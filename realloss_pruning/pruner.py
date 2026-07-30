import copy
import dataclasses

import torch
import torch.nn as nn

from .cws_obs import compute_hinv_cholesky, cws_prune_layer
from .sparsegpt_obs import sparsegpt_prune_layer
from .fisher_hessian import accumulate_fisher_caches, robust_row_hinv
from .hessian import HessianAccumulator
from .real_loss import compute_real_loss


def find_linear_layers(module: nn.Module, prefix: str = "") -> dict[str, nn.Linear]:
    layers = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            layers[full_name] = child
        else:
            layers.update(find_linear_layers(child, full_name))
    return layers


def get_decoder_blocks(model: nn.Module) -> list[nn.Module]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)  # GPT-2 family
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return list(model.model.decoder.layers)  # OPT family
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)  # LLaMA family
    raise ValueError("Unsupported model architecture: could not locate decoder blocks")


class _Catcher(nn.Module):
    """Captures the first block's input args/kwargs, then aborts the forward pass."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.captured_inputs = []
        self.captured_kwargs = []

    def forward(self, hidden_states, *args, **kwargs):
        self.captured_inputs.append(hidden_states)
        self.captured_kwargs.append(kwargs)
        raise _StopForward()


class _StopForward(Exception):
    pass


@dataclasses.dataclass
class PruneLogEntry:
    block_index: int
    cumulative_sparsity: float
    real_loss: float
    damping_used: float


class SequentialPruner:
    """Sequential layer-wise pruner with three interchangeable Hessian
    sources feeding the same OBS-style closed-form selection+correction:

    - 'cws': H = (2/N) X^T X, the local layer-reconstruction Hessian
      (SparseGPT/CWS's usual surrogate), with CWS's full-Hessian
      cancellation-aware selection.
    - 'sparsegpt': same H, real SparseGPT (Frantar & Alistarh, 2023):
      diagonal-only mask selection (shared elimination order across all
      output rows) but the *same* full off-diagonal OBS correction as
      'cws' -- unlike 'cws', it can't detect cancellation groups between
      correlated columns, since diagonal selection ranks purely by
      magnitude, but it still gets the exact closed-form redistribution.
    - 'cws_realloss': H is instead the empirical-Fisher approximation to
      the *real* loss's Hessian w.r.t. each layer's input activations
      (see fisher_hessian.py) -- so the same OBS/CWS closed-form
      redistribution now minimizes real-loss error, not the reconstruction
      surrogate.

    In all cases, real-loss (ELSA-style) evaluation after every block is
    used as a monitoring/adaptive-control signal (damping backoff on loss
    spikes), not as the optimization objective itself.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        blocksize: int | None = None,
        damping: float = 0.01,
        adaptive: bool = True,
        loss_spike_ratio: float = 1.15,
        damping_backoff_multiplier: float = 3.0,
        device: str | None = None,
        method: str = "cws",
    ):
        if method not in ("cws", "sparsegpt", "cws_realloss"):
            raise ValueError(f"method must be 'cws', 'sparsegpt', or 'cws_realloss', got {method!r}")
        self.model = model
        self.sparsity = sparsity
        self.blocksize = blocksize
        self.method = method
        self.base_damping = damping
        self.adaptive = adaptive
        self.loss_spike_ratio = loss_spike_ratio
        self.damping_backoff_multiplier = damping_backoff_multiplier
        self.device = device or next(model.parameters()).device
        self.log: list[PruneLogEntry] = []

    def _capture_block0_inputs(self, calib_batches: list[torch.Tensor], blocks: list[nn.Module]):
        original_block0 = blocks[0]
        catcher = _Catcher(original_block0)
        self._set_block(0, catcher)

        for batch in calib_batches:
            try:
                self.model(batch.to(self.device))
            except _StopForward:
                pass

        self._set_block(0, original_block0)
        blocks[0] = original_block0
        return catcher.captured_inputs, catcher.captured_kwargs

    def _set_block(self, index: int, block: nn.Module):
        blocks_container = self._blocks_container()
        blocks_container[index] = block

    def _blocks_container(self):
        model = self.model
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            return model.model.decoder.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        raise ValueError("Unsupported model architecture")

    def prune(
        self,
        calib_batches: list[torch.Tensor],
        realloss_batches: list[torch.Tensor],
    ) -> list[PruneLogEntry]:
        model = self.model
        model.eval()
        blocks = get_decoder_blocks(model)

        block_inputs, block_kwargs = self._capture_block0_inputs(calib_batches, blocks)

        baseline_loss = compute_real_loss(model, realloss_batches, self.device)
        prev_loss = baseline_loss
        damping = self.base_damping

        for block_idx, block in enumerate(blocks):
            linear_layers = find_linear_layers(block)

            if self.method == "cws_realloss":
                # Needs the full downstream computation graph (this block's
                # output feeds every later block before reaching the loss),
                # so it always runs the whole model from raw token ids,
                # rather than reusing the cached/cropped block activations
                # the forward-hook-based methods below propagate.
                caches = accumulate_fisher_caches(model, linear_layers, calib_batches, self.device)
                for name, layer in linear_layers.items():
                    cache = caches[name]
                    n_rows = layer.weight.shape[0]
                    for row in range(n_rows):
                        Hinv, row_damping = robust_row_hinv(cache, row, damping)
                        pruned_row, _ = cws_prune_layer(
                            layer.weight.data[row : row + 1, :],
                            Hinv,
                            self.sparsity,
                            self.blocksize,
                        )
                        layer.weight.data[row, :] = pruned_row.squeeze(0)
                        if row_damping != damping:
                            print(
                                f"  [cws_realloss] block {block_idx} {name} row {row}: "
                                f"needed damping backoff to {row_damping:.4g} for a stable Cholesky",
                                flush=True,
                            )
                    print(
                        f"  [cws_realloss] block {block_idx} {name}: pruned all {n_rows} rows",
                        flush=True,
                    )

                block_outputs = []
                for i, hidden_states in enumerate(block_inputs):
                    out = block(hidden_states, **block_kwargs[i])
                    block_outputs.append(out[0] if isinstance(out, tuple) else out)
                block_inputs = block_outputs
            else:
                accumulators = {name: HessianAccumulator(layer) for name, layer in linear_layers.items()}
                handles = [
                    layer.register_forward_pre_hook(accumulators[name].make_hook())
                    for name, layer in linear_layers.items()
                ]

                for i, hidden_states in enumerate(block_inputs):
                    block(hidden_states, **block_kwargs[i])

                for h in handles:
                    h.remove()

                for name, layer in linear_layers.items():
                    H = accumulators[name].get_dampened_hessian(damping)
                    Hinv = compute_hinv_cholesky(H)
                    if self.method == "cws":
                        pruned_W, _ = cws_prune_layer(
                            layer.weight.data, Hinv, self.sparsity, self.blocksize
                        )
                    else:
                        pruned_W, _ = sparsegpt_prune_layer(
                            layer.weight.data, Hinv, self.sparsity, self.blocksize
                        )
                    layer.weight.data = pruned_W

                block_outputs = []
                for i, hidden_states in enumerate(block_inputs):
                    out = block(hidden_states, **block_kwargs[i])
                    block_outputs.append(out[0] if isinstance(out, tuple) else out)
                block_inputs = block_outputs

            real_loss = compute_real_loss(model, realloss_batches, self.device)
            cumulative_sparsity = self.sparsity * (block_idx + 1) / len(blocks)

            if self.adaptive and real_loss > prev_loss * self.loss_spike_ratio:
                damping = min(damping * self.damping_backoff_multiplier, 0.5)
            prev_loss = real_loss

            self.log.append(
                PruneLogEntry(
                    block_index=block_idx,
                    cumulative_sparsity=cumulative_sparsity,
                    real_loss=real_loss,
                    damping_used=damping,
                )
            )

        return self.log
