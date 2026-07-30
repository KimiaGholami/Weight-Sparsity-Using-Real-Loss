import torch
import torch.nn as nn
import torch.nn.functional as F

from .cws_obs import compute_hinv_cholesky


def _working_dtype(device) -> torch.dtype:
    # MPS has no float64 support; fall back to float32 there and keep the
    # full double precision everywhere else (CPU/CUDA).
    device = torch.device(device)
    return torch.float32 if device.type == "mps" else torch.float64


def _damp(H: torch.Tensor, damping: float) -> torch.Tensor:
    diag_idx = torch.arange(H.shape[0], device=H.device)
    dead = H[diag_idx, diag_idx] == 0
    H[dead, dead] = 1.0
    damp = damping * H[diag_idx, diag_idx].mean()
    H[diag_idx, diag_idx] += damp
    return H


class FisherActivationGradCache:
    """Caches forward input activations and backward output gradients for a
    linear layer across all calibration batches, then builds a genuinely
    per-output-row empirical-Fisher Hessian on demand.

    For y = Wx, the Gauss-Newton curvature of the real loss w.r.t. output
    row i is (d^2L/dy_i^2) x x^T -- row-specific, since different output
    neurons feed the rest of the network differently and the real loss is
    not equally sensitive to all of them. Averaging (d L/dy_i)^2 across rows
    into one shared scalar (as an earlier version of this module did) throws
    away exactly that row-specific information, which is the whole reason to
    prefer a real-loss-derived Hessian over the reconstruction Hessian in
    the first place -- so this keeps the full (n_tokens, d_out) gradient
    tensor instead of collapsing it, and builds H_i = (1/N) sum_n
    (dL/dy_{n,i})^2 x_n x_n^T separately for each row i.

    Caching (x, grad_output) is cheap: O(n_tokens * (d_in + d_out)) rather
    than O(d_out * d_in^2), so the one unavoidable full-model forward+backward
    pass per calibration batch is still done only once per layer, not once
    per output row. The cost that *does* scale with d_out is downstream: a
    separate (d_in, d_in) Cholesky factorization per row instead of one
    shared factorization for the whole layer -- the same "row-Hessian
    challenge" SparseGPT's shared-H trick exists to avoid, reintroduced here
    because a real-loss Hessian genuinely differs per row where the
    reconstruction Hessian provably does not.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.d_in = layer.weight.shape[1]
        self.d_out = layer.weight.shape[0]
        self.device = layer.weight.device
        self.compute_dtype = _working_dtype(self.device)
        self._x_chunks: list[torch.Tensor] = []
        self._g_chunks: list[torch.Tensor] = []
        self._last_input = None
        self.X: torch.Tensor | None = None
        self.G: torch.Tensor | None = None

    def capture_input(self, x: torch.Tensor) -> None:
        self._last_input = x.detach().reshape(-1, x.shape[-1]).to(self.compute_dtype)

    def capture_output_grad(self, grad_output: torch.Tensor) -> None:
        if self._last_input is None:
            return
        go = grad_output.detach().reshape(-1, grad_output.shape[-1]).to(self.compute_dtype)
        self._x_chunks.append(self._last_input)
        self._g_chunks.append(go)
        self._last_input = None

    def finalize(self) -> None:
        self.X = torch.cat(self._x_chunks, dim=0).to(self.device)
        self.G = torch.cat(self._g_chunks, dim=0).to(self.device)
        self._x_chunks = []
        self._g_chunks = []

    def get_dampened_row_hessian(self, row: int, damping: float = 0.01) -> torch.Tensor:
        weighted_x = self.X * self.G[:, row : row + 1]
        n = weighted_x.shape[0]
        H = (weighted_x.t() @ weighted_x) / max(n, 1)
        return _damp(H, damping)

    def get_dampened_shared_hessian(self, damping: float = 0.01) -> torch.Tensor:
        """One Hessian shared across all output rows, built from the same
        cached (X, G) this cache already holds -- no extra forward/backward
        pass needed. Reintroduces the earlier, cheaper `cws_realloss`
        construction (row-averaged real-loss sensitivity) as an explicit,
        selectable alternative to `get_dampened_row_hessian`'s per-row
        version: one shared Cholesky factorization per layer, same cost
        profile as `cws`/`sparsegpt`, instead of a separate factorization
        per output row. `s_n = RMS(dL/dy_n)` stands in for the row-specific
        `d^2L/dy_i^2`, averaged across output rows -- the same free
        simplification `cws`/`sparsegpt` make for the reconstruction
        Hessian, except here it's a real approximation (see this class's
        docstring), not something the objective already gives you for free.
        """
        s = self.G.pow(2).mean(dim=-1).sqrt()
        weighted_x = self.X * s.unsqueeze(1)
        n = weighted_x.shape[0]
        H = (weighted_x.t() @ weighted_x) / max(n, 1)
        return _damp(H, damping)

    def make_forward_hook(self):
        def hook(module, inputs):
            self.capture_input(inputs[0])

        return hook

    def make_backward_hook(self):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.capture_output_grad(grad_output[0])

        return hook


def _robust_hinv(build_H, damping: float, max_retries: int, backoff: float, label: str):
    current_damping = damping
    for _ in range(max_retries):
        H = build_H(current_damping)
        try:
            return compute_hinv_cholesky(H), current_damping
        except torch.linalg.LinAlgError:
            current_damping *= backoff
    raise RuntimeError(
        f"Cholesky failed for {label} even after {max_retries} damping retries "
        f"(final damping tried: {current_damping})"
    )


def robust_row_hinv(
    cache: FisherActivationGradCache,
    row: int,
    damping: float,
    max_retries: int = 8,
    backoff: float = 10.0,
):
    """Builds a row's Hessian inverse, retrying with progressively larger
    damping if Cholesky fails.

    Per-row Hessians can be nearly singular for output neurons whose
    real-loss gradient is close to zero across most calibration tokens
    (e.g. rarely-active units): the shared, all-rows-averaged Hessian
    'cws'/'sparsegpt' use never has this problem (averaging across rows
    smooths it out), but a single row's own weighted covariance can be, and the
    fixed 1%-of-this-row's-own-scale damping isn't always enough to fix it
    -- a degenerate row has a tiny own-scale to begin with, so 1% of it is
    also tiny. Retrying with geometrically larger damping on failure is a
    standard, cheap way to recover instead of crashing a run that may be
    hours into a per-row sweep over a large layer.

    Returns (Hinv, damping_used).
    """
    return _robust_hinv(
        lambda d: cache.get_dampened_row_hessian(row, d), damping, max_retries, backoff, f"row {row}"
    )


def robust_shared_hinv(
    cache: FisherActivationGradCache,
    damping: float,
    max_retries: int = 8,
    backoff: float = 10.0,
):
    """Same damping-retry safety net as `robust_row_hinv`, for the single
    shared (all-rows-averaged) Hessian instead of a per-row one. Less
    likely to need it in practice -- averaging across rows smooths out the
    near-singular cases a single degenerate row can produce -- but cheap
    to guard against regardless.

    Returns (Hinv, damping_used).
    """
    return _robust_hinv(
        lambda d: cache.get_dampened_shared_hessian(d), damping, max_retries, backoff, "shared Hessian"
    )


def accumulate_fisher_caches(
    model: nn.Module,
    linear_layers: dict[str, nn.Linear],
    batches: list[torch.Tensor],
    device,
) -> dict[str, FisherActivationGradCache]:
    """Runs full forward+backward passes of the real loss through the whole
    (current, partially-pruned) model once per calibration batch, caching
    each of `linear_layers`' (input activation, output gradient) pairs via
    paired forward/backward hooks. This needs the complete downstream
    computation graph to know how sensitive the real loss is to each
    layer's output, so it always runs the full model from raw token ids
    rather than reusing cached/cropped block activations.
    """
    caches = {name: FisherActivationGradCache(layer) for name, layer in linear_layers.items()}
    handles = []
    for name, layer in linear_layers.items():
        handles.append(layer.register_forward_pre_hook(caches[name].make_forward_hook()))
        handles.append(layer.register_full_backward_hook(caches[name].make_backward_hook()))

    for batch in batches:
        input_ids = batch.to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss.backward()

    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)

    for cache in caches.values():
        cache.finalize()

    return caches
