import torch
import torch.nn as nn


@torch.no_grad()
def compute_real_loss(model: nn.Module, batches: list[torch.Tensor], device=None) -> float:
    """Average next-token cross-entropy loss over calibration/held-out batches.

    This is the ELSA-style "real loss" signal: the actual language-modeling
    objective evaluated with a full forward pass through the (possibly
    partially pruned) model, as opposed to the per-layer L2 reconstruction
    error that SparseGPT/CWS use as a surrogate. It is used here purely as a
    monitoring/adaptive-control signal layered on top of the one-shot
    layer-wise pruning loop, not as an optimization objective (see README for
    the scope decision vs. full ELSA ADMM).
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        input_ids = batch.to(device)
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        n_tokens = shift_labels.numel()
        total_loss += loss.item()
        total_tokens += n_tokens

    model.train(was_training)
    return total_loss / max(total_tokens, 1)
