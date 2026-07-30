import math

import torch


@torch.no_grad()
def sparsegpt_prune_layer(
    W: torch.Tensor,
    Hinv: torch.Tensor,
    sparsity: float,
    blocksize: int | None = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real SparseGPT (Frantar & Alistarh, 2023): diagonal-only mask
    selection, but a *full* off-diagonal OBS correction.

    Unlike CWS (`cws_obs.py`), which lets each output row pick its own
    elimination order (needed to detect cancellation groups, at the cost of
    a separate evolving inverse-Hessian per row), SparseGPT fixes one
    left-to-right elimination order shared by every row. That sharing is
    exactly what makes `Hinv` (from `compute_hinv_cholesky`) usable
    directly: row j of `Hinv`, from column j onward, already encodes
    everything needed at elimination step j for every row simultaneously --
    no per-row Schur bookkeeping required, unlike CWS.

    Mask selection is a static snapshot per block: for each block of
    `blocksize` columns, the score `w_{i,j}^2 / [H^-1]_{jj}` is computed
    once (this diagonal doesn't depend on row i, so ranking within a column
    reduces to plain magnitude -- this is the "diagonal-only" limitation
    that misses cancellation groups between correlated columns), and the
    lowest-scoring fraction per row is pruned. Given `Hinv[j,j] =
    sqrt([H^-1]_jj)` (see `cws_obs.py`'s docstring for the derivation), the
    true diagonal needed for scoring is `Hinv[j,j]**2`, not `Hinv[j,j]`
    directly -- squaring only cancels out in the *combined* correction
    formula below, not in isolation for ranking.

    The correction, once a column's weight is marked for pruning, is
    applied to *every* remaining column (both still-in-block and future
    columns) in one step, using `Hinv[j,j:]` directly -- no block/lazy-update
    split is needed here (unlike CWS) because there's only one shared
    Hinv sequence for the whole layer, so there's nothing to batch.

    Args:
        W: (d_out, d_in) weight matrix, modified via a cloned copy.
        Hinv: (d_in, d_in) upper-triangular matrix from `compute_hinv_cholesky`.
        sparsity: fraction of weights to prune per row, per block.
        blocksize: number of columns whose mask is selected from one static
            snapshot before moving to the next block. `None` uses the full
            `d_in` as a single block.

    Returns:
        (pruned_W, mask) where mask is True at kept (unpruned) positions.
    """
    if blocksize is None:
        blocksize = W.shape[1]
    device = W.device
    dtype = W.dtype
    compute_dtype = Hinv.dtype
    d_out, d_in = W.shape
    W = W.clone().to(compute_dtype)
    mask = torch.ones((d_out, d_in), dtype=torch.bool, device=device)

    hinv_diag = torch.diagonal(Hinv).to(compute_dtype)
    d_all = hinv_diag.pow(2)

    n_blocks = math.ceil(d_in / blocksize)
    for b in range(n_blocks):
        start = b * blocksize
        end = min(start + blocksize, d_in)
        B = end - start
        k_prune = int(math.floor(B * sparsity))
        if k_prune == 0:
            continue

        block_w = W[:, start:end]
        block_d = d_all[start:end]
        score = block_w.pow(2) / block_d.clamp_min(1e-10).unsqueeze(0)
        prune_idx = torch.topk(score, k_prune, largest=False, dim=1).indices
        mask[:, start:end].scatter_(1, prune_idx, False)

    for j in range(d_in):
        d_j = Hinv[j, j].to(compute_dtype)
        col = Hinv[j, j:].to(compute_dtype)
        is_pruned = ~mask[:, j]
        w_j = W[:, j].clone()

        err = torch.where(is_pruned, w_j / d_j, torch.zeros_like(w_j))
        W[:, j:] -= err.unsqueeze(1) * col.unsqueeze(0)
        W[:, j] = torch.where(is_pruned, torch.zeros_like(w_j), W[:, j])

    W = W.to(dtype)
    W = W * mask
    return W, mask
