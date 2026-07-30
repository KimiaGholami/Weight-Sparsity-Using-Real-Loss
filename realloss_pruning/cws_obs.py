import math

import torch


def compute_hinv_cholesky(H: torch.Tensor) -> torch.Tensor:
    """SparseGPT's Cholesky-based inverse-Hessian sequence trick.

    Returns an upper-triangular matrix `Hinv` such that `Hinv[j, j:]` equals
    the j-th row of (H_{U_j})^{-1}, where U_j = {j, j+1, ..., d-1} is the set
    of columns not yet eliminated under a fixed left-to-right elimination
    order. This lets a single O(d^3) factorization stand in for the whole
    sequence of partial-inverse Gaussian-elimination steps that OBS needs.
    """
    L = torch.linalg.cholesky(H)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv_full, upper=True)
    return Hinv


@torch.no_grad()
def cws_prune_layer(
    W: torch.Tensor,
    Hinv: torch.Tensor,
    sparsity: float,
    blocksize: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune a weight matrix using CWS's cancellation-aware OBS pruning.

    Unlike SparseGPT, which selects the whole block's mask up front from a
    single static snapshot of diag(H^-1) (the same for every output row,
    since that diagonal doesn't depend on the row), this selects one weight
    at a time per output row, from the argmin of r_j^2 / d_j over the full
    evolving local inverse-Hessian block -- so correlated ("cancellation")
    channels change each other's scores as they get pruned, which a purely
    diagonal criterion can never see.

    Because each row can pick a different elimination order within a block,
    the SparseGPT trick of reusing one shared global Hinv sequence for every
    row no longer applies inside a block (this is CWS's "row-Hessian
    challenge" fix): each row keeps its own local (blocksize x blocksize)
    inverse-Hessian copy, Schur-downdated after every weight it removes.
    Across blocks, the standard SparseGPT lazy update is still exact for
    propagating a block's *net* correction to not-yet-processed columns,
    since OBS's iterative-application result is order-invariant for a fixed
    final removed set (Frantar & Alistarh, 2023, Sec 3.1).

    Args:
        W: (d_out, d_in) weight matrix, modified via a cloned copy.
        Hinv: (d_in, d_in) upper-triangular matrix from `compute_hinv_cholesky`.
        sparsity: fraction of weights to prune per row, per block.
        blocksize: number of input columns processed jointly per greedy pass
            before propagating corrections to the remaining columns. `None`
            (the default) disables this chunking and uses the full `d_in`
            as a single block -- the whole layer's Hessian is used at once,
            with no lazy cross-block propagation needed at all.

    Returns:
        (pruned_W, mask) where mask is True at kept (unpruned) positions.
    """
    if blocksize is None:
        blocksize = W.shape[1]
    device = W.device
    dtype = W.dtype
    # MPS has no float64 support, so the working precision follows whatever
    # dtype the caller already prepared Hinv in (float64 on CPU/CUDA, float32
    # on MPS) rather than hardcoding float64 everywhere.
    compute_dtype = Hinv.dtype
    d_out, d_in = W.shape
    W = W.clone().to(compute_dtype)
    mask = torch.ones((d_out, d_in), dtype=torch.bool, device=device)

    n_blocks = math.ceil(d_in / blocksize)
    for b in range(n_blocks):
        start = b * blocksize
        end = min(start + blocksize, d_in)
        B = end - start
        k_prune = int(math.floor(B * sparsity))
        if k_prune == 0:
            continue

        w_before = W[:, start:end].clone()
        w_local = w_before.clone()
        active = torch.ones((d_out, B), dtype=torch.bool, device=device)

        # `Hinv[start:end, start:end]` is a slice of the *Cholesky factor* of
        # (H_{U_start})^{-1}, not the matrix itself -- it must be multiplied
        # out (U^T U) to recover the true block-local inverse-Hessian that
        # the per-row greedy OBS recursion needs as its starting point.
        U_block = Hinv[start:end, start:end].to(compute_dtype)
        Hinv_block0 = U_block.t() @ U_block
        Hinv_local = Hinv_block0.unsqueeze(0).repeat(d_out, 1, 1)

        for _ in range(k_prune):
            diag = torch.diagonal(Hinv_local, dim1=1, dim2=2)
            score = w_local.pow(2) / diag.clamp_min(1e-10)
            score = score.masked_fill(~active, float("inf"))
            j = torch.argmin(score, dim=1)

            d_j = diag.gather(1, j.unsqueeze(1)).squeeze(1)
            col = torch.gather(
                Hinv_local, 2, j.view(-1, 1, 1).expand(-1, B, 1)
            ).squeeze(2)
            w_j = w_local.gather(1, j.unsqueeze(1)).squeeze(1)

            # `col` is only meaningful at still-active positions: entries of
            # Hinv_local at already-pruned rows/cols are stale leftovers from
            # earlier downdates, so the delta must not touch those weights
            # (they are frozen at 0), or it would silently un-prune them.
            delta = -(w_j / d_j).unsqueeze(1) * col
            w_local = torch.where(active, w_local + delta, w_local)
            active.scatter_(1, j.unsqueeze(1), False)
            w_local.scatter_(1, j.unsqueeze(1), torch.zeros(d_out, 1, device=device, dtype=compute_dtype))

            outer = col.unsqueeze(2) * col.unsqueeze(1)
            Hinv_local = Hinv_local - outer / d_j.view(-1, 1, 1)

        W[:, start:end] = w_local
        mask[:, start:end] = active

        if end < d_in:
            # Same U^T U reconstruction for the block-to-future cross term:
            # (H_{U_start})^{-1}[block, future] = U_block^T @ U_future.
            # This cross term is evaluated once per block (pre-block state);
            # not re-deriving it after every intra-block Schur downdate is
            # the same block/lazy-update approximation SparseGPT itself
            # makes at block boundaries.
            #
            # The conditional-shift formula for a quadratic form, partitioned
            # into (block, future), is Δw_future = -M_ff^{-1} M_fb Δw_block
            # where M = H_{U_start}. Writing (H_{U_start})^{-1} in the same
            # partition as [[A, B], [B^T, D]], the identity
            # M_ff^{-1} M_fb = -B^T A^{-1} gives Δw_future = B^T A^{-1}
            # Δw_block @ ... i.e. it needs an extra Hinv_block0^{-1} factor,
            # not just the raw cross term (verified numerically against a
            # direct block-matrix-inverse computation).
            U_future = Hinv[start:end, end:].to(compute_dtype)
            cross = U_block.t() @ U_future
            E = w_local - w_before
            W[:, end:] += E @ torch.inverse(Hinv_block0) @ cross

    W = W.to(dtype)
    W = W * mask
    return W, mask
