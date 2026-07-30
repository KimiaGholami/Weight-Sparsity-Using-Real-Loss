# Real-Loss CWS Pruning

A one-shot weight-pruning pipeline for causal LMs that combines three ideas
from three papers in this folder:

- **SparseGPT** (`SparseGPT.pdf`): layer-wise OBS (Optimal Brain Surgeon)
  pruning — greedily remove one weight at a time and apply the closed-form
  compensating update to the remaining weights, using calibration-data
  Hessians `H = (2/N) XᵀX`.
- **CWS** (`CWS.pdf`): SparseGPT's weight *selection* is diagonal-only
  (`[H⁻¹]_jj`, the same for every output row), which is blind to
  "cancellation groups" — correlated input channels whose oppositely-signed
  weights partially cancel in the layer output. CWS derives its selection
  criterion from the *full* Hessian instead, via greedy Schur-complement
  updates within fixed-width column blocks, then applies exact OBS
  corrections.
- **ELSA** (`ELSA.pdf`): argues that layer-wise L2 reconstruction error is a
  *surrogate* for the thing that actually matters — the real
  next-token-prediction loss — and that this surrogate can diverge from it,
  especially at high sparsity.

## What this implementation actually does

This is **not** a reimplementation of ELSA's full ADMM training loop. That
would mean optimizing the real loss directly via many gradient steps —
a fundamentally different (and much more expensive) algorithm from one-shot
layer-wise pruning. Instead, this keeps the fast, single-calibration-pass
structure of SparseGPT/CWS, and uses ELSA's core insight — that the real
loss is what matters, not the reconstruction surrogate — in two ways:

1. **Monitoring/adaptive control** (all methods): transformer decoder blocks
   are processed strictly sequentially (block 0, then block 1, ...), exactly
   as SparseGPT/GPTQ do. After each block is pruned, a full forward pass of
   the (partially-pruned) model over held-out data records the actual
   cross-entropy loss (`realloss_pruning/real_loss.py`) — the "real loss" in
   ELSA's sense. If it jumps sharply after a block (`loss_spike_ratio`,
   default 1.15x the previous block's loss), Hessian damping for subsequent
   blocks increases (`damping_backoff_multiplier`) — a lightweight safeguard
   against layer-wise pipelines' compounding-error problem, without adopting
   ELSA's full ADMM machinery.
2. **The correction itself** (`cws_realloss` method): rather than only
   watching real loss after the fact, OBS's closed-form redistribution is
   fed a Hessian *derived from* real loss — see below.

## Three interchangeable methods

All three share the same OBS-style closed-form selection+correction
machinery; they differ only in which Hessian `H` feeds it. Select with
`--method`:

- **`cws`** (`realloss_pruning/cws_obs.py`, default) — `H = (2/N) XᵀX`, the
  local layer-reconstruction Hessian (SparseGPT/CWS's usual surrogate).
  Selection and correction both use the full (off-diagonal-aware)
  block-local inverse-Hessian, so it detects cancellation groups the way
  CWS's paper describes. Because each output row can prune a different set
  of columns, it can't reuse SparseGPT's single shared global elimination
  order; it keeps a separate evolving inverse-Hessian block per row,
  Schur-downdated after every weight removed (see the module docstring for
  the full derivation, including the block-to-future propagation formula,
  which needed an extra `(H_block)⁻¹` factor beyond the naive
  SparseGPT-style cross-term reuse — verified numerically in-repo, not just
  derived).

  This needs `blocksize` chunking (default 128, matching CWS's own paper)
  for both numerical stability and tractability: with no chunking
  (`--blocksize` unset), the per-row inverse-Hessian state is `d_in × d_in`
  in size, e.g. ~54GB for OPT-125M's `fc2` layer alone.

- **`sparsegpt`** (`realloss_pruning/sparsegpt_obs.py`) — real SparseGPT
  (Frantar & Alistarh, 2023): same `H = XᵀX` as `cws`, and the *same* full
  off-diagonal OBS correction, but mask selection is diagonal-only and
  uses one shared elimination order across every output row (not a
  separate per-row order like `cws`), so it can't detect cancellation
  groups between correlated columns — selection reduces to plain magnitude
  ranking within a column, since `[H⁻¹]_jj` doesn't depend on row. Because
  the elimination order is shared, it needs no per-row Schur bookkeeping at
  all: `compute_hinv_cholesky`'s matrix can be read off directly, one shared
  factorization for the whole layer. On synthetic data with genuine
  cancellation groups (correlated, opposite-signed weight pairs), this
  lands cleanly between `cws` and uncorrected magnitude pruning in
  reconstruction error — it gets most of the benefit from OBS's exact
  correction, but misses what `cws`'s full-Hessian selection catches. On
  i.i.d. features (nothing to cancel), it's nearly identical to `cws`.

- **`cws_realloss`** (`realloss_pruning/fisher_hessian.py`) — same OBS/CWS
  closed-form selection+correction machinery as `cws`, but `H` is instead a
  Gauss-Newton empirical-Fisher approximation to the *real loss's* Hessian,
  built **separately per output row**: `H_i ≈ (1/N) Σₙ (∂L/∂y_{n,i})² xₙxₙᵀ`,
  pairing each layer's forward input activation `xₙ` (same quantity
  `cws`/`sparsegpt` already collect) with that specific output row's own
  real-loss sensitivity, captured via a backward hook after a full
  forward+backward pass of the whole (current, partially-pruned) model.
  Each `H_i` has the same `(d_in, d_in)` PSD shape as `cws`'s `H = XᵀX`, so
  it plugs into the identical Cholesky/CWS machinery — the "error being
  redistributed" by OBS's formula is now about the real loss, not the local
  reconstruction surrogate.

  An earlier version of this module instead used
  `gₙ = ∂(real loss)/∂(layer input)` directly as the outer-product vector.
  That's a different, and wrong, quantity: for a linear layer `y=Wx`,
  `∂L/∂x = Wᵀ·∂L/∂y`, so that construction reduced to
  `Wᵀ[(1/N)Σₙ(∂L/∂yₙ)(∂L/∂yₙ)ᵀ]W` — built circularly from the very (dense)
  weight matrix about to be pruned, not a curvature-w.r.t.-weights quantity
  at all. Its symptom was diagnostic: giving it *more* calibration data made
  results *worse* (454 → 635 perplexity at 50% sparsity on OPT-125M),
  which is the signature of a more confident estimate of the wrong thing,
  not noise in an estimate of the right thing.

  The corrected version initially *shared* one Hessian across all of a
  layer's output rows (averaging `(∂L/∂yᵢ)²` into one scalar per sample) --
  the same free simplification `cws`/`sparsegpt` make, except it isn't free here:
  the reconstruction Hessian provably doesn't depend on row, but the real
  loss's sensitivity genuinely does, so sharing it throws away exactly the
  row-specific information that's the whole reason to prefer a real-loss
  Hessian in the first place. The current version keeps a separate `H_i`
  per row instead (`fisher_hessian.py`'s `FisherActivationGradCache`),
  which reintroduces SparseGPT's original "row-Hessian challenge" for real
  loss specifically (a separate Cholesky factorization per output row,
  where the reconstruction Hessian needs only one for the whole layer) --
  this is the main cost driver for this method (see below).

  Row-specific Hessians can also be far more ill-conditioned than the
  shared one (a row whose real-loss gradient is near-zero for most
  calibration tokens has an almost-singular weighted covariance), which
  surfaced as a genuine `Cholesky: not positive-definite` crash ~105 minutes
  into a full run even in float64 -- fixed with `robust_row_hinv`, which
  retries with geometrically increasing damping until the factorization
  succeeds (mathematically guaranteed to terminate: adding any positive
  damping to a PSD matrix makes it strictly PD, so failure is purely a
  float64-rounding-noise-floor issue, not a real singularity).

  Because it needs the whole downstream network to know how loss-sensitive
  a layer's output is, this always runs full forward+backward passes from
  raw token ids per block (not the cached/cropped activation propagation
  the other two methods use). Combined with a separate per-row Cholesky
  factorization, it's substantially slower than `cws`/`sparsegpt`: ~2.5 hours
  for OPT-125M vs. minutes, dominated by `fc2`'s 768 rows each needing their
  own `d_in=3072` factorization. MPS (Apple GPU) does not help here --
  benchmarked slower than plain CPU for this many-small-sequential-ops
  workload, and it has no float64 support at all, which caused an
  independent numerically-real Cholesky failure when tested with float32.

## Measured results

OPT-125M, 50% sparsity, WikiText-2 calibration (8 batches, seqlen 256),
same held-out real-loss set across all runs (dense baseline: 38.94
perplexity):

| Method | Final perplexity | vs. dense |
|---|---|---|
| `cws` | 49.05 | +26% |
| `sparsegpt` | 50.31 | +29% |
| `cws_realloss` (row-level Fisher) | 60.42 | +55% |
| `cws_realloss` (shared/averaged Fisher, superseded) | 61.79 | +59% |

`sparsegpt` lands almost exactly where the synthetic correlated-feature
test predicted: essentially tied with `cws` (50.31 vs. 49.05), since it
gets the same full off-diagonal OBS correction and OPT-125M's real weight
matrices apparently don't have enough exploitable cancellation-group
structure at this scale/sparsity for `cws`'s full-Hessian selection to pull
meaningfully ahead — unlike the synthetic test, which was constructed
specifically to contain strong cancellation groups.

Row-level Hessians give a small, real improvement to `cws_realloss` over
sharing one Hessian across all output rows (61.79 → 60.42), consistent with
row-specific real-loss sensitivity actually mattering -- but `cws_realloss`
still doesn't catch `cws`/`sparsegpt`. That's consistent with the
row-sharing simplification being a secondary effect: the dominant gap is
`cws`/`sparsegpt`'s `H=XᵀX` being an *exact* Hessian of a well-behaved
surrogate objective, versus `cws_realloss`'s Fisher being an
*approximation* to the real loss's curvature that carries more estimation
noise regardless of how it's factored across rows. Whether more
calibration data closes this gap further is untested — the diagnostic
sample-size experiment above was run against the
since-fixed, conceptually wrong construction, so it doesn't say anything
about the corrected version's data efficiency.

### CWS paper's own published results (reference, not reproduced here)

These are the CWS paper's own numbers (`CWS.pdf`, Tables 3/6/10), for
context on how `cws`'s selection criterion is reported to behave across
sparsity levels and model scales — WikiText-2 perplexity, one-shot pruning,
their own baselines (Wanda, RIA, SparseGPT, AWP). Not run by us; included
purely as the paper's reference point, on different (larger) models than
the OPT-125M runs above.

**TinyLlama-1.1B** (dense PPL 7.8):

| Sparsity | CWS | Wanda | RIA | SparseGPT | AWP |
|---|---|---|---|---|---|
| 30% | **8.13** | 8.15 | 8.14 | 8.37 | 8.16 |
| 40% | **8.74** | 8.82 | 8.81 | 9.41 | 8.82 |
| 50% | **10.20** | 10.73 | 10.77 | 11.83 | 10.73 |
| 60% | **14.60** | 19.76 | 19.69 | 17.83 | 19.86 |
| 70% | **34.40** | 94.92 | 96.21 | 62.38 | 95.80 |
| 80% | **217.27** | 615 | 902 | 1,490 | 637 |

**HGRN-1.3B** (dense PPL 11.8; gated recurrent SSM, no attention):

| Sparsity | CWS | Wanda | RIA | SparseGPT | AWP |
|---|---|---|---|---|---|
| 30% | **12.14** | 31.64 | 25.6 | 15.02 | 21.7 |
| 40% | **12.69** | 76.87 | 54.3 | 16.46 | 49.7 |
| 50% | **13.88** | 350 | 348 | 17.4 | 426.2 |
| 60% | **17.07** | 11,552 | 8,239 | 32.43 | 2,616 |
| 70% | **31.11** | 20,592 | 17,457 | 115.4 | 4,440 |
| 80% | **131.27** | 76,051 | 28,615 | 6,956 | 17,195 |

**LLaMA-7B** (dense PPL 6.61):

| Sparsity | CWS | Wanda | RIA | SparseGPT | AWP |
|---|---|---|---|---|---|
| 30% | 6.87 | 6.89 | 6.84 | 6.99 | **6.81** |
| 40% | 7.30 | 7.40 | 7.30 | 7.76 | **7.19** |
| 50% | 8.24 | 8.69 | 8.58 | 9.51 | **8.10** |
| 60% | **10.96** | 14.40 | 14.22 | 15.64 | 11.33 |
| 70% | **26.85** | 77.82 | 102.17 | 67.02 | 31.99 |
| 80% | **245.5** | 1,647.4 | 2,084.1 | 2,071.2 | 211.1 |

The paper's own headline finding here: CWS's advantage is largest on
sub-2B models (TinyLlama, HGRN), where it wins at every sparsity level
tested. At 7B scale the ordering partially inverts — AWP (an iterative
gradient-based mask search) overtakes CWS on raw perplexity at low-to-mid
sparsity, though the paper reports CWS still keeps the best zero-shot task
accuracy at 80% sparsity despite AWP's lower perplexity there, and AWP costs
substantially more compute (up to 200 gradient steps per layer vs. CWS's
single closed-form pass).

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full-Hessian CWS method (default), blockwise for tractability
python run_prune.py --model facebook/opt-125m --method cws --blocksize 128 --sparsity 0.5

# Real SparseGPT: diagonal selection, same full off-diagonal OBS correction
python run_prune.py --model facebook/opt-125m --method sparsegpt --blocksize 128 --sparsity 0.5

# Real-loss Fisher Hessian feeding the same OBS/CWS machinery
python run_prune.py --model facebook/opt-125m --method cws_realloss --blocksize 128 --sparsity 0.5
```

Key flags: `--method {cws,sparsegpt,cws_realloss}`, `--sparsity`,
`--blocksize` (default 128, matching both papers), `--damping`,
`--n-calib`, `--n-realloss`, `--seqlen`, `--no-adaptive` (disable the
damping-backoff control loop), `--out` (per-block real-loss CSV log),
`--save-model`.

Output is a CSV with one row per transformer block: cumulative sparsity,
real loss, perplexity, and the damping value used for that block — this is
the ELSA-style "real loss vs. sparsity" trajectory, at the model level
rather than the layer-reconstruction level.

## Architecture support

Targets standard `nn.Linear`-based decoder-only causal LMs: OPT and LLaMA
families work out of the box (`realloss_pruning/pruner.py`'s
`get_decoder_blocks`/`find_linear_layers`). GPT-2 is **not** supported as-is
— HuggingFace's GPT-2 implementation uses a custom `Conv1D` module with a
transposed weight layout instead of `nn.Linear`, which this code doesn't
special-case.

## Layout

```
realloss_pruning/
  hessian.py         # local-reconstruction Hessian accumulation (cws/sparsegpt)
  cws_obs.py          # full-Hessian cancellation-aware OBS selection+correction
  sparsegpt_obs.py     # real SparseGPT: diagonal selection, full OBS correction
  fisher_hessian.py   # real-loss empirical-Fisher Hessian (cws_realloss)
  real_loss.py        # real next-token cross-entropy loss (monitoring signal)
  pruner.py           # sequential layer-by-layer driver + adaptive control
  data.py             # WikiText-2 calibration/held-out batch loading
run_prune.py           # CLI entry point
```
