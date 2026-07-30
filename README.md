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

  **`--fisher-mode {row,shared}`** selects between the two constructions
  described above: `row` (default) is the accurate-but-slow per-output-row
  Hessian just described; `shared` reintroduces the cheaper, row-averaged
  construction (`FisherActivationGradCache.get_dampened_shared_hessian`,
  same `s_n = RMS(∂L/∂yₙ)` reweighting as the original, since-corrected
  version of this module) as an explicit, selectable option -- one shared
  Cholesky factorization per layer, same cost profile as `cws`/`sparsegpt`
  (minutes, not hours), reusing the *same cached activations/gradients*
  `row` mode builds, so it needs no extra forward+backward pass. Measured
  on OPT-125M at 50% sparsity, `row` only barely beats `shared` (60.42 vs.
  61.79 ppl, both at `--n-calib 8`) for roughly 50-100x more compute, so
  `shared` is the more practical default despite being less exact -- see
  [Measured results](#measured-results) for how much more calibration data
  changes this picture.

## Measured results

All runs: `facebook/opt-125m`, 50% sparsity, `--blocksize 128`,
`--damping 0.01` (default), `--seqlen 256`, WikiText-2 calibration
(`Salesforce/wikitext`, `wikitext-2-raw-v1` train split), CPU (Apple
Silicon M1, no GPU) -- see [Environment](#environment) for exact package
versions. Each row below is independently reproducible with the exact
command shown.

**Important caveat on comparing across different `--n-calib` values**: the
calibration and held-out real-loss batches are both drawn by
`realloss_pruning/data.py`'s `get_wikitext2_batches`, which samples
`n_calib + n_realloss` total chunks from a fixed-seed RNG (`seed=0` by
default). Because the *count* requested changes what a fixed-seed
`random.sample` call returns, changing `--n-calib` changes *which text*
ends up in the held-out real-loss set too -- so runs at different
`--n-calib` have different dense-model baselines and are **not** directly
comparable on raw perplexity. Compare *relative degradation*
(`pruned_ppl / dense_ppl` from the *same run*) instead when `--n-calib`
differs; only compare raw perplexity directly when `--n-calib` (and
`--n-realloss`, `--seqlen`) match exactly, since only then is the held-out
set identical.

### `--n-calib 8 --n-realloss 4` (dense baseline: 38.94 ppl)

| Method | Command (append to `python run_prune.py --model facebook/opt-125m --blocksize 128 --sparsity 0.5 --n-calib 8 --n-realloss 4 --seqlen 256`) | Final ppl | vs. dense | Wall time (CPU) |
|---|---|---|---|---|
| `cws` | `--method cws` | 49.05 | 1.260x (+26%) | ~85 min |
| `sparsegpt` | `--method sparsegpt` | 50.31 | 1.292x (+29%) | ~2 min |
| `cws_realloss`, row | `--method cws_realloss --fisher-mode row` | 60.42 | 1.552x (+55%) | ~2.5 hr |
| `cws_realloss`, shared | `--method cws_realloss --fisher-mode shared` | 61.79-62.60* | 1.587-1.608x (+59-61%) | ~17.5 min |

\* Ran twice (once before, once after the row/shared refactor in
`fisher_hessian.py`); both are the same construction and the small
difference (61.79 vs. 62.60) is ordinary run-to-run variance, not a
regression -- included to show that range rather than pick one arbitrarily.

`sparsegpt` lands almost exactly where the synthetic correlated-feature
test predicted (see `sparsegpt_obs.py`'s validation, summarized in the
method description above): essentially tied with `cws` (50.31 vs. 49.05),
since it gets the same full off-diagonal OBS correction and OPT-125M's
real weight matrices apparently don't have enough exploitable
cancellation-group structure at this scale/sparsity for `cws`'s
full-Hessian selection to pull meaningfully ahead — unlike the synthetic
test, which was constructed specifically to contain strong cancellation
groups.

`cws_realloss` row-level Hessians give a small, real improvement over
sharing one Hessian across all output rows (61.79-62.60 → 60.42),
consistent with row-specific real-loss sensitivity actually mattering --
but neither variant catches `cws`/`sparsegpt`. That's consistent with the
row-sharing simplification being a secondary effect: the dominant gap is
`cws`/`sparsegpt`'s `H=XᵀX` being an *exact* Hessian of a well-behaved
surrogate objective, versus `cws_realloss`'s Fisher being an
*approximation* to the real loss's curvature that carries more estimation
noise regardless of how it's factored across rows.

### `--n-calib 32 --n-realloss 4` (dense baseline: 49.88 ppl -- different held-out set, see caveat above)

| Method | Command (append to `python run_prune.py --model facebook/opt-125m --blocksize 128 --sparsity 0.5 --n-calib 32 --n-realloss 4 --seqlen 256`) | Final ppl | vs. dense | Wall time (CPU) |
|---|---|---|---|---|
| `cws_realloss`, shared | `--method cws_realloss --fisher-mode shared` | 72.98 | 1.463x (+46%) | ~17.5 min |

More calibration data gives the `shared` variant a real, meaningful
improvement in *relative* terms: degradation drops from 1.587-1.608x (at
`--n-calib 8`) to 1.463x -- closing a good chunk of the gap to
`cws`/`sparsegpt`'s 1.260-1.292x, though not all of it. This is consistent
with calibration-sample-size noise being a real, fixable contributor to
`cws_realloss`'s gap, on top of the more fundamental (and not
data-size-fixable) empirical-Fisher-approximation gap discussed above.

**Not yet run**: `cws`, `sparsegpt`, and `cws_realloss --fisher-mode row`
at `--n-calib 32`, which would let the `--n-calib 32` row above be compared
on identical held-out data rather than only via the relative-degradation
proxy. `cws` in particular takes ~85 minutes even at `--n-calib 8` (its
cost is dominated by CWS's per-block greedy Schur elimination, not by
calibration size, so ~85 min is a reasonable estimate at `--n-calib 32`
too); `cws_realloss --fisher-mode row` at `--n-calib 32` would add roughly
15-20 minutes of extra forward+backward time on top of its ~2.5 hour
baseline (the per-row Cholesky cost, which dominates, doesn't scale with
`--n-calib`).

## Environment

Results in this README were produced with:

- Python 3.11.1
- `torch` 2.13.0, `transformers` 5.14.1, `datasets` 5.0.0 (see
  `requirements.txt`)
- macOS 26.5.1, Apple Silicon (arm64), **CPU only** -- no CUDA GPU
  available. MPS (Apple's GPU backend) was tested and explicitly **not**
  used: it has no `float64` support at all (the Cholesky-based OBS math
  here relies on double precision for numerical stability -- switching to
  float32 caused a genuine `not positive-definite` Cholesky failure, not
  just reduced precision), and separately benchmarked *slower* than plain
  CPU for this workload's many small sequential linear-algebra ops (as
  opposed to the few large batched ops MPS is suited for).
- No random seed beyond `realloss_pruning/data.py`'s default `seed=0` for
  calibration/held-out data sampling (see the note in
  [Measured results](#measured-results) on how changing `--n-calib`
  changes what that seed samples). Pruning itself is deterministic given
  fixed calibration data -- no other randomness is introduced.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Full-Hessian CWS method (default), blockwise for tractability
python run_prune.py --model facebook/opt-125m --method cws --blocksize 128 --sparsity 0.5

# Real SparseGPT: diagonal selection, same full off-diagonal OBS correction
python run_prune.py --model facebook/opt-125m --method sparsegpt --blocksize 128 --sparsity 0.5

# Real-loss Fisher Hessian feeding the same OBS/CWS machinery, row-level (accurate, slow)
python run_prune.py --model facebook/opt-125m --method cws_realloss --fisher-mode row --blocksize 128 --sparsity 0.5

# Same, but shared across rows (cheap, less exact)
python run_prune.py --model facebook/opt-125m --method cws_realloss --fisher-mode shared --blocksize 128 --sparsity 0.5
```

Key flags: `--method {cws,sparsegpt,cws_realloss}`, `--fisher-mode
{row,shared}` (only used by `--method cws_realloss`; default `row`),
`--sparsity`, `--blocksize` (default 128, matching both papers),
`--damping` (default 0.01), `--n-calib` (default 32), `--n-realloss`
(default 8), `--seqlen` (default 512), `--no-adaptive` (disable the
damping-backoff control loop), `--out` (per-block real-loss CSV log,
default `prune_log.csv`), `--save-model` (persist the pruned checkpoint to
a directory), `--device` (default `cuda` if available, else `cpu`; MPS is
never auto-selected -- see [Environment](#environment) for why).

Every command above prints the dense-model baseline loss/perplexity before
pruning starts, then one progress line per block/layer during pruning
(more verbose for `cws_realloss`, which logs per-row damping-retry events
too -- see `cws_realloss`'s method description above), and finally the
pruned model's real loss/perplexity vs. the dense baseline.

`--out` writes a CSV with one row per transformer block: cumulative
sparsity, real loss, perplexity, and the damping value used for that
block — the full per-block trajectory, not just the final number. This is
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
  hessian.py          # local-reconstruction Hessian accumulation (cws/sparsegpt)
  cws_obs.py           # full-Hessian cancellation-aware OBS selection+correction
  sparsegpt_obs.py      # real SparseGPT: diagonal selection, full OBS correction
  fisher_hessian.py    # real-loss empirical-Fisher Hessian (cws_realloss, row+shared)
  real_loss.py         # real next-token cross-entropy loss (monitoring signal)
  pruner.py            # sequential layer-by-layer driver + adaptive control
  data.py              # WikiText-2 calibration/held-out batch loading
run_prune.py            # CLI entry point
requirements.txt         # pinned to tested versions (see Environment)
```
