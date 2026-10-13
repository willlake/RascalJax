# RascalJax development notes

Importance-sampled covariance matrices of 2-point correlation functions (a JAX rewrite of RascalC). Read
`README.md` for the user-facing picture and `BENCHMARK.md` for the validation status before changing the numerics.

## Layout and data flow

```
catalog.py   Catalog (randoms, weights, n_data, jackknife labels) / Sample (membership-weighted view of a catalog)
grid.py      cuboid or periodic cell grid, CSR particle layout, n_side auto-tuning to 2-16 particles per filled cell
kernels.py   cell-offset probability cubes (1/r^2 Dawson kernel; xi kernel by quadrature) + alias-table sampler
xi_table.py  xi(s, mu) tables: cubic-spline interpolation of s^2 xi, RascalC extrapolation rules, fine lookup grids
binning.py   Binning: s edges, linear mu bins in [0, 1], Legendre projection matrices, output ordering (s, ell)
integrals.py JAX kernel accumulate_batch (pairs -> triples -> quads), Static (hashable config), PassData (device arrays)
driver.py    run_cov: builds everything, plan_passes decides catalog tuples per pass, distributes subsamples over
             devices, normalizes and assembles blocks into RawCovariance (c4, c3 per shared catalog, c2 per catalog pair)
postprocess.py jackknife/mock shot-noise rescaling, D-matrix precision, N_eff, combine_regions, comparison metrics
io_lsstypes.py / io_pycorr.py  adapters producing CorrelationInput; compat.py  RascalC C++ inputs/outputs
```

Conventions that must stay consistent across modules:

* Flat (s, mu) bin index is `i_s * n_mu + i_mu`; output index is `i_s * n_ell + i_ell` (RascalC order). The DESI /
  pycorr order `(ell, s)` is obtained with `Binning.to_pycorr_order`.
* Weights are normalized to unit sum per catalog and pair counts are *normalized* counts; `n_data / n_random` of a
  catalog ("galaxies per random") enters C3 once and C2 twice (`driver._normalize`).
* All accumulated sums are per draw; `_normalize` divides by draws per primary (`n_draws / n_prim`), so the sums
  estimate sums over all primaries.
* C4 accumulates only the kernel-matched Wick contraction xi(ik) xi(jl); the other contraction comes from the
  term with the first pair's samples swapped (`plan_passes` adds both). Do not add xi(il) xi(jk) per quad: it is
  unbiased but heavy-tailed and ruined the convergence of the DESI runs (N_eff 550 -> 6000 after the change).
  C3 is assembled from the shared-particle tensor `T[X,(Y,Y'),Z]` in four sharing cases. Diagonal blocks are
  symmetrized at the end.
* Jackknife weight tensor and its `1 - sum_A w_aA w_bA` normalization follow RascalC; the EE/RR disconnected
  arrays use the full-cell draw probability and `0.25 n_pairs` normalization (not RascalC's class-conditional one).

## Working rules

* Keep the kernel pure and shape-static: everything that affects shapes or control flow goes into `Static`
  (hashable dataclass, `static_argnames`), arrays into `PassData`. Never put Python scalars in `PassData`.
* GPU scatter-adds into few segments are slow (atomics contention). Accumulate into the full `(n_bins, n_bins)`
  (s, mu) matrix and project afterwards; do not revert to segment sums over `(s, s) x ell^2`.
* JAX caches traces per function object: when monkeypatching module functions for ablations, jit a fresh
  wrapper function (see `scripts/profile_ablation.py`), otherwise the patch silently does nothing.
* `float64` everywhere (`jax_enable_x64` is set on import of `integrals.py`). `run_cov(positions_float32=True)`
  stores positions and computes separations in float32 (verified unbiased against the exact sums), but it gave no
  measurable speedup on A100s (1.83e8 vs 1.79e8 draws/s): the kernel is not limited by the position gathers.
* Use `XLA_PYTHON_CLIENT_PREALLOCATE=false` only on shared GPUs; on an exclusive node leave the default.
* Never run the C++ comparisons in periodic mode as a reference: the RascalC periodic binaries are biased high
  by ~3% (C3) and ~6% (C4) against exact sums; the aperiodic binaries are unbiased.

## Testing

```
JAX_PLATFORMS=cpu python -m pytest -q tests          # ~1 min, unit + small end-to-end runs
python scripts/bruteforce_check.py 4000 512 --rascalc --aperiodic   # exact sums vs RascalJax (and C++), ~5 min
python scripts/gaussian_box.py 200000 32             # analytic Gaussian covariance of multipoles, 1-2 min on a GPU
python scripts/bruteforce_multi.py 3000 256          # multi-sample bookkeeping vs exact sums (slow numpy part)
```
Any change to `integrals.py`, `driver.py` normalization or `plan_passes` must be followed by the brute-force
checks; a change to the post-processing by `tests/` and a rerun of `scripts/desi_dr2_run.py <dir> 32 2 None --skip-run`
on saved raw terms.

## Environments on Perlmutter

* `DESI-MG` conda env (`/pscratch/sd/w/willlake/desi/conda/envs/DESI-MG`): JAX with CUDA, lsstypes; the package is
  pip-installed editable here. No pycorr.
* `cosmodesi` (`source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main`): pycorr, fitsio,
  cosmoprimo, cucount, CPU-only JAX. Use it for `scripts/desi_dr2_prepare.py`, which saves plain arrays for the runs.
* DESI inputs and results: `/pscratch/sd/w/willlake/rascaljax/desi_dr2_<TRACER>_<CAP>_z<zmin>-<zmax>/`.
* `OMP_NUM_THREADS=1` is set in the login shell, so `nproc` reports 1; compute nodes have 128 cores and 4 A100s.

## Not implemented (by decision)

Legendre "accumulated" mode with survey-correction functions, the 3PCF, the power-spectrum mode.
