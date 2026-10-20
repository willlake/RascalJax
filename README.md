# RascalJax

Covariance matrices of 2-point correlation functions by importance-sampled integration over random
catalogs: a rewrite of [RascalC](https://github.com/oliverphilcox/RascalC) in pure Python, with the sampling
and accumulation kernels in JAX (GPU or CPU).

Method references: Philcox et al. 2020 (arXiv:1904.11070), Rashkovetskyi et al. 2023 (arXiv:2306.06320),
Rashkovetskyi et al. 2025 (arXiv:2404.03007).

## What it computes

For every pair of requested correlation functions (auto- and cross-correlations of *samples*), the 2-, 3- and
4-point terms of the covariance of the Landy-Szalay estimator in `(s, mu)` bins, projected to even Legendre
multipoles (`mode="legendre"`, as `pycorr`/`cucount` do) or kept in `(s, mu)` bins (`mode="s_mu"`). The
shot-noise rescaling is fitted to a jackknife covariance (if jackknife pair counts and random labels are given)
or to a mock sample covariance, and the covariance, bias-corrected precision matrix and effective number of
samples are produced as in RascalC.

Differences from RascalC:

* **Samples with shared particles.** A `Sample` is a membership-weighted view of a random `Catalog`. Samples
  on the same catalog with overlapping memberships (e.g. overlapping redshift bins) share particles, and the
  shot-noise cross terms of their covariance come out of the same integrals. Samples on different catalogs
  (different tracers) are the RascalC multi-tracer case. Any number of samples and catalogs is supported.
* **One accumulation kernel.** All terms are accumulated in `(s, mu)` bins and projected to multipoles on the
  fly; the pass planner enumerates the kernel-matched 4-point contractions and the four particle-sharing cases of
  the 3-point term for any number of samples and catalogs.
* **Pure Python**: no compilation; `numpy`/`scipy` for setup and post-processing, JAX for the kernels.

Not (yet) implemented: the "accumulated" Legendre mode with survey-correction functions, the 3PCF, and
the power-spectrum mode.

## Install

```
pip install -e ~/Packages/RascalJax
```
Requires `numpy`, `scipy`, `jax` (with a CUDA build for GPUs). `lsstypes` is optional (input adapters).

## Usage

```python
import numpy as np, rascaljax as rj
from rascaljax.io_lsstypes import correlation_input_from_lsstypes
import lsstypes

counts = lsstypes.read("recon_particle2_correlation_smu_jackknife60_LRG_z0.4-0.6_GCcomb.h5")
corr = correlation_input_from_lsstypes(counts, s_slice=slice(20, 200), rebin=4)   # covariance bins 20-200, 4 Mpc/h
cat = rj.Catalog((ra, dec, dist), weights, n_data=n_galaxies, jackknife=labels, position_type="rdd")
raw = rj.run_cov([rj.Sample(cat)], {(0, 0): corr}, mode="legendre", max_l=4,
                 N2=5, N3=10, N4=20, n_loops=64, loops_per_sample=4, out_dir="out")
res = rj.post_process(raw, skip_s_bins=0, out_file="out/Rescaled_Covariance.npz")
cov = res["full_theory_covariance"]          # ordering (correlation, s, ell)
cov_pycorr = raw.binning.to_pycorr_order(cov)  # ordering (correlation, ell, s)
```

Overlapping redshift bins of one tracer:

```python
sA = rj.Sample.from_redshift_range(cat, z, 0.4, 0.7, "zA")
sB = rj.Sample.from_redshift_range(cat, z, 0.6, 0.9, "zB")
raw = rj.run_cov([sA, sB], {(0, 0): corr_AA, (0, 1): corr_AB, (1, 1): corr_BB}, ...)
```
Here `corr_AB` holds the cross pair counts and cross-correlation function between the two redshift bins. Pair counts
from `pycorr` / `lsstypes` measured on the randoms selected for each bin are normalized per sample, which is the
default `CorrelationInput(rr_normalization="sample")`; `run_cov` converts them to the catalog normalization of the
kernel (counts built from `Sample.effective_weights` are passed with `rr_normalization="catalog"`). The log's
`RR check` line compares the sampled pair counts with the inputs. How the cross terms between the bins are computed
is explained in `OVERLAPPING_REDSHIFT_BINS.md`.

## Examples

* `notebooks/01_quickstart_periodic_box.ipynb` – synthetic box, comparison with the analytic Gaussian covariance;
* `notebooks/02_desi_dr2_jackknife.ipynb` – DESI DR2 LRG per-cap run with jackknife calibration, comparison with the
  official RascalC covariance and the GLAM-Uchuu mocks, NGC+SGC combination;
* `notebooks/03_overlapping_redshift_bins.ipynb` – joint covariance of two overlapping redshift bins.
* `notebooks/04_desi_overlapping_bins_validation.ipynb` – the DESI DR2 LRG overlapping-bin covariance against 350
  GLAM-Uchuu mocks (BENCHMARK.md section 6): variance ratios, cross-bin correlation, metrics, convergence.

Validation scripts in `scripts/`: `gaussian_box.py` (analytic Gaussian covariance), `bruteforce_check.py` and
`bruteforce_multi.py` (exact sums on small catalogs, single and multi-sample), `compare_rascalc_box.py` /
`compare_rascalc_jackknife.py` (RascalC C++ head-to-head; note the C++ periodic mode is biased high by a few
percent in C3/C4, its aperiodic mode is not), `desi_dr2_*.py` (DESI DR2 pipeline).

## Layout

* `catalog.py` – catalogs and samples; `grid.py` – cell grid; `kernels.py` – importance-sampling kernels;
  `xi_table.py` – correlation function tables; `binning.py` – bins and Legendre projection;
* `integrals.py` – JAX kernels; `driver.py` – pass planning and `run_cov`; `postprocess.py` – covariance assembly,
  shot-noise rescaling, precision matrix; `io_lsstypes.py` – `lsstypes` adapters; `compat.py` – RascalC C++ interop.
* `scripts/` – validation scripts (analytic Gaussian box, RascalC comparison); `tests/` – unit tests (`pytest`).
