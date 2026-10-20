# Covariance of overlapping redshift bins

How RascalJax computes the joint covariance of the correlation functions of several redshift bins of one tracer
when the bins share part of their redshift range (and, as a special case, when they do not). Read `README.md`
first for the general picture; the module layout is in `CLAUDE.md` and the validation numbers in `BENCHMARK.md`.

## 1. The idea in one paragraph

RascalJax has no dedicated "overlapping bins" code path. It has one general mechanism: a **`Sample`** is a
membership-weighted view of a random **`Catalog`**. Two redshift bins of the same tracer are two samples on
*one* catalog, each with a per-particle membership weight (1 inside the bin, 0 outside, fractional allowed). The
sampling kernel draws particles from the catalog, never from a sample, and multiplies each accumulated term by
the membership weights of the samples it belongs to. The shot-noise terms of the covariance contain products of
weights *of the same particle* in two samples, `w_j^A w_j^B`, which are nonzero exactly where the bins overlap.
So the cross terms between overlapping bins come out of the same integrals as the auto terms, with no special
case anywhere. In the multi-tracer formulas of RascalC the Kronecker delta that says "tracers X and Y share
particles" is replaced by the membership product `w_j^X w_j^Y` on one catalog.

## 2. Setup: catalog, samples, correlations

```python
import rascaljax as rj

cat = rj.Catalog((ra, dec, dist), weights, n_data=n_gal_total, jackknife=labels, position_type="rdd")
sA = rj.Sample.from_redshift_range(cat, z_random, 0.4, 0.7, "zA")   # membership 1 for 0.4 <= z < 0.7
sB = rj.Sample.from_redshift_range(cat, z_random, 0.6, 0.9, "zB")   # membership 1 for 0.6 <= z < 0.9
raw = rj.run_cov([sA, sB], {(0, 0): corr_AA, (0, 1): corr_AB, (1, 1): corr_BB}, mode="legendre", max_l=4, ...)
res = rj.post_process(raw)
cov = res["full_theory_covariance"]      # joint, ordering (correlation, s, ell), correlations AA, AB, BB
```

* `Catalog` (`rascaljax/catalog.py`): the randoms of the *union* of the redshift ranges, weights normalized to unit
  sum, and **one** `n_data`, the number of data galaxies represented by the whole catalog. Each random particle
  stands for `g = n_data / n_random` galaxies.
* `Sample`: `catalog` plus `membership`, one weight in [0, 1] per catalog particle. The effective weight of particle
  `i` in sample `S` is `w_i^S = catalog.weights[i] * membership[i]`. `Sample.from_redshift_range` builds the mask
  `(z >= z_min) & (z < z_max)` from a per-particle redshift array. Particles in the overlap have membership 1 in
  both samples.
* `correlations`: one `CorrelationInput` per *pair of samples*, keyed `(i, j)` with `i <= j`. **All pairs are
  required**, including the cross pair: the cross-correlation function `xi_AB` enters the 4-point term of every
  mixed block and the cross pair counts `RR^AB` normalize them. Each input carries its own normalized `RR`
  counts in the covariance bins, its own `xi` table, and optionally its own jackknife `xi` and `RR` per region.
  All inputs must share the `s` edges and the number of `mu` bins.
* **Normalization of the counts.** The kernel's pair counts are sums of effective weights normalized to unit sum
  over the *whole catalog*. Counts measured with `pycorr` or `lsstypes` on the randoms selected for each bin are
  normalized by the selected samples' weight sums instead, so they are larger by `1 / (F_A F_B)`, where `F_S` is
  the fraction of the catalog's weight in sample `S` (about 0.6 for a bin holding 60% of the randoms). Declare the
  convention with `CorrelationInput(..., rr_normalization=...)`: `"sample"` (the default) for estimator output,
  which `run_cov` multiplies by `F_A F_B`; `"catalog"` for counts built directly from `Sample.effective_weights`
  (the notebook and the brute-force script do this). The two coincide for a sample covering its whole catalog,
  so single-sample runs are unaffected. `run_cov` logs the Monte Carlo estimate of every correlation's `RR`
  divided by the input (`RR check`) and warns when it deviates by more than 5%: counts in the wrong convention
  show up as a ratio equal to `F_A F_B`, and a mismatch between randoms and samples shows up there too.

Any number of samples and catalogs can be mixed: several redshift bins of several tracers are several samples
on several catalogs. Samples on different catalogs never share particles.

## 3. What is computed

Write `A, B, C, D` for samples, `c1 = (A, B)` and `c2 = (C, D)` for two of the requested correlations, `a` and `b`
for `(s, mu)` bins, `Theta_a(ij)` for the indicator that pair `ij` falls in bin `a`, and `xi_XY` for the
cross-correlation function of samples `X` and `Y` (`xi_XX` is the auto-correlation). All sums run over
*distinct* catalog particles. The covariance block between `xi^c1_a` and `xi^c2_b` is the sum of three terms,
each estimated by importance sampling and divided by the input counts `RR^c1_a RR^c2_b`:

```
C4[c1,c2][a,b] = sum_{i,j,k,l}  w_i^A w_j^B w_k^C w_l^D  Theta_a(ij) Theta_b(kl)
                                [ xi_AC(ik) xi_BD(jl) + xi_AD(il) xi_BC(jk) ]

C3[c1,c2][a,b] = (1/g) sum_{i,j,k} Theta_a(ij) Theta_b(kj)
                                [ w_i^A (w_j^B w_j^D) w_k^C xi_AC(ik) + w_i^A (w_j^B w_j^C) w_k^D xi_AD(ik)
                                + w_i^B (w_j^A w_j^D) w_k^C xi_BC(ik) + w_i^B (w_j^A w_j^C) w_k^D xi_BD(ik) ]

C2[c1,c2][a,b] = delta_ab (1/g^2) sum_{i,j} Theta_a(ij)
                                w_i^A w_j^B ( w_i^C w_j^D + w_i^D w_j^C ) ( 1 + [xi_AB(ij) + xi_CD(ij)] / 2 )
```

In `C2` both galaxies of the pair are shared, so the pair belongs to both correlations and its `1 + xi` is taken
as the average of the two correlations' tables. This keeps the term independent of the order in which the samples
are listed (before this choice the block AA x BB used `xi_AA` or `xi_BB` depending on the labelling). The two
tables are the same function unless the measured correlation functions of the bins differ through redshift
evolution or noise, in which case neither is strictly the correlation at the pair's redshift and the average is the
symmetric choice. `C3` and `C4` need no such choice: their `xi` always connects two unshared particles whose
samples are fixed by the term.

These are the RascalC terms (Philcox et al. 2020, arXiv:1904.11070) with the data treated as a Poisson sampling
of the randoms: `C4` has four distinct galaxies, `C3` one galaxy `j` common to both pairs, `C2` both galaxies in
common. The powers of `1/g` make `C3` scale as `1/n_data` and `C2` as `1/n_data^2`, independent of the number of
randoms.

### Where the overlap enters

The three terms see the overlap in three different ways. With `O = A ∩ B` the overlap region:

| block | `C4` couples through | `C3`: shared particle `j` must lie in | `C2`: nonzero pairs `(i, j)` |
|---|---|---|---|
| AA x AA | `xi_AA xi_AA` | A | both in A |
| AA x AB | `xi_AA xi_AB` | O (two cases), A (two cases) | one in A, the other in O |
| AA x BB | `xi_AB xi_AB` | O (all four cases) | both in O |
| AB x AB | `xi_AA xi_BB`, `xi_AB xi_AB` | A, B, O, O | (A, B) or (O, O) |
| AB x BB | `xi_AB xi_BB` | B (two cases), O (two cases) | one in O, the other in B |
| BB x BB | `xi_BB xi_BB` | B | both in B |

So for the block AA x BB the whole shot-noise part (`C3` and `C2`) comes from the overlap, while `C4` couples
the two bins through the cross-correlation `xi_AB` even if the bins are disjoint. If the bins do not overlap
(`O` empty), the mixed `C2` blocks vanish identically and the mixed `C3` blocks keep only the cases with the
shared particle in `A` or `B` and the third particle in the other bin, i.e. boundary contributions within the
range of `xi_AB`.

## 4. Implementation walk-through

### 4.1 Membership arrays on the device (`driver.run_cov`)

For each catalog, the effective weights of every sample on that catalog are stacked into one array
`membership` of shape `(n_samples, N)` in the cell-sorted particle order of the grid and stored in the
catalog's `GridArrays`. Rows of samples that live on another catalog are zero. Sample indices are global, so the
kernel can index `membership[A]` for any sample `A` on any grid.

The input counts are converted to the kernel's normalization here (`rr_normalization`, section 2); the converted
counts are what `RawCovariance.rr_input` holds and what the jackknife disconnected term uses.

The `xi` lookups are indexed by sample pair: `xi_pair_index[A, B]` gives the row of the fine `xi` table built
from the `CorrelationInput` of `(A, B)`. The importance-sampling kernel for drawing `k` around `i` and `l` around
`j` is one per *catalog pair*, built from the maximum over the isotropic `xi` of all correlations on that catalog
pair (`_kernel_xi_iso`), so one kernel covers AA, AB and BB; the `w / p` weighting corrects for it. The optional
RascalC-style refinement of each `xi` table (`refine_xi_table`) draws pairs on the catalog and weights them by
the memberships of the two samples of that correlation, so the cross table is refined with `w^A` on one leg and
`w^B` on the other.

### 4.2 Pass planning (`driver.plan_passes`)

A *pass* is one run of the sampling kernel with a fixed catalog for each leg `i, j, k, l`. The planner works at the
catalog level only. For two samples on one catalog every leg is catalog 0, so **one pass serves all six blocks**;
it simply accumulates more terms. Running the planner for `plan_passes([0, 0], [(0, 0), (0, 1), (1, 1)])`:

```
pass 0: legs=(0, 0, 0, 0)
  c4_terms (A, B, C, D, corr_AB, corr_CD):
    (0,0,0,0,0,0) (0,0,0,1,0,1) (0,0,1,1,0,2) (0,1,0,1,1,1) (1,0,0,1,1,1) (0,1,1,1,1,2) (1,0,1,1,1,2) (1,1,1,1,2,2)
  t3_terms (X, Y, Y', Z, corr_XY, corr_ZY'):   11 distinct tensors
  c2_terms (A, B, C, D, corr_AB, corr_CD):      6 terms
  rr_terms (A, B, corr_AB):                     (0,0,0) (0,1,1) (1,1,2)
```

For every block `(c1, c2)` the planner records how to assemble it:

* **C4**: two terms, `(A, B, C, D)` and `(B, A, C, D)`. The kernel accumulates only the kernel-matched Wick
  contraction `xi_AC(ik) xi_BD(jl)` (`k` is drawn around `i` and `l` around `j` with `xi` kernels); swapping the
  first pair turns it into `xi_BC(ik) xi_AD(jl)`, which is the other contraction. For `A == B` the same term is
  counted twice. Do not add `xi_AD(il) xi_BC(jk)` per quad: it is unbiased but heavy-tailed (see `CLAUDE.md`).
* **C3**: the four sharing cases `(X, Y, Y', Z) = (A,B,D,C), (A,B,C,D), (B,A,D,C), (B,A,C,D)`, each a tensor
  `T[X, (Y, Y'), Z]` with the shared particle `j` carrying `w_j^Y w_j^Y'`. A case is kept only if `cat(Y) == cat(Y')`
  (otherwise the particle cannot be shared). Identical tensors are deduplicated by `add_unique`, which is why the
  block AA x BB maps to the same tensor `(0, 0, 1, 1)` four times. The record also stores the catalog of the shared
  particle, needed for the shot-noise rescaling.
* **C2**: one term `(A, B, C, D)`, kept only if the two correlations are on the same catalog pair.
* **RR**: a Monte Carlo estimate of each correlation's pair counts, for checks.

With `n` samples on one catalog the number of terms grows (roughly `n^4` for C4) but the number of passes stays
one. By contrast, the same two bins as two *catalogs* need eight passes with one or two terms each, and lose the
overlap entirely (section 5).

### 4.3 The kernel (`integrals.accumulate_batch`)

Per batch of primaries `i` of catalog `legs[0]`, the kernel draws `N2` secondaries `j` with the `1/r^2` cell kernel,
`N3` tertiaries `k` around `i` and `N4` quaternaries `l` around `j` with the `xi` kernels, always from the full
catalog. It then gathers all sample weights of the drawn particles at once:

```python
w_i = gi.membership[:, primary_idx]   # (n_samples, B)
w_j = gj.membership[:, idx_j]         # (n_samples, B, N2)
w_k = gk.membership[:, idx_k]         # (n_samples, B, N2, N3)
w_l = gl.membership[:, idx_l]         # (n_samples, B, N2, N3, N4)
```

and evaluates each planned term as a product of the selected rows times the `xi` lookups of the right sample
pairs, divided by the draw probability and by `RR^c1_a RR^c2_b`:

```python
# C4 term (A, B, C, D): xi_AC(ik) xi_BD(jl)
v = w_i[A] * w_j[B] * w_k[C] * w_l[D] * xi(A, C, r_ik, mu_ik) * xi(B, D, r_jl, mu_jl) / p / (rr[ca, bin_a] * rr[cb, bin_b])
# T tensor (X, Y, Y', Z): shared particle j
v = w_i[X] * w_j[Y] * w_j[Y2] * w_k[Z] * xi(X, Z, r_ik, mu_ik) / p / (rr[ca, bin_ij] * rr[cb, bin_kj])
# C2 term (A, B, C, D): symmetric average of the two correlations' xi for the shared pair
xi_pair = 0.5 * (xi(A, B, r_ij, mu_ij) + xi(C, D, r_ij, mu_ij))
v = w_i[A] * w_j[B] * (w_i[C] * w_j[D] + w_i[D] * w_j[C]) * (1 + xi_pair) / p / (rr[ca, bin_a] * rr[cb, bin_a])
```

A particle outside sample `B` has `w^B = 0` and silently drops out of every term that needs it in `B`; a particle in
the overlap contributes to both. Because all legs are on the same catalog, the exclusions `j != i`, `k != i, j`
and `l != i, j, k` are applied by index comparison, so the sums are over distinct particles as required.
Contributions are scattered into the full `(n_bins, n_bins)` matrix of `(s, mu)` bins and projected to
multipoles afterwards. With jackknives, the same products are also accumulated with the RascalC jackknife weight
tensor built from the per-correlation region weights `w_aA = RR_aA / sum_A RR_aA` of `c1` and `c2`.

### 4.4 Normalization and assembly (`driver._normalize`, `run_cov`)

The accumulated sums are divided by the number of draws per primary, so they estimate sums over all primaries,
and by the galaxies-per-random factors of the leg catalogs: `C2` by `g_i g_j`, `T` by `g_j` (the shared particle's
catalog), `C4` by nothing. For two bins on one catalog `g` is the same number for both, `n_data / n_random` of the
whole catalog.

The blocks are then assembled into a `RawCovariance` with arrays indexed `[subsample, corr1, corr2, bin1, bin2]`:

* `c4`: the two C4 terms added;
* `c3`: the four tensors added, transposed where the pass had the legs in the other order, stored with an extra
  leading axis over the **catalog of the shared particle** (one entry here);
* `c2`: the diagonal `(s, mu)` term projected to the output bins, stored with an extra leading axis over
  **catalog pairs** (one entry here).

Off-diagonal blocks `(c1, c2)` and `(c2, c1)` are each other's transposes; diagonal blocks are symmetrized.

### 4.5 Post-processing (`postprocess.py`)

`CovarianceTerms.total` builds the model

```
C(alpha) = c4 + sum_cat alpha_cat c3[cat] + sum_(a,b) alpha_a alpha_b c2[(a,b)]
```

with one shot-noise rescaling `alpha` per **catalog**, not per sample: both redshift bins share the same `alpha`.
With jackknife inputs the target is the joint jackknife covariance of all requested correlations, cross blocks
included (`jackknife_data_covariance` on the stacked `jack_xi` of AA, AB and BB), and `alpha` is fitted by the
RascalC likelihood on the full joint matrix (`alpha_method="joint"`, default) or on the correlations of that
catalog alone (`"separate"`). The jackknife disconnected term is built per correlation pair from the `EE/RR`
per-region arrays. The precision matrix with the quadratic bias correction, `N_eff` and the convergence checks
act on the joint matrix. Output ordering is `(correlation, s, ell)` with correlations in sorted key order;
`raw.binning.to_pycorr_order(cov, n_corr=3)` gives `(correlation, ell, s)`.

## 5. Why not two catalogs?

The same two bins can be passed as two `Catalog`s (one `Sample` each). Then the planner treats them as two
tracers: eight passes, and no term ever has one particle in both catalogs. The consequences, verified in
`scripts/bruteforce_multi.py`:

* the mixed `C2` blocks (AA x AB, AA x BB, AB x BB) are exactly zero;
* the mixed `C3` blocks keep only boundary contributions (third particle across the boundary within `xi_cut`);
* the `C4` blocks are the same as in the one-catalog case.

For genuinely disjoint bins this is correct and equivalent, just slower. For overlapping bins the two-catalog
treatment double-counts the overlap particles as independent and misses all the shot-noise cross terms. Use
one catalog with memberships.

## 6. Assumptions and limitations

* **One `n_data` per catalog.** The shot noise of each bin is `n_data / n_random` of the whole catalog. This is
  right when the randoms follow the data `n(z)`, so that galaxies per random is the same in every redshift slice
  (true for DESI randoms). `n_data` must count the galaxies in the full range of the catalog, not in one bin.
  Per-sample shot-noise levels or per-sample `alpha` are not supported.
* **Counts must be in the declared convention** (section 2). Estimator output on the selected randoms is
  `"sample"`-normalized, counts from the effective weights are `"catalog"`-normalized; mixing them up scales whole
  blocks by powers of the weight fractions, which no shot-noise rescaling can absorb because `C4` scales too. The
  `RR check` line of the log is the guard.
* **All sample pairs need inputs.** `run_cov` raises if any pair `(i, j)` lacks a `CorrelationInput`, and
  `CorrelationInput` requires strictly positive `RR` in every bin. Two bins separated by a gap larger than `s_max`
  have no cross pairs and cannot be handled as is (a cross input with vanishing counts would be needed).
* **Same binning everywhere.** All correlations share the `s` edges and the number of `mu` bins; jackknife mode
  needs jackknife inputs for all correlations and labels for the catalog.
* **Only the Gaussian plus shot-noise terms.** As in RascalC, no connected 4-point function.
* The membership is a plain weight, so soft redshift boundaries or per-bin weighting schemes are a matter of
  passing a fractional `membership` instead of a mask; nothing else changes.

## 7. Recipe for DESI data

1. Build one random catalog covering the union of the bins (e.g. 0.4 to 1.1 for three LRG bins) with the usual
   FKP weights, the per-random redshift, the jackknife region labels, and `n_data` = total galaxies in the union.
2. Make one `Sample.from_redshift_range` per bin.
3. Measure or load the pair counts of every pair of bins in the covariance binning: the autos `RR^AA`, `RR^BB`, ...
   and the cross counts `RR^AB` between the randoms of bin A and those of bin B, with their correlation functions
   and, for the jackknife calibration, the per-region `xi` and `RR`. `io_pycorr.correlation_input_from_pycorr`
   and `io_lsstypes.correlation_input_from_lsstypes` convert single estimators, autos and the cross alike. Such
   estimators are normalized by the selected samples' weights, which is the default `rr_normalization="sample"`:
   pass them as they come and let `run_cov` rescale them, without rescaling by hand as well. Then read the
   `RR check` line of the log: the ratios must be 1 within the Monte Carlo noise. `scripts/desi_dr2_prepare.py`
   currently prepares one tracer and one bin only, so the cross inputs have to be added for a multi-bin run.
4. Call `run_cov` with the samples and the dictionary of all pairs, then `post_process`. The result is the joint
   covariance of all requested correlations, including the cross-correlation between bins; if only the autos are
   wanted, slice the corresponding blocks (they are correct on their own, the cross-correlation only needs to be
   present in the integrals).

## 8. Validation

`scripts/bruteforce_multi.py` evaluates the exact sums of section 3 with dense `N x N` matrices for two overlapping
slabs of a periodic box and compares every block with the Monte Carlo estimates; the results are in section 2 of
`BENCHMARK.md` (`C2` within 0.5%, `C3` within 1 to 4%, `C4` within its Monte Carlo errors, for all six blocks),
followed by the disjoint two-catalog case. `notebooks/03_overlapping_redshift_bins.ipynb` shows the same setup
end to end, including building the three sets of pair counts with a KD-tree and inspecting the separate terms.
Two unit tests in `tests/test_basic.py` cover the normalization conversion (sample- and catalog-normalized inputs
give identical results, and a wrong declaration is flagged by the `RR check`) and the invariance of the 2-point
term under a relabelling of the samples with different `xi` tables.
