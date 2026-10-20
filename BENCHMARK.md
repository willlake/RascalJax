# RascalJax validation and benchmarks

All numbers from 2026-10-12 on Perlmutter (A100-80GB nodes; RascalC C++ v3.2 from `~/Packages/RascalC`).
"MC error" means the scatter of the independent subsamples divided by the square root of their number.

## 1. Exact sums on small catalogs (`scripts/bruteforce_check.py`)

Uniform randoms in a 300 Mpc/h box (4000 particles, `n_data = 2000`), s bins 20-40-60, 2 mu bins, model xi from a
toy P(k). The 2-, 3- and 4-point sums are evaluated exactly with dense N x N matrices and compared with the
Monte Carlo estimates (512 loops, N2 = N3 = N4 = 10).

| term (diagonal) | RascalJax / exact | MC error | RascalC C++ periodic / exact | RascalC C++ aperiodic / exact |
|---|---|---|---|---|
| C2 | 1.000 - 1.004 | 0.2-0.4% | 1.000 - 1.004 | 0.995 - 1.001 |
| C3 | 0.989 - 1.004 | 0.4% | **1.028 - 1.039** | 0.988 - 1.007 |
| C4 | 0.97 - 0.99 | 1.5-5% | **1.04 - 1.07** | 0.98 - 1.03 |

RascalJax is unbiased in both periodic and aperiodic geometry. The RascalC periodic binaries are high by about
3% in C3 and 6% in C4 (a bias proportional to xi and xi^2; the xi interpolant was verified identical to 1e-5), the
aperiodic binaries are unbiased. The variations with N2/N3/N4 and grid size did not change the C++ periodic bias.

## 2. Multi-sample bookkeeping (`scripts/bruteforce_multi.py`)

Two overlapping slabs of a periodic box as two samples on one catalog (3000 particles, 256 loops, kernel-matched
estimator; rerun on 2026-10-18 after the correction of the exact C4 reference, whose coincident-index terms had been
subtracted with full matrices instead of their diagonals, a <1% effect at this catalog size). All six covariance
blocks between the correlations AA, AB, BB agree with exact sums carrying the membership weights:

| block | C2 / exact | C3 / exact | C4 / exact |
|---|---|---|---|
| AA x AA | 0.998 - 1.003 (0.3%) | 0.999 - 1.024 (1%) | 0.97 - 1.05 (2-3.5%) |
| AA x AB | 0.995 - 1.002 (0.5%) | 1.005 - 1.015 (1.5%) | 0.91 - 1.03 (4-5%) |
| AA x BB | 0.992 - 1.001 (0.6%) | 1.000 - 1.042 (2%) | 0.92 - 1.05 (4-6%) |
| AB x AB | 0.995 - 1.002 (0.4%) | 0.996 - 1.017 (1.5%) | 0.91 - 1.03 (3-4%) |
| AB x BB | 0.994 - 1.001 (0.5%) | 0.985 - 1.028 (1.5%) | 0.92 - 1.03 (3.5-4%) |
| BB x BB | 0.994 - 1.000 (0.3%) | 0.993 - 1.002 (1%) | 0.93 - 1.02 (2.5%) |

(ranges over the diagonal elements; MC errors in parentheses). The low end of every C4 range is the first
separation bin (20-40 Mpc/h, mu < 0.5), 2-2.5 sigma low in the mixed blocks; all blocks are built from the same
quads, so this is one correlated fluctuation of the run (seed 5), not six. With seed 11 the same bin comes out
1.03 - 1.05 (3-5%) in every block and the other bins 0.95 - 1.06, i.e. C2 within 0.5%, C3 within 1.5% and C4
within 1-2 sigma for both seeds.

The same particles split into two disjoint z-slabs treated as two *catalogs* (two sampling passes planned by
`plan_passes`): the auto blocks AA x AA, BB x BB and AB x AB agree with the exact sums (C2 within 0.5%, C3 within
1-3%, C4 within MC errors of 3-9%); the C2 of the mixed blocks (AA x AB, AA x BB, AB x BB) is exactly zero in both,
as it must be for disjoint catalogs; their C3 and C4 are small boundary contributions sampled with 10-50% errors
and are consistent with the exact sums at that level.

## 3. Analytic Gaussian covariance (`scripts/gaussian_box.py`)

Periodic 1000 Mpc/h box, 2e5 randoms, `n_data = 1e5`, isotropic toy P(k), multipoles 0 and 2 projected from 20 mu
bins, 8 s bins of 10 Mpc/h from 20 to 100. Against C_ll(s_a, s_b) = 2(2l+1)/V int k^2 dk/(2 pi^2) (P + 1/n)^2
jbar_l jbar_l (Grieb et al. 2016):

* C3 (the 2P/n term): ratio 0.99-1.01 on all diagonal and off-diagonal elements;
* C2 (the 1/n^2 term): ratio 1.036 uniformly, which is the k-integral cutoff of the analytic delta function (not
  the code; C2 agrees with the exact sums above);
* C4 (the P^2 term): with the kernel-matched estimator, the ell = 0 diagonal agrees to 0.98 - 1.02 in every bin at
  32 loops (6.4e9 quads); with the earlier estimator including the second Wick contraction per quad the same run
  scattered between 0.65 and 1.5 at s > 60 Mpc/h.

## 4. Head-to-head with the RascalC C++ code

Same catalog, xi table and binning as section 3 (periodic box, 32 loops, Legendre projected mode):

| term | RascalJax / RascalC | notes |
|---|---|---|
| C2 | 0.999 - 1.003 (RMS rel. diff. 0.2%) | |
| C3 | 0.94 - 0.96 | the C++ periodic bias of section 1 |
| C4 | 0.77 - 1.04 | C++ periodic bias plus MC noise |

Jackknife mode (27 cubic regions, `s_mu` binning): c2j within 0.4%, c3j within 2.5%, c4j within the MC noise
(pulls with RMS 1.0-1.4). The per-region EE/RR arrays of the disconnected term differ by a factor that depends on
the cell occupancy, because the C++ weights those pairs with the class-conditional cell count (`np1 * sln1`) while
drawing from all particles; RascalJax uses the actual draw probability.

## 5. DESI DR2 LRG 0.4 < z < 0.6, pre-reconstruction, per Galactic cap

Inputs: official desipipe pycorr counts (`allcounts_LRG_{NGC,SGC}_z0.4-0.6_default_FKP_lin_nran8_njack60_split20`),
covariance bins 20-200 Mpc/h in 4 Mpc/h steps, 100 mu bins after wrapping, multipoles 0, 2, 4; xi table from the
same counts rebinned to 2 Mpc/h x 10 mu bins; one random file per cap cut to the redshift bin (NGC: 3.9e6 randoms
for 686k galaxies; SGC: 2.0e6); 60 angular K-means jackknife regions fitted on the cap's data as in desipipe;
N2, N3, N4 = 5, 10, 20. References: the official RascalC covariance (`cov_2pt/rascalc/v1.1`) and the sample
covariance of 1007 GLAM-Uchuu altmtl mocks (measured to s = 180, so compared on the first 40 s bins).
Metrics as in Rashkovetskyi et al. 2025: R_inv (RMS eigenvalue of C1^-1 C2 - 1) and reduced chi2 - 1, quoted
as (C1 = first, C2 = second) and the reverse.

### Estimator note

The first version of the 4-point kernel accumulated both Wick contractions per quad, xi(ik) xi(jl) + xi(il) xi(jk).
The second contraction uses draws that are not kernel-matched (l is drawn around j, not i) and is heavy-tailed: on
DESI geometry it dominated the variance at large s, the covariance was not positive definite even after 256 loops
(1e12 quads, N_eff = 550) and the jackknife fit failed. The kernel now accumulates only xi(ik) xi(jl), and the
other contraction comes from the term with the first pair's samples swapped (RascalC's strategy, generalized to
any samples). Both versions are unbiased against the exact sums (section 1).

### SGC, 32 loops (6.4e10 quads, 4 minutes on 2 A100s), kernel-matched estimator

| quantity | value |
|---|---|
| shot-noise rescaling (jackknife fit) | 1.017 |
| positive definite | yes |
| N_eff | 6352 |
| convergence check (halves of the subsamples) R_inv | 0.54 - 0.62 |
| official RascalC vs RascalJax, ell = 0, 2, 4: R_inv / chi2_red - 1 | 0.21, 0.22 / 0.016, 0.028 |
| official RascalC vs RascalJax, ell = 0, 2: R_inv / chi2_red - 1 | 0.17, 0.17 / 0.020, 0.008 |
| RascalJax vs 1007 GLAM mocks, ell = 0, 2, 4: R_inv / chi2_red - 1 | 0.41, 0.51 / 0.009, 0.17 |
| official RascalC vs the same mocks, ell = 0, 2, 4 | 0.35, 0.45 / -0.007, 0.15 |
| RascalJax vs mocks, ell = 0, 2 | 0.33, 0.40 / -0.014, 0.13 |
| official RascalC vs mocks, ell = 0, 2 | 0.29, 0.36 / -0.014, 0.11 |
| mean reduced chi2 of the mock xi (0, 2, 4) with the RascalJax precision matrix | 1.009 |

The diagonal agrees with the official covariance to 2-4% in all three multipoles below 100 Mpc/h (RascalJax
slightly lower) and within the noise above. Against the mocks both codes show the same pattern: the ell = 2
variance of the mocks is 10-35% larger below 50 Mpc/h (nonlinear RSD beyond the Gaussian + shot-noise model) and
the ell = 0 variance 5-20% smaller than the analytic ones between 80 and 180 Mpc/h.

### SGC, 256 loops (5.2e11 quads, 32 minutes on 2 A100s), kernel-matched estimator

| quantity | value |
|---|---|
| shot-noise rescaling (jackknife fit) | 1.013 |
| positive definite, N_eff | yes, 52 000 |
| convergence check (halves of the subsamples) R_inv / chi2_red - 1 | 0.15 / 0.01 |
| diagonal RascalJax / official, all s, ell = 0 (ell = 2) | 0.985 (0.98 - 0.99), uniform in s |
| official vs RascalJax, ell = 0, 2, 4: R_inv / chi2_red - 1 | 0.086, 0.083 / 0.0002, 0.007 |
| official vs RascalJax, ell = 0, 2 | 0.071, 0.068 / 0.0007, 0.004 |
| RascalJax vs 1007 GLAM mocks, ell = 0, 2, 4 | 0.358, 0.456 / -0.004, 0.148 |
| official RascalC vs the same mocks, ell = 0, 2, 4 | 0.353, 0.454 / -0.007, 0.149 |
| RascalJax vs mocks, ell = 0, 2 | 0.297, 0.358 / -0.012, 0.110 |
| official RascalC vs mocks, ell = 0, 2 | 0.294, 0.357 / -0.014, 0.110 |
| mean reduced chi2 of the mock xi (0, 2, 4) with the RascalJax precision matrix | 0.996 |

The converged RascalJax covariance is 1.5% below the official one everywhere (consistent with the 0.4% difference in
the fitted shot-noise rescaling plus the different random subset and refinement target), and is statistically
indistinguishable from the official RascalC covariance in its agreement with the mocks.

### NGC, 256 loops (1.0e12 quads, 65 minutes on 2 A100s), kernel-matched estimator

| quantity | value |
|---|---|
| shot-noise rescaling (jackknife fit) | 0.985 |
| positive definite, N_eff | yes, 76 000 |
| convergence check (halves of the subsamples) R_inv / chi2_red - 1 | 0.12 / 0.008 |
| diagonal RascalJax / official, all s, ell = 0 | 0.98 - 0.985, uniform in s |
| official vs RascalJax, ell = 0, 2, 4: R_inv / chi2_red - 1 | 0.072, 0.070 / -0.006, 0.011 |
| official vs RascalJax, ell = 0, 2 | 0.060, 0.059 / -0.007, 0.011 |
| RascalJax vs 1007 GLAM mocks, ell = 0, 2, 4 | 0.357, 0.499 / -0.036, 0.194 |
| official RascalC vs the same mocks, ell = 0, 2, 4 | 0.351, 0.501 / -0.045, 0.202 |
| RascalJax vs mocks, ell = 0, 2 | 0.295, 0.375 / -0.041, 0.143 |
| official RascalC vs mocks, ell = 0, 2 | 0.293, 0.380 / -0.049, 0.152 |
| mean reduced chi2 of the mock xi (0, 2, 4) with the RascalJax precision matrix | 0.964 |

### GCcomb = NGC + SGC combined (`scripts/desi_dr2_combine.py`)

RR-weighted combination of the two converged per-cap covariances, against the official GCcomb RascalC
covariance and the GCcomb mock covariance (1007 mocks):

| quantity | value |
|---|---|
| diagonal RascalJax / official, ell = 0 (ell = 2) | 0.982 - 0.985 (0.972 - 0.984) |
| official vs RascalJax, ell = 0, 2, 4: R_inv / chi2_red - 1 | 0.060, 0.058 / -0.004, 0.008 |
| RascalJax vs mocks, ell = 0, 2, 4 | 0.356, 0.475 / -0.022, 0.172 |
| official RascalC vs mocks, ell = 0, 2, 4 | 0.352, 0.477 / -0.028, 0.177 |

Summary: on DESI DR2 data RascalJax reproduces the official RascalC covariance to R_inv ~ 0.06-0.09 (a 1.5-2%
lower overall amplitude, consistent with the different random subset, refinement target and the fitted
rescalings 0.985 / 1.013), and both agree with the mock sample covariance to the same degree. The remaining
RascalJax-vs-mock differences (ell = 2 at s < 50 Mpc/h, ell = 0 at 80-180 Mpc/h) are shared with the official code
and come from the Gaussian + shot-noise model, not from the sampling.

## 6. DESI DR2 LRG, overlapping redshift bins 0.4 < z < 0.55 and 0.45 < z < 0.6 (NGC, pre-reconstruction)

First validation of the multi-sample covariance (`OVERLAPPING_REDSHIFT_BINS.md`) against mocks: two `Sample`s of one
catalog. The catalog is the 0.4 < z < 0.6 random set of GLAM-Uchuu altmtl mock 150 (3.0e6 of the 1.57e7 randoms of
4 files, `n_data` = 694 911 galaxies in 0.4-0.6; bin A holds 480 667, bin B 558 236, and the shared range 0.45-0.55
about two thirds of each). Inputs (`scripts/desi_zbins_measure_mocks.py`, `desi_zbins_prepare.py`,
`desi_zbins_run.py`): pycorr counts (cucount on GPUs, Corrfunc on CPUs) in 1 Mpc/h x 200 mu bins, Landy-Szalay with
the same 2 random files for DR and RR (see the note below); covariance bins 40-160 Mpc/h in 4 Mpc/h steps, 100 mu
bins, multipoles 0, 2, 4; xi tables (2 Mpc/h x 10 mu bins) averaged over, and RR counts taken from, 5 mocks measured
to 250 Mpc/h with all three correlations; sample covariance of the (AA, BB) data vector from 350 mocks measured to
160 Mpc/h (AB is not part of the BAO fit). Run: 256 loops, 16 loops per subsample, N2, N3, N4 = 5, 10, 20, 7.7e11
draws, 72 minutes on 4 A100-40GB (1.8e8 draws/s). The shot-noise rescaling is fitted to the mock covariance (the mock
counts carry no jackknives). Metrics as in section 5, plus D_KL; "expected" values are what the sampling noise of
350 mocks alone produces (Wishart draws from the RascalJax covariance, 104 bins).

| quantity | value |
|---|---|
| RR check (Monte Carlo estimate / input counts, median), AA, AB, BB | 0.999, 0.999, 1.000 |
| shot-noise rescaling, fit on the AA block (fit on AA and BB) | 1.229 (1.216) |
| (AA, BB), ell = 0, 2, 48-152 Mpc/h: positive definite, N_eff | yes, 145 000 |
| same block, halves of the subsamples: R_inv / D_KL / chi2_red - 1 | 0.078 / 0.15 / 0.006 |
| same block, min eigenvalue of C2^-1/2 C4 C2^-1/2 (threshold) | -0.12 (-1.5) |
| mean reduced chi2 of the 350 mock xi with the RascalJax precision, alpha = 1 (AA, BB, joint) | 1.268, 1.240, 1.255 |
| same with alpha = 1.229 | 0.989, 0.971, 0.980 |
| mocks vs RascalJax, (AA, BB) ell = 0, 2: R_inv / D_KL / chi2_red - 1 | 1.07, 0.53 / 14.0, 8.6 / 0.46, -0.025 |
| expected from the mock sampling noise alone (mean +- scatter) | 1.04 +- 0.03, 0.55 +- 0.01 / 13.7 +- 0.5, 8.8 +- 0.2 / 0.43 +- 0.01, 0.00 +- 0.01 |
| diagonal RascalJax / mocks, ell = 0, mean over s < 120 (s > 130): A, B | 1.12 (0.96), 1.05 (1.04); mock noise 0.05 (0.06) |
| diagonal RascalJax / mocks, ell = 2, 48-152 Mpc/h (A, B) | 0.92-1.12, 0.83-1.05 |
| mean offset of the AA x BB correlation coefficient, model - mocks, ell = 0 (ell = 2) | +0.017 (-0.009); expected scatter 0.017 |
| AA x BB correlation coefficient, ell = 0, 40 -> 160 Mpc/h: RascalJax (mocks) | 0.65 -> 0.53 (0.61 -> 0.60, +-0.03) |
| (AA, AB, BB) ell = 0, 2 joint matrix | not positive definite (min eigenvalue -4e-10, max 2e-4); not converged |

Summary: with one rescaling per catalog the RascalJax covariance of the two overlapping bins matches the 350-mock
covariance to the mock noise: the mocks' reduced chi2 is 0.99 / 0.97 / 0.98 (A, B, joint), the D_KL and the reduced
chi2 of the mock-vs-theory comparison are what the finite number of mocks predicts, and the cross-covariance of the
bins, the new ingredient, agrees at every separation and direction (1.0 sigma mean offset). The remaining differences
(monopole variance of A 12% high below 120 Mpc/h, 2.4 sigma; quadrupole of B 10-15% low on 100-140 Mpc/h) are the
usual size of RascalC-vs-mock differences. The AB-inclusive matrix has a near-null direction in C2 (with two thirds of
each bin shared, xi_AB is nearly a combination of xi_AA and xi_BB) that the 4-point sampling noise has not resolved at
256 loops; it is not needed for a BAO fit of (AA, BB).

**Note on the mock measurement (a trap worth recording).** A first sample covariance was built with DR from 10 random
files and RR from 2 (RR being the expensive count, passed as `R1R2`). Against the same RascalJax covariance it showed
the monopole variance 15-25% low above 130 Mpc/h (3.0 sigma for A) and the monopole cross-bin correlation coefficient
0.05-0.10 high (3.7 sigma), with the quadrupole and all transverse pairs fine. Three RascalJax runs with different xi
tables (per-correlation 5-mock tables, a common table, the 350-mock-mean table with a Kaiser extension) were
identical, so the model was excluded. The cause is the estimator: Landy-Szalay cancels the random catalog's edge
noise only when DR and RR use the same randoms. Measured on mock 150, two independent 2-file RR realizations differ
by 3-6e-4 at s = 150 Mpc/h for mu > 0.4 and by < 0.4e-4 transversally, which is 0.13 (s = 42) to 0.3 (s = 100-140)
of the mocks' monopole scatter: about 10% of the variance, coherent in s, radial, and independent between the bins
(different redshift edges). Recomputing DR with the 2 RR files (`scripts/remeasure_dr2.py`, 45 s per mock on an
A100) removed both anomalies. The rescaling rose from 1.145 to 1.227 because the Landy-Szalay random-noise term with
2 files (of order N_D/N_R = 1/11 of the shot-noise part) is now in the mocks' variance; a fully clean sample
covariance would use 4 or more identical random files for DD, DR and RR (5-20 GPU-hours for 350 mocks).
`desi_zbins_measure_mocks.py` now defaults to the same files for DR and RR, and `desi_zbins_remeasure_dr.py` applies
the fix to stored counts. The xi tables and RR inputs (5 long-range mocks) were remeasured the same way and RascalJax
rerun on them: the covariance changed by under 1%, the numbers above are from that final run. Outputs:
`/pscratch/sd/w/willlake/rascaljax/desi_zbins_LRG_z0.4-0.55_z0.45-0.6/NGC/rascaljax/` (first comparison in
`NGC_first/rascaljax/`, intermediate one in `NGC_dr2/`; figure `validation_NGC.png` in each; the contaminated
measurements are kept as `xi_*_NGC_dr10rr2.npy`). The executed notebook
`notebooks/04_desi_overlapping_bins_validation.ipynb` reproduces every number of this section, with the
correlation matrices, the chi2 distribution of the mocks, the Wishart expectations and the before/after comparison.

## 7. Performance

| setting | draws per second |
|---|---|
| RascalC C++, one CPU thread (Perlmutter, periodic box) | 2.5e7 |
| RascalJax, one A100-80GB (box, Legendre, 1000 draws per primary) | 2.0e8 |
| RascalJax, 2 A100s (per-device threads, box, incl. compilation) | 2.9e8 |
| RascalJax, 4 A100s (per-device threads, box, incl. compilation) | 5.0e8 |
| RascalJax, kernel without the final scatter (floor) | 3.6e8 per GPU |

A "draw" is one sampled pair, triple or quad; the scatter-add into the (s, mu)^2 matrix is 40% of the kernel.
Each device is driven by its own host thread (subsamples round-robin over devices); results are bitwise
identical for any number of devices. Before the threads, a shared host loop limited the scaling to 1.6x on 2 GPUs.
Storing positions in float32 (`positions_float32=True`) leaves the exact-sum agreement unchanged (C2 0.1%, C3 0.5%,
C4 1-3%, periodic and aperiodic) but does not speed the kernel up (1.83e8 vs 1.79e8 draws/s on one A100).
