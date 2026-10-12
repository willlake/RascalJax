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

Two overlapping slabs of a periodic box as two samples on one catalog (3000 particles, 256 loops). All six
covariance blocks between the correlations AA, AB, BB agree with exact sums carrying the membership weights:

| block | C2 / exact | C3 / exact | C4 / exact |
|---|---|---|---|
| AA x AA | 0.998 - 1.003 (0.3%) | 0.999 - 1.024 (1%) | 0.95 - 1.05 (4-9%) |
| AA x AB | 0.995 - 1.002 | 1.005 - 1.015 (1.5%) | 0.93 - 1.27 (5-24%) |
| AA x BB | 0.992 - 1.001 | 1.000 - 1.042 (2%) | 0.94 - 1.6 (5-50%) |
| AB x AB | 0.995 - 1.002 | 0.996 - 1.017 (1.5%) | 0.93 - 1.21 (5-20%) |
| AB x BB | 0.994 - 1.001 | 0.985 - 1.028 (1.5%) | 0.95 - 1.46 (5-34%) |
| BB x BB | 0.994 - 1.000 | 0.993 - 1.002 (1%) | 0.86 - 1.24 (4-18%) |

(ranges over the diagonal elements; MC errors in parentheses; C4 at the largest separation is poorly sampled in
this tiny box).

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
* C4 (the P^2 term): agrees within its MC error (the diagonal at large s is the noisiest element of the estimator).

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

### Converged runs

256-loop runs per cap (NGC 1e12 quads, SGC 5e11) with the kernel-matched estimator are in progress; this section
will be completed with their numbers and the NGC+SGC combination (`scripts/desi_dr2_combine.py`).

## 6. Performance

| setting | draws per second |
|---|---|
| RascalC C++, one CPU thread (Perlmutter, periodic box) | 2.5e7 |
| RascalJax, one A100-80GB (box, Legendre, 1000 draws per primary) | 2.0e8 |
| RascalJax, 3 A100s (DESI NGC run) | 3.3e8 |
| RascalJax, kernel without the final scatter (floor) | 3.6e8 per GPU |

A "draw" is one sampled pair, triple or quad; the scatter-add into the (s, mu)^2 matrix is 40% of the kernel.
Multi-GPU scaling is limited by the single-threaded host dispatch (1.6x on 2 GPUs, 1.65x on 3).
