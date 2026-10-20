# RascalJax overlapping-redshift-bin code review

- **Review date:** 2026-10-19
- **Repository:** https://github.com/willlake/RascalJax
- **Reviewed commit:** [`30737fb507d892e4274848fb26998ee1cead41cc`](https://github.com/willlake/RascalJax/commit/30737fb507d892e4274848fb26998ee1cead41cc)
- **Starting document:** [`OVERLAPPING_REDSHIFT_BINS.md`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/OVERLAPPING_REDSHIFT_BINS.md)
- **Scope:** overlapping selections of a common parent catalogue; covariance contractions, normalization, correlation inputs, and numerical validation.

## Assessment

The basic masked-weight construction is consistent with the covariance derivation: samples on the same catalogue share galaxies, and products of their weights correctly select the shared-galaxy contributions. However, the documented end-to-end workflow has two significant correctness issues, and the multi-sample validation script has an incorrect reference calculation. These should be resolved before relying on the workflow for a scientific overlapping-bin analysis.

| ID | Priority | Finding | Evidence |
| --- | --- | --- | --- |
| R1 | High | Parent-normalized sampling weights are incompatible with unconverted sample-normalized RR counts from the adapters. | Controlled Poisson run: covariance reduced to 0.12425625 of the consistently normalized result. |
| R2 | High | The shared-pair term changes under sample relabeling when the bins have different input correlation tables. | Swapping samples changed the tested cross-block C2 by exactly 1.36363636. |
| R3 | Medium | The exact C4 reference subtracts two repeated-index contributions incorrectly. | Explicit enumeration disagrees with the repository reference; corrected expressions agree at floating-point precision. |

All 10 existing repository tests passed. They do not establish correctness for the failing cases described below. The review did not change the repository.

## R1. Inconsistent RR normalization in the documented adapter workflow

### Code locations

- [`catalog.py:85-86`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/rascaljax/catalog.py#L85-L86): catalogue weights are normalized to unit sum.
- [`catalog.py:149-151`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/rascaljax/catalog.py#L149-L151): effective sample weights are the catalogue weights times membership, without sample renormalization.
- [`driver.py:442-449`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/rascaljax/driver.py#L442-L449): these weights are passed to the kernel.
- [`driver.py:508-509`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/rascaljax/driver.py#L508-L509): supplied RR counts are used directly.
- [`io_pycorr.py:31-32`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/rascaljax/io_pycorr.py#L31-L32) and [`io_lsstypes.py:66-68`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/rascaljax/io_lsstypes.py#L66-L68): adapters pass through the estimator's normalized counts.

### Why this is incorrect

Let the unnormalized parent weights be w_i and define

$$
W_P=\sum_{i\in P}w_i,\qquad
u_i^A=\frac{w_iS_A(i)}{W_P},\qquad
u_i^B=\frac{w_iS_B(i)}{W_P}.
$$

The kernel requires pair counts in exactly this normalization:

$$
RR_{\mathrm{kernel},a}^{AB}
=\sum_{i\ne j}u_i^Au_j^B\Theta_a(i,j).
$$

An ordinary estimator measured on selected catalogues A and B instead normalizes its counts using those selected catalogues. Write

$$
f_A=\frac{\sum_iw_iS_A(i)}{W_P},\qquad
f_B=\frac{\sum_iw_iS_B(i)}{W_P}.
$$

Ignoring finite-random self-pair normalization corrections,

$$
RR_{\mathrm{kernel}}^{AB}
=f_Af_B\,RR_{\mathrm{selected}}^{AB}.
$$

The adapters do not perform this conversion. Consequently, directly supplying selected-sample counts to the current masked-weight kernel gives, for the covariance of the two auto-correlations,

$$
C_{\mathrm{wrong}}^{AA,BB}
=f_A^2f_B^2\,C_{\mathrm{correct}}^{AA,BB}.
$$

The same denominator error affects C2, C3, and C4. A shot-noise rescaling cannot generally repair it because C4 is also affected.

### Numerical reproduction

A controlled test used 80 equal-weight random points, 400 represented data galaxies, two selections of 48 points each with 16 shared points, one separation bin containing all distinct pairs, and xi = 0. The parent-normalized counts were

```text
RR_AA = 0.3525
RR_AB = 0.3575
RR_BB = 0.3525
```

Using selected-sample all-pairs auto-counts of one in the denominator, while leaving the kernel weights unchanged, produced

```text
C2_wrong / C2_correct:
  AA x AA = 0.12425625
  AA x BB = 0.12425625
  BB x BB = 0.12425625
```

This is exactly 0.3525 squared: approximately an eightfold underestimate. The two runs used identical sampling configurations and random seeds, so the ratio is not a Monte Carlo fluctuation. In this Poisson example C2 is the full covariance.

The optional pycorr and lsstypes packages were not executed in this test. The adapter issue is established by their direct pass-through code and by reproducing the normalization mismatch at the CorrelationInput boundary.

### Suggested correction

Choose and enforce one normalization convention throughout the input pipeline.

The smallest change preserving the current kernel is to convert full and jackknife RR counts to parent normalization. If the original raw weighted pair count is Q_AB, and the estimator supplies Q_AB / N_AB, the required conversion for the same parent weight normalization is

$$
RR_{\mathrm{kernel}}^{AB}
=RR_{\mathrm{estimator}}^{AB}\frac{N_{AB}}{W_P^2}.
$$

Use the actual estimator normalization N_AB, including its self-pair conventions, rather than treating f_A f_B as an exact finite-random identity. This formula assumes compatible raw weight and pair-ordering conventions; independently subsampled randoms must be mapped consistently to the same normalized selection measure.

An alternative is to normalize the effective sample weights consistently with the selected-sample counts. This also requires a consistent treatment of finite-random pair normalizations.

Add a diagnostic comparing raw.rr_estimate with raw.rr_input within sampling uncertainty. A systematic sample-dependent factor should fail validation.

**Scope:** the existing overlap notebook computes RR directly from parent-normalized weights and therefore avoids R1. The affected path is the documented recipe that combines those weights with ordinary adapter-produced selected-sample counts.

## R2. C2 depends on the order of samples when xi tables differ

### Code location

[`integrals.py:377-381`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/rascaljax/integrals.py#L377-L381):

```python
cache[(A, Bs)] = xi(A, Bs, r_ij, mu_ij).reshape(-1)
ww = (wi[A] * w_j[Bs] * (wi[C] * w_j[D] + wi[D] * w_j[C])).reshape(-1)
v = ww * (1.0 + cache[(A, Bs)]) * inv_p2 / rrnorm(ca, cb, ba, ba)
```

### Why this is incorrect

The correlation factor in the shared-pair contribution is taken from the first correlation in the covariance block.

For Cov(xi_AA, xi_BB), the code uses 1 + xi_AA on pairs whose two endpoints belong to the overlap. Reversing the sample order makes it use 1 + xi_BB for the same physical pairs. The required invariance is

$$
C_{ab}^{AA,BB}=C_{ba}^{BB,AA},
$$

independently of how the input samples are numbered.

The driver fills the reverse block by transposition, so the returned matrix can look symmetric while still failing the stronger test of invariance under relabeling and rerunning.

### Numerical reproduction

The same overlapping selections were supplied with constant correlation tables

$$
\xi_{AA}=0.1,\qquad \xi_{AB}=0.2,\qquad \xi_{BB}=0.5.
$$

After reversing the sample list and consistently remapping the correlation inputs, the C2 contribution to the same physical AA x BB covariance changed by

$$
\frac{C_{2,\mathrm{reversed}}}{C_{2,\mathrm{original}}}
=\frac{1+0.5}{1+0.1}=1.36363636.
$$

This exact factor appeared in both Monte Carlo subsamples. The toy amplitudes demonstrate the failure mechanism; they are not an estimate of its size in DESI data.

### Suggested correction and physical interpretation

For selections of one parent population, a shared pair needs one consistent underlying correlation model evaluated at the locations of that pair.

- For a stationary parent-field approximation, use a common xi table for all selections on that parent catalogue, and document or validate this restriction.
- For redshift evolution, provide a common model such as xi(s, mu; z_i, z_j), and evaluate it for the actual pair. The shared-pair term then samples the overlap automatically through the weight product.
- An overlap-specific effective correlation model may be used as an approximation, but its weighting and accuracy must be justified.

Independent bin-averaged xi_AA, xi_AB, and xi_BB measurements do not uniquely define the local correlation of shared pairs. Averaging the two C2 formulas would enforce symmetry but would not establish a physically correct model.

The current constant/common-xi overlap demonstration is unaffected by R2. The issue becomes relevant when following the recommendation to give each redshift-bin pair its own measured xi table.

## R3. Incorrect repeated-index subtractions in the C4 reference

### Code location

[`scripts/bruteforce_multi.py:104-116`](https://github.com/willlake/RascalJax/blob/30737fb507d892e4274848fb26998ee1cead41cc/scripts/bruteforce_multi.py#L104-L116), specifically the definitions of jk1 and jl2:

```python
jk1 = np.sum(Ma * X * (Mb @ X.T).T)
jl2 = np.sum(Ma * X * (Mb.T @ X.T).T)
```

### Derivation of the correction

For the first Wick contraction,

$$
T_1=\sum_{ijkl}M^a_{ij}M^b_{kl}X_{ik}X_{jl}.
$$

Its k = j contribution is

$$
\sum_{ij}M^a_{ij}X_{ij}\sum_lM^b_{jl}X_{jl}
=\sum_{ij}M^a_{ij}X_{ij}(M^bX^T)_{jj}.
$$

The required factor is the diagonal indexed by j, broadcast over i. The same reasoning applies to the l = j contribution of the other Wick contraction.

The corrected expressions are

```python
jk1 = np.sum(Ma * X * np.diag(Mb @ X.T)[None, :])
jl2 = np.sum(Ma * X * np.diag(Mb.T @ X.T)[None, :])
```

### Independent verification

Every ordered quadruplet of four distinct indices was explicitly enumerated for a 12-point catalogue with overlapping binary selections, nonuniform weights, two separation bins, and a symmetric correlation matrix. This direct enumeration was compared against the repository's c4_exact function and against the corrected calculation.

| Block | Maximum relative discrepancy of repository reference | Maximum relative discrepancy after correction |
| --- | ---: | ---: |
| AA x AA | 7.89% | 1.55e-15 |
| AA x AB | 5.45% | 6.66e-16 |
| AA x BB | 4.06% | 1.33e-15 |
| AB x AB | 11.66% | 6.66e-16 |
| AB x BB | 22.88% | 4.44e-16 |
| BB x BB | 144.18% | 1.33e-15 |

These are maxima over entries in a deliberately small catalogue. They demonstrate an algebraic error in the reference; they do not quantify the error of the production kernel or of the published large-catalogue benchmark.

Correct the reference and rerun the comparisons before using the benchmark as evidence that the production C4 integrals are unbiased.

## Components consistent with the derivation

The following structures were traced through the implementation and are consistent with the masked-weight framework, subject to consistent normalizations and a physically consistent correlation model:

1. Samples sharing one Catalog use products of effective weights at the same particle. The product vanishes outside the intersection.
2. The four single-endpoint identifications for C3 are assembled in plan_passes.
3. C4 combines the kernel-matched Wick contraction with the first-pair-swapped term to obtain both contractions.
4. Distinct particle indices are enforced within each parent catalogue.
5. C3 is divided by one galaxies-per-random factor for the shared catalogue; C2 is divided by two such factors.
6. Each block uses the RR counts associated with its two estimators. For AA x BB the required denominator is RR_AA times RR_BB, not RR_AB squared.
7. The code projects the angularly binned covariance to even multipoles.
8. One shot-noise rescaling per parent catalogue is consistent with the stated common-parent model.

These observations are a review of the implementation structure, not a substitute for end-to-end mock validation.

## Validation performed and remaining limits

### Completed

- Read the overlap design note and traced the relevant catalogue, driver, integral, adapter, post-processing, notebook, and test code.
- Ran the repository test suite on CPU: **10 passed in 95.51 seconds**.
- Ran controlled kernel-level covariance calculations demonstrating R1 and R2 through run_cov.
- Compared the C4 reference with explicit enumeration of distinct quadruplets and verified the corrected subtractions.
- Confirmed the reviewed repository remained unchanged.

The CPU test environment used Python 3.12, JAX 0.11.2, NumPy 2.5.4, SciPy 1.18.1, and pytest 9.1.1. Optional pycorr and lsstypes integrations were inspected statically; their packages were not installed for execution.

### Remaining validation

- Re-run the corrected multi-sample benchmark at the originally reported catalogue sizes and Monte Carlo depth.
- Exercise the complete pycorr/lsstypes adapter path with real selected-sample counts and jackknife inputs.
- Test sample-order invariance with a physically consistent model of redshift evolution.
- Compare the final joint covariance with common-realization mocks for the actual overlapping survey bins.

## Recommended correction order and regression checks

1. **Fix and document RR normalization.** Verify parent-normalized and selected-normalized representations yield the same covariance after explicit conversion. Include nonuniform weights and unequal sample sizes.
2. **Define the correlation model for shared objects.** Add a sample-permutation test after remapping both inputs and outputs. For a common-parent stationary mode, enforce a common xi table or reject incompatible inputs clearly.
3. **Correct the C4 reference.** Include a tiny explicit distinct-quadruplet enumeration test for all six covariance blocks.
4. **Check the physical limits.** Identical selections must reproduce the same covariance blocks; disjoint selections must have zero C2 and C3 for the AA x BB block, while C4 may remain nonzero.
5. **Validate the joint estimator on mocks.** Apply both selections to the same realization and include consistent jackknife partitions and pair-count conventions.

The production masked-weight idea is appropriate. The immediate work is to make its input normalization and shared-pair correlation model consistent, and to repair the independent numerical reference used to validate it.
