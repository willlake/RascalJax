"""Joint covariance of two overlapping redshift bins of one DESI tracer, validated against mocks.

Inputs (prepared by ``desi_zbins_prepare_mocks.py`` in the cosmodesi environment) in ``<dir>``:

* ``randoms.npz``: ``positions`` (N, 3) Cartesian Mpc/h, ``weights``, ``z`` of the randoms covering the union of the
  two bins, ``n_data`` (data galaxies in the union), ``zbins`` ((2, 2) array of the bin edges);
* ``corr_AA.npz``, ``corr_AB.npz``, ``corr_BB.npz``: :func:`rascaljax.io_pycorr.save_correlation_input` files with the
  pair counts (sample-normalized, as pycorr gives them on the selected randoms) and xi tables of the mock mean;
* ``mock_xi.npz``: ``xi_samples`` (n_corr, n_mock, n_s, n_mu) of the mocks in the covariance binning with ``correlations``
  naming them (AA, BB for a BAO data vector; AB optional).

Usage: python desi_zbins_run.py <dir> [n_loops=32] [loops_per_sample=2] [max_randoms=None] [--skip-run]
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj
from rascaljax.io_pycorr import load_correlation_input
from rascaljax.postprocess import CovarianceTerms, compare_covariances, fit_shot_noise_rescaling, sample_covariance

args = [a for a in sys.argv[1:] if not a.startswith("--")]
d = args[0]
n_loops = int(args[1]) if len(args) > 1 else 32
loops_per_sample = int(args[2]) if len(args) > 2 else 2
max_randoms = int(args[3]) if len(args) > 3 and args[3] != "None" else None
skip_run = "--skip-run" in sys.argv
out_dir = os.path.join(d, "rascaljax")
os.makedirs(out_dir, exist_ok=True)
max_l = 4
np.set_printoptions(precision=3, linewidth=160)

# ----- inputs -----
r = np.load(os.path.join(d, "randoms.npz"))
pos, w, z, n_data, zbins = r["positions"], r["weights"], r["z"], float(r["n_data"]), r["zbins"]
if max_randoms is not None and len(pos) > max_randoms:
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pos), max_randoms, replace=False, p=w / w.sum())
    pos, w, z = pos[sel], np.ones(max_randoms), z[sel]
names = ["AA", "AB", "BB"]
inputs = {}
for key, name in zip([(0, 0), (0, 1), (1, 1)], names):
    ci, _ = load_correlation_input(os.path.join(d, f"corr_{name}.npz"))
    inputs[key] = ci
print(f"{len(pos)} randoms in z{zbins[0, 0]:g}-{zbins[1, 1]:g}, n_data={n_data:.0f}, covariance bins {inputs[(0, 0)].rr.shape}, "
      f"xi tables {inputs[(0, 0)].xi_table.xi.shape}, rr_normalization={inputs[(0, 0)].rr_normalization}")
cat = rj.Catalog(pos, w, n_data=n_data, name="LRG")
sA = rj.Sample.from_redshift_range(cat, z, zbins[0, 0], zbins[0, 1], "zA")
sB = rj.Sample.from_redshift_range(cat, z, zbins[1, 0], zbins[1, 1], "zB")
print(f"weight fractions: A {sA.effective_weights.sum():.4f}, B {sB.effective_weights.sum():.4f}, overlap {(sA.effective_weights * sB.membership).sum():.4f}")

# ----- run -----
if not skip_run:
    t0 = time.time()
    raw = rj.run_cov([sA, sB], inputs, mode="legendre", max_l=max_l, N2=5, N3=10, N4=20, n_loops=n_loops, loops_per_sample=loops_per_sample,
                     boxsize=None, xi_cut=250.0, refine_iterations=10, refine_draws=20, quads_per_batch=5_000_000, seed=42, out_dir=out_dir)
    print(f"run_cov done in {time.time() - t0:.0f} s")
else:
    raw = rj.RawCovariance.load(os.path.join(out_dir, "Raw_Covariance.npz"))
b = raw.binning
n_s, n_ell, n_out = b.n_s, b.n_ell, b.n_out
print("RR check (sampled / input, median over bins):", raw.info.get("rr_check"))

# ----- mock covariance over the measured correlations, embedded in the (AA, AB, BB) block structure of the run -----
m = np.load(os.path.join(d, "mock_xi.npz"))
xis = m["xi_samples"]  # (n_avail, n_mock, n_s_m, n_mu)
mock_names = [str(x) for x in m["correlations"]]
avail = [names.index(nm) for nm in mock_names]  # raw block index of each measured correlation
n_mock = xis.shape[1]
if not np.allclose(m["s_edges"], b.s_edges) or xis.shape[3] != b.n_mu:
    raise ValueError("mock binning differs from the covariance binning")
if n_mock < 2:
    res1 = rj.post_process(raw, alphas=1.0, out_file=os.path.join(out_dir, "Rescaled_Covariance_alpha1.npz"))
    print(f"only {n_mock} mock measured: saved the alpha = 1 covariance, no comparison (rerun with --skip-run once more mocks exist)")
    sys.exit(0)
mock_avail = sample_covariance(xis, b)  # (n_avail n_out, n_avail n_out) in (corr, s, ell)
mock = np.full((3 * n_out, 3 * n_out), np.nan)
for ka, ia in enumerate(avail):
    for kb, ib in enumerate(avail):
        mock[ia * n_out:(ia + 1) * n_out, ib * n_out:(ib + 1) * n_out] = mock_avail[ka * n_out:(ka + 1) * n_out, kb * n_out:(kb + 1) * n_out]
idx_avail = np.concatenate([ia * n_out + np.arange(n_out) for ia in avail])
print(f"mock covariance from {n_mock} mocks for {mock_names}: {mock_avail.shape}")

# ----- post-processing: alpha = 1, alpha fitted on the AA x AA block only, alpha fitted on all measured blocks -----
res1 = rj.post_process(raw, alphas=1.0, out_file=os.path.join(out_dir, "Rescaled_Covariance_alpha1.npz"))
terms = CovarianceTerms(raw)
idx_aa = np.arange(n_out)
alpha_aa = fit_shot_noise_rescaling(terms.subset(idx_aa), mock[idx_aa[:, None], idx_aa[None, :]], "joint")
res_aa = rj.post_process(raw, alphas=alpha_aa, out_file=os.path.join(out_dir, "Rescaled_Covariance_alphaAA.npz"))
alpha_j = fit_shot_noise_rescaling(terms.subset(idx_avail), mock[idx_avail[:, None], idx_avail[None, :]], "joint")
res_j = rj.post_process(raw, alphas=alpha_j, out_file=os.path.join(out_dir, "Rescaled_Covariance_mockfit.npz"))
print(f"shot-noise rescaling: AA-block fit {alpha_aa}, fit on {mock_names} {alpha_j}; N_eff (alpha=1) {res1['N_eff']:.0f}")


def block(c, i, j):
    return c[i * n_out:(i + 1) * n_out, j * n_out:(j + 1) * n_out]


def diag_ell(blk, ell_idx):
    return np.diag(blk)[ell_idx::n_ell]


def report(cov, label):
    print(f"===== {label}")
    for (i, j) in [(0, 0), (2, 2), (1, 1), (0, 2), (0, 1), (1, 2)]:
        if i not in avail or j not in avail:
            continue
        cm, cr = block(mock, i, j), block(cov, i, j)
        for ell_idx, ell in enumerate((0, 2)):
            dm, dr = diag_ell(cm, ell_idx), diag_ell(cr, ell_idx)
            if i == j:
                print(f"{names[i]} x {names[j]} ell={ell}: diag ratio RascalJax / mock = {dr / dm}")
            else:
                # off-diagonal blocks: the correlation coefficient between the two correlations at the same (s, ell)
                di, dj = diag_ell(block(mock, i, i), ell_idx), diag_ell(block(mock, j, j), ell_idx)
                ri, rj_ = diag_ell(block(cov, i, i), ell_idx), diag_ell(block(cov, j, j), ell_idx)
                print(f"{names[i]} x {names[j]} ell={ell}: corr. coeff. mock     = {dm / np.sqrt(di * dj)}")
                print(f"{names[i]} x {names[j]} ell={ell}: corr. coeff. RascalJax = {dr / np.sqrt(ri * rj_)}")
    # inversion-based metrics on the (AA, BB) joint block with ell = 0, 2 (needs n_mock well above the dimension)
    # inversion-based metrics on the BAO range (48-152 Mpc/h), ell = 0, 2, for the (AA, BB) data vector and each auto alone
    i_bao = np.flatnonzero((b.s_edges[:-1] >= 48) & (b.s_edges[1:] <= 152))
    keep_a = (i_bao[:, None] * n_ell + np.arange(2)[None, :]).ravel()
    keep = np.concatenate([c * n_out + keep_a for c in (0, 2) if c in avail])
    if n_mock > 2 * len(keep):
        compare_covariances(mock[keep[:, None], keep[None, :]], cov[keep[:, None], keep[None, :]], label="mock vs RascalJax, (AA, BB) BAO range ell 0,2: ")
    if n_mock > 2 * len(keep_a):
        for c, lab in ((0, "AA"), (2, "BB")):
            if c in avail:
                compare_covariances(block(mock, c, c)[keep_a[:, None], keep_a[None, :]], block(cov, c, c)[keep_a[:, None], keep_a[None, :]], label=f"mock vs RascalJax, {lab} BAO range ell 0,2: ")
    # chi2 of the mock xi around their mean with the RascalJax precision: each auto alone, (AA, BB) jointly, all measured
    v = {ia: xis[ka].reshape(n_mock, -1) @ b.projection_matrix() for ka, ia in enumerate(avail)}  # (n_mock, n_out) per block
    tests = [([c], names[c]) for c in avail] + ([([0, 2], "AA+BB joint")] if 0 in avail and 2 in avail else [])
    if len(avail) == 3:
        tests.append(([0, 1, 2], "AA+AB+BB joint"))
    for sel, lab in tests:
        dv = np.concatenate([v[c] - v[c].mean(axis=0) for c in sel], axis=1)
        ii = np.concatenate([c * n_out + np.arange(n_out) for c in sel])
        prec = np.linalg.inv(cov[ii[:, None], ii[None, :]])
        chi2 = np.einsum("ia,ab,ib->i", dv, prec, dv)
        print(f"mean reduced chi2 of the mocks with the RascalJax precision, {lab}: {chi2.mean() / len(ii) * n_mock / (n_mock - 1):.3f} (expect 1)")


report(res1["full_theory_covariance"], "alpha = 1")
report(res_aa["full_theory_covariance"], f"alpha fitted on the AA x AA block ({alpha_aa[0]:.4f})")
np.savez(os.path.join(out_dir, "comparison.npz"), mock=mock, mock_correlations=np.array(mock_names), cov_alpha1=res1["full_theory_covariance"],
         cov_alphaAA=res_aa["full_theory_covariance"], cov_mockfit=res_j["full_theory_covariance"], alpha_aa=alpha_aa, alpha_joint=alpha_j,
         s_edges=b.s_edges, n_mu=b.n_mu, max_l=max_l, n_mock=n_mock)
print("saved", os.path.join(out_dir, "comparison.npz"))
