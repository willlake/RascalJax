"""Run RascalJax on the inputs prepared by desi_dr2_prepare.py and compare with the official RascalC covariance and
the GLAM mock sample covariance.

Usage: python desi_dr2_run.py <dir with inputs.npz> [n_loops=32] [loops_per_sample=2] [max_randoms=None] [--skip-run]
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj
from rascaljax.io_pycorr import load_correlation_input
from rascaljax.postprocess import compare_covariances

d = args[0]
n_loops = int(args[1]) if len(args) > 1 else 32
loops_per_sample = int(args[2]) if len(args) > 2 else 2
args = [a for a in sys.argv[1:] if not a.startswith("--")]
max_randoms = int(args[3]) if len(args) > 3 and args[3] != "None" else None
skip_run = "--skip-run" in sys.argv
out_dir = os.path.join(d, "rascaljax")
max_l = 4

ci, extra = load_correlation_input(os.path.join(d, "inputs.npz"))
pos, w, labels, n_data = extra["positions"], extra["weights"], extra["labels"], float(extra["n_data"])
if max_randoms is not None and len(pos) > max_randoms:
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pos), max_randoms, replace=False, p=w / w.sum())
    pos, w, labels = pos[sel], np.ones(max_randoms), labels[sel]
    # note: weighted subsampling then unit weights keeps the weighted density; counts remain normalized
print(f"{len(pos)} randoms, n_data={n_data:.0f}, covariance bins {ci.rr.shape}, xi table {ci.xi_table.xi.shape}, jackknife {ci.jack_xi.shape}")
cat = rj.Catalog(pos, w, n_data=n_data, jackknife=labels, name="LRG")

if not skip_run:
    raw = rj.run_cov([rj.Sample(cat)], {(0, 0): ci}, mode="legendre", max_l=max_l, N2=5, N3=10, N4=20, n_loops=n_loops, loops_per_sample=loops_per_sample,
                     boxsize=None, xi_cut=250.0, refine_iterations=10, refine_draws=20, quads_per_batch=5_000_000, seed=42, out_dir=out_dir)
else:
    raw = rj.RawCovariance.load(os.path.join(out_dir, "Raw_Covariance.npz"))

res = rj.post_process(raw, disconnected=False, out_file=os.path.join(out_dir, "Rescaled_Covariance.npz"))
cov = res["full_theory_covariance"]
b = raw.binning
cov_ls = b.to_pycorr_order(cov)  # (ell, s) ordering
n_s, n_ell = b.n_s, b.n_ell
np.set_printoptions(precision=3, linewidth=160)
print("shot-noise rescaling:", res["shot_noise_rescaling"], "N_eff:", res["N_eff"])

if "rascalc_cov" in extra:
    off = extra["rascalc_cov"]
    print("official RascalC covariance", off.shape, "projs", extra.get("rascalc_projs"))
    if off.shape == cov_ls.shape:
        print("shot-noise rescaling in the official file (attrs):", {k: v for k, v in extra.items() if k.startswith("rascalc_attr")})
        print("diagonal ratio RascalJax / official (ell=0):", np.diag(cov_ls)[:n_s] / np.diag(off)[:n_s])
        print("diagonal ratio RascalJax / official (ell=2):", np.diag(cov_ls)[n_s:2 * n_s] / np.diag(off)[n_s:2 * n_s])
        print("diagonal ratio RascalJax / official (ell=4):", np.diag(cov_ls)[2 * n_s:] / np.diag(off)[2 * n_s:])
        compare_covariances(off, cov_ls, label="official vs RascalJax (ell 0,2,4; all s): ")
        keep = np.concatenate([np.arange(n_s), n_s + np.arange(n_s)])  # ell = 0, 2 only
        compare_covariances(off[keep[:, None], keep[None, :]], cov_ls[keep[:, None], keep[None, :]], label="official vs RascalJax (ell 0,2): ")
mock_fn = os.path.join(d, "mock_cov_glam.npz")
if os.path.exists(mock_fn):
    m = np.load(mock_fn)
    mock = m["cov_ell_s"]
    n_mock = int(m["n_mocks"])
    print(f"mock covariance from {n_mock} GLAM mocks", mock.shape)
    n_s_mock = (len(m["s_edges"]) - 1)
    if n_s_mock < n_s and np.allclose(m["s_edges"], b.s_edges[: n_s_mock + 1]):
        # mocks measured to a smaller s_max: restrict the RascalJax (and official) covariance to the common bins
        keep_m = np.concatenate([l * n_s + np.arange(n_s_mock) for l in range(n_ell)])
        cov_ls = cov_ls[keep_m[:, None], keep_m[None, :]]
        cov = cov[(np.arange(n_s_mock)[:, None] * n_ell + np.arange(n_ell)[None, :]).ravel()[:, None], (np.arange(n_s_mock)[:, None] * n_ell + np.arange(n_ell)[None, :]).ravel()[None, :]]
        if "rascalc_cov" in extra:
            extra["rascalc_cov"] = extra["rascalc_cov"][keep_m[:, None], keep_m[None, :]]
        n_s = n_s_mock
        b = rj.Binning(m["s_edges"], b.n_mu, b.max_l)
        print(f"restricted to the mocks' {n_s} separation bins (s < {m['s_edges'][-1]:g})")
    n_sm = len(mock) // n_ell  # mocks may cover fewer separation bins (measured to s = 180)
    keep = np.concatenate([l * n_s + np.arange(n_sm) for l in range(n_ell)])
    c = cov_ls[keep[:, None], keep[None, :]]
    print("diagonal ratio RascalJax / mocks (ell=0):", np.diag(c)[:n_sm] / np.diag(mock)[:n_sm])
    print("diagonal ratio RascalJax / mocks (ell=2):", np.diag(c)[n_sm:2 * n_sm] / np.diag(mock)[n_sm:2 * n_sm])
    compare_covariances(c, mock, label="RascalJax (inverted) vs mocks (ell 0,2,4): ")
    if "rascalc_cov" in extra:
        compare_covariances(extra["rascalc_cov"][keep[:, None], keep[None, :]], mock, label="official RascalC (inverted) vs mocks (ell 0,2,4): ")
    keep02 = np.concatenate([np.arange(n_sm), n_sm + np.arange(n_sm)])
    compare_covariances(c[keep02[:, None], keep02[None, :]], mock[keep02[:, None], keep02[None, :]], label="RascalJax (inverted) vs mocks (ell 0,2): ")
    if "rascalc_cov" in extra:
        o = extra["rascalc_cov"][keep[:, None], keep[None, :]]
        compare_covariances(o[keep02[:, None], keep02[None, :]], mock[keep02[:, None], keep02[None, :]], label="official RascalC (inverted) vs mocks (ell 0,2): ")
    # chi2 of mock xi around the mock mean with the RascalJax precision (ell 0, 2 and 4, s up to the mocks' range)
    xis = m["xi_samples"]  # (n_mock, n_sm, n_mu) in (s, mu)
    bm = rj.Binning(m["s_edges"], int(m["n_mu"]), max_l)
    v = xis.reshape(n_mock, -1) @ bm.projection_matrix()  # (n_mock, n_out) in (s, ell) order
    dv = v - v.mean(axis=0)
    keep_se = np.concatenate([i * n_ell + np.arange(n_ell) for i in range(n_sm)])
    prec = np.linalg.inv(cov[keep_se[:, None], keep_se[None, :]])
    chi2 = np.einsum("ia,ab,ib->i", dv, prec, dv)
    print(f"mean reduced chi2 of mocks with the RascalJax covariance: {chi2.mean() / len(keep_se) * n_mock / (n_mock - 1):.3f} (expect 1)")
    np.savez(os.path.join(out_dir, "comparison.npz"), cov_s_ell=cov, cov_ell_s=cov_ls, mock=mock, official=extra.get("rascalc_cov"))
