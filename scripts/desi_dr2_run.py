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

d = sys.argv[1]
n_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 32
loops_per_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 2
max_randoms = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "None" else None
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
    if mock.shape == cov_ls.shape:
        hartlap = (n_mock - len(mock) - 2) / (n_mock - 1)
        print("diagonal ratio RascalJax / mocks (ell=0):", np.diag(cov_ls)[:n_s] / np.diag(mock)[:n_s])
        print("diagonal ratio RascalJax / mocks (ell=2):", np.diag(cov_ls)[n_s:2 * n_s] / np.diag(mock)[n_s:2 * n_s])
        compare_covariances(cov_ls, mock, label="RascalJax (inverted) vs mocks: ")
        if "rascalc_cov" in extra and extra["rascalc_cov"].shape == mock.shape:
            compare_covariances(extra["rascalc_cov"], mock, label="official RascalC (inverted) vs mocks: ")
        keep = np.concatenate([np.arange(n_s), n_s + np.arange(n_s)])
        compare_covariances(cov_ls[keep[:, None], keep[None, :]], mock[keep[:, None], keep[None, :]], label="RascalJax vs mocks (ell 0,2): ")
        # chi2 of mock xi around the mock mean with the RascalJax precision
        xis = m["xi_samples"]  # (n_mock, n_s, n_mu) in (s, mu)
        P = b.projection_matrix()
        v = xis.reshape(n_mock, -1) @ P  # (n_mock, n_out) in (s, ell) order
        dv = v - v.mean(axis=0)
        prec = np.linalg.inv(cov)
        chi2 = np.einsum("ia,ab,ib->i", dv, prec, dv)
        print(f"mean reduced chi2 of mocks with the RascalJax covariance: {chi2.mean() / cov.shape[0] * n_mock / (n_mock - 1):.3f} (expect 1)")
        np.savez(os.path.join(out_dir, "comparison.npz"), cov_s_ell=cov, cov_ell_s=cov_ls, mock=mock, official=extra.get("rascalc_cov"))
