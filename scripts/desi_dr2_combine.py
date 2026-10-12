"""Combine the NGC and SGC RascalJax covariances into GCcomb (RR-weighted, Appendix B.2 of Rashkovetskyi+2025) and
compare with the official GCcomb RascalC covariance and the GCcomb mock covariance.

Usage: python desi_dr2_combine.py <NGC dir> <SGC dir> <GCcomb dir> [official GCcomb cov npy]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj
from rascaljax.postprocess import combine_regions, compare_covariances

d_ngc, d_sgc, d_comb = sys.argv[1:4]
official_fn = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(d_comb, exist_ok=True)
covs, weights, binning = [], [], None
for d in (d_ngc, d_sgc):
    res = np.load(os.path.join(d, "rascaljax", "Rescaled_Covariance.npz"))
    inp = np.load(os.path.join(d, "inputs.npz"))
    raw = rj.RawCovariance.load(os.path.join(d, "rascaljax", "Raw_Covariance.npz"))
    binning = raw.binning
    covs.append(res["full_theory_covariance"])  # (s, ell) ordering, all bins kept
    weights.append(inp["rr_wcounts"].ravel())
    print(d, "alpha", res["shot_noise_rescaling"], "N_eff", res["N_eff"])
cov = combine_regions(covs, weights, binning)
cov_ls = binning.to_pycorr_order(cov)
np.savez(os.path.join(d_comb, "combined_covariance.npz"), cov_s_ell=cov, cov_ell_s=cov_ls, s_edges=binning.s_edges, max_l=binning.max_l)
n_s = binning.n_s
np.set_printoptions(precision=3, linewidth=160)
if official_fn:
    state = np.load(official_fn, allow_pickle=True)[()]
    off = np.asarray(state["value"])
    print("official GCcomb covariance", off.shape)
    print("diag ratio RascalJax / official (ell=0):", np.diag(cov_ls)[:n_s] / np.diag(off)[:n_s])
    print("diag ratio RascalJax / official (ell=2):", np.diag(cov_ls)[n_s:2 * n_s] / np.diag(off)[n_s:2 * n_s])
    compare_covariances(off, cov_ls, label="official vs RascalJax GCcomb (ell 0,2,4): ")
mock_fn = os.path.join(d_comb, "mock_cov_glam.npz")
if os.path.exists(mock_fn):
    m = np.load(mock_fn)
    mock = m["cov_ell_s"]
    n_sm = (len(mock) // binning.n_ell)
    keep = np.concatenate([l * n_s + np.arange(n_sm) for l in range(binning.n_ell)])  # mocks measured to s = 180: cut the covariance
    c = cov_ls[keep[:, None], keep[None, :]]
    print(f"mock covariance from {int(m['n_mocks'])} mocks, {n_sm} s bins")
    print("diag ratio RascalJax / mocks (ell=0):", np.diag(c)[:n_sm] / np.diag(mock)[:n_sm])
    print("diag ratio RascalJax / mocks (ell=2):", np.diag(c)[n_sm:2 * n_sm] / np.diag(mock)[n_sm:2 * n_sm])
    compare_covariances(c, mock, label="RascalJax GCcomb (inverted) vs mocks: ")
    if official_fn:
        compare_covariances(off[keep[:, None], keep[None, :]], mock, label="official GCcomb (inverted) vs mocks: ")
