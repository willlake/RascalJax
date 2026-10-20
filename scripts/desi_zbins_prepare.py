"""Build the RascalJax inputs for two overlapping redshift bins from the mock measurements of
``desi_zbins_measure_mocks.py`` (pycorr files; must run in the cosmodesi environment).

Usage: python desi_zbins_prepare.py <dir> <region: NGC|SGC> [--xi-dir DIR] [--ref-mock 150] [--s-min 40] [--s-max 160]
                                    [--ds 4] [--ds-xi 2] [--n-mu-xi 10] [--max-mocks N] [--sample-correlations AA,BB]

Two sets of measurements are used: the many mocks for the sample covariance (``<dir>/mocks/mock*/xi_{AA,AB,BB}_<region>.npy``,
measured to the covariance ``s_max``) and, optionally, a few mocks measured to larger separations for the inputs of the
covariance integrals (``--xi-dir``, same layout; defaults to ``<dir>``). Writes into ``<dir>/<region>/``:

* ``corr_AA.npz``, ``corr_AB.npz``, ``corr_BB.npz``: the RR counts of the reference mock in the covariance binning
  (sample-normalized, as measured on the randoms selected for each bin) and xi tables averaged over the ``xi-dir`` mocks;
  for the cross-correlation the first separation bin holds the self-pairs of the overlap objects and is replaced by the
  mean of the two auto-correlations;
* ``randoms.npz``: copy of the reference mock's expanded randoms (union of the two bins, from ``xi-dir``);
* ``mock_xi.npz``: ``xi_samples`` (n_corr, n_mock, n_s, n_mu) of the ``<dir>`` mocks in the covariance binning for the
  ``--sample-correlations`` (default AA, BB: a BAO fit does not use the cross-correlation), with their names.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pycorr import TwoPointCorrelationFunction  # noqa: E402

from rascaljax.io_pycorr import _wrap, correlation_input_from_pycorr, rebin, save_correlation_input, xi_table_from_pycorr  # noqa: E402
from rascaljax.xi_table import XiTable  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("dir")
p.add_argument("region", choices=["NGC", "SGC"])
p.add_argument("--xi-dir", default=None, help="directory of the (few) mocks measured to large separations for the xi tables and RR counts")
p.add_argument("--ref-mock", type=int, default=150)
p.add_argument("--s-min", type=int, default=40)
p.add_argument("--s-max", type=int, default=160)
p.add_argument("--ds", type=int, default=4)
p.add_argument("--ds-xi", type=int, default=2)
p.add_argument("--n-mu-xi", type=int, default=10)
p.add_argument("--max-mocks", type=int, default=None)
p.add_argument("--sample-correlations", default="AA,BB", help="correlations entering the mock sample covariance")
a = p.parse_args()
names = ["AA", "AB", "BB"]  # RascalJax order of the correlations of samples (0, 1)
sample_names = [n for n in names if n in a.sample_correlations.split(",")]
out = os.path.join(a.dir, a.region)
os.makedirs(out, exist_ok=True)
xi_dir = a.xi_dir or a.dir


def complete_mocks(base, needed):
    dirs = sorted(glob.glob(os.path.join(base, "mocks", "mock*")), key=lambda d: int(os.path.basename(d)[4:]))
    return [d for d in dirs if all(os.path.exists(os.path.join(d, f"xi_{n}_{a.region}.npy")) for n in needed)]


def load(d, n):
    return TwoPointCorrelationFunction.load(os.path.join(d, f"xi_{n}_{a.region}.npy"))


mock_dirs = complete_mocks(a.dir, sample_names)
if a.max_mocks:
    mock_dirs = mock_dirs[: a.max_mocks]
imocks = [int(os.path.basename(d)[4:]) for d in mock_dirs]
xi_dirs = complete_mocks(xi_dir, names)
xi_imocks = [int(os.path.basename(d)[4:]) for d in xi_dirs]
print(f"{len(imocks)} mocks with {sample_names} for {a.region} (sample covariance); {len(xi_imocks)} with all three correlations in {xi_dir} (xi tables, RR)")
if a.ref_mock not in xi_imocks:
    raise SystemExit(f"reference mock {a.ref_mock} has no complete measurement in {xi_dir}")

# ----- sample covariance: xi of every mock in the covariance binning (s_min-s_max by ds, mu wrapped to [0, 1]) -----
cov_slice = slice(a.s_min, a.s_max, a.ds)  # 1 Mpc/h bins: indices are separations
xis = {n: [] for n in sample_names}
edges = None
for k, d in enumerate(mock_dirs):
    for n in sample_names:
        est = _wrap(rebin(load(d, n), cov_slice))
        if edges is None:
            edges = np.asarray(est.edges[0], dtype=float)
            print("covariance s edges", edges[:3], "...", edges[-1], "n_mu", est.shape[1])
        xi = np.asarray(est.corr, dtype=float)
        if not np.all(np.isfinite(xi)):
            raise RuntimeError(f"non-finite xi in {d} {n}")
        xis[n].append(xi)
    if k % 50 == 0:
        print(k, flush=True)
xi_samples = np.array([xis[n] for n in sample_names])  # (n_corr, n_mock, n_s, n_mu)
np.savez_compressed(os.path.join(out, "mock_xi.npz"), xi_samples=xi_samples, s_edges=edges, n_mu=xi_samples.shape[3], imocks=np.array(imocks),
                    correlations=np.array(sample_names))
print("saved mock_xi.npz", xi_samples.shape)

# ----- inputs: RR of the reference mock in the covariance binning, xi tables averaged over the xi-dir mocks -----
ref = {n: load(xi_dirs[xi_imocks.index(a.ref_mock)], n) for n in names}
n_mu_full = ref["AA"].shape[1]
mu_step = n_mu_full // (2 * a.n_mu_xi)
print(f"xi tables from {len(xi_imocks)} mocks measured to s = {ref['AA'].edges[0][-1]:g} Mpc/h, {a.ds_xi} Mpc/h x {a.n_mu_xi} mu bins")
tables, inputs = {}, {}
for n in names:
    tabs = [xi_table_from_pycorr(rebin(load(d, n), None, a.ds_xi, mu_step)) for d in xi_dirs]
    t0 = tabs[0]
    xi_mean = np.mean([t.xi for t in tabs], axis=0)
    tables[n] = XiTable(np.mean([t.s_vals for t in tabs], axis=0), np.mean([t.mu_vals for t in tabs], axis=0), xi_mean, t0.s_edges)
    ci = correlation_input_from_pycorr(rebin(ref[n], cov_slice), xi=rebin(ref[n], None, a.ds_xi, mu_step))
    ci.xi_table = tables[n]
    inputs[n] = ci
    print(f"{n}: RR {ci.rr.shape} (sum {ci.rr.sum():.4e}), xi table {ci.xi_table.xi.shape}, rr_normalization={ci.rr_normalization}")
# cross-correlation: the first s bin contains the zero-separation self-pairs of the overlap objects
t = tables["AB"]
xi_fixed = t.xi.copy()
xi_fixed[0] = 0.5 * (tables["AA"].xi[0] + tables["BB"].xi[0])
print(f"AB xi table first bin: measured {t.xi[0, :3]} -> replaced by the auto mean {xi_fixed[0, :3]}")
inputs["AB"].xi_table = XiTable(t.s_vals, t.mu_vals, xi_fixed, t.s_edges)
for n in names:
    save_correlation_input(inputs[n], os.path.join(out, f"corr_{n}.npz"), imocks=np.array(xi_imocks), ref_mock=a.ref_mock)
ran_fn = os.path.join(xi_dir, f"randoms_{a.region}_mock{a.ref_mock}.npz")
shutil.copyfile(ran_fn, os.path.join(out, "randoms.npz"))
r = np.load(ran_fn)
print(f"randoms.npz: {len(r['positions'])} randoms, n_data {int(r['n_data'])} (A {int(r['n_data_A'])}, B {int(r['n_data_B'])}), zbins {r['zbins'].tolist()}")
print("wrote", out)
