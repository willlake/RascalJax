"""Sample covariance of xi multipoles from DESI mock measurements (lsstypes files from desi-clustering).

Usage: python desi_mock_covariance.py <glob of mock h5 files> <out npz> [s_min=20] [s_max=200] [ds=4] [max_l=4]
Saves the joint covariance in (s, ell) ordering (RascalJax convention) and in (ell, s) ordering.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lsstypes  # noqa: E402

from rascaljax.binning import Binning  # noqa: E402
from rascaljax.io_lsstypes import wrap_correlation  # noqa: E402
from rascaljax.postprocess import sample_covariance  # noqa: E402

pattern, out = sys.argv[1], sys.argv[2]
s_min = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
s_max = float(sys.argv[4]) if len(sys.argv) > 4 else 200.0
ds = int(sys.argv[5]) if len(sys.argv) > 5 else 4
max_l = int(sys.argv[6]) if len(sys.argv) > 6 else 4
files = sorted(glob.glob(pattern))
print(f"{len(files)} files")
xis = []
edges = None
for i, fn in enumerate(files):
    try:
        c = lsstypes.read(fn)
    except Exception as e:  # noqa: BLE001
        print("skip", fn, e)
        continue
    c = c.select(s=(s_min, s_max))
    c = c.select(s=slice(0, c.shape[0], ds))
    c = wrap_correlation(c)
    e = np.asarray(c.get("RR").edges("s"))
    e = np.concatenate([[e[0, 0]], e[1:, 0]] + [[e[-1, 1]]]) if e.ndim == 2 else e
    if edges is None:
        edges = e
        print("s edges", edges[:3], edges[-1], "n_mu", c.shape[1])
    xi = np.asarray(c.value(), dtype=float)
    if not np.all(np.isfinite(xi)):
        print("skip (non-finite)", fn)
        continue
    xis.append(xi)
    if i % 100 == 0:
        print(i, flush=True)
xis = np.array(xis)
binning = Binning(edges, xis.shape[2], max_l)
cov = sample_covariance(xis[None], binning)
np.savez_compressed(out, cov_s_ell=cov, cov_ell_s=binning.to_pycorr_order(cov), s_edges=edges, n_mu=xis.shape[2], max_l=max_l, n_mocks=len(xis),
                    xi_samples=xis)
print("saved", out, cov.shape, "from", len(xis), "mocks")
