"""Prepare RascalJax inputs for a DESI DR2 (loa-v1 v1.1) pre-reconstruction tracer/redshift bin from the official
desipipe pycorr counts and LSS catalogs. Must run in the cosmodesi environment (pycorr, fitsio, cosmoprimo).

Usage: python desi_dr2_prepare.py <tracer> <region: NGC|SGC> <zmin> <zmax> <out_dir> [n_random_files=1]

The jackknife regions of the official GCcomb counts are per-cap sums, so (as in the DESI RascalC pipeline) the
covariance is computed per cap (NGC, SGC) and combined afterwards.

Writes ``<out_dir>/inputs.npz`` with the covariance pair counts, xi table, jackknife arrays, random positions
(Cartesian, Mpc/h), weights and jackknife labels, the number of data galaxies, and the official RascalC covariance.
"""
import os
import sys
import time

import fitsio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pycorr import KMeansSubsampler, TwoPointCorrelationFunction  # noqa: E402

from rascaljax.io_pycorr import correlation_input_from_pycorr, rebin, save_correlation_input  # noqa: E402

tracer, region, zmin, zmax, out_dir = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
n_ran_files = int(sys.argv[6]) if len(sys.argv) > 6 else 1
s_min, s_max, ds = 20, 200, 4
ds_xi, n_mu_xi = 2, 10
os.makedirs(out_dir, exist_ok=True)
base = "/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/"
cat_dir = "/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/"
zlabel = f"z{zmin:g}-{zmax:g}"
t0 = time.time()

# ----- pair counts -----
fn = base + f"2pt/xi/smu/allcounts_{tracer}_{region}_{zlabel}_default_FKP_lin_nran8_njack60_split20.npy"
print("Loading", fn, flush=True)
allcounts = TwoPointCorrelationFunction.load(fn)
assert allcounts.edges[0][1] - allcounts.edges[0][0] == 1.0, "expected 1 Mpc/h bins"
counts_cov = rebin(allcounts, slice(s_min, s_max, ds))
n_mu_orig = allcounts.shape[1]
counts_xi = rebin(allcounts, None, ds_xi, n_mu_orig // (2 * n_mu_xi))
ci = correlation_input_from_pycorr(counts_cov, counts_xi)
print(f"Covariance binning: {ci.rr.shape} (s, mu) bins; xi table {ci.xi_table.xi.shape}; jackknife {ci.jack_xi.shape}", flush=True)

# ----- data and randoms -----
from cosmoprimo.fiducial import DESI  # noqa: E402

cosmo = DESI()


def read(fn, columns):
    d = fitsio.read(fn, columns=columns)
    sel = (d["Z"] >= zmin) & (d["Z"] < zmax)
    return d[sel]


data = read(cat_dir + f"{tracer}_{region}_clustering.dat.fits", ["RA", "DEC", "Z", "WEIGHT", "WEIGHT_FKP"])
n_data = len(data)
print(f"{n_data} data galaxies in {zlabel}", flush=True)
rans = []
for i in range(n_ran_files):
    rans.append(read(cat_dir + f"{tracer}_{region}_{i}_clustering.ran.fits", ["RA", "DEC", "Z", "WEIGHT", "WEIGHT_FKP"]))
ran = np.concatenate(rans)
print(f"{len(ran)} randoms from {n_ran_files} file(s)", flush=True)

# jackknife regions as in desipipe/xirunpc: angular K-means on the data positions, 60 regions, nside 512, seed 42
for key in [k for k in os.environ if k.startswith("OMP_")]:
    os.environ.pop(key)
os.environ["OMP_NUM_THREADS"] = "1"
d_data = cosmo.comoving_radial_distance(data["Z"])
subsampler = KMeansSubsampler("angular", positions=[data["RA"], data["DEC"], d_data], position_type="rdd", nsamples=60, nside=512, random_state=42, dtype=np.float64)
d_ran = cosmo.comoving_radial_distance(ran["Z"])
labels = subsampler.label([ran["RA"], ran["DEC"], d_ran], position_type="rdd")
data_labels = subsampler.label([data["RA"], data["DEC"], d_data], position_type="rdd")
print("Jackknife labels:", np.unique(labels)[:5], "...", len(np.unique(labels)), "regions", flush=True)
# consistency proxy: fraction of random weight per region vs the file's jackknife RR fraction at the largest separations
w_ran = ran["WEIGHT"] * ran["WEIGHT_FKP"]
frac = np.array([w_ran[labels == A].sum() for A in range(60)]) / w_ran.sum()
jw = ci.jack_weights.reshape(60, ci.rr.shape[0], ci.rr.shape[1])[:, 0, :].mean(axis=1)  # weights at the smallest covariance separation (most local)
print("corr(region random weight fraction, file jackknife weight at s=20):", np.corrcoef(frac, jw)[0, 1])
print("ratio file jackknife weight / 2 x weight fraction (first 10 regions):", (jw / (2 * frac))[:10])

ra, dec = np.radians(ran["RA"]), np.radians(ran["DEC"])
pos = np.stack([d_ran * np.cos(dec) * np.cos(ra), d_ran * np.cos(dec) * np.sin(ra), d_ran * np.sin(dec)], axis=1)

# ----- official RascalC covariance (desilike ObservableCovariance pickle) -----
official = {}
try:
    cov_fn = base + f"cov_2pt/rascalc/v1.1/covariance_correlation_{tracer}_{region}_{zlabel}_default_FKP_lin.npy"
    state = np.load(cov_fn, allow_pickle=True)[()]
    print("Official covariance keys:", list(state.keys()) if isinstance(state, dict) else type(state))
    official["rascalc_cov"] = np.asarray(state["value"])
    obs = state.get("observables", state.get("observable", None))
    if obs is not None:
        o = obs[0] if isinstance(obs, (list, tuple)) else obs
        if isinstance(o, dict):
            official["rascalc_x"] = np.concatenate([np.asarray(x) for x in o["x"]])
            official["rascalc_projs"] = np.array([str(p) for p in o["projs"]], dtype="U")
    print("Official RascalC covariance:", official["rascalc_cov"].shape, flush=True)
except Exception as e:  # noqa: BLE001
    print("Could not load the official RascalC covariance:", e)

from rascaljax.io_pycorr import _wrap  # noqa: E402

rr_wcounts = np.asarray(_wrap(counts_cov).R1R2.wcounts, dtype=float)  # unnormalized, for the region combination weights
save_correlation_input(ci, os.path.join(out_dir, "inputs.npz"), positions=pos, weights=w_ran, labels=labels, n_data=n_data, rr_wcounts=rr_wcounts,
                       data_labels=data_labels, data_weights=data["WEIGHT"] * data["WEIGHT_FKP"], **official)
print(f"Saved {os.path.join(out_dir, 'inputs.npz')} in {time.time() - t0:.0f} s")
