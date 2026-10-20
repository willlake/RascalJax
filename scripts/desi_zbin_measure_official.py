"""2PCF of one redshift bin of DESI mocks with the official desi-clustering settings: DD and DR with all ``--nran`` random
files, RR as the sum of same-split pair counts of the concatenated randoms (``--split-randoms`` = randoms/data ratio per
split, as in desi-clustering's ``compute_particle2_correlation``), s in 1 Mpc/h bins to ``--s-max``, 200 mu bins in
[-1, 1], midpoint line of sight, default-FKP weights. Landy-Szalay estimator saved as pycorr file.

Usage: python desi_zbin_measure_official.py <out_dir> <region> <z_min> <z_max> imock... [--nran 10] [--split-randoms 10]
           [--s-max 180] [--label BB] [--engine corrfunc|cucount] [--nthreads 28] [--skip-existing] [--save-randoms]
Writes <out_dir>/mocks/mock<i>/xi_<label>_<region>.npy (claims <out_dir>/mocks/mock<i>/.claim_<label>_<region>).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/global/homes/w/willlake/Packages/desi-clustering")
p = argparse.ArgumentParser()
p.add_argument("out_dir"); p.add_argument("region"); p.add_argument("z_min", type=float); p.add_argument("z_max", type=float)
p.add_argument("imocks", type=int, nargs="+"); p.add_argument("--nran", type=int, default=10)
p.add_argument("--split-randoms", type=float, default=10.0, help="randoms / data ratio per RR split (desi-clustering convention)")
p.add_argument("--s-max", type=float, default=180.0); p.add_argument("--label", default="BB")
p.add_argument("--engine", default="corrfunc"); p.add_argument("--nthreads", type=int, default=28)
p.add_argument("--skip-existing", action="store_true"); p.add_argument("--save-randoms", action="store_true")
p.add_argument("--tracer", default="LRG"); p.add_argument("--version", default="glam-uchuu-v2-altmtl")
a = p.parse_args()
if a.engine == "cucount":
    import jax
    jax.config.update("jax_enable_x64", True)
from clustering_statistics import tools  # noqa: E402
from pycorr import TwoPointCorrelationFunction, TwoPointCounter, setup_logging  # noqa: E402

setup_logging("warning")
zr = (a.z_min, a.z_max)
opts = dict(version=a.version, tracer=a.tracer, weight="default-FKP", FKP_P0=1e4, nran=a.nran)
expand = {"parent_randoms_fn": tools.get_catalog_fn(kind="parent_randoms", version="data-dr2-v2", tracer=a.tracer, nran=a.nran),
          "from_data": ["Z", "WEIGHT_SYS", "FRAC_TLOBS_TILES"]}
edges = (np.linspace(0.0, a.s_max, int(a.s_max) + 1), np.linspace(-1.0, 1.0, 201))
kw = dict(position_type="pos", los="midpoint", engine=a.engine)
if a.engine == "corrfunc":
    kw["nthreads"] = a.nthreads
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "desi_zbins_measure_mocks.py")).read()
i0 = src.index("def arrays(cat):")
exec(src[i0:src.index("\n\n", i0)])  # noqa: S102  -> arrays(cat): (positions, weights) of a prepared catalog

for imock in a.imocks:
    mdir = os.path.join(a.out_dir, "mocks", f"mock{imock}")
    os.makedirs(mdir, exist_ok=True)
    fn = os.path.join(mdir, f"xi_{a.label}_{a.region}.npy")
    if a.skip_existing and os.path.exists(fn):
        print(f"mock {imock}: exists", flush=True)
        continue
    if a.skip_existing:
        try:
            os.mkdir(os.path.join(mdir, f".claim_{a.label}_{a.region}"))
        except FileExistsError:
            print(f"mock {imock}: claimed", flush=True)
            continue
    t0 = time.time()
    data = tools.prepare_catalog(tools.read_catalog(kind="data", region=a.region, zrange=zr, imock=imock, **opts), kind="data", zrange=zr, region=a.region, **opts)
    randoms = tools.read_catalog(kind="randoms", region=a.region, zrange=zr, imock=imock, expand=expand, concatenate=False, **opts)
    randoms = [tools.prepare_catalog(r, kind="randoms", zrange=zr, region=a.region, **opts) for r in randoms]
    dpos, dw = arrays(data)
    rs = [arrays(r) for r in randoms]
    rpos, rw = np.concatenate([r[0] for r in rs]), np.concatenate([r[1] for r in rs])
    del randoms, rs
    n_data, n_ran = len(dw), len(rw)
    print(f"mock {imock} {a.region} z{zr[0]:g}-{zr[1]:g}: {n_data} data, {n_ran} randoms ({a.nran} files); read in {time.time() - t0:.0f} s", flush=True)
    if a.save_randoms:
        np.savez_compressed(os.path.join(a.out_dir, f"randoms_{a.region}_mock{imock}_z{zr[0]:g}-{zr[1]:g}.npz"), positions=rpos, weights=rw, n_data=n_data, zrange=np.array(zr))
    # RR: same-split pairs of the concatenated randoms, splits of split_randoms x n_data (desi-clustering: random uniform split, seed 42)
    nsplits = max(int(n_ran / (a.split_randoms * n_data)), 1)
    x = np.random.default_rng(42).uniform(0.0, 1.0, n_ran)
    t1 = time.time()
    rr = sum(TwoPointCounter("smu", edges, positions1=rpos[m], weights1=rw[m], **kw) for m in ((x >= i / nsplits) & (x < (i + 1) / nsplits) for i in range(nsplits)))
    t_rr = time.time() - t1
    est = TwoPointCorrelationFunction("smu", edges, data_positions1=dpos, data_weights1=dw, randoms_positions1=rpos, randoms_weights1=rw, R1R2=rr, estimator="landyszalay", **kw)
    est.save(fn)
    print(f"  {a.label}: RR in {nsplits} splits of ~{n_ran // nsplits} randoms in {t_rr:.0f} s; DD+DR in {time.time() - t1 - t_rr:.0f} s -> {fn}", flush=True)
    print(f"mock {imock} done in {time.time() - t0:.0f} s", flush=True)
