"""Rebuild Landy-Szalay estimates with DR (and R1D2) computed from the SAME random files as the stored RR.

The stored pycorr estimators keep their DD and RR counts; only the data-random terms are recomputed, with the first
``nran`` random files (those used for RR), so that the random-catalog edge noise cancels in the estimator.

Usage: python desi_zbins_remeasure_dr.py <out_dir> <region> zA_min zA_max zB_min zB_max imock... [--nran 2]
           [--correlations AA,BB] [--suffix _dr2] [--engine cucount|corrfunc] [--nthreads 28] [--skip-existing]
Reads  <out_dir>/mocks/mock<i>/xi_<corr>_<region>.npy and writes xi_<corr>_<region><suffix>.npy (claims .claim<suffix>_<region>).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/global/homes/w/willlake/Packages/desi-clustering")
p = argparse.ArgumentParser()
p.add_argument("out_dir"); p.add_argument("region"); p.add_argument("zA_min", type=float); p.add_argument("zA_max", type=float)
p.add_argument("zB_min", type=float); p.add_argument("zB_max", type=float); p.add_argument("imocks", type=int, nargs="+")
p.add_argument("--nran", type=int, default=2); p.add_argument("--correlations", default="AA,BB"); p.add_argument("--suffix", default="_dr2")
p.add_argument("--engine", default="corrfunc"); p.add_argument("--nthreads", type=int, default=28); p.add_argument("--skip-existing", action="store_true")
p.add_argument("--tracer", default="LRG"); p.add_argument("--version", default="glam-uchuu-v2-altmtl")
a = p.parse_args()
if a.engine == "cucount":
    import jax
    jax.config.update("jax_enable_x64", True)
from clustering_statistics import tools  # noqa: E402
from pycorr import TwoPointCorrelationFunction, TwoPointCounter, setup_logging  # noqa: E402
from pycorr.twopoint_estimator import get_twopoint_estimator  # noqa: E402

setup_logging("warning")
LS = get_twopoint_estimator("landyszalay")
zA, zB = (a.zA_min, a.zA_max), (a.zB_min, a.zB_max)
z_union = (min(zA[0], zB[0]), max(zA[1], zB[1]))
opts = dict(version=a.version, tracer=a.tracer, weight="default-FKP", FKP_P0=1e4, nran=a.nran)
expand = {"parent_randoms_fn": tools.get_catalog_fn(kind="parent_randoms", version="data-dr2-v2", tracer=a.tracer, nran=a.nran),
          "from_data": ["Z", "WEIGHT_SYS", "FRAC_TLOBS_TILES"]}
kw = dict(position_type="pos", los="midpoint", engine=a.engine)
if a.engine == "corrfunc":
    kw["nthreads"] = a.nthreads
labels = {"AA": ("A", "A"), "BB": ("B", "B"), "AB": ("A", "B")}
labels = {k: labels[k] for k in a.correlations.split(",")}


def arrays(cat):
    pos = np.column_stack([np.asarray(cat[c]) for c in ("X", "Y", "Z")]) if "X" in cat.columns() else np.asarray(cat["POSITION"])
    return pos, np.asarray(cat["WEIGHT"])


# the measurement script's exact conversion (positions, weights) of a prepared catalog
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "desi_zbins_measure_mocks.py")).read()
i0 = src.index("def arrays(cat):")
exec(src[i0:src.index("\n\n", i0)])  # noqa: S102  (redefines arrays)

for imock in a.imocks:
    mdir = os.path.join(a.out_dir, "mocks", f"mock{imock}")
    fns = {n: os.path.join(mdir, f"xi_{n}_{a.region}{a.suffix}.npy") for n in labels}
    if a.skip_existing and all(os.path.exists(f) for f in fns.values()):
        print(f"mock {imock}: exists", flush=True)
        continue
    if a.skip_existing:
        try:
            os.mkdir(os.path.join(mdir, f".claim{a.suffix}_{a.region}"))
        except FileExistsError:
            print(f"mock {imock}: claimed", flush=True)
            continue
    t0 = time.time()
    data = tools.read_catalog(kind="data", region=a.region, zrange=z_union, imock=imock, **opts)
    data = tools.prepare_catalog(data, kind="data", zrange=z_union, region=a.region, **opts)
    randoms = tools.read_catalog(kind="randoms", region=a.region, zrange=z_union, imock=imock, expand=expand, concatenate=False, **opts)
    randoms = [tools.prepare_catalog(r, kind="randoms", zrange=z_union, region=a.region, **opts) for r in randoms]
    cat = {}
    for name, zr in (("A", zA), ("B", zB)):
        dpos, dw = arrays(tools.mask_catalog(data, "data", zrange=zr, region=a.region))
        rs = [arrays(tools.mask_catalog(r, "randoms", zrange=zr, region=a.region)) for r in randoms]
        cat[name] = dict(dpos=dpos, dw=dw, rpos=np.concatenate([r[0] for r in rs]), rw=np.concatenate([r[1] for r in rs]))
    for n, (c1n, c2n) in labels.items():
        if a.skip_existing and os.path.exists(fns[n]):
            continue
        old = TwoPointCorrelationFunction.load(os.path.join(mdir, f"xi_{n}_{a.region}.npy"))
        edges = old.D1D2.edges
        c1, c2 = cat[c1n], cat[c2n]
        if old.R1R2.size1 != len(c1["rw"]):
            raise RuntimeError(f"mock {imock} {n}: stored RR used {old.R1R2.size1} randoms, loaded {len(c1['rw'])} ({a.nran} files)")
        t1 = time.time()
        d1r2 = TwoPointCounter("smu", edges, positions1=c1["dpos"], weights1=c1["dw"], positions2=c2["rpos"], weights2=c2["rw"], **kw)
        if c1n == c2n:
            new = LS(D1D2=old.D1D2, D1R2=d1r2, R1R2=old.R1R2)
        else:
            r1d2 = TwoPointCounter("smu", edges, positions1=c1["rpos"], weights1=c1["rw"], positions2=c2["dpos"], weights2=c2["dw"], **kw)
            new = LS(D1D2=old.D1D2, D1R2=d1r2, R1D2=r1d2, R1R2=old.R1R2)
        new.save(fns[n])
        lo, hi = int(edges[0][0]), int(edges[0][-1])
        d = np.nanmean(np.asarray(new.corr)[max(lo, 40):hi] - np.asarray(old.corr)[max(lo, 40):hi])
        print(f"  {n}: DR with {len(c2['rw'])} randoms in {time.time() - t1:.0f} s; mean(xi_new - xi_old) over 40-{hi} = {d:+.2e}", flush=True)
    print(f"mock {imock} done in {time.time() - t0:.0f} s", flush=True)
