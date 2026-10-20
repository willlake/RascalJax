"""Pre-reconstruction xi(s, mu) of two overlapping redshift bins of one DESI tracer on GLAM-Uchuu v2 altmtl mocks:
the two auto-correlations and their cross-correlation, with pycorr's cucount (GPU) engine. Must run in the cosmodesi
environment (source .../cosmodesi_environment.sh main), one process per GPU.

The random-random term dominates the cost but its Poisson noise is irrelevant, while the data-random term sets the
random-noise contribution to the estimator variance. DD, DR and RR use ``--nran-dr`` random files (DESI default 10) with
the mock's own weights; ``--nran-rr`` can reduce the RR set, but DR and RR must share their randoms for the Landy-Szalay
cancellation of the random-catalog edge noise (a 2-file RR against a 10-file DR added spurious, radial large-scale
variance to the sample covariance: BENCHMARK.md section 6), so the default is the same number of files.

Usage: python desi_zbins_measure_mocks.py <out_dir> <region: NGC|SGC> <zA_min> <zA_max> <zB_min> <zB_max> <imock> [imock ...]
           [--nran-dr 10] [--nran-rr N] [--s-max 160] [--tracer LRG] [--save-randoms] [--save-randoms-files 4] [--skip-existing]
           [--engine cucount|corrfunc] [--nthreads 32] [--correlations AA,BB,AB]

The sample covariance of a BAO fit needs the two auto-correlations only (``--correlations AA,BB``); the cross-correlation
AB is needed, together with large separations, for the few mocks that provide the inputs of the covariance integrals.

Outputs: <out_dir>/mocks/mock<i>/xi_{AA,BB,AB}_<region>.npy (pycorr TwoPointCorrelationFunction, s bins of 1 Mpc/h, 200 mu
bins in [-1, 1]) and, with --save-randoms, <out_dir>/randoms_<region>_mock<i>.npz with the expanded randoms of the first
``--save-randoms-files`` files over the union of the two bins (Cartesian positions in Mpc/h, weights WEIGHT x WEIGHT_FKP,
redshifts) and the number of data galaxies. Mock randoms are stored without positions: they are expanded from the DR2
parent randoms (positions) and the mock data (redshifts), as the official pipeline does.
"""
import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("out_dir")
p.add_argument("region", choices=["NGC", "SGC"])
p.add_argument("zA_min", type=float)
p.add_argument("zA_max", type=float)
p.add_argument("zB_min", type=float)
p.add_argument("zB_max", type=float)
p.add_argument("imocks", type=int, nargs="+")
p.add_argument("--nran-dr", type=int, default=10, help="random files for the data-random term (DESI default 10)")
p.add_argument("--nran-rr", type=int, default=None, help="random files for the random-random term (default: the same as --nran-dr; "
               "a smaller RR set breaks the Landy-Szalay cancellation of the random-catalog edge noise and adds spurious, radial, "
               "large-scale variance to the sample covariance: see BENCHMARK.md section 6)")
p.add_argument("--s-max", type=float, default=160.0)
p.add_argument("--tracer", default="LRG")
p.add_argument("--version", default="glam-uchuu-v2-altmtl")
p.add_argument("--save-randoms", action="store_true")
p.add_argument("--save-randoms-files", type=int, default=4)
p.add_argument("--skip-existing", action="store_true", help="skip measured mocks; several workers can then share one list: each mock is "
               "claimed atomically (<mock dir>/.claim_<region>) by the first worker reaching it")
p.add_argument("--ignore-claims", action="store_true", help="with --skip-existing, re-measure mocks whose claim was left by a dead worker")
p.add_argument("--engine", default="cucount", choices=["cucount", "corrfunc"], help="pycorr engine: cucount (GPU) or corrfunc (CPU)")
p.add_argument("--nthreads", type=int, default=32, help="CPU threads for the corrfunc engine")
p.add_argument("--correlations", default="AA,BB,AB", help="which of AA, BB, AB to measure")
a = p.parse_args()

if a.engine == "corrfunc":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the catalog tools import jax; keep the GPU free for other jobs
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from clustering_statistics import tools  # noqa: E402
from pycorr import TwoPointCorrelationFunction, TwoPointCounter, setup_logging  # noqa: E402

setup_logging("warning")
zA, zB = (a.zA_min, a.zA_max), (a.zB_min, a.zB_max)
z_union = (min(zA[0], zB[0]), max(zA[1], zB[1]))
if a.nran_rr is None:
    a.nran_rr = a.nran_dr
elif a.nran_rr != a.nran_dr:
    print(f"WARNING: --nran-rr {a.nran_rr} != --nran-dr {a.nran_dr}: DR and RR use different randoms, so the random-catalog edge noise of RR does not cancel "
          "in the Landy-Szalay estimator and the sample covariance gains spurious radial large-scale variance (BENCHMARK.md section 6)", flush=True)
nran = max(a.nran_dr, a.nran_rr, a.save_randoms_files if a.save_randoms else 0)
opts = dict(version=a.version, tracer=a.tracer, weight="default-FKP", FKP_P0=1e4, nran=nran)
expand = {"parent_randoms_fn": tools.get_catalog_fn(kind="parent_randoms", version="data-dr2-v2", tracer=a.tracer, nran=nran),
          "from_data": ["Z", "WEIGHT_SYS", "FRAC_TLOBS_TILES"]}  # reused across mocks: caches the parent randoms
edges = (np.linspace(0.0, a.s_max, int(a.s_max) + 1), np.linspace(-1.0, 1.0, 201))
kw = dict(position_type="pos", los="midpoint", engine=a.engine)
if a.engine == "corrfunc":
    kw["nthreads"] = a.nthreads
labels = {"AA": (zA, zA), "BB": (zB, zB), "AB": (zA, zB)}
labels = {k: labels[k] for k in a.correlations.split(",")}


def arrays(cat):
    return np.asarray(cat["POSITION"], dtype=np.float64), np.asarray(cat["INDWEIGHT"], dtype=np.float64)


for imock in a.imocks:
    mdir = os.path.join(a.out_dir, "mocks", f"mock{imock}")
    os.makedirs(mdir, exist_ok=True)
    fns = {k: os.path.join(mdir, f"xi_{k}_{a.region}.npy") for k in labels}
    if a.skip_existing and all(os.path.exists(fn) for fn in fns.values()) and not a.save_randoms:
        print(f"mock {imock}: all outputs exist, skipping", flush=True)
        continue
    if a.skip_existing and not a.ignore_claims:
        try:
            os.mkdir(os.path.join(mdir, f".claim_{a.region}"))
        except FileExistsError:
            print(f"mock {imock}: claimed by another worker, skipping", flush=True)
            continue
    t0 = time.time()
    data = tools.read_catalog(kind="data", region=a.region, zrange=z_union, imock=imock, **opts)
    data = tools.prepare_catalog(data, kind="data", zrange=z_union, region=a.region, **opts)
    randoms = tools.read_catalog(kind="randoms", region=a.region, zrange=z_union, imock=imock, expand=expand, concatenate=False, **opts)
    randoms = [tools.prepare_catalog(r, kind="randoms", zrange=z_union, region=a.region, **opts) for r in randoms]
    n_data = int(data.csize)
    print(f"mock {imock} {a.region}: {n_data} data, {sum(int(r.csize) for r in randoms)} randoms ({nran} files) in z{z_union[0]:g}-{z_union[1]:g}; "
          f"read in {time.time() - t0:.0f} s", flush=True)
    if a.save_randoms:
        sel = randoms[: a.save_randoms_files]
        pos = np.concatenate([np.asarray(r["POSITION"]) for r in sel])
        w = np.concatenate([np.asarray(r["INDWEIGHT"]) for r in sel])
        z = np.concatenate([np.asarray(r["Z"]) for r in sel])
        nA = int(tools.mask_catalog(data, "data", zrange=zA, region=a.region).csize)
        nB = int(tools.mask_catalog(data, "data", zrange=zB, region=a.region).csize)
        fn = os.path.join(a.out_dir, f"randoms_{a.region}_mock{imock}.npz")
        np.savez_compressed(fn, positions=pos, weights=w, z=z, n_data=n_data, n_data_A=nA, n_data_B=nB, zbins=np.array([zA, zB]),
                            region=a.region, imock=imock, nran=len(sel), n_data_weighted=float(np.sum(np.asarray(data["INDWEIGHT"]))))
        print(f"  saved {fn}: {len(pos)} randoms from {len(sel)} files, n_data {n_data} (A {nA}, B {nB})", flush=True)

    # per-bin arrays: data, randoms for DR (nran_dr files) and for RR (nran_rr files)
    cat = {}
    for name, zr in (("A", zA), ("B", zB)):
        dpos, dw = arrays(tools.mask_catalog(data, "data", zrange=zr, region=a.region))
        rs = [arrays(tools.mask_catalog(r, "randoms", zrange=zr, region=a.region)) for r in randoms]
        rpos_dr = np.concatenate([r[0] for r in rs[: a.nran_dr]])
        rw_dr = np.concatenate([r[1] for r in rs[: a.nran_dr]])
        rpos_rr = np.concatenate([r[0] for r in rs[: a.nran_rr]])
        rw_rr = np.concatenate([r[1] for r in rs[: a.nran_rr]])
        cat[name] = dict(dpos=dpos, dw=dw, rpos_dr=rpos_dr, rw_dr=rw_dr, rpos_rr=rpos_rr, rw_rr=rw_rr)
        print(f"  bin {name} z{zr[0]:g}-{zr[1]:g}: {len(dw)} data, {len(rw_dr)} randoms (DR), {len(rw_rr)} randoms (RR)", flush=True)
    del randoms

    for key, (z1, z2) in labels.items():
        if a.skip_existing and os.path.exists(fns[key]):
            continue
        t1 = time.time()
        c1, c2 = cat["A" if z1 == zA else "B"], cat["A" if z2 == zA else "B"]
        if z1 == z2:
            rr = TwoPointCounter("smu", edges, positions1=c1["rpos_rr"], weights1=c1["rw_rr"], **kw)
            t_rr = time.time() - t1
            est = TwoPointCorrelationFunction("smu", edges, data_positions1=c1["dpos"], data_weights1=c1["dw"], randoms_positions1=c1["rpos_dr"],
                                              randoms_weights1=c1["rw_dr"], R1R2=rr, estimator="landyszalay", **kw)
        else:
            rr = TwoPointCounter("smu", edges, positions1=c1["rpos_rr"], weights1=c1["rw_rr"], positions2=c2["rpos_rr"], weights2=c2["rw_rr"], **kw)
            t_rr = time.time() - t1
            est = TwoPointCorrelationFunction("smu", edges, data_positions1=c1["dpos"], data_weights1=c1["dw"], data_positions2=c2["dpos"],
                                              data_weights2=c2["dw"], randoms_positions1=c1["rpos_dr"], randoms_weights1=c1["rw_dr"],
                                              randoms_positions2=c2["rpos_dr"], randoms_weights2=c2["rw_dr"], R1R2=rr, estimator="landyszalay", **kw)
        est.save(fns[key])
        print(f"  {key} (z{z1[0]:g}-{z1[1]:g} x z{z2[0]:g}-{z2[1]:g}) done in {time.time() - t1:.0f} s (RR {t_rr:.0f} s) -> {fns[key]}", flush=True)
    print(f"mock {imock} done in {time.time() - t0:.0f} s", flush=True)
