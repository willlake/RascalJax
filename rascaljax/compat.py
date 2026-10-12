"""Interoperability with the RascalC C++ code: writing its input files, running the binary, and reading its
output matrices in the same ordering as :class:`rascaljax.driver.RawCovariance` (single catalog / single sample).
"""
from __future__ import annotations

import os
import subprocess
from glob import glob

import numpy as np

from .binning import Binning
from .catalog import Catalog
from .utils import Array
from .xi_table import XiTable


def write_rascalc_inputs(out_dir: str, catalog: Catalog, binning: Binning, xi_table: XiTable, rr: Array,
                         jack_weights: Array | None = None, jack_labels: Array | None = None) -> dict:
    """Write the RascalC text inputs for a single-catalog run and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    cols = [catalog.positions[:, 0], catalog.positions[:, 1], catalog.positions[:, 2], catalog.weights]
    if catalog.jackknife is not None:
        cols.append(catalog.jackknife)
    paths["randoms"] = os.path.join(out_dir, "randoms.txt")
    np.savetxt(paths["randoms"], np.column_stack(cols))
    paths["binfile"] = os.path.join(out_dir, "radial_binning_cov.csv")
    np.savetxt(paths["binfile"], np.array((binning.s_edges[:-1], binning.s_edges[1:])).T)
    if xi_table.s_edges is not None:
        cf_edges = xi_table.s_edges
    else:  # edges around the tabulated points (must contain them)
        mid = 0.5 * (xi_table.s_vals[1:] + xi_table.s_vals[:-1])
        cf_edges = np.concatenate([[max(1e-4, xi_table.s_vals[0] - (mid[0] - xi_table.s_vals[0]))], mid, [2 * xi_table.s_vals[-1] - mid[-1]]])
    paths["binfile_cf"] = os.path.join(out_dir, "radial_binning_corr.csv")
    np.savetxt(paths["binfile_cf"], np.array((cf_edges[:-1], cf_edges[1:])).T)
    paths["xi"] = os.path.join(out_dir, "xi_11.dat")
    xi_table.write(paths["xi"])
    paths["rr"] = os.path.join(out_dir, "RR_counts.dat")
    np.savetxt(paths["rr"], np.asarray(rr).reshape(-1, 1))
    if binning.legendre:
        paths["mu_bin_legendre"] = os.path.join(out_dir, f"mu_bin_legendre_factors_m{binning.n_mu}_l{binning.max_l}.txt")
        np.savetxt(paths["mu_bin_legendre"], binning.legendre_factors())
    if jack_weights is not None:
        paths["jackknife"] = os.path.join(out_dir, "jackknife_weights.dat")
        np.savetxt(paths["jackknife"], np.column_stack((jack_labels, jack_weights)))
    return paths


def rascalc_command(binary: str, out_dir: str, paths: dict, catalog: Catalog, binning: Binning, xi_table: XiTable, *,
                    N2: int, N3: int, N4: int, n_loops: int, loops_per_sample: int, nthread: int, xi_cut: float,
                    boxsize: float | None, n_side: int = 301, cf_loops: int = 0, seed: int | None = None) -> list[str]:
    cmd = [binary, "-output", out_dir.rstrip("/") + "/", "-nside", str(n_side), "-rescale", "1", "-nthread", str(nthread),
           "-maxloops", str(n_loops), "-loopspersample", str(loops_per_sample), "-N2", str(N2), "-N3", str(N3), "-N4", str(N4),
           "-xicut", str(xi_cut), "-binfile", paths["binfile"], "-binfile_cf", paths["binfile_cf"], "-mbin_cf", str(len(xi_table.mu_vals)),
           "-cf_loops", str(cf_loops), "-in", paths["randoms"], "-norm", str(catalog.n_data), "-cor", paths["xi"],
           "-RRbin", paths["rr"], "-mbin", str(binning.n_mu)]
    if catalog.effective_n:
        cmd.append("-effective_norm")
    if binning.legendre:
        cmd += ["-max_l", str(binning.max_l), "-mu_bin_legendre_file", paths["mu_bin_legendre"]]
    if boxsize:
        cmd += ["-perbox", "-boxsize", str(boxsize)]
    if "jackknife" in paths:
        cmd += ["-jackknife", paths["jackknife"]]
    if seed is not None:
        cmd += ["-seed", str(seed)]
    return cmd


def run_rascalc(cmd: list[str], out_dir: str, log_file: str | None = None) -> None:
    os.makedirs(os.path.join(out_dir, "CovMatricesAll"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "CovMatricesJack"), exist_ok=True)
    env = dict(os.environ, OMP_PROC_BIND="spread", OMP_PLACES="threads")
    with open(log_file or os.path.join(out_dir, "rascalc_log.txt"), "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)


def read_rascalc_outputs(out_dir: str, binning: Binning, jack: bool = False) -> dict:
    """Read the RascalC single-tracer text outputs into arrays ``c2, c3, c4`` of shape ``(n_sub, n_out, n_out)``
    plus ``*_full``; with ``jack=True`` the jackknife versions and the ``EE``/``RR`` arrays."""
    sub = "CovMatricesJack" if jack else "CovMatricesAll"
    d = os.path.join(out_dir, sub)
    tag = f"n{binning.n_s}_l{binning.max_l}" if binning.legendre else f"n{binning.n_s}_m{binning.n_mu}"
    names = {"c2": f"c2_{tag}_11_", "c3": f"c3_{tag}_1,11_", "c4": f"c4_{tag}_11,11_"}
    out = {}
    for key, prefix in names.items():
        files = sorted(glob(os.path.join(d, prefix + "[0-9]*.txt")), key=lambda f: int(os.path.basename(f)[len(prefix):-4]))
        mats = []
        for f in files:
            m = np.loadtxt(f)
            if m.ndim == 1:
                m = np.diag(m)
            mats.append(m)
        out[key] = np.array(mats)
        full = np.loadtxt(os.path.join(d, prefix + "full.txt"))
        out[key + "_full"] = np.diag(full) if full.ndim == 1 else full
    if not binning.legendre and not jack:
        out["RR_full"] = np.loadtxt(os.path.join(d, f"RR_{tag}_11_full.txt"))
    if jack and not binning.legendre:
        for key in ("EE1", "EE2", "RR1", "RR2"):
            out[key + "_full"] = np.loadtxt(os.path.join(d, f"{key}_{tag}_11_full.txt"))
    return out
