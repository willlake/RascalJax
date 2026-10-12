"""Vary RascalC settings on the brute-force box to localize the C++ C3/C4 bias."""
import os, sys, time, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj
from rascaljax.compat import rascalc_command, read_rascalc_outputs, run_rascalc, write_rascalc_inputs
work0 = "/tmp/claude-102016/-global-u2-w-willlake-Projects-DESI-MultitracerBAO/db282ac8-ac12-4352-9440-1f2f53df60c1/scratchpad/bf_rascalc"
binary = os.path.expanduser("~/Packages/RascalC/RascalC/bin/cov.s_mu_periodic")
# reuse inputs written by bruteforce_check.py (N=4000 run)
paths = {k: os.path.join(work0, v) for k, v in dict(randoms="randoms.txt", binfile="radial_binning_cov.csv", binfile_cf="radial_binning_corr.csv", xi="xi_11.dat", rr="RR_counts.dat").items()}
s_edges = np.array([20.0, 40.0, 60.0]); binning = rj.Binning(s_edges, 2, None)
table = rj.XiTable.read(paths["xi"])
pos = np.loadtxt(paths["randoms"])[:, :3]
cat = rj.Catalog(pos, np.ones(len(pos)), n_data=len(pos) / 2, name="box")
bf = np.load(os.path.join(work0, "..", "bf_reference.npz")) if os.path.exists(os.path.join(work0, "..", "bf_reference.npz")) else None
configs = {"base_N10": dict(N2=10, N3=10, N4=10, n_loops=256, n_side=11), "N3_2_N4_2": dict(N2=10, N3=2, N4=2, n_loops=1024, n_side=11),
           "N2_2": dict(N2=2, N3=10, N4=10, n_loops=512, n_side=11), "nside21": dict(N2=10, N3=10, N4=10, n_loops=256, n_side=21)}
for name, c in configs.items():
    work = work0 + "_" + name
    os.makedirs(work, exist_ok=True)
    cmd = rascalc_command(binary, work, paths, cat, binning, table, N2=c["N2"], N3=c["N3"], N4=c["N4"], n_loops=c["n_loops"], loops_per_sample=1, nthread=1,
                          xi_cut=150.0, boxsize=300.0, n_side=c["n_side"], cf_loops=0, seed=13)
    t0 = time.time(); run_rascalc(cmd, work); el = time.time() - t0
    cpp = read_rascalc_outputs(work, binning)
    n = len(cpp["c3"])
    print(f"== {name} ({el:.0f} s, {n} loops): C3 diag {np.diag(cpp['c3_full'])} +- {np.diag(cpp['c3'].std(axis=0)) / np.sqrt(n)}; C4 diag {np.diag(cpp['c4_full'])} +- {np.diag(cpp['c4'].std(axis=0)) / np.sqrt(n)}", flush=True)
