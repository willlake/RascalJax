"""Term-by-term comparison with the RascalC C++ code on the Gaussian box of gaussian_box.py (same catalog, xi, binning).

Usage: python compare_rascalc_box.py <N_randoms> <n_loops> <nthread>
Runs the C++ code (legendre_projected_periodic) and RascalJax, then compares c2, c3, c4 (full matrices).
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj
from rascaljax.compat import rascalc_command, read_rascalc_outputs, run_rascalc, write_rascalc_inputs

L = 1000.0
N_r = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
n_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 32
nthread = int(sys.argv[3]) if len(sys.argv) > 3 else 1
work = sys.argv[4] if len(sys.argv) > 4 else "/tmp/claude-102016/-global-u2-w-willlake-Projects-DESI-MultitracerBAO/db282ac8-ac12-4352-9440-1f2f53df60c1/scratchpad/cmp_box"
binary = os.path.expanduser("~/Packages/RascalC/RascalC/bin/cov.legendre_projected_periodic")
N_g = 100_000.0

k = np.linspace(1e-4, 2.0, 40001)
P = 2e4 * (k / 0.1) / (1 + (k / 0.1) ** 3.5) * np.exp(-((k * 3.0) ** 2))


def xi_of_r(r):
    r = np.atleast_1d(r)
    return np.trapezoid(k[None, :] ** 2 * P[None, :] * spherical_jn(0, k[None, :] * r[:, None]), k, axis=1) / (2 * np.pi**2)


s_tab = np.arange(1.0, 251.0, 1.0)
mu_tab = np.linspace(0.025, 0.975, 20)
table = rj.XiTable.from_model(s_tab, mu_tab, np.repeat(xi_of_r(s_tab)[:, None], len(mu_tab), axis=1))
s_edges = np.arange(20.0, 101.0, 10.0)
n_mu, max_l = 20, 2
binning = rj.Binning(s_edges, n_mu, max_l)
shell = 4 * np.pi / 3 * (s_edges[1:] ** 3 - s_edges[:-1] ** 3) / L**3
rr = np.repeat((shell / n_mu)[:, None], n_mu, axis=1) * (1 - 1 / N_r)
rng = np.random.default_rng(2)
cat = rj.Catalog(rng.uniform(0, L, size=(N_r, 3)), np.ones(N_r), n_data=N_g, name="box")

os.makedirs(work, exist_ok=True)
cpp_dir = os.path.join(work, "rascalc")
paths = write_rascalc_inputs(cpp_dir, cat, binning, table, rr)
cmd = rascalc_command(binary, cpp_dir, paths, cat, binning, table, N2=10, N3=10, N4=10, n_loops=n_loops, loops_per_sample=1,
                      nthread=nthread, xi_cut=250.0, boxsize=L, n_side=41, cf_loops=0, seed=7)
print("Running RascalC:", " ".join(cmd), flush=True)
t0 = time.time()
run_rascalc(cmd, cpp_dir)
print(f"RascalC done in {time.time() - t0:.1f} s", flush=True)
cpp = read_rascalc_outputs(cpp_dir, binning)

t0 = time.time()
raw = rj.run_cov([rj.Sample(cat, name="all")], {(0, 0): rj.CorrelationInput(s_edges, rr, table)}, mode="legendre", max_l=max_l, n_mu=n_mu,
                 N2=10, N3=10, N4=10, n_loops=n_loops, loops_per_sample=1, boxsize=L, xi_cut=250.0, refine_iterations=0,
                 quads_per_batch=2_000_000, seed=3, out_dir=os.path.join(work, "rascaljax"))
print(f"RascalJax done in {time.time() - t0:.1f} s", flush=True)

n_ell = binning.n_ell
np.set_printoptions(precision=3, linewidth=160)
for name, mine_sub in (("c2", raw.c2[0, :, 0, 0]), ("c3", raw.c3[0, :, 0, 0]), ("c4", raw.c4[:, 0, 0])):
    mine = mine_sub.mean(axis=0)
    err_mine = mine_sub.std(axis=0) / np.sqrt(len(mine_sub))
    theirs = cpp[name + "_full"]
    err_cpp = cpp[name].std(axis=0) / np.sqrt(len(cpp[name]))
    err = np.sqrt(err_mine**2 + err_cpp**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mine / theirs
        pull = (mine - theirs) / err
    print(f"==== {name}: diag ratio (ell=0): {np.diag(ratio)[0::n_ell]}")
    print(f"     diag ratio (ell=2): {np.diag(ratio)[1::n_ell]}")
    big = np.abs(theirs) > 0.05 * np.abs(theirs).max()
    print(f"     pulls on the {big.sum()} largest elements: mean {pull[big].mean():.2f}, rms {np.sqrt((pull[big]**2).mean()):.2f}, max |pull| {np.abs(pull[big]).max():.2f}")
    print(f"     RMS relative difference on those elements: {np.sqrt(np.mean((ratio[big] - 1) ** 2)):.3f}")
np.savez(os.path.join(work, "comparison.npz"), **{"cpp_" + k: v for k, v in cpp.items()}, mine_c2=raw.c2, mine_c3=raw.c3, mine_c4=raw.c4)
