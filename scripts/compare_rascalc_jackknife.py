"""Jackknife-mode comparison with the RascalC C++ code on a periodic box.

Jackknife regions are cubic sub-boxes; the jackknife RR counts (RascalC convention) are computed here with a
KD-tree so that both codes get identical weights. Runs both codes in ``s_mu`` jackknife mode (which also outputs
the EE/RR arrays) or ``legendre`` jackknife mode and compares c2j, c3j, c4j (and EE1/RR1 for s_mu).

Usage: python compare_rascalc_jackknife.py <mode: s_mu|legendre> <N_randoms> <n_loops> <nthread>
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import spherical_jn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj
from rascaljax.compat import rascalc_command, read_rascalc_outputs, run_rascalc, write_rascalc_inputs

mode = sys.argv[1] if len(sys.argv) > 1 else "s_mu"
L = 600.0
N_r = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
n_loops = int(sys.argv[3]) if len(sys.argv) > 3 else 32
nthread = int(sys.argv[4]) if len(sys.argv) > 4 else 1
work = sys.argv[5] if len(sys.argv) > 5 else "/tmp/claude-102016/-global-u2-w-willlake-Projects-DESI-MultitracerBAO/db282ac8-ac12-4352-9440-1f2f53df60c1/scratchpad/cmp_jack_" + mode
binary = os.path.expanduser("~/Packages/RascalC/RascalC/bin/cov." + ("legendre_projected" if mode == "legendre" else "s_mu") + "_jackknife_periodic")
N_g = 30_000.0

k = np.linspace(1e-4, 2.0, 40001)
P = 2e4 * (k / 0.1) / (1 + (k / 0.1) ** 3.5) * np.exp(-((k * 3.0) ** 2))


def xi_of_r(r):
    r = np.atleast_1d(r)
    return np.trapezoid(k[None, :] ** 2 * P[None, :] * spherical_jn(0, k[None, :] * r[:, None]), k, axis=1) / (2 * np.pi**2)


s_tab = np.arange(1.0, 201.0, 1.0)
mu_tab = np.linspace(0.025, 0.975, 20)
table = rj.XiTable.from_model(s_tab, mu_tab, np.repeat(xi_of_r(s_tab)[:, None], len(mu_tab), axis=1))
s_edges = np.arange(20.0, 81.0, 15.0)
n_s = len(s_edges) - 1
n_mu, max_l = 4, 2
binning = rj.Binning(s_edges, n_mu, max_l if mode == "legendre" else None)

rng = np.random.default_rng(5)
pos = rng.uniform(0, L, size=(N_r, 3))
n_jk_side = 3
jk = ((pos // (L / n_jk_side)).astype(int) * np.array([n_jk_side**2, n_jk_side, 1])).sum(axis=1)
n_jack = n_jk_side**3
w = np.ones(N_r) / N_r

# pair counts in (s, mu) bins, total and per jackknife region (RascalC convention: pairs with one particle in the
# region count fully, both particles -> counted twice then halved: RR_A = sum_pairs w_i w_j [ (J_i == A) + (J_j == A) ] / 2)
print("Counting pairs...", flush=True)
tree = cKDTree(pos, boxsize=L)
pairs = tree.query_pairs(s_edges[-1], output_type="ndarray")
i, j = pairs[:, 0], pairs[:, 1]
d = pos[i] - pos[j]
d -= L * np.round(d / L)
r = np.sqrt((d**2).sum(axis=1))
mu = np.abs(d[:, 2]) / r
b = binning.flat_bin(r, mu)
ok = b >= 0
i, j, b = i[ok], j[ok], b[ok]
ww = w[i] * w[j]
rr_total = 2 * np.bincount(b, weights=ww, minlength=binning.n_bins)  # ordered pairs
rr_jack = np.zeros((n_jack, binning.n_bins))
np.add.at(rr_jack, (jk[i], b), ww)
np.add.at(rr_jack, (jk[j], b), ww)
assert np.allclose(rr_jack.sum(axis=0), rr_total)
rr = rr_total.reshape(n_s, n_mu)
jack_weights = rr_jack / rr_total[None, :]
labels = np.arange(n_jack)
print(f"{len(i)} pairs counted; min RR per bin {rr.min():.3e}", flush=True)

cat = rj.Catalog(pos, np.ones(N_r), n_data=N_g, jackknife=jk, name="box")
os.makedirs(work, exist_ok=True)
cpp_dir = os.path.join(work, "rascalc")
paths = write_rascalc_inputs(cpp_dir, cat, binning, table, rr, jack_weights, labels)
cmd = rascalc_command(binary, cpp_dir, paths, cat, binning, table, N2=8, N3=8, N4=8, n_loops=n_loops, loops_per_sample=1,
                      nthread=nthread, xi_cut=200.0, boxsize=L, n_side=25, cf_loops=0, seed=11)
print("Running RascalC:", " ".join(cmd), flush=True)
t0 = time.time()
run_rascalc(cmd, cpp_dir)
print(f"RascalC done in {time.time() - t0:.1f} s", flush=True)
cpp = read_rascalc_outputs(cpp_dir, binning)
cppj = read_rascalc_outputs(cpp_dir, binning, jack=True)

corr = rj.CorrelationInput(s_edges, rr, table, jack_xi=np.zeros_like(rr_jack), jack_rr=rr_jack, jack_labels=labels)
t0 = time.time()
raw = rj.run_cov([rj.Sample(cat, name="all")], {(0, 0): corr}, mode=mode, max_l=max_l, n_mu=n_mu, N2=8, N3=8, N4=8, n_loops=n_loops,
                 loops_per_sample=1, boxsize=L, xi_cut=200.0, refine_iterations=0, quads_per_batch=2_000_000, seed=3,
                 out_dir=os.path.join(work, "rascaljax"))
print(f"RascalJax done in {time.time() - t0:.1f} s", flush=True)

np.set_printoptions(precision=3, linewidth=160)


def compare(name, mine_sub, theirs_sub, theirs_full):
    mine = mine_sub.mean(axis=0)
    err = np.sqrt((mine_sub.std(axis=0) ** 2 + theirs_sub.std(axis=0) ** 2) / len(mine_sub))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mine / theirs_full
        pull = (mine - theirs_full) / err
    big = np.abs(theirs_full) > 0.05 * np.abs(theirs_full).max()
    print(f"==== {name}: diag ratio: {np.diag(ratio) if ratio.ndim == 2 else ratio}")
    print(f"     pulls on the {big.sum()} largest elements: mean {pull[big].mean():.2f}, rms {np.sqrt((pull[big]**2).mean()):.2f}, max |pull| {np.abs(pull[big]).max():.2f}")
    print(f"     RMS relative difference on those elements: {np.sqrt(np.mean((ratio[big] - 1) ** 2)):.3f}")


compare("c2", raw.c2[0, :, 0, 0], cpp["c2"], cpp["c2_full"])
compare("c3", raw.c3[0, :, 0, 0], cpp["c3"], cpp["c3_full"])
compare("c4", raw.c4[:, 0, 0], cpp["c4"], cpp["c4_full"])
compare("c2j", raw.c2j[0, :, 0, 0], cppj["c2"], cppj["c2_full"])
compare("c3j", raw.c3j[0, :, 0, 0], cppj["c3"], cppj["c3_full"])
compare("c4j", raw.c4j[:, 0, 0], cppj["c4"], cppj["c4_full"])
if mode == "s_mu":
    print("RR estimate / input:", raw.rr_estimate.mean(axis=0)[0] / rr.ravel())
    for key, mine in (("EE1", raw.ee[:, 0, 0]), ("EE2", raw.ee[:, 0, 1]), ("RR1", raw.rrj[:, 0, 0]), ("RR2", raw.rrj[:, 0, 1])):
        theirs = cppj[key + "_full"]
        m = mine.mean(axis=0)
        print(f"==== {key}: ratio of sums over regions {m.sum(axis=0) / theirs.sum(axis=0)}; RMS relative difference {np.sqrt(np.mean((m / theirs - 1) ** 2)):.3f}")
np.savez(os.path.join(work, "comparison.npz"), **{"cpp_" + k: v for k, v in cpp.items()}, **{"cppj_" + k: v for k, v in cppj.items()},
         mine_c2=raw.c2, mine_c3=raw.c3, mine_c4=raw.c4, mine_c2j=raw.c2j, mine_c3j=raw.c3j, mine_c4j=raw.c4j, mine_ee=raw.ee, mine_rrj=raw.rrj)
