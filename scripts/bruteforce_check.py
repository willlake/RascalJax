"""Exact (brute-force) evaluation of the 2-, 3- and 4-point sums on a small periodic catalog, compared with the
Monte Carlo estimates of RascalJax and (optionally) RascalC in s_mu mode.

Usage: python bruteforce_check.py [N] [n_loops] [--rascalc]
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
n_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 64
use_rascalc = "--rascalc" in sys.argv
aperiodic = "--aperiodic" in sys.argv
L = 300.0
N_g = N / 2.0
g = N_g / N

k = np.linspace(1e-4, 2.0, 40001)
P = 2e4 * (k / 0.1) / (1 + (k / 0.1) ** 3.5) * np.exp(-((k * 3.0) ** 2))


def xi_of_r(r):
    r = np.atleast_1d(r)
    return np.trapezoid(k[None, :] ** 2 * P[None, :] * spherical_jn(0, k[None, :] * r[:, None]), k, axis=1) / (2 * np.pi**2)


s_tab = np.arange(1.0, 151.0, 1.0)
mu_tab = np.linspace(0.025, 0.975, 20)
table = rj.XiTable.from_model(s_tab, mu_tab, np.repeat(xi_of_r(s_tab)[:, None], len(mu_tab), axis=1))
s_edges = np.array([20.0, 40.0, 60.0])
n_s, n_mu = 2, 2
binning = rj.Binning(s_edges, n_mu, None)
n_bins = binning.n_bins

rng = np.random.default_rng(8)
pos = rng.uniform(0, L, size=(N, 3))
w = np.ones(N) / N

# ----- brute force -----
d = pos[:, None, :] - pos[None, :, :]
if not aperiodic:
    d -= L * np.round(d / L)
r = np.sqrt((d**2).sum(axis=-1))
np.fill_diagonal(r, 1.0)
if aperiodic:  # midpoint line of sight
    los = pos[:, None, :] + pos[None, :, :]
    mu = np.abs((d * los).sum(axis=-1)) / r / np.sqrt((los**2).sum(axis=-1))
else:
    mu = np.abs(d[..., 2]) / r
X = table(r, mu)  # xi matrix with the same interpolant as the code
np.fill_diagonal(X, 0.0)
np.fill_diagonal(r, -1.0)
b = binning.flat_bin(r, mu)  # -1 for self and out of range
rr = np.array([np.sum(w[:, None] * w[None, :] * (b == a)) for a in range(n_bins)])
print("RR (normalized counts):", rr)
# C2_a = 2 sum_{i != j in a} w_i^2 w_j^2 (1 + xi_ij) / (g^2 RR_a^2)
W2 = w[:, None] ** 2 * w[None, :] ** 2
c2_bf = np.array([2 * np.sum(W2 * (1 + X) * (b == a)) for a in range(n_bins)]) / (g**2 * rr**2)
# C3_ab = 4 sum_{i,j,k distinct} w_i w_j^2 w_k xi_ik Theta_a(ij) Theta_b(jk) / (g RR_a RR_b)
M = [w[:, None] * w[None, :] * (b == a) for a in range(n_bins)]  # M_a[i, j]
c3_bf = np.zeros((n_bins, n_bins))
c4_bf = np.zeros((n_bins, n_bins))
for a in range(n_bins):
    for bb in range(n_bins):
        # sum_{i,j,k} M_a[i,j] M_b[k,j] xi_ik, with i != k (xi_ii = 0 already), i != j, k != j (M excludes self)
        c3_bf[a, bb] = 4 * np.sum(M[a] * (X @ M[bb].T)) / (g * rr[a] * rr[bb])  # (X M_b^T)[i, j] = sum_k xi_ik M_b[k, j]... careful: want sum_k M_b[k,j] xi_ik = (X @ M_b)[i, j] since M_b symmetric
        # 4-point: sum_{ijkl} M_a[i,j] M_b[k,l] (xi_ik xi_jl + xi_il xi_jk) with all indices distinct
        T1 = np.sum(M[a] * (X @ M[bb] @ X.T))  # sum_ij M_a[i,j] sum_kl xi_ik M_b[k,l] xi_jl ; xi_ii = 0 kills i=k and j=l
        # subtract i == l and j == k contributions (and add back both)
        il = np.sum(M[a] * X.T * (X @ M[bb].T))  # i = l: sum_{ijk} M_a[i,j] xi_ik xi_ji M_b[k,i] -> sum_ij M_a[i,j] xi_ji (sum_k xi_ik M_b[k,i]) ; (X @ M_b)[i,i] diag -> use elementwise
        # recompute properly:
        XMb = X @ M[bb]  # [i, l] = sum_k xi_ik M_b[k, l]
        il = np.sum(M[a] * X.T * np.diag(XMb)[:, None])  # sum_ij M_a[i,j] xi_ji (XMb)[i,i]
        jk = np.sum(M[a] * X * np.diag(M[bb] @ X.T)[None, :])  # j = k: sum_ij M_a[i,j] xi_ij sum_l M_b[j,l] xi_jl... wait xi_il with i: need sum_l M_b[j,l] xi_il -> depends on i and j
        jk = np.sum(M[a] * X * (M[bb] @ X.T).T)  # (M_b X^T)[j, i] = sum_l M_b[j,l] xi_il
        both = np.sum(M[a] * X * X.T * M[bb].T)  # i = l and j = k: M_a[i,j] xi_ij xi_ji M_b[j,i]
        term1 = T1 - il - jk + both
        c4_bf[a, bb] = 2 * term1 / (rr[a] * rr[bb])  # the second Wick term equals the first by symmetry of the sum
print("brute-force C2:", c2_bf)
print("brute-force C3:\n", c3_bf)
print("brute-force C4:\n", c4_bf)

cat = rj.Catalog(pos, np.ones(N), n_data=N_g, name="box")
corr = rj.CorrelationInput(s_edges, rr.reshape(n_s, n_mu), table)
t0 = time.time()
raw = rj.run_cov([rj.Sample(cat)], {(0, 0): corr}, mode="s_mu", n_mu=n_mu, N2=10, N3=10, N4=10, n_loops=n_loops, loops_per_sample=1, boxsize=None if aperiodic else L,
                 xi_cut=150.0, refine_iterations=0, quads_per_batch=2_000_000, seed=5, n_side=11, log=lambda s: None)
print(f"RascalJax: {time.time() - t0:.1f} s")
np.set_printoptions(precision=4, linewidth=160)
c2 = raw.c2[0, :, 0, 0]
c3 = raw.c3[0, :, 0, 0]
c4 = raw.c4[:, 0, 0]
print("RascalJax C2 / brute force:", np.diag(c2.mean(axis=0)) / c2_bf, "+-", np.diag(c2.std(axis=0)) / np.sqrt(n_loops) / c2_bf)
print("RascalJax C3 / brute force:\n", c3.mean(axis=0) / c3_bf, "\n +-\n", c3.std(axis=0) / np.sqrt(n_loops) / c3_bf)
print("RascalJax C4 / brute force:\n", c4.mean(axis=0) / c4_bf, "\n +-\n", c4.std(axis=0) / np.sqrt(n_loops) / c4_bf)
print("RR estimate / exact:", raw.rr_estimate.mean(axis=0)[0] / rr)

if use_rascalc:
    from rascaljax.compat import rascalc_command, read_rascalc_outputs, run_rascalc, write_rascalc_inputs

    work = "/tmp/claude-102016/-global-u2-w-willlake-Projects-DESI-MultitracerBAO/db282ac8-ac12-4352-9440-1f2f53df60c1/scratchpad/bf_rascalc" + ("_aperiodic" if aperiodic else "")
    binary = os.path.expanduser("~/Packages/RascalC/RascalC/bin/cov.s_mu" + ("" if aperiodic else "_periodic"))
    paths = write_rascalc_inputs(work, cat, binning, table, rr.reshape(n_s, n_mu))
    cmd = rascalc_command(binary, work, paths, cat, binning, table, N2=10, N3=10, N4=10, n_loops=n_loops, loops_per_sample=1, nthread=1,
                          xi_cut=150.0, boxsize=None if aperiodic else L, n_side=11, cf_loops=0, seed=11)
    t0 = time.time()
    run_rascalc(cmd, work)
    print(f"RascalC: {time.time() - t0:.1f} s")
    cpp = read_rascalc_outputs(work, binning)
    print("RascalC C2 / brute force:", np.diag(cpp["c2_full"]) / c2_bf, "+-", np.diag(cpp["c2"].std(axis=0)) / np.sqrt(n_loops) / c2_bf)
    print("RascalC C3 / brute force:\n", cpp["c3_full"] / c3_bf, "\n +-\n", cpp["c3"].std(axis=0) / np.sqrt(n_loops) / c3_bf)
    print("RascalC C4 / brute force:\n", cpp["c4_full"] / c4_bf, "\n +-\n", cpp["c4"].std(axis=0) / np.sqrt(n_loops) / c4_bf)
    print("RascalC RR / exact:", cpp["RR_full"] / rr)
