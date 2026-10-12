"""Brute-force check of the multi-sample machinery: two overlapping samples on one catalog (overlapping "redshift"
slabs of a periodic box) and two disjoint samples as separate catalogs, against exact sums with membership weights.

Usage: python bruteforce_multi.py [N] [n_loops]
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj
from rascaljax.utils import symmetrized

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
n_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 256
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

rng = np.random.default_rng(9)
pos = rng.uniform(0, L, size=(N, 3))
w = np.ones(N) / N
z = pos[:, 2]
memb = {"A": (z < 0.6 * L).astype(float), "B": (z >= 0.4 * L).astype(float)}  # overlapping slabs

# ----- exact sums -----
d = pos[:, None, :] - pos[None, :, :]
d -= L * np.round(d / L)
r = np.sqrt((d**2).sum(axis=-1))
np.fill_diagonal(r, 1.0)
mu = np.abs(d[..., 2]) / r
X = table(r, mu)
np.fill_diagonal(X, 0.0)
np.fill_diagonal(r, -1.0)
b = binning.flat_bin(r, mu)
corrs = [("A", "A"), ("A", "B"), ("B", "B")]


def M(S, T, a):
    """M^{ST}_a[i, j] = w_i^S w_j^T Theta_a(ij)."""
    return (w * memb[S])[:, None] * (w * memb[T])[None, :] * (b == a)


rr = {c: np.array([np.sum(M(c[0], c[1], a)) for a in range(n_bins)]) for c in corrs}


def c2_exact(c1, c2):
    A, B = c1
    C, D = c2
    out = np.zeros(n_bins)
    for a in range(n_bins):
        ww = (w * memb[A])[:, None] * (w * memb[B])[None, :] * ((w * memb[C])[:, None] * (w * memb[D])[None, :] + (w * memb[D])[:, None] * (w * memb[C])[None, :])
        out[a] = np.sum(ww * (1 + X) * (b == a)) / (g**2 * rr[c1][a] * rr[c2][a])
    return out


def t_exact(Xs, Y, Y2, Z, a, bb):
    """T[X,(Y,Y'),Z]_ab = sum_{ijk} w_i^X w_j^Y w_j^Y' w_k^Z xi_ik Theta_a(ij) Theta_b(kj), distinct indices."""
    Ma = (w * memb[Xs])[:, None] * (w * memb[Y])[None, :] * (b == a)  # [i, j]
    Mb = (w * memb[Z])[:, None] * (w * memb[Y2])[None, :] * (b == bb)  # [k, j]
    return np.sum(Ma * (X @ Mb))  # sum_ij Ma[i,j] sum_k xi_ik Mb[k,j]


def c3_exact(c1, c2):
    A, B = c1
    C, D = c2
    out = np.zeros((n_bins, n_bins))
    for a in range(n_bins):
        for bb in range(n_bins):
            tot = t_exact(A, B, D, C, a, bb) + t_exact(A, B, C, D, a, bb) + t_exact(B, A, D, C, a, bb) + t_exact(B, A, C, D, a, bb)
            out[a, bb] = tot / (g * rr[c1][a] * rr[c2][bb])
    return out


def c4_exact(c1, c2):
    A, B = c1
    C, D = c2
    out = np.zeros((n_bins, n_bins))
    for a in range(n_bins):
        for bb in range(n_bins):
            Ma = M(A, B, a)
            Mb = M(C, D, bb)
            # term1: sum M_a[i,j] M_b[k,l] xi_ik xi_jl ; term2: xi_il xi_jk
            T1 = np.sum(Ma * (X @ Mb @ X.T))
            T2 = np.sum(Ma * (X @ Mb.T @ X.T))
            # remove i == l (term1) / i == k (term2 has xi_il with l=i -> xi_ii = 0 kills), j == k (term1) / j == l (term2)
            XMb = X @ Mb
            il1 = np.sum(Ma * X.T * np.diag(XMb)[:, None])  # term1 with l = i
            jk1 = np.sum(Ma * X * (Mb @ X.T).T)  # term1 with k = j
            both1 = np.sum(Ma * X * X.T * Mb.T)
            XMbT = X @ Mb.T
            ik2 = np.sum(Ma * X.T * np.diag(XMbT)[:, None])  # term2 with k = i: M_a[i,j] xi_il xi_ji M_b[i,l]
            jl2 = np.sum(Ma * X * (Mb.T @ X.T).T)  # term2 with l = j: M_a[i,j] xi_ij xi_jk M_b[k,j]
            both2 = np.sum(Ma * X * X.T * Mb)
            out[a, bb] = (T1 - il1 - jk1 + both1 + T2 - ik2 - jl2 + both2) / (rr[c1][a] * rr[c2][bb])
    return out


t0 = time.time()
exact = {}
for i1, c1 in enumerate(corrs):
    for i2, c2 in enumerate(corrs[i1:], i1):
        exact[(i1, i2)] = (c2_exact(c1, c2), c3_exact(c1, c2), c4_exact(c1, c2))
print(f"exact sums in {time.time() - t0:.0f} s")

# ----- RascalJax: two samples on one catalog -----
cat = rj.Catalog(pos, np.ones(N), n_data=N_g, name="box")
sA, sB = rj.Sample(cat, memb["A"], "A"), rj.Sample(cat, memb["B"], "B")
inputs = {(0, 0): rj.CorrelationInput(s_edges, rr[("A", "A")].reshape(n_s, n_mu), table),
          (0, 1): rj.CorrelationInput(s_edges, rr[("A", "B")].reshape(n_s, n_mu), table),
          (1, 1): rj.CorrelationInput(s_edges, rr[("B", "B")].reshape(n_s, n_mu), table)}
kw = dict(mode="s_mu", n_mu=n_mu, N2=10, N3=10, N4=10, n_loops=n_loops, loops_per_sample=1, boxsize=L, xi_cut=150.0, refine_iterations=0,
          quads_per_batch=2_000_000, seed=5, n_side=11, log=lambda s: None)
t0 = time.time()
raw = rj.run_cov([sA, sB], inputs, **kw)
print(f"RascalJax (one catalog, overlapping samples): {time.time() - t0:.1f} s")
np.set_printoptions(precision=3, linewidth=160)


def report(raw, label):
    print(f"===== {label}")
    for (i1, i2), (c2e, c3e, c4e) in exact.items():
        c2 = raw.c2.sum(axis=0)[:, i1, i2]  # sum over catalog pairs (alpha = 1)
        c3 = raw.c3.sum(axis=0)[:, i1, i2]
        c4 = raw.c4[:, i1, i2]
        n = len(c4)
        print(f"block {corrs[i1]} x {corrs[i2]}:")
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.any(c2e != 0):
                print("  C2 diag / exact:", np.diag(c2.mean(axis=0)) / c2e, "+-", np.diag(c2.std(axis=0)) / np.sqrt(n) / c2e)
            else:
                print("  C2 exact is zero; mine max |C2| =", np.abs(c2).max())
            print("  C3 / exact (diag):", np.diag(c3.mean(axis=0) / c3e), "+-", np.diag(c3.std(axis=0) / np.sqrt(n) / np.abs(c3e)))
            print("  C4 / exact (diag):", np.diag(c4.mean(axis=0) / c4e), "+-", np.diag(c4.std(axis=0) / np.sqrt(n) / np.abs(c4e)))
            big = np.abs(c4e) > 0.2 * np.abs(c4e).max()
            print("  C3 / exact (all elements with |C3e| > 20% max):", (c3.mean(axis=0) / c3e)[np.abs(c3e) > 0.2 * np.abs(c3e).max()])
            print("  C4 / exact (all elements with |C4e| > 20% max):", (c4.mean(axis=0) / c4e)[big])


report(raw, "two overlapping samples on one catalog")

# ----- RascalJax: disjoint samples as two catalogs -----
memb["A"] = (z < 0.5 * L).astype(float)
memb["B"] = (z >= 0.5 * L).astype(float)
rr = {c: np.array([np.sum(M(c[0], c[1], a)) for a in range(n_bins)]) for c in corrs}
exact = {}
for i1, c1 in enumerate(corrs):
    for i2, c2 in enumerate(corrs[i1:], i1):
        exact[(i1, i2)] = (c2_exact(c1, c2), c3_exact(c1, c2), c4_exact(c1, c2))
selA, selB = memb["A"] > 0, memb["B"] > 0
catA = rj.Catalog(pos[selA], np.ones(selA.sum()), n_data=N_g * selA.mean(), name="A")
catB = rj.Catalog(pos[selB], np.ones(selB.sum()), n_data=N_g * selB.mean(), name="B")
# with catalog weights normalized per catalog, the exact sums above use w = 1/N for all particles: pass the
# matching normalized counts (rr computed with w = 1/N) and rescale weights so that both conventions agree
inputs = {(0, 0): rj.CorrelationInput(s_edges, rr[("A", "A")].reshape(n_s, n_mu) * (N / selA.sum()) ** 2, table),
          (0, 1): rj.CorrelationInput(s_edges, rr[("A", "B")].reshape(n_s, n_mu) * (N / selA.sum()) * (N / selB.sum()), table),
          (1, 1): rj.CorrelationInput(s_edges, rr[("B", "B")].reshape(n_s, n_mu) * (N / selB.sum()) ** 2, table)}
t0 = time.time()
raw2 = rj.run_cov([rj.Sample(catA, name="A"), rj.Sample(catB, name="B")], inputs, **kw)
print(f"RascalJax (two catalogs): {time.time() - t0:.1f} s; passes: see counts keys {list(raw2.counts.keys())[:8]}")
# convert: with per-catalog unit-sum weights w' = w N/N_S, C4 (ratio of w^4 / RR^2) is invariant, C3 ~ 1/g_cat: g_cat = n_data_cat / N_cat = N_g/N same;
# C2 ~ 1/(g g) same. So blocks should agree directly.
report(raw2, "disjoint samples as two catalogs")
