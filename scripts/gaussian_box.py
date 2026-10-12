"""Validation against the analytic Gaussian covariance of xi multipoles in a periodic box with an isotropic P(k).

The RascalJax terms should match: C4 <-> P^2 part, C3 (alpha=1) <-> 2 P / n part, C2 <-> (1 + xi) / n^2 part.
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj

L = 1000.0
N_r = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
n_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N_g = 100_000.0
n_gal = N_g / L**3
V = L**3

# ----- model P(k) and xi(r) -----
k = np.linspace(1e-4, 2.0, 40001)
P = 2e4 * (k / 0.1) / (1 + (k / 0.1) ** 3.5) * np.exp(-((k * 3.0) ** 2))


def xi_of_r(r):
    r = np.atleast_1d(r)
    out = np.trapezoid(k[None, :] ** 2 * P[None, :] * spherical_jn(0, k[None, :] * r[:, None]), k, axis=1) / (2 * np.pi**2)
    return out


s_tab = np.arange(1.0, 251.0, 1.0)
mu_tab = np.linspace(0.025, 0.975, 20)
xi_tab = np.repeat(xi_of_r(s_tab)[:, None], len(mu_tab), axis=1)
table = rj.XiTable.from_model(s_tab, mu_tab, xi_tab)

s_edges = np.arange(20.0, 101.0, 10.0)
n_s = len(s_edges) - 1
n_mu = 20
max_l = 2
shell = 4 * np.pi / 3 * (s_edges[1:] ** 3 - s_edges[:-1] ** 3) / V
rr = np.repeat((shell / n_mu)[:, None], n_mu, axis=1) * (1 - 1 / N_r)

rng = np.random.default_rng(2)
cat = rj.Catalog(rng.uniform(0, L, size=(N_r, 3)), np.ones(N_r), n_data=N_g, name="box")
sample = rj.Sample(cat, name="all")
corr = rj.CorrelationInput(s_edges, rr, table)

t0 = time.time()
raw = rj.run_cov([sample], {(0, 0): corr}, mode="legendre", max_l=max_l, n_mu=n_mu, N2=10, N3=10, N4=10, n_loops=n_loops, loops_per_sample=1,
                 boxsize=L, xi_cut=250.0, refine_iterations=0, quads_per_batch=2_000_000, seed=3)
print("runtime", time.time() - t0)
raw.save("/tmp/claude-102016/-global-u2-w-willlake-Projects-DESI-MultitracerBAO/db282ac8-ac12-4352-9440-1f2f53df60c1/scratchpad/gaussian_box_raw.npz")

# ----- analytic Gaussian covariance (Grieb et al. 2016 with isotropic P) -----
n_ell = max_l // 2 + 1
ells = np.arange(0, max_l + 1, 2)
s_fine = [np.linspace(s_edges[i], s_edges[i + 1], 41) for i in range(n_s)]


def jbar(ell, i):
    s = s_fine[i]
    w = s**2
    js = spherical_jn(ell, k[None, :] * s[:, None])
    return np.trapezoid(w[:, None] * js, s, axis=0) / np.trapezoid(w, s)


jb = np.array([[jbar(ell, i) for i in range(n_s)] for ell in ells])  # (n_ell, n_s, n_k)


def cov_term(weight_k):
    out = np.zeros((n_s, n_ell, n_s, n_ell))
    for il, ell in enumerate(ells):
        integrand = k**2 * weight_k / (2 * np.pi**2)
        for a in range(n_s):
            for b in range(n_s):
                out[a, il, b, il] = (2 * ell + 1) * 2 / V * np.trapezoid(integrand * jb[il, a] * jb[il, b], k)
    return out.reshape(n_s * n_ell, n_s * n_ell)


C4_th = cov_term(P**2)
C3_th = cov_term(2 * P / n_gal)
C2_th = cov_term(np.ones_like(P) / n_gal**2)
xi_bar = xi_of_r(0.5 * (s_edges[1:] + s_edges[:-1]))
C2_th = C2_th * (1 + np.repeat(xi_bar, n_ell))[:, None]

c4 = raw.c4.mean(axis=0)[0, 0]
c3 = raw.c3.mean(axis=1)[0, 0, 0]
c2 = raw.c2.mean(axis=1)[0, 0, 0]
c4_err = raw.c4.std(axis=0)[0, 0] / np.sqrt(raw.n_sub)
np.set_printoptions(precision=3, linewidth=160, suppress=False)
for name, mine, th in (("C2", c2, C2_th), ("C3", c3, C3_th), ("C4", c4, C4_th)):
    d = np.diag(mine) / np.diag(th)
    print(f"{name} diagonal ratio (ell=0): {d[0::n_ell]}")
    print(f"{name} diagonal ratio (ell=2): {d[1::n_ell]}")
    off = mine[0::n_ell, 0::n_ell]
    print(f"{name} ell=0 block ratio, first row: {off[0] / th[0::n_ell, 0::n_ell][0]}")
print("C4 ell=0 block mine:\n", c4[0::n_ell, 0::n_ell])
print("C4 ell=0 block theory:\n", C4_th[0::n_ell, 0::n_ell])
print("C4 ell=0 block MC error:\n", c4_err[0::n_ell, 0::n_ell])
print("C4 ell=2 block mine:\n", c4[1::n_ell, 1::n_ell])
print("C4 ell=2 block theory:\n", C4_th[1::n_ell, 1::n_ell])
print("C4 ell0-ell2 block mine (theory 0):\n", c4[0::n_ell, 1::n_ell])
