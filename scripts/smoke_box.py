"""Smoke test: uniform randoms in a periodic box with a model xi; checks RR and C2 against analytic expectations."""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj

L = 1000.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
n_data = 50_000.0
rng = np.random.default_rng(1)
pos = rng.uniform(0, L, size=(N, 3))
cat = rj.Catalog(pos, np.ones(N), n_data=n_data, name="box")
sample = rj.Sample(cat, name="all")

# model xi: power law with a smooth cutoff
s_tab = np.arange(1.0, 200.0, 2.0)
mu_tab = np.linspace(0.05, 0.95, 10)
xi_iso = (s_tab / 8.0) ** -1.8 * np.exp(-(s_tab / 120.0) ** 2)
xi_tab = np.repeat(xi_iso[:, None], len(mu_tab), axis=1)
table = rj.XiTable.from_model(s_tab, mu_tab, xi_tab)

s_edges = np.arange(20.0, 101.0, 10.0)
n_mu = 4
# analytic RR for uniform randoms: fraction of (ordered, distinct) pairs in each bin
shell = 4 * np.pi / 3 * (s_edges[1:] ** 3 - s_edges[:-1] ** 3) / L**3
rr = np.repeat((shell / n_mu)[:, None], n_mu, axis=1) * (1 - 1 / N)
corr = rj.CorrelationInput(s_edges, rr, table)

t0 = time.time()
raw = rj.run_cov([sample], {(0, 0): corr}, mode="legendre", max_l=2, N2=5, N3=5, N4=5, n_loops=4, loops_per_sample=1,
                 boxsize=L, xi_cut=150.0, refine_iterations=0, quads_per_batch=1_000_000, seed=1)
print("runtime", time.time() - t0)
rr_est = raw.rr_estimate.mean(axis=0)[0].reshape(len(s_edges) - 1, n_mu)
print("RR estimate / analytic:\n", rr_est / rr)
# C2 in (s, ell) bins: for ell=0 block, C2[s, l=0, s, l=0] should be 2 (1 + xi_bar) / (n_data^2 RR_a) projected: sum over mu bins of (RR_a^-1 ...)
c2 = raw.c2.mean(axis=1)[0, 0, 0]  # (n_out, n_out)
n_ell = raw.binning.n_ell
# expected C2 in (s, mu): diagonal 2 (1 + xi) / (n_data^2 RR_a); project with factors
xi_bar = table(0.5 * (s_edges[1:] + s_edges[:-1]))
c2_smu = np.zeros((raw.binning.n_bins, raw.binning.n_bins))
for i in range(len(s_edges) - 1):
    for m in range(n_mu):
        a = i * n_mu + m
        c2_smu[a, a] = 2 * (1 + xi_bar[i]) / (n_data**2 * rr[i, m])
c2_exp = raw.binning.project_matrix(c2_smu)
d = np.diag(c2)
print("C2 diag ratio (ell=0 entries):", (d / np.diag(c2_exp))[0::n_ell])
print("C2 diag ratio (ell=2 entries):", (d / np.diag(c2_exp))[1::n_ell])
print("counts", {k: v if v.ndim == 1 else v.sum() for k, v in raw.counts.items()})
c4 = raw.c4.mean(axis=0)[0, 0]
c3 = raw.c3.mean(axis=1)[0, 0, 0]
print("C4 diag:", np.diag(c4)[::n_ell])
print("C3 diag:", np.diag(c3)[::n_ell])
print("C2 diag:", d[::n_ell])
