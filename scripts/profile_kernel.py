"""Profile the batch kernel on the Gaussian box setup: throughput vs batch size, with and without the scatter."""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rascaljax as rj
from rascaljax import integrals as I
from rascaljax.driver import _to_kernel, plan_passes
from rascaljax.grid import CellGrid, make_geometry
from rascaljax.kernels import r2_cell_probabilities, xi_cell_probabilities

L = 1000.0
N_r = 200_000
rng = np.random.default_rng(2)
cat = rj.Catalog(rng.uniform(0, L, size=(N_r, 3)), np.ones(N_r), n_data=1e5, name="box")
s_tab = np.arange(1.0, 251.0, 1.0)
mu_tab = np.linspace(0.025, 0.975, 20)
table = rj.XiTable.from_model(s_tab, mu_tab, np.repeat(((s_tab / 8) ** -1.8 * np.exp(-((s_tab / 120) ** 2)))[:, None], 20, axis=1))
s_edges = np.arange(20.0, 101.0, 10.0)
n_mu, max_l = 20, 2
binning = rj.Binning(s_edges, n_mu, max_l)
geom = make_geometry([cat], 41, L, 100.0)
grid = CellGrid.from_catalog(cat, geom)
memb = grid.sort_membership(cat.weights)[None]
ga = I.GridArrays(jnp.asarray(grid.positions), jnp.asarray(grid.jackknife, dtype=jnp.int32), jnp.asarray(grid.rand_class, dtype=jnp.int32),
                  jnp.asarray(geom.cell_3d(grid.positions).astype(np.int32)), jnp.asarray(grid.cell_start), jnp.asarray(grid.cell_count), jnp.asarray(memb))
r2_n_half = int(np.ceil(200 / geom.cellsize))
xi_n_half = int(np.ceil(250 / geom.cellsize))
kr2, _ = _to_kernel(r2_cell_probabilities(r2_n_half, geom.cellsize))
kxi, _ = _to_kernel(xi_cell_probabilities(xi_n_half, geom.cellsize, table.isotropic))
lk = table.lookup(0.25, 201)
shell = 4 * np.pi / 3 * (s_edges[1:] ** 3 - s_edges[:-1] ** 3) / L**3
rr = np.repeat((shell / n_mu)[:, None], n_mu, axis=1)
passes, _ = plan_passes([0], [(0, 0)])
p = passes[0]
N2, N3, N4 = 10, 10, 10
st = I.Static(N2, N3, N4, p.legs, True, binning.n_s, n_mu, binning.n_ell, 1, False, True, True, xi_n_half, r2_n_half, 201,
              tuple(p.c4_terms), tuple(p.t3_terms), tuple(p.c2_terms), tuple(p.rr_terms))
data = I.PassData((ga,), jnp.asarray(geom.n_side), jnp.asarray(geom.boxsize), kr2, kxi, kxi, jnp.asarray(lk.table)[None], jnp.zeros((1, 1), dtype=jnp.int32),
                  jnp.asarray(lk.s_max), jnp.asarray(lk.ds), jnp.asarray(s_edges), jnp.asarray(binning.projection_matrix()), jnp.asarray(rr.ravel())[None],
                  jnp.zeros((1, 1, binning.n_bins)))
key = jax.random.PRNGKey(0)


def bench(batch, label, n_rep=5):
    acc = I.zero_accumulators(st)
    idx = jnp.arange(batch)
    acc = I.accumulate_batch_jit(key, idx, data, st, acc)
    jax.block_until_ready(acc)
    t0 = time.time()
    for i in range(n_rep):
        acc = I.accumulate_batch_jit(jax.random.fold_in(key, i), idx, data, st, acc)
    jax.block_until_ready(acc)
    el = (time.time() - t0) / n_rep
    print(f"{label}: batch {batch} -> {batch * N2 * N3 * N4 / el:.3e} draws/s ({el * 1e3:.1f} ms per batch)", flush=True)


for batch in ([] if __name__ != "__main__" else (1000, 2000, 5000, 10000)):
    bench(batch, "full")

# without scatter
orig = I.scatter_matrix
I.scatter_matrix = lambda acc, values, bin_a, bin_b, valid, st, lf: acc + jnp.sum(values) * 0
I.accumulate_batch_jit = jax.jit(I.accumulate_batch, static_argnames=("st",))
for batch in ([] if __name__ != "__main__" else (2000, 10000)):
    bench(batch, "no scatter")
I.scatter_matrix = orig
