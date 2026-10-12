import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp, numpy as np, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profile_kernel import I, data, st, key, N2, N3, N4
from profile_ablation import bench  # noqa (runs its benchmarks on import? no: guarded below)

bench(st, "full")
orig = I.scatter_matrix
I.scatter_matrix = lambda acc, values, bin_a, bin_b, valid, st_, lf: acc.at[0, 0, 0].add(jnp.sum(values))
bench(st, "c4 values computed, scatter replaced by a sum")
I.scatter_matrix = lambda acc, values, bin_a, bin_b, valid, st_, lf: orig(acc, values, bin_a, bin_b, valid, dataclasses.replace(st_, n_ell=0), lf)[:, : st_.n_out, : st_.n_out] if False else acc.at[0, 0, 0].add(jnp.sum(values * bin_a * bin_b))
bench(st, "c4 values + bins used, no scatter")
I.scatter_matrix = orig
# s_mu scatter instead of legendre
st_smu = dataclasses.replace(st, n_ell=0)
bench(st_smu, "full in s_mu mode (160x160 output, plain segment_sum)")
# legendre scatter via dense one-hot matmul
def scatter_dense(acc, values, bin_a, bin_b, valid, st_, lf):
    n_terms = values.shape[0]
    values = jnp.where(valid[None, :], values, 0.0)
    bin_a = jnp.where(valid, bin_a, 0); bin_b = jnp.where(valid, bin_b, 0)
    s_a, mu_a = bin_a // st_.n_mu, bin_a % st_.n_mu
    s_b, mu_b = bin_b // st_.n_mu, bin_b % st_.n_mu
    n_ell = st_.n_ell
    A = jnp.zeros((values.shape[1], st_.n_out)).at[jnp.arange(values.shape[1])[:, None], s_a[:, None] * n_ell + jnp.arange(n_ell)[None, :]].set(lf[mu_a])
    B = jnp.zeros((values.shape[1], st_.n_out)).at[jnp.arange(values.shape[1])[:, None], s_b[:, None] * n_ell + jnp.arange(n_ell)[None, :]].set(lf[mu_b])
    out = acc
    for t in range(n_terms):
        out = out.at[t].add((A * values[t][:, None]).T @ B)
    return out
I.scatter_matrix = scatter_dense
bench(st, "legendre scatter via dense one-hot GEMM")
I.scatter_matrix = orig
