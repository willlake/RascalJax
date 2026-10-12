import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp, numpy as np, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profile_kernel import I, data, st, key, N2, N3, N4
from profile_ablation import bench
st_smu = dataclasses.replace(st, n_ell=0)
orig = I.scatter_matrix
for rnd in range(2):
    bench(st, "legendre, real scatter")
    bench(st_smu, "s_mu, real scatter")
    I.scatter_matrix = lambda acc, values, bin_a, bin_b, valid, st_, lf: acc.at[0, 0, 0].add(jnp.sum(values))
    bench(st, "legendre, sum instead of scatter")
    bench(st_smu, "s_mu, sum instead of scatter")
    I.scatter_matrix = orig
    bench(dataclasses.replace(st, c4_terms=()), "legendre, no c4 terms")
    bench(dataclasses.replace(st_smu, c4_terms=()), "s_mu, no c4 terms")
