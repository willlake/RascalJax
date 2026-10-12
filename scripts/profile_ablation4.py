import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp, numpy as np, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profile_kernel import I, data, st, key
from profile_ablation import bench
st_smu = dataclasses.replace(st, n_ell=0)
for rnd in range(3):
    bench(st, "legendre")
    bench(st_smu, "s_mu")
# legendre but with the scatter done in s_mu style then projected (reference)
orig = I.scatter_matrix
def scatter_smu_then_project(acc, values, bin_a, bin_b, valid, st_, lf):
    st2 = dataclasses.replace(st_, n_ell=0)
    big = orig(jnp.zeros((values.shape[0], st2.n_out, st2.n_out)), values, bin_a, bin_b, valid, st2, lf)
    P = jnp.zeros((st_.n_s, st_.n_mu, st_.n_s, st_.n_ell)).at[jnp.arange(st_.n_s), :, jnp.arange(st_.n_s), :].set(lf).reshape(st2.n_out, st_.n_out)
    return acc + jnp.einsum("tab,ap,bq->tpq", big, P, P)
I.scatter_matrix = scatter_smu_then_project
bench(st, "legendre via s_mu scatter + projection")
I.scatter_matrix = orig
