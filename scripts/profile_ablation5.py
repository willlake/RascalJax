import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp, numpy as np, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profile_kernel import I, data, st, key
from profile_ablation import bench
orig = I.scatter_matrix
bench(st, "legendre, current segment_sum into (s,s) x ell^2")

def scatter_smu_then_project(acc, values, bin_a, bin_b, valid, st_, lf):
    if not st_.legendre:
        return orig(acc, values, bin_a, bin_b, valid, st_, lf)
    st2 = dataclasses.replace(st_, n_ell=0)
    big = orig(jnp.zeros((values.shape[0], st2.n_out, st2.n_out)), values, bin_a, bin_b, valid, st2, lf)
    P = jnp.zeros((st_.n_s, st_.n_mu, st_.n_s, st_.n_ell)).at[jnp.arange(st_.n_s), :, jnp.arange(st_.n_s), :].set(lf).reshape(st2.n_out, st_.n_out)
    return acc + jnp.einsum("tab,ap,bq->tpq", big, P, P)
I.scatter_matrix = scatter_smu_then_project
bench(st, "legendre via (s,mu)^2 segment_sum + projection")

def scatter_dense(acc, values, bin_a, bin_b, valid, st_, lf):
    n_terms, n = values.shape
    values = jnp.where(valid[None, :], values, 0.0)
    bin_a = jnp.where(valid, bin_a, 0); bin_b = jnp.where(valid, bin_b, 0)
    s_a, mu_a = bin_a // st_.n_mu, bin_a % st_.n_mu
    s_b, mu_b = bin_b // st_.n_mu, bin_b % st_.n_mu
    n_ell = st_.n_ell
    rows = jnp.arange(n)[:, None]
    A = jnp.zeros((n, st_.n_out)).at[rows, s_a[:, None] * n_ell + jnp.arange(n_ell)[None, :]].set(lf[mu_a])
    B = jnp.zeros((n, st_.n_out)).at[rows, s_b[:, None] * n_ell + jnp.arange(n_ell)[None, :]].set(lf[mu_b])
    out = acc
    for t in range(n_terms):
        out = out.at[t].add((A * values[t][:, None]).T @ B)
    return out
I.scatter_matrix = scatter_dense
bench(st, "legendre via dense one-hot GEMM")

def scatter_sorted(acc, values, bin_a, bin_b, valid, st_, lf):
    n_terms, n = values.shape
    values = jnp.where(valid[None, :], values, 0.0)
    bin_a = jnp.where(valid, bin_a, 0); bin_b = jnp.where(valid, bin_b, 0)
    s_a, mu_a = bin_a // st_.n_mu, bin_a % st_.n_mu
    s_b, mu_b = bin_b // st_.n_mu, bin_b % st_.n_mu
    n_ell = st_.n_ell
    seg = s_a * st_.n_s + s_b
    order = jnp.argsort(seg)
    seg_s = seg[order]
    ff = (lf[mu_a][:, :, None] * lf[mu_b][:, None, :]).reshape(n, n_ell * n_ell)[order]
    out = acc
    for t in range(n_terms):
        upd = jax.ops.segment_sum(values[t][order][:, None] * ff, seg_s, num_segments=st_.n_s * st_.n_s, indices_are_sorted=True)
        upd = upd.reshape(st_.n_s, st_.n_s, n_ell, n_ell).transpose(0, 2, 1, 3).reshape(st_.n_out, st_.n_out)
        out = out.at[t].add(upd)
    return out
I.scatter_matrix = scatter_sorted
bench(st, "legendre via sort + sorted segment_sum")

def scatter_chunked(acc, values, bin_a, bin_b, valid, st_, lf, chunk=1024):
    """Two-level: per chunk of contributions a one-hot (chunk x n_seg) reduction via matmul, then sum over chunks."""
    n_terms, n = values.shape
    values = jnp.where(valid[None, :], values, 0.0)
    bin_a = jnp.where(valid, bin_a, 0); bin_b = jnp.where(valid, bin_b, 0)
    s_a, mu_a = bin_a // st_.n_mu, bin_a % st_.n_mu
    s_b, mu_b = bin_b // st_.n_mu, bin_b % st_.n_mu
    n_ell = st_.n_ell
    n_seg = st_.n_s * st_.n_s
    seg = s_a * st_.n_s + s_b
    pad = (-n) % chunk
    seg = jnp.pad(seg, (0, pad)).reshape(-1, chunk)
    ff = (lf[mu_a][:, :, None] * lf[mu_b][:, None, :]).reshape(n, n_ell * n_ell)
    ff = jnp.pad(ff, ((0, pad), (0, 0))).reshape(-1, chunk, n_ell * n_ell)
    onehot = (seg[:, :, None] == jnp.arange(n_seg)[None, None, :]).astype(jnp.float64)  # (n_chunks, chunk, n_seg)
    out = acc
    for t in range(n_terms):
        v = jnp.pad(values[t], (0, pad)).reshape(-1, chunk, 1)
        upd = jnp.einsum("cks,ckf->sf", onehot, ff * v)
        upd = upd.reshape(st_.n_s, st_.n_s, n_ell, n_ell).transpose(0, 2, 1, 3).reshape(st_.n_out, st_.n_out)
        out = out.at[t].add(upd)
    return out
I.scatter_matrix = scatter_chunked
bench(st, "legendre via chunked one-hot einsum")
I.scatter_matrix = orig
