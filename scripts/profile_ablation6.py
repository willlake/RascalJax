"""Scatter variants: float64 segment_sum (current) vs float32 accumulation per batch (added to float64 totals)."""
import os, sys, time
import jax, jax.numpy as jnp, numpy as np, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profile_kernel import I, data, st, key
from profile_ablation import bench
orig = I.scatter_matrix

def scatter_f32(acc, values, bin_a, bin_b, valid, st_, projection):
    n_terms = values.shape[0]
    n_bins = st_.n_bins
    bin_a = jnp.where(valid, bin_a, 0); bin_b = jnp.where(valid, bin_b, 0)
    seg = bin_a * n_bins + bin_b
    out = acc
    for t in range(n_terms):
        v = jnp.where(valid, values[t], 0.0)
        scale = jnp.max(jnp.abs(v)) + 1e-300  # keep float32 in range
        tmp = jax.ops.segment_sum((v / scale).astype(jnp.float32), seg, num_segments=n_bins * n_bins).astype(jnp.float64).reshape(n_bins, n_bins) * scale
        if st_.legendre:
            tmp = projection.T @ tmp @ projection
        out = out.at[t].add(tmp)
    return out

for rnd in range(2):
    bench(st, "float64 segment_sum (current)")
    I.scatter_matrix = scatter_f32
    bench(st, "float32 segment_sum per batch")
    I.scatter_matrix = orig
