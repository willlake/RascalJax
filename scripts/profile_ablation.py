"""Ablation profile of the batch kernel: which part costs the time?"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profile_kernel import I, data, st, key, N2, N3, N4  # noqa: E402  (reuses the setup; profile_kernel runs its own benchmarks on import)
import dataclasses


def bench(st_, label, batch=5000, n_rep=4):
    def fresh(key, primary_idx, data, st, acc):  # new function object each time: defeats JAX's trace cache so monkeypatches take effect
        return I.accumulate_batch(key, primary_idx, data, st, acc)

    fn = jax.jit(fresh, static_argnames=("st",))
    acc = I.zero_accumulators(st_)
    idx = jnp.arange(batch)
    acc = fn(key, idx, data, st_, acc)
    jax.block_until_ready(acc)
    el = np.inf
    for _ in range(5):  # take the best of several rounds (the GPU may be shared)
        t0 = time.time()
        for i in range(n_rep):
            acc = fn(jax.random.fold_in(key, i), idx, data, st_, acc)
        jax.block_until_ready(acc)
        el = min(el, (time.time() - t0) / n_rep)
    per = N2 * (N3 if st_.do_triples else 1) * (N4 if st_.do_quads else 1)
    print(f"{label}: {batch * per / el:.3e} draws/s ({el * 1e3:.1f} ms per batch of {batch * per} draws)", flush=True)


if __name__ == "__main__":
    bench(dataclasses.replace(st, do_triples=False, do_quads=False), "pairs only", batch=200000)
    bench(dataclasses.replace(st, do_quads=False), "pairs+triples", batch=50000)
    bench(st, "full")
    bench(dataclasses.replace(st, c4_terms=()), "full, no c4 terms (sampling + bins only)")

    orig_xi = I.xi_lookup
    I.xi_lookup = lambda table, row, s_max, ds, n_mu, r, mu: 0.01 * jnp.exp(-r / 50.0)
    bench(st, "full, trivial xi")
    I.xi_lookup = orig_xi

    orig_draw = I.draw_offsets
    I.draw_offsets = lambda key, kernel, n_half, shape: (jnp.zeros(tuple(shape) + (3,), dtype=jnp.int32) + jnp.array([1, 0, 0]), jnp.full(shape, 1e-3))
    bench(st, "full, fixed offsets (no kernel draws)")
    I.draw_offsets = orig_draw

    orig_pick = I.draw_particle_in_cell
    I.draw_particle_in_cell = lambda key, grid, n_side, periodic, cell3: (jnp.zeros(cell3.shape[:-1], dtype=jnp.int64), jnp.ones(cell3.shape[:-1]), jnp.ones(cell3.shape[:-1], dtype=bool))
    bench(st, "full, fixed particle picks (no cell gathers)")
    I.draw_particle_in_cell = orig_pick

    orig_sep = I.separation
    I.separation = lambda p1, p2, boxsize, periodic: (jnp.sum(p1 - p2, axis=-1) % 90.0 + 10.0, jnp.full(jnp.broadcast_shapes(p1.shape, p2.shape)[:-1], 0.5))
    bench(st, "full, trivial separations")
    I.separation = orig_sep
