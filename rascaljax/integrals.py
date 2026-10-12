"""JAX kernels: batched importance sampling of particle pairs, triples and quads, and
accumulation of the 2-, 3- and 4-point covariance terms.

Notation (one "pass" has a fixed catalog for each leg i, j, k, l):

* ``i``: primary particle, every particle of the primary catalog is used ``N2`` times per loop;
* ``j``: drawn around ``i`` with the ``1/r^2`` cell kernel (pair ``ij`` in bin ``a``);
* ``k``: drawn around ``i`` with the ``xi`` cell kernel;
* ``l``: drawn around ``j`` with the ``xi`` cell kernel (pair ``kl`` in bin ``b``).

Per-contribution weights are ``w / p`` with ``p`` the sampling probability; the accumulated sums are
normalized by the number of draws in :mod:`rascaljax.driver`.

Accumulated terms, for every requested covariance block between correlations
``c1 = (A, B)`` (bins ``a``) and ``c2 = (C, D)`` (bins ``b``), in flat (s, mu) bins or projected to multipoles:

* ``C4[c1, c2][a, b] = sum w_i^A w_j^B w_k^C w_l^D [xi_AC(ik) xi_BD(jl) + xi_AD(il) xi_BC(jk)] / (RR^c1_a RR^c2_b)``
* ``T[X, (Y, Y'), Z][a, b] = sum w_i^X w_j^Y w_j^Y' w_k^Z xi_XZ(ik) Theta_a(ij) Theta_b(kj) / (RR_a RR_b)``,
  the 3-point term with the particle ``j`` shared by the two pairs (the four sharing cases of a block
  are assembled from ``T`` tensors by the driver);
* ``C2[c1, c2][a] = sum w_i^A w_j^B (w_i^C w_j^D + w_i^D w_j^C) (1 + xi_AB(ij)) Theta_a(ij) / (RR^c1_a RR^c2_a)``;
* ``RR[c][a] = sum w_i^A w_j^B Theta_a(ij)`` (a Monte Carlo estimate of the input pair counts, for checks).

With jackknives, the same terms weighted by the jackknife weight tensor (``C4j``, ``Tj``, ``C2j``) and the
``EE``/``RR`` per-region arrays for the disconnected term are accumulated as well.

All sizes that determine array shapes are static (:class:`Static`); device data lives in :class:`PassData`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------------------------
class GridArrays(NamedTuple):
    """One catalog on the grid (see :class:`rascaljax.grid.CellGrid`)."""

    positions: jax.Array  # (N, 3)
    jackknife: jax.Array  # (N,) int32 region index (zeros without jackknives)
    rand_class: jax.Array  # (N,) int32
    cell3: jax.Array  # (N, 3) int32 cell coordinates of each particle
    cell_start: jax.Array  # (n_cells,)
    cell_count: jax.Array  # (n_cells,)
    membership: jax.Array  # (n_samples, N): effective weight (catalog weight x membership) of each sample


class Kernel(NamedTuple):
    prob: jax.Array  # flat probabilities of offsets
    threshold: jax.Array  # alias-method threshold table
    alias: jax.Array  # alias-method alias table (int32)


class PassData(NamedTuple):
    """Device arrays used by one pass."""

    grids: tuple  # tuple of GridArrays, one per catalog
    n_side: jax.Array  # (3,) int
    boxsize: jax.Array  # (3,)
    kern_r2: Kernel
    kern_xi_ik: Kernel
    kern_xi_jl: Kernel
    xi_table: jax.Array  # (n_corr_xi, n_s_grid, n_mu_grid) fine xi lookups
    xi_pair_index: jax.Array  # (n_samples, n_samples) -> row of xi_table
    xi_s_max: jax.Array  # scalar
    xi_ds: jax.Array  # scalar
    s_edges: jax.Array  # (n_s + 1,)
    projection: jax.Array  # (n_bins, n_out) projection from flat (s, mu) bins to output bins (unused for (s, mu) output)
    rr: jax.Array  # (n_corr, n_bins) pair counts normalizing each correlation
    jk_weights: jax.Array  # (n_corr, n_jack, n_bins) jackknife weights (dummy without jackknives)


@dataclass(frozen=True)
class Static:
    """Static (hashable) configuration of a pass."""

    N2: int
    N3: int
    N4: int
    legs: tuple[int, int, int, int]  # catalog index of legs i, j, k, l
    periodic: bool
    n_s: int
    n_mu: int
    n_ell: int  # 0 for (s, mu) output
    n_jack: int
    jackknife: bool
    do_triples: bool
    do_quads: bool
    xi_n_half: int  # cube half-side of the xi kernels
    r2_n_half: int  # cube half-side of the 1/r^2 kernel
    xi_n_mu: int  # number of mu grid points of the xi lookup
    # term specifications: tuples of (samples..., corr_a, corr_b)
    c4_terms: tuple[tuple[int, int, int, int, int, int], ...]  # (A, B, C, D, corr_AB, corr_CD)
    t3_terms: tuple[tuple[int, int, int, int, int, int], ...]  # (X, Y, Y', Z, corr_XY, corr_ZY')
    c2_terms: tuple[tuple[int, int, int, int, int, int], ...]  # (A, B, C, D, corr_AB, corr_CD)
    rr_terms: tuple[tuple[int, int, int], ...]  # (A, B, corr_AB)

    @property
    def n_bins(self) -> int:
        return self.n_s * self.n_mu

    @property
    def n_out(self) -> int:
        return self.n_s * (self.n_ell if self.n_ell else self.n_mu)

    @property
    def legendre(self) -> bool:
        return self.n_ell > 0


class Accumulators(NamedTuple):
    """Sums accumulated in a pass (unnormalized)."""

    c4: jax.Array  # (n_c4, n_out, n_out)
    t3: jax.Array  # (n_t3, n_out, n_out)
    c2: jax.Array  # (n_c2, n_bins): diagonal in flat (s, mu) bins, projected to output bins by the driver
    rr: jax.Array  # (n_rr, n_bins)
    c4j: jax.Array
    t3j: jax.Array
    c2j: jax.Array  # (n_c2, n_bins)
    ee: jax.Array  # (n_rr, 2, n_jack, n_bins)
    rrj: jax.Array  # (n_rr, 2, n_jack, n_bins)
    n4: jax.Array  # (n_s, n_s) accepted quads per separation-bin pair (diagnostics)
    n3: jax.Array
    n2: jax.Array
    n_draws: jax.Array  # (3,) pair, triple and quad draws (including rejected ones)


def zero_accumulators(st: Static) -> Accumulators:
    z = lambda *s: jnp.zeros(s, dtype=jnp.float64)  # noqa: E731
    no, nb, nj = st.n_out, st.n_bins, max(st.n_jack, 1)
    n4, n3, n2, nr = len(st.c4_terms), len(st.t3_terms), len(st.c2_terms), len(st.rr_terms)
    return Accumulators(z(n4, no, no), z(n3, no, no), z(n2, nb), z(nr, nb), z(n4, no, no), z(n3, no, no), z(n2, nb),
                        z(nr, 2, nj, nb), z(nr, 2, nj, nb), z(st.n_s, st.n_s), z(st.n_s, st.n_s), z(st.n_s, st.n_s), z(3))


def accumulators_to_numpy(acc: Accumulators) -> dict:
    return {k: np.asarray(v) for k, v in acc._asdict().items()}


def add_accumulators(a: Accumulators, b: Accumulators) -> Accumulators:
    return Accumulators(*[x + y for x, y in zip(a, b)])


# ----------------------------------------------------------------------------------------------
# Elementary operations
# ----------------------------------------------------------------------------------------------
def _offsets_from_flat(flat, n_half: int):
    side = 2 * n_half + 1
    z = flat % side
    y = (flat // side) % side
    x = flat // (side * side)
    return jnp.stack([x, y, z], axis=-1) - n_half


def draw_offsets(key, kernel: Kernel, n_half: int, shape):
    """Draw cell offsets ``(..., 3)`` and their probabilities (alias method: two uniforms, two gathers)."""
    n = kernel.prob.shape[0]
    u = jax.random.uniform(key, (2,) + tuple(shape), dtype=jnp.float64)
    i = jnp.minimum((u[0] * n).astype(jnp.int32), n - 1)
    flat = jnp.where(u[1] < kernel.threshold[i], i, kernel.alias[i])
    return _offsets_from_flat(flat, n_half), kernel.prob[flat]


def draw_particle_in_cell(key, grid: GridArrays, n_side, periodic: bool, cell3):
    """Uniformly draw a particle in the given cells. Returns ``(index, 1/n_cell, valid)``."""
    if periodic:
        c = jnp.mod(cell3, n_side)
        inside = jnp.ones(cell3.shape[:-1], dtype=bool)
    else:
        inside = jnp.all((cell3 >= 0) & (cell3 < n_side), axis=-1)
        c = jnp.clip(cell3, 0, n_side - 1)
    flat = (c[..., 0] * n_side[1] + c[..., 1]) * n_side[2] + c[..., 2]
    count = grid.cell_count[flat]
    valid = inside & (count > 0)
    u = jax.random.uniform(key, flat.shape, dtype=jnp.float64)
    pick = jnp.minimum(jnp.floor(u * count).astype(jnp.int64), jnp.maximum(count - 1, 0))
    idx = jnp.where(valid, grid.cell_start[flat] + pick, 0)
    inv_n = jnp.where(valid, 1.0 / jnp.maximum(count, 1), 0.0)
    return idx, inv_n, valid


def separation(p1, p2, boxsize, periodic: bool):
    """``(|r|, |mu|)`` of the pair with the midpoint line of sight (periodic: z axis, minimum image)."""
    d = p1 - p2
    if periodic:
        d = d - boxsize * jnp.round(d / boxsize)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    safe_r = jnp.where(r > 0, r, 1.0)
    if periodic:
        mu = jnp.abs(d[..., 2]) / safe_r
    else:
        los = p1 + p2
        los_norm = jnp.sqrt(jnp.sum(los * los, axis=-1))
        mu = jnp.abs(jnp.sum(d * los, axis=-1)) / safe_r / jnp.where(los_norm > 0, los_norm, 1.0)
    return r, jnp.minimum(mu, 1.0)


def flat_bin(r, mu, s_edges, n_s: int, n_mu: int):
    """Flat (s, mu) bin index, or -1 outside the separation range."""
    i_s = jnp.searchsorted(s_edges, r, side="right") - 1
    ok = (i_s >= 0) & (i_s < n_s)
    i_mu = jnp.clip(jnp.floor(mu * n_mu).astype(jnp.int64), 0, n_mu - 1)
    return jnp.where(ok, i_s * n_mu + i_mu, -1)


def xi_lookup(table, row, s_max, ds, n_mu: int, r, mu):
    """Bilinear interpolation of ``xi`` in the fine table row; ``s^-4`` tail beyond ``s_max``."""
    n_s = table.shape[1]
    x = jnp.minimum(r, s_max) / ds
    i0 = jnp.clip(jnp.floor(x).astype(jnp.int64), 0, n_s - 2)
    fx = jnp.clip(x - i0, 0.0, 1.0)
    y = jnp.clip(mu, 0.0, 1.0) * (n_mu - 1)
    j0 = jnp.clip(jnp.floor(y).astype(jnp.int64), 0, n_mu - 2)
    fy = jnp.clip(y - j0, 0.0, 1.0)
    t = table[row]
    v = ((1 - fx) * (1 - fy) * t[i0, j0] + fx * (1 - fy) * t[i0 + 1, j0]
         + (1 - fx) * fy * t[i0, j0 + 1] + fx * fy * t[i0 + 1, j0 + 1])
    return v * (jnp.maximum(r, s_max) / s_max) ** -4


def jackknife_weight_tensor(w, ji, jj, jk, jl, bin_a, bin_b, corr_a: int, corr_b: int):
    """RascalC jackknife weight tensor for regions ``(ji, jj, jk, jl)`` and bins ``(a, b)`` of correlations
    ``corr_a`` (pair ij) and ``corr_b`` (pair kl):
    ``1/4 [d_ik + d_jk + d_il + d_jl] - 1/2 [w^b_{b,Ji} + w^b_{b,Jj} + w^a_{a,Jk} + w^a_{a,Jl}] + sum_A w^a_{aA} w^b_{bA}``,
    divided by ``1 - sum_A w^a_{aA} w^b_{bA}``.
    """
    wa, wb = w[corr_a], w[corr_b]  # (n_jack, n_bins)
    first = (ji == jk).astype(jnp.float64) + (jj == jk) + (ji == jl) + (jj == jl)
    second = wb[ji, bin_b] + wb[jj, bin_b] + wa[jk, bin_a] + wa[jl, bin_a]
    prod = jnp.sum(wa[:, bin_a] * wb[:, bin_b], axis=0)  # (n_contrib,)
    return (0.25 * first - 0.5 * second + prod) / (1.0 - prod)


# ----------------------------------------------------------------------------------------------
# Scatter helpers: contributions in flat (s, mu) bins -> output bins
# ----------------------------------------------------------------------------------------------
def scatter_matrix(acc, values, bin_a, bin_b, valid, st: Static, projection):
    """Add ``values`` (shape ``(n_terms, n_contrib)``) at flat (s, mu) bins (bin_a, bin_b) into ``acc`` of shape
    ``(n_terms, n_out, n_out)``.

    Contributions are first summed into a temporary ``(n_bins, n_bins)`` matrix (many segments, so the GPU
    scatter-add has little contention) and then projected to the output bins with ``projection`` for Legendre
    output. One term at a time to bound the temporary memory.
    """
    n_terms = values.shape[0]
    n_bins = st.n_bins
    bin_a = jnp.where(valid, bin_a, 0)
    bin_b = jnp.where(valid, bin_b, 0)
    seg = bin_a * n_bins + bin_b
    out = acc
    for t in range(n_terms):
        v = jnp.where(valid, values[t], 0.0)
        tmp = jax.ops.segment_sum(v, seg, num_segments=n_bins * n_bins).reshape(n_bins, n_bins)
        if st.legendre:
            tmp = projection.T @ tmp @ projection
        out = out.at[t].add(tmp)
    return out


def scatter_counts(acc, bin_a, bin_b, valid, st: Static):
    """Count accepted contributions per separation-bin pair."""
    s_a = jnp.where(valid, bin_a // st.n_mu, 0)
    s_b = jnp.where(valid, bin_b // st.n_mu, 0)
    upd = jax.ops.segment_sum(valid.astype(jnp.float64), s_a * st.n_s + s_b, num_segments=st.n_s * st.n_s)
    return acc + upd.reshape(st.n_s, st.n_s)


def scatter_vector(acc, values, bins, valid, n_bins: int):
    """Add ``values`` ``(n_terms, n_contrib)`` at flat bins into ``acc`` ``(n_terms, n_bins)``."""
    values = jnp.where(valid[None, :], values, 0.0)
    bins = jnp.where(valid, bins, 0)
    return acc + jax.ops.segment_sum(values.T, bins, num_segments=n_bins).T


def scatter_jack_vector(acc, values, region, bins, valid, n_jack: int, n_bins: int):
    """Add ``values`` ``(n_terms, n_contrib)`` at (region, bin) into ``acc`` ``(n_terms, n_jack, n_bins)``."""
    values = jnp.where(valid[None, :], values, 0.0)
    flat = jnp.where(valid, region * n_bins + bins, 0)
    upd = jax.ops.segment_sum(values.T, flat, num_segments=n_jack * n_bins)
    return acc + upd.T.reshape(values.shape[0], n_jack, n_bins)


# ----------------------------------------------------------------------------------------------
# The batch kernel
# ----------------------------------------------------------------------------------------------
def accumulate_batch(key, primary_idx, data: PassData, st: Static, acc: Accumulators) -> Accumulators:
    """Process one batch of primary particles ``primary_idx`` (shape ``(B,)``) of catalog ``st.legs[0]``:
    draw ``N2`` secondaries per primary, ``N3`` tertiaries per secondary and ``N4`` quaternaries per tertiary
    and add all contributions to ``acc``."""
    legs = st.legs
    gi, gj, gk, gl = (data.grids[c] for c in legs)
    B = primary_idx.shape[0]
    N2, N3, N4 = st.N2, st.N3, st.N4
    per = st.periodic
    k_off2, k_pick2, k_off3, k_pick3, k_off4, k_pick4 = jax.random.split(key, 6)
    n_bins = st.n_bins
    w_jk = data.jk_weights

    def xi(A, Bs, r, mu):
        return xi_lookup(data.xi_table, data.xi_pair_index[A, Bs], data.xi_s_max, data.xi_ds, st.xi_n_mu, r, mu)

    def bins(r, mu):
        return flat_bin(r, mu, data.s_edges, st.n_s, st.n_mu)

    def rrnorm(corr_a, corr_b, ba, bb):
        return data.rr[corr_a, ba] * data.rr[corr_b, bb]

    # ----- primary -----
    pos_i = gi.positions[primary_idx]  # (B, 3)
    cell_i = gi.cell3[primary_idx]
    w_i = gi.membership[:, primary_idx]  # (n_samples, B)
    jk_i = gi.jackknife[primary_idx]
    rc_i = gi.rand_class[primary_idx]

    # ----- secondary j (pairs) -----
    off2, p_off2 = draw_offsets(k_off2, data.kern_r2, st.r2_n_half, (B, N2))
    idx_j, inv_n_j, ok_j = draw_particle_in_cell(k_pick2, gj, data.n_side, per, cell_i[:, None, :] + off2)
    p_j = p_off2 * inv_n_j
    pos_j = gj.positions[idx_j]
    r_ij, mu_ij = separation(pos_i[:, None, :], pos_j, data.boxsize, per)
    bin_ij = bins(r_ij, mu_ij)
    ok_ij = ok_j & (bin_ij >= 0)
    if legs[0] == legs[1]:
        ok_ij &= idx_j != primary_idx[:, None]
    inv_p_ij = jnp.where(ok_ij, 1.0 / jnp.where(p_j > 0, p_j, 1.0), 0.0)  # (B, N2)
    w_j = gj.membership[:, idx_j]  # (n_samples, B, N2)
    jk_j = gj.jackknife[idx_j]
    rc_j = gj.rand_class[idx_j]

    pairs = ok_ij.reshape(-1)
    ba = jnp.where(pairs, bin_ij.reshape(-1), 0)
    inv_p2 = inv_p_ij.reshape(-1)
    wi = w_i[:, :, None]
    jki2 = jnp.broadcast_to(jk_i[:, None], (B, N2)).reshape(-1)
    jkj2 = jk_j.reshape(-1)

    # RR estimates (and EE/RR per region for the jackknife disconnected term)
    rr_new, ee_new, rrj_new = acc.rr, acc.ee, acc.rrj
    if st.rr_terms:
        ww_rr = jnp.stack([(wi[A] * w_j[Bs]).reshape(-1) * inv_p2 for A, Bs, _ in st.rr_terms])
        rr_new = scatter_vector(acc.rr, ww_rr, ba, pairs, n_bins)
        if st.jackknife:
            xi_rr = jnp.stack([xi(A, Bs, r_ij, mu_ij).reshape(-1) for A, Bs, _ in st.rr_terms])
            ee_vals = 0.5 * ww_rr * xi_rr
            rr_vals = 0.5 * ww_rr
            rci = jnp.broadcast_to(rc_i[:, None], (B, N2)).reshape(-1)
            rcj = rc_j.reshape(-1)
            for cls in (0, 1):
                sel = pairs & (rci == cls) & (rcj == cls)
                e, r_ = ee_new[:, cls], rrj_new[:, cls]
                for reg in (jki2, jkj2):
                    e = scatter_jack_vector(e, ee_vals, reg, ba, sel, st.n_jack, n_bins)
                    r_ = scatter_jack_vector(r_, rr_vals, reg, ba, sel, st.n_jack, n_bins)
                ee_new = ee_new.at[:, cls].set(e)
                rrj_new = rrj_new.at[:, cls].set(r_)

    # 2-point terms
    c2_new, c2j_new = acc.c2, acc.c2j
    if st.c2_terms:
        vals, jvals, cache = [], [], {}
        for A, Bs, C, D, ca, cb in st.c2_terms:
            if (A, Bs) not in cache:
                cache[(A, Bs)] = xi(A, Bs, r_ij, mu_ij).reshape(-1)
            ww = (wi[A] * w_j[Bs] * (wi[C] * w_j[D] + wi[D] * w_j[C])).reshape(-1)
            v = ww * (1.0 + cache[(A, Bs)]) * inv_p2 / rrnorm(ca, cb, ba, ba)
            vals.append(v)
            if st.jackknife:
                jvals.append(v * jackknife_weight_tensor(w_jk, jki2, jkj2, jki2, jkj2, ba, ba, ca, cb))
        c2_new = scatter_vector(acc.c2, jnp.stack(vals), ba, pairs, n_bins)
        if st.jackknife:
            c2j_new = scatter_vector(acc.c2j, jnp.stack(jvals), ba, pairs, n_bins)
    n2_new = scatter_counts(acc.n2, ba, ba, pairs, st)

    if not st.do_triples:
        return acc._replace(c2=c2_new, c2j=c2j_new, rr=rr_new, ee=ee_new, rrj=rrj_new, n2=n2_new,
                            n_draws=acc.n_draws + jnp.array([B * N2, 0, 0], dtype=jnp.float64))

    # ----- tertiary k (triples), drawn around i -----
    off3, p_off3 = draw_offsets(k_off3, data.kern_xi_ik, st.xi_n_half, (B, N2, N3))
    idx_k, inv_n_k, ok_k = draw_particle_in_cell(k_pick3, gk, data.n_side, per, cell_i[:, None, None, :] + off3)
    p_k = p_off3 * inv_n_k
    pos_k = gk.positions[idx_k]  # (B, N2, N3, 3)
    r_ik, mu_ik = separation(pos_i[:, None, None, :], pos_k, data.boxsize, per)
    r_jk, mu_jk = separation(pos_j[:, :, None, :], pos_k, data.boxsize, per)
    bin_jk = bins(r_jk, mu_jk)
    ok_ijk = ok_ij[:, :, None] & ok_k
    if legs[0] == legs[2]:
        ok_ijk &= idx_k != primary_idx[:, None, None]
    if legs[1] == legs[2]:
        ok_ijk &= idx_k != idx_j[:, :, None]
    ok_t3 = ok_ijk & (bin_jk >= 0)
    inv_p_ijk = jnp.where(ok_ijk, inv_p_ij[:, :, None] / jnp.where(p_k > 0, p_k, 1.0), 0.0)
    w_k = gk.membership[:, idx_k]  # (n_samples, B, N2, N3)
    jk_k = gk.jackknife[idx_k]

    t3_new, t3j_new = acc.t3, acc.t3j
    trip = ok_t3.reshape(-1)
    ba3 = jnp.where(trip, jnp.broadcast_to(bin_ij[:, :, None], bin_jk.shape).reshape(-1), 0)
    bb3 = jnp.where(trip, bin_jk.reshape(-1), 0)
    if st.t3_terms:
        inv_p3 = inv_p_ijk.reshape(-1)
        wi3 = w_i[:, :, None, None]
        wj3 = w_j[:, :, :, None]
        jki3 = jnp.broadcast_to(jk_i[:, None, None], bin_jk.shape).reshape(-1)
        jkj3 = jnp.broadcast_to(jk_j[:, :, None], bin_jk.shape).reshape(-1)
        jkk3 = jk_k.reshape(-1)
        vals, jvals, cache = [], [], {}
        for X, Y, Y2, Z, ca, cb in st.t3_terms:
            if (X, Z) not in cache:
                cache[(X, Z)] = xi(X, Z, r_ik, mu_ik).reshape(-1)
            ww = (wi3[X] * wj3[Y] * wj3[Y2] * w_k[Z]).reshape(-1)
            v = ww * cache[(X, Z)] * inv_p3 / rrnorm(ca, cb, ba3, bb3)
            vals.append(v)
            if st.jackknife:
                jvals.append(v * jackknife_weight_tensor(w_jk, jki3, jkj3, jkk3, jkj3, ba3, bb3, ca, cb))
        t3_new = scatter_matrix(acc.t3, jnp.stack(vals), ba3, bb3, trip, st, data.projection)
        if st.jackknife:
            t3j_new = scatter_matrix(acc.t3j, jnp.stack(jvals), ba3, bb3, trip, st, data.projection)
    n3_new = scatter_counts(acc.n3, ba3, bb3, trip, st)

    if not st.do_quads:
        return acc._replace(c2=c2_new, c2j=c2j_new, rr=rr_new, ee=ee_new, rrj=rrj_new, n2=n2_new,
                            t3=t3_new, t3j=t3j_new, n3=n3_new,
                            n_draws=acc.n_draws + jnp.array([B * N2, B * N2 * N3, 0], dtype=jnp.float64))

    # ----- quaternary l (quads), drawn around j -----
    cell_j = gj.cell3[idx_j]
    off4, p_off4 = draw_offsets(k_off4, data.kern_xi_jl, st.xi_n_half, (B, N2, N3, N4))
    idx_l, inv_n_l, ok_l = draw_particle_in_cell(k_pick4, gl, data.n_side, per, cell_j[:, :, None, None, :] + off4)
    p_l = p_off4 * inv_n_l
    pos_l = gl.positions[idx_l]  # (B, N2, N3, N4, 3)
    r_kl, mu_kl = separation(pos_k[:, :, :, None, :], pos_l, data.boxsize, per)
    bin_kl = bins(r_kl, mu_kl)
    ok_q = ok_ijk[..., None] & ok_l & (bin_kl >= 0)
    if legs[0] == legs[3]:
        ok_q &= idx_l != primary_idx[:, None, None, None]
    if legs[1] == legs[3]:
        ok_q &= idx_l != idx_j[:, :, None, None]
    if legs[2] == legs[3]:
        ok_q &= idx_l != idx_k[:, :, :, None]
    inv_p_q = jnp.where(ok_q, inv_p_ijk[..., None] / jnp.where(p_l > 0, p_l, 1.0), 0.0)
    w_l = gl.membership[:, idx_l]
    jk_l = gl.jackknife[idx_l]

    c4_new, c4j_new = acc.c4, acc.c4j
    quad = ok_q.reshape(-1)
    ba4 = jnp.where(quad, jnp.broadcast_to(bin_ij[:, :, None, None], bin_kl.shape).reshape(-1), 0)
    bb4 = jnp.where(quad, bin_kl.reshape(-1), 0)
    if st.c4_terms:
        r_jl, mu_jl = separation(pos_j[:, :, None, None, :], pos_l, data.boxsize, per)
        r_il, mu_il = separation(pos_i[:, None, None, None, :], pos_l, data.boxsize, per)
        inv_p4 = inv_p_q.reshape(-1)
        wi4 = w_i[:, :, None, None, None]
        wj4 = w_j[:, :, :, None, None]
        wk4 = w_k[:, :, :, :, None]
        jki4 = jnp.broadcast_to(jk_i[:, None, None, None], bin_kl.shape).reshape(-1)
        jkj4 = jnp.broadcast_to(jk_j[:, :, None, None], bin_kl.shape).reshape(-1)
        jkk4 = jnp.broadcast_to(jk_k[:, :, :, None], bin_kl.shape).reshape(-1)
        jkl4 = jk_l.reshape(-1)
        cache = {}

        def xi_c(tag, X, Y, r, mu):
            if (tag, X, Y) not in cache:
                cache[(tag, X, Y)] = xi(X, Y, r, mu)
            return cache[(tag, X, Y)]

        vals, jvals = [], []
        for A, Bs, C, D, ca, cb in st.c4_terms:
            term1 = xi_c("ik", A, C, r_ik, mu_ik)[..., None] * xi_c("jl", Bs, D, r_jl, mu_jl)
            term2 = xi_c("il", A, D, r_il, mu_il) * xi_c("jk", Bs, C, r_jk, mu_jk)[..., None]
            ww = wi4[A] * wj4[Bs] * wk4[C] * w_l[D]
            v = (ww * (term1 + term2)).reshape(-1) * inv_p4 / rrnorm(ca, cb, ba4, bb4)
            vals.append(v)
            if st.jackknife:
                jvals.append(v * jackknife_weight_tensor(w_jk, jki4, jkj4, jkk4, jkl4, ba4, bb4, ca, cb))
        c4_new = scatter_matrix(acc.c4, jnp.stack(vals), ba4, bb4, quad, st, data.projection)
        if st.jackknife:
            c4j_new = scatter_matrix(acc.c4j, jnp.stack(jvals), ba4, bb4, quad, st, data.projection)
    n4_new = scatter_counts(acc.n4, ba4, bb4, quad, st)

    return Accumulators(c4_new, t3_new, c2_new, rr_new, c4j_new, t3j_new, c2j_new, ee_new, rrj_new, n4_new, n3_new, n2_new,
                        acc.n_draws + jnp.array([B * N2, B * N2 * N3, B * N2 * N3 * N4], dtype=jnp.float64))


accumulate_batch_jit = jax.jit(accumulate_batch, static_argnames=("st",))


# ----------------------------------------------------------------------------------------------
# Pair sampler for the correlation function refinement (xi-kernel draws, as RascalC's rescale_correlation)
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RefineStatic:
    N2: int
    same_catalog: bool
    periodic: bool
    n_s: int
    n_mu: int
    n_half: int
    xi_n_mu: int


def binned_xi_batch(key, primary_idx, grid_i: GridArrays, grid_j: GridArrays, n_side, boxsize, kern: Kernel,
                    table, s_max, ds, w_i_all, w_j_all, s_edges, rs: RefineStatic):
    """``sum w_i w_j xi(ij) / p``, ``sum w_i w_j / p`` and the number of accepted pairs in the (s, mu) bins of
    the xi table, for primaries ``primary_idx`` with ``N2`` xi-kernel draws each."""
    k_off, k_pick = jax.random.split(key)
    B = primary_idx.shape[0]
    pos_i = grid_i.positions[primary_idx]
    cell_i = grid_i.cell3[primary_idx]
    off, p_off = draw_offsets(k_off, kern, rs.n_half, (B, rs.N2))
    idx_j, inv_n_j, ok_j = draw_particle_in_cell(k_pick, grid_j, n_side, rs.periodic, cell_i[:, None, :] + off)
    p_j = p_off * inv_n_j
    pos_j = grid_j.positions[idx_j]
    r, mu = separation(pos_i[:, None, :], pos_j, boxsize, rs.periodic)
    i_s = jnp.searchsorted(s_edges, r, side="right") - 1
    ok = ok_j & (i_s >= 0) & (i_s < rs.n_s)
    if rs.same_catalog:
        ok &= idx_j != primary_idx[:, None]
    i_mu = jnp.clip(jnp.floor(mu * rs.n_mu).astype(jnp.int64), 0, rs.n_mu - 1)
    flat = jnp.where(ok, i_s * rs.n_mu + i_mu, 0).reshape(-1)
    ww = (w_i_all[primary_idx][:, None] * w_j_all[idx_j]).reshape(-1)
    inv_p = jnp.where(ok, 1.0 / jnp.where(p_j > 0, p_j, 1.0), 0.0).reshape(-1)
    xi = xi_lookup(table, 0, s_max, ds, rs.xi_n_mu, r, mu).reshape(-1)
    okf = ok.reshape(-1)
    rr = jnp.where(okf, ww * inv_p, 0.0)
    n = rs.n_s * rs.n_mu
    return (jax.ops.segment_sum(rr * xi, flat, num_segments=n), jax.ops.segment_sum(rr, flat, num_segments=n),
            jax.ops.segment_sum(okf.astype(jnp.float64), flat, num_segments=n))


binned_xi_batch_jit = jax.jit(binned_xi_batch, static_argnames=("rs",))
