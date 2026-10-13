"""Orchestration: build grids, kernels and tables, run the sampling passes and assemble the raw covariance terms.

The main entry point is :func:`run_cov`; the result is a :class:`RawCovariance` holding the Monte Carlo
estimates of the 2-, 3- and 4-point terms for every pair of requested correlation functions, split into
independent subsamples, plus everything needed by :mod:`rascaljax.postprocess`.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .binning import Binning
from .catalog import Catalog, Sample
from .grid import CellGrid, Geometry, choose_n_side, make_geometry
from .integrals import (Accumulators, GridArrays, Kernel, PassData, RefineStatic, Static, accumulate_batch_jit,
                        accumulators_to_numpy, add_accumulators, binned_xi_batch_jit, zero_accumulators)
from .kernels import OffsetSampler, r2_cell_probabilities, xi_cell_probabilities
from .utils import Array, Logger, symmetrized
from .xi_table import XiTable


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
@dataclass
class CorrelationInput:
    """Inputs describing one measured correlation function (pair of samples).

    Parameters
    ----------
    s_edges : (n_s + 1,) separation bin edges of the covariance.
    rr : (n_s, n_mu) normalized random pair counts in the covariance bins (mu in [0, 1]).
    xi_table : :class:`XiTable` used in the integrals (usually finer binned than the covariance).
    jack_xi : optional (n_jack, n_s * n_mu) jackknife correlation function estimates.
    jack_rr : optional (n_jack, n_s * n_mu) jackknife random pair counts (RascalC convention:
        ``RR_A = RR_{A x all} + ...``, i.e. they sum to ``rr`` over regions).
    jack_labels : optional (n_jack,) region labels matching the catalogs' ``jackknife`` labels.
    """

    s_edges: Array
    rr: Array
    xi_table: XiTable
    jack_xi: Array | None = None
    jack_rr: Array | None = None
    jack_labels: Array | None = None

    def __post_init__(self):
        self.s_edges = np.asarray(self.s_edges, dtype=float).ravel()
        self.rr = np.asarray(self.rr, dtype=float)
        if self.rr.ndim != 2 or self.rr.shape[0] != len(self.s_edges) - 1:
            raise ValueError("rr must be a 2D (n_s, n_mu) array matching s_edges")
        if np.any(self.rr <= 0):
            raise ValueError("All RR counts must be positive")
        if self.jack_xi is not None:
            self.jack_xi = np.asarray(self.jack_xi, dtype=float)
            self.jack_rr = np.asarray(self.jack_rr, dtype=float)
            self.jack_labels = np.asarray(self.jack_labels)
            if self.jack_xi.shape != self.jack_rr.shape or self.jack_xi.shape[1] != self.rr.size:
                raise ValueError("jack_xi and jack_rr must have shape (n_jack, n_s * n_mu)")
            if len(self.jack_labels) != len(self.jack_xi):
                raise ValueError("jack_labels must have one entry per jackknife realization")

    @property
    def jackknife(self) -> bool:
        return self.jack_xi is not None

    @property
    def jack_weights(self) -> Array:
        """``w_aA = RR_aA / sum_A RR_aA``."""
        return self.jack_rr / np.sum(self.jack_rr, axis=0)[None, :]


@dataclass
class RawCovariance:
    """Raw (unrescaled) covariance terms.

    Arrays are indexed ``[i_sample, i_corr1, i_corr2, bin1, bin2]`` for subsamples; ``*_full`` versions are
    the averages over subsamples. ``c3`` has an extra leading axis over the catalog of the shared particle and
    ``c2`` an extra leading axis over catalog pairs (``cat_pairs``), so that shot-noise rescalings per catalog
    can be applied in post-processing. Output bins are ``(s, ell)`` for Legendre or ``(s, mu)`` otherwise.
    """

    binning: Binning
    sample_names: list[str]
    catalog_names: list[str]
    sample_catalog: Array  # (n_samples,) catalog index of each sample
    correlations: list[tuple[int, int]]  # sample index pairs
    cat_pairs: list[tuple[int, int]]  # catalog pairs for the c2 axis
    c4: Array  # (n_sub, n_corr, n_corr, n_out, n_out)
    c3: Array  # (n_cat, n_sub, n_corr, n_corr, n_out, n_out)
    c2: Array  # (n_catpair, n_sub, n_corr, n_corr, n_out, n_out)
    c4j: Array | None = None
    c3j: Array | None = None
    c2j: Array | None = None
    ee: Array | None = None  # (n_sub, n_corr, 2, n_jack, n_bins)
    rrj: Array | None = None  # (n_sub, n_corr, 2, n_jack, n_bins)
    rr_estimate: Array | None = None  # (n_sub, n_corr, n_bins)
    rr_input: Array | None = None  # (n_corr, n_bins)
    jack_weights: Array | None = None  # (n_corr, n_jack, n_bins)
    jack_xi: Array | None = None  # (n_corr, n_jack, n_bins)
    jack_labels: Array | None = None
    counts: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    # ----- convenience -----
    @property
    def n_corr(self) -> int:
        return len(self.correlations)

    @property
    def n_sub(self) -> int:
        return self.c4.shape[0]

    @property
    def jackknife(self) -> bool:
        return self.c4j is not None

    def full(self, name: str) -> Array:
        return np.mean(getattr(self, name), axis=-5)

    def save(self, filename: str) -> None:
        d = dict(s_edges=self.binning.s_edges, n_mu=self.binning.n_mu, max_l=-1 if self.binning.max_l is None else self.binning.max_l,
                 sample_names=np.array(self.sample_names), catalog_names=np.array(self.catalog_names),
                 sample_catalog=self.sample_catalog, correlations=np.array(self.correlations), cat_pairs=np.array(self.cat_pairs),
                 c4=self.c4, c3=self.c3, c2=self.c2)
        for k in ("c4j", "c3j", "c2j", "ee", "rrj", "rr_estimate", "rr_input", "jack_weights", "jack_xi", "jack_labels"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        for k, v in self.counts.items():
            d["counts_" + k] = v
        for k, v in self.info.items():
            d["info_" + k] = np.array(v)
        np.savez_compressed(filename, **d)

    @classmethod
    def load(cls, filename: str) -> "RawCovariance":
        with np.load(filename, allow_pickle=False) as f:
            d = dict(f)
        max_l = int(d["max_l"])
        binning = Binning(d["s_edges"], int(d["n_mu"]), None if max_l < 0 else max_l)
        kw = {k: d.get(k) for k in ("c4j", "c3j", "c2j", "ee", "rrj", "rr_estimate", "rr_input", "jack_weights", "jack_xi", "jack_labels")}
        counts = {k[7:]: v for k, v in d.items() if k.startswith("counts_")}
        info = {k[5:]: v.item() if v.ndim == 0 else v for k, v in d.items() if k.startswith("info_")}
        return cls(binning, list(d["sample_names"]), list(d["catalog_names"]), d["sample_catalog"],
                   [tuple(int(x) for x in c) for c in d["correlations"]], [tuple(int(x) for x in c) for c in d["cat_pairs"]],
                   d["c4"], d["c3"], d["c2"], counts=counts, info=info, **kw)


# ----------------------------------------------------------------------------------------------
# Pass planning
# ----------------------------------------------------------------------------------------------
@dataclass
class PassPlan:
    legs: tuple[int, int, int, int]
    c4_terms: list  # (A, B, C, D, corr_AB, corr_CD)
    t3_terms: list  # (X, Y, Y', Z, corr_XY, corr_ZY')
    c2_terms: list  # (A, B, C, D, corr_AB, corr_CD)
    rr_terms: list  # (A, B, corr_AB)
    do_triples: bool = True
    do_quads: bool = True

    def add_unique(self, name: str, term: tuple) -> int:
        lst = getattr(self, name)
        if term not in lst:
            lst.append(term)
        return lst.index(term)


def plan_passes(sample_catalog: Sequence[int], correlations: Sequence[tuple[int, int]]):
    """Decide which sampling passes (catalog per leg) are needed, which terms each accumulates, and how
    each covariance block is assembled from them.

    Returns ``(passes, assembly)`` where ``assembly[(i_corr1, i_corr2)]`` is a dict with:
    ``"c4": list of (i_pass, i_term)`` (summed), ``"c3": list of (i_pass, i_term, transpose, shared_catalog)``,
    ``"c2": (i_pass, i_term, catalog_pair) or None``.
    """
    cat = list(sample_catalog)
    corr_index = {tuple(c): i for i, c in enumerate(correlations)}

    def cidx(a, b):
        return corr_index[(a, b)] if (a, b) in corr_index else corr_index[(b, a)]

    passes: list[PassPlan] = []

    def get_pass(legs, need_quads=False, need_triples=False):
        for p in passes:
            if p.legs == legs:
                p.do_quads |= need_quads
                p.do_triples |= need_triples or need_quads
                return p
        p = PassPlan(legs, [], [], [], [], need_triples or need_quads, need_quads)
        passes.append(p)
        return p

    def find_pass_prefix(prefix, n):
        for p in passes:
            if p.legs[:n] == prefix and (n < 3 or p.do_triples):
                return p
        return None

    assembly = {}
    n_corr = len(correlations)
    # 4-point passes first (they also serve the pair and triple terms). Both Wick contractions of a block come
    # from kernel-matched terms with the first pair in both orders: (A, B, C, D) gives xi_AC(ik) xi_BD(jl) and
    # (B, A, C, D) gives xi_BC(ik) xi_AD(jl); for A == B the same term is simply counted twice.
    for i1 in range(n_corr):
        for i2 in range(i1, n_corr):
            A, B = correlations[i1]
            C, D = correlations[i2]
            providers = []
            for X, Y in ((A, B), (B, A)):
                p = get_pass((cat[X], cat[Y], cat[C], cat[D]), need_quads=True)
                it = p.add_unique("c4_terms", (X, Y, C, D, i1, i2))
                providers.append((passes.index(p), it))
            assembly[(i1, i2)] = {"c4": providers, "c3": [], "c2": None}
    # 3-point tensors T[X, (Y, Y'), Z] with cat(Y) == cat(Y')
    for i1 in range(n_corr):
        for i2 in range(i1, n_corr):
            A, B = correlations[i1]
            C, D = correlations[i2]
            for X, Y, Y2, Z in ((A, B, D, C), (A, B, C, D), (B, A, D, C), (B, A, C, D)):
                if cat[Y] != cat[Y2]:
                    continue
                prefix = (cat[X], cat[Y], cat[Z])
                p = find_pass_prefix(prefix, 3)
                transpose = False
                if p is None:
                    p = find_pass_prefix((cat[Z], cat[Y], cat[X]), 3)
                    transpose = p is not None
                if p is None:
                    p = get_pass((cat[X], cat[Y], cat[Z], cat[Z]), need_triples=True)
                if transpose:
                    it = p.add_unique("t3_terms", (Z, Y2, Y, X, cidx(Z, Y2), cidx(X, Y)))
                else:
                    it = p.add_unique("t3_terms", (X, Y, Y2, Z, cidx(X, Y), cidx(Z, Y2)))
                assembly[(i1, i2)]["c3"].append((passes.index(p), it, transpose, cat[Y]))
    # 2-point terms and RR estimates
    for i1 in range(n_corr):
        for i2 in range(i1, n_corr):
            A, B = correlations[i1]
            C, D = correlations[i2]
            if {cat[A], cat[B]} != {cat[C], cat[D]} and (cat[A], cat[B]) != (cat[C], cat[D]):
                continue
            if sorted((cat[A], cat[B])) != sorted((cat[C], cat[D])):
                continue
            p = find_pass_prefix((cat[A], cat[B]), 2) or find_pass_prefix((cat[B], cat[A]), 2)
            if p is None:
                p = get_pass((cat[A], cat[B], cat[B], cat[B]))
            if p.legs[:2] == (cat[A], cat[B]):
                it = p.add_unique("c2_terms", (A, B, C, D, i1, i2))
            else:
                it = p.add_unique("c2_terms", (B, A, D, C, i1, i2))
            assembly[(i1, i2)]["c2"] = (passes.index(p), it, tuple(sorted((cat[A], cat[B]))))
    for ic, (A, B) in enumerate(correlations):
        p = find_pass_prefix((cat[A], cat[B]), 2)
        if p is not None:
            p.add_unique("rr_terms", (A, B, ic))
        else:
            p = find_pass_prefix((cat[B], cat[A]), 2) or get_pass((cat[A], cat[B], cat[B], cat[B]))
            if p.legs[:2] == (cat[A], cat[B]):
                p.add_unique("rr_terms", (A, B, ic))
            else:
                p.add_unique("rr_terms", (B, A, ic))
    return passes, assembly


# ----------------------------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------------------------
def _kernel_xi_iso(tables: list[XiTable]):
    """Isotropic xi used for the sampling kernel of a catalog pair: the maximum over the given tables."""

    def f(r):
        return np.max(np.stack([t.isotropic(r) for t in tables]), axis=0)

    return f


def _to_kernel(cube: Array) -> tuple[Kernel, int]:
    s = OffsetSampler.from_cube(cube)
    return Kernel(jnp.asarray(s.prob), jnp.asarray(s.threshold), jnp.asarray(s.alias, dtype=jnp.int32)), s.n_half


def unique_catalogs(samples: list[Sample]) -> list[Catalog]:
    cats = []
    for s in samples:
        if not any(s.catalog is c for c in cats):
            cats.append(s.catalog)
    return cats


def refine_xi_table(table: XiTable, grid_i: GridArrays, grid_j: GridArrays, same_catalog: bool, n_side, boxsize, periodic: bool,
                    kern: Kernel, n_half: int, w_i: Array, w_j: Array, n_iter: int, draws: int, batch: int, key, log,
                    xi_lookup_ds: float, xi_lookup_n_mu: int) -> XiTable:
    """RascalC-style refinement: rescale the table until the bin averages of the interpolant match the input bin values."""
    if table.s_edges is None or n_iter <= 0:
        return table
    target = table.xi.copy()
    s_c = 0.5 * (table.s_edges[1:] + table.s_edges[:-1])
    mu_c = table.mu_vals
    n_s, n_mu = table.xi.shape
    rs = RefineStatic(draws, same_catalog, periodic, n_s, n_mu, n_half, xi_lookup_n_mu)
    s_edges = jnp.asarray(table.s_edges)
    n = int(grid_i.positions.shape[0])
    current = table
    for it in range(n_iter):
        lk = current.lookup(xi_lookup_ds, xi_lookup_n_mu)
        tab = jnp.asarray(lk.table)[None]
        xs = np.zeros(n_s * n_mu)
        rr = np.zeros(n_s * n_mu)
        cs = np.zeros(n_s * n_mu)
        for ib, start in enumerate(range(0, n, batch)):
            idx = jnp.arange(start, min(start + batch, n))
            k = jax.random.fold_in(jax.random.fold_in(key, it), ib)
            a, b, c = binned_xi_batch_jit(k, idx, grid_i, grid_j, n_side, boxsize, kern, tab, jnp.asarray(lk.s_max), jnp.asarray(lk.ds),
                                          w_i, w_j, s_edges, rs)
            xs += np.asarray(a)
            rr += np.asarray(b)
            cs += np.asarray(c)
        if np.any(cs == 0) or np.any(rr == 0):
            bad = np.argwhere((cs == 0) | (rr == 0)).ravel()
            raise RuntimeError(f"No pairs sampled in xi bins {bad.tolist()} during refinement; increase refine_draws, coarsen the xi binning or provide denser randoms")
        est = (xs / rr).reshape(n_s, n_mu)
        old = current(s_c[:, None], mu_c[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(est != 0, target / est, 1.0)
        new_xi = old * ratio
        current = XiTable(s_c, mu_c, new_xi, table.s_edges)
        scale = np.max(np.abs(target))
        log(f"  refinement iteration {it + 1}/{n_iter}: max |est - target| / max |target| = {np.max(np.abs(est - target)) / scale:.3e}, min pairs per bin = {int(cs.min())}")
    return current


def run_cov(samples: list[Sample], correlations: dict[tuple[int, int], CorrelationInput], *,
            mode: str = "legendre", max_l: int = 4, n_mu: int | None = None,
            N2: int = 5, N3: int = 10, N4: int = 20, n_loops: int = 32, loops_per_sample: int = 2,
            boxsize: float | None = None, xi_cut: float = 250.0, n_side: int = 301,
            refine_iterations: int = 10, refine_draws: int = 20,
            quads_per_batch: int = 2_000_000, seed: int = 0, positions_float32: bool = False,
            xi_lookup_ds: float = 0.25, xi_lookup_n_mu: int = 201,
            devices: list | None = None,
            out_dir: str | None = None, log=None) -> RawCovariance:
    """Compute the raw covariance matrix terms.

    Parameters
    ----------
    samples : list of :class:`Sample`
        The samples (membership-weighted views of random catalogs) whose correlation functions are covered.
    correlations : dict ``{(i_sample, j_sample): CorrelationInput}``
        The correlation functions to compute the covariance of (keys with ``i <= j``). All must share the
        separation binning and the number of mu bins.
    mode : ``"legendre"`` (covariance of even multipoles projected from mu bins, as pycorr) or ``"s_mu"``.
    max_l : maximum even multipole for ``mode="legendre"``.
    n_mu : number of mu bins of the pair counts in [0, 1]; inferred from the inputs if None.
    N2, N3, N4 : draws per primary particle, per secondary and per tertiary.
    n_loops, loops_per_sample : number of loops over all primaries, and loops merged into one output subsample.
    boxsize : periodic cubic box side (None for aperiodic / survey geometry).
    xi_cut : separation beyond which the correlation function is not sampled (the ``xi``-kernel range).
    n_side : initial guess of the grid size (adjusted to 2-16 particles per filled cell).
    refine_iterations, refine_draws : xi table refinement (RascalC's rescaling), draws per primary per iteration.
    quads_per_batch : controls the batch size (device memory).
    seed : RNG seed (results are reproducible with the same inputs, seed and batch size).
    positions_float32 : store particle positions and compute separations in float32 (faster, memory-bound kernel);
        separations of survey-size coordinates are then accurate to ~1e-5 relative, far below the bin widths.
        All weights, probabilities and accumulations stay in float64. Default False.
    devices : JAX devices to use (default: all available); subsamples are distributed over them.
    out_dir : if given, the raw covariance is saved there as ``Raw_Covariance.npz`` with a log file.
    """
    t_start = time.time()
    if log is None:
        log = Logger(None if out_dir is None else os.path.join(out_dir, "log.txt"))
    devices = list(jax.devices()) if devices is None else list(devices)
    n_dev = len(devices)
    if mode not in ("legendre", "s_mu"):
        raise ValueError("mode must be 'legendre' or 's_mu'")
    if n_loops % loops_per_sample:
        raise ValueError("loops_per_sample must divide n_loops")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # ----- samples, catalogs, correlations -----
    catalogs = unique_catalogs(samples)
    cat_names = [c.name for c in catalogs]
    sample_catalog = np.array([next(i for i, c in enumerate(catalogs) if s.catalog is c) for s in samples], dtype=int)
    corr_keys = sorted(correlations.keys())
    for a, b in corr_keys:
        if not (0 <= a <= b < len(samples)):
            raise ValueError(f"Invalid correlation key {(a, b)}; use (i, j) with i <= j sample indices")
    corr_inputs = [correlations[k] for k in corr_keys]
    n_corr = len(corr_keys)
    rr0 = corr_inputs[0].rr
    if n_mu is None:
        n_mu = rr0.shape[1]
    for ci in corr_inputs:
        if ci.rr.shape != rr0.shape:
            raise ValueError("All correlations must have the same (n_s, n_mu) pair-count binning")
    for ci in corr_inputs:
        if not np.allclose(ci.s_edges, corr_inputs[0].s_edges):
            raise ValueError("All correlations must have the same separation bin edges")
    binning = Binning(corr_inputs[0].s_edges, n_mu, max_l if mode == "legendre" else None)
    n_bins = binning.n_bins
    jackknife = all(ci.jackknife for ci in corr_inputs)
    if any(ci.jackknife for ci in corr_inputs) and not jackknife:
        raise ValueError("Either all or none of the correlations must have jackknife inputs")
    log(f"RascalJax: {len(samples)} sample(s) on {len(catalogs)} catalog(s), {n_corr} correlation(s), mode={mode}"
        + (f", max_l={max_l}" if mode == "legendre" else "") + f", jackknife={jackknife}, periodic={bool(boxsize)}")
    log(f"Binning: {binning.n_s} s bins from {binning.s_min:g} to {binning.s_max:g}, {n_mu} mu bins; N2={N2}, N3={N3}, N4={N4}, n_loops={n_loops}, loops_per_sample={loops_per_sample}")
    log(f"Devices: {n_dev} ({devices[0].platform})")

    # ----- jackknife labels -----
    jack_labels = None
    n_jack = 1
    if jackknife:
        jack_labels = np.unique(np.concatenate([np.unique(c.jackknife) for c in catalogs if c.jackknife is not None]))
        if any(c.jackknife is None for c in catalogs):
            raise ValueError("Jackknife mode requires jackknife labels for every catalog")
        for ci in corr_inputs:
            if not np.array_equal(np.sort(ci.jack_labels), jack_labels):
                raise ValueError("Jackknife labels of the pair counts must match the union of catalog labels")
        n_jack = len(jack_labels)
        for c in catalogs:
            c.assign_random_classes(seed)

    # ----- grids -----
    n_side_final = choose_n_side(catalogs, boxsize, binning.s_max, n_side, log=log)
    geom = make_geometry(catalogs, n_side_final, boxsize, binning.s_max)
    log(f"Grid: n_side={geom.n_side.tolist()}, cellsize={geom.cellsize:.3f}, box={geom.boxsize.round(2).tolist()}")
    grids = [CellGrid.from_catalog(c, geom, jack_labels) for c in catalogs]
    for c, g in zip(catalogs, grids):
        log(f"  {c.name}: {g.size} particles, {g.n_filled} filled cells, {g.mean_density:.2f} per filled cell, max {g.max_per_cell}")
    grid_arrays = []
    for ic, g in enumerate(grids):
        memb = np.zeros((len(samples), g.size))
        for i, s in enumerate(samples):
            if s.catalog is catalogs[ic]:
                memb[i] = g.sort_membership(s.effective_weights)
        cell3 = geom.cell_3d(g.positions).astype(np.int32)
        pos_dtype = jnp.float32 if positions_float32 else jnp.float64
        grid_arrays.append(GridArrays(jnp.asarray(g.positions, dtype=pos_dtype), jnp.asarray(g.jackknife, dtype=jnp.int32), jnp.asarray(g.rand_class, dtype=jnp.int32),
                                      jnp.asarray(cell3), jnp.asarray(g.cell_start), jnp.asarray(g.cell_count), jnp.asarray(memb)))
    n_side_arr = jnp.asarray(geom.n_side)
    boxsize_arr = jnp.asarray(geom.boxsize)
    periodic = geom.periodic
    if periodic and 2 * (binning.s_max + xi_cut) > boxsize:
        log("WARNING: 2 * (s_max + xi_cut) exceeds the box size; some 4-point configurations may be mis-wrapped")

    # ----- kernels -----
    r2_n_half = int(np.ceil(2 * binning.s_max / geom.cellsize))
    xi_n_half = int(np.ceil(xi_cut / geom.cellsize))
    kern_r2, _ = _to_kernel(r2_cell_probabilities(r2_n_half, geom.cellsize))
    log(f"Kernels: 1/r^2 cube half-side {r2_n_half}, xi cube half-side {xi_n_half}")

    def tables_for_catpair(a, b):
        return [ci.xi_table for (sa, sb), ci in zip(corr_keys, corr_inputs) if {sample_catalog[sa], sample_catalog[sb]} == {a, b}]

    xi_kernels = {}

    def xi_kernel(a, b):
        key = (min(a, b), max(a, b))
        if key not in xi_kernels:
            tabs = tables_for_catpair(*key)
            if not tabs:
                tabs = [ci.xi_table for ci in corr_inputs]
            xi_kernels[key], _ = _to_kernel(xi_cell_probabilities(xi_n_half, geom.cellsize, _kernel_xi_iso(tabs)))
        return xi_kernels[key]

    # ----- xi tables: refinement and lookups -----
    key0 = jax.random.PRNGKey(seed)
    refined = []
    for ic, ((sa, sb), ci) in enumerate(zip(corr_keys, corr_inputs)):
        ca, cb = int(sample_catalog[sa]), int(sample_catalog[sb])
        if ci.xi_table.s_edges is not None and refine_iterations > 0:
            log(f"Refining xi table for correlation {ic} (samples {sa}, {sb})")
            kern = xi_kernel(ca, cb)
            t = refine_xi_table(ci.xi_table, grid_arrays[ca], grid_arrays[cb], ca == cb, n_side_arr, boxsize_arr, periodic, kern, xi_n_half,
                                grid_arrays[ca].membership[sa], grid_arrays[cb].membership[sb], refine_iterations, refine_draws,
                                max(1, quads_per_batch // refine_draws), jax.random.fold_in(key0, 1000 + ic), log, xi_lookup_ds, xi_lookup_n_mu)
            if out_dir is not None:
                t.write(os.path.join(out_dir, f"xi_refined_{sa}{sb}.txt"))
        else:
            t = ci.xi_table
        refined.append(t)
    s_max_lookup = max(t.s_max for t in refined)
    lookups = []
    for t in refined:
        n_grid = int(np.ceil(s_max_lookup / xi_lookup_ds)) + 1
        ds = s_max_lookup / (n_grid - 1)
        s_grid = np.arange(n_grid) * ds
        mu_grid = np.linspace(0, 1, xi_lookup_n_mu)
        lookups.append(t(s_grid[:, None], mu_grid[None, :]))
    xi_table_arr = jnp.asarray(np.stack(lookups))
    pair_index = -np.ones((len(samples), len(samples)), dtype=np.int32)
    for ic, (sa, sb) in enumerate(corr_keys):
        pair_index[sa, sb] = pair_index[sb, sa] = ic
    if np.any(pair_index < 0):
        missing = [(a, b) for a in range(len(samples)) for b in range(a, len(samples)) if pair_index[a, b] < 0]
        raise ValueError(f"Correlation functions are needed for all sample pairs; missing {missing}")

    # ----- normalizing counts and jackknife weights -----
    rr_arr = jnp.asarray(np.stack([ci.rr.ravel() for ci in corr_inputs]))
    if jackknife:
        jw = []
        for ci in corr_inputs:
            order = np.argsort(ci.jack_labels)
            jw.append(ci.jack_weights[order])
        jk_weights = jnp.asarray(np.stack(jw))
    else:
        jk_weights = jnp.zeros((n_corr, 1, n_bins))
    projection = jnp.asarray(binning.projection_matrix() if binning.legendre else np.zeros((1, 1)))

    # ----- passes -----
    passes, assembly = plan_passes([int(c) for c in sample_catalog], corr_keys)
    log(f"Planned {len(passes)} sampling pass(es): " + ", ".join(f"legs={p.legs} quads={p.do_quads} terms(c4,t3,c2,rr)=({len(p.c4_terms)},{len(p.t3_terms)},{len(p.c2_terms)},{len(p.rr_terms)})" for p in passes))
    gpr = np.array([c.galaxies_per_random for c in catalogs])
    n_sub = n_loops // loops_per_sample
    n_out = binning.n_out
    pass_results = []  # per pass: list over subsamples of normalized accumulator dicts
    n_ell = binning.n_ell if binning.legendre else 0

    for ip, p in enumerate(passes):
        st = Static(N2, N3, N4, p.legs, periodic, binning.n_s, n_mu, n_ell, n_jack, jackknife, p.do_triples, p.do_quads,
                    xi_n_half, r2_n_half, xi_lookup_n_mu, tuple(p.c4_terms), tuple(p.t3_terms), tuple(p.c2_terms), tuple(p.rr_terms))
        data = PassData(tuple(grid_arrays), n_side_arr, boxsize_arr, kern_r2, xi_kernel(p.legs[0], p.legs[2]), xi_kernel(p.legs[1], p.legs[3]),
                        xi_table_arr, jnp.asarray(pair_index), jnp.asarray(s_max_lookup), jnp.asarray(ds), jnp.asarray(binning.s_edges),
                        projection, rr_arr, jk_weights)
        n_prim = grids[p.legs[0]].size
        per_prim = N2 * (N3 if p.do_triples else 1) * (N4 if p.do_quads else 1)
        batch = max(1, min(n_prim, quads_per_batch // per_prim))
        log(f"Pass {ip + 1}/{len(passes)}: legs={p.legs}, {n_prim} primaries, batch {batch} ({per_prim} draws per primary), {n_loops} loops")
        key_pass = jax.random.fold_in(key0, ip)
        # subsamples are distributed round-robin over the devices, each driven by its own host thread (JAX dispatch
        # is thread-safe), so the devices run concurrently without the host loop interleaving their batches
        data_dev = [jax.device_put(data, dev) for dev in devices]
        subsamples = [None] * n_sub
        t0 = time.time()
        done = [0]
        lock = threading.Lock()

        def worker(dev_i):
            dev = devices[dev_i]
            for s_idx in range(dev_i, n_sub, n_dev):
                acc = jax.device_put(zero_accumulators(st), dev)
                for il in range(loops_per_sample):
                    key_loop = jax.random.fold_in(key_pass, s_idx * loops_per_sample + il)
                    for ib, start in enumerate(range(0, n_prim, batch)):
                        idx = jax.device_put(jnp.arange(start, min(start + batch, n_prim)), dev)
                        acc = accumulate_batch_jit(jax.device_put(jax.random.fold_in(key_loop, ib), dev), idx, data_dev[dev_i], st, acc)
                result = _normalize(accumulators_to_numpy(acc), p, gpr, n_prim)
                with lock:
                    subsamples[s_idx] = result
                    done[0] += loops_per_sample
                    el = time.time() - t0
                    log(f"  loop {done[0]}/{n_loops} done ({el:.1f} s elapsed, ~{el / done[0] * (n_loops - done[0]):.1f} s left)")

        with ThreadPoolExecutor(max_workers=n_dev) as pool:
            futures = [pool.submit(worker, d) for d in range(min(n_dev, n_sub))]
            for f in futures:
                f.result()
        el = time.time() - t0
        n_q = n_prim * n_loops * per_prim
        log(f"  pass done in {el:.1f} s: {n_q:.3e} draws, {n_q / el:.3e} draws/s")
        pass_results.append(subsamples)

    # ----- assemble blocks -----
    P_out = binning.projection_matrix()  # identity for (s, mu) output
    cat_pairs = sorted({tuple(sorted((int(sample_catalog[a]), int(sample_catalog[b])))) for a, b in corr_keys})
    n_cat = len(catalogs)
    shape = (n_sub, n_corr, n_corr, n_out, n_out)
    c4 = np.zeros(shape)
    c3 = np.zeros((n_cat,) + shape)
    c2 = np.zeros((len(cat_pairs),) + shape)
    c4j = np.zeros(shape) if jackknife else None
    c3j = np.zeros((n_cat,) + shape) if jackknife else None
    c2j = np.zeros((len(cat_pairs),) + shape) if jackknife else None
    for (i1, i2), asm in assembly.items():
        for ip, it in asm["c4"]:
            for s in range(n_sub):
                blk = pass_results[ip][s]["c4"][it]
                c4[s, i1, i2] += blk
                if i1 != i2:
                    c4[s, i2, i1] += blk.T
                if jackknife:
                    blkj = pass_results[ip][s]["c4j"][it]
                    c4j[s, i1, i2] += blkj
                    if i1 != i2:
                        c4j[s, i2, i1] += blkj.T
        for ip, it, tr, shared in asm["c3"]:
            for s in range(n_sub):
                blk = pass_results[ip][s]["t3"][it]
                blk = blk.T if tr else blk
                c3[shared, s, i1, i2] += blk
                if i1 != i2:
                    c3[shared, s, i2, i1] += blk.T
                if jackknife:
                    blkj = pass_results[ip][s]["t3j"][it]
                    blkj = blkj.T if tr else blkj
                    c3j[shared, s, i1, i2] += blkj
                    if i1 != i2:
                        c3j[shared, s, i2, i1] += blkj.T
        if asm["c2"] is not None:
            ip, it, cp = asm["c2"]
            icp = cat_pairs.index(cp)
            for s in range(n_sub):
                blk = P_out.T @ (pass_results[ip][s]["c2"][it][:, None] * P_out)  # diagonal in (s, mu) bins -> output bins
                c2[icp, s, i1, i2] = blk
                c2[icp, s, i2, i1] = blk.T
                if jackknife:
                    blkj = P_out.T @ (pass_results[ip][s]["c2j"][it][:, None] * P_out)
                    c2j[icp, s, i1, i2] = blkj
                    c2j[icp, s, i2, i1] = blkj.T
    # diagonal blocks: symmetrize (expectation is symmetric)
    for i in range(n_corr):
        c4[:, i, i] = symmetrized(c4[:, i, i])
        c3[:, :, i, i] = symmetrized(c3[:, :, i, i])
        c2[:, :, i, i] = symmetrized(c2[:, :, i, i])
        if jackknife:
            c4j[:, i, i] = symmetrized(c4j[:, i, i])
            c3j[:, :, i, i] = symmetrized(c3j[:, :, i, i])
            c2j[:, :, i, i] = symmetrized(c2j[:, :, i, i])

    # RR estimates, EE/RR per region
    rr_est = np.zeros((n_sub, n_corr, n_bins))
    ee = np.zeros((n_sub, n_corr, 2, n_jack, n_bins)) if jackknife else None
    rrj = np.zeros((n_sub, n_corr, 2, n_jack, n_bins)) if jackknife else None
    for ip, p in enumerate(passes):
        for it, (_, _, ic) in enumerate(p.rr_terms):
            for s in range(n_sub):
                rr_est[s, ic] = pass_results[ip][s]["rr"][it]
                if jackknife:
                    ee[s, ic] = pass_results[ip][s]["ee"][it]
                    rrj[s, ic] = pass_results[ip][s]["rrj"][it]
    counts = {}
    for ip, p in enumerate(passes):
        for k in ("n2", "n3", "n4", "n_draws"):
            counts[f"pass{ip}_{k}"] = np.sum([r[k] for r in pass_results[ip]], axis=0)

    raw = RawCovariance(binning, [s.name for s in samples], cat_names, sample_catalog, corr_keys, cat_pairs, c4, c3, c2, c4j, c3j, c2j, ee, rrj,
                        rr_est, np.asarray(rr_arr), np.asarray(jk_weights) if jackknife else None,
                        np.stack([ci.jack_xi[np.argsort(ci.jack_labels)] for ci in corr_inputs]) if jackknife else None,
                        jack_labels, counts,
                        dict(mode=mode, N2=N2, N3=N3, N4=N4, n_loops=n_loops, loops_per_sample=loops_per_sample, seed=seed, xi_cut=xi_cut,
                             n_side=n_side_final, cellsize=geom.cellsize, periodic=periodic, boxsize=0.0 if boxsize is None else boxsize,
                             positions_float32=positions_float32,
                             galaxies_per_random=gpr, runtime=time.time() - t_start))
    if out_dir is not None:
        raw.save(os.path.join(out_dir, "Raw_Covariance.npz"))
        log(f"Saved raw covariance terms to {os.path.join(out_dir, 'Raw_Covariance.npz')}")
    log(f"Done in {time.time() - t_start:.1f} s")
    return raw


def _normalize(acc: dict, p: PassPlan, gpr: Array, n_prim: int) -> dict:
    """Normalize the sums of one subsample by the number of draws per primary particle (the sums over draws
    estimate sums over all primaries) and by the galaxies-per-random factors."""
    n_pairs, n_triples, n_quads = acc["n_draws"] / n_prim
    out = {}
    g_i, g_j = gpr[p.legs[0]], gpr[p.legs[1]]
    out["c2"] = acc["c2"] / (n_pairs * g_i * g_j)
    out["c2j"] = acc["c2j"] / (n_pairs * g_i * g_j)
    out["rr"] = acc["rr"] / n_pairs
    out["ee"] = acc["ee"] / (0.25 * n_pairs)
    out["rrj"] = acc["rrj"] / (0.25 * n_pairs)
    if n_triples > 0:
        out["t3"] = acc["t3"] / (n_triples * g_j)
        out["t3j"] = acc["t3j"] / (n_triples * g_j)
    else:
        out["t3"], out["t3j"] = acc["t3"], acc["t3j"]
    if n_quads > 0:
        out["c4"] = acc["c4"] / n_quads
        out["c4j"] = acc["c4j"] / n_quads
    else:
        out["c4"], out["c4j"] = acc["c4"], acc["c4j"]
    for k in ("n2", "n3", "n4", "n_draws"):
        out[k] = acc[k]
    return out
