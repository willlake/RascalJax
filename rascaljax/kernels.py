"""Importance-sampling kernels over cell offsets.

Given the primary particle's cell, a secondary cell is drawn at an integer offset
``(dx, dy, dz)`` with ``|d| <= n_half`` from a probability cube of side ``2 n_half + 1``:

* :func:`r2_cell_probabilities` -- the ``1/r^2`` kernel (semi-analytic with the Dawson function),
  used to draw the second particle of each pair at any separation up to ``2 s_max``;
* :func:`xi_cell_probabilities` -- the ``xi(r)``-weighted kernel (Gaussian-smoothed over the cell),
  used to draw particles correlated with a given one, up to ``xi_cut``.

Both follow the RascalC definitions (``random_draws.h``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import simpson
from scipy.special import dawsn

from .utils import Array


def _fill_cube(n_half: int, cellsize: float, prob_of_distance: Callable[[Array], Array]) -> Array:
    """Evaluate ``prob_of_distance`` on the unique offset distances and fill the ``(2n+1)^3`` cube."""
    n = n_half
    ax = np.arange(-n, n + 1)
    dx, dy, dz = np.meshgrid(ax, ax, ax, indexing="ij")
    d2 = dx**2 + dy**2 + dz**2
    uniq, inv = np.unique(d2.ravel(), return_inverse=True)
    vals = prob_of_distance(np.sqrt(uniq) * cellsize)
    vals = np.where(vals > 0, vals, 0.0)
    return vals[inv].reshape(d2.shape)


def r2_cell_probabilities(n_half: int, cellsize: float) -> Array:
    """Unnormalized ``1/r^2``-kernel probabilities of cell offsets, shape ``(2n+1,)*3``."""
    a = cellsize

    def prob(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            p = dawsn(d / a) / (a * d)
        return np.where(d > 0, p, a**-2)

    return _fill_cube(n_half, cellsize, prob)


def xi_cell_probabilities(n_half: int, cellsize: float, xi_iso: Callable[[Array], Array], n_quad: int = 4001) -> Array:
    """Unnormalized ``xi``-kernel probabilities of cell offsets.

    For a cell at distance ``d`` from the primary cell, the probability is the integral of ``xi+(x)``
    (``xi`` floored at ``10 / x^2`` where it is below ``1e-10``, as in RascalC) against a Gaussian
    window of width ``R = cellsize / 2`` for the pair separation.
    """
    R = cellsize / 2.0
    x_max = 2.0 * cellsize * n_half
    x = np.linspace(0.0, x_max, n_quad)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = xi_iso(x)
        floor = 10.0 / x**2
    xi_plus = np.where(xi < 1e-10, floor, xi)
    xi_plus = np.where(x > 0, xi_plus, 0.0)

    def prob(d):
        d = np.asarray(d, dtype=float)[:, None]
        f1 = ((x[None, :] + d) / (2 * R)) ** 2
        f2 = ((x[None, :] - d) / (2 * R)) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = x[None, :] / (R * d) * (np.exp(-f2) - np.exp(-f1)) * xi_plus[None, :]
        integrand0 = x[None, :] ** 2 * xi_plus[None, :] / R**3 * np.exp(-((x[None, :] / (2 * R)) ** 2))
        integrand = np.where(d > 0, integrand, integrand0)
        return simpson(integrand, x=x, axis=1)

    return _fill_cube(n_half, cellsize, prob)


def alias_tables(p: Array) -> tuple[Array, Array]:
    """Walker/Vose alias tables for a discrete distribution ``p`` (normalized to 1).

    Returns ``(threshold, alias)``: draw ``i = floor(u1 * n)``; the sample is ``i`` if ``u2 < threshold[i]``,
    otherwise ``alias[i]``.
    """
    n = len(p)
    scaled = p * n
    threshold = np.ones(n)
    alias = np.arange(n)
    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]
    while small and large:
        s = small.pop()
        l = large.pop()
        threshold[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] + scaled[s] - 1.0
        (small if scaled[l] < 1.0 else large).append(l)
    for i in small + large:
        threshold[i] = 1.0
        alias[i] = i
    return threshold, alias


@dataclass
class OffsetSampler:
    """Alias-method sampler of cell offsets from a probability cube."""

    n_half: int
    prob: Array  # normalized flat probabilities, length (2n+1)^3
    threshold: Array
    alias: Array

    @classmethod
    def from_cube(cls, cube: Array) -> "OffsetSampler":
        n_half = (cube.shape[0] - 1) // 2
        p = cube.ravel().astype(np.float64)
        p = p / p.sum()
        thr, al = alias_tables(p)
        return cls(n_half, p, thr, al)

    def draw(self, rng: np.random.Generator, size: int) -> Array:
        """Numpy reference implementation of the draw (for tests)."""
        n = len(self.prob)
        i = np.minimum((rng.uniform(size=size) * n).astype(np.int64), n - 1)
        return np.where(rng.uniform(size=size) < self.threshold[i], i, self.alias[i])

    @property
    def side(self) -> int:
        return 2 * self.n_half + 1

    def offsets_of(self, flat: Array) -> Array:
        """Flat index -> (N, 3) integer offsets."""
        side = self.side
        z = flat % side
        y = (flat // side) % side
        x = flat // (side * side)
        return np.stack([x, y, z], axis=-1) - self.n_half
