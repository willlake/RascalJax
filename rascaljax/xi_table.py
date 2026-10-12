"""Tabulated 2-point correlation function xi(s, mu) with the interpolation and extrapolation rules of RascalC.

The table is interpolated as ``s^2 xi`` with a row ``s = 0, s^2 xi = 0`` prepended, using cubic splines
(bicubic in (s, mu) when there are enough mu points). Outside the tabulated range:

* ``s > s_max``: ``xi(s_max, mu) * (s / s_max)^-4``;
* ``s < s_min``: smooth transition to a constant below ``s_min / 2`` (``s_min`` is the first tabulated s);
* ``mu`` is clamped to the tabulated range.

For the JAX kernels the table is sampled on a fine uniform (s, mu) grid and interpolated bilinearly
(:meth:`XiTable.lookup`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from .utils import Array


def _smooth_transition(x, x1, x2, f1, f2):
    """Smooth (C2) transition from f1 at x <= x1 to f2 at x >= x2."""
    y = np.clip((x2 - x) / (x2 - x1), 0.0, 1.0)
    w1 = y**3 * (10.0 - 15.0 * y + 6.0 * y**2)
    return w1 * f1 + (1.0 - w1) * f2


@dataclass
class XiLookup:
    """Fine-grid representation of a :class:`XiTable` for bilinear interpolation inside JAX kernels."""

    s_max: float  # table maximum separation; beyond it xi falls as s^-4
    ds: float  # grid spacing in s (grid starts at 0)
    n_mu: int  # number of mu grid points (uniform in [0, 1])
    table: Array  # (n_s_grid, n_mu) values of xi


class XiTable:
    """Correlation function table.

    Parameters
    ----------
    s_vals : (n_s,) reference separations (bin centers or mean separations), increasing, > 0.
    mu_vals : (n_mu,) reference mu values in [0, 1], increasing.
    xi : (n_s, n_mu) correlation function values.
    s_edges : optional (n_s + 1,) separation bin edges, needed for the refinement procedure
        (the table is then interpreted as bin averages).
    """

    def __init__(self, s_vals: Array, mu_vals: Array, xi: Array, s_edges: Array | None = None):
        s_vals = np.asarray(s_vals, dtype=float).ravel()
        mu_vals = np.asarray(mu_vals, dtype=float).ravel()
        xi = np.asarray(xi, dtype=float)
        if xi.shape != (len(s_vals), len(mu_vals)):
            raise ValueError("xi must have shape (len(s_vals), len(mu_vals))")
        if np.any(np.diff(s_vals) <= 0) or s_vals[0] <= 0:
            raise ValueError("s_vals must be positive and increasing")
        if len(mu_vals) > 1 and np.any(np.diff(mu_vals) <= 0):
            raise ValueError("mu_vals must be increasing")
        if not np.all(np.isfinite(xi)):
            raise ValueError("xi table contains non-finite values")
        self.s_vals, self.mu_vals, self.xi = s_vals, mu_vals, xi
        self.s_edges = None if s_edges is None else np.asarray(s_edges, dtype=float).ravel()
        if self.s_edges is not None and len(self.s_edges) != len(s_vals) + 1:
            raise ValueError("s_edges must have len(s_vals) + 1 entries")
        self.s_min, self.s_max = float(s_vals[0]), float(s_vals[-1])
        self.mu_min, self.mu_max = float(mu_vals[0]), float(mu_vals[-1])
        self.mudim = len(mu_vals) > 1
        r = np.concatenate([[0.0], s_vals])
        z = np.concatenate([np.zeros((1, len(mu_vals))), xi * s_vals[:, None] ** 2], axis=0)
        self._z1d = CubicSpline(r, z.mean(axis=1), bc_type="natural")
        if self.mudim:
            ky = min(3, len(mu_vals) - 1)
            self._z2d = RectBivariateSpline(r, mu_vals, z, kx=min(3, len(r) - 1), ky=ky, s=0)
        else:
            self._z2d = None

    # ----- evaluation -----
    def _z(self, s, mu):
        if self._z2d is None:
            return self._z1d(s)
        return self._z2d(s, mu, grid=False)

    def __call__(self, s: Array, mu: Array | None = None) -> Array:
        """xi(s, mu) with RascalC extrapolation rules; ``mu=None`` gives the mu-averaged (isotropic) xi."""
        s = np.asarray(s, dtype=float)
        if mu is None or not self.mudim:
            z = self._z1d
            f = lambda r: z(r)  # noqa: E731
        else:
            s, mu_c = np.broadcast_arrays(s, np.clip(np.asarray(mu, dtype=float), self.mu_min, self.mu_max))
            f = lambda r: self._z(r, np.broadcast_to(mu_c, np.shape(r)))  # noqa: E731
        s_in = np.clip(s, self.s_min / 2, self.s_max)
        core = f(s_in) / s_in**2
        low = f(np.full_like(s, self.s_min / 2)) / (self.s_min / 2) ** 2
        out = np.where(s < self.s_min, _smooth_transition(s, self.s_min / 2, self.s_min, low, core), core)
        tail = f(np.full_like(s, self.s_max)) / self.s_max**2 * (np.maximum(s, self.s_max) / self.s_max) ** -4
        return np.where(s > self.s_max, tail, out)

    def isotropic(self, s: Array) -> Array:
        return self(s, None)

    # ----- transforms -----
    def rescaled(self, ratio: Array, s_vals: Array | None = None, mu_vals: Array | None = None) -> "XiTable":
        """Table with values ``ratio * xi`` evaluated at (optionally new) reference points; keeps the edges."""
        s_vals = self.s_vals if s_vals is None else np.asarray(s_vals, dtype=float)
        mu_vals = self.mu_vals if mu_vals is None else np.asarray(mu_vals, dtype=float)
        xi_at = self(s_vals[:, None], mu_vals[None, :])
        return XiTable(s_vals, mu_vals, xi_at * ratio, self.s_edges)

    def lookup(self, ds: float = 0.25, n_mu: int = 201) -> XiLookup:
        """Sample on a fine grid for bilinear interpolation (s from 0 to s_max, mu from 0 to 1)."""
        n_s = int(np.ceil(self.s_max / ds)) + 1
        ds = self.s_max / (n_s - 1)
        s_grid = np.arange(n_s) * ds
        mu_grid = np.linspace(0.0, 1.0, n_mu)
        table = self(s_grid[:, None], mu_grid[None, :])
        return XiLookup(self.s_max, ds, n_mu, np.ascontiguousarray(table))

    # ----- constructors -----
    @classmethod
    def from_model(cls, s_vals: Array, mu_vals: Array, xi: Array) -> "XiTable":
        """A theory table evaluated at grid points (no refinement needed)."""
        return cls(s_vals, mu_vals, xi, None)

    @classmethod
    def from_binned(cls, s_edges: Array, mu_edges: Array, xi: Array, s_avg: Array | None = None, mu_avg: Array | None = None) -> "XiTable":
        """A table of bin averages in (s, mu) bins; refinement will make the interpolant's bin averages match."""
        s_edges = np.asarray(s_edges, dtype=float)
        mu_edges = np.asarray(mu_edges, dtype=float)
        s_vals = 0.5 * (s_edges[1:] + s_edges[:-1]) if s_avg is None else np.asarray(s_avg, dtype=float)
        mu_vals = 0.5 * (mu_edges[1:] + mu_edges[:-1]) if mu_avg is None else np.asarray(mu_avg, dtype=float)
        return cls(s_vals, mu_vals, xi, s_edges)

    def write(self, filename: str) -> None:
        """Write in the RascalC text format (s values, mu values, then rows of xi)."""
        with open(filename, "w") as f:
            f.write("\t".join(f"{v:.8e}" for v in self.s_vals) + "\n")
            f.write("\t".join(f"{v:.8e}" for v in self.mu_vals) + "\n")
            for row in self.xi:
                f.write("\t".join(f"{v:.8e}" for v in row) + "\n")

    @classmethod
    def read(cls, filename: str, s_edges: Array | None = None) -> "XiTable":
        """Read the RascalC text format."""
        with open(filename) as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
        s_vals = np.array(lines[0].split(), dtype=float)
        mu_vals = np.array(lines[1].split(), dtype=float)
        xi = np.array([l.split() for l in lines[2:]], dtype=float)
        return cls(s_vals, mu_vals, xi, s_edges)
