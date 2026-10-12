"""Separation and angular binning, and the projection from (s, mu) bins to Legendre multipoles.

Conventions (shared by the whole package):

* separation bins have arbitrary contiguous edges ``s_edges`` (length ``n_s + 1``);
* angular bins are linear in ``mu = |cos theta|`` between 0 and 1 (``n_mu`` bins);
* the flat (s, mu) bin index is ``i_s * n_mu + i_mu``;
* Legendre bins are ``(s, ell)`` with the flat index ``i_s * n_ell + i_ell`` for even ``ell``
  (the RascalC ordering); :func:`to_pycorr_order` switches to ``(ell, s)`` ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import legendre

from .utils import Array, format_skip_s_bins


def mu_bin_legendre_factors(mu_edges: Array, max_l: int) -> Array:
    """Factors ``F[i_mu, i_ell] = (2 ell + 1) * int_{bin} P_ell(mu) dmu`` projecting mu-binned
    values (uniform within the bin) to even multipoles, i.e. ``xi_ell(s) = sum_mu F[mu, ell] xi(s, mu)``.
    Identical to RascalC's ``compute_mu_bin_legendre_factors``."""
    if max_l < 0 or max_l % 2:
        raise ValueError("max_l must be a non-negative even integer")
    ells = np.arange(0, max_l + 1, 2)
    mu_edges = np.asarray(mu_edges, dtype=float)
    out = np.zeros((len(mu_edges) - 1, len(ells)))
    for i, ell in enumerate(ells):
        integ = np.polyint(legendre(ell))
        out[:, i] = (2 * ell + 1) * np.diff(integ(mu_edges))
    return out


def legendre_to_mu_bin_factors(mu_edges: Array, max_l: int) -> Array:
    """Factors ``G[i_ell, i_mu] = int_{bin} P_ell(mu) dmu / dmu`` giving the bin-averaged multipole values, i.e.
    the (approximate) inverse projection ``xi(s, mu_bin) = sum_ell G[ell, mu] xi_ell(s)``."""
    ells = np.arange(0, max_l + 1, 2)
    mu_edges = np.asarray(mu_edges, dtype=float)
    out = np.zeros((len(ells), len(mu_edges) - 1))
    for i, ell in enumerate(ells):
        integ = np.polyint(legendre(ell))
        out[i] = np.diff(integ(mu_edges)) / np.diff(mu_edges)
    return out


@dataclass(frozen=True)
class Binning:
    """Binning of the covariance (and of the pair counts normalizing it)."""

    s_edges: Array
    n_mu: int
    max_l: int | None = None  # None -> (s, mu) output; even integer -> Legendre output
    mu_edges: Array = field(init=False, repr=False)

    def __post_init__(self):
        s_edges = np.ascontiguousarray(np.asarray(self.s_edges, dtype=float))
        if s_edges.ndim != 1 or len(s_edges) < 2 or np.any(np.diff(s_edges) <= 0):
            raise ValueError("s_edges must be a 1D increasing array of at least two edges")
        if self.n_mu < 1:
            raise ValueError("n_mu must be positive")
        if self.max_l is not None and (self.max_l < 0 or self.max_l % 2):
            raise ValueError("max_l must be None or a non-negative even integer")
        object.__setattr__(self, "s_edges", s_edges)
        object.__setattr__(self, "mu_edges", np.linspace(0.0, 1.0, self.n_mu + 1))

    # ----- sizes -----
    @property
    def n_s(self) -> int:
        return len(self.s_edges) - 1

    @property
    def s_min(self) -> float:
        return float(self.s_edges[0])

    @property
    def s_max(self) -> float:
        return float(self.s_edges[-1])

    @property
    def n_bins(self) -> int:
        """Number of flat (s, mu) bins."""
        return self.n_s * self.n_mu

    @property
    def legendre(self) -> bool:
        return self.max_l is not None

    @property
    def n_ell(self) -> int:
        if self.max_l is None:
            raise ValueError("Not a Legendre binning")
        return self.max_l // 2 + 1

    @property
    def ells(self) -> Array:
        return np.arange(0, self.max_l + 1, 2)

    @property
    def n_out(self) -> int:
        """Number of output bins per correlation: ``n_s * n_ell`` (Legendre) or ``n_s * n_mu``."""
        return self.n_s * (self.n_ell if self.legendre else self.n_mu)

    @property
    def s_mid(self) -> Array:
        return 0.5 * (self.s_edges[1:] + self.s_edges[:-1])

    @property
    def mu_mid(self) -> Array:
        return 0.5 * (self.mu_edges[1:] + self.mu_edges[:-1])

    # ----- projections -----
    def legendre_factors(self) -> Array:
        """``(n_mu, n_ell)`` factors from mu bins to even multipoles."""
        return mu_bin_legendre_factors(self.mu_edges, self.max_l)

    def inverse_legendre_factors(self) -> Array:
        """``(n_ell, n_mu)`` bin-averaged multipole values (see :func:`legendre_to_mu_bin_factors`)."""
        return legendre_to_mu_bin_factors(self.mu_edges, self.max_l)

    def projection_matrix(self) -> Array:
        """Matrix ``P`` of shape ``(n_bins, n_out)`` such that a vector in flat (s, mu) bins
        maps to the output bins as ``v_out = v @ P``; a covariance maps as ``P.T @ C @ P``.
        Identity for (s, mu) output."""
        if not self.legendre:
            return np.eye(self.n_bins)
        f = self.legendre_factors()  # (n_mu, n_ell)
        P = np.zeros((self.n_s, self.n_mu, self.n_s, self.n_ell))
        for i in range(self.n_s):
            P[i, :, i, :] = f
        return P.reshape(self.n_bins, self.n_out)

    def project_matrix(self, cov_smu: Array) -> Array:
        """Project a covariance in flat (s, mu) bins (last two axes) to the output bins."""
        P = self.projection_matrix()
        return np.swapaxes(np.swapaxes(cov_smu @ P, -1, -2) @ P, -1, -2)  # P^T C P, batched over leading axes

    def project_vector(self, v_smu: Array) -> Array:
        """Project a vector (last axis) in flat (s, mu) bins to the output bins."""
        return v_smu @ self.projection_matrix()

    # ----- bin lookups (numpy) -----
    def s_bin(self, s: Array) -> Array:
        """Separation bin index, -1 outside the range."""
        idx = np.searchsorted(self.s_edges, s, side="right") - 1
        return np.where((idx >= 0) & (idx < self.n_s), idx, -1)

    def mu_bin(self, mu: Array) -> Array:
        """Angular bin index for mu in [0, 1] (mu = 1 goes to the last bin)."""
        return np.clip(np.floor(mu * self.n_mu).astype(int), 0, self.n_mu - 1)

    def flat_bin(self, s: Array, mu: Array) -> Array:
        """Flat (s, mu) bin, -1 outside the separation range."""
        i_s = self.s_bin(s)
        return np.where(i_s >= 0, i_s * self.n_mu + self.mu_bin(mu), -1)

    # ----- output selection -----
    def output_filter(self, skip_s_bins: int | tuple[int, int] = 0, skip_l: int = 0) -> Array:
        """Indices of output bins kept after skipping separation bins (from the start and the end)
        and the highest ``skip_l`` multipoles."""
        n0, n1 = format_skip_s_bins(skip_s_bins)
        s_idx = np.arange(n0, self.n_s - n1)
        if self.legendre:
            if skip_l < 0 or skip_l >= self.n_ell:
                raise ValueError("skip_l must be between 0 and n_ell - 1")
            l_idx = np.arange(self.n_ell - skip_l)
            return (s_idx[:, None] * self.n_ell + l_idx[None, :]).ravel()
        if skip_l:
            raise ValueError("skip_l only applies to Legendre binning")
        return (s_idx[:, None] * self.n_mu + np.arange(self.n_mu)[None, :]).ravel()

    def to_pycorr_order(self, cov: Array, n_corr: int = 1, n_ell: int | None = None, n_s: int | None = None) -> Array:
        """Reorder a Legendre covariance from ``(corr, s, ell)`` to ``(corr, ell, s)`` ordering."""
        if not self.legendre:
            return cov
        n_ell = self.n_ell if n_ell is None else n_ell
        n_s = self.n_s if n_s is None else n_s
        n = len(cov)
        if n != n_corr * n_s * n_ell:
            raise ValueError("Covariance size inconsistent with n_corr * n_s * n_ell")
        c = cov.reshape(n_corr, n_s, n_ell, n_corr, n_s, n_ell).transpose(0, 2, 1, 3, 5, 4)
        return c.reshape(n, n)
