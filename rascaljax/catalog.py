"""Random catalogs and the samples (membership-weighted views of a catalog) whose
correlation functions we compute covariances for.

A :class:`Catalog` holds random points representing the selection function of one tracer.
A :class:`Sample` is a catalog plus a per-particle membership weight (0/1 mask, or fractional).
Two samples on the same catalog with overlapping memberships (e.g. overlapping redshift bins)
share particles, so their covariance gets the shot-noise cross terms automatically.
Samples on different catalogs never share particles (distinct tracers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .utils import Array

PositionType = Literal["pos", "xyz", "rdd"]


def format_positions(positions, position_type: PositionType = "pos") -> Array:
    """Return Cartesian positions with shape ``(N, 3)``.

    ``"pos"``: array ``(N, 3)``; ``"xyz"``: three arrays / array ``(3, N)``;
    ``"rdd"``: RA, Dec (degrees) and comoving distance, ``(3, N)`` or a sequence of three arrays.
    """
    if position_type == "pos":
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError('positions must have shape (N, 3) for position_type="pos"')
        return np.ascontiguousarray(pos)
    arrs = [np.asarray(a, dtype=np.float64).ravel() for a in positions]
    if len(arrs) != 3 or len({len(a) for a in arrs}) != 1:
        raise ValueError("positions must be three arrays of equal length")
    if position_type == "xyz":
        return np.ascontiguousarray(np.stack(arrs, axis=1))
    if position_type == "rdd":
        ra, dec, dist = arrs
        ra_r, dec_r = np.radians(ra), np.radians(dec)
        cd = np.cos(dec_r)
        return np.ascontiguousarray(np.stack([dist * cd * np.cos(ra_r), dist * cd * np.sin(ra_r), dist * np.sin(dec_r)], axis=1))
    raise ValueError(f"Unknown position_type {position_type!r}")


@dataclass
class Catalog:
    """Random catalog of one tracer.

    Parameters
    ----------
    positions, position_type
        See :func:`format_positions`.
    weights
        Per-particle weights (FKP etc.). They are normalized to unit sum when ``normalize=True``
        (default), which matches normalized pair counts.
    n_data
        Number of data galaxies represented by this catalog (sets the Gaussian shot noise:
        each random particle stands for ``n_data / n_random`` galaxies).
    jackknife
        Optional per-particle jackknife region labels (integers).
    effective_n
        Use the effective number of randoms ``(sum w)^2 / sum w^2`` instead of the plain count.
    name
        Label used in outputs.
    """

    positions: Array
    weights: Array
    n_data: float
    jackknife: Array | None = None
    position_type: PositionType = "pos"
    normalize: bool = True
    effective_n: bool = False
    name: str = "catalog"
    rand_class: Array = field(init=False, repr=False)

    def __post_init__(self):
        self.positions = format_positions(self.positions, self.position_type)
        self.position_type = "pos"
        n = len(self.positions)
        w = np.asarray(self.weights, dtype=np.float64).ravel()
        if len(w) != n:
            raise ValueError("weights must have one entry per particle")
        if self.normalize:
            w = w / np.sum(w)
        self.weights = w
        if self.jackknife is not None:
            jk = np.asarray(self.jackknife).ravel()
            if len(jk) != n:
                raise ValueError("jackknife labels must have one entry per particle")
            self.jackknife = jk
        if self.n_data is None or self.n_data <= 0:
            raise ValueError("n_data (number of data galaxies represented by the randoms) must be positive")
        # random class 0/1 for the jackknife disconnected term (as in RascalC)
        self.rand_class = np.zeros(n, dtype=np.int8)

    def assign_random_classes(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.rand_class = rng.integers(0, 2, size=len(self.positions)).astype(np.int8)

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def n_random(self) -> float:
        """Number of randoms (effective if requested)."""
        if self.effective_n:
            return float(np.sum(self.weights) ** 2 / np.sum(self.weights**2))
        return float(self.size)

    @property
    def galaxies_per_random(self) -> float:
        """``n_data / n_random``: galaxies represented by each random particle."""
        return self.n_data / self.n_random

    def subset(self, mask: Array) -> "Catalog":
        """Return a new catalog with the masked particles (weights are not renormalized)."""
        c = Catalog(self.positions[mask], self.weights[mask], self.n_data, None if self.jackknife is None else self.jackknife[mask],
                    "pos", False, self.effective_n, self.name)
        c.rand_class = self.rand_class[mask]
        return c


@dataclass
class Sample:
    """A membership-weighted view of a :class:`Catalog`.

    ``membership`` is a per-particle weight in [0, 1] (bool masks accepted); ``None`` means all particles.
    The effective weight of particle ``i`` in this sample is ``catalog.weights[i] * membership[i]``.
    """

    catalog: Catalog
    membership: Array | None = None
    name: str = "sample"

    def __post_init__(self):
        if self.membership is None:
            self.membership = np.ones(self.catalog.size, dtype=np.float64)
        else:
            m = np.asarray(self.membership, dtype=np.float64).ravel()
            if len(m) != self.catalog.size:
                raise ValueError("membership must have one entry per catalog particle")
            if np.any(m < 0):
                raise ValueError("membership weights must be non-negative")
            self.membership = m

    @property
    def effective_weights(self) -> Array:
        return self.catalog.weights * self.membership

    @classmethod
    def from_redshift_range(cls, catalog: Catalog, redshifts: Array, z_min: float, z_max: float, name: str | None = None) -> "Sample":
        """Sample of the catalog particles with ``z_min <= z < z_max``."""
        z = np.asarray(redshifts).ravel()
        if len(z) != catalog.size:
            raise ValueError("redshifts must have one entry per catalog particle")
        return cls(catalog, (z >= z_min) & (z < z_max), name or f"{catalog.name}_z{z_min}-{z_max}")
