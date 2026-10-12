"""Cell grid used to importance-sample particles by cell offsets.

Particles are sorted by cell; each cell stores the start and count of its particle slice
(CSR layout). The grid is either a cuboid bounding box (aperiodic) or a periodic cube.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import Catalog
from .utils import Array

MIN_DENSITY, MAX_DENSITY = 2.0, 16.0  # particles per filled cell, as in RascalC


@dataclass
class Geometry:
    """Common box geometry of all grids used in one run."""

    origin: Array  # (3,) lower corner (aperiodic) or 0 (periodic)
    boxsize: Array  # (3,) box side lengths
    cellsize: float
    n_side: Array  # (3,) number of cells along each axis (int)
    periodic: bool

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.n_side))

    def cell_3d(self, positions: Array) -> Array:
        """Integer cell coordinates of positions."""
        c = np.floor((positions - self.origin) / self.cellsize).astype(np.int64)
        if self.periodic:
            c %= self.n_side
        else:
            c = np.clip(c, 0, self.n_side - 1)
        return c

    def flat_cell(self, c3: Array) -> Array:
        return (c3[:, 0] * self.n_side[1] + c3[:, 1]) * self.n_side[2] + c3[:, 2]


def make_geometry(catalogs: list[Catalog], n_side: int, boxsize: float | None, s_max: float) -> Geometry:
    """Build the geometry for the given number of cells along the longest side."""
    if boxsize:  # periodic cube
        cellsize = boxsize / n_side
        return Geometry(np.zeros(3), np.full(3, float(boxsize)), cellsize, np.full(3, int(n_side)), True)
    pos = np.concatenate([c.positions for c in catalogs], axis=0)
    pmin, pmax = pos.min(axis=0), pos.max(axis=0)
    prange = pmax - pmin
    safety = 1.0 + 1.0 / 4096.0
    cellsize = safety * prange.max() / n_side
    n_side3 = np.maximum(np.ceil(safety * prange / cellsize).astype(int), 1)
    return Geometry(pmin, n_side3 * cellsize, cellsize, n_side3, False)


@dataclass
class CellGrid:
    """A catalog sorted into the cells of a :class:`Geometry`."""

    geometry: Geometry
    order: Array  # permutation sorting the catalog particles by cell
    positions: Array  # sorted positions (periodic: wrapped into [0, L))
    weights: Array  # sorted weights
    jackknife: Array  # sorted jackknife labels (collapsed to 0..n_jack-1), zeros if none
    rand_class: Array  # sorted random classes (0/1)
    cell_of_particle: Array  # flat cell index per sorted particle
    cell_start: Array  # (n_cells,) start of the particle slice
    cell_count: Array  # (n_cells,) number of particles

    @classmethod
    def from_catalog(cls, catalog: Catalog, geometry: Geometry, jackknife_labels: Array | None = None) -> "CellGrid":
        pos = catalog.positions.copy()
        if geometry.periodic:
            pos = np.mod(pos, geometry.boxsize)
        c3 = geometry.cell_3d(pos)
        flat = geometry.flat_cell(c3)
        order = np.argsort(flat, kind="stable")
        flat_sorted = flat[order]
        counts = np.bincount(flat_sorted, minlength=geometry.n_cells)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        if catalog.jackknife is None:
            jk = np.zeros(catalog.size, dtype=np.int32)
        else:
            labels = np.unique(catalog.jackknife) if jackknife_labels is None else np.asarray(jackknife_labels)
            jk = np.searchsorted(labels, catalog.jackknife).astype(np.int32)
            if np.any(labels[jk] != catalog.jackknife):
                raise ValueError("Catalog jackknife labels not in the provided label list")
        return cls(geometry, order, np.ascontiguousarray(pos[order]), catalog.weights[order], jk[order],
                   catalog.rand_class[order].astype(np.int32), flat_sorted, starts.astype(np.int64), counts.astype(np.int64))

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def n_filled(self) -> int:
        return int(np.count_nonzero(self.cell_count))

    @property
    def max_per_cell(self) -> int:
        return int(self.cell_count.max())

    @property
    def mean_density(self) -> float:
        """Mean particles per filled cell."""
        return self.size / max(self.n_filled, 1)

    def sort_membership(self, membership: Array) -> Array:
        return membership[self.order]


def choose_n_side(catalogs: list[Catalog], boxsize: float | None, s_max: float, n_side: int = 301,
                  attempts: int = 8, log=print) -> int:
    """Pick an odd ``n_side`` giving 2-16 particles per filled cell for every catalog (RascalC heuristics)."""
    for _ in range(attempts + 1):
        geom = make_geometry(catalogs, n_side, boxsize, s_max)
        dens = []
        ok = True
        for i, cat in enumerate(catalogs):
            flat = geom.flat_cell(geom.cell_3d(np.mod(cat.positions, geom.boxsize) if geom.periodic else cat.positions))
            n_filled = len(np.unique(flat))
            d = cat.size / n_filled
            dens.append(d)
            if d > MAX_DENSITY or d < MIN_DENSITY:
                if d > MAX_DENSITY:
                    aim = (MAX_DENSITY**2 * MIN_DENSITY) ** (1 / 3)
                else:
                    aim = (MAX_DENSITY * MIN_DENSITY**2) ** (1 / 3)
                if i:
                    aim = np.sqrt(MAX_DENSITY * MIN_DENSITY * d / dens[0])
                approx = (d / aim) ** (1 / 3) * n_side
                n_side = 2 * int(round((approx + 1) / 2)) - 1
                log(f"Grid density {d:.2f} per filled cell for {cat.name}; retrying with n_side={n_side}")
                ok = False
                break
        if ok:
            return n_side
    log(f"WARNING: could not meet the grid density constraints; using n_side={n_side}")
    return n_side
