"""Adapters from ``lsstypes`` 2-point correlation objects (``Count2Correlation`` and ``Count2JackknifeCorrelation``,
as produced by ``cucount`` / ``desi-clustering``) to :class:`rascaljax.driver.CorrelationInput`.

Conventions follow RascalC's ``lsstypes_utils``:

* counts in ``mu`` from -1 to 1 are wrapped to ``|mu|`` (the number of mu bins must be even);
* normalized RR counts are used (weights are assumed normalized to unit sum per catalog);
* the input xi table is the measured correlation function multiplied by ``RR / SS`` when shifted randoms are
  present (post-reconstruction), tabulated at the mean separations of the bins;
* the jackknife realization ``A`` is ``j_A x all + all x j_A`` halved: ``auto + (cross12 + cross21) / 2``,
  with jackknife weights ``w_aA = RR_aA / sum_A RR_aA``.
"""
from __future__ import annotations

from warnings import warn

import numpy as np

from .driver import CorrelationInput
from .utils import Array
from .xi_table import XiTable


def _edges_1d(obj, coord: str) -> Array:
    edges = obj.get("RR").edges(coord)
    edges = np.asarray(edges)
    if edges.ndim == 2:
        if not np.allclose(edges[1:, 0], edges[:-1, 1]):
            raise ValueError("Bins are not adjacent")
        return np.concatenate([[edges[0, 0]], (edges[1:, 0] + edges[:-1, 1]) / 2, [edges[-1, 1]]])
    return edges


def wrap_counts(count):
    """Wrap a ``Count2`` with symmetric mu edges to |mu| (copied from RascalC / lsstypes tests)."""
    import lsstypes

    if count.shape[1] % 2:
        raise ValueError("Cannot wrap counts with an odd number of mu bins")
    mid = count.shape[1] // 2
    sl_neg, sl_pos = slice(mid - 1, None, -1), slice(mid, None, 1)
    coord_name = count._coords_names[1]
    mu_edges = count.edges(coord_name)
    if not np.allclose(mu_edges[sl_neg], -mu_edges[sl_pos, ::-1]):
        raise ValueError("Mu edges are not symmetric; cannot wrap")
    mu_edges = mu_edges[sl_pos]
    counts = count.values("counts")
    counts = counts[..., sl_neg] + counts[..., sl_pos]
    norm = count.values("norm")[..., sl_pos] if np.ndim(count.values("norm")) >= 2 else count.values("norm")
    edges = dict(count.edges())
    edges[coord_name] = mu_edges
    edges = {f"{coord}_edges": edge for coord, edge in edges.items()}
    coords = dict(count.coords())
    coords[coord_name] = count.coords(coord_name)[sl_pos]
    return lsstypes.Count2(counts=counts, norm=norm, **coords, **edges, coords=list(coords), attrs=dict(count.attrs))


def wrap_correlation(corr):
    """Wrap a correlation (and its jackknife counts) to |mu| if its mu range starts below 0."""
    mu_edges = _edges_1d(corr, "mu")
    if mu_edges[0] >= 0:
        return corr
    return corr.map(lambda count: wrap_counts(count), level=None, is_leaf=lambda branch: False)


def rr_counts(corr) -> Array:
    """Normalized RR counts, shape (n_s, n_mu)."""
    return np.asarray(corr.get("RR").values("normalized_counts"), dtype=float)


def input_xi(corr) -> Array:
    """Correlation function for the integrals: ``xi * RR / SS`` when shifted randoms are present."""
    xi = np.asarray(corr.value(), dtype=float)
    if "SS" in corr.count_names:
        xi = xi * rr_counts(corr) / np.asarray(corr.get("SS").values("normalized_counts"), dtype=float)
    return xi


def xi_table_from_lsstypes(corr) -> XiTable:
    """Bin-averaged xi table (with edges for refinement) at the mean separations and mu of the bins."""
    corr = wrap_correlation(corr)
    s_edges = _edges_1d(corr, "s")
    mu_edges = _edges_1d(corr, "mu")
    s_avg = np.asarray(corr.coords("s"), dtype=float)
    mu_avg = np.asarray(corr.coords("mu"), dtype=float)
    if s_avg.ndim == 2:
        s_avg = np.nanmean(s_avg, axis=1)
    if mu_avg.ndim == 2:
        mu_avg = np.nanmean(mu_avg, axis=0)
    xi = input_xi(corr)
    bad = ~np.isfinite(xi)
    if np.any(bad):
        warn(f"{bad.sum()} non-finite xi values replaced by 0")
        xi = np.where(bad, 0.0, xi)
    s_avg = np.where(np.isfinite(s_avg), s_avg, 0.5 * (s_edges[1:] + s_edges[:-1]))
    mu_avg = np.where(np.isfinite(mu_avg), mu_avg, 0.5 * (mu_edges[1:] + mu_edges[:-1]))
    return XiTable.from_binned(s_edges, mu_edges, xi, s_avg, mu_avg)


def jackknife_realization(jack_corr, ii):
    """RascalC-framed jackknife realization ``auto + (cross12 + cross21) / 2`` with the total norm."""
    import lsstypes

    kw = {}
    for name in jack_corr.count_names:
        jc = jack_corr.get(name)
        counts = jc.get(realization=ii, cross="ii").values("counts") + 0.5 * (jc.get(realization=ii, cross="ij").values("counts") + jc.get(realization=ii, cross="ji").values("counts"))
        kw[name] = lsstypes.Count2(counts=counts, norm=jc.values("norm"))
    return lsstypes.Count2Correlation(**kw, estimator=jack_corr.estimator, attrs=jack_corr.attrs)


def jackknife_arrays(jack_corr):
    """``(labels, xi_jack, rr_jack)`` with shapes ``(n_jack,)``, ``(n_jack, n_bins)``, ``(n_jack, n_bins)``."""
    labels = np.asarray(jack_corr.realizations)
    reals = [jackknife_realization(jack_corr, ii) for ii in labels]
    xi_jack = np.array([np.asarray(r.value()).ravel() for r in reals])
    rr_jack = np.array([rr_counts(r).ravel() for r in reals])
    return labels, xi_jack, rr_jack


def correlation_input_from_lsstypes(counts, xi=None, s_slice: slice | None = None, rebin: int | None = None) -> CorrelationInput:
    """Build a :class:`CorrelationInput` from lsstypes objects.

    Parameters
    ----------
    counts : ``Count2Correlation`` or ``Count2JackknifeCorrelation``
        Pair counts (and jackknife counts) in the covariance binning; use ``s_slice``/``rebin`` to select and
        rebin the separations (e.g. ``s_slice=slice(20, 200)``, ``rebin=4`` for 4 Mpc/h bins from 20 to 200).
    xi : optional correlation object used for the xi table (finer binning is fine); ``counts`` if None.
    """
    if s_slice is not None or rebin is not None:
        counts = counts.select(s=s_slice) if s_slice is not None else counts
        if rebin is not None:
            counts = counts.select(s=slice(0, counts.shape[0], rebin))
    wrapped = wrap_correlation(counts)
    s_edges = _edges_1d(wrapped, "s")
    mu_edges = _edges_1d(wrapped, "mu")
    if not np.allclose(mu_edges, np.linspace(0, 1, len(mu_edges))):
        raise ValueError("Mu bins must be linear between 0 and 1 after wrapping")
    rr = rr_counts(wrapped)
    table = xi_table_from_lsstypes(counts if xi is None else xi)
    kw = {}
    if hasattr(wrapped, "realizations"):
        labels, xi_jack, rr_jack = jackknife_arrays(wrapped)
        if not np.allclose(rr_jack.sum(axis=0), rr.ravel()):
            raise ValueError("Jackknife RR counts do not sum to the total RR counts")
        kw = dict(jack_xi=xi_jack, jack_rr=rr_jack, jack_labels=labels)
    return CorrelationInput(s_edges, rr, table, **kw)


def xi_samples_from_lsstypes(corrs, s_slice: slice | None = None, rebin: int | None = None) -> Array:
    """Stack wrapped correlation functions of mocks into an array ``(n_mock, n_s, n_mu)`` for
    :func:`rascaljax.postprocess.sample_covariance`."""
    out = []
    for c in corrs:
        if s_slice is not None:
            c = c.select(s=s_slice)
        if rebin is not None:
            c = c.select(s=slice(0, c.shape[0], rebin))
        out.append(np.asarray(wrap_correlation(c).value(), dtype=float))
    return np.array(out)
