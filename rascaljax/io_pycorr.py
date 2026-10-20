"""Adapters from ``pycorr`` two-point estimators (DESI ``allcounts_*.npy`` files) to
:class:`rascaljax.driver.CorrelationInput`, following RascalC's ``pycorr_utils``.

Only needed to *prepare* inputs; the arrays can be saved and used in an environment without ``pycorr``.
"""
from __future__ import annotations

import numpy as np

from .driver import CorrelationInput
from .utils import Array
from .xi_table import XiTable


def _wrap(est):
    """Wrap to |mu| if the mu range starts below 0."""
    if est.edges[1][0] < 0:
        return est.wrap()
    return est


def rebin(est, s_slice: slice | None = None, s_step: int | None = None, mu_step: int | None = None):
    """Select/rebin separations (``s_slice`` in Mpc/h units as integer bin indices for 1 Mpc/h bins) and rebin mu."""
    if s_slice is not None:
        est = est[s_slice]
    if s_step is not None or mu_step is not None:
        est = est[:: (s_step or 1), :: (mu_step or 1)]
    return est


def rr_counts(est) -> Array:
    return np.asarray(est.R1R2.normalized_wcounts(), dtype=float)


def input_xi(est) -> Array:
    """``xi * RR / SS`` (equal to xi pre-reconstruction)."""
    xi = np.asarray(est.corr, dtype=float)
    if hasattr(est, "S1S2"):
        xi = xi * est.R1R2.normalized_wcounts() / est.S1S2.normalized_wcounts()
    return xi


def xi_table_from_pycorr(est) -> XiTable:
    est = _wrap(est)
    s_edges = np.asarray(est.edges[0], dtype=float)
    mu_edges = np.asarray(est.edges[1], dtype=float)
    s_avg = np.asarray(est.sepavg(axis=0), dtype=float)
    mu_avg = np.asarray(est.sepavg(axis=1), dtype=float)
    xi = input_xi(est)
    bad = ~np.isfinite(xi)
    xi = np.where(bad, 0.0, xi)
    s_avg = np.where(np.isfinite(s_avg), s_avg, 0.5 * (s_edges[1:] + s_edges[:-1]))
    mu_avg = np.where(np.isfinite(mu_avg), mu_avg, 0.5 * (mu_edges[1:] + mu_edges[:-1]))
    return XiTable.from_binned(s_edges, mu_edges, xi, s_avg, mu_avg)


def jackknife_realization(jack_est, i):
    """RascalC-framed jackknife realization ``auto + (cross12 + cross21) / 2`` with the total normalization."""
    cls = jack_est.__class__.__bases__[0]
    kw = {}
    for name in jack_est.count_names:
        counts = getattr(jack_est, name)
        kw[name] = counts.auto[i] + 0.5 * (counts.cross12[i] + counts.cross21[i])
        kw[name].wnorm = counts.wnorm
    return cls(**kw)


def jackknife_arrays(jack_est):
    labels = np.asarray(jack_est.realizations)
    reals = [jackknife_realization(jack_est, i) for i in labels]
    xi_jack = np.array([np.asarray(r.corr, dtype=float).ravel() for r in reals])
    rr_jack = np.array([rr_counts(r).ravel() for r in reals])
    return labels, xi_jack, rr_jack


def correlation_input_from_pycorr(counts, xi=None) -> CorrelationInput:
    """Build a :class:`CorrelationInput` from pycorr estimators already rebinned to the covariance binning
    (``counts``) and to the xi-table binning (``xi``, defaults to ``counts``)."""
    wrapped = _wrap(counts)
    s_edges = np.asarray(wrapped.edges[0], dtype=float)
    mu_edges = np.asarray(wrapped.edges[1], dtype=float)
    if not np.allclose(mu_edges, np.linspace(0, 1, len(mu_edges))):
        raise ValueError("Mu bins must be linear between 0 and 1 after wrapping")
    rr = rr_counts(wrapped)
    table = xi_table_from_pycorr(counts if xi is None else xi)
    kw = {}
    if hasattr(wrapped, "realizations"):
        labels, xi_jack, rr_jack = jackknife_arrays(wrapped)
        if not np.allclose(rr_jack.sum(axis=0), rr.ravel()):
            raise ValueError("Jackknife RR counts do not sum to the total RR counts")
        kw = dict(jack_xi=xi_jack, jack_rr=rr_jack, jack_labels=labels)
    return CorrelationInput(s_edges, rr, table, **kw)


def n_data_from_pycorr(est) -> float:
    return float(est.D1D2.size1)


def save_correlation_input(ci: CorrelationInput, filename: str, **extra) -> None:
    """Save a :class:`CorrelationInput` (and extra arrays) to npz, for use without pycorr."""
    d = dict(s_edges=ci.s_edges, rr=ci.rr, rr_normalization=np.array(ci.rr_normalization), xi_s=ci.xi_table.s_vals,
             xi_mu=ci.xi_table.mu_vals, xi=ci.xi_table.xi)
    if ci.xi_table.s_edges is not None:
        d["xi_s_edges"] = ci.xi_table.s_edges
    if ci.jackknife:
        d.update(jack_xi=ci.jack_xi, jack_rr=ci.jack_rr, jack_labels=ci.jack_labels)
    d.update(extra)
    np.savez_compressed(filename, **d)


def load_correlation_input(filename: str) -> tuple[CorrelationInput, dict]:
    with np.load(filename, allow_pickle=False) as f:
        d = dict(f)
    table = XiTable(d["xi_s"], d["xi_mu"], d["xi"], d.get("xi_s_edges"))
    kw = {k: d[k] for k in ("jack_xi", "jack_rr", "jack_labels") if k in d}
    norm = str(d["rr_normalization"]) if "rr_normalization" in d else "sample"
    ci = CorrelationInput(d["s_edges"], d["rr"], table, rr_normalization=norm, **kw)
    extra = {k: v for k, v in d.items() if k not in ("s_edges", "rr", "rr_normalization", "xi_s", "xi_mu", "xi", "xi_s_edges", "jack_xi", "jack_rr", "jack_labels")}
    return ci, extra
