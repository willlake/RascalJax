"""Post-processing of the raw covariance terms: shot-noise rescaling (from jackknives or mocks),
covariance assembly, precision matrix with the quadratic bias correction, effective number of samples
and convergence checks. A cleaned-up generalization of RascalC's post-processing.

Bin ordering of the assembled matrices: ``(correlation, s, ell)`` for Legendre output (RascalC convention) or
``(correlation, s, mu)``; use :meth:`rascaljax.binning.Binning.to_pycorr_order` to switch to ``(correlation, ell, s)``.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import fmin, minimize

from .binning import Binning
from .driver import RawCovariance
from .utils import Array, symmetrized


# ----------------------------------------------------------------------------------------------
# Jackknife data covariance and disconnected term
# ----------------------------------------------------------------------------------------------
def jackknife_data_covariance(jack_xi: Array, weights: Array) -> Array:
    """Jackknife covariance of the (joint) correlation functions in (s, mu) bins.

    ``jack_xi`` and ``weights`` have shape ``(n_corr, n_jack, n_bins)``; the result has shape
    ``(n_corr * n_bins, n_corr * n_bins)``:
    ``C[a, b] = sum_A w_aA w_bA (xi_aA - mean_a)(xi_bA - mean_b) / (1 - sum_A w_aA w_bA)``.
    """
    n_corr, n_jack, n_bins = jack_xi.shape
    xi = jack_xi.transpose(1, 0, 2).reshape(n_jack, -1)
    w = weights.transpose(1, 0, 2).reshape(n_jack, -1)
    w = w / np.sum(w, axis=0)[None, :]
    mean = np.sum(xi * w, axis=0)
    tmp = w * (xi - mean)
    cov = tmp.T @ tmp
    denom = w.T @ w
    return cov / (1.0 - denom)


def project_joint(cov_smu: Array, binning: Binning, n_corr: int) -> Array:
    """Project a joint covariance in ``(corr, s, mu)`` bins to ``(corr, s, ell)`` (identity for s_mu output)."""
    n_bins = binning.n_bins
    P = binning.projection_matrix()
    big = np.kron(np.eye(n_corr), P)  # block-diagonal projection of the joint vector
    return big.T @ cov_smu @ big


def disconnected_term(raw: RawCovariance) -> Array:
    """The jackknife "disconnected" 4-point term from the ``EE``/``RR`` per-region arrays, for every pair of
    correlations and subsample: shape ``(n_sub, n_corr, n_corr, n_out, n_out)``."""
    n_sub, n_corr, _, n_jack, n_bins = raw.ee.shape
    rr = raw.rr_input  # (n_corr, n_bins)
    w = raw.jack_weights  # (n_corr, n_jack, n_bins)
    out = np.zeros((n_sub, n_corr, n_corr, raw.binning.n_out, raw.binning.n_out))
    for s in range(n_sub):
        for c1 in range(n_corr):
            for c2 in range(n_corr):
                norm = rr[c1][:, None] * rr[c2][None, :] * (1.0 - w[c1].T @ w[c2])
                ee1, rr1 = raw.ee[s, c1, 0], raw.rrj[s, c1, 0]
                ee2, rr2 = raw.ee[s, c2, 1], raw.rrj[s, c2, 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    w1 = np.where(rr1.sum(axis=0) > 0, rr1 / rr1.sum(axis=0), 0.0)
                    w2 = np.where(rr2.sum(axis=0) > 0, rr2 / rr2.sum(axis=0), 0.0)
                diff1 = ee1 - w1 * ee1.sum(axis=0)
                diff2 = ee2 - w2 * ee2.sum(axis=0)
                cx = (diff1.T @ diff2) / norm
                out[s, c1, c2] = raw.binning.project_matrix(cx)
    return symmetrized(out.transpose(0, 1, 3, 2, 4).reshape(n_sub, n_corr * raw.binning.n_out, n_corr * raw.binning.n_out)).reshape(out.shape[0], n_corr, raw.binning.n_out, n_corr, raw.binning.n_out).transpose(0, 1, 3, 2, 4)


# ----------------------------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------------------------
def _joint(blocks: Array) -> Array:
    """``(..., n_corr, n_corr, n, n) -> (..., n_corr * n, n_corr * n)``."""
    *lead, n_corr, _, n, _ = blocks.shape
    return blocks.transpose(*range(len(lead)), len(lead), len(lead) + 2, len(lead) + 1, len(lead) + 3).reshape(*lead, n_corr * n, n_corr * n)


def _filter(cov: Array, keep: Array) -> Array:
    return cov[..., keep[:, None], keep[None, :]]


class CovarianceTerms:
    """The 2-, 3- and 4-point terms of all correlations as joint matrices, with per-catalog shot-noise rescaling.

    ``total(alphas) = c4 + sum_cat alpha_cat c3[cat] + sum_(a,b) alpha_a alpha_b c2[(a,b)]``.
    """

    def __init__(self, raw: RawCovariance | None, jack: bool = False, skip_s_bins=0, skip_l: int = 0, disconnected: bool = True, *, _arrays=None):
        if _arrays is not None:  # internal: build from (c2, c3, c4, n_cat, cat_pairs)
            self.c2, self.c3, self.c4, self.n_cat, self.cat_pairs = _arrays
            self.raw = None
            return
        self.raw = raw
        self.binning = raw.binning
        self.n_corr = raw.n_corr
        self.n_cat = len(raw.catalog_names)
        self.cat_pairs = raw.cat_pairs
        keep1 = self.binning.output_filter(skip_s_bins, skip_l)
        n_out = self.binning.n_out
        self.keep = (np.arange(self.n_corr)[:, None] * n_out + keep1[None, :]).ravel()
        self.n_kept = len(keep1)
        c4 = raw.c4j if jack else raw.c4
        c3 = raw.c3j if jack else raw.c3
        c2 = raw.c2j if jack else raw.c2
        if jack and disconnected:
            c4 = c4 + disconnected_term(raw)
        self.c4 = _filter(_joint(c4), self.keep)  # (n_sub, N, N)
        self.c3 = _filter(_joint(c3), self.keep)  # (n_cat, n_sub, N, N)
        self.c2 = _filter(_joint(c2), self.keep)  # (n_catpair, n_sub, N, N)

    def subset(self, idx: Array) -> "CovarianceTerms":
        """Terms restricted to the given joint-bin indices."""
        return CovarianceTerms(None, _arrays=(self.c2[:, :, idx[:, None], idx[None, :]], self.c3[:, :, idx[:, None], idx[None, :]],
                                              self.c4[:, idx[:, None], idx[None, :]], self.n_cat, self.cat_pairs))

    @property
    def n_sub(self) -> int:
        return self.c4.shape[0]

    @property
    def size(self) -> int:
        return self.c4.shape[-1]

    def total(self, alphas, subsamples: bool = False) -> Array:
        alphas = np.broadcast_to(np.asarray(alphas, dtype=float), (self.n_cat,))
        c4, c3, c2 = (self.c4, self.c3, self.c2) if subsamples else (self.c4.mean(axis=0), self.c3.mean(axis=1), self.c2.mean(axis=1))
        out = c4.copy()
        for ic in range(self.n_cat):
            out += alphas[ic] * c3[ic]
        for ip, (a, b) in enumerate(self.cat_pairs):
            out += alphas[a] * alphas[b] * c2[ip]
        return out

    def term_means(self):
        return self.c2.mean(axis=1), self.c3.mean(axis=1), self.c4.mean(axis=0)


# ----------------------------------------------------------------------------------------------
# Precision matrix, N_eff, checks (ports of RascalC)
# ----------------------------------------------------------------------------------------------
def compute_D_precision_matrix(partial_cov: Array, full_cov: Array) -> tuple[Array, Array]:
    """Quadratic-order bias correction ``D`` and the corrected precision matrix (O'Connell & Eisenstein 2019)."""
    n_samples = len(partial_cov)
    n_bins = len(full_cov)
    sum_partial = np.sum(partial_cov, axis=0)
    tmp = 0.0
    for i in range(n_samples):
        c_excl = (sum_partial - partial_cov[i]) / (n_samples - 1)
        tmp += np.linalg.inv(c_excl) @ partial_cov[i]
    D = (n_samples - 1.0) / n_samples * (-np.eye(n_bins) + tmp / n_samples)
    prec = (np.eye(n_bins) - D) @ np.linalg.inv(full_cov)
    return D, prec


def compute_N_eff(D: Array, log: Callable = print) -> float:
    n_bins = len(D)
    sign, logdet = np.linalg.slogdet(D)
    if sign < 0:
        log("N_eff is negative; setting to zero")
        return 0.0
    D_value = sign * np.exp(logdet / n_bins)
    n_eff = (n_bins + 1.0) / D_value + 1.0
    log(f"Effective number of samples N_eff = {n_eff:.4e}")
    return n_eff


def check_eigval_convergence(c2: Array, c4: Array, alpha: float = 1.0, kind: str = "", log: Callable = print) -> bool:
    """RascalC eigenvalue convergence tests of the 4-point term relative to the 2-point term."""
    ok = True
    kind = (kind + " ") if kind else ""
    min_c2 = np.min(np.linalg.eigvalsh(c2))
    min_c4 = np.min(np.linalg.eigvalsh(c4))
    log(f"{kind}convergence check: min eigenvalue of C2 = {min_c2:.2e}, of C4 = {min_c4:.2e}")
    inv_sqrt_c2 = np.linalg.inv(sqrtm(c2))
    min_comb = np.min(np.linalg.eigvalsh(inv_sqrt_c2 @ c4 @ inv_sqrt_c2))
    log(f"{kind}convergence check: min eigenvalue of C2^-1/2 C4 C2^-1/2 = {min_comb:.3f} (should be > {-alpha**2:.2f})")
    if min_comb <= -alpha**2:
        log(f"WARNING: {kind}4-point term has not converged by the weaker eigenvalue test")
        ok = False
    if min_c4 < -min_c2 * alpha**2:
        log(f"WARNING: {kind}4-point term has not converged by the stronger eigenvalue test")
        ok = False
    return ok


def check_positive_definite(cov: Array, log: Callable = print, strict: bool = False) -> bool:
    """Check positive definiteness; warn (or raise with ``strict=True``) if the covariance is not."""
    ev = np.linalg.eigvalsh(cov)
    if np.any(ev <= 0):
        msg = f"The covariance is not positive definite (min eigenvalue {ev.min():.3e}, max {ev.max():.3e}); insufficient convergence"
        if strict:
            raise ValueError(msg)
        log("WARNING: " + msg)
        return False
    return True


# ----------------------------------------------------------------------------------------------
# Shot-noise rescaling fit
# ----------------------------------------------------------------------------------------------
def neg_log_likelihood(alphas, terms: CovarianceTerms, target: Array) -> float:
    """``tr(Psi C_target) - log det Psi`` with the bias-corrected precision ``Psi`` of the model at ``alphas``.
    Infinite for negative alphas or non-positive precision (as RascalC)."""
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    if np.any(alphas < 0):
        return np.inf
    full = terms.total(alphas)
    partial = terms.total(alphas, subsamples=True)
    try:
        _, psi = compute_D_precision_matrix(partial, full)
    except np.linalg.LinAlgError:
        return np.inf
    sign, logdet = np.linalg.slogdet(psi)
    if sign < 0:
        return np.inf
    return float(np.trace(psi @ target) - logdet)


def fit_shot_noise_rescaling(terms: CovarianceTerms, target: Array, method: str = "joint", log: Callable = print) -> Array:
    """Fit the per-catalog shot-noise rescaling(s) to a target (jackknife or mock) covariance.

    ``method="joint"`` fits all catalogs' alphas at once on the full joint covariance;
    ``method="separate"`` fits each catalog's alpha on the auto-covariance of its own sample
    auto-correlations only (as RascalC multi-tracer).
    """
    n_cat = terms.n_cat
    if n_cat == 1 or method == "joint":
        if n_cat == 1:
            best = fmin(lambda a: neg_log_likelihood(a, terms, target), np.ones(1), disp=False)
        else:
            res = minimize(lambda a: neg_log_likelihood(a, terms, target), np.ones(n_cat), method="Nelder-Mead")
            best = res.x
        best = np.atleast_1d(best)
        log(f"Shot-noise rescaling(s): {best}")
        return best
    if method != "separate":
        raise ValueError("method must be 'joint' or 'separate'")
    raw = terms.raw
    best = np.ones(n_cat)
    n = terms.n_kept
    for ic in range(n_cat):
        sel = [k for k, (a, b) in enumerate(raw.correlations) if raw.sample_catalog[a] == ic and raw.sample_catalog[b] == ic]
        if not sel:
            log(f"Catalog {ic}: no auto-correlation, keeping alpha = 1")
            continue
        idx = np.concatenate([np.arange(k * n, (k + 1) * n) for k in sel])
        sub = terms.subset(idx)
        tgt = target[idx[:, None], idx[None, :]]

        def f(a, ic=ic, sub=sub, tgt=tgt):
            al = np.ones(n_cat)
            al[ic] = a[0]
            return neg_log_likelihood(al, sub, tgt)

        best[ic] = fmin(f, np.ones(1), disp=False)[0]
        log(f"Catalog {ic} ({raw.catalog_names[ic]}): shot-noise rescaling {best[ic]:.6f}")
    return best


# ----------------------------------------------------------------------------------------------
# Mock sample covariance
# ----------------------------------------------------------------------------------------------
def sample_covariance(xi_samples: Array, binning: Binning) -> Array:
    """Sample covariance of correlation functions from mocks.

    ``xi_samples`` has shape ``(n_corr, n_mock, n_s, n_mu)`` in the covariance binning (mu in [0, 1]);
    they are projected to the output bins (multipoles) and the joint ``(corr, s, ell)`` covariance is returned.
    """
    xi_samples = np.asarray(xi_samples, dtype=float)
    n_corr, n_mock = xi_samples.shape[:2]
    v = xi_samples.reshape(n_corr, n_mock, -1) @ binning.projection_matrix()  # (n_corr, n_mock, n_out)
    v = v.transpose(1, 0, 2).reshape(n_mock, -1)
    return np.cov(v.T)


# ----------------------------------------------------------------------------------------------
# Comparison measures (Rashkovetskyi et al. 2023, 2025)
# ----------------------------------------------------------------------------------------------
def rms_eig_inv(C1: Array, C2: Array) -> float:
    """``R_inv``: RMS of the eigenvalues of ``C1^-1 C2 - 1``."""
    t = np.linalg.inv(C1) @ C2 - np.eye(len(C2))
    return float(np.sqrt(np.trace(t @ t) / len(C2)))


def kl_divergence(C1: Array, C2: Array) -> float:
    m = np.linalg.inv(C1) @ C2
    return float((np.trace(m) - len(C2) - np.linalg.slogdet(m)[1]) / 2)


def chi2_reduced(C1: Array, C2: Array) -> float:
    return float(np.trace(np.linalg.inv(C1) @ C2) / len(C2))


def compare_covariances(C1: Array, C2: Array, log: Callable = print, label: str = "") -> dict:
    res = {"R_inv": (rms_eig_inv(C1, C2), rms_eig_inv(C2, C1)), "D_KL": (kl_divergence(C1, C2), kl_divergence(C2, C1)),
           "chi2_red-1": (chi2_reduced(C1, C2) - 1, chi2_reduced(C2, C1) - 1)}
    log(f"{label}R_inv = {res['R_inv'][0]:.2e}, {res['R_inv'][1]:.2e}; D_KL = {res['D_KL'][0]:.2e}, {res['D_KL'][1]:.2e}; chi2_red-1 = {res['chi2_red-1'][0]:.2e}, {res['chi2_red-1'][1]:.2e}")
    return res


def convergence_check_extra(partial_cov: Array, log: Callable = print) -> dict:
    """Compare the covariances averaged over two halves of the subsamples (two splittings)."""
    n = len(partial_cov)
    h = n // 2
    res = {"split1": compare_covariances(partial_cov[:h].mean(axis=0), partial_cov[h:].mean(axis=0), log, "First half vs second half: "),
           "split2": compare_covariances(partial_cov[0::2].mean(axis=0), partial_cov[1::2].mean(axis=0), log, "Even vs odd subsamples: ")}
    return res


# ----------------------------------------------------------------------------------------------
# Main post-processing
# ----------------------------------------------------------------------------------------------
def post_process(raw: RawCovariance, *, mock_cov: Array | None = None, alphas=None, skip_s_bins=0, skip_l: int = 0,
                 disconnected: bool = True, alpha_method: str = "joint", strict: bool = False, out_file: str | None = None,
                 log: Callable = print) -> dict:
    """Assemble the final covariance from raw terms.

    The shot-noise rescaling is fitted to the jackknife covariance if the raw terms contain jackknife estimates,
    or to ``mock_cov`` (a joint covariance of all correlations in the output bins, e.g. from :func:`sample_covariance`)
    if given; otherwise the given ``alphas`` (default 1 for every catalog) are used.

    With ``strict=True`` a non-positive-definite covariance raises instead of warning.

    Returns a dictionary (also saved as ``out_file`` if given) with the full covariance, precision matrix, N_eff,
    the per-subsample covariances, the shot-noise rescalings and, with jackknives, the jackknife data and theory
    covariances.
    """
    binning = raw.binning
    n_corr = raw.n_corr
    terms = CovarianceTerms(raw, jack=False, skip_s_bins=skip_s_bins, skip_l=skip_l)
    keep = terms.keep
    out = {"correlations": np.array(raw.correlations), "sample_names": np.array(raw.sample_names), "catalog_names": np.array(raw.catalog_names),
           "s_edges": binning.s_edges, "n_mu": binning.n_mu, "max_l": -1 if binning.max_l is None else binning.max_l,
           "skip_s_bins": np.array(skip_s_bins), "skip_l": skip_l, "kept_bins": keep}

    if alphas is not None:
        alphas = np.broadcast_to(np.asarray(alphas, dtype=float), (terms.n_cat,)).copy()
    if raw.jackknife and mock_cov is None:
        log("Fitting the shot-noise rescaling to the jackknife covariance")
        jterms = CovarianceTerms(raw, jack=True, skip_s_bins=skip_s_bins, skip_l=skip_l, disconnected=disconnected)
        data_cov = project_joint(jackknife_data_covariance(raw.jack_xi, raw.jack_weights), binning, n_corr)[keep[:, None], keep[None, :]]
        c2j, c3j, c4j = jterms.term_means()
        ok = check_eigval_convergence(c2j.sum(axis=0), c4j, kind="Jackknife", log=log)
        alphas = fit_shot_noise_rescaling(jterms, data_cov, alpha_method, log)
        if ok and np.any(alphas < 1):
            check_eigval_convergence(c2j.sum(axis=0), c4j, float(np.min(alphas)), kind="Jackknife", log=log)
        jack_cov = jterms.total(alphas)
        jack_partial = jterms.total(alphas, subsamples=True)
        _, jack_prec = compute_D_precision_matrix(jack_partial, jack_cov)
        out.update(jackknife_data_covariance=data_cov, jackknife_theory_covariance=jack_cov, jackknife_theory_precision=jack_prec,
                   individual_theory_jackknife_covariances=jack_partial)
    elif mock_cov is not None:
        log("Fitting the shot-noise rescaling to the mock sample covariance")
        mock_cov = np.asarray(mock_cov)[keep[:, None], keep[None, :]]
        c2, c3, c4 = terms.term_means()
        ok = check_eigval_convergence(c2.sum(axis=0), c4, kind="Full", log=log)
        alphas = fit_shot_noise_rescaling(terms, mock_cov, alpha_method, log)
        out["mock_covariance"] = mock_cov
    elif alphas is None:
        alphas = np.ones(terms.n_cat)
        log("No jackknife or mock reference: using shot-noise rescaling 1 for every catalog")

    c2, c3, c4 = terms.term_means()
    check_eigval_convergence(c2.sum(axis=0), c4, float(np.min(alphas)), kind="Full", log=log)
    full_cov = terms.total(alphas)
    out["positive_definite"] = check_positive_definite(full_cov, log, strict)
    partial = terms.total(alphas, subsamples=True)
    D, prec = compute_D_precision_matrix(partial, full_cov)
    n_eff = compute_N_eff(D, log)
    out.update(full_theory_covariance=full_cov, full_theory_precision=prec, full_theory_D_matrix=D, N_eff=n_eff,
               individual_theory_covariances=partial, shot_noise_rescaling=alphas,
               c2_terms=c2, c3_terms=c3, c4_term=c4)
    log("Extra convergence check on the full covariance:")
    out["convergence"] = convergence_check_extra(partial, log)
    if out_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        np.savez_compressed(out_file, **{k: v for k, v in out.items() if k != "convergence"})
        log(f"Saved post-processed results to {out_file}")
    return out


# ----------------------------------------------------------------------------------------------
# Combination of regions (e.g. NGC + SGC), Appendix B.2 of Rashkovetskyi et al. 2025
# ----------------------------------------------------------------------------------------------
def combine_regions(covs: list[Array], rr_weights: list[Array], binning: Binning, n_corr: int = 1) -> Array:
    """Covariance of the correlation function(s) of the union of independent regions whose pair counts are summed.

    ``covs[r]`` is the covariance of region ``r`` in the output bins (``(corr, s, ell)`` or ``(corr, s, mu)`` ordering),
    ``rr_weights[r]`` the region's *unnormalized* RR counts in flat (s, mu) bins (shape ``(n_corr, n_bins)`` or
    ``(n_bins,)``). The combined (s, mu)-binned estimator is the RR-weighted average of the regions', so the Legendre
    covariance combines through the partial derivatives projected back and forth between mu bins and multipoles.
    Correlations between regions are neglected.
    """
    n_bins, n_out = binning.n_bins, binning.n_out
    weights = [np.asarray(w, dtype=float).reshape(n_corr, n_bins) for w in rr_weights]
    total = np.sum(weights, axis=0)
    out = np.zeros((n_corr * n_out, n_corr * n_out))
    if binning.legendre:
        G = binning.inverse_legendre_factors()  # (n_ell, n_mu)
        F = binning.legendre_factors()  # (n_mu, n_ell)
    for cov, w in zip(covs, weights):
        wn = w / total  # (n_corr, n_bins)
        pd = np.zeros((n_corr, n_out, n_corr, n_out))
        for c in range(n_corr):
            wc = wn[c].reshape(binning.n_s, binning.n_mu)
            if binning.legendre:
                # d xi_ell'(s) / d xi_ell(s) of region r: sum_mu G[ell, mu] w[s, mu] F[mu, ell'] (diagonal in s)
                blk = np.einsum("il,sl,lj->sij", G, wc, F)  # (s, ell_in, ell_out)
                full = np.zeros((binning.n_s, binning.n_ell, binning.n_s, binning.n_ell))
                for i_s in range(binning.n_s):
                    full[i_s, :, i_s, :] = blk[i_s]
                pd[c, :, c, :] = full.reshape(n_out, n_out)
            else:
                pd[c, :, c, :] = np.diag(wc.ravel())
        pd = pd.reshape(n_corr * n_out, n_corr * n_out)
        out += pd.T @ cov @ pd
    return out
