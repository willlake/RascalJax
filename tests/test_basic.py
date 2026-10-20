import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

import rascaljax as rj
from rascaljax.binning import mu_bin_legendre_factors
from rascaljax.kernels import OffsetSampler, r2_cell_probabilities, xi_cell_probabilities


def test_legendre_factors():
    f = mu_bin_legendre_factors(np.linspace(0, 1, 10), 4)
    assert f.shape == (9, 3)
    assert np.isclose(f[:, 0].sum(), 1.0)
    assert np.isclose(f[:, 1].sum(), 0.0, atol=1e-12)
    assert np.isclose(f[:, 2].sum(), 0.0, atol=1e-12)
    # a function xi(mu) = P_2(mu) projects to xi_2 = 1 (up to binning) and xi_0 = 0
    mu_mid = np.linspace(0, 1, 10)
    mu_mid = 0.5 * (mu_mid[1:] + mu_mid[:-1])
    p2 = 0.5 * (3 * mu_mid**2 - 1)
    proj = p2 @ f
    assert abs(proj[1] - 1) < 0.02 and abs(proj[0]) < 0.02


def test_binning_projection_and_filter():
    b = rj.Binning(np.array([20.0, 30, 40]), 5, max_l=2)
    P = b.projection_matrix()
    assert P.shape == (10, 4)
    v = np.ones(10)
    out = b.project_vector(v)
    assert np.allclose(out[0::2], 1.0) and np.allclose(out[1::2], 0.0, atol=1e-12)
    assert list(b.output_filter(skip_s_bins=1)) == [2, 3]
    assert list(b.output_filter(skip_l=1)) == [0, 2]
    assert b.flat_bin(np.array([25.0, 35.0, 45.0]), np.array([0.1, 1.0, 0.5])).tolist() == [0, 9, -1]


def test_xi_table_rules():
    s = np.arange(2.0, 100.0, 2.0)
    mu = np.linspace(0.05, 0.95, 10)
    xi = np.outer((s / 5) ** -1.5, 1 + 0.3 * mu)
    t = rj.XiTable.from_model(s, mu, xi)
    assert np.allclose(t(s[:, None], mu[None, :]), xi, rtol=1e-6)
    # power-law tail beyond the table
    assert np.isclose(t(200.0, 0.5), t(98.0, 0.5) * (200 / 98.0) ** -4, rtol=1e-6)
    # constant below s_min / 2, continuous through the transition
    assert np.isclose(t(0.5, 0.5), t(1.0, 0.5))
    assert abs(t(1.9, 0.5) - t(2.0, 0.5)) < 0.1 * abs(t(2.0, 0.5))
    # mu clamping
    assert np.isclose(t(10.0, 1.0), t(10.0, 0.95))
    lk = t.lookup(0.5, 51)
    assert lk.table.shape == (int(np.ceil(98 / 0.5)) + 1, 51)


def test_kernels():
    cube = r2_cell_probabilities(3, 10.0)
    assert cube.shape == (7, 7, 7) and np.all(cube > 0)
    assert np.allclose(cube, cube[::-1]) and np.allclose(cube, cube.transpose(1, 0, 2))
    s = OffsetSampler.from_cube(cube)
    off = s.offsets_of(np.array([0, 3 * 49 + 3 * 7 + 3, 342]))
    assert off.tolist() == [[-3, -3, -3], [0, 0, 0], [3, 3, 3]]
    xi_cube = xi_cell_probabilities(3, 10.0, lambda r: np.where(r > 0, (np.maximum(r, 1) / 5.0) ** -1.5, 1.0))
    assert np.all(xi_cube >= 0) and xi_cube[3, 3, 3] == xi_cube.max()


def _box_setup(N=3000, L=400.0, seed=0):
    rng = np.random.default_rng(seed)
    cat = rj.Catalog(rng.uniform(0, L, size=(N, 3)), np.ones(N), n_data=N / 2, name="box")
    s_tab = np.arange(1.0, 150.0, 2.0)
    mu_tab = np.linspace(0.1, 0.9, 5)
    table = rj.XiTable.from_model(s_tab, mu_tab, np.repeat(((s_tab / 8) ** -1.8 * np.exp(-((s_tab / 80) ** 2)))[:, None], 5, axis=1))
    s_edges = np.array([20.0, 40.0, 60.0])
    n_mu = 2
    shell = 4 * np.pi / 3 * (s_edges[1:] ** 3 - s_edges[:-1] ** 3) / L**3
    rr = np.repeat((shell / n_mu)[:, None], n_mu, axis=1) * (1 - 1 / N)
    return cat, table, s_edges, rr, L


def test_small_run_rr_and_shapes():
    cat, table, s_edges, rr, L = _box_setup()
    raw = rj.run_cov([rj.Sample(cat)], {(0, 0): rj.CorrelationInput(s_edges, rr, table)}, mode="legendre", max_l=2, N2=4, N3=3, N4=3,
                     n_loops=2, loops_per_sample=1, boxsize=L, xi_cut=100.0, refine_iterations=0, quads_per_batch=50_000, seed=1, log=lambda s: None)
    assert raw.c4.shape == (2, 1, 1, 4, 4) and raw.c3.shape == (1, 2, 1, 1, 4, 4) and raw.c2.shape == (1, 2, 1, 1, 4, 4)
    est = raw.rr_estimate.mean(axis=0)[0]
    assert np.allclose(est, rr.ravel(), rtol=0.15)
    res = rj.post_process(raw, log=lambda s: None)
    assert res["full_theory_covariance"].shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(res["full_theory_covariance"]) > 0)


def test_identical_samples_match_single():
    """Two identical samples on one catalog must give identical blocks for all correlation pairs."""
    cat, table, s_edges, rr, L = _box_setup()
    ci = rj.CorrelationInput(s_edges, rr, table)
    kw = dict(mode="s_mu", N2=4, N3=3, N4=3, n_loops=2, loops_per_sample=1, boxsize=L, xi_cut=100.0, refine_iterations=0,
              quads_per_batch=50_000, seed=1, log=lambda s: None)
    raw1 = rj.run_cov([rj.Sample(cat, name="a")], {(0, 0): ci}, **kw)
    raw2 = rj.run_cov([rj.Sample(cat, name="a"), rj.Sample(cat, name="b")], {(0, 0): ci, (0, 1): ci, (1, 1): ci}, **kw)
    assert raw2.n_corr == 3
    from rascaljax.utils import symmetrized
    for i in range(3):
        for j in range(3):
            # off-diagonal blocks are not symmetrized by construction, diagonal ones are
            assert np.allclose(symmetrized(raw2.c4[:, i, j]), raw1.c4[:, 0, 0])
            assert np.allclose(symmetrized(raw2.c3[0, :, i, j]), raw1.c3[0, :, 0, 0])
            assert np.allclose(symmetrized(raw2.c2[0, :, i, j]), raw1.c2[0, :, 0, 0])


def test_disjoint_samples_vs_two_catalogs():
    """Disjoint samples on one catalog vs the same particles as two catalogs: same expectation (here: same seed
    draws differ, so only check structure and rough agreement of the diagonal)."""
    cat, table, s_edges, rr, L = _box_setup(N=6000)
    z = cat.positions[:, 2]
    half = z < L / 2
    ci = rj.CorrelationInput(s_edges, rr * 0.5, table)
    kw = dict(mode="s_mu", N2=4, N3=3, N4=3, n_loops=2, loops_per_sample=1, boxsize=L, xi_cut=100.0, refine_iterations=0,
              quads_per_batch=50_000, seed=1, log=lambda s: None)
    sA, sB = rj.Sample(cat, half, "A"), rj.Sample(cat, ~half, "B")
    raw = rj.run_cov([sA, sB], {(0, 0): ci, (0, 1): ci, (1, 1): ci}, **kw)
    # the cross-correlation of disjoint z-halves has no 2-point (shot noise) term with the auto-correlations
    assert np.allclose(raw.c2[:, :, 0, 1], 0) and np.allclose(raw.c2[:, :, 1, 2], 0)
    assert not np.allclose(raw.c2[:, :, 0, 0], 0)
    catA = cat.subset(half)
    catB = cat.subset(~half)
    catA.n_data = catB.n_data = cat.n_data / 2
    raw2 = rj.run_cov([rj.Sample(catA, name="A"), rj.Sample(catB, name="B")], {(0, 0): ci, (0, 1): ci, (1, 1): ci}, **kw)
    assert len(raw2.catalog_names) == 2 and raw2.c2.shape[0] == 3
    assert np.allclose(raw2.c2[:, :, 0, 1], 0) and np.allclose(raw2.c2[:, :, 1, 2], 0)
    # the per-catalog-pair c2 axis: (0,0) block from pair (0,0), (1,1) block from pair (1,1)
    assert np.allclose(raw2.c2[0, :, 1, 1], 0) and not np.allclose(raw2.c2[0, :, 0, 0], 0)


def test_alias_sampler_statistics():
    from rascaljax.kernels import OffsetSampler, r2_cell_probabilities

    s = OffsetSampler.from_cube(r2_cell_probabilities(4, 10.0))
    assert np.isclose(s.prob.sum(), 1.0)
    rng = np.random.default_rng(0)
    n = 4_000_000
    h = np.bincount(s.draw(rng, n), minlength=len(s.prob))
    expected = s.prob * n
    sel = expected > 50
    pulls = (h[sel] - expected[sel]) / np.sqrt(expected[sel])
    assert abs(pulls.mean()) < 0.2 and 0.8 < pulls.std() < 1.25


def test_jackknife_run_smoke():
    cat, table, s_edges, rr, L = _box_setup(N=4000)
    rng = np.random.default_rng(3)
    cat.jackknife = rng.integers(0, 5, size=cat.size)
    frac = rng.dirichlet(np.ones(5))
    rr_jack = rr.ravel()[None, :] * frac[:, None]
    xi_jack = rng.normal(0, 0.01, size=rr_jack.shape)
    ci = rj.CorrelationInput(s_edges, rr, table, jack_xi=xi_jack, jack_rr=rr_jack, jack_labels=np.arange(5))
    raw = rj.run_cov([rj.Sample(cat)], {(0, 0): ci}, mode="legendre", max_l=2, N2=4, N3=3, N4=3, n_loops=4, loops_per_sample=1,
                     boxsize=L, xi_cut=100.0, refine_iterations=0, quads_per_batch=50_000, seed=1, log=lambda s: None)
    assert raw.jackknife and raw.c4j.shape == raw.c4.shape and raw.ee.shape == (4, 1, 2, 5, 4)
    assert np.all(np.isfinite(raw.c4j)) and np.any(raw.c4j != 0) and np.any(raw.ee != 0)
    res = rj.post_process(raw, log=lambda s: None)
    assert np.isfinite(res["shot_noise_rescaling"]).all() and res["jackknife_theory_covariance"].shape == (4, 4)


def test_combine_regions_limits():
    from rascaljax.postprocess import combine_regions

    b = rj.Binning(np.array([20.0, 30, 40, 50]), 100, 2)
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6)); C1 = A @ A.T
    A = rng.normal(size=(6, 6)); C2 = A @ A.T
    w = np.ones(b.n_bins)
    # equal weights: average of the estimators -> (C1 + C2) / 4 up to the mu-binning projection error
    comb = combine_regions([C1, C2], [w, w], b)
    assert np.allclose(comb, (C1 + C2) / 4, rtol=0, atol=1e-3 * np.abs(C1 + C2).max())
    # one region dominates
    comb = combine_regions([C1, C2], [w, 1e-9 * w], b)
    assert np.allclose(comb, C1, rtol=0, atol=1e-3 * np.abs(C1).max())
    # s_mu output: exact
    bs = rj.Binning(np.array([20.0, 30, 40]), 3, None)
    A = rng.normal(size=(6, 6)); C = A @ A.T
    assert np.allclose(combine_regions([C, C], [np.ones(6), 3 * np.ones(6)], bs), C * (0.25**2 + 0.75**2))


def _overlap_setup(N=3000, L=300.0, seed=1):
    """Two overlapping z-slabs of a periodic box and their exact catalog-normalized RR counts."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, L, size=(N, 3))
    cat = rj.Catalog(pos, np.ones(N), n_data=N / 2, name="box")
    z = pos[:, 2]
    sA, sB = rj.Sample(cat, z < 0.6 * L, "A"), rj.Sample(cat, z >= 0.4 * L, "B")
    s_edges = np.array([20.0, 40.0, 60.0])
    n_mu = 2
    binning = rj.Binning(s_edges, n_mu, None)
    pairs = cKDTree(pos, boxsize=L).query_pairs(s_edges[-1], output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[i] - pos[j]
    d -= L * np.round(d / L)
    r = np.sqrt((d**2).sum(axis=1))
    b = binning.flat_bin(r, np.abs(d[:, 2]) / r)
    ok = b >= 0
    i, j, b = i[ok], j[ok], b[ok]
    w = cat.weights

    def rr(mS, mT):  # both orderings of each pair, as the kernel's ordered-pair sums
        return np.bincount(b, weights=w[i] * w[j] * (mS[i] * mT[j] + mS[j] * mT[i]), minlength=binning.n_bins).reshape(len(s_edges) - 1, n_mu)

    counts = {(0, 0): rr(sA.membership, sA.membership), (0, 1): rr(sA.membership, sB.membership), (1, 1): rr(sB.membership, sB.membership)}
    kw = dict(mode="s_mu", n_mu=n_mu, N2=4, N3=3, N4=3, n_loops=2, loops_per_sample=1, boxsize=L, xi_cut=100.0, refine_iterations=0,
              quads_per_batch=50_000, seed=1, n_side=11, log=lambda s: None)
    return cat, sA, sB, s_edges, counts, kw


def _power_law_table():
    s_tab = np.arange(1.0, 150.0, 2.0)
    return rj.XiTable.from_model(s_tab, np.linspace(0.1, 0.9, 5), np.repeat(((s_tab / 8) ** -1.8 * np.exp(-((s_tab / 80) ** 2)))[:, None], 5, axis=1))


def test_rr_normalization_conventions():
    """Sample-normalized counts (pycorr convention on the randoms selected for each sample) converted inside run_cov
    must give the same result as catalog-normalized counts passed directly, and the RR self-check must flag a mismatch."""
    cat, sA, sB, s_edges, counts, kw = _overlap_setup()
    table = _power_law_table()
    f = {0: sA.effective_weights.sum(), 1: sB.effective_weights.sum()}
    assert 0.5 < f[0] < 0.7 and 0.5 < f[1] < 0.7
    inp_cat = {k: rj.CorrelationInput(s_edges, v, table, rr_normalization="catalog") for k, v in counts.items()}
    inp_smp = {k: rj.CorrelationInput(s_edges, v / (f[k[0]] * f[k[1]]), table) for k, v in counts.items()}  # default "sample"
    raw_c = rj.run_cov([sA, sB], inp_cat, **kw)
    raw_s = rj.run_cov([sA, sB], inp_smp, **kw)
    assert np.allclose(raw_s.rr_input, raw_c.rr_input)
    assert np.allclose(raw_s.c2, raw_c.c2) and np.allclose(raw_s.c3, raw_c.c3) and np.allclose(raw_s.c4, raw_c.c4)
    assert np.allclose(raw_c.info["rr_factors"], 1.0)
    assert np.allclose(raw_s.info["rr_factors"], [f[0] ** 2, f[0] * f[1], f[1] ** 2])
    assert np.allclose(raw_c.info["rr_check"], 1.0, atol=0.1)
    # sample-normalized counts declared as "catalog": the sampled RR disagrees with the inputs by the weight fractions
    inp_bad = {k: rj.CorrelationInput(s_edges, v / (f[k[0]] * f[k[1]]), table, rr_normalization="catalog") for k, v in counts.items()}
    raw_w = rj.run_cov([sA, sB], inp_bad, **kw)
    assert np.allclose(raw_w.info["rr_check"], [f[0] ** 2, f[0] * f[1], f[1] ** 2], atol=0.1)
    with pytest.raises(ValueError):
        rj.CorrelationInput(s_edges, counts[(0, 0)], table, rr_normalization="parent")


def test_c2_invariant_under_sample_relabelling():
    """With a different xi table per correlation, the 2-point term of a block must not depend on which sample is
    listed first (symmetric average of the two correlations' xi for the shared pair)."""
    cat, sA, sB, s_edges, counts, kw = _overlap_setup()
    s_tab = np.arange(1.0, 150.0, 2.0)
    mu_tab = np.linspace(0.1, 0.9, 5)

    def const(c):
        return rj.XiTable.from_model(s_tab, mu_tab, np.full((len(s_tab), len(mu_tab)), c))

    tabs = {(0, 0): const(0.1), (0, 1): const(0.2), (1, 1): const(0.5)}
    inp1 = {k: rj.CorrelationInput(s_edges, counts[k], tabs[k], rr_normalization="catalog") for k in counts}
    swap = {(0, 0): (1, 1), (0, 1): (0, 1), (1, 1): (0, 0)}  # samples listed as [B, A]: (0,0) is BB, (0,1) is BA, (1,1) is AA
    inp2 = {k: rj.CorrelationInput(s_edges, counts[swap[k]], tabs[swap[k]], rr_normalization="catalog") for k in counts}
    r1 = rj.run_cov([sA, sB], inp1, **kw)
    r2 = rj.run_cov([sB, sA], inp2, **kw)
    c2_1, c2_2 = r1.c2.sum(axis=0), r2.c2.sum(axis=0)
    # the same pair draws enter both runs: the physical AA x BB block ((0, 2) in run 1, (2, 0) in run 2) and the
    # AA x AA block must agree to rounding; before the symmetric xi they differed by (1 + 0.5) / (1 + 0.1)
    assert np.allclose(c2_2[:, 2, 0], c2_1[:, 0, 2], rtol=1e-10)
    assert np.allclose(c2_2[:, 2, 2], c2_1[:, 0, 0], rtol=1e-10)
    assert not np.allclose(c2_1[:, 0, 2], 0)
