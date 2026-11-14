"""Cell statistics (sky v2 A0): agreement with the existing estimators and known answers."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from tycho.ri.cells import (cell_stats, chi_matrix, conditional_ri_plugin, eta_information, gate, local_ri_of,
                            subsets_of)
from tycho.ri.estimators import estimate_local_ri
from tycho.ri.fourier import single_mode_ri, walsh_coefficients
from tycho.ri.refined import conditional_ri, local_ri
from tycho.ri.variables import VariableSet


def _rows(n=4000, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, d))
    s = 2 * X - 1
    z = 0.9 * s[:, 0] - 0.5 * s[:, 1] + 0.7 * s[:, 0] * s[:, 2] + 0.3 * s[:, 1] * s[:, 2] * s[:, 3]
    p1 = 1 / (1 + np.exp(-z))
    return X, p1


def test_local_ri_matches_estimator_with_soft_output():
    X, p1 = _rows()
    subsets = [(0,), (0, 1), (0, 2), (1, 2, 3), (0, 1, 2, 3)]
    ours = local_ri_of(cell_stats(X, p1, subsets))
    d = X.shape[1]
    Xa = np.concatenate([X, (p1 > 0.5)[:, None].astype(int)], axis=1)
    soft = {d: np.stack([1 - p1, p1], axis=1)}
    est = estimate_local_ri(Xa, [2] * (d + 1), [tuple(A) + (d,) for A in subsets], soft=soft)
    np.testing.assert_allclose(ours, [e.ri_plugin for e in est], atol=1e-10)


def test_data_moments_match_walsh_coefficients():
    X, p1 = _rows()
    d = X.shape[1]
    vs = VariableSet(X=np.concatenate([X, (p1 > 0.5)[:, None].astype(int)], axis=1), cards=[2] * (d + 1),
                     names=[f"x{i}" for i in range(d)] + ["yhat"], soft={d: np.stack([1 - p1, p1], axis=1)})
    for A in [(0,), (0, 2), (1, 2, 3)]:
        cs = cell_stats(X, p1, [A])[0]
        mom = cs.data_moments()
        f = walsh_coefficients(vs, [tuple(A) + (d,)])[0]
        assert mom[tuple(range(len(A)))] == pytest.approx(f, abs=1e-12)
        assert mom[()] == pytest.approx(np.mean(2 * p1 - 1), abs=1e-12)


def _exact_rows(z_fn, d):
    """All 2^d input patterns once, weight 1 each (uniform population), with p1 = sigmoid(z)."""
    X = np.array(list(itertools.product((0, 1), repeat=d)))
    s = 2 * X - 1
    return X, 1 / (1 + np.exp(-z_fn(s)))


def test_logit_walsh_recovers_log_odds_and_additive_has_no_top_term():
    X, p1 = _exact_rows(lambda s: 2 * (1.0 * s[:, 0] + 0.8 * s[:, 1] + 0.6 * s[:, 2]), 3)
    cs = cell_stats(X, p1, [(0, 1, 2)])[0]
    w = cs.logit_walsh()
    assert w[(0,)] == pytest.approx(1.0, abs=1e-9)        # logit/2 = 1.0 s0 + ...
    assert w[(0, 1, 2)] == pytest.approx(0.0, abs=1e-12)  # additive log-odds: no top term
    f = cs.uniform_cell_walsh()
    assert abs(f[(0, 1, 2)]) > 0.05                        # ... but a top moment (saturating link)
    assert local_ri_of([cs])[0] == pytest.approx(0.0, abs=1e-10)
    assert single_mode_ri(f[(0, 1, 2)]) > 0.001


def test_uniform_cell_walsh_removes_imbalance_bias():
    # unbalanced cells (x0 = 1 in 70 % of rows), output mean far from 0: the uncentred moment is biased
    rng = np.random.default_rng(3)
    n = 20000
    x0 = (rng.random(n) < 0.7).astype(int)
    X = x0[:, None]
    m = 0.6 + 0.05 * (2 * x0 - 1)
    p1 = (1 + m) / 2
    cs = cell_stats(X, p1, [(0,)])[0]
    assert cs.uniform_cell_walsh()[(0,)] == pytest.approx(0.05, abs=1e-12)
    assert abs(cs.data_moments()[(0,)] - 0.05) > 0.1      # f̂ = f_c + 2δ f0 with δ = 0.2, f0 ≈ 0.6


def test_conditional_ri_plugin_matches_exact_table_and_removes_spurious_pair():
    # x1 is a noisy copy target; x2 correlated with x1 but unused
    pmf = {}
    rho = 0.8
    for x1, x2 in itertools.product((0, 1), repeat=2):
        pmf[(x1, x2)] = 0.25 * (1 + rho * (2 * x1 - 1) * (2 * x2 - 1))
    X = np.array(list(pmf), dtype=int)
    w = np.array(list(pmf.values()))
    p1 = np.where(X[:, 0] == 1, 0.9, 0.1)
    cs = cell_stats(X, p1, [(1,)], weights=w)[0]
    assert local_ri_of([cs])[0] > 0.05                     # marginal RI of {x2, ŷ}: spurious
    P = cell_stats(X, p1, [(0, 1)], weights=w)[0].table()
    exact = conditional_ri(P, [1, 2])
    assert exact == pytest.approx(0.0, abs=1e-10)
    rows = np.repeat(X, (w * 10000).astype(int), axis=0)
    prow = np.where(rows[:, 0] == 1, 0.9, 0.1)
    assert conditional_ri_plugin(rows, prow, [1], [0]) == pytest.approx(0.0, abs=1e-8)
    assert conditional_ri_plugin(rows, prow, [0], [1]) > 0.1


def test_eta_information_reduces_to_single_mode_and_handles_base_rate():
    f = np.array([0.0, 0.1, 0.5])
    np.testing.assert_allclose(eta_information(f, 0.0), single_mode_ri(f), atol=1e-12)
    assert eta_information(0.04, 0.76) > eta_information(0.04, 0.0)
    assert np.isnan(eta_information(0.5, 0.6))


def test_chi_matrix_is_orthogonal_and_gate():
    for k in (1, 2, 3):
        H = chi_matrix(k)
        np.testing.assert_allclose(H.T @ H, 2 ** k * np.eye(2 ** k))
        assert len(subsets_of(k)) == 2 ** k
    corr, p = gate(np.array([0.0, 0.01, np.nan]), 1000, 0.001)
    assert corr[0] == 0 and corr[1] == pytest.approx(0.01 - 0.0005) and p[2] == 1.0
    assert local_ri(np.ones((2, 2))) == 0.0


def test_conditional_ri_logistic_matches_ipf_on_correlated_tables():
    from tycho.calibration.correlated import logit_teacher, probe_pmf, product_pmf, sample_rows
    rng = np.random.default_rng(5)
    d = 5
    pmf = product_pmf(probe_pmf("markov", 3, 0.8), probe_pmf("equicorr", 2, 0.5))
    w = {(0,): 0.8, (1,): -0.6, (3,): 0.4, (0, 1): 0.5, (0, 1, 3): 0.4, (5,): 1.2, (6,): -0.9}
    t = logit_teacher(d + 2, w)
    X, p1 = sample_rows(pmf, t, d, 3000, rng)
    for A in [(0,), (2,), (4,), (0, 1), (1, 4), (0, 1, 3)]:
        a = conditional_ri_plugin(X, p1, A, range(d), method="ipf", tol=1e-11)
        b = conditional_ri_plugin(X, p1, A, range(d))
        assert b == pytest.approx(a, rel=1e-2, abs=1e-6)


def test_exact_conditional_ri_is_zero_for_unused_inputs_under_correlation():
    from tycho.calibration.correlated import exact_measures, logit_teacher, probe_pmf
    pmf = probe_pmf("markov", 5, 0.9)
    t = logit_teacher(5 + 2, {(0,): 1.0, (1,): 0.5, (0, 1): 0.4, (5,): 1.5, (6,): -1.0})
    ex = {tuple(e["subset"]): e for e in exact_measures(pmf, t, [(2,), (3,), (2, 3), (0, 2), (0,), (0, 1)],
                                                         d_table=5)}
    for A in [(2,), (3,), (2, 3), (0, 2)]:
        assert ex[A]["ri_cond"] == pytest.approx(0.0, abs=1e-10)
        assert ex[A]["f_u"] == pytest.approx(0.0, abs=1e-12)
    assert ex[(2,)]["ri_marg"] > 1e-3 and abs(ex[(2,)]["m_d"]) > 0.01   # spurious under correlation
    # planted terms stay positive, but strong correlation (redundant inputs) makes the conditional values small
    assert ex[(0,)]["ri_cond"] > 0.005 and ex[(0, 1)]["ri_cond"] > 1e-4
    assert ex[(0, 1)]["ri_cond"] < ex[(0, 1)]["ri_marg"]


def test_parity_information_matches_eta_information_on_balanced_cells_and_is_always_defined():
    from tycho.ri.cells import parity_information
    X, p1 = _exact_rows(lambda s: 1.2 * s[:, 0] + 0.7 * s[:, 0] * s[:, 1] - 0.4 * s[:, 2], 3)
    for A in [(0,), (0, 1), (0, 1, 2)]:
        cs = cell_stats(X, p1, [A])[0]
        mom = cs.data_moments()
        top = tuple(range(len(A)))
        assert parity_information(cs) == pytest.approx(float(eta_information(mom[top], mom[()])), abs=1e-12)
    # strongly unbalanced chi_A: the uncentred moment has no valid single-mode reading, the parity MI does
    w = np.array([0.02, 0.48, 0.48, 0.02])
    Xb = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    pb = np.array([0.01, 0.99, 0.99, 0.01])
    cs = cell_stats(Xb, pb, [(0, 1)], weights=w)[0]
    mom = cs.data_moments()
    assert np.isnan(eta_information(mom[(0, 1)], mom[()]))
    assert 0 < parity_information(cs) < np.log(2)


def test_e8_cell_nulls_are_exactly_null():
    import logging
    from tycho.calibration.correlated import e8_cell
    logging.getLogger("tycho").setLevel(logging.ERROR)
    res = e8_cell("markov", 0.9, 4, 1024, 1.0, 4, 0)
    assert res["max_population_value"] < 1e-10
    for lab in ("cond_corr_order2", "cond_corr_order3", "cond_corr_order4", "cond_pair", "marg_pair"):
        assert res[lab]["count"] > 0


def test_projection_information_matches_local_and_conditional_ri_and_has_link_specific_zero_sets():
    from scipy.optimize import brentq
    from tycho.calibration.correlated import logit_teacher, patterns, probe_pmf, uniform_pmf
    from tycho.ri.cells import projection_information
    t = logit_teacher(5, {(0,): .9, (1,): -.5, (0, 1): .6, (0, 1, 3): .4, (2,): .3})
    pmf = probe_pmf("markov", 5, 0.7)
    X = patterns(5)
    csA = cell_stats(X, t.p1, [(0, 1)], weights=pmf)[0]
    assert projection_information(csA, [0, 1]) == pytest.approx(local_ri_of([csA])[0], abs=1e-10)
    csC = cell_stats(X, t.p1, [tuple(range(5))], weights=pmf)[0]
    assert projection_information(csC, [0, 1]) == pytest.approx(
        conditional_ri_plugin(X, t.p1, [0, 1], range(5), weights=pmf), abs=1e-10)

    X3 = patterns(3)
    chi = np.prod(2 * X3 - 1, axis=1)

    def teacher(b):
        return logit_teacher(3, {(0,): 1.0, (1,): 1.0, (2,): 1.0, (0, 1, 2): b})
    bstar = brentq(lambda b: float(((2 * teacher(b).p1 - 1) * chi).mean()), -10, 10)
    for pm in (uniform_pmf(3), probe_pmf("markov", 3, 0.8), probe_pmf("equicorr", 3, 0.6)):
        for wt in ("counts", "uniform"):
            th_null = cell_stats(X3, teacher(0.0).p1, [(0, 1, 2)], weights=pm)[0]
            et_null = cell_stats(X3, teacher(bstar).p1, [(0, 1, 2)], weights=pm)[0]
            assert projection_information(th_null, [0, 1, 2], link="logit", weights=wt) == pytest.approx(0, abs=1e-9)
            assert projection_information(th_null, [0, 1, 2], link="linear", weights=wt) > 0.003
            assert projection_information(et_null, [0, 1, 2], link="linear", weights=wt) == pytest.approx(0, abs=1e-9)
            assert projection_information(et_null, [0, 1, 2], link="logit", weights=wt) > 0.003


def test_projection_information_eta_values_bounded_and_gaussian():
    from tycho.calibration.correlated import patterns
    from tycho.ri.cells import CellStats, eta_information, projection_information
    cs = CellStats(inputs=(0,), counts=np.array([1.0, 1.0]), sums=np.array([-0.99, 0.99]), n=2.0)
    theta = projection_information(cs, [0], link="logit")
    gauss = projection_information(cs, [0], link="linear", eta_value="gaussian")
    bounded = projection_information(cs, [0], link="linear")
    assert theta == pytest.approx(0.662, abs=1e-3)
    assert gauss == pytest.approx(1.959, abs=1e-3)
    assert bounded == pytest.approx(float(eta_information(0.99)), abs=1e-12)
    assert bounded <= np.log(2)
    from tycho.calibration.correlated import probe_pmf
    rng = np.random.default_rng(3)
    X = patterns(3)
    m = np.tanh(rng.normal(size=8) * 1.5)
    cs3 = cell_stats(X, (1 + m) / 2, [(0, 1, 2)])[0]
    fu = cs3.uniform_cell_walsh()[(0, 1, 2)]
    assert projection_information(cs3, [0, 1, 2], link="linear", weights="uniform") == pytest.approx(
        float(eta_information(fu)), abs=1e-10)
    pm = probe_pmf("markov", 3, 0.7)
    csd = cell_stats(X, (1 + m) / 2, [(0, 1, 2)], weights=pm)[0]
    for c in (cs3, csd):
        for A in ([0, 1], [0, 1, 2], [2]):
            g = projection_information(c, A, link="linear", eta_value="gaussian")
            b = projection_information(c, A, link="linear")
            assert g > 1e-6 and b > 1e-6 and g >= b - 1e-12
    chi = np.prod(2 * X - 1, axis=1)
    m0 = 0.45 * (m - np.mean(m * chi) * chi)
    assert np.all(np.abs(m0) < 1)
    cs0 = cell_stats(X, (1 + m0) / 2, [(0, 1, 2)])[0]
    for ev in ("bounded", "gaussian"):
        assert projection_information(cs0, [0, 1, 2], link="linear", weights="uniform",
                                      eta_value=ev) == pytest.approx(0, abs=1e-12)


def test_projection_information_zero_set_of_a_non_top_set_depends_on_the_weights():
    from scipy.optimize import brentq
    from tycho.calibration.correlated import logit_teacher, patterns, probe_pmf
    from tycho.ri.cells import projection_information
    X = patterns(3)
    s0 = 2 * X[:, 0] - 1

    def teacher(a):
        return logit_teacher(3, {(0,): a, (0, 1): 1.0, (0, 2): 0.5, (0, 1, 2): 0.7, (1,): 0.3})
    astar = brentq(lambda a: float(((2 * teacher(a).p1 - 1) * s0).mean()), -5, 5)
    assert abs(astar) > 0.1
    t = teacher(astar)
    cu = cell_stats(X, t.p1, [(0, 1, 2)])[0]
    for link in ("logit", "linear"):
        assert projection_information(cu, [0], link=link, weights="uniform") == pytest.approx(0, abs=1e-9)
    cd = cell_stats(X, t.p1, [(0, 1, 2)], weights=probe_pmf("markov", 3, 0.7))[0]
    assert projection_information(cd, [0], link="logit") > 1e-3


def test_projection_information_singleton_zero_set_is_shared_by_the_links_at_c_d():
    from scipy.optimize import brentq
    from tycho.calibration.correlated import logit_teacher, patterns, probe_pmf
    from tycho.ri.cells import projection_information
    X = patterns(3)
    pm = probe_pmf("markov", 3, 0.7)
    s0 = 2 * X[:, 0] - 1
    rest = X[:, 1] * 2 + X[:, 2]

    def teacher(a):
        return logit_teacher(3, {(0,): a, (0, 1): 0.8, (0, 1, 2): 0.6, (1,): 0.4, (2,): -0.3})

    def score(a):
        t = teacher(a).p1
        cond = np.zeros_like(t)
        for r in range(4):
            sel = rest == r
            cond[sel] = np.sum(pm[sel] * t[sel]) / np.sum(pm[sel])
        return float(np.sum(pm * (t - cond) * s0))
    astar = brentq(score, -5, 5)
    assert abs(astar) > 0.05
    cd = cell_stats(X, teacher(astar).p1, [(0, 1, 2)], weights=pm)[0]
    for link, ev in (("logit", "bounded"), ("linear", "bounded"), ("linear", "gaussian")):
        assert projection_information(cd, [0], link=link, eta_value=ev) == pytest.approx(0, abs=1e-9)
    assert projection_information(cd, [0], link="logit", weights="uniform") > 1e-4
