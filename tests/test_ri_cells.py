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
