"""Tests of tycho.ri.estimators: plug-in RI, bias correction, LRT p-values and their null calibration, parametric
bootstrap, row-bootstrap CIs, Benjamini-Hochberg, candidate subsets and the scan of a VariableSet."""
from __future__ import annotations

import time
from itertools import combinations
from math import comb

import numpy as np
import pytest
from scipy.stats import chi2, kstest

from tycho.analysis.ricurve import RICurve
from tycho.ri.estimators import bh_fdr, candidate_subsets, estimate_local_ri, ri_df, ri_scan
from tycho.ri.loglinear import joint_table, loglinear_table
from tycho.ri.refined import local_ri, local_ri_from_table
from tycho.ri.variables import VariableSet


def sample_codes(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(P.size, size=n, p=P.ravel())
    return np.column_stack(np.unravel_index(idx, P.shape))


def planted_xor(n: int, rng: np.random.Generator, flip: float = 0.1) -> np.ndarray:
    """Six bits: x2 = x0 XOR x1 (flipped w.p. ``flip``), x3..x5 independent with unequal biases."""
    X = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 2, n), np.zeros(n, dtype=int),
                         (rng.random(n) < 0.3), (rng.random(n) < 0.5), (rng.random(n) < 0.7)]).astype(int)
    X[:, 2] = X[:, 0] ^ X[:, 1] ^ (rng.random(n) < flip)
    return X


# ----------------------------------------------------------------------------------------------- basics
def test_ri_df():
    assert ri_df([2, 2, 2]) == 1
    assert ri_df([3, 2, 2]) == 2
    assert ri_df([3, 3]) == 4
    assert ri_df([4, 3, 2, 2]) == 6
    assert ri_df([2, 1, 3]) == 0  # a single-state variable: no interaction term
    assert ri_df([5]) == 0 and ri_df([]) == 0


def test_bh_fdr_hand_computed_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005, 0.01, 0.03, 0.04 -> m p / rank = 0.02, 0.02, 0.04, 0.04 -> running min from the top
    rej, q = bh_fdr(p, 0.03)
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])
    np.testing.assert_array_equal(rej, [True, False, False, True])
    rej2, q2 = bh_fdr(np.array([0.2, np.nan, 0.01, 0.6]), 0.05)  # NaN: not tested, not counted in m
    np.testing.assert_allclose(q2, [0.3, np.nan, 0.03, 0.6], equal_nan=True)
    np.testing.assert_array_equal(rej2, [False, False, True, False])
    _, q3 = bh_fdr(np.array([0.9, 0.95]), 0.05)
    np.testing.assert_allclose(q3, [0.95, 0.95])
    r4, q4 = bh_fdr(np.array([]), 0.05)
    assert r4.shape == (0,) and q4.shape == (0,)
    with pytest.raises(ValueError):
        bh_fdr(np.array([1.2]), 0.05)


def test_bh_fdr_matches_step_up_rule():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = int(rng.integers(1, 40))
        p = np.where(rng.random(m) < 0.3, rng.random(m) * 1e-3, rng.random(m))
        alpha = 0.1
        rej, q = bh_fdr(p, alpha)
        order = np.sort(p)
        below = np.nonzero(order <= alpha * np.arange(1, m + 1) / m)[0]
        k = below[-1] + 1 if below.size else 0
        expected = p <= order[k - 1] if k else np.zeros(m, dtype=bool)
        np.testing.assert_array_equal(rej, expected)
        assert np.all(q >= p - 1e-15)
        assert np.all(np.diff(q[np.argsort(p, kind="stable")]) >= -1e-15)  # q monotone in p


def test_candidate_subsets_counts_match_combinatorics():
    for d in range(1, 9):
        for lo, hi in [(2, 2), (2, 3), (2, 4), (1, 3), (3, 5)]:
            got = candidate_subsets(d, max_order=hi, min_order=lo)
            assert len(got) == sum(comb(d, k) for k in range(lo, min(hi, d) + 1))
            assert got == sorted(got, key=lambda s: (len(s), s))
    d, must = 10, [9]
    got = candidate_subsets(d, max_order=4, must_include=must)
    assert len(got) == comb(9, 1) + comb(9, 2) + comb(9, 3)
    assert all(9 in S for S in got)
    got = candidate_subsets(d, max_order=4, min_order=2, must_include=[0, 9])
    assert len(got) == 1 + comb(8, 1) + comb(8, 2)
    got = candidate_subsets(d, max_order=3, among=[1, 3, 5, 7], must_include=[9])
    assert len(got) == comb(4, 1) + comb(4, 2) and all(set(S) <= {1, 3, 5, 7, 9} for S in got)
    assert candidate_subsets(5, max_order=2, must_include=[0, 1, 2]) == []
    with pytest.raises(ValueError):
        candidate_subsets(4, max_order=2, among=[0, 4])


# ----------------------------------------------------------------------------------------------- plug-in
def test_plugin_matches_local_ri_of_empirical_table_and_formulas():
    rng = np.random.default_rng(1)
    cards = [2, 3, 2, 2, 4]
    X = np.column_stack([rng.integers(0, k, 1500) for k in cards])
    X[:, 3] = (X[:, 0] + X[:, 1] + (rng.random(1500) < 0.2)) % 2  # some structure
    subsets = [(0, 1), (0, 1, 3), (1, 2, 4), (0, 1, 2, 3), (3,), (2, 4)]
    est = estimate_local_ri(X, cards, subsets)
    for S, e in zip(subsets, est):
        assert e.subset == S and e.order == len(S)
        sub_cards = [cards[i] for i in S]
        expected = local_ri(joint_table(X[:, list(S)], sub_cards)) if len(S) > 1 else 0.0
        assert e.ri_plugin == pytest.approx(expected, abs=1e-12)
        assert e.df == ri_df(sub_cards)
        assert e.n_eff == 1500.0
        assert e.g2 == pytest.approx(2 * 1500 * e.ri_plugin)
        if e.df > 0:
            assert e.p_value == pytest.approx(chi2.sf(e.g2, e.df))
            assert e.ri_corrected == pytest.approx(max(0.0, e.ri_plugin - e.df / 3000.0))
        else:
            assert e.p_value == 1.0 and e.ri_corrected == 0.0
    assert est[1].p_value < 1e-10  # the planted (0, 1, 3) parity-like interaction


def test_soft_one_hot_equals_hard_and_soft_reduces_noise():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, (2000, 4))
    X[:, 3] = rng.integers(0, 3, 2000)
    subsets = [(0, 3), (1, 2, 3), (0, 1, 2, 3), (0, 1)]
    hard = estimate_local_ri(X, [2, 2, 2, 3], subsets)
    soft = estimate_local_ri(X, [2, 2, 2, 3], subsets, soft={3: np.eye(3)[X[:, 3]]})
    for h, s in zip(hard, soft):
        assert s.ri_plugin == pytest.approx(h.ri_plugin, abs=1e-13)
        assert s.p_value == pytest.approx(h.p_value, abs=1e-12)
    # a soft column equal to the true (here uniform) conditional removes that variable's sampling noise
    flat = estimate_local_ri(X, [2, 2, 2, 3], subsets, soft={3: np.full((2000, 3), 1 / 3)})
    for h, s in zip(hard, flat):
        if 3 in h.subset:
            assert s.ri_plugin <= h.ri_plugin + 1e-15
            assert s.ri_plugin == pytest.approx(0.0, abs=1e-12)


def test_soft_and_weighted_tables_match_joint_table():
    rng = np.random.default_rng(14)
    n = 700
    X = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 3, n), rng.integers(0, 2, n),
                         rng.integers(0, 2, n)])
    soft = rng.dirichlet([0.5, 0.5], n)
    w = rng.random(n)
    cards = [2, 3, 2, 2]
    subsets = [(0, 3), (1, 3), (0, 1, 3), (0, 2, 3), (0, 1, 2, 3), (0, 1)]
    est = estimate_local_ri(X, cards, subsets, soft={3: soft}, weights=w)
    for S, e in zip(subsets, est):
        sub_soft = {S.index(3): soft} if 3 in S else None
        table = joint_table(X[:, list(S)], [cards[i] for i in S], soft=sub_soft, weights=w)
        assert e.ri_plugin == pytest.approx(local_ri(table), abs=1e-13)


def test_many_state_variables_use_wide_codes():
    rng = np.random.default_rng(15)
    n = 3000
    X = np.column_stack([rng.integers(0, 300, n), rng.integers(0, 2, n), rng.integers(0, 2, n)])
    X[:, 1] = (X[:, 0] % 2) ^ (rng.random(n) < 0.1)
    est = estimate_local_ri(X, [300, 2, 2], [(0, 1), (0, 1, 2), (1, 2)])
    for S, e in zip([(0, 1), (0, 1, 2), (1, 2)], est):
        table = joint_table(X[:, list(S)], [[300, 2, 2][i] for i in S])
        assert e.ri_plugin == pytest.approx(local_ri(table), abs=1e-12)
    assert est[0].df == 299 and est[0].p_value < 1e-10


def test_weights_equal_row_duplication_and_kish_n_eff():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, (800, 4))
    w = rng.integers(1, 4, 800).astype(float)
    dup = np.repeat(X, w.astype(int), axis=0)
    subsets = [(0, 1), (0, 1, 2), (0, 1, 2, 3)]
    a = estimate_local_ri(X, [2] * 4, subsets, weights=w, n_eff=len(dup))
    b = estimate_local_ri(dup, [2] * 4, subsets)
    for ea, eb in zip(a, b):
        assert ea.ri_plugin == pytest.approx(eb.ri_plugin, abs=1e-13)
        assert ea.p_value == pytest.approx(eb.p_value, abs=1e-12)
        assert ea.n_eff == eb.n_eff
    kish = estimate_local_ri(X, [2] * 4, subsets, weights=w)
    assert kish[0].n_eff == pytest.approx(w.sum() ** 2 / np.sum(w ** 2))
    assert kish[0].ri_plugin == pytest.approx(a[0].ri_plugin)
    zero_w = w.copy()
    zero_w[:400] = 0.0  # zero weight == dropping the rows
    c = estimate_local_ri(X, [2] * 4, subsets, weights=zero_w)
    d = estimate_local_ri(X[400:], [2] * 4, subsets, weights=w[400:])
    for ec, ed in zip(c, d):
        assert ec.ri_plugin == pytest.approx(ed.ri_plugin, abs=1e-13) and ec.n_eff == pytest.approx(ed.n_eff)


def test_mixed_shapes_in_one_call_are_batched_consistently():
    rng = np.random.default_rng(4)
    cards = [2, 3, 2, 2, 3, 2]
    X = np.column_stack([rng.integers(0, k, 1000) for k in cards])
    subsets = candidate_subsets(6, max_order=4)
    together = estimate_local_ri(X, cards, subsets)
    for S, e in zip(subsets, together):
        alone = estimate_local_ri(X, cards, [S])[0]
        assert e.ri_plugin == pytest.approx(alone.ri_plugin, abs=1e-14)
    reversed_order = estimate_local_ri(X, cards, subsets[::-1])
    assert [e.ri_plugin for e in reversed_order] == pytest.approx([e.ri_plugin for e in together[::-1]], abs=0)


def test_edge_cases():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        estimate_local_ri(np.zeros((0, 3), dtype=int), [2, 2, 2], [(0, 1)])  # N = 0
    X = rng.integers(0, 2, (500, 3))
    with pytest.raises(ValueError):
        estimate_local_ri(X, [2, 2, 2], [(0, 1)], null="permutation")
    with pytest.raises(ValueError):
        estimate_local_ri(X, [2, 2, 2], [()])
    with pytest.raises(ValueError):
        estimate_local_ri(X, [2, 2, 2], [(0, 3)])
    with pytest.raises(ValueError):
        estimate_local_ri(X, [2, 2, 2], [(0, 1)], weights=np.zeros(500))
    # a card-1 variable: df = 0 -> p = 1, corrected RI = 0
    Xc = np.column_stack([X, np.zeros(500, dtype=int)])
    e = estimate_local_ri(Xc, [2, 2, 2, 1], [(0, 3), (0, 1, 3)])
    assert all(x.df == 0 and x.p_value == 1.0 and x.ri_corrected == 0.0 and x.ri_plugin == 0.0 for x in e)
    # a constant column with two declared states: RI exactly 0, p = 1
    e = estimate_local_ri(Xc, [2, 2, 2, 2], [(0, 3), (0, 1, 3)])
    assert all(x.df > 0 and x.ri_plugin == 0.0 and x.p_value == 1.0 and x.ri_corrected == 0.0 for x in e)
    # a single observation: all mass on one cell
    e = estimate_local_ri(X[:1], [2, 2, 2], [(0, 1), (0, 1, 2)])
    assert all(x.ri_plugin == 0.0 and np.isfinite(x.p_value) for x in e)
    # order-1 subsets are allowed and trivial
    e = estimate_local_ri(X, [2, 2, 2], [(1,)])[0]
    assert (e.ri_plugin, e.df, e.p_value) == (0.0, 0, 1.0)
    # deterministic relation in data: AND is in the closure of the pairwise model -> RI exactly 0
    Xa = X.copy()
    Xa[:, 2] = Xa[:, 0] & Xa[:, 1]
    e = estimate_local_ri(Xa, [2, 2, 2], [(0, 1, 2), (0, 2)])
    assert e[0].ri_plugin == 0.0 and e[0].p_value == 1.0
    assert e[1].ri_plugin > 0.1 and e[1].p_value < 1e-10


# ----------------------------------------------------------------------------------------------- null calibration
@pytest.mark.parametrize("cards,theta", [
    ((2, 2, 2), {(0, 1): 0.5, (1, 2): -0.4, (0, 2): 0.3, (0,): 0.2}),
    ((3, 2, 2), {(0, 1): np.array([[0.4, -0.4], [0.0, 0.3], [-0.2, 0.1]]), (1, 2): 0.5, (2,): -0.3}),
])
def test_chi2_null_calibration_pairwise_only(cards, theta):
    """Pairwise-only structure (no 3-way term): the LRT p-values of the triple must be ~U(0, 1)."""
    P = loglinear_table(theta, list(cards))
    assert local_ri(P) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(6)
    pvals = np.array([estimate_local_ri(sample_codes(P, 2000, rng), list(cards), [(0, 1, 2)])[0].p_value
                      for _ in range(200)])
    fpr = float(np.mean(pvals < 0.05))
    assert 0.01 <= fpr <= 0.10, fpr
    assert kstest(pvals, "uniform").pvalue > 1e-3


def test_parametric_bootstrap_null():
    rng = np.random.default_rng(7)
    P = loglinear_table({(0, 1): 0.6, (1, 2): 0.4}, [2, 2, 2])
    pvals = np.array([estimate_local_ri(sample_codes(P, 600, rng), [2, 2, 2], [(0, 1, 2)],
                                        null="parametric_bootstrap", n_null=99, rng=rng)[0].p_value
                      for _ in range(80)])
    assert np.all((pvals > 0) & (pvals <= 1))
    assert set(np.round(pvals * 100).astype(int)) <= set(range(1, 101))  # (1 + k) / (n_null + 1)
    assert kstest(pvals, "uniform").pvalue > 1e-3
    X = planted_xor(2000, rng)
    a = estimate_local_ri(X, [2] * 6, [(0, 1, 2), (3, 4, 5)], null="parametric_bootstrap", n_null=99,
                          rng=np.random.default_rng(1))
    b = estimate_local_ri(X, [2] * 6, [(0, 1, 2), (3, 4, 5)], null="parametric_bootstrap", n_null=99,
                          rng=np.random.default_rng(1))
    assert a[0].p_value == pytest.approx(1 / 100) and a[0].null == "parametric_bootstrap"
    assert [e.p_value for e in a] == [e.p_value for e in b]  # deterministic given the generator


def test_row_bootstrap_ci_covers_the_truth():
    rng = np.random.default_rng(8)
    P = loglinear_table({(0, 1, 2): 0.35, (0, 1): 0.2}, [2, 2, 2])
    truth = local_ri(P)
    covered = 0
    for _ in range(20):
        e = estimate_local_ri(sample_codes(P, 1500, rng), [2, 2, 2], [(0, 1, 2)], ci=True, n_boot=100,
                              rng=rng)[0]
        assert e.ci_low <= e.ci_high
        covered += e.ci_low <= truth <= e.ci_high
    assert covered >= 15  # nominal 95 %: P(Binomial(20, 0.95) < 15) < 0.3 %
    e = estimate_local_ri(sample_codes(P, 500, rng), [2, 2, 2], [(0, 1)], ci=True, n_boot=50)
    e2 = estimate_local_ri(sample_codes(P, 500, np.random.default_rng(8)), [2, 2, 2], [(0, 1)])
    assert np.isfinite(e[0].ci_low) and np.isnan(e2[0].ci_low)
    # weighted CI, with most rows carrying zero weight (resamples may draw none of the others)
    X = sample_codes(P, 400, rng)
    w = np.zeros(400)
    w[:3] = 1.0
    ew = estimate_local_ri(X, [2, 2, 2], [(0, 1), (0, 1, 2)], weights=w, ci=True, n_boot=40, rng=rng)
    assert all(np.isfinite(x.ci_low) and x.ci_low <= x.ci_high for x in ew)
    assert ew[0].n_eff == pytest.approx(3.0)


# ----------------------------------------------------------------------------------------------- scan
def test_ri_scan_detects_planted_noisy_xor():
    rng = np.random.default_rng(9)
    false_hits, n_unrelated = 0, 0
    for rep in range(10):
        X = planted_xor(4000, rng)
        vs = VariableSet(X=X, cards=[2] * 6, names=[f"v{i}" for i in range(6)],
                         ground_truth=[{"subset": ["v0", "v1", "v2"], "order": 3}])
        scan = ri_scan(vs, max_order=3, alpha=0.01)
        assert len(scan.subsets) == comb(6, 2) + comb(6, 3)
        i = scan.subsets.index((0, 1, 2))
        assert scan.detected[i] and scan.q_values[i] <= 0.01
        assert scan.estimates[i].ri_corrected > 0.2  # population RI of the 10 %-noisy XOR ≈ log 2 − H(0.1)
        others = [j for j in range(len(scan.subsets)) if j != i]
        false_hits += int(scan.detected[others].sum())
        n_unrelated += len(others)
    assert false_hits / n_unrelated < 0.02


def test_ri_scan_options_records_and_ricurve_compatibility():
    rng = np.random.default_rng(10)
    n = 1500
    X = rng.integers(0, 2, (n, 5))
    soft = np.column_stack([1 - X[:, 0] * 0.8 - 0.1, X[:, 0] * 0.8 + 0.1])  # "model output" tracks x0
    vs = VariableSet(X=X, cards=[2] * 5, names=["a", "b", "c", "d", "yhat"], soft={4: soft}, must_include=[4])
    scan = ri_scan(vs, max_order=3)
    assert all(4 in S for S in scan.subsets)  # must_include defaults to vs.must_include
    assert len(scan.subsets) == comb(4, 1) + comb(4, 2)
    hit = scan.subsets.index((0, 4))
    assert scan.detected[hit]
    free = ri_scan(vs, max_order=2, must_include=[])
    assert len(free.subsets) == comb(5, 2)
    with pytest.raises(ValueError, match="max_subsets"):
        ri_scan(vs, max_order=3, must_include=[], max_subsets=10)
    recs = scan.to_records()
    assert recs[hit]["names"] == ["a", "yhat"] and recs[hit]["detected"] is True
    assert set(recs[0]) >= {"subset", "names", "order", "ri_plugin", "ri_corrected", "df", "n_eff", "g2",
                            "p_value", "q_value", "detected", "ci_low", "ci_high", "null"}
    scan2 = ri_scan(vs, max_order=3)
    curve = RICurve.from_scans([0, 100], [scan, scan2], ground_truth=None, meta={"alpha": 0.01})
    assert curve.ri.shape == (len(scan.subsets), 2) and curve.n_eff[0] == n
    empty = ri_scan(VariableSet(X=X[:, :1], cards=[2], names=["a"]), max_order=3)
    assert empty.subsets == [] and empty.q_values.shape == (0,)


def test_scan_timing_quick():
    rng = np.random.default_rng(11)
    X = rng.integers(0, 2, (2048, 16))
    vs = VariableSet(X=X, cards=[2] * 16, names=[f"b{i}" for i in range(16)])
    t0 = time.perf_counter()
    scan = ri_scan(vs, max_order=4)
    assert time.perf_counter() - t0 < 10.0  # 2,500 subsets; ~0.2 s expected
    assert len(scan.subsets) == comb(16, 2) + comb(16, 3) + comb(16, 4)
    assert scan.detected.sum() <= 2  # independent bits


@pytest.mark.slow
def test_scan_performance_target():
    """~20k subsets of 4 binary variables (N = 4096) in ~1-2 s: generous bound for a shared machine."""
    rng = np.random.default_rng(12)
    X = rng.integers(0, 2, (4096, 28))
    vs = VariableSet(X=X, cards=[2] * 28, names=[f"b{i}" for i in range(28)])
    t0 = time.perf_counter()
    scan = ri_scan(vs, min_order=4, max_order=4)
    elapsed = time.perf_counter() - t0
    assert len(scan.subsets) == comb(28, 4)
    assert elapsed < 3.0, elapsed


def test_population_values_match_local_ri_from_table():
    """With a huge sample the plug-in converges to the population local RI of each subset."""
    rng = np.random.default_rng(13)
    P = loglinear_table({(0, 1, 2): 0.5, (1, 3): 0.4, (0,): 0.2}, [2, 2, 2, 2])
    subsets = [S for k in (2, 3) for S in combinations(range(4), k)]
    truth = local_ri_from_table(P, subsets)
    est = estimate_local_ri(sample_codes(P, 200_000, rng), [2] * 4, subsets)
    for t, e in zip(truth, est):
        assert e.ri_corrected == pytest.approx(t, abs=2e-3)
