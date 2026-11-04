"""Tests of tycho.calibration: planted generators (exact ground truth), samplers, schedules, dichotomized Gaussians and
target-RI planting; the suite's helpers, configuration, end-to-end quick run (files and sane numbers), determinism
across n_jobs, cache/resume; and the report (CALIBRATION.md, figures, calibration.json, min_effect lookup)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from tycho.calibration import (
    EXPERIMENTS,
    CalibrationConfig,
    dichotomized_gaussian_table,
    equicorrelation,
    gaussian_binarized,
    interpolate_min_effect,
    latent_threshold,
    load_calibration,
    load_results,
    planted_binary,
    planted_binary_batch,
    random_interactions,
    recommended_min_effect,
    run_calibration,
    sample_table,
    spin_features,
    switch_on,
    theta_for_local_ri,
    true_local_ri,
    write_report,
)
from tycho.calibration.suite import ZERO_TOL, _chunks, _interp_mde, _isotonic, _rng, _theory_mde
from tycho.ri.loglinear import loglinear_table, marginal, mutual_information, theta_binary
from tycho.ri.refined import local_ri


def h_binary(p: float) -> float:
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------------------------------------------------------------------------- planted tables
def test_planted_binary_spin_convention_and_reference():
    theta = {(0, 1, 2): 0.4, (0, 1): -0.3}
    P = planted_binary(3, theta, biases={2: 0.2})
    np.testing.assert_allclose(P, loglinear_table({**theta, (2,): 0.2}, [2, 2, 2]), rtol=0, atol=1e-15)
    assert P.shape == (2, 2, 2) and P.sum() == pytest.approx(1.0, abs=1e-15)
    # spin coding: theta contributes theta * s0 * s1 with s = 2x - 1
    t = 0.37
    P2 = planted_binary(2, {(0, 1): t})
    assert P2[1, 1] / P2[0, 1] == pytest.approx(math.exp(2 * t), rel=1e-12)
    P1 = planted_binary(1, {}, biases={0: t})
    assert P1[1] / P1[0] == pytest.approx(math.exp(2 * t), rel=1e-12)
    # top Moebius (0/1) coordinate = 2^d * spin coupling, whatever the lower-order terms
    P3 = planted_binary(3, {(0, 1, 2): t, (1, 2): 0.5}, biases={0: -0.4})
    assert theta_binary(P3)[(0, 1, 2)] == pytest.approx(8 * t, rel=1e-10)
    # keys are normalised and duplicates add up
    np.testing.assert_allclose(planted_binary(2, {(0, 1): 0.2, (1, 0): 0.3}), planted_binary(2, {(0, 1): 0.5}))
    for bad in ({(0, 3): 1.0}, {(): 1.0}, {(0, 1): float("nan")}):
        with pytest.raises(ValueError):
            planted_binary(3, bad)


def test_planted_subset_has_ri_and_unrelated_subsets_do_not():
    """Pure 3-way {0,1,2} and pair {3,4}, no biases: exact closed forms, and exact zeros elsewhere."""
    a, b = 0.4, 0.3
    P = planted_binary(6, {(0, 1, 2): a, (3, 4): b})
    planted = true_local_ri(P, [(0, 1, 2), (3, 4)])
    # the parity class s0 s1 s2 = +1 has probability sigma(2a) and uniform margins otherwise
    assert planted[0] == pytest.approx(math.log(2) - h_binary(sigmoid(2 * a)), rel=1e-10)
    assert planted[1] == pytest.approx(math.log(2) - h_binary(sigmoid(2 * b)), rel=1e-10)
    unrelated = [(0, 1), (0, 2), (1, 2),  # proper subsets of a pure synergy
                 (0, 3), (2, 5), (4, 5), (0, 1, 3), (3, 4, 5), (0, 3, 4), (0, 1, 2, 3), (0, 1, 2, 5), (5, 0)]
    np.testing.assert_allclose(true_local_ri(P, unrelated), 0.0, atol=1e-12)
    assert true_local_ri(P, [(3,)])[0] == 0.0
    # marginalisation can create interactions: a bias on the summed-out variable induces a pairwise term
    Pb = planted_binary(3, {(0, 1, 2): a}, biases={2: 0.5})
    assert true_local_ri(Pb, [(0, 1)])[0] > 1e-3


def test_true_local_ri_matches_definitions():
    P = planted_binary(4, random_interactions(4, [1, 2, 3, 4], 0.5, np.random.default_rng(3)))
    ri = true_local_ri(P, [(0, 2), (1, 2, 3), (0, 1, 2, 3)])
    assert ri[0] == pytest.approx(mutual_information(P, [0], [2]), rel=1e-12)
    assert ri[1] == pytest.approx(local_ri(marginal(P, (1, 2, 3))), rel=1e-12)
    assert ri[2] == pytest.approx(local_ri(P), rel=1e-12)


def test_spin_features_and_batch_match_reference():
    F = spin_features(2, [(), (0,), (1,), (0, 1)])
    np.testing.assert_array_equal(F, [[1, 1, 1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]])
    rng = np.random.default_rng(1)
    subsets = [(0,), (2,), (0, 1), (1, 2), (0, 1, 2)]
    thetas = rng.normal(0, 1.0, (5, len(subsets)))
    batch = planted_binary_batch(3, subsets, thetas)
    for r in range(5):
        ref = planted_binary(3, dict(zip(subsets, thetas[r])))
        np.testing.assert_allclose(batch[r], ref, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        planted_binary_batch(3, subsets, thetas[:, :3])


def test_random_interactions():
    rng = np.random.default_rng(0)
    th = random_interactions(5, [1, 3], 0.7, rng)
    assert list(th) == [(i,) for i in range(5)] + list(combinations(range(5), 3))  # graded-lex order
    assert th == random_interactions(5, [3, 1], 0.7, np.random.default_rng(0))  # deterministic, order-insensitive
    assert all(v == 0.0 for v in random_interactions(4, [2], 0.0, rng).values())
    big = np.array([v for _ in range(40) for v in random_interactions(8, [2, 3], 0.5, rng).values()])
    assert big.std() == pytest.approx(0.5, rel=0.05) and abs(big.mean()) < 0.03
    for orders, scale in (([0], 1.0), ([6], 1.0), ([2], -1.0), ([2], float("inf"))):
        with pytest.raises(ValueError):
            random_interactions(5, orders, scale, rng)


def test_sample_table_marginals_support_and_determinism():
    rng = np.random.default_rng(42)
    P = rng.random((2, 3, 2))
    P[1, 2, 0] = 0.0
    P /= P.sum()
    n = 200_000
    X = sample_table(P * 7.0, n, np.random.default_rng(5))  # unnormalised input is normalised
    assert X.shape == (n, 3) and X.dtype == np.int64
    assert X.min() == 0 and np.all(X.max(axis=0) == [1, 2, 1])
    counts = np.zeros(P.shape)
    np.add.at(counts, tuple(X.T), 1)
    assert counts[1, 2, 0] == 0  # a zero-probability cell is never drawn
    pos = P > 0
    gof = stats.chisquare(counts[pos], n * P[pos])
    assert gof.pvalue > 1e-4
    for axis in range(3):  # one-variable marginals within 5 binomial SE
        freq = np.bincount(X[:, axis], minlength=P.shape[axis]) / n
        p = marginal(P, (axis,))
        assert np.all(np.abs(freq - p) <= 5 * np.sqrt(p * (1 - p) / n))
    np.testing.assert_array_equal(X, sample_table(P, n, np.random.default_rng(5)))
    assert sample_table(P, 0, rng).shape == (0, 3)
    with pytest.raises(ValueError):
        sample_table(-P, 10, rng)


# ----------------------------------------------------------------------------------------------- target-RI planting
def test_theta_for_local_ri_plants_exact_target():
    rng = np.random.default_rng(7)
    lower = [S for o in (1, 2) for S in combinations(range(3), o)]
    bg = rng.normal(0, 0.3, (20, len(lower)))
    theta, tables, ri = theta_for_local_ri(0.01, 3, lower, bg)
    assert np.all(theta > 0)
    np.testing.assert_allclose(ri, 0.01, rtol=1e-9)
    np.testing.assert_allclose([local_ri(T) for T in tables], 0.01, rtol=1e-9)
    for r in range(3):  # the tables are exactly the planted ones: background plus the solved top coupling
        ref = planted_binary(3, {**dict(zip(lower, bg[r])), (0, 1, 2): theta[r]})
        np.testing.assert_allclose(tables[r], ref, rtol=1e-12, atol=1e-15)
        assert theta_binary(tables[r])[(0, 1, 2)] == pytest.approx(8 * theta[r], rel=1e-9)
    theta0, _, ri0 = theta_for_local_ri(0.0, 3, lower, bg[:2])
    assert np.all(theta0 == 0) and np.all(ri0 < 1e-15)
    # a target above the largest attainable RI (here > 3 log 2) is flagged, not silently capped
    theta_bad, _, _ = theta_for_local_ri(5.0, 3, lower, bg[:2])
    assert np.all(np.isnan(theta_bad))
    with pytest.raises(ValueError):
        theta_for_local_ri(0.01, 3, [(0, 1, 2)], np.zeros((1, 1)))  # background must be of lower order


# ----------------------------------------------------------------------------------------------- schedules
@pytest.mark.parametrize("width", [0.0, 0.5, 3.0])
def test_switch_on_is_a_monotone_ramp(width):
    t = np.linspace(-40, 60, 2001)
    up = switch_on(t, 10.0, width, 2.0)
    assert np.all(np.diff(up) >= 0)
    assert np.all(np.diff(switch_on(t, 10.0, width, -2.0)) <= 0)
    assert switch_on(np.array([10.0]), 10.0, width, 2.0)[0] == pytest.approx(1.0)
    assert up[0] == pytest.approx(0.0, abs=1e-6) and up[-1] == pytest.approx(2.0, rel=1e-6)
    if width > 0:  # 10 % -> 90 % rise time 2 ln 9 * width
        t10 = 10.0 + width * math.log(1 / 9)
        t90 = 10.0 + width * math.log(9)
        np.testing.assert_allclose(switch_on(np.array([t10, t90]), 10.0, width, 1.0), [0.1, 0.9], rtol=1e-12)
    else:
        np.testing.assert_array_equal(switch_on(np.array([9.999, 10.0, 10.001]), 10.0, 0.0, 1.0), [0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        switch_on(t, 0.0, -1.0, 1.0)


# ----------------------------------------------------------------------------------------------- Gaussian latents
def test_gaussian_binarized_thresholds():
    rng = np.random.default_rng(3)
    n, rho = 100_000, 0.5
    C = equicorrelation(4, rho)
    B = gaussian_binarized(C, n, rng, "median")
    assert B.shape == (n, 4) and B.dtype == np.int64
    np.testing.assert_array_equal(B.sum(axis=0), n // 2)  # continuous data, even n: exactly half above the median
    p11 = np.mean(B[:, 0] & B[:, 1])
    assert abs(p11 - (0.25 + math.asin(rho) / (2 * math.pi))) < 5 * math.sqrt(0.25 / n)
    Bq = gaussian_binarized(C, n, rng, "quantile", q=0.8)
    np.testing.assert_allclose(Bq.mean(axis=0), 0.2, atol=2.0 / n)
    Bz = gaussian_binarized(C, n, rng, "zero", mean=0.5)
    p = stats.norm.cdf(0.5)
    assert np.all(np.abs(Bz.mean(axis=0) - p) < 5 * math.sqrt(p * (1 - p) / n))
    np.testing.assert_array_equal(gaussian_binarized(C, 50, np.random.default_rng(1)),
                                  gaussian_binarized(C, 50, np.random.default_rng(1)))
    with pytest.raises(ValueError):
        gaussian_binarized(np.array([[1.0, 0.5], [0.4, 1.0]]), 10, rng)  # not symmetric
    with pytest.raises(ValueError):
        gaussian_binarized(np.array([[1.0, 2.0], [2.0, 1.0]]), 10, rng)  # not positive semi-definite
    with pytest.raises(ValueError):
        equicorrelation(4, -0.5)


def test_latent_threshold():
    assert latent_threshold("median") == 0.0 and latent_threshold("mean", mean=3.0) == 0.0
    assert latent_threshold("zero", mean=0.5) == -0.5
    assert latent_threshold("quantile", q=0.8) == pytest.approx(stats.norm.ppf(0.8), rel=1e-12)
    with pytest.raises(ValueError):
        latent_threshold("sign")


def test_dichotomized_gaussian_table_exact_values():
    for rho in (0.0, 0.2, 0.5, 0.9):
        T2 = dichotomized_gaussian_table(2, rho, 0.0)
        assert T2[1, 1] == pytest.approx(0.25 + math.asin(rho) / (2 * math.pi), abs=1e-13)  # orthant probability
    T = dichotomized_gaussian_table(4, 0.4, 0.7)
    assert T.sum() == pytest.approx(1.0, abs=1e-14)
    ones = np.indices(T.shape).sum(axis=0)
    for m in range(5):  # exchangeable: depends on the number of ones only
        assert np.ptp(T[ones == m]) < 1e-16
    for axis in range(4):  # each bit is on with probability Phi(-tau)
        assert marginal(T, (axis,))[1] == pytest.approx(stats.norm.sf(0.7), abs=1e-13)
    # matches the empirical frequencies of thresholded Gaussian samples (zero rule with mean -tau == threshold tau)
    n = 200_000
    X = gaussian_binarized(equicorrelation(3, 0.6), n, np.random.default_rng(9), "zero", mean=-0.3)
    P = dichotomized_gaussian_table(3, 0.6, 0.3)
    counts = np.zeros(P.shape)
    np.add.at(counts, tuple(X.T), 1)
    assert stats.chisquare(counts.ravel(), n * P.ravel()).pvalue > 1e-4


def test_median_threshold_kills_odd_orders_only():
    """E6's claim, exactly: a median-thresholded pairwise Gaussian has no 3-way but a 4-way interaction."""
    for rho in (0.2, 0.5, 0.8):
        T4 = dichotomized_gaussian_table(4, rho, 0.0)
        np.testing.assert_allclose(T4, T4[::-1, ::-1, ::-1, ::-1], rtol=0, atol=1e-16)  # flip symmetry
        assert local_ri(dichotomized_gaussian_table(3, rho, 0.0)) < ZERO_TOL
        assert local_ri(T4) > 1e-6
        assert local_ri(dichotomized_gaussian_table(3, rho, 0.84)) > 1e-6  # off-median threshold: odd orders appear
    for k in (2, 3, 4):
        assert local_ri(dichotomized_gaussian_table(k, 0.0, 0.5)) < ZERO_TOL  # rho = 0: independent bits


# ----------------------------------------------------------------------------------------------- suite helpers
def test_chunks_cover_range_and_respect_breaks():
    for total, size, breaks in ((10, 3, ()), (10, 4, (5,)), (7, 10, (2, 2, 9)), (5, 1, (0, 5)), (0, 3, ())):
        chunks = _chunks(total, size, breaks)
        flat = [i for a, b in chunks for i in range(a, b)]
        assert flat == list(range(total))
        assert all(0 < b - a <= size for a, b in chunks)
        assert all(not a < c < b for a, b in chunks for c in breaks)


def test_named_streams_are_reproducible_and_distinct():
    a = _rng(3, "E1|cards=2x2|n=256").random(4)
    np.testing.assert_array_equal(a, _rng(3, "E1|cards=2x2|n=256").random(4))
    assert not np.allclose(a, _rng(3, "E1|cards=2x2|n=512").random(4))
    assert not np.allclose(a, _rng(4, "E1|cards=2x2|n=256").random(4))


def test_isotonic_regression():
    np.testing.assert_allclose(_isotonic(np.array([0.1, 0.3, 0.2, 0.6]), np.ones(4)), [0.1, 0.25, 0.25, 0.6])
    np.testing.assert_allclose(_isotonic(np.array([0.5, 0.1]), np.array([1.0, 3.0])), [0.2, 0.2])
    y = np.array([0.0, 0.2, 0.9, 1.0])
    np.testing.assert_array_equal(_isotonic(y, np.ones(4)), y)


def test_mde_interpolation_recovers_noncentral_chi2_theory():
    n, crit = 4096, stats.chi2.isf(0.01, 1)
    theory = _theory_mde(n, 1, crit, 0.8)
    assert stats.ncx2.sf(crit, 1, 2 * n * theory) == pytest.approx(0.8, abs=1e-8)
    assert theory == pytest.approx(11.68 / (2 * n), rel=0.01)  # (z_0.995 + z_0.8)^2 / (2N)
    for factor in (1.25, 4.0):  # fine and coarse (quick-like) grids of planted RI
        ri = theory * factor ** np.arange(-4, 5, dtype=float)
        power = np.array([stats.ncx2.sf(crit, 1, 2 * n * x) for x in ri])
        mde, censored = _interp_mde(ri, power, np.full(ri.size, 100.0), 0.8)
        assert censored == "" and mde == pytest.approx(theory, rel=0.02 if factor < 2 else 0.12)
    assert _interp_mde(ri[-3:], power[-3:], np.full(3, 100.0), 0.8)[1] == "below"
    assert _interp_mde(ri[:3], power[:3], np.full(3, 100.0), 0.8)[1] == "above"


def test_interpolate_min_effect():
    table = [{"order": 2, "n": 100, "noise": 0.05, "floor": 1e-3}, {"order": 2, "n": 1000, "noise": 0.004, "floor": 1e-3},
             {"order": 3, "n": 100, "noise": 0.06, "floor": 1e-3}, {"order": 3, "n": 1000, "noise": 0.005, "floor": 1e-3}]
    assert interpolate_min_effect(table, 100, 2) == pytest.approx(0.05)
    assert interpolate_min_effect(table, 100) == pytest.approx(0.06)  # whole scan: max over orders
    assert interpolate_min_effect(table, 1000, 2) == pytest.approx(0.004)
    n_mid = math.sqrt(100 * 1000)  # N * noise interpolated linearly in log N: (5 + 4) / 2
    assert interpolate_min_effect(table, n_mid, 2) == pytest.approx(4.5 / n_mid)
    assert interpolate_min_effect(table, 10, 2) == pytest.approx(0.5)  # 1/N scaling below the grid
    assert interpolate_min_effect(table, 1e5, 2) == pytest.approx(1e-3)  # floor above the grid
    assert interpolate_min_effect(table, 1000, 7) == pytest.approx(0.005)  # unknown order: whole-scan value
    with pytest.raises(ValueError):
        interpolate_min_effect([], 100)
    with pytest.raises(ValueError):
        interpolate_min_effect(table, 0)


def test_config_presets_roundtrip_and_validation():
    for cfg in (CalibrationConfig.quick(), CalibrationConfig.full()):
        again = CalibrationConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert again == cfg
    full = CalibrationConfig.full()
    assert full.name == "full" and len(full.n_grid) == 10 and full.e1_reps > CalibrationConfig.quick().e1_reps
    bad = [{"experiments": ("E9_x",)}, {"n_grid": (1024, 256)}, {"seed": -1}, {"e4_final_ri": 1e-4},
           {"e6_methods": ("sign",)}, {"e6_rhos": (1.0,)}, {"e2_ci_reps": 1000}, {"orders": (1,)}]
    for kw in bad:
        with pytest.raises(ValueError):
            CalibrationConfig(**kw)
    with pytest.raises(ValueError):
        CalibrationConfig.from_dict({"not_a_field": 1})


# ----------------------------------------------------------------------------------------------- end-to-end quick run
@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("calibration")
    results = run_calibration(CalibrationConfig.quick(), out)
    md = write_report(results, out)
    return out, results, md


def test_quick_run_writes_all_outputs(quick_run):
    out, results, md = quick_run
    assert md == out / "CALIBRATION.md"
    expected = [f"{name}.csv" for name in EXPERIMENTS] + ["E3_power_mde.csv", "E7_chain_totals.csv",
                                                           "results.json", "calibration.json", "CALIBRATION.md"]
    for name in expected:
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    assert sorted(results["files"]) == sorted(f for f in expected if f not in ("calibration.json", "CALIBRATION.md"))
    figures = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert figures == ["e1.png", "e2.png", "e3_mde.png", "e3_power.png", "e4.png", "e5.png", "e6.png", "e7.png"]
    for p in (out / "figures").glob("*.png"):
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    text = md.read_text(encoding="utf-8")
    for heading in ("## Instrument error budget", "## E1 ", "## E2 ", "## E3 ", "## E4 ", "## E5 ", "## E6 ", "## E7 ",
                    "## Caveats"):
        assert heading in text
    for link in [line.split("](")[1].rstrip(")") for line in text.splitlines() if line.startswith("![")]:
        assert (out / link).is_file(), link
    assert "sign symmetry: VERIFIED" in text
    cfg = CalibrationConfig.quick()
    n_rows = {name: len(results["experiments"][name]["rows"]) for name in EXPERIMENTS}
    assert n_rows["E1_null"] == (len(cfg.orders) + len(cfg.e1_categorical)) * len(cfg.e1_strengths) * len(cfg.n_grid)
    assert n_rows["E2_bias"] == len(cfg.orders) * len(cfg.e2_targets) * len(cfg.n_grid)
    assert n_rows["E3_power"] == n_rows["E2_bias"] * len(cfg.e3_family_sizes)
    assert n_rows["E6_discretization"] == len(cfg.e6_methods) * len(cfg.e6_rhos) * (cfg.e6_max_order - 1)
    with open(out / "E2_bias.csv", encoding="utf-8") as fh:
        assert sum(1 for _ in fh) == n_rows["E2_bias"] + 1


def test_quick_run_null_calibration(quick_run):
    _, results, _ = quick_run
    rows = [r for r in results["experiments"]["E1_null"]["rows"] if r["null"] == "chi2" and r["binary"]]
    regular = [r for r in rows if r["frac_zero"] <= 0.05]
    assert len(regular) >= len(rows) // 2
    for a in results["config"]["fpr_alphas"]:
        col = f"fpr_{a:g}"
        n_tot = sum(r["n_rep"] for r in regular)
        pooled = sum(r[col] * r["n_rep"] for r in regular) / n_tot
        assert abs(pooled - a) < 4 * math.sqrt(a * (1 - a) / n_tot), (a, pooled)  # FPR near alpha
        for r in regular:
            k = round(r[col] * r["n_rep"])
            assert stats.binomtest(k, r["n_rep"], a).pvalue > 1e-4, r
        for r in rows:  # boundary cells may be conservative, never liberal
            assert r[col] <= a + 4 * r[f"{col}_se"]
    for r in rows:
        if r["strength"] == 0:  # the noise floor follows chi2_1(0.99) / 2N
            assert 0.6 < r["ri_plugin_q99"] / r["theory_q99"] < 1.6, r
    s = results["experiments"]["E1_null"]["summary"]
    assert s["ks_max_regular"] < 0.15
    floors = results["budget"]["recommended_min_effect"]["table"]
    assert all(f["value"] >= f["floor"] and f["source"] == "E1" for f in floors)


def test_quick_run_bias_and_power(quick_run):
    _, results, _ = quick_run
    e2 = results["experiments"]["E2_bias"]["rows"]
    np.testing.assert_allclose([r["true_ri"] for r in e2], [r["target_ri"] for r in e2], rtol=1e-8)
    assert all(r["n_invalid"] == 0 for r in e2)
    sd_ratio = [r["sd_plugin"] / r["theory_sd_plugin"] for r in e2]
    assert 0.8 < np.median(sd_ratio) < 1.25
    bias_ratio = [r["bias_plugin"] / r["theory_bias_plugin"] for r in e2 if r["lambda"] <= 50]
    assert 0.5 < np.median(bias_ratio) < 1.6
    assert results["experiments"]["E2_bias"]["summary"]["median_coverage_strong"] >= 0.8
    e3 = results["experiments"]["E3_power"]
    for r in (r for r in e3["rows"] if r["family_size"] == 1):  # empirical power = noncentral chi2 power
        for col, theory in (("power_stat", "theory_power_stat"), ("power", "theory_power")):
            k = round(r[col] * r["n_rep"])
            assert stats.binomtest(k, r["n_rep"], min(max(r[theory], 1e-9), 1 - 1e-9)).pvalue > 1e-4, (col, r)
    # power increases with N: averaged over the planted RI levels, for every order / family size / rule
    for k in results["config"]["orders"]:
        for m in results["config"]["e3_family_sizes"]:
            for col in ("power_stat", "power"):
                curve = [np.mean([r[col] for r in e3["rows"] if r["order"] == k and r["family_size"] == m
                                  and r["n"] == n]) for n in results["config"]["n_grid"]]
                assert np.all(np.diff(curve) > 0), (k, m, col, curve)
    assert e3["summary"]["monotone_in_n"]["violations"] == 0
    mde = [r for r in e3["summary"]["mde"] if r["rule"] == "stat" and r["family_size"] == 1 and not r["censored"]]
    assert mde and all(0.5 < r["mde"] / r["theory_mde"] < 2.0 for r in mde)


def test_quick_run_timing_identity_artifacts_chains(quick_run):
    _, results, _ = quick_run
    cfg = results["config"]
    e4 = results["experiments"]["E4_timing"]
    for curve in e4["summary"]["curves"].values():
        assert np.all(np.diff(curve["true_ri"]) >= -1e-15)  # a monotone schedule gives a monotone true RI
    n_max = max(cfg["e4_n_grid"])
    for r in e4["rows"]:
        if r["width"] == 0:
            assert r["true_onset"] == round(cfg["e4_t_star"])  # step: RI jumps at t*
        if r["n"] == n_max:
            assert r["detect_rate"] >= 0.75 and abs(r["median_error"]) <= 1, r
    e5 = {(r["n"], r["rule"]): r for r in results["experiments"]["E5_identity"]["rows"]}
    big = e5[(max(cfg["e5_n_grid"]), "survey")]
    assert big["recall"] == 1.0 and big["precision_true"] >= 0.9 and big["mean_fp_related"] == 0
    s5 = results["experiments"]["E5_identity"]["summary"]
    assert all(p["true_ri"] > 5e-3 for p in s5["planted"]) and s5["largest_unplanted"][0]["true_ri"] < 2e-3
    assert s5["max_ri_inside_planted"] < 1e-4  # the planted triple stays (almost) a pure synergy
    e6 = results["experiments"]["E6_discretization"]
    for r in e6["rows"]:
        if r["rho"] == 0:
            assert abs(r["ri_population"]) < ZERO_TOL
        elif r["order"] == 3 and r["method"] == "median":
            assert abs(r["ri_population"]) < ZERO_TOL  # odd order vanishes by sign symmetry
            assert abs(r["ri_excess_mean"]) * 2 * r["n"] < 1.0  # and the instrument sees ~ nothing either
        elif r["order"] == 3:
            assert r["ri_population"] > 1e-6  # asymmetric thresholds create a 3-way artefact
        elif r["order"] == 4:
            assert r["ri_population"] > 1e-7  # even orders survive for every threshold
    assert e6["summary"]["median_symmetry"]["verified"] is True
    e7 = results["experiments"]["E7_chain"]
    assert e7["summary"]["max_abs_sum_minus_tc"] < 1e-8
    assert max(abs(t["total"] - t["tc"]) for t in e7["tables"]["totals"]) < 1e-8  # sum of chain RI = TC
    assert max(abs(t["residual"]) for t in e7["tables"]["totals"]) < 1e-12
    for t in {r["table"] for r in e7["rows"]}:
        rows = [r for r in e7["rows"] if r["table"] == t]
        assert sum(r["ri_graded"] for r in rows) == pytest.approx(rows[0]["tc"], abs=1e-8)
        first = next(r for r in rows if r["subset"] == "0-1")
        assert first["ri_graded"] == pytest.approx(first["local_ri"], rel=1e-9)  # first chain step = MI
        top = max(rows, key=lambda r: r["order"])
        assert top["spread"] < 1e-12  # the full set always enters last


def test_quick_run_calibration_json_and_rerender(quick_run, tmp_path):
    out, results, _ = quick_run
    cal = load_calibration(out / "calibration.json")
    for key in ("noise_floor", "recommended_min_effect", "min_detectable_ri", "timing", "artifacts",
                "chain_dependence", "specificity", "estimator_bias", "false_positive_rate"):
        assert key in cal, key
    table = cal["recommended_min_effect"]["table"]
    ns = sorted({r["n"] for r in table})
    values = [recommended_min_effect(out / "calibration.json", n) for n in ns]
    assert all(v >= cal["recommended_min_effect"]["floor"] for v in values)
    assert all(np.diff(values) <= 0)  # more samples, lower (or equal) threshold
    for n in ns:
        assert recommended_min_effect(cal, n) == pytest.approx(max(r["value"] for r in table if r["n"] == n))
    reloaded = load_results(out)
    assert reloaded["budget"] == results["budget"]
    write_report(reloaded, tmp_path)
    again = load_calibration(tmp_path / "calibration.json")
    cal.pop("created_at")
    again.pop("created_at")
    assert again == cal


# ----------------------------------------------------------------------------------------------- determinism, resume
def _tiny_config(**kw) -> CalibrationConfig:
    base = dict(n_grid=(256, 1024), e1_reps=30, e1_strengths=(0.0, 1.0), e1_categorical=((3, 2),),
                e1_bootstrap_reps=4, e1_n_null=19, e2_targets=(4e-3, 3e-2), e2_reps=12, e2_ci_reps=3, e2_n_boot=8,
                e3_family_sizes=(1, 50), e4_checkpoints=8, e4_t_star=4.0, e4_widths=(1.0,), e4_n_grid=(1024,),
                e4_reps=2, e5_n_grid=(512,), e5_reps=2, e6_rhos=(0.4,), e6_n_grid=(512,), e6_reps=2,
                e7_cards=((2, 3, 2),), e7_random_chains=1)
    base.update(kw)
    return CalibrationConfig(**base)


def _content(res: dict) -> str:
    return json.dumps({"experiments": res["experiments"], "budget": res["budget"]}, sort_keys=True)


def test_results_independent_of_n_jobs_and_resumable(tmp_path):
    cfg = _tiny_config()
    serial = run_calibration(cfg, tmp_path / "serial", n_jobs=1, resume=False)
    parallel = run_calibration(cfg, tmp_path / "parallel", n_jobs=2, resume=False)
    assert parallel["n_jobs"] == 2
    assert _content(parallel) == _content(serial)  # bit-identical, whatever the process layout
    cache = sorted((tmp_path / "serial" / "cache").glob("*.npz"))
    assert cache
    cache[0].write_bytes(b"not an npz file")  # a damaged cache entry is recomputed, never fatal
    resumed = run_calibration(cfg, tmp_path / "serial", n_jobs=1, resume=True)
    assert _content(resumed) == _content(serial)
    assert resumed["n_cached"] == resumed["n_items"] - 1 and serial["n_cached"] == 0  # only the damaged one rerun
    assert sum(resumed["compute_s"].values()) < 0.5 * sum(serial["compute_s"].values())
    other_seed = run_calibration(_tiny_config(seed=1, experiments=("E1_null",)), tmp_path / "seed1")
    assert _content(other_seed) != _content(serial)


def test_partial_run_and_report_without_e1(tmp_path):
    cfg = _tiny_config(experiments=("E6_discretization", "E7_chain"))
    res = run_calibration(cfg, tmp_path)
    assert list(res["experiments"]) == ["E6_discretization", "E7_chain"]
    assert "noise_floor" not in res["budget"]
    me = res["budget"]["recommended_min_effect"]
    assert all(r["source"] == "chi2" for r in me["table"])  # analytic fallback when E1 did not run
    expected = max(cfg.min_effect_floor, (stats.chi2.ppf(cfg.min_effect_null_quantile, 1) - 1) / (2 * 256))
    assert interpolate_min_effect(me["table"], 256) == pytest.approx(expected)
    text = write_report(res, tmp_path).read_text(encoding="utf-8")
    assert "## E6 " in text and "## E7 " in text and "## E1 " not in text
    assert sorted(p.name for p in (tmp_path / "figures").glob("*.png")) == ["e6.png", "e7.png"]


def test_import_does_not_load_matplotlib():
    """Survey processes that only read calibration.json, and spawned workers, must not pay for (or reconfigure)
    matplotlib."""
    code = "import sys, tycho.calibration; sys.exit('matplotlib' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


@pytest.mark.slow
def test_null_fpr_precise(tmp_path):
    """E1 with 3000 replicates per cell: every regular cell's FPR within 4 binomial SE of alpha."""
    cfg = _tiny_config(experiments=("E1_null",), n_grid=(512, 4096), e1_reps=3000, e1_strengths=(0.0, 0.5),
                       e1_categorical=(), e1_bootstrap_reps=300, e1_n_null=199)
    rows = run_calibration(cfg, tmp_path)["experiments"]["E1_null"]["rows"]
    for r in rows:
        if r["frac_zero"] <= 0.05:
            for a in cfg.fpr_alphas:
                assert abs(r[f"fpr_{a:g}"] - a) <= 4 * r[f"fpr_{a:g}_se"], r
