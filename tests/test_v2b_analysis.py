"""Tests of the sky v2 Part B offline analysis (tycho/analysis/v2b.py; docs/results/v2b_preregistration.md)."""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from tycho.analysis import v2b
from tycho.analysis.v2a import all_sets, holm
from tycho.ri.cells import cell_stats, projection_information
from tycho.tasks.patterns import pattern_bits

P, I, C, N = v2b.PREV, v2b.IND, v2b.CTX, v2b.NONE
PAIRS6 = list(itertools.combinations(range(6), 2))


def _network(H: int, rng: np.random.Generator, scale: float = 1.5) -> np.ndarray:
    """Log-odds of a random function of H binary heads with terms up to order 3."""
    s = 2.0 * pattern_bits(H) - 1.0
    z = np.full(1 << H, rng.normal(scale=0.3))
    for r in (1, 2, 3):
        for B in itertools.combinations(range(H), r):
            z = z + rng.normal(scale=scale / r) * np.prod(s[:, list(B)], axis=1)
    return z


# ================================================================================================ measures (§1)
def test_table_stats_full_and_marginal_match_row_construction():
    rng = np.random.default_rng(1)
    H = 6
    m = np.clip(np.tanh(_network(H, rng) / 2), -1 + 1e-12, 1 - 1e-12)
    X = pattern_bits(H)
    for inputs in [tuple(range(H)), (1, 4), (0, 2, 5), (3,)]:
        mine = v2b.table_stats(m, inputs)
        ref = cell_stats(X, (1.0 + m) / 2.0, [inputs])[0]
        np.testing.assert_array_equal(mine.counts, ref.counts)
        np.testing.assert_allclose(mine.cell_means(), ref.cell_means(), atol=1e-15)


def test_mu_marginal_is_the_average_of_m_over_the_other_heads():
    rng = np.random.default_rng(2)
    H = 6
    m = np.clip(np.tanh(_network(H, rng) / 2), -1 + 1e-12, 1 - 1e-12)
    X = pattern_bits(H)
    a, b = 1, 4
    cs = v2b.table_stats(m, (a, b))
    assert cs.inputs == (a, b) and cs.n == 64.0
    np.testing.assert_array_equal(cs.counts, [16.0] * 4)
    # cell index = 2 x_a + x_b (first input most significant, as CellStats)
    for xa, xb in itertools.product((0, 1), repeat=2):
        sel = (X[:, a] == xa) & (X[:, b] == xb)
        assert cs.cell_means()[2 * xa + xb] == pytest.approx(np.mean(m[sel]), abs=1e-15)
    # the full table has one pattern per cell
    full = v2b.table_stats(m, range(H))
    np.testing.assert_array_equal(full.counts, np.ones(64))


def test_set_values_match_projection_information_by_hand():
    rng = np.random.default_rng(3)
    lo = _network(6, rng)
    m = v2b.clipped_mean(lo)
    X = pattern_bits(6)
    A = (0, 3)
    vals = v2b.set_values(m, A)
    full = cell_stats(X, (1 + m) / 2, [tuple(range(6))])[0]
    marg = cell_stats(X, (1 + m) / 2, [A])[0]
    for r, (cs, pos) in enumerate(((full, A), (marg, (0, 1)))):
        assert vals[r, 0] == pytest.approx(projection_information(cs, pos, weights="uniform"), rel=1e-9, abs=1e-14)
        assert vals[r, 1] == pytest.approx(projection_information(cs, pos, link="linear", weights="uniform"),
                                           rel=1e-9, abs=1e-14)
        assert vals[r, 2] == pytest.approx(projection_information(cs, pos, link="linear", weights="uniform",
                                                                  eta_value="gaussian"), rel=1e-9, abs=1e-14)


def test_bounded_eta_equal_at_cu_and_mu_on_the_uniform_design():
    rng = np.random.default_rng(4)
    lo = _network(6, rng)[None, :]
    sets = all_sets(6, 3)
    vals = v2b.object_measures(lo, sets)  # (refs, values, 1, sets)
    eb = v2b.VALUES.index("eta_b")
    np.testing.assert_allclose(vals[0, eb, 0], vals[1, eb, 0], atol=1e-12, rtol=0)
    # not vacuous: θ and Gaussian η differ between the references for some pair
    assert np.max(np.abs(vals[0, 0, 0] - vals[1, 0, 0])) > 1e-4
    assert np.max(np.abs(vals[0, 2, 0] - vals[1, 2, 0])) > 1e-6


def test_m_is_clipped_before_averaging():
    lo = np.zeros(8)
    lo[7] = 1e6  # p = 1 for the full pattern
    m = np.clip(np.tanh(lo / 2), -1 + 1e-12, 1 - 1e-12)
    assert m[7] == 1 - 1e-12
    vals = v2b.object_measures(lo[None, :], [(0, 1, 2)])
    assert np.all(np.isfinite(vals))


# ================================================================================================ roles (§3)
def _final(prev, prev2, ind):
    out = {}
    for h in range(3):
        out[f"prev_l0h{h}"], out[f"prev2_l0h{h}"], out[f"ind_l0h{h}"] = prev[h], prev2[h], 0.0
        out[f"prev_l1h{h}"], out[f"prev2_l1h{h}"], out[f"ind_l1h{h}"] = 0.9, 0.9, ind[h]
    return out


def test_role_assignment_including_context_heads():
    roles = v2b.assign_roles(_final(prev=[0.7, 0.3, 0.1], prev2=[0.4, 0.25, 0.05], ind=[0.6, 0.2, 0.49]), 2, 3)
    assert roles == [P, C, N, I, N, N]  # layer-1 prev / prev2 scores never make a previous-token or context head
    # boundaries: prev = 0.5 is previous-token, prev2 = 0.15 is context, ind = 0.5 is induction
    roles = v2b.assign_roles(_final(prev=[0.5, 0.4999, 0.0], prev2=[0.9, 0.15, 0.1499], ind=[0.5, 0.0, 1.0]), 2, 3)
    assert roles == [P, C, N, I, N, I]
    # a 1-layer model gets layer-0 roles only
    assert v2b.assign_roles(_final(prev=[0.9, 0.1, 0.0], prev2=[0, 0.3, 0], ind=[1, 1, 1]), 1, 3) == [P, C, N]


def test_formation_versus_eligibility():
    assert v2b.is_formed([P, C, N, I, N, N], 0.05) and v2b.is_eligible([P, C, N, I, N, N], True)
    # seed-331-like: the third layer-0 head is a second previous-token head; formed, not eligible
    roles = [P, P, C, I, I, I]
    assert v2b.is_formed(roles, 0.05) and not v2b.is_eligible(roles, True)
    assert v2b.is_formed([P, N, N, I, N, N], 0.1)  # loss <= 0.1 inclusive
    assert not v2b.is_formed([P, N, N, I, N, N], 0.1000001)
    assert not v2b.is_formed([C, N, N, I, I, N], 0.01)  # no previous-token head
    assert not v2b.is_formed([P, N, N, N, N, N], 0.01)  # no induction head
    assert not v2b.is_eligible([P, N, N, I, N, N], False)  # not formed, never eligible


# ================================================================================================ pairs (§4 M)
def _named(mask):
    return {p for p, k in zip(PAIRS6, mask, strict=True) if k}


def test_pair_sets_one_context_head():
    roles = [P, C, N, I, N, N]
    pos, neg = v2b.referee_masks(roles, PAIRS6, "i")
    assert _named(pos) == {(0, 3)} and _named(neg) == set(PAIRS6) - {(0, 3)}
    pos, neg = v2b.referee_masks(roles, PAIRS6, "ii")
    assert _named(pos) == {(0, 3)}
    assert _named(neg) == {p for p in PAIRS6 if {2, 4, 5} & set(p)}
    assert set(PAIRS6) - _named(pos) - _named(neg) == {(0, 1), (1, 3)}  # left out under (ii)
    assert v2b.pair_kind(C, N) == "context" and v2b.pair_kind(P, C) == "context"
    assert v2b.pair_kind(N, N) == "remaining" and v2b.pair_kind(I, N) == "remaining"


def test_pair_sets_two_context_heads():
    roles = [C, C, P, I, N, N]
    pos, neg = v2b.referee_masks(roles, PAIRS6, "i")
    assert _named(pos) == {(2, 3)} and (0, 1) in _named(neg)  # a context pair is non-circuit under (i)
    pos, neg = v2b.referee_masks(roles, PAIRS6, "ii")
    left_out = set(PAIRS6) - _named(pos) - _named(neg)
    assert left_out == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)}
    assert _named(neg) == {p for p in PAIRS6 if {4, 5} & set(p)}


def test_pair_sets_two_previous_token_heads_and_composition_only():
    roles = [P, P, N, I, I, N]
    pos, neg = v2b.referee_masks(roles, PAIRS6, "i")
    comp = {(0, 3), (0, 4), (1, 3), (1, 4)}
    assert _named(pos) == comp | {(0, 1), (3, 4)}
    assert _named(neg) == set(PAIRS6) - _named(pos)
    pos, neg = v2b.referee_masks(roles, PAIRS6, "i", composition_only=True)
    assert _named(pos) == comp and _named(neg) == set(PAIRS6) - comp - {(0, 1), (3, 4)}
    pos2, neg2 = v2b.referee_masks(roles, PAIRS6, "ii")
    assert _named(neg2) == _named(neg)  # no context head: both referees agree
    assert [v2b.pair_kind(roles[a], roles[b]) for a, b in [(0, 1), (3, 4), (0, 3)]] == \
        ["redundancy", "redundancy", "composition"]


# ================================================================================================ AUC, rank
def test_auc_with_ties():
    assert v2b.auc([1.0, 2.0], [0.0, 2.0]) == pytest.approx(2.5 / 4)
    assert v2b.auc([1.0], [1.0 + 5e-13]) == 0.5  # within 1e-12: tied
    assert v2b.auc([1.0], [1.0 + 2e-12]) == 0.0
    assert v2b.auc([1.0 + 2e-12], [1.0]) == 1.0
    assert math.isnan(v2b.auc([], [1.0])) and math.isnan(v2b.auc([1.0], []))
    rng = np.random.default_rng(5)
    pos, neg = rng.integers(0, 4, 7).astype(float), rng.integers(0, 4, 9).astype(float)
    u = mannwhitneyu(pos, neg).statistic
    assert v2b.auc(pos, neg) == pytest.approx(u / (pos.size * neg.size))


def test_best_composition_rank():
    vals = np.array([0.5, 0.9, 0.5 + 5e-13, 0.7, 0.1])
    comp = np.array([True, False, False, False, True])
    assert v2b.best_composition_rank(vals, comp) == 3  # 0.9 and 0.7 larger; 0.5 + 5e-13 tied
    assert v2b.best_composition_rank(vals, np.array([False, True, False, False, False])) == 1
    assert v2b.best_composition_rank(vals, np.zeros(5, bool)) is None


# ================================================================================================ bootstrap
def test_joint_seed_resamples_redraw_rule():
    accept = lambda mult: mult[0] > 0 and mult[1] > 0  # noqa: E731
    mult, redrawn = v2b.joint_seed_resamples(4, 300, accept, seed=7)
    assert mult.shape == (300, 4) and np.all(mult.sum(axis=1) == 4)
    assert np.all(mult[:, 0] > 0) and np.all(mult[:, 1] > 0) and redrawn > 0
    # the same stream drawn by hand: rejected draws are skipped, not replaced by something else
    rng = np.random.default_rng(7)
    kept, rej = [], 0
    while len(kept) < 300:
        m = np.bincount(rng.integers(0, 4, size=4), minlength=4)
        if m[0] > 0 and m[1] > 0:
            kept.append(m)
        else:
            rej += 1
    np.testing.assert_array_equal(mult, np.array(kept))
    assert redrawn == rej
    with pytest.raises(RuntimeError):
        v2b.joint_seed_resamples(3, 5, lambda m: False, max_redraws=10)


def test_delta_stat_skips_missing_runs_and_weights_duplicates():
    D = np.array([[0.1, np.nan], [0.3, 0.2], [np.nan, -0.4]])  # labels × sizes
    assert v2b.delta_stat(D, np.ones((1, 3)))[0] == pytest.approx(((0.1 + 0.3) / 2 + (0.2 - 0.4) / 2) / 2)
    W = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
    out = v2b.delta_stat(D, W)
    assert out[0] == pytest.approx(((0.2 + 0.3) / 3 + 0.2) / 2)
    assert out[1] == pytest.approx((0.3 + 0.2) / 2)


def test_holm_over_two():
    np.testing.assert_allclose(holm([0.03, 0.04]), [0.06, 0.06])
    np.testing.assert_allclose(holm([0.01, 0.04]), [0.02, 0.04])
    np.testing.assert_allclose(holm([0.04, 0.01]), [0.04, 0.02])
    np.testing.assert_allclose(holm([0.6, 1.0]), [1.0, 1.0])


# ================================================================================================ labels (§4)
def _comps(spec: dict) -> list[dict]:
    """8 comparisons; ``spec[(ref, referee, ev)] = (delta, p_holm)``, default (0.0, 1.0)."""
    out = []
    for ref in v2b.REFS:
        for referee in v2b.REFEREES:
            for ev in v2b.ETA_VALUES:
                d, p = spec.get((ref, referee, ev), (0.0, 1.0))
                out.append({"reference": ref, "referee": referee, "eta_value": ev, "delta": d, "p_holm": p})
    return out


def _all4(ref, d, p=0.001):
    return {(ref, r, e): (d, p) for r in v2b.REFEREES for e in v2b.ETA_VALUES}


def test_labels_ahead_and_outcomes():
    res = v2b.m_labels(_comps(_all4("C_U", -0.10)))
    assert res["by_reference"]["C_U"]["eta"]["labels"] == ["ahead"] and res["outcome"] == "η ahead on mechanism"
    res = v2b.m_labels(_comps(_all4("M_U", 0.08)))
    assert res["theta_ahead_at"] == ["M_U"] and res["outcome"] == "θ ahead on mechanism"
    res = v2b.m_labels(_comps({**_all4("C_U", -0.1), **_all4("M_U", 0.1)}))
    assert res["outcome"] == "mixed"
    assert v2b.m_labels(_comps({}))["outcome"] == "tie"


def test_labels_referee_and_value_dependent():
    # met under referee (i) only, both η values: referee-dependent, not value-dependent; outcome tie, reported
    res = v2b.m_labels(_comps({("C_U", "i", "eta_b"): (-0.1, 0.001), ("C_U", "i", "eta_g"): (-0.1, 0.001)}))
    assert res["by_reference"]["C_U"]["eta"]["labels"] == ["referee-dependent"]
    assert res["outcome"] == "tie" and len(res["reported_with_outcome"]) == 1
    # met with bounded η only, both referees: value-dependent
    res = v2b.m_labels(_comps({("C_U", "i", "eta_b"): (-0.1, 0.001), ("C_U", "ii", "eta_b"): (-0.1, 0.001)}))
    assert res["by_reference"]["C_U"]["eta"]["labels"] == ["value-dependent"]
    # a single met combination carries both labels
    res = v2b.m_labels(_comps({("M_U", "ii", "eta_g"): (0.2, 0.01)}))
    assert res["by_reference"]["M_U"]["theta"]["labels"] == ["referee-dependent", "value-dependent"]
    assert res["outcome"] == "tie"
    # diagonal and three-of-four patterns have no label in §4: reported as partial, never ahead
    res = v2b.m_labels(_comps({("C_U", "i", "eta_b"): (-0.1, 0.001), ("C_U", "ii", "eta_g"): (-0.1, 0.001)}))
    assert res["by_reference"]["C_U"]["eta"]["labels"] == ["partial"] and res["outcome"] == "tie"
    three = _all4("C_U", -0.1)
    three[("C_U", "ii", "eta_g")] = (-0.1, 0.2)
    assert v2b.m_labels(_comps(three))["by_reference"]["C_U"]["eta"]["labels"] == ["partial"]


def test_met_condition_boundaries():
    base = _all4("C_U", -0.05, 0.049)  # |Δ| = 0.05 and p < 0.05: met
    assert v2b.m_labels(_comps(base))["eta_ahead_at"] == ["C_U"]
    assert v2b.m_labels(_comps(_all4("C_U", -0.0499, 0.001)))["eta_ahead_at"] == []  # |Δ| < 0.05
    assert v2b.m_labels(_comps(_all4("C_U", -0.1, 0.05)))["eta_ahead_at"] == []  # p not below 0.05
    assert v2b.m_labels(_comps(_all4("C_U", math.nan, 1.0)))["outcome"] == "tie"


# ================================================================================================ fake runs
def _fake_run(d, seed, roles, *, region=v2b.MAIN, loss=None, ind=None, prev=None, pair_last=None, n_ck=6,
              steps=None):
    """A Run with a light curve and pair values; ``pair_last`` (15,) sets every measure's last-checkpoint pairs."""
    steps = np.array([0, 1, 3, 10, 30, 100][:n_ck]) if steps is None else np.asarray(steps)
    n_ck = steps.size
    sets = all_sets(6, 3)
    values = np.zeros((2, 3, n_ck, len(sets)))
    if pair_last is not None:
        pidx = [j for j, A in enumerate(sets) if len(A) == 2]
        values[:, :, -1, pidx] = np.asarray(pair_last)
    lc = {"step": steps, "x_loss_rep": np.linspace(4.0, 0.05, n_ck) if loss is None else np.asarray(loss, float)}
    for h in range(3):
        lc[f"x_prev_l0h{h}"] = np.zeros(n_ck) if prev is None or h else np.asarray(prev, float)
        lc[f"x_ind_l1h{h}"] = np.zeros(n_ck) if ind is None or h else np.asarray(ind, float)
        lc[f"x_prev_l1h{h}"] = lc[f"x_prev2_l0h{h}"] = lc[f"x_prev2_l1h{h}"] = np.zeros(n_ck)
        lc[f"x_ind_l0h{h}"] = np.zeros(n_ck)
    rec = {"object_id": f"{region}-{d}-{seed}", "region": region, "seed": seed, "status": "done",
           "task_cfg": {"d_model": d, "n_layer": 2, "n_head": 3}}
    r = v2b.Run(rec=rec, steps=steps, values=values, sets=sets, checks={}, lc=lc, roles=list(roles))
    if region == v2b.MAIN:
        r.formed = v2b.is_formed(r.roles, float(lc["x_loss_rep"][-1]))
        r.eligible = v2b.is_eligible(r.roles, r.formed)
    return r


FORMED = [P, C, N, I, N, N]
UNFORMED = [N, N, N, N, N, N]


def test_b1_slope_on_synthetic_steps():
    x = np.log([48, 48, 96, 96, 192])
    y = np.log(700.0) - 0.8 * (x - np.log(48))
    assert v2b.weighted_slopes(x, y, np.ones((1, 5)))[0] == pytest.approx(-0.8)
    y2 = y + np.array([0.1, -0.1, 0.2, -0.2, 0.0])
    W = np.array([[2.0, 0, 1, 1, 0]])
    xx, yy = np.repeat(x, W[0].astype(int)), np.repeat(y2, W[0].astype(int))
    assert v2b.weighted_slopes(x, y2, W)[0] == pytest.approx(np.polyfit(xx, yy, 1)[0])
    assert math.isnan(v2b.weighted_slopes(x, y2, np.array([[1.0, 1, 0, 0, 0]]))[0])  # one size: no slope


def test_b1_phase_change_and_bootstrap_redraw():
    steps = [0, 10, 100, 300, 1000, 3000]
    # size 48: seed 0 forms (loss <= 1 first at step 1000); size 96: seed 1 forms (first at 300); others do not
    runs = [_fake_run(48, 0, FORMED, loss=[4, 3, 2, 1.5, 1.0, 0.05], steps=steps),
            _fake_run(96, 1, FORMED, loss=[4, 3, 2, 0.9, 0.5, 0.05], steps=steps)]
    runs += [_fake_run(d, s, UNFORMED, loss=[4, 4, 4, 4, 4, 4], steps=steps) for d in (48, 96) for s in (2, 3)]
    runs += [_fake_run(48, 1, FORMED, loss=[4, 3, 2, 1.5, 1.1, 0.2], steps=steps)]  # loss 0.2 > 0.1: not formed
    records = [r.rec for r in runs] + [{"object_id": "x", "region": v2b.MAIN, "seed": 0, "status": "failed",
                                        "task_cfg": {"d_model": 96, "n_layer": 2, "n_head": 3}}]
    res = v2b.b1(runs, records, B=200)
    assert res["per_size"]["48"]["phase_change_steps"] == [1000] and res["per_size"]["96"]["phase_change_steps"] == [300]
    assert res["per_size"]["48"]["n_formed"] == 1 and res["per_size"]["96"]["n_objects"] == 4
    assert res["slope"] == pytest.approx((math.log(300) - math.log(1000)) / math.log(2))
    # every kept resample holds labels 0 and 1 (both sizes): the slope is the same in all of them
    assert res["bootstrap"]["redrawn"] > 0 and res["bootstrap"]["slopes_finite"] == 200
    assert res["slope_interval_95"] == pytest.approx([res["slope"]] * 2)
    assert res["predictions"]["slope_interval_below_0"]["met"]
    assert not res["predictions"]["formation_at_every_size"]["met"]  # 1 of 4 at each size
    assert v2b.phase_change_index([4.0, 1.0000001, 1.0]) == 2 and v2b.phase_change_index([0.5, 2.0]) == 0
    assert v2b.phase_change_index([2.0, 3.0]) is None


def test_b2_same_evaluation_counts_as_not_before():
    runs = [_fake_run(48, 0, FORMED, ind=[0, 0.2, 0.5, 0.9, 0.9, 0.9], prev=[0, 0.1, 0.3, 0.6, 0.9, 0.9]),  # before
            _fake_run(48, 1, FORMED, ind=[0, 0.2, 0.6, 0.9, 0.9, 0.9], prev=[0, 0.1, 0.5, 0.6, 0.9, 0.9]),  # tie
            _fake_run(96, 2, FORMED, ind=[0, 0, 0, 0.4, 0.7, 0.9], prev=[0, 0.5, 0.4, 0.6, 0.9, 0.9]),  # after
            _fake_run(96, 3, UNFORMED, ind=[0.9] * 6, prev=[0.0] * 6)]  # not formed: not counted
    res = v2b.b2(runs, [48, 96])
    assert [x["ind_before_prev"] for x in res["runs"]] == [True, False, False]
    assert res["n_formed"] == 3 and res["share"] == pytest.approx(1 / 3) and not res["prediction_met"]
    assert res["by_size"]["48"] == {"n_formed": 2, "n_before": 1, "share": 0.5}
    assert v2b.first_reach([0.1, 0.4999, 0.5]) == 2 and v2b.first_reach([0.1]) is None


# ================================================================================================ M machinery
def _pairs_for(roles):
    """Last-checkpoint pair values: circuit pairs 1.0, others 0.5 (θ AUC 1 under both referees)."""
    pos, _ = v2b.referee_masks(roles, PAIRS6, "i")
    return np.where(pos, 1.0, 0.5)


def test_m_family_drop_and_redraw_rule():
    labels = [0, 1, 2, 3]
    runs = [_fake_run(48, s, FORMED, pair_last=_pairs_for(FORMED)) for s in labels]
    runs += [_fake_run(96, 0, FORMED, pair_last=_pairs_for(FORMED))]
    runs += [_fake_run(96, s, [P, P, C, I, I, I], pair_last=_pairs_for(FORMED)) for s in (1, 2)]  # not eligible
    runs += [_fake_run(192, s, FORMED, pair_last=_pairs_for(FORMED)) for s in (0, 1)]
    records = [r.rec for r in runs]
    aucs = {r.oid: v2b.run_aucs(r) for r in runs if r.formed}
    res = v2b.m_analysis(runs, records, aucs, B=100, min_eligible=1)
    assert res["family_sizes"] == [48, 96, 192] and res["n_eligible_by_size"] == {"48": 4, "96": 1, "192": 2}
    assert res["bootstrap"]["redrawn"] > 0
    # redrawn resamples: every kept one holds label 0 (the only eligible run at 96)
    held = [np.array([0, 1, 2, 3]), np.array([0]), np.array([0, 1])]
    mult, red = v2b.joint_seed_resamples(4, 100, lambda m: all(m[h].sum() > 0 for h in held))
    assert red == res["bootstrap"]["redrawn"] and np.all(mult[:, 0] > 0)
    # with a minimum of 2, size 96 is dropped and listed
    res2 = v2b.m_analysis(runs, records, aucs, B=50, min_eligible=2)
    assert res2["family_sizes"] == [48, 192] and res2["sizes_dropped"] == [{"d_model": 96, "n_eligible": 1}]
    # identical θ and η values: every AUC 1, Δ = 0, p = 1, tie
    c = res["comparisons"][0]
    assert c["delta"] == 0.0 and c["p_raw"] == 1.0 and res["decision"]["outcome"] == "tie"
    assert {(x["referee"], x["eta_value"], x["reference"]) for x in res["comparisons"]} == set(
        itertools.product(v2b.REFEREES, v2b.ETA_VALUES, v2b.REFS))


def test_m_delta_and_holm_on_a_planted_difference():
    rng = np.random.default_rng(11)
    runs = []
    for d in (48, 96):
        for s in range(6):
            r = _fake_run(d, s, FORMED, pair_last=_pairs_for(FORMED))
            # η ranks one non-circuit pair (l0h2, l1h2) above the circuit pair at C_U only: η AUC < 1 there
            j = PAIRS6.index((2, 5))
            for v in ("eta_b", "eta_g"):
                pidx = [k for k, A in enumerate(r.sets) if len(A) == 2]
                r.values[0, v2b.VALUES.index(v), -1, pidx[j]] = 2.0 + rng.uniform()
            runs.append(r)
    aucs = {r.oid: v2b.run_aucs(r) for r in runs}
    res = v2b.m_analysis(runs, [r.rec for r in runs], aucs, B=400, min_eligible=6)
    by = {(c["referee"], c["eta_value"], c["reference"]): c for c in res["comparisons"]}
    # referee (i): 14 negatives, one ranked above the single positive → η AUC 13/14, Δ = 1/14 at C_U
    assert by[("i", "eta_b", "C_U")]["delta"] == pytest.approx(1 / 14)
    assert by[("ii", "eta_b", "C_U")]["delta"] == pytest.approx(1 / 12)
    assert by[("i", "eta_b", "M_U")]["delta"] == 0.0
    c_cu, c_mu = by[("i", "eta_b", "C_U")], by[("i", "eta_b", "M_U")]
    np.testing.assert_allclose([c_cu["p_holm"], c_mu["p_holm"]], holm([c_cu["p_raw"], c_mu["p_raw"]]))
    assert c_cu["p_raw"] == pytest.approx(2 / 401)  # every resample has Δ* > 0: the floor
    assert res["decision"]["theta_ahead_at"] == ["C_U"] and res["decision"]["outcome"] == "θ ahead on mechanism"


# ================================================================================================ §5
def test_tau_calibration_and_onset():
    pool = np.arange(101.0)
    assert v2b.threshold(pool) == pytest.approx(99.0)
    assert v2b.threshold(np.array([0.0, 1.0])) == pytest.approx(np.percentile([0.0, 1.0], 99, method="linear"))
    assert math.isnan(v2b.threshold(np.zeros(0)))
    steps = np.array([0, 1, 2, 3, 4])
    assert v2b.onset_index(steps, [5.0, 0.5, 1.0, 2.0, 3.0], 1.0) == 3  # step 0 excluded; equal is not exceeding
    assert v2b.onset_index(steps, [5.0, 0.5, 1.0 + 5e-13, 1.0 + 2e-12, 3.0], 1.0) == 3  # within 1e-12: tied
    assert v2b.onset_index(steps, [0.0] * 5, 1.0) is None
    assert v2b.onset_index(steps, [9.0] * 5, math.nan) is None


def test_control_pool_uses_the_largest_pair_and_steps_from_1():
    r = _fake_run(48, 0, UNFORMED, region=v2b.NOREPEAT, steps=[0, 1, 5])
    pidx = [k for k, A in enumerate(r.sets) if len(A) == 2]
    r.values[0, 0, 0, pidx[0]] = 100.0  # step 0: excluded
    r.values[0, 0, 1, pidx[3]] = 0.2
    r.values[0, 0, 1, pidx[4]] = 0.3
    r.values[0, 0, 2, pidx[14]] = 0.7
    r.values[0, 0, 2, [k for k, A in enumerate(r.sets) if len(A) == 3][0]] = 50.0  # a triple: not a pair
    np.testing.assert_array_equal(v2b.control_pool([r], "C_U", "theta"), [0.3, 0.7])


def test_b3_shares():
    kinds = ["composition", "redundancy", "context", "remaining", "remaining"]
    s = v2b.b3_shares(np.array([2.0, 1.0, 0.5, 0.25, 0.25]), kinds)
    assert s == {"composition": 0.5, "redundancy": 0.25, "context": 0.125, "remaining": 0.125}
    assert all(math.isnan(v) for v in v2b.b3_shares(np.zeros(5), kinds).values())


def test_select_replays_one_per_group():
    recs = [{"object_id": f"{g}{i}", "region": reg, "seed": i, "status": "done" if i else "failed",
             "task_cfg": {"d_model": d}} for g, (reg, d) in enumerate([(v2b.MAIN, 48), (v2b.MAIN, 96),
                                                                        (v2b.ONELAYER, 96)]) for i in range(3)]
    sel = v2b.select_replays(recs)
    assert [v2b.config_group(r) for r in sel] == [(v2b.ONELAYER, 96), (v2b.MAIN, 48), (v2b.MAIN, 96)]  # sorted
    assert all(r["status"] == "done" for r in sel) and sel == v2b.select_replays(recs)


# ================================================================================================ end to end
def _write_object(tmp, oid, region, d, seed, n_layer, final_roles, rng, loss_end):
    H = n_layer * 3
    steps = np.array([0, 1, 10, 100])
    lo = np.stack([_network(H, rng, scale=0.5 + t) for t in range(steps.size)])
    np.savez_compressed(tmp / f"p_{oid}.npz", steps=steps, logodds=lo, counts=np.zeros(lo.shape, np.int64),
                        pmf_D=np.full(1 << H, 1 / (1 << H)), pmf_U=np.full(1 << H, 1 / (1 << H)),
                        pmf_S=np.full(1 << H, 1 / (1 << H)),
                        input_names=np.array([f"l{l}h{h}" for l in range(n_layer) for h in range(3)]),
                        n_probe=np.int64(1024))
    lc = {"step": steps, "x_loss_rep": np.array([4.2, 3.0, 0.8, loss_end]), "wall_s": np.zeros(4)}
    for l in range(n_layer):
        for h in range(3):
            role = final_roles[l * 3 + h]
            lc[f"x_prev_l{l}h{h}"] = np.array([0, 0.1, 0.2, 0.8 if role == P else 0.05])
            lc[f"x_prev2_l{l}h{h}"] = np.array([0, 0.1, 0.1, 0.3 if role == C else 0.05])
            lc[f"x_ind_l{l}h{h}"] = np.array([0, 0.1, 0.6, 0.9 if role == I else 0.05])
            lc[f"x_dup_l{l}h{h}"] = np.zeros(4)
    np.savez_compressed(tmp / f"l_{oid}.npz", **lc)
    return {"object_id": oid, "region": region, "task": "induction", "seed": seed, "sky_version": "test",
            "status": "done", "task_cfg": {"d_model": d, "n_layer": n_layer, "n_head": 3},
            "lightcurve": str(tmp / f"l_{oid}.npz"), "patterns": str(tmp / f"p_{oid}.npz")}


def test_analyse_end_to_end(tmp_path):
    rng = np.random.default_rng(12)
    recs = []
    for d in (48, 96):
        for s in (0, 1):
            recs.append(_write_object(tmp_path, f"m{d}{s}", v2b.MAIN, d, s, 2, FORMED, rng, 0.05))
        recs.append(_write_object(tmp_path, f"n{d}", v2b.NOREPEAT, d, 5, 2, UNFORMED, rng, 4.3))
    recs.append(_write_object(tmp_path, "o96", v2b.ONELAYER, 96, 5, 1, [P, C, N], rng, 2.0))
    recs.append({"object_id": "fail", "region": v2b.MAIN, "task": "induction", "seed": 2, "status": "failed",
                 "task_cfg": {"d_model": 96, "n_layer": 2, "n_head": 3}})
    man = tmp_path / "manifest.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in recs))
    cache = tmp_path / "cache"
    for r in recs:
        if r["status"] == "done":
            assert not v2b.stage1_object(r, cache, man)["cached"]
    assert v2b.stage1_object(recs[0], cache, man)["cached"]
    out = v2b.analyse(man, cache, B=100, min_eligible=2, replays=None, n_checkpoints=4)
    json.dumps(v2b.jsonable(out))  # JSON-friendly
    assert out["instrument_checks"]["data_checks"]["passed"], out["instrument_checks"]["data_checks"]
    assert out["instrument_checks"]["IC-2"]["passed"]
    assert out["status"]["withheld_because"] == ["IC-1 skipped"]
    assert out["M"]["decision"]["outcome"].startswith("withheld")
    assert out["B1"]["per_size"]["48"]["n_formed"] == 2 and out["B1"]["per_size"]["96"]["n_objects"] == 3
    assert out["M"]["family_sizes"] == [48, 96] and len(out["M"]["family_runs"]) == 4
    assert out["listing"]["not_done"][0]["object_id"] == "fail"
    run0 = next(x for x in out["runs"] if x["object_id"] == "m480")
    assert run0["roles"] == {"l0h0": P, "l0h1": C, "l0h2": N, "l1h0": I, "l1h1": N, "l1h2": N}
    assert len(run0["last_checkpoint_values"]["C_U"]["theta"]) == 41
    eb = run0["last_checkpoint_values"]
    for k in eb["C_U"]["eta_b"]:
        assert eb["C_U"]["eta_b"][k] == pytest.approx(eb["M_U"]["eta_b"][k], abs=1e-12)
    taus = out["descriptive"]["onsets"]["thresholds"]
    assert set(taus["a_no_repeat_p99"]["C_U:theta"]) == {"48", "96"} and taus["c_fixed"]["M_U:eta_g"] == 1e-3
    one = next(x for x in out["runs"] if x["object_id"] == "o96")
    assert len(one["last_checkpoint_values"]["C_U"]["theta"]) == 7  # 3 + 3 + 1 sets
