"""Tests of tycho.analysis.events: event detection on synthetic RICurves and transient detection."""
from __future__ import annotations

import json
import math
from itertools import combinations

import numpy as np
import pytest

from tycho.analysis.classify import Classification
from tycho.analysis.events import (
    TRANSIENT_KINDS,
    TRANSIENT_SEVERITY,
    RIEvent,
    Transient,
    detect_events,
    find_transients,
)
from tycho.analysis.ricurve import RICurve

STEPS = np.unique(np.round(np.logspace(0, 4, 30)).astype(np.int64))  # log-spaced checkpoints 1 … 10000
T = STEPS.size
ON = 1e-3  # min_effect; off defaults to ON / 2


def onset_row(onset: int, level: float = 0.01) -> np.ndarray:
    """RI row that is 0 before checkpoint ``onset`` and ``level`` from it on."""
    return np.where(np.arange(T) >= onset, level, 0.0)


def significance(ri: np.ndarray) -> np.ndarray:
    """q-values that are significant exactly where RI reaches the effect threshold."""
    return np.where(np.asarray(ri) >= ON, 1e-6, 0.5)


def make_curve(rows, subsets, *, q=None, names=None, ground_truth=None, meta=None, steps=STEPS) -> RICurve:
    ri = np.asarray(rows, dtype=np.float64).reshape(len(subsets), -1)
    q = significance(ri) if q is None else np.asarray(q, dtype=np.float64).reshape(ri.shape)
    if names is None:
        names = [f"x{i}" for i in range(1 + max((max(s) for s in subsets), default=-1))]
    return RICurve(steps=steps[:ri.shape[1]], names=names, subsets=subsets, ri=ri, ri_plugin=ri, p=q, q=q,
                   n_eff=np.full(ri.shape[1], 4096.0), ground_truth=ground_truth,
                   meta={"min_effect": ON, "alpha": 0.01} if meta is None else meta)


def kinds(events) -> list[str]:
    return [e.kind for e in events]


# ---------------------------------------------------------------------------------------------- detection
def test_activation_bracket_on_log_spaced_steps():
    row = onset_row(15)
    row[14] = 4e-4  # sub-threshold rise before the onset
    (ev,) = detect_events(make_curve([row], [(0, 1)]))
    assert ev.kind == "activation" and ev.subset == (0, 1) and ev.subset_names == ("x0", "x1") and ev.order == 2
    assert (ev.step_lo, ev.step, ev.step_hi) == (STEPS[14], STEPS[15], STEPS[15])
    assert ev.ri_before == pytest.approx(2e-4) and ev.ri_after == pytest.approx(0.01) and ev.peak_ri == 0.01
    assert ev.confidence == pytest.approx(1 - 1e-6)
    assert ev.has_precursor is None and ev.matches_ground_truth is None and ev.left_censored is False


@pytest.mark.parametrize("seed", range(10))
def test_noisy_switch_on_is_dated_within_one_checkpoint(seed):
    """Calibration-style (E4): a sigmoid switch-on in log step, observed with sampling noise."""
    rng = np.random.default_rng(seed)
    population = 0.02 * 0.5 * (1 + np.tanh((np.log(STEPS) - math.log(500.0)) / 0.6))
    observed = np.maximum(population + 2e-4 * rng.standard_normal(T), 0.0)
    q = np.array([math.erfc(math.sqrt(4096 * r)) for r in observed])  # LRT p-value of the 1-df chi-square null
    events = detect_events(make_curve([observed], [(0, 1)], q=q))
    assert kinds(events) == ["activation"]
    true_onset = int(np.argmax(population >= ON))
    assert abs(int(np.searchsorted(STEPS, events[0].step)) - true_onset) <= 1


def test_flicker_is_suppressed_by_persistence():
    alternating = np.where(np.arange(T) % 2 == 0, 0.01, 0.0)
    assert detect_events(make_curve([alternating], [(0, 1)])) == []
    flicker = detect_events(make_curve([alternating], [(0, 1)]), persistence=1)
    assert len(flicker) > T // 2 and kinds(flicker)[:3] == ["activation", "extinction", "reactivation"]
    isolated = np.where(np.arange(T) % 5 == 2, 0.05, 0.0)
    assert detect_events(make_curve([isolated], [(0, 1)])) == []


def test_hysteresis_band_keeps_a_subset_active():
    row = onset_row(5)
    row[10:20] = 0.7 * ON  # between off (= ON / 2) and on, and not significant there
    curve = make_curve([row], [(0, 1)])
    assert kinds(detect_events(curve)) == ["activation"]
    assert kinds(detect_events(curve, off=ON)) == ["activation", "extinction", "reactivation"]  # no hysteresis


def test_single_dip_below_off_is_not_an_extinction():
    row = onset_row(5)
    row[12] = 0.0
    assert kinds(detect_events(make_curve([row], [(0, 1)]))) == ["activation"]


def test_extinction_then_reactivation():
    row = np.zeros(T)
    row[4:10] = 0.02
    row[6] = 0.03
    row[15:] = 0.005
    act, ext, react = detect_events(make_curve([row], [(0, 1)]))
    assert kinds([act, ext, react]) == ["activation", "extinction", "reactivation"]
    assert act.step == STEPS[4] and act.peak_ri == 0.03
    assert (ext.step_lo, ext.step, ext.step_hi) == (STEPS[9], STEPS[10], STEPS[10])
    assert ext.ri_before == pytest.approx(0.02) and ext.ri_after == 0.0 and ext.peak_ri == 0.03
    assert ext.confidence == 1.0 and ext.has_precursor is None
    assert react.step == STEPS[15] and react.peak_ri == 0.005 and react.ri_before == 0.0
    row[10:15] = 0.3 * ON  # extinction to 60 % of the off threshold
    ext = detect_events(make_curve([row], [(0, 1)]))[1]
    assert ext.kind == "extinction" and ext.confidence == pytest.approx(1 - 0.3 / 0.5)


def test_q_gates_activation_but_not_persistence_of_activity():
    row = onset_row(5)
    q = np.full(T, 1e-6)
    q[:12] = 0.5  # large but not yet significant
    q[20:] = 0.9  # significance lost later while RI stays large: still active
    (ev,) = detect_events(make_curve([row], [(0, 1)], q=q))
    assert ev.kind == "activation" and ev.step == STEPS[12]
    q[12] = 0.02  # max q in the confirmation window sets the confidence
    q[13] = 0.004
    (ev,) = detect_events(make_curve([row], [(0, 1)], q=q), alpha=0.05)
    assert ev.confidence == pytest.approx(1 - 0.02)


def test_left_censored_activation():
    (ev,) = detect_events(make_curve([np.full(T, 0.01)], [(0, 1)]))
    assert ev.left_censored and ev.step == ev.step_lo == ev.step_hi == STEPS[0] and math.isnan(ev.ri_before)


def test_transitions_need_enough_checkpoints_to_be_confirmed():
    last_only = np.zeros(T)
    last_only[-1] = 0.01
    assert detect_events(make_curve([last_only], [(0, 1)])) == []
    (ev,) = detect_events(make_curve([last_only], [(0, 1)]), persistence=1)
    assert ev.step == STEPS[-1]
    dies_at_end = onset_row(5)
    dies_at_end[-1] = 0.0
    assert kinds(detect_events(make_curve([dies_at_end], [(0, 1)]))) == ["activation"]
    assert detect_events(make_curve([onset_row(T - 3)], [(0, 1)]), persistence=4) == []


def test_empty_curves():
    assert detect_events(RICurve.empty(["a", "b"], [(0, 1)])) == []  # T = 0
    no_subsets = RICurve(steps=STEPS, names=["a"], subsets=[], ri=np.zeros((0, T)), ri_plugin=np.zeros((0, T)),
                         p=np.zeros((0, T)), q=np.zeros((0, T)), n_eff=np.full(T, 4096.0))
    assert detect_events(no_subsets) == []  # S = 0
    assert detect_events(make_curve([[0.01]], [(0, 1)])) == []  # T = 1 < persistence


def test_nan_is_never_active_and_breaks_persistence():
    row = np.full(T, 0.01)
    row[1::2] = np.nan
    assert detect_events(make_curve([row], [(0, 1)])) == []
    q = np.full(T, np.nan)
    assert detect_events(make_curve([np.full(T, 0.01)], [(0, 1)], q=q)) == []
    active_with_holes = onset_row(5)
    active_with_holes[[10, 11, 12]] = np.nan  # missing estimates do not extinguish an active subset
    active_with_holes[15] = 0.04
    (ev,) = detect_events(make_curve([active_with_holes], [(0, 1)]))
    assert ev.peak_ri == 0.04


def test_thresholds_default_to_the_curve_meta():
    row = onset_row(5, level=0.005)
    assert detect_events(make_curve([row], [(0, 1)], meta={"min_effect": 0.01})) == []
    assert len(detect_events(make_curve([row], [(0, 1)], meta={}))) == 1  # fallback 1e-3
    assert len(detect_events(make_curve([row], [(0, 1)], meta={"min_effect": 0.01}), on=0.004)) == 1


@pytest.mark.parametrize("kwargs", [dict(persistence=0), dict(on=1e-3, off=2e-3), dict(alpha=0.0),
                                    dict(alpha=1.5)])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        detect_events(make_curve([onset_row(5)], [(0, 1)]), **kwargs)


def test_steps_must_increase():
    curve = make_curve([onset_row(5)], [(0, 1)], steps=STEPS[::-1].copy())
    with pytest.raises(ValueError, match="increasing"):
        detect_events(curve)


def test_precursors():
    subsets = [(0, 1), (0, 1, 2), (2, 3, 4), (3, 4), (0, 1, 3)]
    rows = [onset_row(5), onset_row(10), onset_row(10), onset_row(12), onset_row(5)]
    events = {e.subset: e for e in detect_events(make_curve(rows, subsets))}
    assert events[(0, 1)].has_precursor is None  # order 2: no lower-order parts
    assert events[(0, 1, 2)].has_precursor is True  # (0, 1) activated strictly earlier
    assert events[(2, 3, 4)].has_precursor is False  # its part (3, 4) activates only later
    assert events[(0, 1, 3)].has_precursor is False  # its part (0, 1) activates at the same checkpoint


def test_ground_truth_matching_is_by_names_and_order_insensitive():
    names = ["x0", "x1", "x2", "yhat"]
    subsets = [(0, 2, 3), (1, 3)]
    rows = [onset_row(8), onset_row(8)]
    gt = [{"subset": ["yhat", "x2", "x0"], "order": 3}]
    match = {e.subset: e.matches_ground_truth for e in detect_events(make_curve(rows, subsets, names=names,
                                                                               ground_truth=gt))}
    assert match == {(0, 2, 3): True, (1, 3): False}
    none = detect_events(make_curve(rows, subsets, names=names, ground_truth=None))
    assert {e.matches_ground_truth for e in none} == {None}
    empty = detect_events(make_curve(rows, subsets, names=names, ground_truth=[]))
    assert {e.matches_ground_truth for e in empty} == {False}


def test_events_are_sorted_and_independent_of_row_order():
    subsets = [(0, 1), (1, 2), (0, 1, 2), (2, 3)]
    rows = [onset_row(9), onset_row(4), onset_row(9), onset_row(2)]
    rows[3][14:] = 0.0
    events = detect_events(make_curve(rows, subsets))
    keys = [(e.step, e.order, e.subset) for e in events]
    assert keys == sorted(keys)
    perm = [2, 0, 3, 1]
    shuffled = detect_events(make_curve([rows[i] for i in perm], [subsets[i] for i in perm]))
    assert [e.to_dict() for e in shuffled] == [e.to_dict() for e in events]


def test_event_serialization_and_validation():
    (ev,) = detect_events(make_curve([onset_row(5)], [(0, 1)], ground_truth=[{"subset": ["x0", "x1"]}]))
    d = json.loads(json.dumps(ev.to_dict()))
    assert d["subset"] == [0, 1] and d["subset_names"] == ["x0", "x1"] and d["matches_ground_truth"] is True
    assert RIEvent(**{**d, "subset": tuple(d["subset"])}) == ev
    base = dict(subset=(0, 1), subset_names=("a", "b"), order=2, kind="activation", step=10, step_lo=5,
                step_hi=10, ri_before=0.0, ri_after=0.01, peak_ri=0.01, confidence=0.99)
    for bad in (dict(kind="birth"), dict(order=3), dict(subset_names=("a",)), dict(step_lo=11)):
        with pytest.raises(ValueError):
            RIEvent(**{**base, **bad})


# ---------------------------------------------------------------------------------------------- transients
def by_kind(transients) -> dict[str, list[Transient]]:
    out: dict[str, list[Transient]] = {}
    for tr in transients:
        out.setdefault(tr.kind, []).append(tr)
    return out


def test_pure_synergy_and_order_inversion():
    plan = {(0, 1, 2): 8, (0, 1): None, (1, 2): None,  # triple whose pairs never activate: pure synergy
            (3, 4, 5): 5, (3, 4): 12, (4, 5): 15,  # triple before all of its active pairs: inversion
            (6, 7, 8): 9, (6, 7): 3,  # assembled from an earlier pair: neither
            (0, 3, 6): 10, (0, 3): 10,  # simultaneous with its pair: not strictly before, so neither
            (9, 10, 11): 10, (9, 10): 5, (10, 11): 15}  # before one pair but not all: neither
    subsets = list(plan)
    rows = [np.zeros(T) if t is None else onset_row(t) for t in plan.values()]
    events = detect_events(make_curve(rows, subsets))
    found = by_kind(find_transients(events, {}, None))
    assert set(found) == {"pure_synergy", "order_inversion"}
    (syn,), (inv,) = found["pure_synergy"], found["order_inversion"]
    assert events[syn.event_index].subset == (0, 1, 2) and syn.severity == 3
    assert events[inv.event_index].subset == (3, 4, 5) and inv.severity == 3
    assert "x3" in inv.message and str(STEPS[12]) in inv.message


def test_extinction_reactivation_and_one_synergy_per_subset():
    row = np.zeros(T)
    row[4:10] = 0.02
    row[15:] = 0.02
    events = detect_events(make_curve([row], [(0, 1, 2)]))
    found = by_kind(find_transients(events, {}, None))
    assert sorted(found) == ["extinction", "pure_synergy", "reactivation"]
    assert len(found["pure_synergy"]) == 1 and found["pure_synergy"][0].event_index == 0
    assert found["extinction"][0].event_index == 1 and found["extinction"][0].severity == 2
    assert found["reactivation"][0].event_index == 2 and found["reactivation"][0].severity == 2


def test_late_activation_after_the_training_set_is_fitted():
    i99 = 20  # train accuracy reached 99 % at STEPS[20]
    subsets = [(0, 1), (1, 2), (2, 3)]
    rows = [onset_row(i99), onset_row(i99 + 1), onset_row(i99 + 4)]
    rows[2][i99 + 4:i99 + 6] = 0.0
    rows[2][i99 + 6:] = 0.0
    events = detect_events(make_curve(rows, subsets))
    summary = {"t_train_acc99": float(STEPS[i99]), "max_train_acc": 1.0, "steps_total": 10000}
    late = by_kind(find_transients(events, summary, None)).get("late_activation", [])
    # (0, 1): bracket (STEPS[19], STEPS[20]] straddles saturation -> not late; (1, 2): bracket after it -> late
    assert [events[t.event_index].subset for t in late] == [(1, 2)]
    assert late[0].severity == 3 and "99%" in late[0].message
    never_fitted = {"t_train_acc99": math.nan, "max_train_acc": 0.8, "steps_total": 10000}
    assert "late_activation" not in by_kind(find_transients(events, never_fitted, None))
    # a curve whose first checkpoint is late (e.g. a dense follow-up window): already-active subsets have an
    # unknown onset and are not late activations
    late_window = make_curve([np.full(5, 0.01), onset_row(2)[:5]], [(0, 1), (1, 2)], steps=STEPS[i99:i99 + 5])
    late = by_kind(find_transients(detect_events(late_window), summary, None)).get("late_activation", [])
    assert [t.message.split(" at step")[0] for t in late] == ["late activation of {x1, x2} (order 2)"]


def test_late_activation_without_accuracies_uses_late_frac():
    subsets = [(0, 1), (1, 2)]
    events = detect_events(make_curve([onset_row(22), onset_row(27)], subsets))  # STEPS[22] ~ 1.1e3, [27] ~ 5.3e3
    no_acc = {"t_train_acc99": math.nan, "max_train_acc": math.nan, "final_train_acc": math.nan,
              "steps_total": 10000}
    late = by_kind(find_transients(events, no_acc, None)).get("late_activation", [])
    assert [events[t.event_index].subset for t in late] == [(1, 2)]
    late = by_kind(find_transients(events, no_acc, None, late_frac=0.05)).get("late_activation", [])
    assert [events[t.event_index].subset for t in late] == [(0, 1), (1, 2)]
    with pytest.raises(ValueError):
        find_transients(events, no_acc, None, late_frac=1.5)


def test_classification_and_summary_transients():
    grok = Classification(label="grokking", score=0.9)
    found = find_transients([], {"loss_spikes": 2, "t_train_acc99": 1000, "t_test_acc90": 20000,
                                 "grok_ratio": 20.0}, grok)
    assert [(t.kind, t.severity, t.event_index) for t in found] == [("loss_spike", 1, None), ("grokking", 1, None)]
    assert "step 20000" in found[1].message and "step 1000," in found[1].message
    unknown = Classification(label="unknown", reasons=["rules-v1: unknown: no rule matched"])
    (t,) = find_transients([], {}, unknown)
    assert t.kind == "unknown_class" and "no rule matched" in t.message
    assert [t.kind for t in find_transients([], {}, {"label": "unknown"})] == ["unknown_class"]
    assert [t.kind for t in find_transients([], None, "grokking")] == ["grokking"]
    assert find_transients([], {"loss_spikes": 0}, Classification(label="smooth")) == []


def test_transient_validation_and_severity_map():
    assert set(TRANSIENT_SEVERITY) == set(TRANSIENT_KINDS)
    assert {k for k, v in TRANSIENT_SEVERITY.items() if v == 3} == {"late_activation", "pure_synergy",
                                                                    "order_inversion"}
    assert {k for k, v in TRANSIENT_SEVERITY.items() if v == 2} == {"extinction", "reactivation"}
    with pytest.raises(ValueError):
        Transient("supernova", 3, "?")
    with pytest.raises(ValueError):
        Transient("grokking", 4, "?")
    t = Transient("extinction", 2, "gone", 3)
    assert json.loads(json.dumps(t.to_dict())) == {"kind": "extinction", "severity": 2, "message": "gone",
                                                   "event_index": 3}


def test_parity_like_positive_control_end_to_end():
    """Order-4 pure synergy {x1, x3, x4, yhat} among all subsets containing yhat, with noisy q-values."""
    rng = np.random.default_rng(7)
    names = [f"x{i}" for i in range(6)] + ["yhat"]
    subsets = [tuple(sorted(c + (6,))) for k in (1, 2, 3) for c in combinations(range(6), k)]
    target = subsets.index((1, 3, 4, 6))
    ri = np.abs(1e-4 * rng.standard_normal((len(subsets), T)))
    ri[target, 18:] = 0.05 * (1 + 0.1 * rng.standard_normal(T - 18))
    q = rng.uniform(0, 1, ri.shape)
    q[target, 18:] = 1e-9
    noise_hits = rng.random(ri.shape) < 0.1  # isolated single-checkpoint false detections (flicker)
    noise_hits[:, 1:] &= ~noise_hits[:, :-1]
    noise_hits[target] = False
    q[noise_hits], ri[noise_hits] = 1e-4, 2e-3
    gt = [{"subset": ["yhat", "x4", "x3", "x1"], "order": 4}]
    curve = make_curve(ri, subsets, q=q, names=names, ground_truth=gt)
    assert noise_hits.sum() > 50  # plenty of flicker, all of it suppressed by persistence
    events = detect_events(curve)
    assert [(e.subset, e.kind) for e in events] == [((1, 3, 4, 6), "activation")]
    gt_event = events[0]
    assert gt_event.matches_ground_truth is True and gt_event.has_precursor is False
    assert (gt_event.step_lo, gt_event.step_hi) == (STEPS[17], STEPS[18])
    found = find_transients(events, {"t_train_acc99": float(STEPS[19]), "max_train_acc": 1.0}, None)
    assert [(t.kind, t.event_index) for t in found] == [("pure_synergy", 0)]  # activation before fitting: not late
    assert "[ground truth]" in found[0].message and "x1, x3, x4, yhat" in found[0].message


def test_min_peak_gates_event_transients():
    """Weak events stay in the catalog but do not raise transients; strong ones still do."""
    from tycho.analysis.events import RIEvent, find_transients

    def ev(subset, kind, step, peak):
        return RIEvent(subset=subset, subset_names=tuple(f"v{i}" for i in subset), order=len(subset), kind=kind,
                       step=step, step_lo=step - 1, step_hi=step, ri_before=0.0, ri_after=peak, peak_ri=peak,
                       confidence=0.99)

    events = [ev((0, 1, 2), "activation", 10, 0.003), ev((0, 1, 2), "extinction", 20, 0.003),
              ev((3, 4, 5), "activation", 30, 0.5)]
    all_kinds = [t.kind for t in find_transients(events, {}, None)]
    gated = find_transients(events, {}, None, min_peak=0.005)
    assert all_kinds.count("pure_synergy") == 2 and "extinction" in all_kinds
    assert [t.kind for t in gated] == ["pure_synergy"] and gated[0].event_index == 2


def test_ground_truth_events_are_informational():
    from tycho.analysis.events import RIEvent, find_transients

    ev = RIEvent(subset=(0, 1, 2), subset_names=("x0", "x1", "yhat"), order=3, kind="activation", step=10,
                 step_lo=9, step_hi=10, ri_before=0.0, ri_after=0.69, peak_ri=0.69, confidence=0.99,
                 matches_ground_truth=True)
    (t,) = find_transients([ev], {}, None)
    assert t.kind == "pure_synergy" and t.severity == 1 and "expected" in t.message
