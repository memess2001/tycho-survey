"""Tests of tycho.analysis.classify: a zoo of realistic synthetic light curves, rule order, thresholds and tags."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tycho.analysis.classify import (
    GROK_MIN_RATIO,
    LABELS,
    NO_LEARN_ACC_GAIN,
    PLATEAU_MIN_LOGFRAC,
    RULESET,
    SCORE_FLOOR,
    UNKNOWN_SCORE,
    Classification,
    classify,
)
from tycho.analysis.summarize import SUMMARY_KEYS, summarize

LC_KEYS = ("step train_loss test_loss train_acc test_acc param_norm grad_norm update_norm dist_from_init "
           "erank_weights erank_hidden rep_cka_prev ntk_trace ntk_top_eig ntk_erank ntk_align ntk_velocity lr "
           "wall_s").split()


def eval_steps(total: int, n_log: int = 60, n_lin: int = 40) -> np.ndarray:
    log = np.round(np.logspace(0, np.log10(total), n_log)).astype(int)
    lin = np.round(np.linspace(0, total, n_lin)).astype(int)
    return np.unique(np.concatenate([[0], log, lin, [total]]))


def make_lc(step, **cols) -> dict[str, np.ndarray]:
    lc = {k: np.full(len(step), np.nan) for k in LC_KEYS}
    lc["step"] = np.asarray(step, dtype=np.int64)
    lc.update({k: np.asarray(v, dtype=np.float64) for k, v in cols.items()})
    return lc


def logistic(z):
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=np.float64) / 2.0))


def log_ramp(t, t0, width):
    """0 → 1 sigmoid in log step centred at t0 (``width`` in ln-step units)."""
    return logistic((np.log1p(t) - math.log(t0)) / width)


# ---------------------------------------------------------------------------------------------- the zoo
def grokking(rng):
    """Modular addition (p = 23): train acc saturates near 1e3 steps, test acc reaches 90 % near 2e4."""
    st, c = eval_steps(30000), 1 / 23
    tr_acc = c + (1 - c) * log_ramp(st, 450, 0.18)
    te_prog = log_ramp(st, 15000, 0.12)
    te_acc = c + (1 - c) * te_prog
    tr_loss = (math.log(23) * (1 - log_ramp(st, 400, 0.3)) + 2e-3) * np.exp(0.01 * rng.standard_normal(st.size))
    te_loss = math.log(23) + 3.0 * log_ramp(st, 600, 0.4) * (1 - te_prog) - (math.log(23) - 0.05) * te_prog
    return make_lc(st, train_loss=tr_loss, test_loss=te_loss,
                   train_acc=np.clip(tr_acc + 0.005 * rng.standard_normal(st.size), 0, 1),
                   test_acc=np.clip(te_acc + 0.005 * rng.standard_normal(st.size), 0, 1)), c


def plateau_drop(rng):
    """Sparse parity: loss sits at ln 2 for ~6000 steps, then falls by three decades within a few hundred."""
    st, c = eval_steps(20000), 0.5
    prog = logistic((st - 6000) / 150)
    loss = (0.002 + 0.691 * (1 - prog)) * np.exp(0.005 * rng.standard_normal(st.size))
    acc = np.clip(0.5 + 0.5 * prog + 0.01 * (1 - prog) * rng.standard_normal(st.size), 0, 1)
    return make_lc(st, train_loss=loss, test_loss=1.02 * loss, train_acc=acc, test_acc=acc), c


def staged(rng):
    """Two separated sharp drops (unigram → bigram-like stages)."""
    st, c = eval_steps(20000), 1 / 3
    loss = (0.60 + 0.35 * logistic((4000 - st) / 200) + 0.15 * logistic((300 - st) / 20)) * \
        np.exp(0.003 * rng.standard_normal(st.size))
    acc = c + 0.15 * logistic((st - 300) / 20) + 0.25 * logistic((st - 4000) / 200)
    return make_lc(st, train_loss=loss, test_loss=1.01 * loss, train_acc=acc, test_acc=acc - 0.01), c


def double_descent(rng):
    """Regression (no accuracies): smooth train loss; the test loss descends, bumps up near step 2000, descends."""
    st = eval_steps(20000)
    tr = (0.02 + (1 + st / 30) ** -0.7) * np.exp(0.003 * rng.standard_normal(st.size))
    te = 0.10 + (1 + st / 30) ** -0.5 + 0.3 * np.exp(-0.5 * ((np.log1p(st) - math.log(2000)) / 0.4) ** 2)
    return make_lc(st, train_loss=tr, test_loss=te), None


def diverged(rng):
    """Loss decreases, then becomes NaN (the runner stops at a non-finite loss)."""
    st, c = eval_steps(5000), 0.1
    loss = 2.3 * (1 + st / 50) ** -0.3
    loss[st > 3000] = np.nan
    acc = np.where(st > 3000, np.nan, 0.3)
    return make_lc(st, train_loss=loss, test_loss=loss, train_acc=acc, test_acc=acc), c


def no_learning(rng):
    """Binary task stuck at chance for the whole run."""
    st, c = eval_steps(20000), 0.5
    noise = rng.standard_normal((4, st.size))
    return make_lc(st, train_loss=0.6931 + 0.002 * noise[0], test_loss=0.6931 + 0.002 * noise[1],
                   train_acc=0.5 + 0.01 * noise[2], test_acc=0.5 + 0.01 * noise[3]), c


def memorization(rng):
    """Train set fitted, test accuracy at chance, test loss rising."""
    st, c = eval_steps(20000), 1 / 23
    return make_lc(st, train_loss=math.log(23) * (1 - log_ramp(st, 1200, 0.3)) + 1e-3,
                   test_loss=math.log(23) + 3 * log_ramp(st, 1500, 0.4),
                   train_acc=c + (1 - c) * log_ramp(st, 1500, 0.25),
                   test_acc=np.clip(c + 0.01 * rng.standard_normal(st.size), 0, 1)), c


def smooth(rng):
    """Saturating power-law decline with matching accuracies."""
    st, c = eval_steps(10000), 0.1
    loss = 0.05 + 2.0 * (1 + st / 20) ** -0.6
    acc = 0.1 + 0.85 * (1 - (1 + st / 20) ** -0.6)
    return make_lc(st, train_loss=loss * np.exp(0.01 * rng.standard_normal(st.size)), test_loss=1.1 * loss,
                   train_acc=acc, test_acc=acc - 0.02), c


ZOO = {"grokking": grokking, "plateau_drop": plateau_drop, "staged": staged, "double_descent": double_descent,
       "diverged": diverged, "no_learning": no_learning, "memorization": memorization, "smooth": smooth}


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("name", sorted(ZOO))
def test_zoo_labels(name, seed):
    lc, chance = ZOO[name](np.random.default_rng(seed))
    s = summarize(lc, chance)
    cls = classify(s, chance_acc=chance)
    assert cls.label == name, cls.reasons
    assert SCORE_FLOOR <= cls.score <= 1.0
    assert cls.reasons[0].startswith(f"{RULESET}: {name}: ")
    assert classify(s, lc, chance).label == name  # recomputing from the light curve agrees


def test_zoo_key_summaries():
    rng = np.random.default_rng(0)
    s = summarize(*grokking(rng))
    assert 600 <= s["t_train_acc99"] <= 2000 and 15000 <= s["t_test_acc90"] <= 25000
    assert s["grok_ratio"] >= 10 and s["grok_delay"] > 10000 and s["test_acc_at_train99"] < 0.2
    s = summarize(*plateau_drop(rng))
    assert s["n_drops"] == 1 and 0.28 <= s["plateau_frac"] <= 0.33 and s["largest_drop_lo"] <= 6000
    assert 5500 <= s["t_train_acc90"] <= 7000 and s["grok_ratio"] < GROK_MIN_RATIO
    s = summarize(*staged(rng))
    assert s["n_drops"] == 2 and s["first_drop_step"] < 1000 < 3000 < s["largest_drop_step"]
    s = summarize(*double_descent(rng))
    assert s["test_bump"] >= 0.1 and s["test_bump_recovery"] >= 1.0 and 1000 <= s["test_bump_step"] <= 4000
    assert s["n_drops"] == 0 and math.isnan(s["t_train_acc90"])
    s = summarize(*diverged(rng))
    assert s["diverged"] is True and s["steps_total"] == 5000
    s = summarize(*no_learning(rng))
    assert s["plateau_frac"] == 1.0 and s["n_drops"] == 0 and math.isnan(s["t_train_acc90"])
    s = summarize(*memorization(rng))
    assert s["final_train_acc"] >= 0.99 and math.isnan(s["t_test_acc90"]) and s["test_bump"] > 0.5
    s = summarize(*smooth(rng))
    assert s["n_drops"] == 0 and s["train_loss_decrease"] > 0.9


@pytest.mark.parametrize("tail", [np.inf, 1e4])
def test_infinite_or_exploding_loss_is_diverged(tail):
    st = eval_steps(5000)
    loss = 2.3 * (1 + st / 50) ** -0.3
    loss[-3:] = tail
    lc = make_lc(st, train_loss=loss, test_loss=loss, train_acc=np.full(st.size, 0.3),
                 test_acc=np.full(st.size, 0.3))
    s = summarize(lc, 0.1)
    assert s["diverged"] is True and classify(s, chance_acc=0.1).label == "diverged"


def test_labels_are_the_spec_list():
    assert LABELS == ["diverged", "no_learning", "memorization", "grokking", "plateau_drop", "staged",
                      "double_descent", "smooth", "unknown"]


# ---------------------------------------------------------------------------------------------- rule order
def base_summary(**over) -> dict:
    """A summary of a smoothly learning classification run (chance 0.1); keyword arguments override keys."""
    s = {k: math.nan for k in SUMMARY_KEYS}
    s.update(n_evals=100, steps_total=10000, chance_acc=0.1, diverged=False, n_drops=0, loss_spikes=0,
             initial_train_loss=2.3, final_train_loss=0.3, min_train_loss=0.3, train_loss_decrease=0.87,
             initial_test_loss=2.3, final_test_loss=0.35, min_test_loss=0.35, max_train_acc=0.95,
             final_train_acc=0.95, final_test_acc=0.93, plateau_frac=0.001, plateau_logfrac=0.1, test_bump=0.0,
             t_train_acc90=800.0, t_test_acc90=900.0)
    s.update(over)
    return s


GROKKING = dict(t_train_acc99=1000.0, t_test_acc90=20000.0, grok_ratio=20.0, test_acc_at_train99=0.05,
                steps_total=30000, max_train_acc=1.0, final_train_acc=1.0, final_test_acc=0.99)


def test_base_summary_is_smooth():
    assert classify(base_summary()).label == "smooth"


def test_diverged_beats_every_other_rule():
    cls = classify(base_summary(**GROKKING, diverged=True, n_drops=2))
    assert cls.label == "diverged" and cls.score == 1.0


def test_grokking_beats_staged_and_staged_beats_plateau_drop():
    assert classify(base_summary(**GROKKING, n_drops=2, drop_width_max=0.4)).label == "grokking"
    s = base_summary(n_drops=2, drop_width_max=0.4, first_drop_pre_frac=0.0, plateau_logfrac=0.9)
    assert classify(s).label == "staged"


def test_plateau_drop_beats_double_descent_which_beats_smooth():
    dd = dict(test_bump=0.3, test_first_descent=0.3, test_bump_recovery=1.0, test_bump_step=3000.0)
    s = base_summary(n_drops=1, drop_width_max=0.4, first_drop_pre_frac=0.05, plateau_logfrac=0.8, **dd)
    cls = classify(s)
    assert cls.label == "plateau_drop" and "test_bump" in cls.tags
    assert classify(base_summary(**dd)).label == "double_descent"


def test_memorization_needs_test_accuracy_never_at_90():
    memo = dict(max_train_acc=1.0, final_train_acc=1.0, final_test_acc=0.12)
    assert classify(base_summary(**memo, t_test_acc90=math.nan)).label == "memorization"
    assert classify(base_summary(**memo)).label != "memorization"  # test acc reached 90 % at some point


def test_loss_based_rules_without_accuracies():
    no_acc = dict(chance_acc=math.nan, max_train_acc=math.nan, final_train_acc=math.nan, final_test_acc=math.nan,
                  t_train_acc90=math.nan, t_test_acc90=math.nan)
    assert classify(base_summary(**no_acc, train_loss_decrease=0.02)).label == "no_learning"
    s = base_summary(**no_acc, train_loss_decrease=0.9, final_train_loss=0.2, final_test_loss=2.4)
    assert classify(s).label == "memorization"
    assert classify(base_summary(**no_acc)).label == "smooth"


# ---------------------------------------------------------------------------------------------- thresholds & score
def test_grok_ratio_threshold_and_monotone_score():
    at = classify(base_summary(**{**GROKKING, "grok_ratio": GROK_MIN_RATIO}))
    assert at.label == "grokking" and at.score == pytest.approx(SCORE_FLOOR)
    below = classify(base_summary(**{**GROKKING, "grok_ratio": 0.99 * GROK_MIN_RATIO}))
    assert below.label != "grokking"
    scores = [classify(base_summary(**{**GROKKING, "grok_ratio": r})).score for r in (3.0, 6.0, 12.0, 30.0, 100.0)]
    assert scores == sorted(scores) and scores[-2] == scores[-1] == 1.0


def test_grokking_requires_generalization_to_start_after_fitting():
    assert classify(base_summary(**{**GROKKING, "test_acc_at_train99": 0.7})).label != "grokking"


def test_no_learning_gain_threshold():
    just_below = 0.1 + 0.9 * NO_LEARN_ACC_GAIN * 0.99
    just_above = 0.1 + 0.9 * NO_LEARN_ACC_GAIN * 1.01
    assert classify(base_summary(max_train_acc=just_below)).label == "no_learning"
    assert classify(base_summary(max_train_acc=just_above)).label != "no_learning"
    assert classify(base_summary(max_train_acc=0.1)).score == 1.0


def test_plateau_logfrac_threshold_separates_plateau_drop_from_early_drop():
    drop = dict(n_drops=1, drop_width_max=0.4, first_drop_pre_frac=0.05, first_drop_step=30.0)
    assert classify(base_summary(**drop, plateau_logfrac=PLATEAU_MIN_LOGFRAC)).label == "plateau_drop"
    early = classify(base_summary(**drop, plateau_logfrac=0.9 * PLATEAU_MIN_LOGFRAC))
    assert early.label == "smooth" and "early_drop" in early.tags and early.score == SCORE_FLOOR
    staged_one = classify(base_summary(**{**drop, "first_drop_pre_frac": 0.3}, plateau_logfrac=0.1))
    assert staged_one.label == "staged"  # a gradual stage followed by one sharp drop


def test_double_descent_requires_recovery_and_a_first_descent():
    dd = dict(test_bump=0.3, test_first_descent=0.3, test_bump_recovery=1.0, test_bump_step=3000.0)
    assert classify(base_summary(**{**dd, "test_bump_recovery": 0.3})).label == "smooth"
    assert classify(base_summary(**{**dd, "test_first_descent": 0.0})).label == "smooth"


def test_unknown_and_insufficient_data():
    rising = classify(base_summary(final_train_loss=3.0))  # loss went up (not 10x): nothing matches
    assert rising.label == "unknown" and rising.score == UNKNOWN_SCORE
    assert "no rule matched" in rising.reasons[0]
    empty = classify({})
    assert empty.label == "unknown" and empty.score == 0.0 and "insufficient" in empty.reasons[0]


def test_scores_in_unit_interval_on_random_summaries():
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = base_summary(max_train_acc=rng.random(), final_train_acc=rng.random(), final_test_acc=rng.random(),
                         n_drops=int(rng.integers(0, 4)), drop_width_max=rng.random(),
                         first_drop_pre_frac=rng.random(), plateau_logfrac=rng.random(),
                         grok_ratio=10 ** rng.uniform(-1, 3), t_train_acc99=1000.0, t_test_acc90=5000.0,
                         test_bump=rng.random(), test_first_descent=rng.random(),
                         test_bump_recovery=2 * rng.random(), train_loss_decrease=rng.random(),
                         final_train_loss=3 * rng.random())
        cls = classify(s)
        assert cls.label in LABELS and 0.0 <= cls.score <= 1.0


# ---------------------------------------------------------------------------------------------- tags
def test_tags():
    s = base_summary(**GROKKING, loss_spikes=3, plateau_frac=0.2, final_test_loss=1.0, min_test_loss=0.3)
    cls = classify(s)
    assert cls.label == "grokking"
    assert cls.tags == ["loss_spikes", "slow_start", "late_generalization", "overfitting"]
    assert len(cls.reasons) == 1 + len(cls.tags) and any("3 evaluations" in r for r in cls.reasons)
    gap = classify(base_summary(final_train_acc=1.0, final_test_acc=0.7))
    assert "generalization_gap" in gap.tags and "late_generalization" not in gap.tags
    slow_both = classify(base_summary(t_train_acc99=9000.0, t_test_acc90=6000.0))  # late, but not after fitting
    assert "late_generalization" not in slow_both.tags
    assert "slow_start" not in classify(base_summary(max_train_acc=0.1, plateau_frac=1.0)).tags


# ---------------------------------------------------------------------------------------------- inputs
def test_catalog_style_summary_with_none_and_int_flags():
    s = {k: None for k in SUMMARY_KEYS}
    s.update(diverged=1)
    assert classify(s).label == "diverged"
    s = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in base_summary().items()}
    s["diverged"] = 0
    assert classify(s).label == "smooth"


def test_light_curve_fills_missing_keys_and_diverged_is_ored():
    lc, chance = plateau_drop(np.random.default_rng(1))
    assert classify({}, lc, chance).label == "plateau_drop"
    assert classify({"chance_acc": chance, "n_drops": None}, lc).label == "plateau_drop"  # None does not mask
    assert classify({"diverged": True}, lc, chance).label == "diverged"


def test_chance_from_summary_and_json_round_trip():
    lc, chance = no_learning(np.random.default_rng(2))
    s = json.loads(json.dumps(summarize(lc, chance)))
    cls = classify(s)  # chance_acc read from the summary
    assert cls.label == "no_learning"
    d = json.loads(json.dumps(cls.to_dict()))
    assert d["label"] == "no_learning" and d["ruleset"] == RULESET
    assert Classification(**d) == cls
