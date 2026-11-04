"""Rule-based phenomenological classification of an object's light curve (rule set "rules-v1", SPEC §5).

The input is a :func:`tycho.analysis.summarize.summarize` dictionary. The rules are evaluated in the SPEC's first-match
order, listed below (note that it tries staged before plateau_drop, whereas :data:`LABELS` lists plateau_drop first);
every threshold is a module constant. "Gain" denotes accuracy above chance in units of the headroom,
(acc − c) / (1 − c), with c the chance accuracy.

diverged        ``summary["diverged"]`` is true.
(no data)       no finite initial train loss and no finite train accuracy: "unknown" with score 0.
no_learning     with accuracies and a known c: the maximum train gain < NO_LEARN_ACC_GAIN; otherwise (losses only):
                train_loss_decrease < NO_LEARN_MIN_LOSS_DECREASE.
memorization    with train and test accuracies and c: final train accuracy ≥ its 90 % threshold, final test gain
                < MEMO_MAX_TEST_GAIN and the test accuracy never reached its 90 % threshold; otherwise (no usable
                accuracies): the train loss fell by ≥ MEMO_MIN_TRAIN_DECREASE (relative) while the final test loss is
                less than MEMO_MAX_TEST_DECREASE below the initial one.
grokking        train accuracy reached 99 % and test accuracy 90 % (``t_train_acc99``, ``t_test_acc90``),
                grok_ratio ≥ GROK_MIN_RATIO, and the test gain at ``t_train_acc99`` was < GROK_MAX_EARLY_TEST_GAIN
                (generalization had not started when the training set was fitted). Absolute thresholds apply when
                c is unknown.
staged          ≥ 2 sharp drops of the train loss, or one sharp drop preceded by ≥ STAGED_MIN_PRE_FRAC of the total
                log-loss decrease (a gradual first stage followed by a sharp second one).
plateau_drop    ≥ 1 sharp drop and an initial plateau covering ≥ PLATEAU_MIN_LOGFRAC of the run in log step.
double_descent  test_bump ≥ DD_MIN_BUMP after a first descent ≥ DD_MIN_FIRST_DESCENT, and ≥ DD_MIN_RECOVERY of the
                bump undone afterwards (a bump that never comes down is tagged "test_bump"/"overfitting" instead).
smooth          the final train loss is ≥ SMOOTH_MIN_DECREASE (relative) below the initial one and there is no sharp
                drop, or a single sharp drop that is the initial descent itself (tag "early_drop").
unknown         nothing matched (score UNKNOWN_SCORE).

Score: a matched rule scores SCORE_FLOOR + (1 − SCORE_FLOOR) · m, where the margin m ∈ [0, 1] measures how far the
deciding statistic lies beyond its threshold, normalised by the documented full-score point (the weakest condition
counts when several decide). ``diverged`` scores 1. The score is a heuristic for triage (low = worth a look), not a
probability.

Tags (secondary observations, given with any label, in this order):
    loss_spikes          summary["loss_spikes"] ≥ 1.
    slow_start           plateau_frac ≥ TAG_SLOW_START_FRAC (not with labels diverged / no_learning).
    late_generalization  t_test_acc90 ≥ TAG_LATE_GEN_FRAC · steps_total and t_test_acc90 > t_train_acc99.
    test_bump            test_bump ≥ DD_MIN_BUMP and the label is not double_descent.
    overfitting          (final_test_loss − min_test_loss) / initial_test_loss ≥ TAG_OVERFIT_MIN_RISE.
    generalization_gap   final_train_acc − final_test_acc ≥ TAG_GEN_GAP_MIN.
    early_drop           label smooth with one sharp drop (the initial descent).
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from tycho.analysis.summarize import DROP_MAX_WIDTH, acc_threshold, summarize

logger = logging.getLogger(__name__)

RULESET = "rules-v1"

LABELS: list[str] = ["diverged", "no_learning", "memorization", "grokking", "plateau_drop", "staged",
                     "double_descent", "smooth", "unknown"]
"""All labels (SPEC order). Rules are tried in the order diverged → no_learning → memorization → grokking → staged →
plateau_drop → double_descent → smooth → unknown."""

SCORE_FLOOR = 0.5
"""Score of a rule that matched exactly at its threshold."""

UNKNOWN_SCORE = 0.25
"""Score of the label "unknown" when the summary has data but no rule matched."""

NO_LEARN_ACC_GAIN = 0.1
"""no_learning: maximum train gain below this (e.g. max train acc < 0.55 at c = 0.5). Full score at gain 0."""

NO_LEARN_MIN_LOSS_DECREASE = 0.05
"""no_learning without accuracies: relative train-loss decrease (L0 − Lmin)/L0 below this. Full score at 0."""

MEMO_TRAIN_LEVEL = 90
"""memorization: final train accuracy at least the threshold of this level (percent, see ``acc_threshold``)."""

MEMO_MAX_TEST_GAIN = 0.2
"""memorization: final test gain below this. Full score at gain 0 (and train accuracy 1)."""

MEMO_MIN_TRAIN_DECREASE = 0.5
"""memorization without accuracies: relative train-loss decrease at least this. Full score at 1."""

MEMO_MAX_TEST_DECREASE = 0.05
"""memorization without accuracies: relative decrease of the final test loss below this. Full score at ≤ 0."""

GROK_MIN_RATIO = 3.0
"""grokking: t_test_acc90 / t_train_acc99 at least this (generalization ≥ 3× later than fitting)."""

GROK_FULL_RATIO = 30.0
"""grokking: ratio at which the score reaches 1 (log-linear between GROK_MIN_RATIO and this)."""

GROK_MAX_EARLY_TEST_GAIN = 0.5
"""grokking: test gain at t_train_acc99 below this."""

STAGED_MIN_PRE_FRAC = 0.2
"""staged with one sharp drop: fraction of the total log-loss decrease achieved before the drop. Full score at 2×."""

PLATEAU_MIN_LOGFRAC = 0.4
"""plateau_drop: plateau_logfrac at least this (for 20k steps: the plateau lasts beyond step ≈ 50)."""

DD_MIN_BUMP = 0.05
"""double_descent: test_bump (bump height / initial test loss) at least this."""

DD_FULL_BUMP = 0.2
"""double_descent: test_bump at which the bump margin reaches 1."""

DD_MIN_FIRST_DESCENT = 0.05
"""double_descent: relative test-loss descent before the bump at least this (excludes initial transients)."""

DD_MIN_RECOVERY = 0.5
"""double_descent: fraction of the bump undone after its peak at least this. Full score at 1."""

SMOOTH_MIN_DECREASE = 0.05
"""smooth: relative decrease of the final train loss below the initial one at least this."""

SMOOTH_FULL_DECREASE = 0.5
"""smooth: relative final decrease at which the score reaches 1."""

TAG_SLOW_START_FRAC = 0.1
"""Tag "slow_start": plateau_frac ≥ this (not given with labels diverged / no_learning)."""

TAG_LATE_GEN_FRAC = 0.5
"""Tag "late_generalization": t_test_acc90 ≥ this fraction of steps_total and after t_train_acc99 (the test set was
generalized to in the second half of the run, after the training set had been fitted)."""

TAG_OVERFIT_MIN_RISE = 0.1
"""Tag "overfitting": (final_test_loss − min_test_loss) / initial_test_loss ≥ this."""

TAG_GEN_GAP_MIN = 0.2
"""Tag "generalization_gap": final_train_acc − final_test_acc ≥ this."""


@dataclass
class Classification:
    """Result of a classifier. ``score`` is a heuristic confidence in [0, 1]; ``reasons`` are human-readable."""

    label: str
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    ruleset: str = RULESET

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_Hit = tuple[float, str]  # (margin in [0, 1], explanation)


def classify(summary: Mapping[str, Any] | None, lc: Mapping[str, Any] | None = None,
             chance_acc: float | None = None) -> Classification:
    """Classify an object with rule set "rules-v1" (see the module docstring).

    Args:
        summary: output of :func:`summarize` (possibly round-tripped through JSON or SQLite: ``None`` reads as NaN).
        lc: optional light curve; when given, a summary is recomputed from it and the defined (non-None, non-NaN)
            entries of ``summary`` override the recomputed ones (``diverged`` is the OR of both), so older or partial
            summaries can be classified.
        chance_acc: chance accuracy; defaults to ``summary["chance_acc"]``.
    """
    summary = dict(summary or {})
    chance = _finite_or_none(chance_acc)
    if chance is None:
        chance = _finite_or_none(_num(summary.get("chance_acc")))
    if lc is not None:
        merged = summarize(lc, chance)
        diverged = _flag(summary.get("diverged")) or _flag(merged["diverged"])
        merged.update({k: v for k, v in summary.items() if not _missing(v)})
        merged["diverged"] = diverged
        summary = merged
    s = _Summary(summary)

    if _flag(summary.get("diverged")):
        label, score, reason = "diverged", 1.0, "train loss became non-finite or exploded"
    elif not (math.isfinite(s["initial_train_loss"]) or math.isfinite(s["max_train_acc"])):
        label, score, reason = "unknown", 0.0, "insufficient data: no finite train loss or train accuracy"
    else:
        label, score, reason = "unknown", UNKNOWN_SCORE, _no_match_reason(s)
        for name, rule in _RULES:
            hit = rule(s, chance)
            if hit is not None:
                margin, reason = hit
                label, score = name, SCORE_FLOOR + (1.0 - SCORE_FLOOR) * _clip01(margin)
                break
    tags, tag_reasons = _tags(s, label)
    cls = Classification(label=label, tags=tags, score=float(score),
                         reasons=[f"{RULESET}: {label}: {reason}", *tag_reasons])
    logger.debug("classified as %s (score %.2f, tags %s)", label, cls.score, tags)
    return cls


# ---------------------------------------------------------------------------------------------- rules
class _Summary:
    """Float view of a summary: missing keys and ``None`` read as NaN."""

    def __init__(self, d: Mapping[str, Any]) -> None:
        self._d = d

    def __getitem__(self, key: str) -> float:
        return _num(self._d.get(key))


def _gain(acc: float, chance: float) -> float:
    return (acc - chance) / (1.0 - chance)


def _has_acc(s: _Summary, chance: float | None) -> bool:
    return chance is not None and chance < 1.0 and math.isfinite(s["max_train_acc"])


def _rule_no_learning(s: _Summary, chance: float | None) -> _Hit | None:
    if _has_acc(s, chance):
        gain = _gain(s["max_train_acc"], chance)
        if gain < NO_LEARN_ACC_GAIN:
            return (1.0 - max(gain, 0.0) / NO_LEARN_ACC_GAIN,
                    f"max train acc {_f(s['max_train_acc'])} is within {NO_LEARN_ACC_GAIN:g} of the headroom above "
                    f"chance {_f(chance)} (gain {_f(gain)})")
        return None
    dec = s["train_loss_decrease"]
    if math.isfinite(dec) and dec < NO_LEARN_MIN_LOSS_DECREASE:
        return (1.0 - max(dec, 0.0) / NO_LEARN_MIN_LOSS_DECREASE,
                f"train loss decreased by only {_f(dec)} (relative) < {NO_LEARN_MIN_LOSS_DECREASE:g}")
    return None


def _rule_memorization(s: _Summary, chance: float | None) -> _Hit | None:
    if _has_acc(s, chance) and math.isfinite(s["final_test_acc"]):
        train_thr = acc_threshold(MEMO_TRAIN_LEVEL, chance)
        train, test = s["final_train_acc"], s["final_test_acc"]
        test_gain = _gain(test, chance)
        if train >= train_thr and test_gain < MEMO_MAX_TEST_GAIN and not math.isfinite(s["t_test_acc90"]):
            margin = min((train - train_thr) / max(1.0 - train_thr, 1e-12),
                         1.0 - max(test_gain, 0.0) / MEMO_MAX_TEST_GAIN)
            return (margin, f"final train acc {_f(train)} >= {_f(train_thr)} but final test acc {_f(test)} stays near "
                            f"chance {_f(chance)} (gain {_f(test_gain)} < {MEMO_MAX_TEST_GAIN:g})")
        return None
    train_dec = s["train_loss_decrease"]
    tl0, tlf = s["initial_test_loss"], s["final_test_loss"]
    if not (math.isfinite(train_dec) and math.isfinite(tl0) and math.isfinite(tlf) and tl0 > 0):
        return None
    test_dec = (tl0 - tlf) / tl0
    if train_dec >= MEMO_MIN_TRAIN_DECREASE and test_dec < MEMO_MAX_TEST_DECREASE:
        margin = min((train_dec - MEMO_MIN_TRAIN_DECREASE) / (1.0 - MEMO_MIN_TRAIN_DECREASE),
                     1.0 - max(test_dec, 0.0) / MEMO_MAX_TEST_DECREASE)
        return (margin, f"train loss fell by {_f(train_dec)} (relative) while the final test loss is only "
                        f"{_f(test_dec)} below its initial value")
    return None


def _rule_grokking(s: _Summary, chance: float | None) -> _Hit | None:
    t99, t90, ratio = s["t_train_acc99"], s["t_test_acc90"], s["grok_ratio"]
    if not (math.isfinite(t99) and math.isfinite(t90) and math.isfinite(ratio)) or ratio < GROK_MIN_RATIO:
        return None
    c = chance if chance is not None and chance < 1.0 else 0.0
    early = s["test_acc_at_train99"]
    if math.isfinite(early) and _gain(early, c) >= GROK_MAX_EARLY_TEST_GAIN:
        return None
    margin = math.log10(ratio / GROK_MIN_RATIO) / math.log10(GROK_FULL_RATIO / GROK_MIN_RATIO)
    return (margin, f"train acc reached 99% at step {_f(t99)}, test acc 90% only at step {_f(t90)} "
                    f"(grok_ratio {_f(ratio)} >= {GROK_MIN_RATIO:g}, test acc {_f(early)} when train acc hit 99%)")


def _sharpness_margin(s: _Summary) -> float:
    width = s["drop_width_max"]
    return 1.0 - width / DROP_MAX_WIDTH if math.isfinite(width) else 0.0


def _rule_staged(s: _Summary, chance: float | None) -> _Hit | None:
    n, pre = s["n_drops"], s["first_drop_pre_frac"]
    if n >= 2:
        return (_sharpness_margin(s), f"{int(n)} sharp drops of the train loss (first at step "
                                      f"{_f(s['first_drop_step'])}, widest W={_f(s['drop_width_max'])})")
    if n == 1 and math.isfinite(pre) and pre >= STAGED_MIN_PRE_FRAC:
        margin = min(_sharpness_margin(s), (pre - STAGED_MIN_PRE_FRAC) / STAGED_MIN_PRE_FRAC)
        return (margin, f"a gradual stage ({_f(pre)} of the log-loss decrease >= {STAGED_MIN_PRE_FRAC:g}) followed by "
                        f"a sharp drop at step {_f(s['first_drop_step'])}")
    return None


def _rule_plateau_drop(s: _Summary, chance: float | None) -> _Hit | None:
    n, logfrac = s["n_drops"], s["plateau_logfrac"]
    if n >= 1 and math.isfinite(logfrac) and logfrac >= PLATEAU_MIN_LOGFRAC:
        margin = min(_sharpness_margin(s), (logfrac - PLATEAU_MIN_LOGFRAC) / (1.0 - PLATEAU_MIN_LOGFRAC))
        return (margin, f"train loss stayed on its initial plateau until step {_f(s['t_plateau_end'])} "
                        f"(plateau_logfrac {_f(logfrac)} >= {PLATEAU_MIN_LOGFRAC:g}, plateau_frac "
                        f"{_f(s['plateau_frac'])}), then dropped sharply at step {_f(s['first_drop_step'])}")
    return None


def _rule_double_descent(s: _Summary, chance: float | None) -> _Hit | None:
    bump, first, rec = s["test_bump"], s["test_first_descent"], s["test_bump_recovery"]
    if not (math.isfinite(bump) and math.isfinite(first) and math.isfinite(rec)):
        return None
    if bump >= DD_MIN_BUMP and first >= DD_MIN_FIRST_DESCENT and rec >= DD_MIN_RECOVERY:
        margin = min((bump - DD_MIN_BUMP) / (DD_FULL_BUMP - DD_MIN_BUMP),
                     (rec - DD_MIN_RECOVERY) / (1.0 - DD_MIN_RECOVERY))
        return (margin, f"test loss descended ({_f(first)}), rose by test_bump {_f(bump)} >= {DD_MIN_BUMP:g} "
                        f"peaking at step {_f(s['test_bump_step'])}, then {_f(rec)} of the bump was undone")
    return None


def _rule_smooth(s: _Summary, chance: float | None) -> _Hit | None:
    l0, lf = s["initial_train_loss"], s["final_train_loss"]
    if not (math.isfinite(l0) and math.isfinite(lf) and l0 > 0):
        return None
    dec = (l0 - lf) / l0
    if dec < SMOOTH_MIN_DECREASE:
        return None
    n = s["n_drops"]
    if n == 0:
        return (dec / SMOOTH_FULL_DECREASE, f"train loss decreased by {_f(dec)} (relative) without sharp drops")
    if n == 1:  # staged / plateau_drop did not match, so the drop is the initial descent
        return (0.0, f"train loss decreased by {_f(dec)} (relative); its only sharp drop is the initial descent "
                     f"(step {_f(s['first_drop_step'])})")
    return None


_RULES: tuple[tuple[str, Callable[[_Summary, float | None], _Hit | None]], ...] = (
    ("no_learning", _rule_no_learning),
    ("memorization", _rule_memorization),
    ("grokking", _rule_grokking),
    ("staged", _rule_staged),
    ("plateau_drop", _rule_plateau_drop),
    ("double_descent", _rule_double_descent),
    ("smooth", _rule_smooth),
)


def _no_match_reason(s: _Summary) -> str:
    return (f"no rule matched (n_drops {_f(s['n_drops'])}, plateau_logfrac {_f(s['plateau_logfrac'])}, "
            f"test_bump {_f(s['test_bump'])}, final train loss {_f(s['final_train_loss'])} vs initial "
            f"{_f(s['initial_train_loss'])})")


# ---------------------------------------------------------------------------------------------- tags
def _tags(s: _Summary, label: str) -> tuple[list[str], list[str]]:
    tags: list[str] = []
    reasons: list[str] = []

    def add(tag: str, why: str) -> None:
        tags.append(tag)
        reasons.append(f"tag {tag}: {why}")

    if s["loss_spikes"] >= 1:
        add("loss_spikes", f"{int(s['loss_spikes'])} evaluations above 2x the running minimum of the train loss")
    if label not in ("diverged", "no_learning") and s["plateau_frac"] >= TAG_SLOW_START_FRAC:
        add("slow_start", f"plateau_frac {_f(s['plateau_frac'])} >= {TAG_SLOW_START_FRAC:g}")
    t90, t99, total = s["t_test_acc90"], s["t_train_acc99"], s["steps_total"]
    if math.isfinite(t90) and math.isfinite(t99) and total > 0 and t90 >= TAG_LATE_GEN_FRAC * total and t90 > t99:
        add("late_generalization", f"test acc reached 90% at step {_f(t90)} >= {TAG_LATE_GEN_FRAC:g} x {_f(total)} "
                                   f"steps, after train acc reached 99% at step {_f(t99)}")
    if label != "double_descent" and s["test_bump"] >= DD_MIN_BUMP:
        add("test_bump", f"test_bump {_f(s['test_bump'])} >= {DD_MIN_BUMP:g} at step {_f(s['test_bump_step'])}")
    tl0 = s["initial_test_loss"]
    if tl0 > 0:
        rise = (s["final_test_loss"] - s["min_test_loss"]) / tl0
        if rise >= TAG_OVERFIT_MIN_RISE:
            add("overfitting", f"final test loss is {_f(rise)} (of the initial) above its minimum")
    gap = s["final_train_acc"] - s["final_test_acc"]
    if gap >= TAG_GEN_GAP_MIN:
        add("generalization_gap", f"final train acc - final test acc = {_f(gap)} >= {TAG_GEN_GAP_MIN:g}")
    if label == "smooth" and s["n_drops"] == 1:
        add("early_drop", f"one sharp initial drop at step {_f(s['first_drop_step'])}")
    return tags, reasons


# ---------------------------------------------------------------------------------------------- helpers
def _num(v: Any) -> float:
    """Float value of a summary entry; None, non-numeric and missing values read as NaN."""
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _finite_or_none(v: float | None) -> float | None:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _flag(v: Any) -> bool:
    """Truthiness of a boolean summary entry; NaN and None read as False."""
    x = _num(v)
    return math.isfinite(x) and x != 0.0


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0)) if math.isfinite(x) else 0.0


def _f(x: float | None) -> str:
    """Integral values (steps, counts) print as integers, others with 4 significant digits."""
    if x is None or not math.isfinite(x):
        return "nan"
    return str(int(x)) if x == round(x) and abs(x) < 1e15 else f"{x:.4g}"
