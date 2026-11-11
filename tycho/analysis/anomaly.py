"""Anomaly scores of light curves within their peer group: the survey's "unknown" exit (sky v1).

The rule-based classifier (:mod:`tycho.analysis.classify`) almost always finds a label: in sky v0 not one of 6,855
objects was "unknown", because ``smooth`` accepts nearly every curve that falls. A curve can still be unlike every
other curve with its label in its region (a "smooth" run whose test loss ends ten times higher than its peers', a
"grokking" run that generalises a decade later than the rest). This module scores that, so that the scheduler can
give such objects RI follow-ups even though their label is ordinary.

Score
    Objects are grouped by (region, label) by the caller. Each object's summary is reduced to the
    :data:`FEATURES` below; for every feature the group's median m and median absolute deviation (MAD) give a
    robust z-score ``|x − m| / (1.4826 · max(MAD, floor))``, the floor keeping a near-constant feature from turning
    tiny differences into huge scores. The object's score is the largest z over its finite features, and the
    feature that attains it is reported. Groups smaller than :data:`MIN_GROUP` get no scores (None): with few peers
    "unusual" means little.

Times that were never reached (a run that never hit 90 % test accuracy) are set to ``log10(2 · steps_total)``, so
"never" counts as later than any observed time but stays finite.

The score is a triage heuristic like the classifier's own score, not a probability. :data:`THRESHOLD` was chosen on
DR0 (sky v0) so that about 1 % of each large group is flagged.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Mapping, Sequence

__all__ = ["FEATURES", "MIN_GROUP", "THRESHOLD", "anomaly_scores", "summary_features"]

MIN_GROUP = 20
"""Objects a (region, label) group needs before its members are scored."""

THRESHOLD = 8.0
"""Robust z-score from which an object counts as anomalous."""

_MAD_TO_SIGMA = 1.4826


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def _log_time(key: str) -> Callable[[Mapping[str, Any]], float]:
    def f(s: Mapping[str, Any]) -> float:
        t = _num(s.get(key))
        if math.isfinite(t):
            return math.log10(max(t, 1.0))
        total = _num(s.get("steps_total"))
        return math.log10(2.0 * total) if math.isfinite(total) and total > 0 else math.nan
    return f


def _log_value(key: str) -> Callable[[Mapping[str, Any]], float]:
    def f(s: Mapping[str, Any]) -> float:
        x = _num(s.get(key))
        return math.log10(max(x, 1e-8)) if math.isfinite(x) else math.nan
    return f


def _value(key: str) -> Callable[[Mapping[str, Any]], float]:
    return lambda s: _num(s.get(key))


FEATURES: dict[str, tuple[Callable[[Mapping[str, Any]], float], float]] = {
    "log_t_train_acc90": (_log_time("t_train_acc90"), 0.1),
    "log_t_test_acc90": (_log_time("t_test_acc90"), 0.1),
    "final_train_acc": (_value("final_train_acc"), 0.02),
    "final_test_acc": (_value("final_test_acc"), 0.02),
    "log_final_train_loss": (_log_value("final_train_loss"), 0.1),
    "log_final_test_loss": (_log_value("final_test_loss"), 0.1),
    "plateau_logfrac": (_value("plateau_logfrac"), 0.05),
    "n_drops": (_value("n_drops"), 0.5),
    "test_bump": (_value("test_bump"), 0.05),
}
"""Feature name → (function of a summary dict, MAD floor in the feature's units)."""


def summary_features(summary: Mapping[str, Any]) -> dict[str, float]:
    """The :data:`FEATURES` of one light-curve summary (NaN where the summary lacks the data)."""
    return {name: f(summary) for name, (f, _) in FEATURES.items()}


def anomaly_scores(summaries: Sequence[Mapping[str, Any] | None], *,
                   min_group: int = MIN_GROUP) -> list[tuple[float, str] | None]:
    """(score, deciding feature) of every summary of one peer group, in input order; None for a missing summary,
    a summary without any finite feature, or every member of a group smaller than ``min_group``."""
    feats = [summary_features(s) if s else None for s in summaries]
    n_valid = sum(f is not None for f in feats)
    out: list[tuple[float, str] | None] = [None] * len(feats)
    if n_valid < min_group:
        return out
    centre: dict[str, tuple[float, float]] = {}
    for name, (_, floor) in FEATURES.items():
        values = [f[name] for f in feats if f is not None and math.isfinite(f[name])]
        if len(values) < min_group:
            continue
        med = statistics.median(values)
        mad = statistics.median(abs(v - med) for v in values)
        centre[name] = (med, _MAD_TO_SIGMA * max(mad, floor))
    for i, f in enumerate(feats):
        if f is None:
            continue
        best: tuple[float, str] | None = None
        for name, (med, scale) in centre.items():
            x = f[name]
            if math.isfinite(x):
                z = abs(x - med) / scale
                if best is None or z > best[0]:
                    best = (z, name)
        out[i] = best
    return out
