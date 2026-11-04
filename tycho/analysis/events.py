"""RI events and transients (SPEC §5): when do interactions switch on, switch off and switch on again?

An :class:`~tycho.analysis.ricurve.RICurve` holds the local RI (nats) and BH q-values of S variable subsets at T
checkpoints. :func:`detect_events` turns each subset's row into discrete events with a two-threshold (hysteresis)
state machine and a persistence requirement:

* A checkpoint is *active* iff q ≤ alpha and RI ≥ ``on`` (``on`` defaults to ``curve.meta["min_effect"]``, else
  1e-3; ``off`` defaults to on / 2). NaN RI or q is never active and never below ``off``, so it breaks persistence.
* inactive → active when ``persistence`` consecutive checkpoints t0 … t0+p−1 are all active. The onset is t0: the
  event's ``step`` is ``steps[t0]`` (the first active checkpoint, = ``step_hi``) and ``step_lo = steps[t0 − 1]`` (the
  last checkpoint before it), so the onset happened in (step_lo, step_hi]. A subset already active at the first
  checkpoint gets ``step_lo = step_hi = steps[0]`` and ``left_censored = True``. The first episode is an
  "activation", later ones are "reactivation"s.
* active → inactive ("extinction") when ``persistence`` consecutive checkpoints have RI < ``off``; only RI matters
  here (a still-large RI whose q rose above alpha keeps the subset active). The bracket is defined the same way.
* Activations that cannot be confirmed because fewer than ``persistence`` checkpoints remain are not reported.

Event fields: ``ri_before`` / ``ri_after`` are mean RIs over the ``persistence`` checkpoints before / from the
transition (NaN when there are none), ``peak_ri`` is the maximum RI of the episode that the event starts (activation,
reactivation) or ends (extinction). ``confidence`` is 1 − max q over the confirmation window for (re)activations
(≥ 1 − alpha by construction) and 1 − max(RI)/off over the window for extinctions (1 when RI fell to zero).

``has_precursor`` (activations and reactivations of order ≥ 3; None otherwise): True iff some proper subset of order
≥ 2 had an activation or reactivation strictly earlier, i.e. the interaction was assembled from lower-order parts.
Proper subsets that are not tracked by the curve count as never active, which is exact for curves pruned with
``RICurve.prune`` (it keeps every subset that was ever detected). ``matches_ground_truth``: the event's variable
names equal the names of a ground-truth entry as sets; None when the curve has no ground truth.

:func:`find_transients` raises alerts from events, the photometry summary and the classification; severities follow
:data:`TRANSIENT_SEVERITY`. Interactions that appear out of the expected low → high order, or after the training set
has been fitted, are the scientifically surprising ones (3); interactions that vanish or come back are notable (2);
known photometric phenomena and the unknown-class queue are informational (1).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from tycho.analysis.ricurve import RICurve

logger = logging.getLogger(__name__)

Subset = tuple[int, ...]

EVENT_KINDS: tuple[str, ...] = ("activation", "extinction", "reactivation")
TRANSIENT_KINDS: tuple[str, ...] = ("late_activation", "extinction", "reactivation", "pure_synergy",
                                    "order_inversion", "grokking", "loss_spike", "unknown_class")

TRANSIENT_SEVERITY: dict[str, int] = {
    "late_activation": 3,  # an interaction switches on after the training set has been fitted
    "pure_synergy": 3,  # an order >= 3 interaction with no lower-order (>= 2) part ever active
    "order_inversion": 3,  # a subset activates before all of its active proper subsets
    "extinction": 2,  # an active interaction switches off
    "reactivation": 2,  # ... and switches on again
    "grokking": 1,  # known photometric phenomenon (classification label)
    "loss_spike": 1,  # train loss above 2x its running minimum after the first drop (summary["loss_spikes"])
    "unknown_class": 1,  # light curve matched no rule; queued for review rather than pushed
}
"""Severity (1-3) of each transient kind. The default notifier threshold (``notify.min_severity`` = 2) pushes
late_activation, pure_synergy, order_inversion, extinction and reactivation."""

DEFAULT_MIN_EFFECT = 1e-3
"""Fallback activation threshold (nats) when neither ``on`` nor ``curve.meta["min_effect"]`` is given."""


@dataclass
class RIEvent:
    """One state change of one subset's RI (see the module docstring for every field's definition)."""

    subset: Subset
    subset_names: tuple[str, ...]
    order: int
    kind: str
    step: int
    step_lo: int
    step_hi: int
    ri_before: float
    ri_after: float
    peak_ri: float
    confidence: float
    has_precursor: bool | None = None
    matches_ground_truth: bool | None = None
    left_censored: bool = False

    def __post_init__(self) -> None:
        self.subset = tuple(int(i) for i in self.subset)
        self.subset_names = tuple(str(n) for n in self.subset_names)
        self.order = int(self.order)
        self.step, self.step_lo, self.step_hi = int(self.step), int(self.step_lo), int(self.step_hi)
        for name in ("ri_before", "ri_after", "peak_ri", "confidence"):
            setattr(self, name, float(getattr(self, name)))
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}; expected one of {EVENT_KINDS}")
        if self.order != len(self.subset) or len(self.subset_names) != len(self.subset):
            raise ValueError("order and subset_names must match the subset length")
        if not self.step_lo <= self.step <= self.step_hi:
            raise ValueError(f"need step_lo <= step <= step_hi, got {self.step_lo}, {self.step}, {self.step_hi}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (lists instead of tuples; NaN floats are kept as NaN)."""
        return {"subset": list(self.subset), "subset_names": list(self.subset_names), "order": self.order,
                "kind": self.kind, "step": self.step, "step_lo": self.step_lo, "step_hi": self.step_hi,
                "ri_before": self.ri_before, "ri_after": self.ri_after, "peak_ri": self.peak_ri,
                "confidence": self.confidence, "has_precursor": self.has_precursor,
                "matches_ground_truth": self.matches_ground_truth, "left_censored": self.left_censored}


@dataclass
class Transient:
    """An alert-worthy observation. ``event_index`` indexes the event list given to :func:`find_transients`."""

    kind: str
    severity: int
    message: str
    event_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in TRANSIENT_KINDS:
            raise ValueError(f"unknown transient kind {self.kind!r}; expected one of {TRANSIENT_KINDS}")
        self.severity = int(self.severity)
        if self.severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {self.severity}")
        if self.event_index is not None:
            self.event_index = int(self.event_index)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "severity": self.severity, "message": self.message,
                "event_index": self.event_index}


# ---------------------------------------------------------------------------------------------- detection
def detect_events(curve: RICurve, *, alpha: float = 0.01, on: float | None = None, off: float | None = None,
                  persistence: int = 2) -> list[RIEvent]:
    """Activation / extinction / reactivation events of every subset, sorted by (step, order, subset).

    Raises:
        ValueError: if persistence < 1, alpha is not in (0, 1], or not 0 ≤ off ≤ on.
    """
    if on is None:
        meta_on = curve.meta.get("min_effect")
        on = DEFAULT_MIN_EFFECT if meta_on is None else float(meta_on)
    on = float(on)
    off = on / 2.0 if off is None else float(off)
    persistence = int(persistence)
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if not 0.0 <= off <= on:
        raise ValueError(f"need 0 <= off <= on (hysteresis), got on={on}, off={off}")
    if np.any(np.diff(curve.steps) <= 0):
        raise ValueError("RICurve steps must be strictly increasing")
    if curve.S == 0 or curve.T < persistence:
        return []

    with np.errstate(invalid="ignore"):
        active = (curve.q <= alpha) & (curve.ri >= on)
        below = curve.ri < off
    confirm_on = np.lib.stride_tricks.sliding_window_view(active, persistence, axis=1).all(axis=2)
    confirm_off = np.lib.stride_tricks.sliding_window_view(below, persistence, axis=1).all(axis=2)
    gt_sets = None if curve.ground_truth is None else {
        frozenset(str(n) for n in g.get("subset", [])) for g in curve.ground_truth}
    steps = [int(s) for s in curve.steps]

    events: list[RIEvent] = []
    for i in np.flatnonzero(confirm_on.any(axis=1)):
        ri, q = curve.ri[i], curve.q[i]
        subset, names = curve.subsets[i], curve.subset_names(i)
        matches = None if gt_sets is None else frozenset(names) in gt_sets
        common = dict(subset=subset, subset_names=names, order=len(subset), matches_ground_truth=matches)
        for k, (t0, t_end) in enumerate(_episodes(confirm_on[i], confirm_off[i], persistence)):
            peak = _nanmax(ri[t0:curve.T if t_end is None else t_end])
            events.append(RIEvent(
                kind="activation" if k == 0 else "reactivation", step=steps[t0], step_lo=steps[max(t0 - 1, 0)],
                step_hi=steps[t0], ri_before=_nanmean(ri[max(t0 - persistence, 0):t0]),
                ri_after=_nanmean(ri[t0:t0 + persistence]), peak_ri=peak,
                confidence=1.0 - float(np.max(q[t0:t0 + persistence])), left_censored=t0 == 0, **common))
            if t_end is not None:
                window = ri[t_end:t_end + persistence]
                conf = 1.0 if off == 0 else float(np.clip(1.0 - np.max(window) / off, 0.0, 1.0))
                events.append(RIEvent(
                    kind="extinction", step=steps[t_end], step_lo=steps[t_end - 1], step_hi=steps[t_end],
                    ri_before=_nanmean(ri[max(t_end - persistence, 0):t_end]), ri_after=_nanmean(window),
                    peak_ri=peak, confidence=conf, **common))
    events.sort(key=lambda e: (e.step, e.order, e.subset, EVENT_KINDS.index(e.kind)))
    _mark_precursors(events)
    logger.debug("detect_events: %d events in %d subsets x %d checkpoints", len(events), curve.S, curve.T)
    return events


def _episodes(confirm_on: np.ndarray, confirm_off: np.ndarray, p: int) -> list[tuple[int, int | None]]:
    """(onset, extinction or None) checkpoint indices of one subset's active episodes.

    ``confirm_on[t]`` / ``confirm_off[t]``: checkpoints t … t+p−1 are all active / all below ``off``.
    """
    episodes: list[tuple[int, int | None]] = []
    onset: int | None = None
    t, n = 0, confirm_on.size
    while t < n:
        if onset is None and confirm_on[t]:
            onset, t = t, t + p
        elif onset is not None and confirm_off[t]:
            episodes.append((onset, t))
            onset, t = None, t + p
        else:
            t += 1
    if onset is not None:
        episodes.append((onset, None))
    return episodes


def _proper_subsets(subset: Subset, min_order: int = 2) -> Iterator[Subset]:
    for k in range(min_order, len(subset)):
        yield from combinations(subset, k)


def _first_activation_steps(events: Sequence[RIEvent]) -> dict[Subset, int]:
    first: dict[Subset, int] = {}
    for ev in events:
        if ev.kind != "extinction":
            first[ev.subset] = min(first.get(ev.subset, ev.step), ev.step)
    return first


def _mark_precursors(events: list[RIEvent]) -> None:
    first = _first_activation_steps(events)
    for ev in events:
        if ev.kind == "extinction" or ev.order < 3:
            ev.has_precursor = None
        else:
            ev.has_precursor = any(first.get(sub, math.inf) < ev.step for sub in _proper_subsets(ev.subset))


# ---------------------------------------------------------------------------------------------- transients
def find_transients(events: Sequence[RIEvent], summary: Mapping[str, Any] | None, classification: Any, *,
                    late_frac: float = 0.5) -> list[Transient]:
    """Transients of one object: event-based ones in event order, then loss_spike, grokking and unknown_class.

    * late_activation: an activation whose whole onset bracket lies after the training set was fitted,
      ``step_lo ≥ summary["t_train_acc99"]``; when the summary has no accuracies at all (regression tasks), after
      ``late_frac`` of the run instead (``step_lo ≥ late_frac · steps_total``). A run with accuracies that never
      reached 99 % has no late activations, and neither has a left-censored activation (its onset is unknown).
    * pure_synergy: the first activation of an order ≥ 3 subset none of whose proper subsets of order ≥ 2 is ever
      active (in ``events``).
    * order_inversion: the first activation of a subset that precedes the first activations of all of its proper
      subsets (order ≥ 2) that ever activate, at least one of which does.
    * extinction / reactivation: one per such event.
    * loss_spike: ``summary["loss_spikes"] > 0``. grokking / unknown_class: classification label "grokking" /
      "unknown".

    ``classification`` may be a :class:`~tycho.analysis.classify.Classification`, a mapping with a "label" key, a
    label string, or None.
    """
    if not 0.0 <= late_frac <= 1.0:
        raise ValueError(f"late_frac must be in [0, 1], got {late_frac}")
    summary = summary or {}
    late = _late_threshold(summary, late_frac)
    first = _first_activation_steps(events)
    out: list[Transient] = []

    def add(kind: str, message: str, index: int | None = None) -> None:
        out.append(Transient(kind=kind, severity=TRANSIENT_SEVERITY[kind], message=message, event_index=index))

    seen: set[Subset] = set()
    for idx, ev in enumerate(events):
        what = _describe(ev)
        if ev.kind == "activation" and not ev.left_censored and late is not None and ev.step_lo >= late[0]:
            add("late_activation", f"late activation of {what} at step {ev.step} (onset in ({ev.step_lo}, "
                                   f"{ev.step_hi}]) {late[1]}", idx)
        if ev.kind != "extinction" and ev.subset not in seen:
            seen.add(ev.subset)
            if ev.order >= 3:
                parts = [sub for sub in _proper_subsets(ev.subset) if sub in first]
                if not parts:
                    add("pure_synergy", f"pure synergy: {what} activated at step {ev.step} and none of its "
                                        f"lower-order (>= 2) parts is ever active", idx)
                elif all(first[sub] > ev.step for sub in parts):
                    add("order_inversion", f"order inversion: {what} activated at step {ev.step} before all "
                                           f"{len(parts)} of its active lower-order parts (earliest at step "
                                           f"{min(first[sub] for sub in parts)})", idx)
        if ev.kind == "extinction":
            add("extinction", f"extinction of {what} at step {ev.step} (RI {ev.ri_before:.3g} -> "
                              f"{ev.ri_after:.3g} nats; peak {ev.peak_ri:.3g})", idx)
        elif ev.kind == "reactivation":
            add("reactivation", f"reactivation of {what} at step {ev.step} (RI {ev.ri_after:.3g} nats)", idx)

    spikes = _num(summary.get("loss_spikes"))
    if math.isfinite(spikes) and spikes > 0:
        add("loss_spike", f"{int(spikes)} loss spike(s): train loss rose above 2x its running minimum after the "
                          f"first drop")
    label = _label_of(classification)
    if label == "grokking":
        add("grokking", f"grokking: train acc 99% at step {_fmt(summary.get('t_train_acc99'))}, test acc 90% at "
                        f"step {_fmt(summary.get('t_test_acc90'))} (grok_ratio {_fmt(summary.get('grok_ratio'))})")
    elif label == "unknown":
        reasons = getattr(classification, "reasons", None) or (
            classification.get("reasons") if isinstance(classification, Mapping) else None)
        add("unknown_class", "light curve matched no known class" + (f": {reasons[0]}" if reasons else ""))
    return out


def _late_threshold(summary: Mapping[str, Any], late_frac: float) -> tuple[float, str] | None:
    """(step threshold, explanation) for late activations, or None when "late" is undefined."""
    has_acc = any(math.isfinite(_num(summary.get(k))) for k in ("max_train_acc", "final_train_acc", "t_train_acc99"))
    if has_acc:
        t99 = _num(summary.get("t_train_acc99"))
        if math.isfinite(t99):
            return t99, f"after train acc reached 99% at step {int(t99)}"
        return None
    total = _num(summary.get("steps_total"))
    if math.isfinite(total) and total > 0:
        return late_frac * total, f"after {late_frac:g} of the {int(total)} training steps"
    return None


def _describe(ev: RIEvent) -> str:
    gt = " [ground truth]" if ev.matches_ground_truth else ""
    return "{" + ", ".join(ev.subset_names) + "}" + f" (order {ev.order}){gt}"


def _label_of(classification: Any) -> str | None:
    if classification is None or isinstance(classification, str):
        return classification
    if isinstance(classification, Mapping):
        return classification.get("label")
    return getattr(classification, "label", None)


def _num(v: Any) -> float:
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _fmt(v: Any) -> str:
    """Integral values (steps) print as integers, others with 4 significant digits."""
    x = _num(v)
    if not math.isfinite(x):
        return "nan"
    return str(int(x)) if x == round(x) and abs(x) < 1e15 else f"{x:.4g}"


def _nanmean(a: np.ndarray) -> float:
    finite = a[np.isfinite(a)]
    return float(finite.mean()) if finite.size else math.nan


def _nanmax(a: np.ndarray) -> float:
    finite = a[np.isfinite(a)]
    return float(finite.max()) if finite.size else math.nan
