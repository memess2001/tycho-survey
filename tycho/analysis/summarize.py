"""Photometry summary: scalar statistics of one object's light curve (SPEC §5).

A light curve ``lc`` maps the keys of SPEC §4.4 to 1-D arrays aligned with ``lc["step"]``. Only ``step``,
``train_loss``, ``test_loss``, ``train_acc``, ``test_acc`` and ``ntk_velocity`` are read; other keys are ignored.
Any of the five metrics may be missing or entirely NaN (accuracies of a regression task, eNTK statistics when
``ntk_every = 0``); every summary that cannot be computed is NaN.

Preprocessing
    Rows are stably sorted by step, and when a step occurs more than once (a resumed run may log it twice) the last
    row wins. Non-finite steps, or metric arrays whose length differs from ``step``, raise ``ValueError``.

Time coordinate
    Evaluation schedules are log-spaced plus linear, and plateaus and drops are scale-free phenomena, so shape
    statistics use the *log step* ``x = ln(1 + step)``. Fractions "of total steps" (``plateau_frac``) use plain steps
    as the SPEC asks; ``plateau_logfrac`` is the log-step counterpart used by the classifier.

Accuracy thresholds (:func:`acc_threshold`)
    For level ℓ ∈ {50, 90, 99}: the absolute accuracy ℓ/100 when the chance accuracy c is unknown or c ≤ 0.5, else
    the rescaled c + (ℓ/100)(1 − c). A crossing time is the first evaluated step with acc ≥ threshold − ACC_TOL.
    Note that for a balanced binary task (c = 0.5) the 50 % threshold equals chance, so ``t_*_acc50`` is then
    typically the first evaluation; the 90/99 levels are the informative ones there.

Sharp drops (:func:`find_drops`, ``n_drops``)
    1. y = ln(L + ε), ε = DROP_LOSS_FLOOR_REL · L_initial: the log loss with a soft floor, so that changes far below
       0.1 % of the initial loss (e.g. the margin-growth tail of a cross-entropy loss) are not resolved.
    2. y is linearly interpolated onto a uniform grid in x (spacing DROP_GRID_DX) and smoothed with a Gaussian of
       standard deviation DROP_SMOOTH_SIGMA (edge values repeated beyond the ends). The robust scatter σ_y of the
       observations around the smoothed curve (1.4826 · MAD of the residuals) measures the curve's own noise.
    3. Steepness s(x) = −dy/dx, the local log-log slope (a power law L ∝ t^−α has s = α everywhere).
    4. Background b(x) = max(0, min of s over x ± DROP_BACKGROUND_HALFWIDTH); excess e = s − b. A power law has
       e ≈ 0; a drop is a localized excess.
    5. Peaks of e are visited from the highest; each claims the contiguous, not yet claimed interval around it where
       e ≥ DROP_EXTENT_FRAC · e_peak. Over that interval: excess depth D = ∫ e dx (nats), width W = D / e_peak
       (ln-step units), new-low depth N = min(y before the interval) − y(end of interval).
    6. The interval is a drop iff W ≤ DROP_MAX_WIDTH (sharp), D ≥ max(DROP_MIN_DEPTH, DROP_MIN_DEPTH_FRAC · R,
       DROP_MIN_SNR · σ_y) with R = y(start) − min y the total decrease of the smoothed log loss (significant), and
       N ≥ DROP_NEW_LOW_FRAC · D (the loss reaches a new low; recoveries from loss spikes are not drops).
    Calibration on synthetic curves (see tests): a logistic transition in linear time with scale w at t* has
    W ≈ 0.4 (w/t* = 0.02) … 0.75 (w/t* = 0.2); exponential relaxations exp(−t/τ) have W ≥ 1.2 for any τ and power laws
    have no excess, so neither produces drops. Two drops less than a factor ≈ 2 apart in step are merged into one.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

SUMMARY_VERSION = "summarize-v1"
"""Stored in every summary so that catalog rows record which definitions produced them."""

ACC_LEVELS: tuple[int, ...] = (50, 90, 99)
"""Accuracy levels (percent) of the ``t_{train,test}_acc{level}`` crossing times."""

ACC_TOL = 1e-9
"""Tolerance of accuracy-threshold comparisons (absorbs float rounding of the rescaled thresholds)."""

EPS = 1e-12
"""Guard against division by zero (``max(test_loss(0), EPS)`` in ``test_bump``)."""

PLATEAU_RANGE_FRAC = 0.1
"""The plateau ends when the train loss first falls below L0 − PLATEAU_RANGE_FRAC · (L0 − Lmin) (SPEC), with the
two noise guards below. For a curve that learns, the SPEC term dominates and the guards change nothing; for a curve
that does not learn, L0 − Lmin is itself noise and without them plateau_frac would be ≈ 0 instead of 1."""

PLATEAU_HEAD = 3
"""For the plateau, L0 is the median of the first PLATEAU_HEAD finite train losses (robust to a noisy first value)."""

PLATEAU_MIN_REL_DROP = 0.01
"""Guard 1: the required fall is at least PLATEAU_MIN_REL_DROP · |L0|."""

PLATEAU_NOISE_K = 6.0
"""Guard 2: the required fall is at least PLATEAU_NOISE_K · σ_rel · |L0|, where σ_rel is the relative evaluation
noise of the train loss: 1.4826 · MAD of the second differences of ln(L + ε) / √6 (ε as for drops). This is
unbiased for multiplicative white noise, the same at every loss scale, and insensitive to a few sharp drops."""

LOSS_SPIKE_RATIO = 2.0
"""An evaluation after the plateau end is a loss spike when train loss > LOSS_SPIKE_RATIO × (running min before it)."""

LOSS_SPIKE_MIN_RISE = 1e-3
"""... and it exceeds the running min by more than LOSS_SPIKE_MIN_RISE · (L0 − Lmin), which keeps relative
fluctuations of an already negligible loss (e.g. 1e-5 → 3e-5 after a decrease from 3.1) from counting as spikes."""

DIVERGED_LOSS_FACTOR = 10.0
"""A finite run whose last train loss exceeds DIVERGED_LOSS_FACTOR × its initial train loss has diverged."""

DROP_LOSS_FLOOR_REL = 1e-3
"""Soft floor ε = DROP_LOSS_FLOOR_REL · L_initial of the log loss used for drop detection."""

DROP_GRID_DX = 0.01
"""Spacing (ln-step units) of the uniform log-step grid used for drop detection."""

DROP_SMOOTH_SIGMA = 0.15
"""Standard deviation (ln-step units) of the Gaussian smoother, i.e. about ±16 % in step."""

DROP_BACKGROUND_HALFWIDTH = 1.5
"""Half-width (ln-step units) of the running-minimum window that defines the background steepness."""

DROP_EXTENT_FRAC = 0.25
"""A drop extends around its peak while the excess steepness is ≥ DROP_EXTENT_FRAC × its peak value."""

DROP_MAX_WIDTH = 0.8
"""Maximum width W = D / e_peak (ln-step units) of a sharp drop; about a factor 2-3 in step for the bulk of it."""

DROP_MIN_DEPTH = 0.03
"""Minimum excess depth of a drop in nats (a 3 % decrease of the loss)."""

DROP_MIN_DEPTH_FRAC = 0.1
"""Minimum excess depth of a drop as a fraction of the total decrease R of the smoothed log loss."""

DROP_MIN_SNR = 5.0
"""Minimum excess depth of a drop in units of the curve's robust scatter σ_y."""

DROP_NEW_LOW_FRAC = 0.75
"""Minimum new-low depth N of a drop as a fraction of its excess depth D."""

DROP_MIN_POINTS = 5
"""Minimum number of finite evaluations at distinct steps for drop detection."""

SUMMARY_KEYS: tuple[str, ...] = (
    "summary_version", "n_evals", "steps_total", "chance_acc",
    "final_train_loss", "final_test_loss", "final_train_acc", "final_test_acc",
    "initial_train_loss", "initial_test_loss", "min_train_loss", "min_test_loss", "train_loss_decrease",
    "max_train_acc", "max_test_acc",
    "t_train_acc50", "t_train_acc90", "t_train_acc99", "t_test_acc50", "t_test_acc90", "t_test_acc99",
    "test_acc_at_train99", "grok_delay", "grok_ratio",
    "plateau_frac", "plateau_logfrac", "t_plateau_end",
    "n_drops", "first_drop_step", "first_drop_pre_frac", "largest_drop_step", "largest_drop_lo", "largest_drop_hi",
    "largest_drop_depth", "drop_width_max",
    "loss_spikes", "test_bump", "test_bump_step", "test_bump_recovery", "test_first_descent",
    "diverged", "max_ntk_velocity", "step_max_ntk_velocity",
)
"""Every key returned by :func:`summarize`, in a stable order."""

_METRICS = ("train_loss", "test_loss", "train_acc", "test_acc", "ntk_velocity")
_INT_KEYS = frozenset({"n_evals", "steps_total", "n_drops", "loss_spikes"})


@dataclass(frozen=True)
class LossDrop:
    """One sharp drop of the train loss (see the module docstring for the definitions).

    Attributes:
        step: step of steepest excess descent of the log loss vs log step (interpolated, rounded to an integer
            step). For a logistic cliff L∞ + ΔL·σ((t* − t)/w) this lies in its exponential tail, a few w after the
            linear midpoint t*, because that is where ln L falls fastest; the bracket below contains both.
        step_lo, step_hi: evaluated steps bracketing the drop (largest step ≤ its start, smallest step ≥ its end).
        depth: excess depth D in nats of log loss.
        width: W = D / peak excess steepness, in ln-step units (smaller is sharper).
        steepness: peak of −d ln(L + ε)/d ln(1 + step), the local log-log slope at the drop.
        pre_frac: fraction of the total decrease R of the smoothed log loss achieved before the drop starts.
    """

    step: int
    step_lo: int
    step_hi: int
    depth: float
    width: float
    steepness: float
    pre_frac: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------------- public helpers
def acc_threshold(level: float, chance_acc: float | None) -> float:
    """Accuracy threshold for ``level`` percent: absolute when chance ≤ 0.5 (or unknown), else rescaled."""
    frac = float(level) / 100.0
    if chance_acc is None or not math.isfinite(chance_acc) or chance_acc <= 0.5:
        return frac
    return float(chance_acc) + frac * (1.0 - float(chance_acc))


def find_drops(step: ArrayLike, loss: ArrayLike) -> list[LossDrop]:
    """Sharp drops of a loss curve as a function of log step, sorted by step (module docstring, steps 1-6).

    Only evaluations with a finite loss and step ≥ 0 are used (sorted; a repeated step keeps its last value);
    negative losses are clipped to 0 (Tycho losses are cross-entropies or squared errors). Returns ``[]`` when fewer
    than DROP_MIN_POINTS evaluations remain or the total decrease R of the smoothed log loss is itself below the
    depth threshold.
    """
    s = np.asarray(step, dtype=np.float64).reshape(-1)
    loss_arr = np.asarray(loss, dtype=np.float64).reshape(-1)
    if s.shape != loss_arr.shape:
        raise ValueError(f"step and loss must have the same length, got {s.shape} and {loss_arr.shape}")
    ok = np.isfinite(s) & np.isfinite(loss_arr) & (s >= 0)
    s, loss_arr = s[ok], loss_arr[ok]
    idx = _sorted_last_rows(s)
    s, loss_arr = s[idx], loss_arr[idx]
    if s.size < DROP_MIN_POINTS:
        return []
    if np.any(loss_arr < 0):
        logger.warning("find_drops: %d negative loss values clipped to 0", int(np.sum(loss_arr < 0)))
        loss_arr = np.maximum(loss_arr, 0.0)
    floor = max(DROP_LOSS_FLOOR_REL * loss_arr[0], EPS)
    x = np.log1p(s)
    y = np.log(loss_arr + floor)
    n_grid = int(np.floor((x[-1] - x[0]) / DROP_GRID_DX)) + 1
    if n_grid < 3:
        return []
    grid = x[0] + DROP_GRID_DX * np.arange(n_grid)
    ys = _gaussian_smooth(np.interp(grid, x, y), DROP_SMOOTH_SIGMA / DROP_GRID_DX)
    total = float(ys[0] - ys.min())
    resid = y - np.interp(x, grid, ys)
    scatter = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    min_depth = max(DROP_MIN_DEPTH, DROP_MIN_DEPTH_FRAC * total, DROP_MIN_SNR * scatter)
    if total < min_depth:
        return []
    steep = -np.gradient(ys, DROP_GRID_DX)
    background = np.maximum(_running_min(steep, int(round(DROP_BACKGROUND_HALFWIDTH / DROP_GRID_DX))), 0.0)
    excess = steep - background

    rising = np.ones(n_grid, dtype=bool)
    rising[1:] = excess[1:] >= excess[:-1]
    falling = np.ones(n_grid, dtype=bool)
    falling[:-1] = excess[:-1] > excess[1:]
    peaks = np.flatnonzero(rising & falling & (excess > 0))
    peaks = peaks[np.argsort(-excess[peaks], kind="stable")]
    claimed = np.zeros(n_grid, dtype=bool)
    drops: list[LossDrop] = []
    for i in peaks:
        if claimed[i]:
            continue
        cut = DROP_EXTENT_FRAC * excess[i]
        lo = hi = int(i)
        while lo > 0 and not claimed[lo - 1] and excess[lo - 1] >= cut:
            lo -= 1
        while hi < n_grid - 1 and not claimed[hi + 1] and excess[hi + 1] >= cut:
            hi += 1
        claimed[lo:hi + 1] = True
        depth = float(np.trapezoid(excess[lo:hi + 1], dx=DROP_GRID_DX))
        width = depth / float(excess[i])
        new_low = float(ys[:lo + 1].min() - ys[hi])
        if width > DROP_MAX_WIDTH or depth < min_depth or new_low < DROP_NEW_LOW_FRAC * depth:
            continue
        t_lo, t_hi = np.expm1(grid[lo]), np.expm1(grid[hi])
        k_lo = max(int(np.searchsorted(s, t_lo + 1e-9, side="right")) - 1, 0)
        k_hi = min(int(np.searchsorted(s, t_hi - 1e-9, side="left")), s.size - 1)
        drops.append(LossDrop(step=int(round(float(np.expm1(grid[i])))), step_lo=int(s[k_lo]), step_hi=int(s[k_hi]),
                              depth=depth, width=width, steepness=float(steep[i]),
                              pre_frac=float(np.clip((ys[0] - ys[lo]) / total, 0.0, 1.0))))
    drops.sort(key=lambda d: d.step)
    return drops


# ---------------------------------------------------------------------------------------------- summarize
def summarize(lc: Mapping[str, Any], chance_acc: float | None = None) -> dict[str, Any]:
    """Scalar summary of a light curve. All keys of :data:`SUMMARY_KEYS` are always present.

    Counts (``n_evals``, ``steps_total``, ``n_drops``, ``loss_spikes``) are ints, ``diverged`` is a bool,
    ``summary_version`` a string and every other value a float (NaN when undefined); step-valued keys are floats.
    L0 / Lmin / TL0 below are the first finite / minimum finite train loss and the first finite test loss.

    SPEC keys:
        final_{train,test}_{loss,acc}: value at the last evaluation (NaN if not finite there).
        min_test_loss: minimum finite test loss.
        t_{train,test}_acc{50,90,99}: first step with acc ≥ :func:`acc_threshold` (NaN if never).
        grok_delay: t_test_acc90 − t_train_acc99 (may be negative). grok_ratio: t_test_acc90 / max(t_train_acc99, 1).
        plateau_frac: t_plateau_end / steps_total, 1.0 if the plateau never ends.
        n_drops: number of sharp drops of the smoothed log train loss vs log step (:func:`find_drops`).
        loss_spikes: evaluations after the plateau end (the "first drop") whose train loss exceeds
            LOSS_SPIKE_RATIO × the running minimum of the earlier evaluations (and the rise floor LOSS_SPIKE_MIN_RISE).
        test_bump: max(0, max_t [TL(t) − min_{s<t} TL(s)]) / max(TL0, EPS); a monotone curve gives 0.
        diverged: a non-finite train loss after a finite one, an infinite train or test loss, or a last finite train
            loss above DIVERGED_LOSS_FACTOR × L0.
        max_ntk_velocity, step_max_ntk_velocity: maximum finite ntk_velocity and the first step attaining it.
        steps_total: last evaluated step (0 for an empty light curve).

    Extra keys:
        summary_version, n_evals (distinct evaluated steps), chance_acc (NaN if None).
        initial_train_loss (L0), initial_test_loss (TL0), min_train_loss (Lmin),
        train_loss_decrease: (L0 − Lmin) / L0 (NaN if L0 ≤ 0).
        max_train_acc, max_test_acc: maxima of the finite accuracies.
        test_acc_at_train99: test accuracy at the evaluation where train accuracy first reached its 99 % threshold.
        t_plateau_end: first step with train loss < L0' − max(PLATEAU_RANGE_FRAC (L0' − Lmin), PLATEAU_MIN_REL_DROP
            |L0'|, PLATEAU_NOISE_K σ_L), L0' = median of the first PLATEAU_HEAD train losses (see those constants).
        plateau_logfrac: ln(1 + t_plateau_end) / ln(1 + steps_total), 1.0 if the plateau never ends.
        first_drop_step, first_drop_pre_frac: ``step`` and ``pre_frac`` of the earliest sharp drop.
        largest_drop_step / _lo / _hi / _depth: the deepest sharp drop and its bracket of evaluated steps
            (the natural window for dense RI follow-up).
        drop_width_max: largest width W among the sharp drops.
        test_bump_step: step of the test-loss bump peak (NaN when test_bump = 0).
        test_bump_recovery: (TL(peak) − min TL after the peak) / bump height; NaN if the peak is the last point.
        test_first_descent: (TL0 − min TL before the peak) / TL0, the descent that preceded the bump.
    """
    step, cols = _prepare(lc)
    chance = float(chance_acc) if chance_acc is not None and math.isfinite(float(chance_acc)) else math.nan
    out: dict[str, Any] = {k: math.nan for k in SUMMARY_KEYS}
    out.update(summary_version=SUMMARY_VERSION, n_evals=int(step.size), steps_total=0, chance_acc=chance,
               n_drops=0, loss_spikes=0, diverged=False)
    if step.size == 0:
        return out
    total = float(step[-1])
    out["steps_total"] = int(step[-1])
    chance_opt = None if math.isnan(chance) else chance

    for split in ("train", "test"):
        loss, acc = cols[f"{split}_loss"], cols[f"{split}_acc"]
        out[f"final_{split}_loss"] = _finite_or_nan(loss[-1])
        out[f"final_{split}_acc"] = _finite_or_nan(acc[-1])
        out[f"max_{split}_acc"] = _nanmax(acc)
        for level in ACC_LEVELS:
            out[f"t_{split}_acc{level}"] = _first_step_at_or_above(step, acc, acc_threshold(level, chance_opt))

    t99, t90 = out["t_train_acc99"], out["t_test_acc90"]
    if math.isfinite(t99):
        out["test_acc_at_train99"] = _finite_or_nan(cols["test_acc"][np.searchsorted(step, t99)])
        if math.isfinite(t90):
            out["grok_delay"] = t90 - t99
            out["grok_ratio"] = t90 / max(t99, 1.0)

    _train_loss_stats(out, step, cols["train_loss"], total)
    _test_loss_stats(out, step, cols["test_loss"])
    out["diverged"] = _diverged(cols["train_loss"], cols["test_loss"])

    ntk = cols["ntk_velocity"]
    finite_ntk = np.isfinite(ntk)
    if finite_ntk.any():
        k = int(np.argmax(np.where(finite_ntk, ntk, -np.inf)))
        out["max_ntk_velocity"] = float(ntk[k])
        out["step_max_ntk_velocity"] = float(step[k])

    return {k: (int(v) if k in _INT_KEYS else v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------- internals
def _prepare(lc: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sorted, de-duplicated steps and the metric columns (NaN-filled when missing)."""
    if "step" not in lc or lc["step"] is None:
        if any(np.size(lc[k]) for k in _METRICS if k in lc and lc[k] is not None):
            raise ValueError("light curve has metric arrays but no 'step' key")
        return np.zeros(0), {k: np.zeros(0) for k in _METRICS}
    step = np.asarray(lc["step"], dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(step)):
        raise ValueError("light-curve steps must be finite")
    cols: dict[str, np.ndarray] = {}
    for key in _METRICS:
        val = lc.get(key)
        if val is None:
            cols[key] = np.full(step.shape, np.nan)
            continue
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        if arr.shape != step.shape:
            raise ValueError(f"light-curve key {key!r} has length {arr.size}, expected {step.size} (len(step))")
        cols[key] = arr
    idx = _sorted_last_rows(step)
    return step[idx], {k: v[idx] for k, v in cols.items()}


def _sorted_last_rows(step: np.ndarray) -> np.ndarray:
    """Row indices that sort ``step`` stably, keeping only the last row of each repeated step."""
    order = np.argsort(step, kind="stable")
    keep = np.ones(order.size, dtype=bool)
    keep[:-1] = step[order][1:] != step[order][:-1]
    return order[keep]


def _train_loss_stats(out: dict[str, Any], step: np.ndarray, loss: np.ndarray, total: float) -> None:
    """Plateau, loss-spike and sharp-drop statistics of the train loss (writes into ``out``)."""
    finite = np.isfinite(loss)
    if not finite.any():
        return
    s, v = step[finite], loss[finite]
    l0, lmin = float(v[0]), float(v.min())
    out["initial_train_loss"] = l0
    out["min_train_loss"] = lmin
    if v.size < 2:  # a single value has no plateau, decrease, spikes or drops
        return
    if l0 > 0:
        out["train_loss_decrease"] = (l0 - lmin) / l0
    level = float(np.median(v[:PLATEAU_HEAD]))
    needed = max(PLATEAU_RANGE_FRAC * (level - lmin), PLATEAU_MIN_REL_DROP * abs(level),
                 PLATEAU_NOISE_K * _relative_noise(v) * abs(level))
    below = np.flatnonzero(v < level - needed)
    t_end = float(s[below[0]]) if below.size else math.nan
    out["t_plateau_end"] = t_end
    if total > 0:
        plateau = t_end if below.size else total
        out["plateau_frac"] = plateau / total
        out["plateau_logfrac"] = math.log1p(plateau) / math.log1p(total)
    if below.size:
        prev_min = np.concatenate([[np.inf], np.minimum.accumulate(v)[:-1]])
        after = np.arange(v.size) > below[0]
        spikes = after & (v > LOSS_SPIKE_RATIO * prev_min) & (v - prev_min > LOSS_SPIKE_MIN_RISE * (l0 - lmin))
        out["loss_spikes"] = int(spikes.sum())
    drops = find_drops(s, v)
    out["n_drops"] = len(drops)
    if drops:
        largest = max(drops, key=lambda d: d.depth)
        out.update(first_drop_step=float(drops[0].step), first_drop_pre_frac=drops[0].pre_frac,
                   largest_drop_step=float(largest.step), largest_drop_lo=float(largest.step_lo),
                   largest_drop_hi=float(largest.step_hi), largest_drop_depth=largest.depth,
                   drop_width_max=max(d.width for d in drops))


def _test_loss_stats(out: dict[str, Any], step: np.ndarray, loss: np.ndarray) -> None:
    """min_test_loss, initial_test_loss and the test-bump statistics (writes into ``out``)."""
    finite = np.isfinite(loss)
    if not finite.any():
        return
    s, v = step[finite], loss[finite]
    out["initial_test_loss"] = float(v[0])
    out["min_test_loss"] = float(v.min())
    if v.size < 2:
        return
    denom = max(float(v[0]), EPS)
    prev_min = np.minimum.accumulate(v)[:-1]  # min over s < t, for t = 1 .. n-1
    rise = v[1:] - prev_min
    k = int(np.argmax(rise))
    height = float(rise[k])
    if height <= 0:
        out["test_bump"] = 0.0
        return
    peak = k + 1
    out["test_bump"] = height / denom
    out["test_bump_step"] = float(s[peak])
    out["test_first_descent"] = (float(v[0]) - float(prev_min[k])) / denom
    if peak < v.size - 1:
        out["test_bump_recovery"] = (float(v[peak]) - float(v[peak + 1:].min())) / height


def _diverged(train_loss: np.ndarray, test_loss: np.ndarray) -> bool:
    if np.isinf(train_loss).any() or np.isinf(test_loss).any():
        return True
    finite = np.isfinite(train_loss)
    if not finite.any():  # not measured: no evidence either way
        return False
    if not finite[int(np.argmax(finite)):].all():  # NaN after the first finite value
        return True
    v = train_loss[finite]
    return bool(v.size >= 2 and v[0] > 0 and v[-1] > DIVERGED_LOSS_FACTOR * v[0])


def _first_step_at_or_above(step: np.ndarray, values: np.ndarray, threshold: float) -> float:
    with np.errstate(invalid="ignore"):
        hit = np.flatnonzero(np.isfinite(values) & (values >= threshold - ACC_TOL))
    return float(step[hit[0]]) if hit.size else math.nan


def _relative_noise(loss: np.ndarray) -> float:
    """Relative evaluation noise of a loss series: 1.4826 · MAD of the second differences of ln(L + ε) / √6."""
    if loss.size < 3:
        return 0.0
    y = np.log(np.maximum(loss, 0.0) + max(DROP_LOSS_FLOOR_REL * abs(float(loss[0])), EPS))
    d2 = np.diff(y, 2)
    return 1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / math.sqrt(6.0)


def _finite_or_nan(x: float) -> float:
    x = float(x)
    return x if math.isfinite(x) else math.nan


def _nanmax(a: np.ndarray) -> float:
    finite = np.isfinite(a)
    return float(a[finite].max()) if finite.any() else math.nan


def _gaussian_smooth(y: np.ndarray, sigma_pts: float) -> np.ndarray:
    """Gaussian smoothing with standard deviation ``sigma_pts`` grid points; edge values repeated beyond the ends."""
    radius = int(math.ceil(4.0 * sigma_pts))
    if radius < 1:
        return y.copy()
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_pts) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(y, radius, mode="edge"), kernel, mode="valid")


def _running_min(a: np.ndarray, half: int) -> np.ndarray:
    """Minimum over the window [i − half, i + half] clipped to the array."""
    if half < 1:
        return a.copy()
    padded = np.pad(a, half, mode="constant", constant_values=np.inf)
    return np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).min(axis=1)
