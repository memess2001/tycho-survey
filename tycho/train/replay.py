"""Re-observation by replay: rerun an object from scratch with denser RI spectroscopy.

Unlike the sky, a training run can be repeated: :func:`replay_object` retrains the object from its spec (bit-for-bit
the same trajectory on CPU with the same thread count, see :mod:`tycho.train.runner`) and takes RI checkpoints at
the usual log-spaced points plus ``n_dense`` linearly spaced points inside a window ``[lo, hi]``, typically the
bracket of the largest train-loss drop (``summary["largest_drop_lo"/"largest_drop_hi"]``). Because the RI probe at
step t depends only on (seed, t), the replayed curve agrees with the in-situ curve at the steps both observed.

:func:`replay_fidelity` checks that a replay reproduced the original light curve.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from tycho.train.runner import run_object
from tycho.train.schedule import eval_steps, subsample_log
from tycho.train.spec import ObjectSpec, RIConfig, RunResult

__all__ = ["FIDELITY_EPS", "dense_points", "replay_fidelity", "replay_object"]

logger = logging.getLogger(__name__)

#: Floor of the denominator of relative deviations in :func:`replay_fidelity`.
FIDELITY_EPS = 1e-12


def dense_points(lo: float, hi: float, n_dense: int, total: int) -> np.ndarray:
    """``n_dense`` linearly spaced integer steps in ``[lo, hi]`` clipped to ``[0, total]`` (sorted, unique)."""
    lo_i, hi_i = sorted((int(round(float(lo))), int(round(float(hi)))))
    lo_i, hi_i = max(0, lo_i), min(int(total), hi_i)
    if n_dense < 1 or lo_i > hi_i:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.round(np.linspace(lo_i, hi_i, int(n_dense))).astype(np.int64))


def replay_object(spec: ObjectSpec, *, work_dir: Path, ri: RIConfig, dense: tuple[int, int] | None = None,
                  n_dense: int = 30, stop_flag: Callable[[], bool] | None = None,
                  on_progress: Callable[[int, int], None] | None = None, resume: bool = True) -> RunResult:
    """Rerun ``spec`` from scratch with RI at ``subsample_log(eval steps, ri.n_points)`` ∪ ``n_dense`` points in
    ``dense = (lo, hi)`` (RI is enabled whatever ``ri.enabled`` says). The RI curve has ``meta["kind"] = "replay"``
    and ``meta["dense"] = [lo, hi]`` (or None). Interrupted replays resume like any run (``work_dir/resume.pt``)."""
    train = spec.train
    steps = eval_steps(train.steps, train.n_log_evals, train.n_lin_evals)
    points = subsample_log(steps, ri.n_points)
    window = None
    if dense is not None:
        extra = dense_points(dense[0], dense[1], n_dense, int(train.steps))
        if extra.size:
            window = [int(extra[0]), int(extra[-1])]
            points = np.union1d(points, extra)
        else:
            logger.warning("replay of %s: empty dense window %s", spec.object_id(), dense)
    return run_object(spec, work_dir=work_dir, ri=dataclasses.replace(ri, enabled=True), stop_flag=stop_flag,
                      on_progress=on_progress, resume=resume, ri_steps=[int(s) for s in points], ri_kind="replay",
                      ri_meta={"dense": window, "n_dense": int(n_dense) if window else 0})


def replay_fidelity(lc_a: Mapping[str, np.ndarray], lc_b: Mapping[str, np.ndarray],
                    keys: Sequence[str] = ("train_loss", "test_loss")) -> float:
    """Maximum relative deviation ``|a − b| / max(|a|, |b|, FIDELITY_EPS)`` between two light curves over their
    common steps and the given keys. 0 means a bit-for-bit replay. Values that are NaN in both count as equal,
    NaN in one only as ``inf``; ``inf`` is also returned when the curves share no step (nothing could be checked).

    Raises:
        ValueError: if a key is missing from either light curve.
    """
    for k in ("step", *keys):
        if k not in lc_a or k not in lc_b:
            raise ValueError(f"light curves must both contain {k!r}")
    steps_a, steps_b = np.asarray(lc_a["step"]), np.asarray(lc_b["step"])
    common, ia, ib = np.intersect1d(steps_a, steps_b, return_indices=True)
    if common.size == 0:
        return math.inf
    worst = 0.0
    for k in keys:
        a = np.asarray(lc_a[k], dtype=np.float64)[ia]
        b = np.asarray(lc_b[k], dtype=np.float64)[ib]
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        if np.any(nan_a != nan_b):
            return math.inf
        ok = ~nan_a
        if not ok.any():
            continue
        a, b = a[ok], b[ok]
        with np.errstate(invalid="ignore"):
            dev = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), FIDELITY_EPS)
        dev = np.where(a == b, 0.0, dev)  # equal infinities
        if not np.all(np.isfinite(dev)):
            return math.inf
        worst = max(worst, float(dev.max()))
    return worst
