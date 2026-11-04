"""Observation and learning-rate schedules.

Evaluations ("photometry") are log-spaced in the step, because training phenomena (plateaus, drops, power laws)
are scale-free, plus a linear grid that keeps late training resolved. RI checkpoints ("spectroscopy") are an
approximately log-uniform subset of the evaluation steps.

Step convention: step t means "after t optimizer updates"; step 0 is the initial model.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from tycho.train.spec import TrainConfig

__all__ = ["LR_SCHEDULES", "eval_steps", "lr_at", "subsample_log"]

#: Learning-rate schedules after the linear warm-up.
LR_SCHEDULES = ("constant", "cosine")


def eval_steps(total: int, n_log: int, n_lin: int) -> np.ndarray:
    """Evaluation steps: sorted unique ints in ``[0, total]`` including 0 and ``total``.

    The union of ``n_log`` log-spaced points from 1 to ``total`` and ``n_lin`` linearly spaced points from 0 to
    ``total`` (each rounded to the nearest int, so duplicates at small steps collapse).
    """
    total, n_log, n_lin = int(total), int(n_log), int(n_lin)
    if total < 0 or n_log < 0 or n_lin < 0:
        raise ValueError(f"total, n_log and n_lin must be >= 0, got {total}, {n_log}, {n_lin}")
    parts = [np.asarray([0, total], dtype=np.int64)]
    if total >= 1 and n_log:
        parts.append(np.round(np.logspace(0.0, math.log10(total), n_log)).astype(np.int64))
    if n_lin:
        parts.append(np.round(np.linspace(0.0, total, n_lin)).astype(np.int64))
    return np.unique(np.concatenate(parts))


def subsample_log(steps: Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    """An approximately log-uniform subset of ``n`` of the given steps that keeps the first and the last.

    Targets are ``n`` points equally spaced in ``ln(1 + step)`` between the first and the last step; each target
    in turn takes the nearest step not yet taken (ties: the smaller step). Returns all steps when
    ``n >= len(steps)``. Output: sorted unique int64.
    """
    s = np.unique(np.asarray(steps, dtype=np.int64))
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if s.size and s[0] < 0:
        raise ValueError("steps must be >= 0")
    if n >= s.size:
        return s
    if n == 0:
        return s[:0]
    if n == 1:
        return s[-1:]
    x = np.log1p(s.astype(np.float64))
    free = np.ones(s.size, dtype=bool)
    free[0] = free[-1] = False
    chosen = [0, s.size - 1]
    for target in np.linspace(x[0], x[-1], n)[1:-1]:
        cand = np.flatnonzero(free)
        i = int(cand[np.argmin(np.abs(x[cand] - target))])
        free[i] = False
        chosen.append(i)
    return s[np.sort(chosen)]


def lr_at(step: int, train: TrainConfig) -> float:
    """Learning rate of the update from step ``step`` to ``step + 1``.

    Linear warm-up ``lr · (step + 1) / warmup_steps`` for ``step < warmup_steps``, then ``"constant"`` (``lr``) or
    ``"cosine"`` (``lr · ½ (1 + cos(π · min(1, (step − warmup) / max(1, steps − warmup))))``, which reaches 0 at
    the last step).

    Raises:
        ValueError: on an unknown schedule.
    """
    if train.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {train.lr_schedule!r}")
    base = float(train.lr)
    warm = int(train.warmup_steps)
    if warm > 0 and step < warm:
        return base * (step + 1) / warm
    if train.lr_schedule == "constant":
        return base
    progress = min(1.0, (step - warm) / max(1, int(train.steps) - warm))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))
