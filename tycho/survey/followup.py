"""Spectroscopic follow-up by replay (SPEC §6.4) and the RI settings of the survey.

Two-tier observation: every object gets photometry; selected objects get RI spectroscopy. Regions with
``ri_policy = always`` (the positive controls) are observed with RI *in situ* by the worker. All other selected objects
are *followed up*: the scheduler (or the agent) requests a follow-up, a worker claims it and :func:`run_followup`
replays the object from its spec. On CPU the replay is bit-for-bit the original trajectory (see
:mod:`tycho.train.replay`), so the RI curve describes the object already in the catalog, and its RI checkpoints can be
placed where they matter: ``ri.n_points`` log-spaced points plus :data:`N_DENSE` points densely covering the bracket of
the largest train-loss drop (``summary["largest_drop_lo"/"largest_drop_hi"]``). Objects without a sharp drop get the
log-spaced points only.

After the replay, :func:`run_followup` checks that the replayed light curve reproduces the stored one
(:func:`tycho.train.replay.replay_fidelity`, stored in the curve's ``meta["replay_fidelity"]``; a deviation above
:data:`REPLAY_FIDELITY_TOL` is logged as a warning, since the RI curve then belongs to a different trajectory),
saves the curve (``data/ri/<id[:2]>/<id>_followup.npz``), records an ``ri_runs`` row of kind :data:`FOLLOWUP_KIND`,
detects activation / extinction / reactivation events and records the event-based transients. Photometric transients
(loss spikes, grokking, unknown class) were recorded when the object was first observed and are not repeated.

RI settings (:func:`resolve_ri_config`)
    The :class:`~tycho.train.spec.RIConfig` of an object is the config's ``ri`` section overridden by its region's
    ``ri`` mapping in the sky file. ``min_effect`` (the effect-size threshold of a detection, nats) follows this
    precedence:

    1. the region's ``ri.min_effect``;
    2. ``ri.min_effect`` set explicitly in ``tycho.yaml`` (or a value differing from the built-in default);
    3. the calibrated value: :func:`tycho.calibration.report.recommended_min_effect` of the calibration file
       (``ri.calibration_file``, default ``reports/calibration/calibration.json``) at the probe size ``n_probe``,
       taking the maximum over orders as ``ri_scan`` applies one threshold per scan;
    4. the built-in default (``DEFAULT_CONFIG["ri"]["min_effect"]``).

    So once ``tycho calibrate`` has measured the instrument's noise floor, detections are judged against it unless
    a threshold was chosen on purpose.

Follow-up status (:func:`process_followup`)
    ``requested → running`` (``Catalog.claim_followup``) ``→ done`` (recorded) ``| failed`` (an exception; the
    traceback is stored) or back to ``requested`` when the replay was interrupted by a graceful stop. The replay's
    resume checkpoint (``data/work/<id>/followup/resume.pt``) is kept, so the next claim continues where it stopped.
    The catalog has no attempt refund for follow-ups (unlike ``mark_interrupted`` for objects), so an interrupted
    follow-up is handed back with ``Catalog.release_worker`` and an unlimited attempt budget: the interruption counts
    as an attempt in ``followups.attempts``, but never makes the follow-up fail. Infrastructure errors
    (:data:`INFRASTRUCTURE_ERRORS`: a database locked beyond its busy timeout, a full or failing disk) are not the
    object's fault and would otherwise fail it for good (an object is never followed up twice), so they propagate:
    the worker hands the follow-up back and it is retried under the ``survey.max_attempts`` rule. A successful
    follow-up removes its (then empty) work directory.
"""
from __future__ import annotations

import functools
import logging
import math
import sqlite3
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from tycho.config import DEFAULT_CONFIG, Paths
from tycho.survey.catalog import Catalog
from tycho.survey.sky import Region
from tycho.train.spec import ObjectSpec, RIConfig

logger = logging.getLogger(__name__)

__all__ = ["FOLLOWUP_KIND", "INFRASTRUCTURE_ERRORS", "INSITU_KIND", "N_DENSE", "REPLAY_FIDELITY_TOL", "FollowupFailed",
           "FollowupInterrupted", "calibrated_min_effect", "dense_window", "process_followup", "resolve_ri_config",
           "run_followup"]

FOLLOWUP_KIND = "followup"
"""``ri_runs.kind`` (and RI file tag) of follow-up curves."""

INSITU_KIND = "insitu"
"""``ri_runs.kind`` (and RI file tag) of curves observed in situ (``ri_policy = always``)."""

N_DENSE = 30
"""Dense RI checkpoints placed linearly inside the bracket of the largest train-loss drop."""

REPLAY_FIDELITY_TOL = 1e-6
"""Largest relative deviation of the replayed train/test loss from the stored light curve that is accepted silently.
On CPU with the survey's fixed thread count the replay is exact (deviation 0); anything above this means the RI curve
observed a different trajectory (other thread count, MPS, or changed code), which is logged as a warning."""

WORK_SUBDIR = "followup"
"""Subdirectory of the object's work directory that holds the replay's resume checkpoint."""

INFRASTRUCTURE_ERRORS: tuple[type[BaseException], ...] = (sqlite3.OperationalError, OSError)
"""Errors that are not the observed object's fault (locked database, full disk): the job is handed back and retried
instead of being marked failed (also used by :mod:`tycho.survey.worker`)."""

_UNLIMITED_ATTEMPTS = 2 ** 31 - 1
"""Attempt budget used when handing back an interrupted follow-up (an interruption must never fail it)."""

_RI_FIELDS = frozenset(f.name for f in fields(RIConfig))


class FollowupInterrupted(RuntimeError):
    """The replay was stopped by ``stop_flag``; its resume checkpoint is kept for the next attempt."""


class FollowupFailed(RuntimeError):
    """The replay failed (the message holds the runner's error or traceback)."""


# ---------------------------------------------------------------------------------------------- RI settings
def resolve_ri_config(cfg: Mapping[str, Any], region: Region | None = None, *, root: Path | str | None = None,
                      enabled: bool = True) -> RIConfig:
    """The RIConfig of an object in ``region`` (None: no region overrides); see the module docstring for the
    ``min_effect`` precedence. ``root`` locates ``tycho.yaml`` and a relative calibration file."""
    ri_section = dict(cfg.get("ri") or {})
    settings = {k: v for k, v in ri_section.items() if k in _RI_FIELDS}
    region_ri = dict(region.ri) if region is not None else {}
    settings.update(region_ri)
    explicit = "min_effect" in region_ri or _explicit_min_effect(ri_section, root)
    if not explicit:
        n_probe = int(settings.get("n_probe", RIConfig.n_probe))
        calibrated = calibrated_min_effect(cfg, root, n_probe)
        if calibrated is not None:
            settings["min_effect"] = calibrated
    settings["enabled"] = bool(enabled)
    return RIConfig.from_dict(settings)


def calibrated_min_effect(cfg: Mapping[str, Any], root: Path | str | None, n_probe: int) -> float | None:
    """The calibrated min_effect (nats) at ``n_probe`` samples, or None without a usable calibration file."""
    rel = (cfg.get("ri") or {}).get("calibration_file")
    if not rel:
        return None
    path = Path(rel).expanduser()
    if not path.is_absolute():
        if root is None:
            return None
        path = Path(root) / path
    try:
        stat = path.stat()
    except OSError:
        return None
    return _calibrated_cached(str(path), stat.st_mtime_ns, stat.st_size, int(n_probe))


@functools.lru_cache(maxsize=32)
def _calibrated_cached(path: str, mtime_ns: int, size: int, n_probe: int) -> float | None:
    """Read the calibration file once per (file version, probe size); a broken file is reported, not fatal."""
    from tycho.calibration.report import recommended_min_effect  # matplotlib-free; imported lazily

    try:
        value = float(recommended_min_effect(path, n_probe))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("calibration file %s is unusable (%s: %s); using the configured min_effect", path,
                       type(exc).__name__, exc)
        return None
    if not (math.isfinite(value) and value >= 0):
        logger.warning("calibration file %s gives min_effect %r at n=%d; ignored", path, value, n_probe)
        return None
    logger.info("min_effect %.3g nats at n_probe=%d from calibration %s", value, n_probe, path)
    return value


def _explicit_min_effect(ri_section: Mapping[str, Any], root: Path | str | None) -> bool:
    """Whether ``ri.min_effect`` was chosen on purpose: written in ``<root>/tycho.yaml``, or a value that differs
    from the built-in default (e.g. a config dict built in code)."""
    value = ri_section.get("min_effect", DEFAULT_CONFIG["ri"]["min_effect"])
    if value != DEFAULT_CONFIG["ri"]["min_effect"]:
        return True
    if root is None:
        return False
    path = Path(root) / "tycho.yaml"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return False
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("cannot read %s (%s); treating ri.min_effect as not set", path, exc)
        return False
    ri_user = user.get("ri") if isinstance(user, Mapping) else None
    return isinstance(ri_user, Mapping) and "min_effect" in ri_user


# ---------------------------------------------------------------------------------------------- follow-up
def dense_window(summary: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """The bracket ``(lo, hi)`` of evaluated steps around the largest sharp train-loss drop, or None when the
    summary has no drop (missing, None or NaN entries, or an empty bracket)."""
    if not summary:
        return None
    lo, hi = _finite(summary.get("largest_drop_lo")), _finite(summary.get("largest_drop_hi"))
    if lo is None or hi is None or hi < lo:
        return None
    return int(round(lo)), int(round(hi))


def run_followup(catalog: Catalog, paths: Paths, object_id: str, ri_cfg: RIConfig, *,
                 stop_flag: Callable[[], bool] | None = None,
                 on_progress: Callable[[int, int], None] | None = None, n_dense: int = N_DENSE) -> int:
    """Replay a finished object with dense RI around its largest train-loss drop and record the result.

    Saves the RI curve, records an ``ri_runs`` row (kind :data:`FOLLOWUP_KIND`), its events and their transients
    (module docstring). Idempotent: if the object already has a follow-up RI run (checked before the replay, and
    again before recording in case another worker finished the same follow-up meanwhile), nothing new is recorded
    and the existing run id is returned.

    Returns:
        The ``ri_run_id``.

    Raises:
        KeyError: unknown object. ValueError: the object is not done.
        FollowupInterrupted: ``stop_flag`` stopped the replay (resume checkpoint kept).
        FollowupFailed: the replay failed or produced no RI curve.
    """
    from tycho.analysis.events import detect_events, find_transients, transient_min_peak  # numpy only
    from tycho.train.replay import replay_fidelity, replay_object  # imports torch

    obj = catalog.get_object(object_id)
    if obj["status"] != "done":
        raise ValueError(f"object {object_id} is {obj['status']}, not done: nothing to follow up")
    existing = _followup_run_id(catalog, object_id)
    if existing is not None:
        logger.warning("follow-up of %s: already recorded as ri_run %d; nothing to do", object_id, existing)
        return existing
    spec = ObjectSpec.from_dict(obj["spec"])
    summary = obj["summary"] or {}
    window = dense_window(summary)
    work_dir = paths.work_dir(object_id) / WORK_SUBDIR
    logger.info("follow-up of %s (%s, seed %d): dense RI window %s", object_id, spec.task, spec.seed,
                "none (log-spaced points only)" if window is None else list(window))
    result = replay_object(spec, work_dir=work_dir, ri=ri_cfg, dense=window, n_dense=n_dense, stop_flag=stop_flag,
                           on_progress=on_progress, resume=True)
    if result.status == "interrupted":
        raise FollowupInterrupted(f"follow-up of {object_id} interrupted; resume checkpoint in {work_dir}")
    if result.status != "done":
        raise FollowupFailed(result.error or f"replay of {object_id} ended with status {result.status}")
    curve = result.ri_curve
    if curve is None:
        raise FollowupFailed(f"replay of {object_id} produced no RI curve")

    fidelity = _replay_fidelity(catalog, obj, result.light_curve, replay_fidelity)
    curve.meta["replay_fidelity"] = fidelity
    curve.meta["followup_reason"] = (obj.get("followup") or {}).get("reason")
    if fidelity is None:
        logger.warning("follow-up of %s: stored light curve unavailable, replay fidelity not checked", object_id)
    elif not fidelity <= REPLAY_FIDELITY_TOL:
        logger.warning("follow-up of %s: the replay deviates from the stored light curve (max relative deviation "
                       "%.3g > %.0e); the RI curve describes a different trajectory (thread count, device or code "
                       "changed?)", object_id, fidelity, REPLAY_FIDELITY_TOL)

    existing = _followup_run_id(catalog, object_id)
    if existing is not None:  # e.g. a second worker replayed an object whose first worker was declared lost
        logger.warning("follow-up of %s: already recorded as ri_run %d; not recording a duplicate", object_id,
                       existing)
        _remove_empty_work_dirs(work_dir)
        return existing
    path = curve.save(paths.ri_path(object_id, FOLLOWUP_KIND))
    ri_run_id = catalog.add_ri_run(object_id, FOLLOWUP_KIND, path, curve)
    events = detect_events(curve, alpha=ri_cfg.alpha)
    event_ids = catalog.add_events(ri_run_id, object_id, events)
    classification = _classification_mapping(obj.get("classification"))
    transients = [t for t in find_transients(events, summary, classification, min_peak=transient_min_peak(curve))
                  if t.event_index is not None]
    catalog.add_transients(object_id, transients, event_ids)
    logger.info("follow-up of %s recorded: ri_run %d, %d subsets x %d checkpoints, %d events, %d transients",
                object_id, ri_run_id, curve.S, curve.T, len(events), len(transients))
    _remove_empty_work_dirs(work_dir)
    return ri_run_id


def process_followup(catalog: Catalog, paths: Paths, object_id: str, ri_cfg: RIConfig, *, worker_id: str,
                     stop_flag: Callable[[], bool] | None = None,
                     on_progress: Callable[[int, int], None] | None = None) -> str:
    """Run a follow-up claimed by ``worker_id`` and set its status; never raises for a per-object problem.

    Returns:
        "done", "failed" or "interrupted" (handed back to ``requested``, see the module docstring).
    """
    try:
        run_followup(catalog, paths, object_id, ri_cfg, stop_flag=stop_flag, on_progress=on_progress)
    except FollowupInterrupted:
        catalog.release_worker(worker_id, max_attempts=_UNLIMITED_ATTEMPTS)
        logger.info("follow-up of %s interrupted; handed back for a later resume", object_id)
        return "interrupted"
    except INFRASTRUCTURE_ERRORS:
        raise  # not the object's fault: the caller hands the follow-up back (module docstring)
    except Exception as exc:  # a per-object failure must never stop the worker
        error = exc.args[0] if isinstance(exc, FollowupFailed) and exc.args else traceback.format_exc()
        logger.error("follow-up of %s failed: %s", object_id, error)
        catalog.finish_followup(object_id, ok=False, error=str(error))
        return "failed"
    catalog.finish_followup(object_id, ok=True)
    return "done"


# ---------------------------------------------------------------------------------------------- helpers
def _followup_run_id(catalog: Catalog, object_id: str) -> int | None:
    rows = catalog.query("SELECT ri_run_id FROM ri_runs WHERE object_id = ? AND kind = ? ORDER BY ri_run_id LIMIT 1",
                         (object_id, FOLLOWUP_KIND))
    return int(rows[0]["ri_run_id"]) if rows else None


def _remove_empty_work_dirs(work_dir: Path) -> None:
    """Remove the replay's work directory and the object's, if empty (the runner deletes its checkpoint when done)."""
    for d in (work_dir, work_dir.parent):
        try:
            d.rmdir()
        except OSError:
            return


def _finite(v: Any) -> float | None:
    if v is None or isinstance(v, (str, bytes, bool)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _replay_fidelity(catalog: Catalog, obj: Mapping[str, Any], replayed: Mapping[str, np.ndarray],
                     fidelity_fn: Callable[..., float]) -> float | None:
    """Max relative deviation between the stored and the replayed light curve, or None if it cannot be checked."""
    lc_row = obj.get("lightcurve")
    if not lc_row:
        return None
    try:
        with np.load(catalog.resolve_path(lc_row["path"]), allow_pickle=False) as z:
            stored = {k: z[k] for k in z.files}
        return float(fidelity_fn(stored, replayed))
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("cannot compare the replay of %s with its stored light curve: %s", obj.get("object_id"), exc)
        return None


def _classification_mapping(latest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The catalog's latest classification as the mapping ``find_transients`` reads (label and reasons; the
    catalog stores the reasons one per line in the note)."""
    if not latest:
        return None
    return {"label": latest.get("label"), "reasons": [r for r in str(latest.get("note") or "").splitlines() if r]}
