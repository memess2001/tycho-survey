"""Survey workers (SPEC §6.4): claim objects and follow-ups from the catalog and observe them, one at a time.

A worker is one process (started by :mod:`tycho.survey.daemon` with the ``spawn`` context) running
:func:`worker_main`. Its loop:

1. Stop when the shared ``stop_event`` is set, on its own SIGTERM or SIGHUP, or when its supervisor is gone (an
   orphan must not keep claiming work). SIGINT is ignored: a Ctrl-C in the terminal reaches every process of the
   group, and the supervisor turns it into ``stop_event``.
2. **Disk guard**: while the survey root's file system has less than ``survey.min_free_gb`` free, claim nothing
   (heartbeat status ``paused: ...``) and re-check every :data:`DISK_RECHECK_S` seconds. The supervisor sends the
   alert.
3. **Follow-ups first** (``Catalog.claim_followup``; RI spectroscopy by replay, :mod:`tycho.survey.followup`), then
   objects (``Catalog.claim_next``), within an optional shared claim budget (``run_survey(max_objects=...)``).
   Idle workers poll every :data:`IDLE_POLL_S` seconds.
4. A per-iteration exception (e.g. the database stayed locked beyond its busy timeout) is logged and retried after
   an exponential backoff; it never ends the worker.

Observing an object (:func:`process_object`)
    ``run_object`` (device ``survey.device``, resume from ``data/work/<id>/resume.pt``; in-situ RI when the region's
    ``ri_policy`` is ``always``, with :func:`~tycho.survey.followup.resolve_ri_config`) → light curve
    ``data/lightcurves/<id[:2]>/<id>.npz`` (written atomically) → rule-based classification → RI curve, events and
    transients (``always`` regions; photometric transients such as grokking, loss spikes and the unknown class for
    every object) → ``Catalog.mark_done``. Marking done comes last, so a crash in between leaves the object running;
    it is requeued and observed again. The products are recorded idempotently (an existing rules classification,
    in-situ RI run or set of transients of the object is not recorded twice), so such a re-observation, or a
    worker that finishes an object after it was requeued, never duplicates catalog rows. A graceful stop gives
    status ``interrupted`` (the runner saved a resume checkpoint; ``mark_interrupted`` refunds the attempt), a
    training exception status ``failed`` with the traceback. An *infrastructure* error while recording (the
    database stayed locked beyond its busy timeout, a full or failing disk: ``sqlite3.OperationalError``,
    ``OSError``) is not the object's fault: it propagates, and the worker hands the object back
    (``Catalog.release_worker``), so it is retried under the ``survey.max_attempts`` rule instead of being lost.
    The in-situ RI settings are part of the resume checkpoint's fingerprint, so an object interrupted before a new
    calibration file appeared (which changes ``min_effect``) restarts from scratch when resumed (logged by the
    runner).

Heartbeats
    A thread writes ``Catalog.heartbeat(worker_id, object_id, status)`` every ``survey.heartbeat_s`` seconds; the
    status shows the job and its progress (``running 42%``). If the job reports no progress for :data:`STALL_S`
    seconds the thread stops writing heartbeats: the supervisor then requeues the job as stale and restarts the hung
    worker.

Threads and torch: ``configure_torch(survey.torch_threads)``; one thread per worker is the default (the survey's
objects are tiny and many single-threaded workers are fastest; exact replay needs a fixed thread count).

:func:`run_followups` is the foreground variant used by ``tycho followup run``: it requests follow-ups from the
scheduler if fewer than asked for are waiting, then claims and replays them one by one in the calling process
(with a heartbeat thread, so a running daemon does not mistake it for a lost worker).
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import logging.handlers
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

import tycho
from tycho.config import Paths, load_config
from tycho.survey.catalog import Catalog
from tycho.survey.followup import INFRASTRUCTURE_ERRORS, INSITU_KIND, process_followup, resolve_ri_config
from tycho.survey.sky import Region, Sky
from tycho.train.spec import ObjectSpec, RIConfig, RunResult

logger = logging.getLogger(__name__)

__all__ = ["DISK_RECHECK_S", "IDLE_POLL_S", "LOG_FORMAT", "STALL_S", "Heartbeat", "code_version", "free_disk_gb",
           "load_sky", "process_object", "region_of", "run_followups", "save_light_curve", "worker_main"]

IDLE_POLL_S = 5.0
"""Seconds between claim attempts while there is nothing to do."""

DISK_RECHECK_S = 60.0
"""Seconds between disk checks while claiming is paused for lack of space."""

STALL_S = 3600.0
"""A job that reports no progress for this long is considered hung: heartbeats are withheld so that the supervisor
requeues it and restarts the worker. The longest silent phase of a v0 object (one RI scan, or the training steps
between two checkpoints) takes seconds to minutes."""

ERROR_BACKOFF_S = 2.0
ERROR_BACKOFF_MAX_S = 300.0
"""Backoff after a failed loop iteration: ERROR_BACKOFF_S · 2^(k−1) for the k-th consecutive failure, capped."""

LOG_FORMAT = "%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s"
"""Format of the survey log lines (shared by the supervisor and its workers)."""

LOG_FILE = "survey.log"
"""The survey log in ``logs/`` (rotated by the supervisor; workers reopen it after a rotation)."""

_SLEEP_SLICE_S = 0.5


# ---------------------------------------------------------------------------------------------- helpers
def free_disk_gb(path: Path | str) -> float:
    """Free space (GB = 1e9 bytes) on the file system holding ``path``."""
    return shutil.disk_usage(path).free / 1e9


@functools.lru_cache(maxsize=None)
def _git_describe(repo: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", repo, "describe", "--always", "--dirty", "--tags"], capture_output=True,
                             text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return text if out.returncode == 0 and text else None


def code_version(cfg: Mapping[str, Any] | None = None) -> str:
    """``survey.code_version`` if configured, else ``<package version>+<git describe>`` when the package is a git
    checkout, else the package version."""
    configured = ((cfg or {}).get("survey") or {}).get("code_version")
    if configured:
        return str(configured)
    repo = Path(tycho.__file__).resolve().parent.parent
    if (repo / ".git").exists():
        described = _git_describe(str(repo))
        if described:
            return f"{tycho.__version__}+{described}"
    return tycho.__version__


def load_sky(paths: Paths, cfg: Mapping[str, Any]) -> Sky:
    """The sky named by ``survey.sky`` (relative to the survey root unless absolute)."""
    sky_path = Path(cfg["survey"]["sky"]).expanduser()
    return Sky.from_yaml(sky_path if sky_path.is_absolute() else paths.root / sky_path)


def region_of(sky: Sky | None, spec: ObjectSpec) -> Region | None:
    """The sky region an object belongs to (by its catalog region name and task), or None (ad-hoc objects,
    regions that are no longer in the sky)."""
    if sky is None:
        return None
    try:
        region = sky.region(spec.region)
    except KeyError:
        return None
    return region if region.task == spec.task else None


def save_light_curve(path: Path, lc: Mapping[str, np.ndarray]) -> Path:
    """Write a light curve with ``np.savez_compressed``, atomically (temporary file, fsync, rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{k: np.asarray(v) for k, v in lc.items()})
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _exists(catalog: Catalog, sql: str, params: tuple) -> bool:
    return bool(catalog.query(sql, params, max_rows=1))


# ---------------------------------------------------------------------------------------------- one object
def process_object(catalog: Catalog, paths: Paths, spec: ObjectSpec, cfg: Mapping[str, Any],
                   stop_flag: Callable[[], bool] | None, *, worker_id: str | None = None, sky: Sky | None = None,
                   on_progress: Callable[[int, int], None] | None = None, version: str | None = None) -> None:
    """Observe one claimed object and record it (see the module docstring).

    Args:
        catalog, paths: the survey's catalog and paths.
        spec: the object (its ``region`` is the catalog region).
        cfg: the survey config (``survey.device``, ``ri.*``, calibration file).
        stop_flag: polled by the runner; when it returns true the object is checkpointed and marked interrupted.
        worker_id: the claiming worker; status changes to failed / interrupted then apply only while the object is
            still assigned to it.
        sky: the sky (default: loaded from ``survey.sky``); decides the region's ``ri_policy``.
        on_progress: forwarded to ``run_object``.
        version: code version recorded with the object (default :func:`code_version`).
    """
    _process(catalog, paths, spec, cfg, stop_flag, worker_id=worker_id,
             sky=sky if sky is not None else load_sky(paths, cfg), on_progress=on_progress,
             version=version if version is not None else code_version(cfg))


def _process(catalog: Catalog, paths: Paths, spec: ObjectSpec, cfg: Mapping[str, Any],
             stop_flag: Callable[[], bool] | None, *, worker_id: str | None, sky: Sky | None,
             on_progress: Callable[[int, int], None] | None, version: str) -> str:
    """:func:`process_object`; returns the outcome "done", "failed" or "interrupted"."""
    from tycho.train.runner import run_object  # imports torch

    oid = spec.object_id()
    region = region_of(sky, spec)
    ri_cfg = resolve_ri_config(cfg, region, root=paths.root) if region is not None and region.ri_policy == "always" \
        else None
    device = str((cfg.get("survey") or {}).get("device") or "cpu")
    logger.info("observing %s (%s, region %s, seed %d, %d steps%s)", oid, spec.task, spec.region, spec.seed,
                spec.train.steps, "" if ri_cfg is None else f", in-situ RI min_effect {ri_cfg.min_effect:.3g}")
    result = run_object(dataclasses.replace(spec, device=device), work_dir=paths.work_dir(oid), ri=ri_cfg,
                        stop_flag=stop_flag, on_progress=on_progress, resume=True)
    if result.status == "interrupted":
        catalog.mark_interrupted(oid, worker_id=worker_id)
        logger.info("%s interrupted at %.0f s of wall time; resume checkpoint saved", oid, result.wall_time_s)
        return "interrupted"
    if result.status != "done":
        catalog.mark_failed(oid, result.error or f"run ended with status {result.status}", worker_id=worker_id)
        return "failed"
    try:
        _record(catalog, paths, result, ri_cfg, version=version)
    except INFRASTRUCTURE_ERRORS:
        raise  # not the object's fault: the worker hands it back (module docstring)
    except Exception:  # a recording bug must cost one object, not the worker
        error = traceback.format_exc()
        logger.error("recording %s failed:\n%s", oid, error)
        catalog.mark_failed(oid, f"recording the result failed:\n{error}", worker_id=worker_id)
        return "failed"
    try:
        paths.work_dir(oid).rmdir()  # empty once run_object removed its checkpoint
    except OSError:
        pass
    return "done"


def _record(catalog: Catalog, paths: Paths, result: RunResult, ri_cfg: RIConfig | None, *, version: str) -> None:
    """Light curve, classification, in-situ RI, transients, then mark_done (idempotent; module docstring)."""
    from tycho.analysis.classify import classify
    from tycho.analysis.events import detect_events, find_transients, transient_min_peak

    oid = result.object_id
    lc_path = save_light_curve(paths.lightcurve_path(oid), result.light_curve)
    summary = result.summary
    cls = classify(summary)
    if not _exists(catalog, "SELECT 1 FROM classifications WHERE object_id = ? AND classifier = ?",
                   (oid, cls.ruleset)):
        catalog.add_classification(oid, cls, classifier=cls.ruleset)
    events: list = []
    event_ids: list[int] = []
    curve = result.ri_curve
    if ri_cfg is not None and curve is not None:
        if _exists(catalog, "SELECT 1 FROM ri_runs WHERE object_id = ? AND kind = ?", (oid, INSITU_KIND)):
            logger.warning("%s: in-situ RI already recorded; not recording a duplicate", oid)
        else:
            ri_path = curve.save(paths.ri_path(oid, INSITU_KIND))
            ri_run_id = catalog.add_ri_run(oid, INSITU_KIND, ri_path, curve)
            events = detect_events(curve, alpha=ri_cfg.alpha)
            event_ids = catalog.add_events(ri_run_id, oid, events)
    if not _exists(catalog, "SELECT 1 FROM transients WHERE object_id = ?", (oid,)):
        catalog.add_transients(oid, find_transients(events, summary, cls, min_peak=transient_min_peak(curve)),
                               event_ids)
    catalog.mark_done(result, lc_path, code_version=version)
    logger.info("%s done in %.1f s: %s (score %.2f)%s", oid, result.wall_time_s, cls.label, cls.score,
                f", {len(events)} RI events" if curve is not None else "")


# ---------------------------------------------------------------------------------------------- heartbeats
class Heartbeat:
    """Background thread writing ``worker_id``'s heartbeat row every ``interval_s`` seconds (see the module
    docstring). :meth:`set` names the current job (None when idle) and its status, :meth:`progress` updates the
    status and the progress clock; while a job reports no progress for ``stall_s`` seconds no heartbeat is written.
    A failed write (e.g. a locked database) is retried at the next beat."""

    def __init__(self, catalog: Catalog, worker_id: str, interval_s: float, stall_s: float = STALL_S) -> None:
        self.catalog = catalog
        self.worker_id = worker_id
        self.interval_s = max(0.1, float(interval_s))
        self.stall_s = float(stall_s)
        self._lock = threading.Lock()
        self._object: str | None = None
        self._status = "starting"
        self._last_progress = time.monotonic()
        self._halt = threading.Event()
        self._stalled = False
        self._failures = 0
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{worker_id}", daemon=True)

    def set(self, object_id: str | None, status: str) -> None:
        with self._lock:
            self._object, self._status = object_id, status
            self._last_progress = time.monotonic()

    def progress(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._last_progress = time.monotonic()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)

    def _run(self) -> None:
        while not self._halt.wait(self.interval_s):
            with self._lock:
                oid, status, silent = self._object, self._status, time.monotonic() - self._last_progress
            if oid is not None and silent > self.stall_s:
                if not self._stalled:
                    logger.error("worker %s: no progress on %s for %.0f s; withholding heartbeats so that the "
                                 "supervisor requeues it and restarts this worker", self.worker_id, oid, silent)
                    self._stalled = True
                continue
            self._stalled = False
            try:
                self.catalog.heartbeat(self.worker_id, oid, status)
                self._failures = 0
            except Exception as exc:  # e.g. a locked database: try again at the next beat
                self._failures += 1
                if self._failures in (1, 10, 100) or self._failures % 1000 == 0:
                    logger.warning("worker %s: heartbeat failed (%d in a row): %s", self.worker_id, self._failures,
                                   exc)


# ---------------------------------------------------------------------------------------------- the worker loop
def worker_main(worker_id: str, root: Path | str, stop_event: Any, *, claim_budget: Any = None,
                parent_pid: int | None = None, configure_logging: bool = False) -> None:
    """Run a worker until ``stop_event`` is set (see the module docstring).

    Args:
        worker_id: stable id (e.g. "w0"); a restarted worker first takes back the jobs its previous incarnation
            left running (``Catalog.release_worker``).
        root: the survey root.
        stop_event: anything with ``is_set()`` (a ``multiprocessing`` or ``threading`` Event).
        claim_budget: optional shared integer (``multiprocessing.Value``) of objects that may still be claimed;
            negative = unlimited. A claim that finds no object, or whose object is interrupted or handed back, is
            refunded.
        parent_pid: the supervisor's pid; the worker stops when its parent changes (orphaned).
        configure_logging: log to ``logs/survey.log`` (set by the supervisor for its spawned workers).
    """
    paths = Paths(root)
    local_stop = threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), lambda signum, frame: local_stop.set())
    if configure_logging:
        _configure_logging(paths)
    from tycho.train.device import configure_torch  # imports torch

    cfg = load_config(paths.root)
    configure_torch(int(cfg["survey"].get("torch_threads") or 1))
    _Worker(worker_id, paths, cfg, stop_event, local_stop, claim_budget, parent_pid).run()


def _configure_logging(paths: Paths) -> None:
    """Log to the shared survey log; WatchedFileHandler reopens it after the supervisor rotates it."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.WatchedFileHandler(paths.logs / LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger = logging.getLogger()
    root_logger.handlers[:] = [handler]
    root_logger.setLevel(logging.INFO)
    logging.captureWarnings(True)


class _Worker:
    """State of one worker process (see :func:`worker_main`)."""

    def __init__(self, worker_id: str, paths: Paths, cfg: Mapping[str, Any], stop_event: Any,
                 local_stop: threading.Event, claim_budget: Any, parent_pid: int | None) -> None:
        survey = cfg["survey"]
        self.worker_id = worker_id
        self.paths = paths
        self.cfg = cfg
        self.stop_event = stop_event
        self.local_stop = local_stop
        self.claim_budget = claim_budget
        self.parent_pid = parent_pid
        self.min_free_gb = float(survey["min_free_gb"])
        self.catalog = Catalog(paths.db, root=paths.root, max_attempts=int(survey["max_attempts"]))
        self.sky = load_sky(paths, cfg)
        self.version = code_version(cfg)
        self.heartbeat = Heartbeat(self.catalog, worker_id, float(survey["heartbeat_s"]))
        self._paused = False
        self._job = "running"
        self._needs_release = False

    # ------------------------------------------------------------------ control
    def stopping(self) -> bool:
        if self.stop_event.is_set() or self.local_stop.is_set():
            return True
        if self.parent_pid is not None and os.getppid() != self.parent_pid:
            if not self.local_stop.is_set():
                logger.error("worker %s: supervisor (pid %d) is gone; stopping", self.worker_id, self.parent_pid)
                self.local_stop.set()
            return True
        return False

    def sleep(self, seconds: float) -> None:
        """Sleep up to ``seconds``, waking early when the worker should stop."""
        deadline = time.monotonic() + seconds
        while not self.stopping():
            left = deadline - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(_SLEEP_SLICE_S, left))

    def _take_claim(self) -> bool:
        if self.claim_budget is None:
            return True
        with self.claim_budget.get_lock():
            if self.claim_budget.value == 0:
                return False
            if self.claim_budget.value > 0:
                self.claim_budget.value -= 1
            return True

    def _refund_claim(self) -> None:
        if self.claim_budget is not None:
            with self.claim_budget.get_lock():
                if self.claim_budget.value >= 0:
                    self.claim_budget.value += 1

    def _on_progress(self, step: int, total: int) -> None:
        self.heartbeat.progress(f"{self._job} {100.0 * step / max(total, 1):.0f}%")

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        released = self.catalog.release_worker(self.worker_id)
        if released:
            logger.warning("worker %s: took back %d job(s) left running by its previous incarnation", self.worker_id,
                           released)
        self.heartbeat.set(None, "idle")
        self.heartbeat.start()
        logger.info("worker %s started (pid %d)", self.worker_id, os.getpid())
        failures = 0
        try:
            while not self.stopping():
                try:
                    self.step()
                    failures = 0
                except Exception:
                    failures += 1
                    delay = min(ERROR_BACKOFF_MAX_S, ERROR_BACKOFF_S * 2 ** (failures - 1))
                    logger.exception("worker %s: loop error (%d in a row); retrying in %.0f s", self.worker_id,
                                     failures, delay)
                    self.heartbeat.set(None, "error")
                    self.sleep(delay)
        finally:
            self.heartbeat.stop()
            try:
                self.catalog.heartbeat(self.worker_id, None, "stopped")
            except Exception:
                logger.exception("worker %s: could not write the final heartbeat", self.worker_id)
            self.catalog.close()
            logger.info("worker %s stopped", self.worker_id)

    def step(self) -> None:
        """Claim and process one job, or wait (disk guard, nothing to do, claim budget used up)."""
        if self._needs_release:  # a job ended with an exception: hand back whatever is still running under our id
            released = self.catalog.release_worker(self.worker_id)
            self._needs_release = False
            if released:
                logger.warning("worker %s: handed back %d job(s) after an error", self.worker_id, released)
        free = free_disk_gb(self.paths.root)
        if free < self.min_free_gb:
            if not self._paused:
                logger.warning("worker %s: %.1f GB free < survey.min_free_gb = %g; claiming paused", self.worker_id,
                               free, self.min_free_gb)
                self._paused = True
            self.heartbeat.set(None, f"paused: {free:.1f} GB free")
            self.sleep(DISK_RECHECK_S)
            return
        if self._paused:
            logger.info("worker %s: %.1f GB free; claiming resumed", self.worker_id, free)
            self._paused = False
        oid = self.catalog.claim_followup(self.worker_id)
        if oid is not None:
            self._followup(oid)
            return
        if not self._take_claim():
            self.heartbeat.set(None, "idle: claim budget used")
            self.sleep(IDLE_POLL_S)
            return
        spec = self.catalog.claim_next(self.worker_id)
        if spec is None:
            self._refund_claim()
            self.heartbeat.set(None, "idle")
            self.sleep(IDLE_POLL_S)
            return
        self._object(spec)

    def _object(self, spec: ObjectSpec) -> None:
        self._job = "running"
        self.heartbeat.set(spec.object_id(), "running")
        try:
            outcome = _process(self.catalog, self.paths, spec, self.cfg, self.stopping, worker_id=self.worker_id,
                               sky=self.sky, on_progress=self._on_progress, version=self.version)
        except Exception:  # not a run failure (those are recorded) but e.g. the catalog: requeue the object
            logger.exception("worker %s: observing %s raised; handing it back", self.worker_id, spec.object_id())
            self._needs_release = True
            outcome = "error"
        finally:
            self.heartbeat.set(None, "idle")
        if outcome not in ("done", "failed"):  # interrupted or handed back: it will be claimed again
            self._refund_claim()

    def _followup(self, object_id: str) -> None:
        self._job = "followup"
        self.heartbeat.set(object_id, "followup")
        try:
            spec = self.catalog.get_spec(object_id)
            ri_cfg = resolve_ri_config(self.cfg, region_of(self.sky, spec), root=self.paths.root)
            process_followup(self.catalog, self.paths, object_id, ri_cfg, worker_id=self.worker_id,
                             stop_flag=self.stopping, on_progress=self._on_progress)
        except Exception:
            logger.exception("worker %s: follow-up of %s raised; handing it back", self.worker_id, object_id)
            self._needs_release = True
        finally:
            self.heartbeat.set(None, "idle")


# ---------------------------------------------------------------------------------------------- foreground follow-ups
def run_followups(paths: Paths, cfg: Mapping[str, Any], n: int, *, stop_flag: Callable[[], bool] | None = None,
                  worker_id: str | None = None) -> list[tuple[str, str]]:
    """Run up to ``n`` follow-ups in this process (``tycho followup run``) and return ``(object_id, outcome)`` pairs.

    If fewer than ``n`` follow-ups are waiting, ``Scheduler.select_followups`` is asked for the difference first (so
    the daily budget and the priority order apply). Each follow-up is then claimed (``Catalog.claim_followup``) and
    replayed with :func:`~tycho.survey.followup.process_followup` under ``worker_id`` (default
    ``cli-<host>-<pid>``), while a :class:`Heartbeat` thread keeps the claim fresh. ``stop_flag`` interrupts the
    current replay (it is handed back and resumes later) and ends the loop.
    """
    from tycho.survey.scheduler import Scheduler, SchedulerPolicy
    from tycho.train.device import configure_torch  # imports torch

    survey = cfg["survey"]
    worker_id = worker_id or f"cli-{socket.gethostname()}-{os.getpid()}"
    stop_flag = stop_flag if stop_flag is not None else (lambda: False)
    sky = load_sky(paths, cfg)
    outcomes: list[tuple[str, str]] = []
    with Catalog(paths.db, root=paths.root, max_attempts=int(survey["max_attempts"])) as catalog:
        waiting = catalog.query("SELECT COUNT(*) AS n FROM followups WHERE status = 'requested'")[0]["n"]
        if waiting < n:
            policy = SchedulerPolicy.from_dict(cfg.get("scheduler"))
            Scheduler(catalog, sky, policy, device=str(survey.get("device") or "cpu")).select_followups(n - waiting)
        configure_torch(int(survey.get("torch_threads") or 1))
        heartbeat = Heartbeat(catalog, worker_id, float(survey["heartbeat_s"]))
        heartbeat.set(None, "idle")
        heartbeat.start()
        try:
            while len(outcomes) < n and not stop_flag():
                oid = catalog.claim_followup(worker_id)
                if oid is None:
                    break
                heartbeat.set(oid, "followup")
                ri_cfg = resolve_ri_config(cfg, region_of(sky, catalog.get_spec(oid)), root=paths.root)
                outcome = process_followup(
                    catalog, paths, oid, ri_cfg, worker_id=worker_id, stop_flag=stop_flag,
                    on_progress=lambda step, total: heartbeat.progress(f"followup {100.0 * step / max(total, 1):.0f}%"))
                heartbeat.set(None, "idle")
                outcomes.append((oid, outcome))
        finally:
            heartbeat.stop()
            catalog.heartbeat(worker_id, None, "stopped")
    return outcomes
