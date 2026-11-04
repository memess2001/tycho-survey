"""The survey daemon (SPEC §6.4): one supervisor process that keeps N workers busy, unattended, for months.

:func:`run_survey` takes the survey lock (``data/daemon.lock``, an ``flock`` held for the daemon's lifetime, so two
daemons can never run on one root), configures the rotating survey log (``logs/survey.log``, shared with the
workers), starts N workers (:func:`tycho.survey.worker.worker_main`) with the ``spawn`` context and runs
:meth:`Supervisor.step` every :data:`LOOP_INTERVAL_S` seconds until a stop is requested. One step:

* **Workers**: a worker that exited is restarted (it first takes back whatever it left running). A slot that keeps
  dying is restarted with exponential backoff (:data:`RESTART_BACKOFF_S` doubling up to
  :data:`RESTART_BACKOFF_MAX_S`), and :data:`CRASH_ALERT_RESTARTS` restarts within an hour raise an alert. A live
  worker whose heartbeat is older than ``survey.stale_after_s`` (it is hung; see ``worker.STALL_S``) is killed and
  restarted.
* **Stale jobs** (every ``survey.heartbeat_s``): ``Catalog.requeue_stale(survey.stale_after_s)`` requeues running
  objects and follow-ups of lost workers (failed after ``survey.max_attempts`` claims). After the machine slept (the
  supervisor's step comes more than :data:`SLEEP_GAP_S` late) every heartbeat looks stale, so requeueing waits two
  heartbeat intervals for the workers to report again.
* **Scheduling** (every ``survey.topup_every_s``): ``Scheduler.top_up`` (coarse coverage, then boundary
  refinement).
* **Follow-ups** (every ``survey.followup_every_s``): ``Scheduler.select_followups`` requests replays at the pace of
  ``scheduler.followups_per_day`` (about followups_per_day · followup_every_s / 86400 per call), keeping at most
  :data:`FOLLOWUP_BACKLOG_PER_WORKER` requested follow-ups per worker waiting (workers prefer follow-ups, so a large
  backlog would starve the survey). ``select_followups`` itself enforces the daily cap and the priority order.
* **Notifications** (every step): new transients with severity ≥ ``notify.min_severity`` are pushed through the
  :class:`~tycho.survey.alerts.Notifier` (rate limited, with digest), weaker ones only logged; all are then marked
  notified. If every push channel fails, the transient stays pending and pushing is retried after
  :data:`NOTIFY_RETRY_S`.
* **Disk guard**: below ``survey.min_free_gb`` the workers pause claiming; the supervisor alerts once, and again only
  after the free space recovered above :data:`DISK_RECOVER_FACTOR` × the threshold and fell again.
* **Status** (every ``survey.status_every_s``, and at start and stop): ``reports/status.json`` and
  ``reports/dashboard.html`` (:mod:`tycho.survey.dashboard`).
* **Backup** (when the newest ``backups/catalog-*.sqlite`` is older than ``survey.backup_every_h``): an online SQLite
  backup; only the newest ``survey.backups_keep`` are kept.

Every periodic task is isolated: an exception is logged (and alerted once after :data:`TASK_ALERT_FAILURES` in a
row) and the loop goes on.

Stopping
    SIGTERM / SIGINT / SIGHUP (launchd's ``bootout``, Ctrl-C, a closed terminal), the ``hours`` limit or the
    ``max_objects`` budget set the shared stop event. Workers then stop at their next poll (within
    :data:`tycho.train.runner.POLL_STEPS` updates), save a resume checkpoint and mark their object *interrupted*;
    interrupted objects are claimed again later and resume exactly. The supervisor waits :data:`SHUTDOWN_TIMEOUT_S`
    for them (launchd's ``ExitTimeOut`` in ``ops/launchd/com.tycho.survey.plist`` is longer), then kills stragglers
    and hands their jobs back. A second signal stops waiting. Signal handlers only set a flag; the loop acts on it
    (no locks are taken in a handler).

``max_objects`` is a claim budget shared by the workers: once it is used up, nothing new is claimed and the daemon
stops when the claimed objects have finished.
"""
from __future__ import annotations

import fcntl
import logging
import logging.handlers
import math
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from tycho.config import Paths, load_config
from tycho.survey.alerts import Notifier, transient_notification
from tycho.survey.catalog import Catalog
from tycho.survey.dashboard import collect_status, write_dashboard, write_json_atomic
from tycho.survey.scheduler import Scheduler, SchedulerPolicy
from tycho.survey.worker import LOG_FILE, LOG_FORMAT, code_version, free_disk_gb, load_sky, worker_main

logger = logging.getLogger(__name__)

__all__ = ["BACKUP_GLOB", "LOCK_FILE", "LOOP_INTERVAL_S", "SHUTDOWN_TIMEOUT_S", "STATUS_FILE", "STOP_SIGNALS",
           "Supervisor", "SurveyLock", "SurveyLockError", "backup_catalog", "list_backups", "prune_backups",
           "resolve_n_workers", "run_survey", "running_daemon_pid"]

LOOP_INTERVAL_S = 2.0
"""Period of the supervisor loop."""

SHUTDOWN_TIMEOUT_S = 90.0
"""How long a stopping supervisor waits for its workers to checkpoint (launchd's ExitTimeOut must be longer)."""

KILL_GRACE_S = 5.0
"""After the shutdown timeout: wait this long after SIGTERM before SIGKILL."""

RESTART_BACKOFF_S = 5.0
RESTART_BACKOFF_MAX_S = 600.0
"""Restart delay of a worker slot whose worker died within :data:`STABLE_RUN_S` of its start: doubles from
RESTART_BACKOFF_S up to RESTART_BACKOFF_MAX_S; a worker that ran stably is restarted at once."""

STABLE_RUN_S = 60.0
"""A worker that ran at least this long before dying resets its slot's backoff."""

CRASH_ALERT_RESTARTS = 5
"""Restarts of one worker slot within an hour that raise an alert (once per episode)."""

SLEEP_GAP_S = 60.0
"""A step that starts this much later than the previous one means the machine slept (or the loop was blocked)."""

TASK_ALERT_FAILURES = 5
"""Consecutive failures of one periodic task that raise an alert (once per episode)."""

NOTIFY_RETRY_S = 600.0
"""Wait before pushing again after every channel failed."""

NOTIFY_BATCH = 50
"""Transients handled per step."""

DISK_RECOVER_FACTOR = 1.1
"""The low-disk alert re-arms once free space exceeds this multiple of ``survey.min_free_gb``."""

FOLLOWUP_BACKLOG_PER_WORKER = 1
"""Requested-but-unclaimed follow-ups allowed per worker before the scheduler is asked for more."""

BACKUP_RETRY_S = 3600.0
"""Wait before retrying a failed backup."""

LOG_MAX_BYTES = 10_000_000
LOG_BACKUPS = 10
"""Rotation of ``logs/survey.log``: 10 files of 10 MB."""

LOCK_FILE = "daemon.lock"
"""Lock file in ``data/`` (holds the daemon's pid)."""

STATUS_FILE = "status.json"
"""Status file in ``reports/``."""

NOTIFY_LOG = "notifications.log"
"""Notification log in ``logs/`` (JSON lines)."""

BACKUP_PREFIX, BACKUP_SUFFIX = "catalog-", ".sqlite"
BACKUP_GLOB = f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"
"""Backups are ``backups/catalog-<UTC timestamp>.sqlite``; names sort chronologically."""

STOP_SIGNALS: tuple[signal.Signals, ...] = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, name))
"""Signals that stop the survey gracefully (launchd's bootout sends SIGTERM; Ctrl-C SIGINT; closing the terminal of
a foreground ``tycho survey run`` SIGHUP)."""

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
"""Thread-count variables of the BLAS/OpenMP runtimes, set (unless already set) to ``survey.torch_threads`` before
workers start, so N workers do not oversubscribe the cores through numpy."""

_HOUR_S = 3600.0
_DAY_S = 86_400.0


# ---------------------------------------------------------------------------------------------- lock
class SurveyLockError(RuntimeError):
    """Another daemon holds the survey lock of this root."""


class SurveyLock:
    """Exclusive, non-blocking ``flock`` on ``data/daemon.lock`` (released automatically if the process dies)."""

    def __init__(self, paths: Paths) -> None:
        self.path = paths.data / LOCK_FILE
        self._fh: Any = None

    def acquire(self) -> "SurveyLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            holder = fh.read().strip() or "unknown"
            fh.close()
            raise SurveyLockError(f"another survey daemon (pid {holder}) is running on {self.path.parent.parent}; "
                                  f"stop it first (launchctl bootout gui/$UID/com.tycho.survey)") from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return self

    def release(self) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "SurveyLock":
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


def running_daemon_pid(paths: Paths) -> int | None:
    """The pid of the daemon holding the survey lock (0 if unreadable), or None when no daemon runs."""
    path = paths.data / LOCK_FILE
    if not path.exists():
        return None
    with open(path, "a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            text = fh.read().strip()
            return int(text) if text.isdigit() else 0
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return None


# ---------------------------------------------------------------------------------------------- backups
def list_backups(paths: Paths) -> list[Path]:
    """Catalog backups, oldest first."""
    return sorted(paths.backups.glob(BACKUP_GLOB), key=lambda p: p.name)


def prune_backups(paths: Paths, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` (at least 1) backups; returns the deleted paths."""
    backups = list_backups(paths)
    doomed = backups[:max(0, len(backups) - max(1, int(keep)))]
    for p in doomed:
        p.unlink(missing_ok=True)
    return doomed


def backup_catalog(catalog: Catalog, paths: Paths, keep: int, *, now: float | None = None) -> Path:
    """Online backup of the catalog to ``backups/catalog-<UTC timestamp>.sqlite``, then prune to ``keep``."""
    stamp = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = paths.backups / f"{BACKUP_PREFIX}{stamp}{BACKUP_SUFFIX}"
    n = 1
    while dest.exists():  # two backups within one second
        dest = paths.backups / f"{BACKUP_PREFIX}{stamp}-{n}{BACKUP_SUFFIX}"
        n += 1
    catalog.backup(dest)
    removed = prune_backups(paths, keep)
    logger.info("catalog backed up to %s (%d old backup(s) removed)", dest, len(removed))
    return dest


# ---------------------------------------------------------------------------------------------- supervisor
def resolve_n_workers(n_workers: int | None, cfg: Mapping[str, Any]) -> int:
    """``n_workers`` if given, else ``survey.n_workers``, else cores − 2 (at least 1)."""
    value = n_workers if n_workers is not None else (cfg.get("survey") or {}).get("n_workers")
    if value is None:
        value = max(1, (os.cpu_count() or 1) - 2)
    value = int(value)
    if value < 0:
        raise ValueError(f"n_workers must be >= 0, got {value}")
    return value


@dataclass
class _Slot:
    """One worker position: its process and restart bookkeeping."""

    worker_id: str
    process: Any = None
    started_at: float = 0.0
    next_start: float = 0.0
    backoff: float = RESTART_BACKOFF_S
    restarts: list[float] = field(default_factory=list)
    alerted: bool = False


class Supervisor:
    """The supervisor loop (see the module docstring); :meth:`step` runs one iteration.

    Args:
        root: the survey root.
        n_workers: worker processes (see :func:`resolve_n_workers`); 0 runs the loop without workers (tests).
        hours: stop after this many hours (None: run until stopped).
        max_objects: claim budget (None: unlimited).
        cfg: the survey config (default: ``load_config(root)``).
        clock: wall-clock source (seconds since the epoch).
        notifier: the notifier (default: one logging to ``logs/notifications.log``).
    """

    def __init__(self, root: Path | str, *, n_workers: int | None = None, hours: float | None = None,
                 max_objects: int | None = None, cfg: Mapping[str, Any] | None = None,
                 clock: Callable[[], float] = time.time, notifier: Notifier | None = None) -> None:
        self.paths = Paths(root).ensure()
        self.cfg = dict(cfg) if cfg is not None else load_config(self.paths.root)
        survey = self.cfg["survey"]
        self.clock = clock
        self.n_workers = resolve_n_workers(n_workers, self.cfg)
        if max_objects is not None and int(max_objects) < 0:
            raise ValueError(f"max_objects must be >= 0, got {max_objects}")
        if hours is not None and not (math.isfinite(float(hours)) and float(hours) > 0):
            raise ValueError(f"hours must be a positive number, got {hours}")
        self.max_objects = None if max_objects is None else int(max_objects)
        self.started_at = clock()
        self.deadline = None if hours is None else self.started_at + float(hours) * _HOUR_S
        self.catalog = Catalog(self.paths.db, root=self.paths.root, max_attempts=int(survey["max_attempts"]))
        self.sky = load_sky(self.paths, self.cfg)
        self.policy = SchedulerPolicy.from_dict(self.cfg.get("scheduler"))
        self.scheduler = Scheduler(self.catalog, self.sky, self.policy, device=str(survey.get("device") or "cpu"))
        self.notifier = notifier if notifier is not None else Notifier(self.cfg.get("notify"),
                                                                       self.paths.logs / NOTIFY_LOG, clock=clock)
        self.version = code_version(self.cfg)
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.claim_budget = self.ctx.Value("q", -1 if self.max_objects is None else self.max_objects)
        self.slots = [_Slot(f"w{i}") for i in range(self.n_workers)]
        self.stop_reason: str | None = None
        self.force = False
        self.signal_received: int | None = None
        self.state = "starting"
        self.disk: dict[str, Any] = {}
        self.counters = {"worker_starts": 0, "worker_restarts": 0, "requeued": 0, "enqueued": 0,
                         "followups_requested": 0, "notified": 0, "backups": 0}
        self._due: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        self._last_step: float | None = None
        self._grace_until = 0.0
        self._notify_retry_at = 0.0
        self._next_backup: float | None = None
        self._disk_alerted = False
        self._idle_budget_steps = 0

    # ------------------------------------------------------------------ control
    @property
    def stopping(self) -> bool:
        return self.stop_reason is not None

    def request_stop(self, reason: str) -> None:
        """Ask workers to checkpoint and stop (idempotent; the first reason is kept)."""
        if self.stop_reason is None:
            self.stop_reason = reason
            self.state = "stopping"
            self.stop_event.set()
            logger.info("stopping the survey: %s", reason)

    def handle_signal(self, signum: int, frame: Any = None) -> None:
        """Signal handler: only records the signal (the loop acts on it). A second signal stops waiting."""
        if self.signal_received is not None:
            self.force = True
        self.signal_received = signum

    def sleep(self, seconds: float) -> None:
        """Sleep in short slices, returning early once a signal arrived."""
        deadline = time.monotonic() + seconds
        seen = self.signal_received
        while time.monotonic() < deadline and self.signal_received == seen:
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))

    def start(self) -> None:
        """Start the workers and write the first status."""
        survey = self.cfg["survey"]
        threads = str(int(survey.get("torch_threads") or 1))
        for name in _THREAD_ENV:
            os.environ.setdefault(name, threads)
        logger.info("survey daemon starting: root %s, sky %s, %d workers, code %s, pid %d", self.paths.root,
                    self.sky.version, self.n_workers, self.version, os.getpid())
        now = self.clock()
        for slot in self.slots:
            self._start_worker(slot, now)
        self.state = "running"
        self._every("status", float(survey["status_every_s"]), now, self._write_status)

    # ------------------------------------------------------------------ one iteration
    def step(self, now: float | None = None) -> None:
        """One supervisor iteration (see the module docstring)."""
        now = self.clock() if now is None else now
        if self.signal_received is not None and not self.stopping:
            self.request_stop(f"signal {signal.Signals(self.signal_received).name}")
        if self._last_step is not None and now - self._last_step > SLEEP_GAP_S:
            self._grace_until = now + 2.0 * float(self.cfg["survey"]["heartbeat_s"])
            logger.warning("supervisor step %.0f s late (system sleep?): waiting %.0f s for heartbeats before "
                           "requeueing stale jobs", now - self._last_step, self._grace_until - now)
        self._last_step = now
        survey = self.cfg["survey"]
        if not self.stopping:
            self._task("workers", self._supervise_workers, now)
            self._every("requeue", float(survey["heartbeat_s"]), now, self._requeue)
            self._every("topup", float(survey["topup_every_s"]), now, self._top_up)
            self._every("followups", float(survey["followup_every_s"]), now, self._followups)
        self._task("disk", self._check_disk, now)
        self._task("notify", self._notify, now)
        self._every("status", float(survey["status_every_s"]), now, self._write_status)
        self._task("backup", self._maybe_backup, now)
        self._check_stop(now)

    def _every(self, name: str, period_s: float, now: float, fn: Callable[[float], None]) -> None:
        if now >= self._due.get(name, -math.inf):
            self._due[name] = now + period_s
            self._task(name, fn, now)

    def _task(self, name: str, fn: Callable[[float], None], now: float) -> None:
        """Run one periodic task; failures are logged and counted, never raised."""
        try:
            fn(now)
        except Exception:
            n = self._failures.get(name, 0) + 1
            self._failures[name] = n
            if n <= 3 or n % 100 == 0:
                logger.exception("supervisor task %r failed (%d in a row)", name, n)
            if n == TASK_ALERT_FAILURES:
                self.notifier.send(f"Tycho: supervisor task '{name}' keeps failing",
                                   f"{n} consecutive failures; last error:\n{traceback.format_exc(limit=5)}",
                                   "high", ("tycho", "daemon"))
        else:
            self._failures[name] = 0

    # ------------------------------------------------------------------ workers
    def _start_worker(self, slot: _Slot, now: float) -> None:
        process = self.ctx.Process(
            target=worker_main, name=f"tycho-{slot.worker_id}",
            args=(slot.worker_id, str(self.paths.root), self.stop_event),
            kwargs={"claim_budget": self.claim_budget, "parent_pid": os.getpid(), "configure_logging": True},
            daemon=True)
        process.start()
        slot.process, slot.started_at = process, now
        self.counters["worker_starts"] += 1
        logger.info("worker %s started (pid %s)", slot.worker_id, process.pid)

    def _supervise_workers(self, now: float) -> None:
        """Restart dead workers (with backoff) and kill hung ones."""
        for slot in self.slots:
            p = slot.process
            if p is not None and not p.is_alive():
                self._reap(slot, now)
            if slot.process is None and now >= slot.next_start:
                self._start_worker(slot, now)
                slot.restarts = [t for t in slot.restarts if now - t < _HOUR_S] + [now]
                self.counters["worker_restarts"] += 1
                if len(slot.restarts) >= CRASH_ALERT_RESTARTS and not slot.alerted:
                    slot.alerted = True
                    self.notifier.send(f"Tycho: worker {slot.worker_id} keeps dying",
                                       f"{len(slot.restarts)} restarts within an hour; see logs/{LOG_FILE}",
                                       "high", ("tycho", "daemon"))

    def _reap(self, slot: _Slot, now: float) -> None:
        p = slot.process
        p.join(timeout=0)
        code = p.exitcode
        slot.process = None
        held = self._release_budget_of(slot.worker_id)
        ran = now - slot.started_at
        if ran >= STABLE_RUN_S:
            slot.backoff = RESTART_BACKOFF_S
            delay = 0.0
            if now - max(slot.restarts, default=-math.inf) >= _HOUR_S:
                slot.alerted = False
        else:
            delay = slot.backoff
            slot.backoff = min(RESTART_BACKOFF_MAX_S, slot.backoff * 2.0)
        slot.next_start = now + delay
        logger.warning("worker %s (pid %s) exited with code %s after %.0f s%s; restarting in %.0f s", slot.worker_id,
                       p.pid, code, ran, f", holding {held} object(s)" if held else "", delay)

    def _release_budget_of(self, worker_id: str) -> int:
        """Refund the claim budget for objects a dead worker still holds (they will be claimed again)."""
        held = self.catalog.query("SELECT COUNT(*) AS n FROM objects WHERE status = 'running' AND worker = ?",
                                  (worker_id,))[0]["n"]
        if held and self.max_objects is not None:
            with self.claim_budget.get_lock():
                self.claim_budget.value += held
        return int(held)

    def _requeue(self, now: float) -> None:
        if now < self._grace_until:
            return
        stale_after = float(self.cfg["survey"]["stale_after_s"])
        n = self.catalog.requeue_stale(stale_after)
        if n:
            self.counters["requeued"] += n
            logger.warning("requeued %d stale job(s)", n)
        beats = {r["worker_id"]: r["updated_at"] for r in self.catalog.query(
            "SELECT worker_id, updated_at FROM heartbeats")}
        for slot in self.slots:
            p = slot.process
            if p is None or not p.is_alive() or now - slot.started_at <= stale_after:
                continue
            seen = beats.get(slot.worker_id)
            if seen is None or now - seen > stale_after:
                logger.error("worker %s (pid %s) sent no heartbeat for %s; killing it (it will be restarted)",
                             slot.worker_id, p.pid, "ever" if seen is None else f"{now - seen:.0f} s")
                p.kill()

    # ------------------------------------------------------------------ scheduling
    def _top_up(self, now: float) -> None:
        n = self.scheduler.top_up()
        if n:
            self.counters["enqueued"] += n
            logger.info("scheduler enqueued %d object(s)", n)

    def _followups(self, now: float) -> None:
        per_day = int(self.policy.followups_per_day)
        if per_day <= 0:
            return
        pending = self.catalog.query("SELECT COUNT(*) AS n FROM followups WHERE status = 'requested'")[0]["n"]
        room = max(1, self.n_workers) * FOLLOWUP_BACKLOG_PER_WORKER - pending
        if room <= 0:
            return
        pace = max(1, math.ceil(per_day * float(self.cfg["survey"]["followup_every_s"]) / _DAY_S))
        selected = self.scheduler.select_followups(min(room, pace))
        if selected:
            self.counters["followups_requested"] += len(selected)
            logger.info("requested %d follow-up(s): %s", len(selected),
                        "; ".join(f"{oid} ({reason})" for oid, reason in selected))

    # ------------------------------------------------------------------ notifications, disk
    def _notify(self, now: float) -> None:
        if now >= self._notify_retry_at:
            min_severity = int((self.cfg.get("notify") or {}).get("min_severity", 2))
            handled = []
            for row in self.catalog.pending_notifications(min_severity=1, limit=NOTIFY_BATCH):
                title, message, priority, tags = transient_notification(row)
                if int(row["severity"]) >= min_severity:
                    if not self.notifier.send(title, message, priority, tags):
                        self._notify_retry_at = now + NOTIFY_RETRY_S
                        logger.warning("notification channels failing; retrying in %.0f s", NOTIFY_RETRY_S)
                        break
                else:
                    self.notifier.record(title, message, priority, tags)
                handled.append(int(row["transient_id"]))
            if handled:
                self.catalog.mark_notified(handled)
                self.counters["notified"] += len(handled)
        self.notifier.flush_digest()

    def _check_disk(self, now: float) -> None:
        min_free = float(self.cfg["survey"]["min_free_gb"])
        free = free_disk_gb(self.paths.root)
        self.disk = {"free_gb": free, "min_free_gb": min_free, "low": free < min_free}
        if free < min_free and not self._disk_alerted:
            self._disk_alerted = True
            logger.error("disk space low: %.1f GB free < survey.min_free_gb = %g; workers pause claiming", free,
                         min_free)
            self.notifier.send("Tycho: disk space low",
                               f"{free:.1f} GB free on {self.paths.root} (survey.min_free_gb = {min_free:g}). Workers "
                               f"have paused claiming new work until space is freed.", "high", ("tycho", "disk"))
        elif self._disk_alerted and free >= DISK_RECOVER_FACTOR * min_free:
            self._disk_alerted = False
            logger.info("disk space recovered: %.1f GB free", free)

    # ------------------------------------------------------------------ status, backups
    def status_extra(self, now: float) -> dict[str, Any]:
        """The daemon part of ``status.json``."""
        return {
            "state": self.state,
            "daemon": {"pid": os.getpid(), "host": socket.gethostname(), "started_at": self.started_at,
                       "uptime_s": now - self.started_at, "code_version": self.version,
                       "sky_version": self.sky.version, "n_workers": self.n_workers, "max_objects": self.max_objects,
                       "claims_left": None if self.max_objects is None else int(self.claim_budget.value),
                       "deadline": self.deadline, "stop_reason": self.stop_reason, "counters": dict(self.counters)},
            "processes": [{"worker_id": s.worker_id, "pid": None if s.process is None else s.process.pid,
                           "alive": bool(s.process is not None and s.process.is_alive()),
                           "restarts_last_hour": len([t for t in s.restarts if now - t < _HOUR_S])}
                          for s in self.slots],
            "notifications": {"channels": self.notifier.channels, "held_for_digest": self.notifier.held},
        }

    def _write_status(self, now: float) -> None:
        status = collect_status(self.catalog, self.paths, self.cfg, now=now, extra=self.status_extra(now))
        write_json_atomic(self.paths.reports / STATUS_FILE, status)
        write_dashboard(self.catalog, self.paths, status=status)

    def _maybe_backup(self, now: float) -> None:
        survey = self.cfg["survey"]
        period = float(survey["backup_every_h"]) * _HOUR_S
        if self._next_backup is None:
            backups = list_backups(self.paths)
            self._next_backup = backups[-1].stat().st_mtime + period if backups else now
        if now < self._next_backup:
            return
        self._next_backup = now + BACKUP_RETRY_S  # if the backup below fails
        backup_catalog(self.catalog, self.paths, int(survey["backups_keep"]), now=now)
        self.counters["backups"] += 1
        self._next_backup = now + period

    # ------------------------------------------------------------------ stopping
    def _check_stop(self, now: float) -> None:
        if self.stopping:
            return
        if self.deadline is not None and now >= self.deadline:
            self.request_stop("time limit reached")
        elif self.max_objects is not None and self.claim_budget.value == 0:
            running = self.catalog.query("SELECT COUNT(*) AS n FROM objects WHERE status = 'running'")[0]["n"]
            # two consecutive idle steps, so that a claim in flight (budget taken, row not yet running) completes
            self._idle_budget_steps = self._idle_budget_steps + 1 if running == 0 else 0
            if self._idle_budget_steps >= 2:
                self.request_stop(f"max_objects={self.max_objects} observed")

    def shutdown(self, timeout_s: float = SHUTDOWN_TIMEOUT_S) -> dict[str, Any]:
        """Stop the workers (graceful, then forced after ``timeout_s``), write the final status and return a
        summary of the run."""
        self.request_stop(self.stop_reason or "shutdown")
        deadline = time.monotonic() + timeout_s
        while any(s.process is not None and s.process.is_alive() for s in self.slots):
            if self.force or time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        stragglers = [s for s in self.slots if s.process is not None and s.process.is_alive()]
        for s in stragglers:
            logger.error("worker %s (pid %s) did not stop in time; terminating it", s.worker_id, s.process.pid)
            s.process.terminate()
        end = time.monotonic() + KILL_GRACE_S
        for s in stragglers:
            s.process.join(timeout=max(0.0, end - time.monotonic()))
            if s.process.is_alive():
                s.process.kill()
                s.process.join(timeout=KILL_GRACE_S)
        for s in self.slots:
            if s.process is not None:
                s.process.join(timeout=0)
        for s in stragglers:  # their objects lost the progress since their last checkpoint
            try:
                n = self.catalog.release_worker(s.worker_id)
                if n:
                    logger.warning("handed back %d job(s) of the killed worker %s", n, s.worker_id)
            except Exception:
                logger.exception("could not hand back the jobs of worker %s", s.worker_id)
        now = self.clock()
        self.state = "stopped"
        self._task("status", self._write_status, now)
        summary = {"started_at": self.started_at, "stopped_at": now, "elapsed_s": now - self.started_at,
                   "stop_reason": self.stop_reason, "n_workers": self.n_workers, "counters": dict(self.counters),
                   "stats": self.catalog.stats()}
        self.catalog.close()
        logger.info("survey daemon stopped (%s)", self.stop_reason)
        return summary


# ---------------------------------------------------------------------------------------------- entry point
def _configure_logging(paths: Paths) -> logging.Handler:
    """Add the rotating survey log to the root logger (removed again by :func:`run_survey`)."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(paths.logs / LOG_FILE, maxBytes=LOG_MAX_BYTES,
                                                   backupCount=LOG_BACKUPS, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    if root_logger.getEffectiveLevel() > logging.INFO:
        root_logger.setLevel(logging.INFO)
    logging.captureWarnings(True)
    return handler


def run_survey(root: Path, *, hours: float | None = None, max_objects: int | None = None,
               n_workers: int | None = None) -> dict:
    """Run the survey daemon on ``root`` until a signal, ``hours`` or ``max_objects`` stops it.

    Returns a summary: start/stop times, stop reason, worker count, counters (worker starts and restarts, requeued
    jobs, enqueued objects, follow-ups requested, transients notified, backups) and the final ``Catalog.stats()``.

    Raises:
        SurveyLockError: another daemon is running on this root.
    """
    paths = Paths(root).ensure()
    with SurveyLock(paths):
        handler = _configure_logging(paths)
        try:
            return _run(paths, hours=hours, max_objects=max_objects, n_workers=n_workers)
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()


def _run(paths: Paths, *, hours: float | None, max_objects: int | None, n_workers: int | None) -> dict:
    try:
        sup = Supervisor(paths.root, n_workers=n_workers, hours=hours, max_objects=max_objects)
    except Exception:
        logger.exception("the survey daemon cannot start")
        _alert_start_failure(paths, traceback.format_exc(limit=8))
        raise
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in STOP_SIGNALS:
            previous[sig] = signal.signal(sig, sup.handle_signal)
    try:
        sup.start()
        while not sup.stopping:
            sup.step()
            if not sup.stopping:
                sup.sleep(LOOP_INTERVAL_S)
    finally:
        try:
            summary = sup.shutdown()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return summary


def _alert_start_failure(paths: Paths, error: str) -> None:
    """Best-effort alert that the daemon cannot start (e.g. a broken sky file); launchd will retry it, and the
    notifier's persisted rate limit keeps a crash loop from flooding the channels."""
    try:
        cfg = load_config(paths.root)
    except Exception:  # a broken tycho.yaml: log-only notifier with the defaults
        cfg = {}
    Notifier(cfg.get("notify"), paths.logs / NOTIFY_LOG).send("Tycho: survey daemon cannot start", error, "urgent",
                                                              ("tycho", "daemon"))
