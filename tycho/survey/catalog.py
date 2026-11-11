"""The survey catalog (SPEC §6.1): one SQLite database holding every object, its photometry summary, light-curve
location, classifications, RI spectroscopy (runs, events, transients), follow-up requests, proposals, the survey
log, worker heartbeats and the agent's audit trail.

Storage
    ``<root>/data/catalog.sqlite`` in WAL mode, plus per-object files written by the workers (light curves as
    ``np.savez_compressed``, RI curves via ``RICurve.save``). File paths are stored **relative to the survey root**
    (POSIX separators) when the file lies inside the root and absolute otherwise; :meth:`Catalog.resolve_path` turns a
    stored path back into an absolute one, so a survey root can be moved, copied or released as a whole. The root is
    the ``root`` argument, or else inferred from the ``Paths`` layout: the parent of the ``data/`` directory holding
    the database (the database's own directory when it is not in a directory named ``data``).

Concurrency model
    * One connection per process, opened lazily on first use (never in ``__init__``). A ``Catalog`` pickles without
      its connection, so it can be handed to ``spawn`` workers; after ``fork()`` the child drops the inherited
      connection unused (SQLite forbids using it, and closing it could disturb the parent's locks) and opens its own.
    * Threads of one process (e.g. a worker and its heartbeat thread) share the connection under a lock.
    * Pragmas: ``journal_mode=WAL`` (readers never block the writer), ``synchronous=NORMAL`` (durable at WAL
      checkpoints; a power cut can lose only the last transactions, never corrupt the file), ``busy_timeout=30000``,
      ``foreign_keys=ON`` (child rows must reference an existing object / RI run / event / agent run).
    * Every method commits its own short transaction. Writes use ``BEGIN IMMEDIATE``, which takes the database's
      single write lock up front and waits for it up to the busy timeout; a deferred transaction that reads and then
      writes can instead fail with SQLITE_BUSY at the upgrade, which the busy timeout does not cover. This is what
      makes :meth:`Catalog.claim_next` safe across processes: select-and-mark happens under the write lock.

Object life cycle (``objects.status``)
    ``queued → running`` (:meth:`claim_next`) ``→ done | failed | interrupted``; ``interrupted`` objects are claimable
    like queued ones and resume from their checkpoint. :meth:`requeue_stale` and :meth:`release_worker` take running
    objects back from lost workers (``running → queued``, or ``failed`` once out of attempts). ``attempts`` counts
    claims, but a graceful interruption (:meth:`mark_interrupted`) refunds its attempt, so ``attempts`` is the number of
    claims that crashed, failed, completed or are still running; a stale object that has used ``max_attempts`` fails
    instead of being requeued. Claims take the highest ``priority`` first, then the oldest (``created_at``, then
    insertion order).

Follow-ups (``followups.status``)
    ``requested → running`` (:meth:`claim_followup`, only for objects that are done) ``→ done | failed``; stale running
    follow-ups go back to ``requested`` (or ``failed``) under the same attempt rule. An object has at most one
    follow-up row ever, which is how "never follow up an object twice" is enforced. *Addition to the SPEC schema:*
    ``followups.attempts`` (the SPEC's "attempts counted" needs a counter; the SPEC table has none).

Heartbeats
    :meth:`heartbeat` upserts one row per worker (pid, host, current object or None, free-form status such as
    "idle", "running", "followup"). :meth:`claim_next` and :meth:`claim_followup` upsert it too, so a fresh claim is
    never mistaken for a stale one. A running job is stale when its worker has no heartbeat row or the last heartbeat
    is at least ``stale_after_s`` old.

JSON conventions
    Summaries, tags, points and other JSON columns hold strict JSON: NaN and ±inf become ``null``, numpy scalars and
    arrays become numbers and lists. The numeric summary columns are NULL when the value is not finite, so SQL
    aggregates skip them. ``ri_events.subset_json`` is the JSON list of the subset's variable *names*.

Schema versioning
    ``meta['schema_version']`` holds :data:`SCHEMA_VERSION`. Opening a database written with another version raises
    :class:`CatalogSchemaError` (migrations are added together with the next schema change).

Read-only queries
    :meth:`Catalog.query` runs one SELECT/WITH statement for the agent and the reports. It is checked lexically (the
    first keyword is SELECT or WITH, a single statement; string literals and comments are skipped) and enforced by an
    SQLite authorizer that denies every action other than reading, so writes hidden in a CTE
    (``WITH x AS (...) DELETE ...``), ATTACH or PRAGMA writes fail at prepare time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import socket
import sqlite3
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

# _canonical is the object-identity normalisation; reusing it makes proposals.point_json equal sky.point_key.
from tycho.train.spec import ObjectSpec, RunResult, _canonical

if TYPE_CHECKING:  # pragma: no cover
    from tycho.analysis.classify import Classification
    from tycho.analysis.events import RIEvent, Transient
    from tycho.analysis.ricurve import RICurve

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
"""Version of the table layout below, stored in ``meta['schema_version']``."""

BUSY_TIMEOUT_MS = 30_000
"""How long a connection waits for another process's write lock before failing (SQLite ``busy_timeout``)."""

DEFAULT_MAX_ATTEMPTS = 3
"""A stale running job that has been claimed this many times is marked failed instead of being requeued."""

MAX_TEXT_CHARS = 20_000
"""Error messages and tool-call previews are truncated to this many characters (head and tail are kept)."""

OBJECT_STATUSES: tuple[str, ...] = ("queued", "running", "done", "failed", "interrupted")
CLAIMABLE_STATUSES: tuple[str, ...] = ("queued", "interrupted")
FOLLOWUP_STATUSES: tuple[str, ...] = ("requested", "running", "done", "failed")

SUMMARY_COLUMNS: tuple[str, ...] = ("final_test_acc", "final_test_loss", "final_train_loss", "t_test_acc90",
                                    "grok_delay", "plateau_frac", "n_drops", "diverged")
"""Summary keys copied into columns of ``summaries`` (the full summary is in ``summaries.json``)."""

AGENT_RUN_FIELDS: frozenset[str] = frozenset({
    "kind", "started_at", "finished_at", "model", "input_tokens", "output_tokens", "cache_read_tokens",
    "cache_write_tokens", "est_cost_usd", "n_tool_calls", "status", "stop_reason", "error"})
"""Columns of ``agent_runs`` that :meth:`Catalog.finish_agent_run` may set."""

_SECONDS_PER_DAY = 86_400.0

_SCHEMA: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)",
    """CREATE TABLE IF NOT EXISTS objects (
        object_id TEXT PRIMARY KEY, task TEXT NOT NULL, region TEXT NOT NULL, sky_version TEXT NOT NULL,
        spec_json TEXT NOT NULL, seed INTEGER NOT NULL, source TEXT, priority REAL NOT NULL DEFAULT 1.0,
        status TEXT NOT NULL DEFAULT 'queued'
            CHECK (status IN ('queued', 'running', 'done', 'failed', 'interrupted')),
        attempts INTEGER NOT NULL DEFAULT 0, worker TEXT, created_at REAL NOT NULL, started_at REAL,
        finished_at REAL, wall_time_s REAL, device TEXT, n_params INTEGER, error TEXT, code_version TEXT,
        random_control INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS summaries (
        object_id TEXT PRIMARY KEY REFERENCES objects(object_id), json TEXT NOT NULL, final_test_acc REAL,
        final_test_loss REAL, final_train_loss REAL, t_test_acc90 REAL, grok_delay REAL, plateau_frac REAL,
        n_drops INTEGER, diverged INTEGER)""",
    """CREATE TABLE IF NOT EXISTS lightcurves (
        object_id TEXT PRIMARY KEY REFERENCES objects(object_id), path TEXT NOT NULL, n_points INTEGER)""",
    """CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY, object_id TEXT NOT NULL REFERENCES objects(object_id), label TEXT NOT NULL,
        tags_json TEXT, score REAL, classifier TEXT, note TEXT, created_at REAL)""",
    """CREATE TABLE IF NOT EXISTS followups (
        object_id TEXT PRIMARY KEY REFERENCES objects(object_id), reason TEXT, priority REAL,
        status TEXT NOT NULL DEFAULT 'requested' CHECK (status IN ('requested', 'running', 'done', 'failed')),
        requested_by TEXT, created_at REAL, started_at REAL, finished_at REAL, worker TEXT, error TEXT,
        attempts INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS ri_runs (
        ri_run_id INTEGER PRIMARY KEY, object_id TEXT NOT NULL REFERENCES objects(object_id), kind TEXT, path TEXT,
        n_subsets INTEGER, n_points INTEGER, max_order INTEGER, alpha REAL, min_effect REAL, created_at REAL)""",
    """CREATE TABLE IF NOT EXISTS ri_events (
        event_id INTEGER PRIMARY KEY, ri_run_id INTEGER NOT NULL REFERENCES ri_runs(ri_run_id),
        object_id TEXT NOT NULL REFERENCES objects(object_id), subset_json TEXT, order_k INTEGER, kind TEXT,
        step INTEGER, step_lo INTEGER, step_hi INTEGER, ri_before REAL, ri_after REAL, peak_ri REAL, confidence REAL,
        has_precursor INTEGER, matches_gt INTEGER, created_at REAL)""",
    """CREATE TABLE IF NOT EXISTS transients (
        transient_id INTEGER PRIMARY KEY, object_id TEXT NOT NULL REFERENCES objects(object_id),
        event_id INTEGER REFERENCES ri_events(event_id), kind TEXT, severity INTEGER, message TEXT, created_at REAL,
        notified INTEGER NOT NULL DEFAULT 0, acknowledged INTEGER NOT NULL DEFAULT 0, note TEXT)""",
    """CREATE TABLE IF NOT EXISTS proposals (
        proposal_id INTEGER PRIMARY KEY, source TEXT, region TEXT, task TEXT, point_json TEXT, seeds INTEGER,
        rationale TEXT, status TEXT, created_at REAL, decided_at REAL, n_enqueued INTEGER)""",
    """CREATE TABLE IF NOT EXISTS survey_log (
        entry_id INTEGER PRIMARY KEY, created_at REAL, author TEXT, kind TEXT, title TEXT, markdown TEXT)""",
    """CREATE TABLE IF NOT EXISTS agent_runs (
        run_id INTEGER PRIMARY KEY, kind TEXT, started_at REAL, finished_at REAL, model TEXT, input_tokens INTEGER,
        output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, est_cost_usd REAL,
        n_tool_calls INTEGER, status TEXT, stop_reason TEXT, error TEXT)""",
    """CREATE TABLE IF NOT EXISTS agent_tool_calls (
        id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES agent_runs(run_id), created_at REAL, tool TEXT,
        input_json TEXT, output_preview TEXT, is_error INTEGER)""",
    """CREATE TABLE IF NOT EXISTS heartbeats (
        worker_id TEXT PRIMARY KEY, pid INTEGER, host TEXT, object_id TEXT, status TEXT, updated_at REAL)""",
    "CREATE INDEX IF NOT EXISTS idx_objects_claim ON objects(status, priority DESC, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_objects_region ON objects(region, status)",
    "CREATE INDEX IF NOT EXISTS idx_objects_finished ON objects(finished_at)",
    "CREATE INDEX IF NOT EXISTS idx_classifications_object ON classifications(object_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_followups_claim ON followups(status, priority DESC, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_ri_runs_object ON ri_runs(object_id)",
    "CREATE INDEX IF NOT EXISTS idx_ri_events_object ON ri_events(object_id)",
    "CREATE INDEX IF NOT EXISTS idx_ri_events_run ON ri_events(ri_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_transients_object ON transients(object_id)",
    "CREATE INDEX IF NOT EXISTS idx_transients_event ON transients(event_id)",
    "CREATE INDEX IF NOT EXISTS idx_transients_pending ON transients(notified, severity)",
    "CREATE INDEX IF NOT EXISTS idx_proposals_source ON proposals(source, region)",
    "CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run ON agent_tool_calls(run_id)",
)


class CatalogSchemaError(RuntimeError):
    """The database was written with a different catalog schema version than this code."""


# ---------------------------------------------------------------------------------------------- process safety
_LIVE_CATALOGS: "weakref.WeakSet[Catalog]" = weakref.WeakSet()

_INHERITED_CONNECTIONS: list[sqlite3.Connection] = []
"""Connections a forked child inherited from its parent. SQLite forbids using them in the child and closing them
could release locks the parent relies on, so the child keeps them referenced, unused, until it exits."""


def _after_fork_in_child() -> None:
    for catalog in list(_LIVE_CATALOGS):
        catalog._reset_after_fork()


if hasattr(os, "register_at_fork"):  # POSIX
    os.register_at_fork(after_in_child=_after_fork_in_child)


# ---------------------------------------------------------------------------------------------- the catalog
class Catalog:
    """SQLite catalog of one survey (see the module docstring for the concurrency model and life cycles).

    Args:
        path: database file (normally ``Paths(root).db``); its directory is created on first use.
        root: survey root that stored file paths are relative to (default: inferred from ``path``, see module doc).
        max_attempts: default attempt limit of :meth:`requeue_stale` / :meth:`release_worker`.
        clock: wall-clock source in seconds since the epoch (injectable for tests; must be picklable for spawn).
    """

    def __init__(self, path: Path | str, *, root: Path | str | None = None,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path).expanduser().resolve()
        self.root = Path(root).expanduser().resolve() if root is not None else _infer_root(self.path)
        if int(max_attempts) < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_attempts = int(max_attempts)
        self._clock = clock
        self._init_process_state()

    # ------------------------------------------------------------------ process state / connection
    def _init_process_state(self) -> None:
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._pid = os.getpid()
        self._host = socket.gethostname()
        _LIVE_CATALOGS.add(self)

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "root": self.root, "max_attempts": self.max_attempts, "clock": self._clock}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.path, self.root = state["path"], state["root"]
        self.max_attempts, self._clock = state["max_attempts"], state["clock"]
        self._init_process_state()

    def __repr__(self) -> str:
        return f"Catalog({str(self.path)!r}, root={str(self.root)!r})"

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close this process's connection (a later call reopens it)."""
        with self._lock:
            if self._pid != os.getpid():
                self._drop_inherited_connection()
            elif self._conn is not None:
                self._conn.close()
            self._conn = None

    def _reset_after_fork(self) -> None:
        """Runs in a forked child (single-threaded): drop the inherited connection and lock."""
        self._drop_inherited_connection()
        self._lock = threading.RLock()

    def _drop_inherited_connection(self) -> None:
        if self._conn is not None:
            _INHERITED_CONNECTIONS.append(self._conn)
        self._conn = None
        self._pid = os.getpid()

    def _connection(self) -> sqlite3.Connection:
        """This process's connection, opened and schema-checked on first use. The caller holds ``self._lock``."""
        if self._pid != os.getpid():  # forked without the at-fork hook having run
            self._drop_inherited_connection()
        if self._conn is None:
            self._conn = self._open()
        return self._conn

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=BUSY_TIMEOUT_MS / 1000.0, isolation_level=None,
                               check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                logger.warning("catalog %s: journal_mode is %s, not WAL (network filesystem?)", self.path, mode)
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            self._ensure_schema(conn)
        except BaseException:
            conn.close()
            raise
        logger.debug("catalog %s opened in pid %d", self.path, os.getpid())
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = None
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone():
                row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
                version = None if row is None else int(row[0])
            if version is not None and version != SCHEMA_VERSION:
                raise CatalogSchemaError(f"catalog {self.path} has schema version {version}; this code writes "
                                         f"version {SCHEMA_VERSION} and has no migration for it")
            for stmt in _SCHEMA:  # idempotent (IF NOT EXISTS); also adds indices missing from older files
                conn.execute(stmt)
            if version is None:
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                             (str(SCHEMA_VERSION),))
                conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', ?)", (repr(self.now()),))
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One short write transaction: BEGIN IMMEDIATE … COMMIT, rolled back on any exception."""
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        """A consistent read snapshot (deferred transaction; never blocks writers in WAL mode)."""
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN")
            try:
                yield conn
            finally:
                if conn.in_transaction:
                    conn.execute("COMMIT")

    def now(self) -> float:
        """Current time from the catalog's clock (seconds since the epoch); all timestamps use it."""
        return float(self._clock())

    # ------------------------------------------------------------------ paths
    def store_path(self, path: Path | str) -> str:
        """The form in which ``path`` is stored: relative to the survey root (POSIX) if inside it, else absolute.

        A relative ``path`` is taken to be relative to the survey root already.
        """
        p = Path(path).expanduser()
        if not p.is_absolute():
            return PurePath(p).as_posix()
        p = p.resolve()
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return p.as_posix()

    def resolve_path(self, stored: str) -> Path:
        """Absolute path of a stored (root-relative or absolute) path."""
        p = Path(stored)
        return p if p.is_absolute() else self.root / p

    # ------------------------------------------------------------------ objects
    def add_objects(self, specs: Sequence[ObjectSpec], *, region: str, source: str, priority: float = 1.0,
                    random_control_frac: float = 0.0, rng: np.random.Generator | None = None) -> int:
        """Enqueue objects (INSERT OR IGNORE by object id) and return how many were new.

        ``region`` is the catalog region of the objects; it replaces ``spec.region`` in the stored spec (region is
        bookkeeping, not identity). Each spec is marked as a random control with probability
        ``random_control_frac``: the uniform draw is ``rng.random()`` per spec (in order) when ``rng`` is given,
        otherwise a deterministic hash of the object id. Marks are fixed at first insertion.
        """
        if not region or not isinstance(region, str):
            raise ValueError("region must be a non-empty string")
        priority = float(priority)
        if not math.isfinite(priority):
            raise ValueError(f"priority must be finite, got {priority}")
        if not 0.0 <= random_control_frac <= 1.0:
            raise ValueError(f"random_control_frac must be in [0, 1], got {random_control_frac}")
        specs = list(specs)
        if not specs:
            return 0
        ids = [s.object_id() for s in specs]
        if random_control_frac > 0.0:
            draws = rng.random(len(specs)) if rng is not None else [_unit_hash("random_control", i) for i in ids]
            marks = [int(u < random_control_frac) for u in draws]
        else:
            marks = [0] * len(specs)
        now = self.now()
        rows = [(oid, s.task, region, s.sky_version, replace(s, region=region).to_json(), int(s.seed), source,
                 priority, now, mark) for oid, s, mark in zip(ids, specs, marks)]
        with self._write() as conn:
            cur = conn.executemany(
                "INSERT OR IGNORE INTO objects (object_id, task, region, sky_version, spec_json, seed, source, "
                "priority, status, attempts, created_at, random_control) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?)", rows)
            n_new = max(cur.rowcount, 0)
        logger.debug("add_objects: %d of %d new in region %s (source %s)", n_new, len(rows), region, source)
        return n_new

    def claim_next(self, worker_id: str) -> ObjectSpec | None:
        """Claim the next queued or interrupted object for ``worker_id`` (None if there is none).

        Highest priority first, then oldest. The object becomes ``running`` with ``attempts + 1`` and the worker's
        heartbeat is refreshed in the same transaction. A stored spec that no longer parses (e.g. written by an
        incompatible code version) is marked failed and skipped, so one bad row cannot stall the survey.
        """
        now = self.now()
        with self._write() as conn:
            while True:
                row = conn.execute(
                    "SELECT object_id, spec_json FROM objects WHERE status IN ('queued', 'interrupted') "
                    "ORDER BY priority DESC, created_at ASC, rowid ASC LIMIT 1").fetchone()
                if row is None:
                    return None
                try:
                    spec = ObjectSpec.from_json(row["spec_json"])
                except (ValueError, KeyError, TypeError) as exc:
                    logger.error("object %s has an unparseable spec and is marked failed: %s", row["object_id"], exc)
                    conn.execute("UPDATE objects SET status = 'failed', finished_at = ?, error = ? "
                                 "WHERE object_id = ?", (now, _truncate(f"unparseable spec: {exc!r}"),
                                                         row["object_id"]))
                    continue
                conn.execute("UPDATE objects SET status = 'running', worker = ?, started_at = ?, "
                             "attempts = attempts + 1 WHERE object_id = ?", (worker_id, now, row["object_id"]))
                self._upsert_heartbeat(conn, worker_id, row["object_id"], "running", now)
                return spec

    def heartbeat(self, worker_id: str, object_id: str | None = None, status: str = "idle") -> None:
        """Record that ``worker_id`` is alive, what it is working on (None when idle) and a free-form status."""
        with self._write() as conn:
            self._upsert_heartbeat(conn, worker_id, object_id, status, self.now())

    def _upsert_heartbeat(self, conn: sqlite3.Connection, worker_id: str, object_id: str | None, status: str,
                          now: float) -> None:
        conn.execute(
            "INSERT INTO heartbeats (worker_id, pid, host, object_id, status, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(worker_id) DO UPDATE SET pid = excluded.pid, host = excluded.host, "
            "object_id = excluded.object_id, status = excluded.status, updated_at = excluded.updated_at",
            (worker_id, os.getpid(), self._host, object_id, status, now))

    def requeue_stale(self, stale_after_s: float, *, max_attempts: int | None = None) -> int:
        """Take back running objects and follow-ups whose worker's last heartbeat is missing or at least
        ``stale_after_s`` old: back to queued / requested, or failed once they have used ``max_attempts`` claims
        (default ``self.max_attempts``). Returns the number of jobs taken back. ``stale_after_s = 0`` releases every
        running job (for a supervisor that knows no worker is alive, e.g. at startup)."""
        if stale_after_s < 0:
            raise ValueError(f"stale_after_s must be >= 0, got {stale_after_s}")
        now = self.now()
        with self._write() as conn:
            return self._release(conn, "(h.updated_at IS NULL OR h.updated_at <= ?)", (now - stale_after_s,),
                                 self._max_attempts(max_attempts), now)

    def release_worker(self, worker_id: str, *, max_attempts: int | None = None) -> int:
        """Take back every job still marked running for ``worker_id`` regardless of heartbeat age (same attempt
        rule as :meth:`requeue_stale`). A worker that restarts under a stable id should call this before its first
        claim: anything still running under its id belongs to a dead previous incarnation, and its fresh heartbeats
        would otherwise keep those jobs from ever looking stale."""
        now = self.now()
        with self._write() as conn:
            return self._release(conn, "x.worker = ?", (worker_id,), self._max_attempts(max_attempts), now)

    def _max_attempts(self, max_attempts: int | None) -> int:
        value = self.max_attempts if max_attempts is None else int(max_attempts)
        if value < 1:
            raise ValueError(f"max_attempts must be >= 1, got {value}")
        return value

    def _release(self, conn: sqlite3.Connection, condition: str, params: tuple, max_attempts: int,
                 now: float) -> int:
        n = 0
        for table, key, back in (("objects", "object_id", "queued"), ("followups", "object_id", "requested")):
            rows = conn.execute(
                f"SELECT x.{key} AS id, x.attempts, x.worker, h.updated_at AS seen FROM {table} x "
                f"LEFT JOIN heartbeats h ON h.worker_id = x.worker WHERE x.status = 'running' AND {condition}",
                params).fetchall()
            for r in rows:
                age = "no heartbeat" if r["seen"] is None else f"last heartbeat {now - r['seen']:.0f} s ago"
                lost = f"worker {r['worker']!r} lost ({age}) during attempt {r['attempts']}/{max_attempts}"
                if r["attempts"] >= max_attempts:
                    conn.execute(f"UPDATE {table} SET status = 'failed', finished_at = ?, error = ? WHERE {key} = ?",
                                 (now, f"gave up: {lost}", r["id"]))
                    logger.warning("%s %s failed: %s", table, r["id"], lost)
                else:
                    conn.execute(f"UPDATE {table} SET status = ?, worker = NULL, error = ? WHERE {key} = ?",
                                 (back, f"requeued: {lost}", r["id"]))
                    logger.info("%s %s requeued: %s", table, r["id"], lost)
                n += 1
        return n

    def mark_done(self, result: RunResult, lc_path: Path | str, *, code_version: str | None = None) -> None:
        """Record a finished observation: the object row, its summary (JSON plus the :data:`SUMMARY_COLUMNS`) and
        its light-curve file (stored root-relative, see :meth:`store_path`). A valid result is recorded whatever the
        object's current status (e.g. a slow worker finishing after its object was requeued)."""
        if result.status != "done":
            raise ValueError(f"mark_done needs a result with status 'done', got {result.status!r} "
                             f"(use mark_failed / mark_interrupted)")
        summary = dict(result.summary or {})
        lc = result.light_curve or {}
        n_points = len(lc["step"]) if "step" in lc else 0
        now = self.now()
        with self._write() as conn:
            self._require_object(conn, result.object_id)
            conn.execute("UPDATE objects SET status = 'done', finished_at = ?, wall_time_s = ?, device = ?, "
                         "n_params = ?, error = NULL, code_version = COALESCE(?, code_version) WHERE object_id = ?",
                         (now, float(result.wall_time_s), result.device, int(result.n_params), code_version,
                          result.object_id))
            conn.execute(f"INSERT OR REPLACE INTO summaries (object_id, json, {', '.join(SUMMARY_COLUMNS)}) "
                         f"VALUES (?, ?, {', '.join('?' * len(SUMMARY_COLUMNS))})",
                         (result.object_id, _dumps(summary), *_summary_columns(summary)))
            conn.execute("INSERT OR REPLACE INTO lightcurves (object_id, path, n_points) VALUES (?, ?, ?)",
                         (result.object_id, self.store_path(lc_path), n_points))

    def mark_failed(self, object_id: str, error: str, *, worker_id: str | None = None) -> bool:
        """Mark an object failed with an error message (truncated to :data:`MAX_TEXT_CHARS`).

        With ``worker_id``, only if the object is still assigned to that worker (a worker that lost its claim must
        not fail an object another worker is running). Returns whether the row changed.
        """
        guard, params = ("", ()) if worker_id is None else (" AND worker = ?", (worker_id,))
        with self._write() as conn:
            self._require_object(conn, object_id)
            cur = conn.execute(f"UPDATE objects SET status = 'failed', finished_at = ?, error = ? "
                               f"WHERE object_id = ?{guard}", (self.now(), _truncate(str(error)), object_id, *params))
        return cur.rowcount > 0

    def mark_interrupted(self, object_id: str, *, worker_id: str | None = None) -> bool:
        """Mark a running (or requeued) object interrupted after a graceful stop: it stays claimable, resumes from
        its checkpoint, and its attempt is refunded. Done and failed objects are left alone; with ``worker_id``,
        only an object still assigned to that worker changes. Returns whether the row changed."""
        guard, params = ("", ()) if worker_id is None else (" AND worker = ?", (worker_id,))
        with self._write() as conn:
            self._require_object(conn, object_id)
            cur = conn.execute(f"UPDATE objects SET status = 'interrupted', attempts = MAX(attempts - 1, 0) "
                               f"WHERE object_id = ? AND status IN ('running', 'queued'){guard}",
                               (object_id, *params))
        if cur.rowcount == 0:
            logger.warning("mark_interrupted(%s): object is not running%s; left unchanged", object_id,
                           "" if worker_id is None else f" under worker {worker_id!r}")
        return cur.rowcount > 0

    def get_spec(self, object_id: str) -> ObjectSpec:
        """The stored :class:`ObjectSpec` of an object (KeyError if unknown)."""
        with self._read() as conn:
            row = conn.execute("SELECT spec_json FROM objects WHERE object_id = ?", (object_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown object {object_id!r}")
        return ObjectSpec.from_json(row["spec_json"])

    def get_object(self, object_id: str) -> dict[str, Any]:
        """Everything about one object (KeyError if unknown): the ``objects`` columns plus ``spec`` (dict),
        ``summary`` (dict or None), ``lightcurve`` ({path (absolute), n_points} or None), ``classification`` (latest,
        see :meth:`latest_classification`), ``followup`` (row or None), ``ri_runs`` (rows, absolute paths),
        ``n_ri_events`` and ``transients`` (rows)."""
        with self._read() as conn:
            row = conn.execute("SELECT * FROM objects WHERE object_id = ?", (object_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown object {object_id!r}")
            out = dict(row)
            out["spec"] = json.loads(out["spec_json"])
            summ = conn.execute("SELECT json FROM summaries WHERE object_id = ?", (object_id,)).fetchone()
            out["summary"] = None if summ is None else json.loads(summ["json"])
            lc = conn.execute("SELECT path, n_points FROM lightcurves WHERE object_id = ?", (object_id,)).fetchone()
            out["lightcurve"] = None if lc is None else {"path": str(self.resolve_path(lc["path"])),
                                                         "n_points": lc["n_points"]}
            out["classification"] = self._latest_classification(conn, object_id)
            fu = conn.execute("SELECT * FROM followups WHERE object_id = ?", (object_id,)).fetchone()
            out["followup"] = None if fu is None else dict(fu)
            runs = [dict(r) for r in conn.execute("SELECT * FROM ri_runs WHERE object_id = ? ORDER BY ri_run_id",
                                                  (object_id,))]
            for r in runs:
                r["path"] = None if r["path"] is None else str(self.resolve_path(r["path"]))
            out["ri_runs"] = runs
            out["n_ri_events"] = conn.execute("SELECT COUNT(*) FROM ri_events WHERE object_id = ?",
                                              (object_id,)).fetchone()[0]
            out["transients"] = [dict(r) for r in conn.execute(
                "SELECT * FROM transients WHERE object_id = ? ORDER BY transient_id", (object_id,))]
        return out

    def queue_size(self) -> int:
        """Number of claimable objects (queued or interrupted)."""
        with self._read() as conn:
            return conn.execute("SELECT COUNT(*) FROM objects WHERE status IN ('queued', 'interrupted')").fetchone()[0]

    def object_ids(self, region: str | None = None) -> set[str]:
        """Ids of all objects in the catalog (of one region if given), whatever their status."""
        with self._read() as conn:
            if region is None:
                rows = conn.execute("SELECT object_id FROM objects")
            else:
                rows = conn.execute("SELECT object_id FROM objects WHERE region = ?", (region,))
            return {r[0] for r in rows}

    def region_objects(self, region: str) -> list[dict[str, Any]]:
        """All objects of a region in insertion order, with what the scheduler needs to aggregate them by point:
        object_id, seed, status, sky_version, spec_json, source, random_control, created_at, finished_at,
        t_test_acc90, final_test_acc, diverged (summary columns; None without a summary), summary_json (the full
        summary as JSON text, None without one), label (latest
        classification or None), has_followup (0/1), n_transients, max_severity and transient_kinds (sorted list)."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT o.object_id, o.seed, o.status, o.sky_version, o.spec_json, o.source, o.random_control, "
                "o.created_at, o.finished_at, s.t_test_acc90, s.final_test_acc, s.diverged, s.json AS summary_json, "
                "(SELECT c.label FROM classifications c WHERE c.object_id = o.object_id ORDER BY c.id DESC LIMIT 1) "
                "AS label, EXISTS (SELECT 1 FROM followups f WHERE f.object_id = o.object_id) AS has_followup, "
                "(SELECT COUNT(*) FROM transients t WHERE t.object_id = o.object_id) AS n_transients, "
                "(SELECT MAX(t.severity) FROM transients t WHERE t.object_id = o.object_id) AS max_severity, "
                "(SELECT GROUP_CONCAT(DISTINCT t.kind) FROM transients t WHERE t.object_id = o.object_id) "
                "AS transient_kinds "
                "FROM objects o LEFT JOIN summaries s ON s.object_id = o.object_id "
                "WHERE o.region = ? ORDER BY o.rowid", (region,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["transient_kinds"] = sorted(d["transient_kinds"].split(",")) if d["transient_kinds"] else []
            out.append(d)
        return out

    def _require_object(self, conn: sqlite3.Connection, object_id: str) -> None:
        if conn.execute("SELECT 1 FROM objects WHERE object_id = ?", (object_id,)).fetchone() is None:
            raise KeyError(f"unknown object {object_id!r}")

    # ------------------------------------------------------------------ classifications
    def add_classification(self, object_id: str, cls: "Classification | Mapping[str, Any]", classifier: str,
                           note: str = "") -> int:
        """Record a classification (a :class:`Classification` or a mapping with label, tags, score, reasons).

        When ``note`` is empty the classifier's reasons are stored as the note (one per line), so the explanation of
        every label stays in the catalog. Returns the row id; the row with the highest id is the latest.
        """
        label = _field(cls, "label")
        if not label or not isinstance(label, str):
            raise ValueError(f"classification needs a non-empty string label, got {label!r}")
        tags = list(_field(cls, "tags") or [])
        score = _finite_or_none(_field(cls, "score"))
        if not note:
            note = "\n".join(str(r) for r in (_field(cls, "reasons") or []))
        with self._write() as conn:
            self._require_object(conn, object_id)
            cur = conn.execute("INSERT INTO classifications (object_id, label, tags_json, score, classifier, note, "
                               "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (object_id, label, _dumps(tags), score, classifier, note, self.now()))
            return int(cur.lastrowid)

    def latest_classification(self, object_id: str) -> dict[str, Any] | None:
        """The most recent classification of an object (by insertion order) as a dict with id, label, tags (list),
        score, classifier, note and created_at; None if it has none."""
        with self._read() as conn:
            return self._latest_classification(conn, object_id)

    @staticmethod
    def _latest_classification(conn: sqlite3.Connection, object_id: str) -> dict[str, Any] | None:
        row = conn.execute("SELECT id, label, tags_json, score, classifier, note, created_at FROM classifications "
                           "WHERE object_id = ? ORDER BY id DESC LIMIT 1", (object_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["tags"] = json.loads(out.pop("tags_json") or "[]")
        return out

    # ------------------------------------------------------------------ follow-ups
    def request_followup(self, object_id: str, reason: str, priority: float = 1.0,
                         requested_by: str = "scheduler") -> bool:
        """Request RI follow-up of an object. Returns False if the object already has a follow-up row (in any
        status): an object is never followed up twice. KeyError if the object is unknown."""
        with self._write() as conn:
            self._require_object(conn, object_id)
            cur = conn.execute("INSERT OR IGNORE INTO followups (object_id, reason, priority, status, requested_by, "
                               "created_at, attempts) VALUES (?, ?, ?, 'requested', ?, ?, 0)",
                               (object_id, reason, float(priority), requested_by, self.now()))
            return cur.rowcount > 0

    def claim_followup(self, worker_id: str) -> str | None:
        """Claim the highest-priority, oldest requested follow-up whose object is done; returns its object id."""
        now = self.now()
        with self._write() as conn:
            row = conn.execute(
                "SELECT f.object_id FROM followups f JOIN objects o ON o.object_id = f.object_id "
                "WHERE f.status = 'requested' AND o.status = 'done' "
                "ORDER BY f.priority DESC, f.created_at ASC, f.rowid ASC LIMIT 1").fetchone()
            if row is None:
                return None
            conn.execute("UPDATE followups SET status = 'running', worker = ?, started_at = ?, "
                         "attempts = attempts + 1 WHERE object_id = ?", (worker_id, now, row["object_id"]))
            self._upsert_heartbeat(conn, worker_id, row["object_id"], "followup", now)
            return str(row["object_id"])

    def finish_followup(self, object_id: str, ok: bool, error: str | None = None) -> None:
        """Mark a follow-up done (``ok``) or failed (with ``error``). KeyError if there is no such follow-up."""
        with self._write() as conn:
            cur = conn.execute("UPDATE followups SET status = ?, finished_at = ?, error = ? WHERE object_id = ?",
                               ("done" if ok else "failed", self.now(),
                                None if error is None else _truncate(str(error)), object_id))
            if cur.rowcount == 0:
                raise KeyError(f"no follow-up for object {object_id!r}")

    def count_followups(self, *, requested_by: str | None = None, since: float | None = None) -> int:
        """Number of follow-up rows (optionally only those requested by ``requested_by`` at or after ``since``)."""
        sql, params = "SELECT COUNT(*) FROM followups WHERE 1 = 1", []
        if requested_by is not None:
            sql += " AND requested_by = ?"
            params.append(requested_by)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(float(since))
        with self._read() as conn:
            return conn.execute(sql, params).fetchone()[0]

    # ------------------------------------------------------------------ RI spectroscopy
    def add_ri_run(self, object_id: str, kind: str, path: Path | str, curve: "RICurve") -> int:
        """Record an RI curve file (kind e.g. "insitu" or "followup"; path stored root-relative)."""
        meta = curve.meta or {}
        max_order = meta.get("max_order")
        if max_order is None and curve.S:
            max_order = int(curve.orders.max())
        with self._write() as conn:
            self._require_object(conn, object_id)
            cur = conn.execute(
                "INSERT INTO ri_runs (object_id, kind, path, n_subsets, n_points, max_order, alpha, min_effect, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (object_id, kind, self.store_path(path), int(curve.S), int(curve.T),
                 None if max_order is None else int(max_order), _finite_or_none(meta.get("alpha")),
                 _finite_or_none(meta.get("min_effect")), self.now()))
            return int(cur.lastrowid)

    def add_events(self, ri_run_id: int, object_id: str, events: Sequence["RIEvent"]) -> list[int]:
        """Record RI events of one RI run; returns their event ids in the order of ``events``.

        ``subset_json`` holds the JSON list of the subset's variable names; ``has_precursor`` / ``matches_gt`` are
        1, 0 or NULL.
        """
        now = self.now()
        with self._write() as conn:
            run = conn.execute("SELECT object_id FROM ri_runs WHERE ri_run_id = ?", (int(ri_run_id),)).fetchone()
            if run is None:
                raise KeyError(f"unknown ri_run_id {ri_run_id}")
            if run["object_id"] != object_id:
                raise ValueError(f"ri_run {ri_run_id} belongs to object {run['object_id']}, not {object_id}")
            ids = []
            for ev in events:
                cur = conn.execute(
                    "INSERT INTO ri_events (ri_run_id, object_id, subset_json, order_k, kind, step, step_lo, step_hi, "
                    "ri_before, ri_after, peak_ri, confidence, has_precursor, matches_gt, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (int(ri_run_id), object_id, _dumps(list(ev.subset_names)), int(ev.order), ev.kind, int(ev.step),
                     int(ev.step_lo), int(ev.step_hi), _finite_or_none(ev.ri_before), _finite_or_none(ev.ri_after),
                     _finite_or_none(ev.peak_ri), _finite_or_none(ev.confidence), _flag_or_none(ev.has_precursor),
                     _flag_or_none(ev.matches_ground_truth), now))
                ids.append(int(cur.lastrowid))
        return ids

    def add_transients(self, object_id: str, transients: Sequence["Transient | Mapping[str, Any]"],
                       event_ids: Sequence[int]) -> list[int]:
        """Record transients of one object. ``transient.event_index`` indexes the event list whose ids
        :meth:`add_events` returned (``event_ids``); None means the transient is not tied to an event."""
        now = self.now()
        rows = []
        for tr in transients:
            idx = _field(tr, "event_index")
            if idx is None:
                event_id = None
            elif 0 <= int(idx) < len(event_ids):
                event_id = int(event_ids[int(idx)])
            else:
                raise ValueError(f"transient event_index {idx} out of range for {len(event_ids)} event ids")
            rows.append((object_id, event_id, _field(tr, "kind"), int(_field(tr, "severity")),
                         str(_field(tr, "message")), now))
        with self._write() as conn:
            self._require_object(conn, object_id)
            ids = []
            for row in rows:
                cur = conn.execute("INSERT INTO transients (object_id, event_id, kind, severity, message, created_at, "
                                   "notified, acknowledged) VALUES (?, ?, ?, ?, ?, ?, 0, 0)", row)
                ids.append(int(cur.lastrowid))
        return ids

    def pending_notifications(self, *, min_severity: int = 1, limit: int | None = None) -> list[dict[str, Any]]:
        """Transients not yet notified (severity ≥ ``min_severity``), most severe first, then oldest; each with the
        object's task, region and seed."""
        sql = ("SELECT t.transient_id, t.object_id, t.event_id, t.kind, t.severity, t.message, t.created_at, "
               "o.task, o.region, o.seed FROM transients t JOIN objects o ON o.object_id = t.object_id "
               "WHERE t.notified = 0 AND t.severity >= ? ORDER BY t.severity DESC, t.created_at, t.transient_id")
        params: list[Any] = [int(min_severity)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._read() as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def mark_notified(self, ids: Iterable[int]) -> int:
        """Mark transients as notified; returns the number of rows changed."""
        rows = [(int(i),) for i in ids]
        if not rows:
            return 0
        with self._write() as conn:
            cur = conn.executemany("UPDATE transients SET notified = 1 WHERE transient_id = ? AND notified = 0", rows)
            return max(cur.rowcount, 0)

    # ------------------------------------------------------------------ proposals, log, agent audit
    def add_proposal(self, source: str, region: str, point: Mapping[str, Any], *, task: str | None = None,
                     seeds: int = 1, rationale: str = "", status: str = "proposed", n_enqueued: int = 0,
                     decided_at: float | None = None) -> int:
        """Record a proposed configuration point (``point_json`` is canonical JSON: sorted keys, integral floats
        as ints, the same normalisation as ``tycho.survey.sky.point_key``). Conventional statuses: proposed,
        enqueued, rejected. Returns the proposal id."""
        with self._write() as conn:
            cur = conn.execute(
                "INSERT INTO proposals (source, region, task, point_json, seeds, rationale, status, created_at, "
                "decided_at, n_enqueued) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source, region, task, _dumps(_canonical(dict(point))), int(seeds), rationale, status, self.now(),
                 decided_at, int(n_enqueued)))
            return int(cur.lastrowid)

    def add_log(self, author: str, kind: str, title: str, markdown: str) -> int:
        """Append a survey-log entry (e.g. the agent's nightly log); returns its id."""
        with self._write() as conn:
            cur = conn.execute("INSERT INTO survey_log (created_at, author, kind, title, markdown) "
                               "VALUES (?, ?, ?, ?, ?)", (self.now(), author, kind, title, markdown))
            return int(cur.lastrowid)

    def start_agent_run(self, kind: str, model: str) -> int:
        """Open an agent-run audit row (status "running"); returns its id."""
        with self._write() as conn:
            cur = conn.execute("INSERT INTO agent_runs (kind, started_at, model, input_tokens, output_tokens, "
                               "cache_read_tokens, cache_write_tokens, est_cost_usd, n_tool_calls, status) "
                               "VALUES (?, ?, ?, 0, 0, 0, 0, 0.0, 0, 'running')", (kind, self.now(), model))
            return int(cur.lastrowid)

    def finish_agent_run(self, run_id: int, **fields: Any) -> None:
        """Update an agent run (only :data:`AGENT_RUN_FIELDS`; ``finished_at`` defaults to now)."""
        unknown = set(fields) - AGENT_RUN_FIELDS
        if unknown:
            raise ValueError(f"unknown agent_runs fields: {sorted(unknown)}")
        fields.setdefault("finished_at", self.now())
        if "error" in fields and fields["error"] is not None:
            fields["error"] = _truncate(str(fields["error"]))
        cols = sorted(fields)
        with self._write() as conn:
            cur = conn.execute(f"UPDATE agent_runs SET {', '.join(f'{c} = ?' for c in cols)} WHERE run_id = ?",
                               (*[fields[c] for c in cols], int(run_id)))
            if cur.rowcount == 0:
                raise KeyError(f"unknown agent run {run_id}")

    def log_tool_call(self, run_id: int, tool: str, input_json: str | Mapping[str, Any], output_preview: str,
                      is_error: bool) -> int:
        """Audit one agent tool call (input as JSON text; the preview is truncated to :data:`MAX_TEXT_CHARS`)."""
        text = input_json if isinstance(input_json, str) else _dumps(input_json)
        row = (int(run_id), self.now(), tool, text, _truncate(str(output_preview)), int(bool(is_error)))
        with self._write() as conn:
            cur = conn.execute("INSERT INTO agent_tool_calls (run_id, created_at, tool, input_json, output_preview, "
                               "is_error) VALUES (?, ?, ?, ?, ?, ?)", row)
            return int(cur.lastrowid)

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: str) -> None:
        """Set a ``meta`` value (text)."""
        if not isinstance(value, str):
            raise TypeError(f"meta values are text; got {type(value).__name__} (json.dumps it first)")
        with self._write() as conn:
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        """A ``meta`` value, or ``default`` if the key is absent."""
        with self._read() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else row["value"]

    def meta_items(self, prefix: str = "") -> dict[str, str]:
        """All ``meta`` entries whose key starts with ``prefix`` (exact prefix match, no wildcards)."""
        with self._read() as conn:
            rows = conn.execute("SELECT key, value FROM meta WHERE substr(key, 1, ?) = ? ORDER BY key",
                                (len(prefix), prefix)).fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict[str, Any]:
        """Counts for status pages and the agent (one consistent snapshot). Every status key is always present.

        Keys: objects {status: n, "total"}, queue (queued + interrupted), by_region {region: {status: n}},
        followups {status: n}, labels {label: n} (latest classification per object), ri_runs, ri_events,
        transients {total, pending_notification, unacknowledged}, proposals {status: n}, done_last_24h,
        failed_last_24h, wall_time_s_done (total compute of done objects), workers (heartbeat rows with age_s),
        schema_version, generated_at.
        """
        now = self.now()
        with self._read() as conn:
            objects = dict.fromkeys(OBJECT_STATUSES, 0)
            objects.update({r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM objects GROUP BY status")})
            by_region: dict[str, dict[str, int]] = {}
            for region, status, n in conn.execute("SELECT region, status, COUNT(*) FROM objects "
                                                  "GROUP BY region, status ORDER BY region"):
                by_region.setdefault(region, dict.fromkeys(OBJECT_STATUSES, 0))[status] = n
            followups = dict.fromkeys(FOLLOWUP_STATUSES, 0)
            followups.update({r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM followups "
                                                               "GROUP BY status")})
            labels = {r[0]: r[1] for r in conn.execute(
                "SELECT c.label, COUNT(*) FROM classifications c JOIN (SELECT MAX(id) AS id FROM classifications "
                "GROUP BY object_id) m ON m.id = c.id GROUP BY c.label ORDER BY c.label")}
            tr = conn.execute("SELECT COUNT(*), COALESCE(SUM(notified = 0), 0), COALESCE(SUM(acknowledged = 0), 0) "
                              "FROM transients").fetchone()
            recent = conn.execute("SELECT COALESCE(SUM(status = 'done'), 0), COALESCE(SUM(status = 'failed'), 0) "
                                  "FROM objects WHERE finished_at >= ?", (now - _SECONDS_PER_DAY,)).fetchone()
            out = {
                "objects": {**objects, "total": sum(objects.values())},
                "queue": objects["queued"] + objects["interrupted"],
                "by_region": by_region,
                "followups": followups,
                "labels": labels,
                "ri_runs": conn.execute("SELECT COUNT(*) FROM ri_runs").fetchone()[0],
                "ri_events": conn.execute("SELECT COUNT(*) FROM ri_events").fetchone()[0],
                "transients": {"total": tr[0], "pending_notification": tr[1], "unacknowledged": tr[2]},
                "proposals": {r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM proposals "
                                                               "GROUP BY status ORDER BY status")},
                "done_last_24h": recent[0],
                "failed_last_24h": recent[1],
                "wall_time_s_done": conn.execute("SELECT COALESCE(SUM(wall_time_s), 0.0) FROM objects "
                                                 "WHERE status = 'done'").fetchone()[0],
                "workers": [{**dict(r), "age_s": now - r["updated_at"]} for r in conn.execute(
                    "SELECT worker_id, pid, host, object_id, status, updated_at FROM heartbeats ORDER BY worker_id")],
                "schema_version": SCHEMA_VERSION,
                "generated_at": now,
            }
        return out

    def query(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = (), *, max_rows: int | None = None,
              timeout_s: float | None = None) -> list[dict[str, Any]]:
        """Run one read-only SELECT/WITH statement and return its rows as dicts.

        Raises ValueError for anything else (other statements, several statements, or a statement that would write
        or attach: see the module docstring), TimeoutError when ``timeout_s`` elapses. ``max_rows`` caps the result.
        """
        _check_select_only(sql)
        with self._lock:
            conn = self._connection()
            conn.set_authorizer(_readonly_authorizer)
            if timeout_s is not None:
                deadline = time.monotonic() + float(timeout_s)
                conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            cur = None
            try:
                cur = conn.execute(sql, params)
                rows = cur.fetchall() if max_rows is None else cur.fetchmany(int(max_rows))
                cols = [d[0] for d in cur.description or ()]
            except sqlite3.DatabaseError as exc:
                if "not authorized" in str(exc):
                    raise ValueError(f"query is not read-only: {exc}") from exc
                if timeout_s is not None and "interrupted" in str(exc):
                    raise TimeoutError(f"query exceeded {timeout_s} s") from exc
                raise
            finally:
                if cur is not None:
                    cur.close()
                conn.set_authorizer(None)
                conn.set_progress_handler(None, 0)
        return [dict(zip(cols, r)) for r in rows]

    def backup(self, dest: Path | str) -> Path:
        """Consistent online copy of the catalog to ``dest`` via the SQLite backup API.

        The copy is written next to ``dest`` first, switched to a self-contained rollback journal (no -wal file),
        checked with ``PRAGMA quick_check`` and then atomically renamed, so ``dest`` is never a partial backup.
        """
        dest = Path(dest).expanduser().resolve()
        if dest == self.path:
            raise ValueError("backup destination is the catalog itself")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        _remove_database_files(tmp)  # leftovers of an interrupted backup
        target = sqlite3.connect(str(tmp), isolation_level=None)
        try:
            with self._lock:
                self._connection().backup(target)
            target.execute("PRAGMA journal_mode = DELETE")
            check = target.execute("PRAGMA quick_check").fetchone()[0]
        except BaseException:
            target.close()
            _remove_database_files(tmp)
            raise
        target.close()
        if check != "ok":
            _remove_database_files(tmp)
            raise RuntimeError(f"backup of {self.path} failed quick_check: {check}")
        os.replace(tmp, dest)
        logger.info("catalog backed up to %s", dest)
        return dest


# ---------------------------------------------------------------------------------------------- read-only queries
_READ_ACTIONS = frozenset({sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE})
_SCHEMA_PRAGMAS = frozenset({"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo",
                             "foreign_key_list"})
"""Introspection pragmas allowed as table-valued functions inside a query (e.g. ``pragma_table_info('objects')``)."""
_DENIED_FUNCTIONS = frozenset({"load_extension"})


def _readonly_authorizer(action: int, arg1: str | None, arg2: str | None, db_name: str | None,
                         trigger: str | None) -> int:
    """SQLite authorizer of :meth:`Catalog.query`: allow reads only."""
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (arg2 or "").lower() in _DENIED_FUNCTIONS else sqlite3.SQLITE_OK
    if action in _READ_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in _SCHEMA_PRAGMAS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _check_select_only(sql: str) -> None:
    """Lexical gate of :meth:`Catalog.query`: one statement whose first keyword is SELECT or WITH.

    Whitespace, ``--`` and ``/* */`` comments, string literals and quoted identifiers are skipped, so semicolons and
    keywords inside them do not count; a trailing semicolon is fine.
    """
    if not isinstance(sql, str):
        raise ValueError("query must be a string")
    i, n = 0, len(sql)
    first_word: str | None = None
    after_semicolon = False
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j + 1
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif after_semicolon:
            raise ValueError("query must be a single statement")
        elif ch == ";":
            if first_word is None:
                raise ValueError("empty statement")
            after_semicolon = True
            i += 1
        elif first_word is None:
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            first_word = sql[i:j].upper()
            if first_word not in ("SELECT", "WITH"):
                raise ValueError(f"only SELECT/WITH statements are allowed, got {first_word or sql[i]!r}")
            i = j
        elif ch in "'\"`":
            i = _skip_quoted(sql, i, ch)
        elif ch == "[":
            j = sql.find("]", i + 1)
            i = n if j < 0 else j + 1
        else:
            i += 1
    if first_word is None:
        raise ValueError("empty query")


def _skip_quoted(sql: str, start: int, quote: str) -> int:
    """Index just after the literal opened at ``start`` (a doubled quote is an escaped quote)."""
    j, n = start + 1, len(sql)
    while j < n:
        if sql[j] == quote:
            if j + 1 < n and sql[j + 1] == quote:
                j += 2
                continue
            return j + 1
        j += 1
    return n


# ---------------------------------------------------------------------------------------------- helpers
def _remove_database_files(path: Path) -> None:
    """Delete a database file and its WAL / shared-memory companions (if present)."""
    for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        p.unlink(missing_ok=True)


def _infer_root(db_path: Path) -> Path:
    """Survey root of a database at ``<root>/data/<file>`` (the ``Paths`` layout), else the database's directory."""
    parent = db_path.parent
    return parent.parent if parent.name == "data" else parent


def _unit_hash(salt: str, key: str) -> float:
    """Deterministic uniform number in [0, 1) from sha256(salt, key)."""
    digest = hashlib.sha256(f"{salt}\x1f{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def _jsonable(o: Any) -> Any:
    """Strict-JSON form of ``o``: NaN/±inf → None; numpy scalars/arrays → numbers/lists; tuples, sets → lists."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Mapping):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (set, frozenset)):
        return [_jsonable(v) for v in sorted(o, key=repr)]
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if is_dataclass(o) and not isinstance(o, type):
        return _jsonable(asdict(o))
    if isinstance(o, PurePath):
        return o.as_posix()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _dumps(o: Any) -> str:
    """Compact strict JSON with sorted keys."""
    return json.dumps(_jsonable(o), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_or_none(v: Any) -> float | None:
    if v is None or isinstance(v, (str, bytes)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _flag_or_none(v: Any) -> int | None:
    """1/0 for a boolean-like value; None for None and NaN."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
        return None
    return int(bool(v))


def _summary_columns(summary: Mapping[str, Any]) -> list[Any]:
    out: list[Any] = []
    for key in SUMMARY_COLUMNS:
        v = summary.get(key)
        if key == "diverged":
            out.append(_flag_or_none(v))
        elif key == "n_drops":
            f = _finite_or_none(v)
            out.append(None if f is None else int(f))
        else:
            out.append(_finite_or_none(v))
    return out


def _field(obj: Any, name: str) -> Any:
    """Attribute ``name`` of a dataclass-like object or key ``name`` of a mapping (None if absent)."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """Keep the head and the tail (where tracebacks end) of an over-long text."""
    if len(text) <= limit:
        return text
    half = (limit - 40) // 2
    return f"{text[:half]}\n... [{len(text) - 2 * half} characters truncated] ...\n{text[-half:]}"
