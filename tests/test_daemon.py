"""Tests of tycho.survey.daemon: the supervisor's one-iteration step (status, scheduling, stale jobs, worker
restarts, notifications, disk guard, backups, stop conditions) with fake processes and a fake notifier; a smoke run
of the real daemon (2 spawned workers on a tiny sky); and a SIGTERM test of a daemon subprocess whose interrupted
objects resume exactly on the next run."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pytest
import torch
from survey_env import make_root, synthetic_catalog

import tycho.survey.daemon as daemon_mod
from tycho.survey.catalog import Catalog
from tycho.survey.daemon import (CRASH_ALERT_RESTARTS, NOTIFY_RETRY_S, RESTART_BACKOFF_S, STATUS_FILE,
                                 TASK_ALERT_FAILURES, Supervisor, SurveyLock, SurveyLockError, list_backups,
                                 resolve_n_workers, run_survey, running_daemon_pid)
from tycho.train.runner import RESUME_FILE, run_object

T0 = 1_800_000_000.0


@pytest.fixture(autouse=True, scope="module")
def _torch_settings():
    """The resume check trains in this process with one thread, as the workers do; restored afterwards."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class FakeNotifier:
    """Stands in for Notifier: records what the supervisor sends."""

    channels = ["fake"]
    held = 0

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[tuple] = []
        self.recorded: list[str] = []

    def send(self, title, message, priority="default", tags=()) -> bool:
        self.sent.append((title, message, priority, tuple(tags)))
        return self.ok

    def record(self, title, message, priority="low", tags=()) -> None:
        self.recorded.append(title)

    def flush_digest(self) -> bool:
        return False

    def titles(self, prefix: str) -> list[str]:
        return [s[0] for s in self.sent if s[0].startswith(prefix)]


class FakeProc:
    """Stands in for a worker process."""

    def __init__(self, pid: int) -> None:
        self.pid, self.alive, self.exitcode, self.killed = pid, True, None, False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout=None) -> None:
        return None

    def die(self, code: int = 1) -> None:
        self.alive, self.exitcode = False, code

    def kill(self) -> None:
        self.killed = True
        self.die(-9)

    def terminate(self) -> None:
        self.die(-15)


@pytest.fixture
def sup_factory(tmp_path):
    made = []

    def make(config=None, *, synthetic=False, **kwargs) -> Supervisor:
        paths = make_root(tmp_path / f"root{len(made)}", config=config)
        if synthetic:
            synthetic_catalog(paths)[0].close()
        kwargs.setdefault("n_workers", 0)
        kwargs.setdefault("notifier", FakeNotifier())
        kwargs.setdefault("clock", lambda: T0)
        sup = Supervisor(paths.root, **kwargs)
        made.append(sup)
        return sup

    yield make
    for sup in made:
        sup.shutdown(timeout_s=0)


def fake_starts(sup: Supervisor) -> list[FakeProc]:
    """Replace process creation by FakeProc objects; returns the list of started fakes."""
    started: list[FakeProc] = []

    def start(slot, now):
        slot.process, slot.started_at = FakeProc(10_000 + len(started)), now
        started.append(slot.process)
        sup.counters["worker_starts"] += 1

    sup._start_worker = start
    return started


def read_status(sup: Supervisor) -> dict:
    return json.loads((sup.paths.reports / STATUS_FILE).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- one iteration
def test_step_tops_up_writes_status_and_backs_up(sup_factory):
    sup = sup_factory()
    sup.start()
    assert read_status(sup)["state"] == "running"
    sup.step(T0)
    assert sup.catalog.queue_size() == 8 and sup.counters["enqueued"] == 8  # target_queue of the tiny config
    assert len(list_backups(sup.paths)) == 1 and sup.counters["backups"] == 1
    sup.step(T0 + 2)  # status_every_s = 1
    status = read_status(sup)
    assert status["objects"]["queued"] == 8 and status["daemon"]["n_workers"] == 0
    assert status["daemon"]["counters"]["enqueued"] == 8 and status["daemon"]["sky_version"] == "tiny-v0"
    assert (sup.paths.reports / "dashboard.html").is_file()
    summary = sup.shutdown(timeout_s=0)
    assert summary["stop_reason"] == "shutdown" and read_status(sup)["state"] == "stopped"


def test_stale_jobs_are_requeued_but_not_right_after_a_sleep(sup_factory):
    sup = sup_factory({"survey": {"stale_after_s": 0.3, "heartbeat_s": 0.5}})
    sup.step(T0)  # enqueues
    with Catalog(sup.paths.db, root=sup.paths.root) as other:
        ghost = other.claim_next("ghost").object_id()  # a worker that dies right after claiming
    time.sleep(0.4)
    sup.step(T0 + 100)  # 100 s after the previous step: the machine slept, heartbeats are given time to resume
    assert sup.catalog.get_object(ghost)["status"] == "running"
    sup.step(T0 + 101.5)
    obj = sup.catalog.get_object(ghost)
    assert obj["status"] == "queued" and obj["attempts"] == 1 and sup.counters["requeued"] == 1


def test_dead_workers_restart_with_backoff_and_hung_workers_are_killed(sup_factory):
    sup = sup_factory(n_workers=2)
    started = fake_starts(sup)
    sup.step(T0)
    assert len(started) == 2
    started[0].die()
    sup.step(T0 + 1)  # reaped; it ran 1 s only: restart after the backoff
    assert sup.slots[0].process is None and len(started) == 2
    sup.step(T0 + 1 + RESTART_BACKOFF_S - 0.5)
    assert len(started) == 2
    sup.step(T0 + 1 + RESTART_BACKOFF_S + 0.5)
    assert len(started) == 3 and sup.slots[0].backoff == 2 * RESTART_BACKOFF_S
    # w1 never wrote a heartbeat: once it is older than stale_after_s (60 s) it counts as hung and is killed
    sup.step(T0 + 61 + RESTART_BACKOFF_S)
    assert started[1].killed and not started[2].killed  # w0's new incarnation is too young to judge


def test_a_crash_looping_worker_raises_one_alert(sup_factory):
    sup = sup_factory(n_workers=1)
    started = fake_starts(sup)
    now = T0
    sup.step(now)
    for _ in range(CRASH_ALERT_RESTARTS + 2):
        started[-1].die()
        now += 1
        sup.step(now)  # reap
        now = sup.slots[0].next_start + 0.1
        sup.step(now)  # restart
    assert sup.notifier.titles("Tycho: worker w0 keeps dying") == ["Tycho: worker w0 keeps dying"]


def test_notifications_respect_min_severity_and_retry(sup_factory):
    sup = sup_factory(synthetic=True, notifier=FakeNotifier(ok=False))
    sup.step(T0)
    assert len(sup.notifier.sent) == 1 and "pure synergy" in sup.notifier.sent[0][0]  # severity 3 is pushed first
    assert len(sup.catalog.pending_notifications()) == 2  # every channel failed: nothing marked notified
    sup.step(T0 + 10)
    assert len(sup.notifier.sent) == 1  # waiting for the retry
    sup.notifier.ok = True
    sup.step(T0 + NOTIFY_RETRY_S + 1)
    assert len(sup.notifier.titles("Tycho transient: pure synergy")) == 2
    assert sup.notifier.recorded == ["Tycho transient: loss spike (severity 1)"]  # below min_severity: logged only
    assert sup.catalog.pending_notifications() == [] and sup.counters["notified"] == 2


def test_disk_guard_alerts_once_and_rearms(sup_factory, monkeypatch):
    sup = sup_factory({"survey": {"min_free_gb": 10.0}})
    free = {"gb": 5.0}
    monkeypatch.setattr(daemon_mod, "free_disk_gb", lambda path: free["gb"])
    for i, gb in enumerate((5.0, 4.0, 10.5, 12.0, 3.0)):
        free["gb"] = gb
        sup.step(T0 + i)
    assert len(sup.notifier.titles("Tycho: disk space low")) == 2  # at 5 GB, and again after recovering to 12 GB
    assert sup.disk == {"free_gb": 3.0, "min_free_gb": 10.0, "low": True}


def test_a_failing_task_is_isolated_and_alerted_once(sup_factory, monkeypatch):
    sup = sup_factory()

    def broken():
        raise RuntimeError("scheduler bug")

    monkeypatch.setattr(sup.scheduler, "top_up", broken)
    for i in range(TASK_ALERT_FAILURES + 3):
        sup.step(T0 + 2 * i)  # topup_every_s = 1: due at every step; the step itself never raises
    assert sup.notifier.titles("Tycho: supervisor task 'topup' keeps failing") == [
        "Tycho: supervisor task 'topup' keeps failing"]
    assert read_status(sup)["objects"]["total"] == 0


def test_followups_are_requested_for_unknown_objects(sup_factory):
    sup = sup_factory(synthetic=True)
    grok_done = sup.catalog.query("SELECT object_id FROM objects WHERE region = 'grok' AND status = 'done'")
    oid = grok_done[0]["object_id"]
    sup.catalog.add_classification(oid, {"label": "unknown", "tags": [], "score": 0.25}, classifier="agent")
    sup.step(T0)
    fu = sup.catalog.get_object(oid)["followup"]
    assert fu["status"] == "requested" and fu["reason"].startswith("unknown_class")
    assert sup.counters["followups_requested"] == 1


def test_backups_follow_the_period_and_retention(sup_factory):
    sup = sup_factory({"survey": {"backup_every_h": 1.0, "backups_keep": 2}})
    for i in range(4):
        sup.step(T0 + i * 3601.0)
    backups = list_backups(sup.paths)
    assert len(backups) == 2 and sup.counters["backups"] == 4
    stamp = datetime.fromtimestamp(T0 + 3 * 3601.0, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    assert backups[-1].name == f"catalog-{stamp}.sqlite"  # named by the step's time (UTC)
    with Catalog(backups[-1]) as copy:
        assert copy.stats()["objects"]["total"] == 8


def test_stop_conditions(sup_factory):
    sup = sup_factory()
    sup.handle_signal(signal.SIGTERM)
    sup.step(T0)
    assert sup.stopping and sup.stop_event.is_set() and sup.stop_reason == "signal SIGTERM"
    timed = sup_factory(hours=1.0)
    timed.step(T0 + 3599)
    assert not timed.stopping
    timed.step(T0 + 3601)
    assert timed.stop_reason == "time limit reached"
    budget = sup_factory(max_objects=0)
    budget.step(T0)
    assert not budget.stopping  # waits one more step for claims in flight
    budget.step(T0 + 2)
    assert budget.stop_reason == "max_objects=0 observed"


def test_survey_lock_and_worker_count(tmp_path):
    paths = make_root(tmp_path)
    assert running_daemon_pid(paths) is None
    with SurveyLock(paths):
        assert running_daemon_pid(paths) == os.getpid()
        with pytest.raises(SurveyLockError, match="another survey daemon"):
            SurveyLock(paths).acquire()
    assert running_daemon_pid(paths) is None
    assert resolve_n_workers(3, {"survey": {"n_workers": 7}}) == 3
    assert resolve_n_workers(None, {"survey": {"n_workers": 7}}) == 7
    assert resolve_n_workers(None, {"survey": {"n_workers": None}}) == max(1, (os.cpu_count() or 1) - 2)
    with pytest.raises(ValueError):
        resolve_n_workers(-1, {})


# ---------------------------------------------------------------------------------------------- real daemon
def test_daemon_smoke_run(tmp_path):
    paths = make_root(tmp_path / "root")
    summary = run_survey(paths.root, n_workers=2, max_objects=6)
    assert summary["stop_reason"] == "max_objects=6 observed"
    stats = summary["stats"]
    assert stats["objects"]["done"] == 6 and stats["objects"]["running"] == 0 and stats["objects"]["failed"] == 0
    assert summary["counters"]["worker_starts"] == 2 and summary["counters"]["worker_restarts"] == 0
    assert stats["ri_runs"] == 4  # the ctrl region (ri_policy always, priority 2) is observed first, with RI
    assert stats["followups"]["running"] == 0 and stats["followups"]["failed"] == 0
    assert {w["worker_id"]: w["status"] for w in stats["workers"]} == {"w0": "stopped", "w1": "stopped"}
    status = json.loads((paths.reports / STATUS_FILE).read_text(encoding="utf-8"))
    assert status["state"] == "stopped" and status["objects"]["done"] == 6
    assert status["daemon"]["claims_left"] == 0 and status["daemon"]["code_version"] == "test"
    assert (paths.reports / "dashboard.html").is_file() and len(list_backups(paths)) == 1
    log = (paths.logs / "survey.log").read_text(encoding="utf-8")
    assert "[tycho-w0]" in log and "[tycho-w1]" in log and "survey daemon stopped" in log
    assert running_daemon_pid(paths) is None and mp.active_children() == []


SLOW_SKY = {
    "version": "slow-v0",
    "regions": [{
        "name": "slow", "task": "sparse_parity", "seeds": 2, "ri_policy": "never",
        "fixed": {"n": 8, "k": 2, "test_size": 256},
        "train": {"steps": 8000, "batch_size": 16, "optimizer": "sgd", "momentum": 0.0, "lr": 0.05,
                  "n_log_evals": 20, "n_lin_evals": 40, "ntk_every": 0, "checkpoint_every_s": 3600.0},
        "knobs": {"width": {"values": [32, 48], "kind": "log"}}}],
}


def test_sigterm_checkpoints_and_the_next_run_resumes(tmp_path):
    paths = make_root(tmp_path / "root", sky=SLOW_SKY)
    proc = subprocess.Popen([sys.executable, "-m", "tycho", "--root", str(paths.root), "survey", "run",
                             "--workers", "2"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 90
        while not paths.db.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        catalog = Catalog(paths.db, root=paths.root)
        mid_run = False
        while not mid_run and time.monotonic() < deadline and proc.poll() is None:
            beats = catalog.query("SELECT status FROM heartbeats WHERE status LIKE 'running %'")
            mid_run = any(2 <= float(b["status"].split()[1].rstrip("%")) <= 70 for b in beats)
            time.sleep(0.05)
        assert mid_run, "no worker was seen in the middle of an object"
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, out
    assert "signal SIGTERM" in out
    objects = catalog.query("SELECT object_id, status, attempts FROM objects")
    interrupted = [o["object_id"] for o in objects if o["status"] == "interrupted"]
    assert interrupted and not [o for o in objects if o["status"] == "running"]
    assert all(o["attempts"] == 0 for o in objects if o["status"] == "interrupted")  # graceful: attempt refunded
    for oid in interrupted:
        assert (paths.work_dir(oid) / RESUME_FILE).is_file()
    assert {w["status"] for w in catalog.stats()["workers"]} == {"stopped"}

    remaining = sum(o["status"] in ("queued", "interrupted") for o in objects)
    summary = run_survey(paths.root, n_workers=2, max_objects=remaining)
    assert summary["stats"]["objects"]["done"] == len(objects)
    log = (paths.logs / "survey.log").read_text(encoding="utf-8")
    assert log.count("resumed at step") >= len(interrupted)
    for oid in interrupted:
        assert not paths.work_dir(oid).exists()

    # the resumed light curve equals an uninterrupted observation, bit for bit
    spec = catalog.get_spec(interrupted[0])
    reference = run_object(spec, work_dir=tmp_path / "reference")
    with np.load(catalog.get_object(interrupted[0])["lightcurve"]["path"]) as lc:
        for key in ("step", "train_loss", "test_loss", "train_acc", "param_norm", "grad_norm"):
            assert np.array_equal(lc[key], reference.light_curve[key], equal_nan=True), key
    catalog.close()
