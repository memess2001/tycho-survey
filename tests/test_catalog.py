"""Tests of tycho.survey.catalog: schema, life cycles, concurrency (threads, fork, spawn), queries, backups."""
from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tycho.analysis.classify import Classification
from tycho.analysis.events import RIEvent, Transient
from tycho.analysis.ricurve import RICurve
from tycho.config import Paths
from tycho.survey.catalog import BUSY_TIMEOUT_MS, SCHEMA_VERSION, Catalog, CatalogSchemaError
from tycho.survey.sky import point_key
from tycho.train.spec import ObjectSpec, RunResult, TrainConfig


class FakeClock:
    """Settable clock (module level, so a catalog using it stays picklable)."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_specs(n: int, *, task: str = "toy", start: int = 0) -> list[ObjectSpec]:
    return [ObjectSpec(task=task, task_cfg={"width": 8}, seed=start + i, train=TrainConfig(steps=10),
                       sky_version="test") for i in range(n)]


def done_result(oid: str, summary: dict | None = None, n_points: int = 5) -> RunResult:
    return RunResult(object_id=oid, status="done", light_curve={"step": np.arange(n_points)},
                     summary=summary or {}, wall_time_s=1.5, device="cpu", n_params=123)


@pytest.fixture
def paths(tmp_path) -> Paths:
    return Paths(tmp_path / "root").ensure()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cat(paths, clock) -> Catalog:
    catalog = Catalog(paths.db, clock=clock)
    yield catalog
    catalog.close()


def raw(catalog: Catalog) -> sqlite3.Connection:
    """The catalog's own connection (to inspect per-connection state or plant inconsistent rows)."""
    return catalog._connection()


# ---------------------------------------------------------------------------------------------- schema / pragmas
def test_lazy_open_pragmas_and_schema_version(paths):
    catalog = Catalog(paths.db)
    assert not paths.db.exists()  # opened lazily
    assert catalog.get_meta("schema_version") == str(SCHEMA_VERSION)
    conn = raw(catalog)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables >= {"objects", "summaries", "lightcurves", "classifications", "followups", "ri_runs", "ri_events",
                      "transients", "proposals", "survey_log", "agent_runs", "agent_tool_calls", "heartbeats", "meta"}
    cols = [r[1] for r in conn.execute("PRAGMA table_info(objects)")]
    assert cols == ["object_id", "task", "region", "sky_version", "spec_json", "seed", "source", "priority", "status",
                    "attempts", "worker", "created_at", "started_at", "finished_at", "wall_time_s", "device",
                    "n_params", "error", "code_version", "random_control"]
    catalog.close()
    assert catalog.root == paths.root  # inferred from <root>/data/catalog.sqlite


def test_foreign_keys_are_enforced(cat):
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        raw(cat).execute("INSERT INTO summaries (object_id, json) VALUES ('nope', '{}')")


def test_schema_version_mismatch_is_refused(paths):
    catalog = Catalog(paths.db)
    catalog.set_meta("schema_version", str(SCHEMA_VERSION + 1))
    catalog.close()
    with pytest.raises(CatalogSchemaError, match="schema version"):
        Catalog(paths.db).queue_size()


# ---------------------------------------------------------------------------------------------- objects
def test_add_objects_is_idempotent_and_counts_new(cat):
    specs = make_specs(5)
    assert cat.add_objects(specs[:3], region="r", source="coarse") == 3
    assert cat.add_objects(specs + specs[:1], region="r", source="coarse") == 2  # duplicates in and across calls
    assert cat.queue_size() == 5
    assert cat.object_ids("r") == {s.object_id() for s in specs} and cat.object_ids("other") == set()
    stored = cat.get_spec(specs[0].object_id())
    assert stored.region == "r" and stored.to_dict() == {**specs[0].to_dict(), "region": "r"}
    with pytest.raises(ValueError):
        cat.add_objects(specs, region="", source="x")
    with pytest.raises(ValueError):
        cat.add_objects(specs, region="r", source="x", random_control_frac=1.5)
    with pytest.raises(KeyError):
        cat.get_spec("missing")


def test_random_control_marks(paths, tmp_path):
    specs = make_specs(2000)

    def marks(catalog: Catalog, **kw) -> list[int]:
        catalog.add_objects(specs, region="r", source="coarse", **kw)
        rows = catalog.query("SELECT object_id, random_control FROM objects")
        by_id = {r["object_id"]: r["random_control"] for r in rows}
        return [by_id[s.object_id()] for s in specs]

    a = marks(Catalog(tmp_path / "a.sqlite"), random_control_frac=0.3, rng=np.random.default_rng(7))
    b = marks(Catalog(tmp_path / "b.sqlite"), random_control_frac=0.3, rng=np.random.default_rng(7))
    c = marks(Catalog(tmp_path / "c.sqlite"), random_control_frac=0.3)  # no rng: hash of the object id
    d = marks(Catalog(tmp_path / "d.sqlite"), random_control_frac=0.3)
    assert a == b and c == d and a != c
    for m in (a, c):
        assert 0.25 < np.mean(m) < 0.35
    assert sum(marks(Catalog(tmp_path / "e.sqlite"))) == 0
    assert sum(marks(Catalog(tmp_path / "f.sqlite"), random_control_frac=1.0)) == len(specs)


def test_claim_order_priority_then_oldest(cat, clock):
    low = make_specs(3, start=0)
    high = make_specs(2, start=10)
    cat.add_objects(low, region="r", source="coarse", priority=1.0)
    clock.t += 1
    cat.add_objects(high, region="r", source="refine", priority=2.0)
    claimed = [cat.claim_next("w").object_id() for _ in range(5)]
    assert claimed == [s.object_id() for s in high + low]  # same created_at within a batch: insertion order
    assert cat.claim_next("w") is None


def test_claim_sets_running_attempts_and_heartbeat(cat, clock):
    (spec,) = make_specs(1)
    cat.add_objects([spec], region="r", source="coarse")
    got = cat.claim_next("w1")
    assert got.object_id() == spec.object_id() and got.region == "r"
    obj = cat.get_object(spec.object_id())
    assert (obj["status"], obj["worker"], obj["attempts"], obj["started_at"]) == ("running", "w1", 1, clock.t)
    (worker,) = cat.stats()["workers"]
    assert worker["worker_id"] == "w1" and worker["object_id"] == spec.object_id() and worker["status"] == "running"


def test_interrupted_objects_resume_first_and_refund_their_attempt(cat, clock):
    a, b = make_specs(2)
    cat.add_objects([a], region="r", source="coarse")
    clock.t += 1
    cat.add_objects([b], region="r", source="coarse")
    assert cat.claim_next("w1").object_id() == a.object_id()
    assert not cat.mark_interrupted(a.object_id(), worker_id="someone-else")  # guarded: not that worker's object
    assert cat.mark_interrupted(a.object_id(), worker_id="w1")
    obj = cat.get_object(a.object_id())
    assert obj["status"] == "interrupted" and obj["attempts"] == 0
    assert cat.queue_size() == 2
    assert cat.claim_next("w2").object_id() == a.object_id()  # older than b
    assert cat.get_object(a.object_id())["attempts"] == 1
    cat.mark_done(done_result(a.object_id()), "x.npz")
    assert not cat.mark_interrupted(a.object_id())  # done objects are left alone
    assert cat.get_object(a.object_id())["status"] == "done"


def test_unparseable_spec_is_failed_and_skipped(cat):
    bad, good = make_specs(2)
    cat.add_objects([bad], region="r", source="coarse", priority=5.0)
    cat.add_objects([good], region="r", source="coarse")
    spec = json.loads(cat.get_object(bad.object_id())["spec_json"])
    spec["train"]["removed_field"] = 1  # e.g. written by an incompatible code version
    raw(cat).execute("UPDATE objects SET spec_json = ? WHERE object_id = ?", (json.dumps(spec), bad.object_id()))
    assert cat.claim_next("w").object_id() == good.object_id()
    obj = cat.get_object(bad.object_id())
    assert obj["status"] == "failed" and "unparseable spec" in obj["error"]


def test_mark_done_records_summary_and_relative_lightcurve_path(cat, paths):
    (spec,) = make_specs(1)
    oid = spec.object_id()
    cat.add_objects([spec], region="r", source="coarse")
    cat.claim_next("w")
    summary = {"final_test_acc": np.float64(0.75), "final_test_loss": float("nan"), "final_train_loss": 0.1,
               "t_test_acc90": float("nan"), "grok_delay": float("inf"), "plateau_frac": 0.2, "n_drops": np.int64(2),
               "diverged": False, "extra": np.array([1.0, np.nan]), "summary_version": "summarize-v1"}
    lc_path = paths.lightcurve_path(oid)
    cat.mark_done(done_result(oid, summary, n_points=7), lc_path, code_version="v1.2")
    row = cat.query("SELECT * FROM summaries WHERE object_id = ?", (oid,))[0]
    assert row["final_test_acc"] == 0.75 and row["final_train_loss"] == 0.1 and row["plateau_frac"] == 0.2
    assert row["final_test_loss"] is None and row["t_test_acc90"] is None and row["grok_delay"] is None  # NaN/inf
    assert row["n_drops"] == 2 and row["diverged"] == 0
    stored = json.loads(row["json"])  # strict JSON: NaN -> null
    assert stored["final_test_loss"] is None and stored["extra"] == [1.0, None] and stored["n_drops"] == 2
    lc = cat.query("SELECT path, n_points FROM lightcurves")[0]
    assert lc == {"path": f"data/lightcurves/{oid[:2]}/{oid}.npz", "n_points": 7}
    assert cat.resolve_path(lc["path"]) == lc_path
    obj = cat.get_object(oid)
    assert obj["status"] == "done" and obj["error"] is None and obj["code_version"] == "v1.2"
    assert obj["wall_time_s"] == 1.5 and obj["n_params"] == 123 and obj["finished_at"] is not None
    assert obj["lightcurve"] == {"path": str(lc_path), "n_points": 7} and obj["summary"]["final_test_acc"] == 0.75
    assert cat.store_path("/elsewhere/lc.npz") == "/elsewhere/lc.npz"  # outside the root: absolute
    assert cat.resolve_path("/elsewhere/lc.npz") == Path("/elsewhere/lc.npz")
    with pytest.raises(ValueError, match="status 'done'"):
        cat.mark_done(RunResult(object_id=oid, status="failed"), lc_path)
    with pytest.raises(KeyError):
        cat.mark_done(done_result("unknown"), lc_path)


def test_mark_failed_truncates_and_guards_by_worker(cat):
    (spec,) = make_specs(1)
    oid = spec.object_id()
    cat.add_objects([spec], region="r", source="coarse")
    cat.claim_next("w1")
    assert not cat.mark_failed(oid, "boom", worker_id="w2")
    assert cat.get_object(oid)["status"] == "running"
    assert cat.mark_failed(oid, "Traceback\n" + "x" * 50_000 + "\nValueError: the end", worker_id="w1")
    obj = cat.get_object(oid)
    assert obj["status"] == "failed" and len(obj["error"]) <= 20_000
    assert obj["error"].startswith("Traceback") and obj["error"].endswith("ValueError: the end")
    with pytest.raises(KeyError):
        cat.mark_failed("unknown", "x")


# ---------------------------------------------------------------------------------------------- stale workers
def test_requeue_stale_by_heartbeat_age_and_max_attempts(paths, clock):
    cat = Catalog(paths.db, clock=clock, max_attempts=2)
    a, b, c = make_specs(3)
    cat.add_objects([a, b, c], region="r", source="coarse")
    assert cat.claim_next("w1").object_id() == a.object_id()
    assert cat.claim_next("w2").object_id() == b.object_id()
    clock.t += 100
    cat.heartbeat("w2", b.object_id(), "running")  # w2 is alive, w1 went silent 100 s ago
    assert cat.requeue_stale(60) == 1
    obj_a = cat.get_object(a.object_id())
    assert obj_a["status"] == "queued" and obj_a["worker"] is None and obj_a["attempts"] == 1
    assert "w1" in obj_a["error"] and "100 s" in obj_a["error"]
    assert cat.get_object(b.object_id())["status"] == "running"
    assert cat.requeue_stale(60) == 0  # nothing else is stale

    assert cat.claim_next("w3").object_id() == a.object_id()  # second attempt
    clock.t += 30
    cat.heartbeat("w2", b.object_id(), "running")
    clock.t += 100  # w3 silent for 100 s (2nd attempt of a), w2 for 100 s
    assert cat.requeue_stale(60) == 2
    obj_a = cat.get_object(a.object_id())
    assert obj_a["status"] == "failed" and obj_a["attempts"] == 2 and obj_a["error"].startswith("gave up")
    assert cat.get_object(b.object_id())["status"] == "queued"  # first attempt of b: requeued
    assert cat.get_object(c.object_id())["status"] == "queued"  # never claimed: untouched
    cat.close()


def test_requeue_stale_without_heartbeat_row_and_zero_threshold(cat, clock):
    a, b = make_specs(2)
    cat.add_objects([a, b], region="r", source="coarse")
    cat.claim_next("ghost")
    cat.claim_next("live")
    raw(cat).execute("DELETE FROM heartbeats WHERE worker_id = 'ghost'")
    assert cat.requeue_stale(3600) == 1  # no heartbeat at all counts as stale
    assert "no heartbeat" in cat.get_object(a.object_id())["error"]
    assert cat.requeue_stale(60) == 0  # "live" heartbeated just now
    assert cat.requeue_stale(0) == 1  # 0 releases every running job, even one claimed this very moment
    assert cat.queue_size() == 2
    with pytest.raises(ValueError):
        cat.requeue_stale(-1)


def test_requeue_stale_followups(paths, clock):
    cat = Catalog(paths.db, clock=clock, max_attempts=2)
    (spec,) = make_specs(1)
    oid = spec.object_id()
    cat.add_objects([spec], region="r", source="coarse")
    cat.claim_next("w")
    cat.mark_done(done_result(oid), "lc.npz")
    assert cat.request_followup(oid, "test")
    for attempt, expected in ((1, "requested"), (2, "failed")):
        assert cat.claim_followup("fw") == oid
        clock.t += 1000
        assert cat.requeue_stale(60) == 1
        row = cat.get_object(oid)["followup"]
        assert row["status"] == expected and row["attempts"] == attempt
    assert cat.get_object(oid)["status"] == "done"  # the object itself is untouched
    cat.close()


def test_release_worker_takes_back_its_jobs_despite_fresh_heartbeats(cat):
    a, b = make_specs(2)
    cat.add_objects([a, b], region="r", source="coarse")
    cat.claim_next("w0")
    cat.claim_next("w1")
    cat.heartbeat("w0", None, "idle")  # a restarted w0 heartbeats while its old object still says running
    assert cat.requeue_stale(60) == 0
    assert cat.release_worker("w0") == 1
    assert cat.get_object(a.object_id())["status"] == "queued"
    assert cat.get_object(b.object_id())["status"] == "running"


# ---------------------------------------------------------------------------------------------- concurrency
def _claim_until_empty(catalog: Catalog, worker_id: str, barrier, results) -> None:
    barrier.wait(timeout=60)
    claimed = []
    while (spec := catalog.claim_next(worker_id)) is not None:
        claimed.append(spec.object_id())
        catalog.heartbeat(worker_id, spec.object_id(), "running")
        time.sleep(0.002)
    results.put((worker_id, claimed))


def test_concurrent_claims_from_spawned_processes_are_exclusive(paths):
    catalog = Catalog(paths.db)
    specs = make_specs(30)
    catalog.add_objects(specs, region="r", source="coarse")
    assert catalog.queue_size() == 30  # the parent has an open connection; the pickled copies must not use it
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(3), ctx.Queue()
    procs = [ctx.Process(target=_claim_until_empty, args=(catalog, f"w{i}", barrier, results)) for i in range(3)]
    for p in procs:
        p.start()
    try:
        claims = dict(results.get(timeout=120) for _ in procs)
    finally:
        for p in procs:
            p.join(timeout=60)
    assert all(p.exitcode == 0 for p in procs)
    everything = [oid for ids in claims.values() for oid in ids]
    assert sorted(everything) == sorted(s.object_id() for s in specs)  # each object claimed exactly once
    assert sum(1 for ids in claims.values() if ids) >= 2  # the processes really competed
    rows = catalog.query("SELECT object_id, status, worker, attempts FROM objects")
    owner = {oid: w for w, ids in claims.items() for oid in ids}
    assert all(r["status"] == "running" and r["attempts"] == 1 and r["worker"] == owner[r["object_id"]]
               for r in rows)
    catalog.close()


def _claim_in_forked_child(catalog: Catalog, results) -> None:
    spec = catalog.claim_next("child")
    catalog.heartbeat("child", spec.object_id(), "running")
    results.put(spec.object_id())


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="fork is not available")
def test_catalog_is_usable_in_a_forked_child_and_afterwards_in_the_parent(paths):
    catalog = Catalog(paths.db)
    a, b, c = make_specs(3)
    catalog.add_objects([a, b, c], region="r", source="coarse")
    assert catalog.claim_next("parent").object_id() == a.object_id()  # connection open before the fork
    ctx = mp.get_context("fork")
    results = ctx.Queue()
    child = ctx.Process(target=_claim_in_forked_child, args=(catalog, results))
    child.start()
    child_claim = results.get(timeout=60)
    child.join(timeout=60)
    assert child.exitcode == 0 and child_claim == b.object_id()
    assert catalog.claim_next("parent").object_id() == c.object_id()  # the parent's connection still works
    assert {w["worker_id"] for w in catalog.stats()["workers"]} == {"parent", "child"}
    catalog.close()


def test_threads_share_the_process_connection(cat):
    cat.add_objects(make_specs(40), region="r", source="coarse")
    errors: list[BaseException] = []

    def beat() -> None:
        try:
            for _ in range(200):
                cat.heartbeat("hb", None, "idle")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    thread = threading.Thread(target=beat)
    thread.start()
    claimed = [cat.claim_next("main") for _ in range(40)]
    thread.join()
    assert not errors and len({s.object_id() for s in claimed}) == 40


def test_pickle_round_trip_reopens(cat):
    cat.add_objects(make_specs(2), region="r", source="coarse")
    clone = pickle.loads(pickle.dumps(cat))
    assert clone.path == cat.path and clone.root == cat.root and clone.queue_size() == 2
    clone.close()


# ---------------------------------------------------------------------------------------------- classifications
def test_classifications_latest_wins(cat, clock):
    (spec,) = make_specs(1)
    oid = spec.object_id()
    cat.add_objects([spec], region="r", source="coarse")
    first = cat.add_classification(oid, Classification(label="smooth", tags=["slow_start"], score=0.7,
                                                       reasons=["rules-v1: smooth: because", "tag slow_start: x"]),
                                   "rules-v1")
    second = cat.add_classification(oid, {"label": "grokking", "score": float("nan")}, "agent", note="looks grokked")
    assert second > first
    latest = cat.latest_classification(oid)  # same timestamp: insertion order decides
    assert latest["label"] == "grokking" and latest["classifier"] == "agent" and latest["score"] is None
    assert latest["note"] == "looks grokked" and latest["tags"] == []
    rows = cat.query("SELECT note, tags_json FROM classifications WHERE id = ?", (first,))
    assert rows[0]["note"] == "rules-v1: smooth: because\ntag slow_start: x"  # reasons kept as the note
    assert json.loads(rows[0]["tags_json"]) == ["slow_start"]
    assert cat.latest_classification("nobody") is None
    with pytest.raises(KeyError):
        cat.add_classification("nobody", Classification(label="smooth"), "rules-v1")
    with pytest.raises(ValueError):
        cat.add_classification(oid, {"label": ""}, "rules-v1")
    assert cat.stats()["labels"] == {"grokking": 1}


# ---------------------------------------------------------------------------------------------- follow-ups
def test_followups_request_claim_finish(cat, clock):
    a, b, c = make_specs(3)
    cat.add_objects([a, b, c], region="r", source="coarse")
    for s in (a, b):
        cat.claim_next("w")
        cat.mark_done(done_result(s.object_id()), "lc.npz")
    assert cat.request_followup(c.object_id(), "not done yet", priority=9.0)
    assert cat.request_followup(a.object_id(), "low", priority=1.0, requested_by="agent")
    clock.t += 1
    assert cat.request_followup(b.object_id(), "high", priority=3.0)
    assert not cat.request_followup(b.object_id(), "again")  # never twice
    with pytest.raises(KeyError):
        cat.request_followup("unknown", "x")
    assert cat.count_followups() == 3
    assert cat.count_followups(requested_by="scheduler") == 2
    assert cat.count_followups(requested_by="scheduler", since=clock.t) == 1
    assert cat.claim_followup("fw") == b.object_id()  # priority 9 is not done yet; then the highest priority
    assert cat.claim_followup("fw") == a.object_id()
    assert cat.claim_followup("fw") is None
    cat.finish_followup(b.object_id(), True)
    cat.finish_followup(a.object_id(), False, error="replay diverged")
    fb, fa = cat.get_object(b.object_id())["followup"], cat.get_object(a.object_id())["followup"]
    assert fb["status"] == "done" and fb["worker"] == "fw" and fb["attempts"] == 1
    assert fa["status"] == "failed" and fa["error"] == "replay diverged"
    with pytest.raises(KeyError):
        cat.finish_followup(c.object_id() + "x", True)
    assert cat.stats()["followups"] == {"requested": 1, "running": 0, "done": 1, "failed": 1}


# ---------------------------------------------------------------------------------------------- RI spectroscopy
def make_curve() -> RICurve:
    ri = np.array([[0.0, 0.002, 0.003], [0.0, 0.0, 0.01]])
    q = np.where(ri > 0, 1e-6, 0.5)
    return RICurve(steps=[10, 100, 1000], names=["x0", "x1", "yhat"], subsets=[(0, 2), (0, 1, 2)], ri=ri,
                   ri_plugin=ri, p=q, q=q, n_eff=[4096.0] * 3, meta={"alpha": 0.01, "min_effect": 1e-3})


def make_event(subset, names, kind="activation", step=100, **kw) -> RIEvent:
    fields = dict(subset=subset, subset_names=names, order=len(subset), kind=kind, step=step, step_lo=10,
                  step_hi=step, ri_before=0.0, ri_after=0.002, peak_ri=0.003, confidence=0.99)
    fields.update(kw)
    return RIEvent(**fields)


def test_ri_runs_events_transients_and_notifications(cat, paths):
    a, b = make_specs(2)
    cat.add_objects([a, b], region="r", source="coarse")
    oid = a.object_id()
    curve_path = paths.ri_path(oid, "insitu")
    run_id = cat.add_ri_run(oid, "insitu", curve_path, make_curve())
    run = cat.get_object(oid)["ri_runs"][0]
    assert run["ri_run_id"] == run_id and run["path"] == str(curve_path)
    assert (run["n_subsets"], run["n_points"], run["max_order"], run["alpha"], run["min_effect"]) == (2, 3, 3, 0.01,
                                                                                                        1e-3)
    events = [make_event((0, 2), ("x0", "yhat"), matches_ground_truth=True),
              make_event((0, 1, 2), ("x0", "x1", "yhat"), step=1000, has_precursor=False)]
    event_ids = cat.add_events(run_id, oid, events)
    rows = cat.query("SELECT * FROM ri_events ORDER BY event_id")
    assert [r["event_id"] for r in rows] == event_ids
    assert json.loads(rows[1]["subset_json"]) == ["x0", "x1", "yhat"] and rows[1]["order_k"] == 3
    assert (rows[0]["matches_gt"], rows[0]["has_precursor"], rows[1]["has_precursor"]) == (1, None, 0)
    with pytest.raises(ValueError, match="belongs to"):
        cat.add_events(run_id, b.object_id(), events)
    with pytest.raises(KeyError):
        cat.add_events(run_id + 99, oid, events)

    transients = [Transient(kind="pure_synergy", severity=3, message="synergy", event_index=1),
                  Transient(kind="grokking", severity=1, message="grokked"),
                  {"kind": "extinction", "severity": 2, "message": "gone", "event_index": 0}]
    tids = cat.add_transients(oid, transients, event_ids)
    linked = {r["transient_id"]: r["event_id"] for r in cat.query("SELECT transient_id, event_id FROM transients")}
    assert [linked[t] for t in tids] == [event_ids[1], None, event_ids[0]]
    with pytest.raises(ValueError, match="out of range"):
        cat.add_transients(oid, [Transient(kind="extinction", severity=2, message="x", event_index=5)], event_ids)

    pending = cat.pending_notifications(min_severity=2)
    assert [p["kind"] for p in pending] == ["pure_synergy", "extinction"]  # most severe first
    assert pending[0]["region"] == "r" and pending[0]["seed"] == a.seed
    assert len(cat.pending_notifications()) == 3
    assert cat.mark_notified([pending[0]["transient_id"]]) == 1
    assert cat.mark_notified([pending[0]["transient_id"]]) == 0
    assert [p["kind"] for p in cat.pending_notifications(min_severity=2)] == ["extinction"]
    stats = cat.stats()
    assert stats["ri_runs"] == 1 and stats["ri_events"] == 2
    assert stats["transients"] == {"total": 3, "pending_notification": 2, "unacknowledged": 3}
    assert cat.get_object(oid)["n_ri_events"] == 2 and len(cat.get_object(oid)["transients"]) == 3


# ---------------------------------------------------------------------------------------------- proposals, log, agent
def test_proposals_log_meta_and_agent_audit(cat):
    point = {"width": 128.0, "train.lr": 0.001, "n_train": None}
    pid = cat.add_proposal("agent", "r", point, task="toy", seeds=3, rationale="test a hypothesis")
    row = cat.query("SELECT * FROM proposals WHERE proposal_id = ?", (pid,))[0]
    assert row["point_json"] == point_key(point) and row["status"] == "proposed" and row["seeds"] == 3
    assert cat.add_log("agent", "nightly", "Night 1", "# all quiet") > 0

    cat.set_meta("a_b/x", "1")
    cat.set_meta("aXb/y", "2")
    cat.set_meta("a%/z", "3")
    assert cat.meta_items("a_b/") == {"a_b/x": "1"}  # no LIKE wildcards
    assert cat.meta_items("a%") == {"a%/z": "3"}
    assert cat.get_meta("missing", "default") == "default"
    with pytest.raises(TypeError):
        cat.set_meta("k", 3)

    run_id = cat.start_agent_run("nightly", "claude-opus-5-5")
    call_id = cat.log_tool_call(run_id, "get_object", {"object_id": "abc"}, "x" * 30_000, False)
    cat.finish_agent_run(run_id, status="done", input_tokens=1200, output_tokens=300, est_cost_usd=0.01,
                         n_tool_calls=1, stop_reason="end_turn")
    run = cat.query("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))[0]
    assert run["status"] == "done" and run["input_tokens"] == 1200 and run["finished_at"] is not None
    call = cat.query("SELECT * FROM agent_tool_calls WHERE id = ?", (call_id,))[0]
    assert json.loads(call["input_json"]) == {"object_id": "abc"} and len(call["output_preview"]) <= 20_000
    assert call["is_error"] == 0
    with pytest.raises(ValueError, match="unknown agent_runs fields"):
        cat.finish_agent_run(run_id, status="done", **{"status = 'x'; --": 1})
    with pytest.raises(KeyError):
        cat.finish_agent_run(run_id + 1, status="done")


# ---------------------------------------------------------------------------------------------- read-only queries
def test_query_reads(cat):
    cat.add_objects(make_specs(3), region="r", source="coarse")
    assert cat.query("SELECT COUNT(*) AS n FROM objects") == [{"n": 3}]
    assert cat.query("select seed from objects where seed >= ? order by seed", (1,)) == [{"seed": 1}, {"seed": 2}]
    assert cat.query("SELECT seed FROM objects WHERE seed = :s", {"s": 2}) == [{"seed": 2}]
    tricky = "  -- leading; comment\n /* SELECT; */ SELECT ';' AS s, \"we;ird\" FROM (SELECT 1 AS \"we;ird\");  "
    assert cat.query(tricky) == [{"s": ";", "we;ird": 1}]
    assert cat.query("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) "
                     "SELECT SUM(n) AS s FROM r") == [{"s": 15}]
    assert {r["name"] for r in cat.query("SELECT name FROM pragma_table_info('heartbeats')")} >= {"worker_id"}
    assert len(cat.query("SELECT seed FROM objects", max_rows=2)) == 2


@pytest.mark.parametrize("sql", [
    "DELETE FROM objects",
    "SELECT 1; DELETE FROM objects",
    "select 1;delete from objects",
    "SELECT 1; SELECT 2",
    "SELECT ';'; DROP TABLE objects",
    "WITH x AS (SELECT 1) DELETE FROM objects",
    "WITH x AS (SELECT 1) INSERT INTO meta (key, value) VALUES ('a', 'b')",
    "WITH x AS (SELECT 1) UPDATE objects SET status = 'done'",
    "WITH x AS (SELECT 1) REPLACE INTO meta (key, value) VALUES ('schema_version', '99')",
    "/* SELECT */ DROP TABLE objects",
    "-- SELECT\nDROP TABLE objects",
    "PRAGMA journal_mode = DELETE",
    "ATTACH DATABASE 'other.db' AS other",
    "SELECT * FROM objects; ATTACH DATABASE 'other.db' AS other",
    "SELECT load_extension('evil')",
    "SELECT * FROM pragma_journal_mode()",  # only schema-introspection pragmas are allowed
    "VACUUM",
    "EXPLAIN SELECT 1",
    "",
    "   ",
    ";",
])
def test_query_rejects_anything_but_one_read(cat, sql):
    cat.add_objects(make_specs(2), region="r", source="coarse")
    with pytest.raises(ValueError):
        cat.query(sql)
    assert cat.queue_size() == 2 and cat.get_meta("schema_version") == str(SCHEMA_VERSION)
    assert raw(cat).execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    cat.set_meta("still", "writable")  # the read-only guard is lifted after the query
    assert cat.get_meta("still") == "writable"


def test_query_timeout(cat):
    with pytest.raises(TimeoutError):
        cat.query("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r",
                  timeout_s=0.05)
    assert cat.query("SELECT 1 AS one") == [{"one": 1}]


# ---------------------------------------------------------------------------------------------- stats / backup
def test_stats(cat, clock):
    specs = make_specs(4)
    cat.add_objects(specs, region="r", source="coarse")
    cat.add_objects(make_specs(1, task="other"), region="q", source="agent")
    cat.claim_next("w")
    cat.mark_done(done_result(specs[0].object_id()), "lc.npz")
    cat.claim_next("w")
    cat.mark_failed(specs[1].object_id(), "boom")
    stats = cat.stats()
    assert stats["objects"] == {"queued": 3, "running": 0, "done": 1, "failed": 1, "interrupted": 0, "total": 5}
    assert stats["queue"] == 3 and stats["done_last_24h"] == 1 and stats["failed_last_24h"] == 1
    assert stats["by_region"]["r"]["queued"] == 2 and stats["by_region"]["q"]["queued"] == 1
    assert stats["wall_time_s_done"] == 1.5 and stats["schema_version"] == SCHEMA_VERSION
    assert stats["workers"][0]["age_s"] == 0.0
    clock.t += 2 * 86_400
    assert cat.stats()["done_last_24h"] == 0


def test_backup_is_a_consistent_standalone_copy(cat, paths):
    specs = make_specs(5)
    cat.add_objects(specs, region="r", source="coarse")
    cat.claim_next("w")
    cat.mark_done(done_result(specs[0].object_id()), paths.lightcurve_path(specs[0].object_id()))
    dest = cat.backup(paths.backups / "catalog-1.sqlite")
    assert dest == paths.backups / "catalog-1.sqlite"
    assert sorted(p.name for p in paths.backups.iterdir()) == ["catalog-1.sqlite"]  # no -wal / partial leftovers
    conn = sqlite3.connect(dest)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 1
    conn.close()
    copy = Catalog(dest, root=paths.root)
    assert copy.get_object(specs[0].object_id())["status"] == "done" and copy.queue_size() == 4
    copy.close()
    cat.add_objects(make_specs(1, start=100), region="r", source="coarse")  # the source keeps working
    assert cat.backup(dest) == dest  # overwrite with a newer backup
    assert sqlite3.connect(dest).execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 6
    with pytest.raises(ValueError):
        cat.backup(paths.db)


def test_failed_backup_leaves_no_partial_file(cat, paths, monkeypatch):
    class FullDisk:
        def backup(self, target):
            raise sqlite3.OperationalError("database or disk is full")

    cat.add_objects(make_specs(1), region="r", source="coarse")
    monkeypatch.setattr(cat, "_connection", lambda: FullDisk())
    with pytest.raises(sqlite3.OperationalError, match="full"):
        cat.backup(paths.backups / "catalog-2.sqlite")
    assert list(paths.backups.iterdir()) == []
