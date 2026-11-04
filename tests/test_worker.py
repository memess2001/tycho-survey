"""Tests of tycho.survey.worker: observing and recording one object (light curve, summary, classification, in-situ
RI, transients), idempotent re-recording, interruption and exact resume, failures, the disk guard, the claim budget
and the worker loop (in a thread)."""
from __future__ import annotations

import multiprocessing as mp
import sqlite3
import threading
import time

import numpy as np
import pytest
import torch
from survey_env import make_root, tiny_config

import tycho.survey.worker as worker_mod
import tycho.train.device as device_mod
from tycho.analysis.events import detect_events
from tycho.analysis.ricurve import RICurve
from tycho.survey.catalog import Catalog
from tycho.survey.followup import INSITU_KIND
from tycho.survey.scheduler import Scheduler, SchedulerPolicy
from tycho.survey.sky import Sky
from tycho.survey.worker import _Worker, load_sky, process_object, worker_main
from tycho.train.runner import RESUME_FILE, run_object


@pytest.fixture(autouse=True, scope="module")
def _torch_settings():
    """One thread (as in the survey); worker_main's process-wide torch settings are undone afterwards."""
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1], warn_only=before[2])


@pytest.fixture
def env(tmp_path):
    paths = make_root(tmp_path / "root")
    cfg = tiny_config(paths)
    sky = load_sky(paths, cfg)
    catalog = Catalog(paths.db, root=paths.root)
    Scheduler(catalog, sky, SchedulerPolicy(target_queue=100)).top_up()  # all 8 objects, ctrl (priority 2) first
    yield paths, cfg, sky, catalog
    catalog.close()


def claim(catalog: Catalog, region: str, worker: str = "w0"):
    spec = catalog.claim_next(worker)
    assert spec is not None and spec.region == region
    return spec


def counts(catalog: Catalog, oid: str) -> dict:
    return {t: catalog.query(f"SELECT COUNT(*) AS n FROM {t} WHERE object_id = ?", (oid,))[0]["n"]
            for t in ("classifications", "ri_runs", "ri_events", "transients", "summaries", "lightcurves")}


def test_process_object_with_insitu_ri(env):
    paths, cfg, sky, catalog = env
    spec = claim(catalog, "ctrl")
    oid = spec.object_id()
    process_object(catalog, paths, spec, cfg, lambda: False, worker_id="w0", sky=sky)
    obj = catalog.get_object(oid)
    assert obj["status"] == "done" and obj["code_version"] == "test" and obj["device"] == "cpu"
    assert obj["n_params"] > 0 and obj["wall_time_s"] > 0 and obj["error"] is None
    lc_path = paths.lightcurve_path(oid)
    assert obj["lightcurve"]["path"] == str(lc_path) and lc_path.is_file()
    with np.load(lc_path) as lc:
        assert lc["step"][0] == 0 and lc["step"][-1] == 300 and len(lc["step"]) == obj["lightcurve"]["n_points"]
        assert {"train_loss", "test_acc", "ntk_trace", "x_support_frac"} <= set(lc.files)
    assert obj["summary"]["steps_total"] == 300 and "t_test_acc90" in obj["summary"]
    cls = obj["classification"]
    assert cls["classifier"] == "rules-v1" and cls["label"] and cls["note"].startswith("rules-v1:")
    (run,) = obj["ri_runs"]
    assert run["kind"] == INSITU_KIND and run["n_points"] == 6 and run["min_effect"] == pytest.approx(1e-3)
    curve = RICurve.load(run["path"])
    assert curve.meta["kind"] == "insitu" and curve.meta["n_probe"] == 512  # region ri overrides the config
    assert obj["n_ri_events"] == len(detect_events(curve, alpha=0.01))
    assert not paths.work_dir(oid).exists()  # the empty work directory is removed


def test_process_object_without_ri_and_idempotent_recording(env):
    paths, cfg, sky, catalog = env
    for _ in range(4):  # skip the ctrl objects
        catalog.mark_failed(catalog.claim_next("w0").object_id(), "skipped")
    spec = claim(catalog, "grok")
    oid = spec.object_id()
    process_object(catalog, paths, spec, cfg, lambda: False, worker_id="w0", sky=sky)
    first = counts(catalog, oid)
    assert first["ri_runs"] == 0 and first["classifications"] == 1 and first["summaries"] == 1
    kinds = {r["kind"] for r in catalog.get_object(oid)["transients"]}
    assert kinds <= {"loss_spike", "grokking", "unknown_class"}  # photometric transients only
    # a re-observation (e.g. after the object was requeued) records nothing twice
    process_object(catalog, paths, spec, cfg, lambda: False, worker_id="w0", sky=sky)
    assert counts(catalog, oid) == first and catalog.get_object(oid)["status"] == "done"


def test_interrupted_object_resumes_exactly(env, tmp_path):
    paths, cfg, sky, catalog = env
    spec = claim(catalog, "ctrl")
    oid = spec.object_id()
    process_object(catalog, paths, spec, cfg, lambda: True, worker_id="w0", sky=sky)  # stop at the first poll
    obj = catalog.get_object(oid)
    assert obj["status"] == "interrupted" and obj["attempts"] == 0  # the graceful stop refunds the attempt
    assert (paths.work_dir(oid) / RESUME_FILE).is_file()
    assert counts(catalog, oid)["summaries"] == 0
    again = catalog.claim_next("w1")
    assert again.object_id() == oid  # interrupted objects are claimed again first (same priority, older)
    process_object(catalog, paths, again, cfg, lambda: False, worker_id="w1", sky=sky)
    assert catalog.get_object(oid)["status"] == "done"
    reference = run_object(spec, work_dir=tmp_path / "ref")
    with np.load(paths.lightcurve_path(oid)) as lc:
        for key in ("step", "train_loss", "test_loss", "param_norm", "ntk_trace"):
            assert np.array_equal(lc[key], reference.light_curve[key], equal_nan=True), key


def test_failed_run_is_recorded_and_does_not_raise(env):
    paths, cfg, sky, catalog = env
    spec = claim(catalog, "ctrl")
    bad = spec.__class__(task=spec.task, task_cfg={**spec.task_cfg, "bogus": 1}, seed=spec.seed, train=spec.train,
                         sky_version=spec.sky_version, region=spec.region)
    catalog.add_objects([bad], region="ctrl", source="test")
    process_object(catalog, paths, bad, cfg, lambda: False, sky=sky)
    obj = catalog.get_object(bad.object_id())
    assert obj["status"] == "failed" and "unknown config keys" in obj["error"]


def test_infrastructure_errors_hand_the_object_back(env, monkeypatch):
    paths, cfg, sky, catalog = env
    w = _Worker("w0", paths, cfg, threading.Event(), threading.Event(), None, None)
    original = worker_mod._record
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(worker_mod, "_record", flaky)
    monkeypatch.setattr(worker_mod, "IDLE_POLL_S", 0.0)
    w.step()  # claims the first ctrl object; recording fails: not marked failed, handed back on the next step
    oid = catalog.query("SELECT object_id FROM objects WHERE attempts = 1")[0]["object_id"]
    assert catalog.get_object(oid)["status"] == "running"
    w.step()  # releases it (queued, attempt counted) and claims it again: done
    obj = catalog.get_object(oid)
    assert obj["status"] == "done" and obj["attempts"] == 2 and len(calls) == 2
    w.catalog.close()


def test_disk_guard_pauses_claiming(env, monkeypatch):
    paths, cfg, sky, catalog = env
    cfg = {**cfg, "survey": {**cfg["survey"], "min_free_gb": 5.0}}
    stop = threading.Event()
    stop.set()  # sleeps return at once
    w = _Worker("w0", paths, cfg, stop, threading.Event(), None, None)
    w.heartbeat.set = lambda oid, status: catalog.heartbeat("w0", oid, status)  # write synchronously
    monkeypatch.setattr(worker_mod, "free_disk_gb", lambda path: 1.25)
    queued = catalog.queue_size()
    w.step()
    assert catalog.queue_size() == queued
    assert catalog.stats()["workers"][0]["status"] == "paused: 1.2 GB free"
    monkeypatch.setattr(worker_mod, "free_disk_gb", lambda path: 50.0)
    stop.clear()
    w.step()  # claims again
    assert catalog.queue_size() == queued - 1
    w.catalog.close()


def test_claim_budget_limits_and_refunds(env, monkeypatch):
    paths, cfg, sky, catalog = env
    monkeypatch.setattr(worker_mod, "IDLE_POLL_S", 0.0)
    budget = mp.get_context("spawn").Value("q", 1)
    w = _Worker("w0", paths, cfg, threading.Event(), threading.Event(), budget, None)
    w.step()
    assert budget.value == 0 and catalog.stats()["objects"]["done"] == 1
    w.step()  # budget used up: nothing claimed
    assert catalog.stats()["objects"]["done"] == 1 and catalog.stats()["objects"]["running"] == 0
    stop = threading.Event()
    stop.set()
    budget.value = 1
    w.stop_event = stop  # the next object is interrupted at once: its claim is refunded
    w.step()
    assert budget.value == 1 and catalog.stats()["objects"]["interrupted"] == 1
    w.catalog.close()


def test_worker_main_in_a_thread(env, monkeypatch):
    paths, cfg, sky, catalog = env
    make_root(paths.root, config={"survey": {"torch_threads": 2}})
    monkeypatch.setattr(worker_mod, "IDLE_POLL_S", 0.1)
    threads = []
    original = device_mod.configure_torch
    monkeypatch.setattr(device_mod, "configure_torch", lambda n, *a, **k: (threads.append(n), original(n, *a, **k)))
    for spec in iter(lambda: catalog.claim_next("setup"), None):  # leave only two queued objects
        catalog.mark_failed(spec.object_id(), "not part of this test")
    sky_obj = Sky.from_yaml(paths.configs / "tiny.yaml")
    specs = [sky_obj.make_spec("grok", p, 7) for p in sky_obj.points("grok")]
    catalog.add_objects(specs, region="grok", source="test")
    stop = threading.Event()
    t = threading.Thread(target=worker_main, args=("wt", paths.root, stop), daemon=True)
    t.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and catalog.stats()["objects"]["done"] < 2:
        time.sleep(0.1)
    stop.set()
    t.join(timeout=30)
    assert not t.is_alive()
    assert catalog.stats()["objects"]["done"] == 2
    (hb,) = [w for w in catalog.stats()["workers"] if w["worker_id"] == "wt"]
    assert hb["status"] == "stopped" and hb["object_id"] is None
    assert threads == [2]  # survey.torch_threads
