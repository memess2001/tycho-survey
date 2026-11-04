"""Tests of tycho.survey.followup: RI settings (min_effect precedence, calibration file), the dense window, and
follow-up by replay on a finished tiny object (RI run, events, transients, idempotency, failure, interruption and
resume), plus the foreground runner ``tycho.survey.worker.run_followups``."""
from __future__ import annotations

import json
import sqlite3

import pytest
import torch
import yaml
from survey_env import make_root, tiny_config

import tycho.survey.followup as followup_mod
from tycho.analysis.events import detect_events
from tycho.analysis.ricurve import RICurve
from tycho.config import DEFAULT_CONFIG
from tycho.survey.catalog import Catalog
from tycho.survey.followup import (FOLLOWUP_KIND, N_DENSE, dense_window, process_followup, resolve_ri_config,
                                   run_followup)
from tycho.survey.scheduler import Scheduler, SchedulerPolicy
from tycho.survey.worker import load_sky, process_object, run_followups
from tycho.train.runner import RESUME_FILE
from tycho.train.schedule import eval_steps, subsample_log


@pytest.fixture(autouse=True, scope="module")
def _torch_settings():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1], warn_only=before[2])


def write_calibration(paths, noise_at_512: float = 0.004) -> None:
    table = [{"order": 2, "n": 512, "noise": noise_at_512, "floor": 1e-3, "value": noise_at_512, "source": "E1"},
             {"order": 2, "n": 4096, "noise": noise_at_512 / 8, "floor": 1e-3, "value": 1e-3, "source": "E1"}]
    doc = {"format": "tycho-calibration/1", "recommended_min_effect": {"rule": "test", "table": table}}
    path = paths.reports / "calibration" / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- RI settings
def test_resolve_ri_config_merges_config_and_region(tmp_path):
    paths = make_root(tmp_path)
    cfg = tiny_config(paths)
    sky = load_sky(paths, cfg)
    ri = resolve_ri_config(cfg, sky.region("ctrl"), root=paths.root)
    assert ri.enabled and ri.n_probe == 512 and ri.n_points == 6  # region ri overrides config ri
    assert ri.alpha == DEFAULT_CONFIG["ri"]["alpha"] and ri.keep_top == DEFAULT_CONFIG["ri"]["keep_top"]
    assert ri.min_effect == pytest.approx(1e-3)  # no calibration: the built-in default
    assert resolve_ri_config(cfg, None, root=paths.root, enabled=False).enabled is False


def test_min_effect_precedence(tmp_path):
    paths = make_root(tmp_path, config={"ri": {"n_probe": 4096}})
    write_calibration(paths, noise_at_512=0.004)
    cfg = tiny_config(paths)
    region = load_sky(paths, cfg).region("ctrl")  # region n_probe 512
    # the calibrated value at the scan's probe size: 0.004 at n = 512; at 4096 the noise (0.0005) is below the floor
    assert resolve_ri_config(cfg, region, root=paths.root).min_effect == pytest.approx(0.004)
    assert resolve_ri_config(cfg, None, root=paths.root).min_effect == pytest.approx(1e-3)
    # an explicit ri.min_effect in tycho.yaml wins over the calibration (even if equal to the built-in default)
    data = yaml.safe_load((paths.root / "tycho.yaml").read_text(encoding="utf-8"))
    data["ri"]["min_effect"] = 1.0e-3
    (paths.root / "tycho.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = tiny_config(paths)
    assert resolve_ri_config(cfg, region, root=paths.root).min_effect == pytest.approx(1e-3)
    # a region's own min_effect wins over everything
    region.ri["min_effect"] = 0.02
    assert resolve_ri_config(cfg, region, root=paths.root).min_effect == pytest.approx(0.02)


def test_unusable_calibration_file_falls_back(tmp_path, caplog):
    paths = make_root(tmp_path)
    cal = paths.reports / "calibration" / "calibration.json"
    cal.parent.mkdir(parents=True)
    cal.write_text('{"format": "something-else"}', encoding="utf-8")
    cfg = tiny_config(paths)
    assert resolve_ri_config(cfg, None, root=paths.root).min_effect == pytest.approx(1e-3)
    assert "unusable" in caplog.text


def test_dense_window():
    assert dense_window({"largest_drop_lo": 30.0, "largest_drop_hi": 100.0}) == (30, 100)
    assert dense_window({"largest_drop_lo": float("nan"), "largest_drop_hi": 100.0}) is None
    assert dense_window({"largest_drop_lo": None, "largest_drop_hi": 100}) is None
    assert dense_window({"largest_drop_lo": 100, "largest_drop_hi": 30}) is None
    assert dense_window(None) is None and dense_window({}) is None


# ---------------------------------------------------------------------------------------------- follow-up
@pytest.fixture
def finished(tmp_path):
    """A survey root whose two grok objects of width 16 are done (observed by the worker)."""
    paths = make_root(tmp_path / "root")
    cfg = tiny_config(paths)
    sky = load_sky(paths, cfg)
    catalog = Catalog(paths.db, root=paths.root)
    region = sky.region("grok")
    specs = [sky.make_spec(region, {"width": 16}, seed) for seed in (0, 1)]
    catalog.add_objects(specs, region="grok", source="coarse")
    for _ in specs:
        spec = catalog.claim_next("w0")
        process_object(catalog, paths, spec, cfg, lambda: False, worker_id="w0", sky=sky)
    oids = [s.object_id() for s in specs]
    assert all(catalog.get_object(o)["status"] == "done" for o in oids)
    yield paths, cfg, sky, catalog, oids
    catalog.close()


def test_run_followup_records_ri_run_events_and_transients(finished):
    paths, cfg, sky, catalog, oids = finished
    oid = oids[0]
    assert catalog.request_followup(oid, "test", requested_by="test")
    assert catalog.claim_followup("w0") == oid
    ri_cfg = resolve_ri_config(cfg, sky.region("grok"), root=paths.root)
    run_id = run_followup(catalog, paths, oid, ri_cfg)
    obj = catalog.get_object(oid)
    (run,) = [r for r in obj["ri_runs"] if r["ri_run_id"] == run_id]
    assert run["kind"] == FOLLOWUP_KIND and run["path"] == str(paths.ri_path(oid, FOLLOWUP_KIND))
    curve = RICurve.load(run["path"])
    assert curve.meta["kind"] == "replay" and curve.meta["replay_fidelity"] == 0.0  # bit-for-bit replay on CPU
    assert curve.meta["followup_reason"] == "test"
    spec = catalog.get_spec(oid)
    log_points = subsample_log(eval_steps(spec.train.steps, spec.train.n_log_evals, spec.train.n_lin_evals), 6)
    window = dense_window(obj["summary"])
    assert set(log_points.tolist()) <= set(curve.steps.tolist())
    assert window is not None  # these objects have a sharp loss drop: dense points in its bracket
    assert curve.meta["dense"] == [window[0], window[1]]
    inside = curve.steps[(curve.steps >= window[0]) & (curve.steps <= window[1])]
    assert inside.size >= min(N_DENSE, window[1] - window[0] + 1)
    events = catalog.query("SELECT * FROM ri_events WHERE ri_run_id = ?", (run_id,))
    assert len(events) >= 1 and len(events) == len(detect_events(curve, alpha=ri_cfg.alpha))
    event_ids = {e["event_id"] for e in events}
    followup_transients = [t for t in obj["transients"] if t["event_id"] is not None]
    assert all(t["event_id"] in event_ids for t in followup_transients)
    before = (len(obj["transients"]), obj["n_ri_events"])
    assert run_followup(catalog, paths, oid, ri_cfg) == run_id  # idempotent: nothing recorded twice
    obj = catalog.get_object(oid)
    assert (len(obj["transients"]), obj["n_ri_events"]) == before and len(obj["ri_runs"]) == 1
    assert not paths.work_dir(oid).exists()  # the replay's work directory is cleaned up


def test_followup_of_an_unfinished_object_is_refused(finished):
    paths, cfg, sky, catalog, oids = finished
    region = sky.region("grok")
    other = sky.make_spec(region, {"width": 32}, 0)
    catalog.add_objects([other], region="grok", source="coarse")
    with pytest.raises(ValueError, match="not done"):
        run_followup(catalog, paths, other.object_id(), resolve_ri_config(cfg, region, root=paths.root))


def test_process_followup_success_failure_and_infrastructure_errors(finished, monkeypatch):
    paths, cfg, sky, catalog, oids = finished
    ri_cfg = resolve_ri_config(cfg, sky.region("grok"), root=paths.root)
    for oid in oids:
        catalog.request_followup(oid, "test")
    first = catalog.claim_followup("w0")
    assert process_followup(catalog, paths, first, ri_cfg, worker_id="w0") == "done"
    assert catalog.get_object(first)["followup"]["status"] == "done"

    second = catalog.claim_followup("w0")
    original = followup_mod.run_followup
    monkeypatch.setattr(followup_mod, "run_followup",
                        lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")))
    with pytest.raises(sqlite3.OperationalError):  # not the object's fault: the worker hands it back
        process_followup(catalog, paths, second, ri_cfg, worker_id="w0")
    assert catalog.get_object(second)["followup"]["status"] == "running"

    monkeypatch.setattr(followup_mod, "run_followup",
                        lambda *a, **k: (_ for _ in ()).throw(followup_mod.FollowupFailed("replay exploded")))
    assert process_followup(catalog, paths, second, ri_cfg, worker_id="w0") == "failed"
    fu = catalog.get_object(second)["followup"]
    assert fu["status"] == "failed" and fu["error"] == "replay exploded"
    monkeypatch.setattr(followup_mod, "run_followup", original)


def test_interrupted_followup_is_handed_back_and_resumes(finished):
    paths, cfg, sky, catalog, oids = finished
    oid = oids[1]
    ri_cfg = resolve_ri_config(cfg, sky.region("grok"), root=paths.root)
    catalog.request_followup(oid, "test")
    assert catalog.claim_followup("w0") == oid
    assert process_followup(catalog, paths, oid, ri_cfg, worker_id="w0", stop_flag=lambda: True) == "interrupted"
    fu = catalog.get_object(oid)["followup"]
    assert fu["status"] == "requested"  # claimable again
    assert (paths.work_dir(oid) / "followup" / RESUME_FILE).is_file()  # the replay resumes from here
    assert catalog.claim_followup("w1") == oid
    assert process_followup(catalog, paths, oid, ri_cfg, worker_id="w1") == "done"
    curve = RICurve.load(catalog.get_object(oid)["ri_runs"][0]["path"])
    assert curve.meta["replay_fidelity"] == 0.0


def test_run_followups_in_the_foreground(finished):
    paths, cfg, sky, catalog, oids = finished
    catalog.request_followup(oids[0], "requested by hand", requested_by="human")
    outcomes = run_followups(paths, cfg, 1, worker_id="cli-test")
    assert outcomes == [(oids[0], "done")]
    (hb,) = [w for w in catalog.stats()["workers"] if w["worker_id"] == "cli-test"]
    assert hb["status"] == "stopped"
    # nothing requested and nothing selectable (two done seeds of one point: no boundary, no unknown class)
    preview = Scheduler(catalog, sky, SchedulerPolicy()).select_followups(5, request=False)
    assert run_followups(paths, cfg, 2, worker_id="cli-test") == [(oid, "done") for oid, _ in preview]
    assert run_followups(paths, cfg, 1, stop_flag=lambda: True, worker_id="cli-test") == []
