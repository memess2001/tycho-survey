"""Tests of tycho.agent.tools: strict schemas, every tool against a seeded catalog, validation, caps, truncation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from tycho.agent.tools import (AGENT_FOLLOWUP_PRIORITY, NIGHTLY_TOOLS, TOOL_SPECS, WEEKLY_TOOLS, SurveyTools,
                               ToolLimits, fit_json, iso_week, power_law_fit, tool_specs, validate_input)
from tycho.analysis.events import RIEvent, Transient
from tycho.analysis.ricurve import RICurve
from tycho.config import Paths
from tycho.survey.catalog import Catalog
from tycho.survey.sky import Sky, point_key
from tycho.train.spec import RunResult

DAY = 86_400.0
T0 = 1_790_000_000.0  # "now" in these tests

TEST_SKY = {
    "version": "agent-test",
    "regions": [
        {"name": "grok", "task": "modular_addition", "seeds": 3, "priority": 1.5, "ri_policy": "followup",
         "fixed": {"depth": 1}, "train": {"steps": 1000},
         "knobs": {"width": {"values": [16, 64, 256], "kind": "log", "refine": True},
                   "train.lr": {"values": [0.001, 0.01], "kind": "log"}}},
        {"name": "ctrl", "task": "sparse_parity", "seeds": 2, "priority": 3.0, "ri_policy": "always",
         "knobs": {"k": {"values": [2, 3], "kind": "ordinal"}}},
        {"name": "dark", "task": "toy_superposition", "seeds": 2, "ri_policy": "never",
         "knobs": {"feature_prob": {"values": [0.01, 0.1], "kind": "log"}}},
    ],
}

DEFAULTS: dict[str, dict] = {
    "list_new_objects": {"since_hours": 0, "region": "", "status": "any", "limit": 0, "offset": 0},
    "get_object": {"include_lightcurve": True},
    "list_unknown_queue": {"region": "", "only_unreviewed": False, "limit": 0},
    "list_transients": {"since_hours": 0, "min_severity": 0, "kind": "any", "unacknowledged_only": False,
                        "region": "", "limit": 0},
    "list_ri_events": {"object_id": "", "region": "", "kind": "any", "min_order": 0, "ground_truth": "any",
                       "since_hours": 0, "limit": 0},
    "query_summaries": {"region": "", "group_by": [], "metrics": ["final_test_acc"], "label": "", "where": [],
                        "finished_within_hours": 0},
    "fit_power_law": {"where": [], "label": "", "aggregate": "point_median", "finished_within_hours": 0},
    "get_sky": {"region": ""},
    "list_proposals": {"since_hours": 0, "source": "any", "region": "", "limit": 0},
    "read_survey_log": {"since_hours": 0, "kind": "any", "limit": 0, "max_chars_per_entry": 0},
    "label_object": {"confidence": 0.8, "tags": []},
}


class FakeClock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def call(tools: SurveyTools, name: str, **kwargs) -> tuple[dict, bool]:
    text, is_error = tools.dispatch(name, {**DEFAULTS.get(name, {}), **kwargs})
    assert len(text) <= tools.limits.max_output_chars
    return json.loads(text), is_error


def pairs(point: dict) -> list[dict]:
    return [{"knob": k, "value": v} for k, v in point.items()]


def proposal(point: dict, *, region: str = "grok", seeds: int = 0, purpose: str = "boundary",
             hypothesis: str = "t_test_acc90 lies between its neighbours", falsified_if: str = "it does not") -> dict:
    return {"region": region, "point": pairs(point), "seeds": seeds, "purpose": purpose, "hypothesis": hypothesis,
            "falsified_if": falsified_if}


# ---------------------------------------------------------------------------------------------- the seeded survey
def _lightcurve(path: Path, final_acc: float) -> dict:
    step = np.unique(np.concatenate([[0], np.round(np.geomspace(1, 1000, 59))])).astype(np.int64)
    lc = {"step": step, "train_loss": 2.0 * np.exp(-step / 300.0) + 0.01,
          "test_loss": 2.2 * np.exp(-step / 400.0) + 0.05, "train_acc": np.minimum(1.0, step / 500.0),
          "test_acc": final_acc * np.minimum(1.0, step / 800.0), "param_norm": 10.0 + step / 100.0,
          "ntk_velocity": np.full(step.shape, np.nan)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **lc)
    return lc


def _observe(env, region: str, point: dict, seeds, *, t90, acc: float, label: str, score: float = 0.8,
             diverged: bool = False) -> list[str]:
    """Done objects at a point (t90: a number, a list per seed, or NaN) with a light curve and a rules-v1 label."""
    specs = [env.sky.make_spec(region, point, s) for s in seeds]
    env.cat.add_objects(specs, region=region, source="coarse")
    ids = []
    for i, spec in enumerate(specs):
        oid = spec.object_id()
        t = t90[i] if isinstance(t90, list) else t90
        summary = {"final_test_acc": acc, "final_train_acc": 1.0, "final_train_loss": 0.01, "final_test_loss": 0.2,
                   "t_test_acc90": t, "grok_delay": 100.0, "plateau_frac": 0.1, "n_drops": 1, "diverged": diverged,
                   "steps_total": 1000, "test_bump": 0.0}
        lc_path = env.paths.lightcurve_path(oid)
        lc = _lightcurve(lc_path, acc)
        env.cat.mark_done(RunResult(object_id=oid, status="done", light_curve=lc, summary=summary, wall_time_s=12.5,
                                    n_params=1000), lc_path)
        env.cat.add_classification(oid, {"label": label, "tags": [], "score": score,
                                         "reasons": [f"rules say {label}"]}, classifier="rules-v1")
        ids.append(oid)
    return ids


def _ri(env, oid: str, *, subset: tuple, matches_gt: bool | None, transient: str | None = None,
        severity: int = 3) -> None:
    names = ["x0", "x1", "yhat"]
    curve = RICurve(steps=[0, 500, 1000], names=names, subsets=[(0, 2), (0, 1, 2)], ri=np.zeros((2, 3)),
                    ri_plugin=np.zeros((2, 3)), p=np.ones((2, 3)), q=np.ones((2, 3)), n_eff=[4096.0] * 3,
                    meta={"alpha": 0.01, "min_effect": 1e-3, "max_order": 3})
    run_id = env.cat.add_ri_run(oid, "insitu", env.paths.ri_path(oid, "insitu"), curve)
    event = RIEvent(subset=subset, subset_names=tuple(names[i] for i in subset), order=len(subset),
                    kind="activation", step=500, step_lo=400, step_hi=500, ri_before=0.0, ri_after=0.6, peak_ri=0.69,
                    confidence=0.99, has_precursor=False, matches_ground_truth=matches_gt)
    event_ids = env.cat.add_events(run_id, oid, [event])
    if transient:
        env.cat.add_transients(oid, [Transient(transient, severity, f"{transient} of {subset}", 0)], event_ids)


@pytest.fixture
def env(tmp_path):
    clock = FakeClock(T0 - 3 * DAY)
    paths = Paths(tmp_path)
    cat = Catalog(paths.db, root=paths.root, clock=clock)
    e = SimpleNamespace(cat=cat, sky=Sky.from_dict(TEST_SKY), paths=paths, clock=clock)
    widths = [16, 64, 256]
    # three days ago: an exact power law t90 = 5000 * width^-0.5 at lr = 0.001, 3 seeds per width
    e.old = {w: _observe(e, "grok", {"width": w, "train.lr": 0.001}, range(3), t90=5000.0 * w ** -0.5, acc=0.97,
                         label="grokking") for w in widths}
    e.cat.add_log("agent", "nightly", "An earlier night", "## QC\n" + "All quiet. " * 600)
    e.cat.add_proposal("scheduler", "grok", {"width": 32, "train.lr": 0.01}, task="modular_addition", seeds=3,
                       rationale="boundary refinement", status="enqueued", n_enqueued=3, decided_at=clock.t)
    e.cat.heartbeat("w-gone", None, "stopped")
    # one hour ago: the new objects of tonight's window
    clock.t = T0 - 3600.0
    e.mem = _observe(e, "grok", {"width": 16, "train.lr": 0.01}, range(3), t90=math.nan, acc=0.3,
                     label="memorization")
    e.g64 = _observe(e, "grok", {"width": 64, "train.lr": 0.01}, range(3), t90=[237.5, 250.0, 262.5], acc=0.95,
                     label="grokking")
    e.g256 = _observe(e, "grok", {"width": 256, "train.lr": 0.01}, range(2), t90=[118.75, 125.0], acc=0.95,
                      label="grokking")
    e.unknown = _observe(e, "grok", {"width": 256, "train.lr": 0.01}, [2], t90=131.25, acc=0.9, label="unknown",
                         score=0.25)[0]
    e.ctrl = {k: _observe(e, "ctrl", {"k": k}, range(2), t90=300.0, acc=1.0, label="plateau_drop") for k in (2, 3)}
    _ri(e, e.ctrl[2][0], subset=(0, 1, 2), matches_gt=True, transient="pure_synergy")
    _ri(e, e.ctrl[2][1], subset=(0, 1, 2), matches_gt=True)
    _ri(e, e.ctrl[3][0], subset=(0, 2), matches_gt=False, transient="extinction", severity=2)
    e.dark = _observe(e, "dark", {"feature_prob": 0.01}, [0], t90=500.0, acc=0.8, label="smooth")[0]
    e.diverged = _observe(e, "dark", {"feature_prob": 0.1}, [0], t90=math.nan, acc=0.0, label="diverged",
                          diverged=True)[0]
    failing = [e.sky.make_spec("grok", {"width": 16, "train.lr": 0.001}, s) for s in (3, 4)]
    failing.append(e.sky.make_spec("ctrl", {"k": 2}, 2))
    e.cat.add_objects(failing[:2], region="grok", source="coarse")
    e.cat.add_objects(failing[2:], region="ctrl", source="coarse")
    for spec in failing[:2]:
        e.cat.mark_failed(spec.object_id(), "RuntimeError: shape mismatch\nTraceback (most recent call last): ...")
    e.cat.mark_failed(failing[2].object_id(), "ValueError: bad config")
    e.failed = [s.object_id() for s in failing]
    e.queued = e.sky.make_spec("grok", {"width": 64, "train.lr": 0.001}, 3).object_id()
    e.cat.add_objects([e.sky.make_spec("grok", {"width": 64, "train.lr": 0.001}, 3)], region="grok", source="refine")
    clock.t = T0 - 600.0
    e.cat.heartbeat("w-stale", e.queued, "running")
    clock.t = T0 - 30.0
    e.cat.heartbeat("w-alive", None, "idle")
    clock.t = T0
    yield e
    cat.close()


def make_tools(env, kind: str = "nightly", **limits) -> SurveyTools:
    return SurveyTools(env.cat, env.sky, env.paths, ToolLimits(**limits), kind=kind, window_start=T0 - DAY)


# ---------------------------------------------------------------------------------------------- schemas
def _walk(schema: dict, path: str = ""):
    """Yield (path, schema) of every schema node."""
    yield path, schema
    for name, sub in schema.get("properties", {}).items():
        yield from _walk(sub, f"{path}.{name}")
    if "items" in schema:
        yield from _walk(schema["items"], f"{path}[]")


UNSUPPORTED_KEYWORDS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
                        "maxLength", "pattern", "format", "minItems", "maxItems", "uniqueItems", "default", "oneOf",
                        "$ref", "not"}


def test_tool_specs_are_strict_with_every_property_required():
    names = [t["name"] for t in TOOL_SPECS]
    assert len(names) == len(set(names)) == 16
    for tool in TOOL_SPECS:
        assert tool["strict"] is True and tool["description"].strip()
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        for path, node in _walk(schema):
            assert not UNSUPPORTED_KEYWORDS & set(node), (tool["name"], path)
            if node.get("type") == "object":
                assert node["additionalProperties"] is False, (tool["name"], path)
                assert node["required"] == list(node["properties"]), (tool["name"], path)
        for name, prop in schema["properties"].items():
            assert prop.get("description", "").strip(), (tool["name"], name)


@pytest.mark.parametrize("kind", ["nightly", "weekly"])
def test_tool_sets_respect_the_api_limits_of_strict_schemas(kind):
    """At most 20 strict tools, no optional parameters, at most 16 union-typed parameters per request."""
    specs = tool_specs(kind)
    assert len(specs) <= 20
    unions = optional = 0
    for tool in specs:
        for _, node in _walk(tool["input_schema"]):
            unions += isinstance(node.get("type"), list) or "anyOf" in node
            if node.get("type") == "object":
                optional += len(set(node["properties"]) - set(node["required"]))
    assert optional == 0
    assert unions <= 16


def test_tool_sets_per_kind_and_copies():
    nightly = [t["name"] for t in tool_specs("nightly")]
    weekly = [t["name"] for t in tool_specs("weekly")]
    assert nightly == list(NIGHTLY_TOOLS) and weekly == list(WEEKLY_TOOLS)
    assert "write_report" not in nightly and "write_survey_log" in nightly and "propose_configs" in nightly
    assert "write_report" in weekly
    assert not {"propose_configs", "label_object", "request_followup", "write_survey_log"} & set(weekly)
    specs = tool_specs("nightly")
    specs[0]["description"] = "mutated"
    assert TOOL_SPECS[0]["description"] != "mutated"
    assert json.dumps(tool_specs("nightly")) == json.dumps(tool_specs("nightly"))
    with pytest.raises(ValueError):
        tool_specs("hourly")


def test_validate_input_reports_schema_violations():
    schema = next(t for t in TOOL_SPECS if t["name"] == "propose_configs")["input_schema"]
    good = {"proposals": [proposal({"width": 32, "train.lr": 0.001})]}
    assert validate_input(good, schema) == []
    bad = {"proposals": [{**proposal({"width": 32}), "seeds": True, "extra": 1}], "more": 2}
    errors = " | ".join(validate_input(bad, schema))
    assert "unexpected property 'more'" in errors
    assert "proposals[0].seeds: expected integer, got boolean" in errors
    assert "unexpected property 'extra'" in errors
    missing = {"proposals": [{k: v for k, v in proposal({"width": 32}).items() if k != "hypothesis"}]}
    assert any("missing property 'hypothesis'" in e for e in validate_input(missing, schema))
    wrong_enum = {"proposals": [{**proposal({"width": 32}), "purpose": "fun"}]}
    assert any("is not one of" in e for e in validate_input(wrong_enum, schema))
    value_types = {"proposals": [proposal({"width": [1, 2]})]}
    assert any("point[0].value" in e for e in validate_input(value_types, schema))
    assert validate_input(float("nan"), {"type": "number"}) and validate_input(True, {"type": "integer"})
    nullable = {"type": ["number", "null"]}
    assert validate_input(3, nullable) == [] and validate_input(None, nullable) == []


def test_dispatch_rejects_unknown_disallowed_and_malformed_calls(env):
    weekly = make_tools(env, "weekly")
    out, err = call(weekly, "propose_configs", proposals=[proposal({"width": 32, "train.lr": 0.001})])
    assert err and "unknown tool 'propose_configs' for a weekly run" in out["error"]
    out, err = call(weekly, "run_shell", command="rm -rf /")
    assert err and "unknown tool" in out["error"]
    text, err = weekly.dispatch("get_sky", ["not", "an", "object"])
    assert err and "must be a JSON object" in json.loads(text)["error"]
    out, err = call(weekly, "get_object", object_id="abc", include_lightcurve="yes")
    assert err and "include_lightcurve: expected boolean" in out["error"]
    assert weekly.call_counts["propose_configs"] == 1 and weekly.call_counts["get_object"] == 1


# ---------------------------------------------------------------------------------------------- QC and listings
def test_survey_overview_reports_qc_state(env):
    tools = make_tools(env)
    out, err = call(tools, "survey_overview")
    assert not err
    assert out["objects"]["done"] == 9 + 9 + 4 + 2 and out["objects"]["failed"] == 3 and out["queue"] == 1
    assert out["throughput"]["done_last_24h"] == 15 and out["throughput"]["done_prev_24h"] == 0
    first = out["failures_last_24h"][0]
    assert (first["error"], first["count"], first["regions"]) == ("RuntimeError: shape mismatch", 2, ["grok"])
    assert sorted(first["examples"]) == sorted(env.failed[:2])
    assert out["failures_last_24h"][1]["error"] == "ValueError: bad config"
    assert out["divergences_last_24h"] == {"dark": 1}
    workers = out["workers"]
    assert workers["alive"] == 1 and workers["silent"] == 1
    assert [w["worker_id"] for w in workers["stale_holding_object"]] == ["w-stale"]
    assert workers["stale_holding_object"][0]["object_id"] == env.queued
    assert {"free_gb", "total_gb", "min_free_gb", "low"} <= set(out["disk"])
    assert out["calibration"]["present"] is False
    assert out["caps"] == {"proposals_left_this_run": 50, "objects_left_today": 400, "followups_left_this_run": 20,
                           "labels_left_this_run": 200}
    flags = " | ".join(out["qc_flags"])
    for expected in ("3 objects failed", "divergences in the last 24 h: 1 in dark", "stale workers holding objects: 1",
                     "no calibration file",
                     "transients not yet acknowledged", "no catalog backup"):
        assert expected in flags
    assert "daemon down" not in flags
    assert "caps" not in make_tools(env, "weekly").overview()


def test_overview_reads_the_calibration_file_and_backups(env):
    calib = env.paths.root / "reports" / "calibration" / "calibration.json"
    calib.parent.mkdir(parents=True)
    calib.write_text(json.dumps({"noise_floor": [{"k": 3, "n": n, "q99": 1.0 / n} for n in range(1, 2000)],
                                 "min_detectable": {"3": 0.01}}))
    env.paths.backups.mkdir(parents=True)
    (env.paths.backups / "catalog-latest.sqlite").write_bytes(b"x")
    out = make_tools(env).overview()
    cal = out["calibration"]
    assert cal["present"] and cal["readable"] and cal["content"]["min_detectable"] == {"3": 0.01}
    assert len(json.dumps(cal["content"], separators=(",", ":"))) <= 6000 and "_truncated" in cal["content"]
    assert out["latest_backup_age_h"] < 1.0
    flags = " | ".join(out["qc_flags"])
    assert "calibration" not in flags and "backup" not in flags


def test_list_new_objects_window_filters_and_paging(env):
    tools = make_tools(env)
    out, err = call(tools, "list_new_objects")
    assert not err and out["total"] == 15 + 3
    assert out["label_counts"]["grokking"] == 5 and out["label_counts"]["unlabelled"] == 3
    oids = [o["object_id"] for o in out["objects"]]
    assert set(env.old[16]).isdisjoint(oids) and env.unknown in oids
    mem = next(o for o in out["objects"] if o["object_id"] == env.mem[0])
    assert mem["point"] == {"width": 16, "train.lr": 0.01} and mem["label"] == "memorization"
    assert mem["summary"]["t_test_acc90"] is None and mem["summary"]["diverged"] is False
    failed, err = call(tools, "list_new_objects", status="failed")
    assert failed["total"] == 3 and failed["objects"][0]["error"] == "RuntimeError: shape mismatch"
    assert "summary" not in failed["objects"][0]
    week, _ = call(tools, "list_new_objects", since_hours=96, region="grok", status="done")
    assert week["total"] == 18
    page1, _ = call(tools, "list_new_objects", since_hours=96, limit=5)
    page2, _ = call(tools, "list_new_objects", since_hours=96, limit=5, offset=5)
    assert page1["returned"] == page2["returned"] == 5
    assert not {o["object_id"] for o in page1["objects"]} & {o["object_id"] for o in page2["objects"]}
    for bad in ({"region": "nowhere"}, {"limit": 1000}, {"since_hours": -1}, {"offset": -3}):
        out, err = call(tools, "list_new_objects", **bad)
        assert err, bad


def test_get_object_with_lightcurve_excerpt_and_history(env):
    tools = make_tools(env)
    oid = env.ctrl[2][0]
    env.cat.add_classification(oid, {"label": "staged", "tags": [], "score": 0.6}, classifier="agent",
                               note="two drops")
    out, err = call(tools, "get_object", object_id=oid)
    assert not err
    assert out["point"] == {"k": 2} and out["summary"]["final_test_acc"] == 1.0
    assert [c["label"] for c in out["classifications"]] == ["staged", "plateau_drop"]
    assert out["ri_events"][0]["subset"] == ["x0", "x1", "yhat"] and out["ri_events"][0]["matches_gt"] is True
    assert out["transients"][0]["kind"] == "pure_synergy"
    lc = out["lightcurve"]
    assert lc["available"] and lc["n_points"] == len(np.load(env.paths.lightcurve_path(oid))["step"])
    steps = lc["excerpt"]["step"]
    assert steps[0] == 0 and steps[-1] == 1000 and len(steps) <= 26 and steps == sorted(steps)
    assert "ntk_velocity" not in lc["excerpt"] and len(lc["excerpt"]["train_loss"]) == len(steps)
    brief, _ = call(tools, "get_object", object_id=oid, include_lightcurve=False)
    assert "lightcurve" not in brief
    env.paths.lightcurve_path(oid).unlink()
    missing, err = call(tools, "get_object", object_id=oid)
    assert not err and missing["lightcurve"]["available"] is False
    out, err = call(tools, "get_object", object_id="0123456789abcdef")
    assert err and "unknown object id" in out["error"]


def test_unknown_queue_and_its_review(env):
    tools = make_tools(env)
    out, _ = call(tools, "list_unknown_queue")
    assert out["total"] == 1 and out["objects"][0]["object_id"] == env.unknown
    assert out["objects"][0]["classifier"] == "rules-v1" and out["objects"][0]["note"] == "rules say unknown"
    labelled, err = call(tools, "label_object", object_id=env.unknown, label="unknown",
                         note="late second drop at step 900; not like its siblings", confidence=0.5,
                         tags=["needs_replay"])
    assert not err and labelled["previous"] == {"label": "unknown", "classifier": "rules-v1"}
    out, _ = call(tools, "list_unknown_queue")
    assert out["objects"][0]["classifier"] == "agent" and out["objects"][0]["tags"] == ["needs_replay"]
    assert call(tools, "list_unknown_queue", only_unreviewed=True)[0]["total"] == 0
    call(tools, "label_object", object_id=env.unknown, label="grokking", note="test acc 0.9 after a delay")
    assert call(tools, "list_unknown_queue")[0]["total"] == 0


def test_list_transients_and_ri_events_filters(env):
    tools = make_tools(env)
    out, _ = call(tools, "list_transients")
    assert [t["kind"] for t in out["transients"]] == ["pure_synergy", "extinction"]
    assert out["transients"][0]["event"] == {"subset": ["x0", "x1", "yhat"], "order": 3, "kind": "activation",
                                             "step": 500}
    assert call(tools, "list_transients", min_severity=3)[0]["total"] == 1
    assert call(tools, "list_transients", kind="extinction")[0]["transients"][0]["object_id"] == env.ctrl[3][0]
    assert call(tools, "list_transients", min_severity=4)[1]
    events, _ = call(tools, "list_ri_events")
    assert events["total"] == 3 and events["by_order"] == {"2": 1, "3": 2}
    assert call(tools, "list_ri_events", ground_truth="match")[0]["total"] == 2
    assert call(tools, "list_ri_events", ground_truth="no_match")[0]["total"] == 1
    assert call(tools, "list_ri_events", min_order=3)[0]["total"] == 2
    one, _ = call(tools, "list_ri_events", object_id=env.ctrl[3][0], since_hours=0.001)
    assert one["total"] == 1 and one["events"][0]["object_id"] == env.ctrl[3][0]
    assert call(tools, "list_ri_events", since_hours=0.5)[0]["total"] == 0


def test_list_proposals_and_survey_log(env):
    tools = make_tools(env)
    out, _ = call(tools, "list_proposals", since_hours=96)
    assert out["proposals"][0]["source"] == "scheduler" and out["proposals"][0]["outcome"] == {"n_objects": 0}
    assert call(tools, "list_proposals")[0]["returned"] == 0
    log, _ = call(tools, "read_survey_log", since_hours=96, max_chars_per_entry=100)
    entry = log["entries"][0]
    assert entry["title"] == "An earlier night" and entry["chars"] > 6000 and len(entry["markdown"]) <= 100
    assert call(tools, "read_survey_log", since_hours=96, kind="weekly")[0]["returned"] == 0
    assert call(tools, "read_survey_log", max_chars_per_entry=50_000)[1]


# ---------------------------------------------------------------------------------------------- analysis
def test_query_summaries_groups_and_aggregates(env):
    tools = make_tools(env)
    out, err = call(tools, "query_summaries", region="grok", group_by=["train.lr"],
                    metrics=["t_test_acc90", "final_test_acc"])
    assert not err and out["n_objects"] == 18 and [g["group"] for g in out["groups"]] == [{"train.lr": 0.001},
                                                                                         {"train.lr": 0.01}]
    fast = out["groups"][1]
    t90 = [237.5, 250.0, 262.5, 118.75, 125.0, 131.25]
    assert fast["metrics"]["t_test_acc90"]["n"] == 6
    assert fast["metrics"]["t_test_acc90"]["median"] == pytest.approx(float(np.median(t90)))
    assert fast["metrics"]["t_test_acc90"]["std"] == pytest.approx(float(np.std(t90, ddof=1)), rel=1e-5)
    assert fast["labels"] == {"grokking": 5, "memorization": 3, "unknown": 1}
    points, _ = call(tools, "query_summaries", region="grok", group_by=["point"], metrics=["final_test_acc"],
                     where=[{"knob": "train.lr", "value": 0.01}])
    assert [g["group"]["point"]["width"] for g in points["groups"]] == [16, 64, 256]
    labelled, _ = call(tools, "query_summaries", group_by=["region"], metrics=["wall_time_s"], label="grokking")
    assert labelled["groups"] == [{"group": {"region": "grok"}, "n": 14, "labels": {"grokking": 14},
                                   "metrics": {"wall_time_s": {"n": 14, "mean": 12.5, "median": 12.5, "std": 0.0,
                                                               "min": 12.5, "max": 12.5}}}]
    for bad in ({"group_by": ["width"]}, {"region": "grok", "group_by": ["colour"]}, {"metrics": []},
                {"metrics": ["final_test_acc", "final_test_acc"]}, {"label": "boring"},
                {"where": [{"knob": "width", "value": 16}]}, {"metrics": ["accuracy"]},
                {"region": "grok", "where": [{"knob": "train.lr", "value": 0.0011}]},
                {"region": "grok", "where": [{"knob": "width", "value": 16}, {"knob": "width", "value": 64}]}):
        out, err = call(tools, "query_summaries", **bad)
        assert err, bad


def test_fit_power_law_recovers_a_planted_exponent(env):
    tools = make_tools(env)
    out, err = call(tools, "fit_power_law", region="grok", metric="t_test_acc90", knob="width",
                    where=[{"knob": "train.lr", "value": 0.001}])
    assert not err
    fit = out["fit"]
    assert fit["slope"] == pytest.approx(-0.5, abs=1e-9) and fit["prefactor"] == pytest.approx(5000.0, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0) and fit["n"] == 3 and fit["n_distinct_x"] == 3
    assert fit["slope_ci"][0] == pytest.approx(-0.5, abs=1e-6) and fit["slope_ci"][1] == pytest.approx(-0.5, abs=1e-6)
    assert out["n_objects"] == 9 and out["n_points"] == 3 and out["min_seeds_per_point"] == 3
    assert out["other_knobs_varying"] == [] and out["excluded"] == {}
    assert out["caveats"][-1].startswith("a candidate only")
    assert len(out["caveats"]) == 1
    per_object, _ = call(tools, "fit_power_law", region="grok", metric="t_test_acc90", knob="width",
                         where=[{"knob": "train.lr", "value": 0.001}], aggregate="objects")
    assert per_object["fit"]["n"] == 9 and per_object["fit"]["slope"] == pytest.approx(-0.5, abs=1e-9)


def test_fit_power_law_reports_exclusions_and_caveats(env):
    tools = make_tools(env)
    out, err = call(tools, "fit_power_law", region="grok", metric="t_test_acc90", knob="width",
                    where=[{"knob": "train.lr", "value": 0.01}], aggregate="objects")
    assert not err and out["excluded"] == {"metric missing or not finite": 3}
    assert out["fit"]["n_distinct_x"] == 2 and out["fit"]["jackknife"] is None
    caveats = " | ".join(out["caveats"])
    assert "only 2 distinct knob values" in caveats and "0.60 decades" in caveats and "censored" in caveats
    pooled, _ = call(tools, "fit_power_law", region="grok", metric="t_test_acc90", knob="width")
    assert pooled["other_knobs_varying"] == ["train.lr"] and "confounding" in " | ".join(pooled["caveats"])
    none, err = call(tools, "fit_power_law", region="ctrl", metric="t_test_acc90", knob="k")
    assert not err and none["fit"] is None and "need >= 3 observations" in none["reason"]
    for bad in ({"region": "grok", "knob": "depth"}, {"region": "nowhere", "knob": "width"},
                {"region": "grok", "knob": "width", "where": [{"knob": "width", "value": 16}]}):
        out, err = call(tools, "fit_power_law", metric="t_test_acc90", **bad)
        assert err, bad


def test_power_law_fit_matches_ols_theory():
    rng = np.random.default_rng(0)
    x = np.repeat([10.0, 30.0, 100.0, 300.0, 1000.0], 4)
    y = 7.0 * x ** 0.75 * np.exp(rng.normal(0.0, 0.1, x.size))
    fit = power_law_fit(x, y)
    ref = stats.linregress(np.log10(x), np.log10(y))
    assert fit["slope"] == pytest.approx(ref.slope) and fit["slope_se"] == pytest.approx(ref.stderr)
    assert fit["intercept_log10"] == pytest.approx(ref.intercept) and fit["r2"] == pytest.approx(ref.rvalue ** 2)
    t = stats.t.ppf(0.975, x.size - 2)
    assert fit["slope_ci"] == pytest.approx([ref.slope - t * ref.stderr, ref.slope + t * ref.stderr])
    assert fit["slope_ci"][0] < 0.75 < fit["slope_ci"][1]
    jack = fit["jackknife"]
    assert len(jack["slope_without_x"]) == 5 and jack["min"] <= fit["slope"] <= jack["max"]
    assert fit["curvature"]["p_value"] > 0.05
    lx = np.log10(x)
    curved = 10 ** (1.0 + 0.5 * lx + 0.4 * (lx - lx.mean()) ** 2) * np.exp(rng.normal(0.0, 0.01, x.size))
    curvature = power_law_fit(x, curved)["curvature"]
    assert curvature["coef"] == pytest.approx(0.4, abs=0.02) and curvature["p_value"] < 1e-6
    for bad in (([1.0, 2.0], [1.0, 2.0]), ([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, 3.0], [1.0, -2.0, 3.0])):
        with pytest.raises(ValueError):
            power_law_fit(*bad)


def test_objects_of_another_sky_version_are_not_read_as_points_of_this_sky(env):
    """Fixed and training settings may differ between sky versions, so knob analyses skip older objects."""
    old_sky = Sky.from_dict({**TEST_SKY, "version": "agent-test-old"})
    spec = old_sky.make_spec("grok", {"width": 64, "train.lr": 0.001}, 0)
    env.cat.add_objects([spec], region="grok", source="coarse")
    env.cat.mark_done(RunResult(object_id=spec.object_id(), status="done", light_curve={"step": np.arange(3)},
                                summary={"t_test_acc90": 1.0e6, "final_test_acc": 0.5}), "lc.npz")
    tools = make_tools(env)
    out, _ = call(tools, "query_summaries", region="grok", group_by=["point"], metrics=["t_test_acc90"])
    assert out["excluded_not_on_current_sky"] == 1 and out["n_objects"] == 18
    everything, _ = call(tools, "query_summaries", region="grok", metrics=["t_test_acc90"])
    assert everything["n_objects"] == 19  # no knob interpretation needed: every done object counts
    fit, _ = call(tools, "fit_power_law", region="grok", metric="t_test_acc90", knob="width",
                  where=[{"knob": "train.lr", "value": 0.001}])
    assert fit["excluded"] == {"not a point of the current sky": 1} and fit["fit"]["slope"] == pytest.approx(-0.5)
    assert call(tools, "get_sky", region="grok")[0]["observed_points"] == 6
    assert call(tools, "get_object", object_id=spec.object_id(), include_lightcurve=False)[0]["point"] is None


def test_get_sky_lists_regions_and_boundaries(env):
    tools = make_tools(env)
    out, _ = call(tools, "get_sky")
    assert [r["name"] for r in out["regions"]] == ["grok", "ctrl", "dark"]
    grok = out["regions"][0]
    assert grok["grid_size"] == 6 and grok["objects"] == {"done": 18, "failed": 2, "queued": 1, "total": 21}
    assert grok["knobs"]["width"] == {"values": [16, 64, 256], "kind": "log", "refine": True, "bounds": [16, 256]}
    detail, _ = call(tools, "get_sky", region="grok")
    assert detail["observed_points"] == 6 and detail["points_with_3_or_more_done_seeds"] == 6
    assert detail["labels_of_done"] == {"grokking": 14, "memorization": 3, "unknown": 1}
    pairs_ = [(p["a"], p["b"]) for p in detail["boundary_pairs"]]
    assert ({"width": 16, "train.lr": 0.01}, {"width": 64, "train.lr": 0.01}) in pairs_
    assert detail["fixed"] == {"depth": 1} and detail["train"] == {"steps": 1000}
    assert call(tools, "get_sky", region="nowhere")[1]


# ---------------------------------------------------------------------------------------------- writes
def test_propose_configs_enqueues_validated_points(env):
    tools = make_tools(env)
    point = {"width": 32, "train.lr": 0.01}
    out, err = call(tools, "propose_configs", proposals=[proposal(point, hypothesis="memorization ends near 32")])
    assert not err and out["accepted"] == 1 and out["rejected"] == 0
    res = out["results"][0]
    expected = [env.sky.make_spec("grok", point, s).object_id() for s in range(3)]
    assert res["status"] == "enqueued" and res["n_enqueued"] == 3 and res["object_ids"] == expected
    rows = env.cat.query("SELECT object_id, source, priority, status, created_at FROM objects WHERE source = 'agent'")
    assert sorted(r["object_id"] for r in rows) == sorted(expected)
    assert all(r["priority"] == 1.5 and r["status"] == "queued" for r in rows)
    prop = env.cat.query("SELECT * FROM proposals WHERE source = 'agent'")[0]
    assert prop["status"] == "enqueued" and prop["n_enqueued"] == 3 and prop["seeds"] == 3
    assert prop["point_json"] == point_key(point)
    assert "memorization ends near 32" in prop["rationale"] and "falsified if: it does not" in prop["rationale"]
    assert out["caps"]["proposals_left_this_run"] == 49 and out["caps"]["objects_left_today"] == 397
    again, err = call(tools, "propose_configs", proposals=[proposal(point)])
    assert not err and again["results"][0]["status"] == "already_in_catalog" and again["accepted"] == 0
    more, _ = call(tools, "propose_configs", proposals=[proposal({"width": 64, "train.lr": 0.01}, seeds=5,
                                                                 purpose="replication")])
    assert more["results"][0]["n_enqueued"] == 2 and more["results"][0]["n_existing"] == 3
    listed, _ = call(tools, "list_proposals", source="agent")
    outcome = listed["proposals"][-1]["outcome"]
    assert outcome["by_status"] == {"queued": 3} and outcome["n_objects"] == 3


@pytest.mark.parametrize("item, message", [
    (proposal({"width": 64, "train.lr": 0.005}), "not a grid value"),
    (proposal({"width": 1000, "train.lr": 0.01}), "outside the bounds"),
    (proposal({"width": 32.5, "train.lr": 0.01}), "not an integer"),
    (proposal({"width": 32}), "missing knob 'train.lr'"),
    (proposal({"width": 32, "train.lr": 0.01, "depth": 2}), "unknown knob 'depth'"),
    (proposal({"width": 32, "train.lr": 0.01}, region="nowhere"), "unknown region"),
    (proposal({"width": 32, "train.lr": 0.01}, seeds=11), "seeds must be"),
    (proposal({"width": 32, "train.lr": 0.01}, hypothesis="  "), "hypothesis must not be empty"),
    ({**proposal({"width": 32, "train.lr": 0.01}),
      "point": pairs({"width": 32, "train.lr": 0.01}) + [{"knob": "width", "value": 48}]}, "given twice"),
])
def test_propose_configs_rejects_invalid_proposals(env, item, message):
    tools = make_tools(env)
    n_objects = env.cat.stats()["objects"]["total"]
    out, err = call(tools, "propose_configs", proposals=[item])
    assert err and out["error"] == "no proposal was accepted"
    assert message in " | ".join(out["results"][0]["errors"])
    assert env.cat.stats()["objects"]["total"] == n_objects and tools.n_proposals == 0
    assert not env.cat.query("SELECT 1 FROM proposals WHERE source = 'agent'")


def test_propose_configs_partial_success_and_caps(env):
    tools = make_tools(env, max_proposals_per_run=2, max_objects_per_day=7)
    batch = [proposal({"width": 32, "train.lr": 0.01}), proposal({"width": 64, "train.lr": 0.005}),
             proposal({"width": 128, "train.lr": 0.01}), proposal({"width": 48, "train.lr": 0.01})]
    out, err = call(tools, "propose_configs", proposals=batch)
    assert not err and out["accepted"] == 2 and out["rejected"] == 2
    assert [r["status"] for r in out["results"]] == ["enqueued", "rejected", "enqueued", "rejected"]
    assert "per-run proposal cap reached (2)" in out["results"][3]["errors"][0]
    fresh = make_tools(env, max_objects_per_day=7)  # a new run on the same day: 6 agent objects already
    out, err = call(fresh, "propose_configs", proposals=[proposal({"width": 48, "train.lr": 0.01})])
    assert err and "daily object cap" in out["results"][0]["errors"][0] and "only 1 of 7" in \
        out["results"][0]["errors"][0]
    out, err = call(fresh, "propose_configs", proposals=[proposal({"width": 48, "train.lr": 0.01}, seeds=1)])
    assert not err and out["results"][0]["n_enqueued"] == 1
    env.clock.t += DAY + 1  # the agent's objects of yesterday no longer count
    assert make_tools(env, max_objects_per_day=7).caps()["objects_left_today"] == 7
    assert call(tools, "propose_configs", proposals=[])[1]
    assert call(tools, "propose_configs", proposals=[proposal({"width": 32, "train.lr": 0.01})] * 21)[1]


def test_label_object_validation_and_cap(env):
    tools = make_tools(env, max_labels_per_run=1)
    out, err = call(tools, "label_object", object_id=env.g64[0], label="staged", note="two drops", confidence=0.7,
                    tags=["second_drop"])
    assert not err and out["previous"] == {"label": "grokking", "classifier": "rules-v1"}
    latest = env.cat.latest_classification(env.g64[0])
    assert latest["label"] == "staged" and latest["classifier"] == "agent" and latest["note"] == "two drops"
    assert latest["score"] == 0.7 and latest["tags"] == ["second_drop"]
    assert "per-run label cap" in call(tools, "label_object", object_id=env.g64[1], label="staged",
                                       note="same")[0]["error"]
    fresh = make_tools(env)
    for bad, message in (({"object_id": env.failed[0]}, "only done objects"),
                         ({"object_id": "ffffffffffffffff"}, "unknown object"),
                         ({"note": " "}, "note must not be empty"),
                         ({"confidence": 1.5}, "confidence must be in [0, 1]"),
                         ({"tags": ["Not OK"]}, "lowercase identifiers"),
                         ({"label": "interesting"}, "is not one of")):
        args = {"object_id": env.g64[1], "label": "staged", "note": "n", **bad}
        out, err = call(fresh, "label_object", **args)
        assert err and message in out["error"], bad
    assert fresh.n_labels == 0


def test_request_followup_rules_and_cap(env):
    tools = make_tools(env, max_followups_per_run=1)
    out, err = call(tools, "request_followup", object_id=env.unknown, reason="does RI show a late 3-way term?")
    assert not err and out["requested"] is True
    row = env.cat.query("SELECT * FROM followups WHERE object_id = ?", (env.unknown,))[0]
    assert row["requested_by"] == "agent" and row["priority"] == AGENT_FOLLOWUP_PRIORITY
    assert row["status"] == "requested" and row["reason"] == "does RI show a late 3-way term?"
    again, err = call(tools, "request_followup", object_id=env.unknown, reason="again")
    assert not err and again["requested"] is False and again["existing"]["requested_by"] == "agent"
    capped, err = call(tools, "request_followup", object_id=env.g64[0], reason="boundary member")
    assert err and "per-run follow-up cap reached (1)" in capped["error"]
    fresh = make_tools(env)
    for oid, message in ((env.failed[0], "only done objects"), (env.dark, "ri_policy 'never'"),
                         ("ffffffffffffffff", "unknown object")):
        out, err = call(fresh, "request_followup", object_id=oid, reason="r")
        assert err and message in out["error"]


def test_write_survey_log_exactly_once(env):
    tools = make_tools(env, max_log_chars=100)
    for bad, message in (({"title": "", "markdown": "x"}, "title must not be empty"),
                         ({"title": "t" * 201, "markdown": "x"}, "title is 201 characters"),
                         ({"title": "t", "markdown": "  "}, "markdown must not be empty"),
                         ({"title": "t", "markdown": "m" * 101}, "the limit is 100")):
        out, err = call(tools, "write_survey_log", **bad)
        assert err and message in out["error"]
    out, err = call(tools, "write_survey_log", title="Night  of\nthe 21st", markdown="## QC\nquiet\n")
    assert not err and out["entry_id"] == tools.log_entry_id
    entry = env.cat.query("SELECT * FROM survey_log WHERE entry_id = ?", (tools.log_entry_id,))[0]
    assert (entry["author"], entry["kind"], entry["title"], entry["markdown"]) == ("agent", "nightly",
                                                                                  "Night of the 21st", "## QC\nquiet")
    out, err = call(tools, "write_survey_log", title="again", markdown="again")
    assert err and "may be called once per run" in out["error"]


def test_write_report_writes_a_file_and_a_log_entry(env):
    tools = make_tools(env, "weekly")
    taken = env.paths.reports / "weekly" / f"{iso_week(T0)}.md"
    taken.parent.mkdir(parents=True)
    taken.write_text("a human's notes")
    out, err = call(tools, "write_report", title="Week 38", markdown="## Summary\nGrokking everywhere.")
    assert not err and out["path"] == f"reports/weekly/{iso_week(T0)}-2.md"
    assert taken.read_text() == "a human's notes"
    assert tools.report_path.read_text() == "# Week 38\n\n## Summary\nGrokking everywhere.\n"
    entry = env.cat.query("SELECT author, kind, title FROM survey_log WHERE entry_id = ?", (out["entry_id"],))[0]
    assert entry == {"author": "agent", "kind": "weekly", "title": "Week 38"}
    out, err = call(tools, "write_report", title="Week 38 again", markdown="x")
    assert err and "may be called once per run" in out["error"]


# ---------------------------------------------------------------------------------------------- outputs
def test_outputs_are_truncated_to_valid_json(env):
    tools = make_tools(env, max_output_chars=2000)
    text, err = tools.dispatch("list_new_objects", {**DEFAULTS["list_new_objects"], "since_hours": 96,
                                                    "limit": 200})
    assert not err and len(text) <= 2000
    out = json.loads(text)
    assert out["total"] == 27 and out["returned"] == 27
    omitted = out["_truncated"]["omitted_items"]["objects"]
    assert len(out["objects"]) + omitted == 27 and len(out["objects"]) >= 1


def test_fit_json_shrinks_the_largest_list_and_keeps_small_outputs():
    small = {"a": [1, 2, 3], "b": "x"}
    assert fit_json(small, 1000) == '{"a":[1,2,3],"b":"x"}'
    big = {"head": "keep", "rows": [{"i": i, "pad": "x" * 50} for i in range(500)], "tags": list(range(10))}
    text = fit_json(big, 3000)
    out = json.loads(text)
    assert len(text) <= 3000 and out["head"] == "keep" and out["tags"] == list(range(10))
    assert [r["i"] for r in out["rows"]] == list(range(len(out["rows"])))
    assert out["_truncated"]["omitted_items"] == {"rows": 500 - len(out["rows"])}
    nested = fit_json([[float("nan"), np.float64(1.234567891)], list(range(2000))], 1000)
    data = json.loads(nested)
    assert len(nested) <= 1000 and data["result"][0] == [None, 1.23457]
    assert fit_json({"s": "y" * 5000}, 1000).endswith("[cut: output exceeded 1000 characters]")


def test_period_summary_and_determinism(env):
    tools = make_tools(env)
    summary = tools.period_summary(T0 - DAY)
    assert summary["objects_finished"]["grok"] == {"done": 9, "failed": 2}
    assert summary["labels_of_done"]["memorization"] == 3
    assert summary["positive_controls"] == {"ctrl": {"objects_with_ri": 3, "with_ground_truth_match": 2}}
    assert {(t["kind"], t["severity"], t["n"]) for t in summary["transients"]} == {("pure_synergy", 3, 1),
                                                                                  ("extinction", 2, 1)}
    assert summary["survey_log"] == []
    other = make_tools(env)
    for name, args in (("survey_overview", {}), ("get_sky", {"region": "grok"}),
                       ("query_summaries", {"region": "grok", "group_by": ["point"]})):
        a = tools.dispatch(name, {**DEFAULTS.get(name, {}), **args})
        b = other.dispatch(name, {**DEFAULTS.get(name, {}), **args})
        if name == "survey_overview":  # disk usage may change between calls
            a, b = (json.loads(a[0]), a[1]), (json.loads(b[0]), b[1])
            a[0].pop("disk"), b[0].pop("disk")
        assert a == b
