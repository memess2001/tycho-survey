"""Tests of tycho.survey.dashboard: the status dict (counts, throughput, labels per task, latest transients and
events, workers) and the self-contained HTML dashboard."""
from __future__ import annotations

import json
import math
import re

import pytest
from survey_env import make_root, synthetic_catalog

from tycho.config import load_config
from tycho.survey.catalog import Catalog
from tycho.survey.dashboard import STATUS_FORMAT, collect_status, render_dashboard, write_dashboard, write_json_atomic


@pytest.fixture
def env(tmp_path):
    paths = make_root(tmp_path / "root")
    catalog, done = synthetic_catalog(paths)
    yield paths, catalog, done
    catalog.close()


def test_collect_status_counts(env):
    paths, catalog, done = env
    status = collect_status(catalog, paths, load_config(paths.root))
    assert status["format"] == STATUS_FORMAT and status["root"] == str(paths.root)
    assert status["objects"] == {"queued": 2, "running": 0, "done": 5, "failed": 1, "interrupted": 0, "total": 8}
    assert status["queue"] == 2 and status["by_region"]["ctrl"]["done"] == 4
    assert status["ri_runs"] == 1 and status["ri_events"] == 1 and status["transients"]["total"] == 2
    tp = status["throughput"]
    assert tp["done_last_24h"] == 5 and tp["done_last_1h"] == 5 and tp["failed_last_24h"] == 1
    assert len(tp["hourly"]["done"]) == 24 and sum(tp["hourly"]["done"]) == 5 and tp["hourly"]["done"][-1] == 5
    assert tp["median_wall_time_s_24h"] == pytest.approx(3.5)  # 1.5 .. 5.5
    by_task = status["labels_by_task"]
    assert set(by_task) == {"sparse_parity", "modular_addition"} and sum(by_task["sparse_parity"].values()) == 4
    assert [t["kind"] for t in status["latest_transients"]] == ["loss_spike", "pure_synergy"]  # newest first
    event = status["latest_events"][0]
    assert event["subset"] == ["x0", "x1", "yhat"] and event["run_kind"] == "insitu" and event["order_k"] == 3
    assert [w["worker_id"] for w in status["workers"]] == ["w0"] and status["workers"][0]["stale"] is False
    assert status["disk"]["free_gb"] > 0 and status["latest_backup"] is None
    assert status["compute_hours_done"] == pytest.approx(sum(1.5 + i for i in range(5)) / 3600.0)
    json.dumps(status, allow_nan=False)  # strict JSON


def test_extra_entries_and_stale_workers(env):
    paths, catalog, done = env
    status = collect_status(catalog, paths, {"survey": {"stale_after_s": 0.0}}, extra={"state": "running"})
    assert status["state"] == "running" and status["workers"][0]["stale"] is True


def test_dashboard_html(env):
    paths, catalog, done = env
    path = write_dashboard(catalog, paths)
    assert path == paths.reports / "dashboard.html"
    page = path.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert "<script" not in page and "http://" not in page and "https://" not in page  # self-contained
    assert '<meta http-equiv="refresh" content="30">' in page  # status_every_s = 1 s, floored at 30 s
    tiles = dict(re.findall(r'<div class="label">([^<]+)</div><div class="value">([^<]+)</div>', page))
    assert tiles["Done"] == "5" and tiles["Queued"] == "2" and tiles["Failed"] == "1"
    assert tiles["RI events"] == "1" and tiles["Transients"] == "2"
    assert "of 8 objects" in page
    for text in ("Objects finished per hour", "Classification by task", "sparse_parity", "modular_addition",
                 "Latest transients", "pure_synergy", "Latest RI events", "{x0, x1, yhat}", "Workers", "w0",
                 "Regions", "ctrl", "grok"):
        assert text in page
    assert "pure synergy of &lt;x0, x1, yhat&gt; &amp; friends" in page  # messages are escaped
    assert "<x0" not in page


def test_dashboard_of_an_empty_catalog(tmp_path):
    paths = make_root(tmp_path / "root")
    with Catalog(paths.db, root=paths.root) as catalog:
        page = write_dashboard(catalog, paths).read_text(encoding="utf-8")
    for text in ("No classified objects yet.", "No transients yet.", "No RI events yet.", "No worker has reported yet.",
                 "The catalog is empty"):
        assert text in page


def test_daemon_section_is_rendered(env):
    paths, catalog, done = env
    status = collect_status(catalog, paths, extra={"state": "running", "daemon": {
        "pid": 4242, "uptime_s": 7200.0, "n_workers": 8, "sky_version": "tiny-v0", "code_version": "0.1.0+abc"}})
    page = render_dashboard(status)
    assert "daemon running (pid 4242, up 2.0 h, 8 workers, sky tiny-v0, code 0.1.0+abc)" in page


def test_write_json_atomic_is_strict(tmp_path):
    path = write_json_atomic(tmp_path / "reports" / "status.json", {"a": math.nan, "b": (1, math.inf), "c": "x"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": None, "b": [1, None], "c": "x"}
    assert not list(path.parent.glob("*.tmp"))
