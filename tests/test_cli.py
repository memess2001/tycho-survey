"""Tests of tycho.cli: subprocess smoke runs of init, run-object, survey status, release, dashboard and
agent nightly --dry-run; in-process tests of value parsing, init fallbacks, enqueue, follow-ups, backup, calibrate
(with the suite mocked), survey run and exit codes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from survey_env import make_root, synthetic_catalog

import tycho.calibration as calibration_pkg
import tycho.cli as cli
import tycho.survey.daemon as daemon_mod
from tycho.cli import SKY_V0_YAML, main, parse_assignments, parse_value
from tycho.config import DEFAULT_CONFIG, load_config
from tycho.survey.catalog import Catalog
from tycho.survey.daemon import SurveyLock
from tycho.survey.sky import Sky

REPO = Path(__file__).resolve().parents[1]


def run_cli(*args: str, root: Path | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    """``python -m tycho [--root ROOT] ARGS`` in a subprocess."""
    cmd = [sys.executable, "-m", "tycho"] + (["--root", str(root)] if root is not None else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(autouse=True, scope="module")
def _torch_settings():
    """run-object configures torch process-wide (threads, deterministic algorithms); undo it afterwards."""
    import torch

    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1], warn_only=before[2])


@pytest.fixture
def synthetic_root(tmp_path):
    paths = make_root(tmp_path / "root")
    catalog, done = synthetic_catalog(paths)
    catalog.close()
    return paths, done


# ---------------------------------------------------------------------------------------------- subprocess smoke
def test_init_creates_a_survey_root_and_is_idempotent(tmp_path):
    root = tmp_path / "survey"
    r = run_cli("init", root=root)
    assert r.returncode == 0, r.stderr
    for d in ("data", "data/lightcurves", "data/ri", "data/work", "reports", "logs", "releases", "backups"):
        assert (root / d).is_dir(), d
    assert (root / "configs" / "sky_v0.yaml").read_text() == (REPO / "configs" / "sky_v0.yaml").read_text()
    assert (root / "tycho.yaml").read_text() == (REPO / "configs" / "tycho.example.yaml").read_text()
    assert (root / "data" / "catalog.sqlite").is_file()
    assert "sky-v0: 6 regions" in r.stdout and "created" in r.stdout
    cfg = load_config(root)
    assert cfg["survey"]["sky"] == "configs/sky_v0.yaml" and "min_effect" not in (
        yaml.safe_load((root / "tycho.yaml").read_text()).get("ri") or {})  # the calibration sets it
    (root / "tycho.yaml").write_text("survey: {n_workers: 3}\n")
    again = run_cli("init", root=root)
    assert again.returncode == 0 and again.stdout.count("kept") == 2
    assert (root / "tycho.yaml").read_text() == "survey: {n_workers: 3}\n"  # never overwritten


def test_run_object_prints_a_summary_with_ri_events(tmp_path, capsys):
    paths = make_root(tmp_path / "root")
    args = ["run-object", "--task", "sparse_parity", "--seed", "0", "--ri", "--set", "n=8", "--set", "k=2",
            "--set", "width=16", "--set", "test_size=256", "--train", "steps=300", "--train", "batch_size=16",
            "--train", "optimizer=sgd", "--train", "lr=5e-2", "--train", "n_log_evals=12", "--train",
            "n_lin_evals=6"]
    r = run_cli(*args, "--json", "--out", str(tmp_path / "out"), root=paths.root)
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["status"] == "done" and report["steps"] == 300 and report["task_cfg"]["width"] == 16
    assert report["classification"]["label"] and report["ri"]["n_subsets"] > 0
    assert report["ri"]["min_effect"] == pytest.approx(1e-3) and isinstance(report["events"], list)
    assert {"lightcurve.npz", "ricurve.npz", "summary.json"} <= {p.name for p in (tmp_path / "out").iterdir()}
    assert main(["--root", str(paths.root), *args]) == 0  # the text form (in process: torch is loaded already)
    text = capsys.readouterr().out
    assert f"object {report['object_id']}: sparse_parity seed 0, 300 steps: done" in text
    assert "classification:" in text and f"RI events: {len(report['events'])}" in text
    with Catalog(paths.db) as catalog:  # ad-hoc objects are not recorded
        assert catalog.stats()["objects"]["total"] == 0


def test_survey_status_release_dashboard_and_agent_dry_run(synthetic_root):
    paths, done = synthetic_root
    r = run_cli("survey", "status", root=paths.root)
    assert r.returncode == 0, r.stderr
    assert "objects:     8 total: 5 done, 2 queued, 0 running, 0 interrupted, 1 failed" in r.stdout
    assert "daemon:      not running" in r.stdout and "w0" in r.stdout and "pure_synergy" in r.stdout
    status = json.loads(run_cli("survey", "status", "--json", root=paths.root).stdout)
    assert status["objects"]["done"] == 5 and status["daemon_process"]["running"] is False

    r = run_cli("release", "--tag", "DR0", root=paths.root)
    assert r.returncode == 0, r.stderr
    assert "manifest verified" in r.stdout and (paths.releases / "DR0" / "objects.parquet").is_file()
    assert run_cli("release", "--tag", "DR0", root=paths.root).returncode == 2  # releases are immutable

    r = run_cli("dashboard", root=paths.root)
    assert r.returncode == 0, r.stderr
    page = (paths.reports / "dashboard.html").read_text(encoding="utf-8")
    assert "Tycho survey" in page and "pure_synergy" in page
    assert not (paths.reports / "status.json").exists()  # only the daemon writes status.json (healthcheck)

    r = run_cli("agent", "nightly", "--dry-run", root=paths.root)
    assert r.returncode == 0, r.stderr
    request = json.loads((paths.reports / "agent" / "dryrun-nightly.json").read_text(encoding="utf-8"))
    assert "dry run" in r.stdout and request


# ---------------------------------------------------------------------------------------------- in process
def test_parse_value_and_assignments():
    assert parse_value("128") == 128 and parse_value("0.5") == 0.5 and parse_value("1e-3") == 1e-3
    assert parse_value("-2.5E+2") == -250.0 and parse_value("null") is None and parse_value("") is None
    assert parse_value("relu") == "relu" and parse_value("[1, 2]") == [1, 2] and parse_value("true") is True
    assert parse_value("1:1.0;1,3:0.8") == "1:1.0;1,3:0.8" and parse_value("{a: 1}") == {"a": 1}
    assert parse_value("[unclosed") == "[unclosed"
    assert parse_assignments(["n=8", "act = relu", "n=9"], "--set") == {"n": 9, "act": "relu"}
    for bad in (["n"], ["=3"]):
        with pytest.raises(cli.CLIError):
            parse_assignments(bad, "--set")


def test_embedded_sky_matches_the_repository_file():
    assert SKY_V0_YAML == (REPO / "configs" / "sky_v0.yaml").read_text(encoding="utf-8")


def test_init_without_the_source_checkout_uses_the_fallbacks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path / "no-checkout")
    root = tmp_path / "survey"
    assert main(["--root", str(root), "init"]) == 0
    assert (root / "configs" / "sky_v0.yaml").read_text(encoding="utf-8") == SKY_V0_YAML
    assert Sky.from_yaml(root / "configs" / "sky_v0.yaml").version == "sky-v0"
    written = yaml.safe_load((root / "tycho.yaml").read_text(encoding="utf-8"))
    assert "min_effect" not in written["ri"]  # an explicit value would override the calibration
    cfg = load_config(root)
    assert {k: v for k, v in cfg.items()} == DEFAULT_CONFIG
    assert "created" in capsys.readouterr().out


def test_usage_errors_exit_with_2(tmp_path, capsys):
    assert main(["--root", str(tmp_path / "nothing"), "survey", "status"]) == 2
    assert "no catalog" in capsys.readouterr().err and not (tmp_path / "nothing").exists()
    paths = make_root(tmp_path / "root")
    assert main(["--root", str(paths.root), "run-object", "--task", "no_such_task", "--seed", "0"]) == 2
    assert "unknown task" in capsys.readouterr().err
    assert main(["--root", str(paths.root), "run-object", "--task", "sparse_parity", "--seed", "0",
                 "--train", "stepz=3"]) == 2
    assert "unknown TrainConfig fields" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["survey"])  # argparse: a subcommand is required
    assert exc.value.code == 2


def test_survey_enqueue(tmp_path, capsys):
    paths = make_root(tmp_path / "root")
    assert main(["--root", str(paths.root), "survey", "enqueue", "--limit", "2"]) == 0
    assert "enqueued 2 object(s) from sky tiny-v0" in capsys.readouterr().out  # one point, its 2 seeds
    assert main(["--root", str(paths.root), "survey", "enqueue"]) == 0
    out = capsys.readouterr().out
    assert "enqueued 6 object(s)" in out and "queue now 8" in out
    assert main(["--root", str(paths.root), "survey", "enqueue"]) == 0
    assert "enqueued 0 object(s)" in capsys.readouterr().out  # idempotent
    with Catalog(paths.db) as catalog:
        assert {r["source"] for r in catalog.query("SELECT source FROM objects")} == {"coarse"}


def test_followup_run_and_backup(synthetic_root, capsys):
    paths, done = synthetic_root
    root = str(paths.root)
    with Catalog(paths.db) as catalog:
        grok = catalog.query("SELECT object_id FROM objects WHERE region = 'grok' AND status = 'done'")[0]["object_id"]
        catalog.request_followup(grok, "requested in a test", requested_by="human")
    assert main(["--root", root, "followup", "run", "--n", "1"]) == 0
    assert f"{grok} done: ri_run" in capsys.readouterr().out
    assert main(["--root", root, "followup", "run"]) == 0
    assert "no follow-up to run" in capsys.readouterr().out
    assert main(["--root", root, "backup"]) == 0
    assert "catalog backed up to" in capsys.readouterr().out and len(list(paths.backups.glob("catalog-*"))) == 1


def test_calibrate_wiring(tmp_path, monkeypatch, capsys):
    paths = make_root(tmp_path / "root")
    calls = {}

    def fake_run(cfg, out_dir, *, n_jobs=1, progress=None):
        calls.update(preset=cfg.name, out=out_dir, jobs=n_jobs)
        out_dir.mkdir(parents=True, exist_ok=True)
        progress(1, 2)
        progress(2, 2)
        return {"fake": True}

    def fake_report(results, out_dir):
        table = [{"order": 2, "n": 512, "noise": 0.003, "floor": 1e-3, "value": 0.003}]
        (out_dir / "calibration.json").write_text(json.dumps(
            {"format": "tycho-calibration/1", "recommended_min_effect": {"table": table}}), encoding="utf-8")
        (out_dir / "CALIBRATION.md").write_text("# report\n", encoding="utf-8")
        return out_dir / "CALIBRATION.md"

    monkeypatch.setattr(calibration_pkg, "run_calibration", fake_run)
    monkeypatch.setattr(calibration_pkg, "write_report", fake_report)
    assert main(["--root", str(paths.root), "calibrate", "--quick", "--jobs", "3"]) == 0
    assert calls == {"preset": "quick", "out": paths.reports / "calibration", "jobs": 3}
    out = capsys.readouterr()
    assert "min_effect at n_probe=512: 0.003 nats" in out.out and "2/2 work items (100%)" in out.err
    assert main(["--root", str(paths.root), "calibrate"]) == 0
    assert calls["preset"] == "full" and calls["jobs"] == 1  # the defaults


def test_survey_run_and_the_survey_lock(tmp_path, monkeypatch, capsys):
    paths = make_root(tmp_path / "root")
    monkeypatch.setattr(daemon_mod, "LOOP_INTERVAL_S", 0.05)
    assert main(["--root", str(paths.root), "survey", "run", "--workers", "0", "--max-objects", "0"]) == 0
    out = capsys.readouterr().out
    assert "survey stopped after" in out and "max_objects=0 observed" in out
    with SurveyLock(paths):
        assert main(["--root", str(paths.root), "survey", "run", "--workers", "0"]) == 3
    assert "another survey daemon" in capsys.readouterr().err


def test_agent_disabled_exits_with_3(tmp_path, capsys):
    paths = make_root(tmp_path / "root", config={"agent": {"enabled": False}})
    with Catalog(paths.db) as catalog:
        catalog.stats()
    assert main(["--root", str(paths.root), "agent", "weekly"]) == 3
    assert "agent unavailable" in capsys.readouterr().err


def test_example_config_lists_every_default():
    """``tycho init`` copies configs/tycho.example.yaml: it must document exactly the built-in defaults, except
    ri.min_effect, which stays commented out so that the calibration file sets it."""
    def flat(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{prefix}{k}."))
            else:
                out[f"{prefix}{k}"] = v
        return out

    example = flat(yaml.safe_load((REPO / "configs" / "tycho.example.yaml").read_text(encoding="utf-8")))
    defaults = flat(DEFAULT_CONFIG)
    assert set(defaults) - set(example) == {"ri.min_effect"} and set(example) <= set(defaults)
    assert {k: v for k, v in example.items()} == {k: v for k, v in defaults.items() if k != "ri.min_effect"}
