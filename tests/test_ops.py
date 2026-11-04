"""Tests of ops/: shell syntax, the launchd templates (rendered and parsed with plistlib, and their command lines
accepted by the CLI), the installer's dry run, that pmset_recommendations.sh never runs sudo, and healthcheck.sh."""
from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tycho.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
OPS = REPO / "ops"
SCRIPTS = ("install_macmini.sh", "uninstall_macmini.sh", "pmset_recommendations.sh", "healthcheck.sh")
LABELS = ("com.tycho.survey", "com.tycho.nightly", "com.tycho.weekly")
ROOT, PYTHON, USER = "/Users/tycho/survey root", "/Users/tycho/tycho/.venv/bin/python", "tycho"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def render(label: str) -> dict:
    text = (OPS / "launchd" / f"{label}.plist").read_text(encoding="utf-8")
    assert all(p in text for p in ("__ROOT__", "__PYTHON__", "__USER__"))
    text = text.replace("__ROOT__", ROOT).replace("__PYTHON__", PYTHON).replace("__USER__", USER)
    return plistlib.loads(text.encode("utf-8"))


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_are_executable_bash(script):
    path = OPS / script
    assert os.access(path, os.X_OK)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    assert subprocess.run([str(path), "--help"], capture_output=True).returncode == 0


def test_launchd_templates():
    jobs = {label: render(label) for label in LABELS}
    parser = build_parser()
    for label, job in jobs.items():
        assert job["Label"] == label
        args = job["ProgramArguments"]
        assert args[:5] == [PYTHON, "-m", "tycho", "--root", ROOT]
        parser.parse_args(args[3:])  # the CLI accepts the command line
        env = job["EnvironmentVariables"]
        assert env["TYCHO_ROOT"] == ROOT and env["PYTHONUNBUFFERED"] == "1" and "/usr/bin" in env["PATH"]
        assert env["USER"] == USER and job["WorkingDirectory"] == ROOT
        for key in ("StandardOutPath", "StandardErrorPath"):
            assert job[key].startswith(f"{ROOT}/logs/")
    survey = jobs["com.tycho.survey"]
    assert survey["ProgramArguments"][5:] == ["survey", "run"]
    assert survey["KeepAlive"] is True and survey["RunAtLoad"] is True and survey["ThrottleInterval"] >= 10
    assert survey["ExitTimeOut"] > 95  # the daemon's shutdown timeout (90 s) plus the kill grace
    assert jobs["com.tycho.nightly"]["ProgramArguments"][5:] == ["agent", "nightly"]
    assert jobs["com.tycho.nightly"]["StartCalendarInterval"] == {"Hour": 3, "Minute": 17}
    assert jobs["com.tycho.weekly"]["ProgramArguments"][5:] == ["agent", "weekly"]
    assert jobs["com.tycho.weekly"]["StartCalendarInterval"] == {"Weekday": 1, "Hour": 7, "Minute": 41}
    assert not jobs["com.tycho.nightly"]["RunAtLoad"] and "KeepAlive" not in jobs["com.tycho.weekly"]


def test_installer_dry_run_changes_nothing(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path / "home")}
    env.pop("TYCHO_ROOT", None)
    root = tmp_path / "survey"
    r = subprocess.run([str(OPS / "install_macmini.sh"), "--dry-run", "--root", str(root)], capture_output=True,
                       text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("launchctl bootstrap") == 3 and "init" in r.stdout and "uv pip install" in r.stdout
    assert not (tmp_path / "home").exists() and not root.exists()
    u = subprocess.run([str(OPS / "uninstall_macmini.sh"), "--dry-run"], capture_output=True, text=True, env=env,
                       timeout=60)
    assert u.returncode == 0 and not (tmp_path / "home").exists()


def test_pmset_recommendations_only_print(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "ran"
    for name in ("sudo", "pmset", "fdesetup"):
        (fake_bin / name).write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
        (fake_bin / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    r = subprocess.run([str(OPS / "pmset_recommendations.sh")], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0
    for line in ("sudo pmset -a sleep 0", "sudo pmset -a disksleep 0", "sudo pmset -a autorestart 1",
                 "sudo pmset -a womp 1"):
        assert line in r.stdout
    assert "FileVault" in r.stdout and "Automatic Updates" in r.stdout
    if os.uname().sysname != "Darwin":
        assert not marker.exists()  # nothing was executed (on macOS it reads the settings with `pmset -g`)


def test_healthcheck(tmp_path):
    status = tmp_path / "reports" / "status.json"
    script = [str(OPS / "healthcheck.sh"), "--root", str(tmp_path)]
    r = subprocess.run(script, capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "does not exist" in r.stdout
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"state": "running", "status_every_s": 300, "disk": {"free_gb": 50.0, "low": False},
                                  "workers": [], "throughput": {"failed_last_24h": 0}}), encoding="utf-8")
    r = subprocess.run(script, capture_output=True, text=True, timeout=30)
    healthy = os.uname().sysname != "Darwin"  # on macOS the launchd job is checked too
    assert ("FAIL" not in r.stdout) == healthy and (r.returncode == 0) == healthy, r.stdout
    assert "daemon state: running" in r.stdout
    old = time.time() - 3600
    os.utime(status, (old, old))
    r = subprocess.run(script + ["--quiet"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "s old (limit 900 s)" in r.stdout and "ok " not in r.stdout
