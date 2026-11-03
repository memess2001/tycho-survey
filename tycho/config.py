"""Paths and configuration for a Tycho survey root directory.

A *survey root* is a directory holding ``tycho.yaml`` (optional overrides), ``configs/`` (sky definitions),
``data/`` (catalog + light curves + RI curves), ``reports/``, ``logs/``, ``releases/`` and ``backups/``.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "survey": {
        "sky": "configs/sky_v0.yaml",
        "n_workers": None,  # None -> max(1, os.cpu_count() - 2)
        "torch_threads": 1,
        "device": "cpu",  # cpu | mps | auto  (cpu is deterministic and usually fastest for tiny models)
        "heartbeat_s": 15.0,
        "stale_after_s": 300.0,
        "topup_every_s": 60.0,
        "followup_every_s": 600.0,
        "status_every_s": 300.0,
        "backup_every_h": 24.0,
        "backups_keep": 14,
        "min_free_gb": 20.0,
        "max_attempts": 3,
        "code_version": None,  # None -> git describe if available
    },
    "scheduler": {
        "target_queue": 200,
        "boundary_refine": True,
        "max_refine_depth": 3,
        "extra_seeds": 2,
        "random_control_frac": 0.05,
        "followups_per_day": 200,
    },
    "ri": {
        "n_probe": 4096,
        "alpha": 0.01,
        "min_effect": 1e-3,
        "null": "chi2",
        "n_points": 40,
        "max_subsets": 50_000,
        "keep_top": 200,
        "calibration_file": "reports/calibration/calibration.json",
    },
    "notify": {
        "max_per_hour": 6,
        "min_severity": 2,
        "channels": {
            "ntfy": {"enabled": False, "server": "https://ntfy.sh", "topic": ""},
            "webhook": {"enabled": False, "url": ""},
            "macos": {"enabled": True},
        },
    },
    "agent": {
        "enabled": True,
        "model": "claude-opus-5-5",
        "nightly_effort": "medium",
        "weekly_effort": "high",
        "max_iterations": 40,
        "max_cost_usd": 3.0,
        "fallbacks": True,
        "language": "en",  # en | zh | ja
        "max_proposals_per_run": 50,
        "max_objects_per_day": 400,
        "max_followups_per_run": 20,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` merged recursively into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(root: Path | str) -> dict:
    """Load ``<root>/tycho.yaml`` merged over :data:`DEFAULT_CONFIG` (missing file -> defaults)."""
    root = Path(root)
    path = root / "tycho.yaml"
    user: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    return deep_merge(DEFAULT_CONFIG, user)


def default_root() -> Path:
    """``$TYCHO_ROOT`` if set, else the current working directory."""
    return Path(os.environ.get("TYCHO_ROOT", os.getcwd())).resolve()


@dataclass(frozen=True)
class Paths:
    """All filesystem locations of a survey root. Call :meth:`ensure` once to create directories."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def lightcurves(self) -> Path:
        return self.data / "lightcurves"

    @property
    def ri(self) -> Path:
        return self.data / "ri"

    @property
    def work(self) -> Path:
        return self.data / "work"

    @property
    def checkpoints(self) -> Path:
        return self.data / "checkpoints"

    @property
    def db(self) -> Path:
        return self.data / "catalog.sqlite"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def backups(self) -> Path:
        return self.root / "backups"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    def ensure(self) -> "Paths":
        for p in (self.data, self.lightcurves, self.ri, self.work, self.checkpoints, self.reports, self.logs,
                  self.releases, self.backups, self.configs):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def lightcurve_path(self, object_id: str) -> Path:
        return self.lightcurves / object_id[:2] / f"{object_id}.npz"

    def ri_path(self, object_id: str, tag: str) -> Path:
        return self.ri / object_id[:2] / f"{object_id}_{tag}.npz"

    def work_dir(self, object_id: str) -> Path:
        return self.work / object_id
