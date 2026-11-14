"""Specification of one survey object (a training run) and of its result. Contract types."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from tycho.analysis.ricurve import RICurve


@dataclass
class TrainConfig:
    """Optimisation and observation schedule of one object. Part of the object's identity."""

    steps: int = 10_000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"  # sgd | adam | adamw
    momentum: float = 0.9  # sgd only
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 0
    lr_schedule: str = "constant"  # constant | cosine
    grad_clip: float | None = None
    n_log_evals: int = 60
    n_lin_evals: int = 40
    ntk_every: int = 5  # compute eNTK statistics at every k-th evaluation (0 = never)
    ntk_probe: int = 32
    checkpoint_every_s: float = 300.0  # wall-clock interval of resume checkpoints

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrainConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig fields: {sorted(unknown)}")
        return cls(**d)


@dataclass
class RIConfig:
    """Settings of RI spectroscopy (in situ or by replay). Not part of the object's identity."""

    enabled: bool = False
    n_probe: int = 4096
    max_order: int | None = None  # None -> task.default_ri_order()
    alpha: float = 0.01
    min_effect: float = 1e-3
    null: str = "chi2"
    n_points: int = 40
    max_subsets: int = 50_000
    keep_top: int = 200
    # "scan": RI scans of the task's RI variables (ri_scan). "patterns": the task's log-odds on every input pattern
    # and the per-pattern counts of n_probe probes drawn from its training distribution, at the same steps
    # (tasks implementing tycho.tasks.patterns.PatternTask; sky v2 Part A). Stored as RunResult.patterns.
    mode: str = "scan"
    # calibration.json that min_effect came from (set by tycho.survey.followup.resolve_ri_config, never from a config
    # file). When set, the runner re-reads the calibrated threshold at the scan's actual number of probe rows, which
    # differs from n_probe for tasks that probe a fixed input set (modular addition probes all p^2 pairs).
    min_effect_calibration: str | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "RIConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _canonical(o: Any) -> Any:
    """Normalise a config value for hashing (numbers, containers, numpy scalars)."""
    if isinstance(o, dict):
        return {str(k): _canonical(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_canonical(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if np.isfinite(f) and f == int(f) and abs(f) < 1e15:
            return int(f)
        return f
    return o


@dataclass
class ObjectSpec:
    """One survey object: a task configuration, a training seed and a training configuration."""

    task: str
    task_cfg: dict
    seed: int
    train: TrainConfig = field(default_factory=TrainConfig)
    sky_version: str = "adhoc"
    region: str = "adhoc"
    device: str = "cpu"

    def identity(self) -> dict:
        """What defines the object (region and device are bookkeeping, not identity)."""
        return {"task": self.task, "task_cfg": _canonical(self.task_cfg), "seed": int(self.seed),
                "train": _canonical(asdict(self.train)), "sky_version": self.sky_version}

    def object_id(self) -> str:
        blob = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"task": self.task, "task_cfg": self.task_cfg, "seed": int(self.seed), "train": asdict(self.train),
                "sky_version": self.sky_version, "region": self.region, "device": self.device}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=_canonical)

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectSpec":
        return cls(task=d["task"], task_cfg=dict(d.get("task_cfg") or {}), seed=int(d["seed"]),
                   train=TrainConfig.from_dict(d.get("train")), sky_version=d.get("sky_version", "adhoc"),
                   region=d.get("region", "adhoc"), device=d.get("device", "cpu"))

    @classmethod
    def from_json(cls, s: str) -> "ObjectSpec":
        return cls.from_dict(json.loads(s))


@dataclass
class RunResult:
    """Outcome of observing one object."""

    object_id: str
    status: str  # done | failed | interrupted
    light_curve: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    ri_curve: "RICurve | None" = None
    patterns: dict[str, np.ndarray] | None = None
    wall_time_s: float = 0.0
    device: str = "cpu"
    n_params: int = 0
    error: str | None = None
