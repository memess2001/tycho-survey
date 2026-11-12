"""The sky (SPEC §6.2): survey regions, each a grid of knobs over one task's configuration and its training.

File format (``configs/sky_v0.yaml``)
    Top level: ``version`` (part of every object's identity), ``description``, ``regions``. A region has ``name``,
    ``task``, ``seeds`` (initial seeds per point), ``priority`` (scheduling weight), ``ri_policy`` (always | followup
    | never), ``ri`` (RIConfig overrides), ``fixed`` (task-config values shared by the region), ``train`` (TrainConfig
    fields shared by the region), an optional ``description``, and ``knobs``: a mapping from knob name to
    ``{values, kind, refine, bounds}`` (a bare list is shorthand for ``{values: [...]}``). Unknown keys are errors, so
    typos fail when the sky is loaded rather than months into a survey. YAML ``null`` is a legal categorical value
    (e.g. ``n_train: null`` = online data). Beware that YAML 1.1 reads ``1e-3`` as a *string* (write ``1.0e-3`` or
    ``0.001``); train values are type-checked against TrainConfig, so such a slip is caught at load time.

Knobs
    A knob name is a task-config key or ``train.<TrainConfig field>``. Kinds:

    =============  ================================================  ========================================
    kind           values                                            coordinate (``coords``)
    =============  ================================================  ========================================
    categorical    any JSON values (incl. null), in the listed order  rank / (n − 1)
    ordinal        numbers (strictly increasing) or strings          rank / (n − 1)
    log            positive numbers, strictly increasing             linear in ln(value) over the bounds
    linear         numbers, strictly increasing                      linear in value over the bounds
    =============  ================================================  ========================================

    ``refine: true`` is allowed for log and linear knobs only. A refinable knob accepts any value within its
    ``bounds`` (default ``[min(values), max(values)]``); the scheduler inserts midpoints (geometric for log,
    arithmetic for linear, rounded half-up for integer knobs, :data:`MIDPOINT_SIG_DIGITS` significant digits
    otherwise). Every other knob accepts grid values only. A knob whose grid values are all integers is an integer
    knob. Values are compared by canonical JSON (``128 == 128.0``, but ``True != 1`` and ``None`` is a value).

Points and object identity
    A *point* maps each knob of a region to a value. :meth:`Sky.make_spec` turns a point into an ObjectSpec:
    ``task_cfg = fixed`` plus the task knobs; the TrainConfig is the region's ``train`` section overridden by the
    ``train.*`` knobs (a knob may not repeat a ``fixed`` or ``train`` key); ``sky_version = sky.version``;
    ``region = region.name``. The object id hashes (task, task_cfg, seed, train, sky_version) after canonical
    normalisation, so it does not depend on knob order or on 128 vs 128.0. :func:`point_key` is the canonical JSON of a
    point under the same normalisation; it identifies a point across its seeds. :meth:`Sky.point_of` recovers the
    point of a spec.

Lattice
    :meth:`Sky.neighbors` returns the points one grid step away along one knob (values in the listed order). For an
    off-grid value of a refinable knob, the neighbours along that knob are the bracketing grid values.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import re
import types
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from tycho.train.spec import ObjectSpec, RIConfig, TrainConfig, _canonical

KNOB_KINDS: tuple[str, ...] = ("categorical", "ordinal", "log", "linear")
REFINABLE_KINDS: tuple[str, ...] = ("log", "linear")
RI_POLICIES: tuple[str, ...] = ("always", "followup", "never")
TRAIN_PREFIX = "train."

MIDPOINT_SIG_DIGITS = 6
"""Significant digits of float midpoints (keeps refined values readable; relative rounding error ≤ 5e-7)."""

_REGION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_KNOB_NAME = re.compile(r"^(train\.)?[A-Za-z_][A-Za-z0-9_]*$")
_TRAIN_FIELDS: frozenset[str] = frozenset(f.name for f in fields(TrainConfig))
_TRAIN_TYPES: dict[str, Any] = typing.get_type_hints(TrainConfig)
_RI_FIELDS: frozenset[str] = frozenset(f.name for f in fields(RIConfig)) - {"min_effect_calibration"}
"""RIConfig fields a region's ``ri`` mapping may set (``min_effect_calibration`` is set by
``tycho.survey.followup.resolve_ri_config`` only)."""
_SKY_KEYS = frozenset({"version", "description", "regions"})
_REGION_KEYS = frozenset({"name", "task", "seeds", "priority", "ri_policy", "ri", "fixed", "train", "knobs",
                          "description"})
_KNOB_KEYS = frozenset({"values", "kind", "refine", "bounds"})


def point_key(point: Mapping[str, Any]) -> str:
    """Canonical JSON of a point: sorted keys, compact, integral floats as ints (the object-identity normalisation).

    Raises ValueError for non-finite numbers.
    """
    return json.dumps(_canonical(dict(point)), sort_keys=True, separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------------------------------------- knobs
@dataclass
class Knob:
    """One axis of a region's grid (see the module docstring for kinds, refinement and bounds)."""

    name: str
    values: list
    kind: str = "ordinal"
    refine: bool = False
    bounds: tuple | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _KNOB_NAME.match(self.name):
            raise ValueError(f"invalid knob name {self.name!r} (a task-config key or 'train.<field>')")
        train_field = self.train_field
        if train_field is not None and train_field not in _TRAIN_FIELDS:
            raise ValueError(f"knob {self.name!r}: TrainConfig has no field {train_field!r}")
        if self.kind not in KNOB_KINDS:
            raise ValueError(f"knob {self.name!r}: kind must be one of {KNOB_KINDS}, got {self.kind!r}")
        if not isinstance(self.values, (list, tuple)) or not self.values:
            raise ValueError(f"knob {self.name!r}: needs a non-empty list of values")
        self.values = list(self.values)
        keys = []
        for v in self.values:
            key = _value_key(v)
            if key is None:
                raise ValueError(f"knob {self.name!r}: value {v!r} is not finite JSON")
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise ValueError(f"knob {self.name!r}: duplicate values in {self.values!r}")
        if train_field is not None:
            for v in self.values:
                err = _train_type_error(train_field, v)
                if err:
                    raise ValueError(f"knob {self.name!r}: {err}")
        numeric = all(_is_number(v) for v in self.values)
        if self.kind in REFINABLE_KINDS and not numeric:
            raise ValueError(f"knob {self.name!r}: {self.kind} values must be finite numbers, got {self.values!r}")
        if self.kind == "ordinal" and not (numeric or all(isinstance(v, str) for v in self.values)):
            raise ValueError(f"knob {self.name!r}: ordinal values must be all numbers or all strings "
                             f"(use kind categorical for mixed values or null)")
        if numeric and self.kind != "categorical" and any(b <= a for a, b in zip(self.values, self.values[1:])):
            raise ValueError(f"knob {self.name!r}: {self.kind} values must be strictly increasing, got {self.values!r}")
        if self.kind == "log" and self.values[0] <= 0:
            raise ValueError(f"knob {self.name!r}: log values must be positive, got {self.values!r}")
        if not isinstance(self.refine, bool):
            raise ValueError(f"knob {self.name!r}: refine must be true or false, got {self.refine!r}")
        if self.refine and self.kind not in REFINABLE_KINDS:
            raise ValueError(f"knob {self.name!r}: refine requires kind log or linear, not {self.kind}")
        if self.bounds is not None:
            self.bounds = self._checked_bounds(self.bounds)
        elif self.refine:
            self.bounds = (self.values[0], self.values[-1])
        self._index = {k: i for i, k in enumerate(keys)}

    def _checked_bounds(self, bounds: Any) -> tuple:
        if not self.refine:
            raise ValueError(f"knob {self.name!r}: bounds only apply to refinable knobs")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or not all(_is_number(b) for b in bounds):
            raise ValueError(f"knob {self.name!r}: bounds must be two finite numbers, got {bounds!r}")
        lo, hi = bounds
        if not (lo <= self.values[0] and self.values[-1] <= hi):
            raise ValueError(f"knob {self.name!r}: bounds {list(bounds)} must contain all values {self.values!r}")
        if self.kind == "log" and lo <= 0:
            raise ValueError(f"knob {self.name!r}: log bounds must be positive, got {list(bounds)}")
        if self.is_integer and not (float(lo).is_integer() and float(hi).is_integer()):
            raise ValueError(f"knob {self.name!r}: bounds of an integer knob must be integers, got {list(bounds)}")
        return (lo, hi)

    @property
    def train_field(self) -> str | None:
        """The TrainConfig field this knob sets, or None for a task-config knob."""
        return self.name[len(TRAIN_PREFIX):] if self.name.startswith(TRAIN_PREFIX) else None

    @property
    def is_integer(self) -> bool:
        """True when every grid value is an integer (then midpoints are rounded to integers), except for a
        ``train.*`` knob whose TrainConfig field is a float (``train.weight_decay: [0, 1]`` is not an integer knob)."""
        if self.train_field is not None and float in _allowed_types(_TRAIN_TYPES[self.train_field]):
            return False
        return all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in self.values)

    def index(self, value: Any) -> int | None:
        """Grid index of ``value`` (canonical-JSON comparison), or None if it is not a grid value."""
        key = _value_key(value)
        return None if key is None else self._index.get(key)

    def check(self, value: Any) -> str | None:
        """Why ``value`` is not acceptable for this knob, or None if it is."""
        if self.refine:
            if not _is_number(value):
                return f"knob {self.name!r}: {value!r} is not a finite number"
            lo, hi = self.bounds
            if not lo <= value <= hi:
                return f"knob {self.name!r}: {value!r} is outside the bounds [{lo}, {hi}]"
            if self.is_integer and not float(value).is_integer():
                return f"knob {self.name!r}: {value!r} is not an integer"
            return None
        if self.index(value) is None:
            return f"knob {self.name!r}: {value!r} is not a grid value (allowed: {self.values!r})"
        return None

    def normalize(self, value: Any) -> Any:
        """The value as it enters the spec: the grid value itself (exact YAML type), or for refinable knobs an int
        (integer knobs) or float. ``value`` must pass :meth:`check`."""
        if self.refine:
            return int(round(float(value))) if self.is_integer else float(value)
        return copy.deepcopy(self.values[self.index(value)])

    def position(self, value: Any) -> float:
        """Normalized coordinate in [0, 1] (see the module docstring); 0 for a single-valued knob."""
        if self.kind in REFINABLE_KINDS:
            lo, hi = self.bounds if self.bounds is not None else (self.values[0], self.values[-1])
            if hi == lo:
                return 0.0
            if self.kind == "log":
                return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return (float(value) - lo) / (hi - lo)
        i = self.index(value)
        if i is None:
            raise ValueError(f"knob {self.name!r}: {value!r} is not a grid value")
        return i / (len(self.values) - 1) if len(self.values) > 1 else 0.0

    def midpoint(self, a: Any, b: Any) -> int | float | None:
        """Refinement midpoint of two values (geometric for log, arithmetic for linear; rounded half-up for integer
        knobs, else to :data:`MIDPOINT_SIG_DIGITS` significant digits). None when no value lies strictly between."""
        if not self.refine:
            raise ValueError(f"knob {self.name!r} is not refinable")
        lo, hi = sorted((float(a), float(b)))
        mid = math.sqrt(lo * hi) if self.kind == "log" else (lo + hi) / 2.0
        value: int | float = int(math.floor(mid + 0.5)) if self.is_integer else float(f"{mid:.{MIDPOINT_SIG_DIGITS}g}")
        return value if lo < value < hi else None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"values": copy.deepcopy(self.values), "kind": self.kind, "refine": self.refine}
        if self.bounds is not None:
            out["bounds"] = list(self.bounds)
        return out


# ---------------------------------------------------------------------------------------------- regions
@dataclass
class Region:
    """A task with a grid of knobs, shared settings, a seed count and scheduling metadata."""

    name: str
    task: str
    knobs: list[Knob]
    fixed: dict
    train: dict
    seeds: int = 3
    ri_policy: str = "followup"
    ri: dict = field(default_factory=dict)
    priority: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _REGION_NAME.match(self.name):
            raise ValueError(f"invalid region name {self.name!r} (letters, digits, '_', '.', '-')")
        where = f"region {self.name!r}"
        if not isinstance(self.task, str) or not self.task:
            raise ValueError(f"{where}: task must be a non-empty string")
        self.knobs = list(self.knobs)
        if not all(isinstance(k, Knob) for k in self.knobs):
            raise TypeError(f"{where}: knobs must be Knob objects (Sky.from_dict parses plain data)")
        names = [k.name for k in self.knobs]
        if len(set(names)) != len(names):
            raise ValueError(f"{where}: duplicate knob names in {names}")
        self.fixed = dict(self.fixed or {})
        bad = sorted(repr(k) for k, v in self.fixed.items() if not isinstance(k, str) or _value_key(v) is None)
        if bad:
            raise ValueError(f"{where}: fixed entries {bad} need string keys and finite JSON values")
        self.train = dict(self.train or {})
        try:
            TrainConfig.from_dict(self.train)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: {exc}") from exc
        for key, value in self.train.items():
            err = _train_type_error(key, value)
            if err:
                raise ValueError(f"{where}: {err}")
        clash = sorted(set(self.fixed) & {k.name for k in self.knobs if k.train_field is None})
        clash += sorted(f"train.{f}" for f in set(self.train) & {k.train_field for k in self.knobs})
        if clash:
            raise ValueError(f"{where}: knobs {clash} also appear in fixed/train")
        if isinstance(self.seeds, bool) or not isinstance(self.seeds, int) or self.seeds < 1:
            raise ValueError(f"{where}: seeds must be a positive integer, got {self.seeds!r}")
        if self.ri_policy not in RI_POLICIES:
            raise ValueError(f"{where}: ri_policy must be one of {RI_POLICIES}, got {self.ri_policy!r}")
        self.ri = dict(self.ri or {})
        unknown = sorted(set(self.ri) - _RI_FIELDS)
        if unknown:
            raise ValueError(f"{where}: unknown RIConfig fields {unknown}")
        if not _is_number(self.priority) or self.priority <= 0:
            raise ValueError(f"{where}: priority must be a positive number, got {self.priority!r}")
        self.priority = float(self.priority)
        self.description = str(self.description or "").strip()

    @property
    def knob_names(self) -> list[str]:
        return [k.name for k in self.knobs]

    @property
    def grid_size(self) -> int:
        """Number of points of the coarse grid."""
        return math.prod(len(k.values) for k in self.knobs)

    def knob(self, name: str) -> Knob:
        for k in self.knobs:
            if k.name == name:
                return k
        raise KeyError(f"region {self.name!r} has no knob {name!r}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "task": self.task, "seeds": self.seeds, "priority": self.priority,
                               "ri_policy": self.ri_policy, "ri": copy.deepcopy(self.ri),
                               "fixed": copy.deepcopy(self.fixed), "train": copy.deepcopy(self.train),
                               "knobs": {k.name: k.to_dict() for k in self.knobs}}
        if self.description:
            out["description"] = self.description
        return out


# ---------------------------------------------------------------------------------------------- the sky
@dataclass
class Sky:
    """A versioned set of regions (see the module docstring)."""

    version: str
    regions: list[Region]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError(f"sky version must be a non-empty string, got {self.version!r}")
        self.regions = list(self.regions)
        if not all(isinstance(r, Region) for r in self.regions):
            raise TypeError("regions must be Region objects (Sky.from_dict parses plain data)")
        names = [r.name for r in self.regions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate region names in {names}")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Sky":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: a sky file must be a YAML mapping")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Sky":
        unknown = sorted(set(d) - _SKY_KEYS)
        if unknown:
            raise ValueError(f"unknown sky keys {unknown}")
        regions = d.get("regions") or []
        if not isinstance(regions, list):
            raise ValueError("sky 'regions' must be a list")
        return cls(version=d.get("version"), regions=[_region_from_dict(r) for r in regions],
                   description=str(d.get("description") or "").strip())

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form; ``Sky.from_dict(sky.to_dict()) == sky``."""
        return {"version": self.version, "description": self.description,
                "regions": [r.to_dict() for r in self.regions]}

    def region(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"unknown region {name!r} (sky {self.version} has {[r.name for r in self.regions]})")

    def _region(self, region: Region | str) -> Region:
        return region if isinstance(region, Region) else self.region(region)

    def points(self, region: Region | str) -> list[dict[str, Any]]:
        """The full coarse grid of a region (knob name → value), knobs varying in declaration order (last fastest)."""
        reg = self._region(region)
        names = reg.knob_names
        return [dict(zip(names, copy.deepcopy(combo))) for combo in itertools.product(*(k.values for k in reg.knobs))]

    def validate_point(self, region_name: Region | str, point: Mapping[str, Any]) -> list[str]:
        """Problems with a point ([] if it is valid): unknown region, unknown or missing knobs, off-grid values of
        non-refinable knobs, non-numeric / out-of-bounds / non-integer values of refinable knobs."""
        try:
            reg = self._region(region_name)
        except KeyError as exc:
            return [str(exc.args[0])]
        if not isinstance(point, Mapping):
            return [f"a point must be a mapping of knob names to values, got {type(point).__name__}"]
        names = set(reg.knob_names)
        errors = [f"unknown knob {k!r} (region {reg.name!r} has knobs {reg.knob_names})" for k in point
                  if k not in names]
        for knob in reg.knobs:
            if knob.name not in point:
                errors.append(f"missing knob {knob.name!r}")
            else:
                err = knob.check(point[knob.name])
                if err:
                    errors.append(err)
        return errors

    def _require_valid(self, reg: Region, point: Mapping[str, Any]) -> None:
        errors = self.validate_point(reg, point)
        if errors:
            raise ValueError(f"invalid point for region {reg.name!r}: " + "; ".join(errors))

    def make_spec(self, region: Region | str, point: Mapping[str, Any], seed: int, device: str = "cpu") -> ObjectSpec:
        """The ObjectSpec of a point and seed (ValueError for an invalid point or a negative seed)."""
        reg = self._region(region)
        self._require_valid(reg, point)
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        task_cfg = copy.deepcopy(reg.fixed)
        train = copy.deepcopy(reg.train)
        for knob in reg.knobs:
            value = knob.normalize(point[knob.name])
            if knob.train_field is None:
                task_cfg[knob.name] = value
            else:
                train[knob.train_field] = value
        return ObjectSpec(task=reg.task, task_cfg=task_cfg, seed=int(seed), train=TrainConfig.from_dict(train),
                          sky_version=self.version, region=reg.name, device=device)

    def point_of(self, region: Region | str, spec: ObjectSpec) -> dict[str, Any]:
        """The knob point of a spec of this region (inverse of :meth:`make_spec` on the knobs). ValueError if the spec
        is of another task or lacks a knob's task-config key. The point is not validated (a spec from an older sky
        version may hold values that are off the current grid)."""
        reg = self._region(region)
        if spec.task != reg.task:
            raise ValueError(f"spec task {spec.task!r} is not region {reg.name!r}'s task {reg.task!r}")
        point = {}
        for knob in reg.knobs:
            if knob.train_field is not None:
                point[knob.name] = getattr(spec.train, knob.train_field)
            elif knob.name in spec.task_cfg:
                point[knob.name] = spec.task_cfg[knob.name]
            else:
                raise ValueError(f"spec has no task_cfg[{knob.name!r}] for region {reg.name!r}")
        return point

    def neighbors(self, region: Region | str, point: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Lattice neighbours: the point moved one grid step along one knob (knob order; lower step first). An
        off-grid value of a refinable knob has the bracketing grid values as neighbours along that knob."""
        reg = self._region(region)
        self._require_valid(reg, point)
        out = []
        for knob in reg.knobs:
            value = point[knob.name]
            i = knob.index(value)
            if i is not None:
                steps = [knob.values[j] for j in (i - 1, i + 1) if 0 <= j < len(knob.values)]
            else:
                pos = knob.position(value)
                below = [v for v in knob.values if knob.position(v) < pos]
                above = [v for v in knob.values if knob.position(v) > pos]
                steps = below[-1:] + above[:1]
            for v in steps:
                nb = dict(point)
                nb[knob.name] = copy.deepcopy(v)
                out.append(nb)
        return out

    def coords(self, region: Region | str, point: Mapping[str, Any]) -> np.ndarray:
        """Normalized coordinates in [0, 1], one per knob (log scale for log knobs, rank for ordinal/categorical)."""
        reg = self._region(region)
        self._require_valid(reg, point)
        return np.asarray([k.position(point[k.name]) for k in reg.knobs], dtype=np.float64)


# ---------------------------------------------------------------------------------------------- parsing helpers
def _region_from_dict(r: Any) -> Region:
    if not isinstance(r, Mapping):
        raise ValueError(f"a region must be a mapping, got {r!r}")
    unknown = sorted(set(r) - _REGION_KEYS)
    if unknown:
        raise ValueError(f"region {r.get('name')!r}: unknown keys {unknown}")
    spec = r.get("knobs") or {}
    if not isinstance(spec, Mapping):
        raise ValueError(f"region {r.get('name')!r}: knobs must be a mapping of knob name to settings")
    knobs = [_knob_from_dict(name, settings) for name, settings in spec.items()]
    return Region(name=r.get("name"), task=r.get("task"), knobs=knobs, fixed=r.get("fixed") or {},
                  train=r.get("train") or {}, seeds=r.get("seeds", 3), ri_policy=r.get("ri_policy", "followup"),
                  ri=r.get("ri") or {}, priority=r.get("priority", 1.0), description=r.get("description") or "")


def _knob_from_dict(name: Any, settings: Any) -> Knob:
    if isinstance(settings, (list, tuple)):
        settings = {"values": list(settings)}
    if not isinstance(settings, Mapping):
        raise ValueError(f"knob {name!r}: settings must be a mapping or a list of values, got {settings!r}")
    unknown = sorted(set(settings) - _KNOB_KEYS)
    if unknown:
        raise ValueError(f"knob {name!r}: unknown keys {unknown}")
    bounds = settings.get("bounds")
    return Knob(name=name, values=settings.get("values") or [], kind=settings.get("kind", "ordinal"),
                refine=settings.get("refine", False), bounds=None if bounds is None else tuple(bounds))


def _value_key(value: Any) -> str | None:
    """Canonical JSON of a value (None if it is not finite JSON)."""
    try:
        return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None


def _is_number(v: Any) -> bool:
    """A finite real number (bools are not numbers here)."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(v))


def _allowed_types(hint: Any) -> tuple:
    """The member types of a type hint (``float | None`` -> (float, NoneType))."""
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        return typing.get_args(hint)
    return (hint,)


def _train_type_error(name: str, value: Any) -> str | None:
    """Why ``value`` cannot be TrainConfig field ``name`` (None if it can). Ints are accepted for floats."""
    hint = _TRAIN_TYPES[name]
    for t in _allowed_types(hint):
        if t is type(None) and value is None:
            return None
        if t is str and isinstance(value, str):
            return None
        if t is bool and isinstance(value, bool):
            return None
        if t is int and isinstance(value, int) and not isinstance(value, bool):
            return None
        if t is float and _is_number(value):
            return None
    return f"train.{name} must be of type {getattr(hint, '__name__', hint)}, got {value!r}"
