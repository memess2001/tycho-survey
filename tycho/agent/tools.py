"""The survey agent's tools (SPEC §7): everything the nightly and weekly Claude agent can see and do.

Safety model
    The agent runs unattended, so its capabilities are exactly the tools defined here:

    * reads of the catalog through fixed, parameterised SQL (run with ``Catalog.query``, which only executes read-only
      statements); the model never supplies SQL, file paths or shell commands, and nothing here touches the network;
    * a few bounded, validated writes, each capped per run (and proposals also per 24 h, counted in the catalog so the
      cap survives restarts): configuration proposals (validated by ``Sky.validate_point``, enqueued with
      ``source="agent"``), RI follow-up requests, classification labels (``classifier="agent"``), one survey-log entry
      per nightly run and one report per weekly run (its file name is derived from the clock, never from the model).

    Every input is validated against the tool's JSON schema (the API enforces the schema too, through strict tool use)
    and then against the semantic limits of :class:`ToolLimits`; violations come back as error results that explain
    what to fix. The runner audits every call in ``agent_tool_calls``.

Tool schemas (:data:`TOOL_SPECS`)
    Every tool is strict (``"strict": True``): each object schema lists all its properties in ``required`` and sets
    ``additionalProperties: false``. Strict schemas cannot express numeric ranges or string/array lengths (the API
    rejects ``minimum``, ``maxLength`` and the like), so those limits are checked here. The API also limits a request
    to 16 parameters with union types (type arrays such as ``["integer", "null"]`` or ``anyOf``) summed over all strict
    tools, so optional inputs are not nullable: they take an explicit default instead (0 for numbers, "" for strings,
    "any" in enumerations), and union types are reserved for knob values, which are polymorphic (number, string,
    boolean or null). ``since_hours = 0`` means "since the start of the run's window" (:attr:`SurveyTools.window_start`,
    the start of the previous completed run of the same kind; the drivers state it in the context packet).

Outputs
    Compact JSON with floats rounded to :data:`FLOAT_SIG_DIGITS` significant digits, NaN/inf as null and timestamps as
    ISO-8601 UTC strings. An output longer than ``limits.max_output_chars`` is shrunk by dropping the tail items of its
    largest lists until it fits, and the dropped counts are reported under ``"_truncated"``, so the result stays valid
    JSON (:func:`fit_json`).

Tool sets
    Nightly runs get every tool except ``write_report``; weekly runs get the read-only tools, ``fit_power_law`` and
    ``write_report`` (the weekly report recommends; the nightly run acts).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from tycho.agent.prompts import KINDS
from tycho.analysis.classify import LABELS
from tycho.analysis.events import EVENT_KINDS, TRANSIENT_KINDS
from tycho.analysis.summarize import SUMMARY_KEYS
from tycho.config import DEFAULT_CONFIG, Paths
from tycho.survey.catalog import SUMMARY_COLUMNS, Catalog
from tycho.survey.scheduler import Scheduler
from tycho.survey.sky import Region, Sky, point_key
from tycho.train.spec import ObjectSpec

logger = logging.getLogger(__name__)

AGENT_SOURCE = "agent"
"""How the agent's writes are attributed: ``objects.source``, ``proposals.source``, ``followups.requested_by``,
``classifications.classifier`` and ``survey_log.author``."""

AGENT_FOLLOWUP_PRIORITY = 3.5
"""Priority of the agent's follow-up requests: above the scheduler's transient-bearing selections (3.0) and below
its unknown-class selections (4.0), so targeted requests run soon without starving the unknown queue."""

FLOAT_SIG_DIGITS = 6
"""Significant digits of floats in tool outputs."""

DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 200
"""Rows returned by a list tool when its ``limit`` is 0, and the largest ``limit`` accepted."""

DEFAULT_LOG_ENTRY_CHARS = 4000
"""Markdown characters per entry returned by ``read_survey_log`` when ``max_chars_per_entry`` is 0."""

MAX_LOOKBACK_HOURS = 24.0 * 366 * 10
"""Largest accepted ``since_hours`` / ``finished_within_hours`` (ten years: effectively "all time")."""

MAX_PROPOSALS_PER_CALL = 20
MAX_WHERE_FILTERS = 10
MAX_GROUP_BY = 4
MAX_METRICS = 8
MAX_TAGS = 8
MAX_TITLE_CHARS = 200
MAX_GROUPS = 300
"""Per-call bounds on list inputs and on the number of groups ``query_summaries`` returns."""

MAX_EVENTS_PER_OBJECT = 40
MAX_CLASSIFICATIONS_PER_OBJECT = 20
MAX_BOUNDARY_PAIRS = 30
MAX_LOG_TITLES = 50
"""Caps on nested lists in single-object and summary outputs."""

ERROR_PREVIEW_CHARS = 400
"""Characters of an object's error message shown in listings (get_object shows up to ``limits.max_note_chars``)."""

LIGHTCURVE_EXCERPT_POINTS = 24
"""Number of (roughly log-spaced) evaluation steps in a light-curve excerpt."""

LIGHTCURVE_EXCERPT_KEYS: tuple[str, ...] = ("step", "train_loss", "test_loss", "train_acc", "test_acc", "param_norm",
                                            "erank_hidden", "ntk_velocity")
"""Light-curve keys included in an excerpt (when present and not all NaN)."""

CALIBRATION_MAX_CHARS = 6000
"""Size budget of the calibration file's content in ``survey_overview`` (shrunk like any over-long output)."""

THROUGHPUT_DROP_FRAC = 0.5
THROUGHPUT_MIN_BASELINE = 10
"""QC flag: objects done in the last 24 h fell below this fraction of the previous 24 h (if that had ≥ the baseline)."""

BACKUP_MAX_AGE_H = 36.0
"""QC flag: no catalog backup newer than this (the daemon backs up every 24 h)."""

QUERY_TIMEOUT_S = 60.0
"""Wall-clock bound of every catalog query made by a tool."""

MIN_DISTINCT_X_FOR_LAW = 3
MIN_SEEDS_PER_POINT = 3
MIN_DECADES = 1.0
MIN_R2 = 0.9
CURVATURE_ALPHA = 0.05
"""Thresholds of the caveats attached to a power-law fit: fewer distinct knob values, fewer seeds in some
configuration, a knob range under MIN_DECADES decades, R² below MIN_R2, or a curvature term significant at
CURVATURE_ALPHA each produce a caveat."""

PROPOSAL_PURPOSES: tuple[str, ...] = ("boundary", "replication", "extension", "control")
"""Why a configuration is proposed: refine a boundary between phenomena, add seeds where a pattern rests on too few,
extend coverage to unobserved points, or re-observe a control."""

GROUP_FIELDS: tuple[str, ...] = ("region", "task", "label", "seed", "sky_version", "source", "random_control")
"""Object-level fields ``query_summaries`` can group by (besides the knobs of a region and "point")."""

METRICS: tuple[str, ...] = tuple(k for k in SUMMARY_KEYS if k != "summary_version") + ("wall_time_s",)
"""Numeric quantities the aggregation tools accept: the photometric summary keys plus the object's wall time."""

NO_CALIBRATION_NOTE = ("no calibration file: the RI instrument has no measured noise floor yet, so every RI detection "
                       "is provisional (run `tycho calibrate`)")

_DAY_S = 86_400.0
_TAG = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")
_LATEST_LABEL_JOIN = ("classifications c ON c.id = "
                      "(SELECT MAX(c2.id) FROM classifications c2 WHERE c2.object_id = o.object_id)")
"""Join condition attaching an object's latest classification (alias ``c``) to ``objects o``."""
_SUMMARY_SELECT = ", ".join(f"s.{c}" for c in SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------------------------- limits and errors
@dataclass(frozen=True)
class ToolLimits:
    """Caps and size limits of the agent's tools (the ``agent`` section of the config may override any field).

    Attributes:
        max_output_chars: maximum length of a tool output (JSON characters).
        max_proposals_per_run: accepted proposals (configuration points) per run.
        max_objects_per_day: objects the agent may enqueue per 24 h (counted in the catalog: ``source = 'agent'``).
        max_followups_per_run: follow-up requests per run.
        max_labels_per_run: classification labels per run.
        max_seeds_per_point: largest ``seeds`` of one proposal.
        max_log_chars: largest survey-log entry or report (markdown characters).
        max_note_chars: largest note, reason or hypothesis.
    """

    max_output_chars: int = 20_000
    max_proposals_per_run: int = 50
    max_objects_per_day: int = 400
    max_followups_per_run: int = 20
    max_labels_per_run: int = 200
    max_seeds_per_point: int = 10
    max_log_chars: int = 60_000
    max_note_chars: int = 2_000

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ToolLimits.{f.name} must be a non-negative integer, got {value!r}")
        if self.max_output_chars < 1000:
            raise ValueError(f"max_output_chars must be >= 1000, got {self.max_output_chars}")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ToolLimits":
        """Limits from a survey config: every ToolLimits field present in ``cfg["agent"]`` overrides the default."""
        agent = cfg.get("agent") or {}
        return cls(**{f.name: int(agent[f.name]) for f in fields(cls) if f.name in agent})


class ToolError(Exception):
    """A tool call that cannot be carried out; the message (and details) go back to the model as an error result."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


# ---------------------------------------------------------------------------------------------- schemas
def _strict_object(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A strict object schema: every property required, no additional properties."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _integer(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def _number(description: str) -> dict[str, Any]:
    return {"type": "number", "description": description}


def _boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _enum(values: Sequence[str], description: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values), "description": description}


def _array(items: dict[str, Any], description: str) -> dict[str, Any]:
    return {"type": "array", "items": items, "description": description}


def _tool(name: str, description: str, properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "description": description, "strict": True, "input_schema": _strict_object(properties)}


_REGION_FILTER = _string('Region name, or "" for all regions.')
_SINCE_HOURS = _number("Look back this many hours from now; 0 = since the start of this run's window.")
_LIMIT = _integer(f"Maximum number of rows (1-{MAX_LIST_LIMIT}); 0 = {DEFAULT_LIST_LIMIT}.")
_FINISHED_WITHIN = _number("Only objects that finished within this many hours; 0 = no time restriction.")
_LABEL_FILTER = _string(f'Only objects whose latest label is this (one of {", ".join(LABELS)}); "" = any label.')
_KNOB_VALUE = _strict_object({
    "knob": _string('Knob name, e.g. "width" or "train.lr".'),
    "value": {"type": ["number", "string", "boolean", "null"],
              "description": "Knob value (number, string, boolean or null, exactly as in get_sky)."},
})
_WHERE = _array(_KNOB_VALUE, "Keep only objects whose knob has this value (needs a region); [] = no filter.")

TOOL_SPECS: tuple[dict[str, Any], ...] = (
    _tool("survey_overview",
          "Current state of the whole survey for quality control: object counts by status and region, queue, "
          "throughput in the last 24 h versus the previous 24 h, failures grouped by error message, divergences, "
          "worker heartbeats (alive, stale while holding an object, silent), free disk space, age of the latest "
          "backup, calibration status, recent agent runs, the caps left for this run, and QC flags. Call it at the "
          "start of a run.",
          {}),
    _tool("list_new_objects",
          "Objects that finished (done or failed) since a time, oldest first, with their knob point, latest "
          "classification (label, score, classifier, tags), key summary statistics and, for failed objects, the "
          "start of the error message. Use it for the classification review and to inspect failures; page with "
          "offset when total exceeds the limit.",
          {"since_hours": _SINCE_HOURS,
           "region": _REGION_FILTER,
           "status": _enum(("any", "done", "failed"), 'Status filter; "any" = done and failed.'),
           "limit": _LIMIT,
           "offset": _integer("Number of rows to skip (for paging); 0 = none.")}),
    _tool("get_object",
          "Everything about one object: configuration and knob point, the full photometric summary, the "
          "classification history, follow-up status, RI runs and events, transients and, with include_lightcurve, "
          f"a light-curve excerpt at about {LIGHTCURVE_EXCERPT_POINTS} log-spaced steps. Call it before labelling an "
          "object or requesting its follow-up.",
          {"object_id": _string("The object id (16 hex characters)."),
           "include_lightcurve": _boolean("Include the light-curve excerpt.")}),
    _tool("list_unknown_queue",
          "The human's review queue: objects whose latest classification is 'unknown', oldest first, with the "
          "classifier and its note, summary statistics and follow-up status. only_unreviewed=true hides objects "
          "you have already reviewed (their latest classification is your own 'unknown' with a note).",
          {"region": _REGION_FILTER,
           "only_unreviewed": _boolean("Hide objects whose latest classification is by the agent."),
           "limit": _LIMIT}),
    _tool("list_transients",
          "Transients (alert-worthy observations), most severe and newest first, with their object, the RI event "
          "they come from (subset, order, step) and whether a human has acknowledged them. Use it in the "
          "spectroscopy review.",
          {"since_hours": _SINCE_HOURS,
           "min_severity": _integer("Minimum severity (1-3); 0 = 1."),
           "kind": _enum(("any",) + TRANSIENT_KINDS, "Transient kind, or any."),
           "unacknowledged_only": _boolean("Only transients not yet acknowledged by a human."),
           "region": _REGION_FILTER,
           "limit": _LIMIT}),
    _tool("list_ri_events",
          "RI events: a variable subset activating, going extinct or reactivating at a training step, with the "
          "onset bracket, RI before, after and at its peak (nats), confidence, and the precursor and ground-truth "
          "flags. Use it to check the positive controls and to judge events against the noise floor.",
          {"object_id": _string('Only events of this object (then since_hours is ignored); "" = any object.'),
           "region": _REGION_FILTER,
           "kind": _enum(("any",) + EVENT_KINDS, "Event kind, or any."),
           "min_order": _integer("Minimum interaction order (subset size); 0 = any."),
           "ground_truth": _enum(("any", "match", "no_match"),
                                 "Filter on the ground-truth flag of positive controls: events that match the "
                                 "planted ground truth, events that do not, or any."),
           "since_hours": _SINCE_HOURS,
           "limit": _LIMIT}),
    _tool("query_summaries",
          "Aggregate photometric summary statistics of done objects by group. group_by takes object fields ("
          + ", ".join(GROUP_FIELDS) + "), knob names of the region, or 'point' (all knobs: one group per "
          "configuration, across its seeds); [] = one group. Returns per group the number of objects, the label "
          "counts and, per metric, n, mean, median, std, min and max. Knob grouping, 'point' and where-filters need "
          "a region. Use it for replication (spread across seeds), boundaries and trends.",
          {"region": _REGION_FILTER,
           "group_by": _array(_string("A group field."), f"Up to {MAX_GROUP_BY} group fields."),
           "metrics": _array(_enum(METRICS, "A summary metric."), f"1 to {MAX_METRICS} metrics to aggregate."),
           "label": _LABEL_FILTER,
           "where": _WHERE,
           "finished_within_hours": _FINISHED_WITHIN}),
    _tool("fit_power_law",
          "Fit a candidate power law metric = A * knob^b by least squares in log-log space over the done objects of "
          "one region, optionally restricted by where-filters (fix the other knobs to avoid confounding) and a "
          "label. aggregate 'point_median' uses one value per configuration (the median over its seeds) and is the "
          "basis for claims; 'objects' treats every seed as an observation. Returns the exponent b with its "
          "standard error and 95 % confidence interval, R^2, the residual scatter, the exponents obtained when "
          "leaving out one knob value at a time, a curvature test (quadratic term in log-log), exclusions "
          "(non-positive or missing values, e.g. objects that never reached a threshold) and caveats. A fit is a "
          "candidate law, never a finding.",
          {"region": _string("Region name."),
           "metric": _enum(METRICS, "The summary metric (y); values must be positive."),
           "knob": _string("A numeric knob of the region (x); values must be positive."),
           "where": _WHERE,
           "label": _LABEL_FILTER,
           "aggregate": _enum(("point_median", "objects"), "One observation per configuration, or per object."),
           "finished_within_hours": _FINISHED_WITHIN}),
    _tool("get_sky",
          "The sky definition: regions with task, knobs (values, kind, whether refinable, bounds), seeds per point, "
          "RI policy, priority, grid size and object counts. With a region name, also its fixed settings and "
          "training settings, how many configurations are observed (and with >= 3 seeds), its label counts, and "
          "the current boundary pairs: neighbouring points whose majority labels or generalization times differ, "
          "the best places for proposals.",
          {"region": _string('A region name for details, or "" for all regions.')}),
    _tool("list_proposals",
          "Proposed configurations (yours or the scheduler's), newest first, with the hypothesis or rationale and "
          "the outcome so far: objects at the proposed point by status, their labels, and the median t_test_acc90 "
          "and final_test_acc. Check it before proposing, to follow up earlier hypotheses and to avoid repeats.",
          {"since_hours": _SINCE_HOURS,
           "source": _enum(("any", "agent", "scheduler"), "Who proposed, or any."),
           "region": _REGION_FILTER,
           "limit": _LIMIT}),
    _tool("read_survey_log",
          "Entries of the survey log (nightly logs, weekly reports, notes), newest first, with their markdown "
          "truncated per entry. Use it for continuity with earlier nights and, for the weekly report, to read the "
          "week's nightly logs.",
          {"since_hours": _SINCE_HOURS,
           "kind": _enum(("any", "nightly", "weekly"), "Entry kind, or any."),
           "limit": _LIMIT,
           "max_chars_per_entry": _integer(f"Markdown characters per entry (1-20000); 0 = "
                                           f"{DEFAULT_LOG_ENTRY_CHARS}.")}),
    _tool("propose_configs",
          "Enqueue new objects at configurations of the sky (source 'agent'). Each proposal names a region, a point "
          "(a value for every knob of the region: a grid value, or for refinable knobs any value within the "
          "bounds), the total number of seeds the point should have (seeds 0..n-1; objects that already exist are "
          "not duplicated; 0 = the region's default), a purpose, the hypothesis it tests and the result that would "
          "falsify it. Points are validated against the sky, and caps per run and per 24 h apply; each proposal "
          f"is accepted or rejected on its own. At most {MAX_PROPOSALS_PER_CALL} proposals per call.",
          {"proposals": _array(_strict_object({
              "region": _string("Region name."),
              "point": _array(_KNOB_VALUE, "One entry per knob of the region."),
              "seeds": _integer("Total seeds the point should have; 0 = the region's default."),
              "purpose": _enum(PROPOSAL_PURPOSES, "Why this point."),
              "hypothesis": _string("The expectation this point tests, with numbers where possible."),
              "falsified_if": _string("The observation that would falsify the hypothesis."),
          }), "The proposals.")}),
    _tool("label_object",
          "Record your classification of a done object (classifier 'agent'). Give a known label only when the light "
          "curve clearly shows that phenomenon and cite the deciding statistics in the note; give 'unknown' with a "
          "note to leave the object in the human's review queue with your observations.",
          {"object_id": _string("The object id."),
           "label": _enum(LABELS, "The label."),
           "note": _string("The evidence: which statistics or curve features decide the label."),
           "confidence": _number("Your confidence in the label, from 0 to 1."),
           "tags": _array(_string("A short lowercase identifier, e.g. needs_replay."), "Tags; [] = none.")}),
    _tool("request_followup",
          "Request RI spectroscopy of a done object: a deterministic replay with dense RI checkpoints around its "
          "largest loss drop. Worth it when the RI timeline would decide between hypotheses (transients, boundary "
          "members, unknowns, controls that did not recover their ground truth). Not available in regions with RI "
          "policy 'never'. An object is followed up at most once; capped per run.",
          {"object_id": _string("The object id."),
           "reason": _string("What the replay should decide.")}),
    _tool("write_survey_log",
          "Write tonight's survey log entry (markdown). Call it exactly once, at the very end of the run, after all "
          "other tool calls.",
          {"title": _string("A one-line title."),
           "markdown": _string("The log in markdown.")}),
    _tool("write_report",
          "Write this week's report (markdown). It is saved as reports/weekly/<year>-W<week>.md and as a survey-log "
          "entry. Call it exactly once, at the very end of the run.",
          {"title": _string("A one-line title."),
           "markdown": _string("The report in markdown.")}),
)
"""All tool definitions, in a fixed order (the request's tool list must be byte-stable for prompt caching)."""

NIGHTLY_TOOLS: tuple[str, ...] = tuple(t["name"] for t in TOOL_SPECS if t["name"] != "write_report")
WEEKLY_TOOLS: tuple[str, ...] = tuple(
    t["name"] for t in TOOL_SPECS
    if t["name"] not in ("propose_configs", "label_object", "request_followup", "write_survey_log"))
_TOOLS_BY_KIND: dict[str, tuple[str, ...]] = {"nightly": NIGHTLY_TOOLS, "weekly": WEEKLY_TOOLS}
_SPEC_BY_NAME: dict[str, dict[str, Any]] = {t["name"]: t for t in TOOL_SPECS}


def tool_specs(kind: str) -> list[dict[str, Any]]:
    """Copies of the tool definitions available to a run of ``kind``, in :data:`TOOL_SPECS` order."""
    if kind not in _TOOLS_BY_KIND:
        raise ValueError(f"unknown agent run kind {kind!r}; expected one of {KINDS}")
    allowed = _TOOLS_BY_KIND[kind]
    return [copy.deepcopy(t) for t in TOOL_SPECS if t["name"] in allowed]


def validate_input(value: Any, schema: Mapping[str, Any], path: str = "input") -> list[str]:
    """Problems of ``value`` against a strict tool schema ([] if it conforms).

    Supports the subset of JSON Schema used by :data:`TOOL_SPECS`: ``type`` (a name or a list of names), ``enum``,
    ``properties`` / ``required`` / ``additionalProperties: false`` and ``items``.
    """
    allowed = schema.get("type")
    allowed = [allowed] if isinstance(allowed, str) else list(allowed or [])
    if allowed and not any(_is_json_type(value, t) for t in allowed):
        return [f"{path}: expected {' or '.join(allowed)}, got {_json_type_name(value)}"]
    if "enum" in schema and not any(_same_json(value, e) for e in schema["enum"]):
        return [f"{path}: {value!r} is not one of {schema['enum']}"]
    errors: list[str] = []
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        errors += [f"{path}: missing property {k!r}" for k in schema.get("required", []) if k not in value]
        for key, item in value.items():
            if key not in properties:
                errors.append(f"{path}: unexpected property {key!r}")
            else:
                errors += validate_input(item, properties[key], f"{path}.{key}")
    elif isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate_input(item, schema["items"], f"{path}[{i}]")
    return errors


def _is_json_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise ValueError(f"unknown JSON schema type {name!r}")


def _json_type_name(value: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _is_json_type(value, name):
            return name
    return type(value).__name__


def _same_json(a: Any, b: Any) -> bool:
    """JSON equality (``True`` is not ``1``)."""
    return type(a) is type(b) and a == b


# ---------------------------------------------------------------------------------------------- JSON output
def _round_sig(x: float, digits: int = FLOAT_SIG_DIGITS) -> float:
    return float(f"{x:.{digits}g}")


def jsonable(o: Any) -> Any:
    """Plain-JSON form of ``o`` (a new structure): floats rounded to :data:`FLOAT_SIG_DIGITS` significant digits,
    NaN/inf → None, numpy scalars/arrays → numbers/lists, tuples/sets → lists, paths → POSIX strings."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return _round_sig(f) if math.isfinite(f) else None
    if isinstance(o, Mapping):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [jsonable(v) for v in sorted(o, key=repr)]
    if isinstance(o, PurePath):
        return o.as_posix()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _dumps(o: Any) -> str:
    """Compact JSON of an already plain-JSON structure."""
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _shrink(payload: Any, limit: int) -> tuple[Any, str]:
    """(plain-JSON payload, its JSON) no longer than ``limit`` characters if lists can be shortened enough.

    While too long, the fewest tail items of the largest list (by serialized size) whose removal covers the excess are
    dropped; the counts dropped per list path are reported under ``"_truncated"`` (a non-dict payload is wrapped as
    ``{"result": ...}``).
    """
    data = jsonable(payload)
    text = _dumps(data)
    if len(text) <= limit:
        return data, text
    if not isinstance(data, dict):
        data = {"result": data}
    omitted: dict[str, int] = {}
    while len(text) > limit:
        found = _largest_list(data, "")
        if found is None:
            break
        _, path, items = found
        excess, n_drop = len(text) - limit, 0
        while n_drop < len(items) and excess > 0:
            n_drop += 1
            excess -= len(_dumps(items[-n_drop])) + 1  # the item and its separating comma
        del items[len(items) - n_drop:]
        omitted[path] = omitted.get(path, 0) + n_drop
        data["_truncated"] = {"omitted_items": omitted,
                              "hint": f"output exceeded {limit} characters; narrow the request (filters, limit)"}
        text = _dumps(data)
    return data, text


def _largest_list(node: Any, path: str) -> tuple[int, str, list] | None:
    """(serialized size, path, list) of the largest non-empty list inside ``node``."""
    best: tuple[int, str, list] | None = None
    if isinstance(node, list):
        if node:
            best = (len(_dumps(node)), path or "result", node)
        children = [(f"{path}[{i}]", child) for i, child in enumerate(node)]
    elif isinstance(node, dict):
        children = [(f"{path}.{key}" if path else str(key), child) for key, child in node.items()]
    else:
        return None
    for child_path, child in children:
        found = _largest_list(child, child_path)
        if found is not None and (best is None or found[0] > best[0]):
            best = found
    return best


def fit_json(payload: Any, limit: int) -> str:
    """JSON text of ``payload`` of at most ``limit`` characters (see :func:`_shrink`). If no list is left to shorten
    (e.g. a single huge string), the text is cut hard and marked; tools keep their strings short, so this is a last
    resort that yields invalid JSON rather than an unbounded output."""
    _, text = _shrink(payload, limit)
    if len(text) > limit:
        marker = f"...[cut: output exceeded {limit} characters]"
        text = text[: max(0, limit - len(marker))] + marker
    return text


# ---------------------------------------------------------------------------------------------- time and files
def iso_utc(t: float | None) -> str | None:
    """ISO-8601 UTC timestamp (seconds precision, ``Z`` suffix) of epoch seconds; None stays None."""
    if t is None or not math.isfinite(float(t)):
        return None
    return datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def local_time(t: float) -> str:
    """Local wall-clock time with its UTC offset, e.g. ``2026-11-04 03:00 JST (UTC+09:00)``."""
    dt = datetime.fromtimestamp(float(t)).astimezone()
    offset = dt.strftime("%z")
    return f"{dt:%Y-%m-%d %H:%M} {dt.tzname()} (UTC{offset[:3]}:{offset[3:]})"


def local_date(t: float) -> str:
    """Local calendar date ``YYYY-MM-DD`` (names of nightly report files)."""
    return datetime.fromtimestamp(float(t)).astimezone().strftime("%Y-%m-%d")


def iso_week(t: float) -> str:
    """Local ISO week ``YYYY-Www`` (names of weekly report files)."""
    year, week, _ = datetime.fromtimestamp(float(t)).astimezone().isocalendar()
    return f"{year}-W{week:02d}"


def unique_path(directory: Path, stem: str, suffix: str = ".md") -> Path:
    """``directory/stem+suffix``, or ``stem-2``, ``stem-3``, ... if taken: reports are never overwritten."""
    path = directory / f"{stem}{suffix}"
    n = 2
    while path.exists():
        path = directory / f"{stem}-{n}{suffix}"
        n += 1
    return path


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary file and an atomic rename (creating the directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------- power-law fit
def power_law_fit(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, *,
                  level: float = 0.95) -> dict[str, Any]:
    """Least-squares fit of the power law y = A·x^b, i.e. of log10 y = log10 A + b·log10 x.

    Args:
        x, y: positive finite observations (the caller filters; ValueError otherwise).
        level: confidence level of the slope interval (Student t with n − 2 degrees of freedom).

    Returns:
        A dict with n, n_distinct_x, slope (b), slope_se, slope_ci, intercept_log10 (log10 A), prefactor (A), r2
        (None when y is constant), residual_sd_dex, df, ``jackknife`` (slopes refitted with all observations at one
        distinct x left out: min, max and the slope per left-out x; None with fewer than 3 distinct x) and
        ``curvature`` (the coefficient c of a quadratic term c·(log10 x − mean)², its standard error and two-sided p;
        None with fewer than 3 distinct x or 4 observations; p is None for an exact fit).

    Raises:
        ValueError: fewer than 3 observations or 2 distinct x values, or non-positive / non-finite data.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y differ in length ({x.size} vs {y.size})")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(x > 0) and np.all(y > 0)):
        raise ValueError("power-law fit needs positive finite x and y")
    lx, ly = np.log10(x), np.log10(y)
    distinct = np.unique(lx)
    if lx.size < 3 or distinct.size < 2:
        raise ValueError(f"need >= 3 observations at >= 2 distinct knob values, got {lx.size} at {distinct.size}")
    slope, intercept, sse = _ols(lx, ly)
    n = lx.size
    df = n - 2
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    se = math.sqrt(sse / df / sxx)
    t_q = float(stats.t.ppf(0.5 + level / 2.0, df))
    sst = float(np.sum((ly - ly.mean()) ** 2))
    out: dict[str, Any] = {
        "n": int(n), "n_distinct_x": int(distinct.size), "slope": slope, "slope_se": se,
        "slope_ci": [slope - t_q * se, slope + t_q * se], "ci_level": level, "intercept_log10": intercept,
        "prefactor": 10.0 ** intercept, "r2": (1.0 - sse / sst) if sst > 0 else None,
        "residual_sd_dex": math.sqrt(sse / df), "df": int(df), "jackknife": None, "curvature": None,
    }
    if distinct.size >= 3:
        by_x = {}
        for value in distinct:
            keep = lx != value
            if keep.sum() >= 3 and np.unique(lx[keep]).size >= 2:
                by_x[float(10.0 ** value)] = _ols(lx[keep], ly[keep])[0]
        if by_x:
            out["jackknife"] = {"min": min(by_x.values()), "max": max(by_x.values()),
                                "slope_without_x": [{"x": k, "slope": v} for k, v in sorted(by_x.items())]}
        if n >= 4:
            out["curvature"] = _curvature(lx, ly)
    return out


def _ols(lx: np.ndarray, ly: np.ndarray) -> tuple[float, float, float]:
    """(slope, intercept, sum of squared residuals) of the simple regression of ly on lx."""
    xm, ym = lx.mean(), ly.mean()
    slope = float(np.sum((lx - xm) * (ly - ym)) / np.sum((lx - xm) ** 2))
    intercept = float(ym - slope * xm)
    sse = float(np.sum((ly - intercept - slope * lx) ** 2))
    return slope, intercept, sse


def _curvature(lx: np.ndarray, ly: np.ndarray) -> dict[str, Any]:
    """Quadratic term of log10 y on centred log10 x: coefficient, standard error and two-sided p (t, n − 3 df)."""
    xc = lx - lx.mean()
    design = np.column_stack([np.ones_like(xc), xc, xc ** 2])
    beta, *_ = np.linalg.lstsq(design, ly, rcond=None)
    df = lx.size - 3
    sse = float(np.sum((ly - design @ beta) ** 2))
    cov = (sse / df) * np.linalg.inv(design.T @ design)
    se = math.sqrt(max(float(cov[2, 2]), 0.0))
    coef = float(beta[2])
    p = float(2.0 * stats.t.sf(abs(coef) / se, df)) if se > 0 else None
    return {"coef": coef, "se": se, "p_value": p, "df": int(df)}


# ---------------------------------------------------------------------------------------------- the tools
class SurveyTools:
    """The tools of one agent run, bound to a catalog, a sky and a survey root.

    Args:
        catalog: the survey catalog (all reads go through ``Catalog.query``; its clock is the tools' clock).
        sky: the current sky (proposals are validated against it).
        paths: the survey root's paths (reports and backups; the calibration file).
        limits: caps and size limits (default :class:`ToolLimits`).
        kind: "nightly" or "weekly"; selects the tool set (:data:`NIGHTLY_TOOLS` / :data:`WEEKLY_TOOLS`).
        config: the survey config (default :data:`tycho.config.DEFAULT_CONFIG`): ``survey.stale_after_s`` and
            ``survey.min_free_gb`` for QC, ``ri.calibration_file`` for the calibration status.
        window_start: start of the run's observation window (epoch seconds); default 24 h before now.

    Per-run state: the counters of accepted writes (``n_proposals``, ``n_followups``, ``n_labels``), the survey-log
    entry or report written (``log_entry_id``, ``report_entry_id``, ``report_path``) and ``call_counts`` per tool.
    """

    def __init__(self, catalog: Catalog, sky: Sky, paths: Paths, limits: ToolLimits | None = None, *,
                 kind: str = "nightly", config: Mapping[str, Any] | None = None,
                 window_start: float | None = None) -> None:
        if kind not in _TOOLS_BY_KIND:
            raise ValueError(f"unknown agent run kind {kind!r}; expected one of {KINDS}")
        self.catalog = catalog
        self.sky = sky
        self.paths = paths
        self.limits = limits if limits is not None else ToolLimits()
        self.kind = kind
        self.config: Mapping[str, Any] = config if config is not None else DEFAULT_CONFIG
        self.window_start = float(window_start) if window_start is not None else self.now() - _DAY_S
        self.n_proposals = 0
        self.n_followups = 0
        self.n_labels = 0
        self.log_entry_id: int | None = None
        self.report_entry_id: int | None = None
        self.report_path: Path | None = None
        self.call_counts: Counter[str] = Counter()
        self._allowed = frozenset(_TOOLS_BY_KIND[kind])
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "survey_overview": lambda args: self.overview(),
            "list_new_objects": self._list_new_objects,
            "get_object": self._get_object,
            "list_unknown_queue": self._list_unknown_queue,
            "list_transients": self._list_transients,
            "list_ri_events": self._list_ri_events,
            "query_summaries": self._query_summaries,
            "fit_power_law": self._fit_power_law,
            "get_sky": self._get_sky,
            "list_proposals": self._list_proposals,
            "read_survey_log": self._read_survey_log,
            "propose_configs": self._propose_configs,
            "label_object": self._label_object,
            "request_followup": self._request_followup,
            "write_survey_log": self._write_survey_log,
            "write_report": self._write_report,
        }

    # ------------------------------------------------------------------ public interface
    @property
    def specs(self) -> list[dict[str, Any]]:
        """The tool definitions of this run's kind (sent with every request)."""
        return tool_specs(self.kind)

    def now(self) -> float:
        return self.catalog.now()

    def dispatch(self, name: str, input: Mapping[str, Any]) -> tuple[str, bool]:  # noqa: A002 (SPEC §7 signature)
        """Run tool ``name`` on ``input``; returns (JSON text, is_error).

        Unknown tools, tools outside this run's set, schema violations and :class:`ToolError` become error results.
        Any other exception propagates (the runner reports it to the model as an internal error and logs it).
        """
        self.call_counts[name] += 1
        limit = self.limits.max_output_chars
        if name not in self._allowed:
            return fit_json({"error": f"unknown tool {name!r} for a {self.kind} run; available: "
                                      f"{sorted(self._allowed)}"}, limit), True
        if not isinstance(input, Mapping):
            return fit_json({"error": f"tool input must be a JSON object, got {_json_type_name(input)}"}, limit), True
        args = dict(input)
        problems = validate_input(args, _SPEC_BY_NAME[name]["input_schema"])
        if problems:
            return fit_json({"error": "invalid input: " + "; ".join(problems[:10])}, limit), True
        try:
            payload = self._handlers[name](args)
        except ToolError as exc:
            return fit_json({"error": exc.message, **exc.details}, limit), True
        return fit_json(payload, limit), False

    # ------------------------------------------------------------------ overview / QC
    def overview(self) -> dict[str, Any]:
        """The survey's state for quality control (the ``survey_overview`` tool and the context packets)."""
        now = self.now()
        st = self.catalog.stats()
        survey_cfg = self.config.get("survey") or {}
        stale_after = float(survey_cfg.get("stale_after_s", 300.0))
        min_free_gb = float(survey_cfg.get("min_free_gb", 20.0))
        workers = st["workers"]
        alive = [w for w in workers if w["age_s"] < stale_after]
        stale = [{"worker_id": w["worker_id"], "object_id": w["object_id"], "status": w["status"],
                  "age_s": w["age_s"]} for w in workers if w["age_s"] >= stale_after and w["object_id"]]
        silent = [w for w in workers if w["age_s"] >= stale_after and not w["object_id"]]
        done_prev = self._q("SELECT COUNT(*) AS n FROM objects WHERE status = 'done' AND finished_at >= ? "
                            "AND finished_at < ?", (now - 2 * _DAY_S, now - _DAY_S))[0]["n"]
        wall = [r["wall_time_s"] for r in self._q(
            "SELECT wall_time_s FROM objects WHERE status = 'done' AND finished_at >= ? AND wall_time_s IS NOT NULL",
            (now - _DAY_S,))]
        throughput = {"done_last_24h": st["done_last_24h"], "done_prev_24h": done_prev,
                      "failed_last_24h": st["failed_last_24h"],
                      "median_wall_time_s_last_24h": float(np.median(wall)) if wall else None}
        divergences = {r["region"]: r["n"] for r in self._q(
            "SELECT o.region AS region, COUNT(*) AS n FROM objects o JOIN summaries s ON s.object_id = o.object_id "
            "WHERE o.status = 'done' AND s.diverged = 1 AND o.finished_at >= ? GROUP BY o.region ORDER BY o.region",
            (now - _DAY_S,))}
        disk = self._disk(min_free_gb)
        backup_age_h = self._latest_backup_age_h()
        calibration = self.calibration_status()
        out: dict[str, Any] = {
            "now": iso_utc(now),
            "window_start": iso_utc(self.window_start),
            "sky_version": self.sky.version,
            "objects": st["objects"], "queue": st["queue"], "by_region": st["by_region"], "labels": st["labels"],
            "followups": st["followups"], "ri_runs": st["ri_runs"], "ri_events": st["ri_events"],
            "transients": st["transients"], "proposals": st["proposals"],
            "throughput": throughput,
            "failures_last_24h": self._failure_groups(now - _DAY_S),
            "divergences_last_24h": divergences,
            "workers": {"alive": len(alive), "stale_holding_object": stale, "silent": len(silent),
                        "stale_after_s": stale_after},
            "disk": disk,
            "latest_backup_age_h": backup_age_h,
            "calibration": calibration,
            "agent_runs": self._recent_agent_runs(now),
        }
        if self.kind == "nightly":
            out["caps"] = self.caps()
        out["qc_flags"] = self._qc_flags(out, now)
        return out

    def caps(self) -> dict[str, int]:
        """What is left of the per-run and per-day write caps."""
        lim = self.limits
        return {"proposals_left_this_run": max(0, lim.max_proposals_per_run - self.n_proposals),
                "objects_left_today": max(0, lim.max_objects_per_day - self._agent_objects_last_day()),
                "followups_left_this_run": max(0, lim.max_followups_per_run - self.n_followups),
                "labels_left_this_run": max(0, lim.max_labels_per_run - self.n_labels)}

    def calibration_status(self) -> dict[str, Any]:
        """Whether the calibration file (``ri.calibration_file``) exists, its age and its content (size-capped)."""
        rel = str((self.config.get("ri") or {}).get("calibration_file", "reports/calibration/calibration.json"))
        path = self.paths.root / rel
        if not path.is_file():
            return {"present": False, "file": rel, "note": NO_CALIBRATION_NOTE}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
        except (OSError, ValueError) as exc:
            return {"present": True, "file": rel, "readable": False, "error": f"{type(exc).__name__}: {exc}"}
        content, _ = _shrink(data, CALIBRATION_MAX_CHARS)
        return {"present": True, "file": rel, "readable": True,
                "age_days": max(0.0, (time.time() - mtime) / _DAY_S), "content": content}

    def period_summary(self, since: float) -> dict[str, Any]:
        """What happened since ``since`` (epoch seconds): finished objects, labels, transients, RI events, follow-ups,
        proposals, agent runs, survey-log entries and positive-control recovery. Used by the context packets."""
        q = self._q
        finished: dict[str, dict[str, int]] = {}
        for r in q("SELECT region, status, COUNT(*) AS n FROM objects WHERE finished_at >= ? "
                   "AND status IN ('done', 'failed') GROUP BY region, status ORDER BY region", (since,)):
            finished.setdefault(r["region"], {"done": 0, "failed": 0})[r["status"]] = r["n"]
        labels = {r["label"]: r["n"] for r in q(
            f"SELECT COALESCE(c.label, 'unlabelled') AS label, COUNT(*) AS n FROM objects o "
            f"LEFT JOIN {_LATEST_LABEL_JOIN} WHERE o.status = 'done' AND o.finished_at >= ? GROUP BY 1 ORDER BY 1",
            (since,))}
        transients = [dict(r) for r in q("SELECT kind, severity, COUNT(*) AS n FROM transients WHERE created_at >= ? "
                                         "GROUP BY kind, severity ORDER BY severity DESC, kind", (since,))]
        events = [{"order": r["order_k"], "kind": r["kind"], "n": r["n"], "matching_gt": r["matching_gt"],
                   "not_matching_gt": r["not_matching_gt"]} for r in q(
            "SELECT order_k, kind, COUNT(*) AS n, COALESCE(SUM(matches_gt = 1), 0) AS matching_gt, "
            "COALESCE(SUM(matches_gt = 0), 0) AS not_matching_gt FROM ri_events WHERE created_at >= ? "
            "GROUP BY order_k, kind ORDER BY order_k, kind", (since,))]
        followups = {
            "requested": {r["requested_by"] or "unknown": r["n"] for r in q(
                "SELECT requested_by, COUNT(*) AS n FROM followups WHERE created_at >= ? GROUP BY requested_by "
                "ORDER BY requested_by", (since,))},
            "finished": {r["status"]: r["n"] for r in q(
                "SELECT status, COUNT(*) AS n FROM followups WHERE finished_at >= ? GROUP BY status ORDER BY status",
                (since,))}}
        proposals = [dict(r) for r in q(
            "SELECT source, status, COUNT(*) AS n, COALESCE(SUM(n_enqueued), 0) AS objects_enqueued FROM proposals "
            "WHERE created_at >= ? GROUP BY source, status ORDER BY source, status", (since,))]
        agent = [dict(r) for r in q(
            "SELECT kind, status, COUNT(*) AS n, COALESCE(SUM(est_cost_usd), 0.0) AS est_cost_usd FROM agent_runs "
            "WHERE started_at >= ? GROUP BY kind, status ORDER BY kind, status", (since,))]
        log = [{"entry_id": r["entry_id"], "created": iso_utc(r["created_at"]), "author": r["author"],
                "kind": r["kind"], "title": r["title"]} for r in q(
            "SELECT entry_id, created_at, author, kind, title FROM survey_log WHERE created_at >= ? "
            "ORDER BY entry_id DESC LIMIT ?", (since, MAX_LOG_TITLES))]
        return {"since": iso_utc(since), "until": iso_utc(self.now()), "objects_finished": finished,
                "labels_of_done": labels, "transients": transients, "ri_events": events, "followups": followups,
                "proposals": proposals, "agent_runs": agent, "survey_log": log,
                "positive_controls": self._control_recovery(since)}

    # ------------------------------------------------------------------ list tools
    def _list_new_objects(self, a: dict[str, Any]) -> dict[str, Any]:
        since = self._since(a["since_hours"])
        limit, offset = self._limit(a["limit"]), self._offset(a["offset"])
        statuses = ("done", "failed") if a["status"] == "any" else (a["status"],)
        where = ["o.finished_at >= ?", f"o.status IN ({', '.join('?' * len(statuses))})"]
        params: list[Any] = [since, *statuses]
        region = self._region_filter(a["region"])
        if region:
            where.append("o.region = ?")
            params.append(region)
        cond = " AND ".join(where)
        total = self._q(f"SELECT COUNT(*) AS n FROM objects o WHERE {cond}", params)[0]["n"]
        label_counts = {r["label"]: r["n"] for r in self._q(
            f"SELECT COALESCE(c.label, 'unlabelled') AS label, COUNT(*) AS n FROM objects o "
            f"LEFT JOIN {_LATEST_LABEL_JOIN} WHERE {cond} GROUP BY 1 ORDER BY 1", params)}
        rows = self._q(
            f"SELECT o.object_id, o.region, o.task, o.seed, o.status, o.source, o.random_control, o.spec_json, "
            f"o.finished_at, o.wall_time_s, o.error, {_SUMMARY_SELECT}, c.label, c.score, c.classifier, c.tags_json "
            f"FROM objects o LEFT JOIN summaries s ON s.object_id = o.object_id LEFT JOIN {_LATEST_LABEL_JOIN} "
            f"WHERE {cond} ORDER BY o.finished_at, o.object_id LIMIT ? OFFSET ?", [*params, limit, offset])
        return {"window_since": iso_utc(since), "total": total, "offset": offset, "returned": len(rows),
                "label_counts": label_counts, "objects": [self._brief(r) for r in rows]}

    def _get_object(self, a: dict[str, Any]) -> dict[str, Any]:
        oid = a["object_id"].strip()
        try:
            obj = self.catalog.get_object(oid)
        except KeyError:
            raise ToolError(f"unknown object id {oid!r}") from None
        spec = obj["spec"]
        out: dict[str, Any] = {
            "object_id": oid, "task": obj["task"], "region": obj["region"], "seed": obj["seed"],
            "status": obj["status"], "source": obj["source"], "priority": obj["priority"],
            "attempts": obj["attempts"], "random_control": bool(obj["random_control"]),
            "sky_version": obj["sky_version"], "created": iso_utc(obj["created_at"]),
            "started": iso_utc(obj["started_at"]), "finished": iso_utc(obj["finished_at"]),
            "wall_time_s": obj["wall_time_s"], "device": obj["device"], "n_params": obj["n_params"],
            "code_version": obj["code_version"],
            "error": _clip(obj["error"], self.limits.max_note_chars),
            "point": self._point_of(obj["region"], obj["spec_json"]),
            "task_cfg": spec.get("task_cfg"), "train": spec.get("train"),
            "summary": obj["summary"],
        }
        out["classifications"] = [
            {"label": r["label"], "tags": json.loads(r["tags_json"] or "[]"), "score": r["score"],
             "classifier": r["classifier"], "note": _clip(r["note"], self.limits.max_note_chars),
             "created": iso_utc(r["created_at"])}
            for r in self._q("SELECT label, tags_json, score, classifier, note, created_at FROM classifications "
                             "WHERE object_id = ? ORDER BY id DESC LIMIT ?", (oid, MAX_CLASSIFICATIONS_PER_OBJECT))]
        fu = obj["followup"]
        out["followup"] = None if fu is None else {
            "status": fu["status"], "reason": fu["reason"], "requested_by": fu["requested_by"],
            "priority": fu["priority"], "requested": iso_utc(fu["created_at"]),
            "finished": iso_utc(fu["finished_at"]), "error": _clip(fu["error"], ERROR_PREVIEW_CHARS)}
        out["ri_runs"] = [{"ri_run_id": r["ri_run_id"], "kind": r["kind"], "n_subsets": r["n_subsets"],
                           "n_points": r["n_points"], "max_order": r["max_order"], "alpha": r["alpha"],
                           "min_effect": r["min_effect"], "created": iso_utc(r["created_at"])}
                          for r in obj["ri_runs"]]
        out["n_ri_events"] = obj["n_ri_events"]
        out["ri_events"] = [self._event(r) for r in self._q(
            "SELECT e.*, r.kind AS run_kind FROM ri_events e JOIN ri_runs r ON r.ri_run_id = e.ri_run_id "
            "WHERE e.object_id = ? ORDER BY e.step, e.event_id LIMIT ?", (oid, MAX_EVENTS_PER_OBJECT))]
        out["transients"] = [{"transient_id": t["transient_id"], "kind": t["kind"], "severity": t["severity"],
                              "message": t["message"], "acknowledged": bool(t["acknowledged"]),
                              "created": iso_utc(t["created_at"])} for t in obj["transients"]]
        if a["include_lightcurve"]:
            lc = obj["lightcurve"]
            out["lightcurve"] = (_lightcurve_excerpt(Path(lc["path"])) if lc is not None
                                 else {"available": False, "reason": "no light curve recorded"})
        return out

    def _list_unknown_queue(self, a: dict[str, Any]) -> dict[str, Any]:
        limit = self._limit(a["limit"])
        where, params = ["c.label = 'unknown'"], []
        region = self._region_filter(a["region"])
        if region:
            where.append("o.region = ?")
            params.append(region)
        if a["only_unreviewed"]:
            where.append("COALESCE(c.classifier, '') != ?")
            params.append(AGENT_SOURCE)
        cond = " AND ".join(where)
        total = self._q(f"SELECT COUNT(*) AS n FROM objects o JOIN {_LATEST_LABEL_JOIN} WHERE {cond}",
                        params)[0]["n"]
        rows = self._q(
            f"SELECT o.object_id, o.region, o.task, o.seed, o.status, o.source, o.random_control, o.spec_json, "
            f"o.finished_at, o.wall_time_s, o.error, {_SUMMARY_SELECT}, c.label, c.score, c.classifier, c.tags_json, "
            f"c.note, c.created_at AS labelled_at, f.status AS followup_status FROM objects o "
            f"JOIN {_LATEST_LABEL_JOIN} LEFT JOIN summaries s ON s.object_id = o.object_id "
            f"LEFT JOIN followups f ON f.object_id = o.object_id "
            f"WHERE {cond} ORDER BY o.finished_at, o.object_id LIMIT ?", [*params, limit])
        objects = []
        for r in rows:
            item = self._brief(r)
            item.update(note=_clip(r["note"], self.limits.max_note_chars), labelled=iso_utc(r["labelled_at"]),
                        followup=r["followup_status"])
            objects.append(item)
        return {"total": total, "returned": len(objects), "objects": objects}

    def _list_transients(self, a: dict[str, Any]) -> dict[str, Any]:
        since = self._since(a["since_hours"])
        limit = self._limit(a["limit"])
        min_severity = a["min_severity"] or 1
        if not 1 <= min_severity <= 3:
            raise ToolError(f"min_severity must be 0 (default) or 1-3, got {min_severity}")
        where, params = ["t.created_at >= ?", "t.severity >= ?"], [since, min_severity]
        if a["kind"] != "any":
            where.append("t.kind = ?")
            params.append(a["kind"])
        if a["unacknowledged_only"]:
            where.append("t.acknowledged = 0")
        region = self._region_filter(a["region"])
        if region:
            where.append("o.region = ?")
            params.append(region)
        cond = " AND ".join(where)
        base = ("FROM transients t JOIN objects o ON o.object_id = t.object_id "
                "LEFT JOIN ri_events e ON e.event_id = t.event_id")
        total = self._q(f"SELECT COUNT(*) AS n {base} WHERE {cond}", params)[0]["n"]
        rows = self._q(
            f"SELECT t.transient_id, t.object_id, t.kind, t.severity, t.message, t.created_at, t.notified, "
            f"t.acknowledged, t.note, o.task, o.region, o.seed, e.subset_json, e.order_k, e.step AS event_step, "
            f"e.kind AS event_kind {base} WHERE {cond} "
            f"ORDER BY t.severity DESC, t.created_at DESC, t.transient_id DESC LIMIT ?", [*params, limit])
        out = []
        for r in rows:
            item = {"transient_id": r["transient_id"], "object_id": r["object_id"], "kind": r["kind"],
                    "severity": r["severity"], "message": r["message"], "created": iso_utc(r["created_at"]),
                    "notified": bool(r["notified"]), "acknowledged": bool(r["acknowledged"]),
                    "task": r["task"], "region": r["region"], "seed": r["seed"]}
            if r["note"]:
                item["note"] = _clip(r["note"], self.limits.max_note_chars)
            if r["event_kind"] is not None:
                item["event"] = {"subset": json.loads(r["subset_json"] or "[]"), "order": r["order_k"],
                                 "kind": r["event_kind"], "step": r["event_step"]}
            out.append(item)
        return {"window_since": iso_utc(since), "total": total, "returned": len(out), "transients": out}

    def _list_ri_events(self, a: dict[str, Any]) -> dict[str, Any]:
        limit = self._limit(a["limit"])
        where: list[str] = []
        params: list[Any] = []
        oid = a["object_id"].strip()
        if oid:
            where.append("e.object_id = ?")
            params.append(oid)
        else:
            where.append("e.created_at >= ?")
            params.append(self._since(a["since_hours"]))
        region = self._region_filter(a["region"])
        if region:
            where.append("o.region = ?")
            params.append(region)
        if a["kind"] != "any":
            where.append("e.kind = ?")
            params.append(a["kind"])
        if a["min_order"] < 0:
            raise ToolError(f"min_order must be >= 0, got {a['min_order']}")
        if a["min_order"] > 0:
            where.append("e.order_k >= ?")
            params.append(a["min_order"])
        if a["ground_truth"] != "any":
            where.append("e.matches_gt = ?")
            params.append(1 if a["ground_truth"] == "match" else 0)
        cond = " AND ".join(where)
        base = ("FROM ri_events e JOIN ri_runs r ON r.ri_run_id = e.ri_run_id "
                "JOIN objects o ON o.object_id = e.object_id")
        total = self._q(f"SELECT COUNT(*) AS n {base} WHERE {cond}", params)[0]["n"]
        by_order = {str(r["order_k"]): r["n"] for r in self._q(
            f"SELECT e.order_k, COUNT(*) AS n {base} WHERE {cond} GROUP BY e.order_k ORDER BY e.order_k", params)}
        rows = self._q(f"SELECT e.*, r.kind AS run_kind, o.task, o.region, o.seed {base} WHERE {cond} "
                       f"ORDER BY e.created_at DESC, e.event_id DESC LIMIT ?", [*params, limit])
        events = []
        for r in rows:
            item = self._event(r)
            item.update(object_id=r["object_id"], task=r["task"], region=r["region"], seed=r["seed"])
            events.append(item)
        return {"total": total, "by_order": by_order, "returned": len(events), "events": events}

    def _list_proposals(self, a: dict[str, Any]) -> dict[str, Any]:
        since = self._since(a["since_hours"])
        limit = self._limit(a["limit"])
        where, params = ["created_at >= ?"], [since]
        if a["source"] != "any":
            where.append("source = ?")
            params.append(a["source"])
        region = self._region_filter(a["region"])
        if region:
            where.append("region = ?")
            params.append(region)
        rows = self._q(f"SELECT * FROM proposals WHERE {' AND '.join(where)} ORDER BY proposal_id DESC LIMIT ?",
                       [*params, limit])
        outcomes: dict[str, dict[str, dict[str, Any]]] = {}
        out = []
        for r in rows:
            point = json.loads(r["point_json"] or "{}")
            if r["region"] not in outcomes:
                outcomes[r["region"]] = self._point_outcomes(r["region"])
            outcome = outcomes[r["region"]].get(point_key(point), {"n_objects": 0})
            out.append({"proposal_id": r["proposal_id"], "source": r["source"], "region": r["region"],
                        "point": point, "seeds": r["seeds"], "status": r["status"], "n_enqueued": r["n_enqueued"],
                        "created": iso_utc(r["created_at"]),
                        "rationale": _clip(r["rationale"], self.limits.max_note_chars), "outcome": outcome})
        return {"window_since": iso_utc(since), "returned": len(out), "proposals": out}

    def _read_survey_log(self, a: dict[str, Any]) -> dict[str, Any]:
        since = self._since(a["since_hours"])
        limit = self._limit(a["limit"])
        max_chars = a["max_chars_per_entry"] or DEFAULT_LOG_ENTRY_CHARS
        if not 1 <= max_chars <= 20_000:
            raise ToolError(f"max_chars_per_entry must be 0 (default) or 1-20000, got {max_chars}")
        where, params = ["created_at >= ?"], [since]
        if a["kind"] != "any":
            where.append("kind = ?")
            params.append(a["kind"])
        rows = self._q(f"SELECT entry_id, created_at, author, kind, title, markdown FROM survey_log "
                       f"WHERE {' AND '.join(where)} ORDER BY entry_id DESC LIMIT ?", [*params, limit])
        entries = [{"entry_id": r["entry_id"], "created": iso_utc(r["created_at"]), "author": r["author"],
                    "kind": r["kind"], "title": r["title"], "chars": len(r["markdown"] or ""),
                    "markdown": _clip(r["markdown"], max_chars)} for r in rows]
        return {"window_since": iso_utc(since), "returned": len(entries), "entries": entries}

    # ------------------------------------------------------------------ analysis tools
    def _query_summaries(self, a: dict[str, Any]) -> dict[str, Any]:
        region_name = self._region_filter(a["region"])
        reg = self.sky.region(region_name) if region_name else None
        metrics = _unique_list(a["metrics"], "metrics")
        if not 1 <= len(metrics) <= MAX_METRICS:
            raise ToolError(f"give 1 to {MAX_METRICS} metrics, got {len(metrics)}")
        group_by = _unique_list(a["group_by"], "group_by")
        if len(group_by) > MAX_GROUP_BY:
            raise ToolError(f"group_by takes at most {MAX_GROUP_BY} fields, got {len(group_by)}")
        knob_names = set(reg.knob_names) if reg is not None else set()
        for g in group_by:
            if g in GROUP_FIELDS:
                continue
            if reg is None:
                raise ToolError(f"group field {g!r} is not one of {list(GROUP_FIELDS)}; knob names and 'point' need "
                                f"a region")
            if g != "point" and g not in knob_names:
                raise ToolError(f"group field {g!r} is neither an object field {list(GROUP_FIELDS)}, 'point', nor a "
                                f"knob of region {reg.name!r} {reg.knob_names}")
        if a["where"] and reg is None:
            raise ToolError("where-filters need a region")
        where = self._where(reg, a["where"]) if reg is not None else []
        label = self._label_filter(a["label"])
        needs_point = bool(where) or any(g == "point" or g in knob_names for g in group_by)
        groups: dict[str, dict[str, Any]] = {}
        n_off_grid = 0
        for r in self._done_rows(region_name, self._finished_since(a["finished_within_hours"])):
            if label is not None and r["label"] != label:
                continue
            point = None
            if needs_point:
                point = self._point_of(r["region"], r["spec_json"])
                if point is None:
                    n_off_grid += 1
                    continue
                if not _matches(point, where):
                    continue
            key = {g: _group_value(g, r, point) for g in group_by}
            gkey = _dumps(jsonable(key))
            group = groups.setdefault(gkey, {"key": key, "n": 0, "labels": Counter(),
                                             "values": {m: [] for m in metrics}})
            group["n"] += 1
            group["labels"][r["label"] or "unlabelled"] += 1
            summary = json.loads(r["summary_json"] or "{}")
            for m in metrics:
                group["values"][m].append(_metric_value(m, summary, r))
        ordered = sorted(groups.values(), key=lambda gr: _order_key([gr["key"][g] for g in group_by]))
        out_groups = [{"group": gr["key"], "n": gr["n"], "labels": dict(sorted(gr["labels"].items())),
                       "metrics": {m: _describe(gr["values"][m]) for m in metrics}}
                      for gr in ordered[:MAX_GROUPS]]
        out: dict[str, Any] = {"region": region_name or "all", "group_by": group_by,
                               "n_objects": sum(gr["n"] for gr in groups.values()), "n_groups": len(groups),
                               "groups": out_groups}
        if len(groups) > MAX_GROUPS:
            out["note"] = f"only the first {MAX_GROUPS} of {len(groups)} groups are shown"
        if n_off_grid:
            out["excluded_not_on_current_sky"] = n_off_grid
        return out

    def _fit_power_law(self, a: dict[str, Any]) -> dict[str, Any]:
        reg = self._region(a["region"])
        try:
            knob = reg.knob(a["knob"])
        except KeyError:
            raise ToolError(f"region {reg.name!r} has no knob {a['knob']!r}; knobs: {reg.knob_names}") from None
        metric = a["metric"]
        where = self._where(reg, a["where"])
        if any(name == knob.name for name, _ in where):
            raise ToolError(f"the fitted knob {knob.name!r} cannot also be fixed by a where-filter")
        label = self._label_filter(a["label"])
        excluded: Counter[str] = Counter()
        obs: list[tuple[str, dict[str, Any], float, float]] = []  # (point key, point, x, y)
        for r in self._done_rows(reg.name, self._finished_since(a["finished_within_hours"])):
            if label is not None and r["label"] != label:
                continue
            point = self._point_of(reg.name, r["spec_json"])
            if point is None:
                excluded["not a point of the current sky"] += 1
                continue
            if not _matches(point, where):
                continue
            x = point[knob.name]
            if not _is_json_type(x, "number") or x <= 0:
                excluded["knob value not a positive number"] += 1
                continue
            y = _metric_value(metric, json.loads(r["summary_json"] or "{}"), r)
            if not math.isfinite(y):
                excluded["metric missing or not finite"] += 1
                continue
            if y <= 0:
                excluded["metric <= 0"] += 1
                continue
            obs.append((point_key(point), point, float(x), y))
        objects_per_x = Counter(x for _, _, x, _ in obs)
        seeds_per_point = Counter(key for key, _, _, _ in obs)
        if a["aggregate"] == "point_median":
            by_point: dict[str, list[tuple[dict[str, Any], float, float]]] = {}
            for key, point, x, y in obs:
                by_point.setdefault(key, []).append((point, x, y))
            data = [(items[0][0], items[0][1], float(10.0 ** np.median(np.log10([y for _, _, y in items]))))
                    for _, items in sorted(by_point.items())]
        else:
            data = [(point, x, y) for _, point, x, y in obs]
        varying = sorted(k for k in reg.knob_names if k != knob.name
                         and len({point_key({k: p[k]}) for p, _, _ in data}) > 1)
        out: dict[str, Any] = {
            "region": reg.name, "metric": metric, "knob": knob.name, "aggregate": a["aggregate"],
            "where": {name: value for name, value in where}, "label": label, "n_objects": len(obs),
            "n_points": len(seeds_per_point), "n_observations": len(data),
            "objects_per_knob_value": {_fmt_num(x): n for x, n in sorted(objects_per_x.items())},
            "min_seeds_per_point": min(seeds_per_point.values(), default=0),
            "other_knobs_varying": varying, "excluded": dict(sorted(excluded.items())),
        }
        try:
            fit = power_law_fit([x for _, x, _ in data], [y for _, _, y in data])
        except ValueError as exc:
            out.update(fit=None, reason=str(exc), caveats=["insufficient data for a fit"])
            return out
        out["fit"] = fit
        out["formula"] = f"{metric} ≈ {_fmt_num(fit['prefactor'])} · {knob.name}^{_fmt_num(fit['slope'])}"
        out["caveats"] = _fit_caveats(fit, sorted(objects_per_x), min(seeds_per_point.values()), varying, excluded)
        return out

    def _get_sky(self, a: dict[str, Any]) -> dict[str, Any]:
        name = a["region"].strip()
        counts: dict[str, dict[str, int]] = {}
        for r in self._q("SELECT region, status, COUNT(*) AS n FROM objects WHERE sky_version = ? "
                         "GROUP BY region, status", (self.sky.version,)):
            counts.setdefault(r["region"], {})[r["status"]] = r["n"]
        if not name:
            return {"version": self.sky.version, "description": self.sky.description,
                    "regions": [self._region_brief(reg, counts.get(reg.name, {})) for reg in self.sky.regions]}
        reg = self._region(name)
        out = self._region_brief(reg, counts.get(reg.name, {}))
        out.update(description=reg.description, fixed=reg.fixed, train=reg.train, ri=reg.ri)
        seeds_by_point: Counter[str] = Counter()
        labels: Counter[str] = Counter()
        for r in self._done_rows(reg.name, None):
            labels[r["label"] or "unlabelled"] += 1
            point = self._point_of(reg.name, r["spec_json"])
            if point is not None:
                seeds_by_point[point_key(point)] += 1
        out["observed_points"] = len(seeds_by_point)
        out["points_with_3_or_more_done_seeds"] = sum(1 for n in seeds_by_point.values() if n >= 3)
        out["labels_of_done"] = dict(sorted(labels.items()))
        pairs = Scheduler(self.catalog, self.sky).boundary_pairs(reg)
        out["n_boundary_pairs"] = len(pairs)
        out["boundary_pairs"] = [{"a": pa, "b": pb, "reason": why} for pa, pb, why in pairs[:MAX_BOUNDARY_PAIRS]]
        return out

    # ------------------------------------------------------------------ write tools
    def _propose_configs(self, a: dict[str, Any]) -> dict[str, Any]:
        items = a["proposals"]
        if not items:
            raise ToolError("proposals is empty")
        if len(items) > MAX_PROPOSALS_PER_CALL:
            raise ToolError(f"at most {MAX_PROPOSALS_PER_CALL} proposals per call, got {len(items)}")
        used_today = self._agent_objects_last_day()
        results = []
        for i, item in enumerate(items):
            result: dict[str, Any] = {"index": i, "region": item["region"]}
            try:
                n_new = self._propose_one(item, result, used_today)
            except ToolError as exc:
                result.update(status="rejected", errors=[exc.message, *exc.details.get("errors", [])])
            else:
                used_today += n_new
            results.append(result)
        accepted = sum(r["status"] == "enqueued" for r in results)
        payload = {"accepted": accepted,
                   "rejected": sum(r["status"] == "rejected" for r in results),
                   "results": results, "caps": self.caps()}
        if accepted == 0 and payload["rejected"]:
            raise ToolError("no proposal was accepted", payload)
        return payload

    def _propose_one(self, item: dict[str, Any], result: dict[str, Any], used_today: int) -> int:
        """Validate and enqueue one proposal (fills ``result``); returns the number of objects enqueued."""
        point = _pairs_to_point(item["point"])
        errors = self.sky.validate_point(item["region"], point)
        if errors:
            raise ToolError("invalid point", {"errors": errors})
        reg = self.sky.region(item["region"])
        seeds = item["seeds"] or reg.seeds
        if not 1 <= seeds <= self.limits.max_seeds_per_point:
            raise ToolError(f"seeds must be 0 (region default {reg.seeds}) or 1-{self.limits.max_seeds_per_point}, "
                            f"got {seeds}")
        hypothesis = self._text(item["hypothesis"], "hypothesis")
        falsifier = self._text(item["falsified_if"], "falsified_if")
        point = {k.name: k.normalize(point[k.name]) for k in reg.knobs}
        result["point"] = point
        if self.n_proposals >= self.limits.max_proposals_per_run:
            raise ToolError(f"per-run proposal cap reached ({self.limits.max_proposals_per_run})")
        specs = [self.sky.make_spec(reg, point, s) for s in range(seeds)]
        ids = [s.object_id() for s in specs]
        existing = {r["object_id"] for r in self._q(
            f"SELECT object_id FROM objects WHERE object_id IN ({', '.join('?' * len(ids))})", ids)}
        new = [s for s, oid in zip(specs, ids) if oid not in existing]
        if not new:
            result.update(status="already_in_catalog", n_existing=len(existing),
                          note=f"all {seeds} seeds of this point are already in the catalog; nothing enqueued")
            return 0
        left = self.limits.max_objects_per_day - used_today
        if len(new) > left:
            raise ToolError(f"daily object cap: this proposal needs {len(new)} new objects but only {max(left, 0)} "
                            f"of {self.limits.max_objects_per_day} are left in the last 24 h")
        n = self.catalog.add_objects(new, region=reg.name, source=AGENT_SOURCE, priority=reg.priority)
        rationale = f"[{item['purpose']}] hypothesis: {hypothesis} | falsified if: {falsifier}"
        pid = self.catalog.add_proposal(AGENT_SOURCE, reg.name, point, task=reg.task, seeds=seeds,
                                        rationale=rationale, status="enqueued", n_enqueued=n,
                                        decided_at=self.now())
        self.n_proposals += 1
        logger.info("agent proposal %d: %d objects at %s in region %s (%s)", pid, n, point_key(point), reg.name,
                    item["purpose"])
        result.update(status="enqueued", proposal_id=pid, n_enqueued=n, n_existing=len(existing),
                      object_ids=[s.object_id() for s in new])
        return n

    def _label_object(self, a: dict[str, Any]) -> dict[str, Any]:
        oid = a["object_id"].strip()
        label = a["label"]
        if label not in LABELS:
            raise ToolError(f"label must be one of {LABELS}, got {label!r}")
        note = self._text(a["note"], "note")
        confidence = a["confidence"]
        if not 0.0 <= confidence <= 1.0:
            raise ToolError(f"confidence must be in [0, 1], got {confidence}")
        tags = _unique_list(a["tags"], "tags")
        bad = [t for t in tags if not _TAG.match(t)]
        if bad or len(tags) > MAX_TAGS:
            raise ToolError(f"tags must be at most {MAX_TAGS} lowercase identifiers ([a-z0-9_.-], <= 40 chars); "
                            f"invalid: {bad}")
        status = self._object_status(oid)
        if status != "done":
            raise ToolError(f"object {oid} is {status}; only done objects can be labelled")
        if self.n_labels >= self.limits.max_labels_per_run:
            raise ToolError(f"per-run label cap reached ({self.limits.max_labels_per_run})")
        previous = self.catalog.latest_classification(oid)
        cid = self.catalog.add_classification(oid, {"label": label, "tags": tags, "score": confidence},
                                              classifier=AGENT_SOURCE, note=note)
        self.n_labels += 1
        logger.info("agent labelled %s as %s (previously %s)", oid, label,
                    None if previous is None else previous["label"])
        return {"classification_id": cid, "object_id": oid, "label": label,
                "previous": None if previous is None else {"label": previous["label"],
                                                           "classifier": previous["classifier"]},
                "labels_left_this_run": self.limits.max_labels_per_run - self.n_labels}

    def _request_followup(self, a: dict[str, Any]) -> dict[str, Any]:
        oid = a["object_id"].strip()
        reason = self._text(a["reason"], "reason")
        rows = self._q("SELECT o.status, o.region, f.status AS followup_status, f.requested_by, f.reason "
                       "FROM objects o LEFT JOIN followups f ON f.object_id = o.object_id WHERE o.object_id = ?",
                       (oid,))
        if not rows:
            raise ToolError(f"unknown object id {oid!r}")
        row = rows[0]
        if row["status"] != "done":
            raise ToolError(f"object {oid} is {row['status']}; only done objects can be followed up")
        policy = next((r.ri_policy for r in self.sky.regions if r.name == row["region"]), None)
        if policy == "never":
            raise ToolError(f"region {row['region']!r} has ri_policy 'never': no RI spectroscopy there")
        if row["followup_status"] is not None:
            return {"requested": False, "object_id": oid,
                    "note": "this object already has a follow-up (an object is followed up at most once)",
                    "existing": {"status": row["followup_status"], "requested_by": row["requested_by"],
                                 "reason": row["reason"]}}
        if self.n_followups >= self.limits.max_followups_per_run:
            raise ToolError(f"per-run follow-up cap reached ({self.limits.max_followups_per_run})")
        requested = self.catalog.request_followup(oid, reason, priority=AGENT_FOLLOWUP_PRIORITY,
                                                  requested_by=AGENT_SOURCE)
        if requested:
            self.n_followups += 1
            logger.info("agent requested RI follow-up of %s: %s", oid, reason)
        return {"requested": requested, "object_id": oid, "priority": AGENT_FOLLOWUP_PRIORITY,
                "followups_left_this_run": self.limits.max_followups_per_run - self.n_followups}

    def _write_survey_log(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.log_entry_id is not None:
            raise ToolError(f"the survey log of this run is already written (entry {self.log_entry_id}); "
                            f"write_survey_log may be called once per run")
        title, markdown = self._title(a["title"]), self._markdown(a["markdown"])
        self.log_entry_id = self.catalog.add_log(AGENT_SOURCE, self.kind, title, markdown)
        logger.info("agent wrote survey-log entry %d: %s", self.log_entry_id, title)
        return {"written": True, "entry_id": self.log_entry_id}

    def _write_report(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.report_path is not None:
            raise ToolError(f"the report of this run is already written ({self._rel(self.report_path)}); "
                            f"write_report may be called once per run")
        title, markdown = self._title(a["title"]), self._markdown(a["markdown"])
        path = unique_path(self.paths.reports / "weekly", iso_week(self.now()))
        write_text_atomic(path, f"# {title}\n\n{markdown}\n")
        self.report_path = path
        self.report_entry_id = self.catalog.add_log(AGENT_SOURCE, self.kind, title, markdown)
        logger.info("agent wrote report %s (survey-log entry %d)", path, self.report_entry_id)
        return {"written": True, "path": self._rel(path), "entry_id": self.report_entry_id}

    # ------------------------------------------------------------------ helpers: input checks
    def _q(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return self.catalog.query(sql, tuple(params), timeout_s=QUERY_TIMEOUT_S)

    def _since(self, hours: float) -> float:
        if hours < 0 or hours > MAX_LOOKBACK_HOURS:
            raise ToolError(f"since_hours must be 0 (window start) or in (0, {MAX_LOOKBACK_HOURS:g}], got {hours}")
        return self.window_start if hours == 0 else self.now() - 3600.0 * hours

    def _finished_since(self, hours: float) -> float | None:
        if hours < 0 or hours > MAX_LOOKBACK_HOURS:
            raise ToolError(f"finished_within_hours must be 0 (no restriction) or in (0, {MAX_LOOKBACK_HOURS:g}], "
                            f"got {hours}")
        return None if hours == 0 else self.now() - 3600.0 * hours

    @staticmethod
    def _limit(value: int) -> int:
        if value == 0:
            return DEFAULT_LIST_LIMIT
        if not 1 <= value <= MAX_LIST_LIMIT:
            raise ToolError(f"limit must be 0 (default) or 1-{MAX_LIST_LIMIT}, got {value}")
        return value

    @staticmethod
    def _offset(value: int) -> int:
        if value < 0:
            raise ToolError(f"offset must be >= 0, got {value}")
        return value

    def _region(self, name: str) -> Region:
        try:
            return self.sky.region(name.strip())
        except KeyError:
            raise ToolError(f"unknown region {name!r}; regions of sky {self.sky.version}: "
                            f"{[r.name for r in self.sky.regions]}") from None

    def _region_filter(self, name: str) -> str | None:
        """A validated region name, or None for "" (all regions)."""
        name = name.strip()
        return self._region(name).name if name else None

    @staticmethod
    def _label_filter(label: str) -> str | None:
        label = label.strip()
        if not label:
            return None
        if label not in LABELS:
            raise ToolError(f"label must be one of {LABELS} or \"\", got {label!r}")
        return label

    def _where(self, reg: Region, pairs: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        """Validated where-filters as (knob name, value) pairs. A value the knob cannot take is an error rather than a
        filter that silently matches nothing."""
        if len(pairs) > MAX_WHERE_FILTERS:
            raise ToolError(f"at most {MAX_WHERE_FILTERS} where-filters, got {len(pairs)}")
        out: list[tuple[str, Any]] = []
        for pair in pairs:
            name = pair["knob"]
            if name not in reg.knob_names:
                raise ToolError(f"where: region {reg.name!r} has no knob {name!r}; knobs: {reg.knob_names}")
            if any(n == name for n, _ in out):
                raise ToolError(f"where: knob {name!r} is given twice")
            problem = reg.knob(name).check(pair["value"])
            if problem:
                raise ToolError(f"where: {problem}")
            out.append((name, pair["value"]))
        return out

    def _text(self, text: str, what: str) -> str:
        text = text.strip()
        if not text:
            raise ToolError(f"{what} must not be empty")
        if len(text) > self.limits.max_note_chars:
            raise ToolError(f"{what} is {len(text)} characters; the limit is {self.limits.max_note_chars}")
        return text

    @staticmethod
    def _title(title: str) -> str:
        title = " ".join(title.split())
        if not title:
            raise ToolError("title must not be empty")
        if len(title) > MAX_TITLE_CHARS:
            raise ToolError(f"title is {len(title)} characters; the limit is {MAX_TITLE_CHARS}")
        return title

    def _markdown(self, markdown: str) -> str:
        markdown = markdown.strip()
        if not markdown:
            raise ToolError("markdown must not be empty")
        if len(markdown) > self.limits.max_log_chars:
            raise ToolError(f"markdown is {len(markdown)} characters; the limit is {self.limits.max_log_chars}")
        return markdown

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.paths.root).as_posix()
        except ValueError:
            return path.as_posix()

    # ------------------------------------------------------------------ helpers: catalog reads
    def _object_status(self, oid: str) -> str:
        rows = self._q("SELECT status FROM objects WHERE object_id = ?", (oid,))
        if not rows:
            raise ToolError(f"unknown object id {oid!r}")
        return rows[0]["status"]

    def _agent_objects_last_day(self) -> int:
        return self._q("SELECT COUNT(*) AS n FROM objects WHERE source = ? AND created_at >= ?",
                       (AGENT_SOURCE, self.now() - _DAY_S))[0]["n"]

    def _done_rows(self, region: str | None, since: float | None) -> list[dict[str, Any]]:
        """Done objects with a summary (optionally of one region / finished since a time) and their latest label."""
        where, params = ["o.status = 'done'"], []
        if region is not None:
            where.append("o.region = ?")
            params.append(region)
        if since is not None:
            where.append("o.finished_at >= ?")
            params.append(since)
        return self._q(
            f"SELECT o.object_id, o.region, o.task, o.seed, o.sky_version, o.source, o.random_control, o.spec_json, "
            f"o.finished_at, o.wall_time_s, s.json AS summary_json, c.label AS label FROM objects o "
            f"JOIN summaries s ON s.object_id = o.object_id LEFT JOIN {_LATEST_LABEL_JOIN} "
            f"WHERE {' AND '.join(where)} ORDER BY o.rowid", params)

    def _point_of(self, region: str, spec_json: str) -> dict[str, Any] | None:
        """The knob point of an object of ``region`` in the current sky; None if the object belongs to another sky
        version (its fixed and training settings may differ, so its knob values would not mean the same) or its region
        or point is not part of the current sky."""
        try:
            spec = ObjectSpec.from_json(spec_json)
            if spec.sky_version != self.sky.version:
                return None
            reg = self.sky.region(region)
            point = self.sky.point_of(reg, spec)
        except (KeyError, ValueError, TypeError):
            return None
        return None if self.sky.validate_point(reg, point) else point

    def _point_outcomes(self, region: str) -> dict[str, dict[str, Any]]:
        """Per point key of ``region`` (current sky version): objects by status, labels of done objects, median
        t_test_acc90 and final_test_acc."""
        acc: dict[str, dict[str, Any]] = {}
        for r in self._q(
                f"SELECT o.status, o.spec_json, s.t_test_acc90, s.final_test_acc, c.label FROM objects o "
                f"LEFT JOIN summaries s ON s.object_id = o.object_id LEFT JOIN {_LATEST_LABEL_JOIN} "
                f"WHERE o.region = ? AND o.sky_version = ?", (region, self.sky.version)):
            point = self._point_of(region, r["spec_json"])
            if point is None:
                continue
            item = acc.setdefault(point_key(point), {"by_status": Counter(), "labels": Counter(), "t90": [],
                                                     "acc": []})
            item["by_status"][r["status"]] += 1
            if r["status"] == "done":
                item["labels"][r["label"] or "unlabelled"] += 1
                item["t90"].append(math.nan if r["t_test_acc90"] is None else r["t_test_acc90"])
                item["acc"].append(math.nan if r["final_test_acc"] is None else r["final_test_acc"])
        out = {}
        for key, item in acc.items():
            t90 = np.asarray(item["t90"], dtype=np.float64)
            out[key] = {"n_objects": sum(item["by_status"].values()), "by_status": dict(item["by_status"]),
                        "labels": dict(sorted(item["labels"].items())),
                        "median_t_test_acc90": _nanmedian(t90),
                        "n_reached_test_acc90": int(np.isfinite(t90).sum()),
                        "median_final_test_acc": _nanmedian(np.asarray(item["acc"], dtype=np.float64))}
        return out

    def _brief(self, r: Mapping[str, Any]) -> dict[str, Any]:
        """Compact description of an object row (list tools)."""
        item: dict[str, Any] = {
            "object_id": r["object_id"], "region": r["region"], "task": r["task"], "seed": r["seed"],
            "status": r["status"], "finished": iso_utc(r["finished_at"]), "wall_time_s": r["wall_time_s"],
            "point": self._point_of(r["region"], r["spec_json"]),
            "label": r["label"], "score": r["score"], "classifier": r["classifier"],
            "tags": json.loads(r["tags_json"] or "[]"),
        }
        if r["source"] != "coarse":
            item["source"] = r["source"]
        if r["random_control"]:
            item["random_control"] = True
        if r["status"] == "done":
            summary = {c: r[c] for c in SUMMARY_COLUMNS}
            summary["diverged"] = None if summary["diverged"] is None else bool(summary["diverged"])
            item["summary"] = summary
        if r["error"] and r["error"].strip():
            item["error"] = _clip(_first_line(r["error"]), ERROR_PREVIEW_CHARS)
        return item

    @staticmethod
    def _event(r: Mapping[str, Any]) -> dict[str, Any]:
        return {"event_id": r["event_id"], "run_kind": r["run_kind"], "subset": json.loads(r["subset_json"] or "[]"),
                "order": r["order_k"], "kind": r["kind"], "step": r["step"], "step_lo": r["step_lo"],
                "step_hi": r["step_hi"], "ri_before": r["ri_before"], "ri_after": r["ri_after"],
                "peak_ri": r["peak_ri"], "confidence": r["confidence"],
                "has_precursor": None if r["has_precursor"] is None else bool(r["has_precursor"]),
                "matches_gt": None if r["matches_gt"] is None else bool(r["matches_gt"])}

    def _region_brief(self, reg: Region, counts: Mapping[str, int]) -> dict[str, Any]:
        return {"name": reg.name, "task": reg.task, "seeds": reg.seeds, "priority": reg.priority,
                "ri_policy": reg.ri_policy, "grid_size": reg.grid_size,
                "knobs": {k.name: k.to_dict() for k in reg.knobs},
                "objects": {**dict(sorted(counts.items())), "total": sum(counts.values())}}

    def _failure_groups(self, since: float) -> list[dict[str, Any]]:
        """Failed objects since ``since`` grouped by the first line of their error message (largest groups first)."""
        groups: dict[str, dict[str, Any]] = {}
        for r in self._q("SELECT object_id, region, error FROM objects WHERE status = 'failed' AND finished_at >= ? "
                         "ORDER BY finished_at", (since,)):
            first = _clip(_first_line(r["error"] or "") or "(no error message)", 200)
            g = groups.setdefault(first, {"error": first, "count": 0, "regions": set(), "examples": []})
            g["count"] += 1
            g["regions"].add(r["region"])
            if len(g["examples"]) < 3:
                g["examples"].append(r["object_id"])
        ordered = sorted(groups.values(), key=lambda g: (-g["count"], g["error"]))
        return [{**g, "regions": sorted(g["regions"])} for g in ordered[:10]]

    def _recent_agent_runs(self, now: float) -> dict[str, Any]:
        rows = self._q("SELECT run_id, kind, status, started_at, est_cost_usd, n_tool_calls, stop_reason "
                       "FROM agent_runs ORDER BY run_id DESC LIMIT 5")
        cost = self._q("SELECT COALESCE(SUM(est_cost_usd), 0.0) AS c FROM agent_runs WHERE started_at >= ?",
                       (now - 7 * _DAY_S,))[0]["c"]
        return {"recent": [{**r, "started_at": iso_utc(r["started_at"])} for r in rows],
                "est_cost_last_7d_usd": cost}

    def _control_recovery(self, since: float) -> dict[str, dict[str, int]]:
        """Per always-observed region (positive controls): done objects since ``since`` with an RI run, and how many
        of them have an activation event that matches the planted ground truth."""
        regions = [r.name for r in self.sky.regions if r.ri_policy == "always"]
        if not regions:
            return {}
        rows = self._q(
            "SELECT o.region AS region, COUNT(DISTINCT o.object_id) AS with_ri, "
            "COUNT(DISTINCT CASE WHEN e.matches_gt = 1 THEN o.object_id END) AS with_gt_match "
            "FROM objects o JOIN ri_runs r ON r.object_id = o.object_id "
            "LEFT JOIN ri_events e ON e.ri_run_id = r.ri_run_id AND e.kind = 'activation' "
            f"WHERE o.status = 'done' AND o.finished_at >= ? AND o.region IN ({', '.join('?' * len(regions))}) "
            "GROUP BY o.region ORDER BY o.region", (since, *regions))
        return {r["region"]: {"objects_with_ri": r["with_ri"], "with_ground_truth_match": r["with_gt_match"]}
                for r in rows}

    def _disk(self, min_free_gb: float) -> dict[str, Any]:
        try:
            usage = shutil.disk_usage(self.paths.root)
        except OSError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        free_gb = usage.free / 1e9
        return {"free_gb": free_gb, "total_gb": usage.total / 1e9, "min_free_gb": min_free_gb,
                "low": free_gb < min_free_gb}

    def _latest_backup_age_h(self) -> float | None:
        """Age in hours of the newest file in the backups directory (wall clock), None if there is none."""
        try:
            mtimes = [e.stat().st_mtime for e in os.scandir(self.paths.backups) if e.is_file()]
        except OSError:
            return None
        return max(0.0, (time.time() - max(mtimes)) / 3600.0) if mtimes else None

    def _qc_flags(self, ov: Mapping[str, Any], now: float) -> list[str]:
        flags = []
        tp = ov["throughput"]
        if tp["failed_last_24h"]:
            flags.append(f"{tp['failed_last_24h']} objects failed in the last 24 h")
        if ov["divergences_last_24h"]:
            per_region = ", ".join(f"{n} in {region}" for region, n in ov["divergences_last_24h"].items())
            flags.append(f"divergences in the last 24 h: {per_region}")
        if tp["done_prev_24h"] >= THROUGHPUT_MIN_BASELINE \
                and tp["done_last_24h"] < THROUGHPUT_DROP_FRAC * tp["done_prev_24h"]:
            flags.append(f"throughput dropped: {tp['done_last_24h']} done in the last 24 h vs "
                         f"{tp['done_prev_24h']} in the previous 24 h")
        workers = ov["workers"]
        if workers["stale_holding_object"]:
            flags.append(f"stale workers holding objects: {len(workers['stale_holding_object'])}")
        running = ov["objects"]["running"]
        if workers["alive"] == 0 and (ov["queue"] > 0 or running > 0):
            flags.append("no worker heartbeat while objects are queued or running: is the survey daemon down?")
        if ov["disk"].get("low"):
            flags.append(f"low disk space: {ov['disk']['free_gb']:.1f} GB free (minimum "
                         f"{ov['disk']['min_free_gb']:.0f} GB)")
        age = ov["latest_backup_age_h"]
        created = self.catalog.get_meta("created_at")
        catalog_age_h = (now - float(created)) / 3600.0 if created else 0.0
        if (age is None and catalog_age_h > BACKUP_MAX_AGE_H) or (age is not None and age > BACKUP_MAX_AGE_H):
            flags.append("no catalog backup in the last 36 h" if age is None
                         else f"latest catalog backup is {age:.0f} h old")
        if not ov["calibration"]["present"]:
            flags.append("no calibration file: RI detections are provisional")
        if ov["transients"]["unacknowledged"]:
            flags.append(f"{ov['transients']['unacknowledged']} transients not yet acknowledged by the human")
        return flags


# ---------------------------------------------------------------------------------------------- module helpers
def _first_line(text: str) -> str:
    """The first non-empty line of ``text`` (stripped), or ""."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _clip(text: str | None, limit: int) -> str | None:
    """``text`` cut to ``limit`` characters (with a marker); None stays None."""
    if text is None or len(text) <= limit:
        return text
    return text[: max(0, limit - 20)] + " …[truncated]"


def _unique_list(values: list[str], what: str) -> list[str]:
    values = [v.strip() for v in values]
    if len(set(values)) != len(values):
        raise ToolError(f"{what} contains duplicates: {values}")
    return values


def _pairs_to_point(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    point: dict[str, Any] = {}
    for pair in pairs:
        if pair["knob"] in point:
            raise ToolError("invalid point", {"errors": [f"knob {pair['knob']!r} is given twice"]})
        point[pair["knob"]] = pair["value"]
    return point


def _matches(point: Mapping[str, Any], where: Sequence[tuple[str, Any]]) -> bool:
    """Whether ``point`` has every (knob, value) of ``where`` (canonical comparison: 128 == 128.0)."""
    return all(point_key({"v": point[name]}) == point_key({"v": value}) for name, value in where)


def _group_value(field: str, row: Mapping[str, Any], point: Mapping[str, Any] | None) -> Any:
    if field == "label":
        return row["label"] or "unlabelled"
    if field == "random_control":
        return bool(row["random_control"])
    if field in GROUP_FIELDS:
        return row[field]
    if field == "point":
        return dict(point)
    return point[field]


def _metric_value(metric: str, summary: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    """A metric of one object as a float (NaN when missing or not numeric; booleans as 0/1)."""
    value = row["wall_time_s"] if metric == "wall_time_s" else summary.get(metric)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan


def _describe(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size >= 2 else None, "min": float(a.min()), "max": float(a.max())}


def _order_key(values: Sequence[Any]) -> tuple:
    """Sort key of a sequence of group values: numbers in numeric order, then strings, booleans and None; a point
    (a dict of knob values) or a list sorts by its values in order."""
    return tuple(_rank(v) for v in values)


def _rank(v: Any) -> tuple:
    if isinstance(v, bool):
        return (2, float(v), "")
    if isinstance(v, (int, float)):
        return (0, float(v), "")
    if isinstance(v, str):
        return (1, 0.0, v)
    if v is None:
        return (3, 0.0, "")
    items = v.values() if isinstance(v, Mapping) else v
    return (4, 0.0, "", _order_key(list(items)))


def _nanmedian(a: np.ndarray) -> float | None:
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def _fmt_num(x: float) -> str:
    return f"{x:.4g}"


def _fit_caveats(fit: Mapping[str, Any], xs: Sequence[float], min_seeds: int, varying: Sequence[str],
                 excluded: Counter) -> list[str]:
    """Warnings about a power-law fit over the distinct knob values ``xs`` whose least-replicated configuration has
    ``min_seeds`` objects (thresholds: the MIN_* constants and CURVATURE_ALPHA)."""
    caveats = []
    if fit["n_distinct_x"] < MIN_DISTINCT_X_FOR_LAW:
        caveats.append(f"only {fit['n_distinct_x']} distinct knob values: a power law cannot be distinguished from "
                       f"other monotone relations")
    decades = math.log10(max(xs) / min(xs))
    if decades < MIN_DECADES:
        caveats.append(f"the knob spans {decades:.2f} decades (< {MIN_DECADES:g}): the exponent is poorly constrained")
    if min_seeds < MIN_SEEDS_PER_POINT:
        caveats.append(f"some configurations have fewer than {MIN_SEEDS_PER_POINT} seeds: replication insufficient")
    if varying:
        caveats.append(f"other knobs vary within the fit ({', '.join(varying)}): the fit pools over them; fix them "
                       f"with where-filters to rule out confounding")
    if fit["r2"] is not None and fit["r2"] < MIN_R2:
        caveats.append(f"R^2 = {fit['r2']:.3f} < {MIN_R2:g}: large scatter or a different functional form")
    curvature = fit.get("curvature")
    if curvature and curvature["p_value"] is not None and curvature["p_value"] < CURVATURE_ALPHA:
        caveats.append(f"significant curvature in log-log (quadratic term p = {curvature['p_value']:.3g}): not a "
                       f"single power law over this range (a regime change?)")
    jack = fit.get("jackknife")
    if jack and not (fit["slope_ci"][0] <= jack["min"] and jack["max"] <= fit["slope_ci"][1]):
        caveats.append("leaving out one knob value moves the exponent outside its confidence interval: the fit "
                       "depends on single values")
    n_excluded = sum(excluded.values())
    if n_excluded:
        caveats.append(f"{n_excluded} objects excluded (see 'excluded'); objects that never reached a threshold are "
                       f"censored, which biases the exponent")
    caveats.append("a candidate only: confirm on held-out points (e.g. midpoints), in other regions and with replays "
                   "before calling it a law")
    return caveats


def _lightcurve_excerpt(path: Path) -> dict[str, Any]:
    """Roughly log-spaced rows of the main light-curve keys of an ``np.savez_compressed`` light curve."""
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: np.asarray(data[k]) for k in data.files}
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": f"cannot read the light curve: {type(exc).__name__}: {exc}"}
    step = arrays.get("step")
    if step is None or step.ndim != 1 or step.size == 0 or not np.issubdtype(step.dtype, np.number):
        return {"available": False, "reason": "the light curve has no usable 'step' array"}
    idx = _log_spaced_indices(step.astype(np.float64), LIGHTCURVE_EXCERPT_POINTS)
    excerpt = {}
    for key in LIGHTCURVE_EXCERPT_KEYS:
        arr = arrays.get(key)
        if arr is None or arr.shape != step.shape or not np.issubdtype(arr.dtype, np.number):
            continue
        values = arr.astype(np.float64)[idx]
        if np.isfinite(values).any():
            excerpt[key] = values
    return {"available": True, "n_points": int(step.size), "excerpt": excerpt}


def _log_spaced_indices(step: np.ndarray, k: int) -> np.ndarray:
    """Indices of about ``k`` rows whose steps are nearest to log-spaced targets (always the first and last row)."""
    n = step.size
    if n <= k:
        return np.arange(n)
    lo = max(float(step[step > 0].min()) if np.any(step > 0) else 1.0, 1.0)
    hi = max(float(step.max()), lo)
    targets = np.geomspace(lo, hi, k - 1)
    nearest = np.abs(step[None, :] - targets[:, None]).argmin(axis=1)
    return np.unique(np.concatenate([[0, n - 1], nearest]))
