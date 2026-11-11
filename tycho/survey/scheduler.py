"""Survey scheduler (SPEC §6.3): what to observe next, and which objects deserve RI spectroscopy.

Restart safety and determinism
    Every decision is a function of the sky and the catalog (the scheduler only caches pure functions of the sky, such
    as object ids, never decisions), and all enqueueing goes through ``Catalog.add_objects`` (INSERT OR IGNORE by
    object id). A daemon can therefore be killed and restarted at any moment without duplicating work, and a fresh
    ``Scheduler`` on the same catalog continues exactly where the previous one stopped. The one piece of persistent
    scheduler state is the refinement depth of midpoints (see Refinement), kept in the catalog's ``meta`` table.

Coarse coverage (:meth:`Scheduler.top_up`)
    The queue is the number of claimable objects (queued + interrupted). When it is below ``target_queue / 2`` it is
    refilled to ``target_queue``, first from unobserved coarse-grid points, then by :meth:`refine`. Each region's grid
    points are visited in a fixed pseudo-random order: sorted by sha256(sky.version, region.name, point_key), a shuffle
    seeded by (sky.version, region.name) that depends neither on numpy's generator algorithms nor on the grid's
    enumeration order. A point is unobserved while any of its initial seeds (0 … n0 − 1, n0 = ``seeds_initial`` or the
    region's ``seeds``) is missing from the catalog; its missing seeds are enqueued together. Regions are interleaved
    by stride scheduling: the next point comes from the region with the fewest objects enqueued in this call per unit
    of region priority. Regions should not overlap: region is not part of an object's identity, so a configuration
    shared by two regions is one object, recorded under the region that enqueued it first and counted once.
    Coarse objects get ``priority = region.priority`` and ``source = "coarse"``; each is marked as
    a random control with probability ``random_control_frac`` (uniform draws from the scheduler's ``rng``). Only
    coarse objects are marked, so the control sample is a uniform sample of the sky's design rather than of the
    boundary-enriched refinement.

Boundaries (:meth:`Scheduler.boundary_pairs`)
    Objects of a region with the current sky version are aggregated by point (``point_key(sky.point_of(...))``;
    objects whose point is not valid in the current sky are ignored). A point is *eligible* with ≥
    :data:`MIN_FINISHED_SEEDS` done seeds. Two eligible points form a pair when they differ in one knob and are
    adjacent along it: no *live* value lies between them on their line (all other knobs fixed). The live values of a
    line are the knob's grid values plus the values of all catalog points on the line, minus values whose objects all
    failed. So a pending midpoint separates its parents (they are compared again through it once it is observed), a
    grid point not yet observed prevents comparing points two steps apart, and a permanently failed point does not
    block anything. A pair is a boundary when

    * the majority labels differ: the majority label of a point is the latest-classification label held by more than
      half of its labelled done seeds, :data:`MIXED_LABEL` if there is none (needs ≥ MIN_FINISHED_SEEDS labelled
      seeds); or
    * the medians of log10(max(t_test_acc90, 1)) over done seeds differ by more than :data:`TIMING_BOUNDARY_DEX`
      decades. A seed that never reached 90 % test accuracy counts as +inf, so "mostly generalizes" vs "mostly never
      does" is a boundary while two points that never do are not.

Refinement (:meth:`Scheduler.refine`)
    For every boundary pair: both endpoints get ``extra_seeds`` additional seeds (seeds 0 … n0 + extra_seeds − 1 in
    total; idempotent), and when the differing knob is refinable a midpoint (``Knob.midpoint``: geometric for log,
    arithmetic for linear, rounded for integers) gets n0 seeds, provided its depth is ≤ ``max_refine_depth``. Depth:
    coarse-grid points, and any point without a record (e.g. agent proposals), have depth 0; a midpoint has depth
    1 + max(depth of its endpoints). Depths are stored in ``meta`` under ``scheduler/refine_depth/<region>/<point_key>``
    *before* the midpoint's objects are enqueued, so a crash between the two never loses a depth (a dangling record
    without objects is harmless). Every point that receives new objects gets an audit row in ``proposals`` (source
    "scheduler", status "enqueued") stating the boundary that motivated it. Refinement objects have ``source =
    "refine"`` and ``priority = region.priority × refine_priority_boost``; regions are refined in priority order.

Follow-up selection (:meth:`Scheduler.select_followups`)
    Candidates are done, non-diverged objects of the current sky version without a follow-up row, in regions with
    ``ri_policy = "followup"`` (``always`` regions are observed in situ, ``never`` regions not at all). Selection
    order (sky v1; in v0 every follow-up went to "grokking"-labelled objects because the label counted as a
    transient, which left no controls):

    1. unknown class (latest label "unknown"), oldest first;
    2. RI-based transients, most severe first. Photometric transient kinds (:data:`PHOTOMETRIC_TRANSIENTS`:
       grokking, loss_spike, unknown_class) do not count: they restate the light curve, which every object has;
    3. everything else, region by region by stride scheduling (the next pick comes from the region with the fewest
       follow-ups so far, counting earlier calls, per unit of region priority). Within a region four streams take
       turns, the first turn rotating with the region's follow-up count so that small calls stay fair:

       * anomaly: objects whose light curve is unlike the rest of their (region, label) group
         (:func:`tycho.analysis.anomaly.anomaly_scores` ≥ :data:`tycho.analysis.anomaly.THRESHOLD`), highest first.
         This is the survey's "unknown" exit for curves the classifier labels but that do not fit the label;
       * matched: points whose done seeds carry two or more labels (same configuration, different outcome, e.g.
         grokking and memorization). Each such point contributes one seed per label, so the RI curves of the
         different outcomes form matched controls; every matched point gets its k-th set before any its (k+1)-th;
       * boundary: all done seeds of points in boundary pairs, round-robin over points by seed rank;
       * stratified: the region's labels take turns, the label with the fewest follow-ups first; within a label
         random-control objects first, then a fixed pseudo-random order (sha256 of sky version, region, object id).

    Each object is selected at most once, and selections are recorded with ``Catalog.request_followup``
    (requested_by "scheduler", priority :data:`FOLLOWUP_PRIORITY` of the stream), which also makes "never twice"
    hold across calls and restarts. At most ``followups_per_day`` scheduler requests are made per 24 h.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from tycho.analysis.anomaly import THRESHOLD as ANOMALY_THRESHOLD, anomaly_scores
from tycho.analysis.classify import LABELS
from tycho.survey.catalog import Catalog
from tycho.survey.sky import Knob, Region, Sky, point_key
from tycho.train.spec import ObjectSpec

logger = logging.getLogger(__name__)

MIN_FINISHED_SEEDS = 2
"""Done seeds a point needs before it can be part of a boundary (and labelled seeds for the label criterion)."""

TIMING_BOUNDARY_DEX = 0.5
"""Two points whose median log10 t_test_acc90 differ by more than this many decades are on a boundary."""

MIXED_LABEL = "mixed"
"""Majority label of a point whose labelled seeds have no strict majority."""

SOURCE_COARSE = "coarse"
SOURCE_REFINE = "refine"
REQUESTER = "scheduler"
"""``source`` of the scheduler's proposals rows and ``requested_by`` of its follow-up requests."""

DEPTH_META_PREFIX = "scheduler/refine_depth/"
"""Meta key prefix of midpoint depths: ``<prefix><region>/<point_key>`` → depth (decimal text)."""

FOLLOWUP_PRIORITY: dict[str, float] = {"unknown_class": 4.0, "transient": 3.0, "anomaly": 2.5, "matched": 2.0,
                                       "boundary": 2.0, "stratified": 1.5, "random_control": 1.5}
"""Follow-up priority by selection class (higher is claimed first by ``Catalog.claim_followup``)."""

PHOTOMETRIC_TRANSIENTS = frozenset({"grokking", "loss_spike", "unknown_class"})
"""Transient kinds derived from the light curve alone; they do not raise follow-up priority."""

REGION_STREAMS = ("anomaly", "matched", "boundary", "stratified")
"""Per-region follow-up streams, in turn order (see Follow-up selection)."""

_SECONDS_PER_DAY = 86_400.0


@dataclass
class SchedulerPolicy:
    """Scheduling policy (the ``scheduler`` section of the survey config, plus ``refine_priority_boost``)."""

    target_queue: int = 200
    seeds_initial: int | None = None  # None -> region.seeds
    boundary_refine: bool = True
    max_refine_depth: int = 3
    extra_seeds: int = 2
    random_control_frac: float = 0.05
    followups_per_day: int = 200
    refine_priority_boost: float = 1.0

    def __post_init__(self) -> None:
        for name, minimum in (("target_queue", 0), ("max_refine_depth", 0), ("extra_seeds", 0),
                              ("followups_per_day", 0)):
            _check_int(name, getattr(self, name), minimum)
        if self.seeds_initial is not None:
            _check_int("seeds_initial", self.seeds_initial, 1)
        if not isinstance(self.boundary_refine, bool):
            raise ValueError(f"boundary_refine must be a bool, got {self.boundary_refine!r}")
        if not 0.0 <= float(self.random_control_frac) <= 1.0:
            raise ValueError(f"random_control_frac must be in [0, 1], got {self.random_control_frac}")
        if not (math.isfinite(float(self.refine_priority_boost)) and self.refine_priority_boost > 0):
            raise ValueError(f"refine_priority_boost must be positive, got {self.refine_priority_boost}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SchedulerPolicy":
        """From the config's ``scheduler`` section (unknown keys are errors)."""
        d = dict(d or {})
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown scheduler policy keys {unknown}")
        return cls(**d)


@dataclass
class _Point:
    """All catalog objects of one point (rows of ``Catalog.region_objects``)."""

    point: dict[str, Any]
    key: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def done(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == "done"]

    @property
    def dead(self) -> bool:
        """Every object of the point failed: the point is permanently unobservable."""
        return bool(self.rows) and all(r["status"] == "failed" for r in self.rows)

    def majority_label(self) -> str | None:
        labels = [r["label"] for r in self.done if r["label"] is not None]
        if len(labels) < MIN_FINISHED_SEEDS:
            return None
        label, count = Counter(labels).most_common(1)[0]
        return label if 2 * count > len(labels) else MIXED_LABEL

    def median_log_t90(self) -> float | None:
        values = [math.log10(max(t, 1.0)) if t is not None and math.isfinite(t) else math.inf
                  for t in (r["t_test_acc90"] for r in self.done)]
        return float(statistics.median(values)) if len(values) >= MIN_FINISHED_SEEDS else None


@dataclass
class _Boundary:
    a: _Point  # lower position along the knob
    b: _Point
    knob: Knob
    reason: str


class Scheduler:
    """Coarse coverage, boundary refinement and follow-up selection over one sky (see the module docstring).

    Args:
        catalog: the survey catalog.
        sky: the sky being surveyed.
        policy: scheduling policy (default ``SchedulerPolicy()``).
        rng: generator for random-control marks (default: seeded from the sky version).
        device: ``ObjectSpec.device`` of enqueued objects (bookkeeping, not identity).
    """

    def __init__(self, catalog: Catalog, sky: Sky, policy: SchedulerPolicy | None = None,
                 rng: np.random.Generator | None = None, *, device: str = "cpu") -> None:
        self.catalog = catalog
        self.sky = sky
        self.policy = policy if policy is not None else SchedulerPolicy()
        self.rng = rng if rng is not None else np.random.default_rng(_hash_int("scheduler", sky.version))
        self.device = device
        # Caches of pure functions of the (immutable) sky, so that the idempotent re-evaluation of boundaries and
        # refinements on every call stays cheap: coarse plans per region, object ids per (region, point key, seed),
        # the point key of each catalog object (None when it is not a valid point of its region) and the point of
        # each key (shared by the seeds of a point).
        self._plans: dict[str, list[tuple[list[ObjectSpec], list[str]]]] = {}
        self._ids: dict[tuple[str, str, int], str] = {}
        self._row_keys: dict[str, str | None] = {}
        self._points: dict[str, dict[str, Any]] = {}

    def initial_seeds(self, region: Region) -> int:
        """Seeds per point before refinement (``policy.seeds_initial`` or the region's ``seeds``)."""
        return self.policy.seeds_initial if self.policy.seeds_initial is not None else region.seeds

    # ------------------------------------------------------------------ coarse coverage
    def top_up(self) -> int:
        """Refill the queue to ``target_queue`` once it falls below half of it: unobserved coarse-grid points first
        (deterministic shuffled order per region, interleaved by region priority), then refinement. Returns the
        number of objects enqueued."""
        target = self.policy.target_queue
        queued = self.catalog.queue_size()
        if queued >= target / 2:
            return 0
        need = target - queued
        added = self._enqueue_coarse(need)
        if added < need and self.policy.boundary_refine:
            added += self.refine(max_new=need - added)
        (logger.info if added else logger.debug)("top_up: queue %d -> %d (target %d)", queued, queued + added,
                                                   target)
        return added

    def enqueue_coarse(self, limit: int) -> int:
        """Enqueue up to ``limit`` objects of unobserved coarse-grid points now, regardless of the queue level
        (used by ``tycho survey enqueue``). Returns the number enqueued."""
        return self._enqueue_coarse(limit) if limit > 0 else 0

    def _plan(self, region: Region) -> list[tuple[list[ObjectSpec], list[str]]]:
        """Coarse grid of a region in visiting order: (specs of the initial seeds, their object ids) per point."""
        plan = self._plans.get(region.name)
        if plan is None:
            n0 = self.initial_seeds(region)
            points = sorted(self.sky.points(region),
                            key=lambda p: _hash_int(self.sky.version, region.name, point_key(p)))
            plan = []
            for p in points:
                specs = [self.sky.make_spec(region, p, seed, self.device) for seed in range(n0)]
                plan.append((specs, [s.object_id() for s in specs]))
            self._plans[region.name] = plan
        return plan

    def _enqueue_coarse(self, need: int) -> int:
        """Enqueue whole unobserved coarse points, regions interleaved by stride scheduling, until ``need`` objects
        were selected (the last point may overshoot by its remaining seeds). Returns the number enqueued."""
        existing = self.catalog.object_ids()
        order = {r.name: i for i, r in enumerate(self.sky.regions)}
        cursors = {r.name: iter(self._plan(r)) for r in self.sky.regions}
        load = dict.fromkeys(order, 0.0)
        active = list(self.sky.regions)
        groups: list[tuple[Region, list[ObjectSpec]]] = []
        n = 0
        while n < need and active:
            region = min(active, key=lambda r: (load[r.name], order[r.name]))
            pending = next(([(s, oid) for s, oid in zip(specs, ids) if oid not in existing]
                            for specs, ids in cursors[region.name] if not existing.issuperset(ids)), None)
            if pending is None:
                active.remove(region)
                continue
            existing.update(oid for _, oid in pending)  # regions that share objects must not count them twice
            new_specs = [s for s, _ in pending]
            if groups and groups[-1][0] is region:
                groups[-1][1].extend(new_specs)
            else:
                groups.append((region, new_specs))
            n += len(new_specs)
            load[region.name] += len(new_specs) / region.priority
        added = 0
        for region, specs in groups:  # in interleaved order, so FIFO claims keep the interleaving
            added += self.catalog.add_objects(specs, region=region.name, source=SOURCE_COARSE,
                                              priority=region.priority,
                                              random_control_frac=self.policy.random_control_frac, rng=self.rng)
        return added

    # ------------------------------------------------------------------ boundaries
    def boundary_pairs(self, region: Region | str) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
        """Neighbouring eligible points of a region whose majority labels or generalization times differ, as
        (point_a, point_b, reason) with point_a lower along the differing knob."""
        reg = self._region(region)
        return [(copy.deepcopy(bd.a.point), copy.deepcopy(bd.b.point), bd.reason)  # the points are cached
                for bd in self._boundaries(reg, self._observe(reg))]

    def _region(self, region: Region | str) -> Region:
        return region if isinstance(region, Region) else self.sky.region(region)

    def _observe(self, region: Region, rows: Sequence[dict[str, Any]] | None = None) -> dict[str, _Point]:
        """Catalog objects of the region (current sky version) aggregated by point, in insertion order."""
        if rows is None:
            rows = self.catalog.region_objects(region.name)
        points: dict[str, _Point] = {}
        for row in rows:
            if row["sky_version"] != self.sky.version:
                continue
            key = self._point_key_of_row(region, row)
            if key is not None:
                points.setdefault(key, _Point(point=self._points[key], key=key)).rows.append(row)
        return points

    def _point_key_of_row(self, region: Region, row: Mapping[str, Any]) -> str | None:
        """Point key of a catalog object (its point is then in ``self._points``), or None if its spec is not a valid
        point of the region."""
        oid = row["object_id"]
        if oid not in self._row_keys:
            key = None
            try:
                point = self.sky.point_of(region, ObjectSpec.from_json(row["spec_json"]))
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("object %s: no point in region %s: %s", oid, region.name, exc)
            else:
                if not self.sky.validate_point(region, point):  # else off the current grid or bounds
                    key = point_key(point)
                    self._points.setdefault(key, point)
            self._row_keys[oid] = key
        return self._row_keys[oid]

    def _object_id(self, region: Region, point: Mapping[str, Any], key: str, seed: int) -> str:
        """Object id of a point and seed (cached)."""
        cache_key = (region.name, key, seed)
        oid = self._ids.get(cache_key)
        if oid is None:
            oid = self._ids[cache_key] = self.sky.make_spec(region, point, seed, self.device).object_id()
        return oid

    def _boundaries(self, region: Region, points: Mapping[str, _Point]) -> list[_Boundary]:
        eligible = {k for k, p in points.items() if len(p.done) >= MIN_FINISHED_SEEDS}
        out: list[_Boundary] = []
        if not eligible:
            return out
        for knob in region.knobs:
            lines: dict[str, list[_Point]] = defaultdict(list)
            for p in points.values():
                lines[point_key({k: v for k, v in p.point.items() if k != knob.name})].append(p)
            for line in sorted(lines):
                on_line = lines[line]
                if not any(p.key in eligible for p in on_line):
                    continue
                by_pos = {knob.position(p.point[knob.name]): p for p in on_line}
                live: dict[float, _Point | None] = {}
                for v in knob.values:  # grid values: live unless observed and all failed
                    pos = knob.position(v)
                    if pos not in by_pos:
                        live[pos] = None
                live.update({pos: p for pos, p in by_pos.items() if not p.dead})
                ordered = sorted(live)
                for lo, hi in zip(ordered, ordered[1:]):
                    a, b = live[lo], live[hi]
                    if a is None or b is None or a.key not in eligible or b.key not in eligible:
                        continue
                    reason = _difference(a, b)
                    if reason:
                        out.append(_Boundary(a=a, b=b, knob=knob, reason=reason))
        return out

    # ------------------------------------------------------------------ refinement
    def refine(self, *, max_new: int | None = None) -> int:
        """Bracket every boundary pair with extra seeds and (on refinable knobs, within ``max_refine_depth``) a
        midpoint. Idempotent; regions in priority order; a pair's objects are enqueued together, and with
        ``max_new`` no further pair is started once that many objects were added. Returns the number enqueued."""
        if not self.policy.boundary_refine or (max_new is not None and max_new <= 0):
            return 0
        total = 0
        order = {r.name: i for i, r in enumerate(self.sky.regions)}
        for region in sorted(self.sky.regions, key=lambda r: (-r.priority, order[r.name])):
            boundaries = self._boundaries(region, self._observe(region))
            if not boundaries:
                continue
            prefix = f"{DEPTH_META_PREFIX}{region.name}/"
            depths = {k[len(prefix):]: int(v) for k, v in self.catalog.meta_items(prefix).items()}
            existing = self.catalog.object_ids(region.name)
            n0 = self.initial_seeds(region)
            for bd in boundaries:
                targets = [(bd.a.point, bd.a.key, n0 + self.policy.extra_seeds, None),
                           (bd.b.point, bd.b.key, n0 + self.policy.extra_seeds, None)]
                mid = self._midpoint(bd, depths)
                if mid is not None:
                    targets.append((mid[0], point_key(mid[0]), n0, mid[1]))
                todo = []
                for point, key, n_seeds, depth in targets:
                    seeds = [s for s in range(n_seeds) if self._object_id(region, point, key, s) not in existing]
                    if seeds:
                        todo.append((point, key, depth,
                                     [self.sky.make_spec(region, point, s, self.device) for s in seeds]))
                n_pair = sum(len(t[3]) for t in todo)
                if n_pair == 0:
                    continue
                if max_new is not None and total > 0 and total + n_pair > max_new:
                    return total
                total += self._enqueue_refinement(region, bd, todo, depths, existing, prefix)
                if max_new is not None and total >= max_new:
                    return total
        if total:
            logger.info("refine: enqueued %d objects", total)
        return total

    def _midpoint(self, bd: _Boundary, depths: Mapping[str, int]) -> tuple[dict[str, Any], int] | None:
        """(midpoint, depth) of a boundary pair, or None (knob not refinable, too deep, or no value in between)."""
        if not bd.knob.refine:
            return None
        depth = 1 + max(depths.get(bd.a.key, 0), depths.get(bd.b.key, 0))
        if depth > self.policy.max_refine_depth:
            return None
        value = bd.knob.midpoint(bd.a.point[bd.knob.name], bd.b.point[bd.knob.name])
        if value is None:
            return None
        point = dict(bd.a.point)
        point[bd.knob.name] = value
        return point, depth

    def _enqueue_refinement(self, region: Region, bd: _Boundary,
                            todo: list[tuple[dict[str, Any], str, int | None, list[ObjectSpec]]],
                            depths: dict[str, int], existing: set[str], prefix: str) -> int:
        added = 0
        for point, key, depth, specs in todo:
            if depth is not None and key not in depths:  # record the depth before the objects exist
                self.catalog.set_meta(prefix + key, str(depth))
                depths[key] = depth
            n = self.catalog.add_objects(specs, region=region.name, source=SOURCE_REFINE,
                                         priority=region.priority * self.policy.refine_priority_boost)
            existing.update(s.object_id() for s in specs)
            if n == 0:  # enqueued meanwhile (e.g. by a region sharing these objects)
                continue
            role = f"midpoint at depth {depth}" if depth is not None else "extra seeds at an endpoint"
            rationale = (f"boundary refinement ({role}) along {bd.knob.name}: {bd.reason}; between {bd.a.key} "
                         f"and {bd.b.key}")
            self.catalog.add_proposal(REQUESTER, region.name, point, task=region.task, seeds=len(specs),
                                      rationale=rationale, status="enqueued", n_enqueued=n,
                                      decided_at=self.catalog.now())
            added += n
        return added

    # ------------------------------------------------------------------ follow-ups
    def select_followups(self, n: int, *, request: bool = True) -> list[tuple[str, str]]:
        """Up to ``n`` objects for RI follow-up as (object_id, reason), in the SPEC priority order (unknown class >
        transient-bearing > boundary members > random control), never an object twice, within the daily budget.

        With ``request`` (default) each selection is recorded with ``Catalog.request_followup``; with
        ``request=False`` nothing is written (a preview).
        """
        used = self.catalog.count_followups(requested_by=REQUESTER, since=self.catalog.now() - _SECONDS_PER_DAY)
        n = min(int(n), self.policy.followups_per_day - used)
        selected: list[tuple[str, str]] = []
        if n <= 0:
            return selected
        seen: set[str] = set()
        for category, oid, reason in self._followup_candidates():
            if oid in seen:
                continue
            seen.add(oid)
            if request and not self.catalog.request_followup(oid, reason, priority=FOLLOWUP_PRIORITY[category],
                                                             requested_by=REQUESTER):
                continue  # requested meanwhile by someone else
            selected.append((oid, reason))
            if len(selected) >= n:
                break
        if selected:
            logger.info("select_followups: %d selected%s", len(selected), "" if request else " (preview)")
        return selected

    def _followup_candidates(self) -> Iterator[tuple[str, str, str]]:
        """(category, object_id, reason) in selection order; may repeat an object across categories."""
        unknown, transient = [], []
        streams: list[tuple[Region, int, Iterator[tuple[str, str, str]]]] = []
        followed: dict[str, int] = {}
        for region_rank, region in enumerate(self.sky.regions):
            if region.ri_policy != "followup":
                continue
            rows = [r for r in self.catalog.region_objects(region.name) if r["sky_version"] == self.sky.version]
            eligible = {r["object_id"] for r in rows
                        if r["status"] == "done" and not r["has_followup"] and r["diverged"] != 1}
            for r in rows:
                if r["object_id"] not in eligible:
                    continue
                age = (r["finished_at"] if r["finished_at"] is not None else math.inf, r["object_id"])
                if r["label"] == "unknown":
                    unknown.append((age, r["object_id"], "unknown_class: light curve matched no known class"))
                kinds = [k for k in r["transient_kinds"] if k not in PHOTOMETRIC_TRANSIENTS]
                if kinds:
                    reason = f"transient: {', '.join(kinds)} (max severity {r['max_severity']})"
                    transient.append(((-(r["max_severity"] or 0), *age), r["object_id"], reason))
            followed[region.name] = sum(1 for r in rows if r["has_followup"])
            streams.append((region, region_rank, self._region_stream(region, rows, eligible, followed[region.name])))
        for category, items in (("unknown_class", unknown), ("transient", transient)):
            for _, oid, reason in sorted(items, key=lambda item: item[0]):
                yield category, oid, reason
        picked: dict[str, int] = defaultdict(int)
        while streams:
            i = min(range(len(streams)), key=lambda j: ((followed[streams[j][0].name] + picked[streams[j][0].name])
                                                        / streams[j][0].priority, streams[j][1]))
            region, _, stream = streams[i]
            item = next(stream, None)
            if item is None:
                streams.pop(i)
                continue
            picked[region.name] += 1
            yield item

    def _region_stream(self, region: Region, rows: Sequence[dict[str, Any]], eligible: set[str],
                       n_followed: int) -> Iterator[tuple[str, str, str]]:
        """The region's anomaly, matched, boundary and stratified streams, taking turns (first turn rotating with
        ``n_followed``); an object is yielded at most once."""
        points = self._observe(region, rows)
        subs = {"anomaly": self._anomaly_stream(region, rows, eligible),
                "matched": self._matched_stream(region, points, eligible),
                "boundary": self._boundary_stream(region, points, eligible),
                "stratified": self._stratified_stream(region, rows, eligible)}
        start = n_followed % len(REGION_STREAMS)
        order = [subs[name] for name in REGION_STREAMS[start:] + REGION_STREAMS[:start]]
        seen: set[str] = set()
        while order:
            for it in list(order):
                for item in it:
                    if item[1] not in seen:
                        seen.add(item[1])
                        yield item
                        break
                else:
                    order.remove(it)

    def _anomaly_stream(self, region: Region, rows: Sequence[dict[str, Any]],
                        eligible: set[str]) -> Iterator[tuple[str, str, str]]:
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r["status"] == "done" and r["label"] is not None and r["diverged"] != 1:
                by_label[r["label"]].append(r)
        found = []
        for label, members in by_label.items():
            summaries = [_summary_of(r) for r in members]
            for r, score in zip(members, anomaly_scores(summaries)):
                if score is not None and score[0] >= ANOMALY_THRESHOLD and r["object_id"] in eligible:
                    found.append((-score[0], r["object_id"], label, score))
        for _, oid, label, (z, feature) in sorted(found):
            yield "anomaly", oid, f"anomaly: unlike the other {label} objects of {region.name} ({feature}, z = {z:.1f})"

    def _matched_stream(self, region: Region, points: Mapping[str, _Point],
                        eligible: set[str]) -> Iterator[tuple[str, str, str]]:
        items = []
        for key, p in points.items():
            labelled = [r for r in p.done if r["label"] is not None and r["diverged"] != 1]
            labels = sorted({r["label"] for r in labelled}, key=_label_rank)
            if len(labels) < 2:
                continue
            point_rank = _hash_int(self.sky.version, region.name, "matched", key)
            for label_rank, label in enumerate(labels):
                members = sorted((r for r in labelled if r["label"] == label
                                  and (r["object_id"] in eligible or r["has_followup"])), key=lambda r: r["seed"])
                for rank, r in enumerate(members):
                    if r["object_id"] in eligible:
                        reason = (f"matched: labels {', '.join(labels)} at one configuration of {region.name}; "
                                  f"this seed {label}")
                        items.append(((rank, point_rank, label_rank), r["object_id"], reason))
        for _, oid, reason in sorted(items):
            yield "matched", oid, reason

    def _boundary_stream(self, region: Region, points: Mapping[str, _Point],
                         eligible: set[str]) -> Iterator[tuple[str, str, str]]:
        items = []
        for bi, bd in enumerate(self._boundaries(region, points)):
            reason = f"boundary along {bd.knob.name}: {bd.reason}"
            for side, p in enumerate((bd.a, bd.b)):
                # ranks count seeds already followed up, so across calls every boundary point gets its k-th
                # replay before any point gets its (k+1)-th
                members = sorted((r for r in p.done if r["object_id"] in eligible or r["has_followup"]),
                                 key=lambda r: r["seed"])
                for rank, r in enumerate(members):
                    if r["object_id"] in eligible:
                        items.append(((rank, bi, side), r["object_id"], reason))
        for _, oid, reason in sorted(items):
            yield "boundary", oid, reason

    def _stratified_stream(self, region: Region, rows: Sequence[dict[str, Any]],
                           eligible: set[str]) -> Iterator[tuple[str, str, str]]:
        queues: dict[str, list[tuple[tuple[int, int], str]]] = defaultdict(list)
        followed: Counter[str] = Counter()
        for r in rows:
            if r["status"] != "done" or r["label"] is None or r["diverged"] == 1:
                continue
            if r["has_followup"]:
                followed[r["label"]] += 1
            elif r["object_id"] in eligible:
                order = (0 if r["random_control"] else 1, _hash_int(self.sky.version, region.name, r["object_id"]))
                queues[r["label"]].append((order, r["object_id"]))
        labels = sorted(queues, key=lambda lab: (followed[lab], _label_rank(lab)))
        iters = [iter(sorted(queues[lab])) for lab in labels]
        for items in itertools.zip_longest(*iters):
            for lab, item in zip(labels, items):
                if item is None:
                    continue
                (control, _), oid = item
                if control == 0:
                    yield "random_control", oid, f"random_control: uniform control sample ({lab})"
                else:
                    yield "stratified", oid, f"stratified: {lab} objects of {region.name} take turns with other labels"


def _summary_of(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """The light-curve summary of a ``Catalog.region_objects`` row (None without one)."""
    text = row.get("summary_json")
    if not text:
        return None
    try:
        out = json.loads(text)
    except (TypeError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def _label_rank(label: str) -> tuple[int, str]:
    return (LABELS.index(label) if label in LABELS else len(LABELS), label)


def _difference(a: _Point, b: _Point) -> str:
    """Why two points are on a boundary ("" if they are not)."""
    reasons = []
    la, lb = a.majority_label(), b.majority_label()
    if la is not None and lb is not None and la != lb:
        reasons.append(f"majority labels {la} vs {lb}")
    ta, tb = a.median_log_t90(), b.median_log_t90()
    if ta is not None and tb is not None and not (math.isinf(ta) and math.isinf(tb)) \
            and abs(ta - tb) > TIMING_BOUNDARY_DEX:
        reasons.append(f"median t_test_acc90 {_dex(ta)} vs {_dex(tb)}")
    return "; ".join(reasons)


def _dex(log_t: float) -> str:
    return "never" if math.isinf(log_t) else f"10^{log_t:.2f}"


def _hash_int(*parts: str) -> int:
    """Stable 64-bit integer from sha256 of the parts (independent of Python's hash randomization)."""
    return int.from_bytes(hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()[:8], "big")


def _check_int(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
