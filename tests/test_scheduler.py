"""Tests of tycho.survey.scheduler: coarse coverage, restart safety, boundaries, refinement and follow-up selection."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tycho.analysis.classify import Classification
from tycho.analysis.events import Transient
from tycho.config import DEFAULT_CONFIG
from tycho.survey.catalog import Catalog
from tycho.survey.scheduler import DEPTH_META_PREFIX, FOLLOWUP_PRIORITY, Scheduler, SchedulerPolicy
from tycho.survey.sky import Sky, point_key
from tycho.train.spec import ObjectSpec, RunResult

SKY_V0 = Path(__file__).resolve().parents[1] / "configs" / "sky_v0.yaml"


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


TEST_SKY = {
    "version": "sched-test",
    "regions": [
        {"name": "line", "task": "toy", "seeds": 3, "ri_policy": "followup", "fixed": {"depth": 1},
         "train": {"steps": 100},
         "knobs": {"train.lr": {"values": [0.001, 0.01, 0.1], "kind": "log", "refine": True},
                   "mode": {"values": ["a", "b"], "kind": "categorical"}}},
        {"name": "grid2", "task": "toy", "seeds": 2, "ri_policy": "never",
         "knobs": {"width": {"values": [32, 128], "kind": "log", "refine": True},
                   "frac": {"values": [0.2, 0.6], "kind": "linear", "refine": True}}},
        {"name": "aux", "task": "toy2", "seeds": 5, "ri_policy": "followup", "knobs": {}},
        {"name": "ctrl", "task": "toy3", "seeds": 2, "priority": 3.0, "ri_policy": "always",
         "knobs": {"width": [8, 16]}},
    ],
}


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sky() -> Sky:
    return Sky.from_dict(TEST_SKY)


@pytest.fixture
def cat(tmp_path, clock) -> Catalog:
    catalog = Catalog(tmp_path / "data" / "catalog.sqlite", clock=clock)
    yield catalog
    catalog.close()


def observe(cat: Catalog, sky: Sky, region: str, point: dict, seeds, *, label: str | None = None,
            t90: float | None = None, diverged: bool = False, random_control: bool = False,
            clock: FakeClock | None = None) -> list[str]:
    """Insert finished objects at a point with a synthetic summary and (optionally) a classification label."""
    specs = [sky.make_spec(region, point, s) for s in seeds]
    cat.add_objects(specs, region=region, source="coarse", random_control_frac=1.0 if random_control else 0.0)
    for spec in specs:
        finish(cat, spec.object_id(), label=label, t90=t90, diverged=diverged)
        if clock is not None:
            clock.t += 1
    return [s.object_id() for s in specs]


def finish(cat: Catalog, oid: str, *, label: str | None = None, t90: float | None = None,
           diverged: bool = False) -> None:
    summary = {"t_test_acc90": math.nan if t90 is None else t90, "diverged": diverged}
    cat.mark_done(RunResult(object_id=oid, status="done", summary=summary, light_curve={"step": np.arange(3)}),
                  f"data/lightcurves/{oid}.npz")
    if label is not None:
        cat.add_classification(oid, Classification(label=label), "rules-v1")


def finish_new(cat: Catalog, sky: Sky, region: str, labels: dict[str, tuple[str, float | None]]) -> int:
    """Finish every queued object of a region with the (label, t90) of its point's ``lr`` (by point key of lr)."""
    n = 0
    for row in cat.region_objects(region):
        if row["status"] != "queued":
            continue
        point = sky.point_of(region, cat.get_spec(row["object_id"]))
        label, t90 = labels[point_key({"lr": point["train.lr"]})]
        finish(cat, row["object_id"], label=label, t90=t90)
        n += 1
    return n


def lr_key(lr: float) -> str:
    return point_key({"lr": lr})


def seeds_at(cat: Catalog, sky: Sky, region: str, point: dict) -> set[int]:
    key = point_key(point)
    return {r["seed"] for r in cat.region_objects(region)
            if point_key(sky.point_of(region, cat.get_spec(r["object_id"]))) == key}


# ---------------------------------------------------------------------------------------------- policy
def test_policy_from_config_and_validation():
    policy = SchedulerPolicy.from_dict(DEFAULT_CONFIG["scheduler"])
    assert policy == SchedulerPolicy()
    with pytest.raises(ValueError, match="unknown"):
        SchedulerPolicy.from_dict({"target_queu": 10})
    for bad in ({"target_queue": -1}, {"seeds_initial": 0}, {"random_control_frac": 2.0}, {"extra_seeds": 1.5},
                {"boundary_refine": 1}, {"refine_priority_boost": 0.0}):
        with pytest.raises(ValueError):
            SchedulerPolicy(**bad)


# ---------------------------------------------------------------------------------------------- coarse coverage
def test_top_up_fills_the_queue_with_complete_coarse_points(tmp_path):
    sky = Sky.from_yaml(SKY_V0)
    cat = Catalog(tmp_path / "catalog.sqlite")
    sch = Scheduler(cat, sky, SchedulerPolicy(target_queue=50))
    added = sch.top_up()
    assert added == cat.queue_size() and 50 <= added < 50 + 5  # whole points: overshoot < max seeds per point
    for region in sky.regions:
        rows = cat.region_objects(region.name)
        by_point: dict[str, set[int]] = {}
        grid = {point_key(p) for p in sky.points(region)}
        for r in rows:
            key = point_key(sky.point_of(region, cat.get_spec(r["object_id"])))
            assert key in grid and r["source"] == "coarse" and r["status"] == "queued"
            by_point.setdefault(key, set()).add(r["seed"])
        assert all(seeds == set(range(region.seeds)) for seeds in by_point.values())
        assert all(json.loads(r["spec_json"])["sky_version"] == "sky-v0" for r in rows)
    assert sch.top_up() == 0  # queue >= target / 2
    assert Scheduler(cat, sky, SchedulerPolicy(target_queue=50)).top_up() == 0
    cat.close()


def _drain(cat: Catalog, worker: str = "w") -> list[str]:
    """Claim everything claimable and mark it done (no labels: no boundaries)."""
    order = []
    while (spec := cat.claim_next(worker)) is not None:
        order.append(spec.object_id())
        finish(cat, spec.object_id())
    return order


def test_top_up_is_deterministic_and_restart_safe(tmp_path):
    sky = Sky.from_yaml(SKY_V0)
    policy = SchedulerPolicy(target_queue=60, random_control_frac=0.0)

    def survey(path: Path, fresh_scheduler_each_round: bool) -> list[str]:
        cat = Catalog(path)
        sch = Scheduler(cat, sky, policy)
        claimed = []
        for _ in range(4):
            if fresh_scheduler_each_round:  # a daemon restart between rounds
                sch = Scheduler(cat, sky, policy)
            sch.top_up()
            sch.top_up()  # repeated calls never duplicate
            claimed += _drain(cat)
        assert cat.stats()["objects"]["total"] == len(claimed)
        cat.close()
        return claimed

    reference = survey(tmp_path / "a.sqlite", False)
    restarted = survey(tmp_path / "b.sqlite", True)
    assert reference == restarted  # same objects in the same order
    assert len(set(reference)) == len(reference) >= 4 * 60


def test_coarse_order_is_a_shuffle_seeded_by_version_and_region(tmp_path):
    region = Sky.from_yaml(SKY_V0).region("grokking").to_dict()  # 108 points

    def point_order(version: str, name: str) -> list[str]:
        sky = Sky.from_dict({"version": version, "regions": [{**region, "name": name}]})
        cat = Catalog(tmp_path / f"{version}-{name}.sqlite")
        Scheduler(cat, sky, SchedulerPolicy(target_queue=10_000, random_control_frac=0.0)).top_up()
        keys = [point_key(sky.point_of(name, ObjectSpec.from_json(r["spec_json"]))) for r in cat.region_objects(name)]
        cat.close()
        assert len(keys) == 3 * 108  # the whole grid, three seeds per point ...
        assert all(keys[i] == keys[i - i % 3] for i in range(len(keys)))  # ... enqueued together
        return keys[::3]

    base = point_order("v1", "pw")
    grid = [point_key(p) for p in Sky.from_dict({"version": "v1", "regions": [{**region, "name": "pw"}]}).points("pw")]
    assert sorted(base) == sorted(grid) and base != grid  # a permutation of the grid, not its enumeration order
    assert point_order("v1", "pw") == base
    assert point_order("v2", "pw") != base and point_order("v1", "pw2") != base


def _two_region_sky(lo_task: str) -> Sky:
    knobs = {"x": {"values": list(range(60)), "kind": "ordinal"}}
    return Sky.from_dict({"version": "w", "regions": [
        {"name": "hi", "task": "t", "seeds": 1, "priority": 3.0, "knobs": knobs},
        {"name": "lo", "task": lo_task, "seeds": 1, "priority": 1.0, "knobs": knobs}]})


def test_regions_are_interleaved_by_priority(cat):
    Scheduler(cat, _two_region_sky("t2"), SchedulerPolicy(target_queue=40)).top_up()
    counts = cat.stats()["by_region"]
    assert counts["hi"]["queued"] == 30 and counts["lo"]["queued"] == 10  # stride scheduling: 3 : 1
    claimed = [cat.claim_next("w").region for _ in range(4)]
    assert claimed == ["hi"] * 4  # higher priority is claimed first


def test_overlapping_regions_still_fill_the_queue(cat):
    """Region is not part of object identity: identical configurations in two regions are one object, recorded under
    the region that enqueued it first, and must not be counted twice."""
    added = Scheduler(cat, _two_region_sky("t"), SchedulerPolicy(target_queue=40)).top_up()
    assert added == cat.queue_size() == 40


def test_coarse_grid_exhaustion_then_nothing_without_boundaries(cat, sky):
    sch = Scheduler(cat, sky, SchedulerPolicy(target_queue=8))
    total = 0
    for _ in range(20):
        total += sch.top_up()
        _drain(cat)
    expected = sum(r.grid_size * r.seeds for r in sky.regions)
    assert total == expected == cat.stats()["objects"]["done"]
    assert sch.top_up() == 0


def test_random_control_marks_coarse_objects_only(cat, sky):
    sch = Scheduler(cat, sky, SchedulerPolicy(target_queue=1000, random_control_frac=1.0))
    sch.top_up()
    assert {r["random_control"] for r in cat.query("SELECT random_control FROM objects")} == {1}
    while (spec := cat.claim_next("w")) is not None:
        lr = sky.point_of(spec.region, spec).get("train.lr")
        finish(cat, spec.object_id(), label="no_learning" if lr == 0.001 else "grokking")
    assert sch.refine() > 0
    marks = {r["random_control"] for r in cat.query("SELECT random_control FROM objects WHERE source = 'refine'")}
    assert marks == {0}


def test_random_control_marks_are_seeded(tmp_path, sky):
    def marks(seed: int, name: str) -> list[int]:
        cat = Catalog(tmp_path / f"{name}.sqlite")
        Scheduler(cat, sky, SchedulerPolicy(target_queue=1000, random_control_frac=0.5),
                  rng=np.random.default_rng(seed)).top_up()
        out = [r["random_control"] for r in cat.query("SELECT random_control FROM objects ORDER BY rowid")]
        cat.close()
        return out

    first = marks(1, "a")
    assert marks(1, "b") == first != marks(2, "c")
    assert 0 < sum(first) < len(first)


# ---------------------------------------------------------------------------------------------- boundaries
def test_boundary_by_majority_label(cat, sky):
    observe(cat, sky, "line", {"train.lr": 0.001, "mode": "a"}, range(3), label="no_learning")
    observe(cat, sky, "line", {"train.lr": 0.01, "mode": "a"}, range(3), label="grokking", t90=1000.0)
    observe(cat, sky, "line", {"train.lr": 0.1, "mode": "a"}, range(3), label="grokking", t90=1100.0)
    sch = Scheduler(cat, sky)
    (pair,) = sch.boundary_pairs("line")
    a, b, reason = pair
    assert a == {"train.lr": 0.001, "mode": "a"} and b == {"train.lr": 0.01, "mode": "a"}
    assert "majority labels no_learning vs grokking" in reason and "never vs 10^3.00" in reason
    a["train.lr"] = 123.0  # callers get copies: mutating them cannot corrupt the scheduler's cached points
    assert sch.boundary_pairs("line")[0][0] == {"train.lr": 0.001, "mode": "a"}
    assert sch.boundary_pairs("grid2") == [] and sch.boundary_pairs(sky.region("aux")) == []


@pytest.mark.parametrize("t90_b, is_boundary", [(10_000.0, True), (2_000.0, False), (None, True)])
def test_boundary_by_generalization_time(cat, sky, t90_b, is_boundary):
    observe(cat, sky, "line", {"train.lr": 0.01, "mode": "a"}, range(3), label="smooth", t90=1000.0)
    observe(cat, sky, "line", {"train.lr": 0.1, "mode": "a"}, range(3), label="smooth", t90=t90_b)
    pairs = Scheduler(cat, sky).boundary_pairs("line")
    assert len(pairs) == int(is_boundary)
    if is_boundary:
        assert "median t_test_acc90 10^3.00 vs" in pairs[0][2] and "majority" not in pairs[0][2]


def test_points_that_never_generalize_are_not_a_timing_boundary(cat, sky):
    observe(cat, sky, "line", {"train.lr": 0.01, "mode": "a"}, range(3), label="smooth")
    observe(cat, sky, "line", {"train.lr": 0.1, "mode": "a"}, range(3), label="smooth")
    assert Scheduler(cat, sky).boundary_pairs("line") == []


def test_boundary_needs_two_finished_seeds_and_mixed_counts_as_a_label(cat, sky):
    observe(cat, sky, "line", {"train.lr": 0.001, "mode": "a"}, [0], label="no_learning")
    observe(cat, sky, "line", {"train.lr": 0.01, "mode": "a"}, range(3), label="grokking")
    sch = Scheduler(cat, sky)
    assert sch.boundary_pairs("line") == []  # one finished seed only
    observe(cat, sky, "line", {"train.lr": 0.001, "mode": "a"}, [1], label="grokking")
    (pair,) = sch.boundary_pairs("line")  # [no_learning, grokking] has no majority
    assert "majority labels mixed vs grokking" in pair[2]


def test_adjacency_respects_unobserved_dead_and_pending_points(cat, sky):
    lo, mid, hi = ({"train.lr": v, "mode": "a"} for v in (0.001, 0.01, 0.1))
    observe(cat, sky, "line", lo, range(3), label="no_learning")
    observe(cat, sky, "line", hi, range(3), label="grokking")
    sch = Scheduler(cat, sky)
    assert sch.boundary_pairs("line") == []  # the grid point in between is not observed yet
    specs = [sky.make_spec("line", mid, s) for s in range(3)]
    cat.add_objects(specs, region="line", source="coarse")
    assert sch.boundary_pairs("line") == []  # ... and is pending
    for s in specs:
        cat.mark_failed(s.object_id(), "crashed")
    (pair,) = sch.boundary_pairs("line")  # a point whose objects all failed does not separate its neighbours
    assert pair[0] == lo and pair[1] == hi


def test_objects_of_other_sky_versions_are_ignored(cat, sky):
    old = Sky.from_dict({**TEST_SKY, "version": "older"})
    observe(cat, old, "line", {"train.lr": 0.001, "mode": "a"}, range(3), label="no_learning")
    observe(cat, sky, "line", {"train.lr": 0.01, "mode": "a"}, range(3), label="grokking")
    assert Scheduler(cat, sky).boundary_pairs("line") == []


# ---------------------------------------------------------------------------------------------- refinement
def test_refine_brackets_a_boundary_and_respects_max_depth(cat, sky):
    policy = SchedulerPolicy(max_refine_depth=2, extra_seeds=2)
    labels = {lr_key(0.001): ("no_learning", None), lr_key(0.01): ("grokking", 1000.0),
              lr_key(0.1): ("grokking", 1100.0)}
    for lr in (0.001, 0.01, 0.1):
        observe(cat, sky, "line", {"train.lr": lr, "mode": "a"}, range(3), label=labels[lr_key(lr)][0],
                t90=labels[lr_key(lr)][1])
    sch = Scheduler(cat, sky, policy)
    lo, hi = {"train.lr": 0.001, "mode": "a"}, {"train.lr": 0.01, "mode": "a"}
    m1 = {"train.lr": 0.00316228, "mode": "a"}  # geometric midpoint of 1e-3 and 1e-2 to 6 significant digits

    assert sch.refine() == 2 + 2 + 3  # extra seeds 3, 4 at both endpoints; seeds 0-2 at the midpoint
    assert seeds_at(cat, sky, "line", lo) == seeds_at(cat, sky, "line", hi) == set(range(5))
    assert seeds_at(cat, sky, "line", m1) == {0, 1, 2}
    assert cat.get_meta(f"{DEPTH_META_PREFIX}line/{point_key(m1)}") == "1"
    refine_rows = cat.query("SELECT source, priority, random_control FROM objects WHERE source = 'refine'")
    assert len(refine_rows) == 7 and all(r["priority"] == 1.0 and r["random_control"] == 0 for r in refine_rows)
    proposals = cat.query("SELECT * FROM proposals ORDER BY proposal_id")
    assert [p["point_json"] for p in proposals] == [point_key(lo), point_key(hi), point_key(m1)]
    assert all(p["source"] == "scheduler" and p["status"] == "enqueued" for p in proposals)
    assert [p["n_enqueued"] for p in proposals] == [2, 2, 3] and "midpoint at depth 1" in proposals[2]["rationale"]
    assert "no_learning vs grokking" in proposals[2]["rationale"]

    assert sch.refine() == 0  # idempotent ...
    assert Scheduler(cat, sky, policy).refine() == 0  # ... also for a fresh scheduler (restart)
    assert sch.boundary_pairs("line") == []  # the pending midpoint separates the pair

    labels[lr_key(0.00316228)] = ("grokking", 1500.0)
    assert finish_new(cat, sky, "line", labels) == 7
    (pair,) = sch.boundary_pairs("line")
    assert pair[0] == lo and pair[1] == m1
    m2 = {"train.lr": float(f"{math.sqrt(0.001 * 0.00316228):.6g}"), "mode": "a"}
    assert sch.refine() == 2 + 3  # lo already has 5 seeds; extra seeds at m1; depth-2 midpoint m2
    assert cat.get_meta(f"{DEPTH_META_PREFIX}line/{point_key(m2)}") == "2"
    assert seeds_at(cat, sky, "line", m2) == {0, 1, 2}

    labels[lr_key(m2["train.lr"])] = ("grokking", 1500.0)
    assert finish_new(cat, sky, "line", labels) == 5
    (pair,) = sch.boundary_pairs("line")
    assert pair[1] == m2
    assert sch.refine() == 2  # depth 3 > max_refine_depth: only the extra seeds at m2
    assert sch.refine() == 0
    assert len(cat.meta_items(DEPTH_META_PREFIX)) == 2


def test_refine_on_non_refinable_knob_adds_extra_seeds_only(cat, sky):
    observe(cat, sky, "line", {"train.lr": 0.01, "mode": "a"}, range(3), label="grokking")
    observe(cat, sky, "line", {"train.lr": 0.01, "mode": "b"}, range(3), label="memorization")
    sch = Scheduler(cat, sky, SchedulerPolicy(extra_seeds=1))
    (pair,) = sch.boundary_pairs("line")
    assert (pair[0]["mode"], pair[1]["mode"]) == ("a", "b")
    assert sch.refine() == 2
    assert cat.meta_items(DEPTH_META_PREFIX) == {}
    assert sch.refine() == 0


def test_refine_integer_and_linear_midpoints(cat, sky):
    observe(cat, sky, "grid2", {"width": 32, "frac": 0.2}, range(2), label="smooth")
    observe(cat, sky, "grid2", {"width": 128, "frac": 0.2}, range(2), label="staged")
    observe(cat, sky, "grid2", {"width": 32, "frac": 0.6}, range(2), label="memorization")
    sch = Scheduler(cat, sky, SchedulerPolicy(extra_seeds=0, refine_priority_boost=2.0))
    assert len(sch.boundary_pairs("grid2")) == 2
    assert sch.refine() == 4  # two midpoints with 2 seeds each, no extra seeds
    specs = [cat.get_spec(r["object_id"]) for r in cat.region_objects("grid2") if r["source"] == "refine"]
    assert sorted({(s.task_cfg["width"], s.task_cfg["frac"]) for s in specs}) == [(32, 0.4), (64, 0.2)]
    assert all(isinstance(s.task_cfg["width"], int) for s in specs)
    assert {r["priority"] for r in cat.query("SELECT priority FROM objects WHERE source = 'refine'")} == {2.0}


def test_refine_max_new_keeps_pairs_together(cat, sky):
    observe(cat, sky, "grid2", {"width": 32, "frac": 0.2}, range(2), label="smooth")
    observe(cat, sky, "grid2", {"width": 128, "frac": 0.2}, range(2), label="staged")
    observe(cat, sky, "grid2", {"width": 32, "frac": 0.6}, range(2), label="memorization")
    sch = Scheduler(cat, sky, SchedulerPolicy(extra_seeds=2))
    first = sch.refine(max_new=1)  # the first pair is enqueued whole even though it exceeds the cap
    assert first == 2 + 2 + 2
    assert sch.refine(max_new=0) == 0
    assert sch.refine() == 2 + 2  # second pair: extra seeds at its other endpoint, and its midpoint
    assert sch.refine() == 0


def test_refine_disabled_by_policy(cat, sky):
    observe(cat, sky, "line", {"train.lr": 0.001, "mode": "a"}, range(3), label="no_learning")
    observe(cat, sky, "line", {"train.lr": 0.01, "mode": "a"}, range(3), label="grokking")
    assert Scheduler(cat, sky, SchedulerPolicy(boundary_refine=False)).refine() == 0


def test_top_up_falls_back_to_refinement_when_the_grid_is_covered(cat, sky):
    sch = Scheduler(cat, sky, SchedulerPolicy(target_queue=1000))
    sch.top_up()
    while (spec := cat.claim_next("w")) is not None:
        point = sky.point_of(spec.region, spec)
        label = "no_learning" if point.get("train.lr") == 0.001 else "grokking"
        finish(cat, spec.object_id(), label=label)
    added = sch.top_up()
    assert added > 0 and cat.query("SELECT COUNT(*) AS n FROM objects WHERE source = 'refine'")[0]["n"] == added


# ---------------------------------------------------------------------------------------------- follow-ups
@pytest.fixture
def followup_sky(cat, sky, clock):
    """Objects of every follow-up class, plus objects that must never be selected."""
    ids: dict[str, list[str]] = {}
    def aux(seed: int, **kw) -> str:
        return observe(cat, sky, "aux", {}, [seed], clock=clock, **kw)[0]

    ids["U"] = aux(0, label="unknown")
    ids["T3"] = aux(1, label="smooth")
    ids["T1"] = aux(2, label="smooth")
    ids["D"] = aux(3, label="unknown", diverged=True)
    ids["UT"] = aux(4, label="unknown")
    cat.add_transients(ids["T3"], [Transient(kind="late_activation", severity=3, message="late")], [])
    cat.add_transients(ids["T1"], [Transient(kind="loss_spike", severity=1, message="spike")], [])
    cat.add_transients(ids["UT"], [Transient(kind="extinction", severity=2, message="gone"),
                                   Transient(kind="unknown_class", severity=1, message="?")], [])
    ids["lo"] = observe(cat, sky, "line", {"train.lr": 0.001, "mode": "a"}, range(3), label="no_learning",
                        clock=clock)
    ids["mid"] = observe(cat, sky, "line", {"train.lr": 0.01, "mode": "a"}, range(3), label="grokking", t90=1000.0,
                         clock=clock)
    ids["R"] = observe(cat, sky, "line", {"train.lr": 0.1, "mode": "a"}, range(3), label="grokking", t90=1100.0,
                       random_control=True, clock=clock)
    ids["ctrl"] = observe(cat, sky, "ctrl", {"width": 8}, [0], label="unknown", clock=clock)  # ri_policy always
    assert cat.request_followup(ids["R"][0], "already requested by the agent", requested_by="agent")
    # unknown (oldest first), RI transients (T1's loss_spike is photometric and does not count), then the line and
    # aux regions by stride scheduling, each alternating its boundary / stratified (random controls first) streams
    expected = [ids["U"], ids["UT"], ids["T3"],
                ids["lo"][0], ids["T1"], ids["lo"][1], ids["mid"][0], ids["R"][2], ids["mid"][1], ids["lo"][2],
                ids["mid"][2], ids["R"][1]]
    return ids, expected


def test_select_followups_priority_order_and_dedup(cat, sky, followup_sky):
    ids, expected = followup_sky
    sch = Scheduler(cat, sky)
    preview = sch.select_followups(100, request=False)
    assert cat.count_followups() == 1  # a preview writes nothing
    selected = sch.select_followups(100)
    assert selected == preview
    assert [oid for oid, _ in selected] == expected  # UT once (as unknown), D/ctrl/R[0] never
    reasons = dict(selected)
    assert reasons[ids["U"]].startswith("unknown_class")
    assert reasons[ids["T3"]] == "transient: late_activation (max severity 3)"
    assert not reasons[ids["T1"]].startswith("transient")  # loss_spike is photometric
    assert reasons[ids["lo"][0]].startswith("boundary along train.lr: majority labels no_learning vs grokking")
    assert reasons[ids["R"][1]].startswith("random_control")
    rows = {r["object_id"]: r for r in cat.query("SELECT * FROM followups")}
    assert rows[ids["U"]]["priority"] == FOLLOWUP_PRIORITY["unknown_class"]
    assert rows[ids["T3"]]["priority"] == FOLLOWUP_PRIORITY["transient"]
    assert rows[ids["T1"]]["priority"] == FOLLOWUP_PRIORITY["stratified"]
    assert rows[ids["mid"][2]]["priority"] == FOLLOWUP_PRIORITY["boundary"]
    assert rows[ids["R"][1]]["priority"] == FOLLOWUP_PRIORITY["random_control"]
    assert rows[ids["R"][2]]["requested_by"] == "scheduler" and rows[ids["R"][2]]["status"] == "requested"
    assert sch.select_followups(100) == []  # never twice
    assert Scheduler(cat, sky).select_followups(100) == []  # not even after a restart


def test_select_followups_respects_n_and_the_daily_budget(cat, sky, clock, followup_sky):
    _, expected = followup_sky
    sch = Scheduler(cat, sky, SchedulerPolicy(followups_per_day=5))
    first = [oid for oid, _ in sch.select_followups(3)]
    assert first == expected[:3]  # unknown and transient classes come first in every call
    second = [oid for oid, _ in sch.select_followups(3)]
    assert len(second) == 2  # budget: 5 per 24 h
    assert sch.select_followups(3) == []
    clock.t += 86_401  # the earlier requests are now more than 24 h old
    third = [oid for oid, _ in sch.select_followups(100)]
    assert len(third) == 5
    picked = first + second + third
    assert len(set(picked)) == len(picked) and set(picked) <= set(expected)  # never twice, only candidates
    assert sch.select_followups(0) == []


# ---------------------------------------------------------------------------------------------- sky v1 follow-ups
def finish_with(cat: Catalog, oid: str, label: str, **summary) -> None:
    base = {"t_test_acc90": 1000.0, "t_train_acc90": 500.0, "final_train_acc": 1.0, "final_test_acc": 0.95,
            "final_train_loss": 0.01, "final_test_loss": 0.1, "plateau_logfrac": 0.2, "n_drops": 1, "test_bump": 0.0,
            "steps_total": 10_000, "diverged": False}
    base.update(summary)
    cat.mark_done(RunResult(object_id=oid, status="done", summary=base, light_curve={"step": np.arange(3)}),
                  f"data/lightcurves/{oid}.npz")
    cat.add_classification(oid, Classification(label=label), "rules-v1")


def test_grokking_label_alone_does_not_win_followups(cat, sky):
    """v0: every follow-up went to grokking objects because the label counted as a transient. Now labels take
    turns, so smooth objects of the region are followed up too."""
    for lr, label in ((0.001, "grokking"), (0.01, "grokking"), (0.1, "smooth")):
        for mode in ("a", "b"):
            specs = [sky.make_spec("line", {"train.lr": lr, "mode": mode}, i) for i in range(3)]
            cat.add_objects(specs, region="line", source="coarse", random_control_frac=0.0)
            for spec in specs:
                finish_with(cat, spec.object_id(), label)
                cat.add_transients(spec.object_id(), [Transient(kind="grokking", severity=1, message="g")]
                                   if label == "grokking" else [], [])
    picks = Scheduler(cat, sky).select_followups(4)
    labels = {cat.query("SELECT label FROM classifications WHERE object_id = ? ORDER BY id DESC LIMIT 1",
                        (oid,))[0]["label"] for oid, _ in picks}
    assert labels == {"grokking", "smooth"}
    assert not any(reason.startswith("transient") for _, reason in picks)


def test_matched_sets_follow_up_every_outcome_of_a_mixed_point(cat, sky):
    specs = [sky.make_spec("aux", {}, i) for i in range(5)]
    cat.add_objects(specs, region="aux", source="coarse", random_control_frac=0.0)
    labels = ["grokking", "grokking", "memorization", "grokking", "memorization"]
    for spec, label in zip(specs, labels):
        finish_with(cat, spec.object_id(), label)
    picks = Scheduler(cat, sky).select_followups(100)
    matched = [oid for oid, reason in picks if reason.startswith("matched")]
    # the first matched set is the lowest seed of each label, labels in classifier order: seed 2 (memorization),
    # seed 0 (grokking)
    assert matched[:2] == [specs[2].object_id(), specs[0].object_id()]
    assert "labels memorization, grokking" in dict(picks)[matched[0]]


def test_anomalous_light_curves_are_the_unknown_exit(cat, sky):
    specs = [sky.make_spec("aux", {}, i) for i in range(30)]
    cat.add_objects(specs, region="aux", source="coarse", random_control_frac=0.0)
    rng = np.random.default_rng(0)
    for i, spec in enumerate(specs):
        loss = 100.0 if i == 17 else float(0.1 * (1 + 0.05 * rng.standard_normal()))
        finish_with(cat, spec.object_id(), "smooth", final_test_loss=loss)
    picks = Scheduler(cat, sky).select_followups(1)
    assert picks[0][0] == specs[17].object_id()
    assert picks[0][1].startswith("anomaly: unlike the other smooth objects of aux (log_final_test_loss")


def test_regions_share_followups_by_priority(cat):
    sky = Sky.from_dict({"version": "share", "regions": [
        {"name": "big", "task": "toy", "seeds": 40, "priority": 3.0, "ri_policy": "followup", "knobs": {}},
        {"name": "small", "task": "toy2", "seeds": 40, "priority": 1.0, "ri_policy": "followup", "knobs": {}}]})
    for region in ("big", "small"):
        specs = [sky.make_spec(region, {}, i) for i in range(40)]
        cat.add_objects(specs, region=region, source="coarse", random_control_frac=0.0)
        for spec in specs:
            finish_with(cat, spec.object_id(), "smooth")
    picks = Scheduler(cat, sky).select_followups(40)
    regions = Counter(cat.query("SELECT region FROM objects WHERE object_id = ?", (oid,))[0]["region"]
                      for oid, _ in picks)
    assert regions == {"big": 30, "small": 10}


def test_followups_ignore_other_sky_versions(cat, sky):
    old = Sky.from_dict({**TEST_SKY, "version": "older"})
    specs = [old.make_spec("aux", {}, i) for i in range(3)]
    cat.add_objects(specs, region="aux", source="coarse", random_control_frac=0.0)
    for spec in specs:
        finish_with(cat, spec.object_id(), "unknown")
    assert Scheduler(cat, sky).select_followups(10) == []
