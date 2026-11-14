"""Tests of tycho.survey.sky: the sky format, grids, point validation, specs and identity, lattice and coordinates."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from tycho.survey.sky import MIDPOINT_SIG_DIGITS, Knob, Region, Sky, point_key
from tycho.train.spec import ObjectSpec, TrainConfig

SKY_V0 = Path(__file__).resolve().parents[1] / "configs" / "sky_v0.yaml"

GRID_SIZES = {"control-parity": 2 * 2, "control-markov": 3, "grokking": 2 * 3 * 2 * 3 * 3, "icl-markov": 2 * 2,
              "superposition": 4 * 2, "parity-wide": 3 * 2 * 3 * 4 * 3 * 3}


@pytest.fixture(scope="module")
def sky() -> Sky:
    return Sky.from_yaml(SKY_V0)


def minimal_sky(**region_overrides) -> dict:
    """A one-region sky definition (plain data) to break in validation tests."""
    region = {"name": "r", "task": "toy", "seeds": 2, "fixed": {"depth": 1}, "train": {"steps": 10},
              "knobs": {"width": {"values": [8, 32, 128], "kind": "log", "refine": True},
                        "mode": {"values": ["a", None], "kind": "categorical"}}}
    region.update(region_overrides)
    return {"version": "t", "regions": [region]}


# ---------------------------------------------------------------------------------------------- parsing
def test_sky_v0_parses(sky):
    assert sky.version == "sky-v0"
    assert [r.name for r in sky.regions] == list(GRID_SIZES)
    assert {r.name: r.ri_policy for r in sky.regions} == {
        "control-parity": "always", "control-markov": "always", "grokking": "followup", "icl-markov": "followup",
        "superposition": "followup", "parity-wide": "followup"}
    pw = sky.region("parity-wide")
    assert pw.task == "sparse_parity" and pw.seeds == 3 and pw.priority == 1.0
    n_train = pw.knob("n_train")
    assert n_train.kind == "categorical" and n_train.values == [1000, 4000, 16000, None] and not n_train.refine
    lr = pw.knob("train.lr")
    assert lr.train_field == "lr" and lr.refine and lr.bounds == (0.01, 0.1)
    assert pw.knob("width").is_integer and not lr.is_integer
    assert sky.region("control-parity").ri == {"n_probe": 4096, "n_points": 30}
    assert sky.region("grokking").train["beta2"] == 0.98
    with pytest.raises(KeyError):
        sky.region("nope")


def test_grid_sizes_per_region(sky):
    for region in sky.regions:
        points = sky.points(region)
        assert region.grid_size == GRID_SIZES[region.name] == len(points)
        assert len({point_key(p) for p in points}) == len(points)  # all distinct
        assert all(set(p) == set(region.knob_names) for p in points)
        assert all(sky.validate_point(region.name, p) == [] for p in points)


def test_sky_v0_configurations_are_known_to_their_tasks(sky):
    """Integration guard: Task rejects unknown config keys, so every key the sky produces must be a task default."""
    tasks = pytest.importorskip("tycho.tasks")
    for region in sky.regions:
        defaults = set(tasks.REGISTRY[region.task].defaults())
        keys = set().union(*(sky.make_spec(region, p, 0).task_cfg for p in sky.points(region)))
        assert keys <= defaults, (region.name, sorted(keys - defaults))


def test_points_vary_last_knob_fastest(sky):
    points = sky.points("control-parity")
    assert points == [{"k": 2, "width": 128}, {"k": 2, "width": 512}, {"k": 3, "width": 128}, {"k": 3, "width": 512}]


def test_dict_and_yaml_round_trip(sky, tmp_path):
    assert Sky.from_dict(sky.to_dict()) == sky
    path = tmp_path / "sky.yaml"
    path.write_text(yaml.safe_dump(sky.to_dict(), sort_keys=False))
    assert Sky.from_yaml(path) == sky


def test_bare_list_knob_shorthand():
    d = minimal_sky(knobs={"k": [2, 3, 4]})
    knob = Sky.from_dict(d).region("r").knob("k")
    assert knob.values == [2, 3, 4] and knob.kind == "ordinal" and not knob.refine


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["regions"][0]["knobs"]["mode"].update(refine=True), "refine requires"),
    (lambda d: d["regions"][0]["knobs"]["width"].update(values=[0, 8]), "positive"),
    (lambda d: d["regions"][0]["knobs"]["width"].update(values=[32, 8]), "strictly increasing"),
    (lambda d: d["regions"][0]["knobs"]["mode"].update(values=["a", "a"]), "duplicate"),
    (lambda d: d["regions"][0]["knobs"]["width"].update(kind="exponential"), "kind"),
    (lambda d: d["regions"][0]["knobs"]["width"].update(step=2), "unknown keys"),
    (lambda d: d["regions"][0]["knobs"]["width"].update(values=[8, float("nan")]), "not finite"),
    (lambda d: d["regions"][0]["knobs"].update(k={"values": [1, None], "kind": "ordinal"}), "ordinal"),
    (lambda d: d["regions"][0]["knobs"]["width"].update(bounds=[16, 64]), "contain"),
    (lambda d: d["regions"][0]["knobs"]["mode"].update(bounds=[0, 1]), "only apply"),
    (lambda d: d["regions"][0]["knobs"].update({"train.lrr": {"values": [0.1]}}), "no field"),
    (lambda d: d["regions"][0]["knobs"].update({"train.lr": {"values": ["1e-3", "1e-2"]}}), "train.lr"),
    (lambda d: d["regions"][0]["knobs"].update(depth=[1, 2]), "also appear"),
    (lambda d: d["regions"][0]["knobs"].update({"train.steps": [10, 20]}), "also appear"),
    (lambda d: d["regions"][0].update(fixed={"depth": float("inf")}), "finite JSON"),
    (lambda d: d["regions"][0].update(train={"stepz": 10}), "unknown TrainConfig"),
    (lambda d: d["regions"][0].update(train={"lr": "1e-3"}), "train.lr"),
    (lambda d: d["regions"][0].update(ri_policy="sometimes"), "ri_policy"),
    (lambda d: d["regions"][0].update(ri={"n_probes": 10}), "RIConfig"),
    (lambda d: d["regions"][0].update(seeds=0), "seeds"),
    (lambda d: d["regions"][0].update(priority=0), "priority"),
    (lambda d: d["regions"][0].update(name="has space"), "region name"),
    (lambda d: d["regions"][0].update(colour="red"), "unknown keys"),
    (lambda d: d["regions"].append(copy.deepcopy(d["regions"][0])), "duplicate region"),
    (lambda d: d.update(version=""), "version"),
    (lambda d: d.update(extra=1), "unknown sky keys"),
])
def test_invalid_skies_are_rejected(mutate, match):
    d = minimal_sky()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        Sky.from_dict(d)


def test_yaml_scientific_notation_pitfall_is_caught(tmp_path):
    """YAML 1.1 reads 1e-3 as a string; the sky must refuse it instead of training with lr='1e-3'."""
    path = tmp_path / "sky.yaml"
    path.write_text("version: t\nregions:\n  - name: r\n    task: toy\n    train: {lr: 1e-3}\n    knobs: {}\n")
    with pytest.raises(ValueError, match="train.lr"):
        Sky.from_yaml(path)


# ---------------------------------------------------------------------------------------------- validation
def test_validate_point_accepts_grid_and_in_bounds_values(sky):
    point = sky.points("parity-wide")[0]
    assert sky.validate_point("parity-wide", point) == []
    assert sky.validate_point("parity-wide", {**point, "n_train": None}) == []  # YAML null is a legal value
    assert sky.validate_point("parity-wide", {**point, "width": 64}) == []  # refinable: any value within bounds
    assert sky.validate_point("parity-wide", {**point, "width": 64.0}) == []  # integral float for an int knob
    assert sky.validate_point("parity-wide", {**point, "train.lr": 0.0123}) == []
    assert sky.validate_point("parity-wide", {**point, "n": 16.0}) == []  # 16.0 is the grid value 16


@pytest.mark.parametrize("change, fragment", [
    ({"bogus": 1}, "unknown knob 'bogus'"),
    ({"n_train": 2000}, "knob 'n_train': 2000 is not a grid value"),
    ({"n_train": "null"}, "is not a grid value"),
    ({"n": 20}, "knob 'n': 20 is not a grid value"),  # ordinal, not refinable
    ({"k": True}, "is not a grid value"),  # True is not 1 or 2 or 3
    ({"train.weight_decay": 0.001}, "is not a grid value"),
    ({"width": 1024}, "knob 'width': 1024 is outside the bounds [32, 512]"),
    ({"width": 16}, "outside the bounds"),
    ({"width": 64.5}, "is not an integer"),
    ({"width": "64"}, "is not a finite number"),
    ({"train.lr": float("nan")}, "is not a finite number"),
    ({"train.lr": 0.2}, "outside the bounds"),
])
def test_validate_point_errors(sky, change, fragment):
    point = {**sky.points("parity-wide")[0], **change}
    errors = sky.validate_point("parity-wide", point)
    assert len(errors) == 1 and fragment in errors[0], errors


def test_validate_point_missing_knob_unknown_region_and_non_mapping(sky):
    point = sky.points("parity-wide")[0]
    del point["k"]
    assert sky.validate_point("parity-wide", point) == ["missing knob 'k'"]
    assert "unknown region" in sky.validate_point("nowhere", {})[0]
    assert "mapping" in sky.validate_point("parity-wide", [1, 2])[0]


# ---------------------------------------------------------------------------------------------- specs and identity
def test_make_spec_maps_fixed_task_and_train_knobs(sky):
    point = {"p": 47, "train_frac": 0.3, "width": 256, "train.lr": 0.003, "train.weight_decay": 1.0}
    spec = sky.make_spec("grokking", point, seed=2, device="mps")
    assert spec.task == "modular_addition" and spec.seed == 2 and spec.device == "mps"
    assert spec.sky_version == "sky-v0" and spec.region == "grokking"
    assert spec.task_cfg == {"depth": 1, "embed": "onehot", "p": 47, "train_frac": 0.3, "width": 256}
    expected = TrainConfig(steps=20000, batch_size=512, optimizer="adamw", beta2=0.98, lr=0.003, weight_decay=1.0)
    assert spec.train == expected
    # the region's fixed/train sections are not mutated by make_spec
    assert sky.region("grokking").fixed == {"depth": 1, "embed": "onehot"}
    assert "lr" not in sky.region("grokking").train


def test_object_id_is_stable_and_independent_of_knob_order(sky):
    point = {"n": 24, "k": 3, "width": 128, "n_train": None, "train.lr": 0.03, "train.weight_decay": 0.0001}
    spec = sky.make_spec("parity-wide", point, seed=1)
    reordered = dict(reversed(list(point.items())))
    assert list(reordered) != list(point)
    assert sky.make_spec("parity-wide", reordered, seed=1).object_id() == spec.object_id()
    as_floats = {**point, "width": 128.0, "n": 24.0}
    spec_f = sky.make_spec("parity-wide", as_floats, seed=1)
    assert spec_f.object_id() == spec.object_id()
    assert isinstance(spec_f.task_cfg["width"], int) and isinstance(spec_f.task_cfg["n"], int)  # model sizes stay ints
    assert sky.make_spec("parity-wide", point, seed=2).object_id() != spec.object_id()
    assert sky.make_spec("parity-wide", {**point, "width": 129}, seed=1).object_id() != spec.object_id()
    # Regression guard: object ids are the survey's primary keys and must never change for an unchanged sky, or a
    # restarted daemon would re-run everything. The value equals the id of the hand-built spec below.
    hand = ObjectSpec(task="sparse_parity", task_cfg={"depth": 1, "act": "relu", "test_size": 2048, "n": 24, "k": 3,
                                                      "width": 128, "n_train": None}, seed=1,
                      train=TrainConfig(steps=30000, batch_size=32, optimizer="sgd", momentum=0.0, lr=0.03,
                                        weight_decay=0.0001), sky_version="sky-v0")
    assert spec.object_id() == hand.object_id() == "d80053ea9b91d947"


def test_make_spec_rejects_invalid_points_and_seeds(sky):
    point = sky.points("control-parity")[0]
    with pytest.raises(ValueError, match="invalid point"):
        sky.make_spec("control-parity", {**point, "k": 7}, seed=0)
    for seed in (-1, True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            sky.make_spec("control-parity", point, seed=seed)


def test_point_of_inverts_make_spec(sky):
    for region in sky.regions:
        for point in sky.points(region):
            spec = sky.make_spec(region, point, seed=0)
            assert point_key(sky.point_of(region, spec)) == point_key(point)
    refined = {**sky.points("parity-wide")[5], "width": 64, "train.lr": 0.0173205}
    spec = ObjectSpec.from_json(sky.make_spec("parity-wide", refined, seed=4).to_json())  # through the catalog's JSON
    assert sky.point_of("parity-wide", spec) == refined


def test_point_of_rejects_foreign_specs(sky):
    spec = sky.make_spec("grokking", sky.points("grokking")[0], seed=0)
    with pytest.raises(ValueError, match="task"):
        sky.point_of("parity-wide", spec)
    spec.task_cfg.pop("p")
    with pytest.raises(ValueError, match="task_cfg"):
        sky.point_of("grokking", spec)


def test_point_key_is_canonical():
    assert point_key({"b": 1, "a": None}) == point_key({"a": None, "b": 1.0}) == '{"a":null,"b":1}'
    assert point_key({"a": True}) != point_key({"a": 1})
    assert point_key({"a": np.int64(3), "b": np.float64(0.5)}) == '{"a":3,"b":0.5}'
    assert json.loads(point_key({"train.lr": 0.0003})) == {"train.lr": 0.0003}
    with pytest.raises(ValueError):
        point_key({"a": float("nan")})


# ---------------------------------------------------------------------------------------------- lattice and coords
def test_neighbors_are_one_grid_step_along_one_knob(sky):
    region = sky.region("parity-wide")
    grid = sky.points(region)
    keys = {point_key(p) for p in grid}
    nbrs = {point_key(p): {point_key(q) for q in sky.neighbors(region, p)} for p in grid}
    for p in grid:
        for q in sky.neighbors(region, p):
            diff = [k for k in p if point_key({k: p[k]}) != point_key({k: q[k]})]
            assert len(diff) == 1
            knob = region.knob(diff[0])
            assert abs(knob.index(p[diff[0]]) - knob.index(q[diff[0]])) == 1
            assert point_key(q) in keys
            assert point_key(p) in nbrs[point_key(q)]  # symmetric
    corner = grid[0]  # first value of every knob: one neighbour per knob
    assert len(sky.neighbors(region, corner)) == len(region.knobs)
    interior = {"n": 24, "k": 2, "width": 128, "n_train": 4000, "train.lr": 0.03, "train.weight_decay": 0.0001}
    assert len(sky.neighbors(region, interior)) == 11  # k has only two values, n_train has 4: 2+1+2+2+2+2


def test_neighbors_of_an_off_grid_point_are_the_bracketing_grid_values(sky):
    point = {**sky.points("parity-wide")[0], "width": 64}
    along_width = [q["width"] for q in sky.neighbors("parity-wide", point)
                   if point_key({**q, "width": 0}) == point_key({**point, "width": 0})]
    assert along_width == [32, 128]
    with pytest.raises(ValueError):
        sky.neighbors("parity-wide", {**point, "width": 4096})


def test_coords(sky):
    point = {"n": 24, "k": 3, "width": 128, "n_train": None, "train.lr": 0.1, "train.weight_decay": 0.0}
    c = sky.coords("parity-wide", point)
    assert c.dtype == np.float64 and c.shape == (6,)
    np.testing.assert_allclose(c, [0.5, 1.0, 0.5, 1.0, 1.0, 0.0])  # log width: 128 is the log-midpoint of 32..512
    c_mid = sky.coords("parity-wide", {**point, "width": 64})
    assert c_mid[2] == pytest.approx(0.25)
    c_lin = sky.coords("grokking", {"p": 23, "train_frac": 0.4, "width": 64, "train.lr": 0.001,
                                    "train.weight_decay": 0.1})
    assert c_lin[1] == pytest.approx(0.25) and c_lin[0] == 0.0
    single = Sky.from_dict(minimal_sky(knobs={"k": {"values": [5], "kind": "linear"}}))
    assert single.coords("r", {"k": 5}).tolist() == [0.0]


# ---------------------------------------------------------------------------------------------- knob midpoints
def test_knob_midpoints():
    lr = Knob("train.lr", [0.0003, 0.001, 0.003], kind="log", refine=True)
    mid = lr.midpoint(0.0003, 0.001)
    assert mid == float(f"{math.sqrt(0.0003 * 0.001):.{MIDPOINT_SIG_DIGITS}g}") == 0.000547723
    assert lr.midpoint(0.001, 0.0003) == mid  # symmetric
    width = Knob("width", [32, 128, 512], kind="log", refine=True)
    assert width.midpoint(32, 128) == 64 and isinstance(width.midpoint(32, 128), int)
    assert width.midpoint(32, 45) == 38  # sqrt(1440) = 37.95 -> 38
    assert width.midpoint(32, 33) is None  # no integer strictly in between
    frac = Knob("train_frac", [0.3, 0.5, 0.7], kind="linear", refine=True)
    assert frac.midpoint(0.3, 0.5) == 0.4
    steps = Knob("n", [10, 20], kind="linear", refine=True)
    assert steps.midpoint(10, 13) == 12  # 11.5 rounds half up
    assert Knob("x", [1.0, 2.0], kind="linear", refine=True).midpoint(1.0, 1.0 + 1e-9) is None  # below 6 digits
    with pytest.raises(ValueError, match="not refinable"):
        Knob("k", [2, 3]).midpoint(2, 3)
    # integer grid values of a float TrainConfig field do not make an integer knob
    wd = Knob("train.weight_decay", [0, 1], kind="linear", refine=True)
    assert not wd.is_integer and wd.midpoint(0, 1) == 0.5
    assert Knob("train.steps", [100, 300], kind="log", refine=True).midpoint(100, 300) == 173


def test_explicit_bounds_widen_the_refinable_range():
    region = Region(name="r", task="toy", knobs=[Knob("width", [32, 128], kind="log", refine=True, bounds=(16, 256))],
                    fixed={}, train={})
    sky = Sky(version="t", regions=[region])
    assert sky.validate_point("r", {"width": 16}) == []
    assert sky.validate_point("r", {"width": 8}) != []
    assert sky.coords("r", {"width": 64})[0] == pytest.approx(0.5)


def test_seed_base_and_data_seed_from_seed():
    from tycho.survey.sky import Sky
    d = {"version": "t", "regions": [
        {"name": "m", "task": "markov_lang", "seeds": 2, "seed_base": 100, "data_seed_from_seed": True,
         "fixed": {"window": 6}, "knobs": {"couplings": ["1:1.0"]}},
        {"name": "plain", "task": "markov_lang", "seeds": 2, "fixed": {"window": 6},
         "knobs": {"couplings": ["1:1.0"]}}]}
    sky = Sky.from_dict(d)
    a = sky.make_spec("m", {"couplings": "1:1.0"}, 1)
    assert a.seed == 101 and a.task_cfg["data_seed"] == 101
    b = sky.make_spec("plain", {"couplings": "1:1.0"}, 1)
    assert b.seed == 1 and "data_seed" not in b.task_cfg
    assert Sky.from_dict(sky.to_dict()).make_spec("m", {"couplings": "1:1.0"}, 1).object_id() == a.object_id()
    for bad in ({"seed_base": -1}, {"seed_base": True}, {"data_seed_from_seed": "yes"},
                {"data_seed_from_seed": True, "fixed": {"window": 6, "data_seed": 3}}):
        r = {**d["regions"][1], **bad}
        with pytest.raises(ValueError):
            Sky.from_dict({"version": "t", "regions": [r]})
