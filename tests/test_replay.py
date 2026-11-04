"""Tests of tycho.train.replay: replays reproduce the original object exactly and densify RI where asked."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tycho.train import ObjectSpec, RIConfig, TrainConfig, replay_fidelity, replay_object, run_object
from tycho.train.replay import dense_points
from tycho.train.runner import RESUME_FILE
from tycho.train.schedule import eval_steps, subsample_log


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


SPEC = ObjectSpec(task="sparse_parity", task_cfg={"n": 8, "k": 2, "width": 16, "test_size": 256}, seed=3,
                  train=TrainConfig(steps=400, batch_size=16, optimizer="sgd", momentum=0.0, lr=0.1, n_log_evals=12,
                                    n_lin_evals=6, ntk_every=4, ntk_probe=8))
RI = RIConfig(enabled=True, n_probe=512, n_points=6)


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    return run_object(SPEC, work_dir=tmp_path_factory.mktemp("orig"), ri=RI)


def test_replay_reproduces_the_light_curve_and_densifies_ri(original, tmp_path):
    dense = (100, 160)
    rep = replay_object(SPEC, work_dir=tmp_path, ri=RIConfig(enabled=False, n_probe=512, n_points=6), dense=dense,
                        n_dense=7)
    assert rep.status == "done"
    assert replay_fidelity(original.light_curve, rep.light_curve) == 0.0
    for k in original.light_curve:
        if k != "wall_s":
            assert np.array_equal(original.light_curve[k], rep.light_curve[k], equal_nan=True), k
    steps = eval_steps(400, 12, 6)
    expected = np.union1d(subsample_log(steps, 6), np.round(np.linspace(100, 160, 7)).astype(int))
    assert np.array_equal(rep.ri_curve.steps, expected)
    assert rep.ri_curve.meta["kind"] == "replay" and rep.ri_curve.meta["dense"] == [100, 160]
    # shared checkpoints: same probe, same model, same spectrum
    a, b = original.ri_curve, rep.ri_curve
    for step in a.steps:
        i, j = list(a.steps).index(step), list(b.steps).index(step)
        for row, S in enumerate(a.subsets):
            r = b.index_of(S)
            if r is not None:
                assert a.ri_plugin[row, i] == b.ri_plugin[r, j] and a.q[row, i] == b.q[r, j]


def test_replay_without_window_and_with_window_from_the_summary(original, tmp_path):
    plain = replay_object(SPEC, work_dir=tmp_path / "plain", ri=RI)
    assert np.array_equal(plain.ri_curve.steps, original.ri_curve.steps) and plain.ri_curve.meta["dense"] is None
    # the follow-up use case: densify around the largest train-loss drop of the original observation
    drop_spec = ObjectSpec(task="sparse_parity", task_cfg={"n": 6, "k": 3, "width": 16, "test_size": 256}, seed=3,
                           train=TrainConfig(steps=800, batch_size=16, optimizer="sgd", momentum=0.0, lr=0.1,
                                             n_log_evals=20, n_lin_evals=10, ntk_every=0))
    first = run_object(drop_spec, work_dir=tmp_path / "first", ri=RI)
    s = first.summary
    assert s["n_drops"] >= 1
    lo, hi = int(s["largest_drop_lo"]), int(s["largest_drop_hi"])
    rep = replay_object(drop_spec, work_dir=tmp_path / "drop", ri=RI, dense=(lo, hi), n_dense=12)
    assert replay_fidelity(first.light_curve, rep.light_curve) == 0.0
    inside = [t for t in rep.ri_curve.steps if lo <= t <= hi]
    assert len(inside) >= 12 and rep.ri_curve.meta["dense"] == [lo, hi]
    assert set(first.ri_curve.steps) <= set(rep.ri_curve.steps)


def test_interrupted_replay_resumes_exactly(tmp_path):
    full = replay_object(SPEC, work_dir=tmp_path / "full", ri=RI, dense=(50, 90), n_dense=5)
    progress = []
    part = replay_object(SPEC, work_dir=tmp_path / "part", ri=RI, dense=(50, 90), n_dense=5,
                         on_progress=lambda s, t: progress.append(s), stop_flag=lambda: len(progress) >= 6)
    assert part.status == "interrupted" and (tmp_path / "part" / RESUME_FILE).exists()
    done = replay_object(SPEC, work_dir=tmp_path / "part", ri=RI, dense=(50, 90), n_dense=5)
    assert replay_fidelity(full.light_curve, done.light_curve, keys=("train_loss", "test_loss", "ntk_trace")) == 0.0
    assert np.array_equal(full.ri_curve.ri, done.ri_curve.ri) and full.ri_curve.subsets == done.ri_curve.subsets


def test_dense_points():
    assert dense_points(10, 20, 6, 100).tolist() == [10, 12, 14, 16, 18, 20]
    assert dense_points(20.4, 9.6, 3, 100).tolist() == [10, 15, 20]  # order-insensitive, rounded
    assert dense_points(90, 150, 4, 100).tolist() == [90, 93, 97, 100]  # clipped to the run
    assert dense_points(10, 12, 10, 100).tolist() == [10, 11, 12]  # unique
    assert dense_points(150, 200, 5, 100).size == 0 and dense_points(1, 5, 0, 100).size == 0


def test_replay_fidelity():
    a = {"step": np.array([0, 10, 20]), "train_loss": np.array([1.0, 0.5, 0.25]),
         "test_loss": np.array([1.0, np.nan, 0.3])}
    assert replay_fidelity(a, a) == 0.0
    b = {k: v.copy() for k, v in a.items()}
    b["train_loss"][1] = 0.55
    assert replay_fidelity(a, b) == pytest.approx(0.05 / 0.55)
    # only common steps are compared, in step order
    c = {"step": np.array([20, 5, 10]), "train_loss": np.array([0.25, 9.0, 0.5]), "test_loss": np.array([0.3, 1, np.nan])}
    assert replay_fidelity(a, c) == 0.0
    d = {k: v.copy() for k, v in a.items()}
    d["test_loss"][1] = 0.4  # NaN in one curve only
    assert replay_fidelity(a, d) == math.inf
    e = {k: v.copy() for k, v in a.items()}
    e["train_loss"][2] = 0.0
    a0 = {k: v.copy() for k, v in a.items()}
    a0["train_loss"][2] = 0.0
    assert replay_fidelity(a0, e) == 0.0  # exact zeros agree
    inf_a = {k: v.copy() for k, v in a.items()}
    inf_a["train_loss"][2] = np.inf
    assert replay_fidelity(inf_a, inf_a) == 0.0 and replay_fidelity(a, inf_a) == math.inf
    assert replay_fidelity(a, {"step": np.array([1, 2]), "train_loss": np.ones(2), "test_loss": np.ones(2)}) == math.inf
    with pytest.raises(ValueError):
        replay_fidelity(a, {"step": a["step"], "train_loss": a["train_loss"]})
