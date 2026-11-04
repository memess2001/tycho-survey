"""Tests of tycho.train.runner (and device / schedule): determinism, non-interference of RI probing, exact resume,
divergence and failure handling, optimizer parity with torch.optim, and the positive controls (slow)."""
from __future__ import annotations

import logging
import math
import shutil

import numpy as np
import pytest
import torch

import tycho.train.runner as runner_mod
from tycho.analysis.events import detect_events
from tycho.ri.estimators import ri_scan
from tycho.tasks import make_task
from tycho.train import ObjectSpec, RIConfig, TrainConfig, run_object
from tycho.train.device import configure_torch, pick_device
from tycho.train.photometry import LIGHT_CURVE_KEYS
from tycho.train.runner import RESUME_FILE, LeanOptimizer
from tycho.train.schedule import eval_steps, lr_at, subsample_log


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def parity_spec(steps: int = 300, seed: int = 0, **train) -> ObjectSpec:
    kw = dict(steps=steps, batch_size=16, optimizer="sgd", momentum=0.9, lr=0.05, n_log_evals=12, n_lin_evals=6,
              ntk_every=3, ntk_probe=8)
    kw.update(train)
    return ObjectSpec(task="sparse_parity", task_cfg={"n": 8, "k": 2, "width": 16, "test_size": 256}, seed=seed,
                      train=TrainConfig(**kw))


RI = RIConfig(enabled=True, n_probe=512, n_points=6)


def same_lc(a: dict, b: dict, *, skip=("wall_s",)) -> bool:
    """Bit-for-bit equality of two light curves (NaN == NaN), ignoring ``skip`` keys."""
    if set(a) != set(b):
        return False
    return all(np.array_equal(a[k], b[k], equal_nan=True) for k in a if k not in skip)


def same_curve(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return (np.array_equal(a.steps, b.steps) and a.subsets == b.subsets and a.names == b.names
            and all(np.array_equal(getattr(a, k), getattr(b, k), equal_nan=True)
                    for k in ("ri", "ri_plugin", "p", "q", "n_eff")) and a.meta == b.meta)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """One uninterrupted run with in-situ RI (shared by the determinism and resume tests)."""
    return run_object(parity_spec(), work_dir=tmp_path_factory.mktemp("ref"), ri=RI)


# ---------------------------------------------------------------------------------------------- basic contract
def test_run_result_contract(reference, tmp_path):
    res = reference
    assert res.status == "done" and res.error is None
    assert res.object_id == parity_spec().object_id() and res.device == "cpu" and res.n_params == 8 * 16 + 16 + 34
    lc = res.light_curve
    assert list(lc)[: len(LIGHT_CURVE_KEYS)] == list(LIGHT_CURVE_KEYS) and "x_support_frac" in lc
    assert lc["step"].dtype == np.int64 and all(lc[k].dtype == np.float64 for k in lc if k != "step")
    assert np.array_equal(lc["step"], eval_steps(300, 12, 6))
    assert math.isnan(lc["grad_norm"][0]) and np.all(np.isfinite(lc["grad_norm"][1:]))
    assert np.all(np.diff(lc["wall_s"]) >= 0) and res.wall_time_s >= lc["wall_s"][-1]
    ntk_rows = np.isfinite(lc["ntk_trace"])
    assert np.array_equal(np.flatnonzero(ntk_rows), np.arange(0, lc["step"].size, 3))  # every 3rd evaluation
    assert res.summary["steps_total"] == 300 and res.summary["diverged"] is False
    curve = res.ri_curve
    assert np.array_equal(curve.steps, subsample_log(eval_steps(300, 12, 6), 6))
    assert curve.meta["kind"] == "insitu" and curve.meta["max_order"] == 3 and curve.meta["object_id"] == res.object_id
    assert curve.ground_truth == make_task("sparse_parity", {"n": 8, "k": 2}, 0).ground_truth()


def test_progress_callback_and_checkpoint_cleanup(tmp_path):
    calls = []
    res = run_object(parity_spec(steps=40), work_dir=tmp_path, ri=RI, on_progress=lambda s, t: calls.append((s, t)))
    assert res.status == "done"
    expected = np.union1d(eval_steps(40, 12, 6), subsample_log(eval_steps(40, 12, 6), 6))
    assert calls == [(int(s), 40) for s in expected]
    assert not (tmp_path / RESUME_FILE).exists()


# ---------------------------------------------------------------------------------------------- determinism
def test_same_spec_gives_bitwise_identical_curves(reference, tmp_path):
    again = run_object(parity_spec(), work_dir=tmp_path, ri=RI)
    assert same_lc(reference.light_curve, again.light_curve)
    assert same_curve(reference.ri_curve, again.ri_curve)
    other_seed = run_object(parity_spec(seed=1), work_dir=tmp_path / "s1")
    assert not np.array_equal(other_seed.light_curve["train_loss"], reference.light_curve["train_loss"])


def test_ri_probing_never_changes_the_training_trajectory(reference, tmp_path):
    no_ri = run_object(parity_spec(), work_dir=tmp_path / "off")
    assert no_ri.ri_curve is None
    assert same_lc(reference.light_curve, no_ri.light_curve)
    # RI at steps that are not evaluation steps: the light curve still does not move, and the probes at shared
    # steps give the same spectrum (the probe depends on (seed, step) only)
    odd = run_object(parity_spec(), work_dir=tmp_path / "odd", ri=RI, ri_steps=[0, 3, 17, 250, 300])
    assert same_lc(reference.light_curve, odd.light_curve)
    assert list(odd.ri_curve.steps) == [0, 3, 17, 250, 300]
    ref_c = reference.ri_curve
    for step in (0, 300):
        i, j = list(ref_c.steps).index(step), list(odd.ri_curve.steps).index(step)
        for row, S in enumerate(odd.ri_curve.subsets):
            r = ref_c.index_of(S)
            if r is not None:
                assert odd.ri_curve.ri_plugin[row, j] == ref_c.ri_plugin[r, i]


# ---------------------------------------------------------------------------------------------- resume
def test_resume_after_stop_flag_reproduces_the_uninterrupted_run(reference, tmp_path):
    progress = []
    first = run_object(parity_spec(), work_dir=tmp_path, ri=RI, on_progress=lambda s, t: progress.append(s),
                       stop_flag=lambda: len(progress) >= 5)
    assert first.status == "interrupted" and (tmp_path / RESUME_FILE).exists()
    assert first.light_curve["step"].size <= 5 and first.ri_curve is not None
    second = run_object(parity_spec(), work_dir=tmp_path, ri=RI)
    assert second.status == "done" and not (tmp_path / RESUME_FILE).exists()
    assert same_lc(reference.light_curve, second.light_curve)
    assert same_curve(reference.ri_curve, second.ri_curve)
    assert second.wall_time_s >= first.wall_time_s


def test_resume_mid_segment_twice(reference, tmp_path, monkeypatch):
    """Interruptions between checkpoint steps (polling every update), resumed twice."""
    monkeypatch.setattr(runner_mod, "POLL_STEPS", 1)
    for stop_after in (37, 120):
        calls = [0]

        def stop() -> bool:
            calls[0] += 1
            return calls[0] >= stop_after

        res = run_object(parity_spec(), work_dir=tmp_path, ri=RI, stop_flag=stop)
        assert res.status == "interrupted"
    final = run_object(parity_spec(), work_dir=tmp_path, ri=RI)
    assert final.status == "done"
    assert same_lc(reference.light_curve, final.light_curve)
    assert same_curve(reference.ri_curve, final.ri_curve)


def test_crash_recovery_from_periodic_checkpoint(reference, tmp_path, monkeypatch):
    """A crash (exception) after periodic checkpoints: the next call resumes from the last one."""
    monkeypatch.setattr(runner_mod, "POLL_STEPS", 7)
    spec = parity_spec(checkpoint_every_s=0.0)  # checkpoint at every poll
    calls = [0]

    def crash() -> bool:
        calls[0] += 1
        if calls[0] == 25:
            raise RuntimeError("simulated crash")
        return False

    res = run_object(spec, work_dir=tmp_path, ri=RI, stop_flag=crash)
    assert res.status == "failed" and "simulated crash" in res.error
    assert (tmp_path / RESUME_FILE).exists()
    final = run_object(spec, work_dir=tmp_path, ri=RI)
    assert final.status == "done"
    ref = run_object(spec, work_dir=tmp_path / "fresh", ri=RI)
    assert same_lc(ref.light_curve, final.light_curve) and same_curve(ref.ri_curve, final.ri_curve)
    assert same_lc(reference.light_curve, final.light_curve)  # checkpoint_every_s is part of the identity only


def test_mismatched_checkpoint_is_ignored(reference, tmp_path, caplog):
    progress = []
    run_object(parity_spec(), work_dir=tmp_path, ri=RIConfig(enabled=True, n_probe=256, n_points=6),
               on_progress=lambda s, t: progress.append(s), stop_flag=lambda: len(progress) >= 3)
    assert (tmp_path / RESUME_FILE).exists()
    with caplog.at_level(logging.WARNING, logger="tycho.train.runner"):
        res = run_object(parity_spec(), work_dir=tmp_path, ri=RI)
    assert "starting from scratch" in caplog.text
    assert same_lc(reference.light_curve, res.light_curve) and same_curve(reference.ri_curve, res.ri_curve)
    (tmp_path / RESUME_FILE).write_bytes(b"not a checkpoint")
    with caplog.at_level(logging.WARNING, logger="tycho.train.runner"):
        res2 = run_object(parity_spec(), work_dir=tmp_path, ri=RI)
    assert res2.status == "done" and same_lc(reference.light_curve, res2.light_curve)


def test_unloadable_checkpoint_is_quarantined(reference, tmp_path, caplog):
    """A matching checkpoint whose model state no longer fits (e.g. after a code change) is moved aside."""
    progress = []
    run_object(parity_spec(), work_dir=tmp_path, ri=RI, on_progress=lambda s, t: progress.append(s),
               stop_flag=lambda: len(progress) >= 3)
    state = torch.load(tmp_path / RESUME_FILE, weights_only=True)
    state["model"]["net.0.weight"] = torch.zeros(3)
    torch.save(state, tmp_path / RESUME_FILE)
    with caplog.at_level(logging.WARNING, logger="tycho.train.runner"):
        res = run_object(parity_spec(), work_dir=tmp_path, ri=RI)
    assert "could not restore" in caplog.text and (tmp_path / (RESUME_FILE + ".bad")).exists()
    assert res.status == "done" and same_lc(reference.light_curve, res.light_curve)
    assert same_curve(reference.ri_curve, res.ri_curve)


def test_resume_false_restarts_from_step_zero(reference, tmp_path):
    progress = []
    run_object(parity_spec(), work_dir=tmp_path, ri=RI, on_progress=lambda s, t: progress.append(s),
               stop_flag=lambda: len(progress) >= 4)
    done_steps = list(progress)
    shutil.copy(tmp_path / RESUME_FILE, tmp_path / "saved.pt")
    fresh = []
    res = run_object(parity_spec(), work_dir=tmp_path, ri=RI, resume=False, on_progress=lambda s, t: fresh.append(s))
    assert fresh[0] == 0 and len(fresh) == len(np.union1d(reference.light_curve["step"], reference.ri_curve.steps))
    assert same_lc(reference.light_curve, res.light_curve)
    shutil.copy(tmp_path / "saved.pt", tmp_path / RESUME_FILE)
    resumed = []
    run_object(parity_spec(), work_dir=tmp_path, ri=RI, on_progress=lambda s, t: resumed.append(s))
    assert resumed[0] > done_steps[-1] and done_steps + resumed == fresh


# ---------------------------------------------------------------------------------------------- photometry values
def test_update_norm_identity_and_clipping(tmp_path):
    lr = 0.05
    plain = run_object(parity_spec(steps=60, momentum=0.0, lr=lr), work_dir=tmp_path / "a").light_curve
    # plain SGD: θ_t − θ_{t−1} = −lr g, so ‖Δθ‖ = lr ‖g‖
    assert np.allclose(plain["update_norm"][1:], lr * plain["grad_norm"][1:], rtol=1e-5)
    clip = 1e-3
    clipped = run_object(parity_spec(steps=60, momentum=0.0, lr=lr, grad_clip=clip), work_dir=tmp_path / "b")
    lc = clipped.light_curve
    # the update is measured as a float32 parameter difference: allow its rounding (~1e-7 · ‖θ‖ per entry)
    assert np.all(lc["update_norm"][1:] <= lr * clip * (1 + 1e-3)) and np.all(lc["grad_norm"][1:] > clip)
    assert np.allclose(lc["update_norm"][1:], lr * clip, rtol=1e-3)


def test_lr_schedule_in_light_curve(tmp_path):
    spec = parity_spec(steps=100, warmup_steps=10, lr_schedule="cosine")
    lc = run_object(spec, work_dir=tmp_path).light_curve
    assert np.allclose(lc["lr"], [lr_at(int(s), spec.train) for s in lc["step"]])
    assert lc["lr"][0] == pytest.approx(0.005) and lc["lr"][-1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------------------------- failure modes
@pytest.mark.parametrize("task,cfg,lr", [("toy_superposition", {"n_features": 6, "m_hidden": 2}, 50.0),
                                         ("sparse_parity", {"n": 8, "k": 2, "width": 16}, 1e30)])
def test_divergence_stops_early_with_done_status(task, cfg, lr, tmp_path):
    spec = ObjectSpec(task=task, task_cfg=cfg, seed=0,
                      train=TrainConfig(steps=200, batch_size=16, optimizer="sgd", momentum=0.0, lr=lr,
                                        n_log_evals=10, n_lin_evals=5, ntk_every=1, ntk_probe=4))
    res = run_object(spec, work_dir=tmp_path, ri=RIConfig(enabled=True, n_probe=128, n_points=8))
    assert res.status == "done" and res.summary["diverged"] is True
    lc = res.light_curve
    assert lc["step"][-1] < 200 and not np.isfinite(lc["train_loss"][-1])
    assert np.all(np.diff(lc["step"]) > 0)
    assert res.ri_curve is None or res.ri_curve.steps[-1] <= lc["step"][-1]
    assert not (tmp_path / RESUME_FILE).exists()


def test_exceptions_give_failed_status(tmp_path):
    bad_opt = run_object(parity_spec(optimizer="rmsprop"), work_dir=tmp_path / "a")
    assert bad_opt.status == "failed" and "optimizer must be one of" in bad_opt.error
    bad_task = run_object(ObjectSpec(task="nope", task_cfg={}, seed=0), work_dir=tmp_path / "b")
    assert bad_task.status == "failed" and "unknown task" in bad_task.error
    bad_steps = run_object(parity_spec(steps=20), work_dir=tmp_path / "c", ri=RI, ri_steps=[5, 25])
    assert bad_steps.status == "failed" and "ri_steps" in bad_steps.error


# ---------------------------------------------------------------------------------------------- optimizer
@pytest.mark.parametrize("cfg", [
    dict(optimizer="sgd", momentum=0.0, weight_decay=0.0), dict(optimizer="sgd", momentum=0.9, weight_decay=0.01),
    dict(optimizer="adam", weight_decay=0.1, beta2=0.98), dict(optimizer="adamw", weight_decay=0.5, beta1=0.8)])
def test_lean_optimizer_matches_torch_optim_bitwise(cfg):
    train = TrainConfig(**cfg)
    torch.manual_seed(0)
    base = [torch.randn(5, 3), torch.randn(3), torch.randn(2, 5)]
    lean_params = [p.clone() for p in base]
    ref_params = [p.clone().requires_grad_() for p in base]
    lean = LeanOptimizer(lean_params, train)
    kw = dict(lr=0.01, weight_decay=train.weight_decay, foreach=True)
    if train.optimizer == "sgd":
        ref = torch.optim.SGD(ref_params, momentum=train.momentum, **kw)
    else:
        cls = torch.optim.Adam if train.optimizer == "adam" else torch.optim.AdamW
        ref = cls(ref_params, betas=(train.beta1, train.beta2), **kw)
    for i in range(25):
        lr = 0.01 * (1 + i % 3)
        grads = [torch.randn_like(p) for p in base]
        lean.step([g.clone() for g in grads], lr)
        for group in ref.param_groups:
            group["lr"] = lr
        for p, g in zip(ref_params, grads):
            p.grad = g.clone()
        ref.step()
    assert all(torch.equal(a, b.detach()) for a, b in zip(lean_params, ref_params))
    clone = LeanOptimizer([p.clone() for p in lean_params], train)
    clone.load_state_dict(lean.state_dict())
    assert clone.t == lean.t == (25 if train.optimizer != "sgd" else 25)


# ---------------------------------------------------------------------------------------------- schedule / device
def test_eval_steps_and_subsample_log():
    st = eval_steps(1000, 60, 40)
    assert st[0] == 0 and st[-1] == 1000 and np.all(np.diff(st) > 0)
    lin = set(np.round(np.linspace(0, 1000, 40)).astype(int).tolist())
    log = set(np.round(np.logspace(0, 3, 60)).astype(int).tolist())
    assert set(st.tolist()) == lin | log | {0, 1000} and {1, 2, 3} <= log
    assert eval_steps(0, 60, 40).tolist() == [0]
    assert eval_steps(10, 0, 0).tolist() == [0, 10]
    sub = subsample_log(st, 10)
    assert sub.size == 10 and sub[0] == 0 and sub[-1] == 1000 and set(sub) <= set(st)
    gaps = np.diff(np.log1p(sub[1:]))  # roughly uniform in log step away from the dense start
    assert gaps.max() < 3 * gaps.min() + 0.2
    assert np.array_equal(subsample_log(st, 10_000), st)
    assert subsample_log([5, 1, 3], 1).tolist() == [5] and subsample_log([5, 1, 3], 0).size == 0
    with pytest.raises(ValueError):
        eval_steps(-1, 3, 3)


def test_lr_at_schedules():
    c = TrainConfig(lr=0.1, steps=100)
    assert lr_at(0, c) == lr_at(99, c) == 0.1
    w = TrainConfig(lr=0.1, steps=100, warmup_steps=4, lr_schedule="cosine")
    assert [lr_at(s, w) for s in range(4)] == pytest.approx([0.025, 0.05, 0.075, 0.1])
    assert lr_at(4, w) == pytest.approx(0.1) and lr_at(52, w) == pytest.approx(0.05) and lr_at(100, w) == 0.0
    with pytest.raises(ValueError):
        lr_at(0, TrainConfig(lr_schedule="step"))


def test_pick_device_and_configure_torch():
    assert pick_device("cpu", 10**9).type == "cpu"
    assert pick_device("auto", 10).type == "cpu"
    if not torch.backends.mps.is_available():
        assert pick_device("mps", 10).type == "cpu" and pick_device("AUTO", 10**9).type == "cpu"
    with pytest.raises(ValueError):
        pick_device("cuda", 10)
    before = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        configure_torch(2)
        assert torch.get_num_threads() == 2 and torch.are_deterministic_algorithms_enabled()
    finally:
        torch.set_num_threads(before)
        torch.use_deterministic_algorithms(deterministic)


# ---------------------------------------------------------------------------------------------- positive controls
@pytest.mark.slow
def test_positive_control_sparse_parity_learns_and_rings_at_the_ground_truth(tmp_path):
    spec = ObjectSpec(task="sparse_parity", task_cfg={"n": 10, "k": 2, "width": 128, "data_seed": 1}, seed=0,
                      train=TrainConfig(steps=3000, batch_size=32, optimizer="sgd", momentum=0.0, lr=0.1,
                                        n_log_evals=40, n_lin_evals=20, ntk_every=5))
    res = run_object(spec, work_dir=tmp_path, ri=RIConfig(enabled=True, n_probe=4096, n_points=15))
    assert res.status == "done" and res.summary["final_test_acc"] > 0.97
    curve = res.ri_curve
    gt = curve.index_of(curve.ground_truth[0]["subset"])
    assert gt is not None
    det = curve.detected()
    assert det[gt, -3:].all()  # detected over the last checkpoints
    assert curve.ri[gt, -1] > 0.5 and curve.ri[gt, -1] == curve.ri[:, -1].max()
    events = detect_events(curve)
    assert any(e.matches_ground_truth and e.kind == "activation" for e in events)


@pytest.mark.slow
def test_positive_control_markov_language_orders_and_true_spectrum(tmp_path):
    cfg = {"couplings": "1:1.0;1,3:0.8", "n_tokens": 100_000}
    spec = ObjectSpec(task="markov_lang", task_cfg=cfg, seed=0,
                      train=TrainConfig(steps=3000, batch_size=64, optimizer="adam", lr=1e-3, n_log_evals=40,
                                        n_lin_evals=20, ntk_every=5))
    res = run_object(spec, work_dir=tmp_path, ri=RIConfig(enabled=True, n_probe=8192, n_points=20))
    assert res.status == "done" and res.light_curve["x_kl_true"][-1] < 0.01
    curve = res.ri_curve
    pair = curve.index_of(["x[t-1]", "xhat"])
    triple = curve.index_of(["x[t-1]", "x[t-3]", "xhat"])
    det = curve.detected()
    assert det[pair, -1] and det[triple, -1]  # both planted terms are present at the end
    onset = {}
    for e in detect_events(curve):
        if e.kind == "activation":
            onset[e.subset_names] = e.step
    assert onset[curve.subset_names(pair)] <= onset[curve.subset_names(triple)]  # pairwise no later than 3-way
    # the final model spectrum matches the true spectrum (same probe distribution, same subsets)
    task = make_task("markov_lang", cfg, 0)
    true_scan = ri_scan(task.true_variables(8192, torch.Generator().manual_seed(0)), max_order=3)
    truth = {tuple(true_scan.names[i] for i in S): e.ri_corrected for S, e in zip(true_scan.subsets,
                                                                                   true_scan.estimates)}
    model_ri = np.asarray([curve.ri[i, -1] for i in range(curve.S)])
    true_ri = np.asarray([truth[curve.subset_names(i)] for i in range(curve.S)])
    assert np.corrcoef(model_ri, true_ri)[0, 1] > 0.95


_CHILD = """
import sys, numpy as np
from tycho.train import run_object, ObjectSpec, TrainConfig, RIConfig, configure_torch
configure_torch(1)
spec = ObjectSpec.from_json(sys.argv[2])
r = run_object(spec, work_dir=sys.argv[3], ri=RIConfig(enabled=True, n_probe=256, n_points=3))
assert r.status == "done", r.error
np.savez(sys.argv[1], ri=r.ri_curve.ri, **{k: v for k, v in r.light_curve.items() if k != "wall_s"})
"""


@pytest.mark.slow
def test_bitwise_determinism_across_processes(tmp_path):
    """Two processes (different hash seeds) observing the same spec produce identical curves."""
    import os
    import subprocess
    import sys

    spec = ObjectSpec(task="icl_markov", task_cfg={"d_model": 8, "seq_len": 16, "eval_size": 16}, seed=2,
                      train=TrainConfig(steps=60, batch_size=8, n_log_evals=6, n_lin_evals=3, ntk_every=2,
                                        ntk_probe=4))
    outs = []
    for i, hashseed in enumerate(("1", "2")):
        out = tmp_path / f"run{i}.npz"
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        subprocess.run([sys.executable, "-c", _CHILD, str(out), spec.to_json(), str(tmp_path / f"w{i}")], check=True,
                       env=env, timeout=300)
        outs.append(np.load(out))
    a, b = outs
    assert set(a.files) == set(b.files)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
