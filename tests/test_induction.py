"""Sky v2 Part B: the induction task (random repeat positions, controls, head scores) and TinyGPT head ablation."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tycho.models.transformer import TinyGPT
from tycho.tasks import make_task
from tycho.tasks.induction import repeat_sequences
from tycho.train import ObjectSpec, RIConfig, TrainConfig, run_object


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_repeat_sequences_copy_and_mark_the_predictable_tokens():
    x, rep, src = repeat_sequences(200, 40, 50, 4, 12, 1.0, torch.Generator().manual_seed(0))
    starts = set()
    for i in range(200):
        t = torch.nonzero(rep[i]).flatten()
        assert t.numel() >= 3
        L = int(t[0] - src[i, t[0]] + 1)
        assert 4 <= L <= 12 and torch.all(t == torch.arange(int(t[0]), int(t[0]) + L - 1))
        s = int(t[0]) - L
        starts.add(s)
        assert torch.equal(x[i, s + L:s + 2 * L], x[i, s:s + L])
        assert torch.equal(x[i, t + 1], x[i, src[i, t]])  # the induction target is the token after the match
        assert torch.all(src[i, ~rep[i]] == -1)
    assert len(starts) > 10  # the repeat starts at varying positions (no positional shortcut)
    _, rep0, _ = repeat_sequences(50, 40, 50, 4, 12, 0.0, torch.Generator().manual_seed(1))
    assert not rep0.any()


def test_head_mask_and_attention_maps():
    torch.manual_seed(0)
    m = TinyGPT(20, 16, 2, 2, 12, attn_only=True)
    x = torch.randint(0, 20, (3, 12))
    with torch.no_grad():
        ref = m(x)
        with m.head_mask([1, 1, 1, 1]):
            assert torch.equal(m(x), ref)
        with m.head_mask([0, 0, 0, 0]):
            ablated = m(x)
        assert torch.equal(m(x), ref)  # restored on exit
    # with every head ablated an attention-only model sees only the current token and its position
    y = x.clone()
    y[:, :-1] = torch.randint(0, 20, (3, 11))
    with torch.no_grad(), m.head_mask([0, 0, 0, 0]):
        assert torch.allclose(m(y)[:, -1], ablated[:, -1])
    with pytest.raises(ValueError), m.head_mask([1, 0]):
        pass
    maps = m.attention_maps(x)
    assert len(maps) == 2 and maps[0].shape == (3, 2, 12, 12)
    assert torch.allclose(maps[1].sum(-1), torch.ones(3, 2, 12))
    assert torch.all(maps[0].triu(1) == 0)
    assert all(b.attn.record is None for b in m.blocks)


def test_pattern_codes_follow_the_head_order():
    t = make_task("induction", {"vocab": 16, "seq_len": 24, "rep_min": 4, "rep_max": 10, "d_model": 16,
                                "n_layer": 2, "n_head": 2, "probe_size": 8}, 0)
    m = t.build_model()
    p = t.pattern_probabilities(m)
    assert p.shape == (16,) and t.pattern_input_names() == ["l0h0", "l0h1", "l1h0", "l1h1"]
    pr = t._probe
    with torch.no_grad(), m.head_mask([0, 1, 1, 0]):  # code 0b0110 = 6: heads 1 and 2 kept
        lp = m(pr["x"]).double().log_softmax(-1).gather(-1, pr["y"][..., None])[..., 0]
    assert p[6] == pytest.approx(float(lp[pr["rep"]].exp().mean()), rel=1e-6)
    z = t.pattern_logodds(m)
    assert np.allclose(z, np.log(p / (1 - p)))
    assert t.pattern_truth() is None and np.allclose(t.pattern_pmfs()["D"], 1 / 16)
    metrics = t.extra_metrics(m)
    assert {"loss_rep", "loss_other", "icl_score", "prev_l0h0", "ind_l1h1"} <= set(metrics)
    with pytest.raises(ValueError):
        t.ri_variables(m, 4, torch.Generator())


def test_control_settings_and_validation():
    t = make_task("induction", {"p_repeat": 0.0, "seq_len": 32, "rep_max": 16}, 0)
    x, _ = t.sample_batch(64, torch.Generator().manual_seed(0))
    assert x.shape == (64, 31)
    assert t._probe["rep"].any()  # the probe set always has repeats
    for bad in ({"rep_max": 40}, {"rep_min": 1}, {"p_repeat": 1.5}):
        with pytest.raises(ValueError):
            make_task("induction", bad, 0)


def test_runner_pattern_mode_on_head_ablations(tmp_path):
    spec = ObjectSpec(task="induction", task_cfg={"vocab": 16, "seq_len": 24, "rep_min": 4, "rep_max": 10,
                                                  "d_model": 16, "n_layer": 2, "n_head": 2, "probe_size": 8},
                      seed=0, train=TrainConfig(steps=40, batch_size=8, lr=1e-3, optimizer="adam", n_log_evals=6,
                                                n_lin_evals=0, ntk_every=0))
    ri = RIConfig(enabled=True, n_probe=64, n_points=4, mode="patterns")
    res = run_object(spec, work_dir=tmp_path / "a", ri=ri)
    assert res.status == "done", res.error
    pat = res.patterns
    assert pat["logodds"].shape == (pat["steps"].size, 16) and "truth" not in pat
    assert "x_ind_l1h0" in res.light_curve
    again = run_object(spec, work_dir=tmp_path / "b", ri=ri)
    assert np.array_equal(again.patterns["logodds"], pat["logodds"])


def test_tiled_sequences_repeat_with_random_period_and_phase():
    from tycho.tasks.induction import tiled_sequences
    x, rep, src = tiled_sequences(100, 40, 50, 4, 12, 1.0, torch.Generator().manual_seed(3))
    periods = set()
    for i in range(100):
        t = torch.nonzero(rep[i]).flatten()
        L = int(t[0])
        periods.add(L)
        assert 4 <= L <= 12 and torch.equal(x[i, L:], x[i, :40 - L])  # period L
        assert torch.all(t == torch.arange(L, 39)) and torch.equal(src[i, t], t - L + 1)
    assert len(periods) > 5
    _, rep0, _ = tiled_sequences(20, 40, 50, 4, 12, 0.0, torch.Generator().manual_seed(4))
    assert not rep0.any()


def test_induction_and_duplicate_key_sets():
    t = make_task("induction", {"vocab": 8, "seq_len": 12, "rep_min": 3, "rep_max": 3, "layout": "tiled",
                                "d_model": 8, "n_layer": 2, "n_head": 1, "probe_size": 2}, 0)
    pr = t._probe
    x, ind, dup = pr["x"][0], pr["ind_keys"][0], pr["dup_keys"][0]
    for q in range(x.numel()):
        for j in range(x.numel()):
            assert bool(ind[q, j]) == (1 <= j <= q and int(x[j - 1]) == int(x[q]))
            assert bool(dup[q, j]) == (j < q and int(x[j]) == int(x[q]))
    m = t.build_model()
    metrics = t.extra_metrics(m)
    assert {"dup_l0h0", "ind_l1h0", "prev_l1h0"} <= set(metrics)
    assert math.isnan(metrics["loss_other"]) or metrics["loss_other"] > 0
    with pytest.raises(ValueError):
        make_task("induction", {"layout": "spiral"}, 0)
