"""Tests of tycho.train.photometry: exact identities of the statistics and the Photometer's bookkeeping."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from tycho.tasks import make_task
from tycho.train.photometry import (
    LIGHT_CURVE_KEYS,
    Photometer,
    effective_rank,
    flat_params,
    linear_cka,
    monitored_weight,
    ntk_gram,
    ntk_stats,
    param_norm,
)


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


# ---------------------------------------------------------------------------------------------- effective rank
def test_effective_rank_identities():
    assert abs(effective_rank(torch.eye(7)) - 7.0) < 1e-9
    assert abs(effective_rank(3.0 * torch.eye(7)) - 7.0) < 1e-9  # scale invariant
    u, v = torch.randn(9, 1), torch.randn(1, 5)
    assert abs(effective_rank(u @ v) - 1.0) < 1e-6
    assert abs(effective_rank(torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0]))) - 2.0) < 1e-9
    # two singular values in ratio 3:1 -> exp(H(3/4, 1/4))
    expected = math.exp(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))
    assert abs(effective_rank(torch.diag(torch.tensor([3.0, 1.0]))) - expected) < 1e-9
    assert effective_rank(torch.zeros(3, 4)) == 0.0
    assert math.isnan(effective_rank(torch.tensor([[1.0, float("nan")], [0.0, 1.0]])))
    with pytest.raises(ValueError):
        effective_rank(torch.ones(3))


# ---------------------------------------------------------------------------------------------- CKA
def test_linear_cka_identities():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(200, 6, generator=g)
    assert abs(linear_cka(X, X) - 1.0) < 1e-12
    Q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    assert abs(linear_cka(X, 2.5 * X.double() @ Q + 3.0) - 1.0) < 1e-9  # rotation, isotropic scale, shift
    Y = torch.randn(200, 4, generator=g)
    assert linear_cka(X, Y) < 0.15
    assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-12
    assert math.isnan(linear_cka(X, torch.ones(200, 3)))
    with pytest.raises(ValueError):
        linear_cka(X, Y[:10])


# ---------------------------------------------------------------------------------------------- eNTK
class _LinearTask:
    """f(x) = w·x + b: per-sample gradient (x, 1), so K = X Xᵀ + 1 exactly."""

    def ntk_output(self, out, y):
        return out[:, 0]


def test_ntk_gram_of_a_linear_model_is_x_xt_plus_one():
    torch.manual_seed(0)
    model = nn.Linear(5, 1)
    x = torch.randn(9, 5)
    y = torch.zeros(9, dtype=torch.long)
    expected = x.double() @ x.double().T + 1.0
    for method in ("func", "loop", "auto"):
        K = ntk_gram(model, x, y, _LinearTask(), method=method)
        assert K.dtype == torch.float64 and torch.allclose(K, expected, atol=1e-5)
    with pytest.raises(ValueError):
        ntk_gram(model, x, y, _LinearTask(), method="exact")


@pytest.mark.parametrize("name,cfg", [
    ("sparse_parity", {"n": 6, "k": 2, "width": 12}),
    ("icl_markov", {"d_model": 8, "seq_len": 14, "eval_size": 8}),
    ("markov_lang", {"model": "transformer", "d_model": 8, "n_head": 2, "n_tokens": 2000, "test_size": 64}),
    ("toy_superposition", {"n_features": 6, "m_hidden": 2}),
])
def test_ntk_gram_func_and_loop_agree_on_task_models(name, cfg):
    task = make_task(name, cfg, seed=0)
    model = task.build_model()
    x, y = task.eval_splits()["train"]
    x, y = x[:6], y[:6]
    K_func = ntk_gram(model, x, y, task, method="func")
    K_loop = ntk_gram(model, x, y, task, method="loop")
    assert torch.allclose(K_func, K_loop, rtol=1e-4, atol=1e-6 * float(K_loop.abs().max()))
    assert torch.allclose(K_func, K_func.T)
    assert float(torch.linalg.eigvalsh(K_func).min()) > -1e-6 * float(K_func.abs().max())


def test_ntk_stats_identities():
    K = torch.diag(torch.tensor([4.0, 1.0, 1.0, 0.0], dtype=torch.float64))
    s = ntk_stats(K)
    assert s["trace"] == 6.0 and s["top_eig"] == 4.0 and math.isnan(s["align"])
    p = torch.tensor([4.0, 1.0, 1.0], dtype=torch.float64) / 6.0
    assert abs(s["erank"] - float(torch.exp(-(p * p.log()).sum()))) < 1e-12
    # the ideal kernel of the centred one-hot labels has alignment 1 (any positive scale)
    y = torch.tensor([0, 2, 1, 0, 2, 2])
    Y = torch.nn.functional.one_hot(y).double()
    Yc = Y - Y.mean(0)
    assert abs(ntk_stats(7.0 * Yc @ Yc.T, y)["align"] - 1.0) < 1e-12
    assert abs(ntk_stats(Y @ Y.T + 3.0, y)["align"] - 1.0) < 1e-12  # centring removes the constant
    # binary labels: one-hot alignment equals centred ±1 alignment
    yb = torch.tensor([0, 1, 1, 0, 1])
    Kb = torch.randn(5, 3, generator=torch.Generator().manual_seed(1)).double()
    Kb = Kb @ Kb.T
    pm = (2.0 * yb - 1.0).double()
    pmc = pm - pm.mean()
    H = torch.eye(5, dtype=torch.float64) - 1.0 / 5
    Kc = H @ Kb @ H
    ref = float(pmc @ Kc @ pmc) / float(torch.linalg.matrix_norm(Kc) * (pmc @ pmc))
    assert abs(ntk_stats(Kb, yb)["align"] - ref) < 1e-12
    nan = ntk_stats(torch.full((3, 3), float("nan")), torch.tensor([0, 1, 0]))
    assert all(math.isnan(v) for v in nan.values())
    with pytest.raises(ValueError):
        ntk_stats(torch.ones(2, 3))


# ---------------------------------------------------------------------------------------------- photometer
def _parity():
    task = make_task("sparse_parity", {"n": 6, "k": 2, "width": 10, "test_size": 300}, seed=0)
    return task, task.build_model(), task.eval_splits()


def test_photometer_row_keys_and_direct_values():
    task, model, splits = _parity()
    ph = Photometer(ntk_probe=8)
    row = ph.measure(0, model, task, splits, grad_norm=float("nan"), lr=0.1, do_ntk=True, wall_s=1.5)
    assert list(row)[: len(LIGHT_CURVE_KEYS)] == list(LIGHT_CURVE_KEYS)
    assert set(row) - set(LIGHT_CURVE_KEYS) == {"x_support_frac"}
    x, y = splits["test"]
    with torch.no_grad():
        out = model(x)
    assert abs(row["test_loss"] - float(task.loss(out, y))) < 1e-6
    assert row["test_acc"] == task.accuracy(out, y)
    assert row["step"] == 0 and row["lr"] == 0.1 and row["wall_s"] == 1.5 and math.isnan(row["grad_norm"])
    assert row["dist_from_init"] == 0.0 and abs(row["param_norm"] - param_norm(model)) < 1e-12
    assert math.isnan(row["rep_cka_prev"]) and math.isnan(row["ntk_velocity"])
    assert row["erank_weights"] == effective_rank(model.first_layer().weight)
    assert monitored_weight(model) is model.first_layer().weight
    assert row["ntk_trace"] > 0 and 0 < row["ntk_erank"] <= 8 and -1 <= row["ntk_align"] <= 1
    # unchanged model: CKA 1, eNTK velocity 0
    row2 = ph.measure(5, model, task, splits, grad_norm=1.0, lr=0.1, do_ntk=True)
    assert abs(row2["rep_cka_prev"] - 1.0) < 1e-12 and abs(row2["ntk_velocity"]) < 1e-12
    # a known parameter displacement
    delta = 0.01
    with torch.no_grad():
        for p in model.parameters():
            p.add_(delta)
    n_params = flat_params(model).numel()
    row3 = ph.measure(9, model, task, splits, grad_norm=1.0, lr=0.1, do_ntk=False)
    assert abs(row3["dist_from_init"] - delta * math.sqrt(n_params)) < 1e-6
    assert math.isnan(row3["ntk_trace"]) and math.isnan(row3["ntk_velocity"])
    assert row3["rep_cka_prev"] < 1.0


def test_photometer_state_roundtrip_reproduces_rows():
    task, model, splits = _parity()
    a = Photometer(ntk_probe=4)
    a.measure(0, model, task, splits, float("nan"), 0.1, True)
    b = Photometer(ntk_probe=4)
    b.load_state_dict({k: (None if v is None else v.clone()) for k, v in a.state_dict().items()})
    with torch.no_grad():
        model.first_layer().weight.mul_(1.1)
    ra = a.measure(3, model, task, splits, 1.0, 0.1, True)
    rb = b.measure(3, model, task, splits, 1.0, 0.1, True)
    assert all((ra[k] == rb[k]) or (math.isnan(ra[k]) and math.isnan(rb[k])) for k in ra)


def test_photometer_survives_non_finite_parameters():
    task, model, splits = _parity()
    ph = Photometer(ntk_probe=4)
    ph.measure(0, model, task, splits, float("nan"), 0.1, True)
    with torch.no_grad():
        model.first_layer().weight[0, 0] = float("nan")
    row = ph.measure(1, model, task, splits, float("nan"), 0.1, True)
    for k in ("train_loss", "param_norm", "erank_weights", "ntk_trace", "ntk_velocity", "x_support_frac"):
        assert math.isnan(row[k]), k


def test_photometer_task_without_accuracy_and_sequence_targets():
    sup = make_task("toy_superposition", {"n_features": 6, "m_hidden": 2, "eval_size": 64}, seed=0)
    m = sup.build_model()
    row = Photometer(ntk_probe=4).measure(0, m, sup, sup.eval_splits(), float("nan"), 0.1, True)
    assert math.isnan(row["train_acc"]) and math.isnan(row["ntk_align"]) and row["ntk_trace"] > 0
    assert {"x_n_represented", "x_superposition", "x_dim_per_feature"} <= set(row)
    icl = make_task("icl_markov", {"d_model": 8, "seq_len": 14, "eval_size": 8}, seed=0)
    row = Photometer(ntk_probe=4).measure(0, icl.build_model(), icl, icl.eval_splits(), float("nan"), 0.1, True)
    assert math.isnan(row["ntk_align"]) and np.isfinite(row["ntk_top_eig"]) and np.isfinite(row["x_kl_to_bigram"])
    assert 0.0 <= row["train_acc"] <= 1.0
