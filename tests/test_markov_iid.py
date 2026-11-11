"""Tests of the Markov-language option ``train_context = "iid"``: uniform i.i.d. training contexts with targets from
the chain's true conditional, next to an unchanged default (``"chain"``) path.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np
import pytest
import torch
from torch import nn

from tycho.tasks import make_task
from tycho.tasks.markov_lang import TRAIN_CONTEXTS
from tycho.train import ObjectSpec, RIConfig, TrainConfig, run_object

COUPLINGS = "1:1.0;1,3:0.8"


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _task(ctx: str | None, **cfg):
    """Markov task with the test couplings; ``ctx = None`` leaves ``train_context`` out of the config."""
    base = {"couplings": COUPLINGS, "n_tokens": 60_000, "test_size": 4096}
    if ctx is not None:
        base["train_context"] = ctx
    return make_task("markov_lang", {**base, **cfg}, 0)


def _spins(lag: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(lag, dtype=np.float64) - 1.0


def _data_digest(task) -> str:
    d = hashlib.sha256()
    c, te = task._corpus, task._test
    for a in (c["lag"], c["targets"].numpy(), te["inputs"].numpy(), te["targets"].numpy(), te["p_true"]):
        d.update(np.ascontiguousarray(a).tobytes())
    return d.hexdigest()[:16]


class Oracle(nn.Module):
    """MLP-input oracle: logits (−h, h), so softmax gives the true conditional σ(2h)."""

    def __init__(self, couplings, bias):
        super().__init__()
        self.couplings, self.bias = couplings, bias
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, s):  # s: (B, W) spins in lag order
        h = torch.full((s.shape[0],), self.bias, dtype=s.dtype)
        for L, beta in self.couplings.items():
            h = h + beta * s[:, [lag - 1 for lag in L]].prod(dim=1)
        return torch.stack([-h, h], dim=1) + 0.0 * self.dummy


# ---------------------------------------------------------------------------------------------- contexts
def test_iid_windows_are_uniform_and_uncorrelated_unlike_the_chain():
    chain, iid = _task("chain"), _task("iid")
    s_chain, s_iid = _spins(chain._corpus["lag"]), _spins(iid._corpus["lag"])
    assert s_chain.shape == s_iid.shape == (60_000 - 6, 6)  # same number of examples as the chain corpus
    assert chain._corpus["n_pos"] == iid._corpus["n_pos"] == 60_000 - 6
    # the chain's pairwise coupling {1} correlates the lags: s(t-1)·s(t-3) = +1 in ~85 % of the contexts
    e13_chain, e13_iid = float(np.mean(s_chain[:, 0] * s_chain[:, 2])), float(np.mean(s_iid[:, 0] * s_iid[:, 2]))
    assert e13_chain > 0.6
    assert abs(e13_iid) < 0.02  # sd ≈ 1/√60000 ≈ 0.004
    # every bit is a fair coin and every pair is uncorrelated (training and test split)
    for s, tol in ((s_iid, 0.02), (iid._test["inputs"].numpy().astype(np.float64), 0.08)):
        assert np.abs(s.mean(axis=0)).max() < tol
        second = s.T @ s / s.shape[0]
        assert np.abs(second - np.eye(6)).max() < tol
    # all 2^W windows are equally frequent (multinomial, 64 cells of ≈ 937)
    codes = (iid._corpus["lag"].astype(np.int64) << np.arange(6)).sum(axis=1)
    counts = np.bincount(codes, minlength=64)
    expected = codes.size / 64
    assert np.abs(counts - expected).max() < 5.0 * math.sqrt(expected)


def test_iid_targets_follow_the_true_conditional():
    task = _task("iid")
    lag, tgt = task._corpus["lag"], task._corpus["targets"].numpy()
    p_true = task.true_conditional(lag)
    bits = np.asarray(list(itertools.product([0, 1], repeat=6)))
    s = 2.0 * bits - 1.0
    expected = 1.0 / (1.0 + np.exp(-2.0 * (1.0 * s[:, 0] + 0.8 * s[:, 0] * s[:, 2])))
    assert np.allclose(task.true_conditional(bits), expected)  # the conditional does not depend on train_context
    codes = (lag.astype(np.int64) << np.arange(6)).sum(axis=1)
    for code in range(64):  # every window is frequent (≈ 937 examples)
        sel = codes == code
        p = p_true[sel][0]
        assert abs(tgt[sel].mean() - p) < 4.5 * math.sqrt(p * (1 - p) / sel.sum()) + 1e-3
    # pooled over the (x[t-1], x[t-3]) cells, as in the chain test
    for b1, b3 in itertools.product([0, 1], repeat=2):
        sel = (lag[:, 0] == b1) & (lag[:, 2] == b3)
        s1, s3 = 2 * b1 - 1, 2 * b3 - 1
        p = 1.0 / (1.0 + math.exp(-2.0 * (s1 + 0.8 * s1 * s3)))
        assert abs(tgt[sel].mean() - p) < 4.0 * math.sqrt(p * (1 - p) / sel.sum()) + 1e-3
    # the test split carries p_true of its own (iid) windows, and its targets follow it
    test = task._test
    test_bits = ((test["inputs"].numpy() + 1.0) / 2.0).astype(np.int64)
    assert np.array_equal(test["p_true"], task.true_conditional(test_bits))
    y, p = test["targets"].numpy().astype(np.float64), test["p_true"]
    assert abs((y - p).sum()) < 4.0 * math.sqrt((p * (1 - p)).sum())
    # kl_true works: the oracle is perfect, a constant predictor is not
    assert task.extra_metrics(Oracle(task.couplings, task.bias))["kl_true"] < 1e-12
    assert task.extra_metrics(Oracle({}, 0.0))["kl_true"] > 0.05


def test_iid_true_variables_and_ground_truth_unchanged():
    chain, iid = _task("chain", n_tokens=2000), _task("iid", n_tokens=2000)
    assert chain.ground_truth() == iid.ground_truth()
    assert chain.default_ri_order() == iid.default_ri_order()
    a, b = chain.true_variables(2000, torch.Generator().manual_seed(4)), iid.true_variables(
        2000, torch.Generator().manual_seed(4))
    assert np.array_equal(a.X, b.X) and np.array_equal(a.soft[6], b.soft[6])
    # ri_context = "chain" draws probes from the training examples: uniform under iid training
    agree = {}
    for ctx in TRAIN_CONTEXTS:
        task = make_task("markov_lang", {"couplings": "1:1.5", "n_tokens": 20_000, "ri_context": "chain",
                                         "train_context": ctx}, 0)
        vs = task.true_variables(4000, torch.Generator().manual_seed(1))
        agree[ctx] = float(np.mean(vs.X[:, 0] == vs.X[:, 1]))
    assert abs(agree["chain"] - 1.0 / (1.0 + math.exp(-3.0))) < 0.05  # P(x_t = x_{t-1}) = σ(2β)
    assert abs(agree["iid"] - 0.5) < 0.05


# ---------------------------------------------------------------------------------------------- determinism
def test_iid_data_is_deterministic_in_data_seed_only():
    a, b = _task("iid", n_tokens=5000), _task("iid", n_tokens=5000)
    other_seed = make_task("markov_lang", {"couplings": COUPLINGS, "n_tokens": 5000, "train_context": "iid"}, 7)
    assert _data_digest(a) == _data_digest(b) == _data_digest(other_seed)  # training seed: same problem
    assert _data_digest(_task("iid", n_tokens=5000, data_seed=1)) != _data_digest(a)
    assert _data_digest(_task("chain", n_tokens=5000)) != _data_digest(a)
    for k in ("train", "test"):
        sa, sb = a.eval_splits()[k], b.eval_splits()[k]
        assert torch.equal(sa[0], sb[0]) and torch.equal(sa[1], sb[1])
    xa, ya = a.sample_batch(64, torch.Generator().manual_seed(3))
    xb, yb = b.sample_batch(64, torch.Generator().manual_seed(3))
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    # n_train restricts training to a prefix of the same examples
    sub = _task("iid", n_tokens=5000, n_train=700)
    assert sub._corpus["n_pos"] == 700
    assert np.array_equal(sub._corpus["lag"], a._corpus["lag"])
    assert torch.equal(sub._corpus["targets"], a._corpus["targets"])
    frac = float(a._corpus["targets"][:700].float().mean())
    assert sub.chance_accuracy() == max(frac, 1.0 - frac)


def test_chain_default_is_unchanged():
    omitted, chain = _task(None, n_tokens=5000), _task("chain", n_tokens=5000)
    assert omitted.cfg["train_context"] == "chain" and omitted.train_context == "chain"
    assert _data_digest(omitted) == _data_digest(chain)
    for k in ("train", "test"):
        so, sc = omitted.eval_splits()[k], chain.eval_splits()[k]
        assert torch.equal(so[0], sc[0]) and torch.equal(so[1], sc[1])
    xo, yo = omitted.sample_batch(64, torch.Generator().manual_seed(3))
    xc, yc = chain.sample_batch(64, torch.Generator().manual_seed(3))
    assert torch.equal(xo, xc) and torch.equal(yo, yc)
    # regression pin: digest of the chain corpus and test split as generated before train_context existed
    pinned = make_task("markov_lang", {"couplings": COUPLINGS, "n_tokens": 3000, "test_size": 200}, 0)
    assert _data_digest(pinned) == "ebd7bb057304b53a"
    assert pinned._corpus["targets"][:16].tolist() == [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_invalid_train_context_raises():
    for bad in ("uniform", "IID", "", "markov"):
        with pytest.raises(ValueError, match="train_context"):
            make_task("markov_lang", {"train_context": bad, "n_tokens": 2000}, 0)


# ---------------------------------------------------------------------------------------------- end to end
def test_iid_transformer_inputs_are_in_temporal_order():
    task = make_task("markov_lang", {"couplings": COUPLINGS, "n_tokens": 3000, "train_context": "iid",
                                     "model": "transformer", "d_model": 8, "n_head": 2}, 0)
    lag, inputs = task._corpus["lag"], task._corpus["inputs"]
    assert inputs.dtype == torch.long and inputs.shape == (3000 - 6, 6)
    assert torch.equal(inputs, torch.as_tensor(lag[:, ::-1].copy(), dtype=torch.long))
    x, y = task.sample_batch(16, torch.Generator().manual_seed(0))
    model = task.build_model()
    assert task.loss(model(x), y).dim() == 0 and task.ntk_output(model(x), y).shape == (16,)


@pytest.mark.parametrize("model_cfg", [{"model": "mlp", "width": 16},
                                       {"model": "transformer", "d_model": 8, "n_head": 2}])
def test_iid_trains_end_to_end_with_spectroscopy(model_cfg, tmp_path):
    cfg = {"couplings": COUPLINGS, "n_tokens": 5000, "test_size": 256, "train_context": "iid", **model_cfg}
    spec = ObjectSpec(task="markov_lang", task_cfg=cfg, seed=0,
                      train=TrainConfig(steps=200, batch_size=32, optimizer="adam", lr=3e-3, n_log_evals=5,
                                        n_lin_evals=3, ntk_every=2, ntk_probe=8))
    res = run_object(spec, work_dir=tmp_path, ri=RIConfig(enabled=True, n_probe=256, n_points=3))
    assert res.status == "done", res.error
    lc = res.light_curve
    assert lc["step"][-1] == 200 and np.all(np.isfinite(lc["train_loss"]))
    kl = np.asarray(lc["x_kl_true"])
    assert np.all(np.isfinite(kl)) and kl[-1] < kl[0]  # learning toward the true conditional
    curve = res.ri_curve
    assert curve is not None and curve.ri.shape[1] == len(curve.steps)
    for gt in curve.ground_truth or []:
        assert set(gt["subset"]) <= set(curve.names)
