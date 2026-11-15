"""Tests of tycho.tasks: the Task contract for every registered task, and each task's ground truth.

Positive-control identities are checked exactly where possible: oracle models (that compute the target function or
the true conditional) must show the known RI spectrum, and the Markov-language population spectrum is computed from
the exact joint table.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
from torch import nn

from tycho.ri.estimators import ri_scan
from tycho.ri.refined import local_ri_from_table
from tycho.tasks import REGISTRY, make_task
from tycho.tasks.base import derive_seed, logit_margin, torch_generator
from tycho.tasks.icl_markov import (
    MIN_CONTEXT,
    _counts,
    _stationary,
    sample_given,
    strategy_predictions,
    strategy_variables,
)
from tycho.tasks.markov_lang import parse_couplings
from tycho.tasks.modular_addition import fourier_frequency_counts
from tycho.tasks.toy_superposition import superposition_metrics
from tycho.train import ObjectSpec, RIConfig, TrainConfig, run_object
from tycho.train.schedule import eval_steps, subsample_log

SMALL: dict[str, dict] = {
    "sparse_parity": {"n": 8, "k": 2, "width": 16, "test_size": 128},
    "markov_lang": {"n_tokens": 5000, "width": 16, "test_size": 256},
    "modular_addition": {"p": 7, "width": 16, "ri_units": 4},
    "icl_markov": {"d_model": 8, "seq_len": 16, "eval_size": 16},
    "toy_superposition": {"n_features": 8, "m_hidden": 3, "eval_size": 128},
}


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


#: Tasks observed in pattern mode only (no RI scans); their contract is tested in their own files
#: (induction: tests/test_induction.py).
PATTERN_ONLY = {"induction"}


def test_registry_covers_spec_tasks():
    assert set(REGISTRY) == set(SMALL) | PATTERN_ONLY
    assert all(REGISTRY[name].name == name for name in REGISTRY)


# ---------------------------------------------------------------------------------------------- generic contract
@pytest.mark.parametrize("name", sorted(SMALL))
def test_task_contract(name):
    task = make_task(name, SMALL[name], seed=1)
    model = task.build_model()
    x, y = task.sample_batch(6, torch.Generator().manual_seed(0))
    out = model(x)
    loss = task.loss(out, y)
    assert loss.dim() == 0 and math.isfinite(float(loss.detach()))
    acc = task.accuracy(out, y)
    assert acc is None or 0.0 <= acc <= 1.0
    assert (acc is None) == (task.chance_accuracy() is None)
    assert task.ntk_output(out, y).shape == (6,)
    splits = task.eval_splits()
    assert {"train", "test"} <= set(splits)
    again = task.eval_splits()
    for k in ("train", "test"):
        assert torch.equal(splits[k][0], again[k][0]) and torch.equal(splits[k][1], again[k][1])
        assert splits[k][0].shape[0] == splits[k][1].shape[0] > 0
    assert task.default_ri_order() >= 2
    rep = task.hidden_representation(model, splits["test"][0][:5])
    assert rep.dim() == 2 and rep.shape[0] >= 5
    extras = task.extra_metrics(model)
    assert set(extras) == set(task.extra_metrics(model)) and all(isinstance(v, float) for v in extras.values())

    vs = task.ri_variables(model, 64, torch.Generator().manual_seed(2))
    expected_n = task.p ** 2 if name == "modular_addition" else 64
    assert vs.N == expected_n and len(set(vs.names)) == vs.d
    for gt in task.ground_truth() or []:
        assert set(gt["subset"]) <= set(vs.names) and gt["order"] == len(gt["subset"])
    if vs.soft:
        (j, probs), = vs.soft.items()
        assert np.allclose(probs.sum(axis=1), 1.0) and vs.must_include == [j]
    if name == "modular_addition":  # no model randomness is consumed: the probe is all pairs
        assert vs.must_include is None and task.ground_truth() is None


@pytest.mark.parametrize("name", sorted(SMALL))
def test_task_trains_50_steps_with_spectroscopy(name, tmp_path):
    spec = ObjectSpec(task=name, task_cfg=SMALL[name], seed=0,
                      train=TrainConfig(steps=50, batch_size=16, n_log_evals=5, n_lin_evals=3, ntk_every=2,
                                        ntk_probe=8))
    res = run_object(spec, work_dir=tmp_path, ri=RIConfig(enabled=True, n_probe=256, n_points=3))
    assert res.status == "done", res.error
    lc = res.light_curve
    extras = make_task(name, SMALL[name], 0).extra_metrics(make_task(name, SMALL[name], 0).build_model())
    assert {f"x_{k}" for k in extras} <= set(lc)
    assert lc["step"][0] == 0 and lc["step"][-1] == 50
    assert np.all(np.isfinite(lc["train_loss"])) and np.all(np.isfinite(lc["erank_hidden"]))
    curve = res.ri_curve
    expected = subsample_log(eval_steps(50, 5, 3), 3)
    assert curve is not None and np.array_equal(curve.steps, expected) and expected[0] == 0 and expected[-1] == 50
    for gt in curve.ground_truth or []:
        assert set(gt["subset"]) <= set(curve.names)
        assert curve.index_of(gt["subset"]) is not None  # ground-truth rows survive pruning


@pytest.mark.parametrize("name", sorted(SMALL))
def test_seeds_batches_and_problem_sharing(name):
    t0, t1 = make_task(name, SMALL[name], seed=0), make_task(name, SMALL[name], seed=1)
    s0, s1 = t0.build_model().state_dict(), t1.build_model().state_dict()
    s0b = make_task(name, SMALL[name], seed=0).build_model().state_dict()
    assert all(torch.equal(s0[k], s0b[k]) for k in s0)
    assert not all(torch.equal(s0[k], s1[k]) for k in s0)
    b1 = t0.sample_batch(8, torch.Generator().manual_seed(5))
    b2 = t0.sample_batch(8, torch.Generator().manual_seed(5))
    assert torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1])
    # the problem (evaluation data) depends on data_seed only, not on the training seed
    e0, e1 = t0.eval_splits()["test"], t1.eval_splits()["test"]
    assert torch.equal(e0[0], e1[0])
    other = make_task(name, {**SMALL[name], "data_seed": 7}, seed=0).eval_splits()["test"]
    assert not torch.equal(other[0], e0[0]) or not torch.equal(other[1], e0[1])


def test_unknown_names_and_keys_raise():
    with pytest.raises(ValueError, match="unknown task"):
        make_task("nope", {}, 0)
    with pytest.raises(ValueError, match="unknown config keys"):
        make_task("sparse_parity", {"widht": 3}, 0)
    with pytest.raises(ValueError):
        make_task("sparse_parity", {"n": 3, "k": 4}, 0)
    with pytest.raises(ValueError):
        make_task("modular_addition", {"train_frac": 1.0}, 0)


def test_seed_derivation_is_stable_and_separates_streams():
    assert derive_seed(3, "data") == derive_seed(3, "data")
    assert len({derive_seed(3, "data"), derive_seed(3, "ri"), derive_seed(4, "data"), derive_seed(3, "ri", 0)}) == 4
    assert 0 <= derive_seed(2**40, "x") < 2**63
    a = torch.randn(3, generator=torch_generator(1, "a"))
    assert torch.equal(a, torch.randn(3, generator=torch_generator(1, "a")))


def test_logit_margin_identities():
    logits = torch.tensor([[2.0, -1.0], [0.0, 3.0], [1.0, 2.0]])
    y = torch.tensor([0, 1, 0])
    assert torch.allclose(logit_margin(logits, y), torch.tensor([3.0, 3.0, -1.0]))
    lg = torch.randn(5, 4)
    yy = torch.randint(0, 4, (5,))
    p = torch.softmax(lg, -1).gather(1, yy[:, None]).squeeze(1)
    assert torch.allclose(logit_margin(lg, yy), torch.log(p) - torch.log1p(-p), atol=1e-5)


# ---------------------------------------------------------------------------------------------- sparse parity
class ParityOracle(nn.Module):
    """Logits (a, −a) with a = 20 ∏_{i∈S} x_i: predicts y = 1 iff the parity is odd."""

    def __init__(self, support):
        super().__init__()
        self.support = list(support)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        a = 20.0 * x[:, self.support].prod(dim=1)
        return torch.stack([a, -a], dim=1) + 0.0 * self.dummy


def test_sparse_parity_support_labels_and_oracle_spectrum():
    task = make_task("sparse_parity", {"n": 10, "k": 3, "data_seed": 4}, seed=0)
    assert task.support == sorted(np.random.default_rng(4).choice(10, 3, replace=False).tolist())
    x, y = task.sample_batch(500, torch.Generator().manual_seed(1))
    assert set(np.unique(x.numpy())) == {-1.0, 1.0}
    assert torch.equal(y, (x[:, task.support].prod(dim=1) < 0).long())
    oracle = ParityOracle(task.support)
    out = oracle(x)
    assert task.accuracy(out, y) == 1.0
    vs = task.ri_variables(oracle, 4096, torch.Generator().manual_seed(0))
    scan = ri_scan(vs, max_order=task.default_ri_order())
    gt = vs.ground_truth_subsets()[0]
    by_subset = {S: e for S, e in zip(scan.subsets, scan.estimates)}
    assert abs(by_subset[gt].ri_corrected - math.log(2)) < 0.02  # a pure 4th-order line of log 2 nats
    detected = {S for S, d in zip(scan.subsets, scan.detected) if d}
    assert detected == {gt}
    assert max(e.ri_corrected for S, e in by_subset.items() if S != gt) < 0.01


def test_sparse_parity_finite_training_set_and_support_frac():
    task = make_task("sparse_parity", {"n": 6, "k": 2, "n_train": 40, "width": 4}, seed=0)
    x_tr, y_tr = task.eval_splits()["train"]
    assert x_tr.shape == (40, 6)
    xb, yb = task.sample_batch(200, torch.Generator().manual_seed(0))
    rows = {tuple(r) for r in x_tr.tolist()}
    assert all(tuple(r) in rows for r in xb.tolist())
    model = task.build_model()
    with torch.no_grad():
        w = model.first_layer().weight
        w.zero_()
        w[:, task.support] = 1.0
    assert task.extra_metrics(model)["support_frac"] == 1.0
    with torch.no_grad():
        w.fill_(0.3)
    assert abs(task.extra_metrics(model)["support_frac"] - 2 / 6) < 1e-12


# ---------------------------------------------------------------------------------------------- markov language
def test_parse_couplings_forms_and_errors():
    a = parse_couplings({"1": 1.0, "3,1": 0.8}, window=6)
    b = parse_couplings(" 1 : 1.0 ; 1, 3 : 0.8 ;", window=6)
    c = parse_couplings({1: 1.0, (3, 1): 0.8}, window=6)
    assert a == b == c == {(1,): 1.0, (1, 3): 0.8}
    assert list(parse_couplings("1,2,3:1;2:0.5", 3)) == [(2,), (1, 2, 3)]
    for bad in ["0:1.0", "7:1.0", "1,1:1.0", "1:1.0;1:2.0", "1-1.0", {"1": "x"}, 3.0]:
        with pytest.raises(ValueError):
            parse_couplings(bad, window=6)


class MarkovOracle(nn.Module):
    """MLP-input oracle: logits (−h, h) with h the planted log-odds/2, so softmax gives P_true."""

    def __init__(self, couplings, bias):
        super().__init__()
        self.couplings, self.bias = couplings, bias
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, s):  # s: (B, W) spins in lag order
        h = torch.full((s.shape[0],), self.bias, dtype=s.dtype)
        for L, beta in self.couplings.items():
            h = h + beta * s[:, [lag - 1 for lag in L]].prod(dim=1)
        return torch.stack([-h, h], dim=1) + 0.0 * self.dummy


def test_markov_conditional_corpus_statistics_and_oracle():
    task = make_task("markov_lang", {"couplings": "1:1.0;1,3:0.8", "n_tokens": 60_000, "test_size": 2000}, 0)
    bits = np.asarray(list(itertools.product([0, 1], repeat=6)))
    s = 2.0 * bits - 1.0
    h = 1.0 * s[:, 0] + 0.8 * s[:, 0] * s[:, 2]
    assert np.allclose(task.true_conditional(bits), 1.0 / (1.0 + np.exp(-2.0 * h)))
    # the corpus follows the source: empirical P(x_t = 1 | x[t-1], x[t-3]) matches σ(2h)
    corpus = task._corpus
    lag, tgt = corpus["lag"], corpus["targets"].numpy()
    for b1, b3 in itertools.product([0, 1], repeat=2):
        sel = (lag[:, 0] == b1) & (lag[:, 2] == b3)
        s1, s3 = 2 * b1 - 1, 2 * b3 - 1
        p = 1.0 / (1.0 + math.exp(-2.0 * (s1 + 0.8 * s1 * s3)))
        assert abs(tgt[sel].mean() - p) < 4.0 * math.sqrt(p * (1 - p) / sel.sum()) + 1e-3
    oracle = MarkovOracle(task.couplings, task.bias)
    assert task.extra_metrics(oracle)["kl_true"] < 1e-12
    gen = torch.Generator().manual_seed(3)
    vs_model = task.ri_variables(oracle, 500, gen)
    vs_true = task.true_variables(500, torch.Generator().manual_seed(3))
    assert np.array_equal(vs_model.X[:, :-1], vs_true.X[:, :-1])
    assert np.allclose(vs_model.soft[6], vs_true.soft[6], atol=1e-12)


def _population_ri(task) -> dict[tuple[str, ...], float]:
    """Exact local RI of every subset containing xhat under uniform probe contexts (2^W windows enumerated)."""
    W = task.window
    bits = np.asarray(list(itertools.product([0, 1], repeat=W)))
    p1 = task.true_conditional(bits)
    P = np.zeros((2,) * (W + 1))
    for row, p in zip(bits, p1):
        P[tuple(row) + (1,)] = p / 2 ** W
        P[tuple(row) + (0,)] = (1.0 - p) / 2 ** W
    subsets = [A + (W,) for r in range(1, W + 1) for A in itertools.combinations(range(W), r)]
    names = task.lag_names + ["xhat"]
    ri = local_ri_from_table(P, subsets)
    return {tuple(names[i] for i in S): float(v) for S, v in zip(subsets, ri)}


@pytest.mark.parametrize("couplings", ["1:1.0;1,3:0.8", "1:1.0;1,2,3:1.0", "1,3:0.8"])
def test_markov_population_spectrum_is_exactly_the_planted_one(couplings):
    """Uniform contexts, bias 0, at most two couplings: RI > 0 on the planted subsets, exactly 0 elsewhere."""
    task = make_task("markov_lang", {"couplings": couplings, "n_tokens": 2000}, 0)
    ri = _population_ri(task)
    planted = {tuple(gt["subset"]) for gt in task.ground_truth()}
    assert all(ri[S] > 0.05 for S in planted)
    assert max(v for S, v in ri.items() if S not in planted) < 1e-10


@pytest.mark.parametrize("cfg,induced", [
    ({"couplings": "1:1.0;1,3:0.8", "bias": 0.5}, ("x[t-3]", "xhat")),
    ({"couplings": "1,2:1.0;1,4:1.0;3,4:1.0"}, ("x[t-2]", "x[t-3]", "xhat")),
])
def test_markov_marginalisation_can_induce_unplanted_terms(cfg, induced):
    """Outside the documented regime the exact spectrum has small unplanted lines (documented limitation)."""
    task = make_task("markov_lang", {**cfg, "n_tokens": 2000}, 0)
    ri = _population_ri(task)
    assert ri[induced] > 1e-5
    assert all(gt["subset"] != list(induced) for gt in task.ground_truth())


def test_markov_true_variables_scan_detects_exactly_the_planted_subsets():
    task = make_task("markov_lang", {"couplings": "1:1.0;1,3:0.8", "n_tokens": 2000}, 0)
    vs = task.true_variables(20_000, torch.Generator().manual_seed(0))
    scan = ri_scan(vs, max_order=task.default_ri_order())
    detected = {vs.subset_names(S) for S, d in zip(scan.subsets, scan.detected) if d}
    assert detected == {("x[t-1]", "xhat"), ("x[t-1]", "x[t-3]", "xhat")}
    assert task.default_ri_order() == 3


def test_markov_probe_contexts_uniform_vs_chain_and_transformer_inputs():
    cfg = {"couplings": "1:1.5", "n_tokens": 20_000}
    agree = {}
    for ctx in ("uniform", "chain"):
        task = make_task("markov_lang", {**cfg, "ri_context": ctx}, 0)
        vs = task.true_variables(4000, torch.Generator().manual_seed(1))
        agree[ctx] = float(np.mean(vs.X[:, 0] == vs.X[:, 1]))
    assert abs(agree["uniform"] - 0.5) < 0.05
    assert abs(agree["chain"] - 1.0 / (1.0 + math.exp(-3.0))) < 0.05  # P(x_t = x_{t-1}) = σ(2β)
    tr = make_task("markov_lang", {**cfg, "model": "transformer", "d_model": 8, "n_head": 2}, 0)
    x, y = tr.sample_batch(16, torch.Generator().manual_seed(0))
    assert x.dtype == torch.long and x.shape == (16, 6)
    lag = tr._corpus["lag"]
    # temporal order: the last input token is x[t-1], the first is x[t-W]
    assert torch.equal(tr._corpus["inputs"][:, -1], torch.as_tensor(lag[:, 0], dtype=torch.long))
    assert torch.equal(tr._corpus["inputs"][:, 0], torch.as_tensor(lag[:, 5], dtype=torch.long))
    model = tr.build_model()
    assert tr.loss(model(x), y).dim() == 0 and tr.ntk_output(model(x), y).shape == (16,)


# ---------------------------------------------------------------------------------------------- modular addition
def test_modular_addition_data_split_and_batches():
    task = make_task("modular_addition", {"p": 11, "train_frac": 0.3}, seed=0)
    assert task.n_train == round(0.3 * 121)
    assert set(task.train_idx) | set(task.test_idx) == set(range(121)) and not set(task.train_idx) & set(task.test_idx)
    x, y = task._all
    assert torch.all(x.sum(dim=1) == 2)
    a, b = x[:, :11].argmax(dim=1), x[:, 11:].argmax(dim=1)
    assert torch.equal(y, (a + b) % 11) and torch.equal(a * 11 + b, torch.arange(121))
    other_seed = make_task("modular_addition", {"p": 11, "train_frac": 0.3}, seed=5)
    assert np.array_equal(task.train_idx, other_seed.train_idx)
    gen = torch.Generator().manual_seed(0)
    state = gen.get_state()
    xb, yb = task.sample_batch(10_000, gen)  # batch >= training set: full batch, no randomness used
    assert torch.equal(xb, task.eval_splits()["train"][0]) and torch.equal(gen.get_state(), state)
    xs, _ = task.sample_batch(20, gen)
    assert len({tuple(r) for r in xs.tolist()}) == 20  # without replacement


def test_modular_addition_ri_units_are_frozen_and_correct_flag():
    task = make_task("modular_addition", {"p": 7, "width": 32, "ri_units": 5}, seed=0)
    model = task.build_model()
    vs = task.ri_variables(model, 0, torch.Generator())
    assert vs.N == 49 and vs.names[-1] == "correct" and len(vs.names) == 6
    with torch.no_grad():
        logits = model(task._all[0])
    assert np.array_equal(vs.X[:, -1], (logits.argmax(1) == task._all[1]).numpy().astype(int))
    frozen = vs.meta["frozen"]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    vs2 = task.ri_variables(model, 0, torch.Generator(), frozen=frozen)
    assert vs2.names == vs.names
    units = frozen["units"]
    hidden = model.hidden(task._all[0])[-1].detach().numpy()
    assert np.array_equal(vs2.X[:, :-1], (hidden[:, units] > 0).astype(int))


def test_fourier_frequency_counts():
    p, width = 13, 10
    a = np.arange(p)
    phases = np.linspace(0, 2, width)[:, None]

    def wave(freqs):
        return np.cos(2 * np.pi * np.asarray(freqs)[:, None] * a[None, :] / p + phases)

    one = wave([3] * width)
    assert np.allclose(fourier_frequency_counts(np.concatenate([one, one], 1), p), (1.0, 1.0))
    spread = wave([(j % 5) + 1 for j in range(width)])  # each unit single-frequency, five frequencies in use
    g, u = fourier_frequency_counts(np.concatenate([spread, spread], 1), p)
    assert abs(g - 5.0) < 1e-9 and abs(u - 1.0) < 1e-9
    two = one + np.sin(2 * np.pi * 5 * a[None, :] / p + phases)  # every unit uses two equal-power frequencies
    assert np.allclose(fourier_frequency_counts(np.concatenate([two, np.zeros_like(two)], 1), p), (2.0, 2.0))
    rand = np.random.default_rng(0).normal(size=(2048, 2 * 23))
    g, u = fourier_frequency_counts(rand, 23)
    assert g > 0.99 * 11 and 0.75 * 11 < u < 0.88 * 11
    nyquist = np.cos(np.pi * np.arange(8))[None, :].repeat(3, 0)  # p = 8: frequency 4 has no partner
    assert np.allclose(fourier_frequency_counts(np.concatenate([nyquist, nyquist], 1), 8), (1.0, 1.0))
    assert all(math.isnan(v) for v in fourier_frequency_counts(np.ones((3, 2 * p)), p))  # no non-DC power


# ---------------------------------------------------------------------------------------------- in-context Markov
def test_icl_stationary_and_sampling_given_transitions():
    rng = np.random.default_rng(0)
    T = rng.dirichlet(np.ones(4), size=(5, 4))
    pi = _stationary(T)
    assert np.allclose(np.einsum("mi,mij->mj", pi, T), pi) and np.allclose(pi.sum(1), 1.0)
    cycle = np.zeros((2, 3, 3))
    cycle[:, [0, 1, 2], [1, 2, 0]] = 1.0  # deterministic 0 → 1 → 2 → 0
    seq = sample_given(cycle, np.asarray([[1.0, 0, 0], [0, 0, 1.0]]), 7, rng)
    assert seq.tolist() == [[0, 1, 2, 0, 1, 2, 0], [2, 0, 1, 2, 0, 1, 2]]
    trans = np.broadcast_to(np.asarray([[0.9, 0.1], [0.3, 0.7]]), (400, 2, 2))
    long = sample_given(trans, np.full((400, 2), 0.5), 200, rng)
    prev, nxt = long[:, :-1].ravel(), long[:, 1:].ravel()
    assert abs(np.mean(nxt[prev == 0] == 1) - 0.1) < 0.01 and abs(np.mean(nxt[prev == 1] == 1) - 0.7) < 0.01


def _brute_counts(seq, t, vocab):
    uni = np.bincount(seq[: t + 1], minlength=vocab)
    succ = np.zeros(vocab)
    for s in range(t):
        if seq[s] == seq[t]:
            succ[seq[s + 1]] += 1
    return uni, succ


def test_icl_counts_strategies_and_variables_match_brute_force():
    rng = np.random.default_rng(1)
    vocab, alpha = 3, 0.7
    seq = rng.integers(0, vocab, size=(6, 20))
    uni, succ = _counts(seq, vocab)
    unigram, bigram = strategy_predictions(seq, vocab, alpha)
    for i in range(6):
        for t in range(20):
            u, s = _brute_counts(seq[i], t, vocab)
            assert np.array_equal(uni[i, t], u) and np.array_equal(succ[i, t], s)
            assert np.allclose(unigram[i, t], (u + 0.5) / (t + 1 + 1.5))
            assert np.allclose(bigram[i, t], (s + alpha) / (s.sum() + vocab * alpha))
    pos = rng.integers(MIN_CONTEXT, 20, size=6)
    x_t, u_t, b_t = strategy_variables(seq, pos, vocab, rng)
    for i in range(6):
        u, s = _brute_counts(seq[i], pos[i], vocab)
        assert x_t[i] == seq[i, pos[i]]
        assert u[u_t[i]] == u.max()  # an argmax (ties broken at random)
        assert (s[b_t[i]] == s.max()) if s.sum() > 0 else (b_t[i] == u_t[i])


class BigramOracle(nn.Module):
    """Predicts exactly the in-context bigram (or unigram) strategy."""

    def __init__(self, vocab, alpha, which):
        super().__init__()
        self.vocab, self.alpha, self.which = vocab, alpha, which
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        uni, bi = strategy_predictions(idx.numpy(), self.vocab, self.alpha)
        return torch.log(torch.as_tensor(bi if self.which == "bigram" else uni, dtype=torch.float32)) + self.dummy


def test_icl_strategy_kls_vanish_for_the_matching_oracle():
    task = make_task("icl_markov", {"seq_len": 24, "eval_size": 64}, seed=0)
    kb = task.extra_metrics(BigramOracle(3, 1.0, "bigram"))
    ku = task.extra_metrics(BigramOracle(3, 1.0, "unigram"))
    assert kb["kl_to_bigram"] < 1e-6 < kb["kl_to_unigram"]
    assert ku["kl_to_unigram"] < 1e-6 < ku["kl_to_bigram"]
    vs = task.ri_variables(BigramOracle(3, 1.0, "bigram"), 2000, torch.Generator().manual_seed(0))
    scan = ri_scan(vs, max_order=2)
    est = {vs.subset_names(S): e for S, e in zip(scan.subsets, scan.estimates)}
    # the bigram oracle's prediction is far more informative about b_t than about u_t
    assert est[("b_t", "xhat")].ri_corrected > est[("u_t", "xhat")].ri_corrected > 0


# ---------------------------------------------------------------------------------------------- superposition
def test_superposition_metrics_known_geometries():
    m = superposition_metrics(np.eye(3))
    assert m == {"n_represented": 3.0, "superposition": 0.0, "dim_per_feature": 1.0}
    anti = superposition_metrics(np.asarray([[1.0, -1.0, 0.1]]))  # antipodal pair + an unrepresented feature
    assert anti["n_represented"] == 2.0
    assert abs(anti["superposition"] - (1.0 + 0.01)) < 1e-12  # interference from the partner and the small one
    assert abs(anti["dim_per_feature"] - 1.0 / (1.0 + 1.0 + 0.01)) < 1e-12
    none = superposition_metrics(np.zeros((2, 4)))
    assert none["n_represented"] == 0.0 and math.isnan(none["superposition"]) and math.isnan(none["dim_per_feature"])


def test_superposition_data_loss_and_variables():
    task = make_task("toy_superposition", {"n_features": 10, "feature_prob": 0.2}, seed=0)
    x, y = task.sample_batch(20_000, torch.Generator().manual_seed(0))
    assert torch.equal(x, y)
    assert abs(float((x > 0).float().mean()) - 0.2) < 0.01 and float(x.max()) < 1.0
    model = task.build_model()
    out = model(x[:50])
    manual = ((x[:50] - out) ** 2 * 0.9 ** torch.arange(10.0)).sum(1).mean()
    assert torch.allclose(task.loss(out, x[:50]), manual)
    assert torch.allclose(task.ntk_output(out, x[:50]), -((x[:50] - out) ** 2 * 0.9 ** torch.arange(10.0)).sum(1))
    vs = task.ri_variables(model, 300, torch.Generator().manual_seed(1))
    assert vs.names == [f"a{i}" for i in range(5)] + [f"r{i}" for i in range(5)]
    assert vs.must_include is None and set(np.unique(vs.X)) <= {0, 1}
