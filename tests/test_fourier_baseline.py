"""Tests of the Fourier (Walsh) baseline stored next to every RI curve (tycho.ri.fourier, RICurve.fourier)."""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
import torch

from tycho.analysis.ricurve import RICurve
from tycho.ri.estimators import ri_scan
from tycho.ri.fourier import single_mode_ri, walsh_coefficients, walsh_output
from tycho.ri.variables import VariableSet
from tycho.tasks import make_task
from tycho.train import ObjectSpec, RIConfig, TrainConfig, run_object


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def binary_entropy(p: float) -> float:
    return -sum(x * math.log(x) for x in (p, 1.0 - p) if x > 0)


def planted_vs(n_inputs: int, planted: tuple[int, ...], eps: float, N: int, seed: int = 0) -> VariableSet:
    """Uniform binary inputs x0..x{n-1} and a soft output with ``P(yhat = 1 | x) = (1 + ε χ_A(x))/2``."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(N, n_inputs))
    chi = np.prod(2 * bits[:, list(planted)] - 1, axis=1)
    p1 = (1.0 + eps * chi) / 2.0
    probs = np.stack([1.0 - p1, p1], axis=1)
    X = np.concatenate([bits, (p1 > 0.5).astype(np.int64)[:, None]], axis=1)
    return VariableSet(X=X, cards=[2] * (n_inputs + 1), names=[f"x{i}" for i in range(n_inputs)] + ["yhat"],
                       soft={n_inputs: probs}, must_include=[n_inputs])


# ---------------------------------------------------------------------------------------------- the coefficients
def test_matches_a_brute_force_loop_with_weights_and_nan_rules():
    rng = np.random.default_rng(1)
    N, o = 257, 3
    X = np.column_stack([rng.integers(0, 2, N), rng.integers(0, 2, N), rng.integers(0, 3, N), np.zeros(N, int),
                         rng.integers(0, 2, N)])
    p1 = rng.random(N)
    vs = VariableSet(X=X, cards=[2, 2, 3, 2, 2], names=["a", "b", "tri", "out", "c"],
                     soft={o: np.stack([1 - p1, p1], axis=1)})
    assert walsh_output(vs) == o
    w = rng.random(N) * 3
    subsets = [(3,), (0, 3), (1, 3), (0, 1, 3), (0, 3, 4), (0, 1, 3, 4), (0, 2, 3), (0, 1), (2, 3), (0, 1, 4)]
    got = walsh_coefficients(vs, subsets, weights=w)
    for row, S in enumerate(subsets):
        if o not in S or 2 in S:
            assert math.isnan(got[row]), S
            continue
        terms = [w[n] * (2 * p1[n] - 1) * math.prod(2 * X[n, i] - 1 for i in S if i != o) for n in range(N)]
        assert got[row] == pytest.approx(sum(terms) / w.sum(), abs=1e-12), S
    # unweighted = uniform weights; integer weights = duplicated rows
    assert np.allclose(walsh_coefficients(vs, subsets), walsh_coefficients(vs, subsets, weights=np.ones(N)),
                       equal_nan=True)
    k = rng.integers(0, 3, N)
    dup = VariableSet(X=np.repeat(X, k, axis=0), cards=vs.cards, names=vs.names,
                      soft={o: np.repeat(vs.soft[o], k, axis=0)})
    assert np.allclose(walsh_coefficients(vs, subsets, weights=k.astype(float)), walsh_coefficients(dup, subsets),
                       equal_nan=True)
    assert np.all(np.isnan(walsh_coefficients(vs, subsets, weights=np.zeros(N))))  # zero total weight
    with pytest.raises(ValueError, match="outside"):
        walsh_coefficients(vs, [(0, 3, 5)])


def test_all_nan_without_a_soft_binary_output():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, size=(64, 3))
    hard = VariableSet(X=X, cards=[2, 2, 2], names=["h0", "h1", "correct"])  # modular_addition-like
    assert walsh_output(hard) is None
    assert np.all(np.isnan(walsh_coefficients(hard, [(0, 2), (1, 2), (0, 1, 2)])))
    probs = rng.dirichlet(np.ones(3), size=64)
    ternary = VariableSet(X=np.column_stack([X[:, :2], probs.argmax(1)]), cards=[2, 2, 3], names=["a", "b", "y"],
                          soft={2: probs})
    assert walsh_output(ternary) is None
    assert np.all(np.isnan(walsh_coefficients(ternary, [(0, 2), (0, 1, 2)])))
    assert walsh_coefficients(hard, []).shape == (0,)


def test_single_mode_ri_formula():
    for eps in (0.0, 0.3, -0.7, 1.0):
        assert single_mode_ri(eps) == pytest.approx(math.log(2) - binary_entropy((1 + eps) / 2), abs=1e-15)
    assert math.isnan(float(single_mode_ri(np.nan)))


@pytest.mark.parametrize("planted, eps", [((2,), 0.4), ((1, 3), 0.6), ((0, 2, 4), -0.5)])
def test_planted_mode_gives_epsilon_and_the_matching_ri(planted, eps):
    """Exact check on synthetic data: f̂(A) = ε, other coefficients ≈ 0, RI(A ∪ {yhat}) ≈ ln 2 − h((1 + ε)/2)."""
    N, n_in = 20_000, 6
    vs = planted_vs(n_in, planted, eps, N)
    scan = ri_scan(vs, max_order=4, alpha=0.01, min_effect=1e-3)
    fourier = walsh_coefficients(vs, scan.subsets)
    curve = RICurve.from_scans([0, 10], [scan, scan], fourier=[fourier, fourier])
    assert curve.fourier.shape == (curve.S, curve.T) == (len(scan.subsets), 2)
    row = curve.index_of(planted + (n_in,))
    # (2 p1 − 1) χ_A = ε χ_A² = ε on every row: exact up to rounding
    assert curve.fourier[row, 0] == pytest.approx(eps, abs=1e-12)
    others = np.delete(curve.fourier[:, 0], row)
    assert np.all(np.isfinite(others)) and np.max(np.abs(others)) < 5 * abs(eps) / math.sqrt(N)
    target = math.log(2) - binary_entropy((1 + eps) / 2)
    assert float(single_mode_ri(curve.fourier[row, 0])) == pytest.approx(target, abs=1e-12)
    # the RI scan sees the same mode: plug-in and corrected RI agree with the single-mode formula
    assert curve.ri_plugin[row, 0] == pytest.approx(target, rel=0.03, abs=2e-3)
    assert curve.ri[row, 0] == pytest.approx(target, rel=0.03, abs=2e-3)
    assert bool(scan.detected[row])
    spurious = np.delete(curve.ri_plugin[:, 0], row)
    assert np.max(spurious) < 0.1 * target


# ---------------------------------------------------------------------------------------------- RICurve
def _scan(S: int, value: float):
    class Est:
        ri_corrected = ri_plugin = value
        p_value = 0.5
        n_eff = 100.0

    class Scan:
        subsets = [(0, 2), (1, 2), (0, 1, 2)][:S]
        estimates = [Est()] * S
        q_values = np.full(S, 0.5)

    return Scan()


def test_curve_append_backfills_validates_and_survives_prune():
    curve = RICurve.empty(["x0", "x1", "yhat"], [(0, 2), (1, 2), (0, 1, 2)])
    assert curve.fourier is None
    curve = curve.append(0, _scan(3, 0.0))
    assert curve.fourier is None  # no checkpoint has a baseline yet
    curve = curve.append(5, _scan(3, 0.1), np.array([0.2, -0.3, np.nan]))
    assert curve.fourier.shape == (3, 2)
    assert np.all(np.isnan(curve.fourier[:, 0])) and np.allclose(curve.fourier[:2, 1], [0.2, -0.3])
    curve = curve.append(9, _scan(3, 0.3))  # a checkpoint without the baseline gets a NaN column
    assert curve.fourier.shape == (3, 3) and np.all(np.isnan(curve.fourier[:, 2]))
    with pytest.raises(ValueError, match="fourier"):
        curve.append(12, _scan(3, 0.3), np.zeros(2))
    with pytest.raises(ValueError, match="fourier"):
        RICurve(steps=curve.steps, names=curve.names, subsets=curve.subsets, ri=curve.ri, ri_plugin=curve.ri_plugin,
                p=curve.p, q=curve.q, n_eff=curve.n_eff, fourier=np.zeros((curve.T - 1, curve.S)))
    curve.ri[0] = [0.0, 0.0, 0.05]  # subset 0 now has the smallest peak RI
    pruned = curve.prune(keep_top=2, alpha=0.01, min_effect=1.0)
    assert pruned.subsets == [(1, 2), (0, 1, 2)]
    assert np.array_equal(pruned.fourier, curve.fourier[[1, 2]], equal_nan=True)


def test_curve_files_with_and_without_fourier(tmp_path):
    curve = RICurve.empty(["x0", "x1", "yhat"], [(0, 2), (1, 2)]).append(0, _scan(2, 0.1), np.array([0.5, np.nan]))
    back = RICurve.load(curve.save(tmp_path / "with.npz"))
    assert np.array_equal(back.fourier, curve.fourier, equal_nan=True) and back.fourier.dtype == np.float64
    plain = RICurve.empty(["x0", "x1", "yhat"], [(0, 2), (1, 2)]).append(0, _scan(2, 0.1))
    with np.load(plain.save(tmp_path / "without.npz")) as z:
        assert "fourier" not in z.files
    assert RICurve.load(tmp_path / "without.npz").fourier is None
    # a file in the format written before the baseline existed (exactly the old key set)
    old = tmp_path / "old.npz"
    np.savez_compressed(old, steps=plain.steps, ri=plain.ri, ri_plugin=plain.ri_plugin, p=plain.p, q=plain.q,
                        n_eff=plain.n_eff, subsets_len=np.array([2, 2]), subsets_flat=np.array([0, 2, 1, 2]),
                        header=np.array('{"names": ["x0", "x1", "yhat"], "ground_truth": null, "meta": {}}'))
    legacy = RICurve.load(old)
    assert legacy.fourier is None and legacy.subsets == [(0, 2), (1, 2)] and np.array_equal(legacy.ri, plain.ri)
    assert legacy.prune(keep_top=1).fourier is None


# ---------------------------------------------------------------------------------------------- tasks and runner
TASKS = {  # (config, expected output column of the baseline)
    "sparse_parity": ({"n": 8, "k": 2, "width": 16, "test_size": 128}, "yhat"),
    "markov_lang": ({"n_tokens": 5000, "width": 16, "test_size": 256}, "xhat"),
    "modular_addition": ({"p": 7, "width": 16, "ri_units": 4}, None),  # hard "correct" column
    "icl_markov": ({"d_model": 8, "seq_len": 16, "eval_size": 16}, None),  # vocab 3: non-binary output
    "toy_superposition": ({"n_features": 8, "m_hidden": 3, "eval_size": 128}, None),
}


@pytest.mark.parametrize("name", sorted(TASKS))
def test_which_tasks_get_values(name):
    cfg, output = TASKS[name]
    task = make_task(name, cfg, seed=0)
    model = task.build_model().eval()
    vs = task.ri_variables(model, 256, torch.Generator().manual_seed(0))
    o = walsh_output(vs)
    assert (None if o is None else vs.names[o]) == output
    subsets = [S for k in (2, 3) for S in combinations(range(vs.d), k)]
    f = walsh_coefficients(vs, subsets)
    has_value = np.array([o is not None and o in S for S in subsets])
    assert np.all(np.isfinite(f[has_value])) and np.all(np.abs(f[has_value]) <= 1 + 1e-12)
    assert np.all(np.isnan(f[~has_value]))


def test_sparse_parity_run_stores_the_baseline_end_to_end(tmp_path):
    spec = ObjectSpec(task="sparse_parity", task_cfg={"n": 8, "k": 2, "width": 16, "test_size": 256}, seed=0,
                      train=TrainConfig(steps=120, batch_size=16, optimizer="sgd", momentum=0.9, lr=0.05,
                                        n_log_evals=8, n_lin_evals=4, ntk_every=0))
    res = run_object(spec, work_dir=tmp_path / "w", ri=RIConfig(enabled=True, n_probe=512, n_points=5))
    assert res.status == "done", res.error
    curve = res.ri_curve
    assert curve.meta["fourier_output"] == "yhat"
    assert curve.fourier is not None and curve.fourier.shape == (curve.S, curve.T) and curve.T >= 3
    out = curve.names.index("yhat")
    assert all(out in S for S in curve.subsets)  # every scanned subset contains the output ...
    assert np.all(np.isfinite(curve.fourier)) and np.all(np.abs(curve.fourier) <= 1)  # ... so all values are finite
    # the same probe at the same step gives the same coefficients (probe rows depend on (seed, step) only)
    again = run_object(spec, work_dir=tmp_path / "again", ri=RIConfig(enabled=True, n_probe=512, n_points=5))
    assert np.array_equal(again.ri_curve.fourier, curve.fourier)
    small = curve.prune(keep_top=3, alpha=0.01, min_effect=10.0)
    assert small.S < curve.S and small.fourier.shape == (small.S, small.T)
    for i, S in enumerate(small.subsets):
        assert np.array_equal(small.fourier[i], curve.fourier[curve.index_of(S)])
    for c in (curve, small):
        back = RICurve.load(c.save(tmp_path / f"curve_{c.S}.npz"))
        assert back.subsets == c.subsets and np.array_equal(back.fourier, c.fourier, equal_nan=True)
        assert back.meta["fourier_output"] == "yhat"


def test_modular_addition_run_stores_an_all_nan_baseline(tmp_path):
    spec = ObjectSpec(task="modular_addition", task_cfg={"p": 7, "train_frac": 0.5, "width": 16, "ri_units": 4},
                      seed=0, train=TrainConfig(steps=20, batch_size=16, optimizer="adamw", lr=1e-3, n_log_evals=4,
                                                n_lin_evals=0, ntk_every=0))
    res = run_object(spec, work_dir=tmp_path, ri=RIConfig(enabled=True, n_probe=256, n_points=3))
    assert res.status == "done", res.error
    curve = res.ri_curve
    assert curve.meta["fourier_output"] is None
    assert curve.fourier.shape == (curve.S, curve.T) and np.all(np.isnan(curve.fourier))
