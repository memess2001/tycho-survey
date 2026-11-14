"""Sky v2 Part A engineering: exact pmfs, pattern observation of the Markov and parity tasks, correlated parity,
the runner's pattern mode (with exact resume) and the worker's pattern file."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tycho.tasks import make_task
from tycho.tasks.base import torch_generator
from tycho.tasks.markov_lang import _generate, _lag_windows
from tycho.tasks.patterns import binary_losses, chain_pmf, markov_window_pmf, pattern_bits, sample_codes
from tycho.train import ObjectSpec, RIConfig, TrainConfig, run_object
from tycho.train.runner import RESUME_FILE


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


# ---------------------------------------------------------------------------------------------- pmfs
def test_chain_pmf_has_uniform_marginals_and_the_requested_correlation():
    for rho in (-0.5, 0.0, 0.5, 0.8):
        pmf = chain_pmf(6, rho)
        s = 2 * pattern_bits(6) - 1
        assert pmf.sum() == pytest.approx(1.0)
        assert np.allclose(pmf @ s, 0.0, atol=1e-12)
        for lag in (1, 2, 3):
            assert pmf @ (s[:, 0] * s[:, lag]) == pytest.approx(rho ** lag, abs=1e-12)
    assert np.allclose(chain_pmf(4, 0.0), 1 / 16)


@pytest.mark.parametrize("couplings", ["1:1.0;1,2,3:1.0", "1,3:0.8", "1:2.0;1,3:1.6"])
def test_markov_window_pmf_is_stationary_and_matches_a_long_corpus(couplings):
    task = make_task("markov_lang", {"couplings": couplings}, 0)
    pi = markov_window_pmf(task.p_table, 6)
    codes = np.arange(64)
    T = np.zeros((64, 64))
    T[codes, (codes << 1) & 63] += 1 - task.p_table
    T[codes, ((codes << 1) | 1) & 63] += task.p_table
    assert np.allclose(pi @ T, pi, atol=1e-13) and pi.sum() == pytest.approx(1.0)
    seq = _generate(task.p_table, 6, 400_000, np.random.default_rng(1))
    lag = _lag_windows(seq, 6).astype(np.int64)
    freq = np.bincount((lag << np.arange(6)).sum(axis=1), minlength=64) / lag.shape[0]
    assert 0.5 * np.abs(freq - pi).sum() < 0.03


def test_sample_codes_is_deterministic_and_follows_the_pmf():
    pmf = chain_pmf(8, 0.8)
    a = sample_codes(pmf, 200_000, torch_generator(3, "ri", 7))
    b = sample_codes(pmf, 200_000, torch_generator(3, "ri", 7))
    assert np.array_equal(a, b)
    freq = np.bincount(a, minlength=256) / a.size
    assert 0.5 * np.abs(freq - pmf).sum() < 0.02


def test_binary_losses_are_kl_and_squared_error():
    z = np.array([0.0, 2.0, -1.0])
    p = np.array([0.5, 1.0, 0.2])
    w = np.array([0.2, 0.3, 0.5])
    q = 1 / (1 + np.exp(-z))
    kl = [0.0, -np.log(q[1]), 0.2 * np.log(0.2 / q[2]) + 0.8 * np.log(0.8 / (1 - q[2]))]
    got_kl, got_sq = binary_losses(z, p, w)
    assert got_kl == pytest.approx(float(w @ np.array(kl)))
    assert got_sq == pytest.approx(float(w @ (q - p) ** 2))


# ---------------------------------------------------------------------------------------------- tasks
def test_markov_pattern_logodds_match_the_model_on_the_same_windows():
    task = make_task("markov_lang", {"couplings": "1:1.0;1,2,3:1.0", "exact_metrics": True}, 4)
    model = task.build_model()
    z = task.pattern_logodds(model)
    bits = pattern_bits(6)  # column j = lag j + 1
    with torch.no_grad():
        logits = model(task._model_inputs(bits))
    z32 = (logits[:, 1] - logits[:, 0]).double().numpy()
    assert z.dtype == np.float64 and np.allclose(z, z32, atol=1e-5)
    # the code convention of the task's own conditional table
    assert np.allclose(task.pattern_truth(), task.true_conditional(bits))
    # light-curve metrics use the stationary pmf for chain training and the uniform one for iid training
    m = task.extra_metrics(model)
    assert m["lossD"] == pytest.approx(binary_losses(z, task.p_table, markov_window_pmf(task.p_table, 6))[0])
    iid = make_task("markov_lang", {"couplings": "1:1.0;1,2,3:1.0", "train_context": "iid"}, 4)
    assert np.allclose(iid.pattern_pmfs()["D"], 1 / 64)
    assert "lossD" not in iid.extra_metrics(model)


def test_correlated_parity_sampler_labels_and_patterns():
    t = make_task("sparse_parity", {"n": 8, "k": 3, "support": [1, 3, 6], "rho": 0.5, "exact_metrics": True}, 0)
    assert t.support == [1, 3, 6]
    x, y = t._sample(100_000, torch.Generator().manual_seed(0))
    xn = x.numpy()
    assert np.mean(xn[:, 2:] * xn[:, :-2]) == pytest.approx(0.25, abs=0.01)
    assert np.array_equal(y.numpy(), (xn[:, [1, 3, 6]].prod(axis=1) < 0).astype(np.int64))
    bits = (xn > 0).astype(np.int64)
    codes = (bits << np.arange(8)).sum(axis=1)
    assert np.array_equal(t.pattern_truth()[codes], y.numpy().astype(np.float64))
    model = t.build_model()
    z = t.pattern_logodds(model)
    with torch.no_grad():
        logits = model(torch.as_tensor(2.0 * pattern_bits(8) - 1, dtype=torch.float32))
    assert np.allclose(z, (logits[:, 1] - logits[:, 0]).double().numpy(), atol=1e-5)
    assert set(t.extra_metrics(model)) >= {"lossD", "lossU", "lossS", "sqD", "sqU", "sqS", "support_frac"}


def test_parity_rho_zero_keeps_the_v1_sampler_bit_for_bit():
    a = make_task("sparse_parity", {"n": 8, "k": 2}, 0)
    b = make_task("sparse_parity", {"n": 8, "k": 2, "rho": 0.0}, 0)
    for t in (a, b):
        t._x, t._y = t._sample(64, torch.Generator().manual_seed(5))
    assert torch.equal(a._x, b._x) and torch.equal(a._y, b._y)
    with pytest.raises(ValueError):
        make_task("sparse_parity", {"n": 8, "k": 2, "support": [1, 1]}, 0)


# ---------------------------------------------------------------------------------------------- runner
def pattern_spec(seed: int = 0) -> ObjectSpec:
    return ObjectSpec(task="sparse_parity",
                      task_cfg={"n": 8, "k": 2, "support": [2, 5], "rho": 0.5, "width": 16, "test_size": 256,
                                "exact_metrics": True},
                      seed=seed, train=TrainConfig(steps=300, batch_size=16, optimizer="sgd", momentum=0.0, lr=0.1,
                                                   n_log_evals=12, n_lin_evals=0, ntk_every=0))


PRI = RIConfig(enabled=True, n_probe=1024, n_points=8, mode="patterns")


def test_pattern_mode_records_logodds_and_probe_counts(tmp_path):
    res = run_object(pattern_spec(), work_dir=tmp_path / "a", ri=PRI)
    assert res.status == "done" and res.ri_curve is None
    pat = res.patterns
    n = pat["steps"].size
    assert n >= 6 and pat["logodds"].shape == (n, 256) and pat["counts"].shape == (n, 256)
    assert np.all(pat["counts"].sum(axis=1) == 1024)
    assert np.allclose(pat["pmf_D"], chain_pmf(8, 0.5)) and pat["truth"].shape == (256,)
    # the last row is the final network; the light-curve exact loss at the last step agrees with it
    lc = res.light_curve
    last = int(np.flatnonzero(lc["step"] == pat["steps"][-1])[0])
    assert lc["x_lossD"][last] == pytest.approx(binary_losses(pat["logodds"][-1], pat["truth"], pat["pmf_D"])[0])
    # probe counts depend on (seed, step) only
    again = run_object(pattern_spec(), work_dir=tmp_path / "b", ri=PRI)
    assert np.array_equal(again.patterns["counts"], pat["counts"])
    assert np.array_equal(again.patterns["logodds"], pat["logodds"])
    bad = RIConfig(enabled=True, n_points=4, mode="patterns")
    out = run_object(ObjectSpec(task="sparse_parity", task_cfg={"n": 20, "k": 2}, seed=0,
                                train=TrainConfig(steps=10)), work_dir=tmp_path / "c", ri=bad)
    assert out.status == "failed" and "n <= 12" in out.error
    unsupported = run_object(ObjectSpec(task="modular_addition", task_cfg={}, seed=0, train=TrainConfig(steps=10)),
                             work_dir=tmp_path / "d", ri=bad)
    assert unsupported.status == "failed" and "does not support" in unsupported.error


def test_pattern_mode_resumes_exactly(tmp_path):
    ref = run_object(pattern_spec(), work_dir=tmp_path / "ref", ri=PRI)
    progress = []
    first = run_object(pattern_spec(), work_dir=tmp_path / "r", ri=PRI, on_progress=lambda s, t: progress.append(s),
                       stop_flag=lambda: len(progress) >= 4)
    assert first.status == "interrupted" and (tmp_path / "r" / RESUME_FILE).exists()
    second = run_object(pattern_spec(), work_dir=tmp_path / "r", ri=PRI)
    assert second.status == "done"
    for key in ("steps", "logodds", "counts"):
        assert np.array_equal(second.patterns[key], ref.patterns[key])


# ---------------------------------------------------------------------------------------------- worker
def test_worker_records_the_pattern_file(tmp_path):
    from survey_env import make_root, tiny_config

    from tycho.survey.catalog import Catalog
    from tycho.survey.worker import PATTERNS_TAG, load_sky, process_object

    sky_dict = {"version": "v2a-test", "description": "pattern mode", "regions": [{
        "name": "pc", "task": "sparse_parity", "seeds": 1, "seed_base": 200, "data_seed_from_seed": True,
        "ri_policy": "always", "ri": {"n_probe": 512, "n_points": 5, "mode": "patterns"},
        "fixed": {"n": 8, "k": 2, "support": [2, 5], "width": 16, "test_size": 128, "exact_metrics": True},
        "train": {"steps": 100, "batch_size": 16, "optimizer": "sgd", "momentum": 0.0, "lr": 0.1,
                  "n_log_evals": 8, "n_lin_evals": 0, "ntk_every": 0},
        "knobs": {"rho": [0.5]}}]}
    paths = make_root(tmp_path / "root", sky=sky_dict)
    cfg = tiny_config(paths)
    sky = load_sky(paths, cfg)
    spec = sky.make_spec("pc", {"rho": 0.5}, 0)
    assert spec.seed == 200 and spec.task_cfg["data_seed"] == 200
    catalog = Catalog(paths.db, root=paths.root)
    try:
        catalog.add_objects([spec], region="pc", source="coarse")
        claimed = catalog.claim_next("w0")
        process_object(catalog, paths, claimed, cfg, lambda: False, worker_id="w0", sky=sky)
        oid = spec.object_id()
        assert catalog.get_object(oid)["status"] == "done"
        path = paths.ri_path(oid, PATTERNS_TAG)
        with np.load(path) as f:
            assert f["logodds"].shape[1] == 256 and f["counts"].sum(axis=1).tolist() == [512] * f["steps"].size
            assert list(f["input_names"]) == [f"x{i}" for i in range(8)]
        assert catalog.get_object(oid)["ri_runs"] == []
        # a data release carries the pattern file
        import tarfile

        import pandas as pd

        from tycho.survey.release import make_release
        target = make_release(catalog, paths, "DRt", cfg=cfg)
        with tarfile.open(target / "patterns.tar.gz") as tar:
            assert tar.getnames() == [f"patterns/{oid}.npz"]
        objects = pd.read_parquet(target / "objects.parquet")
        assert list(objects["patterns_file"]) == [f"patterns/{oid}.npz"]
        assert "patterns.tar.gz" in (target / "README.md").read_text()
    finally:
        catalog.close()
