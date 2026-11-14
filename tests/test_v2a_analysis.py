"""Tests of the sky v2 Part A offline analysis (tycho/analysis/v2a.py; docs/results/v2a_preregistration.md)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from tycho.analysis import v2a
from tycho.ri.cells import cell_stats, eta_information, projection_information
from tycho.tasks.patterns import chain_pmf, pattern_bits


def _network(d: int, rng: np.random.Generator, scale: float = 2.0) -> np.ndarray:
    """Log-odds of a random function with terms up to order 3 (a stand-in for a network)."""
    s = 2.0 * pattern_bits(d) - 1.0
    z = rng.normal(scale=0.3)
    for r in (1, 2, 3):
        for B in itertools.combinations(range(d), r):
            z = z + rng.normal(scale=scale / r) * np.prod(s[:, list(B)], axis=1)
    return z


def _library(m: np.ndarray, pmf: np.ndarray, sets, d: int) -> np.ndarray:
    """(2 tables M/C, 3 values, sets) from the task statement's construction, through the row-based library path:
    one row per pattern with the pmf as row weights (counts = Σ pmf, sums = Σ pmf·m)."""
    X = pattern_bits(d)
    p1 = (1.0 + m) / 2.0
    full = cell_stats(X, p1, [tuple(range(d))], weights=pmf)[0]
    out = np.empty((2, 3, len(sets)))
    for j, A in enumerate(sets):
        marg = cell_stats(X, p1, [A], weights=pmf)[0]
        for t, (cs, pos) in enumerate(((marg, range(len(A))), (full, A))):
            out[t, 0, j] = projection_information(cs, pos, link="logit")
            out[t, 1, j] = projection_information(cs, pos, link="linear", eta_value="bounded")
            out[t, 2, j] = projection_information(cs, pos, link="linear", eta_value="gaussian")
    return out


# ------------------------------------------------------------------------------------------------ Walsh basis
def test_walsh_basis_and_fwht_match_the_chi_definition():
    for k in (1, 3, 5):
        H = v2a.walsh_matrix(k)
        s = 2.0 * pattern_bits(k) - 1.0
        for B in range(1 << k):
            members = [j for j in range(k) if B >> j & 1]
            np.testing.assert_array_equal(H[:, B], np.prod(s[:, members], axis=1) if members else 1.0)
        v = np.random.default_rng(k).normal(size=(3, 1 << k))
        np.testing.assert_allclose(v2a.walsh_coefficients(v), v @ H / (1 << k), atol=1e-14)
        np.testing.assert_allclose(H.T @ H, (1 << k) * np.eye(1 << k))
    assert len(v2a.all_sets(6)) == 41 and len(v2a.all_sets(8)) == 92


# ------------------------------------------------------------------------------------------------ measures
@pytest.mark.parametrize("d, rho", [(4, 0.6), (5, -0.5)])
def test_measures_equal_projection_information_on_the_preregistered_tables(d, rho):
    rng = np.random.default_rng(d)
    logodds = np.stack([_network(d, rng), _network(d, rng, scale=6.0)])  # the second is near-saturated
    m = v2a.clipped_mean(logodds)
    pmf_D, pmf_U = chain_pmf(d, rho), np.full(1 << d, 1.0 / (1 << d))
    sets = v2a.all_sets(d)
    got = v2a.object_measures(logodds, pmf_D, pmf_U, None, sets)
    for t in range(2):
        for pmf, (iM, iC) in ((pmf_D, (0, 2)), (pmf_U, (1, 3))):
            ref = _library(m[t], pmf, sets, d)
            for vi, val in enumerate(v2a.VALUES):
                np.testing.assert_allclose(got[val][iM, t], ref[0, vi], atol=1e-10, rtol=0)
                np.testing.assert_allclose(got[val][iC, t], ref[1, vi], atol=1e-10, rtol=0)


def test_sampled_theta_with_counts_as_weights_matches_the_probe_rows_including_empty_cells():
    d = 5
    rng = np.random.default_rng(7)
    logodds = _network(d, rng, scale=4.0)
    counts = rng.multinomial(300, chain_pmf(d, 0.8))  # many empty cells
    assert np.sum(counts == 0) > 3
    sets = v2a.all_sets(d)
    got = v2a.object_measures(logodds[None], chain_pmf(d, 0.8), np.full(32, 1 / 32), counts[None], sets)
    lib = v2a.library_sampled_theta(logodds, counts, sets)
    np.testing.assert_allclose(got["theta_sampled"][:, 0], lib, atol=1e-9, rtol=0)


def test_m_u_top_set_is_eta_information_of_the_uniform_walsh_coefficient_and_d_equals_u_when_d_is_uniform():
    d = 6
    rng = np.random.default_rng(3)
    logodds = _network(d, rng)[None]
    m = v2a.clipped_mean(logodds)
    sets = v2a.all_sets(d)
    U = np.full(1 << d, 1.0 / (1 << d))
    got = v2a.object_measures(logodds, U, U, None, sets)
    fu = v2a.walsh_coefficients(m)[0]
    for j, A in enumerate(sets):
        assert got["eta_b"][1, 0, j] == pytest.approx(float(eta_information(fu[v2a.mask_of(A)])), abs=1e-12)
    # IC-4: with D uniform, every D-weighted measure equals its uniform counterpart
    for val in v2a.VALUES:
        assert np.max(np.abs(got[val][0] - got[val][1])) <= 1e-12
        assert np.max(np.abs(got[val][2] - got[val][3])) <= 1e-12


def test_fit_falls_back_to_the_zero_start_when_the_warm_start_stalls():
    """A near-saturated function under the concentrated D of "1:2.0;1,2,3:2.0" (the case that stalled a warm start
    on pilot data): every θ at C_D still equals projection_information."""
    d = 6
    D = v2a.exact_markov_pmf("1:2.0;1,2,3:2.0", "chain")
    s = 2.0 * pattern_bits(d) - 1.0
    rng = np.random.default_rng(11)
    logodds = 2 * (2.0 * s[:, 0] + 2.0 * s[:, 0] * s[:, 1] * s[:, 2]) + 3.0 * rng.normal(size=1 << d)
    m = v2a.clipped_mean(logodds)
    sets = v2a.all_sets(d)
    # the fallback is exercised: some lo-model warm starts (hi solution without χ_A) do not converge in 50 iterations
    W, T = (D / D.sum())[None], ((1.0 + m) / 2.0)[None]
    beta_full = v2a.walsh_coefficients(2.0 * np.arctanh(m))[None]
    masks, stalled = np.arange(1 << d), 0
    for A in sets:
        a = v2a.mask_of(A)
        lo = masks[(masks & a) != a]
        hi = np.sort(np.append(lo, a))
        _, b_hi, _ = v2a._fit(d, hi, W, T, beta_full[:, hi])
        warm = b_hi[:, hi != a]
        if v2a._ce(d, lo, warm, W, T)[0] < math.log(2):
            stalled += int(not v2a._newton(d, lo, W, T, warm, max_iter=50)[3][0])
    assert stalled >= 1
    got = v2a.conditional_measures(m[None], [D], sets)[0]["theta"][0]
    X = pattern_bits(d)
    full = cell_stats(X, (1.0 + m) / 2.0, [tuple(range(d))], weights=D)[0]
    ref = np.array([projection_information(full, A, link="logit") for A in sets])
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0)


def test_support_and_spurious_sets():
    assert v2a.support_inputs("markov_lang", {"couplings": "1:1.0;1,2,3:1.0"}) == (0, 1, 2)
    assert v2a.support_inputs("markov_lang", {"couplings": "1,2,3,5:0.5"}) == (0, 1, 2, 4)
    assert v2a.support_inputs("sparse_parity", {"support": [3, 1, 6]}) == (1, 3, 6)
    r = v2a.Run(rec={"object_id": "x", "task": "markov_lang", "region": "markov-v2", "seed": 1,
                     "task_cfg": {"couplings": "1:1.0", "train_context": "chain"}}, z={}, support=(0,),
                sets=v2a.all_sets(6))
    assert r.spurious(2).sum() == 35 and r.spurious(1).sum() == 40


# ------------------------------------------------------------------------------------------------ §4 rules
def test_window_rule():
    steps = np.array([0, 1, 2, 3, 4])
    assert v2a.window_start(steps, np.array([1.0, 0.6, 0.5, 0.7, 0.4])) == 2  # ≤ ½ loss(0); then to the end
    assert v2a.window_start(steps, np.array([1.0, 0.9, 0.8, 0.7, 0.51])) is None
    assert v2a.window_start(steps, np.array([0.0, 0.0, 0.0, 0.0, 0.0])) == 1  # step 0 is never in the window
    with pytest.raises(ValueError):
        v2a.window_start(np.array([1, 2]), np.array([1.0, 0.1]))


def test_spearman_ties_and_constant_predictor_rule():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 40).astype(float)
    y = x + rng.integers(0, 3, 40)
    assert v2a.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)
    assert v2a.spearman(np.full(10, 0.3), np.arange(10.0)) == 0.0
    assert v2a.spearman_info(np.full(10, 0.3), np.arange(10.0))[1] == "constant predictor"
    assert v2a.spearman([1.0], [2.0]) == 0.0
    assert v2a.spearman(np.arange(5.0), -np.arange(5.0)) == pytest.approx(-1.0)  # signed


def test_weighted_midranks_equal_ranks_of_the_duplicated_sample():
    rng = np.random.default_rng(1)
    v = rng.integers(0, 6, 12).astype(float)
    W = rng.integers(0, 3, size=(5, 12)).astype(float)
    R = v2a._midranks(v2a._rank_prep(v), W)
    for b in range(5):
        dup = np.repeat(np.arange(12), W[b].astype(int))
        ranks = rankdata(v[dup])
        for i in np.unique(dup):
            assert np.all(ranks[dup == i] == R[b, i])


def test_bootstrap_p_value_formula_and_holm():
    assert v2a.boot_pvalue(np.array([-1.0, 0.0, 1.0, 2.0])) == 1.0  # 2·min(2+1, 3+1)/5 capped at 1
    assert v2a.boot_pvalue(np.arange(1.0, 10.0)) == pytest.approx(2 * 1 / 10)  # B = 9, none ≤ 0
    assert v2a.boot_pvalue(-np.arange(1.0, 20.0)) == pytest.approx(2 * 1 / 20)
    assert v2a.boot_pvalue(np.zeros(9)) == 1.0  # zeros count on both sides
    np.testing.assert_allclose(v2a.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(v2a.holm([0.5, 0.9]), [1.0, 1.0])


def test_joint_seed_resampling_moves_a_seeds_runs_together_across_configurations():
    rng = np.random.default_rng(2)
    seeds = [100, 101, 102, 103]
    cfgs = []
    for name, seeds_in in (("A", [0, 1, 2, 3]), ("B", [0, 1, 3])):  # seed 102 excluded (IC-5) in B
        c = v2a.Cfg(name=name)
        for si in seeds_in:
            n = 4
            c.add(si, f"{name}{si}", np.arange(n), {"k": rng.normal(size=n)}, rng.normal(size=n), rng.normal(size=n))
        cfgs.append(c.finalize())
    mult = v2a.resample_multiplicities(len(seeds), 300, 20261114)
    np.testing.assert_array_equal(mult.sum(axis=1), len(seeds))
    np.testing.assert_array_equal(mult, v2a.resample_multiplicities(len(seeds), 300, 20261114))
    rho, _ = v2a._boot_rhos(cfgs, ["k"], mult)
    for b in range(0, 300, 37):
        for ci, c in enumerate(cfgs):  # the same multiset of seed labels in both configurations
            dup = np.concatenate([np.flatnonzero(c.seed_idx == si).repeat(int(mult[b, si]))
                                  for si in range(len(seeds))]).astype(int)
            assert rho["k"][ci, b] == pytest.approx(v2a.spearman(c.X["k"][dup], c.Y[dup]), abs=1e-12)
    # an all-ones resample is the original data
    one, _ = v2a._boot_rhos(cfgs, ["k"], np.ones((1, len(seeds))))
    for ci, c in enumerate(cfgs):
        assert one["k"][ci, 0] == pytest.approx(v2a.spearman(c.X["k"], c.Y), abs=1e-12)


def test_identifiability_numbers_match_section_4():
    rows = {r["twin_set"]: r for r in v2a.identifiability_table()}
    n = 0
    for (wi, ref), printed in v2a.PREREG_IDENT.items():
        wo = "1:1.0" if wi.startswith("1:1.0") else "1:2.0"
        row = rows[f"{wo} vs {wi} | chain"]
        match = v2a.prereg_matches(row[ref], printed)
        assert match["double_rounding"], (wi, ref, row[ref], printed)
        assert row[f"enters_{ref}"] == (printed >= 0.01)
        n += match["single_rounding"]
    assert n == 7  # 0.0061477 (C_D, "1:1.0;1,2,3:1.0") is printed 0.0062: 0.00615 rounded again
    # exclusions as printed: C_D keeps only "1:1.0;1,3:0.8" among chain twin sets; M_D excludes only 1,2,3:2.0
    chain = [r for r in rows.values() if r["twin_set"].endswith("chain")]
    assert [r["twin_set"] for r in chain if r["enters_C_D"]] == ["1:1.0 vs 1:1.0;1,3:0.8 | chain"]
    assert [r["twin_set"] for r in chain if not r["enters_M_D"]] == ["1:2.0 vs 1:2.0;1,2,3:2.0 | chain"]
    for r in rows.values():  # residual variance 1 at the uniform references and for iid twins
        assert r["M_U"] == pytest.approx(1.0) and r["C_U"] == pytest.approx(1.0)
        if r["twin_set"].endswith("iid"):
            assert r["M_D"] == pytest.approx(1.0) and r["C_D"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------ pipeline wiring
def _fake_run(couplings: str, seed: int, rng: np.random.Generator, *, ic5_kl: float = 0.001) -> v2a.Run:
    """A chain Markov run whose spurious-set sums are +Y for θ and −Y for both η values at every reference."""
    n_ck = 12
    sets = v2a.all_sets(6)
    support = v2a.support_inputs("markov_lang", {"couplings": couplings})
    kl = np.empty((3, n_ck))
    kl[0] = np.geomspace(1.0, ic5_kl, n_ck)
    Y = np.cumsum(rng.uniform(0.1, 1.0, n_ck)) * 0.01
    kl[2] = kl[0] + Y
    kl[1] = kl[0]
    rec = {"object_id": f"{couplings}-{seed}", "task": "markov_lang", "region": "markov-v2", "seed": seed,
           "task_cfg": {"couplings": couplings, "train_context": "chain"}}
    r = v2a.Run(rec=rec, z={}, support=support, sets=sets)
    sp2 = r.spurious(2)
    z = {"steps": np.arange(n_ck), "kl": kl, "sq": kl * 0.5}
    for val, sign in (("theta", 1.0), ("eta_b", -1.0), ("eta_g", -1.0)):
        a = np.zeros((4, n_ck, len(sets)))
        a[:, :, np.flatnonzero(sp2)[0]] = sign * Y + 1.0
        z[val] = a
    r.z = z
    r.ic5_value, r.ic5_pass = float(kl[0, -1]), bool(kl[0, -1] < v2a.IC5_KL)
    r.win = v2a.window_start(z["steps"], kl[0])
    return r


def test_t1_statistic_bootstrap_and_exclusions_on_a_known_answer():
    rng = np.random.default_rng(5)
    seeds = list(range(100, 106))
    runs = [_fake_run(c, s, rng) for c in ("1:1.0", "1,3:1.0") for s in seeds]
    runs += [_fake_run("1:2.0", s, rng, ic5_kl=0.05 if s < 102 else 0.001) for s in seeds]  # 4 seeds pass IC-5
    cfgs = v2a.build_t1(runs, "T1_markov", seeds)
    assert [c.n_valid for c in cfgs] == [6, 6, 4]
    comps = v2a.evaluate_family("T1_markov", cfgs, seeds, B=199, min_seeds=5)
    for c in comps:
        assert c["configurations"] == ["1,3:1.0 | chain", "1:1.0 | chain"]  # "1:2.0" dropped (4 < 5 seeds)
        assert c["dropped"][0]["name"] == "1:2.0 | chain"
        assert c["rhobar_theta"] == pytest.approx(1.0) and c["rhobar_eta_b"] == pytest.approx(-1.0)
        assert c["delta_eta_b"] == pytest.approx(2.0) and c["delta_eta_g"] == pytest.approx(2.0)
        assert c["p_raw_eta_b"] == pytest.approx(2 / 200)  # every Δ* > 0
        assert c["secondary"]["pooled_Y_theta"] == pytest.approx(1.0)


def _comp(name: str, db: float, dg: float, ds: float, p: float = 1e-4) -> dict:
    return {"comparison": name, "delta_eta_b": db, "delta_eta_g": dg, "delta_sens_eta_b": ds,
            "p_raw_eta_b": p, "p_raw_eta_g": p}


def test_decision_table_and_labels():
    def run(special):
        comps = [_comp(f"c{i}", 0.01, 0.01, 0.01, p=0.5) for i in range(12)]
        comps[:len(special)] = special
        return v2a.apply_decision(comps), comps
    d, comps = run([_comp("a", 0.2, 0.15, 0.05)])
    assert d["outcome"] == "gate 2 passed (θ), behavioural referees" and comps[0]["label"] == "θ wins"
    d, _ = run([_comp("a", -0.2, -0.15, -0.05)])
    assert d["outcome"] == "η wins"
    d, _ = run([_comp("a", 0.2, 0.15, 0.05), _comp("b", -0.2, -0.15, -0.05)])
    assert d["outcome"] == "mixed"
    d, comps = run([_comp("a", 0.2, 0.05, 0.05)])  # Gaussian margin not met
    assert d["outcome"] == "tie" and comps[0]["conditions"]["theta"]["value_dependent"]
    assert comps[0]["label"].startswith("value-dependent")
    d, comps = run([_comp("a", -0.2, -0.3, 0.02)])  # sensitivity outcome favours θ
    assert d["outcome"] == "tie" and comps[0]["conditions"]["eta"]["outcome_dependent"]
    d, comps = run([_comp("a", 0.2, 0.15, 0.05, p=0.01)])  # Holm over 12: 0.12 ≥ 0.05
    assert d["outcome"] == "tie" and comps[0]["p_holm_eta_b"] == pytest.approx(0.12)
    d, comps = run([_comp("a", math.nan, math.nan, math.nan)])
    assert comps[0]["label"] == "not computable" and d["outcome"] == "tie"


def test_ic1_sample_is_five_percent_of_each_preregistered_region_and_deterministic():
    recs = [{"object_id": f"{i:016x}", "task": "markov_lang", "region": "markov-v2", "status": "done"}
            for i in range(320)]
    recs += [{"object_id": f"{1000 + i:016x}", "task": "sparse_parity", "region": f"parity-corr-k{2 + i % 2}",
              "status": "done"} for i in range(120)]
    sel = v2a.select_replays(recs, 0.05)
    assert sum(r["task"] == "markov_lang" for r in sel) == 16 and sum(r["task"] == "sparse_parity" for r in sel) == 6
    assert [r["object_id"] for r in sel] == [r["object_id"] for r in v2a.select_replays(recs, 0.05)]


def test_stage1_object_end_to_end_on_a_synthetic_parity_object(tmp_path):
    from tycho.tasks.base import torch_generator
    from tycho.tasks.patterns import binary_losses, sample_codes

    d, seed, n_probe = 4, 7, 512
    rng = np.random.default_rng(0)
    steps = np.array([0, 1, 3, 10, 30])
    pmf_D, pmf_U, pmf_S = chain_pmf(d, 0.5), chain_pmf(d, 0.0), chain_pmf(d, -0.5)
    spins = 2 * pattern_bits(d) - 1
    truth = (spins[:, 1] * spins[:, 3] < 0).astype(float)
    logodds = np.stack([(t / 10.0) * 4 * (2 * truth - 1) + rng.normal(scale=0.3, size=1 << d) for t in steps])
    counts = np.stack([np.bincount(sample_codes(pmf_D, n_probe, torch_generator(seed, "ri", int(s))), minlength=16)
                       for s in steps])
    np.savez(tmp_path / "p.npz", steps=steps, logodds=logodds, counts=counts, pmf_D=pmf_D, pmf_U=pmf_U, pmf_S=pmf_S,
             truth=truth, input_names=np.array([f"x{i}" for i in range(d)]), n_probe=np.int64(n_probe))
    rec = {"object_id": "00000000000000aa", "region": "parity-corr-k2", "task": "sparse_parity", "seed": seed,
           "task_cfg": {"k": 2, "support": [1, 3], "rho": 0.5}, "status": "done", "patterns": str(tmp_path / "p.npz")}
    cache = tmp_path / "cache"
    first = v2a.stage1_object(rec, cache, ic2a_fraction=0.4)
    assert not first["cached"] and v2a.stage1_object(rec, cache, ic2a_fraction=0.4)["cached"]
    z = dict(np.load(cache / f"{rec['object_id']}.npz"))
    assert z["theta"].shape == (4, 5, 14) and z["theta_sampled"].shape == (2, 5, 14)
    sel = z["ic2a_idx"]
    assert sel.size == 2 and np.all(z["ic2a_regen_ok"])
    np.testing.assert_allclose(z["theta_sampled"][:, sel], np.transpose(z["ic2a_lib"], (1, 0, 2)), atol=1e-9)
    np.testing.assert_allclose(z["theta"][0], z["local_ri"], atol=1e-9)
    assert z["kl"][2, -1] == pytest.approx(binary_losses(logodds[-1], truth, pmf_S)[0])
    runs = v2a.build_runs([rec], cache)
    assert runs[0].ic5_pass and runs[0].win is not None
    ic = v2a.ic_results(runs, None)
    assert ic["IC-2a"]["passed"] and ic["IC-4"]["passed_local_ri"]
