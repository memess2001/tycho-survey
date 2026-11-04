"""Tests of tycho.ri.refined: chain decompositions, greedy selection and local RI (exact fibre solver + batched IPF)."""
from __future__ import annotations

import time
from itertools import combinations, product

import numpy as np
import pytest

from tycho.ri.hypergraph import graded_order
from tycho.ri.loglinear import kl, loglinear_table, m_project, marginal, mutual_information, total_correlation
from tycho.ri.refined import (
    chain_decomposition,
    graded_chain,
    greedy_selection,
    local_projection_batch,
    local_ri,
    local_ri_batch,
    local_ri_from_table,
)

LOG2 = float(np.log(2.0))


def random_table(rng: np.random.Generator, cards, conc: float = 1.0) -> np.ndarray:
    return rng.dirichlet(np.full(int(np.prod(cards)), conc)).reshape(cards)


def boundary_generators(m: int) -> list[tuple[int, ...]]:
    return [tuple(i for i in range(m) if i != j) for j in range(m)]


def ri_by_projection(P: np.ndarray) -> float:
    """Local RI computed independently: per-table m_project onto ∂S, then KL."""
    return kl(P, m_project(P, boundary_generators(P.ndim)).Q)


def plain_ipf_kl(P: np.ndarray, sweeps: int) -> float:
    """Reference: KL from P to plain IPF (uniform start, no facial reduction, extended precision) onto ∂S.

    Every IPF iterate lies in the model, so this is an upper bound on the local RI that decreases to it.
    """
    Pl = np.asarray(P, dtype=np.longdouble)
    m = P.ndim
    targets = [Pl.sum(axis=j, keepdims=True) for j in range(m)]
    Q = np.full(P.shape, 1.0 / P.size, dtype=np.longdouble)
    for _ in range(sweeps):
        for j in range(m):
            cur = Q.sum(axis=j, keepdims=True)
            ratio = np.zeros_like(cur)
            np.divide(targets[j], cur, out=ratio, where=cur > 0)
            Q *= ratio
    return kl(P, Q.astype(np.float64))


def deterministic_table(fn, cards_in, k_out=2) -> np.ndarray:
    """Uniform inputs, output = fn(inputs)."""
    P = np.zeros(tuple(cards_in) + (k_out,))
    for x in product(*[range(k) for k in cards_in]):
        P[x + (fn(*x),)] = 1.0
    return P / P.sum()


def parity_table(m: int) -> np.ndarray:
    P = np.zeros((2,) * m)
    for x in product(range(2), repeat=m):
        if sum(x) % 2 == 0:
            P[x] = 1.0
    return P / P.sum()


# ----------------------------------------------------------------------------------------------- local RI
def test_local_ri_of_two_variables_is_mutual_information():
    rng = np.random.default_rng(0)
    for cards in [(3, 4), (2, 2), (5, 2)]:
        P = random_table(rng, cards)
        assert local_ri(P) == pytest.approx(mutual_information(P, (0,), (1,)), abs=1e-14)
    P4 = random_table(rng, (2, 3, 2, 2))
    pairs = list(combinations(range(4), 2))
    vals = local_ri_from_table(P4, pairs)
    for (a, b), v in zip(pairs, vals):
        assert v == pytest.approx(mutual_information(P4, (a,), (b,)), abs=1e-14)


def test_local_ri_from_table_matches_margins():
    rng = np.random.default_rng(1)
    P = random_table(rng, (2, 3, 2, 2, 2))
    subsets = [(0,), (1, 3), (0, 1, 2), (2, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4), (0, 1, 2, 3, 4)]
    vals = local_ri_from_table(P, subsets)
    assert vals[0] == 0.0
    for S, v in zip(subsets[1:], vals[1:]):
        assert v == pytest.approx(ri_by_projection(marginal(P, S)), abs=1e-12)
    with pytest.raises(ValueError):
        local_ri_from_table(P, [()])
    with pytest.raises(ValueError):
        local_ri_from_table(P, [(0, 5)])


def test_pure_parity_three_and_four_bits():
    P3 = parity_table(3)  # uniform on even parity
    assert local_ri(P3) == pytest.approx(LOG2, abs=1e-14)
    assert local_ri(P3, method="ipf") == pytest.approx(LOG2, abs=1e-12)
    np.testing.assert_allclose(local_ri_from_table(P3, [(0, 1), (0, 2), (1, 2)]), 0.0, atol=1e-15)
    P4 = parity_table(4)
    lower = graded_order(4, 2, 3)
    np.testing.assert_allclose(local_ri_from_table(P4, lower), 0.0, atol=1e-15)
    assert local_ri(P4) == pytest.approx(LOG2, abs=1e-14)
    # the whole total correlation sits in the top-order term
    assert total_correlation(P4) == pytest.approx(LOG2, abs=1e-14)


@pytest.mark.parametrize("name,fn", [("and", lambda a, b: a & b), ("or", lambda a, b: a | b)])
def test_deterministic_and_or_boundary(name, fn):
    """AND/OR are linearly separable: they lie in the closure of the pairwise model, so the triple's RI is 0 exactly.

    The tables have zeros and the projection is a boundary point, where plain IPF converges only sublinearly; the
    reference (long, extended-precision plain IPF) is an upper bound that must approach our exact value.
    """
    P = deterministic_table(fn, (2, 2))
    exact_fiber = local_ri(P)
    exact_ipf = local_ri(P, method="ipf")
    assert exact_fiber == 0.0 and exact_ipf == 0.0
    ref_short, ref_long = plain_ipf_kl(P, 2000), plain_ipf_kl(P, 8000)
    assert 0.0 <= exact_fiber <= ref_long + 1e-15
    assert ref_long < ref_short and ref_long < 2e-5
    # its pairs carry the information: RI(x_i, y) = MI, and the chain total is the total correlation
    vals = local_ri_from_table(P, [(0, 2), (1, 2)])
    assert np.all(np.isfinite(vals)) and np.all(vals > 0.1)
    cd = chain_decomposition(P, graded_chain(3))
    assert cd.total == pytest.approx(total_correlation(P), abs=1e-12)
    assert cd.as_dict()[(0, 1, 2)] == pytest.approx(0.0, abs=1e-15)
    assert all(np.isfinite(s.ri) for s in cd.steps)


def test_three_input_and_and_noisy_relations_against_long_ipf():
    P = deterministic_table(lambda a, b, c: a & b & c, (2, 2, 2))
    assert local_ri(P) == 0.0  # also linearly separable
    # a deterministic relation with an interior solution: XOR of three inputs has RI log 2 at order 4
    X = deterministic_table(lambda a, b, c: a ^ b ^ c, (2, 2, 2))
    assert local_ri(X) == pytest.approx(LOG2, abs=1e-14)
    # noisy AND: strictly positive, near the boundary; the exact solver must agree with a long IPF
    noisy = 0.98 * deterministic_table(lambda a, b: a & b, (2, 2)) + 0.02 / 8
    exact = local_ri(noisy)
    assert exact == pytest.approx(plain_ipf_kl(noisy, 3000), abs=1e-12)
    assert exact == pytest.approx(local_ri(noisy, method="ipf"), abs=1e-13)


@pytest.mark.parametrize("name,fn,expect_zero", [
    ("mod2", lambda a, b: (a + b) % 2, False),  # parity-like: interior projection, RI > 0
    ("threshold", lambda a, b: int(a + b >= 2), True),  # separable: boundary point in the closure, RI = 0
])
def test_non_binary_deterministic_relations(name, fn, expect_zero):
    P = deterministic_table(fn, (3, 2))
    ours = local_ri(P)  # IPF path with the exact facial start
    assert np.isfinite(ours)
    ref = plain_ipf_kl(P, 4000)
    assert ours <= ref + 1e-14
    if expect_zero:
        assert ours == pytest.approx(0.0, abs=1e-15)
        assert 0.0 < ref < 1e-3  # plain IPF still creeping towards 0
    else:
        assert ours > 0.1
        assert ours == pytest.approx(ref, abs=1e-10)


def test_batched_equals_per_table_binary_and_mixed():
    rng = np.random.default_rng(2)
    tables = rng.dirichlet(np.full(16, 0.8), size=60).reshape(60, 2, 2, 2, 2)
    batch = local_ri_batch(tables)
    per = np.array([ri_by_projection(t) for t in tables])
    np.testing.assert_allclose(batch, per, atol=1e-12)
    np.testing.assert_allclose(local_ri_batch(tables, method="ipf"), batch, atol=1e-12)
    mixed = rng.dirichlet(np.full(24, 0.8), size=40).reshape(40, 3, 2, 2, 2)
    np.testing.assert_allclose(local_ri_batch(mixed), [ri_by_projection(t) for t in mixed], atol=1e-12)
    # projections returned by the batch have the generator margins of the input (IPF and fibre alike)
    for tabs in (tables, mixed):
        Q, _ = local_projection_batch(tabs)
        P = tabs / tabs.reshape(len(tabs), -1).sum(axis=1).reshape((-1,) + (1,) * (tabs.ndim - 1))
        for j in range(1, tabs.ndim):
            np.testing.assert_allclose(Q.sum(axis=j), P.sum(axis=j), atol=1e-12)


def test_sparse_and_extreme_tables_are_finite_and_consistent():
    rng = np.random.default_rng(3)
    counts = np.stack([rng.multinomial(30, rng.dirichlet(np.full(8, 0.3))) for _ in range(300)])
    tables = counts.reshape(300, 2, 2, 2).astype(float)
    fib = local_ri_batch(tables)
    ipf = local_ri_batch(tables, method="ipf")
    assert np.all(np.isfinite(fib)) and np.all(fib >= 0)
    np.testing.assert_allclose(fib, ipf, atol=1e-10)
    # cells of 1e-250 next to O(1) cells: the root sits extremely close to the end of the fibre
    P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2)
    P[0, 0, 1], P[0, 1, 0] = 1e-200, 1e-250
    val = local_ri(P)
    assert np.isfinite(val) and val == pytest.approx(local_ri(P, method="ipf"), abs=1e-10)


def test_local_ri_is_invariant_to_relabelling_and_axis_order():
    rng = np.random.default_rng(4)
    P = random_table(rng, (2, 3, 2))
    base = local_ri(P)
    assert local_ri(P.transpose(2, 0, 1)) == pytest.approx(base, abs=1e-12)
    assert local_ri(P[:, ::-1, :]) == pytest.approx(base, abs=1e-12)
    assert local_ri(P[::-1, :, ::-1]) == pytest.approx(base, abs=1e-12)


def test_edge_cases():
    rng = np.random.default_rng(5)
    assert local_ri(np.array([0.2, 0.8])) == 0.0
    with pytest.raises(ValueError):
        local_ri(np.array(1.0))
    card1 = random_table(rng, (2, 1, 3)).reshape(2, 1, 3)
    assert local_ri(card1) == pytest.approx(0.0, abs=1e-15)
    single = np.zeros((2, 2, 2, 2))
    single[1, 0, 1, 1] = 1.0
    assert local_ri(single) == 0.0 and local_ri(single, method="ipf") == 0.0
    assert local_ri(np.zeros((3, 3, 2)) + np.eye(3)[:, :, None]) == pytest.approx(0.0, abs=1e-15)
    Q, ri = local_projection_batch(np.zeros((0, 2, 2, 2)))
    assert Q.shape == (0, 2, 2, 2) and ri.shape == (0,)
    with pytest.raises(ValueError):
        local_ri_batch(np.zeros((2, 2, 2, 2)))  # zero-mass tables
    with pytest.raises(ValueError):
        local_ri_batch(np.ones((1, 3, 2, 2)), method="fiber")
    with pytest.raises(ValueError):
        local_ri_batch(np.ones((1, 2, 2, 2)), method="newton")
    with pytest.raises(ValueError):
        local_ri_batch(np.ones(4))


# ----------------------------------------------------------------------------------------------- chains
@pytest.mark.parametrize("cards", [(2, 3, 2, 2), (2, 2, 2, 2, 2), (3, 3)])
def test_graded_chain_decomposes_total_correlation(cards):
    rng = np.random.default_rng(6)
    P = random_table(rng, cards, conc=0.6)
    cd = chain_decomposition(P, graded_chain(len(cards)))
    tc = total_correlation(P)
    assert cd.kl_base == pytest.approx(tc, abs=1e-12)
    assert cd.residual == pytest.approx(0.0, abs=1e-15)
    assert cd.total == pytest.approx(tc, abs=1e-9)
    assert len(cd.steps) == 2 ** len(cards) - len(cards) - 1
    for step in cd.steps:
        assert step.ri >= 0 and step.order == len(step.subset)
        assert step.ri == pytest.approx(step.kl_before - step.kl_after, abs=1e-9)  # Pythagoras per step
    if len(cards) == 2:
        assert cd.steps[0].ri == pytest.approx(mutual_information(P, (0,), (1,)), abs=1e-12)


def test_truncated_chain_and_custom_base():
    rng = np.random.default_rng(7)
    P = random_table(rng, (2, 2, 3, 2))
    cd = chain_decomposition(P, graded_chain(4, max_order=2))
    pairs_model = m_project(P, graded_order(4, 2, 2)).Q
    assert cd.residual == pytest.approx(kl(P, pairs_model), abs=1e-10)
    assert cd.kl_base == pytest.approx(cd.total + cd.residual, abs=1e-9)
    based = chain_decomposition(P, [(0, 1, 2)], base=[(0, 1), (0, 2), (1, 2)])
    assert based.base == ((3,), (0, 1), (0, 2), (1, 2))
    assert based.steps[0].ri == pytest.approx(local_ri(marginal(P, (0, 1, 2))), abs=1e-10)


def test_non_admissible_chain_raises():
    P = random_table(np.random.default_rng(8), (2, 2, 2))
    with pytest.raises(ValueError, match="not admissible"):
        chain_decomposition(P, [(0, 1, 2)])
    with pytest.raises(ValueError, match="not admissible"):
        chain_decomposition(P, [(0, 1), (0, 2), (0, 1, 2)])
    with pytest.raises(ValueError, match="already"):
        chain_decomposition(P, [(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="order"):
        chain_decomposition(P, [(1,)])
    with pytest.raises(ValueError):
        chain_decomposition(P, [(0, 3)])


def test_chain_dependence_of_refined_information():
    """Different orders of incomparable subsets change the per-subset RI but not the total (= TC)."""
    rng = np.random.default_rng(9)
    P = random_table(rng, (2, 3, 2), conc=0.5)
    chain_a = [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
    chain_b = [(1, 2), (0, 2), (0, 1), (0, 1, 2)]
    a = chain_decomposition(P, chain_a).as_dict()
    b = chain_decomposition(P, chain_b).as_dict()
    tc = total_correlation(P)
    assert sum(a.values()) == pytest.approx(tc, abs=1e-9)
    assert sum(b.values()) == pytest.approx(tc, abs=1e-9)
    assert abs(a[(0, 1)] - b[(0, 1)]) > 1e-3  # first in chain a, last pair in chain b
    assert a[(0, 1)] == pytest.approx(mutual_information(P, (0,), (1,)), abs=1e-12)
    # the top-order step comes after all its proper subsets in both chains: equal to the local RI
    assert a[(0, 1, 2)] == pytest.approx(b[(0, 1, 2)], abs=1e-10)
    assert a[(0, 1, 2)] == pytest.approx(local_ri(P), abs=1e-10)


def test_chain_ri_equals_local_ri_when_added_right_after_its_subsets():
    rng = np.random.default_rng(10)
    P = random_table(rng, (2, 2, 2, 3), conc=0.5)
    # (0, 1, 2) enters right after its proper subsets; nothing else involves variables 0..2 yet,
    # although (3,) interacts with nothing so far either
    cd = chain_decomposition(P, [(0, 1), (0, 2), (1, 2), (0, 1, 2), (0, 3)])
    assert cd.as_dict()[(0, 1, 2)] == pytest.approx(local_ri(marginal(P, (0, 1, 2))), abs=1e-10)


# ----------------------------------------------------------------------------------------------- greedy selection
def test_greedy_selection_complete_and_first_step_is_max_mi():
    rng = np.random.default_rng(11)
    P = random_table(rng, (2, 2, 3, 2), conc=0.5)
    gd = greedy_selection(P)
    assert gd.total == pytest.approx(total_correlation(P), abs=1e-9)
    assert gd.residual == pytest.approx(0.0, abs=1e-12)
    pairs = list(combinations(range(4), 2))
    mis = [mutual_information(P, (a,), (b,)) for a, b in pairs]
    assert gd.steps[0].subset == pairs[int(np.argmax(mis))]
    assert gd.steps[0].ri == pytest.approx(max(mis), abs=1e-12)
    again = greedy_selection(P)
    assert [s.subset for s in again.steps] == [s.subset for s in gd.steps]  # deterministic


def test_greedy_selection_reaches_pure_synergy_only_with_zero_min_gain():
    P = parity_table(3)
    gd = greedy_selection(P)
    assert [s.subset for s in gd.steps] == [(0, 1), (0, 2), (1, 2), (0, 1, 2)]  # zero-gain ties: graded-lex
    assert gd.as_dict()[(0, 1, 2)] == pytest.approx(LOG2, abs=1e-12)
    stopped = greedy_selection(P, min_gain=1e-6)
    assert stopped.steps == [] and stopped.residual == pytest.approx(LOG2, abs=1e-12)
    limited = greedy_selection(P, max_order=2)
    assert all(s.order == 2 for s in limited.steps) and limited.residual == pytest.approx(LOG2, abs=1e-12)
    assert len(greedy_selection(P, max_steps=2).steps) == 2


def test_planted_interaction_is_selected_first():
    theta = {(1, 3): 1.2, (0,): 0.3, (0, 2): 0.2}
    P = loglinear_table(theta, [2, 2, 2, 2])
    assert greedy_selection(P, max_steps=1).steps[0].subset == (1, 3)


# ----------------------------------------------------------------------------------------------- performance
def test_batch_timing_quick():
    rng = np.random.default_rng(12)
    tables = rng.dirichlet(np.ones(16), size=5000).reshape(5000, 2, 2, 2, 2)
    t0 = time.perf_counter()
    local_ri_batch(tables)
    assert time.perf_counter() - t0 < 2.0  # generous; ~0.05 s expected


@pytest.mark.slow
def test_batch_performance_target():
    """SPEC §2.4: local_ri_batch on (20_000, 2, 2, 2, 2) in < 2 s on one core (also for forced IPF on
    realistic near-model tables)."""
    rng = np.random.default_rng(13)
    tables = rng.dirichlet(np.ones(16), size=20_000).reshape(20_000, 2, 2, 2, 2)
    t0 = time.perf_counter()
    local_ri_batch(tables)
    assert time.perf_counter() - t0 < 2.0
    marg = rng.dirichlet(np.full(2, 5.0), size=(20_000, 4))
    near = np.einsum("bi,bj,bk,bl->bijkl", *[marg[:, i] for i in range(4)])
    near = rng.multinomial(4096, near.reshape(20_000, 16) / near.reshape(20_000, 16).sum(1, keepdims=True))
    near = near.reshape(20_000, 2, 2, 2, 2) + 0.5
    t0 = time.perf_counter()
    local_ri_batch(near, method="ipf")
    assert time.perf_counter() - t0 < 2.0
