"""Tests of tycho.ri.loglinear: tables, m-projection (IPF + exact facial support), information measures and
binary coordinates (spin vs 0/1 coding)."""
from __future__ import annotations

import logging
from itertools import product

import numpy as np
import pytest

from tycho.ri.hypergraph import graded_order, nonempty_subsets
from tycho.ri.loglinear import (
    entropy,
    eta_binary,
    facial_support,
    joint_table,
    kl,
    loglinear_table,
    m_project,
    marginal,
    mutual_information,
    theta_01_to_spin,
    theta_binary,
    theta_spin_to_01,
    total_correlation,
)


def random_table(rng: np.random.Generator, cards, conc: float = 1.0) -> np.ndarray:
    return rng.dirichlet(np.full(int(np.prod(cards)), conc)).reshape(cards)


def plain_ipf(P: np.ndarray, gens, sweeps: int) -> np.ndarray:
    """Reference IPF from the uniform table, without facial reduction, in extended precision."""
    P = np.asarray(P, dtype=np.longdouble)
    d = P.ndim
    others = [tuple(i for i in range(d) if i not in g) for g in gens]
    targets = [P.sum(axis=o, keepdims=True) for o in others]
    Q = np.full(P.shape, 1.0 / P.size, dtype=np.longdouble)
    for _ in range(sweeps):
        for o, t in zip(others, targets):
            cur = Q.sum(axis=o, keepdims=True)
            ratio = np.zeros_like(cur)
            np.divide(t, cur, out=ratio, where=cur > 0)
            Q *= ratio
    return Q.astype(np.float64)


def and_table() -> np.ndarray:
    P = np.zeros((2, 2, 2))
    for a, b in product(range(2), repeat=2):
        P[a, b, a & b] = 0.25
    return P


# ----------------------------------------------------------------------------------------------- tables
def test_joint_table_matches_loop_count():
    rng = np.random.default_rng(0)
    cards = [2, 3, 4]
    X = np.column_stack([rng.integers(0, k, 500) for k in cards])
    w = rng.random(500)
    T = joint_table(X, cards, weights=w, normalize=False)
    ref = np.zeros(cards)
    for x, wi in zip(X, w):
        ref[tuple(x)] += wi
    np.testing.assert_allclose(T, ref, rtol=1e-12)
    Tn = joint_table(X, cards, weights=w)
    assert Tn.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(Tn, ref / ref.sum(), rtol=1e-12)
    Tp = joint_table(X, cards, normalize=False, pseudocount=0.5)
    assert Tp.sum() == pytest.approx(500 + 0.5 * 24)


def test_joint_table_integer_weights_equal_row_duplication():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, (300, 3))
    w = rng.integers(0, 4, 300)
    dup = np.repeat(X, w, axis=0)
    np.testing.assert_allclose(joint_table(X, [2, 2, 2], weights=w), joint_table(dup, [2, 2, 2]), atol=1e-15)


def test_joint_table_soft_one_hot_equals_hard_and_ignores_soft_codes():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.integers(0, 2, 400), rng.integers(0, 3, 400), rng.integers(0, 2, 400)])
    one_hot = np.eye(3)[X[:, 1]]
    garbage = X.copy()
    garbage[:, 1] = 99  # the codes of a soft column are ignored
    np.testing.assert_allclose(joint_table(garbage, [2, 3, 2], soft={1: one_hot}), joint_table(X, [2, 3, 2]),
                               atol=1e-15)
    soft = rng.dirichlet(np.ones(3), 400)
    T = joint_table(X, [2, 3, 2], soft={1: soft}, normalize=False)
    ref = np.zeros((2, 3, 2))
    for n in range(400):
        ref[X[n, 0], :, X[n, 2]] += soft[n]
    np.testing.assert_allclose(T, ref, rtol=1e-12)


def test_joint_table_errors():
    X = np.zeros((0, 2), dtype=int)
    with pytest.raises(ValueError):
        joint_table(X, [2, 2])  # N = 0: nothing to normalise
    assert joint_table(X, [2, 2], normalize=False).sum() == 0.0
    np.testing.assert_allclose(joint_table(X, [2, 2], pseudocount=1.0), np.full((2, 2), 0.25))
    with pytest.raises(ValueError):
        joint_table(np.array([[0, 2]]), [2, 2])  # code out of range
    with pytest.raises(ValueError):
        joint_table(np.array([[0.5, 1]]), [2, 2])  # non-integer code
    with pytest.raises(ValueError):
        joint_table(np.array([[0, 1]]), [2, 2], soft={0: np.ones((1, 2)) / 2, 1: np.ones((1, 2)) / 2})
    with pytest.raises(ValueError):
        joint_table(np.array([[0, 1]]), [2, 2], soft={1: np.array([[0.5, 0.6]])})  # not a probability row
    with pytest.raises(ValueError):
        joint_table(np.array([[0, 1]]), [2, 2], weights=np.array([-1.0]))


def test_marginal_keeps_sorted_axes():
    rng = np.random.default_rng(3)
    P = random_table(rng, (2, 3, 4, 5))
    M = marginal(P, (3, 1))
    assert M.shape == (3, 5)
    np.testing.assert_allclose(M, P.sum(axis=(0, 2)))
    assert float(marginal(P, ())) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        marginal(P, (4,))


# ----------------------------------------------------------------------------------------------- m-projection
def test_fast_paths_full_model_and_independence():
    rng = np.random.default_rng(4)
    P = random_table(rng, (2, 3, 2))
    full = m_project(P, [(0, 1, 2)])
    assert full.n_iter == 0 and full.converged
    np.testing.assert_array_equal(full.Q, P)
    ind = m_project(P, [])
    expected = np.einsum("i,j,k->ijk", marginal(P, (0,)), marginal(P, (1,)), marginal(P, (2,)))
    np.testing.assert_allclose(ind.Q, expected, rtol=1e-13)
    assert kl(P, ind.Q) == pytest.approx(total_correlation(P), rel=1e-12)


def test_ipf_matches_generator_margins_of_a_cycle_model():
    rng = np.random.default_rng(5)
    P = random_table(rng, (2, 3, 2, 2))
    H = [(0, 1), (1, 2), (2, 3), (0, 3)]  # a 4-cycle: not decomposable, needs several sweeps
    res = m_project(P, H)
    assert res.converged and res.n_iter > 1 and res.max_err < 1e-12
    for G in H:
        np.testing.assert_allclose(marginal(res.Q, G), marginal(P, G), atol=1e-12)
    # m-projection = maximum entropy with those margins: D(P‖Q) = H(Q) − H(P), since P has the same margins
    assert kl(P, res.Q) == pytest.approx(entropy(res.Q) - entropy(P), abs=1e-12)


def test_ipf_equals_closed_form_of_decomposable_model():
    rng = np.random.default_rng(6)
    P = random_table(rng, (3, 2, 4))
    res = m_project(P, [(0, 1), (1, 2)])
    closed = marginal(P, (0, 1))[:, :, None] * marginal(P, (1, 2))[None, :, :] / marginal(P, (1,))[None, :, None]
    np.testing.assert_allclose(res.Q, closed, atol=1e-13)


def test_projection_lies_in_the_model_binary_moebius_coordinates_vanish():
    rng = np.random.default_rng(7)
    P = random_table(rng, (2, 2, 2, 2))
    H = [(0, 1), (1, 2), (0, 2), (2, 3)]
    Q = m_project(P, H).Q
    theta = theta_binary(Q)
    family = {T for G in H for T in nonempty_subsets(G)}
    for S, value in theta.items():
        if S not in family:
            assert abs(value) < 1e-9, (S, value)


@pytest.mark.parametrize("cards", [(2, 3, 2, 2), (3, 2, 4), (2, 2, 2, 2, 2)])
def test_pythagorean_identity_mixed_cardinalities(cards):
    rng = np.random.default_rng(8)
    d = len(cards)
    pairs = [S for S in graded_order(d, 2, 2)]
    for _ in range(5):
        P = random_table(rng, cards, conc=0.7)
        chains = [([], pairs), (pairs[:2], pairs), (pairs, pairs + [(0, 1, 2)]), ([(0, 1)], [(0, 1, 2)])]
        for H, H2 in chains:
            Q = m_project(P, H).Q
            Q2 = m_project(P, H2).Q
            lhs = kl(P, Q)
            rhs = kl(P, Q2) + kl(Q2, Q)
            assert lhs == pytest.approx(rhs, abs=1e-10)
            assert lhs >= kl(P, Q2) - 1e-12  # projecting onto a larger family never increases the error


def test_warm_start_equals_cold_start():
    rng = np.random.default_rng(9)
    P = random_table(rng, (2, 3, 2, 2))
    small = m_project(P, [(0, 1), (2, 3)]).Q
    big = [(0, 1), (1, 2), (2, 3), (0, 3)]
    cold = m_project(P, big)
    warm = m_project(P, big, init=small)
    np.testing.assert_allclose(warm.Q, cold.Q, atol=1e-12)
    with pytest.raises(ValueError):
        bad = small.copy()
        bad[0, 0, 0, 0] = 0.0
        m_project(P, big, init=bad)


def test_non_convergence_is_reported_and_logged(caplog):
    rng = np.random.default_rng(10)
    P = random_table(rng, (2, 2, 2))
    with caplog.at_level(logging.WARNING, logger="tycho.ri.loglinear"):
        res = m_project(P, [(0, 1), (1, 2), (0, 2)], max_iter=1)
    assert not res.converged and res.n_iter == 1 and res.max_err > 1e-12
    assert "not converged" in caplog.text


def test_unnormalised_input_keeps_its_mass():
    rng = np.random.default_rng(11)
    C = 250.0 * random_table(rng, (2, 3, 2))
    res = m_project(C, [(0, 1), (1, 2), (0, 2)])
    assert res.Q.sum() == pytest.approx(250.0)
    np.testing.assert_allclose(res.Q / 250.0, m_project(C / 250.0, [(0, 1), (1, 2), (0, 2)]).Q, atol=1e-13)


def test_and_relation_boundary_solution_is_exact():
    """X3 = X1 AND X2 lies in the closure of the pairwise model, so its projection onto all pairs is itself.

    Plain IPF from the uniform start only creeps towards it (KL error ≈ 1/(12 t)); the facial start gives it at once.
    """
    P = and_table()
    pairs = [(0, 1), (0, 2), (1, 2)]
    res = m_project(P, pairs)
    assert res.converged and res.n_iter <= 2
    np.testing.assert_allclose(res.Q, P, atol=1e-15)
    assert kl(P, res.Q) == 0.0
    slow_1k, slow_2k = kl(P, plain_ipf(P, pairs, 1000)), kl(P, plain_ipf(P, pairs, 2000))
    assert slow_2k < slow_1k < 2e-4  # the reference is still converging, towards our exact value 0
    assert slow_1k / slow_2k == pytest.approx(2.0, rel=0.05)  # sublinear: error ∝ 1/t


def test_facial_support():
    pairs = [(0, 1), (0, 2), (1, 2)]
    P = and_table()
    np.testing.assert_array_equal(facial_support(P, pairs), P > 0)  # boundary: support of P itself
    xor = np.zeros((2, 2, 2))
    for a, b in product(range(2), repeat=2):
        xor[a, b, a ^ b] = 0.25
    assert facial_support(xor, pairs).all()  # parity is not in the closure: interior projection
    implied = np.ones((2, 2, 2))
    implied[0, 0, :] = 0.0  # zeros implied by the (0, 1) margin: no LP needed
    np.testing.assert_array_equal(facial_support(implied, pairs), implied > 0)
    assert facial_support(np.ones((3, 2)), [(0,), (1,)]).all()


def brute_facial_support(P: np.ndarray, gens) -> np.ndarray:
    """Independent check (primal LP per cell): x is in the support iff max R(x) > 0 over the fibre
    {R >= 0 : R_G = P_G for every generator G}."""
    from scipy.optimize import linprog

    cells = list(np.ndindex(P.shape))
    rows, rhs = [], []
    for G in gens:
        M = marginal(P, G)
        for y in np.ndindex(M.shape):
            rows.append([1.0 if tuple(x[i] for i in G) == y else 0.0 for x in cells])
            rhs.append(M[y])
    out = np.zeros(P.shape, dtype=bool)
    for k, x in enumerate(cells):
        c = np.zeros(len(cells))
        c[k] = -1.0
        res = linprog(c, A_eq=np.array(rows), b_eq=np.array(rhs), bounds=[(0, None)] * len(cells), method="highs")
        assert res.status == 0
        out[x] = -res.fun > 1e-9
    return out


@pytest.mark.parametrize("shape,gens", [
    ((3, 2, 2), [(0, 1), (0, 2), (1, 2)]),
    ((2, 2, 2, 2), [(0, 1), (1, 2), (2, 3), (0, 3)]),
    ((3, 3, 2), [(0, 1), (1, 2)]),
])
def test_facial_support_matches_brute_force_on_sparse_tables(shape, gens):
    rng = np.random.default_rng(16)
    n_lp_cases = 0
    for _ in range(16):
        counts = rng.multinomial(int(rng.integers(3, 9)), rng.dirichlet(np.full(int(np.prod(shape)), 0.4)))
        P = counts.reshape(shape) / counts.sum()
        support = facial_support(P, gens)
        np.testing.assert_array_equal(support, brute_facial_support(P, gens))
        implied = np.zeros(shape, dtype=bool)
        for G in gens:
            implied |= P.sum(axis=tuple(i for i in range(P.ndim) if i not in G), keepdims=True) == 0
        n_lp_cases += int(np.any((P == 0) & ~implied))  # zeros no margin explains: the LP decides them
        res = m_project(P, gens)  # IPF on the exact support converges and matches every generator margin
        assert res.converged
        for G in gens:
            np.testing.assert_allclose(marginal(res.Q, G), marginal(P, G), atol=1e-12)
    assert n_lp_cases >= 2  # the LP path was exercised


# ----------------------------------------------------------------------------------------------- information
def test_information_measures():
    rng = np.random.default_rng(12)
    P = random_table(rng, (2, 3, 4))
    assert kl(P, P) == 0.0
    assert entropy(np.ones((4, 4))) == pytest.approx(np.log(16))
    Q = P.copy()
    Q[0, 0, 0] = 0.0
    assert kl(P, Q) == np.inf
    assert kl(Q, P) < np.inf  # 0 log 0 = 0
    H = entropy
    mi = mutual_information(P, (0,), (1, 2))
    assert mi == pytest.approx(H(marginal(P, (0,))) + H(marginal(P, (1, 2))) - H(P), abs=1e-12)
    tc = total_correlation(P)
    assert tc == pytest.approx(sum(H(marginal(P, (i,))) for i in range(3)) - H(P), abs=1e-12)
    # normalisation invariance
    assert kl(3 * P, 7 * Q + 1e-3) == pytest.approx(kl(P, (7 * Q + 1e-3) / (7 * Q + 1e-3).sum()))
    assert mutual_information(5 * P, (0,), (2,)) == pytest.approx(mutual_information(P, (0,), (2,)))
    independent = np.einsum("i,j->ij", [0.2, 0.8], [0.1, 0.6, 0.3])
    assert mutual_information(independent, (0,), (1,)) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        mutual_information(P, (0, 1), (1,))
    with pytest.raises(ValueError):
        mutual_information(P, (), (1,))
    with pytest.raises(ValueError):
        entropy(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        kl(P, np.ones((2, 3)))


# ----------------------------------------------------------------------------------------------- coordinates
def test_loglinear_table_spin_and_array_terms():
    theta = 0.7
    P = loglinear_table({(0, 1): theta}, [2, 2])
    # spin coding: log q = θ s0 s1 with s = 2x − 1 -> aligned cells have weight e^θ, others e^−θ
    assert P[1, 1] / P[0, 1] == pytest.approx(np.exp(2 * theta))
    assert P[0, 0] == pytest.approx(P[1, 1])
    A = np.array([[0.0, 1.0, -2.0], [0.5, 0.0, 0.3]])  # shape (cards[2], cards[0]) = (2, 3) for key (2, 0)
    Q = loglinear_table({(2, 0): A}, [3, 2, 2])
    L = np.log(Q)
    for x0, x1, x2 in product(range(3), range(2), range(2)):
        assert L[x0, x1, x2] - L[0, x1, 0] == pytest.approx(A[x2, x0] - A[0, 0])
    with pytest.raises(ValueError):
        loglinear_table({(0,): 1.0}, [3, 2])  # scalar (spin) term on a non-binary variable
    with pytest.raises(ValueError):
        loglinear_table({(0, 1): np.zeros((2, 2))}, [3, 2])
    assert loglinear_table({}, [2, 3]) == pytest.approx(np.full((2, 3), 1 / 6))


def test_theta_binary_recovers_spin_parameters():
    rng = np.random.default_rng(13)
    d = 4
    phi = {S: float(rng.normal(scale=0.6)) for S in graded_order(d, 1)}
    P = loglinear_table(phi, [2] * d)
    theta = theta_binary(P)
    expected = theta_spin_to_01(phi, d)
    assert set(theta) == set(expected) == set(graded_order(d, 1))
    for S in theta:
        assert theta[S] == pytest.approx(expected[S], abs=1e-10)
    # top-order term: θ_[d] = 2^d φ_[d]
    assert theta[(0, 1, 2, 3)] == pytest.approx(2 ** d * phi[(0, 1, 2, 3)], abs=1e-10)
    back = theta_01_to_spin(theta, d)
    for S in phi:
        assert back[S] == pytest.approx(phi[S], abs=1e-10)
    # one variable: θ_1 = log(p1/p0) = 2 φ_1
    assert theta_binary(loglinear_table({(0,): 0.3}, [2]))[(0,)] == pytest.approx(0.6)
    # θ is invariant to the normalisation of P
    th2 = theta_binary(17.0 * P)
    assert all(th2[S] == pytest.approx(theta[S], abs=1e-12) for S in theta)
    with pytest.raises(ValueError):
        theta_binary(and_table())  # zeros
    with pytest.raises(ValueError):
        theta_binary(np.ones((2, 3)))


def test_spin_and_01_conversions_are_inverse():
    rng = np.random.default_rng(14)
    d = 5
    theta = {S: float(rng.normal()) for S in graded_order(d, 1) if rng.random() < 0.6}
    phi = theta_01_to_spin(theta, d)
    back = theta_spin_to_01(phi, d)
    for S in graded_order(d, 1):
        assert back[S] == pytest.approx(theta.get(S, 0.0), abs=1e-12)


def test_eta_binary_matches_direct_probabilities():
    rng = np.random.default_rng(15)
    P = random_table(rng, (2, 2, 2))
    eta = eta_binary(3.0 * P)  # normalised internally
    for S in graded_order(3, 1):
        direct = sum(P[x] for x in product(range(2), repeat=3) if all(x[i] == 1 for i in S))
        assert eta[S] == pytest.approx(direct, abs=1e-14)
    uniform = eta_binary(np.ones((2, 2, 2, 2)))
    assert all(v == pytest.approx(0.5 ** len(S)) for S, v in uniform.items())
