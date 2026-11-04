"""Regression tests against the definitions and worked examples of Enouen & Sugiyama (TMLR 2026, arXiv:2410.11964v2)."""
import itertools

import numpy as np
import pytest

from tycho.ri import (
    chain_decomposition,
    conditional_ri,
    j_information,
    kl,
    local_ri_from_table,
    marginal_ri,
    mutual_information,
)


def _figure10_table():
    """p(a,b,c) = exp{log3·δ_ab + log3·δ_ac}/32: A uniform, B = A w.p. 3/4, C = A w.p. 3/4 (paper Fig. 10)."""
    P = np.zeros((2, 2, 2))
    for a, b, c in itertools.product(range(2), repeat=3):
        P[a, b, c] = 3.0 ** (a == b) * 3.0 ** (a == c) / 32
    return P


def test_figure10_values():
    P = _figure10_table()
    assert P.sum() == pytest.approx(1.0)
    # RI_{∅,A,B,C},AB = 0.1308 ; RI_{∅,A,B,C,AC,BC},AB = 0.0992 ; RI_{∅,A,B,C},BC = 0.0316 ; RI_{∅,A,B,C,AB,AC},BC = 0
    assert marginal_ri(P, (0, 1)) == pytest.approx(0.1308, abs=5e-5)
    assert conditional_ri(P, (0, 1)) == pytest.approx(0.0992, abs=5e-5)
    assert marginal_ri(P, (0, 2)) == pytest.approx(0.1308, abs=5e-5)
    assert conditional_ri(P, (0, 2)) == pytest.approx(0.0992, abs=5e-5)
    assert marginal_ri(P, (1, 2)) == pytest.approx(0.0316, abs=5e-5)
    assert conditional_ri(P, (1, 2)) == pytest.approx(0.0, abs=1e-10)


def test_conditional_ri_matches_chain_value():
    """RI^cond_S equals the chain step of S added after every subset not containing S."""
    rng = np.random.default_rng(3)
    P = rng.dirichlet(np.ones(24)).reshape(2, 3, 2, 2)
    S = (0, 2)
    others = [T for r in range(1, 5) for T in itertools.combinations(range(4), r)
              if not set(S) <= set(T)]
    chain = sorted(others, key=lambda T: (len(T), T)) + [S]
    cd = chain_decomposition(P, chain, start="empty")
    assert cd.steps[-1].subset == S
    assert conditional_ri(P, S) == pytest.approx(cd.steps[-1].ri, abs=1e-9)


def test_marginal_ri_is_mutual_information_and_local_ri():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(18)).reshape(3, 2, 3)
    assert marginal_ri(P, (0, 2)) == pytest.approx(mutual_information(P, (0,), (2,)), abs=1e-12)
    assert marginal_ri(P, (0, 1, 2)) == pytest.approx(local_ri_from_table(P, [(0, 1, 2)])[0], abs=1e-12)


def test_complete_chain_from_empty_decomposes_kl_to_uniform():
    """Eq. (6): along a complete hierarchical chain from {∅}, Σ RI = D_KL(p; u); singleton steps are D(p_k‖u_k)."""
    rng = np.random.default_rng(7)
    P = rng.dirichlet(np.ones(12)).reshape(2, 3, 2)
    chain = [(1,), (0,), (0, 1), (2,), (1, 2), (0, 2), (0, 1, 2)]
    cd = chain_decomposition(P, chain, start="empty")
    u = np.full(P.shape, 1.0 / P.size)
    assert cd.kl_base == pytest.approx(kl(P, u), abs=1e-12)
    assert cd.total == pytest.approx(kl(P, u), abs=1e-9)
    assert cd.residual == pytest.approx(0.0, abs=1e-10)
    p1 = P.sum(axis=(0, 2))
    assert cd.steps[0].ri == pytest.approx(kl(p1, np.full(3, 1 / 3)), abs=1e-12)
    assert all(step.ri >= -1e-12 for step in cd.steps)


def test_empty_start_rejects_non_hierarchical_chain():
    P = np.full((2, 2), 0.25)
    with pytest.raises(ValueError):
        chain_decomposition(P, [(0, 1)], start="empty")
    with pytest.raises(ValueError):
        chain_decomposition(P, [(0,)], start="bogus")


def test_j_information_figure1_xor():
    """Fig. 1: for X3 = X1 xor X2, J is 1 bit on {1,2,3} and 0 on every pair and singleton."""
    P = np.zeros((2, 2, 2))
    for a, b in itertools.product(range(2), repeat=2):
        P[a, b, a ^ b] = 0.25
    assert j_information(P, (0, 1, 2)) == pytest.approx(np.log(2), abs=1e-12)
    for S in [(0,), (1,), (0, 1), (1, 2)]:
        assert j_information(P, S) == pytest.approx(0.0, abs=1e-12)
    assert marginal_ri(P, (0, 1, 2)) == pytest.approx(np.log(2), abs=1e-9)
