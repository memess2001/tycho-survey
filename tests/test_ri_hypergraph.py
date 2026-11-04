"""Tests of tycho.ri.hypergraph: subset normalisation, closures, generators, admissibility, graded order."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from tycho.ri.hypergraph import (
    admissible_candidates,
    down_closure,
    graded_key,
    graded_order,
    is_admissible,
    is_hierarchical,
    maximal_elements,
    nonempty_subsets,
    normalize_subset,
)


def _brute_down_closure(H, d=None):
    out = set()
    for S in H:
        S = tuple(sorted(S))
        for k in range(1, len(S) + 1):
            out.update(combinations(S, k))
    if d is not None:
        out.update((i,) for i in range(d))
    return out


def _random_hierarchical(rng: np.random.Generator, d: int, n_gens: int) -> frozenset:
    gens = []
    for _ in range(n_gens):
        k = int(rng.integers(1, d + 1))
        gens.append(tuple(sorted(rng.choice(d, size=k, replace=False).tolist())))
    return down_closure(gens, d)


def test_normalize_subset_sorts_and_validates():
    assert normalize_subset([3, 0, 2]) == (0, 2, 3)
    assert normalize_subset(np.array([2, 1], dtype=np.int32)) == (1, 2)
    assert normalize_subset(()) == ()
    with pytest.raises(ValueError):
        normalize_subset([1, 1])
    with pytest.raises(ValueError):
        normalize_subset([-1, 2])
    with pytest.raises(TypeError):
        normalize_subset([1.5])


def test_nonempty_subsets_graded_lex_and_counts():
    assert nonempty_subsets((2, 0, 1)) == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    assert nonempty_subsets((2, 0, 1), proper=True) == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    for n in range(0, 7):
        S = tuple(range(n))
        assert len(nonempty_subsets(S)) == 2 ** n - 1
        assert len(nonempty_subsets(S, proper=True)) == max(0, 2 ** n - 2)
        subs = nonempty_subsets(S)
        assert subs == sorted(subs, key=graded_key)


def test_down_closure_matches_brute_force_and_adds_singletons():
    assert down_closure([(0, 2)]) == frozenset({(0,), (2,), (0, 2)})
    assert down_closure([(0, 2)], d=3) == frozenset({(0,), (1,), (2,), (0, 2)})
    assert down_closure([(), (1,)]) == frozenset({(1,)})  # the empty set is never a member
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = int(rng.integers(1, 7))
        H = [tuple(rng.choice(d, size=int(rng.integers(1, d + 1)), replace=False)) for _ in range(3)]
        got = down_closure(H, d)
        assert got == _brute_down_closure(H, d)
        assert is_hierarchical(got)
        assert down_closure(got, d) == got  # idempotent
    with pytest.raises(ValueError):
        down_closure([(0, 3)], d=3)


def test_is_hierarchical():
    assert is_hierarchical([])
    assert is_hierarchical([(0,), (1,), (0, 1)])
    assert not is_hierarchical([(0, 1)])  # singletons are not implicit here
    assert not is_hierarchical([(0,), (1,), (2,), (0, 1, 2)])  # pairs missing
    assert not is_hierarchical([(0,), (1,), (2,), (0, 1), (0, 2), (0, 1, 2)])  # (1, 2) missing
    assert is_hierarchical([(1, 0), (0,), (1,)])  # unsorted input is normalised


def test_maximal_elements():
    H = down_closure([(0, 1), (1, 2, 3), (4,)], d=6)
    assert maximal_elements(H) == [(4,), (5,), (0, 1), (1, 2, 3)]
    assert maximal_elements([(0,), (0, 1), (1,)]) == [(0, 1)]
    rng = np.random.default_rng(1)
    for _ in range(30):
        d = int(rng.integers(2, 7))
        H = _random_hierarchical(rng, d, 3)
        gens = maximal_elements(H)
        assert down_closure(gens, d) == H  # generators determine the family
        for A, B in combinations(gens, 2):  # an antichain
            assert not set(A) <= set(B) and not set(B) <= set(A)


def test_is_admissible_uses_singleton_convention():
    empty = frozenset()
    assert is_admissible(empty, (0, 1))  # singletons count as members
    assert not is_admissible(empty, (0,))  # order < 2
    assert not is_admissible(empty, (0, 1, 2))  # pairs missing
    H = down_closure([(0, 1), (0, 2), (1, 2)], 3)
    assert is_admissible(H, (2, 0, 1))
    assert not is_admissible(H, (0, 1))  # already present
    H2 = down_closure([(0, 1), (0, 2)], 3)
    assert not is_admissible(H2, (0, 1, 2))


def test_admissible_candidates_matches_brute_force():
    assert admissible_candidates(frozenset(), 4) == list(combinations(range(4), 2))
    H = down_closure([(0, 1), (0, 2), (1, 2)], 4)
    assert admissible_candidates(H, 4) == [(0, 3), (1, 3), (2, 3), (0, 1, 2)]
    assert admissible_candidates(H, 4, max_order=2) == [(0, 3), (1, 3), (2, 3)]
    rng = np.random.default_rng(2)
    for _ in range(40):
        d = int(rng.integers(2, 7))
        H = _random_hierarchical(rng, d, int(rng.integers(1, 4)))
        for max_order in (None, 2, 3):
            brute = [S for S in graded_order(d, 2, max_order) if is_admissible(H, S)]
            assert admissible_candidates(H, d, max_order) == brute
            for S in brute:  # adding an admissible subset keeps the family hierarchical
                assert is_hierarchical(H | {S})
    with pytest.raises(ValueError):
        admissible_candidates(frozenset({(0, 5)}), 4)


def test_graded_order_counts_and_order():
    for d in range(0, 8):
        for lo, hi in [(2, None), (1, 3), (0, 2), (3, 3)]:
            got = graded_order(d, lo, hi)
            top = d if hi is None else min(d, hi)
            assert len(got) == sum(comb(d, k) for k in range(max(1, lo), top + 1))
            assert got == sorted(got, key=graded_key)
            assert len(set(got)) == len(got)


def test_graded_chain_steps_are_admissible():
    d = 5
    H = down_closure([], d)
    for S in graded_order(d, 2):
        assert is_admissible(H, S)
        H = H | {S}
    assert len(H) == 2 ** d - 1
