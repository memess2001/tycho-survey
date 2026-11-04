"""Combinatorics of hierarchical families of variable subsets.

Hierarchical log-linear models (:mod:`tycho.ri.loglinear`) and refined-information chains
(:mod:`tycho.ri.refined`) are indexed by families of variable subsets. This module holds the set-system
operations they need.

Conventions
-----------
* A *subset* is a sorted tuple of distinct non-negative axis indices (:data:`Subset`). Every function accepts
  any sequence of integers and normalises it with :func:`normalize_subset`; repeated or negative indices raise
  ``ValueError``.
* The empty set is never a member of a family (the normaliser ψ(θ) of the log-linear model plays its role);
  empty input subsets are ignored.
* A family ``H`` is *hierarchical* (an order ideal, or down-set, of the Boolean lattice) if ``S ∈ H`` and
  ``∅ ≠ T ⊂ S`` imply ``T ∈ H``. :func:`is_hierarchical` checks exactly this, with no implicit members.
* By convention every model contains all singletons (main effects). :func:`down_closure` adds the singletons
  of all axes ``< d`` when ``d`` is given, and :func:`is_admissible` / :func:`admissible_candidates` treat
  every singleton as a member of ``H``.
* *Graded-lex* order sorts subsets by ``(len(S), S)``. Every list returned here is in that order, so results
  are deterministic.

A hierarchical family is determined by its maximal elements (*generators*): they are the margins that
iterative proportional fitting matches. A subset ``S`` is *admissible* with respect to ``H`` if ``|S| ≥ 2``,
``S ∉ H`` and every proper nonempty subset of ``S`` is in ``H``. Then ``H ∪ {S}`` is again hierarchical, which
is one step of a complete hierarchical chain (:func:`tycho.ri.refined.chain_decomposition`).
"""
from __future__ import annotations

import operator
from itertools import combinations
from typing import Iterable, Sequence

Subset = tuple[int, ...]

__all__ = [
    "Subset",
    "normalize_subset",
    "graded_key",
    "nonempty_subsets",
    "down_closure",
    "is_hierarchical",
    "maximal_elements",
    "is_admissible",
    "admissible_candidates",
    "graded_order",
]


def normalize_subset(S: Sequence[int]) -> Subset:
    """Return ``S`` as a sorted tuple of ints.

    Raises:
        TypeError: if an element is not an integer (``operator.index`` fails, e.g. for ``1.5``).
        ValueError: on negative or repeated indices.
    """
    out = tuple(sorted(operator.index(i) for i in S))
    if out and out[0] < 0:
        raise ValueError(f"negative axis index in subset {tuple(S)!r}")
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis index in subset {tuple(S)!r}")
    return out


def graded_key(S: Subset) -> tuple[int, Subset]:
    """Sort key of the graded-lex order: ``(len(S), S)``."""
    return (len(S), S)


def _family(H: Iterable[Sequence[int]]) -> set[Subset]:
    """Normalised members of ``H``, without the empty set."""
    return {s for s in (normalize_subset(S) for S in H) if s}


def nonempty_subsets(S: Sequence[int], proper: bool = False) -> list[Subset]:
    """All nonempty subsets of ``S`` (excluding ``S`` itself if ``proper``), in graded-lex order."""
    S = normalize_subset(S)
    top = len(S) - 1 if proper else len(S)
    return [c for k in range(1, top + 1) for c in combinations(S, k)]


def down_closure(H: Iterable[Sequence[int]], d: int | None = None) -> frozenset[Subset]:
    """The smallest hierarchical family containing ``H``; with ``d``, also all singletons ``(i,)``, ``i < d``.

    Raises:
        ValueError: if ``d`` is given and a member of ``H`` uses an axis ``>= d``.
    """
    fam: set[Subset] = set()
    # Largest first: a subset already present had its whole down-set added with a superset.
    for S in sorted(_family(H), key=lambda s: (-len(s), s)):
        if d is not None and S[-1] >= d:
            raise ValueError(f"subset {S} uses an axis >= d={d}")
        if S not in fam:
            fam.update(nonempty_subsets(S))
    if d is not None:
        fam.update((i,) for i in range(d))
    return frozenset(fam)


def is_hierarchical(H: Iterable[Sequence[int]]) -> bool:
    """True iff every proper nonempty subset of every member of ``H`` is itself a member (no implicit members).

    It suffices to check the subsets of co-size one: by induction on size they generate all proper subsets.
    """
    fam = _family(H)
    return all(S[:i] + S[i + 1:] in fam for S in fam if len(S) >= 2 for i in range(len(S)))


def maximal_elements(H: Iterable[Sequence[int]]) -> list[Subset]:
    """Members of ``H`` not strictly contained in another member (the generators), in graded-lex order."""
    kept: list[frozenset[int]] = []
    out: list[Subset] = []
    # Largest first: a non-maximal member lies inside some maximal member, which is already kept.
    for S in sorted(_family(H), key=lambda s: (-len(s), s)):
        fs = frozenset(S)
        if not any(fs < K for K in kept):
            kept.append(fs)
            out.append(S)
    return sorted(out, key=graded_key)


def _admissible(fam: set[Subset] | frozenset[Subset], S: Subset) -> bool:
    """Admissibility of a normalised ``S`` against a normalised family; singletons count as members."""
    if len(S) < 2 or S in fam:
        return False
    return all(T in fam for T in nonempty_subsets(S, proper=True) if len(T) >= 2)


def is_admissible(H: frozenset[Subset], S: Sequence[int]) -> bool:
    """True iff ``|S| >= 2``, ``S ∉ H`` and every proper nonempty subset of ``S`` is in ``H``.

    Singletons are treated as members of ``H`` (every model contains all main effects), so a pair is
    admissible iff it is not already in ``H``.
    """
    return _admissible(_family(H), normalize_subset(S))


def admissible_candidates(H: frozenset[Subset], d: int, max_order: int | None = None) -> list[Subset]:
    """All subsets of ``range(d)`` of order ``<= max_order`` that are admissible w.r.t. ``H`` (graded-lex).

    Singletons of all ``d`` axes are treated as members of ``H``.

    Raises:
        ValueError: if a member of ``H`` uses an axis ``>= d``.
    """
    fam = _family(H)
    if any(S[-1] >= d for S in fam):
        raise ValueError(f"H uses an axis >= d={d}")
    top = d if max_order is None else min(int(max_order), d)
    bases = fam | {(i,) for i in range(d)}
    found: set[Subset] = set()
    # Every admissible S of order k >= 2 extends one of its (k-1)-subsets, all of which are in ``bases``.
    for T in bases:
        if len(T) + 1 > top:
            continue
        members = set(T)
        for j in range(d):
            if j in members:
                continue
            S = tuple(sorted(T + (j,)))
            if S not in found and _admissible(fam, S):
                found.add(S)
    return sorted(found, key=graded_key)


def graded_order(d: int, min_order: int = 2, max_order: int | None = None) -> list[Subset]:
    """All subsets of ``range(d)`` with ``min_order <= |S| <= max_order`` (default ``d``), graded-lex.

    Orders below 1 are clamped to 1 (the empty set is never listed).
    """
    lo = max(1, int(min_order))
    hi = d if max_order is None else min(d, int(max_order))
    return [c for k in range(lo, hi + 1) for c in combinations(range(d), k)]
