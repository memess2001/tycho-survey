"""Empirical Walsh–Fourier coefficients of a model's soft binary output: the Fourier baseline of RI curves.

Definition
----------
For a :class:`~tycho.ri.variables.VariableSet` whose soft column ``o`` is binary (the model's predictive
distribution over two output codes, e.g. ``yhat`` / ``xhat``) and a subset ``S ∋ o`` whose other columns
``A = S \\ {o}`` are all binary (card 2, code 1 ↦ spin +1), the empirical Walsh coefficient is

    f̂(A) = Σ_n w_n (2 p1_n − 1) Π_{i∈A} (2 x_{n,i} − 1) / Σ_n w_n,

with ``p1_n = soft[o][n, 1]`` and probe weights ``w_n`` (default 1; :class:`VariableSet` currently has no weights
field, so rows are unweighted, as in the runner's :func:`~tycho.ri.estimators.ri_scan`). A subset that does not
contain ``o``, or that has a non-binary column, gets NaN; so does every subset when ``vs`` has no soft column, when
the soft column is not binary (e.g. ``icl_markov`` with ``vocab > 2``) or when the total weight is 0. Only the soft
column counts as the output: a hard column (``modular_addition``'s ``correct``, ``sparse_parity`` with
``ri_soft = False``) gives NaN everywhere.

The sums go through one matrix product per order and chunk of shared prefixes (BLAS), so values are reproducible
on one machine and may differ in the last bits between BLAS builds.

Relation to RI
--------------
With independent uniform binary probe inputs, ``E[2 p1 − 1 | x_A] = Σ_{B⊆A} f(B) χ_B(x_A)``, so the marginal
``P(x_A, ŷ)`` and hence the local RI of ``A ∪ {o}`` are functions of the Walsh coefficients on the subsets of
``A``. For a single mode (``2 p1 − 1 = ε χ_A(x)``) the local RI of ``A ∪ {o}`` is ``ln 2 − h((1 + ε)/2)`` (``h``
the binary entropy in nats). With correlated probe inputs (Markov-chain contexts) the two measures can differ.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from tycho.ri.loglinear import _check_weights
from tycho.ri.variables import VariableSet

__all__ = ["walsh_output", "walsh_coefficients", "single_mode_ri"]

#: Upper bound on (prefix, sample) products held per chunk (~32 MB of float64 working memory).
_CHUNK = 1 << 22


def walsh_output(vs: VariableSet) -> int | None:
    """Index of ``vs``'s soft column if it is binary (the output of the Walsh coefficients), else None."""
    if not vs.soft:
        return None
    (o,) = vs.soft
    return int(o) if vs.cards[o] == 2 else None


def walsh_coefficients(vs: VariableSet, subsets: Sequence[Sequence[int]], *,
                       weights: np.ndarray | None = None) -> np.ndarray:
    """Empirical Walsh coefficient ``f̂(S \\ {o})`` of the soft binary output ``o`` for each subset (module docstring).

    Args:
        vs: the probe rows (binary codes in ``X``, the output's probabilities in ``vs.soft``).
        subsets: column-index tuples (e.g. ``ScanResult.subsets``).
        weights: optional (N,) non-negative probe weights; default ``vs.weights`` if the VariableSet has such an
            attribute, else uniform.

    Returns:
        ``(len(subsets),)`` float64 array in ``[−1, 1]``, NaN where the coefficient is undefined.

    Raises:
        ValueError: on malformed weights or a column index outside ``[0, d)``.
    """
    subs = [tuple(int(i) for i in s) for s in subsets]
    out = np.full(len(subs), np.nan)
    o = walsh_output(vs)
    if o is None or not subs:
        return out
    if weights is None:
        weights = getattr(vs, "weights", None)
    w = _check_weights(weights, vs.N)
    total = float(vs.N) if w is None else float(w.sum())
    if not total > 0:
        return out
    signal = 2.0 * vs.soft[o][:, 1] - 1.0
    if w is not None:
        signal = signal * w
    signal = signal / total
    binary = np.asarray(vs.cards) == 2
    groups: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for row, S in enumerate(subs):
        if any(not 0 <= i < vs.d for i in S):
            raise ValueError(f"subset {S} has a column index outside [0, {vs.d})")
        if o not in S:
            continue
        A = tuple(i for i in S if i != o)
        if all(binary[i] for i in A):
            groups[len(A)].append((row, A))
    spins = 2.0 * vs.X - 1.0  # (N, d); only binary columns (codes 0/1) are read
    spins_t = np.ascontiguousarray(spins.T)
    for k, members in groups.items():
        rows = np.asarray([r for r, _ in members], dtype=np.int64)
        if k == 0:
            out[rows] = signal.sum()
            continue
        # f̂(P ∪ {c}) = Σ_n [signal_n Π_{i∈P} z_ni] z_nc: one row of products per distinct prefix P = A minus its
        # last column, then a single matrix product gives every last column c at once
        A = np.asarray([cols for _, cols in members], dtype=np.int64)  # (m, k), sorted rows
        if k == 1:
            prefixes, which = np.zeros((1, 0), dtype=np.int64), np.zeros(len(A), dtype=np.int64)
        else:
            prefixes, which = np.unique(A[:, :-1], axis=0, return_inverse=True)
            which = which.reshape(-1)
        last = A[:, -1]
        step = max(1, _CHUNK // max(1, vs.N))
        for lo in range(0, len(prefixes), step):
            hi = min(len(prefixes), lo + step)
            prod = np.repeat(signal[None, :], hi - lo, axis=0)
            for j in range(k - 1):
                prod *= spins_t[prefixes[lo:hi, j]]
            F = prod @ spins  # (hi − lo, d)
            sel = (which >= lo) & (which < hi)
            out[rows[sel]] = F[which[sel] - lo, last[sel]]
    return out


def single_mode_ri(eps: np.ndarray | float) -> np.ndarray:
    """``ln 2 − h((1 + ε)/2)`` in nats: the local RI of ``A ∪ {o}`` when ``2 p1 − 1 = ε χ_A(x)`` on uniform inputs."""
    e = np.clip(np.abs(np.asarray(eps, dtype=np.float64)), 0.0, 1.0)
    p, q = (1.0 + e) / 2.0, (1.0 - e) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(p) + np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0))
    return np.log(2.0) - h
