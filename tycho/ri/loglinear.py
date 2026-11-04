"""Hierarchical log-linear models on probability tables: m-projection, information measures, coordinates.

Math
----
Variables ``X = (X_1..X_d)``, ``X_i`` categorical with ``k_i`` states; a joint distribution is a table ``P`` of
shape ``(k_1, ..., k_d)``. For a hierarchical family ``H`` of subsets (see :mod:`tycho.ri.hypergraph`; every
model contains all singletons) the hierarchical log-linear model is the exponential (e-flat) family

    E_H = { q :  log q(x) = Σ_{S ∈ H} θ_S(x_S) − ψ(θ) }.

The **m-projection** ``P_H = argmin_{Q ∈ cl(E_H)} D(P‖Q)`` is the unique member of the closure ``cl(E_H)`` whose
``S``-margins equal those of ``P`` for every ``S ∈ H``; equivalently the maximum-entropy table with those margins,
or the MLE of the model when ``P`` is an empirical table. For ``H ⊆ H'`` the Pythagorean relation holds:
``D(P‖P_H) = D(P‖P_H') + D(P_H'‖P_H)``.

Algorithm (:func:`m_project`): iterative proportional fitting (IPF) over the generators ``G`` (maximal elements)
of ``H``: each sweep rescales ``Q ← Q · P_G / Q_G`` for every generator in graded-lex order (``0/0 := 0``).
From any start in ``cl(E_H)`` that is positive on the support of the solution, the iterates stay in ``cl(E_H)``
and converge to ``P_H``.

Boundary (facial) solutions. If ``P`` has zero cells, ``P_H`` may be zero on cells where no generator margin of
``P`` vanishes (example: ``X_3 = X_1 AND X_2`` lies in the closure of the pairwise model, so its projection onto
all pairs is ``P`` itself, with four zero cells). IPF from the uniform start then converges only sublinearly: the
cells that must vanish decay like ``1/t`` (for the AND table the KL error after ``t`` sweeps is ``≈ 1/(12 t)``).
The support of ``P_H`` is the *facial set* of ``supp(P)`` in the marginal polytope (Fienberg & Rinaldo 2012,
"Maximum likelihood estimation in log-linear models", Ann. Stat.): a cell is zero in ``P_H`` iff it is zero in
every non-negative table with ``P``'s ``H``-margins. :func:`facial_support` decides this exactly (one small linear
program when some zero of ``P`` is not implied by a vanishing generator margin), and :func:`m_project` starts IPF
from the uniform distribution on that support, where convergence is linear.

Conventions
-----------
* Natural logarithms; all information quantities are in nats. ``0 log 0 = 0``.
* :func:`kl`, :func:`entropy`, :func:`mutual_information` and :func:`total_correlation` normalise their inputs
  to total mass 1 first (identical to the plain formulas for normalised tables); a zero-mass table raises.
* :func:`marginal` keeps the remaining axes in sorted order.

Binary coordinates
------------------
Two codings of a binary variable are used, and they must not be confused:

* **0/1 coding** (Möbius / Sugiyama–Amari coordinates; :func:`theta_binary`, :func:`eta_binary`): with
  ``x_i ∈ {0, 1}`` and ``x^T = Π_{i∈T} x_i``, ``log p(x) = Σ_{T} θ_T x^T``. Möbius inversion on the Boolean
  lattice gives ``θ_S = Σ_{T⊆S} (−1)^{|S∖T|} log p(1_T)`` (``1_T`` = the configuration with ones exactly on ``T``)
  and ``η_S = P(X_i = 1 ∀ i ∈ S)``.
* **Spin coding** (:func:`loglinear_table` with scalar parameters; used to construct planted distributions):
  ``s_i = 2 x_i − 1 ∈ {−1, +1}`` and ``log q(x) = Σ_S φ_S Π_{i∈S} s_i − ψ``.

Conversion (:func:`theta_spin_to_01`, :func:`theta_01_to_spin`), for nonempty ``S`` and ``T``:

    θ_T = 2^{|T|} Σ_{S ⊇ T} (−1)^{|S|−|T|} φ_S          φ_S = Σ_{T ⊇ S} 2^{−|T|} θ_T

In particular the top-order coefficients satisfy ``θ_[d] = 2^d φ_[d]``.
"""
from __future__ import annotations

import logging
import math
import operator
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from tycho.ri.hypergraph import (
    Subset,
    down_closure,
    graded_order,
    maximal_elements,
    nonempty_subsets,
    normalize_subset,
)

logger = logging.getLogger(__name__)

__all__ = [
    "joint_table",
    "marginal",
    "normalize_table",
    "ProjectionResult",
    "facial_support",
    "m_project",
    "kl",
    "entropy",
    "mutual_information",
    "total_correlation",
    "loglinear_table",
    "theta_binary",
    "eta_binary",
    "theta_spin_to_01",
    "theta_01_to_spin",
]

#: Tolerance on the row sums of a soft column (rows must be probability vectors).
SOFT_ROW_ATOL = 1e-6
#: LP solution value above which a zero cell is classified as outside the facial support. The optimal LP
#: solution is 0/1-valued (see :func:`facial_support`), so any threshold in (0, 1) is safe.
_LP_THRESHOLD = 0.5


# ----------------------------------------------------------------------------------------------- validation
def _as_table(P: np.ndarray, name: str = "P") -> np.ndarray:
    """``P`` as a float64 array with at least one axis, finite, non-negative and with positive mass."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 0:
        raise ValueError(f"{name} must have at least one axis")
    if not np.all(np.isfinite(P)):
        raise ValueError(f"{name} must be finite")
    if np.any(P < 0):
        raise ValueError(f"{name} must be non-negative")
    if not P.sum() > 0:
        raise ValueError(f"{name} has zero total mass")
    return P


def normalize_table(P: np.ndarray, name: str = "P") -> np.ndarray:
    """Validate a table (float64, at least one axis, finite, non-negative, positive mass) and divide by its sum."""
    P = _as_table(P, name)
    return P / P.sum()


def _check_axes(S: Subset, d: int) -> Subset:
    if S and S[-1] >= d:
        raise ValueError(f"subset {S} uses an axis >= d={d}")
    return S


def _complement(S: Subset, d: int) -> tuple[int, ...]:
    members = set(S)
    return tuple(i for i in range(d) if i not in members)


def _ratio(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """``target / current`` with ``x/0 := 0`` (IPF rescaling factor)."""
    out = np.zeros(np.broadcast_shapes(target.shape, current.shape))
    np.divide(target, current, out=out, where=current > 0)
    return out


# ----------------------------------------------------------------------------------------------- tables
def _check_soft(soft: Mapping[int, np.ndarray] | None, n: int,
                cards: Sequence[int]) -> tuple[int | None, np.ndarray | None]:
    if not soft:
        return None, None
    if len(soft) > 1:
        raise ValueError("at most one soft column is supported")
    ((j, arr),) = soft.items()
    j = operator.index(j)
    if not 0 <= j < len(cards):
        raise ValueError(f"soft column index {j} out of range")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (n, cards[j]):
        raise ValueError(f"soft[{j}] must have shape {(n, cards[j])}, got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"soft[{j}] must be finite and non-negative")
    if n and not np.allclose(arr.sum(axis=1), 1.0, rtol=0.0, atol=SOFT_ROW_ATOL):
        raise ValueError(f"soft[{j}] rows must be probability vectors")
    return j, arr


def _check_weights(weights: np.ndarray | None, n: int) -> np.ndarray | None:
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape {(n,)}, got {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    return w


def _as_codes(X: np.ndarray, cards: Sequence[int], soft_col: int | None) -> tuple[np.ndarray, list[int]]:
    """Validate integer codes ``X`` (N, d) against ``cards``; the soft column's codes are ignored (set to 0)."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, d), got shape {X.shape}")
    cards = [operator.index(k) for k in cards]
    if len(cards) != X.shape[1]:
        raise ValueError(f"cards has length {len(cards)} but X has {X.shape[1]} columns")
    if any(k < 1 for k in cards):
        raise ValueError("cards must be >= 1")
    if X.dtype == np.bool_ or np.issubdtype(X.dtype, np.integer):
        X = X.astype(np.int64)
    else:
        Xf = X.astype(np.float64)
        if not np.all(np.isfinite(Xf)) or np.any(Xf != np.round(Xf)):
            raise ValueError("X must contain integer codes")
        X = Xf.astype(np.int64)
    if soft_col is not None:
        X[:, soft_col] = 0
    if X.size:
        if np.any(X < 0) or np.any(X >= np.asarray(cards)[None, :]):
            raise ValueError("codes out of range [0, card)")
    return X, cards


def _strides(shape: Sequence[int]) -> np.ndarray:
    """Row-major (C-order) strides, in cells, of a table of the given shape."""
    out = np.ones(len(shape), dtype=np.int64)
    for p in range(len(shape) - 2, -1, -1):
        out[p] = out[p + 1] * shape[p + 1]
    return out


def joint_table(X: np.ndarray, cards: Sequence[int], *, weights: np.ndarray | None = None,
                soft: dict[int, np.ndarray] | None = None, normalize: bool = True,
                pseudocount: float = 0.0) -> np.ndarray:
    """Empirical joint table of integer-coded observations.

    Args:
        X: (N, d) integer codes, column ``i`` in ``[0, cards[i])`` (the soft column's codes are ignored).
        cards: number of states of each column.
        weights: optional (N,) non-negative sample weights ``w_n`` (default 1).
        soft: optional ``{j: (N, cards[j]) row-stochastic array}`` (at most one soft column). Sample ``n``
            contributes ``w_n * soft[j][n, c]`` to cell ``(x_{n,-j}, c)``.
        normalize: divide by the total mass.
        pseudocount: added to every cell before normalisation (in units of samples).

    Returns:
        float64 array of shape ``tuple(cards)``; sums to 1 if ``normalize`` else to ``Σw + pseudocount·n_cells``.

    Raises:
        ValueError: on malformed input, or if ``normalize`` and the total mass is zero (e.g. ``N = 0``).
    """
    X0 = np.asarray(X)
    if X0.ndim != 2:
        raise ValueError(f"X must be 2-D (N, d), got shape {X0.shape}")
    n = X0.shape[0]
    soft_col, soft_arr = _check_soft(soft, n, list(cards))
    X, cards = _as_codes(X0, cards, soft_col)
    if not cards:
        raise ValueError("need at least one variable")
    if not (math.isfinite(pseudocount) and pseudocount >= 0):
        raise ValueError("pseudocount must be finite and >= 0")
    w = _check_weights(weights, n)
    shape = tuple(cards)
    n_cells = int(np.prod(shape))
    strides = _strides(shape)
    base = X @ strides  # the soft column's codes are 0, so it contributes nothing here
    if soft_col is None:
        counts = np.bincount(base, weights=w, minlength=n_cells).astype(np.float64)
    else:
        mass = soft_arr if w is None else soft_arr * w[:, None]
        cells = base[:, None] + np.arange(cards[soft_col])[None, :] * strides[soft_col]
        counts = np.bincount(cells.ravel(), weights=mass.ravel(), minlength=n_cells)
    T = counts.reshape(shape) + pseudocount
    if normalize:
        total = T.sum()
        if not total > 0:
            raise ValueError("table has zero total mass (no samples?)")
        T = T / total
    return T


def marginal(P: np.ndarray, S: Sequence[int]) -> np.ndarray:
    """Margin of ``P`` on the axes ``S``, kept in sorted order (``S = ()`` gives the total mass)."""
    P = np.asarray(P, dtype=np.float64)
    S = _check_axes(normalize_subset(S), P.ndim)
    return P.sum(axis=_complement(S, P.ndim))


# ----------------------------------------------------------------------------------------------- m-projection
@dataclass
class ProjectionResult:
    """Result of :func:`m_project`.

    Attributes:
        Q: the projected table (same total mass as the input).
        n_iter: IPF sweeps performed (0 for the closed-form fast paths).
        max_err: largest absolute deviation of a generator margin after the last sweep (probability scale).
        converged: ``max_err < tol``.
    """

    Q: np.ndarray
    n_iter: int
    max_err: float
    converged: bool


def _facial_zero_lp(zero: np.ndarray, generators: Sequence[Subset]) -> np.ndarray | None:
    """Cells outside the facial set of the support ``~zero`` (``None`` if the LP solver fails).

    A cell ``x`` lies outside the facial set iff some function ``f(x) = Σ_G φ_G(x_G)`` in the span of the
    generator indicators satisfies ``f = 0`` on the support, ``f >= 0`` everywhere and ``f(x) > 0``. These
    functions form a convex cone, so one LP finds them all: maximise ``Σ s_x`` subject to ``f = 0`` on the support,
    ``f(x) >= s_x`` and ``0 <= s_x <= 1`` on the zero cells. Scaling makes ``s_x = 1`` attainable for every cell
    outside the facial set simultaneously, while ``s_x = 0`` is forced inside it; the optimum is 0/1-valued.
    """
    # scipy.optimize and scipy.sparse are only needed for boundary cases; importing them lazily keeps
    # ``import tycho.ri`` light for workers that never meet such a table.
    from scipy import sparse
    from scipy.optimize import linprog

    shape = zero.shape
    n_cells = zero.size
    cells = np.indices(shape).reshape(len(shape), n_cells)
    cols, offset = [], 0
    for G in generators:
        dims = [shape[i] for i in G]
        idx = np.ravel_multi_index(tuple(cells[i] for i in G), dims) if G else np.zeros(n_cells, np.int64)
        cols.append(offset + idx)
        offset += int(np.prod(dims))
    rows = np.tile(np.arange(n_cells), len(generators))
    A = sparse.csr_matrix((np.ones(rows.size), (rows, np.concatenate(cols))), shape=(n_cells, offset))
    flat_zero = zero.ravel()
    zero_idx = np.flatnonzero(flat_zero)
    n_zero = zero_idx.size
    A_eq = sparse.hstack([A[~flat_zero], sparse.csr_matrix((n_cells - n_zero, n_zero))], format="csr")
    A_ub = sparse.hstack([-A[zero_idx], sparse.identity(n_zero, format="csr")], format="csr")
    c = np.concatenate([np.zeros(offset), -np.ones(n_zero)])
    bounds = [(None, None)] * offset + [(0.0, 1.0)] * n_zero
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n_zero), A_eq=A_eq, b_eq=np.zeros(A_eq.shape[0]),
                  bounds=bounds, method="highs")
    if res.status != 0:
        return None
    outside = np.zeros(n_cells, dtype=bool)
    outside[zero_idx[res.x[offset:] > _LP_THRESHOLD]] = True
    return outside.reshape(shape)


def facial_support(P: np.ndarray, generators: Iterable[Sequence[int]]) -> np.ndarray:
    """Boolean mask of the cells that are positive in the m-projection of ``P`` onto the model ``generators``.

    The support of the m-projection is the facial set of ``supp(P)``: the cells that are positive in at least
    one non-negative table with ``P``'s generator margins (the maximum-entropy member of that fibre is
    positive wherever any member is). Zeros of ``P`` where some generator margin vanishes are always outside;
    only when other zeros exist is the linear program of :func:`_facial_zero_lp` solved. The result depends
    on the zero pattern of ``P`` only.

    Args:
        P: non-negative table.
        generators: the generating subsets (e.g. the maximal elements of a hierarchical family).
    """
    P = _as_table(P)
    zero = P <= 0
    if not zero.any():
        return np.ones(P.shape, dtype=bool)
    gens = [_check_axes(normalize_subset(G), P.ndim) for G in generators]
    implied = np.zeros(P.shape, dtype=bool)
    for G in gens:
        implied |= P.sum(axis=_complement(G, P.ndim), keepdims=True) <= 0
    if np.array_equal(zero, implied):
        return ~zero
    outside = _facial_zero_lp(zero, gens)
    if outside is None:
        logger.warning("facial-set LP failed; falling back to plain IPF support (convergence may be slow)")
        return ~implied
    return ~outside


def m_project(P: np.ndarray, H: Iterable[Sequence[int]], *, tol: float = 1e-12, max_iter: int = 10_000,
              init: np.ndarray | None = None) -> ProjectionResult:
    """m-projection of ``P`` onto the hierarchical log-linear model ``E_H`` (IPF over the generators).

    The family is ``down_closure(H ∪ singletons)``. Fast paths: if it contains the full set, ``Q = P``; if its
    generators are all singletons, ``Q`` is the product of the one-variable margins. Otherwise IPF sweeps the
    generators in graded-lex order, starting from the uniform distribution on the facial support of ``P``
    (:func:`facial_support`; this equals the plain uniform start whenever every zero of ``P`` is implied by a
    vanishing generator margin) or from ``init`` restricted to that support.

    Args:
        P: non-negative table with positive mass (normally a probability table); ``Q`` has the same mass.
        H: subsets generating the model (axes ``< P.ndim``).
        tol: converged when, after a full sweep, every generator margin of ``Q`` deviates from that of ``P`` by
            less than ``tol`` in absolute value (both normalised to mass 1).
        max_iter: maximum number of sweeps; a warning is logged if not converged.
        init: optional warm start: a table in ``cl(E_H')`` for some ``H' ⊆ H`` (e.g. the projection onto a smaller
            family), positive on the support of the solution. A projection onto a sub-family always qualifies.

    Raises:
        ValueError: on malformed input, or if ``init`` vanishes on a cell of the solution's support.
    """
    P = _as_table(P)
    mass = float(P.sum())
    Pn = P / mass
    d = P.ndim
    family = down_closure(H, d)
    gens = maximal_elements(family)
    if tuple(range(d)) in family:
        return ProjectionResult(P.copy(), 0, 0.0, True)
    if all(len(G) == 1 for G in gens):
        Q = np.ones(P.shape)
        for i in range(d):
            shape = [1] * d
            shape[i] = P.shape[i]
            Q = Q * marginal(Pn, (i,)).reshape(shape)
        return ProjectionResult(Q * mass, 0, 0.0, True)
    others = [_complement(G, d) for G in gens]
    targets = [Pn.sum(axis=o, keepdims=True) for o in others]
    support = facial_support(Pn, gens)
    if init is None:
        Q = support.astype(np.float64)
    else:
        Q0 = _as_table(init, "init")
        if Q0.shape != P.shape:
            raise ValueError(f"init must have shape {P.shape}, got {Q0.shape}")
        if np.any(Q0[support] <= 0):
            raise ValueError("init must be positive on the support of the projection")
        Q = np.where(support, Q0, 0.0)
    Q = Q / Q.sum()
    err = math.inf
    n_iter = 0
    converged = False
    for n_iter in range(1, int(max_iter) + 1):
        for o, t in zip(others, targets):
            Q *= _ratio(t, Q.sum(axis=o, keepdims=True))
        err = max(float(np.max(np.abs(Q.sum(axis=o, keepdims=True) - t))) for o, t in zip(others, targets))
        if err < tol:
            converged = True
            break
    if not converged:
        logger.warning("m_project: IPF not converged after %d sweeps (max margin error %.3g, tol %.3g)",
                       n_iter, err, tol)
    return ProjectionResult(Q * mass, n_iter, err, converged)


# ----------------------------------------------------------------------------------------------- information
def kl(P: np.ndarray, Q: np.ndarray) -> float:
    """Kullback-Leibler divergence ``D(P‖Q) = Σ p log(p/q)`` in nats (inputs normalised; ``0 log 0 = 0``).

    Returns ``inf`` if ``q = 0 < p`` somewhere. The result is clamped at 0 against rounding.
    """
    P = normalize_table(P, "P")
    Q = normalize_table(Q, "Q")
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: {P.shape} vs {Q.shape}")
    pos = P > 0
    if np.any(Q[pos] <= 0):
        return math.inf
    p = P[pos]
    return max(0.0, float(np.sum(p * np.log(p / Q[pos]))))


def entropy(P: np.ndarray) -> float:
    """Shannon entropy in nats of the normalised table ``P``."""
    p = normalize_table(P).ravel()
    p = p[p > 0]
    return max(0.0, float(-np.sum(p * np.log(p))))


def _expand(M: np.ndarray, S: Subset, U: Subset) -> np.ndarray:
    """Reshape a table on the sorted axes ``S`` to broadcast over the sorted superset ``U``."""
    members = set(S)
    return M.reshape([M.shape[S.index(i)] if i in members else 1 for i in U])


def mutual_information(P: np.ndarray, A: Sequence[int], B: Sequence[int]) -> float:
    """``I(X_A; X_B)`` in nats, computed as ``D(P_{A∪B} ‖ P_A ⊗ P_B)`` (no entropy cancellation).

    Raises:
        ValueError: if ``A`` or ``B`` is empty or they overlap.
    """
    P = normalize_table(P)
    A = _check_axes(normalize_subset(A), P.ndim)
    B = _check_axes(normalize_subset(B), P.ndim)
    if not A or not B:
        raise ValueError("A and B must be nonempty")
    if set(A) & set(B):
        raise ValueError("A and B must be disjoint")
    U = tuple(sorted(A + B))
    prod = _expand(marginal(P, A), A, U) * _expand(marginal(P, B), B, U)
    return kl(marginal(P, U), prod)


def total_correlation(P: np.ndarray) -> float:
    """Total correlation (multi-information) ``Σ_i H(X_i) − H(X) = D(P ‖ ⊗_i P_i)`` in nats."""
    P = normalize_table(P)
    d = P.ndim
    prod = np.ones(P.shape)
    for i in range(d):
        prod = prod * _expand(marginal(P, (i,)), (i,), tuple(range(d)))
    return kl(P, prod)


# ----------------------------------------------------------------------------------------------- coordinates
def _spin_axis(i: int, d: int) -> np.ndarray:
    """``s_i = 2 x_i − 1`` as a broadcastable array along axis ``i``."""
    shape = [1] * d
    shape[i] = 2
    return np.array([-1.0, 1.0]).reshape(shape)


def loglinear_table(theta: dict[Subset, float | np.ndarray], cards: Sequence[int]) -> np.ndarray:
    """Normalised table ``q ∝ exp(Σ_S θ_S(x_S))``.

    A scalar ``θ_S`` requires ``cards[i] == 2`` for ``i ∈ S`` and uses spin coding: it contributes
    ``θ_S · Π_{i∈S} s_i`` with ``s_i = 2 x_i − 1``. An array ``θ_S`` has shape ``tuple(cards[i] for i in S)``
    (axes in the order the key lists them) and is added directly. ``S = ()`` adds a constant (no effect).
    """
    cards = [operator.index(k) for k in cards]
    if not cards or any(k < 1 for k in cards):
        raise ValueError("cards must be a nonempty list of positive integers")
    d = len(cards)
    L = np.zeros(cards)
    for key, value in theta.items():
        raw = tuple(operator.index(i) for i in key)
        S = _check_axes(normalize_subset(raw), d)
        arr = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"theta[{key}] must be finite")
        if arr.ndim == 0:
            if any(cards[i] != 2 for i in S):
                raise ValueError(f"scalar theta[{key}] (spin coding) requires binary variables")
            term = np.full([1] * d, float(arr))
            for i in S:
                term = term * _spin_axis(i, d)
        else:
            expected = tuple(cards[i] for i in raw)
            if arr.shape != expected:
                raise ValueError(f"theta[{key}] must have shape {expected}, got {arr.shape}")
            arr = arr.transpose(np.argsort(raw))
            term = arr.reshape([cards[i] if i in S else 1 for i in range(d)])
        L = L + term
    L = L - L.max()
    Q = np.exp(L)
    return Q / Q.sum()


def _check_binary(P: np.ndarray) -> None:
    if any(k != 2 for k in P.shape):
        raise ValueError(f"binary table required (all axes of size 2), got shape {P.shape}")


def _cell_of(S: Subset, d: int) -> tuple[int, ...]:
    """Index of the configuration ``1_S`` (ones exactly on ``S``)."""
    members = set(S)
    return tuple(1 if i in members else 0 for i in range(d))


def theta_binary(P: np.ndarray) -> dict[Subset, float]:
    """Möbius θ-coordinates (0/1 coding) of a strictly positive binary table, for every nonempty ``S``.

    ``θ_S = Σ_{T⊆S} (−1)^{|S∖T|} log p(1_T)``. They do not depend on the normalisation of ``P``. See the module
    docstring for the conversion to spin-coded parameters.

    Raises:
        ValueError: if an axis is not binary or some cell is not strictly positive.
    """
    P = _as_table(P)
    _check_binary(P)
    if np.any(P <= 0):
        raise ValueError("theta_binary requires a strictly positive table")
    L = np.log(P)
    for ax in range(P.ndim):  # Möbius inversion along each axis: [a0, a1] -> [a0, a1 - a0]
        a0 = np.take(L, [0], axis=ax)
        L = np.concatenate([a0, np.take(L, [1], axis=ax) - a0], axis=ax)
    return {S: float(L[_cell_of(S, P.ndim)]) for S in graded_order(P.ndim, 1)}


def eta_binary(P: np.ndarray) -> dict[Subset, float]:
    """η-coordinates ``η_S = P(X_i = 1 ∀ i ∈ S)`` of a binary table (normalised first), every nonempty ``S``."""
    E = normalize_table(P)
    _check_binary(E)
    for ax in range(E.ndim):  # zeta transform along each axis: [a0, a1] -> [a0 + a1, a1]
        a1 = np.take(E, [1], axis=ax)
        E = np.concatenate([np.take(E, [0], axis=ax) + a1, a1], axis=ax)
    return {S: float(E[_cell_of(S, E.ndim)]) for S in graded_order(E.ndim, 1)}


def _coordinate_dict(coef: Mapping[Subset, float], d: int) -> dict[Subset, float]:
    out = {S: 0.0 for S in graded_order(d, 1)}
    for key, value in coef.items():
        S = _check_axes(normalize_subset(key), d)
        if S:
            out[S] += float(value)
    return out


def theta_spin_to_01(phi: Mapping[Subset, float], d: int) -> dict[Subset, float]:
    """Convert spin-coded parameters ``φ`` to Möbius θ (0/1 coding) over ``d`` binary variables.

    ``θ_T = 2^{|T|} Σ_{S ⊇ T} (−1)^{|S|−|T|} φ_S`` for every nonempty ``T ⊆ range(d)`` (from
    ``Π_{i∈S}(2x_i − 1) = Σ_{T⊆S} 2^{|T|} (−1)^{|S|−|T|} x^T``). ``φ_∅`` only shifts the normaliser and is ignored.
    """
    phi = _coordinate_dict(phi, d)
    out = {T: 0.0 for T in phi}
    for S, value in phi.items():
        if value:
            for T in nonempty_subsets(S):
                out[T] += 2.0 ** len(T) * (-1.0) ** (len(S) - len(T)) * value
    return out


def theta_01_to_spin(theta: Mapping[Subset, float], d: int) -> dict[Subset, float]:
    """Convert Möbius θ (0/1 coding) to spin-coded parameters ``φ`` over ``d`` binary variables.

    ``φ_S = Σ_{T ⊇ S} 2^{−|T|} θ_T`` for every nonempty ``S`` (from ``x^T = 2^{−|T|} Π_{i∈T}(1 + s_i)``); the
    inverse of :func:`theta_spin_to_01`.
    """
    theta = _coordinate_dict(theta, d)
    out = {S: 0.0 for S in theta}
    for T, value in theta.items():
        if value:
            for S in nonempty_subsets(T):
                out[S] += 2.0 ** (-len(T)) * value
    return out
