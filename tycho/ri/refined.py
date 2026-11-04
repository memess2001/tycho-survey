"""Refined information (RI): chain decompositions of the KL error and local (top-order) RI.

All quantities are in nats. ``P`` is a probability table over ``d`` categorical variables (see
:mod:`tycho.ri.loglinear` for hierarchical log-linear models and m-projections).

Chain refined information
-------------------------
A complete hierarchical chain ``H_0 ⊂ H_1 ⊂ … ⊂ H_m`` adds one admissible subset at a time,
``H_j = H_{j−1} ∪ {S_j}`` (``|S_j| ≥ 2``, ``S_j ∉ H_{j−1}``, all proper nonempty subsets of ``S_j`` in ``H_{j−1}``).
The refined information of ``S_j`` along the chain is the divergence drop

    RI_C(S_j) = D(P‖P_{H_{j−1}}) − D(P‖P_{H_j}) = D(P_{H_j} ‖ P_{H_{j−1}}) ≥ 0.

The equality is the Pythagorean relation; :func:`chain_decomposition` evaluates the right-hand side, which is
non-negative by construction and free of cancellation. Summing along a chain from ``H_0`` = singletons to the full
power set completely decomposes the KL error of the independence model, i.e. the total correlation:

    TC(X) = Σ_i H(X_i) − H(X) = D(P‖P_{H_0}) = Σ_j RI_C(S_j).

For ``d = 2`` the single step is the mutual information. **RI_C depends on the chain**: when incomparable subsets
are ordered differently, their individual RI values change and only the total is invariant. The chain is therefore
always explicit (:func:`graded_chain`, a user-given chain, or :func:`greedy_selection`).

Local refined information
-------------------------
``RI(S) = D(P_S ‖ (P_S)_{∂S})`` where ``∂S`` is the family of all proper nonempty subsets of ``S`` (generators: the
``|S|`` subsets of size ``|S| − 1``; for ``|S| = 2`` the projection is the product of the two margins, so RI = MI).
It is the top-order term for ``S`` and, **unlike RI_C, does not depend on any chain**: it equals ``RI_C(S)`` for every
chain in which ``S`` enters right after its proper subsets with nothing else involving ``S``'s variables. It is
``≥ 0``, and ``0`` iff ``P_S`` has no ``|S|``-th order interaction (for binary variables: Möbius ``θ_S = 0``, or ``P_S``
on the boundary of that model). It coincides with Amari's k-cut mixed-coordinate divergence of top order and with
Schneidman et al.'s connected information of top order. It is the calibrated "spectral line" of the survey.

How local RI is computed (:func:`local_projection_batch`)
---------------------------------------------------------
* ``|S| = 2``: closed form (product of margins).
* all variables binary, ``|S| ≥ 3`` (default): exact one-dimensional solve. The tables sharing ``P``'s
  ``(m−1)``-margins form the segment ``P + t·u`` with ``u(x) = (−1)^{m−|x|}`` (the only direction with vanishing
  ``(m−1)``-margins; ``|x|`` = number of ones). The projection is its maximum-entropy point, the root of
  ``θ_top(t) = Σ_x u(x) log(P(x) + t u(x)) = 0`` (top Möbius coordinate), a strictly increasing function of ``t``.
  If ``P`` has zeros in both sign classes of ``u`` the segment is the single point ``P`` and ``RI = 0``: deterministic
  AND/OR relations, for example, are linearly separable and lie in the closure of the lower-order model. This is the
  same fixed point that IPF converges to, but exact, and without IPF's sublinear convergence at such boundary points.
* otherwise: IPF over the ``m`` generators, vectorised across the batch, with per-table convergence and the exact
  facial start of :func:`tycho.ri.loglinear.facial_support`.

Alignment with Enouen & Sugiyama (TMLR 2026)
--------------------------------------------
Checked against the paper (arXiv:2410.11964v2): **Definition 1** ``RI_{I,S}(p) = D_KL(p_{I+S}; p_I)`` for a
hierarchical collection ``I`` (which contains ``∅``) and the complete chain ``{∅} = I_0 ⊊ … ⊊ I_T = P([d])``, whose
terms sum to ``D_KL(p; u)`` (``u`` uniform), eq. (6). :func:`chain_decomposition` with ``start="empty"`` reproduces
this exactly (singletons are then chain steps with ``RI = D(p_k‖u_k) = log I_k − H(X_k)``); the default
``start="singletons"`` skips those first-order steps and decomposes the total correlation instead.
The paper's two **order-invariant canonical values** (Appendix C.1) are both implemented:

* ``RI^marg_S = RI_{I^min_S, S}`` with ``I^min_S = P(S) − S``: the **marginal refined information**. This is what
  this code calls *local RI* (:func:`local_ri`, :func:`marginal_ri`); for ``|S| = 2`` it is the mutual information.
* ``RI^cond_S = RI_{I^max_S, S}`` with ``I^max_S = P([d]) − {T ⊇ S}``: the **conditional (purified) refined
  information** (:func:`conditional_ri`), which depends on which other variables are in the table.

The paper's Figure 3 shows that ``RI_{I,S}`` of one subset can vary by orders of magnitude across contexts ``I``;
the survey therefore reports the marginal value (estimable from small sub-tables, with a known χ² null) for event
detection, and the conditional value only for small variable sets. :func:`j_information` is the paper's cheap
proxy ``J(p_S) = Σ_{T⊆S} (−1)^{|S|−|T|} D_KL(p_T; u_T)`` used as the selection score of MAHGenTa.
:func:`greedy_selection` is a strong-heredity, exact-gain variant of the paper's mode interaction selection (the
paper selects by ``|J_S|`` with a weak-heredity threshold τ = 30 % and fits θ by gradient descent with Gibbs/AIS).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tycho.ri.hypergraph import (
    Subset,
    admissible_candidates,
    down_closure,
    graded_order,
    maximal_elements,
    nonempty_subsets,
    normalize_subset,
)
from tycho.ri.loglinear import facial_support, kl, m_project, marginal, normalize_table

logger = logging.getLogger(__name__)

__all__ = [
    "RIStep",
    "ChainDecomposition",
    "chain_decomposition",
    "graded_chain",
    "greedy_selection",
    "local_ri",
    "local_ri_batch",
    "local_projection_batch",
    "local_ri_from_table",
    "marginal_ri",
    "conditional_ri",
    "j_information",
]

#: Methods of :func:`local_projection_batch`.
METHODS = ("auto", "fiber", "ipf")
#: Maximum Newton iterations of the binary fibre solver (typical tables need 3-10; boundary-hugging ones ~30).
FIBER_MAX_ITER = 200
#: Fibre solver convergence: the last Newton step in λ satisfies ``|Δλ| <= FIBER_LAMBDA_TOL·(1 + |λ|)``.
FIBER_LAMBDA_TOL = 1e-13
#: Largest Newton step in λ (a factor e^50 in the boundary distances), against wild steps far from the root.
FIBER_MAX_STEP = 50.0
#: Gains (nats) within this tolerance of the best are ties in :func:`greedy_selection`; the first candidate in
#: graded-lex order wins, which keeps the selection deterministic under floating-point noise.
GREEDY_TIE_TOL = 1e-12


# ----------------------------------------------------------------------------------------------- results
@dataclass
class RIStep:
    """One step of a chain: ``subset`` was added; ``ri = D(P_{H_j}‖P_{H_{j−1}})``.

    ``kl_before = D(P‖P_{H_{j−1}})`` and ``kl_after = D(P‖P_{H_j})``; ``kl_before − kl_after ≈ ri`` (Pythagoras).
    """

    subset: Subset
    order: int
    ri: float
    kl_before: float
    kl_after: float


@dataclass
class ChainDecomposition:
    """Refined information along one chain.

    Attributes:
        base: generators (maximal elements) of the starting family ``H_0``.
        steps: the chain steps in order.
        kl_base: ``D(P‖P_{H_0})`` (the total correlation when ``H_0`` = singletons).
        residual: ``D(P‖P_{H_m})`` of the final family (0 when the chain reaches the full set).
    """

    base: tuple[Subset, ...]
    steps: list[RIStep]
    kl_base: float
    residual: float

    @property
    def total(self) -> float:
        """``Σ step.ri``; invariant: ``kl_base ≈ total + residual``."""
        return float(sum(s.ri for s in self.steps))

    def as_dict(self) -> dict[Subset, float]:
        """``{subset: ri}`` in chain order."""
        return {s.subset: s.ri for s in self.steps}


# ----------------------------------------------------------------------------------------------- chains
def graded_chain(d: int, max_order: int | None = None) -> list[Subset]:
    """The graded-lex chain: all subsets of ``range(d)`` with ``2 <= |S| <= max_order`` sorted by ``(len, tuple)``.

    Every subset appears after all of its proper subsets, so the chain is admissible from the singletons.
    """
    return graded_order(d, 2, max_order)


def _project_family(Pn: np.ndarray, family: set[Subset], *, tol: float,
                    init: np.ndarray | None = None) -> np.ndarray:
    """m-projection onto a hierarchical family that need not contain every singleton.

    Axes covered by no member of ``family`` are uniform and independent of the rest (they carry no θ coordinate).
    With the empty family the projection is the uniform distribution ``u``.
    """
    d = Pn.ndim
    covered = sorted({i for S in family for i in S})
    if len(covered) == d:
        return m_project(Pn, family, tol=tol, init=init).Q
    shape = Pn.shape
    if not covered:
        return np.full(shape, 1.0 / Pn.size)
    pos = {a: k for k, a in enumerate(covered)}
    Qc = m_project(marginal(Pn, covered), {tuple(pos[a] for a in S) for S in family}, tol=tol).Q
    uncovered = [a for a in range(d) if a not in pos]
    n_unc = int(np.prod([shape[a] for a in uncovered]))
    Q = np.multiply.outer(Qc, np.full([shape[a] for a in uncovered], 1.0 / n_unc))
    return np.transpose(Q, np.argsort(covered + uncovered))


def chain_decomposition(P: np.ndarray, chain: Sequence[Sequence[int]], *,
                        base: Sequence[Sequence[int]] | None = None, tol: float = 1e-12,
                        start: str = "singletons") -> ChainDecomposition:
    """Refined information of each subset along an explicit chain.

    Args:
        P: probability table (normalised internally).
        chain: subsets ``S_1..S_m`` added in order; each must be admissible when it is added.
        base: subsets generating ``H_0`` in addition to the start family.
        tol: IPF tolerance of each projection (each warm-started from the previous one when possible).
        start: ``"singletons"`` (default): ``H_0`` contains all singletons, chain elements have order ≥ 2 and the
            terms decompose the total correlation. ``"empty"``: ``H_0 = {∅}`` as in Enouen & Sugiyama (2026), chain
            elements may be singletons, and a complete chain decomposes ``D(P‖u)``.

    Raises:
        ValueError: if a chain element is not admissible at its position (too small, already present, or a proper
            subset missing), uses an axis ``>= P.ndim``, or ``start`` is unknown.
    """
    if start not in ("singletons", "empty"):
        raise ValueError(f"start must be 'singletons' or 'empty', got {start!r}")
    Pn = normalize_table(P)
    d = Pn.ndim
    raw_base = () if base is None else base
    if start == "singletons":
        base_family = down_closure(raw_base, d)
    else:
        base_family = down_closure(raw_base) if raw_base else frozenset()
    min_order = 2 if start == "singletons" else 1
    family = set(base_family)
    Q = _project_family(Pn, family, tol=tol)
    kl_base = kl(Pn, Q)
    kl_prev = kl_base
    steps: list[RIStep] = []
    for pos, raw in enumerate(chain):
        S = normalize_subset(raw)
        if S and S[-1] >= d:
            raise ValueError(f"chain element {pos} {S} uses an axis >= d={d}")
        if len(S) < min_order:
            raise ValueError(f"chain element {pos} {S} has order < {min_order}"
                             + (" (singletons belong to the base)" if start == "singletons" else ""))
        if S in family:
            raise ValueError(f"chain element {pos} {S} is already in the family")
        missing = [T for T in nonempty_subsets(S, proper=True) if T not in family]
        if missing:
            raise ValueError(f"chain element {pos} {S} is not admissible: missing proper subsets {missing}")
        family.add(S)
        Q_new = _project_family(Pn, family, tol=tol, init=Q)
        kl_after = kl(Pn, Q_new)
        steps.append(RIStep(subset=S, order=len(S), ri=kl(Q_new, Q), kl_before=kl_prev, kl_after=kl_after))
        Q, kl_prev = Q_new, kl_after
    return ChainDecomposition(base=tuple(maximal_elements(base_family)) if base_family else (), steps=steps,
                              kl_base=kl_base, residual=kl_prev)


def greedy_selection(P: np.ndarray, *, max_order: int | None = None, max_steps: int | None = None,
                     min_gain: float = 0.0, tol: float = 1e-12) -> ChainDecomposition:
    """Mode interaction selection: grow a chain from the singletons, adding at each step the admissible subset with
    the largest RI gain ``D(P_{H∪{S}} ‖ P_H)``.

    Stops when no admissible candidate (of order ``<= max_order``) remains, after ``max_steps`` steps, or when the
    best gain is ``< min_gain``. With the default ``min_gain = 0`` the chain runs to completion, so its total is the
    total correlation (restricted to ``max_order``). Caveat: admissibility forces lower-order subsets in first, and a
    pure synergy (e.g. parity) has zero gain on all of them, so ``min_gain > 0`` can stop before reaching it.
    Near-ties (within :data:`GREEDY_TIE_TOL`) go to the first candidate in graded-lex order.
    """
    Pn = normalize_table(P)
    d = Pn.ndim
    family = set(down_closure((), d))
    Q = m_project(Pn, family).Q
    kl_base = kl(Pn, Q)
    kl_prev = kl_base
    steps: list[RIStep] = []
    while max_steps is None or len(steps) < max_steps:
        candidates = admissible_candidates(frozenset(family), d, max_order)
        if not candidates:
            break
        projections = [m_project(Pn, family | {S}, tol=tol, init=Q).Q for S in candidates]
        gains = np.asarray([kl(Qs, Q) for Qs in projections])
        best = int(np.flatnonzero(gains >= gains.max() - GREEDY_TIE_TOL)[0])
        if gains[best] < min_gain:
            break
        S, Q_new = candidates[best], projections[best]
        family.add(S)
        kl_after = kl(Pn, Q_new)
        steps.append(RIStep(subset=S, order=len(S), ri=float(gains[best]), kl_before=kl_prev, kl_after=kl_after))
        Q, kl_prev = Q_new, kl_after
    return ChainDecomposition(base=tuple((i,) for i in range(d)), steps=steps, kl_base=kl_base, residual=kl_prev)


# ----------------------------------------------------------------------------------------------- local RI
def _kl_rows(P: np.ndarray, log_P: np.ndarray, log_Q: np.ndarray) -> np.ndarray:
    """Row-wise ``Σ_{p>0} p (log p − log q)`` of (B, C) arrays, clamped at 0 against rounding."""
    with np.errstate(invalid="ignore"):
        terms = np.where(P > 0, P * (log_P - log_Q), 0.0)
    return np.maximum(terms.sum(axis=1), 0.0)


def _parity_signs(m: int) -> np.ndarray:
    """``u(x) = (−1)^{m−|x|}`` over the cells of a ``(2,)*m`` table in C order (``u(1..1) = +1``)."""
    weight = np.indices((2,) * m).reshape(m, -1).sum(axis=0)
    return np.where((m - weight) % 2 == 0, 1.0, -1.0)


def _fiber_eval(lam: np.ndarray, log_w: np.ndarray, log_dp: np.ndarray,
                log_dn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """θ_top and its λ-derivative on the fibre, plus ``log q`` of the ``u = +1`` and ``u = −1`` cells.

    Parametrisation (see :func:`_fiber_binary`): ``δ = W σ(λ)``, ``γ = W σ(−λ)``, ``q_+ = Δ_+ + δ``,
    ``q_− = Δ_− + γ``; everything is evaluated in log space so that roots extremely close to a boundary stay exact.
    """
    log_delta = log_w - np.logaddexp(0.0, -lam)
    log_gamma = log_w - np.logaddexp(0.0, lam)
    lqp = np.logaddexp(log_dp, log_delta[:, None])
    lqn = np.logaddexp(log_dn, log_gamma[:, None])
    g = lqp.sum(axis=1) - lqn.sum(axis=1)
    log_c = (log_delta + log_gamma - log_w)[:, None]  # log(dδ/dλ) = log(δγ/W)
    gp = np.exp(log_c - lqp).sum(axis=1) + np.exp(log_c - lqn).sum(axis=1)
    return g, gp, lqp, lqn


def _fiber_binary(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact m-projection of binary tables onto ``∂S`` and their local RI. ``P``: (B, 2, …, 2), normalised.

    The fibre is ``q(t) = P + t u`` for ``t ∈ [−a, b]`` with ``a = min_{u=+1} P`` and ``b = min_{u=−1} P``. Write
    ``W = a + b``, ``t = −a + δ = b − γ`` (``δ, γ`` = distances to the two ends) and ``Δ_± = P_± − min P_±``; then
    ``q_+ = Δ_+ + δ`` and ``q_− = Δ_− + γ`` are computed without cancellation. With ``δ = W σ(λ)`` the equation
    ``θ_top = Σ log q_+ − Σ log q_− = 0`` is smooth and strictly increasing in ``λ ∈ ℝ`` and asymptotically linear at
    both ends, so safeguarded Newton (bisection once a bracket exists) converges quickly. ``W = 0`` (zeros on both
    sign classes) means the fibre is the single point ``P``: ``Q = P`` and RI = 0.
    """
    B = P.shape[0]
    m = P.ndim - 1
    flat = P.reshape(B, -1)
    u = _parity_signs(m)
    pos, neg = np.flatnonzero(u > 0), np.flatnonzero(u < 0)
    P_pos, P_neg = flat[:, pos], flat[:, neg]
    a, b = P_pos.min(axis=1), P_neg.min(axis=1)
    with np.errstate(divide="ignore"):
        log_flat = np.log(flat)
    log_q = log_flat.copy()  # tables whose fibre is a single point keep Q = P
    live = np.flatnonzero(a + b > 0)
    if live.size:
        a, b = a[live], b[live]
        with np.errstate(divide="ignore"):
            log_dp = np.log(P_pos[live] - a[:, None])  # -inf at the cells attaining the minimum
            log_dn = np.log(P_neg[live] - b[:, None])
            log_w = np.log(a + b)
            # start at t = 0 (Q = P) when that is interior, else at the midpoint of the segment
            lam = np.where((a > 0) & (b > 0), np.log(a) - np.log(b), 0.0)
        lo = np.full(live.size, -np.inf)
        hi = np.full(live.size, np.inf)
        act = np.arange(live.size)
        for _ in range(FIBER_MAX_ITER):
            cur = lam[act]
            g, gp, _, _ = _fiber_eval(cur, log_w[act], log_dp[act], log_dn[act])
            lo[act] = np.where(g < 0, cur, lo[act])
            hi[act] = np.where(g > 0, cur, hi[act])
            new = cur - np.clip(g / gp, -FIBER_MAX_STEP, FIBER_MAX_STEP)
            # A Newton step can leave the bracket only when both ends are finite; then bisect (the midpoint of a
            # half-open bracket is never selected, so its inf - inf is harmless).
            with np.errstate(invalid="ignore"):
                mid = 0.5 * (lo[act] + hi[act])
            new = np.where((new > lo[act]) & (new < hi[act]), new, mid)
            done = (g == 0) | (np.abs(new - cur) <= FIBER_LAMBDA_TOL * (1.0 + np.abs(cur)))
            lam[act] = np.where(g == 0, cur, new)
            act = act[~done]
            if not act.size:
                break
        if act.size:
            logger.warning("binary fibre solver: %d of %d tables not converged after %d iterations",
                           act.size, B, FIBER_MAX_ITER)
        _, _, lqp, lqn = _fiber_eval(lam, log_w, log_dp, log_dn)
        log_q[np.ix_(live, pos)] = lqp
        log_q[np.ix_(live, neg)] = lqn
    return np.exp(log_q).reshape(P.shape), _kl_rows(flat, log_flat, log_q)


def _layouts(shape: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """``(pre, k, post)`` of each axis ``j`` of a C-ordered table: the axis as the middle of a 3-D view."""
    return [(int(np.prod(shape[:j])), shape[j], int(np.prod(shape[j + 1:]))) for j in range(len(shape))]


def _drop_axis_margin(Q: np.ndarray, layout: tuple[int, int, int]) -> np.ndarray:
    """Margin of every table of the batch ``Q`` (B, ...) over one axis, as a (B·pre, post) array.

    Adding the slices of the (B·pre, k, post) view is several times faster than ``ndarray.sum`` over a small middle
    axis, which dominates the cost of batched IPF on small tables.
    """
    _, k, post = layout
    R = Q.reshape(-1, k, post)
    acc = R[:, 0, :].copy()
    for v in range(1, k):
        acc += R[:, v, :]
    return acc


def _facial_start(P: np.ndarray, targets: list[np.ndarray], layouts: list[tuple[int, int, int]]) -> np.ndarray:
    """IPF start for each table: uniform on its facial support (plain uniform unless it has non-implied zeros)."""
    B, m = P.shape[0], P.ndim - 1
    start = np.ones(P.shape)  # C order: _ipf_batch updates it in place through reshaped views
    zero = P <= 0
    if zero.any():
        implied = np.zeros(P.shape, dtype=bool)
        for t, (pre, k, post) in zip(targets, layouts):
            implied |= np.broadcast_to((t <= 0).reshape(-1, 1, post), (B * pre, k, post)).reshape(P.shape)
        extra = np.flatnonzero((zero & ~implied).reshape(B, -1).any(axis=1))
        if extra.size:
            gens = [tuple(i for i in range(m) if i != j) for j in range(m)]
            cache: dict[bytes, np.ndarray] = {}  # the facial support depends on the zero pattern only
            for k in extra:
                key = zero[k].tobytes()
                if key not in cache:
                    cache[key] = facial_support(P[k], gens)
                start[k] = cache[key]
    return start / start.reshape(B, -1).sum(axis=1).reshape((B,) + (1,) * m)


def _ipf_batch(P: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """IPF onto ``∂S`` for every table of ``P`` (B, k_1..k_m), m >= 3; generator ``j`` = all axes but ``j``.

    Converged tables are frozen and removed from the working set. After a sweep the last generator is fitted
    exactly, so the convergence check covers generators ``0..m−2``; the margin of generator 0 computed for the check
    is reused by the next sweep's first update. Margins are kept as (B, pre·post) arrays (see :func:`_layouts`).
    """
    B, shape = P.shape[0], P.shape[1:]
    layouts = _layouts(shape)
    targets = [_drop_axis_margin(P, lay).reshape(B, -1) for lay in layouts]
    Q = _facial_start(P, targets, layouts)
    out = np.empty_like(P)
    act = np.arange(B)
    first = _drop_axis_margin(Q, layouts[0])
    for _ in range(int(max_iter)):
        for j, (_, k, post) in enumerate(layouts):
            cur = first if j == 0 else _drop_axis_margin(Q, layouts[j])
            ratio = np.zeros_like(cur)
            np.divide(targets[j].reshape(-1, post), cur, out=ratio, where=cur > 0)
            view = Q.reshape(-1, k, post, copy=False)  # Q is C-contiguous: a view, so the update is in place
            view *= ratio[:, None, :]
        first = _drop_axis_margin(Q, layouts[0])
        err = np.abs(first.reshape(act.size, -1) - targets[0]).max(axis=1)
        for j in range(1, len(layouts) - 1):
            dev = np.abs(_drop_axis_margin(Q, layouts[j]).reshape(act.size, -1) - targets[j]).max(axis=1)
            err = np.maximum(err, dev)
        done = err < tol
        if done.any():
            out[act[done]] = Q[done]
            keep = ~done
            first = first.reshape(act.size, -1)[keep].reshape(-1, layouts[0][2])
            act, Q = act[keep], Q[keep]
            targets = [t[keep] for t in targets]
            if not act.size:
                break
    if act.size:
        out[act] = Q
        logger.warning("batched IPF: %d of %d tables not converged after %d sweeps (tol %.3g); their RI is an "
                       "upper bound", act.size, B, max_iter, tol)
    return out


def local_projection_batch(tables: np.ndarray, *, tol: float = 1e-12, max_iter: int = 10_000,
                           method: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """m-projections of a batch of tables onto ``∂S`` and their local RI.

    Args:
        tables: (B, k_1, ..., k_m) non-negative tables, each normalised internally (each needs positive mass).
        tol, max_iter: IPF convergence tolerance (absolute, on generator margins) and sweep limit.
        method: ``"auto"`` (exact fibre solver for all-binary tables with m >= 3, IPF otherwise), ``"fiber"``
            (binary only) or ``"ipf"`` (force IPF). ``m = 1`` (RI = 0) and ``m = 2`` (product of margins) are closed
            forms for every method.

    Returns:
        ``(Q, ri)``: the normalised projections (B, k_1..k_m) and the RI values (B,) in nats.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    T = np.asarray(tables, dtype=np.float64)
    if T.ndim < 2:
        raise ValueError(f"tables must have shape (B, k_1, ..., k_m) with m >= 1, got {T.shape}")
    if not np.all(np.isfinite(T)) or np.any(T < 0):
        raise ValueError("tables must be finite and non-negative")
    B, shape = T.shape[0], T.shape[1:]
    m = len(shape)
    binary = all(k == 2 for k in shape)
    if method == "fiber" and not binary:
        raise ValueError(f"method 'fiber' requires binary tables, got shape {shape}")
    if B == 0:
        return T.copy(), np.zeros(0)
    mass = T.reshape(B, -1).sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("every table needs positive mass")
    P = np.ascontiguousarray(T / mass.reshape((B,) + (1,) * m))  # inputs may be transposed views
    if m == 1:
        return P, np.zeros(B)
    if m >= 3 and binary and method != "ipf":
        return _fiber_binary(P)
    if m == 2:
        Q = P.sum(axis=2, keepdims=True) * P.sum(axis=1, keepdims=True)
    else:
        Q = _ipf_batch(P, tol, max_iter)
    flat_P, flat_Q = P.reshape(B, -1), Q.reshape(B, -1)
    with np.errstate(divide="ignore"):
        ri = _kl_rows(flat_P, np.log(flat_P), np.log(flat_Q))
    return Q, ri


def local_ri_batch(tables: np.ndarray, *, tol: float = 1e-12, max_iter: int = 10_000,
                   method: str = "auto") -> np.ndarray:
    """Local RI (nats) of each table of a (B, k_1..k_m) batch; see :func:`local_projection_batch`."""
    return local_projection_batch(tables, tol=tol, max_iter=max_iter, method=method)[1]


def local_ri(P: np.ndarray, *, tol: float = 1e-12, max_iter: int = 10_000, method: str = "auto") -> float:
    """Local RI of the set of all axes of ``P`` (its top-order term); 0 for a 1-D table."""
    Pn = normalize_table(P)
    if Pn.ndim == 1:
        return 0.0
    return float(local_ri_batch(Pn[None], tol=tol, max_iter=max_iter, method=method)[0])


def local_ri_from_table(P: np.ndarray, subsets: Sequence[Sequence[int]]) -> np.ndarray:
    """Population local RI ``RI(S)`` of each subset, from the margins of the full table ``P`` (batched by shape).

    Subsets of order 1 have RI 0. Raises ``ValueError`` on an empty subset or an axis ``>= P.ndim``.
    """
    Pn = normalize_table(P)
    subs = [normalize_subset(S) for S in subsets]
    out = np.zeros(len(subs))
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, S in enumerate(subs):
        if not S:
            raise ValueError("empty subset")
        if S[-1] >= Pn.ndim:
            raise ValueError(f"subset {S} uses an axis >= d={Pn.ndim}")
        if len(S) >= 2:
            groups.setdefault(tuple(Pn.shape[j] for j in S), []).append(i)
    for idx in groups.values():
        out[idx] = local_ri_batch(np.stack([marginal(Pn, subs[i]) for i in idx]))
    return out


def marginal_ri(P: np.ndarray, S: Sequence[int]) -> float:
    """Marginal refined information ``RI^marg_S = D(p_S ‖ (p_S)_{P(S)−S})`` (Enouen & Sugiyama 2026, App. C.1).

    Identical to the local RI of ``S`` (see :func:`local_ri_from_table`); for ``|S| = 2`` it is the mutual information
    and for ``|S| = 1`` it is 0 (the survey's per-subset quantity measures interactions, not marginal skew).
    """
    return float(local_ri_from_table(P, [S])[0])


def conditional_ri(P: np.ndarray, S: Sequence[int], *, tol: float = 1e-12) -> float:
    """Conditional (purified) refined information ``RI^cond_S = D(p_{I+S} ‖ p_I)``, ``I = P([d]) − {T ⊇ S}``.

    ``I`` contains every subset that does not contain ``S``; its generators are ``[d] − {s}`` for ``s ∈ S``. The value
    depends on all variables of ``P`` (e.g. it vanishes for two variables that are conditionally independent given a
    third), so it is computed on the full table: use it for small variable sets only.
    """
    Pn = normalize_table(P)
    d = Pn.ndim
    S = normalize_subset(S)
    if not S:
        raise ValueError("empty subset")
    if S[-1] >= d:
        raise ValueError(f"subset {S} uses an axis >= d={d}")
    full = tuple(range(d))
    gens = [tuple(a for a in full if a != s) for s in S]
    lower = set(down_closure([g for g in gens if g]))
    Q_lo = _project_family(Pn, lower, tol=tol)
    upper = lower | {S}
    Q_hi = _project_family(Pn, upper, tol=tol, init=Q_lo if len(S) > 1 else None)
    return float(kl(Q_hi, Q_lo))


def j_information(P: np.ndarray, S: Sequence[int]) -> float:
    """``J(p_S) = Σ_{T⊆S} (−1)^{|S|−|T|} D_KL(p_T; u_T)`` (Enouen & Sugiyama 2026, eq. 3), in nats.

    Equals ``(−1)^{|S|} I(p_S)`` (multiple mutual information) for ``|S| ≥ 2``; may be negative. The paper uses
    ``|J_S|`` as a cheap proxy for refined information when selecting mode interactions.
    """
    Pn = normalize_table(P)
    S = normalize_subset(S)
    if not S:
        raise ValueError("empty subset")
    total = 0.0
    for T in nonempty_subsets(S):
        pT = marginal(Pn, T)
        total += (-1) ** (len(S) - len(T)) * kl(pT, np.full(pT.shape, 1.0 / pT.size))
    return float(total)
