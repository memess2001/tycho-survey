"""Finite-sample estimation of local refined information, and the subset scan used for RI spectroscopy.

Estimator
---------
For a subset ``S`` with empirical marginal table ``P̂_S`` (optionally weighted, optionally with one soft column), the
plug-in estimate is the local RI of that table (:mod:`tycho.ri.refined`):

    RI_plugin(S) = D(P̂_S ‖ (P̂_S)_{∂S}).

The m-projection onto ``∂S`` is the MLE of the model without the top-order term, so ``RI_plugin`` is the
per-sample log-likelihood ratio of the nested hierarchical log-linear models ``∂S ⊂ 2^S`` and

    G² = 2 n_eff RI_plugin

is the likelihood-ratio statistic for H0 "no ``|S|``-th order interaction". Under H0, ``G² → χ²_df`` with
``df = Π_{i∈S} (k_i − 1)`` (the number of free parameters of the top-order term), so
``p_value = chi2.sf(G², df)``. ``E[G²] ≈ df`` gives the first-order null bias ``E[RI_plugin] ≈ df / (2 n_eff)``
(to first order the same bias holds under alternatives, as for the Miller-Madow correction of plug-in mutual
information), hence

    RI_corrected = max(0, RI_plugin − df / (2 n_eff)).

``df = 0`` (a variable with a single state, or ``|S| < 2``) means there is no interaction term: ``p = 1`` and
``RI_corrected = 0``.

Null calibration
----------------
The χ² approximation degrades for sparse tables (expected cell counts ≲ 5). With structural zeros the effective df
is smaller, which makes the χ² p-value conservative. ``null="parametric_bootstrap"`` draws ``n_null`` multinomial
tables of size ``round(n_eff)`` from the fitted null (the m-projection of ``P̂_S`` onto ``∂S``), recomputes RI and
reports the Monte Carlo p-value ``(1 + #{RI_null ≥ RI_obs}) / (n_null + 1)``.

Weights, soft columns, n_eff
----------------------------
Sample weights ``w_n`` default to ``n_eff = (Σw)² / Σw²`` (Kish). A soft column (e.g. a model's predictive
distribution) contributes probability mass instead of a sampled code, which removes that variable's sampling noise.
For subsets containing it, ``G²`` is therefore stochastically smaller than ``χ²_df`` under H0: both nulls are
conservative and the bias correction over-corrects. ``n_eff`` can be overridden.

Scans
-----
:func:`ri_scan` estimates every candidate subset (:func:`candidate_subsets`), controls the false discovery rate
across them with Benjamini-Hochberg (:func:`bh_fdr`), and marks a subset *detected* iff
``q <= alpha and RI_corrected >= min_effect``. Candidates are grouped by table shape and soft-column position. The
marginal tables of a group are counted together (one ``bincount`` per chunk of subsets), and their RI comes from
the vectorised :func:`tycho.ri.refined.local_projection_batch` (exact fibre solver for binary tables, batched IPF
otherwise). Randomness (bootstrap) comes only from the ``rng`` argument. When it is ``None`` a generator seeded
with :data:`DEFAULT_SEED` is used, so results are reproducible; pass your own generator for independent draws
across calls.
"""
from __future__ import annotations

import logging
import math
import operator
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from tycho.ri.hypergraph import Subset, graded_key, normalize_subset
from tycho.ri.loglinear import _as_codes, _check_soft, _check_weights, _strides
from tycho.ri.refined import local_projection_batch, local_ri_batch
from tycho.ri.variables import VariableSet

logger = logging.getLogger(__name__)

__all__ = [
    "NULLS",
    "DEFAULT_SEED",
    "ri_df",
    "RIEstimate",
    "estimate_local_ri",
    "bh_fdr",
    "candidate_subsets",
    "ScanResult",
    "ri_scan",
]

#: Supported null distributions.
NULLS = ("chi2", "parametric_bootstrap")
#: Seed of the generator used when ``rng`` is None and randomness is needed (reproducible default).
DEFAULT_SEED = 0
#: Upper bound on (sample, subset) pairs counted per chunk, and on cells per bootstrap batch (bounds the working
#: memory to ~100 MB regardless of the number of subsets).
_CHUNK = 1 << 22
#: Absolute tolerance (nats) for "null statistic ≥ observed" in the bootstrap p-value. Near-ties count as
#: exceedances, which keeps the Monte Carlo p-value conservative (e.g. both values 0 up to rounding).
BOOTSTRAP_TIE_TOL = 1e-12


def ri_df(cards: Sequence[int]) -> int:
    """Degrees of freedom ``Π (k_i − 1)`` of the top-order term of a subset with these cardinalities.

    Fewer than two variables have no interaction term (main effects are always in the model): df = 0.
    """
    cards = [int(k) for k in cards]
    if len(cards) < 2:
        return 0
    return int(math.prod(k - 1 for k in cards))


@dataclass
class RIEstimate:
    """Estimated local RI of one subset (nats). See the module docstring for the definitions."""

    subset: Subset
    order: int
    ri_plugin: float
    ri_corrected: float
    df: int
    n_eff: float
    g2: float
    p_value: float
    ci_low: float = math.nan
    ci_high: float = math.nan
    null: str = "chi2"


@dataclass(frozen=True)
class _Data:
    """Validated observations in counting layout.

    Attributes:
        XT: (d, N) codes, transposed and C-contiguous so that gathering the columns of many subsets copies whole
            rows; uint8 when every cardinality is <= 256 (else int64). The soft column's codes are 0.
        cards: cardinalities.
        soft_col, soft: the soft column index and its (N, k) probabilities, or None.
    """

    XT: np.ndarray
    cards: list[int]
    soft_col: int | None
    soft: np.ndarray | None

    @property
    def n(self) -> int:
        return int(self.XT.shape[1])


def _code_dtype(n_cells: int) -> type[np.integer]:
    """Smallest unsigned dtype holding cell codes ``0..n_cells−1`` (int64 beyond uint16)."""
    if n_cells <= 1 << 8:
        return np.uint8
    if n_cells <= 1 << 16:
        return np.uint16
    return np.int64


def _count_tables(data: _Data, S_arr: np.ndarray, shape: tuple[int, ...], soft_pos: int,
                  w: np.ndarray | None) -> np.ndarray:
    """Weighted marginal count tables (C, *shape) of the C subsets in ``S_arr`` (C, m), one ``bincount`` per call.

    Cell codes are built in the smallest integer type (memory traffic dominates), then subset ``c``'s cells are
    offset by ``c·n_cells`` so that all C tables share one ``bincount``. ``soft_pos`` is the position of the soft
    column within the subsets (−1 if absent); its state ``v`` receives mass ``w_n soft[n, v]``.
    """
    n = data.n
    C, m = S_arr.shape
    n_cells = int(np.prod(shape))
    strides = _strides(shape)
    ctype = _code_dtype(n_cells)
    codes = np.zeros((C, n), dtype=ctype)
    for p in range(m):
        if p != soft_pos:
            term = data.XT[S_arr[:, p]].astype(ctype, copy=False)
            term *= ctype(strides[p])
            codes += term
    flat = codes.astype(np.int64)
    flat += np.arange(C, dtype=np.int64)[:, None] * n_cells
    if soft_pos < 0:
        wts = None if w is None else np.broadcast_to(w[None, :], (C, n)).ravel()
        counts = np.bincount(flat.ravel(), weights=wts, minlength=C * n_cells).astype(np.float64)
    else:
        counts = np.zeros(C * n_cells)
        for v in range(shape[soft_pos]):
            mass = data.soft[:, v] if w is None else w * data.soft[:, v]
            counts += np.bincount((flat + v * strides[soft_pos]).ravel(),
                                  weights=np.broadcast_to(mass[None, :], (C, n)).ravel(), minlength=C * n_cells)
    return counts.reshape((C,) + shape)


def _group_subsets(subsets: list[Subset], cards: list[int],
                   soft_col: int | None) -> dict[tuple[tuple[int, ...], int], np.ndarray]:
    """Indices of the subsets of order >= 2, grouped by (table shape, position of the soft column or −1)."""
    groups: dict[tuple[tuple[int, ...], int], list[int]] = {}
    for i, S in enumerate(subsets):
        if len(S) >= 2:
            key = (tuple(cards[j] for j in S), S.index(soft_col) if soft_col in S else -1)
            groups.setdefault(key, []).append(i)
    return {key: np.asarray(idx, dtype=np.int64) for key, idx in groups.items()}


def _bootstrap_pvalues(Q: np.ndarray, ri_obs: np.ndarray, n: int, n_null: int, pseudocount: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Parametric-bootstrap p-values of C subsets from their fitted null tables ``Q`` (C, *shape)."""
    C, shape = Q.shape[0], Q.shape[1:]
    flat = Q.reshape(C, -1)
    flat = flat / flat.sum(axis=1, keepdims=True)
    exceed = np.zeros(C)
    batch = max(1, _CHUNK // (n_null * flat.shape[1]))
    for lo in range(0, C, batch):
        pv = flat[lo:lo + batch]
        draws = rng.multinomial(n, pv, size=(n_null, pv.shape[0])).astype(np.float64) + pseudocount
        null_ri = local_ri_batch(draws.reshape((-1,) + shape)).reshape(n_null, pv.shape[0])
        exceed[lo:lo + batch] = (null_ri >= ri_obs[lo:lo + batch][None, :] - BOOTSTRAP_TIE_TOL).sum(axis=0)
    return (1.0 + exceed) / (n_null + 1.0)


def _plugin_ri(data: _Data, subsets: list[Subset], groups: dict[tuple[tuple[int, ...], int], np.ndarray],
               w: np.ndarray | None, pseudocount: float,
               bootstrap: tuple[int, int, np.random.Generator] | None) -> tuple[np.ndarray, np.ndarray]:
    """Plug-in local RI of every subset (0 for order < 2) and, with ``bootstrap = (n, n_null, rng)``, the
    parametric-bootstrap p-values (1 where not computed)."""
    ri = np.zeros(len(subsets))
    p_boot = np.ones(len(subsets))
    for (shape, soft_pos), members in groups.items():
        S_arr = np.asarray([subsets[i] for i in members], dtype=np.int64)
        per_subset = data.n * (shape[soft_pos] if soft_pos >= 0 else 1)
        chunk = max(1, _CHUNK // max(1, per_subset))
        for lo in range(0, members.size, chunk):
            sel = members[lo:lo + chunk]
            tables = _count_tables(data, S_arr[lo:lo + chunk], shape, soft_pos, w) + pseudocount
            Q, r = local_projection_batch(tables)
            ri[sel] = r
            if bootstrap is not None:
                n, n_null, rng = bootstrap
                p_boot[sel] = _bootstrap_pvalues(Q, r, n, n_null, pseudocount, rng)
    return ri, p_boot


def _corrected(ri: np.ndarray, df: np.ndarray, n_eff: float) -> np.ndarray:
    return np.where(df > 0, np.maximum(0.0, ri - df / (2.0 * n_eff)), 0.0)


def estimate_local_ri(X: np.ndarray, cards: Sequence[int], subsets: Sequence[Sequence[int]], *,
                      soft: dict[int, np.ndarray] | None = None, weights: np.ndarray | None = None,
                      null: str = "chi2", n_null: int = 200, ci: bool = False, n_boot: int = 200,
                      ci_level: float = 0.95, rng: np.random.Generator | None = None, pseudocount: float = 0.0,
                      n_eff: float | None = None) -> list[RIEstimate]:
    """Estimate the local RI of each subset from samples.

    Args:
        X: (N, d) integer codes; column ``i`` in ``[0, cards[i])`` (a soft column's codes are ignored).
        cards: cardinalities of the d columns.
        subsets: index subsets (any order; normalised to sorted tuples). Order-1 subsets get RI 0, df 0, p 1.
        soft: optional ``{j: (N, cards[j]) row-stochastic}``, at most one column (see :func:`joint_table`).
        weights: optional (N,) non-negative sample weights.
        null: ``"chi2"`` (asymptotic LRT) or ``"parametric_bootstrap"`` (``n_null`` draws from the fitted null).
        ci: add a row-bootstrap percentile CI (``n_boot`` resamples, level ``ci_level``) of ``ri_corrected``.
        rng: generator for the bootstrap(s); default ``np.random.default_rng(DEFAULT_SEED)``.
        pseudocount: added to every cell of every marginal table (sample units), observed and bootstrap alike.
        n_eff: effective sample size; default N, or ``(Σw)²/Σw²`` with weights.

    Returns:
        One :class:`RIEstimate` per subset, in input order.

    Raises:
        ValueError: on malformed input, N = 0, zero total weight, or an unknown ``null``.
    """
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")
    X0 = np.asarray(X)
    if X0.ndim != 2:
        raise ValueError(f"X must be 2-D (N, d), got shape {X0.shape}")
    n = X0.shape[0]
    if n == 0:
        raise ValueError("no samples (N = 0)")
    soft_col, soft_arr = _check_soft(soft, n, [operator.index(k) for k in cards])
    codes, cards = _as_codes(X0, cards, soft_col)
    small = max(cards, default=1) <= 1 << 8
    XT = np.ascontiguousarray(codes.T, dtype=np.uint8 if small else np.int64)
    data = _Data(XT, cards, soft_col, soft_arr)
    w = _check_weights(weights, n)
    if w is not None and not w.sum() > 0:
        raise ValueError("weights have zero total")
    if not (math.isfinite(pseudocount) and pseudocount >= 0):
        raise ValueError("pseudocount must be finite and >= 0")
    if n_eff is None:
        n_eff_val = float(n) if w is None else float(w.sum() ** 2 / np.sum(w ** 2))
    else:
        n_eff_val = float(n_eff)
        if not (math.isfinite(n_eff_val) and n_eff_val > 0):
            raise ValueError("n_eff must be finite and > 0")
    subs = [normalize_subset(S) for S in subsets]
    for S in subs:
        if not S:
            raise ValueError("empty subset")
        if S[-1] >= len(cards):
            raise ValueError(f"subset {S} uses a column >= d={len(cards)}")
    needs_rng = null == "parametric_bootstrap" or ci
    if needs_rng:
        if null == "parametric_bootstrap" and n_null < 1:
            raise ValueError("n_null must be >= 1")
        if ci and (n_boot < 1 or not 0.0 < ci_level < 1.0):
            raise ValueError("ci needs n_boot >= 1 and 0 < ci_level < 1")
        rng = np.random.default_rng(DEFAULT_SEED) if rng is None else rng
    groups = _group_subsets(subs, cards, soft_col)
    bootstrap = (max(1, int(round(n_eff_val))), int(n_null), rng) if null == "parametric_bootstrap" else None
    ri, p_boot = _plugin_ri(data, subs, groups, w, pseudocount, bootstrap)
    df = np.zeros(len(subs), dtype=np.int64)  # order-1 subsets keep df = 0
    for (shape, _), members in groups.items():
        df[members] = ri_df(shape)
    g2 = 2.0 * n_eff_val * ri
    if null == "chi2":
        p = np.where(df > 0, chi2.sf(g2, np.maximum(df, 1)), 1.0)
    else:
        p = np.where(df > 0, p_boot, 1.0)
    ri_corr = _corrected(ri, df, n_eff_val)
    ci_low = np.full(len(subs), math.nan)
    ci_high = np.full(len(subs), math.nan)
    if ci:
        reps = np.empty((n_boot, len(subs)))
        for b in range(n_boot):
            mult = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
            wb = mult if w is None else mult * w
            if not wb.sum() > 0:  # only zero-weight rows drawn: no table to estimate from
                reps[b] = np.nan
                continue
            ri_b, _ = _plugin_ri(data, subs, groups, wb, pseudocount, None)
            if n_eff is not None:
                n_eff_b = n_eff_val
            elif w is None:
                n_eff_b = float(n)
            else:  # Kish formula over the resample (row n repeated mult[n] times)
                n_eff_b = float(wb.sum() ** 2 / np.sum(mult * w ** 2))
            reps[b] = _corrected(ri_b, df, n_eff_b)
        tail = (1.0 - ci_level) / 2.0
        ci_low, ci_high = np.nanquantile(reps, [tail, 1.0 - tail], axis=0)
    return [RIEstimate(subset=S, order=len(S), ri_plugin=float(ri[i]), ri_corrected=float(ri_corr[i]),
                       df=int(df[i]), n_eff=n_eff_val, g2=float(g2[i]), p_value=float(p[i]),
                       ci_low=float(ci_low[i]), ci_high=float(ci_high[i]), null=null)
            for i, S in enumerate(subs)]


def bh_fdr(pvals: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure.

    ``q_(i) = min_{j ≥ i} min(1, m p_(j) / j)`` over the ``m`` non-NaN p-values sorted ascending; a hypothesis is
    rejected iff ``q <= alpha`` (equivalent to the step-up rule at level alpha). NaN p-values are not tested: they do
    not count in ``m``, get ``q = NaN`` and are never rejected.

    Returns:
        ``(rejected, q_values)`` with the shape of ``pvals``.
    """
    p = np.asarray(pvals, dtype=np.float64)
    flat = p.ravel()
    q = np.full(flat.shape, np.nan)
    rejected = np.zeros(flat.shape, dtype=bool)
    ok = ~np.isnan(flat)
    m = int(ok.sum())
    if m:
        pv = flat[ok]
        if np.any((pv < 0) | (pv > 1)):
            raise ValueError("p-values must lie in [0, 1]")
        order = np.argsort(pv, kind="stable")
        scaled = pv[order] * m / np.arange(1, m + 1)
        q_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
        q_ok = np.empty(m)
        q_ok[order] = q_sorted
        q[ok] = q_ok
        rejected[ok] = q_ok <= alpha
    return rejected.reshape(p.shape), q.reshape(p.shape)


def candidate_subsets(d: int, *, max_order: int, min_order: int = 2, must_include: Sequence[int] | None = None,
                      among: Sequence[int] | None = None) -> list[Subset]:
    """Subsets ``S`` with ``min_order <= |S| <= max_order`` that contain every index of ``must_include``.

    Members are drawn from ``among`` (default all ``d`` columns) plus the ``must_include`` columns; orders below 1
    are clamped to 1. With a pool of ``n`` columns of which ``r`` are required, the count is
    ``Σ_k C(n − r, k − r)`` over ``max(min_order, r) <= k <= max_order``. Graded-lex order.
    """
    must = sorted({int(i) for i in (must_include or ())})
    pool = sorted(set(range(d) if among is None else (int(i) for i in among)) | set(must))
    if pool and (pool[0] < 0 or pool[-1] >= d):
        raise ValueError(f"column indices must lie in [0, {d})")
    rest = [i for i in pool if i not in set(must)]
    out: list[Subset] = []
    for k in range(max(int(min_order), len(must), 1), min(int(max_order), len(pool)) + 1):
        for extra in combinations(rest, k - len(must)):
            out.append(tuple(sorted(must + list(extra))))
    out.sort(key=graded_key)
    return out


@dataclass
class ScanResult:
    """Local RI of all candidate subsets of one :class:`VariableSet` (one spectrum).

    Attributes:
        names: variable names (subset indices refer to this list).
        subsets: candidate subsets (graded-lex order).
        estimates: one :class:`RIEstimate` per subset.
        q_values: Benjamini-Hochberg q-values across the subsets.
        detected: ``(q <= alpha) & (ri_corrected >= min_effect)``.
        alpha, min_effect: the detection thresholds used.
    """

    names: list[str]
    subsets: list[Subset]
    estimates: list[RIEstimate]
    q_values: np.ndarray
    detected: np.ndarray
    alpha: float = math.nan
    min_effect: float = math.nan

    def to_records(self) -> list[dict]:
        """One JSON-friendly dict per subset (indices, names, estimates, q-value, detection flag)."""
        records = []
        for S, est, q, det in zip(self.subsets, self.estimates, self.q_values, self.detected):
            records.append({
                "subset": list(S), "names": [self.names[i] for i in S], "order": est.order,
                "ri_plugin": est.ri_plugin, "ri_corrected": est.ri_corrected, "df": est.df, "n_eff": est.n_eff,
                "g2": est.g2, "p_value": est.p_value, "q_value": float(q), "detected": bool(det),
                "ci_low": est.ci_low, "ci_high": est.ci_high, "null": est.null,
            })
        return records


def ri_scan(vs: VariableSet, *, max_order: int = 3, min_order: int = 2, must_include: Sequence[int] | None = None,
            alpha: float = 0.01, min_effect: float = 1e-3, null: str = "chi2", max_subsets: int = 200_000,
            rng: np.random.Generator | None = None, weights: np.ndarray | None = None, n_null: int = 200,
            among: Sequence[int] | None = None) -> ScanResult:
    """Scan all candidate subsets of ``vs`` for local RI (one RI "spectrum").

    ``must_include`` defaults to ``vs.must_include`` (pass ``[]`` to drop it). ``detected = (q <= alpha) &
    (ri_corrected >= min_effect)`` with BH q-values across all candidates. ``weights``, ``n_null`` and ``among`` are
    passed to :func:`estimate_local_ri` / :func:`candidate_subsets`.

    Raises:
        ValueError: if the number of candidates exceeds ``max_subsets`` (or as :func:`estimate_local_ri`).
    """
    must = vs.must_include if must_include is None else must_include
    cands = candidate_subsets(vs.d, max_order=max_order, min_order=min_order, must_include=must, among=among)
    if len(cands) > max_subsets:
        raise ValueError(f"{len(cands)} candidate subsets exceed max_subsets={max_subsets}")
    estimates = estimate_local_ri(vs.X, vs.cards, cands, soft=vs.soft, weights=weights, null=null, n_null=n_null,
                                  rng=rng) if cands else []
    p = np.asarray([e.p_value for e in estimates], dtype=np.float64)
    _, q = bh_fdr(p, alpha)
    ri = np.asarray([e.ri_corrected for e in estimates], dtype=np.float64)
    detected = (q <= alpha) & (ri >= min_effect)
    logger.debug("ri_scan: %d subsets, %d detected", len(cands), int(detected.sum()))
    return ScanResult(names=list(vs.names), subsets=cands, estimates=estimates, q_values=q, detected=detected,
                      alpha=float(alpha), min_effect=float(min_effect))
