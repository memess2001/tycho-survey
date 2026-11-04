"""Planted ground truth for instrument calibration (SPEC §3).

Every generator here produces a distribution whose refined-information structure is known exactly, so the RI
estimators of :mod:`tycho.ri` can be checked against the truth before they are pointed at neural networks.

Binary tables (spin coding)
---------------------------
For ``d`` binary variables with codes ``x_i ∈ {0, 1}`` and spins ``s_i = 2 x_i − 1`` (SPEC §2.1),

    log P(x) = Σ_S θ_S Π_{i∈S} s_i − ψ(θ).

A *planted interaction* is a nonzero ``θ_S`` (:func:`planted_binary`, :func:`planted_binary_batch`). Facts the
calibration relies on (all checked in ``tests/test_calibration.py``):

* For a table over exactly the variables of ``S`` (``d = |S|``), the top Möbius coordinate is ``2^{|S|} θ_S``, so the
  local RI of ``S`` (:func:`tycho.ri.refined.local_ri`) is positive iff ``θ_S ≠ 0``, whatever the lower-order terms.
* A pure ``θ_S`` (no other terms, no biases) leaves every proper sub-margin of ``S`` uniform: all proper subsets have
  RI exactly 0 (pure synergy). Variables in another, independent block have RI 0 with any subset of ``S`` because the
  table factorises.
* Marginalising over variables outside a subset can create interactions inside it (a pure 3-way term plus a bias on
  the summed-out variable induces a pairwise term). The "true" RI of a subset of a larger table is therefore always
  computed from the population table (:func:`true_local_ri`), never assumed.

Planting a target RI
--------------------
:func:`theta_for_local_ri` finds, for each of many lower-order backgrounds, the top-order coupling ``θ_top >= 0`` at
which the local RI of the full variable set equals a target value (vectorised bisection on the batched exact solver
:func:`tycho.ri.refined.local_ri_batch`). The calibration grids are therefore indexed by the *true RI* in nats, the
quantity the survey reports, rather than by a coupling whose meaning depends on the background.

Switch-on schedules
-------------------
:func:`switch_on` is the logistic ramp ``θ(t) = θ_max σ((t − t*)/w)``: ``θ(t*) = θ_max/2`` and the 10 %→90 % rise
takes ``2 ln 9 · w ≈ 4.39 w`` time units; ``w = 0`` is the step function (value ``θ_max/2`` exactly at ``t*``).

Dichotomized Gaussian (discretisation artefacts, experiment E6)
---------------------------------------------------------------
:func:`gaussian_binarized` samples a Gaussian latent and thresholds it with :func:`tycho.ri.discretize.binarize`, as
the survey does with hidden units. :func:`dichotomized_gaussian_table` gives the *population* table of
``1[Z_i > τ]`` for an equicorrelated, unit-variance latent with correlation ``ρ ∈ [0, 1)``, through the one-factor
representation ``Z_i = √ρ W + √(1−ρ) E_i``: conditional on ``W = w`` the bits are independent with
``p(w) = Φ((√ρ w − τ)/√(1−ρ))``, so a configuration with ``m`` ones has probability
``∫ φ(w) p(w)^m (1 − p(w))^{k−m} dw``, evaluated by a composite Gauss-Legendre rule to ~1e-15. The latent has only
pairwise structure, yet the bits carry higher-order log-linear interactions. With the threshold at the median
(``τ = 0``) the bit distribution is invariant under flipping all bits, so every odd-order interaction vanishes
exactly while even orders do not; asymmetric thresholds (zero with a biased latent, the 0.8-quantile) create odd
orders too. :func:`latent_threshold` converts a :func:`~tycho.ri.discretize.binarize` method into the threshold
``τ`` on the standardised latent.
"""
from __future__ import annotations

import functools
import logging
import math
import operator
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import special

from tycho.ri.discretize import BINARIZE_METHODS, binarize
from tycho.ri.hypergraph import Subset, normalize_subset
from tycho.ri.loglinear import loglinear_table, normalize_table
from tycho.ri.refined import local_ri_batch, local_ri_from_table

logger = logging.getLogger(__name__)

__all__ = [
    "planted_binary",
    "planted_binary_batch",
    "spin_features",
    "random_interactions",
    "sample_table",
    "true_local_ri",
    "theta_for_local_ri",
    "switch_on",
    "gaussian_binarized",
    "equicorrelation",
    "latent_threshold",
    "dichotomized_gaussian_table",
]

#: Largest top-order coupling :func:`theta_for_local_ri` tries before declaring a target unreachable. At
#: ``|θ| = 64`` the planted term dominates any background used in calibration by many orders of magnitude.
THETA_CAP = 64.0
#: Relative bracket width at which the bisection of :func:`theta_for_local_ri` stops (the RI then matches the
#: target to ~1e-12 relative, far below Monte Carlo error).
THETA_RTOL = 1e-13
#: Safety bound on bisection steps of :func:`theta_for_local_ri` (~50 are needed for targets >= 1e-6 nats).
BISECT_MAX_ITER = 200
#: Composite Gauss-Legendre rule of :func:`dichotomized_gaussian_table`: ``GL_PANELS`` equal panels of
#: ``GL_NODES`` nodes on ``[−GL_HALF_WIDTH, GL_HALF_WIDTH]``. ``φ(12) ≈ 2e-32``, so truncation is far below double
#: precision, and the panels (width 0.09) resolve the steepest conditional probability used (ρ = 0.95, scale 0.23).
#: Agreement with adaptive quadrature: ~1e-15.
GL_NODES = 20
GL_PANELS = 256
GL_HALF_WIDTH = 12.0


# ----------------------------------------------------------------------------------------------- binary tables
def _check_d(d: int) -> int:
    d = operator.index(d)
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    return d


def planted_binary(d: int, interactions: dict[Subset, float], *,
                   biases: dict[int, float] | None = None) -> np.ndarray:
    """Normalised table over ``d`` binary variables with spin-coded couplings.

    ``log P(x) = Σ_S interactions[S] Π_{i∈S} s_i + Σ_i biases[i] s_i − ψ`` with ``s_i = 2 x_i − 1``. A bias is the
    same as an order-1 interaction (both are added if given twice).

    Args:
        d: number of binary variables (the table has shape ``(2,) * d``).
        interactions: ``{subset: θ_S}``; subsets are any sequences of distinct axes ``< d`` (normalised to sorted
            tuples; duplicates after normalisation are summed).
        biases: optional ``{i: b_i}``.

    Raises:
        ValueError: on an empty subset, an axis ``>= d`` or a non-finite coupling.
    """
    d = _check_d(d)
    theta: dict[Subset, float] = {}
    items = list(interactions.items()) + [((i,), b) for i, b in (biases or {}).items()]
    for key, value in items:
        S = normalize_subset(key)
        if not S:
            raise ValueError("empty subset in interactions")
        if S[-1] >= d:
            raise ValueError(f"subset {S} uses an axis >= d={d}")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"coupling of {S} is not finite")
        theta[S] = theta.get(S, 0.0) + v
    return loglinear_table(theta, [2] * d)


def spin_features(d: int, subsets: Sequence[Sequence[int]]) -> np.ndarray:
    """``F[j, c] = Π_{i∈S_j} s_i(c)`` for every cell ``c`` of a ``(2,) * d`` table in C order (``F`` = ±1 floats).

    The empty subset gives a row of ones. ``planted_binary_batch`` uses ``log P ∝ θ @ F``.
    """
    d = _check_d(d)
    spins = 2.0 * np.indices((2,) * d).reshape(d, -1) - 1.0
    rows = []
    for raw in subsets:
        S = normalize_subset(raw)
        if S and S[-1] >= d:
            raise ValueError(f"subset {S} uses an axis >= d={d}")
        rows.append(np.prod(spins[list(S)], axis=0) if S else np.ones(2 ** d))
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), 2 ** d)


def planted_binary_batch(d: int, subsets: Sequence[Sequence[int]], thetas: np.ndarray) -> np.ndarray:
    """A batch of planted tables: ``P_r ∝ exp(Σ_j thetas[r, j] Π_{i∈S_j} s_i)``, shape ``(R,) + (2,) * d``.

    The vectorised twin of :func:`planted_binary` (identical up to rounding) for many coupling vectors at once.
    """
    F = spin_features(d, subsets)
    th = np.atleast_2d(np.asarray(thetas, dtype=np.float64))
    if th.shape[1] != F.shape[0]:
        raise ValueError(f"thetas must have {F.shape[0]} columns (one per subset), got shape {th.shape}")
    if not np.all(np.isfinite(th)):
        raise ValueError("thetas must be finite")
    L = th @ F
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(axis=1, keepdims=True)
    return P.reshape((th.shape[0],) + (2,) * d)


def random_interactions(d: int, orders: Sequence[int], scale: float, rng: np.random.Generator) -> dict[Subset, float]:
    """Random spin couplings ``θ_S ~ Normal(0, scale²)``, i.i.d., for every subset ``S ⊆ range(d)`` with
    ``|S| ∈ orders``.

    Subsets are drawn in graded-lex order, so the result is a deterministic function of the generator state. Order 1
    gives biases. ``scale = 0`` gives all-zero couplings (the draws are still consumed, keeping streams aligned).

    Raises:
        ValueError: if an order is outside ``[1, d]`` or ``scale`` is negative or not finite.
    """
    d = _check_d(d)
    if not (math.isfinite(scale) and scale >= 0):
        raise ValueError(f"scale must be finite and >= 0, got {scale}")
    ks = sorted({operator.index(k) for k in orders})
    if ks and (ks[0] < 1 or ks[-1] > d):
        raise ValueError(f"orders must lie in [1, d={d}], got {list(orders)}")
    subsets = [S for k in ks for S in combinations(range(d), k)]
    values = scale * rng.standard_normal(len(subsets))
    return {S: float(v) for S, v in zip(subsets, values)}


def sample_table(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` i.i.d. draws from the probability table ``P`` as an ``(n, d)`` int64 array of codes (``d = P.ndim``).

    Inverse-CDF sampling over the cells in C order; cells of probability 0 are never drawn. ``P`` is normalised first.
    """
    Pn = normalize_table(P)
    n = operator.index(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    cdf = np.cumsum(Pn.ravel())
    cdf /= cdf[-1]  # exactly 1 at the end, so every u in [0, 1) falls inside the table
    cells = np.searchsorted(cdf, rng.random(n), side="right")
    return np.stack(np.unravel_index(cells, Pn.shape), axis=1).astype(np.int64, copy=False)


def true_local_ri(P: np.ndarray, subsets: Sequence[Sequence[int]]) -> np.ndarray:
    """Population local RI (nats) of each subset, computed from the margins of the full table ``P``.

    Order-1 subsets have RI 0. This is the ground truth the finite-sample estimators are compared with.
    """
    return local_ri_from_table(P, subsets)


def theta_for_local_ri(target: float, d: int, background_subsets: Sequence[Sequence[int]],
                       background_thetas: np.ndarray, *, theta_cap: float = THETA_CAP,
                       rtol: float = THETA_RTOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-order couplings that plant a given local RI on the full variable set, for a batch of backgrounds.

    For each row ``r`` of ``background_thetas`` (couplings of ``background_subsets``, all of order ``< d``), find
    ``θ_r >= 0`` such that the table ``P_r ∝ exp(Σ_j bg[r, j] s_{S_j} + θ_r s_{0..d−1})`` has local RI of
    ``range(d)`` equal to ``target``. Vectorised bisection: the bracket ``[0, hi]`` is grown by doubling until
    ``RI(hi) >= target`` (``RI(0) = 0`` because the background has no top-order term), then halved until its width
    is below ``rtol·hi``. Bisection needs only the sign change, not monotonicity. Rows whose RI stays below the target
    up to ``theta_cap`` get NaN.

    Args:
        target: local RI in nats (``>= 0``; 0 gives ``θ = 0``).
        d: number of binary variables (``>= 2``).
        background_subsets: subsets of order ``< d``.
        background_thetas: ``(R, len(background_subsets))`` spin couplings.

    Returns:
        ``(theta, tables, ri)``: the couplings ``(R,)`` (NaN where unreachable), the planted tables
        ``(R,) + (2,) * d`` (the background-only table where unreachable) and their exact local RI ``(R,)``.
    """
    d = _check_d(d)
    if d < 2:
        raise ValueError("planting an interaction needs d >= 2")
    if not (math.isfinite(target) and target >= 0):
        raise ValueError(f"target must be finite and >= 0, got {target}")
    top = tuple(range(d))
    subs = [normalize_subset(S) for S in background_subsets]
    if any(len(S) >= d for S in subs):
        raise ValueError("background subsets must have order < d")
    bg = np.atleast_2d(np.asarray(background_thetas, dtype=np.float64)).reshape(-1, len(subs))
    R = bg.shape[0]
    F_bg = spin_features(d, subs)
    f_top = spin_features(d, [top])[0]
    L0 = bg @ F_bg

    def tables(theta: np.ndarray) -> np.ndarray:
        L = L0 + theta[:, None] * f_top[None, :]
        L -= L.max(axis=1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(axis=1, keepdims=True)
        return P.reshape((R,) + (2,) * d)

    def ri_of(theta: np.ndarray) -> np.ndarray:
        return local_ri_batch(tables(theta))

    if target == 0.0 or R == 0:
        theta = np.zeros(R)
        P = tables(theta)
        return theta, P, local_ri_batch(P) if R else np.zeros(0)
    lo = np.zeros(R)  # RI(0) = 0 < target: the background has no top-order term
    hi = np.ones(R)
    reached = ri_of(hi) >= target
    while not reached.all() and hi.max() < theta_cap:
        grow = ~reached
        hi[grow] = np.minimum(2.0 * hi[grow], theta_cap)
        reached[grow] = ri_of(hi)[grow] >= target
    for _ in range(BISECT_MAX_ITER):
        width = hi - lo
        if np.all(width[reached] <= rtol * hi[reached]):
            break
        mid = 0.5 * (lo + hi)
        below = ri_of(mid) < target
        lo = np.where(reached & below, mid, lo)
        hi = np.where(reached & ~below, mid, hi)
    theta = np.where(reached, 0.5 * (lo + hi), np.nan)
    P = tables(np.where(reached, theta, 0.0))
    if not reached.all():
        logger.warning("theta_for_local_ri: target RI %.3g unreachable for %d of %d backgrounds (theta_cap=%g)",
                       target, int((~reached).sum()), R, theta_cap)
    return theta, P, local_ri_batch(P)


# ----------------------------------------------------------------------------------------------- schedules
def switch_on(t: np.ndarray, t_star: float, width: float, theta_max: float) -> np.ndarray:
    """Logistic switch-on ramp ``θ(t) = θ_max σ((t − t*) / width)`` (float64 array of the shape of ``t``).

    Monotone in ``t`` (non-decreasing for ``θ_max >= 0``), ``θ(t*) = θ_max / 2``, limits 0 and ``θ_max``. The
    10 %→90 % rise takes ``2 ln 9 · width ≈ 4.39 width``. ``width = 0`` is the step ``θ_max·1[t > t*]`` with value
    ``θ_max/2`` at ``t = t*`` (the limit of the ramp).

    Raises:
        ValueError: if ``width`` is negative or a parameter is not finite.
    """
    if not all(math.isfinite(float(v)) for v in (t_star, width, theta_max)):
        raise ValueError("t_star, width and theta_max must be finite")
    if width < 0:
        raise ValueError(f"width must be >= 0, got {width}")
    t = np.asarray(t, dtype=np.float64)
    if width == 0:
        return theta_max * np.where(t > t_star, 1.0, np.where(t == t_star, 0.5, 0.0))
    return theta_max * special.expit((t - t_star) / width)


# ----------------------------------------------------------------------------------------------- Gaussian latents
def equicorrelation(d: int, rho: float) -> np.ndarray:
    """``d × d`` correlation matrix with unit diagonal and every off-diagonal entry ``rho``.

    Positive definite for ``−1/(d−1) < rho < 1``.
    """
    d = _check_d(d)
    if not math.isfinite(rho) or (d > 1 and not -1.0 / (d - 1) < rho < 1.0):
        raise ValueError(f"rho={rho} does not give a positive definite {d}x{d} equicorrelation matrix")
    C = np.full((d, d), float(rho))
    np.fill_diagonal(C, 1.0)
    return C


def gaussian_binarized(corr: np.ndarray, n: int, rng: np.random.Generator, method: str = "median", *,
                       q: float = 0.5, mean: float | np.ndarray = 0.0) -> np.ndarray:
    """Sample ``Z ~ Normal(mean, corr)`` ``n`` times and binarise each column with :func:`tycho.ri.discretize.binarize`.

    Args:
        corr: ``(m, m)`` symmetric positive semi-definite covariance (normally a correlation matrix).
        n: number of samples.
        method: a :data:`~tycho.ri.discretize.BINARIZE_METHODS` rule (``"median"``, ``"zero"``, ``"mean"``,
            ``"quantile"``); thresholds are per-column *sample* statistics, as in the survey.
        q: quantile level for ``"quantile"``.
        mean: latent mean (scalar or ``(m,)``); only ``"zero"`` depends on it (the others are shift invariant).

    Returns:
        ``(n, m)`` int64 array of 0/1 codes.
    """
    C = np.asarray(corr, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.all(np.isfinite(C)):
        raise ValueError(f"corr must be a finite square matrix, got shape {C.shape}")
    if not np.allclose(C, C.T, rtol=0.0, atol=1e-12):
        raise ValueError("corr must be symmetric")
    if method not in BINARIZE_METHODS:
        raise ValueError(f"method must be one of {BINARIZE_METHODS}, got {method!r}")
    n = operator.index(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    m = C.shape[0]
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(C)
        if evals.min() < -1e-10 * max(1.0, evals.max()):
            raise ValueError("corr must be positive semi-definite") from None
        L = evecs * np.sqrt(np.clip(evals, 0.0, None))[None, :]
    mu = np.broadcast_to(np.asarray(mean, dtype=np.float64), (m,))
    Z = rng.standard_normal((n, m)) @ L.T + mu[None, :]
    return binarize(Z, method=method, q=q).astype(np.int64)


def latent_threshold(method: str, *, q: float = 0.5, mean: float = 0.0) -> float:
    """Population threshold ``τ`` on the standardised latent ``Z − mean`` (unit variance) of a binarisation rule.

    ``binarize`` on a column of ``Normal(mean, 1)`` samples converges to ``1[Z − mean > τ]`` with ``τ = 0`` for
    ``"median"`` and ``"mean"``, ``τ = −mean`` for ``"zero"`` and ``τ = Φ^{-1}(q)`` for ``"quantile"``. The bit is 1
    with probability ``Φ(−τ)``.
    """
    if method in ("median", "mean"):
        return 0.0
    if method == "zero":
        return -float(mean)
    if method == "quantile":
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must lie in (0, 1), got {q}")
        return float(special.ndtri(q))
    raise ValueError(f"method must be one of {BINARIZE_METHODS}, got {method!r}")


@functools.lru_cache(maxsize=1)
def _gl_rule() -> tuple[np.ndarray, np.ndarray]:
    """Nodes ``w`` and weights ``ω·φ(w)`` of the composite Gauss-Legendre rule for ``∫ φ(w) f(w) dw`` (read-only)."""
    x, wx = special.roots_legendre(GL_NODES)
    edges = np.linspace(-GL_HALF_WIDTH, GL_HALF_WIDTH, GL_PANELS + 1)
    a, b = edges[:-1, None], edges[1:, None]
    nodes = (0.5 * (b - a) * x[None, :] + 0.5 * (a + b)).ravel()
    weights = (0.5 * (b - a) * wx[None, :]).ravel() * np.exp(-0.5 * nodes ** 2) / math.sqrt(2.0 * math.pi)
    nodes.flags.writeable = False
    weights.flags.writeable = False
    return nodes, weights


def dichotomized_gaussian_table(k: int, rho: float, threshold: float) -> np.ndarray:
    """Population table ``(2,) * k`` of the bits ``1[Z_i > threshold]``, ``Z`` equicorrelated ``Normal(0, C_ρ)``.

    Uses the one-factor representation (module docstring); valid for ``0 <= rho < 1``. The table is exchangeable
    (it depends on the number of ones only). For ``threshold = 0`` it is invariant under flipping all bits.
    """
    k = _check_d(k)
    if not (math.isfinite(rho) and 0.0 <= rho < 1.0):
        raise ValueError(f"rho must lie in [0, 1), got {rho}")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if rho == 0.0:
        z = np.array([-threshold])
        weights = np.array([1.0])
    else:
        nodes, weights = _gl_rule()
        z = (math.sqrt(rho) * nodes - threshold) / math.sqrt(1.0 - rho)
    log_p1, log_p0 = special.log_ndtr(z), special.log_ndtr(-z)
    m = np.arange(k + 1)[:, None]
    per_count = (np.exp(m * log_p1[None, :] + (k - m) * log_p0[None, :]) * weights[None, :]).sum(axis=1)
    ones = np.indices((2,) * k).sum(axis=0)
    table = per_count[ones]
    return table / table.sum()
