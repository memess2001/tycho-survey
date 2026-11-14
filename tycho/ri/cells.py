"""Per-cell statistics of a binary-input, binary-output subset, and the θ and η quantities they determine (sky v2, A0).

Setting. Binary inputs x ∈ {0, 1}^d (spin s = 2x − 1), a binary output ŷ given as a soft probability
p1 = P(ŷ = 1 | x) per probe row, and a probe distribution D (the rows). For a subset A of the inputs (|A| = k), the
**cell statistics** are, for every cell x_A ∈ {0, 1}^k,

    N_x = number (or weight) of probe rows in the cell,     S_x = Σ_{rows in the cell} (2 p1 − 1).

They are sufficient for everything on the marginal table of A ∪ {ŷ}, which is
P(x_A, ŷ = 1) = (N_x + S_x) / (2N), P(x_A, ŷ = 0) = (N_x − S_x) / (2N). With m(x) = S_x / N_x (the cell mean of
2p1 − 1) and χ_B(x) = Π_{i∈B} s_i:

η view (moments)
    * mean output f(∅) = Σ_x S_x / N;
    * data moment m_D(B) = Σ_x S_x χ_B(x) / N = E_D[(2p1 − 1) χ_B], for B ⊆ A (the uncentred sample average; it
      depends on D and is biased by f(∅) when the cells are unbalanced);
    * uniform-cell coefficient f_cell(B) = 2^−k Σ_x m(x) χ_B(x), the Walsh coefficient of the cell means with every
      cell weighted equally. Under uniform probes it estimates the function's Walsh coefficient f_u(B) without the
      bias of the uncentred average (sky v2 A3). Undefined (NaN) if a cell is empty.

θ view (log-linear)
    * logit coefficient w(B) = 2^−k Σ_x atanh(m(x)) χ_B(x): the spin coefficient of the term B ∪ {ŷ} in the log-linear
      expansion of the marginal table (log-odds / 2). The top coefficient w(A) is 0 iff the local RI of A ∪ {ŷ} is 0.
      Undefined (NaN) if a cell is empty or saturated (|m| = 1).
    * local (marginal) RI of A ∪ {ŷ}: the plug-in RI of the marginal table, :func:`tycho.ri.refined.local_ri`.

Conditional (purified) RI, RI^cond (Enouen & Sugiyama 2026, App. C.1), needs the joint table of the whole
conditioning set, not only A. :func:`conditional_ri_plugin` builds it from the rows and solves the projection as a
weighted logistic regression (default). The iterative proportional fitting of
:func:`tycho.ri.refined.conditional_ri` is available as ``method="ipf"``, but it is inaccurate on near-saturated
tables. For binary variables the top-order term has 1 degree of freedom, so G² = 2 N RI^cond is asymptotically
χ²_1 under H0. That approximation needs non-sparse tables. Its calibration is experiment E8 of sky v2
(``tycho.calibration.correlated``).

Parity information. :func:`parity_information` is the mutual information between ŷ and the parity feature χ_A
under the cells' weights: the information that A's top Walsh feature carries. It is the η view's information value
on data probes. It is always defined, and with balanced χ_A it equals ``eta_information(f(A), f(∅))``.

Information scale of a coefficient. ``eta_information(f, f0)`` is the mutual information of a binary input with an
output whose mean is f0 + f·s: h((1 + f0)/2) − ½[h((1 + f0 + f)/2) + h((1 + f0 − f)/2)] nats. At f0 = 0 it equals
the single-mode value ln 2 − h((1 + |f|)/2) of :func:`tycho.ri.fourier.single_mode_ri`. With f0 ≠ 0 it accounts
for the output's base rate (sky v2 A3, item 4).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from tycho.ri.refined import conditional_ri, local_ri_batch

__all__ = [
    "CellStats",
    "cell_stats",
    "chi_matrix",
    "subsets_of",
    "eta_information",
    "conditional_ri_plugin",
    "gate",
    "parity_information",
]


def subsets_of(k: int) -> list[tuple[int, ...]]:
    """All subsets of ``range(k)`` (positions within A), the empty one first, then by size and lexicographically."""
    return [B for r in range(k + 1) for B in itertools.combinations(range(k), r)]


def chi_matrix(k: int) -> np.ndarray:
    """(2^k cells, 2^k subsets) matrix of χ_B(x): cells in C order over {0,1}^k (first position most significant),
    subsets in the order of :func:`subsets_of`."""
    cells = np.array(list(itertools.product((0, 1), repeat=k)), dtype=np.float64).reshape(-1, k) * 2 - 1
    return np.stack([np.prod(cells[:, list(B)], axis=1) if B else np.ones(len(cells)) for B in subsets_of(k)], axis=1)


@dataclass
class CellStats:
    """Cell counts and output sums of one input subset (see the module docstring).

    Attributes:
        inputs: input column indices of A, in the order that defines the cells (first = most significant bit).
        counts: (2^k,) number (or weight) of probe rows per cell.
        sums: (2^k,) Σ (2 p1 − 1) per cell.
        n: total number (or weight) of rows.
    """

    inputs: tuple[int, ...]
    counts: np.ndarray
    sums: np.ndarray
    n: float

    @property
    def k(self) -> int:
        return len(self.inputs)

    def table(self) -> np.ndarray:
        """The marginal table P(x_A, ŷ) of shape (2,)*k + (2,), ŷ on the last axis."""
        p0 = (self.counts - self.sums) / (2.0 * self.n)
        p1 = (self.counts + self.sums) / (2.0 * self.n)
        return np.clip(np.stack([p0, p1], axis=-1), 0.0, None).reshape((2,) * self.k + (2,))

    def mean_output(self) -> float:
        return float(self.sums.sum() / self.n)

    def cell_means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.sums / np.where(self.counts > 0, self.counts, 1.0), np.nan)

    def data_moments(self) -> dict[tuple[int, ...], float]:
        """m_D(B) for every B ⊆ A (positions within A; () is f(∅))."""
        v = chi_matrix(self.k).T @ self.sums / self.n
        return dict(zip(subsets_of(self.k), map(float, v)))

    def uniform_cell_walsh(self) -> dict[tuple[int, ...], float]:
        """f_cell(B) for every B ⊆ A; NaN everywhere if a cell is empty."""
        m = self.cell_means()
        if not np.all(np.isfinite(m)):
            return {B: math.nan for B in subsets_of(self.k)}
        v = chi_matrix(self.k).T @ m / 2 ** self.k
        return dict(zip(subsets_of(self.k), map(float, v)))

    def logit_walsh(self, *, tol: float = 1e-12) -> dict[tuple[int, ...], float]:
        """w(B) = Walsh coefficient of atanh(m) for every B ⊆ A; NaN everywhere if a cell is empty or saturated."""
        m = self.cell_means()
        if not np.all(np.isfinite(m)) or np.any(np.abs(m) >= 1 - tol):
            return {B: math.nan for B in subsets_of(self.k)}
        v = chi_matrix(self.k).T @ np.arctanh(m) / 2 ** self.k
        return dict(zip(subsets_of(self.k), map(float, v)))


def cell_stats(X: np.ndarray, p1: np.ndarray, subsets: Sequence[Sequence[int]], *,
               weights: np.ndarray | None = None) -> list[CellStats]:
    """Cell statistics of each input subset from probe rows.

    Args:
        X: (N, d) 0/1 input codes.
        p1: (N,) soft output probabilities P(ŷ = 1 | row).
        subsets: input column index tuples (the order within a tuple defines the cells).
        weights: optional (N,) non-negative row weights.
    """
    X = np.asarray(X)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != p1.shape[0]:
        raise ValueError(f"X must be (N, d) with N = len(p1); got {X.shape} and {p1.shape}")
    w = np.ones(len(p1)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    y = (2.0 * p1 - 1.0) * w
    out = []
    for A in subsets:
        A = tuple(int(a) for a in A)
        k = len(A)
        codes = (X[:, list(A)].astype(np.int64) * (1 << np.arange(k)[::-1])).sum(axis=1) if k else np.zeros(len(p1),
                                                                                                          np.int64)
        out.append(CellStats(inputs=A, counts=np.bincount(codes, weights=w, minlength=1 << k),
                             sums=np.bincount(codes, weights=y, minlength=1 << k), n=float(w.sum())))
    return out


def local_ri_of(stats: Sequence[CellStats]) -> np.ndarray:
    """Plug-in local RI of A ∪ {ŷ} for each element (batched by k)."""
    out = np.zeros(len(stats))
    by_k: dict[int, list[int]] = {}
    for i, cs in enumerate(stats):
        by_k.setdefault(cs.k, []).append(i)
    for k, idx in by_k.items():
        if k == 0:
            continue
        tabs = np.stack([np.clip(stats[i].table(), 1e-300, None) for i in idx])
        out[idx] = local_ri_batch(tabs)
    return out


__all__.append("local_ri_of")


def _hb(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(q > 0, -q * np.log(np.where(q > 0, q, 1.0)), 0.0)
        b = np.where(q < 1, -(1 - q) * np.log(np.where(q < 1, 1 - q, 1.0)), 0.0)
    return a + b


def eta_information(f: np.ndarray | float, f0: np.ndarray | float = 0.0) -> np.ndarray:
    """MI (nats) of a binary input with an output of mean f0 + f·s (see the module docstring); NaN if
    |f0| + |f| > 1 (not a valid conditional)."""
    f = np.abs(np.asarray(f, dtype=np.float64))
    f0 = np.asarray(f0, dtype=np.float64)
    valid = np.abs(f0) + f <= 1 + 1e-12
    v = _hb((1 + f0) / 2) - 0.5 * (_hb((1 + f0 + f) / 2) + _hb((1 + f0 - f) / 2))
    return np.where(valid, np.maximum(v, 0.0), np.nan)


def _weighted_logistic_loss(F: np.ndarray, w: np.ndarray, t: np.ndarray, *, tol: float = 1e-12,
                            max_iter: int = 100) -> float:
    """min over β of −Σ_x w_x [t_x log σ(F β) + (1 − t_x) log(1 − σ(F β))] by damped Newton (soft targets t)."""
    beta = np.zeros(F.shape[1])

    def loss(b):
        z = F @ b
        return float(np.sum(w * (np.logaddexp(0.0, z) - t * z)))
    cur = loss(beta)
    for _ in range(max_iter):
        z = F @ beta
        sig = 0.5 * (1.0 + np.tanh(0.5 * z))
        g = F.T @ (w * (sig - t))
        if np.max(np.abs(g)) < tol:
            break
        H = F.T @ (F * (w * sig * (1 - sig))[:, None]) + 1e-12 * np.eye(F.shape[1])
        step = np.linalg.solve(H, g)
        a = 1.0
        while a > 1e-8:
            new = loss(beta - a * step)
            if new <= cur + 1e-15:
                break
            a *= 0.5
        beta = beta - a * step
        if cur - new < 1e-16 and a == 1.0:
            cur = new
            break
        cur = new
    return cur


def parity_information(cs: CellStats, *, uniform_cells: bool = False) -> float:
    """MI (nats) between ŷ and χ_A, with the cells weighted by their counts (data probes) or, with
    ``uniform_cells``, equally (the uniform-cell analogue of :meth:`CellStats.uniform_cell_walsh`). NaN if a cell
    is empty in the uniform-cell case."""
    chiA = chi_matrix(cs.k)[:, -1] if cs.k else np.ones(1)
    if uniform_cells:
        m = cs.cell_means()
        if not np.all(np.isfinite(m)):
            return math.nan
        w = np.full(len(m), 1.0 / len(m))
        y = w * m
    else:
        w = cs.counts / cs.n
        y = cs.sums / cs.n
    tab = np.zeros((2, 2))
    for sign, row in ((-1, 0), (1, 1)):
        sel = chiA == sign
        tab[row, 0] = (w[sel].sum() - y[sel].sum()) / 2
        tab[row, 1] = (w[sel].sum() + y[sel].sum()) / 2
    tab = np.clip(tab, 0.0, None)
    tab /= tab.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        prod = tab.sum(axis=1, keepdims=True) * tab.sum(axis=0, keepdims=True)
        terms = np.where(tab > 0, tab * np.log(tab / prod), 0.0)
    return float(max(terms.sum(), 0.0))


def conditional_ri_plugin(X: np.ndarray, p1: np.ndarray, A: Sequence[int], cond: Sequence[int], *,
                          method: str = "logistic", tol: float = 1e-10, weights: np.ndarray | None = None) -> float:
    """Plug-in conditional RI of A ∪ {ŷ} given the inputs ``cond`` (the table holds U = cond ∪ A and ŷ).

    The table is built from the rows with the soft output. ``method="ipf"`` projects the full table with
    :func:`tycho.ri.refined.conditional_ri`; ``tol`` applies only to it. ``method="logistic"`` (default; faster,
    and exact up to the Newton tolerance, including near saturation, where IPF stops early) uses the equivalent
    form for a binary output: in the model without any term ⊇ A ∪ {ŷ} every
    input-only term is free, so the input margin is fitted exactly and the projection is a weighted logistic
    regression of ŷ on the features χ_B, B ⊆ U, B ⊉ A, with cell weights N_x/N and soft targets (the cell means
    of p1). RI^cond is the drop in weighted cross-entropy when χ_A is added. Cost grows as 2^|U|: small sets only.
    With ``weights`` (e.g. a pmf over all patterns, one row each) the value is the exact population RI^cond.
    """
    U = sorted(set(int(c) for c in cond) | set(int(a) for a in A))
    cs = cell_stats(X, p1, [tuple(U)], weights=weights)[0]
    if method == "ipf":
        P = np.clip(cs.table(), 0.0, None)
        axes = [U.index(int(a)) for a in A] + [len(U)]
        return float(conditional_ri(P, axes, tol=tol))
    if method != "logistic":
        raise ValueError(f"unknown method {method!r}")
    k = len(U)
    pos = {U.index(int(a)) for a in A}
    keep = cs.counts > 0
    w = cs.counts[keep] / cs.n
    t = np.clip((cs.counts[keep] + cs.sums[keep]) / (2.0 * cs.counts[keep]), 0.0, 1.0)
    H = chi_matrix(k)[keep]
    subs = subsets_of(k)
    lo = [j for j, B in enumerate(subs) if not pos <= set(B)]
    hi = lo + [subs.index(tuple(sorted(pos)))]
    return max(0.0, _weighted_logistic_loss(H[:, lo], w, t) - _weighted_logistic_loss(H[:, hi], w, t))


def gate(info_plugin: np.ndarray, n: float, min_effect: float, *, df: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(corrected value, χ² p-value) for plug-in information values: corrected = max(0, v − df/(2n)),
    p = χ²_df.sf(2 n v). Used for RI (marginal and conditional) and, for an equal footing, for η information."""
    v = np.asarray(info_plugin, dtype=np.float64)
    corr = np.maximum(0.0, v - df / (2.0 * n))
    p = chi2.sf(2.0 * n * np.nan_to_num(v, nan=0.0), df)
    p = np.where(np.isfinite(v), p, 1.0)
    return corr, p
