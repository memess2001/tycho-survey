"""Discretisation of continuous observables (activations, logits) into binary RI variables.

RI is defined on categorical variables, so continuous quantities such as hidden-unit activations must be
discretised before spectroscopy. :func:`binarize` thresholds each column, and :func:`select_informative` picks a
manageable number of columns (the number of candidate subsets grows like ``C(k, order)``).

Discretisation is a systematic error of the instrument. Thresholding a Gaussian latent with only pairwise
correlation induces higher-order log-linear interactions in the bits (calibration experiment E6 quantifies this).
Ties matter too. With ``"median"`` on a column whose median value is repeated (e.g. a ReLU unit that is exactly 0
on more than half of the inputs), ``A > median`` is strictly unbalanced; ``"zero"`` is the natural choice for ReLU
units ("active or not").
"""
from __future__ import annotations

import numpy as np

__all__ = ["BINARIZE_METHODS", "SELECT_METHODS", "binarize", "select_informative"]

#: Thresholding rules of :func:`binarize`.
BINARIZE_METHODS = ("zero", "median", "mean", "quantile")
#: Column-selection rules of :func:`select_informative`.
SELECT_METHODS = ("entropy", "random")


def binarize(A: np.ndarray, method: str = "median", q: float = 0.5) -> np.ndarray:
    """Threshold each column: 1 where the value is strictly above the column's threshold, else 0.

    Methods: ``"zero"`` (``A > 0``), ``"median"`` (``A >`` per-column median), ``"mean"`` (``A >`` per-column mean),
    ``"quantile"`` (``A >`` per-column ``q``-quantile, linear interpolation).

    Args:
        A: (N, m) array (a 1-D array is treated as one column and returned 1-D).
        method: one of :data:`BINARIZE_METHODS`.
        q: quantile level in [0, 1] for ``"quantile"``.

    Returns:
        int8 array of the shape of ``A``.

    Raises:
        ValueError: on an unknown method, a non-finite value, ``A`` with more than two axes, or ``q`` outside [0, 1].
    """
    if method not in BINARIZE_METHODS:
        raise ValueError(f"method must be one of {BINARIZE_METHODS}, got {method!r}")
    A = np.asarray(A, dtype=np.float64)
    if A.ndim not in (1, 2):
        raise ValueError(f"A must be 1-D or 2-D, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    if method == "quantile" and not 0.0 <= q <= 1.0:
        raise ValueError(f"q must lie in [0, 1], got {q}")
    if A.shape[0] == 0 or method == "zero":
        return (A > 0.0).astype(np.int8)
    if method == "median":
        threshold = np.median(A, axis=0)
    elif method == "mean":
        threshold = A.mean(axis=0)
    else:
        threshold = np.quantile(A, q, axis=0)
    return (A > threshold).astype(np.int8)


def _column_entropy(col: np.ndarray) -> float:
    """Plug-in Shannon entropy (nats) of the empirical distribution of one discrete column (0 if empty)."""
    if col.size == 0:
        return 0.0
    _, counts = np.unique(col, return_counts=True)
    p = counts / col.size
    return float(-np.sum(p * np.log(p)))


def select_informative(B: np.ndarray, k: int, *, method: str = "entropy", rng: np.random.Generator | None = None,
                       exclude_constant: bool = True) -> np.ndarray:
    """Choose up to ``k`` columns of a discrete matrix ``B`` (N, m).

    ``"entropy"``: the ``k`` columns of highest empirical entropy (ties broken by column index). ``"random"``: ``k``
    columns uniformly without replacement (``rng``, default ``np.random.default_rng(0)``). With
    ``exclude_constant`` (default), columns taking a single value are never chosen, so fewer than ``k`` indices are
    returned if fewer non-constant columns exist.

    Returns:
        Sorted int64 array of column indices.
    """
    if method not in SELECT_METHODS:
        raise ValueError(f"method must be one of {SELECT_METHODS}, got {method!r}")
    B = np.asarray(B)
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D (N, m), got shape {B.shape}")
    if k < 0:
        raise ValueError("k must be >= 0")
    H = np.asarray([_column_entropy(B[:, j]) for j in range(B.shape[1])])
    cand = np.arange(B.shape[1])
    if exclude_constant:
        cand = cand[H > 0]
    k = min(int(k), cand.size)
    if method == "entropy":
        chosen = cand[np.lexsort((cand, -H[cand]))[:k]]
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        chosen = rng.choice(cand, size=k, replace=False)
    return np.sort(chosen).astype(np.int64)
