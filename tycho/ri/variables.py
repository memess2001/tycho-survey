"""The discrete variables an RI analysis runs on (contract type shared by tasks, estimators and the survey)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class VariableSet:
    """N joint observations of d discrete variables.

    Attributes:
        X: (N, d) integer codes; column i takes values in ``[0, cards[i])``. For a *soft* column the codes are
            ignored by the estimators (store e.g. the argmax there).
        cards: number of states of each variable.
        names: unique variable names (used for ground truth matching and for the catalog).
        soft: optional ``{j: (N, cards[j]) row-stochastic array}``. At most one soft column. Sample n then
            contributes probability mass ``soft[j][n, c]`` to state c of variable j (e.g. a model's predictive
            distribution, which avoids sampling noise in the output variable).
        must_include: candidate subsets must contain all of these column indices (e.g. the output variable).
        ground_truth: list of dicts ``{"subset": [names...], "order": k, ...}`` describing interactions that are
            known to exist (positive controls). Entries whose names are not all present are ignored.
        meta: free-form metadata (task name, step, binarization method, ...).
    """

    X: np.ndarray
    cards: list[int]
    names: list[str]
    soft: dict[int, np.ndarray] | None = None
    must_include: list[int] | None = None
    ground_truth: list[dict] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, d), got shape {X.shape}")
        if X.dtype == np.bool_:
            X = X.astype(np.int64)
        elif not np.issubdtype(X.dtype, np.integer):
            Xf = X.astype(np.float64)
            if X.size and (not np.all(np.isfinite(Xf)) or not np.all(Xf == np.round(Xf))):
                raise ValueError("X must contain integer codes")
        self.X = X.astype(np.int64, copy=False)
        self.cards = [int(c) for c in self.cards]
        self.names = [str(n) for n in self.names]
        n, d = self.X.shape
        if len(self.cards) != d or len(self.names) != d:
            raise ValueError(f"cards ({len(self.cards)}) and names ({len(self.names)}) must have length d={d}")
        if len(set(self.names)) != d:
            raise ValueError("variable names must be unique")
        if any(c < 1 for c in self.cards):
            raise ValueError("cards must be >= 1")
        soft = dict(self.soft or {})
        if len(soft) > 1:
            raise ValueError("at most one soft column is supported")
        for j, pj in list(soft.items()):
            j = int(j)
            if not 0 <= j < d:
                raise ValueError(f"soft column index {j} out of range")
            pj = np.asarray(pj, dtype=np.float64)
            if pj.shape != (n, self.cards[j]):
                raise ValueError(f"soft[{j}] must have shape {(n, self.cards[j])}, got {pj.shape}")
            if np.any(pj < -1e-12) or not np.allclose(pj.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"soft[{j}] rows must be probability vectors")
            soft = {j: np.clip(pj, 0.0, None)}
        self.soft = soft or None
        hard_cols = [i for i in range(d) if not (self.soft and i in self.soft)]
        if n and hard_cols:
            Xh = self.X[:, hard_cols]
            lo = Xh.min(axis=0)
            hi = Xh.max(axis=0)
            cards_h = np.asarray([self.cards[i] for i in hard_cols])
            if np.any(lo < 0) or np.any(hi >= cards_h):
                raise ValueError("codes out of range [0, card)")
        if self.must_include is not None:
            self.must_include = sorted({int(i) for i in self.must_include})
            if any(not 0 <= i < d for i in self.must_include):
                raise ValueError("must_include index out of range")

    @property
    def N(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    def index(self, name: str) -> int:
        return self.names.index(name)

    def subset_names(self, S: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.names[i] for i in S)

    def subset_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(sorted(self.index(n) for n in names))

    def ground_truth_subsets(self) -> list[tuple[int, ...]]:
        """Index tuples of ground-truth subsets whose variables are all present."""
        out = []
        for gt in self.ground_truth or []:
            names = gt.get("subset", [])
            if all(n in self.names for n in names):
                out.append(self.subset_indices(names))
        return out

    def select(self, cols: Sequence[int]) -> "VariableSet":
        """Return a VariableSet restricted to ``cols`` (in the given order)."""
        cols = [int(c) for c in cols]
        remap = {old: new for new, old in enumerate(cols)}
        soft = None
        if self.soft:
            soft = {remap[j]: p for j, p in self.soft.items() if j in remap} or None
        must = None
        if self.must_include is not None:
            must = [remap[i] for i in self.must_include if i in remap]
        names = [self.names[c] for c in cols]
        gt = None
        if self.ground_truth is not None:
            gt = [g for g in self.ground_truth if all(n in names for n in g.get("subset", []))]
        return VariableSet(X=self.X[:, cols], cards=[self.cards[c] for c in cols], names=names, soft=soft,
                           must_include=must, ground_truth=gt, meta=dict(self.meta))
