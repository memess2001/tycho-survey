"""Time-resolved refined information of many variable subsets (the "spectral light curve" of one object)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


@dataclass
class RICurve:
    """RI of ``S`` subsets at ``T`` checkpoints.

    Attributes:
        steps: (T,) training steps of the checkpoints.
        names: variable names (indices in ``subsets`` refer to this list).
        subsets: S sorted index tuples.
        ri: (S, T) bias-corrected local RI in nats (``RIEstimate.ri_corrected``).
        ri_plugin: (S, T) plug-in estimate.
        p: (S, T) p-values; q: (S, T) Benjamini-Hochberg q-values computed within each checkpoint.
        n_eff: (T,) effective sample sizes.
        ground_truth: optional list of ``{"subset": [names...], "order": k, ...}``.
        meta: alpha, min_effect, max_order, null, object_id, kind, task, ...
    """

    steps: np.ndarray
    names: list[str]
    subsets: list[tuple[int, ...]]
    ri: np.ndarray
    ri_plugin: np.ndarray
    p: np.ndarray
    q: np.ndarray
    n_eff: np.ndarray
    ground_truth: list[dict] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.steps = np.asarray(self.steps, dtype=np.int64).reshape(-1)
        self.names = [str(n) for n in self.names]
        self.subsets = [tuple(int(i) for i in s) for s in self.subsets]
        S, T = len(self.subsets), len(self.steps)
        for name in ("ri", "ri_plugin", "p", "q"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.size == 0:
                arr = arr.reshape(S, T)
            if arr.shape != (S, T):
                raise ValueError(f"{name} must have shape {(S, T)}, got {arr.shape}")
            setattr(self, name, arr)
        self.n_eff = np.asarray(self.n_eff, dtype=np.float64).reshape(-1)
        if self.n_eff.shape != (T,):
            raise ValueError(f"n_eff must have shape {(T,)}, got {self.n_eff.shape}")

    # ------------------------------------------------------------------ basic accessors
    @property
    def S(self) -> int:
        return len(self.subsets)

    @property
    def T(self) -> int:
        return len(self.steps)

    @property
    def orders(self) -> np.ndarray:
        return np.asarray([len(s) for s in self.subsets], dtype=np.int64)

    def subset_names(self, i: int) -> tuple[str, ...]:
        return tuple(self.names[j] for j in self.subsets[i])

    def index_of(self, subset: Sequence[int] | Sequence[str]) -> int | None:
        """Row index of a subset given as indices or names (order-insensitive); None if absent."""
        items = list(subset)
        if items and isinstance(items[0], str):
            try:
                idx = tuple(sorted(self.names.index(n) for n in items))
            except ValueError:
                return None
        else:
            idx = tuple(sorted(int(i) for i in items))
        try:
            return self.subsets.index(idx)
        except ValueError:
            return None

    def ground_truth_indices(self) -> list[int]:
        out = []
        for gt in self.ground_truth or []:
            i = self.index_of(gt.get("subset", []))
            if i is not None:
                out.append(i)
        return out

    def detected(self, alpha: float | None = None, min_effect: float | None = None) -> np.ndarray:
        alpha = self.meta.get("alpha", 0.01) if alpha is None else alpha
        min_effect = self.meta.get("min_effect", 1e-3) if min_effect is None else min_effect
        with np.errstate(invalid="ignore"):
            return (self.q <= alpha) & (self.ri >= min_effect)

    # ------------------------------------------------------------------ construction
    @classmethod
    def empty(cls, names: Sequence[str], subsets: Sequence[Sequence[int]], *, ground_truth=None,
              meta: dict | None = None) -> "RICurve":
        S = len(subsets)
        z = np.zeros((S, 0))
        return cls(steps=np.zeros(0, dtype=np.int64), names=list(names), subsets=[tuple(s) for s in subsets],
                   ri=z, ri_plugin=z.copy(), p=z.copy(), q=z.copy(), n_eff=np.zeros(0),
                   ground_truth=ground_truth, meta=dict(meta or {}))

    def append(self, step: int, scan: Any) -> "RICurve":
        """Return a new curve with one more checkpoint taken from a ``ScanResult``-like object.

        ``scan`` must expose ``subsets`` (identical to ``self.subsets``), ``estimates`` (objects with
        ``ri_corrected``, ``ri_plugin``, ``p_value``, ``n_eff``) and ``q_values``.
        """
        subs = [tuple(int(i) for i in s) for s in scan.subsets]
        if subs != self.subsets:
            raise ValueError("scan subsets differ from curve subsets")
        ri = np.asarray([e.ri_corrected for e in scan.estimates], dtype=np.float64)[:, None]
        rp = np.asarray([e.ri_plugin for e in scan.estimates], dtype=np.float64)[:, None]
        p = np.asarray([e.p_value for e in scan.estimates], dtype=np.float64)[:, None]
        q = np.asarray(scan.q_values, dtype=np.float64).reshape(-1, 1)
        n_eff = float(scan.estimates[0].n_eff) if scan.estimates else float("nan")
        return RICurve(steps=np.append(self.steps, int(step)), names=self.names, subsets=self.subsets,
                       ri=np.hstack([self.ri, ri]), ri_plugin=np.hstack([self.ri_plugin, rp]),
                       p=np.hstack([self.p, p]), q=np.hstack([self.q, q]), n_eff=np.append(self.n_eff, n_eff),
                       ground_truth=self.ground_truth, meta=dict(self.meta))

    @classmethod
    def from_scans(cls, steps: Sequence[int], scans: Sequence[Any], *, names: Sequence[str] | None = None,
                   ground_truth=None, meta: dict | None = None) -> "RICurve":
        if not scans:
            raise ValueError("need at least one scan")
        names = list(names if names is not None else scans[0].names)
        curve = cls.empty(names, scans[0].subsets, ground_truth=ground_truth, meta=meta)
        for st, sc in zip(steps, scans):
            curve = curve.append(int(st), sc)
        return curve

    def prune(self, keep_top: int = 200, alpha: float | None = None, min_effect: float | None = None) -> "RICurve":
        """Keep subsets that were ever detected, the ``keep_top`` largest by max RI, and ground-truth subsets."""
        if self.S == 0:
            return self
        keep = set(np.nonzero(self.detected(alpha, min_effect).any(axis=1))[0].tolist()) if self.T else set()
        if self.T:
            peak = np.nan_to_num(self.ri, nan=-np.inf).max(axis=1)
            keep |= set(np.argsort(-peak, kind="stable")[: max(0, keep_top)].tolist())
        else:
            keep |= set(range(min(self.S, keep_top)))
        keep |= set(self.ground_truth_indices())
        idx = sorted(keep)
        meta = dict(self.meta)
        meta["n_candidates"] = int(meta.get("n_candidates", self.S))
        return RICurve(steps=self.steps, names=self.names, subsets=[self.subsets[i] for i in idx],
                       ri=self.ri[idx], ri_plugin=self.ri_plugin[idx], p=self.p[idx], q=self.q[idx],
                       n_eff=self.n_eff, ground_truth=self.ground_truth, meta=meta)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lens = np.asarray([len(s) for s in self.subsets], dtype=np.int64)
        flat = np.asarray([i for s in self.subsets for i in s], dtype=np.int64)
        header = json.dumps({"names": self.names, "ground_truth": self.ground_truth, "meta": self.meta},
                            default=_json_default)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, steps=self.steps, ri=self.ri, ri_plugin=self.ri_plugin, p=self.p, q=self.q,
                            n_eff=self.n_eff, subsets_len=lens, subsets_flat=flat, header=np.array(header))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "RICurve":
        with np.load(Path(path), allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            lens = z["subsets_len"].tolist()
            flat = z["subsets_flat"].tolist()
            subsets, pos = [], 0
            for ln in lens:
                subsets.append(tuple(flat[pos:pos + ln]))
                pos += ln
            return cls(steps=z["steps"], names=header["names"], subsets=subsets, ri=z["ri"],
                       ri_plugin=z["ri_plugin"], p=z["p"], q=z["q"], n_eff=z["n_eff"],
                       ground_truth=header.get("ground_truth"), meta=header.get("meta") or {})
