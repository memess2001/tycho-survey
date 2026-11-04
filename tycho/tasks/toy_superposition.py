"""Toy model of superposition (Elhage et al. 2022): sparse features squeezed through a narrow ReLU autoencoder.

Problem
-------
``n_features`` independent features, ``x_i ~ U(0, 1)`` with probability ``feature_prob`` and 0 otherwise. Model
:class:`~tycho.models.ToyReLU` ``x̂ = ReLU(Wᵀ W x + b)`` with ``W ∈ R^{m_hidden × n_features}``. Loss: importance-
weighted squared error ``mean_batch Σ_i I_i (x_i − x̂_i)²`` with ``I_i = importance_decay^i`` (feature 0 is the
most important). As features get sparser, more of them are represented, in superposition (non-orthogonal
directions arranged in polytopes). Accuracy is undefined. "train" and "test" are fixed samples of ``eval_size``
inputs from two streams of ``data_seed``.

Extra light-curve metrics (W_i = column i of W, Ŵ_i = W_i / ‖W_i‖)
-------------------------------------------------------------------
* ``n_represented``: number of features with ``‖W_i‖ > REPRESENTED_NORM``;
* ``superposition``: mean over represented i of ``Σ_{j≠i} (Ŵ_i · W_j)²`` (interference on feature i);
* ``dim_per_feature``: mean over represented i of the feature dimensionality
  ``D_i = ‖W_i‖⁴ / Σ_j (W_i · W_j)²`` (1 for an orthogonal feature, 1/k when k features share one direction).
The last two are NaN when no feature is represented.

RI variables
------------
For the ``RI_FEATURES`` most important features: ``a{i}`` (feature present, ``x_i > 0``) and ``r{i}`` (feature
reconstructed, ``x̂_i > RECON_THRESHOLD``), on ``n`` fresh inputs from the data distribution. No ground truth; no
required variable. Interference shows up as interactions between the presence of one feature and the
reconstruction of another.

eNTK output
-----------
The per-sample negative weighted squared error ``−Σ_i I_i (x_i − x̂_i)²`` (there are no labels, so kernel-target
alignment is NaN).
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
from torch import Tensor, nn

from tycho.models.superposition import ToyReLU
from tycho.ri.variables import VariableSet
from tycho.tasks.base import Task, model_device, torch_generator

__all__ = ["RECON_THRESHOLD", "REPRESENTED_NORM", "RI_FEATURES", "ToySuperposition", "superposition_metrics"]

#: A feature counts as represented when its column norm exceeds this value (Elhage et al. use ‖W_i‖ ≈ 1 vs ≈ 0).
REPRESENTED_NORM = 0.5
#: A feature counts as reconstructed (RI variable ``r{i}``) when ``x̂_i`` exceeds this value.
RECON_THRESHOLD = 0.1
#: Number of most important features observed by RI spectroscopy.
RI_FEATURES = 5


def superposition_metrics(W: np.ndarray) -> dict[str, float]:
    """``n_represented``, ``superposition`` and ``dim_per_feature`` of a weight matrix ``W`` (m × n)."""
    W = np.asarray(W, dtype=np.float64)
    norms = np.linalg.norm(W, axis=0)
    rep = np.flatnonzero(norms > REPRESENTED_NORM)
    out = {"n_represented": float(rep.size), "superposition": float("nan"), "dim_per_feature": float("nan")}
    if rep.size:
        G = W.T @ W  # G[i, j] = W_i · W_j
        sq = np.square(G[rep])  # (r, n)
        interference = (sq.sum(axis=1) - np.square(norms[rep] ** 2)) / norms[rep] ** 2
        out["superposition"] = float(interference.mean())
        out["dim_per_feature"] = float((norms[rep] ** 4 / sq.sum(axis=1)).mean())
    return out


class ToySuperposition(Task):
    """Sparse features through a ReLU bottleneck (see the module docstring)."""

    name = "toy_superposition"

    @classmethod
    def defaults(cls) -> dict:
        return {"n_features": 20, "m_hidden": 5, "feature_prob": 0.1, "importance_decay": 0.9, "data_seed": 0,
                "eval_size": 4096}

    def __init__(self, cfg: dict | None, seed: int) -> None:
        super().__init__(cfg, seed)
        c = self.cfg
        self.n_features, self.m_hidden = int(c["n_features"]), int(c["m_hidden"])
        self.feature_prob, self.decay = float(c["feature_prob"]), float(c["importance_decay"])
        self.data_seed, self.eval_size = int(c["data_seed"]), int(c["eval_size"])
        if self.n_features < 1 or self.m_hidden < 1:
            raise ValueError("n_features and m_hidden must be >= 1")
        if not 0.0 < self.feature_prob <= 1.0:
            raise ValueError(f"feature_prob must lie in (0, 1], got {self.feature_prob}")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"importance_decay must lie in (0, 1], got {self.decay}")
        if self.eval_size < 1:
            raise ValueError(f"eval_size must be >= 1, got {self.eval_size}")
        self.importance = self.decay ** torch.arange(self.n_features, dtype=torch.float32)
        self.n_ri = min(RI_FEATURES, self.n_features)

    # ------------------------------------------------------------------------------------------ data
    def _sample(self, m: int, gen: torch.Generator) -> Tensor:
        value = torch.rand(m, self.n_features, generator=gen)
        present = torch.rand(m, self.n_features, generator=gen) < self.feature_prob
        return value * present

    def sample_batch(self, batch_size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        x = self._sample(batch_size, gen)
        return x, x

    @cached_property
    def _splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        out = {}
        for name in ("train", "test"):
            x = self._sample(self.eval_size, torch_generator(self.data_seed, f"toy_superposition/{name}"))
            out[name] = (x, x)
        return out

    def eval_splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        return dict(self._splits)

    # ------------------------------------------------------------------------------------------ model / loss
    def build_model(self) -> nn.Module:
        torch.manual_seed(self.seed)
        return ToyReLU(self.n_features, self.m_hidden)

    def _weighted_sq_error(self, out: Tensor, y: Tensor) -> Tensor:
        return ((y - out).square() * self.importance.to(out.device)).sum(dim=-1)

    def loss(self, out: Tensor, y: Tensor) -> Tensor:
        return self._weighted_sq_error(out, y).mean()

    def ntk_output(self, out: Tensor, y: Tensor) -> Tensor:
        return -self._weighted_sq_error(out, y)

    # ------------------------------------------------------------------------------------------ observation
    def default_ri_order(self) -> int:
        return 3

    def ri_variables(self, model: nn.Module, n: int, gen: torch.Generator, *,
                     frozen: dict | None = None) -> VariableSet:
        x = self._sample(int(n), gen)
        with torch.no_grad():
            x_hat = model(x.to(model_device(model))).cpu()
        k = self.n_ri
        present = (x[:, :k] > 0).numpy().astype(np.int64)
        recon = (x_hat[:, :k] > RECON_THRESHOLD).numpy().astype(np.int64)
        names = [f"a{i}" for i in range(k)] + [f"r{i}" for i in range(k)]
        return VariableSet(X=np.concatenate([present, recon], axis=1), cards=[2] * (2 * k), names=names,
                           must_include=None, ground_truth=None,
                           meta={"task": self.name, "probe": "data", "recon_threshold": RECON_THRESHOLD})

    def extra_metrics(self, model: nn.Module) -> dict[str, float]:
        return superposition_metrics(model.W.detach().to("cpu", torch.float64).numpy())
