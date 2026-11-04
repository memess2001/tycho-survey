"""Sparse parity: the positive control with a pure (k+1)-order synergy and "hidden progress".

Problem
-------
Inputs ``x ∈ {−1, +1}^n`` are uniform. The label depends on a hidden support ``S*`` of ``k`` coordinates,
``S* = sorted(np.random.default_rng(data_seed).choice(n, k, replace=False))``, through their parity:
``y = 1`` iff ``∏_{i∈S*} x_i = −1``. An MLP with two logits is trained with cross-entropy, either online on fresh
samples (``n_train = None``) or on a fixed training set of ``n_train`` samples (with-replacement minibatches).
SGD learns k-sparse parities after a long loss plateau during which the weights slowly concentrate on ``S*``
(Barak et al. 2022, "Hidden progress in deep learning"); ``x_support_frac`` tracks that hidden progress.

RI variables (the spectroscopy target)
--------------------------------------
Probe inputs are *fresh* uniform bits, so the ``n`` input variables ``x0 … x{n−1}`` (code 1 iff ``x_i = +1``) are
mutually independent, and ``yhat`` is the model's prediction: by default a *soft* column holding the predictive
distribution (no sampling noise in the output variable), else the argmax. Scans require ``yhat`` in every subset.
For the target function the (k+1)-subset ``{x_i : i ∈ S*} ∪ {yhat}`` carries exactly ``log 2`` nats of local RI
and every other subset containing ``yhat`` carries 0 (parity is a pure top-order interaction of uniform bits), so a
model that has learned the task shows a single spectral line of order k+1 at the ground-truth subset.

Extra light-curve metric
------------------------
``support_frac`` = ``‖W₁[:, S*]‖²_F / ‖W₁‖²_F`` for the first-layer weight ``W₁``: the share of input-weight mass on
the relevant coordinates (``≈ k/n`` at initialisation, → 1 when the network has localised the support).
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tycho.models.mlp import MLP
from tycho.ri.variables import VariableSet
from tycho.tasks.base import Task, logit_margin, model_device, softmax_np, torch_generator

__all__ = ["SparseParity"]


class SparseParity(Task):
    """k-sparse parity of n uniform ±1 bits, learned by an MLP (see the module docstring)."""

    name = "sparse_parity"

    @classmethod
    def defaults(cls) -> dict:
        return {"n": 20, "k": 3, "width": 256, "depth": 1, "act": "relu", "n_train": None, "test_size": 2048,
                "data_seed": 0, "ri_soft": True}

    def __init__(self, cfg: dict | None, seed: int) -> None:
        super().__init__(cfg, seed)
        c = self.cfg
        self.n, self.k = int(c["n"]), int(c["k"])
        self.width, self.depth, self.act = int(c["width"]), int(c["depth"]), str(c["act"])
        self.n_train = None if c["n_train"] is None else int(c["n_train"])
        self.test_size, self.data_seed, self.ri_soft = int(c["test_size"]), int(c["data_seed"]), bool(c["ri_soft"])
        if not 1 <= self.k <= self.n:
            raise ValueError(f"need 1 <= k <= n, got n={self.n}, k={self.k}")
        if self.n_train is not None and self.n_train < 1:
            raise ValueError(f"n_train must be None or >= 1, got {self.n_train}")
        if self.test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {self.test_size}")
        self.support: list[int] = sorted(int(i) for i in
                                         np.random.default_rng(self.data_seed).choice(self.n, self.k, replace=False))
        self._support_idx = torch.tensor(self.support, dtype=torch.long)

    # ------------------------------------------------------------------------------------------ data
    def _sample(self, m: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """``m`` uniform ±1 inputs (float32) and their parity labels (int64)."""
        x = torch.randint(0, 2, (m, self.n), generator=gen, dtype=torch.float32).mul_(2.0).sub_(1.0)
        y = x.index_select(1, self._support_idx).prod(dim=1).lt(0).long()
        return x, y

    @cached_property
    def _train_set(self) -> tuple[Tensor, Tensor] | None:
        if self.n_train is None:
            return None
        return self._sample(self.n_train, torch_generator(self.data_seed, "sparse_parity/train"))

    def sample_batch(self, batch_size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        if self._train_set is None:
            return self._sample(batch_size, gen)
        x, y = self._train_set
        idx = torch.randint(0, x.shape[0], (batch_size,), generator=gen)
        return x[idx], y[idx]

    @cached_property
    def _splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        test = self._sample(self.test_size, torch_generator(self.data_seed, "sparse_parity/test"))
        if self._train_set is not None:
            train = self._train_set
        else:  # online training: a fixed sample of the training distribution
            train = self._sample(self.test_size, torch_generator(self.data_seed, "sparse_parity/train-eval"))
        return {"train": train, "test": test}

    def eval_splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        return dict(self._splits)

    # ------------------------------------------------------------------------------------------ model / loss
    def build_model(self) -> nn.Module:
        torch.manual_seed(self.seed)
        return MLP(self.n, self.width, 2, depth=self.depth, act=self.act)

    def loss(self, out: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(out, y)

    def accuracy(self, out: Tensor, y: Tensor) -> float | None:
        return float((out.argmax(dim=-1) == y).float().mean().item())

    def chance_accuracy(self) -> float | None:
        return 0.5

    def ntk_output(self, out: Tensor, y: Tensor) -> Tensor:
        return logit_margin(out, y)

    # ------------------------------------------------------------------------------------------ observation
    def ground_truth(self) -> list[dict] | None:
        return [{"subset": [f"x{i}" for i in self.support] + ["yhat"], "order": self.k + 1}]

    def default_ri_order(self) -> int:
        return self.k + 1

    def ri_variables(self, model: nn.Module, n: int, gen: torch.Generator, *,
                     frozen: dict | None = None) -> VariableSet:
        x, _ = self._sample(int(n), gen)
        with torch.no_grad():
            logits = model(x.to(model_device(model)))
        probs = softmax_np(logits)
        bits = (x.numpy() > 0).astype(np.int64)
        X = np.concatenate([bits, probs.argmax(axis=1)[:, None]], axis=1)
        names = [f"x{i}" for i in range(self.n)] + ["yhat"]
        return VariableSet(X=X, cards=[2] * (self.n + 1), names=names,
                           soft={self.n: probs} if self.ri_soft else None, must_include=[self.n],
                           ground_truth=self.ground_truth(),
                           meta={"task": self.name, "probe": "uniform", "soft_output": self.ri_soft})

    def extra_metrics(self, model: nn.Module) -> dict[str, float]:
        w = model.first_layer().weight.detach().to("cpu", torch.float64)  # (MPS has no float64)
        total = float(w.square().sum())
        on_support = float(w[:, self._support_idx].square().sum())
        return {"support_frac": on_support / total if total > 0 else float("nan")}
