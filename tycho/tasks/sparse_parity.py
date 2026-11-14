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

Correlated inputs and fixed supports (sky v2 Part A)
----------------------------------------------------
``rho ≠ 0`` draws the training, train-eval and test inputs from a stationary binary Markov chain along the bit
index: ``x_0`` uniform and ``P(x_{i+1} = x_i) = (1 + rho)/2`` (spin correlation rho^|i−j|, uniform marginals).
``rho = 0`` keeps the uniform sampler above bit for bit. ``support`` (a list of k indices) fixes ``S*`` instead of
drawing it from ``data_seed``. RI probes stay uniform. The task implements
:class:`~tycho.tasks.patterns.PatternTask` for n ≤ 12: code bit j = 1 iff x_j = +1, the output is y = 1; the pmfs
are D = the training chain, U = uniform and S = the same chain with ``shift_rho``. With ``exact_metrics = True`` the
light curve also holds ``x_lossD``/``U``/``S`` (log-loss of the true label) and ``x_sqD``/``U``/``S`` (its Brier
score).
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
from tycho.tasks.patterns import MAX_PATTERN_INPUTS, PatternTask, chain_pmf, float64_copy, pattern_bits

__all__ = ["SparseParity"]


class SparseParity(PatternTask, Task):
    """k-sparse parity of n uniform ±1 bits, learned by an MLP (see the module docstring)."""

    name = "sparse_parity"

    @classmethod
    def defaults(cls) -> dict:
        return {"n": 20, "k": 3, "width": 256, "depth": 1, "act": "relu", "n_train": None, "test_size": 2048,
                "data_seed": 0, "ri_soft": True, "rho": 0.0, "support": None, "shift_rho": -0.5,
                "exact_metrics": False}

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
        self.rho, self.shift_rho = float(c["rho"]), float(c["shift_rho"])
        for name, value in (("rho", self.rho), ("shift_rho", self.shift_rho)):
            if not -1.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (-1, 1), got {value}")
        self.exact = bool(c["exact_metrics"])
        if self.exact and self.n > MAX_PATTERN_INPUTS:
            raise ValueError(f"exact_metrics needs n <= {MAX_PATTERN_INPUTS}, got n={self.n}")
        if c["support"] is None:
            self.support: list[int] = sorted(int(i) for i in np.random.default_rng(self.data_seed).choice(
                self.n, self.k, replace=False))
        else:
            self.support = sorted(int(i) for i in c["support"])
            if len(self.support) != self.k or len(set(self.support)) != self.k or not all(
                    0 <= i < self.n for i in self.support):
                raise ValueError(f"support must be k={self.k} distinct indices in 0..{self.n - 1}, got {c['support']}")
        self._support_idx = torch.tensor(self.support, dtype=torch.long)

    # ------------------------------------------------------------------------------------------ data
    def _sample(self, m: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """``m`` ±1 inputs (float32; uniform, or the Markov chain when rho ≠ 0) and their parity labels (int64)."""
        if self.rho == 0.0:
            x = torch.randint(0, 2, (m, self.n), generator=gen, dtype=torch.float32).mul_(2.0).sub_(1.0)
        else:
            x0 = torch.randint(0, 2, (m, 1), generator=gen, dtype=torch.float32).mul_(2.0).sub_(1.0)
            flips = torch.rand((m, self.n - 1), generator=gen, dtype=torch.float32) < (1.0 - self.rho) / 2.0
            signs = torch.cat([x0, 1.0 - 2.0 * flips.to(torch.float32)], dim=1)
            x = torch.cumprod(signs, dim=1)
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
        out = {"support_frac": on_support / total if total > 0 else float("nan")}
        if self.exact:
            out.update(self.exact_metrics(model))
        return out

    # ------------------------------------------------------------------------------------------ all patterns
    def pattern_dim(self) -> int:
        if self.n > MAX_PATTERN_INPUTS:
            raise ValueError(f"pattern observation needs n <= {MAX_PATTERN_INPUTS}, got n={self.n}")
        return self.n

    def pattern_input_names(self) -> list[str]:
        return [f"x{i}" for i in range(self.n)]

    def pattern_truth(self) -> np.ndarray:
        """P(y = 1 | pattern) ∈ {0, 1}: y = 1 iff Π_{i∈S*} x_i = −1."""
        spins = 2 * pattern_bits(self.pattern_dim()) - 1
        return (np.prod(spins[:, self.support], axis=1) < 0).astype(np.float64)

    def pattern_pmfs(self) -> dict[str, np.ndarray]:
        n = self.pattern_dim()
        return {"D": chain_pmf(n, self.rho), "U": chain_pmf(n, 0.0), "S": chain_pmf(n, self.shift_rho)}

    def pattern_logodds(self, model: nn.Module) -> np.ndarray:
        """float64 log-odds of y = 1 on every pattern, in code order (a float64 copy of the model)."""
        m64 = float64_copy(model)
        x = torch.as_tensor(2 * pattern_bits(self.pattern_dim()) - 1, dtype=torch.float64)
        with torch.no_grad():
            logits = m64(x.to(model_device(m64)))
        return (logits[:, 1] - logits[:, 0]).to("cpu", torch.float64).numpy()
