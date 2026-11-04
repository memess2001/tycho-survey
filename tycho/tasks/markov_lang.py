"""Markov language with planted interaction orders: the token-level positive control.

Source
------
Binary tokens ``x_t ∈ {0, 1}`` with spins ``s_t = 2x_t − 1`` form a stationary order-W Markov chain with

    P(x_t = 1 | context) = σ(2h),     h = bias + Σ_{L ∈ couplings} β_L ∏_{ℓ ∈ L} s_{t−ℓ},

i.e. ``log P(x_t | ctx) = s_t h − log(2 cosh h)``: each coupling ``L`` (a set of lags in ``1..W``) is an
interaction of order ``|L| + 1`` between x_t and the lagged tokens. ``couplings`` is given as a dict
``{"1": 1.0, "1,3": 0.8}`` (keys = comma-separated lags) or as the string ``"1:1.0;1,3:0.8"`` (the sky format).
A corpus of ``n_tokens`` tokens is generated once from ``data_seed`` after a burn-in of :data:`BURN_IN` tokens
(a uniform random initial window); a training example is a random position t of the corpus (``n_train`` restricts
the positions to the first ``n_train``), with input the window ``x[t-1] … x[t-W]`` and target x_t. The MLP reads the
window as ±1 floats in lag order; the transformer (``model = "transformer"``, a :class:`~tycho.models.TinyGPT` with
``d_model``, ``n_layer``, ``n_head``) reads token ids in temporal order and predicts at the last position.

RI variables and why the probe contexts are uniform
---------------------------------------------------
:meth:`MarkovLanguage.ri_variables` returns the window variables ``x[t-1] … x[t-W]`` (code 1 = token 1 = spin +1)
and ``xhat``, a *soft* column holding the model's predictive distribution of x_t; scans require ``xhat`` in every
subset. With ``ri_context = "uniform"`` (default) the probe windows are i.i.d. uniform random bits, not windows of
the chain. The context variables are then mutually independent, so the local RI of a subset containing ``xhat``
reflects how the *model's prediction* depends on those lags, not correlations of the source.

For the true conditional this yields an exactly known spectrum. Let y be the spin of the predicted token, A a set of
lags and B the other lags. With uniform contexts ``P(y, s_A) = 2^{−|A|} P(y | s_A)`` with
``P(y = +1 | s_A) = E_{s_B}[σ(2h(s))]``, and ``log P(y, s_A) = c(s_A) + (y/2) · logit P(y = +1 | s_A)``. So the
top-order interaction of the marginal on ``A ∪ {y}`` (and hence its local RI) is non-zero iff the Walsh coefficient
of ``logit P(y = +1 | s_A)`` on the full set A is non-zero. Consequences:

* If A contains every coupled lag, ``logit P(y = +1 | s_A) = 2h`` exactly, whose Walsh coefficients are the
  ``2β_L``: ``RI(A ∪ {xhat}) > 0`` iff A is a coupling set.
* If A misses a lag ℓ, averaging over ``s_ℓ`` turns each coupling term containing ℓ into a uniform random sign
  independent of ``s_A``. With ``bias = 0`` and at most two couplings (this covers every coupling set of sky v0),
  ``P(y = +1 | s_A)`` is then either 1/2 or a function of the one coupling product ``∏_{ℓ∈L} s_ℓ`` with ``L ⊆ A``
  (any function of a ±1 product is affine in it). **In this regime the population local RI is positive exactly on
  the planted subsets** ``{x[t-ℓ] : ℓ ∈ L} ∪ {xhat}`` **and exactly 0 on every other subset containing xhat**
  (tested with :meth:`MarkovLanguage.true_variables` and with the exact population table).
* Outside that regime, marginalising the unobserved lags can induce small top-order terms on unplanted subsets.
  A bias makes ``P(y | s_A)`` depend on ``|β₁ + β₂ χ|`` (couplings {1}, {1,3} with bias ≠ 0 induce
  ``{x[t-3], xhat}``), and three couplings can combine (the path {1,2}, {1,4}, {3,4} induces
  ``{x[t-2], x[t-3], xhat}``). :meth:`true_variables` gives the exact population reference in every case; the
  ground truth lists the planted couplings only.

With ``ri_context = "chain"`` probe windows are random windows of the training corpus. The context variables are
then correlated through the source and the local RI of subsets containing xhat mixes the model's structure with
those correlations (use it to see the model on its own data distribution).

Extra light-curve metric
------------------------
``kl_true``: mean over the test split of ``KL(P_true(x_t | ctx) ‖ P_model(x_t | ctx))`` in nats, the model's excess
cross-entropy over the Bayes-optimal predictor (0 for a perfect model).
"""
from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tycho.models.mlp import MLP
from tycho.models.transformer import TinyGPT
from tycho.ri.variables import VariableSet
from tycho.tasks.base import Task, logit_margin, model_device, numpy_rng, softmax_np, torch_generator

__all__ = ["BURN_IN", "MODELS", "RI_CONTEXTS", "MarkovLanguage", "parse_couplings"]

#: Tokens generated and discarded before a corpus starts (the chain forgets its uniform initial window).
BURN_IN = 1000
#: Model families.
MODELS = ("mlp", "transformer")
#: Probe-context distributions of the RI variables (see the module docstring).
RI_CONTEXTS = ("uniform", "chain")

Coupling = tuple[int, ...]


def parse_couplings(spec: Any, window: int) -> dict[Coupling, float]:
    """Parse couplings given as ``{"1": 1.0, "1,3": 0.8}`` or ``"1:1.0;1,3:0.8"`` into ``{(1,): 1.0, (1, 3): 0.8}``.

    Dict keys may also be ints or sequences of ints. Lags must lie in ``1..window``, be distinct within a coupling,
    and each coupling may appear once. The result is sorted by (order, lags).

    Raises:
        ValueError: on malformed input.
    """
    if isinstance(spec, str):
        items: list[tuple[Any, Any]] = []
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"coupling {part!r} must look like 'lag,lag,...:beta'")
            lags, beta = part.rsplit(":", 1)
            items.append((lags, beta))
    elif isinstance(spec, dict):
        items = list(spec.items())
    else:
        raise ValueError(f"couplings must be a dict or a string, got {type(spec).__name__}")
    out: dict[Coupling, float] = {}
    for key, beta in items:
        try:
            if isinstance(key, str):
                lags = [int(s) for s in key.replace(" ", "").split(",") if s]
            elif isinstance(key, (int, np.integer)):
                lags = [int(key)]
            else:
                lags = [int(v) for v in key]
            value = float(beta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed coupling {key!r}: {beta!r}") from exc
        L = tuple(sorted(lags))
        if not L or len(set(L)) != len(L) or L[0] < 1 or L[-1] > window:
            raise ValueError(f"coupling lags {key!r} must be distinct and lie in 1..{window}")
        if not np.isfinite(value):
            raise ValueError(f"coupling {key!r} has a non-finite strength {beta!r}")
        if L in out:
            raise ValueError(f"coupling {L} given twice")
        out[L] = value
    return dict(sorted(out.items(), key=lambda kv: (len(kv[0]), kv[0])))


def _conditional_table(window: int, couplings: dict[Coupling, float], bias: float) -> np.ndarray:
    """``P(x_t = 1 | context code c)`` for all ``2^W`` codes ``c = Σ_ℓ x_{t−ℓ} 2^{ℓ−1}`` (float64)."""
    codes = np.arange(1 << window)
    spins = 2.0 * ((codes[:, None] >> np.arange(window)[None, :]) & 1) - 1.0  # column ℓ−1 = spin of lag ℓ
    h = np.full(codes.size, float(bias))
    for L, beta in couplings.items():
        h += beta * np.prod(spins[:, [lag - 1 for lag in L]], axis=1)
    return 1.0 / (1.0 + np.exp(-2.0 * h))


def _generate(p_table: np.ndarray, window: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` tokens (uint8) of the chain after :data:`BURN_IN` burn-in tokens from a uniform random window."""
    total = n + BURN_IN
    u = rng.random(total).tolist()
    p = p_table.tolist()
    mask = (1 << window) - 1
    c = int(rng.integers(0, 1 << window))
    out = bytearray(total)
    for t in range(total):  # sequential by nature; plain Python ints are the fastest option without numba
        b = 1 if u[t] < p[c] else 0
        out[t] = b
        c = ((c << 1) | b) & mask
    return np.frombuffer(out, dtype=np.uint8, offset=BURN_IN).copy()


def _lag_windows(seq: np.ndarray, window: int) -> np.ndarray:
    """Lag-order windows ``(len(seq) − W, W)``: row i, column ℓ−1 = ``seq[W + i − ℓ]``; targets are ``seq[W:]``."""
    temporal = np.lib.stride_tricks.sliding_window_view(seq, window)[:-1]
    return np.ascontiguousarray(temporal[:, ::-1])


class MarkovLanguage(Task):
    """Order-W binary Markov source with planted couplings (see the module docstring)."""

    name = "markov_lang"

    @classmethod
    def defaults(cls) -> dict:
        return {"window": 6, "couplings": {"1": 1.0, "1,3": 0.8}, "bias": 0.0, "n_tokens": 200_000, "width": 128,
                "depth": 1, "act": "relu", "model": "mlp", "d_model": 32, "n_layer": 2, "n_head": 2,
                "data_seed": 0, "n_train": None, "test_size": 4096, "ri_context": "uniform"}

    def __init__(self, cfg: dict | None, seed: int) -> None:
        super().__init__(cfg, seed)
        c = self.cfg
        self.window = int(c["window"])
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        self.couplings = parse_couplings(c["couplings"], self.window)
        self.bias = float(c["bias"])
        self.n_tokens, self.test_size, self.data_seed = int(c["n_tokens"]), int(c["test_size"]), int(c["data_seed"])
        self.n_train = None if c["n_train"] is None else int(c["n_train"])
        self.model_kind, self.ri_context = str(c["model"]), str(c["ri_context"])
        if self.model_kind not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model_kind!r}")
        if self.ri_context not in RI_CONTEXTS:
            raise ValueError(f"ri_context must be one of {RI_CONTEXTS}, got {self.ri_context!r}")
        if self.n_tokens <= self.window:
            raise ValueError(f"n_tokens must exceed window, got {self.n_tokens}")
        if self.n_train is not None and self.n_train < 1:
            raise ValueError(f"n_train must be None or >= 1, got {self.n_train}")
        if self.test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {self.test_size}")
        self.p_table = _conditional_table(self.window, self.couplings, self.bias)
        self.lag_names = [f"x[t-{lag}]" for lag in range(1, self.window + 1)]

    # ------------------------------------------------------------------------------------------ source
    def true_conditional(self, lag_bits: np.ndarray) -> np.ndarray:
        """``P(x_t = 1 | window)`` for lag-order 0/1 windows ``(m, W)`` (column ℓ−1 = token at lag ℓ)."""
        bits = np.asarray(lag_bits, dtype=np.int64)
        codes = (bits << np.arange(self.window)[None, :]).sum(axis=1)
        return self.p_table[codes]

    def _model_inputs(self, lag_bits: np.ndarray | Tensor) -> Tensor:
        """Model inputs for lag-order 0/1 windows: ±1 floats in lag order (MLP) or temporal-order token ids."""
        bits = torch.as_tensor(np.asarray(lag_bits), dtype=torch.long)
        if self.model_kind == "mlp":
            return bits.to(torch.float32).mul_(2.0).sub_(1.0)
        return bits.flip(1).contiguous()

    @cached_property
    def _corpus(self) -> dict[str, Any]:
        """Training corpus: lag windows (uint8), model inputs, targets, and the number of training positions."""
        seq = _generate(self.p_table, self.window, self.n_tokens, numpy_rng(self.data_seed, "markov_lang/corpus"))
        lag = _lag_windows(seq, self.window)
        n_pos = lag.shape[0] if self.n_train is None else min(self.n_train, lag.shape[0])
        return {"lag": lag, "inputs": self._model_inputs(lag), "targets": torch.as_tensor(seq[self.window:],
                                                                                          dtype=torch.long),
                "n_pos": n_pos}

    @cached_property
    def _test(self) -> dict[str, Any]:
        seq = _generate(self.p_table, self.window, self.test_size + self.window,
                        numpy_rng(self.data_seed, "markov_lang/test"))
        lag = _lag_windows(seq, self.window)
        return {"inputs": self._model_inputs(lag), "targets": torch.as_tensor(seq[self.window:], dtype=torch.long),
                "p_true": self.true_conditional(lag)}

    # ------------------------------------------------------------------------------------------ data
    def sample_batch(self, batch_size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        corpus = self._corpus
        idx = torch.randint(0, corpus["n_pos"], (batch_size,), generator=gen)
        return corpus["inputs"][idx], corpus["targets"][idx]

    @cached_property
    def _splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        corpus = self._corpus
        m = min(self.test_size, corpus["n_pos"])
        idx = torch.randperm(corpus["n_pos"], generator=torch_generator(self.data_seed, "markov_lang/train-eval"))[:m]
        return {"train": (corpus["inputs"][idx], corpus["targets"][idx]),
                "test": (self._test["inputs"], self._test["targets"])}

    def eval_splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        return dict(self._splits)

    # ------------------------------------------------------------------------------------------ model / loss
    def build_model(self) -> nn.Module:
        torch.manual_seed(self.seed)
        if self.model_kind == "mlp":
            return MLP(self.window, int(self.cfg["width"]), 2, depth=int(self.cfg["depth"]), act=str(self.cfg["act"]))
        return TinyGPT(2, int(self.cfg["d_model"]), int(self.cfg["n_layer"]), int(self.cfg["n_head"]),
                       max_len=self.window)

    @staticmethod
    def _last(out: Tensor) -> Tensor:
        """Logits of the prediction for x_t: the transformer's last position, or the MLP output."""
        return out[:, -1] if out.dim() == 3 else out

    def loss(self, out: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(self._last(out), y)

    def accuracy(self, out: Tensor, y: Tensor) -> float | None:
        return float((self._last(out).argmax(dim=-1) == y).float().mean().item())

    def chance_accuracy(self) -> float | None:
        """Majority-token frequency of the training corpus."""
        frac = float(self._corpus["targets"][: self._corpus["n_pos"]].float().mean())
        return max(frac, 1.0 - frac)

    def ntk_output(self, out: Tensor, y: Tensor) -> Tensor:
        return logit_margin(self._last(out), y)

    # ------------------------------------------------------------------------------------------ observation
    def ground_truth(self) -> list[dict] | None:
        """One entry per planted coupling with β ≠ 0 (an empty list for a source without couplings)."""
        return [{"subset": [f"x[t-{lag}]" for lag in L] + ["xhat"], "order": len(L) + 1, "beta": beta}
                for L, beta in self.couplings.items() if beta != 0.0]

    def default_ri_order(self) -> int:
        """One more than the largest planted coupling (at least 2)."""
        return max([2] + [len(L) + 1 for L, beta in self.couplings.items() if beta != 0.0])

    def _probe_windows(self, n: int, gen: torch.Generator) -> np.ndarray:
        """``n`` lag-order 0/1 probe windows: uniform bits, or random windows of the training corpus."""
        if self.ri_context == "uniform":
            return torch.randint(0, 2, (n, self.window), generator=gen).numpy()
        idx = torch.randint(0, self._corpus["n_pos"], (n,), generator=gen).numpy()
        return self._corpus["lag"][idx].astype(np.int64)

    def _variable_set(self, lag_bits: np.ndarray, probs: np.ndarray, kind: str) -> VariableSet:
        X = np.concatenate([lag_bits.astype(np.int64), probs.argmax(axis=1)[:, None]], axis=1)
        return VariableSet(X=X, cards=[2] * (self.window + 1), names=self.lag_names + ["xhat"],
                           soft={self.window: probs}, must_include=[self.window], ground_truth=self.ground_truth(),
                           meta={"task": self.name, "probe": self.ri_context, "output": kind})

    def ri_variables(self, model: nn.Module, n: int, gen: torch.Generator, *,
                     frozen: dict | None = None) -> VariableSet:
        lag_bits = self._probe_windows(int(n), gen)
        with torch.no_grad():
            logits = self._last(model(self._model_inputs(lag_bits).to(model_device(model))))
        return self._variable_set(lag_bits, softmax_np(logits), "model")

    def true_variables(self, n: int, gen: torch.Generator) -> VariableSet:
        """The same variables with the *true* conditional as the soft ``xhat`` column (the "true spectrum")."""
        lag_bits = self._probe_windows(int(n), gen)
        p1 = self.true_conditional(lag_bits)
        return self._variable_set(lag_bits, np.stack([1.0 - p1, p1], axis=1), "true")

    def extra_metrics(self, model: nn.Module) -> dict[str, float]:
        test = self._test
        with torch.no_grad():
            logits = self._last(model(test["inputs"].to(model_device(model))))
        log_q = torch.log_softmax(logits.detach().to("cpu", torch.float64), dim=-1).numpy()
        p1 = test["p_true"]
        p = np.stack([1.0 - p1, p1], axis=1)
        safe_p = np.where(p > 0.0, p, 1.0)  # 0 log 0 = 0 (σ(2h) saturates in float64 for very strong couplings)
        kl = np.sum(np.where(p > 0.0, p * (np.log(safe_p) - log_q), 0.0), axis=1)
        return {"kl_true": float(kl.mean())}
