"""Task interface of the survey objects (SPEC §4.1) and helpers shared by the task modules.

A :class:`Task` defines one learning problem: its data, model family, loss, evaluation splits, task-specific
light-curve metrics and the *RI variables* (the discrete variables RI spectroscopy runs on).

Randomness
----------
Three independent sources, all explicit:

* **Problem randomness** (which parity subset, which train/test split, the Markov couplings' corpus, the fixed
  evaluation sets) is derived from ``cfg["data_seed"]`` only, so all training seeds of one configuration face the
  same problem. Streams are separated with :func:`derive_seed` / :func:`torch_generator`.
* **Training randomness**: the training seed sets the initial weights (``torch.manual_seed(seed)`` right before
  model construction, :meth:`Task.build_model`) and, through the generator the runner passes to
  :meth:`Task.sample_batch`, the batch order.
* **Probe randomness** for RI comes from the generator passed to :meth:`Task.ri_variables`; the runner derives it
  from the spec seed and the checkpoint step, never from the training generator, so observing an object never
  changes its trajectory.

Tensors and devices
-------------------
Tasks generate data on the CPU. ``sample_batch`` and ``eval_splits`` return CPU tensors (the runner moves them to
the model's device). Methods that receive a model (``ri_variables``, ``extra_metrics``) move their inputs to the
model's device and return numpy / Python values.

Frozen RI variable definitions
------------------------------
An RI curve needs the same variables at every checkpoint. When a task's variables depend on the model (e.g.
modular addition picks the most informative hidden units), :meth:`Task.ri_variables` stores the choice it made as a
JSON-friendly dict in ``vs.meta["frozen"]``; callers pass that dict back as ``frozen=`` at later checkpoints of the
same object (the runner does this and saves it in its resume checkpoint).
"""
from __future__ import annotations

import abc
import hashlib
from typing import Any, ClassVar

import numpy as np
import torch
from torch import Tensor, nn

from tycho.ri.variables import VariableSet

__all__ = ["Task", "derive_seed", "torch_generator", "numpy_rng", "model_device", "logit_margin",
           "softmax_np", "split_seed_from"]

_SEED_MASK = (1 << 63) - 1


def derive_seed(*keys: int | str) -> int:
    """A 63-bit seed from a tuple of keys (ints and strings); distinct tuples give unrelated seeds.

    The keys are encoded unambiguously (a type tag and a length prefix per key, so ``(1, "2")``, ``(12,)`` and
    ``(1, 2)`` differ, as do tuples that differ only by trailing zeros) and hashed with SHA-256, which is stable
    across processes, platforms and Python versions. Example: ``derive_seed(seed, "data")``,
    ``derive_seed(seed, "ri", step)``.
    """
    h = hashlib.sha256()
    for key in keys:
        if isinstance(key, str):
            tag, data = b"s", key.encode("utf-8")
        else:
            tag, data = b"i", str(int(key)).encode("ascii")
        h.update(tag + len(data).to_bytes(8, "little") + data)
    return int.from_bytes(h.digest()[:8], "little") & _SEED_MASK


def torch_generator(*keys: int | str) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded with :func:`derive_seed` of ``keys``."""
    return torch.Generator().manual_seed(derive_seed(*keys))


def numpy_rng(*keys: int | str) -> np.random.Generator:
    """A ``numpy.random.Generator`` seeded with :func:`derive_seed` of ``keys``."""
    return np.random.default_rng(derive_seed(*keys))


def split_seed_from(gen: torch.Generator) -> int:
    """Draw one 62-bit integer from a torch generator (to seed a numpy generator for samplers torch lacks)."""
    return int(torch.randint(0, 1 << 62, (1,), generator=gen).item())


def model_device(model: nn.Module) -> torch.device:
    """Device of the model's first parameter (CPU for a parameterless module)."""
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def logit_margin(logits: Tensor, y: Tensor) -> Tensor:
    """True-class logit minus logsumexp of the other logits (the binary margin for two classes).

    Works on any leading shape (``logits (..., C)``, ``y (...)``) and under ``torch.func.vmap``. It equals
    ``log p(y) − log(1 − p(y))``: positive iff the true class has probability > 1/2.
    """
    true = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    is_true = torch.arange(logits.shape[-1], device=logits.device) == y.unsqueeze(-1)
    others = logits.masked_fill(is_true, float("-inf"))
    return true - torch.logsumexp(others, dim=-1)


def softmax_np(logits: Tensor) -> np.ndarray:
    """Row-wise softmax of a logits tensor (..., C) as a float64 numpy array (rows sum to 1 to ~1e-16)."""
    return torch.softmax(logits.detach().to("cpu", torch.float64), dim=-1).numpy()


class Task(abc.ABC):
    """One learning problem of the survey (SPEC §4.1). Subclasses register in :data:`tycho.tasks.REGISTRY`.

    ``self.cfg = {**defaults(), **cfg}``; keys not in :meth:`defaults` raise ``ValueError``. ``self.seed`` is the
    training seed (initialisation and, via the runner's generator, batch order).
    """

    name: ClassVar[str] = "task"

    def __init__(self, cfg: dict | None, seed: int) -> None:
        cfg = dict(cfg or {})
        known = self.defaults()
        unknown = sorted(set(cfg) - set(known))
        if unknown:
            raise ValueError(f"{self.name}: unknown config keys {unknown}; known keys: {sorted(known)}")
        self.cfg: dict[str, Any] = {**known, **cfg}
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")

    # ------------------------------------------------------------------------------------------ definition
    @classmethod
    def defaults(cls) -> dict:
        """Default configuration (every accepted key)."""
        return {}

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """A freshly initialised model; calls ``torch.manual_seed(self.seed)`` first (deterministic init)."""

    @abc.abstractmethod
    def sample_batch(self, batch_size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """One training batch ``(inputs, targets)`` on the CPU, drawn with ``gen`` only."""

    @abc.abstractmethod
    def eval_splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Fixed evaluation sets (CPU tensors, identical on every call); always contains "train" and "test"."""

    @abc.abstractmethod
    def loss(self, out: Tensor, y: Tensor) -> Tensor:
        """Scalar training loss, averaged over the batch."""

    def accuracy(self, out: Tensor, y: Tensor) -> float | None:
        """Fraction of correct predictions, or None when the task has no accuracy."""
        return None

    def chance_accuracy(self) -> float | None:
        """Accuracy of an uninformed predictor (None when accuracy is undefined)."""
        return None

    @abc.abstractmethod
    def ntk_output(self, out: Tensor, y: Tensor) -> Tensor:
        """Per-sample scalar whose parameter gradients define the empirical NTK; shape ``(B,)``. Must be
        ``torch.func.vmap``-compatible (no data-dependent shapes or Python control flow on values)."""

    @abc.abstractmethod
    def ri_variables(self, model: nn.Module, n: int, gen: torch.Generator, *,
                     frozen: dict | None = None) -> VariableSet:
        """The RI variables of ``model`` on ``n`` probe samples drawn with ``gen`` (see the module docstring for
        ``frozen``). Must not modify the model or use any other source of randomness."""

    def ground_truth(self) -> list[dict] | None:
        """Interactions known to exist (``[{"subset": [names...], "order": k, ...}]``), or None."""
        return None

    def extra_metrics(self, model: nn.Module) -> dict[str, float]:
        """Task-specific light-curve metrics (keys without the ``x_`` prefix); the key set must not vary."""
        return {}

    @abc.abstractmethod
    def default_ri_order(self) -> int:
        """Maximum subset order of RI scans when ``RIConfig.max_order`` is None."""

    # ------------------------------------------------------------------------------------------ photometry hooks
    def hidden_representation(self, model: nn.Module, x: Tensor) -> Tensor:
        """Representation used for ``erank_hidden`` and ``rep_cka_prev``: a 2-D ``(rows, features)`` tensor.

        Default: the last hidden layer of a model with a ``hidden(x)`` method; for a transformer, the final
        residual-stream state at every position (rows = batch × positions).
        """
        if hasattr(model, "hidden"):
            h = model.hidden(x)[-1]
        else:
            _, states = model(x, return_hidden=True)
            h = states[-1]
        return h.reshape(-1, h.shape[-1])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, cfg={self.cfg})"
