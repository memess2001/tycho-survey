"""Modular addition: the grokking object (Power et al. 2022; Gromov 2023 for one-hot MLPs).

Problem
-------
All ``p²`` pairs ``(a, b)`` with target ``(a + b) mod p``; input ``concat(onehot(a), onehot(b))`` (2p floats). A
fixed random split (``data_seed``) puts ``round(train_frac · p²)`` pairs in the training set and the rest in the
test set. Minibatches are uniform random subsets of the training set without replacement; a batch size of at
least the training-set size means full-batch training (no randomness). With weight decay the test accuracy
typically jumps long after the training set has been fitted ("grokking").

RI variables
------------
The probe is the whole input space (all ``p²`` pairs, so ``n`` is ignored and the RI values are exact for the
model; the generator is unused). Variables: the units of the last hidden layer binarised with
``binarize(method="zero")`` (ReLU unit active or not), of which the ``ri_units`` with the highest entropy are kept
(``select_informative``), named ``h{unit}``, plus ``correct`` (1 iff the argmax prediction is right). The unit
choice is made at the first call for an object and then *frozen* (returned in ``vs.meta["frozen"]``, passed back
as ``frozen=`` by the runner), so that the RI curve follows the same units at every checkpoint. There is no ground
truth and no required variable.

Extra light-curve metrics
-------------------------
Fourier structure of the first-layer weight ``W₁`` (``width × 2p``; row j = hidden unit j). For unit j and frequency
f, ``P_j(f) = |DFT(W₁[j, :p])(f)|² + |DFT(W₁[j, p:])(f)|²`` (both operands, DFT along the token axis). Frequencies
f and p − f are paired (equal power for real weights) and f = 0 is dropped, leaving ``⌊p/2⌋`` frequency pairs.

* ``fourier_unit_nfreq``: per-unit effective number of frequency pairs ``exp(H(P_j / Σ P_j))``, averaged over units
  with weights ``Σ_f P_j(f)``. Random weights give about 0.8 · ⌊p/2⌋ (8.95 for p = 23); the generalising solutions
  of modular addition make every unit use one or a few frequencies (about 2 for the sky-v0 ReLU MLPs), and the
  value falls steadily during the memorisation plateau, before the test accuracy rises: a hidden-progress measure.
* ``fourier_nfreq``: effective number of frequency pairs of the total power ``Σ_j P_j``: how many distinct
  frequencies the network uses. A wide one-hot MLP typically spreads its units over all frequencies (value near
  ⌊p/2⌋ even after grokking), whereas a narrow embedding bottleneck concentrates on a few key frequencies
  (Nanda et al. 2023).

Both are NaN when the non-constant power is below :data:`POWER_FLOOR_REL` of the total power.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tycho.models.mlp import MLP
from tycho.ri.discretize import binarize, select_informative
from tycho.ri.variables import VariableSet
from tycho.tasks.base import Task, logit_margin, model_device

__all__ = ["EMBEDDINGS", "POWER_FLOOR_REL", "ModularAddition", "fourier_frequency_counts"]

#: Supported input embeddings.
EMBEDDINGS = ("onehot",)
#: Non-constant Fourier power below this fraction of the total is rounding noise (the Fourier metrics are NaN).
POWER_FLOOR_REL = 1e-12


def _exp_entropy(q: np.ndarray, axis: int = -1) -> np.ndarray:
    """``exp(entropy)`` of distributions along ``axis`` (0 log 0 = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0), axis=axis)
    return np.exp(h)


def fourier_frequency_counts(w_in: np.ndarray, p: int) -> tuple[float, float]:
    """``(fourier_nfreq, fourier_unit_nfreq)`` of input weights ``w_in`` (``width × 2p``); see the module docstring.
    Both are NaN if the non-constant (f ≠ 0) power is below :data:`POWER_FLOOR_REL` of the total power."""
    w_in = np.asarray(w_in, dtype=np.float64)
    spec = np.square(np.abs(np.fft.fft(w_in[:, :p], axis=1))) + np.square(np.abs(np.fft.fft(w_in[:, p:2 * p], axis=1)))
    half = p // 2
    pairs = spec[:, 1:half + 1].copy()  # frequency pairs f = 1..⌊p/2⌋
    partner = p - np.arange(1, half + 1)
    distinct = partner != np.arange(1, half + 1)  # f = p/2 (p even) has no partner
    pairs[:, distinct] += spec[:, partner[distinct]]
    unit_power = pairs.sum(axis=1)
    total = float(unit_power.sum())
    if not np.isfinite(total) or total <= POWER_FLOOR_REL * float(spec.sum()):
        return float("nan"), float("nan")
    global_n = float(_exp_entropy(pairs.sum(axis=0) / total))
    live = unit_power > 0
    unit_n = _exp_entropy(pairs[live] / unit_power[live, None], axis=1)
    return global_n, float(np.sum(unit_power[live] * unit_n) / total)


class ModularAddition(Task):
    """``(a + b) mod p`` with one-hot inputs and an MLP (see the module docstring)."""

    name = "modular_addition"

    @classmethod
    def defaults(cls) -> dict:
        return {"p": 23, "train_frac": 0.5, "width": 128, "depth": 1, "act": "relu", "data_seed": 0,
                "embed": "onehot", "ri_units": 10}

    def __init__(self, cfg: dict | None, seed: int) -> None:
        super().__init__(cfg, seed)
        c = self.cfg
        self.p, self.train_frac = int(c["p"]), float(c["train_frac"])
        self.width, self.depth, self.act = int(c["width"]), int(c["depth"]), str(c["act"])
        self.data_seed, self.ri_units, self.embed = int(c["data_seed"]), int(c["ri_units"]), str(c["embed"])
        if self.p < 2:
            raise ValueError(f"p must be >= 2, got {self.p}")
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError(f"train_frac must lie in (0, 1), got {self.train_frac}")
        if self.embed not in EMBEDDINGS:
            raise ValueError(f"embed must be one of {EMBEDDINGS}, got {self.embed!r}")
        if self.ri_units < 1:
            raise ValueError(f"ri_units must be >= 1, got {self.ri_units}")
        n_pairs = self.p * self.p
        self.n_train = int(min(max(round(self.train_frac * n_pairs), 1), n_pairs - 1))
        perm = np.random.default_rng(self.data_seed).permutation(n_pairs)
        self.train_idx, self.test_idx = np.sort(perm[:self.n_train]), np.sort(perm[self.n_train:])

    # ------------------------------------------------------------------------------------------ data
    @cached_property
    def _all(self) -> tuple[Tensor, Tensor]:
        """Inputs (p², 2p) and targets (p²,) of all pairs, pair index ``a·p + b``."""
        a, b = np.divmod(np.arange(self.p * self.p), self.p)
        x = torch.zeros(self.p * self.p, 2 * self.p)
        rows = torch.arange(self.p * self.p)
        x[rows, torch.as_tensor(a)] = 1.0
        x[rows, torch.as_tensor(self.p + b)] = 1.0
        return x, torch.as_tensor((a + b) % self.p, dtype=torch.long)

    @cached_property
    def _splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        x, y = self._all
        tr, te = torch.as_tensor(self.train_idx), torch.as_tensor(self.test_idx)
        return {"train": (x[tr], y[tr]), "test": (x[te], y[te])}

    def eval_splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        return dict(self._splits)

    def sample_batch(self, batch_size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        x, y = self._splits["train"]
        if batch_size >= self.n_train:
            return x, y
        idx = torch.randperm(self.n_train, generator=gen)[:batch_size]
        return x[idx], y[idx]

    # ------------------------------------------------------------------------------------------ model / loss
    def build_model(self) -> nn.Module:
        torch.manual_seed(self.seed)
        return MLP(2 * self.p, self.width, self.p, depth=self.depth, act=self.act)

    def loss(self, out: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(out, y)

    def accuracy(self, out: Tensor, y: Tensor) -> float | None:
        return float((out.argmax(dim=-1) == y).float().mean().item())

    def chance_accuracy(self) -> float | None:
        return 1.0 / self.p

    def ntk_output(self, out: Tensor, y: Tensor) -> Tensor:
        return logit_margin(out, y)

    # ------------------------------------------------------------------------------------------ observation
    def default_ri_order(self) -> int:
        return 3

    def ri_variables(self, model: nn.Module, n: int, gen: torch.Generator, *,
                     frozen: dict | None = None) -> VariableSet:
        x, y = self._all
        with torch.no_grad():
            x_dev = x.to(model_device(model))
            hidden = model.hidden(x_dev)[-1].to("cpu", torch.float64).numpy()
            correct = (model(x_dev).argmax(dim=-1).cpu() == y).numpy().astype(np.int64)
        active = binarize(hidden, method="zero")
        if frozen is not None:
            units = np.asarray(frozen["units"], dtype=np.int64)
        else:
            units = select_informative(active, self.ri_units, method="entropy")
        X = np.concatenate([active[:, units].astype(np.int64), correct[:, None]], axis=1)
        names = [f"h{int(u)}" for u in units] + ["correct"]
        return VariableSet(X=X, cards=[2] * X.shape[1], names=names, must_include=None, ground_truth=None,
                           meta={"task": self.name, "probe": "all_pairs", "binarize": "zero",
                                 "frozen": {"units": [int(u) for u in units]}})

    def extra_metrics(self, model: nn.Module) -> dict[str, float]:
        w = model.first_layer().weight.detach().to("cpu", torch.float64).numpy()
        global_n, unit_n = fourier_frequency_counts(w, self.p)
        return {"fourier_nfreq": global_n, "fourier_unit_nfreq": unit_n}
