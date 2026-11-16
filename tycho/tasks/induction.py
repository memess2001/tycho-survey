"""Induction heads in a small transformer (sky v2 Part B; plan: docs/results/v2b_pilot_plan.md).

Problem
-------
A sequence of ``seq_len`` tokens is drawn uniformly from ``vocab`` tokens. With probability ``p_repeat`` it has
repeats, in one of two layouts (``layout``):

* ``"single"``: one repeated segment of length ``L`` uniform in ``[rep_min, rep_max]`` and start ``s`` uniform in
  ``[0, seq_len − 2L]``; tokens ``s+L … s+2L−1`` copy tokens ``s … s+L−1``.
* ``"tiled"``: a random segment of length ``L`` uniform in ``[rep_min, rep_max]`` is repeated to fill the sequence,
  starting at a random phase. Every token from position ``L`` on is predictable. The Part B pilot found that the
  single layout forms the circuit in only some runs; the tiled layout formed it in 6 of 6 (plan, §2).

 A causal
:class:`~tycho.models.TinyGPT` reads tokens ``0..T−2`` and is trained with the mean next-token cross-entropy. Inside
a repeat, the next token is predictable from content only: the segment's length and position vary, so a head cannot
find the earlier occurrence by position (with a fixed offset a single positional head solves the task; see the
plan, §2). The mechanism that forms is the two-layer induction circuit: a previous-token head in an early layer and
an induction head that attends to the token after the previous occurrence of the current token. ``p_repeat = 0``
(no repeats in training: uniform random tokens) and ``n_layer = 1`` are the negative controls.

Evaluation
----------
"train" and "test" are fixed sets of ``eval_size`` sequences of the training distribution (two ``data_seed``
streams). A separate *probe set* of ``probe_size`` sequences always contains a repeat (whatever ``p_repeat``); it
serves the extra metrics and the pattern observation, so that the controls are measured on the same footing.

Extra light-curve metrics (on the probe set)
--------------------------------------------
* ``loss_rep``: mean next-token loss on the predictable repeated tokens; ``loss_other``: on all other targets
  (``nan`` in the tiled layout when there are none); ``icl_score = loss_other − loss_rep``.
* Per head (``l<layer>h<head>``): ``prev_…`` = mean attention from position t to t − 1 (t ≥ 1); ``ind_…`` = mean
  attention mass from a predictable position t to the keys j ≤ t whose previous token equals the current one
  (``x[j−1] = x[t]``), i.e. to the tokens that followed earlier occurrences of the current token; ``dup_…`` = mean
  attention mass from t to earlier positions holding the current token itself (a duplicate-token head). These are
  the attention-pattern scores of the circuits literature, used as the mechanism referee: neither the θ nor the η
  view uses them.

Pattern mode: head-ablation patterns
------------------------------------
The variables are the ``H = n_layer · n_head`` heads, each kept (1) or zero-ablated (0); pattern code
``c = Σ_j keep_j 2^j`` with head ``j = l · n_head + h``. The output is "the model predicts the repeated token": for
pattern c, p(c) is the mean probability of the correct token over the probe set's predictable repeated tokens, and
``pattern_logodds`` returns log(p / (1 − p)). The pmfs are uniform (the ablation design is ours: D = U = S). There is
no ground-truth conditional (``pattern_truth`` is None). Every pattern is enumerated, so ``H ≤ 12``; one forward pass
of the probe set per pattern.

RI scans (``ri.mode = "scan"``) are not supported; use ``ri.mode = "patterns"``.
"""
from __future__ import annotations

import math
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tycho.models.transformer import TinyGPT
from tycho.ri.variables import VariableSet
from tycho.tasks.base import Task, logit_margin, model_device, torch_generator
from tycho.tasks.patterns import MAX_PATTERN_INPUTS, PatternTask, pattern_bits, uniform_pmf

__all__ = ["LAYOUTS", "Induction", "repeat_sequences", "tiled_sequences"]

#: Clip of the pattern output probability before the log-odds.
P_CLIP = 1e-12


#: Sequence layouts (module docstring).
LAYOUTS = ("single", "tiled")


def tiled_sequences(m: int, seq_len: int, vocab: int, rep_min: int, rep_max: int, p_repeat: float,
                    gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """``m`` sequences of a random segment repeated to fill ``seq_len`` (with probability ``p_repeat``; otherwise
    uniform tokens). Returns ``(x, rep, src)`` as :func:`repeat_sequences`; ``src`` is the most recent match."""
    x = torch.randint(0, vocab, (m, seq_len), generator=gen)
    has = torch.rand(m, generator=gen) < p_repeat
    L = torch.randint(rep_min, rep_max + 1, (m,), generator=gen)
    seg = torch.randint(0, vocab, (m, rep_max), generator=gen)
    phase = (torch.rand(m, generator=gen) * L.to(torch.float32)).floor().to(torch.long)
    idx = torch.arange(seq_len)[None, :]
    tiled = seg.gather(1, (idx + phase[:, None]) % L[:, None])
    x = torch.where(has[:, None], tiled, x)
    t = torch.arange(seq_len - 1)[None, :]
    rep = has[:, None] & (t >= L[:, None])
    src = torch.where(rep, t - L[:, None] + 1, torch.full_like(t.expand(m, -1), -1))
    return x, rep, src


def repeat_sequences(m: int, seq_len: int, vocab: int, rep_min: int, rep_max: int, p_repeat: float,
                     gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """``m`` sequences (int64, ``(m, seq_len)``) with at most one repeated segment each (module docstring).

    Returns ``(x, rep, src)``: ``rep (m, seq_len − 1)`` marks the input positions t whose target ``x[t+1]`` is a
    predictable repeated token, and ``src (m, seq_len − 1)`` is the position an induction head should attend to from
    t (``t − L + 1``), −1 elsewhere."""
    x = torch.randint(0, vocab, (m, seq_len), generator=gen)
    has = torch.rand(m, generator=gen) < p_repeat
    L = torch.randint(rep_min, rep_max + 1, (m,), generator=gen)
    s = (torch.rand(m, generator=gen) * (seq_len - 2 * L + 1).to(torch.float32)).floor().to(torch.long)
    idx = torch.arange(seq_len)[None, :]
    in_copy = has[:, None] & (idx >= (s + L)[:, None]) & (idx < (s + 2 * L)[:, None])
    copied = x.gather(1, (idx - L[:, None]).clamp(min=0))
    x = torch.where(in_copy, copied, x)
    t = torch.arange(seq_len - 1)[None, :]
    rep = has[:, None] & (t >= (s + L)[:, None]) & (t <= (s + 2 * L - 2)[:, None])
    src = torch.where(rep, t - L[:, None] + 1, torch.full_like(t.expand(m, -1), -1))
    return x, rep, src


class Induction(PatternTask, Task):
    """Next-token prediction on random tokens with a repeated segment (see the module docstring)."""

    name = "induction"

    @classmethod
    def defaults(cls) -> dict:
        return {"vocab": 64, "seq_len": 64, "rep_min": 12, "rep_max": 32, "p_repeat": 1.0, "d_model": 64,
                "n_layer": 2, "n_head": 4, "attn_only": True, "mlp_ratio": 4, "data_seed": 0, "eval_size": 512,
                "probe_size": 128, "layout": "single"}

    def __init__(self, cfg: dict | None, seed: int) -> None:
        super().__init__(cfg, seed)
        c = self.cfg
        self.vocab, self.seq_len = int(c["vocab"]), int(c["seq_len"])
        self.rep_min, self.rep_max, self.p_repeat = int(c["rep_min"]), int(c["rep_max"]), float(c["p_repeat"])
        self.n_layer, self.n_head = int(c["n_layer"]), int(c["n_head"])
        self.data_seed, self.eval_size, self.probe_size = int(c["data_seed"]), int(c["eval_size"]), int(c["probe_size"])
        self.layout = str(c["layout"])
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        if self.vocab < 2:
            raise ValueError(f"vocab must be >= 2, got {self.vocab}")
        if not 2 <= self.rep_min <= self.rep_max or 2 * self.rep_max > self.seq_len:
            raise ValueError(f"need 2 <= rep_min <= rep_max and 2 * rep_max <= seq_len, got {self.rep_min}, "
                             f"{self.rep_max}, {self.seq_len}")
        if not 0.0 <= self.p_repeat <= 1.0:
            raise ValueError(f"p_repeat must lie in [0, 1], got {self.p_repeat}")
        if min(self.eval_size, self.probe_size) < 1:
            raise ValueError("eval_size and probe_size must be >= 1")

    # ------------------------------------------------------------------------------------------ data
    def _draw(self, m: int, gen: torch.Generator, p_repeat: float | None = None) -> tuple[Tensor, Tensor, Tensor]:
        fn = tiled_sequences if self.layout == "tiled" else repeat_sequences
        return fn(m, self.seq_len, self.vocab, self.rep_min, self.rep_max,
                  self.p_repeat if p_repeat is None else p_repeat, gen)

    def sample_batch(self, batch_size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        x, _, _ = self._draw(batch_size, gen)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    @cached_property
    def _splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        out = {}
        for name in ("train", "test"):
            x, _, _ = self._draw(self.eval_size, torch_generator(self.data_seed, f"induction/{name}"))
            out[name] = (x[:, :-1].contiguous(), x[:, 1:].contiguous())
        return out

    def eval_splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        return dict(self._splits)

    @cached_property
    def _probe(self) -> dict[str, Tensor]:
        """The probe set: every sequence has a repeat (``p_repeat`` = 1), whatever the training distribution."""
        x, rep, src = self._draw(self.probe_size, torch_generator(self.data_seed, "induction/probe"), 1.0)
        inp = x[:, :-1].contiguous()
        T = inp.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        prev_tok = torch.cat([torch.full((inp.shape[0], 1), -1, dtype=inp.dtype), inp[:, :-1]], dim=1)
        ind_keys = (prev_tok[:, None, :] == inp[:, :, None]) & causal[None]  # key j: x[j−1] = x[t], j ≤ t
        dup_keys = (inp[:, None, :] == inp[:, :, None]) & causal.tril(-1)[None]  # key j < t: x[j] = x[t]
        return {"x": inp, "y": x[:, 1:].contiguous(), "rep": rep, "src": src, "ind_keys": ind_keys,
                "dup_keys": dup_keys}

    # ------------------------------------------------------------------------------------------ model / loss
    def build_model(self) -> nn.Module:
        torch.manual_seed(self.seed)
        c = self.cfg
        return TinyGPT(self.vocab, int(c["d_model"]), self.n_layer, self.n_head, max_len=self.seq_len - 1,
                       attn_only=bool(c["attn_only"]), mlp_ratio=int(c["mlp_ratio"]))

    def loss(self, out: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))

    def accuracy(self, out: Tensor, y: Tensor) -> float | None:
        return float((out.argmax(dim=-1) == y).float().mean().item())

    def chance_accuracy(self) -> float | None:
        return 1.0 / self.vocab

    def ntk_output(self, out: Tensor, y: Tensor) -> Tensor:
        return logit_margin(out, y).mean(dim=-1)

    # ------------------------------------------------------------------------------------------ observation
    def ground_truth(self) -> list[dict] | None:
        return None

    def default_ri_order(self) -> int:
        return 3

    def ri_variables(self, model: nn.Module, n: int, gen: torch.Generator, *,
                     frozen: dict | None = None) -> VariableSet:
        raise ValueError("induction: RI scans are not supported; use ri.mode = 'patterns' (head-ablation patterns)")

    def _token_nll(self, model: nn.Module) -> tuple[Tensor, Tensor]:
        pr = self._probe
        with torch.no_grad():
            logits = model(pr["x"].to(model_device(model))).to("cpu", torch.float64)
        nll = -logits.log_softmax(-1).gather(-1, pr["y"][..., None])[..., 0]
        return nll, pr["rep"]

    def extra_metrics(self, model: nn.Module) -> dict[str, float]:
        nll, rep = self._token_nll(model)
        other = nll[~rep]
        out = {"loss_rep": float(nll[rep].mean()), "loss_other": float(other.mean()) if other.numel() else math.nan}
        out["icl_score"] = out["loss_other"] - out["loss_rep"]
        pr = self._probe
        maps = model.attention_maps(pr["x"].to(model_device(model)))
        T = pr["x"].shape[1]
        ind_keys, dup_keys = pr["ind_keys"].to(torch.float64), pr["dup_keys"].to(torch.float64)
        for layer, att in enumerate(maps):
            att = att.to("cpu", torch.float64)
            for h in range(att.shape[1]):
                a = att[:, h]
                out[f"prev_l{layer}h{h}"] = float(a[:, torch.arange(1, T), torch.arange(0, T - 1)].mean())
                out[f"ind_l{layer}h{h}"] = float((a * ind_keys).sum(-1)[rep].mean())
                out[f"dup_l{layer}h{h}"] = float((a * dup_keys).sum(-1)[rep].mean())
        return out

    # ------------------------------------------------------------------------------------------ head ablation
    def pattern_dim(self) -> int:
        H = self.n_layer * self.n_head
        if not 1 <= H <= MAX_PATTERN_INPUTS:
            raise ValueError(f"pattern observation needs 1 <= n_layer * n_head <= {MAX_PATTERN_INPUTS}, got {H}")
        return H

    def pattern_input_names(self) -> list[str]:
        return [f"l{layer}h{h}" for layer in range(self.n_layer) for h in range(self.n_head)]

    def pattern_truth(self) -> np.ndarray | None:
        return None

    def pattern_pmfs(self) -> dict[str, np.ndarray]:
        U = uniform_pmf(self.pattern_dim())
        return {"D": U, "U": U.copy(), "S": U.copy()}

    def pattern_probabilities(self, model: nn.Module) -> np.ndarray:
        """p(c): mean probability of the correct repeated token under head-ablation pattern c, in code order."""
        pr = self._probe
        dev = model_device(model)
        x = pr["x"].to(dev)
        rep = pr["rep"].to(dev)
        y = pr["y"].to(dev)[rep]
        was = model.training
        model.eval()
        out = np.empty(1 << self.pattern_dim())
        try:
            for code, keep in enumerate(pattern_bits(self.pattern_dim())):
                with model.head_mask(keep), torch.no_grad():
                    logits = model(x)[rep].to(torch.float64)  # only the predictable repeated tokens
                    lp = logits.log_softmax(-1).gather(-1, y[:, None])[:, 0]
                out[code] = float(lp.exp().mean())
        finally:
            model.train(was)
        return out

    def pattern_logodds(self, model: nn.Module) -> np.ndarray:
        p = np.clip(self.pattern_probabilities(model), P_CLIP, 1.0 - P_CLIP)
        return np.log(p) - np.log1p(-p)
