"""In-context Markov chains (Edelman et al. 2024, "The evolution of statistical induction heads").

Problem
-------
Every sequence has its own random transition matrix over ``vocab`` tokens: each row is drawn from
``Dirichlet(alpha · 1)``. The first token comes from the chain's stationary distribution (``start =
"stationary"``) or uniformly (``"uniform"``); then ``seq_len − 1`` transitions are sampled. A causal
:class:`~tycho.models.TinyGPT` reads tokens ``0..T−2`` and is trained with the mean next-token cross-entropy over
positions ``1..T−1``. Transformers learn this in stages: first the in-context *unigram* statistics, then the
*bigram* (transition) statistics through an induction-head-like circuit, with plateaus in between.

Evaluation
----------
"train" and "test" are fixed sets of ``eval_size`` sequences from two different streams of ``data_seed`` (training
batches are fresh sequences from the training generator, so "train" is a sample of the training distribution).
``accuracy`` is next-token argmax accuracy over all predicted positions; ``chance_accuracy`` is ``1 / vocab``.

Strategies and extra light-curve metrics
----------------------------------------
At position t (having seen ``x_0..x_t``, predicting ``x_{t+1}``):

* unigram strategy: ``(n_v + UNIGRAM_SMOOTHING) / (t + 1 + vocab · UNIGRAM_SMOOTHING)`` with ``n_v`` the count of
  token v in ``x_0..x_t`` (add-1/2, Krichevsky–Trofimov);
* bigram strategy: ``(m_v + alpha) / (Σ_v m_v + vocab · alpha)`` with ``m_v`` the number of transitions
  ``x_s = x_t → x_{s+1} = v`` with ``s + 1 ≤ t``. Add-``alpha`` is the posterior predictive of the transition row
  under its Dirichlet(alpha) prior, i.e. Bayes-optimal for this source up to the information in the start token.

``kl_to_unigram`` / ``kl_to_bigram`` = mean ``KL(strategy ‖ model)`` (nats) over the test sequences and positions
``t ≥ MIN_CONTEXT``.

RI variables
------------
For each probe sample a fresh sequence and a uniform random position ``t ∈ [MIN_CONTEXT, T−2]``: ``x_t`` (current
token), ``u_t`` (argmax of the in-context unigram counts), ``b_t`` (argmax of the successor counts of ``x_t``,
falling back to ``u_t`` when ``x_t`` has no earlier successor), all with ``vocab`` states and ties broken uniformly
at random, and ``xhat`` (soft: the model's predictive distribution at t). Scans require ``xhat``. Ground truth:
``{u_t, xhat}`` (stage 1, unigram) and ``{b_t, xhat}`` (stage 2, bigram).
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tycho.models.transformer import TinyGPT
from tycho.ri.variables import VariableSet
from tycho.tasks.base import Task, logit_margin, model_device, numpy_rng, split_seed_from

__all__ = ["MIN_CONTEXT", "STARTS", "UNIGRAM_SMOOTHING", "ICLMarkov", "sample_chains", "sample_given",
           "strategy_predictions", "strategy_variables"]

#: First position (0-based index of the current token) whose prediction enters the strategy KLs and RI probes.
MIN_CONTEXT = 8
#: Add-constant of the in-context unigram strategy (Krichevsky–Trofimov).
UNIGRAM_SMOOTHING = 0.5
#: Distributions of the first token.
STARTS = ("stationary", "uniform")
#: Sequences per forward pass when probing many sequences (bounds memory).
_CHUNK = 2048


def _stationary(trans: np.ndarray) -> np.ndarray:
    """Stationary distributions ``(m, V)`` of row-stochastic matrices ``(m, V, V)`` (π T = π, Σπ = 1)."""
    m, V, _ = trans.shape
    A = np.transpose(trans, (0, 2, 1)) - np.eye(V)[None]
    A[:, -1, :] = 1.0  # replace one balance equation by the normalisation
    b = np.zeros((m, V))
    b[:, -1] = 1.0
    pi = np.linalg.solve(A, b[..., None])[..., 0]
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum(axis=1, keepdims=True)


def sample_given(trans: np.ndarray, first: np.ndarray, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Sequences ``(m, seq_len)`` (int64) of the chains with transition matrices ``trans (m, V, V)`` and first-token
    distributions ``first (m, V)`` (inverse-CDF sampling)."""
    m, vocab = first.shape
    cdf = np.cumsum(trans, axis=2)
    u = rng.random((m, seq_len))
    seq = np.empty((m, seq_len), dtype=np.int64)
    seq[:, 0] = np.minimum((u[:, :1] > np.cumsum(first, axis=1)).sum(axis=1), vocab - 1)
    rows = np.arange(m)
    for t in range(1, seq_len):
        seq[:, t] = np.minimum((u[:, t, None] > cdf[rows, seq[:, t - 1]]).sum(axis=1), vocab - 1)
    return seq


def sample_chains(m: int, seq_len: int, vocab: int, alpha: float, start: str,
                  rng: np.random.Generator) -> np.ndarray:
    """``m`` sequences ``(m, seq_len)`` (int64), each from its own Dirichlet(alpha) transition matrix."""
    trans = rng.dirichlet(np.full(vocab, alpha), size=(m, vocab))  # (m, V, V), rows sum to 1
    first = _stationary(trans) if start == "stationary" else np.full((m, vocab), 1.0 / vocab)
    return sample_given(trans, first, seq_len, rng)


def _counts(seq: np.ndarray, vocab: int) -> tuple[np.ndarray, np.ndarray]:
    """In-context counts for every position t of sequences ``(m, L)``.

    Returns ``uni (m, L, V)`` (token counts in ``x_0..x_t``) and ``succ (m, L, V)`` (successor counts of the
    current token ``x_t`` over transitions ``s → s+1`` with ``s + 1 ≤ t``).
    """
    m, L = seq.shape
    onehot = np.eye(vocab)[seq]  # (m, L, V)
    uni = np.cumsum(onehot, axis=1)
    pair = onehot[:, :-1, :, None] * onehot[:, 1:, None, :]  # transition s → s+1, completed at position s + 1
    trans = np.zeros((m, L, vocab, vocab))
    trans[:, 1:] = np.cumsum(pair, axis=1)
    succ = np.take_along_axis(trans, seq[:, :, None, None].repeat(vocab, axis=3), axis=2)[:, :, 0, :]
    return uni, succ


def strategy_predictions(seq: np.ndarray, vocab: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Unigram and bigram strategy distributions ``(m, L, V)`` at every position of ``seq`` (module docstring)."""
    uni, succ = _counts(seq, vocab)
    unigram = (uni + UNIGRAM_SMOOTHING) / (uni.sum(axis=2, keepdims=True) + vocab * UNIGRAM_SMOOTHING)
    bigram = (succ + alpha) / (succ.sum(axis=2, keepdims=True) + vocab * alpha)
    return unigram, bigram


def strategy_variables(seq: np.ndarray, pos: np.ndarray, vocab: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x_t, u_t, b_t)`` at position ``pos[i]`` of sequence ``seq[i]`` (module docstring; ties broken with
    ``rng``)."""
    n = seq.shape[0]
    uni, succ = _counts(seq, vocab)
    rows = np.arange(n)
    jitter = rng.random((n, 2, vocab)) * 0.5  # < 1: reorders equal integer counts only
    x_t = seq[rows, pos]
    u_t = np.argmax(uni[rows, pos] + jitter[:, 0], axis=1)
    s = succ[rows, pos]
    b_t = np.where(s.sum(axis=1) > 0, np.argmax(s + jitter[:, 1], axis=1), u_t)
    return x_t, u_t, b_t


def _kl_rows(p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """``KL(p ‖ q)`` along the last axis (p > 0 everywhere for smoothed strategies)."""
    return np.sum(p * (np.log(p) - log_q), axis=-1)


class ICLMarkov(Task):
    """In-context learning of random Markov chains by a small transformer (see the module docstring)."""

    name = "icl_markov"

    @classmethod
    def defaults(cls) -> dict:
        return {"vocab": 3, "seq_len": 48, "alpha": 1.0, "d_model": 32, "n_layer": 2, "n_head": 1,
                "attn_only": True, "mlp_ratio": 4, "start": "stationary", "data_seed": 0, "eval_size": 512}

    def __init__(self, cfg: dict | None, seed: int) -> None:
        super().__init__(cfg, seed)
        c = self.cfg
        self.vocab, self.seq_len, self.alpha = int(c["vocab"]), int(c["seq_len"]), float(c["alpha"])
        self.start, self.data_seed, self.eval_size = str(c["start"]), int(c["data_seed"]), int(c["eval_size"])
        if self.vocab < 2:
            raise ValueError(f"vocab must be >= 2, got {self.vocab}")
        if self.seq_len < MIN_CONTEXT + 2:
            raise ValueError(f"seq_len must be >= {MIN_CONTEXT + 2}, got {self.seq_len}")
        if not self.alpha > 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}, got {self.start!r}")
        if self.eval_size < 1:
            raise ValueError(f"eval_size must be >= 1, got {self.eval_size}")

    # ------------------------------------------------------------------------------------------ data
    def _sequences(self, m: int, rng: np.random.Generator) -> np.ndarray:
        return sample_chains(m, self.seq_len, self.vocab, self.alpha, self.start, rng)

    @staticmethod
    def _xy(seq: np.ndarray) -> tuple[Tensor, Tensor]:
        s = torch.as_tensor(seq, dtype=torch.long)
        return s[:, :-1].contiguous(), s[:, 1:].contiguous()

    def sample_batch(self, batch_size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        return self._xy(self._sequences(batch_size, np.random.default_rng(split_seed_from(gen))))

    @cached_property
    def _eval_seqs(self) -> dict[str, np.ndarray]:
        return {name: self._sequences(self.eval_size, numpy_rng(self.data_seed, f"icl_markov/{name}"))
                for name in ("train", "test")}

    @cached_property
    def _splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        return {name: self._xy(seq) for name, seq in self._eval_seqs.items()}

    def eval_splits(self) -> dict[str, tuple[Tensor, Tensor]]:
        return dict(self._splits)

    @cached_property
    def _test_strategies(self) -> tuple[np.ndarray, np.ndarray]:
        """Unigram / bigram strategy distributions of the test inputs ``(m, T−1, V)``."""
        return strategy_predictions(self._eval_seqs["test"][:, :-1], self.vocab, self.alpha)

    # ------------------------------------------------------------------------------------------ model / loss
    def build_model(self) -> nn.Module:
        torch.manual_seed(self.seed)
        c = self.cfg
        return TinyGPT(self.vocab, int(c["d_model"]), int(c["n_layer"]), int(c["n_head"]), max_len=self.seq_len - 1,
                       attn_only=bool(c["attn_only"]), mlp_ratio=int(c["mlp_ratio"]))

    def loss(self, out: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))

    def accuracy(self, out: Tensor, y: Tensor) -> float | None:
        return float((out.argmax(dim=-1) == y).float().mean().item())

    def chance_accuracy(self) -> float | None:
        return 1.0 / self.vocab

    def ntk_output(self, out: Tensor, y: Tensor) -> Tensor:
        """Mean true-token logit margin over the positions ``t ≥ MIN_CONTEXT`` of each sequence."""
        return logit_margin(out, y)[..., MIN_CONTEXT:].mean(dim=-1)

    def _log_probs(self, model: nn.Module, x: Tensor) -> np.ndarray:
        """Model log-probabilities ``(m, T−1, V)`` (float64), computed in chunks."""
        dev = model_device(model)
        out = []
        with torch.no_grad():
            for lo in range(0, x.shape[0], _CHUNK):
                logits = model(x[lo:lo + _CHUNK].to(dev))
                out.append(torch.log_softmax(logits.to("cpu", torch.float64), dim=-1).numpy())
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------------------------------ observation
    def ground_truth(self) -> list[dict] | None:
        return [{"subset": ["u_t", "xhat"], "order": 2, "stage": 1},
                {"subset": ["b_t", "xhat"], "order": 2, "stage": 2}]

    def default_ri_order(self) -> int:
        return 3

    def ri_variables(self, model: nn.Module, n: int, gen: torch.Generator, *,
                     frozen: dict | None = None) -> VariableSet:
        n = int(n)
        rng = np.random.default_rng(split_seed_from(gen))
        seq = self._sequences(n, rng)[:, :-1]  # model inputs: tokens 0..T−2
        pos = rng.integers(MIN_CONTEXT, seq.shape[1], size=n)
        x_t, u_t, b_t = strategy_variables(seq, pos, self.vocab, rng)
        probs = np.exp(self._log_probs(model, torch.as_tensor(seq))[np.arange(n), pos])
        probs /= probs.sum(axis=1, keepdims=True)
        X = np.stack([x_t, u_t, b_t, probs.argmax(axis=1)], axis=1).astype(np.int64)
        return VariableSet(X=X, cards=[self.vocab] * 4, names=["x_t", "u_t", "b_t", "xhat"], soft={3: probs},
                           must_include=[3], ground_truth=self.ground_truth(),
                           meta={"task": self.name, "probe": "fresh_sequences", "min_context": MIN_CONTEXT})

    def extra_metrics(self, model: nn.Module) -> dict[str, float]:
        x, _ = self._splits["test"]
        log_q = self._log_probs(model, x)[:, MIN_CONTEXT:]
        unigram, bigram = self._test_strategies
        return {"kl_to_unigram": float(_kl_rows(unigram[:, MIN_CONTEXT:], log_q).mean()),
                "kl_to_bigram": float(_kl_rows(bigram[:, MIN_CONTEXT:], log_q).mean())}
