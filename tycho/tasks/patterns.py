"""Exact observation on every input pattern (sky v2 Part A, ``docs/results/v2a_preregistration.md``).

Tasks with a small binary input space (the Markov language's 6-token window, parity on 8 bits) can be observed on
*every* input pattern. A pattern is an integer **code** ``c = Σ_j b_j 2^j`` with ``b_j ∈ {0, 1}`` the bit of input j
(code 1 = token 1 = spin +1, as for the RI variables). With the network's log-odds on all patterns and the exact
pmfs of the input distributions, every measure of the pre-registration is an exact population value.

This module holds the task-independent pieces:

* :func:`pattern_bits` — the ``(2^d, d)`` bit matrix in code order;
* :func:`chain_pmf` — a stationary binary Markov chain over the d positions with uniform marginals and
  ``P(b_{j+1} = b_j) = (1 + ρ)/2`` (spin correlation ρ^|i−j|);
* :func:`markov_window_pmf` — the exact stationary distribution of the window of an order-W binary Markov source;
* :func:`sample_codes` — i.i.d. draws from a pmf with a torch generator (deterministic);
* :func:`binary_losses` — exact expected losses of a binary predictor under a pmf.

A task that supports it implements :class:`PatternTask` (see :class:`tycho.tasks.markov_lang.MarkovLanguage` and
:class:`tycho.tasks.sparse_parity.SparseParity`).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

__all__ = ["MAX_PATTERN_INPUTS", "PatternTask", "binary_losses", "chain_pmf", "float64_copy", "markov_window_pmf",
           "pattern_bits", "sample_codes", "uniform_pmf"]

#: Largest number of binary inputs observed on all patterns (2^12 = 4,096 patterns).
MAX_PATTERN_INPUTS = 12


def pattern_bits(d: int) -> np.ndarray:
    """All ``2^d`` patterns as an int64 ``(2^d, d)`` matrix; row c, column j = bit j of c."""
    if not 1 <= d <= MAX_PATTERN_INPUTS:
        raise ValueError(f"pattern observation needs 1 <= d <= {MAX_PATTERN_INPUTS}, got {d}")
    codes = np.arange(1 << d, dtype=np.int64)
    return (codes[:, None] >> np.arange(d, dtype=np.int64)[None, :]) & 1


def uniform_pmf(d: int) -> np.ndarray:
    """The uniform pmf over the ``2^d`` codes."""
    return np.full(1 << d, 1.0 / (1 << d))


def chain_pmf(d: int, rho: float) -> np.ndarray:
    """Stationary binary Markov chain over positions 0 … d−1: uniform marginals, ``P(b_{j+1} = b_j) = (1 + ρ)/2``."""
    rho = float(rho)
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [-1, 1], got {rho}")
    bits = pattern_bits(d)
    same = (bits[:, 1:] == bits[:, :-1])
    p_same, p_diff = (1.0 + rho) / 2.0, (1.0 - rho) / 2.0
    pmf = 0.5 * np.prod(np.where(same, p_same, p_diff), axis=1)
    return pmf / pmf.sum()


def markov_window_pmf(p_table: np.ndarray, window: int) -> np.ndarray:
    """Exact stationary pmf of the window code ``c = Σ_ℓ x_{t−ℓ} 2^{ℓ−1}`` of an order-W binary Markov source.

    ``p_table[c] = P(x_t = 1 | window c)``; the next window is ``((c << 1) | x_t) & (2^W − 1)``. The stationary
    distribution solves ``π T = π, Σ π = 1`` (a direct linear solve, so slowly mixing chains are exact too). Raises
    ValueError when the chain has no unique stationary distribution (an all-0 or all-1 conditional somewhere that
    traps it).
    """
    p = np.asarray(p_table, dtype=np.float64)
    n = 1 << window
    if p.shape != (n,):
        raise ValueError(f"p_table must have {n} entries, got {p.shape}")
    mask = n - 1
    codes = np.arange(n)
    T = np.zeros((n, n))
    T[codes, (codes << 1) & mask] += 1.0 - p
    T[codes, ((codes << 1) | 1) & mask] += p
    A = T.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the chain has no unique stationary distribution") from exc
    if np.any(pi < -1e-12) or not np.allclose(pi @ T, pi, atol=1e-12):
        raise ValueError("the chain has no unique stationary distribution")
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def sample_codes(pmf: np.ndarray, n: int, gen: torch.Generator) -> np.ndarray:
    """``n`` codes drawn i.i.d. from ``pmf`` with ``gen`` (inverse-CDF sampling on float64 uniforms)."""
    pmf = np.asarray(pmf, dtype=np.float64)
    cdf = np.cumsum(pmf)
    cdf /= cdf[-1]
    u = torch.rand(int(n), generator=gen, dtype=torch.float64).numpy()
    return np.minimum(np.searchsorted(cdf, u, side="right"), pmf.size - 1).astype(np.int64)


def binary_losses(logodds: np.ndarray, p_true: np.ndarray, pmf: np.ndarray) -> tuple[float, float]:
    """Exact expected losses of a binary predictor with log-odds ``logodds`` (of outcome 1) against the true
    probability ``p_true`` of outcome 1, under ``pmf``:

    * ``kl`` = Σ_x pmf(x) KL(p*(· | x) ‖ q(· | x)), the excess cross-entropy over the Bayes predictor (for a
      deterministic target, p* ∈ {0, 1}, this is the log-loss of the true label);
    * ``sq`` = Σ_x pmf(x) (q(1 | x) − p*(1 | x))² (for a deterministic target, the Brier score of the true label).
    """
    z = np.asarray(logodds, dtype=np.float64)
    p = np.asarray(p_true, dtype=np.float64)
    w = np.asarray(pmf, dtype=np.float64)
    log_q1 = -np.logaddexp(0.0, -z)
    log_q0 = -np.logaddexp(0.0, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - log_q1), 0.0)
        t0 = np.where(p < 1, (1 - p) * (np.log(np.where(p < 1, 1 - p, 1.0)) - log_q0), 0.0)
    q1 = 1.0 / (1.0 + np.exp(-z))
    return float(np.sum(w * (t1 + t0))), float(np.sum(w * (q1 - p) ** 2))


def float64_copy(model: nn.Module) -> nn.Module:
    """A float64 copy of ``model`` in eval mode (the same weights at higher precision, for exact log-odds)."""
    m = copy.deepcopy(model).to(dtype=torch.float64)
    m.eval()
    return m


class PatternTask:
    """Mixin interface of tasks that can be observed on every input pattern (all methods return numpy arrays).

    * ``pattern_dim()`` — number of binary inputs d (``2^d`` patterns, codes as in this module);
    * ``pattern_logodds(model)`` — float64 log-odds of output 1 on every pattern, in code order, computed with a
      float64 copy of the model;
    * ``pattern_truth()`` — the true P(output 1 | pattern), or None when there is none (then no exact losses);
    * ``pattern_pmfs()`` — ``{"D": training distribution, "U": uniform, "S": shift distribution}``;
    * ``pattern_input_names()`` — the RI variable names of the d inputs, in bit order.
    """

    def pattern_dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def pattern_logodds(self, model: nn.Module) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def pattern_truth(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def pattern_pmfs(self) -> dict[str, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def pattern_input_names(self) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def exact_metrics(self, model: nn.Module) -> dict[str, float]:
        """Exact losses under D, U and S (KL from truth and squared error), for the light curve."""
        z = self.pattern_logodds(model)
        p = self.pattern_truth()
        out = {}
        for name, pmf in self.pattern_pmfs().items():
            kl, sq = binary_losses(z, p, pmf)
            out[f"loss{name}"] = kl
            out[f"sq{name}"] = sq
        return out
