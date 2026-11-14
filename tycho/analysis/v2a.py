"""Offline analysis of sky v2 Part A (``docs/results/v2a_preregistration.md``, binding; § numbers refer to it).

What it computes
----------------
* **Stage 1, per object** (:func:`stage1_object`, cached to npz): for every set A of 1 to 3 inputs (41 Markov sets,
  92 parity sets) and every stored checkpoint, the θ value and both η values (bounded and Gaussian, from one weighted
  least-squares fit) of ``tycho.ri.cells.projection_information`` at the four references of §1:

  * M_X: the marginal table of A's inputs, each cell weighted by pmf_X summed over the other inputs
    (counts = Σ pmf, sums = Σ pmf·m), positions = all of A;
  * C_X: the full table (counts = pmf, sums = pmf·m), positions = A;

  with X ∈ {D, U} and m = tanh(log-odds/2) clipped to [−1 + 1e-12, 1 − 1e-12] (§1 "Numerics"). Also: the exact
  losses (KL and squared error) under D, U and S, the accuracies, the sampled θ_M_D and θ_C_D from the stored probe
  counts, the saturation and lower-order energies of §5 Q2, the inputs of IC-2(a), IC-3 and IC-4, and Q5.
* **Stage 2** (:func:`analyse`): IC-1 to IC-5, T1 and T2 with the identifiability rule, the joint-seed cluster
  bootstrap and its p-value, Holm over the 12 comparisons per η value, the three win conditions, the labels, the
  decision table, the secondary statistics, and Q1 to Q5.

Numerics. The fits reproduce ``projection_information`` (tests: ``tests/test_v2a_analysis.py``) but are organised
for speed: all checkpoints of a set are one batch; the features are columns of the Walsh basis, so the logits,
gradients and Hessians come from fast Walsh-Hadamard transforms (the Hessian of a column set is a gather of the
transformed weights, (Hᵀ diag(v) H)[B, C] = (Hᵀ v)[B ⊕ C]); the damped Newton of ``projection_information`` starts
from a warm start when that start has the lower loss; the η values come from one weighted least-squares projection
(ΔRSS = (r_m · r_A)² / |r_A|², with r the weighted residuals on the smaller model's features); the saturated
("hi") model at the marginal references has its minimum in closed form.

The choices the pre-registration leaves open are listed in :data:`DECISIONS` (and appended to this docstring).
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tycho.ri.cells import eta_information
from tycho.tasks.patterns import binary_losses, pattern_bits

__all__ = ["DECISIONS", "REFS", "VALUES", "all_sets", "analyse", "boot_pvalue", "clipped_mean",
           "conditional_measures", "holm", "marginal_measures", "residual_variance", "spearman", "stage1_object",
           "walsh_coefficients", "window_start"]

DECISIONS: tuple[str, ...] = (
    # ---------------------------------------------------------------- measures (§1)
    ('Sets: all sets of 1-3 inputs, by size then lexicographically (itertools.combinations); Markov input j is the'
     ' token at lag j+1, parity input j is bit j.'),
    ('Cell weights are normalised to sum 1 within each table (as projection_information does). Cells of zero '
     'weight (empty probe cells) are kept with weight 0, which is equivalent to projection_information dropping '
     'them.'),
    ("Logistic fits replicate projection_information's damped Newton (gradient tolerance 1e-12, ridge 1e-12, at "
     'most 100 iterations, the same backtracking rule) but start from a warm start when it has a lower loss than '
     "beta = 0: the Walsh coefficients of the clipped logit restricted to the model's columns (hi model, and the "
     'lo model at the marginal references), or the hi solution without chi_A (lo model at the conditional '
     'references). A warm-started fit that has not converged (gradient test or a full step improving by < 1e-16) '
     'within 50 iterations is redone from beta = 0 with the 100-iteration limit, as projection_information does '
     '(on pilot data a warm start of a lo model stalled where beta = 0 converges). When the backtracking finds no '
     'decrease down to a <= 1e-8 the fit stops at the current point (projection_information takes that tiny step '
     'and continues). At the marginal references the hi model is saturated and its minimum, sum_x w H(t_x), is '
     'computed in closed form.'),
    ("Sampled theta_M_D and theta_C_D: the runner stores per-pattern probe counts (probe seed (seed, 'ri', step)),"
     ' not sampled values. Stage 1 computes them with the exact-value code using the probe counts as cell weights.'
     ' IC-2(a) recomputes them independently for 5 % of (run, checkpoint) pairs from the probe rows themselves '
     '(rows rebuilt from the counts) with cell_stats + projection_information, and requires agreement to 1e-9. The'
     ' 5 % are ceil(0.05 * n_checkpoints) checkpoints per run (3 of 60), drawn with a generator seeded from '
     '(20261114, object id). IC-2(a) also regenerates the counts from the probe seed and requires them to equal '
     'the stored counts (an extra check, not in §3).'),
    ('Losses are tycho.tasks.patterns.binary_losses of the stored float64 log-odds (KL from truth and squared '
     'error); for parity the KL equals the log-loss of the true label and the squared error its Brier score (§2).'),
    # ---------------------------------------------------------------- regions and instrument checks (§2, §3)
    ("Regions are those of §2: 'markov-v2' and 'parity-corr' (k = 2 and 3 together; configs/sky_v2a.yaml splits it"
     " into parity-corr-k2 and parity-corr-k3). They group IC-1's sample and IC-2(b)'s pooling."),
    ('IC-1: ceil(5 %) of the done objects of each §2 region (at least one per region), drawn with seed 20261114; '
     "each is replayed with tycho.train.run_object (spec rebuilt from the manifest and the region's train settings"
     " in configs/sky_v2a.yaml, its object id checked; RIConfig from the region's ri with enabled=True; torch with"
     ' one thread, as survey workers) and compared bit for bit on every light-curve key except wall_s, and on the '
     'pattern arrays steps, logodds and counts.'),
    ('IC-2(b) pools all done runs of a region (IC-5 failures included) and all checkpoints, step 0 included. The '
     'share within 0.002 nats is reported per configuration (Markov: couplings x context; parity: k x rho).'),
    ('IC-3: for each chain run, the corpus is regenerated with make_task(...)._corpus; TV = 1/2 sum |pmf_D - '
     'window frequency| over its training windows; the transition check uses the binomial SE of p*: |k/n - p*| <= '
     '5 sqrt(p*(1 - p*)/n) for every window seen >= 1,000 times. The exact pmf and p* are recomputed from the '
     'couplings and compared with the stored ones.'),
    ('IC-4 compares M_D with M_U and C_D with C_U for theta and both eta values at every set and checkpoint of the'
     ' iid Markov runs and the parity rho = 0 runs; the local_ri check runs at every set and checkpoint of every '
     'run, on the table P(x_A, yhat) = CellStats(counts = D marginal, sums = D marginal of pmf*m).table().'),
    ("IC-5: the parity prediction is output 1 iff log-odds > 0 (argmax, ties to class 0); 'Markov runs' and "
     "'parity runs' in the pass-rate rule are all done runs of each kind."),
    # ---------------------------------------------------------------- tests (§4)
    ("Support (for 'spurious'): Markov, the union of the lags of every planted coupling with beta != 0 (4-lag "
     "couplings included), as inputs lag - 1; parity, the planted set task_cfg['support']."),
    ("Windows use the KL loss under D for the primary and the squared-error outcomes alike (T1: the run's own "
     "loss_D; T2: the 'with' network's KL loss under D_with)."),
    ('Spearman: average ranks, ties are exact floating-point equality (no rounding). rho = 0 when the predictor is'
     ' constant (§4), and also when the outcome is constant or a configuration has fewer than 2 points (in a '
     "resample, possibly none); none of these occurred unless reported under 'degenerate'."),
    ('Bootstrap: for each family (T1 Markov, T1 parity, T2) a fresh numpy default_rng(20261114) draws a B x '
     'n_seeds matrix of seed indices; the same resamples serve the four references and both eta values of the '
     "family. Seed labels are the sorted distinct seeds of the family's runs in the manifest. A resampled seed "
     'contributes all window points of its (IC-5-passing) run in every configuration, or of both twins of its '
     'pair; resampled duplicates enter the Spearman as tied copies (weighted mid-ranks, weighted Pearson of '
     'ranks). Bootstrap SD is the sample SD (ddof = 1) of Delta*, the CI the 2.5 and 97.5 % percentiles.'),
    ('Configurations dropped from rho-bar (fewer than --min-seeds seeds passing IC-5 with at least one window '
     'point) are dropped from every resample too. A comparison with no configuration left has Delta = NaN, p = 1 '
     "in Holm (the family stays at 12), and is labelled 'not computable'."),
    ('Labels per link L: win = conditions 1, 2 and 3 for L; value-dependent = exactly one of conditions 1 and 2 '
     'holds for L; outcome-dependent = 1 and 2 hold for L but not 3. Delta = 0 favours neither link.'),
    ('Secondary statistics: the pooled Spearman uses the points of the configurations that enter rho-bar; the '
     'seed-level statistic is the unweighted mean over all (configuration, checkpoint) cells with >= 3 seeds whose'
     ' window contains the checkpoint; the secondary T1 predictor (all spurious sets, singletons included) and the'
     ' squared-error outcome are reported as point estimates.'),
    ('T2: a twin pair needs both objects done and both passing IC-5; its outcome uses binary losses with p_true = '
     "sigmoid(log-odds of the 'with' network). The identifiability residual variance is computed from the exact "
     "stationary pmf of the 'with' language (markov_window_pmf of its conditional table; uniform for iid), which "
     'is also checked against the stored pmf_D.'),
    # ---------------------------------------------------------------- questions (§5)
    ('Q1: last checkpoint; the predictions are scored on the pooled runs (chain Markov and parity rho in {0.5, '
     "0.8}, passing IC-5) and also reported per configs/sky_v2a.yaml region; 'smaller in total' = sum over runs of"
     ' the C_D count < that of the M_D count; eta presence uses the bounded value.'),
    ('Q2: all runs passing IC-5 (both regions, both contexts, every rho), sets with |A| >= 2, every checkpoint '
     'including step 0. Bins: saturation [0, 0.5, 0.74, 0.9, 0.99, 1], lower-order energy [0, 0.1, 0.3, 0.5, 1, '
     "inf). 'Per run' = number of C_D divergences (bounded value) / number of runs of the group, both contexts; "
     "'more' is strict (equal rates do not confirm the prediction)."),
    ("Q3: consecutive = adjacent checkpoints of the run's grid, step 0 excluded; shares are over seeds passing "
     'IC-5.'),
    ('Q4: pairs (couplings, seed) with both runs done and passing IC-5; ties (difference exactly 0) dropped; '
     'one-sided binomial sign test, p = P(Bin(n_pos + n_neg, 1/2) >= n_pos); the eta link is scored with the '
     'bounded value, the Gaussian value is reported.'),
    ('Q5 (exploratory): last checkpoint of every run passing IC-5; each link (theta, eta bounded, eta Gaussian) '
     'ranks the sets by its C_D value (ties by set order); a logistic model with an intercept and chi_A of the top'
     " K sets is fitted to the network's clipped soft output under D (weighted, soft targets) and scored by E_S "
     'KL(p_net || q) (and E_D).'),
)
__doc__ = (__doc__ or "") + "\nDECISIONS\n---------\n" + "\n".join(f"{i}. {d}" for i, d in enumerate(DECISIONS, 1))

#: Bump when stage-1 outputs change (invalidates caches).
STAGE1_VERSION = "v2a-stage1-1"
REFS = ("M_D", "M_U", "C_D", "C_U")
VALUES = ("theta", "eta_b", "eta_g")
ETA_VALUES = ("eta_b", "eta_g")
CLIP = 1e-12
RIDGE = 1e-12
PRESENCE = 1e-3
DIVERGE = 2e-3
MARGIN = 0.10
ALPHA = 0.05
BOOT_SEED = 20261114
N_BOOT = 10_000
MIN_SEEDS = 5
IDENT_MIN = 0.01
IC_TOL = 1e-9
IC2_MEDIAN = 5e-4
IC2_BAND = 2e-3
IC2A_FRACTION = 0.05
IC3_MIN_COUNT = 1000
IC3_Z = 5.0
IC5_KL = 0.01
IC5_ACC = 0.95
IC5_SHARE = {"markov_lang": 0.95, "sparse_parity": 0.90}
Q5_K = (3, 5, 8)
#: T2 twins (§4): (without, with, added lags).
TWIN_PAIRS = (("1:1.0", "1:1.0;1,2,3:1.0", (1, 2, 3)), ("1:1.0", "1:1.0;1,3:0.8", (1, 3)),
              ("1:2.0", "1:2.0;1,2,3:2.0", (1, 2, 3)), ("1:2.0", "1:2.0;1,3:1.6", (1, 3)))
#: Residual variances printed in §4 (chain twin sets), for the check of :func:`identifiability_table`.
PREREG_IDENT = {("1:1.0;1,2,3:1.0", "C_D"): 0.0062, ("1:2.0;1,2,3:2.0", "C_D"): 2e-6,
                ("1:2.0;1,3:1.6", "C_D"): 0.0028, ("1:1.0;1,3:0.8", "C_D"): 0.106,
                ("1:1.0;1,2,3:1.0", "M_D"): 0.018, ("1:2.0;1,2,3:2.0", "M_D"): 7e-6,
                ("1:1.0;1,3:0.8", "M_D"): 0.43, ("1:2.0;1,3:1.6", "M_D"): 0.016}
Q2_STRONG = ("1:2.0", "1,2,3:2.0", "1:2.0;1,3:1.6", "1:2.0;1,2,3:2.0")
Q2_WEAK = ("1:1.0", "1,3:1.0", "1,2,4:1.0", "1,2,3,5:1.0", "1:1.0;1,3:0.8", "1,2,3:1.0", "1:1.0;1,2,3:1.0")
Q2_SAT_EDGES = (0.0, 0.5, 0.74, 0.9, 0.99, 1.0 + 1e-12)
Q2_ENERGY_EDGES = (0.0, 0.1, 0.3, 0.5, 1.0, math.inf)
Q3_COUPLINGS, Q3_LAGS = "1:1.0;1,2,3:1.0", (1, 2, 3)


# ================================================================================================ Walsh basis
def all_sets(d: int, max_size: int = 3) -> list[tuple[int, ...]]:
    """All sets of 1..max_size of the d inputs (41 for d = 6, 92 for d = 8), by size then lexicographically."""
    return [A for r in range(1, max_size + 1) for A in itertools.combinations(range(d), r)]


def mask_of(A: Iterable[int]) -> int:
    return int(sum(1 << int(i) for i in A))


@functools.cache
def _sign(k: int) -> np.ndarray:
    """(−1)^|B| for every bit mask B of k bits: χ_B = sign(B) · (standard Walsh function of B)."""
    return np.where(np.bitwise_count(np.arange(1 << k, dtype=np.int64)) % 2 == 1, -1.0, 1.0)


@functools.cache
def walsh_matrix(k: int) -> np.ndarray:
    """H[c, B] = Π_{j∈B} s_j(c), s_j = 2·bit_j(c) − 1, for pattern codes c and masks B (both in code order)."""
    s = 2.0 * pattern_bits(k) - 1.0
    masks = pattern_bits(k).astype(bool)
    H = np.stack([np.prod(np.where(masks[B][None, :], s, 1.0), axis=1) for B in range(1 << k)], axis=1)
    H.setflags(write=False)
    return H


def fwht(a: np.ndarray) -> np.ndarray:
    """Unnormalised Walsh-Hadamard transform along the last axis: out[B] = Σ_c (−1)^popcount(c & B) a[c]."""
    a = np.array(a, dtype=np.float64, copy=True)
    n = a.shape[-1]
    lead = a.shape[:-1]
    h = 1
    while h < n:
        a = a.reshape(lead + (n // (2 * h), 2, h))
        x, y = a[..., 0, :], a[..., 1, :]
        s, dd = x + y, x - y
        a[..., 0, :] = s
        a[..., 1, :] = dd
        h *= 2
    return a.reshape(lead + (n,))


def walsh_coefficients(v: np.ndarray) -> np.ndarray:
    """f(B) = 2^−k Σ_c v(c) χ_B(c) along the last axis (uniform Walsh coefficients, masks in code order)."""
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[-1]
    return _sign(int(n).bit_length() - 1) * fwht(v) / n


def clipped_mean(logodds: np.ndarray) -> np.ndarray:
    """m = tanh(log-odds / 2) clipped to [−1 + 1e-12, 1 − 1e-12] (§1 "Numerics")."""
    return np.clip(np.tanh(np.asarray(logodds, dtype=np.float64) / 2.0), -1.0 + CLIP, 1.0 - CLIP)


# ================================================================================================ fits
def _logits(k: int, cols: np.ndarray, beta: np.ndarray) -> np.ndarray:
    full = np.zeros((beta.shape[0], 1 << k))
    full[:, cols] = beta * _sign(k)[cols]
    return fwht(full)


def _ce(k: int, cols: np.ndarray, beta: np.ndarray, W: np.ndarray, T: np.ndarray) -> np.ndarray:
    z = _logits(k, cols, beta)
    return np.einsum("sn,sn->s", W, np.logaddexp(0.0, z) - T * z)


def _newton(k: int, cols: np.ndarray, W: np.ndarray, T: np.ndarray, beta: np.ndarray, *, tol: float = 1e-12,
            max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched damped Newton of ``tycho.ri.cells._weighted_logistic_loss`` on the Walsh columns ``cols`` of a
    k-bit table: min_β Σ_x W_x [log(1 + e^{Fβ}) − T_x Fβ] for each row of W, T (weights summing to 1, soft
    targets).
    Returns (loss, β, iterations, converged); converged means the gradient test or a full step with an improvement
    below 1e-16 ended the fit (not the iteration limit or a failed backtracking)."""
    p = cols.size
    sign = _sign(k)
    xor = (cols[:, None] ^ cols[None, :]).ravel()
    eye = RIDGE * np.eye(p)
    beta = np.array(beta, dtype=np.float64, copy=True)
    cur = _ce(k, cols, beta, W, T)
    active = np.ones(beta.shape[0], bool)
    conv_ok = np.zeros(beta.shape[0], bool)
    iters = np.zeros(beta.shape[0], np.int64)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        b, w, t = beta[idx], W[idx], T[idx]
        sig = 0.5 * (1.0 + np.tanh(0.5 * _logits(k, cols, b)))
        g = (fwht(w * (sig - t)) * sign)[:, cols]
        iters[idx] += 1
        conv = np.max(np.abs(g), axis=1) < tol
        if conv.any():
            active[idx[conv]] = False
            conv_ok[idx[conv]] = True
            nc = ~conv
            idx, b, w, t, g, sig = idx[nc], b[nc], w[nc], t[nc], g[nc], sig[nc]
            if idx.size == 0:
                break
        V = fwht(w * sig * (1.0 - sig)) * sign
        hess = V[:, xor].reshape(-1, p, p) + eye
        step = np.linalg.solve(hess, g[..., None])[..., 0]
        c = cur[idx]
        a = np.ones(idx.size)
        new = _ce(k, cols, b - step, w, t)
        pend = ~(new <= c + 1e-15)
        gave_up = np.zeros(idx.size, bool)
        while pend.any():
            a[pend] *= 0.5
            stop = pend & (a <= 1e-8)
            gave_up |= stop
            pend &= ~stop
            pi = np.flatnonzero(pend)
            if pi.size == 0:
                break
            new[pi] = _ce(k, cols, b[pi] - a[pi, None] * step[pi], w[pi], t[pi])
            pend[pi[new[pi] <= c[pi] + 1e-15]] = False
        moved = ~gave_up
        beta[idx[moved]] = b[moved] - a[moved, None] * step[moved]
        cur[idx[moved]] = new[moved]
        done = moved & (c - new < 1e-16) & (a == 1.0)
        conv_ok[idx[done]] = True
        active[idx[gave_up | done]] = False
    return cur, beta, iters, conv_ok


def _fit(k: int, cols: np.ndarray, W: np.ndarray, T: np.ndarray, warm: np.ndarray, *,
         chunk_elems: int = 4_000_000, warm_max_iter: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum cross-entropy of the logistic model on ``cols``. Each problem starts from the warm start when it has
    a lower loss than β = 0; a warm-started fit that does not converge within ``warm_max_iter`` iterations is redone
    from β = 0 with projection_information's limit of 100 iterations (so the result is never worse than the
    library's own start). Chunked to bound the Hessian stack. Returns (loss, β, iterations)."""
    cols = np.asarray(cols, dtype=np.int64)
    S, p = warm.shape
    zero = np.zeros_like(warm)
    use = _ce(k, cols, warm, W, T) < _ce(k, cols, zero, W, T)
    size = max(1, chunk_elems // max(1, p * p))
    out_l, out_b, out_i = np.empty(S), np.empty((S, p)), np.zeros(S, np.int64)

    def run(idx, start, max_iter):
        conv = np.zeros(idx.size, bool)
        for s0 in range(0, idx.size, size):
            sl = idx[s0:s0 + size]
            out_l[sl], out_b[sl], it, conv[s0:s0 + size] = _newton(k, cols, W[sl], T[sl], start[sl],
                                                                    max_iter=max_iter)
            out_i[sl] += it
        return conv

    warm_idx = np.flatnonzero(use)
    if warm_idx.size:
        conv = run(warm_idx, warm, warm_max_iter)
        redo = warm_idx[~conv]
        if redo.size:
            run(redo, zero, 100)
    zero_idx = np.flatnonzero(~use)
    if zero_idx.size:
        run(zero_idx, zero, 100)
    return out_l, out_b, out_i


def _linear(F_lo: np.ndarray, f_a: np.ndarray, w: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, ...]:
    """One weighted least-squares projection of the cell means ``M`` (cells × checkpoints) and of χ_A on the smaller
    model's features, with shared weights ``w`` (sum 1). Returns (ΔRSS, RSS_lo, RSS_hi, residual variance of χ_A),
    RSS including the Bernoulli variance Σ w (1 − m²) as in projection_information."""
    keep = w > 0
    sw = np.sqrt(w[keep])
    Fw = F_lo[keep] * sw[:, None]
    Q, R = np.linalg.qr(Fw)
    dg = np.abs(np.diag(R))
    if dg.size and dg.min() <= 1e-10 * dg.max():  # rank-deficient (only possible with empty cells): SVD basis
        U, s, _ = np.linalg.svd(Fw, full_matrices=False)
        Q = U[:, s > s[0] * max(Fw.shape) * np.finfo(float).eps]
    Y = M[keep] * sw[:, None]
    Rm = Y - Q @ (Q.T @ Y)
    a = f_a[keep] * sw
    ra = a - Q @ (Q.T @ a)
    va = float(ra @ ra)
    base = w[keep] @ (1.0 - M[keep] ** 2)
    rss_lo = base + np.sum(Rm * Rm, axis=0)
    drss = (ra @ Rm) ** 2 / va if va > 0 else np.zeros(M.shape[1])
    return drss, rss_lo, rss_lo - drss, va


def _eta_values(drss: np.ndarray, rss_lo: np.ndarray, rss_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bounded, Gaussian) η values of projection_information from ΔRSS, RSS_lo and RSS_hi."""
    bounded = eta_information(np.sqrt(np.clip(drss, 0.0, 1.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        gauss = np.where(rss_hi > 0, np.maximum(0.0, 0.5 * np.log(rss_lo / rss_hi)), np.nan)
    return np.asarray(bounded, dtype=np.float64), gauss


def _norm_weights(w: np.ndarray, n_ck: int) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        w = np.broadcast_to(w, (n_ck, w.size))
    return w / w.sum(axis=1, keepdims=True)


def _sub_codes(A: Sequence[int], d: int) -> np.ndarray:
    """Cell of every pattern in the marginal table of A: Σ_i bit_{A[i]} 2^i."""
    bits = pattern_bits(d)
    return (bits[:, list(A)] << np.arange(len(A))[None, :]).sum(axis=1)


def marginal_tables(m: np.ndarray, w: np.ndarray, sets: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Cell weights (counts = Σ w over the other inputs) and cell means (sums / counts; 0 in empty cells) of the
    marginal tables of ``sets`` (all of one size k): arrays (checkpoints, sets, 2^k)."""
    m = np.atleast_2d(m)
    n_ck, n = m.shape
    d = n.bit_length() - 1
    k = len(sets[0])
    P = np.zeros((len(sets), n, 1 << k))
    for j, A in enumerate(sets):
        P[j, np.arange(n), _sub_codes(A, d)] = 1.0
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        cnt = np.broadcast_to(np.einsum("c,scj->sj", w, P), (n_ck, len(sets), 1 << k))
        sums = np.einsum("tc,scj->tsj", w[None, :] * m, P)
    else:
        cnt = np.einsum("tc,scj->tsj", w, P)
        sums = np.einsum("tc,scj->tsj", w * m, P)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(cnt > 0, sums / np.where(cnt > 0, cnt, 1.0), 0.0)
    return np.array(cnt), np.clip(means, -1.0 + CLIP, 1.0 - CLIP)


def marginal_measures(m: np.ndarray, weightings: Sequence[np.ndarray], sets: Sequence[Sequence[int]], *,
                      linear: Sequence[bool] | None = None) -> list[dict[str, np.ndarray]]:
    """θ (and, where ``linear``, both η values and the residual variance of χ_A) at a marginal reference, for each
    weighting: a pmf (2^d,) shared by all checkpoints, or per-checkpoint weights (checkpoints, 2^d) such as probe
    counts (θ only). ``m``: (checkpoints, 2^d) clipped means. Returns one dict per weighting with arrays
    (checkpoints, sets) and ``resid`` (sets,)."""
    m = np.atleast_2d(m)
    n_ck = m.shape[0]
    linear = [np.asarray(w).ndim == 1 for w in weightings] if linear is None else list(linear)
    out = [{"theta": np.full((n_ck, len(sets)), np.nan), "iters": np.zeros(len(sets))} for _ in weightings]
    for o, lin in zip(out, linear, strict=True):
        if lin:
            o.update(eta_b=np.full((n_ck, len(sets)), np.nan), eta_g=np.full((n_ck, len(sets)), np.nan),
                     resid=np.full(len(sets), np.nan))
    for k in sorted({len(A) for A in sets}):
        idx = [j for j, A in enumerate(sets) if len(A) == k]
        sub = [tuple(sets[j]) for j in idx]
        nc = 1 << k
        lo_cols = np.arange(nc - 1, dtype=np.int64)
        Hk = walsh_matrix(k)
        for o, w, lin in zip(out, weightings, linear, strict=True):
            cnt, mA = marginal_tables(m, w, sub)
            W = cnt / cnt.sum(axis=2, keepdims=True)
            T = (1.0 + mA) / 2.0
            L = 2.0 * np.arctanh(mA)
            loss_hi = np.sum(W * (np.logaddexp(0.0, L) - T * L), axis=2)
            beta_sat = walsh_coefficients(L)
            l_lo, _, it = _fit(k, lo_cols, W.reshape(-1, nc), T.reshape(-1, nc), beta_sat[..., lo_cols].reshape(
                -1, nc - 1))
            o["theta"][:, idx] = np.maximum(0.0, l_lo.reshape(n_ck, len(idx)) - loss_hi)
            o["iters"][idx] = it.reshape(n_ck, len(idx)).mean(axis=0)
            if lin:
                if np.asarray(w).ndim != 1:
                    raise ValueError("the η values need weights shared by all checkpoints")
                for jj, j in enumerate(idx):
                    drss, lo_, hi_, va = _linear(Hk[:, lo_cols], Hk[:, nc - 1], W[0, jj], mA[:, jj, :].T)
                    o["eta_b"][:, j], o["eta_g"][:, j] = _eta_values(drss, lo_, hi_)
                    o["resid"][j] = va
    return out


def conditional_measures(m: np.ndarray, weightings: Sequence[np.ndarray], sets: Sequence[Sequence[int]], *,
                         linear: Sequence[bool] | None = None) -> list[dict[str, np.ndarray]]:
    """As :func:`marginal_measures` at a conditional reference (the full table of all inputs, positions = A)."""
    m = np.atleast_2d(m)
    n_ck, n = m.shape
    d = n.bit_length() - 1
    linear = [np.asarray(w).ndim == 1 for w in weightings] if linear is None else list(linear)
    nw = len(weightings)
    Ws = np.concatenate([_norm_weights(w, n_ck) for w in weightings], axis=0)
    T = np.tile((1.0 + m) / 2.0, (nw, 1))
    beta_full = np.tile(walsh_coefficients(2.0 * np.arctanh(m)), (nw, 1))
    H = walsh_matrix(d)
    masks = np.arange(n, dtype=np.int64)
    out = [{"theta": np.full((n_ck, len(sets)), np.nan), "iters": np.zeros(len(sets))} for _ in weightings]
    for o, lin in zip(out, linear, strict=True):
        if lin:
            o.update(eta_b=np.full((n_ck, len(sets)), np.nan), eta_g=np.full((n_ck, len(sets)), np.nan),
                     resid=np.full(len(sets), np.nan))
    for j, A in enumerate(sets):
        a = mask_of(A)
        lo_cols = masks[(masks & a) != a]
        hi_cols = np.sort(np.append(lo_cols, a))
        lo_in_hi = np.flatnonzero(hi_cols != a)
        l_hi, b_hi, it_hi = _fit(d, hi_cols, Ws, T, beta_full[:, hi_cols])
        l_lo, _, it_lo = _fit(d, lo_cols, Ws, T, b_hi[:, lo_in_hi])
        th = np.maximum(0.0, l_lo - l_hi)
        for i, (o, w, lin) in enumerate(zip(out, weightings, linear, strict=True)):
            o["theta"][:, j] = th[i * n_ck:(i + 1) * n_ck]
            o["iters"][j] = 0.5 * (it_hi[i * n_ck:(i + 1) * n_ck].mean() + it_lo[i * n_ck:(i + 1) * n_ck].mean())
            if lin:
                if np.asarray(w).ndim != 1:
                    raise ValueError("the η values need weights shared by all checkpoints")
                drss, lo_, hi_, va = _linear(H[:, lo_cols], H[:, a], Ws[i * n_ck], m.T)
                o["eta_b"][:, j], o["eta_g"][:, j] = _eta_values(drss, lo_, hi_)
                o["resid"][j] = va
    return out


def residual_variance(pmf: np.ndarray, A: Sequence[int], ref: str) -> float:
    """§4 identifiability: the pmf-weighted residual variance of χ_A after weighted least squares on the features
    of the smaller model at reference ``ref`` ("M_*": the proper subsets of A on A's marginal table; "C_*": every
    B ⊉ A on the full table)."""
    pmf = np.asarray(pmf, dtype=np.float64)
    m0 = np.zeros((1, pmf.size))
    if ref.startswith("M"):
        return float(marginal_measures(m0, [pmf], [tuple(A)])[0]["resid"][0])
    return float(conditional_measures(m0, [pmf], [tuple(A)])[0]["resid"][0])


# ================================================================================================ stage 1
def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(p: str | os.PathLike, manifest: Path | None = None) -> Path:
    """A manifest path: absolute, or relative to the working directory, the repository, or the manifest's folder."""
    p = Path(p)
    if p.is_absolute():
        return p
    repo = Path(__file__).resolve().parents[2]
    cands = [Path.cwd() / p, repo / p]
    if manifest is not None:
        cands += [manifest.parent / p, manifest.parent / p.parent.name / p.name]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def library_sampled_theta(logodds: np.ndarray, counts: np.ndarray, sets: Sequence[Sequence[int]]) -> np.ndarray:
    """IC-2(a) reference path: θ_M_D and θ_C_D (2, sets) of one checkpoint from the probe rows themselves (rows
    rebuilt from the per-pattern counts, soft output (1 + m)/2 per row), with ``cell_stats`` and
    ``projection_information``."""
    from tycho.ri.cells import cell_stats, projection_information

    n = logodds.size
    d = n.bit_length() - 1
    p1 = (1.0 + clipped_mean(logodds)) / 2.0
    rows = np.repeat(np.arange(n), np.asarray(counts, dtype=np.int64))
    X = pattern_bits(d)[rows]
    P1 = p1[rows]
    out = np.empty((2, len(sets)))
    full = cell_stats(X, P1, [tuple(range(d))])[0]
    for j, A in enumerate(sets):
        cs = cell_stats(X, P1, [tuple(A)])[0]
        out[0, j] = projection_information(cs, range(len(A)), link="logit")
        out[1, j] = projection_information(full, tuple(A), link="logit")
    return out


def _accuracy(logodds: np.ndarray, truth: np.ndarray, pmf: np.ndarray) -> float:
    pred = logodds > 0
    return float(np.sum(pmf * (pred == (truth > 0.5))))


def support_inputs(task: str, task_cfg: dict) -> tuple[int, ...]:
    """Inputs in the generating support: Markov, lags of planted couplings with β ≠ 0 (as inputs lag − 1);
    parity, the planted set."""
    if task == "markov_lang":
        from tycho.tasks.markov_lang import parse_couplings

        cp = parse_couplings(task_cfg["couplings"], int(task_cfg.get("window", 6)))
        return tuple(sorted({lag - 1 for L, beta in cp.items() if beta != 0.0 for lag in L}))
    if task == "sparse_parity":
        return tuple(sorted(int(i) for i in task_cfg["support"]))
    raise ValueError(f"unknown task {task!r}")


def _ic3(rec: dict, pmf_D: np.ndarray, truth: np.ndarray) -> dict:
    """IC-3 for one chain Markov run: TV between the exact window pmf and the corpus, and the transition check."""
    from tycho.tasks import make_task
    from tycho.tasks.patterns import markov_window_pmf

    task = make_task("markov_lang", rec["task_cfg"], int(rec["seed"]))
    corpus = task._corpus
    n_pos = corpus["n_pos"]
    lag = corpus["lag"][:n_pos].astype(np.int64)
    tg = corpus["targets"][:n_pos].numpy().astype(np.float64)
    W = lag.shape[1]
    codes = (lag << np.arange(W)[None, :]).sum(axis=1)
    nc = np.bincount(codes, minlength=1 << W).astype(np.float64)
    kc = np.bincount(codes, weights=tg, minlength=1 << W)
    exact = markov_window_pmf(task.p_table, W)
    tv = 0.5 * float(np.abs(exact - nc / nc.sum()).sum())
    sel = nc >= IC3_MIN_COUNT
    p = task.p_table[sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (kc[sel] / nc[sel] - p) / np.sqrt(p * (1.0 - p) / nc[sel])
    return {"tv": tv, "n_windows_checked": int(sel.sum()), "max_abs_z": float(np.max(np.abs(z))) if z.size else 0.0,
            "n_fail": int(np.sum(~(np.abs(z) <= IC3_Z))), "pmf_max_diff": float(np.max(np.abs(exact - pmf_D))),
            "truth_max_diff": float(np.max(np.abs(task.p_table - truth))), "n_windows": int(n_pos)}


def _q5(logodds: np.ndarray, m: np.ndarray, pmf_D: np.ndarray, pmf_S: np.ndarray, sets, cd: dict) -> dict:
    """Q5 at one checkpoint: top-K sets at C_D per link, logistic model on them under D, KL under S and D."""
    n = m.size
    d = n.bit_length() - 1
    t = (1.0 + m) / 2.0
    p_net = 0.5 * (1.0 + np.tanh(0.5 * np.asarray(logodds, dtype=np.float64)))
    W = (pmf_D / pmf_D.sum())[None, :]
    res: dict[str, Any] = {}
    for val in VALUES:
        order = np.argsort(-cd[val], kind="stable")
        for K in Q5_K:
            top = [sets[i] for i in order[:K]]
            cols = np.array(sorted([0] + [mask_of(A) for A in top]), dtype=np.int64)
            _, beta, _ = _fit(d, cols, W, t[None, :], np.zeros((1, cols.size)))
            zq = _logits(d, cols, beta)[0]
            res[f"{val}_K{K}_kl_S"] = binary_losses(zq, p_net, pmf_S)[0]
            res[f"{val}_K{K}_kl_D"] = binary_losses(zq, p_net, pmf_D)[0]
            res[f"{val}_K{K}_sets"] = [list(A) for A in top]
    return res


def object_measures(logodds: np.ndarray, pmf_D: np.ndarray, pmf_U: np.ndarray, counts: np.ndarray | None,
                    sets: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    """All measures of one object: ``theta``, ``eta_b``, ``eta_g`` (4 refs × checkpoints × sets, refs in
    :data:`REFS` order) and, with ``counts``, ``theta_sampled`` (2 × checkpoints × sets: M_D, C_D)."""
    m = clipped_mean(np.atleast_2d(logodds))
    ws = [pmf_D, pmf_U]
    lin = [True, True]
    if counts is not None:
        ws.append(np.asarray(counts, dtype=np.float64))
        lin.append(False)
    M = marginal_measures(m, ws, sets, linear=lin)
    C = conditional_measures(m, ws, sets, linear=lin)
    out = {v: np.stack([M[0][v], M[1][v], C[0][v], C[1][v]]) for v in VALUES}
    out["iters"] = np.stack([M[0]["iters"], M[1]["iters"], C[0]["iters"], C[1]["iters"]])
    if counts is not None:
        out["theta_sampled"] = np.stack([M[2]["theta"], C[2]["theta"]])
    return out


def stage1_object(rec: dict, cache_dir: str | os.PathLike, manifest: str | os.PathLike | None = None, *,
                  force: bool = False, ic2a_fraction: float = IC2A_FRACTION, ic3: bool = True,
                  ic4_local_ri: bool = True) -> dict:
    """Stage 1 of one done object, cached at ``cache_dir/<object_id>.npz`` (recomputed when the stage-1 version or
    the patterns file changed). Returns {"object_id", "path", "seconds", "cached"}."""
    t_start = time.perf_counter()
    oid = rec["object_id"]
    cache = Path(cache_dir) / f"{oid}.npz"
    ppath = resolve_path(rec["patterns"], Path(manifest) if manifest else None)
    sha = _file_sha1(ppath)
    if cache.exists() and not force:
        try:
            with np.load(cache, allow_pickle=False) as z:
                chain = rec["task"] == "markov_lang" and rec["task_cfg"].get("train_context", "chain") == "chain"
                ok = (str(z["version"]) == STAGE1_VERSION and str(z["patterns_sha1"]) == sha
                      and (ic2a_fraction <= 0 or z["ic2a_idx"].size > 0)
                      and (not ic4_local_ri or "local_ri" in z.files)
                      and (not (ic3 and chain) or json.loads(str(z["ic3_json"])) is not None))
                if ok:  # the cache holds at least what this call asks for
                    return {"object_id": oid, "path": str(cache), "seconds": float(z["seconds_total"]),
                            "cached": True}
        except Exception:  # unreadable cache: recompute
            pass
    with np.load(ppath, allow_pickle=False) as z:
        P = {k: z[k] for k in z.files}
    steps, lo = P["steps"].astype(np.int64), P["logodds"].astype(np.float64)
    counts = P["counts"].astype(np.int64)
    pmf_D, pmf_U, pmf_S, truth = (P[k].astype(np.float64) for k in ("pmf_D", "pmf_U", "pmf_S", "truth"))
    n_ck, n = lo.shape
    d = n.bit_length() - 1
    sets = all_sets(d)
    m = clipped_mean(lo)

    t0 = time.perf_counter()
    meas = object_measures(lo, pmf_D, pmf_U, counts, sets)
    t_meas = time.perf_counter() - t0

    kl = np.empty((3, n_ck))
    sq = np.empty((3, n_ck))
    acc = np.empty((3, n_ck))
    for i, pmf in enumerate((pmf_D, pmf_U, pmf_S)):
        for t in range(n_ck):
            kl[i, t], sq[i, t] = binary_losses(lo[t], truth, pmf)
            acc[i, t] = _accuracy(lo[t], truth, pmf)
    fu = walsh_coefficients(m)  # (checkpoints, masks)
    lo_energy = np.empty((n_ck, len(sets)))
    for j, A in enumerate(sets):
        proper = [mask_of(B) for r in range(1, len(A)) for B in itertools.combinations(A, r)]
        lo_energy[:, j] = np.sqrt(np.sum(fu[:, proper] ** 2, axis=1)) if proper else 0.0
    fu_sets = fu[:, [mask_of(A) for A in sets]]

    out: dict[str, Any] = {"version": STAGE1_VERSION, "object_id": oid, "patterns_sha1": sha, "steps": steps,
                           "sets": np.array([list(A) + [-1] * (3 - len(A)) for A in sets], dtype=np.int64),
                           "kl": kl, "sq": sq, "acc": acc, "saturation": np.max(np.abs(m), axis=1),
                           "lo_energy": lo_energy, "fu_sets": fu_sets, "seconds_measures": t_meas,
                           "n_probe": int(P.get("n_probe", counts[0].sum())),
                           "counts_total_ok": bool(np.all(counts.sum(axis=1) == counts[0].sum()))}
    for v in VALUES:
        out[v] = meas[v]
    out["theta_sampled"] = meas["theta_sampled"]
    out["iters"] = meas["iters"]

    # IC-4, second part: θ_M_D against tycho.ri.refined.local_ri of the exact marginal.
    t0 = time.perf_counter()
    if ic4_local_ri:
        from tycho.ri.cells import CellStats
        from tycho.ri.refined import local_ri

        lri = np.empty((n_ck, len(sets)))
        for k in sorted({len(A) for A in sets}):
            idx = [j for j, A in enumerate(sets) if len(A) == k]
            cnt, mA = marginal_tables(m, pmf_D, [sets[j] for j in idx])
            for t in range(n_ck):
                for jj, j in enumerate(idx):
                    cs = CellStats(inputs=tuple(sets[j]), counts=cnt[t, jj], sums=cnt[t, jj] * mA[t, jj], n=1.0)
                    lri[t, j] = local_ri(cs.table())
        out["local_ri"] = lri
    out["seconds_ic4"] = time.perf_counter() - t0

    # IC-2(a): the library path on the probe rows, at ceil(5 %) of the checkpoints; probe counts from the probe seed.
    t0 = time.perf_counter()
    n_sel = math.ceil(ic2a_fraction * n_ck) if ic2a_fraction > 0 else 0
    rng = np.random.default_rng([BOOT_SEED, int(oid, 16) % (1 << 63)])
    sel = np.sort(rng.choice(n_ck, size=min(n_sel, n_ck), replace=False)) if n_sel else np.zeros(0, np.int64)
    lib = np.stack([library_sampled_theta(lo[t], counts[t], sets) for t in sel]) if sel.size else np.zeros(
        (0, 2, len(sets)))
    regen_ok = np.zeros(sel.size, bool)
    if sel.size:
        from tycho.tasks.base import torch_generator
        from tycho.tasks.patterns import sample_codes

        for i, t in enumerate(sel):
            codes = sample_codes(pmf_D, int(out["n_probe"]), torch_generator(int(rec["seed"]), "ri", int(steps[t])))
            regen_ok[i] = bool(np.array_equal(np.bincount(codes, minlength=n), counts[t]))
    out.update(ic2a_idx=sel.astype(np.int64), ic2a_lib=lib, ic2a_regen_ok=regen_ok)
    out["seconds_ic2a"] = time.perf_counter() - t0

    # IC-3 for chain Markov runs.
    t0 = time.perf_counter()
    if ic3 and rec["task"] == "markov_lang" and rec["task_cfg"].get("train_context", "chain") == "chain":
        out["ic3_json"] = json.dumps(_ic3(rec, pmf_D, truth))
    else:
        out["ic3_json"] = json.dumps(None)
    out["seconds_ic3"] = time.perf_counter() - t0

    # Q5 at the last checkpoint.
    cd = {v: meas[v][2, -1] for v in VALUES}
    out["q5_json"] = json.dumps(_q5(lo[-1], m[-1], pmf_D, pmf_S, sets, cd))
    out["seconds_total"] = time.perf_counter() - t_start
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in out.items()})
    os.replace(tmp, cache)
    return {"object_id": oid, "path": str(cache), "seconds": out["seconds_total"], "cached": False}


# ================================================================================================ IC-1 replays
def replay_object(rec: dict, sky_yaml: str, work_dir: str, manifest: str | None = None) -> dict:
    """IC-1: replay one object with :func:`tycho.train.run_object` and compare its light curve (every key except
    ``wall_s``) and its pattern arrays with the stored files, bit for bit."""
    import tempfile

    from tycho.survey.sky import Sky
    from tycho.train import run_object
    from tycho.train.device import configure_torch
    from tycho.train.spec import ObjectSpec, RIConfig, TrainConfig

    t0 = time.perf_counter()
    configure_torch(1)
    region = Sky.from_yaml(sky_yaml).region(rec["region"])
    spec = ObjectSpec(task=rec["task"], task_cfg=dict(rec["task_cfg"]), seed=int(rec["seed"]),
                      train=TrainConfig.from_dict(region.train), sky_version=rec["sky_version"], region=rec["region"])
    out: dict[str, Any] = {"object_id": rec["object_id"], "region": rec["region"],
                           "spec_id_matches": spec.object_id() == rec["object_id"]}
    if not out["spec_id_matches"]:
        out.update(passed=False, error=f"rebuilt spec has object id {spec.object_id()}")
        return out
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        res = run_object(spec, work_dir=Path(tmp), ri=RIConfig.from_dict({**region.ri, "enabled": True}),
                         resume=False)
    out["status"] = res.status
    man = Path(manifest) if manifest else None
    mismatches = []
    with np.load(resolve_path(rec["lightcurve"], man)) as z:
        stored = {k: z[k] for k in z.files}
    keys = sorted(set(stored) | set(res.light_curve))
    for k in keys:
        if k == "wall_s":
            continue
        a, b = stored.get(k), res.light_curve.get(k)
        if a is None or b is None or np.asarray(a).dtype != np.asarray(b).dtype or \
                np.asarray(a).tobytes() != np.asarray(b).tobytes():
            mismatches.append(f"lightcurve:{k}")
    with np.load(resolve_path(rec["patterns"], man)) as z:
        stored_p = {k: z[k] for k in z.files}
    for k in ("steps", "logodds", "counts", "pmf_D", "pmf_U", "pmf_S", "truth"):
        a, b = stored_p.get(k), (res.patterns or {}).get(k)
        if a is None or b is None or np.asarray(a).tobytes() != np.asarray(b).tobytes():
            mismatches.append(f"patterns:{k}")
    out.update(n_lightcurve_keys=len(keys) - ("wall_s" in keys), mismatches=mismatches,
               passed=res.status == "done" and not mismatches, seconds=time.perf_counter() - t0)
    return out


# ================================================================================================ stage 2: runs
def prereg_region(task: str) -> str:
    """The region of §2: "markov-v2" or "parity-corr" (configs/sky_v2a.yaml splits the latter by k)."""
    return "markov-v2" if task == "markov_lang" else "parity-corr"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


@dataclass(eq=False)
class Run:
    """One done object with its stage-1 arrays."""

    rec: dict
    z: dict
    support: tuple[int, ...]
    sets: list[tuple[int, ...]]
    ic5_value: float = math.nan
    ic5_pass: bool = False
    win: int | None = None

    @property
    def oid(self) -> str:
        return self.rec["object_id"]

    @property
    def task(self) -> str:
        return self.rec["task"]

    @property
    def region(self) -> str:
        return self.rec["region"]

    @property
    def seed(self) -> int:
        return int(self.rec["seed"])

    @property
    def context(self) -> str | None:
        return self.rec["task_cfg"].get("train_context", "chain") if self.task == "markov_lang" else None

    @property
    def couplings(self) -> str | None:
        return self.rec["task_cfg"]["couplings"] if self.task == "markov_lang" else None

    @property
    def rho(self) -> float | None:
        return float(self.rec["task_cfg"].get("rho", 0.0)) if self.task == "sparse_parity" else None

    @property
    def config(self) -> str:
        if self.task == "markov_lang":
            return f"{self.couplings} | {self.context}"
        return f"{self.region} rho={self.rho:g}"

    @property
    def steps(self) -> np.ndarray:
        return self.z["steps"]

    def measure(self, ref: str, value: str) -> np.ndarray:
        """(checkpoints, sets) values of one measure."""
        return self.z[value][REFS.index(ref)]

    def spurious(self, min_size: int = 1) -> np.ndarray:
        sup = set(self.support)
        return np.array([len(A) >= min_size and any(i not in sup for i in A) for A in self.sets])


def window_start(steps: np.ndarray, loss: np.ndarray) -> int | None:
    """§4 window: index of the first checkpoint t >= 1 with loss(t) <= ½ loss(0) (None if never); step 0 never in."""
    steps = np.asarray(steps)
    if steps.size == 0 or steps[0] != 0:
        raise ValueError("the first checkpoint must be step 0")
    thr = 0.5 * float(loss[0])
    for i in range(1, steps.size):
        if steps[i] >= 1 and loss[i] <= thr:
            return i
    return None


def load_manifest(path: str | os.PathLike) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_stage1(path: str | os.PathLike) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build_runs(records: Sequence[dict], cache_dir: str | os.PathLike) -> list[Run]:
    """Runs of the done records with their stage-1 arrays, IC-5 value and window."""
    runs = []
    for rec in records:
        if rec.get("status") != "done":
            continue
        z = _load_stage1(Path(cache_dir) / f"{rec['object_id']}.npz")
        sets = [tuple(int(i) for i in row if i >= 0) for row in z["sets"]]
        r = Run(rec=rec, z=z, support=support_inputs(rec["task"], rec["task_cfg"]), sets=sets)
        if r.task == "markov_lang":
            r.ic5_value = float(z["kl"][0, -1])
            r.ic5_pass = r.ic5_value < IC5_KL
        else:
            r.ic5_value = float(z["acc"][0, -1])
            r.ic5_pass = r.ic5_value >= IC5_ACC
        r.win = window_start(z["steps"], z["kl"][0])
        runs.append(r)
    return runs


# ================================================================================================ Spearman
def _rank_prep(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort order and, per sorted position, the [start, end) of its tie group (exact equality)."""
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    order = np.argsort(v, kind="mergesort")
    if n == 0:
        return order, np.zeros(0, np.int64), np.zeros(0, np.int64)
    sv = v[order]
    new = np.empty(n, bool)
    new[0] = True
    new[1:] = sv[1:] != sv[:-1]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n)
    return order, starts[gid], ends[gid]


def _midranks(prep: tuple[np.ndarray, np.ndarray, np.ndarray], Wm: np.ndarray) -> np.ndarray:
    """Mid-ranks of the multiset in which point i appears Wm[b, i] times (rows b = resamples)."""
    order, gs, ge = prep
    B, n = Wm.shape
    C = np.zeros((B, n + 1))
    np.cumsum(Wm[:, order], axis=1, out=C[:, 1:])
    before = C[:, gs]
    R = np.empty((B, n))
    R[:, order] = before + (C[:, ge] - before + 1.0) / 2.0
    return R


def _wcorr(Rx: np.ndarray, Ry: np.ndarray, Wm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Pearson correlation of rank rows; 0 where a side is constant or there are < 2 points. Returns
    (ρ, degenerate flag)."""
    N = Wm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = (Wm * Rx).sum(axis=1) / N
        my = (Wm * Ry).sum(axis=1) / N
        dx = Rx - mx[:, None]
        dy = Ry - my[:, None]
        vx = (Wm * dx * dx).sum(axis=1)
        vy = (Wm * dy * dy).sum(axis=1)
        c = (Wm * dx * dy).sum(axis=1)
        rho = c / np.sqrt(vx * vy)
    ok = (N >= 2) & (vx > 0) & (vy > 0)
    return np.where(ok, rho, 0.0), ~ok


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Signed Spearman ρ with average ranks; 0 if x or y is constant or there are fewer than 2 points (§4)."""
    return spearman_info(x, y)[0]


def spearman_info(x: np.ndarray, y: np.ndarray) -> tuple[float, str | None]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("x and y differ in length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("non-finite values in a Spearman correlation")
    if x.size < 2:
        return 0.0, "fewer than 2 points"
    W = np.ones((1, x.size))
    rho, _ = _wcorr(_midranks(_rank_prep(x), W), _midranks(_rank_prep(y), W), W)
    if np.all(x == x[0]):
        return 0.0, "constant predictor"
    if np.all(y == y[0]):
        return 0.0, "constant outcome"
    return float(rho[0]), None


def boot_pvalue(deltas: np.ndarray) -> float:
    """§4: min(1, 2·min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / (B + 1))."""
    d = np.asarray(deltas, dtype=np.float64)
    B = d.size
    return float(min(1.0, 2.0 * min(np.sum(d <= 0) + 1, np.sum(d >= 0) + 1) / (B + 1)))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def resample_multiplicities(n_seeds: int, B: int, seed: int = BOOT_SEED) -> np.ndarray:
    """(B, n_seeds) multiplicities of the seed labels in B resamples of n_seeds labels drawn with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_seeds, size=(B, n_seeds))
    mult = np.zeros((B, n_seeds))
    np.add.at(mult, (np.repeat(np.arange(B), n_seeds), draws.ravel()), 1.0)
    return mult


# ================================================================================================ T1 / T2
PRIMARY_KEYS = tuple(f"{ref}:{val}" for ref in REFS for val in VALUES)


@dataclass(eq=False)
class Cfg:
    """The window points of one configuration (T1) or twin set (T2)."""

    name: str
    seed_idx: list = field(default_factory=list)
    ckpt: list = field(default_factory=list)
    unit: list = field(default_factory=list)
    X: dict = field(default_factory=dict)
    Y: list = field(default_factory=list)
    Ysq: list = field(default_factory=list)
    units_in: list = field(default_factory=list)
    seeds_in: list = field(default_factory=list)
    excluded_ic5: list = field(default_factory=list)
    no_window: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add(self, seed_i: int, unit: str, ckpts: np.ndarray, X: dict, Y: np.ndarray, Ysq: np.ndarray) -> None:
        n = len(ckpts)
        self.seed_idx += [seed_i] * n
        self.ckpt += [int(c) for c in ckpts]
        self.unit += [unit] * n
        for k, v in X.items():
            self.X.setdefault(k, []).extend(np.asarray(v, dtype=np.float64).tolist())
        self.Y += np.asarray(Y, dtype=np.float64).tolist()
        self.Ysq += np.asarray(Ysq, dtype=np.float64).tolist()
        self.units_in.append(unit)

    def finalize(self) -> Cfg:
        self.seed_idx = np.asarray(self.seed_idx, dtype=np.int64)
        self.ckpt = np.asarray(self.ckpt, dtype=np.int64)
        self.unit = np.asarray(self.unit)
        self.X = {k: np.asarray(v, dtype=np.float64) for k, v in self.X.items()}
        self.Y = np.asarray(self.Y, dtype=np.float64)
        self.Ysq = np.asarray(self.Ysq, dtype=np.float64)
        return self

    @property
    def n_valid(self) -> int:
        """Seeds passing IC-5 with at least one window point."""
        return len(self.units_in)


def build_t1(runs: Sequence[Run], family: str, seed_labels: Sequence[int]) -> list[Cfg]:
    """T1 configurations of ``family`` ("T1_markov": chain Markov runs; "T1_parity": parity rho in {0.5, 0.8})."""
    if family == "T1_markov":
        sel = [r for r in runs if r.task == "markov_lang" and r.context == "chain"]
    elif family == "T1_parity":
        sel = [r for r in runs if r.task == "sparse_parity" and r.rho in (0.5, 0.8)]
    else:
        raise ValueError(family)
    cfgs = []
    for name in sorted({r.config for r in sel}):
        cfg = Cfg(name=name)
        for r in sorted((r for r in sel if r.config == name), key=lambda r: r.seed):
            if not r.ic5_pass:
                cfg.excluded_ic5.append({"object_id": r.oid, "seed": r.seed, "ic5_value": r.ic5_value})
                continue
            if r.win is None:
                cfg.no_window.append({"object_id": r.oid, "seed": r.seed})
                continue
            idx = np.arange(r.win, r.steps.size)
            sp2, spa = r.spurious(2), r.spurious(1)
            X = {}
            for ref in REFS:
                for val in VALUES:
                    M = r.measure(ref, val)[idx]
                    X[f"{ref}:{val}"] = M[:, sp2].sum(axis=1)
                    X[f"all|{ref}:{val}"] = M[:, spa].sum(axis=1)
            kl, sq = r.z["kl"], r.z["sq"]
            cfg.add(list(seed_labels).index(r.seed), r.oid, idx, X, kl[2, idx] - kl[0, idx], sq[2, idx] - sq[0, idx])
            cfg.seeds_in.append(r.seed)
        r0 = next(r for r in sel if r.config == name)
        cfg.info = {"n_spurious_ge2": int(r0.spurious(2).sum()), "n_spurious_all": int(r0.spurious(1).sum()),
                    "support_inputs": list(r0.support)}
        cfgs.append(cfg.finalize())
    return cfgs


def exact_markov_pmf(couplings: str, context: str, window: int = 6) -> np.ndarray:
    """D of a Markov language: the exact stationary window pmf (chain) or uniform (iid)."""
    from tycho.tasks.markov_lang import _conditional_table, parse_couplings
    from tycho.tasks.patterns import markov_window_pmf, uniform_pmf

    if context == "iid":
        return uniform_pmf(window)
    return markov_window_pmf(_conditional_table(window, parse_couplings(couplings, window), 0.0), window)


def identifiability_table(window: int = 6) -> list[dict]:
    """§4 identifiability residual variances of every twin set at the four references, from the exact pmfs of the
    "with" languages, with the values printed in §4 for comparison."""
    out = []
    for wo, wi, lags in TWIN_PAIRS:
        A = tuple(lag - 1 for lag in lags)
        for ctx in ("chain", "iid"):
            pmf = exact_markov_pmf(wi, ctx, window)
            row = {"twin_set": f"{wo} vs {wi} | {ctx}", "added_lags": list(lags), "added_inputs": list(A)}
            for ref in REFS:
                w = pmf if ref.endswith("D") else np.full(pmf.size, 1.0 / pmf.size)
                v = residual_variance(w, A, ref)
                row[ref] = v
                row[f"enters_{ref}"] = bool(v >= IDENT_MIN)
                if ctx == "chain" and (wi, ref) in PREREG_IDENT:
                    row[f"prereg_{ref}"] = PREREG_IDENT[(wi, ref)]
            out.append(row)
    return out


def _sig_digits(x: float) -> int:
    mant = repr(float(x)).split("e")[0].replace("-", "").replace(".", "").lstrip("0")
    return max(1, len(mant))


def _round_half_up(x: float, digits: int) -> float:
    from decimal import ROUND_HALF_UP, Decimal

    if x == 0:
        return 0.0
    q = Decimal(1).scaleb(math.floor(math.log10(abs(x))) - digits + 1)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def prereg_matches(value: float, printed: float) -> dict:
    """Whether ``value`` gives the number printed in §4 at its significant digits: by one rounding, or by rounding
    first to one more digit (a double rounding, e.g. 0.0061477 -> 0.00615 -> 0.0062)."""
    n = _sig_digits(printed)
    single = _round_half_up(value, n) == float(printed)
    double = _round_half_up(_round_half_up(value, n + 1), n) == float(printed)
    return {"single_rounding": bool(single), "double_rounding": bool(double),
            "rel_diff": float(abs(value - printed) / abs(printed)) if printed else math.nan}


def build_t2(runs: Sequence[Run], seed_labels: Sequence[int], manifest: Path | None) -> tuple[list[Cfg], dict]:
    """T2 twin sets: per pair and context, the window points of every seed with both twins done and passing IC-5."""
    idx = {(r.couplings, r.context, r.seed): r for r in runs if r.task == "markov_lang"}
    ident = {row["twin_set"]: row for row in identifiability_table()}
    checks = {"pmf_with_max_diff": 0.0, "iid_without_recompute_max_diff": 0.0}
    cfgs = []
    for wo_c, wi_c, lags in TWIN_PAIRS:
        A = tuple(lag - 1 for lag in lags)
        for ctx in ("chain", "iid"):
            name = f"{wo_c} vs {wi_c} | {ctx}"
            cfg = Cfg(name=name)
            cfg.info = {"added_lags": list(lags), "resid": {ref: ident[name][ref] for ref in REFS}}
            seeds = sorted({s for (c, x, s) in idx if x == ctx and c in (wo_c, wi_c)})
            exact = exact_markov_pmf(wi_c, ctx)
            for s in seeds:
                wo, wi = idx.get((wo_c, ctx, s)), idx.get((wi_c, ctx, s))
                if wo is None or wi is None:
                    cfg.missing.append({"seed": s, "without": None if wo is None else wo.oid,
                                        "with": None if wi is None else wi.oid})
                    continue
                if not (wo.ic5_pass and wi.ic5_pass):
                    cfg.excluded_ic5.append({"seed": s, "without": [wo.oid, wo.ic5_value, wo.ic5_pass],
                                             "with": [wi.oid, wi.ic5_value, wi.ic5_pass]})
                    continue
                if not np.array_equal(wo.steps, wi.steps):
                    raise ValueError(f"twins {wo.oid} and {wi.oid} have different checkpoints")
                if wi.win is None:
                    cfg.no_window.append({"seed": s, "with": wi.oid})
                    continue
                with np.load(resolve_path(wo.rec["patterns"], manifest)) as z:
                    z_wo = z["logodds"].astype(np.float64)
                with np.load(resolve_path(wi.rec["patterns"], manifest)) as z:
                    z_wi = z["logodds"].astype(np.float64)
                    pmf_with = z["pmf_D"].astype(np.float64)
                checks["pmf_with_max_diff"] = max(checks["pmf_with_max_diff"], float(np.abs(pmf_with - exact).max()))
                t_idx = np.arange(wi.win, wi.steps.size)
                jA = wi.sets.index(A)
                m_wo = clipped_mean(z_wo[t_idx])
                under = {"M_D": marginal_measures(m_wo, [pmf_with], [A])[0],
                         "C_D": conditional_measures(m_wo, [pmf_with], [A])[0]}
                X = {}
                for ref in REFS:
                    for val in VALUES:
                        a_with = wi.measure(ref, val)[t_idx, jA]
                        if ref in under:
                            a_wo = under[ref][val][:, 0]
                            if ctx == "iid":
                                checks["iid_without_recompute_max_diff"] = max(
                                    checks["iid_without_recompute_max_diff"],
                                    float(np.abs(a_wo - wo.measure(ref, val)[t_idx, jA]).max()))
                        else:
                            a_wo = wo.measure(ref, val)[t_idx, jA]
                        X[f"{ref}:{val}"] = a_with - a_wo
                Y = np.empty(t_idx.size)
                Ysq = np.empty(t_idx.size)
                for i, t in enumerate(t_idx):
                    Y[i], Ysq[i] = binary_losses(z_wo[t], _sigmoid(z_wi[t]), pmf_with)
                cfg.add(list(seed_labels).index(s), f"{wo.oid}+{wi.oid}", t_idx, X, Y, Ysq)
                cfg.seeds_in.append(s)
            cfgs.append(cfg.finalize())
    return cfgs, checks


def _family_kept(cfgs: Sequence[Cfg], family: str, min_seeds: int) -> dict[str, list[bool]]:
    kept = {}
    for ref in REFS:
        row = []
        for c in cfgs:
            ok = c.n_valid >= min_seeds
            if family == "T2" and ref in ("M_D", "C_D"):
                ok = ok and c.info["resid"][ref] >= IDENT_MIN
            row.append(bool(ok))
        kept[ref] = row
    return kept


def _boot_rhos(cfgs: Sequence[Cfg], keys: Sequence[str], mult: np.ndarray, *, chunk: int = 250
               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-configuration Spearman ρ of each predictor key with Y in every resample: {key: (configs, B)}, and the
    count of resamples in which ρ was set to 0 by the degenerate-case rule."""
    B = mult.shape[0]
    out = {k: np.zeros((len(cfgs), B)) for k in keys}
    degen = {k: np.zeros(len(cfgs), np.int64) for k in keys}
    for ci, cfg in enumerate(cfgs):
        if cfg.Y.size == 0:
            for k in keys:
                degen[k][ci] = B
            continue
        py = _rank_prep(cfg.Y)
        px = {k: _rank_prep(cfg.X[k]) for k in keys}
        for b0 in range(0, B, chunk):
            Wm = mult[b0:b0 + chunk][:, cfg.seed_idx]
            Ry = _midranks(py, Wm)
            for k in keys:
                out[k][ci, b0:b0 + chunk], dg = _wcorr(_midranks(px[k], Wm), Ry, Wm)
                degen[k][ci] += int(dg.sum())
    return out, degen


def _seed_level(cfgs: Sequence[Cfg], key: str, outcome: str = "Y") -> tuple[float, int]:
    """Mean over (configuration, checkpoint) cells with >= 3 seeds of the Spearman across seeds."""
    vals = []
    for c in cfgs:
        Y = c.Y if outcome == "Y" else c.Ysq
        for t in np.unique(c.ckpt):
            sel = c.ckpt == t
            if sel.sum() >= 3:
                vals.append(spearman(c.X[key][sel], Y[sel]))
    return (float(np.mean(vals)) if vals else math.nan), len(vals)


def _rbar(per_cfg: Sequence[dict], col: str) -> float:
    """Unweighted mean of a per-configuration ρ over the configurations that enter ρ̄."""
    v = [row[col] for row in per_cfg if row["kept"]]
    return float(np.mean(v)) if v else math.nan


def evaluate_family(family: str, cfgs: Sequence[Cfg], seed_labels: Sequence[int], *, B: int, min_seeds: int
                    ) -> list[dict]:
    """The four comparisons (one per reference) of a family: point estimates, bootstrap, raw p-values, secondaries."""
    kept = _family_kept(cfgs, family, min_seeds)
    mult = resample_multiplicities(len(seed_labels), B, BOOT_SEED) if B > 0 else np.zeros((0, len(seed_labels)))
    boot, degen = _boot_rhos(cfgs, PRIMARY_KEYS, mult) if B > 0 else ({}, {})
    point = {}
    point_reason = {}
    for c in cfgs:
        for k in c.X:
            for outcome in ("Y", "Ysq"):
                rho, why = spearman_info(c.X[k], getattr(c, outcome))
                point[(c.name, k, outcome)] = rho
                if why:
                    point_reason[(c.name, k, outcome)] = why
    res = []
    for ref in REFS:
        kc = [c for c, ok in zip(cfgs, kept[ref], strict=True) if ok]
        ki = [i for i, ok in enumerate(kept[ref]) if ok]
        comp: dict[str, Any] = {"family": family, "reference": ref, "comparison": f"{family} {ref}",
                                "configurations": [c.name for c in kc],
                                "dropped": [{"name": c.name, "n_valid_seeds": c.n_valid,
                                             "reason": ("identifiability" if family == "T2" and ref in ("M_D", "C_D")
                                                        and c.info["resid"][ref] < IDENT_MIN else
                                                        f"fewer than {min_seeds} seeds passing IC-5 with a window "
                                                        "point")}
                                            for c, ok in zip(cfgs, kept[ref], strict=True) if not ok]}
        per_cfg = []
        for c, ok in zip(cfgs, kept[ref], strict=True):
            row = {"name": c.name, "kept": ok, "n_valid_seeds": c.n_valid, "n_points": int(c.Y.size)}
            for val in VALUES:
                row[f"rho_{val}"] = point.get((c.name, f"{ref}:{val}", "Y"), 0.0)
                row[f"rho_sens_{val}"] = point.get((c.name, f"{ref}:{val}", "Ysq"), 0.0)
                if family != "T2":
                    row[f"rho_allspurious_{val}"] = point.get((c.name, f"all|{ref}:{val}", "Y"), 0.0)
                why = point_reason.get((c.name, f"{ref}:{val}", "Y"))
                if why:
                    row.setdefault("degenerate", {})[val] = why
            if family == "T2":
                row["resid_var"] = c.info["resid"][ref]
            per_cfg.append(row)
        comp["per_configuration"] = per_cfg

        for val in VALUES:
            comp[f"rhobar_{val}"] = _rbar(per_cfg, f"rho_{val}")
            comp[f"rhobar_sens_{val}"] = _rbar(per_cfg, f"rho_sens_{val}")
            if family != "T2":
                comp[f"rhobar_allspurious_{val}"] = _rbar(per_cfg, f"rho_allspurious_{val}")
        for ev in ETA_VALUES:
            comp[f"delta_{ev}"] = comp["rhobar_theta"] - comp[f"rhobar_{ev}"]
            comp[f"delta_sens_{ev}"] = comp["rhobar_sens_theta"] - comp[f"rhobar_sens_{ev}"]
            if family != "T2":
                comp[f"delta_allspurious_{ev}"] = comp["rhobar_allspurious_theta"] - comp[f"rhobar_allspurious_{ev}"]
            if B > 0 and kc:
                th = boot[f"{ref}:theta"][ki].mean(axis=0)
                et = boot[f"{ref}:{ev}"][ki].mean(axis=0)
                d = th - et
                comp[f"boot_{ev}"] = {"B": int(B), "sd": float(np.std(d, ddof=1)),
                                      "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                      "mean": float(d.mean()), "p_raw": boot_pvalue(d),
                                      "degenerate_resamples": {
                                          "theta": int(degen[f"{ref}:theta"][ki].sum()),
                                          ev: int(degen[f"{ref}:{ev}"][ki].sum())}}
                comp[f"p_raw_{ev}"] = comp[f"boot_{ev}"]["p_raw"]
            else:
                comp[f"p_raw_{ev}"] = 1.0
                comp[f"boot_{ev}"] = None
        # secondaries
        sec: dict[str, Any] = {}
        if kc:
            for val in VALUES:
                X = np.concatenate([c.X[f"{ref}:{val}"] for c in kc])
                for outcome in ("Y", "Ysq"):
                    Y = np.concatenate([getattr(c, outcome) for c in kc])
                    sec[f"pooled_{outcome}_{val}"] = spearman(X, Y)
                    sec[f"seedlevel_{outcome}_{val}"], sec[f"seedlevel_cells_{outcome}"] = _seed_level(
                        kc, f"{ref}:{val}", outcome)
            for ev in ETA_VALUES:
                sec[f"pooled_delta_{ev}"] = sec["pooled_Y_theta"] - sec[f"pooled_Y_{ev}"]
                sec[f"seedlevel_delta_{ev}"] = sec["seedlevel_Y_theta"] - sec[f"seedlevel_Y_{ev}"]
            if family == "T2":
                chain = [row for row in per_cfg if row["kept"] and row["name"].endswith("| chain")]
                for val in VALUES:
                    sec[f"chain_only_rhobar_{val}"] = float(np.mean([r[f"rho_{val}"] for r in chain])) if chain \
                        else math.nan
                for ev in ETA_VALUES:
                    sec[f"chain_only_delta_{ev}"] = sec["chain_only_rhobar_theta"] - sec[f"chain_only_rhobar_{ev}"]
                sec["chain_only_twin_sets"] = [r["name"] for r in chain]
        comp["secondary"] = sec
        res.append(comp)
    return res


def _meets(c: dict, ev: str, sgn: float) -> bool:
    """Condition 1 (eta_b) or 2 (eta_g) of §4 for the link with sign ``sgn`` (+1 θ, −1 η)."""
    d = c[f"delta_{ev}"]
    return bool(sgn * d > 0 and abs(d) >= MARGIN and c[f"p_holm_{ev}"] < ALPHA)


def apply_decision(comps: list[dict]) -> dict:
    """Holm over the 12 comparisons per η value, the three win conditions, labels and the §4 decision table."""
    for ev in ETA_VALUES:
        adj = holm([c[f"p_raw_{ev}"] if math.isfinite(c.get(f"delta_{ev}", math.nan)) else 1.0 for c in comps])
        for c, a in zip(comps, adj, strict=True):
            c[f"p_holm_{ev}"] = float(a)
    wins = {"theta": [], "eta": []}
    for c in comps:
        cond = {}
        computable = all(math.isfinite(c.get(f"delta_{ev}", math.nan)) for ev in ETA_VALUES)
        for link, sgn in (("theta", 1.0), ("eta", -1.0)):
            c1, c2 = (computable and _meets(c, ev, sgn) for ev in ETA_VALUES)
            c3 = bool(computable and sgn * c["delta_sens_eta_b"] > 0)
            cond[link] = {"1_bounded": c1, "2_gaussian": c2, "3_sensitivity": c3, "win": c1 and c2 and c3,
                          "value_dependent": c1 != c2, "outcome_dependent": c1 and c2 and not c3}
            if c1 and c2 and c3:
                wins[link].append(c["comparison"])
        c["conditions"] = cond
        if not computable:
            label = "not computable"
        else:
            parts = []
            for link, name in (("theta", "θ"), ("eta", "η")):
                if cond[link]["win"]:
                    parts.append(f"{name} wins")
                if cond[link]["value_dependent"]:
                    parts.append(f"value-dependent ({name}: condition {'1' if cond[link]['1_bounded'] else '2'} "
                                 f"only)")
                if cond[link]["outcome_dependent"]:
                    parts.append(f"outcome-dependent ({name})")
            label = "; ".join(parts) if parts else "no win"
        c["label"] = label
    if wins["theta"] and not wins["eta"]:
        outcome = "gate 2 passed (θ), behavioural referees"
    elif wins["eta"] and not wins["theta"]:
        outcome = "η wins"
    elif wins["theta"] and wins["eta"]:
        outcome = "mixed"
    else:
        outcome = "tie"
    return {"outcome": outcome, "theta_wins": wins["theta"], "eta_wins": wins["eta"]}


# ================================================================================================ instrument checks
def ic_results(runs: Sequence[Run], replays: Sequence[dict] | None) -> dict:
    """IC-1 to IC-5 (§3)."""
    ic: dict[str, Any] = {}
    # IC-1
    if replays is None:
        ic["IC-1"] = {"skipped": True, "passed": None}
    else:
        ok = [bool(r.get("passed")) for r in replays]
        ic["IC-1"] = {"skipped": False, "n_replayed": len(replays), "n_passed": sum(ok),
                      "passed": bool(ok) and all(ok), "objects": list(replays)}
    # IC-2(a)
    worst = {"M_D": 0.0, "C_D": 0.0}
    worst_at = {"M_D": None, "C_D": None}
    n_pairs, regen_ok, n_regen = 0, True, 0
    for r in runs:
        sel = r.z["ic2a_idx"]
        if sel.size == 0:
            continue
        mine = r.z["theta_sampled"][:, sel, :]  # (2, pairs, sets)
        lib = np.transpose(r.z["ic2a_lib"], (1, 0, 2))
        for i, ref in enumerate(("M_D", "C_D")):
            dmax = float(np.max(np.abs(mine[i] - lib[i])))
            if dmax > worst[ref]:
                worst[ref] = dmax
                j = np.unravel_index(np.argmax(np.abs(mine[i] - lib[i])), mine[i].shape)
                worst_at[ref] = {"object_id": r.oid, "step": int(r.steps[sel[j[0]]]), "set": list(r.sets[j[1]]),
                                 "stage1": float(mine[i][j]), "library": float(lib[i][j])}
        n_pairs += sel.size
        regen_ok &= bool(np.all(r.z["ic2a_regen_ok"]))
        n_regen += int(np.sum(r.z["ic2a_regen_ok"]))
    ic["IC-2a"] = {"n_pairs": n_pairs, "max_abs_diff_theta_M_D": worst["M_D"], "max_abs_diff_theta_C_D": worst["C_D"],
                   "worst": worst_at, "tolerance": IC_TOL,
                   "passed": bool(n_pairs > 0 and max(worst.values()) <= IC_TOL),
                   "extra_counts_regenerated_from_probe_seed": {"n_ok": n_regen, "n": n_pairs, "passed": regen_ok}}
    # IC-2(b)
    b: dict[str, Any] = {}
    for region in sorted({prereg_region(r.task) for r in runs}):
        rr = [r for r in runs if prereg_region(r.task) == region]
        row: dict[str, Any] = {"config_regions": sorted({r.region for r in rr})}
        for i, (ref, ri) in enumerate((("M_D", 0), ("C_D", 2))):
            errs = [(r.z["theta_sampled"][i] - r.z["theta"][ri]).ravel() for r in rr]
            e = np.concatenate(errs)
            per_cfg = {}
            for cfgname in sorted({r.config for r in rr}):
                ec = np.concatenate([err for r, err in zip(rr, errs, strict=True) if r.config == cfgname])
                per_cfg[cfgname] = float(np.mean(np.abs(ec) <= IC2_BAND))
            row[f"theta_{ref}"] = {"n_points": int(e.size), "median_signed_error": float(np.median(e)),
                                   "passed": bool(abs(float(np.median(e))) <= IC2_MEDIAN),
                                   "share_within_0.002": float(np.mean(np.abs(e) <= IC2_BAND)),
                                   "share_within_0.002_per_configuration": per_cfg,
                                   "mean_signed_error": float(np.mean(e)), "max_abs_error": float(np.max(np.abs(e)))}
        b[region] = row
    ic["IC-2b"] = {"regions": b, "passed": all(v["passed"] for row in b.values() for v in row.values()
                                               if isinstance(v, dict))}
    # IC-3
    rows3 = []
    for r in runs:
        j = json.loads(str(r.z["ic3_json"]))
        if j is not None:
            rows3.append({"object_id": r.oid, "configuration": r.config, "seed": r.seed, **j})
    ic["IC-3"] = {"n_corpora": len(rows3), "passed": bool(rows3) and all(x["n_fail"] == 0 for x in rows3),
                  "max_abs_z": max((x["max_abs_z"] for x in rows3), default=math.nan),
                  "tv_by_corpus": rows3,
                  "extra_pmf_and_truth_match_stored": all(x["pmf_max_diff"] <= 1e-12 and x["truth_max_diff"] <= 1e-12
                                                          for x in rows3)} if rows3 else {"n_corpora": 0,
                                                                                            "passed": None}
    # IC-4
    uni = [r for r in runs if (r.task == "markov_lang" and r.context == "iid") or
           (r.task == "sparse_parity" and r.rho == 0.0)]
    dmax, at = 0.0, None
    for r in uni:
        for val in VALUES:
            for a, bb in (("M_D", "M_U"), ("C_D", "C_U")):
                dd = np.abs(r.measure(a, val) - r.measure(bb, val))
                if np.nanmax(dd) > dmax or (not np.all(np.isfinite(dd))):
                    dmax = float(np.nanmax(dd)) if np.any(np.isfinite(dd)) else math.inf
                    at = {"object_id": r.oid, "value": val, "pair": f"{a}-{bb}"}
    lmax, lat = 0.0, None
    for r in runs:
        if "local_ri" not in r.z:
            continue
        dd = np.abs(r.measure("M_D", "theta") - r.z["local_ri"])
        if float(np.max(dd)) > lmax:
            lmax = float(np.max(dd))
            j = np.unravel_index(np.argmax(dd), dd.shape)
            lat = {"object_id": r.oid, "step": int(r.steps[j[0]]), "set": list(r.sets[j[1]]),
                   "theta_M_D": float(r.measure("M_D", "theta")[j]), "local_ri": float(r.z["local_ri"][j])}
    ic["IC-4"] = {"uniform_D_runs": len(uni), "max_abs_D_minus_U": dmax, "worst": at,
                  "passed_D_equals_U": bool(uni) and dmax <= IC_TOL,
                  "local_ri_runs": sum("local_ri" in r.z for r in runs), "max_abs_theta_M_D_minus_local_ri": lmax,
                  "local_ri_worst": lat, "passed_local_ri": lmax <= IC_TOL, "tolerance": IC_TOL}
    ic["IC-4"]["passed"] = bool(ic["IC-4"]["passed_D_equals_U"] and ic["IC-4"]["passed_local_ri"])
    # IC-5
    rows5 = [{"object_id": r.oid, "configuration": r.config, "seed": r.seed, "value": r.ic5_value,
              "passed": r.ic5_pass} for r in runs]
    share = {}
    for task, need in IC5_SHARE.items():
        tr = [r for r in runs if r.task == task]
        if tr:
            s = float(np.mean([r.ic5_pass for r in tr]))
            share[task] = {"n": len(tr), "n_passed": int(sum(r.ic5_pass for r in tr)), "share": s, "required": need,
                           "shortfall": s < need}
    ic["IC-5"] = {"criterion": {"markov_lang": f"final KL under D < {IC5_KL}",
                                "sparse_parity": f"final accuracy under D >= {IC5_ACC}"},
                  "share": share, "failing_runs": [x for x in rows5 if not x["passed"]], "runs": rows5}
    ic["bug_checks_passed"] = {k: ic[k]["passed"] for k in ("IC-1", "IC-2a", "IC-2b", "IC-3", "IC-4")}
    return ic


# ================================================================================================ questions (§5)
def q1(runs: Sequence[Run]) -> dict:
    sel = [r for r in runs if r.ic5_pass and ((r.task == "markov_lang" and r.context == "chain") or
                                               (r.task == "sparse_parity" and r.rho in (0.5, 0.8)))]
    per_run = []
    for r in sel:
        spa = r.spurious(1)
        row = {"object_id": r.oid, "configuration": r.config, "seed": r.seed, "region": r.region}
        for ref in REFS:
            for link, val in (("theta", "theta"), ("eta", "eta_b")):
                row[f"{ref}:{link}"] = int(np.sum(r.measure(ref, val)[-1, spa] >= PRESENCE))
        per_run.append(row)

    def score(rows):
        out = {"n_runs": len(rows)}
        for link in ("theta", "eta"):
            if not rows:
                out[link] = None
                continue
            c = {ref: np.array([x[f"{ref}:{link}"] for x in rows]) for ref in REFS}
            md_any = float(np.mean(c["M_D"] >= 1))
            cd_le = float(np.mean(c["C_D"] <= c["M_D"]))
            cu_ge = float(np.mean(c["C_U"] >= c["C_D"]))
            out[link] = {"mean_counts": {ref: float(c[ref].mean()) for ref in REFS},
                         "total_counts": {ref: int(c[ref].sum()) for ref in REFS},
                         "share_M_D_at_least_one": md_any, "pred_M_D": md_any >= 0.80,
                         "share_C_D_le_M_D": cd_le, "C_D_total_smaller": int(c["C_D"].sum()) < int(c["M_D"].sum()),
                         "pred_C_D": cd_le >= 0.90 and int(c["C_D"].sum()) < int(c["M_D"].sum()),
                         "share_C_U_ge_C_D": cu_ge, "pred_C_U": cu_ge >= 0.90}
        return out
    return {"pooled": score(per_run),
            "by_region": {g: score([x for x in per_run if x["region"] == g]) for g in sorted({x["region"]
                                                                                              for x in per_run})},
            "per_run": per_run}


def q2(runs: Sequence[Run]) -> dict:
    sel = [r for r in runs if r.ic5_pass]
    counts: dict[str, Any] = {}
    nb_s, nb_e = len(Q2_SAT_EDGES) - 1, len(Q2_ENERGY_EDGES) - 1
    hist = {ev: np.zeros((nb_s, nb_e), np.int64) for ev in ETA_VALUES}
    pred1 = {ev: [0, 0] for ev in ETA_VALUES}
    per_run_cd = {ev: {} for ev in ETA_VALUES}
    for r in sel:
        s2 = np.array([len(A) >= 2 for A in r.sets])
        group = f"{r.task}:{r.context}" if r.task == "markov_lang" else f"{r.region}:rho={r.rho:g}"
        for ref in REFS:
            th = r.measure(ref, "theta")[:, s2]
            for ev in ETA_VALUES:
                et = r.measure(ref, ev)[:, s2]
                t_only = (th >= DIVERGE) & (et < PRESENCE)
                e_only = (et >= DIVERGE) & (th < PRESENCE)
                key = f"{ref}:{ev}"
                g = counts.setdefault(key, {}).setdefault(group, {"theta_only": 0, "eta_only": 0, "points": 0})
                g["theta_only"] += int(t_only.sum())
                g["eta_only"] += int(e_only.sum())
                g["points"] += int(th.size)
                if ref == "C_D":
                    div = t_only | e_only
                    sat = np.broadcast_to(r.z["saturation"][:, None], div.shape)[div]
                    en = r.z["lo_energy"][:, s2][div]
                    hist[ev] += np.histogram2d(sat, en, bins=[Q2_SAT_EDGES, Q2_ENERGY_EDGES])[0].astype(np.int64)
                    pred1[ev][0] += int(np.sum((sat >= 0.74) & (en >= 0.3)))
                    pred1[ev][1] += int(div.sum())
                    per_run_cd[ev][r.oid] = (r, int(div.sum()))
    out: dict[str, Any] = {"counts": counts,
                           "bins": {"saturation": list(Q2_SAT_EDGES), "energy": list(Q2_ENERGY_EDGES)}}
    for ev in ETA_VALUES:
        n_ok, n_all = pred1[ev]
        share = n_ok / n_all if n_all else math.nan
        strong = [n for (r, n) in per_run_cd[ev].values() if r.task == "markov_lang" and r.couplings in Q2_STRONG]
        weak = [n for (r, n) in per_run_cd[ev].values() if r.task == "markov_lang" and r.couplings in Q2_WEAK]
        ms = float(np.mean(strong)) if strong else math.nan
        mw = float(np.mean(weak)) if weak else math.nan
        out[f"C_D_{ev}"] = {"n_divergences": n_all, "histogram_sat_x_energy": hist[ev].tolist(),
                            "share_sat_ge_0.74_and_energy_ge_0.3": share,
                            "pred_share_ge_0.95": (share >= 0.95) if n_all else None,
                            "strong_runs": len(strong), "strong_per_run": ms, "weak_runs": len(weak),
                            "weak_per_run": mw,
                            "pred_strong_more": (ms > mw) if (strong and weak) else None}
    out["predictions_scored_with"] = "C_D, eta_b (bounded); C_D eta_g reported"
    return out


def q3(runs: Sequence[Run]) -> dict:
    A = tuple(lag - 1 for lag in Q3_LAGS)
    sel = [r for r in runs if r.ic5_pass and r.task == "markov_lang" and r.couplings == Q3_COUPLINGS]
    out: dict[str, Any] = {"set_inputs": list(A), "per_seed": []}
    for ref in REFS:
        for ev in ETA_VALUES:
            res = {}
            for ctx in ("chain", "iid"):
                rr = [r for r in sel if r.context == ctx]
                hits = []
                for r in rr:
                    j = r.sets.index(A)
                    cond = (r.measure(ref, ev)[:, j] >= DIVERGE) & (r.measure(ref, "theta")[:, j] < PRESENCE)
                    cond[0] = False
                    hit = bool(np.any(cond[1:-1] & cond[2:])) if cond.size > 2 else False
                    hits.append(hit)
                    out["per_seed"].append({"object_id": r.oid, "context": ctx, "seed": r.seed, "reference": ref,
                                            "eta": ev, "divergent_checkpoints": int(cond.sum()),
                                            "two_consecutive": hit})
                res[ctx] = {"n_seeds": len(rr), "share": float(np.mean(hits)) if hits else math.nan}
            if ref == "M_U":
                res["prediction"] = (res["chain"]["share"] >= 0.80 and res["iid"]["share"] <= 0.20) \
                    if res["chain"]["n_seeds"] and res["iid"]["n_seeds"] else None
            out[f"{ref}:{ev}"] = res
    return out


def q4(runs: Sequence[Run]) -> dict:
    from scipy.stats import binom

    idx = {(r.couplings, r.context, r.seed): r for r in runs if r.task == "markov_lang"}
    pairs = []
    for (c, ctx, s), rc in sorted(idx.items()):
        if ctx != "chain" or (c, "iid", s) not in idx:
            continue
        ri = idx[(c, "iid", s)]
        if not (rc.ic5_pass and ri.ic5_pass):
            continue
        spa = rc.spurious(1)
        pairs.append({"couplings": c, "seed": s, **{
            val: float(rc.measure("C_U", val)[-1, spa].sum() - ri.measure("C_U", val)[-1, spa].sum())
            for val in VALUES}})
    out: dict[str, Any] = {"n_pairs": len(pairs), "pairs": pairs}
    for val in VALUES:
        d = np.array([p[val] for p in pairs])
        npos, nneg = int(np.sum(d > 0)), int(np.sum(d < 0))
        n = npos + nneg
        p = float(binom.sf(npos - 1, n, 0.5)) if n else math.nan
        out[val] = {"n_positive": npos, "n_negative": nneg, "n_ties": int(np.sum(d == 0)), "p_one_sided": p,
                    "median_difference": float(np.median(d)) if d.size else math.nan,
                    "prediction": (p < 0.05) if n else None}
    return out


def q5(runs: Sequence[Run]) -> dict:
    rows = []
    for r in runs:
        if not r.ic5_pass:
            continue
        j = json.loads(str(r.z["q5_json"]))
        rows.append({"object_id": r.oid, "configuration": r.config, "group": f"{r.region}" + (
            f":{r.context}" if r.context else f":rho={r.rho:g}"), **{k: v for k, v in j.items() if "_sets" not in k}})
    out: dict[str, Any] = {"exploratory": True, "n_runs": len(rows)}
    for K in Q5_K:
        for grp in sorted({x["group"] for x in rows}) + ["all"]:
            rr = rows if grp == "all" else [x for x in rows if x["group"] == grp]
            res = {}
            for val in VALUES:
                v = np.array([x[f"{val}_K{K}_kl_S"] for x in rr])
                res[f"mean_kl_S_{val}"] = float(v.mean()) if v.size else math.nan
                res[f"median_kl_S_{val}"] = float(np.median(v)) if v.size else math.nan
            for ev in ETA_VALUES:
                th = np.array([x[f"theta_K{K}_kl_S"] for x in rr])
                et = np.array([x[f"{ev}_K{K}_kl_S"] for x in rr])
                res[f"share_theta_better_than_{ev}"] = float(np.mean(th < et)) if th.size else math.nan
                res[f"share_tied_with_{ev}"] = float(np.mean(th == et)) if th.size else math.nan
            out[f"K{K}:{grp}"] = res
    out["per_run"] = rows
    return out


# ================================================================================================ driver
def select_replays(records: Sequence[dict], fraction: float, seed: int = BOOT_SEED) -> list[dict]:
    """IC-1: ceil(fraction · n) done objects of each §2 region (at least one), drawn with ``seed``."""
    rng = np.random.default_rng(seed)
    out = []
    for region in sorted({prereg_region(r["task"]) for r in records}):
        done = sorted((r for r in records if prereg_region(r["task"]) == region and r.get("status") == "done"),
                      key=lambda r: r["object_id"])
        if not done or fraction <= 0:
            continue
        k = min(len(done), max(1, math.ceil(fraction * len(done))))
        out += [done[i] for i in sorted(rng.choice(len(done), size=k, replace=False))]
    return out


def timing_summary(runs: Sequence[Run]) -> dict:
    out = {}
    for region in sorted({r.region for r in runs}):
        rr = [r for r in runs if r.region == region]
        per = np.array([float(r.z["seconds_measures"]) / r.steps.size for r in rr])
        tot = np.array([float(r.z["seconds_total"]) for r in rr])
        out[region] = {"n_objects": len(rr), "measures_s_per_checkpoint_mean": float(per.mean()),
                       "measures_s_per_checkpoint_median": float(np.median(per)),
                       "measures_s_per_checkpoint_max": float(per.max()),
                       "stage1_total_s_per_object_mean": float(tot.mean()),
                       "ic2a_s_per_object_mean": float(np.mean([float(r.z["seconds_ic2a"]) for r in rr])),
                       "ic4_s_per_object_mean": float(np.mean([float(r.z["seconds_ic4"]) for r in rr])),
                       "ic3_s_per_object_mean": float(np.mean([float(r.z["seconds_ic3"]) for r in rr])),
                       "mean_newton_iterations_by_ref": dict(zip(
                           REFS, np.mean([r.z["iters"].mean(axis=1) for r in rr], axis=0).tolist(), strict=True))}
    return out


def analyse(manifest: str | os.PathLike, cache_dir: str | os.PathLike, *, B: int = N_BOOT,
            min_seeds: int = MIN_SEEDS, replays: Sequence[dict] | None = None) -> dict:
    """Stage 2 on the stage-1 caches of the done objects of ``manifest``."""
    manifest = Path(manifest)
    t0 = time.perf_counter()
    records = load_manifest(manifest)
    runs = build_runs(records, cache_dir)
    fam_seeds = {
        "T1_markov": sorted({r.seed for r in runs if r.task == "markov_lang" and r.context == "chain"}),
        "T1_parity": sorted({r.seed for r in runs if r.task == "sparse_parity" and r.rho in (0.5, 0.8)}),
        "T2": sorted({r.seed for r in runs if r.task == "markov_lang" and r.couplings in
                      {c for p in TWIN_PAIRS for c in p[:2]}}),
    }
    cfgs = {"T1_markov": build_t1(runs, "T1_markov", fam_seeds["T1_markov"]),
            "T1_parity": build_t1(runs, "T1_parity", fam_seeds["T1_parity"])}
    cfgs["T2"], t2_checks = build_t2(runs, fam_seeds["T2"], manifest)
    comps = []
    for fam in ("T1_markov", "T1_parity", "T2"):
        comps += evaluate_family(fam, cfgs[fam], fam_seeds[fam], B=B, min_seeds=min_seeds)
    decision = apply_decision(comps)
    ident = identifiability_table()
    for row in ident:
        for ref in ("M_D", "C_D"):
            if f"prereg_{ref}" in row:
                row[f"match_{ref}"] = prereg_matches(row[ref], row[f"prereg_{ref}"])
    exclusions = {
        "not_done": [{"object_id": r["object_id"], "region": r["region"], "status": r.get("status")}
                     for r in records if r.get("status") != "done"],
        "ic5_failures": [{"object_id": r.oid, "configuration": r.config, "seed": r.seed, "value": r.ic5_value}
                         for r in runs if not r.ic5_pass],
    }
    for fam, cl in cfgs.items():
        exclusions[fam] = [{"name": c.name, "n_valid_seeds": c.n_valid, "seeds_in": c.seeds_in,
                            "excluded_ic5": c.excluded_ic5, "no_window": c.no_window, "missing_twin": c.missing,
                            "kept_by_reference": {ref: bool(k[i]) for ref, k in
                                                  _family_kept(cl, fam, min_seeds).items()},
                            **({"resid_var": c.info["resid"]} if fam == "T2" else {})}
                           for i, c in enumerate(cl)]
    out = {
        "preregistration": "docs/results/v2a_preregistration.md",
        "settings": {"bootstrap_B": B, "bootstrap_seed": BOOT_SEED, "min_seeds": min_seeds, "margin": MARGIN,
                     "alpha": ALPHA, "presence": PRESENCE, "divergence_high": DIVERGE,
                     "identifiability_min": IDENT_MIN, "seed_labels": fam_seeds},
        "n_objects": {"manifest": len(records), "done": len(runs)},
        "instrument_checks": ic_results(runs, replays),
        "identifiability": ident,
        "t2_consistency_checks": t2_checks,
        "comparisons": comps,
        "decision": decision,
        "exclusions": exclusions,
        "questions": {"Q1": q1(runs), "Q2": q2(runs), "Q3": q3(runs), "Q4": q4(runs), "Q5": q5(runs)},
        "timing_stage1": timing_summary(runs),
        "DECISIONS": list(DECISIONS),
    }
    out["timing_stage2_s"] = time.perf_counter() - t0
    return out


def jsonable(o: Any) -> Any:
    """JSON-friendly copy: numpy scalars and arrays to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o
