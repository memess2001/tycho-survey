"""Known answers under correlated probes for sky v2 Part A: probe distributions, constructed teachers, exact
θ and η measures, their estimates, and calibration experiment E8.

A *teacher* is a conditional P(ŷ = 1 | x) over binary inputs x ∈ {0, 1}^d, given on all 2^d patterns: what a network
with known weights computes, without training. The first ``d_table`` inputs are the *table inputs* (the variables
the instrument sees). The remaining ``d − d_table`` are *nuisance inputs* that the teacher may use but the
instrument never sees. They are uniform and independent of everything, and they are what makes the estimates noisy:
with a soft output and every input in the table, a cell mean is exact. A *probe distribution* is a pmf over the
table inputs: uniform, a binary Markov chain (spin correlation ρ^|i−j|), an equicorrelated Ising model (pairwise
spin correlation ρ), or a product of independent blocks of these. For a teacher and a probe distribution D the joint
table P(x_table, ŷ) = Σ_z D(x_table) 2^−d_z P(ŷ | x_table, z) is exact, so every measure of :mod:`tycho.ri.cells`
has an exact population value. Inputs use spin s = 2x − 1, and m = 2 p1 − 1.

Measures for an input subset A (exact: :func:`exact_measures`; estimated: :func:`estimate_measures`):

* ``ri_marg``: local (marginal) RI of A ∪ {ŷ} under D (the survey's spectral line).
* ``ri_cond``: conditional RI of A ∪ {ŷ} given all d inputs, under D.
* ``w``: logit coefficient of A, the Walsh coefficient of atanh(m) on A over all inputs. It is a property of the
  teacher, not of D, and it is the θ coordinate that ``ri_cond`` tests (``ri_cond`` = 0 when ``w`` = 0 for A and all
  its supersets).
* ``f_u``: Walsh coefficient of m on A under uniform inputs, a property of the teacher. It is estimated on separate
  uniform probes with the uniform-cell estimator.
* ``m_d``: data moment E_D[m χ_A], which depends on D.
* ``f0_u``, ``f0_d``: mean outputs on uniform and data probes.
* Information values (nats) on one scale: ``ri_marg`` and ``ri_cond`` directly; ``eta_u`` =
  eta_information(f_u, f0_u) (uniform inputs, so χ_A is balanced); ``eta_d`` = the mutual information between ŷ and
  the parity feature χ_A under D (:func:`tycho.ri.cells.parity_information`). The last is always defined, whereas
  eta_information(m_d, f0_d) is not when χ_A is unbalanced.

E8 (null calibration under correlated probes): for marginal and conditional RI, the false-positive rate of the χ²_1
p-value and the noise floor (quantiles of the plug-in and corrected values) when the population value is exactly 0,
over probe distributions, table sizes, sample sizes and nuisance strengths. Two designs.

*Correlated nulls*, for conditional RI only. All table inputs form one correlated block. The teacher uses a random
half of them (additive log-odds plus random pair terms) and the nuisance inputs. Each tested set (one, two or three
inputs) contains an unused input, which is correlated with used ones. Its conditional RI is exactly 0; its marginal
RI is not (that is the spurious case conditional RI exists for).

*Block nulls*, for both RI variants. The table inputs form two independent blocks of equal size, each correlated
inside according to the probe kind. The teacher uses the first block (additive
log-odds and random pair terms) and four nuisance inputs, and ignores the second block. So every subset that
contains a second-block input has conditional RI exactly 0. Every subset made only of second-block inputs also has
marginal RI exactly 0. The exact population values of the tested subsets are computed and reported
(``max_population_value``).
"""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from tycho.ri.cells import (cell_stats, conditional_ri_plugin, eta_information, gate, local_ri_of,
                            parity_information)
from tycho.ri.estimators import bh_fdr
from tycho.ri.refined import local_ri

__all__ = [
    "patterns", "uniform_pmf", "markov_pmf", "equicorrelated_pmf", "probe_pmf", "product_pmf",
    "Teacher", "logit_teacher", "walsh_teacher", "soft_and_teacher",
    "exact_measures", "estimate_measures", "sample_rows", "table_mean", "detect", "e8_cell", "rng_for",
]


def patterns(d: int) -> np.ndarray:
    """(2^d, d) 0/1 patterns in C order (first input most significant)."""
    return np.array(list(itertools.product((0, 1), repeat=d)), dtype=np.int64).reshape(-1, d)


def uniform_pmf(d: int) -> np.ndarray:
    return np.full(1 << d, 1.0 / (1 << d))


def markov_pmf(d: int, rho: float) -> np.ndarray:
    """Stationary symmetric binary Markov chain: x_1 uniform, x_{t+1} = x_t with probability (1 + ρ)/2, so the spin
    correlation of inputs i and j is ρ^|i−j|."""
    s = patterns(d) * 2 - 1
    stay = (1 + rho) / 2
    p = np.full(len(s), 0.5)
    for t in range(1, d):
        p *= np.where(s[:, t] == s[:, t - 1], stay, 1 - stay)
    return p / p.sum()


def _ising_equicorr(d: int, J: float) -> np.ndarray:
    s = patterns(d) * 2 - 1
    tot = s.sum(axis=1)
    e = J * (tot ** 2 - d) / 2.0
    p = np.exp(e - e.max())
    return p / p.sum()


def equicorrelated_pmf(d: int, rho: float) -> np.ndarray:
    """Ising model P(s) ∝ exp(J Σ_{i<j} s_i s_j) with J chosen so that every pair has spin correlation ρ (0 ≤ ρ < 1)."""
    if rho == 0:
        return uniform_pmf(d)
    s = patterns(d) * 2 - 1

    def corr(J):
        p = _ising_equicorr(d, J)
        return float((p * s[:, 0] * s[:, 1]).sum()) - rho
    J = brentq(corr, 0.0, 10.0)
    return _ising_equicorr(d, J)


def product_pmf(*pmfs: np.ndarray) -> np.ndarray:
    """pmf of independent blocks (the first block's inputs first), in the C order of :func:`patterns`."""
    out = np.ones(1)
    for p in pmfs:
        out = np.outer(out, p).reshape(-1)
    return out


def probe_pmf(kind: str, d: int, rho: float = 0.0) -> np.ndarray:
    if kind == "uniform":
        return uniform_pmf(d)
    if kind == "markov":
        return markov_pmf(d, rho)
    if kind == "equicorr":
        return equicorrelated_pmf(d, rho)
    raise ValueError(f"unknown probe kind {kind!r}")


@dataclass
class Teacher:
    """A conditional P(ŷ = 1 | x) on all 2^d patterns, with the sets that are planted in its log-odds."""

    d: int
    p1: np.ndarray
    name: str
    planted: list[tuple[int, ...]]

    @property
    def mean(self) -> np.ndarray:
        return 2 * self.p1 - 1


def _chi(s: np.ndarray, A: Sequence[int]) -> np.ndarray:
    return np.prod(s[:, list(A)], axis=1) if len(A) else np.ones(len(s))


def logit_teacher(d: int, w: dict[tuple[int, ...], float], name: str = "logit") -> Teacher:
    """log-odds / 2 = Σ_T w_T χ_T(s), so m = tanh(Σ w χ) and ``w`` are exactly the logit coefficients."""
    s = patterns(d) * 2 - 1
    z = sum(v * _chi(s, T) for T, v in w.items())
    p1 = (1 + np.tanh(z)) / 2
    return Teacher(d, p1, name, sorted({tuple(sorted(T)) for T, v in w.items() if v != 0 and T}))


def walsh_teacher(d: int, f: dict[tuple[int, ...], float], name: str = "walsh") -> Teacher:
    """m = Σ_B f_B χ_B(s) (must stay in [−1, 1])."""
    s = patterns(d) * 2 - 1
    m = sum(v * _chi(s, B) for B, v in f.items())
    if np.any(np.abs(m) > 1 + 1e-12):
        raise ValueError("Walsh coefficients give |m| > 1")
    return Teacher(d, (1 + np.clip(m, -1, 1)) / 2, name, sorted({tuple(sorted(B)) for B, v in f.items() if v and B}))


def soft_and_teacher(d: int, inputs: Sequence[int], eps: float) -> Teacher:
    """P(ŷ = 1) = eps + (1 − 2 eps) Π_{i ∈ inputs} x_i: a probability-scale product (soft AND)."""
    x = patterns(d)
    prod = np.prod(x[:, list(inputs)], axis=1)
    return Teacher(d, eps + (1 - 2 * eps) * prod, f"and(eps={eps})", [tuple(inputs)])


def _walsh_on_uniform(values: np.ndarray, d: int, A: Sequence[int]) -> float:
    s = patterns(d) * 2 - 1
    return float((values * _chi(s, A)).mean())


def table_mean(teacher: Teacher, d_table: int) -> np.ndarray:
    """m(x_table) = E_z[2 p1 − 1] over the uniform nuisance inputs, on the 2^d_table table patterns."""
    d_z = teacher.d - d_table
    return teacher.mean.reshape(1 << d_table, 1 << d_z).mean(axis=1)


def exact_measures(pmf: np.ndarray, teacher: Teacher, subsets: Sequence[Sequence[int]], *,
                   d_table: int | None = None, with_cond: bool = True) -> list[dict]:
    """Exact population value of every measure for each subset of the table inputs (see the module docstring).
    ``pmf`` is over the table inputs; the nuisance inputs (if any) are uniform and averaged out."""
    d = teacher.d if d_table is None else d_table
    s = patterns(d) * 2 - 1
    m = table_mean(teacher, d)
    p1 = (1 + m) / 2
    P = np.stack([pmf * (1 - p1), pmf * p1], axis=-1).reshape((2,) * d + (2,))
    z = np.arctanh(np.clip(m, -1 + 1e-15, 1 - 1e-15))
    f0_u = float(m.mean())
    f0_d = float((pmf * m).sum())
    out = []
    for A in subsets:
        A = tuple(A)
        marg = np.moveaxis(P, [*A, d], list(range(len(A) + 1)))
        marg = marg.reshape(marg.shape[:len(A) + 1] + (-1,)).sum(axis=-1)
        f_u = _walsh_on_uniform(m, d, A)
        m_d = float((pmf * m * _chi(s, A)).sum())
        chiA = _chi(s, A)
        tab = np.array([[(pmf[chiA == c] * (1 - p1[chiA == c])).sum(), (pmf[chiA == c] * p1[chiA == c]).sum()]
                        for c in (-1, 1)])
        prod = tab.sum(axis=1, keepdims=True) * tab.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta_d = float(max(np.where(tab > 0, tab * np.log(tab / prod), 0.0).sum(), 0.0))
        out.append(dict(
            subset=list(A), ri_marg=float(local_ri(marg)),
            ri_cond=(conditional_ri_plugin(patterns(d), p1, A, range(d), weights=pmf) if with_cond
                     else math.nan),
            w=_walsh_on_uniform(z, d, A), f_u=f_u, m_d=m_d, f0_u=f0_u, f0_d=f0_d,
            eta_u=float(eta_information(f_u, f0_u)), eta_d=eta_d,
            planted=A in teacher.planted))
    return out


def sample_rows(pmf: np.ndarray, teacher: Teacher, d_table: int, n: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n probe rows: table inputs from ``pmf``, nuisance inputs uniform; returns (table codes, soft p1)."""
    d_z = teacher.d - d_table
    it = rng.choice(len(pmf), size=n, p=pmf)
    iz = rng.integers(0, 1 << d_z, size=n) if d_z else np.zeros(n, dtype=np.int64)
    return patterns(d_table)[it], teacher.p1[(it << d_z) + iz]


def estimate_measures(pmf: np.ndarray, teacher: Teacher, subsets: Sequence[Sequence[int]], n: int,
                      rng: np.random.Generator, *, d_table: int | None = None, n_uniform: int | None = None,
                      with_cond: bool = True) -> list[dict]:
    """Plug-in estimates of every measure from ``n`` data probes (table inputs drawn from ``pmf``) and
    ``n_uniform`` uniform probes (default n), with the teacher's soft output, as the survey would record them."""
    d = teacher.d if d_table is None else d_table
    Xd, pd_ = sample_rows(pmf, teacher, d, n, rng)
    Xu, pu = sample_rows(uniform_pmf(d), teacher, d, n_uniform or n, rng)
    subs = [tuple(A) for A in subsets]
    cd = cell_stats(Xd, pd_, subs)
    cu = cell_stats(Xu, pu, subs)
    ri_marg = local_ri_of(cd)
    out = []
    for A, csd, csu, rm in zip(subs, cd, cu, ri_marg):
        top = tuple(range(len(A)))
        m_d = csd.data_moments()[top]
        f_u = csu.uniform_cell_walsh()[top]
        f0_d, f0_u = csd.mean_output(), csu.mean_output()
        rc = conditional_ri_plugin(Xd, pd_, A, range(d)) if with_cond else math.nan
        out.append(dict(subset=list(A), ri_marg=float(rm), ri_cond=float(rc), f_u=float(f_u), m_d=float(m_d),
                        f0_u=f0_u, f0_d=f0_d, eta_u=float(eta_information(f_u, f0_u)),
                        eta_d=parity_information(csd), w=float(csu.logit_walsh()[top])))
    return out


MEASURES = ("ri_marg", "ri_cond", "eta_u", "eta_d")


def detect(est: list[dict], n: int, min_effect: dict[str, float], *, alpha: float = 0.01) -> dict[str, np.ndarray]:
    """Detections per measure within one family of subsets: BH q ≤ alpha and corrected value ≥ min_effect[measure].
    All four measures use the same gate (corrected = value − 1/(2n), p = χ²_1 of 2n·value)."""
    out = {}
    for meas in MEASURES:
        v = np.array([e[meas] for e in est], dtype=np.float64)
        corr, p = gate(v, n, min_effect[meas])
        rej, q = bh_fdr(p, alpha)
        out[meas] = (q <= alpha) & (corr >= min_effect[meas]) & np.isfinite(v)
    return out


def rng_for(seed: int, name: str) -> np.random.Generator:
    """Independent stream per named cell (reproducible regardless of order or parallelism)."""
    h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, h]))


# ---------------------------------------------------------------------------------------------------- E8
N_NUISANCE = 4


def _e8_teacher(u: int, v: int, nuisance_scale: float, rng: np.random.Generator) -> Teacher:
    """Uses block 1 (inputs 0..u−1: additive log-odds plus random pair terms) and N_NUISANCE nuisance inputs (after
    the table inputs); ignores block 2 (inputs u..u+v−1)."""
    d = u + v + N_NUISANCE
    w: dict[tuple[int, ...], float] = {(i,): float(rng.standard_normal()) for i in range(u)}
    for pr in itertools.combinations(range(u), 2):
        if rng.random() < 0.3:
            w[pr] = 0.5 * float(rng.standard_normal())
    for k in range(N_NUISANCE):
        w[(u + v + k,)] = nuisance_scale * float(rng.standard_normal())
    return logit_teacher(d, w, "e8-null")


def e8_cell(probe: str, rho: float, d_table: int, n: int, nuisance_scale: float, reps: int, seed: int) -> dict:
    """One E8 grid cell (see the module docstring): null replicates of conditional RI (all table inputs in the table)
    and marginal RI (subsets of the unused block), with χ²_1 p-values and corrected values."""
    name = f"E8|{probe}|rho={rho}|d={d_table}|n={n}|nuisance={nuisance_scale}"
    rng = rng_for(seed, name)
    u = v = d_table // 2
    block = probe_pmf(probe, u, rho)
    pmf = product_pmf(block, probe_pmf(probe, v, rho))
    tests_cond = [(u,), (u + 1,), (0, u), (u, u + 1)][: 4 if v >= 2 else 2]
    tests_marg = [(u,), (u, u + 1)] if v >= 2 else [(u,)]
    vals = {"cond_pair": [], "cond_triple": [], "marg_pair": [], "marg_triple": []}
    max_pop = 0.0
    for r in range(reps):
        t = _e8_teacher(u, v, nuisance_scale, rng)
        if r < 3:  # exact check of the null on the first replicates (identical structure in all of them)
            ex = exact_measures(pmf, t, tests_cond, d_table=d_table)
            max_pop = max(max_pop, max(e["ri_cond"] for e in ex))
            ex = exact_measures(pmf, t, tests_marg, d_table=d_table, with_cond=False)
            max_pop = max(max_pop, max(e["ri_marg"] for e in ex))
        X, p1 = sample_rows(pmf, t, d_table, n, rng)
        for A in tests_cond:
            vals["cond_pair" if len(A) == 1 else "cond_triple"].append(
                conditional_ri_plugin(X, p1, A, range(d_table)))
        for A, val in zip(tests_marg, local_ri_of(cell_stats(X, p1, tests_marg))):
            vals["marg_pair" if len(A) == 1 else "marg_triple"].append(float(val))
    # correlated-null design: one correlated block; the teacher uses a random half of it (additive + random pairs)
    # plus the nuisance inputs; tested sets contain an unused input, which is correlated with used ones
    pmf_c = probe_pmf(probe, d_table, rho)
    for k in (1, 2, 3):
        vals[f"cond_corr_order{k + 1}"] = []
    for r in range(reps):
        used = sorted(rng.choice(d_table, size=d_table // 2, replace=False).tolist())
        unused = [i for i in range(d_table) if i not in used]
        w: dict[tuple[int, ...], float] = {(i,): float(rng.standard_normal()) for i in used}
        for pr in itertools.combinations(used, 2):
            if rng.random() < 0.3:
                w[pr] = 0.5 * float(rng.standard_normal())
        for k in range(N_NUISANCE):
            w[(d_table + k,)] = nuisance_scale * float(rng.standard_normal())
        t = logit_teacher(d_table + N_NUISANCE, w, "e8-corr-null")
        j = unused[int(rng.integers(len(unused)))]
        others = [i for i in range(d_table) if i != j]
        tests = [(j,), tuple(sorted((j, int(rng.choice(others))))),
                 tuple(sorted([j] + rng.choice(others, size=2, replace=False).tolist()))]
        if r < 3:
            ex = exact_measures(pmf_c, t, tests, d_table=d_table)
            max_pop = max(max_pop, max(e["ri_cond"] for e in ex))
        X, p1 = sample_rows(pmf_c, t, d_table, n, rng)
        for A in tests:
            vals[f"cond_corr_order{len(A) + 1}"].append(conditional_ri_plugin(X, p1, A, range(d_table)))
    res = dict(name=name, probe=probe, rho=rho, d_table=d_table, n=n, nuisance_scale=nuisance_scale, reps=reps,
               max_population_value=max_pop)
    q = stats.chi2.ppf(0.99, 1) / (2 * n)
    for label, vv in vals.items():
        if not vv:
            continue
        a = np.array(vv)
        corr, p = gate(a, n, 0.0)
        res[label] = dict(
            count=int(len(a)), fpr_alpha_0_01=float((p <= 0.01).mean()), fpr_alpha_0_05=float((p <= 0.05).mean()),
            ks_uniform=float(stats.kstest(p, "uniform").statistic),
            plugin_mean=float(a.mean()), chi2_mean=1 / (2 * n),
            plugin_q99=float(np.quantile(a, 0.99)), chi2_q99=float(q),
            corrected_q99=float(np.quantile(corr, 0.99)), corrected_q999=float(np.quantile(corr, 0.999)))
    return res
