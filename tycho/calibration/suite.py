"""Instrument calibration (SPEC §3): seven experiments that measure the error budget of the RI instrument.

"Build Tycho's instrument before the survey." An anomaly in the catalog means something only relative to a known
instrument error, so every experiment plants a known truth (:mod:`tycho.calibration.planted`), observes it the way the
survey does (the estimators, scans and event detector of :mod:`tycho.ri` and :mod:`tycho.analysis.events`), and
compares. All information values are in nats.

Experiments
-----------
E1 null / false-positive rate
    Tables over exactly k variables with lower-order structure only, so the local RI of the k-set is exactly 0:
    binary tables with spin couplings ``~ Normal(0, strength²)`` on every subset of order < k
    (:func:`~tycho.calibration.planted.random_interactions`), and categorical extras with i.i.d.
    ``Normal(0, strength²)`` potentials on every cell of every margin of order < k. A fresh null table per replicate.
    For each (cards, strength, N): false-positive rate of the χ² p-value at each α, Kolmogorov-Smirnov distance of the
    p-values from U(0, 1), and quantiles of RI_plugin and RI_corrected (the noise floor), next to the χ² prediction
    ``χ²_df quantile / (2N)``. Optionally a parametric-bootstrap null on the first replicates (same data as χ²).
E2 bias / RMSE / CI coverage
    For each order k and target RI: a fresh random lower-order background per replicate (spin scale
    ``e2_background``) and the top-order coupling solved so that the population local RI equals the target
    (:func:`~tycho.calibration.planted.theta_for_local_ri`). Bias, SD and RMSE of RI_plugin and RI_corrected (with the
    first-order predictions bias ``df/(2N)`` and SD ``√(2(df + 2λ))/(2N)``, ``λ = 2N·RI``) and coverage of the
    row-bootstrap percentile CI.
E3 power
    The replicates of E2. A detection is ``q <= alpha and RI_corrected >= min_effect``, where q is the
    Benjamini-Hochberg q-value of the planted subset tested together with ``m − 1`` true nulls whose p-values are
    U(0, 1) (the worst case for BH: no other signal), for each family size m. Rule ``"stat"``: q only. Rule
    ``"survey"``: also RI_corrected >= the recommended min_effect at N (maximum over orders: ``ri_scan`` applies one
    threshold to a whole scan). The minimum detectable RI (MDE) at the target
    power is interpolated in log RI on the isotonic power curve and compared with the noncentral-χ² prediction
    ``λ*/(2N)`` (Bonferroni ``α/m`` approximation of BH).
E4 timing
    ``θ_S(t) = switch_on(t, t*, width, θ_max)`` for the top-order subset of ``e4_order`` binary variables over a fixed
    lower-order background, observed at T checkpoints with N fresh samples each: an
    :func:`~tycho.ri.estimators.ri_scan` of all subsets per checkpoint, assembled into an
    :class:`~tycho.analysis.ricurve.RICurve`, then :func:`~tycho.analysis.events.detect_events`. Onset error =
    detected − true onset, in checkpoints; true onset := first checkpoint whose population RI >= min_effect. The fixed
    ``cfg.min_effect`` defines both the truth and the detector's ``on`` threshold, so the error isolates the estimator.
E5 identity / specificity
    d = 8 binary variables: a planted pure 3-way {0,1,2} and a pairwise {3,4}, plus a weak random singleton and
    pairwise background elsewhere; an ri_scan to order 4 (154 subsets, BH within the scan). Precision and recall of
    the detected set against the planted set. False positives are split into *spurious* (population RI < min_effect),
    *real but unplanted* (population RI >= min_effect) and *related* (a proper sub- or superset of a planted subset:
    an error of order rather than of identity).
E6 discretisation artefacts (RQ0)
    An equicorrelated Gaussian latent (ρ grid), binarised by median, zero (with latent mean ``e6_latent_mean``, i.e.
    a biased pre-activation) or the ``e6_quantile`` quantile. Exact population local RI at orders 2..e6_max_order
    (:func:`~tycho.calibration.planted.dichotomized_gaussian_table`) and what the instrument sees at sample size N
    (mean RI over all subsets of each order, detection rate). The latent has only pairwise structure, so every
    order >= 3 term is an artefact of thresholding. Median thresholding is symmetric under flipping all bits: odd
    orders vanish exactly, even orders do not; the summary checks both.
E7 chain dependence
    Random positive tables with mixed cardinalities (i.i.d. ``Normal(0, e7_scale²)`` log-potentials per cell): chain
    RI of every subset along the graded, reverse-graded (reverse lexicographic within each order), greedy (mode
    interaction selection) and random admissible chains. Spread across chains, comparison with the chain-independent
    local RI, and the check Σ RI = TC for every chain.

Recommended min_effect
----------------------
``min_effect(k, N) = max(min_effect_floor, Q_q(RI_corrected | H0, k, N))`` with ``q = min_effect_null_quantile``,
the quantile taken over E1's binary null replicates (maximum over strengths). The floor is a policy choice
(interactions weaker than it are not interesting at any N), not a calibration output; the quantile term is the
noise floor, so at most a fraction ``1 − q`` of null subsets pass the effect-size filter even before the FDR filter.
Without E1 the quantile comes from the χ² approximation. :func:`interpolate_min_effect` evaluates the table at any N
using the 1/N scaling of the noise floor between and beyond the grid points.

Execution and reproducibility
-----------------------------
Each experiment is split into work items (one per grid cell). Every random stream of an item derives from
``SeedSequence([cfg.seed, sha256(name)])`` where the name identifies the cell, not its position in the grid, so a
cell's result does not depend on ``n_jobs``, on scheduling order or on which other cells are in the grid. Null
tables and backgrounds are keyed without N, so all sample sizes of a cell share them (common random numbers give
smooth curves in N). With ``n_jobs > 1`` items run in a ``multiprocessing`` pool with the spawn context (largest
first). Finished items are cached in ``out_dir/cache`` (keyed by the item parameters, the seed and
:data:`ITEM_CODE_VERSION`), so an interrupted run resumes where it stopped. Summaries are computed in the parent.

Outputs (``out_dir``)
---------------------
``<experiment>.csv`` (one row per grid cell; E3 adds ``E3_power_mde.csv``, E7 adds ``E7_chain_totals.csv``) and
``results.json`` (config, rows, summaries and the error budget; reload with :func:`load_results`). The report
(``CALIBRATION.md``, ``calibration.json``, figures) is written by :func:`tycho.calibration.report.write_report`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import time
import zipfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from tycho.analysis.events import detect_events
from tycho.analysis.ricurve import RICurve
from tycho.calibration.planted import (
    dichotomized_gaussian_table,
    equicorrelation,
    gaussian_binarized,
    latent_threshold,
    planted_binary,
    planted_binary_batch,
    random_interactions,
    sample_table,
    switch_on,
    theta_for_local_ri,
    true_local_ri,
)
from tycho.ri.discretize import BINARIZE_METHODS
from tycho.ri.estimators import bh_fdr, candidate_subsets, estimate_local_ri, ri_df, ri_scan
from tycho.ri.hypergraph import Subset, admissible_candidates
from tycho.ri.loglinear import loglinear_table, total_correlation
from tycho.ri.refined import chain_decomposition, graded_chain, greedy_selection, local_ri, local_ri_from_table
from tycho.ri.variables import VariableSet

logger = logging.getLogger(__name__)

__all__ = [
    "EXPERIMENTS",
    "CalibrationConfig",
    "run_calibration",
    "load_results",
    "interpolate_min_effect",
    "jsonable",
]

#: Experiment names, in run and report order. ``<name>.csv`` is written for each experiment that runs.
EXPERIMENTS: tuple[str, ...] = ("E1_null", "E2_bias", "E3_power", "E4_timing", "E5_identity", "E6_discretization",
                                "E7_chain")
#: Format tag of ``results.json``.
RESULTS_FORMAT = "tycho-calibration-results/1"
#: Version of the work-item computations. Bump it whenever an item function changes its output, which invalidates
#: the cache of earlier runs (the cache key includes it).
ITEM_CODE_VERSION = 1
#: Upper bound on the (N × columns) int64 entries of one stacked estimation call (16 MB), which bounds the memory of
#: a work item independently of the number of replicates.
CHUNK_ENTRIES = 2_000_000
#: Null quantiles reported as the noise floor.
NOISE_QUANTILES: tuple[float, ...] = (0.5, 0.95, 0.99)
#: Number of equal-width bins of the pooled p-value histograms (E1 summary, for the report's figure).
PVALUE_BINS = 20
#: |RI| below which a population value counts as zero (E6 symmetry check). Exact zeros come out ~1e-16; the smallest
#: genuine artefact in the default grids (median, order 4, ρ = 0.1) is ~5e-8.
ZERO_TOL = 1e-12
#: Noncentrality λ = 2N·RI above which E2 calls an interaction clearly detectable (single df = 1 test at α = 0.01
#: has power ≈ 97 % there); bias and CI coverage are summarised separately above and below it.
LAMBDA_STRONG = 20.0
#: Progress is logged at INFO level each time another this fraction of the work items has finished.
PROGRESS_STEP = 0.05
#: Fewest replicates for which a standard error from replicate means is quoted (E6); with fewer, its own
#: uncertainty (a t distribution with reps − 1 degrees of freedom) makes it meaningless.
MIN_REPS_FOR_SE = 10
#: Groups in E3 whose power at a larger N falls below the power at the next smaller N by more than this many Monte
#: Carlo standard errors count as violations of monotonicity in N.
MONOTONE_SE = 3.0
#: E1 cells in which more than this fraction of null replicates has RI_plugin = 0 exactly are reported as *boundary*
#: cells: the empirical table has a sampling zero in both parity classes of the top-order contrast, lies in the
#: closure of the lower-order model, and the test can neither reject (conservative) nor detect (blind).
BOUNDARY_FRAC = 0.05
#: Null tables per (cards, strength) used to describe how concentrated E1's tables are (median effective number of
#: cells exp(H)); regenerated from the same named streams as the work items, so they are the items' first tables.
EFFECTIVE_CELLS_SAMPLE = 2000
#: One-sided tail probability at which a cell's false-positive count is flagged as too high (liberal) or too low
#: (conservative) by an exact binomial test against α (E1); 0.00135 is the one-sided 3-sigma tail, so for large
#: replicate counts this is "more than 3 standard errors from α", and for small counts it stays exact.
FPR_TAIL = 0.00135


# ----------------------------------------------------------------------------------------------- configuration
@dataclass
class CalibrationConfig:
    """Grids and sizes of the calibration suite. The defaults are the :meth:`quick` sizes.

    Shared:
        name: preset name (``quick`` | ``full`` | ``custom``), recorded in the outputs.
        seed: master seed (non-negative int); every random stream derives from it and a cell name.
        experiments: which of :data:`EXPERIMENTS` to run (E2 and E3 share their simulations).
        alpha: detection level (BH q-value) of E3-E6, the survey default.
        fpr_alphas: levels at which E1 reports false-positive rates.
        orders: interaction orders k (binary k-subsets) of E1-E3.
        n_grid: sample sizes N of E1-E3 (and of the min_effect table).
        budget_n: N of the headline numbers in the report (the survey's default probe size; nearest grid value).
        min_effect: fixed effect threshold (nats) for E4's truth and detector and for E5's "real interaction" label.
        min_effect_floor: policy floor of the recommended min_effect (nats).
        min_effect_null_quantile: null quantile of RI_corrected behind the recommended min_effect.
    E1:
        e1_strengths: scales of the random lower-order structure.
        e1_reps: replicates (null tables) per cell.
        e1_categorical: extra non-binary cardinality patterns (validate χ² with df > 1).
        e1_bootstrap_reps: replicates (the first ones) that also get a parametric-bootstrap p-value (0 = off).
        e1_n_null: bootstrap draws per parametric-bootstrap p-value.
    E2/E3:
        e2_targets: planted true local RI values (nats).
        e2_background: spin scale of the random lower-order background. 0.2 keeps every target up to 0.1 nats
            attainable for > 99 % of backgrounds at k = 4; with 0.5 the 14 lower-order terms concentrate the table so
            much that the attainable top-order RI saturates near 0.03 nats (see E1's boundary cells).
        e2_reps: replicates per (k, target, N).
        e2_ci_reps: replicates (the first ones) that get a bootstrap CI; e2_n_boot resamples; e2_ci_level level.
        e3_family_sizes: numbers m of simultaneous tests for the BH q-value.
        e3_power: power level of the minimum detectable RI.
    E4:
        e4_order: order of the switching interaction; e4_checkpoints T; e4_t_star the ramp midpoint (checkpoints);
        e4_widths ramp widths (checkpoints; 0 = step); e4_n_grid sample sizes; e4_final_ri population RI at full
        strength; e4_background spin scale of the fixed lower-order background; e4_persistence of detect_events;
        e4_reps replicates per cell.
    E5:
        e5_d variables; e5_theta3 / e5_theta2 spin couplings of the planted {0,1,2} and {3,4}; e5_background spin
        scale of the weak background; e5_max_order scan order; e5_n_grid sample sizes; e5_reps replicates per N.
    E6:
        e6_rhos latent correlations; e6_methods binarisation rules; e6_quantile level of the quantile rule;
        e6_latent_mean latent mean (only "zero" depends on it); e6_max_order; e6_d latent dimension of the sampled
        scans; e6_n_grid sample sizes; e6_reps replicates per cell.
    E7:
        e7_cards cardinality patterns; e7_tables random tables per pattern; e7_scale log-potential scale;
        e7_random_chains random admissible chains per table.
    """

    name: str = "custom"
    seed: int = 0
    experiments: tuple[str, ...] = EXPERIMENTS
    alpha: float = 0.01
    fpr_alphas: tuple[float, ...] = (0.01, 0.05)
    orders: tuple[int, ...] = (2, 3, 4)
    n_grid: tuple[int, ...] = (256, 1024, 4096)
    budget_n: int = 4096
    min_effect: float = 1e-3
    min_effect_floor: float = 1e-3
    min_effect_null_quantile: float = 0.99
    # E1
    e1_strengths: tuple[float, ...] = (0.0, 0.5, 1.0)
    e1_reps: int = 300
    e1_categorical: tuple[tuple[int, ...], ...] = ((3, 3),)
    e1_bootstrap_reps: int = 0
    e1_n_null: int = 199
    # E2 / E3
    e2_targets: tuple[float, ...] = (1e-3, 4e-3, 1.6e-2, 6.4e-2)
    e2_background: float = 0.2
    e2_reps: int = 100
    e2_ci_reps: int = 20
    e2_n_boot: int = 50
    e2_ci_level: float = 0.95
    e3_family_sizes: tuple[int, ...] = (1, 1000)
    e3_power: float = 0.8
    # E4
    e4_order: int = 3
    e4_checkpoints: int = 30
    e4_t_star: float = 15.0
    e4_widths: tuple[float, ...] = (0.0, 3.0)
    e4_n_grid: tuple[int, ...] = (1024, 8192)
    e4_final_ri: float = 0.02
    e4_background: float = 0.3
    e4_persistence: int = 2
    e4_reps: int = 8
    # E5
    e5_d: int = 8
    e5_theta3: float = 0.15
    e5_theta2: float = 0.15
    e5_background: float = 0.02
    e5_max_order: int = 4
    e5_n_grid: tuple[int, ...] = (1024, 8192)
    e5_reps: int = 5
    # E6
    e6_rhos: tuple[float, ...] = (0.0, 0.3, 0.6)
    e6_methods: tuple[str, ...] = ("median", "zero", "quantile")
    e6_quantile: float = 0.8
    e6_latent_mean: float = 0.5
    e6_max_order: int = 4
    e6_d: int = 6
    e6_n_grid: tuple[int, ...] = (4096,)
    e6_reps: int = 3
    # E7
    e7_cards: tuple[tuple[int, ...], ...] = ((2, 3, 2, 3), (2, 2, 3, 2, 2))
    e7_tables: int = 1
    e7_scale: float = 1.0
    e7_random_chains: int = 2

    def __post_init__(self) -> None:
        for f in fields(self):  # lists from JSON/YAML -> tuples (hashable, canonical)
            v = getattr(self, f.name)
            if isinstance(v, list):
                setattr(self, f.name, tuple(tuple(x) if isinstance(x, list) else x for x in v))
        self._validate()

    def _validate(self) -> None:
        def require(ok: bool, msg: str) -> None:
            if not ok:
                raise ValueError(f"CalibrationConfig: {msg}")

        require(isinstance(self.seed, int) and self.seed >= 0, "seed must be a non-negative int")
        unknown = set(self.experiments) - set(EXPERIMENTS)
        require(not unknown, f"unknown experiments {sorted(unknown)}; choose from {EXPERIMENTS}")
        require(0.0 < self.alpha < 1.0 and all(0.0 < a < 1.0 for a in self.fpr_alphas), "alphas must lie in (0, 1)")
        require(bool(self.orders) and all(2 <= k <= 8 for k in self.orders), "orders must lie in [2, 8]")
        for name in ("n_grid", "e4_n_grid", "e5_n_grid", "e6_n_grid"):
            grid = getattr(self, name)
            require(bool(grid) and all(int(n) >= 2 for n in grid) and list(grid) == sorted(set(grid)),
                    f"{name} must be a strictly increasing tuple of sample sizes >= 2")
        require(self.min_effect >= 0 and self.min_effect_floor >= 0, "min_effect values must be >= 0")
        require(0.5 <= self.min_effect_null_quantile < 1.0, "min_effect_null_quantile must lie in [0.5, 1)")
        require(all(s >= 0 for s in self.e1_strengths), "e1_strengths must be >= 0")
        require(all(len(c) >= 2 and min(c) >= 2 for c in self.e1_categorical), "e1_categorical needs >= 2 axes >= 2")
        require(self.e1_reps >= 2 and 0 <= self.e1_bootstrap_reps <= self.e1_reps and self.e1_n_null >= 1,
                "E1 replicate counts invalid")
        require(bool(self.e2_targets) and all(t > 0 for t in self.e2_targets), "e2_targets must be > 0")
        require(self.e2_reps >= 2 and 0 <= self.e2_ci_reps <= self.e2_reps and self.e2_n_boot >= 1,
                "E2 replicate counts invalid")
        require(0.0 < self.e2_ci_level < 1.0 and 0.0 < self.e3_power < 1.0, "levels must lie in (0, 1)")
        require(bool(self.e3_family_sizes) and all(m >= 1 for m in self.e3_family_sizes), "family sizes >= 1")
        require(2 <= self.e4_order <= 6 and self.e4_checkpoints >= 3 and self.e4_persistence >= 1,
                "E4 order/checkpoints/persistence invalid")
        require(all(w >= 0 for w in self.e4_widths) and self.e4_final_ri > self.min_effect,
                "E4 widths must be >= 0 and e4_final_ri > min_effect")
        require(self.e5_d >= 5 and 3 <= self.e5_max_order <= self.e5_d, "E5 needs d >= 5 and 3 <= max_order <= d")
        require(set(self.e6_methods) <= set(BINARIZE_METHODS), f"e6_methods must be in {BINARIZE_METHODS}")
        require(all(0.0 <= r < 1.0 for r in self.e6_rhos), "e6_rhos must lie in [0, 1)")
        require(0.0 < self.e6_quantile < 1.0 and 2 <= self.e6_max_order <= self.e6_d, "E6 quantile/order invalid")
        require(all(len(c) >= 2 and min(c) >= 2 for c in self.e7_cards), "e7_cards needs >= 2 axes >= 2")
        for name in ("e1_reps", "e2_reps", "e4_reps", "e5_reps", "e6_reps", "e7_tables"):
            require(getattr(self, name) >= 1, f"{name} must be >= 1")

    @classmethod
    def quick(cls) -> "CalibrationConfig":
        """Smoke-test sizes: the whole suite takes ~15-25 s on one core. Every code path runs, but with few
        replicates the numbers carry large Monte Carlo errors; use :meth:`full` for the real calibration."""
        return cls(name="quick")

    @classmethod
    def full(cls) -> "CalibrationConfig":
        """Production sizes, for ``n_jobs`` = all cores of the survey machine.

        Expected runtime: ≈15 CPU-hours of work items, measured per item on one x86-64 Linux core (E1 ≈ 4.9 h,
        E2/E3 ≈ 7.7 h, E4 ≈ 1.3 h, E5-E7 ≈ 1.1 h), i.e. roughly 1.5-3 h wall on a 10-core Apple-silicon Mac mini
        with ``n_jobs=10`` (its six efficiency cores are about half as fast as its four performance cores). The
        largest single item takes ~10 min. Interrupted runs resume from the item cache. Monte Carlo standard
        errors: false-positive rate at α = 0.01 ±0.05 %, null 99th percentile ±5 %, power ±0.6 %, CI coverage
        ±1 %."""
        return cls(
            name="full",
            n_grid=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536),
            e1_reps=40_000, e1_bootstrap_reps=4_000, e1_n_null=999,
            e2_targets=tuple(float(f"{10 ** (-4.5 + 0.25 * i):.3g}") for i in range(15)),
            e2_reps=8_000, e2_ci_reps=500, e2_n_boot=200,
            e4_checkpoints=40, e4_t_star=20.0, e4_widths=(0.0, 1.0, 3.0, 8.0),
            e4_n_grid=(512, 1024, 2048, 4096, 8192, 16384, 32768, 65536), e4_reps=1_000,
            e5_n_grid=(512, 1024, 2048, 4096, 8192, 16384, 32768, 65536), e5_reps=2_000,
            e6_rhos=tuple(round(0.1 * i, 1) for i in range(10)), e6_n_grid=(4096, 16384, 65536), e6_reps=1_000,
            e7_cards=((2, 3, 2, 3), (3, 2, 2, 2), (2, 2, 3, 2, 3), (2, 3, 2, 2, 2)), e7_tables=100,
            e7_random_chains=50,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (tuples become lists); :meth:`from_dict` inverts it."""
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibrationConfig":
        """Build from a dict such as :meth:`to_dict` output. Unknown keys raise ``ValueError``."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown CalibrationConfig fields: {sorted(unknown)}")
        return cls(**dict(d))


# ----------------------------------------------------------------------------------------------- utilities
def jsonable(obj: Any) -> Any:
    """Recursively convert to strict-JSON types: numpy scalars/arrays to Python, tuples to lists, NaN/inf to None."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def _rng(seed: int, *parts: object) -> np.random.Generator:
    """Generator for a named stream: ``SeedSequence([seed, sha256("|".join(parts))[:16 bytes as 4 words]])``."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([int(seed), *words]))


def _fmt(v: Any) -> str:
    """Canonical text of a cell parameter inside an item name (sequences as ``2x3``, floats by ``repr``)."""
    if isinstance(v, (list, tuple)):
        return "x".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _name(kind: str, **ident: Any) -> str:
    """Name of a cell, e.g. ``E1|cards=2x2x2|n=1024|strength=0.5``; seeds and cache keys derive from it."""
    return kind + "|" + "|".join(f"{k}={_fmt(v)}" for k, v in sorted(ident.items()))


def _subset_str(S: Sequence[int]) -> str:
    return "-".join(str(int(i)) for i in S)


def _chunks(total: int, size: int, breaks: Sequence[int] = ()) -> list[tuple[int, int]]:
    """``[a, b)`` chunks of ``range(total)`` of at most ``size`` items, also split at every break point."""
    cuts = sorted({0, total, *(b for b in breaks if 0 < b < total)})
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        out.extend((a, min(a + size, hi)) for a in range(lo, hi, size))
    return out


def _quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if x.size else math.nan


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else math.nan


def _binom_se(p: float, n: int) -> float:
    return math.sqrt(p * (1.0 - p) / n) if n > 0 and math.isfinite(p) else math.nan


def _se_of_rows(x: np.ndarray) -> float:
    """Standard error of the grand mean of an (R, S) array whose rows (replicates) are independent and whose columns
    (subsets of one scan, sharing variables) are not: the SE of the R row means."""
    if x.shape[0] < 2:
        return math.nan
    return float(np.std(x.mean(axis=1), ddof=1) / math.sqrt(x.shape[0]))


def _finite_positive(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def _isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted non-decreasing least-squares fit (pool-adjacent-violators)."""
    blocks: list[list[float]] = []  # [value, weight, count]
    for yi, wi in zip(y.tolist(), w.tolist()):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            wt = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / wt if wt > 0 else 0.5 * (v1 + v2), wt, c1 + c2])
    return np.repeat([b[0] for b in blocks], [int(b[2]) for b in blocks]).astype(np.float64)


def interpolate_min_effect(table: Sequence[Mapping[str, Any]], n: float, order: int | None = None) -> float:
    """Recommended min_effect (nats) at sample size ``n`` from a table of records ``{order, n, value, noise, floor}``.

    The noise part scales like 1/N (the null distribution of ``2N·RI_plugin`` is ~χ²_df), so ``n·noise`` is
    interpolated linearly in ``log n`` between grid points and held constant beyond them, then ``max(floor, ·)`` is
    taken. ``order=None`` returns the maximum over orders (one threshold for a whole scan). An order missing from the
    table falls back to that maximum.

    Raises:
        ValueError: on an empty table or ``n <= 0``.
    """
    if not table:
        raise ValueError("empty min_effect table")
    if not n > 0:
        raise ValueError(f"n must be > 0, got {n}")
    orders = sorted({int(r["order"]) for r in table})
    if order is not None and int(order) in orders:
        orders = [int(order)]
    values = []
    for k in orders:
        rows = sorted((r for r in table if int(r["order"]) == k), key=lambda r: float(r["n"]))
        ns = np.log([float(r["n"]) for r in rows])
        scaled = np.asarray([float(r["n"]) * float(r["noise"]) for r in rows])
        noise = float(np.interp(math.log(n), ns, scaled)) / n
        floor = max(float(r["floor"]) for r in rows)
        values.append(max(floor, noise))
    return float(max(values))


# ----------------------------------------------------------------------------------------------- shared simulation
def _replicate_estimates(tables: np.ndarray, n: int, rng: np.random.Generator, *, null_reps: int = 0,
                         n_null: int = 199, null_rng: np.random.Generator | None = None, ci_reps: int = 0,
                         n_boot: int = 200, ci_level: float = 0.95,
                         ci_rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Draw ``n`` samples from each of R tables and estimate the local RI of each table's full variable set.

    The replicates of a chunk are stacked as disjoint column groups of one data matrix, so one
    :func:`~tycho.ri.estimators.estimate_local_ri` call (the survey's own estimator, χ² null) handles many independent
    replicates. The first ``null_reps`` replicates also get a parametric-bootstrap p-value on the same data, and the
    first ``ci_reps`` a row-bootstrap CI (rows are resampled jointly across the stacked groups, which leaves each
    replicate's bootstrap distribution unchanged).
    """
    R = tables.shape[0]
    cards = [int(c) for c in tables.shape[1:]]
    k = len(cards)
    out = {"ri_plugin": np.full(R, np.nan), "ri_corrected": np.full(R, np.nan), "p_value": np.full(R, np.nan),
           "p_boot": np.full(null_reps, np.nan), "ci_low": np.full(R, np.nan), "ci_high": np.full(R, np.nan)}
    per_chunk = max(1, CHUNK_ENTRIES // (n * k))
    for a, b in _chunks(R, per_chunk, breaks=(ci_reps, null_reps)):
        X = np.hstack([sample_table(tables[r], n, rng) for r in range(a, b)])
        subsets = [tuple(range(j * k, (j + 1) * k)) for j in range(b - a)]
        all_cards = cards * (b - a)
        with_ci = a < ci_reps
        est = estimate_local_ri(X, all_cards, subsets, ci=with_ci, n_boot=n_boot, ci_level=ci_level, rng=ci_rng)
        out["ri_plugin"][a:b] = [e.ri_plugin for e in est]
        out["ri_corrected"][a:b] = [e.ri_corrected for e in est]
        out["p_value"][a:b] = [e.p_value for e in est]
        if with_ci:
            out["ci_low"][a:b] = [e.ci_low for e in est]
            out["ci_high"][a:b] = [e.ci_high for e in est]
        if a < null_reps:
            boot = estimate_local_ri(X, all_cards, subsets, null="parametric_bootstrap", n_null=n_null, rng=null_rng)
            out["p_boot"][a:b] = [e.p_value for e in boot]
    return out


def _lower_subsets(k: int) -> list[Subset]:
    """All subsets of ``range(k)`` of order 1..k−1 in graded-lex order (the keys of ``random_interactions``)."""
    return [S for o in range(1, k) for S in combinations(range(k), o)]


def _background_thetas(k: int, scale: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """(reps, n_lower) spin couplings: one :func:`random_interactions` draw (orders 1..k−1) per replicate."""
    subsets = _lower_subsets(k)
    rows = [random_interactions(k, range(1, k), scale, rng) for _ in range(reps)]
    return np.asarray([[th[S] for S in subsets] for th in rows], dtype=np.float64).reshape(reps, len(subsets))


def _null_tables(cards: Sequence[int], strength: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """``reps`` random tables with structure of order < len(cards) only (zero top-order local RI)."""
    k = len(cards)
    if all(c == 2 for c in cards):
        return planted_binary_batch(k, _lower_subsets(k), _background_thetas(k, strength, reps, rng))
    out = np.empty((reps,) + tuple(cards))
    lower = _lower_subsets(k)
    for r in range(reps):
        theta = {S: strength * rng.standard_normal(tuple(cards[i] for i in S)) for S in lower}
        out[r] = loglinear_table(theta, list(cards))
    return out


# ----------------------------------------------------------------------------------------------- work items
def _item_null(p: dict, seed: int, name: str) -> dict[str, np.ndarray]:
    """E1 cell: null replicates for one (cards, strength, N)."""
    cards = [int(c) for c in p["cards"]]
    tables = _null_tables(cards, float(p["strength"]), int(p["reps"]), _rng(seed, p["table_name"]))
    est = _replicate_estimates(tables, int(p["n"]), _rng(seed, name, "data"), null_reps=int(p["boot_reps"]),
                               n_null=int(p["n_null"]), null_rng=_rng(seed, name, "bootstrap-null"))
    return {"ri_plugin": est["ri_plugin"], "ri_corrected": est["ri_corrected"], "p_chi2": est["p_value"],
            "p_boot": est["p_boot"], "df": np.array(ri_df(cards))}


def _item_planted(p: dict, seed: int, name: str) -> dict[str, np.ndarray]:
    """E2/E3 cell: replicates with a planted top-order RI for one (k, target, N)."""
    k, reps, n = int(p["order"]), int(p["reps"]), int(p["n"])
    bg = _background_thetas(k, float(p["background"]), reps, _rng(seed, p["table_name"]))
    theta, tables, true_ri = theta_for_local_ri(float(p["target"]), k, _lower_subsets(k), bg)
    est = _replicate_estimates(tables, n, _rng(seed, name, "data"), ci_reps=int(p["ci_reps"]),
                               n_boot=int(p["n_boot"]), ci_level=float(p["ci_level"]), ci_rng=_rng(seed, name, "ci"))
    bh_rng = _rng(seed, name, "bh")
    fams = [int(m) for m in p["family_sizes"]]
    q = np.empty((len(fams), reps))
    for i, m in enumerate(fams):
        if m == 1:
            q[i] = est["p_value"]
            continue
        for r in range(reps):
            pv = np.concatenate([est["p_value"][r:r + 1], bh_rng.random(m - 1)])
            q[i, r] = bh_fdr(pv, 1.0)[1][0]  # q-values do not depend on the level argument
    return {"theta": theta, "true_ri": true_ri, "valid": np.isfinite(theta), "ri_plugin": est["ri_plugin"],
            "ri_corrected": est["ri_corrected"], "p_value": est["p_value"], "q": q, "ci_low": est["ci_low"],
            "ci_high": est["ci_high"], "df": np.array(ri_df([2] * k))}


def _e4_tables(p: Mapping[str, Any]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Schedule θ(t), the population tables and their true top-order RI for an E4 cell (deterministic)."""
    k, T = int(p["order"]), int(p["checkpoints"])
    theta_t = switch_on(np.arange(T), float(p["t_star"]), float(p["width"]), float(p["theta_max"]))
    bg = {tuple(S): float(v) for S, v in p["background"]}
    top = tuple(range(k))
    tables = [planted_binary(k, {**bg, top: float(th)}) for th in theta_t]
    return theta_t, tables, np.asarray([local_ri(P) for P in tables])


def _item_timing(p: dict, seed: int, name: str) -> dict[str, np.ndarray]:
    """E4 cell: onset detection of a switching interaction for one (width, N)."""
    k, T, n, reps = int(p["order"]), int(p["checkpoints"]), int(p["n"]), int(p["reps"])
    alpha, min_effect = float(p["alpha"]), float(p["min_effect"])
    theta_t, tables, true_ri = _e4_tables(p)
    steps = np.arange(T)
    names = [f"x{i}" for i in range(k)]
    top = tuple(range(k))
    rng = _rng(seed, name, "data")
    onset = np.full(reps, -1, dtype=np.int64)
    n_events = np.zeros(reps, dtype=np.int64)
    for r in range(reps):
        scans = [ri_scan(VariableSet(X=sample_table(P, n, rng), cards=[2] * k, names=names), max_order=k,
                         alpha=alpha, min_effect=min_effect) for P in tables]
        curve = RICurve.from_scans(steps, scans, meta={"alpha": alpha, "min_effect": min_effect})
        events = [e for e in detect_events(curve, alpha=alpha, on=min_effect, persistence=int(p["persistence"]))
                  if e.subset == top]
        n_events[r] = len(events)
        activations = [e.step for e in events if e.kind == "activation"]
        if activations:
            onset[r] = activations[0]
    reached = np.flatnonzero(true_ri >= min_effect)
    return {"theta": theta_t, "true_ri": true_ri, "true_onset": np.array(int(reached[0]) if reached.size else -1),
            "onset": onset, "n_events": n_events}


def _e5_table(p: Mapping[str, Any]) -> np.ndarray:
    d = int(p["d"])
    theta = {tuple(S): float(v) for S, v in p["background"]}
    theta.update({tuple(S): float(v) for S, v in p["planted"]})
    return planted_binary(d, theta)


def _scan_replicates(sampler: Callable[[], np.ndarray], d: int, max_order: int, alpha: float,
                     reps: int) -> dict[str, np.ndarray]:
    """``reps`` full scans (all subsets of order 2..max_order, BH within the scan) of freshly sampled data."""
    names = [f"x{i}" for i in range(d)]
    S = len(candidate_subsets(d, max_order=max_order))
    out = {key: np.empty((reps, S)) for key in ("ri_plugin", "ri_corrected", "q")}
    for r in range(reps):
        scan = ri_scan(VariableSet(X=sampler(), cards=[2] * d, names=names), max_order=max_order, alpha=alpha,
                       min_effect=0.0)
        out["ri_plugin"][r] = [e.ri_plugin for e in scan.estimates]
        out["ri_corrected"][r] = [e.ri_corrected for e in scan.estimates]
        out["q"][r] = scan.q_values
    return out


def _item_identity(p: dict, seed: int, name: str) -> dict[str, np.ndarray]:
    """E5 cell: full scans of the planted d-variable table at one N."""
    P = _e5_table(p)
    rng = _rng(seed, name, "data")
    n = int(p["n"])
    return _scan_replicates(lambda: sample_table(P, n, rng), int(p["d"]), int(p["max_order"]), float(p["alpha"]),
                            int(p["reps"]))


def _item_discretization(p: dict, seed: int, name: str) -> dict[str, np.ndarray]:
    """E6 cell: population artefacts and sampled scans for one (method, ρ, N)."""
    method, rho = str(p["method"]), float(p["rho"])
    q, mean = float(p["quantile"]), float(p["latent_mean"])
    tau = latent_threshold(method, q=q, mean=mean)
    ri_pop = np.asarray([local_ri(dichotomized_gaussian_table(j, rho, tau))
                         for j in range(2, int(p["max_order"]) + 1)])
    corr = equicorrelation(int(p["d"]), rho)
    rng = _rng(seed, name, "data")
    n = int(p["n"])
    out = _scan_replicates(lambda: gaussian_binarized(corr, n, rng, method, q=q, mean=mean), int(p["d"]),
                           int(p["max_order"]), float(p["alpha"]), int(p["reps"]))
    out.update({"ri_pop": ri_pop, "tau": np.array(tau)})
    return out


#: Fixed chains of E7, before the random ones (``random_0``, ``random_1``, ...).
E7_FIXED_CHAINS: tuple[str, ...] = ("graded", "reverse_graded", "greedy")


def _random_chain(d: int, rng: np.random.Generator) -> list[Subset]:
    """A uniformly random admissible step at every position, from the singletons to the full power set."""
    family: set[Subset] = set()
    chain: list[Subset] = []
    while True:
        cands = admissible_candidates(frozenset(family), d)
        if not cands:
            return chain
        S = cands[int(rng.integers(len(cands)))]
        family.add(S)
        chain.append(S)


def _item_chain(p: dict, seed: int, name: str) -> dict[str, np.ndarray]:
    """E7 cell: chain RI of every subset of one random table along all chains."""
    cards = tuple(int(c) for c in p["cards"])
    d = len(cards)
    rng = _rng(seed, name, "table")
    P = np.exp(float(p["scale"]) * rng.standard_normal(cards))
    P /= P.sum()
    subsets = graded_chain(d)
    index = {S: j for j, S in enumerate(subsets)}
    reverse = [S for o in range(2, d + 1) for S in reversed([T for T in subsets if len(T) == o])]
    decompositions = [chain_decomposition(P, subsets), chain_decomposition(P, reverse), greedy_selection(P)]
    chain_rng = _rng(seed, name, "chains")
    decompositions += [chain_decomposition(P, _random_chain(d, chain_rng)) for _ in range(int(p["random_chains"]))]
    ri = np.full((len(decompositions), len(subsets)), np.nan)
    for c, dec in enumerate(decompositions):
        for step in dec.steps:
            ri[c, index[step.subset]] = step.ri
    return {"ri": ri, "total": np.asarray([dec.total for dec in decompositions]),
            "residual": np.asarray([dec.residual for dec in decompositions]),
            "local_ri": local_ri_from_table(P, subsets), "tc": np.array(total_correlation(P))}


_ITEM_FUNCS: dict[str, Callable[[dict, int, str], dict[str, np.ndarray]]] = {
    "E1": _item_null, "E23": _item_planted, "E4": _item_timing, "E5": _item_identity, "E6": _item_discretization,
    "E7": _item_chain,
}


@dataclass(frozen=True)
class _Item:
    """One work item: ``kind`` selects the item function, ``name`` seeds it, ``params`` (JSON) fully specify it."""

    kind: str
    name: str
    params: dict
    cost: float  # rough relative cost, for largest-first scheduling only


def _execute(task: tuple[int, str, str, dict, int]) -> tuple[int, dict[str, np.ndarray], float]:
    """Pool entry point (top level so that spawned workers can import it)."""
    index, kind, name, params, seed = task
    t0 = time.perf_counter()
    out = _ITEM_FUNCS[kind](params, seed, name)
    return index, out, time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------- planning
def _plan_e1(cfg: CalibrationConfig) -> list[_Item]:
    items = []
    cases = [(2,) * k for k in cfg.orders] + [tuple(c) for c in cfg.e1_categorical]
    for cards in cases:
        for s in cfg.e1_strengths:
            table_name = _name("E1-tables", cards=cards, strength=float(s))
            for n in cfg.n_grid:
                params = {"cards": list(cards), "strength": float(s), "n": int(n), "reps": cfg.e1_reps,
                          "boot_reps": cfg.e1_bootstrap_reps, "n_null": cfg.e1_n_null, "table_name": table_name}
                cost = cfg.e1_reps * (n * len(cards) * 2e-7 + 5e-5) + cfg.e1_bootstrap_reps * cfg.e1_n_null * 2e-5
                items.append(_Item("E1", _name("E1", cards=cards, strength=float(s), n=int(n)), params, cost))
    return items


def _plan_e23(cfg: CalibrationConfig) -> list[_Item]:
    ci_reps = cfg.e2_ci_reps if "E2_bias" in cfg.experiments else 0
    items = []
    for k in cfg.orders:
        table_name = _name("E23-tables", order=k)
        for target in cfg.e2_targets:
            for n in cfg.n_grid:
                params = {"order": int(k), "target": float(target), "n": int(n), "reps": cfg.e2_reps,
                          "ci_reps": ci_reps, "n_boot": cfg.e2_n_boot, "ci_level": cfg.e2_ci_level,
                          "background": cfg.e2_background, "family_sizes": list(cfg.e3_family_sizes),
                          "table_name": table_name}
                cost = cfg.e2_reps * (n * k * 2e-7 + 1e-4) + ci_reps * cfg.e2_n_boot * n * k * 1e-7
                items.append(_Item("E23", _name("E23", order=k, target=float(target), n=int(n)), params, cost))
    return items


def _e4_design(cfg: CalibrationConfig) -> dict[str, Any]:
    """Fixed background and full-strength coupling of E4 (deterministic in the seed)."""
    k = cfg.e4_order
    bg = random_interactions(k, range(1, k), cfg.e4_background, _rng(cfg.seed, "E4-background"))
    subsets = list(bg)
    theta, _, ri = theta_for_local_ri(cfg.e4_final_ri, k, subsets, np.asarray([[bg[S] for S in subsets]]))
    if not np.isfinite(theta[0]):
        raise ValueError(f"e4_final_ri={cfg.e4_final_ri} is not reachable with the E4 background")
    return {"order": k, "background": [[list(S), v] for S, v in bg.items()], "theta_max": float(theta[0]),
            "final_ri": float(ri[0])}


def _plan_e4(cfg: CalibrationConfig) -> list[_Item]:
    design = _e4_design(cfg)
    items = []
    for w in cfg.e4_widths:
        for n in cfg.e4_n_grid:
            params = {**design, "checkpoints": cfg.e4_checkpoints, "t_star": cfg.e4_t_star, "width": float(w),
                      "n": int(n), "reps": cfg.e4_reps, "alpha": cfg.alpha, "min_effect": cfg.min_effect,
                      "persistence": cfg.e4_persistence}
            cost = cfg.e4_reps * cfg.e4_checkpoints * (n * cfg.e4_order * 2e-7 + 1e-3)
            items.append(_Item("E4", _name("E4", width=float(w), n=int(n)), params, cost))
    return items


def _e5_design(cfg: CalibrationConfig) -> dict[str, Any]:
    """Planted couplings and the weak background of E5 (deterministic in the seed).

    The background has biases on every variable and couplings on every pair except the planted pair and the pairs
    inside the planted triple, so the triple stays a pure synergy: a detected pair inside it is an order error.
    """
    planted = {(0, 1, 2): cfg.e5_theta3, (3, 4): cfg.e5_theta2}
    bg = random_interactions(cfg.e5_d, [1, 2], cfg.e5_background, _rng(cfg.seed, "E5-background"))
    for S in [S for S in bg if len(S) > 1 and any(set(S) <= set(G) for G in planted)]:
        del bg[S]
    return {"d": cfg.e5_d, "planted": [[list(S), float(v)] for S, v in planted.items()],
            "background": [[list(S), v] for S, v in bg.items()], "max_order": cfg.e5_max_order}


def _plan_e5(cfg: CalibrationConfig) -> list[_Item]:
    design = _e5_design(cfg)
    n_sub = len(candidate_subsets(cfg.e5_d, max_order=cfg.e5_max_order))
    return [_Item("E5", _name("E5", n=int(n)), {**design, "n": int(n), "reps": cfg.e5_reps, "alpha": cfg.alpha},
                  cfg.e5_reps * (n * n_sub * 3e-8 + 2e-3)) for n in cfg.e5_n_grid]


def _plan_e6(cfg: CalibrationConfig) -> list[_Item]:
    n_sub = len(candidate_subsets(cfg.e6_d, max_order=cfg.e6_max_order))
    items = []
    for method in cfg.e6_methods:
        for rho in cfg.e6_rhos:
            for n in cfg.e6_n_grid:
                params = {"method": method, "rho": float(rho), "quantile": cfg.e6_quantile,
                          "latent_mean": cfg.e6_latent_mean, "max_order": cfg.e6_max_order, "d": cfg.e6_d,
                          "n": int(n), "reps": cfg.e6_reps, "alpha": cfg.alpha}
                items.append(_Item("E6", _name("E6", method=method, rho=float(rho), n=int(n)), params,
                                   cfg.e6_reps * (n * n_sub * 3e-8 + 2e-3)))
    return items


def _plan_e7(cfg: CalibrationConfig) -> list[_Item]:
    items = []
    for cards in cfg.e7_cards:
        n_sub = 2 ** len(cards) - len(cards) - 1
        for t in range(cfg.e7_tables):
            params = {"cards": list(cards), "scale": cfg.e7_scale, "random_chains": cfg.e7_random_chains,
                      "table": t}
            items.append(_Item("E7", _name("E7", cards=cards, table=t), params,
                               (cfg.e7_random_chains + 3 + n_sub) * n_sub * 4e-3))
    return items


def _plan(cfg: CalibrationConfig) -> dict[str, list[_Item]]:
    want = set(cfg.experiments)
    plan: dict[str, list[_Item]] = {}
    if "E1_null" in want:
        plan["E1"] = _plan_e1(cfg)
    if want & {"E2_bias", "E3_power"}:
        plan["E23"] = _plan_e23(cfg)
    if "E4_timing" in want:
        plan["E4"] = _plan_e4(cfg)
    if "E5_identity" in want:
        plan["E5"] = _plan_e5(cfg)
    if "E6_discretization" in want:
        plan["E6"] = _plan_e6(cfg)
    if "E7_chain" in want:
        plan["E7"] = _plan_e7(cfg)
    return plan


# ----------------------------------------------------------------------------------------------- execution + cache
def _cache_blob(item: _Item, seed: int) -> str:
    return json.dumps({"version": ITEM_CODE_VERSION, "seed": seed, "kind": item.kind, "name": item.name,
                       "params": item.params}, sort_keys=True, default=str)


def _cache_path(cache_dir: Path, blob: str, kind: str) -> Path:
    return cache_dir / f"{kind}_{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:32]}.npz"


def _cache_load(path: Path, blob: str) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["__item__"]) != blob:
                return None
            return {k: z[k] for k in z.files if k != "__item__"}
    except FileNotFoundError:
        return None
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile) as exc:  # a damaged cache never fails a run
        logger.warning("ignoring unreadable cache file %s (%s)", path, exc)
        return None


def _cache_save(path: Path, blob: str, out: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, __item__=np.array(blob), **{k: np.asarray(v) for k, v in out.items()})
    os.replace(tmp, path)


def _run_items(items: Sequence[_Item], seed: int, *, n_jobs: int, cache_dir: Path, resume: bool,
               progress: Callable[[int, int], None] | None) -> tuple[list[dict[str, np.ndarray]], dict[str, float], int]:
    """Run (or load from cache) every item: results in item order, compute seconds per kind, number cached."""
    results: list[dict[str, np.ndarray] | None] = [None] * len(items)
    blobs = [_cache_blob(it, seed) for it in items]
    seconds: dict[str, float] = {}
    pending = []
    for i, (it, blob) in enumerate(zip(items, blobs)):
        cached = _cache_load(_cache_path(cache_dir, blob, it.kind), blob) if resume else None
        if cached is not None:
            results[i] = cached
        else:
            pending.append(i)
    total, done = len(items), len(items) - len(pending)
    logger.info("calibration: %d work items (%d cached), n_jobs=%d", total, done, n_jobs)
    if progress is not None:
        progress(done, total)
    pending.sort(key=lambda i: -items[i].cost)
    tasks = [(i, items[i].kind, items[i].name, items[i].params, seed) for i in pending]

    t_start = time.perf_counter()
    next_log = done + max(1, math.ceil(PROGRESS_STEP * total))

    def collect(index: int, out: dict[str, np.ndarray], secs: float) -> None:
        nonlocal done, next_log
        results[index] = out
        kind = items[index].kind
        seconds[kind] = seconds.get(kind, 0.0) + secs
        _cache_save(_cache_path(cache_dir, blobs[index], kind), blobs[index], out)
        done += 1
        logger.debug("item %s done in %.2fs (%d/%d)", items[index].name, secs, done, total)
        if done >= next_log or done == total:
            logger.info("calibration: %d/%d work items done (%.0f s)", done, total, time.perf_counter() - t_start)
            next_log = done + max(1, math.ceil(PROGRESS_STEP * total))
        if progress is not None:
            progress(done, total)

    if n_jobs > 1 and len(tasks) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(n_jobs, len(tasks))) as pool:
            for index, out, secs in pool.imap_unordered(_execute, tasks, chunksize=1):
                collect(index, out, secs)
    else:
        for task in tasks:
            collect(*_execute(task))
    missing = [items[i].name for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(f"work items without results: {missing[:5]}")
    return results, seconds, total - len(pending)  # type: ignore[return-value]


# ----------------------------------------------------------------------------------------------- E1 summary
def _null_row(cfg: CalibrationConfig, p: Mapping[str, Any], df: int, ri_plugin: np.ndarray, ri_corr: np.ndarray,
              pvals: np.ndarray, null: str) -> dict[str, Any]:
    n, R = int(p["n"]), int(pvals.size)
    row: dict[str, Any] = {"order": len(p["cards"]), "cards": _fmt(p["cards"]), "df": df,
                           "binary": all(int(c) == 2 for c in p["cards"]), "strength": float(p["strength"]), "n": n,
                           "null": null, "n_rep": R}
    for a in cfg.fpr_alphas:
        row[f"fpr_{a:g}"] = _mean((pvals <= a).astype(float))
        row[f"fpr_{a:g}_se"] = _binom_se(a, R)
    ks = stats.kstest(pvals, "uniform")
    row["ks_stat"], row["ks_pvalue"] = float(ks.statistic), float(ks.pvalue)
    for qq in NOISE_QUANTILES:
        tag = f"q{round(100 * qq)}"
        row[f"ri_plugin_{tag}"] = _quantile(ri_plugin, qq)
        row[f"ri_corrected_{tag}"] = _quantile(ri_corr, qq)
        row[f"theory_{tag}"] = float(stats.chi2.ppf(qq, df) / (2.0 * n))
    row["mean_g2_over_df"] = float(np.mean(2.0 * n * ri_plugin) / df)
    row["frac_zero"] = _mean((ri_plugin <= 0).astype(float))
    return row


def _summarize_e1(cfg: CalibrationConfig, items: Sequence[_Item],
                  results: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    rows = []
    for it, res in zip(items, results):
        df = int(res["df"])
        rows.append(_null_row(cfg, it.params, df, res["ri_plugin"], res["ri_corrected"], res["p_chi2"], "chi2"))
        B = int(it.params["boot_reps"])
        if B:
            rows.append(_null_row(cfg, it.params, df, res["ri_plugin"][:B], res["ri_corrected"][:B], res["p_boot"],
                                  "parametric_bootstrap"))
    chi_bin = [r for r in rows if r["null"] == "chi2" and r["binary"]]
    noise_floor = []
    for k in cfg.orders:
        for n in cfg.n_grid:
            cell = [r for r in chi_bin if r["order"] == k and r["n"] == n]
            rec = {"order": k, "n": n, "df": 1, "n_rep": min(r["n_rep"] for r in cell)}
            for qq in NOISE_QUANTILES:
                tag = f"q{round(100 * qq)}"
                rec[tag] = max(r[f"ri_plugin_{tag}"] for r in cell)
                rec[f"{tag}_corrected"] = max(r[f"ri_corrected_{tag}"] for r in cell)
                rec[f"theory_{tag}"] = cell[0][f"theory_{tag}"]
            noise_floor.append(rec)
    fpr: dict[str, Any] = {}
    for null in sorted({r["null"] for r in rows}):
        sel = [r for r in rows if r["null"] == null]
        per_alpha = {}
        for a in cfg.fpr_alphas:
            col = f"fpr_{a:g}"
            counts = [(round(r[col] * r["n_rep"]), r["n_rep"]) for r in sel]
            liberal = [stats.binomtest(k, n, a, alternative="greater").pvalue < FPR_TAIL for k, n in counts]
            conservative = [stats.binomtest(k, n, a, alternative="less").pvalue < FPR_TAIL for k, n in counts]
            worst = max(sel, key=lambda r: r[col])
            per_alpha[f"{a:g}"] = {
                "pooled": float(np.average([r[col] for r in sel], weights=[r["n_rep"] for r in sel])),
                "min": min(r[col] for r in sel), "max": max(r[col] for r in sel), "n_cells": len(sel),
                "n_liberal": int(sum(liberal)), "n_conservative": int(sum(conservative)), "tail": FPR_TAIL,
                "max_cell": {key: worst[key] for key in ("cards", "strength", "n", col, f"{col}_se")},
            }
        fpr[null] = per_alpha
    hist = {}
    edges = np.linspace(0.0, 1.0, PVALUE_BINS + 1)
    for k in cfg.orders:
        pooled = np.concatenate([res["p_chi2"] for it, res in zip(items, results) if it.params["cards"] == [2] * k])
        hist[str(k)] = {"edges": edges.tolist(), "counts": np.histogram(pooled, bins=edges)[0].tolist()}
    effective_cells = []
    for it in items:
        p = it.params
        if p["n"] == cfg.n_grid[0]:  # one record per (cards, strength); the tables do not depend on N
            T = _null_tables(p["cards"], p["strength"], min(int(p["reps"]), EFFECTIVE_CELLS_SAMPLE),
                             _rng(cfg.seed, p["table_name"])).reshape(-1, int(np.prod(p["cards"])))
            with np.errstate(divide="ignore", invalid="ignore"):
                H = -np.where(T > 0, T * np.log(T), 0.0).sum(axis=1)
            effective_cells.append({"cards": _fmt(p["cards"]), "strength": p["strength"],
                                    "n_cells": T.shape[1], "median_effective_cells": float(np.median(np.exp(H)))})
    chi = [r for r in rows if r["null"] == "chi2"]
    regular = [r for r in chi if r["frac_zero"] <= BOUNDARY_FRAC]
    boundary = sorted((r for r in chi if r["frac_zero"] > BOUNDARY_FRAC), key=lambda r: -r["frac_zero"])
    summary = {"noise_floor": noise_floor, "fpr": fpr, "pvalue_hist": hist,
               "ks_median": float(np.median([r["ks_stat"] for r in chi])),
               "ks_max_regular": max((r["ks_stat"] for r in regular), default=math.nan),
               "mean_g2_over_df_regular": [min((r["mean_g2_over_df"] for r in regular), default=math.nan),
                                           max((r["mean_g2_over_df"] for r in regular), default=math.nan)],
               "boundary_frac": BOUNDARY_FRAC, "n_cells": len(chi), "effective_cells": effective_cells,
               "boundary_cells": [{key: r[key] for key in ("cards", "strength", "n", "frac_zero")}
                                  for r in boundary]}
    return {"rows": rows, "summary": summary}


def _min_effect_table(cfg: CalibrationConfig, items: Sequence[_Item] | None,
                      results: Sequence[Mapping[str, np.ndarray]] | None) -> dict[str, Any]:
    """Recommended min_effect per (order, N): max(floor, null quantile of RI_corrected) (see module docstring)."""
    q = cfg.min_effect_null_quantile
    table = []
    for k in cfg.orders:
        for n in cfg.n_grid:
            if items is not None and results is not None:
                cells = [res["ri_corrected"] for it, res in zip(items, results)
                         if it.params["cards"] == [2] * k and it.params["n"] == n]
                noise, source = max(_quantile(c, q) for c in cells), "E1"
            else:
                noise, source = max(0.0, float((stats.chi2.ppf(q, 1) - 1.0) / (2.0 * n))), "chi2"
            table.append({"order": k, "n": n, "noise": noise, "floor": cfg.min_effect_floor,
                          "value": max(cfg.min_effect_floor, noise), "source": source})
    by_n = [{"n": n, "value": interpolate_min_effect(table, n)} for n in cfg.n_grid]
    return {"rule": "max(floor, null quantile of RI_corrected)", "floor": cfg.min_effect_floor, "null_quantile": q,
            "table": table, "by_n": by_n}


# ----------------------------------------------------------------------------------------------- E2 / E3 summaries
def _summarize_e2(cfg: CalibrationConfig, items: Sequence[_Item],
                  results: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    rows = []
    for it, res in zip(items, results):
        p = it.params
        n, df = int(p["n"]), int(res["df"])
        ok = res["valid"].astype(bool)
        tr, rp, rc = res["true_ri"][ok], res["ri_plugin"][ok], res["ri_corrected"][ok]
        truth = _mean(tr)
        lam = 2.0 * n * truth
        ci_ok = ok & np.isfinite(res["ci_low"])
        covered = (res["ci_low"][ci_ok] <= res["true_ri"][ci_ok]) & (res["true_ri"][ci_ok] <= res["ci_high"][ci_ok])
        row = {"order": int(p["order"]), "df": df, "target_ri": float(p["target"]), "true_ri": truth, "n": n,
               "lambda": lam, "n_rep": int(ok.sum()), "n_invalid": int((~ok).sum()),
               "mean_plugin": _mean(rp), "mean_corrected": _mean(rc),
               "bias_plugin": _mean(rp - tr), "bias_corrected": _mean(rc - tr),
               "theory_bias_plugin": df / (2.0 * n),
               "rel_bias_plugin": _mean(rp - tr) / truth, "rel_bias_corrected": _mean(rc - tr) / truth,
               "sd_plugin": float(np.std(rp, ddof=1)) if rp.size > 1 else math.nan,
               "theory_sd_plugin": math.sqrt(2.0 * (df + 2.0 * lam)) / (2.0 * n),
               "rmse_plugin": float(np.sqrt(np.mean((rp - tr) ** 2))) if rp.size else math.nan,
               "rmse_corrected": float(np.sqrt(np.mean((rc - tr) ** 2))) if rc.size else math.nan,
               "bias_se": float(np.std(rp - tr, ddof=1) / math.sqrt(rp.size)) if rp.size > 1 else math.nan,
               "n_ci": int(ci_ok.sum()), "coverage": _mean(covered.astype(float)),
               "coverage_se": _binom_se(cfg.e2_ci_level, int(ci_ok.sum())),
               "ci_width": _mean(res["ci_high"][ci_ok] - res["ci_low"][ci_ok])}
        row["rel_rmse_corrected"] = row["rmse_corrected"] / truth
        rows.append(row)
    strong = [r for r in rows if r["lambda"] >= LAMBDA_STRONG]
    weak = [r for r in rows if r["lambda"] < LAMBDA_STRONG]
    cov_strong = [r["coverage"] for r in strong if r["n_ci"] > 0]
    cov_weak = [r["coverage"] for r in weak if r["n_ci"] > 0]

    def median(x: list[float]) -> float:
        return float(np.median(x)) if x else math.nan

    summary = {
        "lambda_split": LAMBDA_STRONG,
        "max_abs_rel_bias_corrected_strong": max((abs(r["rel_bias_corrected"]) for r in strong), default=math.nan),
        "median_bias_ratio_plugin": median([r["bias_plugin"] / r["theory_bias_plugin"] for r in rows]),
        "median_sd_ratio_plugin": median([r["sd_plugin"] / r["theory_sd_plugin"] for r in rows]),
        "coverage_strong": [min(cov_strong, default=math.nan), max(cov_strong, default=math.nan)],
        "coverage_weak": [min(cov_weak, default=math.nan), max(cov_weak, default=math.nan)],
        "median_coverage_strong": median(cov_strong), "median_coverage_weak": median(cov_weak),
        "n_cells_strong": len(strong), "n_cells_weak": len(weak),
        "n_invalid_replicates": int(sum(r["n_invalid"] for r in rows)),
    }
    return {"rows": rows, "summary": summary}


def _crit(alpha: float, m: int, df: int, n: int, min_effect: float) -> float:
    """G² threshold of a detection: χ² critical value at α/m (Bonferroni view of BH), and RI_corr >= min_effect."""
    return max(float(stats.chi2.isf(alpha / m, df)), 2.0 * n * min_effect + df)


def _theory_power(ri: float, n: int, df: int, crit: float) -> float:
    lam = 2.0 * n * ri
    return float(stats.ncx2.sf(crit, df, lam)) if lam > 0 else float(stats.chi2.sf(crit, df))


def _theory_mde(n: int, df: int, crit: float, power: float) -> float:
    """Smallest RI with P(noncentral χ²_df(2N·RI) >= crit) = power."""
    def f(lam: float) -> float:
        return float(stats.ncx2.sf(crit, df, lam)) - power
    if float(stats.chi2.sf(crit, df)) >= power:
        return 0.0
    hi = 1.0
    while f(hi) < 0.0:
        hi *= 2.0
    return brentq(f, 1e-12, hi, xtol=1e-10, rtol=1e-12) / (2.0 * n)  # f(1e-12) ~ chi2.sf(crit) - power < 0


def _interp_mde(ri: np.ndarray, power: np.ndarray, weights: np.ndarray, level: float) -> tuple[float, str]:
    """RI at which the power curve reaches ``level``, with a censoring flag (``"below"`` / ``"above"`` the grid).

    The power estimates are first made non-decreasing in RI (weighted isotonic regression, weights = replicates),
    then interpolated linearly in (log RI, probit(power)), the coordinates in which a power curve is close to a
    straight line (dose-response practice). Powers of exactly 0 or 1 are clipped to ``0.5/n`` and ``1 − 0.5/n``.
    """
    keep = np.isfinite(ri) & (ri > 0) & np.isfinite(power)
    ri, power, weights = ri[keep], power[keep], weights[keep]
    if not ri.size:
        return math.nan, "above"
    order = np.argsort(ri)
    x, y = np.log(ri[order]), _isotonic(power[order], weights[order])
    if y[0] >= level:
        return float(ri[order][0]), "below"
    hit = np.flatnonzero(y >= level)
    if not hit.size:
        return math.nan, "above"
    i = int(hit[0])
    eps = 0.5 / max(float(np.min(weights)), 1.0)
    z0, z1 = stats.norm.ppf(np.clip([y[i - 1], y[i]], eps, 1.0 - eps))
    zl = float(stats.norm.ppf(level))
    t = (zl - z0) / (z1 - z0) if z1 > z0 else 1.0
    return float(math.exp(x[i - 1] + min(max(t, 0.0), 1.0) * (x[i] - x[i - 1]))), ""


def _summarize_e3(cfg: CalibrationConfig, items: Sequence[_Item], results: Sequence[Mapping[str, np.ndarray]],
                  min_effect: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    fams = list(cfg.e3_family_sizes)
    for it, res in zip(items, results):
        p = it.params
        k, n, df = int(p["order"]), int(p["n"]), int(res["df"])
        ok = res["valid"].astype(bool)
        truth = _mean(res["true_ri"][ok])
        me = interpolate_min_effect(min_effect["table"], n)  # one threshold per scan, as ri_scan applies it
        rc = res["ri_corrected"][ok]
        for i, m in enumerate(fams):
            sig = res["q"][i][ok] <= cfg.alpha
            power_stat, power = _mean(sig.astype(float)), _mean((sig & (rc >= me)).astype(float))
            rows.append({"order": k, "df": df, "target_ri": float(p["target"]), "true_ri": truth, "n": n,
                         "family_size": m, "min_effect": me, "n_rep": int(ok.sum()),
                         "power_stat": power_stat, "power_stat_se": _binom_se(power_stat, int(ok.sum())),
                         "power": power, "power_se": _binom_se(power, int(ok.sum())),
                         "theory_power_stat": _theory_power(truth, n, df, _crit(cfg.alpha, m, df, n, 0.0)),
                         "theory_power": _theory_power(truth, n, df, _crit(cfg.alpha, m, df, n, me))})
    mde_rows = []
    violations, groups = 0, 0
    for k in cfg.orders:
        for m in fams:
            for rule, col in (("stat", "power_stat"), ("survey", "power")):
                for n in cfg.n_grid:
                    cell = [r for r in rows if r["order"] == k and r["family_size"] == m and r["n"] == n]
                    ri = np.asarray([r["true_ri"] for r in cell])
                    pw = np.asarray([r[col] for r in cell])
                    mde, censored = _interp_mde(ri, pw, np.asarray([r["n_rep"] for r in cell], float), cfg.e3_power)
                    me = cell[0]["min_effect"] if rule == "survey" else 0.0
                    df = cell[0]["df"]
                    mde_rows.append({"order": k, "n": n, "family_size": m, "rule": rule, "min_effect": me,
                                     "power_level": cfg.e3_power, "mde": mde, "censored": censored,
                                     "theory_mde": _theory_mde(n, df, _crit(cfg.alpha, m, df, n, me), cfg.e3_power)})
                for target in cfg.e2_targets:
                    seq = sorted((r for r in rows if r["order"] == k and r["family_size"] == m
                                  and r["target_ri"] == float(target)), key=lambda r: r["n"])
                    for a, b in zip(seq[:-1], seq[1:]):
                        groups += 1
                        se = math.hypot(a[f"{col}_se"], b[f"{col}_se"]) or 1.0 / max(a["n_rep"], 1)
                        violations += int(b[col] < a[col] - MONOTONE_SE * se)
    ratios = [r["mde"] / r["theory_mde"] for r in mde_rows if not r["censored"] and r["theory_mde"] > 0]
    summary = {"mde": mde_rows, "power_level": cfg.e3_power, "family_sizes": fams,
               "monotone_in_n": {"violations": violations, "pairs": groups, "tolerance_se": MONOTONE_SE},
               "mde_over_theory_median": float(np.median(ratios)) if ratios else math.nan,
               "mde_over_theory_range": [min(ratios), max(ratios)] if ratios else [math.nan, math.nan]}
    return {"rows": rows, "summary": summary, "tables": {"mde": mde_rows}}


# ----------------------------------------------------------------------------------------------- E4 summary
def _summarize_e4(cfg: CalibrationConfig, items: Sequence[_Item],
                  results: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    rows, curves = [], {}
    for it, res in zip(items, results):
        p = it.params
        true_onset = int(res["true_onset"])
        det = res["onset"] >= 0
        # the error is undefined when the population RI never reaches min_effect within the window
        err = (res["onset"][det] - true_onset).astype(float) if true_onset >= 0 else np.zeros(0)
        R = int(res["onset"].size)
        rows.append({"width": float(p["width"]), "n": int(p["n"]), "checkpoints": int(p["checkpoints"]),
                     "t_star": float(p["t_star"]), "true_onset": true_onset, "theta_max": float(p["theta_max"]),
                     "final_ri": float(p["final_ri"]), "min_effect": float(p["min_effect"]), "n_rep": R,
                     "detect_rate": float(det.mean()), "median_error": float(np.median(err)) if err.size else math.nan,
                     "mean_error": _mean(err), "mae": _mean(np.abs(err)), "q10": _quantile(err, 0.10),
                     "q25": _quantile(err, 0.25), "q75": _quantile(err, 0.75), "q90": _quantile(err, 0.90),
                     "frac_early": _mean((err < 0).astype(float)), "frac_exact": _mean((err == 0).astype(float)),
                     "frac_within_1": _mean((np.abs(err) <= 1).astype(float)),
                     "mean_events": _mean(res["n_events"].astype(float))})
        curves[f"{float(p['width']):g}"] = {"theta": res["theta"].tolist(), "true_ri": res["true_ri"].tolist(),
                                            "true_onset": true_onset}
    return {"rows": rows, "summary": {"curves": curves, "persistence": cfg.e4_persistence,
                                      "min_effect": cfg.min_effect, "order": cfg.e4_order}}


# ----------------------------------------------------------------------------------------------- E5 summary
def _summarize_e5(cfg: CalibrationConfig, items: Sequence[_Item], results: Sequence[Mapping[str, np.ndarray]],
                  min_effect: Mapping[str, Any]) -> dict[str, Any]:
    p0 = items[0].params
    P = _e5_table(p0)
    subsets = candidate_subsets(int(p0["d"]), max_order=int(p0["max_order"]))
    truth = true_local_ri(P, subsets)
    planted = [tuple(S) for S, _ in p0["planted"]]
    is_planted = np.asarray([S in planted for S in subsets])
    related = np.asarray([not is_planted[j] and any(set(S) < set(G) or set(G) < set(S) for G in planted)
                          for j, S in enumerate(subsets)])
    real = truth >= cfg.min_effect
    rows, frequent = [], {}
    for it, res in zip(items, results):
        n = int(it.params["n"])
        for rule, me in (("survey", interpolate_min_effect(min_effect["table"], n)), ("stat", 0.0)):
            det = (res["q"] <= cfg.alpha) & (res["ri_corrected"] >= me)
            tp = det[:, is_planted].sum(axis=1)
            nd = det.sum(axis=1)
            fp = det & ~is_planted
            with np.errstate(invalid="ignore", divide="ignore"):
                precision = np.where(nd > 0, tp / np.maximum(nd, 1), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                precision_true = np.where(nd > 0, (det & real).sum(axis=1) / np.maximum(nd, 1), np.nan)
            row = {"n": n, "rule": rule, "min_effect": me, "n_rep": int(det.shape[0]),
                   "recall": _mean(tp / len(planted)),
                   "precision": float(np.nanmean(precision)) if np.isfinite(precision).any() else math.nan,
                   "precision_true": float(np.nanmean(precision_true)) if np.isfinite(precision_true).any()
                   else math.nan,
                   "exact_recovery": _mean((det == is_planted[None, :]).all(axis=1).astype(float)),
                   "mean_detected": _mean(nd.astype(float)), "mean_false_positives": _mean(fp.sum(axis=1)),
                   "mean_fp_spurious": _mean((fp & ~real).sum(axis=1)),
                   "mean_fp_real_unplanted": _mean((fp & real).sum(axis=1)),
                   "mean_fp_related": _mean((det & related).sum(axis=1))}
            for S, G in zip(planted, np.flatnonzero(is_planted)):
                row[f"recall_{_subset_str(S)}"] = _mean(det[:, G].astype(float))
            rows.append(row)
            if rule == "survey":
                rate = fp.mean(axis=0)
                top = [j for j in np.argsort(-rate, kind="stable")[:3] if rate[j] > 0]
                frequent[str(n)] = [{"subset": _subset_str(subsets[j]), "rate": float(rate[j]),
                                     "true_ri": float(truth[j])} for j in top]
    unplanted = np.flatnonzero(~is_planted)
    top_unplanted = unplanted[np.argsort(-truth[unplanted], kind="stable")[:5]]
    summary = {"planted": [{"subset": _subset_str(S), "order": len(S), "theta": float(v),
                            "true_ri": float(truth[subsets.index(tuple(S))])} for S, v in p0["planted"]],
               "n_subsets": len(subsets), "n_real_unplanted": int((real & ~is_planted).sum()),
               "max_ri_inside_planted": max((float(truth[j]) for j, S in enumerate(subsets)
                                             if any(set(S) < set(G) for G in planted)), default=0.0),
               "largest_unplanted": [{"subset": _subset_str(subsets[j]), "true_ri": float(truth[j])}
                                     for j in top_unplanted],
               "frequent_false_positives": frequent}
    return {"rows": rows, "summary": summary}


# ----------------------------------------------------------------------------------------------- E6 summary
def _summarize_e6(cfg: CalibrationConfig, items: Sequence[_Item], results: Sequence[Mapping[str, np.ndarray]],
                  min_effect: Mapping[str, Any]) -> dict[str, Any]:
    subsets = candidate_subsets(cfg.e6_d, max_order=cfg.e6_max_order)
    orders = np.asarray([len(S) for S in subsets])
    rows = []
    for it, res in zip(items, results):
        p = it.params
        n = int(p["n"])
        me = interpolate_min_effect(min_effect["table"], n)
        tau = float(res["tau"])
        ri2 = float(res["ri_pop"][0])
        for j in range(2, int(p["max_order"]) + 1):
            sel = orders == j
            rc = res["ri_corrected"][:, sel]
            q = res["q"][:, sel]
            # plug-in minus its first-order bias df/2N (df = 1): unbiased to first order, unlike max(0, .)
            excess = res["ri_plugin"][:, sel] - 1.0 / (2.0 * n)
            rows.append({"method": p["method"], "rho": float(p["rho"]), "threshold": tau,
                         "p_one": float(stats.norm.sf(tau)), "order": j,
                         "ri_population": float(res["ri_pop"][j - 2]),
                         "ratio_to_order2": float(res["ri_pop"][j - 2]) / ri2 if ri2 > 0 else math.nan,
                         "n": n, "n_rep": int(rc.shape[0]), "n_subsets": int(sel.sum()),
                         "ri_plugin_mean": _mean(res["ri_plugin"][:, sel]), "ri_corrected_mean": _mean(rc),
                         "ri_excess_mean": _mean(excess), "ri_excess_se": _se_of_rows(excess),
                         "min_effect": me, "detect_rate": _mean(((q <= cfg.alpha) & (rc >= me)).astype(float)),
                         "detect_rate_stat": _mean((q <= cfg.alpha).astype(float))})
    pop = {(r["method"], r["rho"], r["order"]): r["ri_population"] for r in rows}
    summary: dict[str, Any] = {"latent_mean": cfg.e6_latent_mean, "quantile": cfg.e6_quantile}
    if "median" in cfg.e6_methods:
        odd = [abs(v) for (m, rho, j), v in pop.items() if m == "median" and j % 2 == 1]
        even = [v for (m, rho, j), v in pop.items() if m == "median" and j % 2 == 0 and j >= 4 and rho > 0]
        n_max = max(cfg.e6_n_grid)
        sample3 = [r for r in rows if r["method"] == "median" and r["order"] == 3 and r["n"] == n_max]
        z3 = [r["ri_excess_mean"] / r["ri_excess_se"] for r in sample3
              if _finite_positive(r["ri_excess_se"])]
        summary["median_symmetry"] = {
            "max_abs_odd_order_ri": max(odd, default=math.nan),
            "min_even_order_ri": min(even, default=math.nan),
            "odd_orders_vanish": bool(odd) and max(odd) <= ZERO_TOL,
            "even_orders_nonzero": bool(even) and min(even) > ZERO_TOL,
            "sample_order3_max_abs_excess": max((abs(r["ri_excess_mean"]) for r in sample3), default=math.nan),
            "sample_order3_max_abs_z": max((abs(z) for z in z3), default=math.nan),
            "sample_n": n_max, "sample_reps": cfg.e6_reps,
        }
        summary["median_symmetry"]["verified"] = bool(summary["median_symmetry"]["odd_orders_vanish"]
                                                      and summary["median_symmetry"]["even_orders_nonzero"])
    summary["max_order3_ri"] = {m: max((v for (mm, rho, j), v in pop.items() if mm == m and j == 3), default=math.nan)
                                for m in cfg.e6_methods}
    return {"rows": rows, "summary": summary}


# ----------------------------------------------------------------------------------------------- E7 summary
def _summarize_e7(cfg: CalibrationConfig, items: Sequence[_Item],
                  results: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    rows, totals = [], []
    share_ranges: dict[int, list[float]] = {}
    for t, (it, res) in enumerate(zip(items, results)):
        cards = tuple(int(c) for c in it.params["cards"])
        d = len(cards)
        subsets = graded_chain(d)
        names = list(E7_FIXED_CHAINS) + [f"random_{j}" for j in range(int(it.params["random_chains"]))]
        ri, tc = res["ri"], float(res["tc"])
        for c, name in enumerate(names):
            totals.append({"table": t, "cards": _fmt(cards), "chain": name, "total": float(res["total"][c]),
                           "tc": tc, "abs_error": abs(float(res["total"][c]) - tc),
                           "residual": float(res["residual"][c])})
        lo, hi, mean = ri.min(axis=0), ri.max(axis=0), ri.mean(axis=0)
        for j, S in enumerate(subsets):
            rows.append({"table": t, "cards": _fmt(cards), "d": d, "subset": _subset_str(S), "order": len(S),
                         "full_set": len(S) == d, "tc": tc, "local_ri": float(res["local_ri"][j]),
                         **{f"ri_{name}": float(ri[c, j]) for c, name in enumerate(E7_FIXED_CHAINS)},
                         "ri_min": float(lo[j]), "ri_max": float(hi[j]), "ri_mean": float(mean[j]),
                         "spread": float(hi[j] - lo[j]), "spread_over_tc": float((hi[j] - lo[j]) / tc),
                         "spread_over_mean": float((hi[j] - lo[j]) / mean[j]) if mean[j] > 0 else math.nan})
        orders = np.asarray([len(S) for S in subsets])
        for o in range(2, d):  # the full set (order d) enters every chain last: its share never varies
            share = ri[:, orders == o].sum(axis=1) / tc
            share_ranges.setdefault(o, []).append(float(share.max() - share.min()))
    # Chain dependence is summarised without the full variable set: it is always the last step of a complete chain,
    # so its chain RI is the same in every chain (D(P || P_H) for H = all proper subsets) by construction.
    by_order = []
    for o in sorted({r["order"] for r in rows if not r["full_set"]}):
        sel = [r for r in rows if r["order"] == o and not r["full_set"]]
        rel = np.asarray([r["spread_over_tc"] for r in sel])
        rel_mean = np.asarray([r["spread_over_mean"] for r in sel], dtype=float)
        dev_local = np.asarray([abs(r["ri_graded"] - r["local_ri"]) / r["tc"] for r in sel])
        by_order.append({"order": o, "n_subsets": len(sel), "median_spread_over_tc": float(np.median(rel)),
                         "p90_spread_over_tc": float(np.quantile(rel, 0.9)), "max_spread_over_tc": float(rel.max()),
                         "median_spread_over_mean": float(np.nanmedian(rel_mean)),
                         "median_graded_minus_local_over_tc": float(np.median(dev_local)),
                         "median_order_share_range": float(np.median(share_ranges[o]))})
    full = [r for r in rows if r["full_set"]]
    summary = {"n_tables": len(items), "chains": list(E7_FIXED_CHAINS) + [f"random x{cfg.e7_random_chains}"],
               "max_abs_sum_minus_tc": max(r["abs_error"] for r in totals),
               "max_residual": max(abs(r["residual"]) for r in totals), "by_order": by_order,
               "max_full_set_spread": max((r["spread"] for r in full), default=math.nan)}
    return {"rows": rows, "summary": summary, "tables": {"totals": totals}}


# ----------------------------------------------------------------------------------------------- error budget
def _nearest(values: Sequence[int], target: int) -> int:
    return min(values, key=lambda v: (abs(math.log(v) - math.log(target)), v))


def _error_budget(cfg: CalibrationConfig, exps: Mapping[str, Mapping[str, Any]],
                  min_effect: Mapping[str, Any]) -> dict[str, Any]:
    """Machine-readable error budget (the body of ``calibration.json``)."""
    budget: dict[str, Any] = {"units": "nats", "alpha": cfg.alpha, "budget_n": _nearest(cfg.n_grid, cfg.budget_n),
                              "orders": list(cfg.orders), "n_grid": list(cfg.n_grid),
                              "recommended_min_effect": min_effect}
    if "E1_null" in exps:
        s = exps["E1_null"]["summary"]
        budget["noise_floor"] = s["noise_floor"]
        budget["false_positive_rate"] = s["fpr"]
    if "E2_bias" in exps:
        keep = ("order", "n", "target_ri", "true_ri", "bias_plugin", "bias_corrected", "rmse_plugin",
                "rmse_corrected", "coverage", "n_ci")
        budget["estimator_bias"] = [{k: r[k] for k in keep} for r in exps["E2_bias"]["rows"]]
    if "E3_power" in exps:
        budget["min_detectable_ri"] = exps["E3_power"]["summary"]["mde"]
    if "E4_timing" in exps:
        keep = ("width", "n", "true_onset", "detect_rate", "median_error", "q25", "q75", "mean_error", "mae",
                "frac_early", "frac_within_1")
        budget["timing"] = [{k: r[k] for k in keep} for r in exps["E4_timing"]["rows"]]
    if "E5_identity" in exps:
        budget["specificity"] = exps["E5_identity"]["rows"]
    if "E6_discretization" in exps:
        keep = ("method", "rho", "threshold", "order", "ri_population", "n", "ri_corrected_mean", "detect_rate")
        budget["artifacts"] = [{k: r[k] for k in keep} for r in exps["E6_discretization"]["rows"]]
        budget["artifact_symmetry"] = exps["E6_discretization"]["summary"].get("median_symmetry")
    if "E7_chain" in exps:
        s = exps["E7_chain"]["summary"]
        budget["chain_dependence"] = {"by_order": s["by_order"], "max_abs_sum_minus_tc": s["max_abs_sum_minus_tc"]}
    lam = {}
    for m in cfg.e3_family_sizes:
        crit = float(stats.chi2.isf(cfg.alpha / m, 1))
        lam[str(m)] = _theory_mde(1, 1, crit, cfg.e3_power) * 2.0
    budget["analytic"] = {
        "noise_floor": "Q_q(RI_plugin | H0) ~= chi2.ppf(q, df) / (2 N)",
        "mde_stat": "MDE ~= lambda* / (2 N) with ncx2.sf(chi2.isf(alpha/m, df), df, lambda*) = power",
        "lambda_star_df1": lam, "power_level": cfg.e3_power,
    }
    return budget


# ----------------------------------------------------------------------------------------------- driver
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for r in rows:
        fieldnames.extend(k for k in r if k not in fieldnames)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_jobs(n_jobs: int) -> int:
    n_jobs = int(n_jobs)
    return max(1, os.cpu_count() or 1) if n_jobs <= 0 else n_jobs


def run_calibration(cfg: CalibrationConfig, out_dir: Path, *, n_jobs: int = 1, resume: bool = True,
                    progress: Callable[[int, int], None] | None = None) -> dict:
    """Run the selected experiments and write ``<experiment>.csv`` files and ``results.json`` into ``out_dir``.

    Args:
        cfg: grids and sizes (:meth:`CalibrationConfig.quick`, :meth:`CalibrationConfig.full` or custom).
        out_dir: output directory (created). Work items are cached in ``out_dir/cache`` (about 0.6 GB for the full
            preset, a few MB for quick); the cache can be deleted once ``results.json`` exists.
        n_jobs: worker processes (spawn context); ``<= 0`` uses all CPUs. Results do not depend on it. As always
            with the spawn start method, a script that calls this with ``n_jobs > 1`` must guard its entry point
            with ``if __name__ == "__main__":``.
        resume: reuse cached items of an earlier (possibly interrupted) run with identical item parameters.
        progress: optional callback ``(items_done, items_total)``.

    Returns:
        ``{"format", "config", "preset", "n_jobs", "started_at", "finished_at", "wall_s", "compute_s" (seconds per
        item kind, computed in this call), "n_items", "n_cached", "platform", "experiments": {name: {"rows",
        "summary"[, "tables"]}}, "budget", "files"}``; pass it to
        :func:`tycho.calibration.report.write_report`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = _resolve_jobs(n_jobs)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    plan = _plan(cfg)
    order = list(plan)
    items = [it for kind in order for it in plan[kind]]
    results, seconds, n_cached = _run_items(items, cfg.seed, n_jobs=n_jobs, cache_dir=out_dir / "cache",
                                            resume=resume, progress=progress)
    grouped: dict[str, tuple[list[_Item], list[dict[str, np.ndarray]]]] = {}
    pos = 0
    for kind in order:
        size = len(plan[kind])
        grouped[kind] = (plan[kind], results[pos:pos + size])
        pos += size

    exps: dict[str, dict[str, Any]] = {}
    if "E1" in grouped:
        exps["E1_null"] = _summarize_e1(cfg, *grouped["E1"])
    min_effect = _min_effect_table(cfg, *(grouped["E1"] if "E1" in grouped else (None, None)))
    if "E2_bias" in cfg.experiments:
        exps["E2_bias"] = _summarize_e2(cfg, *grouped["E23"])
    if "E3_power" in cfg.experiments:
        exps["E3_power"] = _summarize_e3(cfg, *grouped["E23"], min_effect)
    if "E4" in grouped:
        exps["E4_timing"] = _summarize_e4(cfg, *grouped["E4"])
    if "E5" in grouped:
        exps["E5_identity"] = _summarize_e5(cfg, *grouped["E5"], min_effect)
    if "E6" in grouped:
        exps["E6_discretization"] = _summarize_e6(cfg, *grouped["E6"], min_effect)
    if "E7" in grouped:
        exps["E7_chain"] = _summarize_e7(cfg, *grouped["E7"])
    exps = {name: exps[name] for name in EXPERIMENTS if name in exps}

    files = []
    for name, exp in exps.items():
        _write_csv(out_dir / f"{name}.csv", exp["rows"])
        files.append(f"{name}.csv")
        for tname, trows in exp.get("tables", {}).items():
            _write_csv(out_dir / f"{name}_{tname}.csv", trows)
            files.append(f"{name}_{tname}.csv")
    out = {"format": RESULTS_FORMAT, "preset": cfg.name, "config": cfg.to_dict(), "n_jobs": n_jobs,
           "started_at": started.isoformat(timespec="seconds"),
           "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "wall_s": time.perf_counter() - t0, "compute_s": seconds, "n_items": len(items), "n_cached": n_cached,
           "platform": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
                        "system": platform.system()},
           "experiments": exps, "budget": _error_budget(cfg, exps, min_effect), "files": files + ["results.json"]}
    out = jsonable(out)
    tmp = out_dir / "results.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, out_dir / "results.json")
    logger.info("calibration finished in %.1fs (%s)", out["wall_s"], ", ".join(f"{k} {v:.1f}s" for k, v in
                                                                                 seconds.items()))
    return out


def load_results(out_dir: Path) -> dict:
    """Read ``out_dir/results.json`` (the dict :func:`run_calibration` returned)."""
    with open(Path(out_dir) / "results.json", "r", encoding="utf-8") as fh:
        res = json.load(fh)
    if res.get("format") != RESULTS_FORMAT:
        raise ValueError(f"unexpected results format {res.get('format')!r}")
    return res
