"""Sky v1, pre-registered question 4: how does the onset of the planted parity interaction scale with n, per order k?

Pre-registration: docs/results/v1_preregistration.md (question 4; Amendment 1 items A and C).

For each region (k = 2, 3, 4) and learning rate (0.03, 0.1) this fits log(median onset over seeds) on log n and
bootstraps the slope (2,000 resamples, fixed seed, seeds resampled within each n, refit). Four onset measures:

  (i)   registered onset: the planted subset's first activation event (ri_events, kind == "activation").
  (ii)  amended onset (Amendment 1 C, decides for RI): the first activation event (activation or reactivation) that is
        not left-censored and at whose checkpoint the planted subset's Fourier coefficient has the sign of its final
        value (last checkpoint).
  (iii) gated Fourier onset (Amendment 1 A): the first of two consecutive checkpoints where
        R = ln 2 - h((1 + |f|)/2) satisfies R - 1/(2N) >= min_effect and min(1, m * chi2_1.sf(2 N R)) <= alpha,
        m = meta["n_candidates"], N = meta["min_effect_n"].
  (iv)  t_test_acc90 (photometric reference; objects table, cross-checked against the light curves).

Runs without an onset within the budget are censored at the step budget (train_steps), never dropped.
Event steps are the first active checkpoint (tycho/analysis/events.py: step = steps[t0] = step_hi); step_lo is the
checkpoint before. Supplementary measures: the bare Fourier onset of the registered check-2 definition
(|f| >= eps with ln 2 - h((1 + eps)/2) = min_effect), (iii) with Amendment C's conditions added, (ii) with step_lo or
the geometric bracket midpoint instead of step, and alternative readings of C.

Usage (cwd = repository root):
  PYTHONPATH=. /home/user/tycho/.venv/bin/python scripts/analysis_v1/parity_scaling.py [DR1_EXTRACTED_DIR]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2

import tycho
from tycho.analysis.events import detect_events
from tycho.analysis.ricurve import RICurve

REPO = Path(__file__).resolve().parents[2]
assert Path(tycho.__file__).resolve().is_relative_to(REPO), f"wrong tycho package: {tycho.__file__}"
REL = REPO / "releases" / "DR1"
REL0 = REPO / "releases" / "DR0"
EXTRACTED = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "dr1"  # extracted DR1 archives; see README.md
OUT = REPO / "docs" / "results" / "v1_analysis" / "parity_scaling.json"

B = 2000
BOOT_SEED = 20261112
LRS = (0.03, 0.1)
KS = (2, 3, 4)
REGION = {2: "parity-k2", 3: "parity-k3", 4: "parity-k4"}
MEASURES = ("i_registered", "ii_amended", "iii_fourier_gated", "iv_t_test_acc90")
FIXED_F = (0.0447, 0.0808, 0.2, 0.5, 0.9)
"""Fixed |f| thresholds of the supplementary threshold sweep (signed, persistence 2, not at the first checkpoint):
the bare eps of min_effect, the Amendment-A gate at the largest candidate count of any parity cell (m = 43,744),
and three higher levels. A fixed threshold removes the gate's dependence on n (m grows with n)."""
SUPP = ("v_fourier_bare", "iii_fourier_gated_C", "ii_amended_step_lo", "ii_amended_geomid",
        "ii_amended_C_both_confirming", "ii_amended_C_signaware_recompute", "t_test_acc90_ri_grid") + tuple(
    f"fixed_f_{c:g}" for c in FIXED_F) + ("fixed_gate_min_m", "fixed_gate_max_m")
GATE_M_RANGE = {2: (136, 2080), 3: (696, 43744), 4: (2516, 41448)}
"""Smallest / largest n_candidates per region (n = 16 / largest n); checked against the curves' meta in load_runs."""
LN2 = math.log(2.0)
V0_HINT = {2: "about 1.0", 3: "1.2-1.6"}


def hb(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats."""
    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(1.0 - p, 1e-300, 1.0)
    return -(p * np.log(p) + q * np.log(q))


def ri_of_f(f: np.ndarray) -> np.ndarray:
    """Single-mode RI of a Walsh coefficient: ln 2 - h((1 + |f|)/2)."""
    return LN2 - hb((1.0 + np.abs(f)) / 2.0)


def eps_of_ri(ri: float) -> float:
    return brentq(lambda e: float(ri_of_f(np.array([e]))[0]) - ri, 1e-9, 1 - 1e-12)


def gate_abs_f(m: int, alpha: float = 0.01, me: float = 1e-3, N: int = 4096) -> float:
    """Smallest |f| passing both Amendment-A gate conditions for m candidates (the gate is monotone in |f|)."""
    R = max(chi2.isf(alpha / m, 1) / (2 * N), me + 1.0 / (2 * N))
    return eps_of_ri(R)


def first_pair(cond: np.ndarray, start: int = 0) -> int | None:
    """First index t >= start with cond[t] and cond[t + 1] (persistence 2)."""
    for t in range(start, cond.size - 1):
        if cond[t] and cond[t + 1]:
            return t
    return None


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not math.isfinite(x) else x
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ------------------------------------------------------------------------------------------------ per-run table
def load_runs() -> tuple[pd.DataFrame, dict]:
    o = pd.read_parquet(REL / "objects.parquet")
    runs = pd.read_parquet(REL / "ri_runs.parquet")
    ev = pd.read_parquet(REL / "ri_events.parquet")
    p = o[o.region.isin(REGION.values())].copy()
    cfg = p.task_cfg_json.map(json.loads)
    p["n"] = cfg.map(lambda c: int(c["n"]))
    p["k"] = cfg.map(lambda c: int(c["k"]))
    p["width"] = cfg.map(lambda c: int(c["width"]))
    grids: dict = {}
    checks = dict(n_objects=len(p), recompute_mismatch=0, gt_missing=0, gt_events_table_vs_recomputed="identical",
                  t_test_acc90_mismatch=0, ri_grid_subset_of_lc_grid=True)
    rows = []
    for _, ob in p.sort_values(["k", "train_lr", "n", "seed"]).iterrows():
        rr = runs[runs.object_id == ob.object_id]
        assert len(rr) == 1 and rr.iloc[0].kind == "insitu", ob.object_id
        rr = rr.iloc[0]
        c = RICurve.load(EXTRACTED / rr.ricurve_file)
        gti = c.ground_truth_indices()
        if len(gti) != 1 or len(c.ground_truth) != 1:
            checks["gt_missing"] += 1
            raise RuntimeError(f"ground truth not found once in {ob.object_id}: {c.ground_truth} {gti}")
        g = gti[0]
        names = c.subset_names(g)
        assert len(names) == ob.k + 1 and names[-1] == "yhat", names
        steps = c.steps.astype(np.int64)
        T = steps.size
        assert steps[0] == 0 and steps[-1] == ob.train_steps, (steps[0], steps[-1], ob.train_steps)
        ri, q, f = c.ri[g], c.q[g], c.fourier[g]
        meta = c.meta
        alpha, me, N, m = float(meta["alpha"]), float(meta["min_effect"]), int(meta["min_effect_n"]), int(
            meta["n_candidates"])
        assert alpha == 0.01 and np.isfinite(f).all()
        # events: table rows of this run's planted subset, cross-checked against a recomputation
        tab = ev[(ev.ri_run_id == rr.ri_run_id) & (ev.subset_json.map(lambda s: frozenset(json.loads(s)))
                                                   == frozenset(names))].sort_values(["step", "event_id"])
        rec = [e for e in detect_events(c, alpha=0.01) if e.subset == c.subsets[g]]
        if [(e.kind, e.step, e.step_lo) for e in rec] != list(zip(tab.kind, tab.step, tab.step_lo)):
            checks["recompute_mismatch"] += 1
        assert bool(tab.matches_gt.all()) if len(tab) else True
        idx = {int(s): i for i, s in enumerate(steps)}
        f_final = float(f[-1])
        sgn_final = np.sign(f_final)
        budget = int(ob.train_steps)

        # (i) registered: first event of kind "activation"
        acts = tab[tab.kind == "activation"]
        if len(acts):
            a0 = acts.iloc[0]
            i_step, i_lo, i_cens = int(a0.step), int(a0.step_lo), False
            i_left = bool(idx[i_step] == 0)
        else:
            i_step, i_lo, i_cens, i_left = budget, np.nan, True, False
        # (ii) amended (C): first activation/reactivation not left-censored with sign(f[t0]) == sign(final)
        starts = tab[tab.kind.isin(["activation", "reactivation"])]
        ii = ii_both = ii_actonly = None
        for _, e in starts.iterrows():
            t0 = idx[int(e.step)]
            if t0 == 0:
                continue
            ok = np.sign(f[t0]) == sgn_final and sgn_final != 0
            ok2 = ok and t0 + 1 < T and np.sign(f[t0 + 1]) == sgn_final
            if ok and ii is None:
                ii = (t0, int(e.step), int(e.step_lo), e.kind)
            if ok2 and ii_both is None:
                ii_both = t0
            if ok and e.kind == "activation" and ii_actonly is None:
                ii_actonly = t0
        # sign-aware recomputation: an active checkpoint must also have the final sign (persistence 2, t0 > 0)
        with np.errstate(invalid="ignore"):
            active = (q <= alpha) & (ri >= me)
        t_sa = first_pair(active & (np.sign(f) == sgn_final), start=1)
        # (iii) gated Fourier onset (A)
        R = ri_of_f(f)
        with np.errstate(invalid="ignore"):
            gate = (R - 1.0 / (2 * N) >= me) & (np.minimum(1.0, m * chi2.sf(2 * N * R, 1)) <= alpha)
        t_A = first_pair(gate)
        t_AC = first_pair(gate & (np.sign(f) == sgn_final), start=1)
        # (v) bare Fourier onset (registered check-2 definition)
        eps = eps_of_ri(me)
        bare = np.flatnonzero(np.abs(f) >= eps)
        t_bare = int(bare[0]) if bare.size else None
        t_fixed = {c: first_pair(sgn_final * f >= c, start=1) for c in FIXED_F}
        assert GATE_M_RANGE[int(ob.k)][0] <= m <= GATE_M_RANGE[int(ob.k)][1], (ob.k, m)
        t_gmin = first_pair(sgn_final * f >= gate_abs_f(GATE_M_RANGE[int(ob.k)][0]), start=1)
        t_gmax = first_pair(sgn_final * f >= gate_abs_f(GATE_M_RANGE[int(ob.k)][1]), start=1)
        # (iv) t_test_acc90, cross-checked from the light curve; also on the RI grid
        lc = np.load(EXTRACTED / ob.lightcurve_file)
        ls, ta = lc["step"].astype(np.int64), lc["test_acc"]
        hit = np.flatnonzero(np.isfinite(ta) & (ta >= 0.9 - 1e-9))
        t90_lc = float(ls[hit[0]]) if hit.size else np.nan
        t90_lo = float(ls[hit[0] - 1]) if hit.size and hit[0] > 0 else np.nan
        t90 = float(ob.t_test_acc90)
        if not ((math.isnan(t90) and math.isnan(t90_lc)) or t90 == t90_lc):
            checks["t_test_acc90_mismatch"] += 1
        if not set(steps.tolist()) <= set(ls.tolist()):
            checks["ri_grid_subset_of_lc_grid"] = False
        ta_ri = np.array([ta[np.searchsorted(ls, s)] for s in steps])
        if ob.region not in grids:
            ratio = steps[1:] / np.maximum(steps[:-1], 1)
            big = steps[1:] >= 10
            grids[ob.region] = dict(ri_points=int(steps.size), lc_points=int(ls.size),
                                    ri_step_ratio_min=float(ratio[big].min()), ri_step_ratio_max=float(ratio[big].max()))
        assert grids[ob.region]["ri_points"] == steps.size and grids[ob.region]["lc_points"] == ls.size
        hit_ri = np.flatnonzero(np.isfinite(ta_ri) & (ta_ri >= 0.9 - 1e-9))
        t90_ri_idx = int(hit_ri[0]) if hit_ri.size else None

        def st(t):
            return budget if t is None else int(steps[t])

        ii_t = None if ii is None else ii[0]
        row = dict(
            object_id=ob.object_id, region=ob.region, k=int(ob.k), lr=float(ob.train_lr), n=int(ob.n), seed=int(ob.seed),
            source=ob.source, budget=budget, label=ob.label, ri_run_id=int(rr.ri_run_id), gt_subset=list(names),
            n_candidates=m, N=N, min_effect=me, alpha=alpha, eps_bare=eps,
            n_gt_events=len(tab), gt_event_kinds=",".join(tab.kind.tolist()),
            i_registered=float(i_step), i_censored=i_cens, i_left_censored=i_left, i_step_lo=i_lo,
            ii_amended=float(st(ii_t)), ii_censored=ii is None, ii_event_kind=None if ii is None else ii[3],
            ii_amended_step_lo=float(budget if ii is None else ii[2]),
            ii_amended_geomid=float(budget if ii is None else math.sqrt(max(ii[2], 1) * ii[1])),
            ii_amended_C_both_confirming=float(st(ii_both)), ii_amended_C_activation_only=float(st(ii_actonly)),
            ii_amended_C_signaware_recompute=float(st(t_sa)),
            iii_fourier_gated=float(st(t_A)), iii_censored=t_A is None, iii_at_first_checkpoint=t_A == 0,
            iii_fourier_gated_C=float(st(t_AC)), iii_C_censored=t_AC is None,
            v_fourier_bare=float(st(t_bare)), v_censored=t_bare is None, v_at_first_checkpoint=t_bare == 0,
            iv_t_test_acc90=float(budget if math.isnan(t90) else t90), iv_censored=bool(math.isnan(t90)),
            iv_step_lo=t90_lo,
            fixed_gate_min_m=float(st(t_gmin)), fixed_gate_min_m_censored=t_gmin is None,
            fixed_gate_max_m=float(st(t_gmax)), fixed_gate_max_m_censored=t_gmax is None,
            **{f"fixed_f_{c:g}": float(st(t)) for c, t in t_fixed.items()},
            **{f"fixed_f_{c:g}_censored": t is None for c, t in t_fixed.items()},
            t_test_acc90_ri_grid=float(st(t90_ri_idx)),
            idx_ii=ii_t, idx_iii=t_A, idx_v=t_bare, idx_t90_ri_grid=t90_ri_idx,
            rise_ckpts_0p0808_to_0p5=(None if t_fixed[0.0808] is None or t_fixed[0.5] is None
                                      else int(t_fixed[0.5] - t_fixed[0.0808])),
            f_at_ii=None if ii_t is None else float(f[ii_t]), f_final=f_final, f_first_checkpoint=float(f[0]),
            ri_at_ii=None if ii_t is None else float(ri[ii_t]), q_at_ii=None if ii_t is None else float(q[ii_t]),
            standin_at_ii=None if ii_t is None else float(min(1.0, m * chi2.sf(2 * N * R[ii_t], 1))),
            abs_f_at_ii=None if ii_t is None else float(abs(f[ii_t])),
            ri_first_checkpoint=float(ri[0]), q_first_checkpoint=float(q[0]), max_abs_f_init=float(abs(f[0])),
            test_acc_at_ii=None if ii_t is None else float(ta_ri[ii_t]),
            final_test_acc=float(ob.final_test_acc), t_train_acc90=float(ob.t_train_acc90),
        )
        rows.append(row)
    d = pd.DataFrame(rows)
    d["lead_steps"] = d.iv_t_test_acc90 - d.ii_amended
    d["ratio_onset_over_t90"] = d.ii_amended / d.iv_t_test_acc90
    d["lead_ri_checkpoints"] = d.idx_t90_ri_grid - d.idx_ii
    d["ii_minus_iii_checkpoints"] = d.idx_ii - d.idx_iii
    d["iii_minus_v_checkpoints"] = d.idx_iii - d.idx_v
    checks["grids"] = grids
    return d, checks


# ------------------------------------------------------------------------------------------------ statistics
def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm = x.mean()
    s = ((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum()
    return float(s), float(y.mean() - s * xm)


def boot_slopes(logmed: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised OLS slopes of (B, n_x) log medians on x."""
    xc = x - x.mean()
    return (logmed - logmed.mean(axis=1, keepdims=True)) @ xc / (xc ** 2).sum()


def ci(a: np.ndarray) -> list[float]:
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def cell_stats(v: np.ndarray, cens: np.ndarray) -> dict:
    return dict(n_runs=int(v.size), n_censored=int(cens.sum()), median=float(np.median(v)),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), min=float(v.min()),
                max=float(v.max()), median_is_censored_bound=bool(cens.sum() * 2 >= v.size))


CENS_COL = {"i_registered": "i_censored", "ii_amended": "ii_censored", "iii_fourier_gated": "iii_censored",
            "iv_t_test_acc90": "iv_censored", "v_fourier_bare": "v_censored", "iii_fourier_gated_C": "iii_C_censored",
            "ii_amended_step_lo": "ii_censored", "ii_amended_geomid": "ii_censored",
            "ii_amended_C_both_confirming": None, "ii_amended_C_signaware_recompute": None,
            "t_test_acc90_ri_grid": None, **{f"fixed_f_{c:g}": f"fixed_f_{c:g}_censored" for c in FIXED_F},
            "fixed_gate_min_m": "fixed_gate_min_m_censored", "fixed_gate_max_m": "fixed_gate_max_m_censored"}


def analyse(d: pd.DataFrame) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    jrng = np.random.default_rng(BOOT_SEED + 2)
    allm = MEASURES + SUPP
    out: dict = {}
    for k in KS:
        for lr in LRS:
            g = d[(d.k == k) & (d.lr == lr)]
            ns = sorted(int(v) for v in g.n.unique())
            x = np.log(np.array(ns, dtype=float))
            cells = {n: g[g.n == n].sort_values("seed") for n in ns}
            # one set of resample indices per (k, lr): all measures are paired
            draws = {n: rng.integers(0, len(cells[n]), size=(B, len(cells[n]))) for n in ns}
            res = dict(k=k, lr=lr, n=ns, seeds_per_n=[len(cells[n]) for n in ns], measures={})
            boot_store = {}
            for meas in allm:
                cc = CENS_COL[meas]
                stats = {}
                meds = []
                for n in ns:
                    v = cells[n][meas].to_numpy(float)
                    cens = cells[n][cc].to_numpy(bool) if cc else np.zeros(v.size, bool)
                    stats[str(n)] = cell_stats(v, cens)
                    meds.append(np.median(v))
                meds = np.array(meds)
                y = np.log(meds)
                slope, icpt = ols(x, y)
                resid = y - (icpt + slope * x)
                ss_tot = ((y - y.mean()) ** 2).sum()
                bm = np.stack([np.median(cells[n][meas].to_numpy(float)[draws[n]], axis=1) for n in ns], axis=1)
                blog = np.log(bm)
                bs = boot_slopes(blog, x)
                boot_store[meas] = (blog, bs)
                fit = dict(slope=slope, intercept=intercept_safe(icpt), ci95=ci(bs), boot_se=float(np.std(bs, ddof=1)),
                           r2=float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else None,
                           residuals_log=resid.tolist(), max_abs_residual_log=float(np.abs(resid).max()),
                           rmse_log=float(np.sqrt((resid ** 2).mean())),
                           reference_exponent=k - 1, reference_in_ci=bool(ci(bs)[0] <= k - 1 <= ci(bs)[1]))
                # sub-range slopes
                sub = {}
                ranges = ([(16, 24, 32), (32, 48, 64)] if len(ns) == 5 else [(16, 24), (24, 32)])
                for rng_n in ranges:
                    cols = [ns.index(n) for n in rng_n]
                    xs = x[cols]
                    s_, _ = ols(xs, y[cols])
                    bss = boot_slopes(blog[:, cols], xs)
                    sub[f"{rng_n[0]}-{rng_n[-1]}"] = dict(n=list(rng_n), slope=s_, ci95=ci(bss))
                    boot_store[(meas, rng_n)] = bss
                a, b_ = ranges
                diff = boot_store[(meas, b_)] - boot_store[(meas, a)]
                sub["upper_minus_lower"] = dict(slope=sub[f"{b_[0]}-{b_[-1]}"]["slope"] - sub[f"{a[0]}-{a[-1]}"]["slope"],
                                                ci95=ci(diff))
                fit["subrange"] = sub
                # per-run OLS of log onset on log n (supplementary)
                lv = np.log(g[meas].to_numpy(float))
                lx = np.log(g.n.to_numpy(float))
                fit["per_run_ols_slope"] = ols(lx, lv)[0]
                res["measures"][meas] = dict(cells=stats, medians=meds.tolist(), fit=fit)
            # paired slope differences
            pairs = [("ii_amended", "iii_fourier_gated"), ("ii_amended", "iv_t_test_acc90"),
                     ("iii_fourier_gated", "iv_t_test_acc90"), ("i_registered", "ii_amended"),
                     ("ii_amended", "v_fourier_bare")]
            res["paired_slope_differences"] = {
                f"{a}-{b}": dict(diff=res["measures"][a]["fit"]["slope"] - res["measures"][b]["fit"]["slope"],
                                 ci95=ci(boot_store[a][1] - boot_store[b][1])) for a, b in pairs}
            # lead of the amended RI onset over t_test_acc90
            lead = {}
            for n in ns:
                c_ = cells[n]
                lead[str(n)] = dict(
                    lead_steps_median=float(c_.lead_steps.median()), lead_steps_q25=float(c_.lead_steps.quantile(.25)),
                    lead_steps_q75=float(c_.lead_steps.quantile(.75)),
                    ratio_median=float(c_.ratio_onset_over_t90.median()),
                    ratio_q25=float(c_.ratio_onset_over_t90.quantile(.25)),
                    ratio_q75=float(c_.ratio_onset_over_t90.quantile(.75)),
                    lead_ri_checkpoints_median=float(c_.lead_ri_checkpoints.median()),
                    n_lead_pos=int((c_.lead_steps > 0).sum()), n_lead_zero=int((c_.lead_steps == 0).sum()),
                    n_lead_neg=int((c_.lead_steps < 0).sum()), test_acc_at_onset_median=float(c_.test_acc_at_ii.median()),
                    ratio_of_medians=float(c_.ii_amended.median() / c_.iv_t_test_acc90.median()))
            res["lead_vs_t_test_acc90"] = lead
            res["lead_vs_t_test_acc90_pooled_over_n"] = dict(
                n_runs=int(len(g)), lead_steps_median=float(g.lead_steps.median()),
                ratio_median=float(g.ratio_onset_over_t90.median()),
                ratio_q25=float(g.ratio_onset_over_t90.quantile(.25)),
                ratio_q75=float(g.ratio_onset_over_t90.quantile(.75)),
                lead_ri_checkpoints_median=float(g.lead_ri_checkpoints.median()),
                n_same_ri_checkpoint=int((g.lead_ri_checkpoints == 0).sum()),
                n_ri_checkpoints_ahead_ge1=int((g.lead_ri_checkpoints >= 1).sum()),
                n_ri_after_on_ri_grid=int((g.lead_ri_checkpoints < 0).sum()),
                n_lead_steps_pos=int((g.lead_steps > 0).sum()), n_lead_steps_zero=int((g.lead_steps == 0).sum()),
                n_lead_steps_neg=int((g.lead_steps < 0).sum()))
            # bracket-jittered bootstrap: each resampled onset drawn log-uniformly within its bracket (lo, step]
            jit = {}
            for meas, lo_col in (("ii_amended", "ii_amended_step_lo"), ("iv_t_test_acc90", "iv_step_lo")):
                bm = []
                for n in ns:
                    lo = cells[n][lo_col].to_numpy(float)
                    hi = cells[n][meas].to_numpy(float)
                    assert np.all(lo > 0) and np.all(lo < hi), (meas, n)
                    dr = draws[n]
                    u = jrng.random(dr.shape)
                    bm.append(np.median(np.exp(np.log(lo[dr]) + u * (np.log(hi[dr]) - np.log(lo[dr]))), axis=1))
                blog = np.log(np.stack(bm, axis=1))
                bs = boot_slopes(blog, x)
                entry = dict(ci95=ci(bs), boot_median_slope=float(np.median(bs)), subrange={})
                ranges = ([(16, 24, 32), (32, 48, 64)] if len(ns) == 5 else [(16, 24), (24, 32)])
                sbs = []
                for rng_n in ranges:
                    cols = [ns.index(n) for n in rng_n]
                    sb = boot_slopes(blog[:, cols], x[cols])
                    sbs.append(sb)
                    entry["subrange"][f"{rng_n[0]}-{rng_n[-1]}"] = dict(ci95=ci(sb), boot_median=float(np.median(sb)))
                entry["subrange"]["upper_minus_lower"] = dict(ci95=ci(sbs[1] - sbs[0]),
                                                              boot_median=float(np.median(sbs[1] - sbs[0])))
                jit[meas] = entry
            res["bracket_jittered_bootstrap"] = jit
            out[f"k{k}_lr{lr}"] = res
    return out


def intercept_safe(v: float) -> float:
    return float(v)


def v0_baseline() -> dict:
    """v0 hint recomputed from DR0 with scripts/analyze_dr0.py's rule: parity-wide, online data, learned runs
    (final test acc >= 0.9), median t_test_acc90 per n, n = 16, 24, 32; pooled widths and width 512 only."""
    o = pd.read_parquet(REL0 / "objects.parquet")
    p = o[o.region == "parity-wide"].copy()
    cfg = pd.json_normalize(p.task_cfg_json.map(json.loads))
    cfg.index = p.index
    p = pd.concat([p, cfg.add_prefix("c_")], axis=1)
    p = p[p.c_n_train.isna()]
    rng = np.random.default_rng(BOOT_SEED + 1)
    out = {}
    for k in (2, 3):
        for lr in LRS:
            for w in (None, 512):
                g = p[(p.c_k == k) & (p.train_lr == lr) & p.c_n.isin([16, 24, 32])]
                if w:
                    g = g[g.c_width == w]
                n_all = len(g)
                g = g[g.final_test_acc >= 0.9]
                ns = [16, 24, 32]
                x = np.log(np.array(ns, float))
                vals = [g[g.c_n == n].t_test_acc90.to_numpy(float) for n in ns]
                meds = np.array([np.median(v) for v in vals])
                s, _ = ols(x, np.log(meds))
                bm = np.stack([np.median(v[rng.integers(0, v.size, (B, v.size))], axis=1) for v in vals], axis=1)
                bs = boot_slopes(np.log(bm), x)
                out[f"k{k}_lr{lr}_{'width512' if w else 'pooled_widths'}"] = dict(
                    k=k, lr=lr, widths=sorted(g.c_width.unique().tolist()), n=ns,
                    train_steps=sorted(int(v) for v in g.train_steps.unique()), runs=[int(v.size) for v in vals],
                    runs_before_learned_filter=n_all, median_t_test_acc90=meds.tolist(), slope=s, ci95=ci(bs))
    return out


# ------------------------------------------------------------------------------------------------ items
def _sl(fit: dict) -> str:
    return f"{fit['slope']:.2f} [{fit['ci95'][0]:.2f}, {fit['ci95'][1]:.2f}]"


def _cells(r: dict, m: str) -> str:
    return "; ".join(f"n={n}: {c['median']:g} (IQR {c['q25']:g}-{c['q75']:g})"
                     for n, c in r["measures"][m]["cells"].items())


def build_items(res: dict, sanity: dict, cens: dict, v0: dict, d: pd.DataFrame) -> list[dict]:
    K = [(k, lr) for k in KS for lr in LRS]
    R = {kl: res[f"k{kl[0]}_lr{kl[1]}"] for kl in K}
    F = lambda kl, m: R[kl]["measures"][m]["fit"]  # noqa: E731
    n_runs = {kl: int(sum(R[kl]["seeds_per_n"])) for kl in K}
    tot_cens = {m: sum(v.values()) for m, v in cens.items()}
    sizes = "; ".join(f"k={k} lr={lr}: seeds per n {R[(k, lr)]['seeds_per_n']} ({n_runs[(k, lr)]} runs)"
                      for k, lr in K)
    lr_ratio = {k: [R[(k, 0.03)]["measures"]["ii_amended"]["medians"][i] / R[(k, 0.1)]["measures"]["ii_amended"][
        "medians"][i] for i in range(len(R[(k, 0.1)]["n"]))] for k in KS}
    san = sanity
    rbg = san["ri_before_gate"]
    sw = lambda kk, c: [F((kk, lr), f"fixed_f_{c:g}")["slope"] for lr in LRS]  # noqa: E731
    step_lo_drop = max(F(kl, "ii_amended")["slope"] - F(kl, "ii_amended_step_lo")["slope"] for kl in K)
    step_lo_drop_kl = max(K, key=lambda kl: F(kl, "ii_amended")["slope"] - F(kl, "ii_amended_step_lo")["slope"])
    gate_fixed = {kl: [F(kl, "fixed_gate_min_m")["slope"], F(kl, "fixed_gate_max_m")["slope"]] for kl in K}
    gate_drop = {kl: F(kl, "ii_amended")["slope"] - max(gate_fixed[kl]) for kl in K}
    compat = [f"{m_} k={k} lr={lr} ({_sl(F((k, lr), m_))})" for k, lr in K for m_ in MEASURES[1:]
              if F((k, lr), m_)["reference_in_ci"]]
    sub2 = {kl: F(kl, "ii_amended")["subrange"] for kl in K}
    rng_ = lambda vals: (f"{min(vals):.2f}" if f"{min(vals):.2f}" == f"{max(vals):.2f}"  # noqa: E731
                         else f"{min(vals):.2f}-{max(vals):.2f}")
    sl_of = lambda m_, ks_: [F((k, lr), m_)["slope"] for k in ks_ for lr in LRS]  # noqa: E731
    low_meas = ("i_registered", "ii_amended", "iii_fourier_gated", "v_fourier_bare", "fixed_f_0.0447",
                "fixed_f_0.0808", "fixed_gate_min_m", "fixed_gate_max_m")
    low_all = [F(kl, m_)["slope"] for kl in K for m_ in low_meas]
    gate_absdiff_k34 = max(abs(F((k, lr), "ii_amended")["slope"] - v) for k in (3, 4) for lr in LRS
                           for v in gate_fixed[(k, lr)])
    diff_iii = {kl: R[kl]["paired_slope_differences"]["ii_amended-iii_fourier_gated"]["diff"] for kl in K}
    off_k2 = {int(a): b for a, b in san["ii_minus_iii_by_k"]["2"].items()}
    lead_ck = {k: {int(a): b for a, b in san["lead_ri_checkpoints_by_k"][str(k)].items()} for k in KS}
    ratio_n = {kl: [v["ratio_median"] for v in R[kl]["lead_vs_t_test_acc90"].values()] for kl in K}
    common_checks = (
        f"Checks: planted subset found once in {san['n_objects'] - san['gt_missing']} of {san['n_objects']} curves; "
        f"ri_events rows of the planted subset equal detect_events(curve, alpha=0.01) in "
        f"{san['n_objects'] - san['recompute_mismatch']} of {san['n_objects']}; objects.t_test_acc90 equals the "
        f"light-curve recomputation in {san['n_objects'] - san['t_test_acc90_mismatch']} of {san['n_objects']}; "
        f"final Walsh coefficient negative in {san['n_f_final_negative']} of {san['n_objects']} "
        f"(range {san['f_final_range'][0]:.5f} to {san['f_final_range'][1]:.5f}; learned parity = -1 convention); "
        f"minimum final test accuracy {san['final_test_acc_min']:g}.")
    items = []

    # ---------------------------------------------------------------- q4 main
    lit = "; ".join(f"k={k} lr={lr}: {_sl(F((k, lr), 'i_registered'))}" for k, lr in K)
    amd = "; ".join(f"k={k} lr={lr}: {_sl(F((k, lr), 'ii_amended'))}" for k, lr in K)
    ref = "; ".join(f"k={k} lr={lr}: reference {k - 1} {'inside' if F((k, lr), 'ii_amended')['reference_in_ci'] else 'outside'} CI"
                    for k, lr in K)
    kn = [f"slope (ii) amended onset k={k} lr={lr}: {_sl(F((k, lr), 'ii_amended'))} (R2 {F((k, lr), 'ii_amended')['r2']:.3f})"
          for k, lr in K]
    kn += [f"medians (ii) k={k} lr={lr}: {[round(v, 1) for v in R[(k, lr)]['measures']['ii_amended']['medians']]} at n={R[(k, lr)]['n']}"
           for k, lr in K]
    kn += [f"(ii) median (IQR) k={k} lr={lr}: " + _cells(R[(k, lr)], "ii_amended") for k, lr in K]
    kn += [f"(i) per-cell medians identical to (ii) in every cell: "
           f"{all(R[kl]['measures']['i_registered']['cells'] == R[kl]['measures']['ii_amended']['cells'] for kl in K)}"]
    kn += [f"runs where (ii) != (i): {san['n_ii_differs_from_i']} of {san['n_objects']}; left-censored planted activations: "
           f"{san['n_registered_left_censored']}; wrong sign at activation: {san['n_registered_wrong_sign_at_onset']}",
           f"censored runs (no onset within budget): (i) {tot_cens['i_registered']}, (ii) {tot_cens['ii_amended']} of {san['n_objects']}",
           f"v1 lr=0.1 n=16-32 slope (ii): k=2 {_sl(R[(2, 0.1)]['measures']['ii_amended']['fit']['subrange']['16-32'])}, "
           f"k=3 {_sl(R[(3, 0.1)]['measures']['ii_amended']['fit']['subrange']['16-32'])}",
           "median onset ratio lr0.03/lr0.1 per n: " + "; ".join(
               f"k={k}: {[round(x, 2) for x in lr_ratio[k]]}" for k in KS)]
    items.append(dict(
        id="q4",
        question="Q4 (pre-registered): sparse parity, how does the onset of the planted interaction scale with n, per "
                 "order k? Fit log(median onset over seeds) on log n for each k and lr with a bootstrap over seeds; "
                 "reference n^(k-1) (exponents 1, 2, 3); v0 hint about 1.0 (k=2) and 1.2-1.6 (k=3).",
        definition_used=(
            "Literal (registered): onset = step of the planted subset's first ri_events activation (kind == "
            "'activation'); the event step is the first active checkpoint steps[t0] (events.py), so the onset lies in "
            "(step_lo, step]. Amended (Amendment 1 C, decides): the first activation or reactivation event that is not "
            "left-censored (t0 > 0) and at whose checkpoint t0 the planted subset's Walsh coefficient has the sign of "
            "its value at the last checkpoint. Median over all seeds of the current sky (coarse + refinement) at each "
            "(k, lr, n): " + sizes + ". OLS of ln(median) on ln(n) over n = 16, 24, 32, 48, 64 (k=4: 16, 24, 32). "
            f"95 % percentile bootstrap, {B} resamples, seed {BOOT_SEED}, seeds resampled with replacement within each "
            "n, medians recomputed and refit; the same resample indices for every measure within (k, lr), so slope "
            "differences are paired. Runs without an onset are set to the step budget (20k/60k/150k). Ambiguities "
            "and alternatives: (a) 'activation event' read as activation or reactivation (alternative: kind == "
            "'activation' only); (b) sign checked at t0 (alternatives: at both confirming checkpoints t0 and t0+1; "
            "a sign-aware recomputation from the curve, q <= 0.01, ri >= min_effect and final sign at two "
            "consecutive checkpoints, t0 > 0). Every alternative gives the same onset in "
            f"{san['n_objects'] - max(san['n_ii_variants_differ'].values())} of {san['n_objects']} runs."),
        literal_result="Slope [95 % CI] of ln(median registered onset) on ln n: " + lit + f". Censored runs: "
                       f"{tot_cens['i_registered']} of {san['n_objects']}.",
        literal_verdict="descriptive",
        amended_result="Identical to the literal result: Amendment C changes no parity onset ("
                       f"{san['n_ii_differs_from_i']} of {san['n_objects']} runs differ). Every planted subset has "
                       "exactly one event (an activation; no extinction or reactivation), none at the first "
                       "checkpoint (RI = 0 and q > 0.01 at step 0 in all runs; max |f| at step 0 = "
                       f"{san['abs_f_first_checkpoint_max']:.4f}), and the coefficient at the activation checkpoint "
                       "has the final (negative) sign in every run. Slopes: " + amd + ". Reference: " + ref + ".",
        amended_verdict="descriptive",
        deciding="amended",
        key_numbers="\n".join(kn),
        interpretation=(
            "The planted-interaction onset grows as a power of n with exponent "
            f"{rng_(sl_of('ii_amended', (2,)))} for k=2, {rng_(sl_of('ii_amended', (3,)))} for k=3 and "
            f"{rng_(sl_of('ii_amended', (4,)))} for k=4 (3 points). The exponent rises only weakly with k, far less than the n^(k-1) "
            "reference: for k=2 it is well above 1 (CIs exclude 1), for k=3 the CI includes 2 at lr 0.03 "
            f"({_sl(F((3, 0.03), 'ii_amended'))}) but not at lr 0.1 ({_sl(F((3, 0.1), 'ii_amended'))}), and for "
            f"k=4 it is well below 3 ({_sl(F((4, 0.1), 'ii_amended'))}). The exponent barely depends on the learning "
            "rate for k=3 and k=4; lr 0.03 onsets are "
            f"{min(min(v) for v in lr_ratio.values()):.1f}-{max(max(v) for v in lr_ratio.values()):.1f} times the "
            "lr 0.1 onsets (lr ratio 3.3). Against the v0 hint, "
            "the RI onset exponent on the v0 range (n=16-32, lr 0.1) is about 2 for both k=2 and k=3, larger than "
            "the v0 numbers, which were exponents of the steps to 90 % test accuracy (see q4_photometric: that "
            "measure reproduces v0). Measures compatible with the reference (CI contains k-1): "
            + ("; ".join(compat) if compat else "none") + ". The onset is a low-threshold crossing (detection threshold |f| "
            f"{min(v['abs_f'] for v in san['gate_abs_f_threshold_by_cell'].values()):.3f}-"
            f"{max(v['abs_f'] for v in san['gate_abs_f_threshold_by_cell'].values()):.3f}; median |f| at the onset "
            "checkpoint " + ", ".join(f"k={k}: {san['abs_f_at_onset_by_k'][str(k)]['median']:.3f}" for k in KS)
            + "), and its exponent depends on the threshold (q4_fit_strength): for k=2 the exponent falls from "
            f"{rng_(sw(2, 0.0447))} at |f| >= 0.0447 to {rng_(sw(2, 0.9))} at |f| >= 0.9, so 'the' exponent of "
            "parity learning is not a single number here."),
        caveats=(
            "Three- and five-point log-log fits over a factor of 4 (k=4: factor 2) in n are weak evidence of a power "
            "law; for k=2 the local slope falls significantly from n=16-32 to n=32-64 (see q4_fit_strength). Onsets "
            "are quantised to the 60-point log grid: above step 10, consecutive checkpoints differ by a factor "
            f"{min(g_['ri_step_ratio_min'] for g_ in san['grids'].values()):.2f}-"
            f"{max(g_['ri_step_ratio_max'] for g_ in san['grids'].values()):.2f}, i.e. "
            f"{math.log(min(g_['ri_step_ratio_min'] for g_ in san['grids'].values())):.2f}-"
            f"{math.log(max(g_['ri_step_ratio_max'] for g_ in san['grids'].values())):.2f} in ln, comparable to the "
            "fit residuals; the percentile bootstrap of grid-valued medians is "
            "lumpy and can be asymmetric around the point estimate (e.g. k=2 lr 0.03: "
            f"{_sl(F((2, 0.03), 'ii_amended'))}). Using step_lo instead of the event step lowers slopes by up to "
            f"{step_lo_drop:.2f} (k={step_lo_drop_kl[0]} lr={step_lo_drop_kl[1]}); a bracket-jittered bootstrap is "
            "reported in q4_fit_strength. For k=2, holding the detection threshold fixed in n gives exponents "
            f"{rng_(gate_fixed[(2, 0.03)])} (lr 0.03) and {rng_(gate_fixed[(2, 0.1)])} (lr 0.1), "
            f"{gate_drop[(2, 0.03)]:.2f} and {gate_drop[(2, 0.1)]:.2f} below the RI exponent. So part of the k=2 "
            "exponent comes from a detection rule that changes with n: the number of scanned candidates grows from "
            f"m = {GATE_M_RANGE[2][0]} at n=16 to {GATE_M_RANGE[2][1]} at n=64. " + common_checks)))

    # ---------------------------------------------------------------- q4 Fourier (A)
    dif = "; ".join(f"k={k} lr={lr}: (ii)-(iii) {R[(k, lr)]['paired_slope_differences']['ii_amended-iii_fourier_gated']['diff']:.3f} "
                    f"[{R[(k, lr)]['paired_slope_differences']['ii_amended-iii_fourier_gated']['ci95'][0]:.3f}, "
                    f"{R[(k, lr)]['paired_slope_differences']['ii_amended-iii_fourier_gated']['ci95'][1]:.3f}]" for k, lr in K)
    off = san["ii_minus_iii_by_k"]
    off_s = "; ".join(f"k={k}: {off[str(k)]}" for k in KS)
    bare = "; ".join(f"k={k} lr={lr}: {_sl(F((k, lr), 'v_fourier_bare'))}" for k, lr in K)
    gated = "; ".join(f"k={k} lr={lr}: {_sl(F((k, lr), 'iii_fourier_gated'))}" for k, lr in K)
    gthr = san["gate_abs_f_threshold_by_cell"]
    items.append(dict(
        id="q4_fourier_A",
        question="Q4 with the Fourier baseline: does the onset exponent from the Walsh coefficient of the planted "
                 "inputs agree with the RI exponent (it should, with uniform probes)? Amendment 1 A defines the "
                 "gated Fourier onset.",
        definition_used=(
            "Registered Fourier onset (literal, check-2 definition): first checkpoint with |f| >= eps, eps = "
            f"{san['eps_bare']:.4f} solving ln 2 - h((1+eps)/2) = min_effect = 0.001 (no persistence). Amended "
            "(A, decides): first of two consecutive checkpoints where R = ln 2 - h((1+|f|)/2) satisfies R - 1/(2N) >= "
            "min_effect and min(1, m chi2_1.sf(2NR)) <= alpha = 0.01, with m = meta n_candidates (136 to 43,744 "
            "depending on k and n) and N = meta min_effect_n = 4096; f = the stored Walsh coefficient of the planted "
            "inputs. The gate equals |f| >= " + ", ".join(f"{kk}: {v['abs_f']:.4f}" for kk, v in gthr.items()) +
            ". Same fit and paired bootstrap as q4. Alternative: the gate plus Amendment C (t0 > 0 and final sign) "
            f"changes the onset in {san['n_iii_C_differs_from_iii']} of {san['n_objects']} runs; the gate fires at "
            f"the first checkpoint in {san['n_iii_at_first_checkpoint']} runs."),
        literal_result="Bare-threshold Fourier onset slopes: " + bare + f". Censored: {tot_cens['v_fourier_bare']}.",
        literal_verdict="descriptive",
        amended_result="Gated Fourier onset slopes: " + gated + f". Censored: {tot_cens['iii_fourier_gated']}. "
                       "Paired slope differences RI (ii) minus gated Fourier (iii): " + dif + ". Per-run checkpoint "
                       "offset (ii) - (iii) (negative = RI earlier): " + off_s + ".",
        amended_verdict="descriptive",
        deciding="amended",
        key_numbers="\n".join(
            [f"gated Fourier slope k={k} lr={lr}: {_sl(F((k, lr), 'iii_fourier_gated'))}" for k, lr in K]
            + [f"(iii) median (IQR) k={k} lr={lr}: " + _cells(R[(k, lr)], "iii_fourier_gated") for k, lr in K]
            + [f"(ii)-(iii) slope diff k={k} lr={lr}: {R[(k, lr)]['paired_slope_differences']['ii_amended-iii_fourier_gated']['diff']:.3f} "
               f"CI {[round(c, 3) for c in R[(k, lr)]['paired_slope_differences']['ii_amended-iii_fourier_gated']['ci95']]}" for k, lr in K]
            + [f"runs with (ii) and (iii) on the same checkpoint: {sum(v.get(0, v.get('0', 0)) for v in [san['ii_minus_iii_checkpoints']])} "
               f"of {san['n_objects']} (k=2: {off['2'].get(0, off['2'].get('0', 0))} of 100, k=3: "
               f"{off['3'].get(0, off['3'].get('0', 0))} of 120, k=4: {off['4'].get(0, off['4'].get('0', 0))} of 68)",
               "gated minus bare Fourier onset in checkpoints by k: " + "; ".join(
                   f"k={k}: {san['iii_minus_v_by_k'][str(k)]}" for k in KS)]),
        interpretation=(
            "For k=3 and k=4 the RI and gated-Fourier onsets have identical medians in every (lr, n) cell, so the "
            "exponents are identical. For k=2 the RI exponent is larger by "
            f"{diff_iii[(2, 0.03)]:.2f} (lr 0.03) and {diff_iii[(2, 0.1)]:.2f} (lr 0.1); the paired CI excludes 0 at "
            f"lr 0.1 ({R[(2, 0.1)]['paired_slope_differences']['ii_amended-iii_fourier_gated']['ci95'][0]:.3f} to "
            f"{R[(2, 0.1)]['paired_slope_differences']['ii_amended-iii_fourier_gated']['ci95'][1]:.3f}). The cause is "
            f"that the RI event fires {-max(o for o in off_k2 if o < 0)}-{-min(off_k2)} checkpoints before the gate in "
            f"{rbg['n_runs_k2']} of 100 k=2 runs ({rbg['n_runs_k2_n_le_32']} of them at n <= 32; by n: "
            f"{rbg['n_runs_k2_by_n']}): the gate uses m * p, a conservative stand-in for the BH q-value, and the stored "
            "q is smaller when other subsets are also significant. At the RI onset checkpoint of the "
            f"{rbg['n_runs']} runs (all k) where RI fires first, the median stored q is {rbg['median_q_at_ri_onset']:.4f} "
            f"against a median stand-in of {rbg['median_standin_at_ri_onset']:.4f}; the stand-in is above alpha in "
            f"{rbg['n_standin_above_alpha_at_ri_onset']} of them. Over all runs, q < stand-in at the RI onset in "
            f"{rbg['n_q_below_standin_at_ri_onset_all_runs']} of {san['n_objects']}. With a gradually growing "
            "coefficient, a one-checkpoint shift at small n moves the slope. So RI and Fourier carry the same information here, as expected with uniform probes. The "
            "remaining k=2 difference comes from the gate's definition, not from the instrument. The bare "
            "Fourier onset (registered check-2 threshold) precedes the gated one by 0-5 checkpoints for k=2, as "
            "Amendment A anticipated, and by 0-1 for k=3 and k=4 (distribution in key_numbers)."),
        caveats="The gate approximates the BH q-value of the planted subset by m * p (Bonferroni). The stored q-values "
                "come from the full scan, so the RI event can pass before the stand-in does. Exponents of "
                "threshold crossings depend on the threshold level (q4_fit_strength).")) 

    # ---------------------------------------------------------------- q4 photometric reference
    t90 = "; ".join(f"k={k} lr={lr}: {_sl(F((k, lr), 'iv_t_test_acc90'))}" for k, lr in K)
    v0s = "; ".join(f"{key}: medians {[round(x) for x in v['median_t_test_acc90']]} slope {v['slope']:.2f} "
                    f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] ({sum(v['runs'])} runs)" for key, v in v0.items()
                    if v["lr"] == 0.1)
    v1s = "; ".join(f"k={k} lr=0.1: medians {[round(x) for x in R[(k, 0.1)]['measures']['iv_t_test_acc90']['medians'][:3]]} "
                    f"slope {_sl(R[(k, 0.1)]['measures']['iv_t_test_acc90']['fit']['subrange']['16-32'])}" for k in (2, 3))
    items.append(dict(
        id="q4_photometric",
        question="Q4 photometric reference: how does t_test_acc90 (first light-curve step with test accuracy >= 0.9) "
                 "scale with n, and does it reproduce the v0 hint?",
        definition_used=(
            "t_test_acc90 from objects.parquet (equal to the recomputation from the light curves in every run); never "
            f"reached = censored at the budget ({tot_cens['iv_t_test_acc90']} runs). Same fit and bootstrap as q4. v0 "
            "baseline recomputed from DR0 with scripts/analyze_dr0.py's rule: parity-wide, online data, learned runs "
            "(final test acc >= 0.9; at these settings all were learned), median t_test_acc90 at n = 16, 24, 32, "
            "pooled over widths (the published hint) and width 512 only (the v1 width), same bootstrap."),
        literal_result="t_test_acc90 slopes n=16-64 (k=4: 16-32): " + t90 + ".",
        literal_verdict="descriptive",
        key_numbers="\n".join(
            [f"t_test_acc90 slope k={k} lr={lr}: {_sl(F((k, lr), 'iv_t_test_acc90'))}; medians "
             f"{[round(x) for x in R[(k, lr)]['measures']['iv_t_test_acc90']['medians']]}" for k, lr in K]
            + [f"(iv) median (IQR) k={k} lr={lr}: " + _cells(R[(k, lr)], "iv_t_test_acc90") for k, lr in K]
            + ["v0 (DR0) lr=0.1: " + v0s, "v1 lr=0.1 n=16-32: " + v1s]
            + [f"slope diff (ii) - (iv) k={k} lr={lr}: {R[(k, lr)]['paired_slope_differences']['ii_amended-iv_t_test_acc90']['diff']:.3f} "
               f"CI {[round(c, 3) for c in R[(k, lr)]['paired_slope_differences']['ii_amended-iv_t_test_acc90']['ci95']]}" for k, lr in K]),
        interpretation=(
            "v1 reproduces v0 on the same configuration: at width 512, lr 0.1, n = 16-32 the t_test_acc90 medians "
            "and exponents agree (" + v1s + "; v0 width 512 in key_numbers). The v0 hint was an exponent of "
            "steps to 90 % test accuracy pooled over widths; recomputed from DR0 (lr 0.1) it is "
            f"{v0['k2_lr0.1_pooled_widths']['slope']:.2f} for k=2 and {v0['k3_lr0.1_pooled_widths']['slope']:.2f} for "
            "k=3. Over the wider v1 range, t_test_acc90 grows as "
            f"n^{rng_(sl_of('iv_t_test_acc90', (2,)))} for k=2 (concave: local slope "
            f"{rng_([F((2, lr), 'iv_t_test_acc90')['subrange']['32-64']['slope'] for lr in LRS])} from n=32 to 64), "
            f"n^{rng_(sl_of('iv_t_test_acc90', (3,)))} for k=3 and n^{rng_(sl_of('iv_t_test_acc90', (4,)))} for k=4. "
            "For k=3 and k=4 every t_test_acc90 CI excludes the n^(k-1) reference: "
            + "; ".join(f"k={k} lr={lr} {_sl(F((k, lr), 'iv_t_test_acc90'))}" for k in (3, 4) for lr in LRS) + ". "
            "The RI onset exponent is larger than the t_test_acc90 exponent in every (k, lr); the paired CIs "
            "exclude 0 except: " + (", ".join(
                f"k={k} lr={lr}" for k, lr in K
                if R[(k, lr)]['paired_slope_differences']['ii_amended-iv_t_test_acc90']['ci95'][0] <= 0) or "none")
            + ". So the RI lead over accuracy shrinks in relative terms as n grows (q4_lead)."),
        caveats=(f"t_test_acc90 lives on the light-curve grid ("
                 + ", ".join(f"{r_}: {g_['lc_points']} points" for r_, g_ in san["grids"].items())
                 + "), which is finer than the 60-point RI grid and contains it; a run's t_test_acc90 can therefore "
                 "fall between two RI checkpoints. The v0 baseline used a "
                 f"{v0['k2_lr0.1_pooled_widths']['train_steps']} step budget and its own evaluation grid, so the v0 "
                 "comparison is approximate."))) 

    # ---------------------------------------------------------------- q4 lead
    pooled = {kl: R[kl]["lead_vs_t_test_acc90_pooled_over_n"] for kl in K}
    ls = san["lead_steps_sign_by_k"]
    items.append(dict(
        id="q4_lead",
        question="Q4: how does the RI onset relate to t_test_acc90 (lead in steps, as a ratio, in checkpoints)?",
        definition_used=(
            "Per run: lead = t_test_acc90 - amended onset (ii) in steps; ratio = onset (ii) / t_test_acc90; lead in "
            "RI checkpoints = index of the first RI checkpoint with test accuracy >= 0.9 (the RI grid is a subset of "
            "the light-curve grid, so test accuracy is known there) minus the index of the onset checkpoint. "
            "Medians and IQR per (k, lr, n) and pooled over n."),
        literal_result="; ".join(
            f"k={k} lr={lr}: median lead {pooled[(k, lr)]['lead_steps_median']:g} steps, median ratio "
            f"{pooled[(k, lr)]['ratio_median']:.2f} (IQR {pooled[(k, lr)]['ratio_q25']:.2f}-{pooled[(k, lr)]['ratio_q75']:.2f}), "
            f"median {pooled[(k, lr)]['lead_ri_checkpoints_median']:g} RI checkpoints; same RI checkpoint in "
            f"{pooled[(k, lr)]['n_same_ri_checkpoint']} of {pooled[(k, lr)]['n_runs']}, RI after accuracy on the RI grid "
            f"in {pooled[(k, lr)]['n_ri_after_on_ri_grid']}" for k, lr in K),
        literal_verdict="descriptive",
        key_numbers="\n".join(
            [f"lead sign in steps by k: " + "; ".join(f"k={k}: {v['pos']} ahead, {v['zero']} same step, {v['neg']} after"
                                                     for k, v in ls.items())]
            + [f"lead in RI checkpoints by k: " + "; ".join(f"k={k}: {san['lead_ri_checkpoints_by_k'][str(k)]}" for k in KS)]
            + [f"ratio onset/t90 per n k={k} lr={lr}: " + ", ".join(
                f"n={n} {v['ratio_median']:.2f}" for n, v in R[(k, lr)]['lead_vs_t_test_acc90'].items()) for k, lr in K]
            + [f"median lead steps per n k={k} lr={lr}: " + ", ".join(
                f"n={n} {v['lead_steps_median']:g}" for n, v in R[(k, lr)]['lead_vs_t_test_acc90'].items()) for k, lr in K]
            + ["test accuracy at the onset checkpoint by k: " + "; ".join(
                f"k={k}: median {v['median']:.3f}, below 0.9 in {v['n_below_0p9']}" for k, v in san['test_acc_at_onset_by_k'].items())]),
        interpretation=(
            "The RI onset comes after the 90 % accuracy crossing on the common RI grid in "
            f"{int((d.lead_ri_checkpoints < 0).sum())} of {len(d)} runs. For k=2 it leads clearly: "
            f"{sum(v for c, v in lead_ck[2].items() if c > 0)} of {sum(lead_ck[2].values())} runs, by "
            f"{min(lead_ck[2])}-{max(lead_ck[2])} RI checkpoints, with median test accuracy "
            f"{san['test_acc_at_onset_by_k']['2']['median']:.2f} at the onset. The ratio onset/t_test_acc90 rises "
            f"with n (lr 0.1: {ratio_n[(2, 0.1)][0]:.2f} at n=16 to {ratio_n[(2, 0.1)][-1]:.2f} at n=64; lr 0.03: "
            f"{ratio_n[(2, 0.03)][0]:.2f} to {ratio_n[(2, 0.03)][-1]:.2f}), because the onset exponent exceeds the "
            "accuracy exponent. For k=3 and k=4 the coefficient jumps: from |f| >= 0.0808 to |f| >= 0.5 "
            "takes a median of " + ", ".join(
                f"k={k}: {san['rise_ckpts_0p0808_to_0p5_by_k'][str(k)]['median']:g}" for k in KS)
            + " RI checkpoints. The onset is then typically one RI checkpoint ahead (pooled median ratio "
            f"{rng_([pooled[(k, lr)]['ratio_median'] for k in (3, 4) for lr in LRS])}, about one grid step) or on the "
            f"same checkpoint (k=3: {lead_ck[3].get(0, 0)} of {sum(lead_ck[3].values())}; k=4: "
            f"{lead_ck[4].get(0, 0)} of {sum(lead_ck[4].values())}). So the lead for "
            "k >= 3 is at the grid resolution. As noted in v0, this is a low threshold on the same Walsh "
            "coefficient being crossed before a high one, not evidence that RI sees more than accuracy."),
        caveats=f"In steps, {ls['3']['neg']} k=3 and {ls['4']['neg']} k=4 runs show a negative lead, because t_test_acc90 falls on a light-curve step between "
                "two RI checkpoints; on the RI grid these are the same checkpoint. Ratios for k >= 3 take few "
                "discrete values set by the grid spacing.")) 

    # ---------------------------------------------------------------- q4 fit strength
    sub_lines = []
    for k, lr in K:
        sub = F((k, lr), "ii_amended")["subrange"]
        names = [x for x in sub if x != "upper_minus_lower"]
        j = R[(k, lr)]["bracket_jittered_bootstrap"]["ii_amended"]
        sub_lines.append(
            f"k={k} lr={lr} (ii): {names[0]} {_sl(sub[names[0]])}, {names[1]} {_sl(sub[names[1]])}, difference "
            f"{_sl(sub['upper_minus_lower'])}; max |residual| {F((k, lr), 'ii_amended')['max_abs_residual_log']:.3f} ln units; "
            f"bracket-jittered CI full range [{j['ci95'][0]:.2f}, {j['ci95'][1]:.2f}], difference CI "
            f"[{j['subrange']['upper_minus_lower']['ci95'][0]:.2f}, {j['subrange']['upper_minus_lower']['ci95'][1]:.2f}]")
    sweep = []
    for k, lr in K:
        sweep.append(f"k={k} lr={lr}: " + ", ".join(
            f"|f|>={c:g}: {F((k, lr), f'fixed_f_{c:g}')['slope']:.2f}" for c in FIXED_F)
            + f"; fixed gate at min m {F((k, lr), 'fixed_gate_min_m')['slope']:.2f}, at max m "
              f"{F((k, lr), 'fixed_gate_max_m')['slope']:.2f}")
    items.append(dict(
        id="q4_fit_strength",
        question="Q4: how strong is the evidence for a power law, and how robust is the exponent?",
        definition_used=(
            "(1) Residuals of the ln-ln OLS fit. (2) Sub-range slopes from the same bootstrap: n = 16-32 vs 32-64 for "
            "k=2,3 and 16-24 vs 24-32 for k=4, with a paired CI of their difference. (3) Bracket-jittered bootstrap: "
            "each resampled onset is drawn log-uniformly within its bracket (step_lo, step], so the CI includes the "
            "checkpoint quantisation; same for t_test_acc90 with its light-curve bracket. (4) Onset at step_lo or the "
            "geometric bracket midpoint. (5) Threshold sweep (supplementary, not a criterion): the first of two "
            "consecutive checkpoints t0 > 0 with sign-corrected f >= c for fixed c, and the Amendment-A gate "
            "threshold held fixed at the region's smallest / largest m (this removes the gate's n-dependence)."),
        literal_result="; ".join(sub_lines),
        literal_verdict="descriptive",
        key_numbers="\n".join(sub_lines + ["threshold sweep slopes: " + s_ for s_ in sweep]
                              + [f"step_lo / geomid slopes (ii) k={k} lr={lr}: {F((k, lr), 'ii_amended_step_lo')['slope']:.2f} / "
                                 f"{F((k, lr), 'ii_amended_geomid')['slope']:.2f}" for k, lr in K]
                              + [f"per-run OLS slope (ii) k={k} lr={lr}: {F((k, lr), 'ii_amended')['per_run_ols_slope']:.2f}" for k, lr in K]),
        interpretation=(
            "The five-point fits have R2 >= "
            f"{min(F(kl, 'ii_amended')['r2'] for kl in K if kl[0] < 4):.3f}, but residuals reach "
            f"{max(F(kl, 'ii_amended')['max_abs_residual_log'] for kl in K):.3f} in ln units, about one checkpoint "
            "gap. For k=2 a single power law does not hold: the local slope falls from "
            f"{rng_([sub2[(2, lr)]['16-32']['slope'] for lr in LRS])} (n=16-32) to "
            f"{rng_([sub2[(2, lr)]['32-64']['slope'] for lr in LRS])} (n=32-64), with difference CIs excluding 0 "
            "at both learning rates, also under bracket jitter. For k=3 the sub-range slopes are compatible "
            "(difference CIs include 0), but each sub-range CI is "
            f"{rng_([sub2[(3, lr)][r]['ci95'][1] - sub2[(3, lr)][r]['ci95'][0] for lr in LRS for r in ('16-32', '32-64')])} "
            "wide. The k=4 three-point fit (R2 = 1.000 with 1 degree of freedom) is no evidence of a power law: its "
            "sub-range CIs span "
            f"{min(sub2[(4, lr)][r]['ci95'][0] for lr in LRS for r in ('16-24', '24-32')):.2f}-"
            f"{max(sub2[(4, lr)][r]['ci95'][1] for lr in LRS for r in ('16-24', '24-32')):.2f}. The exponent also "
            "depends on which threshold defines 'onset'. For k=2 it falls from "
            f"{rng_(sw(2, 0.0447))} at |f| >= 0.0447 to {rng_(sw(2, 0.9))} at |f| >= 0.9; for k=3 from "
            f"{rng_(sw(3, 0.0447))} to {rng_(sw(3, 0.9))}; for k=4 from {rng_(sw(4, 0.0447))} to "
            f"{rng_(sw(4, 0.9))}. Holding the gate threshold fixed in n gives k=2 exponents "
            f"{rng_(gate_fixed[(2, 0.03)])} (lr 0.03) and {rng_(gate_fixed[(2, 0.1)])} (lr 0.1), "
            f"{rng_([gate_drop[(2, lr)] for lr in LRS])} below the RI onset exponent. For k=3 and k=4 the "
            f"fixed-gate exponents differ from the RI exponent by at most {gate_absdiff_k34:.2f}. The robust "
            "statements are three. Low-threshold onset exponents (RI, gated and bare Fourier, fixed |f| 0.0447 and "
            f"0.0808, fixed gates) lie between {min(low_all):.2f} and {max(low_all):.2f} across all k and lr, rising "
            "only weakly with k. No measure approaches 3 for k=4. The measures whose CI contains k-1 are listed in "
            "q4. A three- or five-point fit over a factor 2-4 in n cannot distinguish a power law from a "
            "curved relation."),
        caveats="The sub-range and sweep analyses are added robustness checks, not pre-registered criteria. "
                "Bootstrap CIs of grid-valued medians are lumpy; the jittered CIs are the more honest ones."))
    return items


def main() -> None:
    d, checks = load_runs()
    eps = float(d.eps_bare.iloc[0])
    res = analyse(d)
    v0 = v0_baseline()
    # v1 restricted to n = 16-32, lr 0.1, for the v0 comparison (t_test_acc90 and the RI onsets)
    v1_1632 = {f"k{k}_lr{lr}": {m: res[f"k{k}_lr{lr}"]["measures"][m]["fit"]["subrange"].get(
        "16-32", {"slope": res[f"k{k}_lr{lr}"]["measures"][m]["fit"]["slope"],
                  "ci95": res[f"k{k}_lr{lr}"]["measures"][m]["fit"]["ci95"]}) for m in MEASURES}
        for k in KS for lr in LRS}
    # censoring summary
    cens = {m: {f"k{k}_lr{lr}": int(d[(d.k == k) & (d.lr == lr)][CENS_COL[m]].sum()) for k in KS for lr in LRS}
            for m in ("i_registered", "ii_amended", "iii_fourier_gated", "iv_t_test_acc90", "v_fourier_bare",
                      "iii_fourier_gated_C", "fixed_gate_min_m", "fixed_gate_max_m")
            + tuple(f"fixed_f_{c:g}" for c in FIXED_F)}
    sanity = dict(
        checks,
        n_runs_by_k_lr={f"k{k}_lr{lr}": int(((d.k == k) & (d.lr == lr)).sum()) for k in KS for lr in LRS},
        gt_event_kinds=d.gt_event_kinds.value_counts().to_dict(),
        n_registered_left_censored=int(d.i_left_censored.sum()),
        n_registered_wrong_sign_at_onset=int((np.sign(d.f_at_ii.astype(float)) != np.sign(d.f_final)).sum()),
        n_ii_differs_from_i=int((d.ii_amended != d.i_registered).sum()),
        n_ii_variants_differ=dict(
            both_confirming=int((d.ii_amended_C_both_confirming != d.ii_amended).sum()),
            activation_only=int((d.ii_amended_C_activation_only != d.ii_amended).sum()),
            signaware_recompute=int((d.ii_amended_C_signaware_recompute != d.ii_amended).sum())),
        f_final_range=[float(d.f_final.min()), float(d.f_final.max())],
        n_f_final_negative=int((d.f_final < 0).sum()),
        abs_f_first_checkpoint_max=float(d.max_abs_f_init.max()),
        ri_first_checkpoint_max=float(d.ri_first_checkpoint.max()),
        n_q_first_checkpoint_le_alpha=int((d.q_first_checkpoint <= 0.01).sum()),
        n_iii_at_first_checkpoint=int(d.iii_at_first_checkpoint.sum()),
        n_iii_C_differs_from_iii=int((d.iii_fourier_gated_C != d.iii_fourier_gated).sum()),
        n_v_at_first_checkpoint=int(d.v_at_first_checkpoint.sum()),
        eps_bare=eps,
        abs_f_at_onset_by_k={str(k): dict(median=float(d[d.k == k].abs_f_at_ii.median()),
                                          q25=float(d[d.k == k].abs_f_at_ii.quantile(.25)),
                                          q75=float(d[d.k == k].abs_f_at_ii.quantile(.75)),
                                          max=float(d[d.k == k].abs_f_at_ii.max())) for k in KS},
        rise_ckpts_0p0808_to_0p5_by_k={str(k): dict(median=float(d[d.k == k].rise_ckpts_0p0808_to_0p5.median()),
                                                    min=int(d[d.k == k].rise_ckpts_0p0808_to_0p5.min()),
                                                    max=int(d[d.k == k].rise_ckpts_0p0808_to_0p5.max()))
                                       for k in KS},
        ri_before_gate=dict(
            n_runs=int((d.ii_minus_iii_checkpoints < 0).sum()),
            n_runs_k2=int(((d.ii_minus_iii_checkpoints < 0) & (d.k == 2)).sum()),
            n_runs_k2_n_le_32=int(((d.ii_minus_iii_checkpoints < 0) & (d.k == 2) & (d.n <= 32)).sum()),
            n_runs_k2_by_n=d[(d.ii_minus_iii_checkpoints < 0) & (d.k == 2)].n.value_counts().sort_index().to_dict(),
            median_q_at_ri_onset=float(d[d.ii_minus_iii_checkpoints < 0].q_at_ii.median()),
            median_standin_at_ri_onset=float(d[d.ii_minus_iii_checkpoints < 0].standin_at_ii.median()),
            n_standin_above_alpha_at_ri_onset=int((d[d.ii_minus_iii_checkpoints < 0].standin_at_ii > 0.01).sum()),
            n_q_below_standin_at_ri_onset_all_runs=int((d.q_at_ii < d.standin_at_ii).sum())),
        gate_abs_f_threshold_by_cell={f"k{k}_n{n}": dict(n_candidates=int(m), abs_f=gate_abs_f(int(m)))
                                      for (k, n), m in d.groupby(["k", "n"]).n_candidates.first().items()},
        final_test_acc_min=float(d.final_test_acc.min()),
        ii_minus_iii_checkpoints=d.ii_minus_iii_checkpoints.value_counts().sort_index().to_dict(),
        ii_minus_iii_by_k={str(k): d[d.k == k].ii_minus_iii_checkpoints.value_counts().sort_index().to_dict()
                           for k in KS},
        iii_minus_v_by_k={str(k): d[d.k == k].iii_minus_v_checkpoints.value_counts().sort_index().to_dict()
                          for k in KS},
        lead_ri_checkpoints_by_k={str(k): d[d.k == k].lead_ri_checkpoints.value_counts().sort_index().to_dict()
                                  for k in KS},
        lead_steps_sign_by_k={str(k): dict(pos=int((d[d.k == k].lead_steps > 0).sum()),
                                           zero=int((d[d.k == k].lead_steps == 0).sum()),
                                           neg=int((d[d.k == k].lead_steps < 0).sum())) for k in KS},
        test_acc_at_onset_by_k={str(k): dict(median=float(d[d.k == k].test_acc_at_ii.median()),
                                             max=float(d[d.k == k].test_acc_at_ii.max()),
                                             n_below_0p9=int((d[d.k == k].test_acc_at_ii < 0.9).sum()))
                                for k in KS},
        ratio_onset_over_t90_by_k={str(k): dict(median=float(d[d.k == k].ratio_onset_over_t90.median()),
                                                q25=float(d[d.k == k].ratio_onset_over_t90.quantile(.25)),
                                                q75=float(d[d.k == k].ratio_onset_over_t90.quantile(.75)))
                                   for k in KS},
    )
    chart = {}
    for k in KS:
        for lr in LRS:
            r = res[f"k{k}_lr{lr}"]
            mm = r["measures"]
            cell = lambda m, key: [mm[m]["cells"][str(n)][key] for n in r["n"]]  # noqa: E731
            chart[f"k{k}_lr{lr}"] = dict(
                k=k, lr=lr, n=r["n"], onset_ii_median=cell("ii_amended", "median"),
                onset_ii_q25=cell("ii_amended", "q25"), onset_ii_q75=cell("ii_amended", "q75"),
                t_test_acc90_median=cell("iv_t_test_acc90", "median"),
                t_test_acc90_q25=cell("iv_t_test_acc90", "q25"), t_test_acc90_q75=cell("iv_t_test_acc90", "q75"),
                fourier_gated_median=cell("iii_fourier_gated", "median"),
                registered_median=cell("i_registered", "median"),
                slope_ii=mm["ii_amended"]["fit"]["slope"], slope_ii_ci95=mm["ii_amended"]["fit"]["ci95"],
                slope_t90=mm["iv_t_test_acc90"]["fit"]["slope"], slope_t90_ci95=mm["iv_t_test_acc90"]["fit"]["ci95"],
                reference_exponent=k - 1, budget=int(d[d.k == k].budget.iloc[0]))
    per_run_cols = ["standin_at_ii", "abs_f_at_ii", "rise_ckpts_0p0808_to_0p5", "object_id", "region", "k", "lr", "n", "seed", "source", "label", "ri_run_id", "gt_subset",
                    "n_candidates", "gt_event_kinds", "i_registered", "i_step_lo", "i_left_censored",
                    "ii_amended", "ii_event_kind", "ii_amended_step_lo", "iii_fourier_gated", "iii_fourier_gated_C",
                    "v_fourier_bare", "iv_t_test_acc90", "iv_step_lo", "t_test_acc90_ri_grid", "idx_ii", "idx_iii", "idx_v",
                    "idx_t90_ri_grid", "f_at_ii", "f_final", "f_first_checkpoint", "ri_at_ii", "q_at_ii",
                    "ri_first_checkpoint", "q_first_checkpoint", "test_acc_at_ii", "lead_steps",
                    "ratio_onset_over_t90", "lead_ri_checkpoints", "ii_minus_iii_checkpoints", "final_test_acc",
                    "budget"] + [f"fixed_f_{c:g}" for c in FIXED_F] + ["fixed_gate_min_m", "fixed_gate_max_m"]
    items = build_items(res, sanity, cens, v0, d)
    doc = dict(
        key="q4_parity_scaling",
        items=items,
        description="Sky v1 pre-registered question 4 (sparse parity: onset scaling with n per order k), with "
                    "Amendment 1 items A and C. Generated by scripts/analysis_v1/parity_scaling.py.",
        data=dict(release="DR1", objects=str(REL / "objects.parquet"), extracted=str(EXTRACTED),
                  dr0_objects=str(REL0 / "objects.parquet")),
        settings=dict(bootstrap_resamples=B, bootstrap_seed=BOOT_SEED, unit="seeds resampled within each n, "
                      "paired across measures within (k, lr)", ci="percentile 95 %", median="numpy (mean of the two "
                      "middle values for even seed counts)", log="natural", onset_step="event step = first active "
                      "checkpoint (events.py step = steps[t0])", censoring="no onset within budget -> budget value"),
        sanity=sanity, censored_runs=cens, results=res, v0_baseline_from_DR0=v0, v1_n16_32_slopes=v1_1632,
        chart=chart, per_run=d[per_run_cols].to_dict(orient="records"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(jsonable(doc), indent=1))
    # ------------------------------------------------------------------ console summary
    print("sanity:", json.dumps(jsonable({k_: v for k_, v in sanity.items()}), indent=None)[:3000])
    print("censored:", cens)
    for key, r in res.items():
        print(f"\n== {key}  n={r['n']} seeds={r['seeds_per_n']}")
        for m in MEASURES + SUPP:
            fit = r["measures"][m]["fit"]
            meds = [round(v) for v in r["measures"][m]["medians"]]
            sub = {kk: (round(vv["slope"], 2), [round(c, 2) for c in vv["ci95"]]) for kk, vv in fit["subrange"].items()}
            print(f"  {m:34s} med={meds} slope={fit['slope']:.3f} CI=[{fit['ci95'][0]:.2f},{fit['ci95'][1]:.2f}] "
                  f"R2={fit['r2']:.3f} maxres={fit['max_abs_residual_log']:.3f} sub={sub} perrun={fit['per_run_ols_slope']:.2f}")
        for kk, vv in r["paired_slope_differences"].items():
            print(f"  diff {kk}: {vv['diff']:.3f} CI={[round(c, 3) for c in vv['ci95']]}")
        for n, l in r["lead_vs_t_test_acc90"].items():
            print(f"  lead n={n}: {l}")
        print("  lead pooled:", r["lead_vs_t_test_acc90_pooled_over_n"])
        print("  jitter:", json.dumps(jsonable(r["bracket_jittered_bootstrap"])))
    print("\nv0:", json.dumps(jsonable(v0), indent=None))
    print("\nITEMS_JSON_BEGIN")
    print(json.dumps(jsonable(items), indent=1))


if __name__ == "__main__":
    main()
