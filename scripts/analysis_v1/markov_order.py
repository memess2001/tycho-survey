"""Sky v1, pre-registered question 5 (Markov language: does the order set the onset?), incl. Amendment 1 C and D.

Pre-registration: docs/results/v1_preregistration.md (written before the run; Amendment 1 written after launch,
before v1 data were read). Data: releases/DR1 (tables) and the extracted RI curves / light curves.

Runs
  Region markov-v1, the eight one-coupling sets with planted order 2..5 at two strengths:
      beta = 1.0: "1:1.0" (order 2), "1,3:1.0" (3), "1,2,4:1.0" (4), "1,2,3,5:1.0" (5)
      beta = 0.5: "1:0.5", "1,3:0.5", "1,2,4:0.5", "1,2,3,5:0.5"
  Both train contexts (chain, iid); every seed of the current sky (here 10 per configuration, 160 runs).
  Planted subset = {x[t-l] : l in lags} + {xhat}: the curve's single ground-truth entry.

Onset of the planted subset, per run (all onsets are checkpoint steps of the 60-point in-situ RI grid)
  registered (literal)   the first "activation" event of the planted subset (DR1 ri_events, cross-checked against
                         detect_events(curve, alpha=0.01)). Its step is the FIRST of the two consecutive active
                         checkpoints that confirm it (events.py), so the onset lies in (step_lo, step]. An
                         activation at the first checkpoint (step 0, left-censored) counts with step 0, as written.
  amended (Amendment 1 C) the first (re)activation event of the planted subset that is not left-censored and at whose
                         checkpoint (the event's step) the planted Fourier coefficient has the sign of its final
                         value (last checkpoint, step 30,000). Reactivations are eligible: after a left-censored or
                         wrong-signed first episode the next episode is a "reactivation" in events.py.
      alternatives reported: (a) only the first "activation" event is eligible (if it fails C the run has no
                         amended onset and is censored); (b) the sign must hold at both confirming checkpoints.
  Runs without an onset are censored at the step budget (30,000) and kept (counted).
  Amendment C: order-2 onsets are at the scale of the initial offset and are reported as at the resolution floor
  (<= 8 steps); the actual values are given next to that, with any that exceed 8 steps.

Criterion (registered text, unchanged by Amendment D): at beta = 1.0, in both contexts, median onsets increase with
order 2 < 3 < 4 < 5 (strictly) AND the order-wise seed ranges [min, max] over seeds do not overlap (closed intervals,
every pair of orders; two orders sharing a checkpoint overlap). Evaluated as written (orders 2-5) and for orders 3-5
(Amendment C), with the registered and the amended onset. Per the amendment ("where this amendment changes a
definition, both results are reported and the amended one decides") the amended onset decides; the criterion
itself and its order set (2-5, as written) are unchanged.

Amendment D (resolution): per context at beta = 1.0 (and 0.5, descriptive) the checkpoint indices of the order-4 and
order-5 onsets, the grid around them with the step ratios, which checkpoints both orders share, and the overlap
depth d = max index(order k) - min index(order k+1) for consecutive orders: d < 0 no overlap; d = 0 the orders meet
only on a shared checkpoint (resolution limit); d = 1 a higher-order seed one checkpoint before a lower-order seed
(adjacent, within the grid's resolution in the sense of the task); d >= 2 genuine (a higher-order seed earlier by
more than one checkpoint). All seed pairs and same-seed pairs are also counted.

Exploratory (no criterion): the step at which the sign-corrected planted Fourier coefficient g = f * sign(f_final)
first reaches a level (eps = 0.0447, the coefficient at RI = min_effect; 25 % and 50 % of the population value),
interpolated linearly in g against log(step) between the two bracketing checkpoints. It hints whether orders 4 and 5
are ordered inside a shared grid interval; a dense replay is the registered follow-up.

Sanity checks: tycho imported from this repository; one grid for all runs; planted subset stored in every curve;
table events = recomputed events; population Walsh coefficient E_uniform[tanh(h(s)) prod_A s] by enumerating the
2^6 windows vs the final coefficient; final sign; final kl_true from the light curve.

Bootstrap: 2,000 resamples, fixed seed, seeds resampled within a configuration (beta, context, order); percentile
95 % intervals of the median onset and of the difference of medians between consecutive orders (descriptive).

Usage (cwd = repository root):
    PYTHONPATH=. python scripts/analysis_v1/markov_order.py [--release releases/DR1] [--extracted DIR] [--out JSON]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import tycho
from tycho.analysis.events import detect_events
from tycho.analysis.ricurve import RICurve
from tycho.ri.fourier import single_mode_ri
from tycho.tasks.markov_lang import parse_couplings

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACTED = REPO / "work" / "dr1"  # extracted DR1 archives (git-ignored); see README.md
DEFAULT_OUT = REPO / "docs/results/v1_analysis/markov_order.json"
REGION = "markov-v1"
SETS = {  # (beta, order) -> coupling set
    (1.0, 2): "1:1.0", (1.0, 3): "1,3:1.0", (1.0, 4): "1,2,4:1.0", (1.0, 5): "1,2,3,5:1.0",
    (0.5, 2): "1:0.5", (0.5, 3): "1,3:0.5", (0.5, 4): "1,2,4:0.5", (0.5, 5): "1,2,3,5:0.5",
}
BETAS = (1.0, 0.5)
ORDERS = (2, 3, 4, 5)
CONTEXTS = ("chain", "iid")
WINDOW = 6
ALPHA = 0.01
BUDGET = 30000
ORDER2_FLOOR = 8
BOOT_N = 2000
BOOT_SEED = 20261112
EPS_MIN_EFFECT = 0.0447  # |f| at which ln2 - h((1+|f|)/2) = 0.001 (checked below)


# ---------------------------------------------------------------------------------------------- helpers
def fnum(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return None if not math.isfinite(v) else v


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return fnum(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def lags_of_names(names) -> tuple[int, ...]:
    return tuple(sorted(int(n[len("x[t-"):-1]) for n in names if n.startswith("x[t-")))


def population_coefficient(couplings: str, lags: tuple[int, ...]) -> float:
    """E_uniform[tanh(h(s)) prod_{l in lags} s_l] over the 2^6 windows, h(s) = sum_L beta_L prod_{l in L} s_{t-l}."""
    cp = parse_couplings(couplings, WINDOW)
    tot = 0.0
    for s in itertools.product((-1.0, 1.0), repeat=WINDOW):  # s[l-1] = s_{t-l}
        h = sum(b * np.prod([s[l - 1] for l in L]) for L, b in cp.items())
        tot += math.tanh(h) * np.prod([s[l - 1] for l in lags])
    return tot / 2 ** WINDOW


def interp_crossing(steps: np.ndarray, g: np.ndarray, level: float) -> float | None:
    """First step where g reaches level, linear in g against log(step) between the bracketing checkpoints."""
    hit = np.flatnonzero(g >= level)
    if hit.size == 0:
        return None
    t = int(hit[0])
    if t == 0:
        return float(steps[0])
    s0, s1, g0, g1 = float(steps[t - 1]), float(steps[t]), float(g[t - 1]), float(g[t])
    w = (level - g0) / (g1 - g0) if g1 != g0 else 1.0
    if s0 <= 0:
        return s0 + w * (s1 - s0)
    return float(math.exp(math.log(s0) + w * (math.log(s1) - math.log(s0))))


def boot_median(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(x), size=(BOOT_N, len(x)))
    return np.median(x[idx], axis=1)


# ---------------------------------------------------------------------------------------------- per run
def analyse_run(row: pd.Series, tab_events: pd.DataFrame, extracted: Path) -> dict:
    c = RICurve.load(extracted / row.ricurve_file)
    steps = c.steps
    gt = c.ground_truth or []
    assert len(gt) == 1, f"{row.object_id}: expected one ground-truth entry, got {gt}"
    gt_names = tuple(gt[0]["subset"])
    gi = c.index_of(list(gt_names))
    stored = gi is not None
    assert stored, f"{row.object_id}: planted subset not stored"
    lags = lags_of_names(gt_names)
    assert len(gt_names) == row.order and "xhat" in gt_names
    assert set(lags) == set(parse_couplings(row.couplings, WINDOW).keys().__iter__().__next__())
    me = float(c.meta["min_effect"])
    assert abs(me - 1e-3) < 1e-12 and c.meta.get("kind") == "insitu"

    ri, q, f = c.ri[gi], c.q[gi], c.fourier[gi]
    f_final = float(f[-1])
    sgn_final = float(np.sign(f_final))

    # events: recomputed and from the table
    rec = [e for e in detect_events(c, alpha=ALPHA) if e.matches_ground_truth]
    rec = [e for e in rec if set(e.subset_names) == set(gt_names)]
    tab = tab_events[tab_events.subset_json.map(lambda s: set(json.loads(s)) == set(gt_names))].sort_values(
        ["step", "event_id"])
    rec_sig = [(e.kind, e.step, e.step_lo) for e in rec]
    tab_sig = [(k, int(s), int(lo)) for k, s, lo in zip(tab.kind, tab.step, tab.step_lo)]
    events_match = rec_sig == tab_sig
    assert bool(tab.matches_gt.all()) if len(tab) else True

    T = len(steps)
    step_list = [int(s) for s in steps]
    on_events = [e for e in rec if e.kind != "extinction"]

    def t_of(e):
        return step_list.index(e.step)

    # registered onset: first activation event (table; recomputed must agree)
    acts_tab = tab[tab.kind == "activation"]
    if len(acts_tab):
        reg_step = int(acts_tab.step.iloc[0])
        reg_lo = int(acts_tab.step_lo.iloc[0])
        reg_t = step_list.index(reg_step)
        reg_left = bool(reg_t == 0)
        assert reg_left == bool(np.isnan(acts_tab.ri_before.iloc[0]))
    else:
        reg_step = reg_lo = reg_t = None
        reg_left = False

    # amended onset (C): first (re)activation not left-censored with sign(f[t0]) == sign(final f)
    def amended(events, window_sign: bool):
        skipped = []
        for e in events:
            t0 = t_of(e)
            if e.left_censored:
                skipped.append({"kind": e.kind, "step": e.step, "reason": "left-censored", "f": float(f[t0])})
                continue
            ts = [t0, t0 + 1] if window_sign else [t0]
            ok = all(np.isfinite(f[t]) and np.sign(f[t]) == sgn_final and sgn_final != 0 for t in ts if t < T)
            if not ok:
                skipped.append({"kind": e.kind, "step": e.step, "reason": "wrong sign", "f": float(f[t0])})
                continue
            return e, t0, skipped
        return None, None, skipped

    e_am, t_am, skipped_am = amended(on_events, window_sign=False)
    e_alt_a, t_alt_a, _ = amended([e for e in on_events if e.kind == "activation"][:1], window_sign=False)
    e_alt_b, t_alt_b, _ = amended(on_events, window_sign=True)

    # Fourier onset with the bare threshold eps (sign-corrected, two consecutive checkpoints): descriptive
    g = f * sgn_final
    pair = (g[:-1] >= EPS_MIN_EFFECT) & (g[1:] >= EPS_MIN_EFFECT)
    t_four = int(np.flatnonzero(pair)[0]) if pair.any() else None

    pop = population_coefficient(row.couplings, lags)
    interp = {
        "eps_0.0447": interp_crossing(steps, g, EPS_MIN_EFFECT),
        "25pct_pop": interp_crossing(steps, g, 0.25 * pop),
        "50pct_pop": interp_crossing(steps, g, 0.50 * pop),
    }
    lc = np.load(extracted / row.lightcurve_file)
    kl = lc["x_kl_true"]
    kl_final = float(kl[np.isfinite(kl)][-1]) if np.isfinite(kl).any() else float("nan")

    def st(t):
        return None if t is None else step_list[t]

    return {
        "object_id": row.object_id, "ri_run_id": int(row.ri_run_id), "couplings": row.couplings,
        "beta": float(row.beta), "order": int(row.order), "context": row.context, "seed": int(row.seed),
        "source": row.source, "lightcurve_file": row.lightcurve_file, "ricurve_file": row.ricurve_file,
        "planted_subset": list(gt_names), "planted_stored": stored, "n_subsets": c.S, "n_candidates": c.meta.get(
            "n_candidates"), "grid": tuple(step_list),
        "events_planted": [{"kind": e.kind, "step": e.step, "step_lo": e.step_lo, "idx": t_of(e),
                            "left_censored": e.left_censored, "f_at_step": float(f[t_of(e)]),
                            "ri_after": e.ri_after, "peak_ri": e.peak_ri} for e in rec],
        "events_table_match_recomputed": events_match, "n_events_table": len(tab),
        # registered
        "reg_onset_step": reg_step, "reg_onset_idx": reg_t, "reg_onset_step_lo": reg_lo,
        "reg_left_censored": reg_left, "reg_censored": reg_step is None,
        "reg_onset_value": BUDGET if reg_step is None else reg_step,
        "reg_f_at_onset": None if reg_t is None else float(f[reg_t]),
        "reg_sign_matches_final": None if reg_t is None else bool(np.sign(f[reg_t]) == sgn_final),
        # amended C
        "am_onset_step": st(t_am), "am_onset_idx": t_am, "am_onset_step_lo": None if e_am is None else e_am.step_lo,
        "am_event_kind": None if e_am is None else e_am.kind, "am_censored": e_am is None,
        "am_onset_value": BUDGET if e_am is None else step_list[t_am],
        "am_skipped_events": skipped_am,
        "am_f_at_onset": None if t_am is None else float(f[t_am]),
        "am_ri_at_onset": None if t_am is None else float(ri[t_am]),
        "am_q_at_onset": None if t_am is None else float(q[t_am]),
        # which gate condition blocked the checkpoint before the onset; max q over the two confirming checkpoints
        "am_prev_blocked_by": None if (t_am is None or t_am == 0) else (
            "ri<min_effect" if ri[t_am - 1] < me else ("q>alpha" if q[t_am - 1] > ALPHA else "persistence")),
        "am_q_max_confirming": None if t_am is None else float(np.nanmax(q[t_am:t_am + 2])),
        "am_differs_from_reg": (t_am != reg_t),
        # sensitivity: onset read at the confirming (second active) checkpoint instead of the event's step
        "am_confirm_value": BUDGET if (t_am is None or t_am + 1 >= T) else step_list[t_am + 1],
        "am_confirm_idx": None if (t_am is None or t_am + 1 >= T) else t_am + 1,
        # alternatives of C
        "alt_activation_only_value": BUDGET if e_alt_a is None else step_list[t_alt_a],
        "alt_activation_only_idx": t_alt_a, "alt_activation_only_censored": e_alt_a is None,
        "alt_window_sign_value": BUDGET if e_alt_b is None else step_list[t_alt_b],
        "alt_window_sign_idx": t_alt_b, "alt_window_sign_censored": e_alt_b is None,
        # Fourier and sanity
        "fourier_eps_onset_step": st(t_four), "fourier_eps_onset_idx": t_four,
        "interp_crossing_step": interp,
        "f_initial": float(f[0]), "ri_initial": float(ri[0]), "q_initial": float(q[0]),
        "f_final": f_final, "ri_final": float(ri[-1]), "sign_final": sgn_final,
        "population_coefficient": pop, "population_ri": float(single_mode_ri(pop)),
        "final_over_population": f_final / pop, "kl_true_final": kl_final,
        "peak_ri": float(np.nanmax(ri)),
    }


# ---------------------------------------------------------------------------------------------- groups
def group_stats(d: pd.DataFrame, col: str, idx_col: str, cens_col: str, rng_seed: int) -> dict:
    d = d.sort_values("seed")
    v = d[col].to_numpy(float)
    rng = np.random.default_rng(rng_seed)
    bm = boot_median(v, rng)
    idx = d[idx_col]
    return {
        "n": int(len(d)), "seeds": d.seed.astype(int).tolist(), "onsets": [int(x) for x in v],
        "checkpoint_idx": [None if pd.isna(x) else int(x) for x in idx],
        "n_censored": int(d[cens_col].sum()),
        "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
        "idx_min": None if idx.isna().all() else int(idx.min()), "idx_max": None if idx.isna().all() else int(
            idx.max()),
        "median_boot_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
        "_boot": bm,
    }


def evaluate(groups: dict, orders: tuple[int, ...]) -> dict:
    """Registered criterion on one context: strictly increasing medians and pairwise non-overlapping ranges."""
    med = [groups[k]["median"] for k in orders]
    rngs = {k: (groups[k]["min"], groups[k]["max"]) for k in orders}
    medians_increasing = all(a < b for a, b in zip(med, med[1:]))
    overlaps = []
    for a, b in itertools.combinations(orders, 2):
        lo_a, hi_a = rngs[a]
        lo_b, hi_b = rngs[b]
        if max(lo_a, lo_b) <= min(hi_a, hi_b):
            overlaps.append([a, b])
    ordered = all(rngs[a][1] < rngs[b][0] for a, b in zip(orders, orders[1:]))
    return {"orders": list(orders), "medians": med, "ranges": {str(k): list(rngs[k]) for k in orders},
            "medians_strictly_increasing": medians_increasing, "overlapping_pairs": overlaps,
            "ranges_disjoint_and_ordered": ordered, "pass": bool(medians_increasing and not overlaps)}


def overlap_class(ga: dict, gb: dict, steps: list[int]) -> dict:
    """Overlap of consecutive orders a < b on checkpoint indices (censored runs: index = len(grid))."""
    ia = [len(steps) if x is None else x for x in ga["checkpoint_idx"]]
    ib = [len(steps) if x is None else x for x in gb["checkpoint_idx"]]
    depth = max(ia) - min(ib)
    if depth < 0:
        cls = "none"
    elif depth == 0:
        cls = "same checkpoint only (resolution limit)"
    elif depth == 1:
        cls = "adjacent: a higher-order seed one checkpoint before a lower-order seed (within one checkpoint)"
    else:
        cls = "genuine: a higher-order seed earlier by more than one checkpoint"
    shared = sorted(set(ia) & set(ib))
    pairs = [(x, y) for x in ia for y in ib]
    same_seed = {}
    for s, x in zip(ga["seeds"], ia):
        if s in gb["seeds"]:
            same_seed[s] = ib[gb["seeds"].index(s)] - x
    lo = max(0, min(min(ia), min(ib)) - 2)
    hi = min(len(steps) - 1, max(max(ia), max(ib)) + 2)
    return {
        "orders": [ga["order"], gb["order"]], "overlap_depth_idx": int(depth), "classification": cls,
        "shared_checkpoints": [{"idx": i, "step": steps[i] if i < len(steps) else None,
                                "n_lower": ia.count(i), "n_higher": ib.count(i)} for i in shared],
        "idx_counts_lower": {str(i): ia.count(i) for i in sorted(set(ia))},
        "idx_counts_higher": {str(i): ib.count(i) for i in sorted(set(ib))},
        "all_seed_pairs": {"n": len(pairs), "higher_later": sum(y > x for x, y in pairs),
                           "same_checkpoint": sum(y == x for x, y in pairs),
                           "higher_earlier_by_1": sum(y == x - 1 for x, y in pairs),
                           "higher_earlier_by_more_than_1": sum(y < x - 1 for x, y in pairs)},
        "same_seed_idx_difference_higher_minus_lower": {str(k): int(v) for k, v in sorted(same_seed.items())},
        "grid_around": [{"idx": i, "step": steps[i],
                         "ratio_to_previous": None if i == 0 or steps[i - 1] == 0 else steps[i] / steps[i - 1],
                         "n_lower": ia.count(i), "n_higher": ib.count(i)} for i in range(lo, hi + 1)],
    }


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases/DR1")
    ap.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    assert Path(tycho.__file__).resolve().is_relative_to(REPO), tycho.__file__
    assert abs(float(single_mode_ri(EPS_MIN_EFFECT)) - 1e-3) < 2e-6

    objs = pd.read_parquet(args.release / "objects.parquet")
    runs = pd.read_parquet(args.release / "ri_runs.parquet")
    evs = pd.read_parquet(args.release / "ri_events.parquet")
    m = objs[objs.region == REGION].copy()
    cfg = m.task_cfg_json.map(json.loads)
    m["couplings"] = cfg.map(lambda c: c.get("couplings"))
    m["context"] = cfg.map(lambda c: c.get("train_context"))
    inv = {v: k for k, v in SETS.items()}
    m = m[m.couplings.isin(inv)].copy()
    m["beta"] = m.couplings.map(lambda s: inv[s][0])
    m["order"] = m.couplings.map(lambda s: inv[s][1])
    assert (m.train_steps == BUDGET).all() and (m.steps_total == BUDGET).all()
    m = m.merge(runs[["object_id", "ri_run_id", "kind", "ricurve_file", "n_points", "min_effect", "alpha"]],
                on="object_id", how="left", validate="one_to_one")
    assert m.ri_run_id.notna().all() and (m.kind == "insitu").all() and (m.n_points == 60).all()

    rows = []
    for _, row in m.sort_values(["beta", "order", "context", "seed"], ascending=[False, True, True, True]).iterrows():
        rows.append(analyse_run(row, evs[evs.ri_run_id == row.ri_run_id], args.extracted))
    df = pd.DataFrame(rows)
    grids = set(df.grid)
    assert len(grids) == 1, "runs do not share one checkpoint grid"
    steps = list(next(iter(grids)))

    checks = {
        "tycho_file": str(Path(tycho.__file__).resolve()),
        "n_runs": int(len(df)), "n_objects_markov_v1": int((objs.region == REGION).sum()),
        "runs_per_configuration": {f"{r.couplings}|{r.context}": int(r.n) for r in
                                   df.groupby(["couplings", "context"]).size().rename("n").reset_index().itertuples()},
        "one_grid_for_all_runs": True, "grid_steps": steps, "n_checkpoints": len(steps),
        "planted_stored": f"{int(df.planted_stored.sum())} of {len(df)}",
        "table_events_equal_recomputed": f"{int(df.events_table_match_recomputed.sum())} of {len(df)}",
        "eps_at_ri_min_effect": EPS_MIN_EFFECT, "single_mode_ri(eps)": float(single_mode_ri(EPS_MIN_EFFECT)),
        "population_coefficient_by_beta": {str(b): sorted(set(np.round(df[df.beta == b].population_coefficient,
                                                                         6))) for b in BETAS},
        "tanh_beta": {str(b): math.tanh(b) for b in BETAS},
        "final_sign_positive": f"{int((df.sign_final > 0).sum())} of {len(df)}",
        "final_over_population_min_max": [float(df.final_over_population.min()),
                                          float(df.final_over_population.max())],
        "kl_true_final_max": float(df.kl_true_final.max()),
        "kl_true_final_lt_0.01": f"{int((df.kl_true_final < 0.01).sum())} of {len(df)}",
        "reg_censored": f"{int(df.reg_censored.sum())} of {len(df)}",
        "am_censored": f"{int(df.am_censored.sum())} of {len(df)}",
        "reg_left_censored": f"{int(df.reg_left_censored.sum())} of {len(df)}",
        "reg_onset_wrong_sign": f"{int((df.reg_sign_matches_final == False).sum())} of {len(df)}",  # noqa: E712
        "am_differs_from_reg": f"{int(df.am_differs_from_reg.sum())} of {len(df)}",
        "am_differs_from_reg_runs": [f"{r.couplings}|{r.context}|seed {r.seed}: {r.reg_onset_value} -> "
                                     f"{r.am_onset_value} ({r.am_event_kind})" for r in df.itertuples()
                                     if r.am_differs_from_reg],
        "alt_activation_only_censored": f"{int(df.alt_activation_only_censored.sum())} of {len(df)}",
        "alt_window_sign_differs_from_am": f"{int((df.alt_window_sign_value != df.am_onset_value).sum())} of "
                                           f"{len(df)}",
        # same seed number = same initial weights across beta and context (initial planted coefficient identical)
        "same_seed_initial_f_max_spread_within_order": float(
            df.groupby(["order", "seed"]).f_initial.agg(lambda s: s.max() - s.min()).max()),
    }
    # BH q-values scale linearly with the number of candidates m (q = min(1, q * m'/m)); the candidate count grows
    # with the planted order (max_order = order), so check that it cannot move the onsets of orders 3-5
    hi = df[df.order >= 3]
    m_by_order = {int(k): sorted(set(v)) for k, v in df.groupby("order").n_candidates}
    m5 = max(m_by_order[5])
    checks["n_candidates_by_order"] = m_by_order
    checks["orders3to5_prev_checkpoint_blocked_by"] = {str(k): int(v) for k, v in
                                                       hi.am_prev_blocked_by.value_counts().items()}
    checks["orders3to4_max_q_confirming_rescaled_to_order5_m"] = float(
        max(r.am_q_max_confirming * m5 / r.n_candidates for r in hi[hi.order < 5].itertuples()))
    checks["candidate_count_cannot_move_onsets"] = bool(
        (hi.am_prev_blocked_by == "ri<min_effect").all()
        and checks["orders3to4_max_q_confirming_rescaled_to_order5_m"] <= ALPHA)
    # descriptive: does the untrained network's planted coefficient set the seed order within an order?
    init_offset = {}
    for b in BETAS:
        for ctx in CONTEXTS:
            for k in ORDERS:
                d = df[(df.beta == b) & (df.context == ctx) & (df.order == k)]
                const = d.am_onset_value.nunique() < 2
                init_offset[f"beta={b}|{ctx}|order={k}"] = {
                    "spearman_f_initial_vs_amended_onset": None if const else float(
                        d.f_initial.corr(d.am_onset_value.astype(float), method="spearman")),
                    "n_f_initial_negative": int((d.f_initial < 0).sum()),
                    "median_onset_f_initial_pos": float(np.median(d[d.f_initial > 0].am_onset_value))
                    if (d.f_initial > 0).any() else None,
                    "median_onset_f_initial_neg": float(np.median(d[d.f_initial < 0].am_onset_value))
                    if (d.f_initial < 0).any() else None}

    # ------------------------------------------------------------------ groups and criterion
    measures = {
        "registered": ("reg_onset_value", "reg_onset_idx", "reg_censored"),
        "amended": ("am_onset_value", "am_onset_idx", "am_censored"),
        "amended_alt_activation_only": ("alt_activation_only_value", "alt_activation_only_idx",
                                        "alt_activation_only_censored"),
        "amended_alt_window_sign": ("alt_window_sign_value", "alt_window_sign_idx", "alt_window_sign_censored"),
        "amended_alt_confirm_checkpoint": ("am_confirm_value", "am_confirm_idx", "am_censored"),
    }
    G: dict = {}
    seed_ctr = 0
    for meas, (col, icol, ccol) in measures.items():
        for b in BETAS:
            for ctx in CONTEXTS:
                for k in ORDERS:
                    d = df[(df.beta == b) & (df.context == ctx) & (df.order == k)]
                    gs = group_stats(d, col, icol, ccol, BOOT_SEED + seed_ctr)
                    seed_ctr += 1
                    gs["order"] = k
                    gs["n_left_censored_first_activation"] = int(d.reg_left_censored.sum())
                    G[(meas, b, ctx, k)] = gs

    criterion = {}
    for meas in measures:
        for b in BETAS:
            for oset in ((2, 3, 4, 5), (3, 4, 5)):
                key = f"{meas}|beta={b}|orders={'-'.join(map(str, oset))}"
                per_ctx = {ctx: evaluate({k: G[(meas, b, ctx, k)] for k in ORDERS}, oset) for ctx in CONTEXTS}
                criterion[key] = {"per_context": per_ctx,
                                  "supported": bool(all(per_ctx[c]["pass"] for c in CONTEXTS)),
                                  "role": ("deciding" if (meas == "amended" and b == 1.0 and oset == (2, 3, 4, 5))
                                           else "literal (registered onset, as written)" if (
                                      meas == "registered" and b == 1.0 and oset == (2, 3, 4, 5))
                                           else "amendment C (orders 3-5)" if b == 1.0 and meas in (
                                      "registered", "amended") else "sensitivity" if b == 1.0 else
                                           "descriptive (beta 0.5)")}

    # differences of medians between consecutive orders (bootstrap, descriptive)
    med_diffs = {}
    for meas in ("registered", "amended"):
        for b in BETAS:
            for ctx in CONTEXTS:
                for a, c_ in zip(ORDERS, ORDERS[1:]):
                    ga, gb = G[(meas, b, ctx, a)], G[(meas, b, ctx, c_)]
                    diff = gb["_boot"] - ga["_boot"]
                    med_diffs[f"{meas}|beta={b}|{ctx}|{c_}-{a}"] = {
                        "median_difference": gb["median"] - ga["median"],
                        "boot_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                        "boot_frac_le_0": float(np.mean(diff <= 0))}

    # ------------------------------------------------------------------ Amendment D: resolution
    resolution = {}
    for meas in ("amended", "registered", "fourier_eps"):
        for b in BETAS:
            for ctx in CONTEXTS:
                for a, c_ in zip(ORDERS, ORDERS[1:]):
                    if meas == "fourier_eps":
                        def gfor(k):
                            d = df[(df.beta == b) & (df.context == ctx) & (df.order == k)].sort_values("seed")
                            return {"order": k, "seeds": d.seed.astype(int).tolist(),
                                    "checkpoint_idx": [None if pd.isna(x) else int(x) for x in
                                                       d.fourier_eps_onset_idx]}
                        ga, gb = gfor(a), gfor(c_)
                    else:
                        ga, gb = G[(meas, b, ctx, a)], G[(meas, b, ctx, c_)]
                    resolution[f"{meas}|beta={b}|{ctx}|{a}-{c_}"] = overlap_class(ga, gb, steps)

    # exploratory: interpolated crossings of orders 4 vs 5 (same seed)
    interp_cmp = {}
    for b in BETAS:
        for ctx in CONTEXTS:
            d4 = df[(df.beta == b) & (df.context == ctx) & (df.order == 4)].set_index("seed")
            d5 = df[(df.beta == b) & (df.context == ctx) & (df.order == 5)].set_index("seed")
            out = {}
            for lvl in ("eps_0.0447", "25pct_pop", "50pct_pop"):
                x4 = d4.interp_crossing_step.map(lambda z: z[lvl])
                x5 = d5.interp_crossing_step.map(lambda z: z[lvl])
                common = sorted(set(x4.index) & set(x5.index))
                ratio = [x5[s] / x4[s] for s in common]
                out[lvl] = {"order4_median": float(np.median(x4)), "order5_median": float(np.median(x5)),
                            "order4_range": [float(x4.min()), float(x4.max())],
                            "order5_range": [float(x5.min()), float(x5.max())],
                            "same_seed_ratio_5_over_4": {str(s): float(r) for s, r in zip(common, ratio)},
                            "n_seeds_order5_later": f"{sum(r > 1 for r in ratio)} of {len(ratio)}",
                            "median_same_seed_ratio": float(np.median(ratio))}
            interp_cmp[f"beta={b}|{ctx}"] = out

    # ------------------------------------------------------------------ order 2 at the floor
    o2 = df[df.order == 2]
    order2 = {}
    for b in BETAS:
        for ctx in CONTEXTS:
            d = o2[(o2.beta == b) & (o2.context == ctx)].sort_values("seed")
            order2[f"beta={b}|{ctx}"] = {
                "registered": d.reg_onset_value.astype(int).tolist(), "amended": d.am_onset_value.astype(int).tolist(),
                "seeds": d.seed.astype(int).tolist(),
                "n_registered_left_censored": int(d.reg_left_censored.sum()),
                "n_amended_le_floor": f"{int((d.am_onset_value <= ORDER2_FLOOR).sum())} of {len(d)}",
                "amended_above_floor": [{"seed": int(r.seed), "onset": int(r.am_onset_value),
                                         "event": r.am_event_kind, "f_initial": r.f_initial,
                                         "skipped": r.am_skipped_events} for r in d.itertuples()
                                        if r.am_onset_value > ORDER2_FLOOR],
                "n_initial_f_negative": int((d.f_initial < 0).sum())}

    # ------------------------------------------------------------------ chart
    chart = {"description": "Onset of the planted subset per seed, per train context, beta and planted order. "
                            "onset_amended = Amendment 1 C onset (step of the first not-left-censored (re)activation "
                            "with the final Fourier sign); onset_registered = first activation event. Censored runs "
                            f"(none expected) would carry {BUDGET} and censored=true. Order-2 onsets are at the "
                            f"resolution floor (<= {ORDER2_FLOOR} steps per Amendment C; actual values shown).",
             "x": "planted order", "y": "onset step (log scale suggested; step 0 = first checkpoint)",
             "grid_steps": steps, "data": {}}
    for ctx in CONTEXTS:
        chart["data"][ctx] = {}
        for b in BETAS:
            chart["data"][ctx][str(b)] = {}
            for k in ORDERS:
                d = df[(df.beta == b) & (df.context == ctx) & (df.order == k)].sort_values("seed")
                chart["data"][ctx][str(b)][str(k)] = {
                    "couplings": SETS[(b, k)], "seeds": d.seed.astype(int).tolist(),
                    "onset_amended": d.am_onset_value.astype(int).tolist(),
                    "checkpoint_idx_amended": [None if pd.isna(x) else int(x) for x in d.am_onset_idx],
                    "censored": d.am_censored.astype(bool).tolist(),
                    "onset_registered": d.reg_onset_value.astype(int).tolist(),
                    "median_amended": float(np.median(d.am_onset_value)),
                    "range_amended": [int(d.am_onset_value.min()), int(d.am_onset_value.max())]}

    # ------------------------------------------------------------------ items
    def rng_str(meas, b, ctx, k):
        g = G[(meas, b, ctx, k)]
        return (f"o{k}: med {g['median']:g} [{g['min']:g}, {g['max']:g}] idx {g['idx_min']}-{g['idx_max']} "
                f"(n={g['n']}, cens {g['n_censored']})")

    def crit_str(key):
        cr = criterion[key]
        parts = []
        for ctx in CONTEXTS:
            pc = cr["per_context"][ctx]
            parts.append(f"{ctx}: medians {pc['medians']} increasing={pc['medians_strictly_increasing']}, "
                         f"overlapping pairs {pc['overlapping_pairs']} -> {'pass' if pc['pass'] else 'fail'}")
        return "; ".join(parts) + f". Supported: {cr['supported']}"

    def verdict(key):
        return "supported" if criterion[key]["supported"] else "not_supported"

    k_lit = "registered|beta=1.0|orders=2-3-4-5"
    k_lit35 = "registered|beta=1.0|orders=3-4-5"
    k_am = "amended|beta=1.0|orders=2-3-4-5"
    k_am35 = "amended|beta=1.0|orders=3-4-5"
    k_alt_a = "amended_alt_activation_only|beta=1.0|orders=2-3-4-5"
    k_alt_b = "amended_alt_window_sign|beta=1.0|orders=2-3-4-5"
    k_alt_a35 = "amended_alt_activation_only|beta=1.0|orders=3-4-5"
    k_alt_b35 = "amended_alt_window_sign|beta=1.0|orders=3-4-5"

    k_alt_c = "amended_alt_confirm_checkpoint|beta=1.0|orders=2-3-4-5"
    k_alt_c35 = "amended_alt_confirm_checkpoint|beta=1.0|orders=3-4-5"

    out = {
        "analysis": "v1 pre-registered question 5 (Markov: does the order set the onset?) incl. Amendment 1 C, D",
        "script": "scripts/analysis_v1/markov_order.py",
        "preregistration": "docs/results/v1_preregistration.md",
        "release": str(args.release.relative_to(REPO) if args.release.is_relative_to(REPO) else args.release),
        "budget_steps": BUDGET, "alpha": ALPHA, "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED,
                                                              "unit": "seeds within a configuration"},
        "checks": checks,
        "groups": {f"{meas}|beta={b}|{ctx}|order={k}": {kk: vv for kk, vv in g.items() if kk != "_boot"}
                   for (meas, b, ctx, k), g in G.items()},
        "criterion": criterion, "median_differences": med_diffs, "resolution_D": resolution,
        "interpolated_crossings_order4_vs_5_exploratory": interp_cmp, "order2_floor": order2,
        "initial_offset_vs_onset": init_offset, "chart": chart,
        "runs": [{k: v for k, v in r.items() if k != "grid"} for r in rows],
    }
    ctx_ = dict(G=G, criterion=criterion, resolution=resolution, interp_cmp=interp_cmp, order2=order2, df=df,
                steps=steps, checks=checks, med_diffs=med_diffs, init_offset=init_offset,
                keys=dict(lit=k_lit, lit35=k_lit35, am=k_am, am35=k_am35, a=k_alt_a, a35=k_alt_a35, b=k_alt_b,
                          b35=k_alt_b35, c=k_alt_c, c35=k_alt_c35))
    out["items"] = build_items(**ctx_)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(out), indent=1, allow_nan=False))
    print(json.dumps(jsonable(out["items"]), indent=1, ensure_ascii=False))
    print(f"wrote {args.out}")


# ---------------------------------------------------------------------------------------------- items (text)
def build_items(G, criterion, resolution, interp_cmp, order2, df, steps, checks, med_diffs, init_offset,
                keys) -> list[dict]:
    """Structured result items; every number in them is taken from the computed tables."""

    def g(meas, b, ctx, k):
        return G[(meas, b, ctx, k)]

    def fmt(x):
        return f"{x:g}"

    def rng_str(meas, b, ctx, k):
        x = g(meas, b, ctx, k)
        ci = x["median_boot_ci95"]
        return (f"o{k}: median {fmt(x['median'])} (boot 95% {fmt(ci[0])}-{fmt(ci[1])}), range [{fmt(x['min'])}, "
                f"{fmt(x['max'])}] = checkpoints {x['idx_min']}-{x['idx_max']}, n={x['n']}, censored "
                f"{x['n_censored']}")

    def crit_str(key):
        cr = criterion[key]
        parts = []
        for ctx in CONTEXTS:
            pc = cr["per_context"][ctx]
            parts.append(f"{ctx}: medians {[fmt(v) for v in pc['medians']]} strictly increasing="
                         f"{pc['medians_strictly_increasing']}, ranges {pc['ranges']}, overlapping order pairs "
                         f"{pc['overlapping_pairs']} -> {'pass' if pc['pass'] else 'fail'}")
        return "; ".join(parts) + f" => {'SUPPORTED' if cr['supported'] else 'NOT SUPPORTED'}"

    def verdict(key):
        return "supported" if criterion[key]["supported"] else "not_supported"

    def short(key):
        cr = criterion[key]
        return ("supported" if cr["supported"] else "not supported") + " (" + ", ".join(
            f"{c}: overlaps {cr['per_context'][c]['overlapping_pairs']}" for c in CONTEXTS) + ")"

    def res(meas, b, ctx, a, c_):
        return resolution[f"{meas}|beta={b}|{ctx}|{a}-{c_}"]

    def onsets_by_seed(meas, b, ctx, k):
        x = g(meas, b, ctx, k)
        return ", ".join(f"s{s}:{o}" for s, o in zip(x["seeds"], x["onsets"]))

    items = []
    # ------------------------------------------------------------------ q5
    all_verdicts = {k: verdict(v) for k, v in keys.items()}
    kn = []
    for meas in ("registered", "amended"):
        for ctx in CONTEXTS:
            kn.append(f"beta=1.0 {ctx} {meas}: " + "; ".join(rng_str(meas, 1.0, ctx, k) for k in ORDERS))
    for ctx in CONTEXTS:
        md = med_diffs[f"amended|beta=1.0|{ctx}|5-4"]
        r45 = res("amended", 1.0, ctx, 4, 5)
        kn.append(f"beta=1.0 {ctx} amended: median(order5) - median(order4) = {fmt(md['median_difference'])} "
                  f"(boot 95% {fmt(md['boot_ci95'][0])} to {fmt(md['boot_ci95'][1])}); orders 4-5 share checkpoints "
                  + ", ".join(f"step {s['step']} (o4 {s['n_lower']}, o5 {s['n_higher']} seeds)"
                              for s in r45["shared_checkpoints"]))
    kn.append("verdicts at beta=1.0: " + ", ".join(f"{k}={v}" for k, v in all_verdicts.items()))
    kn.append(f"censored (no onset within {BUDGET} steps): registered {checks['reg_censored']}, amended "
              f"{checks['am_censored']}")
    kn.append(f"amended differs from registered in {checks['am_differs_from_reg']} runs (all order 2, left-censored "
              f"wrong-sign first activation): {checks['am_differs_from_reg_runs']}")

    o2r = {c: g("registered", 1.0, c, 2) for c in CONTEXTS}
    o2a = {c: g("amended", 1.0, c, 2) for c in CONTEXTS}
    o3 = {c: g("amended", 1.0, c, 3) for c in CONTEXTS}
    o4 = {c: g("amended", 1.0, c, 4) for c in CONTEXTS}
    o5 = {c: g("amended", 1.0, c, 5) for c in CONTEXTS}
    n_above = {c: sum(v > ORDER2_FLOOR for v in o2a[c]["onsets"]) for c in CONTEXTS}
    interp = (
        f"Not supported, under every evaluation (registered and amended onset; orders 2-5 and 3-5; all sensitivity "
        f"readings). The failure is orders 4 vs 5: at beta = 1.0 their median onsets are equal (chain "
        f"{fmt(o4['chain']['median'])} vs {fmt(o5['chain']['median'])}, iid {fmt(o4['iid']['median'])} vs "
        f"{fmt(o5['iid']['median'])} steps) and their seed ranges overlap (chain [{fmt(o4['chain']['min'])}, "
        f"{fmt(o4['chain']['max'])}] vs [{fmt(o5['chain']['min'])}, {fmt(o5['chain']['max'])}], iid "
        f"[{fmt(o4['iid']['min'])}, {fmt(o4['iid']['max'])}] vs [{fmt(o5['iid']['min'])}, {fmt(o5['iid']['max'])}]). "
        f"Below order 4 the order does set the onset: order 3 comes after order 2 (amended medians chain "
        f"{fmt(o2a['chain']['median'])} -> {fmt(o3['chain']['median'])}, iid {fmt(o2a['iid']['median'])} -> "
        f"{fmt(o3['iid']['median'])}) and order 4 after order 3 (-> {fmt(o4['chain']['median'])} / "
        f"{fmt(o4['iid']['median'])}), with disjoint order-3 / order-4 ranges in both contexts (order-3 max "
        f"{fmt(o3['chain']['max'])} / {fmt(o3['iid']['max'])} < order-4 min {fmt(o4['chain']['min'])} / "
        f"{fmt(o4['iid']['min'])}). Orders 2 and 3 are disjoint with the registered onset (order-2 max "
        f"{fmt(o2r['chain']['max'])} / {fmt(o2r['iid']['max'])} < order-3 min {fmt(o3['chain']['min'])} / "
        f"{fmt(o3['iid']['min'])}) but overlap with the amended onset, because {n_above['chain']} chain and "
        f"{n_above['iid']} iid order-2 seeds whose untrained network carries the planted pair with the wrong sign "
        f"reach the amended onset at up to {fmt(o2a['chain']['max'])} / {fmt(o2a['iid']['max'])} steps "
        f"(Amendment C's initial-offset effect; see q5_order2_floor). Whether the 4-5 overlap is a resolution limit "
        f"is in q5_resolution_D. Train context: beta = 1.0 amended medians chain vs iid "
        + ", ".join(f"o{k} {fmt(g('amended', 1.0, 'chain', k)['median'])} vs {fmt(g('amended', 1.0, 'iid', k)['median'])}"
                    for k in ORDERS) + ".")
    # qualitative claims in the text above, checked against the data
    assert not criterion[keys["am"]]["supported"] and all(v == "not_supported" for v in all_verdicts.values())
    for c in CONTEXTS:
        assert o3[c]["max"] < o4[c]["min"] and o2r[c]["max"] < o3[c]["min"] and o2a[c]["max"] >= o3[c]["min"]
        assert o2a[c]["median"] < o3[c]["median"] < o4[c]["median"] == o5[c]["median"]
    io = init_offset
    items.append({
        "id": "q5",
        "question": "Q5 Markov: does the order set the onset? At fixed strength and context, median onsets should "
                    "increase 2 < 3 < 4 < 5; supported if the order-wise seed ranges do not overlap at beta = 1.0 "
                    "in both contexts.",
        "definition_used": (
            f"Runs: markov-v1 one-coupling sets 1:b, 1,3:b, 1,2,4:b, 1,2,3,5:b (planted orders 2-5), b = 1.0 and "
            f"0.5, train_context chain and iid, every seed ({len(df)} runs, 10 per configuration; boundary "
            f"refinement added none here). Planted subset = the curve's ground truth {{x[t-l]}} + xhat (stored in "
            f"{checks['planted_stored']} curves). REGISTERED onset = step of the first 'activation' event of the "
            f"planted subset (DR1 ri_events; equal to detect_events(curve, alpha=0.01) in "
            f"{checks['table_events_equal_recomputed']} runs). An event's step is the FIRST of the two consecutive "
            f"active checkpoints (q <= 0.01 and RI >= min_effect = 0.001) that confirm it, so the onset lies in "
            f"(step_lo, step]; a left-censored activation (already active at step 0) counts as step 0, as written. "
            f"AMENDED onset (Amendment 1 C) = first activation or reactivation event of the planted subset that is "
            f"not left-censored and at whose step the planted Fourier coefficient has the sign of its value at "
            f"step 30,000 (positive in {checks['final_sign_positive']} runs, = planted sign). Runs without an onset "
            f"would be censored at 30,000 (none). CRITERION (registered text, unchanged by D): in each context, "
            f"medians strictly increasing in order AND the closed seed ranges [min, max] of all order pairs "
            f"disjoint (two orders sharing a checkpoint overlap); supported iff both contexts pass at beta = 1.0. "
            f"Evaluated for orders 2-5 (as written) and 3-5 (C), with both onsets. DECIDING: the registered "
            f"criterion (orders 2-5) applied to the amended onset (the amendment's rule: where it changes a "
            f"definition the amended one decides); all other evaluations are reported and agree. Sensitivity "
            f"readings of C: (a) only the first 'activation' event eligible (else censored at 30,000; "
            f"{checks['alt_activation_only_censored']} runs censored this way); (b) sign required at both confirming "
            f"checkpoints ({checks['alt_window_sign_differs_from_am']} runs change); (c) onset read at the "
            f"confirming (second) checkpoint instead of the event step. Medians: over seeds (10 -> mean of 5th and "
            f"6th). Bootstrap: {BOOT_N} resamples of seeds within a configuration, seed {BOOT_SEED}."),
        "literal_result": (f"Registered onset, orders 2-5 (as written): {crit_str(keys['lit'])}. Orders 3-5: "
                           f"{crit_str(keys['lit35'])}."),
        "literal_verdict": verdict(keys["lit"]),
        "amended_result": (f"Amended onset (C), orders 2-5: {crit_str(keys['am'])}. Orders 3-5 (C): "
                           f"{crit_str(keys['am35'])}. Sensitivity, orders 2-5 / 3-5: (a) activation-only "
                           f"{short(keys['a'])} / {short(keys['a35'])}; (b) sign at both confirming checkpoints "
                           f"{short(keys['b'])} / {short(keys['b35'])}; (c) confirming checkpoint "
                           f"{short(keys['c'])} / {short(keys['c35'])}."),
        "amended_verdict": verdict(keys["am"]),
        "deciding": "amended",
        "key_numbers": "\n".join(kn),
        "interpretation": interp,
        "caveats": (
            "Onsets are checkpoint steps of a 60-point log grid (true onset in (step_lo, step]); medians of 10 seeds "
            "can fall between checkpoints. The same seed number has the same initial weights in every coupling set "
            f"and context (initial planted coefficient identical; max spread "
            f"{checks['same_seed_initial_f_max_spread_within_order']:g}), so chain/iid and beta comparisons are "
            f"paired and the 10 seeds are the independent unit. Within an order the seed order follows the untrained "
            f"network's planted coefficient: e.g. order 3 at beta = 1.0, Spearman(initial f, onset) = "
            f"{io['beta=1.0|chain|order=3']['spearman_f_initial_vs_amended_onset']:.2f} (chain), "
            f"{io['beta=1.0|iid|order=3']['spearman_f_initial_vs_amended_onset']:.2f} (iid); median onset "
            f"{io['beta=1.0|chain|order=3']['median_onset_f_initial_pos']:g} vs "
            f"{io['beta=1.0|chain|order=3']['median_onset_f_initial_neg']:g} steps (chain) for positive vs negative "
            f"initial coefficient, so part of each order's seed range is the initial offset that Amendment C "
            f"describes. Orders 4-5 cannot be separated on this grid (Amendment D); this is a statement about the "
            f"60-point grid, not evidence that their onsets are equal."),
    })

    # ------------------------------------------------------------------ D
    kn, lit = [], []
    cls_short = {}
    for b in BETAS:
        for ctx in CONTEXTS:
            r = res("amended", b, ctx, 4, 5)
            p = r["all_seed_pairs"]
            kn.append(f"beta={b} {ctx}: order-4 onsets {{{onsets_by_seed('amended', b, ctx, 4)}}}; order-5 onsets "
                      f"{{{onsets_by_seed('amended', b, ctx, 5)}}}")
            kn.append(f"beta={b} {ctx}: checkpoint counts o4 {r['idx_counts_lower']}, o5 {r['idx_counts_higher']}; "
                      f"overlap depth {r['overlap_depth_idx']} -> {r['classification']}; all {p['n']} seed pairs: "
                      f"o5 later {p['higher_later']}, same checkpoint {p['same_checkpoint']}, o5 earlier by 1 "
                      f"{p['higher_earlier_by_1']}, o5 earlier by >1 {p['higher_earlier_by_more_than_1']}; same-seed "
                      f"idx(o5)-idx(o4) {r['same_seed_idx_difference_higher_minus_lower']}")
            cls_short[(b, ctx)] = r
            if b == 1.0:
                lit.append(f"{ctx}: o4 on checkpoints {sorted(int(i) for i in r['idx_counts_lower'])}, o5 on "
                           f"{sorted(int(i) for i in r['idx_counts_higher'])}; shared "
                           f"{[s['step'] for s in r['shared_checkpoints']]}; depth {r['overlap_depth_idx']} = "
                           f"{r['classification']}; seed pairs with o5 earlier by >1 checkpoint "
                           f"{p['higher_earlier_by_more_than_1']} of {p['n']}, by exactly 1 "
                           f"{p['higher_earlier_by_1']} of {p['n']}")
    grid = res("amended", 1.0, "chain", 4, 5)["grid_around"]
    lo_i = min(x["idx"] for x in grid)
    hi_i = max(x["idx"] for x in grid)
    kn.append(f"grid around the order-4/5 onsets (index:step, ratio to previous checkpoint), idx {lo_i}-{hi_i}: " +
              ", ".join(f"{x['idx']}:{x['step']} (x{x['ratio_to_previous']:.3f})" for x in grid))
    for ctx in CONTEXTS:
        r = res("registered", 1.0, ctx, 4, 5)
        rf = res("fourier_eps", 1.0, ctx, 4, 5)
        kn.append(f"beta=1.0 {ctx}: registered onset depth {r['overlap_depth_idx']} (identical to amended for "
                  f"orders 4-5); bare-eps Fourier onset (sign-corrected f >= 0.0447 at two consecutive checkpoints) "
                  f"depth {rf['overlap_depth_idx']}, checkpoint counts o4 {rf['idx_counts_lower']} o5 "
                  f"{rf['idx_counts_higher']}")
    for b in (1.0,):
        for ctx in CONTEXTS:
            ic = interp_cmp[f"beta={b}|{ctx}"]
            kn.append(f"exploratory log-step interpolated crossing of the planted coefficient, beta={b} {ctx}: " +
                      "; ".join(f"{lvl}: o4 median {v['order4_median']:.1f}, o5 median {v['order5_median']:.1f}, "
                                f"o5 later in {v['n_seeds_order5_later']} same-seed pairs (median ratio "
                                f"{v['median_same_seed_ratio']:.3f})" for lvl, v in ic.items()))
    kn.append(f"instrument: BH candidates per order {checks['n_candidates_by_order']}; the checkpoint before every "
              f"order-3-5 onset was blocked by {checks['orders3to5_prev_checkpoint_blocked_by']}; order-3/4 "
              f"confirming q rescaled to the order-5 candidate count <= "
              f"{checks['orders3to4_max_q_confirming_rescaled_to_order5_m']:.4f} -> the candidate count cannot move "
              f"any onset: {checks['candidate_count_cannot_move_onsets']}")
    for a, c_ in ((2, 3), (3, 4)):
        for ctx in CONTEXTS:
            r = res("amended", 1.0, ctx, a, c_)
            rr = res("registered", 1.0, ctx, a, c_)
            kn.append(f"beta=1.0 {ctx} orders {a}-{c_}: amended depth {r['overlap_depth_idx']} "
                      f"({r['classification']}), registered depth {rr['overlap_depth_idx']} ({rr['classification']})")
    rc, ri_ = cls_short[(1.0, "chain")], cls_short[(1.0, "iid")]
    pc, pi_ = rc["all_seed_pairs"], ri_["all_seed_pairs"]
    gen = [ctx for ctx, r in (("chain", rc), ("iid", ri_)) if r["overlap_depth_idx"] >= 2]
    ic_c, ic_i = interp_cmp["beta=1.0|chain"], interp_cmp["beta=1.0|iid"]
    # the steps of the genuine inversions (chain)
    inv = []
    for ctx, r in (("chain", rc), ("iid", ri_)):
        ga, gb = g("amended", 1.0, ctx, 4), g("amended", 1.0, ctx, 5)
        pairs = sorted({(x4, x5) for x4, i4 in zip(ga["onsets"], ga["checkpoint_idx"])
                        for x5, i5 in zip(gb["onsets"], gb["checkpoint_idx"]) if i5 < i4 - 1})
        inv.append((ctx, pairs))
    shared_txt = "; ".join(f"{c}: shared checkpoints {[s_['step'] for s_ in r['shared_checkpoints']]}, o4 on "
                           f"{[steps[int(i)] for i in sorted(int(i) for i in r['idx_counts_lower'])]}, o5 on "
                           f"{[steps[int(i)] for i in sorted(int(i) for i in r['idx_counts_higher'])]}"
                           for c, r in (("chain", rc), ("iid", ri_)))
    gr = [x["ratio_to_previous"] for x in grid if x["n_lower"] or x["n_higher"]]
    assert ic_i["25pct_pop"]["median_same_seed_ratio"] < min(gr)
    n_c25 = int(ic_c["25pct_pop"]["n_seeds_order5_later"].split()[0])
    assert 3 <= n_c25 <= 7, "chain interpolation is not 'no consistent order'"
    for b_ in BETAS:
        for c in CONTEXTS:
            if b_ == 0.5:
                p_ = cls_short[(0.5, c)]["all_seed_pairs"]
                assert p_["higher_earlier_by_1"] + p_["higher_earlier_by_more_than_1"] == 0
    d_interp = (
        f"Orders 4 and 5 fall on the same few consecutive checkpoints in both contexts at beta = 1.0 "
        f"({shared_txt}), as the Amendment D preview said; the grid spacing there is a factor "
        f"{min(x['ratio_to_previous'] for x in grid if x['n_lower'] or x['n_higher']):.2f}-"
        f"{max(x['ratio_to_previous'] for x in grid if x['n_lower'] or x['n_higher']):.2f}. "
        + (f"By the task's definition the overlap is GENUINE in {', '.join(gen)}: "
           + "; ".join(f"order-5 seeds at step {sorted({b_ for _, b_ in p})} vs order-4 seeds at step "
                       f"{sorted({a_ for a_, _ in p})}" for ctx, p in inv if p)
           + f" ({pc['higher_earlier_by_more_than_1']} of {pc['n']} chain seed pairs, a 2-checkpoint = "
             f"1.30x inversion); " if gen else "No order-5 seed is more than one checkpoint earlier than an order-4 "
                                               "seed in either context; ")
        + (f"in iid it is not genuine: no order-5 seed is more than one checkpoint earlier than an order-4 seed "
           f"({pi_['higher_earlier_by_more_than_1']} of {pi_['n']} pairs earlier by >1, "
           f"{pi_['higher_earlier_by_1']} by exactly one checkpoint, {pi_['same_checkpoint']} on the same "
           f"checkpoint), i.e. within the grid's resolution. " if ri_["overlap_depth_idx"] <= 1 else "")
        + f"Exploratory interpolation of the planted Fourier coefficient between checkpoints: in iid order 5 "
          f"crosses later than order 4 in {ic_i['25pct_pop']['n_seeds_order5_later']} same-seed pairs at 25 % of "
          f"tanh(beta) (median ratio {ic_i['25pct_pop']['median_same_seed_ratio']:.2f}; "
          f"{ic_i['eps_0.0447']['n_seeds_order5_later']} at eps), a lag smaller than one grid step; in chain there "
          f"is no consistent order ({ic_c['25pct_pop']['n_seeds_order5_later']} at 25 %, median ratio "
          f"{ic_c['25pct_pop']['median_same_seed_ratio']:.2f}). So: iid is consistent with order 5 trailing order 4 "
          f"by less than the grid resolves; chain shows orders 4 and 5 learned together with seed scatter larger "
          f"than any order effect, including small genuine inversions. At beta = 0.5 the two orders meet only on "
          f"shared checkpoints (depth {cls_short[(0.5, 'chain')]['overlap_depth_idx']} chain, "
          f"{cls_short[(0.5, 'iid')]['overlap_depth_idx']} iid; o5 never earlier). A dense replay of these objects "
          f"(the registered follow-up) is needed to order them; equal onsets are not demonstrated.")
    items.append({
        "id": "q5_resolution_D",
        "question": "Amendment D: at beta = 1.0 do the onsets of orders 4 and 5 fall on the same checkpoints of the "
                    "60-point grid, and is their overlap a resolution limit (same checkpoint) or genuine (order-5 "
                    "seeds earlier than order-4 seeds by more than one checkpoint)?",
        "definition_used": (
            "Checkpoint index of each run's amended onset (orders 4-5: identical to the registered onset). Overlap "
            "depth for consecutive orders k < k+1: d = max index(order k) - min index(order k+1). d < 0 no overlap; "
            "d = 0 the orders meet only on shared checkpoints (resolution limit); d = 1 the worst inversion is one "
            "checkpoint (within one checkpoint: not genuine by the task's definition, reported separately); d >= 2 "
            "genuine (an order-(k+1) seed earlier than an order-k seed by more than one checkpoint). Also counted: "
            "all 100 cross-seed pairs and the 10 same-seed pairs (same seed = same initial weights). Cross-check: "
            "bare-eps Fourier onset. Exploratory, no criterion: crossing step of the sign-corrected planted Fourier "
            "coefficient at eps = 0.0447 (RI = min_effect), 25 % and 50 % of tanh(beta), interpolated linearly in "
            "the coefficient against log(step) between the bracketing checkpoints."),
        "literal_result": "beta = 1.0, amended onsets, orders 4-5: " + " | ".join(lit),
        "literal_verdict": "descriptive",
        "key_numbers": "\n".join(kn),
        "interpretation": d_interp,
        "caveats": ("Instrument: the number of scanned candidates (hence the BH correction) grows with the planted "
                    "order, but it cannot move any order-3-5 onset (key_numbers), so the 4-5 tie is not a q-value "
                    "artefact. The genuine/resolution split follows the task's checkpoint-count definition; a 2-checkpoint "
                    "inversion here is 96 vs 125 steps (factor 1.30), comparable to seed scatter within one order. "
                    "The interpolation assumes a smooth coefficient between checkpoints and is not a registered "
                    "measure. Pre-registered criterion unaffected: it fails either way."),
        "deciding": "both_reported",
    })

    # ------------------------------------------------------------------ beta 0.5
    kn = []
    for meas in ("registered", "amended"):
        for ctx in CONTEXTS:
            kn.append(f"beta=0.5 {ctx} {meas}: " + "; ".join(rng_str(meas, 0.5, ctx, k) for k in ORDERS))
    for ctx in CONTEXTS:
        r = res("amended", 0.5, ctx, 4, 5)
        kn.append(f"beta=0.5 {ctx}: orders 4-5 checkpoint counts o4 {r['idx_counts_lower']} o5 "
                  f"{r['idx_counts_higher']}, depth {r['overlap_depth_idx']}; seed pairs o5 later "
                  f"{r['all_seed_pairs']['higher_later']}, same {r['all_seed_pairs']['same_checkpoint']}, o5 "
                  f"earlier {r['all_seed_pairs']['higher_earlier_by_1'] + r['all_seed_pairs']['higher_earlier_by_more_than_1']}")
    ratio_lines = []
    for ctx in CONTEXTS:
        ratio_lines.append(f"{ctx}: " + ", ".join(
            f"o{k} {fmt(g('amended', 0.5, ctx, k)['median'])} vs {fmt(g('amended', 1.0, ctx, k)['median'])}"
            for k in ORDERS))
    kn.append("amended median onset beta=0.5 vs beta=1.0: " + "; ".join(ratio_lines))
    k5, k5r = "amended|beta=0.5|orders=2-3-4-5", "registered|beta=0.5|orders=2-3-4-5"
    k5_35 = "amended|beta=0.5|orders=3-4-5"
    b5i = {c: [g("amended", 0.5, c, k) for k in ORDERS] for c in CONTEXTS}
    for c in CONTEXTS:  # claims of the interpretation below
        x2, x3, x4, x5 = b5i[c]
        assert x2["median"] < x3["median"] < x4["median"] == x5["median"] and x3["max"] < x4["min"]
        for k in (3, 4, 5):
            assert g("amended", 0.5, c, k)["median"] > g("amended", 1.0, c, k)["median"]
        assert res("amended", 0.5, c, 4, 5)["overlap_depth_idx"] == 0
    n162 = {k: sum(o == 162 for c in CONTEXTS for o in g("amended", 0.5, c, k)["onsets"]) for k in (4, 5)}
    items.append({
        "id": "q5_beta05",
        "question": "Q5 at beta = 0.5 (descriptive; the registered criterion is at beta = 1.0): onsets by order and "
                    "context.",
        "definition_used": "Same onsets (registered and amended), same arithmetic (strictly increasing medians, "
                           "pairwise disjoint closed seed ranges) applied to the beta = 0.5 coupling sets 1:0.5, "
                           "1,3:0.5, 1,2,4:0.5, 1,2,3,5:0.5; descriptive only.",
        "literal_result": f"Registered onset, orders 2-5: {crit_str(k5r)}.",
        "literal_verdict": "descriptive",
        "amended_result": f"Amended onset, orders 2-5: {crit_str(k5)}. Orders 3-5: {crit_str(k5_35)}.",
        "amended_verdict": "descriptive",
        "deciding": "both_reported",
        "key_numbers": "\n".join(kn),
        "interpretation": (
            "Same pattern as at beta = 1.0, shifted later for orders >= 3: order 3 after order 2, orders 4 and 5 "
            "after order 3 with disjoint ranges, but orders 4 and 5 not separated (equal medians "
            f"{fmt(b5i['chain'][2]['median'])}/{fmt(b5i['chain'][3]['median'])} chain, "
            f"{fmt(b5i['iid'][2]['median'])}/{fmt(b5i['iid'][3]['median'])} iid; they meet on one shared checkpoint "
            "and order 5 is never earlier than order 4). Weaker coupling delays orders 3-5 (medians in key_numbers) "
            "but leaves order 2 at the initial-offset scale. Chain vs iid amended medians: " + ", ".join(
                f"o{k} {fmt(g('amended', 0.5, 'chain', k)['median'])} vs {fmt(g('amended', 0.5, 'iid', k)['median'])}"
                for k in ORDERS) + "."),
        "caveats": (f"Descriptive; same grid limits as at beta = 1.0: the single checkpoint at step 162 holds "
                    f"{n162[4]} of 20 order-4 and {n162[5]} of 20 order-5 onsets (both contexts)."),
    })

    # ------------------------------------------------------------------ order 2 floor
    kn = []
    for key, v in order2.items():
        kn.append(f"{key}: registered {{{', '.join(f's{s}:{x}' for s, x in zip(v['seeds'], v['registered']))}}}; "
                  f"amended {{{', '.join(f's{s}:{x}' for s, x in zip(v['seeds'], v['amended']))}}}; amended <= "
                  f"{ORDER2_FLOOR} in {v['n_amended_le_floor']}; left-censored first activation in "
                  f"{v['n_registered_left_censored']}; initial planted coefficient < 0 in {v['n_initial_f_negative']}")
    above = [(key, a["seed"], a["onset"], a["event"], round(a["f_initial"], 3)) for key, v in order2.items()
             for a in v["amended_above_floor"]]
    kn.append(f"amended onsets above {ORDER2_FLOOR} steps (key, seed, step, event, initial f): {above}")
    n2 = int((df.order == 2).sum())
    seeds_above = sorted({a[1] for a in above})
    lc_runs = df[df.reg_left_censored]
    assert (lc_runs.order == 2).all() and (lc_runs.reg_sign_matches_final == False).all()  # noqa: E712
    lc2 = int(df[df.order == 2].reg_left_censored.sum())
    items.append({
        "id": "q5_order2_floor",
        "question": "Amendment C: order-2 onsets reported at the resolution floor (<= 8 steps); left-censored and "
                    "wrong-sign initial activations.",
        "definition_used": f"Order-2 runs (1:1.0 and 1:0.5, both contexts, {n2} runs). Amendment C: order-2 onsets "
                           f"are at the scale of the initial offset and are reported as <= {ORDER2_FLOOR} steps. "
                           f"Registered and amended onsets listed per seed; amended onsets above the floor flagged.",
        "literal_result": (f"Registered: left-censored (activation at step 0) in {lc2} of {n2} order-2 runs "
                           f"({checks['reg_left_censored']} of all runs; every one has a negative = wrong-sign "
                           f"planted coefficient at step 0); ranges " + "; ".join(
                               f"{k}: [{min(v['registered'])}, {max(v['registered'])}]" for k, v in order2.items())
                           + f" -> all <= {ORDER2_FLOOR} steps."),
        "literal_verdict": "descriptive",
        "amended_result": ("Amended: " + "; ".join(
            f"{k}: [{min(v['amended'])}, {max(v['amended'])}], <= {ORDER2_FLOOR} in {v['n_amended_le_floor']}"
            for k, v in order2.items()) + f". Above the floor: {len(above)} of {n2} runs."),
        "amended_verdict": "descriptive",
        "deciding": "both_reported",
        "key_numbers": "\n".join(kn),
        "interpretation": (
            f"Order-2 onsets are at the initial-offset scale, as Amendment C says. The registered onsets are all "
            f"<= {ORDER2_FLOOR} steps, but {lc2} of them are step-0 activations of the planted pair with the wrong "
            f"sign. Removing those (C) moves these runs to their first right-signed reactivation at "
            f"{min(a[2] for a in above) if above else 'n/a'}-{max(a[2] for a in above) if above else 'n/a'} steps, "
            f"so 'order 2 <= 8 steps' does not hold for every amended onset ({len(above)} of {n2} exceed it). These "
            f"are seeds {seeds_above} (same initial weights in every order-2 configuration). Reported at the floor "
            f"(<= {ORDER2_FLOOR}), order 2 would be disjoint from order 3 at beta = 1.0 (order-3 min "
            f"{fmt(g('amended', 1.0, 'chain', 3)['min'])} chain / {fmt(g('amended', 1.0, 'iid', 3)['min'])} iid), "
            f"but the question is decided by orders 4-5 either way."),
        "caveats": "The amended onset of these runs measures when the network undoes a random initial "
                   "anti-correlation, not when it first learns the coupling; neither reading gives a "
                   "well-resolved order-2 onset on this grid.",
    })

    # ------------------------------------------------------------------ censored
    items.append({
        "id": "q5_censored",
        "question": f"Runs that never activate the planted subset within the budget (censored at {BUDGET} steps).",
        "definition_used": f"Censored = no onset within the {BUDGET}-step budget (registered: no activation event of "
                           f"the planted subset; amended: no qualifying (re)activation); such runs would enter "
                           f"medians and ranges at {BUDGET}, never dropped.",
        "literal_result": f"Registered: {checks['reg_censored']} runs censored. Amended: {checks['am_censored']}.",
        "literal_verdict": "descriptive",
        "key_numbers": (f"censored: registered {checks['reg_censored']}, amended {checks['am_censored']} (sensitivity "
                        f"(a) activation-only: {checks['alt_activation_only_censored']}, the left-censored order-2 "
                        f"runs)\nplanted subset stored in {checks['planted_stored']} curves\nfinal planted "
                        f"coefficient positive (planted sign) in {checks['final_sign_positive']}; final / population "
                        f"tanh(beta) in [{checks['final_over_population_min_max'][0]:.3f}, "
                        f"{checks['final_over_population_min_max'][1]:.3f}] (population values "
                        f"{checks['population_coefficient_by_beta']} by enumeration of the 2^6 windows)\nfinal "
                        f"kl_true < 0.01 in {checks['kl_true_final_lt_0.01']} (max "
                        f"{checks['kl_true_final_max']:.4f} nats)"),
        "interpretation": "Every run of the eight coupling sets activates and learns its planted interaction within "
                          "the budget; no censoring enters the medians or ranges.",
        "caveats": "",
        "deciding": "both_reported",
    })
    return items


if __name__ == "__main__":
    main()
