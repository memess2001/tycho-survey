"""Sky v1, pre-registered question 6 (Markov: is the v0 delay a context-correlation effect?), incl. Amendment 1 C, E.

Pre-registration: docs/results/v1_preregistration.md (written before the run; Amendment 1 written after launch,
before v1 data were read). Data: releases/DR1 (tables) and the extracted RI curves / light curves. v0 baseline:
releases/DR0 (event table only).

Comparisons (region markov-v1, both train contexts chain and iid, every seed of the current sky)
  c13  : "1,3:0.8" (alone)   vs "1:1.0;1,3:0.8"   (with the pairwise term); subset {x[t-1], x[t-3], xhat}
  c123 : "1,2,3:1.0" (alone) vs "1:1.0;1,2,3:1.0" (with the pairwise term); subset {x[t-1], x[t-2], x[t-3], xhat}
  R = median onset (with) / median onset (alone), per context. 95 % bootstrap CI: 2,000 resamples, seeds resampled
  independently within each arm (couplings x context), percentile interval. The same resample indices of an arm are
  reused for every onset definition (so the definitions are compared on identical resamples), and R_chain / R_iid
  uses the four arms' independent resamples. Runs without an onset are censored at the budget (30,000) and kept.

Onset definitions (all onsets are checkpoint steps of the 60-point in-situ grid)
  registered   first "activation" event of the subset (DR1 ri_events; cross-checked against
               detect_events(curve, alpha=0.01)). events.py: the event's step is steps[t0], the FIRST of the two
               consecutive active checkpoints (q <= 0.01 and RI >= min_effect = 0.001), so the onset lies in
               (step_lo, step]; the state machine confirms it one checkpoint later (steps[t0+1], recorded as
               confirm_step). A left-censored activation (t0 = 0) would count as step 0, as written.
  amended C    Amendment 1 C: the first (re)activation event that is not left-censored and at whose checkpoint (the
               event's step) the Fourier coefficient has the sign of its final value (step 30,000). Alternatives:
               (a) only the first "activation" event is eligible; (b) the sign must hold at both confirming
               checkpoints. (Same reading as scripts/analysis_v1/markov_order.py, question 5.)
  amended E    Amendment 1 E (DECIDES, at 25 %): the first checkpoint where g = f * sign(population value) (the
               planted sign) reaches 25 % of the POPULATION Walsh coefficient
                   pop(A) = E_uniform[tanh(h(s)) prod_{l in A} s_l],  h(s) = sum_L beta_L prod_{l in L} s_{t-l},
               computed exactly by enumerating the 2^6 windows (checked: 0.664 for "1,3:0.8" and 0.375 for
               "1:1.0;1,3:0.8" on {1,3}). Also 10 % and 50 %. Literal reading: a single checkpoint. Alternatives:
               (a) two consecutive checkpoints at or above the level (the events' persistence rule), (b) the crossing
               interpolated linearly in g against log(step) between the bracketing checkpoints (sub-grid resolution,
               descriptive), (c) |f| instead of the planted sign.

Interpretation (registered branches): R_iid ~ 1 (CI includes 1) while R_chain > 1 (CI above 1): the delay came
from correlated contexts. R_iid > 1 (CI above 1): lower-order structure delays higher-order learning even with
independent contexts. Any other combination is reported as neither branch. Amendment E's pilot preview
(R_iid > 1 under every definition, 1.5 to 3.2; R_chain larger, 2.9 to 6.1) and Amendment B's preview
({1,2,3}:1.0 onset 162-273 with chain contexts vs 96-125 iid) are compared with the real values.

Also reported: R_chain / R_iid with its CI (how much the context correlation adds to the delay); the v0 baseline
R (DR0 events, chain contexts, 7 seeds, registered onset); and, descriptively, why the RI onset of the
four-way term lags its Fourier crossing in chain runs with the pairwise term (the stored Fourier rows of every
B subset of A reconstruct the RI of {x_A, xhat} under uniform probes; lower-order chain-induced coefficients
cancel the top-order log-linear term).

Usage (cwd = repository root):
    PYTHONPATH=. python scripts/analysis_v1/markov_context.py [--release releases/DR1] [--extracted DIR] [--out JSON]
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
from tycho.ri.refined import local_ri
from tycho.tasks.markov_lang import parse_couplings

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACTED = REPO / "work" / "dr1"  # extracted DR1 archives (git-ignored); see README.md
DEFAULT_OUT = REPO / "docs/results/v1_analysis/markov_context.json"
REGION = "markov-v1"
COMPARISONS = {
    "c13": {"alone": "1,3:0.8", "with": "1:1.0;1,3:0.8", "subset": ("x[t-1]", "x[t-3]", "xhat"),
            "label": "{1,3}:0.8 alone vs with {1}:1.0"},
    "c123": {"alone": "1,2,3:1.0", "with": "1:1.0;1,2,3:1.0", "subset": ("x[t-1]", "x[t-2]", "x[t-3]", "xhat"),
             "label": "{1,2,3}:1.0 alone vs with {1}:1.0"},
}
ARMS = ("alone", "with")
CONTEXTS = ("chain", "iid")
WINDOW = 6
ALPHA = 0.01
BUDGET = 30000
LEVELS = (0.10, 0.25, 0.50)
BOOT_N = 2000
BOOT_SEED = 20261112
EPS_MIN_EFFECT = 0.0447  # |f| at which ln2 - h((1+|f|)/2) = 0.001 (checked below)
PREVIEW_E = {"R_iid_range": [1.5, 3.2], "R_chain_range": [2.9, 6.1]}
PREVIEW_B = {"chain": [162, 273], "iid": [96, 125]}  # {1,2,3}:1.0 onset, Amendment 1 B (pilot)
V0_REPORTED = {"alone": [14, 36], "with": [96, 133]}  # v0_findings.md


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
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return fnum(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def lags_of_names(names) -> tuple[int, ...]:
    return tuple(sorted(int(n[len("x[t-"):-1]) for n in names if n.startswith("x[t-")))


def population_coefficient(couplings: str, lags: tuple[int, ...]) -> float:
    """E_uniform[tanh(h(s)) prod_{l in lags} s_l] over the 2^6 windows (bias 0)."""
    cp = parse_couplings(couplings, WINDOW)
    tot = 0.0
    for s in itertools.product((-1.0, 1.0), repeat=WINDOW):  # s[l-1] = s_{t-l}
        h = sum(b * np.prod([s[l - 1] for l in L]) for L, b in cp.items())
        tot += math.tanh(h) * np.prod([s[l - 1] for l in lags])
    return tot / 2 ** WINDOW


def first_crossing(g: np.ndarray, level: float, persistence: int = 1) -> int | None:
    """First checkpoint index t with g[t..t+persistence-1] >= level (NaN never reaches)."""
    with np.errstate(invalid="ignore"):
        ok = g >= level
    T = len(g)
    for t in range(T - persistence + 1):
        if ok[t:t + persistence].all():
            return t
    return None


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


def reconstructed_ri(c: RICurve, names: tuple[str, ...], t: int) -> float:
    """Local RI of {x_A, xhat} rebuilt from the stored Walsh coefficients f(B), B a nonempty subset of A.

    Under uniform probes E[2 p1 - 1 | x_A] = sum_{B subset A} f(B) chi_B(x_A); f(empty) (the mean output) is not
    stored and is set to 0. So this is an approximation (also the probe's x_A marginal is only nearly uniform).
    """
    lagn = [n for n in names if n != "xhat"]
    k = len(lagn)
    coef = {}
    for r in range(1, k + 1):
        for B in itertools.combinations(range(k), r):
            i = c.index_of([lagn[j] for j in B] + ["xhat"])
            coef[B] = float(c.fourier[i, t])
    P = np.zeros((2,) * (k + 1))
    for codes in itertools.product((0, 1), repeat=k):
        s = [2 * x - 1 for x in codes]
        m = sum(v * np.prod([s[j] for j in B]) for B, v in coef.items())
        p1 = min(max((1.0 + m) / 2.0, 1e-12), 1 - 1e-12)
        P[codes + (1,)] = p1 / 2 ** k
        P[codes + (0,)] = (1 - p1) / 2 ** k
    return float(local_ri(P))


# ---------------------------------------------------------------------------------------------- per run
def analyse_run(row: pd.Series, tab_events: pd.DataFrame, extracted: Path, subset: tuple[str, ...]) -> dict:
    c = RICurve.load(extracted / row.ricurve_file)
    steps = c.steps
    step_list = [int(s) for s in steps]
    T = len(steps)
    assert c.meta.get("kind") == "insitu" and abs(float(c.meta["min_effect"]) - 1e-3) < 1e-12
    gt_sets = [frozenset(g["subset"]) for g in (c.ground_truth or [])]
    in_gt = frozenset(subset) in gt_sets
    gi = c.index_of(list(subset))
    assert gi is not None, f"{row.object_id}: subset {subset} not stored"
    assert in_gt, f"{row.object_id}: subset {subset} is not a ground-truth entry {c.ground_truth}"
    lags = lags_of_names(subset)
    ri, q, f = c.ri[gi], c.q[gi], c.fourier[gi]
    assert np.isfinite(f).all(), f"{row.object_id}: NaN Fourier row"
    pop = population_coefficient(row.couplings, lags)
    planted_sign = float(np.sign(pop))
    f_final = float(f[-1])
    sgn_final = float(np.sign(f_final))

    # events of the subset: recomputed and from the table
    rec = [e for e in detect_events(c, alpha=ALPHA) if set(e.subset_names) == set(subset)]
    tab = tab_events[tab_events.subset_json.map(lambda s: set(json.loads(s)) == set(subset))].sort_values(
        ["step", "event_id"])
    rec_sig = [(e.kind, e.step, e.step_lo) for e in rec]
    tab_sig = [(k, int(s), int(lo)) for k, s, lo in zip(tab.kind, tab.step, tab.step_lo)]
    events_match = rec_sig == tab_sig
    tab_gt_flag = bool(tab.matches_gt.all()) if len(tab) else None
    on_events = [e for e in rec if e.kind != "extinction"]

    def t_of(e):
        return step_list.index(e.step)

    # registered: first activation event (table)
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

    # amended C
    def amended(events, window_sign: bool):
        skipped = []
        for e in events:
            t0 = t_of(e)
            if e.left_censored:
                skipped.append({"kind": e.kind, "step": e.step, "reason": "left-censored", "f": float(f[t0])})
                continue
            ts = [t0, t0 + 1] if window_sign else [t0]
            ok = all(np.sign(f[t]) == sgn_final and sgn_final != 0 for t in ts if t < T)
            if not ok:
                skipped.append({"kind": e.kind, "step": e.step, "reason": "wrong sign", "f": float(f[t0])})
                continue
            return e, t0, skipped
        return None, None, skipped

    e_am, t_am, skipped_am = amended(on_events, window_sign=False)
    e_aa, t_aa, _ = amended([e for e in on_events if e.kind == "activation"][:1], window_sign=False)
    e_aw, t_aw, _ = amended(on_events, window_sign=True)

    # amended E
    g = f * planted_sign
    E = {}
    for lv in LEVELS:
        level = lv * abs(pop)
        t1 = first_crossing(g, level, 1)
        t2 = first_crossing(g, level, 2)
        ta = first_crossing(np.abs(f), level, 1)
        E[lv] = {"level": level, "t": t1, "t_two": t2, "t_abs": ta, "interp": interp_crossing(steps, g, level),
                 "below_after_t": None if t1 is None else bool((g[t1:] < level).any())}

    # Fourier at the bare threshold eps (two consecutive, sign-corrected; descriptive)
    pair = (g[:-1] >= EPS_MIN_EFFECT) & (g[1:] >= EPS_MIN_EFFECT)
    t_eps = int(np.flatnonzero(pair)[0]) if pair.any() else None

    lc = np.load(extracted / row.lightcurve_file)
    kl = lc["x_kl_true"]
    kl_final = float(kl[np.isfinite(kl)][-1]) if np.isfinite(kl).any() else float("nan")

    def st(t):
        return None if t is None else step_list[t]

    def val(t):
        return BUDGET if t is None else step_list[t]

    # RI vs Fourier at the E25 crossing: lower-order coefficients and the reconstructed RI
    t25 = E[0.25]["t"]
    lower = {}
    if t25 is not None:
        lagn = [n for n in subset if n != "xhat"]
        for r in range(1, len(lagn)):
            for B in itertools.combinations(lagn, r):
                i = c.index_of(list(B) + ["xhat"])
                lower["{" + ",".join(n[2:-1] for n in B) + "}"] = float(c.fourier[i, t25])
    recon_curve = [reconstructed_ri(c, subset, t) for t in range(T)]

    out = {
        "object_id": row.object_id, "ri_run_id": int(row.ri_run_id), "couplings": row.couplings,
        "comparison": row.comparison, "arm": row.arm, "context": row.context, "seed": int(row.seed),
        "source": row.source, "label": row.label, "ricurve_file": row.ricurve_file,
        "lightcurve_file": row.lightcurve_file, "subset": list(subset), "subset_stored": True,
        "subset_in_ground_truth": in_gt, "ground_truth": [list(g_["subset"]) for g_ in (c.ground_truth or [])],
        "n_subsets": c.S, "n_candidates": c.meta.get("n_candidates"), "grid": tuple(step_list),
        "events_subset": [{"kind": e.kind, "step": e.step, "step_lo": e.step_lo, "idx": t_of(e),
                           "left_censored": e.left_censored, "f_at_step": float(f[t_of(e)]),
                           "ri_after": e.ri_after, "peak_ri": e.peak_ri} for e in rec],
        "events_table_match_recomputed": events_match, "events_table_matches_gt_flag": tab_gt_flag,
        "n_events_table": len(tab),
        # registered
        "reg_onset_step": reg_step, "reg_onset_idx": reg_t, "reg_onset_step_lo": reg_lo,
        "reg_confirm_step": None if reg_t is None or reg_t + 1 >= T else step_list[reg_t + 1],
        "reg_left_censored": reg_left, "reg_censored": reg_step is None,
        "reg_onset_value": BUDGET if reg_step is None else reg_step,
        "reg_f_at_onset": None if reg_t is None else float(f[reg_t]),
        "reg_ri_at_onset": None if reg_t is None else float(ri[reg_t]),
        "reg_sign_matches_final": None if reg_t is None else bool(np.sign(f[reg_t]) == sgn_final),
        # amended C
        "C_onset_value": val(t_am), "C_onset_idx": t_am, "C_censored": e_am is None,
        "C_event_kind": None if e_am is None else e_am.kind, "C_skipped_events": skipped_am,
        "C_differs_from_reg": t_am != reg_t,
        "C_alt_activation_only_value": val(t_aa), "C_alt_activation_only_idx": t_aa,
        "C_alt_activation_only_censored": e_aa is None,
        "C_alt_window_sign_value": val(t_aw), "C_alt_window_sign_idx": t_aw,
        "C_alt_window_sign_censored": e_aw is None,
        # Fourier
        "population_coefficient": pop, "planted_sign": planted_sign,
        "population_single_mode_ri": float(single_mode_ri(pop)),
        "f_initial": float(f[0]), "f_final": f_final, "sign_final": sgn_final,
        "final_over_population": f_final / pop, "f_min": float(f.min()), "f_max": float(f.max()),
        "ri_initial": float(ri[0]), "ri_final": float(ri[-1]), "peak_ri": float(np.nanmax(ri)),
        "kl_true_final": kl_final,
        "fourier_eps_onset_step": st(t_eps), "fourier_eps_onset_idx": t_eps,
        "ri_at_E25": None if t25 is None else float(ri[t25]),
        "f_at_E25": None if t25 is None else float(f[t25]),
        "q_at_E25": None if t25 is None else float(q[t25]),
        "single_mode_ri_at_E25": None if t25 is None else float(single_mode_ri(f[t25])),
        "reconstructed_ri_at_E25": None if t25 is None else recon_curve[t25],
        "lower_order_f_at_E25": lower,
        "reconstructed_vs_stored_ri_max_abs_diff": float(np.max(np.abs(np.asarray(recon_curve) - c.ri_plugin[gi]))),
        "reconstructed_vs_stored_ri_median_abs_diff": float(np.median(np.abs(np.asarray(recon_curve)
                                                                             - c.ri_plugin[gi]))),
        "stored_ri_at_max_reconstruction_diff": float(c.ri_plugin[gi, int(np.argmax(np.abs(
            np.asarray(recon_curve) - c.ri_plugin[gi])))]),
        "reconstructed_ri_at_E25_minus_stored_plugin": None if t25 is None else recon_curve[t25] - float(
            c.ri_plugin[gi, t25]),
        "f_E25_to_E50_range": None if (t25 is None or E[0.5]["t"] is None) else [
            float(g[t25:E[0.5]["t"]].min()), float(g[t25:E[0.5]["t"]].max())],
        "steps_E25_to_E50": None if (t25 is None or E[0.5]["t"] is None) else step_list[E[0.5]["t"]] - step_list[
            t25],
        "reconstructed_ri_at_reg_onset": None if reg_t is None else recon_curve[reg_t],
        "gap_reg_minus_E25_checkpoints": None if (reg_t is None or t25 is None) else reg_t - t25,
    }
    for lv in LEVELS:
        tag = f"E{int(round(lv * 100))}"
        e_ = E[lv]
        out[f"{tag}_level"] = e_["level"]
        out[f"{tag}_onset_value"] = val(e_["t"])
        out[f"{tag}_onset_idx"] = e_["t"]
        out[f"{tag}_censored"] = e_["t"] is None
        out[f"{tag}_left_censored"] = e_["t"] == 0
        out[f"{tag}_dips_below_later"] = e_["below_after_t"]
        out[f"{tag}_two_onset_value"] = val(e_["t_two"])
        out[f"{tag}_two_onset_idx"] = e_["t_two"]
        out[f"{tag}_two_censored"] = e_["t_two"] is None
        out[f"{tag}_abs_onset_value"] = val(e_["t_abs"])
        out[f"{tag}_abs_onset_idx"] = e_["t_abs"]
        out[f"{tag}_abs_censored"] = e_["t_abs"] is None
        out[f"{tag}_interp_onset_value"] = BUDGET if e_["interp"] is None else e_["interp"]
        out[f"{tag}_interp_onset_idx"] = e_["t"]
        out[f"{tag}_interp_censored"] = e_["interp"] is None
    return out


# ---------------------------------------------------------------------------------------------- definitions
DEFS = {  # name -> (value col, idx col, censored col, role)
    "registered": ("reg_onset_value", "reg_onset_idx", "reg_censored", "literal (registered)"),
    "amended_C": ("C_onset_value", "C_onset_idx", "C_censored", "Amendment 1 C"),
    "amended_C_alt_activation_only": ("C_alt_activation_only_value", "C_alt_activation_only_idx",
                                      "C_alt_activation_only_censored", "Amendment C, alternative reading (a)"),
    "amended_C_alt_window_sign": ("C_alt_window_sign_value", "C_alt_window_sign_idx", "C_alt_window_sign_censored",
                                  "Amendment C, alternative reading (b)"),
}
for _lv in LEVELS:
    _tag = f"E{int(round(_lv * 100))}"
    DEFS[_tag] = (f"{_tag}_onset_value", f"{_tag}_onset_idx", f"{_tag}_censored",
                  "Amendment 1 E (DECIDES)" if _tag == "E25" else "Amendment 1 E, other level")
    DEFS[f"{_tag}_two"] = (f"{_tag}_two_onset_value", f"{_tag}_two_onset_idx", f"{_tag}_two_censored",
                           "Amendment E, alternative (a): two consecutive checkpoints")
    DEFS[f"{_tag}_interp"] = (f"{_tag}_interp_onset_value", f"{_tag}_interp_onset_idx", f"{_tag}_interp_censored",
                              "Amendment E, alternative (b): interpolated in log(step) (descriptive)")
    DEFS[f"{_tag}_abs"] = (f"{_tag}_abs_onset_value", f"{_tag}_abs_onset_idx", f"{_tag}_abs_censored",
                           "Amendment E, alternative (c): |f| instead of the planted sign")
MAIN_DEFS = ("registered", "amended_C", "E10", "E25", "E50")


def ci95(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def branch(ci_iid: list[float], ci_chain: list[float], r_iid: float, r_chain: float) -> dict:
    iid_above = ci_iid[0] > 1
    iid_incl = ci_iid[0] <= 1 <= ci_iid[1]
    chain_above = ci_chain[0] > 1
    if iid_above:
        b = "lower-order delays higher-order even with iid contexts (registered branch 2)"
        key = "branch2"
    elif iid_incl and chain_above:
        b = "delay came from correlated contexts (registered branch 1, the confound)"
        key = "branch1"
    elif iid_incl:
        b = "neither branch: R_iid CI includes 1 and R_chain CI does not lie above 1"
        key = "neither_no_chain_delay"
    else:
        b = "neither branch: R_iid CI lies below 1 (the pairwise term speeds up the higher-order term)"
        key = "neither_iid_faster"
    # point-estimate reading (alternative)
    if r_iid > 1 and not iid_incl:
        pt = "branch2"
    elif r_chain > 1 and iid_incl:
        pt = "branch1"
    else:
        pt = key
    return {"branch": key, "description": b, "iid_ci_above_1": iid_above, "iid_ci_includes_1": iid_incl,
            "chain_ci_above_1": chain_above, "chain_greater_than_iid_point": r_chain > r_iid,
            "point_reading_R_chain_gt_1": bool(r_chain > 1), "point_reading_R_iid_gt_1": bool(r_iid > 1)}


# ---------------------------------------------------------------------------------------------- v0 baseline
def v0_baseline(dr0: Path, rng_seed: int) -> dict:
    o = pd.read_parquet(dr0 / "objects.parquet")
    e = pd.read_parquet(dr0 / "ri_events.parquet")
    r = pd.read_parquet(dr0 / "ri_runs.parquet")
    m = o[o.task == "markov_lang"].copy()
    cfg = m.task_cfg_json.map(json.loads)
    m["couplings"] = cfg.map(lambda c: c.get("couplings"))
    m["context"] = cfg.map(lambda c: c.get("train_context", "chain (default; key absent in v0)"))
    m = m.merge(r[["object_id", "ri_run_id", "n_points", "min_effect", "alpha", "kind"]], on="object_id")
    out = {"source": "releases/DR0 ri_events (registered onset: first activation event); v0 had chain contexts "
                     "only, 30 log-spaced checkpoints, 15,000 steps, 7 seeds", "arms": {}}
    sub = {"x[t-1]", "x[t-3]", "xhat"}
    vals = {}
    for arm, cp in (("alone", "1,3:0.8"), ("with", "1:1.0;1,3:0.8")):
        d = m[m.couplings == cp].sort_values("seed")
        on, lc = [], []
        for _, row in d.iterrows():
            ee = e[(e.ri_run_id == row.ri_run_id) & (e.kind == "activation")]
            ee = ee[ee.subset_json.map(lambda s: set(json.loads(s)) == sub)].sort_values("step")
            if len(ee):
                on.append(int(ee.step.iloc[0]))
                lc.append(bool(np.isnan(ee.ri_before.iloc[0])))
            else:
                on.append(int(row.train_steps))
                lc.append(False)
        vals[arm] = np.asarray(on, float)
        out["arms"][arm] = {"couplings": cp, "seeds": d.seed.astype(int).tolist(), "onsets": on,
                            "n": len(on), "n_left_censored": int(sum(lc)),
                            "n_censored": int(sum(o_ == int(d.train_steps.iloc[0]) for o_ in on)),
                            "median": float(np.median(on)), "range": [min(on), max(on)],
                            "context": sorted(set(d.context)), "train_steps": sorted(set(d.train_steps.astype(int))),
                            "n_points": sorted(set(d.n_points.astype(int)))}
    rng = np.random.default_rng([rng_seed, 999])
    ia = rng.integers(0, len(vals["alone"]), size=(BOOT_N, len(vals["alone"])))
    iw = rng.integers(0, len(vals["with"]), size=(BOOT_N, len(vals["with"])))
    bR = np.median(vals["with"][iw], axis=1) / np.median(vals["alone"][ia], axis=1)
    out["R"] = float(np.median(vals["with"]) / np.median(vals["alone"]))
    out["R_boot_ci95"] = ci95(bR)
    out["reported_in_v0_findings"] = V0_REPORTED
    out["ranges_agree_with_v0_findings"] = (out["arms"]["alone"]["range"] == V0_REPORTED["alone"]
                                           and out["arms"]["with"]["range"] == V0_REPORTED["with"])
    return out


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases/DR1")
    ap.add_argument("--dr0", type=Path, default=REPO / "releases/DR0")
    ap.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    assert Path(tycho.__file__).resolve().is_relative_to(REPO), tycho.__file__
    assert abs(float(single_mode_ri(EPS_MIN_EFFECT)) - 1e-3) < 2e-6

    # population coefficients required by the task
    pop_checks = {
        "1,3:0.8 on {1,3}": population_coefficient("1,3:0.8", (1, 3)),
        "1:1.0;1,3:0.8 on {1,3}": population_coefficient("1:1.0;1,3:0.8", (1, 3)),
        "1,2,3:1.0 on {1,2,3}": population_coefficient("1,2,3:1.0", (1, 2, 3)),
        "1:1.0;1,2,3:1.0 on {1,2,3}": population_coefficient("1:1.0;1,2,3:1.0", (1, 2, 3)),
    }
    assert abs(pop_checks["1,3:0.8 on {1,3}"] - 0.664) < 5e-4, pop_checks
    assert abs(pop_checks["1:1.0;1,3:0.8 on {1,3}"] - 0.375) < 5e-4, pop_checks
    closed_form = {"tanh(0.8)": math.tanh(0.8), "(tanh(1.8)-tanh(0.2))/2": (math.tanh(1.8) - math.tanh(0.2)) / 2,
                   "tanh(1.0)": math.tanh(1.0), "tanh(2)/2": math.tanh(2.0) / 2}
    assert abs(pop_checks["1:1.0;1,3:0.8 on {1,3}"] - closed_form["(tanh(1.8)-tanh(0.2))/2"]) < 1e-12
    assert abs(pop_checks["1:1.0;1,2,3:1.0 on {1,2,3}"] - closed_form["tanh(2)/2"]) < 1e-12

    objs = pd.read_parquet(args.release / "objects.parquet")
    runs = pd.read_parquet(args.release / "ri_runs.parquet")
    evs = pd.read_parquet(args.release / "ri_events.parquet")
    m = objs[objs.region == REGION].copy()
    cfg = m.task_cfg_json.map(json.loads)
    m["couplings"] = cfg.map(lambda c: c.get("couplings"))
    m["context"] = cfg.map(lambda c: c.get("train_context"))
    m["bias"] = cfg.map(lambda c: c.get("bias", 0.0))
    arm_of = {}
    for ck, cv in COMPARISONS.items():
        for arm in ARMS:
            arm_of[cv[arm]] = (ck, arm)
    m = m[m.couplings.isin(arm_of)].copy()
    m["comparison"] = m.couplings.map(lambda s: arm_of[s][0])
    m["arm"] = m.couplings.map(lambda s: arm_of[s][1])
    assert (m.bias == 0.0).all()
    assert (m.train_steps == BUDGET).all() and (m.steps_total == BUDGET).all()
    assert set(m.context) == set(CONTEXTS)
    m = m.merge(runs[["object_id", "ri_run_id", "kind", "ricurve_file", "n_points", "min_effect", "alpha"]],
                on="object_id", how="left", validate="one_to_one")
    assert m.ri_run_id.notna().all() and (m.kind == "insitu").all() and (m.n_points == 60).all()

    rows = []
    for _, row in m.sort_values(["comparison", "arm", "context", "seed"]).iterrows():
        subset = COMPARISONS[row.comparison]["subset"]
        rows.append(analyse_run(row, evs[evs.ri_run_id == row.ri_run_id], args.extracted, subset))
    df = pd.DataFrame(rows)
    grids = set(df.grid)
    assert len(grids) == 1, "runs do not share one checkpoint grid"
    steps = list(next(iter(grids)))

    # ------------------------------------------------------------------ bootstrap indices per arm
    arm_keys = [(ck, arm, ctx) for ck in COMPARISONS for arm in ARMS for ctx in CONTEXTS]
    boot_idx, arm_df = {}, {}
    for j, key in enumerate(arm_keys):
        ck, arm, ctx = key
        d = df[(df.comparison == ck) & (df.arm == arm) & (df.context == ctx)].sort_values("seed").reset_index(
            drop=True)
        arm_df[key] = d
        rng = np.random.default_rng([BOOT_SEED, j])
        boot_idx[key] = rng.integers(0, len(d), size=(BOOT_N, len(d)))

    def arm_stats(key, dname):
        col, icol, ccol, _ = DEFS[dname]
        d = arm_df[key]
        v = d[col].to_numpy(float)
        bm = np.median(v[boot_idx[key]], axis=1)
        return {"n": int(len(d)), "seeds": d.seed.astype(int).tolist(), "onsets": [fnum(x) for x in v],
                "checkpoint_idx": [None if pd.isna(x) else int(x) for x in d[icol]],
                "censored": d[ccol].astype(bool).tolist(), "n_censored": int(d[ccol].sum()),
                "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                "median_boot_ci95": ci95(bm)}, v, bm

    results = {}
    for ck in COMPARISONS:
        results[ck] = {}
        for dname in DEFS:
            per_ctx, boots = {}, {}
            for ctx in CONTEXTS:
                sa, va, ba = arm_stats((ck, "alone", ctx), dname)
                sw, vw, bw = arm_stats((ck, "with", ctx), dname)
                with np.errstate(divide="ignore", invalid="ignore"):
                    R = float(np.median(vw) / np.median(va)) if np.median(va) > 0 else float("nan")
                    bR = bw / ba
                boots[ctx] = bR
                per_ctx[ctx] = {"alone": sa, "with": sw, "R": R, "R_boot_ci95": ci95(bR),
                                "R_boot_frac_le_1": float(np.mean(bR <= 1)),
                                "R_boot_frac_eq_1": float(np.mean(bR == 1)),
                                "R_boot_frac_lt_1": float(np.mean(bR < 1)),
                                "n_with_onsets_before_min_alone": int((vw < va.min()).sum()),
                                "n_with_onsets_at_or_before_max_alone": int((vw <= va.max()).sum()),
                                "R_boot_nonfinite": int((~np.isfinite(bR)).sum()),
                                "ranges_disjoint_with_later": bool(sw["min"] > sa["max"])}
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = boots["chain"] / boots["iid"]
            ratio = {"R_chain_over_R_iid": per_ctx["chain"]["R"] / per_ctx["iid"]["R"],
                     "boot_ci95": ci95(rr), "boot_frac_le_1": float(np.mean(rr <= 1))}
            results[ck][dname] = {
                "definition_role": DEFS[dname][3], "per_context": per_ctx, "chain_over_iid": ratio,
                "branch": branch(per_ctx["iid"]["R_boot_ci95"], per_ctx["chain"]["R_boot_ci95"],
                                 per_ctx["iid"]["R"], per_ctx["chain"]["R"])}

    # ------------------------------------------------------------------ checks
    def cnt(mask):
        return f"{int(mask.sum())} of {len(mask)}"

    checks = {
        "tycho_file": str(Path(tycho.__file__).resolve()),
        "n_runs": int(len(df)),
        "runs_per_arm": {f"{ck}|{arm}|{ctx}": int(len(arm_df[(ck, arm, ctx)])) for ck, arm, ctx in arm_keys},
        "seeds_per_arm": {f"{ck}|{arm}|{ctx}": arm_df[(ck, arm, ctx)].seed.astype(int).tolist()
                          for ck, arm, ctx in arm_keys},
        "sources": {f"{r.couplings}|{r.context}|{r.source}": int(r.n) for r in
                    df.groupby(["couplings", "context", "source"]).size().rename("n").reset_index().itertuples()},
        "labels": df.label.value_counts().to_dict(),
        "one_grid_for_all_runs": True, "grid_steps": steps, "n_checkpoints": len(steps),
        "grid_ratios_around_onsets": {str(steps[i]): steps[i] / steps[i - 1] for i in range(1, len(steps))
                                      if 20 <= steps[i] <= 15623},
        "subset_stored_and_ground_truth": cnt(df.subset_stored & df.subset_in_ground_truth),
        "table_events_equal_recomputed": cnt(df.events_table_match_recomputed),
        "table_matches_gt_flag_true": cnt(df.events_table_matches_gt_flag.fillna(False).astype(bool)),
        "population_coefficients": pop_checks, "closed_forms": closed_form,
        "population_coefficient_per_run_distinct": {cp: sorted(set(np.round(d.population_coefficient, 6)))
                                                     for cp, d in df.groupby("couplings")},
        "eps_at_ri_min_effect": EPS_MIN_EFFECT, "single_mode_ri(eps)": float(single_mode_ri(EPS_MIN_EFFECT)),
        "planted_sign_positive": cnt(df.planted_sign > 0),
        "final_sign_equals_planted_sign": cnt(df.sign_final == df.planted_sign),
        "final_over_population_by_arm": {f"{cp}|{ctx}": [float(d.final_over_population.min()),
                                                         float(d.final_over_population.max())]
                                         for (cp, ctx), d in df.groupby(["couplings", "context"])},
        "f_initial_range": [float(df.f_initial.min()), float(df.f_initial.max())],
        "f_initial_negative": cnt(df.f_initial < 0),
        "kl_true_final_max": float(df.kl_true_final.max()),
        "kl_true_final_by_arm_max": {f"{cp}|{ctx}": float(d.kl_true_final.max())
                                     for (cp, ctx), d in df.groupby(["couplings", "context"])},
        "n_events_of_subset": df.n_events_table.value_counts().to_dict(),
        "reg_censored": cnt(df.reg_censored), "reg_left_censored": cnt(df.reg_left_censored),
        "reg_onset_wrong_sign": cnt(df.reg_sign_matches_final == False),  # noqa: E712
        "C_censored": cnt(df.C_censored), "C_differs_from_reg": cnt(df.C_differs_from_reg),
        "C_alt_activation_only_differs_from_C": cnt(df.C_alt_activation_only_value != df.C_onset_value),
        "C_alt_window_sign_differs_from_C": cnt(df.C_alt_window_sign_value != df.C_onset_value),
        "registered_step_is_first_active_checkpoint": "events.py: step = steps[t0] (first of the two confirming "
                                                      "active checkpoints), step_lo = steps[t0-1]; confirmation at "
                                                      "steps[t0+1] (reg_confirm_step per run)",
    }
    for lv in LEVELS:
        tag = f"E{int(round(lv * 100))}"
        checks[f"{tag}_censored"] = cnt(df[f"{tag}_censored"])
        checks[f"{tag}_left_censored"] = cnt(df[f"{tag}_left_censored"])
        checks[f"{tag}_two_differs_from_single"] = cnt(df[f"{tag}_two_onset_value"] != df[f"{tag}_onset_value"])
        checks[f"{tag}_abs_differs_from_signed"] = cnt(df[f"{tag}_abs_onset_value"] != df[f"{tag}_onset_value"])
        checks[f"{tag}_dips_below_level_after_crossing"] = cnt(df[f"{tag}_dips_below_later"].fillna(False).astype(
            bool))

    # ------------------------------------------------------------------ previews
    preview = {"amendment_E": {"preview": PREVIEW_E, "per_comparison": {}}, "amendment_B": {}}
    for ck in COMPARISONS:
        pc = {}
        for dname in MAIN_DEFS:
            r_ = results[ck][dname]["per_context"]
            ri_, rc_ = r_["iid"]["R"], r_["chain"]["R"]
            pc[dname] = {"R_iid": ri_, "R_iid_ci95": r_["iid"]["R_boot_ci95"], "R_chain": rc_,
                         "R_chain_ci95": r_["chain"]["R_boot_ci95"],
                         "R_iid_gt_1_point": ri_ > 1, "R_iid_ci_above_1": r_["iid"]["R_boot_ci95"][0] > 1,
                         "R_iid_in_preview_range": PREVIEW_E["R_iid_range"][0] <= ri_ <= PREVIEW_E["R_iid_range"][1],
                         "R_chain_gt_R_iid": rc_ > ri_,
                         "R_chain_in_preview_range": PREVIEW_E["R_chain_range"][0] <= rc_ <= PREVIEW_E[
                             "R_chain_range"][1]}
        preview["amendment_E"]["per_comparison"][ck] = pc
    for ctx in CONTEXTS:
        d = arm_df[("c123", "alone", ctx)]
        preview["amendment_B"][ctx] = {
            "preview_range": PREVIEW_B[ctx], "registered_range": [int(d.reg_onset_value.min()),
                                                                  int(d.reg_onset_value.max())],
            "registered_median": float(np.median(d.reg_onset_value)),
            "within_preview_range": cnt(d.reg_onset_value.between(*PREVIEW_B[ctx])),
            "amended_C_range": [int(d.C_onset_value.min()), int(d.C_onset_value.max())]}

    # ------------------------------------------------------------------ RI vs Fourier gap (descriptive)
    gap = {}
    for ck in COMPARISONS:
        for arm in ARMS:
            for ctx in CONTEXTS:
                d = arm_df[(ck, arm, ctx)]
                gap[f"{ck}|{arm}|{ctx}"] = {
                    "couplings": COMPARISONS[ck][arm],
                    "gap_reg_minus_E25_checkpoints": d.gap_reg_minus_E25_checkpoints.astype(int).tolist(),
                    "median_gap_checkpoints": float(d.gap_reg_minus_E25_checkpoints.median()),
                    "ri_at_E25_median": float(d.ri_at_E25.median()),
                    "f_at_E25_range": [float(d.f_at_E25.min()), float(d.f_at_E25.max())],
                    "ri_at_E25_range": [float(d.ri_at_E25.min()), float(d.ri_at_E25.max())],
                    "single_mode_ri_at_E25_median": float(d.single_mode_ri_at_E25.median()),
                    "reconstructed_ri_at_E25_median": float(d.reconstructed_ri_at_E25.median()),
                    "reconstructed_ri_at_E25_range": [float(d.reconstructed_ri_at_E25.min()),
                                                      float(d.reconstructed_ri_at_E25.max())],
                    "reconstructed_vs_stored_ri_plugin_max_abs_diff_over_curve": float(
                        d["reconstructed_vs_stored_ri_max_abs_diff"].max()),
                    "stored_ri_at_max_reconstruction_diff_median": float(
                        d.stored_ri_at_max_reconstruction_diff.median()),
                    "reconstructed_vs_stored_ri_plugin_median_abs_diff_over_curve_median": float(
                        d["reconstructed_vs_stored_ri_median_abs_diff"].median()),
                    "n_reconstruction_within_0.002_of_stored_at_E25": cnt(
                        d.reconstructed_ri_at_E25_minus_stored_plugin.abs() <= 0.002),
                    "reconstructed_minus_stored_plugin_at_E25_range": [
                        float(d.reconstructed_ri_at_E25_minus_stored_plugin.min()),
                        float(d.reconstructed_ri_at_E25_minus_stored_plugin.max())],
                    "reg_ri_onset_after_E25_crossing": cnt(d.gap_reg_minus_E25_checkpoints > 0),
                    "E25_dips_below_level_later": cnt(d.E25_dips_below_later.fillna(False).astype(bool)),
                    "E10_dips_below_level_later": cnt(d.E10_dips_below_later.fillna(False).astype(bool)),
                    "f_between_E25_and_E50_crossings_range": [
                        float(min(x[0] for x in d.f_E25_to_E50_range)), float(max(x[1] for x in d.f_E25_to_E50_range))],
                    "steps_from_E25_to_E50_median": float(d.steps_E25_to_E50.median()),
                    "lower_order_f_at_E25_median": {k: float(np.median([x[k] for x in d.lower_order_f_at_E25]))
                                                    for k in d.lower_order_f_at_E25.iloc[0]},
                    "n_ri_at_E25_below_min_effect": cnt(d.ri_at_E25 < 1e-3)}

    v0 = v0_baseline(args.dr0, BOOT_SEED)

    # ------------------------------------------------------------------ chart
    chart = {"description": "Per-seed onset of the planted higher-order subset in both arms (alone, with the pairwise "
                            "{1}:1.0 term), per comparison, train context and onset definition. registered = first "
                            "activation event; amended_C = first not-left-censored (re)activation with the final "
                            "Fourier sign; E10/E25/E50 = first checkpoint where the planted-sign Fourier "
                            "coefficient reaches 10/25/50 % of its population value (E25 decides). Censored runs "
                            f"(none occurred) would carry {BUDGET} and censored=true.",
             "x": "arm (alone | with), grouped by context", "y": "onset step (log scale suggested)",
             "grid_steps": steps, "definitions": list(MAIN_DEFS) + ["E25_two"], "data": {}}
    for ck, cv in COMPARISONS.items():
        chart["data"][ck] = {"label": cv["label"], "subset": list(cv["subset"]),
                             "couplings": {"alone": cv["alone"], "with": cv["with"]}, "contexts": {}}
        for ctx in CONTEXTS:
            chart["data"][ck]["contexts"][ctx] = {}
            for dname in chart["definitions"]:
                pc = results[ck][dname]["per_context"][ctx]
                chart["data"][ck]["contexts"][ctx][dname] = {
                    arm: {"seeds": pc[arm]["seeds"], "onsets": pc[arm]["onsets"],
                          "checkpoint_idx": pc[arm]["checkpoint_idx"], "censored": pc[arm]["censored"],
                          "median": pc[arm]["median"]} for arm in ARMS}
                chart["data"][ck]["contexts"][ctx][dname]["R"] = pc["R"]
                chart["data"][ck]["contexts"][ctx][dname]["R_boot_ci95"] = pc["R_boot_ci95"]

    # ------------------------------------------------------------------ items
    def pcx(ck, dname, ctx):
        return results[ck][dname]["per_context"][ctx]

    def ci_s(ci):
        return f"[{ci[0]:.3g}, {ci[1]:.3g}]"

    def rfmt(ck, dname, ctx):
        pc = pcx(ck, dname, ctx)
        a, w = pc["alone"], pc["with"]
        return (f"R_{ctx} = {pc['R']:.3g} {ci_s(pc['R_boot_ci95'])} (with: median {w['median']:g}, range "
                f"{w['min']:g}-{w['max']:g}, n={w['n']}, censored {w['n_censored']}; alone: median {a['median']:g}, "
                f"range {a['min']:g}-{a['max']:g}, n={a['n']}, censored {a['n_censored']})")

    def ratio_fmt(ck, dname):
        co = results[ck][dname]["chain_over_iid"]
        return f"R_chain/R_iid = {co['R_chain_over_R_iid']:.3g} {ci_s(co['boot_ci95'])}"

    def dfmt(ck, dname):
        return (f"{dname}: {rfmt(ck, dname, 'chain')}; {rfmt(ck, dname, 'iid')}; {ratio_fmt(ck, dname)} -> "
                f"{results[ck][dname]['branch']['branch']}")

    verd_map = {"branch1": "supported", "branch2": "not_supported"}

    def verdict(ck, dname):
        return verd_map.get(results[ck][dname]["branch"]["branch"], "inconclusive")

    def short(ck, dname, ctx):
        pc = pcx(ck, dname, ctx)
        return f"{pc['R']:.3g} {ci_s(pc['R_boot_ci95'])}"

    def keynums(ck, defs):
        lines = []
        for dname in defs:
            for ctx in CONTEXTS:
                pc = pcx(ck, dname, ctx)
                lines.append(f"{ck} {dname} {ctx}: median with {pc['with']['median']:g} (n={pc['with']['n']}) / median "
                             f"alone {pc['alone']['median']:g} (n={pc['alone']['n']}) = R {pc['R']:.4g}, CI95 "
                             f"[{pc['R_boot_ci95'][0]:.4g}, {pc['R_boot_ci95'][1]:.4g}], boot frac R<=1 "
                             f"{pc['R_boot_frac_le_1']:.4g} (=1: {pc['R_boot_frac_eq_1']:.4g}), censored "
                             f"{pc['with']['n_censored']}+{pc['alone']['n_censored']}")
            co = results[ck][dname]["chain_over_iid"]
            lines.append(f"{ck} {dname} R_chain/R_iid {co['R_chain_over_R_iid']:.4g} CI95 [{co['boot_ci95'][0]:.4g}, "
                         f"{co['boot_ci95'][1]:.4g}] -> {results[ck][dname]['branch']['branch']}")
        return "\n".join(lines)

    n_cens_all = {d: int(df[DEFS[d][2]].sum()) for d in DEFS}
    cens_txt = (f"Censored runs (no onset by 30,000 steps): {n_cens_all['registered']} of {len(df)} registered, "
                f"{n_cens_all['amended_C']} amended C, {n_cens_all['E10']}/{n_cens_all['E25']}/{n_cens_all['E50']} "
                f"at E10/E25/E50; left-censored onsets: {int(df.reg_left_censored.sum())} registered, "
                f"{int(df.E10_left_censored.sum())}/{int(df.E25_left_censored.sum())}/{int(df.E50_left_censored.sum())} "
                f"at E10/E25/E50.")
    def_text = ("R = median onset (with {1}:1.0) / median onset (alone), per train context; 95 % percentile bootstrap "
                f"CI, {BOOT_N} resamples, seed {BOOT_SEED}, seeds resampled independently within each arm (the same "
                "resamples for every definition; R_chain/R_iid from the four arms' resamples). Onsets: registered = "
                "first activation event of the subset (events.py: step = first of the two confirming active "
                "checkpoints, q <= 0.01 and RI >= 0.001); amended C = first not-left-censored (re)activation whose "
                "Fourier coefficient at the event's checkpoint has the final sign (alternatives: first activation "
                "only; sign at both confirming checkpoints); amended E (decides at 25 %) = first single checkpoint "
                "where f * sign(pop) >= level * pop, pop = E_uniform[tanh(h) prod_A s] from the couplings by "
                "enumerating 64 windows ("
                + "; ".join(f"{k}: {v:.4f}" for k, v in pop_checks.items()) + "); "
                "levels 10/25/50 %; alternatives: two consecutive checkpoints (the events' persistence), log-step "
                "interpolated crossing (sub-grid, descriptive), |f|. Censored runs would count as 30,000. Branch "
                "rule (as registered): R_iid CI includes 1 (closed interval) and R_chain CI lower bound > 1 -> "
                "branch 1 (the delay came from correlated contexts; verdict 'supported' for the context-correlation "
                "explanation); R_iid CI lower bound > 1 -> branch 2 (lower-order structure delays higher order even "
                "with iid contexts; verdict 'not_supported'); anything else 'inconclusive'. Most plausible "
                "alternative reading of the CI condition: a sub-grid onset (interpolated crossing) so that ties on "
                "the 60-point grid do not put mass exactly at R = 1.")

    items = []

    # --- q6: the v0 delay ({1,3}:0.8)
    ck = "c13"
    frac13 = max(pcx(ck, d, "iid")["R_boot_frac_le_1"] for d in MAIN_DEFS)
    alt13 = {d: results[ck][d]["branch"]["branch"] for d in DEFS}
    n_branch2_13 = sum(b == "branch2" for b in alt13.values())
    items.append({
        "id": "q6",
        "question": "Q6 (Markov): is the v0 delay a context-correlation effect? {1,3}:0.8 alone ('1,3:0.8') vs with "
                    "the pairwise term ('1:1.0;1,3:0.8'), subset {x[t-1], x[t-3], xhat}; R = median onset with / "
                    "alone, chain and iid training contexts.",
        "definition_used": def_text,
        "literal_result": dfmt(ck, "registered") + ". " + cens_txt,
        "literal_verdict": verdict(ck, "registered"),
        "amended_result": " || ".join(dfmt(ck, d) for d in ("E25", "E10", "E50", "amended_C"))
                          + f" || all {len(DEFS)} definitions incl. alternatives: {n_branch2_13} give branch 2 "
                            f"({', '.join(sorted(k for k, v in alt13.items() if v != 'branch2')) or 'no exceptions'})",
        "amended_verdict": verdict(ck, "E25"),
        "deciding": "amended",
        "key_numbers": keynums(ck, MAIN_DEFS + ("E25_two", "E25_interp", "E25_abs")),
        "interpretation": (
            f"Registered branch 2 under the deciding definition: with independent (iid) training contexts the "
            f"pairwise term still delays the three-way {{1,3}} term, R_iid = {short(ck, 'E25', 'iid')} at E25 "
            f"(registered {short(ck, 'registered', 'iid')}; E10 {short(ck, 'E10', 'iid')}; E50 "
            f"{short(ck, 'E50', 'iid')}); the largest share of bootstrap resamples with R_iid <= 1 under any of the "
            f"five main definitions is {frac13:.3g}. So the v0 delay is not only a context-correlation effect: "
            f"lower-order structure delays the higher-order term even with independent contexts. The chain "
            f"correlation adds to it: R_chain = {short(ck, 'E25', 'chain')} at E25 ({short(ck, 'registered', 'chain')} "
            f"registered), {ratio_fmt(ck, 'E25')} at E25 ({ratio_fmt(ck, 'registered')} registered); on a log scale "
            f"the iid factor is {math.log(pcx(ck, 'E25', 'iid')['R']) / math.log(pcx(ck, 'E25', 'chain')['R']):.0%} of "
            f"the chain factor at E25 ({math.log(pcx(ck, 'registered', 'iid')['R']) / math.log(pcx(ck, 'registered', 'chain')['R']):.0%} "
            f"registered). Amendment E's effect-size correction matters: normalising the threshold by the "
            f"population coefficient moves R_chain from {pcx(ck, 'registered', 'chain')['R']:.3g} to "
            f"{pcx(ck, 'E25', 'chain')['R']:.3g} and R_iid from {pcx(ck, 'registered', 'iid')['R']:.3g} to "
            f"{pcx(ck, 'E25', 'iid')['R']:.3g} (registered -> E25), but does not remove the iid delay. Amendment C changes no onset here "
            f"({checks['C_differs_from_reg']} runs differ), so the amended C result equals the registered one."),
        "caveats": (
            f"Onsets are on the 60-point grid (ratios {min(checks['grid_ratios_around_onsets'].values()):.3g}-"
            f"{max(checks['grid_ratios_around_onsets'].values()):.3g} between checkpoints at 20-15,623 steps), so a "
            f"factor 1.5 is about two to three checkpoints; the interpolated E25 crossing gives R_iid "
            f"{short(ck, 'E25_interp', 'iid')}, R_chain {short(ck, 'E25_interp', 'chain')}. 10 seeds per arm. "
            f"The E25 and E50 R_iid ({pcx(ck, 'E25', 'iid')['R']:.3g}, {pcx(ck, 'E50', 'iid')['R']:.3g}) are "
            f"just below the pilot preview's 1.5-3.2 (see q6_preview). E thresholds differ between the arms by "
            f"design (25 % of {pop_checks['1,3:0.8 on {1,3}']:.4f} = {0.25 * pop_checks['1,3:0.8 on {1,3}']:.4f} alone vs "
            f"25 % of {pop_checks['1:1.0;1,3:0.8 on {1,3}']:.4f} = {0.25 * pop_checks['1:1.0;1,3:0.8 on {1,3}']:.4f} "
            f"with {{1}}). {cens_txt}"),
    })

    # --- q6_123
    ck = "c123"
    alt123 = {d: results[ck][d]["branch"]["branch"] for d in DEFS}
    pi = pcx(ck, "E25", "iid")
    gp = gap["c123|with|chain"]
    items.append({
        "id": "q6_123",
        "question": "Q6, the same comparison for {1,2,3}:1.0 alone ('1,2,3:1.0') vs with the pairwise term "
                    "('1:1.0;1,2,3:1.0'), subset {x[t-1], x[t-2], x[t-3], xhat}.",
        "definition_used": def_text,
        "literal_result": dfmt(ck, "registered") + ". " + cens_txt,
        "literal_verdict": verdict(ck, "registered"),
        "amended_result": " || ".join(dfmt(ck, d) for d in ("E25", "E10", "E50", "amended_C"))
                          + " || alternatives at 25 %: two consecutive -> " + alt123["E25_two"]
                          + ", |f| -> " + alt123["E25_abs"] + ", interpolated (sub-grid) -> " + alt123["E25_interp"]
                          + f" (R_iid {short(ck, 'E25_interp', 'iid')})",
        "amended_verdict": verdict(ck, "E25"),
        "deciding": "amended",
        "key_numbers": keynums(ck, MAIN_DEFS + ("E25_two", "E25_interp", "E25_abs", "E10_interp", "E50_interp")),
        "interpretation": (
            f"Boundary case. Under the deciding E25 onset R_iid = {short(ck, 'E25', 'iid')}: the CI includes 1 only at "
            f"its lower edge, because {pi['R_boot_frac_eq_1'] * 100:.3g} % of resamples put both medians on the same "
            f"checkpoint (R exactly 1) and {pi['R_boot_frac_lt_1'] * 100:.3g} % fall below 1; no iid run with the "
            f"pairwise term crosses earlier than the earliest alone run ({pi['n_with_onsets_before_min_alone']} of "
            f"{pi['with']['n']}). R_chain = {short(ck, 'E25', 'chain')} lies above 1. By the letter of the "
            f"registered rule this is branch 1 (the delay of the {{1,2,3}} term comes from correlated contexts); "
            f"the same holds at E10 ({short(ck, 'E10', 'iid')}). The registered onset ({short(ck, 'registered', 'iid')}), "
            f"amended C (identical), E50 ({short(ck, 'E50', 'iid')}) and the sub-grid interpolated E25 crossing "
            f"({short(ck, 'E25_interp', 'iid')}) put R_iid's CI above 1 (branch 2). Read together: with iid contexts "
            f"the pairwise term delays the four-way term by a small factor (R_iid "
            f"{min(pcx(ck, d, 'iid')['R'] for d in MAIN_DEFS + ('E25_interp',)):.3g}-"
            f"{max(pcx(ck, d, 'iid')['R'] for d in MAIN_DEFS + ('E25_interp',)):.3g} across definitions), "
            f"which the deciding definition cannot separate from 1 on this grid; most of the delay comes from the "
            f"chain contexts: {ratio_fmt(ck, 'E25')} at E25, {ratio_fmt(ck, 'registered')} registered, "
            f"{ratio_fmt(ck, 'E50')} at E50."),
        "caveats": (
            f"(1) In chain runs with the pairwise term the {{1,2,3}} coefficient lingers near the 25 % level "
            f"({0.25 * pop_checks['1:1.0;1,2,3:1.0 on {1,2,3}']:.3g}): a median of "
            f"{gp['steps_from_E25_to_E50_median']:g} steps pass between its 25 % and 50 % crossings (alone chain "
            f"{gap['c123|alone|chain']['steps_from_E25_to_E50_median']:g}, with iid "
            f"{gap['c123|with|iid']['steps_from_E25_to_E50_median']:g}), its values in between span "
            f"{gp['f_between_E25_and_E50_crossings_range'][0]:.3g}-{gp['f_between_E25_and_E50_crossings_range'][1]:.3g}, "
            f"and it dips back below 25 % after first crossing in {gp['E25_dips_below_level_later']} runs, so R_chain "
            f"is level-sensitive (E10 "
            f"{short(ck, 'E10', 'chain')}, E25 {short(ck, 'E25', 'chain')}, E25 two consecutive "
            f"{short(ck, 'E25_two', 'chain')}, E50 {short(ck, 'E50', 'chain')}). (2) In those runs the registered RI "
            f"onset comes after the E25 crossing in {gp['reg_ri_onset_after_E25_crossing']} runs (median "
            f"{gp['median_gap_checkpoints']:g} checkpoints later; RI at the crossing < min_effect in "
            f"{gp['n_ri_at_E25_below_min_effect']}): chain-induced lower-order coefficients (median f{{2}} = "
            f"{gp['lower_order_f_at_E25_median']['{t-2}']:.3g}, f{{3}} = {gp['lower_order_f_at_E25_median']['{t-3}']:.3g} "
            f"with f{{1}} = {gp['lower_order_f_at_E25_median']['{t-1}']:.3g}) cancel the top-order log-linear term "
            f"(see q6_ri_vs_fourier), which inflates the registered R_chain ({pcx(ck, 'registered', 'chain')['R']:.3g}). "
            f"(3) The iid alone arm has 10 seeds, the other three 12 (boundary refinement). {cens_txt}"),
    })

    # --- q6_chain_over_iid
    co_lines = []
    for ck_ in COMPARISONS:
        for d in MAIN_DEFS + ("E25_interp",):
            co = results[ck_][d]["chain_over_iid"]
            co_lines.append(f"{ck_} {d}: R_chain/R_iid {co['R_chain_over_R_iid']:.4g} CI95 [{co['boot_ci95'][0]:.4g}, "
                            f"{co['boot_ci95'][1]:.4g}], boot frac <= 1 {co['boot_frac_le_1']:.4g}")
    items.append({
        "id": "q6_chain_over_iid",
        "question": "Q6 (requested addition): how much of the delay does the context correlation add? R_chain / R_iid "
                    "with its bootstrap CI, per comparison and onset definition.",
        "definition_used": "R_chain / R_iid with R as in q6; each of the four arms resampled independently (the same "
                           "resamples as the R CIs), 2,000 resamples, percentile 95 % CI. Descriptive: no criterion "
                           "was registered for this ratio.",
        "literal_result": f"registered: c13 {ratio_fmt('c13', 'registered')}; c123 {ratio_fmt('c123', 'registered')}",
        "literal_verdict": "descriptive",
        "amended_result": "; ".join(f"{ck_} {d}: {ratio_fmt(ck_, d)}" for ck_ in COMPARISONS
                                    for d in ("E25", "E10", "E50", "amended_C")),
        "amended_verdict": "descriptive",
        "deciding": "amended",
        "key_numbers": "\n".join(co_lines),
        "interpretation": (
            f"The chain contexts multiply the delay in both comparisons, with every CI above 1 except the c123 E10 "
            f"ratio, whose CI starts at {results['c123']['E10']['chain_over_iid']['boot_ci95'][0]:.3g}: for "
            f"{{1,3}}:0.8 by {results['c13']['E25']['chain_over_iid']['R_chain_over_R_iid']:.3g} at E25 "
            f"({min(results['c13'][d]['chain_over_iid']['R_chain_over_R_iid'] for d in MAIN_DEFS):.3g}-"
            f"{max(results['c13'][d]['chain_over_iid']['R_chain_over_R_iid'] for d in MAIN_DEFS):.3g} across the five "
            f"main definitions), for {{1,2,3}}:1.0 by "
            f"{results['c123']['E25']['chain_over_iid']['R_chain_over_R_iid']:.3g} at E25, but by "
            f"{results['c123']['registered']['chain_over_iid']['R_chain_over_R_iid']:.3g} with the registered RI onset "
            f"and {results['c123']['E50']['chain_over_iid']['R_chain_over_R_iid']:.3g} at E50: in chain runs with the "
            f"pairwise term a median of {gap['c123|with|chain']['steps_from_E25_to_E50_median']:g} steps pass between "
            f"the 25 % and 50 % crossings (alone chain: {gap['c123|alone|chain']['steps_from_E25_to_E50_median']:g}; "
            f"with iid: {gap['c123|with|iid']['steps_from_E25_to_E50_median']:g})."),
        "caveats": "Ratio of ratios of medians on a discrete grid; the c123 value depends strongly on the level "
                   "(see q6_123 caveats).",
    })

    # --- q6_preview
    pv = preview["amendment_E"]["per_comparison"]

    def pv_count(ck_, key):
        return f"{sum(bool(pv[ck_][d][key]) for d in MAIN_DEFS)} of {len(MAIN_DEFS)}"

    pv_lines = []
    for ck_ in COMPARISONS:
        for d in MAIN_DEFS:
            x = pv[ck_][d]
            pv_lines.append(f"{ck_} {d}: R_iid {x['R_iid']:.3g} {ci_s(x['R_iid_ci95'])}, R_chain {x['R_chain']:.3g} "
                            f"{ci_s(x['R_chain_ci95'])}")
    pb = preview["amendment_B"]
    items.append({
        "id": "q6_preview",
        "question": "Amendment 1 E disclosed a pilot preview (R_iid > 1 under every definition, 1.5 to 3.2; R_chain "
                    "larger, 2.9 to 6.1); Amendment 1 B one for {1,2,3}:1.0 alone (onset 162-273 chain vs 96-125 "
                    "iid). Do the real values agree?",
        "definition_used": "The five main definitions (registered, amended C, E10, E25, E50). 'R_iid > 1' checked as "
                           "point estimate and as CI lower bound > 1; ranges as closed intervals. The preview does "
                           "not say which comparison; both are checked. Amendment B: registered onsets of the "
                           "'1,2,3:1.0' runs per context.",
        "literal_result": (
            f"c13: R_iid > 1 (point) {pv_count('c13', 'R_iid_gt_1_point')}, CI above 1 "
            f"{pv_count('c13', 'R_iid_ci_above_1')}, R_iid in 1.5-3.2 {pv_count('c13', 'R_iid_in_preview_range')}, "
            f"R_chain > R_iid {pv_count('c13', 'R_chain_gt_R_iid')}, R_chain in 2.9-6.1 "
            f"{pv_count('c13', 'R_chain_in_preview_range')}. c123: R_iid > 1 (point) "
            f"{pv_count('c123', 'R_iid_gt_1_point')}, CI above 1 {pv_count('c123', 'R_iid_ci_above_1')}, R_iid in "
            f"1.5-3.2 {pv_count('c123', 'R_iid_in_preview_range')}, R_chain > R_iid "
            f"{pv_count('c123', 'R_chain_gt_R_iid')}, R_chain in 2.9-6.1 {pv_count('c123', 'R_chain_in_preview_range')}. "
            f"Amendment B: '1,2,3:1.0' registered onsets chain {pb['chain']['registered_range']} "
            f"({pb['chain']['within_preview_range']} within 162-273), iid {pb['iid']['registered_range']} "
            f"({pb['iid']['within_preview_range']} within 96-125)."),
        "literal_verdict": "descriptive",
        "key_numbers": "\n".join(pv_lines),
        "interpretation": (
            f"For the v0 comparison ({{1,3}}:0.8) the preview's direction is confirmed: R_iid > 1 with the CI above 1 "
            f"in {pv_count('c13', 'R_iid_ci_above_1')} definitions and R_chain > R_iid in "
            f"{pv_count('c13', 'R_chain_gt_R_iid')}. Magnitudes: R_iid "
            f"{min(pv['c13'][d]['R_iid'] for d in MAIN_DEFS):.3g}-{max(pv['c13'][d]['R_iid'] for d in MAIN_DEFS):.3g} "
            f"(outside 1.5-3.2: {', '.join(d for d in MAIN_DEFS if not pv['c13'][d]['R_iid_in_preview_range']) or 'none'}), "
            f"R_chain {min(pv['c13'][d]['R_chain'] for d in MAIN_DEFS):.3g}-"
            f"{max(pv['c13'][d]['R_chain'] for d in MAIN_DEFS):.3g} (outside 2.9-6.1: "
            f"{', '.join(d for d in MAIN_DEFS if not pv['c13'][d]['R_chain_in_preview_range']) or 'none'}); the "
            f"deciding E25 values sit just below the previewed ranges. For {{1,2,3}}:1.0 R_iid is smaller than "
            f"previewed (point estimates {min(pv['c123'][d]['R_iid'] for d in MAIN_DEFS):.3g}-"
            f"{max(pv['c123'][d]['R_iid'] for d in MAIN_DEFS):.3g}, in 1.5-3.2: {pv_count('c123', 'R_iid_in_preview_range')}) "
            f"and its CI lies above 1 only for {', '.join(d for d in MAIN_DEFS if pv['c123'][d]['R_iid_ci_above_1'])}, "
            f"while R_chain > R_iid in {pv_count('c123', 'R_chain_gt_R_iid')}. The Amendment B preview of the "
            f"alone {{1,2,3}}:1.0 onsets is reproduced exactly ({pb['chain']['within_preview_range']} chain, "
            f"{pb['iid']['within_preview_range']} iid within the previewed ranges)."),
        "caveats": "The preview did not name the comparison or the levels it used; agreement is judged on both "
                   "comparisons and all five main definitions.",
    })

    # --- q6_onset_checks
    items.append({
        "id": "q6_onset_checks",
        "question": "Sanity checks behind Q6: subset presence, event table vs recomputed events, sign conventions, "
                    "population coefficients, left-censoring and censoring, Amendment C's effect on these onsets.",
        "definition_used": "Planted subset looked up by name in every curve and required to be a ground-truth entry; "
                           "events recomputed with detect_events(curve, alpha=0.01) and compared with ri_events "
                           "(kind, step, step_lo); planted sign = sign of the population coefficient (Markov code 1 = "
                           "spin +1, beta > 0, so +); final sign = sign of f at step 30,000; population coefficients "
                           "by exact 64-window enumeration, checked against closed forms.",
        "literal_result": (
            f"{len(df)} runs ({', '.join(f'{k}: {v}' for k, v in checks['runs_per_arm'].items())}); subset stored and "
            f"ground truth {checks['subset_stored_and_ground_truth']}; table events = recomputed "
            f"{checks['table_events_equal_recomputed']}; one activation event and no extinction per run "
            f"({checks['n_events_of_subset']}); final sign = planted sign {checks['final_sign_equals_planted_sign']}; "
            f"f(final)/pop in {min(v[0] for v in checks['final_over_population_by_arm'].values()):.3g}-"
            f"{max(v[1] for v in checks['final_over_population_by_arm'].values()):.3g}; initial f in "
            f"{checks['f_initial_range'][0]:.3g}..{checks['f_initial_range'][1]:.3g} (negative in "
            f"{checks['f_initial_negative']}); population coefficients "
            + ", ".join(f"{k} = {v:.4f}" for k, v in pop_checks.items())
            + f"; final kl_true max {checks['kl_true_final_max']:.3g} nats; registered: censored "
              f"{checks['reg_censored']}, left-censored {checks['reg_left_censored']}, wrong sign at onset "
              f"{checks['reg_onset_wrong_sign']}; amended C differs from registered {checks['C_differs_from_reg']} "
              f"(alternatives differ from C: {checks['C_alt_activation_only_differs_from_C']}, "
              f"{checks['C_alt_window_sign_differs_from_C']}); E10/E25/E50 censored {checks['E10_censored']}/"
              f"{checks['E25_censored']}/{checks['E50_censored']}, left-censored {checks['E10_left_censored']}/"
              f"{checks['E25_left_censored']}/{checks['E50_left_censored']}; two-consecutive differs from single "
              f"{checks['E10_two_differs_from_single']}/{checks['E25_two_differs_from_single']}/"
              f"{checks['E50_two_differs_from_single']}; |f| differs from signed {checks['E10_abs_differs_from_signed']}/"
              f"{checks['E25_abs_differs_from_signed']}/{checks['E50_abs_differs_from_signed']}."),
        "literal_verdict": "descriptive",
        "key_numbers": "\n".join([f"{k}: {checks[k]}" for k in (
            "n_runs", "subset_stored_and_ground_truth", "table_events_equal_recomputed",
            "final_sign_equals_planted_sign", "reg_censored", "reg_left_censored", "reg_onset_wrong_sign",
            "C_differs_from_reg", "E10_censored", "E25_censored", "E50_censored", "E25_two_differs_from_single",
            "E25_dips_below_level_after_crossing")] + [f"pop {k}: {v:.6f}" for k, v in pop_checks.items()]),
        "interpretation": "The inputs behave as designed: every curve holds the planted subset, the event table "
                          "matches a recomputation, no onset is left-censored or censored, and the final Fourier "
                          "coefficients carry the planted (+) sign and reach about their population values. "
                          "Amendment C's correction (initial wrong-sign or left-censored detections) does not apply "
                          "to these order-3 and order-4 subsets, so C equals the registered onset in every run. The "
                          f"required population values are reproduced ({pop_checks['1,3:0.8 on {1,3}']:.4f}, "
                          f"{pop_checks['1:1.0;1,3:0.8 on {1,3}']:.4f}; closed forms tanh(0.8) and "
                          f"(tanh(1.8) - tanh(0.2))/2).",
        "caveats": "The event 'step' is the first of the two consecutive active checkpoints (the onset, in "
                   "(step_lo, step]); the state machine confirms it one checkpoint later (reg_confirm_step in "
                   "per_run). E crossings use the same convention (first checkpoint at or above the level).",
    })

    # --- q6_ri_vs_fourier
    g_lines = []
    for k_, v_ in gap.items():
        g_lines.append(f"{k_}: reg - E25 (checkpoints) median {v_['median_gap_checkpoints']:g}; RI at E25 median "
                       f"{v_['ri_at_E25_median']:.3g} (single-mode {v_['single_mode_ri_at_E25_median']:.3g}, "
                       f"reconstructed {v_['reconstructed_ri_at_E25_median']:.3g}); RI < min_effect at E25 "
                       f"{v_['n_ri_at_E25_below_min_effect']}")
    items.append({
        "id": "q6_ri_vs_fourier",
        "question": "Descriptive: why does the registered RI onset diverge from the Fourier (E) onset, most strongly "
                    "for {1,2,3} with the pairwise term in chain contexts?",
        "definition_used": "Per run, the checkpoint gap registered onset - E25 crossing; the stored RI, the single-mode "
                           "RI of f(A), and an RI rebuilt from the stored Walsh coefficients f(B) of every nonempty B "
                           "in A (uniform probes: E[2p-1 | x_A] = sum_B f(B) chi_B; f(empty) not stored, set to 0), "
                           "all at the E25 crossing; the lower-order coefficients there.",
        "literal_result": "; ".join(g_lines),
        "literal_verdict": "descriptive",
        "key_numbers": "\n".join(g_lines),
        "interpretation": (
            f"In every other arm the RI activation precedes the E25 crossing, by "
            f"{-max(max(v_['gap_reg_minus_E25_checkpoints']) for k_, v_ in gap.items() if k_ != 'c123|with|chain')} to "
            f"{-min(min(v_['gap_reg_minus_E25_checkpoints']) for k_, v_ in gap.items() if k_ != 'c123|with|chain')} "
            f"checkpoints (RI's detection threshold corresponds to a lower coefficient). In "
            f"chain runs of '1:1.0;1,2,3:1.0' it follows it ({gp['reg_ri_onset_after_E25_crossing']} runs, median "
            f"{gp['median_gap_checkpoints']:g} checkpoints later): at the crossing f{{1,2,3}} is "
            f"{gp['f_at_E25_range'][0]:.3g}-{gp['f_at_E25_range'][1]:.3g}, whose "
            f"single-mode RI would be {gp['single_mode_ri_at_E25_median']:.3g} nats (median), but the stored RI is "
            f"{gp['ri_at_E25_median']:.3g} (median; below min_effect in {gp['n_ri_at_E25_below_min_effect']}). The "
            f"RI rebuilt from all stored coefficients (median {gp['reconstructed_ri_at_E25_median']:.3g}; within 0.002 "
            f"nats of the stored value in {gp['n_reconstruction_within_0.002_of_stored_at_E25']}) "
            f"reproduces the near-zero value from the chain-induced lower-order coefficients (medians f{{2}} = "
            f"{gp['lower_order_f_at_E25_median']['{t-2}']:.3g}, f{{3}} = {gp['lower_order_f_at_E25_median']['{t-3}']:.3g} "
            f"next to f{{1}} = {gp['lower_order_f_at_E25_median']['{t-1}']:.3g}), which cancel the top-order "
            f"log-linear term. So this is "
            f"the instrument measuring a different (log-domain) quantity, not a bug, and it is why the registered "
            f"R_chain for {{1,2,3}} ({pcx('c123', 'registered', 'chain')['R']:.3g}) is far above the E25 value "
            f"({pcx('c123', 'E25', 'chain')['R']:.3g}). RI onset after the E25 crossing in the other arms: "
            + ", ".join(f"{k_} {v_['reg_ri_onset_after_E25_crossing']}" for k_, v_ in gap.items()
                        if k_ != 'c123|with|chain') + "."),
        "caveats": f"The rebuild is approximate (f(empty) unknown, the probe's x_A marginal only nearly uniform) and "
                   f"sensitive near cancellation: in chain runs with the pairwise term its difference from the stored "
                   f"plug-in RI at the E25 crossing ranges "
                   f"{gp['reconstructed_minus_stored_plugin_at_E25_range'][0]:.3g} to "
                   f"{gp['reconstructed_minus_stored_plugin_at_E25_range'][1]:.3g} nats; its largest error over all "
                   f"checkpoints is {gp['reconstructed_vs_stored_ri_plugin_max_abs_diff_over_curve']:.3g} nats, "
                   f"reached where the stored RI is already large (median "
                   f"{gp['stored_ri_at_max_reconstruction_diff_median']:.3g} nats there).",
    })

    # --- q6_v0_baseline
    items.append({
        "id": "q6_v0_baseline",
        "question": "Descriptive: the v0 delay R (DR0, chain contexts) next to v1's chain result.",
        "definition_used": "DR0 ri_events, first activation of {x[t-1], x[t-3], xhat} in the 7 '1,3:0.8' and 7 "
                           "'1:1.0;1,3:0.8' control-markov runs (v0: chain contexts only, 15,000 steps, 30 "
                           "checkpoints); same bootstrap as v1.",
        "literal_result": (
            f"v0: alone {v0['arms']['alone']['onsets']} (median {v0['arms']['alone']['median']:g}), with "
            f"{v0['arms']['with']['onsets']} (median {v0['arms']['with']['median']:g}); R_v0 = {v0['R']:.3g} "
            f"{ci_s(v0['R_boot_ci95'])}; ranges match v0_findings ({v0['ranges_agree_with_v0_findings']}). v1 chain "
            f"registered R = {short('c13', 'registered', 'chain')}."),
        "literal_verdict": "descriptive",
        "key_numbers": f"R_v0 {v0['R']:.4g} CI95 {ci_s(v0['R_boot_ci95'])}\nR_v1_chain_registered "
                       f"{pcx('c13', 'registered', 'chain')['R']:.4g}\nR_v1_iid_registered "
                       f"{pcx('c13', 'registered', 'iid')['R']:.4g}",
        "interpretation": (
            f"v1's chain runs reproduce the v0 delay (R_v0 {v0['R']:.3g} {ci_s(v0['R_boot_ci95'])} vs v1 chain "
            f"{short('c13', 'registered', 'chain')}). With independent contexts the registered R is "
            f"{pcx('c13', 'registered', 'iid')['R']:.3g}, and {pcx('c13', 'E25', 'iid')['R']:.3g} with the "
            f"effect-size-normalised E25 onset (chain: {pcx('c13', 'E25', 'chain')['R']:.3g}): part of the v0 delay "
            f"was the context correlation, part the smaller target coefficient, and a factor of "
            f"{pcx('c13', 'E25', 'iid')['R']:.3g} remains with iid contexts (q6)."),
        "caveats": "v0 differs in budget (15k steps), grid (30 points) and seeds (7); a qualitative cross-check only.",
    })

    payload = {
        "question": "Pre-registered question 6 (Markov: is the v0 delay a context-correlation effect?), incl. "
                    "Amendment 1 C and E",
        "preregistration": "docs/results/v1_preregistration.md",
        "script": "scripts/analysis_v1/markov_context.py",
        "release": str(args.release), "extracted": str(args.extracted),
        "budget_steps": BUDGET, "alpha": ALPHA,
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "unit": "seeds within each arm (couplings x context), "
                                                              "independently per arm; percentile 95 % CI"},
        "comparisons": {ck_: {"label": cv["label"], "alone": cv["alone"], "with": cv["with"],
                              "subset": list(cv["subset"])} for ck_, cv in COMPARISONS.items()},
        "definitions": {k: {"value_col": v[0], "role": v[3]} for k, v in DEFS.items()},
        "items": items,
        "checks": checks, "results": results, "previews": preview, "ri_vs_fourier_gap": gap,
        "v0_baseline": v0, "chart": chart,
        "per_run": [{k: v for k, v in r.items() if k != "grid"} for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(payload), indent=1))
    print(f"wrote {args.out}")
    for it in items:
        print(f"[{it['id']}] literal={it['literal_verdict']} amended={it.get('amended_verdict')}")


if __name__ == "__main__":
    main()
