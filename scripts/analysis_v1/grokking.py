#!/usr/bin/env python
"""Sky v1 analysis: pre-registered questions 7 and 8 (grokking on modular addition), with Amendment 1 items F and G.

Pre-registration: docs/results/v1_preregistration.md. Output: docs/results/v1_analysis/grokking.json.

Usage (from the repository root):
    PYTHONPATH=. /home/user/tycho/.venv/bin/python scripts/analysis_v1/grokking.py \
        [--dr1-extracted DIR] [--scratch DIR] [--out JSON]

Inputs (read-only): releases/DR1/{objects,ri_runs,ri_events}.parquet, the extracted DR1 RI curves and light curves
(``--dr1-extracted``, paths in the tables are relative to it), releases/DR0/{objects,ri_runs,ri_events}.parquet and
the DR0 light curves of the 145 v0 objects on the v1 grid points, which this script extracts from
releases/DR0/lightcurves.tar.gz into ``--scratch`` (only those members).

Definitions used (see the JSON "items" for the full statement of each):
* Q7 per-run measure ("strong late activation"): an event of kind activation or reactivation with
  peak_ri >= 0.02 nats whose onset bracket lies after the training set was fitted, step_lo >= t_train_acc99
  (v0's late_activation definition). Runs that never reach 99 % train accuracy have none. Alternative (v0's exact
  code, scripts/analyze_dr0.py and tycho.analysis.events.find_transients): activations only.
* Bootstrap: 2,000 resamples, numpy default_rng(SEED), percentile 95 % CI; the resampled unit is the configuration
  (same-configuration comparison, descriptive rates), the configuration pair (neighbouring comparison) or the grid
  point (Q8).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tycho  # noqa: E402
from tycho.analysis.events import detect_events  # noqa: E402
from tycho.analysis.ricurve import RICurve  # noqa: E402
from tycho.train.spec import TrainConfig  # noqa: E402

SCRATCH_ROOT = REPO / "work"  # extracted releases and scratch files (git-ignored); see README.md
PEAK = 0.02  # nats: "strong" activation (pre-registered)
ALPHA = 0.01
BUDGET = 50_000  # v1 step budget of grokking-v1 (censoring value)
V0_BUDGET = 20_000
N_BOOT = 2000
SEED = 20261112
REGION = "grokking-v1"
KEY = ["c_p", "c_train_frac", "c_width", "train_lr", "train_weight_decay"]
KEY_NAMES = ["p", "train_frac", "width", "lr", "weight_decay"]
COARSE_TF = (0.3, 0.4, 0.5)
COARSE_WD = (0.1, 0.3, 1.0)
GRID = dict(c_p=[47], c_train_frac=list(COARSE_TF), c_width=[64, 256], train_lr=[0.001, 0.003],
            train_weight_decay=list(COARSE_WD))


# ------------------------------------------------------------------------------------------------ helpers
def jsonable(x):
    """Recursively convert numpy / pandas values to JSON types (NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NA or x is None:
        return None
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def records(df: pd.DataFrame) -> list[dict]:
    return [jsonable(r) for r in df.to_dict(orient="records")]


def with_cfg(o: pd.DataFrame) -> pd.DataFrame:
    cfg = pd.json_normalize(o["task_cfg_json"].map(json.loads))
    cfg.index = o.index
    return pd.concat([o, cfg.add_prefix("c_")], axis=1)


def cfg_str(row) -> str:
    return (f"tf={row['c_train_frac']:g} w={int(row['c_width'])} lr={row['train_lr']:g} "
            f"wd={row['train_weight_decay']:g}")


def pct(k, n) -> float:
    return float("nan") if n == 0 else 100.0 * k / n


def ci(a) -> list[float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def cluster_boot_diff(kg, ng, kc, nc, rng, n_boot=N_BOOT):
    """Bootstrap over clusters (configurations or pairs): pooled frac(g) - pooled frac(c), and the mean over clusters
    of the per-cluster difference. Returns dict with point estimates and percentile CIs."""
    kg, ng, kc, nc = (np.asarray(v, dtype=float) for v in (kg, ng, kc, nc))
    C = kg.size
    out = {"n_clusters": int(C)}
    pooled = kg.sum() / ng.sum() - kc.sum() / nc.sum()
    per = kg / ng - kc / nc
    out["pooled_frac_g"] = kg.sum() / ng.sum()
    out["pooled_frac_c"] = kc.sum() / nc.sum()
    out["pooled_diff"] = pooled
    out["mean_cluster_diff"] = float(per.mean())
    if C < 2:
        out["pooled_diff_ci95"] = [float("nan"), float("nan")]
        out["mean_cluster_diff_ci95"] = [float("nan"), float("nan")]
        out["ci_note"] = "fewer than 2 clusters: no bootstrap CI"
        return out
    idx = rng.integers(0, C, size=(n_boot, C))
    with np.errstate(invalid="ignore", divide="ignore"):
        bp = kg[idx].sum(1) / ng[idx].sum(1) - kc[idx].sum(1) / nc[idx].sum(1)
    bm = per[idx].mean(1)
    out["pooled_diff_ci95"] = ci(bp)
    out["mean_cluster_diff_ci95"] = ci(bm)
    return out


def verdict_from_ci(lo, hi, n_clusters, few=10):
    """Amendment 1 F.3 rule: CI above 0 -> supported; CI below 0 -> not_supported; CI including 0 (or no CI) with
    few clusters -> inconclusive."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "inconclusive"
    if lo > 0:
        return "supported"
    if hi < 0:
        return "not_supported"
    return "inconclusive" if n_clusters < few else "not_supported"


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dr1", default=str(REPO / "releases/DR1"))
    ap.add_argument("--dr0", default=str(REPO / "releases/DR0"))
    ap.add_argument("--dr1-extracted", default=str(SCRATCH_ROOT / "dr1"))
    ap.add_argument("--scratch", default=str(SCRATCH_ROOT / "grokking"))
    ap.add_argument("--out", default=str(REPO / "docs/results/v1_analysis/grokking.json"))
    args = ap.parse_args()
    DR1, DR0, X1, SCR = Path(args.dr1), Path(args.dr0), Path(args.dr1_extracted), Path(args.scratch)
    SCR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    checks: dict = {"tycho_file": tycho.__file__,
                    "tycho_under_repo": str(Path(tycho.__file__).resolve()).startswith(str(REPO))}
    assert checks["tycho_under_repo"], tycho.__file__

    # ------------------------------------------------------------------ DR1 tables
    o_all = pd.read_parquet(DR1 / "objects.parquet")
    runs_all = pd.read_parquet(DR1 / "ri_runs.parquet")
    ev_all = pd.read_parquet(DR1 / "ri_events.parquet")
    g = with_cfg(o_all[o_all.region == REGION].copy()).reset_index(drop=True)
    g["cfg"] = g.apply(cfg_str, axis=1)
    g["cfg_key"] = list(zip(*[g[k] for k in KEY]))
    runs = runs_all[runs_all.object_id.isin(g.object_id)]
    checks["n_objects"] = int(len(g))
    checks["labels"] = g.label.value_counts().to_dict()
    checks["n_configurations"] = int(g.cfg_key.nunique())
    checks["followup_runs"] = int((runs.kind == "followup").sum())
    checks["objects_with_followup"] = int(runs[runs.kind == "followup"].object_id.nunique())
    checks["objects_with_more_than_one_ri_run"] = int((runs.groupby("object_id").size() > 1).sum())
    checks["ri_run_kinds"] = runs.kind.value_counts().to_dict()
    assert checks["objects_with_followup"] == len(g)
    run_of = runs[runs.kind == "followup"].set_index("object_id")
    g["ri_run_id"] = g.object_id.map(run_of.ri_run_id)
    g["ricurve_file"] = g.object_id.map(run_of.ricurve_file)
    g["n_points"] = g.object_id.map(run_of.n_points)
    ev = ev_all[ev_all.ri_run_id.isin(g.ri_run_id)].merge(
        g[["object_id", "label", "t_train_acc99", "t_test_acc90", "cfg"]], on="object_id", how="left")
    checks["n_events"] = int(len(ev))
    checks["events_by_kind"] = ev.kind.value_counts().to_dict()
    checks["events_with_correct_column"] = int(ev.subset_json.str.contains('"correct"').sum())
    checks["t_train_acc99_missing_by_label"] = g.groupby("label").t_train_acc99.apply(lambda s: int(s.isna().sum())).to_dict()
    checks["t_test_acc90_reached_by_label"] = g.groupby("label").t_test_acc90.apply(lambda s: int(s.notna().sum())).to_dict()
    checks["n_points_by_label"] = {k: v.value_counts().to_dict() for k, v in g.groupby("label").n_points}
    checks["objects_with_sharp_drop (dense window)"] = int(g.largest_drop_lo.notna().sum())

    # ------------------------------------------------------------------ RI curves: fidelity, cross-check of events
    meta_rows = []
    mism = []
    n_recomputed_strong_late = {}
    for row in g.itertuples():
        c = RICurve.load(X1 / row.ricurve_file)
        m = c.meta
        steps = [int(s) for s in c.steps]
        rec = detect_events(c, alpha=ALPHA)
        rec_set = {(frozenset(e.subset_names), e.kind, e.step, e.step_lo, round(e.peak_ri, 9)) for e in rec}
        tab = ev[ev.object_id == row.object_id]
        tab_set = {(frozenset(json.loads(s)), k, int(st), int(lo), round(float(p), 9))
                   for s, k, st, lo, p in zip(tab.subset_json, tab.kind, tab.step, tab.step_lo, tab.peak_ri)}
        # step_lo must be the checkpoint just before step (or = step for a left-censored first-checkpoint onset)
        pos = {s: i for i, s in enumerate(steps)}
        bad_bracket = int(sum(1 for st, lo in zip(tab.step, tab.step_lo)
                              if not (int(st) in pos and int(lo) in pos and
                                      (pos[int(st)] - pos[int(lo)] == 1 or (int(st) == int(lo) == steps[0])))))
        t99 = row.t_train_acc99
        n_rec = sum(1 for e in rec if e.kind != "extinction" and e.peak_ri >= PEAK and math.isfinite(t99)
                    and e.step_lo >= t99)
        n_recomputed_strong_late[row.object_id] = n_rec
        if rec_set != tab_set or bad_bracket or len(rec) != len(tab):
            mism.append({"object_id": row.object_id, "only_recomputed": len(rec_set - tab_set),
                         "only_table": len(tab_set - rec_set), "bad_bracket": bad_bracket,
                         "n_recomputed": len(rec), "n_table": len(tab)})
        meta_rows.append({"object_id": row.object_id, "replay_fidelity": m.get("replay_fidelity"),
                          "ri_kind": m.get("kind"), "min_effect": m.get("min_effect"),
                          "min_effect_n": m.get("min_effect_n"), "n_candidates": m.get("n_candidates"),
                          "n_stored_subsets": c.S, "T": c.T, "first_step": steps[0], "last_step": steps[-1],
                          "dense": m.get("dense"), "followup_reason": m.get("followup_reason"),
                          "n_ckpt_ge_t99": int(sum(1 for st in steps if math.isfinite(t99) and st >= t99)),
                          "names": ",".join(c.names)})
    meta = pd.DataFrame(meta_rows)
    g = g.merge(meta, on="object_id", how="left")
    fid = pd.to_numeric(g.replay_fidelity, errors="coerce")
    checks["replay_fidelity"] = {"n_zero": int((fid == 0).sum()), "n_missing": int(fid.isna().sum()),
                                 "max": float(fid.max()) if fid.notna().any() else None}
    checks["curve_meta"] = {"ri_kind": g.ri_kind.value_counts().to_dict(),
                            "min_effect": g.min_effect.value_counts().to_dict(),
                            "min_effect_n": g.min_effect_n.value_counts().to_dict(),
                            "n_candidates": g.n_candidates.value_counts().to_dict(),
                            "T": g["T"].value_counts().to_dict(),
                            "first_step": g.first_step.value_counts().to_dict(),
                            "last_step": g.last_step.value_counts().to_dict(),
                            "n_variables_names_example": g.names.iloc[0]}
    checks["event_table_vs_detect_events"] = {"objects_compared": int(len(g)), "objects_mismatching": len(mism),
                                              "mismatches": mism[:20]}
    reason = g.followup_reason.fillna("").str.split(":").str[0]
    checks["followup_reason_stream"] = reason.value_counts().to_dict()
    checks["failed_followups"] = ("none: all 689 grokking-v1 objects have a follow-up RI curve in DR1 (DR1 holds no "
                                  "follow-up request table, so failed attempts that were later retried are not visible)")

    # ------------------------------------------------------------------ per-run Q7 measures
    ev["strong"] = ev.kind.isin(["activation", "reactivation"]) & (ev.peak_ri >= PEAK)
    ev["late"] = ev.strong & ev.t_train_acc99.notna() & (ev.step_lo >= ev.t_train_acc99)
    ev["late_act"] = ev.late & (ev.kind == "activation")
    t90c = ev.t_test_acc90.fillna(BUDGET)  # never generalised: censored at the budget
    ev["late_before_gen"] = ev.late & (ev.step_lo < t90c)
    ev["late_act_before_gen"] = ev.late_act & (ev.step_lo < t90c)
    ev["left_censored"] = ev.ri_before.isna() & (ev.kind == "activation")
    checks["left_censored_activations"] = int(ev.left_censored.sum())
    checks["left_censored_strong_activations"] = int((ev.left_censored & ev.strong).sum())
    checks["strong_events_straddling_t99 (step_lo < t99 <= step)"] = int(
        (ev.strong & (ev.step_lo < ev.t_train_acc99) & (ev.step >= ev.t_train_acc99)).sum())
    cnt = lambda col: g.object_id.map(ev[ev[col]].groupby("object_id").size()).fillna(0).astype(int)  # noqa: E731
    g["n_strong_any"] = cnt("strong")
    g["n_late"] = cnt("late")
    g["n_late_act"] = cnt("late_act")
    g["n_late_before_gen"] = cnt("late_before_gen")
    g["n_late_act_before_gen"] = cnt("late_act_before_gen")
    g["n_events"] = g.object_id.map(ev.groupby("object_id").size()).fillna(0).astype(int)
    g["n_activations_all"] = g.object_id.map(ev[ev.kind == "activation"].groupby("object_id").size()).fillna(0).astype(int)
    for col in ["late", "late_act", "late_before_gen", "late_act_before_gen", "strong_any"]:
        g["m_" + col] = g["n_" + col] > 0
    first_late = ev[ev.late].sort_values("step").groupby("object_id").first()
    g["first_late_step_lo"] = g.object_id.map(first_late.step_lo)
    g["first_late_step"] = g.object_id.map(first_late.step)
    g["first_late_peak"] = g.object_id.map(first_late.peak_ri)
    g["max_peak_late"] = g.object_id.map(ev[ev.late].groupby("object_id").peak_ri.max())
    g["recomputed_n_late"] = g.object_id.map(n_recomputed_strong_late)
    checks["n_late_table_equals_recomputed"] = int((g.recomputed_n_late == g.n_late).sum())
    checks["never_reached_train99"] = {"total": int(g.t_train_acc99.isna().sum()),
                                       "by_label": g.groupby("label").t_train_acc99.apply(
                                           lambda s: int(s.isna().sum())).to_dict()}

    # ------------------------------------------------------------------ descriptive by label (whole region)
    def label_rates(col):
        out = {}
        for lab, sub in g.groupby("label"):
            per = sub.groupby("cfg_key")[col].agg(["sum", "size"])
            k, n = int(per["sum"].sum()), int(per["size"].sum())
            idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
            bs = per["sum"].values[idx].sum(1) / per["size"].values[idx].sum(1)
            out[lab] = {"k": k, "n": n, "pct": pct(k, n), "ci95_pct_resampling_configs": [100 * v for v in ci(bs)],
                        "n_configs": int(len(per))}
        return out

    by_label = {
        "strong_late (act+react; primary)": label_rates("m_late"),
        "strong_late activation-only (v0 code)": label_rates("m_late_act"),
        "strong_late_before_generalisation (t90 censored at 50k)": label_rates("m_late_before_gen"),
        "any strong (>= 0.02) activation or reactivation, any time": label_rates("m_strong_any"),
    }
    n_ev_by_label = {lab: {"mean": float(s.mean()), "median": float(s.median()),
                           "q25": float(s.quantile(.25)), "q75": float(s.quantile(.75)), "max": int(s.max()),
                           "sum": int(s.sum()), "n_runs": int(s.size)}
                     for lab, s in g.groupby("label").n_late}
    n_evact_by_label = {lab: {"mean": float(s.mean()), "median": float(s.median()), "sum": int(s.sum())}
                        for lab, s in g.groupby("label").n_late_act}
    # whole-region label differences (confounded by configuration), configurations resampled
    cfg_lab = g.groupby(["cfg_key", "label"]).m_late.agg(["sum", "size"]).unstack(fill_value=0)

    def region_diff(a, b, col="m_late"):
        t = g.groupby(["cfg_key", "label"])[col].agg(["sum", "size"]).unstack(fill_value=0)
        ka, na = t[("sum", a)].values.astype(float), t[("size", a)].values.astype(float)
        kb, nb = t[("sum", b)].values.astype(float), t[("size", b)].values.astype(float)
        idx = rng.integers(0, len(t), size=(N_BOOT, len(t)))
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = ka[idx].sum(1) / na[idx].sum(1) - kb[idx].sum(1) / nb[idx].sum(1)
        return {"diff_pct": 100 * (ka.sum() / na.sum() - kb.sum() / nb.sum()),
                "ci95_pct_resampling_all_101_configs": [100 * v for v in ci(bs)]}

    region_diffs = {f"{a} - {b}": region_diff(a, b) for a, b in
                    [("grokking", "memorization"), ("grokking", "smooth"), ("smooth", "memorization")]}
    del cfg_lab
    # descriptive context: the measure by final test accuracy, order of the late events, late checkpoints available
    acc_bins = [0.0, 0.05, 0.22, 0.5, 0.8, 0.9, 1.0001]
    bin_labels = ["[0, 0.05)", "[0.05, 0.22)", "[0.22, 0.5)", "[0.5, 0.8)", "[0.8, 0.9)", "[0.9, 1]"]
    g["final_test_acc_bin"] = pd.cut(g.final_test_acc, acc_bins, right=False, labels=bin_labels).astype(str)
    by_acc = g.groupby(["final_test_acc_bin", "label"]).m_late.agg(["sum", "size"]).reset_index()
    by_acc_all = g.groupby("final_test_acc_bin").m_late.agg(["sum", "size"]).reset_index()
    late_order = {f"{lab} order {o}": int(n) for (lab, o), n in ev[ev.late].groupby(["label", "order"]).size().items()}
    late_kind = {f"{lab} {k}": int(n) for (lab, k), n in ev[ev.late].groupby(["label", "kind"]).size().items()}
    ckpt_after = {lab: {"median_ckpt_ge_t99": float(s.n_ckpt_ge_t99.median()),
                        "median_t_train_acc99": float(s.t_train_acc99.median()),
                        "min_t_train_acc99": float(s.t_train_acc99.min()),
                        "max_t_train_acc99": float(s.t_train_acc99.max())} for lab, s in g.groupby("label")}

    # ------------------------------------------------------------------ F.1 / literal Q7: same configuration
    labs = g.groupby("cfg_key").label.agg(lambda s: sorted(set(s)))
    mixed_keys = [k for k, v in labs.items() if "grokking" in v and len(v) > 1]
    disagree_keys = [k for k, v in labs.items() if len(v) > 1]
    sc = g[g.cfg_key.isin(mixed_keys)].copy()
    sc["group"] = np.where(sc.label == "grokking", "grokking", "control")
    cfg_rows = []
    for k in mixed_keys:
        s = sc[sc.cfg_key == k]
        gr, ct = s[s.group == "grokking"], s[s.group == "control"]
        cfg_rows.append({"cfg": s.cfg.iloc[0], **dict(zip(KEY_NAMES, k)),
                         "n_grokking": len(gr), "k_grokking": int(gr.m_late.sum()),
                         "n_control": len(ct), "k_control": int(ct.m_late.sum()),
                         "k_grokking_act_only": int(gr.m_late_act.sum()), "k_control_act_only": int(ct.m_late_act.sum()),
                         "k_grokking_before_gen": int(gr.m_late_before_gen.sum()),
                         "k_control_before_gen": int(ct.m_late_before_gen.sum()),
                         "sum_events_grokking": int(gr.n_late.sum()), "sum_events_control": int(ct.n_late.sum()),
                         "control_labels": ct.label.value_counts().to_dict(),
                         "grokking_seeds": sorted(gr.seed.tolist()), "control_seeds": sorted(ct.seed.tolist())})
    cfg_tab = pd.DataFrame(cfg_rows)
    controls = sc[sc.group == "control"][["object_id", "cfg", "seed", "label", "final_test_acc", "max_test_acc",
                                          "final_train_acc", "t_train_acc99", "n_late", "m_late", "m_late_act",
                                          "first_late_step_lo", "max_peak_late"]].sort_values(["cfg", "seed"])
    grok_sc = sc[sc.group == "grokking"][["object_id", "cfg", "seed", "label", "final_test_acc", "t_train_acc99",
                                          "t_test_acc90", "n_late", "m_late", "m_late_act", "m_late_before_gen",
                                          "first_late_step_lo", "max_peak_late"]].sort_values(["cfg", "seed"])

    def sc_boot(kcol_g, kcol_c):
        return cluster_boot_diff(cfg_tab[kcol_g], cfg_tab.n_grokking, cfg_tab[kcol_c], cfg_tab.n_control, rng)

    same_cfg = {
        "primary (act+react)": sc_boot("k_grokking", "k_control"),
        "alternative activation-only (v0 code)": sc_boot("k_grokking_act_only", "k_control_act_only"),
        "late and before generalisation (t90 censored at 50k)": sc_boot("k_grokking_before_gen", "k_control_before_gen"),
    }
    # (a) number of such events per run
    same_cfg_nev = cluster_boot_diff(cfg_tab.sum_events_grokking, cfg_tab.n_grokking, cfg_tab.sum_events_control,
                                     cfg_tab.n_control, rng)
    same_cfg_nev = {"mean_events_per_run_grokking": same_cfg_nev["pooled_frac_g"],
                    "mean_events_per_run_control": same_cfg_nev["pooled_frac_c"],
                    "diff": same_cfg_nev["pooled_diff"], "diff_ci95": same_cfg_nev["pooled_diff_ci95"],
                    "median_grokking": float(sc[sc.group == "grokking"].n_late.median()),
                    "median_control": float(sc[sc.group == "control"].n_late.median())}
    prim = same_cfg["primary (act+react)"]
    alt = same_cfg["alternative activation-only (v0 code)"]
    lo, hi = prim["pooled_diff_ci95"]
    literal_v = "supported" if lo > 0 else "not_supported"
    amended_v = verdict_from_ci(lo, hi, prim["n_clusters"])

    # ------------------------------------------------------------------ F.2: neighbouring-point comparison
    pts = g.groupby("cfg_key").label.value_counts().unstack(fill_value=0)
    for lab in ("grokking", "memorization", "smooth"):
        if lab not in pts:
            pts[lab] = 0
    pts = pts.reset_index()
    pts[KEY] = pd.DataFrame(pts.cfg_key.tolist(), index=pts.index)
    tf_global = sorted(g.c_train_frac.unique())
    wd_global = sorted(g.train_weight_decay.unique())
    pidx = {k: i for i, k in enumerate(pts.cfg_key)}

    def neighbours(key, mode):
        p, tf, w, lr, wd = key
        out = []
        for knob, vals_global, pos in (("train_frac", tf_global, 1), ("weight_decay", wd_global, 4)):
            if mode == "global":
                i = vals_global.index(key[pos])
                cand = [vals_global[j] for j in (i - 1, i + 1) if 0 <= j < len(vals_global)]
            elif mode == "line":
                line = sorted(v[pos] for v in pidx if all(v[j] == key[j] for j in range(5) if j != pos))
                i = line.index(key[pos])
                cand = [line[j] for j in (i - 1, i + 1) if 0 <= j < len(line)]
            elif mode == "coarse":
                grid = COARSE_TF if pos == 1 else COARSE_WD
                if key[1] not in COARSE_TF or key[4] not in COARSE_WD:
                    return []
                i = grid.index(key[pos])
                cand = [grid[j] for j in (i - 1, i + 1) if 0 <= j < len(grid)]
            for v in cand:
                nk = tuple(v if j == pos else key[j] for j in range(5))
                if nk in pidx:
                    out.append((knob, nk))
        return out

    def neighbour_analysis(mode):
        pairs = []
        for k in pts.cfg_key[pts.grokking > 0]:
            for knob, nk in neighbours(k, mode):
                if pts.loc[pidx[nk], "memorization"] > 0:
                    gr = g[(g.cfg_key == k) & (g.label == "grokking")]
                    me = g[(g.cfg_key == nk) & (g.label == "memorization")]
                    pairs.append({"grok_point": gr.cfg.iloc[0], "mem_point": me.cfg.iloc[0], "knob": knob,
                                  "grok_point_labels": g[g.cfg_key == k].label.value_counts().to_dict(),
                                  "mem_point_labels": g[g.cfg_key == nk].label.value_counts().to_dict(),
                                  "n_grokking": len(gr), "k_grokking": int(gr.m_late.sum()),
                                  "n_memorization": len(me), "k_memorization": int(me.m_late.sum()),
                                  "k_grokking_act_only": int(gr.m_late_act.sum()),
                                  "k_memorization_act_only": int(me.m_late_act.sum()),
                                  "mem_final_test_acc": [round(float(x), 4) for x in me.final_test_acc],
                                  "grok_object_ids": gr.object_id.tolist(), "mem_object_ids": me.object_id.tolist()})
        res = {"n_pairs": len(pairs), "pairs": pairs}
        if pairs:
            pt = pd.DataFrame(pairs)
            res["n_grok_points"] = int(pt.grok_point.nunique())
            res["n_mem_points"] = int(pt.mem_point.nunique())
            res["unique_grokking_runs"] = len({x for ids in pt.grok_object_ids for x in ids})
            res["unique_memorization_runs"] = len({x for ids in pt.mem_object_ids for x in ids})
            res["primary (act+react)"] = cluster_boot_diff(pt.k_grokking, pt.n_grokking, pt.k_memorization,
                                                           pt.n_memorization, rng)
            res["alternative activation-only"] = cluster_boot_diff(pt.k_grokking_act_only, pt.n_grokking,
                                                                   pt.k_memorization_act_only, pt.n_memorization, rng)
            b = res["primary (act+react)"]
            res["verdict_amended_rule"] = verdict_from_ci(*b["pooled_diff_ci95"], b["n_clusters"])
        else:
            res["verdict_amended_rule"] = "inconclusive"
            res["note"] = "no memorization point is adjacent to a grokking point under this definition"
        return res

    neigh = {"global value list (primary)": neighbour_analysis("global"),
             "nearest present value on the line (alternative)": neighbour_analysis("line"),
             "registered coarse grid only (sensitivity)": neighbour_analysis("coarse")}

    # ------------------------------------------------------------------ Q7 (b): grokking runs, before generalisation
    gk = g[g.label == "grokking"]
    late_g = ev[ev.late & (ev.label == "grokking")].copy()
    late_g["onset_over_t90"] = late_g.step / late_g.t_test_acc90
    first_g = late_g.sort_values("step").groupby("object_id").first()
    sec_b = {
        "n_grokking_runs": int(len(gk)),
        "k_late (act+react)": int(gk.m_late.sum()),
        "k_late_before_gen (act+react, step_lo < t_test_acc90)": int(gk.m_late_before_gen.sum()),
        "k_late_act_only": int(gk.m_late_act.sum()),
        "k_late_act_only_before_gen (v0 definition; v0 81/137 = 59 %)": int(gk.m_late_act_before_gen.sum()),
        "pct_late_before_gen": pct(gk.m_late_before_gen.sum(), len(gk)),
        "pct_late_act_only_before_gen": pct(gk.m_late_act_before_gen.sum(), len(gk)),
        "among_runs_with_late: k_before_gen / k_late": [int(gk.m_late_before_gen.sum()), int(gk.m_late.sum())],
        "onset_over_t_test_acc90_all_late_events_quantiles_10_25_50_75_90":
            np.round(late_g.onset_over_t90.quantile([.1, .25, .5, .75, .9]).values, 3).tolist(),
        "n_late_events_grokking": int(len(late_g)),
        "first_late_onset_over_t90_median": float((first_g.step / first_g.t_test_acc90).median()),
        "same_config_subset": {"n": int(len(grok_sc)), "k_before_gen": int(grok_sc.m_late_before_gen.sum())},
    }

    # ------------------------------------------------------------------ v0 reproduction (code check)
    o0 = pd.read_parquet(DR0 / "objects.parquet")
    r0 = pd.read_parquet(DR0 / "ri_runs.parquet")
    e0 = pd.read_parquet(DR0 / "ri_events.parquet")
    g0 = o0[o0.region == "grokking"]
    f0 = r0[(r0.kind == "followup") & r0.object_id.isin(g0.object_id)]
    e0 = e0[e0.ri_run_id.isin(f0.ri_run_id)].merge(g0[["object_id", "label", "t_train_acc99", "t_test_acc90"]],
                                                   on="object_id")
    gid0 = set(f0.merge(g0[["object_id", "label"]], on="object_id").query("label == 'grokking'").object_id)
    v0rep = {}
    for name, kinds in (("activation-only", ["activation"]), ("act+react", ["activation", "reactivation"])):
        s = e0[e0.kind.isin(kinds) & (e0.peak_ri >= PEAK) & (e0.step_lo >= e0.t_train_acc99)
               & e0.object_id.isin(gid0)]
        v0rep[name] = {"n_grokking_followed": len(gid0), "k_late": int(s.object_id.nunique()),
                       "k_late_before_gen": int(s[s.step_lo < s.t_test_acc90].object_id.nunique())}

    # ------------------------------------------------------------------ Q8: memorization fraction, point by point
    v0g = with_cfg(o0[o0.region == "grokking"].copy())
    on_grid = lambda d: np.logical_and.reduce([d[k].isin(v) for k, v in GRID.items()])  # noqa: E731
    v0grid = v0g[on_grid(v0g)].copy()
    v1grid = g[on_grid(g)].copy()
    v0grid["cfg_key"] = list(zip(*[v0grid[k] for k in KEY]))
    for d in (v0grid, v1grid):
        d["mem"] = d.label == "memorization"
    q8_literal = {"v1_region_memorization": [int(g.label.eq("memorization").sum()), int(len(g))],
                  "v0_all_regions_memorization": [int(o0.label.eq("memorization").sum()), int(len(o0))],
                  "v0_grokking_region_memorization": [int(v0g.label.eq("memorization").sum()), int(len(v0g))],
                  "registered_baseline_as_written": "866 of 2,166"}
    q8_literal["v1_pct"] = pct(*q8_literal["v1_region_memorization"])
    q8_literal["v0_as_written_pct (866/2166)"] = pct(866, 2166)
    q8_literal["v0_grokking_region_pct"] = pct(*q8_literal["v0_grokking_region_memorization"])
    pts0 = v0grid.groupby("cfg_key").agg(v0_n=("mem", "size"), v0_mem=("mem", "sum"),
                                         v0_labels=("label", lambda s: s.value_counts().to_dict()))
    pts1 = v1grid.groupby("cfg_key").agg(v1_n=("mem", "size"), v1_mem=("mem", "sum"),
                                         v1_labels=("label", lambda s: s.value_counts().to_dict()),
                                         cfg=("cfg", "first"))
    v1grid04 = v1grid[v1grid.seed <= 4]
    pts1b = v1grid04.groupby("cfg_key").agg(v1_seeds04_n=("mem", "size"), v1_seeds04_mem=("mem", "sum"))
    # same seeds as v0 at the point
    v0seeds = v0grid.groupby("cfg_key").seed.apply(set)
    v1same = v1grid[[s in v0seeds.get(k, set()) for k, s in zip(v1grid.cfg_key, v1grid.seed)]]
    pts1c = v1same.groupby("cfg_key").agg(v1_sameseeds_n=("mem", "size"), v1_sameseeds_mem=("mem", "sum"))
    q8pts = pts1.join(pts1b).join(pts1c).join(pts0, how="left").reset_index()
    q8pts[KEY_NAMES] = pd.DataFrame(q8pts.cfg_key.tolist(), index=q8pts.index)
    q8pts["v0_frac"] = q8pts.v0_mem / q8pts.v0_n
    q8pts["v1_frac"] = q8pts.v1_mem / q8pts.v1_n
    q8pts["v1_seeds04_frac"] = q8pts.v1_seeds04_mem / q8pts.v1_seeds04_n
    q8pts["v1_sameseeds_frac"] = q8pts.v1_sameseeds_mem / q8pts.v1_sameseeds_n
    both = q8pts[q8pts.v0_n.notna()].copy()
    q8 = {"n_points_v1": int(len(q8pts)), "n_points_v0": int(both.shape[0]), "n_points_both": int(both.shape[0]),
          "v0_mem_on_grid": [int(v0grid.mem.sum()), int(len(v0grid))],
          "v1_mem_on_grid_all_seeds": [int(v1grid.mem.sum()), int(len(v1grid))],
          "v1_mem_on_grid_seeds0_4": [int(v1grid04.mem.sum()), int(len(v1grid04))]}
    b1 = v1grid[v1grid.cfg_key.isin(set(both.cfg_key))]
    q8["v1_mem_on_shared_points_all_seeds"] = [int(b1.mem.sum()), int(len(b1))]
    q8["v1_mem_on_shared_points_same_seeds_as_v0"] = [int(both.v1_sameseeds_mem.sum()), int(both.v1_sameseeds_n.sum())]
    q8["v1_points_absent_in_v0"] = records(q8pts[q8pts.v0_n.isna()][["cfg", "v1_n", "v1_mem", "v1_labels"]])
    for name, col in (("all_seeds", "v1_frac"), ("seeds0_4", "v1_seeds04_frac"), ("same_seeds_as_v0", "v1_sameseeds_frac")):
        d = both[col].values - both.v0_frac.values
        idx = rng.integers(0, len(both), size=(N_BOOT, len(both)))
        q8["point_pooled_" + name] = {
            "mean_frac_v0": float(both.v0_frac.mean()), "mean_frac_v1": float(both[col].mean()),
            "mean_diff_v1_minus_v0": float(d.mean()), "ci95_resampling_points": ci(d[idx].mean(1)),
            "n_points_v1_lower": int((d < 0).sum()), "n_points_equal": int((d == 0).sum()),
            "n_points_v1_higher": int((d > 0).sum())}
    q8["points_with_v0_all_mem_and_v1_none"] = int(((both.v0_frac == 1) & (both.v1_frac == 0)).sum())
    q8["n_equal_all_mem"] = int(((both.v0_frac == 1) & (both.v1_frac == 1)).sum())
    q8["n_equal_no_mem"] = int(((both.v0_frac == 0) & (both.v1_frac == 0)).sum())
    q8_table = q8pts[["cfg", *KEY_NAMES, "v0_n", "v0_mem", "v0_frac", "v0_labels", "v1_n", "v1_mem", "v1_frac",
                      "v1_labels", "v1_seeds04_n", "v1_seeds04_mem", "v1_sameseeds_n", "v1_sameseeds_mem"]]

    # ------------------------------------------------------------------ Q8 paired: same configuration and seed
    lc_dir = SCR / "dr0_lc"
    need = set(v0g.lightcurve_file)  # all v0 grokking-region light curves that have a v1 twin (selected below)
    v0g["cfg_key"] = list(zip(*[v0g[k] for k in KEY]))
    pair = v0g.merge(g, on=KEY + ["seed"], suffixes=("_0", "_1"))
    need = set(pair.lightcurve_file_0)
    missing = [m for m in need if not (lc_dir / m).exists()]
    if missing:
        lc_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DR0 / "lightcurves.tar.gz") as t:
            for mem in t:
                if mem.name in need and mem.isfile():
                    t.extract(mem, lc_dir, filter="data")
    tc_fields = [f.name for f in dataclasses.fields(TrainConfig)]
    diff_fields = {}
    for f in tc_fields:
        if f"train_{f}" in KEY:  # lr, weight_decay: equal by construction (merge keys)
            continue
        a, b = pair[f"train_{f}_0"], pair[f"train_{f}_1"]
        neq = ~((a == b) | (a.isna() & b.isna()))
        if neq.any():
            diff_fields[f] = {"v0": sorted(map(str, a[neq].unique())), "v1": sorted(map(str, b[neq].unique())),
                              "n_pairs_differing": int(neq.sum())}
    task_diff = {}
    for a, b in zip(pair.task_cfg_json_0, pair.task_cfg_json_1):
        da, db = json.loads(a), json.loads(b)
        for k in set(da) | set(db):
            if da.get(k) != db.get(k):
                task_diff.setdefault(k, set()).add((str(da.get(k)), str(db.get(k))))
    lcrows = []
    shared_steps_seen: set = set()
    for r in pair.itertuples():
        with np.load(lc_dir / r.lightcurve_file_0) as za, np.load(X1 / r.lightcurve_file_1) as zb:
            a = {k: za[k] for k in za.files}
            b = {k: zb[k] for k in zb.files}
        common, ia, ib = np.intersect1d(a["step"], b["step"], return_indices=True)
        shared_steps_seen.add(tuple(int(x) for x in common))
        row = {"object_id_v0": r.object_id_0, "object_id_v1": r.object_id_1, "n_shared_steps": int(common.size),
               "max_shared_step": int(common.max()), "n_shared_steps_ge_1000": int((common >= 1000).sum())}
        for k in ("train_loss", "test_loss", "train_acc", "test_acc", "param_norm"):
            x, y = a[k][ia].astype(float), b[k][ib].astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            row["maxabs_" + k] = float(np.max(np.abs(x[ok] - y[ok]))) if ok.any() else float("nan")
            row["nan_mismatch_" + k] = int((np.isfinite(x) != np.isfinite(y)).sum())
        # v1 run truncated at the v0 budget: does it reach 90 % test accuracy by 20k?
        m20 = b["step"] <= V0_BUDGET
        row["v1_test_acc_max_upto_20k"] = float(np.nanmax(b["test_acc"][m20]))
        row["v1_test_acc_at_last_eval_le_20k"] = float(b["test_acc"][m20][-1])
        row["v0_final_test_acc"] = float(a["test_acc"][-1])
        lcrows.append(row)
    lct = pd.DataFrame(lcrows)
    pair = pair.merge(lct, left_on="object_id_0", right_on="object_id_v0")
    pair["on_grid"] = on_grid(pair)
    dev_cols = [c for c in lct.columns if c.startswith("maxabs_")]
    pair["max_dev"] = pair[dev_cols].max(axis=1)
    pair["identical"] = (pair.max_dev == 0) & (pair[[c for c in lct.columns if c.startswith("nan_mismatch_")]].sum(axis=1) == 0)
    coincide_grid = bool(pair[pair.on_grid].identical.all())
    paired = {
        "train_config_fields_differing": diff_fields,
        "task_cfg_keys_differing": {k: sorted(v) for k, v in task_diff.items()},
        "code_version_v0": sorted(pair.code_version_0.unique()), "code_version_v1": sorted(pair.code_version_1.unique()),
        "n_pairs_all_v0_grokking_region_with_v1_twin": int(len(pair)),
        "n_pairs_on_grid": int(pair.on_grid.sum()),
        "n_grid_v0_objects_without_v1_twin": int(len(v0grid) - pair.on_grid.sum()),
        "n_pairs_bit_identical_on_shared_steps (all)": int(pair.identical.sum()),
        "n_pairs_bit_identical_on_shared_steps (grid)": int(pair[pair.on_grid].identical.sum()),
        "max_abs_deviation_any_key_any_pair": float(pair.max_dev.max()),
        "shared_steps_per_pair": {"min": int(pair.n_shared_steps.min()), "max": int(pair.n_shared_steps.max()),
                                  "max_shared_step_min": int(pair.max_shared_step.min()),
                                  "shared_steps_ge_1000_min": int(pair.n_shared_steps_ge_1000.min())},
        "trajectories_coincide": coincide_grid,
        "shared_step_lists": [list(x) for x in sorted(shared_steps_seen)],
    }
    gp = pair[pair.on_grid]
    paired["label_transitions_grid (v0 -> v1)"] = {f"{a}->{b}": int(n) for (a, b), n in
                                                   gp.groupby(["label_0", "label_1"]).size().items()}
    paired["label_transitions_all_pairs (v0 -> v1)"] = {f"{a}->{b}": int(n) for (a, b), n in
                                                        pair.groupby(["label_0", "label_1"]).size().items()}
    # consistency: a v0 memorization run cannot reach 90 % test accuracy by 20k in v1 if the runs coincide
    mem0 = gp[gp.label_0 == "memorization"]
    paired["consistency_v0_mem_with_v1_t90_le_20k"] = int((mem0.t_test_acc90_1 <= V0_BUDGET).sum())
    paired["consistency_v0_mem_v1_max_test_acc_upto_20k_max"] = float(mem0.v1_test_acc_max_upto_20k.max())
    conv = mem0[mem0.t_test_acc90_1 > V0_BUDGET]
    convg = mem0[mem0.label_1 == "grokking"]

    def dist(s):
        s = s.dropna()
        if s.empty:
            return None
        return {"n": int(s.size), "min": float(s.min()), "q25": float(s.quantile(.25)), "median": float(s.median()),
                "q75": float(s.quantile(.75)), "max": float(s.max()), "values": sorted(float(x) for x in s)}

    paired["v0_memorization_runs_on_grid"] = int(len(mem0))
    paired["v0_mem_v1_labels"] = mem0.label_1.value_counts().to_dict()
    paired["v0_mem_with_v1_t_test_acc90_gt_20k"] = int(len(conv))
    paired["v0_mem_labelled_grokking_in_v1"] = int(len(convg))
    paired["v1_t_test_acc90_of_converted"] = dist(conv.t_test_acc90_1)
    paired["v1_t_test_acc90_of_converted_by_point"] = {
        c: sorted(int(x) for x in s) for c, s in conv.groupby("cfg").t_test_acc90_1}
    paired["v0_mem_v1_final_test_acc_by_v1_label"] = {
        lab: dist(s) for lab, s in mem0.groupby("label_1").final_test_acc_1}
    mem0_all = pair[pair.label_0 == "memorization"]
    paired["all_pairs_v0_mem"] = {"n": int(len(mem0_all)), "v1_labels": mem0_all.label_1.value_counts().to_dict(),
                                  "v1_t90_gt_20k": int((mem0_all.t_test_acc90_1 > V0_BUDGET).sum()),
                                  "v1_t90_dist": dist(mem0_all[mem0_all.t_test_acc90_1 > V0_BUDGET].t_test_acc90_1)}
    # v0 grokking runs: same t90 crossing in v1?
    gk0 = gp[gp.label_0 == "grokking"]
    paired["v0_grokking_on_grid"] = {"n": int(len(gk0)), "v1_labels": gk0.label_1.value_counts().to_dict(),
                                     "t90_v0_vs_v1_equal": int((gk0.t_test_acc90_0 == gk0.t_test_acc90_1).sum())}
    pair_table = gp[["cfg", "seed", "object_id_0", "object_id_1", "label_0", "label_1", "t_train_acc99_0",
                     "t_train_acc99_1", "t_test_acc90_0", "t_test_acc90_1", "final_test_acc_0", "final_test_acc_1",
                     "n_shared_steps", "max_shared_step", "max_dev", "identical",
                     "v1_test_acc_max_upto_20k"]].sort_values(["cfg", "seed"])

    # ------------------------------------------------------------------ items
    def fmt_ci(c, scale=100):
        return f"[{scale * c[0]:+.1f}, {scale * c[1]:+.1f}] pp" if all(map(math.isfinite, c)) else "[n/a]"

    n_cfg, n_g, n_c = len(cfg_tab), int(cfg_tab.n_grokking.sum()), int(cfg_tab.n_control.sum())
    kg, kc = int(cfg_tab.k_grokking.sum()), int(cfg_tab.k_control.sum())
    kga, kca = int(cfg_tab.k_grokking_act_only.sum()), int(cfg_tab.k_control_act_only.sum())
    ctrl_labels = controls.label.value_counts().to_dict()
    ctrl_acc = controls.final_test_acc
    items = []
    items.append({
        "id": "q7",
        "question": ("Q7 (registered) + Amendment F.1/F.3: at configurations whose seeds disagree, is the fraction of "
                     "runs with a strong activation (peak >= 0.02 nats) after the training set is fitted higher in "
                     "grokking seeds than in the other seeds (bootstrap CI above 0)?"),
        "definition_used": (
            "Per run: has >= 1 event of kind activation or reactivation with peak_ri >= 0.02 nats and step_lo >= "
            "t_train_acc99 (onset bracket (step_lo, step] entirely after the first light-curve evaluation with train "
            "acc >= 0.99; left-censored onsets have step_lo = 0 and never qualify). Events from the DR1 ri_events "
            "table (follow-up replay, alpha 0.01, min_effect 0.005, persistence 2), cross-checked against "
            "detect_events(curve, alpha=0.01) on all 689 curves. Configuration = (p, train_frac, width, lr, wd). "
            "Comparison set = every configuration whose seeds include 'grokking' and >= 1 other label (all its "
            "seeds; the registered 'configurations whose seeds disagree' restricted to those that contain a grokking "
            "seed, since only they allow a grokking-vs-control contrast). Statistic: pooled fraction in grokking "
            "seeds minus pooled fraction in the other seeds; 95 % percentile CI from 2,000 bootstrap resamples of "
            f"configurations (seed {SEED}). Alternatives reported: activation-only events (v0's exact code, which "
            "reproduces v0's 85/137 = 62 % and 81/137 = 59 % on DR0), and the mean over configurations of the "
            "per-configuration difference. Literal verdict: supported iff CI lower bound > 0, else 'not supported "
            "(property of the label-selected sample)'. Amended (F.3, decides): CI above 0 supported, below 0 not "
            "supported, including 0 with few (< 10) configurations inconclusive."),
        "literal_result": (
            f"{n_cfg} configurations ({sorted(cfg_tab.width.unique().tolist())} width), {n_g + n_c} runs: grokking "
            f"{kg}/{n_g} ({pct(kg, n_g):.1f} %) vs other seeds {kc}/{n_c} ({pct(kc, n_c):.1f} %); difference "
            f"{100 * prim['pooled_diff']:+.1f} pp, 95 % CI {fmt_ci(prim['pooled_diff_ci95'])}. "
            + ("CI above 0 -> literal reading: supported. " if literal_v == "supported" else
               "CI not above 0 -> literal reading: not supported (the registered text calls this 'a property of the "
               "label-selected sample'). ")
            + f"Activation-only: {kga}/{n_g} vs {kca}/{n_c}, {100 * alt['pooled_diff']:+.1f} pp, CI "
            f"{fmt_ci(alt['pooled_diff_ci95'])}."),
        "literal_verdict": literal_v,
        "amended_result": (
            f"Same numbers; every control is near-grokking: labels {ctrl_labels}, final test accuracy "
            f"{ctrl_acc.min():.3f}-{ctrl_acc.max():.3f} (median {ctrl_acc.median():.3f}); no memorization seed shares "
            f"a configuration with a grokking seed (memorization at these {n_cfg} configurations: "
            f"{int((sc.label == 'memorization').sum())} runs). "
            + (f"With {n_cfg} configurations and a CI including 0 the result is inconclusive (F.3), not evidence that "
               "the v0 observation was a property of the label-selected sample. " if amended_v == "inconclusive" else
               f"Amended verdict: {amended_v}. ")
            + f"Mean of per-configuration differences {100 * prim['mean_cluster_diff']:+.1f} pp, CI "
            f"{fmt_ci(prim['mean_cluster_diff_ci95'])}; activation-only CI {fmt_ci(alt['pooled_diff_ci95'])} "
            "(same verdict: " + verdict_from_ci(*alt['pooled_diff_ci95'], alt['n_clusters']) + ")."),
        "amended_verdict": amended_v,
        "deciding": "amended",
        "key_numbers": "\n".join([
            f"configurations with grokking + other label: {n_cfg} (runs {n_g + n_c}: {n_g} grokking, {n_c} control)",
            f"configurations whose seeds disagree (any labels): {len(disagree_keys)}",
            f"grokking with measure: {kg}/{n_g}",
            f"controls with measure: {kc}/{n_c} (labels {ctrl_labels})",
            f"pooled diff {100 * prim['pooled_diff']:+.2f} pp, CI95 {fmt_ci(prim['pooled_diff_ci95'])}",
            f"activation-only: {kga}/{n_g} vs {kca}/{n_c}, diff {100 * alt['pooled_diff']:+.2f} pp, CI95 "
            f"{fmt_ci(alt['pooled_diff_ci95'])}",
            f"control final test acc min/median/max {ctrl_acc.min():.4f}/{ctrl_acc.median():.4f}/{ctrl_acc.max():.4f}",
        ]),
        "interpretation": (
            "Within a configuration the only contrast available is grokking vs smooth seeds that stopped just short "
            "of 90 % test accuracy at 50k steps (the same generalising phase, as Amendment F anticipated). Strong "
            f"late activations are about as common in those near-grokking controls ({pct(kc, n_c):.0f} %) as in the "
            f"grokking seeds ({pct(kg, n_g):.0f} %), and the number of such events per run is similar "
            f"({same_cfg_nev['mean_events_per_run_grokking']:.2f} vs {same_cfg_nev['mean_events_per_run_control']:.2f}). "
            "The CI is wide and spans 0, so by the amended rule the question is inconclusive; the literal rule "
            "would call the v0 observation a property of the label-selected sample. Either way v1 gives no evidence "
            "that late hidden-unit interactions are specific to grokking seeds."),
        "caveats": (
            f"{n_cfg} configurations, {n_g + n_c} runs; grokking:control seeds per configuration "
            + ", ".join(f"{a}:{b}" for a, b in zip(cfg_tab.n_grokking, cfg_tab.n_control))
            + ", so several configurations add a single run to one side. 'Few configurations' is not quantified in "
            f"Amendment F; with {n_cfg} configurations and a CI {100 * (prim['pooled_diff_ci95'][1] - prim['pooled_diff_ci95'][0]):.0f} "
            "pp wide it is treated as few (threshold used: < 10). "
            "The measure fires on any order-2/3 hidden-unit interaction with peak >= 0.02 after the early fit "
            f"(median t_train_acc99 {ckpt_after['grokking']['median_t_train_acc99']:.0f} steps in grokking runs, "
            f"{ckpt_after['smooth']['median_t_train_acc99']:.0f} in smooth runs), so it is not specific to the hidden "
            "phase. Only 10 hidden units per run are binarised (plus the 'correct' column, which is in no strong "
            f"event). {int((g.n_points == 60).sum())} of {len(g)} replays have only the 60 log-spaced checkpoints "
            "(the dense window needs a detected sharp loss drop, found in 1 object). Every grokking-v1 object was "
            "followed up (689/689, replay bit-identical to the stored light curve), so the matched sets are complete."),
    })
    na = neigh["global value list (primary)"]
    nb_ = neigh["nearest present value on the line (alternative)"]
    nc_ = neigh["registered coarse grid only (sensitivity)"]

    def neigh_txt(r):
        if r["n_pairs"] == 0:
            return "0 pairs: no memorization point adjacent to a grokking point; comparison not estimable"
        b = r["primary (act+react)"]
        pt = pd.DataFrame(r["pairs"])
        return (f"{r['n_pairs']} pairs ({r['n_grok_points']} grokking points, {r['n_mem_points']} memorization "
                f"points; {r['unique_grokking_runs']} grokking runs, {r['unique_memorization_runs']} memorization "
                f"runs): grokking {int(pt.k_grokking.sum())}/{int(pt.n_grokking.sum())} vs memorization "
                f"{int(pt.k_memorization.sum())}/{int(pt.n_memorization.sum())} (pooled over pairs), diff "
                f"{100 * b['pooled_diff']:+.1f} pp, CI95 {fmt_ci(b['pooled_diff_ci95'])}")

    items.append({
        "id": "q7_F2_neighbour",
        "question": ("Amendment F.2 (confounded by configuration): memorization runs at a grid point adjacent (one "
                     "knob step in train_frac or weight decay) to a point with grokking runs, matched on width and "
                     "lr; same measure, CI resampling configuration pairs."),
        "definition_used": (
            "Primary adjacency: one step in the sorted list of the knob's values present anywhere in grokking-v1 "
            f"(train_frac {len(tf_global)} values, weight decay {len(wd_global)} values, including refinement "
            "midpoints), other knobs equal, the neighbouring point must exist and contain >= 1 memorization run. "
            "Alternative: the nearest train_frac (weight decay) value present on the same line (same width, lr and "
            "other knob). Sensitivity: adjacency on the registered coarse grid only (tf {0.3,0.4,0.5} x wd "
            "{0.1,0.3,1.0}). Unit = (grokking point, memorization point) pair; grokking runs of the grokking point "
            "vs memorization runs of the memorization point; pooled-over-pairs difference, 2,000 pair resamples."),
        "literal_result": "not part of the registered question (added by Amendment 1 F.2)",
        "literal_verdict": "not_applicable",
        "amended_result": ("Primary: " + neigh_txt(na) + ". Alternative (line): " + neigh_txt(nb_)
                           + ". Sensitivity (coarse grid): " + neigh_txt(nc_) + "."),
        "amended_verdict": na["verdict_amended_rule"],
        "deciding": "amended",
        "key_numbers": "\n".join([f"global: {neigh_txt(na)}", f"line: {neigh_txt(nb_)}", f"coarse: {neigh_txt(nc_)}"]),
        "interpretation": (
            "After boundary refinement, grokking points and memorization points are never one knob step apart: on "
            "every train_frac and weight-decay line the nearest sampled neighbours of grokking points are smooth or "
            "grokking points, except one line (lr 0.001, wd 0.5477, width 256) where train_frac 0.3375-0.3625 was "
            "not sampled and a 1-grokking-seed point faces a memorization point. The amended neighbour comparison is "
            "therefore not estimable (inconclusive, 0 pairs; 1 pair under the alternative). Only on the coarse "
            "registered grid, outside the amendment's definition, do pairs exist; there grokking runs carry the "
            "measure far more often than memorization runs, but that contrast is between phases at different "
            "configurations and is confounded by configuration."),
        "caveats": ("Confounded by configuration by design; the coarse-grid sensitivity has only 6 pairs (5 grokking "
                    "points), so its bootstrap CI understates the uncertainty."),
    })
    bl = by_label["strong_late (act+react; primary)"]
    bla = by_label["strong_late activation-only (v0 code)"]
    items.append({
        "id": "q7_by_label",
        "question": "Descriptive: fraction of runs with a strong late activation, by label, over all 689 grokking-v1 runs.",
        "definition_used": ("Same per-run measure as q7; all 689 objects were followed up. CIs resample "
                            "configurations within the label. Differences between labels resample all 101 "
                            "configurations and are confounded by configuration."),
        "literal_result": "; ".join(f"{lab} {v['k']}/{v['n']} ({v['pct']:.1f} %, CI {v['ci95_pct_resampling_configs'][0]:.1f}-"
                                    f"{v['ci95_pct_resampling_configs'][1]:.1f})" for lab, v in bl.items())
                          + " | activation-only: " + "; ".join(f"{lab} {v['k']}/{v['n']}" for lab, v in bla.items()),
        "literal_verdict": "descriptive",
        "key_numbers": "\n".join(
            [f"{lab}: {v['k']}/{v['n']} = {v['pct']:.1f} % (configs {v['n_configs']})" for lab, v in bl.items()]
            + [f"diff {k}: {v['diff_pct']:+.1f} pp CI {v['ci95_pct_resampling_all_101_configs'][0]:+.1f}.."
               f"{v['ci95_pct_resampling_all_101_configs'][1]:+.1f}" for k, v in region_diffs.items()]
            + [f"mean late events per run {lab}: {v['mean']:.2f} (median {v['median']:.0f})" for lab, v in n_ev_by_label.items()]
            + [f"runs never reaching 99 % train acc: {checks['never_reached_train99']['total']}"]),
        "interpretation": ("Across the region the measure is most frequent in grokking runs and least in "
                           "memorization runs, and it rises steadily with final test accuracy ("
                           + ", ".join(f"{r['final_test_acc_bin']}: {int(r['sum'])}/{int(r['size'])}"
                                       for r in records(by_acc_all))
                           + "), so the label contrast is largely a contrast in how far runs generalise. Labels sit "
                           "at different configurations, so this is descriptive only."),
        "deciding": "literal",
        "caveats": ("Labels are confounded with configuration (train_frac, wd, width, lr). Grokking runs fit the "
                    f"training set earlier (median t_train_acc99 {ckpt_after['grokking']['median_t_train_acc99']:.0f} vs "
                    f"{ckpt_after['memorization']['median_t_train_acc99']:.0f} steps), leaving slightly more RI "
                    f"checkpoints after the fit (median {ckpt_after['grokking']['median_ckpt_ge_t99']:.0f} vs "
                    f"{ckpt_after['memorization']['median_ckpt_ge_t99']:.0f} of 60)."),
    })
    items.append({
        "id": "q7_secondary",
        "question": ("Q7 secondary: (a) number of strong late events per run (grokking vs controls at the same "
                     "configurations), (b) does a strong late activation start before generalisation "
                     "(step_lo < t_test_acc90) in grokking runs (v0: 59 %)?"),
        "definition_used": ("(a) count of qualifying events (activation + reactivation) per run; difference of means "
                            "per run, configurations resampled. (b) per grokking run: >= 1 qualifying event with "
                            "t_train_acc99 <= step_lo < t_test_acc90; v0's 59 % used activations only (81/137, "
                            "reproduced from DR0 with this code); both reported."),
        "literal_result": (
            f"(a) same-configuration set: mean {same_cfg_nev['mean_events_per_run_grokking']:.2f} events/run in "
            f"grokking vs {same_cfg_nev['mean_events_per_run_control']:.2f} in controls, diff "
            f"{same_cfg_nev['diff']:+.2f}, CI95 [{same_cfg_nev['diff_ci95'][0]:+.2f}, {same_cfg_nev['diff_ci95'][1]:+.2f}]. "
            f"(b) all {sec_b['n_grokking_runs']} grokking runs: {sec_b['k_late_before_gen (act+react, step_lo < t_test_acc90)']} "
            f"({sec_b['pct_late_before_gen']:.1f} %) have one before generalisation (act+react); activation-only "
            f"{sec_b['k_late_act_only_before_gen (v0 definition; v0 81/137 = 59 %)']} ({sec_b['pct_late_act_only_before_gen']:.1f} %), "
            "v0 81/137 (59.1 %)."),
        "literal_verdict": "descriptive",
        "key_numbers": "\n".join([
            f"(a) mean events/run grokking {same_cfg_nev['mean_events_per_run_grokking']:.3f}, control "
            f"{same_cfg_nev['mean_events_per_run_control']:.3f}",
            f"(b) act+react before gen: {sec_b['k_late_before_gen (act+react, step_lo < t_test_acc90)']}/{sec_b['n_grokking_runs']}",
            f"(b) activation-only before gen: {sec_b['k_late_act_only_before_gen (v0 definition; v0 81/137 = 59 %)']}/{sec_b['n_grokking_runs']}",
            f"(b) late at all (act+react): {sec_b['k_late (act+react)']}/{sec_b['n_grokking_runs']}; activation-only {sec_b['k_late_act_only']}",
            f"median onset/t_test_acc90 of late events in grokking runs: {sec_b['onset_over_t_test_acc90_all_late_events_quantiles_10_25_50_75_90'][2]}",
            f"v0 reproduction (DR0): {v0rep}",
        ]),
        "interpretation": (f"{sec_b['k_late_before_gen (act+react, step_lo < t_test_acc90)']} of "
                           f"{sec_b['n_grokking_runs']} grokking runs have a strong post-fit interaction that switches "
                           "on before test accuracy reaches 90 % (activation-only "
                           f"{sec_b['pct_late_act_only_before_gen']:.0f} % vs v0's 59 %), but at the same "
                           "configurations the near-grokking controls show as many such events per run, so the "
                           "observation does not single out grokking."),
        "deciding": "literal",
        "caveats": ("t_test_acc90 is never reached by non-grokking runs (censored at 50k), so (b) is only defined for "
                    "grokking runs; v0 used activations only (its 59 % is reproduced exactly from DR0 by this code), "
                    "the task's measure also counts reactivations."),
    })
    q8p = q8["point_pooled_all_seeds"]
    items.append({
        "id": "q8",
        "question": ("Q8 (registered) + Amendment G: fraction of runs labelled memorization at 50k steps vs v0 at 20k; "
                     "amended baseline: v0 objects on the v1 grid points, compared point by point."),
        "definition_used": (
            "Literal: v1 region memorization count / all grokking-v1 runs vs the registered '866 of 2,166' (and the "
            "corrected 742 of 2,166 of v0's grokking region). Amended (G, decides): v0 region 'grokking' with p 47, "
            "train_frac {0.3,0.4,0.5}, width {64,256}, lr {0.001,0.003}, wd {0.1,0.3,1.0}; v1 runs at the same 36 "
            "points (all seeds incl. refinement seeds 5-6; also seeds 0-4 only and the exact v0 seeds). Pool by "
            "point: each point's memorization fraction, mean over the points present in both, v1 minus v0, CI from "
            "2,000 resamples of points."),
        "literal_result": (f"v1 {q8_literal['v1_region_memorization'][0]}/{q8_literal['v1_region_memorization'][1]} "
                           f"({q8_literal['v1_pct']:.1f} %) vs registered 866/2,166 ({q8_literal['v0_as_written_pct (866/2166)']:.1f} %); "
                           f"866 is actually v0's all-region count out of {q8_literal['v0_all_regions_memorization'][1]}; "
                           f"v0 grokking region {q8_literal['v0_grokking_region_memorization'][0]}/{q8_literal['v0_grokking_region_memorization'][1]} "
                           f"({q8_literal['v0_grokking_region_pct']:.1f} %). Different grids, so not like for like."),
        "literal_verdict": "descriptive",
        "amended_result": (
            f"v0: {q8['v0_mem_on_grid'][0]}/{q8['v0_mem_on_grid'][1]} memorization on {q8['n_points_both']} of 36 "
            f"points. v1 on the same {q8['n_points_both']} points: {q8['v1_mem_on_shared_points_all_seeds'][0]}/"
            f"{q8['v1_mem_on_shared_points_all_seeds'][1]} (all seeds), {q8['v1_mem_on_shared_points_same_seeds_as_v0'][0]}/"
            f"{q8['v1_mem_on_shared_points_same_seeds_as_v0'][1]} (same seeds as v0); all 36 points: "
            f"{q8['v1_mem_on_grid_all_seeds'][0]}/{q8['v1_mem_on_grid_all_seeds'][1]}. Pooled by point: mean "
            f"memorization fraction {100 * q8p['mean_frac_v0']:.1f} % (v0) -> {100 * q8p['mean_frac_v1']:.1f} % (v1), "
            f"diff {100 * q8p['mean_diff_v1_minus_v0']:+.1f} pp, CI {fmt_ci(q8p['ci95_resampling_points'])}; points "
            f"lower/equal/higher in v1: {q8p['n_points_v1_lower']}/{q8p['n_points_equal']}/{q8p['n_points_v1_higher']}."),
        "amended_verdict": "descriptive",
        "deciding": "amended",
        "key_numbers": "\n".join([
            f"v1 region: {q8_literal['v1_region_memorization']}", f"v0 all regions: {q8_literal['v0_all_regions_memorization']}",
            f"v0 grokking region: {q8_literal['v0_grokking_region_memorization']}",
            f"v0 grid: {q8['v0_mem_on_grid']} on {q8['n_points_both']} points",
            f"v1 grid all seeds: {q8['v1_mem_on_grid_all_seeds']}; seeds 0-4: {q8['v1_mem_on_grid_seeds0_4']}",
            f"v1 shared points all seeds: {q8['v1_mem_on_shared_points_all_seeds']}; same seeds: "
            f"{q8['v1_mem_on_shared_points_same_seeds_as_v0']}",
            f"point-pooled v0 {q8p['mean_frac_v0']:.4f} v1 {q8p['mean_frac_v1']:.4f} diff {q8p['mean_diff_v1_minus_v0']:+.4f} "
            f"CI {q8p['ci95_resampling_points']}",
        ]),
        "interpretation": (f"On the same grid points the memorization fraction falls from {100 * q8p['mean_frac_v0']:.0f} % "
                           f"to {100 * q8p['mean_frac_v1']:.0f} % (point-pooled); it falls at "
                           f"{q8p['n_points_v1_lower']} points, rises at none, and is unchanged at "
                           f"{q8p['n_points_equal']} ({q8['n_equal_all_mem']} all-memorization in both, "
                           f"{q8['n_equal_no_mem']} memorization-free in both). Part of v0's memorization was "
                           "censoring, but most of it persists at 50k steps."),
        "caveats": "v1 adds refinement seeds 5-6 at 27 of the 36 points; results with seeds 0-4 and with v0's exact seeds are in the JSON.",
    })
    items.append({
        "id": "q8_paired",
        "question": ("Q8 paired (Amendment G follow-on): do v0 (20k) and v1 (50k) runs with the same configuration "
                     "and seed follow the same trajectory up to 20k steps, and if so how many v0 memorization runs "
                     "became grokking (v1 t_test_acc90 > 20,000)?"),
        "definition_used": (
            "Pairs = v0 grokking-region object and v1 grokking-v1 object with equal (p, train_frac, width, lr, wd, "
            "seed). Compared every TrainConfig field and task_cfg key, and train_loss, test_loss, train_acc, "
            "test_acc, param_norm at every evaluation step present in both light curves (exact equality). If all "
            "grid pairs coincide: v0 memorization runs on the grid whose v1 t_test_acc90 > 20,000, their v1 labels "
            "and the distribution of their v1 t_test_acc90."),
        "literal_result": (
            f"Config differences: {diff_fields}; task_cfg differences: { {k: sorted(v) for k, v in task_diff.items()} }. "
            f"{paired['n_pairs_bit_identical_on_shared_steps (grid)']}/{paired['n_pairs_on_grid']} grid pairs "
            f"(and {paired['n_pairs_bit_identical_on_shared_steps (all)']}/{paired['n_pairs_all_v0_grokking_region_with_v1_twin']} "
            f"of all matched pairs) are bit-identical on {paired['shared_steps_per_pair']['min']}-"
            f"{paired['shared_steps_per_pair']['max']} shared steps (last shared step >= "
            f"{paired['shared_steps_per_pair']['max_shared_step_min']}). Of {paired['v0_memorization_runs_on_grid']} v0 "
            f"memorization runs on the grid, {paired['v0_mem_with_v1_t_test_acc90_gt_20k']} reach 90 % test accuracy "
            f"after 20k in v1 ({paired['v0_mem_labelled_grokking_in_v1']} labelled grokking); v1 labels "
            f"{paired['v0_mem_v1_labels']}; v1 t_test_acc90 of those: "
            + (f"median {paired['v1_t_test_acc90_of_converted']['median']:.0f}, range "
               f"{paired['v1_t_test_acc90_of_converted']['min']:.0f}-{paired['v1_t_test_acc90_of_converted']['max']:.0f}"
               if paired['v1_t_test_acc90_of_converted'] else "none") + "."),
        "literal_verdict": "descriptive",
        "key_numbers": "\n".join([
            f"pairs on grid: {paired['n_pairs_on_grid']} (identical {paired['n_pairs_bit_identical_on_shared_steps (grid)']})",
            f"all matched pairs: {paired['n_pairs_all_v0_grokking_region_with_v1_twin']} (identical {paired['n_pairs_bit_identical_on_shared_steps (all)']})",
            f"max abs deviation: {paired['max_abs_deviation_any_key_any_pair']}",
            f"v0 mem on grid: {paired['v0_memorization_runs_on_grid']}; v1 labels {paired['v0_mem_v1_labels']}",
            f"v1 t90 > 20k: {paired['v0_mem_with_v1_t_test_acc90_gt_20k']}; labelled grokking: {paired['v0_mem_labelled_grokking_in_v1']}",
            f"consistency (v0 mem with v1 t90 <= 20k): {paired['consistency_v0_mem_with_v1_t90_le_20k']}",
            f"transitions (grid): {paired['label_transitions_grid (v0 -> v1)']}",
        ]),
        "interpretation": ("The paired comparison is valid: v1 reruns v0's grid objects bit for bit up to 20k steps "
                           "(constant lr schedule, no warmup; only the step budget, n_log_evals and the task_cfg key "
                           "ri_units differ, and the bit-identical light curves show that none of them changes the optimisation). Of "
                           f"the {paired['v0_memorization_runs_on_grid']} v0 memorization runs on the grid, "
                           f"{paired['v0_mem_v1_labels'].get('memorization', 0)} are still memorization at 50k, "
                           f"{paired['v0_mem_v1_labels'].get('smooth', 0)} became smooth (test accuracy rose but never "
                           f"reached 90 %) and {paired['v0_mem_labelled_grokking_in_v1']} grokked, all late "
                           f"(v1 t_test_acc90 {paired['v1_t_test_acc90_of_converted']['min']:.0f}-"
                           f"{paired['v1_t_test_acc90_of_converted']['max']:.0f}; "
                           f"{sum(1 for v in paired['v1_t_test_acc90_of_converted']['values'] if v == BUDGET)} exactly "
                           "at the last evaluation, 50,000)." if paired['v1_t_test_acc90_of_converted'] else
                           "The paired comparison is valid; no v0 memorization run grokked by 50k."),
        "deciding": "literal",
        "caveats": ("Equality was checked at the evaluation steps both light curves share ("
                    + "; ".join(", ".join(map(str, x)) for x in sorted(shared_steps_seen))
                    + "), not at every step; "
                    f"{sum(1 for v in (paired['v1_t_test_acc90_of_converted'] or {'values': []})['values'] if v == BUDGET)} "
                    f"of the {paired['v0_mem_with_v1_t_test_acc90_gt_20k']} conversions reach 90 % test accuracy only at "
                    "the final evaluation (50,000). Over all matched v0/v1 pairs incl. refinement midpoints: "
                    f"{paired['all_pairs_v0_mem']['n']} v0 memorization runs -> v1 {paired['all_pairs_v0_mem']['v1_labels']}."),
    })

    out = {
        "analysis": "sky v1, pre-registered questions 7 and 8 (grokking-v1), with Amendment 1 items F and G",
        "script": str(Path(__file__).resolve()),
        "constants": {"peak_ri_strong": PEAK, "alpha": ALPHA, "budget": BUDGET, "n_boot": N_BOOT, "seed": SEED},
        "items": items,
        "checks": checks,
        "v0_reproduction_on_DR0": v0rep,
        "q7_same_configuration": {"n_configurations": n_cfg, "configurations": records(cfg_tab),
                                  "bootstrap": same_cfg, "events_per_run": same_cfg_nev,
                                  "controls": records(controls), "grokking_runs": records(grok_sc),
                                  "configs_disagreeing_any_labels": [dict(zip(KEY_NAMES, k)) | {"labels": labs[k]}
                                                                     for k in disagree_keys]},
        "q7_neighbour": neigh,
        "q7_by_label": {"rates": by_label, "late_events_per_run": n_ev_by_label,
                        "late_activation_only_events_per_run": n_evact_by_label, "label_differences": region_diffs,
                        "by_final_test_acc": {"by_bin_and_label": records(by_acc), "by_bin": records(by_acc_all)},
                        "late_events_by_order": late_order, "late_events_by_kind": late_kind,
                        "checkpoints_and_t99_by_label": ckpt_after},
        "q7_secondary_before_generalisation": sec_b,
        "q8": {"literal": q8_literal, "amended": q8, "points": records(q8_table)},
        "q8_paired": {"summary": paired, "pairs_on_grid": records(pair_table)},
        "per_run": records(g[["object_id", "cfg", *KEY, "seed", "source", "label", "label_score", "final_train_acc",
                              "final_test_acc", "max_test_acc", "t_train_acc99", "t_test_acc90", "grok_ratio",
                              "ri_run_id", "n_points", "replay_fidelity", "followup_reason", "n_events",
                              "n_activations_all", "n_strong_any", "n_late", "n_late_act", "n_late_before_gen",
                              "n_late_act_before_gen", "m_late", "m_late_act", "m_late_before_gen",
                              "first_late_step_lo", "first_late_step", "first_late_peak", "max_peak_late"]]
                           .rename(columns=dict(zip(KEY, KEY_NAMES)))),
        "chart_data": {
            "late_fraction_by_label": {lab: v["pct"] for lab, v in bl.items()},
            "same_config_fraction": [{"cfg": r["cfg"], "grok": r["k_grokking"] / r["n_grokking"],
                                      "control": r["k_control"] / r["n_control"]} for r in cfg_rows],
            "q8_points": [{"cfg": r.cfg, "v0": r.v0_frac, "v1": r.v1_frac} for r in q8pts.itertuples()],
            "v1_t90_of_v0_mem_converted": paired["v1_t_test_acc90_of_converted"],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(jsonable(out), fh, indent=1)
    print(json.dumps(jsonable({"items": items, "checks": checks}), indent=1)[:20000])
    print("wrote", args.out)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
