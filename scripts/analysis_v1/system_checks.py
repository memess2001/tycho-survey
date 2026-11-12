"""Sky v1, pre-registered system checks 9 and 10, incl. Amendment 1 items H and I.

Pre-registration: docs/results/v1_preregistration.md (criteria fixed before the run; Amendment 1 written after the
launch, before v1 data were read).

Check 9 (follow-up balance, grokking-v1)
  Registered: "Follow-ups in grokking-v1: no label takes more than 50 % of them while other labels still have
  candidates." Amendment 1 H: it can fail by design (the matched and boundary streams do not look at labels); it is
  reported literally over the request order (followups.rowid), with this as a declared deviation.
  The release has no followups table. Order used: ri_runs.ri_run_id of the grokking-v1 follow-up curves, i.e. the
  order in which follow-ups were RECORDED (completion order), a proxy for the request order. The stream of each
  follow-up is read from its curve's meta["followup_reason"]. Label = the latest classification label (checked:
  every label was assigned before the follow-up was recorded, and the label named in the reason agrees).
  Candidates of a label at time t: grokking-v1 objects with that label, finished (objects.finished_at <= t) and not
  yet followed up (no follow-up recorded at or before t; the object followed up at t itself is not a candidate).
  Literal evaluation over the sequence i = 1..N (t = created_at of follow-up i): violation at i iff the largest
  cumulative label share of follow-ups 1..i exceeds 50 % and at least one other label has >= 1 candidate at t.
  Reported: first violation, every spell of consecutive violations (length in follow-ups and seconds, the largest
  share and the largest surplus of the majority label), cumulative label shares over time, stream mix by label.
  Sensitivity (same criterion, other readings of "the order"):
    (S1) request batches: the follow-ups were requested in scheduler calls of 70 (daemon pace
         ceil(10,000 x 600 s / 86,400 s) = 70 per 600 s call); the recorded follow-ups fall into blocks of 70
         separated by gaps of > 300 s, and every block is complete before the next starts, so the cumulative label
         counts at the END of each block are exact in request order. Candidates are evaluated at the call time,
         estimated with the daemon's period (t_daemon + 600 s x k, t_daemon = the smallest offset at which every
         call's picks had finished; checked against the first completion of each block), and at the bracket ends:
         lo = the latest finish among the call's picks (and not before the previous call's lo), hi = the call's
         first completion. Candidates (finished, not in an earlier call) grow with time inside the bracket, so the
         count at lo is a lower bound.
    (S2) reconstructed request order: within each block the region stream's round robin (anomaly, matched,
         boundary, stratified; first turn rotating with the region's follow-up count, scheduler.py) applied to the
         block's items of each stream in their recorded order.
    (S3) per-call share: a label holding > 50 % of one call's 70 picks while other labels had candidates.
    (S4) final share over all follow-ups.

Check 10 (anomaly rate)
  Registered: "Anomalies flagged: about 1 % per (region, label) group; each one is looked at by hand."
  Amendment 1 I: the scheduler scores only grokking-v1, so the rate is evaluated offline on the final catalog for
  every v1 (region, label) group; the per-group result is reported literally, the overall rate next to it as a
  declared deviation.
  Offline flags: tycho.analysis.anomaly.anomaly_scores on DR1 objects grouped by (region, label) (done, labelled,
  not diverged, as the scheduler groups them), summary = the object's row of objects.parquet (it holds every key
  anomaly.summary_features reads); flagged iff score >= anomaly.THRESHOLD (8.0). Groups smaller than MIN_GROUP (20)
  are not scored.
  "About 1 %" (ambiguous, interpretation chosen): a group's flagged count is consistent with a 1 % rate, i.e. the
  exact two-sided binomial test of k flagged of n scored against p = 0.01 has p >= 0.05. Alternatives reported:
  (a) one-sided, too many only (P(X >= k) >= 0.05); (b) a fixed band 0.5 % <= k/n <= 2 %.
  Overall rate: flagged / scored, with a Clopper-Pearson interval and a bootstrap over configurations (2,000
  resamples, fixed seed; flags held fixed) because flags cluster by configuration.
  Every flagged object: label, deciding feature value vs the group median (feature units and raw), and a look at
  its light curve (final losses and accuracies, minimum test loss, maximum test accuracy, test-loss bump, drops,
  task metrics), with a hand-written note.
  Follow-ups whose reason starts with "anomaly": counted over all follow-up curves and compared with the offline
  flags; the scheduler's view is reconstructed at each call time (objects finished by then, scored by label group)
  to see whether any eligible object was flagged when the scheduler looked.

Usage (cwd = repository root):
    PYTHONPATH=. python scripts/analysis_v1/system_checks.py [--release releases/DR1] [--extracted DIR] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import tycho
from tycho.analysis import anomaly as A
from tycho.analysis.summarize import acc_threshold

REPO = Path(__file__).resolve().parents[2]
SCRATCH = REPO / "work"  # extracted releases (git-ignored); see README.md
REGION_G = "grokking-v1"
LABELS_G = ("grokking", "memorization", "smooth")
STREAMS = ("anomaly", "matched", "boundary", "stratified")  # scheduler.REGION_STREAMS
CALL_PERIOD_S = 600.0  # survey.followup_every_s (configs/tycho.example.yaml; start_v1.sh keeps it)
PACE = math.ceil(10_000 * CALL_PERIOD_S / 86_400.0)  # followups_per_day = 10,000 (start_v1.sh) -> 70 per call
BATCH_GAP_S = 200.0
N_BOOT = 2000
BOOT_SEED = 20261112


# ---------------------------------------------------------------------------------------------- helpers
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NA or x is None:
        return None
    return x


def summary_of(row: dict) -> dict:
    """An objects.parquet row as a summary mapping (pd.NA -> None, which anomaly._num turns into NaN)."""
    return {k: (None if v is pd.NA else v) for k, v in row.items()}


def curve_header(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["header"]))


def stream_of(reason: str | None) -> str:
    if not reason:
        return "none"
    head = reason.split(":", 1)[0].strip()
    return "boundary" if head.startswith("boundary") else head


def label_in_reason(reason: str) -> str | None:
    """The label the scheduler named in a follow-up reason (None for boundary reasons, which name none)."""
    s = stream_of(reason)
    if s == "matched":
        return reason.rsplit("this seed", 1)[1].strip()
    if s == "stratified":
        return reason.split(":", 1)[1].strip().split(" ")[0]
    if s == "random_control":
        return reason[reason.rfind("(") + 1: reason.rfind(")")]
    if s == "anomaly":
        return reason.split("unlike the other", 1)[1].strip().split(" ")[0]
    return None


def spells(flags: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs (inclusive) of runs of True."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def evaluate_sequence(labels: list[str], times: np.ndarray, done_at_or_before, followed_before) -> dict:
    """The literal criterion over an ordered sequence of follow-ups.

    labels[i], times[i]: label and time of follow-up i (0-based). candidates(i, lab) = objects of label lab finished
    by times[i] and not followed up among follow-ups 0..i (computed by the two callables)."""
    n = len(labels)
    counts = {lab: 0 for lab in LABELS_G}
    rows = []
    for i in range(n):
        counts[labels[i]] += 1
        k = i + 1
        shares = {lab: counts[lab] / k for lab in LABELS_G}
        top = max(LABELS_G, key=lambda lab: (shares[lab], lab))
        cands = {lab: int(done_at_or_before(times[i], lab) - followed_before(i, lab)) for lab in LABELS_G}
        others = {lab: c for lab, c in cands.items() if lab != top}
        viol = shares[top] > 0.5 and any(c > 0 for c in others.values())
        rows.append({"i": k, "t": float(times[i]), "label": labels[i], **{f"n_{lab}": counts[lab] for lab in LABELS_G},
                     **{f"share_{lab}": shares[lab] for lab in LABELS_G},
                     **{f"cand_{lab}": cands[lab] for lab in LABELS_G},
                     "top_label": top, "top_share": shares[top], "violation": bool(viol)})
    flags = np.array([r["violation"] for r in rows], dtype=bool)
    sp = []
    for a, b in spells(flags):
        seg = rows[a: b + 1]
        labs = sorted({r["top_label"] for r in seg})
        best = max(seg, key=lambda r: r["top_share"])
        surplus = max(2 * r[f"n_{r['top_label']}"] - r["i"] for r in seg)
        sp.append({"first_i": rows[a]["i"], "last_i": rows[b]["i"], "n_followups": b - a + 1,
                   "t_first": rows[a]["t"], "t_last": rows[b]["t"], "duration_s": rows[b]["t"] - rows[a]["t"],
                   "majority_labels": labs, "max_share": best["top_share"], "max_share_at_i": best["i"],
                   "max_share_label": best["top_label"],
                   "max_excess_pp": 100 * (best["top_share"] - 0.5),
                   "max_surplus_followups": int(surplus),
                   "other_label_candidates_at_first_i": {lab: rows[a][f"cand_{lab}"] for lab in LABELS_G
                                                         if lab != rows[a]["top_label"]}})
    first = rows[int(np.argmax(flags))] if flags.any() else None
    return {"rows": rows, "n_violating": int(flags.sum()), "n": n, "first_violation": first, "spells": sp,
            "violating_excluding_first_k": {str(m): int(flags[m:].sum()) for m in (1, 2, 5, 10, 70)}}


# ---------------------------------------------------------------------------------------------- check 9
def check9(objects: pd.DataFrame, runs: pd.DataFrame, extracted: Path) -> dict:
    t_start = float(objects["created_at"].min())  # first top_up of the daemon (proxy for its start)
    fu_all = runs[runs["kind"] == "followup"].merge(objects[["object_id", "region"]], on="object_id")
    region_counts = fu_all["region"].value_counts().to_dict()

    reasons_all = {}
    for rr in runs.itertuples():
        h = curve_header(extracted / rr.ricurve_file)
        reasons_all[int(rr.ri_run_id)] = (h["meta"].get("followup_reason"), h["meta"].get("kind"))

    g = objects[objects["region"] == REGION_G].copy()
    fu = runs[runs["kind"] == "followup"].merge(
        g[["object_id", "label", "finished_at", "labelled_at", "seed", "source", "random_control", "diverged",
           "task_cfg_json", "train_lr", "train_weight_decay"]], on="object_id")
    fu = fu.sort_values("ri_run_id").reset_index(drop=True)
    fu["reason"] = [reasons_all[int(i)][0] for i in fu["ri_run_id"]]
    fu["curve_kind"] = [reasons_all[int(i)][1] for i in fu["ri_run_id"]]
    fu["stream"] = fu["reason"].map(stream_of)
    fu["stream_detail"] = fu["reason"].map(lambda r: r.split(":", 1)[0])
    fu["label_reason"] = fu["reason"].map(label_in_reason)
    fu["t"] = fu["created_at"] - t_start
    fu["i"] = np.arange(1, len(fu) + 1)

    sanity = {
        "tycho_file": tycho.__file__,
        "followup_runs_by_region": region_counts,
        "grokking_objects": int(len(g)),
        "grokking_followups": int(len(fu)),
        "grokking_objects_followed_up_once": int(fu["object_id"].nunique()),
        "curve_kind_counts": fu["curve_kind"].value_counts().to_dict(),
        "diverged_followed_up": int((fu["diverged"] == True).sum()),  # noqa: E712
        "followed_up_before_finished": int((fu["finished_at"] > fu["created_at"]).sum()),
        "labelled_after_followup_recorded": int((fu["labelled_at"] > fu["created_at"]).sum()),
        "reason_label_known": int(fu["label_reason"].notna().sum()),
        "reason_label_disagrees_with_latest_label": int(((fu["label_reason"].notna())
                                                         & (fu["label_reason"] != fu["label"])).sum()),
        "ri_run_id_order_equals_created_at_order": bool(fu["created_at"].is_monotonic_increasing),
        "followup_reason_starts_with_anomaly_all_regions": int(sum(1 for r, k in reasons_all.values()
                                                                   if r and r.startswith("anomaly"))),
        "insitu_curves_with_followup_reason": int(sum(1 for r, k in reasons_all.values() if r and k != "replay")),
    }

    # batches (scheduler calls)
    gap = fu["t"].diff().fillna(np.inf)
    fu["batch"] = np.cumsum(gap.values > BATCH_GAP_S)
    # Call k (1-based) happened after every one of its picks had finished and before its first completion.
    # Periodic model: the daemon runs the follow-up task at its start and then every 600 s, so call k is at
    # t_daemon + 600 k (the call at t_daemon found nothing done). The smallest offset t_daemon consistent with every
    # call's picks having finished is used; the model is checked against the upper ends.
    grp_b = list(fu.groupby("batch"))
    latest = {int(b): float((sub["finished_at"] - t_start).max()) for b, sub in grp_b}
    first_c = {int(b): float(sub["t"].min()) for b, sub in grp_b}
    offset = max(latest[b] - CALL_PERIOD_S * b for b in latest)
    offset_max = min(first_c[b] - CALL_PERIOD_S * b for b in first_c)
    batches = []
    lo_prev = 0.0
    for b, sub in grp_b:
        lo = max(latest[b], lo_prev)  # calls are ordered in time
        lo_prev = lo
        call_est = offset + CALL_PERIOD_S * b
        batches.append({"batch": int(b), "n": int(len(sub)), "first_i": int(sub["i"].min()),
                        "last_i": int(sub["i"].max()), "t_first_completion": first_c[b],
                        "t_last_completion": float(sub["t"].max()), "bracket_lo": lo,
                        "call_time_est": call_est, "latest_finish_of_picks": latest[b],
                        "picks_finished_by_call_est": bool(latest[b] <= call_est),
                        "call_est_in_bracket": bool(lo <= call_est <= first_c[b]),
                        "lag_first_completion_after_call_est_s": float(first_c[b] - call_est)})
    sanity["call_model_offset_s"] = {"used": offset, "max_consistent": offset_max,
                                     "consistent": bool(offset <= offset_max)}
    sanity["batch_sizes"] = [b["n"] for b in batches]
    sanity["pace_per_call"] = PACE
    sanity["batches_complete_before_next"] = all(
        batches[k]["t_last_completion"] < batches[k + 1]["t_first_completion"] for k in range(len(batches) - 1))

    fin = {lab: np.sort((g.loc[g["label"] == lab, "finished_at"] - t_start).values) for lab in LABELS_G}

    def done_by(t: float, lab: str) -> int:
        return int(np.searchsorted(fin[lab], t, side="right"))

    # ---- literal: completion order
    lab_seq = fu["label"].tolist()
    cum = {lab: np.cumsum(np.array(lab_seq) == lab) for lab in LABELS_G}
    # followed up among 0..i AND finished by t_i: every followed-up object finished before its run was recorded
    lit = evaluate_sequence(lab_seq, fu["t"].values, done_by, lambda i, lab: int(cum[lab][i]))

    # ---- S1: batch ends (exact in request order at call boundaries)
    s1 = []
    for bt in batches:
        k_end = bt["last_i"]
        counts = {lab: int(cum[lab][k_end - 1]) for lab in LABELS_G}
        prev_counts = {lab: int(cum[lab][bt["first_i"] - 2]) if bt["first_i"] > 1 else 0 for lab in LABELS_G}
        shares = {lab: counts[lab] / k_end for lab in LABELS_G}
        top = max(LABELS_G, key=lambda lab: shares[lab])
        batch_labels = Counter(fu.loc[fu["batch"] == bt["batch"], "label"])
        rec = {"batch": bt["batch"], "after_i": k_end, **{f"n_{lab}": counts[lab] for lab in LABELS_G},
               **{f"share_{lab}": shares[lab] for lab in LABELS_G}, "top_label": top, "top_share": shares[top],
               **{f"batch_n_{lab}": int(batch_labels.get(lab, 0)) for lab in LABELS_G}}
        for name, tt in (("lo", bt["bracket_lo"]), ("call_est", bt["call_time_est"]),
                         ("hi", bt["t_first_completion"])):
            before = {lab: done_by(tt, lab) - prev_counts[lab] for lab in LABELS_G}
            after = {lab: before[lab] - int(batch_labels.get(lab, 0)) for lab in LABELS_G}
            rec[f"cand_before_call_{name}"] = before
            rec[f"cand_left_after_call_{name}"] = after
        left_lo = rec["cand_left_after_call_lo"]
        left_est = rec["cand_left_after_call_call_est"]
        rec["violation_lo"] = bool(shares[top] > 0.5 and any(left_lo[lab] > 0 for lab in LABELS_G if lab != top))
        rec["violation_call_est"] = bool(shares[top] > 0.5 and any(left_est[lab] > 0 for lab in LABELS_G
                                                                    if lab != top))
        # S3 per call
        btop, bcount = batch_labels.most_common(1)[0]
        before_est = rec["cand_before_call_call_est"]
        rec["call_top_label"] = btop
        rec["call_top_share"] = bcount / bt["n"]
        rec["call_other_labels_had_candidates"] = {lab: before_est[lab] - int(batch_labels.get(lab, 0)) > 0
                                                   for lab in LABELS_G if lab != btop}
        rec["call_violation"] = bool(bcount / bt["n"] > 0.5
                                     and any(rec["call_other_labels_had_candidates"].values()))
        s1.append(rec)

    # ---- S2: reconstructed request order (round robin of the streams within each call)
    recon = []
    n_before = 0
    for bt in batches:
        sub = fu[fu["batch"] == bt["batch"]]
        per = {s: [] for s in STREAMS}
        for r in sub.itertuples():
            s = "stratified" if r.stream == "random_control" else r.stream
            per[s].append(r.Index)
        start = n_before % len(STREAMS)
        order = [per[s] for s in STREAMS[start:] + STREAMS[:start]]
        pos = [0] * len(order)
        while any(pos[j] < len(order[j]) for j in range(len(order))):
            for j in range(len(order)):
                if pos[j] < len(order[j]):
                    recon.append(order[j][pos[j]])
                    pos[j] += 1
        n_before += len(sub)
    fr = fu.loc[recon].reset_index(drop=True)
    # time of a reconstructed request = estimated call time (all picks of a call are made at one instant)
    call_t = {bt["batch"]: bt["call_time_est"] for bt in batches}
    fr_t = np.array([call_t[b] for b in fr["batch"]])
    cum_r = {lab: np.cumsum(np.array(fr["label"]) == lab) for lab in LABELS_G}
    s2 = evaluate_sequence(fr["label"].tolist(), fr_t, done_by, lambda i, lab: int(cum_r[lab][i]))

    # ---- S4 final share
    final_counts = fu["label"].value_counts().to_dict()
    final = {"counts": final_counts, "shares": {k: v / len(fu) for k, v in final_counts.items()},
             "max_share": max(final_counts.values()) / len(fu)}

    # ---- stream mix by label
    ct = pd.crosstab(fu["stream"], fu["label"], margins=True)
    ct_detail = pd.crosstab(fu["stream_detail"], fu["label"], margins=True)
    stream_mix = {"by_stream_label": {s: {lab: int(ct.loc[s, lab]) for lab in ct.columns} for s in ct.index},
                  "by_stream_detail_label": {s: {lab: int(ct_detail.loc[s, lab]) for lab in ct_detail.columns}
                                             for s in ct_detail.index},
                  "label_blind_share": float(fu["stream"].isin(["matched", "boundary"]).mean()),
                  "label_blind_n": int(fu["stream"].isin(["matched", "boundary"]).sum()),
                  "per_batch": {int(b): sub["stream"].value_counts().to_dict() for b, sub in fu.groupby("batch")},
                  "per_batch_label": {int(b): sub["label"].value_counts().to_dict()
                                      for b, sub in fu.groupby("batch")},
                  "per_batch_stream_label": {int(b): {s: x["label"].value_counts().to_dict()
                                                      for s, x in sub.groupby("stream")}
                                             for b, sub in fu.groupby("batch")},
                  "label_blind_share_by_batch": {int(b): float(sub["stream"].isin(["matched", "boundary"]).mean())
                                                 for b, sub in fu.groupby("batch")}}
    # label composition of all finished grokking-v1 objects at each call (what a label-blind pick would mirror)
    avail = []
    for bt in batches:
        d = {lab: done_by(bt["call_time_est"], lab) for lab in LABELS_G}
        tot = sum(d.values())
        avail.append({"batch": bt["batch"], "finished": d, "finished_total": tot,
                      "finished_shares": {lab: d[lab] / tot if tot else None for lab in LABELS_G}})

    per_run = fu[["i", "ri_run_id", "object_id", "t", "batch", "label", "label_reason", "stream", "stream_detail",
                  "reason", "seed", "source", "random_control"]].to_dict("records")
    for rec, lr in zip(per_run, lit["rows"]):
        rec.update({k: lr[k] for k in lr if k.startswith(("n_", "share_", "cand_")) or k in ("violation",)})
    return {"sanity": sanity, "batches": batches, "literal_completion_order": {k: v for k, v in lit.items()
                                                                              if k != "rows"},
            "S1_batch_ends": s1, "S2_reconstructed_request_order": {k: v for k, v in s2.items() if k != "rows"},
            "S2_rows_compact": [{k: r[k] for k in ("i", "label", "top_label", "top_share", "violation")}
                                for r in s2["rows"]],
            "S4_final": final, "stream_mix": stream_mix, "finished_composition_at_calls": avail,
            "per_run": per_run, "t_start_unix": t_start}


# ---------------------------------------------------------------------------------------------- check 10
def lc_look(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    step = d["step"].astype(float)
    out = {"n_points": int(len(step)), "steps_total": float(step[-1])}
    for k in ("train_loss", "test_loss", "train_acc", "test_acc"):
        v = d[k].astype(float)
        out[f"final_{k}"] = float(v[-1])
        fin = np.isfinite(v)
        if fin.any():
            out[f"min_{k}"] = float(np.nanmin(v))
            out[f"max_{k}"] = float(np.nanmax(v))
            out[f"step_max_{k}"] = float(step[int(np.nanargmax(v))])
            out[f"step_min_{k}"] = float(step[int(np.nanargmin(v))])
    ta = d["train_acc"].astype(float)
    hit = np.flatnonzero(np.isfinite(ta) & (ta >= 0.9))
    out["first_step_train_acc_ge_0p9"] = float(step[hit[0]]) if hit.size else None
    out["train_acc_at_step0"] = float(ta[0])
    tl = d["test_loss"].astype(float)
    runmin =np.minimum.accumulate(np.where(np.isfinite(tl), tl, np.inf))
    rise = tl - np.concatenate([[tl[0]], runmin[:-1]])
    j = int(np.nanargmax(rise))
    out["test_loss_initial"] = float(tl[0])
    out["test_loss_bump_peak_step"] = float(step[j])
    out["test_loss_bump_peak"] = float(tl[j])
    out["test_loss_min_before_peak"] = float(runmin[j - 1]) if j > 0 else float(tl[0])
    out["test_loss_bump_rel"] = float(max(0.0, rise[j]) / max(tl[0], 1e-12))
    for k in ("x_kl_true", "x_support_frac"):
        if k in d:
            v = d[k].astype(float)
            fin = np.isfinite(v)
            if fin.any():
                out[f"final_{k}"] = float(v[fin][-1])
                out[f"min_{k}"] = float(np.nanmin(v))
    # compact curve for charts: ~20 log-spaced points
    idx = np.unique(np.round(np.geomspace(1, len(step), 20)).astype(int) - 1)
    out["curve"] = {k: [float(x) for x in d[k][idx]] for k in ("step", "train_loss", "test_loss", "train_acc",
                                                               "test_acc") if k in d}
    return out


def markov_bayes_accuracy(couplings: str, window: int = 6) -> dict:
    """Bayes accuracy of the true conditional under uniform windows and under the chain's exact stationary
    distribution of windows (64-state Markov chain), and the chain's stationary token frequency."""
    coup = []
    for part in couplings.split(";"):
        lags, beta = part.split(":")
        coup.append(([int(x) for x in lags.split(",")], float(beta)))
    W = window
    states = np.array([[1 if (w >> (W - 1 - j)) & 1 else -1 for j in range(W)] for w in range(2 ** W)])  # lag 1..W
    h = np.array([sum(beta * np.prod([s[lag - 1] for lag in lags]) for lags, beta in coup) for s in states])
    p1 = 1 / (1 + np.exp(-2 * h))
    acc_state = np.maximum(p1, 1 - p1)
    index = {tuple(s): i for i, s in enumerate(states)}
    P = np.zeros((2 ** W, 2 ** W))
    for i, s in enumerate(states):
        for new, pr in ((1, p1[i]), (-1, 1 - p1[i])):
            P[i, index[(new, *s[:-1])]] += pr
    pi = np.full(2 ** W, 1 / 2 ** W)
    for _ in range(20000):
        nxt = pi @ P
        if np.abs(nxt - pi).max() < 1e-15:
            pi = nxt
            break
        pi = nxt
    return {"bayes_acc_uniform_windows": float(acc_state.mean()), "bayes_acc_chain_stationary": float(pi @ acc_state),
            "chain_P_token1": float(pi @ (states[:, 0] == 1))}


def check10(objects: pd.DataFrame, runs: pd.DataFrame, extracted: Path, fu_reasons: pd.DataFrame,
            t_start: float, batches: list[dict], followed_batch: dict) -> dict:
    o = objects.copy()
    o["cfg"] = o["task_cfg_json"].map(json.loads)
    o["config_key"] = (o["region"] + "|" + o["task_cfg_json"] + "|" + o["train_lr"].astype(str) + "|"
                       + o["train_weight_decay"].astype(str) + "|" + o["train_steps"].astype(str))
    o = o[(o["label"].notna()) & (o["diverged"] != True)].copy()  # noqa: E712  (scheduler grouping)
    o["score"] = np.nan
    o["feature"] = None
    groups, centres = [], {}
    for (reg, lab), grp in o.groupby(["region", "label"], sort=True):
        recs = [summary_of(r) for r in grp.to_dict("records")]
        sc = A.anomaly_scores(recs)
        o.loc[grp.index, "score"] = [s[0] if s else np.nan for s in sc]
        o.loc[grp.index, "feature"] = [s[1] if s else None for s in sc]
        feats = [A.summary_features(r) for r in recs]
        cen = {}
        for name, (_, floor) in A.FEATURES.items():
            vals = [f[name] for f in feats if math.isfinite(f[name])]
            if len(vals) >= A.MIN_GROUP:
                med = statistics.median(vals)
                mad = statistics.median(abs(v - med) for v in vals)
                cen[name] = {"median": med, "mad": mad, "scale": 1.4826 * max(mad, floor), "n_finite": len(vals),
                             "mad_floored": mad < floor}
        centres[(reg, lab)] = cen
        n = len(grp)
        n_scored = sum(s is not None for s in sc)
        flagged = [s for s in sc if s is not None and s[0] >= A.THRESHOLD]
        k = len(flagged)
        rec = {"region": reg, "label": lab, "n": n, "n_scored": n_scored, "n_flagged": k,
               "frac_flagged": k / n_scored if n_scored else None,
               "deciding_features_flagged": dict(Counter(f for _, f in flagged)),
               "deciding_features_all": dict(Counter(s[1] for s in sc if s is not None)),
               "top5_scores": sorted([float(s[0]) for s in sc if s is not None])[-5:],
               "n_within_1_of_threshold": int(sum(1 for s in sc if s is not None and A.THRESHOLD - 1 <= s[0]
                                                  < A.THRESHOLD)),
               "n_configurations": int(grp["config_key"].nunique())}
        if n_scored:
            bt = binomtest(k, n_scored, 0.01)
            rec["binom_two_sided_p"] = float(bt.pvalue)
            rec["binom_p_too_many"] = float(binomtest(k, n_scored, 0.01, alternative="greater").pvalue)
            rec["binom_p_too_few"] = float(binomtest(k, n_scored, 0.01, alternative="less").pvalue)
            ci = bt.proportion_ci(confidence_level=0.95, method="exact")
            rec["frac_ci95_clopper_pearson"] = [float(ci.low), float(ci.high)]
            rec["expected_at_1pct"] = 0.01 * n_scored
            rec["about_1pct_primary"] = bool(bt.pvalue >= 0.05)
            rec["about_1pct_alt_a_not_too_many"] = bool(rec["binom_p_too_many"] >= 0.05)
            rec["about_1pct_alt_b_band"] = bool(0.005 <= k / n_scored <= 0.02)
            rec["min_nonzero_frac"] = 1 / n_scored
        else:
            rec.update({"about_1pct_primary": None, "about_1pct_alt_a_not_too_many": None,
                        "about_1pct_alt_b_band": None, "note": f"not scored: group smaller than MIN_GROUP = "
                                                               f"{A.MIN_GROUP}"})
        rec["feature_centres"] = cen
        groups.append(rec)

    scored = o[o["score"].notna()]
    flag = scored[scored["score"] >= A.THRESHOLD]
    n_s, k_s = len(scored), len(flag)
    bt = binomtest(k_s, n_s, 0.01)
    ci = bt.proportion_ci(confidence_level=0.95, method="exact")
    # bootstrap over configurations (flags fixed)
    rng = np.random.default_rng(BOOT_SEED)
    by_cfg = scored.groupby("config_key").agg(n=("score", "size"), k=("score", lambda s: int((s >= A.THRESHOLD)
                                                                                         .sum())))
    nn, kk = by_cfg["n"].values, by_cfg["k"].values
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(nn), len(nn))
        boots.append(kk[idx].sum() / nn[idx].sum())
    overall = {"n_objects": int(len(objects)), "n_grouped": int(len(o)), "n_scored": n_s, "n_flagged": k_s,
               "frac_of_scored": k_s / n_s, "frac_of_all_objects": k_s / len(objects),
               "ci95_clopper_pearson": [float(ci.low), float(ci.high)], "binom_two_sided_p_vs_1pct": float(bt.pvalue),
               "n_unscored": int(len(o) - n_s), "unscored_groups": [f"{g['region']}/{g['label']} ({g['n']})"
                                                                    for g in groups if not g["n_scored"]],
               "bootstrap_configs": {"n_configurations": int(len(nn)), "n_resamples": N_BOOT, "seed": BOOT_SEED,
                                     "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]},
               "n_configurations_with_flags": int((kk > 0).sum()),
               "n_groups_scored": int(sum(1 for g in groups if g["n_scored"])),
               "n_groups_primary_ok": int(sum(1 for g in groups if g["about_1pct_primary"])),
               "n_groups_alt_a_ok": int(sum(1 for g in groups if g["about_1pct_alt_a_not_too_many"])),
               "n_groups_alt_b_ok": int(sum(1 for g in groups if g["about_1pct_alt_b_band"]))}

    # ---- every flagged object
    group_medians = (o.groupby(["region", "label"])[["final_train_loss", "final_test_loss", "final_train_acc",
                                                     "final_test_acc", "test_bump", "n_drops", "t_train_acc90",
                                                     "t_test_acc90"]].median())
    flagged_rows = []
    bayes_cache = {}
    for r in flag.sort_values(["region", "label", "score"], ascending=[True, True, False]).itertuples():
        reg, lab, feat = r.region, r.label, r.feature
        cen = centres[(reg, lab)][feat]
        f_val = A.summary_features(summary_of(objects.loc[r.Index].to_dict()))[feat]
        raw_key = {"log_t_train_acc90": "t_train_acc90", "log_t_test_acc90": "t_test_acc90",
                   "log_final_train_loss": "final_train_loss", "log_final_test_loss": "final_test_loss"}.get(feat, feat)
        grp = o[(o["region"] == reg) & (o["label"] == lab)]
        raw_val = objects.loc[r.Index, raw_key]
        raw_med = grp[raw_key].median()
        look = lc_look(extracted / r.lightcurve_file)
        cfg = r.cfg
        rec = {"object_id": r.object_id, "region": reg, "label": lab, "seed": int(r.seed), "config": cfg,
               "train_lr": float(r.train_lr), "train_steps": int(r.train_steps), "source": r.source,
               "score": float(r.score), "feature": feat, "feature_value": f_val, "group_median": cen["median"],
               "group_mad": cen["mad"], "group_scale": cen["scale"], "mad_floored": cen["mad_floored"],
               "raw_key": raw_key, "raw_value": None if pd.isna(raw_val) else float(raw_val),
               "raw_group_median": None if pd.isna(raw_med) else float(raw_med),
               "raw_group_n_finite": int(grp[raw_key].notna().sum()), "raw_group_n": int(len(grp)),
               "group_medians": {k: float(v) for k, v in group_medians.loc[(reg, lab)].items() if pd.notna(v)},
               "summary": {k: (None if pd.isna(objects.loc[r.Index, k]) else float(objects.loc[r.Index, k]))
                           for k in ("final_train_loss", "final_test_loss", "final_train_acc", "final_test_acc",
                                     "t_train_acc90", "t_test_acc90", "test_bump", "test_bump_step",
                                     "test_bump_recovery", "n_drops", "largest_drop_step", "plateau_logfrac",
                                     "chance_acc", "max_train_acc")},
               "lightcurve": look,
               "followed_up": r.object_id in followed_batch, "followup_batch": followed_batch.get(r.object_id)}
        if reg == "markov-v1":
            key = cfg["couplings"]
            if key not in bayes_cache:
                bayes_cache[key] = markov_bayes_accuracy(key, int(cfg.get("window", 6)))
            rec["markov_bayes"] = bayes_cache[key]
            same = o[(o["region"] == reg) & (o["task_cfg_json"] == r.task_cfg_json)]
            rec["same_configuration"] = {"n": int(len(same)), "n_flagged": int((same["score"] >= A.THRESHOLD).sum()),
                                         "labels": same["label"].value_counts().to_dict()}
            twin_cfg = dict(cfg)
            twin_cfg["train_context"] = "iid" if cfg["train_context"] == "chain" else "chain"
            tw = objects[(objects["region"] == reg) & (objects["cfg_key"] == json.dumps(twin_cfg, sort_keys=True))
                         & (objects["seed"] == r.seed)]
            if len(tw):
                t = tw.iloc[0]
                rec["iid_twin"] = {"object_id": t["object_id"], "label": t["label"],
                                   "final_train_acc": float(t["final_train_acc"]),
                                   "final_test_acc": float(t["final_test_acc"]),
                                   "t_train_acc90": None if pd.isna(t["t_train_acc90"]) else float(t["t_train_acc90"])}
        else:
            same = o[(o["region"] == reg) & (o["task_cfg_json"] == r.task_cfg_json) & (o["train_lr"] == r.train_lr)]
            rec["same_configuration"] = {"n": int(len(same)), "n_flagged": int((same["score"] >= A.THRESHOLD).sum()),
                                         "labels": same["label"].value_counts().to_dict(),
                                         "raw_feature_values": sorted(float(x) for x in same[raw_key]),
                                         "final_test_acc_values": sorted(float(x) for x in same["final_test_acc"])}
            gg = grp.assign(n_in=grp["cfg"].map(lambda c: c.get("n")))
            rec["group_raw_feature_by_config"] = [
                {"n": int(n_in), "train_lr": float(lr), "count": int(len(x)), "median": float(x[raw_key].median()),
                 "min": float(x[raw_key].min()), "max": float(x[raw_key].max()),
                 "max_score": float(x["score"].max())}
                for (n_in, lr), x in gg.groupby(["n_in", "train_lr"])]
            rec["max_test_loss_over_initial"] = look["max_test_loss"] / look["test_loss_initial"]
        if raw_key.startswith("t_") and "acc" in raw_key:
            rec["acc_threshold"] = acc_threshold(float(raw_key[-2:]), float(objects.loc[r.Index, "chance_acc"]))
        rec["auto_description"] = describe(rec)
        rec["hand_note"] = HAND_NOTES.get((reg, lab, feat), "")
        flagged_rows.append(rec)

    # markov: which configurations can reach the (rescaled) 90 % train-accuracy threshold at all
    mk = objects[objects["region"] == "markov-v1"].copy()
    mk["couplings"] = mk["cfg_key"].map(lambda s: json.loads(s)["couplings"])
    mk["context"] = mk["cfg_key"].map(lambda s: json.loads(s)["train_context"])
    markov_table = []
    for (cp, ctx), x in mk.groupby(["couplings", "context"]):
        if cp not in bayes_cache:
            bayes_cache[cp] = markov_bayes_accuracy(cp)
        ch = float(x["chance_acc"].median())
        markov_table.append({"couplings": cp, "train_context": ctx, "n": int(len(x)),
                             "labels": x["label"].value_counts().to_dict(),
                             "chance_acc": ch, "threshold_90": acc_threshold(90, ch),
                             "max_train_acc_max": float(x["max_train_acc"].max()),
                             "final_train_acc_median": float(x["final_train_acc"].median()),
                             "bayes_acc_training_contexts": bayes_cache[cp]["bayes_acc_chain_stationary" if ctx == "chain"
                                                                            else "bayes_acc_uniform_windows"],
                             "n_reach_t_train_acc90": int(x["t_train_acc90"].notna().sum()),
                             "n_flagged": int((o.loc[o.index.intersection(x.index), "score"] >= A.THRESHOLD).sum())})

    # ---- anomaly follow-ups vs offline flags; the scheduler's view at each call
    n_anom = int(fu_reasons["reason"].fillna("").str.startswith("anomaly").sum())
    g = o[o["region"] == REGION_G]
    g_flag = g[g["score"] >= A.THRESHOLD]
    sched_view = []
    for bt in batches:
        row = {"batch": bt["batch"]}
        for name, tt in (("lo", bt["bracket_lo"]), ("call_est", bt["call_time_est"]),
                         ("hi", bt["t_first_completion"])):
            done = g[(g["finished_at"] - t_start) <= tt]
            n_el_flag, top, per_label = 0, [], {}
            for lab, grp in done.groupby("label"):
                sc = A.anomaly_scores([summary_of(x) for x in grp.to_dict("records")])
                per_label[lab] = {"n_done": int(len(grp)), "scored": bool(any(s is not None for s in sc))}
                for oid, s in zip(grp["object_id"], sc):
                    if s is None:
                        continue
                    top.append((float(s[0]), oid, lab, s[1]))
                    eligible = followed_batch.get(oid, 10 ** 9) >= bt["batch"]
                    if s[0] >= A.THRESHOLD and eligible:
                        n_el_flag += 1
            top.sort(reverse=True)
            row[name] = {"t": tt, "n_done": int(len(done)), "labels": per_label, "n_flagged_eligible": n_el_flag,
                         "max_score": top[0][0] if top else None,
                         "max_score_object": top[0][1] if top else None, "max_score_label": top[0][2] if top else None,
                         "max_score_feature": top[0][3] if top else None}
        sched_view.append(row)
    near = []
    for lab in LABELS_G:
        x = g[g["label"] == lab].sort_values("score", ascending=False).head(3)
        for r in x.itertuples():
            near.append({"label": lab, "object_id": r.object_id, "score": float(r.score), "feature": r.feature,
                         "final_test_acc": float(r.final_test_acc), "t_test_acc90": None if pd.isna(r.t_test_acc90)
                         else float(r.t_test_acc90), "test_bump": float(r.test_bump), "n_drops": int(r.n_drops)})
    anomaly_followups = {"n_followups_reason_anomaly": n_anom, "n_followups_grokking": int(len(fu_reasons)),
                         "offline_flagged_grokking_final_catalog": int(len(g_flag)),
                         "offline_flagged_grokking_ids": g_flag["object_id"].tolist(),
                         "grokking_top3_scores_per_label": near,
                         "offline_flags_by_region": flag["region"].value_counts().to_dict(),
                         "offline_flagged_followed_up": int(sum(1 for f in flagged_rows if f["followed_up"])),
                         "max_score_grokking_by_label": {lab: float(g.loc[g["label"] == lab, "score"].max())
                                                         for lab in LABELS_G},
                         "scheduler_view_at_calls": sched_view,
                         "max_flagged_eligible_any_call": int(max(max(v[k]["n_flagged_eligible"]
                                                                      for k in ("lo", "call_est", "hi"))
                                                                  for v in sched_view))}
    per_object = o[["object_id", "region", "label", "seed", "score", "feature"]].copy()
    per_object["flagged"] = per_object["score"] >= A.THRESHOLD
    return {"groups": groups, "overall": overall, "flagged": flagged_rows, "anomaly_followups": anomaly_followups,
            "markov_threshold_table": markov_table,
            "per_object_scores": per_object.to_dict("records"), "threshold": A.THRESHOLD, "min_group": A.MIN_GROUP,
            "features": list(A.FEATURES)}


def describe(rec: dict) -> str:
    lc, s = rec["lightcurve"], rec["summary"]
    feat = rec["feature"]
    if rec["raw_value"] is None:
        val = f"{rec['raw_key']} never reached"
    else:
        val = f"{rec['raw_key']} = {rec['raw_value']:.4g}"
    if "acc_threshold" in rec:
        val += f" (threshold: train accuracy >= {rec['acc_threshold']:.4f}, rescaled from chance " \
               f"{s['chance_acc']:.3f})"
    if rec["raw_group_n_finite"] < rec["raw_group_n"]:
        med = (f"only {rec['raw_group_n_finite']} of the {rec['raw_group_n']} group members reach it; the other "
               f"{rec['raw_group_n'] - rec['raw_group_n_finite']} never do and sit at the censored value "
               f"log10(2 x steps_total), which is the group median ({rec['group_median']:.3f}), with MAD "
               f"{rec['group_mad']:.3g}") if rec["raw_group_n_finite"] * 2 < rec["raw_group_n"] else \
              (f"group median {rec['raw_group_median']:.4g} over the {rec['raw_group_n_finite']} of "
               f"{rec['raw_group_n']} members where it is defined")
    else:
        med = f"group median {rec['raw_group_median']:.4g} (MAD {rec['group_mad']:.3g})"
    txt = (f"{rec['label']} in {rec['region']} (seed {rec['seed']}): z = {rec['score']:.1f} on {feat} "
           f"(value {rec['feature_value']:.3f} vs group median {rec['group_median']:.3f}, scale "
           f"{rec['group_scale']:.3f}{', MAD below floor' if rec['mad_floored'] else ''}); {val}, {med}. "
           f"Light curve ({lc['n_points']} points to step {lc['steps_total']:.0f}): final train/test loss "
           f"{lc['final_train_loss']:.4g}/{lc['final_test_loss']:.4g}, final train/test acc "
           f"{lc['final_train_acc']:.4f}/{lc['final_test_acc']:.4f}, max train acc {lc['max_train_acc']:.4f}, "
           f"min test loss {lc['min_test_loss']:.4g} at step {lc['step_min_test_loss']:.0f}; largest test-loss rise "
           f"from {lc['test_loss_min_before_peak']:.4g} to {lc['test_loss_bump_peak']:.4g} at step "
           f"{lc['test_loss_bump_peak_step']:.0f} (test_bump {s['test_bump']:.3f}); n_drops {s['n_drops']:.0f}.")
    if "final_x_kl_true" in lc:
        txt += f" Final kl_true {lc['final_x_kl_true']:.2g} nats."
    if "final_x_support_frac" in lc:
        txt += f" Final support_frac {lc['final_x_support_frac']:.3g}."
    return txt


HAND_NOTES: dict[tuple[str, str, str], str] = {
    ("markov-v1", "smooth", "log_t_train_acc90"):
        "Looked at by hand: a configuration effect, not an odd curve. All 12 seeds of couplings 1:1.0;1,2,3:1.0 "
        "with chain contexts are flagged, and no other markov-v1 object. They are the only markov-v1 runs whose "
        "train accuracy reaches the rescaled 90 % threshold (0.9505 for chance 0.505): under the chain's own "
        "correlated contexts the true conditional is nearly deterministic (exact stationary Bayes accuracy 0.965), "
        "while with uniform contexts (the iid twin, same couplings and seed) it is 0.741 and the twins end at "
        "0.733-0.750. Every other markov configuration has a Bayes accuracy below its threshold "
        "(markov_threshold_table). The light curve itself is an ordinary smooth fit. Train accuracy passes 0.9 "
        "within the first 8 steps (already at step 0 in 4 of the 12 seeds, at 0.91-0.92). It reaches 0.9505 at step 142 or 162 and ends at "
        "0.966-0.968 (test 0.960-0.963). The final kl_true is 2e-4 to 1.2e-3 nats, so the true conditional is "
        "learned. There are no drops and no test-loss bump (test_bump <= 0.005). The other 234 smooth markov runs never "
        "reach the threshold, so the group median of the feature is the censored value log10(60,000) = 4.778 "
        "with MAD 0. The floor 0.1 then sets the scale (0.148), and any run that reaches the threshold scores "
        "about 17. The score flags a different task regime pooled into one (region, label) group.",
    ("parity-k4", "smooth", "test_bump"):
        "Looked at by hand: a successful parity run with large early loss spikes. During the initial plateau "
        "near ln 2 (test loss 0.699 at step 0), the train/test losses spike to 1.054/1.006 at step 4 and to "
        "1.148/1.193 at step 145. The run then returns to the plateau (test loss about 0.73 at step 196 and 0.65 "
        "at step 657) and drops to the solution: test accuracy reaches 90 % at step 1,033 (group median 1,624) and "
        "ends at 1.0 with test loss 2.6e-5, like its peers. The group pools lr = 0.03 runs (17 runs, test_bump "
        "<= 0.067; they set the group median 0.067) with lr = 0.1 runs (16 runs, 0.22 to 0.71; "
        "group_raw_feature_by_config). The same configuration (k = 4, n = 24, "
        "lr = 0.1; 12 seeds, 9 plateau_drop and 3 smooth) has bumps of 0.21 to 0.45 in the other 11 seeds, so "
        "early instability at lr = 0.1 is normal there, and this seed has the largest. Borderline: z = 8.6 "
        "against the threshold of 8.",
}


# ---------------------------------------------------------------------------------------------- items
def build_items(c9: dict, c10: dict) -> list[dict]:
    lit = c9["literal_completion_order"]
    fv = lit["first_violation"]
    sp = lit["spells"]
    s1 = c9["S1_batch_ends"]
    s2 = c9["S2_reconstructed_request_order"]
    mix = c9["stream_mix"]
    fin = c9["S4_final"]
    san = c9["sanity"]
    longest = max(sp, key=lambda x: x["n_followups"]) if sp else None
    nontrivial = [x for x in sp if not (x["first_i"] == 1 and x["last_i"] == 1)]
    max_nontriv = max(nontrivial, key=lambda x: x["max_share"]) if nontrivial else None
    last_viol_i = max(x["last_i"] for x in sp) if sp else None
    s1_viol = [r["batch"] for r in s1 if r["violation_lo"]]
    s1_viol_est = [r["batch"] for r in s1 if r["violation_call_est"]]
    call_viol = [r for r in s1 if r["call_violation"]]
    s2sp = s2["spells"]
    s2_nontriv = [x for x in s2sp if not (x["first_i"] == 1 and x["last_i"] == 1)]
    s2_max = max(s2_nontriv, key=lambda x: x["max_share"]) if s2_nontriv else None
    s2_last = max(x["last_i"] for x in s2sp) if s2sp else None
    max_end_share = max(r["top_share"] for r in s1)
    max_end = max(s1, key=lambda r: r["top_share"])
    calls_in_bracket = all(b["call_est_in_bracket"] and b["picks_finished_by_call_est"] for b in c9["batches"])
    by = mix["by_stream_label"]
    pbsl = mix["per_batch_stream_label"]

    def spell_txt(x):
        return (f"follow-ups {x['first_i']}-{x['last_i']} ({x['n_followups']} follow-up(s), "
                f"{x['duration_s']:.0f} s), majority {'/'.join(x['majority_labels'])}, max share "
                f"{100 * x['max_share']:.1f} % at i = {x['max_share_at_i']} ({x['max_share_label']}, "
                f"+{x['max_excess_pp']:.1f} pp, surplus {x['max_surplus_followups']} follow-ups)")

    def call_txt(r):
        left = {lab: r["cand_left_after_call_call_est"][lab] for lab in LABELS_G if lab != r["call_top_label"]}
        return (f"call {r['batch']}: {r['call_top_label']} {r[f'batch_n_' + r['call_top_label']]} of "
                f"{sum(r[f'batch_n_{lab}'] for lab in LABELS_G)} ({100 * r['call_top_share']:.1f} %), candidates "
                f"left of other labels {left}")

    key9 = [f"grokking-v1 follow-ups: {lit['n']} (all {san['grokking_objects']} objects, each once); follow-up curves "
            f"by region: {san['followup_runs_by_region']}",
            "follow-up labels: " + ", ".join(f"{k} {v}" for k, v in fin["counts"].items()),
            f"scheduler calls: {len(s1)} blocks of sizes {san['batch_sizes']} (pace {san['pace_per_call']} per call), "
            f"each complete before the next: {san['batches_complete_before_next']}; estimated call times inside "
            f"their brackets and after every pick finished: {calls_in_bracket}",
            (f"literal (completion order): {lit['n_violating']} of {lit['n']} prefixes violate; first at i = "
             f"{fv['i']} ({fv['top_label']} {100 * fv['top_share']:.0f} %); last at i = {last_viol_i}")
            if fv else "literal: no violation",
            f"spells: {len(sp)}: " + "; ".join(spell_txt(x) for x in sp),
            "violating prefixes after excluding the first 1/2/5/10/70: "
            + "/".join(str(lit['violating_excluding_first_k'][k]) for k in ('1', '2', '5', '10', '70')),
            f"S2 reconstructed request order: {s2['n_violating']} of {s2['n']} prefixes violate, {len(s2sp)} spells, "
            f"last at i = {s2_last}, max share excluding i = 1: "
            + (f"{100 * s2_max['max_share']:.1f} % at i = {s2_max['max_share_at_i']}" if s2_max else "-"),
            f"S1 call boundaries (exact): violations after calls {s1_viol_est or 'none'} (candidates at estimated "
            f"call time), {s1_viol or 'none'} (bracket lower end); largest cumulative share at a call end "
            f"{100 * max_end_share:.1f} % ({max_end['top_label']}, after call {max_end['batch']})",
            "S1 cumulative shares after each call (g/m/s %): " + ", ".join(
                f"{r['batch']}: {100 * r['share_grokking']:.1f}/{100 * r['share_memorization']:.1f}/"
                f"{100 * r['share_smooth']:.1f}" for r in s1),
            "S3 per call (> 50 % of one call while other labels had candidates): "
            + ("; ".join(call_txt(r) for r in call_viol) if call_viol else "none"),
            f"S4 final share: max {100 * fin['max_share']:.1f} % (no candidates left at the end)",
            f"label-blind streams (matched + boundary): {mix['label_blind_n']} of {lit['n']} "
            f"({100 * mix['label_blind_share']:.1f} %); per call: " + ", ".join(
                f"{b}: {100 * v:.0f} %" for b, v in mix["label_blind_share_by_batch"].items()),
            "stream x label: " + "; ".join(f"{s}: " + ", ".join(f"{lab} {by[s].get(lab, 0)}" for lab in LABELS_G)
                                          for s in by if s != "All"),
            f"follow-ups with reason 'anomaly': {san['followup_reason_starts_with_anomaly_all_regions']}"]

    g10 = c10["groups"]
    ov = c10["overall"]
    af = c10["anomaly_followups"]
    fl = c10["flagged"]
    key10 = [f"{g['region']}/{g['label']}: {g['n_flagged']} of {g['n_scored']} scored"
             + (f" ({100 * g['frac_flagged']:.2f} %, expected {g['expected_at_1pct']:.2f} at 1 %, two-sided binomial "
                f"p = {g['binom_two_sided_p']:.3g}, deciding features of flags {g['deciding_features_flagged'] or '-'}"
                f", top score {g['top5_scores'][-1]:.2f})" if g['n_scored'] else
                f" (group of {g['n']} < MIN_GROUP = {c10['min_group']}, not scored)") for g in g10]
    key10 += [f"overall: {ov['n_flagged']} of {ov['n_scored']} scored = {100 * ov['frac_of_scored']:.2f} % "
              f"(of all {ov['n_objects']} objects {100 * ov['frac_of_all_objects']:.2f} %); Clopper-Pearson 95 % "
              f"[{100 * ov['ci95_clopper_pearson'][0]:.2f}, {100 * ov['ci95_clopper_pearson'][1]:.2f}] %; "
              f"bootstrap over {ov['bootstrap_configs']['n_configurations']} configurations (2,000 resamples) "
              f"[{100 * ov['bootstrap_configs']['ci95'][0]:.2f}, {100 * ov['bootstrap_configs']['ci95'][1]:.2f}] %",
              f"flags fall in {ov['n_configurations_with_flags']} of {ov['bootstrap_configs']['n_configurations']} "
              f"configurations",
              f"groups 'about 1 %' (primary, two-sided binomial p >= 0.05): {ov['n_groups_primary_ok']} of "
              f"{ov['n_groups_scored']}; alt (a) not too many: {ov['n_groups_alt_a_ok']} of {ov['n_groups_scored']};"
              f" alt (b) 0.5-2 % band: {ov['n_groups_alt_b_ok']} of {ov['n_groups_scored']}",
              f"follow-ups with reason starting 'anomaly': {af['n_followups_reason_anomaly']} of "
              f"{af['n_followups_grokking']}; offline flags in grokking-v1: {af['offline_flagged_grokking_final_catalog']}"
              f"; offline flags by region: {af['offline_flags_by_region']} (in situ regions, never followed up); "
              "max grokking-v1 scores: " + ", ".join(f"{k} {v:.2f}" for k, v in af["max_score_grokking_by_label"].items()),
              f"scheduler view reconstructed at the {len(af['scheduler_view_at_calls'])} calls: flagged eligible "
              f"objects at most {af['max_flagged_eligible_any_call']}"]
    for f in fl:
        key10.append(f"flagged {f['object_id']} {f['region']}/{f['label']} seed {f['seed']}: z {f['score']:.2f} on "
                     f"{f['feature']} ({f['feature_value']:.3f} vs group median {f['group_median']:.3f})")

    primary_fail = [f"{g['region']}/{g['label']}" for g in g10 if g["about_1pct_primary"] is False]
    alt_a_fail = [f"{g['region']}/{g['label']}" for g in g10 if g["about_1pct_alt_a_not_too_many"] is False]
    alt_b_ok = [f"{g['region']}/{g['label']}" for g in g10 if g["about_1pct_alt_b_band"]]
    gk = [g for g in g10 if g["region"] == REGION_G]
    mk_flag = [f for f in fl if f["region"] == "markov-v1"]
    other_flag = [f for f in fl if f["region"] != "markov-v1"]
    mk_cfgs = sorted({(f["config"]["couplings"], f["config"]["train_context"]) for f in mk_flag})
    mk_same = mk_flag[0]["same_configuration"] if mk_flag else None
    mtab = c10["markov_threshold_table"]
    reach = [m for m in mtab if m["n_reach_t_train_acc90"]]

    interp9 = (
        f"Read literally, check 9 fails, but only at the very start. In completion order {lit['n_violating']} of "
        f"{lit['n']} prefixes violate, all within the first scheduler call (i <= {last_viol_i} of its 70 picks). "
        f"Prefix 1 is trivial (a single pick is 100 %). Without it, the majority label "
        + (f"({'/'.join(sorted({l for x in nontrivial for l in x['majority_labels']}))}) never exceeds "
           f"{100 * max_nontriv['max_share']:.1f} % and is never more than "
           f"{max(x['max_surplus_followups'] for x in nontrivial)} picks ahead of the other labels combined"
           if max_nontriv else "does not exceed 50 %")
        + f", while other labels had candidates. In the reconstructed request order the picture is the same "
        f"({s2['n_violating']} prefixes, all at i <= {s2_last}; max "
        + (f"{100 * s2_max['max_share']:.1f} % at i = {s2_max['max_share_at_i']}" if s2_max else "-")
        + f"). At every call boundary, where the cumulative "
        f"counts are exact in request order, no label exceeds 50 %. The largest end-of-call share is "
        f"{100 * max_end_share:.1f} % ({max_end['top_label']}, after the last call, when nothing is left). The "
        f"failure Amendment 1 H anticipated, one label dominating the region, does not occur: smooth is "
        f"{100 * fin['shares'].get('smooth', 0):.1f} % of the region. The by-design mechanism is still visible "
        f"per call: in calls {[r['batch'] for r in call_viol]} smooth took "
        + ", ".join(f"{100 * r['call_top_share']:.1f} %" for r in call_viol)
        + " of the picks while memorization (and in the first of them grokking) still had candidates. That "
        "happened as the grokking candidates ran out and the label-blind boundary/matched picks followed the "
        "smooth-heavy remainder.")
    interp10 = (
        f"Overall, the offline flag rate is about the intended 1 % ({ov['n_flagged']} of {ov['n_scored']} = "
        f"{100 * ov['frac_of_scored']:.2f} %). The per-group criterion still fails, in one group: "
        f"{', '.join(primary_fail) or 'none'}. There, "
        f"the {len(mk_flag)} flags are all {mk_same['n'] if mk_same else 0} seeds of a single configuration "
        f"({', '.join(f'{a} {b}' for a, b in mk_cfgs)}). Its runs are ordinary smooth fits, but they are the only "
        f"Markov runs whose train accuracy reaches the rescaled 90 % threshold (reached in "
        f"{len(reach)} of {len(mtab)} coupling x context cells). The flag is therefore a grouping artefact: the "
        "(region, label) group pools 24 task regimes, and the censored-time feature has MAD 0. These are not "
        f"curves that defy their label. The remaining flag ({', '.join(f['region'] + '/' + f['label'] for f in other_flag)}) "
        "is a borderline early loss spike in a successful run. The grokking-v1 groups have 0 flags (largest "
        "scores " + ", ".join(f"{k} {v:.2f}" for k, v in af["max_score_grokking_by_label"].items()) + "). That "
        f"matches the {af['n_followups_reason_anomaly']} anomaly follow-ups, and so does the reconstructed "
        "scheduler view at each call. The 'unknown' exit thus gave no grokking follow-up in v1, and the 13 "
        "offline flags lie in in-situ regions the scheduler never scores.")

    items = [
        {"id": "check9",
         "question": "System check 9: follow-ups in grokking-v1 - no label takes more than 50 % of them while other "
                     "labels still have candidates (Amendment 1 H: may fail by design; reported literally over the "
                     "request order as a declared deviation).",
         "definition_used": (
             "Order: ri_runs.ri_run_id of the grokking-v1 follow-up curves (the order in which follow-ups were "
             "recorded = completion order, a proxy for followups.rowid, which DR1 lacks; ri_run_id order equals "
             "created_at order). Label: latest label (no label was assigned after its follow-up was recorded; the "
             f"label named in {san['reason_label_known']} reasons agrees with it in every case, boundary reasons "
             "name none). Stream: the curve's meta followup_reason. Candidates of a label at follow-up i: "
             "grokking-v1 objects of that label with finished_at <= created_at of follow-up i and not followed up "
             "at or before i. Violation at i: the largest cumulative label share over follow-ups 1..i is > 50 % "
             "and some other label has >= 1 candidate. Evaluated over every prefix (literal, i = 1 included). "
             "Sensitivity readings: S1 the cumulative share at the end of each scheduler call (the follow-ups form "
             f"{len(s1)} disjoint blocks of {san['batch_sizes'][0]} (last {san['batch_sizes'][-1]}), one per 600 s "
             "call, so these points are exact in request order); candidates at the estimated call time (daemon's "
             "period: t_daemon + 600 s x k, with t_daemon the smallest offset at which every call's picks had "
             "finished), and at a lower bound (the latest finish among the call's picks). S2 the request order "
             "reconstructed inside each call from the scheduler's stream round robin. S3 the per-call share. S4 the "
             "final share."),
         "literal_result": (
             (f"{lit['n_violating']} of {lit['n']} prefixes violate; first violation at i = {fv['i']} "
              f"({fv['top_label']}, trivial single pick); {len(sp)} spells: " + "; ".join(spell_txt(x) for x in sp)
              + f". Last violation at i = {last_viol_i} (first call). Excluding i = 1: "
              f"{lit['violating_excluding_first_k']['1']} violating prefixes. S2 reconstructed request order: "
              f"{s2['n_violating']} violating prefixes, all at i <= {s2_last}. S1 call ends: violations after calls "
              f"{s1_viol_est or 'none'}. Final shares: "
              + ", ".join(f"{k} {100 * v:.1f} %" for k, v in fin["shares"].items()))
             if fv else "no violation"),
         "literal_verdict": "fail" if lit["n_violating"] else "pass",
         "amended_result": (
             "Amendment 1 H keeps the criterion and asks for a literal report over the request order, with the "
             "expected failure declared as a deviation. Over the reconstructed request order the check fails "
             f"({s2['n_violating']} prefixes, all in the first call). At exact call boundaries it holds (max "
             f"{100 * max_end_share:.1f} %). Per call: " + ("; ".join(call_txt(r) for r in call_viol) or "none")
             + f". The label-blind matched and boundary streams made {mix['label_blind_n']} of {lit['n']} picks "
             f"({100 * mix['label_blind_share']:.1f} %, about 2/3 in calls 2-7). No anomaly-stream follow-ups."),
         "amended_verdict": "fail" if s2["n_violating"] else "pass",
         "deciding": "literal",
         "key_numbers": "\n".join(key9),
         "interpretation": interp9,
         "caveats": (
             "DR1 has no followups table, so request order is unknown and completion order is a proxy. Within a "
             "call, requests are claimed by priority (matched/boundary 2.0 before stratified/random_control 1.5), "
             "so stratified picks complete last in each block. Completion order is therefore biased against label "
             "balance inside a call. The S2 reconstruction assumes that order within a stream equals completion "
             "order. Call-end points do not depend on either. Call times are estimated, but every estimate falls "
             "inside its bracket and after all of that call's picks had finished. Prefix i = 1 is a trivial "
             "violation. Labels: every latest label was recorded within 0.04 s of the object's finish, before its "
             "follow-up, and agrees with the label named in the follow-up reason; earlier labels are not in DR1.")},
        {"id": "check9_stream_mix",
         "question": "Check 9 context: stream mix of the grokking-v1 follow-ups by label and by scheduler call.",
         "definition_used": "Stream = followup_reason prefix (boundary along any knob pooled); label = latest label.",
         "literal_result": "stream x label: " + "; ".join(
             f"{s}: " + ", ".join(f"{lab} {by[s].get(lab, 0)}" for lab in LABELS_G) for s in by),
         "literal_verdict": "descriptive",
         "key_numbers": "\n".join(
             [f"call {b}: " + "; ".join(f"{s} " + ",".join(f"{lab[:4]} {v.get(lab, 0)}" for lab in LABELS_G)
                                        for s, v in d.items()) for b, d in pbsl.items()]
             + ["finished composition at each call (g/m/s): " + ", ".join(
                 f"{a['batch']}: {a['finished']['grokking']}/{a['finished']['memorization']}/"
                 f"{a['finished']['smooth']}" for a in c9["finished_composition_at_calls"])]),
         "interpretation": (
             "Matched and boundary picks ignore labels. The stratified stream gives the next turn to the label "
             "with the fewest follow-ups. Grokking (17.9 % of the region) was over-sampled early, because "
             "the stratified stream equalised the labels, and was exhausted by call 8. After that the "
             "remaining picks could only be memorization or smooth."),
         "caveats": "Descriptive; no criterion."},
        {"id": "check10",
         "question": "System check 10: anomalies flagged at about 1 % per (region, label) group, each looked at by "
                     "hand (Amendment 1 I: evaluated offline on the final catalog for every v1 group; per-group "
                     "result literal, overall rate as a declared deviation).",
         "definition_used": (
             "Offline: tycho.analysis.anomaly.anomaly_scores per (region, label) group of the DR1 objects (all "
             f"{ov['n_grouped']} are done, labelled and not diverged, as the scheduler requires). Summary mapping = "
             "the object's objects.parquet row, which holds every key summary_features reads (t_train_acc90, "
             "t_test_acc90, steps_total, final_train_acc, final_test_acc, final_train_loss, final_test_loss, "
             "plateau_logfrac, n_drops, test_bump). Flagged iff score >= THRESHOLD = 8.0. Groups with fewer than "
             f"MIN_GROUP = {c10['min_group']} objects are not scored. 'About 1 %' is ambiguous; chosen reading: "
             "the group's flag count is consistent with a 1 % rate (exact two-sided binomial test vs p = 0.01, "
             "p >= 0.05). Alternatives: (a) not too many (one-sided P(X >= k) >= 0.05); (b) a fixed band "
             "0.5 % <= k/n <= 2 %. Literal (as run) reading: the flags the survey itself acted on, i.e. "
             "scheduler follow-ups whose reason starts with 'anomaly'. The scheduler scores grokking-v1 only."),
         "literal_result": (
             f"As run: {af['n_followups_reason_anomaly']} anomaly follow-ups of {af['n_followups_grokking']} "
             f"grokking-v1 follow-ups. That is 0 % in each grokking-v1 label group: 0/"
             + ", 0/".join(str(g['n_scored']) for g in gk)
             + ", consistent with 1 % under the binomial reading (p = "
             + ", ".join(f"{g['binom_two_sided_p']:.3g}" for g in gk)
             + "), outside the 0.5-2 % band. The other regions were not scored at run time, so 7 of the 10 groups "
             "cannot be evaluated from run records. The reconstructed scheduler view at each call finds at most "
             f"{af['max_flagged_eligible_any_call']} eligible flagged object."),
         "literal_verdict": "inconclusive",
         "amended_result": (
             "Offline, every group: " + "; ".join(
                 f"{g['region']}/{g['label']} {g['n_flagged']}/{g['n_scored']}" if g["n_scored"] else
                 f"{g['region']}/{g['label']} not scored (n = {g['n']})" for g in g10)
             + f". Groups not 'about 1 %': primary {primary_fail or 'none'}; alt (a) {alt_a_fail or 'none'}; alt "
               f"(b): {len(alt_b_ok)} of {ov['n_groups_scored']} groups inside the band. Overall (declared "
               f"deviation): {ov['n_flagged']} of {ov['n_scored']} = {100 * ov['frac_of_scored']:.2f} % (CP 95 % "
               f"[{100 * ov['ci95_clopper_pearson'][0]:.2f}, {100 * ov['ci95_clopper_pearson'][1]:.2f}] %; "
               f"configuration bootstrap [{100 * ov['bootstrap_configs']['ci95'][0]:.2f}, "
               f"{100 * ov['bootstrap_configs']['ci95'][1]:.2f}] %). All {ov['n_flagged']} flagged objects were "
               f"looked at by hand: {len(mk_flag)} are every seed of Markov "
               f"{', '.join(f'{a} ({b})' for a, b in mk_cfgs)}, a different accuracy regime, and "
             + ", ".join(f"1 is {f['region']}/{f['label']} {f['object_id']} (early loss spike, z {f['score']:.1f})"
                         for f in other_flag) + "."),
         "amended_verdict": "fail" if primary_fail else "pass",
         "deciding": "amended",
         "key_numbers": "\n".join(key10),
         "interpretation": interp10,
         "caveats": (
             "The pre-registration does not define 'about 1 %'. With 33 to 330 scored objects per group, a 1 % "
             "rate means 0.3 to 3.3 expected flags, so 0 is the typical outcome for small groups. Under the "
             "0.5-2 % band every scored group fails; under the binomial reading small groups can pass. Flags "
             f"cluster by configuration ({ov['n_configurations_with_flags']} configurations hold all "
             f"{ov['n_flagged']}), so the effective sample for the overall rate is the number of configurations "
             "(hence the wide bootstrap interval). Features come from objects.parquet, not the catalog's "
             "summary_json; the keys are the same. parity-k2/double_descent (11 objects) is below MIN_GROUP and "
             "never scored. The scheduler's call times are estimated.")},
        {"id": "check10_anomaly_followups",
         "question": "Check 10 context: follow-ups whose reason starts with 'anomaly' vs the offline flags.",
         "definition_used": ("Count of follow-up curves whose meta followup_reason starts with 'anomaly' (all "
                             "regions). The scheduler's view is reconstructed at each call: grokking-v1 objects "
                             "finished by the estimated call time (and at both ends of its bracket) are scored by "
                             "label group, and flagged objects not yet followed up are counted."),
         "literal_result": (f"{af['n_followups_reason_anomaly']} anomaly follow-ups; offline flags in grokking-v1: "
                            f"{af['offline_flagged_grokking_final_catalog']}; flagged eligible objects at any call: "
                            f"{af['max_flagged_eligible_any_call']}; offline flags elsewhere: "
                            f"{af['offline_flags_by_region']} (ri_policy always: in situ RI, no follow-ups)"),
         "literal_verdict": "descriptive",
         "key_numbers": "\n".join(
             [f"call {v['batch']}: done {v['call_est']['n_done']}, max score {v['call_est']['max_score']:.2f}"
              f" ({v['call_est']['max_score_label']}, {v['call_est']['max_score_feature']}), flagged eligible "
              f"{v['call_est']['n_flagged_eligible']}" for v in af["scheduler_view_at_calls"]]
             + [f"top grokking-v1 score {x['label']}: {x['object_id']} {x['score']:.2f} ({x['feature']})"
                for x in af["grokking_top3_scores_per_label"]]),
         "interpretation": ("Consistent: the scheduler found nothing to flag in grokking-v1, at any call or on "
                            "the final catalog. The anomaly stream was therefore empty, and its turn in the round "
                            "robin passed to the other streams."),
         "caveats": "Call times estimated; labels assumed fixed from the end of training (labelled_at = finish)."},
    ]
    return items


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases/DR1")
    ap.add_argument("--extracted", type=Path, default=SCRATCH / "dr1")
    ap.add_argument("--out", type=Path, default=REPO / "docs/results/v1_analysis/system_checks.json")
    args = ap.parse_args()
    assert Path(tycho.__file__).resolve().is_relative_to(REPO), tycho.__file__

    objects = pd.read_parquet(args.release / "objects.parquet")
    objects["cfg_key"] = objects["task_cfg_json"].map(lambda s: json.dumps(json.loads(s), sort_keys=True))
    runs = pd.read_parquet(args.release / "ri_runs.parquet")

    c9 = check9(objects, runs, args.extracted)
    followed_batch = {r["object_id"]: r["batch"] for r in c9["per_run"]}
    fu_reasons = pd.DataFrame(c9["per_run"])[["object_id", "reason"]]
    c10 = check10(objects, runs, args.extracted, fu_reasons, c9["t_start_unix"], c9["batches"], followed_batch)
    items = build_items(c9, c10)

    # chart data: cumulative shares (literal order), per-call composition, per-group flag rates
    chart = {"check9_cumulative_shares": [{k: r[k] for k in ("i", "t", "batch", "label", "share_grokking",
                                                             "share_memorization", "share_smooth",
                                                             "cand_grokking", "cand_memorization", "cand_smooth",
                                                             "violation")} for r in c9["per_run"]],
             "check9_calls": [{"call": r["batch"], **{f"n_{lab}": r[f"batch_n_{lab}"] for lab in LABELS_G},
                               **{f"cum_share_{lab}": r[f"share_{lab}"] for lab in LABELS_G}}
                              for r in c9["S1_batch_ends"]],
             "check10_groups": [{"group": f"{g['region']}/{g['label']}", "n_scored": g["n_scored"],
                                 "n_flagged": g["n_flagged"], "frac": g["frac_flagged"]} for g in c10["groups"]]}
    out = {"title": "Sky v1 system checks 9 and 10 (incl. Amendment 1 H and I)",
           "preregistration": "docs/results/v1_preregistration.md",
           "script": "scripts/analysis_v1/system_checks.py", "release": str(args.release),
           "tycho_file": tycho.__file__, "items": items, "check9": c9, "check10": c10, "chart_data": chart}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(out), indent=1, allow_nan=False))
    for it in items:
        print("=" * 100)
        for k in ("id", "literal_verdict", "amended_verdict", "deciding"):
            print(f"{k}: {it.get(k)}")
        print("literal:", it["literal_result"])
        print("amended:", it.get("amended_result"))
        print(it["key_numbers"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
