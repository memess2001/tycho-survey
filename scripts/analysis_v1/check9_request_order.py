"""System check 9 over the exact request order of the grokking-v1 follow-ups (Amendment 1 H).

Check 9: no label takes more than 50 % of the follow-ups while other labels still have candidates. Amendment H asks
for the check over the request order (``followups.rowid``). DR1 has no follow-up request table, but the order can be
recovered exactly: each scheduler call (a block of at most 70 follow-ups) is replayed with the scheduler's own stream
code (:class:`tycho.survey.scheduler.Scheduler`) on the catalog state that DR1 implies at the call time. A replay
counts as exact when it reproduces the call's set of objects and every follow-up reason string. This script was
written by the independent verifier of ``system_checks.py``, whose completion-order reconstruction it corrects, and
then made portable (paths as arguments).

Run from the repository root, after extracting DR1 (see README.md):
``PYTHONPATH=. python scripts/analysis_v1/check9_request_order.py``. Writes ``check9_request_order.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path

import numpy as np
import pandas as pd

import tycho
from tycho.survey import scheduler as S
from tycho.survey.sky import Sky, point_key

REPO = Path(__file__).resolve().parents[2]
assert Path(tycho.__file__).resolve().is_relative_to(REPO), f"wrong tycho package: {tycho.__file__}"
LABELS = ["grokking", "memorization", "smooth"]
CALL_SIZE = 70
NON_SUMMARY = {"object_id", "task", "region", "sky_version", "seed", "task_cfg_json", "source", "random_control",
               "priority", "attempts", "created_at", "started_at", "finished_at", "wall_time_s", "device", "n_params",
               "code_version", "lightcurve_points", "label", "label_tags_json", "label_score", "classifier",
               "labelled_at", "lightcurve_file"}


def clean(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if v is pd.NA or (isinstance(v, (float, np.floating)) and not math.isfinite(v)):
        return None
    if isinstance(v, np.floating):
        return float(v)
    return v


def load_followups(release: Path, extracted: Path, obj: pd.DataFrame) -> pd.DataFrame:
    """grokking-v1 follow-ups in ri_run_id (= created_at) order, with the reason stored in each curve's header."""
    runs = pd.read_parquet(release / "ri_runs.parquet")
    fu = runs[runs.kind == "followup"].copy()
    reasons = []
    for f in fu.ricurve_file:
        with np.load(extracted / f, allow_pickle=False) as z:
            reasons.append((json.loads(str(z["header"])).get("meta") or {}).get("followup_reason"))
    fu["followup_reason"] = reasons
    fu = fu.merge(obj[["object_id", "region", "label"]], on="object_id")
    fu = fu[fu.region == "grokking-v1"].sort_values("ri_run_id").reset_index(drop=True)
    assert (np.diff(fu.created_at.values) > 0).all(), "ri_run_id order differs from created_at order"
    fu["pos"] = np.arange(1, len(fu) + 1)
    fu["call"] = (fu.pos - 1) // CALL_SIZE + 1
    fu["stream"] = fu.followup_reason.str.split(":").str[0].str.replace(r"^boundary along .*", "boundary", regex=True)
    return fu


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases" / "DR1")
    ap.add_argument("--extracted", type=Path, default=REPO / "work" / "dr1")
    ap.add_argument("--sky", type=Path, default=REPO / "configs" / "sky_v1.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v1_analysis" / "check9_request_order.json")
    args = ap.parse_args(argv)

    obj = pd.read_parquet(args.release / "objects.parquet")
    fu = load_followups(args.release, args.extracted, obj)
    sky = Sky.from_yaml(args.sky)
    region = next(r for r in sky.regions if r.name == "grokking-v1")
    g = obj[obj.region == "grokking-v1"].copy().sort_values("created_at")
    assert len(fu) == len(g) == fu.object_id.nunique(), "expected one follow-up per grokking-v1 object"
    summary_keys = [c for c in obj.columns if c not in NON_SUMMARY and not c.startswith("train_")]
    call_of = dict(zip(fu.object_id, fu.call))
    reason_of = dict(zip(fu.object_id, fu.followup_reason))

    def point_of(r):
        cfg = json.loads(r.task_cfg_json)
        return {"train_frac": cfg["train_frac"], "width": cfg["width"], "train.lr": r.train_lr,
                "train.weight_decay": r.train_weight_decay}

    g["pt"] = [point_of(r) for r in g.itertuples()]
    g["key"] = [point_key(p) for p in g.pt]

    def rows_at(t: float, call: int) -> list[dict]:
        rows = []
        for r in g.itertuples():
            if r.created_at > t:
                continue
            done = r.finished_at <= t
            rows.append(dict(object_id=r.object_id, status="done" if done else "running",
                             label=r.label if done else None, diverged=0, seed=int(r.seed),
                             random_control=int(r.random_control), finished_at=r.finished_at if done else None,
                             t_test_acc90=(clean(r.t_test_acc90) if done else None), sky_version=sky.version,
                             has_followup=int(call_of[r.object_id] < call), key=r.key, point=r.pt,
                             summary_json=json.dumps({k: clean(getattr(r, k)) for k in summary_keys}) if done else None))
        return rows

    fake = types.SimpleNamespace(sky=types.SimpleNamespace(version=sky.version))
    fake._boundaries = lambda reg, pts: S.Scheduler._boundaries(fake, reg, pts)

    def replay(t: float, call: int, n: int) -> list[tuple]:
        """The scheduler's picks for one region at time t, as (priority, object_id, reason), in request order."""
        rows = rows_at(t, call)
        eligible = {r["object_id"] for r in rows if r["status"] == "done" and not r["has_followup"] and r["diverged"] != 1}
        points: dict = {}
        for r in rows:
            points.setdefault(r["key"], S._Point(point=r["point"], key=r["key"])).rows.append(r)
        n_followed = sum(1 for r in rows if r["has_followup"])
        subs = {"anomaly": S.Scheduler._anomaly_stream(fake, region, rows, eligible),
                "matched": S.Scheduler._matched_stream(fake, region, points, eligible),
                "boundary": S.Scheduler._boundary_stream(fake, region, points, eligible),
                "stratified": S.Scheduler._stratified_stream(fake, region, rows, eligible)}
        start = n_followed % 4
        order = [subs[s] for s in S.REGION_STREAMS[start:] + S.REGION_STREAMS[:start]]
        seen, out = set(), []
        while order and len(out) < n:
            for it in list(order):
                for item in it:
                    if item[1] not in seen:
                        seen.add(item[1])
                        out.append(item)
                        break
                else:
                    order.remove(it)
                if len(out) >= n:
                    break
        return out

    def exact(out, sub) -> bool:
        return ({o for _, o, _ in out} == set(sub.object_id)
                and all(reason_of[o] == rs for _, o, rs in out))

    finish_times = sorted(g.finished_at)
    calls, variants_per_call = [], []
    for c, sub in fu.groupby("call"):
        # the call came after every picked object had finished and before its first follow-up was recorded
        lo = g.set_index("object_id").loc[sub.object_id, "finished_at"].max()
        hi = sub.created_at.min()
        ts = sorted({lo, hi, *[t for t in finish_times if lo <= t <= hi]})
        good: list[tuple[float, tuple]] = []
        for t in ts:
            out = replay(t, c, len(sub))
            if exact(out, sub):
                good.append((t, tuple(o for _, o, _ in out)))
        if not good:
            raise SystemExit(f"call {c}: no replay reproduces the recorded picks and reasons")
        variants: dict[tuple, float] = {}
        for t, order in good:
            variants.setdefault(order, t)  # each distinct order with the earliest call time that gives it
        nxt = [t for t in finish_times if t > max(t for t, _ in good)]
        calls.append(dict(call=int(c), n=int(len(sub)), candidate_times=len(ts), exact_times=len(good),
                          distinct_orders=len(variants),
                          window=[float(min(t for t, _ in good)), float(min(nxt[0] if nxt else hi, hi))]))
        variants_per_call.append(sorted(variants.items(), key=lambda kv: kv[1]))

    t_first = float(obj.created_at.min())
    for c in calls:
        c["window_rel_s"] = [round(x - t_first, 1) for x in c.pop("window")]
    gfin, glab = g.finished_at.values, g.label.values
    label_of = dict(zip(fu.object_id, fu.label))

    def evaluate(choice: list[tuple[tuple, float]]) -> pd.DataFrame:
        """Check 9 over the request order given by one order (and call time) per call."""
        request = [(o, ci + 1, t) for ci, (order, t) in enumerate(choice) for o in order]
        pos = {o: i + 1 for i, (o, _, _) in enumerate(request)}
        gpos = g.object_id.map(pos).values
        counts = dict.fromkeys(LABELS, 0)
        recs = []
        for i, (o, call, t) in enumerate(request, start=1):
            lab = label_of[o]
            counts[lab] += 1
            cand = {L: int(((glab == L) & (gfin <= t) & (gpos > i)).sum()) for L in LABELS}
            maj = max(LABELS, key=lambda L: counts[L])
            share = counts[maj] / i
            viol = share > 0.5 and sum(cand[L] for L in LABELS if L != maj) > 0
            recs.append(dict(i=i, call=call, object_id=o, label=lab, stream=stream_of[o], reason=reason_of[o],
                             majority=maj, share=round(share, 4), surplus=int(2 * counts[maj] - i),
                             violates=bool(viol), **{f"candidates_{L}": cand[L] for L in LABELS}))
        return pd.DataFrame(recs)

    stream_of = dict(zip(fu.object_id, fu.stream))
    combos = [[]]
    for vs in variants_per_call:
        combos = [prev + [v] for prev in combos for v in vs]
    evaluated = [evaluate(cb) for cb in combos]
    ev = evaluated[0]  # the earliest exact call time in every call
    alternatives = [{"violating_prefixes": int(e.violates.sum()), "violating_i": [int(i) for i in e.i[e.violates]]}
                    for e in evaluated]
    v = ev.violates.values
    spells = int(np.sum(v & ~np.r_[False, v[:-1]]))
    ends = ev.groupby("call").tail(1)
    e1 = ev[ev.i > 1]
    result = {
        "description": __doc__.split("\n\n")[0],
        "n_followups": int(len(ev)),
        "calls": calls,
        "note_on_ambiguity": "Where several call times inside a call's window reproduce the recorded picks and "
                             "reasons exactly but in different orders, the primary result uses the earliest such "
                             "time; 'alternatives' gives the result for every combination.",
        "alternatives": alternatives,
        "violating_prefixes": int(v.sum()),
        "violating_i": [int(i) for i in ev.i[v]],
        "spells": spells,
        "max_share_excluding_i1": {"share": float(e1.share.max()), "i": int(ev.i[e1.share.idxmax()]),
                                   "label": ev.majority[e1.share.idxmax()]},
        "call_end_max_share": [{"call": int(r.call), "i": int(r.i), "majority": r.majority, "share": float(r.share)}
                               for r in ends.itertuples()],
        "first_labels": "".join(ev.label.str[0][:45]),
        "first_streams": "".join(ev.stream.str[0][:45]),
        "per_call_label_counts": {int(c): sub.label.value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()
                                  for c, sub in ev.groupby("call")},
        "request_order": ev.to_dict(orient="records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"exact replay of {len(calls)} calls; violating prefixes {result['violating_prefixes']} of {len(ev)} "
          f"(i = {result['violating_i']}); call-end max share "
          f"{max(c['share'] for c in result['call_end_max_share']):.3f}; first labels {result['first_labels'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
