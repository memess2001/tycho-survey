"""Supplementary checks of the sky v1 analysis, raised by the completeness review (not pre-registered).

1. ``operations``: refinement size per region, summed training time, the run's time line, follow-up densities, and
   strong order-2 hidden-unit interactions that are already present at initialisation in grokking-v1.
2. ``parity_lower_order``: activations of proper subsets of the planted parity set (which the target does not
   contain) and of subsets outside it.
3. ``markov_magnitude``: Fourier crossing steps (10, 25, 50 % of the population coefficient, single checkpoint) of
   fixed lag sets at several coupling strengths, alone and with the pairwise term, per training context; separates a
   smaller target from interference by the pairwise term (Q6).
4. ``grokking_still_moving``: are the 50k-step grokking-v1 runs still changing at the end (Q8)?

Run from the repository root, after extracting DR1 (see README.md):
``PYTHONPATH=. python scripts/analysis_v1/supplementary.py``. Writes ``supplementary.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tycho
from tycho.analysis.ricurve import RICurve

REPO = Path(__file__).resolve().parents[2]
assert Path(tycho.__file__).resolve().is_relative_to(REPO), f"wrong tycho package: {tycho.__file__}"


def _py(x):
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def operations(o: pd.DataFrame, r: pd.DataFrame, ev: pd.DataFrame) -> dict:
    t0 = o.created_at.min()
    x = r.merge(o[["object_id", "region", "largest_drop_step"]], on="object_id")
    g = o[o.region == "grokking-v1"][["object_id", "label"]]
    e = ev.merge(g, on="object_id")
    lc = e[(e.kind == "activation") & e.ri_before.isna()]
    strong = lc[lc.peak_ri >= 0.02]
    return {
        "objects_by_region_and_source": {f"{a}|{b}": int(n) for (a, b), n in o.groupby(["region", "source"]).size().items()},
        "hours_enqueue_span": (o.created_at.max() - t0) / 3600,
        "hours_last_object_finished": (o.finished_at.max() - t0) / 3600,
        "hours_last_ri_run_recorded": (r.created_at.max() - t0) / 3600,
        "summed_object_wall_time_hours": o.wall_time_s.sum() / 3600,
        "summed_wall_time_hours_by_region": (o.groupby("region").wall_time_s.sum() / 3600).to_dict(),
        "ri_points_per_run": {f"{a}|{b}|{c}": int(n) for (a, b, c), n in
                              x.groupby(["region", "kind", "n_points"]).size().items()},
        "grokking_objects_with_largest_drop": int(x[x.region == "grokking-v1"].largest_drop_step.notna().sum()),
        "grokking_left_censored_activations": int(len(lc)),
        "grokking_left_censored_strong": int(len(strong)),
        "grokking_runs_with_left_censored_strong": int(strong.object_id.nunique()),
        "grokking_runs": int(len(g)),
        "grokking_left_censored_strong_by_order": strong.order.value_counts().to_dict(),
        "grokking_left_censored_strong_median_ri_after": strong.ri_after.median(),
    }


def parity_lower_order(o, r, ev, extracted: Path) -> dict:
    p = o[o.region.str.startswith("parity")].merge(r[["object_id", "ricurve_file"]], on="object_id")
    rows = []
    for row in p.itertuples():
        c = RICurve.load(extracted / row.ricurve_file)
        gt = set(c.ground_truth[0]["subset"])
        e = ev[(ev.object_id == row.object_id) & (ev.kind != "extinction")]
        sets = e.subset_json.apply(lambda s: frozenset(json.loads(s)))
        proper = e[sets.apply(lambda s: s < gt and "yhat" in s and len(s) >= 2)]
        other = e[sets.apply(lambda s: not s <= gt)]
        onset = e[sets.apply(lambda s: s == gt)].step.min()
        best_f = 0.0
        for i, sub in enumerate(c.subsets):
            nm = {c.names[j] for j in sub}
            if nm < gt and "yhat" in nm and len(nm) >= 2:
                best_f = max(best_f, float(np.nanmax(np.abs(c.fourier[i]))))
        rows.append(dict(region=row.region, n_proper=len(proper),
                         proper_peak=proper.peak_ri.max() if len(proper) else 0.0,
                         proper_before_onset=int((proper.step < onset).sum()),
                         other_peak=other.peak_ri.max() if len(other) else 0.0, best_proper_f=best_f))
    d = pd.DataFrame(rows)
    out = {}
    for reg, x in d.groupby("region"):
        out[reg] = {"runs": len(x), "runs_with_proper_subset_event": int((x.n_proper > 0).sum()),
                    "runs_with_proper_peak_ge_0.01": int((x.proper_peak >= 0.01).sum()),
                    "max_proper_peak": x.proper_peak.max(), "max_proper_abs_f": x.best_proper_f.max(),
                    "proper_events": int(x.n_proper.sum()), "proper_events_before_planted_onset":
                        int(x.proper_before_onset.sum()),
                    "runs_with_other_peak_ge_0.01": int((x.other_peak >= 0.01).sum())}
    return out


def _parse(coupling: str):
    terms = []
    for term in coupling.split(";"):
        lags, b = term.split(":")
        terms.append((tuple(int(v) for v in lags.split(",")), float(b)))
    return terms


def _population(coupling: str, lags) -> float:
    """E_uniform[tanh(h) * prod s_l] over the 2^6 windows: the population Walsh coefficient of the lag set."""
    terms, tot = _parse(coupling), 0.0
    for s in itertools.product([-1, 1], repeat=6):
        h = sum(b * np.prod([s[l - 1] for l in L]) for L, b in terms)
        tot += np.tanh(h) * np.prod([s[l - 1] for l in lags])
    return tot / 64


def markov_magnitude(o, r, extracted: Path) -> list[dict]:
    m = o[o.region == "markov-v1"].merge(r[["object_id", "ricurve_file"]], on="object_id")
    cfg = m.task_cfg_json.apply(json.loads)
    m["coup"], m["ctx"] = cfg.apply(lambda c: c["couplings"]), cfg.apply(lambda c: c["train_context"])
    plan = [((1,), ["1:0.5", "1:1.0"]), ((1, 3), ["1,3:0.5", "1,3:0.8", "1,3:1.0", "1:1.0;1,3:0.8"]),
            ((1, 2, 3), ["1,2,3:1.0", "1:1.0;1,2,3:1.0"]), ((1, 2, 4), ["1,2,4:0.5", "1,2,4:1.0"]),
            ((1, 2, 3, 5), ["1,2,3,5:0.5", "1,2,3,5:1.0"])]
    rows = []
    for lags, coups in plan:
        for coup in coups:
            pv = _population(coup, lags)
            for ctx in ["chain", "iid"]:
                sub = m[(m.coup == coup) & (m.ctx == ctx)]
                curves = [RICurve.load(extracted / f) for f in sub.ricurve_file]
                for level in (0.10, 0.25, 0.50):
                    steps, cens = [], 0
                    for c in curves:
                        idx = tuple(sorted([c.names.index(f"x[t-{l}]") for l in lags] + [c.names.index("xhat")]))
                        f = c.fourier[c.index_of(idx)]
                        hit = np.where(f * np.sign(pv) >= level * abs(pv))[0]
                        steps.append(c.steps[hit[0]] if len(hit) else c.steps[-1])
                        cens += int(len(hit) == 0)
                    st = np.array(steps, dtype=float)
                    rows.append(dict(lags=list(lags), coupling=coup, context=ctx, level=level, population=pv,
                                     n=len(st), median=float(np.median(st)), lo=float(st.min()), hi=float(st.max()),
                                     censored=cens))
    return rows


def grokking_still_moving(o, extracted: Path) -> dict:
    g = o[o.region == "grokking-v1"]
    out = []
    for row in g.itertuples():
        lc = np.load(extracted / row.lightcurve_file)
        st, ta, tl = lc["step"], lc["test_acc"], lc["test_loss"]
        i25, i40 = int(np.argmin(np.abs(st - 25000))), int(np.argmin(np.abs(st - 40000)))
        out.append(dict(label=row.label, a25=ta[i25], a40=ta[i40], aend=ta[-1], s25=st[i25], s40=st[i40],
                        send=st[-1], l25=tl[i25], lend=tl[-1]))
    d = pd.DataFrame(out)
    res = {"eval_steps": {"near_25k": sorted(set(d.s25.tolist())), "near_40k": sorted(set(d.s40.tolist())),
                          "end": sorted(set(d.send.tolist()))}}
    for lab, x in d.groupby("label"):
        res[lab] = {"runs": len(x), "final_test_acc_median": x.aend.median(), "final_test_acc_max": x.aend.max(),
                    "test_acc_rise_25k_to_end_gt_0.05": int(((x.aend - x.a25) > 0.05).sum()),
                    "test_acc_rise_25k_to_end_gt_0.01": int(((x.aend - x.a25) > 0.01).sum()),
                    "test_acc_fall_25k_to_end_gt_0.01": int(((x.aend - x.a25) < -0.01).sum()),
                    "test_acc_rise_40k_to_end_gt_0.02": int(((x.aend - x.a40) > 0.02).sum()),
                    "test_loss_lower_at_end_than_25k": int((x.lend < x.l25).sum())}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases" / "DR1")
    ap.add_argument("--extracted", type=Path, default=REPO / "work" / "dr1")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v1_analysis" / "supplementary.json")
    args = ap.parse_args(argv)
    o = pd.read_parquet(args.release / "objects.parquet")
    r = pd.read_parquet(args.release / "ri_runs.parquet")
    ev = pd.read_parquet(args.release / "ri_events.parquet")
    result = {"description": __doc__.split("\n\n")[0],
              "operations": operations(o, r, ev),
              "parity_lower_order": parity_lower_order(o, r, ev, args.extracted),
              "markov_magnitude": markov_magnitude(o, r, args.extracted),
              "grokking_still_moving": grokking_still_moving(o, args.extracted)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_py(result), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(_py({k: v for k, v in result.items() if k != "markov_magnitude"}), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
