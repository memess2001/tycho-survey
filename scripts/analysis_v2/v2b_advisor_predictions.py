"""Exploratory, not pre-registered: check on DR2b the predictions about Part B that the advisor session wrote on
branch ``claude/lucid-archimedes-av5rk7`` before the confirmatory data existed.

Sources (``notes/discussion_2026-11-16_outside_reviews.md`` on that branch, §7 item 7):
* commit cf3287b (16 November 2026): predictions 1 and 2 and the "leave loss" measurement;
* commit 41d4d7a (17 November 2026): Antigravity's prediction about the leave loss.
The confirmatory runs started on 20 November 2026 (``started_at`` in ``releases/DR2b/objects.csv``).

1. The 1-layer controls end (step 12,000) with a repeat loss between 1.56 and 2.0 nats.
2. At d_model 96, before the phase change, the 2-layer runs' repeat loss stays within 0.2 nats of the median curve
   of the 1-layer controls. Two readings are reported: up to the last evaluation before the repeat loss first falls
   to ≤ 1.0 (the pre-registration's phase change), and up to the leave point.
3. Leave loss: the repeat loss at the last evaluation before it first drops by more than 0.3 nats within one
   evaluation step. Antigravity predicted that it rises or stays flat with d_model.

Usage: python scripts/analysis_v2/v2b_advisor_predictions.py [--release releases/DR2b] [--out FILE]
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def load_curves(release: Path) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    with tarfile.open(release / "lightcurves.tar.gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".npz"):
                z = np.load(io.BytesIO(tar.extractfile(m).read()))
                out[Path(m.name).stem] = {"step": z["step"], "loss_rep": z["x_loss_rep"]}
    return out


def leave_index(loss: np.ndarray, drop: float = 0.3) -> int:
    idx = np.where(loss[:-1] - loss[1:] > drop)[0]
    return int(idx[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", default=str(REPO / "releases" / "DR2b"))
    ap.add_argument("--out", default=str(REPO / "docs" / "results" / "v2_analysis" / "v2b_advisor_predictions.json"))
    args = ap.parse_args()
    release = Path(args.release)
    objs = pd.read_csv(release / "objects.csv")
    cfg = objs.task_cfg_json.map(json.loads)
    objs = objs.assign(d_model=cfg.map(lambda c: c["d_model"]), n_layer=cfg.map(lambda c: c["n_layer"]),
                       p_repeat=cfg.map(lambda c: c["p_repeat"]))
    curves = load_curves(release)

    one = objs[objs.n_layer == 1]
    main_runs = objs[(objs.n_layer == 2) & (objs.p_repeat == 1.0)]
    steps = curves[one.object_id.iloc[0]]["step"]
    for oid in objs.object_id:
        assert np.array_equal(curves[oid]["step"], steps), oid

    final1 = {oid: float(curves[oid]["loss_rep"][-1]) for oid in one.object_id}
    p1 = {"rule": "1-layer final repeat loss in [1.56, 2.0]", "values": sorted(final1.values()),
          "n_in": sum(1.56 <= v <= 2.0 for v in final1.values()), "n": len(final1)}

    med1 = np.median(np.array([curves[o]["loss_rep"] for o in one.object_id]), axis=0)
    p2 = {"rule": "d_model 96: max |2-layer - 1-layer median| <= 0.2 before the phase change", "runs": []}
    for oid in main_runs[main_runs.d_model == 96].object_id:
        loss = curves[oid]["loss_rep"]
        k_pc = int(np.argmax(loss <= 1.0))
        k_leave = leave_index(loss)
        dev_pc = np.abs(loss[:k_pc] - med1[:k_pc])
        dev_leave = np.abs(loss[:k_leave + 1] - med1[:k_leave + 1])
        p2["runs"].append({"object_id": oid,
                           "to_phase_change": {"max_dev": float(dev_pc.max()), "at_step": int(steps[dev_pc.argmax()])},
                           "to_leave_point": {"max_dev": float(dev_leave.max()),
                                              "at_step": int(steps[dev_leave.argmax()])}})
    for reading in ("to_phase_change", "to_leave_point"):
        v = [r[reading]["max_dev"] for r in p2["runs"]]
        p2[reading] = {"n_within": sum(x <= 0.2 for x in v), "n": len(v), "min": min(v), "max": max(v)}
    p2["one_layer_median_vs_two_layer_median"] = {
        int(steps[k]): {"one_layer_median": float(med1[k]),
                        "two_layer_d96_median": float(np.median([curves[o]["loss_rep"][k]
                                                                 for o in main_runs[main_runs.d_model == 96].object_id]))}
        for k in range(len(steps)) if steps[k] in (57, 103, 166, 211, 237, 267, 301)}

    p3 = {"rule": "leave loss rises or stays flat with d_model (Antigravity)", "by_size": {}}
    for d in sorted(main_runs.d_model.unique()):
        vals, at = [], []
        for oid in main_runs[main_runs.d_model == d].object_id:
            loss = curves[oid]["loss_rep"]
            j = leave_index(loss)
            vals.append(float(loss[j]))
            at.append(int(steps[j]))
        p3["by_size"][int(d)] = {"n": len(vals), "median": float(np.median(vals)), "min": min(vals), "max": max(vals),
                                 "leave_step_median": float(np.median(at)), "leave_step_min": min(at),
                                 "leave_step_max": max(at)}

    result = {"status": "exploratory; not pre-registered", "release": str(release.relative_to(REPO)
                                                                          if release.is_relative_to(REPO) else release),
              "first_run_started_utc": pd.Timestamp(objs.started_at.min(), unit="s", tz="UTC").isoformat(),
              "P1": p1, "P2": p2, "P3": p3}
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(f"P1: {p1['n_in']}/{p1['n']} in [1.56, 2.0]; values {np.round(p1['values'], 3).tolist()}")
    for reading in ("to_phase_change", "to_leave_point"):
        s = p2[reading]
        print(f"P2 ({reading}): {s['n_within']}/{s['n']} within 0.2; max deviation {s['min']:.3f} to {s['max']:.3f}")
    for d, s in p3["by_size"].items():
        print(f"P3 d_model {d}: leave loss median {s['median']:.3f} ({s['min']:.3f} to {s['max']:.3f}); "
              f"steps {s['leave_step_min']} to {s['leave_step_max']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
