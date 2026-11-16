"""Exploratory analysis of the Part B pilot (data release DR2b-pilot). Not confirmatory: its results are disclosed in
the Part B pre-registration.

Per object: configuration, final loss on repeated tokens, the largest previous-token score among layer-0 heads and
the largest induction score among last-layer heads, the first evaluation step with the repeat loss at most half its
initial value, and wall time. For the 2-layer runs trained on repeats: the top head pairs at the last checkpoint by θ,
bounded η and Gaussian η (conditional reference, uniform ablation design), and whether the top pair is a
(previous-token head, induction head) pair by the attention scores (≥ 0.5).

Usage: ``python scripts/analysis_v2/v2b_pilot_analysis.py --release releases/DR2b-pilot --work work/dr2b_pilot
--out docs/results/v2_analysis/v2b_pilot.json`` (extracts the archives into --work when needed).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tycho.ri.cells import cell_stats, projection_information  # noqa: E402
from tycho.tasks.patterns import pattern_bits  # noqa: E402

ROLE = 0.5  # attention score that marks a previous-token or induction head


def head_pairs(logodds: np.ndarray, names: list[str]) -> list[dict]:
    H = len(names)
    p = 1.0 / (1.0 + np.exp(-logodds))
    cs = cell_stats(pattern_bits(H), p, [tuple(range(H))])[0]
    out = []
    for A in itertools.combinations(range(H), 2):
        out.append({"pair": [names[i] for i in A],
                    "theta": projection_information(cs, A, weights="uniform"),
                    "eta_b": projection_information(cs, A, link="linear", weights="uniform"),
                    "eta_g": projection_information(cs, A, link="linear", weights="uniform", eta_value="gaussian")})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", default=str(REPO / "releases" / "DR2b-pilot"))
    ap.add_argument("--work", default=str(REPO / "work" / "dr2b_pilot"))
    ap.add_argument("--out", default=str(REPO / "docs" / "results" / "v2_analysis" / "v2b_pilot.json"))
    args = ap.parse_args(argv)
    rel, work = Path(args.release), Path(args.work)
    for name in ("patterns", "lightcurves"):
        if not (work / name).is_dir():
            work.mkdir(parents=True, exist_ok=True)
            with tarfile.open(rel / f"{name}.tar.gz") as tar:
                tar.extractall(work, filter="data")
    objs = pd.read_parquet(rel / "objects.parquet")
    runs = []
    for _, r in objs.iterrows():
        cfg = json.loads(r.task_cfg_json)
        nl, nh = int(cfg["n_layer"]), int(cfg["n_head"])
        lc = np.load(work / "lightcurves" / f"{r.object_id}.npz")
        prev = [float(lc[f"x_prev_l0h{h}"][-1]) for h in range(nh)]
        ind = [float(lc[f"x_ind_l{nl - 1}h{h}"][-1]) for h in range(nh)]
        lr, st = lc["x_loss_rep"], lc["step"]
        half = np.flatnonzero(lr <= 0.5 * lr[0])
        rec = {"object_id": r.object_id, "seed": int(r.seed), "n_layer": nl, "n_head": nh,
               "rep_min": int(cfg["rep_min"]), "p_repeat": float(cfg["p_repeat"]), "loss_rep": float(lr[-1]),
               "prev_max": max(prev), "ind_max": max(ind), "step_half_loss": int(st[half[0]]) if half.size else None,
               "wall_time_s": float(r.wall_time_s),
               "circuit": "full" if max(prev) >= ROLE and max(ind) >= ROLE else
                          ("partial" if max(prev) >= 0.2 and max(ind) >= 0.2 else "none")}
        if nl == 2 and cfg["p_repeat"] == 1.0:
            z = np.load(work / "patterns" / f"{r.object_id}.npz")
            names = [str(n) for n in z["input_names"]]
            pairs = head_pairs(z["logodds"][-1], names)
            prev_heads = {f"l0h{h}" for h in range(nh) if prev[h] >= ROLE}
            ind_heads = {f"l1h{h}" for h in range(nh) if ind[h] >= ROLE}
            rec["heads"] = {"prev_token": sorted(prev_heads), "induction": sorted(ind_heads)}
            for v in ("theta", "eta_b", "eta_g"):
                top = sorted(pairs, key=lambda x: -x[v])[:3]
                rec[f"top_pairs_{v}"] = [{"pair": t["pair"], "value": t[v]} for t in top]
                a, b = top[0]["pair"]
                rec[f"top_pair_is_circuit_{v}"] = bool({a, b} & prev_heads and {a, b} & ind_heads)
        runs.append(rec)
    df = pd.DataFrame(runs)
    rep2 = df[(df.n_layer == 2) & (df.p_repeat == 1.0)]
    summary = {
        "n_objects": len(df),
        "circuit_counts_2layer_repeat": rep2.groupby(["rep_min", "n_head"])["circuit"].value_counts().unstack(
            fill_value=0).reset_index().to_dict(orient="records"),
        "controls_final_loss_rep": {
            "no_repeat": [float(df[df.p_repeat == 0.0].loss_rep.min()), float(df[df.p_repeat == 0.0].loss_rep.max())],
            "one_layer": [float(df[(df.n_layer == 1) & (df.p_repeat == 1.0)].loss_rep.min()),
                          float(df[(df.n_layer == 1) & (df.p_repeat == 1.0)].loss_rep.max())]},
        "wall_time_s_by_heads": df.assign(H=df.n_layer * df.n_head).groupby("H").wall_time_s.mean().round(0).to_dict(),
        "top_pair_is_circuit_in_full_circuits": {
            v: int(rep2[rep2.circuit == "full"][f"top_pair_is_circuit_{v}"].sum()) for v in ("theta", "eta_b", "eta_g")},
        "n_full_circuits": int((rep2.circuit == "full").sum()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=1, default=str) + "\n")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
