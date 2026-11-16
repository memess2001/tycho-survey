"""Exploratory side check on DR2a (not pre-registered): do the cheap light-curve indicators lead the transition?

Asked in a review of the project: does a "photometric" indicator (eNTK statistics, effective rank, distance from
initialisation, parameter norm) move *before* the function changes, by a constant factor, so that it could announce
an interaction before the expensive measurement sees it?

Runs: the single-coupling Markov runs with iid contexts (12 configurations × 10 seeds). They have one clean
transition and no intermediate projection, and its time moves by a factor of about 20 with the order and strength of
the coupling. For each run:

* ``t_loss(f)``: the step at which the exact loss under D (``x_lossD``) has completed a fraction f of its total fall;
* ``t_ind(f)``: the step at which an indicator has completed a fraction f of its total change (from its first value to
  the mean of its last five), both interpolated linearly in log(step) (step 0 counted as 0.5);
* ``t_peak``: the step at which the indicator changes fastest per unit of log(step);
* the share of evaluation intervals in which the indicator moves in the direction of its total change.

A precursor would have ``t_ind(f) / t_loss(f)`` well below 1 for small f, with about the same value in every
configuration. The eNTK series exist at 15 of the 72 evaluation steps (a factor of about 1.9 apart), so a ratio
between 0.5 and 2 cannot be resolved for them; the other indicators are stored at every evaluation.

Usage, from the repository root: ``python scripts/analysis_v2/lightcurve_lead_lag.py`` (options ``--release``,
``--work``, ``--out``). Numpy and the standard library only. Writes
``docs/results/v2_analysis/lightcurve_lead_lag.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
INDICATORS = ("ntk_align", "ntk_top_eig", "ntk_trace", "ntk_erank", "erank_hidden", "erank_weights",
              "dist_from_init", "param_norm")
FRACS = (0.1, 0.25, 0.5)


def frac_time(steps: np.ndarray, y: np.ndarray, frac: float) -> float:
    y0, y1 = float(y[0]), float(np.mean(y[-5:]))
    if not (math.isfinite(y0) and math.isfinite(y1)) or abs(y1 - y0) < 1e-12:
        return float("nan")
    prog = (y - y0) / (y1 - y0)
    for i in range(1, len(prog)):
        if prog[i] >= frac and prog[i - 1] < frac:
            a, b = math.log(max(float(steps[i - 1]), 0.5)), math.log(float(steps[i]))
            return math.exp(a + (frac - prog[i - 1]) / (prog[i] - prog[i - 1]) * (b - a))
    return float("nan")


def peak_time(steps: np.ndarray, y: np.ndarray) -> float:
    ls = np.log(np.maximum(steps, 0.5))
    i = int(np.nanargmax(np.abs(np.diff(y) / np.diff(ls))))
    return math.exp(0.5 * (ls[i] + ls[i + 1]))


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok])))[0, 1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases" / "DR2a")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "dr2a")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "lightcurve_lead_lag.json")
    args = ap.parse_args(argv)
    if not (args.work / "lightcurves").is_dir():
        args.work.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.release / "lightcurves.tar.gz") as tar:
            tar.extractall(args.work, filter="data")

    rows = defaultdict(list)
    with open(args.release / "objects.csv", newline="", encoding="utf-8") as fh:
        for o in csv.DictReader(fh):
            if o["task"] != "markov_lang":
                continue
            cfg = json.loads(o["task_cfg_json"])
            if cfg["train_context"] != "iid" or ";" in cfg["couplings"]:
                continue
            with np.load(args.work / "lightcurves" / f"{o['object_id']}.npz", allow_pickle=False) as f:
                step = np.asarray(f["step"], dtype=float)
                r = {"seed": int(o["seed"])}
                for fr in FRACS:
                    r[f"loss@{fr}"] = frac_time(step, np.asarray(f["x_lossD"], dtype=float), fr)
                for name in INDICATORS:
                    y = np.asarray(f[name], dtype=float)
                    ok = np.isfinite(y)
                    for fr in FRACS:
                        r[f"{name}@{fr}"] = frac_time(step[ok], y[ok], fr)
                    r[f"{name}@peak"] = peak_time(step[ok], y[ok])
                    r[f"{name}@mono"] = float(np.mean(np.sign(np.diff(y[ok])) == np.sign(y[ok][-1] - y[ok][0])))
            rows[cfg["couplings"]].append(r)

    med = lambda c, key: float(np.nanmedian([r[key] for r in rows[c]]))  # noqa: E731
    key_of = lambda c: (len(c.split(":")[0].split(",")), float(c.split(":")[1]))  # noqa: E731
    out = {"status": "exploratory, post hoc, not pre-registered", "release": args.release.name,
           "indicators": list(INDICATORS), "configurations": {}}
    for c in sorted(rows, key=key_of):
        entry = {"n_seeds": len(rows[c]), "t_loss": {str(fr): med(c, f"loss@{fr}") for fr in FRACS}, "indicators": {}}
        for name in INDICATORS:
            entry["indicators"][name] = {
                "t": {str(fr): med(c, f"{name}@{fr}") for fr in FRACS},
                "ratio": {str(fr): float(np.nanmedian([r[f"{name}@{fr}"] / r[f"loss@{fr}"] for r in rows[c]]))
                          for fr in FRACS},
                "t_peak": med(c, f"{name}@peak"), "monotone_share": med(c, f"{name}@mono"),
                "spearman_t25_with_loss50_across_seeds": spearman([r[f"{name}@0.25"] for r in rows[c]],
                                                                 [r["loss@0.5"] for r in rows[c]]),
            }
        out["configurations"][c] = entry
    ratios = {fr: [e["indicators"][n]["ratio"][str(fr)] for e in out["configurations"].values() for n in INDICATORS]
              for fr in FRACS}
    out["summary"] = {
        "cells": len(ratios[0.25]),
        **{f"ratio_below_0.8_at_{fr}": int(np.sum(np.asarray(ratios[fr]) < 0.8)) for fr in FRACS},
        **{f"smallest_ratio_at_{fr}": float(np.nanmin(ratios[fr])) for fr in FRACS},
    }

    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    args.out.write_text(json.dumps(clean(out), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))
    print(f"{'couplings':>12s} {'t_loss 25%':>10s} " + " ".join(f"{n[:12]:>12s}" for n in INDICATORS)
          + "   (ratio t_ind / t_loss at 25 %)")
    for c, e in out["configurations"].items():
        print(f"{c:>12s} {e['t_loss']['0.25']:10.0f} "
              + " ".join(f"{e['indicators'][n]['ratio']['0.25']:12.2f}" for n in INDICATORS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
