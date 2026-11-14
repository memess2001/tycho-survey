"""Sky v2 A0, calibration experiment E8: marginal and conditional RI under correlated probes, when the population
value is exactly 0. See ``tycho/calibration/correlated.py`` (module docstring) for the design.

For every grid cell it reports the false-positive rate of the χ²_1 p-value, the KS distance of the p-values from
U(0, 1), and the noise floor (quantiles of the plug-in and corrected values) next to the χ² predictions. The
recommended min_effect for conditional RI follows v1's rule: max(floor 0.001, 0.99 quantile of the corrected null
values), taken over the probe kinds, correlations and nuisance strengths at each (n, table size).

Run from the repository root: ``PYTHONPATH=. python scripts/analysis_v2/a0_e8_calibration.py [--reps 300]``.
Writes ``docs/results/v2_analysis/a0_e8_calibration.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import tycho
from tycho.calibration.correlated import e8_cell

REPO = Path(__file__).resolve().parents[2]
assert Path(tycho.__file__).resolve().is_relative_to(REPO), f"wrong tycho package: {tycho.__file__}"
PROBES = [("uniform", 0.0), ("markov", 0.5), ("markov", 0.9), ("equicorr", 0.3), ("equicorr", 0.6)]
TABLES = [4, 6, 8]
NS = [1024, 4096, 16384]
NUISANCE = [0.5, 2.0]
FLOOR = 0.001
QUANTILE_KEY = "corrected_q99"


def _run(args):
    logging.getLogger("tycho").setLevel(logging.ERROR)
    probe, rho, d, n, nu, reps, seed = args
    return e8_cell(probe, rho, d, n, nu, reps, seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "a0_e8_calibration.json")
    args = ap.parse_args(argv)
    grid = [(p, r, d, n, nu, args.reps, args.seed)
            for (p, r), d, n, nu in itertools.product(PROBES, TABLES, NS, NUISANCE)]
    grid.sort(key=lambda g: -g[2] * g[3])  # largest first
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        cells = pool.map(_run, grid, chunksize=1)
    cells.sort(key=lambda c: (c["n"], c["d_table"], c["probe"], c["rho"], c["nuisance_scale"]))
    summary = []
    for n, d in itertools.product(NS, TABLES):
        sub = [c for c in cells if c["n"] == n and c["d_table"] == d]
        row = {"n": n, "d_table": d}
        for label in ("cond_pair", "cond_triple", "marg_pair", "marg_triple"):
            vals = [c[label] for c in sub if label in c]
            row[label] = {"max_fpr_alpha_0_01": max(v["fpr_alpha_0_01"] for v in vals),
                          "mean_fpr_alpha_0_01": sum(v["fpr_alpha_0_01"] for v in vals) / len(vals),
                          "max_ks": max(v["ks_uniform"] for v in vals),
                          "max_plugin_mean_over_chi2_mean": max(v["plugin_mean"] / v["chi2_mean"] for v in vals),
                          "max_corrected_q99": max(v[QUANTILE_KEY] for v in vals)}
        noise = max(row["cond_pair"]["max_corrected_q99"], row["cond_triple"]["max_corrected_q99"])
        row["recommended_min_effect_conditional"] = max(FLOOR, noise)
        summary.append(row)
    result = {"description": __doc__.split("\n\n")[0], "reps": args.reps, "seed": args.seed,
              "grid": {"probes": PROBES, "tables": TABLES, "n": NS, "nuisance": NUISANCE},
              "max_population_value": max(c["max_population_value"] for c in cells),
              "summary": summary, "cells": cells, "seconds": round(time.time() - t0, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for r in summary:
        print(r["n"], r["d_table"], "min_effect_cond=%.5f" % r["recommended_min_effect_conditional"],
              {k: (round(v["max_fpr_alpha_0_01"], 4), round(v["max_corrected_q99"], 6)) for k, v in r.items()
               if isinstance(v, dict)})
    print("max population value of the nulls:", result["max_population_value"], " seconds:", result["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
