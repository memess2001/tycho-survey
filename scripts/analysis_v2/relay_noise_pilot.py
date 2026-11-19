"""Pilot (exploratory; local runs): what does the noise scale lr/√B act on?

``relay_period_pilot.py`` found that at rep_min 12 the circuit an unperturbed run ends with is ordered by the
learning rate over the root of the batch size (Adam): textbook circuits at 0.88e-4, relays at 1.77e-4, whatever the
learning rate and the batch size are separately. The reviewer's seventh reply (19 November, to commit 8351c3c) gives
an account of this in three steps, supported by measurements at steps 100 to 300 of five settings:

1. Early on Adam's denominators are set by the gradient noise, so a step moves a weight by about the learning rate
   whatever the batch size, with a drift of lr·√B·(mean over standard deviation of the gradient). Counted in
   sequences seen, N = B·t, the drift and the random walk of a weight are then both lr/√B times a function of N.
   (This is the square-root scaling rule for Adam of Malladi, Lyu, Panigrahi and Arora 2022, arXiv:2205.10287,
   Definition 5.2 and Theorem 5.3, without that rule's adjustment of the two decay rates and of ε. The reply does
   not cite it.)
2. The residual stream is normalised before each attention layer and before the unembedding, so only the direction
   of the embeddings matters; once their random walk exceeds the initial norm, the direction turns at a rate that
   no longer depends on the learning rate. Progress is therefore counted in sequences: a circuit forms after about
   the same N whatever the learning rate and the batch size.
3. The query and key weights are not normalised away. At equal N they have moved in proportion to lr/√B, so a
   larger lr/√B sharpens layer 0's attention on the band earlier in N, before layer 1 has begun to match content,
   and the relay is locked in.

Three checks.

a. Equal N, no new runs: the attention statistics and the repeat loss of the eight cells of
   ``relay_period_pilot.py`` at equal numbers of sequences. Steps 1 and 2 predict that cells with the same lr/√B
   coincide.
b. Displacement: the first 38,400 sequences of the eight cells again (these repeat the cached runs exactly),
   recording for seven groups of weights the norm, the distance from the initial weights and the cosine to them,
   every 3,200 sequences. Step 1 predicts a distance at equal N proportional to lr/√B: fitted as
   lr^a · B^b across the eight cells, a = 1 and b = −1/2 (an Adam without noise would give b = −1; a distance set
   by the number of steps alone, b = −1/2 with a = 1 is the same as a random walk of lr per step).
c. Intervention on the baseline (learning rate 1e-3, batch 32: 0 of 6 textbook circuits; at learning rate 5e-4:
   6 of 6): the learning rate is halved for layer 0's query and key weights only ("qk0_half"), and for every
   weight except those ("rest_half"). If step 3 is what decides (the noise acts through layer 0's hold on the
   band), qk0_half gives textbook circuits and rest_half relays. If the noise acts instead by keeping layer 1's
   content match from forming, it is the other way round. Both, or neither, flipping would mean that no single
   group carries the effect.

Recorded before the runs of (b) and (c) were started: the equal-N comparison (a) had been made (cells with batch 8
and 32 agree at equal N, cells with batch 128 lag by a few thousand sequences); neither outcome of (c) was known
or expected with any confidence.

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/relay_noise_pilot.py
--jobs 4`` (``--skip-runs`` analyses the cache). Needs the cached runs and the output of
``relay_period_pilot.py``. Runs are cached in ``work/relay_band``. Writes
``docs/results/v2_analysis/relay_noise_pilot.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("relay_period_pilot", Path(__file__).with_name("relay_period_pilot.py"))
pp = importlib.util.module_from_spec(_spec)
sys.modules["relay_period_pilot"] = pp
_spec.loader.exec_module(pp)
wp, rb, fp = pp.wp, pp.rb, pp.fp

#: the eight cells of the noise table of relay_period_pilot.py (rep_min 12, d_model 64), by lr/√B
CELLS = ("lr2.5e-4_b8", "lr5e-4", "b128", "b64", "lr5e-4_b8", "base", "lr2e-3_b128", "lr2e-3")
LOW, HIGH = ("lr2.5e-4_b8", "lr5e-4", "b128"), ("lr5e-4_b8", "base", "lr2e-3_b128")   # lr/√B 0.88e-4 and 1.77e-4
N_STEP, N_LAST = 3_200, 38_400                          # sequences between measurements, and in all
GROUPS = ("pos", "tok", "qk0", "ov0", "qk1", "ov1", "head")
KEYS = ("band_l0", "band_l1", "source_l1", "offsets_1_2_l0", "loss_rep")
#: intervention arms: learning-rate factors by group of weights (see relay_band_pilot.weight_groups)
ARMS = {"qk0_half": {"qk0": 0.5}, "rest_half": {"all": 0.5, "qk0": 1.0}}


def cell_of(setting: str) -> tuple[float, int]:
    change = pp.SETTINGS[setting]
    return change.get("lr", 1e-3), change.get("batch_size", 32)


def displacement_stats():
    """A measurement function for ``rb.train``: the band statistics and, by group of weights, the norm, the distance
    from the first weights it sees (step 0) and the cosine to them."""
    import torch

    first = {}

    def fn(model, task) -> dict:
        out = rb.band_stats(model, task)
        for g, parts in rb.weight_groups(model).items():
            w = torch.cat([p.detach()[rows].reshape(-1) for p, rows in parts])
            w0 = first.setdefault(g, w.clone())
            out[f"norm_{g}"] = float(w.norm())
            out[f"moved_{g}"] = float((w - w0).norm())
            out[f"cos_{g}"] = float((w @ w0) / (w.norm() * w0.norm()))
        return out

    return fn


def clock_jobs() -> list[dict]:
    jobs = []
    for cell in CELLS:
        _, batch = cell_of(cell)
        for j in pp.arm_jobs(f"{cell}_natural"):
            jobs.append({**j, "cell": cell, "steps": N_LAST // batch, "every": N_STEP // batch})
    return jobs


def run_clock(args: tuple[dict, str]) -> str:
    job, work = args
    out = Path(work) / f"noise_clock_{job['cell']}_{job['base']}.json"
    if out.exists():
        return f"cached {out.name}"
    at = tuple(range(0, job["steps"] + 1, job["every"]))
    _, _, stats, _ = rb.train(job, job["steps"], at, block_from=0, block_until=0, layer=0, forbid="band",
                              stats_fn=displacement_stats())
    out.write_text(json.dumps({"cell": job["cell"], "run": job["base"], "step": list(at),
                               **{k: [stats[s][k] for s in at] for k in stats[0]}}) + "\n", encoding="utf-8")
    return f"done clock {job['cell']} {job['base']}"


def arm_jobs(arm: str) -> list[dict]:
    return [{**j, "name": f"noise_{arm}_{j['base']}", "arm": f"noise_{arm}", "steps": pp.FULL_STEPS,
             "train": {**j["train"], "steps": pp.FULL_STEPS}, "lr_scale": ARMS[arm]}
            for j in pp.arm_jobs("base_natural")]


def power_fit(cells: dict, value: dict) -> dict:
    """value ~ lr^a · B^b across the cells (least squares on the logarithms)."""
    names = [c for c in cells if value.get(c) and value[c] > 0]
    x = np.array([[1.0, math.log(cells[c][0]), math.log(cells[c][1])] for c in names])
    y = np.log([value[c] for c in names])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"lr_exponent": float(coef[1]), "batch_exponent": float(coef[2]),
            "rms_log_residual": float(np.sqrt(np.mean((x @ coef - y) ** 2))), "cells": len(names)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_band")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    ap.add_argument("--only", choices=("clock", "arms"), help="run only this part")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_noise_pilot.json")
    args = ap.parse_args(argv)
    clock = clock_jobs()
    arms = {arm: arm_jobs(arm) for arm in ARMS}
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            if args.only != "arms":
                for msg in pool.imap(run_clock, [(j, str(args.work)) for j in clock]):
                    print(msg, flush=True)
            if args.only != "clock":
                for msg in pool.imap(pp.run_arm, [(j, str(args.work)) for js in arms.values() for j in js]):
                    print(msg, flush=True)
    cells = {c: cell_of(c) for c in CELLS}
    grid = list(range(0, N_LAST + 1, N_STEP))

    # (a) the cached unperturbed runs at equal numbers of sequences
    cached = {}
    for cell, (lr, batch) in cells.items():
        js = [j for j in pp.arm_jobs(f"{cell}_natural") if (args.work / f"period_{j['name']}.npz").exists()]
        cs = [pp.curves(j, args.work) for j in js]
        cached[cell] = {k: [float(np.median([np.interp(n / batch, c[0], c[1][k]) for c in cs])) for n in grid]
                        for k in KEYS} if cs else None
    equal_n = {"sequences": grid, "cells": {
        c: {"learning_rate": cells[c][0], "batch": cells[c][1], "lr_over_root_batch": cells[c][0] / math.sqrt(
            cells[c][1]), **cached[c]} for c in CELLS if cached[c]}}
    spread = {}
    if all(cached[c] for c in LOW + HIGH):
        for k in KEYS:
            lo = np.array([cached[c][k] for c in LOW])
            hi = np.array([cached[c][k] for c in HIGH])
            spread[k] = {"within_low": (lo.max(0) - lo.min(0)).tolist(),
                         "within_high": (hi.max(0) - hi.min(0)).tolist(), "between": (hi.mean(0) - lo.mean(0)).tolist(),
                         "within_low_batch_8_and_32": np.abs(lo[0] - lo[1]).tolist(),
                         "within_high_batch_8_and_32": np.abs(hi[0] - hi[1]).tolist()}
        # how far the batch-128 cell of each level is behind the other two, in sequences: the shift that brings its
        # band_l1 curve onto the mean of the other two (over the stretch where the curves rise)
        lag = {}
        for name, group in (("low", LOW), ("high", HIGH)):
            ref = np.mean([cached[c]["band_l1"] for c in group[:2]], axis=0)
            mine = np.array(cached[group[2]]["band_l1"])
            top = int(np.argmax(ref))                    # up to the reference curve's maximum
            shifts = np.arange(0, 12_801, 400)
            err = [float(np.mean((np.interp(np.array(grid[:top + 1]) + s, grid, mine) - ref[:top + 1]) ** 2))
                   for s in shifts]
            lag[name] = {"cell": group[2], "sequences": int(shifts[int(np.argmin(err))]),
                         "steps_of_batch_128": int(shifts[int(np.argmin(err))]) // 128}
        equal_n["batch_128_lag"] = lag
    equal_n["spread"] = spread

    # (b) displacement of the weights at equal numbers of sequences
    runs = {}
    for j in clock:
        f = args.work / f"noise_clock_{j['cell']}_{j['base']}.json"
        if f.exists():
            runs.setdefault(j["cell"], []).append(json.loads(f.read_text()))
    measured = {c: {k: np.median([r[k] for r in rs], axis=0).tolist() for k in rs[0]
                    if k not in ("cell", "run", "step")} for c, rs in runs.items() if len(rs) == 6}
    repeat = None                                        # the new short runs against the cached ones
    if measured and all(cached.get(c) for c in measured):
        repeat = float(max(abs(a - b) for c in measured for k in KEYS
                           for a, b, n in zip(measured[c][k], cached[c][k], grid)
                           if (n / cells[c][1]) % 50 == 0))
    displacement = {"sequences": grid, "cells": measured, "largest_difference_from_the_cached_runs": repeat,
                    "fit_lr_a_times_batch_b": {}, "moved_times_root_batch_over_lr": {},
                    "one_minus_cosine_fit": {}}
    if len(measured) == len(CELLS):
        for g in GROUPS:
            fits, coll, turn = {}, {}, {}
            for i, n in enumerate(grid):
                if i == 0:
                    continue
                fits[str(n)] = power_fit(cells, {c: measured[c][f"moved_{g}"][i] for c in CELLS})
                scaled = {c: measured[c][f"moved_{g}"][i] * math.sqrt(cells[c][1]) / cells[c][0] for c in CELLS}
                coll[str(n)] = {"by_cell": scaled, "largest_over_smallest": max(scaled.values()) / min(scaled.values()),
                                "largest_over_smallest_with_lr_over_batch": (lambda v: max(v) / min(v))(
                                    [measured[c][f"moved_{g}"][i] * cells[c][1] / cells[c][0] for c in CELLS])}
                turn[str(n)] = power_fit(cells, {c: 1.0 - measured[c][f"cos_{g}"][i] for c in CELLS})
            displacement["fit_lr_a_times_batch_b"][g] = fits
            displacement["moved_times_root_batch_over_lr"][g] = coll
            displacement["one_minus_cosine_fit"][g] = turn

    # (c) the baseline with the learning rate halved for one group of weights
    samples = fp.sample(12, 32)
    reference = json.loads((REPO / "docs" / "results" / "v2_analysis" / "relay_period_pilot.json").read_text())
    intervention = {"reference": {
        "learning_rate_1e-3": {k: reference["noise_scale"]["cells"]["base"][k] for k in ("textbook", "of")},
        "learning_rate_5e-4": {k: reference["noise_scale"]["cells"]["lr5e-4"][k] for k in ("textbook", "of")}},
        "arms": {}}
    rows_out = {}
    for arm, js in arms.items():
        have = [j for j in js if (args.work / f"period_{j['name']}.npz").exists()]
        rows = []
        for j in have:
            r = wp.row_of({**j, "name": f"period_{j['name']}"}, args.work, samples)
            step, st = pp.curves(j, args.work)
            r.update({"name": j["name"], "d1_l0_max": float(st["d1_l0_max"][-1]),
                      "d2_l0_max": float(st["d2_l0_max"][-1]),
                      **{f"{k}_at_equal_sequences": [float(np.interp(n / 32, step, st[k])) for n in grid]
                         for k in KEYS}})
            rows.append(r)
        rows_out[arm] = rows
        intervention["arms"][arm] = {
            "learning_rate_factors": ARMS[arm], "textbook": int(sum(r["textbook"] for r in rows)),
            "relays": int(sum(r["formed"] and not r["textbook"] for r in rows)),
            "not_formed": int(sum(not r["formed"] for r in rows)), "of": len(rows), "complete": len(rows) == len(js),
            "by_heads": {f"h{nh}": int(sum(r["textbook"] for r in rows if r["n_head"] == nh)) for nh in rb.HEADS},
            "relay_deltas": [r["relay_delta"] for r in rows],
            "deep_decoy_over_clean": [r["deep_decoy_over_clean"] for r in rows],
            "first_step_with_loss_rep_below_2": [r["first_step_with_loss_rep_below_2"] for r in rows],
            "medians_at_equal_sequences": {k: np.median([r[f"{k}_at_equal_sequences"] for r in rows], axis=0).tolist()
                                           for k in KEYS} if rows else {}}

    result = {"status": "exploratory pilot; local runs; not independently recomputed",
              "missing": {"clock": [f"{j['cell']} {j['base']}" for j in clock if not (
                  args.work / f"noise_clock_{j['cell']}_{j['base']}.json").exists()],
                          "arms": [j["name"] for js in arms.values() for j in js if not (
                              args.work / f"period_{j['name']}.npz").exists()]},
              "equal_numbers_of_sequences": equal_n, "displacement": displacement, "intervention": intervention,
              "runs": rows_out}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    pick = [i for i, n in enumerate(grid) if n in (6400, 12800, 19200, 25600, 38400)]
    print("== (a) cached runs at equal numbers of sequences N (medians of six runs); N =", [grid[i] for i in pick])
    for k in KEYS:
        for c in CELLS:
            if cached[c]:
                print(f"   {k:>15s} {c:>12s} lr/sqrt(B) {cells[c][0] / math.sqrt(cells[c][1]):.2e}: "
                      + " ".join(f"{cached[c][k][i]:7.3f}" for i in pick))
    if spread:
        for k in ("band_l0", "band_l1"):
            print(f"   {k}: within the low level", [round(spread[k]["within_low"][i], 3) for i in pick],
                  "| within the high level", [round(spread[k]["within_high"][i], 3) for i in pick],
                  "| between", [round(spread[k]["between"][i], 3) for i in pick])
        print("   batch-128 cells behind the other two of their level:", equal_n["batch_128_lag"])
    if displacement["fit_lr_a_times_batch_b"]:
        print("== (b) distance from the initial weights at equal N, fitted as lr^a B^b over the eight cells "
              "(largest difference of the new runs from the cached ones:", repeat, ")")
        for g in GROUPS:
            f = displacement["fit_lr_a_times_batch_b"][g]
            c = displacement["moved_times_root_batch_over_lr"][g]
            print(f"   {g:>5s}: " + " | ".join(
                f"N {n}: a {f[str(n)]['lr_exponent']:+.2f} b {f[str(n)]['batch_exponent']:+.2f} (spread of "
                f"moved*sqrt(B)/lr x{c[str(n)]['largest_over_smallest']:.2f}; of moved*B/lr "
                f"x{c[str(n)]['largest_over_smallest_with_lr_over_batch']:.2f})" for n in (3200, 12800, 38400)))
        for g in ("pos", "tok"):
            t = displacement["one_minus_cosine_fit"][g]
            print(f"   1 - cosine to the initial {g}: " + " | ".join(
                f"N {n}: lr^{t[str(n)]['lr_exponent']:+.2f} B^{t[str(n)]['batch_exponent']:+.2f}"
                for n in (3200, 12800, 38400)))
    print("== (c) the baseline with the learning rate halved for one group of weights (reference: 0 of 6 textbook "
          "circuits at 1e-3, 6 of 6 at 5e-4)")
    for arm, o in intervention["arms"].items():
        print(f"   {arm:>10s} {o['learning_rate_factors']}: textbook {o['textbook']}/{o['of']} (2 heads "
              f"{o['by_heads']['h2']}, 4 heads {o['by_heads']['h4']}) | relay positions {o['relay_deltas']} | decoy "
              f"{[round(x, 2) for x in o['deep_decoy_over_clean']]} | loss<2 at "
              f"{o['first_step_with_loss_rep_below_2']}")
    print("missing:", {k: len(v) for k, v in result["missing"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
