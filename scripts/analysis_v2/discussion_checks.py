"""Quick checks made during the discussion of 16 November 2026 (exploratory; not pre-registered and not independently
recomputed). The discussion is recorded in ``notes/discussion_2026-11-16_outside_reviews.md``; every number quoted there
that is not in ``hierarchy_projections.json`` or ``lightcurve_lead_lag.json`` comes from this script.

A. Boundary refinement in sky v1's grokking region (DR1). Where did the refinement objects go, how many of them
   reach 90 % test accuracy within the 50,000-step budget, and how late do the seeds that reach it arrive in the
   configurations where seeds disagree? Then the uncensored quantity: the step T at which test accuracy first reaches
   90 %, against the train fraction (a straight-line fit of 1/T) and against the learning rate.
B. The tiled induction task of Part B (vocabulary 64, 64 tokens, segment length 4 to 16): the loss on predictable
   tokens of simple predictors that use no content matching, computed from the data distribution alone: uniform;
   uniform over the distinct tokens seen so far; in-context counting; the best fixed mixture over "copy the token d
   positions back"; the same with weights that depend on the position. The counting and fixed-copier values are also
   given for pilot 1's single-repeat layout.
C. Pilot 2 (DR2b-pilot2, tiled layout): the repeat loss of the 1-layer and 2-layer runs over training, against the
   losses of B. Does the 2-layer network dwell at the loss of the content-free solution before the circuit forms?
D. Pilot 1 (DR2b-pilot, single-repeat layout): were the runs without a full circuit still improving when training
   stopped at 12,000 steps?
E. Two heads under a uniform ablation design: the θ value (information added by the pair term to a logistic model
   with the two single terms) and the bounded η value (the same for least squares on the mean) for AND-like and
   OR-like tables.

Usage, from the repository root: ``python scripts/analysis_v2/discussion_checks.py``. Numpy and the standard library
only (it loads ``hierarchy_projections.py`` from its own directory for the logistic fit). Writes
``docs/results/v2_analysis/discussion_checks_2026-11-16.json``. Well under a minute.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("hierarchy_projections",
                                               Path(__file__).with_name("hierarchy_projections.py"))
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)

BUDGET = 50_000


def read_objects(release: Path) -> list[dict]:
    with open(release / "objects.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def lightcurves(release: Path, work: Path) -> Path:
    if not (work / "lightcurves").is_dir():
        work.mkdir(parents=True, exist_ok=True)
        with tarfile.open(release / "lightcurves.tar.gz") as tar:
            tar.extractall(work, filter="data")
    return work / "lightcurves"


def quartiles(xs) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(list(xs), dtype=float), [0, 0.25, 0.5, 0.75, 1])]


# ---------------------------------------------------------------------------------------------------- A
def refinement(dr1: Path) -> dict:
    rows = []
    for o in read_objects(dr1):
        if o["region"] != "grokking-v1":
            continue
        cfg = json.loads(o["task_cfg_json"])
        t = float(o["t_test_acc90"]) if o["t_test_acc90"] not in ("", "nan") else None
        rows.append({"key": (int(cfg["width"]), float(o["train_lr"]), float(o["train_weight_decay"]),
                             float(cfg["train_frac"])), "source": o["source"], "T": t,
                     "acc": float(o["final_test_acc"]), "label": o["label"]})
    cfgs = defaultdict(list)
    for r in rows:
        cfgs[r["key"]].append(r)
    n_reached = {k: sum(r["T"] is not None for r in v) for k, v in cfgs.items()}
    refine = [r for r in rows if r["source"] == "refine"]
    coarse = [r for r in rows if r["source"] == "coarse"]
    mixed = [k for k, v in cfgs.items() if 0 < n_reached[k] < len(v)]
    late = [r["T"] / BUDGET for k in mixed for r in cfgs[k] if r["T"] is not None]
    short = [r["acc"] for k in mixed for r in cfgs[k] if r["T"] is None]
    out = {
        "objects": len(rows), "configurations": len(cfgs),
        "refine_objects": len(refine), "refine_reached": sum(r["T"] is not None for r in refine),
        "coarse_objects": len(coarse), "coarse_reached": sum(r["T"] is not None for r in coarse),
        "refine_objects_by_configuration": {
            "all_seeds_reach": sum(n_reached[r["key"]] == len(cfgs[r["key"]]) for r in refine),
            "some_seeds_reach": sum(0 < n_reached[r["key"]] < len(cfgs[r["key"]]) for r in refine),
            "no_seed_reaches": sum(n_reached[r["key"]] == 0 for r in refine)},
        "configurations_with_label_disagreement": sum(len({r["label"] for r in v}) > 1 for v in cfgs.values()),
        "mixed_configurations": len(mixed),
        "mixed_T_over_budget_quartiles": quartiles(late), "mixed_reaching_seeds": len(late),
        "mixed_final_accuracy_of_seeds_that_do_not_reach_quartiles": quartiles(short),
    }
    # the uncensored quantity: median T where every seed reaches 90 %
    med_T = {k: float(np.median([r["T"] for r in v])) for k, v in cfgs.items() if n_reached[k] == len(v)}
    fits = []
    for w, lr, wd in sorted({k[:3] for k in med_T}):
        pts = sorted((k[3], T) for k, T in med_T.items() if k[:3] == (w, lr, wd))
        if len(pts) < 3:
            continue
        x, y = np.array([p[0] for p in pts]), 1.0 / np.array([p[1] for p in pts])
        a, b = np.polyfit(x, y, 1)
        frac_c = -b / a
        r2 = 1.0 - float(np.sum((y - (a * x + b)) ** 2) / np.sum((y - y.mean()) ** 2))
        cens = sorted(k[3] for k in cfgs if k[:3] == (w, lr, wd) and n_reached[k] == 0)
        nxt = cens[-1] if cens else None
        fits.append({"width": w, "lr": lr, "weight_decay": wd, "points": [{"train_frac": p[0], "T": p[1]} for p in pts],
                     "slope": float(a), "train_frac_at_zero_rate": float(frac_c), "r2": r2,
                     "largest_train_frac_that_does_not_reach": nxt,
                     "its_median_final_accuracy": None if nxt is None else float(
                         np.median([r["acc"] for r in cfgs[(w, lr, wd, nxt)]])),
                     "predicted_T_there": None if nxt is None or nxt <= frac_c else float(1.0 / (a * (nxt - frac_c)))})
    ratios = []
    for (w, lr, wd, tf), T in sorted(med_T.items()):
        if lr == 0.003 and (w, 0.001, wd, tf) in med_T:
            ratios.append({"width": w, "weight_decay": wd, "train_frac": tf, "T_lr_0.001": med_T[(w, 0.001, wd, tf)],
                           "T_lr_0.003": T, "ratio": med_T[(w, 0.001, wd, tf)] / T})
    out["rate_fits"] = fits
    out["lr_ratios"] = ratios
    return out


# ---------------------------------------------------------------------------------------------------- B
def induction_rungs(vocab: int = 64, seq_len: int = 64, rep_min: int = 4, rep_max: int = 16, m: int = 12_000,
                    seed: int = 0) -> dict:
    """Losses of content-free predictors on the predictable tokens of the tiled layout (as
    ``tycho.tasks.induction.tiled_sequences``, re-implemented with numpy)."""
    rng = np.random.default_rng(seed)
    L = rng.integers(rep_min, rep_max + 1, m)
    seg = rng.integers(0, vocab, (m, rep_max))
    phase = np.floor(rng.random(m) * L).astype(int)
    x = np.take_along_axis(seg, (np.arange(seq_len)[None, :] + phase[:, None]) % L[:, None], axis=1)
    t = np.arange(seq_len - 1)[None, :]
    rep = t >= L[:, None]                                  # input position t, target x[t+1], predictable
    ms, ts = np.nonzero(rep)
    y = x[ms, ts + 1]
    # in-context statistics of x[0..t]
    onehot = np.zeros((m, seq_len, vocab), dtype=np.int32)
    onehot[np.arange(m)[:, None], np.arange(seq_len)[None, :], x] = 1
    counts = np.cumsum(onehot, axis=1)                     # counts[i, t, v] = occurrences of v in x[0..t]
    c_y = counts[ms, ts, y]
    distinct = (counts[ms, ts] > 0).sum(axis=1)
    # copy by position: does the token d back from the target equal the target? d = 1 .. D
    D = seq_len - 1
    match = np.zeros((len(ms), D), dtype=bool)
    for d in range(1, D + 1):
        ok = ts + 1 - d >= 0
        match[ok, d - 1] = x[ms[ok], ts[ok] + 1 - d] == y[ok]

    def em(mt: np.ndarray, iters: int = 200) -> np.ndarray:
        w = np.full(mt.shape[1], 1.0 / mt.shape[1])
        for _ in range(iters):
            num = mt * w[None, :]
            den = num.sum(axis=1, keepdims=True)
            den[den == 0] = 1e-300
            w = (num / den).mean(axis=0)
            w /= w.sum()
        return w

    w_fixed = em(match[:, :40])
    p_fixed = (match[:, :40] * w_fixed[None, :]).sum(axis=1)
    nll_pos = np.zeros(len(ms))
    for tt in np.unique(ts):
        sel = ts == tt
        mt = match[sel][:, : tt + 1]
        p = (mt * em(mt, 100)[None, :]).sum(axis=1)
        nll_pos[sel] = -np.log(0.98 * p + 0.02 / vocab)
    return {
        "sequences": m, "predictable_targets_per_sequence": float(rep.sum(axis=1).mean()),
        "uniform": math.log(vocab),
        "uniform_over_distinct_tokens_seen": float(np.mean(np.log(distinct))),
        "in_context_counting": float(np.mean(-np.log(c_y / (ts + 1)))),
        "fixed_offset_copier": float(np.mean(-np.log(np.maximum(p_fixed, 1e-12)))),
        "position_dependent_offset_copier": float(np.mean(nll_pos)),
        "fixed_offset_weights_4_to_16": [float(v) for v in w_fixed[3:16]],
    }


def single_repeat_rungs(rep_min: int, rep_max: int = 32, vocab: int = 64, seq_len: int = 64, m: int = 40_000,
                        seed: int = 1) -> dict:
    """The same for pilot 1's single-repeat layout (as ``tycho.tasks.induction.repeat_sequences``): counting, and the
    best fixed mixture over "copy the token d positions back"."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, vocab, (m, seq_len))
    L = rng.integers(rep_min, rep_max + 1, m)
    s = np.floor(rng.random(m) * (seq_len - 2 * L + 1)).astype(int)
    idx = np.arange(seq_len)[None, :]
    in_copy = (idx >= (s + L)[:, None]) & (idx < (s + 2 * L)[:, None])
    x = np.where(in_copy, np.take_along_axis(x, np.clip(idx - L[:, None], 0, None), axis=1), x)
    t = np.arange(seq_len - 1)[None, :]
    rep = (t >= (s + L)[:, None]) & (t <= (s + 2 * L - 2)[:, None])
    ms, ts = np.nonzero(rep)
    y = x[ms, ts + 1]
    D = seq_len - 1
    match = np.zeros((len(ms), D), dtype=bool)
    for d in range(1, D + 1):
        ok = ts + 1 - d >= 0
        match[ok, d - 1] = x[ms[ok], ts[ok] + 1 - d] == y[ok]
    w = np.full(D, 1.0 / D)
    for _ in range(200):
        num = match * w[None, :]
        den = num.sum(axis=1, keepdims=True)
        den[den == 0] = 1e-300
        w = (num / den).mean(axis=0)
        w /= w.sum()
    p = (match * w[None, :]).sum(axis=1)
    count = np.array([np.mean(x[i, : tt + 1] == yy) for i, tt, yy in zip(ms[:20000], ts[:20000], y[:20000])])
    return {"rep_min": rep_min, "rep_max": rep_max, "uniform": math.log(vocab),
            "in_context_counting": float(np.mean(-np.log(count))),
            "fixed_offset_copier": float(np.mean(-np.log(0.98 * p + 0.02 / vocab)))}


# ------------------------------------------------------------------------------------------------- C and D
def at(step: np.ndarray, y: np.ndarray, s: int) -> float:
    return float(y[int(np.argmin(np.abs(step - s)))])


def pilot2(release: Path, work: Path) -> dict:
    lc = lightcurves(release, work)
    show = (0, 50, 100, 150, 200, 250, 300, 350, 400, 500, 700, 1000, 2000, 4000, 12000)
    runs = []
    for o in read_objects(release):
        cfg = json.loads(o["task_cfg_json"])
        if float(cfg["p_repeat"]) != 1.0:
            continue
        with np.load(lc / f"{o['object_id']}.npz", allow_pickle=False) as f:
            step, y = f["step"].astype(float), f["x_loss_rep"].astype(float)
        band = np.where((y >= 1.6) & (y <= 2.3))[0]
        runs.append({"n_layer": int(cfg["n_layer"]), "lr": float(o["train_lr"]), "seed": int(o["seed"]),
                     "loss_rep": {str(s): at(step, y, s) for s in show}, "final": float(y[-1]),
                     "evaluations_with_loss_in_1.6_to_2.3": int(len(band)),
                     "first_step_below_1.0": float(step[np.argmax(y <= 1.0)]) if np.any(y <= 1.0) else None})
    runs.sort(key=lambda r: (r["lr"], r["n_layer"], r["seed"]))
    out = {"runs": runs}
    for lr in sorted({r["lr"] for r in runs}):
        two = [r for r in runs if r["lr"] == lr and r["n_layer"] == 2]
        one = [r for r in runs if r["lr"] == lr and r["n_layer"] == 1]
        out[f"lr_{lr}"] = {
            "two_layer_loss_at": {s: [round(r["loss_rep"][s], 3) for r in two] for s in ("250", "300", "400", "500")},
            "one_layer_loss_at": {s: [round(r["loss_rep"][s], 3) for r in one]
                                  for s in ("250", "300", "400", "500", "1000", "4000", "12000")},
            "two_layer_evaluations_in_band": [r["evaluations_with_loss_in_1.6_to_2.3"] for r in two],
            "one_layer_evaluations_in_band": [r["evaluations_with_loss_in_1.6_to_2.3"] for r in one],
        }
    return out


def pilot1(release: Path, work: Path, labels_json: Path) -> dict:
    lc = lightcurves(release, work)
    circuit = {r["object_id"]: r["circuit"] for r in json.loads(labels_json.read_text())["runs"]}
    runs = []
    for o in read_objects(release):
        cfg = json.loads(o["task_cfg_json"])
        if int(cfg["n_layer"]) != 2 or float(cfg["p_repeat"]) != 1.0:
            continue
        with np.load(lc / f"{o['object_id']}.npz", allow_pickle=False) as f:
            step, y = f["step"].astype(float), f["x_loss_rep"].astype(float)
        runs.append({"rep_min": int(cfg["rep_min"]), "n_head": int(cfg["n_head"]), "seed": int(o["seed"]),
                     "circuit": circuit.get(o["object_id"]),
                     "loss_rep": {str(s): at(step, y, s) for s in (2000, 4000, 6000, 8000, 10000, 12000)}})
    runs.sort(key=lambda r: (r["rep_min"], r["n_head"], r["seed"]))
    rest = [r for r in runs if r["circuit"] != "full"]
    change = [r["loss_rep"]["12000"] - r["loss_rep"]["8000"] for r in rest]
    return {"runs": runs, "without_full_circuit": len(rest),
            "final_loss_range_without_full_circuit": [min(r["loss_rep"]["12000"] for r in rest),
                                                      max(r["loss_rep"]["12000"] for r in rest)],
            "still_falling_by_more_than_0.02_between_8k_and_12k": int(sum(c < -0.02 for c in change)),
            "still_falling_by_more_than_0.05_between_8k_and_12k": int(sum(c < -0.05 for c in change))}


# ---------------------------------------------------------------------------------------------------- E
def two_heads() -> list[dict]:
    phi, deg = hp.features(2)                 # cells in the order (0,0), (1,0), (0,1), (1,1): 1 = head kept
    uni = np.full(4, 0.25)
    p0 = 1.0 / 64

    def h2(q: float) -> float:
        return 0.0 if q <= 0 or q >= 1 else -q * math.log(q) - (1 - q) * math.log(1 - q)

    out = []
    for name, p in (("AND: both heads needed, floor at chance", [p0, p0, p0, 0.96]),
                    ("OR: either head suffices (0.99; both 0.9999)", [p0, 0.99, 0.99, 0.9999]),
                    ("OR: either head suffices (0.96; both 0.96)", [p0, 0.96, 0.96, 0.96]),
                    ("each head alone gives 0.2", [p0, 0.2, 0.2, 0.96]),
                    ("each head alone gives 0.5", [p0, 0.5, 0.5, 0.96])):
        p = np.array(p)
        z_main, _, _ = hp.wlogit(phi[:, deg <= 1], uni, p)
        c = float(np.mean((2 * p - 1) * phi[:, 3]))
        out.append({"table": name, "p": [float(v) for v in p], "theta_nats": hp.bern_kl(p, z_main, uni),
                    "eta_bounded_nats": math.log(2) - h2((1 + abs(c)) / 2), "pair_coefficient_of_the_mean": c})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--releases", type=Path, default=REPO / "releases")
    ap.add_argument("--work", type=Path, default=REPO / "work")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" /
                    "discussion_checks_2026-11-16.json")
    args = ap.parse_args(argv)
    result = {
        "status": "exploratory quick checks; not pre-registered; not independently recomputed",
        "A_refinement_dr1_grokking": refinement(args.releases / "DR1"),
        "B_tiled_induction_content_free_losses": induction_rungs(),
        "B_single_repeat_content_free_losses": [single_repeat_rungs(4), single_repeat_rungs(12)],
        "C_pilot2_tiled": pilot2(args.releases / "DR2b-pilot2", args.work / "dr2b_pilot2"),
        "D_pilot1_single_repeat": pilot1(args.releases / "DR2b-pilot", args.work / "dr2b_pilot",
                                         REPO / "docs" / "results" / "v2_analysis" / "v2b_pilot.json"),
        "E_two_heads": two_heads(),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    a, b, d = (result["A_refinement_dr1_grokking"], result["B_tiled_induction_content_free_losses"],
               result["D_pilot1_single_repeat"])
    print(json.dumps({k: v for k, v in a.items() if k not in ("rate_fits", "lr_ratios")}, indent=1))
    for f in a["rate_fits"]:
        print("rate fit:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items() if k != "points"},
              [(p["train_frac"], int(p["T"])) for p in f["points"]])
    print("lr ratios:", [round(r["ratio"], 2) for r in a["lr_ratios"]])
    print("B:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items() if not isinstance(v, list)})
    print("B single repeat:",
          [{k: round(v, 3) for k, v in r.items()} for r in result["B_single_repeat_content_free_losses"]])
    for lr, v in result["C_pilot2_tiled"].items():
        if lr != "runs":
            print("C", lr, json.dumps(v))
    print("D:", {k: v for k, v in d.items() if k != "runs"})
    for r in result["E_two_heads"]:
        print(f"E: {r['table']:<46s} theta {r['theta_nats']:.4f}  eta {r['eta_bounded_nats']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
