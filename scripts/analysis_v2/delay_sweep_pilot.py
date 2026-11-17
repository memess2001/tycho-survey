"""Pilot of the delay sweep (exploratory; not pre-registered by the project; local runs on pilot seeds).

``docs/results/v2_hierarchy_projections.md`` §6 wrote down, on 16 November 2026 and before any of these runs, how much
a lower-order term should delay a higher-order term of the Markov language (iid contexts): "rule A", the inverse of the
output gain at the order-1 projection, and "rule B", the ratio of the term's Walsh coefficients. This script runs those
configurations with the repository's own runner and measures the delays.

Arms
* ``adam``: sky v2a's Markov settings (MLP width 128, Adam 1e-3, batch 64, 30,000 steps, iid contexts). The nine
  configurations of §6, the two twins of DR2a again on the pilot seeds, and the two "alone" sources.
* ``sgd``: the same with plain SGD (learning rate 0.03, no momentum). Asked in review: is the part of the delay that
  the gain does not explain an effect of Adam's second-moment estimate? If so it should vanish here. The rules' values
  are the same as for Adam (they come from the source), but they were written down with sky v2a's Adam settings in
  mind, so the SGD rows do not count as predictions on record.
* Second round (17 November; the reviewer's predictions are in ``REVIEWER`` and were on record before the runs).
  Under SGD, two lower-order terms on the lags the order-3 term uses ({1} and {2}) against the same on lags it does
  not use ({4} and {5}): the proposed mechanism (a ReLU unit whose gate is set by an input it shares with the
  higher-order term can build that term at first order) says the second delays more. And Adam with a short (beta2
  0.9) or long (0.9999) second-moment memory and with a batch of 4,096, on the twin {1}:2.0 + {1,3}:0.8: the
  proposed mechanism (the second moment relaxes to the plateau's gradient noise, so Adam's step grows there) says a
  short memory and a large batch shorten the delay and a long memory brings it back towards SGD.

Seeds 600 to 607 with data_seed = seed, outside every registered range. Pattern mode: the network's log-odds on all 64
windows at 60 log-spaced checkpoints, so every value is exact.

Measures, per twin and paired by seed (definitions as in ``hierarchy_projections.py``)
* delay of the order content of the higher-order term at 25, 50 and 75 % of its final value (the mean of the run's
  last five checkpoints), and the same against the exact target, the term's step length r_k in the source: with a
  strong lower-order term the content at stake is small and the run's own final value is noisy;
* the same for the term's uniform Walsh coefficient, in raw steps and in gain-weighted time;
* rule A and rule B, computed from the source alone;
* a row counts as reliable when the term's content ends within 10 % of its exact target in every seed and the seeds'
  delays at 50 % agree within a factor of two (with the strongest lower-order terms the content at stake is a few
  thousandths of a nat, and under SGD one configuration has not converged in 30,000 steps);
* two paired comparisons: the lower-order term on lag 4 against the same on lag 1 (does sharing an input coordinate
  with the higher-order term matter?), and each twin's delay under SGD against its delay under Adam.

Usage, from the repository root (needs torch):
``PYTHONPATH=. python scripts/analysis_v2/delay_sweep_pilot.py --jobs 4``. Runs are cached in ``work/delay_sweep``.
Writes ``docs/results/v2_analysis/delay_sweep_pilot.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("hierarchy_projections",
                                               Path(__file__).with_name("hierarchy_projections.py"))
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)

SEEDS = tuple(range(600, 608))
FRACS = (0.25, 0.5, 0.75)
ALONE = {"1,3:0.8": 2, "1,2,3:1.0": 3}                      # source -> order of the term
#: (alone, with, was the prediction on record before these runs?)
TWINS = {
    "adam": (("1,3:0.8", "1:0.25;1,3:0.8", True), ("1,3:0.8", "1:0.5;1,3:0.8", True),
             ("1,3:0.8", "1:1.0;1,3:0.8", False), ("1,3:0.8", "1:1.5;1,3:0.8", True),
             ("1,3:0.8", "1:2.0;1,3:0.8", True), ("1,3:0.8", "1:3.0;1,3:0.8", True),
             ("1,3:0.8", "4:1.0;1,3:0.8", True), ("1,3:0.8", "4:2.0;1,3:0.8", True),
             ("1,2,3:1.0", "1:1.0;1,2,3:1.0", False), ("1,2,3:1.0", "1:2.0;1,2,3:1.0", True),
             ("1,2,3:1.0", "1:3.0;1,2,3:1.0", True)),
    "sgd": (("1,3:0.8", "1:0.5;1,3:0.8", False), ("1,3:0.8", "1:1.0;1,3:0.8", False),
            ("1,3:0.8", "1:1.5;1,3:0.8", False), ("1,3:0.8", "1:2.0;1,3:0.8", False),
            ("1,3:0.8", "1:3.0;1,3:0.8", False), ("1,3:0.8", "4:1.0;1,3:0.8", False),
            ("1,3:0.8", "4:2.0;1,3:0.8", False), ("1,2,3:1.0", "1:1.0;1,2,3:1.0", False),
            ("1,2,3:1.0", "1:2.0;1,2,3:1.0", False), ("1,2,3:1.0", "1:3.0;1,2,3:1.0", False),
            # second round: two lower-order terms on lags the order-3 term uses, or on lags it does not use
            ("1,2,3:1.0", "1:1.0;2:1.0;1,2,3:1.0", False), ("1,2,3:1.0", "4:1.0;5:1.0;1,2,3:1.0", False)),
    # second round: Adam's second-moment memory and the batch size, on one twin
    "adam_b2_0.9": (("1,3:0.8", "1:2.0;1,3:0.8", False),),
    "adam_b2_0.9999": (("1,3:0.8", "1:2.0;1,3:0.8", False),),
    "adam_b2_0.9_batch4096": (("1,3:0.8", "1:2.0;1,3:0.8", False),),
}
OPTIM = {"adam": {"optimizer": "adam", "lr": 1e-3}, "sgd": {"optimizer": "sgd", "lr": 0.03, "momentum": 0.0},
         "adam_b2_0.9": {"optimizer": "adam", "lr": 1e-3, "beta2": 0.9},
         "adam_b2_0.9999": {"optimizer": "adam", "lr": 1e-3, "beta2": 0.9999},
         "adam_b2_0.9_batch4096": {"optimizer": "adam", "lr": 1e-3, "beta2": 0.9, "batch_size": 4096}}
#: The reviewer's predictions for the second round (reply of 17 November to commit 41d4d7a), on record before the runs.
REVIEWER = {
    "double_gate": "SGD: the delay of {1,2,3} at 50 % with {4}:1.0 + {5}:1.0 is at least 1.20 times the delay with "
                   "{1}:1.0 + {2}:1.0, raw and gain-weighted; fails if it is not longer",
    "beta2": "Adam, 1:2.0;1,3:0.8, delay at 50 % (3.19 at beta2 0.999): at most 2.60 at beta2 0.9 and at least 4.30 "
             "at beta2 0.9999; fails if it does not fall monotonically from 0.9999 to 0.9 or changes by less than 0.3",
    "batch": "Adam, beta2 0.9, batch 4096 instead of 64: delay at 50 % at most 2.05; fails if it does not fall below "
             "2.30",
}


def slug(couplings: str) -> str:
    return couplings.replace(";", "_").replace(":", "-").replace(",", "")


def run_one(job: tuple[str, str, int, str]) -> str:
    arm, couplings, seed, work = job
    out = Path(work) / arm / f"{slug(couplings)}_s{seed}.npz"
    if out.exists():
        return f"cached {out.name}"
    import torch

    from tycho.train.runner import run_object
    from tycho.train.spec import ObjectSpec, RIConfig, TrainConfig

    torch.set_num_threads(1)
    cfg = {"couplings": couplings, "data_seed": seed, "depth": 1, "exact_metrics": True, "n_tokens": 200_000,
           "shift_rho": -0.5, "train_context": "iid", "width": 128, "window": 6}
    opt = dict(OPTIM[arm])
    train = TrainConfig(steps=30_000, batch_size=int(opt.pop("batch_size", 64)), weight_decay=0.0, n_log_evals=80,
                        n_lin_evals=0, ntk_every=0, **opt)
    spec = ObjectSpec(task="markov_lang", task_cfg=cfg, seed=seed, train=train, sky_version="pilot-delay-sweep")
    res = run_object(spec, work_dir=Path(work) / "resume" / f"{arm}_{slug(couplings)}_s{seed}",
                     ri=RIConfig(enabled=True, mode="patterns", n_probe=8192, n_points=60), resume=False)
    if res.status != "done":
        return f"FAILED {arm} {couplings} {seed}: {res.error}"
    out.parent.mkdir(parents=True, exist_ok=True)
    pat = res.patterns
    np.savez_compressed(out, steps=np.asarray(pat["steps"]), logodds=np.asarray(pat["logodds"]),
                        pmf_D=np.asarray(pat["pmf_D"]), truth=np.asarray(pat["truth"]))
    return f"done {arm} {couplings} seed {seed}"


def higher_term(couplings: str) -> str:
    return max(couplings.split(";"), key=lambda part: len(part.split(":")[0].split(",")))


def load_run(work: Path, arm: str, couplings: str, seed: int, order: int, lags: list[int], phi, deg) -> dict:
    with np.load(work / arm / f"{slug(couplings)}_s{seed}.npz") as f:
        steps, z, pmf, truth = f["steps"].astype(float), f["logodds"], f["pmf_D"], f["truth"]
    L = hp.data_rungs(pmf, truth, phi, deg, 6)["L"]
    target = float(L[order - 1] - L[order])                 # the step length r_k of the source: known in advance
    pt = hp.sigmoid(z)
    m = np.tanh(0.5 * z)
    content = np.zeros(len(steps))
    for t in range(len(steps)):
        lo, _, _ = hp.wlogit(phi[:, deg <= order - 1], pmf, pt[t])
        hi, _, _ = hp.wlogit(phi[:, deg <= order], pmf, pt[t])
        content[t] = hp.bern_kl(pt[t], lo, pmf) - hp.bern_kl(pt[t], hi, pmf)
    chi = phi[:, sum(1 << (lag - 1) for lag in lags)]
    walsh = (m * chi[None, :]) @ pmf
    gain = (1.0 - m ** 2) @ pmf
    loss = np.array([hp.bern_kl(truth, z[t], pmf) for t in range(len(steps))])
    out = {"final_loss": float(loss[-1]), "content_last5_rise": float(content[-1] / max(content[-5], 1e-300) - 1.0),
           "content_final_over_target": float(np.mean(content[-5:]) / target)}
    for f in FRACS:
        tw = hp.half_time(steps, walsh, f)
        out[f] = {"content": hp.half_time(steps, content, f), "walsh": tw,
                  "tau": hp.gain_weighted_time(steps, gain, tw),
                  "content_target": first_crossing(steps, content, f * target)}
    return out


def first_crossing(steps: np.ndarray, y: np.ndarray, level: float) -> float:
    for i in range(1, len(y)):
        if y[i] >= level and y[i - 1] < level:
            a, b = math.log(max(float(steps[i - 1]), 0.5)), math.log(float(steps[i]))
            return math.exp(a + (level - y[i - 1]) / (y[i] - y[i - 1]) * (b - a))
    return float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "delay_sweep")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "delay_sweep_pilot.json")
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    args = ap.parse_args(argv)

    jobs = []
    for arm, twins in TWINS.items():
        for c in sorted({a for a, _, _ in twins} | {w for _, w, _ in twins}):
            jobs += [(arm, c, s, str(args.work)) for s in SEEDS]
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            for i, msg in enumerate(pool.imap_unordered(run_one, jobs), 1):
                if msg.startswith("FAILED") or i % 20 == 0:
                    print(f"[{i}/{len(jobs)}] {msg}", flush=True)

    phi, deg = hp.features(6)
    uni = np.full(64, 1.0 / 64)
    result = {"status": "exploratory pilot; local runs; not pre-registered by the project; not independently "
                        "recomputed",
              "seeds": list(SEEDS), "optimisers": OPTIM, "twins": []}
    cache: dict = {}

    def get(arm: str, c: str, order: int, lags: list[int]) -> dict:
        key = (arm, c, order, tuple(lags))
        if key not in cache:
            cache[key] = {s: load_run(args.work, arm, c, s, order, lags, phi, deg) for s in SEEDS}
        return cache[key]

    for arm, twins in TWINS.items():
        for alone, with_c, on_record in twins:
            order = ALONE[alone]
            lags = [int(x) for x in higher_term(with_c).split(":")[0].split(",")]
            a, w = get(arm, alone, order, lags), get(arm, with_c, order, lags)
            rg = hp.data_rungs(uni, hp.truth_table(with_c, 6), phi, deg, 6)
            chi = phi[:, sum(1 << (lag - 1) for lag in lags)]
            f_with = float(np.mean((2 * hp.truth_table(with_c, 6) - 1) * chi))
            f_alone = float(np.mean((2 * hp.truth_table(alone, 6) - 1) * chi))
            row = {"arm": arm, "alone": alone, "with": with_c, "order": order,
                   "prediction_on_record_before_the_runs": on_record,
                   "rule_A": float(1.0 / rg["gain"][1]), "rule_B": f_alone / f_with,
                   "plateau_L1": float(rg["L"][1]), "L0": float(rg["L"][0]),
                   "final_loss_max": {"alone": max(a[s]["final_loss"] for s in SEEDS),
                                      "with": max(w[s]["final_loss"] for s in SEEDS)},
                   "with_content_last5_rise_max": max(w[s]["content_last5_rise"] for s in SEEDS),
                   "with_content_final_over_target": [min(w[s]["content_final_over_target"] for s in SEEDS),
                                                      max(w[s]["content_final_over_target"] for s in SEEDS)],
                   "delay": {}}
            for f in FRACS:
                def stats(xs):
                    xs = np.asarray(xs, dtype=float)
                    return {"median": float(np.nanmedian(xs)), "min": float(np.nanmin(xs)),
                            "max": float(np.nanmax(xs)), "n": int(np.isfinite(xs).sum())}
                row["delay"][str(f)] = {
                    "order_content": stats([w[s][f]["content"] / a[s][f]["content"] for s in SEEDS]),
                    "order_content_vs_target": stats([w[s][f]["content_target"] / a[s][f]["content_target"]
                                                      for s in SEEDS]),
                    "walsh_raw": stats([w[s][f]["walsh"] / a[s][f]["walsh"] for s in SEEDS]),
                    "walsh_gain_weighted": stats([w[s][f]["tau"] / a[s][f]["tau"] for s in SEEDS]),
                    "t_alone_median": float(np.nanmedian([a[s][f]["content"] for s in SEEDS])),
                    "t_with_median": float(np.nanmedian([w[s][f]["content"] for s in SEEDS]))}
            lo, hi = row["with_content_final_over_target"]
            spread = row["delay"]["0.5"]["order_content"]
            # converged (the term's content ends within 10 % of its exact target in every seed) and the seeds agree
            row["reliable"] = bool(lo >= 0.9 and hi <= 1.1 and spread["max"] <= 2.0 * spread["min"])
            result["twins"].append(row)

    def paired(xs) -> dict:
        xs = np.asarray(xs, dtype=float)
        return {"median": float(np.median(xs)), "min": float(xs.min()), "max": float(xs.max()),
                "n_above_1": int((xs > 1.0).sum()), "n": len(xs)}

    # does it matter whether the lower-order term sits on a lag that the higher-order term uses? (paired by seed)
    result["lag_4_over_lag_1"] = []
    for arm in ("adam", "sgd"):
        for strength in ("1.0", "2.0"):
            w1 = get(arm, f"1:{strength};1,3:0.8", 2, [1, 3])
            w4 = get(arm, f"4:{strength};1,3:0.8", 2, [1, 3])
            result["lag_4_over_lag_1"].append({
                "arm": arm, "strength": float(strength),
                "half_time_ratio": {str(f): paired([w4[s][f]["content"] / w1[s][f]["content"] for s in SEEDS])
                                    for f in FRACS}})
    # the same twin under SGD and under Adam: ratio of the two delays (same seeds and data, different optimiser)
    result["delay_sgd_over_adam"] = []
    sgd_twins = {w for _, w, _ in TWINS["sgd"]}
    for alone, with_c, _ in TWINS["adam"]:
        if with_c not in sgd_twins:
            continue
        order = ALONE[alone]
        lags = [int(x) for x in higher_term(with_c).split(":")[0].split(",")]
        d = {arm: {s: get(arm, with_c, order, lags)[s][0.5]["content"] / get(arm, alone, order, lags)[s][0.5]["content"]
                   for s in SEEDS} for arm in ("adam", "sgd")}
        result["delay_sgd_over_adam"].append({"with": with_c, "at_50_percent":
                                              paired([d["sgd"][s] / d["adam"][s] for s in SEEDS])})
    # second round. Two lower-order terms on lags 4 and 5 against the same on lags 1 and 2 (the order-3 term's own)
    result["reviewer_predictions"] = REVIEWER
    shared = get("sgd", "1:1.0;2:1.0;1,2,3:1.0", 3, [1, 2, 3])
    apart = get("sgd", "4:1.0;5:1.0;1,2,3:1.0", 3, [1, 2, 3])
    result["double_gate_lags_4_5_over_lags_1_2"] = {
        "arm": "sgd",
        "content_half_time_ratio": {str(f): paired([apart[s][f]["content"] / shared[s][f]["content"] for s in SEEDS])
                                    for f in FRACS},
        "gain_weighted_ratio": {str(f): paired([apart[s][f]["tau"] / shared[s][f]["tau"] for s in SEEDS])
                                for f in FRACS}}
    # Adam's second-moment memory and the batch size: delay of 1:2.0;1,3:0.8 at 50 %, per setting
    by_twin = {(r["arm"], r["with"]): r for r in result["twins"]}
    result["adam_beta2_and_batch"] = [
        {"arm": arm, "beta2": OPTIM[arm].get("beta2", 0.999), "batch_size": OPTIM[arm].get("batch_size", 64),
         "delay_at_50_percent": by_twin[arm, "1:2.0;1,3:0.8"]["delay"]["0.5"]["order_content"],
         "t_alone_median": by_twin[arm, "1:2.0;1,3:0.8"]["delay"]["0.5"]["t_alone_median"],
         "t_with_median": by_twin[arm, "1:2.0;1,3:0.8"]["delay"]["0.5"]["t_with_median"],
         "reliable": by_twin[arm, "1:2.0;1,3:0.8"]["reliable"]}
        for arm in ("adam_b2_0.9_batch4096", "adam_b2_0.9", "adam", "adam_b2_0.9999", "sgd")]

    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    args.out.write_text(json.dumps(clean(result), indent=1) + "\n", encoding="utf-8")
    print(f"{'arm':>21s} {'with':>21s} {'ruleA':>6s} {'ruleB':>6s} | order content 25/50/75 % | "
          f"vs target 25/50/75 %  | Walsh raw 25 % | gain-weighted 25/50/75 % | reliable")
    for r in result["twins"]:
        d = r["delay"]
        print(f"{r['arm']:>21s} {r['with']:>21s} {r['rule_A']:6.2f} {r['rule_B']:6.2f} | "
              + " ".join(f"{d[str(f)]['order_content']['median']:6.2f}" for f in FRACS) + "     | "
              + " ".join(f"{d[str(f)]['order_content_vs_target']['median']:6.2f}" for f in FRACS)
              + f" | {d['0.25']['walsh_raw']['median']:6.2f}         | "
              + " ".join(f"{d[str(f)]['walsh_gain_weighted']['median']:5.2f}" for f in FRACS)
              + f"        | {'yes' if r['reliable'] else 'no'}")
    print("half-time with the lower-order term on lag 4 over the same on lag 1 (paired by seed; 25/50/75 %)")
    for r in result["lag_4_over_lag_1"]:
        h = r["half_time_ratio"]
        print(f"  {r['arm']:>4s} strength {r['strength']}: " + "  ".join(
            f"{h[str(f)]['median']:.2f} [{h[str(f)]['min']:.2f}, {h[str(f)]['max']:.2f}] "
            f"({h[str(f)]['n_above_1']}/8 > 1)"
            for f in FRACS))
    print("delay under SGD over delay under Adam (paired by seed; 50 %)")
    for r in result["delay_sgd_over_adam"]:
        h = r["at_50_percent"]
        print(f"  {r['with']:>16s}: {h['median']:.2f} [{h['min']:.2f}, {h['max']:.2f}] ({h['n_above_1']}/8 > 1)")
    print("SGD, two lower-order terms: lags 4 and 5 over lags 1 and 2 (paired by seed; 25/50/75 %)")
    for key in ("content_half_time_ratio", "gain_weighted_ratio"):
        h = result["double_gate_lags_4_5_over_lags_1_2"][key]
        print(f"  {key:>24s}: " + "  ".join(
            f"{h[str(f)]['median']:.2f} [{h[str(f)]['min']:.2f}, {h[str(f)]['max']:.2f}] "
            f"({h[str(f)]['n_above_1']}/8 > 1)" for f in FRACS))
    print("delay of 1:2.0;1,3:0.8 at 50 % against Adam's beta2 and the batch size")
    for r in result["adam_beta2_and_batch"]:
        h = r["delay_at_50_percent"]
        print(f"  {r['arm']:>21s} beta2 {r['beta2']:<6} batch {r['batch_size']:>4d}: {h['median']:.2f} "
              f"[{h['min']:.2f}, {h['max']:.2f}]  t_alone {r['t_alone_median']:.0f} t_with {r['t_with_median']:.0f}"
              f"  reliable {r['reliable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
