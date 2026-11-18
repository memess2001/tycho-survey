"""Pilot (exploratory; local runs): does the critical period move with the learning rate and the width?

``relay_window_pilot.py`` found that at rep_min 12 (learning rate 1e-3, d_model 64) keeping layer 0 off the
positional band for 500 steps flips the final circuit if the restriction starts by step 400, and does not if it
starts at step 500 or later. The reviewer's fifth reply (18 November, to commit 4b44e01) gives an account of this
critical period and predicts where it lies at two other learning rates and two other widths. It states that it ran
none of these settings; its predictions are on record here before the runs.

Its account. A masked attention logit gets no gradient, so what layer 0 has learned about the band before the mask
stays in its weights during the mask ("frozen, not erased"). Within the masked window the textbook circuit has to be
built in three stages that wait for each other: layer 1 returns to the source position, a layer-0 head grows on
offset 1, a second one on offset 2. Started at step 400 the three fit into 500 steps, started at step 500 they do
not. Two quantities of the unperturbed run, read before step 400, are said to locate the critical step:

    t_crit_loss = t2 + (0.295 − (loss(0) − loss(t2))) / v,   v = (loss(t1) − loss(t2)) / (t2 − t1)
    t_crit_attn = t2 + log(o(t2) / 0.0085) / u,              u = (log o(t1) − log o(t2)) / (t2 − t1)

with loss the repeat loss and o layer 0's attention on offsets 1 and 2. The two constants (0.295 nats, 0.0085) were
set on the six baseline runs, where the formulas give medians of 451 and 433.

Predictions (rep_min 12; 2 and 4 heads; seeds 300–302; band mask on layer 0 during [start, end); a "flip" is a
textbook circuit at 12,000 steps: relay at the source and a two-token key):

* 1A, learning rate 5e-4. At step 300: layer 0's band attention (median of the six runs) in [0.52, 0.65], its
  attention on offsets 1–2 in [0.028, 0.043]; t_crit_loss from steps 150 and 350 in [760, 980]. Window [600, 1600):
  at least 5 of 6 flip; [700, 1700): at least 4; [1050, 2050): at most 1.
* 1A, learning rate 2e-3. t_crit_loss and t_crit_attn from steps 100 and 200 in [200, 265]; at step 300 the band
  attention is at least 0.85 and the attention on offsets 1–2 at most 0.010. Window [300, 800): at most 1 of 6 flip;
  [150, 650): at least 5.
  1A fails if [600, 1600) at 5e-4 flips 3 or fewer, if [300, 800) at 2e-3 flips 3 or more, or if a median of
  t_crit_loss is outside its interval.
* 1B, d_model 128. At step 300: band attention at least 0.82, attention on offsets 1–2 at most 0.013, drop of the
  repeat loss at least 0.23 nats; t_crit_attn and t_crit_loss in [260, 350]. Window [400, 900): at most 2 of 6 flip
  (6 of 6 at d_model 64); [240, 740): at least 5.
* 1B, d_model 32. At step 300: band attention in [0.54, 0.67], attention on offsets 1–2 at least 0.026, drop of
  the repeat loss in [0.09, 0.14]; t_crit in [560, 760]. Window [500, 1000): at least 4 of 6 flip (0 of 6 at
  d_model 64).
  1B fails if the band attention at step 300 is not higher at d_model 128 and lower at d_model 32 than at 64, if
  [400, 900) at d_model 128 still flips 5 or more, or if [500, 1000) at d_model 32 flips 1 or fewer.

Recorded here before the runs, for comparison. (a) If training simply runs at a speed proportional to the learning
rate, the critical step is 800–1,000 at 5e-4 and 200–250 at 2e-3, and every window of 1A comes out as the reviewer
predicts; 1A cannot tell its account from that. (b) If the width did not matter, [400, 900) would flip 6 of 6 at
d_model 128 and [500, 1000) 0 of 6 at d_model 32: 1B is the informative half. (c) One window is added at d_model 32,
[900, 1400), to see whether there is a critical period at all at that width (the reviewer's interval for t_crit
implies that it fails).

Also run, to look inside the window: the six baseline runs without a mask and with the windows [400, 900) and
[500, 1000), for 2,500 steps, measured every 50 steps with the layer-0 attention on offsets 1 and 2 by head. These
repeat the first 2,500 steps of runs of ``relay_window_pilot.py`` exactly.

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/relay_period_pilot.py
--jobs 4`` (``--skip-runs`` analyses the cache). Needs the outputs of ``relay_followup_pilot.py``,
``relay_band_pilot.py`` and ``relay_window_pilot.py``. Runs are cached in ``work/relay_band``. Writes
``docs/results/v2_analysis/relay_period_pilot.json``.
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
_spec = importlib.util.spec_from_file_location("relay_window_pilot", Path(__file__).with_name("relay_window_pilot.py"))
wp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp)
rb, fp, rp = wp.rb, wp.fp, wp.rp

FULL_STEPS, SHORT_STEPS = 12_000, 2_500
LOSS_DROP, NEAR_FLOOR = 0.295, 0.0085                   # the reviewer's two constants (set on the baseline runs)
SETTINGS = {"base": {}, "lr5e-4": {"lr": 5e-4}, "lr2e-3": {"lr": 2e-3}, "d128": {"d_model": 128},
            "d32": {"d_model": 32}}
#: arm -> (setting, window [start, end) of the band mask on layer 0 or None, steps); in order of priority
ARMS = {
    "base_natural": ("base", None, SHORT_STEPS), "base_w400_900": ("base", (400, 900), SHORT_STEPS),
    "base_w500_1000": ("base", (500, 1000), SHORT_STEPS),
    "d32_natural": ("d32", None, FULL_STEPS), "d32_w500_1000": ("d32", (500, 1000), FULL_STEPS),
    "d32_w900_1400": ("d32", (900, 1400), FULL_STEPS),
    "d128_natural": ("d128", None, FULL_STEPS), "d128_w400_900": ("d128", (400, 900), FULL_STEPS),
    "d128_w240_740": ("d128", (240, 740), FULL_STEPS),
    "lr2e-3_natural": ("lr2e-3", None, FULL_STEPS), "lr2e-3_w300_800": ("lr2e-3", (300, 800), FULL_STEPS),
    "lr2e-3_w150_650": ("lr2e-3", (150, 650), FULL_STEPS),
    "lr5e-4_natural": ("lr5e-4", None, FULL_STEPS), "lr5e-4_w600_1600": ("lr5e-4", (600, 1600), FULL_STEPS),
    "lr5e-4_w700_1700": ("lr5e-4", (700, 1700), FULL_STEPS),
    "lr5e-4_w1050_2050": ("lr5e-4", (1050, 2050), FULL_STEPS),
}
#: the steps (t1, t2) the reviewer names for its two formulas, by setting: (loss formula, attention formula)
FORMULA_STEPS = {"base": ((100, 300), (200, 300)), "lr5e-4": ((150, 350), (150, 350)),
                 "lr2e-3": ((100, 200), (100, 200)), "d128": ((100, 300), (200, 300)),
                 "d32": ((100, 300), (200, 300))}
STATS = ("band_l0", "band_l1", "source_l1", "offsets_1_2_l0", "loss_rep", "d1_l0_max", "d2_l0_max")


def measure_at(steps: int) -> tuple[int, ...]:
    return tuple(sorted(set(range(0, min(steps, SHORT_STEPS) + 1, 50)) | set(range(0, steps + 1, 250))))


def detail_stats(model, task) -> dict:
    """``band_stats`` of ``relay_band_pilot.py`` and, for layer 0, the largest attention of a head on offset 1 and
    on offset 2 (from the predictable positions of the probe set)."""
    out = rb.band_stats(model, task)
    pr = task._probe
    a0 = model.attention_maps(pr["x"])[0].numpy()        # (B, H, T, T)
    bi, ti = np.nonzero(pr["rep"].numpy())
    out["d1_l0_max"] = float(a0[bi, :, ti, ti - 1].mean(0).max())
    out["d2_l0_max"] = float(a0[bi, :, ti, ti - 2].mean(0).max())
    return out


def arm_jobs(arm: str) -> list[dict]:
    setting, window, steps = ARMS[arm]
    base = {j["name"]: j for j in rp.jobs_list()}
    change = SETTINGS[setting]
    jobs = []
    for nh in rb.HEADS:
        for seed in rb.SEEDS:
            t = base[f"single_rm12_h{nh}_s{seed}"]
            jobs.append({"name": f"{arm}_single_rm12_h{nh}_s{seed}", "base": f"single_rm12_h{nh}_s{seed}",
                         "arm": arm, "setting": setting, "seed": seed, "window": window or (0, 0), "steps": steps,
                         "cfg": {**t["cfg"], **{k: v for k, v in change.items() if k == "d_model"}},
                         "train": {**t["train"], **{k: v for k, v in change.items() if k == "lr"}, "steps": steps},
                         "released_id": None})
    return jobs


def run_arm(args: tuple[dict, str]) -> str:
    job, work = args
    out = Path(work) / f"period_{job['name']}.npz"
    if out.exists():
        return f"cached {out.name}"
    at = measure_at(job["steps"])
    model, task, stats, _ = rb.train(job, job["steps"], at, block_from=job["window"][0],
                                     block_until=job["window"][1], layer=0, forbid="band", stats_fn=detail_stats)
    model.eval()
    arrays = {f"param/{k}": v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update({f"stat_{k}": np.asarray([stats[s][k] for s in at]) for k in STATS})
    np.savez_compressed(out, logodds=task.pattern_logodds(model)[None, :], steps=np.asarray([job["steps"]]),
                        lc_step=np.asarray(at), lc_loss_rep=arrays["stat_loss_rep"],
                        object_id=np.asarray(f"{job['arm']}:{job['base']}"), **arrays)
    return f"done {job['name']}"


def curves(job: dict, work: Path) -> tuple[np.ndarray, dict]:
    with np.load(work / f"period_{job['name']}.npz") as f:
        return f["lc_step"].astype(float), {k: f[f"stat_{k}"] for k in STATS}


def first_crossing(step: np.ndarray, y: np.ndarray, level: float, rising: bool) -> float | None:
    """The first step at which y passes the level (linear interpolation between measurements)."""
    hit = (y >= level) if rising else (y <= level)
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    if i == 0:
        return float(step[0])
    return float(step[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (step[i] - step[i - 1]))


def early(job: dict, work: Path) -> dict:
    """What an unperturbed run shows early on: the reviewer's quantities at step 300, its two extrapolated critical
    steps, and where the same two levels are actually crossed."""
    step, st = curves(job, work)
    at = {int(s): i for i, s in enumerate(step)}
    loss, near = st["loss_rep"], st["offsets_1_2_l0"]
    (a1, a2), (b1, b2) = FORMULA_STEPS[job["setting"]]
    v = (loss[at[a1]] - loss[at[a2]]) / (a2 - a1)
    u = (math.log(near[at[b1]]) - math.log(near[at[b2]])) / (b2 - b1)
    after = step >= 100                                  # the two layers are equal at step 0
    behind = np.nonzero(after & (st["band_l1"] >= st["band_l0"]))[0]
    return {"band_l0_at_300": float(st["band_l0"][at[300]]), "offsets_1_2_l0_at_300": float(near[at[300]]),
            "loss_drop_at_300": float(loss[0] - loss[at[300]]),
            "t_crit_loss": float(a2 + (LOSS_DROP - (loss[0] - loss[at[a2]])) / v) if v > 0 else None,
            "t_crit_attn": float(b2 + math.log(near[at[b2]] / NEAR_FLOOR) / u) if u > 0 else None,
            "measured": {"loss_drop_reaches_0.295": first_crossing(step, loss[0] - loss, LOSS_DROP, True),
                         "offsets_1_2_l0_falls_to_0.0085": first_crossing(step, near, NEAR_FLOOR, False),
                         "last_step_with_layer_1_ahead_in_the_band": float(step[behind[-1]]) if len(behind) else None,
                         "band_l1_falls_to_0.33": first_crossing(step[after], st["band_l1"][after], 0.33, False),
                         "loss_rep_falls_below_2": first_crossing(step, loss, 2.0, False)}}


def median(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_band")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_period_pilot.json")
    args = ap.parse_args(argv)
    jobs = {arm: arm_jobs(arm) for arm in ARMS}
    args.work.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            for msg in pool.imap(run_arm, [(j, str(args.work)) for js in jobs.values() for j in js]):
                print(msg, flush=True)
    have = {arm: [j for j in js if (args.work / f"period_{j['name']}.npz").exists()] for arm, js in jobs.items()}
    smp = fp.sample(12, 32)

    # final circuits of the full-length arms
    rows = {}
    for arm, js in have.items():
        if ARMS[arm][2] != FULL_STEPS:
            continue
        rows[arm] = []
        for j in js:
            r = wp.row_of({**j, "name": f"period_{j['name']}"}, args.work, smp)
            step, st = curves(j, args.work)
            r.update({"name": j["name"], "d1_l0_max": float(st["d1_l0_max"][-1]),
                      "d2_l0_max": float(st["d2_l0_max"][-1])})
            rows[arm].append(r)

    def outcome(arm: str) -> dict:
        rs = rows.get(arm, [])
        return {"window": list(ARMS[arm][1]) if ARMS[arm][1] else None,
                "textbook": int(sum(r["textbook"] for r in rs)),
                "relays": int(sum(r["formed"] and not r["textbook"] for r in rs)),
                "not_formed": int(sum(not r["formed"] for r in rs)), "of": len(rs),
                "complete": len(rs) == len(jobs[arm]), "relay_deltas": [r["relay_delta"] for r in rs],
                "deep_decoy_over_clean": [r["deep_decoy_over_clean"] for r in rs],
                "loss_all": [r["loss_all"] for r in rs], "d2_l0_max": [r["d2_l0_max"] for r in rs],
                "first_step_with_loss_rep_below_2": [r["first_step_with_loss_rep_below_2"] for r in rs]}

    settings = {}
    for setting in SETTINGS:
        nat = f"{setting}_natural"
        per_run = {j["base"]: early(j, args.work) for j in have[nat]}
        keys = ("band_l0_at_300", "offsets_1_2_l0_at_300", "loss_drop_at_300", "t_crit_loss", "t_crit_attn")
        settings[setting] = {
            "formula_steps": {"loss": list(FORMULA_STEPS[setting][0]), "attention": list(FORMULA_STEPS[setting][1])},
            "unperturbed_runs": per_run,
            "medians": {**{k: median([r[k] for r in per_run.values()]) for k in keys},
                        "measured": {k: median([r["measured"][k] for r in per_run.values()])
                                     for k in next(iter(per_run.values()))["measured"]} if per_run else {}},
            "arms": {arm: outcome(arm) for arm in ARMS if ARMS[arm][0] == setting and ARMS[arm][2] == FULL_STEPS}}

    def flips(arm: str) -> int | None:
        o = settings[ARMS[arm][0]]["arms"][arm]
        return o["textbook"] if o["complete"] else None

    def med(setting: str, key: str):
        return settings[setting]["medians"].get(key)

    def within(x, lo, hi) -> bool | None:
        return None if x is None else bool(lo <= x <= hi)

    def test(x, fn) -> bool | None:
        return None if x is None else bool(fn(x))

    prediction_1a = {
        "lr5e-4": {"band_l0_at_300_in_0.52_0.65": within(med("lr5e-4", "band_l0_at_300"), 0.52, 0.65),
                   "offsets_1_2_l0_at_300_in_0.028_0.043": within(med("lr5e-4", "offsets_1_2_l0_at_300"), 0.028, 0.043),
                   "t_crit_loss_in_760_980": within(med("lr5e-4", "t_crit_loss"), 760, 980),
                   "w600_1600_at_least_5": test(flips("lr5e-4_w600_1600"), lambda n: n >= 5),
                   "w700_1700_at_least_4": test(flips("lr5e-4_w700_1700"), lambda n: n >= 4),
                   "w1050_2050_at_most_1": test(flips("lr5e-4_w1050_2050"), lambda n: n <= 1)},
        "lr2e-3": {"t_crit_loss_in_200_265": within(med("lr2e-3", "t_crit_loss"), 200, 265),
                   "t_crit_attn_in_200_265": within(med("lr2e-3", "t_crit_attn"), 200, 265),
                   "band_l0_at_300_at_least_0.85": test(med("lr2e-3", "band_l0_at_300"), lambda x: x >= 0.85),
                   "offsets_1_2_l0_at_300_at_most_0.010": test(med("lr2e-3", "offsets_1_2_l0_at_300"),
                                                               lambda x: x <= 0.010),
                   "w300_800_at_most_1": test(flips("lr2e-3_w300_800"), lambda n: n <= 1),
                   "w150_650_at_least_5": test(flips("lr2e-3_w150_650"), lambda n: n >= 5)},
        "fails_by_its_own_criteria": None}
    f600, f300 = flips("lr5e-4_w600_1600"), flips("lr2e-3_w300_800")
    if f600 is not None and f300 is not None:
        prediction_1a["fails_by_its_own_criteria"] = bool(
            f600 <= 3 or f300 >= 3 or prediction_1a["lr5e-4"]["t_crit_loss_in_760_980"] is False
            or prediction_1a["lr2e-3"]["t_crit_loss_in_200_265"] is False)
    b64, b128, b32 = (med(s, "band_l0_at_300") for s in ("base", "d128", "d32"))
    prediction_1b = {
        "d128": {"band_l0_at_300_at_least_0.82": test(b128, lambda x: x >= 0.82),
                 "offsets_1_2_l0_at_300_at_most_0.013": test(med("d128", "offsets_1_2_l0_at_300"),
                                                             lambda x: x <= 0.013),
                 "loss_drop_at_300_at_least_0.23": test(med("d128", "loss_drop_at_300"), lambda x: x >= 0.23),
                 "t_crit_loss_in_260_350": within(med("d128", "t_crit_loss"), 260, 350),
                 "t_crit_attn_in_260_350": within(med("d128", "t_crit_attn"), 260, 350),
                 "w400_900_at_most_2": test(flips("d128_w400_900"), lambda n: n <= 2),
                 "w240_740_at_least_5": test(flips("d128_w240_740"), lambda n: n >= 5)},
        "d32": {"band_l0_at_300_in_0.54_0.67": within(b32, 0.54, 0.67),
                "offsets_1_2_l0_at_300_at_least_0.026": test(med("d32", "offsets_1_2_l0_at_300"), lambda x: x >= 0.026),
                "loss_drop_at_300_in_0.09_0.14": within(med("d32", "loss_drop_at_300"), 0.09, 0.14),
                "t_crit_loss_in_560_760": within(med("d32", "t_crit_loss"), 560, 760),
                "t_crit_attn_in_560_760": within(med("d32", "t_crit_attn"), 560, 760),
                "w500_1000_at_least_4": test(flips("d32_w500_1000"), lambda n: n >= 4)},
        "fails_by_its_own_criteria": None}
    f400, f500 = flips("d128_w400_900"), flips("d32_w500_1000")
    if None not in (f400, f500, b64, b128, b32):
        prediction_1b["fails_by_its_own_criteria"] = bool(b128 <= b64 or b32 >= b64 or f400 >= 5 or f500 <= 1)

    # inside the window: medians over the six baseline runs, every 100 steps
    inside = {}
    for arm in ("base_natural", "base_w400_900", "base_w500_1000"):
        cs = [curves(j, args.work) for j in have[arm]]
        if not cs:
            continue
        step = cs[0][0]
        pick = [i for i, s in enumerate(step) if s % 100 == 0 and 300 <= s <= SHORT_STEPS]
        inside[arm] = {"step": [int(step[i]) for i in pick],
                       **{k: [float(np.median([c[1][k][i] for c in cs])) for i in pick] for k in STATS},
                       "runs": {j["base"]: {k: [float(c[1][k][i]) for i in pick] for k in STATS}
                                for j, c in zip(have[arm], cs)}}

    result = {"status": "exploratory pilot; local runs; not independently recomputed",
              "missing": [j["name"] for arm, js in jobs.items() for j in js if j not in have[arm]],
              "constants": {"loss_drop": LOSS_DROP, "offsets_1_2_floor": NEAR_FLOOR},
              "settings": settings, "prediction_1A": prediction_1a, "prediction_1B": prediction_1b,
              "inside_the_window": inside, "runs": rows}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    for setting, blk in settings.items():
        m = blk["medians"]
        if m.get("band_l0_at_300") is None:
            continue
        print(f"== {setting}: at step 300 band_l0 {m['band_l0_at_300']:.3f}, offsets 1-2 "
              f"{m['offsets_1_2_l0_at_300']:.4f}, loss drop {m['loss_drop_at_300']:.3f}; t_crit_loss "
              f"{m['t_crit_loss']}, t_crit_attn {m['t_crit_attn']}")
        print("   measured (medians):", json.dumps(m["measured"]))
        for name, r in blk["unperturbed_runs"].items():
            print(f"     {name}: band_l0(300) {r['band_l0_at_300']:.3f} near {r['offsets_1_2_l0_at_300']:.4f} drop "
                  f"{r['loss_drop_at_300']:.3f} t_crit loss/attn {r['t_crit_loss']}/{r['t_crit_attn']} | "
                  + json.dumps(r["measured"]))
        for arm, o in blk["arms"].items():
            print(f"   {arm}: textbook {o['textbook']}/{o['of']}, relays {o['relays']}, not formed {o['not_formed']}; "
                  f"δ {o['relay_deltas']}; decoy {[round(x, 2) for x in o['deep_decoy_over_clean']]}; "
                  f"d2 {[round(x, 2) for x in o['d2_l0_max']]}; first<2 {o['first_step_with_loss_rep_below_2']}")
    print("prediction 1A:", json.dumps(prediction_1a))
    print("prediction 1B:", json.dumps(prediction_1b))
    for arm, blk in inside.items():
        print(f"== inside, {arm} (medians): step / latent band_l0 / band_l1 / source_l1 / d1 max / d2 max / loss")
        for i, s in enumerate(blk["step"]):
            print(f"   {s:>5d} {blk['band_l0'][i]:.2f} {blk['band_l1'][i]:.2f} {blk['source_l1'][i]:.3f} "
                  f"{blk['d1_l0_max'][i]:.3f} {blk['d2_l0_max'][i]:.3f} {blk['loss_rep'][i]:.2f}")
    print("missing:", len(result["missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
