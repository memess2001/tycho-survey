"""Pilot (exploratory; local runs): controls for the band intervention, its time window, the transition, tiled.

``relay_band_pilot.py`` found that keeping one layer out of the positional band for the first 500 steps decides which
circuit the single-repeat induction task ends with. The reviewer's fourth reply (17 November, to commit f6dedbf) lists
the controls that are still missing and makes three predictions; all are on record before these runs.

Controls (each: 6 runs, restriction during steps 0–499, then free training to 12,000; "sham" forbids everything
except offsets 0–3 and the band, so it removes more offsets than the band mask and leaves the band open):

* ``sham_l0_rm12``: sham mask on layer 0, rep_min 12. Expected: at most 1 of 6 textbook circuits (the relays stay).
  The reading "the band decides" fails if 3 or more turn into textbook circuits.
* ``band_l1_rm20``: band mask on layer 1, rep_min 20. Expected: at least 5 of 6 relays, at δ 18–19; fails below 4.
* ``sham_l1_rm20``: sham mask on layer 1, rep_min 20. Expected: at most 1 of 6 relays; fails at 2 or more.
* ``band_l1_rm12``: band mask on layer 1 where the relay forms anyway (rep_min 12): a control for the shock of lifting
  a mask. Expected: 0 of 6 textbook circuits; fails at 2 or more.

Prediction 2b, the window (band mask on layer 0, rep_min 12; textbook = relay at the source and a two-token key):

* steps 0–99 only: at least 5 of 6 textbook circuits, with a whole-sequence loss of at most 2.92;
* steps 200–699 and 400–899: at least 5 of 6;
* steps 500–999 and 600–1,099: at most 1 of 6. The prediction fails if 0–99 or 400–899 gives fewer than 5, or if
  500–999 still gives 3 or more (no cliff between a start at 400 and a start at 500).

Prediction 2a, the transition (no intervention; rep_min 13, 14, 15; 2 and 4 heads; seeds 300–302): textbook circuits
in 3 of 6 (allowed 2–4), 5 of 6 (4–6) and 5 of 6 (4–6), 13 of 18 in all (11–15), fewer at 13 than at 15; the other
runs are relays at δ in [rep_min − 2, rep_min − 1]. And at step 300, before any circuit has formed, layer 1's
attention on the source position (threshold 0.031, read off the 24 runs of ``relay_band_pilot.py``) tells the final
circuit in at least 15 of the 18 new runs.

Prediction 2c, the tiled layout (the three tiled runs of ``relay_offsets_pilot.py``; layer 1 restricted to offsets
0–3 for the first 500 steps): none of the 3 ends as a textbook circuit; relay δ in {1, 2} for every layer-1 head; a
layer-0 head with its top offset in 11–13 and no layer-0 head with more than 0.25 at offset 1; layer 1's attention
on the source at most 0.06; repeat loss between 0.035 and 0.095. Fails if a run returns to δ = 0, if δ leaves 1–3,
if a layer-0 head has 0.40 or more at offset 1, or if the loss is outside 0.025–0.20.

Note on the record: the reviewer had run predictions 2a–2c itself before stating them (its appended script: about
2,000 steps for 2a and 2b, 12,000 for 2c), so here they are a reproduction at 12,000 steps. The four controls were
not in that script.

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/relay_window_pilot.py
--jobs 4`` (``--skip-runs`` analyses the cache; runs that are missing are left out and listed). Needs the outputs
of ``relay_followup_pilot.py`` and ``relay_band_pilot.py``. Runs are cached in ``work/relay_band``. Writes
``docs/results/v2_analysis/relay_window_pilot.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("relay_band_pilot", Path(__file__).with_name("relay_band_pilot.py"))
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
fp, rp = rb.fp, rb.rp

FULL_STEPS = rb.FULL_STEPS
SOURCE_THRESHOLD = 0.031                                # the reviewer's threshold for layer 1's attention on the source
#: The reviewer's run-by-run prediction for the transition: the runs that end as relays; all others textbook.
PREDICTED_RELAYS = ("single_rm13_h2_s302", "single_rm13_h4_s301", "single_rm13_h4_s302", "single_rm14_h4_s302",
                    "single_rm15_h4_s302")
#: arm -> (rep_min or "tiled", restricted layer, forbidden offsets, first step, end step); in order of priority
ARMS = {
    "sham_l0_rm12": (12, 0, "sham", 0, 500), "band_l1_rm20": (20, 1, "band", 0, 500),
    "sham_l1_rm20": (20, 1, "sham", 0, 500), "band_l1_rm12": (12, 1, "band", 0, 500),
    "tiled_l1_far": ("tiled", 1, "far", 0, 500),
    "w0_100": (12, 0, "band", 0, 100), "w400_900": (12, 0, "band", 400, 900), "w500_1000": (12, 0, "band", 500, 1000),
    "natural_rm13": (13, 0, "band", 0, 0), "natural_rm14": (14, 0, "band", 0, 0), "natural_rm15": (15, 0, "band", 0, 0),
    "w200_700": (12, 0, "band", 200, 700), "w600_1100": (12, 0, "band", 600, 1100),
}
MEASURE_AT = tuple(sorted(set(range(0, FULL_STEPS + 1, 250)) | {200, 300}))
STATS = ("band_l0", "band_l1", "source_l1", "offsets_1_2_l0", "loss_rep")


def arm_jobs(arm: str) -> list[dict]:
    setting, layer, forbid, start, end = ARMS[arm]
    base = {j["name"]: j for j in rp.jobs_list()}
    if setting == "tiled":
        templates = [base[f"tiled_h3_s{seed}"] for seed in (310, 311, 312)]
        sources = [{"name": t["name"], "cfg": dict(t["cfg"]), "seed": t["seed"], "train": dict(t["train"])}
                   for t in templates]
    else:
        sources = [{"name": f"single_rm{setting}_h{nh}_s{seed}",
                    "cfg": {**base[f"single_rm12_h{nh}_s{seed}"]["cfg"], "rep_min": setting}, "seed": seed,
                    "train": dict(base[f"single_rm12_h{nh}_s{seed}"]["train"])}
                   for nh in rb.HEADS for seed in rb.SEEDS]
    return [{**s, "name": f"{arm}_{s['name']}", "base": s["name"], "arm": arm, "layer": layer, "forbid": forbid,
             "window": (start, end), "train": {**s["train"], "steps": FULL_STEPS}, "released_id": None}
            for s in sources]


def run_arm(args: tuple[dict, str]) -> str:
    job, work = args
    out = Path(work) / f"{job['name']}.npz"
    if out.exists():
        return f"cached {out.name}"
    model, task, stats, _ = rb.train(job, FULL_STEPS, MEASURE_AT, block_from=job["window"][0],
                                     block_until=job["window"][1], layer=job["layer"], forbid=job["forbid"])
    model.eval()
    arrays = {f"param/{k}": v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update({f"stat_{k}": np.asarray([stats[s][k] for s in MEASURE_AT]) for k in STATS})
    np.savez_compressed(out, logodds=task.pattern_logodds(model)[None, :], steps=np.asarray([FULL_STEPS]),
                        lc_step=np.asarray(MEASURE_AT), lc_loss_rep=arrays["stat_loss_rep"],
                        object_id=np.asarray(f"{job['arm']}:{job['base']}"), **arrays)
    return f"done {job['name']}"


def row_of(job: dict, work: Path, smp: dict | None) -> dict:
    summary = rp.attention_summary(job, work)
    with np.load(work / f"{job['name']}.npz") as f:
        step = f["lc_step"]
        stat = {k: f[f"stat_{k}"] for k in STATS}
    at = {int(s): i for i, s in enumerate(step)}
    row = {"arm": job["arm"], "name": job["name"], "n_head": summary["n_head"], "relay_delta": summary["relay_delta"],
           "p_all": summary["p_all"], "p_best_with_one_layer0_head": summary["p_best_with_one_layer0_head"],
           "loss_rep_probe": float(stat["loss_rep"][-1]),
           "first_step_with_loss_rep_below_2": (int(step[np.argmax(stat["loss_rep"] < 2.0)])
                                                if (stat["loss_rep"] < 2.0).any() else None),
           "at_step_300": {k: float(stat[k][at[300]]) for k in STATS},
           "source_l1_at_the_end": float(stat["source_l1"][-1]),
           "layer0": {h["head"]: {"top_offsets": [[t["d"], round(t["attention"], 3)] for t in h["top_offsets"]],
                                  "attention_d1": h["attention_d1"]}
                      for h in summary["heads"] if h["head"].startswith("l0")},
           "layer1_modal_deltas": [h["modal_top_delta"] for h in summary["heads"] if h["head"].startswith("l1")]}
    if smp is not None:
        k = fp.key_use(job, work, smp, summary)["intact"]
        formed = bool(k["loss_rep"] < 2.0)                 # a circuit has formed (an untrained model has 4.0–4.2)
        row.update({"deep_decoy_over_clean": k["deep_decoy_over_clean"], "loss_all": k["loss_all"],
                    "loss_rep": k["loss_rep"], "loss_other": k["loss_other"], "formed": formed,
                    "textbook": bool(formed and summary["relay_delta"] == 0 and k["deep_decoy_over_clean"] >= 0.9)})
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_band")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_window_pilot.json")
    args = ap.parse_args(argv)
    jobs = [j for arm in ARMS for j in arm_jobs(arm)]
    args.work.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            for msg in pool.imap(run_arm, [(j, str(args.work)) for j in jobs]):
                print(msg, flush=True)
    have = [j for j in jobs if (args.work / f"{j['name']}.npz").exists()]
    results = REPO / "docs" / "results" / "v2_analysis"
    final = {r["name"]: r for r in json.loads((results / "relay_followup_pilot.json").read_text())["runs"]}
    natural_tiled = {r["name"]: r for r in json.loads((results / "relay_offsets_pilot.json").read_text())["runs"]
                     if r["name"].startswith("tiled")}
    samples: dict = {}
    rows = []
    for j in have:
        rm = ARMS[j["arm"]][0]
        if rm != "tiled" and rm not in samples:
            samples[rm] = fp.sample(rm, 32)
        row = row_of(j, args.work, None if rm == "tiled" else samples[rm])
        if j["base"] in final:                           # the same run without the intervention (12,000 steps)
            base = final[j["base"]]
            row["without_intervention"] = {
                "relay_delta": base["relay_delta"], "loss_all": base["intact"]["loss_all"],
                "deep_decoy_over_clean": base["intact"]["deep_decoy_over_clean"],
                "textbook": bool(base["relay_delta"] == 0 and base["intact"]["deep_decoy_over_clean"] >= 0.9)}
        elif j["base"] in natural_tiled:
            nat = natural_tiled[j["base"]]
            task, model = fp.load_model({"name": j["base"], "cfg": j["cfg"], "seed": j["seed"]},
                                        args.work.parent / "relay_offsets")
            st = rb.band_stats(model, task)
            row["without_intervention"] = {
                "relay_delta": nat["relay_delta"], "p_all": nat["p_all"], "loss_rep_probe": st["loss_rep"],
                "source_l1_at_the_end": st["source_l1"],
                "layer1_modal_deltas": [h["modal_top_delta"] for h in nat["heads"] if h["head"].startswith("l1")],
                "layer0": {h["head"]: {"top_offsets": [[t["d"], round(t["attention"], 3)] for t in h["top_offsets"]],
                                       "attention_d1": h["attention_d1"]}
                           for h in nat["heads"] if h["head"].startswith("l0")}}
        natural = args.work.parent / "relay_offsets" / f"{j['base']}.npz"
        if "without_intervention" in row and natural.exists():
            with np.load(natural) as f:                  # the runner's own evaluation grid (finer than every 250)
                below = f["lc_loss_rep"] < 2.0
                row["without_intervention"]["first_step_with_loss_rep_below_2"] = (
                    int(f["lc_step"][np.argmax(below)]) if below.any() else None)
        rows.append(row)
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}

    def count(arm: str, test) -> dict:
        rs = by_arm[arm]
        return {"count": int(sum(test(r) for r in rs)), "of": len(rs), "complete": len(rs) == len(arm_jobs(arm))}

    relay = lambda r: r["formed"] and not r["textbook"]   # noqa: E731
    tb = lambda r: r["textbook"]                          # noqa: E731

    def outcome(arm: str) -> dict:                       # what the six runs of a control arm ended as
        rs = by_arm[arm]
        return {"textbook": count(arm, tb), "relays": count(arm, relay),
                "not_formed": count(arm, lambda r: not r["formed"]),
                "relay_deltas": [r["relay_delta"] for r in rs],
                "first_step_with_loss_rep_below_2": [r["first_step_with_loss_rep_below_2"] for r in rs],
                "without_intervention": [r["without_intervention"]["first_step_with_loss_rep_below_2"] for r in rs]}

    controls = {
        "sham_l0_rm12": {**outcome("sham_l0_rm12"), "expected": "at most 1 textbook", "fails_at": "3 textbook"},
        "band_l1_rm20": {**outcome("band_l1_rm20"), "expected": "at least 5 relays", "fails_below": "4 relays"},
        "sham_l1_rm20": {**outcome("sham_l1_rm20"), "expected": "at most 1 relay", "fails_at": "2 relays"},
        "band_l1_rm12": {**outcome("band_l1_rm12"), "expected": "0 textbook", "fails_at": "2 textbook"}}
    windows = {arm: {"steps": list(ARMS[arm][3:]), "textbook": count(arm, tb),
                     "textbook_and_loss_at_most_2.92": count(arm, lambda r: r["textbook"] and r["loss_all"] <= 2.92),
                     "formed": count(arm, lambda r: r["formed"]),
                     "relay_deltas": [r["relay_delta"] for r in by_arm[arm]]}
               for arm in ("w0_100", "w200_700", "w400_900", "w500_1000", "w600_1100")}
    windows["fails_by_its_own_criteria"] = bool(
        windows["w0_100"]["textbook"]["count"] < 5 or windows["w400_900"]["textbook"]["count"] < 5
        or windows["w500_1000"]["textbook"]["count"] >= 3)
    transition = {}
    for rm in (13, 14, 15):
        rs = by_arm[f"natural_rm{rm}"]
        transition[str(rm)] = {
            "textbook": count(f"natural_rm{rm}", tb), "textbook_runs": [r["name"] for r in rs if r["textbook"]],
            "relay_deltas_of_the_others": [r["relay_delta"] for r in rs if not r["textbook"]],
            "source_l1_at_300": {r["name"]: r["at_step_300"]["source_l1"] for r in rs},
            "already_formed_at_300": int(sum(r["at_step_300"]["loss_rep"] < 2.0 for r in rs))}
    new = [r for rm in (13, 14, 15) for r in by_arm[f"natural_rm{rm}"]]
    base_name = lambda r: r["name"][len(r["arm"]) + 1:]   # noqa: E731
    counts = {rm: sum(r["textbook"] for r in by_arm[f"natural_rm{rm}"]) for rm in (13, 14, 15)}
    others = [(r, ARMS[r["arm"]][0]) for r in new if not r["textbook"]]

    def best_threshold(key) -> dict:                     # the best single threshold on a step-300 quantity, in sample
        v = np.asarray([key(r) for r in new], dtype=float)
        lab = np.asarray([r["textbook"] for r in new], dtype=bool)
        acc = [(int(((v >= t) == lab).sum()), float(t)) for t in np.unique(v)] or [(0, float("nan"))]
        return {"correct": max(acc)[0], "threshold": max(acc)[1], "always_textbook": int(lab.sum()),
                "never_textbook": int((~lab).sum())}

    transition["all"] = {
        "textbook": int(sum(r["textbook"] for r in new)), "of": len(new), "predicted": 13, "allowed": [11, 15],
        "per_rep_min_within_the_allowed_range": {"13": 2 <= counts[13] <= 4, "14": counts[14] >= 4,
                                                 "15": counts[15] >= 4},
        "fewer_at_13_than_at_15": counts[13] < counts[15],
        "runs_as_predicted": int(sum((base_name(r) not in PREDICTED_RELAYS) == r["textbook"] for r in new)),
        "others_formed": int(sum(r["formed"] for r, _ in others)),
        "others_with_relay_delta_in_rep_min_minus_2_or_1": int(sum(rm - 2 <= r["relay_delta"] <= rm - 1
                                                                   for r, rm in others)),
        "none_formed_at_300": not any(r["at_step_300"]["loss_rep"] < 2.0 for r in new),
        "source_threshold": SOURCE_THRESHOLD,
        "correct_by_source_l1_at_300": int(sum((r["at_step_300"]["source_l1"] >= SOURCE_THRESHOLD) == r["textbook"]
                                               for r in new)),
        "correct_by_sign_of_band_difference_at_300": int(sum(
            ((r["at_step_300"]["band_l1"] - r["at_step_300"]["band_l0"]) > 0) == r["textbook"] for r in new)),
        "correct_by_either": int(sum(
            ((r["at_step_300"]["source_l1"] >= SOURCE_THRESHOLD)
             or (r["at_step_300"]["band_l1"] - r["at_step_300"]["band_l0"]) > 0) == r["textbook"] for r in new)),
        "predicted_at_least": 15,
        "best_threshold_in_sample": {"source_l1": best_threshold(lambda r: r["at_step_300"]["source_l1"]),
                                     "band_difference": best_threshold(
                                         lambda r: r["at_step_300"]["band_l1"] - r["at_step_300"]["band_l0"])}}
    tiled_runs = [{"name": r["name"], "relay_delta": r["relay_delta"],
                   "layer1_modal_deltas": r["layer1_modal_deltas"], "layer0": r["layer0"],
                   "max_layer0_attention_d1": max(v["attention_d1"] for v in r["layer0"].values()),
                   "a_layer0_head_with_top_offset_11_to_13": any(11 <= v["top_offsets"][0][0] <= 13
                                                                 for v in r["layer0"].values()),
                   "source_l1_at_the_end": r["source_l1_at_the_end"], "loss_rep_probe": r["loss_rep_probe"],
                   "p_all": r["p_all"], "p_best_with_one_layer0_head": r["p_best_with_one_layer0_head"],
                   "without_intervention": r.get("without_intervention")} for r in by_arm["tiled_l1_far"]]
    tiled = {"runs": tiled_runs,
             "not_textbook": int(sum(r["relay_delta"] != 0 for r in tiled_runs)), "of": len(tiled_runs),
             "relay_delta_in_1_or_2": int(sum(r["relay_delta"] in (1, 2) for r in tiled_runs)),
             "all_layer1_heads_in_1_or_2": int(sum(all(d in (1, 2) for d in r["layer1_modal_deltas"])
                                                   for r in tiled_runs)),
             "layer0_as_predicted": int(sum(r["a_layer0_head_with_top_offset_11_to_13"]
                                            and r["max_layer0_attention_d1"] <= 0.25 for r in tiled_runs)),
             "source_l1_at_most_0.06": int(sum(r["source_l1_at_the_end"] <= 0.06 for r in tiled_runs)),
             "loss_rep_within_0.035_and_0.095": int(sum(0.035 <= r["loss_rep_probe"] <= 0.095 for r in tiled_runs)),
             "fails_by_its_own_criteria": bool(any(
                 r["relay_delta"] == 0 or not 1 <= r["relay_delta"] <= 3 or r["max_layer0_attention_d1"] >= 0.40
                 or not 0.025 <= r["loss_rep_probe"] <= 0.20 for r in tiled_runs))}
    result = {"status": "exploratory pilot; local runs; not independently recomputed",
              "missing": [j["name"] for j in jobs if j not in have],
              "controls": controls, "prediction_2b_window": windows, "prediction_2a_transition": transition,
              "prediction_2c_tiled": tiled,
              "ladder_whole_sequence_loss": {
                  f"rep {rm}..32": {("inf" if m is None else str(m)): fp.score(p, smp)["loss_all"]
                                    for m, p in smp["ladder"].items()} for rm, smp in samples.items()},
              "runs": rows}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    for r in rows:
        if "textbook" in r:
            kind = "textbook" if r["textbook"] else "relay" if r["formed"] else "unformed"
            print(f"{r['arm']:>13s} {r['name'][len(r['arm']) + 1:]:>20s} δ={r['relay_delta']:>2d} "
                  f"{kind:>8s} decoy {r['deep_decoy_over_clean']:.2f} one-L0 "
                  f"{r['p_best_with_one_layer0_head']:.2f} loss {r['loss_rep']:.2f}/all {r['loss_all']:.3f} first<2 "
                  f"{r['first_step_with_loss_rep_below_2']} | at 300: source {r['at_step_300']['source_l1']:.4f} "
                  f"band {r['at_step_300']['band_l0']:.2f}/{r['at_step_300']['band_l1']:.2f}")
        else:
            print(f"{r['arm']:>13s} {r['name'][len(r['arm']) + 1:]:>20s} δ={r['relay_delta']:>2d} L1 deltas "
                  f"{r['layer1_modal_deltas']} loss {r['loss_rep_probe']:.3f} source at end "
                  f"{r['source_l1_at_the_end']:.3f} | L0: "
                  + "; ".join(f"{h} top d={v['top_offsets'][0][0]} ({v['top_offsets'][0][1]:.2f}), d1 "
                              f"{v['attention_d1']:.2f}" for h, v in r["layer0"].items()))
    print("controls:", json.dumps(controls))
    print("windows:", json.dumps({k: {"textbook": v["textbook"], "with_loss": v["textbook_and_loss_at_most_2.92"]}
                                  for k, v in windows.items() if isinstance(v, dict)}),
          "| fails:", windows["fails_by_its_own_criteria"])
    print("tiled:", json.dumps({k: v for k, v in tiled.items() if k != "runs"}))
    print("transition:", json.dumps({k: (v if k == "all" else {"textbook": v["textbook"], "others": v[
        "relay_deltas_of_the_others"]}) for k, v in transition.items()}))
    print("missing:", len(result["missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
