"""Pilot (exploratory; local runs): which layer takes the positional band first, and an intervention on it.

``relay_followup_pilot.py`` found that the circuit which forms in the single-repeat induction task flips with the
shortest repeat: displaced relays (or unformed circuits) for rep_min up to 12, the textbook circuit for rep_min 16
and 20. The reviewer's account (third reply of 17 November, to commit b656849; on record before these runs): both
layers can copy by position, that is, attend to the band of offsets d in [rep_min − 1, 31] at which a token can be
the next one. Layer 1 enters the band first. Then either layer 1, from inside the band, pulls a layer-0 head to
offset 1 (the textbook circuit), or layer 0 takes the band over, layer 1 is pushed out of it, and the relay is built
on layer 0's band offsets. Its predictions:

* 1A. With B_l(t) the attention mass of layer l on the band (probe set, predictable positions, mean over heads) and
  ΔB = B_1 − B_0: at step 200, ΔB > 0 in at least 22 of the 24 runs (rep_min 8, 12, 16, 20; 2 and 4 heads; seeds
  300–302). At step 600, ΔB tells the circuit found at 12,000 steps in at least 22 of 24: ΔB > +0.15 (median at
  least +0.45) for the 11 textbook runs, ΔB < −0.15 (median at most −0.45) for the 13 others. Fails below 20 of 24.
* 1B. Keep layer 0 out of the band at the start: in the six rep_min 12 runs, restrict layer 0's attention to offsets
  0–3 for the first 500 steps and train normally from there to 12,000. At least 5 of the 6, all displaced relays
  without the intervention, become textbook circuits (best ablation pattern with one layer-0 head at least 0.50,
  decoy ratio at least 0.90, whole-sequence loss at most 2.92). Fails below 4 of 6.

Two further arms, added by the analyst after the first one had run (expectations written before these runs):

* "band": as 1B, but layer 0 is kept out of the band only (offsets rep_min − 1 to 31 forbidden, all others allowed).
  The first arm also forces layer 0 onto the offsets the textbook circuit needs. If what matters is keeping layer 0
  out of the band, this arm should flip most runs as well (expected: at least 4 of 6).
* "mirror": the reverse intervention where the textbook circuit forms by itself. In the six rep_min 20 runs layer 1
  is restricted to offsets 0–3 for the first 500 steps. If the account is right, layer 0 takes the band meanwhile
  and most runs end as displaced relays (expected: at least 4 of 6), near δ = 19 if the relay sits at the band's
  lower edge.

The training loop here mirrors the runner's update (same initial weights, batch stream and optimiser); the script
checks that on one run by comparing its weights at step 600 with a 600-step run of the runner itself. The runs
without intervention are those of ``relay_offsets_pilot.py`` and ``relay_followup_pilot.py``; their final circuits
are read from ``relay_followup_pilot.json``.

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/relay_band_pilot.py
--jobs 4`` (``--skip-runs`` analyses the cache). Runs are cached in ``work/relay_band`` and ``work/relay_offsets``.
Writes ``docs/results/v2_analysis/relay_band_pilot.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("relay_followup_pilot",
                                               Path(__file__).with_name("relay_followup_pilot.py"))
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)
rp = fp.rp

REP_MINS, HEADS, SEEDS = (8, 12, 16, 20), (2, 4), (300, 301, 302)
EARLY_STEPS = (0, 100, 200, 300, 400, 600)
BLOCK_STEPS, NEAR, FULL_STEPS = 500, 4, 12_000
#: Interventions: (prefix of the cached run, rep_min, restricted layer, forbidden offsets: "far" = NEAR and beyond,
#: "band" = rep_min − 1 .. rep_max − 1).
ARMS = {"near": ("blocked", 12, 0, "far"), "band": ("bandblocked", 12, 0, "band"),
        "mirror": ("l1blocked", 20, 1, "far")}
CHECK = "single_rm12_h2_s300"                           # the run whose loop is checked against the runner


def jobs_list() -> list[dict]:
    base = {j["name"]: j for j in rp.jobs_list()}
    jobs = []
    for rm in REP_MINS:
        for nh in HEADS:
            for seed in SEEDS:
                t = base[f"single_rm12_h{nh}_s{seed}"]
                jobs.append({"name": f"single_rm{rm}_h{nh}_s{seed}", "cfg": {**t["cfg"], "rep_min": rm}, "seed": seed,
                             "train": dict(t["train"]), "released_id": None})
    return jobs


def band_stats(model, task) -> dict:
    """Attention of each layer from the predictable positions of the probe set: mass on the positional band, on
    offsets 1 and 2, and on the source position; mean over the layer's heads. Also the repeat loss."""
    pr = task._probe
    rep, src = pr["rep"].numpy(), pr["src"].numpy()
    maps = [a.numpy() for a in model.attention_maps(pr["x"])]          # per layer: (B, H, T, T)
    bi, ti = np.nonzero(rep)
    out = {}
    for layer, a in enumerate(maps):
        band = np.zeros((len(bi), a.shape[1]))
        for d in range(task.rep_min - 1, task.rep_max):   # the band: offsets rep_min − 1 .. rep_max − 1
            ok = ti >= d
            band[ok] += a[bi[ok], :, ti[ok], ti[ok] - d]
        near = a[bi, :, ti, np.maximum(ti - 1, 0)] * (ti >= 1)[:, None] + a[bi, :, ti, np.maximum(ti - 2, 0)] * (
            ti >= 2)[:, None]
        out[f"band_l{layer}"] = float(band.mean())
        out[f"offsets_1_2_l{layer}"] = float(near.mean())
        out[f"source_l{layer}"] = float(a[bi, :, ti, src[bi, ti]].mean())
    nll, mask = task._token_nll(model)
    out["loss_rep"] = float(nll[mask].mean())
    return out


def weight_groups(model) -> dict:
    """Named groups of weights, as lists of (parameter, rows): the query/key rows of each layer's attention
    ("qk0", "qk1"), its value rows and output projection ("ov0", "ov1"), the embeddings ("pos", "tok") and the
    unembedding ("head"). The LayerNorm parameters are in no group."""
    d = model.d_model
    groups = {"pos": [(model.pos.weight, slice(None))], "tok": [(model.tok.weight, slice(None))],
              "head": [(model.head.weight, slice(None))]}
    for i, block in enumerate(model.blocks):
        a = block.attn
        groups[f"qk{i}"] = [(a.qkv.weight, slice(0, 2 * d)), (a.qkv.bias, slice(0, 2 * d))]
        groups[f"ov{i}"] = [(a.qkv.weight, slice(2 * d, 3 * d)), (a.qkv.bias, slice(2 * d, 3 * d)),
                            (a.proj.weight, slice(None)), (a.proj.bias, slice(None))]
    return groups


def train(job: dict, steps: int, measure_at: tuple[int, ...], block_until: int = 0, keep_state_at: int | None = None,
          layer: int = 0, forbid: str = "far", block_from: int = 0, stats_fn=None, lr_scale: dict | None = None,
          out_scale: dict | None = None, out_switch: dict | None = None):
    """The runner's update loop on one induction job. While block_from <= step < block_until, ``layer`` may not
    attend to the forbidden offsets: "far" = NEAR and beyond; "band" = rep_min − 1 .. rep_max − 1; "sham" =
    everything except the near offsets and the band. Measurements are taken without the restriction, with
    ``stats_fn(model, task)`` (default: :func:`band_stats`). ``lr_scale`` multiplies the learning rate of some
    weights: {"all": factor for every weight, group of :func:`weight_groups`: factor for that group}; Adam's step
    is linear in the learning rate and its averages do not depend on it, so each step is scaled after it is taken.
    ``out_scale`` = {layer: factor} multiplies the output of that layer's attention by a fixed factor throughout,
    in training and in the measurements; before returning, the factor is folded into the layer's output projection,
    so the returned model computes the same function without it.
    ``out_switch`` = {"step": s, "fold": bool} takes the factors of ``out_scale`` away from step s on (before the
    update of that step and after its measurement). With ``fold`` the factor goes into the output projection and
    Adam's two averages for the projection are rescaled to match (first moment / factor, second / factor²), so the
    function is unchanged at the switch and training continues as it would without a factor from these weights.
    Without ``fold`` the factor is simply dropped: the layer's output jumps by 1 / factor at step s.
    Returns (model, task, {step: statistics}, state at ``keep_state_at``)."""
    import torch

    from tycho.tasks import make_task
    from tycho.tasks.base import torch_generator
    from tycho.train.runner import LeanOptimizer
    from tycho.train.spec import TrainConfig

    torch.set_num_threads(1)
    tc = TrainConfig.from_dict({**job["train"], "steps": steps})
    task = make_task("induction", job["cfg"], job["seed"])
    model = task.build_model()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = LeanOptimizer(params, tc)
    gen = torch_generator(job["seed"], "data")
    attn = model.blocks[layer].attn
    plain = attn.forward
    idx = torch.arange(model.max_len)
    offset = idx[:, None] - idx[None, :]
    band = (offset >= task.rep_min - 1) & (offset <= task.rep_max - 1)
    allowed = {"far": offset < NEAR, "band": ~band, "sham": (offset < NEAR) | band}[forbid]
    blocked = {"on": False}

    def restricted(x, mask):
        return plain(x, mask & allowed[: mask.shape[0], : mask.shape[1]] if blocked["on"] else mask)

    attn.forward = restricted
    factors = {int(lyr): float(factor) for lyr, factor in (out_scale or {}).items()}
    for lyr in factors:
        inner = model.blocks[lyr].attn.forward
        model.blocks[lyr].attn.forward = (lambda x, mask, inner=inner, lyr=lyr: factors[lyr] * inner(x, mask))
    scale = None
    if lr_scale:
        scale = [torch.full_like(p, float(lr_scale.get("all", 1.0))) for p in params]
        where = {id(p): i for i, p in enumerate(params)}
        for group, factor in lr_scale.items():
            if group != "all":
                for p, rows in weight_groups(model)[group]:
                    scale[where[id(p)]][rows] = float(factor)
    stats, kept = {}, None
    for step in range(steps + 1):
        if step in measure_at:
            blocked["on"] = False
            model.eval()
            stats[step] = (stats_fn or band_stats)(model, task)
            model.train()
        if step == keep_state_at:
            kept = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == steps:
            break
        if out_switch and step == int(out_switch["step"]):
            where_p = {id(p): i for i, p in enumerate(params)}
            for lyr, factor in factors.items():
                if out_switch["fold"] and opt.exp_avg is None:
                    raise ValueError("out_switch with fold needs Adam")
                if out_switch["fold"]:
                    a = model.blocks[lyr].attn
                    with torch.no_grad():
                        for w in (a.proj.weight, a.proj.bias):
                            w.mul_(factor)
                            opt.exp_avg[where_p[id(w)]].div_(factor)
                            opt.exp_avg_sq[where_p[id(w)]].div_(factor * factor)
                factors[lyr] = 1.0
        blocked["on"] = block_from <= step < block_until
        x, y = task.sample_batch(tc.batch_size, gen)
        loss = task.loss(model(x), y)
        before = [p.detach().clone() for p in params] if scale else None
        opt.step(list(torch.autograd.grad(loss, params, allow_unused=True, materialize_grads=True)), tc.lr)
        if scale:
            with torch.no_grad():
                for p, old, f in zip(params, before, scale):
                    p.copy_(old + f * (p - old))
    blocked["on"] = False
    for lyr, factor in factors.items():                  # fold the factor into the weights; the function is unchanged
        a = model.blocks[lyr].attn
        a.forward = type(a).forward.__get__(a)
        with torch.no_grad():
            a.proj.weight.mul_(factor)
            a.proj.bias.mul_(factor)
    return model, task, stats, kept


def run_early(args: tuple[dict, str]) -> str:
    job, work = args
    out = Path(work) / f"early_{job['name']}.json"
    if out.exists():
        return f"cached {out.name}"
    _, _, stats, kept = train(job, max(EARLY_STEPS), EARLY_STEPS, keep_state_at=max(EARLY_STEPS))
    res = {"name": job["name"], "steps": {str(s): v for s, v in stats.items()}}
    if job["name"] == CHECK:                            # the loop against the runner itself, at the last early step
        from tycho.train.runner import run_object
        from tycho.train.spec import ObjectSpec, RIConfig, TrainConfig

        spec = ObjectSpec(task="induction", task_cfg=job["cfg"], seed=job["seed"],
                          train=TrainConfig.from_dict({**job["train"], "steps": max(EARLY_STEPS)}),
                          sky_version="pilot-relay-band")
        r = run_object(spec, work_dir=Path(work) / "resume" / f"check_{job['name']}",
                       ri=RIConfig(enabled=True, mode="patterns", n_probe=1024, n_points=2), resume=False)
        res["max_abs_weight_difference_from_the_runner"] = float(max(
            np.abs(np.asarray(r.patterns[f"param/{k}"]) - v.numpy()).max() for k, v in kept.items()))
    out.write_text(json.dumps(res) + "\n", encoding="utf-8")
    return f"done early {job['name']}"


def run_blocked(args: tuple[dict, str]) -> str:
    job, work = args
    out = Path(work) / f"{job['name']}.npz"
    if out.exists():
        return f"cached {out.name}"
    at = tuple(range(0, FULL_STEPS + 1, 250))
    model, task, stats, _ = train(job, FULL_STEPS, at, block_until=BLOCK_STEPS, layer=job["layer"],
                                  forbid=job["forbid"])
    model.eval()
    arrays = {f"param/{k}": v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(out, logodds=task.pattern_logodds(model)[None, :], steps=np.asarray([FULL_STEPS]),
                        lc_step=np.asarray(at), lc_loss_rep=np.asarray([stats[s]["loss_rep"] for s in at]),
                        object_id=np.asarray(f"blocked:{job['name']}"),
                        band=np.asarray([[stats[s][f"band_l{layer}"] for layer in (0, 1)] for s in at]), **arrays)
    return f"done blocked {job['name']}"


def blocked_jobs(jobs: list[dict], arm: str) -> list[dict]:
    prefix, rep_min, layer, forbid = ARMS[arm]
    return [{**j, "name": f"{prefix}_{j['name']}", "base": j["name"], "layer": layer, "forbid": forbid,
             "train": {**j["train"], "steps": FULL_STEPS}} for j in jobs if j["cfg"]["rep_min"] == rep_min]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_band")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_band_pilot.json")
    args = ap.parse_args(argv)
    jobs = jobs_list()
    arms = {arm: blocked_jobs(jobs, arm) for arm in ARMS}
    args.work.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(run_blocked, [(j, str(args.work)) for a in arms.values() for j in a]):
                print(msg, flush=True)
            for msg in pool.imap_unordered(run_early, [(j, str(args.work)) for j in jobs]):
                print(msg, flush=True)

    # the circuits found at 12,000 steps without intervention
    follow = json.loads((REPO / "docs" / "results" / "v2_analysis" / "relay_followup_pilot.json").read_text())
    final = {r["name"]: r for r in follow["runs"]}

    def textbook(r: dict) -> bool:                      # the relay sits at the source and the key has two tokens
        return r["relay_delta"] == 0 and r["intact"]["deep_decoy_over_clean"] >= 0.9

    early, check = [], None
    for j in jobs:
        e = json.loads((args.work / f"early_{j['name']}.json").read_text())
        check = e.get("max_abs_weight_difference_from_the_runner", check)
        s = e["steps"]
        early.append({"name": j["name"], "rep_min": j["cfg"]["rep_min"], "n_head": j["cfg"]["n_head"],
                      "textbook_at_12000": textbook(final[j["name"]]),
                      "relay_delta_at_12000": final[j["name"]]["relay_delta"],
                      "delta_band": {st: s[st]["band_l1"] - s[st]["band_l0"] for st in s}, "steps": s})
    at200 = [r["delta_band"]["200"] for r in early]

    def split(step: str) -> dict:
        tb = [r["delta_band"][step] for r in early if r["textbook_at_12000"]]
        other = [r["delta_band"][step] for r in early if not r["textbook_at_12000"]]
        formed = [r["name"] for r in early if r["steps"][step]["loss_rep"] < 2.0]
        return {"textbook": {"n": len(tb), "above_plus_0.15": int(sum(x > 0.15 for x in tb)),
                             "median": float(np.median(tb)), "min": float(min(tb)), "max": float(max(tb))},
                "other": {"n": len(other), "below_minus_0.15": int(sum(x < -0.15 for x in other)),
                          "median": float(np.median(other)), "min": float(min(other)), "max": float(max(other))},
                "correct_by_the_stated_thresholds": int(sum(x > 0.15 for x in tb) + sum(x < -0.15 for x in other)),
                "correct_by_sign": int(sum(x > 0 for x in tb) + sum(x < 0 for x in other)), "of": len(early),
                "runs_whose_repeat_loss_is_already_below_2": len(formed)}

    prediction_1a = {
        "layer1_ahead_at_step_200": {"count": int(sum(x > 0 for x in at200)), "of": len(at200),
                                     "predicted_at_least": 22},
        "step_600": {**split("600"), "predicted_at_least": 22, "fails_below": 20},
        "step_400_for_comparison": split("400"), "step_300_for_comparison": split("300")}

    samples = {rm: fp.sample(rm, 32) for rm in sorted({v[1] for v in ARMS.values()})}
    references = {rm: {"inf" if m is None else str(m): fp.score(p, smp) for m, p in smp["ladder"].items()}
                  for rm, smp in samples.items()}
    rows = []
    for arm, j in ((arm, j) for arm, js in arms.items() for j in js):
        if not (args.work / f"{j['name']}.npz").exists():
            continue
        smp = samples[ARMS[arm][1]]
        summary = rp.attention_summary(j, args.work)
        k = fp.key_use(j, args.work, smp, summary)
        with np.load(args.work / f"{j['name']}.npz") as f:
            lc_step, lc_loss, band = f["lc_step"], f["lc_loss_rep"], f["band"]
        base = final[j["base"]]
        row = {"arm": arm, "name": j["name"], "n_head": summary["n_head"], "relay_delta": summary["relay_delta"],
               "p_all": summary["p_all"], "p_best_with_one_layer0_head": summary["p_best_with_one_layer0_head"],
               "deep_decoy_over_clean": k["intact"]["deep_decoy_over_clean"], "loss_all": k["intact"]["loss_all"],
               "loss_rep": k["intact"]["loss_rep"], "loss_other": k["intact"]["loss_other"],
               "first_step_with_loss_rep_below_2": (int(lc_step[np.argmax(lc_loss < 2.0)])
                                                    if (lc_loss < 2.0).any() else None),
               "band_layer0_layer1_at_500_and_750": [[float(x) for x in band[2]], [float(x) for x in band[3]]],
               "layer0_top_offsets": {h["head"]: [[t["d"], round(t["attention"], 3)] for t in h["top_offsets"]]
                                      for h in summary["heads"] if h["head"].startswith("l0")},
               "without_intervention": {"relay_delta": base["relay_delta"], "loss_all": base["intact"]["loss_all"],
                                        "deep_decoy_over_clean": base["intact"]["deep_decoy_over_clean"],
                                        "p_best_with_one_layer0_head":
                                            base["attention"]["p_best_with_one_layer0_head"]}}
        row["textbook_by_the_stated_criteria"] = bool(
            row["p_best_with_one_layer0_head"] >= 0.5 and row["deep_decoy_over_clean"] >= 0.9
            and row["loss_all"] <= 2.92)
        row["textbook_by_position_and_key"] = bool(row["relay_delta"] == 0 and row["deep_decoy_over_clean"] >= 0.9)
        rows.append(row)
    near = [r for r in rows if r["arm"] == "near"]
    prediction_1b = {"flipped_by_the_stated_criteria": int(sum(r["textbook_by_the_stated_criteria"] for r in near)),
                     "flipped_by_position_and_key": int(sum(r["textbook_by_position_and_key"] for r in near)),
                     "of": len(near), "predicted_at_least": 5, "fails_below": 4}
    further = {arm: {"textbook_by_position_and_key": int(sum(r["textbook_by_position_and_key"] for r in rows
                                                             if r["arm"] == arm)),
                     "relay_deltas": [r["relay_delta"] for r in rows if r["arm"] == arm],
                     "of": sum(r["arm"] == arm for r in rows)} for arm in ("band", "mirror")}
    result = {"status": "exploratory pilot; local runs; not independently recomputed",
              "loop_check": {"run": CHECK, "step": max(EARLY_STEPS),
                             "max_abs_weight_difference_from_the_runner": check},
              "prediction_1A": prediction_1a, "prediction_1B": prediction_1b, "further_arms": further,
              "ladder_whole_sequence_loss": {f"rep {rm}..32": {m: v["loss_all"] for m, v in ref.items()}
                                             for rm, ref in references.items()},
              "interventions": rows, "early": early}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    print("loop check: largest weight difference from the runner at step", max(EARLY_STEPS), "=", check)
    print("band mass by layer (layer 0 / layer 1) and their difference, before the circuit forms")
    for r in early:
        s = r["steps"]
        print(f"  {r['name']:>20s} {'textbook' if r['textbook_at_12000'] else 'other':>8s} | "
              + " | ".join(f"{st}: {s[st]['band_l0']:.2f}/{s[st]['band_l1']:.2f} ({r['delta_band'][st]:+.2f})"
                           for st in ("0", "200", "400", "600"))
              + f" | at 600: L0 offsets 1-2 {s['600']['offsets_1_2_l0']:.2f}, L1 source {s['600']['source_l1']:.2f},"
              + f" loss {s['600']['loss_rep']:.2f}")
    print("prediction 1A:", json.dumps(prediction_1a))
    print("interventions for the first", BLOCK_STEPS, "steps; ladder (whole-sequence loss):",
          result["ladder_whole_sequence_loss"])
    for r in rows:
        w = r["without_intervention"]
        print(f"  {r['arm']:>6s} {r['name']:>32s} relay δ={r['relay_delta']:>2d} one-L0 "
              f"{r['p_best_with_one_layer0_head']:.2f} decoy {r['deep_decoy_over_clean']:.2f} loss {r['loss_rep']:.2f}/"
              f"{r['loss_other']:.2f}/all {r['loss_all']:.3f} first<2 at {r['first_step_with_loss_rep_below_2']} | "
              f"without: δ={w['relay_delta']} one-L0 {w['p_best_with_one_layer0_head']:.2f} decoy "
              f"{w['deep_decoy_over_clean']:.2f} all {w['loss_all']:.3f}")
    print("prediction 1B:", json.dumps(prediction_1b), "| further arms:", json.dumps(further))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
