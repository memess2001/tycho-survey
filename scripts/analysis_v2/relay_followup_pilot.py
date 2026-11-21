"""Follow-up to relay_offsets_pilot.py (exploratory; local runs): three checks from the review round of 17 November.

1. Where does the relay sit when the shortest repeat changes? The reviewer's prediction, on record before these runs
   (reply of 17 November to commit 41d4d7a): the layer-0 heads are first drawn to offsets d ≥ rep_min − 1, the only
   ones at which a token predicts the next one directly, and a relay that works for every repeat length needs
   δ ≤ rep_min − 1, so the relay is pinned at δ = rep_min − 1. With rep_max 32: δ = 15 ± 1 for rep_min 16 and
   δ = 19 ± 1 for rep_min 20; the prediction fails if δ stays below 13 or scatters. Runs: pilot 1's settings with
   rep_min 16, 20 and (no prediction on record) 8; 2 and 4 heads per layer; seeds 300–302; 12,000 steps.

2. Does a run key on more than the current token? ``relay_offsets_pilot.py`` read a rising probability on the
   predictable targets as a sign of a second key token; the reviewer objected that a one-token matcher that hedges
   less scores higher there as well. Two measurements that do not have this problem, on 2,000 fresh sequences:
   * the loss on the other targets, next to the loss on the predictable ones (a predictor that hedges less pays
     there), for every cached run, against the key-length ladder on the same sequences;
   * decoys. A predictable position is "deep" when the previous token is a copy too. A decoy is an earlier
     occurrence of the current token at an admissible lag that is not the source. A one-token key cannot tell a
     decoy from the source, a two-token key can (the decoy's previous token differs, except by chance; those cases
     are left out). So the probability of the correct token at deep positions with a decoy, relative to deep
     positions without one, is about 0.6 for a one-token key and about one for a two-token key (the ladder's
     predictors on the same sequences give the reference values). Positive control: the runs with the textbook
     circuit, intact, with the t − 2 head removed, and under their best ablation pattern that keeps the
     previous-token head as the only layer-0 head.
   The losses are computed on these common sequences. The light curves' ``loss_rep`` is computed on each seed's own
   probe set of 128 sequences, and those sets differ: runs with data seed 302 read about 0.2 nats lower there.

3. What does a layer-1 head that attends to its own position do? The reviewer's proposal: it removes from the output
   what the layer-0 heads wrote at the current position. Test: remove that head and see which tokens gain
   probability (the tokens 1, 2, 11, 12, 13 positions back, at positions where none of them is the answer).

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/relay_followup_pilot.py
--jobs 4`` (``--skip-runs`` analyses the cache). Runs are cached in ``work/relay_offsets`` with those of
``relay_offsets_pilot.py``, which must have been run first. Writes
``docs/results/v2_analysis/relay_followup_pilot.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("relay_offsets_pilot",
                                               Path(__file__).with_name("relay_offsets_pilot.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

#: Shortest repeats of the sweep. 16 and 20 carry the reviewer's prediction; 8 has none on record.
REP_MINS = (16, 20, 8)
N_SEQ = 2000
OFFSETS = (0, 1, 2, 11, 12, 13)                        # tokens whose probability the cleaner test follows
STRATA = ("deep_clean", "deep_decoy", "first_clean", "first_decoy")


def sweep_jobs() -> list[dict]:
    base = {j["name"]: j for j in rp.jobs_list()}
    jobs = []
    for rm in REP_MINS:
        for nh in (2, 4):
            for seed in (300, 301, 302):
                t = base[f"single_rm12_h{nh}_s{seed}"]
                jobs.append({"name": f"single_rm{rm}_h{nh}_s{seed}", "cfg": {**t["cfg"], "rep_min": rm}, "seed": seed,
                             "train": dict(t["train"]), "sky_version": "pilot-relay-followup", "released_id": None})
    return jobs


def run_one(args: tuple[dict, str]) -> str:
    return rp.run_one(args)


# ------------------------------------------------------------------------------------------------- measurements
def load_model(job: dict, work: Path):
    import torch

    from tycho.tasks.induction import Induction

    with np.load(work / f"{job['name']}.npz") as f:
        state = {k[6:]: torch.as_tensor(f[k]) for k in f.files if k.startswith("param/")}
    task = Induction(job["cfg"], job["seed"])
    model = task.build_model()
    model.load_state_dict(state)
    model.eval()
    return task, model


def sample(rep_min: int, rep_max: int) -> dict:
    """Fresh sequences of the single layout with their strata and the ladder's probabilities on them."""
    from tycho.tasks.base import torch_generator
    from tycho.tasks.induction import Induction

    task = Induction({"vocab": rp.VOCAB, "seq_len": rp.SEQ_LEN, "rep_min": rep_min, "rep_max": rep_max,
                      "layout": "single"}, 0)
    x, rep, src = task._draw(N_SEQ, torch_generator(777, "induction/key-use"), 1.0)
    xs, rep, src = x.numpy(), rep.numpy(), src.numpy()
    N, T = rep.shape
    t = np.arange(T)[None, :]
    lag = np.where(rep, t - src + 1, 0)
    first = rep.argmax(1)[:, None]                      # the first predictable position of each sequence
    depth = np.where(rep, t - first + 1, 0)
    cur = xs[:, :T]
    n1, n2 = np.zeros((N, T), dtype=int), np.zeros((N, T), dtype=int)
    for d in range(rep_min, rep_max + 1):               # decoys: the current token d positions back, d not the lag
        same = np.zeros((N, T), dtype=bool)
        same[:, d:] = cur[:, d:] == cur[:, :-d]
        same &= lag != d
        both = np.zeros((N, T), dtype=bool)             # ... whose previous token matches as well
        both[:, d + 1:] = same[:, d + 1:] & (cur[:, d:-1] == cur[:, :-d - 1])
        n1 += same
        n2 += both
    strata = {"deep_clean": rep & (depth >= 2) & (n1 == 0), "deep_decoy": rep & (depth >= 2) & (n1 >= 1) & (n2 == 0),
              "first_clean": rep & (depth == 1) & (n1 == 0), "first_decoy": rep & (depth == 1) & (n1 >= 1)}
    ladder = rp.ladder_probabilities(xs, "single", rep_min, rep_max)
    return {"x": x, "xs": xs, "rep": rep, "lag": lag, "strata": strata, "ladder": ladder}


def probabilities(model, x, keep=None) -> np.ndarray:
    """Softmax outputs (N, T, vocab) of the model on the inputs x[:, :-1], with an optional head mask."""
    import torch

    out = []
    with torch.no_grad(), model.head_mask(keep):
        for i in range(0, len(x), 250):
            out.append(model(x[i:i + 250, :-1]).to(torch.float64).softmax(-1).numpy())
    return np.concatenate(out)


def score(p_true: np.ndarray, smp: dict) -> dict:
    rep = smp["rep"]
    out = {"loss_rep": float(-np.log(p_true[rep]).mean()), "loss_other": float(-np.log(p_true[~rep]).mean()),
           "loss_all": float(-np.log(p_true).mean()), "p_rep": float(p_true[rep].mean())}
    for name in STRATA:
        out[f"p_{name}"] = float(p_true[smp["strata"][name]].mean())
    out["deep_decoy_over_clean"] = out["p_deep_decoy"] / out["p_deep_clean"]
    out["first_decoy_over_clean"] = out["p_first_decoy"] / out["p_first_clean"]
    return out


def true_token(probs: np.ndarray, smp: dict) -> np.ndarray:
    return np.take_along_axis(probs, smp["xs"][:, 1:, None], axis=2)[..., 0]


def key_use(job: dict, work: Path, smp: dict, summary: dict) -> dict:
    """Losses and decoy strata of one run; for a textbook circuit also without its t − 2 head; for a run with a
    layer-1 head on its own position, what removing that head does."""
    task, model = load_model(job, work)
    nh = summary["n_head"]
    out = {"name": job["name"], "rep_min": summary["rep_min"], "n_head": nh, "steps": summary["steps"],
           "relay_delta": summary["relay_delta"], "intact": score(true_token(probabilities(model, smp["x"]), smp), smp)}
    l0 = [h for h in summary["heads"] if h["head"].startswith("l0")]
    l1 = [h for h in summary["heads"] if h["head"].startswith("l1")]
    prev = [i for i, h in enumerate(l0) if h["attention_d1"] >= 0.5]
    back2 = [i for i, h in enumerate(l0) if h["attention_d2"] >= 0.5]
    if prev and back2:                                   # textbook circuit with a t − 2 head: remove that head
        keep = np.ones(2 * nh)
        keep[back2] = 0.0
        out["without_t_minus_2_head"] = score(true_token(probabilities(model, smp["x"], keep), smp), smp)
        # ... and the best ablation pattern (by the run's own table) with the previous-token head as the only
        # layer-0 head: some layer-1 heads rely on the t − 2 head and have to go with it
        with np.load(work / f"{job['name']}.npz") as f:
            table = f["logodds"][-1]
        codes = np.arange(1 << (2 * nh))
        if table.size == codes.size:                     # no table is kept for more than 12 heads in all
            only_prev = (codes & ((1 << nh) - 1)) == (1 << prev[0])
            best = int(codes[only_prev][np.argmax(table[only_prev])])
            keep = (best >> np.arange(2 * nh)) & 1
            out["previous_token_head_only"] = {
                "kept": [f"l{j // nh}h{j % nh}" for j in range(2 * nh) if keep[j]],
                **score(true_token(probabilities(model, smp["x"], keep.astype(float)), smp), smp)}
    own = [i for i, h in enumerate(l1) if h["attention_to_own_position"] >= 0.3]
    if own:                                              # the "cleaner" test
        xs, rep, lag = smp["xs"], smp["rep"], smp["lag"]
        sel = rep & (lag >= 15)                          # none of the tokens followed below is the answer here
        T = rep.shape[1]
        keep = np.ones(2 * nh)
        keep[[nh + i for i in own]] = 0.0
        res = {"layer1_heads_removed": [f"l1h{i}" for i in own], "positions": int(sel.sum())}
        for label, k in (("intact", None), ("removed", keep)):
            probs = probabilities(model, smp["x"], k)
            row = {"p_correct": float(true_token(probs, smp)[sel].mean())}
            for d in OFFSETS:
                tok = np.zeros((len(xs), T), dtype=np.int64)
                tok[:, d:] = xs[:, :T - d] if d else xs[:, :T]
                ok = sel & (np.arange(T)[None, :] >= d)
                row[f"p_token_{d}_back"] = float(np.take_along_axis(probs, tok[..., None], axis=2)[..., 0][ok].mean())
            res[label] = row
        out["own_position_head"] = res
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_offsets")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_followup_pilot.json")
    args = ap.parse_args(argv)
    sweep = sweep_jobs()
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(run_one, [(j, str(args.work)) for j in sweep]):
                print(msg, flush=True)
    sweep = [j for j in sweep if (args.work / f"{j['name']}.npz").exists()]
    earlier = [j for j in rp.jobs_list() if j["cfg"].get("layout", "single") == "single"] + rp.long_jobs()
    earlier = [j for j in earlier if (args.work / f"{j['name']}.npz").exists()]

    samples, reference, runs = {}, {}, []
    for job in sweep + earlier:
        rm, hi = int(job["cfg"]["rep_min"]), int(job["cfg"]["rep_max"])
        if (rm, hi) not in samples:
            samples[rm, hi] = smp = sample(rm, hi)
            reference[f"repeat length {rm}..{hi}"] = {
                "strata_sizes": {k: int(v.sum()) for k, v in smp["strata"].items()},
                "key_length": {"inf" if m is None else str(m): score(p, smp) for m, p in smp["ladder"].items()}}
        summary = rp.attention_summary(job, args.work)
        row = key_use(job, args.work, samples[rm, hi], summary)
        row["sweep"] = job in sweep
        row["attention"] = {
            "loss_rep_probe": summary["loss_rep"], "p_all": summary["p_all"],
            "p_best_with_one_layer0_head": summary["p_best_with_one_layer0_head"],
            "layer0_top_offsets": {h["head"]: [[t["d"], round(t["attention"], 3)] for t in h["top_offsets"]]
                                   for h in summary["heads"] if h["head"].startswith("l0")},
            "layer1": {h["head"]: {"modal_top_delta": h["modal_top_delta"],
                                   "share_of_lengths": h["share_of_lengths_with_modal_delta"],
                                   "median_attention": h["median_attention_at_modal_delta"],
                                   "own_position": h["attention_to_own_position"]}
                       for h in summary["heads"] if h["head"].startswith("l1")}}
        runs.append(row)
    result = {"status": "exploratory pilot; local runs; not independently recomputed",
              "prediction_on_record": {"rep_min 16": "relay delta 15 +- 1", "rep_min 20": "relay delta 19 +- 1",
                                       "fails_if": "delta stays below 13 or scatters", "rep_min 8": "none"},
              "n_sequences": N_SEQ, "reference": reference, "runs": runs}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    print("relay position against the shortest repeat (sweep runs)")
    for r in runs:
        if r["sweep"]:
            a = r["attention"]
            l0 = "; ".join(f"{h}: " + ", ".join(f"d={d} ({v:.2f})" for d, v in tops[:2])
                           for h, tops in a["layer0_top_offsets"].items())
            print(f"  {r['name']:>22s} loss {a['loss_rep_probe']:.2f} p_all {a['p_all']:.2f} one-L0 "
                  f"{a['p_best_with_one_layer0_head']:.2f} relay δ={r['relay_delta']:>2d} | {l0}")
    print("losses and decoys (probability of the correct token; ratio = deep with decoy / deep without)")
    for label, ref in reference.items():
        print(f"  reference, {label}: " + "   ".join(
            f"m={m}: {v['loss_rep']:.3f} / {v['loss_other']:.3f} / all {v['loss_all']:.3f}, ratio "
            f"{v['deep_decoy_over_clean']:.2f} (first {v['first_decoy_over_clean']:.2f})"
            for m, v in ref["key_length"].items()))
    for r in runs:
        for label in ("intact", "without_t_minus_2_head", "previous_token_head_only"):
            if label in r:
                v = r[label]
                print(f"  {r['name']:>27s} {label:>24s}: loss {v['loss_rep']:.2f} / {v['loss_other']:.2f} / all "
                      f"{v['loss_all']:.3f} p {v['p_rep']:.2f} | deep {v['p_deep_clean']:.2f} / "
                      f"{v['p_deep_decoy']:.2f} ratio {v['deep_decoy_over_clean']:.2f}"
                      f" | first {v['p_first_clean']:.2f} / {v['p_first_decoy']:.2f} ratio "
                      f"{v['first_decoy_over_clean']:.2f}")
    print("layer-1 heads on their own position: probabilities with the head and without it")
    for r in runs:
        if "own_position_head" in r:
            o = r["own_position_head"]
            for label in ("intact", "removed"):
                print(f"  {r['name']:>27s} {'+'.join(o['layer1_heads_removed'])} {label:>8s}: correct "
                      f"{o[label]['p_correct']:.3f} | " + " ".join(f"{d} back {o[label][f'p_token_{d}_back']:.3f}"
                                                                   for d in OFFSETS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
