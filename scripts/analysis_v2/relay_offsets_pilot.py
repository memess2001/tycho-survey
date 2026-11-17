"""Pilot (exploratory; local re-runs): what the induction runs without the textbook circuit compute instead.

Asked in review (17 November 2026): the single-repeat runs of pilot 1 that have no textbook induction circuit still
predict the repeated token far better than any content-free solution, and their head-ablation tables show that they
need two layer-0 heads. The proposed mechanism is a *displaced relay*. In the textbook circuit a layer-1 head attends
from position t to the token after the previous occurrence, j = t − L + 1, where a previous-token head (offset 1) has
written the key. In a relay displaced by δ the layer-1 head attends to j + δ instead, and two layer-0 heads have to
carry the key (offset δ + 1) and the value (offset δ) there. The proposal also called the state a trap with a loss
bound near 0.8 nats. This script measures three things.

1. Attention offsets. The released pilots store no weights, so the networks are trained again with the repository's
   runner (same specifications and object ids; not bit-identical with the server runs) and their attention is read on
   the task's probe set: layer 0, the mean attention from t to t − d; layer 1, from the predictable positions, the
   mean attention to j + δ, overall and separately for every repeat length L.
   Runs: pilot 1's two-layer objects (single layout, rep_min 4 or 12, 2 or 4 heads per layer, seeds 300–302) and
   three runs of Part B's setting (tiled layout, 3 heads per layer, d_model 96, learning rate 3e-4, seeds 310–312).

2. A reference from the task alone: the key-length ladder. All these circuits find the earlier occurrence by matching
   a key of m tokens (m = 1: the current token; m = 2: the current and the previous token, which needs one more
   layer-0 head). For every lag the predictor "with a key of m tokens" knows only whether the last min(m, ·) tokens
   match at that lag; it is the Bayes predictor of the task given that information, so its loss is what a circuit of
   that kind reaches when it minimises the training loss. m = ∞ is the Bayes predictor of the task. The loss is
   reported on the predictable targets; a predictor that trusted every match more would do better there and worse
   on the other targets, so the loss on the other targets is given as well (a uniform guess costs log 64 there). The
   content-free references are in ``discussion_checks.py``, part B.

3. Longer training. The single-layout runs without the textbook circuit are trained on to 60,000 steps (five times
   pilot 1's budget; the learning rate is constant, so the first 12,000 steps are the same run), with the ablation
   table at 14 checkpoints: does the loss stop near 0.8, does it approach the m = 1 value, does the circuit change?

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/relay_offsets_pilot.py
--jobs 4`` (``--skip-runs`` analyses the cached runs only). Runs are cached in ``work/relay_offsets``. Writes
``docs/results/v2_analysis/relay_offsets_pilot.json``.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D_MAX, DELTAS = 40, tuple(range(-4, 32))
LONG_STEPS = 60_000
#: Runs trained on to LONG_STEPS, most informative first (the pool works through them in this order).
LONG = ("single_rm12_h2_s302", "single_rm12_h2_s300", "single_rm12_h2_s301", "single_rm12_h4_s302",
        "single_rm4_h2_s300", "single_rm4_h2_s302", "single_rm12_h4_s301", "single_rm12_h4_s300")
#: Key lengths of the ladder (None = the Bayes predictor of the task) and the settings it is computed for.
KEY_LENGTHS = (1, 2, 3, None)
LADDER_SETTINGS = {"single, repeat length 4..32": ("single", 4, 32), "single, repeat length 12..32": ("single", 12, 32),
                   "tiled, repeat length 4..16": ("tiled", 4, 16)}
VOCAB, SEQ_LEN = 64, 64


# ---------------------------------------------------------------------------------------------------------- runs
def jobs_list() -> list[dict]:
    from tycho.train.spec import TrainConfig

    fields = {f.name for f in dataclasses.fields(TrainConfig)}

    def num(v: str):
        if v == "":
            return None
        return int(float(v)) if float(v).is_integer() and "." not in v and "e" not in v else float(v)

    jobs = []
    with open(REPO / "releases" / "DR2b-pilot" / "objects.csv", newline="", encoding="utf-8") as fh:
        for o in csv.DictReader(fh):
            cfg = json.loads(o["task_cfg_json"])
            if int(cfg["n_layer"]) != 2 or float(cfg["p_repeat"]) != 1.0:
                continue
            train = {k[6:]: (o[k] if k[6:] in ("optimizer", "lr_schedule") else num(o[k]))
                     for k in o if k.startswith("train_") and k[6:] in fields}
            jobs.append({"name": f"single_rm{cfg['rep_min']}_h{cfg['n_head']}_s{o['seed']}", "cfg": cfg,
                         "seed": int(o["seed"]), "train": train, "sky_version": o["sky_version"],
                         "released_id": o["object_id"]})
    for seed in (310, 311, 312):
        cfg = {"vocab": VOCAB, "seq_len": SEQ_LEN, "rep_min": 4, "rep_max": 16, "layout": "tiled", "d_model": 96,
               "attn_only": True, "probe_size": 128, "n_head": 3, "n_layer": 2, "p_repeat": 1.0, "data_seed": seed}
        train = {"steps": 12000, "batch_size": 32, "optimizer": "adam", "lr": 3e-4, "n_log_evals": 80, "n_lin_evals": 0,
                 "ntk_every": 0}
        jobs.append({"name": f"tiled_h3_s{seed}", "cfg": cfg, "seed": seed, "train": train,
                     "sky_version": "pilot-relay-offsets", "released_id": None})
    return sorted(jobs, key=lambda j: j["name"])


def long_jobs() -> list[dict]:
    """The single-layout runs without the textbook circuit, trained five times longer (same seeds, constant learning
    rate: the first 12,000 steps are the same run), with the ablation table at 14 log-spaced checkpoints."""
    base = {j["name"]: j for j in jobs_list()}
    out = []
    for name in LONG:
        j = dict(base[name])
        j["train"] = {**j["train"], "steps": LONG_STEPS}
        j.update(name=f"long_{name}", base=name, n_points=14, sky_version="pilot-relay-offsets", released_id=None)
        out.append(j)
    return out


def run_one(args: tuple[dict, str]) -> str:
    job, work = args
    out = Path(work) / f"{job['name']}.npz"
    if out.exists():
        return f"cached {job['name']}"
    import torch

    from tycho.train.runner import run_object
    from tycho.train.spec import ObjectSpec, RIConfig, TrainConfig

    torch.set_num_threads(1)
    spec = ObjectSpec(task="induction", task_cfg=job["cfg"], seed=job["seed"],
                      train=TrainConfig.from_dict(job["train"]), sky_version=job["sky_version"])
    res = run_object(spec, work_dir=Path(work) / "resume" / job["name"],
                     ri=RIConfig(enabled=True, mode="patterns", n_probe=1024, n_points=int(job.get("n_points", 2))),
                     resume=False)
    if res.status != "done":
        return f"FAILED {job['name']}: {res.error}"
    pat, lc = res.patterns, res.light_curve
    arrays = {k: np.asarray(v) for k, v in pat.items() if k.startswith("param/")}
    arrays.update({f"lc_{k}": np.asarray(lc[k]) for k in lc if k.startswith(("x_prev", "x_ind", "x_dup"))})
    np.savez_compressed(out, logodds=np.asarray(pat["logodds"]), steps=np.asarray(pat["steps"]),
                        lc_step=np.asarray(lc["step"]), lc_loss_rep=np.asarray(lc["x_loss_rep"]),
                        object_id=np.asarray(spec.object_id()), **arrays)
    return f"done {job['name']} (id {spec.object_id()}; released {job['released_id']})"


# ------------------------------------------------------------------------------------------- the key-length ladder
def run_lengths(x: np.ndarray, lag: int) -> np.ndarray:
    """r[t] = the number of consecutive positions i = t, t − 1, … with x[i] == x[i − lag]."""
    r = np.zeros(len(x), dtype=np.int64)
    for t in range(lag, len(x)):
        r[t] = r[t - 1] + 1 if x[t] == x[t - lag] else 0
    return r


def key_ladder(layout: str, rep_min: int, rep_max: int, n_seq: int) -> dict:
    """Loss and mean probability of the correct token, on the predictable targets, of the predictors with a key of
    m tokens (module docstring), for m in KEY_LENGTHS; from ``n_seq`` sequences drawn by the task's own generator.

    A hypothesis is a repeat length L (and, in the single layout, a start s). Given a prefix x[0..t] it has n copied
    positions; the prefix has likelihood ∝ 64^n if they all match their sources and 0 otherwise. With a key of m
    tokens only the last min(n, m) positions of a copy that is still running are checked (and count); the posterior
    over hypotheses then predicts x[t + 1 − L] where the next position is a copy and a uniform token elsewhere.
    """
    from tycho.tasks.base import torch_generator
    from tycho.tasks.induction import Induction

    task = Induction({"vocab": VOCAB, "seq_len": SEQ_LEN, "rep_min": rep_min, "rep_max": rep_max, "layout": layout}, 0)
    x, rep, _ = task._draw(n_seq, torch_generator(12345, "induction/key-ladder"), 1.0)
    xs, rep = x.numpy(), rep.numpy()
    S, T = SEQ_LEN, SEQ_LEN - 1                         # inputs 0..T−1, targets 1..T
    lags = np.arange(rep_min, rep_max + 1)
    t = np.arange(T)
    probs = {m: [] for m in KEY_LENGTHS}
    others = {m: [] for m in KEY_LENGTHS}
    for i in range(n_seq):
        seq = xs[i]
        acc = {m: [np.zeros(T), np.zeros(T)] for m in KEY_LENGTHS}   # [mass on the true token, total mass]
        for L in lags:
            r = run_lengths(seq, L)
            hit = np.zeros(T, dtype=bool)                # would copying at lag L give the right next token?
            hit[L - 1:] = seq[np.arange(L - 1, T) + 1 - L] == seq[np.arange(L - 1, T) + 1]
            if layout == "tiled":
                n = np.clip(t - L + 1, 0, None)
                covers = t + 1 >= L
                for m in KEY_LENGTHS:
                    c = n if m is None else np.minimum(n, m)
                    w = np.where(r[:T] >= c, 64.0 ** np.minimum(c, 150), 0.0) / len(lags)
                    acc[m][0] += np.where(covers, w * hit, w / VOCAB)
                    acc[m][1] += w
                continue
            s = np.arange(0, S - 2 * L + 1)[:, None]
            prior = 1.0 / (len(lags) * (S - 2 * L + 1))
            n = np.clip(t[None, :] - s - L + 1, 0, L)
            running = (t[None, :] >= s + L) & (t[None, :] <= s + 2 * L - 1)
            ended = t[None, :] > s + 2 * L - 1
            covers = (t[None, :] + 1 >= s + L) & (t[None, :] + 1 <= s + 2 * L - 1)
            whole = r[s[:, 0] + 2 * L - 1][:, None] >= L  # the whole copy matches its source
            for m in KEY_LENGTHS:
                if m is None:
                    w = np.where(running, np.where(r[None, :T] >= n, 64.0 ** n, 0.0),
                                 np.where(ended, np.where(whole, 64.0 ** L, 0.0), 1.0)) * prior
                else:
                    c = np.minimum(n, m)
                    w = np.where(running, np.where(r[None, :T] >= c, 64.0 ** c, 0.0), 1.0) * prior
                a, b = (w * covers).sum(0), (w * ~covers).sum(0)
                acc[m][0] += a * hit + b / VOCAB
                acc[m][1] += a + b
        for m in KEY_LENGTHS:
            probs[m].append((acc[m][0] / acc[m][1])[rep[i]])
            others[m].append((acc[m][0] / acc[m][1])[~rep[i]])
    out = {"n_sequences": n_seq, "n_predictable_targets": int(rep.sum()), "n_other_targets": int((~rep).sum()),
           "loss_other_uniform": float(np.log(VOCAB)), "key_length": {}}
    for m in KEY_LENGTHS:
        p, q = np.concatenate(probs[m]), np.concatenate(others[m])
        out["key_length"]["inf" if m is None else str(m)] = {"loss": float(-np.log(p).mean()), "p": float(p.mean()),
                                                             "loss_other": float(-np.log(q).mean())}
    return out


# ------------------------------------------------------------------------------------------------------ analysis
def attention_summary(job: dict, work: Path) -> dict:
    """Attention offsets of the final network of one cached run, and its ablation table at the end."""
    import torch

    from tycho.tasks.induction import Induction

    with np.load(Path(work) / f"{job['name']}.npz") as f:
        state = {k[6:]: torch.as_tensor(f[k]) for k in f.files if k.startswith("param/")}
        p = 1.0 / (1.0 + np.exp(-f["logodds"][-1]))
        loss_rep = float(f["lc_loss_rep"][-1])
        object_id = str(f["object_id"])
    task = Induction(job["cfg"], job["seed"])
    model = task.build_model()
    model.load_state_dict(state)
    model.eval()
    pr = task._probe
    x, rep, src = pr["x"], pr["rep"].numpy(), pr["src"].numpy()
    maps = [a.numpy() for a in model.attention_maps(x)]          # per layer: (B, H, T, T)
    T = x.shape[1]
    nh = int(job["cfg"]["n_head"])
    out = {"name": job["name"], "object_id": object_id, "same_id_as_release": object_id == job["released_id"],
           "rep_min": int(job["cfg"]["rep_min"]), "rep_max": int(job["cfg"]["rep_max"]), "n_head": nh,
           "layout": job["cfg"].get("layout", "single"), "seed": job["seed"], "steps": int(job["train"]["steps"]),
           "loss_rep": loss_rep, "heads": []}
    b = np.arange(1 << (2 * nh))
    n0 = sum((b >> i) & 1 for i in range(nh))
    out["p_all"], out["p_best_with_one_layer0_head"] = float(p[-1]), float(p[n0 == 1].max())
    t_idx = np.arange(T)
    for h in range(nh):                                           # layer 0: attention by offset
        a = maps[0][:, h]
        prof = np.array([a[:, t_idx[d:], t_idx[d:] - d].mean() for d in range(D_MAX + 1)])
        top = np.argsort(-prof)[:3]
        out["heads"].append({"head": f"l0h{h}",
                             "top_offsets": [{"d": int(d), "attention": float(prof[d])} for d in top],
                             "attention_d0": float(prof[0]), "attention_d1": float(prof[1]),
                             "attention_d2": float(prof[2])})
    bi, ti = np.nonzero(rep)
    lag = ti - src[bi, ti] + 1                                    # the repeat length of each predictable position
    best = None
    for h in range(nh):                                           # layer 1: attention relative to the induction target
        a = maps[1][:, h]

        def profile(sel: np.ndarray) -> np.ndarray:
            prof = np.full(len(DELTAS), np.nan)
            for k, dl in enumerate(DELTAS):
                j = src[bi[sel], ti[sel]] + dl
                ok = (j >= 0) & (j <= ti[sel])
                if ok.any():
                    prof[k] = a[bi[sel][ok], ti[sel][ok], j[ok]].mean()
            return prof

        prof = profile(np.ones(len(bi), dtype=bool))
        top = np.argsort(-np.nan_to_num(prof))[:3]
        lags = sorted(set(lag.tolist()))
        by_lag = np.array([profile(lag == L) for L in lags])
        top_by_lag = [DELTAS[int(np.nanargmax(row))] for row in by_lag]
        modal, count = Counter(top_by_lag).most_common(1)[0]
        at_modal = by_lag[:, DELTAS.index(modal)]
        head = {"head": f"l1h{h}",
                "top_deltas": [{"delta": int(DELTAS[i]), "attention": float(prof[i])} for i in top],
                "attention_delta0": float(prof[DELTAS.index(0)]),
                "attention_to_own_position": float(a[bi, ti, ti].mean()),
                "by_repeat_length": {"L": lags, "top_delta": top_by_lag,
                                     "attention_at_top": [float(np.nanmax(row)) for row in by_lag]},
                "modal_top_delta": int(modal), "share_of_lengths_with_modal_delta": count / len(lags),
                "share_of_lengths_with_top_at_own_position": float(np.mean([d == L - 1 for d, L in
                                                                            zip(top_by_lag, lags)])),
                "median_attention_at_modal_delta": float(np.nanmedian(at_modal))}
        out["heads"].append(head)
        if best is None or head["median_attention_at_modal_delta"] > best["median_attention_at_modal_delta"]:
            best = head
    out["relay_delta"] = best["modal_top_delta"]                   # of the layer-1 head that attends most sharply
    return out


def long_summary(job: dict, work: Path) -> dict:
    """Loss, attention scores and ablation table over the long run, and the attention offsets at its end."""
    out = attention_summary(job, work)
    nh = out["n_head"]
    with np.load(Path(work) / f"{job['name']}.npz") as f:
        steps, loss = f["lc_step"].astype(float), f["lc_loss_rep"].astype(float)
        p = 1.0 / (1.0 + np.exp(-f["logodds"]))
        psteps = f["steps"].astype(float)
        prev = np.max([f[k] for k in f.files if k.startswith("lc_x_prev_l0")], axis=0)
        ind = np.max([f[k] for k in f.files if k.startswith("lc_x_ind_l1")], axis=0)
    b = np.arange(1 << (2 * nh))
    n0 = sum((b >> i) & 1 for i in range(nh))
    n1 = sum((b >> i) & 1 for i in range(nh, 2 * nh))
    at = lambda s: int(np.argmin(np.abs(steps - s)))               # noqa: E731
    keep = steps >= 1000
    out.update({
        "base": job["base"],
        "loss_rep_at": {str(s): float(loss[at(s)]) for s in (2_000, 6_000, 12_000, 24_000, 36_000, 48_000, 60_000)},
        "loss_rep_curve": {"step": [int(s) for s in steps[keep]], "loss_rep": [float(v) for v in loss[keep]]},
        "loss_rep_change_last_20_percent_of_steps": float(loss[-1] - loss[at(0.8 * steps[-1])]),
        "previous_token_score_max_layer0": {"at_12000": float(prev[at(12_000)]), "at_end": float(prev[-1])},
        "induction_score_max_layer1": {"at_12000": float(ind[at(12_000)]), "at_end": float(ind[-1])},
        "ablation": {"step": [int(s) for s in psteps], "p_all": [float(v) for v in p[:, -1]],
                     "p_best_with_one_layer0_head": [float(v) for v in p[:, n0 == 1].max(1)],
                     "p_best_with_two_layer0_heads_and_one_layer1_head":
                         [float(v) for v in p[:, (n0 == 2) & (n1 == 1)].max(1)]}})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_offsets")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ladder-sequences", type=int, default=1000)
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_offsets_pilot.json")
    args = ap.parse_args(argv)
    jobs, longs = jobs_list(), long_jobs()
    args.work.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(run_one, [(j, str(args.work)) for j in jobs + longs]):
                print(msg, flush=True)

    ladder = {label: key_ladder(layout, lo, hi, args.ladder_sequences)
              for label, (layout, lo, hi) in LADDER_SETTINGS.items()}
    runs = [attention_summary(j, args.work) for j in jobs]
    have = [j for j in longs if (args.work / f"{j['name']}.npz").exists()]
    long_runs = [long_summary(j, args.work) for j in have]
    result = {"status": "exploratory pilot; local re-runs, not bit-identical with the server runs; not independently "
                        "recomputed", "key_length_ladder": ladder, "runs": runs,
              "long_runs": {"steps": LONG_STEPS, "missing": [j["name"] for j in longs if j not in have],
                            "runs": long_runs}}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    print("key-length ladder (loss / mean p of the correct token, on the predictable targets)")
    for label, lad in ladder.items():
        print(f"  {label:>30s}: " + "   ".join(f"m={m}: {v['loss']:.3f} / {v['p']:.3f} (other {v['loss_other']:.3f})"
                                               for m, v in lad["key_length"].items()))

    def heads_line(r: dict) -> str:
        l0 = "; ".join(f"{hd['head']}: "
                       + ", ".join(f"d={t['d']} ({t['attention']:.2f})" for t in hd["top_offsets"][:2])
                       for hd in r["heads"] if hd["head"].startswith("l0"))
        l1 = "; ".join(f"{hd['head']}: δ={hd['modal_top_delta']} for "
                       f"{hd['share_of_lengths_with_modal_delta']:.0%} of L "
                       f"({hd['median_attention_at_modal_delta']:.2f}), own {hd['attention_to_own_position']:.2f}"
                       for hd in r["heads"] if hd["head"].startswith("l1"))
        return f"{l0} | {l1}"

    print("attention at the end of training")
    for r in runs:
        print(f"{r['name']:>22s} loss {r['loss_rep']:.2f} p_all {r['p_all']:.2f} one-L0 "
              f"{r['p_best_with_one_layer0_head']:.2f} relay δ={r['relay_delta']} | {heads_line(r)}")
    print(f"long runs ({LONG_STEPS} steps)")
    for r in long_runs:
        la = r["loss_rep_at"]
        print(f"{r['name']:>27s} loss at 12k/24k/36k/48k/60k: " + "/".join(f"{la[str(s)]:.2f}" for s in
                                                                           (12_000, 24_000, 36_000, 48_000, 60_000))
              + f" | p_all {r['p_all']:.2f} one-L0 {r['p_best_with_one_layer0_head']:.2f} relay δ={r['relay_delta']}"
              + f" | prev {r['previous_token_score_max_layer0']['at_end']:.2f}"
              + f" ind {r['induction_score_max_layer1']['at_end']:.2f} | {heads_line(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
