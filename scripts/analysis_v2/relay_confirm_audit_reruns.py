"""Independent audit of the relay confirmation study, the part that needs torch: implementation tests and re-runs.

Companion of ``relay_confirm_audit.py`` (which recomputes the verdicts from the committed rows). The saved weights of
the confirmation runs are on the server, so this script checks the code paths directly and re-trains a sample of runs
on another machine:

* ``tests``: the output factor on a layer's attention and on its MLP (the function during training against the model
  after the factor is folded into the weights, and against the model reloaded from its saved weights), the rotary
  position embedding of ``relay_stress_tests.py`` (against a complex-number implementation; the attention score must
  depend on the offset only; the position embedding must stay zero), the learning-rate and initial-value factors on
  layer 0's output projection, the warm-up schedule, and whether the runner builds the same run as the exploratory
  pilot for the same seed (weights after 400 steps).
* ``pilot``: the frozen classifier, end to end, on the weights of exploratory pilot runs in the local cache
  (``work/relay_band``), against the pilot's own committed analysis.
* ``rerun``: re-train named runs with the committed code. A job is ``frozen:CELL:SEED:HEADS`` (the pre-registered
  runner; any seed), ``resopt:ARM:SEED:HEADS`` (``relay_mlp_res_opt.py``) or ``stress:CELL:SEED:HEADS``
  (``relay_stress_tests.py``).
* ``compare``: classify the re-trained runs with the committed classifiers and compare them, run by run, with the
  committed rows. Writes ``docs/results/v2_analysis/relay_confirm_audit_reruns.json``.

Usage, from the repository root (needs torch):
``PYTHONPATH=. python scripts/analysis_v2/relay_confirm_audit_reruns.py tests``;
``... rerun --jobs 4 frozen:A1-6:1000:2 stress:ROPE-attn-mid:1000:2``; ``... compare``.
Re-trained runs are cached in ``work/relay_confirm_audit``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "results" / "v2_analysis"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = load("relay_confirm")
an = load("relay_confirm_analyze")
ro = load("relay_mlp_res_opt")
st = load("relay_stress_tests")
KINDS = {"frozen": "relay_confirm_results.json", "resopt": "relay_mlp_res_opt.json",
         "stress": ("relay_stress_rope.json", "relay_stress_depth.json")}


# ---------------------------------------------------------------------------------------------------- tests
def run_tests() -> dict:
    import torch

    from tycho.tasks import make_task
    from tycho.train.schedule import lr_at
    from tycho.train.spec import TrainConfig

    torch.set_num_threads(1)
    out: dict = {}
    # 1. the factor on the attention output: statistics with the factor active against the folded model
    fold = {}
    for cid, nh in (("A3-1", 2), ("A6-1", 2), ("B2-mid-c05", 4), ("B1-WU-low-c2", 2)):
        job = rc.make_job(rc.CELL_BY_ID[cid], 1000 if nh == 2 else 1012, nh, steps_override=30)
        model, task, _, stats = rc.train_confirm(job, measure_at=(0, 30))
        after = rc.confirm_stats(model, task)
        fold[cid] = max(abs(after[k] - stats[30][k]) for k in stats[30])
    out["attention_factor_hooked_against_folded"] = fold
    # 2. the factor on the MLP output (post hoc script)
    job = rc.make_job(ro.BASE_SPEC, 1000, 2, steps_override=30)
    job["out_scale"], job["mlp_out_scale"] = {0: 0.25}, {0: 0.25}
    model, task, at, stats = ro.train_with_mlp_scale(job)
    after = rc.confirm_stats(model, task)
    zero = rc.make_job(ro.BASE_SPEC, 1000, 2, steps_override=0)
    plain = ro.train_with_mlp_scale({**zero, "out_scale": None, "mlp_out_scale": None})[3][0]
    scaled = ro.train_with_mlp_scale({**zero, "out_scale": {0: 0.25}, "mlp_out_scale": {0: 0.25}})[3][0]
    out["mlp_factor"] = {"hooked_against_folded": max(abs(after[k] - stats[at[-1]][k]) for k in stats[at[-1]]),
                         "at_step_0_ratio_of_layer_0_mlp_output": scaled["rms_m0"] / plain["rms_m0"],
                         "at_step_0_ratio_of_layer_0_attention_output": scaled["rms_y0"] / plain["rms_y0"],
                         "at_step_0_ratio_of_layer_1_mlp_output": scaled["rms_m1"] / plain["rms_m1"]}
    # 3. rotary position embedding
    seq, heads, d_head = 64, 4, 16
    cos, sin = st.rope_cos_sin(seq, d_head)
    x = torch.randn(3, heads, seq, d_head, generator=torch.Generator().manual_seed(0))
    theta = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, dtype=torch.float64) / d_head))
    angle = torch.outer(torch.arange(seq, dtype=torch.float64), theta)
    z = torch.complex(x[..., 0::2].double(), x[..., 1::2].double()) * torch.polar(torch.ones_like(angle), angle)[None, None]
    reference = torch.stack((z.real, z.imag), dim=-1).flatten(-2)
    g = torch.Generator().manual_seed(1)
    q0, k0 = torch.randn(1, heads, 1, d_head, generator=g), torch.randn(1, heads, 1, d_head, generator=g)
    q = st.apply_rope(q0.expand(1, heads, seq, d_head).contiguous(), cos, sin)
    k = st.apply_rope(k0.expand(1, heads, seq, d_head).contiguous(), cos, sin)
    score = q @ k.transpose(-2, -1)                      # the content is fixed: the score may depend on the offset only
    spread = max(float((torch.diagonal(score, offset=-d, dim1=-2, dim2=-1)
                        - torch.diagonal(score, offset=-d, dim1=-2, dim2=-1)[..., :1]).abs().max()) for d in range(40))
    model = st.train_stress(st.make_stress_job(st.CELL_MAP["ROPE-attn-mid"], 1000, 2, steps_override=20))[0]
    out["rotary"] = {"largest_difference_from_a_complex_number_implementation": float((st.apply_rope(x, cos, sin).double() - reference).abs().max()),
                     "largest_variation_of_the_score_along_one_offset": spread,
                     "position_embedding_after_20_steps": float(model.pos.weight.abs().max()),
                     "position_embedding_is_trained": bool(model.pos.weight.requires_grad)}
    saved = {}
    for cid in ("ROPE-mlp-wu-both05", "L4-rope-mlp-wu-l01_05", "L4-abs-mlp-wu-l01_05"):
        job = st.make_stress_job(st.CELL_MAP[cid], 1000, 2, steps_override=50)
        model, task, at, stats = st.train_stress(job)
        folded = st.stress_stats(model, task)
        again = task.build_model()
        again.load_state_dict(model.state_dict())
        if job["use_rope"]:
            st.patch_model_for_rope(again)
        again.eval()
        reloaded = st.stress_stats(again, task)
        saved[cid] = {"hooked_against_folded": max(abs(folded[k] - stats[at[-1]][k]) for k in folded),
                      "folded_against_reloaded": max(abs(reloaded[k] - folded[k]) for k in folded)}
    out["stress_runs_as_saved"] = saved
    # 4. layer 0's output projection: learning-rate factor (A6-7) and initial-value factor (A6-8)
    task = make_task("induction", rc.make_job(rc.CELL_BY_ID["A1-1"], 1000, 2)["cfg"], 1000)
    start = task.build_model().state_dict()
    one = lambda cid, steps: rc.train_confirm(rc.make_job(rc.CELL_BY_ID[cid], 1000, 2, steps_override=steps), measure_at=(0,))[0].state_dict()
    base, slow, small = one("A1-1", 1), one("A6-7", 1), one("A6-8", 0)
    key = "blocks.0.attn.proj.weight"
    others = [k for k in start if k != key and not k.startswith("blocks.0.attn.proj")]
    out["projection_options"] = {
        "first_update_of_the_projection_A6-7_over_A1-1": float((slow[key] - start[key]).abs().sum() / (base[key] - start[key]).abs().sum()),
        "largest_difference_in_every_other_weight_A6-7_against_A1-1": max(float((slow[k] - base[k]).abs().max()) for k in others),
        "initial_projection_A6-8_over_default": float(small[key].abs().sum() / start[key].abs().sum()),
        "largest_difference_in_every_other_initial_weight": max(float((small[k] - start[k]).abs().max()) for k in others)}
    # 5. schedules
    tc = TrainConfig.from_dict(rc.make_job(rc.CELL_BY_ID["B1-WU-mid"], 1000, 2)["train"])
    out["warm_up"] = {"optimizer": tc.optimizer, "weight_decay": tc.weight_decay,
                      "lr_at_steps_0_99_199_200": [lr_at(s, tc) for s in (0, 99, 199, 200)]}
    # 6. the runner against the exploratory pilot, same seed
    pp = load("relay_period_pilot")
    same = {}
    for arm, cid, idx in (("base_natural", "A1-6", 0), ("lr5e-4_natural", "A1-1", 4), ("rm20_lr2e-3_natural", "A2-4", 1)):
        old = pp.arm_jobs(arm)[idx]
        new = rc.make_job(rc.CELL_BY_ID[cid], old["seed"], old["cfg"]["n_head"], steps_override=400)
        m_old = pp.rb.train({**old, "train": {**old["train"], "steps": 400}}, 400, (0, 400), block_from=0, block_until=0,
                            layer=0, forbid="band")[0].state_dict()
        m_new = rc.train_confirm(new, measure_at=(0, 400))[0].state_dict()
        same[f"{old['name']} against {cid}"] = max(float((m_old[k] - m_new[k]).abs().max()) for k in m_old)
    out["runner_against_pilot_after_400_steps"] = same
    return out


# ------------------------------------------------------------------------------- the classifier on pilot weights
def classifier_on_pilot(cache: Path, limit: int) -> dict:
    pp = load("relay_period_pilot")
    pilot = json.loads((RESULTS / "relay_period_pilot.json").read_text())["runs"]

    def pilot_category(r: dict) -> str:
        if not r["formed"]:
            return "U"
        if r["textbook"]:
            return "T"
        if r["relay_delta"] != 0:
            return "R"
        return "S" if r["source_l1_at_the_end"] > 0.5 else "M"

    table, n, worst = Counter(), 0, {"decoy": 0.0, "loss": 0.0, "offset": 0}
    tmp = Path(tempfile.mkdtemp()) / "run.npz"
    for arm, rows in pilot.items():
        jobs = {j["name"]: j for j in pp.arm_jobs(arm)}
        for r in rows:
            f = cache / f"period_{r['name']}.npz"
            if n >= limit or not f.exists():
                continue
            job = jobs[r["name"]]
            with np.load(f) as z:
                arrays = {k: z[k] for k in z.files if k.startswith(("param/", "stat_")) or k == "lc_step"}
            for k in rc.STAT_KEYS:
                arrays.setdefault(f"stat_{k}", np.zeros(len(arrays["lc_step"])))
            meta = {"name": job["name"], "cell_id": arm, "suite": "pilot", "group": "pilot", "seed": job["seed"],
                    "cfg": job["cfg"], "train": {"lr": job["train"]["lr"], "batch_size": job["train"]["batch_size"]}}
            np.savez(tmp, meta_json=np.asarray(json.dumps(meta)), **arrays)
            row = an.classify_npz(tmp)
            n += 1
            table[f"{pilot_category(r)}->{row['cat']}"] += 1
            worst["decoy"] = max(worst["decoy"], abs(row["deep_decoy_over_clean"] - r["deep_decoy_over_clean"]))
            worst["loss"] = max(worst["loss"], abs(row["loss_rep"] - r["loss_rep"]))
            worst["offset"] += row["relay_delta"] != r["relay_delta"]
    return {"runs": n, "pilot_category_to_frozen_category": dict(sorted(table.items())),
            "largest_difference_of_the_decoy_ratio": worst["decoy"], "largest_difference_of_the_loss": worst["loss"],
            "runs_with_another_relay_offset": worst["offset"]}


# --------------------------------------------------------------------------------------------------- re-runs
def rerun_one(args: tuple[str, str]) -> str:
    item, work = args
    kind, cid, seed, heads = item.split(":")
    seed, heads = int(seed), int(heads)
    out = Path(work) / kind
    out.mkdir(parents=True, exist_ok=True)
    if kind == "frozen":
        return rc.run_one_job((rc.make_job(rc.CELL_BY_ID[cid], seed, heads), str(out), False))
    if kind == "resopt":
        if cid == ro.BASE_SPEC.cell_id:                  # the baseline arm was run with the pre-registered runner
            return rc.run_one_job((rc.make_job(ro.BASE_SPEC, seed, heads), str(out), False))
        a_scale, m_scale = {c: (a, m) for c, a, m in ro.NEW_ARMS}[cid]
        spec = rc.CellSpec(cid, "B", "MLP-res-opt", "exploratory", lr=1e-3, batch_size=32, attn_only=False, mlp_ratio=4,
                           optimizer="adamw", weight_decay=0.1, warmup_steps=200, out_scale=a_scale)
        job = rc.make_job(spec, seed, heads)
        job["mlp_out_scale"] = dict(m_scale) if m_scale else None
        return ro.run_one((job, str(out)))
    if kind == "stress":
        return st.run_one((st.make_stress_job(st.CELL_MAP[cid], seed, heads), str(out)))
    raise ValueError(item)


def compare(work: Path) -> dict:
    out: dict = {}
    for kind, names in KINDS.items():
        folder = work / kind
        if not folder.exists():
            continue
        committed = {}
        for name in ([names] if isinstance(names, str) else names):
            committed.update({r["name"]: r for r in json.loads((RESULTS / name).read_text())["runs"]})
        classify = st.classify_stress_npz if kind == "stress" else an.classify_npz
        rows = []
        for f in sorted(p for p in folder.glob("*.npz") if not p.name.endswith(".tmp.npz")):
            row = classify(f)
            ref = committed.get(row["name"])
            entry = {"run": row["name"], "here": row["cat"], "loss_here": row["loss_rep"],
                     "decoy_here": row["deep_decoy_over_clean"], "relay_offset_here": row["relay_delta"]}
            if ref is not None:
                entry.update({"committed": ref["cat"], "same_category": row["cat"] == ref["cat"],
                              "same_relay_offset": row["relay_delta"] == ref["relay_delta"],
                              "loss_difference": abs(row["loss_rep"] - ref["loss_rep"]),
                              "same_steps": all(row.get(k) == ref.get(k) for k in ("t0_band85", "t1_band85", "source_step", "second_key_step") if k in ref)})
            rows.append(entry)
        both = [r for r in rows if "committed" in r]
        out[kind] = {"compared": len(both), "same_category": sum(r["same_category"] for r in both),
                     "same_relay_offset": sum(r["same_relay_offset"] for r in both),
                     "largest_loss_difference_where_the_category_agrees": max((r["loss_difference"] for r in both if r["same_category"]), default=None),
                     "runs": rows}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("tests", "pilot", "rerun", "compare"))
    ap.add_argument("items", nargs="*", help="for rerun: frozen:CELL:SEED:HEADS, resopt:ARM:SEED:HEADS, stress:CELL:SEED:HEADS")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_confirm_audit")
    ap.add_argument("--pilot-cache", type=Path, default=REPO / "work" / "relay_band")
    ap.add_argument("--limit", type=int, default=120, help="for pilot: the number of pilot runs to classify")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=RESULTS / "relay_confirm_audit_reruns.json")
    args = ap.parse_args(argv)
    if args.what == "rerun":
        args.work.mkdir(parents=True, exist_ok=True)
        with Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(rerun_one, [(item, str(args.work)) for item in args.items], chunksize=1):
                print(msg, flush=True)
        return 0
    result = json.loads(args.out.read_text()) if args.out.exists() else {
        "status": "audit re-runs on another machine; a sample, not a recomputation of the study"}
    if args.what == "tests":
        result["tests"] = run_tests()
        print(json.dumps(result["tests"], indent=1))
    elif args.what == "pilot":
        result["frozen_classifier_on_pilot_weights"] = classifier_on_pilot(args.pilot_cache, args.limit)
        print(json.dumps(result["frozen_classifier_on_pilot_weights"], indent=1))
    else:
        import torch

        result["reruns"] = {"torch_version": torch.__version__, **compare(args.work)}
        for kind, v in result["reruns"].items():
            if isinstance(v, dict):
                print(f"{kind}: compared {v['compared']}, same category {v['same_category']}, same relay offset {v['same_relay_offset']}")
                for r in v["runs"]:
                    print("   ", r["run"], "here", r["here"], "| committed", r.get("committed", "-"),
                          f"| loss {r['loss_here']:.3f}" + (f" (difference {r['loss_difference']:.1e})" if "loss_difference" in r else ""))
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
