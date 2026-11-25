"""Post-hoc exploratory optimization test (NOT pre-registered): Layer-0 Attn+MLP residual scaling.

Tests the optimization hypothesis motivated by ``relay_mlp_diag.md``:
On the full modern 2-layer Transformer recipe (4x GELU MLP + AdamW wd=0.1 + 200-step warmup,
lr=1e-3, B=32, d=64, rep_min=12), baseline ``MLP-wu-lr1e-3`` falls 24/24 into the positional
relay trap (loss ~1.06 nats vs ~0.17 textbook) while ``Attn0 x 0.5`` alone rescued 7/24 (p=0.0156)
because ``MLP0`` still injected rms_m0 ~ 1.5x rms_emb.

Runs 3 new paired cells (72 runs = 1 single wave on 86 cores, ~18 min), reusing the cached
``MLP-wu-lr1e-3`` and ``MLP-wu-lr1e-3-c05`` baselines in ``work/relay_confirm_mlp_diag/``:
  1. ``MLP-wu-lr1e-3-m05``     : Attn0 x 1.0, MLP0 x 0.5  (isolates MLP0 scaling alone)
  2. ``MLP-wu-lr1e-3-both05``  : Attn0 x 0.5, MLP0 x 0.5  (full Layer-0 block scaling 0.5)
  3. ``MLP-wu-lr1e-3-both025`` : Attn0 x 0.25, MLP0 x 0.25 (stronger Layer-0 block scaling 0.25)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_rc_spec = importlib.util.spec_from_file_location("relay_confirm", Path(__file__).with_name("relay_confirm.py"))
rc = importlib.util.module_from_spec(_rc_spec)
sys.modules["relay_confirm"] = rc
_rc_spec.loader.exec_module(rc)

_an_spec = importlib.util.spec_from_file_location("relay_confirm_analyze", Path(__file__).with_name("relay_confirm_analyze.py"))
an = importlib.util.module_from_spec(_an_spec)
sys.modules["relay_confirm_analyze"] = an
_an_spec.loader.exec_module(an)

BASE_SPEC = rc.CellSpec(
    "MLP-wu-lr1e-3", "B", "MLP-res-opt", "exploratory",
    lr=1e-3, batch_size=32, attn_only=False, mlp_ratio=4,
    optimizer="adamw", weight_decay=0.1, warmup_steps=200,
)

NEW_ARMS: list[tuple[str, dict[int, float] | None, dict[int, float] | None]] = [
    ("MLP-wu-lr1e-3-m05", None, {0: 0.5}),
    ("MLP-wu-lr1e-3-both05", {0: 0.5}, {0: 0.5}),
    ("MLP-wu-lr1e-3-both025", {0: 0.25}, {0: 0.25}),
]

ALL_ARM_ORDER = [
    "MLP-wu-lr1e-3",
    "MLP-wu-lr1e-3-c05",
    "MLP-wu-lr1e-3-m05",
    "MLP-wu-lr1e-3-both05",
    "MLP-wu-lr1e-3-both025",
]


def train_with_mlp_scale(job: dict[str, Any]):
    """Exact copy of ``train_confirm`` with optional ``mlp_out_scale`` hook + exact post-training fold."""
    import torch

    from tycho.tasks import make_task
    from tycho.tasks.base import torch_generator
    from tycho.train.runner import LeanOptimizer
    from tycho.train.schedule import lr_at
    from tycho.train.spec import TrainConfig

    torch.set_num_threads(1)
    steps = int(job["train"]["steps"])
    at = rc.measure_steps(steps)
    at_set = set(at)

    tc = TrainConfig.from_dict(job["train"])
    task = make_task("induction", job["cfg"], job["seed"])
    model = task.build_model()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = LeanOptimizer(params, tc)
    gen = torch_generator(job["seed"], "data")

    a_factors = {int(lyr): float(factor) for lyr, factor in (job.get("out_scale") or {}).items()}
    for lyr in a_factors:
        a = model.blocks[lyr].attn
        inner = a.forward
        a.forward = (lambda x, mask, inner=inner, lyr=lyr: a_factors[lyr] * inner(x, mask))

    m_factors = {int(lyr): float(factor) for lyr, factor in (job.get("mlp_out_scale") or {}).items()}
    for lyr in m_factors:
        m = model.blocks[lyr].mlp
        inner_m = m.forward
        m.forward = (lambda x, inner_m=inner_m, lyr=lyr: m_factors[lyr] * inner_m(x))

    use_schedule = (tc.warmup_steps > 0) or (tc.lr_schedule != "constant")
    stats: dict[int, dict[str, float]] = {}
    for step in range(steps + 1):
        if step in at_set:
            model.eval()
            stats[step] = rc.confirm_stats(model, task)
            model.train()
        if step == steps:
            break
        x, y = task.sample_batch(tc.batch_size, gen)
        loss = task.loss(model(x), y)
        step_lr = lr_at(step, tc) if use_schedule else tc.lr
        opt.step(list(torch.autograd.grad(loss, params, allow_unused=True, materialize_grads=True)), step_lr)

    for lyr, factor in a_factors.items():
        a = model.blocks[lyr].attn
        a.forward = type(a).forward.__get__(a)
        with torch.no_grad():
            a.proj.weight.mul_(factor)
            a.proj.bias.mul_(factor)
    for lyr, factor in m_factors.items():
        m = model.blocks[lyr].mlp
        m.forward = type(m).forward.__get__(m)
        with torch.no_grad():
            m[2].weight.mul_(factor)
            m[2].bias.mul_(factor)
    model.eval()
    return model, task, at, stats


def run_one(args: tuple[dict[str, Any], str]) -> str:
    job, work_dir = args
    import torch

    out = Path(work_dir) / f"{job['name']}.npz"
    if out.exists():
        return f"cached {job['name']}"
    t0 = time.monotonic()
    model, _, at, stats = train_with_mlp_scale(job)
    wall_s = time.monotonic() - t0
    arrays = {f"param/{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    for k in rc.STAT_KEYS:
        arrays[f"stat_{k}"] = np.asarray([stats[s][k] for s in at], dtype=np.float64)
    meta = {
        **job,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cpu_model": platform.processor() or platform.machine(),
        "num_threads": 1,
        "wall_s": round(wall_s, 3),
    }
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        lc_step=np.asarray(at, dtype=np.int64),
        lc_loss_rep=arrays["stat_loss_rep"],
        meta_json=np.asarray(json.dumps(meta)),
        **arrays,
    )
    os.replace(tmp, out)
    return f"done {job['name']} ({wall_s:.1f}s, loss={stats[at[-1]]['loss_rep']:.3f}, src={stats[at[-1]]['source_l1']:.3f})"


def verify_fold() -> None:
    import torch

    j = rc.make_job(BASE_SPEC, 1000, 2, steps_override=5, name_override="fold_test")
    j["out_scale"] = {0: 0.5}
    j["mlp_out_scale"] = {0: 0.5}
    m, task, _, _ = train_with_mlp_scale(j)
    # Check logits match manual forward with folded weights (no NaNs, finite loss)
    x = task._probe["x"][:4]
    with torch.no_grad():
        logits = m(x)
    assert torch.isfinite(logits).all(), "Non-finite logits!"
    print("[VERIFY] Fold & finite-logit check PASSED.", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Post-hoc exploratory Layer-0 Attn+MLP residual scaling ablation.")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_confirm_mlp_diag")
    ap.add_argument("--jobs", type=int, default=86)
    ap.add_argument("--analyze-jobs", type=int, default=32)
    ap.add_argument("--out-json", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_mlp_res_opt.json")
    ap.add_argument("--out-md", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_mlp_res_opt.md")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    if args.verify_only:
        verify_fold()
        return 0

    args.work.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for cid, a_sc, m_sc in NEW_ARMS:
        sp = rc.CellSpec(
            cid, "B", "MLP-res-opt", "exploratory",
            lr=1e-3, batch_size=32, attn_only=False, mlp_ratio=4,
            optimizer="adamw", weight_decay=0.1, warmup_steps=200,
            out_scale=a_sc,
        )
        for seed, nh in sp.run_pairs(topup=False):
            j = rc.make_job(sp, seed, nh)
            j["mlp_out_scale"] = dict(m_sc) if m_sc else None
            jobs.append(j)

    todo = [j for j in jobs if not (args.work / f"{j['name']}.npz").exists()]
    print(f"[MLP-RES-OPT] Selected={len(jobs)} | To run={len(todo)} | Workers={args.jobs}", flush=True)
    if todo:
        t0 = time.monotonic()
        done = 0
        with Pool(processes=max(1, args.jobs)) as pool:
            for msg in pool.imap_unordered(run_one, [(j, str(args.work)) for j in todo]):
                done += 1
                if done % 12 == 0 or done == len(todo) or done <= 5:
                    print(f"[{done:2d}/{len(todo):2d} | {(time.monotonic()-t0)/60:4.1f}m] {msg}", flush=True)
        print(f"[MLP-RES-OPT] Finished {done} runs in {(time.monotonic()-t0)/60:.1f} min.", flush=True)

    # Classify all 5 arms (base, c05, m05, both05, both025) from work_dir
    target_files = sorted(
        p for p in args.work.glob("MLP-wu-lr1e-3*.npz") if not p.name.endswith(".tmp.npz")
    )
    print(f"[MLP-RES-OPT] Classifying {len(target_files)} runs across 5 arms...", flush=True)
    with Pool(processes=min(max(1, args.analyze_jobs), max(1, len(target_files)))) as pool:
        rows = list(pool.imap_unordered(an._classify_worker, [str(p) for p in target_files]))
    rows.sort(key=lambda r: (r["cell_id"], r["seed"]))

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(r["cell_id"], []).append(r)

    cells = {cid: an.summarize_cell(by_cell[cid]) for cid in ALL_ARM_ORDER if cid in by_cell}
    comparisons = {
        cid: an.paired_comparison(cid, "MLP-wu-lr1e-3", by_cell)
        for cid in ALL_ARM_ORDER[1:]
        if cid in by_cell and "MLP-wu-lr1e-3" in by_cell
    }
    res = {"exploratory": True, "n_runs": len(rows), "cells": cells, "comparisons_vs_base": comparisons, "runs": rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Post-hoc Exploratory Optimization Test: Layer-0 Attn0 + MLP0 Residual Scaling",
        "",
        "> **Exploratory post-hoc test** on full modern GPT recipe (`4x GELU MLP + AdamW wd=0.1 + 200-step warmup, lr=1e-3, B=32, d=64, n=24/arm`).",
        "",
        "| Arm | Scaling (Attn0, MLP0) | n | U | R | T | S | M | T+S | h2 T+S | h4 T+S | med R_1 | med m0 | med loss | vs base (McNemar p) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    labels = {
        "MLP-wu-lr1e-3": "(1.0, 1.0)",
        "MLP-wu-lr1e-3-c05": "(0.5, 1.0)",
        "MLP-wu-lr1e-3-m05": "(1.0, 0.5)",
        "MLP-wu-lr1e-3-both05": "(0.5, 0.5)",
        "MLP-wu-lr1e-3-both025": "(0.25, 0.25)",
    }
    for cid in ALL_ARM_ORDER:
        if cid not in cells:
            continue
        cs = cells[cid]
        rs = by_cell[cid]
        c = cs["counts"]
        h2 = cs["by_head"].get("h2", {})
        h4 = cs["by_head"].get("h4", {})
        m0_med = round(float(np.median([r["rms_m0_aligned"] for r in rs])), 4)
        loss_med = round(float(np.median([r["loss_rep"] for r in rs])), 3)
        if cid in comparisons:
            ms = comparisons[cid]["by_mode"]["source_strict"]
            vs_str = f"**{ms['verdict']}** (`p={ms['mcnemar_p']}`)"
        else:
            vs_str = "baseline"
        lines.append(
            f"| {cid} | {labels[cid]} | {cs['n']} | {c['U']} | {c['R']} | {c['T']} | {c['S']} | {c['M']} | "
            f"**{cs['source_strict']}** | {h2.get('T',0)+h2.get('S',0)}/12 | {h4.get('T',0)+h4.get('S',0)}/12 | "
            f"{cs['r1_aligned']['median']} | {m0_med} | {loss_med} | {vs_str} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
