"""Post-hoc exploratory diagnostic (NOT pre-registered): MLP low-LR & modern-recipe sweep.

Suite B2 (``attn_only=False, mlp_ratio=4``, Adam, no warmup) showed that adding a 4x GELU MLP
pushed ``B2-low`` (1e-3/128) to 24/24 R and ``B2-mid/high`` (1e-3/32, 2e-3/32) into the
instability wall (22-24/24 U). This isolated exploratory diagnostic probes lower learning rates
(3e-4, 5e-4, 7e-4), bidirectional c levers (0.5, 2.0), and modern AdamW + 200-step warmup on
``B = 32``, saving strictly to ``work/relay_confirm_mlp_diag/`` so the pre-registered cache in
``work/relay_confirm/`` remains untouched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

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

DIAG_CELLS: list[rc.CellSpec] = [
    rc.CellSpec("MLP-lr3e-4", "B", "MLP-diag", "exploratory", lr=3e-4, batch_size=32, attn_only=False, mlp_ratio=4),
    rc.CellSpec("MLP-lr3e-4-c2", "B", "MLP-diag", "exploratory", lr=3e-4, batch_size=32, attn_only=False, mlp_ratio=4, out_scale={0: 2.0}),
    rc.CellSpec("MLP-lr5e-4", "B", "MLP-diag", "exploratory", lr=5e-4, batch_size=32, attn_only=False, mlp_ratio=4),
    rc.CellSpec("MLP-lr5e-4-c05", "B", "MLP-diag", "exploratory", lr=5e-4, batch_size=32, attn_only=False, mlp_ratio=4, out_scale={0: 0.5}),
    rc.CellSpec("MLP-lr7e-4", "B", "MLP-diag", "exploratory", lr=7e-4, batch_size=32, attn_only=False, mlp_ratio=4),
    rc.CellSpec("MLP-wu-lr1e-3", "B", "MLP-diag", "exploratory", lr=1e-3, batch_size=32, attn_only=False, mlp_ratio=4, optimizer="adamw", weight_decay=0.1, warmup_steps=200),
    rc.CellSpec("MLP-wu-lr1e-3-c05", "B", "MLP-diag", "exploratory", lr=1e-3, batch_size=32, attn_only=False, mlp_ratio=4, optimizer="adamw", weight_decay=0.1, warmup_steps=200, out_scale={0: 0.5}),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Post-hoc exploratory MLP low-LR diagnostic (7 cells, 168 runs).")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_confirm_mlp_diag")
    ap.add_argument("--jobs", type=int, default=86)
    ap.add_argument("--analyze-jobs", type=int, default=32)
    ap.add_argument("--out-json", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_mlp_diag.json")
    ap.add_argument("--out-md", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_mlp_diag.md")
    ap.add_argument("--skip-runs", action="store_true")
    args = ap.parse_args(argv)

    args.work.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for sp in DIAG_CELLS:
        for seed, nh in sp.run_pairs(topup=False):
            jobs.append(rc.make_job(sp, seed, nh))

    if not args.skip_runs:
        todo = [j for j in jobs if not (args.work / f"{j['name']}.npz").exists()]
        print(f"[MLP-DIAG] Selected={len(jobs)} | To run={len(todo)} | Workers={args.jobs}", flush=True)
        if todo:
            t0 = time.monotonic()
            done = 0
            with Pool(processes=max(1, args.jobs)) as pool:
                for msg in pool.imap_unordered(rc.run_one_job, [(j, str(args.work), False) for j in todo]):
                    done += 1
                    if done % 20 == 0 or done == len(todo) or done <= 5:
                        print(f"[{done:3d}/{len(todo):3d} | {(time.monotonic()-t0)/60:4.1f}m] {msg}", flush=True)
            print(f"[MLP-DIAG] Finished {done} runs in {(time.monotonic()-t0)/60:.1f} min.", flush=True)

    npz_files = sorted(p for p in args.work.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    print(f"[MLP-DIAG] Classifying {len(npz_files)} runs...", flush=True)
    with Pool(processes=min(max(1, args.analyze_jobs), max(1, len(npz_files)))) as pool:
        rows = list(pool.imap_unordered(an._classify_worker, [str(p) for p in npz_files]))
    rows.sort(key=lambda r: (r["cell_id"], r["seed"]))

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(r["cell_id"], []).append(r)

    cells = {cid: an.summarize_cell(by_cell[cid]) for cid in [sp.cell_id for sp in DIAG_CELLS] if cid in by_cell}
    comparisons = {
        "lr3e-4_vs_lr7e-4": an.paired_comparison("MLP-lr3e-4", "MLP-lr7e-4", by_cell),
        "lr3e-4_vs_lr5e-4": an.paired_comparison("MLP-lr3e-4", "MLP-lr5e-4", by_cell),
        "lever_c2_at_3e-4": an.paired_comparison("MLP-lr3e-4", "MLP-lr3e-4-c2", by_cell),
        "lever_c05_at_5e-4": an.paired_comparison("MLP-lr5e-4-c05", "MLP-lr5e-4", by_cell),
        "lever_c05_at_wu1e-3": an.paired_comparison("MLP-wu-lr1e-3-c05", "MLP-wu-lr1e-3", by_cell),
    }
    res = {"exploratory": True, "n_runs": len(rows), "cells": cells, "comparisons": comparisons, "runs": rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Post-hoc Exploratory MLP Diagnostic (4x GELU MLP, d=64, B=32, n=24/cell)",
        "",
        "> **Exploratory post-hoc diagnostic** (not pre-registered; cached separately in `work/relay_confirm_mlp_diag/`).",
        "",
        "| Cell | n | U | R | T | S | M | T+S | T+S+M | h2 T+S | h4 T+S | R_1 med |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cid, cs in cells.items():
        c = cs["counts"]
        h2 = cs["by_head"].get("h2", {})
        h4 = cs["by_head"].get("h4", {})
        h2_ts = h2.get("T", 0) + h2.get("S", 0)
        h4_ts = h4.get("T", 0) + h4.get("S", 0)
        lines.append(
            f"| {cid} | {cs['n']} | {c['U']} | {c['R']} | {c['T']} | {c['S']} | {c['M']} | "
            f"{cs['source_strict']} | {cs['source_inclusive']} | {h2_ts}/12 | {h4_ts}/12 | {cs['r1_aligned']['median']} |"
        )
    lines.extend(["", "## Paired Comparisons (T+S)", ""])
    for name, cmp_d in comparisons.items():
        if cmp_d.get("n", 0):
            ms = cmp_d["by_mode"]["source_strict"]
            lines.append(f"- **{name}** (`{cmp_d['x']}` vs `{cmp_d['y']}`): `{ms['kx']} vs {ms['ky']}` (**{ms['verdict']}**, McNemar `p={ms['mcnemar_p']}`)")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
