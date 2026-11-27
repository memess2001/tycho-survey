"""Crash-proof multi-arm Stage 2 suite runner & live cross-arm scorecard generator.

Designed for Colab G4 (RTX PRO 6000 96GB) or lab RTX PRO 6000:
  - Runs the 5 pre-registered arms in priority order:
      post_olmo2 -> post_matched_gamma -> post_matched_emb -> post_frozen_gamma -> pre_ln
  - Automatically skips completed arms (DONE sentinel) and resumes interrupted runs (--resume)
  - Mirrors metrics.jsonl, run_meta.json, DONE, and rolling latest.pt to --sync-root (Google Drive)
  - Prints a clean live comparison table & writes summary_scorecard.json after every run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_ARMS = [
    "post_olmo2",
    "post_matched_gamma",
    "post_matched_emb",
    "post_frozen_gamma",
    "pre_ln",
]


def summarize_run(metrics_path: Path) -> Optional[dict]:
    if not metrics_path.exists():
        return None
    rows = [json.loads(ln) for ln in metrics_path.read_text().splitlines() if ln.strip()]
    if not rows:
        return None
    post0 = [r for r in rows if r["step"] > 0] or rows
    mid_idx = len(rows) // 2
    # Early causal depth: around ~10-25% of training (where trap is deepest in Arm 2)
    early_rows = [r for r in rows if 0.08 * rows[-1]["step"] <= r["step"] <= 0.30 * rows[-1]["step"]] or post0[:1]
    return {
        "n_evals": len(rows),
        "last_step": rows[-1]["step"],
        "last_tokens_M": round(rows[-1]["tokens"] / 1e6, 1),
        "wall_min": round(rows[-1]["wall_s"] / 60.0, 2),
        "val_bpb_mid": rows[mid_idx]["val_bpb"],
        "val_bpb_end": rows[-1]["val_bpb"],
        "R0_start": rows[0]["R_0"],
        "R0_end": rows[-1]["R_0"],
        "peak_dA0": max(r["dA0"] for r in post0),
        "end_dA0": rows[-1]["dA0"],
        "min_cosA": min(r["min_cosA_stream"] for r in post0),
        "peak_E_sL0": max(r["E_same_L0"] for r in post0),
        "early_causD": round(sum(r["causal_depth"] for r in early_rows) / len(early_rows), 3),
        "end_causD": rows[-1]["causal_depth"],
        "max_outI": max(r["max_outlier_int"] for r in post0),
        "ic_all_pass": all(bool(r.get("ic_v3_2_pass", True)) for r in rows),
    }


def print_scorecard(out_root: Path, sync_root: Optional[Path], arms: List[str], seeds: List[int]) -> None:
    summary: Dict[str, dict] = {}
    print("\n" + "=" * 124)
    print(
        f"{'Run':24s} | {'Step':>6s} | {'Min':>5s} | {'bpb(end)':>8s} | {'R0 0->end':>12s} | "
        f"{'peak dA0':>8s} | {'end dA0':>7s} | {'minCosA':>7s} | {'pk E_sL0':>8s} | {'causD e->f':>11s} | {'maxOutI':>7s} | IC"
    )
    print("-" * 124)
    for seed in seeds:
        for arm in arms:
            tag = f"{arm}_s{seed}"
            mp = out_root / tag / "metrics.jsonl"
            if not mp.exists() and sync_root is not None:
                smp = sync_root / tag / "metrics.jsonl"
                if smp.exists():
                    mp = smp
            s = summarize_run(mp)
            if s is None:
                continue
            summary[tag] = s
            print(
                f"{tag:24s} | {s['last_step']:6d} | {s['wall_min']:5.1f} | {s['val_bpb_end']:8.4f} | "
                f"{s['R0_start']:5.1f}->{s['R0_end']:<5.2f} | {s['peak_dA0']:8.2f} | {s['end_dA0']:7.2f} | "
                f"{s['min_cosA']:+7.3f} | {s['peak_E_sL0']:+8.2f} | {s['early_causD']:.2f}->{s['end_causD']:.2f}   | "
                f"{s['max_outI']:7.1f} | {'OK' if s['ic_all_pass'] else 'FAIL'}"
            )
    print("=" * 124 + "\n", flush=True)

    sc_path = out_root / "summary_scorecard.json"
    sc_path.write_text(json.dumps(summary, indent=2) + "\n")
    if sync_root is not None:
        sync_root.mkdir(parents=True, exist_ok=True)
        tmp = sync_root / "summary_scorecard.json.tmp"
        shutil.copy2(sc_path, tmp)
        os.replace(tmp, sync_root / "summary_scorecard.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--preset", default="stage2_70m", choices=["stage2_70m", "tiny_smoke"])
    ap.add_argument("--dataset", default="bin", choices=["bin", "fineweb-edu", "synthetic"])
    ap.add_argument("--train-bin", type=Path, default=None)
    ap.add_argument("--val-bin", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--sync-root", type=Path, default=None, help="Google Drive mirror root")
    ap.add_argument("--sync-latest", action="store_true", help="Mirror latest.pt (~700MB) to Drive for cross-session resume")
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--save-latest-every", type=int, default=1000)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.sync_root is not None:
        args.sync_root.mkdir(parents=True, exist_ok=True)

    train_script = Path(__file__).resolve().parent / "train_stage2.py"

    for seed in args.seeds:
        for arm in args.arms:
            tag = f"{arm}_s{seed}"
            run_dir = args.out_root / tag
            cmd = [
                sys.executable,
                "-u",
                str(train_script),
                "--arm",
                arm,
                "--preset",
                args.preset,
                "--dataset",
                args.dataset,
                "--seed",
                str(seed),
                "--peak-lr",
                str(args.peak_lr),
                "--save-latest-every",
                str(args.save_latest_every),
                "--device",
                args.device,
                "--out-dir",
                str(run_dir),
                "--resume",
            ]
            if args.train_bin is not None:
                cmd += ["--train-bin", str(args.train_bin)]
            if args.val_bin is not None:
                cmd += ["--val-bin", str(args.val_bin)]
            if args.total_steps is not None:
                cmd += ["--total-steps", str(args.total_steps)]
            if args.compile:
                cmd += ["--compile"]
            if args.sync_root is not None:
                cmd += ["--sync-dir", str(args.sync_root / tag)]
                if args.sync_latest:
                    cmd += ["--sync-latest"]

            print(f"\n>>> Launching Stage 2 run: {tag}", flush=True)
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                print(f"[ERROR] Run {tag} exited with code {rc}", flush=True)
                return rc
            print_scorecard(args.out_root, args.sync_root, args.arms, args.seeds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
