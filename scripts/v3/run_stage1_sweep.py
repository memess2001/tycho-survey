"""NUMA-aware, crash-proof, disk-evicting Stage 1 sweep orchestrator for Tycho Survey v3.

Runs `scripts/v3/scan_checkpoint.py` across Pythia (14M..1B + multi-seed) and OLMo-1/2
trajectories using isolated per-job temporary HuggingFace caches (evicted immediately after
each checkpoint completes so disk usage stays bounded < 20 GB across 500+ checkpoints).

Supports pinning each worker to a dedicated NUMA socket (e.g., 4x22-core Xeon on s82)
via `taskset`, exponential-backoff retries on HF downloads, and resumable skipping of
already-verified (`ic_v3.all_pass == True`) checkpoints.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCAN_SCRIPT = SCRIPT_DIR / "scan_checkpoint.py"

PYTHIA_36 = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128", "step256", "step512",
    "step1000", "step2000", "step3000", "step4000", "step5000", "step6000", "step8000", "step10000",
    "step12000", "step16000", "step20000", "step25000", "step32000", "step40000", "step50000",
    "step64000", "step80000", "step96000", "step112000", "step128000", "step135000",
    "step140000", "step141000", "step142000", "step143000",
]

PYTHIA_24 = [
    "step0", "step16", "step64", "step256", "step512",
    "step1000", "step2000", "step4000", "step6000", "step8000", "step10000",
    "step12000", "step16000", "step20000", "step25000", "step32000", "step40000",
    "step50000", "step64000", "step80000", "step96000", "step112000", "step128000", "step143000",
]

PYTHIA_18 = [
    "step0", "step64", "step256", "step512",
    "step1000", "step2000", "step4000", "step8000", "step12000",
    "step20000", "step32000", "step50000", "step64000", "step80000",
    "step96000", "step112000", "step128000", "step143000",
]

# Verified live against HF Refs API on 2026-11-27
OLMO1_28 = [
    "step0-tokens0B", "step1000-tokens2B", "step2000-tokens4B", "step3000-tokens6B",
    "step4000-tokens8B", "step5000-tokens10B", "step6000-tokens12B", "step8000-tokens16B",
    "step10000-tokens20B", "step12000-tokens25B", "step16000-tokens33B", "step20000-tokens41B",
    "step25000-tokens52B", "step32000-tokens67B", "step40000-tokens83B", "step50000-tokens104B",
    "step64000-tokens134B", "step80000-tokens167B", "step96000-tokens201B", "step112000-tokens234B",
    "step128000-tokens268B", "step143000-tokens299B", "step200000-tokens419B", "step300000-tokens628B",
    "step500000-tokens1048B", "step750000-tokens1572B", "step1000000-tokens2096B", "step1454000-tokens3048B",
]

OLMO2_18 = [
    "stage1-step0-tokens0B", "stage1-step300-tokens1B", "stage1-step10000-tokens21B",
    "stage1-step20000-tokens42B", "stage1-step30000-tokens63B", "stage1-step40000-tokens84B",
    "stage1-step50000-tokens105B", "stage1-step60000-tokens126B", "stage1-step80000-tokens168B",
    "stage1-step100000-tokens210B", "stage1-step120000-tokens252B", "stage1-step140000-tokens294B",
    "stage1-step200000-tokens420B", "stage1-step300000-tokens630B", "stage1-step500000-tokens1049B",
    "stage1-step1000000-tokens2098B", "stage1-step1500000-tokens3146B", "stage1-step1907359-tokens4001B",
]

TIERS: dict[str, list[tuple[str, list[str]]]] = {
    "verify_olmo": [
        ("allenai/OLMo-1B-0724-hf", ["step1000-tokens2B"]),
        ("allenai/OLMo-2-0425-1B", ["stage1-step300-tokens1B"]),
    ],
    "tier1_pythia_small": [
        ("EleutherAI/pythia-14m", PYTHIA_36),
        ("EleutherAI/pythia-31m", PYTHIA_36),
        ("EleutherAI/pythia-70m", PYTHIA_36),
        ("EleutherAI/pythia-160m", PYTHIA_36),
    ],
    "tier2_pythia_seeds": [
        ("EleutherAI/pythia-70m-seed1", PYTHIA_36),
        ("EleutherAI/pythia-70m-seed2", PYTHIA_36),
        ("EleutherAI/pythia-70m-seed3", PYTHIA_36),
        ("EleutherAI/pythia-160m-seed1", PYTHIA_36),
        ("EleutherAI/pythia-160m-seed2", PYTHIA_36),
        ("EleutherAI/pythia-160m-seed3", PYTHIA_36),
    ],
    "tier3_pythia_large": [
        ("EleutherAI/pythia-410m", PYTHIA_36),
        ("EleutherAI/pythia-410m-seed1", PYTHIA_18),
        ("EleutherAI/pythia-1b", PYTHIA_24),
    ],
    "tier4_olmo_cross": [
        ("allenai/OLMo-1B-0724-hf", OLMO1_28),
        ("allenai/OLMo-2-0425-1B", OLMO2_18),
    ],
}


def is_done(out_dir: Path, model_id: str, revision: str) -> bool:
    slug = model_id.replace("/", "__")
    model_dir = out_dir / slug
    jpath = model_dir / f"{slug}__{revision}.json"
    npath = model_dir / f"{slug}__{revision}.npz"
    if not (jpath.exists() and npath.exists()):
        return False
    try:
        d = json.loads(jpath.read_text())
        return bool(d.get("ic_v3", {}).get("all_pass", False))
    except Exception:
        return False


def run_one_job(
    worker_id: int,
    model_id: str,
    revision: str,
    out_dir: Path,
    *,
    python_exe: str,
    threads: int,
    numa_cores: str | None,
    cache_root: Path,
    core_seqs: int,
    causal_seqs: int,
    seq_len: int,
    probe_n: int,
    gram_m: int,
    batch_size: int,
    max_retries: int = 3,
) -> dict:
    slug = model_id.replace("/", "__")
    model_dir = out_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    if is_done(out_dir, model_id, revision):
        return {"model": model_id, "revision": revision, "status": "skipped", "wall_s": 0.0}

    t0 = time.perf_counter()
    last_err = ""
    for attempt in range(1, max_retries + 1):
        tmp_cache = cache_root / f"hf_w{worker_id}_{os.getpid()}_{int(time.time()*1000)}"
        tmp_cache.mkdir(parents=True, exist_ok=True)
        try:
            env = os.environ.copy()
            env["HF_HOME"] = str(tmp_cache)
            env["OMP_NUM_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)
            env["OPENBLAS_NUM_THREADS"] = str(threads)
            env["TOKENIZERS_PARALLELISM"] = "false"

            cmd = [
                python_exe,
                str(SCAN_SCRIPT),
                "--model", model_id,
                "--revisions", revision,
                "--out-dir", str(model_dir),
                "--device", "auto",
                "--dtype", "auto",
                "--threads", str(threads),
                "--core-seqs", str(core_seqs),
                "--causal-seqs", str(causal_seqs),
                "--seq-len", str(seq_len),
                "--probe-n", str(probe_n),
                "--gram-m", str(gram_m),
                "--batch-size", str(batch_size),
            ]
            if numa_cores and shutil.which("taskset"):
                cmd = ["taskset", "-c", numa_cores] + cmd

            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            if res.returncode == 0 and is_done(out_dir, model_id, revision):
                wall = time.perf_counter() - t0
                return {
                    "model": model_id,
                    "revision": revision,
                    "status": "ok",
                    "attempt": attempt,
                    "worker": worker_id,
                    "wall_s": round(wall, 2),
                }
            last_err = (res.stderr or res.stdout or f"rc={res.returncode}")[-600:]
        except Exception as e:
            last_err = repr(e)
        finally:
            shutil.rmtree(tmp_cache, ignore_errors=True)

        if attempt < max_retries:
            backoff = 15 * (3 ** (attempt - 1))
            time.sleep(backoff)

    wall = time.perf_counter() - t0
    return {
        "model": model_id,
        "revision": revision,
        "status": "failed",
        "worker": worker_id,
        "wall_s": round(wall, 2),
        "error": last_err,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tycho v3 Stage 1 multi-worker sweep orchestrator")
    ap.add_argument("--tiers", nargs="+", choices=list(TIERS.keys()), help="Named tiers to execute in order")
    ap.add_argument("--custom-model", help="Optional custom HF model ID instead of --tiers")
    ap.add_argument("--custom-revisions", nargs="+", help="Optional custom revisions for --custom-model")
    ap.add_argument("--out-dir", default="work/v3_stage1", help="Root output directory")
    ap.add_argument("--cache-root", default="/tmp/hf_v3_sweep", help="Scratch root for per-job HF cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads-per-worker", type=int, default=22)
    ap.add_argument(
        "--numa-slices",
        nargs="*",
        default=["0-21", "22-43", "44-65", "66-87"],
        help="CPU core slices for taskset pinning (empty to disable)",
    )
    ap.add_argument("--core-seqs", type=int, default=32)
    ap.add_argument("--causal-seqs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--probe-n", type=int, default=16)
    ap.add_argument("--gram-m", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    jobs: list[tuple[str, str]] = []
    if args.custom_model and args.custom_revisions:
        for rev in args.custom_revisions:
            jobs.append((args.custom_model, rev))
    elif args.tiers:
        for tier in args.tiers:
            for model_id, revs in TIERS[tier]:
                for rev in revs:
                    jobs.append((model_id, rev))
    else:
        ap.error("Specify either --tiers or --custom-model + --custom-revisions")

    pending = [(m, r) for (m, r) in jobs if not is_done(out_dir, m, r)]
    print(
        f"[sweep] Total jobs={len(jobs)}, already_done={len(jobs)-len(pending)}, pending={len(pending)} "
        f"| workers={args.workers}, threads/worker={args.threads_per_worker}",
        flush=True,
    )
    if not pending:
        print("[sweep] All jobs already complete.", flush=True)
        return 0

    t_start = time.perf_counter()
    done_cnt = 0
    ok_cnt = 0
    fail_cnt = 0

    # Dedicated worker slots so each concurrent process keeps its own NUMA socket
    import queue

    slot_q: queue.Queue[int] = queue.Queue()
    for w in range(args.workers):
        slot_q.put(w)

    def _wrapped(job: tuple[str, str]) -> dict:
        wid = slot_q.get()
        try:
            numa = args.numa_slices[wid] if (args.numa_slices and wid < len(args.numa_slices)) else None
            return run_one_job(
                wid,
                job[0],
                job[1],
                out_dir,
                python_exe=sys.executable,
                threads=args.threads_per_worker,
                numa_cores=numa,
                cache_root=cache_root,
                core_seqs=args.core_seqs,
                causal_seqs=args.causal_seqs,
                seq_len=args.seq_len,
                probe_n=args.probe_n,
                gram_m=args.gram_m,
                batch_size=args.batch_size,
            )
        finally:
            slot_q.put(wid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_wrapped, j) for j in pending]
        for fut in concurrent.futures.as_completed(futs):
            rec = fut.result()
            done_cnt += 1
            if rec["status"] in ("ok", "skipped"):
                ok_cnt += 1
            else:
                fail_cnt += 1
            rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(manifest_path, "a") as mf:
                mf.write(json.dumps(rec) + "\n")
            elapsed = time.perf_counter() - t_start
            rate = done_cnt / max(elapsed, 1e-6)
            eta_min = (len(pending) - done_cnt) / max(rate, 1e-6) / 60.0
            print(
                f"[{done_cnt:3d}/{len(pending):3d}] {rec['status']:7s} {rec['model']} @ {rec['revision']} "
                f"({rec['wall_s']:5.1f}s, w={rec.get('worker','-')}) | elapsed={elapsed/60:.1f}m ETA={eta_min:.1f}m"
                + (f" ERR={rec.get('error','')[:120]}" if rec["status"] == "failed" else ""),
                flush=True,
            )

    total_m = (time.perf_counter() - t_start) / 60.0
    print(f"[sweep] FINISHED in {total_m:.1f} min: ok={ok_cnt}, failed={fail_cnt}", flush=True)
    return 0 if fail_cnt == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
