"""Offline analysis of sky v2 Part A, as pre-registered in docs/results/v2a_preregistration.md (binding).

Stage 1 (per object, in parallel, cached in ``--cache``): all measures of §1 for every set of 1-3 inputs at every
checkpoint, the exact losses, the sampled values and the per-object inputs of the instrument checks
(:func:`tycho.analysis.v2a.stage1_object`). IC-1 replays run in the same pool. Stage 2: the instrument checks, T1, T2,
the bootstrap, Holm, the decision table and Q1-Q5 (:func:`tycho.analysis.v2a.analyse`). Everything is written to one
JSON file (``--out``), with the DECISIONS list of the choices the pre-registration leaves open and the run metadata.

BLAS and OpenMP are pinned to one thread per process before numpy is imported (pre-registration §6).

Usage::

    python scripts/analysis_v2/v2a_analysis.py --manifest work/v2a/manifest.jsonl --out work/v2a/analysis.json \\
        --jobs 4 --cache work/v2a/stage1

Pilot (2 seeds per configuration)::

    python scripts/analysis_v2/v2a_analysis.py --manifest work/v2a_pilot/manifest.jsonl \\
        --out work/v2a_pilot/analysis.json --min-seeds 2
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import datetime  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from tycho.analysis import v2a  # noqa: E402


def _stage1(rec: dict, cache: str, manifest: str, force: bool, ic2a: float, ic3: bool, local_ri: bool) -> dict:
    try:
        return v2a.stage1_object(rec, cache, manifest, force=force, ic2a_fraction=ic2a, ic3=ic3,
                                 ic4_local_ri=local_ri)
    except Exception:
        return {"object_id": rec["object_id"], "error": traceback.format_exc()}


def _replay(rec: dict, sky: str, work: str, manifest: str) -> dict:
    try:
        return v2a.replay_object(rec, sky, work, manifest)
    except Exception:
        return {"object_id": rec["object_id"], "region": rec.get("region"), "passed": False,
                "error": traceback.format_exc()}


def _git_describe() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty", "--tags"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:  # not a git checkout
        return f"unavailable ({exc})"


def _source_versions() -> dict:
    """SHA-256 and git status of the analysis sources (untracked or modified files are not covered by git describe)."""
    out = {}
    for rel in ("tycho/analysis/v2a.py", "scripts/analysis_v2/v2a_analysis.py", "tycho/ri/cells.py",
                "tycho/tasks/patterns.py"):
        path = REPO / rel
        try:
            status = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=REPO, capture_output=True,
                                    text=True, check=True).stdout.strip()[:2].strip() or "clean"
        except Exception:
            status = "unknown"
        out[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                    "git_status": status}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="manifest JSONL (v2a_pilot.py / v2a_manifest.py format)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker processes (spawn)")
    ap.add_argument("--cache", default=None, help="stage-1 cache directory (default: <out dir>/v2a_stage1)")
    ap.add_argument("--bootstrap", type=int, default=v2a.N_BOOT, help="bootstrap resamples (pre-registered: 10000)")
    ap.add_argument("--min-seeds", type=int, default=v2a.MIN_SEEDS,
                    help="seeds per configuration or twin set needed to enter rho-bar (pre-registered: 5)")
    ap.add_argument("--replays", type=float, default=0.05, help="IC-1: fraction of objects replayed per region")
    ap.add_argument("--sky", default=str(REPO / "configs" / "sky_v2a.yaml"), help="sky config for IC-1 replays")
    ap.add_argument("--skip-replays", action="store_true", help="skip IC-1")
    ap.add_argument("--skip-ic2a", action="store_true", help="skip the IC-2(a) library recomputation in stage 1")
    ap.add_argument("--skip-ic3", action="store_true", help="skip the IC-3 corpus regeneration in stage 1")
    ap.add_argument("--skip-local-ri", action="store_true", help="skip the IC-4 local_ri comparison in stage 1")
    ap.add_argument("--force", action="store_true", help="recompute stage 1 even when cached")
    ap.add_argument("--stage1-only", action="store_true", help="stop after stage 1 (and the replays)")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    manifest = Path(args.manifest).resolve()
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out.parent / "v2a_stage1"
    cache.mkdir(parents=True, exist_ok=True)
    records = v2a.load_manifest(manifest)
    done = [r for r in records if r.get("status") == "done"]
    print(f"{len(records)} objects in {manifest} ({len(done)} done); cache {cache}; {args.jobs} workers", flush=True)

    replay_recs = [] if args.skip_replays else v2a.select_replays(records, args.replays)
    stage1_errors, replays = [], []
    t1 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, args.jobs)) as pool:
        rp = [pool.apply_async(_replay, (r, args.sky, str(cache / "replays"), str(manifest))) for r in replay_recs]
        fn = functools.partial(_stage1, cache=str(cache), manifest=str(manifest), force=args.force,
                               ic2a=0.0 if args.skip_ic2a else v2a.IC2A_FRACTION, ic3=not args.skip_ic3,
                               local_ri=not args.skip_local_ri)
        for i, res in enumerate(pool.imap_unordered(fn, done), 1):
            if "error" in res:
                stage1_errors.append(res)
                print(f"[stage 1 {i}/{len(done)}] {res['object_id']} FAILED\n{res['error']}", flush=True)
            else:
                print(f"[stage 1 {i}/{len(done)}] {res['object_id']} {res['seconds']:.1f} s"
                      f"{' (cached)' if res['cached'] else ''}", flush=True)
        t_stage1 = time.perf_counter() - t1
        for r in rp:
            res = r.get()
            replays.append(res)
            print(f"[IC-1] {res['object_id']} {res.get('region')}: {'pass' if res.get('passed') else 'FAIL'} "
                  f"{res.get('mismatches', res.get('error', ''))}", flush=True)
    t_pool = time.perf_counter() - t1
    if stage1_errors:  # listed as not done in the output; the decision is withheld (instrument_checks)
        print(f"{len(stage1_errors)} stage-1 failures: {[e['object_id'] for e in stage1_errors]}", file=sys.stderr)
    if args.stage1_only:
        return 1 if stage1_errors else 0

    t2 = time.perf_counter()
    skipped = [name for name, flag in (("IC-1", args.skip_replays), ("IC-2a", args.skip_ic2a),
                                       ("IC-3", args.skip_ic3), ("IC-4 local_ri", args.skip_local_ri)) if flag]
    result = v2a.analyse(manifest, cache, B=args.bootstrap, min_seeds=args.min_seeds,
                         replays=None if args.skip_replays else replays,
                         stage1_failed={e["object_id"]: e["error"] for e in stage1_errors}, skipped_checks=skipped)
    t_stage2 = time.perf_counter() - t2
    result["run"] = {
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "code_version": _git_describe(), "sources": _source_versions(), "stage1_version": v2a.STAGE1_VERSION,
        "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:] if argv is None else argv, "jobs": args.jobs, "cache": str(cache),
        "skipped": {"IC-1": args.skip_replays, "IC-2a": args.skip_ic2a, "IC-3": args.skip_ic3,
                    "IC-4 local_ri": args.skip_local_ri},
        "python": platform.python_version(), "numpy": np.__version__,
        "blas_threads": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                         "MKL_NUM_THREADS")},
        "timing_s": {"stage1_wall": t_stage1, "stage1_and_replays_wall": t_pool, "stage2_wall": t_stage2,
                     "total_wall": time.perf_counter() - t_start},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v2a.jsonable(result), indent=1, ensure_ascii=False) + "\n")
    d = result["decision"]
    ic = result["instrument_checks"]["bug_checks_passed"]
    print(f"instrument checks: {ic}")
    for c in result["comparisons"]:
        print(f"{c['comparison']:<16} Δb={c['delta_eta_b']:+.3f} Δg={c['delta_eta_g']:+.3f} "
              f"p_holm_b={c.get('p_holm_eta_b', float('nan')):.3g} p_holm_g={c.get('p_holm_eta_g', float('nan')):.3g}"
              f"  {c['label']}")
    print(f"decision: {d['outcome']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
