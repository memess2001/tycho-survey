"""Offline analysis of sky v2 Part B, as pre-registered in docs/results/v2b_preregistration.md (binding).

Stage 1 (per object, in parallel, cached in ``--cache``): θ, bounded η and Gaussian η at C_U and M_U for every set of
1-3 heads at every checkpoint (:func:`tycho.analysis.v2b.stage1_object`). IC-1 replays (one per configuration group)
run in the same pool. Stage 2: roles, formation, eligibility, IC-2, B1, B2, M with the joint-seed bootstrap and Holm,
the secondaries and the descriptive §5 (:func:`tycho.analysis.v2b.analyse`). Everything is written to one JSON file
(``--out``), with the DECISIONS list of the choices the pre-registration leaves open and the run metadata.

BLAS and OpenMP are pinned to one thread per process before numpy is imported.

Usage::

    python scripts/analysis_v2/v2b_analysis.py --manifest work/v2b/manifest.jsonl --out work/v2b/analysis.json \\
        --jobs 8

Pilot (2 seeds per size; the pre-registered minimum of eligible seeds per size is 10)::

    python scripts/analysis_v2/v2b_analysis.py --manifest work/v2b_pilot/manifest.jsonl \\
        --out work/v2b_pilot/analysis.json --min-eligible 2
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

from tycho.analysis import v2b  # noqa: E402


def _stage1(rec: dict, cache: str, manifest: str, force: bool) -> dict:
    try:
        return v2b.stage1_object(rec, cache, manifest, force=force)
    except Exception:
        return {"object_id": rec["object_id"], "error": traceback.format_exc()}


def _replay(rec: dict, sky: str, work: str, manifest: str) -> dict:
    try:
        return v2b.replay_object(rec, sky, work, manifest)
    except Exception:
        return {"object_id": rec["object_id"], "region": rec.get("region"), "group": list(v2b.config_group(rec)),
                "passed": False, "error": traceback.format_exc()}


def _git_describe() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty", "--tags"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:  # not a git checkout
        return f"unavailable ({exc})"


def _source_versions() -> dict:
    """SHA-256 and git status of the analysis sources (untracked or modified files are not covered by git describe)."""
    out = {}
    for rel in ("tycho/analysis/v2b.py", "scripts/analysis_v2/v2b_analysis.py", "tycho/analysis/v2a.py",
                "tycho/ri/cells.py", "tycho/tasks/induction.py"):
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
    ap.add_argument("--manifest", required=True, help="manifest JSONL (v2b_pilot.py / v2a_manifest.py format)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker processes (spawn)")
    ap.add_argument("--cache", default=None, help="stage-1 cache directory (default: <out dir>/v2b_stage1)")
    ap.add_argument("--bootstrap", type=int, default=v2b.N_BOOT, help="bootstrap resamples (pre-registered: 10000)")
    ap.add_argument("--min-eligible", type=int, default=v2b.MIN_ELIGIBLE,
                    help="eligible seeds a size needs to enter M's Delta (pre-registered: 10)")
    ap.add_argument("--checkpoints", type=int, default=v2b.N_CHECKPOINTS,
                    help="expected pattern checkpoints per object (pre-registered: 71; 0: not checked)")
    ap.add_argument("--sky", default=str(REPO / "configs" / "sky_v2b.yaml"), help="sky config for IC-1 replays")
    ap.add_argument("--skip-replays", action="store_true", help="skip IC-1 (withholds the verdicts)")
    ap.add_argument("--force", action="store_true", help="recompute stage 1 even when cached")
    ap.add_argument("--stage1-only", action="store_true", help="stop after stage 1 (and the replays)")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    manifest = Path(args.manifest).resolve()
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out.parent / "v2b_stage1"
    cache.mkdir(parents=True, exist_ok=True)
    records = v2b.load_manifest(manifest)
    done = [r for r in records if r.get("status") == "done"]
    print(f"{len(records)} objects in {manifest} ({len(done)} done); cache {cache}; {args.jobs} workers", flush=True)

    replay_recs = [] if args.skip_replays else v2b.select_replays(records)
    stage1_errors, replays, stage1_seconds = [], [], []
    t1 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, args.jobs)) as pool:
        rp = [pool.apply_async(_replay, (r, args.sky, str(cache / "replays"), str(manifest))) for r in replay_recs]
        fn = functools.partial(_stage1, cache=str(cache), manifest=str(manifest), force=args.force)
        for i, res in enumerate(pool.imap_unordered(fn, done), 1):
            if "error" in res:
                stage1_errors.append(res)
                print(f"[stage 1 {i}/{len(done)}] {res['object_id']} FAILED\n{res['error']}", flush=True)
            else:
                stage1_seconds.append(res["seconds"])
                print(f"[stage 1 {i}/{len(done)}] {res['object_id']} {res['seconds']:.1f} s"
                      f"{' (cached)' if res['cached'] else ''}", flush=True)
        t_stage1 = time.perf_counter() - t1
        for r in rp:
            res = r.get()
            replays.append(res)
            print(f"[IC-1] {res['object_id']} {res.get('group')}: {'pass' if res.get('passed') else 'FAIL'} "
                  f"{res.get('mismatches', res.get('error', ''))} {res.get('seconds', 0):.0f} s", flush=True)
    t_pool = time.perf_counter() - t1
    if stage1_errors:  # listed as not done in the output; the verdicts are withheld
        print(f"{len(stage1_errors)} stage-1 failures: {[e['object_id'] for e in stage1_errors]}", file=sys.stderr)
    if args.stage1_only:
        return 1 if stage1_errors else 0

    t2 = time.perf_counter()
    result = v2b.analyse(manifest, cache, B=args.bootstrap, min_eligible=args.min_eligible,
                         replays=None if args.skip_replays else replays,
                         stage1_failed={e["object_id"]: e["error"] for e in stage1_errors},
                         n_checkpoints=args.checkpoints or None)
    t_stage2 = time.perf_counter() - t2
    result["run"] = {
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "code_version": _git_describe(), "sources": _source_versions(), "stage1_version": v2b.STAGE1_VERSION,
        "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:] if argv is None else argv, "jobs": args.jobs, "cache": str(cache),
        "skipped": {"IC-1": args.skip_replays},
        "python": platform.python_version(), "numpy": np.__version__,
        "blas_threads": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                         "MKL_NUM_THREADS")},
        "timing_s": {"stage1_wall": t_stage1, "stage1_and_replays_wall": t_pool, "stage2_wall": t_stage2,
                     "total_wall": time.perf_counter() - t_start,
                     "stage1_per_object_mean": float(np.mean(stage1_seconds)) if stage1_seconds else None},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v2b.jsonable(result), indent=1, ensure_ascii=False) + "\n")

    ic = result["instrument_checks"]
    print(f"IC-1: {'skipped' if ic['IC-1']['skipped'] else ('pass' if ic['IC-1']['passed'] else 'FAIL')}; "
          f"IC-2: {'pass' if ic['IC-2']['passed'] else 'FAIL'}; data checks: "
          f"{'pass' if ic['data_checks']['passed'] else 'FAIL'}")
    for d, s in result["B1"]["per_size"].items():
        print(f"B1 d_model {d}: formed {s['n_formed']}/{s['n_objects']}, phase-change steps {s['phase_change_steps']}")
    b1 = result["B1"]
    print(f"B1 slope {b1['slope']:+.3f}, 95 % interval {b1['slope_interval_95']}")
    print(f"B2 share {result['B2']['n_before']}/{result['B2']['n_formed']}")
    m = result["M"]
    print(f"M family sizes {m['family_sizes']} (eligible by size {m['n_eligible_by_size']})")
    for c in m["comparisons"]:
        print(f"M ({c['referee']:>2}, {c['eta_value']}, {c['reference']}): Δ={c['delta']:+.4f} "
              f"p={c['p_raw']:.3g} p_holm={c['p_holm']:.3g}")
    print(f"M outcome: {m['decision']['outcome']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
