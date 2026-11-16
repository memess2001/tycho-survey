"""Pilot of sky v2 Part B on non-confirmatory seeds, to test the offline analysis before the run.

Runs a subset of configs/sky_v2b.yaml with the region settings unchanged except the sky version
(``sky-v2b-pilot3``), the first seed (default 900, outside the confirmatory ranges 400–419 and 450–459) and the
number of seeds. Each object is trained in pattern mode exactly as a survey worker would and written to
``<out>/lightcurves/<id>.npz`` and ``<out>/patterns/<id>.npz``, with one line per object in ``<out>/manifest.jsonl``
(the format of ``v2a_manifest.py``, which also serves Part B).

Usage: ``python scripts/analysis_v2/v2b_pilot.py --out work/v2b_pilot --jobs 4``
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tycho.survey.sky import Sky  # noqa: E402
from tycho.train.spec import ObjectSpec, RIConfig  # noqa: E402

#: Pilot points per region (d_model 192 left out to keep the pilot short).
PILOT_POINTS = {"induction-main": [{"d_model": 48}, {"d_model": 96}],
                "induction-norepeat": [{"d_model": 48}, {"d_model": 96}],
                "induction-1layer": [{}]}
CONFIRMATORY = set(range(400, 420)) | set(range(450, 460))


def _run(job: tuple[dict, dict, str, str]) -> dict:
    import torch

    from tycho.survey.worker import save_light_curve
    from tycho.train import run_object

    torch.set_num_threads(1)
    spec_d, ri_d, out, region = job
    spec = ObjectSpec.from_dict(spec_d)
    oid = spec.object_id()
    out = Path(out)
    t0 = time.perf_counter()
    res = run_object(spec, work_dir=out / "work" / oid, ri=RIConfig.from_dict(ri_d))
    rec = {"object_id": oid, "region": region, "task": spec.task, "task_cfg": spec.task_cfg, "seed": spec.seed,
           "sky_version": spec.sky_version, "status": res.status, "wall_time_s": time.perf_counter() - t0,
           "error": res.error}
    if res.status == "done":
        rec["lightcurve"] = str(save_light_curve(out / "lightcurves" / f"{oid}.npz", res.light_curve))
        rec["patterns"] = str(save_light_curve(out / "patterns" / f"{oid}.npz", res.patterns))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / "work" / "v2b_pilot"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args(argv)
    if CONFIRMATORY & set(range(args.seed_base, args.seed_base + args.seeds)):
        ap.error("pilot seeds must lie outside the confirmatory ranges 400-419 and 450-459")
    sky = Sky.from_yaml(REPO / "configs" / "sky_v2b.yaml")
    sky = Sky(version="sky-v2b-pilot3", description="Part B analysis pilot",
              regions=[dataclasses.replace(r, seed_base=args.seed_base, seeds=args.seeds) for r in sky.regions])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = {json.loads(line)["object_id"] for line in manifest.read_text().splitlines()} if manifest.exists() else set()
    jobs = []
    for region in sky.regions:
        for point in PILOT_POINTS[region.name]:
            for rank in range(region.seeds):
                spec = sky.make_spec(region, point, rank)
                if spec.object_id() not in done:
                    jobs.append((spec.to_dict(), {**region.ri, "enabled": True}, str(out), region.name))
    print(f"{len(jobs)} objects to run", flush=True)
    with mp.get_context("spawn").Pool(args.jobs) as pool, manifest.open("a") as fh:
        for rec in pool.imap_unordered(_run, jobs):
            fh.write(json.dumps(rec, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
            fh.flush()
            print(f"{rec['region']} seed {rec['seed']} {rec['status']} {rec['wall_time_s']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
