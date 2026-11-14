"""Write the manifest of sky v2a objects from a survey catalog, in the format of ``v2a_pilot.py``.

One JSON line per object of the sky version: ``object_id``, ``region``, ``task``, ``task_cfg``, ``seed``,
``sky_version``, ``status``, ``wall_time_s``, ``error``, and, for done objects, the paths ``lightcurve``
(``data/lightcurves/<id[:2]>/<id>.npz``) and ``patterns`` (``data/ri/<id[:2]>/<id>_patterns.npz``).

Usage: ``python scripts/analysis_v2/v2a_manifest.py --root ~/tycho-survey --out work/v2a/manifest.jsonl``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tycho.config import Paths  # noqa: E402
from tycho.survey.catalog import Catalog  # noqa: E402
from tycho.survey.worker import PATTERNS_TAG  # noqa: E402
from tycho.train.spec import ObjectSpec  # noqa: E402


def manifest_rows(root: Path, sky_version: str) -> list[dict]:
    paths = Paths(root)
    catalog = Catalog(paths.db, root=paths.root)
    try:
        rows = catalog.query("SELECT object_id, region, sky_version, seed, spec_json, status, wall_time_s, error "
                             "FROM objects WHERE sky_version = ? ORDER BY region, object_id", (sky_version,))
    finally:
        catalog.close()
    out = []
    for r in rows:
        spec = ObjectSpec.from_json(r["spec_json"])
        rec = {"object_id": r["object_id"], "region": r["region"], "task": spec.task, "task_cfg": spec.task_cfg,
               "seed": int(r["seed"]), "sky_version": r["sky_version"], "status": r["status"],
               "wall_time_s": r["wall_time_s"], "error": r["error"]}
        if r["status"] == "done":
            rec["lightcurve"] = str(paths.lightcurve_path(r["object_id"]))
            rec["patterns"] = str(paths.ri_path(r["object_id"], PATTERNS_TAG))
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--sky-version", default="sky-v2a")
    ap.add_argument("--out", default=str(REPO / "work" / "v2a" / "manifest.jsonl"))
    args = ap.parse_args(argv)
    rows = manifest_rows(Path(args.root), args.sky_version)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    done = sum(r["status"] == "done" for r in rows)
    print(f"{len(rows)} objects ({done} done) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
