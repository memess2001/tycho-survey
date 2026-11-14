"""Data releases (SPEC §6.4): versioned, checksummed, self-describing snapshots of the catalog, in the spirit of
SDSS data releases, so that others can look for their own empirical laws in the survey.

``make_release(catalog, paths, "DR1")`` writes ``releases/DR1/``:

==========================  ===========================================================================================
``objects.parquet/.csv``    one row per *done* object: identity, configuration (``task_cfg_json``; ``train_*``
                            columns), bookkeeping, every photometry summary key (:data:`SUMMARY_KEYS`) and the latest
                            classification (``label``, ``label_tags_json``, ``label_score``, ``classifier``)
``ri_runs.parquet/.csv``    RI spectroscopy runs (in situ or follow-up) with their estimator settings
``ri_events.parquet/.csv``  activation / extinction / reactivation events of variable subsets
``transients.parquet/.csv`` alerts raised from events, photometry and classification
``lightcurves.tar.gz``      ``lightcurves/<object_id>.npz`` (``np.load``; keys in the datasheet)
``ricurves.tar.gz``         ``ricurves/<object_id>_<kind>_<ri_run_id>.npz`` (``tycho.analysis.ricurve.RICurve.load``)
``patterns.tar.gz``         ``patterns/<object_id>.npz``: pattern-mode observations (sky v2 Part A; keys in
                            ``tycho.train.runner``, "Pattern mode"); only when some released object has one
                            An archive larger than :data:`ARCHIVE_PART_BYTES` is split into
                            ``<name>.part001.tar.gz``, ``<name>.part002.tar.gz``, ... (each a complete tar.gz), so
                            that every file fits GitHub's 100 MB limit; extract all parts into one directory.
``sky.yaml``                the sky definition (regions, knobs) in use when the release was made
``calibration.json``        the instrument's error budget (if the survey was calibrated)
``README.md``               the datasheet: contents, schema with units, provenance, licence (CC-BY-4.0), how to verify
``MANIFEST.sha256``         SHA-256 of every other file, in ``sha256sum`` format
==========================  ===========================================================================================

Selection: ``sky_versions`` (CLI ``--sky-version``, repeatable) restricts a release to objects of those sky versions
(and their runs, events and transients), e.g. DR1 = sky v1 only, since DR0 already holds sky v0.

Consistency: the tables are read from an online backup of the catalog taken at the start, so a release made while
the daemon is running is one consistent snapshot. Only done objects are released (queued and running objects are
not observations yet; failed objects are infrastructure failures, kept in the catalog), and only events, runs and
transients of released objects. Releases are immutable: an existing tag is an error unless ``overwrite=True``. The
release is assembled in a temporary directory and renamed into place, so a crash never leaves a partial release
under the tag (a leftover ``releases/.<tag>.partial-<pid>`` directory is removed by the next attempt). Archives
are reproducible (sorted members, zero timestamps and owners), so re-releasing unchanged data gives byte-identical
archives.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import tycho
from tycho.analysis.summarize import SUMMARY_KEYS
from tycho.config import Paths, load_config
from tycho.survey.catalog import SCHEMA_VERSION, Catalog
from tycho.train.spec import TrainConfig

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["LICENSE", "MANIFEST", "RELEASE_FORMAT", "make_release", "verify_release"]

RELEASE_FORMAT = "tycho-release/1"
"""Format tag recorded in the datasheet."""

LICENSE = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

MANIFEST = "MANIFEST.sha256"

ARCHIVE_PART_BYTES = 90_000_000
"""Compressed size after which an archive continues in a new part (GitHub refuses files over 100 MB; a part can
exceed this by at most one member)."""
"""Checksum file (``sha256sum -c`` / ``shasum -a 256 -c`` format)."""

TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
"""Allowed release tags (a single path component)."""

_COUNT_KEYS = frozenset({"n_evals", "steps_total", "n_drops", "loss_spikes"})
_TRAIN_FIELDS = tuple(f.name for f in fields(TrainConfig))

OBJECT_COLUMNS: dict[str, str] = {
    "object_id": "object identity: sha1 of (task, task_cfg, seed, train, sky_version), 16 hex digits",
    "task": "task name (tycho.tasks.REGISTRY)",
    "region": "sky region the object was observed in",
    "sky_version": "version of the sky definition (part of the identity)",
    "seed": "training seed (initial weights and batch order)",
    "task_cfg_json": "task configuration as canonical JSON (all keys of the task's defaults that were set)",
    "source": "why it was observed: coarse (grid), refine (boundary refinement) or agent (proposal)",
    "random_control": "1 if the object belongs to the uniform random control sample",
    "priority": "scheduling priority",
    "attempts": "claims that crashed, failed, completed or are running (graceful interruptions excluded)",
    "created_at": "enqueued (Unix time, s, UTC)",
    "started_at": "last claimed (Unix time, s, UTC)",
    "finished_at": "observation finished (Unix time, s, UTC)",
    "wall_time_s": "wall-clock time of the observation (s; resumed runs add up their sessions)",
    "device": "torch device of the observation",
    "n_params": "trainable parameters of the model",
    "code_version": "Tycho version (+ git describe) of the observing code",
    "lightcurve_points": "evaluations in the light curve",
    "lightcurve_file": "member of lightcurves.tar.gz (empty when not included)",
    "patterns_file": "member of patterns.tar.gz: log-odds on every input pattern and probe counts per RI step "
                     "(pattern-mode objects only; empty otherwise)",
    "label": "latest classification label (rules-v1: diverged, no_learning, memorization, grokking, plateau_drop, "
             "staged, double_descent, smooth, unknown; or a label set by the agent)",
    "label_tags_json": "JSON list of secondary tags of the latest classification",
    "label_score": "heuristic confidence of the latest classification in [0, 1] (not a probability)",
    "classifier": "who classified: the rule set (rules-v1) or agent",
    "labelled_at": "time of the latest classification (Unix time, s, UTC)",
}
RI_RUN_COLUMNS: dict[str, str] = {
    "ri_run_id": "RI run id", "object_id": "object", "kind": "insitu (observed during training) or followup "
    "(deterministic replay with dense checkpoints around the largest loss drop)",
    "n_subsets": "subsets kept in the curve (pruned: ever detected, the strongest, ground truth)",
    "n_points": "RI checkpoints (training steps)", "max_order": "largest subset order scanned",
    "alpha": "FDR level (Benjamini-Hochberg q-value) of a detection", "min_effect": "effect-size threshold of a "
    "detection (bias-corrected local RI, nats)", "created_at": "recorded (Unix time, s, UTC)",
    "ricurve_file": "member of ricurves.tar.gz (empty when not included)",
}
EVENT_COLUMNS: dict[str, str] = {
    "event_id": "event id", "ri_run_id": "RI run the event was detected in", "object_id": "object",
    "subset_json": "JSON list of the variable names of the subset", "order": "subset size (interaction order)",
    "kind": "activation, extinction or reactivation", "step": "first checkpoint in the new state (training step)",
    "step_lo": "last checkpoint before the change: the change happened in (step_lo, step]",
    "step_hi": "= step", "ri_before": "mean local RI over the checkpoints before (nats)",
    "ri_after": "mean local RI over the confirming checkpoints (nats)", "peak_ri": "maximum RI of the episode (nats)",
    "confidence": "1 - max q over the confirmation window (activations); for extinctions 1 - max(RI)/off",
    "has_precursor": "order >= 3 activations: a proper subset of order >= 2 activated earlier",
    "matches_gt": "the subset is a planted ground-truth interaction (positive controls)",
    "created_at": "recorded (Unix time, s, UTC)",
}
TRANSIENT_COLUMNS: dict[str, str] = {
    "transient_id": "transient id", "object_id": "object", "event_id": "triggering event (empty for photometric "
    "transients)", "kind": "late_activation, pure_synergy, order_inversion, extinction, reactivation, grokking, "
    "loss_spike or unknown_class", "severity": "1 (info) to 3 (scientifically surprising)", "message": "description",
    "created_at": "recorded (Unix time, s, UTC)", "acknowledged": "1 once reviewed by the researcher",
    "note": "researcher's note",
}


# ---------------------------------------------------------------------------------------------- public API
def make_release(catalog: Catalog, paths: Paths, tag: str, *, include_lightcurves: bool = True,
                 include_ri_curves: bool = True, overwrite: bool = False,
                 cfg: Mapping[str, Any] | None = None, sky_versions: Sequence[str] | None = None) -> Path:
    """Write the data release ``releases/<tag>/`` (see the module docstring) and return its directory.
    ``sky_versions`` restricts it to objects of those sky versions (None: all).

    Raises:
        ValueError: an invalid tag, or no done object of the requested sky versions. FileExistsError: the release
        exists and ``overwrite`` is false.
    """
    if not isinstance(tag, str) or not TAG_PATTERN.match(tag):
        raise ValueError(f"invalid release tag {tag!r}: letters, digits, '.', '_' and '-' (at most 64 characters)")
    cfg = cfg if cfg is not None else load_config(paths.root)
    target = paths.releases / tag
    if target.exists() and not overwrite:
        raise FileExistsError(f"release {tag} already exists at {target} (releases are immutable)")
    paths.releases.mkdir(parents=True, exist_ok=True)
    for stale in paths.releases.glob(f".{tag}.partial-*"):  # left behind by an interrupted release of this tag
        shutil.rmtree(stale, ignore_errors=True)
    work = paths.releases / f".{tag}.partial-{os.getpid()}"
    work.mkdir()
    try:
        snapshot_path = work / ".catalog-snapshot.sqlite"
        catalog.backup(snapshot_path)
        if sky_versions:
            _restrict_snapshot(snapshot_path, list(sky_versions))
        with Catalog(snapshot_path, root=catalog.root) as snapshot:
            _build(snapshot, paths, cfg, tag, work, include_lightcurves=include_lightcurves,
                   include_ri_curves=include_ri_curves)
        for p in (snapshot_path, Path(f"{snapshot_path}-wal"), Path(f"{snapshot_path}-shm")):
            p.unlink(missing_ok=True)
        _write_manifest(work)
        if target.exists():
            shutil.rmtree(target)
        os.replace(work, target)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    logger.info("data release %s written to %s", tag, target)
    return target


def _restrict_snapshot(snapshot_path: Path, sky_versions: list[str]) -> None:
    """Keep only objects of ``sky_versions`` in the release: every table is read through done objects, so marking
    the others "failed" in the temporary snapshot (never in the catalog) excludes them and their runs, events and
    transients."""
    import sqlite3

    marks = ",".join("?" * len(sky_versions))
    conn = sqlite3.connect(snapshot_path)
    try:
        kept = conn.execute(f"SELECT COUNT(*) FROM objects WHERE status = 'done' AND sky_version IN ({marks})",
                            sky_versions).fetchone()[0]
        if not kept:
            raise ValueError(f"no done objects of sky version(s) {', '.join(sky_versions)}")
        conn.execute(f"UPDATE objects SET status = 'failed' WHERE sky_version NOT IN ({marks})", sky_versions)
        conn.commit()
    finally:
        conn.close()


def verify_release(release_dir: Path | str) -> list[str]:
    """Check every file against ``MANIFEST.sha256``; returns the problems (empty when the release is intact)."""
    release_dir = Path(release_dir)
    problems = []
    listed = set()
    for line in (release_dir / MANIFEST).read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed.add(name)
        path = release_dir / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif _sha256(path) != digest:
            problems.append(f"checksum mismatch: {name}")
    extra = {p.relative_to(release_dir).as_posix() for p in release_dir.rglob("*") if p.is_file()} - listed - {MANIFEST}
    problems += [f"not in the manifest: {name}" for name in sorted(extra)]
    return problems


# ---------------------------------------------------------------------------------------------- building
def _build(snapshot: Catalog, paths: Paths, cfg: Mapping[str, Any], tag: str, work: Path, *,
           include_lightcurves: bool, include_ri_curves: bool) -> None:
    import pandas as pd  # imported on use: the daemon and the CLI need it only for releases

    objects = _objects_table(snapshot)
    released = set(objects["object_id"])
    runs = _table(snapshot, "SELECT r.ri_run_id, r.object_id, r.kind, r.path, r.n_subsets, r.n_points, r.max_order, "
                            "r.alpha, r.min_effect, r.created_at FROM ri_runs r JOIN objects o "
                            "ON o.object_id = r.object_id WHERE o.status = 'done' ORDER BY r.ri_run_id",
                  list(RI_RUN_COLUMNS) + ["path"])
    events = _table(snapshot, "SELECT e.event_id, e.ri_run_id, e.object_id, e.subset_json, e.order_k AS \"order\", "
                              "e.kind, e.step, e.step_lo, e.step_hi, e.ri_before, e.ri_after, e.peak_ri, e.confidence, "
                              "e.has_precursor, e.matches_gt, e.created_at FROM ri_events e JOIN objects o "
                              "ON o.object_id = e.object_id WHERE o.status = 'done' ORDER BY e.event_id",
                    list(EVENT_COLUMNS))
    transients = _table(snapshot, "SELECT t.transient_id, t.object_id, t.event_id, t.kind, t.severity, t.message, "
                                  "t.created_at, t.acknowledged, t.note FROM transients t JOIN objects o "
                                  "ON o.object_id = t.object_id WHERE o.status = 'done' ORDER BY t.transient_id",
                        list(TRANSIENT_COLUMNS))
    for col in ("has_precursor", "matches_gt"):
        events[col] = events[col].astype("boolean")
    runs["max_order"] = pd.to_numeric(runs["max_order"]).astype("Int64")
    transients["event_id"] = pd.to_numeric(transients["event_id"]).astype("Int64")

    missing_lc = missing_ri = 0
    lc_files: list[str] = []
    ri_files: list[str] = []
    lc_members: list[tuple[str, Path]] = []
    if include_lightcurves:
        for oid, stored in zip(objects["object_id"], objects.pop("lightcurve_path")):
            member = f"lightcurves/{oid}.npz"
            path = None if stored is None else snapshot.resolve_path(stored)
            if path is not None and path.is_file():
                lc_members.append((member, path))
            else:
                missing_lc += 1
        present = {m for m, _ in lc_members}
        objects["lightcurve_file"] = [f"lightcurves/{oid}.npz" if f"lightcurves/{oid}.npz" in present else None
                                      for oid in objects["object_id"]]
        lc_files = _tar(work / "lightcurves.tar.gz", lc_members)
    else:
        objects.pop("lightcurve_path")
        objects["lightcurve_file"] = None
    ri_members: list[tuple[str, Path]] = []
    files = []
    for run_id, oid, kind, stored in zip(runs["ri_run_id"], runs["object_id"], runs["kind"], runs.pop("path")):
        member = f"ricurves/{oid}_{kind}_{int(run_id)}.npz"
        path = None if stored is None else snapshot.resolve_path(stored)
        if include_ri_curves and path is not None and path.is_file():
            ri_members.append((member, path))
            files.append(member)
        else:
            missing_ri += int(include_ri_curves)
            files.append(None)
    runs["ricurve_file"] = files
    if include_ri_curves:
        ri_files = _tar(work / "ricurves.tar.gz", ri_members)
    from tycho.survey.worker import PATTERNS_TAG  # lazy: the worker module imports more than a release needs

    pat_members: list[tuple[str, Path]] = []
    if include_ri_curves:
        for oid in objects["object_id"]:
            path = paths.ri_path(oid, PATTERNS_TAG)
            if path.is_file():
                pat_members.append((f"patterns/{oid}.npz", path))
    present_pat = {m for m, _ in pat_members}
    objects["patterns_file"] = [f"patterns/{oid}.npz" if f"patterns/{oid}.npz" in present_pat else None
                                for oid in objects["object_id"]]
    pat_files = _tar(work / "patterns.tar.gz", pat_members) if pat_members else []
    tail = ["lightcurve_file", "patterns_file"]
    objects = objects[[c for c in objects.columns if c not in tail] + tail]

    for name, df in (("objects", objects), ("ri_runs", runs), ("ri_events", events), ("transients", transients)):
        df.to_parquet(work / f"{name}.parquet", index=False)
        df.to_csv(work / f"{name}.csv", index=False)

    sky_rel = Path(cfg["survey"]["sky"]).expanduser()
    sky_path = sky_rel if sky_rel.is_absolute() else paths.root / sky_rel
    if sky_path.is_file():
        shutil.copyfile(sky_path, work / "sky.yaml")
    else:
        logger.warning("release %s: sky file %s not found; not included", tag, sky_path)
    cal_rel = (cfg.get("ri") or {}).get("calibration_file")
    cal_path = None
    if cal_rel:
        cal_path = Path(cal_rel).expanduser()
        cal_path = cal_path if cal_path.is_absolute() else paths.root / cal_path
        if cal_path.is_file():
            shutil.copyfile(cal_path, work / "calibration.json")
        else:
            cal_path = None
    readme = _datasheet(tag, snapshot, objects, runs, events, transients, released=len(released),
                        lightcurves=len(lc_members) if include_lightcurves else None, missing_lc=missing_lc,
                        ricurves=len(ri_members) if include_ri_curves else None, missing_ri=missing_ri,
                        lc_files=lc_files, ri_files=ri_files, pat_files=pat_files, patterns=len(pat_members),
                        sky=sky_path.name if sky_path.is_file() else None, calibrated=cal_path is not None, work=work)
    (work / "README.md").write_text(readme, encoding="utf-8")


def _table(catalog: Catalog, sql: str, columns: Sequence[str]) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(catalog.query(sql), columns=list(columns))


def _objects_table(snapshot: Catalog) -> pd.DataFrame:
    import pandas as pd

    rows = snapshot.query(
        "SELECT o.object_id, o.task, o.region, o.sky_version, o.seed, o.spec_json, o.source, o.random_control, "
        "o.priority, o.attempts, o.created_at, o.started_at, o.finished_at, o.wall_time_s, o.device, o.n_params, "
        "o.code_version, s.json AS summary_json, l.path AS lightcurve_path, l.n_points AS lightcurve_points, "
        "c.label, c.tags_json AS label_tags_json, c.score AS label_score, c.classifier, c.created_at AS labelled_at "
        "FROM objects o LEFT JOIN summaries s ON s.object_id = o.object_id "
        "LEFT JOIN lightcurves l ON l.object_id = o.object_id "
        "LEFT JOIN classifications c ON c.id = (SELECT MAX(c2.id) FROM classifications c2 "
        "WHERE c2.object_id = o.object_id) WHERE o.status = 'done' ORDER BY o.created_at, o.object_id")
    records = []
    for r in rows:
        spec = json.loads(r.pop("spec_json"))
        summary = json.loads(r.pop("summary_json") or "{}")
        rec = {k: r[k] for k in ("object_id", "task", "region", "sky_version", "seed")}
        rec["task_cfg_json"] = json.dumps(spec.get("task_cfg") or {}, sort_keys=True, separators=(",", ":"))
        train = spec.get("train") or {}
        rec.update({f"train_{k}": train.get(k) for k in _TRAIN_FIELDS})
        rec.update({k: r[k] for k in ("source", "random_control", "priority", "attempts", "created_at", "started_at",
                                      "finished_at", "wall_time_s", "device", "n_params", "code_version",
                                      "lightcurve_points")})
        rec.update({k: summary.get(k) for k in SUMMARY_KEYS})
        rec.update({k: r[k] for k in ("label", "label_tags_json", "label_score", "classifier", "labelled_at")})
        rec["lightcurve_path"] = r["lightcurve_path"]
        records.append(rec)
    columns = (["object_id", "task", "region", "sky_version", "seed", "task_cfg_json"]
               + [f"train_{k}" for k in _TRAIN_FIELDS]
               + ["source", "random_control", "priority", "attempts", "created_at", "started_at", "finished_at",
                  "wall_time_s", "device", "n_params", "code_version", "lightcurve_points"]
               + list(SUMMARY_KEYS) + ["label", "label_tags_json", "label_score", "classifier", "labelled_at",
                                       "lightcurve_path"])
    df = pd.DataFrame(records, columns=columns)
    for key in SUMMARY_KEYS:  # typed columns (JSON null -> NaN / <NA>)
        if key == "summary_version":
            continue
        if key == "diverged":
            df[key] = df[key].astype("boolean")
        elif key in _COUNT_KEYS:
            df[key] = pd.to_numeric(df[key]).astype("Int64")
        else:
            df[key] = pd.to_numeric(df[key], errors="coerce").astype("float64")
    df["train_grad_clip"] = pd.to_numeric(df["train_grad_clip"]).astype("float64")
    df["lightcurve_points"] = pd.to_numeric(df["lightcurve_points"]).astype("Int64")
    return df


def _tar(path: Path, members: Iterable[tuple[str, Path]], *, part_bytes: int | None = None) -> list[str]:
    """Reproducible .tar.gz: sorted members, zero mtimes/owners, gzip header without timestamp or name. When the
    compressed size passes ``part_bytes`` (default :data:`ARCHIVE_PART_BYTES`) the archive continues in a new part:
    ``<stem>.part001.tar.gz``, ... (each a complete archive); one part keeps the plain name. Returns the file names."""
    limit = ARCHIVE_PART_BYTES if part_bytes is None else int(part_bytes)
    stem = path.name[: -len(".tar.gz")] if path.name.endswith(".tar.gz") else path.name
    ordered = sorted(members)
    written: list[Path] = []
    i = 0
    while True:
        part = path.parent / f"{stem}.part{len(written) + 1:03d}.tar.gz"
        with open(part, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
                tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            while i < len(ordered):
                member, src = ordered[i]
                data = src.read_bytes()
                info = tarfile.TarInfo(member)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
                i += 1
                gz.flush()
                if raw.tell() >= limit and i < len(ordered):
                    break
        written.append(part)
        if i >= len(ordered):
            break
    if len(written) == 1:
        written[0].replace(path)
        return [path.name]
    return [p.name for p in written]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_manifest(work: Path) -> None:
    names = sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file() and p.name != MANIFEST)
    lines = [f"{_sha256(work / name)}  {name}" for name in names]
    (work / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------- datasheet
def _counts(series: pd.Series) -> str:
    counts = series.value_counts().sort_index()
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"


def _columns_table(columns: Mapping[str, str], df: pd.DataFrame) -> str:
    lines = ["| column | type | description |", "|---|---|---|"]
    for col in df.columns:
        if col in columns:
            lines.append(f"| `{col}` | {df[col].dtype} | {columns[col]} |")
    return "\n".join(lines)


def _datasheet(tag: str, snapshot: Catalog, objects: pd.DataFrame, runs: pd.DataFrame, events: pd.DataFrame,
               transients: pd.DataFrame, *, released: int, lightcurves: int | None, missing_lc: int,
               ricurves: int | None, missing_ri: int, sky: str | None, calibrated: bool, work: Path,
               lc_files: Sequence[str] = ("lightcurves.tar.gz",),
               ri_files: Sequence[str] = ("ricurves.tar.gz",), pat_files: Sequence[str] = (),
               patterns: int = 0) -> str:
    from tycho.survey.worker import code_version  # git describe of the releasing code

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sizes = {p.name: p.stat().st_size for p in work.iterdir() if p.is_file() and not p.name.startswith(".")}
    files = [("objects.parquet / objects.csv", "objects ⋈ summaries ⋈ latest classification (done objects)"),
             ("ri_runs.parquet / ri_runs.csv", "RI spectroscopy runs"),
             ("ri_events.parquet / ri_events.csv", "RI events"),
             ("transients.parquet / transients.csv", "transients")]
    if lightcurves is not None:
        files.append((" / ".join(lc_files), f"{lightcurves} light curves (`lightcurves/<object_id>.npz`)"
                      + (f"; {missing_lc} missing on disk" if missing_lc else "")
                      + (f"; split into {len(lc_files)} parts" if len(lc_files) > 1 else "")))
    if ricurves is not None:
        files.append((" / ".join(ri_files), f"{ricurves} RI curves (`ricurves/<object_id>_<kind>_<ri_run_id>.npz`)"
                      + (f"; {missing_ri} missing on disk" if missing_ri else "")
                      + (f"; split into {len(ri_files)} parts" if len(ri_files) > 1 else "")))
    if pat_files:
        files.append((" / ".join(pat_files), f"{patterns} pattern-mode observations (`patterns/<object_id>.npz`: "
                      "steps, logodds, counts, pmf_D, pmf_U, pmf_S, truth, input_names, n_probe)"
                      + (f"; split into {len(pat_files)} parts" if len(pat_files) > 1 else "")))
    if sky:
        files.append(("sky.yaml", f"the sky definition (copied from `{sky}`)"))
    if calibrated:
        files.append(("calibration.json", "the RI instrument's error budget (noise floor, power, timing, artefacts)"))
    files += [("README.md", "this datasheet"), (MANIFEST, "SHA-256 checksums of all other files")]
    file_lines = "\n".join(f"| `{name}` | {desc} |" for name, desc in files)
    code_versions = sorted({str(v) for v in objects["code_version"].dropna()}) or ["unknown"]
    sky_versions = sorted({str(v) for v in objects["sky_version"].dropna()}) or ["none"]
    total_size = sum(sizes.values())
    return f"""# Tycho data release {tag}

A catalog of neural-network training runs ("objects") observed by the Tycho survey: every object with photometry
(light curves of loss, accuracy, norms, effective rank and eNTK statistics), a selection with RI spectroscopy
(time-resolved refined information: which subsets of variables start interacting, at which order, at which training
step), events and transients.

* Release: **{tag}** (format `{RELEASE_FORMAT}`), created {created} by Tycho {code_version()}.
* Objects: **{released}** done objects; by task: {_counts(objects["task"])}.
* By region: {_counts(objects["region"])}.
* Labels (latest classification): {_counts(objects["label"].fillna("unlabelled"))}.
* RI runs: {len(runs)} ({_counts(runs["kind"])}); RI events: {len(events)} ({_counts(events["kind"])});
  transients: {len(transients)} ({_counts(transients["kind"])}).
* Sky versions: {", ".join(sky_versions)}. Observing code versions: {", ".join(code_versions)}.
* Catalog schema version {SCHEMA_VERSION}. Total size {total_size / 1e6:.1f} MB.
* Licence: **{LICENSE}** ({LICENSE_URL}). You may share and adapt the data with attribution to the Tycho survey
  and this release tag.

## Files

| file | content |
|---|---|
{file_lines}

Verify the release with `shasum -a 256 -c {MANIFEST}` (macOS) or `sha256sum -c {MANIFEST}` (Linux).
Archives larger than {ARCHIVE_PART_BYTES // 1_000_000} MB are split into parts (`<name>.part001.tar.gz`, ...), each a
complete tar.gz, so that every file stays under GitHub's 100 MB limit. Extract all parts into one directory:
`for f in lightcurves*.tar.gz ricurves*.tar.gz; do tar -xzf "$f"; done`.

## Loading

```python
import glob, io, tarfile
import numpy as np, pandas as pd
objects = pd.read_parquet("objects.parquet")
events = pd.read_parquet("ri_events.parquet")

def member(name, pattern):  # works whether or not the archive was split into parts
    for part in sorted(glob.glob(pattern)):
        with tarfile.open(part) as tar:
            try:
                return io.BytesIO(tar.extractfile(name).read())
            except KeyError:
                continue
    raise KeyError(name)

lc = np.load(member(objects.lightcurve_file[0], "lightcurves*.tar.gz"))
train_loss = lc["train_loss"]          # aligned with lc["step"]
```

RI curves load with `tycho.analysis.ricurve.RICurve.load(path)` after extraction (arrays `steps`, `ri`,
`ri_plugin`, `p`, `q`, `n_eff` and a JSON header with the subsets' variable names). Curves from sky v1 on also hold
`fourier`: the Walsh coefficient of the model's soft binary output on each subset's inputs, on the same probe rows
and checkpoints (NaN where undefined; absent in older curves).

## Units and conventions

* **Steps** count optimizer updates; step 0 is the initial model.
* **Losses**: cross-entropy in nats (classification and token tasks); importance-weighted squared error
  (`toy_superposition`). **Accuracies** in [0, 1]; NaN when the task has none.
* **Information**: local refined information (RI) in **nats** (natural logarithms). `ri_corrected` /
  `min_effect` refer to the bias-corrected estimate `max(0, RI_plugin − df/(2N))`. A subset is *detected* at a
  checkpoint when its Benjamini–Hochberg q-value ≤ `alpha` and its corrected RI ≥ `min_effect`.
* **Times**: Unix time in seconds (UTC); wall times in seconds.
* Missing values are empty in CSV and null in parquet.

## Provenance

Each object is fully specified by its configuration (`task_cfg_json`, `train_*`, `seed`, `sky_version`); its id is
a hash of exactly these. Training is deterministic on CPU for a fixed thread count, so any object can be replayed
bit for bit with `tycho.train.replay.replay_object` (follow-up RI runs were taken that way). Photometry summaries
(`summary_version`) and classifications (rule set in `classifier`) are computed from the light curves by
`tycho.analysis.summarize` and `tycho.analysis.classify`; events and transients by `tycho.analysis.events`. The RI
estimator's error budget (noise floor, power, timing resolution, discretisation artefacts) is in
`calibration.json` when present: interpret RI values and detections relative to it.

## Schema

### objects

Photometry summary columns (`final_*`, `t_*_acc*`, `grok_*`, `plateau_*`, `*drop*`, `test_bump*`, ...) are the keys
of `tycho.analysis.summarize.summarize` (see its docstring for definitions); step-valued keys are in steps.
`train_*` columns are the fields of `tycho.train.spec.TrainConfig`.

{_columns_table(OBJECT_COLUMNS, objects)}

### ri_runs

{_columns_table(RI_RUN_COLUMNS, runs)}

### ri_events

{_columns_table(EVENT_COLUMNS, events)}

### transients

{_columns_table(TRANSIENT_COLUMNS, transients)}

### Light-curve files

`lightcurves/<object_id>.npz` holds one array per key, aligned with `step`: `train_loss`, `test_loss`,
`train_acc`, `test_acc`, `param_norm`, `grad_norm`, `update_norm`, `dist_from_init`, `erank_weights`,
`erank_hidden`, `rep_cka_prev`, `ntk_trace`, `ntk_top_eig`, `ntk_erank`, `ntk_align`, `ntk_velocity`, `lr`,
`wall_s`, plus task-specific metrics `x_<name>` (NaN where not measured).
"""
