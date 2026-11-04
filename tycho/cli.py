"""Command-line interface (SPEC §8): ``tycho [--root DIR] [-v] <command> ...``.

The survey root is ``--root``, else ``$TYCHO_ROOT``, else the current directory. Commands:

==========================================  ==========================================================================
``init``                                    create the root's layout, ``configs/sky_v0.yaml`` and ``tycho.yaml``
                                            (existing files are never overwritten) and the catalog
``calibrate [--quick|--full] [--jobs N]``   run the instrument calibration into ``reports/calibration/`` (full by
                                            default; resumable)
``run-object --task T --seed S ...``        observe one ad-hoc object (``--set k=v`` task config, ``--train k=v``
                                            TrainConfig fields, values parsed as YAML; ``--ri`` in-situ RI) and print
                                            a summary with its classification and RI events; nothing is recorded in
                                            the catalog (``--out DIR`` saves the light curve, RI curve and summary)
``survey enqueue [--sky PATH] [--limit N]`` enqueue unobserved coarse-grid objects now (the daemon also tops up)
``survey run [--workers N] [--hours H]``    run the survey daemon in the foreground (what launchd runs)
``  [--max-objects N]``
``survey status [--json]``                  counts, throughput, queue, workers, latest transients
``followup run [--n N]``                    run RI follow-ups in this process
``agent nightly|weekly [--dry-run]``        the agent runs (``--dry-run`` needs no credentials)
``release --tag TAG``                       write the data release ``releases/<TAG>/``
``dashboard``                               regenerate ``reports/dashboard.html`` now
``backup``                                  online catalog backup into ``backups/`` (with retention)
==========================================  ==========================================================================

Exit codes: 0 success, 1 failure (an error, a failed object, a broken release), 2 usage errors, 3 unavailable
(another daemon holds the survey lock; the agent is disabled or has no credentials), 130 interrupted (Ctrl-C).

Logging: warnings go to stderr (``-v`` adds info, ``-vv`` debug). Commands other than ``survey run`` also log at
info level to ``logs/cli.log`` (rotated) when the root has a ``logs/`` directory; ``survey run`` logs to
``logs/survey.log`` (see :mod:`tycho.survey.daemon`). Heavy dependencies (torch, matplotlib) are imported only by
the commands that need them, so ``tycho survey status`` is quick.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import math
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

import tycho
from tycho.config import DEFAULT_CONFIG, Paths, default_root, load_config

logger = logging.getLogger(__name__)

__all__ = ["CLIError", "build_parser", "format_status", "main", "parse_assignments", "parse_value"]

EXIT_OK, EXIT_FAILURE, EXIT_USAGE, EXIT_UNAVAILABLE, EXIT_INTERRUPTED = 0, 1, 2, 3, 130

CLI_LOG = "cli.log"
"""Log file of the non-daemon commands in ``logs/`` (rotated: CLI_LOG_MAX_BYTES × CLI_LOG_BACKUPS)."""
CLI_LOG_MAX_BYTES = 5_000_000
CLI_LOG_BACKUPS = 3

N_STATUS_TRANSIENTS = 5
"""Transients listed by ``tycho survey status``."""
N_PRINTED_EVENTS = 20
"""RI events listed by ``tycho run-object``."""

REPO_ROOT = Path(__file__).resolve().parent.parent
"""The checkout this package was imported from (holds ``configs/`` when running from a source tree)."""

_EXPONENT_FLOAT = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+")
"""Floats in exponent notation without a decimal point, which YAML 1.1 reads as strings (``1e-3``)."""


class CLIError(Exception):
    """A user error with a message for the terminal (exit code 2)."""


# ---------------------------------------------------------------------------------------------- parsing helpers
def parse_value(text: str) -> Any:
    """A command-line value parsed as YAML (``128`` → int, ``null`` → None, ``[1, 2]`` → list, ``relu`` → str),
    with exponent floats such as ``1e-3`` read as floats (YAML 1.1 would keep them as strings)."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, str) and _EXPONENT_FLOAT.fullmatch(value.strip()):
        return float(value)
    return value


def parse_assignments(items: Sequence[str] | None, option: str) -> dict[str, Any]:
    """``["k=v", ...]`` → ``{k: parse_value(v)}``; a later assignment of the same key wins."""
    out: dict[str, Any] = {}
    for item in items or ():
        key, sep, text = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise CLIError(f"{option}: expected key=value, got {item!r}")
        out[key] = parse_value(text)
    return out


# ---------------------------------------------------------------------------------------------- shared helpers
def _catalog(paths: Paths, cfg: Mapping[str, Any], *, create: bool = False) -> Any:
    """The survey catalog; without ``create`` a missing catalog is a usage error (the root is not initialised)."""
    from tycho.survey.catalog import Catalog

    if not create and not paths.db.exists():
        raise CLIError(f"no catalog at {paths.db}: is {paths.root} a survey root? Run `tycho --root {paths.root} init`")
    return Catalog(paths.db, root=paths.root, max_attempts=int(cfg["survey"]["max_attempts"]))


def _sky_path(paths: Paths, cfg: Mapping[str, Any], override: str | None = None) -> Path:
    path = Path(override if override is not None else cfg["survey"]["sky"]).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path if override is not None else paths.root / path)


@contextmanager
def _stop_on_signals() -> Iterator[threading.Event]:
    """An event set by SIGINT / SIGTERM while the block runs (previous handlers restored afterwards)."""
    stop = threading.Event()
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda signum, frame: stop.set())
    try:
        yield stop
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None or isinstance(x, bool):
        return {None: "–", True: "yes", False: "no"}[x]
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "–"
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.{digits}g}"


def _ago(seconds: Any) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(s):
        return "–"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 172_800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86_400:.1f} d"


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str))


# ---------------------------------------------------------------------------------------------- init
def _repo_text(relative: str) -> str | None:
    """Text of a file of the source checkout (None when the package is installed without it)."""
    path = REPO_ROOT / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _fallback_tycho_yaml() -> str:
    """``tycho.yaml`` from the built-in defaults, used when ``configs/tycho.example.yaml`` is unavailable.
    ``ri.min_effect`` is left out on purpose: an explicit value would override the calibrated default."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["ri"].pop("min_effect", None)
    header = ("# Tycho survey configuration (generated from the built-in defaults; see configs/tycho.example.yaml\n"
              "# in the Tycho repository for a commented version).\n")
    return header + yaml.safe_dump(cfg, sort_keys=False)


def cmd_init(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.survey.catalog import Catalog
    from tycho.survey.sky import Sky

    paths.ensure()
    sky_rel = DEFAULT_CONFIG["survey"]["sky"]
    files = [(paths.root / sky_rel, _repo_text("configs/sky_v0.yaml") or SKY_V0_YAML),
             (paths.root / "tycho.yaml", _repo_text("configs/tycho.example.yaml") or _fallback_tycho_yaml())]
    for path, text in files:
        if path.exists():
            print(f"kept     {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        print(f"created  {path}")
    cfg = load_config(paths.root)  # both files must parse
    sky = Sky.from_yaml(_sky_path(paths, cfg))
    with Catalog(paths.db, root=paths.root) as catalog:
        n = catalog.stats()["objects"]["total"]
    print(f"catalog  {paths.db} ({n} objects)")
    print(f"sky      {sky.version}: {len(sky.regions)} regions, "
          f"{sum(r.grid_size * r.seeds for r in sky.regions)} coarse objects")
    print(f"Survey root ready at {paths.root}. Next: tycho calibrate --full --jobs N, then tycho survey run.")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------- calibrate
def cmd_calibrate(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.calibration import CalibrationConfig, recommended_min_effect, run_calibration, write_report

    cfg = load_config(paths.root)
    preset = CalibrationConfig.quick() if args.quick else CalibrationConfig.full()
    out = paths.reports / "calibration"
    print(f"calibration ({preset.name}) with {args.jobs} job(s) into {out}; interrupted runs resume", flush=True)
    last = [-1]

    def progress(done: int, total: int) -> None:
        pct = int(100 * done / max(total, 1))
        if pct // 10 > last[0] // 10 or done == total:
            last[0] = pct
            print(f"  {done}/{total} work items ({pct}%)", file=sys.stderr, flush=True)

    results = run_calibration(preset, out, n_jobs=args.jobs, progress=progress)
    report = write_report(results, out)
    n_probe = int(cfg["ri"]["n_probe"])
    value = recommended_min_effect(out / "calibration.json", n_probe)
    print(f"report   {report}")
    print(f"min_effect at n_probe={n_probe}: {value:.3g} nats (used by the survey unless ri.min_effect is set)")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------- run-object
def cmd_run_object(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.analysis.classify import classify
    from tycho.analysis.events import detect_events, find_transients, transient_min_peak
    from tycho.survey.followup import resolve_ri_config
    from tycho.tasks import make_task
    from tycho.train.device import configure_torch
    from tycho.train.runner import run_object
    from tycho.train.spec import ObjectSpec, TrainConfig

    cfg = load_config(paths.root)
    survey = cfg["survey"]
    try:
        spec = ObjectSpec(task=args.task, task_cfg=parse_assignments(args.set, "--set"), seed=int(args.seed),
                          train=TrainConfig.from_dict(parse_assignments(args.train, "--train")),
                          device=str(args.device or survey.get("device") or "cpu"))
        make_task(spec.task, spec.task_cfg, spec.seed)  # unknown task or config keys: a usage error, not a run
    except (TypeError, ValueError) as exc:
        raise CLIError(str(exc)) from exc
    ri = resolve_ri_config(cfg, None, root=paths.root) if args.ri else None
    configure_torch(int(args.threads or survey.get("torch_threads") or 1))
    out = Path(args.out).expanduser().resolve() if args.out else None
    work = Path(tempfile.mkdtemp(prefix="tycho-run-"))
    try:
        result = run_object(spec, work_dir=work, ri=ri, resume=False)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    summary = result.summary or {}
    cls = classify(summary) if summary else None
    curve = result.ri_curve
    events = detect_events(curve, alpha=ri.alpha) if curve is not None and ri is not None else []
    transients = (find_transients(events, summary, cls, min_peak=transient_min_peak(curve))
                  if cls is not None else [])
    report = {
        "object_id": result.object_id, "status": result.status, "task": spec.task, "seed": spec.seed,
        "task_cfg": spec.task_cfg, "steps": spec.train.steps, "wall_time_s": result.wall_time_s,
        "device": result.device, "n_params": result.n_params, "error": result.error,
        "summary": {k: summary.get(k) for k in ("final_train_loss", "final_test_loss", "final_train_acc",
                                                 "final_test_acc", "t_train_acc99", "t_test_acc90", "grok_delay",
                                                 "plateau_frac", "n_drops", "loss_spikes", "diverged")},
        "classification": None if cls is None else cls.to_dict(),
        "ri": None if curve is None else {"n_subsets": curve.S, "n_points": curve.T,
                                          "max_order": curve.meta.get("max_order"),
                                          "min_effect": curve.meta.get("min_effect"), "alpha": ri.alpha},
        "events": [e.to_dict() for e in events],
        "transients": [t.to_dict() for t in transients],
    }
    if out is not None:
        _save_run(out, result, report)
        report["out"] = str(out)
    if args.json:
        _print_json(report)
    else:
        print(_format_run(report))
    return EXIT_OK if result.status == "done" else EXIT_FAILURE


def _save_run(out: Path, result: Any, report: Mapping[str, Any]) -> None:
    from tycho.survey.worker import save_light_curve

    out.mkdir(parents=True, exist_ok=True)
    if result.light_curve:
        save_light_curve(out / "lightcurve.npz", result.light_curve)
    if result.ri_curve is not None:
        result.ri_curve.save(out / "ricurve.npz")
    full = dict(report)
    full["summary"] = result.summary
    (out / "summary.json").write_text(json.dumps(full, indent=1, sort_keys=True, default=_json_default) + "\n",
                                      encoding="utf-8")


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _format_run(r: Mapping[str, Any]) -> str:
    s = r["summary"]
    lines = [f"object {r['object_id']}: {r['task']} seed {r['seed']}, {r['steps']} steps: {r['status']} in "
             f"{r['wall_time_s']:.1f} s on {r['device']} ({r['n_params']} parameters)"]
    if r["status"] == "failed" and r.get("error"):
        lines.append("error: " + str(r["error"]).strip().splitlines()[-1])
    lines.append(f"  final train loss {_fmt(s['final_train_loss'])}, test loss {_fmt(s['final_test_loss'])}, "
                 f"train acc {_fmt(s['final_train_acc'])}, test acc {_fmt(s['final_test_acc'])}")
    lines.append(f"  t_train_acc99 {_fmt(s['t_train_acc99'])}, t_test_acc90 {_fmt(s['t_test_acc90'])}, plateau_frac "
                 f"{_fmt(s['plateau_frac'], 3)}, sharp drops {_fmt(s['n_drops'])}, diverged {_fmt(s['diverged'])}")
    cls = r["classification"]
    if cls is not None:
        tags = f"; tags {', '.join(cls['tags'])}" if cls["tags"] else ""
        lines.append(f"  classification: {cls['label']} (score {cls['score']:.2f}, {cls['ruleset']}){tags}")
    ri = r["ri"]
    if ri is not None:
        lines.append(f"  RI: {ri['n_subsets']} subsets x {ri['n_points']} checkpoints, max order {ri['max_order']}, "
                     f"alpha {_fmt(ri['alpha'])}, min_effect {_fmt(ri['min_effect'], 3)} nats")
        events = r["events"]
        lines.append(f"  RI events: {len(events)}")
        for e in events[:N_PRINTED_EVENTS]:
            gt = " [ground truth]" if e.get("matches_ground_truth") else ""
            lines.append(f"    {e['kind']:<12} {{{', '.join(e['subset_names'])}}} order {e['order']} at step "
                         f"{e['step']} (onset in ({e['step_lo']}, {e['step_hi']}]), peak {_fmt(e['peak_ri'], 3)} "
                         f"nats, confidence {_fmt(e['confidence'], 3)}{gt}")
        if len(events) > N_PRINTED_EVENTS:
            lines.append(f"    ... {len(events) - N_PRINTED_EVENTS} more (use --json)")
    for t in r["transients"]:
        lines.append(f"  transient {t['kind']} (severity {t['severity']}): {t['message']}")
    if r.get("out"):
        lines.append(f"  saved to {r['out']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- survey
def cmd_survey_enqueue(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.survey.scheduler import Scheduler, SchedulerPolicy
    from tycho.survey.sky import Sky

    cfg = load_config(paths.root)
    sky_path = _sky_path(paths, cfg, args.sky)
    configured = _sky_path(paths, cfg)
    if args.sky is not None and sky_path.resolve() != configured.resolve():
        logger.warning("enqueueing from %s, but the daemon observes survey.sky = %s: regions it does not know get no "
                       "in-situ RI and no follow-ups", sky_path, configured)
    sky = Sky.from_yaml(sky_path)
    paths.ensure()
    with _catalog(paths, cfg, create=True) as catalog:
        scheduler = Scheduler(catalog, sky, SchedulerPolicy.from_dict(cfg.get("scheduler")),
                              device=str(cfg["survey"].get("device") or "cpu"))
        total = sum(r.grid_size * scheduler.initial_seeds(r) for r in sky.regions)
        limit = total if args.limit is None else int(args.limit)
        if limit < 0:
            raise CLIError("--limit must be >= 0")
        before = {r: sum(c.values()) for r, c in catalog.stats()["by_region"].items()}
        # Coarse coverage only, in the scheduler's deterministic order and region interleaving (top_up would wait for
        # the queue to run low and add boundary refinement).
        added = scheduler.enqueue_coarse(limit) if limit else 0
        after = catalog.stats()
    print(f"enqueued {added} object(s) from sky {sky.version} ({sky_path}); queue now {after['queue']}")
    for region, counts in sorted(after["by_region"].items()):
        new = sum(counts.values()) - before.get(region, 0)
        if new:
            print(f"  {region}: +{new}")
    return EXIT_OK


def cmd_survey_run(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.survey.daemon import SurveyLockError, run_survey

    try:
        summary = run_survey(paths.root, hours=args.hours, max_objects=args.max_objects, n_workers=args.workers)
    except SurveyLockError as exc:
        print(f"tycho: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    objects = summary["stats"]["objects"]
    counters = summary["counters"]
    print(f"survey stopped after {_ago(summary['elapsed_s'])}: {summary['stop_reason']}")
    print(f"  objects: {objects['done']} done, {objects['queued']} queued, {objects['interrupted']} interrupted, "
          f"{objects['failed']} failed, {objects['running']} running")
    print("  " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in counters.items()))
    return EXIT_OK


def _daemon_section(paths: Paths) -> dict[str, Any]:
    """Daemon state for status output: pid if a daemon holds the lock, and the daemon part of ``status.json``."""
    from tycho.survey.daemon import STATUS_FILE, running_daemon_pid

    pid = running_daemon_pid(paths)
    out: dict[str, Any] = {"running": pid is not None, "pid": pid}
    path = paths.reports / STATUS_FILE
    try:
        written = json.loads(path.read_text(encoding="utf-8"))
        out["status_file_age_s"] = max(0.0, time.time() - path.stat().st_mtime)
    except (OSError, ValueError):
        return out
    if pid is not None:  # a stopped daemon's last status is history, not current state
        for key in ("state", "daemon", "processes", "notifications"):
            if key in written:
                out[key] = written[key]
    return out


def cmd_survey_status(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.survey.dashboard import collect_status

    cfg = load_config(paths.root)
    with _catalog(paths, cfg) as catalog:
        status = collect_status(catalog, paths, cfg, n_latest=N_STATUS_TRANSIENTS)
    status["daemon_process"] = _daemon_section(paths)
    if args.json:
        _print_json(status)
    else:
        print(format_status(status))
    return EXIT_OK


def format_status(status: Mapping[str, Any]) -> str:
    """Terminal rendering of a :func:`tycho.survey.dashboard.collect_status` dict (plus ``daemon_process``)."""
    o, tp, fu, tr = status["objects"], status["throughput"], status["followups"], status["transients"]
    dp = status.get("daemon_process") or {}
    lines = [f"Tycho survey at {status['root']} ({status['generated_at_utc']})"]
    if dp.get("running"):
        d = dp.get("daemon") or {}
        lines.append(f"daemon:      running (pid {dp.get('pid')}, {d.get('n_workers', '?')} workers, up "
                     f"{_ago(d.get('uptime_s'))}, sky {d.get('sky_version', '?')}); status.json written "
                     f"{_ago(dp.get('status_file_age_s'))} ago")
    else:
        lines.append("daemon:      not running")
    lines.append(f"objects:     {o['total']} total: {o['done']} done, {o['queued']} queued, {o['running']} running, "
                 f"{o['interrupted']} interrupted, {o['failed']} failed")
    lines.append(f"queue:       {status['queue']} claimable")
    median = tp.get("median_wall_time_s_24h")
    lines.append(f"throughput:  {tp['done_last_1h']} done in the last hour, {tp['done_last_24h']} in 24 h "
                 f"({tp['objects_per_hour_24h']:.1f}/h), {tp['failed_last_24h']} failed in 24 h; median wall time "
                 f"{_fmt(median, 3)} s")
    lines.append(f"follow-ups:  {fu['requested']} requested, {fu['running']} running, {fu['done']} done, "
                 f"{fu['failed']} failed")
    lines.append(f"RI:          {status['ri_runs']} runs, {status['ri_events']} events; transients {tr['total']} "
                 f"({tr['pending_notification']} not yet notified, {tr['unacknowledged']} unacknowledged)")
    labels = status.get("labels") or {}
    if labels:
        lines.append("labels:      " + ", ".join(f"{k} {v}" for k, v in sorted(labels.items(), key=lambda kv: -kv[1])))
    disk = status.get("disk") or {}
    if "free_gb" in disk:
        low = "  LOW: claiming paused" if disk.get("low") else ""
        lines.append(f"disk:        {disk['free_gb']:.1f} GB free (minimum {disk['min_free_gb']:g} GB){low}")
    backup = status.get("latest_backup")
    lines.append("backup:      none" if not backup else
                 f"backup:      {backup['file']} ({_ago(3600.0 * float(backup['age_h']))} ago)")
    workers = status.get("workers") or []
    lines.append(f"workers:     {len(workers)} reporting" + ("" if workers else " (none)"))
    for w in workers:
        flag = "STALE" if w.get("stale") else "ok"
        lines.append(f"  {w['worker_id']:<10} {flag:<5} {str(w.get('status') or ''):<16} "
                     f"{w.get('object_id') or '':<16}  pid {w.get('pid')}  {_ago(w.get('age_s'))} ago")
    trs = status.get("latest_transients") or []
    lines.append("latest transients:" + ("" if trs else " none"))
    for t in trs:
        lines.append(f"  [{t['severity']}] {t['kind']:<16} {t['object_id']} ({t['task']}, {t['region']}): "
                     f"{t['message']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- follow-ups
def cmd_followup_run(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.survey.worker import run_followups

    cfg = load_config(paths.root)
    if int(args.n) < 1:
        raise CLIError("--n must be >= 1")
    _catalog(paths, cfg).close()  # the root must be initialised
    with _stop_on_signals() as stop:
        outcomes = run_followups(paths, cfg, int(args.n), stop_flag=stop.is_set)
    if not outcomes:
        print("no follow-up to run (none requested and none selectable within the daily budget)")
        return EXIT_OK
    with _catalog(paths, cfg) as catalog:
        for oid, outcome in outcomes:
            rows = catalog.query("SELECT r.ri_run_id, r.n_subsets, r.n_points, (SELECT COUNT(*) FROM ri_events e "
                                 "WHERE e.ri_run_id = r.ri_run_id) AS n_events FROM ri_runs r WHERE r.object_id = ? "
                                 "AND r.kind = 'followup'", (oid,))
            detail = (f": ri_run {rows[0]['ri_run_id']}, {rows[0]['n_subsets']} subsets x {rows[0]['n_points']} "
                      f"checkpoints, {rows[0]['n_events']} events") if outcome == "done" and rows else ""
            print(f"{oid} {outcome}{detail}")
    return EXIT_FAILURE if any(o == "failed" for _, o in outcomes) else EXIT_OK


# ---------------------------------------------------------------------------------------------- agent
def cmd_agent(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.agent import AgentUnavailable, run_nightly, run_weekly

    cfg = load_config(paths.root)
    _catalog(paths, cfg).close()
    run = run_nightly if args.kind == "nightly" else run_weekly
    try:
        path = run(paths.root, dry_run=args.dry_run)
    except AgentUnavailable as exc:
        print(f"tycho: agent unavailable: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    if args.dry_run:
        print(f"dry run: the first request of the {args.kind} run was written to {path}")
    else:
        print(f"{args.kind} run finished: {path}")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------- release, backup
def cmd_release(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.survey.release import make_release, verify_release

    cfg = load_config(paths.root)
    with _catalog(paths, cfg) as catalog:
        try:
            target = make_release(catalog, paths, args.tag, include_lightcurves=not args.no_lightcurves,
                                  overwrite=args.overwrite, cfg=cfg)
        except (ValueError, FileExistsError) as exc:
            raise CLIError(str(exc)) from exc
    problems = verify_release(target)
    files = sorted(p.name for p in target.iterdir() if p.is_file())
    print(f"release {args.tag} written to {target}: {', '.join(files)}")
    if problems:
        print("manifest check FAILED: " + "; ".join(problems), file=sys.stderr)
        return EXIT_FAILURE
    print(f"manifest verified ({len(files) - 1} files)")
    return EXIT_OK


def cmd_dashboard(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.survey.dashboard import collect_status, write_dashboard

    cfg = load_config(paths.root)
    daemon = _daemon_section(paths)
    extra = {k: daemon[k] for k in ("state", "daemon", "processes", "notifications") if k in daemon}
    with _catalog(paths, cfg) as catalog:
        status = collect_status(catalog, paths, cfg, extra=extra)
        path = write_dashboard(catalog, paths, status=status)
    print(f"dashboard written to {path}")
    return EXIT_OK


def cmd_backup(args: argparse.Namespace, paths: Paths) -> int:
    from tycho.survey.daemon import backup_catalog

    cfg = load_config(paths.root)
    with _catalog(paths, cfg) as catalog:
        dest = backup_catalog(catalog, paths, int(cfg["survey"]["backups_keep"]))
    print(f"catalog backed up to {dest} (keeping the newest {int(cfg['survey']['backups_keep'])})")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------- parser and main
def build_parser() -> argparse.ArgumentParser:
    """The argument parser (each leaf command sets ``func``)."""
    p = argparse.ArgumentParser(prog="tycho", description="Tycho: a 24x7 sky survey of neural-network training "
                                "dynamics with refined information (RI) spectroscopy.")
    p.add_argument("--root", help="survey root (default: $TYCHO_ROOT or the current directory)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="more log output on stderr (-v info, -vv debug)")
    p.add_argument("--version", action="version", version=f"tycho {tycho.__version__}")
    sub = p.add_subparsers(dest="command", metavar="command", required=True)

    s = sub.add_parser("init", help="create the survey root layout, sky, tycho.yaml and catalog")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("calibrate", help="calibrate the RI instrument (writes reports/calibration/)")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="smoke-test sizes (~20 s; large Monte Carlo errors)")
    g.add_argument("--full", action="store_true", help="production sizes (the default; hours, resumable)")
    s.add_argument("--jobs", type=int, default=1, help="worker processes (<= 0: all cores; default 1)")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("run-object", help="observe one ad-hoc object and print its summary (not recorded)")
    s.add_argument("--task", required=True, help="task name (sparse_parity, markov_lang, modular_addition, ...)")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="task configuration value (YAML-parsed)")
    s.add_argument("--train", action="append", metavar="KEY=VALUE", help="TrainConfig field (YAML-parsed)")
    s.add_argument("--seed", type=int, required=True, help="training seed")
    s.add_argument("--ri", action="store_true", help="in-situ RI spectroscopy (settings from the config's ri section)")
    s.add_argument("--device", choices=("cpu", "mps", "auto"), help="device policy (default: survey.device)")
    s.add_argument("--threads", type=int, help="torch threads (default: survey.torch_threads)")
    s.add_argument("--out", help="directory for lightcurve.npz, ricurve.npz and summary.json")
    s.add_argument("--json", action="store_true", help="print the result as JSON")
    s.set_defaults(func=cmd_run_object)

    survey = sub.add_parser("survey", help="enqueue, run and inspect the survey")
    ssub = survey.add_subparsers(dest="survey_command", metavar="survey_command", required=True)
    s = ssub.add_parser("enqueue", help="enqueue unobserved coarse-grid objects now")
    s.add_argument("--sky", help="sky file (default: survey.sky of tycho.yaml)")
    s.add_argument("--limit", type=int, help="at most about this many objects (whole points; default: all)")
    s.set_defaults(func=cmd_survey_enqueue)
    s = ssub.add_parser("run", help="run the survey daemon in the foreground (supervisor + workers)")
    s.add_argument("--workers", type=int, help="worker processes (default: survey.n_workers or cores - 2)")
    s.add_argument("--hours", type=float, help="stop after this many hours")
    s.add_argument("--max-objects", type=int, help="stop after claiming this many objects")
    s.set_defaults(func=cmd_survey_run)
    s = ssub.add_parser("status", help="counts, throughput, queue, workers, latest transients")
    s.add_argument("--json", action="store_true", help="print the full status as JSON")
    s.set_defaults(func=cmd_survey_status)

    followup = sub.add_parser("followup", help="RI follow-ups by replay")
    fsub = followup.add_subparsers(dest="followup_command", metavar="followup_command", required=True)
    s = fsub.add_parser("run", help="run follow-ups in this process")
    s.add_argument("--n", type=int, default=1, help="number of follow-ups (default 1)")
    s.set_defaults(func=cmd_followup_run)

    agent = sub.add_parser("agent", help="the survey agent")
    asub = agent.add_subparsers(dest="kind", metavar="kind", required=True)
    for kind, what in (("nightly", "QC, classification review, proposals, survey log"), ("weekly", "weekly report")):
        s = asub.add_parser(kind, help=f"the {kind} run: {what}")
        s.add_argument("--dry-run", action="store_true", help="write the first request to reports/agent/ only")
        s.set_defaults(func=cmd_agent)

    s = sub.add_parser("release", help="write a data release to releases/<TAG>/")
    s.add_argument("--tag", required=True, help="release tag, e.g. DR1")
    s.add_argument("--no-lightcurves", action="store_true", help="leave out the light-curve archive")
    s.add_argument("--overwrite", action="store_true", help="replace an existing release with this tag")
    s.set_defaults(func=cmd_release)

    s = sub.add_parser("dashboard", help="regenerate reports/dashboard.html")
    s.set_defaults(func=cmd_dashboard)

    s = sub.add_parser("backup", help="online backup of the catalog into backups/")
    s.set_defaults(func=cmd_backup)
    return p


def _configure_logging(verbosity: int, paths: Paths, *, log_file: bool) -> list[logging.Handler]:
    """stderr at WARNING (INFO with -v, DEBUG with -vv); the rotating ``logs/cli.log`` at INFO if ``log_file`` and
    the root has a ``logs/`` directory. Returns the handlers added to the root logger."""
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(level)
    stderr.setFormatter(fmt)
    handlers: list[logging.Handler] = [stderr]
    if log_file and paths.logs.is_dir():
        fh = logging.handlers.RotatingFileHandler(paths.logs / CLI_LOG, maxBytes=CLI_LOG_MAX_BYTES,
                                                  backupCount=CLI_LOG_BACKUPS, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s"))
        handlers.append(fh)
    root_logger = logging.getLogger()
    for h in handlers:
        root_logger.addHandler(h)
    root_logger.setLevel(min(h.level for h in handlers))
    logging.captureWarnings(True)
    return handlers


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``tycho`` console script; returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else default_root()
    paths = Paths(root)
    is_daemon = getattr(args, "func", None) is cmd_survey_run
    handlers = _configure_logging(args.verbose, paths, log_file=not is_daemon)
    func: Callable[[argparse.Namespace, Paths], int] = args.func
    try:
        return int(func(args, paths))
    except CLIError as exc:
        print(f"tycho: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("tycho: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception:  # the traceback goes to stderr and logs/cli.log
        logger.error("tycho %s failed", args.command, exc_info=True)
        return EXIT_FAILURE
    finally:
        root_logger = logging.getLogger()
        for h in handlers:
            root_logger.removeHandler(h)
            h.close()


# ---------------------------------------------------------------------------------------------- embedded sky v0
SKY_V0_YAML = r'''# Sky v0 of the Tycho survey: small objects (seconds to minutes on one CPU core).
# Each region is a grid over "knobs". A knob name is a task-config key, or "train.<TrainConfig field>".
# kind: categorical | ordinal | log | linear.  refine: the scheduler may insert midpoints near boundaries.
# ri_policy: always (in-situ RI spectroscopy for every object) | followup (RI only by replay when selected) | never
version: sky-v0
description: >
  First sky. Positive controls with known interaction structure (sparse parity, Markov language with planted
  orders), known phenomena (grokking on modular addition, staged in-context learning, superposition), and a broad
  parity grid for boundary hunting.
regions:
  # ---------------------------------------------------------------- positive controls (always observed with RI)
  - name: control-parity
    task: sparse_parity
    seeds: 5
    priority: 3.0
    ri_policy: always
    ri: {n_probe: 4096, n_points: 30}
    fixed: {n: 20, depth: 1, act: relu, test_size: 2048}
    train: {steps: 20000, batch_size: 32, optimizer: sgd, momentum: 0.0, lr: 0.05, n_log_evals: 60, n_lin_evals: 40}
    knobs:
      k: {values: [2, 3], kind: ordinal}
      width: {values: [128, 512], kind: log}
  - name: control-markov
    task: markov_lang
    seeds: 5
    priority: 3.0
    ri_policy: always
    ri: {n_probe: 8192, n_points: 30}
    fixed: {window: 6, width: 128, depth: 1, n_tokens: 200000}
    train: {steps: 15000, batch_size: 64, optimizer: adam, lr: 0.001}
    knobs:
      couplings: {values: ["1:1.0;1,3:0.8", "1:1.0;1,2,3:1.0", "1,3:0.8"], kind: categorical}
  # ---------------------------------------------------------------- known phenomena
  - name: grokking
    task: modular_addition
    seeds: 3
    priority: 1.5
    ri_policy: followup
    fixed: {depth: 1, embed: onehot}
    train: {steps: 20000, batch_size: 512, optimizer: adamw, beta2: 0.98}
    knobs:
      p: {values: [23, 47], kind: ordinal}
      train_frac: {values: [0.3, 0.5, 0.7], kind: linear, refine: true}
      width: {values: [64, 256], kind: log}
      train.lr: {values: [0.0003, 0.001, 0.003], kind: log, refine: true}
      train.weight_decay: {values: [0.1, 0.3, 1.0], kind: log, refine: true}
  - name: icl-markov
    task: icl_markov
    seeds: 3
    priority: 1.0
    ri_policy: followup
    fixed: {vocab: 3, seq_len: 48, n_layer: 2, n_head: 1, attn_only: true}
    train: {steps: 6000, batch_size: 32, optimizer: adamw, weight_decay: 0.0}
    knobs:
      d_model: {values: [16, 32], kind: log}
      train.lr: {values: [0.001, 0.003], kind: log, refine: true}
  - name: superposition
    task: toy_superposition
    seeds: 3
    priority: 1.0
    ri_policy: followup
    fixed: {n_features: 20, m_hidden: 5}
    train: {steps: 8000, batch_size: 1024, optimizer: adam, lr: 0.001, ntk_every: 0}
    knobs:
      feature_prob: {values: [0.01, 0.03, 0.1, 0.3], kind: log, refine: true}
      importance_decay: {values: [0.7, 0.9], kind: linear}
  # ---------------------------------------------------------------- broad survey region
  - name: parity-wide
    task: sparse_parity
    seeds: 3
    priority: 1.0
    ri_policy: followup
    fixed: {depth: 1, act: relu, test_size: 2048}
    train: {steps: 30000, batch_size: 32, optimizer: sgd, momentum: 0.0}
    knobs:
      n: {values: [16, 24, 32], kind: ordinal}
      k: {values: [2, 3], kind: ordinal}
      width: {values: [32, 128, 512], kind: log, refine: true}
      n_train: {values: [1000, 4000, 16000, null], kind: categorical}
      train.lr: {values: [0.01, 0.03, 0.1], kind: log, refine: true}
      train.weight_decay: {values: [0.0, 0.0001, 0.01], kind: ordinal}
'''
"""Copy of ``configs/sky_v0.yaml`` used by ``tycho init`` when the package runs without its source checkout (a test
keeps it identical to the file)."""
