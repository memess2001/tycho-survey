"""Survey status and dashboard (SPEC §6.4): ``reports/status.json`` and a self-contained ``reports/dashboard.html``.

:func:`collect_status` gathers the survey's state from the catalog in one place: object, queue, follow-up, RI and
transient counts (``Catalog.stats``), throughput (objects finished per clock hour over the last 24 h, failures, median
wall time), the label distribution per task (latest classification of done objects), the latest transients and RI
events, worker heartbeats (stale when older than ``survey.stale_after_s``), free disk space and the age of the newest
backup. The daemon adds its own section (``extra``) and writes the result as ``status.json``; ``tycho survey
status`` prints it; :func:`render_dashboard` turns it into HTML.

The HTML page has no scripts and no external resources (it is opened from disk or copied off the machine), reloads
itself every ``survey.status_every_s`` seconds and follows the system's light or dark mode. Chart conventions: one
series in one hue, thin bars with rounded data ends on a hairline grid, values also in a table view; transient
severity and worker health use the status colours with a text label, never colour alone.
"""
from __future__ import annotations

import html
import json
import math
import os
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tycho.config import Paths, load_config
from tycho.survey.catalog import Catalog

__all__ = ["DASHBOARD_FILE", "STATUS_FORMAT", "collect_status", "render_dashboard", "write_dashboard",
           "write_json_atomic"]

STATUS_FORMAT = "tycho-status/1"
"""Format tag of ``status.json``."""

DASHBOARD_FILE = "dashboard.html"
"""The dashboard in ``reports/``."""

N_LATEST = 15
"""Rows of the latest-transients and latest-events lists."""

THROUGHPUT_HOURS = 24
"""Clock hours in the throughput histogram (the current, partial hour included)."""

_LABEL_ORDER = ("diverged", "no_learning", "memorization", "grokking", "plateau_drop", "staged", "double_descent",
                "smooth", "unknown")
"""Column order of the label table (``tycho.analysis.classify.LABELS``); other labels (e.g. the agent's) follow."""

_SEVERITY_NAMES = {3: "high", 2: "medium", 1: "info"}


# ---------------------------------------------------------------------------------------------- status
def collect_status(catalog: Catalog, paths: Paths, cfg: Mapping[str, Any] | None = None, *,
                   now: float | None = None, extra: Mapping[str, Any] | None = None,
                   n_latest: int = N_LATEST) -> dict[str, Any]:
    """The survey's state as a JSON-friendly dict (see the module docstring); ``extra`` entries are added on top."""
    cfg = cfg if cfg is not None else load_config(paths.root)
    survey = cfg.get("survey") or {}
    now = catalog.now() if now is None else float(now)
    st = catalog.stats()
    stale_after = float(survey.get("stale_after_s", 300.0))
    min_free = float(survey.get("min_free_gb", 20.0))
    status: dict[str, Any] = {
        "format": STATUS_FORMAT,
        "generated_at": now,
        "generated_at_utc": _utc(now),
        "root": str(paths.root),
        "objects": st["objects"],
        "queue": st["queue"],
        "by_region": st["by_region"],
        "followups": st["followups"],
        "labels": st["labels"],
        "labels_by_task": _labels_by_task(catalog),
        "ri_runs": st["ri_runs"],
        "ri_events": st["ri_events"],
        "transients": st["transients"],
        "proposals": st["proposals"],
        "throughput": _throughput(catalog, now, st),
        "compute_hours_done": float(st["wall_time_s_done"] or 0.0) / 3600.0,
        "workers": [{**w, "stale": w["age_s"] >= stale_after} for w in st["workers"]],
        "stale_after_s": stale_after,
        "disk": _disk(paths, min_free),
        "latest_backup": _latest_backup(paths, now),
        "latest_transients": catalog.query(
            "SELECT t.transient_id, t.created_at, t.kind, t.severity, t.message, t.notified, t.acknowledged, "
            "t.object_id, o.task, o.region, o.seed FROM transients t JOIN objects o ON o.object_id = t.object_id "
            "ORDER BY t.transient_id DESC LIMIT ?", (int(n_latest),)),
        "latest_events": [_event_row(r) for r in catalog.query(
            "SELECT e.event_id, e.created_at, e.object_id, o.task, o.region, r.kind AS run_kind, e.subset_json, "
            "e.order_k, e.kind, e.step, e.step_lo, e.step_hi, e.peak_ri, e.confidence, e.matches_gt "
            "FROM ri_events e JOIN objects o ON o.object_id = e.object_id JOIN ri_runs r ON r.ri_run_id = e.ri_run_id "
            "ORDER BY e.event_id DESC LIMIT ?", (int(n_latest),))],
        "status_every_s": float(survey.get("status_every_s", 300.0)),
    }
    status.update(extra or {})
    return _jsonable(status)


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _labels_by_task(catalog: Catalog) -> dict[str, dict[str, int]]:
    rows = catalog.query(
        "SELECT o.task AS task, COALESCE(c.label, 'unlabelled') AS label, COUNT(*) AS n FROM objects o "
        "LEFT JOIN classifications c ON c.id = (SELECT MAX(c2.id) FROM classifications c2 "
        "WHERE c2.object_id = o.object_id) WHERE o.status = 'done' GROUP BY 1, 2 ORDER BY 1, 2")
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out.setdefault(r["task"], {})[r["label"]] = int(r["n"])
    return out


def _throughput(catalog: Catalog, now: float, st: Mapping[str, Any]) -> dict[str, Any]:
    """Done objects per local clock hour (oldest first; the last bin is the current, partial hour)."""
    current = datetime.fromtimestamp(now).replace(minute=0, second=0, microsecond=0).timestamp()
    starts = [current - 3600.0 * (THROUGHPUT_HOURS - 1 - i) for i in range(THROUGHPUT_HOURS)]
    rows = catalog.query("SELECT finished_at, wall_time_s FROM objects WHERE status = 'done' AND finished_at >= ?",
                         (min(starts[0], now - 86_400.0),))
    counts = [0] * THROUGHPUT_HOURS
    for r in rows:
        i = int((r["finished_at"] - starts[0]) // 3600.0)
        if 0 <= i < THROUGHPUT_HOURS:
            counts[i] += 1
    walls = [r["wall_time_s"] for r in rows if r["wall_time_s"] is not None and r["finished_at"] >= now - 86_400.0]
    done_1h = sum(1 for r in rows if r["finished_at"] >= now - 3600.0)
    return {"done_last_1h": done_1h, "done_last_24h": st["done_last_24h"], "failed_last_24h": st["failed_last_24h"],
            "objects_per_hour_24h": st["done_last_24h"] / 24.0,
            "median_wall_time_s_24h": float(statistics.median(walls)) if walls else None,
            "hourly": {"start": starts, "done": counts}}


def _disk(paths: Paths, min_free_gb: float) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(paths.root)
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "min_free_gb": min_free_gb}
    free = usage.free / 1e9
    return {"free_gb": free, "total_gb": usage.total / 1e9, "min_free_gb": min_free_gb, "low": free < min_free_gb}


def _latest_backup(paths: Paths, now: float) -> dict[str, Any] | None:
    """Newest file in ``backups/`` (by modification time) and its age, or None."""
    try:
        entries = [(e.stat().st_mtime, e.name) for e in os.scandir(paths.backups) if e.is_file()]
    except OSError:
        return None
    if not entries:
        return None
    mtime, name = max(entries)
    return {"file": name, "age_h": max(0.0, (time.time() - mtime) / 3600.0), "n_files": len(entries)}


def _event_row(r: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(r)
    try:
        out["subset"] = json.loads(out.pop("subset_json") or "[]")
    except ValueError:
        out["subset"] = []
    return out


def _jsonable(o: Any) -> Any:
    """Strict-JSON form: NaN/inf -> None, tuples -> lists, mapping keys -> str."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, Mapping):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def write_json_atomic(path: Path | str, obj: Any) -> Path:
    """Write strict JSON atomically (temporary file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------------------------- HTML
def write_dashboard(catalog: Catalog, paths: Paths, *, status: Mapping[str, Any] | None = None) -> Path:
    """Write ``reports/dashboard.html`` (atomically) from ``status`` (default: :func:`collect_status`)."""
    status = status if status is not None else collect_status(catalog, paths)
    path = paths.reports / DASHBOARD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(render_dashboard(status), encoding="utf-8")
    os.replace(tmp, path)
    return path


_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --series-1: #2a78d6; --track: #e1e0d9;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --series-1: #3987e5; --track: #2c2c2a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --series-1: #3987e5; --track: #2c2c2a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.meta, .sub { color: var(--ink-2); margin: 0 0 12px; }
.sub { font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin: 16px 0; }
.tile, .card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; }
.tile { padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 12px; }
.card { padding: 16px; margin: 16px 0; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; white-space: nowrap; }
.empty { color: var(--muted); }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
.dot.good { background: var(--good); } .dot.warning { background: var(--warning); }
.dot.serious { background: var(--serious); } .dot.critical { background: var(--critical); }
.dot.neutral { background: var(--muted); }
.share { height: 4px; border-radius: 2px; background: var(--track); margin-top: 3px; min-width: 40px; }
.share > span { display: block; height: 4px; border-radius: 2px; background: var(--series-1); }
svg.chart { width: 100%; height: auto; display: block; }
svg.chart .gridline { stroke: var(--grid); stroke-width: 1; }
svg.chart .baseline { stroke: var(--axis); stroke-width: 1; }
svg.chart .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg.chart .value-label { fill: var(--ink-2); font-size: 11px; font-weight: 600; }
svg.chart .bar { fill: var(--series-1); }
svg.chart .bar:hover { opacity: 0.75; }
details { margin-top: 8px; color: var(--ink-2); }
summary { cursor: pointer; font-size: 13px; }
footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
"""


def render_dashboard(status: Mapping[str, Any]) -> str:
    """The dashboard page for a :func:`collect_status` dict."""
    objects = status.get("objects") or {}
    followups = status.get("followups") or {}
    transients = status.get("transients") or {}
    throughput = status.get("throughput") or {}
    refresh = max(30, int(status.get("status_every_s") or 300))
    generated = float(status.get("generated_at") or time.time())
    parts = [
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<meta http-equiv="refresh" content="{refresh}">', "<title>Tycho survey</title>",
        f"<style>{_CSS}</style>", "</head>", "<body>", "<main>",
        "<header><h1>Tycho survey</h1>",
        f'<p class="meta">Generated {_e(_local(generated))} ({_e(status.get("generated_at_utc", ""))}) · '
        f'{_e(status.get("root", ""))}{_daemon_line(status)}</p></header>',
        '<section class="tiles">',
        _tile("Done", objects.get("done", 0), f"of {_compact(objects.get('total', 0))} objects"),
        _tile("Queued", status.get("queue", 0), f"{_compact(objects.get('interrupted', 0))} interrupted"),
        _tile("Running", objects.get("running", 0), f"{len(status.get('workers') or [])} workers reporting"),
        _tile("Failed", objects.get("failed", 0), f"{_compact(throughput.get('failed_last_24h', 0))} in 24 h"),
        _tile("Done in 24 h", throughput.get("done_last_24h", 0),
              f"{throughput.get('objects_per_hour_24h', 0) or 0:.1f} per hour"),
        _tile("Follow-ups done", followups.get("done", 0),
              f"{_compact(followups.get('requested', 0))} requested, {_compact(followups.get('running', 0))} running"),
        _tile("RI events", status.get("ri_events", 0), f"in {_compact(status.get('ri_runs', 0))} RI runs"),
        _tile("Transients", transients.get("total", 0),
              f"{_compact(transients.get('unacknowledged', 0))} unacknowledged"),
        "</section>",
        _throughput_card(throughput),
        _labels_card(status.get("labels_by_task") or {}),
        _transients_card(status.get("latest_transients") or []),
        _events_card(status.get("latest_events") or []),
        _workers_card(status),
        _regions_card(status.get("by_region") or {}),
        _footer(status),
        "</main>", "</body>", "</html>",
    ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------------------------- HTML pieces
def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _compact(n: Any) -> str:
    """Stat-tile style numbers: 1,284 / 12.9K / 4.2M / 1.1B."""
    try:
        v = float(n)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(v):
        return "–"
    for threshold, divisor, suffix in ((999_950_000, 1e9, "B"), (999_950, 1e6, "M"), (1e4, 1e3, "K")):
        if abs(v) >= threshold:
            text = f"{v / divisor:.1f}"
            return (text[:-2] if text.endswith(".0") else text) + suffix
    return f"{int(v):,}" if v == int(v) else f"{v:,.2f}"


def _local(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return "–"


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


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "–"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _e(x)
    if not math.isfinite(v):
        return "–"
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.{digits}g}"


def _tile(label: str, value: Any, note: str) -> str:
    return (f'<div class="tile"><div class="label">{_e(label)}</div><div class="value">{_e(_compact(value))}</div>'
            f'<div class="note">{_e(note)}</div></div>')


def _table(header: Sequence[tuple[str, bool]], rows: Sequence[Sequence[str]], empty: str) -> str:
    """A table; ``header`` items are (title, numeric); cells are pre-escaped HTML."""
    if not rows:
        return f'<p class="empty">{_e(empty)}</p>'
    head = "".join(f'<th class="num">{_e(h)}</th>' if num else f"<th>{_e(h)}</th>" for h, num in header)
    body = []
    for row in rows:
        cells = "".join(f'<td class="num">{c}</td>' if num else f"<td>{c}</td>" for c, (_, num) in zip(row, header))
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _daemon_line(status: Mapping[str, Any]) -> str:
    daemon = status.get("daemon")
    if not daemon:
        return ""
    state = status.get("state", "unknown")
    return (f" · daemon {_e(state)} (pid {_e(daemon.get('pid'))}, up {_e(_ago(daemon.get('uptime_s')))}, "
            f"{_e(daemon.get('n_workers'))} workers, sky {_e(daemon.get('sky_version'))}, code "
            f"{_e(daemon.get('code_version'))})")


def _nice_step(top: float) -> float:
    """A round tick step giving 3-5 gridlines up to ``top``."""
    raw = max(top, 1.0) / 4.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= raw:
            return max(1.0, m * mag)
    return 10.0 * mag


def _bar_path(x: float, y: float, w: float, base: float, r: float = 4.0) -> str:
    """Bar from the baseline up to ``y`` with rounded top corners (radius clamped to the bar's size)."""
    r = min(r, w / 2.0, base - y)
    return (f"M{x:.1f},{base:.1f}V{y + r:.1f}A{r:.1f},{r:.1f} 0 0 1 {x + r:.1f},{y:.1f}H{x + w - r:.1f}"
            f"A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f}V{base:.1f}Z")


def _throughput_card(tp: Mapping[str, Any]) -> str:
    hourly = tp.get("hourly") or {}
    starts, counts = list(hourly.get("start") or []), [int(c) for c in hourly.get("done") or []]
    median = tp.get("median_wall_time_s_24h")
    sub = (f"{_compact(tp.get('done_last_24h', 0))} done and {_compact(tp.get('failed_last_24h', 0))} failed in the "
           f"last 24 h; median wall time {_fmt(median)} s per object. Local clock hours; the last bar is the current "
           f"hour so far.")
    if not counts:
        return f'<section class="card"><h2>Objects finished per hour</h2><p class="sub">{_e(sub)}</p></section>'
    width, height, left, right, top, bottom = 720.0, 200.0, 40.0, 8.0, 14.0, 24.0
    plot_w, base = width - left - right, height - bottom
    step = _nice_step(max(counts))
    y_max = step * max(1, math.ceil(max(counts) / step))
    slot = plot_w / len(counts)
    bar_w = min(24.0, slot - 4.0)
    svg = [f'<svg class="chart" viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
           f'aria-label="Objects finished per hour over the last {len(counts)} hours">']
    tick = 0.0
    while tick <= y_max + 1e-9:
        y = base - (base - top) * tick / y_max
        cls = "baseline" if tick == 0 else "gridline"
        svg.append(f'<line class="{cls}" x1="{left:.1f}" x2="{width - right:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        svg.append(f'<text class="tick" x="{left - 6:.1f}" y="{y + 4:.1f}" text-anchor="end">{_compact(tick)}</text>')
        tick += step
    peak = max(range(len(counts)), key=lambda i: (counts[i], i))
    for i, (start, n) in enumerate(zip(starts, counts)):
        x = left + i * slot + (slot - bar_w) / 2.0
        label = f"{_hour(start)}–{_hour(start + 3600)}: {n} object{'s' if n != 1 else ''}"
        if i == len(counts) - 1:
            label += " (current hour so far)"
        if n > 0:
            y = base - (base - top) * n / y_max
            svg.append(f'<path class="bar" d="{_bar_path(x, y, bar_w, base)}"><title>{_e(label)}</title></path>')
            if i == peak:
                svg.append(f'<text class="value-label" x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" '
                           f'text-anchor="middle">{n}</text>')
        if i % 4 == 0 or i == len(counts) - 1:
            svg.append(f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{height - 6:.1f}" '
                       f'text-anchor="middle">{_e(_hour(start))}</text>')
    svg.append("</svg>")
    rows = [[_e(f"{_hour(s)}–{_hour(s + 3600)}"), str(n)] for s, n in zip(starts, counts)]
    table = _table([("Hour (local)", False), ("Objects done", True)], rows, "")
    return (f'<section class="card"><h2>Objects finished per hour</h2><p class="sub">{_e(sub)}</p>{"".join(svg)}'
            f"<details><summary>Table view</summary>{table}</details></section>")


def _hour(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def _labels_card(by_task: Mapping[str, Mapping[str, int]]) -> str:
    present = {label for counts in by_task.values() for label in counts}
    columns = [lbl for lbl in _LABEL_ORDER if lbl in present] + sorted(present - set(_LABEL_ORDER))
    header = [("Task", False)] + [(lbl.replace("_", " "), True) for lbl in columns] + [("Total", True)]
    rows = []
    for task in sorted(by_task):
        counts = by_task[task]
        total = sum(counts.values())
        cells = [_e(task)]
        for lbl in columns:
            n = counts.get(lbl, 0)
            if n == 0:
                cells.append('<span class="empty">0</span>')
                continue
            share = 100.0 * n / total
            cells.append(f'{n}<div class="share" title="{share:.0f}% of {_e(task)}">'
                         f'<span style="width:{share:.1f}%"></span></div>')
        cells.append(str(total))
        rows.append(cells)
    return ('<section class="card"><h2>Classification by task</h2><p class="sub">Latest label of every done object '
            '(rule set or agent); the bar under a count is its share of the task.</p>'
            f'{_table(header, rows, "No classified objects yet.")}</section>')


def _severity_cell(severity: Any) -> str:
    sev = int(severity or 1)
    cls = {3: "critical", 2: "serious"}.get(sev, "neutral")
    return f'<span class="dot {cls}"></span>{sev} {_SEVERITY_NAMES.get(sev, "")}'


def _transients_card(rows: Sequence[Mapping[str, Any]]) -> str:
    body = [[_e(_local(r.get("created_at"))), _severity_cell(r.get("severity")), _e(r.get("kind")),
             f'<span class="mono">{_e(r.get("object_id"))}</span><br><span class="empty">{_e(r.get("task"))} · '
             f'{_e(r.get("region"))} · seed {_e(r.get("seed"))}</span>', _e(r.get("message")),
             "yes" if r.get("notified") else "pending"] for r in rows]
    header = [("Time", False), ("Severity", False), ("Kind", False), ("Object", False), ("Message", False),
              ("Notified", False)]
    return ('<section class="card"><h2>Latest transients</h2><p class="sub">Newest first. Severity 3 and 2 are '
            'pushed when at or above notify.min_severity.</p>'
            f'{_table(header, body, "No transients yet.")}</section>')


def _events_card(rows: Sequence[Mapping[str, Any]]) -> str:
    def gt(v: Any) -> str:
        return "–" if v is None else ("yes" if v else "no")

    body = [[_e(_local(r.get("created_at"))), _e(r.get("kind")), _e("{" + ", ".join(map(str, r.get("subset") or []))
                                                                    + "}"),
             str(r.get("order_k")), _e(f"{_fmt(r.get('step'))} ({_fmt(r.get('step_lo'))}, {_fmt(r.get('step_hi'))}]"),
             _fmt(r.get("peak_ri")), _fmt(r.get("confidence")), gt(r.get("matches_gt")),
             f'<span class="mono">{_e(r.get("object_id"))}</span><br><span class="empty">{_e(r.get("task"))} · '
             f'{_e(r.get("run_kind"))}</span>'] for r in rows]
    header = [("Time", False), ("Event", False), ("Subset", False), ("Order", True), ("Step (bracket]", False),
              ("Peak RI (nats)", True), ("Confidence", True), ("Ground truth", False), ("Object", False)]
    return ('<section class="card"><h2>Latest RI events</h2><p class="sub">Activations, extinctions and '
            'reactivations of variable subsets, newest first; the onset lies in the step bracket.</p>'
            f'{_table(header, body, "No RI events yet.")}</section>')


def _workers_card(status: Mapping[str, Any]) -> str:
    rows = []
    for w in status.get("workers") or []:
        stale = bool(w.get("stale"))
        state = (f'<span class="dot {"critical" if stale else "good"}"></span>'
                 f'{"stale" if stale else "ok"}')
        rows.append([_e(w.get("worker_id")), state, _e(w.get("status")),
                     f'<span class="mono">{_e(w.get("object_id") or "")}</span>', _e(w.get("host")),
                     _e(w.get("pid")), _e(_ago(w.get("age_s")))])
    header = [("Worker", False), ("Health", False), ("Status", False), ("Object", False), ("Host", False),
              ("PID", True), ("Last heartbeat", True)]
    sub = f"Stale after {_fmt(status.get('stale_after_s'))} s without a heartbeat."
    return (f'<section class="card"><h2>Workers</h2><p class="sub">{_e(sub)}</p>'
            f'{_table(header, rows, "No worker has reported yet.")}</section>')


def _regions_card(by_region: Mapping[str, Mapping[str, int]]) -> str:
    statuses = ("done", "queued", "running", "interrupted", "failed")
    rows = [[_e(region)] + [str(counts.get(s, 0)) for s in statuses] + [str(sum(counts.values()))]
            for region, counts in sorted(by_region.items())]
    header = [("Region", False)] + [(s, True) for s in statuses] + [("Total", True)]
    return ('<section class="card"><h2>Regions</h2>'
            f'{_table(header, rows, "The catalog is empty: run tycho survey enqueue.")}</section>')


def _footer(status: Mapping[str, Any]) -> str:
    disk = status.get("disk") or {}
    backup = status.get("latest_backup")
    items = []
    if "free_gb" in disk:
        low = " (below survey.min_free_gb: claiming paused)" if disk.get("low") else ""
        items.append(f"Disk: {disk['free_gb']:.1f} GB free of {disk.get('total_gb', 0):.0f} GB{low}")
    items.append("Latest backup: none" if not backup else
                 f"Latest backup: {backup['file']} ({_ago(3600.0 * float(backup['age_h']))} ago)")
    items.append(f"Compute in done objects: {float(status.get('compute_hours_done') or 0):.1f} CPU-hours")
    return f"<footer>{_e(' · '.join(items))}</footer>"
