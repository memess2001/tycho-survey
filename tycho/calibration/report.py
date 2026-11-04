"""Calibration report: ``CALIBRATION.md`` (instrument error budget + one section per experiment), PNG figures and
``calibration.json`` (the machine-readable error budget the survey reads to set ``min_effect``).

Input is the dict returned by :func:`tycho.calibration.suite.run_calibration` (or reloaded with
:func:`tycho.calibration.suite.load_results`); missing values arrive as ``None`` (strict JSON). Every number in the
report is computed from that dict, so the text stays true for quick and full runs alike; wherever a statement is
conditional ("within Monte Carlo error", "verified"), the condition is evaluated, not assumed.

``calibration.json`` (format ``tycho-calibration/1``) holds, besides provenance:

* ``recommended_min_effect``: ``{"rule", "floor", "null_quantile", "table": [{order, n, noise, floor, value,
  source}], "by_n": [{n, value}]}``; read it with :func:`recommended_min_effect`;
* ``noise_floor``: ``[{order, n, df, n_rep, q50, q95, q99, q*_corrected, theory_q*}]`` (null quantiles of RI, nats);
* ``min_detectable_ri``: ``[{order, n, family_size, rule, min_effect, power_level, mde, censored, theory_mde}]``;
* ``timing``: ``[{width, n, true_onset, detect_rate, median_error, q25, q75, ...}]`` (checkpoints);
* ``artifacts``: ``[{method, rho, threshold, order, ri_population, n, ri_corrected_mean, detect_rate}]``;
* ``chain_dependence``, ``specificity``, ``estimator_bias``, ``false_positive_rate``, ``analytic`` (formulas).
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from tycho.calibration.suite import MIN_REPS_FOR_SE, ZERO_TOL, interpolate_min_effect, jsonable

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

__all__ = ["CALIBRATION_FORMAT", "write_report", "calibration_document", "load_calibration",
           "recommended_min_effect"]

#: Format tag of ``calibration.json``.
CALIBRATION_FORMAT = "tycho-calibration/1"
#: Figure resolution (dots per inch).
DPI = 120
#: Top of the p-value density axis (E1 panel C); a taller bar (the p = 1 point mass of boundary tables) is cut there.
P_HIST_YMAX = 2.5

# Chart tokens (reference palette of the dataviz method): categorical slots in fixed order, an ordinal blue ramp for
# ordered series (sample sizes), ink and chrome colours.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LINE_W, MARKER = 1.6, 5.5


def _pyplot() -> Any:
    """``matplotlib.pyplot`` on the non-interactive Agg backend, imported on first use: importing tycho.calibration
    (e.g. in survey processes that only read calibration.json, or in spawned calibration workers) never loads
    matplotlib or changes its backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ----------------------------------------------------------------------------------------------- formatting
def _ok(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _g(x: Any, digits: int = 3) -> str:
    """General number, ``digits`` significant digits; '–' when missing."""
    return f"{x:.{digits}g}" if _ok(x) else "–"


def _e(x: Any, digits: int = 2) -> str:
    """Compact scientific notation (``8.1e-4``); plain for 0 and for 1e-2 <= |x| < 1e3."""
    if not _ok(x):
        return "–"
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e3:
        return f"{x:.{digits}g}"
    mant, exp = f"{x:.{digits - 1}e}".split("e")
    return f"{mant}e{int(exp)}"


def _pct(x: Any, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f} %" if _ok(x) else "–"


def _n(x: Any) -> str:
    return f"{int(x):,}" if _ok(x) else "–"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _exp(results: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    return results.get("experiments", {}).get(name)


def _budget_n(results: Mapping[str, Any]) -> int:
    return int(results["budget"]["budget_n"])


def _sorted_unique(values: Any) -> list:
    return sorted({v for v in values})


# ----------------------------------------------------------------------------------------------- calibration.json
def calibration_document(results: Mapping[str, Any]) -> dict:
    """The content of ``calibration.json``: provenance plus the error budget of ``results``."""
    doc = {"format": CALIBRATION_FORMAT, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "preset": results.get("preset"), "seed": results.get("config", {}).get("seed"),
           "results_started_at": results.get("started_at"),
           "experiments_run": list(results.get("experiments", {}))}
    doc.update(results["budget"])
    return jsonable(doc)


def load_calibration(path: Path | str) -> dict:
    """Read a ``calibration.json`` written by :func:`write_report`."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("format") != CALIBRATION_FORMAT:
        raise ValueError(f"{path}: unexpected calibration format {doc.get('format')!r}")
    return doc


def recommended_min_effect(calibration: Mapping[str, Any] | Path | str, n: float, order: int | None = None) -> float:
    """Recommended ``min_effect`` (nats) for a scan with ``n`` probe samples.

    ``order=None`` gives the maximum over orders (one threshold per scan, as ``ri_scan`` applies it). Between and
    beyond the calibrated sample sizes the noise part is scaled like 1/N; the policy floor always applies (see
    :func:`tycho.calibration.suite.interpolate_min_effect`).

    Args:
        calibration: a ``calibration.json`` path or its parsed content.
    """
    doc = load_calibration(calibration) if isinstance(calibration, (str, Path)) else calibration
    return interpolate_min_effect(doc["recommended_min_effect"]["table"], n, order)


# ----------------------------------------------------------------------------------------------- figures
def _style(ax: Axes, xlog: bool = False, ylog: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=AXIS)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)


def _short_n(n: int) -> str:
    """Tick label of a sample size: ``512``, ``4k`` (multiples of 1024 above 1000), else ``12,000``."""
    n = int(n)
    if n < 1000:
        return str(n)
    return f"{n // 1024}k" if n % 1024 == 0 else f"{n:,}"


def _n_axis(ax: Axes, ns: Sequence[int], label: str = "sample size N") -> None:
    """Log x axis with one labelled tick per grid value (no minor labels, which collide on short decades)."""
    from matplotlib.ticker import NullFormatter, NullLocator

    ax.set_xscale("log")
    ax.set_xticks(list(ns), [_short_n(n) for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(label + (" (k = 1024)" if any(int(n) >= 1000 and int(n) % 1024 == 0 for n in ns) else ""))


def _title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9.5, color=INK, pad=6)


def _legend(ax: Axes, **kw: Any) -> None:
    leg = ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, **kw)
    if leg is not None:
        for h in leg.legend_handles:
            h.set_alpha(1.0)


def _ref_line(ax: Axes, y: float, label: str) -> None:
    """A solid muted reference line with a direct label at its right end."""
    ax.axhline(y, color=MUTED, linewidth=0.9, zorder=1)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=7, color=INK2)


def _ramp(k: int) -> list[str]:
    """``k`` colours of the ordinal ramp, light to dark (no lighter than the ramp's first step)."""
    if k <= 1:
        return [RAMP[6]]
    idx = np.linspace(0, len(RAMP) - 1, k).round().astype(int)
    return [RAMP[i] for i in idx]


def _save(fig: Figure, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")  # keeps figure-level legends inside
    _pyplot().close(fig)


def _fig_e1(results: Mapping[str, Any], path: Path) -> bool:
    exp = _exp(results, "E1_null")
    if not exp:
        return False
    rows = [r for r in exp["rows"] if r["null"] == "chi2" and r["binary"]]
    orders = _sorted_unique(r["order"] for r in rows)
    ns = _sorted_unique(r["n"] for r in rows)
    alphas = results["config"]["fpr_alphas"]
    fig, axes = _pyplot().subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    _style(ax)
    a = max(alphas)
    col = f"fpr_{a:g}"
    for i, k in enumerate(orders):
        mean = [np.mean([r[col] for r in rows if r["order"] == k and r["n"] == n]) for n in ns]
        lo = [min(r[col] for r in rows if r["order"] == k and r["n"] == n) for n in ns]
        hi = [max(r[col] for r in rows if r["order"] == k and r["n"] == n) for n in ns]
        ax.fill_between(ns, lo, hi, color=SERIES[i], alpha=0.12, linewidth=0)
        ax.plot(ns, mean, color=SERIES[i], lw=LINE_W, marker="o", ms=MARKER, mec=SURFACE, mew=1.2,
                label=f"order {k}")
    _ref_line(ax, a, f"nominal α = {a:g}")
    _n_axis(ax, ns)
    ax.set_ylabel(f"false-positive rate at α = {a:g}")
    ax.set_ylim(0, max(2.5 * a, ax.get_ylim()[1]))
    _title(ax, "A  Null FPR (line: mean over strengths; band: range)")
    _legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    ax = axes[1]
    _style(ax, ylog=True)
    nf = results["budget"].get("noise_floor", [])
    for i, k in enumerate(orders):
        sel = sorted((r for r in nf if r["order"] == k), key=lambda r: r["n"])
        ax.plot([r["n"] for r in sel], [r["q99"] for r in sel], color=SERIES[i], lw=LINE_W, marker="o", ms=MARKER,
                mec=SURFACE, mew=1.2, label=f"order {k}")
    th = sorted((r for r in nf if r["order"] == orders[0]), key=lambda r: r["n"])
    ax.plot([r["n"] for r in th], [r["theory_q99"] for r in th], color=MUTED, lw=1.0, label="χ²₁ theory  χ²(0.99)/2N")
    _n_axis(ax, ns)
    ax.set_ylabel("99th pct of null RI_plugin (nats)")
    _title(ax, "B  Noise floor")
    _legend(ax, loc="upper right")
    ax = axes[2]
    _style(ax)
    hist = exp["summary"]["pvalue_hist"]
    top = P_HIST_YMAX
    spikes = []
    for i, k in enumerate(orders):
        h = hist[str(k)]
        counts = np.asarray(h["counts"], float)
        edges = np.asarray(h["edges"])
        dens = counts / counts.sum() / np.diff(edges)
        ax.stairs(np.minimum(dens, top), edges, color=SERIES[i], lw=LINE_W, label=f"order {k}")
        if dens[-1] > top:
            spikes.append(f"order {k}: {counts[-1] / counts.sum():.0%} in the last bin")
    _ref_line(ax, 1.0, "uniform")
    if spikes:  # boundary tables give p = 1 exactly; the bar is cut at the axis top and its mass is stated
        ax.annotate("cut at top: " + "; ".join(spikes), xy=(0.02, 0.97), xycoords="axes fraction", va="top",
                    fontsize=7, color=INK2)
    ax.set_xlabel("χ² p-value under H0")
    ax.set_ylabel("density")
    ax.set_ylim(0, top)
    _title(ax, "C  Null p-values (pooled over strengths and N)")
    _legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    _save(fig, path)
    return True


def _fig_e2(results: Mapping[str, Any], path: Path) -> bool:
    exp = _exp(results, "E2_bias")
    if not exp:
        return False
    rows = [r for r in exp["rows"] if _ok(r["lambda"]) and r["lambda"] > 0]
    lam = np.asarray([r["lambda"] for r in rows])
    fig, axes = _pyplot().subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    _style(ax, xlog=True)
    se = np.asarray([r["bias_se"] / r["theory_bias_plugin"] if _ok(r["bias_se"]) else np.nan for r in rows])
    for i, (key, label) in enumerate((("bias_plugin", "RI_plugin"), ("bias_corrected", "RI_corrected"))):
        y = np.asarray([r[key] / r["theory_bias_plugin"] if _ok(r[key]) else np.nan for r in rows])
        x = lam * (1.0 + 0.06 * (2 * i - 1))  # side by side, so the two estimators' bars do not overlap
        ax.errorbar(x, y, yerr=se, fmt="o", ms=3.5, color=SERIES[i], mec=SURFACE, mew=0.6, elinewidth=0.8,
                    capsize=0, label=label, zorder=3)
    _ref_line(ax, 1.0, "df/2N")
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlabel("noncentrality λ = 2N · true RI")
    ax.set_ylabel("bias / (df / 2N)")
    _title(ax, "A  Bias in units of df/2N (bars: ±1 Monte Carlo SE)")
    _legend(ax, loc="upper left")
    ax = axes[1]
    _style(ax, xlog=True)
    y = np.asarray([r["sd_plugin"] / r["theory_sd_plugin"] if _ok(r["sd_plugin"]) else np.nan for r in rows])
    ax.scatter(lam, y, s=14, color=SERIES[0], edgecolors=SURFACE, linewidths=0.6, zorder=3)
    _ref_line(ax, 1.0, "theory √(2(df+2λ))/2N")
    ax.set_xlabel("noncentrality λ = 2N · true RI")
    ax.set_ylabel("SD(RI_plugin) / theory")
    _title(ax, "B  Spread of RI_plugin vs noncentral χ²")
    ax = axes[2]
    _style(ax, xlog=True)
    cov = [r for r in rows if r["n_ci"]]
    if cov:
        ax.scatter([r["lambda"] for r in cov], [r["coverage"] for r in cov], s=14, color=SERIES[0],
                   edgecolors=SURFACE, linewidths=0.6, zorder=3)
    level = results["config"]["e2_ci_level"]
    _ref_line(ax, level, f"nominal {level:g}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("noncentrality λ = 2N · true RI")
    ax.set_ylabel("CI coverage")
    _title(ax, "C  Row-bootstrap percentile CI of RI_corrected")
    _save(fig, path)
    return True


def _fig_e3(results: Mapping[str, Any], path_power: Path, path_mde: Path) -> list[str]:
    exp = _exp(results, "E3_power")
    if not exp:
        return []
    rows = exp["rows"]
    orders = _sorted_unique(r["order"] for r in rows)
    ns = _sorted_unique(r["n"] for r in rows)
    m_max = max(results["config"]["e3_family_sizes"])
    level = results["config"]["e3_power"]
    colors = _ramp(len(ns))
    fig, axes = _pyplot().subplots(1, len(orders), figsize=(4.2 * len(orders), 3.6), squeeze=False)
    for ax, k in zip(axes[0], orders):
        _style(ax, xlog=True)
        for c, n in zip(colors, ns):
            sel = sorted((r for r in rows if r["order"] == k and r["n"] == n and r["family_size"] == m_max),
                         key=lambda r: r["true_ri"])
            ax.plot([r["true_ri"] for r in sel], [r["power"] for r in sel], color=c, lw=LINE_W, marker="o",
                    ms=4, mec=SURFACE, mew=0.8, label=f"N = {n:,}")
        _ref_line(ax, level, f"{level:.0%} power")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("true local RI (nats)")
        ax.set_ylabel("detection probability")
        _title(ax, f"Order {k}: survey rule, scan of {m_max:,} subsets")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7.5,
               labelcolor=INK2)
    _save(fig, path_power)
    mde = exp["summary"]["mde"]
    fams = results["config"]["e3_family_sizes"]
    panels = [(min(fams), "stat", "single subset, q ≤ α only"), (m_max, "survey", f"scan of {m_max:,}, survey rule")]
    fig, axes = _pyplot().subplots(1, 2, figsize=(10.5, 3.8))
    for ax, (m, rule, label) in zip(axes, panels):
        _style(ax, ylog=True)
        for i, k in enumerate(orders):
            sel = sorted((r for r in mde if r["order"] == k and r["family_size"] == m and r["rule"] == rule),
                         key=lambda r: r["n"])
            pts = [(r["n"], r["mde"]) for r in sel if _ok(r["mde"]) and not r["censored"]]
            if pts:
                ax.plot(*zip(*pts), color=SERIES[i], lw=LINE_W, marker="o", ms=MARKER, mec=SURFACE, mew=1.2,
                        label=f"order {k}")
            cens = [(r["n"], r["mde"]) for r in sel if _ok(r["mde"]) and r["censored"] == "below"]
            if cens:
                ax.scatter(*zip(*cens), marker="v", s=30, color=SERIES[i], edgecolors=SURFACE, linewidths=0.8,
                           zorder=3)
        th = sorted((r for r in mde if r["order"] == orders[0] and r["family_size"] == m and r["rule"] == rule),
                    key=lambda r: r["n"])
        ax.plot([r["n"] for r in th], [r["theory_mde"] for r in th], color=MUTED, lw=1.0, label="noncentral χ² theory")
        _n_axis(ax, ns)
        ax.set_ylabel(f"min. detectable RI at {level:.0%} power (nats)")
        _title(ax, f"{'AB'[panels.index((m, rule, label))]}  {label}")
        _legend(ax, loc="upper right")
    _save(fig, path_mde)
    return [path_power.name, path_mde.name]


def _fig_e4(results: Mapping[str, Any], path: Path) -> bool:
    exp = _exp(results, "E4_timing")
    if not exp:
        return False
    rows = exp["rows"]
    widths = _sorted_unique(r["width"] for r in rows)
    ns = _sorted_unique(r["n"] for r in rows)
    me = exp["summary"]["min_effect"]
    fig, axes = _pyplot().subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    _style(ax, ylog=True)
    for i, w in enumerate(widths):
        cur = exp["summary"]["curves"][f"{w:g}"]
        ri = np.asarray([v if _ok(v) else np.nan for v in cur["true_ri"]])
        t = np.arange(ri.size)
        shown = ri > ZERO_TOL  # below this the "RI" of a nearly switched-off coupling is rounding noise
        ax.plot(t[shown], ri[shown], color=SERIES[i], lw=LINE_W, label=f"width {w:g}")
    _ref_line(ax, me, f"min_effect = {me:g}")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("population RI (nats)")
    _title(ax, "A  Planted switch-on curves")
    _legend(ax, loc="lower right")
    for ax, key, ylabel, title in ((axes[1], "median_error", "onset error (checkpoints)", "B  Median error, IQR bars"),
                                   (axes[2], "detect_rate", "activation detected", "C  Detection rate")):
        _style(ax)
        for i, w in enumerate(widths):
            sel = sorted((r for r in rows if r["width"] == w), key=lambda r: r["n"])
            x = np.asarray([r["n"] for r in sel], float) * (1.0 + 0.04 * (i - (len(widths) - 1) / 2))
            y = np.asarray([r[key] if _ok(r[key]) else np.nan for r in sel])
            if key == "median_error":
                lo = y - np.asarray([r["q25"] if _ok(r["q25"]) else np.nan for r in sel])
                hi = np.asarray([r["q75"] if _ok(r["q75"]) else np.nan for r in sel]) - y
                ax.errorbar(x, y, yerr=[lo, hi], color=SERIES[i], lw=LINE_W, marker="o", ms=MARKER, mec=SURFACE,
                            mew=1.2, capsize=0, elinewidth=1.0, label=f"width {w:g}")
            else:
                ax.plot(x, y, color=SERIES[i], lw=LINE_W, marker="o", ms=MARKER, mec=SURFACE, mew=1.2,
                        label=f"width {w:g}")
        if key == "median_error":
            ax.axhline(0.0, color=MUTED, linewidth=0.9, zorder=1)
        else:
            ax.set_ylim(-0.02, 1.02)
        _n_axis(ax, ns, "sample size N per checkpoint")
        ax.set_ylabel(ylabel)
        _title(ax, title)
        _legend(ax, loc="best")
    _save(fig, path)
    return True


def _fig_e5(results: Mapping[str, Any], path: Path) -> bool:
    exp = _exp(results, "E5_identity")
    if not exp:
        return False
    rows = sorted((r for r in exp["rows"] if r["rule"] == "survey"), key=lambda r: r["n"])
    ns = [r["n"] for r in rows]
    fig, axes = _pyplot().subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    _style(ax)
    for i, (key, label) in enumerate((("recall", "recall"), ("precision", "precision vs planted"),
                                      ("precision_true", "precision vs population truth"),
                                      ("exact_recovery", "exact recovery"))):
        ax.plot(ns, [r[key] if _ok(r[key]) else np.nan for r in rows], color=SERIES[i], lw=LINE_W, marker="o",
                ms=MARKER, mec=SURFACE, mew=1.2, label=label)
    ax.set_ylim(-0.02, 1.02)
    _n_axis(ax, ns)
    ax.set_ylabel("fraction")
    _title(ax, "A  Identity of planted subsets (survey rule)")
    _legend(ax, loc="lower right")
    ax = axes[1]
    _style(ax)
    for i, (key, label) in enumerate((("mean_fp_spurious", "spurious (true RI < min_effect)"),
                                      ("mean_fp_real_unplanted", "real but unplanted"),
                                      ("mean_fp_related", "sub/superset of a planted subset"))):
        ax.plot(ns, [r[key] for r in rows], color=SERIES[i], lw=LINE_W, marker="o", ms=MARKER, mec=SURFACE,
                mew=1.2, label=label)
    _n_axis(ax, ns)
    ax.set_ylabel("false positives per scan")
    _title(ax, "B  False positives by kind")
    _legend(ax, loc="upper left")
    _save(fig, path)
    return True


def _fig_e6(results: Mapping[str, Any], path: Path) -> bool:
    exp = _exp(results, "E6_discretization")
    if not exp:
        return False
    n0 = min(r["n"] for r in exp["rows"])
    rows = [r for r in exp["rows"] if r["n"] == n0]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    orders = _sorted_unique(r["order"] for r in rows)
    fig, axes = _pyplot().subplots(1, len(methods), figsize=(4.2 * len(methods), 3.6), squeeze=False, sharey=True)
    for ax, method in zip(axes[0], methods):
        _style(ax, ylog=True)
        zero_orders = []
        for i, j in enumerate(orders):
            sel = sorted((r for r in rows if r["method"] == method and r["order"] == j), key=lambda r: r["rho"])
            pts = [(r["rho"], r["ri_population"]) for r in sel if _ok(r["ri_population"])
                   and r["ri_population"] > ZERO_TOL]
            if any(r["rho"] > 0 for r in sel) and not any(rho > 0 for rho, _ in pts):
                zero_orders.append(j)
            if pts:
                ax.plot(*zip(*pts), color=SERIES[i], lw=LINE_W, marker="o", ms=MARKER, mec=SURFACE, mew=1.2,
                        label=f"order {j}")
        tau = rows[[r["method"] for r in rows].index(method)]["threshold"]
        note = f"\norder {', '.join(map(str, zero_orders))}: zero for every ρ" if zero_orders else ""
        _title(ax, f"{method}: threshold τ = {tau:+.2f} (std. units){note}")
        ax.set_xlabel("latent correlation ρ")
        ax.set_ylabel("population local RI (nats)")
        _legend(ax, loc="lower right")
    _save(fig, path)
    return True


def _fig_e7(results: Mapping[str, Any], path: Path) -> bool:
    exp = _exp(results, "E7_chain")
    if not exp:
        return False
    rows = exp["rows"]
    fig, axes = _pyplot().subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    _style(ax, ylog=True)
    orders = _sorted_unique(r["order"] for r in rows if not r["full_set"])
    data = [np.maximum([r["spread_over_tc"] for r in rows if r["order"] == o and not r["full_set"]], 1e-12)
            for o in orders]
    bp = ax.boxplot(data, positions=range(len(orders)), widths=0.35, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "linewidth": 1.2}, whiskerprops={"color": INK2},
                    capprops={"color": INK2}, flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": MUTED,
                                                          "markeredgecolor": "none"})
    for patch, c in zip(bp["boxes"], SERIES):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    ax.set_xticks(range(len(orders)), [f"order {o}" for o in orders])
    ax.set_ylabel("(max − min chain RI) / TC")
    _title(ax, "A  Chain dependence of each subset's RI (full set excluded)")
    ax = axes[1]
    _style(ax)
    err = np.asarray([r["abs_error"] for r in exp["tables"]["totals"]])
    ax.hist(np.log10(np.maximum(err, 1e-17)), bins=30, color=SERIES[0], alpha=0.8, edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlabel("log10 |Σ chain RI − TC| (nats)")
    ax.set_ylabel("chains")
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _title(ax, "B  Completeness: Σ RI = TC on every chain")
    _save(fig, path)
    return True


def _figures(results: Mapping[str, Any], figdir: Path) -> dict[str, str]:
    figdir.mkdir(parents=True, exist_ok=True)
    made: dict[str, str] = {}
    for key, fn in (("E1", _fig_e1), ("E2", _fig_e2), ("E4", _fig_e4), ("E5", _fig_e5), ("E6", _fig_e6),
                    ("E7", _fig_e7)):
        path = figdir / f"{key.lower()}.png"
        if fn(results, path):
            made[key] = f"{figdir.name}/{path.name}"
    e3 = _fig_e3(results, figdir / "e3_power.png", figdir / "e3_mde.png")
    if e3:
        made["E3"] = f"{figdir.name}/{e3[0]}"
        made["E3_mde"] = f"{figdir.name}/{e3[1]}"
    return made


# ----------------------------------------------------------------------------------------------- markdown
def _intro(results: Mapping[str, Any]) -> str:
    cfg = results["config"]
    n_b = _budget_n(results)
    compute_h = sum((results.get("compute_s") or {}).values()) / 3600.0
    return f"""# Tycho instrument calibration

*Preset `{results.get('preset')}`, seed {cfg['seed']}, run started {results.get('started_at')}; wall time
{_g((results.get('wall_s') or 0) / 60, 3)} min with {results.get('n_jobs')} worker(s); {_n(results.get('n_items'))} work
items, {_n(results.get('n_cached'))} of them reused from the cache of an earlier run, {_g(compute_h, 3)} CPU-hours
computed in this run. Source data: `results.json` and one CSV per experiment; machine-readable error budget:
`calibration.json`.*

## How to read this report

The survey's instrument is the **marginal refined information** RI(S) = RI^marg_S of a subset S of discrete
variables (Enouen & Sugiyama, TMLR 2026, Appendix C.1; called *local RI* in the code): the top-order term of the
S-marginal, `RI(S) = D(P_S ‖ (P_S)_∂S)`, in nats (`tycho/ri/refined.py`). It is 0 exactly
when the variables of S have no |S|-th-order interaction, and it equals the mutual information for |S| = 2. For
scale: log 2 = 0.693 nats is the RI of a perfect parity, and a pair of binary variables with correlation r has
RI ≈ r²/2 (so 1e-3 nats ≈ r = 0.045).

From N samples the instrument reports `RI_plugin` (RI of the empirical table), `RI_corr = max(0, RI_plugin −
df/2N)` (first-order bias correction; df = Π(k_i − 1), so df = 1 for binary variables), a p-value from
`G² = 2N·RI_plugin ~ χ²_df` and a Benjamini–Hochberg q-value across the subsets of a scan. A subset is
**detected** when `q ≤ α` *and* `RI_corr ≥ min_effect`. Here α = {cfg['alpha']:g}, and N = {n_b:,} (the survey's
default probe size) is used for the headline numbers.

Each experiment plants a known truth, observes it exactly as the survey would, and compares. All numbers are
Monte Carlo estimates; where a ± appears it is one standard error. "λ" denotes the noncentrality 2N·RI, the
quantity that asymptotic theory says governs both power and precision.
"""


def _budget_section(results: Mapping[str, Any]) -> str:
    b = results["budget"]
    cfg = results["config"]
    n_b = _budget_n(results)
    rows: list[list[str]] = []
    if "noise_floor" in b:
        at = sorted((r for r in b["noise_floor"] if r["n"] == n_b), key=lambda r: r["order"])
        rows.append(["Noise floor: 99th pct of null RI_plugin (orders " + "/".join(str(r["order"]) for r in at) + ")",
                     " / ".join(_e(r["q99"]) for r in at) + f" (χ² theory {_e(at[0]['theory_q99'])})",
                     "E1", "An RI estimate below this is what pure sampling noise produces for 1 % of subsets "
                           "with no interaction. It scales as 1/N."])
    rme = b["recommended_min_effect"]
    me_b = interpolate_min_effect(rme["table"], n_b)
    rows.append([f"Recommended min_effect (whole scan, N = {n_b:,})", _e(me_b, 3), "E1 + policy",
                 f"max(policy floor {_e(rme['floor'])}, {rme['null_quantile']:g} null quantile of RI_corr). "
                 "The survey reads it from calibration.json."])
    if "false_positive_rate" in b:
        f = b["false_positive_rate"]["chi2"]
        for a in cfg["fpr_alphas"]:
            s = f[f"{a:g}"]
            rows.append([f"False-positive rate at α = {a:g} (χ² p-value)",
                         f"{_pct(s['pooled'], 2)} pooled; cells {_pct(s['min'], 1)}–{_pct(s['max'], 1)}", "E1",
                         f"Of {s['n_cells']} (cards, strength, N) cells, {s['n_liberal']} have significantly more "
                         f"false alarms than α and {s['n_conservative']} significantly fewer (conservative); exact "
                         f"one-sided binomial tests at {s['tail']:g}, the 3σ tail."])
    if "min_detectable_ri" in b:
        fams = cfg["e3_family_sizes"]
        for m, rule, what in ((min(fams), "stat", "one targeted subset, q ≤ α"),
                              (max(fams), "survey", f"within a scan of {max(fams):,} subsets, full survey rule")):
            sel = sorted((r for r in b["min_detectable_ri"] if r["n"] == n_b and r["family_size"] == m
                          and r["rule"] == rule), key=lambda r: r["order"])
            if sel:
                rows.append([f"Minimum detectable RI at {cfg['e3_power']:.0%} power ({what})",
                             " / ".join(_mde_cell(r) for r in sel) + f" (theory {_e(sel[0]['theory_mde'])})", "E3",
                             "Smallest true RI found 80 % of the time. Orders listed as in the noise-floor row."])
    if "estimator_bias" in b:
        e2 = _exp(results, "E2_bias")["summary"]
        rows.append(["Bias of RI_plugin, in units of df/2N (median over cells)", _g(e2["median_bias_ratio_plugin"]),
                     "E2", "1 means the plug-in overestimates by exactly the first-order term df/2N, which "
                           "RI_corr removes."])
        rows.append(["Relative bias of RI_corr when λ ≥ 20 (worst cell)",
                     _pct(e2["max_abs_rel_bias_corrected_strong"]), "E2",
                     "Worst |mean RI_corr − RI| / RI over cells with a clearly detectable interaction (λ = 2N·RI "
                     "≥ 20), Monte Carlo error included."])
        rows.append([f"Bootstrap {cfg['e2_ci_level']:.0%} CI coverage (median; λ ≥ 20 / λ < 20)",
                     f"{_pct(e2['median_coverage_strong'], 0)} / {_pct(e2['median_coverage_weak'], 0)}", "E2",
                     "Fraction of row-bootstrap percentile intervals of RI_corr that contain the true RI "
                     f"(nominal {cfg['e2_ci_level']:.0%})."])
    if "timing" in b:
        t = [r for r in b["timing"] if r["n"] == _nearest_in([r["n"] for r in b["timing"]], n_b, log=True)]
        if t:
            rows.append([f"Onset timing error, median [IQR] in checkpoints (N = {t[0]['n']:,})",
                         "; ".join(f"width {r['width']:g}: {_g(r['median_error'])} [{_g(r['q25'])}, "
                                   f"{_g(r['q75'])}]" for r in sorted(t, key=lambda r: r["width"])), "E4",
                         "Detected minus true onset. Positive = reported late. The true onset is where the "
                         "population RI first reaches min_effect."])
    if "specificity" in b:
        sp = [r for r in b["specificity"] if r["rule"] == "survey"]
        r = min(sp, key=lambda r: (abs(math.log(r["n"] / n_b)), r["n"]))
        n_sub = _exp(results, "E5_identity")["summary"]["n_subsets"]
        rows.append([f"Identity of planted subsets, recall / precision (N = {r['n']:,})",
                     f"{_pct(r['recall'], 0)} / {_pct(r['precision'], 0)} vs planted, "
                     f"{_pct(r['precision_true'], 0)} vs population truth", "E5",
                     f"{_g(r['mean_false_positives'])} unplanted detections per {n_sub}-subset scan on average; "
                     "the population truth counts every subset whose true RI ≥ min_effect."])
    if "artifacts" in b:
        art = b["artifacts"]
        rho_star = _nearest_in([r["rho"] for r in art if r["rho"] > 0] or [0.0], 0.5)
        n0 = min(r["n"] for r in art)
        parts = []
        for method in dict.fromkeys(r["method"] for r in art):
            vals = {r["order"]: r["ri_population"] for r in art if r["method"] == method and r["rho"] == rho_star
                    and r["n"] == n0}
            parts.append(f"{method}: {_e(vals.get(3))} / {_e(vals.get(4))}")
        rows.append([f"Discretisation artefact, order-3 / order-4 RI at ρ = {rho_star:g}", "; ".join(parts), "E6",
                     "Interactions created by thresholding a purely pairwise Gaussian latent. They are real "
                     "properties of the bits, but not of the latent."])
    if "chain_dependence" in b:
        cd = b["chain_dependence"]
        rows.append(["Chain dependence: median (max − min)/TC of a subset's chain RI, by order",
                     "; ".join(f"order {r['order']}: {_pct(r['median_spread_over_tc'])}" for r in cd["by_order"]),
                     "E7", "How much one subset's chain RI moves when the order of incomparable subsets changes. "
                           "Local RI has no such dependence."])
        rows.append(["Completeness |Σ chain RI − TC| (worst chain)", _e(cd["max_abs_sum_minus_tc"]), "E7",
                     "Numerical error of the complete decomposition (exact in theory)."])
    text = ["## Instrument error budget\n",
            _table(["Quantity", "Value", "Source", "How to read it"], rows), ""]
    text.append(_detail_tables(results))
    return "\n".join(text)


def _nearest_in(values: Sequence[float], target: float, *, log: bool = False) -> float:
    """The value closest to ``target`` (in log distance if ``log``, e.g. for sample sizes); ties go low."""
    vals = sorted(set(values))
    if log:
        return min(vals, key=lambda v: (abs(math.log(v / target)), v))
    return min(vals, key=lambda v: (abs(v - target), v))


def _zero_star(x: Any) -> str:
    """``0*`` for values that are zero to machine precision, else compact scientific notation."""
    return "0*" if _ok(x) and abs(x) < ZERO_TOL else _e(x)


def _mde_cell(r: Mapping[str, Any]) -> str:
    if r["censored"] == "below":
        return "≤ " + _e(r["mde"])
    if r["censored"] == "above":
        return "> max"
    return _e(r["mde"])


def _detail_tables(results: Mapping[str, Any]) -> str:
    b = results["budget"]
    cfg = results["config"]
    out = []
    if "noise_floor" in b:
        nf = b["noise_floor"]
        ns = _sorted_unique(r["n"] for r in nf)
        rows = []
        for k in _sorted_unique(r["order"] for r in nf):
            cells = {r["n"]: r for r in nf if r["order"] == k}
            rows.append([f"order {k}"] + [_e(cells[n]["q99"]) for n in ns])
        rows.append(["χ²₁ theory"] + [_e(next(r for r in nf if r["n"] == n)["theory_q99"]) for n in ns])
        out += ["### Noise floor: 99th percentile of null RI_plugin (nats)\n",
                "Maximum over the three strengths of lower-order structure. Read: the value of RI_plugin that only "
                "1 % of truly interaction-free subsets exceed at that N (before any multiple-testing correction).\n",
                _table(["", *[f"N = {n:,}" for n in ns]], rows), ""]
    rme = b["recommended_min_effect"]
    ns = _sorted_unique(r["n"] for r in rme["table"])
    rows = []
    for k in _sorted_unique(r["order"] for r in rme["table"]):
        cells = {r["n"]: r for r in rme["table"] if r["order"] == k}
        rows.append([f"order {k}"] + [_e(cells[n]["value"]) for n in ns])
    rows.append(["**whole scan (max)**"] + [f"**{_e(r['value'])}**" for r in sorted(rme["by_n"], key=lambda r: r["n"])])
    out += ["### Recommended min_effect (nats)\n",
            f"`max({_e(rme['floor'])}, Q_{rme['null_quantile']:g}(RI_corr | H0))`. The floor is a policy choice "
            "(effects weaker than it are not scientifically interesting); the quantile term is the measured noise "
            "floor and dominates at small N. Source of the quantile: "
            f"{'E1 simulations' if rme['table'][0]['source'] == 'E1' else 'χ² approximation (E1 not run)'}.\n",
            _table(["", *[f"N = {n:,}" for n in ns]], rows), ""]
    if "min_detectable_ri" in b:
        mde = b["min_detectable_ri"]
        fams = cfg["e3_family_sizes"]
        ns = _sorted_unique(r["n"] for r in mde)
        for m, rule, what in ((min(fams), "stat", "a single targeted subset, q ≤ α only"),
                              (max(fams), "survey", f"a subset inside a scan of {max(fams):,} subsets, "
                                                    "q ≤ α and RI_corr ≥ recommended min_effect")):
            rows = []
            for k in _sorted_unique(r["order"] for r in mde):
                cells = {r["n"]: r for r in mde if r["order"] == k and r["family_size"] == m and r["rule"] == rule}
                rows.append([f"order {k}"] + [_mde_cell(cells[n]) for n in ns])
            th = {r["n"]: r for r in mde if r["order"] == min(r2["order"] for r2 in mde) and r["family_size"] == m
                  and r["rule"] == rule}
            rows.append(["noncentral χ² theory"] + [_e(th[n]["theory_mde"]) for n in ns])
            out += [f"### Minimum detectable RI at {cfg['e3_power']:.0%} power (nats): {what}\n",
                    "\"≤ x\": already detected at the smallest planted RI x; \"> max\": not reached at the largest. "
                    "Theory: noncentral χ²₁ with λ = 2N·RI (Bonferroni α/m for the scan).\n",
                    _table(["", *[f"N = {n:,}" for n in ns]], rows), ""]
    if "timing" in b:
        tm = b["timing"]
        ns = _sorted_unique(r["n"] for r in tm)
        rows = []
        for w in _sorted_unique(r["width"] for r in tm):
            cells = {r["n"]: r for r in tm if r["width"] == w}
            rows.append([f"width {w:g}"] + [f"{_g(cells[n]['median_error'])} [{_g(cells[n]['q25'])}, "
                                             f"{_g(cells[n]['q75'])}] · {_pct(cells[n]['detect_rate'], 0)}"
                                             for n in ns])
        out += ["### Onset timing error (checkpoints): median [25th, 75th percentile] · detection rate\n",
                _table(["ramp", *[f"N = {n:,}" for n in ns]], rows), ""]
    if "artifacts" in b:
        art = b["artifacts"]
        n0 = min(r["n"] for r in art)
        rhos = _sorted_unique(r["rho"] for r in art)
        rows = []
        for method in dict.fromkeys(r["method"] for r in art):
            for j in _sorted_unique(r["order"] for r in art):
                cells = {r["rho"]: r for r in art if r["method"] == method and r["order"] == j and r["n"] == n0}
                rows.append([method, str(j)] + [_zero_star(cells[rho]["ri_population"]) if rho in cells else "–"
                                                for rho in rhos])
        out += ["### Discretisation artefacts: population local RI of the thresholded bits (nats)\n",
                "Order 2 is the genuine pairwise dependence of the latent; every order ≥ 3 value is created by "
                f"thresholding. 0* = zero to machine precision (|RI| < {ZERO_TOL:g}).\n",
                _table(["binarisation", "order", *[f"ρ = {rho:g}" for rho in rhos]], rows), ""]
    if "chain_dependence" in b:
        rows = [[str(r["order"]), _pct(r["median_spread_over_tc"]), _pct(r["p90_spread_over_tc"]),
                 _pct(r["max_spread_over_tc"]), _pct(r["median_spread_over_mean"], 0),
                 _pct(r["median_order_share_range"])] for r in b["chain_dependence"]["by_order"]]
        out += ["### Chain dependence of chain RI\n",
                _table(["order", "median spread/TC", "90th pct", "max", "median spread/mean RI",
                        "median range of the order's share of TC"], rows), ""]
    return "\n".join(out)


def _section_e1(results: Mapping[str, Any], figs: Mapping[str, str]) -> str:
    exp = _exp(results, "E1_null")
    if not exp:
        return ""
    s, cfg = exp["summary"], results["config"]
    f = s["fpr"]["chi2"]
    a_lines = []
    for a in cfg["fpr_alphas"]:
        x = f[f"{a:g}"]
        w = x["max_cell"]
        a_lines.append(f"* α = {a:g}: pooled FPR {_pct(x['pooled'], 2)} over {x['n_cells']} cells (range "
                       f"{_pct(x['min'])}–{_pct(x['max'])}). Cells with significantly too many false alarms (exact "
                       f"binomial test, one-sided {x['tail']:g}, the 3σ tail): {x['n_liberal']}; significantly too "
                       f"few (conservative): {x['n_conservative']}. Highest cell: cards "
                       f"{w['cards']}, strength {_g(w['strength'])}, N = {_n(w['n'])}: {_pct(w[f'fpr_{a:g}'], 2)} "
                       f"± {_pct(w[f'fpr_{a:g}_se'], 2)}.")
    boundary = s["boundary_cells"]
    if boundary:
        worst = boundary[0]
        bnd = (f"* **Boundary cells** ({len(boundary)} of {s['n_cells']}): more than "
               f"{_pct(s['boundary_frac'], 0)} of the replicates have RI_plugin = 0 exactly, up to "
               f"{_pct(worst['frac_zero'], 0)} (cards {worst['cards']}, strength {_g(worst['strength'])}, N = "
               f"{_n(worst['n'])}). There the sample has an empty cell in both parity classes of the top-order "
               "contrast, so the empirical table lies in the closure of the lower-order model and p = 1. Such a "
               "cell cannot produce a false positive, but it is also blind: a real interaction of the same order "
               "would be missed. It happens when strong lower-order structure concentrates the table on few cells, "
               "and it fades as N grows.")
    else:
        bnd = "* No boundary cells (RI_plugin = 0 exactly in at most " f"{_pct(s['boundary_frac'], 0)} of replicates)."
    boot = ""
    if "parametric_bootstrap" in s["fpr"]:
        bb = s["fpr"]["parametric_bootstrap"]
        boot = ("\n**Parametric-bootstrap null** (first " f"{_n(cfg['e1_bootstrap_reps'])} replicates of every cell, "
                f"{cfg['e1_n_null']} draws each, same data as χ²): pooled FPR " + ", ".join(
                    f"{_pct(bb[f'{a:g}']['pooled'], 2)} at α = {a:g} ({bb[f'{a:g}']['n_liberal']} liberal cells)"
                    for a in cfg["fpr_alphas"]) +
                ". Its p-values live on the grid (1 + j)/(n_null + 1), which limits the resolution near small α.\n")
    cat = [r for r in exp["rows"] if not r["binary"] and r["null"] == "chi2"]
    cat_txt = ""
    if cat:
        top = max(cfg["fpr_alphas"])
        cat_txt = (f"\n**Categorical check** (cards {', '.join(sorted({r['cards'] for r in cat}))}, df = "
                   f"{cat[0]['df']}): FPR at α = {top:g} ranges {_pct(min(r[f'fpr_{top:g}'] for r in cat))}–"
                   f"{_pct(max(r[f'fpr_{top:g}'] for r in cat))}, and the 99th percentile of RI_plugin follows "
                   "χ²_df(0.99)/2N (`ri_plugin_q99` vs `theory_q99` in `E1_null.csv`).\n")
    lo, hi = s["mean_g2_over_df_regular"]
    conc = "; ".join(f"{e['cards']} at strength {_g(e['strength'])}: {_g(e['median_effective_cells'], 2)} of "
                     f"{e['n_cells']}" for e in s["effective_cells"] if e["strength"] > 0)
    return f"""## E1 Null: false-positive rate and noise floor

**Question.** When a subset has *no* top-order interaction, how often does the instrument claim one, and how large
are the RI values that pure sampling noise produces?

**Design.** Tables over exactly k variables whose log-probability has only terms of order < k (random spin couplings
of scale "strength" ∈ {{{', '.join(_g(x) for x in cfg['e1_strengths'])}}}), so the true local RI of the k-set is
exactly 0. A fresh random table per replicate, {_n(cfg['e1_reps'])} replicates per (k, strength, N); N ∈
{{{', '.join(_n(n) for n in cfg['n_grid'])}}}. How concentrated the null tables are (median effective number of
cells, exp(entropy)): {conc}. The largest strength is a deliberately harsh case.

**Result.**

{chr(10).join(a_lines)}
{bnd}
* Outside boundary cells: Kolmogorov–Smirnov distance of the χ² p-values from U(0, 1) at most
  {_g(s['ks_max_regular'])} (median over all cells {_g(s['ks_median'])}); mean of G²/df (theory 1) between
  {_g(lo)} and {_g(hi)}.
{boot}{cat_txt}
![E1]({figs.get('E1', '')})

**How to read.** Panel A is the calibration of the p-value: the lines should sit on the nominal α (a band reaching
far below α comes from boundary cells). Panel B is the noise floor: because G² = 2N·RI_plugin is χ²-distributed,
the floor falls as 1/N and is (for binary variables, df = 1) the same for every order. Higher orders are not
noisier per subset; they are costlier because there are many more candidate subsets (multiplicity, see E3), and
because sparse tables (boundary cells) appear at larger N. Panel C should be flat; a bar at p = 1 is the boundary
point mass.
"""


def _section_e2(results: Mapping[str, Any], figs: Mapping[str, str]) -> str:
    exp = _exp(results, "E2_bias")
    if not exp:
        return ""
    s, cfg = exp["summary"], results["config"]
    n_b = _budget_n(results)
    at = sorted((r for r in exp["rows"] if r["n"] == n_b), key=lambda r: (r["order"], r["target_ri"]))
    rows = [[str(r["order"]), _e(r["true_ri"]), _e(r["bias_plugin"]), _e(r["bias_corrected"]),
             _e(r["rmse_corrected"]), _pct(r["rel_rmse_corrected"], 0), _pct(r["coverage"], 0)] for r in at]
    return f"""## E2 Bias, RMSE and CI coverage

**Question.** When an interaction of known size is present, how accurate is the reported RI, and can its
bootstrap confidence interval be trusted?

**Design.** For each order k and each target true RI ∈ {{{', '.join(_e(t) for t in cfg['e2_targets'])}}} nats: a fresh
random lower-order background (spin scale {_g(cfg['e2_background'])}) per replicate, and the top-order coupling solved
so that the population local RI equals the target exactly. {_n(cfg['e2_reps'])} replicates per cell; the first
{_n(cfg['e2_ci_reps'])} also get a row-bootstrap {cfg['e2_ci_level']:.0%} percentile CI ({cfg['e2_n_boot']} resamples).

**Result.** Median plug-in bias = {_g(s['median_bias_ratio_plugin'])} × df/2N (theory: 1). Median SD of RI_plugin =
{_g(s['median_sd_ratio_plugin'])} × the noncentral-χ² prediction √(2(df + 2λ))/2N. Once λ = 2N·RI ≥ 20, the worst
relative bias of RI_corr is {_pct(s['max_abs_rel_bias_corrected_strong'])}. CI coverage: median
{_pct(s['median_coverage_strong'], 0)} for λ ≥ 20 (range {_pct(s['coverage_strong'][0], 0)}–{_pct(s['coverage_strong'][1], 0)})
and {_pct(s['median_coverage_weak'], 0)} for λ < 20 (range {_pct(s['coverage_weak'][0], 0)}–{_pct(s['coverage_weak'][1], 0)}).

At N = {n_b:,}:

{_table(["order", "true RI", "bias plug-in", "bias corrected", "RMSE corrected", "RMSE / RI", "CI coverage"], rows)}

![E2]({figs.get('E2', '')})

**How to read.** Asymptotic theory predicts that, plotted against λ, all orders and sample sizes collapse onto one
curve: the plug-in estimate is high by df/2N (panel A, blue points at 1) and the corrected estimate removes that
(orange points at 0), except at small λ, where the truncation max(0, ·) biases RI_corr upward. The error bars are
the Monte Carlo standard errors; at large λ the sampling spread of RI dwarfs df/2N, so those points are noisy in
these units. Panel B compares the observed spread with the noncentral-χ² prediction (1 = agreement). Panel C: the
nominal coverage line is the target. Near RI = 0 the estimator sits on the boundary RI ≥ 0, where percentile
intervals have no coverage guarantee, so CIs are most meaningful for detected subsets.
"""


def _section_e3(results: Mapping[str, Any], figs: Mapping[str, str]) -> str:
    exp = _exp(results, "E3_power")
    if not exp:
        return ""
    s, cfg = exp["summary"], results["config"]
    mono = s["monotone_in_n"]
    lam = results["budget"]["analytic"]["lambda_star_df1"]
    return f"""## E3 Power and minimum detectable RI

**Question.** How large must an interaction be for the survey to see it?

**Design.** The replicates of E2. The planted subset is tested together with m − 1 interaction-free subsets
(m ∈ {{{', '.join(_n(m) for m in cfg['e3_family_sizes'])}}}; their p-values are U(0, 1), the worst case for
Benjamini–Hochberg because nothing else is discovered), and detected when q ≤ α (rule "stat") or when also
RI_corr ≥ the recommended min_effect (rule "survey"). The minimum detectable RI (MDE) is where the power curve
(made monotone by isotonic regression) crosses {cfg['e3_power']:.0%}, interpolated linearly in log RI and probit
power.

**Result.** Empirical MDE / theory: median {_g(s['mde_over_theory_median'])} (range
{_g(s['mde_over_theory_range'][0])}–{_g(s['mde_over_theory_range'][1])}) over uncensored cells. Power is monotone in N:
{mono['violations']} of {mono['pairs']} adjacent-N comparisons drop by more than {mono['tolerance_se']:g} SE.
For df = 1 the theory says MDE ≈ λ*/(2N) with λ* = {', '.join(f'{_g(v)} (m = {k})' for k, v in lam.items())}
at α = {cfg['alpha']:g}.

![E3 power]({figs.get('E3', '')})

![E3 MDE]({figs.get('E3_mde', '')})

**How to read.** Each curve in the first figure is the probability that a planted interaction of the given true RI
is detected, for one N. In the second figure the gray line is the theory: the statistical MDE falls as λ*/2N.
Under the survey rule it cannot fall below the min_effect floor, because then the effect-size filter rather than the
significance test decides; at large N more samples no longer lower the detection threshold, by design. Within a
scan of m subsets the multiplicity correction raises λ*, roughly in proportion to log m.
"""


def _section_e4(results: Mapping[str, Any], figs: Mapping[str, str]) -> str:
    exp = _exp(results, "E4_timing")
    if not exp:
        return ""
    s, cfg = exp["summary"], results["config"]
    r0 = exp["rows"][0]
    return f"""## E4 Timing of activation events

**Question.** When an interaction switches on during training, how accurately does the event detector date the
onset?

**Design.** A {cfg['e4_order']}-way spin coupling θ(t) = θ_max·σ((t − t*)/width) over a fixed random lower-order
background, on {cfg['e4_checkpoints']} checkpoints (t* = {_g(cfg['e4_t_star'])}; width 0 is a step), with θ_max set
so that the final population RI is {_e(r0['final_ri'])} nats. At every checkpoint N fresh samples are scanned
(`ri_scan`, all subsets up to order {cfg['e4_order']}), the scans form an `RICurve`, and `detect_events` runs with
on = min_effect = {_e(s['min_effect'])}, off = on/2 and persistence {s['persistence']}. The true onset is the first
checkpoint whose population RI ≥ min_effect. {_n(cfg['e4_reps'])} replicates per (width, N).

![E4]({figs.get('E4', '')})

**How to read.** Error = detected − true onset, in checkpoints (positive = late). The detector needs RI_corr ≥
min_effect *and* a significant q-value for {s['persistence']} consecutive checkpoints. Expected pattern: at small
N the RI must first climb above the noise floor, so onsets are reported late, and more so for gradual ramps, whose
RI spends more checkpoints near the threshold; at large N the error shrinks toward 0, and negative errors (early
detections) appear when noise lifts an RI that is just below min_effect. The bracket [step_lo, step_hi] of an event
is one checkpoint wide; a replay with denser checkpoints narrows it.
"""


def _section_e5(results: Mapping[str, Any], figs: Mapping[str, str]) -> str:
    exp = _exp(results, "E5_identity")
    if not exp:
        return ""
    s, cfg = exp["summary"], results["config"]
    planted = "; ".join(f"{{{p['subset'].replace('-', ',')}}} (order {p['order']}, θ = {_g(p['theta'])}, true RI "
                        f"{_e(p['true_ri'])})" for p in s["planted"])
    largest = ", ".join(f"{{{p['subset'].replace('-', ',')}}} {_e(p['true_ri'])}" for p in s["largest_unplanted"][:3])
    rows = [[_n(r["n"]), r["rule"], _pct(r["recall"], 0), _pct(r["precision"], 0), _pct(r["precision_true"], 0),
             _pct(r["exact_recovery"], 0), _g(r["mean_false_positives"]), _g(r["mean_fp_spurious"]),
             _g(r["mean_fp_real_unplanted"]), _g(r["mean_fp_related"])]
            for r in sorted(exp["rows"], key=lambda r: (r["rule"] != "survey", r["n"]))]
    freq = "; ".join(f"N = {_n(int(n))}: " + ", ".join(f"{{{x['subset'].replace('-', ',')}}} {_pct(x['rate'], 0)}"
                                                         for x in v)
                     for n, v in s["frequent_false_positives"].items() if v) or "none"
    return f"""## E5 Identity and specificity

**Question.** Does the instrument report the *right* subsets at the *right* order, and nothing else?

**Design.** d = {cfg['e5_d']} binary variables. Planted: {planted}. A weak random background (spin scale
{_g(cfg['e5_background'])}) of biases on all variables and couplings on all other pairs; no background inside the
planted triple, which therefore stays a pure synergy. Full scan to order {cfg['e5_max_order']} ({s['n_subsets']}
subsets, BH within the scan), {_n(cfg['e5_reps'])} replicates per N. Rule "survey": q ≤ α and RI_corr ≥ the
recommended min_effect; rule "stat": q ≤ α only. Largest unplanted population RIs: {largest}
({s['n_real_unplanted']} unplanted subsets have RI ≥ min_effect = {_e(cfg['min_effect'])}).

**Result.**

{_table(["N", "rule", "recall", "precision (planted)", "precision (population)", "exact recovery", "FP / scan",
         "spurious", "real unplanted", "related"], rows)}

Most frequent false positives (survey rule): {freq}.

![E5]({figs.get('E5', '')})

**How to read.** Recall is the fraction of the two planted subsets found. Precision (planted) is the fraction of
detections that are planted; precision (population) is the fraction whose population RI is at least min_effect,
planted or not, which is the instrument's own precision (a detected real background interaction is not an
instrument error). "Spurious" false positives have population RI below min_effect; "related" ones are proper sub- or
supersets of a planted subset, i.e. errors of order rather than of identity. Inside the planted subsets the largest
population RI of a proper subset is {_e(s['max_ri_inside_planted'])} nats (induced only by the weak biases), so the
3-way interaction must be found as a triple, without lower-order precursors; a detected {{0,1}} would be an order
error.
"""


def _section_e6(results: Mapping[str, Any], figs: Mapping[str, str]) -> str:
    exp = _exp(results, "E6_discretization")
    if not exp:
        return ""
    s, cfg = exp["summary"], results["config"]
    ms = s.get("median_symmetry")
    verdict = ""
    if ms:
        status = "VERIFIED" if ms["verified"] else "NOT VERIFIED"
        verdict = (f"**Median thresholding, sign symmetry: {status}.** Largest |population RI| at odd orders: "
                   f"{_e(ms['max_abs_odd_order_ri'])} nats (zero to machine precision; tolerance "
                   f"{ZERO_TOL:g}). Smallest even-order (≥ 4) RI over ρ > 0: {_e(ms['min_even_order_ri'])} nats. "
                   f"In the sampled scans at N = {_n(ms['sample_n'])}, the bias-corrected mean RI of order-3 subsets "
                   f"(mean RI_plugin − df/2N, first-order unbiased) is at most "
                   f"{_e(ms['sample_order3_max_abs_excess'])} nats in absolute value"
                   + (f", {_g(ms['sample_order3_max_abs_z'])} SE from 0" if ms["sample_reps"] >= MIN_REPS_FOR_SE
                      else f" ({ms['sample_reps']} replicates: too few for a standard error)")
                   + ". (The mean of RI_corr itself is positive even at a true zero, because max(0, ·) truncates "
                   "the noise.)\n")
    asym = ", ".join(f"{m}: {_e(v)}" for m, v in s["max_order3_ri"].items())
    n_max = max(r["n"] for r in exp["rows"])
    det = [r for r in exp["rows"] if r["n"] == n_max and r["order"] >= 3]
    worst = max(det, key=lambda r: r["detect_rate"]) if det else None
    det_txt = ""
    if worst:
        det_txt = (f" At N = {_n(n_max)} the highest detection rate of an artefact (order ≥ 3) is "
                   f"{_pct(worst['detect_rate'], 1)} of subsets ({worst['method']}, ρ = {_g(worst['rho'])}, order "
                   f"{worst['order']}, population RI {_e(worst['ri_population'])}).")
    return f"""## E6 Discretisation artefacts (RQ0)

**Question.** RI needs discrete variables, so continuous activations are thresholded. How much higher-order
"interaction" does thresholding create by itself?

**Design.** An equicorrelated Gaussian latent (correlation ρ, purely pairwise structure) is binarised by the
median, by zero with latent mean {_g(cfg['e6_latent_mean'])} (a biased pre-activation; the bit is on with probability
Φ({_g(cfg['e6_latent_mean'])})), or at the {_g(cfg['e6_quantile'])} quantile. Exact population local RI by order from the
one-factor integral, plus sampled scans of {cfg['e6_d']} variables to order {cfg['e6_max_order']}.

**Result.** {verdict}
Largest order-3 RI by binarisation: {asym} nats.{det_txt}

![E6]({figs.get('E6', '')})

**How to read.** This is a systematic error of the instrument, not a sampling error: more samples make it *more*
detectable, not less. The mechanism: thresholding a Gaussian is a nonlinear map, and the log-probabilities of the
resulting bits contain higher-order terms even though the latent has only pairwise couplings (the "dichotomized
Gaussian" of Amari et al. 2003 and Macke et al. 2009). With the threshold at the median the bit distribution is
invariant under flipping every bit, which forces every odd-order log-linear coefficient to zero; even orders
survive. Moving the threshold off the median breaks the symmetry and odd orders appear. Practical rule: for
binarised hidden units, treat order-3/4 events whose RI is comparable to these values (at the units' measured
correlations) as possible artefacts.
"""


def _section_e7(results: Mapping[str, Any], figs: Mapping[str, str]) -> str:
    exp = _exp(results, "E7_chain")
    if not exp:
        return ""
    s, cfg = exp["summary"], results["config"]
    cards = ", ".join("×".join(str(c) for c in cc) for cc in cfg["e7_cards"])
    return f"""## E7 Chain dependence and completeness

**Question.** Chain RI (the complete decomposition of the KL error along a hierarchical chain) depends on the
order in which incomparable subsets enter the chain. How much?

**Design.** {s['n_tables']} random positive tables (cards {cards}; log-potentials i.i.d. Normal(0,
{_g(cfg['e7_scale'])}²) per cell, so interactions of every order are present). Chains: graded (graded-lex), reverse
graded (reverse lexicographic within each order), greedy (largest RI gain first: mode interaction selection) and
{cfg['e7_random_chains']} random admissible chains per table.

**Result.** Σ RI over every chain equals the total correlation to {_e(s['max_abs_sum_minus_tc'])} nats (worst case
over all chains and tables). By order, the median spread (max − min over chains) of a subset's chain RI is
{', '.join(f"{_pct(r['median_spread_over_tc'])} of TC (order {r['order']})" for r in s['by_order'])}; relative to
the subset's own mean chain RI the median spread is
{', '.join(f"{_pct(r['median_spread_over_mean'], 0)} (order {r['order']})" for r in s['by_order'])}. The graded
chain's value differs from the chain-independent local RI by a median of
{', '.join(f"{_pct(r['median_graded_minus_local_over_tc'])} of TC (order {r['order']})" for r in s['by_order'])}.
The full variable set is excluded from these statistics: it is the last step of every complete chain, so its chain
RI is the same in all of them (largest spread observed: {_e(s['max_full_set_spread'])} nats).

![E7]({figs.get('E7', '')})

**How to read.** Completeness (panel B) is a numerical identity check: the chain terms always add up to the total
correlation, whichever chain is used. The spread (panel A) is a property of the definition, not an estimation error:
the same subset can be credited with very different amounts of information depending on what entered the chain
before it. This is why the survey's events use the chain-independent local RI, and why every chain RI that is
reported must name its chain.
"""


def _caveats(results: Mapping[str, Any]) -> str:
    return """## Caveats and open items

* **Soft output columns are not calibrated here.** All experiments sample hard codes. The survey's positive controls
  enter the model's predictive distribution as a soft column, which removes that variable's sampling noise. For a
  subset whose table is fully determined by such a column the null is degenerate (RI_plugin = 0 exactly); in general
  `tycho/ri/estimators.py` states that the χ² null is then conservative. A soft-column null experiment is still to
  be done.
* **Independent samples.** Every experiment draws i.i.d. samples. Probe sets in the survey are i.i.d. by construction
  (fresh inputs per checkpoint); dependent samples (e.g. tokens of one sequence) would need an effective N.
* **Binary noise floor.** The (order, N) tables are for binary variables (df = 1). For categorical variables the
  floor scales with the χ²_df quantile (E1's categorical rows check df > 1).
* **RI convention.** The local RI and the chain RI follow `docs/SPEC.md` §2.1; the canonical chain of Enouen &
  Sugiyama (TMLR 2026) still has to be checked against the paper.
* **Monte Carlo error.** Quick runs exist to exercise the code; only a `full` run supports quantitative claims.
"""


def write_report(results: dict, out_dir: Path) -> Path:
    """Write ``CALIBRATION.md``, ``calibration.json`` and ``figures/*.png`` into ``out_dir``.

    Args:
        results: the dict returned by :func:`~tycho.calibration.suite.run_calibration` (or
            :func:`~tycho.calibration.suite.load_results`).
        out_dir: output directory (created).

    Returns:
        The path of ``CALIBRATION.md``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = jsonable(results)
    figs = _figures(results, out_dir / "figures")
    doc = calibration_document(results)
    tmp = out_dir / "calibration.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, out_dir / "calibration.json")
    sections = [_intro(results), _budget_section(results)]
    sections += [fn(results, figs) for fn in (_section_e1, _section_e2, _section_e3, _section_e4, _section_e5,
                                              _section_e6, _section_e7)]
    sections.append(_caveats(results))
    md = "\n".join(s for s in sections if s)
    path = out_dir / "CALIBRATION.md"
    path.write_text(md, encoding="utf-8")
    logger.info("calibration report written to %s (%d figures)", path, len(figs))
    return path
