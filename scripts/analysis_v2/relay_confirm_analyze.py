"""Frozen analysis script for the 88-core relay vs. textbook confirmation study.

Specification: ``docs/results/relay_confirmation_preregistration.md``.

Computes:
  * Per-run 5-way classification {U, R, T, S, M} from saved final weights and curves.
  * Dual reporting on every cell & comparison:
      - ``source_strict`` (T + S): recommended primary (excludes ambiguous M runs).
      - ``source_inclusive`` (T + S + M): draft primary (all formed runs with delta = 0).
      - ``textbook`` (T): primary for C4 and sensitivity analysis elsewhere.
    Any verdict that changes across definitions is flagged explicitly.
  * 3-tier verdicts (flip / shift / unconfirmed) for n = 24 and (if present) n = 48 top-up,
    plus Wilson 95% CIs, exact two-sided McNemar p-values, exact 3x2 Freeman-Halton test (S1b),
    10,000-resample fixed-seed bootstrap CIs for medians, 2h vs 4h split flags (|diff| >= 6),
    and U > 6 unstable cell flags.
  * Instrument check IC-3 (``--ic3``): verifies the frozen classifier on the exploratory pilot
    JSONs against every exploratory count in §3 of ``notes/relay_confirmation_plan_draft_2026-11-22.md``.

Usage (from repository root):
  PYTHONPATH=. python scripts/analysis_v2/relay_confirm_analyze.py --ic3
  PYTHONPATH=. python scripts/analysis_v2/relay_confirm_analyze.py --work work/relay_confirm --jobs 16
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if "relay_confirm" in sys.modules:
    rc = sys.modules["relay_confirm"]
else:
    _rc_spec = importlib.util.spec_from_file_location("relay_confirm", Path(__file__).with_name("relay_confirm.py"))
    rc = importlib.util.module_from_spec(_rc_spec)
    sys.modules["relay_confirm"] = rc
    _rc_spec.loader.exec_module(rc)

if "relay_followup_pilot" in sys.modules:
    fp = sys.modules["relay_followup_pilot"]
else:
    _fp_spec = importlib.util.spec_from_file_location("relay_followup_pilot", Path(__file__).with_name("relay_followup_pilot.py"))
    fp = importlib.util.module_from_spec(_fp_spec)
    sys.modules["relay_followup_pilot"] = fp
    _fp_spec.loader.exec_module(fp)

DELTAS = tuple(range(-4, 32))
D_MAX = 40
SECOND_KEY_LOSS = 0.4
BAND_LEVEL = 0.85
BOOT_N = 10_000
BOOT_SEED = 20261124

_SAMPLE_CACHE: dict[int, dict] = {}


def get_sample(rep_min: int) -> dict:
    """2,000 common single-layout sequences (seed 777), identical to ``relay_followup_pilot.sample``."""
    if rep_min not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[rep_min] = fp.sample(rep_min, 32)
    return _SAMPLE_CACHE[rep_min]


# -------------------------------------------------------------------------------------- classification
def classify_metrics(loss_rep: float, relay_delta: int, deep_decoy_over_clean: float, source_l1_end: float) -> str:
    """Exact 5-way category {U, R, T, S, M} from prereg §2.3."""
    if not math.isfinite(loss_rep) or loss_rep >= 2.0:
        return "U"
    if int(relay_delta) != 0:
        return "R"
    if math.isfinite(deep_decoy_over_clean) and deep_decoy_over_clean >= 0.9:
        return "T"
    if math.isfinite(source_l1_end) and source_l1_end > 0.5:
        return "S"
    return "M"


def classify_npz(npz_path: Path, samples: dict[int, dict] | None = None) -> dict[str, Any]:
    """Classify one confirmation .npz from its final weights and recorded curves."""
    import torch

    from tycho.tasks.induction import Induction

    with np.load(npz_path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta_json"]))
        state = {k[6:]: torch.as_tensor(f[k]) for k in f.files if k.startswith("param/")}
        steps = f["lc_step"].astype(np.int64)
        stats = {k: f[f"stat_{k}"].astype(np.float64) for k in rc.STAT_KEYS if f"stat_{k}" in f.files}

    cfg = meta["cfg"]
    seed = int(meta["seed"])
    rep_min = int(cfg["rep_min"])
    nh = int(cfg["n_head"])

    task = Induction(cfg, seed)
    model = task.build_model()
    model.load_state_dict(state)
    model.eval()

    # Exact attention_summary (same DELTAS, D_MAX, and modal_top_delta rule as relay_offsets_pilot)
    pr = task._probe
    x_pr, rep_pr, src_pr = pr["x"], pr["rep"].numpy(), pr["src"].numpy()
    maps = [a.numpy() for a in model.attention_maps(x_pr)]
    T = x_pr.shape[1]
    t_idx = np.arange(T)

    l0_heads = []
    for h in range(nh):
        a = maps[0][:, h]
        prof = np.array([a[:, t_idx[d:], t_idx[d:] - d].mean() for d in range(D_MAX + 1)])
        top = np.argsort(-prof)[:3]
        l0_heads.append({
            "head": f"l0h{h}",
            "top_offsets": [[int(d), round(float(prof[d]), 4)] for d in top],
            "attention_d1": float(prof[1]),
            "attention_d2": float(prof[2]),
        })

    bi, ti = np.nonzero(rep_pr)
    lag = ti - src_pr[bi, ti] + 1
    lags = sorted(set(lag.tolist()))
    best_l1 = None
    l1_modal_deltas = []
    for h in range(nh):
        a = maps[1][:, h]

        def profile(sel: np.ndarray) -> np.ndarray:
            prof = np.full(len(DELTAS), np.nan)
            for k, dl in enumerate(DELTAS):
                j = src_pr[bi[sel], ti[sel]] + dl
                ok = (j >= 0) & (j <= ti[sel])
                if ok.any():
                    prof[k] = a[bi[sel][ok], ti[sel][ok], j[ok]].mean()
            return prof

        by_lag = np.array([profile(lag == L) for L in lags])
        top_by_lag = [DELTAS[int(np.nanargmax(row))] for row in by_lag]
        modal, _ = Counter(top_by_lag).most_common(1)[0]
        at_modal = by_lag[:, DELTAS.index(modal)]
        med_att = float(np.nanmedian(at_modal))
        l1_modal_deltas.append(int(modal))
        if best_l1 is None or med_att > best_l1["median_attention_at_modal_delta"]:
            best_l1 = {"head": f"l1h{h}", "modal_top_delta": int(modal), "median_attention_at_modal_delta": med_att}
    relay_delta = int(best_l1["modal_top_delta"])

    # Exact common-sequence decoy & loss evaluation (2,000 sequences, seed 777)
    smp = samples[rep_min] if (samples and rep_min in samples) else get_sample(rep_min)
    probs = fp.probabilities(model, smp["x"])
    sc = fp.score(fp.true_token(probs, smp), smp)

    source_l1_end = float(stats["source_l1"][-1])
    cat = classify_metrics(sc["loss_rep"], relay_delta, sc["deep_decoy_over_clean"], source_l1_end)

    # Curve-derived quantities
    step_idx = {int(s): i for i, s in enumerate(steps)}
    lr = float(meta["train"]["lr"])
    bs = int(meta["train"]["batch_size"])
    r1_step_target = int(round(0.3 / lr))  # 300 @ 1e-3, 600 @ 5e-4, 150 @ 2e-3
    r1_idx = int(np.argmin(np.abs(steps - r1_step_target)))
    n9600_target = int(round(9600 / bs))
    n9600_idx = int(np.argmin(np.abs(steps - n9600_target)))

    # First layer past 0.85 band attention (S3c)
    b0 = stats["band_l0"]
    b1 = stats["band_l1"]
    t0_idx = np.flatnonzero(b0 >= BAND_LEVEL)
    t1_idx = np.flatnonzero(b1 >= BAND_LEVEL)
    t0_85 = int(steps[t0_idx[0]]) if len(t0_idx) else None
    t1_85 = int(steps[t1_idx[0]]) if len(t1_idx) else None
    if t0_85 is None and t1_85 is None:
        l1_first_85 = None
    elif t0_85 is None:
        l1_first_85 = True
    elif t1_85 is None:
        l1_first_85 = False
    elif t1_85 == t0_85:
        l1_first_85 = None  # tie on the same grid step -> undecidable
    else:
        l1_first_85 = bool(t1_85 < t0_85)

    # Two-stage assembly steps (P4c)
    on_src = np.flatnonzero(stats["source_l1"] > 0.5)
    source_step = int(steps[on_src[0]]) if len(on_src) else None
    low_loss = stats["loss_rep"] < SECOND_KEY_LOSS
    stay_low = np.flatnonzero(low_loss & (np.cumsum(~low_loss[::-1])[::-1] == 0))
    second_key_step = int(steps[stay_low[0]]) if len(stay_low) else None

    return {
        "name": meta["name"],
        "cell_id": meta["cell_id"],
        "suite": meta["suite"],
        "group": meta["group"],
        "seed": seed,
        "n_head": nh,
        "rep_min": rep_min,
        "d_model": int(cfg["d_model"]),
        "lr": lr,
        "batch_size": bs,
        "cat": cat,
        "loss_rep": float(sc["loss_rep"]),
        "loss_other": float(sc["loss_other"]),
        "loss_all": float(sc["loss_all"]),
        "deep_decoy_over_clean": float(sc["deep_decoy_over_clean"]),
        "relay_delta": relay_delta,
        "source_l1_end": source_l1_end,
        "loss_rep_probe_end": float(stats["loss_rep"][-1]),
        "r1_aligned_step": int(steps[r1_idx]),
        "r1_aligned": float(stats["r1"][r1_idx]),
        "r1_at_n9600_step": int(steps[n9600_idx]),
        "r1_at_n9600": float(stats["r1"][n9600_idx]),
        "rms_y1_at_300": float(stats["rms_y1"][step_idx[300]]) if 300 in step_idx else float(stats["rms_y1"][r1_idx]),
        "rms_m0_aligned": float(stats["rms_m0"][r1_idx]) if "rms_m0" in stats else 0.0,
        "rms_m1_aligned": float(stats["rms_m1"][r1_idx]) if "rms_m1" in stats else 0.0,
        "t0_band85": t0_85,
        "t1_band85": t1_85,
        "l1_first_85": l1_first_85,
        "source_step": source_step,
        "second_key_step": second_key_step,
        "l0_heads": l0_heads,
        "l1_modal_deltas": l1_modal_deltas,
        "torch_version": meta.get("torch_version"),
        "wall_s": meta.get("wall_s"),
    }


def _classify_worker(npz_str: str) -> dict[str, Any]:
    return classify_npz(Path(npz_str))


# -------------------------------------------------------------------------------------- statistics
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def mcnemar_exact_p(xs: list[bool], ys: list[bool]) -> float:
    """Two-sided exact McNemar p-value on paired binary outcomes."""
    b = sum(x and (not y) for x, y in zip(xs, ys))
    c = sum((not x) and y for x, y in zip(xs, ys))
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    # P(Bin(n, 0.5) <= m) * 2
    tail = sum(math.comb(n, i) for i in range(m + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def freeman_halton_3x2_p(k1: int, n1: int, k2: int, n2: int, k3: int, n3: int) -> float:
    """Exact Freeman-Halton test p-value for a 3x2 contingency table."""
    K = k1 + k2 + k3
    N = n1 + n2 + n3
    denom = math.comb(N, K)

    def prob(a: int, b: int, c: int) -> float:
        return (math.comb(n1, a) * math.comb(n2, b) * math.comb(n3, c)) / denom

    p_obs = prob(k1, k2, k3)
    p_sum = 0.0
    for a in range(max(0, K - n2 - n3), min(n1, K) + 1):
        rem = K - a
        for b in range(max(0, rem - n3), min(n2, rem) + 1):
            c = rem - b
            pr = prob(a, b, c)
            if pr <= p_obs + 1e-12:
                p_sum += pr
    return float(min(1.0, p_sum))


def bootstrap_median_ci(values: list[float], n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    meds = np.median(arr[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {"n": int(arr.size), "median": round(float(np.median(arr)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Exact ROC AUC (Mann-Whitney U) predicting ``labels`` (higher score -> True)."""
    pairs = [(s, bool(y)) for s, y in zip(scores, labels) if s is not None and math.isfinite(s)]
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return round(wins / (len(pos) * len(neg)), 4)


def three_tier(kx: int, ky: int, n: int) -> str:
    """Prereg §2.4 & §2.5 three-tier verdict (n=24: 16/8 & diff>=8; n=48: 32/16 & diff>=16)."""
    if n == 24:
        hi, lo, diff = 16, 8, 8
    elif n == 48:
        hi, lo, diff = 32, 16, 16
    else:
        hi = math.ceil(n * 2 / 3)
        lo = math.floor(n / 3)
        diff = math.ceil(n / 3)
    if kx >= hi and ky <= lo:
        return "flip"
    if (kx - ky) >= diff:
        return "shift"
    return "unconfirmed"


def metric_flag(row: dict[str, Any], mode: str) -> bool:
    if mode == "source_strict":      # T + S
        return row["cat"] in ("T", "S")
    if mode == "source_inclusive":   # T + S + M
        return row["cat"] in ("T", "S", "M")
    if mode == "textbook":           # T only
        return row["cat"] == "T"
    raise ValueError(mode)


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = {k: sum(r["cat"] == k for r in rows) for k in ("U", "R", "T", "S", "M")}
    by_h: dict[str, dict[str, int]] = {}
    for nh in sorted({r["n_head"] for r in rows}):
        sub = [r for r in rows if r["n_head"] == nh]
        by_h[f"h{nh}"] = {k: sum(r["cat"] == k for r in sub) for k in ("U", "R", "T", "S", "M")}
    ts = counts["T"] + counts["S"]
    tsm = ts + counts["M"]
    t_only = counts["T"]
    head_gap_flag = False
    if "h2" in by_h and "h4" in by_h:
        g_ts = abs((by_h["h2"]["T"] + by_h["h2"]["S"]) - (by_h["h4"]["T"] + by_h["h4"]["S"]))
        g_tsm = abs((by_h["h2"]["T"] + by_h["h2"]["S"] + by_h["h2"]["M"]) - (by_h["h4"]["T"] + by_h["h4"]["S"] + by_h["h4"]["M"]))
        head_gap_flag = (g_ts >= 6) or (g_tsm >= 6)
    return {
        "n": n,
        "counts": counts,
        "by_head": by_h,
        "source_strict": ts,
        "source_strict_rate": round(ts / n, 4) if n else None,
        "source_strict_ci95": wilson_ci(ts, n),
        "source_inclusive": tsm,
        "source_inclusive_rate": round(tsm / n, 4) if n else None,
        "source_inclusive_ci95": wilson_ci(tsm, n),
        "textbook": t_only,
        "textbook_rate": round(t_only / n, 4) if n else None,
        "unstable": counts["U"] > (6 if n <= 24 else 12),
        "head_gap_ge_6": head_gap_flag,
        "r1_aligned": bootstrap_median_ci([r["r1_aligned"] for r in rows]),
    }


def paired_comparison(cid_x: str, cid_y: str, by_cell: dict[str, list[dict[str, Any]]],
                      *, primary_mode: str = "source_strict", diff_only: bool = False) -> dict[str, Any]:
    """Paired comparison X (expected higher) vs Y (expected lower) across all 3 outcome definitions."""
    rx = {r["seed"]: r for r in by_cell.get(cid_x, [])}
    ry = {r["seed"]: r for r in by_cell.get(cid_y, [])}
    common = sorted(set(rx) & set(ry))
    n = len(common)
    if n == 0:
        return {"x": cid_x, "y": cid_y, "n": 0, "status": "missing"}
    u_x = sum(rx[s]["cat"] == "U" for s in common)
    u_y = sum(ry[s]["cat"] == "U" for s in common)
    unstable = (u_x > (6 if n <= 24 else 12)) or (u_y > (6 if n <= 24 else 12))

    out: dict[str, Any] = {"x": cid_x, "y": cid_y, "n": n, "unstable": unstable, "by_mode": {}}
    for mode in ("source_strict", "source_inclusive", "textbook"):
        bx = [metric_flag(rx[s], mode) for s in common]
        by = [metric_flag(ry[s], mode) for s in common]
        kx, ky = sum(bx), sum(by)
        p_mc = mcnemar_exact_p(bx, by)
        if unstable:
            verdict = "unstable"
        elif diff_only:
            thresh = 8 if n <= 24 else 16
            verdict = "pass" if (kx - ky) >= thresh else "fail"
        else:
            verdict = three_tier(kx, ky, n)
        out["by_mode"][mode] = {
            "kx": kx,
            "ky": ky,
            "diff": kx - ky,
            "mcnemar_p": round(p_mc, 6),
            "verdict": verdict,
        }
    out["primary_mode"] = primary_mode
    out["verdict"] = out["by_mode"][primary_mode]["verdict"]
    out["definition_sensitive"] = len({out["by_mode"][m]["verdict"] for m in ("source_strict", "source_inclusive")}) > 1
    out["textbook_sensitive"] = out["by_mode"][primary_mode]["verdict"] != out["by_mode"]["textbook"]["verdict"]
    return out


# -------------------------------------------------------------------------------------- IC-3 check
def run_ic3_check() -> dict[str, Any]:
    """Verify frozen classifier reproduces §3 exploratory table counts from pilot JSONs (IC-3)."""
    base = REPO / "docs" / "results" / "v2_analysis"
    dp = json.loads((base / "relay_period_pilot.json").read_text())["runs"]
    dn = json.loads((base / "relay_noise_pilot.json").read_text())["runs"]
    dw_list = json.loads((base / "relay_window_pilot.json").read_text())["runs"]
    dw = {}
    for r in dw_list:
        dw.setdefault(r["arm"], []).append(r)

    def count_runs(runs: list[dict]) -> tuple[int, int, int]:
        cats = [
            classify_metrics(
                r["loss_rep"] if r.get("loss_rep") is not None else r["loss_rep_probe"],
                r["relay_delta"],
                r["deep_decoy_over_clean"],
                r["source_l1_at_the_end"],
            )
            for r in runs
        ]
        t = sum(c == "T" for c in cats)
        ts = sum(c in ("T", "S") for c in cats)
        tsm = sum(c in ("T", "S", "M") for c in cats)
        return tsm, t, ts

    # Expected (T+S+M, T) from §3 of notes/relay_confirmation_plan_draft_2026-11-22.md
    expected = [
        ("A1-1", dp["lr5e-4_natural"], 6, 6),
        ("A1-2", dp["b128_natural"], 6, 6),
        ("A1-3", dp["b64_natural"], 6, 5),
        ("A1-7", dp["lr2e-3_b128_natural"], 0, 0),
        ("A1-8", dp["lr5e-4_b8_natural"], 1, 1),
        ("A1-9", dp["lr2e-3_natural"], 0, 0),
        ("A2-2", dp["rm16_lr2e-3_natural"], 0, 0),
        ("A2-4", dp["rm20_lr2e-3_natural"], 5, 5),
        ("A2-5", dp["rm24_base_natural"], 6, 6),
        ("A2-7", dp["rm4_b64_natural"], 6, 4),
        ("A2-9", dp["rm4_lr2e-3_b64_natural"], 0, 0),
        ("A2-10", dp["rm8_b64_natural"], 5, 4),
        ("A2-12", dp["rm20_lr2e-3_b16_natural"], 1, 1),
        ("A2-13", dp["rm24_lr1.414e-3_b8_natural"], 4, 4),
        ("A2-14", dp["rm24_lr2e-3_b8_natural"], 0, 0),
        ("A3-1", dn["base_y0_half"], 6, 6),
        ("A3-2", dn["lr5e-4_y0_times2"], 0, 0),
        ("A3-3", dn["lr2e-3_y0_quarter"], 6, 4),
        ("A3-5", dn["ov1_double"], 6, 6),
        ("A3-6", dn["lr5e-4_ov1_quarter"], 0, 0),
        ("A4-1", dn["qk0_half"], 6, 6),
        ("A4-2", dn["qk1_double"], 6, 6),
        ("A4-4", dn["lr2e-3_qk0_quarter"], 1, 1),
        ("A5-2", dp["base_far_w1000_3000"], 0, 0),
        ("A5-3", dw["band_l1_rm20"], 0, 0),
    ]
    mismatches = []
    checked = []
    for cid, runs, exp_tsm, exp_t in expected:
        got_tsm, got_t, got_ts = count_runs(runs)
        ok = (got_tsm == exp_tsm) and (got_t == exp_t)
        row = {"cell": cid, "expected_TSM_T": [exp_tsm, exp_t], "got_TSM_T_TS": [got_tsm, got_t, got_ts], "ok": ok}
        checked.append(row)
        if not ok:
            mismatches.append(row)
    res = {"n_cells_checked": len(checked), "n_mismatches": len(mismatches), "pass": len(mismatches) == 0, "mismatches": mismatches}
    print(f"[IC-3] Checked {len(checked)} exploratory cells against §3: mismatches={len(mismatches)}, pass={res['pass']}")
    if mismatches:
        for m in mismatches:
            print("  MISMATCH:", m)
    return res


# -------------------------------------------------------------------------------------- full analysis
def _lowest_tier(verdicts: list[str]) -> str:
    order = {"missing": -1, "unstable": 0, "unconfirmed": 1, "shift": 2, "flip": 3}
    if not verdicts or any(v not in order for v in verdicts):
        return "incomplete"
    return min(verdicts, key=lambda v: order[v])


def analyze_all(work_dir: Path, jobs: int, primary_mode: str = "source_strict") -> dict[str, Any]:
    npz_files = sorted(p for p in work_dir.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    print(f"[ANALYZE] Classifying {len(npz_files)} run files in {work_dir}...", flush=True)
    if jobs > 1 and len(npz_files) > 8:
        with Pool(processes=min(jobs, len(npz_files))) as pool:
            rows = list(pool.imap_unordered(_classify_worker, [str(p) for p in npz_files]))
    else:
        rows = [classify_npz(p) for p in npz_files]
    rows.sort(key=lambda r: (r["cell_id"], r["seed"]))

    by_cell_all: dict[str, list[dict[str, Any]]] = {}
    by_cell_main: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cell_all.setdefault(r["cell_id"], []).append(r)
        if r["seed"] < 1024:
            by_cell_main.setdefault(r["cell_id"], []).append(r)
    for cid, rlist in by_cell_all.items():
        if cid not in by_cell_main:
            by_cell_main[cid] = rlist

    cell_summaries: dict[str, dict[str, Any]] = {}
    for cid in sorted(by_cell_all):
        cs = summarize_cell(by_cell_main[cid])
        if len(by_cell_all[cid]) > len(by_cell_main[cid]):
            cs["pooled_all"] = summarize_cell(by_cell_all[cid])
        cell_summaries[cid] = cs

    main_rows = [r for r in rows if r["seed"] < 1024] or rows

    # ---- Helper for paired R1 direction (C2 P2d/P2e) ----
    def r1_paired(cid_lo: str, cid_hi: str) -> dict[str, Any]:
        rx = {r["seed"]: r for r in by_cell_all.get(cid_lo, [])}
        ry = {r["seed"]: r for r in by_cell_all.get(cid_hi, [])}
        common = sorted(set(rx) & set(ry))
        if not common:
            return {"n": 0, "status": "missing"}
        n_ok = sum(rx[s]["r1_aligned"] < ry[s]["r1_aligned"] for s in common)
        ratios = [rx[s]["r1_aligned"] / ry[s]["r1_aligned"] for s in common if ry[s]["r1_aligned"] > 0]
        thresh = 20 if len(common) == 24 else int(math.ceil(len(common) * 20 / 24))
        return {
            "n": len(common),
            "pairs_lo_lt_hi": n_ok,
            "threshold": thresh,
            "pass": n_ok >= thresh,
            "median_ratio_lo_over_hi": bootstrap_median_ci(ratios),
        }

    # ---- Suite A: Claims C1–C4 ----
    c1 = {
        "P1a": paired_comparison("A1-2", "A1-6", by_cell_all, primary_mode=primary_mode),
        "P1b": paired_comparison("A1-1", "A1-7", by_cell_all, primary_mode=primary_mode),
        "P1c": paired_comparison("A2-1", "A1-6", by_cell_all, primary_mode=primary_mode),
        "P1d": paired_comparison("A2-4", "A2-2", by_cell_all, primary_mode=primary_mode),
    }
    c1["claim_verdict"] = _lowest_tier([c1[k].get("verdict", "missing") for k in ("P1a", "P1b", "P1c", "P1d")])

    # S1a, S1b, S1c, S1d (evaluated on main seeds 1000..1023)
    s1: dict[str, Any] = {}
    for cid in ("A1-8", "A1-9"):
        if cid in cell_summaries:
            cs = cell_summaries[cid]
            s1[f"S1a_{cid}"] = {"source_strict": cs["source_strict"], "source_inclusive": cs["source_inclusive"],
                                "pass_strict": cs["source_strict"] <= 8, "pass_inclusive": cs["source_inclusive"] <= 8}
    if all(cid in cell_summaries for cid in ("A1-3", "A1-4", "A1-5")):
        for mode in ("source_strict", "source_inclusive"):
            ks = [cell_summaries[cid][mode] for cid in ("A1-3", "A1-4", "A1-5")]
            ns = [cell_summaries[cid]["n"] for cid in ("A1-3", "A1-4", "A1-5")]
            p_fh = freeman_halton_3x2_p(ks[0], ns[0], ks[1], ns[1], ks[2], ns[2])
            s1[f"S1b_{mode}"] = {"counts": ks, "ns": ns, "exact_3x2_p": round(p_fh, 5),
                                 "all_over_half": all(k > n / 2 for k, n in zip(ks, ns)),
                                 "pass": p_fh >= 0.05 and all(k > n / 2 for k, n in zip(ks, ns))}
    if "A2-6" in cell_summaries:
        cs = cell_summaries["A2-6"]
        s1["S1c"] = {"source_strict": cs["source_strict"], "source_inclusive": cs["source_inclusive"],
                     "pass_strict": cs["source_strict"] >= 16, "pass_inclusive": cs["source_inclusive"] >= 16}
    if "A2-8" in by_cell_main and "A1-6" in by_cell_main:
        cmp_s1d = paired_comparison("A2-8", "A1-6", by_cell_main, primary_mode=primary_mode)
        if cmp_s1d.get("n", 0) > 0:
            s1["S1d"] = {**cmp_s1d, "pass_strict": cmp_s1d["by_mode"]["source_strict"]["diff"] >= 6,
                         "pass_inclusive": cmp_s1d["by_mode"]["source_inclusive"]["diff"] >= 6}
        else:
            s1["S1d"] = cmp_s1d
    c1["secondary"] = s1

    c2 = {
        "P2a": paired_comparison("A3-1", "A1-6", by_cell_all, primary_mode=primary_mode),
        "P2b": paired_comparison("A1-1", "A3-2", by_cell_all, primary_mode=primary_mode),
        "P2c": paired_comparison("A3-3", "A1-9", by_cell_all, primary_mode=primary_mode),
        "P2d": {**paired_comparison("A3-5", "A1-6", by_cell_all, primary_mode=primary_mode), "r1_check": r1_paired("A3-5", "A1-6")},
        "P2e": {**paired_comparison("A1-1", "A3-6", by_cell_all, primary_mode=primary_mode), "r1_check": r1_paired("A1-1", "A3-6")},
        "S2a_A3-4": cell_summaries.get("A3-4"),
        "A3-7_exploratory": cell_summaries.get("A3-7"),
    }
    c2["claim_verdict"] = _lowest_tier([c2[k].get("verdict", "missing") for k in ("P2a", "P2b", "P2c", "P2d", "P2e")])
    # Descriptive R1 AUC across A1 + A3 (main seeds)
    a1_a3 = [r for r in main_rows if r["group"] in ("A1", "A3") and r["cat"] != "U"]
    c2["descriptive_r1_auc"] = {
        "n_formed": len(a1_a3),
        "aligned_step_vs_relay": roc_auc([r["r1_aligned"] for r in a1_a3], [r["cat"] == "R" for r in a1_a3]),
        "n9600_vs_relay": roc_auc([r["r1_at_n9600"] for r in a1_a3], [r["cat"] == "R" for r in a1_a3]),
    }

    c3 = {
        "P3a": paired_comparison("A4-1", "A1-6", by_cell_all, primary_mode=primary_mode),
        "P3b": paired_comparison("A4-2", "A1-6", by_cell_all, primary_mode=primary_mode),
        "P3c": paired_comparison("A1-1", "A4-3", by_cell_all, primary_mode=primary_mode),
        "P3d": paired_comparison("A5-1", "A5-2", by_cell_all, primary_mode=primary_mode),
        "S3a_A4-4": cell_summaries.get("A4-4"),
        "S3b": paired_comparison("A2-3", "A5-3", by_cell_main, primary_mode=primary_mode),
    }
    c3["claim_verdict"] = _lowest_tier([c3[k].get("verdict", "missing") for k in ("P3a", "P3b", "P3c", "P3d")])
    # S3c: first layer past 0.85 across A1 + A3 + A4 (480 main runs)
    s3c_pool = [r for r in main_rows if r["group"] in ("A1", "A3", "A4")]
    decidable = [r for r in s3c_pool if r["l1_first_85"] is not None]
    s3c_res = {"n_total": len(s3c_pool), "n_decidable": len(decidable),
               "decidable_rate": round(len(decidable) / len(s3c_pool), 4) if s3c_pool else None}
    for mode in ("source_strict", "source_inclusive"):
        acc = sum(r["l1_first_85"] == metric_flag(r, mode) for r in decidable)
        rate = round(acc / len(decidable), 4) if decidable else None
        s3c_res[f"agree_{mode}"] = acc
        s3c_res[f"rate_{mode}"] = rate
        s3c_res[f"pass_{mode}"] = bool(s3c_pool and (len(decidable) / len(s3c_pool) >= 0.90) and rate is not None and rate >= 0.85)
    c3["S3c_band85"] = s3c_res

    c4 = {
        "P4a": paired_comparison("A6-3", "A6-1", by_cell_main, primary_mode="textbook", diff_only=True),
        "P4b": paired_comparison("A6-4", "A6-1", by_cell_main, primary_mode="textbook", diff_only=True),
        "S4a_A6-5": cell_summaries.get("A6-5"),
        "decomp_exploratory": {cid: cell_summaries.get(cid) for cid in ("A6-6", "A6-7", "A6-8")},
    }
    tb_pool = [r for r in main_rows if r["group"] in ("A1", "A3", "A4") and r["cat"] == "T"]
    ordered = [r for r in tb_pool if r["source_step"] is not None and r["second_key_step"] is not None and r["second_key_step"] > r["source_step"]]
    ratios_c4 = [r["second_key_step"] / r["source_step"] for r in ordered if r["source_step"] > 0]
    med_c4 = bootstrap_median_ci(ratios_c4)
    c4["P4c"] = {
        "n_textbook": len(tb_pool),
        "n_ordered": len(ordered),
        "ordered_rate": round(len(ordered) / len(tb_pool), 4) if tb_pool else None,
        "ratio_median": med_c4,
        "pass": bool(tb_pool and (len(ordered) / len(tb_pool) >= 0.90) and med_c4["median"] is not None and 2.0 <= med_c4["median"] <= 4.5),
    }
    c4_parts = [c4["P4a"].get("verdict") == "pass", c4["P4b"].get("verdict") == "pass", c4["P4c"]["pass"]]
    if c4["P4a"].get("n", 0) == 0 or c4["P4b"].get("n", 0) == 0:
        c4["claim_verdict"] = "incomplete"
    elif all(c4_parts):
        c4["claim_verdict"] = "pass_all"
    elif any(c4_parts):
        c4["claim_verdict"] = "partial"
    else:
        c4["claim_verdict"] = "fail_all"

    # ---- Suite B: Generality ----
    suite_b = {}
    for grp in ("B1-W", "B1-U", "B1-WU", "B2"):
        mid_cs = cell_summaries.get(f"{grp}-mid")
        low_cs = cell_summaries.get(f"{grp}-low")
        mid_shift = None
        if mid_cs is not None and mid_cs["n"] > 0:
            v = mid_cs[primary_mode]
            hi_t = math.ceil(mid_cs["n"] * 2 / 3)
            lo_t = math.floor(mid_cs["n"] / 3)
            mid_shift = "threshold_unchanged" if v <= lo_t else ("threshold_shifted_up" if v >= hi_t else "on_threshold")
        entry: dict[str, Any] = {
            "low_vs_high": paired_comparison(f"{grp}-low", f"{grp}-high", by_cell_main, primary_mode=primary_mode),
            "mid": mid_cs,
            "mid_threshold_reading": mid_shift,
        }
        if grp in ("B1-WU", "B2"):
            lc05 = paired_comparison(f"{grp}-mid-c05", f"{grp}-mid", by_cell_main, primary_mode=primary_mode, diff_only=True)
            if mid_cs is not None and mid_cs["n"] > 0 and mid_cs[primary_mode] >= math.ceil(mid_cs["n"] * 2 / 3):
                lc05["ref_already_at_target"] = True
                lc05["prereg_note"] = "na_ref_at_target"
            else:
                lc05["ref_already_at_target"] = False
            lc2 = paired_comparison(f"{grp}-low", f"{grp}-low-c2", by_cell_main, primary_mode=primary_mode, diff_only=True)
            if low_cs is not None and low_cs["n"] > 0 and low_cs[primary_mode] <= math.floor(low_cs["n"] / 3):
                lc2["ref_already_at_target"] = True
                lc2["prereg_note"] = "na_ref_at_target"
            else:
                lc2["ref_already_at_target"] = False
            entry["lever_c05"] = lc05
            entry["lever_c2"] = lc2
        suite_b[grp] = entry
    suite_b["B3"] = {
        "early_vs_late": paired_comparison("B3-early", "B3-late", by_cell_main, primary_mode=primary_mode),
        "early": cell_summaries.get("B3-early"),
        "late": cell_summaries.get("B3-late"),
    }

    # ---- Suite C: Width bridge & GO / NO-GO / GRAY gate on d = 256 ----
    suite_c: dict[str, Any] = {
        "d32": {t: cell_summaries.get(f"C-d32-{t}") for t in ("low", "mid", "high")},
        "d64_reused_from_A": {
            "low": cell_summaries.get("A1-2"), "mid": cell_summaries.get("A1-6"),
            "high": cell_summaries.get("A1-9"), "mid_c05": cell_summaries.get("A3-1"), "high_c05": cell_summaries.get("A3-4"),
        },
        "d128": {t: cell_summaries.get(f"C-d128-{t}") for t in ("low", "mid", "high", "mid-c05", "high-c05")},
        "d256": {t: cell_summaries.get(f"C-d256-{t}") for t in ("low", "mid", "high", "mid-c05", "high-c05")},
    }
    d256_base = [cell_summaries[f"C-d256-{t}"] for t in ("low", "mid", "high") if f"C-d256-{t}" in cell_summaries]
    stable_256 = [cs for cs in d256_base if not cs["unstable"]]
    if len(d256_base) == 3 and stable_256:
        max_r = max(cs["counts"]["R"] for cs in stable_256)
        max_m = max(cs["counts"]["M"] for cs in stable_256)
        mid_ok = (
            "C-d256-mid" in cell_summaries and "C-d256-mid-c05" in cell_summaries
            and not cell_summaries["C-d256-mid"]["unstable"]
            and not cell_summaries["C-d256-mid-c05"]["unstable"]
        )
        high_ok = (
            "C-d256-high" in cell_summaries and "C-d256-high-c05" in cell_summaries
            and not cell_summaries["C-d256-high"]["unstable"]
            and not cell_summaries["C-d256-high-c05"]["unstable"]
        )
        raw_red_mid = (
            cell_summaries["C-d256-mid"]["counts"]["R"] - cell_summaries["C-d256-mid-c05"]["counts"]["R"]
            if ("C-d256-mid" in cell_summaries and "C-d256-mid-c05" in cell_summaries) else 0
        )
        raw_red_high = (
            cell_summaries["C-d256-high"]["counts"]["R"] - cell_summaries["C-d256-high-c05"]["counts"]["R"]
            if ("C-d256-high" in cell_summaries and "C-d256-high-c05" in cell_summaries) else 0
        )
        lever_red_mid = raw_red_mid if mid_ok else 0
        lever_red_high = raw_red_high if high_ok else 0
        if max_r >= 12 and max(lever_red_mid, lever_red_high) >= 8:
            gate = "GO"
        elif max_r <= 4 and max_m <= 4:
            gate = "NO-GO"
        else:
            gate = "GRAY"
        suite_c["gate_d256"] = {
            "decision": gate,
            "max_R_in_stable_cells": max_r,
            "max_M_in_stable_cells": max_m,
            "lever_R_reduction_mid": lever_red_mid,
            "lever_R_reduction_high": lever_red_high,
            "raw_R_reduction_mid": raw_red_mid,
            "raw_R_reduction_high": raw_red_high,
        }
    else:
        suite_c["gate_d256"] = {"decision": "incomplete"}

    # ---- §2.5 Automatic Top-up Check ----
    primary_pairs = [
        ("P1a", c1["P1a"]), ("P1b", c1["P1b"]), ("P1c", c1["P1c"]), ("P1d", c1["P1d"]),
        ("P2a", c2["P2a"]), ("P2b", c2["P2b"]), ("P2c", c2["P2c"]), ("P2d", c2["P2d"]), ("P2e", c2["P2e"]),
        ("P3a", c3["P3a"]), ("P3b", c3["P3b"]), ("P3c", c3["P3c"]), ("P3d", c3["P3d"]),
    ]
    topup_cells = sorted({
        cid
        for _, cmp_d in primary_pairs
        if cmp_d.get("n") == 24 and cmp_d.get("verdict") not in ("flip", "unstable")
        for cid in (cmp_d["x"], cmp_d["y"])
    })

    return {
        "n_runs_analyzed": len(rows),
        "n_cells_present": len(cell_summaries),
        "primary_mode": primary_mode,
        "claim_verdicts": {
            "C1": c1["claim_verdict"],
            "C2": c2["claim_verdict"],
            "C3": c3["claim_verdict"],
            "C4": c4["claim_verdict"],
        },
        "cells": cell_summaries,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "Suite_B": suite_b,
        "Suite_C": suite_c,
        "topup_needed_cells": topup_cells,
        "runs": rows,
    }


def write_markdown_summary(res: dict[str, Any], out_md: Path) -> None:
    cv = res.get("claim_verdicts", {})
    lines = [
        "# Relay Confirmation Study — Frozen Analysis Summary",
        "",
        f"- Runs analyzed: **{res['n_runs_analyzed']}** across **{res['n_cells_present']}** cells.",
        f"- Primary source definition: **`{res['primary_mode']}`** (`T + S`), with `T + S + M` and `T` reported side-by-side.",
        f"- Claim rollups: **C1=`{cv.get('C1')}`**, **C2=`{cv.get('C2')}`**, **C3=`{cv.get('C3')}`**, **C4=`{cv.get('C4')}`**.",
        "",
        "## Primary Comparisons (Suite A)",
        "",
        "| ID | X vs Y | n | T+S (kx vs ky) | Verdict (T+S) | T+S+M (kx vs ky) | Verdict (T+S+M) | T only | McNemar p (T+S) |",
        "|---|---|---:|---|---|---|---:|---|---:|",
    ]
    pairs = [
        ("P1a", res["C1"]["P1a"]), ("P1b", res["C1"]["P1b"]), ("P1c", res["C1"]["P1c"]), ("P1d", res["C1"]["P1d"]),
        ("P2a", res["C2"]["P2a"]), ("P2b", res["C2"]["P2b"]), ("P2c", res["C2"]["P2c"]), ("P2d", res["C2"]["P2d"]), ("P2e", res["C2"]["P2e"]),
        ("P3a", res["C3"]["P3a"]), ("P3b", res["C3"]["P3b"]), ("P3c", res["C3"]["P3c"]), ("P3d", res["C3"]["P3d"]),
        ("P4a", res["C4"]["P4a"]), ("P4b", res["C4"]["P4b"]),
    ]
    for pid, d in pairs:
        if d.get("n", 0) == 0:
            lines.append(f"| {pid} | {d['x']} vs {d['y']} | 0 | missing | — | — | — | — | — |")
            continue
        ms = d["by_mode"]["source_strict"]
        mi = d["by_mode"]["source_inclusive"]
        mt = d["by_mode"]["textbook"]
        flag = " ⚠" if d.get("definition_sensitive") else ""
        lines.append(
            f"| {pid} | {d['x']} vs {d['y']} | {d['n']} | {ms['kx']} vs {ms['ky']} | **{ms['verdict']}**{flag} | "
            f"{mi['kx']} vs {mi['ky']} | {mi['verdict']} | {mt['kx']} vs {mt['ky']} ({mt['verdict']}) | {ms['mcnemar_p']} |"
        )

    lines.extend(["", "## Suite B (Generality)", ""])
    for grp in ("B1-W", "B1-U", "B1-WU", "B2"):
        d = res["Suite_B"][grp]["low_vs_high"]
        if d.get("n", 0) == 0:
            lines.append(f"- **{grp}**: not yet run")
        else:
            ms = d["by_mode"]["source_strict"]
            mi = d["by_mode"]["source_inclusive"]
            lines.append(f"- **{grp}** (`low` vs `high`): T+S `{ms['kx']} vs {ms['ky']}` (**{ms['verdict']}**), T+S+M `{mi['kx']} vs {mi['ky']}` ({mi['verdict']})")
    b3 = res["Suite_B"]["B3"]["early_vs_late"]
    if b3.get("n", 0):
        ms = b3["by_mode"]["source_strict"]
        lines.append(f"- **B3** (`early` vs `late` data window): T+S `{ms['kx']} vs {ms['ky']}` (**{ms['verdict']}**)")

    lines.extend([
        "",
        "## Suite C (Width Bridge & d = 256 Gate)",
        "",
        f"- Gate decision: **`{res['Suite_C']['gate_d256']['decision']}`** (`{json.dumps(res['Suite_C']['gate_d256'])}`)",
        "",
        "## §2.5 Top-up Needed (Primary Comparisons Not Reaching Flip at n = 24)",
        "",
    ])
    if res["topup_needed_cells"]:
        cmd = "PYTHONPATH=. python scripts/analysis_v2/relay_confirm.py --topup " + " ".join(res["topup_needed_cells"]) + " --jobs 86"
        lines.append(f"```bash\n{cmd}\n```")
    else:
        lines.append("None (or no n=24 primary comparisons present yet).")

    lines.extend([
        "",
        "## All Present Cells",
        "",
        "| Cell | n | U | R | T | S | M | T+S | T+S+M | R_1 median | Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for cid, cs in res["cells"].items():
        c = cs["counts"]
        flags = []
        if cs["unstable"]:
            flags.append("UNSTABLE")
        if cs["head_gap_ge_6"]:
            flags.append("HEAD_GAP>=6")
        r1m = cs["r1_aligned"]["median"]
        lines.append(
            f"| {cid} | {cs['n']} | {c['U']} | {c['R']} | {c['T']} | {c['S']} | {c['M']} | "
            f"{cs['source_strict']} | {cs['source_inclusive']} | {r1m} | {', '.join(flags)} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Frozen analysis for the 88-core relay confirmation study.")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_confirm")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--primary-mode", choices=("source_strict", "source_inclusive"), default="source_strict")
    ap.add_argument("--out-json", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_confirm_results.json")
    ap.add_argument("--out-md", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_confirm_summary.md")
    ap.add_argument("--ic3", action="store_true", help="Run IC-3 verification against exploratory pilot JSONs and exit")
    args = ap.parse_args(argv)

    if args.ic3:
        res = run_ic3_check()
        return 0 if res["pass"] else 2

    res = analyze_all(args.work, max(1, args.jobs), primary_mode=args.primary_mode)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    write_markdown_summary(res, args.out_md)
    print(f"[ANALYZE] Wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
