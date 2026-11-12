"""Sky v1, pre-registered instrument check 3 (Markov language, one planted coupling), incl. Amendment 1 item B,
and the chain-induced (transient) non-planted interactions as a result.

Pre-registration: docs/results/v1_preregistration.md (criteria fixed before the run; Amendment 1 written after
launch, before v1 data were read).

Runs: region markov-v1, the ten coupling sets with ONE planted coupling ("1:1.0", "1,3:1.0", "1,2,4:1.0",
"1,2,3,5:1.0", "1:0.5", "1,3:0.5", "1,2,4:0.5", "1,2,3,5:0.5", "1,3:0.8", "1,2,3:1.0"), both train contexts, every
seed of the current sky (boundary refinement added seeds).

Check 3 (literal, registered)
  (a) the planted subset {x[t-l] : l in L} + {xhat} activates in >= 90 % of runs whose final kl_true < 0.01
      ("final" = last value of the light curve's x_kl_true). "Activates" = the curve has an activation event of the
      planted subset (ri_events, kind "activation"; cross-checked with detect_events(curve, alpha=0.01)).
      Alternatives reported: (a2) a (re)activation that is not left-censored and at whose checkpoint the planted
      Fourier coefficient has the sign of its final value (Amendment 1 C's onset rule); (a3) planted subset active
      at the end of training. Also: how many runs reach kl_true < 0.01 at any light-curve point.
  (b) a non-planted subset (any stored subset other than the planted one) with peak RI >= 10 x min_effect occurs in
      <= 5 % of runs. Peak RI = max over all checkpoints of the stored bias-corrected RI (curve.ri). Denominator: all
      one-coupling runs (alternative: runs with final kl_true < 0.01; identical here). Alternative peak: the
      peak_ri of a non-planted activation/reactivation event (needs q <= alpha as well).
Check 3 (amended, Amendment 1 B, decides for the non-planted part)
  (c) at the end of training, the largest non-planted RI over the last five checkpoints is <= 10 x min_effect in
      >= 95 % of runs, in both contexts. Largest = max over non-planted subsets and over the last five checkpoints
      (alternatives: max over subsets of the mean over the last five; the final checkpoint only).
All three are reported separately for chain and iid training contexts.

Result (not a check): transient non-planted interactions in chain-trained runs (non-planted subsets whose peak RI
reaches 10 x min_effect): subsets, runs, peak RI and its step, rise (first activation event) and decay (first
extinction event after it) relative to the planted onset, and the iid twin (same coupling set and seed) on the same
subset. Explanation from the chain: from the actual training corpus (MarkovLanguage, same config, data_seed 0) the
correlation rho_A = E[s_t prod_{l in A} s_{t-l}], the analytic value tanh(beta)^|J| (s_t chi_L(t) = e_t are i.i.d.
with mean tanh(beta); s_t chi_A = prod_{j in J} e_{t-j} when such a finite J exists, else 0), the Bayes-optimal
marginal predictor g_A(s_A) = E_chain[s_t | s_A] with its Walsh coefficient and population local RI under uniform
probes (the RI a model would show if it predicted from the lags of A alone), and the share of the planted term's
gradient signal E[(s_t - g_A) chi_L] / E[s_t chi_L] left once the shortcut g_A is fitted. Delay: planted onset chain
vs iid per coupling set under (i) the registered onset (first activation event), (ii) Amendment 1 C's onset, (iii)
the planted Fourier coefficient (planted sign) reaching 25 % of its population value tanh(beta) (Amendment 1 E's
form); medians over seeds, ratio chain / iid with a bootstrap over seeds within context (2,000 resamples, fixed
seed). Runs without an onset are censored at the step budget (30,000) and counted.

Conventions: code 1 = token 1 = spin +1; fourier = Walsh coefficient of 2 P(xhat = 1) - 1, so a learned coupling
with beta > 0 has a positive coefficient tanh(beta) on its lags. An event's step is the FIRST checkpoint of the two
consecutive checkpoints that confirm it (events.py), step_lo the checkpoint before.

Usage (cwd = repository root):
    PYTHONPATH=. python scripts/analysis_v1/markov_instrument.py [--release releases/DR1] [--extracted DIR] [--out JSON]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import tycho
from tycho.analysis.events import detect_events
from tycho.analysis.ricurve import RICurve
from tycho.ri.fourier import single_mode_ri
from tycho.ri.refined import local_ri_from_table
from tycho.tasks.markov_lang import MarkovLanguage, parse_couplings

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACTED = REPO / "work" / "dr1"  # extracted DR1 archives (git-ignored); see README.md
REGION = "markov-v1"
SINGLE = ["1:1.0", "1,3:1.0", "1,2,4:1.0", "1,2,3,5:1.0", "1:0.5", "1,3:0.5", "1,2,4:0.5", "1,2,3,5:0.5", "1,3:0.8",
          "1,2,3:1.0"]
CONTEXTS = ("chain", "iid")
KL_BAR = 0.01
A_BAR, B_BAR, C_BAR = 0.90, 0.05, 0.95
NONPLANTED_FACTOR = 10.0
LAST_K = 5
BOOT_N = 2000
BOOT_SEED = 20261112
E_FRAC = 0.25  # Amendment 1 E: planted coefficient reaches 25 % of its population value


# ---------------------------------------------------------------------------------------------- helpers
def fnum(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return fnum(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def frac(k, n) -> dict:
    return {"num": int(k), "den": int(n), "frac": (k / n) if n else None}


def first_pair(mask: np.ndarray, start: int = 0) -> int | None:
    m = mask[:-1] & mask[1:]
    idx = np.flatnonzero(m[start:])
    return int(idx[0] + start) if idx.size else None


def first_true(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def lags_of(names) -> tuple[int, ...]:
    return tuple(sorted(int(n[len("x[t-"):-1]) for n in names if n != "xhat"))


def names_of(lags) -> tuple[str, ...]:
    return tuple(f"x[t-{l}]" for l in sorted(lags)) + ("xhat",)


def sname(names) -> str:
    return "{" + ", ".join(names) + "}"


def boot_fraction(flags_by_group: list[np.ndarray], unit: str, seed: int = BOOT_SEED) -> list[float]:
    """95 % percentile CI of the pooled fraction. unit "seed": resample runs within each coupling set (stratified);
    unit "config": resample coupling sets with replacement."""
    rng = np.random.default_rng(seed)
    reps = np.empty(BOOT_N)
    for b in range(BOOT_N):
        if unit == "config":
            pick = rng.integers(0, len(flags_by_group), size=len(flags_by_group))
            vals = np.concatenate([flags_by_group[i] for i in pick])
        else:
            vals = np.concatenate([g[rng.integers(0, g.size, size=g.size)] for g in flags_by_group])
        reps[b] = vals.mean()
    return [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]


def boot_ratio_of_medians(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    """median(a) / median(b) with a 95 % percentile CI; a and b resampled independently (seeds within context)."""
    rng = np.random.default_rng(seed)
    reps = np.empty(BOOT_N)
    for i in range(BOOT_N):
        ra = np.median(a[rng.integers(0, a.size, a.size)])
        rb = np.median(b[rng.integers(0, b.size, b.size)])
        with np.errstate(divide="ignore", invalid="ignore"):
            reps[i] = ra / rb if rb > 0 else (np.inf if ra > 0 else np.nan)
    reps = reps[~np.isnan(reps)]
    return {"ratio": float(np.median(a) / np.median(b)),
            "ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
            "n_boot": BOOT_N, "seed": seed, "unit": "seeds within context (chain and iid resampled independently)",
            "n_resamples_with_zero_iid_median": int(BOOT_N - reps.size + np.isinf(reps).sum())}


# ---------------------------------------------------------------------------------------------- chain statistics
def analytic_shift_representation(L: tuple[int, ...], A: tuple[int, ...], jmax: int = 400) -> list[int] | None:
    """Shifts J with s_t chi_A = prod_{j in J} e_{t-j}, e_u = s_u prod_{l in L} s_{u-l} (i.i.d., mean tanh beta).

    Greedy GF(2) elimination: pattern P_j = {j} u (j + L) has minimum j, so the residual's minimum fixes the next
    shift. Returns None when no finite representation exists (then the stationary correlation is 0).
    """
    R = {0} | set(A)
    J: list[int] = []
    while R:
        m = min(R)
        if m > jmax:
            return None
        J.append(m)
        R ^= {m} | {m + l for l in L}
    return J


def chain_statistics(cfg: dict, L: tuple[int, ...], beta: float, subsets: list[tuple[int, ...]]) -> dict:
    """Correlations and Bayes-optimal marginal predictors from the actual training corpus (chain) and from the iid
    training examples of the same coupling set (data_seed 0)."""
    out = {}
    for ctx in CONTEXTS:
        task = MarkovLanguage({**cfg, "train_context": ctx}, seed=0)
        corpus = task._corpus
        lag = corpus["lag"].astype(np.int64)  # (n, 6) lag order, column l-1 = token at lag l
        s = 2.0 * lag - 1.0
        y = 2.0 * corpus["targets"].numpy().astype(np.float64) - 1.0
        chiL = np.prod(s[:, [l - 1 for l in L]], axis=1)
        rho_L = float(np.mean(y * chiL))
        rows = {}
        for A in subsets:
            chiA = np.prod(s[:, [l - 1 for l in A]], axis=1)
            rho = float(np.mean(y * chiA))
            # Bayes-optimal marginal predictor g_A(s_A) = E[s_t | s_A] (empirical, 2^|A| cells)
            code = (lag[:, [l - 1 for l in A]] << np.arange(len(A))[None, :]).sum(axis=1)
            ncell = 1 << len(A)
            cnt = np.bincount(code, minlength=ncell).astype(float)
            sy = np.bincount(code, weights=y, minlength=ncell)
            g = np.where(cnt > 0, sy / np.maximum(cnt, 1), 0.0)
            cell_bits = (np.arange(ncell)[:, None] >> np.arange(len(A))[None, :]) & 1
            cell_spins = 2.0 * cell_bits - 1.0
            chi_cell = np.prod(cell_spins, axis=1)
            f_unif = float(np.mean(g * chi_cell))  # top Walsh coefficient of g under uniform probes
            # population local RI of A u {xhat} under uniform probes with P(xhat = 1 | s_A) = (1 + g) / 2
            P = np.zeros((2,) * (len(A) + 1))
            for c_ in range(ncell):
                idx = tuple(int(b) for b in cell_bits[c_])
                P[idx + (1,)] = (1.0 + g[c_]) / 2.0 / ncell
                P[idx + (0,)] = (1.0 - g[c_]) / 2.0 / ncell
            ri_g = float(local_ri_from_table(P, [tuple(range(len(A) + 1))])[0])
            # residual planted-term signal once g_A is fitted (squared-loss gradient on chi_L)
            resid = float(np.mean((y - g[code]) * chiL)) / rho_L if rho_L != 0 else float("nan")
            # lower-order Walsh content of g under uniform probes (largest |coef| over proper subsets)
            lower = 0.0
            for r in range(0, len(A)):
                for B in itertools.combinations(range(len(A)), r):
                    chiB = np.prod(cell_spins[:, list(B)], axis=1) if B else np.ones(ncell)
                    lower = max(lower, abs(float(np.mean(g * chiB))))
            chain_pA_max_dev = float(np.max(np.abs(cnt / cnt.sum() - 1.0 / ncell)) * ncell)
            J = analytic_shift_representation(L, A)
            rows[A] = {"rho": rho, "rho_analytic": (math.tanh(beta) ** len(J)) if J is not None else 0.0,
                       "shifts_J": J, "g_walsh_top_uniform": f_unif, "g_walsh_lower_max_abs": lower,
                       "ri_bayes_marginal_uniform_probe": ri_g, "single_mode_ri_of_rho": float(single_mode_ri(rho)),
                       "planted_signal_left_after_g": resid, "p_sA_max_rel_dev_from_uniform": chain_pA_max_dev,
                       "n_examples": int(y.size)}
        out[ctx] = {"rho_planted": rho_L, "rho_planted_analytic": math.tanh(beta), "subsets": rows,
                    "n_examples": int(y.size)}
    return out


# ---------------------------------------------------------------------------------------------- per run
def analyse_run(obj: pd.Series, run: pd.Series, evs: pd.DataFrame, extracted: Path) -> tuple[dict, RICurve]:
    c = RICurve.load(extracted / run.ricurve_file)
    cfg = json.loads(obj.task_cfg_json)
    coup = parse_couplings(cfg["couplings"], int(cfg["window"]))
    assert len(coup) == 1
    (L, beta), = coup.items()
    planted = names_of(L)
    meta = c.meta
    me, alpha = float(meta["min_effect"]), float(meta["alpha"])
    thr = NONPLANTED_FACTOR * me
    steps = c.steps
    T = c.T
    budget = int(obj.train_steps)
    # ground truth found
    gt = c.ground_truth or []
    gt_ok = len(gt) == 1 and tuple(sorted(gt[0]["subset"])) == tuple(sorted(planted))
    gi = c.index_of(list(planted))
    if gi is None:
        raise RuntimeError(f"{obj.object_id}: planted subset {planted} not stored")
    nonp = [i for i in range(c.S) if i != gi]
    all_stored = c.S == int(meta["n_candidates"])
    ri_g, f_g, q_g = c.ri[gi], c.fourier[gi], c.q[gi]
    f_final = float(f_g[-1])
    sgn_final = float(np.sign(f_final))

    # ---------------- light curve
    lc = np.load(extracted / obj.lightcurve_file)
    kl, lstep = lc["x_kl_true"], lc["step"]
    kl_final = float(kl[-1])
    t_kl = first_true(kl < KL_BAR)

    # ---------------- events: table and recomputation
    evs = evs.copy()
    evs["names"] = evs.subset_json.apply(lambda s: tuple(json.loads(s)))
    tab = {(frozenset(n), k, int(s), int(lo)) for n, k, s, lo in zip(evs.names, evs.kind, evs.step, evs.step_lo)}
    rec_events = detect_events(c, alpha=0.01)
    rec = {(frozenset(e.subset_names), e.kind, e.step, e.step_lo) for e in rec_events}
    events_match = tab == rec
    gt_ev = [e for e in rec_events if frozenset(e.subset_names) == frozenset(planted)]
    gt_tab = evs[evs.names.apply(frozenset) == frozenset(planted)].sort_values("step")
    assert (gt_tab.matches_gt.fillna(False)).all()
    acts = [e for e in gt_ev if e.kind == "activation"]
    lit_act = len(acts) > 0
    lit_step = acts[0].step if acts else None
    lit_left = bool(acts and acts[0].left_censored)
    # Amendment C onset: first (re)activation not left-censored with sign(f[t0]) == sign(final f)
    tC, kindC = None, None
    for e in gt_ev:
        if e.kind == "extinction" or e.left_censored:
            continue
        t0 = int(np.searchsorted(steps, e.step))
        if np.sign(f_g[t0]) == sgn_final and sgn_final != 0:
            tC, kindC = t0, e.kind
            break
    final_active = bool(gt_ev) and gt_ev[-1].kind != "extinction"
    # Amendment E form: planted coefficient (planted sign = sign(beta)) reaches 25 % of tanh(beta)
    f_pop = math.tanh(beta)
    with np.errstate(invalid="ignore"):
        tE = first_true(np.sign(beta) * f_g >= E_FRAC * abs(f_pop))
        tE10 = first_true(np.sign(beta) * f_g >= 0.10 * abs(f_pop))
        tE50 = first_true(np.sign(beta) * f_g >= 0.50 * abs(f_pop))

    def st(t):
        return budget if t is None else int(steps[t])

    # ---------------- non-planted peaks
    ri_np = c.ri[nonp]
    peak_np = np.nanmax(ri_np, axis=1)
    j = int(np.nanargmax(peak_np))
    peak_np_max = float(peak_np[j])
    peak_np_sub = c.subset_names(nonp[j])
    peak_np_t = int(np.nanargmax(ri_np[j]))
    n_np_over = int((peak_np >= thr).sum())
    # event-based alternative
    np_ev = [e for e in rec_events if frozenset(e.subset_names) != frozenset(planted) and e.kind != "extinction"]
    ev_peak = max([e.peak_ri for e in np_ev], default=0.0)
    # amended: last five checkpoints
    last = ri_np[:, -LAST_K:]
    l5_max_per = np.nanmax(last, axis=1)
    jl = int(np.nanargmax(l5_max_per))
    l5_max = float(l5_max_per[jl])
    l5_mean_max = float(np.nanmax(np.nanmean(last, axis=1)))
    final_max = float(np.nanmax(ri_np[:, -1]))

    out = {
        "object_id": obj.object_id, "coupling": cfg["couplings"], "lags": list(L), "beta": beta,
        "planted_order": len(L) + 1, "context": cfg["train_context"], "seed": int(obj.seed), "source": obj.source,
        "ri_run_id": int(run.ri_run_id), "T": T, "budget": budget, "last_checkpoint": int(steps[-1]),
        "min_effect": me, "alpha": alpha, "N": int(meta["min_effect_n"]), "n_candidates": int(meta["n_candidates"]),
        "max_order": int(meta["max_order"]), "n_stored": c.S, "all_candidates_stored": bool(all_stored),
        "planted": list(planted), "gt_header_ok": bool(gt_ok), "gt_found": True,
        # light curve
        "kl_final": kl_final, "kl_final_finite": bool(np.isfinite(kl_final)), "kl_min": float(np.nanmin(kl)),
        "kl_final_ok": bool(kl_final < KL_BAR), "kl_ever_ok": t_kl is not None,
        "kl_first_step_below": None if t_kl is None else int(lstep[t_kl]), "lc_last_step": int(lstep[-1]),
        "final_test_acc": fnum(obj.final_test_acc), "label": obj.label,
        # planted
        "gt_activation": lit_act, "gt_activation_step": lit_step, "gt_activation_left_censored": lit_left,
        "gt_event_kinds": [e.kind for e in gt_ev], "gt_event_steps": [e.step for e in gt_ev],
        "gt_onset_C": tC is not None, "gt_onset_C_step": st(tC), "gt_onset_C_idx": tC, "gt_onset_C_kind": kindC,
        "gt_onset_C_censored": tC is None, "gt_final_active": final_active,
        "gt_onset_lit_step_or_budget": budget if lit_step is None else lit_step,
        "gt_onset_E25_step": st(tE), "gt_onset_E25_censored": tE is None,
        "gt_onset_E10_step": st(tE10), "gt_onset_E50_step": st(tE50),
        "gt_f_pop": f_pop, "gt_f_final": f_final, "gt_f_at0": fnum(f_g[0]), "gt_ri_final": float(ri_g[-1]),
        "gt_ri_peak": float(np.nanmax(ri_g)), "gt_ri_at0": float(ri_g[0]),
        # non-planted
        "np_peak_max": peak_np_max, "np_peak_subset": list(peak_np_sub), "np_peak_step": int(steps[peak_np_t]),
        "np_peak_idx": peak_np_t, "np_n_subsets_over": n_np_over, "np_over": bool(peak_np_max >= thr),
        "np_event_peak_max": float(ev_peak), "np_event_over": bool(ev_peak >= thr),
        "np_peak_max_excl_t0": float(np.nanmax(ri_np[:, 1:])),
        "np_last5_max": l5_max, "np_last5_subset": list(c.subset_names(nonp[jl])), "np_last5_ok": bool(l5_max <= thr),
        "np_last5_meanmax": l5_mean_max, "np_last5_mean_ok": bool(l5_mean_max <= thr),
        "np_final_max": final_max, "np_final_ok": bool(final_max <= thr),
        "events_table_matches_recompute": bool(events_match), "n_events_table": len(tab), "n_events_recomputed": len(rec),
        "init_ri_max": float(np.nanmax(c.ri[:, 0])), "init_ri_argmax": list(c.subset_names(int(np.nanargmax(c.ri[:, 0])))),
    }
    return out, c


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases" / "DR1")
    ap.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v1_analysis" / "markov_instrument.json")
    args = ap.parse_args()
    assert str(Path(tycho.__file__).resolve()).startswith(str(REPO)), tycho.__file__

    objects = pd.read_parquet(args.release / "objects.parquet")
    ri_runs = pd.read_parquet(args.release / "ri_runs.parquet")
    events = pd.read_parquet(args.release / "ri_events.parquet")
    mk = objects[objects.region == REGION].copy()
    mk["cfg"] = mk.task_cfg_json.apply(json.loads)
    mk["coupling"] = mk.cfg.apply(lambda c: c["couplings"])
    mk["context"] = mk.cfg.apply(lambda c: c["train_context"])
    n_region = len(mk)
    sel = mk[mk.coupling.isin(SINGLE)].sort_values(["coupling", "context", "seed"])
    assert set(sel.coupling) == set(SINGLE)

    rows, curves = [], {}
    for _, obj in sel.iterrows():
        runs = ri_runs[ri_runs.object_id == obj.object_id]
        assert len(runs) == 1, obj.object_id
        run = runs.iloc[0]
        evs = events[events.ri_run_id == run.ri_run_id]
        r, c = analyse_run(obj, run, evs, args.extracted)
        rows.append(r)
        curves[(r["coupling"], r["context"], r["seed"])] = c
    df = pd.DataFrame(rows)
    me = float(df.min_effect.iloc[0])
    assert (df.min_effect == me).all()
    thr = NONPLANTED_FACTOR * me

    # ------------------------------------------------------------------ sanity checks
    sanity = {
        "tycho_file": str(Path(tycho.__file__).resolve()),
        "n_markov_region_objects": n_region,
        "n_one_coupling_runs": len(df),
        "runs_by_context": df.groupby("context").size().to_dict(),
        "all_candidates_stored": frac(df.all_candidates_stored.sum(), len(df)),
        "gt_header_ok": frac(df.gt_header_ok.sum(), len(df)),
        "planted_subset_stored": frac(df.gt_found.sum(), len(df)),
        "events_table_matches_detect_events_alpha001": frac(df.events_table_matches_recompute.sum(), len(df)),
        "kl_final_finite": frac(df.kl_final_finite.sum(), len(df)),
        "last_checkpoint_equals_budget": frac((df.last_checkpoint == df.budget).sum(), len(df)),
        "lc_last_step_equals_budget": frac((df.lc_last_step == df.budget).sum(), len(df)),
        "T_values": sorted(df["T"].unique().tolist()),
        "planted_final_f_positive": frac((df.gt_f_final > 0).sum(), len(df)),
        "planted_final_f_over_tanh_beta": {"min": float((df.gt_f_final / df.gt_f_pop).min()),
                                           "median": float((df.gt_f_final / df.gt_f_pop).median()),
                                           "max": float((df.gt_f_final / df.gt_f_pop).max())},
        "planted_activation_left_censored": frac(df.gt_activation_left_censored.sum(), len(df)),
        "planted_activation_left_censored_by_coupling": df[df.gt_activation_left_censored].groupby(
            ["coupling", "context"]).size().reset_index(name="n").to_dict("records"),
        "planted_f_at_step0_negative": frac((df.gt_f_at0 < 0).sum(), len(df)),
    }
    # initialisation offsets: identical across chain / iid twins (same seed -> same initial weights and probes)
    same0, n_tw = 0, 0
    for (cp, ctx, sd), c in curves.items():
        if ctx != "chain" or (cp, "iid", sd) not in curves:
            continue
        ci = curves[(cp, "iid", sd)]
        n_tw += 1
        same0 += int(c.subsets == ci.subsets and np.allclose(c.ri[:, 0], ci.ri[:, 0]))
    sanity["step0_ri_identical_in_chain_iid_twins"] = frac(same0, n_tw)
    sanity["init_ri_max_over_runs"] = float(df.init_ri_max.max())
    sanity["init_ri_max_subset_of_max"] = df.loc[df.init_ri_max.idxmax(), "init_ri_argmax"]

    # ------------------------------------------------------------------ check 3a
    def by_coupling_table(d: pd.DataFrame) -> list[dict]:
        out = []
        for (cp, ctx), g in d.groupby(["coupling", "context"], sort=False):
            out.append(dict(
                coupling=cp, context=ctx, n_runs=len(g), seeds=sorted(g.seed.tolist()),
                n_kl_final_below=int(g.kl_final_ok.sum()), n_kl_ever_below=int(g.kl_ever_ok.sum()),
                kl_final_median=float(g.kl_final.median()), kl_final_max=float(g.kl_final.max()),
                kl_first_step_below_median=fnum(g.kl_first_step_below.median()),
                kl_first_step_below_range=[fnum(g.kl_first_step_below.min()), fnum(g.kl_first_step_below.max())],
                n_planted_activation=int(g.gt_activation.sum()),
                n_planted_activation_among_kl_ok=int((g.gt_activation & g.kl_final_ok).sum()),
                n_planted_activation_left_censored=int(g.gt_activation_left_censored.sum()),
                n_planted_onset_C=int(g.gt_onset_C.sum()), n_planted_final_active=int(g.gt_final_active.sum()),
                n_np_peak_over=int(g.np_over.sum()), np_peak_max=float(g.np_peak_max.max()),
                n_np_event_peak_over=int(g.np_event_over.sum()),
                n_np_last5_ok=int(g.np_last5_ok.sum()), np_last5_max=float(g.np_last5_max.max()),
                gt_ri_final_median=float(g.gt_ri_final.median()), gt_f_final_median=float(g.gt_f_final.median()),
                f_pop=float(g.gt_f_pop.iloc[0])))
        return out

    order_key = {c: i for i, c in enumerate(SINGLE)}
    df["_ord"] = df.coupling.map(order_key)
    df = df.sort_values(["_ord", "context", "seed"]).drop(columns="_ord").reset_index(drop=True)
    table = by_coupling_table(df)

    def groups(d, col):
        return [g[col].to_numpy(dtype=float) for _, g in d.groupby("coupling", sort=True)]

    check_a, check_b, check_c = {}, {}, {}
    for ctx in CONTEXTS + ("all",):
        d = df if ctx == "all" else df[df.context == ctx]
        den = d[d.kl_final_ok]
        check_a[ctx] = {
            "runs": len(d), "kl_final_below": frac(d.kl_final_ok.sum(), len(d)),
            "kl_ever_below": frac(d.kl_ever_ok.sum(), len(d)),
            "literal_planted_activates": frac(den.gt_activation.sum(), len(den)),
            "literal_pass": bool(len(den) and den.gt_activation.mean() >= A_BAR),
            "alt_onset_C_rule": frac(den.gt_onset_C.sum(), len(den)),
            "alt_onset_C_pass": bool(len(den) and den.gt_onset_C.mean() >= A_BAR),
            "alt_final_active": frac(den.gt_final_active.sum(), len(den)),
            "left_censored_first_activation": frac(den.gt_activation_left_censored.sum(), len(den)),
        }
        check_b[ctx] = {
            "runs": len(d), "literal_np_peak_over_10x": frac(d.np_over.sum(), len(d)),
            "literal_pass": bool(d.np_over.mean() <= B_BAR),
            "ci95_seed_stratified": boot_fraction(groups(d, "np_over"), "seed"),
            "ci95_config": boot_fraction(groups(d, "np_over"), "config"),
            "alt_denominator_kl_final_ok": frac(d[d.kl_final_ok].np_over.sum(), int(d.kl_final_ok.sum())),
            "alt_event_peak_over_10x": frac(d.np_event_over.sum(), len(d)),
            "alt_event_pass": bool(d.np_event_over.mean() <= B_BAR),
            "diag_peak_over_10x_excluding_step0": frac((d.np_peak_max_excl_t0 >= thr).sum(), len(d)),
            "max_np_peak": float(d.np_peak_max.max()),
            "runs_over_by_coupling": d[d.np_over].groupby("coupling").size().to_dict(),
        }
        check_c[ctx] = {
            "runs": len(d), "amended_last5_max_le_10x": frac(d.np_last5_ok.sum(), len(d)),
            "amended_pass": bool(d.np_last5_ok.mean() >= C_BAR),
            "ci95_seed_stratified": boot_fraction(groups(d, "np_last5_ok"), "seed"),
            "max_last5": float(d.np_last5_max.max()),
            "subset_of_max_last5": d.loc[d.np_last5_max.idxmax(), "np_last5_subset"],
            "median_last5_max": float(d.np_last5_max.median()),
            "alt_mean_last5_le_10x": frac(d.np_last5_mean_ok.sum(), len(d)),
            "alt_final_checkpoint_le_10x": frac(d.np_final_ok.sum(), len(d)),
        }
    check_a["pass_both_contexts"] = all(check_a[c]["literal_pass"] for c in CONTEXTS)
    check_b["pass_both_contexts"] = all(check_b[c]["literal_pass"] for c in CONTEXTS)
    check_b["pass_pooled"] = check_b["all"]["literal_pass"]
    check_c["pass_both_contexts"] = all(check_c[c]["amended_pass"] for c in CONTEXTS)

    # ------------------------------------------------------------------ transients (chain): subsets over 10x
    trans = []
    for _, r in df[df.context == "chain"].iterrows():
        c = curves[(r.coupling, "chain", r.seed)]
        gi = c.index_of(r.planted)
        evs = detect_events(c, alpha=0.01)
        tw = curves.get((r.coupling, "iid", r.seed))
        for i in range(c.S):
            if i == gi:
                continue
            ri = c.ri[i]
            pk = float(np.nanmax(ri))
            if pk < thr:
                continue
            names = c.subset_names(i)
            tpk = int(np.nanargmax(ri))
            sub_ev = [e for e in evs if e.subset_names == names]
            act = [e for e in sub_ev if e.kind == "activation"]
            ext_after = [e for e in sub_ev if e.kind == "extinction" and e.step > c.steps[tpk]]
            ext_any = [e for e in sub_ev if e.kind == "extinction"]
            on_t = r.gt_onset_C_idx
            on_lit = r.gt_activation_step
            # last checkpoint where RI >= 10x min_effect, and the first after the peak below min_effect / 2 (2 in a row)
            above = np.flatnonzero(ri >= thr)
            below_half = ri < me / 2
            t_dec = first_pair(below_half, start=tpk)
            rec = dict(
                coupling=r.coupling, beta=r.beta, lags_planted=r.lags, seed=int(r.seed), subset=list(names),
                subset_lags=list(lags_of(names)), order=len(names),
                peak_ri=pk, peak_step=int(c.steps[tpk]), peak_idx=tpk, f_at_peak=fnum(c.fourier[i, tpk]),
                ri_at0=float(ri[0]), f_at0=fnum(c.fourier[i, 0]),
                first_step_ge_10x=int(c.steps[above[0]]), last_step_ge_10x=int(c.steps[above[-1]]),
                activation_step=act[0].step if act else None, activation_step_lo=act[0].step_lo if act else None,
                activation_left_censored=bool(act and act[0].left_censored),
                extinction_step=ext_after[0].step if ext_after else (ext_any[-1].step if ext_any else None),
                n_reactivations=sum(e.kind == "reactivation" for e in sub_ev),
                decay_below_half_min_effect_step=None if t_dec is None else int(c.steps[t_dec]),
                final_ri=float(ri[-1]), last5_max=float(np.nanmax(ri[-LAST_K:])),
                planted_onset_C_step=r.gt_onset_C_step, planted_onset_C_idx=on_t,
                planted_onset_lit_step=on_lit, planted_onset_E25_step=r.gt_onset_E25_step,
                ri_at_planted_onset_C=None if on_t is None else float(ri[on_t]),
                planted_ri_at_peak=float(c.ri[gi, tpk]), planted_f_at_peak=fnum(c.fourier[gi, tpk]),
            )
            if on_t is not None:
                rec["peak_minus_onset_idx"] = tpk - on_t
                rec["peak_minus_onset_steps"] = int(c.steps[tpk]) - int(c.steps[on_t])
                rec["peak_over_onset_step_ratio"] = (c.steps[tpk] / c.steps[on_t]) if c.steps[on_t] > 0 else None
                rec["first_ge_10x_minus_onset_idx"] = int(above[0]) - on_t
                if act:
                    ta = int(np.searchsorted(c.steps, act[0].step))
                    rec["activation_minus_onset_idx"] = ta - on_t
                    rec["activation_minus_onset_steps"] = act[0].step - int(c.steps[on_t])
                if rec["extinction_step"] is not None:
                    te = int(np.searchsorted(c.steps, rec["extinction_step"]))
                    rec["extinction_minus_onset_idx"] = te - on_t
                    rec["extinction_minus_onset_steps"] = rec["extinction_step"] - int(c.steps[on_t])
                    rec["extinction_over_onset_step_ratio"] = rec["extinction_step"] / int(c.steps[on_t])
            # early growth ratio vs the planted coefficient (from step 0 to the planted E25 checkpoint)
            tE = int(np.searchsorted(c.steps, r.gt_onset_E25_step)) if not r.gt_onset_E25_censored else None
            if tE is not None and tE > 0:
                dA = c.fourier[i, tE] - c.fourier[i, 0]
                dL = c.fourier[gi, tE] - c.fourier[gi, 0]
                rec["growth_ratio_dfA_over_dfL_at_planted_E25"] = fnum(dA / dL) if dL != 0 else None
            # same ratio at the first checkpoint where the planted coefficient has GROWN by >= 10 % of tanh(beta)
            grow = np.sign(r.beta) * (c.fourier[gi] - c.fourier[gi, 0])
            tG = first_true(grow >= 0.10 * abs(r.gt_f_pop))
            if tG is not None and tG > 0:
                rec["growth_ratio_dfA_over_dfL_at_planted_growth10"] = fnum(
                    (c.fourier[i, tG] - c.fourier[i, 0]) / (c.fourier[gi, tG] - c.fourier[gi, 0]))
                rec["growth10_step"] = int(c.steps[tG])
            # initialisation-level crossing: peak within the first three checkpoints and >= 90 % of it present at step 0
            rec["init_level"] = bool(tpk <= 2 and ri[0] >= 0.9 * pk)
            # iid twin on the same subset
            if tw is not None:
                j = tw.index_of(list(names))
                rj = tw.ri[j]
                late = tw.steps >= 10
                rec.update(iid_peak_from_step10=float(np.nanmax(rj[late])),
                           iid_detected_steps=[int(x) for x in tw.steps[tw.detected()[j]]])
                rec.update(iid_twin=True, iid_peak_ri=float(np.nanmax(rj)), iid_peak_step=int(tw.steps[int(np.nanargmax(rj))]),
                           iid_peak_excl_step0=float(np.nanmax(rj[1:])), iid_ri_at_chain_peak_step=float(rj[tpk]),
                           iid_ri_at0=float(rj[0]), iid_last5_max=float(np.nanmax(rj[-LAST_K:])),
                           iid_f_at_chain_peak_step=fnum(tw.fourier[j, tpk]),
                           iid_detected_ever=bool(tw.detected()[j].any()))
            else:
                rec.update(iid_twin=False)
            trans.append(rec)
    tr = pd.DataFrame(trans)

    # ------------------------------------------------------------------ chain statistics per coupling set
    chain_stats = {}
    stats_rows = []
    for cp in SINGLE:
        g = df[df.coupling == cp]
        L, beta = tuple(g.lags.iloc[0]), float(g.beta.iloc[0])
        maxo = int(g.max_order.iloc[0])
        cfg = json.loads(sel[sel.coupling == cp].task_cfg_json.iloc[0])
        subsets = [A for k in range(1, maxo) for A in itertools.combinations(range(1, 7), k) if A != L]
        cs = chain_statistics(cfg, L, beta, subsets)
        chain_stats[cp] = cs
        # model peaks per subset: median over seeds, chain vs iid; runs over 10x
        for A in subsets:
            names = list(names_of(A))
            pk = {}
            for ctx in CONTEXTS:
                vals, vals1 = [], []
                for sd in g[g.context == ctx].seed:
                    c = curves[(cp, ctx, sd)]
                    i = c.index_of(names)
                    vals.append(float(np.nanmax(c.ri[i])))
                    vals1.append(float(np.nanmax(c.ri[i, 1:])))
                pk[ctx] = (np.array(vals), np.array(vals1))
            sc, si = cs["chain"]["subsets"][A], cs["iid"]["subsets"][A]
            stats_rows.append(dict(
                coupling=cp, beta=beta, subset=names, subset_lags=list(A), order=len(A) + 1,
                rho_chain_sim=sc["rho"], rho_chain_analytic=sc["rho_analytic"], shifts_J=sc["shifts_J"],
                rho_iid_sim=si["rho"], ri_bayes_marginal_chain=sc["ri_bayes_marginal_uniform_probe"],
                g_walsh_top_chain=sc["g_walsh_top_uniform"], g_walsh_lower_max_abs_chain=sc["g_walsh_lower_max_abs"],
                planted_signal_left_after_g=sc["planted_signal_left_after_g"],
                chain_peak_median=float(np.median(pk["chain"][0])), chain_peak_max=float(np.max(pk["chain"][0])),
                chain_peak_excl0_median=float(np.median(pk["chain"][1])),
                iid_peak_median=float(np.median(pk["iid"][0])), iid_peak_max=float(np.max(pk["iid"][0])),
                iid_peak_excl0_median=float(np.median(pk["iid"][1])),
                n_chain_over_10x=int((pk["chain"][0] >= thr).sum()), n_chain=int(pk["chain"][0].size),
                n_iid_over_10x=int((pk["iid"][0] >= thr).sum()), n_iid=int(pk["iid"][0].size)))
    st_df = pd.DataFrame(stats_rows)
    # how well the chain statistics explain the model's chain-only excess
    st_df["excess_chain_minus_iid_median_excl0"] = st_df.chain_peak_excl0_median - st_df.iid_peak_excl0_median
    nz = st_df.rho_chain_analytic.abs() > 0
    rho_sp = spearmanr(st_df.rho_chain_sim.abs(), st_df.excess_chain_minus_iid_median_excl0)
    explain = {
        "n_subset_coupling_pairs": len(st_df),
        "n_pairs_with_nonzero_analytic_rho": int(nz.sum()),
        "max_abs_rho_sim_where_analytic_zero": float(st_df[~nz].rho_chain_sim.abs().max()),
        "max_abs_sim_minus_analytic": float((st_df.rho_chain_sim - st_df.rho_chain_analytic).abs().max()),
        "max_abs_rho_iid_sim": float(st_df.rho_iid_sim.abs().max()),
        "chain_excess_median_where_rho_nonzero": float(st_df[nz].excess_chain_minus_iid_median_excl0.median()),
        "chain_excess_median_where_rho_zero": float(st_df[~nz].excess_chain_minus_iid_median_excl0.median()),
        "spearman_absrho_vs_chain_excess": {"rho": float(rho_sp.statistic), "p": float(rho_sp.pvalue),
                                            "n": len(st_df)},
        "runs_over_10x_on_rho_nonzero_subsets": int(st_df[nz].n_chain_over_10x.sum()),
        "runs_over_10x_on_rho_zero_subsets": int(st_df[~nz].n_chain_over_10x.sum()),
        "iid_runs_over_10x_total": int(st_df.n_iid_over_10x.sum()),
    }
    sp_nz = spearmanr(st_df[nz].ri_bayes_marginal_chain, st_df[nz].chain_peak_excl0_median)
    explain["spearman_bayes_marginal_ri_vs_chain_peak_median_where_rho_nonzero"] = {
        "rho": float(sp_nz.statistic), "p": float(sp_nz.pvalue), "n": int(nz.sum())}
    explain["n_pairs_rho_nonzero_with_any_chain_run_over_10x"] = int((st_df[nz].n_chain_over_10x > 0).sum())
    explain["pairs_rho_nonzero_with_any_chain_run_over_10x"] = [
        f"{r.coupling} {sname(r.subset)}" for r in st_df[nz & (st_df.n_chain_over_10x > 0)].itertuples()]
    explain["beta_of_pairs_over_10x"] = sorted(set(st_df[st_df.n_chain_over_10x > 0].beta.tolist()))
    explain["max_ri_bayes_marginal_for_beta_below_1"] = float(st_df[st_df.beta < 1].ri_bayes_marginal_chain.max())
    explain["max_chain_peak_for_beta_below_1"] = float(st_df[st_df.beta < 1].chain_peak_max.max())
    # attach chain statistics to each transient
    if len(tr):
        key = {(r.coupling, tuple(r.subset_lags)): r for r in st_df.itertuples()}
        for col_src, col_dst in (("rho_chain_sim", "rho_chain_sim"), ("rho_chain_analytic", "rho_chain_analytic"),
                                 ("ri_bayes_marginal_chain", "ri_bayes_marginal_chain"),
                                 ("g_walsh_top_chain", "g_walsh_top_chain"),
                                 ("planted_signal_left_after_g", "planted_signal_left_after_g"),
                                 ("rho_iid_sim", "rho_iid_sim")):
            tr[col_dst] = [getattr(key[(cp, tuple(l))], col_src) for cp, l in zip(tr.coupling, tr.subset_lags)]
        tr["peak_over_bayes_marginal_ri"] = tr.peak_ri / tr.ri_bayes_marginal_chain
        tr["f_peak_sign_matches_rho"] = np.sign(tr.f_at_peak) == np.sign(tr.rho_chain_sim)
        tr["rho_ratio_A_over_L"] = tr.rho_chain_sim / [chain_stats[cp]["chain"]["rho_planted"] for cp in tr.coupling]

    # aggregate transients by (coupling, subset)
    agg = []
    for (cp, sub), g in tr.groupby(["coupling", tr.subset.apply(tuple)], sort=False):
        n_chain = int((df.coupling == cp).mul(df.context == "chain").sum())
        tw = g[g.iid_twin]

        def rng_(col, d=g):
            v = d[col].dropna().astype(float)
            return [fnum(v.min()), fnum(v.median()), fnum(v.max())] if len(v) else None
        agg.append(dict(
            coupling=cp, subset=list(sub), n_runs=len(g), n_chain_runs=n_chain, seeds=sorted(g.seed.tolist()),
            peak_ri_min_med_max=rng_("peak_ri"), peak_step_min_med_max=rng_("peak_step"),
            f_at_peak_min_med_max=rng_("f_at_peak"),
            activation_step_min_med_max=rng_("activation_step"), extinction_step_min_med_max=rng_("extinction_step"),
            n_activation_left_censored=int(g.activation_left_censored.sum()),
            planted_onset_C_step_min_med_max=rng_("planted_onset_C_step"),
            activation_minus_onset_idx_min_med_max=rng_("activation_minus_onset_idx"),
            peak_minus_onset_idx_min_med_max=rng_("peak_minus_onset_idx"),
            extinction_minus_onset_idx_min_med_max=rng_("extinction_minus_onset_idx"),
            n_peak_before_planted_onset=int((g.peak_minus_onset_idx < 0).sum()),
            n_peak_at_or_after_planted_onset=int((g.peak_minus_onset_idx >= 0).sum()),
            n_activation_before_planted_onset=int((g.activation_minus_onset_idx < 0).sum()),
            n_extinct_by_end=int(g.extinction_step.notna().sum()), last5_max_max=float(g.last5_max.max()),
            ri_at_planted_onset_min_med_max=rng_("ri_at_planted_onset_C"),
            n_iid_twins=len(tw), iid_peak_min_med_max=rng_("iid_peak_ri", tw),
            iid_peak_excl_step0_min_med_max=rng_("iid_peak_excl_step0", tw),
            iid_ri_at_chain_peak_step_min_med_max=rng_("iid_ri_at_chain_peak_step", tw),
            n_iid_detected_ever=int(tw.iid_detected_ever.sum()) if len(tw) else 0,
            rho_chain_sim=float(g.rho_chain_sim.iloc[0]), rho_chain_analytic=float(g.rho_chain_analytic.iloc[0]),
            rho_iid_sim=float(g.rho_iid_sim.iloc[0]),
            ri_bayes_marginal_chain=float(g.ri_bayes_marginal_chain.iloc[0]),
            g_walsh_top_chain=float(g.g_walsh_top_chain.iloc[0]),
            peak_over_bayes_marginal_min_med_max=rng_("peak_over_bayes_marginal_ri"),
            planted_signal_left_after_g=float(g.planted_signal_left_after_g.iloc[0]),
            n_f_sign_matches_rho=int(g.f_peak_sign_matches_rho.sum()),
            rho_ratio_A_over_L=float(g.rho_ratio_A_over_L.iloc[0]),
            growth_ratio_min_med_max=rng_("growth_ratio_dfA_over_dfL_at_planted_E25")
            if "growth_ratio_dfA_over_dfL_at_planted_E25" in g else None,
            growth10_ratio_min_med_max=rng_("growth_ratio_dfA_over_dfL_at_planted_growth10")
            if "growth_ratio_dfA_over_dfL_at_planted_growth10" in g else None,
            first_step_ge_10x_min_med_max=rng_("first_step_ge_10x"), last_step_ge_10x_min_med_max=rng_("last_step_ge_10x"),
            first_ge_10x_minus_onset_idx_min_med_max=rng_("first_ge_10x_minus_onset_idx"),
            iid_peak_from_step10_min_med_max=rng_("iid_peak_from_step10", tw),
            n_init_level=int(g.init_level.sum())))
    agg_df = pd.DataFrame(agg)

    # ------------------------------------------------------------------ delay of the planted term: chain vs iid
    delay = []
    for k_cp, cp in enumerate(SINGLE):
        g = df[df.coupling == cp]
        tri = tr[(tr.coupling == cp) & ~tr.init_level]
        rowd = {"coupling": cp, "has_transients_over_10x": bool(len(tri)),
                "n_chain_runs_with_transients": int(tri.seed.nunique()),
                "n_chain_runs_over_10x_incl_init_level": int(tr[tr.coupling == cp].seed.nunique())}
        for k_m, (meas, cens) in enumerate((("gt_onset_lit_step_or_budget", None), ("gt_onset_C_step", "gt_onset_C_censored"),
                                            ("gt_onset_E25_step", "gt_onset_E25_censored"),
                                            ("gt_onset_E10_step", None), ("gt_onset_E50_step", None))):
            a = g[g.context == "chain"][meas].to_numpy(float)
            b = g[g.context == "iid"][meas].to_numpy(float)
            d_ = {"chain_median": float(np.median(a)), "iid_median": float(np.median(b)),
                  "chain_range": [float(a.min()), float(a.max())], "iid_range": [float(b.min()), float(b.max())],
                  "n_chain": int(a.size), "n_iid": int(b.size)}
            if cens:
                d_["n_censored_chain"] = int(g[g.context == "chain"][cens].sum())
                d_["n_censored_iid"] = int(g[g.context == "iid"][cens].sum())
            if np.median(b) > 0:
                d_["ratio_chain_over_iid"] = boot_ratio_of_medians(a, b, BOOT_SEED + 100 * k_cp + k_m)
            else:
                d_["ratio_chain_over_iid"] = None
            if meas == "gt_onset_lit_step_or_budget":
                d_["n_left_censored_chain"] = int(g[g.context == "chain"].gt_activation_left_censored.sum())
                d_["n_left_censored_iid"] = int(g[g.context == "iid"].gt_activation_left_censored.sum())
            rowd[meas.replace("gt_onset_", "onset_").replace("_step_or_budget", "").replace("_step", "")] = d_
        delay.append(rowd)
    # within-chain association for the strongest case
    assoc = {}
    for cp in sorted(set(tr.coupling)):
        g = tr[tr.coupling == cp]
        main_sub = g.groupby(g.subset.apply(tuple)).size().idxmax()
        gg = g[g.subset.apply(tuple) == main_sub]
        if len(gg) >= 5:
            s1 = spearmanr(gg.peak_ri, gg.planted_onset_C_step)
            s2 = spearmanr(gg.peak_step, gg.planted_onset_C_step)
            assoc[cp] = {"subset": list(main_sub), "n": len(gg),
                         "spearman_peak_ri_vs_planted_onset": [fnum(s1.statistic), fnum(s1.pvalue)],
                         "spearman_peak_step_vs_planted_onset": [fnum(s2.statistic), fnum(s2.pvalue)]}

    # ------------------------------------------------------------------ chart data
    chart = {}
    for cp in sorted(set(tr.coupling)):
        g = tr[tr.coupling == cp]
        main_sub = list(g.groupby(g.subset.apply(tuple)).size().idxmax())
        planted = df[df.coupling == cp].planted.iloc[0]
        series = {}
        for ctx in CONTEXTS:
            runs = []
            for sd in sorted(df[(df.coupling == cp) & (df.context == ctx)].seed):
                c = curves[(cp, ctx, sd)]
                i, gi = c.index_of(main_sub), c.index_of(planted)
                runs.append({"seed": int(sd), "transient_ri": c.ri[i].tolist(), "transient_f": c.fourier[i].tolist(),
                             "planted_ri": c.ri[gi].tolist(), "planted_f": c.fourier[gi].tolist()})
            arr_t = np.array([r_["transient_ri"] for r_ in runs])
            arr_p = np.array([r_["planted_ri"] for r_ in runs])
            series[ctx] = {"runs": runs, "median_transient_ri": np.median(arr_t, axis=0).tolist(),
                           "median_planted_ri": np.median(arr_p, axis=0).tolist()}
        chart[cp] = {"steps": curves[(cp, "chain", int(df[df.coupling == cp].seed.iloc[0]))].steps.tolist(),
                     "transient_subset": main_sub, "planted_subset": list(planted), "series": series}
    chart_bars = [{"coupling": r["coupling"], "context": r["context"], "np_peak_max_per_run":
                   df[(df.coupling == r["coupling"]) & (df.context == r["context"])].np_peak_max.round(6).tolist(),
                   "np_last5_max_per_run":
                   df[(df.coupling == r["coupling"]) & (df.context == r["context"])].np_last5_max.round(6).tolist()}
                  for r in table]

    # ------------------------------------------------------------------ items (structured result)
    a_c, a_i, a_all = check_a["chain"], check_a["iid"], check_a["all"]
    b_c, b_i, b_all = check_b["chain"], check_b["iid"], check_b["all"]
    c_c, c_i, c_all = check_c["chain"], check_c["iid"], check_c["all"]

    def fr(x):
        return f"{x['num']} of {x['den']}"

    def rg(v, fmt="{:.4g}"):
        """[min, median, max] -> 'min-max (median m)'."""
        if v is None:
            return "n/a"
        lo, md, hi = v
        if lo == hi:
            return fmt.format(lo)
        return f"{fmt.format(lo)} to {fmt.format(hi)} (median {fmt.format(md)})"

    ci_chain = df[df.context == "chain"]
    lc_runs = df[df.gt_activation_left_censored]
    lc_desc = ", ".join(f"{r.coupling} {r.context} seed {r.seed} (kinds {'/'.join(r.gt_event_kinds)} at steps "
                        f"{r.gt_event_steps}; f(step 0) = {r.gt_f_at0:+.3f})" for r in lc_runs.itertuples())
    lc_ext_max = max(st_ for r in lc_runs.itertuples() for k_, st_ in zip(r.gt_event_kinds, r.gt_event_steps)
                     if k_ == "extinction")
    lc_react_max = int(lc_runs.gt_onset_C_step.max())
    init_runs = tr[tr.init_level]
    chain_induced = tr[~tr.init_level]
    n_runs_induced = chain_induced.groupby(["coupling", "seed"]).ngroups
    only_init = sorted(set(map(tuple, init_runs[["coupling", "seed"]].values.tolist())) -
                       set(map(tuple, chain_induced[["coupling", "seed"]].values.tolist())))
    items = []
    items.append(dict(
        id="check3a",
        question="Check 3 (a), literal: does the planted subset activate in >= 90 % of the one-coupling Markov runs "
                 "whose final kl_true < 0.01, in each training context? And how many runs reach kl_true < 0.01 at all?",
        definition_used=f"Runs: markov-v1, the 10 one-coupling sets x {{chain, iid}}, every seed of the sky ({len(df)} "
                        f"runs: chain {len(ci_chain)}, iid {len(df) - len(ci_chain)}; 1,2,3:1.0 chain has 12 seeds). "
                        f"Final kl_true = last value of the light curve's x_kl_true (step 30,000; measured on the test "
                        f"split of the run's own train_context). 'Activates' = the planted subset {{x[t-l], l in L}} + "
                        f"{{xhat}} has an event of kind 'activation' in ri_events (identical to detect_events(curve, "
                        f"alpha=0.01) for all {len(df)} runs), left-censored activations included. Plausible "
                        f"alternatives: (a2) Amendment 1 C's onset rule, a (re)activation that is not left-censored and "
                        f"at whose checkpoint the planted Fourier coefficient has the sign of its final value; (a3) the "
                        f"planted subset is active at the end of training. 'Reach kl_true < 0.01 at all' = any "
                        f"light-curve point below 0.01.",
        literal_result=f"chain: {fr(a_c['literal_planted_activates'])} runs with final kl_true < 0.01 have a planted "
                       f"activation; iid: {fr(a_i['literal_planted_activates'])}; pooled "
                       f"{fr(a_all['literal_planted_activates'])}. Final kl_true < 0.01 in "
                       f"{fr(a_c['kl_final_below'])} chain and {fr(a_i['kl_final_below'])} iid runs; kl_true < 0.01 "
                       f"reached at some point in {fr(a_c['kl_ever_below'])} chain and {fr(a_i['kl_ever_below'])} iid "
                       f"runs, i.e. every seed of every coupling set (per-set counts and first crossing steps in "
                       f"by_coupling_context). Largest final kl_true: chain {ci_chain.kl_final.max():.5f}, iid "
                       f"{df[df.context == 'iid'].kl_final.max():.5f} nats.",
        literal_verdict="pass" if check_a["pass_both_contexts"] else "fail",
        amended_result="Not changed by Amendment 1. Alternatives give the same counts: (a2) C-rule chain "
                       f"{fr(a_c['alt_onset_C_rule'])}, iid {fr(a_i['alt_onset_C_rule'])}; (a3) active at the end: "
                       f"chain {fr(a_c['alt_final_active'])}, iid {fr(a_i['alt_final_active'])}.",
        amended_verdict="not_applicable", deciding="literal",
        key_numbers="\n".join([
            f"chain: planted activation in {fr(a_c['literal_planted_activates'])} runs with final kl_true < 0.01 (bar 90 %)",
            f"iid: planted activation in {fr(a_i['literal_planted_activates'])}",
            f"final kl_true < 0.01: chain {fr(a_c['kl_final_below'])}, iid {fr(a_i['kl_final_below'])}",
            f"kl_true < 0.01 at any point: chain {fr(a_c['kl_ever_below'])}, iid {fr(a_i['kl_ever_below'])}",
            f"max final kl_true: chain {ci_chain.kl_final.max():.5f}, iid {df[df.context == 'iid'].kl_final.max():.5f}",
            f"C-rule alternative: chain {fr(a_c['alt_onset_C_rule'])}, iid {fr(a_i['alt_onset_C_rule'])}",
            f"first planted activation left-censored (step 0): chain {fr(a_c['left_censored_first_activation'])}, iid "
            f"{fr(a_i['left_censored_first_activation'])}"]),
        interpretation="The instrument finds the planted coupling in every converged run of both contexts; check 3 (a) "
                       "passes. The kl_true conditioning removes nothing: every run converged.",
        caveats=f"{len(lc_runs)} runs (all order 2, coupling sets 1:1.0 and 1:0.5) have a left-censored first planted "
                f"activation at step 0: the untrained network's planted-pair RI is significant there with the wrong "
                f"sign (Amendment 1 C). Each goes extinct by step {lc_ext_max} and re-activates with the right sign "
                f"by step {lc_react_max}, so the C-rule alternative gives the same counts: {lc_desc}."))
    items.append(dict(
        id="check3b",
        question="Check 3 (b), literal: does a non-planted subset with peak RI >= 10 x min_effect occur in <= 5 % of "
                 "the one-coupling runs, in each training context?",
        definition_used=f"Non-planted = every stored subset other than the planted one; every Markov curve stores all "
                        f"n_candidates scanned subsets (xhat plus 1 to max_order - 1 lags; checked for {len(df)} of "
                        f"{len(df)} runs). Peak RI = max over all 60 checkpoints (step 0 included) of the stored "
                        f"bias-corrected RI; threshold 10 x min_effect = {thr:g} nats (min_effect = {me:g} in every "
                        f"run). Denominator: all one-coupling runs of the context (plausible alternative: runs with "
                        f"final kl_true < 0.01; identical here). Alternative peak: peak_ri of a non-planted "
                        f"(re)activation event, which also requires q <= 0.01. 95 % bootstrap CIs (2,000 resamples, "
                        f"seed {BOOT_SEED}) resample seeds within coupling set, or coupling sets.",
        literal_result=f"chain: {fr(b_c['literal_np_peak_over_10x'])} runs ({100 * b_c['literal_np_peak_over_10x']['frac']:.1f} %, "
                       f"CI {100 * b_c['ci95_seed_stratified'][0]:.1f}-{100 * b_c['ci95_seed_stratified'][1]:.1f} % "
                       f"over seeds, {100 * b_c['ci95_config'][0]:.1f}-{100 * b_c['ci95_config'][1]:.1f} % over "
                       f"coupling sets); by set {b_c['runs_over_by_coupling']}. iid: "
                       f"{fr(b_i['literal_np_peak_over_10x'])} (largest non-planted peak {b_i['max_np_peak']:.5f} "
                       f"nats, the initialisation offset). Pooled {fr(b_all['literal_np_peak_over_10x'])}. The "
                       f"event-based peak and the kl-conditioned denominator give the same counts.",
        literal_verdict="fail" if not check_b["pass_both_contexts"] else "pass",
        amended_result="Superseded by (c) (Amendment 1 B, which decides): see check3c.",
        amended_verdict="not_applicable", deciding="amended",
        key_numbers="\n".join([
            f"chain: {fr(b_c['literal_np_peak_over_10x'])} runs with a non-planted peak >= {thr:g} nats (bar <= 5 %) -> "
            f"{'pass' if b_c['literal_pass'] else 'fail'}",
            f"iid: {fr(b_i['literal_np_peak_over_10x'])} -> {'pass' if b_i['literal_pass'] else 'fail'}",
            f"pooled: {fr(b_all['literal_np_peak_over_10x'])}",
            f"chain runs over threshold by coupling set: {b_c['runs_over_by_coupling']}",
            f"largest non-planted peak: chain {b_c['max_np_peak']:.4f}, iid {b_i['max_np_peak']:.5f} nats",
            f"event-based peak: chain {fr(b_c['alt_event_peak_over_10x'])}, iid {fr(b_i['alt_event_peak_over_10x'])}",
            f"chain CI (seeds) {b_c['ci95_seed_stratified'][0]:.3f}-{b_c['ci95_seed_stratified'][1]:.3f}; "
            f"(coupling sets) {b_c['ci95_config'][0]:.3f}-{b_c['ci95_config'][1]:.3f}",
            f"chain runs over threshold through an initialisation-level value only: {len(only_init)} "
            f"({', '.join(f'{a} seed {b}' for a, b in only_init)})",
            f"largest step-0 (untrained) RI of any subset: {sanity['init_ri_max_over_runs']:.5f} nats"]),
        interpretation="The registered peak criterion fails for chain training and passes for iid training. The "
                       "failures are real interactions learned transiently from the chain's correlated contexts "
                       "(result_chain_transients), not instrument noise: iid twins with the same initial weights and "
                       "probes stay below the bar. Amendment 1 B anticipated exactly this and replaced the criterion.",
        caveats=f"The untrained network's RI reaches {sanity['init_ri_max_over_runs']:.4f} nats at step 0 (on "
                f"{sname(sanity['init_ri_max_subset_of_max'])}, seed 7, identical in chain and iid twins), just "
                f"below the 0.01 bar; one chain run (1,3:0.5 seed 7) crosses the bar only through that offset "
                f"(0.0101 at step 2). Without it the chain count is {b_c['literal_np_peak_over_10x']['num'] - len(only_init)} "
                f"of {b_c['runs']}; the verdict is unchanged."))
    items.append(dict(
        id="check3c",
        question="Check 3 (c), Amendment 1 B (decides): at the end of training, is the largest non-planted RI over the "
                 "last five checkpoints <= 10 x min_effect in >= 95 % of runs, in both contexts?",
        definition_used=f"Per run: max over all stored non-planted subsets and over the last five checkpoints "
                        f"(steps 15,623, 17,800, 20,282, 26,330 and 30,000) of the bias-corrected RI; a run passes if "
                        f"it is <= {thr:g} nats. Plausible alternatives: max over subsets of the per-subset mean over "
                        f"the last five checkpoints; the final checkpoint only.",
        literal_result="(the registered criterion is (b), reported in check3b)",
        literal_verdict="not_applicable",
        amended_result=f"chain: {fr(c_c['amended_last5_max_le_10x'])} runs pass (largest value {c_c['max_last5']:.5f} "
                       f"nats, on {sname(c_c['subset_of_max_last5'])}); iid: {fr(c_i['amended_last5_max_le_10x'])} "
                       f"(largest {c_i['max_last5']:.5f}). Alternatives: mean over the last five chain "
                       f"{fr(c_c['alt_mean_last5_le_10x'])}, iid {fr(c_i['alt_mean_last5_le_10x'])}; final checkpoint "
                       f"chain {fr(c_c['alt_final_checkpoint_le_10x'])}, iid {fr(c_i['alt_final_checkpoint_le_10x'])}.",
        amended_verdict="pass" if check_c["pass_both_contexts"] else "fail",
        deciding="amended",
        key_numbers="\n".join([
            f"chain: {fr(c_c['amended_last5_max_le_10x'])} runs with last-five non-planted max <= {thr:g} (bar 95 %)",
            f"iid: {fr(c_i['amended_last5_max_le_10x'])}",
            f"largest last-five non-planted RI: chain {c_c['max_last5']:.5f}, iid {c_i['max_last5']:.5f} nats",
            f"median per-run last-five max: chain {c_c['median_last5_max']:.5f}, iid {c_i['median_last5_max']:.5f}"]),
        interpretation=f"Every chain-induced interaction has vanished by the end of training: the largest late "
                       f"non-planted RI is {max(c_c['max_last5'], c_i['max_last5']) / me:.1f} x min_effect, far below "
                       f"the 10 x bar, in both contexts. With (a) passing, check 3 passes under the amended (deciding) "
                       f"definition.",
        caveats="The margin is wide, so max versus mean over the last five checkpoints does not matter."))
    items.append(dict(
        id="check3",
        question="Check 3 overall (Markov instrument check): (a) together with the non-planted criterion.",
        definition_used="Literal = (a) and (b); amended = (a) and (c); Amendment 1 B states that the amended one "
                         "decides. Each evaluated per training context.",
        literal_result=f"(a) passes in both contexts; (b) chain {fr(b_c['literal_np_peak_over_10x'])} -> fail, iid "
                       f"{fr(b_i['literal_np_peak_over_10x'])} -> pass. Literal check 3: fails (chain), passes (iid).",
        literal_verdict="fail" if not (check_a["pass_both_contexts"] and check_b["pass_both_contexts"]) else "pass",
        amended_result=f"(a) passes; (c) chain {fr(c_c['amended_last5_max_le_10x'])}, iid "
                       f"{fr(c_i['amended_last5_max_le_10x'])}. Amended check 3: passes in both contexts.",
        amended_verdict="pass" if (check_a["pass_both_contexts"] and check_c["pass_both_contexts"]) else "fail",
        deciding="amended",
        key_numbers="\n".join([
            f"(a) chain {fr(a_c['literal_planted_activates'])}, iid {fr(a_i['literal_planted_activates'])}",
            f"(b) chain {fr(b_c['literal_np_peak_over_10x'])}, iid {fr(b_i['literal_np_peak_over_10x'])}",
            f"(c) chain {fr(c_c['amended_last5_max_le_10x'])}, iid {fr(c_i['amended_last5_max_le_10x'])}"]),
        interpretation="No sign of a bug or a mis-set threshold: iid runs pass every version; chain runs fail only "
                       "the peak version, because of real, transient, chain-induced interactions, which Amendment 1 "
                       "B re-classified as a result.",
        caveats="Sanity checks behind all three: tycho imported from the repository; every curve has 60 checkpoints "
                "ending at the 30,000-step budget; the planted subset is stored and named in the ground truth of "
                "every curve; the final planted coefficient is positive in every run (median "
                f"{sanity['planted_final_f_over_tanh_beta']['median']:.3f} x tanh(beta)); events table = "
                "detect_events recomputation in every run; step-0 RI identical in chain/iid twins."))

    # ---- result: transients
    lines = [f"chain runs with a non-planted peak >= {thr:g} nats: {b_c['literal_np_peak_over_10x']['num']} of "
             f"{b_c['runs']}; chain-induced (not initialisation-level): {n_runs_induced} of {b_c['runs']}; iid runs: "
             f"{b_i['literal_np_peak_over_10x']['num']} of {b_i['runs']}"]
    for r in agg_df.itertuples():
        if r.n_init_level == r.n_runs:
            lines.append(f"{r.coupling} {sname(r.subset)}: {r.n_runs} of {r.n_chain_runs} chain runs, "
                         f"initialisation-level only (peak {rg(r.peak_ri_min_med_max)} at step "
                         f"{rg(r.peak_step_min_med_max, '{:.0f}')}; iid twin {rg(r.iid_peak_min_med_max)}; rho_chain "
                         f"{r.rho_chain_sim:.3f}, analytic {r.rho_chain_analytic:.3f})")
            continue
        lines.append(
            f"{r.coupling} {sname(r.subset)}: {r.n_runs} of {r.n_chain_runs} chain runs; peak RI "
            f"{rg(r.peak_ri_min_med_max)} nats at steps {rg(r.peak_step_min_med_max, '{:.0f}')} (Fourier at peak "
            f"{rg(r.f_at_peak_min_med_max, '{:+.3f}')}); first >= {thr:g} at steps "
            f"{rg(r.first_step_ge_10x_min_med_max, '{:.0f}')}; activation event at steps "
            f"{rg(r.activation_step_min_med_max, '{:.0f}')}; extinction at steps "
            f"{rg(r.extinction_step_min_med_max, '{:.0f}')}; planted onset (C) at steps "
            f"{rg(r.planted_onset_C_step_min_med_max, '{:.0f}')}; peak minus planted onset "
            f"{rg(r.peak_minus_onset_idx_min_med_max, '{:+.0f}')} checkpoints ({r.n_peak_before_planted_onset} of "
            f"{r.n_runs} peaks before the onset); iid twins ({r.n_iid_twins}) peak {rg(r.iid_peak_min_med_max)}, "
            f"from step 10 on {rg(r.iid_peak_from_step10_min_med_max)}; chain E[s_t chi_A] = {r.rho_chain_sim:.3f} "
            f"(analytic {r.rho_chain_analytic:.3f}; iid {r.rho_iid_sim:+.4f}); Bayes-optimal marginal predictor: "
            f"Walsh {r.g_walsh_top_chain:.3f}, RI {r.ri_bayes_marginal_chain:.4f} nats; model peak / that RI "
            f"{rg(r.peak_over_bayes_marginal_min_med_max, '{:.2f}')}; early growth ratio dfA/dfL (planted "
            f"coefficient grown by 10 % of tanh(beta)) {rg(r.growth10_ratio_min_med_max, '{:.2f}')} vs "
            f"rho_A/rho_L = {r.rho_ratio_A_over_L:.2f}"
            + (f" ({r.n_init_level} initialisation-level)" if r.n_init_level else ""))
    ci_tw = chain_induced[chain_induced.iid_twin.astype(bool)]
    iid_at0 = int((ci_tw.iid_peak_step == 0).sum())
    not_x3 = ci_tw[~((ci_tw.coupling == "1:1.0") & (ci_tw.subset.apply(tuple) == ("x[t-3]", "xhat")))]
    lines.append(f"iid twins on the same subsets ({len(ci_tw)} chain-induced (run, subset) pairs with a twin): peak "
                 f"{ci_tw.iid_peak_ri.min():.5f} to {ci_tw.iid_peak_ri.max():.5f} nats ({iid_at0} at step 0); from step "
                 f"10 on <= {ci_tw.iid_peak_from_step10.max():.5f}; <= {not_x3.iid_peak_ri.max():.5f} outside 1:1.0 "
                 f"{{x[t-3], xhat}}, whose step-0 offsets are {ci_tw[~ci_tw.index.isin(not_x3.index)].iid_ri_at0.min():.4f}"
                 f"-{ci_tw[~ci_tw.index.isin(not_x3.index)].iid_ri_at0.max():.4f}")
    r1 = chain_induced[chain_induced.coupling == "1,2,3:1.0"]
    r2 = chain_induced[chain_induced.coupling != "1,2,3:1.0"]
    ex = explain
    lines.append(f"subset x coupling pairs scanned: {ex['n_subset_coupling_pairs']}; analytic chain correlation non-zero "
                 f"for {ex['n_pairs_with_nonzero_analytic_rho']}; |simulated - analytic| <= "
                 f"{ex['max_abs_sim_minus_analytic']:.4f}; (run, subset) crossings on correlated subsets "
                 f"{ex['runs_over_10x_on_rho_nonzero_subsets']}, on uncorrelated ones "
                 f"{ex['runs_over_10x_on_rho_zero_subsets']}")
    lines.append(f"chain-minus-iid median peak (step 0 excluded): {ex['chain_excess_median_where_rho_nonzero']:.5f} "
                 f"nats on correlated subsets vs {ex['chain_excess_median_where_rho_zero']:.6f} on uncorrelated ones; "
                 f"Spearman(Bayes-marginal RI, chain median peak) over the {ex['n_pairs_with_nonzero_analytic_rho']} "
                 f"correlated pairs = {ex['spearman_bayes_marginal_ri_vs_chain_peak_median_where_rho_nonzero']['rho']:.2f}")
    lines.append(f"beta < 1 sets: Bayes-marginal RI <= {ex['max_ri_bayes_marginal_for_beta_below_1']:.4f}, model "
                 f"chain peaks <= {ex['max_chain_peak_for_beta_below_1']:.4f} (no chain-induced crossing)")
    s123 = agg_df[(agg_df.coupling == "1,2,3:1.0")]
    s123 = s123.iloc[0] if len(s123) else None
    items.append(dict(
        id="result_chain_transients",
        question="Which non-planted interactions do chain-trained one-coupling runs learn transiently (subsets, runs, "
                 "peak RI and step, rise and decay relative to the planted onset, iid twins), and do the chain's "
                 "statistics explain them?",
        definition_used=f"Transient = a non-planted stored subset whose RI reaches 10 x min_effect = {thr:g} nats in "
                        f"a chain run. Rise = its first activation event (detect_events, alpha 0.01; the event step is "
                        f"the first of the two confirming checkpoints) and its first checkpoint >= {thr:g}; decay = its "
                        f"first extinction event after the peak (RI < min_effect/2 on two consecutive checkpoints). "
                        f"Planted onset = Amendment 1 C onset (first non-left-censored (re)activation with the final "
                        f"sign). Timing differences in checkpoint indices of the 60-point log grid. iid twin = same "
                        f"coupling set and seed trained on iid contexts (same initial weights and probes; seeds 10 and "
                        f"11 of 1,2,3:1.0 have none). 'Initialisation-level' = peak within the first three checkpoints "
                        f"with >= 90 % of it already at step 0. Chain statistics from the actual training corpus "
                        f"(MarkovLanguage, same config, data_seed 0, {chain_stats['1:1.0']['chain']['n_examples']:,} "
                        f"windows): rho_A = E[s_t prod_(l in A) s_(t-l)]; analytic value tanh(beta)^|J| with "
                        f"e_u = s_u chi_L(u) i.i.d. of mean tanh(beta) and s_t chi_A = prod_(j in J) e_(t-j) (0 when no "
                        f"finite J exists); Bayes-optimal marginal predictor g_A(s_A) = E_chain[s_t | s_A], its Walsh "
                        f"coefficient and exact local RI of A + xhat under uniform probes (what a model predicting from "
                        f"the lags of A alone would show).",
        literal_result="; ".join(lines[1:1 + len(agg_df)]),
        literal_verdict="descriptive",
        key_numbers="\n".join(lines),
        interpretation=(
            "Every chain-induced transient sits on a subset A whose lags correlate with x_t in the chain: with one "
            "coupling L, e_t = s_t chi_L(t) is i.i.d. with mean tanh(beta), so products of shifted copies of the planted "
            "pattern give E[s_t chi_A] = tanh(beta)^k (e.g. {1,2,3}: s_t s_(t-4) = e_t e_(t-1), tanh^2(1) = 0.58; {1}: "
            "lag l gives tanh^l; {1,2,3,5}: {4,5,6} and {1,2,4}: {3,4,5} give tanh^2; {1,3}: {4,5} and {2,6} give tanh^3). "
            "Simulation of the training corpus matches these values, and iid contexts have none (|rho| < 0.007). Two "
            "regimes: (1) where the shortcut is simpler than the planted term ({1,2,3}:1.0, a single lag against a "
            "3-lag product) the model first learns almost the whole Bayes-optimal marginal predictor on x[t-4] "
            f"(peak / marginal RI {rg(s123.peak_over_bayes_marginal_min_med_max, '{:.2f}') if s123 is not None else 'n/a'}), "
            f"which peaks {-int(r1.peak_minus_onset_idx.max())} to {-int(r1.peak_minus_onset_idx.min())} checkpoints "
            f"before the planted onset in {int((r1.peak_minus_onset_idx < 0).sum())} of {len(r1)} runs and is unlearned "
            f"once the planted term explains it (extinct at steps {int(r1.extinction_step.min()):,} to "
            f"{int(r1.extinction_step.max()):,}); (2) where the shortcut is not simpler ({{1}}, {{1,3}}, {{1,2,4}}, "
            f"{{1,2,3,5}} at beta = 1) it grows alongside or after the planted term (early growth ratios dfA/dfL of the "
            f"order of rho_A/rho_L), reaches only {100 * r2.peak_over_bayes_marginal_ri.min():.0f} to "
            f"{100 * r2.peak_over_bayes_marginal_ri.max():.0f} % of the marginal-predictor RI ({r2.peak_ri.min():.4f} to "
            f"{r2.peak_ri.max():.4f} nats) and peaks after the planted onset ({int((r2.peak_minus_onset_idx > 0).sum())} "
            f"of {len(r2)}), before vanishing by step {int(r2.extinction_step.max()):,}. At beta = 0.5 and 0.8 the "
            f"correlations (tanh^2(0.5) = {math.tanh(0.5) ** 2:.2f}) give marginal RIs <= "
            f"{ex['max_ri_bayes_marginal_for_beta_below_1']:.3f} nats and no chain-induced transient reaches the bar. "
            f"The iid twins, with no chain correlation (|rho| <= {ex['max_abs_rho_iid_sim']:.4f}), stay far below the "
            f"bar on the same subsets (<= {ci_tw.iid_peak_ri.max():.4f} nats, the largest values being the shared "
            f"step-0 offsets; <= {ci_tw.iid_peak_from_step10.max():.4f} from step 10 on)."),
        caveats="The model's coefficient on A is a learning-dynamics quantity, so the Bayes-optimal marginal RI is an "
                "upper reference, not a prediction; the early-growth ratio dfA/dfL (stored per run) is only roughly "
                "tanh(beta)^(k-1) for the {1} set because lag 1 explains the others away while both grow. The 1,3:0.5 "
                "seed 7 and 1:1.0 seed 7 {x[t-5], xhat} crossings are initialisation-level (offset 0.0092 at step 0, "
                "same in the iid twin). Peak steps are limited by the 60-point log grid (adjacent checkpoints differ by "
                "factors up to 1.3).",
        amended_result="", amended_verdict="descriptive", deciding="literal"))

    # ---- result: delay
    dl = {r["coupling"]: r for r in delay}

    def dline(cp, key):
        x = dl[cp][key]
        rr = x["ratio_chain_over_iid"]
        rs = "n/a" if rr is None else f"{rr['ratio']:.2f} [{rr['ci95'][0]:.2f}, {rr['ci95'][1]:.2f}]"
        return (f"chain {x['chain_median']:g} ({x['chain_range'][0]:g}-{x['chain_range'][1]:g}) vs iid "
                f"{x['iid_median']:g} ({x['iid_range'][0]:g}-{x['iid_range'][1]:g}), ratio {rs}")
    no_tr = [c for c in SINGLE if not dl[c]["has_transients_over_10x"]]
    dlines = []
    for cp in SINGLE:
        tag = (f"transients in {dl[cp]['n_chain_runs_with_transients']} chain runs" if dl[cp]["has_transients_over_10x"]
               else "no chain-induced transient >= 0.01")
        dlines.append(f"{cp} ({tag}): C onset {dline(cp, 'onset_C')}; E25 onset {dline(cp, 'onset_E25')}")
    n_cens = sum(dl[cp]["onset_C"]["n_censored_chain"] + dl[cp]["onset_C"]["n_censored_iid"] +
                 dl[cp]["onset_E25"]["n_censored_chain"] + dl[cp]["onset_E25"]["n_censored_iid"] for cp in SINGLE)
    dlines.append(f"censored onsets (no onset within 30,000 steps), C and E25 together: {n_cens}")
    a123 = assoc.get("1,2,3:1.0")
    if a123:
        dlines.append(f"within chain 1,2,3:1.0 (n = {a123['n']}): Spearman(transient peak RI, planted onset) = "
                      f"{a123['spearman_peak_ri_vs_planted_onset'][0]:.2f} (p = {a123['spearman_peak_ri_vs_planted_onset'][1]:.2f})")
    items.append(dict(
        id="result_chain_delay",
        question="Does the chain-induced transient delay the planted term? Planted onset with chain vs iid contexts "
                 "for the same coupling set.",
        definition_used="Per coupling set, median planted onset over seeds per context and the ratio chain / iid of "
                        "the medians, with a 95 % bootstrap CI (2,000 resamples, seeds resampled within each context "
                        f"independently, seed {BOOT_SEED} + offsets). Onsets: (ii) Amendment 1 C (first (re)activation "
                        "not left-censored with the final Fourier sign; primary, as for questions 4 to 6); (i) the "
                        "registered first activation event (left-censored ones at step 0 included); (iii) the planted "
                        "Fourier coefficient reaching 25 % of its population value tanh(beta) (Amendment 1 E's form; "
                        "also 10 % and 50 % in the JSON). No run is censored.",
        literal_result="; ".join(dlines[:len(SINGLE)]),
        literal_verdict="descriptive",
        key_numbers="\n".join(dlines),
        interpretation=(
            "Only 1,2,3:1.0, the set whose transient is a strong first-order shortcut that peaks before the planted "
            f"onset, shows a clear delay: the planted onset roughly doubles with chain contexts and the seed ranges do "
            f"not overlap (C onset {dl['1,2,3:1.0']['onset_C']['chain_range'][0]:g}-{dl['1,2,3:1.0']['onset_C']['chain_range'][1]:g} "
            f"vs {dl['1,2,3:1.0']['onset_C']['iid_range'][0]:g}-{dl['1,2,3:1.0']['onset_C']['iid_range'][1]:g} steps, "
            "as in the Amendment 1 pilot). A mechanism consistent with "
            "this: once the marginal predictor rho s_(t-4) is fitted, the squared-loss signal left for the planted "
            f"term is E[(s_t - g) chi_L] / E[s_t chi_L] = {chain_stats['1,2,3:1.0']['chain']['subsets'][(4,)]['planted_signal_left_after_g']:.2f} "
            "(= 1 - tanh^2(1)) of its full value. Where the transient rises after the planted onset (1,2,3,5:1.0, "
            "1,2,4:1.0, 1,3:1.0) the RI onset is not delayed (C ratios "
            + ", ".join(f"{dl[c]['onset_C']['ratio_chain_over_iid']['ratio']:.2f}" for c in ("1,2,3,5:1.0", "1,2,4:1.0", "1,3:1.0"))
            + ", CIs include 1); the coefficient's 25 % crossing is at most one checkpoint later (E25 ratios "
            + ", ".join(f"{dl[c]['onset_E25']['ratio_chain_over_iid']['ratio']:.2f}" for c in ("1,2,3,5:1.0", "1,2,4:1.0", "1,3:1.0"))
            + f"). For 1:1.0 the coefficient-based onsets are about "
            f"{dl['1:1.0']['onset_E25']['ratio_chain_over_iid']['ratio']:.1f} x later with chain contexts (E25 CI "
            f"[{dl['1:1.0']['onset_E25']['ratio_chain_over_iid']['ci95'][0]:.2f}, "
            f"{dl['1:1.0']['onset_E25']['ratio_chain_over_iid']['ci95'][1]:.2f}]), but RI onsets there are at the "
            f"initialisation scale (<= {max(dl['1:1.0']['onset_C']['chain_range'][1], dl['1:1.0']['onset_C']['iid_range'][1]):g} "
            "steps). Sets without chain-induced transients show C and E25 ratios of "
            + f"{min(dl[c][k]['ratio_chain_over_iid']['ratio'] for c in no_tr for k in ('onset_C', 'onset_E25')):.2f} to "
            + f"{max(dl[c][k]['ratio_chain_over_iid']['ratio'] for c in no_tr for k in ('onset_C', 'onset_E25')):.2f}, "
            "all CIs including 1."),
        caveats="Chain and iid training differ in more than the transient (the whole context distribution changes), so "
                "the delay is attributed to the chain-induced shortcut by association across coupling sets and by the "
                "signal argument, not by an intervention. Within 1,2,3:1.0 chain runs, the size of the transient does "
                "not significantly predict the onset. Medians sit on a 60-point log grid (adjacent checkpoints differ "
                "by 1.14-1.3 x), so bootstrap CIs are coarse and can be degenerate (e.g. [1.29, 1.38]).",
        amended_result="", amended_verdict="descriptive", deciding="literal"))

    out = {
        "meta": {"title": "Sky v1 check 3 (Markov language), Amendment 1 B, chain-induced transients",
                 "preregistration": "docs/results/v1_preregistration.md",
                 "script": "scripts/analysis_v1/markov_instrument.py", "release": str(args.release),
                 "extracted": str(args.extracted), "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
                 "threshold_10x_min_effect": thr, "kl_bar": KL_BAR, "last_k": LAST_K,
                 "coupling_sets": SINGLE},
        "sanity": sanity,
        "check3a": check_a, "check3b": check_b, "check3c": check_c,
        "by_coupling_context": table,
        "transients_chain_runs": tr.to_dict("records"),
        "transients_by_coupling_subset": agg_df.to_dict("records"),
        "chain_statistics_all_subsets": st_df.to_dict("records"),
        "chain_statistics_explanatory_power": explain,
        "chain_statistics_planted": {cp: {ctx: {"rho_planted_sim": chain_stats[cp][ctx]["rho_planted"],
                                                "rho_planted_analytic": chain_stats[cp][ctx]["rho_planted_analytic"],
                                                "n_examples": chain_stats[cp][ctx]["n_examples"]}
                                          for ctx in CONTEXTS} for cp in SINGLE},
        "planted_onset_chain_vs_iid": delay,
        "within_chain_association": assoc,
        "per_run": df.to_dict("records"),
        "chart_data": {"transient_curves": chart, "per_run_bars": chart_bars},
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(out), indent=1, allow_nan=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
