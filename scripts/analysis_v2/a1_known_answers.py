"""Sky v2 A1: constructed known-answer cases for the θ and η views (instrument checks, no training).

Teachers and measures: ``tycho/calibration/correlated.py``. Four families:

1. **Attribution map** (3 inputs, uniform probes). log-odds / 2 = s (χ0 + λ χ1 + λ χ2) + b χ012, with saturation s
   and lower-order strength λ as crossed knobs. b takes three values: 0 (θ top term absent), b* (η top coefficient
   exactly 0, by root finding) and 0.3 (both present). Exact values are reported without nuisance inputs, where
   the constructions are clean. Exact values and estimates are reported with four nuisance inputs (scale 1.0),
   where the estimates are noisy as in a network.
2. **Probability-scale products** (3 inputs, uniform): soft AND (ε = 0.05, 0.2) and soft XOR (m = c χ012, c = 0.3,
   0.9). Exact values for every subset of {0, 1, 2}.
3. **The v1 cancellation profile**: the exact cell means of the DR1 case where the views diverge (A3: object
   898791…, step 2206), as uniform-cell Walsh coefficients. Exact values only.
4. **Correlated inputs** (6 table inputs, four nuisance inputs at scale 1.0). Probes: uniform, Markov chain
   ρ = 0.5, 0.8, 0.95, equicorrelated ρ = 0.3, 0.6. Four teachers that use inputs 0 and 2 only. For all 21 subsets
   of one or two inputs: the exact value of every measure, and detections in replicate estimates. Spurious sets
   contain an input the teacher does not use: there ``ri_cond`` and ``f_u`` are exactly 0, whatever the
   correlation.

Detection (estimates): BH over the subsets of the family at α = 0.01, and corrected value ≥ min_effect. Marginal RI
and the η measures use v1's calibrated min_effect at n. Conditional RI uses E8's recommended value at (n, table
size). Exact presence uses the same min_effect.

Run from the repository root after E8: ``PYTHONPATH=. python scripts/analysis_v2/a1_known_answers.py``. Writes
``docs/results/v2_analysis/a1_known_answers.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import tycho
from tycho.calibration.correlated import (MEASURES, Teacher, detect, estimate_measures, exact_measures,
                                          logit_teacher, probe_pmf, rng_for, soft_and_teacher, uniform_pmf,
                                          walsh_teacher)
from tycho.survey.followup import calibrated_min_effect_at

REPO = Path(__file__).resolve().parents[2]
assert Path(tycho.__file__).resolve().is_relative_to(REPO), f"wrong tycho package: {tycho.__file__}"
N = 4096
NUISANCE_W = {0: 0.8, 1: -0.6, 2: 0.5, 3: -0.4}  # weights of the four nuisance inputs (logit / 2)
ALPHA = 0.01


def with_nuisance(d_table: int, w: dict, scale: float = 1.0) -> dict:
    out = dict(w)
    for k, v in NUISANCE_W.items():
        out[(d_table + k,)] = scale * v
    return out


def min_effects(e8_path: Path, n: int, d_table: int) -> dict[str, float]:
    marg = calibrated_min_effect_at(REPO / "reports" / "calibration" / "calibration.json", n) or 0.001
    e8 = json.loads(e8_path.read_text())
    rows = [r for r in e8["summary"] if r["n"] == n]
    row = min(rows, key=lambda r: (abs(r["d_table"] - d_table), -r["d_table"]))
    return {"ri_marg": marg, "eta_u": marg, "eta_d": marg, "ri_cond": row["recommended_min_effect_conditional"],
            "_cond_from_d_table": row["d_table"]}


def classify(theta: float, eta: float, me: float) -> str:
    t, e = theta >= me, eta >= me
    return "both" if t and e else "theta_only" if t else "eta_only" if e else "neither"


# ------------------------------------------------------------------------------------------------ family 1
def attribution_teacher(s: float, lam: float, b: float, nuisance: bool) -> Teacher:
    w = {(0,): s, (1,): s * lam, (2,): s * lam, (0, 1, 2): b}
    return logit_teacher(3 + (4 if nuisance else 0), with_nuisance(3, w) if nuisance else w, f"s={s},lam={lam},b={b}")


def eta_null_b(s: float, lam: float) -> float:
    f = lambda b: exact_measures(uniform_pmf(3), attribution_teacher(s, lam, b, False), [(0, 1, 2)],  # noqa: E731
                                 with_cond=False)[0]["f_u"]
    return brentq(f, -10.0, 10.0, xtol=1e-12)


def family1(me: dict, reps: int, seed: int) -> list[dict]:
    rows = []
    for s, lam in itertools.product((0.25, 0.5, 1.0, 2.0, 3.0), (0.3, 0.6, 1.0)):
        bstar = eta_null_b(s, lam)
        for label, b in (("theta_null", 0.0), ("eta_null", bstar), ("both", 0.3)):
            clean = exact_measures(uniform_pmf(3), attribution_teacher(s, lam, b, False), [(0, 1, 2)])[0]
            t = attribution_teacher(s, lam, b, True)
            ex = exact_measures(uniform_pmf(3), t, [(0, 1, 2)], d_table=3)[0]
            rng = rng_for(seed, f"A1.1|{s}|{lam}|{label}")
            est_cls, vals = [], {k: [] for k in ("ri_marg", "eta_u")}
            for _ in range(reps):
                e = estimate_measures(uniform_pmf(3), t, [(0, 1, 2)], N, rng, d_table=3, with_cond=False)
                det = detect(e, N, me, alpha=ALPHA)
                est_cls.append(classify(1.0 if det["ri_marg"][0] else 0.0, 1.0 if det["eta_u"][0] else 0.0, 0.5))
                for k in vals:
                    vals[k].append(e[0][k])
            exact_cls = classify(ex["ri_marg"], ex["eta_u"], me["ri_marg"])
            rows.append(dict(
                s=s, lam=lam, construction=label, b=b,
                clean=dict(ri=clean["ri_marg"], w_top=clean["w"], f_u=clean["f_u"], eta_u=clean["eta_u"],
                           cls=classify(clean["ri_marg"], clean["eta_u"], me["ri_marg"]),
                           saturation=float(np.abs(2 * attribution_teacher(s, lam, b, False).p1 - 1).max())),
                nuisance_exact=dict(ri=ex["ri_marg"], w_top=ex["w"], f_u=ex["f_u"], eta_u=ex["eta_u"], cls=exact_cls),
                estimates=dict(class_agreement=float(np.mean([c == exact_cls for c in est_cls])),
                               class_counts={c: est_cls.count(c) for c in sorted(set(est_cls))},
                               ri_mean=float(np.mean(vals["ri_marg"])), ri_sd=float(np.std(vals["ri_marg"])),
                               eta_mean=float(np.mean(vals["eta_u"])), eta_sd=float(np.std(vals["eta_u"])))))
    return rows


# ------------------------------------------------------------------------------------------------ family 2, 3
def family2(me: dict) -> list[dict]:
    subs = [A for r in (1, 2, 3) for A in itertools.combinations(range(3), r)]
    teachers = [soft_and_teacher(3, (0, 1, 2), 0.05), soft_and_teacher(3, (0, 1, 2), 0.2),
                walsh_teacher(3, {(0, 1, 2): 0.3}, "xor(c=0.3)"), walsh_teacher(3, {(0, 1, 2): 0.9}, "xor(c=0.9)")]
    rows = []
    for t in teachers:
        for e in exact_measures(uniform_pmf(3), t, subs):
            rows.append(dict(teacher=t.name, subset=e["subset"], ri=e["ri_marg"], w=e["w"], f_u=e["f_u"],
                             eta_u=e["eta_u"], cls=classify(e["ri_marg"], e["eta_u"], me["ri_marg"])))
    return rows


# Exact cell means of the DR1 case (object 898791…, step 2206, {x[t-1], x[t-2], x[t-3], xhat}) from the exact
# reconstruction of A3 (regenerated probes, fitted f(∅) = 0.0099), as Walsh coefficients over uniform cells. The
# stored (uncentred) coefficients differ from these. Rebuilt with f(∅) = 0 and uniform cells, they give the wrong
# RI 0.0057 (A3, item 3).
V1_PROFILE = {(): -0.009562, (0,): 0.756746, (1,): -0.179917, (2,): 0.191573, (0, 1): -0.005092, (0, 2): 0.011785,
              (1, 2): 0.00679, (0, 1, 2): 0.143216}


def family3(me: dict) -> list[dict]:
    t = walsh_teacher(3, V1_PROFILE, "v1-profile")
    subs = [A for r in (1, 2, 3) for A in itertools.combinations(range(3), r)]
    return [dict(subset=e["subset"], ri=e["ri_marg"], w=e["w"], f_u=e["f_u"], eta_u=e["eta_u"],
                 cls=classify(e["ri_marg"], e["eta_u"], me["ri_marg"])) for e in exact_measures(uniform_pmf(3), t, subs)]


# ------------------------------------------------------------------------------------------------ family 4
TEACHERS4 = {
    "copy0": {(0,): 1.1},
    "additive02": {(0,): 0.8, (2,): 0.8},
    "main0_pair02": {(0,): 0.6, (0, 2): 0.8},
    "pair02": {(0, 2): 1.0},
}
PROBES4 = [("uniform", 0.0), ("markov", 0.5), ("markov", 0.8), ("markov", 0.95), ("equicorr", 0.3), ("equicorr", 0.6)]
USED4 = {0, 2}


def family4(me: dict, reps: int, seed: int) -> list[dict]:
    d = 6
    subs = [A for r in (1, 2) for A in itertools.combinations(range(d), r)]
    spurious = np.array([not set(A) <= USED4 for A in subs])
    rows = []
    for (kind, rho), (tname, w) in itertools.product(PROBES4, TEACHERS4.items()):
        pmf = probe_pmf(kind, d, rho)
        t = logit_teacher(d + 4, with_nuisance(d, w), tname)
        ex = exact_measures(pmf, t, subs, d_table=d)
        exact_present = {m: np.array([e[m] >= me[m] for e in ex]) for m in MEASURES}
        rng = rng_for(seed, f"A1.4|{kind}|{rho}|{tname}")
        det_sum = {m: np.zeros(len(subs)) for m in MEASURES}
        for _ in range(reps):
            est = estimate_measures(pmf, t, subs, N, rng, d_table=d)
            for m, v in detect(est, N, me, alpha=ALPHA).items():
                det_sum[m] += v
        rate = {m: det_sum[m] / reps for m in MEASURES}
        rows.append(dict(
            probe=kind, rho=rho, teacher=tname,
            exact={m: {"spurious_present": int((exact_present[m] & spurious).sum()),
                       "nonspurious_present": int((exact_present[m] & ~spurious).sum()),
                       "max_spurious_value": float(max(e[m] for e, sp in zip(ex, spurious) if sp))}
                   for m in MEASURES},
            estimated={m: {
                # does the estimate reproduce the exact answer?
                "recall_where_exact_present": (float(rate[m][exact_present[m]].mean())
                                               if exact_present[m].any() else None),
                "detection_rate_where_exact_absent": (float(rate[m][~exact_present[m]].mean())
                                                      if (~exact_present[m]).any() else None),
                # does the measure report sets that contain an unused input?
                "mean_spurious_detections": float(rate[m][spurious].sum()),
                "spurious_detection_rate": float(rate[m][spurious].mean())}
                for m in MEASURES},
            exact_values={m: {",".join(map(str, A)): float(e[m]) for A, e in zip(subs, ex)} for m in MEASURES},
            detection_rates={m: {",".join(map(str, A)): float(r) for A, r in zip(subs, rate[m])} for m in MEASURES},
            n_spurious_sets=int(spurious.sum())))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e8", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "a0_e8_calibration.json")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "a1_known_answers.json")
    args = ap.parse_args(argv)
    logging.getLogger("tycho").setLevel(logging.ERROR)
    me3 = min_effects(args.e8, N, 3)
    me6 = min_effects(args.e8, N, 6)
    result = {"description": __doc__.split("\n\n")[0], "n": N, "alpha": ALPHA, "reps": args.reps,
              "min_effect": {"family1_3": me3, "family4": me6},
              "family1_attribution": family1(me3, args.reps, args.seed),
              "family2_products": family2(me3),
              "family3_v1_profile": family3(me3),
              "family4_correlated": family4(me6, args.reps, args.seed)}

    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return None if not math.isfinite(float(v)) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean(result), indent=1) + "\n", encoding="utf-8")
    print("min_effect", me3, me6)
    for r in result["family1_attribution"]:
        print("F1", r["s"], r["lam"], r["construction"], "clean", r["clean"]["cls"],
              "ri %.5f eta %.5f" % (r["clean"]["ri"], r["clean"]["eta_u"]), "| nuisance", r["nuisance_exact"]["cls"],
              "agree %.2f" % r["estimates"]["class_agreement"])
    def rnd(x):
        return None if x is None else round(x, 2)
    for r in result["family4_correlated"]:
        print("F4", r["probe"], r["rho"], r["teacher"],
              {m: (r["exact"][m]["spurious_present"], rnd(r["estimated"][m]["mean_spurious_detections"]),
                   rnd(r["estimated"][m]["recall_where_exact_present"]),
                   rnd(r["estimated"][m]["detection_rate_where_exact_absent"])) for m in MEASURES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
