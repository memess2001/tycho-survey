"""Independent audit of the relay confirmation study: every verdict recomputed from the per-run rows.

The confirmation study (``docs/results/relay_confirmation_preregistration.md``; runner ``relay_confirm.py``, frozen
analysis ``relay_confirm_analyze.py``) was run on the 88-core server and committed with its per-run rows in
``docs/results/v2_analysis/relay_confirm_results.json``. This script shares no code with the frozen analysis. From the
per-run rows alone it

* checks the design (76 cells, seeds and head counts, the cells that were topped up),
* re-derives every run's category from its stored measures and recomputes the registered comparisons, the secondary
  predictions, the instrument numbers of C2 to C4 and the width gate,
* scans for classification problems (runs near a threshold, relays that behave like a textbook circuit),
* and adds analyses that were not registered (labelled as such in the output): learning rate against batch size on
  the nine unchanged rep_min 12 cells, head count, the dependence of the outcome on the seed, the late mask seed by
  seed, the loss of each kind of circuit by width, and the post hoc cells re-read with one behavioural rule.

The saved weights are on the server, so the categories themselves are not recomputed from weights here; a sample of
runs was re-trained locally instead (``relay_confirm_audit_reruns.py``).

Usage, from the repository root (numpy only): ``python scripts/analysis_v2/relay_confirm_audit.py``. Writes
``docs/results/v2_analysis/relay_confirm_audit.json``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "results" / "v2_analysis"
MODES = {"T+S": ("T", "S"), "T+S+M": ("T", "S", "M"), "T": ("T",)}
PRIMARY = {"P1a": ("A1-2", "A1-6"), "P1b": ("A1-1", "A1-7"), "P1c": ("A2-1", "A1-6"), "P1d": ("A2-4", "A2-2"),
           "P2a": ("A3-1", "A1-6"), "P2b": ("A1-1", "A3-2"), "P2c": ("A3-3", "A1-9"), "P2d": ("A3-5", "A1-6"),
           "P2e": ("A1-1", "A3-6"), "P3a": ("A4-1", "A1-6"), "P3b": ("A4-2", "A1-6"), "P3c": ("A1-1", "A4-3"),
           "P3d": ("A5-1", "A5-2")}
CLAIMS = {"C1": ("P1a", "P1b", "P1c", "P1d"), "C2": ("P2a", "P2b", "P2c", "P2d", "P2e"),
          "C3": ("P3a", "P3b", "P3c", "P3d")}
FIXED_HEADS = {"A6-1": 2, "A6-2": 4, "A6-3": 8, "A6-4": 2, "A6-5": 2}
GOOD_LOSS = 0.3                                          # "works like the textbook circuit": decoy >= 0.9, loss below this


def category(r: dict) -> str:
    """The five categories of the pre-registration (§1.3), from a row's stored measures."""
    loss = r["loss_rep"]
    if not math.isfinite(loss) or loss >= 2.0:
        return "U"
    if r["relay_delta"] != 0:
        return "R"
    if math.isfinite(r["deep_decoy_over_clean"]) and r["deep_decoy_over_clean"] >= 0.9:
        return "T"
    return "S" if r["source_l1_end"] > 0.5 else "M"


def mcnemar(xs: list[bool], ys: list[bool]) -> float:
    b = sum(x and not y for x, y in zip(xs, ys))
    c = sum(y and not x for x, y in zip(xs, ys))
    if b + c == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))


def tier(kx: int, ky: int, n: int) -> str:
    hi, lo, diff = (16, 8, 8) if n == 24 else (32, 16, 16)
    return "flip" if kx >= hi and ky <= lo else "shift" if kx - ky >= diff else "unconfirmed"


def exact_r_by_2(ks: list[int], ns: list[int]) -> float:
    """Freeman-Halton exact test of equal rates in independent groups."""
    total, denom = sum(ks), math.comb(sum(ns), sum(ks))
    prob = lambda a: math.prod(math.comb(n, k) for n, k in zip(ns, a)) / denom
    p0 = prob(ks)
    return sum(prob(a) for a in product(*[range(n + 1) for n in ns]) if sum(a) == total and prob(a) <= p0 * (1 + 1e-9))


def cochran_q(table: np.ndarray) -> tuple[float, float]:
    """Cochran's Q for the same units under k conditions (rows = units); p from chi-square with k − 1 = 2 d.f."""
    k = table.shape[1]
    col, row, total = table.sum(0), table.sum(1), table.sum()
    q = (k - 1) * (k * (col ** 2).sum() - total ** 2) / (k * total - (row ** 2).sum())
    return float(q), float(math.exp(-q / 2)) if k == 3 else float("nan")


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, t in zip(scores, labels) if t]
    neg = [s for s, t in zip(scores, labels) if not t]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def logit_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Maximum-likelihood logistic regression (Newton steps): coefficients, standard errors, log-likelihood."""
    b = np.zeros(x.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-x @ b))
        hess = (x * (p * (1 - p))[:, None]).T @ x + 1e-9 * np.eye(x.shape[1])
        step = np.linalg.solve(hess, x.T @ (y - p))
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-x @ b)), 1e-12, 1 - 1e-12)
    se = np.sqrt(np.diag(np.linalg.inv((x * (p * (1 - p))[:, None]).T @ x + 1e-9 * np.eye(x.shape[1]))))
    return b, se, float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def pilot_calibration(entries: list[tuple[int, int, float]], joint: float, draws: int = 100_000) -> dict:
    """How unusual the pilot's counts are if its runs were independent draws at the confirmation's rates. The joint
    probability of any particular set of counts is small, so it is compared with its own distribution; the second
    statistic is how many cell-by-head entries were unanimous (0 of 3 or 3 of 3)."""
    rng = np.random.default_rng(20261125)
    pmf = lambda k, n, p: math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    sim_joint, sim_unanimous = np.ones(draws), np.zeros(draws)
    for _, n, p in entries:
        k = rng.binomial(n, p, size=draws)
        sim_joint *= np.array([pmf(i, n, p) for i in range(n + 1)])[k]
        sim_unanimous += (k == 0) | (k == n)
    unanimous = sum(k in (0, n) for k, n, _ in entries)
    return {"joint_probability_of_the_pilot_counts": joint,
            "p_of_a_joint_probability_this_small": float((sim_joint <= joint).mean()),
            "unanimous_entries": [unanimous, len(entries)], "expected_unanimous_entries": float(sim_unanimous.mean()),
            "p_of_this_many_unanimous_entries": float((sim_unanimous >= unanimous).mean())}


def works(r: dict) -> bool:
    """One behavioural rule for every architecture: the two-token key is used and the loss is at the textbook level."""
    return bool(math.isfinite(r["loss_rep"]) and r["loss_rep"] < GOOD_LOSS and r["deep_decoy_over_clean"] >= 0.9)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=RESULTS / "relay_confirm_audit.json")
    args = ap.parse_args(argv)
    stored = json.loads((RESULTS / "relay_confirm_results.json").read_text())
    rows = stored["runs"]
    by: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        if r["seed"] in by[r["cell_id"]]:
            raise ValueError(f"duplicate run {r['name']}")
        by[r["cell_id"]][r["seed"]] = r
    main_seeds = set(range(1000, 1024))
    hit = lambda r, mode: r["cat"] in MODES[mode]
    out: dict = {"status": "independent audit of the committed rows; not a re-classification from weights"}

    # ------------------------------------------------------------------------------------------ design
    standard = {**{s: 2 for s in range(1000, 1012)}, **{s: 4 for s in range(1012, 1024)}}
    reserve = {**{s: 2 for s in range(1024, 1036)}, **{s: 4 for s in range(1036, 1048)}}
    design_problems = []
    for cid, d in by.items():
        want = {s: FIXED_HEADS[cid] for s in main_seeds} if cid in FIXED_HEADS else standard
        if {s: r["n_head"] for s, r in d.items() if s in main_seeds} != want:
            design_problems.append(f"{cid}: main seeds or head counts")
        top = {s: r["n_head"] for s, r in d.items() if s not in main_seeds}
        if top and top != reserve:
            design_problems.append(f"{cid}: reserve seeds or head counts")
    recategorised = [r["name"] for r in rows if category(r) != r["cat"]]
    out["design"] = {"cells": len(by), "runs": len(rows), "problems": design_problems,
                     "runs_whose_stored_category_differs_from_the_rule": recategorised,
                     "torch_versions": dict(Counter(r["torch_version"] for r in rows)),
                     "core_hours": round(sum(r["wall_s"] for r in rows) / 3600, 1)}

    # ------------------------------------------------------------------------- registered comparisons
    def compare(x: str, y: str, mode: str, main_only: bool = False) -> dict:
        common = sorted(s for s in by[x] if s in by[y] and (not main_only or s in main_seeds))
        xs, ys = [hit(by[x][s], mode) for s in common], [hit(by[y][s], mode) for s in common]
        return {"n": len(common), "x": sum(xs), "y": sum(ys), "mcnemar_p": mcnemar(xs, ys),
                "tier": tier(sum(xs), sum(ys), len(common)) if len(common) in (24, 48) else None}

    stored_cmp = {k: stored[c][k] for c, ks in CLAIMS.items() for k in ks}
    comparisons, disagreements = {}, []
    for k, (x, y) in PRIMARY.items():
        mine = {mode: compare(x, y, mode) for mode in MODES}
        first = compare(x, y, "T+S", main_only=True)
        s = stored_cmp[k]["by_mode"]["source_strict"]
        if (stored_cmp[k]["n"], s["kx"], s["ky"], s["verdict"]) != (mine["T+S"]["n"], mine["T+S"]["x"], mine["T+S"]["y"],
                                                                 mine["T+S"]["tier"]) or abs(s["mcnemar_p"] - mine["T+S"]["mcnemar_p"]) > 1e-6:
            disagreements.append(k)
        comparisons[k] = {"x": x, "y": y, **mine, "at_24_main_seeds": first}
    order = {"unconfirmed": 0, "shift": 1, "flip": 2}
    claims = {c: min((comparisons[k]["T+S"]["tier"] for k in ks), key=order.get) for c, ks in CLAIMS.items()}
    needs_top_up = sorted({c for k, (x, y) in PRIMARY.items() if comparisons[k]["at_24_main_seeds"]["tier"] != "flip"
                           for c in (x, y)})
    out["registered"] = {
        "comparisons": comparisons, "claims": claims, "stored_claims": stored["claim_verdicts"],
        "comparisons_that_differ_from_the_stored_analysis": disagreements,
        "cells_the_rule_tops_up": needs_top_up, "cells_topped_up": sorted(c for c in by if len(by[c]) == 48)}

    count = lambda cid, mode: sum(hit(r, mode) for s, r in by[cid].items() if s in main_seeds)
    pool = [r for r in rows if r["seed"] in main_seeds and r["group"] in ("A1", "A3", "A4")]
    decided = [r for r in pool if r["l1_first_85"] is not None]
    textbook = [r for r in pool if r["cat"] == "T"]
    ordered = [r for r in textbook if r["source_step"] and r["second_key_step"] and r["second_key_step"] > r["source_step"]]
    equal_sigma = ("A1-3", "A1-4", "A1-5")
    seeds24 = sorted(main_seeds)
    secondary = {
        "S1a": {c: count(c, "T+S") for c in ("A1-8", "A1-9")},
        "S1b": {mode: {"counts": [count(c, mode) for c in equal_sigma],
                       "exact_p_as_independent_groups": exact_r_by_2([count(c, mode) for c in equal_sigma], [24] * 3),
                       "cochran_q_and_p_on_the_same_seeds": cochran_q(np.array(
                           [[hit(by[c][s], mode) for c in equal_sigma] for s in seeds24], float))} for mode in MODES},
        "S1c_A2-6": count("A2-6", "T+S"),
        "S1d_A2-8_minus_A1-6": {mode: count("A2-8", mode) - count("A1-6", mode) for mode in MODES},
        "S2a_A3-4": dict(Counter(r["cat"] for r in by["A3-4"].values())),
        "S3a_A4-4": count("A4-4", "T+S"), "S3b": {"A5-3": count("A5-3", "T+S"), "A2-3": count("A2-3", "T+S")},
        "S3c": {"runs": len(pool), "decided": len(decided),
                **{mode: sum(r["l1_first_85"] == hit(r, mode) for r in decided) for mode in MODES}},
        "P4a": compare("A6-3", "A6-1", "T", True), "P4b": compare("A6-4", "A6-1", "T", True),
        "P4c": {"textbook": len(textbook), "second_key_after_the_source": len(ordered),
                "median_ratio": float(np.median([r["second_key_step"] / r["source_step"] for r in ordered]))},
        "S4a_A6-5": dict(Counter(r["cat"] for r in by["A6-5"].values())),
        "decomposition_by_heads": {c: {f"h{nh}": dict(Counter(r["cat"] for r in by[c].values() if r["n_head"] == nh))
                                       for nh in (2, 4)} for c in ("A6-6", "A6-7", "A6-8", "A1-1")}}
    r1 = {}
    for lo, hi in (("A3-5", "A1-6"), ("A1-1", "A3-6")):
        common = sorted(set(by[lo]) & set(by[hi]))
        ratios = [by[lo][s]["r1_aligned"] / by[hi][s]["r1_aligned"] for s in common]
        r1[f"{lo}_below_{hi}"] = {"pairs": len(common), "in_the_predicted_direction": sum(x < 1 for x in ratios),
                                  "median_ratio": float(np.median(ratios))}
    formed = [r for r in rows if r["seed"] in main_seeds and r["group"] in ("A1", "A3") and r["cat"] != "U"]
    relay = lambda rs: [r["cat"] == "R" for r in rs]
    a1_only = [r for r in formed if r["group"] == "A1"]
    base = [r for r in by["A1-6"].values() if r["cat"] != "U"]
    secondary["R_1"] = {
        "pairs": r1,
        "auc_for_relay": {"A1_and_A3_pooled": auc([r["r1_aligned"] for r in formed], relay(formed)),
                          "A1_and_A3_pooled_at_9600_sequences": auc([r["r1_at_n9600"] for r in formed], relay(formed)),
                          "A1_only_no_factor_no_speed_change": auc([r["r1_aligned"] for r in a1_only], relay(a1_only)),
                          "inside_the_baseline_cell": auc([r["r1_aligned"] for r in base], relay(base)),
                          **{f"inside_the_baseline_cell_h{nh}": auc([r["r1_aligned"] for r in base if r["n_head"] == nh],
                                                                   relay([r for r in base if r["n_head"] == nh]))
                             for nh in (2, 4)}}}
    d256 = {t: Counter(r["cat"] for r in by[f"C-d256-{t}"].values()) for t in ("low", "mid", "high", "mid-c05", "high-c05")}
    stable = [t for t in ("low", "mid", "high") if d256[t]["U"] <= 6]
    cut = {t: d256[t]["R"] - d256[f"{t}-c05"]["R"] if d256[t]["U"] <= 6 and d256[f"{t}-c05"]["U"] <= 6 else 0
           for t in ("mid", "high")}
    max_r = max(d256[t]["R"] for t in stable)
    gate = "GO" if max_r >= 12 and max(cut.values()) >= 8 else (
        "NO-GO" if max_r <= 4 and max(d256[t]["M"] for t in stable) <= 4 else "GRAY")
    secondary["gate"] = {"mine": gate, "stored": stored["Suite_C"]["gate_d256"]["decision"], "stable_base_cells": stable,
                         "largest_relay_count": max_r, "relays_removed_by_the_factor": cut}
    out["registered"]["secondary"] = secondary

    # -------------------------------------------------------------------------- classification scans
    by_cat = {c: [r for r in rows if r["cat"] == c] for c in "TSMRU"}
    quant = lambda v, qs: [round(float(x), 3) for x in np.nanpercentile(v, qs)]
    good_relays = [r for r in by_cat["R"] if r["deep_decoy_over_clean"] >= 0.9]
    near = [r for r in rows if abs(r["loss_rep"] - 2.0) < 0.2
            or (r["relay_delta"] == 0 and abs(r["deep_decoy_over_clean"] - 0.9) < 0.05)
            or (r["cat"] in "SM" and abs(r["source_l1_end"] - 0.5) < 0.05)]
    out["classification"] = {
        "counts": {c: len(v) for c, v in by_cat.items()},
        "loss_5_50_95": {c: quant([r["loss_rep"] for r in v], [5, 50, 95]) for c, v in by_cat.items()},
        "decoy_ratio_5_50_95": {c: quant([r["deep_decoy_over_clean"] for r in v], [5, 50, 95]) for c, v in by_cat.items()},
        "relays_with_decoy_ratio_at_least_0.9": {"n": len(good_relays), "by_cell": dict(Counter(r["cell_id"] for r in good_relays)),
                                                 "loss_median": float(np.median([r["loss_rep"] for r in good_relays]))},
        "relay_source_attention_max": max(r["source_l1_end"] for r in by_cat["R"]),
        "relays_with_another_layer_1_head_on_the_source": sum(0 in r["l1_modal_deltas"] for r in by_cat["R"]),
        "relay_offsets_by_rep_min": {rm: dict(sorted(Counter(r["relay_delta"] for r in by_cat["R"] if r["rep_min"] == rm
                                                             and r["suite"] == "A").items())) for rm in (4, 8, 12, 16, 20, 24)},
        "runs_near_a_threshold": {"n": len(near), "in_the_three_reference_cells": [r["name"] for r in near if r["cell_id"]
                                                                                   in ("A1-1", "A1-2", "A1-6")]}}

    # ------------------------------------------------------------------- not registered: structure
    unchanged = [r for c in ("A1-1", "A1-2", "A1-3", "A1-4", "A1-5", "A1-6", "A1-7", "A1-8", "A1-9") for r in by[c].values()]
    table = {f"{lr:g}": {str(b): [sum(hit(r, "T+S") for r in unchanged if r["lr"] == lr and r["batch_size"] == b),
                                  sum(r["lr"] == lr and r["batch_size"] == b for r in unchanged)]
                         for b in sorted({r["batch_size"] for r in unchanged if r["lr"] == lr})}
             for lr in sorted({r["lr"] for r in unchanged})}
    fits = {}
    for label, keep in (("all", lambda r: True), ("2 heads", lambda r: r["n_head"] == 2), ("4 heads", lambda r: r["n_head"] == 4)):
        rs = [r for r in unchanged if keep(r)]
        y = np.array([hit(r, "T+S") for r in rs], float)
        lr2, b2 = np.array([math.log2(r["lr"] / 1e-3) for r in rs]), np.array([math.log2(r["batch_size"] / 32) for r in rs])
        head = [np.array([r["n_head"] == 4 for r in rs], float)] if label == "all" else []
        cols = lambda *xs: np.column_stack([np.ones(len(rs)), *head, *xs])
        coef, se, ll = logit_fit(cols(lr2, b2), y)
        k = 1 + len(head)
        chi1 = lambda other: math.erfc(math.sqrt(max(ll - other, 0.0)))        # likelihood-ratio test, 1 d.f.
        fits[label] = {"runs": len(rs), "per_doubling_of_lr": [float(coef[k]), float(se[k])],
                       "per_doubling_of_batch": [float(coef[k + 1]), float(se[k + 1])],
                       "batch_exponent_relative_to_lr": float(-coef[k + 1] / coef[k]),
                       "p_against_the_free_fit": {"lr_over_sqrt_batch": chi1(logit_fit(cols(lr2 - 0.5 * b2), y)[2]),
                                                  "lr_alone": chi1(logit_fit(cols(lr2), y)[2]),
                                                  "lr_over_batch": chi1(logit_fit(cols(lr2 - b2), y)[2])}}
    heads = {}
    for cid, d in sorted(by.items()):
        rs = list(d.values())
        if cid[0] == "A" and {r["n_head"] for r in rs} == {2, 4} and 0.1 <= sum(hit(r, "T+S") for r in rs) / len(rs) <= 0.9:
            heads[cid] = {f"h{nh}": [sum(hit(r, "T+S") for r in rs if r["n_head"] == nh), sum(r["n_head"] == nh for r in rs)]
                          for nh in (2, 4)}
    more2 = sum(v["h2"][0] / v["h2"][1] > v["h4"][0] / v["h4"][1] for v in heads.values())
    # the seed: cells that share the baseline's early trajectory are tied to it by construction, so the test uses
    # cells whose training differs from the first step, and separately those that differ in the batch size only
    families = {
        "same_early_trajectory_as_the_baseline": ["A1-6", "B1-W-mid", "B3-late", "A5-2"],
        "different_from_the_first_step": ["A1-6", "A1-2", "A1-3", "A1-1", "A1-4", "A1-8", "A1-5", "A1-7", "B1-U-mid",
                                          "A2-8", "A2-10", "A2-11", "A2-4", "A2-6", "A2-13"],
        "rep_min_12_other_batch_sizes_only": ["A1-6", "A1-2", "A1-3", "A1-4", "A1-8", "A1-5", "A1-7"]}
    rng = np.random.default_rng(20261125)
    seed_effect = {}
    for family, cells in families.items():
        seed_effect[family] = {}
        for nh, seeds in ((2, range(1000, 1012)), (4, range(1012, 1024))):
            m = np.array([[hit(by[c][s], "T+S") for c in cells] for s in seeds], float)
            mixed = (m.mean(0) > 0.1) & (m.mean(0) < 0.9)
            m, rate = m[:, mixed], m.mean(0)[mixed]
            sims = np.array([(rng.random(m.shape) < rate).sum(1).var(ddof=1) for _ in range(20_000)])
            seed_effect[family][f"h{nh}"] = {
                "cells_with_a_mixed_outcome": int(mixed.sum()), "per_seed_totals": [int(x) for x in m.sum(1)],
                "variance": float(m.sum(1).var(ddof=1)), "expected_if_the_seed_did_not_matter": float((rate * (1 - rate)).sum()),
                "p_by_simulation": float((sims >= m.sum(1).var(ddof=1)).mean())}
    # the exploratory pilot (seeds 300-302) against the confirmation's rates, cell by cell and head count by head count
    pilot_runs = json.loads((RESULTS / "relay_period_pilot.json").read_text())["runs"]
    followup = json.loads((RESULTS / "relay_followup_pilot.json").read_text())["runs"]

    def pilot_source(r: dict) -> bool:
        return bool(r["formed"] and r["relay_delta"] == 0 and (r["textbook"] or r["source_l1_at_the_end"] > 0.5))

    pilot_cells = {"A1-1": pilot_runs["lr5e-4_natural"], "A1-2": pilot_runs["b128_natural"], "A1-3": pilot_runs["b64_natural"],
                   "A1-7": pilot_runs["lr2e-3_b128_natural"], "A1-8": pilot_runs["lr5e-4_b8_natural"]}
    pilot_table, joint, entries = {}, 1.0, []
    for cid in ("A1-6", *pilot_cells):
        pilot_table[cid] = {}
        for nh in (2, 4):
            if cid == "A1-6":                            # the baseline's six runs of the earlier pilots: all relays
                k = sum(r["rep_min"] == 12 and r["n_head"] == nh and r["steps"] == 12_000 and r["relay_delta"] == 0
                        and r["intact"]["loss_rep"] < 2 for r in followup)
                n = sum(r["rep_min"] == 12 and r["n_head"] == nh and r["steps"] == 12_000 for r in followup)
            else:
                k = sum(pilot_source(r) for r in pilot_cells[cid] if r["n_head"] == nh)
                n = sum(r["n_head"] == nh for r in pilot_cells[cid])
            conf = [r for r in by[cid].values() if r["n_head"] == nh]
            rate = (sum(hit(r, "T+S") for r in conf) + 0.5) / (len(conf) + 1)
            prob = math.comb(n, k) * rate ** k * (1 - rate) ** (n - k)
            joint *= prob
            entries.append((k, n, rate))
            pilot_table[cid][f"h{nh}"] = {"pilot_source_of_n": [k, n], "confirmation_source_of_n": [sum(hit(r, "T+S") for r in conf), len(conf)],
                                          "probability_of_the_pilot_count_at_the_confirmation_rate": prob}
    wd = Counter((hit(by["A1-6"][s], "T+S"), hit(by["B1-W-mid"][s], "T+S")) for s in main_seeds)
    late = {f"h{nh}": {f"{a}->{b}": n for (a, b), n in sorted(Counter(
        (by["A1-6"][s]["cat"], by["A5-2"][s]["cat"]) for s in by["A5-2"] if by["A1-6"][s]["n_head"] == nh).items())}
        for nh in (2, 4)}
    width = {}
    for cid in ("C-d32-low", "C-d32-mid", "A1-2", "A1-6", "C-d128-low", "C-d128-mid", "C-d256-low", "C-d256-mid"):
        rs = [r for s, r in by[cid].items() if s in main_seeds]
        width[cid] = {"d_model": rs[0]["d_model"], "works_like_textbook": sum(works(r) for r in rs),
                      **{c: {"n": sum(r["cat"] == c for r in rs), "loss_median": float(np.median([r["loss_rep"] for r in rs if r["cat"] == c]))}
                         for c in "TR" if any(r["cat"] == c for r in rs)}}
    out["not_registered"] = {
        "learning_rate_against_batch": {"source_of_n_by_lr_and_batch": table, "logistic_fits": fits},
        "heads": {"cells": heads, "cells_where_2_heads_give_more_source": more2, "cells_where_they_give_fewer": len(heads) - more2},
        "seed": {"families": families, **seed_effect,
                 "baseline_against_weight_decay_0.1_same_seed": {"both": wd[(True, True)], "neither": wd[(False, False)],
                                                                 "differ": wd[(True, False)] + wd[(False, True)]}},
        "pilot_against_confirmation": {"cells": pilot_table, **pilot_calibration(entries, joint)},
        "late_mask_against_baseline": late, "by_width": width}

    # --------------------------------------------------------- post hoc cells with one behavioural rule
    post = {}
    for name in ("relay_mlp_diag", "relay_mlp_res_opt", "relay_stress_rope", "relay_stress_depth"):
        cells = defaultdict(list)
        for r in json.loads((RESULTS / f"{name}.json").read_text())["runs"]:
            cells[r["cell_id"]].append(r)
        post[name] = {cid: {"as_committed": dict(Counter(r["cat"] for r in rs)),
                            "works_like_textbook": {"all": sum(works(r) for r in rs),
                                                    **{f"h{nh}": sum(works(r) for r in rs if r["n_head"] == nh) for nh in (2, 4)}},
                            "T_with_loss_at_least_0.3": sum(r["cat"] == "T" and r["loss_rep"] >= GOOD_LOSS for r in rs),
                            "seeds": [min(r["seed"] for r in rs), max(r["seed"] for r in rs)]} for cid, rs in cells.items()}
    out["post_hoc_cells"] = post
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------------------------------ print
    d = out["design"]
    print(f"design: {d['cells']} cells, {d['runs']} runs, problems {d['problems']}; categories that differ from the rule: "
          f"{len(d['runs_whose_stored_category_differs_from_the_rule'])}; {d['core_hours']} core hours")
    reg = out["registered"]
    print(f"registered comparisons that differ from the stored analysis: {reg['comparisons_that_differ_from_the_stored_analysis']}; "
          f"claims {reg['claims']} (stored {reg['stored_claims']}); top-up as the rule says: "
          f"{reg['cells_the_rule_tops_up'] == reg['cells_topped_up']}; gate {secondary['gate']['mine']}")
    for k, c in comparisons.items():
        print(f"  {k} {c['x']:5s} vs {c['y']:5s} n={c['T+S']['n']}: " + " | ".join(
            f"{mode} {c[mode]['x']} vs {c[mode]['y']} {c[mode]['tier']}" for mode in MODES) + f" | p {c['T+S']['mcnemar_p']:.2g}")
    s1b = secondary["S1b"]
    print("S1b:", {m: (v["counts"], round(v["exact_p_as_independent_groups"], 5), round(v["cochran_q_and_p_on_the_same_seeds"][1], 5))
                   for m, v in s1b.items()})
    print("lr against batch (source of n):", table)
    for label, f in fits.items():
        print(f"  {label}: lr {f['per_doubling_of_lr'][0]:+.2f} ± {f['per_doubling_of_lr'][1]:.2f}, batch {f['per_doubling_of_batch'][0]:+.2f} ± "
              f"{f['per_doubling_of_batch'][1]:.2f}, exponent {f['batch_exponent_relative_to_lr']:.2f}, p against free: "
              + ", ".join(f"{k} {v:.2g}" for k, v in f["p_against_the_free_fit"].items()))
    print(f"heads: 2 heads give more source in {more2} of {len(heads)} mixed cells")
    for family, v in seed_effect.items():
        print(f"  seed, {family}: " + "; ".join(f"{h}: {w['cells_with_a_mixed_outcome']} cells, variance {w['variance']:.2f} against "
                                                 f"{w['expected_if_the_seed_did_not_matter']:.2f}, p {w['p_by_simulation']:.4f}" for h, w in v.items()))
    print("pilot against confirmation:", {c: {h: (w["pilot_source_of_n"], w["confirmation_source_of_n"],
                                                  round(w["probability_of_the_pilot_count_at_the_confirmation_rate"], 4)) for h, w in v.items()}
                                          for c, v in pilot_table.items()},
          {k: v for k, v in out["not_registered"]["pilot_against_confirmation"].items() if k != "cells"})
    print("late mask:", late)
    print("by width:", width)
    cl = out["classification"]
    print(f"classification: {cl['counts']}; relays with decoy >= 0.9: {cl['relays_with_decoy_ratio_at_least_0.9']}; near a threshold: {cl['runs_near_a_threshold']}")
    for name, cells in post.items():
        for cid, v in cells.items():
            print(f"  {name} {cid:28s} {v['as_committed']} | works {v['works_like_textbook']} | T with high loss {v['T_with_loss_at_least_0.3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
