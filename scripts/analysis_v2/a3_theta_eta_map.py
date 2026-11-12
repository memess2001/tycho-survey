"""A3 of the sky v2 design: where do the θ view (RI) and the η view (Walsh coefficient) disagree in DR1?

Exploratory analysis of stored data, not pre-registered. Its findings inform v2 Part A and are to be disclosed in
the v2 pre-registration. Design: ``docs/results/v2_design_draft.md`` §4, A3.

Every in-situ RI curve of a binary-input region (parity-k2/k3/k4, markov-v1; the probes are uniform) stores, for
each kept subset A ∪ {out} and checkpoint, the plug-in local RI (top-order θ of the subset's marginal) and the
empirical Walsh coefficient f(A) of the model's soft output. Under uniform probes,

    E[2 p1 − 1 | x_A] = Σ_{B ⊆ A} f(B) χ_B(x_A),

so the local RI of A ∪ {out} is a function of the coefficients f(B), B ⊆ A. Three quantities are compared:

* ``ri_plugin``: the stored plug-in RI (what the instrument measured);
* ``R(f)``: the single-mode value ln 2 − h((1 + |f(A)|)/2), the RI the subset would have if its top coefficient were
  the only one (the η view expressed in nats);
* ``rebuilt``: the local RI of the table rebuilt from all stored f(B), B ⊆ A, assuming exactly uniform probe cells and
  a mean output f(∅) = 0 (f(∅) is not stored). Only for subsets whose every sub-subset is stored. Both assumptions
  fail near saturation (the trained models' f(∅) is not 0, and the probe cells are a few per cent unbalanced), so
  the rebuild is only an approximation; ``a3_exact_reconstruction.py`` regenerates the probes and is exact.

A disagreement is classified at each checkpoint, with m = the curve's min_effect (0.001 nats here):

* η-only: R(f) ≥ 10 m while ri_plugin < m (a large top coefficient, no top-order log-linear term);
* θ-only: ri_plugin ≥ 10 m while R(f) < m (a top-order term with a small top coefficient).

With the lower-order coefficients set to 0 the rebuild reduces to R(f). The gap ri_plugin − R(f) therefore combines
the lower-order coefficients with the effects of f(∅) and of unbalanced probe cells; for pairs it consists of those
effects only.

Run from the repository root after extracting DR1 into work/dr1 (scripts/analysis_v1/README.md):
``PYTHONPATH=. python scripts/analysis_v2/a3_theta_eta_map.py``. Writes ``docs/results/v2_analysis/a3_theta_eta_map.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import tycho
from tycho.analysis.ricurve import RICurve
from tycho.ri.refined import local_ri_batch

REPO = Path(__file__).resolve().parents[2]
assert Path(tycho.__file__).resolve().is_relative_to(REPO), f"wrong tycho package: {tycho.__file__}"
REGIONS = ("parity-k2", "parity-k3", "parity-k4", "markov-v1")
LN2 = math.log(2.0)
STRONG = 10.0  # multiples of min_effect


def single_mode(f: np.ndarray) -> np.ndarray:
    """ln 2 − h((1 + |f|)/2) in nats (h the binary entropy)."""
    q = np.clip((1.0 + np.abs(f)) / 2.0, 1e-15, 1.0)
    r = 1.0 - q
    h = -(q * np.log(q) + np.where(r > 0, r * np.log(np.clip(r, 1e-300, None)), 0.0))
    return np.maximum(LN2 - h, 0.0)


def chi_matrix(m: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """(2^m, 2^m − 1) matrix of χ_B(x) for x in C order over {0,1}^m and B the nonempty subsets of range(m)."""
    Bs = [B for k in range(1, m + 1) for B in itertools.combinations(range(m), k)]
    xs = np.array(list(itertools.product((0, 1), repeat=m)), dtype=np.float64) * 2 - 1
    return np.stack([np.prod(xs[:, list(B)], axis=1) for B in Bs], axis=1), Bs


CHI = {m: chi_matrix(m) for m in range(1, 6)}


def rebuild(coefs: np.ndarray, m: int) -> np.ndarray:
    """Local RI of {x_A, out} for each row of ``coefs`` (N, 2^m − 1), uniform x_A, f(∅) = 0."""
    chi, _ = CHI[m]
    mean = coefs @ chi.T                                     # (N, 2^m)
    p1 = np.clip((1.0 + mean) / 2.0, 1e-12, 1 - 1e-12)
    tab = np.stack([1 - p1, p1], axis=-1) / 2 ** m           # (N, 2^m, 2): x axes flattened, out last
    tab = tab.reshape((len(coefs),) + (2,) * m + (2,))
    out = np.empty(len(coefs))
    for i in range(0, len(coefs), 20000):
        out[i:i + 20000] = local_ri_batch(tab[i:i + 20000])
    return out


def theta_eta_profile(c: RICurve, row: int, t: int) -> dict | None:
    """η and θ coefficients of the subset at checkpoint t, over every B ⊆ A (A = the subset's inputs).

    With uniform x_A and m(x) = E[2 p1 − 1 | x_A] = Σ_B f(B) χ_B(x) (f(∅) = 0), the η coefficients are f(B) and the
    θ coefficients of the terms that involve the output are the Walsh coefficients of atanh(m(x)) = logit(p1)/2.
    None if a sub-subset is not stored.
    """
    oi = c.names.index(c.meta["fourier_output"])
    s = c.subsets[row]
    A = [j for j in s if j != oi]
    m = len(A)
    chi, Bs = CHI[m]
    idx = [c.index_of([c.names[A[b]] for b in B] + [c.names[oi]]) for B in Bs]
    if any(i is None for i in idx):
        return None
    f = np.array([c.fourier[i, t] for i in idx])
    mean = np.clip(chi @ f, -1 + 1e-9, 1 - 1e-9)
    w = chi.T @ np.arctanh(mean) / 2 ** m
    lab = ["{" + ",".join(c.names[A[b]] for b in B) + "}" for B in Bs]
    max_abs = float(np.abs(chi @ f).max())
    return {"eta_f": {k: round(float(v), 4) for k, v in zip(lab, f)},
            # invalid where the approximate cell means leave [-1, 1]: atanh is then only the clip value
            "theta_w": ({k: round(float(v), 4) for k, v in zip(lab, w)} if max_abs < 1 else None),
            "max_abs_mean": round(max_abs, 4),
            "note": "approximate (uniform cells, f(∅) = 0); exact values in a3_exact_reconstruction.json"}


def relation(names_set: frozenset, gts: list[frozenset]) -> str:
    """Relation to the ground-truth sets, by priority: planted, inside a planted set, containing one, other. A subset
    that is inside one planted set and contains another (e.g. {x[t-1], x[t-2], xhat} with planted {x[t-1], xhat} and
    {x[t-1], x[t-2], x[t-3], xhat}) is labelled inside_planted."""
    if any(names_set == g for g in gts):
        return "planted"
    if any(names_set < g for g in gts):
        return "inside_planted"
    if any(names_set > g for g in gts):
        return "contains_planted"
    return "other"


def analyse_curve(c: RICurve) -> dict:
    out_name = c.meta.get("fourier_output")
    oi = c.names.index(out_name)
    me = float(c.meta["min_effect"])
    gts = [frozenset(g["subset"]) for g in (c.ground_truth or [])]
    index = {s: i for i, s in enumerate(c.subsets)}
    rows = []
    groups = defaultdict(list)  # m -> list of (row index, sub-coefficient row indices)
    for i, s in enumerate(c.subsets):
        if oi not in s or len(s) < 2:
            continue
        A = [j for j in s if j != oi]
        m = len(A)
        _, Bs = CHI[m]
        idx = []
        for B in Bs:
            key = tuple(sorted([A[b] for b in B] + [oi]))
            idx.append(index.get(key, -1))
        rec = dict(i=i, m=m, names=frozenset(c.names[j] for j in s), complete=all(k >= 0 for k in idx))
        rows.append(rec)
        if rec["complete"]:
            groups[m].append((len(rows) - 1, idx))
    T = len(c.steps)
    rebuilt, sat, low = {}, {}, {}
    for m, items in groups.items():
        coefs = np.stack([c.fourier[np.array(idx)].T for _, idx in items])     # (n, T, 2^m − 1)
        rb = rebuild(coefs.reshape(-1, coefs.shape[-1]), m).reshape(len(items), T)
        chi, _ = CHI[m]
        mx = np.abs(coefs @ chi.T).max(axis=-1)                                 # (n, T): max over cells of |m(x)|
        lo = np.sqrt((coefs[..., :-1] ** 2).sum(axis=-1))                        # proper nonempty B ⊂ A (last = A)
        for (r, _), v, a, b in zip(items, rb, mx, lo):
            rebuilt[r], sat[r], low[r] = v, a, b
    return dict(rows=rows, rebuilt=rebuilt, sat=sat, low=low, me=me, gts=gts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases" / "DR1")
    ap.add_argument("--extracted", type=Path, default=REPO / "work" / "dr1")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "a3_theta_eta_map.json")
    args = ap.parse_args(argv)

    o = pd.read_parquet(args.release / "objects.parquet")
    r = pd.read_parquet(args.release / "ri_runs.parquet")
    x = r[r.kind == "insitu"].merge(o[["object_id", "region", "task_cfg_json", "label"]], on="object_id")
    x = x[x.region.isin(REGIONS)].sort_values(["region", "object_id"])

    pts = []          # one row per (run, subset, checkpoint)
    cases = []        # exemplar records
    for row in x.itertuples():
        c = RICurve.load(args.extracted / row.ricurve_file)
        assert c.fourier is not None and c.meta.get("kind") == "insitu"
        res = analyse_curve(c)
        me = res["me"]
        T = len(c.steps)
        for k, rec in enumerate(res["rows"]):
            i = rec["i"]
            f = c.fourier[i]
            rf = single_mode(f)
            rb = res["rebuilt"].get(k)
            sa, lo = res["sat"].get(k), res["low"].get(k)
            rel = relation(rec["names"], res["gts"])
            for t in range(T):
                pts.append((row.region, row.object_id, i, rec["m"] + 1, rel, rec["complete"], t, int(c.steps[t]),
                            float(c.ri_plugin[i, t]), float(c.ri[i, t]), float(c.q[i, t]), float(f[t]), float(rf[t]),
                            float(rb[t]) if rb is not None else math.nan, me,
                            float(sa[t]) if sa is not None else math.nan, float(lo[t]) if lo is not None else math.nan))
            eta_only = (rf >= STRONG * me) & (c.ri_plugin[i] < me)
            theta_only = (c.ri_plugin[i] >= STRONG * me) & (rf < me)
            for kind, mask in (("eta_only", eta_only), ("theta_only", theta_only)):
                if mask.any():
                    t = int(np.argmax(np.where(mask, np.abs(f), -1))) if kind == "eta_only" else \
                        int(np.argmax(np.where(mask, c.ri_plugin[i], -1)))
                    A = [n for n in sorted(rec["names"]) if n != c.meta["fourier_output"]]
                    sub = {}
                    for kk in range(1, len(A)):
                        for B in itertools.combinations(A, kk):
                            j = c.index_of(list(B) + [c.meta["fourier_output"]])
                            if j is not None:
                                sub["{" + ",".join(B) + "}"] = round(float(c.fourier[j, t]), 4)
                    cases.append(dict(kind=kind, region=row.region, object_id=row.object_id,
                                      task_cfg=json.loads(row.task_cfg_json), subset=sorted(rec["names"]),
                                      relation=rel, n_checkpoints=int(mask.sum()), step=int(c.steps[t]),
                                      f=round(float(f[t]), 4), R_f=float(rf[t]), ri_plugin=float(c.ri_plugin[i, t]),
                                      rebuilt=(float(rb[t]) if rb is not None else None),
                                      lower_order_coefficients=sub))
    cols = ["region", "object_id", "row", "order", "relation", "complete", "t", "step", "ri_plugin", "ri", "q", "f",
            "R_f", "rebuilt", "min_effect", "saturation", "lower_energy"]
    d = pd.DataFrame(pts, columns=cols)
    me = d.min_effect
    d["eta_only"] = (d.R_f >= STRONG * me) & (d.ri_plugin < me)
    d["theta_only"] = (d.ri_plugin >= STRONG * me) & (d.R_f < me)
    d["gap"] = d.ri_plugin - d.R_f

    def uniq(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(["object_id", "row"])

    def per_region(df: pd.DataFrame) -> dict:
        runs = df.groupby("object_id")
        pairs = df.groupby(["object_id", "row"])
        out = {"runs": int(df.object_id.nunique()), "subset_rows": int(pairs.ngroups), "points": int(len(df))}
        for kind in ("eta_only", "theta_only"):
            pa = pairs[kind].any()
            out[kind] = {"points": int(df[kind].sum()), "subset_rows": int(pa.sum()),
                         "runs": int(runs[kind].any().sum()),
                         "by_relation": {str(k): int(v) for k, v in uniq(df[df[kind]]).groupby("relation").size().items()},
                         "by_order": {int(k): int(v) for k, v in uniq(df[df[kind]]).groupby("order").size().items()},
                         "at_final_checkpoint_subset_rows": int(df[df[kind] & (df.t == df.t.max())].groupby(
                             ["object_id", "row"]).ngroups)}
        comp = df[df.complete & np.isfinite(df.rebuilt)]
        diff = (comp.rebuilt - comp.ri_plugin).abs()
        out["rebuild_check"] = {
            "points": int(len(comp)), "share_of_points": float(len(comp) / max(len(df), 1)),
            "abs_diff_median": float(diff.median()) if len(comp) else None,
            "abs_diff_p99": float(diff.quantile(0.99)) if len(comp) else None,
            "abs_diff_max": float(diff.max()) if len(comp) else None,
            "within_0.001": float((diff <= 0.001).mean()) if len(comp) else None,
        }
        big = df[(df.ri_plugin >= STRONG * df.min_effect) | (df.R_f >= STRONG * df.min_effect)]
        out["gap_where_either_strong"] = {
            "points": int(len(big)),
            "median_abs_gap": float(big.gap.abs().median()) if len(big) else None,
            "share_abs_gap_over_half_of_larger": float((big.gap.abs() > 0.5 * np.maximum(big.ri_plugin, big.R_f)).mean())
            if len(big) else None,
        }
        return out

    summary = {reg: per_region(d[d.region == reg]) for reg in REGIONS}
    summary["all"] = per_region(d)
    # markov: split by coupling and context
    mk = d[d.region == "markov-v1"].merge(
        x[x.region == "markov-v1"][["object_id", "task_cfg_json"]].assign(
            couplings=lambda z: z.task_cfg_json.map(lambda s: json.loads(s)["couplings"]),
            context=lambda z: z.task_cfg_json.map(lambda s: json.loads(s)["train_context"]))[
            ["object_id", "couplings", "context"]], on="object_id")
    markov_cells = {}
    for (cp, ctx), g in mk.groupby(["couplings", "context"]):
        markov_cells[f"{cp}|{ctx}"] = {
            "runs": int(g.object_id.nunique()),
            "eta_only_runs": int(g.groupby("object_id").eta_only.any().sum()),
            "eta_only_subset_rows": int(g[g.eta_only].groupby(["object_id", "row"]).ngroups),
            "theta_only_runs": int(g.groupby("object_id").theta_only.any().sum()),
            "theta_only_subset_rows": int(g[g.theta_only].groupby(["object_id", "row"]).ngroups),
        }
    # threshold sweep: eta-only = R(f) >= k m and ri_plugin < m; theta-only = ri_plugin >= k m and R(f) < m
    sweep = {}
    for kk in (2.0, 3.0, 5.0, 10.0):
        eo = (d.R_f >= kk * d.min_effect) & (d.ri_plugin < d.min_effect)
        to = (d.ri_plugin >= kk * d.min_effect) & (d.R_f < d.min_effect)
        sweep[f"{kk:g}x"] = {
            reg: {"eta_only_runs": int(d[(d.region == reg) & eo].object_id.nunique()),
                  "eta_only_subset_rows": int(uniq(d[(d.region == reg) & eo]).shape[0]),
                  "theta_only_runs": int(d[(d.region == reg) & to].object_id.nunique()),
                  "theta_only_subset_rows": int(uniq(d[(d.region == reg) & to]).shape[0])}
            for reg in REGIONS}
        cells = mk[(mk.R_f >= kk * mk.min_effect) & (mk.ri_plugin < mk.min_effect)].assign(
            cell=lambda z: z.couplings + "|" + z.context)
        sweep[f"{kk:g}x"]["markov_eta_only_by_cell"] = {
            c: int(g.object_id.nunique()) for c, g in cells.groupby("cell")}
        sweep[f"{kk:g}x"]["markov_eta_only_relations"] = {
            str(k): int(v) for k, v in uniq(cells).groupby("relation").size().items()}
    # cases at the 2x level: the point of largest |gap| per (run, subset), with η and θ coefficients
    cases2 = []
    curves = {}
    fmap = dict(zip(x.object_id, x.ricurve_file))
    for kind, mask in (("eta_only_2x", (d.R_f >= 2 * d.min_effect) & (d.ri_plugin < d.min_effect)),
                       ("theta_only_2x", (d.ri_plugin >= 2 * d.min_effect) & (d.R_f < d.min_effect))):
        sel = d[mask].assign(ag=lambda z: z.gap.abs()).sort_values("ag", ascending=False).drop_duplicates(
            ["object_id", "row"])
        for rr in sel.itertuples():
            if rr.object_id not in curves:
                curves[rr.object_id] = RICurve.load(args.extracted / fmap[rr.object_id])
            c = curves[rr.object_id]
            cases2.append(dict(kind=kind, region=rr.region, object_id=rr.object_id,
                               subset=[c.names[j] for j in c.subsets[rr.row]], relation=rr.relation, step=rr.step,
                               f=rr.f, R_f=rr.R_f, ri_plugin=rr.ri_plugin, ri_corrected=rr.ri, q=rr.q,
                               rebuilt=rr.rebuilt, n_checkpoints=int(mask[(d.object_id == rr.object_id)
                                                                        & (d.row == rr.row)].sum()),
                               profile=theta_eta_profile(c, rr.row, int(rr.t))))
    # where do divergences occur? subsets with at least two inputs, binned by saturation and lower-order energy
    multi = d[d.complete & (d.order >= 3) & np.isfinite(d.saturation)].copy()
    multi["div2"] = ((multi.R_f >= 2 * multi.min_effect) & (multi.ri_plugin < multi.min_effect)) | (
        (multi.ri_plugin >= 2 * multi.min_effect) & (multi.R_f < multi.min_effect))
    multi["sat_bin"] = pd.cut(multi.saturation, [0, 0.5, 0.8, 0.9, 0.95, 10], include_lowest=True,
                              labels=["<0.5", "0.5-0.8", "0.8-0.9", "0.9-0.95", ">=0.95"])
    multi["low_bin"] = pd.cut(multi.lower_energy, [0, 0.1, 0.3, 10], include_lowest=True,
                              labels=["<0.1", "0.1-0.3", ">=0.3"])
    multi["sig"] = multi.R_f >= 2 * multi.min_effect  # a top coefficient that matters
    multi["rb_absdiff"] = (multi.rebuilt - multi.ri_plugin).abs()
    tab = multi[multi.sig].groupby(["sat_bin", "low_bin"], observed=False).agg(
        size=("div2", "size"), divergent=("div2", "sum"), rebuild_absdiff_median=("rb_absdiff", "median"))
    sat_table = [{"saturation": str(a), "lower_energy": str(b), "points_with_R_f_ge_2m": int(v["size"]),
                  "divergent": int(v["divergent"]),
                  "rebuild_minus_plugin_abs_median": (None if not np.isfinite(v["rebuild_absdiff_median"])
                                                      else float(v["rebuild_absdiff_median"]))}
                 for (a, b), v in tab.iterrows()]
    # theta-only pairs: a biased mean output raises the mutual information of a given covariance
    def h_b(q):
        q = min(max(q, 1e-15), 1 - 1e-15)
        return -(q * math.log(q) + (1 - q) * math.log(1 - q))

    def mi_pair(f0, fi):
        return h_b((1 + f0) / 2) - 0.5 * (h_b((1 + f0 + fi) / 2) + h_b((1 + f0 - fi) / 2))

    def implied_f0(fi, target):
        g = lambda f0: mi_pair(f0, fi) - target  # noqa: E731
        hi = 1 - abs(fi) - 1e-9
        if g(0.0) >= 0:
            return 0.0  # the MI at f(∅) = 0 already reaches the stored value
        if g(hi) <= 0:
            return math.nan
        lo_, hi_ = 0.0, hi
        for _ in range(80):
            mid = (lo_ + hi_) / 2
            lo_, hi_ = (mid, hi_) if g(mid) < 0 else (lo_, mid)
        return (lo_ + hi_) / 2

    pair_checks = []
    for cs in cases2:
        if cs["kind"] != "theta_only_2x" or len(cs["subset"]) != 2:
            continue
        c = curves[cs["object_id"]]
        t = list(c.steps).index(cs["step"])
        oi = c.names.index(c.meta["fourier_output"])
        others = [implied_f0(c.fourier[i, t], c.ri_plugin[i, t]) for i, sb in enumerate(c.subsets)
                  if len(sb) == 2 and oi in sb and abs(c.fourier[i, t]) > 0.03 and c.subsets[i] != tuple(
                      sorted(c.names.index(n) for n in cs["subset"]))]
        others = [v for v in others if math.isfinite(v)]
        pair_checks.append(dict(object_id=cs["object_id"], region=cs["region"], subset=cs["subset"], step=cs["step"],
                                implied_abs_f0_this_pair=implied_f0(cs["f"], cs["ri_plugin"]),
                                implied_abs_f0_other_pairs_median=float(np.median(others)) if others else None,
                                n_other_pairs=len(others)))
    # the largest rebuild-vs-plug-in differences
    comp = d[d.complete & np.isfinite(d.rebuilt)].copy()
    comp["absdiff"] = (comp.rebuilt - comp.ri_plugin).abs()
    worst = comp.sort_values("absdiff", ascending=False).head(15)[
        ["region", "object_id", "order", "relation", "step", "f", "R_f", "ri_plugin", "rebuilt", "absdiff"]]
    diff_by_f = comp.assign(fbin=pd.cut(comp.f.abs(), [0, 0.1, 0.5, 0.9, 0.99, 1.0], include_lowest=True)).groupby(
        "fbin", observed=True).absdiff.agg(["size", "median", "max"])
    # exemplars: strongest of each kind per region
    ex = {}
    cdf = pd.DataFrame(cases)
    if len(cdf):
        for (kind, reg), g in cdf.groupby(["kind", "region"]):
            key = "f" if kind == "eta_only" else "ri_plugin"
            g = g.reindex(g[key].abs().sort_values(ascending=False).index)
            ex[f"{kind}|{reg}"] = g.head(8).to_dict(orient="records")
    # by step: share of eta-only points per checkpoint index (pooled over regions)
    by_t = d.groupby(["region", "t"]).agg(step=("step", "median"), eta_only=("eta_only", "mean"),
                                           theta_only=("theta_only", "mean"))
    result = {
        "description": __doc__.split("\n\n")[0],
        "thresholds": {"strong_multiple_of_min_effect": STRONG, "min_effect_values": sorted(set(d.min_effect))},
        "summary": summary,
        "markov_cells": markov_cells,
        "share_by_checkpoint": {reg: {"step": g.step.astype(int).tolist(), "eta_only": g.eta_only.round(6).tolist(),
                                      "theta_only": g.theta_only.round(6).tolist()}
                                for reg, g in by_t.reset_index().groupby("region")},
        "exemplars": ex,
        "threshold_sweep": sweep,
        "cases_2x": cases2,
        "divergence_by_saturation_and_lower_energy": sat_table,
        "theta_only_pairs_mean_output_check": pair_checks,
        "rebuild_worst": worst.to_dict(orient="records"),
        "rebuild_absdiff_by_abs_f": {str(k): {"points": int(v["size"]), "median": float(v["median"]),
                                              "max": float(v["max"])} for k, v in diff_by_f.iterrows()},
        "n_cases": {k: int(v) for k, v in Counter(c["kind"] for c in cases).items()},
    }

    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(w) for w in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return None if not math.isfinite(float(v)) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean(result), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(clean({k: v for k, v in result.items()
                            if k in ("threshold_sweep", "divergence_by_saturation_and_lower_energy")}),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
