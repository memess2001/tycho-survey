"""A3, exact check: regenerate the RI probes and reconstruct the stored plug-in tables at every divergent checkpoint.

``a3_theta_eta_map.py`` rebuilds RI from the stored Walsh coefficients assuming exactly uniform probe cells and a
mean output f(∅) = 0. Near saturation those assumptions fail. The probes, however, are deterministic: at checkpoint
step t the runner draws them from ``torch_generator(seed, "ri", t)`` (parity: ``torch.randint(0, 2, (n_probe, n),
dtype=float32)``; Markov: ``torch.randint(0, 2, (n_probe, window))``). With the regenerated rows, the empirical
table of a subset A ∪ {out} follows exactly from the stored coefficients and one scalar f(∅) per checkpoint:

    N_x = probe count of cell x,  D(x) = (1/N) Σ_{n in x} (2 p1_n − 1) = 2^-|A| (f(∅) + Σ_{B ≠ ∅} f(B) χ_B(x)),
    P(x, 1) = (N_x / N + D(x)) / 2,  P(x, 0) = (N_x / N − D(x)) / 2.

f(∅) is fitted so that the reconstructed tables reproduce the stored plug-in RI of every complete subset at that
checkpoint. Then, for each point flagged by the naive 2× rule, two exact quantities follow from the cell means
m(x) = D(x) / (N_x / N):

* the η coefficient on uniform cells, f_u(A) = 2^-|A| Σ_x m(x) χ_A(x), and its single-mode value R(f_u);
* the θ top coefficient, the Walsh coefficient on A of atanh(m(x)) = logit/2.

The point is re-classified with R(f_u) in place of R(f). The stored f is the uncentred sample coefficient, which
is biased when the mean output is far from 0 and the probe cells are unbalanced. Method contributed by the
independent verifier of A3.

Run from the repository root after extracting DR1 into work/dr1:
``PYTHONPATH=. python scripts/analysis_v2/a3_exact_reconstruction.py``. Writes
``docs/results/v2_analysis/a3_exact_reconstruction.json``.
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
from scipy.optimize import minimize_scalar

import tycho
from tycho.analysis.ricurve import RICurve
from tycho.ri.refined import local_ri_batch
from tycho.tasks.base import torch_generator

REPO = Path(__file__).resolve().parents[2]
assert Path(tycho.__file__).resolve().is_relative_to(REPO), f"wrong tycho package: {tycho.__file__}"
REGIONS = ("parity-k2", "parity-k3", "parity-k4", "markov-v1")
LN2 = math.log(2.0)
LEVEL = 2.0


def single_mode(f):
    q = min(max((1.0 + abs(f)) / 2.0, 1e-15), 1.0)
    r = 1.0 - q
    return max(LN2 + q * math.log(q) + (r * math.log(r) if r > 0 else 0.0), 0.0)


def probe_bits(c: RICurve, region: str, step: int) -> np.ndarray:
    gen = torch_generator(int(c.meta["seed"]), "ri", int(step))
    n = int(c.meta["n_probe"])
    w = len(c.names) - 1
    if region.startswith("parity"):
        x = torch.randint(0, 2, (n, w), generator=gen, dtype=torch.float32)
    else:
        x = torch.randint(0, 2, (n, w), generator=gen)
    return x.numpy().astype(np.int64)


class Checkpoint:
    """Reconstructed empirical tables of every complete subset (with the output) at one checkpoint."""

    def __init__(self, c: RICurve, t: int, bits: np.ndarray):
        self.c, self.t, self.N = c, t, bits.shape[0]
        self.o = c.names.index(c.meta["fourier_output"])
        index = {s: j for j, s in enumerate(c.subsets)}
        self.subs = {}
        for j, s in enumerate(c.subsets):
            if self.o not in s or len(s) < 2:
                continue
            A = [v for v in s if v != self.o]
            k = len(A)
            cells = np.array(list(itertools.product((0, 1), repeat=k)))
            codes = (bits[:, A] * (1 << np.arange(k)[::-1])).sum(axis=1)
            Nx = np.bincount(codes, minlength=1 << k).astype(float)
            lower = np.zeros(1 << k)
            ok = True
            for r in range(1, k + 1):
                for Bpos in itertools.combinations(range(k), r):
                    key = tuple(sorted([A[p] for p in Bpos] + [self.o]))
                    if key not in index:
                        ok = False
                        break
                    chi = np.prod(2 * cells[:, list(Bpos)] - 1, axis=1)
                    lower += c.fourier[index[key], t] * chi
                if not ok:
                    break
            if ok:
                self.subs[j] = (k, Nx, lower, cells)

    def tables(self, f0: float) -> dict:
        out = {}
        for j, (k, Nx, lower, _) in self.subs.items():
            D = (f0 + lower) / 2 ** k
            out[j] = np.stack([(Nx / self.N - D) / 2, (Nx / self.N + D) / 2], axis=-1).reshape((2,) * k + (2,))
        return out

    def ri(self, f0: float) -> dict:
        tabs = self.tables(f0)
        res, by = {}, {}
        for j, tb in tabs.items():
            by.setdefault(tb.shape, []).append(j)
        for js in by.values():
            vals = local_ri_batch(np.stack([np.clip(tabs[j], 1e-300, None) for j in js]))
            res.update(zip(js, vals))
        return res

    def fit_f0(self) -> tuple[float, float]:
        def loss(f0):
            tabs = self.tables(f0)
            pen = sum(np.clip(-tb, 0, None).sum() for tb in tabs.values())
            r = self.ri(f0)
            return sum((r[j] - self.c.ri_plugin[j, self.t]) ** 2 for j in r) + 1e3 * pen
        grid = np.linspace(-0.99, 0.99, 199)
        g0 = grid[int(np.argmin([loss(g) for g in grid]))]
        res = minimize_scalar(loss, bounds=(g0 - 0.01, g0 + 0.01), method="bounded", options={"xatol": 1e-12})
        f0 = float(res.x)
        r = self.ri(f0)
        err = max(abs(r[j] - self.c.ri_plugin[j, self.t]) for j in r)
        return f0, float(err)

    def exact_point(self, j: int, f0: float) -> dict:
        k, Nx, lower, cells = self.subs[j]
        D = (f0 + lower) / 2 ** k
        mean = np.clip(D / np.maximum(Nx / self.N, 1e-300), -1 + 1e-12, 1 - 1e-12)
        chiA = np.prod(2 * cells - 1, axis=1)
        f_u = float((mean * chiA).mean())
        theta_top = float((np.arctanh(mean) * chiA).mean())
        return dict(f_u=f_u, R_f_u=single_mode(f_u), theta_top=theta_top,
                    saturation=float(np.abs(mean).max()), cell_count_min=int(Nx.min()), cell_count_max=int(Nx.max()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases" / "DR1")
    ap.add_argument("--extracted", type=Path, default=REPO / "work" / "dr1")
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / "results" / "v2_analysis" / "a3_exact_reconstruction.json")
    args = ap.parse_args(argv)
    o = pd.read_parquet(args.release / "objects.parquet")
    r = pd.read_parquet(args.release / "ri_runs.parquet")
    x = r[r.kind == "insitu"].merge(o[["object_id", "region", "seed", "task_cfg_json"]], on="object_id")
    x = x[x.region.isin(REGIONS)].sort_values(["region", "object_id"])

    points, fits = [], []
    for row in x.itertuples():
        c = RICurve.load(args.extracted / row.ricurve_file)
        assert int(c.meta["seed"]) == int(row.seed)
        me = float(c.meta["min_effect"])
        oi = c.names.index(c.meta["fourier_output"])
        rows = [j for j, s in enumerate(c.subsets) if oi in s and len(s) >= 2]
        f = c.fourier[rows]
        rf = np.vectorize(single_mode)(f)
        rp = c.ri_plugin[rows]
        eta = (rf >= LEVEL * me) & (rp < me)
        theta = (rp >= LEVEL * me) & (rf < me)
        flagged_t = sorted(set(np.nonzero((eta | theta).any(axis=0))[0].tolist()))
        if not flagged_t:
            continue
        cfg = json.loads(row.task_cfg_json)
        for t in flagged_t:
            cp = Checkpoint(c, t, probe_bits(c, row.region, int(c.steps[t])))
            f0, err = cp.fit_f0()
            fits.append(dict(region=row.region, object_id=row.object_id, step=int(c.steps[t]), f0=f0,
                             max_abs_error=err, n_subsets=len(cp.subs)))
            for ii, j in enumerate(rows):
                for kind, mask in (("eta_only", eta), ("theta_only", theta)):
                    if not mask[ii, t]:
                        continue
                    if j not in cp.subs:
                        points.append(dict(kind=kind, region=row.region, object_id=row.object_id, step=int(c.steps[t]),
                                           subset=[c.names[v] for v in c.subsets[j]], reconstructed=False))
                        continue
                    ex = cp.exact_point(j, f0)
                    still = (ex["R_f_u"] >= LEVEL * me and rp[ii, t] < me) if kind == "eta_only" else \
                        (rp[ii, t] >= LEVEL * me and ex["R_f_u"] < me)
                    points.append(dict(kind=kind, region=row.region, object_id=row.object_id, step=int(c.steps[t]),
                                       couplings=cfg.get("couplings"), context=cfg.get("train_context"),
                                       subset=[c.names[v] for v in c.subsets[j]], reconstructed=True,
                                       f_stored=float(f[ii, t]), R_f_stored=float(rf[ii, t]),
                                       ri_plugin=float(rp[ii, t]), f0=f0, still_divergent=bool(still), **ex))
    P = pd.DataFrame(points)
    F = pd.DataFrame(fits)
    rec = P[P.reconstructed]
    summary = {}
    for (kind, reg), g in rec.groupby(["kind", "region"]):
        summary[f"{kind}|{reg}"] = {
            "points": int(len(g)), "still_divergent": int(g.still_divergent.sum()),
            "runs": int(g.object_id.nunique()), "runs_still_divergent": int(g[g.still_divergent].object_id.nunique()),
            "median_abs_f0": float(g.f0.abs().median()), "median_saturation": float(g.saturation.median()),
            "median_abs_theta_top": float(g.theta_top.abs().median()),
        }
    result = {
        "description": __doc__.split("\n\n")[0],
        "level": LEVEL,
        "fit": {"checkpoints": int(len(F)), "max_abs_error": float(F.max_abs_error.max()),
                "median_abs_error": float(F.max_abs_error.median()),
                "by_region": {reg: {"checkpoints": int(len(g)), "max_abs_error": float(g.max_abs_error.max()),
                                    "median_abs_f0": float(g.f0.abs().median())} for reg, g in F.groupby("region")}},
        "not_reconstructed_points": int((~P.reconstructed).sum()),
        "summary": summary,
        "points": P.to_dict(orient="records"),
        "fits": F.to_dict(orient="records"),
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
    print(json.dumps(clean({k: v for k, v in result.items() if k in ("fit", "not_reconstructed_points", "summary")}),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
