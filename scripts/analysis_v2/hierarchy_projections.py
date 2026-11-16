"""Exploratory analysis of DR2a (not pre-registered): does the training trajectory pass near the hierarchy of
projections of the data conditional?

Setting: the Markov language of sky v2a. Six binary lags (64 windows, code ``c = Σ_j b_j 2^j``, input j = lag j + 1,
spin ``s_j = 2 b_j − 1``), the exact training distribution D of the window, the exact true conditional p*, and the
network's log-odds on every window at every checkpoint (pattern mode). Everything below is an exact population value.

Definitions (KL between Bernoulli distributions, in nats; σ the logistic function; χ_B(c) = Π_{j∈B} s_j):

* ``F_k``: predictors whose log-odds are ``Σ_{|B| ≤ k} w_B χ_B(c)`` (interactions of at most k lags with the output);
* projection ``P_k[p]`` of a conditional p onto F_k under D: the member of F_k that minimises
  ``Σ_c D(c) KL(p(c) ‖ q(c))`` (a weighted logistic regression with soft targets; the m-projection of D·p onto the
  hierarchical log-linear model with every input term free and input-output terms up to k lags);
* data rungs ``p^(k) = P_k[p*]``; loss left at rung k, ``L_k = Σ_c D(c) KL(p* ‖ p^(k))``; step length
  ``r_k = L_{k−1} − L_k = Σ_c D(c) KL(p^(k) ‖ p^(k−1))`` (Pythagoras; equal to the refined information of the order-k
  input-output terms summed along any chain);
* network q_t at checkpoint t: ``loss(t) = Σ_c D(c) KL(p* ‖ q_t)`` and ``d_k(t) = Σ_c D(c) KL(p^(k) ‖ q_t)``;
* visit of rung k: ``v_k = min_t d_k(t) / L_k`` (how close the network gets to rung k, relative to how far rung k is
  from the truth); the same along the straight line in log-odds from the first to the last checkpoint (all orders
  learned in proportion) is the reference ``v_line_k``;
* intermediate rung of a configuration: ``1 ≤ k ≤ 5`` with ``L_k ≥ 0.002`` and ``r_k ≥ 0.002``;
* order content of the network: ``M_k(t) = Σ_c D(c) KL(q_t ‖ P_k[q_t])``, ``c_k(t) = M_{k−1}(t) − M_k(t)``: how much
  order-k structure the network itself has, with no threshold and no choice of subsets;
* half-time ``H(f)`` of a series: the first upward crossing of f × (mean of the last 5 checkpoints), interpolated
  linearly in log(step) (step 0 counted as 0.5);
* gain at a rung: ``G_k = Σ_c D(c) (1 − m_k(c)²)`` with ``m_k = 2 p^(k) − 1``. In a kernel picture with uniform
  inputs, a term that is not yet learned relaxes toward its target at a rate proportional to G at the state the
  network sits in, so a higher-order term learned from rung k rather than from an untrained network is delayed by
  about ``1 / G_k`` ("rule A"). The ratio of the term's Walsh coefficients alone and with the lower-order term is
  reported as an alternative ("rule B");
* gain-weighted time, a stricter version of rule A with no free choice: ``τ(T) = ∫_0^T G(t) dt`` with
  ``G(t) = Σ_c D(c) (1 − m_t(c)²)`` measured on the network itself (linear between checkpoints). If output saturation
  were the whole delay, the planted term's uniform Walsh coefficient would reach a given fraction of its final value
  at the same τ with and without the lower-order term (ratio 1).

The script deliberately does not import the tycho package (numpy and the standard library only), so that it is an
independent implementation and runs without torch.

Usage, from the repository root: ``python scripts/analysis_v2/hierarchy_projections.py`` (options ``--release``,
``--work``, ``--out``). It extracts the release's pattern and light-curve archives into ``--work`` when needed and
writes ``docs/results/v2_analysis/hierarchy_projections.json``. About 90 s on one core.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MIN_RUNG = 0.002      # nats: a rung is intermediate if it is this far from the rung below and from the truth
N_LINE = 401          # points of the straight reference path
FRACS = (0.25, 0.5, 0.75)
TINY = 1e-300

#: twin comparisons (iid contexts): (alone, with the lower-order term, order of the higher-order term, stand-in?)
TWINS = (
    ("1,3:0.8", "1:1.0;1,3:0.8", 2, False),
    ("1,2,3:1.0", "1:1.0;1,2,3:1.0", 3, False),
    ("1,2,3:2.0", "1:2.0;1,2,3:2.0", 3, False),
    # DR2a has no "1,3:1.6" run; the alone time does not depend on the strength (see "alone_times"), so the
    # strongest available alone run stands in. Rule A's value for this row was computed before the row was measured.
    ("1,3:1.0", "1:2.0;1,3:1.6", 2, True),
)

#: configurations that have not been run, with rule A's and rule B's predictions (see the note, "A prospective test")
PROPOSED = (
    ("1,3:0.8", "1:0.25;1,3:0.8", 2),
    ("1,3:0.8", "1:0.5;1,3:0.8", 2),
    ("1,3:0.8", "1:1.5;1,3:0.8", 2),
    ("1,3:0.8", "1:2.0;1,3:0.8", 2),
    ("1,3:0.8", "1:3.0;1,3:0.8", 2),
    ("1,3:0.8", "4:1.0;1,3:0.8", 2),       # a lower-order term on a lag outside the higher-order set
    ("1,3:0.8", "4:2.0;1,3:0.8", 2),
    ("1,2,3:1.0", "1:2.0;1,2,3:1.0", 3),
    ("1,2,3:1.0", "1:3.0;1,2,3:1.0", 3),
)


# ------------------------------------------------------------------------------------------------- numerics
def features(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Walsh features χ_B on all 2^d codes (column B, as a bit mask) and the size |B| of each."""
    codes = np.arange(1 << d)
    spins = 2.0 * ((codes[:, None] >> np.arange(d)[None, :]) & 1) - 1.0
    phi = np.ones((1 << d, 1 << d))
    deg = np.zeros(1 << d, dtype=int)
    for b in range(1 << d):
        idx = [j for j in range(d) if (b >> j) & 1]
        deg[b] = len(idx)
        if idx:
            phi[:, b] = np.prod(spins[:, idx], axis=1)
    return phi, deg


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def bern_kl(p: np.ndarray, z: np.ndarray, w: np.ndarray) -> float:
    """Σ_c w_c KL(Bern(p_c) ‖ Bern(σ(z_c))), from the log-odds z."""
    log_q1 = -np.logaddexp(0.0, -z)
    log_q0 = -np.logaddexp(0.0, z)
    t1 = np.where(p > 0, p * (np.log(np.maximum(p, TINY)) - log_q1), 0.0)
    t0 = np.where(p < 1, (1 - p) * (np.log(np.maximum(1 - p, TINY)) - log_q0), 0.0)
    return float(np.sum(w * (t1 + t0)))


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def wlogit(phi: np.ndarray, w: np.ndarray, p: np.ndarray, ridge: float = 1e-13, tol: float = 1e-13,
           iters: int = 500) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted logistic regression with soft targets p and cell weights w (damped Newton).

    Returns the fitted log-odds, the coefficients and the largest gradient component left. The tiny ridge only fixes
    the solution in directions that D does not see (cells of negligible mass); D-weighted quantities do not depend on
    it.
    """
    beta = np.zeros(phi.shape[1])

    def obj(b: np.ndarray) -> float:
        z = phi @ b
        return float(np.sum(w * (np.logaddexp(0.0, z) - p * z))) + 0.5 * ridge * float(b @ b)

    f = obj(beta)
    for _ in range(iters):
        q = sigmoid(phi @ beta)
        g = phi.T @ (w * (q - p)) + ridge * beta
        if np.max(np.abs(g)) < tol:
            break
        h = (phi * (w * q * (1 - q))[:, None]).T @ phi + ridge * np.eye(phi.shape[1])
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(h, g, rcond=None)[0]
        t = 1.0
        f_new = f
        while t > 1e-12:
            f_new = obj(beta - t * step)
            if f_new <= f - 1e-4 * t * float(g @ step):
                break
            t *= 0.5
        beta = beta - t * step
        done = abs(f - f_new) < 1e-16 * max(1.0, abs(f))
        f = f_new
        if done:
            break
    z = phi @ beta
    return z, beta, float(np.max(np.abs(phi.T @ (w * (sigmoid(z) - p)))))


def wls(phi: np.ndarray, w: np.ndarray, m: np.ndarray) -> np.ndarray:
    """D-weighted least squares of the mean m on the features: the fitted mean (the "η" rung)."""
    a = phi * np.sqrt(w)[:, None]
    return phi @ np.linalg.lstsq(a, m * np.sqrt(w), rcond=None)[0]


def half_time(steps: np.ndarray, y: np.ndarray, frac: float) -> float:
    target = frac * float(np.mean(y[-5:]))
    for i in range(1, len(y)):
        if y[i] >= target and y[i - 1] < target:
            a, b = math.log(max(float(steps[i - 1]), 0.5)), math.log(float(steps[i]))
            return math.exp(a + (target - y[i - 1]) / (y[i] - y[i - 1]) * (b - a))
    return float("nan")


def gain_weighted_time(steps: np.ndarray, gain: np.ndarray, t_end: float) -> float:
    """τ(t_end) = ∫_0^t_end G(t) dt, with G linear between checkpoints."""
    tau = 0.0
    for i in range(1, len(steps)):
        a, b = float(steps[i - 1]), float(steps[i])
        if t_end <= a:
            break
        hi = min(b, t_end)
        g_hi = gain[i - 1] + (gain[i] - gain[i - 1]) * (hi - a) / (b - a)
        tau += 0.5 * (gain[i - 1] + g_hi) * (hi - a)
    return tau


# --------------------------------------------------------------------------------------------- the source
def truth_table(couplings: str, window: int) -> np.ndarray:
    """P(x_t = 1 | window code) = σ(2h), h = Σ_L β_L Π_{ℓ∈L} s_{t−ℓ} (as tycho.tasks.markov_lang)."""
    codes = np.arange(1 << window)
    spins = 2.0 * ((codes[:, None] >> np.arange(window)[None, :]) & 1) - 1.0
    h = np.zeros(codes.size)
    for part in couplings.split(";"):
        lags, beta = part.rsplit(":", 1)
        h += float(beta) * np.prod(spins[:, [int(x) - 1 for x in lags.split(",")]], axis=1)
    return 1.0 / (1.0 + np.exp(-2.0 * h))


def stationary_pmf(p_table: np.ndarray, window: int) -> np.ndarray:
    """Exact stationary pmf of the window of the chain (as tycho.tasks.patterns.markov_window_pmf)."""
    n = 1 << window
    codes = np.arange(n)
    t = np.zeros((n, n))
    t[codes, (codes << 1) & (n - 1)] += 1.0 - p_table
    t[codes, ((codes << 1) | 1) & (n - 1)] += p_table
    a = t.T - np.eye(n)
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.clip(np.linalg.solve(a, b), 0.0, None)
    return pi / pi.sum()


def data_rungs(pmf: np.ndarray, p_true: np.ndarray, phi: np.ndarray, deg: np.ndarray, d: int) -> dict:
    """The rungs p^(k), k = 0..d, of the θ hierarchy (log-odds) and of the η hierarchy (means), with L_k."""
    z, beta, m_eta, gmax = [], [], [], 0.0
    for k in range(d + 1):
        cols = deg <= k
        zk, bk, g = wlogit(phi[:, cols], pmf, p_true)
        full = np.zeros(phi.shape[1])
        full[cols] = bk
        z.append(zk)
        beta.append(full)
        m_eta.append(wls(phi[:, cols], pmf, 2.0 * p_true - 1.0))
        gmax = max(gmax, g)
    z = np.array(z)
    L = np.array([bern_kl(p_true, z[k], pmf) for k in range(d + 1)])
    pyth = max(abs((L[k - 1] - L[k]) - bern_kl(sigmoid(z[k]), z[k - 1], pmf)) for k in range(1, d + 1))
    gain = np.array([float(np.sum(pmf * (1.0 - np.tanh(0.5 * z[k]) ** 2))) for k in range(d + 1)])
    return {"z": z, "beta": np.array(beta), "m_eta": np.array(m_eta), "L": L, "pyth": pyth, "gmax": gmax,
            "gain": gain}


def rung_terms(beta: np.ndarray, deg: np.ndarray, d: int, floor: float = 0.05, top: int = 6) -> list[dict]:
    """The largest log-odds coefficients of a rung, as lag sets (the coefficient is of log-odds / 2, like h)."""
    out = []
    for b in np.argsort(-np.abs(beta)):
        if deg[b] == 0 or abs(beta[b]) / 2.0 < floor or len(out) >= top:
            continue
        out.append({"lags": [j + 1 for j in range(d) if (b >> j) & 1], "h_coefficient": float(beta[b] / 2.0)})
    return out


# ------------------------------------------------------------------------------------------- one run
def analyse_run(npz, phi: np.ndarray, deg: np.ndarray, d: int) -> dict:
    steps = np.asarray(npz["steps"], dtype=np.int64)
    zt = np.asarray(npz["logodds"], dtype=np.float64)
    pmf = np.asarray(npz["pmf_D"], dtype=np.float64)
    p_true = np.asarray(npz["truth"], dtype=np.float64)
    T = len(steps)
    rg = data_rungs(pmf, p_true, phi, deg, d)
    p_theta = sigmoid(rg["z"])
    loss = np.array([bern_kl(p_true, zt[t], pmf) for t in range(T)])
    dist = np.array([[bern_kl(p_theta[k], zt[t], pmf) for t in range(T)] for k in range(d + 1)])
    lam = np.linspace(0.0, 1.0, N_LINE)
    zl = (1 - lam)[:, None] * zt[0][None, :] + lam[:, None] * zt[-1][None, :]
    dist_line = np.array([[bern_kl(p_theta[k], zl[i], pmf) for i in range(N_LINE)] for k in range(d + 1)])
    # θ rungs and η rungs in two metrics (KL, and squared error on the probability)
    pt = sigmoid(zt)
    p_eta = np.clip(0.5 * (1.0 + rg["m_eta"]), 0.0, 1.0)
    pe = np.clip(p_eta, 1e-9, 1 - 1e-9)
    sq = lambda a, b: float(np.sum(pmf * (a - b) ** 2))  # noqa: E731
    cmp = {
        "A_theta": [float(np.min(dist[k]) / max(rg["L"][k], TINY)) for k in range(d + 1)],
        "A_eta": [float(min(bern_kl(pe[k], zt[t], pmf) for t in range(T)) /
                        max(bern_kl(p_true, logit(pe[k]), pmf), TINY)) for k in range(d + 1)],
        "S_theta": [float(min(sq(p_theta[k], pt[t]) for t in range(T)) / max(sq(p_theta[k], p_true), TINY))
                    for k in range(d + 1)],
        "S_eta": [float(min(sq(p_eta[k], pt[t]) for t in range(T)) / max(sq(p_eta[k], p_true), TINY))
                  for k in range(d + 1)],
    }
    # order content of the network
    M = np.zeros((d + 1, T))
    for t in range(T):
        for k in range(d):
            zk, _, _ = wlogit(phi[:, deg <= k], pmf, pt[t])
            M[k, t] = bern_kl(pt[t], zk, pmf)
    content = np.vstack([M[k - 1] - M[k] for k in range(1, d + 1)])
    return {"steps": steps, "pmf": pmf, "p_true": p_true, "rungs": rg, "loss": loss, "dist": dist,
            "dist_line": dist_line, "cmp": cmp, "content": content, "z": zt}


# ------------------------------------------------------------------------------------------------ main
def ensure_extracted(release: Path, work: Path, name: str) -> Path:
    if not (work / name).is_dir():
        work.mkdir(parents=True, exist_ok=True)
        with tarfile.open(release / f"{name}.tar.gz") as tar:
            tar.extractall(work, filter="data")
    return work / name


def med(xs) -> float:
    return float(np.nanmedian(np.asarray(list(xs), dtype=float)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", type=Path, default=REPO / "releases" / "DR2a")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "dr2a")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" /
                    "hierarchy_projections.json")
    args = ap.parse_args(argv)
    pat_dir = ensure_extracted(args.release, args.work, "patterns")
    lc_dir = ensure_extracted(args.release, args.work, "lightcurves")
    d = 6
    phi, deg = features(d)

    with open(args.release / "objects.csv", newline="", encoding="utf-8") as fh:
        objects = [r for r in csv.DictReader(fh) if r["task"] == "markov_lang"]
    runs = defaultdict(list)          # (couplings, context) -> run records
    checks = {"pythagoras_max_abs": 0.0, "projection_gradient_max": 0.0, "L6_max_abs": 0.0,
              "loss_vs_lightcurve_max_abs": 0.0, "loss_vs_lightcurve_points": 0, "stationary_pmf_max_abs": 0.0}
    for o in objects:
        cfg = json.loads(o["task_cfg_json"])
        key = (cfg["couplings"], cfg["train_context"])
        with np.load(pat_dir / f"{o['object_id']}.npz", allow_pickle=False) as npz:
            r = analyse_run(npz, phi, deg, d)
        r.update({"object_id": o["object_id"], "seed": int(o["seed"])})
        runs[key].append(r)
        rg = r["rungs"]
        checks["pythagoras_max_abs"] = max(checks["pythagoras_max_abs"], rg["pyth"])
        checks["projection_gradient_max"] = max(checks["projection_gradient_max"], rg["gmax"])
        checks["L6_max_abs"] = max(checks["L6_max_abs"], abs(float(rg["L"][d])))
        # the stored truth and pmf_D against this script's own source model
        p_table = truth_table(cfg["couplings"], d)
        assert np.max(np.abs(p_table - r["p_true"])) < 1e-12, "truth table mismatch"
        pmf_own = stationary_pmf(p_table, d) if cfg["train_context"] == "chain" else np.full(1 << d, 1.0 / (1 << d))
        checks["stationary_pmf_max_abs"] = max(checks["stationary_pmf_max_abs"],
                                               float(np.max(np.abs(pmf_own - r["pmf"]))))
        # the loss against the light curve's exact x_lossD (another data path) at the common steps
        with np.load(lc_dir / f"{o['object_id']}.npz", allow_pickle=False) as lc:
            at = {int(s): float(v) for s, v in zip(lc["step"], lc["x_lossD"])}
        for s, v in zip(r["steps"], r["loss"]):
            if int(s) in at:
                checks["loss_vs_lightcurve_max_abs"] = max(checks["loss_vs_lightcurve_max_abs"], abs(v - at[int(s)]))
                checks["loss_vs_lightcurve_points"] += 1
    assert checks["pythagoras_max_abs"] < 1e-9 and checks["projection_gradient_max"] < 1e-8
    assert checks["loss_vs_lightcurve_max_abs"] < 1e-9 and checks["stationary_pmf_max_abs"] < 1e-9

    # ---- configurations: rungs, visits, θ against η
    configurations, pairs = {}, []
    for key in sorted(runs):
        g = sorted(runs[key], key=lambda r: r["seed"])
        rg = g[0]["rungs"]
        L = rg["L"]
        entry = {"couplings": key[0], "context": key[1], "n_seeds": len(g), "L": [float(v) for v in L],
                 "gain": [float(v) for v in rg["gain"]], "intermediate": []}
        for k in range(1, d):
            if L[k] < MIN_RUNG or L[k - 1] - L[k] < MIN_RUNG:
                continue
            idx = [int(np.argmin(r["dist"][k])) for r in g]
            v = [r["dist"][k][i] / L[k] for r, i in zip(g, idx)]
            v_line = [float(np.min(r["dist_line"][k])) / L[k] for r in g]
            item = {
                "k": k, "step_length": float(L[k - 1] - L[k]), "L_k": float(L[k]), "gain": float(rg["gain"][k]),
                "terms": rung_terms(rg["beta"][k], deg, d),
                "v_median": med(v), "v_min": float(min(v)), "v_max": float(max(v)), "v_line_median": med(v_line),
                "t_star_min": int(min(r["steps"][i] for r, i in zip(g, idx))),
                "t_star_max": int(max(r["steps"][i] for r, i in zip(g, idx))),
                "loss_ratio_median": med(r["loss"][i] / L[k] for r, i in zip(g, idx)),
                "theta_vs_eta": {name: med(r["cmp"][name][k] for r in g)
                                 for name in ("A_theta", "A_eta", "S_theta", "S_eta")},
            }
            entry["intermediate"].append(item)
            pairs.append(item)
        configurations[f"{key[0]}|{key[1]}"] = entry
    vs = np.array([p["v_median"] for p in pairs])
    vl = np.array([p["v_line_median"] for p in pairs])
    same = lambda p, a, b: abs(p["theta_vs_eta"][a] - p["theta_vs_eta"][b]) <= 0.001  # noqa: E731
    overall = {
        "intermediate_rungs": len(pairs), "median_v": float(np.median(vs)), "median_v_line": float(np.median(vl)),
        "closer_than_line": int(np.sum(vs < vl)), "v_below_0.1": int(np.sum(vs < 0.1)),
        "v_below_0.25": int(np.sum(vs < 0.25)),
        "theta_eta_rungs_coincide_kl": int(sum(same(p, "A_theta", "A_eta") for p in pairs)),
        "theta_eta_rungs_coincide_sq": int(sum(same(p, "S_theta", "S_eta") for p in pairs)),
    }

    # ---- order content: half-times, twin delays and the two rules
    def half_times(couplings: str, order: int, frac: float) -> dict[int, float]:
        return {r["seed"]: half_time(r["steps"], r["content"][order - 1], frac) for r in runs[(couplings, "iid")]}

    def rules(alone_strength: str, with_c: str, order_lags: list[int]) -> dict:
        """Rule A (1 / gain of the order-1 rung of the 'with' source) and rule B (ratio of Walsh coefficients)."""
        uni = np.full(1 << d, 1.0 / (1 << d))
        p_with = truth_table(with_c, d)
        rg = data_rungs(uni, p_with, phi, deg, d)
        chi = phi[:, sum(1 << (lag - 1) for lag in order_lags)]
        f_with = float(np.mean((2 * p_with - 1) * chi))
        f_alone = float(np.mean((2 * truth_table(alone_strength, d) - 1) * chi))
        return {"gain_rung1": float(rg["gain"][1]), "rule_A": float(1.0 / rg["gain"][1]),
                "rule_B": float(f_alone / f_with), "max_abs_h_rung1": float(np.max(np.abs(rg["z"][1])) / 2.0)}

    def higher_term(c: str) -> str:
        return max(c.split(";"), key=lambda part: len(part.split(":")[0].split(",")))

    delay = []
    for alone, with_c, order, stand_in in TWINS:
        lags = [int(x) for x in higher_term(with_c).split(":")[0].split(",")]
        row = {"alone": alone, "with": with_c, "order": order, "alone_is_stand_in": stand_in,
               **rules(higher_term(with_c), with_c, lags), "ratios": {}}
        for frac in FRACS:
            a, b = half_times(alone, order, frac), half_times(with_c, order, frac)
            R = np.array([b[s] / a[s] for s in sorted(a)])
            row["ratios"][str(frac)] = {"median": float(np.median(R)), "min": float(R.min()), "max": float(R.max()),
                                        "H_alone_median": med(a.values()), "H_with_median": med(b.values())}
        # the stricter test: the planted term's uniform Walsh coefficient against raw and gain-weighted time
        chi = phi[:, sum(1 << (lag - 1) for lag in lags)]
        by_seed = lambda c: {r["seed"]: r for r in runs[(c, "iid")]}  # noqa: E731
        A, W = by_seed(alone), by_seed(with_c)
        row["walsh_progress"] = {}
        for frac in FRACS:
            raw, eff = [], []
            for s in sorted(A):
                pair = []
                for r in (A[s], W[s]):
                    m = np.tanh(0.5 * r["z"])
                    t_half = half_time(r["steps"], (m * chi[None, :]) @ r["pmf"], frac)
                    pair.append((t_half, gain_weighted_time(r["steps"], (1.0 - m ** 2) @ r["pmf"], t_half)))
                raw.append(pair[1][0] / pair[0][0])
                eff.append(pair[1][1] / pair[0][1])
            row["walsh_progress"][str(frac)] = {
                "raw_median": med(raw), "raw_min": float(min(raw)), "raw_max": float(max(raw)),
                "gain_weighted_median": med(eff), "gain_weighted_min": float(min(eff)),
                "gain_weighted_max": float(max(eff))}
        delay.append(row)
    alone_times = {c: {str(f): med(half_times(c, o, f).values()) for f in FRACS}
                   for c, o in (("1,3:0.5", 2), ("1,3:0.8", 2), ("1,3:1.0", 2), ("1,2,3:1.0", 3), ("1,2,3:2.0", 3),
                                ("1,2,4:0.5", 3), ("1,2,4:1.0", 3))}

    # ---- chain against iid for single couplings: not a clean test of rule A (the whole context distribution
    # differs), reported to show where the rule does not carry over
    chain_vs_iid = []
    for c in ("1,2,3:1.0", "1,2,3:2.0", "1,2,3,5:0.5", "1,2,3,5:1.0", "1,3:1.0", "1,2,4:1.0"):
        lags = [int(x) for x in c.split(":")[0].split(",")]
        chi = phi[:, sum(1 << (lag - 1) for lag in lags)]
        t = {ctx: {r["seed"]: {f: half_time(r["steps"], np.tanh(0.5 * r["z"]) @ chi / chi.size, f) for f in FRACS}
                   for r in runs[(c, ctx)]} for ctx in ("chain", "iid")}
        inter = configurations[f"{c}|chain"]["intermediate"]
        row = {"couplings": c, "chain_rung": inter[0]["k"] if inter else None,
               "inverse_gain_at_chain_rung": float(1.0 / inter[0]["gain"]) if inter else None, "ratios": {}}
        for f in FRACS:
            R = np.array([t["chain"][s][f] / t["iid"][s][f] for s in sorted(t["chain"])])
            row["ratios"][str(f)] = {"median": float(np.median(R)), "min": float(R.min()), "max": float(R.max())}
        chain_vs_iid.append(row)

    # ---- predictions for configurations that have not been run (iid contexts)
    predictions = []
    for alone, with_c, order in PROPOSED:
        lags = [int(x) for x in higher_term(with_c).split(":")[0].split(",")]
        uni = np.full(1 << d, 1.0 / (1 << d))
        rg = data_rungs(uni, truth_table(with_c, d), phi, deg, d)
        predictions.append({"alone": alone, "with": with_c, "order": order, "context": "iid",
                            "L": [float(v) for v in rg["L"][: order + 1]],
                            **rules(higher_term(with_c), with_c, lags)})
    # a shorter window removes the chain shortcut of "1,2,3:1.0" without touching the source
    window3 = {}
    for c in ("1,2,3:1.0", "1,2,3:2.0"):
        p3 = truth_table(c, 3)
        phi3, deg3 = features(3)
        rg3 = data_rungs(stationary_pmf(p3, 3), p3, phi3, deg3, 3)
        window3[c] = {"context": "chain", "window": 3, "L": [float(v) for v in rg3["L"]],
                      "L_window6": configurations[f"{c}|chain"]["L"][:4]}

    # ---- example trajectories (seed-sorted first run of every configuration with an intermediate rung)
    examples = {}
    for name, entry in configurations.items():
        if not entry["intermediate"]:
            continue
        r = sorted(runs[(entry["couplings"], entry["context"])], key=lambda x: x["seed"])[0]
        k = entry["intermediate"][0]["k"]
        examples[name] = {"seed": r["seed"], "object_id": r["object_id"], "k": k,
                          "steps": [int(s) for s in r["steps"]], "loss": [float(v) for v in r["loss"]],
                          "d_k": [float(v) for v in r["dist"][k]],
                          "content": [[float(v) for v in row] for row in r["content"]]}

    per_run = [{"object_id": r["object_id"], "seed": r["seed"], "couplings": key[0], "context": key[1],
                "v": {str(k): float(np.min(r["dist"][k]) / r["rungs"]["L"][k]) for k in range(1, d)
                      if r["rungs"]["L"][k] >= MIN_RUNG and r["rungs"]["L"][k - 1] - r["rungs"]["L"][k] >= MIN_RUNG},
                "half_times": {str(o): {str(f): half_time(r["steps"], r["content"][o - 1], f) for f in FRACS}
                               for o in range(1, d + 1) if float(np.mean(r["content"][o - 1][-5:])) >= MIN_RUNG}}
               for key in sorted(runs) for r in sorted(runs[key], key=lambda x: x["seed"])]

    result = {
        "status": "exploratory, post hoc, not pre-registered",
        "release": args.release.name, "n_runs": len(objects), "min_rung_nats": MIN_RUNG,
        "checks": checks, "overall": overall, "configurations": configurations, "delay": delay,
        "alone_times": alone_times, "chain_vs_iid": chain_vs_iid, "predictions_unrun": predictions,
        "window3_chain": window3,
        "examples": examples, "runs": per_run,
    }

    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x

    args.out.write_text(json.dumps(clean(result), indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "overall": overall}, indent=1))
    print(f"{'configuration':>28s} {'k':>2s} {'r_k':>7s} {'L_k':>7s} {'v':>6s} {'v_line':>6s} {'1/G_k':>6s}  terms")
    for name, entry in configurations.items():
        for it in entry["intermediate"]:
            terms = ", ".join("{" + ",".join(map(str, t["lags"])) + f"}}:{t['h_coefficient']:+.2f}" for t in it["terms"])
            print(f"{name:>28s} {it['k']:>2d} {it['step_length']:7.4f} {it['L_k']:7.4f} {it['v_median']:6.3f} "
                  f"{it['v_line_median']:6.3f} {1 / it['gain']:6.2f}  {terms}")
    for row in delay:
        print(f"delay {row['with']:>16s} / {row['alone']:>9s}: measured "
              + " ".join(f"{row['ratios'][str(f)]['median']:.2f}" for f in FRACS)
              + f" (25/50/75 %);  rule A {row['rule_A']:.2f}  rule B {row['rule_B']:.2f};  Walsh coefficient,"
              + " raw "
              + " ".join(f"{row['walsh_progress'][str(f)]['raw_median']:.2f}" for f in FRACS) + ", gain-weighted "
              + " ".join(f"{row['walsh_progress'][str(f)]['gain_weighted_median']:.2f}" for f in FRACS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
