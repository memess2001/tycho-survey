"""Pilot (exploratory; local runs): is the reviewer's law for the growth of the kernel a law?

The reviewer's fifth reply (18 November, to commit 4b44e01) derives its second formula for the kernel's part of the
delay (``kernel_gate_pilot.py``) from three statements about the one-hidden-layer ReLU network
z = Σ_j a_j relu(w_j·s + b_j), with a_j the difference of unit j's two read-out weights:

1. Under gradient flow a_j² − 2 (‖w_j‖² + b_j²) is conserved for every unit.
2. When a coefficient F grows from zero in a direction whose kernel at initialisation is λ0, the kernel in that
   direction follows λ(F) = λ0 + S(F; λ0), S(F; λ0) = 2 λ0 (√(1 + F² / λ0) − 1), with no fitted constant. It checked
   this on the term {1,3}:0.8 alone, at four points of the median curve (within 0.012).
3. Learning a first-order coefficient F_1 raises the read-out norm Σ_j a_j² by S(F_1; λ1), with λ1 the kernel at
   initialisation in a first-order direction (not by an amount proportional to F_1²).

Checks here.

* Statement 1: the drift of a_j² − 2 ‖w̃_j‖² under the SGD of the pilots (learning rate 0.03, batch 64).
* Statement 2 at the width of the pilots (128), per seed and along the whole trajectory; and at other widths
  (32, 64, 256, 512), where λ0 changes by a factor of 16. Recorded before the runs: for small F and frozen unit
  directions the conservation law gives λ − λ0 ≈ (4 n̄ / λ0) F², with n̄ a per-unit quantity (the squared gradient of
  a unit's {1,3} coefficient, about 0.05) that does not depend on the width, while λ0 is proportional to the width.
  The reviewer's law gives λ − λ0 ≈ F² at every width. The two agree near width 128 and differ by a factor of two at
  widths 64 and 256.
* Statement 3: Σ_j a_j² against F_1 while a strong first-order term is being learned (strength 3.0 on input {4} or
  {1}, and 1.5 on {4}), next to S(F_1; λ1) and to F_1².
* What statement 1 does give (no fitted constant; this exact version was added after the runs above, whose
  outcome matched the scaling recorded before them, with a constant of 0.10 in place of the rough 0.05). Write
  g_j for unit j's own coefficient in the
  direction (so F = Σ_j a_j g_j) and n_j = ‖∇ g_j‖² for the squared gradient of g_j with respect to the unit's
  weights and bias; n_j depends on the unit's direction only. If the directions stay as they are at initialisation
  and the residual has that one direction, then for every unit a_j' = 2 g_j and g_j' = n_j a_j, so
  P_j = 2 g_j² + n_j a_j² (its share of the kernel) and Q_j = a_j g_j (its share of F) obey P_j² − 8 n_j Q_j² = const
  and Q_j' = P_j. Summing the units gives the kernel as a function of F from quantities at initialisation alone
  (``frozen_directions``). For small F this is λ − λ0 ≈ 4 (Σ_j n_j P_j) F² / λ0², and for a first-order direction with
  n_j = 1/4 (half of the gates open) the read-out norm grows by 2 (√(λ1² + 2 F_1²) − λ1).

Second round (the reviewer's sixth reply, 19 November, to commit 6062933): the kernel's contribution K at other
widths. K had been measured at width 128 only. The reviewer gives 40 values for widths 64 and 256 (strengths 0.5 to
2.5, both inputs, 25 % and 50 %) and states that it ran no training at those widths. Its main prediction comes from
integrating, from the network at initialisation, the gradient flow in which the residual has two directions only
(the lower-order input and the term) and every unit is free to turn: ``two_direction_flow`` does the same here, as
gradient descent with the pilots' learning rate. Its second set of values is a formula with constants per width.
Its criteria: a mean absolute error of at most 0.025 over the 40 values (failure beyond 0.035), clearly better than
taking K from width 128; with the shared input K rises with the width at strengths 1.0 to 2.5, with K at most 0.48
at width 64 and at least 0.78 at width 256 for strength 2.0 at 25 % (failure if above 0.50 at width 64); without
the shared input K at width 64 stays within 0.045 of width 128 on average over strengths 1.0 to 2.5, and at width
256 it is at least 0.08 higher at strengths 2.0 and 2.5.
Recorded here before the runs. (i) The flow is the network's own training with the sampling noise and the other 62
directions of the residual left out; agreement shows that these do not matter for K, and it is a simulation, not a
formula. (ii) A plain alternative that needs no simulation: the excess 1 / K − 1 scales as one over the width
(the kernel at initialisation is proportional to the width), so K(m) = 1 / (1 + (128 / m)(1 / K(128) − 1)). It
agrees with the reviewer on the direction at width 256 and differs most where the reviewer predicts that K without
the shared input barely moves at width 64 (strength 2.0 at 25 %: 0.62 by the plain scaling, 0.77 by the reviewer).

Runs: the Markov settings of ``kernel_gate_pilot.py`` (SGD, learning rate 0.03, batch 64, iid contexts, seeds
600–607). The term alone for 12,000 steps at widths 32, 64, 128, 256 and 512; the three sources with a first-order
term for 3,000 steps at width 128; and, for the second round, the term with input {1} or {4} at strengths 0.5,
1.0, 1.5, 2.0 and 2.5 for 12,000 steps at widths 64 and 256.

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/kernel_law_pilot.py
--jobs 4``. Runs are cached in ``work/kernel_law``. Writes ``docs/results/v2_analysis/kernel_law_pilot.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("kernel_gate_pilot", Path(__file__).with_name("kernel_gate_pilot.py"))
kg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kg)

SEEDS = kg.SEEDS
ALONE = kg.ALONE
WIDTHS = (32, 64, 128, 256, 512)
FIRST_ORDER = ("4:3.0;1,3:0.8", "1:3.0;1,3:0.8", "4:1.5;1,3:0.8")
#: (source, width, steps)
K_WIDTHS, K_STRENGTHS, K_PROGRESS = (64, 256), ("0.5", "1.0", "1.5", "2.0", "2.5"), (0.25, 0.5)
JOBS = ([(ALONE, w, 12_000) for w in WIDTHS] + [(c, 128, 3_000) for c in FIRST_ORDER]
        + [(kg.WITH[st][lag], w, 12_000) for w in K_WIDTHS for st in K_STRENGTHS for lag in ("lag 1", "lag 4")])
#: The reviewer's 40 values: width -> strength -> (lag 1 at 25 %, lag 1 at 50 %, lag 4 at 25 %, lag 4 at 50 %), from
#: its two-direction flow and from its formula.
REVIEWER_FLOW = {
    64: {"0.5": (0.9526, 0.9664, 0.9926, 0.9851), "1.0": (0.7755, 0.8370, 0.9424, 0.9302),
         "1.5": (0.5620, 0.6552, 0.8571, 0.8576), "2.0": (0.4014, 0.4985, 0.7733, 0.7973),
         "2.5": (0.2950, 0.3878, 0.7105, 0.7524)},
    256: {"0.5": (1.0050, 0.9956, 0.9950, 0.9961), "1.0": (0.9860, 0.9653, 0.9845, 0.9851),
          "1.5": (0.9312, 0.9050, 0.9587, 0.9604), "2.0": (0.8479, 0.8234, 0.9156, 0.9223),
          "2.5": (0.7550, 0.7363, 0.8631, 0.8763)}}
REVIEWER_FORMULA = {
    64: {"0.5": (0.9460, 0.9624, 0.9838, 0.9854), "1.0": (0.7835, 0.8417, 0.9331, 0.9393),
         "1.5": (0.5741, 0.6659, 0.8582, 0.8703), "2.0": (0.4001, 0.4982, 0.7794, 0.7966),
         "2.5": (0.2812, 0.3692, 0.7076, 0.7284)},
    256: {"0.5": (0.9935, 0.9924, 0.9965, 0.9968), "1.0": (0.9690, 0.9639, 0.9831, 0.9845),
          "1.5": (0.9198, 0.9071, 0.9558, 0.9595), "2.0": (0.8489, 0.8269, 0.9158, 0.9225),
          "2.5": (0.7665, 0.7357, 0.8677, 0.8778)}}
CELLS = (("lag 1", "25 %"), ("lag 1", "50 %"), ("lag 4", "25 %"), ("lag 4", "50 %"))
F_POINTS = (0.16, 0.4, 0.8, 1.2, 1.5)                   # the term's coefficient (target 1.6)
F1_POINTS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)              # the first-order coefficient


def growth(f, lam0):
    """The reviewer's S(F; λ0)."""
    return 2.0 * lam0 * (np.sqrt(1.0 + np.asarray(f, dtype=float) ** 2 / lam0) - 1.0)


def run_one(job: tuple[str, int, int, int, str]) -> str:
    couplings, width, steps, seed, work = job
    out = Path(work) / f"{kg.slug(couplings)}_w{width}_s{seed}.npz"
    if out.exists():
        return f"cached {out.name}"
    import torch

    from tycho.tasks import make_task
    from tycho.tasks.base import torch_generator
    from tycho.tasks.patterns import pattern_bits
    from tycho.train.runner import LeanOptimizer
    from tycho.train.spec import TrainConfig

    torch.set_num_threads(1)
    cfg = {"couplings": couplings, "data_seed": seed, "depth": 1, "exact_metrics": True, "n_tokens": 200_000,
           "shift_rho": -0.5, "train_context": "iid", "width": width, "window": 6}
    train = TrainConfig(steps=steps, batch_size=64, weight_decay=0.0, optimizer="sgd", lr=0.03, momentum=0.0,
                        n_log_evals=80, n_lin_evals=0, ntk_every=0)
    task = make_task("markov_lang", cfg, seed)
    model = task.build_model()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = LeanOptimizer(params, train)
    gen = torch_generator(seed, "data")
    x_all = task._model_inputs(pattern_bits(task.window))
    phi, _ = kg.hp.features(task.window)
    chi = torch.as_tensor(phi[:, kg.MASK_13], dtype=torch.float32)
    singles = [part.split(":")[0] for part in couplings.split(";") if "," not in part.split(":")[0]]
    psi = torch.as_tensor(phi[:, 1 << (int(singles[0]) - 1)], dtype=torch.float32) if singles else None
    hidden, read = model.net[0], model.net[2]

    def invariant() -> tuple[np.ndarray, float, float]:  # a_j² − 2 (‖w_j‖² + b_j²) per unit; Σ a_j²; Σ ‖w̃_j‖²
        a = (read.weight[1] - read.weight[0]).detach().to(torch.float64)
        w2 = (hidden.weight.detach().to(torch.float64) ** 2).sum(1) + hidden.bias.detach().to(torch.float64) ** 2
        return (a * a - 2.0 * w2).numpy(), float((a * a).sum()), float(w2.sum())

    inv0 = invariant()[0]
    log_steps = sorted(set(np.unique(np.round(np.geomspace(1, steps, 160)).astype(int)).tolist()) | {0, steps})
    rows: dict[str, list] = {k: [] for k in ("z", "kernel", "kernel_psi", "sum_a2", "sum_w2", "drift_mean",
                                             "drift_max")}
    step_set = set(log_steps)
    for step in range(steps + 1):
        if step in step_set:
            logits = model(x_all)
            z = logits[:, 1] - logits[:, 0]
            g = torch.autograd.grad((z * chi).mean(), params, retain_graph=psi is not None)
            rows["kernel"].append(float(sum((t.to(torch.float64) ** 2).sum() for t in g)))
            if psi is not None:
                g = torch.autograd.grad((z * psi).mean(), params)
                rows["kernel_psi"].append(float(sum((t.to(torch.float64) ** 2).sum() for t in g)))
            else:
                rows["kernel_psi"].append(float("nan"))
            rows["z"].append(z.detach().to(torch.float64).numpy())
            inv, a2, w2 = invariant()
            rows["sum_a2"].append(a2)
            rows["sum_w2"].append(w2)
            rows["drift_mean"].append(float(np.abs(inv - inv0).mean()))
            rows["drift_max"].append(float(np.abs(inv - inv0).max()))
        if step == steps:
            break
        x, y = task.sample_batch(train.batch_size, gen)
        loss = task.loss(model(x), y)
        opt.step(list(torch.autograd.grad(loss, params)), train.lr)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, steps=np.asarray(log_steps), invariant_abs_mean_at_step_0=float(np.abs(inv0).mean()),
                        **{k: np.asarray(v) for k, v in rows.items()})
    return f"done {couplings} width {width} seed {seed}"


def frozen_directions(seed: int, width: int) -> dict:
    """The kernel of the term's direction as a function of its coefficient if every unit kept its direction
    (module docstring), from the network at initialisation: the coefficient of F² for small F, and the kernel at
    F_POINTS above the initial coefficient."""
    import torch

    from tycho.tasks import make_task
    from tycho.tasks.patterns import pattern_bits

    torch.set_num_threads(1)
    cfg = {"couplings": ALONE, "data_seed": seed, "depth": 1, "exact_metrics": True, "n_tokens": 200_000,
           "shift_rho": -0.5, "train_context": "iid", "width": width, "window": 6}
    task = make_task("markov_lang", cfg, seed)
    model = task.build_model()
    x = task._model_inputs(pattern_bits(task.window)).to(torch.float64)
    phi, _ = kg.hp.features(task.window)
    chi = torch.as_tensor(phi[:, kg.MASK_13], dtype=torch.float64)
    hidden, read = model.net[0], model.net[2]
    pre = x @ hidden.weight.detach().to(torch.float64).T + hidden.bias.detach().to(torch.float64)
    gate = (pre > 0).to(torch.float64)
    g = (pre.clamp(min=0) * chi[:, None]).mean(0).numpy()                       # per unit
    grad_w = torch.einsum("su,si,s->ui", gate, x, chi) / x.shape[0]
    n = ((grad_w ** 2).sum(1) + ((gate * chi[:, None]).mean(0)) ** 2).numpy()
    a = (read.weight[1] - read.weight[0]).detach().to(torch.float64).numpy()
    live = n > 0                                         # a unit whose gate never opens has no share in the kernel
    g, n, a = g[live], n[live], a[live]
    share, q0, const = 2.0 * g * g + n * a * a, a * g, np.abs(n * a * a - 2.0 * g * g)
    lam0 = float(share.sum())
    omega = np.sqrt(8.0 * n)
    theta = np.arcsinh(omega * q0 / const)
    tau = np.linspace(0.0, 12.0, 24_001)[:, None]
    coef = (const / omega * np.sinh(omega * tau + theta)).sum(1) - float(q0.sum())
    kernel = (const * np.cosh(omega * tau + theta)).sum(1)
    return {"kernel_at_step_0": lam0, "small_f_coefficient": float(4.0 * (n * share).sum() / lam0 ** 2),
            "mean_n": float(n.mean()), "share_weighted_n": float((n * share).sum() / lam0),
            "kernel": {str(f): float(np.interp(f, coef, kernel)) for f in F_POINTS}}


def two_direction_flow(seed: int, width: int, couplings: str, lr: float = 0.03, max_steps: int = 14_000,
                       stop_at: float = 0.5) -> dict:
    """Gradient descent from the network at initialisation when the residual has two directions only: the
    lower-order input ψ (if the source has one) and the term χ, with the residuals of ``kernel_gate_pilot.residuals``
    at the current pair of coefficients. Every unit moves freely, so its gate pattern can change. Returns the
    step, the term's coefficient and the kernel in its direction, up to ``stop_at`` of the term's target."""
    import torch

    from tycho.tasks import make_task
    from tycho.tasks.patterns import pattern_bits

    torch.set_num_threads(1)
    cfg = {"couplings": ALONE, "data_seed": seed, "depth": 1, "exact_metrics": True, "n_tokens": 200_000,
           "shift_rho": -0.5, "train_context": "iid", "width": width, "window": 6}
    task = make_task("markov_lang", cfg, seed)             # the initial network does not depend on the source
    model = task.build_model()
    x = task._model_inputs(pattern_bits(task.window)).to(torch.float64).numpy()
    phi, _ = kg.hp.features(task.window)
    chi = phi[:, kg.MASK_13]
    singles = [part.split(":")[0] for part in couplings.split(";") if "," not in part.split(":")[0]]
    psi = phi[:, 1 << (int(singles[0]) - 1)] if singles else None
    beta1 = float(couplings.split(";")[0].split(":")[1]) if singles else 0.0
    w = model.net[0].weight.detach().to(torch.float64).numpy().copy()
    b = model.net[0].bias.detach().to(torch.float64).numpy().copy()
    a = (model.net[2].weight[1] - model.net[2].weight[0]).detach().to(torch.float64).numpy().copy()
    target, n = 2.0 * kg.TERM_STRENGTH, x.shape[0]
    steps, coef, kernel = [], [], []
    for step in range(max_steps + 1):
        pre = x @ w.T + b
        act, gate = np.maximum(pre, 0.0), (pre > 0).astype(float)
        g_chi = act.T @ chi / n
        dw_chi, db_chi = (gate * chi[:, None]).T @ x / n, gate.T @ chi / n
        f = float(a @ g_chi)
        steps.append(step)
        coef.append(f)
        kernel.append(float((2.0 * g_chi ** 2 + a * a * ((dw_chi ** 2).sum(1) + db_chi ** 2)).sum()))
        if f >= stop_at * target or step == max_steps:
            break
        if psi is None:
            r_chi = 0.5 * (math.tanh(target / 2.0) - math.tanh(f / 2.0))
            da, dw, db = 2.0 * r_chi * g_chi, r_chi * dw_chi, r_chi * db_chi
        else:
            g_psi = act.T @ psi / n
            r_psi, r_chi = kg.residuals(float(a @ g_psi), f, beta1)
            da = 2.0 * (r_psi * g_psi + r_chi * g_chi)
            dw = r_psi * (gate * psi[:, None]).T @ x / n + r_chi * dw_chi
            db = r_psi * gate.T @ psi / n + r_chi * db_chi
        a, w, b = a + lr * da, w + lr * a[:, None] * dw, b + lr * a * db
    return {"steps": np.asarray(steps, dtype=float), "coef": np.asarray(coef), "kernel": np.asarray(kernel)}


def flow_job(job: tuple[int, int, str, float, str]) -> str:
    seed, width, couplings, stop_at, work = job
    out = Path(work) / f"flow_{kg.slug(couplings)}_w{width}_s{seed}.npz"
    if not out.exists():
        np.savez_compressed(out, **two_direction_flow(seed, width, couplings, stop_at=stop_at))
    return out.name


def first_step_at(steps: np.ndarray, y: np.ndarray, level: float) -> float:
    """The step at which y first reaches the level (linear interpolation)."""
    hit = np.nonzero(y >= level)[0]
    if len(hit) == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(steps[0])
    return float(steps[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (steps[i] - steps[i - 1]))


def load(work: Path, couplings: str, width: int, seed: int, phi: np.ndarray) -> dict:
    with np.load(work / f"{kg.slug(couplings)}_w{width}_s{seed}.npz") as f:
        d = {k: f[k] for k in f.files}
    singles = [part.split(":")[0] for part in couplings.split(";") if "," not in part.split(":")[0]]
    d["coef"] = (d["z"] * phi[:, kg.MASK_13][None, :]).mean(1)
    d["low"] = (d["z"] * phi[:, 1 << (int(singles[0]) - 1)][None, :]).mean(1) if singles else None
    return d


def at_level(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """y where x first rises through the level (linear interpolation along the trajectory)."""
    hit = np.nonzero(x >= level)[0]
    if len(hit) == 0 or hit[0] == 0:
        return float("nan")
    i = int(hit[0])
    return float(y[i - 1] + (level - x[i - 1]) / (x[i] - x[i - 1]) * (y[i] - y[i - 1]))


def stats(xs) -> dict:
    xs = np.asarray([x for x in xs if not math.isnan(x)], dtype=float)
    if len(xs) == 0:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"median": float(np.median(xs)), "min": float(xs.min()), "max": float(xs.max()), "n": len(xs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "kernel_law")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "kernel_law_pilot.json")
    args = ap.parse_args(argv)
    todo = [(c, w, n, s, str(args.work)) for c, w, n in JOBS for s in SEEDS]
    with Pool(args.jobs) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_one, todo), 1):
            if i % 16 == 0:
                print(f"[{i}/{len(todo)}] {msg}", flush=True)
    phi, _ = kg.hp.features(6)

    # statement 2: the kernel of the term alone against the reviewer's law, by width
    law = {}
    for width in WIDTHS:
        runs = [load(args.work, ALONE, width, s, phi) for s in SEEDS]
        lam0 = [float(r["kernel"][0]) for r in runs]
        by_f = {}
        for f in F_POINTS:
            measured = [at_level(r["coef"], r["kernel"], f) for r in runs]
            by_f[str(f)] = {"measured": stats(measured),
                            "law": stats([l0 + float(growth(f, l0)) for l0 in lam0]),
                            "measured_over_law": stats([m / (l0 + float(growth(f, l0)))
                                                        for m, l0 in zip(measured, lam0)]),
                            "growth_over_f_squared": stats([(m - l0) / (f * f) for m, l0 in zip(measured, lam0)]),
                            "law_growth_over_f_squared": stats([float(growth(f, l0)) / (f * f) for l0 in lam0])}
        frozen = [frozen_directions(s, width) for s in SEEDS]
        for f in F_POINTS:
            start = [float(r["coef"][0]) for r in runs]  # the measured kernel at F above the run's own start
            measured = [at_level(r["coef"] - c0, r["kernel"], f) for r, c0 in zip(runs, start)]
            by_f[str(f)]["measured_over_frozen_directions"] = stats([m / fr["kernel"][str(f)]
                                                                    for m, fr in zip(measured, frozen)])
        law[str(width)] = {"kernel_at_step_0": stats(lam0), "by_coefficient": by_f,
                           "frozen_directions": {
                               "kernel_at_step_0_from_the_units": stats([fr["kernel_at_step_0"] for fr in frozen]),
                               "small_f_coefficient": stats([fr["small_f_coefficient"] for fr in frozen]),
                               "mean_n": stats([fr["mean_n"] for fr in frozen]),
                               "share_weighted_n": stats([fr["share_weighted_n"] for fr in frozen])}}

    # statement 1: the drift of the per-unit invariant, relative to its mean size at step 0
    drift = {}
    for width in WIDTHS:
        runs = [load(args.work, ALONE, width, s, phi) for s in SEEDS]
        drift[str(width)] = {
            f"at {int(p * 100)} % of the term": stats([at_level(r["coef"], r["drift_mean"], p * 1.6)
                                                       / float(r["invariant_abs_mean_at_step_0"]) for r in runs])
            for p in (0.25, 0.5, 0.75)}
        drift[str(width)]["at the last step"] = stats([float(r["drift_mean"][-1])
                                                       / float(r["invariant_abs_mean_at_step_0"]) for r in runs])
        drift[str(width)]["invariant_abs_mean_at_step_0"] = stats([float(r["invariant_abs_mean_at_step_0"])
                                                                  for r in runs])

    # statement 3: the read-out norm against the first-order coefficient while it is being learned
    readout = {}
    for couplings in FIRST_ORDER:
        runs = [load(args.work, couplings, 128, s, phi) for s in SEEDS]
        lam1 = [float(r["kernel_psi"][0]) for r in runs]
        by_f1 = {}
        for f1 in F1_POINTS:
            if f1 > 2.0 * float(couplings.split(";")[0].split(":")[1]) * 0.9:
                continue                                  # beyond what this source reaches before the term grows
            a2 = [at_level(r["low"], r["sum_a2"], f1) - float(r["sum_a2"][0]) for r in runs]
            w2 = [at_level(r["low"], r["sum_w2"], f1) - float(r["sum_w2"][0]) for r in runs]
            kp = [at_level(r["low"], r["kernel_psi"], f1) for r in runs]
            term = [at_level(r["low"], r["coef"], f1) for r in runs]
            by_f1[str(f1)] = {
                "read_out_growth": stats(a2), "first_layer_growth_times_2": stats([2.0 * x for x in w2]),
                "reviewer": stats([float(growth(f1, l1)) for l1 in lam1]), "f1_squared": f1 * f1,
                "half_gates_open": stats([2.0 * (math.sqrt(l1 * l1 + 2.0 * f1 * f1) - l1) for l1 in lam1]),
                "read_out_growth_over_half_gates_open": stats(
                    [x / (2.0 * (math.sqrt(l1 * l1 + 2.0 * f1 * f1) - l1)) for x, l1 in zip(a2, lam1)]),
                "kernel_over_its_start_half_gates_open": stats([math.sqrt(1.0 + 2.0 * f1 * f1 / (l1 * l1))
                                                               for l1 in lam1]),
                "read_out_growth_over_reviewer": stats([x / float(growth(f1, l1)) for x, l1 in zip(a2, lam1)]),
                "read_out_growth_over_f1_squared": stats([x / (f1 * f1) for x in a2]),
                "kernel_in_the_first_order_direction": stats(kp),
                "kernel_in_the_first_order_direction_over_its_start": stats([x / l1 for x, l1 in zip(kp, lam1)]),
                "term_coefficient_then": stats(term)}
        readout[couplings] = {"kernel_at_step_0_in_the_first_order_direction": stats(lam1),
                              "read_out_norm_at_step_0": stats([float(r["sum_a2"][0]) for r in runs]),
                              "by_first_order_coefficient": by_f1}

    # second round: the kernel's contribution K at widths 64 and 256, against the reviewer's 40 values
    gate_pilot = json.loads((REPO / "docs" / "results" / "v2_analysis" / "kernel_gate_pilot.json").read_text())

    def k_at_128(strength: str, lag: str, k: str) -> float:
        return gate_pilot["strength"][strength]["closed_form"][
            "steps_over_alone_divided_by_the_delay_from_the_residual_alone"][lag][k]

    flow_widths = (64, 128, 256)
    flow_jobs = ([(s, w, ALONE, 0.95, str(args.work)) for w in WIDTHS for s in SEEDS]
                 + [(s, w, kg.WITH[st][lag], 0.5, str(args.work)) for w in flow_widths for st in K_STRENGTHS
                    for lag in ("lag 1", "lag 4") for s in SEEDS])
    with Pool(args.jobs) as pool:
        for _ in pool.imap_unordered(flow_job, flow_jobs, chunksize=4):
            pass

    def flow(couplings: str, width: int, seed: int) -> dict:
        with np.load(args.work / f"flow_{kg.slug(couplings)}_w{width}_s{seed}.npz") as f:
            return {k: f[k] for k in f.files}

    target = 2.0 * kg.TERM_STRENGTH
    flow_alone = {w: [flow(ALONE, w, s) for s in SEEDS] for w in WIDTHS}
    sgd_alone = {w: [load(args.work, ALONE, w, s, phi) for s in SEEDS] for w in WIDTHS}
    # the flow against the measured kernel of the term alone (the reviewer's first check of the flow)
    flow_check = {str(w): {str(f): stats([at_level(r["coef"], r["kernel"], f) / float(np.interp(
        f, fl["coef"], fl["kernel"])) for r, fl in zip(sgd_alone[w], flow_alone[w])]) for f in F_POINTS}
        for w in WIDTHS}

    cells = {}
    for width in flow_widths:
        for strength in K_STRENGTHS:
            closed = kg.closed_form(float(strength))["progress"]
            for i, (lag, k) in enumerate(CELLS):
                level = float(k.split()[0]) / 100.0 * target
                delay = closed[k]["delay_from_the_residual_alone"]
                ratios, factors = [], []
                for fa, s in zip(flow_alone[width], SEEDS):
                    fw = flow(kg.WITH[strength][lag], width, s)
                    ta = first_step_at(fa["steps"], fa["coef"], level)
                    tw = first_step_at(fw["steps"], fw["coef"], level)
                    ratios.append(tw / ta)
                    factors.append(float(np.interp(ta, fa["steps"], fa["kernel"]))
                                   / float(np.interp(tw, fw["steps"], fw["kernel"])))
                cell = {"flow": float(np.median(ratios)) / delay, "kernel_factor_flow": float(np.median(factors)),
                        "width_128_measured": k_at_128(strength, lag, k)}
                if width == 128:
                    cell["measured"] = cell["width_128_measured"]
                    cell["kernel_factor_measured"] = gate_pilot["strength"][strength][lag][
                        "kernel_alone_over_with"][k]["median"]
                else:
                    ratios, factors = [], []
                    for ra, s in zip(sgd_alone[width], SEEDS):
                        rw = load(args.work, kg.WITH[strength][lag], width, s, phi)
                        ta = kg.crossing(ra["steps"].astype(float), ra["coef"], level)
                        tw = kg.crossing(rw["steps"].astype(float), rw["coef"], level)
                        ratios.append(tw / ta)
                        factors.append(float(np.interp(ta, ra["steps"], ra["kernel"]))
                                       / float(np.interp(tw, rw["steps"], rw["kernel"])))
                    cell.update({"measured": float(np.median(ratios)) / delay,
                                 "measured_min_max_over_seeds": [float(min(ratios)) / delay,
                                                                 float(max(ratios)) / delay],
                                 "kernel_factor_measured": float(np.median(factors)),
                                 "reviewer_flow": REVIEWER_FLOW[width][strength][i],
                                 "reviewer_formula": REVIEWER_FORMULA[width][strength][i],
                                 "plain_scaling": 1.0 / (1.0 + 128.0 / width * (1.0 / cell["width_128_measured"]
                                                                                 - 1.0))})
                cells[f"{width} {strength} {lag} {k}"] = cell

    def k_errors(widths: tuple[int, ...], which: str, strengths: tuple[str, ...] = K_STRENGTHS) -> dict:
        e = [abs(c[which] - c["measured"]) for key, c in cells.items()
             if int(key.split()[0]) in widths and key.split()[1] in strengths]
        return {"mean_abs_error": float(np.mean(e)), "max_abs_error": float(np.max(e)), "n": len(e)}

    candidates = ("reviewer_flow", "reviewer_formula", "flow", "plain_scaling", "width_128_measured")

    def k_of(width: int, strength: str, lag: str, k: str) -> float:
        return cells[f"{width} {strength} {lag} {k}"]["measured"]

    strong = ("1.0", "1.5", "2.0", "2.5")
    criteria = {
        "shared_input_rises_with_width": bool(all(k_of(64, st, "lag 1", k) < k_of(128, st, "lag 1", k)
                                                  < k_of(256, st, "lag 1", k) for st in strong
                                                  for k in ("25 %", "50 %"))),
        "shared_input_strength_2.0_at_25": {"width_64": k_of(64, "2.0", "lag 1", "25 %"), "at_most": 0.48,
                                            "fails_above": 0.50, "width_256": k_of(256, "2.0", "lag 1", "25 %"),
                                            "at_least": 0.78},
        "not_shared_width_64_mean_shift": {"value": float(np.mean([abs(k_of(64, st, "lag 4", k)
                                                                       - k_of(128, st, "lag 4", k))
                                                                   for st in strong for k in ("25 %", "50 %")])),
                                           "at_most": 0.045},
        "shared_width_64_mean_shift": float(np.mean([k_of(128, st, "lag 1", k) - k_of(64, st, "lag 1", k)
                                                     for st in strong for k in ("25 %", "50 %")])),
        "not_shared_width_256_rise_at_2.0_and_2.5": {
            "values": [k_of(256, st, "lag 4", k) - k_of(128, st, "lag 4", k) for st in ("2.0", "2.5")
                       for k in ("25 %", "50 %")], "at_least": 0.08}}
    k_by_width = {
        "cells": cells,
        "errors": {name: {which: k_errors(widths, which) for which in candidates}
                   for name, widths in (("width 64", (64,)), ("width 256", (256,)), ("both", (64, 256)))},
        "errors_at_strengths_1.0_and_2.0": {which: k_errors((64, 256), which, ("1.0", "2.0")) for which in candidates},
        "reviewer_flow_minus_flow_here": {
            "mean_abs": float(np.mean([abs(c["reviewer_flow"] - c["flow"]) for c in cells.values()
                                       if "reviewer_flow" in c])),
            "max_abs": float(np.max([abs(c["reviewer_flow"] - c["flow"]) for c in cells.values()
                                     if "reviewer_flow" in c]))},
        "flow_at_width_128": {"K": k_errors((128,), "flow"),
                              "kernel_factor": {"mean_abs_error": float(np.mean(
                                  [abs(c["kernel_factor_flow"] - c["kernel_factor_measured"])
                                   for key, c in cells.items() if key.startswith("128 ")]))}},
        "kernel_factor_errors_of_the_flow": {str(w): float(np.mean(
            [abs(c["kernel_factor_flow"] - c["kernel_factor_measured"]) for key, c in cells.items()
             if key.startswith(f"{w} ")])) for w in (64, 256)},
        "passes_if": "mean abs error <= 0.025 over the 40 values, clearly below that of width 128's values",
        "fails_if": "mean abs error > 0.035, or K with the shared input at width 64 above 0.50 for strength 2.0 at "
                    "25 %, or K at width 256 not rising towards 1",
        "criteria": criteria}

    result = {"status": "exploratory pilot; local runs; not independently recomputed", "seeds": list(SEEDS),
              "statement_2_kernel_law_by_width": law, "statement_1_invariant_drift": drift,
              "statement_3_read_out_growth": readout,
              "second_round_flow_against_the_kernel_of_the_term_alone": flow_check,
              "second_round_K_by_width": k_by_width}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    print("statement 2: kernel of the term alone, measured / law (median over seeds [min, max]) at F =", F_POINTS)
    for width, blk in law.items():
        print(f"  width {width:>3s}: kernel at step 0 {blk['kernel_at_step_0']['median']:.4f} | "
              + " | ".join(f"{v['measured_over_law']['median']:.3f} [{v['measured_over_law']['min']:.3f}, "
                           f"{v['measured_over_law']['max']:.3f}]" for v in blk["by_coefficient"].values()))
        print("             growth / F², measured (law): "
              + " | ".join(f"{v['growth_over_f_squared']['median']:.3f} "
                           f"({v['law_growth_over_f_squared']['median']:.3f})"
                           for v in blk["by_coefficient"].values()))
        fr = blk["frozen_directions"]
        print(f"             frozen directions: coefficient of F² {fr['small_f_coefficient']['median']:.3f} "
              f"(mean n {fr['mean_n']['median']:.4f}, share-weighted {fr['share_weighted_n']['median']:.4f}; kernel "
              f"at step 0 from the units {fr['kernel_at_step_0_from_the_units']['median']:.4f}); measured / frozen: "
              + " | ".join(f"{v['measured_over_frozen_directions']['median']:.3f}"
                           for v in blk["by_coefficient"].values()))
    print("statement 1: drift of the per-unit invariant over its mean size at step 0 (median over seeds)")
    for width, blk in drift.items():
        print(f"  width {width:>3s}: " + "; ".join(f"{k} {v['median']:.4f}" for k, v in blk.items()))
    print("statement 3: growth of the read-out norm at first-order coefficient F_1: measured / reviewer's S / F_1² "
          "(ratio to S; ratio to F_1²) | kernel in that direction over its start | the term's coefficient then")
    for couplings, blk in readout.items():
        print(f"  {couplings}: kernel at step 0 {blk['kernel_at_step_0_in_the_first_order_direction']['median']:.3f}, "
              f"read-out norm at step 0 {blk['read_out_norm_at_step_0']['median']:.3f}")
        for f1, v in blk["by_first_order_coefficient"].items():
            print(f"     F_1 {f1}: {v['read_out_growth']['median']:.3f} / {v['reviewer']['median']:.3f} / "
                  f"{v['f1_squared']:.2f} ({v['read_out_growth_over_reviewer']['median']:.3f}; "
                  f"{v['read_out_growth_over_f1_squared']['median']:.3f}); half the gates open "
                  f"{v['half_gates_open']['median']:.3f} ({v['read_out_growth_over_half_gates_open']['median']:.3f}) | "
                  f"{v['kernel_in_the_first_order_direction_over_its_start']['median']:.3f} | "
                  f"{v['term_coefficient_then']['median']:.3f} | 2 x first-layer growth "
                  f"{v['first_layer_growth_times_2']['median']:.3f}")
    print("second round: kernel of the term alone, measured (SGD) over the two-direction flow, at F =", F_POINTS)
    for width, blk in flow_check.items():
        print(f"  width {width:>3s}: " + " | ".join(f"{v['median']:.3f} [{v['min']:.3f}, {v['max']:.3f}]"
                                                    for v in blk.values()))
    print("second round: K measured / reviewer's flow / reviewer's formula / flow here / plain scaling / width 128")
    for key, c in cells.items():
        if "reviewer_flow" in c:
            print(f"  {key}: {c['measured']:.4f} [{c['measured_min_max_over_seeds'][0]:.3f}, "
                  f"{c['measured_min_max_over_seeds'][1]:.3f}] / {c['reviewer_flow']:.4f} / "
                  f"{c['reviewer_formula']:.4f} / {c['flow']:.4f} / {c['plain_scaling']:.4f} / "
                  f"{c['width_128_measured']:.4f}")
    for name, by in k_by_width["errors"].items():
        print(f"  {name}: " + "; ".join(f"{which} {v['mean_abs_error']:.4f} (max {v['max_abs_error']:.4f})"
                                         for which, v in by.items()))
    print("  strengths 1.0 and 2.0 only:", "; ".join(
        f"{which} {v['mean_abs_error']:.4f}" for which, v in k_by_width["errors_at_strengths_1.0_and_2.0"].items()))
    print("  reviewer's flow minus the flow here:", json.dumps(k_by_width["reviewer_flow_minus_flow_here"]),
          "| flow at width 128:", json.dumps(k_by_width["flow_at_width_128"]))
    print("  criteria:", json.dumps(criteria))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
