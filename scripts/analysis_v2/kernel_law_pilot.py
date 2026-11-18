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

Runs: the Markov settings of ``kernel_gate_pilot.py`` (SGD, learning rate 0.03, batch 64, iid contexts, seeds
600–607). The term alone for 12,000 steps at widths 32, 64, 128, 256 and 512; the three sources with a first-order
term for 3,000 steps at width 128.

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
JOBS = [(ALONE, w, 12_000) for w in WIDTHS] + [(c, 128, 3_000) for c in FIRST_ORDER]
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

    result = {"status": "exploratory pilot; local runs; not independently recomputed", "seeds": list(SEEDS),
              "statement_2_kernel_law_by_width": law, "statement_1_invariant_drift": drift,
              "statement_3_read_out_growth": readout}
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
