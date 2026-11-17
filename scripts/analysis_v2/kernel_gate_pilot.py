"""Pilot (exploratory; local runs): what is left of the delay after the output gain, and why the lag matters.

``delay_sweep_pilot.py`` found two things that the gain rule does not cover. Under plain SGD a lower-order term still
delays a higher-order one by 1.02 to 1.39 after time is weighted by the output gain. And the delay is longer when the
lower-order term sits on an input that the higher-order term does not use (lag 4 against lag 1, for the term {1,3}).

The reviewer's proposal (reply of 17 November to commit 41d4d7a, on record before these runs): a ReLU unit whose gate
is set by a strong first-order input s_a is linear in the other inputs. If a is one of the higher-order term's own
inputs (a = 1), the unit still produces s_1 s_3 at first order in its weight on s_3; if it is not (a = 4), the unit
produces no s_1 s_3 at all ("dead capacity"). So the tangent kernel in the {1,3} direction should be lower in the
lag-4 twin. Predictions: at step 150 the kernel of 1:2.0;1,3:0.8 is at least 1.20 times that of 4:2.0;1,3:0.8 (the
proposal fails below 1.05), and with the kernel in the clock the ratio of the two twins' delays at 50 % (1.27 in
gain-weighted time) shrinks to within [0.95, 1.08].

What is measured. Write F for the coefficient of the log-odds z in the Walsh direction χ of {1,3} (its target is the
term's strength, 1.6, whatever the lower-order term), λ = ‖∇_θ F‖² for the tangent kernel in that direction and
ρ = E[(p* − q) χ] for the part of the probability residual in that direction. Under gradient flow
dF/dt = lr · (λ ρ + cross terms from the other directions), so at equal progress F the rate of two runs differs by a
kernel factor and a residual factor. The residual factor is where the output gain acts; it is fixed by the source and
the progress. The kernel factor is what the gain rule leaves out. The script reads both at 10, 25, 50 and 75 % of
the target, the exact rate (the drift of F under the population gradient) to check the factorisation, and three
statistics of the hidden units' gates: the share of (unit, setting of the other inputs) pairs whose gate flips with
s_3, and the share of units whose gate is a function of s_1 alone or of s_4 alone. The kernel is also split by
parameter group. Its part from the first layer's weights on s_3 is Σ_j a_j² E[gate_j · s_1]²: a unit turns a weight
on s_3 into s_1 s_3 to the extent that its gate is correlated with s_1, which needs no gate to be set by s_1 alone.

Second round (the reviewer's third reply, to commit b656849): a closed form for the residual factor, from the source
alone (``closed_form``). The script compares it with the measured residual factor and with the lower-order
coefficient, at the two strengths of the first round and at two more (0.5 and 1.5) that were run after the formula's
values were on record, and divides the measured delay by the delay the residual alone would give: what is left is
the kernel's contribution.

Runs: sky v2a's Markov settings with plain SGD (learning rate 0.03, no momentum, batch 64), iid contexts; the term
{1,3}:0.8 alone and with {1} or {4} at strengths 0.5, 1.0, 1.5 and 2.0; seeds 600–607; 12,000 steps. The loop
mirrors the runner's update: on 4:2.0;1,3:0.8, seed 600, it reproduces the log-odds of the
``delay_sweep_pilot.py`` run at its 46 checkpoints up to step 3,000 to 1.2e-6.

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/kernel_gate_pilot.py
--jobs 4``. Runs are cached in ``work/kernel_gate``. Writes ``docs/results/v2_analysis/kernel_gate_pilot.json``.
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
_spec = importlib.util.spec_from_file_location("hierarchy_projections",
                                               Path(__file__).with_name("hierarchy_projections.py"))
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)

SEEDS = tuple(range(600, 608))
STEPS = 12_000
ALONE = "1,3:0.8"
#: Strengths 1.0 and 2.0 are the first round. 0.5 and 1.5 were added for the closed form's prediction (second round).
WITH = {"0.5": {"lag 1": "1:0.5;1,3:0.8", "lag 4": "4:0.5;1,3:0.8"},
        "1.0": {"lag 1": "1:1.0;1,3:0.8", "lag 4": "4:1.0;1,3:0.8"},
        "1.5": {"lag 1": "1:1.5;1,3:0.8", "lag 4": "4:1.5;1,3:0.8"},
        "2.0": {"lag 1": "1:2.0;1,3:0.8", "lag 4": "4:2.0;1,3:0.8"}}
TERM_STRENGTH = 0.8
SOURCES = (ALONE,) + tuple(c for by_lag in WITH.values() for c in by_lag.values())
MASK_13 = 0b000101                                     # the Walsh direction of the term {1,3}: lags 1 and 3
PROGRESS = (0.1, 0.25, 0.5, 0.75)
GATES = ("gate_flips_with_s3", "gate_set_by_s1", "gate_set_by_s4")
#: The kernel split by parameter group: the first layer's weights on each input, its biases, the read-out layer.
PARTS = ("weights_on_s1", "weights_on_s3", "weights_on_s4", "weights_on_other_inputs", "hidden_biases", "read_out")


def slug(couplings: str) -> str:
    return couplings.replace(";", "_").replace(":", "-").replace(",", "")


def run_one(job: tuple[str, int, str]) -> str:
    couplings, seed, work = job
    out = Path(work) / f"{slug(couplings)}_s{seed}.npz"
    if out.exists():
        return f"cached {out.name}"
    import torch
    import torch.nn.functional as F

    from tycho.tasks import make_task
    from tycho.tasks.base import torch_generator
    from tycho.tasks.patterns import pattern_bits
    from tycho.train.runner import LeanOptimizer
    from tycho.train.spec import TrainConfig

    torch.set_num_threads(1)
    cfg = {"couplings": couplings, "data_seed": seed, "depth": 1, "exact_metrics": True, "n_tokens": 200_000,
           "shift_rho": -0.5, "train_context": "iid", "width": 128, "window": 6}
    train = TrainConfig(steps=STEPS, batch_size=64, weight_decay=0.0, optimizer="sgd", lr=0.03, momentum=0.0,
                        n_log_evals=80, n_lin_evals=0, ntk_every=0)
    task = make_task("markov_lang", cfg, seed)
    model = task.build_model()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = LeanOptimizer(params, train)
    gen = torch_generator(seed, "data")
    bits = pattern_bits(task.window)
    x_all = task._model_inputs(bits)                    # the 64 windows as ±1 inputs, in code order
    phi, _ = hp.features(task.window)
    chi = torch.as_tensor(phi[:, MASK_13], dtype=torch.float32)
    p_true = torch.as_tensor(task.pattern_truth(), dtype=torch.float32)
    pmf = torch.as_tensor(task.pattern_pmfs()["D"], dtype=torch.float32)
    codes = np.arange(64)
    lo3 = codes[(codes >> 2) & 1 == 0]                   # codes with s_3 = −1; their partner has s_3 = +1
    log_steps = sorted(set(np.unique(np.round(np.geomspace(1, STEPS, 140)).astype(int)).tolist()) | {0, 150, STEPS})
    rows: dict[str, list] = {k: [] for k in ("z", "kernel", "drift", "kernel_parts") + GATES}
    step_set = set(log_steps)
    for step in range(STEPS + 1):
        if step in step_set:
            logits = model(x_all)
            z = logits[:, 1] - logits[:, 0]
            g_f = torch.autograd.grad((z * chi).mean(), params, retain_graph=True)
            g_l = torch.autograd.grad((pmf * (F.softplus(z) - p_true * z)).sum(), params)   # the population loss
            rows["z"].append(z.detach().to(torch.float64).numpy())
            rows["kernel"].append(float(sum((g.to(torch.float64) ** 2).sum() for g in g_f)))
            col = (g_f[0].to(torch.float64) ** 2).sum(0).numpy()         # first-layer weights, by input
            rows["kernel_parts"].append([col[0], col[2], col[3], col[1] + col[4] + col[5],
                                         float((g_f[1].to(torch.float64) ** 2).sum()),
                                         float(sum((g.to(torch.float64) ** 2).sum() for g in g_f[2:]))])
            rows["drift"].append(float(-sum((a.to(torch.float64) * b.to(torch.float64)).sum()
                                            for a, b in zip(g_f, g_l))))
            gate = (model.net[0](x_all) > 0).numpy()     # (64 windows, units)
            rows["gate_flips_with_s3"].append(float((gate[lo3] != gate[lo3 | 4]).mean()))
            for name, bit in (("gate_set_by_s1", 0), ("gate_set_by_s4", 3)):
                off, on = gate[(codes >> bit) & 1 == 0], gate[(codes >> bit) & 1 == 1]
                same = (off.min(0) == off.max(0)) & (on.min(0) == on.max(0)) & (off[0] != on[0])
                rows[name].append(float(same.mean()))
        if step == STEPS:
            break
        x, y = task.sample_batch(train.batch_size, gen)
        loss = task.loss(model(x), y)
        opt.step(list(torch.autograd.grad(loss, params)), train.lr)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, steps=np.asarray(log_steps), truth=np.asarray(task.pattern_truth()),
                        **{k: np.asarray(v) for k, v in rows.items()})
    return f"done {couplings} seed {seed}"


def crossing(steps: np.ndarray, y: np.ndarray, level: float) -> float:
    """The step at which y last rises through the level (log-interpolated)."""
    below = np.nonzero(y < level)[0]
    if len(below) == 0 or below[-1] == len(y) - 1:
        return float("nan")
    i = int(below[-1]) + 1
    a, b = math.log(max(float(steps[i - 1]), 0.5)), math.log(float(steps[i]))
    return math.exp(a + (level - y[i - 1]) / (y[i] - y[i - 1]) * (b - a))


def closed_form(beta1: float, beta2: float = TERM_STRENGTH) -> dict:
    """The reviewer's closed form for the residual factor (third reply of 17 November), from the source alone.

    Log-odds z = F_1 ψ + F χ with ψ the lower-order input and χ the term's direction; targets F_1* = 2 β_1 and
    F* = 2 β_2. The means are m_1 = sinh F_1 / (cosh F_1 + cosh F) and m_2 = sinh F / (cosh F_1 + cosh F). While F
    grows, F_1 is taken to follow the value at which the first-order residual vanishes (m_1 = m_1*). Then the
    residual along χ is known at every progress r = F / F*, for the twin with the lower-order term and for the term
    alone, and so is their ratio. It starts at rule B and ends at (1 + cosh F_1* cosh F*) / (1 + cosh F*).
    Also returned: rule A, which is 1 / (1 − m_1*²), and the delay the residual alone would give if the kernel
    were the same in both twins and constant, ∫_0^r dr'/ρ_with over ∫_0^r dr'/ρ_alone.
    """
    f1, f2 = 2.0 * beta1, 2.0 * beta2
    den = math.cosh(f1) + math.cosh(f2)
    m1, m2 = math.sinh(f1) / den, math.sinh(f2) / den

    def point(r):
        c, sh = np.cosh(r * f2), np.sinh(r * f2)
        big_d = (c + np.sqrt(1.0 + m1 * m1 * sh * sh)) / (1.0 - m1 * m1)
        return np.arcsinh(m1 * big_d), 0.5 * (m2 - sh / big_d), 0.5 * (math.tanh(f2 / 2.0) - np.tanh(r * f2 / 2.0))

    grid = np.linspace(0.0, 0.999, 9991)
    _, rho_w, rho_a = point(grid)

    def cum(rho: np.ndarray) -> np.ndarray:             # ∫ dr / ρ along the grid, trapezoid rule
        return np.concatenate([[0.0], np.cumsum(0.5 * (1 / rho[1:] + 1 / rho[:-1]) * np.diff(grid))])

    cum_w, cum_a = cum(rho_w), cum(rho_a)
    out = {"rule_A": 1.0 / (1.0 - m1 * m1), "rule_B": math.tanh(f2 / 2.0) / m2,
           "residual_factor_at_the_end": (1.0 + math.cosh(f1) * math.cosh(f2)) / (1.0 + math.cosh(f2)),
           "first_order_coefficient_at_the_start": float(point(0.0)[0]), "first_order_target": f1, "progress": {}}
    for pr in PROGRESS:
        low, rw, ra = point(pr)
        out["progress"][f"{int(pr * 100)} %"] = {
            "first_order_coefficient": float(low), "residual_factor": float(ra / rw),
            "delay_from_the_residual_alone": float(np.interp(pr, grid, cum_w) / np.interp(pr, grid, cum_a))}
    return out


def load(work: Path, couplings: str, seed: int, phi: np.ndarray) -> dict:
    with np.load(work / f"{slug(couplings)}_s{seed}.npz") as f:
        d = {k: f[k] for k in f.files}
    steps, z, truth = d["steps"].astype(float), d["z"], d["truth"]
    chi = phi[:, MASK_13]
    singles = [part.split(":")[0] for part in couplings.split(";") if "," not in part.split(":")[0]]
    low = (z * phi[:, 1 << (int(singles[0]) - 1)][None, :]).mean(1) if singles else np.full(len(steps), np.nan)
    coef = (z * chi[None, :]).mean(1)                                    # F: the log-odds coefficient
    target = float((np.log(truth / (1.0 - truth)) * chi).mean())
    m = np.tanh(0.5 * z)
    rho = ((truth[None, :] - 0.5 * (1.0 + m)) * chi[None, :]).mean(1)    # the probability residual along chi
    gain = (1.0 - m ** 2).mean(1)
    walsh = (m * chi[None, :]).mean(1)                                   # the coefficient of the mean, as in the sweep

    def at(arr: np.ndarray, s: float) -> float:
        return float(np.interp(s, steps, arr))

    def integral(rate: np.ndarray, s: float) -> float:                   # trapezoid rule up to step s
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(steps))])
        return at(cum, s)

    parts = d["kernel_parts"]
    out = {"target": target, "final_over_target": float(coef[-1] / target),
           "at_step_150": {"kernel": at(d["kernel"], 150.0), "progress": at(coef, 150.0) / target,
                           **{g: at(d[g], 150.0) for g in GATES},
                           **{name: at(parts[:, i], 150.0) for i, name in enumerate(PARTS)}},
           "at_step_0": {"kernel": float(d["kernel"][0]), **{g: float(d[g][0]) for g in GATES},
                         **{name: float(parts[0, i]) for i, name in enumerate(PARTS)}}}
    for pr in PROGRESS:
        t = crossing(steps, coef, pr * target)
        out[pr] = {"t": t, "kernel": at(d["kernel"], t), "rho": at(rho, t), "drift": at(d["drift"], t),
                   "low": at(low, t),
                   **{g: at(d[g], t) for g in GATES}, **{name: at(parts[:, i], t) for i, name in enumerate(PARTS)}}
    # the sweep's own half-time (the mean's coefficient at half its final value) and three clocks at that time
    t_half = crossing(steps, walsh, 0.5 * float(np.mean(walsh[-5:])))
    out["sweep_half_time"] = {"steps": t_half, "gain": integral(gain, t_half),
                              "gain_x_kernel": integral(gain * d["kernel"], t_half)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "kernel_gate")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "kernel_gate_pilot.json")
    args = ap.parse_args(argv)
    with Pool(args.jobs) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_one, [(c, s, str(args.work)) for c in SOURCES for s in SEEDS]),
                                1):
            if i % 10 == 0:
                print(f"[{i}/{len(SOURCES) * len(SEEDS)}] {msg}", flush=True)
    phi, _ = hp.features(6)
    runs = {c: {s: load(args.work, c, s, phi) for s in SEEDS} for c in SOURCES}

    def stats(xs) -> dict:
        xs = np.asarray(xs, dtype=float)
        return {"median": float(np.median(xs)), "min": float(xs.min()), "max": float(xs.max()),
                "n_above_1": int((xs > 1.0).sum()), "n": len(xs)}

    def per_progress(fn) -> dict:
        return {f"{int(pr * 100)} %": stats([fn(s, pr) for s in SEEDS]) for pr in PROGRESS}

    a = runs[ALONE]
    result = {"status": "exploratory pilot; local runs; not independently recomputed",
              "reviewer_predictions": {
                  "kernel": "kernel of 1:2.0;1,3:0.8 over kernel of 4:2.0;1,3:0.8 at step 150 is at least 1.20; the "
                            "proposal fails below 1.05",
                  "clock": "with the kernel in the clock, the delay at 50 % of the lag-4 twin over that of the lag-1 "
                           "twin (1.27 in gain-weighted time) lies within [0.95, 1.08]",
                  "mechanism": "units whose gate is set by the lower-order input: first-order route to {1,3} when "
                               "that input is s_1, none when it is s_4",
                  "third reply, 2A": "the lower-order log-odds coefficient at 10/25/50/75 % of the term follows the "
                                     "closed form within 0.10 (strength 2.0) and 0.06 (strength 1.0), and at 50 and "
                                     "75 % it is larger in the lag-1 twin than in the lag-4 twin in at least 6 of 8 "
                                     "seeds (strength 2.0); fails beyond 0.15 or if lag 1 is not ahead",
                  "third reply, 2B": "new strengths: the residual factor follows the closed form within 8 % at "
                                     "strength 0.5 (each lag, all four points, rising) and within 10 % at strength "
                                     "1.5 (mean of the two lags), with lag 4 above lag 1 at 75 %; fails beyond 15 % "
                                     "at strength 1.5"},
              "seeds": list(SEEDS), "steps": STEPS,
              "final_over_target": {c: stats([runs[c][s]["final_over_target"] for s in SEEDS]) for c in SOURCES},
              "alone": {"kernel": {"at step 0": stats([a[s]["at_step_0"]["kernel"] for s in SEEDS]),
                                   "at step 150": stats([a[s]["at_step_150"]["kernel"] for s in SEEDS]),
                                   **per_progress(lambda s, pr: a[s][pr]["kernel"])},
                        "step": per_progress(lambda s, pr: a[s][pr]["t"]),
                        "rate_over_kernel_x_residual": per_progress(
                            lambda s, pr: a[s][pr]["drift"] / (a[s][pr]["kernel"] * a[s][pr]["rho"])),
                        "gates": {g: {"at step 0": stats([a[s]["at_step_0"][g] for s in SEEDS]),
                                      "at step 150": stats([a[s]["at_step_150"][g] for s in SEEDS]),
                                      "at 10 %": stats([a[s][0.1][g] for s in SEEDS])} for g in GATES},
                        "kernel_parts": {name: {"at step 0": stats([a[s]["at_step_0"][name] for s in SEEDS]),
                                                "at 10 %": stats([a[s][0.1][name] for s in SEEDS]),
                                                "at 25 %": stats([a[s][0.25][name] for s in SEEDS])}
                                         for name in PARTS}},
              "strength": {}}
    for strength, by_lag in WITH.items():
        w = {name: runs[c] for name, c in by_lag.items()}
        block = {"kernel_lag1_over_lag4": {
            "at step 150": stats([w["lag 1"][s]["at_step_150"]["kernel"] / w["lag 4"][s]["at_step_150"]["kernel"]
                                  for s in SEEDS]),
            **per_progress(lambda s, pr: w["lag 1"][s][pr]["kernel"] / w["lag 4"][s][pr]["kernel"])},
            "progress_at_step_150": {name: stats([w[name][s]["at_step_150"]["progress"] for s in SEEDS]) for name in w},
            "steps_lag4_over_lag1": per_progress(lambda s, pr: w["lag 4"][s][pr]["t"] / w["lag 1"][s][pr]["t"]),
            "sweep_half_time_lag4_over_lag1": {
                clock: stats([w["lag 4"][s]["sweep_half_time"][clock] / w["lag 1"][s]["sweep_half_time"][clock]
                              for s in SEEDS]) for clock in ("steps", "gain", "gain_x_kernel")}}
        for name, r in w.items():
            block[name] = {
                "steps_over_alone": per_progress(lambda s, pr: r[s][pr]["t"] / a[s][pr]["t"]),
                "rate_alone_over_with": per_progress(lambda s, pr: a[s][pr]["drift"] / r[s][pr]["drift"]),
                "residual_alone_over_with": per_progress(lambda s, pr: a[s][pr]["rho"] / r[s][pr]["rho"]),
                "kernel_alone_over_with": per_progress(lambda s, pr: a[s][pr]["kernel"] / r[s][pr]["kernel"]),
                "rate_over_kernel_x_residual": per_progress(
                    lambda s, pr: r[s][pr]["drift"] / (r[s][pr]["kernel"] * r[s][pr]["rho"])),
                "sweep_half_time_over_alone": {
                    clock: stats([r[s]["sweep_half_time"][clock] / a[s]["sweep_half_time"][clock] for s in SEEDS])
                    for clock in ("steps", "gain", "gain_x_kernel")},
                "gates": {g: {"at step 150": stats([r[s]["at_step_150"][g] for s in SEEDS]),
                              "at 10 %": stats([r[s][0.1][g] for s in SEEDS])} for g in GATES},
                "kernel_parts": {name: {"at 10 %": stats([r[s][0.1][name] for s in SEEDS]),
                                        "at 25 %": stats([r[s][0.25][name] for s in SEEDS])} for name in PARTS},
                "low_order_coefficient": per_progress(lambda s, pr: r[s][pr]["low"])}
        # the closed form for the residual factor, and what it leaves for the kernel
        cf = closed_form(float(strength))
        keys = [f"{int(pr * 100)} %" for pr in PROGRESS]
        measured = {k: 0.5 * (block["lag 1"]["residual_alone_over_with"][k]["median"]
                              + block["lag 4"]["residual_alone_over_with"][k]["median"]) for k in keys}
        ahead = {k: [w["lag 1"][s][pr]["low"] - w["lag 4"][s][pr]["low"] for s in SEEDS]
                 for k, pr in zip(keys, PROGRESS)}
        block["closed_form"] = {
            **cf,
            "residual_factor_measured_over_closed_form": {
                name: {k: block[name]["residual_alone_over_with"][k]["median"] / cf["progress"][k]["residual_factor"]
                       for k in keys} for name in ("lag 1", "lag 4")},
            "mean_of_the_two_lags_over_closed_form": {k: measured[k] / cf["progress"][k]["residual_factor"]
                                                      for k in keys},
            "low_order_coefficient_minus_closed_form": {
                name: {k: (block[name]["low_order_coefficient"][k]["median"]
                           - cf["progress"][k]["first_order_coefficient"]) for k in keys}
                for name in ("lag 1", "lag 4")},
            "low_order_coefficient_lag1_minus_lag4": {
                k: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)),
                    "n_above_0": int(sum(x > 0 for x in v)), "n": len(v)} for k, v in ahead.items()},
            "steps_over_alone_divided_by_the_delay_from_the_residual_alone": {
                name: {k: (block[name]["steps_over_alone"][k]["median"]
                           / cf["progress"][k]["delay_from_the_residual_alone"]) for k in keys}
                for name in ("lag 1", "lag 4")}}
        result["strength"][strength] = block
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    def fmt(h: dict) -> str:
        return f"{h['median']:.2f} [{h['min']:.2f}, {h['max']:.2f}]"

    def line(d: dict) -> str:
        return " / ".join(fmt(v) for v in d.values())

    print("final log-odds coefficient of {1,3} over its target:",
          {c: round(v["median"], 3) for c, v in result["final_over_target"].items()})
    al = result["alone"]
    print("alone twin, kernel at step 0 / step 150 / 10 / 25 / 50 / 75 % of the target:", line(al["kernel"]))
    print("alone twin, step at 10 / 25 / 50 / 75 %:", " / ".join(f"{v['median']:.0f}" for v in al["step"].values()))
    print("alone twin, exact rate over kernel x residual:", line(al["rate_over_kernel_x_residual"]))
    for g in GATES:
        print(f"alone twin, {g} at step 0 / 150 / 10 %:", line(al["gates"][g]))
    print("alone twin, kernel by parameter group (median) at step 0 / 10 % / 25 %: "
          + "; ".join(f"{name} " + "/".join(f"{v['median']:.3f}" for v in al["kernel_parts"][name].values())
                      for name in PARTS))
    for strength, block in result["strength"].items():
        print(f"== lower-order strength {strength}")
        k = block["kernel_lag1_over_lag4"]
        print("  kernel, lag-1 twin over lag-4 twin, at step 150 / 10 / 25 / 50 / 75 %: "
              + " / ".join(f"{fmt(v)} ({v['n_above_1']}/8)" for v in k.values()))
        print("  progress of the term at step 150: " + "; ".join(f"{n} {fmt(v)}"
                                                                 for n, v in block["progress_at_step_150"].items()))
        print("  steps, lag 4 over lag 1, at 10 / 25 / 50 / 75 %: "
              + " / ".join(f"{fmt(v)} ({v['n_above_1']}/8)" for v in block["steps_lag4_over_lag1"].values()))
        print("  the sweep's half-time, lag 4 over lag 1, in steps / gain / gain x kernel: "
              + " / ".join(f"{fmt(v)} ({v['n_above_1']}/8)" for v in block["sweep_half_time_lag4_over_lag1"].values()))
        for name in ("lag 1", "lag 4"):
            r = block[name]
            print(f"  {name}: steps over the alone twin's:        {line(r['steps_over_alone'])}")
            print(f"  {name}: rate, alone over with:              {line(r['rate_alone_over_with'])}")
            print(f"  {name}:   residual factor, alone over with: {line(r['residual_alone_over_with'])}")
            print(f"  {name}:   kernel factor, alone over with:   {line(r['kernel_alone_over_with'])}")
            print(f"  {name}: exact rate over kernel x residual:  {line(r['rate_over_kernel_x_residual'])}")
            print(f"  {name}: the sweep's half-time over the alone twin's, steps / gain / gain x kernel: "
                  + line(r["sweep_half_time_over_alone"]))
            for g in GATES:
                print(f"  {name}: {g} at step 150 / at 10 %: {line(r['gates'][g])}")
            print(f"  {name}: kernel by parameter group (median) at 10 % / 25 %: "
                  + "; ".join(f"{part} " + "/".join(f"{v['median']:.3f}" for v in r["kernel_parts"][part].values())
                              for part in PARTS))
        cf = block["closed_form"]
        ks = list(cf["progress"])
        print(f"  closed form: rule A {cf['rule_A']:.3f}, rule B {cf['rule_B']:.3f}, end "
              f"{cf['residual_factor_at_the_end']:.3f}; residual factor at 10/25/50/75 %: "
              + " / ".join(f"{cf['progress'][k]['residual_factor']:.3f}" for k in ks)
              + "; delay from the residual alone: "
              + " / ".join(f"{cf['progress'][k]['delay_from_the_residual_alone']:.2f}" for k in ks))
        for name in ("lag 1", "lag 4"):
            print(f"  {name}: residual factor measured / closed form: "
                  + " / ".join(f"{cf['residual_factor_measured_over_closed_form'][name][k]:.3f}" for k in ks)
                  + "; lower-order coefficient " + " / ".join(f"{block[name]['low_order_coefficient'][k]['median']:.3f}"
                                                              for k in ks)
                  + " (closed form " + " / ".join(f"{cf['progress'][k]['first_order_coefficient']:.3f}" for k in ks)
                  + "); steps over alone / residual-only delay: "
                  + " / ".join(f"{cf['steps_over_alone_divided_by_the_delay_from_the_residual_alone'][name][k]:.2f}"
                               for k in ks))
        print("  mean of the two lags / closed form: "
              + " / ".join(f"{cf['mean_of_the_two_lags_over_closed_form'][k]:.3f}" for k in ks)
              + "; lower-order coefficient, lag 1 minus lag 4: "
              + " / ".join(f"{v['median']:+.3f} ({v['n_above_0']}/8)"
                           for v in cf["low_order_coefficient_lag1_minus_lag4"].values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
