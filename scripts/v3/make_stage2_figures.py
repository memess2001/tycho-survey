"""Generate publication-grade 6-panel Stage 2 Causal Surgery figure from Pilot Seed 42 runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "work" / "v3_stage2" / "runs"
OUT_DIR = ROOT / "docs" / "results" / "v3_stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9.8,
    "axes.titlesize": 10.6,
    "axes.labelsize": 9.5,
    "legend.fontsize": 7.9,
    "figure.dpi": 165,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

ARMS = [
    ("post_matched_gamma_s42", r"Arm 3: Matched-$\gamma$ ($\gamma_0=0.02, \mathcal{R}_0=1$) [OURS]", "#1b9e77", "o", "-", 2.6),
    ("post_olmo2_s42", r"Arm 2: Default OLMo-2 ($\gamma_0=1, \mathcal{R}_0=50$)", "#d95f02", "s", "-", 2.2),
    ("post_frozen_gamma_s42", r"Arm 5: Frozen-$\gamma$ ($\gamma\equiv 1$, Peri-LN ctrl)", "#e7298a", "v", "--", 2.0),
    ("post_matched_emb_s42", r"Arm 4: Unit-Emb ($\mathrm{emb}=1, \mathcal{R}_0=1$)", "#7570b3", "D", "-.", 1.9),
    ("pre_ln_s42", "Arm 1: Modern Pre-LN (Llama-style)", "#377eb8", "^", ":", 2.0),
]


def load_arm(tag: str) -> list[dict]:
    p = RUNS_DIR / tag / "metrics.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def make_fig5() -> None:
    data = {tag: load_arm(tag) for tag, *_ in ARMS}
    base = data["post_olmo2_s42"]
    toks_all = np.array([r["tokens"] / 1e6 for r in base])
    m = toks_all > 0  # post-init steps (3.3M -> 1000M)
    t_post = toks_all[m]

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2))

    # Panel A: Validation bpb Advantage relative to Default OLMo-2
    ax = axes[0, 0]
    b_bpb = np.array([r["val_bpb"] for r in base])[m]
    ax.axhline(0.0, color="#d95f02", ls="-", lw=1.8, alpha=0.9)
    for tag, lbl, col, mk, ls, lw in ARMS:
        if tag == "post_olmo2_s42":
            continue
        y = np.array([r["val_bpb"] for r in data[tag]])[m] - b_bpb
        ax.plot(t_post, y, marker=mk, ms=4.4, ls=ls, lw=lw, color=col)
    ax.fill_between(t_post, -0.06, 0.0, color="#1b9e77", alpha=0.07)
    ax.set_xscale("log")
    ax.set_ylim(-0.058, 0.032)
    ax.set_xlabel("Training Tokens (Millions)")
    ax.set_ylabel(r"$\Delta$ val_bpb vs Default OLMo-2 (lower = better)")
    ax.set_title("A. Validation bpb: Matched-Gamma Wins 18/18 Checkpoints")
    ax.annotate("18/18 wins!\n(-0.052 early,\n-0.0047 final ≈ 11% speedup)", xy=(26.2, -0.0518), xytext=(110, -0.049),
                fontsize=8.0, color="#1b9e77", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.0))

    # Panel B: Order Parameter R_0(t) across Post-RMSNorm arms
    ax = axes[0, 1]
    for tag, lbl, col, mk, ls, lw in ARMS:
        if tag == "pre_ln_s42":
            continue
        y = np.array([r["R_0"] for r in data[tag]])[m]
        ax.plot(t_post, y, marker=mk, ms=4.4, ls=ls, lw=lw, color=col)
    ax.axhline(1.0, ls="--", color="black", lw=1.1, alpha=0.8)
    ax.axhspan(0.8, 2.5, color="#ffdd57", alpha=0.22)
    ax.text(4.0, 1.35, r"Stage 1 transition band $\mathcal{R}_0 \in [0.8, 2.5]$", fontsize=7.6, color="#7a5d00")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training Tokens (Millions)")
    ax.set_ylabel(r"Order Parameter $\mathcal{R}_0 = \gamma_{A0} / \mathrm{RMS}(E_{\mathrm{tok}})$")
    ax.set_title(r"B. Order Parameter $\mathcal{R}_0(t)$: Frozen Locked vs Matched Ideal")

    # Panel C: Layer-0 Causal Bottleneck dA0(t)
    ax = axes[0, 2]
    for tag, lbl, col, mk, ls, lw in ARMS:
        y = np.array([r["dA0"] for r in data[tag]])[m]
        ax.plot(t_post, y, marker=mk, ms=4.4, ls=ls, lw=lw, color=col)
    ax.set_xscale("log")
    ax.set_ylim(-0.4, 9.4)
    ax.set_xlabel("Training Tokens (Millions)")
    ax.set_ylabel("Layer-0 Attn Mean-Ablation Damage dA0 (nats)")
    ax.set_title(r"C. Layer-0 Bottleneck Eliminated (144x) When $\mathcal{R}_0(0)=1$")
    ax.annotate("Both R_0(0)=1 arms\nflat at 0.00–0.03!", xy=(491, 0.08), xytext=(45, 1.7),
                fontsize=8.0, color="#1b9e77", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.0))

    # Panel D: Anti-Residual Erasure minCosA(t)
    ax = axes[1, 0]
    for tag, lbl, col, mk, ls, lw in ARMS:
        y = np.array([r["min_cosA_stream"] for r in data[tag]])[m]
        ax.plot(t_post, y, marker=mk, ms=4.4, ls=ls, lw=lw, color=col)
    ax.axhline(0.0, ls=":", color="gray", lw=1.0)
    ax.set_xscale("log")
    ax.set_ylim(-0.66, 0.10)
    ax.set_xlabel("Training Tokens (Millions)")
    ax.set_ylabel(r"Min Write-Stream Cosine $\min_l \cos(\Delta h_{\mathrm{attn}, l}, h_{\mathrm{in}, l})$")
    ax.set_title(r"D. High-$\mathcal{R}_0$ Dwell Time Drives Anti-Residual Erasure (-0.58)")
    ax.annotate("Frozen-gamma locked\nat R_0=7.4 -> -0.584!", xy=(1000, -0.584), xytext=(35, -0.60),
                fontsize=8.0, color="#e7298a", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e7298a", lw=1.0))

    # Panel E: Normalized Causal Depth causD(t)
    ax = axes[1, 1]
    for tag, lbl, col, mk, ls, lw in ARMS:
        y = np.array([r["causal_depth"] for r in data[tag]])[m]
        ax.plot(t_post, y, marker=mk, ms=4.4, ls=ls, lw=lw, color=col)
    ax.set_xscale("log")
    ax.set_ylim(0.08, 0.70)
    ax.set_xlabel("Training Tokens (Millions)")
    ax.set_ylabel("Normalized Causal Depth (Mean Ablation)")
    ax.set_title("E. Immediate 3x Deep Causal Utilization (0.61 vs 0.19)")
    ax.annotate("Matched-gamma:\n0.59–0.61 from start!", xy=(262, 0.61), xytext=(15, 0.615),
                fontsize=8.0, color="#1b9e77", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.0))

    # Panel F: Endpoint (1.0B tok) Layer-by-Layer Total Causal Importance (dA + dM) + Outlier Comparison
    ax = axes[1, 2]
    layers = np.arange(12)
    handles, labels = [], []
    for tag, lbl, col, mk, ls, lw in ARMS:
        end = data[tag][-1]
        tot = np.array(end["dA"]) + np.array(end["dM"])
        short = lbl.split("(")[0].strip() + f" (outI={end['max_outlier_int']:.0f}x)"
        (line,) = ax.plot(layers, tot, marker=mk, ms=4.8, ls=ls, lw=lw, color=col, label=short)
        handles.append(line)
        labels.append(lbl)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8.5)
    ax.set_ylim(-0.3, 9.2)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Layer Causal Ablation Damage dA[l] + dM[l] (nats)")
    ax.set_title("F. 1.0B-Token Layer Profile & Interior Outlier Ratio")
    ax.annotate("Matched-gamma:\nbalanced 12-layer\nhighway (L0=0.06!)", xy=(0, 0.06), xytext=(2.8, 3.3),
                fontsize=8.0, color="#1b9e77", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.0))
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.92)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=8.6,
    )
    fig.suptitle(
        r"Tycho v3 Stage 2 (Pilot Seed 42, 89.4M × 1.0B FineWeb-Edu): 1-Line Init Fix ($\gamma_0 = \mathrm{emb\_std}$) Eliminates L0 Trap, Erasure & Beats Default OLMo-2",
        fontsize=12.4,
        fontweight="bold",
        y=0.988,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.885])
    out_p = OUT_DIR / "fig5_stage2_causal_surgery_seed42.png"
    fig.savefig(out_p)
    plt.close(fig)
    print(f"Saved {out_p} ({out_p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    make_fig5()
