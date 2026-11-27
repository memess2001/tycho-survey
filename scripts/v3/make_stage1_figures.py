"""Generate publication-grade summary figures for Tycho Survey v3 Stage 1 Overnight Report."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "work" / "v3_stage1"
OUT_DIR = ROOT / "docs" / "results" / "v3_stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _parse_tok(rev: str) -> float:
    m = re.search(r"tokens(\d+(?:\.\d+)?)B", rev)
    if m:
        v = float(m.group(1))
        if "step300-tokens1B" in rev:
            return 0.63
        return v
    return 0.0


def fig1_olmo2_r0_evacuation() -> None:
    """Master Figure 1: Dimensionless Order Parameter R_0 = gamma_A0 / RMS(E_tok) governs Layer-0 evacuation across 1B & 7B."""
    o1 = json.loads((DATA_DIR / "olmo2_1b_mechanism.json").read_text())
    o7 = json.loads((DATA_DIR / "olmo2_7b_mechanism.json").read_text())

    def extract(rows):
        toks = np.array([max(0.3, _parse_tok(r["revision"])) for r in rows])
        r0 = np.array([r["gamma_attn"][0] / r["embed_rms"] for r in rows])
        dA0 = np.array([r["causal_nats"]["dA0"] for r in rows])
        dM0 = np.array([r["causal_nats"]["dM0"] for r in rows])
        dA0M0 = np.array([r["causal_nats"]["dA0M0"] for r in rows])
        e_s0 = np.array([r["causal_nats"]["E_same_L0"] for r in rows])
        min_ca = np.array([r["min_cosA_stream"] for r in rows])
        gA0 = np.array([r["gamma_attn"][0] for r in rows])
        gAlast = np.array([r["gamma_attn"][-1] for r in rows])
        return dict(toks=toks, r0=r0, dA0=dA0, dM0=dM0, dA0M0=dA0M0, e_s0=e_s0, min_ca=min_ca, gA0=gA0, gAlast=gAlast)

    d1, d7 = extract(o1), extract(o7)
    # Exclude step0 (t=0.3 dummy) from post-init dynamic curves where helpful
    m1 = d1["toks"] >= 0.5
    m7 = d7["toks"] >= 0.5

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    # Panel A: R_0(t) vs tokens (log-log)
    ax = axes[0, 0]
    ax.plot(d1["toks"][m1], d1["r0"][m1], "o-", lw=2.2, color="#1f77b4", label="OLMo-2-1B (16L, d=2048)")
    ax.plot(d7["toks"][m7], d7["r0"][m7], "s-", lw=2.2, color="#d62728", label="OLMo-2-7B (32L, d=4096)")
    ax.axhline(1.0, ls="--", color="black", lw=1.2, alpha=0.8, label=r"Critical threshold $\mathcal{R}_0 = 1$")
    ax.axhspan(0.8, 2.2, color="#ffdd57", alpha=0.22)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pretraining Tokens (Billions)")
    ax.set_ylabel(r"Order Parameter $\mathcal{R}_0 = \mathrm{RMS}(\gamma_{A0}) / \mathrm{RMS}(E_{\mathrm{tok}})$")
    ax.set_title("A. Post-RMSNorm / Embedding Scale Ratio Collapse (50x -> 0.3x)")
    ax.legend(loc="lower left")

    # Panel B: Layer-0 Causal Damage dA0 vs tokens
    ax = axes[0, 1]
    ax.plot(d1["toks"][m1], d1["dA0"][m1], "o-", lw=2.2, color="#1f77b4", label="OLMo-2-1B: Ablate Attn0")
    ax.plot(d7["toks"][m7], d7["dA0"][m7], "s-", lw=2.2, color="#d62728", label="OLMo-2-7B: Ablate Attn0")
    ax.plot(d1["toks"][m1], d1["e_s0"][m1], "o--", lw=1.5, color="#1f77b4", alpha=0.65, label="OLMo-2-1B: Hydra E(Attn0, MLP0)")
    ax.plot(d7["toks"][m7], d7["e_s0"][m7], "s--", lw=1.5, color="#d62728", alpha=0.65, label="OLMo-2-7B: Hydra E(Attn0, MLP0)")
    ax.axhline(0.0, ls=":", color="gray", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Pretraining Tokens (Billions)")
    ax.set_ylabel("Mean-Ablation Loss Delta / Excess Joint Loss (nats)")
    ax.set_title("B. Layer-0 Bottleneck Evacuation via Transient Hydra Bridge")
    ax.legend(loc="upper right", fontsize=8)

    # Panel C: Universal Data Collapse — dA0 & minCosA plotted directly against R_0!
    ax = axes[1, 0]
    # Skip initial warmup step (1B) before L0 bottleneck forms (use tok >= 10B)
    u1 = d1["toks"] >= 10.0
    u7 = d7["toks"] >= 10.0
    ax.plot(d1["r0"][u1], d1["dA0"][u1], "o-", lw=2.2, color="#1f77b4", label="OLMo-2-1B (dA0, 21B-4001B)")
    ax.plot(d7["r0"][u7], d7["dA0"][u7], "s-", lw=2.2, color="#d62728", label="OLMo-2-7B (dA0, 10B-3896B)")
    ax.axvline(1.0, ls="--", color="black", lw=1.2, alpha=0.8)
    ax.axvspan(0.8, 2.5, color="#ffdd57", alpha=0.22, label=r"Transition band $\mathcal{R}_0 \in [0.8, 2.5]$")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"Order Parameter $\mathcal{R}_0(t) = \gamma_{A0} / \mathrm{RMS}(E_{\mathrm{tok}})$  (training flows left $\to$ right)")
    ax.set_ylabel("Layer-0 Attn Ablation Damage dA0 (nats)")
    ax.set_title("C. Scale Collapse (1B vs 7B): Evacuation Locks to R_0 ~ 1")
    ax.legend(loc="upper right")

    # Panel D: Anti-Residual Erasure minCosA vs tokens
    ax = axes[1, 1]
    ax.plot(d1["toks"][m1], d1["min_ca"][m1], "o-", lw=2.2, color="#1f77b4", label="OLMo-2-1B min cos(Attn_l, h_in)")
    ax.plot(d7["toks"][m7], d7["min_ca"][m7], "s-", lw=2.2, color="#d62728", label="OLMo-2-7B min cos(Attn_l, h_in)")
    ax.axhline(0.0, ls=":", color="gray", lw=1.0)
    ax.set_xscale("log")
    ax.set_ylim(-0.98, 0.1)
    ax.set_xlabel("Pretraining Tokens (Billions)")
    ax.set_ylabel(r"Minimum Layer Write-Stream Cosine $\min_l \cos(\Delta h_{\mathrm{attn}, l}, h_{\mathrm{in}, l})$")
    ax.set_title("D. Early Anti-Residual Erasure (-0.85) Relaxes Upon Evacuation")
    ax.legend(loc="lower right")

    fig.suptitle("Discovery 1: Post-RMSNorm Embedding-Skip Mismatch Trap & Self-Evacuation (OLMo-2 1B & 7B)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig1_olmo2_R0_evacuation_1B_7B.png")
    plt.close(fig)


def fig2_triple_family_depth_hydra() -> None:
    """Figure 2: Three 1B Families Compared — Deepward Migration, Pre-LN Permanent L0 Trap vs OLMo-2 Escape, & Universal Hydra Redundancy."""
    rep = json.loads((DATA_DIR / "stage1_analysis.json").read_text())
    ms = rep["models"]
    fams = [
        ("EleutherAI/pythia-1b", "Pythia-1B (Parallel Pre-LN)", "#2ca02c", "o"),
        ("allenai/OLMo-1B-0724-hf", "OLMo-1B-0724 (Seq Pre-LN)", "#ff7f0e", "^"),
        ("allenai/OLMo-2-0425-1B", "OLMo-2-0425-1B (Post-RMSNorm)", "#1f77b4", "s"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))

    for mid, lbl, col, mk in fams:
        s = ms[mid]
        toks = np.array(s["tokens_B"])
        m = toks >= 0.5
        axes[0].plot(toks[m], np.array(s["ch1"]["pred_depth"])[m], marker=mk, ms=4, lw=2, color=col, label=lbl)
        axes[1].plot(toks[m], np.array(s["ch2"]["causal_depth"])[m], marker=mk, ms=4, lw=2, color=col, label=lbl)
        axes[2].plot(toks[m], np.array(s["ch2"]["pair_groups"]["adj_attn"]["frac_redundant_gt_0p02"])[m],
                     marker=mk, ms=4, lw=2, color=col, label=lbl)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Pretraining Tokens (B)")
    axes[0].set_ylabel("Normalized Predictive Depth (Logit Lens)")
    axes[0].set_title("A. Universal Deepward Migration of Predictions")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Pretraining Tokens (B)")
    axes[1].set_ylabel("Normalized Causal Depth (Mean Ablation)")
    axes[1].set_title("B. Pre-LN Stays L0-Trapped; Post-Norm Escapes")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].set_xscale("log")
    axes[2].set_xlabel("Pretraining Tokens (B)")
    axes[2].set_ylabel("Fraction of Adjacent Attn Pairs with E > +0.02")
    axes[2].set_title("C. Universal Adjacent-Attention Hydra Emergence")
    axes[2].set_ylim(-0.05, 1.08)
    axes[2].legend(loc="lower right", fontsize=8)

    fig.suptitle("Discovery 2 & 3: Triple-Family 1B Structural Comparison (Pythia vs OLMo-1 vs OLMo-2)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_triple_family_depth_hydra.png", bbox_inches="tight")
    plt.close(fig)


def fig3_massive_activation_and_fincam() -> None:
    """Figure 3: Non-Monotonic Interior-to-BOS Massive Activation Migration (1B) + Scale-Ordered Pythia Late Final-Layer Co-Alignment."""
    rep = json.loads((DATA_DIR / "stage1_analysis.json").read_text())
    ms = rep["models"]
    ms_agg = rep["multiseed"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # Panel A: Interior massive activation boom-and-bust at 1B across all 3 families
    ax = axes[0]
    for mid, lbl, col, mk in [
        ("EleutherAI/pythia-1b", "Pythia-1B (Parallel Pre-LN)", "#2ca02c", "o"),
        ("allenai/OLMo-1B-0724-hf", "OLMo-1B-0724 (Seq Pre-LN)", "#ff7f0e", "^"),
        ("allenai/OLMo-2-0425-1B", "OLMo-2-0425-1B (Post-RMSNorm)", "#1f77b4", "s"),
    ]:
        s = ms[mid]
        toks = np.array(s["tokens_B"])
        m = toks >= 0.5
        ax.plot(toks[m], np.array(s["ch4"]["max_outlier_int"])[m], marker=mk, ms=4.5, lw=2, color=col, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pretraining Tokens (Billions)")
    ax.set_ylabel("Max Interior Outlier Ratio (t >= 1, excl. final L)")
    ax.set_title("A. Interior Massive Activations Rise 300x Then Collapse 12x-50x")
    ax.legend(loc="upper left")

    # Panel B: Pythia scale-ordered final-layer Attn-MLP co-alignment (4 seeds at 70M/160M, 2 at 410M, 1 at 1B)
    ax = axes[1]
    for gkey, lbl, col in [
        ("pythia-70m_4seeds", "Pythia-70M (4 seeds mean±SD)", "#9467bd"),
        ("pythia-160m_4seeds", "Pythia-160M (4 seeds mean±SD)", "#1f77b4"),
        ("pythia-410m_2seeds", "Pythia-410M (2 seeds mean±SD)", "#ff7f0e"),
    ]:
        g = ms_agg[gkey]
        toks = np.array(g["tokens_B"])
        m = toks >= 10.0
        mu = np.array(g["final_cos_am"]["mean"])[m]
        sd = np.array(g["final_cos_am"]["sd"])[m]
        ax.plot(toks[m], mu, "o-", ms=3.5, lw=2, color=col, label=lbl)
        ax.fill_between(toks[m], mu - sd, mu + sd, color=col, alpha=0.18)
    s1b = ms["EleutherAI/pythia-1b"]
    t1b = np.array(s1b["tokens_B"])
    m1b = t1b >= 10.0
    ax.plot(t1b[m1b], np.array(s1b["ch1"]["final_cos_am"])[m1b], "s-", ms=4, lw=2, color="#2ca02c", label="Pythia-1B (parallel)")
    so1 = ms["allenai/OLMo-1B-0724-hf"]
    to1 = np.array(so1["tokens_B"])
    mo1 = (to1 >= 10.0) & (to1 <= 320.0)
    ax.plot(to1[mo1], np.array(so1["ch1"]["final_cos_am"])[mo1], "^--", ms=4, lw=1.6, color="gray", label="OLMo-1B (sequential control)")
    ax.set_xlabel("Pretraining Tokens (Billions)")
    ax.set_ylabel(r"Final-Layer $\cos(\Delta h_{\mathrm{attn}, L-1}, \Delta h_{\mathrm{mlp}, L-1})$")
    ax.set_title("B. Parallel-Residual Late Co-Alignment (Scale-Ordered in Cosine Tail)")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Discovery 4 & 5: Non-Monotonic Interior Outliers & Parallel-Residual Cosine-Tail Co-Alignment", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_massive_activation_and_fincam.png", bbox_inches="tight")
    plt.close(fig)


def fig4_probes_and_multiseed() -> None:
    """Figure 4: Channel 3 Micro-Probe Trajectories (P3 Copy vs Decoy, P6 Parametric Memory vs ICL Tug-of-War)."""
    rep = json.loads((DATA_DIR / "stage1_analysis.json").read_text())
    ms = rep["models"]
    ms_agg = rep["multiseed"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))

    # Panel A: P6 Memory vs ICL (margin_bpb_ctx_b: negative = ICL overrides parametric, positive = parametric wins)
    ax = axes[0]
    for mid, lbl, col, mk in [
        ("EleutherAI/pythia-1b", "Pythia-1B", "#2ca02c", "o"),
        ("allenai/OLMo-1B-0724-hf", "OLMo-1B-0724", "#ff7f0e", "^"),
        ("allenai/OLMo-2-0425-1B", "OLMo-2-0425-1B", "#1f77b4", "s"),
    ]:
        s = ms[mid]
        toks = np.array(s["tokens_B"])
        m = toks >= 0.5
        ax.plot(toks[m], np.array(s["ch3"]["curves"]["P6_memory_vs_icl"]["margin_bpb_ctx_b"])[m],
                marker=mk, ms=4, lw=2, color=col, label=lbl)
    g160 = ms_agg["pythia-160m_4seeds"]
    t160 = np.array(g160["tokens_B"])
    m160 = t160 >= 0.5
    mu = np.array(g160["P6_icl_margin_bpb"]["mean"])[m160]
    sd = np.array(g160["P6_icl_margin_bpb"]["sd"])[m160]
    ax.plot(t160[m160], mu, "--", lw=1.6, color="#9467bd", label="Pythia-160M (4 seeds)")
    ax.fill_between(t160[m160], mu - sd, mu + sd, color="#9467bd", alpha=0.15)
    ax.axhline(0.0, ls=":", color="black", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Pretraining Tokens (Billions)")
    ax.set_ylabel("Parametric - Counterfactual ICL Margin (bpb)\n(< 0: ICL wins; > 0: Parametric memory wins)")
    ax.set_title("A. P6 Parametric Memory vs In-Context Override Tug-of-War")
    ax.legend(loc="upper right", fontsize=8)

    # Panel B: P3 Copy vs Decoy Interference Shift across families
    ax = axes[1]
    for mid, lbl, col, mk in [
        ("EleutherAI/pythia-1b", "Pythia-1B", "#2ca02c", "o"),
        ("allenai/OLMo-1B-0724-hf", "OLMo-1B-0724", "#ff7f0e", "^"),
        ("allenai/OLMo-2-0425-1B", "OLMo-2-0425-1B", "#1f77b4", "s"),
    ]:
        s = ms[mid]
        toks = np.array(s["tokens_B"])
        m = toks >= 0.5
        ax.plot(toks[m], np.array(s["ch3"]["curves"]["P3_copy_decoy"]["shift_bpb_a_minus_b"])[m],
                marker=mk, ms=4, lw=2, color=col, label=lbl)
    ax.axhline(0.0, ls=":", color="black", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Pretraining Tokens (Billions)")
    ax.set_ylabel("Decoy Interference Penalty (bpb, Clean - Decoy)")
    ax.set_title("B. P3 Real-Text Decoy Interference Remains Persistent Through 4T")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Channel 3 Tokenizer-Agnostic Micro-Probes Across Three 1B Families", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_probes_p6_p3.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_olmo2_r0_evacuation()
    fig2_triple_family_depth_hydra()
    fig3_massive_activation_and_fincam()
    fig4_probes_and_multiseed()
    print("Generated 4 figures in", OUT_DIR)
