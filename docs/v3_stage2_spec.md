# Tycho Survey `v3` Stage 2 — Pre-Registered Causal Surgery Specification

> **Branch:** `origin/v3-roadmap` (built atop Stage 1 three-scale discovery `7767a4b`; `docs/ROADMAP.md` and `origin/relay-confirm` @ `88aeb84` strictly untouched).
> **Target Hardware:** `1× RTX PRO 6000` (lab server or Colab PRO 6000 100h budget). Estimated total compute: **~25–35 GPU-hours** (`15` core runs across `5` arms × `3` seeds, ~1.5–2.2 h/run at `89.4M` total / `37.9M` non-embedding params × `1.0B` tokens).
> **Status:** Pre-registered BEFORE launching any GPU training run (zero moving goalposts).

---

## 1. Scientific Question & Literature Positioning

Stage 1 ([v3_stage1_overnight_report.md](results/v3_stage1_overnight_report.md)) established a three-scale (`OLMo-2-1B / 7B / 13B`, 13× parameter span, 32 checkpoints) natural-experiment triple dissociation showing that inside-branch Post-RMSNorm (`h = x + RMSNorm(SubLayer(x))`) models initialize with a **50.8× scale mismatch** between Post-RMSNorm gain ($\gamma_0 = 1.0$) and token embeddings ($\mathrm{RMS}(E_{\mathrm{tok}}) = 0.0197$), severing the Layer-0 skip connection (`dA0 = +6.85 / +8.16 / +9.53 nats`) until the dimensionless order parameter
$$\mathcal{R}_0(t) \equiv \frac{\mathrm{RMS}(\gamma_{\text{post-attn}, 0}(t))}{\mathrm{RMS}(E_{\mathrm{tok}}(t))}$$
crosses $\mathcal{R}_0 \approx 1$, triggering a **138×–163× Layer-0 causal collapse** mediated by a universal **transient `Attn0–MLP0` Hydra redundancy bridge (`E = +2.45 / +4.18 / +2.86 nats`)**.

### Direct Positioning Against Prior Literature (Audited 2026-11-27)
1. **Sun et al. (NeurIPS 2025), *The Curse of Depth in LLMs***: Showed Pre-LN variance explosion renders deep layers ineffective; did not study Post-RMSNorm's early Layer-0 mismatch trap or $\mathcal{R}_0 \approx 1$ self-evacuation.
2. **Kim et al. (ICML 2025), *Peri-LN*** & **Gemma 2 (2024)**: Peri-LN showed empirically that freezing output-norm $\gamma \equiv 1$ degrades perplexity vs learnable $\gamma$, and adopted EmbedNorm (while Gemma 2 adopted $\sqrt{d}$ embedding scaling) without identifying the Layer-0 skip-severing trap or $\mathcal{R}_0 \approx 1$ phase transition. Our Stage 2 Arms 3–5 directly test whether **$\mathcal{R}_0(0)$ alone** explains *both* why Gemma's $\sqrt{d}$ scaling avoids the trap and why Peri-LN's frozen $\gamma \equiv 1$ suffers when unscaled!
3. **Lad et al. (2024), *The Remarkable Robustness of LLMs: Stages of Inference***: Argued from static Pre-LN checkpoints that catastrophic Layer-0 ablation fragility is an intrinsic property of "detokenization". Stage 1 showed `OLMo-2` (`1B/7B/13B`) exhibits `dA0 = 0.05–0.07` and `dM0 = 0.00–0.05` once $\mathcal{R}_0 < 1$. Stage 2 tests this causally from Step 0.
4. **McGrath et al. (2023), *The Hydra Effect*** & **Rushing & Nanda (2024)**: Documented static self-repair in trained models; Stage 2 tests our causal prediction that the Layer-0 **transient developmental Hydra bridge** appears *if and only if* a model starts at $\mathcal{R}_0(0) \gg 1$ and later crosses $\mathcal{R}_0 \approx 1$.

---

## 2. Model, Data, and Optimizer Protocol

| Component | Core Specification (`~70M` Modern LLM) | Rationale |
|---|---|---|
| Architecture | Decoder-only Transformer: `L = 12`, `d_model = 512`, `n_heads = 8` (`d_head = 64`), `d_ff = 1376` (**SwiGLU**), **RoPE** ($\theta = 10^4$), **QK-Norm** (RMSNorm on Q & K), untied `lm_head`, no biases | Matches exact modern `OLMo-2` block design; fast enough for 15+ full runs on 1× PRO 6000 |
| Tokenizer & Data | GPT-2 BPE (`V = 50,257`), `seq_len = 512`, streaming or pre-tokenized **`HuggingFaceFW/fineweb-edu`** (`sample-10BT`), **fixed deterministic validation shard** (`256` sequences × `512` tokens) | Clean, standard, modern web-edu corpus; fixed identical data stream across arms for each seed |
| Token Budget | **`1.0B` tokens** per run (`~15×` Chinchilla-optimal for `~70M`, `batch_tokens = 65,536` → **`15,260` steps**, warmup `400` steps, cosine decay to `0.1×` peak LR) | Matches `OLMo-2` long-horizon overtraining regime; gives `~15k` steps for $\gamma_0$ to cross $\mathcal{R}_0 \approx 1$ under `lr = 1e-3, wd = 0.1` |
| Optimizer | `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$, grad clip `1.0`), `peak_lr = 1.0e-3`, `weight_decay = 0.1` on 2D weights **and** RMSNorm `gamma` (matching `OLMo-2`), `weight_decay = 0.0` on `embed_tokens` | Reproduces exact `OLMo-2` AdamW relaxation clock ($\lambda \eta t \approx 1.5$ by step 15k, plus gradient shrinkage → $\mathcal{R}_0: 50 \to <0.5$) |
| Precision | `bf16` autocast (GPU) with `fp32` master weights & `fp32` diagnostic eval | High throughput on PRO 6000 + exact `IC-v3` instrument verification |
| Seeds | **`3` fresh seeds** (`42, 137, 2026`) per arm, sharing identical data order per seed | Enforces Stage 1 Gate 1 multi-seed discipline |

---

## 3. Pre-Registered 5-Arm Causal Surgery Matrix

Every arm is **identical** in architecture, parameter count, data stream, LR schedule, and random seed, changing **ONLY** the normalization placement / initialization line specified below:

| Arm ID | Description | Exact 1-Line Change | Initial $\mathcal{R}_0(0)$ | Primary Pre-Registered Causal Predictions |
|---|---|---|---|---|
| **Arm 1: `pre_ln`** | Modern Pre-LN baseline (`Llama`-style) | `Pre-RMSNorm`, `emb_std=0.02`, `gamma=1.0` | N/A | **P1a:** Permanent Layer-0 bottleneck (`dA0 + dM0 > 2.0 nats` at 1B tok, `causD < 0.38`). **P1b:** Strong interior massive activations (`maxOutI > 100×`). |
| **Arm 2: `post_olmo2`** | Default `OLMo-2` Post-RMSNorm | `Post-RMSNorm`, `emb_std=0.02`, `gamma=1.0` | **`50.0`** | **P2a:** Reproduces Stage 1 trap in miniature: early `dA0 > 2.0 nats` & early `L1..L6` anti-residual erasure (`minCosA < -0.45`). **P2b:** As $\mathcal{R}_0(t)$ crosses `~1–2`, fires positive transient Hydra bridge (`max E_sL0 > +0.30 nats`), then evacuates Layer 0 (`dA0` drops `>5×`, `causD` rises `>+0.15`). |
| **Arm 3: `post_matched_gamma`** | **Matched-Skip $\gamma$ (LayerScale-matched)** | `Post-RMSNorm`, `emb_std=0.02`, **`gamma_init = 0.02`** | **`1.0`** | **P3a:** **Eliminates Layer-0 trap from Step 0** (`max_t dA0 < 0.4 × Arm2_peak` and `causD(early) > Arm2(early) + 0.12`). **P3b:** Eliminates severe anti-residual erasure (`min_t minCosA > Arm2 + 0.20`) and transient Hydra bridge (`max E_sL0 < 0.25`). **P3c (Efficiency Test):** Test whether early/final `val_bpb` improves over Arm 2 (two-sided, reported transparently regardless of sign!). |
| **Arm 4: `post_matched_emb`** | **Unit-Scale Embedding Control (Gemma-Equivalent $\mathcal{R}_0=1$)** | `Post-RMSNorm`, **`emb_std = 1.0`**, `gamma_init = 1.0` *(zero AdamW LR confound vs `embed_scale=50`)* | **`1.0`** | **P4 (Dimensionless Ratio Proof):** Mathematically identical to Arm 3 at Step 0 up to a global factor of 50 (cancelled by RMSNorm!), reproducing **Arm 3** (zero Layer-0 trap, zero severe erasure, high early `causD`) despite having `gamma_init = 1.0`! Proves $\mathcal{R}_0(0)$ alone governs the trap. |
| **Arm 5: `post_frozen_gamma`** | **Mechanistic Lock-In (`Peri-LN` $\gamma \equiv 1$ Control)** | `Post-RMSNorm`, `emb_std=0.02`, **`gamma = 1.0` frozen (`requires_grad=False`)** | **`50.0` (locked)** | **P5:** Prevents $\gamma_0$ shrinkage → keeps $\mathcal{R}_0(t) \gg 1$ far longer (or permanently), **trapping the model in high `dA0` + severe `minCosA` + delayed/absent evacuation** and degrading final `val_bpb` vs Arm 2 & Arm 3 (mechanistically explaining Kim et al. ICML 2025's frozen-$\gamma$ ablation!). |

### Optional Follow-Up Sub-Arms (Only After Core 5 Arms Complete, ~6 GPU-hours)
- **Sub-Arm 6 (`post_depth_graded`):** Initialize $\gamma_l(0)$ directly to the self-organized `OLMo-2` late depth profile (`0.02 → 0.12` linearly across layers `0..L-1`).
- **Sub-Arm 7 (`high_lr_stress`):** Compare Arm 2 (`R_0=50`) vs Arm 3 (`R_0=1`) at `3×` peak LR (`3.0e-3`) to directly test whether matching $\mathcal{R}_0(0)=1$ eliminates `OLMo-2-13B`'s `step4k` high-LR early loss spike!

---

## 4. Telemetry & Online Instrument Verification (`IC-v3`)

Evaluated at **22 log-spaced + linear-transition checkpoints** per run (`step = 0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7500, 9000, 10500, 12000, 13500, 15260`), logging to `metrics.jsonl`:
- **Optimization & Loss:** `val_bpb`, `train_bpb`, `grad_norm`, `lr`
- **Order Parameter & Scales:** `embed_rms`, `gamma_attn[0..L-1]`, `gamma_mlp[0..L-1]`, `R_0 = gamma_attn[0] / embed_rms`, `wR_0 = wRMS_attn[0] / embed_rms`
- **Residual Geometry:** `wRMS_attn[l]`, `wRMS_mlp[l]`, `cosA_stream[l]`, `cosM_stream[l]`, `min_cosA_stream`, `max_outlier_int`, `max_outlier_pos0`, `pred_depth`
- **Causal Structure (Full Layer-Wise Mean Ablation):** `dA[0..L-1]`, `dM[0..L-1]`, `causD` (normalized causal depth), `dA0M0`, **`E_same_L0 = dA0M0 - dA0 - dM0`**, plus `adj_attn` Hydra redundancy fraction & mean `E_norm`!
- **Instrument Gate (`IC-v3.2`):** Exact residual reconstruction relative error `max |h_out - (h_in + dA + dM)| / max(|h_in|, |dA|, |dM|, |h_out|) < 1e-5` checked at EVERY logged checkpoint.

---

## 5. Pre-Registered Decision Criteria (Zero Moving Goalposts)

1. **Strong Confirmation (Publishable Main-Track Paper):**
   - Across `3/3` seeds, Arm 2 reproduces the trap + evacuation + Hydra bridge (`peak dA0 > 2.0`, `minCosA < -0.45`, evacuation drop `>5×`, `max E_sL0 > +0.30`);
   - BOTH Arm 3 AND Arm 4 simultaneously eliminate the trap (`peak dA0 < 0.4× Arm 2`), eliminate severe erasure (`minCosA > Arm 2 + 0.20`), and raise early `causD` (`>+0.12`);
   - Arm 5 remains locked in high `dA0` / lower `causD` and worse `val_bpb` than Arm 3.
   - *(Bonus practical kicker)* Arm 3/4 achieves statistically consistent `val_bpb` gain (lower loss at equal tokens, especially early/mid training) or higher max stable LR in Sub-Arm 7!
2. **Mechanistic Confirmation Without Loss Gain (Still High-Value Mechanistic Paper):**
   - All structural predictions (`P2–P5`) hold cleanly across `3/3` seeds, while final `val_bpb` of Arm 2 vs Arm 3/4 is tied (`|Δbpb| < 0.005`). Report transparently: the mismatch trap wastes early capacity on erasure & creates transient fragility, even if AdamW eventually recovers terminal loss after sufficient integrated LR.
3. **Falsification / Pivot Condition:**
   - If Arm 2 fails to reproduce any Layer-0 trap at `70M` scale (or Arm 3/4 shows the *exact same* trap as Arm 2), stop and investigate scale-threshold / QK-Norm / batch-size interaction before scaling to `160M–300M`.
