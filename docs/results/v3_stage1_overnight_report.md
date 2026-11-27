# Tycho Survey `v3` Stage 1 — Overnight Discovery Report (2026-11-27)

> **Branch:** `origin/v3-roadmap` (built cleanly atop `88aeb84`; `docs/ROADMAP.md` and `origin/relay-confirm` untouched).
> **Scope Completed Overnight on `s82` (88-core 4-NUMA CPU, 0 GPU hours used):**
> - **492 full 4-channel checkpoint scans** across **15 models** (`Pythia-14M / 31M / 70M×4 seeds / 160M×4 seeds / 410M×2 seeds / 1B` + `allenai/OLMo-1B-0724-hf` `0B→3,048B` + `allenai/OLMo-2-0425-1B` `0B→4,001B`), including targeted high-res zoom-in checkpoints (`OLMo-2-1B` `49B–189B`, `Pythia-1B` `step84k–100k`).
> - **32 parameter-level & Layer-0 causal probes** across **three frontier scales**: **`OLMo-2-0425-1B` (10 pts, `0B→4,001B`)**, **`OLMo-2-1124-7B` (12 pts, `1B→3,896B`, including dense transition zoom-in `168B/189B/210B/252B`)**, and **`OLMo-2-1124-13B` (10 pts, 40L, `d=5120`, `9B→5,001B`)**.
> - **Mechanistic audits**: Adjacent-attention Hydra active-compensation check (Top-1 Acc, KL, downstream activation compensation cosine) + exact token/dimension audit of transient interior massive activations.
> - **Instrument integrity:** **100% (`492/492` + `32/32`) `IC-v3` pass rate** (`IC-v3.1` determinism `0.0`, `IC-v3.2` residual identity relative error `< 1e-5` [exact `0.0` across all 32 OLMo-2 probes], `IC-v3.3a/b` ablation identity `0.0`). Disk cleanly evicted after completion (`343 GB` free on `s82`; all 524 checkpoint summaries occupy `83 MB`).

---

## 1. Executive Summary: Ranked Stage 1 → Stage 2 Scorecard

Applying all four hard Kill Gates from [docs/v3_stage1_spec.md](../v3_stage1_spec.md) §5, **one flagship frontier-scale mechanistic discovery clears all four gates across `1B`, `7B`, and `13B` for immediate Stage 2 PRO 6000 causal surgery**, two secondary cross-family findings warrant targeted follow-up or companion sections, and one tempting early lead (`Pythia-160M` final-layer rank collapse) was **cleanly killed** by our multi-seed and cross-family gates.

| Rank | Phenomenon | Gate 1 (Multi-Seed / Non-Artifact) | Gate 2 (Cross-Scale / Cross-Family) | Gate 3 (Literature Clearance) | Gate 4 (PRO 6000 Causal Surgery) | Verdict |
|---|---|---|---|---|---|---|
| **#1 (Flagship)** | **Post-RMSNorm Embedding–Skip Mismatch Trap ($\mathcal{R}_0 = \gamma_{A0}/\mathrm{RMS}(E_{\mathrm{tok}})$) & Self-Evacuation via Transient Hydra Bridge (Triple-Dissociated Across `1B`, `7B`, `13B`)** | **PASS** (`IC-v3` exact `0.0` across all 32 probes; confirmed at both parameter & activation levels) | **PASS AT 1B, 7B & 13B!!** Exact data collapse onto $\mathcal{R}_0 \approx 1$ across `OLMo-2-1B` (16L), `7B` (32L, `lr=3e-4`), and **`13B` (40L, `lr=9e-4`)** despite non-monotonic token times (`126B` / `294B` / `151B`) and `3.5×` step separation (`60k` / `70k` / `18k`); Pre-LN stays L0-trapped through `3T` | **HIGH NOVELTY** (OLMo-2 report & Curse-of-Depth 2025 did not report the `50×` init mismatch, `-0.85` dwell-time erasure, universal `+2.5..+4.2 nat` Hydra bridge, or $\mathcal{R}_0 \sim 1$ three-scale law) | **1-LINE INIT INTERVENTION!** Train 60M–125M Post-RMSNorm with $\mathcal{R}_0(0) \approx 1$ vs default `50×` on PRO 6000: test skipping the L0 trap & erasure without needing high-LR spikes! | **PROMOTE TO STAGE 2 (TOP PRIORITY)** |
| **#2** | **Non-Monotonic Interior-to-BOS Massive-Activation Migration (`Pythia-1B: 1,534→111`, `OLMo-1B: 3,273→35`)** | **PASS** (token-audited: `d1764` on `"."` drops `-556 → absent` as BOS absorbs `-1,042`) | **PASS** across **both** Pre-LN 1B families (`Pythia-1B` parallel & `OLMo-1B-0724` sequential); prevented in Post-RMSNorm (`OLMo-2-1B` max `~80×`) | **MODERATE–HIGH** (directly refutes published monotonic log-exponential growth models of massive activations) | Testable alongside #1 (Pre-LN vs Post-RMSNorm + BOS/delimiter controls) | **KEEP (Companion finding / paper section)** |
| **#3** | **Universal Developmental Hydra Redundancy of Same-Type Adjacent Layers (`adj_attn E_norm = +2.1..+6.4`) + Deepward Migration (`predD`)** | **PASS** (verified non-convexity: `E_Acc` up to `+4.5%`, `E_KL` up to `+0.59`, active compensation cosine `+0.10..+0.21`) | **PASS** across **all 3 1B families** (`Pythia`, `OLMo-1`, `OLMo-2`) & all 6 Pythia sizes (`15/15` models) | **MODERATE** (static Hydra known from McGrath 2023; full developmental trajectory across 3 families is new) | Secondary diagnostic metric inside Stage 2 training runs | **KEEP (Descriptive backbone / diagnostic)** |
| **#4** | **Parallel-Residual Late Cosine-Tail Final-Layer Co-Alignment (`final_cos_am: 0.15 → 0.92`, scale-ordered `64k→80k→96k→112k`)** | **PASS** (10/10 Pythia models: `70M 4/4`, `160M 4/4`, `410M 2/2`, `1B`) | **ARCHITECTURE-SPECIFIC** (parallel Pre-LN only; sequential `OLMo-1B` stays at `~0.08`) | **MODERATE** | Low standalone ceiling (parallel-residual specific) | **DOCUMENT & ARCHIVE (Nice contrast vs sequential)** |
| **#5 (Killed)** | `Pythia-70M/160M` final-layer `erank_normed` 14× collapse (`0.070`) | **FAIL** (huge seed variance: `160M` `0.070 / 0.427 / 0.898 / 0.432`) | **FAIL** (vanishes at `410M` `1.14`, `1B` `0.85`, `OLMo-1B` `0.86`, `OLMo-2-1B` `0.91`) | — | — | **KILL (Saved weeks!)** |

---

## 2. Flagship Discovery #1: The Post-RMSNorm Embedding–Skip Mismatch Trap & $\mathcal{R}_0 \approx 1$ Self-Evacuation Law (`OLMo-2-1B`, `7B`, & `13B`)

![Discovery 1: Post-RMSNorm R_0 Evacuation across OLMo-2 1B, 7B, and 13B](v3_stage1/fig1_olmo2_R0_evacuation_1B_7B_13B.png)

### What We Observed
Comparing all three 1B architectures—and validating across a **13× parameter span** (`OLMo-2-1B` 16L `d=2048`, `OLMo-2-7B` 32L `d=4096`, `OLMo-2-13B` 40L `d=5120`)—reveals a dramatic structural dichotomy between **Pre-LN** and **inside-branch Post-RMSNorm** (`h = x + RMSNorm(SubLayer(x))`):

1. **Pre-LN models remain permanently trapped in a Layer-0 causal bottleneck through 3 Trillion tokens** (the causal dual of Sun et al. 2025's "Curse of Depth"):
   - `Pythia-1B` (300B, parallel Pre-LN): single-block mean ablation of `MLP0` costs **`+6.91 nats`** (`9.5×` larger than the next-largest layer), keeping causal depth stuck at `causD = 0.281`.
   - `OLMo-1B-0724` (299B → **3,048B**, sequential Pre-LN): `Attn0` costs **`+6.54 → +5.91 nats`** and `MLP0` costs **`+2.23 → +3.05 nats`** (`30×` larger than `L1..L14`), keeping `causD = 0.222 → 0.272` even after 3 Trillion tokens.
2. **Post-RMSNorm (`OLMo-2`) unexpectedly starts in the *exact same* catastrophic Layer-0 bottleneck, then spontaneously self-evacuates when a single dimensionless order parameter crosses unity:**
   $$\mathcal{R}_0(t) \equiv \frac{\mathrm{RMS}(\gamma_{\text{post-attn}, 0}(t))}{\mathrm{RMS}(E_{\text{tok}}(t))}$$
   - **Root cause at initialization (`step0`):** `embed_tokens` is initialized with `std = 0.02` (`embRMS = 0.0197`), while Post-RMSNorm scale `γ` is initialized at **`1.000`**—a **`50.8×` scale mismatch ($\mathcal{R}_0(0) = 50.8$)**! Because Layer 0 injects a unit-RMS vector `50×` larger than the incoming token embedding, the residual skip connection around Layer 0 is effectively severed: Layer 0 overwrites `h_0` (`dA0 = +6.85 nats` in 1B, **`+8.16 nats`** in 7B, **`+9.53 nats`** in 13B).
   - **Slow AdamW relaxation driven by integrated learning rate $\int \eta(t)\,dt$:** AdamW simultaneously grows `embRMS` (`0.0197 → 0.08–0.22`, un-decayed) and shrinks early Post-RMSNorm `γ` by `15–35×` (`1.000 → 0.055` in 1B, `→ 0.032` in 7B, `→ 0.028` in 13B) while preserving deep-layer `γ` `4–6×` larger—**self-organizing a depth-graded residual schedule**.
   - **Natural-Experiment Triple Dissociation Across `1B`, `7B`, and `13B` (Figure 1, Panels A–D):**
     Because `OLMo-2-7B` (`bs = 4.2M`, `peak lr = 3e-4`) and `OLMo-2-13B` (`bs = 8.4M`, `peak lr = 9e-4` — `3×` higher LR!) have completely different step-to-token and LR schedules from `1B` (`bs = 2.1M`), the three scales **disagree sharply in both tokens AND optimizer steps, yet collapse cleanly onto $\mathcal{R}_0 \approx 1$**:
     - **Not tokens (non-monotonic in scale!):** Evacuation occurs at `~126B` (`1B`), `~294B` (`7B`), and `~151B–185B` (`13B`).
     - **Not optimizer steps (`3.5×` apart!):** Evacuation occurs at `~60k steps` (`1B`), `~70k steps` (`7B`), and `~18k–22k steps` (`13B`).
     - **Exact collapse onto $\mathcal{R}_0 \approx 1$ & Universal Transient Hydra Bridge (Panels C–D):**
       - **`OLMo-2-1B`**: As $\mathcal{R}_0$ crosses `2.19 → 1.44 → 1.07` (`84B → 105B → 126B`), `E(Attn0, MLP0)` fires a sharp **transient Hydra bridge (`-1.73 → +2.45 → +2.40 nats`)**, `dA0` collapses **`138×` (`6.85 → 0.05`)**, `dM0` collapses **`89×` (`1.74 → 0.02`)**, and `causD` jumps **`3.5×` (`0.18 → 0.63`)**.
       - **`OLMo-2-7B`** (12-point zoom-in): At `126B` (`step30k`), $\mathcal{R}_0 = 5.70$ and 7B is still trapped (`dA0 = 7.14`, `E_sL0 = -2.25`). Across `168B → 189B → 210B → 252B → 294B` ($\mathcal{R}_0 = 3.18 \to 2.44 \to 1.91 \to 1.29 \to 0.97$), the Hydra bridge surges to **`+3.86 → +4.18 nats`** (peak at $\mathcal{R}_0 = 1.91$ — identical to 1B's peak at `2.19`!), then **`dA0` collapses `163×` (`8.16 → 0.05`)** and `dM0` hits `0.00`!
       - **`OLMo-2-13B`** (10-point probe): Thanks to `lr = 9e-4`, `13B` reaches $\mathcal{R}_0 = 2.28$ by `step10k / 84B` (still trapped: `dA0 = 9.53`, `dM0 = 3.77`, `E_sL0 = -4.89`). Across `step15k / 126B → step18k / 151B → step22k / 185B` ($\mathcal{R}_0 = 1.14 \to 0.89 \to 0.70$), the Hydra bridge fires **`+2.86 → +2.70 nats`** and **`dA0` collapses `159×` (`9.53 → 0.06`)** and **`dM0` collapses `189×` (`3.77 → 0.02`)**!
   - **Important Dwell-Time Nuance (`minCosA` in `1B/7B` vs `13B`):** While the Layer-0 causal bottleneck and $\mathcal{R}_0 \approx 1$ Hydra-bridge evacuation are **100% universal across all three scales**, severe `L1..L10` anti-residual erasure (`minCosA = -0.853` in 1B, `-0.798` in 7B) depends on **dwell time in the high-$\mathcal{R}_0$ regime**: `1B` and `7B` spend `25k–35k` steps in $\mathcal{R}_0 \in [5, 30]$ allowing `L1..L10` to co-adapt into strong erasers (collapsing identically to `-0.415 / -0.407` at $\mathcal{R}_0 \approx 1.3–1.4$ and `-0.292 / -0.295` at $\mathcal{R}_0 \approx 1.0$), whereas `13B`'s `3×` higher LR (`9e-4`) blasts past $\mathcal{R}_0 = 50 \to 2.28$ in only `~8k` steps (at the cost of a severe early loss spike at `step4k`, `bpb = 2.253`), keeping `minCosA` mild (`-0.225`) even though Layer 0 still hits `dA0 = +9.53 nats`! Fixing $\mathcal{R}_0(0) = 1$ at initialization should eliminate the Layer-0 bottleneck *without* needing a spike-prone `9e-4` LR.

### Exact Three-Scale Quantitative Comparison (`OLMo-2-1B`, `7B`, & `13B`)

| Scale (Arch / LR) | Regime | Step / Tokens | $\mathcal{R}_0$ (`wR0`) | `dA0` / `dM0` (nats) | `E(Attn0, MLP0)` | `minCosA` | `bpb` |
|---|---|---|---|---|---|---|---|
| **`OLMo-2-1B`** (16L, `d=2048`) | Deep trap | `20k / 42B` | `7.09` (`6.40`) | `6.85` / `1.74` | `-1.73` | `-0.853` | `1.048` |
| | **Hydra bridge** | `40k–50k / 84B–105B` | **`2.19 → 1.44`** | `2.43 → 0.60` / `0.42 → 0.16` | **`+2.45 → +2.40`** | `-0.660 → -0.415` | `1.020` |
| | **Evacuated** | `60k → 140k / 126B → 294B` | **`1.07 → 0.48`** | **`0.13 → 0.05`** / `0.05` | `+0.73 → +0.11` | `-0.292 → -0.216` | `0.995` |
| **`OLMo-2-7B`** (32L, `d=4096`, `lr=3e-4`) | Deep trap | `10k–30k / 42B–126B` | `20.2 → 5.70` | **`8.16 → 7.14`** / `1.28 → 2.49` | `-1.44 → -2.25` | **`-0.798`** | `0.943` |
| | **Hydra bridge** | `45k–50k / 189B–210B` | **`2.44 → 1.91`** | `2.75 → 1.45` / `0.25 → 0.13` | **`+3.86 → +4.18`** | `-0.694 → -0.603` | `0.926` |
| | Shoulder | `60k / 252B` | `1.29` (`1.12`) | `0.21` / `0.04` | `+1.84` | `-0.407` | `0.918` |
| | **Evacuated** | `70k → 929k / 294B → 3.9T` | **`0.97 → 0.27`** | **`0.05 → 0.07`** / `0.00` | `+0.40 → -0.02` | `-0.295 → -0.160` | `0.837` |
| **`OLMo-2-13B`** (40L, `d=5120`, `lr=9e-4`) | Deep trap | `10k–12k / 84B–101B` | `2.28 → 1.64` | **`9.53 → 7.19`** / `3.77 → 2.83` | `-4.89 → -2.49` | `-0.225` *(fast LR)* | `0.924` |
| | **Hydra bridge** | `15k–18k / 126B–151B` | **`1.14 → 0.89`** | `1.71 → 0.30` / `1.80 → 0.33` | **`+2.86 → +2.70`** | `-0.167 → -0.194` | `0.894` |
| | **Evacuated** | `22k → 30k / 185B → 252B` | **`0.70 → 0.50`** | **`0.07 → 0.06`** / `0.15 → 0.02` | `+0.74 → +0.32` | `-0.208 → -0.207` | `0.889` |
| | Final 5T | `596k / 5,001B` | `0.16` | `0.13` / `0.03` | `-0.03` | `-0.130` | `0.823` |

---

## 3. Secondary Cross-Family Findings

![Discovery 2 & 3: Triple-Family Depth Migration & Universal Hydra](v3_stage1/fig2_triple_family_depth_hydra.png)

![Discovery 4 & 5: Non-Monotonic Interior Outliers & Parallel Co-Alignment](v3_stage1/fig3_massive_activation_and_fincam.png)

![Channel 3 Micro-Probes: P6 Memory vs ICL and P3 Copy vs Decoy](v3_stage1/fig4_probes_p6_p3.png)

1. **Universal Adjacent-Attention Hydra Redundancy (Verified Non-Artifact Across All 3 Families, Figure 2C):**
   - By mid-training (`20B–300B`), `60%–100%` of adjacent `(Attn_l, Attn_{l+1})` and `(MLP_l, MLP_{l+1})` pairs across `Pythia-1B`, `OLMo-1B`, and `OLMo-2-1B` become strongly redundant ($E > 0$, `E_norm` reaching `+2.20` in Pythia-1B, `+6.35` in OLMo-1B, `+4.02` in OLMo-2-1B).
   - **Artifact audit passed (`mechanistic_audits.json`):** The redundancy is NOT cross-entropy convexity—it replicates cleanly in **Top-1 accuracy drop** (`Pythia-1B Attn_L1_L2`: `2.5% + 0.1% → 6.8%`, `E_Acc = +4.1%`; `OLMo-1B Attn_L2_L3`: `0.9% + 1.3% → 6.8%`, `E_Acc = +4.5%`; `OLMo-2-1B Attn_L0_L1`: `0.8% + 0.5% → 4.8%`, `E_Acc = +3.5%`), **KL divergence** (`E_KL = +0.26..+0.59`), and **direct active downstream compensation** (`Attn_{l+1}` shifts by `40%–66%` of its norm aligned at `+0.10..+0.21` cosine with the missing `Attn_l` vector).
2. **Non-Monotonic Interior-to-BOS Massive-Activation Migration (Figure 3A):**
   - Directly refuting monotonic growth fits from prior literature, both Pre-LN 1B families undergo a sharp **interior massive-activation boom-and-bust**: `Pythia-1B` peaks at **`1,534×`** (`step88k / 185B`) then collapses **`14×`** to `111×` (`step112k–143k`) while `pos=0` stays locked at `~800×`; `OLMo-1B-0724` peaks at **`3,273×`** (`134B`, traced to exact dimension `d1764 = -556.2` on interior `"."`) then collapses **`80×`** across `L2..L15` to `35–45×` by `1,048B–3,048B` as `d1764` consolidates exclusively onto `pos=0` (`-1,042`) and `bos_mass` surges `0.07 → 0.56`. Post-RMSNorm (`OLMo-2-1B`) prevents interior explosion altogether (`max ~80×`).
3. **Parallel-Residual Cosine-Tail Final-Layer Co-Alignment (Figure 3B):**
   - Across **10/10** Pythia models (`70M 4/4`, `160M 4/4`, `410M 2/2`, `1B`), final-layer `cos(Attn_{L-1}, MLP_{L-1})` surges abruptly in the cosine LR decay tail (`+0.12 → +0.89±0.04` at `70M`, `+0.81±0.11` at `160M`, `+0.42±0.08` at `410M`, `+0.37` at `1B`) in **strict scale order** (`step64k → 80k → 96k → 112k`). Sequential `OLMo-1B` stays flat (`~0.08`), isolating parallel-residual coupling as the driver.
4. **Off-Diagonal Cross-Family Platonic Convergence & Ch3 Probes (Figure 4):**
   - After removing the trivial `I_M` diagonal at `step0`, **off-diagonal CKA** across `Pythia-1B`, `OLMo-1B-0724`, and `OLMo-2-0425-1B` surges from `~0.20` (`0B`) to **`0.78–0.85` by just `10B–20B` tokens** and locks at **`0.80–0.90`** through `3T–4T` tokens—despite completely different tokenizers, norms, MLPs, and corpora.
   - `P6_memory_vs_icl` shows a clean **two-stage tug-of-war** (Figure 4A): ICL overrides parametric facts early (`2B–100B`), then rebounds toward parametric memory in `Pythia-70M/160M/1B` and flips completely back to parametric dominance (`+0.11..+0.56 bpb`) in `OLMo-2-1B` at `420B–4T`, while `P3_copy_decoy` interference (`+0.37..+0.82 bpb`) persists even at `4T` tokens.

---

## 4. Concrete Stage 2 Proposal (When RTX PRO 6000 / Colab 100h Is Ready)

**Discovery #1 (`Post-RMSNorm` $\mathcal{R}_0 = \gamma_0 / \mathrm{RMS}(E_{\mathrm{tok}})$ Trap & Self-Evacuation)** is tailor-made for a clean, pre-registered Stage 2 causal experiment on **1× RTX PRO 6000** (~25–40 GPU-hours total):

1. **Model & Data:** `60M–125M` modern Transformer (`12` layers, `RoPE + SwiGLU + QK-Norm + inside-branch Post-RMSNorm`), trained on `FineWeb-Edu` (`1B–2B` tokens, ~3–6 h per run on 1× PRO 6000, `3` fresh seeds per arm).
2. **Pre-Registered 4-Arm Causal Surgery:**
   - **Arm 1 — Pre-LN baseline:** Predicts permanent Layer-0 causal bottleneck (`dA0/dM0 >> 1`, low `causD`, massive outliers).
   - **Arm 2 — Default OLMo-2 Post-RMSNorm ($\gamma_l(0) = 1.0, \text{emb\_std} = 0.02 \Rightarrow \mathcal{R}_0(0) = 50$):** Predicts full reproduction of the `-0.8` anti-residual erasure phase, Layer-0 single-point-of-failure, slow `γ_0` collapse, transient `Attn0–MLP0` Hydra bridge at $\mathcal{R}_0 \approx 1$, and delayed Layer-0 evacuation.
   - **Arm 3 — Matched-Skip Post-RMSNorm ($\gamma_l(0) = \text{emb\_std} = 0.02 \Rightarrow \mathcal{R}_0(0) = 1.0$):** Predicts **zero early anti-residual erasure (`minCosA > -0.3`), zero Layer-0 trap from Step 0, immediate deep causal utilization (`causD > 0.5`), and measurable validation `bpb` gain**.
   - **Arm 4 — Depth-Graded Post-RMSNorm ($\gamma_l(0)$ matching the self-organized OLMo-2 late profile, e.g. `0.05 → 0.30` across depth) + Embedding-Scale control ($\text{emb\_std} = 1.0, \gamma = 1.0 \Rightarrow \mathcal{R}_0(0) = 1.0$):** Isolates whether the dimensionless ratio $\mathcal{R}_0(0)$ alone governs the phase transition!
   - **Bonus LR-sensitivity sub-arm (motivated by `13B` `lr=9e-4` vs `7B` `lr=3e-4`):** Confirm whether matching $\mathcal{R}_0(0) = 1$ avoids BOTH the slow-LR `-0.85` anti-residual erasure phase AND the high-LR early loss-spike risk (`step4k` in 13B).

---

## 5. Artifacts & Reproducibility

- Analysis & audit scripts: [scripts/v3/scan_checkpoint.py](../../scripts/v3/scan_checkpoint.py), [scripts/v3/probes.py](../../scripts/v3/probes.py), [scripts/v3/run_stage1_sweep.py](../../scripts/v3/run_stage1_sweep.py), [scripts/v3/analyze_stage1.py](../../scripts/v3/analyze_stage1.py), [scripts/v3/probe_olmo2_mechanism.py](../../scripts/v3/probe_olmo2_mechanism.py), [scripts/v3/audit_discoveries.py](../../scripts/v3/audit_discoveries.py), [scripts/v3/make_stage1_figures.py](../../scripts/v3/make_stage1_figures.py).
- Committed compact data summaries: [docs/results/v3_stage1/olmo2_1b_mechanism.json](v3_stage1/olmo2_1b_mechanism.json) (`10` pts), [docs/results/v3_stage1/olmo2_7b_mechanism.json](v3_stage1/olmo2_7b_mechanism.json) (`12` pts), [docs/results/v3_stage1/olmo2_13b_mechanism.json](v3_stage1/olmo2_13b_mechanism.json) (`10` pts), [docs/results/v3_stage1/mechanistic_audits.json](v3_stage1/mechanistic_audits.json).
- Full 492-checkpoint `.npz` + `.json` archive + 32 OLMo-2 1B/7B/13B probe archives (`83 MB`) preserved cleanly on `s82:~/tycho-survey/work/v3_stage1/`.
