# Tycho Survey `v3` — Stage 1 Technical Specification (Draft v0.1)

> **Parent Roadmap:** [docs/NEW_ROADMAP.md](NEW_ROADMAP.md)
> **Status:** Draft v0.1 (ready for prototype validation on `s82` / RTX PRO 6000 and review by Claude before launching the full Stage 1 sweep).
> **Design Philosophy:** Measure structural, causal, and representational dynamics across **two independent modern LLM pretraining trajectories** (`Pythia` and `OLMo-1/2`) using **tokenizer-agnostic** and **architecture-unified** observables, with strict instrument checks (`IC-v3`) and hard Stage 1 → Stage 2 Kill Gates.

---

## 1. Target Sky: Verified Model Families & Checkpoint Grids

All HuggingFace repository IDs and branch naming schemes below were verified live against the HuggingFace Hub Refs API on 2026-11-27.

### 1.1 Dual-Family Architecture & Corpus Matrix

| Family | Repo ID(s) | Architecture | Corpus | Total Checkpoints | Primary Role |
|---|---|---|---|---|---|
| **Pythia (scale ladder)** | `EleutherAI/pythia-{14m,31m,70m,160m,410m,1b}` | **Parallel** Attn+MLP, Pre-LN, GELU, partial RoPE | The Pile (`~300B` tok; `2.097M` tok/step) | `154` (`step0..step143000`) | Scale trajectory (`14M → 1B`) & early log-step resolution (`step0..512`) |
| **Pythia (multi-seed)** | `EleutherAI/pythia-{14m,70m,160m,410m}-seed{1,2,3}` | Same as above | Same as above (different init & batch order) | `154` each | Seed-variance & fluke rejection |
| **OLMo-1 (0724)** | `allenai/OLMo-1B-0724-hf` | **Sequential** Pre-LN, SwiGLU, full RoPE | Dolma v1.7 (`3,048B` tok; `~2.1B` tok/1k steps) | `1,446` (`step0-tokens0B .. step1454000-tokens3048B`) | Matched `0–300B` token cross-family control + long-horizon `3T` tail |
| **OLMo-2 (0425)** | `allenai/OLMo-2-0425-1B` | **Sequential** Post-RMSNorm + **QK-Norm**, SwiGLU, full RoPE | OLMo-2-mix-1124 (`4,001B` tok, Stage 1) | `195` (`stage1-step0-tokens0B .. stage1-step1907359-tokens4001B`) | Modern post-norm + QK-Norm control |

**Why three distinct normalization/residual topologies matter:** Any phenomenon that appears across Pythia (parallel Pre-LN + GELU + Pile), OLMo-1 (sequential Pre-LN + SwiGLU + Dolma), and OLMo-2 (sequential Post-RMSNorm + QK-Norm + SwiGLU + OLMo-2-mix) cannot be dismissed as a corpus-order glitch, parallel-residual quirk, or lack of QK-Norm.

### 1.2 Canonical Stage 1 Checkpoint Grids

To capture both rapid early circuit formation (`0–10B` tokens) and slow mid/late reorganization (`10B–300B+` tokens) within **5–15 GPU-hours** total:

- **Pythia Grid (36 checkpoints per model):**
  - Ultra-early log grid (11): `step0, step1, step2, step4, step8, step16, step32, step64, step128, step256, step512` (`0` to `1.07B` tokens)
  - Early-to-mid grid (15): `step1000, step2000, step3000, step4000, step5000, step6000, step8000, step10000, step12000, step16000, step20000, step25000, step32000, step40000, step50000` (`2.1B` to `105B` tokens)
  - Late grid (10): `step64000, step80000, step96000, step112000, step128000, step135000, step140000, step141000, step142000, step143000` (`134B` to `300B` tokens, densified at the final cosine-decay tail)
- **OLMo-1B-0724 Grid (34 checkpoints, matched token budget + late tail):**
  - Matched `0–300B` window (22): `step0-tokens0B, step1000-tokens2B, step2000-tokens4B, step3000-tokens6B, step4000-tokens8B, step5000-tokens10B, step6000-tokens12B, step8000-tokens16B, step10000-tokens20B, step12000-tokens25B, step16000-tokens33B, step20000-tokens41B, step25000-tokens52B, step32000-tokens67B, step40000-tokens83B, step50000-tokens104B, step64000-tokens134B, step80000-tokens167B, step96000-tokens201B, step112000-tokens234B, step128000-tokens268B, step143000-tokens299B`
  - Long-horizon `300B–3T` tail (12): roughly log-spaced checkpoints up to `step1454000-tokens3048B` (exact branch names resolved programmatically via HF Refs API before launch).
- **OLMo-2-0425-1B Stage-1 Grid (26 checkpoints):**
  - `stage1-step0-tokens0B, stage1-step300-tokens1B, stage1-step10000-tokens21B, stage1-step20000-tokens42B, stage1-step30000-tokens63B, ..., stage1-step140000-tokens294B` plus 10 log-spaced points to `stage1-step1907359-tokens4001B`.

*(Multi-seed Pythia runs `seed1..seed3` run first on `160m` at a 16-point subset, and expand to full 36 points / `70m` / `410m` only on checkpoints bracketing candidate anomalies.)*

---

## 2. Unified Residual-Block Abstraction (Parallel / Pre-LN / Post-RMSNorm)

A silent bug trap when comparing Pythia, OLMo-1, and OLMo-2 is hooking raw sub-layer outputs before post-normalization. To ensure **exact mathematical equivalence** across all three architectures, every Transformer layer $l \in \{0, \dots, L-1\}$ is wrapped in a unified residual-delta interface:

- **Parallel residual (Pythia / `GPTNeoXLayer` with `use_parallel_residual=True`):**
  $$h_{l+1} = h_l + \Delta h_{\text{attn}, l}(h_l) + \Delta h_{\text{mlp}, l}(h_l)$$
  Input to both blocks is $h_{\text{in}, l, \text{attn}} = h_{\text{in}, l, \text{mlp}} = h_l$.
- **Sequential Pre-LN (OLMo-1 / `OlmoDecoderLayer`):**
  $$h_{l+1/2} = h_l + \Delta h_{\text{attn}, l}(h_l), \qquad h_{l+1} = h_{l+1/2} + \Delta h_{\text{mlp}, l}(h_{l+1/2})$$
  Inputs are $h_{\text{in}, l, \text{attn}} = h_l$ and $h_{\text{in}, l, \text{mlp}} = h_{l+1/2}$.
- **Sequential Post-RMSNorm (OLMo-2 / `Olmo2DecoderLayer`):**
  $$\Delta h_{\text{attn}, l} \equiv \text{PostRMSNorm}_{1,l}(\text{SelfAttn}_l(h_l)), \qquad \Delta h_{\text{mlp}, l} \equiv \text{PostRMSNorm}_{2,l}(\text{MLP}_l(h_{l+1/2}))$$
  Both capture and ablation operate strictly on the **post-norm residual write** $\Delta h_{l,b}$—never before post-norm (where zeroing would divide by $\epsilon$ or leave norm-scale artifacts).

Every model run automatically executes **`IC-v3.2` (Residual Identity Check)** verifying `max |h_{l+1} - (h_l + delta_attn + delta_mlp)| < tol` across all layers before recording any metrics.

---

## 3. The Four Observation Channels

All 4 channels run inside a single checkpoint pass (`scripts/v3/scan_checkpoint.py`) with zero training. Core batch $D_{\text{core}}$ is a frozen, seeded set of `N_core` diverse real-text passages (`N_core = 64`, `T = 128` tokens in standard GPU sweep; configurable `--core-seqs` / `--seq-len` for CPU smoke tests). Position `t = 0` (BOS / first token) is excluded from bulk averages and tracked explicitly in Channel 4 so attention-sink / massive-activation spikes do not pollute interior statistics.

### 3.1 Channel 1 — Layer-wise Attn vs MLP Write & Circuit-Migration Dynamics

For each layer $l \in \{0, \dots, L-1\}$ and block $b \in \{\text{attn}, \text{mlp}\}$ on $D_{\text{core}}$ ($t \ge 1$):

1. **Residual Write RMS:**
   $$w_{l,b} = \sqrt{\mathbb{E}_{x,t \ge 1}\!\left[\frac{1}{d}\|\Delta h_{l,b}(x,t)\|_2^2\right]}$$
2. **Relative Perturbation Ratio:**
   $$\rho_{l,b} = \mathbb{E}_{x,t \ge 1}\!\left[\frac{\|\Delta h_{l,b}(x,t)\|_2}{\|h_{\text{in},l,b}(x,t)\|_2 + 10^{-8}}\right]$$
3. **Attn vs MLP Write Energy Share:**
   $$\alpha_l = \frac{w_{l,\text{attn}}^2}{w_{l,\text{attn}}^2 + w_{l,\text{mlp}}^2} \in [0, 1]$$
4. **Write–Stream Cosine Alignment** (negative values flag erasure / cancellation writes):
   $$c_{l,b} = \mathbb{E}_{x,t \ge 1}\!\left[\frac{\langle \Delta h_{l,b},\, h_{\text{in},l,b}\rangle}{\|\Delta h_{l,b}\|_2 \|h_{\text{in},l,b}\|_2 + 10^{-8}}\right]$$
5. **Within-Layer Attn–MLP Write Cosine** (cooperative vs opposing writes):
   $$c^{\text{AM}}_l = \mathbb{E}_{x,t \ge 1}\!\left[\frac{\langle \Delta h_{l,\text{attn}},\, \Delta h_{l,\text{mlp}}\rangle}{\|\Delta h_{l,\text{attn}}\|_2 \|\Delta h_{l,\text{mlp}}\|_2 + 10^{-8}}\right]$$
6. **Logit-Lens Depth Profile & Marginal Gain:**
   $$\mathcal{L}_{\text{lens}}(l) = \text{CE}\!\left(\text{Unembed}(\text{LN}_f(h_{l+1})),\, y\right), \qquad \Delta\mathcal{L}_{\text{lens}}(l) = \mathcal{L}_{\text{lens}}(l-1) - \mathcal{L}_{\text{lens}}(l)$$
   (with $h_0$ giving $\mathcal{L}_{\text{lens}}(-1)$, the embedding-only baseline), reported in **bits per byte (`bpb`)** as well as nats/token for tokenizer invariance.
7. **Direct Logit Attribution (DLA) of Each Block:**
   Projecting $\Delta h_{l,b}$ directly through the final layer norm scale (frozen from clean $h_L$) and unembedding matrix onto the correct next-token logit minus mean logit:
   $$\text{DLA}_{l,b} = \mathbb{E}_{x,t \ge 1}\!\left[\left\langle W_{U, y_t} - \bar{W}_U,\; \frac{\gamma_f \odot \Delta h_{l,b}}{\text{RMS}(h_L(x,t)) + \epsilon}\right\rangle\right]$$
   Separates **direct output writers** from **upstream feature builders**.

### 3.2 Channel 2 — Block-Level Causal Necessity & Pairwise Synergy / Redundancy

Evaluated on a deterministic sub-batch $D_{\text{causal}} \subset D_{\text{core}}$ (`16` passages × `128` tokens = `2,048` positions; exact subset fixed across all checkpoints).

- **Primary intervention — Mean Ablation:** Replace $\Delta h_{l,b}(x, t)$ at all $t \ge 1$ with its batch-and-position mean vector $\bar{\mu}_{l,b} = \mathbb{E}_{x, t \ge 1}[\Delta h_{l,b}(x, t)]$ (preserving DC offset so downstream LayerNorms stay in-distribution while destroying token-specific information).
- **Secondary audit intervention — Zero Ablation:** Replace $\Delta h_{l,b}(x, t)$ with $0$ for single-block passes only, so any DC-bias / sink-only layer is immediately flagged by large `ΔL_zero` alongside tiny `ΔL_mean`.
- **Single-Block Causal Necessity** (`2L` passes, measured in both nats and `bpb`):
  $$\Delta \mathcal{L}(b) = \mathcal{L}(\text{ablate } b) - \mathcal{L}_{\text{clean}}$$
- **Pairwise Block Interaction (Budget-Bounded $O(L)$ Set $\mathcal{P}$):**
  Defined via **Excess Joint Loss** (explicit sign convention!):
  $$E(i, j) \equiv \Delta \mathcal{L}(\{i, j\}) - \Delta \mathcal{L}(i) - \Delta \mathcal{L}(j), \qquad \tilde{E}(i, j) \equiv \frac{E(i, j)}{\max(|\Delta \mathcal{L}(i)| + |\Delta \mathcal{L}(j)|,\, 10^{-3})}$$
  - **$E(i, j) > 0$ — Redundancy / Backup / Hydra compensation:** ablating either block alone is partially compensated; removing both exposes the shared load.
  - **$E(i, j) < 0$ — Serial / AND-gate Composition:** either block alone already breaks the joint computation; removing the second adds less marginal damage.
  - **$E(i, j) \approx 0$ — Additive / Independent contributions.**
- **Structured Pair Set $\mathcal{P}$ (total $5L - 4$ pairs: `56` pairs at `12L`, `76` at `16L`, `116` at `24L`):**
  1. All $L$ same-layer `(Attn_l, MLP_l)` pairs;
  2. All $L - 1$ adjacent `(Attn_l, Attn_{l+1})` pairs;
  3. All $L - 1$ adjacent `(MLP_l, MLP_{l+1})` pairs;
  4. All $L - 1$ cross-layer adjacent `(Attn_l, MLP_{l+1})` pairs and all $L - 1$ adjacent `(MLP_l, Attn_{l+1})` pairs.

### 3.3 Channel 3 — Matched Stress-Test Micro-Probe Divergence (Tokenizer-Agnostic)

All probes are stored as deterministic UTF-8 text pairs (`scripts/v3/probes.py`), evaluated via **exact continuation bits-per-byte (`bpb`)** and **pairwise log-probability margin** $\Delta\log P = \log P(\text{target} \mid \text{ctx}_A) - \log P(\text{target} \mid \text{ctx}_B)$ (or $\log P(\text{good} \mid \text{ctx}) - \log P(\text{bad} \mid \text{ctx})$), making every curve **directly comparable across Pythia, OLMo-1, and OLMo-2 despite different tokenizers**:

| Probe ID | Paired Contrast | Metric | Phenomenon / Crossover Targeted |
|---|---|---|---|
| `P1_order` | Clean natural passage vs local-window word-shuffled passage (target: final clause) | $\Delta\text{bpb} = \text{bpb}_{\text{shuf}} - \text{bpb}_{\text{clean}}$ | Syntactic / word-order sensitivity vs bag-of-words |
| `P2_horizon` | Full paragraph context vs truncated 12-word suffix (same target clause) | $\Delta\text{bpb} = \text{bpb}_{\text{short}} - \text{bpb}_{\text{long}}$ | Long-range context utilization |
| `P3_copy_decoy` | Clean key-value retrieval (`Key A -> Val1 ... Query A -> Val1`) vs **decoy interference** (near-duplicate distractor key/value placed before or after) | $\Delta\log P(\text{Val1}) - \Delta\log P(\text{Decoy})$, accuracy | Direct real-text generalization of `relay` decoy binding vs surface copying |
| `P4_induction_vs_break` | Repeated sequence continuation (`A B C D ... A B C -> D`) vs **anti-repeat** broken continuation (`A B C D ... A B C -> X`) | Margin shift & anti-repeat penalty | Pure induction bias vs contextual override |
| `P5_agreement_attractor` | Grammatical vs ungrammatical verb agreement (`is/are`, `was/were`, `has/have`), **simple** vs **intervening plural/singular attractor** | $\Delta\log P_{\text{simple}}$ vs $\Delta\log P_{\text{attractor}}$ | Hierarchical syntax vs local recency heuristic |
| `P6_memory_vs_icl` | Clean parametric fact (`"The capital of France is Paris"`) vs **in-context counterfactual override** (`"In this story, the capital of France is Lyon..."`) | $\log P(\text{ICL}) - \log P(\text{Parametric})$ | Exact collision / crossover between parametric memory and in-context copying |
| `P7_binding_2_vs_5` | Entity-attribute variable binding with **2 entities** vs **5 entities** (same query format) | Accuracy & margin gap ($\text{2-ent} - \text{5-ent}$) | Multi-entity binding capacity scaling & interference |
| `P8_arith_carry` | 2-digit addition **no-carry** (`23 + 41 = 64`) vs **with-carry** (`28 + 47 = 75`) | Exact-answer `bpb` & margin gap | Compositional algorithmic structure vs local digit heuristics |

*(Standard GPU sweep: `32–64` items per probe pair; CPU smoke test: `16–32` items per pair, all deterministically indexed so smoke subsets are exact prefixes of full probes.)*

### 3.4 Channel 4 — Dark Representation Velocity & Geometry Transients

1. **Tokenizer- & Width-Agnostic Prompt-Endpoint Gram Matrices ($K_l \in \mathbb{R}^{M \times M}$):**
   - Take $M = 64$ (full sweep: `128`) frozen mid-sentence prompt prefixes sliced from `CORE_PASSAGES` so that every prompt ends on a **distinct alphabetic content word** (avoiding repeated terminal punctuation like `.` or `->`, which would otherwise collapse $K_0$ into a trivial rank-1 token-embedding cluster).
   - Extract each layer's residual vector $h_{l+1}(x_m, t_{\text{last}})$ at the **exact final token of prompt $m$** (so row $m$ corresponds to the same semantic prediction point across Pythia, OLMo-1, and OLMo-2!).
   - Mean-center across the $M$ prompts ($\tilde{H}_l$) and store the exact centered Gram matrix $K_l = \tilde{H}_l \tilde{H}_l^\top$ in FP32 (`128 × 128 × 4 B = 64 KB/layer`, or `256 × 256 × 4 B = 256 KB/layer` — only `0.8–4 MB` per checkpoint, **independent of $d_{\text{model}}$**!).
   - Offline on `s82`, compute:
     - **Temporal representation velocity:** $v_l(t) = 1 - \text{CKA}(K_l(t), K_l(t+1))$ where $\text{CKA}(A, B) = \frac{\langle A, B\rangle_F}{\|A\|_F \|B\|_F}$;
     - **Dark-transition ratio:** $v_l(t) / \max(|\Delta\text{bpb}_{\text{core}}(t)|, 10^{-3})$;
     - **Cross-layer alignment matrix** $\text{CKA}(K_{l_1}(t), K_{l_2}(t))$;
     - **Cross-family representational convergence** $\text{CKA}(K^{\text{Pythia}}_{l_1}(t), K^{\text{OLMo}}_{l_2}(t))$ at matched token budgets!
2. **Per-Layer Effective Rank (Raw & Outlier-Decoupled):**
   Log both raw $\text{eRank}(H_l) = \exp\!\left(-\sum_i p_i \ln p_i\right)$ on centered interior states ($t \ge 1$) and **feature-standardized $\text{eRank}_{\text{normed}}(H_l)$** (dividing each coordinate by its empirical standard deviation across interior tokens) so true subspace dimensionality is decoupled from 1–2 massive-activation coordinates.
3. **Massive-Activation / Outlier Ratio (Tracked Separately for `t = 0` vs `t >= 1`):**
   $r^{\text{outlier}}_l = \max |h_l| / (\text{median}|h_l| + 10^{-8})$ — explicitly isolates Sun et al. (2024) massive-activation / attention-sink formation so it is never mistaken for an unexplained anomaly.
4. **Attention Entropy & BOS-Sink Mass:**
   Mean normalized attention entropy $\bar{H}_{\text{attn}, l} = \mathbb{E}_{h, t \ge 4}\!\left[\frac{-\sum_{j \le t} A_{h,t,j}\log A_{h,t,j}}{\log(t+1)}\right]$ and mean attention mass on position `0` ($A_{h, t \ge 4, 0}$), per layer and per head.

---

## 4. Instrument Checks (`IC-v3`)

Every scanner invocation runs and logs four built-in instrument checks before trusting outputs:

| Check | Definition | Pass Criterion |
|---|---|---|
| **`IC-v3.1` (Determinism)** | Re-run core forward pass & Channel 1 on the same batch | `max |diff| <= 1e-6` (FP32) / `<= 1e-4` (BF16) |
| **`IC-v3.2` (Residual Identity)** | Verify `h_{l+1} == h_l + delta_attn + delta_mlp` at every layer $l$ (handles Pythia parallel, OLMo-1 Pre-LN, OLMo-2 Post-RMSNorm) | `max_rel_err = max |diff| / (max |h_out| + 1e-8) <= 1e-6` (FP32) / `<= 1e-2` (BF16) (absolute error also logged; accounts for massive-activation FP32 ULPs) |
| **`IC-v3.3` (Ablation Identity)** | Verify (a) empty ablation matches clean loss (`0.0` diff) and (b) zero-ablating ALL $2L$ blocks reproduces embedding-only logit-lens loss $\mathcal{L}_{\text{lens}}(-1)$ | (a) `== 0.0`; (b) `|diff| <= 1e-5` (FP32) / `<= 1e-2` (BF16) |
| **`IC-v3.4` (Step-0 Null Sanity)** | On `step0` checkpoint, verify clean loss matches random vocab entropy (`≈ ln V`), probe margins `≈ 0`, and mean-ablation `|ΔL|` is tiny | Logged explicitly on `step0` baseline |

---

## 5. Stage 1 → Stage 2 Hard Kill Gates

An anomaly surfaced by the offline `s82` detector from Stage 1 is **promoted to Stage 2 training ONLY IF it clears all four gates**:

1. **Gate 1 — Non-Artifact & Multi-Seed Survival:** Survives `IC-v3.1–3.4`, bootstrap resampling of probe items (95% CI excludes 0), and replicates across `pythia-160m-seed{1,2,3}` (≥ 3/4 seeds including main).
2. **Gate 2 — Cross-Family & Cross-Scale Survival:** Replicates in **both Pythia AND at least one OLMo family** (`OLMo-1B-0724` or `OLMo-2-0425-1B`) at comparable training stages, and persists or follows a coherent scaling trend across Pythia `70M → 160M → 410M → 1B`.
3. **Gate 3 — Literature Clearance:** Deep read of primary papers confirms the phenomenon or its causal mechanism is not already documented/solved (explicitly filtering out known induction-head onset, Sun et al. massive activations, attention sinks, and standard grokking/loss-schedule steps).
4. **Gate 4 — Falsifiable Causal Handle:** Admits a concrete training-time intervention testable at `50M–125M` scale within `30–60` PRO 6000 GPU-hours.

If zero anomalies pass all 4 gates in Round 1, we **do not** launch blind Stage 2 training; we upgrade the telescope channels (e.g., gradient/Hessian alignment or SAE-feature interaction tracking) and re-run a cheap Stage 1 scan.

---

## 6. Artifact & Storage Layout

- Code: `scripts/v3/probes.py` (frozen tokenizer-agnostic probe generator), `scripts/v3/scan_checkpoint.py` (unified 4-channel scanner + `IC-v3`), `scripts/v3/analyze_stage1.py` (offline `s82` trajectory & anomaly analyzer, to be added once multi-checkpoint scans begin).
- Per-checkpoint output: one compact `.npz` (`~1–4 MB`, holding all matrices & Gram sketches) + one human-readable `.json` (`~30–60 KB`, holding scalar summaries, Channel 3 probe margins, timing, memory, and `IC-v3` report).
- Heavy HuggingFace checkpoint weight caches live strictly on local scratch (`/tmp` or `HF_HOME` outside git) and are cleaned/LRU-evicted automatically so disk stays well under `50 GB`.
