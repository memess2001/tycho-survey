# Tycho Survey — New Roadmap (`v3`: Modern-Architecture & Scale Discovery)

> **Status:** Adopted by the researcher on **2026-11-26** (following full autonomy from Prof. Mahito Sugiyama to set the research direction).
> **Relation to `docs/ROADMAP.md`:** [docs/ROADMAP.md](ROADMAP.md) is **intentionally preserved unmodified** as the historical record of `v0–v2b`. This document supersedes it as the active operational roadmap going forward.
> **Origin & Decision Context:** See [notes/conversation_2026-11-25_26_phase2_triage_v3.md](../notes/conversation_2026-11-25_26_phase2_triage_v3.md) for the verbatim discussion (Phase 2 confirmation, zero-training rapid triage of all `DR0–DR2b` anomalies, hardware audit, and the pivot to `v3`).

---

## 1. Executive Summary

The first phase of Tycho (`v0`, `v1`, `v2a`, `v2b`, and the `relay` confirmation branch) successfully built a rigorous experimental methodology—pre-registration, bit-identical replay checks (`IC-*`), common-probe verification, and independent two-agent auditing—and proved that small toy models (`1–2` layers, MLPs, attention-only learned-abs-pos Transformers) are **the wrong telescope for discovering major, previously unknown laws of modern LLM training dynamics**.

Every existing small-model anomaly in `DR0–DR2b` has now been resolved and closed out (§2). Going forward, Tycho pivots cleanly to **`v3`: surveying real modern Transformers (`RoPE + MLP/SwiGLU + multi-layer`) on real text**, executed as a **two-stage "Telescope → Surgical Lab" (A → B) pipeline** matched to our exact compute portfolio:

1. **Stage 1 — Cross-Family Public-Trajectory Telescope (Route A, ~1–2 weeks, 5–15 h on 1× RTX PRO 6000):** Scan full pretraining checkpoint trajectories across **two independent modern LLM families** (**Pythia 14M–1B** and **OLMo / OLMo-2 ~1B**) with 4 targeted structural/causal observation channels, offloading heavy post-hoc analysis to `s82` (`88 CPU cores / 400 GB RAM`).
2. **Stage 2 — Controlled Modern-Architecture Surgical Lab (Route B, ~2–4 weeks, 30–60 h on 1× RTX PRO 6000):** Only after an anomaly survives Stage 1's cross-family, cross-scale, and literature kill gates, reproduce it from scratch in **50M–125M modern Transformers (`6–12` layers, `RoPE + SwiGLU + RMSNorm`, real text)** and run pre-registered training-time causal interventions to establish mechanism.

---

## 2. Historical Ledger: Closing Out `v0–v2b` and the `relay` Branch

### 2.1 `relay` Branch Status (Close Out Independently — Do Not Block `v3`)

Branch `origin/relay-confirm` (`88aeb84`) completed both pre-registered phases on `s82`:
- **Phase 1 (`6942207`, `2a62fdc`; audited by Claude at `a884378`, [notes/relay_confirm_audit_2026-11-25.md](../notes/relay_confirm_audit_2026-11-25.md)):** 2,088 runs establishing the two-basin (`T` vs `R`) dynamics, two-stage assembly (`C4`), and `lr / B^0.32` scaling, while falsifying single-parameter closed-form formulas from the 3-seed exploratory rounds.
- **Phase 2 (`c285275` pre-reg → `88aeb84` results, [docs/results/v2_analysis/relay_confirm_phase2_summary.md](results/v2_analysis/relay_confirm_phase2_summary.md)):** 384 runs on 24 fresh seeds (`1048..1071`) under Claude's unified behavioral criterion `B_T` (`loss_rep < 0.3` and `decoy >= 0.9`), confirming:
  - 2L MLP joint Layer-0 scaling (`Attn0 × 0.5 + MLP0 × 0.5`) holds across fresh seeds, plain Adam `lr=5e-4`, and blind width `d=128` (`23/24`, `22/24`, `24/24` vs `0/24`);
  - 2L RoPE boundary holds (`1e-3`: `24/24`; blind `2e-3`: `17/24` with `3` relays; RoPE+MLP `2e-3` shows a `h2` flip);
  - 4L depth at `24k` steps confirms `D1-rope-24k` (`24/24`) stays relay-free while `D1-abs-24k` (`14/24`) vs `D1-abs-l01_05-24k` (`15/24`) confirms 4L scaling is not a net fix.
- **Disposition:** Await Claude's final Phase 2 audit, then package cleanly as a standalone **TMLR / Workshop** note or short paper. Because the phenomenon largely vanishes under `RoPE + 4L` and wider models (`d=256`) suffer negligible relay loss penalty, **no further GPU budget or weeks of theoretical patch-work will be spent trying to force `relay` into a main-conference paper**.

### 2.2 Zero-Training Rapid Triage of All `DR0–DR2b` Anomalies (2026-11-26)

Before pivoting to `v3`, all 8 open leads from `DR0`, `DR1`, `DR2a`, `DR2b`, and [notes/reframing_draft_2026-11-20.md](../notes/reframing_draft_2026-11-20.md) §5 were audited directly against raw tables, light curves, saved weights (`releases/DR2b/patterns.tar.gz`), and exact analytical gradients. **All 8 are resolved with zero further training needed:**

| # | Release & Anomaly | Zero-Training Diagnosis | Disposition |
|---|---|---|---|
| 1 | `DR2b` `d=192` sudden `9/20` strong `t−2` heads ([v2b_results.md](results/v2b_results.md) §7) | Saved-weight ablation on `1,024` sequences (`12.7%` intra-tile collision positions) shows core collision disambiguation at all widths is performed by the **diffuse multi-lag context head** (`+0.87` to `+2.03` nats on collisions). The sharp `t−2` head (`δ=2` mass `0.70–0.81`) only crystallizes late (step `3,245 → 4,636`, after loss `< 0.06`) in the spare 3rd L0 head and carries only **`+0.037` nats** single-head ablation cost (weak redundant backup; final loss `0.0064` vs `0.0075`). | **KILL / Archive** |
| 2 | `DR1` sparse parity `k=4` as fast as `k=3`, flat slope `~1.8–2.2` ([v1_results.md](results/v1_results.md) §2) | **PyTorch `nn.Linear` default bias init (`w, b ~ U(-1/√n, 1/√n)`) × ReLU odd/even symmetry.** Because `ReLU'(z) = 1(z>0)` is odd, its even derivatives (`k−1 = 2, 4` for odd `k = 3, 5`) vanish at `b = 0`, penalizing `k = 3` by an extra `b ~ n^(-1/2)`. Exact initial population gradient RMS under default init: `k = 3` (`7.34e-3 / 2.42e-3 / 8.20e-4`, `n^-1.59`) = `k = 4` (`6.86e-3 / 2.53e-3 / 8.03e-4`, `n^-1.56`) to two significant figures. Switching to `b ~ U(-1, 1)` immediately restores textbook `n^(-(k-1)/2)` (`-0.51 / -1.07 / -1.62 / -2.15`). | **KILL / Archive** |
| 3 | `DR2b` `6/10` 1-layer controls below `1.56` nats ([v2b_results.md](results/v2b_results.md) §8) | Re-evaluating all 10 saved 1-layer weights on a common `2,048`-sequence probe shifts the median from `1.549` to **`1.5756`** (mean `1.5725`), with `7/10 ≥ 1.56`. Caused by `128`-sequence probe variance plus 1-layer heads suppressing `δ = 0..3` to `0.00` (exploiting `rep_min = 4`). | **KILL / Archive** |
| 4 | `DR2b` leave loss falling `3.21 → 3.01 → 2.37` with `d_model` ([v2b_advisor_predictions.py](../scripts/analysis_v2/v2b_advisor_predictions.py)) | Standard-parameterization (SP) fixed-`lr` width speed-up: by step `51` (before induction moves, `max_ind ≈ 0.12`), `d = 192` has already dropped to `3.235` vs `d = 48` at `4.075`. Content-free bag-of-tokens learning accelerates faster (`~d^1`) than induction onset (`~d^-0.81`). | **KILL / Archive** |
| 5 | `DR2b` prev-token head cleanup delayed thousands of steps (`7,458 / 4,636 / 4,636`) ([v2b_results.md](results/v2b_results.md) §4) | Genuine, but overlaps with Phase 1 `C4` two-stage assembly (and Hoogland 2024 / Singh 2024); identical `4,636` medians at `d = 96, 192` reflect a shared log-grid checkpoint. | **Fold into `relay` note; do not open a new track** |
| 6 | `DR0` (6,791) & `DR1` (2,028) transients + Check 10's 13 offline anomalies ([v0_findings.md](results/v0_findings.md), [v1_results.md](results/v1_results.md) §6) | `DR0`: `98.3%` (6,674) are grokking ReLU-binarization (`E6`) noise. `DR1`: non-grokking alerts are already-analyzed Markov chain shortcuts and normal parity transitions; Check 10's 13 anomalies are 12 MAD=0 pooling artifacts + 1 early loss spike. | **KILL / Archive** |
| 7 | `DR1` grokking boundary refinement & `Q7` late interactions ([v1_results.md](results/v1_results.md) §5) | `Q7` inconclusive (`+7.4 pp [-10.1, +21.5]`); seed disagreement is 50k censoring; `1/T` linear scaling already covered by Nanda / Liu / Varma / Power ([discussion_2026-11-16_outside_reviews.md](../notes/discussion_2026-11-16_outside_reviews.md) §5.1). | **KILL / Archive** |
| 8 | `DR2a` `θ` vs `η` coordinate divergences on 6–8 bit MLPs & parity proper subsets (`≤ 0.055` nats) ([v2a_results.md](results/v2a_results.md) §7) | Gate 2 did not pass; toy binary-MLP measurement-coordinate detail that does not transfer to modern Transformers. | **KILL / Archive** |

---

## 3. Compute Portfolio & Role Assignment

Our hardware dictates the workflow: **ample single-GPU high-VRAM capacity + massive idle CPU/RAM, without depending on contested multi-GPU queues.**

| Resource | Availability | Role in `v3` |
|---|---|---|
| **Colab RTX PRO 6000 (96 GB VRAM)** | Guaranteed **100 h / month** | Primary GPU for Stage 1 public-checkpoint scanning and Stage 2 controlled `50M–125M` training runs. |
| **Lab 1× RTX PRO 6000 server (96 GB)** | Sometimes idle | Opportunistic mirror of Colab workload (faster local disk/checkpoint caching when free). |
| **`s82` (88-core CPU, 400 GB RAM)** | Routinely idle | Dedicated **offline analysis engine**: SVD/effective rank, CKA trajectories, probe regressions, synergy/redundancy matrices, bootstrap/pre-reg verification, anomaly detection, and plotting. Never waste GPU hours on post-processing. |
| **Lab 5× A100 server** | Rarely idle (heavily shared) | **Never a blocking dependency.** Opportunistic burst only (e.g., a single `300M+` scale anchor if Stage 2 succeeds). |

---

## 4. `v3` Architecture: Two-Stage "Telescope → Surgical Lab" (A → B)

### 4.1 Stage 1 — Cross-Family Public-Trajectory Telescope (Route A)

**Budget & Timeline:** ~1–2 weeks; **5–15 GPU-hours** on 1× RTX PRO 6000 + offline compute on `s82`.

**Target Sky (Dual Independent Families from Day 1):**
To eliminate corpus-order, learning-rate-schedule, and architecture-quirk false positives immediately, every observation channel runs across **two independent open-checkpoint LLM families**:
1. **Pythia family (EleutherAI):** `14M`, `31M`, `70M`, `160M`, `410M`, `1B` (plus multi-seed `14M–160M` for seed variance; RoPE + parallel Attn/MLP + LayerNorm, trained on The Pile; 154 public checkpoints, subsampled on a ~30–40 point log-plus-critical-window grid).
2. **OLMo / OLMo-2 family (AI2):** Comparable `~1B` (and smaller/larger siblings where public intermediate checkpoints exist; RoPE + sequential Attn/MLP + SwiGLU + RMSNorm, trained on Dolma / OLMo-2 mix).

**4 Lightweight, High-Discovery Observation Channels:**
Rather than repeating standard loss curves, benchmark tracking, or basic induction-head plots already covered in the Pythia literature, every scanned checkpoint logs four complementary structural channels (each checkpoint takes seconds on a 96 GB PRO 6000):

1. **Channel 1 — Layer-wise Attn vs MLP Write & Circuit-Migration Dynamics**
   - *Motivation:* Directly generalizes the sharpest lesson from `relay` (early Attn vs MLP residual write competition routes circuit formation) to deep modern LLMs.
   - *Measurements:* Per-layer residual write norms (`‖Δh_attn‖`, `‖Δh_mlp‖`, and their ratio), cosine alignment with the running residual stream (constructive vs cancellation/erasure writes), and per-layer logit-lens / tuned-projection contribution across training steps.
   - *Target anomaly:* Smooth total loss hiding **inter-layer role migration** (functions relocating from deep to shallow layers or shifting from Attention to MLP mid-training).
2. **Channel 2 — Block-Level Causal Necessity & Pairwise Synergy / Redundancy**
   - *Motivation:* Brings Tycho's original higher-order interaction spirit to 1B real LLMs at the **block (`Attn_l`, `MLP_l`)** level, where `2L = 24..48` makes exact knock-out cheap instead of exponentially intractable (`2^N`).
   - *Measurements:* Single-block zero/mean-ablation loss delta `ΔL(b)` for all `2L` blocks at every checkpoint, plus pairwise block interaction `I(i, j) = ΔL({i, j}) − ΔL(i) − ΔL(j)` (synergy vs redundancy) across key adjacent, same-layer (Attn–MLP), and top-necessity pairs.
   - *Target anomaly:* Abrupt **load-bearing phase transitions** (layers switching from redundant to indispensable, or Attn–MLP flipping between cooperative synergy and competitive cancellation) that scale systematically from `70M` to `1B`.
3. **Channel 3 — Matched Stress-Test Micro-Probe Divergence**
   - *Motivation:* Generalizes `v2a`'s twin-referee and `relay`'s decoy diagnostic to real text.
   - *Measurements:* A fixed, compact battery of **8–10 matched probe pairs** (`128–256` sequences each; `< 2 s` total per checkpoint): e.g., natural text vs local-order-shuffled, long-range copy vs distractor/decoy copy, frequent collocation vs compositional recombination, syntactic binding vs semantic-conflict, and repeated-template vs novel-context probes.
   - *Target anomaly:* **Non-monotonic trajectories or sharp divergence/crossovers** between matched probes (indicating hidden strategy switches or competing sub-circuits).
4. **Channel 4 — Dark Representation Velocity vs Loss Velocity**
   - *Measurements:* Per-layer checkpoint-to-checkpoint representation velocity `1 − CKA(t, t+1)`, residual effective rank, attention entropy spectrum, and outlier/massive-activation emergence timing—specifically normalized against local loss drop `|ΔLoss(t, t+1)|`.
   - *Target anomaly:* **High-velocity internal restructuring spikes during flat loss segments** ("dark transitions").

**Storage & Engineering Discipline:**
- Cache only compact summary tensors (`.npz` / `.parquet`, MBs per model, never full activation dumps) from PRO 6000 to `s82` / repo analysis workspace.
- Every probe set is frozen, seeded, and versioned before scanning; every channel includes a quick determinism/precision sanity check (`IC-v3`).

---

### 4.2 Stage 1 → Stage 2 Promotion Gate (Hard Kill Gate)

A candidate anomaly from Stage 1 is **promoted to Stage 2 training ONLY IF it passes ALL FOUR gates**:

1. **Non-artifact:** Survives probe-set resampling, numerical precision / LayerNorm checks, and Pythia multi-seed consistency (not a 1-seed fluke or single-checkpoint batch glitch).
2. **Cross-family & cross-scale:** Appears in **both Pythia and OLMo** (ruling out Pile-specific corpus order or parallel-residual quirks) and persists or follows a coherent trend across `70M → 410M → 1B`.
3. **Literature-cleared:** After a dedicated deep-read literature audit (not superficial keyword search), the phenomenon or its causal mechanism is genuinely unreported / unresolved.
4. **Actionable causal handle:** Admits a concrete, falsifiable training-time intervention that can be tested cleanly on a `50M–125M` modern model.

**If NO anomaly passes all 4 gates in Round 1:** **Do NOT force Stage 2.** Archive the negative Stage 1 scan honestly, diagnose why the 4 channels were blind, and iterate the telescope (e.g., adding gradient/Hessian alignment, SAE feature-interaction tracking, or targeted code/math/reasoning corpora) at low GPU cost before spending training hours.

---

### 4.3 Stage 2 — Controlled Modern-Architecture Surgical Lab (Route B)

**Budget & Timeline:** ~2–4 weeks; **30–60 GPU-hours** on 1× RTX PRO 6000 (`bf16 + FlashAttention + torch.compile`).

**Reference Model & Corpus:**
- Modern Transformer baseline: `50M–125M` parameters (`6–12` layers, `RoPE + SwiGLU + RMSNorm`, optional μP width scaling), trained on real text (`FineWeb-Edu` / curated web-text shards) for `1B–2.5B` tokens (~2–4 h per `50M` run, ~10–15 h per `125M` run on 1× PRO 6000).
- **Step 2a — Reproduction Anchor (3–5 seeds):** Confirm that the Stage 1 phenomenon reproduces cleanly in our own controlled training run before any intervention.
- **Step 2b — Pre-Registered Causal Surgery (fresh held-out seeds):** Freeze hypotheses, metrics, success/failure thresholds, and exact intervention arms (e.g., targeted layer write scaling, architectural swap, data-order/curriculum control, LR/batch/width scaling) **in git prior to running**, matching the rigor of `relay` Phase 1/2 (`IC-*` checks + independent audit).

**Stage 2 Venue Gate:**
- **Pass (reproduced + causal mechanism confirmed across fresh seeds + cross-family/cross-scale generality up to `1B`):** Target **NeurIPS / ICML / ICLR main track** (optionally adding one `300M+` anchor if the lab `5× A100` happens to be free).
- **Falsified / narrow scope:** Publish honestly as a short report / workshop / negative result and cycle immediately back to Stage 1—never salvage a falsified mechanism with post-hoc curve fitting.

---

## 5. Standing Methodological Rules (Non-Negotiable Lessons from `v0–v2b + relay`)

1. **No post-hoc laws from ≤ 3 seeds:** Exploratory observations generate hypotheses only; every claim destined for a paper requires pre-registered confirmation on fresh held-out seeds.
2. **Modern architecture & scale checks up front:** Never build weeks of theory on an artifact before checking `RoPE + MLP + multi-layer` and width/μP scaling.
3. **Common probes & exact replay (`IC-*`):** Always evaluate comparisons on common, sufficiently large probe sets and verify deterministic replay before trusting small numeric differences.
4. **Honest kill gates beat sunk-cost persistence:** Killing a dead lead in 30 minutes (as in the 2026-11-26 rapid triage) is a win, not a failure.
5. **Immutable history:** Keep [docs/ROADMAP.md](ROADMAP.md) untouched; record major revisions to `v3` explicitly in dated sections at the bottom of this file.

---

## 6. Immediate Next Actions

1. **Close out `relay` asynchronously:** Once Claude's quota resets, send the `origin/relay-confirm` (`88aeb84`) Phase 2 handoff and `origin/v3-roadmap` pointers for his audit and feedback.
2. **Design & freeze `v3` Stage 1 Specification (`docs/v3_stage1_spec.md`):**
   - Pin exact Pythia (`14M..1B` + multi-seed) and OLMo checkpoint lists and HuggingFace revisions;
   - Finalize the 8–10 matched micro-probe suites and exact definitions for Channels 1–4;
   - Benchmark 1 checkpoint (`Pythia-160M` and `Pythia-1B`) end-to-end on 1× RTX PRO 6000 to verify wall-time (`< 30 s/ckpt`) and disk footprint before launching the full Stage 1 sweep.
