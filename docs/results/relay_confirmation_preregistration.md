# Relay vs. Textbook Induction Circuits: Pre-Registered Confirmation Study (88-Core CPU Matrix)

Written 24 November 2026, before any confirmatory seeds (`1000..1023`, reserve `1024..1047`) are trained to completion.

> **Status: Pre-registered protocol (code and rules frozen prior to confirmatory execution).**
> Upgraded from the researcher-reviewed draft `notes/relay_confirmation_plan_draft_2026-11-22.md` (commit `d646b00`)
> together with the independent audit and blind predictions recorded on 22 November 2026.
> `README.md` and `docs/ROADMAP.md` remain untouched until confirmatory runs finish, independent audit is completed,
> and the researcher signs off on the final results document.

Frozen implementation committed alongside this document:
* Runner & instrument checks: [`scripts/analysis_v2/relay_confirm.py`](../../scripts/analysis_v2/relay_confirm.py)
* Frozen classifier & statistical analyzer: [`scripts/analysis_v2/relay_confirm_analyze.py`](../../scripts/analysis_v2/relay_confirm_analyze.py)
* 88-core Linux launcher: [`ops/linux/run_relay_confirm.sh`](../../ops/linux/run_relay_confirm.sh)

Background:
* `notes/relay_confirmation_plan_draft_2026-11-22.md`: ten-round review and draft matrix;
* `notes/discussion_2026-11-16_outside_reviews.md` §16–§20: exploratory synthesis;
* `docs/results/v2_analysis/relay_*_pilot.json`: exploratory pilot caches (seeds `300..302`, `300..305` for 8 heads).

---

## 0. What is known before this pre-registration (not blind) and refinements over the draft

### 0.1 Exploratory basis (not blind)

All Suite A claims (**C1–C4**) and cell coordinates were formulated after ten rounds of local exploratory experiments
on seeds `300..302` (and `300..305` for 8 heads), totaling 444 full-step runs across `relay_period_pilot.json` (288),
`relay_noise_pilot.json` (156), and `relay_window_pilot.json` (`T = 235`, `R = 163`, `U = 15`, `S = 13`, `M = 18`).
Accordingly:
* Suite A cells with exploratory counts listed in §3 are **confirmatory replications on fresh seeds (`1000..1023`)**, not blind predictions.
* Cells marked **New (blind)** in Suite A (`A1-4`, `A1-5`, `A2-6`, `A3-4`, `A3-7`, `A6-6`, `A6-7`, `A6-8`), all 18 cells of **Suite B**, and 11 of 13 cells of **Suite C** (`d = 32` low/high, `d = 128` low/high/levers, all `d = 256` cells) have **zero prior training runs** and are genuinely blind.
* No confirmatory seed (`≥ 1000`) has been trained past a 150-step code-path smoke test (loss still `> 4.1`, all `U`), and all temporary smoke caches were deleted before commit.

### 0.2 Three concrete refinements over `notes/relay_confirmation_plan_draft_2026-11-22.md`

The 76-cell matrix, seed design, three-tier cutoffs (`16/8` & `diff ≥ 8`), and GO / NO-GO / GRAY gates are **unchanged**
from the 22 November draft. Three precision refinements are frozen here:

1. **Primary source definition (`T + S`) with mandatory dual reporting (`T + S + M` and `T`).**
   An independent audit of all 18 exploratory `M` runs (`loss_rep < 2.0`, `relay_delta = 0`, not `T`, `source_l1 ≤ 0.5`)
   showed that `9/18` carry an explicit L1 relay head (`δ ∈ [10, 25]`), `13/18` have `source_l1 < 0.40` (down to `0.09–0.19`),
   `15/18` occur in 4-head models (including `2/3` of 4-head runs in `d=128`), and in `rep_min = 12` unmasked runs
   `10/10` `S` runs crossed `0.85` at L1 first while `2/2` `M` runs crossed at L0 first.
   Therefore, the frozen analyzer uses **`source_strict = T + S`** as its default primary source count, while
   **always** computing and reporting **`source_inclusive = T + S + M`** (the draft definition) and **`textbook = T`**
   side-by-side on every cell and every comparison. Any comparison whose verdict changes between `T + S` and `T + S + M`
   is flagged automatically (`definition_sensitive = true`) and reported under both definitions.
2. **Strict pooled-48 rule (§2.5) and main-seed isolation.**
   When a primary comparison fails to reach `flip` at `n = 24` and triggers §2.5 top-up (`1024..1047`), its final verdict
   uses **strictly** the pooled `n = 48` sample (never `max(n=24, n=48)`), keeping the two-stage null false-positive rate
   for `shift/flip` at `≤ 0.62%` (vs. `1.47%` single-stage). Secondary single-cell checks (`S1a–S1d`, `S2a`, `S3a–S3c`, `S4a`),
   `P4c`, descriptive `R_1` AUC, and Suite C's reuse of `d = 64` cells strictly evaluate on main seeds `1000..1023`, so
   topping up a shared anchor cell (`A1-6` or `A1-9`) never double-weights it in pooled or secondary metrics.
3. **Side-by-side blind predictions (Claude vs. Antigravity)** recorded prior to execution in §4.3 below, including an
   explicit opposite prediction on `A6-7` (`proj0` learning rate `× 1/16` without `out_scale`).

---

## 1. General Setup and Measurements

### 1.1 Model, task, and training loop

* **Model**: 2-layer pre-LayerNorm `TinyGPT` (`tycho/models/transformer.py`), vocabulary `64`, sequence length `64`,
  learned positional embeddings, attention-only (`attn_only = True`) except Suite B2 (`attn_only = False`, `mlp_ratio = 4`, GELU).
  Default width `d_model = 64` (varied in Suite C: `32, 64, 128, 256`).
* **Task**: `induction` (`tycho/tasks/induction.py`), `layout = "single"`, `p_repeat = 1.0`, repeat length uniform in
  `[rep_min, 32]`, probe set `128` fixed sequences (`data_seed = seed`). Default `rep_min = 12`.
* **Optimizer & schedule**: `LeanOptimizer` with Adam (`β₁ = 0.9`, `β₂ = 0.999`, `weight_decay = 0.0`, constant LR,
  `warmup_steps = 0`) for `12,000` steps (`24,000` steps where specified in `A2-13`, `A2-14`, `A6-6..A6-8`).
  Suite B1 varies `optimizer = "adamw"` (`weight_decay = 0.1`) and/or `warmup_steps = 200` linear warmup via `tycho.train.schedule.lr_at`.
* **Baseline & noise scale**: Baseline is `lr = 1e-3`, `batch_size = 32`. Noise scale is `σ = lr / √batch_size`.
* **Training loop**: `scripts/analysis_v2/relay_confirm.py::train_confirm`, single-threaded per worker (`torch.set_num_threads(1)`,
  `OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=1`), bit-identical to `relay_band_pilot.train` when new options are off (IC-4).
* **Measurement grid**: Every `50` steps from `0` to `2,500`, then every `250` steps to the final step. At every measurement point:
  * `band_l0`, `band_l1`: mean attention across heads/predictable tokens on the positional band `[rep_min − 1, rep_max − 1]`;
  * `source_l1`, `offsets_1_2_l0`, `loss_rep` (probe set);
  * `rms_emb`, `rms_y0` (including active `out_scale` factor `c`), `rms_y1`, and **`R_1 = rms_y0 / rms_emb`** on predictable probe tokens;
  * `rms_m0`, `rms_m1`: MLP output RMS on predictable probe tokens (`0.0` when `attn_only = True`; recorded separately in B2).

### 1.2 Seeds and per-cell sample size

* **Standard cell (`n = 24`)**: seeds `1000..1011` with `n_head = 2`, seeds `1012..1023` with `n_head = 4`.
  Every standard cell shares the exact same 24 `(seed, n_head)` pairs, enabling exact paired comparisons between any two cells.
  Each seed uniquely determines weight initialization, training data stream, and probe set.
* **C4 head-count ladder (`A6-1..A6-5`)**: all 24 seeds `1000..1023` with fixed head count (`2`, `4`, or `8`).
* **§2.5 top-up reserve (`n = 24`)**: seeds `1024..1035` (`n_head = 2`) and `1036..1047` (`n_head = 4`). Excluded for `A6-1..A6-5`.
* No seed `≥ 1000` has ever been used in any prior Tycho survey or pilot (prior max seed `901`).

### 1.3 Five-way outcome classification

Evaluated on final saved weights (`param/*`) using `scripts/analysis_v2/relay_confirm_analyze.py::classify_npz` on the
common `2,000`-sequence single-layout evaluation set (`seed = 777`, identical to `relay_followup_pilot.sample`):

| Category | Definition |
|---|---|
| **`U` (Unformed)** | `loss_rep ≥ 2.0` (or non-finite) on the common evaluation set |
| **`R` (Relay)** | Formed (`loss_rep < 2.0`), and the sharpest L1 head's modal top offset `relay_delta ≠ 0` |
| **`T` (Textbook)** | Formed, `relay_delta = 0`, and `deep_decoy_over_clean ≥ 0.9` |
| **`S` (Single-token key)** | Formed, `relay_delta = 0`, not `T`, and `source_l1_end > 0.5` |
| **`M` (Other / Mixed)** | Formed, `relay_delta = 0`, neither `T` nor `S` |

* **Primary source count (`source_strict`)**: `T + S`.
* **Sensitivity counts (always co-reported)**: `source_inclusive = T + S + M` (draft definition) and `textbook = T` (primary for C4).
* **Unstable cell rule**: A cell with `U > 6` at `n = 24` (`U > 12` at `n = 48`) is flagged `UNSTABLE` and excluded from
  three-tier verdicts (reported descriptively).

### 1.4 Three-tier decision rule, §2.5 top-up, and reporting

For each paired comparison `X` (predicted source) vs. `Y` (predicted relay) on `n` paired seeds, let `s(X)` and `s(Y)`
be the source counts:

| Tier | Condition at `n = 24` | Condition at pooled `n = 48` (§2.5) | Interpretation |
|---|---|---|---|
| **`flip`** | `s(X) ≥ 16` **and** `s(Y) ≤ 8` | `s(X) ≥ 32` **and** `s(Y) ≤ 16` | Intervention flips majority from one basin to the other |
| **`shift`** | Not `flip`, and `s(X) − s(Y) ≥ 8` | Not `flip`, and `s(X) − s(Y) ≥ 16` | Expected direction, smaller than a full flip |
| **`unconfirmed`** | `s(X) − s(Y) < 8` | `s(X) − s(Y) < 16` | Effect not confirmed |

* **Claim-level rollup**: Each claim C1–C3 takes the **lowest tier** across its primary comparisons.
* **C4 comparisons (`P4a`, `P4b`)**: Use `textbook = T` counts with a single threshold `T(X) − T(Y) ≥ 8` (`n = 24`, no top-up).
* **§2.5 Automatic Top-up**: If any primary comparison in `P1a..P3d` does not reach `flip` (and is not `unstable`) at `n = 24`,
  run reserve seeds `1024..1047` on both cells of that pair and re-evaluate **strictly** on the pooled `n = 48` sample. Done at most once.
* **Statistical reporting**:
  * Every cell: 5-way counts split by `h2`/`h4`, source rate with Wilson 95% CI, `|h2 − h4| ≥ 6` flag, `U > 6` unstable flag.
  * Every paired comparison: counts, difference, exact two-sided McNemar p-value under `T+S`, `T+S+M`, and `T`.
  * `S1b`: exact `3 × 2` Freeman-Halton test p-value.
  * All medians (`R_1`, step ratios): 10,000-resample bootstrap 95% CI with fixed seed `20261124`.
* **Cell roles**: 21 `primary`, 10 `secondary`, 10 `descriptive`, 22 `exploratory` (4 in Suite A + 18 in Suite B), 13 `gate` (Suite C) = **76 cells**.

---

## 2. Suite A — Frozen Claims C1–C4 (45 cells, 1,080 runs)

In the tables below, **Expl.** lists exploratory `T+S+M (T)` out of 6 runs (`New` = zero prior runs, blind).

### C1 — Noise scale (`σ = lr / √B`) and `rep_min` threshold ordering

**Claim.** At `rep_min = 12, d_model = 64`, source vs. relay routing is ordered by `σ = lr / √B`, not by `lr` alone or
`lr / B`: cells at `σ = 0.88e-4` route to source and cells at `σ ≥ 1.77e-4` route to relay across `lr ∈ [5e-4, 2e-3]`.
The threshold rises with `rep_min` from `12` upward: below `1.77e-4` at `rep_min = 12`, between `1.77e-4` and `3.54e-4` at
`rep_min = 16`, and above `3.54e-4` at `rep_min = 20`.

| Cell | Setting (`lr / B`, extras) | `σ` (`1e-4`) | Expl. | Prediction | Role |
|---|---|---:|---|---|---|
| `A1-1` | `5e-4 / 32` | `0.88` | `6 (6)` | Source | primary |
| `A1-2` | `1e-3 / 128` | `0.88` | `6 (6)` | Source | primary |
| `A1-3` | `1e-3 / 64` | `1.25` | `6 (5)` | Equal rates across `A1-3..A1-5`, majority source | secondary |
| `A1-4` | `5e-4 / 16` | `1.25` | **New** | Same as `A1-3` | secondary |
| `A1-5` | `2e-3 / 256` | `1.25` | **New** | Same as `A1-3` | secondary |
| `A1-6` | `1e-3 / 32` (baseline) | `1.77` | `0 (0)` | Relay | primary |
| `A1-7` | `2e-3 / 128` | `1.77` | `0 (0)` | Relay | primary |
| `A1-8` | `5e-4 / 8` | `1.77` | `1 (1)` | Relay (`≤ 8/24`) | secondary |
| `A1-9` | `2e-3 / 32` | `3.54` | `0 (0)` | Relay (`≤ 8/24`) | primary |
| `A2-1` | `rep_min=16, 1e-3 / 32` | `1.77` | `6 (5)` | Source | primary |
| `A2-2` | `rep_min=16, 2e-3 / 32` | `3.54` | `0 (0)` | Relay | primary |
| `A2-3` | `rep_min=20, 1e-3 / 32` | `1.77` | `6 (6)` | Source | descriptive |
| `A2-4` | `rep_min=20, 2e-3 / 32` | `3.54` | `5 (5)` | Source | primary |
| `A2-5` | `rep_min=24, 1e-3 / 32` | `1.77` | `6 (6)` | Source | descriptive |
| `A2-6` | `rep_min=24, 2e-3 / 32` | `3.54` | **New** | Source (`≥ 16/24`) | secondary |
| `A2-7` | `rep_min=4, 1e-3 / 64` | `1.25` | `6 (4)` | — | descriptive |
| `A2-8` | `rep_min=4, 1e-3 / 32` | `1.77` | `4 (2)` | More source than `A1-6` (`diff ≥ 6`) | secondary |
| `A2-9` | `rep_min=4, 2e-3 / 64` | `2.50` | `0 (0)` | — | descriptive |
| `A2-10` | `rep_min=8, 1e-3 / 64` | `1.25` | `5 (4)` | — | descriptive |
| `A2-11` | `rep_min=8, 1e-3 / 32` | `1.77` | `1 (0)` | — | descriptive |
| `A2-12` | `rep_min=20, 2e-3 / 16` | `5.00` | `1 (1), 3 U` | Formation wall (report `U` & early `rms_y1`) | descriptive |
| `A2-13` | `rep_min=24, 1.414e-3 / 8, 24k` | `5.00` | `4 (4), 2 U` | Formation wall | descriptive |
| `A2-14` | `rep_min=24, 2e-3 / 8, 24k` | `7.07` | `0 (0), 4 U` | Formation wall | descriptive |

* **Primary comparisons**:
  * `P1a`: `A1-2` vs `A1-6` (same `lr = 1e-3`, `B` differs `4×` — rules out `lr`-only ordering).
  * `P1b`: `A1-1` vs `A1-7` (same `lr / B = 1.56e-5` — rules out `lr / B` ordering).
  * `P1c`: `A2-1` vs `A1-6` (same `1e-3 / 32`, `rep_min 16` vs `12`).
  * `P1d`: `A2-4` vs `A2-2` (same `2e-3 / 32`, `rep_min 20` vs `16`).
* **Secondary predictions**:
  * `S1a`: `A1-8 ≤ 8/24` and `A1-9 ≤ 8/24`.
  * `S1b`: `A1-3`, `A1-4`, `A1-5` (all `σ = 1.25e-4`): exact `3 × 2` Freeman-Halton `p ≥ 0.05` and each `> 12/24`.
    If `p < 0.05`, add caveat that `σ` orders only up to `√2` and `lr`/`B` retain separate effects right on the threshold.
  * `S1c`: `A2-6 ≥ 16/24`.
  * `S1d`: `A2-8 − A1-6 ≥ 6`.
* **Wording downgrades**: If `P1a` or `P1b` is `shift`, weaken sentence 1 to "roughly ordered by `σ`, weaker than in exploration";
  if `unconfirmed`, retract sentence 1. If `P1c` or `P1d` is below `flip`, downgrade the corresponding half of sentence 2.

---

### C2 — Relative scale of L0 attention output is a bidirectional routing lever

**Claim.** Multiplying L0 attention output by a fixed scalar `c` flips routing in both directions, with `c` trading off
against `σ` at least linearly (not as `√c`). Changing the learning rate of L1's OV weights also flips routing in both
directions, while early measured `R_1` moves in the corresponding direction.

| Cell | Setting | `σ·c` (`1e-4`) | Expl. | Prediction | Role |
|---|---|---:|---|---|---|
| `A3-1` | Baseline, `c = 0.5` | `0.88` | `6 (6)` | Source | primary |
| `A3-2` | `5e-4 / 32, c = 2.0` | `1.77` | `0 (0)` | Relay | primary |
| `A3-3` | `2e-3 / 32, c = 0.25` | `0.88` | `6 (4)` | Source | primary |
| `A3-4` | `2e-3 / 32, c = 0.5` | `1.77` | **New** | Linear (`c¹`): relay (`≤ 8/24`); super-linear: higher | secondary |
| `A3-5` | Baseline, `ov1` LR `× 2` | — | `6 (6)` | Source; `R_1(300)` lower than `A1-6` | primary |
| `A3-6` | `5e-4 / 32, ov1` LR `× 0.25` | — | `0 (0)` | Relay; `R_1(600)` higher than `A1-1` | primary |
| `A3-7` | Baseline, `ov0` LR `× 0.5` | — | **New** | Exploratory (see §4.3) | exploratory |

* **Primary comparisons** (`R_1` aligned step `= round(0.3 / lr)`: step `300` at `1e-3`, `600` at `5e-4`, `150` at `2e-3`):
  * `P2a`: `A3-1` vs `A1-6`.
  * `P2b`: `A1-1` vs `A3-2` (under `√c`, `A3-2` would be `1.25e-4` -> majority source).
  * `P2c`: `A3-3` vs `A1-9` (under `√c`, `A3-3` would be `1.77e-4` -> relay).
  * `P2d`: `A3-5` vs `A1-6` **and** `R_1(300, A3-5) < R_1(300, A1-6)` in `≥ 20/24` pairs (`≥ 40/48` if topped up).
  * `P2e`: `A1-1` vs `A3-6` **and** `R_1(600, A3-6) > R_1(600, A1-1)` in `≥ 20/24` pairs (`≥ 40/48` if topped up).
* **Secondary & descriptive**:
  * `S2a` (`A3-4`, blind): `≤ 8/24` supports linear (`c¹`); `≥ 16/24` indicates super-linear leverage (still compatible with "at least linearly", precludes "approximately linear"); `9..15/24` leaves exponent indeterminate.
  * Descriptive: full `R_1(t)` curves and pooled ROC-AUC across `A1 + A3` (reported only; no pass/fail cutoff).
* **Explicitly excluded**: Any claim that `R_1` is a causal mediator or obeys a universal cross-LR threshold.

---

### C3 — Relative QK speed across layers and early critical window

**Claim.** At baseline, slowing L0 QK or speeding L1 QK flips relay to source; at `5e-4 / 32`, doubling L0 QK speed flips
source to relay. Blocking L0 far attention (`offset ≥ 4`) during steps `0–500` flips baseline to source, whereas the same
mask during steps `1,000–3,000` does not. The QK lever has a bounded range: at `σ = 3.54e-4`, slowing L0 QK to `0.25×`
remains relay.

| Cell | Setting | Expl. | Prediction | Role |
|---|---|---:|---|---|
| `A4-1` | Baseline, `qk0` LR `× 0.5` | `6 (6)` | Source | primary |
| `A4-2` | Baseline, `qk1` LR `× 2.0` | `6 (6)` | Source | primary |
| `A4-3` | `5e-4 / 32, qk0` LR `× 2.0` | `0 (0)` | Relay | primary |
| `A4-4` | `2e-3 / 32, qk0` LR `× 0.25` | `1 (1)` | Relay (`≤ 8/24`, boundary) | secondary |
| `A5-1` | Baseline, L0 forbid `far` (`offset ≥ 4`), steps `0–500` | `6 (6)` | Source | primary |
| `A5-2` | Baseline, L0 forbid `far` (`offset ≥ 4`), steps `1,000–3,000` | `0 (0)` | Relay | primary |
| `A5-3` | `rep_min=20, 1e-3 / 32`, L1 forbid `band`, steps `0–500` | `0 (0)` | Relay (`≤ 8/24`, mirror of `A2-3 ≥ 16/24`) | secondary |

* **Primary comparisons**: `P3a` (`A4-1` vs `A1-6`), `P3b` (`A4-2` vs `A1-6`), `P3c` (`A1-1` vs `A4-3`), `P3d` (`A5-1` vs `A5-2`).
* **Secondary predictions**:
  * `S3a`: `A4-4 ≤ 8/24` (if majority source, retract final sentence and note discrepancy with exploration).
  * `S3b`: `A5-3 ≤ 8/24` and `A2-3 ≥ 16/24`.
  * `S3c` (out-of-sample test of `0.85` band-crossing order across all `480` main runs in `A1 + A3 + A4`):
    decidable rate (non-tied crossing) `≥ 90%`, and agreement between `"L1 crosses 0.85 before L0"` and `"routes to source"` `≥ 85%`.

---

### C4 — Two-stage textbook assembly (L1 source lock-in before second key token)

**Claim.** Textbook induction assembles in two stages: L1 first locks onto the source token (single-token key), and the
second key token arrives later. At `2e-3 / 32, c = 1/16`, 2-head models mostly stall at stage 1 within `12,000` steps while
8-head models mostly complete stage 2. Folding `c` into L0's output projection at step `1,000` (function unchanged at the
switch, normal projection update rate thereafter) lets 2-head models mostly finish stage 2 as well.

| Cell | Setting (`lr / B`, heads, seeds) | Expl. (`T`) | Role |
|---|---|---:|---|
| `A6-1` | `2e-3 / 32, c = 1/16`, `2h × 24` (`s1000..1023`) | `0/3` | primary |
| `A6-2` | `2e-3 / 32, c = 1/16`, `4h × 24` (`s1000..1023`) | `2/3` | descriptive |
| `A6-3` | `2e-3 / 32, c = 1/16`, `8h × 24` (`s1000..1023`) | `6/6` | primary |
| `A6-4` | `2e-3 / 32, c = 1/16`, `2h × 24`, fold at step `1,000` | `2/3` (`6/8` pooled) | primary |
| `A6-5` | `2e-3 / 32, c = 1/16`, `2h × 24`, drop at step `1,000` | `3/3` (`8/8` pooled) | secondary |
| `A6-6` | `5e-4 / 32, 24k` steps, `c = 1/16` (`2h×12 + 4h×12`) | **New** | exploratory (decomp ref) |
| `A6-7` | `5e-4 / 32, 24k` steps, `proj0` LR `× 1/16` (`2h×12 + 4h×12`) | **New** | exploratory (decomp LR-only) |
| `A6-8` | `5e-4 / 32, 24k` steps, `proj0` init `× 1/16` (`2h×12 + 4h×12`) | **New** | exploratory (decomp init-only) |

* **Primary comparisons** (`textbook = T` counts):
  * `P4a`: `T(A6-3) − T(A6-1) ≥ 8`.
  * `P4b`: `T(A6-4) − T(A6-1) ≥ 8` (exact paired comparison; trajectories are bit-identical up to step `1,000`).
  * `P4c`: Across all main-seed `T` runs in `A1 + A3 + A4`, the first step where `loss_rep` stays permanently `< 0.4`
    occurs strictly after the first step where `source_l1 > 0.5` in `≥ 90%` of runs, with median ratio in `[2.0, 4.5]`.
* **Secondary**: `S4a`: `A6-5` has `T ≥ 16/24` and `R ≤ 2/24`.
* **Excluded from frozen claims**: All closed-form pre-training formulas (`K` formulas, critical-window formula, two-clock
  analytic forms, composite score `S`), "locking" terminology, ignition-time equations, analytic two-branch `σ_crit(rep_min)`
  forms, and causal-mediator / universal-threshold interpretations of `R_1`.

---

## 3. Suite B — Generality Probes (18 exploratory cells, 432 runs)

Zero prior exploratory data exist for Suite B; all readings below are pre-committed interpretations, not confirmations.
Every `noise3` group uses the three fixed cells `low = 1e-3 / 128` (`σ = 0.88e-4`), `mid = 1e-3 / 32` (`1.77e-4`),
`high = 2e-3 / 32` (`3.54e-4`) at `rep_min = 12, d_model = 64, n = 24`.

| Group | Modification | Cells |
|---|---|---|
| `B1-W` | `AdamW`, `weight_decay = 0.1`, no warmup | `low`, `mid`, `high` (3) |
| `B1-U` | `Adam`, `200`-step linear warmup | `low`, `mid`, `high` (3) |
| `B1-WU` | `AdamW` (`wd = 0.1`) + `200`-step linear warmup | `low`, `mid`, `high`, `mid-c05`, `low-c2` (5) |
| `B2` | Add 4× GELU MLP in both blocks (`attn_only = False`), `Adam` | `low`, `mid`, `high`, `mid-c05`, `low-c2` (5) |
| `B3` | Baseline `1e-3 / 32`, sampling `rep_min = 20` inside window (`rep_min = 12` outside) | `B3-early` `[0, 1000)`, `B3-late` `[1000, 3000)` (2) |

### 3.1 Pre-committed readings

* **Noise-scale transfer** (`B1-W`, `B1-U`, `B1-WU`, `B2`): Compare `low` vs `high` via §1.4 three-tier rule
  (`flip` = transfers cleanly; `shift` = attenuated; `unconfirmed` = absent in this range).
* **Middle cell threshold reading**: `mid ≤ 8/24` -> threshold unchanged; `mid ≥ 16/24` -> threshold shifted up by `≥ √2×`;
  `9..15/24` -> right on threshold.
* **Lever transfer** (`B1-WU`, `B2`): `mid-c05 − mid ≥ 8` and `low − low-c2 ≥ 8`. If the reference cell is already on
  the lever's target side (`mid ≥ 16/24` for `c = 0.5`, or `low ≤ 8/24` for `c = 2`), that half is marked `na_ref_at_target`
  rather than failed.
* **`B3` data-side window**: `B3-early ≥ 16/24` and `B3-late ≤ 8/24` -> data-side early window confirmed; both on same side -> absent.

### 3.2 Side-by-side blind predictions (recorded 22 November 2026, before any runs)

| Cell / Group | Claude's blind guess (draft `d646b00`) | Antigravity's blind prediction (audit Nov 22) |
|---|---|---|
| **`A3-4`** (`2e-3/32, c=0.5`) | Linear `c¹`: relay (`≤ 8/24`) | Intermediate leaning relay (`~6–14/24`, center `9–11/24`; effective exponent `~1.15–1.3`) |
| **`A3-7`** (base, `ov0` LR `×0.5`) | Over half source, lower `R_1` (low confidence) | Majority source (`16–22/24`), significantly lower `R_1` |
| **`A6-6`** (`5e-4/32, c=1/16, 24k`) | `2h` mostly stalls at stage 1; `4h` mostly finishes | Same: `2h` `T = 0–3/12`, `4h` `T = 8–11/12` |
| **`A6-7`** (`5e-4/32, proj0` LR `×1/16, 24k`) | Similar to `A6-6` (`2h` mostly stalls) | **Opposite**: `2h` mostly finishes (`T = 8–12/12`), `4h` `10–12/12`, close to `A1-1` (key discriminator!) |
| **`A6-8`** (`5e-4/32, proj0` init `×1/16, 24k`) | Close to `A1-1` | Agree: indistinguishable from `A1-1` (`T = 22–24/24`) |
| **`B1-W`** (`AdamW, wd=0.1`) | Same as Adam | Agree: `flip` preserved, `mid ≤ 8/24` (center `0–4/24`) |
| **`B1-U` / `B1-WU`** (`200`-step warmup) | `flip` preserved; `mid` shifts toward source | `flip` preserved; `mid` shifts strongly to `~14–22/24` (likely `≥ 16/24`); `c = 2` flips cleanly, `c = 0.5` may hit `na_ref_at_target` |
| **`B2`** (`+4×` GELU MLP) | `flip` preserved; levers transfer | `flip` preserved; `mid` slightly higher (`~6–15/24`); levers transfer |
| **`B3`** (data window `rep_min=20`) | Early source, late relay | Strong clean split: `early ≥ 20/24` (`22–24`), `late ≤ 4/24` (`0–2`) |

---

## 4. Suite C — Width Bridge & GPU Gate (13 cells, 312 runs)

| `d_model` | Cells (`n = 24`: `12 × 2h + 12 × 4h`) | Notes |
|---:|---|---|
| `32` | `low`, `mid`, `high` (3) | Expl. baseline `0/6` source |
| `64` | Reuses Suite A: `A1-2`, `A1-6`, `A1-9`, `A3-1`, `A3-4` (0 new) | Main seeds `1000..1023` only |
| `128` | `low`, `mid`, `high`, `mid-c05`, `high-c05` (5) | Expl. baseline: `2h` `3/3 T`, `4h` `2 M + 1 R` |
| `256` | `low`, `mid`, `high`, `mid-c05`, `high-c05` (5) | **Decision gate below** |

### Pre-registered GPU Gate (evaluated strictly on `d_model = 256` stable cells `U ≤ 6`)

* **`GO`**: At least one stable base cell (`low/mid/high`) has `R ≥ 12/24`, **and** on a stable pair (`mid` vs `mid-c05`
  or `high` vs `high-c05`), `c = 0.5` reduces `R` by `≥ 8`.
* **`NO-GO`**: Every stable base cell has `R ≤ 4/24` **and** `M ≤ 4/24`. Conclusion: positional relay is a narrow-model
  phenomenon; close C1–C4 as a small-model result and do not carry this task to GPU.
* **`GRAY`** (all remaining outcomes): Hold GPU scaling; if `M` dominates, develop a continuous non-modal metric first;
  if `R ∈ [5, 11]`, add CPU noise cells first. Final decision rests with the researcher.

---

## 5. Instrument Checks, Frozen Analysis, Data Release, and Amendments

### 5.1 Mandatory instrument checks (run before Suite A; abort immediately on failure)

* **IC-1 (Same-machine repeatability)**: Replay two runs (`A1-6 h2 s300`, `A1-1 h4 s300`, 1,500 steps) twice on the
  execution machine -> `max_weight_diff == 0.0` and `max_stat_diff == 0.0`.
* **IC-2 (Cross-machine agreement)**: Re-run 18 exploratory runs (`seeds 300..302, h2 & h4` on `A1-6`, `A1-1`, `A3-5`, full 12k steps;
  exploratory counts `0/6`, `6/6`, `6/6`) -> `≤ 1` total flip under both `T+S` and `T+S+M`.
* **IC-3 (Frozen classifier calibration)**: Run `relay_confirm_analyze.py --ic3` against the exploratory pilot JSONs -> `0` mismatches across all 25 checked §3 exploratory cells.
* **IC-4 (No-op bit-identity)**: With all new options off, `train_confirm` reproduces `relay_band_pilot.train` at step `600` (`A1-6, h2, s300`) with `max_weight_diff == 0.0` and `max_stat_diff == 0.0`.

*Pre-commit local verification (macOS, Python 3.9, single-thread CPU):*
`IC-3` **PASSED (`25/25`, 0 mismatches)**, `IC-4` **PASSED (`0.0` weight diff, `0.0` stat diff)**, `IC-1` **PASSED (`2/2` pairs bit-identical)**, and 14-path `--smoke` + analyzer end-to-end tests **PASSED**. Full `IC-3 + IC-4 + IC-1 + IC-2` executes automatically on the 88-core Linux server at the start of `ops/linux/run_relay_confirm.sh`.

### 5.2 Data release, independent verification, and amendments

* Every run saves final weights (`param/*`), full measurement curves (`stat_*`), and exact metadata (`torch_version`,
  `platform`, `cpu_model`, `num_threads`, `wall_s`) atomically to `work/relay_confirm/<name>.npz`.
* After execution finishes, another agent session (Claude on `claude/lucid-archimedes-av5rk7`) independently inspects the
  committed runner/analyzer, re-classifies saved `.npz` weights, and verifies every table and gate verdict.
* **Amendments**: Any change after this commit must be logged in §7 below with date and commit hash **before** affected
  runs start. Any post-hoc analysis run after inspecting confirmatory data is labeled exploratory.

---

## 6. Compute Scale and Execution Order

Execution order enforced by `ops/linux/run_relay_confirm.sh`:
1. `IC-3` + `IC-4` + `IC-1` + `IC-2` (~15 min on 86 workers; hard stop on failure)
2. **Suite A** — 45 cells, `1,080` runs (~121 core-h -> `1.5–2.0 h` wall)
3. **Suite B** — 18 cells, `432` runs (~71 core-h -> `~1.2 h` wall)
4. **Suite C** — 13 cells, `312` runs (~102 core-h -> `~1.8 h` wall)
5. Frozen analysis (`relay_confirm_analyze.py`) -> writes `docs/results/v2_analysis/relay_confirm_results.json` and `relay_confirm_summary.md` (and prints exact `--topup` command if §2.5 triggers).

**Total**: **76 cells, 1,824 runs, ~294 core-hours -> ~3.4 h ideal, ~5 h wall time with 40% margin on 86 workers.**
Jobs are scheduled longest-processing-time-first (`d_model² × steps × batch_size × mlp_factor`) to avoid straggler tails,
and every `.npz` is written atomically via `os.replace` so interrupted runs resume cleanly with zero recomputation.

---

## 7. Amendments Log

*(None — initial pre-registration commit.)*
