# Relay vs. Textbook Induction Circuits — Phase 2 Pre-Registered Confirmation (MLP Residual Scaling, RoPE & 4-Layer Boundaries)

Written 26 November 2026, before any Phase 2 confirmatory seeds (`1048..1071`) are run.

> **Status: Pre-registered protocol (committed and pushed prior to running any seed in `1048..1071`).**
> Directly implements the follow-up confirmation recommended in the independent audit
> `notes/relay_confirm_audit_2026-11-25.md` (§6.1), upgrading the post-hoc exploratory findings of
> `relay_mlp_res_opt.md` and `relay_stress_{rope,depth}.md` onto fresh seeds, additional settings,
> a unified behavioral criterion (`B_T`), and a 24,000-step budget for 4-layer models.

Frozen implementation committed alongside this document:
* Runner, IC-P2 & frozen analyzer: [`scripts/analysis_v2/relay_confirm_phase2.py`](../../scripts/analysis_v2/relay_confirm_phase2.py)
* 88-core Linux launcher: [`ops/linux/run_relay_confirm_phase2.sh`](../../ops/linux/run_relay_confirm_phase2.sh)

---

## 0. What is known before Phase 2 (not blind) vs. genuinely blind

* **Known from post-hoc exploratory runs on Phase 1 seeds `1000..1023` (and Claude's 4 fresh seeds `2000..2003`) — NOT blind**:
  * **M1 (`2L abs-pos 4× MLP, AdamW wd=0.1, 200wu, d=64, lr=1e-3, 12k`)**: Unscaled `(1,1)` gave `0/24 B_T`; `Attn0×0.5` alone `7/24`; `MLP0×0.5` alone `1/24`; joint `(0.5,0.5)` `21/24`; joint `(0.25,0.25)` `24/24` (and `0/4 -> 4/4` on Claude's seeds `2000..2003`).
  * **M2a base (`2L abs-pos 4× MLP, plain Adam, no wu, d=64, lr=5e-4, 12k`)**: Unscaled base gave `0/24 B_T (24 R, 0 U)` in `relay_mlp_diag.md`; joint `(0.25,0.25)` at this setting has **not** been run before (blind arm!).
  * **R1 (`2L RoPE`)**: At `1e-3` attention-only gave `24/24 B_T`; at `2e-3` `4× MLP + AdamW+wu` gave `14/24 B_T` (`4/12 h2`, `10/12 h4`, `7 R`) unscaled vs `22/24 B_T` (`10/12 h2`, `12/12 h4`, `0 R`) with `(0.25,0.25)`.
  * **D1 (`4L`, `12k` steps)**: At `12k` steps `L4-abs` gave `10/24 B_T` (`1/12 h2`, `9/12 h4`, `9 R`, `3 U`), `L4-rope` gave `23/24 B_T` (`12/12 h2`, `11/12 h4`, `0 R`, `0 U`), and `L4-abs-l01_05` gave `13/24 B_T` (`6/12 h2`, `7/12 h4`, `1 R`, `6 U`, with audit §3.5 noting late-forming truncation near 12k steps).
* **Genuinely blind (zero prior runs at these settings)**:
  * **`M2a-lr5e4-both025`**: joint `(0.25,0.25)` under plain Adam `lr=5e-4` (no warmup).
  * **`M2b-d128-base` & `M2b-d128-both025`**: second width `d=128` with `4× MLP + AdamW+wu, lr=5e-4`.
  * **`R1-attn-2e3`**: `2L RoPE` pure attention at high `lr=2e-3` (requested by audit §3.4 & §6.1).
  * **`D1-*-24k`**: all three `4L` cells at the expanded `24,000`-step budget.
* **No confirmatory seed in `1048..1071` has ever been run** (local `--verify-only` and server `--ic` use only seed `300`).

---

## 1. Unified Outcome Criterion, Seeds, and Decision Rules

* **Fresh seeds (`n = 24` per cell)**: seeds `1048..1059` with `n_head = 2`, seeds `1060..1071` with `n_head = 4`. All 384 runs execute on the **same machine (`s82`, single-thread per worker)** to eliminate cross-machine float drift in 4-layer models (audit §3.5).
* **Primary outcome variable (`B_T`, unified behavioral textbook — audit §3.2)**:
  * **`B_T = True`** iff **`loss_rep < 0.3` AND `deep_decoy_over_clean >= 0.9`** on the fixed 2,000-sequence single-layout evaluation set (`seed = 777`).
  * Auxiliary 5-way breakdown `{U, R, T_beh, S, M}` (with `U: loss_rep >= 2.0`, `T_beh = B_T`, `R: formed & not B_T & best_upper_delta != 0`) is co-reported on every cell, split by `h2` and `h4`.
* **Three-tier paired decision rule (`n = 24`, identical to Phase 1 §1.4)**:
  * **`flip`**: `B_T(X) >= 16` AND `B_T(Y) <= 8` (for `h2`-only `n = 12`: `>= 8` AND `<= 4`)
  * **`shift`**: not `flip`, AND `B_T(X) - B_T(Y) >= 8` (for `n = 12`: `>= 4`)
  * **`unconfirmed`**: `B_T(X) - B_T(Y) < 8` (for `n = 12`: `< 4`)
  * **`unstable`**: either cell has `U > 6` (for `n = 12`: `U > 3`); reported descriptively.
* Exact two-sided McNemar p-values, Wilson 95% CIs, and 10,000-resample fixed-seed (`20261124`) bootstrap median CIs are reported on every comparison/cell. No Phase 2 top-up is planned; verdicts stand on `n = 24`.

---

## 2. Frozen 16-Cell Matrix (384 Fresh Runs)

All cells use `rep_min = 12, rep_max = 32, seq_len = 64, vocab = 64, layout = "single", batch_size = 32, mlp_ratio = 4` (when `attn_only = False`).

| Cell | Group | Role | L | Pos | MLP | `d` | `lr` | Opt / WU | Steps | `(Attn, MLP)` scale |
|---|---|---|---:|---|---|---:|---:|---|---:|---|
| `M1-base` | M1 | primary | 2 | abs | 4× | 64 | `1e-3` | AdamW 0.1 + 200wu | 12k | `(1.0, 1.0)` |
| `M1-attn05` | M1 | primary | 2 | abs | 4× | 64 | `1e-3` | AdamW 0.1 + 200wu | 12k | `L0:(0.5, 1.0)` |
| `M1-mlp05` | M1 | primary | 2 | abs | 4× | 64 | `1e-3` | AdamW 0.1 + 200wu | 12k | `L0:(1.0, 0.5)` |
| `M1-both05` | M1 | primary | 2 | abs | 4× | 64 | `1e-3` | AdamW 0.1 + 200wu | 12k | `L0:(0.5, 0.5)` |
| `M1-both025` | M1 | primary | 2 | abs | 4× | 64 | `1e-3` | AdamW 0.1 + 200wu | 12k | `L0:(0.25, 0.25)` |
| `M2a-lr5e4-base` | M2a | primary | 2 | abs | 4× | 64 | `5e-4` | Adam (no wu) | 12k | `(1.0, 1.0)` |
| `M2a-lr5e4-both025` | M2a | primary | 2 | abs | 4× | 64 | `5e-4` | Adam (no wu) | 12k | `L0:(0.25, 0.25)` |
| `M2b-d128-base` | M2b | exploratory | 2 | abs | 4× | 128 | `5e-4` | AdamW 0.1 + 200wu | 12k | `(1.0, 1.0)` |
| `M2b-d128-both025` | M2b | exploratory | 2 | abs | 4× | 128 | `5e-4` | AdamW 0.1 + 200wu | 12k | `L0:(0.25, 0.25)` |
| `R1-attn-1e3` | R1 | secondary | 2 | RoPE | no | 64 | `1e-3` | Adam (no wu) | 12k | `(1.0, —)` |
| `R1-attn-2e3` | R1 | exploratory | 2 | RoPE | no | 64 | `2e-3` | Adam (no wu) | 12k | `(1.0, —)` |
| `R1-mlp-2e3-base` | R1 | primary | 2 | RoPE | 4× | 64 | `2e-3` | AdamW 0.1 + 200wu | 12k | `(1.0, 1.0)` |
| `R1-mlp-2e3-both025` | R1 | primary | 2 | RoPE | 4× | 64 | `2e-3` | AdamW 0.1 + 200wu | 12k | `L0:(0.25, 0.25)` |
| `D1-abs-24k` | D1 | primary | 4 | abs | 4× | 64 | `1e-3` | AdamW 0.1 + 200wu | 24k | `(1.0, 1.0)` |
| `D1-rope-24k` | D1 | primary | 4 | RoPE | 4× | 64 | `1e-3` | AdamW 0.1 + 200wu | 24k | `(1.0, 1.0)` |
| `D1-abs-l01_05-24k` | D1 | exploratory | 4 | abs | 4× | 64 | `1e-3` | AdamW 0.1 + 200wu | 24k | `L0,L1:(0.5, 0.5)` |

---

## 3. Pre-Registered Claims, Comparisons, and Blind Predictions

1. **Claim P2-M1 (2L MLP joint Layer-0 scaling requires damping both branches)**:
   * `PM1a`: `M1-both05` vs `M1-base` -> predicted **`flip`**.
   * `PM1b`: `M1-both025` vs `M1-base` -> predicted **`flip`**.
   * `PM1c`: `M1-both05` vs `M1-attn05` -> predicted **`shift` or `flip`** (`diff >= 8`).
   * `PM1d`: `M1-both05` vs `M1-mlp05` -> predicted **`flip`**.
   * Secondary `SM1`: `M1-attn05 <= 12/24` and `M1-mlp05 <= 8/24` (neither branch alone flips).
   * *Note*: No "super-additivity" claim is made (audit §3.3); claim is strictly that neither branch alone at `0.5` suffices whereas joint damping flips the outcome.
2. **Claim P2-M2 (Generalization across optimizer/LR and width)**:
   * `PM2a`: `M2a-lr5e4-both025` vs `M2a-lr5e4-base` (plain Adam `5e-4`, no warmup) -> predicted **`flip`**.
   * `EM2b` (**blind exploratory**, `d=128`): blind prediction — `M2b-d128-base` has `B_T <= 8/24` (heavy relay trap) and `M2b-d128-both025` increases `B_T` by `>= 8` (reaching `>= 16/24`).
3. **Claim P2-R1 (2L RoPE boundary)**:
   * Secondary `SR1a`: `R1-attn-1e3` has `B_T >= 16/24` and `R <= 2/24`.
   * Exploratory `ER1b` (**blind**): `R1-attn-2e3` tests whether pure-attention RoPE remains relay-free at `2e-3` (blind guess: `B_T >= 16/24`, `R <= 4/24`, possibly some `U`).
   * `PR1`: `R1-mlp-2e3-both025` vs `R1-mlp-2e3-base` -> on `h2` (`n = 12`), predicted **`flip` or `shift`** (`B_T(both025) >= 8/12` and `diff >= 4/12`, with `R` reduced to `<= 2/12`); overall `n = 24`, `B_T(both025) >= 16/24` and `R <= 2/24`.
4. **Claim P2-D1 (4L RoPE vs 4L learned absolute `pos` at `24,000` steps)**:
   * `PD1`: `D1-rope-24k` vs `D1-abs-24k` -> predicted `D1-rope-24k` reaches `B_T >= 18/24` (`h2 >= 10/12`), with `h2` **`flip`** (`>= 8/12` vs `<= 4/12`) and overall `diff >= 8` (note: if `D1-abs-24k`'s `h4` stays near `10–11/12`, overall `n = 24` will be `shift` rather than `flip`, driven by `h2`).
   * Exploratory `ED1` (`D1-abs-l01_05-24k` vs `D1-abs-24k`): tests whether doubling training steps from `12k` to `24k` eliminates the `U` increase seen at `12k` (blind guess: `U` drops from `6` to `<= 2`, `h2` rises to `8–10/12`, `h4` recovers to `8–11/12`, testing whether shallow scaling was merely step-truncated at `12k`).

---

## 4. Mandatory Instrument Check (`IC-P2`) and Execution Order

Enforced by `ops/linux/run_relay_confirm_phase2.sh`:
1. **`IC-P2` (hard stop on failure)**: RoPE norm & relative-invariance check (`< 1e-5`), pre- vs. post-fold logit identity (`< 1e-5`) and reload identity (`0.0`) on 2L abs MLP, 2L RoPE MLP, and 4L RoPE MLP (seed `300`, 4 steps), plus same-machine repeatability on `s82` (`M1-base` 2L MLP and `D1-rope` 4L RoPE, seed `300`, 600 steps replayed twice -> `0.0` weight and curve diff).
2. **`--run`**: All 384 runs in `work/relay_confirm_phase2/`, LPT-sorted (4L `24k` and `d=128` launch first), ~2 hours wall time on 86 single-thread workers.
3. **`--analyze`**: Frozen analysis writing `docs/results/v2_analysis/relay_confirm_phase2_{results.json,summary.md}`.

---

## 5. Amendments Log

*(None — initial pre-registration commit.)*
