# Independent review of the Part B pre-registration draft (16 November 2026)

An independent agent reviewed the first draft of [v2b_preregistration.md](../v2b_preregistration.md) (commits
61754e2 and 045380d). It edited no repository files.

**Pilots it ran** (seeds outside the confirmatory ranges; lr 3e-4, tiled, 3 heads per layer, 60 pattern
checkpoints, weights saved at every checkpoint):

| configuration | seeds |
|---|---|
| d96 main | 330 |
| d48 main | 330, 331 |
| d96 no-repeat 2-layer | 330–332 |
| d48 no-repeat 2-layer | 330 |
| d96 1-layer control, 3 heads | 330 |
| d96 1-layer control, 6 heads | 330 |

It also reanalysed the local 3-head runs and DR2b-pilot2.

## Findings

1. **R2 cannot discriminate as drafted.**
   * A threshold calibrated on no-repeat controls sits at untrained-network level: about 5e-9 nats for θ at
     d_model 96.
   * Every link and reference crossed it at the same checkpoint, between steps 0 and 11, in every pilot run. So Δ_R2
     was 0 in all 12 runs.
   * The event k* cancels from lead_θ − lead_η = k_η − k_θ, so the statistic does not test prediction.
   * Calibrating on 1-layer controls instead gives a verdict that depends on the control: with 6 heads, a tie; with 3
     heads, θ later by 2 to 3 checkpoints.
   * No calibration gave a lead of more than one checkpoint before the loss drop.
2. **R1's expected η win rests on one labelling choice.**
   * Every run has a layer-0 head that the draft's scores call role-less. It attends to t − 2 (0.20 to 0.34, against
     ≤ 0.10 for the other role-less heads).
   * Its single ablation costs 0.07 to 0.11 in p, more than any induction head. It interacts strongly with the
     previous-token head (+3.0 in log-odds).
   * θ's top non-circuit pair at C_U is (previous-token head, this head) in 9 of 9 runs.
   * θ − bounded η AUC at C_U, lr 3e-4 runs:

     | referee | mean |
     |---|---|
     | draft referee | −0.10 (SD 0.03) |
     | helper-head pairs excluded | −0.008 (a tie) |
     | at M_U | −0.015 |

   * Zero against mean ablation, and the choice of output, do not drive the result.
3. **The decision table contradicted the design draft.** Its §3 lets gate 2 pass only on R2 or R3, and says R1 by
   itself does not pass it. The draft let a θ win on R1 alone pass gate 2. With finding 1, θ could not pass at all.
   The reachable outcomes were "η wins" (R1 at C_U) and "tie", and this was not disclosed.
4. **Disclosure was incomplete.**
   * Tiled two-layer runs: 22 of 22 formed the circuit (not 12 of 12).
   * θ's lag in the R2-like preview was 1 to 4 checkpoints depending on the reference and the η value.
   * The M_U preview was a tie (−0.003).
   * In DR2b-pilot2 at C_U, θ's AUC was 0.78–0.89 against η's 1.00.
   * The earlier top-pair previews all favoured η.
   * 3 of the 6 R1 preview runs used lr 1e-3.
   * Counting redundancy pairs was decided after seeing θ's top pairs.
   * No intervention referee (R3), and no reason given.
   * At d_model 96 the loss first falls to ≤ 1 at step 339 in 4 of 4 runs.
   * The size check rests on 2 runs per size.

## Should consider

* **Checkpoint grid.** The pattern grid (60 of the 71 evaluation steps) is irregular and skips step 339; use all
  evaluation steps.
* **η at the two references.** Under the uniform design, bounded η is identical at C_U and M_U, and bounded and
  Gaussian η gave identical R1 AUCs in 5 of 6 runs. So the comparisons are far from independent.
* **Power.** The per-run SD of ΔAUC is about 0.03, so p reaches the bootstrap floor; bounded η's AUC is 1.0 in half
  the runs (a ceiling).
* **Wording to fix.** τ (percentile method, strict inequality, step 0), resamples with an empty size, M_U averaging
  in probability, rank ties, B1 slope (OLS of ln step on ln d_model, light-curve grid), B2 (ties count as "not
  before"), B3 by link and reference, IC-2 for 1-layer runs, the unused probe counts.
* **Store the final weights** so that referees can be recomputed.

## What was done

* **Engineering (both done in this commit):**
  * a t − 2 attention score (`prev2_…`) in the induction task's light curve;
  * the final weights stored with the pattern observations (`param/<name>` arrays), and so in data releases.
* **The design decision** that follows from findings 1 and 3 (whether Part B can be a second test of gate 2 at all)
  was put to the researcher before the pre-registration is revised.
