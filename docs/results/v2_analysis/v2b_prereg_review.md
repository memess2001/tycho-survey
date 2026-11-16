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
* **The design decision** that follows from findings 1 and 3, whether Part B can be a second test of gate 2 at all,
  was put to the researcher. The answer (16 November 2026): run Part B as a study of how the circuit forms (B1,
  B2), and report the θ/η mechanism comparison without counting it toward gate 2. Gate 2 stays open for a later test
  with a real prediction or intervention referee, designed with Prof. Sugiyama.
* **The revised pre-registration** applies the remaining findings:
  * R2 is descriptive, with three calibrations.
  * The mechanism comparison uses two referees, the context head counted as role-less or excluded. Being ahead
    under one only is "referee-dependent".
  * §0 discloses everything in finding 4 and the review's own runs.
  * Smaller fixes:
    * the pattern grid equals the evaluation grid;
    * a tie tolerance;
    * η at C_U and M_U noted as not independent;
    * exact wording for B1, B2, the bootstrap, M_U and IC-2;
    * the final weights are stored.

## Consistency check of the revision (16 November 2026)

A second agent checked the revised text (commit 5009258) against this record and the pilot data. It edited nothing.
It confirmed:
* the counts, the pattern grid (all 71 evaluation steps with `n_points` 80) and the gate-2 reading of the design
  draft;
* most numbers, including bounded η being identical at C_U and M_U (largest difference 2e-16);
* the context-head rule picking the ablation helper in 3 of 3 models with saved weights.

It found, and the pre-registration now fixes:

| # | finding | fix |
|---|---|---|
| 1 | "No R2 can tell the views apart" and "an R3 would be circular" overstated the review. With 1-layer 3-head thresholds, θ crossed 2 to 3 checkpoints after η at d_model 96 and 2 to 8 at 48. | Reworded: what the review found, and why an ablation-based R3 on the same output would be circular. |
| 2 | The onset preview said every calibration crossed together. | The three calibrations are reported separately. |
| 3 | The previous-token head timing at d_model 96 was wrong (it is 4,600 to 5,200; the quoted 5,900 to 7,500 is from the 2-head d_model 64 runs). | Corrected. |
| 4 | "9 of 9 attend to t − 2" overstated: attention was measured only in 3 models. | Reworded. New evidence added: the 6 local d_model 64 models each have one such head (0.29 to 0.46). |
| 5 | The formation precondition required a non-context role-less head. That fails in a run that formed the circuit (d_model 48, seed 331) and in every 2-head model. | Separated into **formation** (B1, B2) and **M eligibility**. The (ii) preview corrected to −0.007 (5 runs); the M_U (ii) preview −0.027 added. |
| 6 | Pair definitions could be read two ways (two context heads as a same-role pair). | Circuit pairs and both referees' non-circuit pairs defined exactly. |
| 7 | "Tie" was described as equal agreement. | "No difference under both referees and both η values; referee- or value-dependent differences reported with it." |

Minor points also applied:
* labels for partial patterns;
* the composition-only preview (−0.15);
* the rank tie tolerance;
* the τ pool;
* the probe size (6,700 to 6,800 tokens);
* IC-1, one replay per configuration group, including the weights;
* the review's d_model 48 steps (614, 692);
* the ceiling of bounded η's AUC, which makes "θ ahead" nearly unreachable;
* the 2-head preview marked as referee (i);
* the run-time note.
