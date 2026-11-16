# Sky v2, Part B: how induction circuits form; design, pre-registered criteria and run-time estimate

Written 16 November 2026, before any confirmatory Part B data exist.

**What Part B is.** A study of how the two-layer induction circuit forms in small attention-only transformers, and
how its formation changes with model size. It records the θ view (log-odds; RI) and the η view (means) of the
heads' interactions through training, and compares how well each agrees with head roles read from attention
patterns.

**What Part B is not: a test of gate 2.**
* The design draft (`v2_design_draft.md` §3) lets gate 2 pass only on a prediction referee (R2) or an intervention
  referee (R3), and says a mechanism referee (R1) by itself does not pass it.
* The independent review of the first draft (`v2_analysis/v2b_prereg_review.md`) found that its prediction referee
  (R2) could not discriminate:
  * With thresholds calibrated on no-repeat controls, every view crossed at the same untrained-level checkpoint.
  * With thresholds from 1-layer controls, the verdict depended on which control was used.
  * No calibration gave a lead of more than one checkpoint before the loss drop.
* The first draft had no intervention referee (R3). An R3 that ablates heads and reads the same output on the same
  probe set would be circular, because both views are computed from exactly those 64 ablations. A non-circular R3,
  with another intervention or another output, has not been designed.
* On 16 November 2026 the researcher decided to run Part B as a formation study and leave gate 2 open. Gate 2 will
  be tested later with a real prediction or intervention referee, designed with Prof. Sugiyama.
* The mechanism comparison of §4 is reported, but no outcome of it passes or fails gate 2.

Background:
* [v2b_pilot_plan.md](v2b_pilot_plan.md): the pilots;
* [v2_analysis/v2b_prereg_review.md](v2_analysis/v2b_prereg_review.md): the review of the first draft;
* [v2a_results.md](v2a_results.md): Part A, "η wins" by its rule (1 of 12 comparisons, a weak win);
* [v2_design_draft.md](v2_design_draft.md) §3 and §5.

## 0. What is known before this pre-registration, and so not blind

* **Part A.** On behavioural referees the two views were largely interchangeable. Where they differed (saturated
  outputs), θ tracked behaviour worse. By Part A's decision table, η is the survey's detector, with RI as a
  descriptive companion. The first draft of this document planned a second gate-2 test; it was withdrawn (above).
* **The pilots** (`v2b_pilot_plan.md` §2; releases DR2b-pilot and DR2b-pilot2; local runs on seeds 300–321; the
  review's runs on seeds 330–332):
  * **Formation.**
    * With one repeated segment per sequence the circuit forms in few runs (2 of 12 on the server).
    * Tiling a random segment over the sequence formed it in 22 of 22 two-layer runs tried: 6 local runs with 2 or 4
      heads per layer (d_model 64, lr 1e-3), 6 server runs with 2 heads, 6 local runs with 3 heads (d_model 96),
      and 4 runs of a size check.
    * The review's runs add 1 at d_model 96 and 2 at d_model 48.
  * **Timing at learning rate 3e-4.**
    * The loss on repeated tokens first falls to ≤ 1.0 at step 339 in 4 of 4 runs at d_model 96 (light-curve grid).
    * The size check: at d_model 48 it falls at step 692 (2 of 2), at 192 at steps 211 and 237.
    * The induction scores rise with the drop. The previous-token head becomes clean (score ≥ 0.5) thousands of
      steps later:
      * steps 4,600 to 5,200 at d_model 96 (4 runs);
      * 7,500 to 8,400 at 48 (2 runs);
      * 5,900 to 7,500 in the 2-head d_model 64 runs of DR2b-pilot2.
    * The size check rests on 2 runs per size. The review's d_model 48 runs fell at steps 614 and 692.
  * **A context head.** In 9 of 9 three-head runs, the role-less head with the largest single-ablation cost is in
    layer 0.
    * Its ablation costs 0.07 to 0.11 in p, more than any induction head.
    * It interacts with the previous-token head: +2.3 to +3.3 in log-odds.
    * θ's top non-circuit pair at C_U is (previous-token head, this head).
    * Its attention to t − 2 could be measured only where weights were saved: 0.20 to 0.34 in the review's 3 models,
      against ≤ 0.10 for the other role-less heads.
    * In the 6 local d_model 64 models (2 or 4 heads per layer), exactly one such head attends to t − 2 at 0.29 to
      0.46, with the other role-less heads ≤ 0.08.
  * **Previews of the mechanism comparison** (§4), θ − bounded η AUC at C_U, lr 3e-4, 6 runs:

    | referee | mean |
    |---|---|
    | (i) counting the context head as role-less, at C_U | −0.10 (SD 0.03) |
    | (ii) as defined in §4, at C_U | −0.007 (5 runs) |
    | (i), at M_U | −0.015 (−0.003 in an earlier preview, which included lr 1e-3 runs) |
    | (ii), at M_U | −0.027 |

    * Referee (ii) is undefined in the sixth run (d_model 48, seed 331), which has no role-less head besides the
      context head. The review's own variant, which drops every pair with the context head, gave −0.008.
    * Where no weights were saved, the context head was taken as the review's ablation helper.
    * The composition-only AUC under referee (i) at C_U: −0.15.

    * In the 2-head runs of DR2b-pilot2, under referee (i): θ 0.78–0.89 against η 1.00 at C_U; all 1.00 at M_U.
      Referee (ii) is undefined in 2-head models.
    * The earlier top-pair previews (the first probe, pilot 1 and the local 4-head runs) all favoured η. They counted
      the context head as role-less.
    * Counting redundancy pairs (same-role pairs) as circuit pairs was decided after seeing that θ's top pairs were
      redundancy pairs.
  * **Onset previews.**
    * The first checkpoint at which a composition pair reaches 0.001 nats came for θ 1 to 4 checkpoints after η,
      depending on the reference and the η value.
    * With thresholds calibrated on no-repeat controls, every view crossed at about the same early checkpoint,
      between steps 0 and 11.
    * With thresholds from the 1-layer, 3-head control, θ crossed 2 to 3 pilot checkpoints after η at d_model 96,
      and 2 to 8 after it at d_model 48 (d_model 96 thresholds).
    * With a 1-layer, 6-head control the views crossed at about the same checkpoint.
    * No calibration gave a lead of more than one checkpoint before the loss drop.
* **Not blind.** The design, the referees and the expectations of §4 were written after these previews. The
  confirmatory seeds (400–419 and 450–459) are new.

## 1. Variables and measures

* **Variables.** The H = 6 attention heads (2 layers × 3), each kept (1) or zero-ablated (0). The ablation design is
  uniform over all 2^6 = 64 patterns, so D = U, and two references remain:
  * **C_U** (conditional, primary): the full table of the 6 heads. The measure of a pair A is the information its term
    adds given every term that does not contain A.
  * **M_U** (marginal): the table of the 2 heads of A alone. Cell means average 2p − 1 over the other heads, in
    probability space.
* **Output.** For pattern c, p(c) = the mean probability of the correct token over the probe set's predictable
  repeated tokens: 128 fixed sequences with repeats, about 6,700 to 6,800 tokens. It is treated as a soft binary output, as in
  Part A's pattern mode. The runner's sampled probe counts are not used.
* **Measures.** `tycho.ri.cells.projection_information`, exactly as Part A §1: link θ (logistic, cross-entropy drop)
  and link η (least squares), with both η values (bounded, primary; Gaussian, sensitivity). m = 2p − 1 is clipped to
  ±(1 − 1e-12). Values within 1e-12 nats count as tied.
* **Not independent.** Under the uniform design, bounded η is identical at C_U and M_U (the Walsh features are
  orthogonal). In the pilots the bounded and Gaussian η values gave the same AUCs in 5 of 6 runs. So the comparisons
  of §4 are far from independent, and "wins with both η values" adds little for this design.
* **Sets.** All 15 head pairs. Singletons and triples are stored and reported but do not enter the tests.
* **Exact values.** All 64 patterns (8 for the 1-layer controls, which have 3 heads) are evaluated at every
  checkpoint, so every measure is exact for the probe set.

## 2. Sky v2b

Configuration file `configs/sky_v2b.yaml` (§6).

* **Task.** `induction`, tiled layout:
  * vocabulary 64, sequences of 64 tokens;
  * segment length uniform in 4 to 16, random phase;
  * attention-only TinyGPT, 2 layers, 3 heads per layer.
* **Size knob.** d_model ∈ {48, 96, 192}: 16, 32 and 64 dimensions per head.
* **Training.**
  * Adam, learning rate 3e-4, batch 32, 12,000 steps;
  * 80 log-spaced evaluations, giving 71 distinct steps;
  * patterns observed at every evaluation step (`ri.n_points` 80), so the pattern grid is the evaluation grid:
    consecutive steps differ by a factor of about 1.13 from step 100 on.
* **Storage.** The final weights of every object are stored with its patterns (`param/…` arrays) and released.
* **Seeds.** 400–419 (20 per size), data_seed = seed.
* **Controls,** seeds 450–459, 10 per configuration:
  * no repeats (p_repeat = 0), 2 layers, at each size;
  * 1 layer, 3 heads, d_model 96, with repeats.
* **Objects.** 60 main + 30 no-repeat + 10 one-layer = 100.

## 3. Instrument checks, preconditions and roles

* **IC-1.** One replay per configuration group (7 groups, so 7 % of objects) is bit-identical to the stored light
  curve (all keys except `wall_s`), pattern arrays and `param/…` arrays. A failure is a bug, and the analysis waits
  for the fix.
* **IC-2, controls behave.**
  * Every no-repeat run ends with a repeat loss ≥ 4.0.
  * No 1-layer run has, at the end, both a head with previous-token score ≥ 0.5 and a head with induction score ≥ 0.5.
    For 1-layer runs both scores are read from layer 0.
  * A failure is reported and the analysis still runs; the report discusses it.
* **Roles,** fixed at the last checkpoint from the attention scores of `tycho.tasks.induction`. Neither view uses
  them.

  | role | layer | rule |
  |---|---|---|
  | previous-token head | 0 | prev ≥ 0.5 |
  | induction head | 1 | ind ≥ 0.5 |
  | context head | 0 | prev < 0.5 and prev2 ≥ 0.15 (the cut comes from 3 pilot models: context heads 0.20–0.34, other role-less heads ≤ 0.10) |
  | role-less | | any other head |
* **Formation** (main runs; B1, B2). At the last checkpoint: at least one previous-token head, at least one induction
  head, and a repeat loss ≤ 0.1.
* **M eligibility.** A formed run enters M only if it also has at least one role-less head that is not a context
  head; without one, referee (ii) has no non-circuit pair. In the pilots this failed in 1 of the 3 runs with saved
  weights (d_model 48, seed 331: the third layer-0 head became a second previous-token head, 0.73). It fails in every
  2-head model.
* **Listing and dropping.**
  * Runs that do not form, and formed runs that are not eligible, are listed.
  * Runs that do not form enter neither the B1 slope nor B2.
  * A size with fewer than 10 eligible seeds is dropped from M's Δ (both referees) and listed.
  * B1 and B2 use every formed run at every size.

## 4. Pre-registered questions and predictions

1. **B1, formation and scale.** The formation rate (formation, §3) and the phase-change step per size.
   * **Phase-change step:** the first light-curve evaluation with repeat loss ≤ 1.0.
   * **Slope:** ordinary least squares of ln(step) on ln(d_model) over the formed runs.
   * **Interval:** 95 % percentile interval from the joint-seed resampling described under M (10,000 resamples, seed
     20261116). Runs that did not form are skipped, and a resample with fewer than two sizes holding a formed run is
     redrawn.
   * **Predicted:**
     * formation in at least 18 of 20 seeds at every size. The pilots formed the circuit in 25 of 25 two-layer
       tiled runs; at 48 and 192 that is only 4 and 2 runs.
     * a slope interval entirely below 0: larger models change earlier (previews about 690, 340 and 220 steps).
2. **B2, order of events.** Among formed runs, the share in which the largest induction score (layer 1) reaches
   0.5 before the largest previous-token score (layer 0) does. Read on the light-curve grid; reaching both at the
   same evaluation counts as "not before".
   * **Predicted:** at least 80 %. The pilots showed the previous-token head becoming clean thousands of steps after
     the phase change.
3. **M, mechanism agreement** (θ against η; reported, not part of gate 2).
   * **Pairs.**
     * Circuit pairs: every (previous-token head, induction head) pair, which is composition, and every pair of two
       previous-token heads or of two induction heads, which is redundancy. A context head is never in a circuit
       pair.
     * Non-circuit pairs under referee (i): every pair with at least one role-less or context head.
     * Non-circuit pairs under referee (ii): every pair with at least one role-less head that is not a context head.
       Pairs whose heads are all context, previous-token or induction heads, with at least one context head, are
       left out.
   * **Per run.** The AUC with which μ, at the last checkpoint, ranks circuit pairs above non-circuit pairs (ties count
     one half).
   * **Statistic.** Δ = the mean over sizes of the mean over eligible runs of (AUC_θ − AUC_η). Positive favours θ.
     There is one Δ for each referee, reference (C_U, M_U) and η value.
   * **Bootstrap.**
     * 10,000 resamples, seed 20261116.
     * Each resample draws one multiset of the 20 seed labels with replacement and uses it for every size, because a
       seed's runs share their seed and data_seed across sizes. Runs that are not eligible are skipped.
     * The sizes in the family are fixed from the observed data. A resample with no eligible run at a size is
       redrawn.
     * p = min(1, 2 · min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / (B + 1)).
   * **Comparisons.** Two, at C_U and M_U. Holm is applied over the two, separately for each referee and η value.
   * **Labels.** A (referee, η value) combination is met when Δ favours the view, |Δ| ≥ 0.05 and the Holm p is
     below 0.05.
     * Met in all four combinations: **ahead**.
     * Met under one referee only (with one or both η values): **referee-dependent**.
     * Met with one η value only (under one or both referees): **value-dependent**.
     * A single met combination carries both labels. Neither label counts as ahead.
   * **Outcomes.**

     | outcome | condition | meaning |
     |---|---|---|
     | **θ ahead on mechanism** | θ ahead at C_U or M_U, η at neither | the log-odds view agrees better with the attention-score roles |
     | **η ahead on mechanism** | η ahead somewhere, θ nowhere | the mean view agrees better with them |
     | **mixed** | both ahead somewhere | report which |
     | **tie** | neither ahead | no difference holds under both referees and both η values; any referee- or value-dependent difference is reported with it |

     None of these passes or fails gate 2.
   * **Secondary.**
     * The same with composition pairs only (redundancy pairs left out).
     * The rank of the best composition pair: 1 + the number of pairs whose value is larger by more than 1e-12.
   * **Expected, from the previews:** **tie**. Under referee (i) η was ahead by about 0.10; under (ii) the difference
     was −0.007. So "referee-dependent" at C_U is the expected label.
   * **Ceiling.** Bounded η's AUC was at its ceiling of 1.0 in 3 of 6 pilot runs under (i) and in 4 of 5 under (ii).
     θ can be ahead only where η falls below 1, so "θ ahead" is nearly unreachable on this design.

## 5. Descriptive (no predictions)

* **Onsets against the phase change.** For each view, size, reference and η value: the first checkpoint (step ≥ 1) at
  which the largest pair value exceeds a threshold τ. τ is computed with `numpy.percentile`, method "linear", and
  "exceed" means strictly greater. Its pool is the largest pair value over all checkpoints with step ≥ 1 of all
  controls of that configuration. The 1-layer controls have only 3 pairs and 8 patterns. Three calibrations:
  * (a) the 99th percentile in the no-repeat controls of that size;
  * (b) the 99th percentile in the 1-layer controls (d_model 96 thresholds);
  * (c) a fixed 0.001 nats.
  The step is compared with the phase-change step. The review found no lead of more than one checkpoint in the
  pilots, which is why this is descriptive.
* **B3, composition against redundancy.** For each view, reference and η value: the share of its total pair
  information at the last checkpoint that falls on composition pairs, on redundancy pairs, on pairs with a context
  head, and on the remaining pairs.
* **Trajectories.** θ and η of the composition, redundancy and context pairs through training, by size. Part B's
  description of formation.

## 6. What must be built before the run

1. `configs/sky_v2b.yaml` and `ops/linux/start_v2b.sh`. A test already builds every point of every sky file.
2. The offline analysis (`tycho/analysis/v2b.py`, `scripts/analysis_v2/v2b_analysis.py`):
   * the measures for all pairs at C_U and M_U, with both η values, at every checkpoint;
   * roles, IC-1, IC-2, B1, B2 and M with the joint-seed bootstrap and Holm;
   * the descriptive part of §5.

   The code is verified independently on pilot data before the run, as in Part A.
3. A data release after the run (patterns with final weights, light curves). An independent agent recomputes the
   results from it.

## 7. Run-time estimate on the 88-core server

| size | runs | training per run (measured) | 64 patterns × 71 checkpoints per run (from the pilots) | total CPU |
|---|---|---|---|---|
| d_model 48 | 30 | ≈ 3 min | ≈ 4 min | ≈ 3.5 h |
| d_model 96 | 40 | ≈ 4.5 min | ≈ 6 min | ≈ 7 h |
| d_model 192 | 30 | ≈ 12 min | ≈ 14 min (estimated; not measured) | ≈ 13 h |
| **total** | 100 | | | **≈ 24 CPU hours** |

About 30 to 35 minutes of wall time on 86 workers: 100 runs leave 14 for a second wave, and the longest single run
(d_model 192) takes about 26 minutes. The patterns were measured at 1.3 to 3.4 minutes per run at d_model 96, so
the table is conservative. The analysis is small: 15 pairs × 71 checkpoints × 100 runs.

## Amendments

* **16 November 2026, before binding.** The first draft (commits 61754e2 and 045380d) planned a second test of gate 2.
  * An independent review found it could not serve as one (`v2_analysis/v2b_prereg_review.md`).
  * The researcher decided to run Part B as a formation study.
  * A consistency check of the revision then found:
    * the formation precondition coupled to M (now separated into formation and M eligibility);
    * pair definitions that could be read two ways (now fixed);
    * five overstated or wrong statements: the R2/R3 attribution, the onset previews, the d_model 96 timing, the
      context-head evidence, and the meaning of "tie";
    * minor wording.

    All are applied. Record: the review file, section "Consistency check of the revision".
* **16 November 2026, clarifications from implementing the analysis** (before any confirmatory data; no definition
  changes):
  * **Labels.** The labels of §4 M are made exhaustive.
    * Combinations met under exactly one referee: referee-dependent.
    * Met with exactly one η value: value-dependent. A single met combination carries both labels.
    * Met combinations spanning both referees and both η values without being all four (three combinations, or a
      "diagonal" pair): **partial**.
    * Only "ahead" counts as ahead.
  * **Onsets (§5).** "Exceed" applies the tie tolerance of §1: a value exceeds τ when it is larger by more than
    1e-12.
  * **B1's bootstrap** uses its own stream started from seed 20261116, with the same joint-seed scheme as M and B1's
    redraw rule.
  * **Trajectories.** "Context pairs" are the pairs that contain a context head, as in B3.
  * **The B1 interval** can be very narrow: phase-change steps snap to the evaluation grid (about 13 % apart), and in
    the analysis pilot both seeds of each size gave the same step. "Entirely below 0" then mostly reflects the
    per-size steps, and the report will say so.
  * **The analysis code** is `tycho/analysis/v2b.py` and `scripts/analysis_v2/v2b_analysis.py`. Every choice the text
    leaves open is listed in its DECISIONS (20 items). It was run on a pilot of 10 objects (seeds 900–901, sky
    version sky-v2b-pilot3):
    * IC-1 5 of 5, IC-2 passed;
    * 4 of 4 main runs formed;
    * M: "tie", with η referee-dependent at C_U.
    * Verification by an independent agent follows before the run.
