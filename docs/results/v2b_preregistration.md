# Sky v2, Part B: how induction circuits form; design, pre-registered criteria and run-time estimate

**Status: revised after independent review; consistency check pending. It becomes binding when this line is
removed.**

Written 16 November 2026, before any confirmatory Part B data exist.

**What Part B is.** A study of how the two-layer induction circuit forms in small attention-only transformers, and
how its formation changes with model size. It records the θ view (log-odds; RI) and the η view (means) of the
heads' interactions through training, and compares how well each agrees with head roles read from attention
patterns.

**What Part B is not: a test of gate 2.**
* The design draft (`v2_design_draft.md` §3) lets gate 2 pass only on a prediction referee (R2) or an intervention
  referee (R3), and says a mechanism referee (R1) by itself does not pass it.
* The independent review of the first draft (`v2_analysis/v2b_prereg_review.md`) showed two things:
  * On this task no R2 can tell the views apart. The circuit forms abruptly, and every measure crosses any
    control-calibrated threshold at the same checkpoint.
  * An R3 would be circular, because both views are computed from head ablations.
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
      steps later: steps 5,900 to 7,500 at d_model 96, and 7,500 to 8,400 at 48.
    * The size check rests on 2 runs per size.
  * **A context head.** In 9 of 9 runs checked, one layer-0 head without a previous-token, duplicate or induction
    role attends to t − 2 (0.20 to 0.34, against ≤ 0.10 for the other role-less heads).
    * Its single ablation costs 0.07 to 0.11 in p, more than any induction head.
    * It interacts strongly with the previous-token head (+3.0 in log-odds).
  * **Previews of the mechanism comparison** (§4), θ − bounded η AUC at C_U, lr 3e-4, 6 runs:

    | referee | mean |
    |---|---|
    | counting the context head as role-less | −0.10 (SD 0.03) |
    | excluding pairs with the context head | −0.008 |
    | at M_U | −0.015 (−0.003 in an earlier preview, which included lr 1e-3 runs) |

    * In the 2-head runs of DR2b-pilot2: θ 0.78–0.89 against η 1.00 at C_U; all 1.00 at M_U.
    * The earlier top-pair previews (the first probe, pilot 1 and the local 4-head runs) all favoured η. They counted
      the context head as role-less.
    * Counting redundancy pairs (same-role pairs) as circuit pairs was decided after seeing that θ's top pairs were
      redundancy pairs.
  * **Onset previews.** The first checkpoint at which a composition pair reaches 0.001 nats came for θ 1 to 4
    checkpoints after η, depending on the reference and the η value. With control-calibrated thresholds every view
    crossed at the same early checkpoint, between steps 0 and 11.
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
  repeated tokens: 128 fixed sequences with repeats, about 6,000 tokens. It is treated as a soft binary output, as in
  Part A's pattern mode. The runner's sampled probe counts are not used.
* **Measures.** `tycho.ri.cells.projection_information`, exactly as Part A §1: link θ (logistic, cross-entropy drop)
  and link η (least squares), with both η values (bounded, primary; Gaussian, sensitivity). m = 2p − 1 is clipped to
  ±(1 − 1e-12). Values within 1e-12 nats count as tied.
* **Not independent.** Under the uniform design, bounded η is identical at C_U and M_U (the Walsh features are
  orthogonal). In the pilots the bounded and Gaussian η values gave the same AUCs in 5 of 6 runs. So the comparisons
  of §4 are far from independent, and "wins with both η values" adds little for this design.
* **Sets.** All 15 head pairs. Singletons and triples are stored and reported but do not enter the tests.
* **Exact values.** All 64 patterns are evaluated at every checkpoint, so every measure is exact for the probe set.

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

* **IC-1.** Replays of 5 % of objects (at least one per configuration group) are bit-identical to the stored light
  curves (all keys except `wall_s`) and pattern arrays. A failure is a bug, and the analysis waits for the fix.
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
* **Formation precondition** (main runs). At the last checkpoint, all of:
  * at least one previous-token head;
  * at least one induction head;
  * at least one role-less head that is not a context head;
  * a repeat loss ≤ 0.1.

  Runs that fail enter neither §4 nor B2, and are listed. A size with fewer than 10 qualifying seeds is dropped from
  the §4 family statistic and listed.

## 4. Pre-registered questions and predictions

1. **B1, formation and scale.** The formation rate (the precondition of §3) and the phase-change step per size.
   * **Phase-change step:** the first light-curve evaluation with repeat loss ≤ 1.0.
   * **Slope:** ordinary least squares of ln(step) on ln(d_model) over the qualifying runs, with a 95 % joint-seed
     bootstrap percentile interval.
   * **Predicted:**
     * formation in at least 18 of 20 seeds at every size (the pilots: 2 of 2 at 48 and 192 in the size check, plus
       the review's 2 at 48; every run at 96);
     * a slope interval entirely below 0: larger models change earlier (previews about 690, 340 and 220 steps).
2. **B2, order of events.** Among qualifying runs, the share in which the largest induction score (layer 1) reaches
   0.5 before the largest previous-token score (layer 0) does. Read on the light-curve grid; reaching both at the
   same evaluation counts as "not before".
   * **Predicted:** at least 80 %. The pilots showed the previous-token head becoming clean thousands of steps after
     the phase change.
3. **M, mechanism agreement** (θ against η; reported, not part of gate 2).
   * **Pairs.**
     * Circuit pairs: every (previous-token head, induction head) pair, which is composition, and every pair of two
       heads with the same role, which is redundancy.
     * Non-circuit pairs: every pair with at least one role-less head.
   * **Two referees.**
     * (i) Context heads count as role-less.
     * (ii) Pairs that contain a context head and no other role-less head are excluded.
   * **Per run.** The AUC with which μ, at the last checkpoint, ranks circuit pairs above non-circuit pairs (ties count
     one half).
   * **Statistic.** Δ = the mean over sizes of the mean over qualifying runs of (AUC_θ − AUC_η). Positive favours θ.
     There is one Δ for each referee, reference (C_U, M_U) and η value.
   * **Bootstrap.**
     * 10,000 resamples, seed 20261116.
     * Each resample draws one multiset of the 20 seed labels with replacement and uses it for every size, because a
       seed's runs share their seed and data_seed across sizes. Runs that do not qualify are skipped.
     * The sizes in the family are fixed from the observed data. A resample with no qualifying run at a size is
       redrawn.
     * p = min(1, 2 · min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / (B + 1)).
   * **Comparisons.** Two, at C_U and M_U. Holm is applied over the two, separately for each referee and η value.
   * **Ahead.** A view is **ahead** at a reference if, under both referees and with both η values, Δ favours it,
     |Δ| ≥ 0.05 and the Holm-adjusted p is below 0.05.
     * Ahead under one referee only: **referee-dependent**.
     * Ahead with one η value only: **value-dependent**.
     * Neither counts as ahead.
   * **Outcomes.**

     | outcome | condition | meaning |
     |---|---|---|
     | **θ ahead on mechanism** | θ ahead at C_U or M_U, η at neither | the log-odds view agrees better with the attention-score roles |
     | **η ahead on mechanism** | η ahead somewhere, θ nowhere | the mean view agrees better with them |
     | **mixed** | both ahead somewhere | report which |
     | **tie** | neither ahead | the views agree equally well with the roles |

     None of these passes or fails gate 2.
   * **Secondary.**
     * The same with composition pairs only.
     * The rank of the best composition pair (1 + the number of pairs with a strictly larger value).
   * **Expected, from the previews:** **tie**. Under referee (i) η was ahead by about 0.10; under (ii) the difference
     was −0.008. So "referee-dependent" at C_U is the expected label.

## 5. Descriptive (no predictions)

* **Onsets against the phase change.** For each view, size and reference: the first checkpoint (step ≥ 1) at which the
  largest pair value exceeds a threshold τ (`numpy.percentile`, method "linear"; exceed means >), under three
  calibrations:
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
| d_model 192 | 30 | ≈ 12 min | ≈ 14 min | ≈ 13 h |
| **total** | 100 | | | **≈ 24 CPU hours** |

About 30 minutes of wall time on 86 workers; the longest single run (d_model 192) takes about 26 minutes. The
analysis is small: 15 pairs × 71 checkpoints × 100 runs.
