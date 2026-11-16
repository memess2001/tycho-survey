# Sky v2, Part B: induction circuits; design, pre-registered criteria and run-time estimate

**Status: draft under independent review. It becomes binding when this line is removed.**

Written 16 November 2026, before any confirmatory Part B data exist.

Part B studies how the two-layer induction circuit forms in small attention-only transformers, and it is the second
test of gate 2: does the θ view (log-odds) single out what the network computes better than the η view (means)?
Part A's referees were behavioural and weak. Part B's referees are the circuit itself (R1, mechanism) and the timing
of its formation (R2, prediction).

Background:
* [v2b_pilot_plan.md](v2b_pilot_plan.md): the pilots, all disclosed below;
* [v2a_results.md](v2a_results.md): Part A, "η wins" by its rule (1 of 12 comparisons; a weak win);
* [v2_design_draft.md](v2_design_draft.md) §3 and §5.

## 0. What is known before this pre-registration, and so not blind

* **Part A.** On behavioural referees the two views were largely interchangeable; where they differed (saturated
  outputs), θ tracked behaviour worse. By Part A's decision table, η became the survey's detector, with RI as a
  descriptive companion. The researcher decided on 16 November 2026 to run this second test, to give θ a fairer
  chance with stronger referees. **Two tests have now been planned. A θ win here is a win in the
  second of two tests, and it will be reported as such.**
* **The pilots** (`v2b_pilot_plan.md` §2; data releases DR2b-pilot and DR2b-pilot2; local runs on seeds 300–321):
  * With one repeated segment per sequence the circuit forms in few runs (2 of 12 on the server). Tiling a random
    segment over the sequence forms it in every run tried: 12 of 12 two-layer runs with 2 or 3 heads per layer.
  * The phase change (the loss on repeated tokens falls from about 3.2 to below 1) comes at steps 300 to 430 at
    learning rate 3e-4. The induction scores rise with it. The previous-token head becomes clean only thousands of
    steps later.
  * **Previews of the referees,** on 6 runs with 3 heads per layer (d_model 96):
    * R1-like (§4): AUC of circuit pairs against pairs with a head outside the circuit, θ 0.63–0.94 (mean about
      0.85), η 0.76–1.00 (mean about 0.94); η ahead in 6 of 6.
    * R2-like: the first checkpoint at which a composition pair reaches 0.001 nats came at or one checkpoint after the
      loss drop for η, and one or two checkpoints later for θ. That preview used fixed roles and a fixed threshold,
      unlike R2 below.
    * In the 2-head runs every measure put the redundant induction-head pair first.
  * **A size check** (training only, seeds 320–321, 3 heads per layer, learning rate 3e-4). The circuit formed at
    d_model 48 (2 of 2, phase change at step 692) and 192 (2 of 2, steps 211 and 237). At 96 the pilots put it at
    steps 339 to 430. Training took about 3 minutes at 48 and 12 minutes at 192 on one core.
* **Not blind.** The design, the referees and the expectations of §4 were written after these previews. The
  confirmatory seeds (400–419 and 450–459) are new.

## 1. Variables and measures

* **Variables.** The H = 6 attention heads (2 layers × 3), each kept (1) or zero-ablated (0). The ablation design is
  uniform over all 2^6 = 64 patterns, so D = U and only two references differ:
  * **C_U** (conditional): the full table of the 6 heads; the measure of a pair A is the information its term adds
    given every term that does not contain A. Primary.
  * **M_U** (marginal): the table of the 2 heads of A alone, with the output averaged uniformly over the other heads.
* **Output.** For pattern c, p(c) = the mean probability of the correct token over the probe set's predictable
  repeated tokens (128 fixed sequences of the training distribution with repeats; about 6,000 tokens). Treated as a
  soft binary output, exactly as in Part A's pattern mode.
* **Measures.** `tycho.ri.cells.projection_information` with link θ (logistic, cross-entropy drop) and link η
  (least squares), η with both values (bounded, primary; Gaussian, sensitivity), exactly as Part A §1. m = 2p − 1
  clipped to ±(1 − 1e-12).
* **Sets.** All 15 head pairs. Singletons and triples are stored and reported but do not enter the tests.
* **Exact values.** All 64 patterns are evaluated at every checkpoint, so every measure is exact for the probe set.

## 2. Sky v2b

Configuration file `configs/sky_v2b.yaml`, to be written with the engineering (§6).

* **Task.** `induction`, tiled layout: vocabulary 64, sequences of 64 tokens, segment length uniform in 4 to 16,
  random phase. Attention-only TinyGPT, 2 layers, 3 heads per layer.
* **Size knob.** d_model ∈ {48, 96, 192} (16, 32 and 64 dimensions per head).
* **Training.** Adam, learning rate 3e-4, batch 32, 12,000 steps, 80 log-spaced evaluations, 60 log-spaced pattern
  checkpoints (the runner's subsample of the evaluations).
* **Seeds.** 400–419 (20 per size), data_seed = seed.
* **Controls,** seeds 450–459 (10 per configuration):
  * no repeats (p_repeat = 0), 2 layers, at each size: these calibrate R2's threshold;
  * 1 layer, 3 heads, d_model 96, with repeats: a model that cannot form the circuit.
* **Objects.** 60 main + 30 no-repeat + 10 one-layer = 100.

## 3. Instrument checks and preconditions

* **IC-1.** Replays of 5 % of objects (at least one per configuration group) are bit-identical to the stored light
  curves (all keys except `wall_s`) and pattern arrays. A failure is a bug; the tests wait for the fix.
* **IC-2, controls behave.** Every no-repeat run ends with a repeat loss ≥ 4.0. No 1-layer run has a previous-token
  head and an induction head (scores ≥ 0.5) at the end. A failure is reported and the tests still run (a control can
  fail for a scientific reason); the report discusses it.
* **Formation precondition** (main runs). At the last checkpoint: at least one layer-0 head with previous-token score
  ≥ 0.5, at least one layer-1 head with induction score ≥ 0.5, at least one head with neither role, and a repeat loss
  ≤ 0.1. Runs that fail enter neither test and are listed. A size with fewer than 10 qualifying seeds is dropped from
  the family statistics and listed. The formation rate per size is a result (B1).
* **Roles** are fixed at the last checkpoint from the attention scores (the scores of `tycho.tasks.induction`):
  previous-token heads (layer 0, score ≥ 0.5), induction heads (layer 1, score ≥ 0.5), heads with neither role. Neither
  view uses the attention scores.

## 4. Confirmatory tests of gate 2

**R1, mechanism.**
* **Pairs.** Circuit pairs: every (previous-token head, induction head) pair (composition) and every pair of two
  heads with the same role (redundancy). Non-circuit pairs: every pair with at least one head without a role.
* **Per run.** The AUC with which μ, at the last checkpoint, ranks circuit pairs above non-circuit pairs (ties count
  one half).
* **Statistic.** Δ_R1 = mean over sizes of the mean over qualifying runs of (AUC_θ − AUC_η). Positive favours θ.
* **Secondary.** The same with composition pairs only; and the rank of the best composition pair.

**R2, prediction.**
* **Event.** k* = the first pattern checkpoint at which the repeat loss (light curve, same step) is ≤ 1.0.
* **Threshold, calibrated per measure and size.** τ_μ = the 99th percentile, over all checkpoints of all no-repeat
  control runs of that size, of the largest value of μ over the 15 head pairs.
* **Onset.** k_μ = the first checkpoint at which the largest value of μ over the 15 pairs exceeds τ_μ. No knowledge of
  the final roles is used. A run whose measure never exceeds τ_μ gets k_μ = 60 (the number of checkpoints).
* **Per run.** lead_μ = k* − k_μ, in checkpoints (one checkpoint is a factor of about 1.17 in steps).
* **Statistic.** Δ_R2 = mean over sizes of the mean over qualifying runs of (lead_θ − lead_η). Positive favours θ
  (an earlier warning).
* **Secondary.** The false-alarm share: 1-layer control runs in which μ exceeds τ_μ (d_model 96 threshold) at any
  checkpoint; and the lead with the event defined as loss ≤ 2.0.

**Comparisons and decision.** Four comparisons: R1 and R2, each at C_U and M_U.
* **Bootstrap.** 10,000 resamples, seed 20261116. Each resample draws one multiset of the 20 seed labels with
  replacement and uses it for every size (a seed's runs share their seed and data_seed across sizes). Runs that do
  not qualify are skipped. p = min(1, 2 · min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / (B + 1)).
* **Margins.** R1: |Δ| ≥ 0.05 (AUC). R2: |Δ| ≥ 1 checkpoint.
* **A link wins a comparison** if, with the bounded η value, Δ favours it, |Δ| reaches the margin and the
  Holm-adjusted p over the four comparisons is below 0.05, and the same holds with the Gaussian value (Holm over its
  own four). A comparison that meets one value only is **value-dependent** and is not a win.

| outcome | condition | meaning | next |
|---|---|---|---|
| **gate 2 passed (θ)** | θ wins at least one comparison and η none | the log-odds view singles out the circuit, or warns of it, better | RI returns as the primary view for v3 |
| **η wins** | η wins at least one and θ none | the mean view is the better detector of circuits too | RI stays a descriptive companion; v3 uses the η view; the theory question (why) becomes a research line of its own |
| **mixed** | both win somewhere | depends on referee or reference | report which; v3 pre-registers the split |
| **tie** | no win | interchangeable on these referees | use the cheaper view; RI's value is description |

**Expectations, informed by the previews (§0).** η wins R1 at C_U (the preview Δ is about −0.09, beyond the 0.05
margin). R2 is uncertain: the previews showed θ later by one or two checkpoints, but with a different threshold.
Overall "η wins" is expected; a θ win is not.

## 5. Pre-registered questions (estimates; any outcome is reported)

1. **B1, formation and scale.** The formation rate and the phase-change step per size.
   * Predicted: formation in at least 18 of 20 seeds at every size.
   * Predicted: a negative log-log slope of the phase-change step against d_model, with its bootstrap interval
     below 0. Larger models change earlier: the previews gave about 690, 340 and 220 steps.
2. **B2, order of events.** The share of qualifying runs in which the induction score reaches 0.5 before the
   previous-token score does. Predicted: at least 80 % (the pilots showed the previous-token head becoming clean
   thousands of steps after the phase change).
3. **B3, composition against redundancy (exploratory).** For each view, the share of its total pair information at the
   last checkpoint that falls on composition pairs, on redundancy pairs and on non-circuit pairs.

## 6. What must be built before the run

1. `configs/sky_v2b.yaml` and `ops/linux/start_v2b.sh`; a test builds every point (already in place for all sky
   files).
2. The offline analysis (`tycho/analysis/v2b.py`, `scripts/analysis_v2/v2b_analysis.py`): measures for all pairs at
   C_U and M_U with both η values at every checkpoint, roles, R1, R2 with the control-calibrated thresholds, the
   joint-seed bootstrap and Holm, the decision, B1–B3, IC-1 and IC-2. Independent verification of the code on pilot
   data before the run, as in Part A.
3. A data release after the run (patterns and light curves), from which an independent agent recomputes the results.

## 7. Run-time estimate on the 88-core server

| size | runs | training per run (measured) | 64 patterns × 60 checkpoints per run (estimated) | total CPU |
|---|---|---|---|---|
| d_model 48 | 30 | ≈ 3 min | ≈ 3 min | ≈ 3 h |
| d_model 96 | 40 | ≈ 4.5 min | ≈ 5 min | ≈ 6.5 h |
| d_model 192 | 30 | ≈ 12 min | ≈ 12 min | ≈ 12 h |
| **total** | 100 | | | **≈ 21 CPU hours** |

About 30 minutes of wall time on 86 workers; the longest single run (d_model 192) takes about 25 minutes.

The analysis is small: 15 pairs × 60 checkpoints × 100 runs.
