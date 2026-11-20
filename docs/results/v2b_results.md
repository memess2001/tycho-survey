# Sky v2, Part B: results (how induction circuits form)

Pre-registered analysis of sky v2 Part B, run on the server on 20 November 2026. Output:
`v2_analysis/v2b_analysis.json`, code at commit 55ba69f. Data release: `releases/DR2b` (light curves, patterns with
final weights).

**Verified independently.** A second agent recomputed every confirmatory number from the raw data release with its
own code, and the exploratory numbers of §8 as well (record:
[v2_analysis/v2b_results_verification.md](v2_analysis/v2b_results_verification.md)). Every count, B1, B2 and M number
matched. Its corrections to the wording and to some descriptive statements are applied.

Pre-registration: [v2b_preregistration.md](v2b_preregistration.md), binding, with its amendments. Every rule below
was fixed before the confirmatory data existed: formation, roles, the phase-change step, the predictions of B1 and
B2, the referees and labels of M, the bootstrap and the onset calibrations. Part B is a formation study, not a test
of gate 2.

## 1. Summary

* **The run and the checks.**
  * 100 of 100 objects done; nothing withheld.
  * IC-1: 7 of 7 replays bit-identical on the server. IC-2: every control behaved. Every data check passed.
* **B1, formation and scale: both predictions met.**
  * The circuit formed in 20 of 20 seeds at every size (60 of 60).
  * The phase change (repeat loss first ≤ 1.0) came at a median step of 692 at d_model 48, 339 at 96 and 211 at 192.
  * The slope of ln(step) on ln(d_model) is −0.814, 95 % interval −0.835 to −0.792.
  * **Caveat.** Every run of a size falls on one or two points of the evaluation grid, which are about 13 % apart, so
    the narrow interval mostly reflects the grid (amendment of 16 November). Three sizes span a factor of 4. This is
    not evidence of a power law.
* **B2, order of events: prediction met in 60 of 60.** The largest induction score reaches 0.5 before the largest
  previous-token score does, in every formed run.
  * Induction ≥ 0.5: median steps 545, 301 and 211.
  * Previous-token ≥ 0.5: median steps 7,458, 4,636 and 4,636, thousands of steps after the phase change.
* **M, mechanism agreement (θ against η): "tie", the expected outcome.**
  * Referee (i) at C_U: Δ favours η by 0.068 (bounded) and 0.067 (Gaussian), Holm p 0.0004. It is met under referee
    (i) only, so it is "referee-dependent", which does not count as ahead.
  * Under referee (ii), and at M_U under both referees, |Δ| ≤ 0.009 and nothing is met.
  * d_model 192 is dropped from M: only 7 of its 20 runs are eligible (minimum 10).
* **Descriptive.**
  * No view's onset leads the phase change by more than one checkpoint, except with thresholds from the no-repeat
    controls. Those sit at untrained-network level and are crossed early (median onset steps 22, 3 and 1; range 1 to
    31).
  * At C_U, θ puts less of its pair information than η on composition pairs (0.17 to 0.23 against 0.44 to 0.49) and
    more on redundancy pairs and on pairs with the context head (0.16 to 0.20 against 0.02 to 0.04). The context
    share is what referee (i) penalises.
* **Not predicted: a strong t − 2 head at d_model 192.**
  * In 9 of 20 runs at d_model 192, a layer-0 head attends to t − 2 with a score of 0.51 to 0.68.
  * At 48 and 96 the largest is 0.33.
  * 12 runs at 192 have two context heads (the 9 with a strong t − 2 head and 3 others), and one more has two
    previous-token heads. None of these 13 has a role-less head, so only 7 are eligible for M.

## 2. Instrument checks

| check | result |
|---|---|
| IC-1 replays | 7 of 7 groups bit-identical on the server: light curves (all keys except `wall_s`), pattern arrays and `param/…` arrays. Bit-identity holds on the server's hardware and thread count only: the verifier's replay of another object on a different machine differed by 3e-10 in the log-odds at step 0, growing to 2e-3 in the repeat loss, with the same phase change, roles and formation |
| IC-2, no-repeat | 30 of 30 end with repeat loss ≥ 4.0 (4.158 to 4.160; chance 4.159) |
| IC-2, 1 layer | 10 of 10 pass: no layer-0 head reaches previous-token 0.5 (largest 0.03) or induction 0.5 (largest 0.23) |
| data checks | all passed: H matches the configuration, input names, steps strictly increasing from 0, uniform design, finite values, pattern grid = light-curve grid, 71 checkpoints |

## 3. B1, formation and scale

| d_model | formed | phase-change steps (light-curve grid) | median | geometric mean |
|---|---|---|---|---|
| 48 | 20 of 20 | 614 (7 runs), 692 (13) | 692 | 664 |
| 96 | 20 of 20 | 339 (20) | 339 | 339 |
| 192 | 20 of 20 | 211 (17), 237 (3) | 211 | 215 |

* Slope −0.814 (OLS over 60 runs); joint-seed bootstrap (10,000, no redraws): 95 % interval −0.835 to −0.792. The
  verifier's other bootstrap seeds give the same interval to 1e-4.
* The predictions (formation in at least 18 of 20 at every size; interval entirely below 0) are met.
* Final repeat loss of formed runs: 0.008 to 0.068 at 48, 0.005 to 0.010 at 96, 0.003 to 0.010 at 192.
* **Roles at the last checkpoint.** All 60 runs have three induction heads, that is, every layer-1 head.

  | layer 0 (previous-token, context, role-less) | d_model 48 | 96 | 192 |
  |---|---|---|---|
  | 1, 1, 1 | 16 | 19 | 7 |
  | 1, 2, 0 | 1 | 0 | 12 |
  | 2, 1, 0 | 1 | 1 | 1 |
  | 2, 0, 1 | 1 | 0 | 0 |
  | 1, 0, 2 | 1 | 0 | 0 |

## 4. B2, order of events

| d_model | induction before previous-token | induction ≥ 0.5, median (range) | previous-token ≥ 0.5, median (range) |
|---|---|---|---|
| 48 | 20 of 20 | 545 (484 to 614) | 7,458 (5,221 to 12,000) |
| 96 | 20 of 20 | 301 (301) | 4,636 (3,655 to 5,221) |
| 192 | 20 of 20 | 211 (187 to 211) | 4,636 (3,245 to 10,655) |

* Predicted: at least 80 %. Observed 60 of 60.
* **Descriptive, not pre-registered.** The induction step shrinks with size nearly as fast as the phase change (OLS
  slope of ln step on ln d_model −0.70, against −0.81). The previous-token step shrinks much less (−0.27), with a wide spread at 48
  and 192. In one run at 48 the previous-token head reaches 0.5 only at the last evaluation (step 12,000).

## 5. M, mechanism agreement

Δ = mean over sizes of the mean over eligible runs of (AUC_θ − AUC_η); negative favours η. Family: d_model 48 (18
eligible runs) and 96 (19). Margin 0.05, Holm over the two references for each referee and η value.

| referee | η value | reference | Δ | Δ at 48 | Δ at 96 | 95 % interval | Holm p | met |
|---|---|---|---|---|---|---|---|---|
| (i) | bounded | C_U | **−0.068** | −0.044 | −0.093 | −0.080 to −0.056 | 0.0004 | η |
| (i) | bounded | M_U | −0.005 | +0.001 | −0.011 | −0.014 to +0.006 | 0.34 | no |
| (i) | Gaussian | C_U | **−0.067** | −0.044 | −0.090 | −0.079 to −0.054 | 0.0004 | η |
| (i) | Gaussian | M_U | +0.007 | +0.005 | +0.010 | −0.001 to +0.017 | 0.11 | no |
| (ii) | bounded | C_U | +0.002 | +0.005 | 0.000 | −0.010 to +0.015 | 0.69 | no |
| (ii) | bounded | M_U | −0.009 | −0.009 | −0.009 | −0.020 to +0.003 | 0.29 | no |
| (ii) | Gaussian | C_U | +0.002 | +0.005 | 0.000 | −0.010 to +0.015 | 0.69 | no |
| (ii) | Gaussian | M_U | −0.006 | −0.007 | −0.005 | −0.016 to +0.005 | 0.52 | no |

* **Outcome: tie.** η is met at C_U under referee (i) only (both η values), so it is "referee-dependent" there.
  Nothing else is met. This is the outcome the pre-registration expected from the pilots.
* **Mean AUCs** (eligible runs), referee (i) at C_U: θ 0.89 and 0.88, bounded η 0.94 and 0.97 at 48 and 96. Under
  referee (ii) at C_U, both views average 0.96 to 0.98.
* **Ceiling.** Under referee (ii) at C_U, bounded η's AUC is exactly 1 in 13 of 18 runs at 48 and 15 of 19 at 96; θ's
  in 11 and 12. As the pre-registration noted, "θ ahead" was nearly unreachable.
* **Secondary, composition pairs only.** All eight Δ favour η.
  * Referee (i) at C_U: −0.117 (both η values; −0.099 at 48, −0.135 at 96), Holm p 0.0004.
  * Referee (i) at M_U: −0.010 (bounded) and −0.011 (Gaussian; Holm p 0.017), below the margin.
  * Referee (ii): −0.021 at C_U and −0.011 at M_U (both η values), Holm p 0.054, below the margin.
* **Rank of the best composition pair** (all 60 formed runs). For bounded η at C_U it is 2 in 52 runs, 1 in 4 and 3
  in 4. For θ at C_U the median is 3 (range 1 to 7); at M_U it is 2. Gaussian η at M_U puts a composition pair first
  in 14 of 20 runs at 48, 2 of 20 at 96 and 12 of 20 at 192 (28 of 60).
* **The top pair** at C_U is a redundancy pair of two induction heads in 59 of 60 runs for θ and 55 of 60 for bounded
  η (η: a composition pair in 4, two previous-token heads in 1).

## 6. Descriptive (no predictions)

**Onsets against the phase change.** The first checkpoint at which the largest pair value exceeds τ; "lead" is the
number of checkpoints before the phase change. C_U shown; M_U differences are listed below the table.

| calibration | d_model | θ: median onset step; lead range | bounded η | Gaussian η | phase change |
|---|---|---|---|---|---|
| (a) no-repeat p99 | 48 / 96 / 192 | 22 / 3 / 1 (leads 26 to 39) | same | same | 692 / 339 / 211 |
| (b) 1-layer p99 | 48 | 3,450; −23 to −8 | 692; −2 to 0 | 692; −1 to +1 | 692 |
| (b) | 96 | 1,590; −26 to −10 | 430; −3 to −1 | 382; −2 to 0 | 339 |
| (b) | 192 | 4,636 (17 of 20 cross); −34 to −4 | 237; −1 to 0 | 211; −1 to 0 | 211 |
| (c) fixed 0.001 nats | 48 | 692; −1 to 0 | 692; −1 to +1 | 614; 0 to +1 | 692 |
| (c) | 96 | 339; −1 to 0 | 382; −2 to 0 | 339; 0 to +1 | 339 |
| (c) | 192 | 211; −1 to 0 | 211; −1 to 0 | 211; 0 to +1 | 211 |

* (a) is crossed at untrained-network level, as the review of the first draft found.
* With (b) and (c), no view leads by more than one checkpoint, at C_U or at M_U. θ's 1-layer threshold (0.0069 nats
  at C_U, against 0.0013 to 0.0019 for η) is crossed long after the phase change.
* **M_U.** θ under (b) crosses earlier than at C_U, though still well after the phase change: median steps 2,144,
  1,412 and 1,183 at 48, 96 and 192 (leads −16 to −6, −15 to −7, −22 to −3; 20 of 20 cross at 192). θ under (c):
  median 382 at 96 and 237 at 192. Gaussian η under (b) at 96: leads −4 to 0. The other cells equal C_U's.

**B3, where the pair information sits at the last checkpoint** (mean share over formed runs).

| d_model | reference | view | composition | redundancy | context | other |
|---|---|---|---|---|---|---|
| 48 | C_U | θ | 0.23 | 0.57 | 0.16 | 0.03 |
| 48 | C_U | η bounded | 0.49 | 0.44 | 0.04 | 0.03 |
| 96 | C_U | θ | 0.17 | 0.61 | 0.20 | 0.02 |
| 96 | C_U | η bounded | 0.44 | 0.51 | 0.03 | 0.02 |
| 192 | C_U | θ | 0.21 | 0.62 | 0.16 | 0.00 |
| 192 | C_U | η bounded | 0.44 | 0.53 | 0.02 | 0.00 |
| 48 / 96 / 192 | M_U | θ | 0.34 / 0.37 / 0.45 | 0.59 / 0.57 / 0.53 | 0.02 to 0.04 | ≤ 0.04 |

Bounded η is the same at M_U as at C_U (§1 of the pre-registration). At C_U, θ puts less than η on composition pairs
(a gap of 0.23 to 0.27, the largest) and more on redundancy pairs (0.09 to 0.13) and on pairs with the context head,
the layer-0 head that attends to t − 2 (0.12 to 0.17). The context share is what referee (i) penalises. Trajectories of the composition,
redundancy and context pairs through training are in the JSON (`descriptive.trajectories`).

## 7. The strong t − 2 head at d_model 192 (not predicted)

* The context-head rule (layer 0, previous-token score < 0.5, t − 2 score ≥ 0.15) found one context head in most
  runs at 48 and 96, with t − 2 scores 0.16 to 0.26 and 0.26 to 0.33.
* At 192, 9 of 20 runs have a layer-0 head with a t − 2 score of 0.51 to 0.68, and 12 of 20 have two context heads
  (the 9 and 3 others). In 10 of those 12 the weaker context head scores 0.21 to 0.28; in the other two it scores
  0.17 and 0.19. One more run has two previous-token heads. These 13 runs have no role-less head and are not eligible.
* **Is this an artefact of the 0.15 cut?** The verifier reran M with other cuts (exploratory, same bootstrap).
  * d_model 192 would enter M (10 eligible) only with a cut of 0.21 or more. Every cut up to 0.20, which the pilot
    evidence allows, leaves it with 9 or fewer.
  * With cuts of 0.10, 0.20 or 0.21, or with 192 forced into the family, the outcome stays "tie": η is met only under
    referee (i) at C_U (Δ −0.059 to −0.077), and referee (ii) is never met.
  * Referee (i) does not depend on the cut except through eligibility, because it treats context and role-less
    heads alike.
* These runs still form the textbook circuit on time (phase change at 211 or 237).
* What the t − 2 head does was not measured. The advisor session's notes on the other branch describe induction
  circuits that relay through other positions; whether this head is one is open.

## 8. Exploratory: the advisor session's predictions written before the data

The advisor session (branch `claude/lucid-archimedes-av5rk7`, `notes/discussion_2026-11-16_outside_reviews.md` §7
item 7) wrote these on 16 and 17 November (commits cf3287b and 41d4d7a). The confirmatory runs started on 20
November. They are not part of the pre-registration. Checked by `scripts/analysis_v2/v2b_advisor_predictions.py`;
output `v2_analysis/v2b_advisor_predictions.json`.

| prediction | result |
|---|---|
| 1. The 1-layer controls end with repeat loss 1.56 to 2.0 (the computed losses of content-free solutions) | **not met**: 4 of 10. The others are 1.49 to 1.55, below 1.56, the lowest content-free loss the advisor computed. The prediction came from 2-head d_model 64 pilots (1.65 to 1.89); these controls have 3 heads at d_model 96 |
| 2. At d_model 96, before the phase change, the 2-layer repeat loss stays within 0.2 nats of the 1-layer median | **not met as worded**, under either reading. Up to the pre-registered phase change: 0 of 20 (largest gap 1.04 to 1.63, at step 301, where the 2-layer runs are already falling). Up to the leave point, a near miss: 17 of 20 (largest gap 0.15 to 0.22, in 19 of 20 runs at steps 57 to 81). The 2-layer runs sit below the 1-layer median from early on, not along it: by medians, 0.18 nats lower at steps 57 to 72, 0.11 at 166, 0.05 at 237 |
| 3. Antigravity: the leave loss rises or stays flat with d_model | **not met**: median 3.21 at 48, 3.01 at 96, 2.37 at 192 (ranges 3.00–3.32, 2.73–3.10, 2.20–2.59). By Antigravity's own criterion this rejects the mechanism it proposed |

* **Caveats.** The leave point depends on the 0.3-nat rule and on the grid. `loss_rep` is measured on each seed's
  own probe set, which the advisor flagged as a source of cross-seed differences. Commit dates are declared by the
  author; git shows the order of the commits, and that item 7 has not changed since 41d4d7a, but not when they were
  pushed. The verifier recomputed these numbers with its own code; they agree.
* **Prediction 1.** Six 1-layer controls end below 1.56. That figure is the lowest loss the advisor computed for one
  family of content-free predictors, over the whole data distribution; it is not a bound on every content-free
  predictor. Three explanations are open:
  * the family is narrower than what one attention layer can do without content;
  * the 1-layer models use some content;
  * probe-set variation. Each `loss_rep` is measured on that seed's own 128 probe sequences. On the ten regenerated
    probe sets, a simple content-free predictor (uniform over the distinct tokens seen so far; 2.12 over the whole
    distribution) scores 2.07 to 2.19 (SD 0.04). The shortfall below 1.56 (0.01 to 0.075) is of the same order.

  This was not examined further.

## 9. Disclosure and limits

* **Not blind.** The pilots (§0 of the pre-registration) had shown formation in 25 of 25 tiled runs, the timings,
  and the order of events. B1 and B2 were safe predictions. Their value is the confirmation on 60 new seeds.
* **Narrow setting.** One task (tiled induction), one architecture (attention-only, 2 layers, 3 heads per layer),
  one optimiser and learning rate, three sizes over a factor of 4. Nothing here shows that the timings or the
  order of events hold elsewhere.
* **Grid.** Steps are read on a grid with points about 13 % apart. Both the B1 interval and the onset leads are
  limited by it.
* **M is not gate 2.** It compares agreement with attention-score roles. Under both referees the context head is
  outside the circuit. They differ only in whether pairs of a circuit head with the context head count as
  non-circuit pairs (i) or are left out (ii), and that choice decides whether the views differ. The η AUC is at its
  ceiling in most runs under (ii).
* **The context-head cut** (t − 2 ≥ 0.15) came from 3 pilot models, where context heads scored ≥ 0.20 and other
  role-less heads ≤ 0.10. That gap does not hold at d_model 48: 7 of its 19 context heads score 0.155 to 0.176, and
  the largest role-less score there is 0.129. One run at 48 (seed 408) is ineligible only because both of its
  context heads (0.159 and 0.173) are above the cut. At 192, two second context heads are near it (0.17, 0.19). §7
  gives the verifier's check of the cut.

## 10. What follows

* By the pre-registration, Part B's questions are answered: the circuit forms reliably at every size, larger
  models change earlier, and the induction heads form before the previous-token head becomes clean. The mechanism
  comparison is a tie; whether η is favoured at C_U depends on one labelling choice (how pairs with the context head
  are counted).
* The direction after Part B is under discussion ([notes/reframing_draft_2026-11-20.md](../../notes/reframing_draft_2026-11-20.md)).
  Part B leaves four unexplained observations that can go on that note's list of candidate phenomena:
  * the previous-token head becomes clean thousands of steps after the phase change, and its timing depends much less
    on size than the phase change does (§4);
  * the strong t − 2 head that appears at d_model 192 (§7);
  * the 1-layer controls going below the computed content-free losses (§8);
  * the leave loss falling with size (§8).
