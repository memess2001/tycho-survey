# Sky v1 results: pre-registered analysis of DR1

12 November 2026. Data: release DR1 (sky v1 only; 1,223 objects, 0 failed; code 0.1.0+f977553). Criteria:
[v1_preregistration.md](v1_preregistration.md), including Amendment 1, which was written after the launch and before
anyone read v1 data. Scripts: [scripts/analysis_v1](../../scripts/analysis_v1/README.md). Numbers: the JSON files in
[v1_analysis](v1_analysis/).

## Summary

* **The instrument passes on known-answer tasks.** The planted interaction is detected in every parity run (288 of
  288) and every one-coupling Markov run (202 of 202). Two registered instrument criteria failed as written: check 2
  (216 of 288) and check 3(b) for chain-trained runs (36 of 102). Both pass under the Amendment 1 definitions
  (284 of 288; 102 of 102 and 100 of 100). The amendments were written from pilot runs before the data were read.
  Both results are given below.
* **RI and Fourier agree, as they must with uniform probes.** Check 2 is a pipeline consistency test. It cannot
  show that RI sees more than a Walsh–Fourier analysis, and v1 does not show that.
* **Parity onsets do not scale as n^(k−1).** The log-log slope of the median onset on n is 1.75 to 2.18 for k = 2, 3
  and 4. On a matched range (n = 16 to 32) there is no detectable dependence on k, and k = 4 learns about as fast as
  k = 3 at the same n. The k = 2 relation is curved, and the exponent depends strongly on the threshold that defines
  an onset. No power law or order-dependent exponent is established.
* **Markov order: not supported.** Onsets increase from order 2 to 3 to 4. Orders 4 and 5 cannot be separated on
  the 60-point grid, which Amendment 1 D anticipated. A dense replay is needed. Each order has only one lag set, so
  order is confounded with lag geometry.
* **The v0 delay is partly a context effect, partly not.** For {1,3}:0.8, a pairwise term still delays the three-way
  term with independent contexts (R_iid = 1.48 [1.38, 1.93]), and chain contexts roughly double the delay. A smaller
  target alone does not explain the iid delay. For {1,2,3}:1.0 the iid delay sits on the boundary between the two
  registered branches.
* **Chain contexts teach transient shortcuts.** Chain-trained models learn and later unlearn interactions on lag sets
  that the chain makes informative. In {1,2,3}:1.0 the shortcut (x[t−4], x̂) reaches 0.14 to 0.17 nats before the
  planted term, and the planted onset comes about twice as late as with iid contexts.
* **RI and the Walsh coefficient can disagree on the same subset.** In chain runs of {1}:1.0 with {1,2,3}:1.0, the
  coefficient f{1,2,3} is 0.12 to 0.14 while RI stays near 1e-4. Learned lower-order terms cancel the top-order
  log-linear term. This is the first case in the survey where the two measures differ substantively.
* **Grokking: no evidence that late interactions are specific to grokking seeds.** The only same-configuration
  controls are near-grokking smooth runs in 9 configurations. The difference is +7.4 pp [−10.1, +21.5]: inconclusive.
  Across the region, the measure tracks how far a run generalises.
* **Training to 50k steps lowers memorization by about 15 pp** on the same grid points (a bit-identical
  continuation of the v0 runs). Most memorization runs are still improving at 50k, so the budget still censors.
* **System checks.** Follow-up balance fails only in the first 9 of 689 requests, as the pre-registration expected.
  The anomaly flag rate is about 1 % overall (13 of 1,212), but the per-group criterion fails because of a grouping
  artefact. The "unknown" exit stayed empty.

## How the analysis was done

Seven analyses follow the pre-registration: parity instrument, parity scaling, Markov instrument, Markov order, Markov
context, grokking and system checks. Every registered item is reported literally. Where Amendment 1 changed a
definition, both results are reported and the amended one decides, as the amendment fixed in advance.

The scripts were written by AI agents (one per analysis). For each analysis a second, independent agent re-derived
every result from the pre-registration and the data before reading the first agent's code, then ran that code and
compared. All seven scripts reproduce their JSON exactly. Of 37 reported items, 34 were confirmed as written. Three
were corrected. In two of them (Q4) the numbers stand but a statement was wrong. In the third (check 9) the request
order had been reconstructed wrongly, and the verifier recovered it exactly. A final reviewer checked coverage,
consistency across the analyses and overclaims. This report uses the corrected statements. The record is in
[verification.json](v1_analysis/verification.json). The checks the report relies on beyond the seven scripts are in
`check9_request_order.py` and `supplementary.py`. The scripts were rerun from the repository and reproduce the
JSON. Check 10 asks that each anomaly be looked at by hand. That was done by an agent and still needs a human look.

Conventions. An RI onset is the step of an activation event: the first of two consecutive active checkpoints, so the
true onset lies in (previous checkpoint, step]. Checkpoints are 60 log-spaced steps. Medians are over seeds.
Bootstraps use 2,000 resamples with fixed seeds; the resampled unit is named per item. No run in any region was
censored at its step budget.

## Scorecard

| # | Item | Registered result | Amendment 1 result | Verdict (deciding) |
|---|---|---|---|---|
| 1 | Parity: planted set activates in ≥ 95 % of runs reaching 90 % test accuracy | 288 / 288 | not changed | **pass** |
| 2 | Parity: RI and Fourier onsets within one checkpoint in ≥ 90 % of runs | 216 / 288 (75.0 %): fail | A, gated Fourier onset: 284 / 288 (98.6 %) | **pass** (amended) |
| 3 | Markov: (a) planted subset activates in ≥ 90 % of converged runs; (b) non-planted peak ≥ 10 × min_effect in ≤ 5 % | (a) 202 / 202; (b) chain 36 / 102, iid 0 / 100: fail | B, end of training: chain 102 / 102, iid 100 / 100 | **pass** (amended) |
| Q4 | Parity: onset exponent per k, reference n^(k−1) | slopes 1.75 to 2.18 | C changes 0 of 288 onsets | estimate: far from n^(k−1); no k-dependence detectable |
| Q5 | Markov: onset ranges disjoint for orders 2 < 3 < 4 < 5 at β = 1.0, both contexts | orders 4 and 5 overlap | C: orders 4–5 and 2–3 overlap; orders 3–5: 4–5 overlap | **not supported** |
| Q6 | Markov: R_iid ≈ 1 (context effect) or R_iid > 1 (lower order delays higher order) | {1,3}: R_iid 2.97; {1,2,3}: 1.47 | E25: {1,3} 1.48 [1.38, 1.93]; {1,2,3} 1.29 [1.000, 1.479] | {1,3}: **R_iid > 1**. {1,2,3}: boundary |
| Q7 | Grokking: late strong interactions more frequent in grokking seeds (CI above 0) | +7.4 pp [−10.1, +21.5]: not supported | F: 9 configurations, near-grokking controls; F.2 has 0 pairs | **inconclusive** (amended) |
| Q8 | Grokking: memorization at 50k vs v0 at 20k | 236 / 689 (34.3 %) vs 866 / 2,166 | G, same grid points: 66.5 % → 51.6 %, −14.8 pp [−25.9, −4.9] | estimate |
| 9 | No label > 50 % of follow-ups while others have candidates | 8 of 689 request prefixes violate (i ≤ 9) | H: reported literally; expected | **fail** (as anticipated) |
| 10 | Anomalies about 1 % per (region, label) group | as run: 0 anomaly follow-ups | I, offline: markov-v1/smooth 12 / 246; overall 13 / 1,212 | **fail** (grouping artefact) |

## 1. Instrument checks

**Check 1 (pass).** All 288 parity runs reached 100 % test accuracy, and every one activates the planted set
{x_S, ŷ}: parity-k2 100 of 100, k3 120 of 120, k4 68 of 68. Each run has exactly one event on the planted set, an
activation that never switches off. Because every run learned, the check cannot test runs that fail to learn.

**Check 2 (literal fail, amended pass).** The registered Fourier onset crosses a bare threshold |f| ≥ 0.0447
(RI = min_effect). The RI event also needs a BH q-value ≤ 0.01 and two consecutive checkpoints. Under the registered
definition 216 of 288 runs agree within one checkpoint. Every disagreement is k = 2, where the coefficient grows
gradually and RI trails the bare threshold by 2 to 5 checkpoints (parity-k2 28 of 100; k3 120 of 120; k4 68 of 68).
Amendment 1 A gives the Fourier onset the same gate. Then 284 of 288 agree (98.6 %; bootstrap over configurations
96.7 to 100 %). In the 4 remaining runs RI fires 2 to 3 checkpoints earlier, at n = 16 or 32. In three of them the
stored BH q-value is below the gate's conservative Bonferroni stand-in m·p. In the fourth the single-mode value R(f)
dips once.

With uniform probes RI on a subset is a function of the Walsh coefficients on its sub-subsets, so check 2 tests the
pipeline's consistency, not the value of RI. The pass also depends on the event indexing: with the onset taken as
the checkpoint before the event, 258 of 288 agree.

**Check 3 (literal fail, amended pass).** All 202 one-coupling runs converge (final kl_true < 0.01, largest 0.0027
nats). The planted subset activates in all of them: chain 102 of 102, iid 100 of 100. The registered criterion (b)
fails for chain training: a non-planted subset peaks at ≥ 0.01 nats in 36 of 102 chain runs and 0 of 100 iid runs.
35 of the 36 are interactions that the chain teaches transiently (next item). The other one crosses only through the
untrained network's offset (0.0101 nats at step 2; the bar is 0.01). Amendment 1 B judges the end of training: over
the last five checkpoints the largest non-planted RI is ≤ 0.01 in 102 of 102 chain and 100 of 100 iid runs (largest
0.0016). Check 3 passes.

**Result: chain contexts teach transient shortcuts.** With one planted coupling L, the chain makes e_t = s_t·χ_L(t)
i.i.d. with mean tanh β, so a lag set A that is a product of k shifted copies of the pattern has
E[s_t χ_A] = tanh(β)^k. This was checked analytically (GF(2) division) and against the exact 64-state stationary
distribution. Every chain-induced transient sits on such a set, and the iid twins (same initial weights and probes)
show none of them (≤ 0.0072 nats, the largest values being the shared step-0 offset).
* {1,2,3}:1.0. The model first learns the single-lag shortcut (x[t−4], x̂): 0.141 to 0.171 nats, 79 to 96 % of
  the RI of the Bayes-optimal predictor that uses only x[t−4]. It peaks at steps 74 to 162, before the planted onset
  in 12 of 12 runs, and is gone by steps 2,206 to 6,267. The planted onset comes at steps 162 to 273 with chain
  contexts against 96 to 125 with iid contexts (ratio 1.92 [1.48, 2.84]).
* Other sets at β = 1.0 ({1}, {1,3}, {1,2,4}, {1,2,3,5}). The transient grows alongside or after the planted
  term, reaches 0.011 to 0.024 nats, and vanishes by step 3,718. RI onsets show no detectable chain delay. Onsets
  defined on the coefficient (50 % of its population value) are about one checkpoint later (1.3 to 1.5 times).
* At β = 0.5 and 0.8 no chain-induced transient reaches 0.01 nats.

The delay is attributed to the shortcut by association across coupling sets, not by an intervention: chain and iid
training differ in the whole context distribution.

## 2. Parity: how does the onset scale with n? (Q4)

<!-- FIG:parity -->

Onset = the RI activation of the planted set. Amendment 1 C changes none of the 288 onsets (none is left-censored or
has the wrong sign). Slopes of ln(median onset) on ln n, with 95 % bootstrap intervals over seeds:

| k | n range | lr 0.03 | lr 0.1 | reference |
|---|---|---|---|---|
| 2 | 16–64 | 1.88 [1.70, 1.92] | 1.75 [1.62, 1.88] | 1 |
| 3 | 16–64 | 1.92 [1.68, 2.10] | 1.80 [1.54, 1.93] | 2 |
| 4 | 16–32 | 2.18 [1.91, 2.41] | 2.18 [1.91, 2.38] | 3 |
| all | 16–32 (matched) | k2 2.17, k3 2.01, k4 2.18 | k2 1.96, k3 1.99, k4 2.18 | |

* **No detectable k-dependence.** The higher k = 4 value comes from its smaller n range. On the matched range every
  difference between k has a CI that includes 0.
* **Far from n^(k−1)** for k = 2 (CIs exclude 1) and k = 4 (CIs exclude 3). The reference 2 lies inside the k = 3
  interval at lr 0.03 only.
* **k = 4 is about as fast as k = 3.** RI onset at lr 0.1 for n = 16, 24, 32: 418, 1,033, 1,888 steps (k = 4)
  against 458, 928.5, 1,845 (k = 3). Steps to 90 % test accuracy: 611, 1,201, 2,195 against 606, 1,057, 1,845.
* **Not a single power law for k = 2.** The local slope falls from 1.96–2.17 (n = 16 to 32) to 1.46–1.62 (n = 32
  to 64); the difference CIs exclude 0 at both learning rates.
* **The exponent depends on the threshold.** The onset is a low-threshold crossing (|f| about 0.06 to 0.08). For
  k = 2 the exponent falls from 1.71–1.75 at |f| ≥ 0.0447 to 0.35–0.36 at |f| ≥ 0.9; for k = 3 from 1.86–2.00 to
  1.24–1.53.
* **Fourier gives the same exponents.** With the Amendment A gate, RI and Fourier exponents are identical for k = 3
  and 4. For k = 2 the RI exponent is 0.04 (lr 0.03) and 0.17 (lr 0.1) larger, from the BH q-value against the
  gate's Bonferroni stand-in.
* **Steps to 90 % test accuracy** grow more slowly (slopes: k = 2 1.23 / 0.94; k = 3 1.80 / 1.52; k = 4 1.75 / 1.83
  at lr 0.03 / 0.1). At width 512, lr 0.1, n = 16 to 32 this reproduces v0 (k = 2: 1.26; k = 3: 1.59).
* **The RI onset never comes after the 90 % accuracy crossing** on the RI grid (0 of 288). For k = 2 it leads by 2
  to 15 checkpoints. For k ≥ 3 the coefficient jumps, and the lead is about one checkpoint: a low threshold on the
  same coefficient is crossed before a high one.

## 3. Markov: does the order set the onset? (Q5)

<!-- FIG:q5 -->

160 runs: one planted coupling of order 2 to 5, two strengths, two contexts, 10 seeds. None is censored.
Median onsets at β = 1.0 (Amendment C onset; registered onset differs only for order 2):

| context | order 2 ({1}) | order 3 ({1,3}) | order 4 ({1,2,4}) | order 5 ({1,2,3,5}) |
|---|---|---|---|---|
| chain | 5.5 (2–15) | 21.5 (9–30) | 110 (96–125) | 110 (96–162) |
| iid | 4.5 (1–12) | 18 (9–30) | 110 (96–110) | 110 (96–125) |

* **Not supported, under every reading** (registered and amended onset, orders 2–5 and 3–5). The decision rests on
  orders 4 and 5: equal medians and overlapping ranges in both contexts.
* Order 3 and order 4 are well separated (order-3 maximum 30, order-4 minimum 96).
* Under the amended onset orders 2 and 3 also overlap. 7 of 40 order-2 runs start with the planted pair active
  with the wrong sign at step 0 (random initial weights) and reach the amended onset only at steps 10 to 15. Reported
  at the ≤ 8-step floor of Amendment C, order 2 would be disjoint from order 3.
* Orders 4 and 5 share checkpoints 96, 110 and 125, where the grid steps by a factor 1.14 to 1.30. In iid runs no
  order-5 seed is more than one checkpoint earlier than an order-4 seed. In chain runs 6 of 100 seed pairs are 2
  checkpoints inverted. Between checkpoints an exploratory interpolation of the coefficient puts order 5 after order
  4 in 9 of 10 same-seed iid pairs, by a median factor of 1.11, less than one grid step.
* With a coefficient-based onset (25 % of the population value, the Q6 definition, not registered for Q5) iid medians
  increase strictly: 16.5 < 65 < 211 < 273, and only orders 4 and 5 overlap.
* Each order has one lag set (maximum lag 1, 3, 4, 5), so order is confounded with which lags are planted.
* At β = 0.5 the pattern is the same and later for orders 3 to 5 (10 of 10 paired seeds in each order).

## 4. Markov: is the v0 delay a context effect? (Q6)

<!-- FIG:q6 -->

R = median onset with the pairwise term {1}:1.0 / median onset alone. Amendment 1 E decides: the onset is the planted
coefficient reaching 25 % of its population value (E25), because the pairwise term lowers the population coefficient
of {1,3} from 0.664 to 0.375. The registered RI onset and 10 % and 50 % levels are reported next to it.

| comparison | definition | R_chain | R_iid | R_chain / R_iid |
|---|---|---|---|---|
| {1,3}:0.8 (the v0 delay) | registered | 5.5 [3.62, 6.53] | 2.97 [1.92, 3.39] | 1.85 [1.23, 2.85] |
| | **E25 (decides)** | 2.66 [2.18, 2.85] | **1.48 [1.38, 1.93]** | 1.80 [1.29, 2.06] |
| {1,2,3}:1.0 | registered | 18.9 [13.6, 23.0] | 1.47 [1.30, 1.69] | 12.8 [8.66, 17.6] |
| | **E25 (decides)** | 3.04 [2.49, 4.79] | **1.29 [1.000, 1.479]** | 2.35 [1.91, 4.01] |

* **{1,3}:0.8: R_iid > 1 (registered branch 2).** With independent contexts the pairwise term still delays the
  three-way term, under all 16 onset definitions tried. Chain contexts add a factor of about 1.8. v1 chain runs
  reproduce v0 (R_v0 = 5.1 [2.7, 9.5]).
* **The iid delay is not a smaller-target effect** (supplementary check, not registered). Alone, {1,3} reaches 25 %
  of its population value at a median of 61, 65 and 65 steps at β = 0.5, 0.8 and 1.0 (population coefficients 0.46
  to 0.76): flat in target size. With {1}:1.0 it takes 96 steps (range 96 to 110, disjoint from all three). The 50 %
  level gives the same picture. The caveat is that 0.375 is below the smallest coefficient tested alone.
* **{1,2,3}:1.0: between the branches.** Under E25 the CI of R_iid touches 1 only because 13 % of resamples put both
  medians on the same checkpoint; none falls below 1. The registered onset, the 50 % level and the interpolated
  crossing all put the CI above 1. The size of the iid delay (1.07 to 1.47 across definitions) equals what halving
  the coupling does to the order-4 set {1,2,4} (25 % level: 211 → 273 steps), although {1,2,3,5} shows no such effect.
* **RI and the Walsh coefficient diverge in one arm.** In chain runs of {1}:1.0 with {1,2,3}:1.0 the RI activation
  comes 1 to 9 checkpoints after the coefficient's 25 % crossing (12 of 12 runs); in every other arm it comes before.
  The chain-trained model carries lower-order coefficients f{2} ≈ −0.16 and f{3} ≈ +0.17. Together with f{1} they
  cancel the top-order log-linear term, so RI stays near 1e-4 while f{1,2,3} is 0.12 to 0.14. Rebuilding RI from the
  stored coefficients reproduces this; with f{2} = f{3} = 0 the rebuilt RI is 0.020 nats. This is why the registered
  R_chain for {1,2,3} is 18.9.

## 5. Grokking (Q7, Q8)

All 689 grokking-v1 objects (p = 47, 50k steps) were followed up with a bit-identical replay and RI on 10 binarised
hidden units. "Strong late interaction" = an activation or reactivation with peak ≥ 0.02 nats after the training set
is fitted (99 % train accuracy).

<!-- FIG:q7 -->

**Q7 (inconclusive).** Seeds disagree on the label at 30 configurations, and grokking meets another label at 9 of
them (all width 256). Every control there is a smooth run that stopped just short of 90 % test accuracy (final 0.81
to 0.90). No memorization seed shares a configuration with a grokking seed.

| comparison | grokking | controls | difference [95 % CI] |
|---|---|---|---|
| same configuration (9 configurations, 63 runs) | 23 / 32 (71.9 %) | 20 / 31 (64.5 %) | +7.4 pp [−10.1, +21.5] |
| all 30 disagreeing configurations (registered wording; confounded) | 23 / 32 | 93 / 176 | +19.0 pp, CI lower bound −1 to −4 pp |
| neighbouring memorization point (F.2) | no memorization point is one knob step from a grokking point | | not estimable |

The literal rule calls this "not supported"; Amendment F.3 calls it inconclusive. Either way v1 gives no evidence
that late interactions are specific to grokking seeds. The events per run are also similar (1.31 vs 1.39). Across
the region the measure rises with final test accuracy: from 27 % of runs below 0.05 to 84 % at 0.9 or above
(flat at 56 to 57 % between 0.22 and 0.8). The untrained hidden layer is already near the bar: 479 strong order-2 activations are
present at step 0 in 283 of 689 runs (median 0.013 nats). Only 1 of 689 replays received the planned dense window
around the largest loss drop, so the timing is resolved only on the 60-point grid.

**Q8 (estimate).** The registered comparison, 236 of 689 (34.3 %) against "866 of 2,166", mixes two counts; v0's
grokking region had 742 of 2,166 (34.3 %), on a different grid. On the 31 grid points that v0 also ran, memorization
falls from 66.5 % (v0, 20k steps) to 51.6 % (v1, 50k): −14.8 pp [−25.9, −4.9] resampling points. It is lower at 8
points, equal at 23 and higher at none.
* The comparison is a continuation. v1 runs are bit-identical to their v0 twins at all 24 shared evaluation steps
  up to step 17,949 (215 of 215 pairs).
* Of the 93 v0 memorization runs, 70 are still memorization at 50k, 17 became smooth and 6 grokked, all late
  (steps 39,744 to 50,000; 3 exactly at the last evaluation).
* **50k still censors.** Every memorization run has lower test loss at 50k than at 24k (236 of 236). Test accuracy
  rose by more than 0.05 over that span in 126 of 236 runs, and by more than 0.02 over the last 10k steps in 123.
  The −15 pp is a lower bound on how much of v0's memorization was censoring.

## 6. System checks

**Check 9 (fail, as anticipated).** DR1 has no follow-up request table. The request order was recovered exactly by
replaying the scheduler's own stream code on the catalog state at each call: the replay reproduces all 689 picks and
their reason strings. Over that order 8 of 689 prefixes violate the rule (i = 1 to 5 and 7 to 9, all in the first
call). The first three requests were memorization objects from the label-blind matched, boundary and control streams.
After request 9 no prefix violates. At every call end the largest cumulative share is at most 47.9 % (smooth, at the
end). A rule applied to every prefix fails at i = 1 for any scheduler, so the criterion needs a minimum prefix size.
Per call, smooth took 51 to 57 % of the picks in calls 7 to 9, after grokking candidates ran out.

**Check 10 (fail).** As run, the scheduler scored only grokking-v1 and flagged nothing, so there were 0 anomaly
follow-ups. Scored offline for every (region, label) group, two groups have flags. markov-v1/smooth has 12 of 246
(4.9 %), and these are all 12 seeds of one configuration ({1}:1.0 with {1,2,3}:1.0, chain). It is a grouping artefact:
the group pools 24 task regimes, and those are the only Markov runs that reach the rescaled accuracy threshold, on a
feature with MAD 0. parity-k4/smooth has 1 of 33, an early loss spike. Overall the rate is 13 of 1,212 = 1.07 %. With
33 to 330 objects per group the per-group criterion has little power. No object was labelled unknown, and the anomaly
exit produced no follow-up.

## 7. Deviations from the plan

* **Refinement was much larger than planned:** 543 refined objects instead of about 150 (grokking-v1 509, parity-k3
  20, parity-k4 8, markov-v1 6). The 6 Markov objects are seeds 10 and 11 in three cells chosen by the scheduler
  after outcomes were known. They enter Q6 and check 3. With seeds 0 to 9 only, no verdict changes (for {1,2,3} the
  E25 R_iid stays 1.29; R_chain moves from 3.04 to 3.24).
* **Dense follow-up windows** (30 extra points around the largest loss drop) were added in 1 of 689 replays, because
  only one grokking object had a detected drop.
* **Run time:** summed training time was 44.1 h (grokking-v1 30.0 h) against the estimated ≈ 30 CPU hours. Wall
  clock matched the estimate: the last object finished 0.65 h after the first was enqueued, and the last follow-up was
  recorded at 1.71 h. Memory (Amendment K) is not recorded in DR1.
* **Not done:** a systematic comparison of RI with the single-mode R(f) over all stored subsets (Amendment J). The
  list of failed follow-ups that Amendment F asks for cannot be checked from DR1, which has no request table; all 689
  follow-ups are present and bit-identical.

## 8. Errata to earlier documents

* [v0_findings.md](v0_findings.md), §2 table: the median onset 14 for {1,3}:0.8 alone is the median first
  activation of any order-3 subset in those runs. For the planted subset it is 26 (range 14 to 36), so the v0 delay
  ratio is 133 / 26 = 5.1, not 9.5.
* `configs/sky_v1.yaml`, comment: the v0 k = 3 medians at width 512 are 642 / 1,085 / 1,832 steps, not
  642 / 911 / 1,832. The config is kept as it was run.
* `v1_analysis/system_checks.json`, check 9: the "36 violating prefixes in the reconstructed request order" is
  superseded by `check9_request_order.json` (8 prefixes). `parity_scaling.json`, Q4: the statements that the exponent
  "rises weakly with k" and that onsets "grow as a power of n" are superseded by the matched-range result above. The
  corrected statements are in `verification.json`.

## 9. What v1 does not show, and what comes next

v1 does not establish RI's added value over Fourier analysis, an order effect beyond order 4, a causal role of
lower-order structure or of context correlation, or a grokking-specific mechanism. The open questions, and what they
need:

| question | stored data (DR1) | new runs |
|---|---|---|
| When do RI (a top-order log-linear, θ-type quantity) and the Walsh coefficient (a moment, η-type) disagree? Map RI against R(f) and against RI rebuilt from coefficients, over all stored subsets. | yes, on subsets RI kept | |
| Parity: proper subsets of the planted set, which the target lacks, activate in 26 of 100 (k = 2), 56 of 120 (k = 3) and 50 of 68 (k = 4) runs, up to 0.055 nats, often before the planted onset. What are they? | yes | |
| Why is k = 4 as fast as k = 3, and is n^(k−1) the right reference at fixed lr, width and batch? | describe only | an lr ∝ 1/n arm, k = 5 |
| Is R_iid > 1 a smaller-target effect? Alone runs at matched population coefficients ({1,3} at 0.375; {1,2,3} at 0.482). | partly (above) | yes |
| Order 4 vs order 5 | | dense replay of stored objects (compute only); a second lag set per order |
| Does the chain shortcut cause the {1,2,3} delay? iid contexts with the s_t·s_{t−4} correlation injected. | | yes |
| Grokking specificity: within-configuration dose-response against final test accuracy; per-run null at initialisation; dense window around the test-accuracy rise. | first two | dense re-replay |
| How many 50k memorization runs would generalise with more steps? | extrapolation | longer runs |
| Re-specify checks 9 and 10 (minimum prefix size; anomaly groups by configuration). | yes | |

## Reproducing

See [scripts/analysis_v1/README.md](../../scripts/analysis_v1/README.md). Each script reads `releases/DR1` and the
extracted archives, checks that `tycho` is imported from the checkout, and writes its JSON; reruns reproduce the
committed files exactly.
