# Sky v0: first-look findings (DR0, 10 November 2026)

Data: `releases/DR0` (6,855 objects, 191 RI runs, 19,222 RI events), full calibration in `reports/calibration/`.
Run on an 88-core lab server, 70 workers, about 12 h. Analysis: `python scripts/analyze_dr0.py releases/DR0`.

## 1. Solid: the instrument works on trained networks

* **Positive controls: 49/49.** Every control object (28 sparse-parity and 21 Markov-language objects, 5–7 seeds per
  configuration) has an activation event on its planted subset at the planted order.
* **Separation from spurious detections.** A control object has a median of 3 non-planted activations. Their peak RI
  has a median of 0.0036 nats, against 0.36 nats for the planted subsets: a factor of 100.
* **Full calibration done** (preset `full`). The numbers agree with the pilot.

## 2. Promising, needs a dedicated test in v1

* **Interaction order sets onset time in the Markov language.** Median onset steps, with 7 seeds per structure:

  | coupling structure | order 2 | order 3 | order 4 |
  |---|---|---|---|
  | {1}: 1.0, {1,3}: 0.8 | 10 | 133 | — |
  | {1}: 1.0, {1,2,3}: 1.0 | 6 | — | 4,072 |
  | {1,3}: 0.8 only | — | 14 | — |

  The seed ranges of different orders never overlap.
* **A lower-order term delays a higher-order term of the same strength.** The {1,3} three-way term, at the same
  coupling 0.8, is learned at step 14–36 when it is alone but at step 96–133 when a pairwise {1} term is present.
  So staging is not just "higher order is intrinsically harder": structure that has already been learned shapes
  when the next order switches on.
* **RI leads accuracy in sparse parity.**
  * For k = 2, the planted interaction switches on 61–90 steps before test accuracy reaches 90 %, in 14 of 14 runs.
  * For k = 3, it leads in 9 of 14 runs (median lead 175–250 steps). The rest fall on the same or the next
    checkpoint, because 30 log-spaced checkpoints are too coarse there. A dense replay should settle this.
* **Parity learning time vs input dimension n.** Median steps to 90 % test accuracy grow roughly as n^1.0 for
  k = 2 and n^1.2–1.6 for k = 3. The exponent rises with k, in the direction of the n^(k−1) expectation for SGD.
  But there are only three n values (16, 24, 32), so this is a hint, not a law.
* **Grokking, as seen through hidden-unit interactions.**
  * Of 137 grokking objects followed up, 62 % have strong late activations (peak ≥ 0.02 nats) after the training
    set is fit.
  * Their onsets fall mostly before generalisation: the median onset is 0.29 × the step of 90 % test accuracy.
    This looks like interactions forming during the "hidden" phase between memorisation and generalisation.

## 3. Not yet interpretable: known problems

* **No control group for grokking.** All grokking-region follow-ups were grokking-labelled objects, because the
  label-based "grokking" transient gave them follow-up priority. Without memorisation-only and smooth objects of
  the same configurations, the late activations cannot be attributed to grokking.
* **Hidden-unit variables are noisy.** There are about 80 activations per grokking object, with peak-RI median
  0.005 nats, near the floor. Calibration E6 shows that binarised correlated units create spurious interactions.
  The transient channel therefore produced 6,791 transients, almost all from this region.
* **Training budget censors grokking.** Grokking delays cluster at 15–19k of the 20k steps. Some "memorization"
  objects (866) are probably grokking that had not happened yet. For the same reason, scaling of the delay with
  weight decay cannot be measured (R² < 0.4).
* **The biggest region was barely followed up.** parity-wide has 4,595 objects but only 5 RI follow-ups.

## 4. Implications for sky v1

1. Test "order sets onset time" directly in the Markov language:
   * vary the coupling strengths, add order 5, run ≥ 10 seeds;
   * include the "higher-order term alone vs with lower-order terms" comparison;
   * use dense checkpoints.
2. Parity: n from 16 to 64 with k = 2, 3, 4 and ≥ 10 seeds, to fit the exponent of onset vs n per order.
3. Grokking:
   * at least 50k steps;
   * RI follow-ups with matched controls (grokking, memorization and smooth at the same configuration);
   * hidden-unit thresholds set from the E6 artefact table.
4. Scheduler: follow-up quotas per region and per label. Photometric labels should not count as transients for
   priority.
