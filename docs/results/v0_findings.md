# Sky v0: first-look findings (DR0, 10 November 2026)

Data: `releases/DR0` (6,855 objects, 191 RI runs, 19,222 RI events), full calibration in `reports/calibration/`.
Run on an 88-core lab server with 70 workers: the 6,855 training runs took 46 min; the 142 RI follow-up replays (2 every 10 min) ran for about 12 h. Analysis: `python scripts/analyze_dr0.py releases/DR0`.

> **Correction (same day, after re-reading the 4 November 1-on-1 report).** Every control in v0 probes the network
> with independent, uniform binary inputs and a soft binary output. In that regime the marginal RI of {x_S, ŷ} is a
> function of the Fourier coefficients of the model on the subsets of S only. For a single mode it is
> ln 2 − h((1 + ε)/2), and it is unchanged when coefficients outside S are added (checked numerically). This is
> slide 3 of that report. So the controls verify the instrument at scale but cannot show that RI sees more than
> Fourier analysis. "RI switches on before accuracy" is a low threshold (≈ ε = 0.045) being crossed before a high
> one, as in Barak et al.'s hidden progress, not evidence. The Markov-language "delay" has a confound; see §2.

## 1. Solid: the instrument works on trained networks

* **Positive controls: 49/49.** Every control object (28 sparse-parity and 21 Markov-language objects, 5–7 seeds per
  configuration) has an activation event on its planted subset at the planted order.
* **Separation from spurious detections.** A control object has a median of 3 non-planted activations. Their peak RI
  has a median of 0.0036 nats, against 0.36 nats for the planted subsets: a factor of 100.
* **Full calibration done** (preset `full`): noise floor 8.1–8.3e-4 (theory 8.1e-4), false positives 0.92 % at α = 0.01,
  minimum detectable RI 1.4e-3 (theory 1.4e-3). The binarisation artefact (E6) is the known dichotomised-Gaussian
  effect (Amari et al. 2003; Macke et al. 2009).

## 2. Promising, needs a dedicated test in v1

* **Interaction order sets onset time in the Markov language.** Median onset steps, with 7 seeds per structure:

  | coupling structure | order 2 | order 3 | order 4 |
  |---|---|---|---|
  | {1}: 1.0, {1,3}: 0.8 | 10 | 133 | — |
  | {1}: 1.0, {1,2,3}: 1.0 | 6 | — | 4,072 |
  | {1,3}: 0.8 only | — | 14 | — |

  The seed ranges of different orders never overlap.
* **A lower-order term seems to delay a higher-order term of the same strength, but this is confounded.** The
  {1,3} three-way term, at the same coupling 0.8, is learned at step 14–36 alone but at step 96–133 when a pairwise
  {1} term is present. There are two confounds:
  * The pairwise term makes the training contexts correlated. In the chain, s(t−1)·s(t−3) = +1 in 85 % of contexts
    (E = +0.71), against 50 % without the term. The three-way term is therefore exercised far less often in training.
  * The final marginal RI of the three-way term differs, 0.121 vs 0.240 nats, and onsets are threshold crossings.

  The effect is therefore at least partly a correlated-inputs effect, which is the open question of the PhD plan.
  It is not yet evidence that "learned lower-order structure delays higher order".
* **RI leads accuracy in sparse parity.** This is expected rather than evidence (see the correction above): it
  amounts to a low threshold on the Fourier coefficient being crossed before a high one.
  * For k = 2, the planted interaction switches on 61–90 steps before test accuracy reaches 90 %, in 14 of 14 runs.
  * For k = 3, it leads in 9 of 14 runs (median lead 175–250 steps). The rest fall on the same or the next
    checkpoint, because 30 log-spaced checkpoints are too coarse there. A dense replay should settle this.
* **Parity learning time vs input dimension n.** Median steps to 90 % test accuracy grow roughly as n^1.0 for
  k = 2 and n^1.2–1.6 for k = 3. The exponent rises with k, in the direction of the n^(k−1) expectation for SGD.
  But there are only three n values (16, 24, 32), so this is a hint, not a law.
* **Grokking, as seen through hidden-unit interactions.**
  * Of 137 grokking objects followed up, 62 % have strong late activations (peak ≥ 0.02 nats) after the training
    set is fit, and 59 % have one before generalisation.
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
  objects (742 in the grokking region; 866 in all regions) are probably grokking that had not happened yet. For the same reason, scaling of the delay with
  weight decay cannot be measured (R² < 0.4).
* **The biggest region was barely followed up.** parity-wide has 4,595 objects but only 5 RI follow-ups.

## 4. How the survey relates to the 4 November plan

The survey is the research program. The 4 November plan was provisional and covers only its first two stages:
the instrument (RI code; open questions on ordering and comparing related variables) and the choice of targets
(the lesson that on simple inputs RI adds nothing new over standard tools, and the candidate targets B, in-context
Markov chains, and C, hidden features). This week built and ran all five stages once: instrument with calibration,
targets (sky v0), observation (6,855 runs), catalog (DR0), and first hints of patterns, none RI-specific yet.
Next: return to stage 2 with that lesson, and aim the survey where RI can add something (B and C).

## 5. Implications for sky v1

All four are implemented in sky v1; see [v1_preregistration.md](v1_preregistration.md).

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
