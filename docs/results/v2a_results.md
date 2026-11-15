# Sky v2, Part A: results

Pre-registered analysis of sky v2 Part A, run on the server on 15 November 2026. Output:
`v2_analysis/v2a_analysis.json`, code at commit 4ca774f. Data release: `releases/DR2a`.

**Verified independently.** A second agent recomputed every confirmatory number from the raw data release with its
own code (record: [v2_analysis/v2a_results_verification.md](v2_analysis/v2a_results_verification.md)). The decision
and all 12 comparison labels were reproduced. Its corrections to the wording of this report are applied.

Pre-registration: [v2a_preregistration.md](v2a_preregistration.md), binding, with its amendments. Every rule below
was fixed before any confirmatory data existed: the measures, referees, windows, bootstrap, the 0.10 margin, Holm
over 12 comparisons, and the three win conditions.

## 1. Summary

* **Decision by the pre-registered rule: "η wins".** η wins 1 of the 12 comparisons (T2 at C_U); θ wins none.
  * Four comparisons are value-dependent: they pass with one η value and not the other, so they do not count.
  * Seven are ties.
* **What the numbers mean.** On these behavioural referees, the two views are largely interchangeable.
  * Within 0.03 (largest 0.031), at every reference: the correlations of θ and η with behaviour agree in
    * 8 of the 16 Markov chain configurations;
    * parity k = 3 at ρ_train = 0.5;
    * the four iid twin sets;
    * the chain twin set "1:1.0 vs 1:1.0;1,3:0.8".
  * Gaps above 0.5 occur in six configurations, all with a largest |m| of at least 0.98:
    * "1:2.0", "1:1.0;1,2,3:1.0" and "1:2.0;1,2,3:2.0" (chain);
    * parity k = 2 at ρ_train = 0.8;
    * the two chain twin sets that add {1, 2, 3}.

    Saturation is necessary for a large gap but not sufficient. For example, parity k = 3 at ρ_train = 0.5 has a
    largest |m| of 0.998 and a gap of 0.014.
  * At the reference of each largest gap, θ tracks behaviour worse than bounded η in all six configurations, and
    worse than Gaussian η in five. The exception is parity k = 2 at ρ_train = 0.8, at C_D: θ +0.40, bounded
    η +0.78, Gaussian η −0.19.
  * At other references θ is sometimes ahead. For example, T1 "1:2.0;1,2,3:2.0" at M_D: θ +0.92 against η +0.29
    and +0.71.
* **The one win is weak.** T2 at C_U, Δ −0.16.
  * About 0.14 of the Δ comes from one twin set, "1:2.0 vs 1:2.0;1,2,3:2.0" (chain). Without that set, Δ is −0.027
    and −0.022, a tie.
  * In that twin set, θ's predictor at C_U stays below 1e-4 nats at every window point, far below the 0.001 presence
    floor. Δη reaches 0.019. ρ_θ = −0.49 therefore ranks values that θ itself would call absent.
  * Under D_with the added term is close to unidentifiable in that twin set: the rule of §4 excludes it at M_D and
    C_D, with residual variances 7e-6 and 2e-6. At C_U, with uniform weights, it enters.
  * The pooled secondary statistic for T2 at C_U is about 0.
* **Gate 2 is not passed.** On these referees, RI does not show a sensitivity to behaviour that the mean view
  lacks. By the decision table, RI's role in the next sky is attribution and description. It is not the primary
  detector.
* **Instrument.** Every bug check passed (IC-1 to IC-4), and all 440 runs met the IC-5 precondition. Of 13.0
  million logistic fits, none needed the fallback added after the analysis-code verification.

## 2. The 12 comparisons

Δ = ρ̄_θ − ρ̄_η; a negative Δ favours η. A link wins a comparison only if all three conditions hold:
* |Δ| ≥ 0.10 in the link's favour, with both η values;
* Holm p < 0.05, with both η values;
* the same sign with the squared-error outcome.

| comparison | ρ̄_θ | ρ̄_η (bounded) | ρ̄_η (Gaussian) | Δ bounded | Δ Gaussian | Δ squared-error outcome (bounded η) | result |
|---|---|---|---|---|---|---|---|
| T1 Markov M_D | −0.034 | −0.060 | −0.048 | +0.026 | +0.014 | +0.025 | tie |
| T1 Markov M_U | +0.511 | +0.547 | +0.564 | −0.036 | −0.053 | −0.041 | tie |
| T1 Markov C_D | +0.408 | +0.487 | +0.482 | −0.079 | −0.074 | −0.081 | tie |
| T1 Markov C_U | +0.424 | +0.547 | +0.514 | **−0.123** | −0.090 | −0.122 | value-dependent (η) |
| T1 parity M_D | −0.627 | −0.623 | −0.635 | −0.004 | +0.008 | −0.007 | tie |
| T1 parity M_U | +0.591 | +0.556 | +0.584 | +0.035 | +0.007 | −0.010 | tie |
| T1 parity C_D | +0.620 | +0.814 | +0.363 | **−0.193** | **+0.258** | −0.242 | value-dependent (both links) |
| T1 parity C_U | +0.474 | +0.556 | +0.346 | −0.082 | **+0.128** | −0.143 | value-dependent (θ) |
| T2 M_D | +0.754 | +0.835 | +0.839 | −0.081 | −0.085 | −0.040 | tie |
| T2 M_U | +0.702 | +0.807 | +0.797 | **−0.105** | −0.095 | −0.048 | value-dependent (η) |
| T2 C_D | +0.883 | +0.897 | +0.895 | −0.015 | −0.013 | −0.037 | tie |
| T2 C_U | +0.646 | +0.807 | +0.801 | **−0.161** | **−0.155** | −0.140 | **η wins** |

* **Bootstrap.**
  * The SDs of Δ over seeds are 0.001 to 0.017.
  * Raw p-values are at or next to the floor of 10,000 resamples: 0.0002 in 20 of the 24 tests (12 comparisons ×
    2 η values) and 0.0004 in two.
  * The Holm-adjusted p is therefore 0.0024 in 22 of the 24. Both exceptions use the Gaussian value: T1 Markov M_D
    (raw 0.0040, Holm 0.0080) and T1 parity M_U (0.33).
  * The 0.10 margin decides every comparison, as the review predicted.
* **Inference is about these configurations.** Only seeds were resampled.
* **Secondary statistics.**
  * For the comparisons that favour η, the secondaries agree in direction, though they are small for T2.
    * T1 Markov C_U: pooled −0.11 and −0.14; seed-level −0.08 and −0.08.
    * T2 C_U: chain twin sets only −0.30 and −0.30; pooled about 0 (−0.004 and +0.005); seed-level −0.02.
  * For the parity comparisons that favour θ with the Gaussian value, they do not agree.
    * T1 parity C_D: +0.26 against seed-level −0.16.
    * T1 parity C_U: +0.13 against seed-level −0.15.

## 3. Where the differences come from

Each configuration's largest gap between ρ_θ and either ρ_η, over the references at which it enters, sorts the
configurations into three groups.

* **At most 0.031.**
  * Markov chain: "1,2,3,5:0.5", "1,2,3,5:1.0", "1,3:0.5", "1,2,4:0.5", "1,3:0.8", "1,3:1.0", "1,2,4:1.0" and
    "1:0.5" (8 of 16 configurations).
  * Parity k = 3 at ρ_train = 0.5.
  * The four iid twin sets, and the chain twin set "1:1.0 vs 1:1.0;1,3:0.8".
* **0.08 to 0.33.**
  * Markov chain: "1,2,3:1.0", "1:1.0", "1:2.0;1,3:1.6", "1,2,3:2.0" and "1:1.0;1,3:0.8".
  * Parity k = 2 at ρ_train = 0.5, and k = 3 at 0.8.
  * The chain twin set "1:2.0 vs 1:2.0;1,3:1.6".
* **Above 0.5.**

  | configuration | largest gap |
  |---|---|
  | "1:2.0" (chain) | 0.51 |
  | parity k = 2, ρ_train = 0.8 | 0.59 |
  | twin set "1:1.0 vs 1:1.0;1,2,3:1.0" (chain) | 0.64, at M_D |
  | "1:1.0;1,2,3:1.0" (chain) | 0.87 |
  | "1:2.0;1,2,3:2.0" (chain) | 0.97 |
  | twin set "1:2.0 vs 1:2.0;1,2,3:2.0" (chain) | 1.10, at C_U; its 1.13 at C_D does not count, because the set is excluded there |

All six configurations with a gap above 0.5 have a largest |m| of at least 0.98. But saturation alone does not
produce a gap: "1:2.0;1,3:1.6" (chain) has 99 % of D's mass at |m| ≥ 0.99 and a gap of 0.20.

**Examples at each configuration's reference of largest gap:**

| configuration | reference | ρ_θ | ρ_η bounded | ρ_η Gaussian |
|---|---|---|---|---|
| T1 Markov "1:1.0;1,2,3:1.0" (chain) | C_U | −0.10 | +0.58 | +0.78 |
| T1 Markov "1:2.0;1,2,3:2.0" (chain) | C_U | −0.01 | +0.72 | +0.95 |
| T1 Markov "1:2.0" (chain) | C_D | +0.24 | +0.74 | +0.36 |
| T1 parity k = 2, ρ_train = 0.8 | C_D | +0.40 | +0.78 | −0.19 |
| T2 "1:1.0 vs 1:1.0;1,2,3:1.0" (chain) | M_D | +0.05 | +0.61 | +0.69 |
| T2 "1:2.0 vs 1:2.0;1,2,3:2.0" (chain) | C_U | −0.49 | +0.61 | +0.60 |

**The {1, 2, 3} configurations** are the kind that v1 and A3 singled out. "1:2.0" and the parity configurations are
not, and Q2 found no large lower-order coefficients in trained-network divergences.

* **"1:2.0;1,2,3:2.0".** The chain-trained network's θ on {1, 2, 3} stays near 0 at every reference: the median at
  the end is below 1e-4 nats, while the teacher's θ at M_U is 0.21. η sees a small term at M_U and C_U: 0.01
  bounded, 0.05 Gaussian.
* **"1:1.0;1,2,3:1.0".** The same holds only for part of training, roughly steps 500 to 2,500 of 30,000 (Q3). At the
  end θ (0.15) exceeds bounded η (0.10), close to the teacher's 0.18 and 0.12.
* **In T2**, the referee says the added term matters: the twins behave differently, and the difference moves with
  η's reading of {1, 2, 3}. In the decisive twin set θ's reading at C_U stays below 1e-4 nats, and under D_with
  the added term is close to unidentifiable (§1).
* **T1 does not involve {1, 2, 3}**, which is planted support there. In those configurations:
  * η's spurious sum at C_U tracks fragility (ρ +0.58 and +0.72), and about half of it comes from three-way sets
    with lag 4: {1, 3, 4}, {1, 2, 4} and {2, 3, 4};
  * θ's spurious sum, spread over pairs, does not (−0.10 and −0.01).
* **Parity k = 2 at ρ_train = 0.8.** The two η values disagree in sign, so the comparison there is about the scale
  of the η value as much as about the link.

## 4. Instrument checks

| check | result |
|---|---|
| IC-1 replays | 22 of 22 bit-identical (all light-curve keys except wall_s; pattern arrays) |
| IC-2(a) code check | 1,320 (run, checkpoint) pairs; largest difference 3.1e-13 (tolerance 1e-9) |
| IC-2(b) sampling | median signed error ≤ 5.6e-6 in both regions (bound ±5e-4). Share within 0.002 nats, reported without a threshold: θ_M_D 0.985 (Markov) and 0.945 (parity), lowest configuration 0.88; θ_C_D ≥ 0.986 everywhere |
| IC-3 corpora | 160 chain corpora; every transition test within 5 SE (largest \|z\| 3.6); TV 0.0004 to 0.15 (reported only; the largest are the metastable coupling-2.0 chains) |
| IC-4 | D-weighted equals uniform in 200 runs (largest difference 0.0); θ_M_D equals local_ri to 1.1e-14 in 440 runs |
| IC-5 | 320 of 320 Markov and 120 of 120 parity runs pass |
| T2 identifiability | as in §4 of the pre-registration: three chain twin sets excluded at C_D, one at M_D |

The independent verification recomputed IC-4, IC-5 and the identifiability values from the raw data. IC-1, IC-2
and IC-3 were checked against the JSON only.

## 5. Pre-registered questions

1. **Spurious structure by reference: all predictions hold.** Mean number of spurious sets present per run at the
   end of training, over the 240 correlated runs:

   | link | M_D | M_U | C_D | C_U |
   |---|---|---|---|---|
   | θ | 14.2 | 0.63 | 0.00 | 1.28 |
   | η | 13.4 | 0.65 | 0.00 | 0.65 |

   * The marginal reference under D flags spurious sets in every run, with both links.
   * The conditional reference under D flags none. Because C_D is 0 for both links, the predictions "C_D at most
     M_D" and "C_U at least C_D" hold trivially.
   * At C_U, θ flags twice as many spurious sets as η pooled (1.28 against 0.65). The ratio is 1.3 in Markov, 1.4
     in parity k = 2 and 4.7 in parity k = 3.
2. **The divergence law on trained networks: both predictions fail.**
   * **Bounded η value.**
     * There are 408 divergences at C_D. All are θ-only, and all come from one configuration, parity k = 2 at
       ρ_train = 0.5.
     * All have saturation ≥ 0.9, but every one has lower-order energy < 0.1. The pre-registered condition was
       energy ≥ 0.3, so the share meeting it is 0, against a predicted ≥ 0.95.
     * Neither the coupling-2.0 nor the coupling-1.0 Markov configurations have any divergences, so "more in
       coupling 2.0" fails.
   * **Gaussian value.**
     * There are 2,056 divergences. All are η-only; 1,678 of them are in parity k = 2 at ρ_train = 0.8.
     * 16 % meet the energy condition.
     * There are 3.7 per run in coupling-2.0 against 0.25 in coupling-1.0.
   * **Reading.** The direction is that of the review preview: θ-only with the bounded value, η-only with the
     Gaussian one. The law that A1 derived on constructed teachers does not carry over to trained networks in this
     form. All divergences need saturation; large lower-order coefficients within the set are not needed.
3. **Replication of the DR1 divergence on new seeds: holds.** The set is {1, 2, 3} in "1:1.0;1,2,3:1.0".
   * At M_U (the prediction) it is η-only in 10 of 10 chain seeds and 0 of 10 iid seeds, with either η value. As
     noted in §3, this happens only for part of training.
   * At C_U: 10 of 10 chain seeds with both η values.
   * At M_D and C_D: 10 of 10 chain seeds with the Gaussian value, 0 of 10 with the bounded value.
   * iid seeds: 0 of 10 at every reference.
4. **Off-distribution behaviour: holds.** The C_U sum over spurious sets is larger in the chain run than in the
   matched iid run in:

   | link | pairs (of 160) | one-sided sign-test p | median difference |
   |---|---|---|---|
   | θ | 124 | 8e-13 | 2e-4 nats |
   | bounded η | 123 | 2.7e-12 | 1.1e-4 nats |
5. **Exploratory: transport of a sparse description to the shifted distribution.** The evaluation favours θ by
   construction, so no conclusion is drawn.
   * Ties between the top-K sets of θ and of η (bounded / Gaussian):
     * K = 3: 73 % / 81 % of all runs, and 81 % / 88 % of Markov chain runs;
     * K = 5: 61 % / 74 %;
     * K = 8: 37 % / 50 %.
   * Mean transport losses are close overall, with exceptions. In parity k = 3 at ρ_train = 0.8, Gaussian η's top 3
     sets transport far better than θ's: 0.19 against 0.71, better in 20 of 20 runs.

## 6. Disclosure and limits

* **Previews.** The review pilot and the engineering pilot both previewed "η wins" before the run, on other seeds.
  Both are disclosed in the pre-registration. The confirmatory seeds (100–109, 200–219) and data seeds are new.
* **Scope.** These are small MLPs on two task families with 6 or 8 binary inputs. The referees are behavioural
  (concurrent fragility and data twins), not interventions on the network. The result says nothing yet about
  transformers or larger models.
* **The single win is weak.** It comes mostly from one twin set; without that set, T2 at C_U would be a tie. In that
  set θ's predictor stays below the presence floor, and the added term is close to unidentifiable under D_with (§1).
  The pre-registered rule counts the win, and this report states it as it is.
* **Scale.** In the parity comparisons the verdict depends on the η value. "Value-dependent" was pre-registered for
  this reason and is not counted.

## 7. What follows

* **RI's role.**
  * By the pre-registered decision table, Part B (small transformers) adopts the η view as its detector. RI stays as
    a descriptive companion: which inputs a learned term involves, and in which link.
  * The result does not say RI is wrong. Where the views differ, they describe the same network in different
    coordinates. On these referees the mean coordinates line up slightly better with behaviour, mainly in saturated
    configurations.
* **Worth a closer look** (exploratory, not part of the gate):
  * In the {1, 2, 3} chain configurations, θ on {1, 2, 3} stays near 0 while the twins behave differently (§3).
  * In T1, η's C_U spurious sum, which includes three-way sets with lag 4, tracks fragility where θ's does not.
  * Q2: divergences on trained networks need saturation but not large lower-order coefficients within the set.
* **Next sky.** The Part B pilot on the CPU server, then the GPU servers. It needs its own pre-registration. That
  pre-registration should include a referee that intervenes on the network (ablation), because Part A's referees
  were weaker than the design draft's R2 and R3.
