# Sky v2, Part A: results

**Status: pending independent verification.** The numbers come from the pre-registered analysis run on the server
on 15 November 2026 (`v2_analysis/v2a_analysis.json`, code at commit 4ca774f). An independent agent will recompute
them from the data release DR2a. This line is removed when that is done.

Pre-registration: [v2a_preregistration.md](v2a_preregistration.md) (binding, with its amendments). Every rule below
(measures, referees, windows, bootstrap, the 0.10 margin, Holm over 12 comparisons, the three win conditions) was
fixed before any confirmatory data existed.

## 1. Summary

* **Decision by the pre-registered rule: "η wins".** η wins one of the 12 comparisons (T2 at C_U). θ wins none.
  * Four comparisons are value-dependent: they pass with one η value and not the other, so they do not count.
  * Seven are ties.
* **What the numbers mean.** On these behavioural referees the two views are largely interchangeable.
  * In 8 of the 16 Markov chain configurations, in every iid twin set and in parity k = 3 at ρ_train = 0.5, the
    correlations of θ and η with behaviour agree to within 0.03 at every reference.
  * Gaps above 0.5 occur only where the output saturates: "1:2.0", "1:1.0;1,2,3:1.0" and "1:2.0;1,2,3:2.0" (chain),
    parity k = 2 at ρ_train = 0.8, and the two {1, 2, 3} chain twin sets. In the {1, 2, 3} sets, lower-order terms
    also cancel the three-way term.
  * There θ tracks behaviour worse than η. This is the same kind of configuration where v1 and A3 found θ and η to
    disagree, and the referees side with η.
* **The one win rests mostly on one twin set.** T2 at C_U: the twin set "1:2.0 vs 1:2.0;1,2,3:2.0" (chain)
  contributes about 0.14 of the Δ of −0.16. There ρ_θ = −0.49 and ρ_η = +0.61. The pooled secondary statistic for
  the same comparison is about 0.
* **Gate 2 is not passed.** On these referees, RI does not show a sensitivity to behaviour that the mean view lacks.
  By the decision table, RI's role in the next sky is attribution and description; it is not the primary detector.
* **Instrument.** Every bug check passed (IC-1 to IC-4), and all 440 runs met the IC-5 precondition. Of 13.0
  million logistic fits, none needed the fallback added after the analysis-code verification.

## 2. The 12 comparisons

Δ = ρ̄_θ − ρ̄_η; a negative Δ favours η. A win needs |Δ| ≥ 0.10 in the link's favour with both η values, Holm
p < 0.05 with both, and the same sign with the squared-error outcome.

| comparison | ρ̄_θ | ρ̄_η (bounded) | ρ̄_η (Gaussian) | Δ bounded | Δ Gaussian | Δ squared-error outcome | result |
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

* **Bootstrap.** Over seeds, the SDs of Δ are 0.001 to 0.016. Raw p-values reach the floor of 10,000 resamples,
  so the Holm-adjusted p is 0.0024 in 23 of the 24 tests (12 comparisons × 2 η values). The exception is T1
  parity M_U with the Gaussian value (0.33). The 0.10 margin decides every comparison, as the review predicted.
* **Inference is about these configurations.** Only seeds were resampled.
* **Secondary statistics agree in direction where there is a difference.**
  * T1 Markov C_U: pooled Δ −0.11 and −0.15, seed-level −0.08 and −0.09.
  * T2 C_U: the mean over chain twin sets only is −0.31 and −0.30, the pooled Δ about 0, and the seed-level Δ −0.02.

## 3. Where the differences come from

The largest gap between ρ_θ and either ρ_η, over the four references, sorts the configurations:

* **At most 0.03.** 8 of 16 Markov chain configurations: "1,2,3,5:0.5", "1,2,3,5:1.0", "1,3:0.5", "1,2,4:0.5",
  "1,3:0.8", "1,3:1.0", "1,2,4:1.0", "1:0.5". Also every iid twin set, the chain twin set "1:1.0 vs 1:1.0;1,3:0.8",
  and parity k = 3 at ρ_train = 0.5.
* **0.08 to 0.33.** "1,2,3:1.0", "1:1.0", "1:2.0;1,3:1.6", "1,2,3:2.0", "1:1.0;1,3:0.8" (chain); parity k = 2 at
  ρ_train = 0.5 and k = 3 at 0.8; the chain twin set "1:2.0 vs 1:2.0;1,3:1.6".
* **Above 0.5.** "1:2.0" (0.51), "1:1.0;1,2,3:1.0" (0.88), "1:2.0;1,2,3:2.0" (0.97); parity k = 2 at
  ρ_train = 0.8 (0.60); the chain twin sets with {1, 2, 3} added (0.64 and 1.14).

All of the largest gaps are in configurations whose outputs saturate. Examples:

| configuration | reference | ρ_θ | ρ_η bounded | ρ_η Gaussian |
|---|---|---|---|---|
| T1 Markov "1:1.0;1,2,3:1.0" (chain) | C_U | −0.10 | +0.58 | +0.78 |
| T1 Markov "1:2.0;1,2,3:2.0" (chain) | C_U | −0.01 | +0.72 | +0.95 |
| T1 Markov "1:2.0" (chain) | C_D | +0.24 | +0.74 | +0.36 |
| T1 parity k = 2, ρ_train = 0.8 | C_D | +0.40 | +0.78 | −0.19 |
| T2 "1:1.0 vs 1:1.0;1,2,3:1.0" (chain) | M_D | +0.05 | +0.61 | +0.69 |
| T2 "1:2.0 vs 1:2.0;1,2,3:2.0" (chain) | C_U | −0.49 | +0.61 | +0.60 |

The {1, 2, 3} chain configurations are the case that v1 and A3 singled out. There the network learns the three-way
term mostly through saturation and lower-order log-odds, so θ on {1, 2, 3} stays near 0 while the mean view sees
a term. The referees say that this term matters for behaviour: the "with" and "without" networks differ in what
they do, and fragility under the shift moves with η's reading. In the parity configuration at ρ_train = 0.8, the
two η values disagree with each other in sign. The comparison there is about the scale of the η value as much as
about the link.

## 4. Instrument checks

| check | result |
|---|---|
| IC-1 replays | 22 of 22 bit-identical (all light-curve keys except wall_s; pattern arrays) |
| IC-2(a) code check | 1,320 (run, checkpoint) pairs; largest difference 3.1e-13 (tolerance 1e-9) |
| IC-2(b) sampling | median signed error ≤ 5.6e-6 in both regions (bound ±5e-4). The share within 0.002 nats is reported without a threshold: θ_M_D 0.985 (Markov) and 0.945 (parity), lowest configuration 0.88; θ_C_D ≥ 0.986 everywhere |
| IC-3 corpora | 160 chain corpora; every transition test within 5 SE (largest \|z\| 3.6); TV 0.0004 to 0.15 (reported; the largest are the metastable coupling-2.0 chains) |
| IC-4 | D-weighted equals uniform in 200 runs (largest difference 0.0); θ_M_D equals local_ri to 1.1e-14 in 440 runs |
| IC-5 | 320 of 320 Markov and 120 of 120 parity runs pass |
| T2 identifiability | as in §4 of the pre-registration: at C_D three chain twin sets excluded; at M_D one |

## 5. Pre-registered questions

1. **Spurious structure by reference: all predictions hold.** Mean number of spurious sets present per run at the
   end of training, over the 240 correlated runs:

   | link | M_D | M_U | C_D | C_U |
   |---|---|---|---|---|
   | θ | 14.2 | 0.63 | 0.00 | 1.28 |
   | η | 13.4 | 0.65 | 0.00 | 0.65 |

   * The marginal reference under D flags spurious sets in every run, with both links.
   * The conditional reference under D flags none.
   * C_U ≥ C_D in every run.
   * θ at C_U flags twice as many spurious sets as η.
2. **The divergence law on trained networks: both predictions fail.** With the bounded η value there are 408
   divergences at C_D. All have saturation ≥ 0.9, but every one has lower-order energy < 0.1. The pre-registered
   requirement was lower-order energy ≥ 0.3, so the share meeting it is 0, against the predicted ≥ 0.95. Neither the
   coupling-2.0 nor the coupling-1.0 Markov configurations have any divergences, so "more in coupling 2.0" fails.
   With the Gaussian value there are 2,056 divergences, 16 % meeting the condition, and 3.7 per run in coupling-2.0
   against 0.25 in coupling-1.0. The law that A1 derived on constructed teachers does not carry over to trained
   networks in this form. Saturation is common to all divergences; large lower-order coefficients within the set
   are not.
3. **Replication of the DR1 divergence on new seeds: holds.** {1, 2, 3} in "1:1.0;1,2,3:1.0" at M_U is η-only in 10
   of 10 chain seeds and 0 of 10 iid seeds, with either η value.
4. **Off-distribution behaviour: holds.** The C_U sum over spurious sets is larger in the chain run than in the
   matched iid run in 124 of 160 pairs for θ and 123 for bounded η (one-sided sign test p ≈ 1e-12). The median
   difference is small: 2e-4 nats.
5. **Exploratory.** For transport of a sparse description to the shifted distribution, the top-K sets chosen by θ
   and by η give the same model in most runs; for example 81 % tied for K = 3 in Markov chain runs. No difference
   worth reporting.

## 6. Disclosure and limits

* **Previews.** The review pilot and the engineering pilot both previewed "η wins" before the run, on other seeds.
  Both are disclosed in the pre-registration. The confirmatory seeds (100–109, 200–219) and data seeds are new.
* **Scope.** These are small MLPs on two task families with 6 or 8 binary inputs, and the referees are behavioural
  (concurrent fragility and data twins), not interventions on the network. The result says nothing yet about
  transformers or larger models.
* **The single win is fragile.** It comes mostly from one twin set; with that set removed, T2 at C_U would be a tie.
  The pre-registered rule counts it, and this report states it as it is.
* **Scale.** In the parity comparisons the verdict depends on the η value. "Value-dependent" was pre-registered for
  this reason and is not counted.

## 7. What follows

* **For RI's role.**
  * By the pre-registered decision table, Part B (small transformers) adopts the η view as its detector and keeps
    RI as a descriptive companion: which inputs a learned term involves, and in which link.
  * The result does not say RI is wrong. Where the views differ, they describe the same network in different
    coordinates, and on these referees the mean coordinates line up better with behaviour.
* **What is worth a closer look** (exploratory, not part of the gate):
  * the configurations of §3: saturation plus cancellation, where θ and behaviour part ways;
  * the failure of the divergence law in Q2: divergences on trained networks need saturation but not large
    lower-order coefficients within the set.
* **Next sky.** The Part B pilot on the CPU server, then the GPU servers. It needs its own pre-registration, which
  should include a referee that intervenes on the network (ablation), since Part A's referees were weaker than the
  design draft's R2 and R3.
