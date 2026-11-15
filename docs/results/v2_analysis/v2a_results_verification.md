# Verification of the Part A results from DR2a (15 November 2026)

## Method

An independent agent rebuilt every confirmatory number of sky v2 Part A:
* **From:** the raw data release (`releases/DR2a`: 440 pattern files, objects table; all 14 checksums verified).
* **Code:** its own. It did not import `tycho/analysis/v2a.py`.
* **Measures:** its own exact implementation. At the conditional references the logistic fit splits into a
  one-dimensional problem per setting of the inputs outside A; the η values have closed forms. Against
  `projection_information` these agree to at most 2.2e-13, over about 9,600 values, including the most saturated
  checkpoints (|log-odds| 13.7).
* **Cost:** about 20 CPU-minutes, with no subsampling.
* **Compared against:** the server output (`v2a_analysis.json`, code 4ca774f) and the report
  ([../v2a_results.md](../v2a_results.md)).

**Result.** The JSON is right, and the decision ("η wins") and all 12 comparison labels stand. The report had wrong
counts, double-rounded values, and interpretation that went further than the data. None of this changed a label or
the decision. All of it is corrected in the report.

## Agreement

| quantity | scope | largest difference |
|---|---|---|
| Run list, seeds, data_seed = seed, configuration counts | 440 objects | exact |
| Losses vs light-curve values; stored pmf_D vs exact stationary pmf | all objects and checkpoints | 1.1e-16; 8.6e-16 |
| IC-5 | 440 runs | all pass (largest final KL 0.0034; smallest parity accuracy 0.9999) |
| D-weighted = uniform (IC-4) | 200 runs | 0.0 |
| Identifiability residual variances and exclusions | 8 twin sets × 4 references | exact |
| Windows and point counts | 28 configurations and twin sets | exact |
| Per-configuration ρ, ρ̄, Δ (primary, squared-error, all-spurious); pooled, seed-level and chain-only secondaries | 1,380 values | 1.1e-16 |
| Bootstrap SD, 95 % CI, raw and Holm p | 24 tests | identical (same resample matrix) |
| Win conditions, labels, decision | 12 comparisons | identical |
| Q1, Q2, Q3, Q4 | all counts, histograms, shares, sign tests | exact (Q4 differences 1.9e-15) |
| Q5 | tie shares (no fits redone) | match |

IC-1 replays, IC-2 probe sampling and IC-3 corpora were checked against the JSON only, not recomputed.

## Corrections to the report, all applied

1. **Bootstrap counts (§2).**
   * Holm p is 0.0024 in 22 of the 24 tests, not 23. T1 Markov M_D with the Gaussian value has raw p 0.0040 and
     Holm 0.0080.
   * The SDs run to 0.017, not 0.016.
2. **Secondary statistics (§2).** "They agree in direction where there is a difference" was false. For the parity
   comparisons that favour θ with the Gaussian value, the seed-level statistics point the other way. Three values
   were double-rounded: −0.15 should be −0.14, −0.09 should be −0.08, and −0.31 should be −0.30.
3. **"There θ tracks behaviour worse than η … the referees side with η" (§1): overstated.** At the reference of each
   largest gap, θ is worse than bounded η in all six configurations but worse than Gaussian η in only five. At
   other references θ is sometimes ahead.
4. **The {1, 2, 3} paragraph (§3): partly wrong, and it mixed up T1 and T2.**
   * θ on {1, 2, 3} stays near 0 throughout only in "1:2.0;1,2,3:2.0". In "1:1.0;1,2,3:1.0" this holds only for
     part of training.
   * T1 does not involve {1, 2, 3}: it is planted support there. What tracks fragility in T1 is η's spurious sum at
     C_U, about half of it three-way sets with lag 4.
   * In the decisive T2 twin set, θ's predictor stays below 1e-4 nats, under the 0.001 presence floor, and the added
     term is close to unidentifiable under D_with. The report now says so in §1 and §6.
5. **Gap values (§3).**
   * Corrected values: 0.87 (not 0.88), 0.59 (not 0.60), and 1.10 at C_U (not 1.14).
   * The 1.13 at C_D does not count, because the twin set is excluded there.
6. **Q5 (§5): "No difference worth reporting" was overstated.**
   * Ties fall to 37 % / 50 % at K = 8.
   * In parity k = 3 at ρ_train = 0.8, Gaussian η's top three sets transport far better than θ's.

**Minor points, applied:**
* saturation is necessary but not sufficient for a large gap;
* the "within 0.03" group's largest value is 0.031;
* the Q2 directions are given for both η values;
* the other references of Q3 are reported;
* Q4 medians and p-values are given per link;
* the Q1 ratio is given per region;
* the label of the squared-error column;
* "the same kind of configuration as v1/A3" applies to the {1, 2, 3} sets only.
