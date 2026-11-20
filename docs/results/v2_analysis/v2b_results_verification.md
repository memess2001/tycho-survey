# Verification of the Part B results (20 November 2026)

The results report [v2b_results.md](../v2b_results.md) was written by one agent from the server's analysis output
(`v2b_analysis.json`, code 55ba69f). A second agent recomputed the results from the raw data release
`releases/DR2b` with its own code and checked the report's wording. It edited nothing in the repository.

## Method

* **Its own θ/η implementation**, written from §1 of the pre-registration. It agrees with
  `tycho.ri.cells.projection_information` to 3.3e-16 (25 objects, about 8 checkpoints each, all pairs, both
  references, θ and both η values), and with the JSON's last-checkpoint pair values to 3.3e-16.
* **Roles, formation, eligibility, B1, B2, M** (AUCs, joint-seed bootstrap, Holm, labels), the §5 onsets and B3,
  recomputed from the light curves and pattern arrays without `tycho.analysis.v2b`.
* **The exploratory §8**, recomputed from the light curves with its own code. It also checked the dates and text of
  the advisor session's predictions on branch `claude/lucid-archimedes-av5rk7`.

## Result

No confirmatory number is wrong.

| quantity | agreement |
|---|---|
| counts by configuration; data checks; IC-2 | identical |
| roles, formation (60 of 60), eligibility (18, 19, 7), the role table | identical |
| B1 phase-change steps, medians, geometric means, slope −0.814 | identical |
| B1 interval (−0.835 to −0.792; no redraws) | identical (< 1e-15); other bootstrap seeds agree to 1e-4 |
| B2 (60 of 60), medians and ranges, descriptive slopes −0.70 and −0.27 | identical |
| M: 8 Δ, by-size Δ, intervals, raw and Holm p, labels, outcome "tie" | Δ ≤ 3e-17; p and intervals identical |
| composition-only Δ; mean AUCs; ceiling counts; ranks; top pairs | identical |
| onsets (1,080 per-run values) and the C_U table; B3 means | identical |
| §7 counts and scores; §8 numbers | identical |

* The tie rule never mattered: no two last-checkpoint pair values were within 1e-9.
* **DECISIONS:** all 21 are legitimate readings of the pre-registration. None changes a result here.

## Corrections, all applied

| # | report text | correction |
|---|---|---|
| 1 | final repeat loss 0.004 to 0.010 at 192 | the minimum is 0.0035: 0.003 to 0.010 |
| 2 | "η ahead by 0.068 with both η values" | "ahead" is a defined label that this result does not earn; the Gaussian Δ is −0.067. Same misuse in ROADMAP, which also misdescribed the referees (under both, the context head is outside the circuit; they differ in how its pairs with circuit heads count) |
| 3 | no-repeat thresholds "crossed at steps 1 to 22" | those are medians; the range is 1 to 31 |
| 4 | "at 48 and 96 no layer-0 head exceeds 0.33" | the largest is 0.3303 |
| 5 | "12 of 20 have two context heads, and that is why 192 is dropped" | 9 of the 12 have the strong head and 3 do not; one more run has two previous-token heads; 13 are ineligible |
| 6 | onsets: "M_U is similar" | not for θ under the 1-layer calibration; the M_U differences are now listed |
| 7 | B3: "the views differ mainly in the context share" | the largest gap is on composition pairs (0.23 to 0.27), then context (0.12 to 0.17) and redundancy (0.09 to 0.13) |
| 8 | advisor prediction 2: "depends on the reading" | not met as worded under either reading; a near miss up to the leave point |
| 9 | prediction 1: two explanations | add probe-set variation (the verifier regenerated the ten probe sets: a simple content-free predictor varies by SD 0.04 across them, the same order as the shortfall); 1.56 is not a bound on all content-free predictors |
| 10 | composition pair first under Gaussian η at M_U: 48 and 192 only | add 2 of 20 at 96 |
| 11 | "a tie that depends on one labelling choice" | the tie does not; whether η is favoured at C_U does |
| 12 | induction step "about as fast as the phase change" | "nearly as fast" (−0.70 against −0.81) |
| 13 | slope −0.81 in the summary, −0.814 in §3 | −0.814 in both |
| 14 | the context-head cut: only the two runs at 192 mentioned | at 48, 7 of 19 context heads score 0.155 to 0.176 and one run is ineligible only because of the cut |

## Added at its suggestion

* **The cut, checked** (exploratory rerun with the same bootstrap). With cuts of 0.10, 0.20 or 0.21, or with d_model
  192 forced into the family, the outcome stays "tie"; η is met only under referee (i) at C_U (Δ −0.059 to −0.077).
  192 enters the family only with a cut of 0.21 or more. Now in report §7.
* **Composition-only under referee (ii):** Δ −0.021 at C_U and −0.011 at M_U (Holm p 0.054). All eight
  composition-only Δ favour η. Now in report §5.
* **Commit dates are declared by the author.** Git shows the order and that the predictions' text is unchanged since
  41d4d7a, not the push time. Now in report §8.

## Not checkable from DR2b

* **IC-1 bit-identity** needs the server's hardware and thread count. The verifier's replay of another object
  (d_model 48, seed 400) on a different machine (1 thread) was not bit-identical: the log-odds differed by about 3e-10
  at step 0, growing to 2e-3 in the repeat loss and 0.06 in the final log-odds. The phase change, roles, formation and
  final scores (to 0.001) were unchanged. This is consistent with float32 kernel differences between machines. The
  report now says that the bit-identity holds on the server.
* **Trajectories:** the report states no numbers from them, so they were not checked.
