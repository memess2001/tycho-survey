# A3 independent verification (12 November 2026)

An independent agent re-derived A3's headline numbers with its own code, written from the stated definitions
before it read the analyst's script. It then read the script and the JSON. Its scratch code is not in the
repository. Its exact-reconstruction method was ported to `scripts/analysis_v2/a3_exact_reconstruction.py`. The
A3 report (`docs/results/v2_a3_stored_data.md`) was corrected accordingly. Its report, condensed:

## Claim 1, the threshold sweep: confirmed

* 2×: parity-k2 0/8, k3 0/9, k4 0/8, markov 12/9 (η-only/θ-only runs).
* 3×: parity η 0 throughout and θ 0/1/0; markov η 12, θ 3.
* 5×: only markov η, 12 runs. 10×: only markov η, 6 runs.
* All markov η-only runs are in "1:1.0;1,2,3:1.0" with chain contexts. Its iid twin has none.
* Of the 9 markov θ-only runs at 2×, 8 are in that cell and 1 is in "1:1.0;1,3:0.8" (chain).
* Every parity θ-only point is a pair: 31 points in 25 runs.

## Claim 2, saturation × lower-order energy: numbers confirmed, wording corrected

* The counts reproduce: 4/129, 33/270, 323/1017, and 0 in every other cell. All 360 points are from the one Markov
  cell.
* The bins are right-closed (`pandas.cut` defaults to `right=True`), not left-closed. No value lies on an edge.
* The map can only contain η-only points, because θ-only requires R(f) < m.

## Claim 3, rebuild consistency: confirmed

* 99.655 % of points are within 0.001 nats; the maximum difference is 0.0650. Where |f| > 0.99 there are 2,047
  points, with a median difference of 0.0230.
* The total includes 427,800 pair points, where the rebuild equals R(f) by construction. Without pairs the share is
  99.59 %.
* 2,616 points have a naive cell mean above 1, which only the rebuild's assumptions can produce.

## Claim 4, the rebuild failure in η-only cases: numbers confirmed, mechanism corrected

* **Probes regenerated.** The probe rows can be regenerated exactly with `torch_generator(seed, "ri", step)`. With
  them, the stored Walsh coefficients and one fitted f(∅) per checkpoint reproduce every stored plug-in RI:
  Markov at all 217 checkpoints with an η-only point to ≤ 6.3e-11, parity pairs to ≤ 3e-16. There is no pipeline
  inconsistency.
* **The cause is the rebuild's assumptions.** The stored coefficients are exact sample values, so sampling errors
  in them are not the cause. The rebuild fails because it assumes uniform probe cells and f(∅) = 0. Mean effects
  over the 360 points:

  | table | mean RI |
  |---|---|
  | uniform cells, true f(∅) | 6.0e-4 |
  | true cells, f(∅) = 0 | 2.7e-4 |
  | full naive rebuild | 6.6e-4 |
  | stored plug-in | 1.0e-4 |

* **Simulation.** 400 draws of 8,192 uniform probes from the example's population (population RI 0.000172):
  * plug-in RI 0.000172 ± 0.000002;
  * R(f̂) 0.0104 ± 0.0013;
  * the naive rebuild ranges from 2e-7 to 0.007.

  The plug-in is reliable and the η-only divergence is real.
* **Reweighting.** With tables reweighted to uniform cells, RI < m at all 360 points, and R ≥ 2m at 85 % of them.
* **Overreach.** Only 51 of the 360 points differ from the plug-in by more than 1e-3 in absolute terms.

## Claim 5, θ-only pairs as a mean-output effect: partly refuted

* The pair's own implied |f(∅)| (0.77) exceeds the other pairs' median at every checkpoint, by 0.13 to 0.84. The
  exact f(∅) has a median of 0.58.
* At the exact f(∅), the pair's MI on uniform inputs is below 2m in 30 of 31 cases. The mean output alone is
  therefore not the explanation.
* The rest is a bias of the uncentred Walsh coefficient, f̂ = f_c + 2δ·f(∅), with a median |δ| of 0.0115. With
  the centred coefficient, R(f_c) ≥ m in 29 of 31 cases. These points are artefacts of the η estimator; the stored
  MI is correct.

## Script issues (fixed in the commit after this verification)

* Docstring statements on f(∅) = 0 and on what the R(f) gap contains.
* θ profiles that are meaningless where the naive cell means exceed 1.
* The relation-label priority was undocumented.
* `implied_f0` dropped the cases with no positive solution.
* The per-checkpoint shares were pooled across regions.

The χ ordering, the table axis mapping, the completeness test, the lower-order energy, the saturation and the
threshold logic were found correct.
