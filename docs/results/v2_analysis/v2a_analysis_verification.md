# Verification of the Part A analysis code (14 November 2026)

The offline analysis of sky v2 Part A (`tycho/analysis/v2a.py`, `scripts/analysis_v2/v2a_analysis.py`) was written
by one agent from the binding pre-registration ([v2a_preregistration.md](../v2a_preregistration.md)) and then
checked by a second agent with its own code. The second agent edited nothing in the repository. The data were the
pilot: 32 objects on seeds 900–901, outside the confirmatory ranges, run with `--min-seeds 2`.

## What was recomputed independently

| check | scope | largest difference |
|---|---|---|
| Measures, with `projection_information` on every point | 32 objects × 60 checkpoints × 41 or 92 sets × 4 references × 3 values | θ 3.0e-13; bounded η 1.9e-15; Gaussian η 1.7e-13 |
| Losses (KL and squared error) under D, U and S | all objects | 4.4e-16 relative |
| Windows, spurious sets, predictor sums, point counts | all configurations | identical |
| ρ per configuration, ρ̄ and Δ (both outcomes; the all-spurious secondary) | 12 comparisons × 3 values | 2.2e-16 |
| Bootstrap (10,000 resamples, replicated exactly from the documented random stream): p, Holm, SD, CI | all comparisons | 3.3e-16; labels and decision identical |
| A synthetic case: 10 seeds, IC-5 exclusions, ties, constant predictors | brute force with duplicated rows and scipy | 2.2e-16 |
| Identifiability residual variances (own weighted least squares on power-iteration pmfs) | 8 values | identical |
| T2 "without" network evaluated under D_with | M_D, C_D | 8.9e-16 |
| Q1 counts, Q2 histograms, Q3 shares, Q4 sign tests | pilot | identical |

The verifier judged 23 of the 24 DECISIONS legitimate readings of the text. The exception was the line-search rule in
DECISION 3 (below).

## Must fix, and what was done

**A stalled logistic fit was not flagged.** When the batched Newton's step-halving found no decrease down to a step
of 1e-8, the fit stopped where it was, with nothing recorded. `projection_information` instead takes the tiny step
and continues.
* Example: the true logit of "1,2,4:1.0" (chain D) plus the error trajectory of a pilot network, set {0, 1, 3} at
  C_D. The fast code gave θ = 0.0303; the reference and an independent high-precision optimiser give 0.0239. The
  Hessian's smallest eigenvalue was 1.7e-12 at the stall.
* Rate on semi-synthetic inputs (16 coupling sets × 20 pilot error trajectories × 60 checkpoints × 41 sets): 18 of
  787,200 C_D θ values wrong at the realistic logit scale (by 0.006 to 0.015 nats), more at larger scales. None on
  the pilot itself. Every wrong value came from a fit flagged as not converged.
* **Fixed:** every fit that has not converged is redone with `projection_information`'s own algorithm and the lower
  loss is kept. Counts are stored per object and reported per region (`timing_stage1.<region>.logistic_fits`). The
  stage-1 version was bumped, which invalidates old caches. A regression test uses the case above
  (`tests/data/v2a_stall_case.npz`).

## Minor points, and what was done

* **Gaussian η precision.** RSS_hi was computed as RSS_lo − ΔRSS, which loses digits when the output is saturated
  (up to 1.2e-6 nats at 5× parity logits; 1.7e-13 on the pilot). **Fixed:** RSS_hi is computed directly.
* **IC-5 pass rate.** Runs that did not finish were left out of the denominator. **Fixed:** they count as failing.
* **The decision was written even when a bug check failed or was skipped.** **Fixed:** it is then withheld, as §3
  requires ("the tests wait for the fix"); the outcome the numbers would give is kept separately.
* **One missing or corrupt file stopped the analysis.** **Fixed:** such an object is listed as not done, counts as an
  IC-5 failure, and withholds the decision.
* **A T1 configuration with no done runs was not listed.** **Fixed:** listed under
  `exclusions.configurations_without_done_runs`.
* **Seed labels.** DECISION 15 now says what the code does: the family's done runs.
* **Not changed:** a configuration with no valid seed in a resample gets ρ* = 0 and stays in the mean (DECISIONS 14
  and 16). With at least 5 of 10 seeds valid this happens in at most 0.1 % of resamples. Q5 differs from the
  verifier's own fit by up to 1.6e-7 nats; Q5 is exploratory.

## Performance at full scale

Nothing grows quadratically. The bootstrap for all three families takes about one minute with 10,000 resamples and
uses about 300 MB. Stage 1 takes about 2.5 s per Markov object and 90 s per parity object, including the
instrument-check work: about 50 minutes on 4 cores, or a few minutes on the 88-core server.
