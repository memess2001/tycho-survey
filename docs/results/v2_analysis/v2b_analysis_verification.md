# Verification of the Part B analysis code (16 November 2026)

The offline analysis of sky v2 Part B (`tycho/analysis/v2b.py`, `scripts/analysis_v2/v2b_analysis.py`) was written by
one agent from the binding pre-registration ([v2b_preregistration.md](../v2b_preregistration.md)) and then checked by
a second agent with its own code. The second agent edited nothing.

## Data

* **The pilot:** 10 objects on seeds 900–901 (sky version `sky-v2b-pilot3`), run with `--min-eligible 2`.
* **A synthetic full-scale dataset:** 100 objects built from perturbed and head-permuted pilot files. It includes
  every edge case the verifier could construct:
  * failed objects;
  * a run that did not form the circuit;
  * a size with too few eligible seeds;
  * a run without a context head;
  * a run with two previous-token heads;
  * ties in B2.

## Result

Nothing must be fixed before the confirmatory run. Every quantity agreed with the verifier's own computation, exactly
or to floating-point rounding:

| quantity | largest difference |
|---|---|
| pair measures at C_U and M_U (θ and both η values; all pairs at all 71 checkpoints) | 3.3e-16 |
| bounded η at C_U against M_U (the identity of §1) | 2.2e-16 |
| roles, formation, eligibility, IC-2, per-run AUCs, family sizes, redraw counts | 0 mismatches |
| Δ, raw and Holm p (primary and composition-only), labels, outcome | 0 |
| bootstrap intervals | 1.4e-17 |
| B1 steps, rate and slope; B2 | 0; 2.2e-16 |
| onset thresholds and onsets; B3; trajectories | 1.1e-16; 0; 1.1e-14; 9.3e-17 |

Further checks:
* **Label rule:** checked exhaustively in 6,561 constructed cases, with 0 disagreements. The boundaries hold:
  |Δ| = 0.05 counts as met, and a Holm p of exactly 0.05 does not.
* **Bootstrap:** on a synthetic case with 6 seeds and 2 sizes, the p-values agree with the exact resampling law (all
  462 multisets enumerated) within Monte Carlo error.
* **DECISIONS:** all 20 are legitimate readings of the text.

## Minor points, and what was done

1. **The pilot JSON came from an older commit.** **Rerun** at the current code.
2. **An empty M family was reported as "tie".** **Fixed:** it is now "not evaluable".
3. **A missing light curve or stage-1 cache crashed stage 2.** **Fixed:** it now counts as a stage-1 failure and
   withholds the verdicts.
4. **IC-2 passed when there were no control runs.** **Fixed:** it is now "not evaluable".
5. **B1's slope prediction read "not met" when the slope could not be computed.** **Fixed:** it is now "not
   evaluable".
6. **Withholding on failed data checks was not documented.** **Fixed:** it is now DECISION 21, which also lists the
   "not evaluable" cases.
7. **Objects that do not finish count as not formed** (DECISION 11). This is a legitimate reading, but a crashed run
   could flip the formation prediction. **Procedure:** re-run any failed object before the analysis. The JSON also
   gives the rate over done objects.

The fixes have tests (`test_analyse_edge_cases_are_reported_not_evaluable`).
