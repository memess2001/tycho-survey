# A0 and A1 independent verification (14 November 2026)

An independent agent rebuilt every check from the definitions with its own code (not in the repository), then read
the analyst's code and report. All exact values and E8 summaries recomputed. No error changed a reported result.
The corrections below were applied in the commit that follows this verification: code, a rerun of E8 and A1, and
the report `docs/results/v2_a0_a1.md`. Condensed report:

1. **Conditional RI as a logistic cross-entropy drop: the mathematics is confirmed.**
   * Model I contains every input-only subset, so the projection keeps the input margin exactly. D(p‖q) is then the
     weighted soft-target cross-entropy.
   * On 37 tables (Dirichlet, correlated Ising, near-saturated, and sparse empirical with up to 114 of 128 cells
     empty), the logistic value matches an independent full log-linear Newton solver to within 8.5e-14.
   * The existing IPF (`refined.conditional_ri`) is the inaccurate one near saturation. It stops at 10,000 sweeps,
     with errors up to 5e-5 nats, or 87 % relative for tiny values. "Within 1 % of IPF" holds only off saturation.
     The speed-up is 2 to 100 times, depending on the table.
2. **`eta_information` and the logit coefficient: confirmed.** The logit coefficient is the spin log-linear
   coefficient of B ∪ {ŷ} to within 4e-16, and w(A) = 0 gives local RI 0.
3. **The family 1 attribution map: confirmed**, all 45 classes. A finer scan finds divergence at min_effect 0.001
   from a largest |m| of about 0.74, not 0.8. The boundary depends on the threshold, and exact divergence exists at
   any saturation, below 0.001.
4. **Family 4: values confirmed.** Within 3.4e-13, and the power-loss table reproduces: 194 times at ρ = 0.95.
   Every subset with an unused input has ri_cond ≤ 5e-17 and f_u = 0. Corrections:
   * "Marginal RI and the data moment report every correlated unused input" is false for the pair teachers. Many of
     those sets are exactly 0 by symmetry.
   * The copy teacher uses only x0, but the script treated x2 as used. The spurious counts are 149 and 192, not 144
     and 187.
5. **Family 3** (the v1 profile): confirmed. RI 0.000172, η 0.0103, top logit coefficient 0.066.
6. **E8: the numbers are confirmed**, and the verifier's own nulls at n = 1,024 and 4,096 agree.
   * **Main flaw.** In the block design no tested null input is ever correlated with an input the teacher uses,
     which is the case conditional RI exists for. With that case added, the test still holds at α = 0.01 but is
     less conservative: pooled up to 0.73 %, and q99 near the recommended min_effect at n = 1,024. **Fixed:** E8
     now has a correlated-null design, with sets of order 2 to 4.
   * **Further points.** Sets of order 4 were not tested in the block design. Each cell's q99 rests on 3 to 6
     extreme values. Outputs are soft only.
7. **Code defects.**
   * The `cells.py` docstring still described IPF as the default.
   * `tol` is ignored by the logistic method.
   * `eta_d` (eta_information of the uncentred data moment) becomes NaN under strong correlation, when
     |f(∅)| + |m_D| > 1. **Fixed:** `eta_d` is now the mutual information between ŷ and the parity feature χ_A
     under D, which is always defined and equals the old value when χ_A is balanced.
8. **Overclaims in the report**, all corrected:
   * ri_cond's independence of D (only its zero set is independent of D);
   * the 1 % agreement and the 75× speed-up;
   * "one cell above nominal" (six cell–null combinations exceed 1 %);
   * the KS range (0.03 to 0.42);
   * "valid under every probe kind" (beyond the design then);
   * the E8 marginal q99 at n = 1,024 (up to 0.0044, above v1's 0.0029);
   * the family 1 estimates test the nuisance-marginalised classes, which differ from the clean map in 17 of 45
     constructions;
   * the 0.8 boundary;
   * family 4's "every correlated input".
