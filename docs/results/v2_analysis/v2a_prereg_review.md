# Independent review of the Part A pre-registration draft (14 November 2026)

An independent agent reviewed the first draft of [v2a_preregistration.md](../v2a_preregistration.md) (commit
de2affa). It built its own code outside the repository, ran a pilot and made no repository changes.

**The pilot.** 72 Markov runs with the v1 settings: the six coupling sets of the twin pairs in both contexts, plus
six more chain sets, seeds 0–3. 24 parity runs: k ∈ {2, 3} × ρ_train ∈ {0, 0.5, 0.8} × seeds 0–3. The output on all
patterns was stored at the v1 checkpoint grid, and all 8 exact measures were computed for every set. These runs
preview the draft's tests, so they are disclosed in §0 of the pre-registration. The Markov runs reproduce DR1's
networks: seed 0 of "1:1.0;1,2,3:1.0" (chain) matches DR1 object 056f3bde514a798b at step 185 to 7e-9.

**Preview of the draft's tests** (Δ = ρ_θ − ρ_η; negative favours η):

| | pooled, draft η value | per configuration, draft η value | per configuration, bounded η value |
|---|---|---|---|
| T1 Markov: M_D / M_U / C_D / C_U | −0.09 / −0.12 / −0.39 / −0.20 | +0.03 / −0.11 / −0.07 / −0.14 | +0.05 / −0.07 / −0.06 / −0.17 |
| T1 parity: C_D / C_U | +0.07 / +0.06 | +0.25 / +0.11 | −0.09 / −0.13 |
| T2, KL referee: C_D / C_U | −0.05 / −0.04 | −0.20 / −0.14 | not computed |

The bootstrap SD of Δ was only 0.005 to 0.017, so the 0.10 margin alone decided every comparison.

## Must fix, and what was done

| # | finding | evidence | change |
|---|---|---|---|
| M1 | The pooled Spearman ranks configurations (Simpson's paradox). | Fragility Y differs about 100-fold between coupling sets (0.007 nats in "1:0.5", 0.81 in "1:2.0;1,2,3:2.0"). At C_D the pooled ρ_θ is −0.42, but θ's mean within-run ρ is +0.43. | Per-configuration Spearman, unweighted mean over configurations (or twin sets). Pooled reported as secondary. |
| M2 | The draft's η value, ½ log(RSS_lo/RSS_hi), grows without bound as the output saturates, so it mixes link with scale. | m = ±0.99: θ 0.662, η 1.959. All C_D divergences are η-only with that value and θ-only with a bounded one; parity T1 changes sign. | Bounded value ln 2 − h((1 + √ΔRSS)/2) is primary; Gaussian value is sensitivity; a win needs both. |
| M3 | The registered seeds reproduce DR1's networks, so Q3 was not a replication; all seeds of a configuration shared one corpus. | Match to 7e-9 (above). | Seeds 10–19 (Markov) and 20–39 (parity), data_seed = seed. Pilot disclosed. |
| M4 | IC-2 (0.002 nats at 99 % of points) fails by sampling error alone. | 98.2 % with exact-D probes; 94.9 % in one configuration with corpus probes. | (a) code check to 1e-9 with the probe counts as weights; (b) ≤ 0.002 at ≥ 95 %, median signed error within ±0.0005. |
| M5 | IC-3 (TV ≤ 0.02) fails by design for metastable chains. | "1:2.0;1,2,3:2.0": TV 0.005 to 0.21 across data seeds; 44 of 64 windows never occur. | Reported for all corpora; required only where the stationary pmf is not concentrated on two windows. |
| M6 | At the D references the added T2 term is barely identifiable in three of four chain twin sets. | Residual variance of χ_A under D_with after the smaller model: 2e-6, 0.0028, 0.0062, 0.106. | Twin set enters at M_D and C_D only if that residual variance is ≥ 0.01; exclusions listed. |
| M7 | Q4 true by construction (C_U − C_D ≡ 0 for iid runs); Q1 mis-aimed (M_U flags in 27 % of chain runs); Q2 and Q3 undefined. | Pilot counts. | Q1–Q4 redefined as proposed (pre-registration §5). |
| M8 | False statements in §1: the links differ only for \|A\| ≥ 2 (true for zero sets, not values); zero sets are function properties (only for the top set); "differ only in the link". | Counterexample: log-odds/2 = 0.318 s0 + s0s1 + 0.5 s0s2 + 0.7 s0s1s2 + 0.3 s1 gives θ_C_U({0}) = 0 and θ_C_D({0}) = 0.013 under Markov(0.7). | §1 rewritten; a test of the singleton example added. |
| M9 | Free choices left to the analyst. | 2,000 resamples put the smallest Holm threshold on 8 resamples; step 0 gives 80 exact ties in T2. | Bootstrap (10,000, scheme and p-value), windows, Markov loss, parity numbering and label, sign, IC-5 fallback, Q1 set sizes fixed. |

## Should consider, and what was done

1. **The T2 referee is θ's own loss** (at convergence θ_C_D of the "with" network equals the KL referee minimised over
   the smaller model). The preview showed no θ advantage from it. **Adopted:** a squared-error sensitivity outcome
   for T1 and T2; a win needs the same sign with it.
2. **T1 is not R2 and T2 is not R3** as the design draft defines them. **Adopted:** relabelled as behavioural
   referees; a pass is "gate 2 passed on behavioural referees".
3. **Within-configuration ρ mostly measures a shared training-time trend**; the seed-level statistic was only 0.04
   to 0.12 for both links in Markov. **Adopted** as a secondary statistic.
4. **Parity fragility is transient**: all pilot runs converged by step 270. **Adopted:** 2,000 training steps.
5. **The spurious-set sum double-counts.** With the per-configuration statistic the number of summed sets is fixed
   within a configuration, so sum and mean give the same ranks. **No change.**
6. **The iid twin sets cannot separate the links** (|Δ| ≤ 0.01 in the preview). **Not adopted**, to avoid an
   outcome-informed restriction; the chain-only mean is reported as secondary.
7. **BLAS threads must be pinned** (about 600 times slower otherwise in parallel workers). **Adopted** in §6.
8. **Minor.** IC-4 extended to parity ρ = 0. The 4-lag planted sets are not measured: stated in §1. Inference is
   limited to these configurations: stated with the bootstrap. Exact zeros and ties: a rule added (ρ = 0 for a
   constant predictor). IC-5 passed in every pilot run.

The revisions were made for the reasons above. Under the revised rules the preview still points to "η wins", as it
did under the draft's rules. Taken together the revisions do not move the preview toward θ: the per-configuration
statistic alone did, for parity with the draft's η value (+0.25 and +0.11), but the requirement to win with both η
values turns those comparisons into "value-dependent".
