# Independent review of the Part A pre-registration draft (14 November 2026)

Two independent checks were made before the pre-registration became binding: a review of the first draft, with a
pilot (below), and a consistency check of the revision (last section).

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

Other preview facts: all 24 parity runs reached accuracy 1.000 under D, uniform and S by step 270, with final
fragility 0.0006 to 0.03 nats. Q3's divergence ({1, 2, 3} in "1:1.0;1,2,3:1.0" at M_U: η ≥ 0.002 and θ < 0.001 at
two or more consecutive checkpoints after step 0) appeared in 4 of 4 chain seeds (10 to 14 checkpoints) and 0 of 4
iid seeds, with either η value. IC-5 passed in all 96 pilot runs.

## Must fix, and what was done

| # | finding | evidence | change |
|---|---|---|---|
| M1 | The pooled Spearman ranks configurations (Simpson's paradox). | Fragility Y differs about 100-fold between coupling sets (0.007 nats in "1:0.5", 0.81 in "1:2.0;1,2,3:2.0"). At C_D the pooled ρ_θ is −0.42, but θ's mean within-run ρ is +0.43. | Per-configuration Spearman, unweighted mean over configurations (or twin sets). Pooled reported as secondary. |
| M2 | The draft's η value, ½ log(RSS_lo/RSS_hi), grows without bound as the output saturates, so it mixes link with scale. | m = ±0.99: θ 0.662, η 1.959. All C_D divergences are η-only with that value and θ-only with a bounded one; parity T1 changes sign. | Bounded value ln 2 − h((1 + √ΔRSS)/2) is primary; Gaussian value is sensitivity; a win needs both. |
| M3 | The registered seeds reproduce DR1's networks, so Q3 was not a replication; all seeds of a configuration shared one corpus. | Match to 7e-9 (above). | New seeds with data_seed = seed (finally Markov 100–109, parity 200–219; see the consistency check). Pilot disclosed. |
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

The revisions moved individual preview comparisons both ways. Under the draft's rules the preview pointed to
"η wins" on three comparisons (Markov T1 at M_U, C_D and C_U); as far as it computed them, the revised rules point to
"η wins" on one (Markov T1 at C_U, −0.14 and −0.17). The per-configuration statistic moved Markov T1 toward θ at
every reference and parity toward θ with the draft's η value (+0.25, +0.11), which the requirement to win with both
η values makes value-dependent; it moved T2 toward η. The preview did not compute T2 with the bounded value, the
squared-error outcome or the identifiability rule. The revisions were made for the reasons above, not for their
effect on the preview.

## Consistency check of the revision

A second independent agent checked the revised text against this record, recomputed the counts and verified the
mathematical claims of §1 and §4 numerically (bounded η ≤ ln 2 and equal to `eta_information(f_u(A))` at M_U; the
Gaussian value is never below the bounded one, so they share a zero set; singletons have one zero set for both links
at all four references; residual variance 1 at M_U and C_U; the ±0.99 example; the review's M8 example). Every item
above was found implemented. It found, and the pre-registration now fixes:

| # | finding | change |
|---|---|---|
| 1 | IC-3 fails by slow mixing alone: regenerating 160 chain corpora (data seeds 10–19) gave TV 0.02003 for "1,2,3:2.0", which is not exempt, and 0.0194 for "1:1.0;1,2,3:1.0". Transition frequencies pass on all 160 (largest \|z\| = 3.5). | TV reported without a threshold; the bug check uses transition frequencies (5 binomial SEs for windows seen ≥ 1,000 times). |
| 2 | IC-2(b) fails for parity by sampling alone: with 4,096 probes, the share of θ_M_D within 0.002 nats is 0.67 to 0.89 (SD up to 0.018 nats); θ_C_D passes. §2 stored no sampled θ_C_D. | Median signed error is the bug check; the share is reported. Sampled θ_M_D and θ_C_D are stored. |
| 3 | The T2 identifiability rule, computed from the exact pmfs, removes 3 of 4 chain twin sets at C_D but only 1 of 4 at M_D. | The exclusions are listed in the text; the T2 expectation corrected. |
| 4 | §0 claimed more than the preview supports ("do not move the preview toward θ"); two preview facts were not in this record; "not blind" omitted Q4 and the revised test definitions. | §0 rewritten (and the last paragraph of the review section above); the facts added above. |
| 5 | DR1 has Markov refinement runs with seeds 10 and 11, so Q3's "new seeds" was not true for seeds 10–11. | Seeds moved to Markov 100–109 and parity 200–219. |
| 6 | Two §1 statements false as worded: at M_D the zero set depends on D (θ_M_U({2, 4}) = 0 but θ_M_D({2, 4}) = 0.036 in "1:1.0;1,3:0.8"); singletons can disagree about presence at the 0.001 floor (means 0.86 and 0.94: θ 0.0043, bounded η 0.0008). | Corrected; a test of the C_D singleton zero set added. |
| 7 | Resampling seeds within each configuration understates the SE: a seed's runs share initial weights and data_seed across configurations, and two twin sets share each "without" network. | One multiset of seed labels per resample, used for every configuration and twin set. |
| 8 | Q2's predictions could be scored in two ways (reference, η value, which configurations are "coupling-1.0"). | Fixed: C_D, bounded value, the 4 and 7 configurations named. |

Minor points also applied: the T1 Markov expectation (M_U may be value-dependent); the scale of the two η values
(neither matches θ's); M_U and C_U wording; DR1 networks only in DR1's coupling sets; n_tokens 200,000; S defined
fully; condition 3 worded exactly, with "outcome-dependent" for a failure of it; IC-5 seed counting, Q3 as a share
of passing seeds, Q4 ties dropped; IC-5 values exact; the seed-level secondary uses only seeds whose window contains
the checkpoint, and T2 reports the same secondaries; log-odds stored in float64 and m clipped, so the Gaussian value
is finite; the wall-clock estimate; a test that compared a weighting with itself; the amendment summary.
