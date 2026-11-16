# Independent recomputation of the hierarchy-projection note (16 November 2026)

Two independent agents recomputed the numbers of [v2_hierarchy_projections.md](../v2_hierarchy_projections.md).
Each was given the definitions and the DR2a files and nothing else: not the numbers, not the scripts, not the note.
Each wrote its own code outside the repository and made no repository changes. Their numbers were then compared
with [hierarchy_projections.json](hierarchy_projections.json) and
[lightcurve_lead_lag.json](lightcurve_lead_lag.json) item by item.

Both verifiers are AI agents of the same kind as the one that did the analysis. This checks the arithmetic and the
definitions as written. It does not check whether the question is well posed or whether the result is new.

## What was recomputed, and how well it agrees

| part of the note | recomputed | agreement |
|---|---|---|
| Rungs and visits (§2.1, §2.2) | L_0 to L_6 of all 32 configurations; which rungs are intermediate; for each of the 11: step length, L_k, closest approach (median, minimum, maximum over seeds), straight-path median, steps of closest approach, loss ratio; the overall medians and counts | the same 11 rungs and the same steps; every value to within 3e-10 |
| θ and η rungs (§2.4) | the four closest-approach ratios of every intermediate rung; in how many the two versions agree to 0.001 | 8 of 11 for both metrics, the same three exceptions; values to within 2e-10 |
| Order content and delays (§2.5, §2.6) | c_k(t) for every run; half-times; per-seed delay ratios of the first three twins; G_1 and 1/G_1 | to within 5e-13 |
| Stand-in twin, alone times, example run (§2.3, §2.5, §2.6) | recomputed from the first verifier's own order-content arrays with the note's half-time rule; rung coefficients from its own rungs | delays and half-times to 6 digits; the example run to 2e-13; coefficients to the digits printed |
| Walsh progress and gain-weighted time (§2.6) | raw and gain-weighted ratios of the four twins at three fractions (median, minimum, maximum) | identical to floating-point precision |
| Chain against iid (§2.6) | ratios for the six coupling sets; 1/G at the chain rung | ratios identical; 1/G to within 7e-11 |
| Light-curve indicators (§3) | loss times and ratios for 12 configurations × 8 indicators × 3 fractions; counts below 0.8 and the smallest ratios | all 324 medians identical; the same counts (0, 1, 4) and the same smallest cells |
| Predictions (§6) | L_0, L_1, rule A and rule B for the 9 proposed sources and the 4 twins; the rungs for a 3-lag window | to within 1e-12, except rule A of the most saturated source (27.54), which agrees to 1e-6 |

The first verifier also checked its own projections three ways: a second solver written separately (agreement to
6e-13 on every array), closed forms for the iid two-coupling sources (12 digits), and 28,512 random perturbations
inside F_k, none of which lowered the objective. Its Pythagoras error is 4e-16 and its largest gradient 1e-15.

## What the verifiers pointed out

These are now stated in the note.

1. **The count "below 0.10 in 6" depends on the median convention.** {1}:1.0 + {1,2,3}:1.0 (chain) has a median of
   0.1013 with its two middle seeds at 0.098 and 0.104; a lower median would give 7.
2. **Two of the 11 rungs lie next to the truth** (L_1 = 0.0035 and 0.0037 nats). The network is closest to them
   late, with its loss already below L_1, and for one of them the straight path is closest at or near its end
   point. They pass the 0.002-nat rule without being waiting states.
3. **The KL comparison for η rungs depends on a clip constant.** In the three cases where the θ and η rungs
   differ, the least-squares fit leaves [0, 1] on 18, 2 and 3 windows (D mass 1.5e-3, 8.4e-4 and 7.4e-4). For
   {1}:1.0 + {1,2,3}:1.0 (chain), 90 % of the η rung's KL distance from the truth comes from the windows clipped to
   1e-9; that distance is 0.0130 at 1e-9, 0.0088 at 1e-6 and 0.0172 at 1e-12. The squared-error comparison does not
   have this problem.
4. **The checkpoint grid is uneven.** From step 74 on, twelve of the gaps are about twice as wide as their
   neighbours (74→96, 125→162, 211→273, …), and several closest-approach steps and half-times fall next to one.
5. **Seeds share their initial network.** The log-odds at step 0 are bit-identical across all 32 configurations for
   a given seed. Seeds are a true pairing across configurations and not independent between them.
6. **Median of ratios, not ratio of medians.** The delays are medians of per-seed ratios. The ratios of the two
   medians differ slightly: 1.51 / 1.56 / 1.64, 1.33 / 1.38 / 1.45 and 1.38 / 1.44 / 1.49 for the first three twins.
7. **The {1,2,3}:2.0 chain runs have not converged off the chain.** The uniform Walsh coefficient of the planted
   term averages 0.91 of its asymptote over the last five checkpoints (0.86 to 0.95 across seeds) and is still
   rising by 4 to 9 % over them. The chain-over-iid ratio of §2.6 is therefore taken against an unconverged final
   value; against the asymptote it is 7.60 / 9.39 / 15.3 instead of 7.32 / 8.68 / 12.0. Every other configuration is
   within about 2 % of its asymptote.
8. **The five light-curve cells below 0.8 are ill-defined.** Each is a rise-and-fall series whose net change is
   0.06 to 0.22 of its range. The smallest (the eNTK trace of {1,2,3,5}:0.5, 0.43) has per-seed ratios from 0.11 to
   55, with the sign of the net change differing between seeds. With the loss time taken on the eNTK's own 15
   checkpoints, the counts below 0.8 are 1 at 25 % and 1 at 50 %; with the eNTK final value taken at its last
   evaluation alone, 0 at 25 % and 3 at 50 %. With the ratio of medians instead of the median of ratios the counts
   are unchanged (0, 1, 4).
9. **At the 10 % level the loss time of weak couplings is in the initial transient.** For {1,2,3,5}:0.5 it ranges
   from 2.6 to 87 steps across seeds (median 15), against 152 at 25 %. One run of that configuration crosses the
   25 % level twice.
10. **Conventions that change a number slightly.** Starting the gain-weighted integral at step 0.5 instead of 0
    changes the ratios by at most 0.4 %. Rule B for the stand-in twin uses an alone source at the same strength
    (1.6), which gives 2.98; with the strength of the stand-in run (1.0) it would be 2.46.
11. **Closed forms.** For the single-coupling chain sources the gain at the rung is 1 − tanh⁴β to 1e-15, and for
    the two-coupling iid sources L_0, L_1, G_1 and rule B match hand-derived values to 12 digits.
12. **Nothing in the data looked wrong.** `truth` equals the source's conditional to 3e-13, the iid D is exactly
    uniform, and the chain D is the exact stationary distribution of the window to 7e-16. The light curves' exact
    `x_lossD` equals the loss recomputed from the pattern files at all 60 shared steps. The chain D can be very
    concentrated: masses go down to 1.9e-11, and {1}:2.0 + {1,3}:1.6 has 98.8 % of its mass on one window. Six
    chain runs of that source have no eNTK alignment values; they are not used here.

## Not covered

* The sketch of §2.6 (why the gain should set the delay) is an argument, not a computation; nobody checked it.
* The statements about earlier work in §5.
* Whether the definitions chosen after the fact (the 0.002-nat rule, the straight reference path, the fractions)
  flatter the result. The note reports them as post hoc.
