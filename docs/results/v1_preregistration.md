# Sky v1: design, pre-registered criteria and run-time estimate (written 11 November 2026, before the run)

Sky file: `configs/sky_v1.yaml`. Start on the lab server: `git pull && bash ops/linux/start_v1.sh` (see
`docs/OPS_LINUX.md`).

v0 was a system test. v1 asks sharper questions on tasks whose answers are known, and fixes the problems v0 exposed
(`docs/results/v0_findings.md` §3, §5). It does **not** test whether RI sees more than Fourier analysis: every RI probe
in v1 is still uniform, where the two carry the same information. That question needs correlated inputs or hidden
variables (v2 and v3).

## What changed since v0

| v0 problem | v1 change |
|---|---|
| Every follow-up went to grokking-labelled objects (the label counted as a transient), so there were no controls. | Photometric labels no longer raise follow-up priority. Within a region, four streams take turns: anomalies, matched sets (every outcome at a configuration whose seeds disagree), boundary members, and labels in turn. Regions share the budget by priority. |
| No "unknown" exit: 0 of 6,855 objects were unknown. | An anomaly score (robust z within region and label) flags curves unlike their peers; about 1 % on DR0. |
| Follow-ups capped at 200 per day. | 10,000 per day (`start_v1.sh` sets it); the backlog limit of one per worker still protects the survey. |
| Hidden-unit RI used the threshold calibrated for 4,096 probe rows, but modular addition probes all p² pairs (529 at p = 23). | A calibrated `min_effect` is re-read at the scan's actual row count. grokking-v1 also sets 0.005 nats: twice the largest higher-order binarisation artefact of calibration E6. |
| Markov "lower order delays higher order" was confounded by correlated training contexts. | `train_context: iid` trains on independent windows with the same conditional; every Markov configuration runs both ways. |
| 20k steps censored grokking. | 50k steps. |
| 30 log-spaced RI checkpoints were too coarse for onsets. | 60 in situ (follow-ups add 30 dense points around the largest loss drop). |
| No Fourier comparison inside the survey. | Every RI curve of a binary-input task stores the Walsh–Fourier coefficient of the model's output on the same subsets at the same checkpoints. |

## Regions

| region | task | grid | seeds | objects | steps | RI |
|---|---|---|---|---|---|---|
| parity-k2 | sparse parity, k = 2 | n ∈ {16, 24, 32, 48, 64} × lr ∈ {0.03, 0.1} | 10 | 100 | 20k | in situ, 60 points |
| parity-k3 | sparse parity, k = 3 | same | 10 | 100 | 60k | in situ |
| parity-k4 | sparse parity, k = 4 | n ∈ {16, 24, 32} × lr | 10 | 60 | 150k | in situ |
| markov-v1 | Markov language, window 6 | 12 coupling sets × train_context {chain, iid} | 10 | 240 | 30k | in situ |
| grokking-v1 | modular addition, p = 47 | train_frac {0.3, 0.4, 0.5} × width {64, 256} × lr {1e-3, 3e-3} × wd {0.1, 0.3, 1.0} | 5 | 180 | 50k | follow-up |

680 objects before boundary refinement (extra seeds at boundaries; midpoints on train_frac and weight decay).

## Pre-registered questions and criteria

Onset of a subset = its first RI activation event. Fourier onset = the first checkpoint where |f̂(A)| reaches the
ε at which ln 2 − h((1 + ε)/2) equals the run's `min_effect` (the same threshold expressed as a coefficient).
Medians are over seeds; runs that never reach an onset are censored at the step budget, not dropped.

### Instrument checks (a failure means a bug or a mis-set threshold, not a finding)

1. Parity: the planted set {x_S, ŷ} activates in ≥ 95 % of runs that reach 90 % test accuracy.
2. Parity: RI onset and Fourier onset fall on the same or adjacent checkpoints in ≥ 90 % of runs. With uniform
   probes the two are functions of each other, so a disagreement is a bug.
3. Markov, one planted coupling: the planted subset activates in ≥ 90 % of runs whose final `kl_true` < 0.01, and a
   non-planted subset with peak RI ≥ 10 × `min_effect` occurs in ≤ 5 % of runs (its population RI is exactly 0).

### Questions (estimates; any outcome is reported)

4. **Parity: how does the onset scale with n, per order?** Fit log(median onset) on log n for each k and lr, with a
   bootstrap over seeds. Reference: SGD needs of order n^(k−1) samples, so exponents near 1, 2 and 3; the v0 hint was
   lower (≈ 1.0 and 1.2–1.6 for k = 2, 3 from three n values).
5. **Markov: does the order set the onset?** At fixed strength and context, median onsets should increase with order
   2 < 3 < 4 < 5. Supported if the order-wise seed ranges do not overlap at β = 1.0 in both contexts.
6. **Markov: is the v0 delay a context-correlation effect?** v0: {1,3}:0.8 activated at steps 14–36 alone and 96–133
   with {1}:1.0. Compute R = median onset with the pairwise term / median onset alone, for chain and for iid
   training, with bootstrap CIs.
   * R_iid ≈ 1 while R_chain > 1: the delay came from correlated contexts (the confound).
   * R_iid > 1 as well: lower-order structure delays higher-order learning even with independent contexts.
   The same comparison is made for {1,2,3}:1.0 alone vs with {1}:1.0.
7. **Grokking: do hidden-unit interactions form before generalisation in grokking runs more than in matched
   controls?** At configurations whose seeds disagree, compare the fraction of runs with a strong activation
   (peak ≥ 0.02 nats) after the training set is fitted, grokking vs memorization or smooth seeds. The v0
   observation (59 % of grokking runs) counts as supported only if the difference has a bootstrap CI above 0.
   Otherwise it was a property of the label-selected sample.
8. **Grokking: censoring.** Fraction of runs labelled memorization at 50k steps, compared with v0's 866 of 2,166 at
   20k.

### System checks

9. Follow-ups in grokking-v1: no label takes more than 50 % of them while other labels still have candidates.
10. Anomalies flagged: about 1 % per (region, label) group; each one is looked at by hand.

## Run-time estimate on the 88-core server (70 workers)

Per-step costs are from DR0 (measured on the server under full load) and from local timing. RI scan costs were
measured locally: about 4 s per checkpoint for the largest scans (≈ 43,700 subsets: parity k = 3 at n = 64, k = 4
at n = 32).

| region | objects | per object (median, CPU s) | CPU hours |
|---|---|---|---|
| parity-k2 | 100 | 30 (20k steps + small scans) | 0.8 |
| parity-k3 | 100 | 60–300 (scan cost grows with n) | 4 |
| parity-k4 | 60 | 170–400 | 4.4 |
| markov-v1 | 240 | 20 | 1.3 |
| grokking-v1 | 180 | 70–160 | 5.8 |
| refinement (≈ 50 % more grokking and parity objects) | ≈ 150 | | ≈ 5 |
| grokking follow-ups (replay + RI) | ≈ 250 | 70–160 | ≈ 8 |
| **total** | | | **≈ 30 CPU hours** |

Memory: the largest parity scans hold about 0.13 GB of RI arrays per object until the curve is pruned at the end
(≈ 43,700 subsets × 60 checkpoints × 6 arrays), so 70 such objects at once would need about 9 GB.

On 70 workers this is about **30 minutes** of compute. The longest single runs take about 7 minutes. Follow-ups
start as grokking objects finish and are requested at up to 70 per 10 minutes. Expected wall-clock time is
**1 to 2 hours** if the server is otherwise idle. Other users' jobs on the shared server stretch this
proportionally.

## Amendment 1 (12 November 2026: after the launch, before anyone read v1 data)

Sky v1 was launched on the server on 12 November at about 15:10 JST. While it ran, an independent audit examined
commit f977553: five auditors, and a separate skeptic who tried to refute each finding. Its evidence came from
three sources: DR0 (v0 data), synthetic catalogs, and pilot runs of v1 objects in another machine. It did not use the
server catalog. Nobody has read v1 data from the server before this amendment.

Some pilot runs preview results. Those previews are disclosed below where they bear on a question, so this
amendment is not blind to them. The criteria above stay as written and are reported literally. Where this amendment
changes a definition, both results are reported and the amended one decides. None of the changes needs a rerun:
every series they use is stored. That includes the RI, p-value, q-value and Fourier rows of every ground-truth
subset at every checkpoint, the curve metadata, labels and follow-up records.

**A. Check 2 (parity, RI onset vs Fourier onset).**
* The problem: the Fourier onset above crosses a bare threshold ε = 0.0447 (RI = min_effect = 0.001). An RI
  activation also needs three other things:
  * a Benjamini–Hochberg q ≤ 0.01 over all scanned subsets;
  * bias correction;
  * two consecutive checkpoints.

  At 4,096 probe rows the q-value is the binding condition. It needs RI ≥ 0.0019 to 0.0033 nats, depending on the
  number of candidates, which is |f̂| ≥ 0.062 to 0.081.
* The effect: for k = 2 the coefficient grows gradually, so the RI onset trails the bare Fourier onset by 2 to 5
  checkpoints. Pilot runs agreed within one checkpoint in 6 of 18 and 2 of 21. For k = 3 the coefficient jumps
  within one checkpoint, and 9 of 9 agreed.
* Amended definition: the Fourier onset uses the same gate as the RI event. It is the first of two consecutive
  checkpoints where R = ln 2 − h((1 + |f̂|)/2) satisfies both conditions below, where m = `meta["n_candidates"]`
  and N = `meta["min_effect_n"]`:
  * R − 1/(2N) ≥ min_effect;
  * min(1, m · χ²₁.sf(2NR)) ≤ α (a conservative stand-in for that subset's BH q-value).

  Pilot runs then agreed in 11 of 12.

**B. Check 3 (Markov, non-planted subsets).**
* The problem: chain-trained models really do learn non-planted interactions for a while, because the chain makes
  unplanted lags informative. For {1,2,3}, s_t correlates with s_{t−4}. In pilot runs (x[t−4], x̂) reached
  0.16 nats before the planted term activated, then decayed to 0. With {1}:1.0, pairs (x[t−2], x̂) and (x[t−3], x̂)
  peaked at 0.012 to 0.023 nats and vanished by step 500. iid-trained twins stayed below 0.005.
* Amended criterion: at the end of training, the largest non-planted RI over the last five checkpoints is
  ≤ 10 × min_effect in ≥ 95 % of runs, in both contexts.
* The peak version is also reported literally. These transient chain-induced interactions are reported as a result:
  they also delay the planted term in chain runs ({1,2,3}:1.0 onset at steps 162 to 273 with chain contexts against
  96 to 125 with iid), which matters for questions 5 and 6.

**C. Onsets (questions 4 to 6).**
* The problem: the untrained network already shows small significant pair RI (up to 0.009 nats) from its random
  initial weights. This is sometimes on the planted pair with the wrong sign, which RI cannot see.
* Amended definition: an onset is the first activation event that is not left-censored (it does not start at the
  first checkpoint) and at which the Fourier coefficient has the sign of its final value.
* Order-2 Markov onsets (1 to 8 steps) are at the scale of that initial offset. They are reported as ≤ 8 steps.
  Question 5's non-overlap criterion is evaluated as written and also for orders 3 to 5.

**D. Question 5, preview (no change to the criterion).**
* What the pilots showed: at β = 1.0, seeds 0 to 3 put the onsets of orders 4 and 5 on the same checkpoints (96, 110,
  125). There the 60-point grid has gaps of 1.14 to 1.3 times.
* What follows: the registered "supported" criterion is therefore unlikely to be met for orders 4 and 5. That would
  be a resolution limit, not evidence that their onsets are equal. A dense replay of these objects is the follow-up.

**E. Question 6 (is the delay a context effect?).**
* The problem: R mixes the delay with effect size. With {1}:1.0 present, the population coefficient of the {1,3}
  term under uniform probes falls from 0.664 to 0.375, while an onset is the crossing of one fixed threshold.
* Added definition, which decides: the onset is the planted coefficient, with the planted sign, reaching 25 % of its
  population value. The population value is computed from the couplings, not from the run's final value.
* The registered R is reported next to it, as is R at 10 % and 50 %.
* Preview: in pilot runs R_iid > 1 under every definition (1.5 to 3.2), and R_chain was larger (2.9 to 6.1).

**F. Question 7 (grokking controls).**
* The problem: in modular addition, seeds at one configuration disagree only where the 90 % test-accuracy crossing
  falls near the end of the budget. In DR0 the 126 non-grokking seeds at such configurations were smooth or
  plateau_drop, with test accuracy 0.66 to 0.88: the same generalising phase, never memorization. On the v1 grid,
  DR0 had no configuration where grokking and another label met.
* Amended analysis:
  1. The same-configuration comparison is kept, but its controls are called near-grokking, and each control's final
     test accuracy is reported.
  2. A second comparison is added and labelled as confounded by configuration: memorization objects at the
     neighbouring grid point across a boundary, matched on width and learning rate.
  3. A confidence interval that includes 0 with few configurations is reported as inconclusive, with the counts. It
     is not read as "a property of the label-selected sample".
* Sample: with 10,000 follow-ups per day every grokking-v1 object can be followed up, and the analysis uses all of
  them. A matched set counts as complete only when every label's follow-up is done; failed follow-ups are listed
  with their error.

**G. Question 8 (baseline).** "866 of 2,166" mixed two counts. 866 memorization labels were in all v0 regions; the
grokking region had 742 of 2,166 (34 %). The like-for-like baseline is the v0 objects on the v1 grid points: 93 of
145 memorization (64 %), on 31 of the 36 points, at 20k steps. The comparison is made point by point.

**H. Check 9 (follow-up balance).**
* Expected result: it can fail by design when one label dominates the region. The matched and boundary streams make
  about two thirds of the picks and do not look at labels.
* How it is reported: literally, over the request order (`followups.rowid`), with this as a declared deviation.

**I. Check 10 (anomaly rate).**
* The problem: the scheduler scores only grokking-v1, so nothing is recorded for the other regions. The rate also
  differs a lot by group: on DR0 it is 1.1 % overall but 0 to 10 % by group.
* How it is evaluated: offline on the final catalog for every v1 (region, label) group. The per-group result is
  reported literally, with the overall rate next to it as a declared deviation.

**J. What the stored Fourier baseline can show.** After pruning, Fourier rows exist only for subsets kept by RI:
every subset ever detected, the 200 strongest, and the ground truth. The baseline can compare RI and Fourier on
those subsets, but it cannot show a subset that Fourier flags and RI does not. With uniform probes the two are
equivalent anyway.

**K. Resources.**
* Memory: peak memory per worker on the largest parity objects was measured at 0.5 to 0.7 GB, not the 0.13 GB
  estimated above.
* Workers: the daemon runs 86 workers (the default, cores minus two), not 70.
* Combined: up to about 40 GB if all 60 large parity objects run at once.
