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
