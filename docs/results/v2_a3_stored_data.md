# Sky v2, step A3: where the θ and η views disagree in DR1

12 November 2026. Exploratory analysis of stored v1 data (DR1), done before any v2 design is fixed. It is not
pre-registered. Its findings inform v2 Part A and are disclosed for the v2 pre-registration. Design:
[v2_design_draft.md](v2_design_draft.md), §4 A3. Script: `scripts/analysis_v2/a3_theta_eta_map.py`. Numbers:
[v2_analysis/a3_theta_eta_map.json](v2_analysis/a3_theta_eta_map.json).

## Question

Every in-situ RI curve of a binary-input region stores, for each kept subset A ∪ {out} and checkpoint, two things:
* the plug-in local RI (the top-order log-linear term, the θ view);
* the Walsh coefficient f(A) of the model's soft output (the η view).

The regions are parity k = 2, 3, 4 and the Markov language, 534 curves in all, with uniform probes. Where, and how
often, do the two views disagree?

With a binary output and uniform probes, both views are expansions of the same function
m(x) = E[2ŷ − 1 | x]:
* the η coefficients are the Walsh coefficients of m;
* the θ coefficients of the terms that involve the output are the Walsh coefficients of atanh(m) = logit/2.

So the two views can only differ through the nonlinearity of atanh: where the output is close to 0 or 1 and several
coefficients are large. To compare a coefficient with an RI value, the coefficient is expressed in nats as
R(f) = ln 2 − h((1 + |f|)/2), the RI the subset would have if f(A) were its only coefficient. With m = min_effect
(0.001 nats):
* **η-only** means R(f) ≥ k·m while RI < m;
* **θ-only** means RI ≥ k·m while R(f) < m.

## Results

**1. The two views agree almost everywhere.** 3.84 million (run, subset, checkpoint) points were examined:

| level k | parity-k2, k3, k4: η-only runs | parity: θ-only runs | markov-v1: η-only runs | markov-v1: θ-only runs |
|---|---|---|---|---|
| 2 × | 0, 0, 0 | 8, 9, 8 | 12 | 9 |
| 3 × | 0, 0, 0 | 0, 1, 0 | 12 | 3 |
| 5 × | 0, 0, 0 | 0, 0, 0 | 12 | 0 |
| 10 × | 0, 0, 0 | 0, 0, 0 | 6 | 0 |

Every Markov η-only run, at every level, is in one configuration: {1}:1.0 with {1,2,3}:1.0, chain contexts (12 of
its 12 runs). Its iid twin has none. The subsets are the planted {x[t−1], x[t−2], x[t−3], x̂} (12 runs) and sets
that contain x[t−1] together with a lag the chain makes informative, such as {x[t−1], x[t−3], x[t−4], x̂}. This is
the case v1 found (v1_results.md §4), now shown to be the only one in DR1.

**2. An η-only divergence needs a saturated output and large lower-order coefficients together.** Points of
subsets with at least two inputs and a non-negligible top coefficient (R(f) ≥ 2m) were counted by two quantities
computed from the stored coefficients. Saturation is the largest |m(x)| over the subset's cells. Lower-order energy
is the root sum of squares of f(B) over proper sub-subsets B:

| saturation | lower-order energy < 0.1 | 0.1 to 0.3 | ≥ 0.3 |
|---|---|---|---|
| < 0.5 | 0 / 4,014 | 0 / 156 | 0 / 9 |
| 0.5 to 0.8 | 0 / 2,948 | 0 / 34 | 0 / 220 |
| 0.8 to 0.9 | 0 / 890 | 0 / 12 | 4 / 129 |
| 0.9 to 0.95 | 0 / 94 | 0 / 3 | 33 / 270 |
| ≥ 0.95 | 0 / 2,622 | none | **323 / 1,017** |

Each cell gives η-only points at 2× over all points in that cell. Saturation alone does not produce a divergence:
learned parity is fully saturated, with |f| ≈ 1, and gives 0 of 2,622. Lower-order structure alone does not either
(0 of 229 points with lower-order energy ≥ 0.3 and saturation below 0.8). This is the mechanism expected from the tanh link: a saturating output
turns first-order log-odds into higher-order moments, and the reverse. Parity targets have no lower-order terms, so
parity never diverges. The chain-trained {1}+{1,2,3} models carry f{1} ≈ 0.75, f{2} ≈ −0.16 and f{3} ≈ +0.18 at
near-saturated cells, so they do.

**3. The stored data cannot give the model's θ coefficients in exactly those cases.** RI rebuilt from the stored
coefficients agrees with the stored plug-in RI within 0.001 nats at 99.65 % of the 1.55 million points where every
sub-subset is stored. It fails where outputs saturate:
* the median error is 0.023 nats where |f| > 0.99;
* the median error is 0.002 to 0.003 nats in the divergent cells, the same size as the divergences themselves.

Example: plug-in 0.00017 against rebuilt 0.0057. The pair MIs at the same checkpoint match their coefficients
exactly, so the mean output is not the cause. Small sampling errors in the coefficients are amplified by atanh near
saturation. In these cases the plug-in RI, computed directly on the probe table, is the reliable quantity. Its
θ-coefficient decomposition is not recoverable from what DR1 stores.

**4. The θ-only cases are small and mostly a base-rate effect.**
* Parity θ-only cases are single pairs {x_i, ŷ} at about 2 × min_effect, all early in training (steps 2 to 399).
* Solving the pair's mutual information for the mean output gives an implied |f(∅)| of about 0.77 (median). Other
  pairs at the same checkpoint imply about 0.56, so the early output was strongly biased. A biased output makes the
  same covariance carry more information. This is a normalisation difference, not an interaction.
* The 11 Markov θ-only cases are all triples {x[t−1], x[t−k], x̂} in chain-trained cells: 10 in {1}+{1,2,3}, 1 in
  {1}+{1,3}:0.8. Their RI is 0.0020 to 0.0048 nats (2 to 5 × min_effect) and their saturation 0.62 to 0.92. A
  log-odds interaction is present there, but the moment is too small to register: the reverse of item 2, at a
  smaller size.

## What this means for v2

1. **Gate 2 has to be tested where saturation and lower-order structure coexist.** In every other regime of DR1 the
   two views are interchangeable, so no referee could tell them apart. v1's tasks reach that regime in one
   configuration only. Part A should make it a knob: saturation, through the output's logit scale or confidence,
   crossed with lower-order strength. Confident classifiers with many features are in this regime by default, which
   is one more reason to move to realistic data.
2. **Store what both views need, exactly.** For each kept subset, store the model's mean output per cell of the
   subset, or for small inputs the output on every distinct probe pattern, together with f(∅). Then both θ and η
   coefficients are exact rather than rebuilt from pruned, noisy coefficients. The Fourier baseline should also
   include the mean output.
3. **Compare information with information.** The η view should be put in nats with the output's base rate taken
   into account, or base-rate effects will look like θ-only findings.
4. **A first referee target exists.** In the {1}+{1,2,3} chain cell the two views disagree on when the three-way
   term is present: RI registers it 1 to 9 checkpoints after the coefficient's 25 % crossing. A rerun of that cell
   with exact storage and an ablation of the three-way component (R3) is a natural first A2 experiment.
5. **Conditional RI.** DR1 stores no full tables, so A3 says nothing about it. In v2, marginal and conditional RI
   will be computed side by side. A0 and A1 will establish what each reports on constructed cases, and the referees
   decide between them.

## Limits

* Exploratory. Thresholds (2 to 10 × min_effect; the bins of item 2) were chosen while looking at the data. That is
  why the full threshold sweep is reported.
* Uniform probes only. Correlated probes, the main topic of v2, cannot be studied from DR1.
* Kept subsets only: after pruning, a subset that Fourier would flag and RI never detected is not stored (v1
  Amendment J). Parity k = 3 and 4 also lack many sub-subsets, so items 2 and 3 cover 40 % of all points (every
  Markov point, 68 % of parity k = 2, 16 % and 11 % of k = 3 and 4).
* Saturation and lower-order energy are computed from the stored coefficients, which are themselves noisy near
  saturation (item 3). They are used only for binning.
