# Sky v2, step A3: where the θ and η views disagree in DR1

12 November 2026. Exploratory analysis of stored v1 data (DR1), done before any v2 design is fixed. It is not
pre-registered. Its findings inform v2 Part A and are disclosed for the v2 pre-registration. Design:
[v2_design_draft.md](v2_design_draft.md), §4 A3. Scripts: `scripts/analysis_v2/a3_theta_eta_map.py` (the map) and
`a3_exact_reconstruction.py` (an exact check of every divergent checkpoint). Numbers:
[a3_theta_eta_map.json](v2_analysis/a3_theta_eta_map.json) and
[a3_exact_reconstruction.json](v2_analysis/a3_exact_reconstruction.json). An independent verifier re-derived the
map's numbers and contributed the exact check. Items 3 and 4 below are corrected after that verification.

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

**1. The two views agree almost everywhere.** 3.84 million (run, subset, checkpoint) points were examined. The table
counts runs with at least one point flagged by the rule above, using the stored coefficient:

| level k | parity-k2, k3, k4: η-only runs | parity: θ-only runs | markov-v1: η-only runs | markov-v1: θ-only runs |
|---|---|---|---|---|
| 2 × | 0, 0, 0 | 8, 9, 8 | 12 | 9 |
| 3 × | 0, 0, 0 | 0, 1, 0 | 12 | 3 |
| 5 × | 0, 0, 0 | 0, 0, 0 | 12 | 0 |
| 10 × | 0, 0, 0 | 0, 0, 0 | 6 | 0 |

Every Markov η-only run, at every level, is in one configuration: {1}:1.0 with {1,2,3}:1.0, chain contexts (12 of
its 12 runs). Its iid twin has none. The subsets are the planted {x[t−1], x[t−2], x[t−3], x̂} (12 runs) and sets
that contain x[t−1] together with a lag the chain makes informative, such as {x[t−1], x[t−3], x[t−4], x̂}. This is
the case v1 found (v1_results.md §4).

The exact check (item 3) confirms that the divergences in this cell are real, in both directions:
* η-only: 306 of the 360 flagged points, in all 12 runs;
* θ-only: 11 of 15 flagged points, in 7 runs.

The parity θ-only flags and the one Markov flag outside this cell are not divergences (item 4). **Every θ/η
divergence in DR1 is in this one configuration.**

**2. A divergence needs a saturated output and large lower-order coefficients together.** Points of
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

Each cell gives η-only points at 2× over all points in that cell. Only η-only points can appear here, because
θ-only requires R(f) < m. Bins are closed on the right; no value lies on an edge. All 360 points come from the
{1}+{1,2,3} chain cell. The real θ-only points (item 4) also have saturation 0.86 to 0.96. Saturation alone does
not produce a divergence:
learned parity is fully saturated, with |f| ≈ 1, and gives 0 of 2,622. Lower-order structure alone does not either
(0 of 229 points with lower-order energy ≥ 0.3 and saturation below 0.8). This is the mechanism expected from the tanh link: a saturating output
turns first-order log-odds into higher-order moments, and the reverse. Parity targets have no lower-order terms, so
parity never diverges. The chain-trained {1}+{1,2,3} models carry f{1} ≈ 0.75, f{2} ≈ −0.16 and f{3} ≈ +0.18 at
near-saturated cells, so they do.

**3. A naive rebuild fails near saturation; an exact reconstruction works.**
* **The naive rebuild.** It uses exactly uniform probe cells and a mean output f(∅) = 0. It agrees with the stored
  plug-in RI within 0.001 nats at 99.65 % of the 1.55 million points where every sub-subset is stored (99.59 %
  without pairs). It fails where outputs saturate: the median error is 0.023 nats where |f| > 0.99, and 0.002 to
  0.003 nats in the divergent cells. Example: plug-in 0.00017 against rebuilt 0.0057.
* **Why it fails.** The stored coefficients are exact sample values. The failure comes from the rebuild's
  assumptions. The probe cells are a few per cent unbalanced, and the trained models' f(∅) is not 0. Near
  saturation, where 1 − |cell mean| is about 0.025, both effects matter, and the cell imbalance is the larger one.
* **The exact reconstruction.** The probes are deterministic: the runner draws them from a generator seeded by
  (seed, "ri", step). With the probe rows regenerated, the stored coefficients plus one fitted f(∅) per checkpoint
  reproduce every stored plug-in RI at all 258 checkpoints with a flagged point, to within 7e-11 nats. The stored
  RI and coefficients are consistent, and the exact θ and η coefficients are recoverable from DR1.
* **Is the plug-in reliable?** In the verifier's simulation (its own scratch code, not in the repository; 400 draws
  of 8,192 probes from the example's
  population), the plug-in RI is 0.000172 ± 0.000002 against a population value of 0.000172, and R(f) stays far
  above min_effect. The η-only divergence is real, and the plug-in is reliable.

At the flagged η-only points, the exact uniform-cell coefficient gives R(f_u) ≥ 2m in 306 of 360 points (median
0.0035 nats). The exact θ top coefficient there is small (median |θ| = 0.017) while RI stays below min_effect.

**4. Most θ-only flags are artefacts of the η estimator; the rest are in the same configuration.**
* **Parity: no real θ-only divergence.** The 31 flags are single pairs {x_i, ŷ} at about 2 × min_effect, all early
  in training (steps 2 to 399), when the output was strongly biased: the exact f(∅) is about 0.58 to 0.63. The
  stored coefficient is the uncentred sample value. When the mean output is far from 0 and the probe cells are
  unbalanced, it is biased: for a pair, f̂ = f_c + 2δ·f(∅), with δ the imbalance of the input's cell. With the exact
  uniform-cell coefficient, 29 of 31 flags disappear. The other 2 disappear once the base rate is included: the
  pair MI at the fitted f(∅) of 0.76 equals the stored RI. A first attempt to explain these flags by the mean output
  alone was inconsistent, and the verifier showed it does not suffice.
* **Markov: 11 real θ-only points.** They lie in the {1}+{1,2,3} chain cell, in 7 runs, on triples
  {x[t−1], x[t−k], x̂}. Their RI is 0.0021 to 0.0048 nats (2 to 5 × min_effect), the exact θ top coefficient is 0.10
  to 0.20 in absolute value, R(f_u) is below 0.0009 nats, and saturation is 0.86 to 0.96. A log-odds interaction is
  present while the moment is too small to register: the reverse of item 2, at a smaller size. The single flag in
  {1}+{1,3}:0.8 disappears under the exact check.

## What this means for v2

1. **Gate 2 has to be tested where saturation and lower-order structure coexist.** In every other regime of DR1 the
   two views are interchangeable, so no referee could tell them apart. v1's tasks reach that regime in one
   configuration only. Part A should make it a knob: saturation, through the output's logit scale or confidence,
   crossed with lower-order strength. Confident classifiers with many features are in this regime by default, which
   is one more reason to move to realistic data.
2. **Make both views exact, cheaply.** Store f(∅) with the Fourier rows; it costs one row. Compute the η
   coefficient on uniform cells, or centred, rather than as the raw sample average. Keep the probes deterministic,
   so that exact tables can always be reconstructed. For correlated (data) probes, where cells are far from
   uniform, store the per-cell counts and output sums of each kept subset as well.
3. **Compare information with information.** The η view should be put in nats with the output's base rate taken
   into account. Otherwise a biased output makes base-rate effects look like θ-only findings, as in parity here.
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
* Saturation and lower-order energy in item 2 come from the naive rebuild's cell means, which can exceed 1 near
  saturation. They are used only for binning. The exact check (item 3) covers every flagged checkpoint, but not
  points the naive rule did not flag.
