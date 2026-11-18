# Sky v2, exploratory: the training trajectory and the hierarchy of projections of the data

16 November 2026. Exploratory analysis of stored v2a data (DR2a). It is **not pre-registered** and needed no new
training. It was done by an AI agent during an outside review of the project, after Part A's results were known, so
every definition and threshold below was chosen with DR2a in view. Scripts:
`scripts/analysis_v2/hierarchy_projections.py` and `scripts/analysis_v2/lightcurve_lead_lag.py` (numpy and the
standard library only; they do not import tycho). Numbers:
[hierarchy_projections.json](v2_analysis/hierarchy_projections.json) and
[lightcurve_lead_lag.json](v2_analysis/lightcurve_lead_lag.json). Checks and independent recomputation: §7 and
[hierarchy_projections_verification.md](v2_analysis/hierarchy_projections_verification.md).

## Summary

* **Question.** Part A asked which view, θ or η, tracks behaviour better, and found little difference
  ([v2a_results.md](v2a_results.md)). This note asks something else. For each order k, the best fit to the data that
  uses interactions of at most k lags can be computed before any training. Does the training trajectory go where
  those fits say?
* **The trajectory passes close to the intermediate fits.** DR2a has 11 cases in which a fit of intermediate order
  differs from the one below it and from the truth. In them the network comes within a median of 6.8 % of the fit's
  own distance from the truth, and within 1.3 to 1.9 % in the four cleanest cases. A straight path in log-odds from
  the initial to the final network stays at a median of 58 %. The network is closer than the straight path in 11 of
  11.
* **The v1 shortcut is the order-1 fit.** For {1,2,3}:1.0 with chain contexts, the best predictor with single-lag
  terms uses lag 4 alone and leaves 0.149 nats of loss. All 10 networks come within 0.002 to 0.004 nats of it, at
  steps 96 to 125 and with a loss of 0.127 to 0.136 nats, before the planted term is learned.
* **The θ and η versions of these fits coincide in 8 of the 11 cases**, and in the other 3 neither is consistently
  the closer one. Both match the same low-order moments; they differ only in how they fill in the higher orders.
* **Order content: a curve without a threshold.** Projecting the network itself gives one number per order and
  checkpoint (6 numbers here, against 41 subset curves with gates). It needs no onset definition.
* **A lower-order term delays a higher-order one by about the inverse of the output gain at the fit where the
  network waits.** Measured delays in iid contexts are 1.48, 1.33, 1.39 and 1.75; the rule gives 1.49, 1.30, 1.33 and
  1.90 (the fourth value was computed before that delay was measured). A stricter test with no free choice,
  weighting time by the network's measured gain, removes most of the delay but not all of it: ratios of 1.27 to 1.59
  become 1.02 to 1.24.
* **Side check.** No light-curve indicator in DR2a (eNTK statistics, effective rank, distance from initialisation,
  parameter norm) leads the transition. Where they follow it at all, they move with the loss or after it.
* **Limits.** Six binary inputs, one-hidden-layer MLPs, one optimiser, post hoc. Higher-order fits are approached
  less closely (17 to 24 % at order 3), and two of the 11 fits lie so close to the truth that passing them means
  little. The delay rule does not carry over from iid to chain contexts.
* **Not new as a phenomenon.** That networks learn low-order statistics first is established (§5). What may be worth
  testing is the exact, in-advance version: plateau heights, which shortcut appears, and how long the network waits.
  §6 gives predictions for configurations that have not been run.
* **Added 17 November: a first test of the delay rule (§6.1; local pilot, exploratory).** On new seeds and with the
  predictions of §6 fixed first, the rule holds where it predicts a delay of at most 1.5 (measured 1.03, 1.10 and
  1.35 to 1.48 against 1.02, 1.09 and 1.49) and overshoots for stronger lower-order terms (predicted 6.0, measured
  2.7 to 3.6). With plain SGD in place of Adam the measured delays stay within about a quarter of the rule up to a
  predicted 6: the rule describes gradient flow, and Adam shortens the delay.
* **Added 17 November, second round (§6.2): the rule is an outcome, not a mechanism.** Under SGD the rate of the
  higher-order term factorises into a probability residual and a tangent kernel. The lower-order term cuts the
  residual by more than the rule says and raises the kernel, more so when the two terms share an input; the delay is
  the product.
* **Added 17 November, third round (§6.3): the residual part has a closed form.** A reviewer derived the residual
  factor from the source alone. It matches the measurements within 6 % up to half of the term's growth at four
  strengths, two of them run after the formula's values were on record, and is off by up to 27 % at three quarters.
  For a weak lower-order term it gives the measured delay within 1 %; for strong ones the kernel takes off up to 38 %.
* **Added 17 November, fourth round (§6.4): where the closed form bends, and the kernel's part.** At seven strengths
  the closed form holds within 6 % up to half of the term's growth. Its deviation at three quarters is traced to the
  lower-order coefficient being off its assumed value by up to 0.05. At initialisation the tangent kernel is about
  ten times larger in a first-order direction than in the term's direction, and that ratio alone reproduces where
  the lower-order coefficient is when the term starts to grow. The kernel's contribution K varies smoothly with the
  strength; the reviewer's formula for it passes its own test, and so does a straight line between neighbouring
  strengths.
* **Added 18 November, fifth round (§6.5): a second formula for K, and what its derivation rests on.** On five
  strengths that the reviewer had not run, the second formula is within 0.018 on 19 of 20 values of K and 0.039 off
  on one, which is beyond its own failure bound; outside the fitted range it follows the bend that the first
  formula missed. Of the three statements it is derived from, the conserved quantity holds; the law for the growth
  of the kernel "with no fitted constant" holds at the width of the pilots only (at half or twice that width it is
  off by up to a third); the growth of the read-out norm has the stated shape at half the stated size. What the
  conserved quantity does give, with no fitted constant, is the growth of the kernel for small coefficients at every
  width tried, and the whole curve for wide networks.

## 1. Definitions

The setting is the Markov language of sky v2a: 6 binary lags, 64 windows, the exact training distribution D of the
window (the chain's stationary distribution, or uniform for iid contexts), the exact true conditional p*, and the
network's log-odds on every window at each of 60 checkpoints. All quantities are exact population values. KL is the
divergence between Bernoulli distributions in nats; χ_B is the product of the spins of the lags in B.

* **F_k**: the predictors whose log-odds are Σ_{|B| ≤ k} w_B χ_B, that is, interactions of at most k lags with the
  output. F_0 is a constant; F_6 is every predictor.
* **Projection P_k[p]** of a conditional p onto F_k: the member of F_k that minimises Σ_c D(c) KL(p(c) ‖ q(c)). It
  is a weighted logistic regression with soft targets. In the language of the instrument, it is the m-projection of
  the joint D·p onto the hierarchical log-linear model with every input term free and input–output terms of up to k
  lags.
* **Rungs** p^(k) = P_k[p*]. **Loss left at rung k**: L_k = Σ_c D(c) KL(p* ‖ p^(k)), so L_0 is the loss of the best
  constant predictor and L_6 = 0. **Step length**: r_k = L_{k−1} − L_k = Σ_c D(c) KL(p^(k) ‖ p^(k−1)) (the
  Pythagorean relation). r_k is the refined information of the order-k input–output terms summed along any chain;
  the sum does not depend on the chain, although its single terms do (calibration E7).
* **Network** q_t at checkpoint t: loss(t) = Σ_c D(c) KL(p* ‖ q_t) and distance to rung k,
  d_k(t) = Σ_c D(c) KL(p^(k) ‖ q_t).
* **Closest approach** v_k = min_t d_k(t) / L_k: how near the network comes to rung k, relative to how far rung k is
  from the truth. The reference is the same quantity along the straight line in log-odds from the first to the last
  checkpoint, a path on which all orders are learned in proportion.
* **Intermediate rung**: 1 ≤ k ≤ 5 with L_k ≥ 0.002 and r_k ≥ 0.002 nats. Other rungs coincide with a neighbour or
  with the truth, and visiting them says nothing.
* **Order content of the network**: M_k(t) = Σ_c D(c) KL(q_t ‖ P_k[q_t]) and c_k(t) = M_{k−1}(t) − M_k(t). This is
  how much order-k structure the network itself has under D. At convergence c_k equals r_k.
* **Half-time** H(f) of a series: its first upward crossing of f times its final value (the mean of the last 5
  checkpoints), interpolated in log(step).
* **Gain** at rung k: G_k = Σ_c D(c) (1 − m_k(c)²), with m_k = 2p^(k) − 1. It is 1 for an uninformed predictor and
  falls toward 0 as the output saturates.

## 2. Results

### 2.1 Which configurations have an intermediate rung

Of the 32 configurations (16 coupling sets × 2 contexts), 11 have one intermediate rung each and 21 have none.

* **None:** the 12 single-coupling iid configurations (nothing of lower order is informative, so L_k = L_0 up to the
  planted order) and 9 chain configurations: the three single-lag sources, whose first rung is the truth; the five
  {1,3} and {1,2,4} sources, whose lower-order fits are uninformative; and {1}:2.0 + {1,2,3}:2.0, whose order-1 fit is
  already within 0.002 nats of the truth.
* **One:** the four two-coupling iid configurations, where the rung is the best fit with single-lag terms (it uses
  lag 1), and seven chain configurations, where the chain makes a lower-order set informative.

The rungs depend only on the source and the context, not on the network or the seed.

### 2.2 The trajectory passes close to the intermediate rungs

Medians and ranges are over the 10 seeds. "Terms" lists the rung's coefficients of h = log-odds / 2 that are at least
0.05, by lag set.

| couplings | context | k | terms of the rung | r_k | L_k | closest approach v (range) | straight path | step of closest approach | loss there / L_k |
|---|---|---|---|---|---|---|---|---|---|
| {1,2,3}:1.0 | chain | 1 | {4}: +0.66 | 0.179 | 0.149 | 0.019 (0.014–0.028) | 0.58 | 96–125 | 0.89 |
| {1,2,3}:2.0 | chain | 1 | {4}: +1.65 | 0.540 | 0.063 | 0.015 (0.012–0.023) | 0.80 | 162–273 | 0.85 |
| {1}:1.0 + {1,2,3}:1.0 | iid | 1 | {1}: +0.53 | 0.121 | 0.180 | 0.015 (0.009–0.024) | 0.41 | 65–125 | 0.93 |
| {1}:2.0 + {1,2,3}:2.0 | iid | 1 | {1}: +0.55 | 0.131 | 0.214 | 0.013 (0.008–0.022) | 0.39 | 65–125 | 0.93 |
| {1}:1.0 + {1,3}:0.8 | iid | 1 | {1}: +0.65 | 0.174 | 0.121 | 0.068 (0.032–0.097) | 0.60 | 65–96 | 0.79 |
| {1}:2.0 + {1,3}:1.6 | iid | 1 | {1}: +0.85 | 0.261 | 0.119 | 0.068 (0.046–0.091) | 0.90 | 74–110 | 0.71 |
| {1}:1.0 + {1,2,3}:1.0 | chain | 1 | {1}: +1.75, {3}: +0.79, {2}: −0.73, {4}: +0.20 | 0.579 | 0.0035 | 0.101 (0.069–0.165) | 0.47 | 1,149–6,267 | 0.67 |
| {1,2,3,5}:0.5 | chain | 3 | {4,5,6}: +0.22 | 0.023 | 0.088 | 0.167 (0.155–0.203) | 0.31 | 211–405 | 0.73 |
| {1}:1.0 + {1,3}:0.8 | chain | 1 | {1}: +0.79, {5}: +0.16, {2}: +0.16, {4}: +0.15, {3}: −0.06 | 0.206 | 0.054 | 0.194 (0.161–0.268) | 0.95 | 96–125 | 0.62 |
| {1,2,3,5}:1.0 | chain | 3 | {4,5,6}: +0.66 | 0.179 | 0.149 | 0.244 (0.186–0.369) | 0.56 | 312–405 | 0.49 |
| {1}:2.0 + {1,3}:1.6 | chain | 1 | {1}: +1.58, {5}: +0.28, {2}: +0.27, {4}: +0.24, … | 0.029 | 0.0037 | 0.298 (0.269–0.399) | 1.81 | 405–777 | 0.45 |

* **Over the 11 rungs** the median closest approach is 0.068 against 0.577 for the straight path. The network is
  closer than the straight path in 11 of 11, below 0.10 in 6 and below 0.25 in 10. The count of 6 depends on the
  median convention: {1}:1.0 + {1,2,3}:1.0 (chain) has its two middle seeds at 0.098 and 0.104.
* **Two rungs are not waiting states.** The order-1 rungs of {1}:1.0 + {1,2,3}:1.0 and {1}:2.0 + {1,3}:1.6 with
  chain contexts lie within 0.004 nats of the truth. The network is closest to them late (steps 405 to 6,267), with
  its loss at 0.45 to 0.67 of L_k, and for the first of them the straight path is closest at or near its end point.
  They meet the 0.002-nat rule but say little. Without them the median over the other 9 rungs is still 0.068.
* **Where it is close, the loss sits near the predicted plateau.** In the four cleanest cases the loss at the closest
  approach is 0.85 to 0.93 of L_k. It is below L_k because the next order has already started.
* **Where it is less close.** The order-3 rungs of the {1,2,3,5} chains (0.17 and 0.24: the network starts on the
  planted order-4 term before it has finished the order-3 shortcut) and the rung of {1}:1.0 + {1,3}:0.8 with chain
  contexts, which has several single-lag terms (0.19). Saturation alone does not spoil it: {1,2,3}:2.0 with chain
  contexts is among the closest.
* **Without an intermediate rung there is one transition.** The iid twin of {1,2,3}:1.0 has L_1 = L_2 = L_0 =
  0.328. In its seed-100 run the loss stays between 0.31 and 0.34 from step 5 to step 65 and then falls in a single
  transition.

### 2.3 The v1 shortcut is the order-1 rung

v1 found that chain-trained {1,2,3}:1.0 networks first learn the single-lag pair (x[t−4], x̂), reaching 79 to 96 % of
the RI of the Bayes-optimal predictor that uses x[t−4] alone, and later unlearn it under uniform probes
([v1_results.md](v1_results.md), §1). In the present terms that predictor is the order-1 rung. Its step length,
0.179 nats, and the loss it leaves, 0.149 nats, follow from the source alone.

One run (seed 100, object 0503e8c2beb9aa72):

| step | loss | distance to the order-1 rung | order-1 content | order-3 content |
|---|---|---|---|---|
| 0 | 0.366 | 0.217 | 0.008 | 0.000 |
| 20 | 0.218 | 0.073 | 0.030 | 0.000 |
| 44 | 0.157 | 0.017 | 0.099 | 0.000 |
| 65 | 0.139 | 0.004 | 0.156 | 0.001 |
| 110 | 0.128 | 0.004 | 0.198 | 0.001 |
| 312 | 0.081 | 0.016 | 0.206 | 0.013 |
| 885 | 0.011 | 0.108 | 0.192 | 0.103 |
| 4,237 | 0.001 | 0.173 | 0.188 | 0.149 |

Under D the order-1 content does not go away: the truth's own order-1 projection has that dependence on lag 4
(r_1 = 0.179). What v1 saw vanish is the function's dependence on lag 4 off the chain, under uniform probes. On the
chain, "uses lag 4" and "uses lags 1 to 3" agree about lag 4, and the network moves from the first to the second.

### 2.4 The θ and η versions of the rungs mostly coincide

The η counterpart of a rung is the D-weighted least-squares fit of the mean 2p* − 1 by a polynomial of degree at
most k. Both versions match the same moments E_D[(2ŷ − 1) χ_B] for |B| ≤ k: these are the stationarity conditions of
both fits. They differ only in what they assume above order k: no log-odds terms, or no terms in the mean. When the
rung turns out to involve a single lag set, the two are the same predictor: both reproduce the conditional mean given
that set.

Closest approach to each version, in two metrics (KL as above, and squared error on the probability), each relative
to the rung's own distance from the truth in that metric. In 8 of the 11 cases the θ and η values agree to within
0.001. The other three:

| couplings | context | KL: θ rung | KL: η rung | squared error: θ rung | squared error: η rung |
|---|---|---|---|---|---|
| {1}:1.0 + {1,2,3}:1.0 | chain | 0.101 | 0.062 | 0.172 | 0.236 |
| {1}:1.0 + {1,3}:0.8 | chain | 0.194 | 0.128 | 0.218 | 0.168 |
| {1}:2.0 + {1,3}:1.6 | chain | 0.298 | 0.343 | 0.426 | 0.440 |

Neither version is consistently closer. In these three cases the least-squares fit leaves [0, 1] on some windows
(18, 2 and 3 of the 64, carrying about 0.1 % of D), so the η rung is not a probability there. It is clipped to
[0, 1] for the squared error and to [1e-9, 1 − 1e-9] for KL, and the KL column then depends on that constant: in the
first row about 90 % of the η rung's own distance from the truth comes from the clipped windows. The squared-error
columns are the more trustworthy comparison.

On this evidence the useful object is the hierarchy by order, which both coordinate systems define together (moments
matched up to order k, log-linear interactions zero above it), and not the choice of link.

### 2.5 Order content: a curve without a threshold

c_k(t) gives, for each checkpoint, how much structure of each order the network has. In the run of §2.3 the order-1
content rises between steps 10 and 110 and the order-3 content between steps 200 and 1,500; the order-2 content stays
below 0.002 nats throughout, as it should for this source. No gate, q-value, persistence rule or subset list is
involved, and the values converge to the step lengths r_k of the data.

Times taken from these curves are fractions of the final value, so they do not depend on a detection threshold. In
iid contexts the time for a single planted term does not depend on its strength: the order-2 content of {1,3} reaches
half its final value at steps 137, 137 and 138 for strengths 0.5, 0.8 and 1.0, and the order-3 content of {1,2,3} at
373 and 375 for strengths 1.0 and 2.0. (v1's exponents moved with the onset threshold; §2 of
[v1_results.md](v1_results.md).)

### 2.6 How much a lower-order term delays a higher-order one

v1's Q6 found that a pairwise term delays a higher-order term even with iid contexts, by a factor of about 1.3 to
1.5, and that a smaller target does not explain it. The rungs suggest why, and how much.

**A rule.** While the network waits at the order-1 rung, its output is partly saturated. A term that is not yet
learned then moves the probability less per unit of log-odds, by the factor G_1. So the higher-order term should take
about 1/G_1 times as long as when it is learned alone, from an unsaturated network ("rule A"). An alternative
("rule B") is the ratio of the term's Walsh coefficients alone and with the lower-order term, the drop in its initial
gradient.

**A sketch of why** (not a derivation). Take a tangent kernel that depends on two inputs only through their inner
product, as an MLP's does at initialisation in the wide limit; with uniform inputs it is diagonal in the Walsh basis,
with eigenvalue λ_χ on the term χ. The mean m = tanh(log-odds / 2) follows dm/dt = (1 − m²)/2 · d(log-odds)/dt, and
for a term χ whose
coefficient is still small, dm_χ/dt ≈ (λ_χ/4) · G · (f*_χ − m_χ) when 1 − m² is constant over the inputs, as it is at
a single-lag rung. The term relaxes exponentially toward its target at a rate proportional to G. The time to a given
fraction of the final value therefore scales as 1/G and does not depend on the size of the target. This also gives
the strength-independence of §2.5, where G ≈ 1 while the term is small.

**Measured against the rules** (iid contexts, paired by seed, median over 10 seeds; times of the order content of
the higher-order term):

| with the lower-order term | alone | delay at 25 % / 50 % / 75 % of the final value | rule A, 1/G_1 | rule B |
|---|---|---|---|---|
| {1}:1.0 + {1,3}:0.8 | {1,3}:0.8 | 1.48 / 1.51 / 1.60 | 1.49 | 1.77 |
| {1}:1.0 + {1,2,3}:1.0 | {1,2,3}:1.0 | 1.33 / 1.38 / 1.50 | 1.30 | 1.58 |
| {1}:2.0 + {1,2,3}:2.0 | {1,2,3}:2.0 | 1.39 / 1.46 / 1.54 | 1.33 | 1.93 |
| {1}:2.0 + {1,3}:1.6 | {1,3}:1.0 (stand-in) | 1.75 / 1.90 / 2.30 | 1.90 | 2.98 |

* **Rule A is within about 0.15 of the 25 % delay in all four rows (0.005, 0.03, 0.06 and 0.15); rule B is too
  large in all four.**
* **Doubling both couplings barely changes the delay** (rows 2 and 3), although the nominal lower-order coupling
  goes from 1.0 to 2.0. The rule explains this: the order-1 rung is the projection, not the nominal term, and its
  coefficient is 0.53 against 0.55. A first guess that used the nominal coupling predicted factors of 2.4 and 14.
* **Row 4 is partly out of sample.** DR2a has no {1,3}:1.6 run, so the strongest alone run stands in, which §2.5
  supports. Rule A's 1.90 was computed before this delay was measured; rows 1 to 3 were fitted after the fact.
* **The delay grows with the fraction.** As the higher-order term grows it saturates the output further, more so
  in the run with the lower-order term.

**A stricter test, with no free choice.** If saturation were the whole delay, the planted term's uniform Walsh
coefficient would reach a given fraction of its final value at the same gain-weighted time τ = ∫ G(t) dt, with G(t)
measured on the network itself, with and without the lower-order term:

| with the lower-order term | raw time ratio at 25 % / 50 % / 75 % | gain-weighted time ratio |
|---|---|---|
| {1}:1.0 + {1,3}:0.8 | 1.46 / 1.46 / 1.48 | 1.24 / 1.17 / 1.14 |
| {1}:1.0 + {1,2,3}:1.0 | 1.27 / 1.32 / 1.36 | 1.02 / 1.06 / 1.12 |
| {1}:2.0 + {1,2,3}:2.0 | 1.33 / 1.39 / 1.44 | 1.05 / 1.12 / 1.22 |
| {1}:2.0 + {1,3}:1.6 | 1.59 / 1.69 / 1.83 | 1.24 / 1.18 / 1.21 |

Saturation accounts for most of the delay, nearly all of it for the {1,2,3} pairs, and leaves a factor of about 1.2
for the {1,3} pairs. Rule A's close match in the table above is therefore partly luck: the network has not fully
reached the rung when the higher-order term starts, which lowers the delay, and something else raises it.

**Across contexts the rule is unreliable.** Comparing chain with iid training for one coupling set is not a clean
test, because the whole context distribution differs; configurations with no intermediate rung are already 1.12 to
1.16 times slower with chain contexts ({1,3}:1.0 and {1,2,4}:1.0, at 25 %). For the planted term's uniform Walsh
coefficient, chain time over iid time at 25 %, against 1/G at the chain rung:

| couplings | chain rung | measured (range) | 1/G at the rung |
|---|---|---|---|
| {1,2,3}:2.0 | 1 | 7.32 (6.34–8.17) | 7.34 |
| {1,2,3,5}:0.5 | 3 | 1.02 (0.96–1.12) | 1.05 |
| {1,2,3}:1.0 | 1 | 2.17 (1.82–2.43) | 1.51 |
| {1,2,3,5}:1.0 | 3 | 1.09 (1.03–1.13) | 1.51 |

Two of four are close and two are off by roughly 30 to 45 %. The {1,2,3}:2.0 chain runs have not converged off the
chain by step 30,000: the coefficient's last five checkpoints average 0.91 of its asymptote tanh 2 (0.86 to 0.95
across seeds) and are still rising. Measured against the asymptote instead of the run's own final value, that ratio
is 7.60; the other three stay at 2.18, 1.09 and 1.02.

## 3. Side check: do the light-curve indicators lead the transition?

The review asked whether a cheap indicator moves before the function changes, by a constant factor, so that it could
announce an interaction before RI or the loss sees it. Tested on the 12 single-coupling iid configurations, which
have one clean transition whose time moves more than 20-fold with order and strength. For each run and indicator, the
step at which the indicator has completed a fraction of its total change is divided by the step at which the exact
loss under D has completed the same fraction of its fall. A precursor has a ratio well below 1, the same in every
configuration.

Ratios at 25 % in the five configurations where the indicators follow the transition at all (median over seeds):

| couplings | loss, step at 25 % | eNTK alignment | eNTK effective rank | eNTK trace | distance from init | parameter norm | hidden effective rank |
|---|---|---|---|---|---|---|---|
| {1,2,3}:1.0 | 145 | 1.75 | 1.26 | 1.02 | 1.24 | 1.01 | 45.8 |
| {1,2,4}:1.0 | 139 | 1.71 | 1.27 | 0.99 | 1.28 | 0.99 | 47.7 |
| {1,2,3}:2.0 | 157 | 1.61 | 1.18 | 1.41 | 1.13 | 1.51 | 3.29 |
| {1,2,3,5}:1.0 | 167 | 1.56 | 1.25 | 1.18 | 1.13 | 1.15 | 13.5 |
| {1,2,3,5}:0.5 | 152 | 2.34 | 2.09 | 0.43 | 1.80 | 72.1 | 51.8 |

* **No indicator leads.** Over the 96 cells (12 configurations × 8 indicators) no ratio is below 0.8 at the 10 %
  level (smallest 0.87), one is at 25 % (the eNTK trace of {1,2,3,5}:0.5) and four are at 50 % (smallest 0.75).
* **Those five cells are not precursors.** Each is a series that rises and falls back, with a net change of 0.06 to
  0.22 of its range, so "the fraction of its change completed" is ill-defined for it. The 0.43 cell has per-seed
  ratios from 0.11 to 55, and the sign of its net change differs between seeds. The counts move with the convention:
  with the loss time taken on the same 15 checkpoints as the eNTK series, one cell is below 0.8 at 25 % and one at
  50 %; with the eNTK final value taken at its last evaluation alone, none at 25 % and three at 50 %.
* **Where the indicators follow the transition, they move with the loss or after it.** In the four configurations
  of the table with strength at least 1.0, the loss has fallen by half at steps 214 to 258. The distance from
  initialisation changes fastest at steps 292 to 333, and the eNTK trace and effective rank between the eNTK
  evaluations at steps 273 and 525.
* **For low orders and weak couplings they are not event markers at all.** They keep drifting for thousands of
  steps after the function has converged (ratios in the hundreds).
* **Limits.** The eNTK series exist at 15 of the 72 evaluation steps, a factor of about 1.9 apart, so ratios between
  0.5 and 2 are not resolved for them; the other indicators are stored at every evaluation. At the 10 % level the
  loss time of the weak couplings falls in the initial transient (for {1,2,3,5}:0.5 it ranges from 3 to 87 steps
  across seeds), so the 25 % level is the cleaner one. This tests the eight stored indicators on these MLPs only. A
  weight-space progress measure aimed at the planted lags, which the light curves do not store, is not ruled out.

## 4. Limits

* **Exploratory and post hoc.** The 0.002-nat rule for an intermediate rung, the straight reference path, the
  half-time fractions and the twin comparisons were all chosen after DR2a was read. Only rule A's value for the
  fourth twin was fixed before its measurement.
* **A very small setting.** Six binary inputs, 64 windows, MLPs of width 128 and depth 1, Adam at one learning rate,
  two task families of which only the Markov language is analysed. Nothing here says the picture holds for larger
  inputs, categorical tokens, transformers or other optimisers.
* **A first look at a transformer goes against the strong form** (added later on 16 November; quick check, not
  recomputed independently: `scripts/analysis_v2/discussion_checks.py`). For Part B's tiled induction task the
  solutions that use no content matching can be computed from the data: they cost 1.56 to 1.96 nats, and the 1-layer
  controls of DR2b-pilot2 end at 1.65 to 1.89. The 2-layer networks follow the 1-layer path only up to a loss of
  3.2 to 3.4 and then form the circuit; they spend at most one evaluation between 1.6 and 2.3 nats. So candidate
  waiting states can be computed in advance, but whether a network waits at one depends on how fast it learns that
  state relative to the next, which this note does not predict. On these MLPs the lower order is learned faster; in
  the attention-only transformer the content-free solution is learned more slowly than the circuit. See
  `notes/discussion_2026-11-16_outside_reviews.md`, §5.4 and §6.
* **Exact projections need the truth.** L_k and the rungs are computed from the known source. For real data the
  hierarchy has to be estimated or approximated, for example by n-gram or k-body models.
* **The picture is weaker at higher orders and for rungs with several terms.** Order-3 rungs are approached to 17
  to 24 %, the multi-term rung of {1}:1.0 + {1,3}:0.8 (chain) to 19 %, and two rungs lie so close to the truth that
  passing them means little (§2.2). That leaves six clean cases out of eleven.
* **"Closer than a straight line" is a low bar.** That a gradient-trained network learns an informative single-lag
  term before a three-lag term is what a simplicity bias predicts. The content is in the numbers: how close, which
  term, at what loss.
* **The delay rule rests on four twin comparisons** in iid contexts, three of them fitted after the fact, and it
  fails as a cross-context rule (§2.6). A local pilot has since tested it on new seeds and configurations (§6.1): it
  holds for weak lower-order terms and, under Adam, not for strong ones.
* **Time resolution.** The 60 checkpoints are a factor of 1.14 apart, with twelve gaps of about 1.3 from step 74
  on. Half-times are interpolated in log(step), and several of them fall next to a gap. A delay of 1.3 to 1.5 is two
  to three grid steps, so the second decimal of the delays is not resolved.
* **Seeds are a pairing, not independent replicates.** A seed has the same initial network in every configuration
  (the log-odds at step 0 are bit-identical across all 32), which is what makes the twin ratios paired. All delays
  are medians of per-seed ratios; no interval or test is attached to them.
* **Eleven rungs, seven sources.** The configurations were chosen for Part A, not for this question.

## 5. Earlier work

That networks learn lower-order statistics of their data before higher-order ones is established under the name
distributional simplicity bias: Refinetti, Ingrosso and Goldt (ICML 2023); Belrose et al. (ICML 2024), who evaluate
Pythia checkpoints on samples from maximum-entropy n-gram models; Rende, Gerace, Laio and Goldt (NeurIPS 2024), who
train transformers on clones of text with interactions up to a chosen order. A maximum-entropy model with matched
moments up to order k is the m-projection onto the order-k log-linear family, so the rungs here are the same
objects. Staged learning with saddle-to-saddle dynamics is analysed by Abbe, Boix-Adserà and Misiakiewicz (COLT
2023), and Kunin et al. (NeurIPS 2025) predict the order, timing and size of loss drops in two-layer networks.
Slower learning of further features once the output is partly fitted under cross-entropy is gradient starvation
(Pezeshki et al., NeurIPS 2021); rule A may already follow from their analysis. None of these papers is cited
elsewhere in this repository yet.

No literature search was done for this note beyond confirming that these papers exist and what their abstracts
claim. Before anything here is presented as new, the overlap has to be checked, in particular for the plateau
heights (§2.2) and the gain rule (§2.6).

## 6. A prospective test

The claims of §2 can be tested on configurations that have not been run, with the predictions fixed first. The
values below come from the source alone (`predictions_unrun` and `window3_chain` in the JSON). They are a proposal;
if the project adopts the test, the criteria and margins belong in its own pre-registration.

**Delay sweep (iid contexts; alone twin {1,3}:0.8 or {1,2,3}:1.0, same seeds).** Predicted plateau L_1 and delay of
the higher-order term at 25 % of its final order content:

| configuration | L_0 | plateau L_1 | rule A | rule B |
|---|---|---|---|---|
| {1}:0.25 + {1,3}:0.8 | 0.240 | 0.230 | 1.02 | 1.04 |
| {1}:0.5 + {1,3}:0.8 | 0.242 | 0.201 | 1.09 | 1.15 |
| {1}:1.5 + {1,3}:0.8 | 0.417 | 0.057 | 2.69 | 3.53 |
| {1}:2.0 + {1,3}:0.8 | 0.538 | 0.023 | 6.02 | 8.35 |
| {1}:3.0 + {1,3}:0.8 | 0.658 | 0.0033 | 39.8 | 57.1 |
| {4}:1.0 + {1,3}:0.8 | 0.295 | 0.121 | 1.49 | 1.77 |
| {4}:2.0 + {1,3}:0.8 | 0.538 | 0.023 | 6.02 | 8.35 |
| {1}:2.0 + {1,2,3}:1.0 | 0.502 | 0.038 | 4.38 | 6.52 |
| {1}:3.0 + {1,2,3}:1.0 | 0.647 | 0.0056 | 27.5 | 43.2 |

* The sweep spans predicted delays from 1.0 to 40, where the existing four lie between 1.3 and 1.9.
* The {4} rows put the lower-order term on a lag outside the higher-order set. Both rules predict the same delay as
  for {1}; a mechanism based on competition for shared inputs would not.
* Given §2.6, rule A should overshoot at the strong couplings (the network needs time to reach a strongly saturated
  rung). The gain-weighted time of §2.6 is the sharper statement: its ratio should stay near 1, within the 1.0 to
  1.25 seen so far.

**Removing the shortcut without touching the source (chain contexts).** With `window: 3` the network cannot see lag
4, and the chain is unchanged. The rungs become L_0 = L_1 = L_2 = 0.328 and L_3 = 0 for {1,2,3}:1.0, against
0.328, 0.149, 0.149, 0 with the 6-lag window (0.603 throughout, against 0.603, 0.063, 0.063, 0, for {1,2,3}:2.0).
Predicted: no plateau near 0.149 (or 0.063), one transition, and an earlier planted term than in the 6-lag chain
run. By how much is an open question, since the gain rule did not carry over to chain contexts in §2.6.

**Intervening on the mechanism (needs a loss option in the task).** The gain factor comes from the cross-entropy
link. With a squared-error loss on a linear output, each Walsh mode of the mean relaxes on its own in the kernel
picture, with no factor 1 − m². The twins of §2.6, trained that way, should show the delay fall from 1.3–1.9 to what
the gain-weighted analysis leaves, about 1.0 to 1.25. This is the one test here that intervenes on the cause rather
than on the data, and the fixed seeds make it a paired comparison: same initial weights and data, only the loss
changed.

### 6.1 First outcome: a local pilot of the delay sweep (17 November)

Status: exploratory. Local runs with the repository's runner (`scripts/analysis_v2/delay_sweep_pilot.py` →
[delay_sweep_pilot.json](v2_analysis/delay_sweep_pilot.json)); seeds 600 to 607, outside every registered range;
sky v2a's Markov settings, 30,000 steps, pattern mode at 60 checkpoints; 200 runs. Not pre-registered by the project
and not recomputed independently. The table of predictions above was committed on 16 November, before these runs.
A second arm with plain SGD (learning rate 0.03, no momentum) was added because a reviewer proposed that the part of
the delay the gain does not explain is an effect of Adam's second-moment estimate and should vanish under SGD. The
rules' values are the same for SGD, but they were written with Adam in mind, so the SGD rows are not predictions on
record.

Delay of the higher-order term's order content: the median over 8 seeds of the per-seed ratio "with the lower-order
term" over "alone", at 25 / 50 / 75 % of the final value. Gain-weighted: the same ratio for the term's Walsh
coefficient in gain-weighted time (§2.6).

| with | rule A | Adam | SGD | gain-weighted, Adam | gain-weighted, SGD |
|---|---|---|---|---|---|
| {1}:0.25 + {1,3}:0.8 | 1.02 | 1.03 / 1.03 / 1.05 | not run | 1.03 / 1.02 / 1.02 | not run |
| {1}:0.5 + {1,3}:0.8 | 1.09 | 1.10 / 1.11 / 1.11 | 1.11 / 1.14 / 1.18 | 1.08 / 1.04 / 1.05 | 1.02 / 1.04 / 1.07 |
| {1}:1.0 + {1,3}:0.8 (DR2a twin, new seeds) | 1.49 | 1.47 / 1.51 / 1.60 | 1.50 / 1.66 / 1.77 | 1.19 / 1.14 / 1.15 | 1.10 / 1.12 / 1.21 |
| {1}:1.5 + {1,3}:0.8 | 2.69 | 1.88 / 2.12 / 2.54 | 2.41 / 2.77 / 3.21 | 1.19 / 1.07 / 1.07 | 1.16 / 1.12 / 1.20 |
| {1}:2.0 + {1,3}:0.8 | 6.02 | 2.73 / 3.19 / 3.90 | 4.43 / 5.58 / 7.25 | 1.27 / 1.05 / 0.98 | 1.34 / 1.07 / 1.19 |
| {4}:1.0 + {1,3}:0.8 | 1.49 | 1.35 / 1.48 / 1.56 | 1.59 / 1.74 / 1.86 | 1.10 / 1.08 / 1.14 | 1.13 / 1.16 / 1.26 |
| {4}:2.0 + {1,3}:0.8 | 6.02 | 3.03 / 3.55 / 3.85 | 5.99 / 7.09 / 8.67 | 1.22 / 1.08 / 1.04 | 1.39 / 1.29 / 1.39 |
| {1}:1.0 + {1,2,3}:1.0 (DR2a twin, new seeds) | 1.30 | 1.27 / 1.35 / 1.42 | 1.49 / 1.56 / 1.65 | 0.97 / 1.00 / 1.07 | 1.09 / 1.15 / 1.22 |
| {1}:2.0 + {1,2,3}:1.0 | 4.38 | 2.77 / 3.12 / 3.82 | 4.81 / 5.56 / 6.60 | 0.78 / 0.78 / 0.88 | 0.93 / 1.15 / 1.39 |

* **Weak lower-order terms: the rule holds.** §6 stated the prediction at 25 %. There the three configurations
  predicted on record with a rule A of at most 1.5 give 1.03, 1.10 and 1.35, against 1.02, 1.09 and 1.49. The third,
  with the lower-order term on lag 4, is 9 % low at 25 % and within 1 % at 50 %. The two DR2a twins give 1.47 and
  1.27 on the new seeds (DR2a: 1.48 and 1.33).
* **Strong lower-order terms: it overshoots under Adam**, as §6 expected. At 25 %: 2.69 → 1.88, 6.02 → 2.73 and
  3.03, 4.38 → 2.77. Rule B overshoots more in every row.
* **Under SGD the rule holds much further.** The SGD delays are within −26 % and +15 % of rule A at 25 %, and
  within −7 % and +27 % at 50 %, over the whole table. Paired by seed, the delay at 50 % under SGD is 1.01, 1.08,
  1.29 and 1.72 times the delay under Adam as the lower-order strength goes from 0.5 to 2.0 (8 of 8 seeds above 1
  from 1.5 on). The sketch behind rule A is a gradient-flow argument. Adam's normalisation of the step makes up for
  part of the lost gain; it does not add delay.
* **Gain-weighted time.** §6 expected the ratio to stay within 1.0 to 1.25. For the {1,3} twins under Adam it is 0.98
  to 1.27. It fails for {1}:2.0 + {1,2,3}:1.0 (0.78 to 0.88: in gain-weighted time the term arrives early). Under
  SGD it is 1.02 to 1.39 with one value of 0.93: the part of the delay that the gain does not explain is there
  without Adam.
* **The lag of the lower-order term matters a little, and against "competition for shared inputs".** Paired by seed,
  the half-time with the lower-order term on lag 4 over the same with lag 1 is, at 25 / 50 / 75 %: under SGD 1.03 /
  1.07 / 1.05 at strength 1.0 and 1.35 / 1.27 / 1.20 at strength 2.0 (8 of 8 seeds above 1); under Adam 0.91 / 0.92
  / 0.98 at strength 1.0 (7 of 8 seeds below 1 at 25 and 50 %) and 1.07 / 1.10 / 1.00 at strength 2.0. Neither
  rule has a lag dependence. A lower-order term on an input that the higher-order term also uses delays it less
  under SGD, not more.
* **Not usable.** The three configurations with {1}:3.0 leave 0.003 to 0.006 nats for the higher-order term. The
  seeds disagree by more than a factor of two for {1,3} (both optimisers), the term's content ends 13 % short of its
  target in one Adam seed for {1,2,3}, and the SGD runs of that twin have not converged in 30,000 steps. The JSON
  marks them `reliable: false`; rule A's 28 and 40 are untested.
* **Limits.** One width and depth, one learning rate per optimiser, iid contexts, eight seeds that share initial
  networks across configurations (a pairing, as in §4), local runs, no independent recomputation. The squared-error
  intervention proposed above has not been done; the SGD arm shows that the optimiser is a second place to
  intervene.

### 6.2 Second round: what the delay is made of (17 November)

Status as in §6.1: exploratory, local runs, not recomputed independently. A reviewer answered §6.1 with proposed
mechanisms and predictions; the second round of `scripts/analysis_v2/delay_sweep_pilot.py` (64 further runs) and
`scripts/analysis_v2/kernel_gate_pilot.py` (40 runs → [kernel_gate_pilot.json](v2_analysis/kernel_gate_pilot.json))
test them. The predictions and their outcomes are in `notes/discussion_2026-11-16_outside_reviews.md`, §10.5 and
§10.6; this section keeps what bears on the delay rule.

Notation, for the term {1,3} under plain SGD: F is the coefficient of the log-odds in the term's Walsh direction χ
(its target is the term's strength), λ = ‖∇_θ F‖² is the tangent kernel in that direction, ρ = E[(p* − q) χ] is the
probability residual in it. Under gradient flow dF/dt = lr · λ ρ up to cross terms; the exact population rate over
λ ρ is 0.85 to 1.07 here. So at equal progress F two runs differ in rate by a residual factor and a kernel factor.

| lower-order strength 2.0; at 10 / 25 / 50 / 75 % of the target | with {1} (shared input) | with {4} |
|---|---|---|
| steps, over the alone twin's | 6.3 / 5.6 / 6.6 / 8.2 | 6.2 / 6.9 / 8.4 / 10.5 |
| residual, alone over with | 9.1 / 9.8 / 11.6 / 14.4 | 9.1 / 9.8 / 12.7 / 17.6 |
| kernel, alone over with | 0.54 / 0.61 / 0.77 / 0.85 | 0.78 / 0.85 / 0.93 / 0.96 |

At strength 1.0 the residual factor is 1.9 to 2.6 (shared) and 1.9 to 2.9, the kernel factor 0.88 to 0.97 and 0.95
to 0.99. Rule A is 1.49 and 6.02 for the two strengths, rule B 1.77 and 8.35.

* **The output side costs more than rule A.** The residual factor starts at about rule B (1.9 against 1.77, 9.1
  against 8.35) and grows as the term is learned.
* **The kernel works the other way.** The lower-order term does not lower the kernel in the {1,3} direction, it raises
  it: by a factor of up to 1.85 when the lower-order term is on lag 1, up to 1.28 when it is on lag 4.
* **So the delay rule of §2.6 is the net of two factors of opposite sign.** That it comes out within a quarter of
  1 / gain under SGD (§6.1) is an outcome. It should not be stated as the mechanism, and "what the gain does not
  explain" (the 1.02 to 1.39 of §6.1) is not a kernel loss.
* **Why the lag matters.** At step 150 the kernel of the lag-1 twin is 1.37 times that of the lag-4 twin at strength
  2.0 (8 of 8 seeds, 1.29 to 1.43; the reviewer's prediction was at least 1.20) and 1.08 times at strength 1.0. The
  kernel's part from the first-layer weights on s_3 equals Σ_j a_j² E[gate_j · s_1]²: a unit turns a weight on s_3
  into s_1 s_3 to the extent that its gate is correlated with s_1. At 10 % of the target that part is 0.020 alone,
  0.075 with {4} and 0.203 with {1}. The reviewer's stronger form, units whose gate is set by the lower-order input
  and are dead for the term otherwise, does not happen: no unit's gate is a function of s_1 alone or of s_4 alone in
  any run, and the share of gates that flip with s_3 stays at its initial 0.26 to 0.27.
* **Two lower-order terms.** Under SGD the order-3 term {1,2,3}:1.0 is delayed by 3.29 (at 50 %) with {1}:1.0 and
  {2}:1.0, and by 1.52 with {4}:1.0 and {5}:1.0; paired by seed the second half-time is 0.46 times the first (0 of 8
  seeds above 1). The order-1 rungs differ: with the lower-order terms on the term's own lags the best first-order
  fit also uses lag 3 (log-odds coefficient −1.22), leaves 0.092 nats instead of 0.183, and rule A is 2.17 against
  1.63. The reviewer had predicted the opposite order; the analyst had taken the two rules to be equal.
* **Adam's second moment.** On {1}:2.0 + {1,3}:0.8 the delay at 50 % is 3.19 with β₂ = 0.999, 3.23 with 0.9999, 2.19
  with 0.9, and 1.74 with β₂ = 0.9 and a batch of 4,096 (SGD: 5.58). A second moment that follows the gradient faster,
  and a less noisy gradient, shorten the delay; a longer memory does not lengthen it.
* **Limits.** One term, one width, SGD at one learning rate for the factorisation, eight seeds; the progress variable
  here is the log-odds coefficient, in §6.1 it is the order content, and the two give different delays (6.6 against
  5.6 at 50 % for the lag-1 twin at strength 2.0).

### 6.3 Third round: a closed form for the output side (17 November)

Status as in §6.1. The reviewer's third reply gave a closed form for the residual factor of §6.2;
`scripts/analysis_v2/kernel_gate_pilot.py` computes it and compares it with the runs, including two strengths (0.5
and 1.5) that were run after the formula's values were on record. Record: `notes/discussion_2026-11-16_outside_reviews.md`,
§11.3. The derivation is the reviewer's; it was checked here and its numbers reproduced.

Take the log-odds to have two coefficients, z = F_1 ψ + F χ, with ψ the lower-order input, χ the term's direction and
targets F_1* = 2 β_1, F* = 2 β_2. The means along ψ and χ are m_1 = sinh F_1 / (cosh F_1 + cosh F) and
m_2 = sinh F / (cosh F_1 + cosh F). Assume that while F grows, F_1 stays where the first-order residual vanishes
(m_1 = m_1*). Then at progress r = F / F*

    ρ_with(r) = [m_2* − sinh(r F*) / D(r)] / 2,   D(r) = [cosh(r F*) + √(1 + m_1*² sinh²(r F*))] / (1 − m_1*²),
    ρ_alone(r) = [tanh(F*/2) − tanh(r F*/2)] / 2.

* **The two rules of §2.6 are two numbers of this solution.** Rule A is 1 / (1 − m_1*²), the ratio of the slopes of
  the two residuals at the plateau. Rule B is ρ_alone(0) / ρ_with(0), the ratio of the residuals there. Since B
  exceeds A, the residual factor ρ_alone / ρ_with starts at rule B and grows; it ends at
  (1 + cosh F_1* cosh F*) / (1 + cosh F*).
* **Measured residual factor over the closed form** (SGD, medians over 8 seeds, at 10 / 25 / 50 / 75 % of the term):

  | lower-order strength | rule A / rule B / end | closed form | with {1}: measured / closed | with {4}: measured / closed |
  |---|---|---|---|---|
  | 0.5 (values on record first) | 1.09 / 1.15 / 1.39 | 1.16 / 1.18 / 1.22 / 1.29 | 1.01 / 1.00 / 1.00 / 0.99 | 1.01 / 1.00 / 0.99 / 0.99 |
  | 1.0 | 1.49 / 1.77 / 2.99 | 1.82 / 1.92 / 2.16 / 2.52 | 1.03 / 1.00 / 0.98 / 1.01 | 1.03 / 1.01 / 1.00 / 1.17 |
  | 1.5 (values on record first) | 2.69 / 3.54 / 7.53 | 3.69 / 4.02 / 4.84 / 6.03 | 1.05 / 1.00 / 0.98 / 1.15 | 1.05 / 1.01 / 1.02 / 1.27 |
  | 2.0 | 6.02 / 8.35 / 19.95 | 8.81 / 9.78 / 12.21 / 15.68 | 1.04 / 1.00 / 0.95 / 0.92 | 1.03 / 1.01 / 1.04 / 1.12 |

  Within 6 % up to 50 %. At 75 % the deviations reach +27 %, and the reviewer's own bound for strength 1.5 (10 %,
  failure beyond 15 %) is missed there (+21 % for the mean of the two lags). At 10 % the measured values are 3 to
  5 % high: the lower-order coefficient is still 0.14 to 0.31 below the assumed value, because the lower-order term
  has not reached the plateau when the higher-order one is at 10 %. From 25 % on the lower-order coefficient follows
  the closed form within 0.04.
* **The delay as a product.** If the kernel were the same in both twins, the delay would be
  D(r) = ∫_0^r dr′ / ρ_with ÷ ∫_0^r dr′ / ρ_alone, from the closed form. The measured ratio of steps over D is the
  kernel's contribution K:

  | lower-order strength | D at 10 / 25 / 50 / 75 % | K with {1} | K with {4} |
  |---|---|---|---|
  | 0.5 | 1.16 / 1.16 / 1.19 / 1.23 | 1.00 / 0.99 / 0.99 / 0.99 | 0.99 / 0.99 / 0.99 / 1.01 |
  | 1.0 | 1.80 / 1.84 / 1.96 / 2.17 | 0.97 / 0.92 / 0.92 / 0.91 | 0.95 / 0.96 / 0.97 / 0.96 |
  | 1.5 | 3.61 / 3.77 / 4.17 / 4.87 | 0.87 / 0.76 / 0.78 / 0.81 | 0.84 / 0.87 / 0.89 / 0.90 |
  | 2.0 | 8.58 / 9.03 / 10.22 / 12.29 | 0.73 / 0.62 / 0.65 / 0.67 | 0.72 / 0.77 / 0.82 / 0.85 |

  For the weak lower-order term the source alone gives the delay within 1 %. The stronger the lower-order term, the
  more the kernel takes off, and more so when the input is shared. So under SGD the delay is D × K, with D computed
  from the source before training and K the part that belongs to the learner; K is measured here, not predicted.
* **Limits.** One term, one width, SGD at one learning rate, eight seeds; progress is the log-odds coefficient (in
  §6.1 it is the order content); the closed form assumes two coefficients and a lower-order coefficient that is
  always in place.

### 6.4 Fourth round: where the closed form bends, the start, and the kernel's part (17 November)

Status as in §6.1. The reviewer's fourth reply (to commit `f6dedbf`) explained the two deviations of §6.3 and proposed
a formula for K. `scripts/analysis_v2/kernel_gate_pilot.py` now runs three more strengths (1.25, 1.75 and 2.5; 48
runs) and makes three checks. Record: `notes/discussion_2026-11-16_outside_reviews.md`, §12.3.

* **Seven strengths.** Up to half of the term's growth the measured residual factor is within 6.2 % of the closed
  form at all seven strengths, for factors between 1.16 and 32. The three new rows (as in §6.3: at 10 / 25 / 50 /
  75 %):

  | lower-order strength | rule A / rule B / end | closed form | with {1}: measured / closed | with {4}: measured / closed |
  |---|---|---|---|---|
  | 1.25 | 1.93 / 2.43 / 4.70 | 2.52 / 2.70 / 3.16 / 3.83 | 1.04 / 1.00 / 0.99 / 0.94 | 1.04 / 1.01 / 1.04 / 1.05 |
  | 1.75 | 3.94 / 5.35 / 12.22 | 5.62 / 6.19 / 7.62 / 9.67 | 1.04 / 1.00 / 0.97 / 1.10 | 1.05 / 1.00 / 1.06 / 1.15 |
  | 2.5 | 15.11 / 21.46 / 53.75 | 22.74 / 25.45 / 32.27 / 41.93 | 1.04 / 1.01 / 0.97 / 0.81 | 1.04 / 1.00 / 1.01 / 1.01 |

* **Why it bends at 75 %.** The ratio measured / closed form is the product of three factors, each measured: (a) the
  alone twin's residual over its one-coefficient formula; (b) the effect of the lower-order coefficient not being at
  the value the closed form assumes; (c) the with twin's two-coefficient formula, evaluated at its measured
  coefficients, over its exact residual. Ranges over the seven strengths and two lags (medians over seeds):

  | progress | (a) alone twin | (b) lower-order coefficient off | (c) with twin, other directions |
  |---|---|---|---|
  | 10 % | 1.00 | 1.00 to 1.05 | 1.00 to 1.01 |
  | 25 % | 1.00 | 0.98 to 1.01 | 1.00 to 1.02 |
  | 50 % | 1.01 | 0.92 to 1.02 | 0.99 to 1.03 |
  | 75 % | 1.03 | 0.71 to 1.18 | 0.98 to 1.12 |

  The two-coefficient picture holds to about 10 %. What fails at 75 % is the assumption that the lower-order
  coefficient sits exactly where its own residual vanishes. It is off by up to 0.05, behind in some twins and
  ahead in others (further ahead when the input is shared), and the residual is by then so small that this is
  amplified up to 30 %. This is the reviewer's first explanation, and it is larger than the reviewer estimated (6 to
  11 %). Its second explanation, log-odds in the other 62 Walsh directions, is the smaller part: factors (a) and (c).
* **The start.** The tangent kernel at initialisation, by the order of the Walsh direction (median over 8 seeds of
  the mean within the order): 22.3, 4.06, 0.40, 0.024, 0.024, 0.019, 0.019 for orders 0 to 6. In the lower-order
  direction it is 10.2 times (range over seeds 8.5 to 14.2; input {1}) and 10.7 times (7.8 to 13.6; input {4}) the
  kernel in the term's direction; the cross terms between the two directions are within ±0.13 of the root of the
  product. The reviewer's start correction took the two kernels to be equal; in none of its readings does it
  reproduce the measured coefficient. Following the two coefficients from zero under a constant diagonal kernel
  with the measured ratio does, with no fitted constant. Lower-order coefficient when the term is at 10 %:

  | lower-order strength | measured, with {1} / with {4} | closed form (assumes it is in place) | constant kernel from the initialisation | equal kernels |
  |---|---|---|---|---|
  | 0.5 | 0.45 / 0.45 | 0.59 | 0.45 / 0.46 | 0.08 |
  | 1.0 | 1.05 / 1.04 | 1.31 | 1.10 / 1.12 | 0.24 |
  | 1.25 | 1.42 / 1.43 | 1.72 | 1.51 / 1.53 | 0.41 |
  | 1.5 | 1.91 / 1.86 | 2.16 | 1.97 / 1.99 | 0.67 |
  | 1.75 | 2.41 / 2.39 | 2.63 | 2.44 / 2.46 | 1.07 |
  | 2.0 | 2.91 / 2.96 | 3.10 | 2.93 / 2.95 | 1.61 |
  | 2.5 | 3.91 / 3.95 | 4.08 | 3.92 / 3.94 | 2.77 |

  The closed form is 0.13 to 0.31 too high there; the constant-kernel path is within −0.01 to +0.13. From 25 % on
  the closed form is the better of the two (within 0.03; the constant-kernel path is 0.005 to 0.05 too low). Caveat:
  the network's log-odds are not zero at initialisation (the term's coefficient starts between −0.03 and +0.08), so
  the 10 % point (0.16) is noisy per seed; the table gives medians. The separation by a factor of ten is what makes
  "the lower-order coefficient is in place" a good assumption after the start. That on binary inputs every Walsh
  direction is an eigenfunction of the kernel of a wide network, with an eigenvalue that depends only on its order,
  is known (Yang and Salman 2019); what is used here is the measured ratio for this network.
* **The kernel's contribution K.** The reviewer's formula is K = 1 / (1 + α F_1(0)²), with F_1(0) the lower-order
  coefficient at the start of the closed form and α fitted, per lag and progress, to the four strengths of §6.3
  (0.064 and 0.032 at 25 %, 0.0575 and 0.023 at 50 %, for inputs {1} and {4}). Its prediction for 1.25 and 1.75 was on
  record before those runs; 2.5 was added here as a point outside the fitted range, together with the plainest
  alternative, a straight line in the strength through the two nearest fitted strengths (computed from §6.3's
  table).

  | lower-order strength | K at 25 % and 50 %, with {1}: measured / formula / line | with {4}: measured / formula / line |
  |---|---|---|
  | 1.25 | 0.846 / 0.842 / 0.842 and 0.837 / 0.856 / 0.849 | 0.913 / 0.914 / 0.913 and 0.939 / 0.937 / 0.930 |
  | 1.75 | 0.695 / 0.695 / 0.692 and 0.710 / 0.718 / 0.714 | 0.820 / 0.820 / 0.817 and 0.867 / 0.864 / 0.859 |
  | 2.5 | 0.489 / 0.486 / 0.487 and 0.521 / 0.512 / 0.510 | 0.697 / 0.654 / 0.668 and 0.772 / 0.724 / 0.756 |

  On the eight predicted cells the formula's mean absolute error is 0.005 and its largest 0.018, inside the
  reviewer's bounds (0.025 and 0.040). The straight line gives 0.005 and 0.011: the predicted strengths lie between
  fitted ones, so this test does not tell the formula from interpolation. At 2.5 the formula holds for the shared
  input (0.003 and 0.008) and is 0.043 and 0.047 too low for the other; the line's errors are 0.002 to 0.029. So K is
  a smooth function of the strength that can be interpolated; a quadratic law in F_1(0) describes the shared-input
  twin up to 2.5 and bends for the other. The ratio of the two α that the reviewer gave as 2 and 2.5 is a fitted
  number: least squares on the four strengths give 0.9, 2.0, 2.5 and 2.6 at 10, 25, 50 and 75 %.
* **Limits.** As in §6.3. The start uses the kernel at initialisation only, while the kernel in the term's direction
  grows by a factor of three during the term's growth; the split into three factors is exact per seed but is
  reported as medians; no alternative with a derivation was tested against the formula for K.

### 6.5 Fifth round: a second formula for K, and what its derivation rests on (18 November)

Status as in §6.1. The reviewer's fifth reply (to commit `4b44e01`) gave a second formula for K with a derivation, and
values of K for five strengths that it stated it had not run. `scripts/analysis_v2/kernel_gate_pilot.py` now runs
them (0.25, 0.35, 2.25, 2.75 and 3.0; 80 runs), and `scripts/analysis_v2/kernel_law_pilot.py` (new; 64 runs →
[kernel_law_pilot.json](v2_analysis/kernel_law_pilot.json)) tests the three statements the derivation rests on.
Record: `notes/discussion_2026-11-16_outside_reviews.md`, §13.3.

* **The formula.** S = 2 λ1 (√(1 + F_1(0)² / λ1) − 1), with λ1 = 4.06 the tangent kernel at initialisation in a
  first-order direction. Then K = 1 / (1 + b_un S) when the lower-order term does not share an input with the
  term, and K = 1 / (1 + b_un S / 2 + b_al F_1(0)²) when it does. The two constants are fitted per progress to
  strengths 0.5 to 2.0. For a small F_1(0), S is F_1(0)² and this is the first formula; for a large one S grows
  linearly.
* **The blind test.** K at 25 % and 50 %: measured / second formula / a straight line through the two nearest
  strengths measured before (2.0 and 2.5). The reviewer's own comparison line goes through 1.5 and 2.0; both lines
  were fixed before the runs.

  | lower-order strength | with {1}: measured / formula / line | with {4}: measured / formula / line |
  |---|---|---|
  | 2.25 | 0.547 / 0.558 / 0.556 and 0.575 / 0.580 / 0.583 | 0.730 / 0.732 / 0.732 and 0.784 / 0.792 / 0.798 |
  | 2.75 | 0.430 / 0.444 / 0.421 and 0.473 / 0.465 / 0.458 | 0.649 / 0.658 / 0.661 and 0.745 / 0.729 / 0.745 |
  | 3.0 | 0.393 / 0.397 / 0.354 and 0.429 / 0.416 / 0.396 | 0.631 / 0.625 / 0.626 and 0.739 / 0.699 / 0.719 |

  Over the 20 blind values the formula's mean absolute error is 0.010 (the reviewer's bound: 0.015); 19 are within
  0.018 and one is 0.039 off (strength 3.0, input {4}, 50 %), which is beyond the bound the reviewer set for any
  single value (0.030, failure beyond 0.035). On the 12 values of the high end: second formula 0.011 (largest
  0.039), first formula 0.033 (0.109), the reviewer's line 0.028 (0.063), the line through 2.0 and 2.5 0.014
  (0.039). With the shared input the formula is the best of these (0.009 against 0.019 for the nearer line); without
  it the nearer line is better (0.009 against 0.014). At the low end (0.25 and 0.35) the measured K is between
  0.997 and 1.012 and every candidate is within 0.02. So the second formula follows the bend that the first one
  missed, and for the shared input it holds out to 3.0; against a line through the two nearest measurements it is
  no better overall.
* **The three statements behind it**, for the network z = Σ_j a_j relu(w_j·s + b_j) with a_j the difference of unit
  j's two read-out weights.
  1. *a_j² − 2 (‖w_j‖² + b_j²) is conserved for every unit.* This is the balance property of homogeneous networks
     (Du, Hu and Lee 2018). Under the SGD of these pilots it drifts by less than 0.15 % of its mean size while the
     term grows, and by 2 to 3 % over 12,000 steps, at widths 32 to 512.
  2. *The kernel in a direction grows as λ0 + S(F; λ0), with no fitted constant.* For the term alone this holds at
     the width of the pilots, along the whole growth and within 1.2 % in the median (2 to 9 % per seed). It does not
     hold at other widths. Measured kernel over the law at F = 0.4, 0.8 and 1.5 (medians over 8 seeds):

     | width | kernel at step 0 | measured / law | growth / F² at small F: measured, from the conserved quantity |
     |---|---|---|---|
     | 32 | 0.121 | 1.75 / 1.84 / 1.83 | 3.02, 3.00 |
     | 64 | 0.215 | 1.26 / 1.35 / 1.37 | 1.66, 1.84 |
     | 128 | 0.388 | 1.01 / 1.00 / 0.99 | 0.93, 1.01 |
     | 256 | 0.786 | 0.93 / 0.83 / 0.74 | 0.50, 0.53 |
     | 512 | 1.51 | 0.93 / 0.80 / 0.64 | 0.27, 0.27 |

     The last column is what statement 1 gives (added here; no fitted constant; "small F" is F = 0.16, and 0.4 at
     width 512). Write g_j for unit j's own coefficient in the direction and n_j for the squared gradient of g_j with
     respect to the unit's weights and bias. If the units keep their directions, a_j' = 2 g_j and g_j' = n_j a_j, so
     each unit's share of the kernel, P_j = 2 g_j² + n_j a_j², and of the coefficient, Q_j = a_j g_j, obey
     P_j² − 8 n_j Q_j² = const and Q_j' = P_j. For small F this gives λ − λ0 ≈ 4 (Σ_j n_j P_j) F² / λ0². The
     share-weighted mean of n_j is 0.09 to 0.10 at every width, a property of single units, while λ0 is proportional
     to the width; the coefficient of F² is therefore about 0.4 / λ0, and it is 1, as S has it, only near width 128.
     The whole curve computed this way from the network at initialisation is within 2 % of the measured kernel at
     width 512 and within 6 % at width 256; at widths 32 to 128 it is 15 to 24 % low by the end of the growth,
     because there the units turn towards the term's inputs (the lazy and the feature-learning regimes of Chizat,
     Oyallon and Bach 2019).
  3. *Learning a first-order coefficient F_1 raises the read-out norm Σ_j a_j² by S(F_1; λ1), not by an amount
     proportional to F_1².* Measured while a term of strength 3.0 is learned: the growth of the read-out norm over
     S is 0.40, 0.46, 0.49, 0.50, 0.51, 0.52 at F_1 = 0.5, 1, 2, 3, 4, 5, and over F_1² it is 0.40, 0.44, 0.41,
     0.36, 0.32, 0.28. So the growth is proportional to S within 6 % from F_1 = 1 on, at half of S, and it is linear
     rather than quadratic for large F_1: that is the bend. The conserved quantity with half of the gates open
     gives 2 (√(λ1² + 2 F_1²) − λ1) with no fitted constant; the measured growth is 0.83 to 1.00 of it.

  The parts of the kernel that exceed the alone twin's at 25 % agree with the reviewer's account at all twelve
  strengths: without a shared input the excess over S is 0.0136 to 0.0144 for strengths 1.5 to 3.0 (over F_1(0)² it
  falls from 0.0115 to 0.0073); with the shared input the part from the first layer's weights on the term's other
  input is 0.0195 to 0.0226 times F_1(0)² from 0.5 to 3.0. So the structure of the derivation stands: a conserved
  quantity, a read-out norm that grows linearly in a large first-order coefficient, and an extra term in F_1(0)²
  when the input is shared. Its particular function S with λ1 as the only scale, and the claim that nothing is
  fitted, do not: the variant of the formula with constants taken from the initialisation has a mean error of 0.03
  at the high end (largest 0.06).
* **Also from the five strengths.** The closed form of §6.3 is within 9 % of the measured residual factor up to
  half of the term's growth at all twelve strengths (factors from 1.04 to 87); at three quarters the ratio is 0.81
  to 1.33. The constant-kernel start of §6.4, which was written before these runs, gives the lower-order coefficient
  at 10 % within −0.08 to +0.04 on the five new strengths (the closed form is 0.05 to 0.18 too high).
* **Limits.** As in §6.3. The law pilot varies the width for the term alone only; K itself was measured at one
  width. The frozen-direction calculation assumes a residual with the term's direction alone.

## 7. Checks and reproduction

* **Internal checks** (`checks` in the JSON; the script stops if they fail).
  * Pythagoras: L_{k−1} − L_k equals Σ_c D(c) KL(p^(k) ‖ p^(k−1)) to 8e-12 over all runs and orders.
  * The largest gradient left at a projection is 8e-10; L_6 is 0 to 1e-14.
  * The loss computed here equals the light curves' exact `x_lossD` at all 19,200 common (run, checkpoint) points,
    to 1e-16. The two come from different files.
  * The script's own source model reproduces the stored `truth` (to 1e-12) and the stored chain `pmf_D` (to 9e-16).
* **Independent recomputation.** Two further agents recomputed the numbers from the definitions and the DR2a files
  alone, without the scripts, the numbers or this note
  ([hierarchy_projections_verification.md](v2_analysis/hierarchy_projections_verification.md)). Every number in
  §2, §3 and §6 was reproduced, except the steps of fastest change in §3: the projection results to within 3e-10,
  the rest to floating-point precision. Their
  remarks are in the text above: the count that depends on the median convention and the two rungs next to the
  truth (§2.2), the clip constant (§2.4), the unconverged {1,2,3}:2.0 chain runs (§2.6), the ill-defined indicator
  cells (§3), the uneven checkpoint grid and the shared initial networks (§4). The verifiers are AI agents of the
  same kind as the analyst, so this checks the arithmetic, not the framing.
* **Reproduce.** From the repository root, `python scripts/analysis_v2/hierarchy_projections.py` (about 90 s) and
  `python scripts/analysis_v2/lightcurve_lead_lag.py` (seconds). Both extract the DR2a archives into `work/dr2a`
  when needed and rewrite their JSON files; reruns reproduce the committed files byte for byte.
* **The pilot of §6.1** needs torch: `PYTHONPATH=. python scripts/analysis_v2/delay_sweep_pilot.py --jobs 4`
  (264 runs with the second round, cached in `work/delay_sweep`; `--skip-runs` analyses the cache). §6.2 also needs
  `PYTHONPATH=. python scripts/analysis_v2/kernel_gate_pilot.py --jobs 4` (200 runs with the strengths of §6.3 to
  §6.5) and, for §6.5, `PYTHONPATH=. python scripts/analysis_v2/kernel_law_pilot.py --jobs 4` (64 runs, cached in
  `work/kernel_law`). None of these was recomputed independently.

## References

* Refinetti, M., Ingrosso, A., Goldt, S. (2023). Neural networks trained with SGD learn distributions of increasing
  complexity. *ICML*. arXiv:2211.11567.
* Belrose, N., Pope, Q., Quirke, L., Mallen, A., Fern, X. (2024). Neural networks learn statistics of increasing
  complexity. *ICML*. arXiv:2402.04362.
* Rende, R., Gerace, F., Laio, A., Goldt, S. (2024). A distributional simplicity bias in the learning dynamics of
  transformers. *NeurIPS*. arXiv:2410.19637.
* Abbe, E., Boix-Adserà, E., Misiakiewicz, T. (2023). SGD learning on neural networks: leap complexity and
  saddle-to-saddle dynamics. *COLT*. arXiv:2302.11055.
* Kunin, D., et al. (2025). Alternating gradient flows: a theory of feature learning in two-layer neural networks.
  *NeurIPS*. arXiv:2506.06489.
* Pezeshki, M., Kaba, S.-O., Bengio, Y., Courville, A., Precup, D., Lajoie, G. (2021). Gradient starvation: a
  learning proclivity in neural networks. *NeurIPS*. arXiv:2011.09468.
* Yang, G., Salman, H. (2019). A fine-grained spectral perspective on neural networks. arXiv:1907.10599.
* Du, S. S., Hu, W., Lee, J. D. (2018). Algorithmic regularization in learning deep homogeneous models: layers are
  automatically balanced. *NeurIPS*. arXiv:1806.00900.
* Chizat, L., Oyallon, E., Bach, F. (2019). On lazy training in differentiable programming. *NeurIPS*.
  arXiv:1812.07956.
* Amari, S. (2001). Information geometry on hierarchy of probability distributions. *IEEE Trans. Inf. Theory*.
