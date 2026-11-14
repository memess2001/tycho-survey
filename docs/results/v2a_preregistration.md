# Sky v2, Part A: design, pre-registered criteria and run-time estimate

**Status: revised after independent review; final consistency check pending. It becomes binding when this line is
removed.**

Written 14 November 2026, before any confirmatory Part A data exist. Part A tests gate 2 on trained networks: does
the θ view (log-odds) see something the η view (means) does not, judged by a referee that is neither? Background:
[v2_design_draft.md](v2_design_draft.md), [v2_a3_stored_data.md](v2_a3_stored_data.md) and
[v2_a0_a1.md](v2_a0_a1.md). The review of the first draft:
[v2_analysis/v2a_prereg_review.md](v2_analysis/v2a_prereg_review.md).

## 0. What is known before this pre-registration, and so not blind

* **DR1 (v1).** Its Markov runs overlap several Part A configurations. v1 found chain-induced transient shortcuts,
  the delays of Q6, and one θ/η divergence ({1}:1.0 + {1,2,3}:1.0, chain).
* **A3.** In DR1 the views diverge only with a saturated output and large lower-order coefficients, and only in that
  one configuration.
* **A1.** On constructed teachers the views diverge in both directions above a largest |m| of about 0.74.
  Conditional references report no spurious sets, but they lose power when inputs are redundant. The uniform
  reference's clean record relies on the teacher being defined off the data distribution.
* **The review pilot (a preview of the draft's tests).** The independent review of the first draft trained 72 Markov
  runs (seeds 0–3: the six coupling sets of the twin pairs in both contexts and six more chain sets) and 24 parity
  runs (k ∈ {2, 3} × ρ_train ∈ {0, 0.5, 0.8} × seeds 0–3), and computed the draft's measures and tests on them. The
  Markov runs are DR1's networks (the same seeds and data_seed 0). None are confirmatory networks: Part A uses new
  seeds (§2). What the preview showed, as Δ = ρ_θ − ρ_η (negative favours η):

  | | pooled, draft η value | per configuration, draft η value | per configuration, bounded η value |
  |---|---|---|---|
  | T1 Markov: M_D / M_U / C_D / C_U | −0.09 / −0.12 / −0.39 / −0.20 | +0.03 / −0.11 / −0.07 / −0.14 | +0.05 / −0.07 / −0.06 / −0.17 |
  | T1 parity: C_D / C_U | +0.07 / +0.06 | +0.25 / +0.11 | −0.09 / −0.13 |
  | T2, KL referee: C_D / C_U | −0.05 / −0.04 | −0.20 / −0.14 | not computed |

  Under the draft's rules the preview pointed to "η wins". Under the rules below it still does (Markov T1 at C_U
  favours η with both η values; the parity results depend on the η value and so do not count). The revisions were
  made for the methodological reasons listed in the review record. Taken together they do not move the preview
  toward θ (the per-configuration statistic alone did, for parity with the draft's η value; the requirement to win
  with both η values removes that).
  Other preview facts used below: shift fragility differs about 100-fold between Markov coupling sets; all parity
  runs reach accuracy 1.000 under D, uniform and S by step 270; Q3's divergence appeared in 4 of 4 chain seeds and
  0 of 4 iid seeds.
* **Not blind.** The predictions of Q1 to Q3 (§5) and the expectations of §4 were written after the preview.

## 1. The measure factorial

A single family of measures replaces A0's four measures for Part A's tests (those four stay as descriptive output).
For a set A of inputs, the information added by the term χ_A to a model of the output that has every term
χ_B, B ⊉ A, is `tycho.ri.cells.projection_information`:

* **link θ**: weighted logistic regression of ŷ; the value is the drop in weighted cross-entropy.
* **link η**: weighted least squares of the mean m = E[2ŷ − 1 | x] on the same features, scored by the expected
  squared error of 2ŷ − 1. With ΔRSS the reduction of that error from adding χ_A (ΔRSS ≤ 1), η has two values:
  * **bounded (primary)**: η = ln 2 − h((1 + √ΔRSS)/2), with h the binary entropy in nats. It is at most ln 2,
    like θ for a binary output. At the M_U reference with A the whole table it equals A3's R(f_u) =
    `eta_information(f_u(A))`.
  * **Gaussian (sensitivity)**: ½ log(RSS_lo / RSS_hi). RSS includes the Bernoulli variance Σ w (1 − m²), which
    goes to 0 as the output saturates, so this value grows without bound: two equiprobable cells with m = ±0.99
    give θ = 0.662 and Gaussian η = 1.959 nats, against bounded η = 0.662.

  Both values have the same zero set (ΔRSS = 0). A link wins a comparison only if it wins with both (§4).
* **How the links differ.** In the link function, in the loss (Bernoulli deviance against squared error), and in
  the scale of the value. The η values above are chosen so that the scale is comparable, not identical.

Each link is computed at four **references**, one per combination of table and weighting:

| reference | table | cell weights | θ measure | η measure |
|---|---|---|---|---|
| M_D | the inputs of A only (marginal) | training distribution D | local RI under D | linear analogue |
| M_U | the inputs of A only | uniform distribution of all inputs | local RI under uniform probes (v1's spectral line) | linear analogue |
| C_D | all table inputs (conditional) | D | conditional RI under D | linear analogue |
| C_U | all table inputs | uniform | conditional RI under uniform weights | linear analogue |

At M_U each cell's mean averages the output uniformly over the inputs outside A.

* **Zero sets.**
  * When A is every input of the table (always so at the marginal references), the θ value is 0 iff the cells'
    log-odds have no χ_A term, and the η value is 0 iff the cell means have none. With every cell weighted these
    are properties of the function, whatever the weights.
  * At the conditional references, for a smaller A, only one direction holds: no term on A or on any superset
    gives 0. The converse fails, and the zero set depends on the weights. Example: for a singleton {i} at C_U both
    links are 0 iff the mean's uniform Walsh coefficient f_u({i}) is 0, even when the log-odds have a χ_i term;
    the same function gives θ_C_D({i}) > 0 under a correlated D.
  * For a singleton the two links have the same zero set at all four references. They can disagree about presence
    only for |A| ≥ 2; their values differ for every A.
  * Tests: `tests/test_ri_cells.py` (top sets under several weightings, the singleton example, the two η values).
* **Exact values.** Every Part A task has a small input space: 6 Markov lags (64 patterns) and 8 parity bits (256
  patterns). At every checkpoint the network's output on every pattern is stored. With the exact pmfs of D, of the
  uniform distribution and of the shift distribution (§2), all measures are exact population values, with no
  sampling noise. Sampled-probe estimates are computed as well, for instrument check IC-2 only.
* **Sets.** All sets of at most 3 inputs: 41 Markov sets (with x̂) and 92 parity sets (with ŷ). The planted
  4-lag sets of "1,2,3,5:1.0" and "1,2,3,5:0.5" are therefore not measured as sets; their lags count as support.
* **Presence.** For descriptive counts a value is present when it is at least 0.001 nats, the policy floor, with
  the bounded η value. The confirmatory tests use the values themselves, with no threshold.

## 2. Sky v2a

Configuration file `configs/sky_v2a.yaml`, to be written with the engineering (§6).

**Seeds.** Markov runs use seeds 10–19 and parity runs seeds 20–39, with data_seed equal to the seed, so no Part A
network repeats a DR1 network or a review-pilot network, and each seed has its own corpus. The same seed gives the
same initial weights in every configuration of a region, as in v1; both twins of a T2 pair share seed and
data_seed.

**Region markov-v2.** The v1 Markov language: window 6, MLP width 128, depth 1, Adam lr 1e-3, batch 64, 30,000
steps, 60 log-spaced checkpoints.
* Couplings: v1's 12 sets plus four saturated ones: "1:2.0", "1,2,3:2.0", "1:2.0;1,3:1.6" and
  "1:2.0;1,2,3:2.0".
* Training contexts: chain and iid. 10 seeds. 16 × 2 × 10 = 320 runs.
* Distributions: D is the exact stationary distribution of the 6-token window under the language (chain) or uniform
  (iid). The shift distribution S is a first-order binary Markov chain over the window positions with
  ρ = −0.5 (anti-correlated). The target conditional p* is the same under D, uniform and S.
* Loss under a distribution Q: Σ_x Q(x) KL(p*(· | x) ‖ p_net(· | x)).

**Region parity-corr.** Sparse parity on correlated inputs.
* Inputs: n = 8 bits x_i ∈ {−1, +1}, numbered 0–7 along a stationary binary Markov chain with uniform marginals
  and P(x_{i+1} = x_i) = (1 + ρ_train)/2.
* Planted sets: S = {2, 5} for k = 2 and {1, 3, 6} for k = 3. The label is y = 1 iff Π_{i∈S} x_i = −1, as in
  `tycho.tasks.sparse_parity`.
* Training: MLP width 256, depth 1, ReLU, SGD lr 0.1, batch 32, fresh samples every step, 2,000 steps (the first
  draft had 10,000; the preview showed every run converged by step 270, so most of a long run is margin growth),
  60 log-spaced checkpoints.
* Knobs: k ∈ {2, 3} and ρ_train ∈ {0, 0.5, 0.8}, with 20 seeds. 2 × 3 × 20 = 120 runs.
* Distributions: D is the training chain. S is a binary Markov chain with ρ = −0.5.
* Loss under Q: E_Q[−log p_net(y(x) | x)].

**Stored per checkpoint.** The output on all patterns (64 or 256 numbers), the light-curve metrics, and the
exact losses under D, uniform and S, with their squared-error counterparts (Markov: Σ_x Q(x)(p_net(1 | x) −
p*(1 | x))²; parity: the Brier score E_Q[(1 − p_net(y(x) | x))²]). The v1-style sampled RI curve is also stored,
with 8,192 (Markov) or 4,096 (parity) probes drawn i.i.d. from the exact pmf of D, for IC-2.

## 3. Instrument checks

A failure of IC-1, IC-2 or IC-4 means a bug, and the tests wait for the fix.

* **IC-1.** Replays of 5 % of objects are bit-identical to the stored light curves.
* **IC-2.** (a) Code check: for 5 % of (run, checkpoint) pairs, the exact-value code run with the probe rows' cell
  counts as weights reproduces the stored sampled values of θ_M_D and θ_C_D to within 1e-9. (b) Sampling check:
  |sampled − exact| ≤ 0.002 nats at ≥ 95 % of (set, checkpoint) points, and the median signed error is within
  ±0.0005.
* **IC-3.** For chain runs, the total variation between the exact stationary window pmf and the corpus's window
  frequencies is reported for every corpus. It must be ≤ 0.02, except for the coupling sets whose stationary pmf
  puts ≥ 90 % of its mass on two windows ("1:2.0", "1:2.0;1,3:1.6", "1:2.0;1,2,3:2.0"), where it is reported only.
  A failure elsewhere is a bug.
* **IC-4.** For iid Markov runs and parity runs with ρ_train = 0, where D is uniform, every D-weighted measure equals
  its uniform counterpart to within 1e-9. In every run, θ_M_D equals `tycho.ri.refined.local_ri` of the exact
  marginal to within 1e-9.
* **IC-5** (a precondition, not a bug check). Final KL from truth under D below 0.01 for Markov runs; final parity
  accuracy under D at least 0.95 for parity runs. A failing run is excluded from T1 and Q1–Q4; a T2 pair is
  excluded if either twin fails; a configuration or twin set with fewer than 5 remaining seeds is dropped from ρ̄
  (§4) and listed. If fewer than 95 % of Markov runs or 90 % of parity runs pass, the shortfall is reported with
  the results.

## 4. Confirmatory tests of gate 2

Both tests compare the two links at the same reference against a behavioural referee that is neither view. They
are weaker than the design draft's R2 and R3: T1 measures predictor and outcome at the same checkpoint, so it is a
concurrent test, not a forecast; T2 intervenes on the data-generating process, not on the network. A pass is
reported as "gate 2 passed on behavioural referees". A referee that intervenes on the network (ablation) is not part
of this pre-registration.

**Common definitions.**
* **Window.** A run's points are its checkpoints from the first checkpoint t ≥ 1 with loss_D(t) ≤ ½ loss_D(0) to
  the end. Runs that never reach it contribute no points and are counted. Step 0 is never in the window.
* **Spearman ρ** is signed; a negative ρ counts against the link. If a predictor is constant within a configuration,
  its ρ there is 0.

**T1: shift fragility.**
* **Runs.** Runs trained on correlated inputs: Markov chain runs (16 configurations, 160 runs) and parity runs with
  ρ_train ∈ {0.5, 0.8} (4 configurations, 80 runs).
* **Predictor.** S_μ(t), the sum over spurious sets A with |A| ≥ 2 of measure μ at checkpoint t. A set is spurious
  if it contains an input outside the generating support: lags in no planted coupling, or bits outside S. Sets
  with one input are excluded from the primary predictor because the links cannot disagree about their presence
  (§1). The sum over all spurious sets is reported as secondary. Within a configuration the number of summed sets is
  fixed, so the sum and the mean give the same ranks.
* **Outcome.** Y(t) = loss_S(t) − loss_D(t), exact. Sensitivity outcome: the same difference of the squared-error
  losses.
* **Statistic.** For each configuration c, ρ_μ,c is the Spearman correlation between S_μ and Y over all (seed,
  checkpoint) points of c in the window. The family statistic is ρ̄_μ, the unweighted mean of ρ_μ,c over
  configurations, and Δ = ρ̄_θ − ρ̄_η at each reference. Secondary, not deciding: the pooled Spearman over all points
  of the family, and a seed-level statistic (for each configuration and checkpoint, the Spearman across seeds,
  averaged). The primary statistic asks whether a measure tracks fragility over training within a configuration;
  the seed-level one asks whether it ranks networks of one configuration at one time. The pooled one mainly ranks
  configurations, whose fragility differs about 100-fold, and is not used to decide.

**T2: twin effect.**
* **Twins.** Markov twins that differ by one planted term in the data, with the same seed and data_seed:
  * "1:1.0" vs "1:1.0;1,2,3:1.0";
  * "1:1.0" vs "1:1.0;1,3:0.8";
  * "1:2.0" vs "1:2.0;1,2,3:2.0";
  * "1:2.0" vs "1:2.0;1,3:1.6".

  Each pair is trained in both contexts, giving 8 twin sets × 10 seeds.
* **Window.** The common rule applied to the "with" network, with loss under D_with.
* **Outcome.** Y_twin(t) = E_{x∼D_with} KL(p_with(· | x) ‖ p_without(· | x)), the behavioural difference between
  the two networks, exact over the 64 windows under the distribution D_with of the twin with the added term.
  Sensitivity outcome: E_{x∼D_with}(p_with(1 | x) − p_without(1 | x))².
* **Predictor.** Δμ(t) = μ_with(A_added, t) − μ_without(A_added, t), with both networks evaluated under D_with.
* **Identifiability.** At M_D and C_D a twin set enters T2 only if the D_with-weighted residual variance of χ_A,
  after weighted least squares on the features of the smaller model at that reference, is ≥ 0.01. Without this the
  added term is untestable rather than absent; in saturated chains D_with puts nearly all its mass on two windows.
  Excluded twin sets are listed. At M_U and C_U the residual variance is 1, so every twin set enters.
* **Statistic.** As in T1, with c a twin set: ρ_μ,c is the Spearman correlation between Δμ and Y_twin over all
  (seed, checkpoint) points of the twin set, ρ̄_μ their unweighted mean, Δ = ρ̄_θ − ρ̄_η. The mean over the chain
  twin sets only is reported as secondary.

**Bootstrap.** 10,000 resamples, seed 20261114. Seeds are resampled with replacement within each configuration or
twin set; all checkpoints of a resampled run, and both twins of a pair, are kept. Both ρ̄ are computed on the same
resample. The two-sided p-value of Δ is min(1, 2 · min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / (B + 1)), with B the number
of resamples. Inference is about these configurations: the configurations are fixed, and only seeds are resampled.

**Comparisons and decision.** There are 12 comparisons: T1 Markov, T1 parity and T2, each at 4 references. A link
**wins** a comparison if all three hold:
1. with the bounded η value: Δ̂ favours the link, |Δ̂| ≥ 0.10, and the Holm-adjusted p-value over the 12 is below
   0.05;
2. the same with the Gaussian η value (Holm over its own 12);
3. Δ̂ has the same sign with the sensitivity outcome (bounded η).

A comparison that meets 1 or 2 but not both is reported as **value-dependent** and is not a win.

| outcome | condition | meaning | next |
|---|---|---|---|
| **gate 2 passed (θ), behavioural referees** | θ wins at least one comparison and η wins none | the log-odds view tracks behaviour that the mean view misses | carry RI to Part B and beyond as the primary view; add a network-intervention referee |
| **η wins** | η wins at least one and θ none | the mean view tracks behaviour better | reconsider RI as the survey's spectral line; Part B adopts the η view, with RI as a descriptive companion |
| **mixed** | both win somewhere | the answer depends on reference or task | report which; Part B pre-registers the reference-specific hypotheses |
| **tie** | no win | the views are interchangeable on these referees | RI's value is attribution and description, not detection; Part B uses whichever is cheaper |

**Expectations, informed by the preview (§0) and written down so they can be wrong.**
* **T1 Markov.** η wins at C_U; the other references tie.
* **T1 parity.** Value-dependent or tie: the preview's sign flipped with the η value.
* **T2.** η favoured or tie. The four iid twin sets barely separate the links in the preview (|Δ| ≤ 0.01), which
  pulls ρ̄ toward 0, and the identifiability rule removes most chain twin sets at the D references.
* **Overall.** "η wins" is the expected outcome; a θ win is not. The tests are run because they are fair, not
  because they favour the view this research started from.

## 5. Pre-registered questions (estimates; any outcome is reported)

Presence is ≥ 0.001 nats with the bounded η value, and spurious sets of all sizes count unless stated.

1. **Spurious structure by reference** (end of training; chain-trained Markov runs and parity runs with
   ρ_train ∈ {0.5, 0.8}). The number of spurious sets present for each of the 8 measures. Predictions, for each
   link:
   * M_D: at least one spurious set present in ≥ 80 % of runs;
   * M_U: no prediction;
   * the C_D count is at most the M_D count in ≥ 90 % of runs, and smaller in total;
   * the C_U count is at least the C_D count in ≥ 90 % of runs.
2. **The divergence law on trained networks.** Over all (run, set with |A| ≥ 2, checkpoint) points, a divergence at
   a reference is one link ≥ 0.002 and the other < 0.001. At C_D, divergences are binned by saturation (the largest
   |m| over all 64 or 256 patterns at that checkpoint) and by lower-order energy (√Σ_{∅≠B⊊A} f_u(B)², with f_u
   the uniform Walsh coefficients of m over all patterns). Predictions:
   * at least 95 % of divergent points have saturation ≥ 0.74 and lower-order energy ≥ 0.3;
   * the coupling-2.0 configurations have more divergent points per run than the coupling-1.0 ones.

   The direction is reported for both η values and not predicted: in the preview every C_D divergence was η-only
   with the Gaussian value and θ-only with the bounded value.
3. **Replication of the DR1 divergence, on new seeds.** For "1:1.0;1,2,3:1.0", set {1, 2, 3}, at M_U (v1's line):
   η ≥ 0.002 and θ < 0.001 at two or more consecutive checkpoints after step 0. Predicted in at least 8 of 10
   chain seeds and at most 2 of 10 iid seeds, with each η value. The other three references are reported without
   a prediction.
4. **Off-distribution behaviour.** The C_U sum over spurious sets at the end of training, chain run minus the iid
   run with the same coupling set and seed (160 pairs), for each link. Predicted positive: one-sided sign test over
   pairs, p < 0.05. (For iid runs D is uniform, so C_U − C_D is 0 by construction and cannot be the contrast.)
5. **Exploratory, not part of the gate.** Sparse-description transport. Each link picks its top-K sets (K = 3, 5, 8)
   at C_D. A logistic model of the network's output on those features is fitted under D and evaluated under S.
   Because the evaluation model is logistic, this favours θ, and it is reported as exploratory only.

## 6. What must be built before the run (engineering, with tests)

1. Storage of the network's output on all input patterns at every checkpoint.
2. Exact pmfs: the stationary window distribution of the Markov language, the binary Markov chain over parity
   bits, and the shift distributions.
3. The parity task variant with Markov-chain inputs, a ρ knob and fixed planted sets, sampled online.
4. Light-curve metrics: losses from truth under D, uniform and S, and their squared-error counterparts.
5. Seeds: a per-region first seed (Markov 10, parity 20) and data_seed tied to the seed.
6. The offline analysis: the 8 measures with both η values for every set and checkpoint, the identifiability
   filter, the T1 and T2 statistics with the cluster bootstrap and Holm adjustment, and the questions of §5. It
   pins BLAS to one thread per process (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`): the
   review measured multithreaded logistic fits in parallel workers running about 600 times slower.
7. `configs/sky_v2a.yaml` and a start script. A pilot of 2 runs per region checks IC-1 to IC-4 before the launch;
   the pilot uses seeds outside the confirmatory ranges.

Done already: both η values in `projection_information` (`eta_value="bounded"` or `"gaussian"`), with tests.

Any change to the definitions in §1 to §5 after this document is committed is made by a dated amendment before
Part A data are read, or reported as post hoc.

## 7. Run-time estimate on the 88-core server

| region | runs | training (CPU per run) | exact measures (CPU per run) | total CPU |
|---|---|---|---|---|
| markov-v2 | 320 | ≈ 25 s (v1: 1.6 h for 246 runs) | ≈ 20 s (0.26 s per checkpoint; review pilot: 10 s per run) | ≈ 4 h |
| parity-corr | 120 | ≈ 5 s (2,000 steps) | ≈ 3 to 6 min (2.5 to 5.7 s per checkpoint: 92 sets on 256-cell tables) | ≈ 6 to 12 h |
| **total** | 440 | | | **≈ 10 to 16 CPU hours** |

On 86 workers this is about 15 to 25 minutes of wall-clock time if the server is otherwise idle. The parity
measures dominate and run after training, so they can also run offline on the 4-core analysis machine (about 2 to 3
hours). The bootstrap (10,000 resamples × 12 comparisons × 2 η values) adds minutes. Memory is small: the largest
stored array is 256 × 60 per run.

## Amendments

* **14 November 2026.** First draft revised after the independent review: per-configuration statistic, bounded η
  value with the Gaussian value as sensitivity, new seeds with data_seed = seed, IC-2/IC-3/IC-5 rewritten, the T2
  identifiability rule, Q1–Q4 redefined, §1 statements corrected, bootstrap and windows fixed, a squared-error
  sensitivity outcome, parity training shortened to 2,000 steps. Details and the reasons:
  [v2_analysis/v2a_prereg_review.md](v2_analysis/v2a_prereg_review.md).
