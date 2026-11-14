# Sky v2, Part A: design, pre-registered criteria and run-time estimate

**Status: draft under independent review. It becomes binding when this line is removed.**

Written 14 November 2026, before any Part A data exist. Part A tests gate 2 on trained networks: does the θ view
(log-odds) see something the η view (means) does not, judged by a referee that is neither? Background:
[v2_design_draft.md](v2_design_draft.md), [v2_a3_stored_data.md](v2_a3_stored_data.md) and
[v2_a0_a1.md](v2_a0_a1.md).

**What is known before this pre-registration, and so not blind:**
* DR1 (v1). Its Markov runs overlap several Part A configurations. v1 found chain-induced transient shortcuts, the
  delays of Q6, and one θ/η divergence ({1}:1.0 + {1,2,3}:1.0, chain).
* A3. In DR1 the views diverge only with a saturated output and large lower-order coefficients, and only in that one
  configuration.
* A1. On constructed teachers the views diverge in both directions above a largest |m| of about 0.74. Conditional
  references report no spurious sets, but they lose power when inputs are redundant. The uniform reference's clean
  record relies on the teacher being defined off the data distribution.
* No trained network has yet been measured with the definitions below. Pilot runs made while the code is built will
  be disclosed in a dated amendment if they bear on a question, as in v1.

## 1. The measure factorial

A single family of measures replaces A0's four measures for Part A's tests (those four stay as descriptive output).
For a set A of inputs, the information added by the term χ_A to a model of the output that has every term
χ_B, B ⊉ A, is `tycho.ri.cells.projection_information`:

* **link θ**: weighted logistic regression; the drop in cross-entropy.
* **link η**: weighted least squares of the mean m = E[2ŷ − 1 | x], scored by the expected squared error of 2ŷ − 1;
  the value is ½ log(RSS_lo / RSS_hi).

Each link is computed at four **references**, one per combination of table and weighting:

| reference | table | cell weights | θ measure | η measure |
|---|---|---|---|---|
| M_D | the inputs of A only (marginal) | training distribution D | local RI under D | linear analogue |
| M_U | the inputs of A only | uniform inputs | local RI under uniform probes (v1's spectral line) | linear analogue |
| C_D | all table inputs (conditional) | D | conditional RI under D | linear analogue |
| C_U | all table inputs | uniform | conditional RI under uniform weights | linear analogue |

* **Zero sets.** At a fixed reference, the two links differ only in the link. Their zero sets are properties of the
  network's function when every cell has weight: no log-odds term (θ) or no mean term (η) on A, once the supersets
  of A are excluded. Their values depend on the weights. Verified on the A1 constructions (`tests/test_ri_cells.py`).
* **Exact values.** Every Part A task has a small input space: 6 Markov lags (64 patterns) and 8 parity bits (256
  patterns). At every checkpoint the network's output on every pattern is stored. With the exact pmfs of D, of the
  uniform distribution and of the shift distribution (§2), all eight measures are exact population values, with no
  sampling noise. Sampled-probe estimates are computed as well, for instrument check IC-2 only.
* **Sets.** All sets of at most 3 inputs: 41 Markov sets (with x̂) and 92 parity sets (with ŷ).
* **Presence.** For descriptive counts a value is present when it is at least 0.001 nats, the policy floor. The
  confirmatory tests use the values themselves, with no threshold.

## 2. Sky v2a

Configuration file `configs/sky_v2a.yaml`, to be written with the engineering (§6).

**Region markov-v2.** The v1 Markov language: window 6, MLP width 128, depth 1, Adam lr 1e-3, batch 64, 30,000
steps, 60 log-spaced checkpoints.
* Couplings: v1's 12 sets plus four saturated ones: "1:2.0", "1,2,3:2.0", "1:2.0;1,3:1.6" and
  "1:2.0;1,2,3:2.0".
* Training contexts: chain and iid. 10 seeds; the same seed number gives the same initial weights in every
  configuration, as in v1. 16 × 2 × 10 = 320 runs.
* Distributions: D is the exact stationary distribution of the 6-token window under the language (chain) or uniform
  (iid). The shift distribution S is a first-order binary Markov chain over the window positions with
  ρ = −0.5 (anti-correlated). The target conditional is the same under D, uniform and S.

**Region parity-corr.** Sparse parity on correlated inputs: n = 8 bits drawn from a stationary binary Markov chain
over positions with spin correlation ρ_train.
* Planted sets: S = {2, 5} for k = 2 and {1, 3, 6} for k = 3. The label is Π_{i∈S} s_i.
* Training: MLP width 256, depth 1, ReLU, SGD lr 0.1, batch 32, fresh samples every step, 10,000 steps, 60
  checkpoints.
* Knobs: k ∈ {2, 3} and ρ_train ∈ {0, 0.5, 0.8}, with 20 seeds. 2 × 3 × 20 = 120 runs.
* Distributions: D is the training chain. S is a binary Markov chain with ρ = −0.5.

**Stored per checkpoint.** The output on all patterns (64 or 256 numbers), the light-curve metrics, and the
exact-truth losses under D, uniform and S:
* Markov: KL from the true conditional.
* Parity: expected log-loss of the true label.

The v1-style sampled RI curve is also stored, with 8,192 (Markov) or 4,096 (parity) probes from D, for IC-2.

## 3. Instrument checks (a failure means a bug, and the tests wait for the fix)

* **IC-1.** Replays of 5 % of objects are bit-identical to the stored light curves.
* **IC-2.** The sampled-probe plug-in values of θ_M_D and θ_C_D are within 0.002 nats of the exact values at 99 %
  of (set, checkpoint) points.
* **IC-3.** For chain runs, the exact stationary window pmf is within total variation 0.02 of the corpus's window
  frequencies.
* **IC-4.** For iid runs, where D is uniform, every D-weighted measure equals its uniform counterpart to within
  1e-9. In every run, θ_M_D equals `tycho.ri.refined.local_ri` of the exact marginal to within 1e-9.
* **IC-5** (a precondition). Final KL from truth under D is below 0.01 in at least 95 % of Markov runs, and final
  parity accuracy under D is at least 0.95 in at least 90 % of parity runs. Otherwise the tests use converged runs
  only, and the shortfall is reported.

## 4. Confirmatory tests of gate 2

Each test compares the two links at the same reference with a referee that is neither: behaviour under a shift
of the input distribution (R2), and the behavioural effect of an intervention on the data-generating process (R3).

**T1: shift fragility (R2).**
* **Runs.** Runs trained on correlated inputs: Markov chain runs (160) and parity runs with ρ_train ∈ {0.5, 0.8}
  (80).
* **Checkpoints.** From the first checkpoint at which the loss under D has fallen to half its initial value, to the
  end.
* **Predictor.** S_μ(t), the sum over spurious sets A with |A| ≥ 2 of measure μ at checkpoint t. A set is spurious
  if it contains an input outside the generating support: lags in no planted coupling, or bits outside S. Sets
  with a single input are excluded, because the two links can differ only for |A| ≥ 2. The sum over all spurious
  sets is reported as secondary.
* **Outcome.** Y(t) = loss under S − loss under D (KL from truth for Markov, log-loss for parity), exact.
* **Statistic.** Spearman ρ_μ between S_μ and Y over all (run, checkpoint) pairs of a family. The link difference
  is Δ = ρ_θ − ρ_η at each reference. Its 95 % interval and p-value come from a bootstrap over runs (2,000
  resamples, seed 20261114).

**T2: twin effect (R3).**
* **Twins.** Markov twins that differ by one planted term in the data, with the same seed and so the same initial
  weights:
  * "1:1.0" vs "1:1.0;1,2,3:1.0";
  * "1:1.0" vs "1:1.0;1,3:0.8";
  * "1:2.0" vs "1:2.0;1,2,3:2.0";
  * "1:2.0" vs "1:2.0;1,3:1.6".

  Each pair is trained in both contexts, giving 8 twin sets × 10 seeds.
* **Outcome.** Y_twin(t) = E_{x∼D_with} KL(p_with(· | x) ‖ p_without(· | x)), the behavioural difference between
  the two networks, exact over the 64 windows under the distribution of the twin with the added term.
* **Predictor.** Δμ(t) = μ_with(A_added, t) − μ_without(A_added, t), with both networks evaluated under D_with.
* **Statistic.** Spearman between Δμ and Y_twin over all (twin set, seed, checkpoint) triples. The link difference
  is Δ = ρ_θ − ρ_η at each reference, with a bootstrap over (twin set, seed) clusters as in T1.

**Comparisons and decision.** There are 12 comparisons: T1 Markov, T1 parity and T2, each at 4 references. A link
**wins** a comparison if |Δ| ≥ 0.10 in its favour and the Holm-adjusted bootstrap p-value over the 12 is below 0.05.

| outcome | condition | meaning | next |
|---|---|---|---|
| **gate 2 passed (θ)** | θ wins at least one comparison and η wins none | the log-odds view tracks behaviour that the mean view misses | carry RI to Part B and beyond as the primary view |
| **η wins** | η wins at least one and θ none | the mean view tracks behaviour better | reconsider RI as the survey's spectral line; Part B adopts the η view, with RI as a descriptive companion |
| **mixed** | both win somewhere | the answer depends on reference or task | report which; Part B pre-registers the reference-specific hypotheses |
| **tie** | no win | the views are interchangeable on these referees | RI's value is attribution and description, not detection; Part B uses whichever is cheaper |

**Expectations, written down so they can be wrong.**
* **T1.** At the marginal references both links see correlations whether or not the network relies on them, so
  both will correlate weakly with fragility. At the conditional references both see actual dependence. Link
  differences are expected only through saturated multi-input sets, so T1 is expected to tie.
* **T2.** In the saturated twins, the added term's behavioural effect is expressed partly through lower-order
  log-odds plus saturation. The η view may track it better there, and an η win in T2 would not surprise us.
* A θ win is not the expected outcome of these tests. The tests are run because they are fair, not because they
  favour the view this research started from.

## 5. Pre-registered questions (estimates; any outcome is reported)

1. **Spurious structure by reference** (end of training, chain-trained Markov runs and correlated parity runs). The
   number of spurious sets present for each of the 8 measures. Predictions:
   * the marginal references flag spurious sets in most runs, with both links;
   * the conditional references flag fewer;
   * C_U ≥ C_D, because the uniform weights expose behaviour on rare and unseen patterns.
2. **The divergence law on trained networks.** Over all (run, set, checkpoint) points, divergences at the C_D
   reference (one link ≥ 2m, the other < m) are binned by saturation (largest |m| over the patterns) and by
   lower-order energy. Predictions:
   * divergences occur only where saturation is at least 0.74 and lower-order energy at least 0.3;
   * the coupling-2.0 configurations have more of them than the coupling-1.0 ones.
3. **Replication of the DR1 divergence.** For "1:1.0;1,2,3:1.0", η-only on {1, 2, 3} at the M_U reference (v1's
   line) at some checkpoint. Predicted in at least 8 of 10 chain seeds and at most 2 of 10 iid seeds. The other
   three references are reported without a prediction.
4. **Off-distribution behaviour.** The difference between the uniform and data-weighted conditional references
   (C_U − C_D) on spurious sets, chain against iid runs. Predicted to be larger in chain runs, because iid
   training covers every pattern.
5. **Exploratory, not part of the gate.** Sparse-description transport. Each link picks its top-K sets (K = 3, 5, 8)
   at C_D. A logistic model of the network's output on those features is fitted under D and evaluated under S.
   Because the evaluation model is logistic, this favours θ, and it is reported as exploratory only.

## 6. What must be built before the run (engineering, with tests)

1. Storage of the network's output on all input patterns at every checkpoint.
2. Exact pmfs: the stationary window distribution of the Markov language, the binary Markov chain over parity
   bits, and the shift distributions.
3. The parity task variant with Markov-chain inputs and a ρ knob, sampled online.
4. Light-curve metrics: losses from truth under D, uniform and S.
5. The offline analysis: the 8 measures for every set and checkpoint, the T1 and T2 statistics with the cluster
   bootstrap and Holm adjustment, and the questions of §5.
6. `configs/sky_v2a.yaml` and a start script. A pilot of 2 runs per region checks IC-1 to IC-4 before the launch.

Any change to the definitions in §1 to §5 after this document is committed is made by a dated amendment before
Part A data are read, or reported as post hoc.

## 7. Run-time estimate on the 88-core server

| region | runs | training (CPU per run) | exact measures (CPU per run, measured) | total CPU |
|---|---|---|---|---|
| markov-v2 | 320 | ≈ 25 s (v1: 1.6 h for 246 runs) | ≈ 20 s (0.26 s per checkpoint for the conditional references) | ≈ 4 h |
| parity-corr | 120 | ≈ 15 s (v1 parity-k2: 26 s for 20k steps at width 512) | ≈ 6 min (5.7 s per checkpoint: 92 sets on 256-cell tables) | ≈ 12 h |
| **total** | 440 | | | **≈ 16 CPU hours** |

On 86 workers this is about 15 to 25 minutes of wall-clock time if the server is otherwise idle. The parity
measures dominate and run after training, so they can also run offline on the 4-core analysis machine (about 3
hours). Memory is small: the largest stored array is 256 × 60 per run.
