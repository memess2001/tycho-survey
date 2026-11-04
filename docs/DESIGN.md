# Tycho: a calibrated sky survey of training dynamics, observed with refined information

*Design document, v0.1, November 2026. For discussion with Prof. Sugiyama.*
*Code: this repository (`tycho/`). Implementation contract: `docs/SPEC.md`. Operations: `docs/OPS_MACMINI.md`.*

---

## 0. Summary

We propose a long-running, mostly autonomous research platform. It trains thousands of small neural networks
under systematically varied conditions and records the full **developmental timeline** of each, using one fixed,
calibrated measurement. The measurement is **refined information (RI)** (Enouen & Sugiyama, TMLR 2026): a
non-negative, higher-order generalisation of mutual information, defined through m-projections onto hierarchical
log-linear models. It tells us *which subset of variables starts to interact, at which order, at which training
step*.

The platform is organised like an astronomical sky survey (SDSS rather than a single observation):

1. **Instrument first.** Before any survey, we characterise the RI estimator's error budget: noise floor,
   detection power, timing resolution, and the artefacts introduced by discretisation. Positive controls have
   known answers. The calibration is part of the deliverable.
2. **Two-tier observation.** Every training run (an *object*) gets cheap **photometry**: light curves of
   loss, accuracy, norms, effective rank and eNTK statistics. Only a selected subset gets expensive **RI
   spectroscopy**: positive controls, anomalous objects, objects at phase boundaries, plus a random 5 % control
   sample against selection bias.
3. **Re-observation by replay.** Unlike the sky, a training run can be replayed bit-for-bit on CPU. Any object
   can be re-observed later with denser RI sampling around a transition. It can also be perturbed and re-run,
   which turns observation into experiment.
4. **A catalog with a transient channel and public data releases.** Each object stores its configuration, seed,
   light curves, RI events (subset, order, onset step with bracket, confidence) and classification. Unusual events
   are pushed to the researcher immediately: late activation, activation followed by extinction, order inversion,
   and pure synergy without lower-order precursors.
5. **An agent that keeps the survey honest.** A Claude-based agent runs nightly and weekly. Each night it does
   quality control, classifies new objects into known phenomena (grokking, double descent, plateau, staged
   learning) or leaves them in an *unknown* queue for the human, proposes the next batch of configurations, and
   writes a survey log. Each week it writes a report that includes candidate empirical laws and their caveats.

The platform runs 24×7 on a single Mac mini (Apple silicon) for weeks to months. The first sky consists of
seconds-to-minutes objects, so the survey reaches tens of thousands of objects per month.

---

## 1. Why a survey, and why now

The empirical science of deep learning is not at a single stage of development. Its objects of study are at
different stages:

| Object of study | Stage | Astronomical analogue |
|---|---|---|
| Trained models as a function of scale | Empirical laws (scaling laws), no accepted mechanism | after Kepler, before Newton |
| What is inside one model | New instruments (probes, SAEs, attribution graphs) revealing new objects; few regularities | Galileo's first telescope |
| How training unfolds in time | Scattered observations (grokking, double descent, induction-head formation, hidden progress in parity); no common ruler, no catalog | before Tycho |

Phenomena of training dynamics were each discovered in one particular setting. Whether they persist under other
configurations, and where their boundaries lie, has rarely been measured systematically, and never with a common,
calibrated instrument. This is the situation before Tycho: observations are scattered, their errors are unknown,
and they cannot be compared.

Two lessons from the history of astronomy shape the design:

* **Precision before quantity.** Kepler abandoned circular orbits because of an 8-arc-minute discrepancy for Mars.
  He trusted that discrepancy only because Tycho's instrumental error was smaller. An anomaly in our catalog is
  meaningful only relative to a known instrument error. Hence calibration (Project 1) is the first phase of the
  survey (Project 3), not a side project.
* **Catalogs falsify beautiful stories.** Kepler's first hypothesis, nested Platonic solids, was elegant and
  wrong. A systematic catalog lets us falsify appealing narratives (for example "interactions are always learned
  from low to high order") as well as discover laws.

We are luckier than Tycho in one respect: the sky does not change when we look again, but we can rerun a training
with another seed or one changed parameter. The methodology is therefore *survey plus experimental biology*: find
phenomena by surveying, then verify mechanisms by intervention and replay.

**In plain words:** we record, with one ruler, the complete development of thousands of networks trained
under different conditions, and look for regularities in the resulting table. The Framingham Heart Study is the
analogy: thousands of people measured with one protocol over decades, rather than case reports.

---

## 2. Research questions

The numbering follows the existing project plan where it is known (RQ0, RQ3). The other items are placeholders
to be reconciled with that plan.

* **RQ0: variables and discretisation.** RI is defined for discrete variables. What are the right variables
  (input coordinates and outputs, binarised units, SAE features, tokens), and how much interaction structure does
  the discretisation itself create or destroy? Calibration experiment E6 below measures this directly.
* **Calibration.** What is the instrument's error budget (noise floor, power, timing resolution, specificity of
  subset identity and order)? Does it recover planted ground truth in synthetic distributions and in trained
  positive-control networks?
* **Phenomenology.** When do k-th-order interactions switch on during training? How does the onset depend on data
  size, width, learning rate and weight decay? Are onsets abrupt or gradual? Do interactions switch off again?
* **RQ3: features versus interactions.** SAE-Track reports that individual SAE features form mostly
  *smoothly* during LLM pre-training. We hypothesise that interactions *between* features can still switch on
  *abruptly*. Both can be true at once. RI sees the second phenomenon; SAEs alone do not.
* **Empirical laws ("Kepler moments").** Cross-configuration regularities that only a catalog can reveal: for
  example, "the onset step of k-th-order interactions scales as a power of the data size", or "interactions are
  learned from low to high order except in region X".

---

## 3. The instrument: refined information

### 3.1 Definitions

Let X = (X_1, …, X_d) be discrete variables with joint distribution P, viewed as a probability tensor. For a
hierarchical family H of subsets (closed under taking non-empty subsets, containing all singletons), the
hierarchical log-linear model is

  E_H = { q : log q(x) = Σ_{S∈H} θ_S(x_S) − ψ(θ) }.

E_H is e-flat. The m-projection P_H = argmin_{Q∈E_H} D(P‖Q) is the unique member of E_H that reproduces P's
marginals on every S ∈ H; equivalently it is the maximum-entropy distribution with those marginals. We compute it
by iterative proportional fitting over the maximal elements of H. For H ⊆ H′ the Pythagorean relation
D(P‖P_H) = D(P‖P_{H′}) + D(P_{H′}‖P_H) holds.

**Refined information (Definition 1 of the paper).** For a hierarchical collection I and a subset S whose proper
subsets are in I,

  RI_{I,S}(p) = D(p_{I+S} ‖ p_I) = D(p‖p_I) − D(p‖p_{I+S}) ≥ 0.

Along a complete chain {∅} = I_0 ⊊ I_1 ⊊ … ⊊ I_T = P([d]) that adds one subset at a time, the terms sum to
D(p‖u), the KL error of the uniform model (eq. 6). This is a *complete* decomposition into non-negative
contributions, one per mode interaction. The singleton steps give D(p_k‖u_k); starting the chain from the
singletons decomposes the total correlation Σ_i H(X_i) − H(X). RI_{I,S} depends on the context I: the paper's
Figure 3 shows values of one subset spread over orders of magnitude. The code exposes the chain explicitly
(`chain_decomposition(start="empty" | "singletons")`, and `greedy_selection` as a strong-heredity, exact-gain
variant of mode interaction selection).

**The two canonical values (Appendix C.1).** The paper defines two order-invariant values per subset:

* the **marginal refined information** RI^marg_S = RI_{P(S)−S, S} = D(p_S ‖ (p_S)_{∂S}), with ∂S the proper
  non-empty subsets of S. It equals the mutual information for |S| = 2 and vanishes iff p_S has no |S|-th-order
  interaction. It coincides with the top-order term of Amari's hierarchical decomposition and with Schneidman et
  al.'s connected information of top order.
* the **conditional refined information** RI^cond_S = RI_{P([d])−{T⊇S}, S}: the gain from S once every
  interaction not containing S is in the model. For example, it vanishes for two variables that are conditionally
  independent given a third.

The survey uses **RI^marg as the per-subset "spectral line"** for event detection. It is computed from the
small sub-table of S alone, so all subsets up to order 4–5 of tens of variables can be scanned, and its estimator
has a known χ² null (§3.2). RI^cond needs the full joint table and is reported only for small variable sets.
`tests/test_ri_paper.py` reproduces the paper's worked examples: Figure 10 gives RI^marg_AB = 0.1308,
RI^cond_AB = 0.0992, RI^marg_BC = 0.0316, RI^cond_BC = 0, and Figure 1 gives J = 1 bit for the 3-bit XOR.

### 3.2 Estimation and significance

From N samples, the plug-in estimate on the empirical marginal table gives the classical likelihood-ratio
statistic of nested hierarchical log-linear models:

  G² = 2N · RI_hat(S) → χ²_df under "no |S|-th-order interaction", df = Π_{i∈S}(k_i − 1).

This yields p-values and a first-order bias correction RI_corr = max(0, RI_hat − df/2N). For small or sparse
tables we use a parametric bootstrap from the fitted null model, which is also what calibrates the χ²
approximation. At each checkpoint we control the false discovery rate across candidate subsets
(Benjamini–Hochberg). A subset is **detected** when q ≤ α *and* RI_corr ≥ ε_min, where ε_min comes from
calibration: with large N, tiny interactions become significant without being interesting.

### 3.3 Events

For each object, RI is measured for every candidate subset at T checkpoints (log-spaced in training steps,
densified around transitions on replay), giving an S × T "RI light curve". An **activation event** is the first
checkpoint after which the subset stays detected for at least m consecutive checkpoints. Hysteresis (separate on
and off thresholds) suppresses flicker. The onset is reported with a bracket [last inactive, first active], and
replay can narrow the bracket. **Extinction** and **reactivation** are defined analogously.

Each event records whether a proper subset activated earlier (a *precursor*) and whether it matches known ground
truth (positive controls). The following events are pushed through the transient channel: late activation (after
the training set is fit), pure synergy (an order ≥ 3 activation with no lower-order precursor), order inversion
(a set activates before all of its subsets that eventually activate), extinction, and reactivation.

Two gates keep the channel useful:

* Only events whose peak RI reaches 5 × min_effect raise transients. Weaker events stay in the catalog. In the
  first end-to-end run, this cut alerts from 22 to 7 per 8 objects, because most were weak interactions of
  intermediate solutions flickering near the threshold.
* Events that match a positive control's ground truth are downgraded to informational. They confirm the
  instrument rather than surprise us.

### 3.4 Variables

The survey keeps the variable layer pluggable (`VariableSet`):

* **Inputs and output (positive controls).** Input coordinates are sampled independently at probe time (uniform
  bits or i.i.d. token windows), and the model's predictive distribution enters as a *soft* output column. With
  independent inputs, the marginal RI of a subset that contains the output isolates the model's own interaction
  structure, and the ground truth is exact. For the Markov-language control with spin-coded conditional
  log-odds Σ_L β_L ∏_{ℓ∈L} s_{t−ℓ}, the population marginal RI is positive exactly on the planted subsets
  {lags ∈ L} ∪ {x̂} and zero on every other subset containing x̂.
* **Binarised hidden units.** For example, ReLU on/off, or a threshold at the median. RQ0 is the question of how
  to binarise.
* **SAE features (planned bridge).** SAE features are sparse and close to monosemantic, so they binarise naturally
  as "active or not", which sidesteps the hardest part of RQ0. Feature identity can be kept across checkpoints with
  the SAE-Track recipe (initialise each checkpoint's SAE from the previous one). RI then measures *which
  combinations of concepts start being used together, and when*.

**Known limitations, measured rather than assumed:**

* RI^marg is marginal-based, so it reflects the dependence structure of the probe distribution as well as the
  model;
* discretisation can create spurious interactions, for example even-order ones from thresholding a Gaussian;
* chain RI is order-dependent;
* sample complexity grows with Π k_i, so higher orders need large N or binary variables.

---

## 4. Calibration (Project 1 = survey phase 1)

`tycho calibrate --full` runs seven experiments and writes `reports/calibration/CALIBRATION.md` plus a
machine-readable `calibration.json`, which the survey reads to set ε_min:

| Exp. | Question | Design |
|---|---|---|
| E1 null | False-positive rate, p-value uniformity, noise floor | Data with only lower-order structure (strength 0, 0.5, 1), orders k = 2–4, N grid |
| E2 bias / RMSE / coverage | Accuracy of RI_hat and RI_corr; bootstrap CI coverage | Planted θ on a k-subset on top of random background |
| E3 power | Minimum detectable RI at 80 % power, per (k, N) | From E2 replicates |
| E4 timing | Onset error in checkpoints | θ(t) switched on with a sigmoid of varying width, observed on T checkpoints |
| E5 identity | Precision and recall of *which* subset at *which* order | d = 8, planted pure 3-way + pairwise, full scan to order 4 |
| E6 discretisation (RQ0) | Interactions created by thresholding | Equicorrelated Gaussian latent, median / zero / 0.8-quantile thresholds |
| E7 chain dependence | Spread of chain RI across chains; Σ RI = TC | Random tables, mixed cardinalities |

**Positive controls with trained networks:**

* *Sparse parity* (n bits, k relevant). The label is a pure k-th-order function, so after learning the
  (k+1)-subset {x_S*, ŷ} carries exactly log 2 nats of RI and every lower-order subset carries zero. The known
  plateau-then-drop dynamics ("hidden progress") gives a timing reference.
* *Markov language with planted orders* (the token-level control). A binary order-W source whose conditional
  log-odds contain, for example, a strong pairwise term on lag 1 and a weaker three-way term on lags {1, 3}. The
  answer is known (which subsets, which orders), and the "true spectrum" can be computed from the true
  conditional. Staged learning predicts the pairwise activation before the three-way one.
* *In-context Markov chains* (Edelman et al. 2024): the unigram → bigram stage transition as a timing control.

---

## 5. The survey

### 5.1 The sky

The sky is a set of **regions**, each a task with a grid of knobs and a number of seeds (`configs/sky_v0.yaml`).

| Region | Task | Knobs (v0) | Role |
|---|---|---|---|
| control-parity | sparse parity | k, width | positive control (always RI) |
| control-markov | Markov language | coupling structure | token-level positive control (always RI) |
| grokking | modular addition | p, train fraction, width, lr, weight decay | known phenomenon |
| icl-markov | in-context Markov chains | width, lr | staged learning |
| superposition | toy models of superposition | sparsity, importance decay | geometry phases |
| parity-wide | sparse parity | n, k, width, data size, lr, weight decay | broad survey, boundary hunting |

Every object is identified by a hash of (task, configuration, seed, training configuration, sky version). This
makes enqueueing idempotent and replays exact.

### 5.2 Observation strategy

1. **Coarse uniform coverage** of every region, with ≥ 3 seeds per point ("repeated exposures"), so that signal
   can be separated from seed noise.
2. **Boundary refinement.** Lattice-neighbouring points whose majority classifications differ, or whose
   generalisation times differ by more than half a decade, are bracketed with extra seeds and midpoints on
   refinable knobs (geometric for log-scaled knobs), up to a maximum depth.
3. **Spectroscopic follow-up** by deterministic replay, with RI checkpoints densified around the largest loss
   drop. Priority order: unknown classification > transient-bearing > boundary members > a random control sample.
4. **Agent proposals.** Validated against the sky's bounds and capped per night, each with a stated hypothesis.

### 5.3 Catalog and data releases

SQLite (WAL) plus compressed per-object files. Main tables: `objects`, `summaries`, `lightcurves`,
`classifications`, `ri_runs`, `ri_events`, `transients`, `proposals`, `survey_log`, plus agent audit tables.
`tycho release --tag DR1` exports a versioned, checksummed data release: parquet and CSV tables, light-curve
archive, sky definition, calibration file, and a datasheet. Like SDSS, regular releases let others look for their
own empirical laws in the catalog.

### 5.4 The agent

The nightly run uses Claude via the Anthropic API with tool use. Its tools are read-only catalog queries plus a
small set of bounded, validated writes: proposals inside the sky, follow-up requests, labels, and log entries. It
has no shell, no file-system or network access, and every tool call is audited in the catalog. A per-run cost cap
applies; at typical sizes a nightly run costs well under a dollar. Its instructions encode the survey's epistemics:
no discovery claims without ≥ 3 seeds and a replay, numbers always relative to the calibrated noise floor, and a
preference for falsification.

---

## 6. Operations

The platform is designed to run unattended for months on one Mac mini:

* a `launchd` agent with KeepAlive runs the survey daemon;
* the supervisor restarts dead workers and re-queues stale jobs;
* SIGTERM triggers a graceful stop, and interrupted objects resume exactly from checkpoints;
* a disk guard pauses work when space runs low;
* the catalog is backed up nightly;
* a status page and dashboard are regenerated every few minutes;
* transients and health alerts go out through rate-limited push notifications (ntfy, webhook or macOS
  notifications).

Throughput is set by Python overhead, not by FLOPs: the v0 objects are tiny, so N single-threaded CPU workers
(N ≈ cores − 2) are typically faster than MPS. MPS is reserved for larger transformer objects. See
`docs/OPS_MACMINI.md`.

---

## 7. Roadmap

| Phase | Weeks | Deliverable |
|---|---|---|
| 1. Instrument | 0–2 | Calibration report (error budget); positive controls recovered; RI convention fixed with the RI authors |
| 2. Pilot survey | 2–6 | Sky v0 covered with ≥ 3 seeds; classifier and transient channel tuned; first boundary maps |
| 3. Full survey + DR1 | 6–12 | Adaptive refinement; RI follow-up of boundaries and unknowns; data release 1; first candidate laws |
| 4. Bridges | 12+ | SAE features as RI variables (RQ3); interventions by replay (perturb at the onset, measure hysteresis); public LLM checkpoint suites (e.g. Pythia) observed with the same instrument |

## 8. What a "Kepler moment" could look like

* A power law between the onset step of k-th-order interactions and the data size (or width), with an exponent
  that depends on k.
* An ordering rule: interactions are learned from low to high order, except in identifiable regions
  (for example, pure-synergy targets, or large weight decay).
* A "Hertzsprung–Russell diagram" of training: objects plotted by (onset time, final RI) fall on a few bands that
  correspond to distinct developmental programs.

Each of these is falsifiable by the catalog. Each would also give the mechanism-level theories of scaling (data
power laws, the quantization model, superposition geometry) a quantitative target: the quantization model's
"quanta learned one at a time" corresponds directly to RI activation events.

## 9. Preliminary results (pilot, 4 November 2026)

These are from a **pilot calibration**, about 1/20 of the full preset: 4 minutes on 2 cores. The full preset is
about 15 CPU-hours, i.e. 2–3 h on the Mac mini. The full report is `docs/results/calibration_pilot/CALIBRATION.md`.

**Instrument error budget** (binary variables, N = 4,096 probe samples, α = 0.01; RI^marg in nats):

| Quantity | Measured | Theory / target |
|---|---|---|
| Noise floor (99th percentile of null RI, orders 2/3/4) | 8.5e-4 / 8.4e-4 / 8.5e-4 | χ²₁: 8.1e-4 |
| False-positive rate at α = 0.01, pooled over 48 cells | 0.90 % | 1 % |
| Minimum detectable RI, 80 % power, one targeted subset | 1.3e-3 | 1.4e-3 |
| Same, inside a scan of 1,000 subsets with the full detection rule | 3.2e-3 | 3.4e-3 |
| Bias of the plug-in estimate | exactly df/2N (removed by the correction); ≤ 4 % residual | — |
| Bootstrap 95 % CI coverage | 94 % | 95 % |
| Onset-time error (median, in checkpoints) | 0 for ramp widths 0–8 | — |
| Identity of planted subsets (recall / precision) | 100 % / 95 % vs population truth (85 % vs planted only) | — |
| Chain dependence of chain RI (median spread / TC) | 2.5–2.9 % (max 16 %) | 0 for RI^marg |

Two results matter for the design:

* **Discretisation creates interactions (RQ0).** A Gaussian latent with only pairwise correlation ρ, thresholded
  at the median, gives order-3 RI of exactly 0 (sign symmetry), but order-4 RI of 1.4e-3 at ρ = 0.8. That is above
  the detection threshold. Thresholding at zero or at a quantile also creates order-3 terms (up to 1.4e-3).
  Strongly correlated binarised units therefore produce *instrument-made* higher-order events. The survey must
  report the discretisation and compare against this artefact table before interpreting such events.
* **The χ² theory is accurate down to N ≈ 256.** Noise floor, power and false-positive rates follow the asymptotic
  formulas. min_effect can therefore be set from theory and checked by calibration, rather than tuned by hand.

**Positive controls with trained networks** (single seed each; in-situ RI at 30 log-spaced checkpoints):

* *Sparse parity* (n = 20, k = 3, MLP width 512, SGD lr 0.05). Of 1,350 candidate subsets (all subsets up to order 4
  containing ŷ), exactly one is detected at the end of training: the planted {x10, x12, x15, ŷ}. Its RI^marg is
  0.688 nats (theory log 2 = 0.693). It has no lower-order precursor, i.e. it is a pure-synergy event. The onset is
  bracketed in steps [975, 1363], *before* test accuracy reaches 90 % at step 1,538.
* *Markov language* (window 6, β_{1} = 1.0, β_{1,3} = 0.8, MLP width 128, Adam). The pairwise term
  {x[t−1], x̂} rises from step ≈ 14. The three-way term {x[t−1], x[t−3], x̂} activates later, bracketed in
  [69, 96], and keeps growing until ≈ 10k steps: staged learning from low to high order, as predicted. At the end
  only the two planted subsets are detected among 21 candidates. Their values, 0.1727 and 0.1142, match the
  true spectrum computed from the source, 0.1728 and 0.1209. During the transition (steps ≈ 100–1,000), several
  non-planted subsets activate and then go extinct. These are interactions of the *model's* intermediate solution,
  not of the data, and they are exactly the kind of transient the catalog is designed to record.

## 10. Questions for discussion

1. The survey detects events with RI^marg (cheap and scannable, with a χ² null) and reports RI^cond only for
   small variable sets. Is this the right division? Should onsets of RI^cond, which measures interactions given
   all others, be tracked as a separate kind of event for small models?
2. How should higher orders be estimated with sparse tables? Is the MAHGenTa Monte-Carlo technique appropriate for
   large variable sets, beyond exhaustive scans of small subsets?
3. Binary versus categorical variables: should token variables stay categorical (df grows as Π(k−1)) or be
   binarised?
4. Temporal multiple testing across checkpoints: is BH-per-checkpoint plus persistence adequate, or should we use
   a sequential or change-point test?
5. Relation to the local learning coefficient (developmental interpretability) and to Harsanyi-interaction
   dynamics (Zhang et al.): which comparisons would be most informative?

## References

* Amari, S. (2001). Information geometry on hierarchy of probability distributions. *IEEE Trans. Inf. Theory*.
* Sugiyama, M., Nakahara, H., Tsuda, K. (2016). Information decomposition on structured space. *ISIT*.
* Enouen, J., Sugiyama, M. (2026). A complete decomposition of KL error using refined information and mode
  interaction selection. *TMLR*. arXiv:2410.11964.
* Schneidman, E., Still, S., Berry, M. J., Bialek, W. (2003). Network information and connected correlations.
  *Phys. Rev. Lett.*
* Barak, B., Edelman, B., Goel, S., Kakade, S., Malach, E., Zhang, C. (2022). Hidden progress in deep learning:
  SGD learns parities near the computational limit. *NeurIPS*.
* Power, A., et al. (2022). Grokking: generalization beyond overfitting on small algorithmic datasets.
* Nanda, N., et al. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR*.
* Nakkiran, P., et al. (2020). Deep double descent. *ICLR*.
* Edelman, B. L., Edelman, E., Goel, S., Malach, E., Tsilivis, N. (2024). The evolution of statistical induction
  heads: in-context learning Markov chains.
* Elhage, N., et al. (2022). Toy models of superposition.
* Michaud, E. J., et al. (2023). The quantization model of neural scaling. *NeurIPS*.
* Bahri, Y., et al. (2021/2024). Explaining neural scaling laws. *PNAS*.
* Liu, Y., Liu, Z., Gore, J. (2025). Superposition yields robust neural scaling. *NeurIPS*.
* *Tracking the feature dynamics in LLM training: a mechanistic study* (SAE-Track).
* *Two-phase dynamics of interactions explains the starting point of a DNN learning over-fitted features*
  (arXiv:2405.10262); *Towards the dynamics of a DNN learning symbolic interactions* (arXiv:2407.19198).
