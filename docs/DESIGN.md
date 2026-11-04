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

**Chain refined information.** Along a complete hierarchical chain H_0 ⊂ H_1 ⊂ … in which each step adds one
subset S_j whose proper subsets are already present,

  RI_C(S_j) = D(P‖P_{H_{j−1}}) − D(P‖P_{H_j}) = D(P_{H_j}‖P_{H_{j−1}}) ≥ 0.

From the independence model to the saturated model, these terms sum to the total correlation
Σ_i H(X_i) − H(X). This is a *complete* decomposition of the KL error into non-negative contributions, one per
mode interaction. For two variables it is the mutual information. The values depend on how incomparable subsets
are ordered in the chain. The code exposes the chain explicitly (`graded_chain`, `greedy_selection` for mode
interaction selection).

**Local refined information** of a subset S (the per-subset "spectral line" used for event detection) is the
top-order term of S's own marginal:

  RI(S) = D(P_S ‖ (P_S)_{∂S}),  where ∂S = all proper non-empty subsets of S.

It equals RI_C(S) in every chain in which, when S is added, the only interactions involving S's variables are S's
own proper subsets. It is chain-independent and non-negative. It equals the mutual information for |S| = 2 and
vanishes iff P_S has no |S|-th-order interaction. It also coincides with the top-order term of Amari's
hierarchical decomposition and with Schneidman et al.'s connected information of top order.

> **For discussion.** We could not access the full text of the TMLR paper from the build environment, so the
> canonical chain and the notation should be checked against it. The implementation keeps both
> `chain_decomposition` (any chain) and `local_ri`, so we can adopt whichever convention is canonical.

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

### 3.4 Variables

The survey keeps the variable layer pluggable (`VariableSet`):

* **Inputs and output (positive controls).** Input coordinates are sampled independently at probe time (uniform
  bits or i.i.d. token windows), and the model's predictive distribution enters as a *soft* output column. With
  independent inputs, the local RI of a subset that contains the output isolates the model's own interaction
  structure, and the ground truth is exact. For the Markov-language control with spin-coded conditional
  log-odds Σ_L β_L ∏_{ℓ∈L} s_{t−ℓ}, the population local RI is positive exactly on the planted subsets
  {lags ∈ L} ∪ {x̂} and zero on every other subset containing x̂.
* **Binarised hidden units.** For example, ReLU on/off, or a threshold at the median. RQ0 is the question of how
  to binarise.
* **SAE features (planned bridge).** SAE features are sparse and close to monosemantic, so they binarise naturally
  as "active or not", which sidesteps the hardest part of RQ0. Feature identity can be kept across checkpoints with
  the SAE-Track recipe (initialise each checkpoint's SAE from the previous one). RI then measures *which
  combinations of concepts start being used together, and when*.

**Known limitations, measured rather than assumed:**

* local RI is marginal-based, so it reflects the dependence structure of the probe distribution as well as the
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

## 9. Preliminary results

*To be filled from `reports/calibration/CALIBRATION.md` and the positive-control runs (see §4).*

## 10. Questions for discussion

1. Which RI convention should the survey adopt for per-subset events: local RI, or chain RI under a canonical
   chain? Is there a preferred chain (graded, or greedy mode-interaction selection) for the decomposition reports?
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
