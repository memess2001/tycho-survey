# Roadmap

A living document: what each sky is for, the current plan, and a dated record of every change to the plan with its
reasons. Past presentations (for example `docs/presentation/weekly-2026-11-10/`) are kept as they were shown.

## Principle

Instrument first. A sky is designed to answer a question its predecessor could not. The research line passes five
gates before it can say anything useful about language models:

1. the instrument is trusted on known answers;
2. the θ view (RI, log-linear interactions) sees something the η view (moments, Walsh–Fourier) does not, judged by a
   referee that is neither: mechanism, prediction or intervention;
3. a measurement predicts a later event, across settings;
4. an intervention based on it works in a toy setting;
5. it transfers: small transformers, then pre-trained models.

## Current plan (12 November 2026)

| sky | goal | status | documents |
|---|---|---|---|
| v0 | system test: the platform runs end to end | done; DR0 | [v0_findings.md](results/v0_findings.md) |
| v1 | sharper known-answer tasks with uniform probes (gate 1) | done; DR1 | [v1_preregistration.md](results/v1_preregistration.md), [v1_results.md](results/v1_results.md) |
| v2 | gate 2: θ vs η with referees. Part A: correlated inputs and constructed cases (CPU). Part B: induction heads in small transformers (pilot on the CPU server, then the GPU servers) | A0, A1, A3 done; Part A run: "η wins" (1 of 12; see v2a_results.md); Part B run as a formation study (DR2b): B1 and B2 met, mechanism comparison a tie (see v2b_results.md) | [v2_design_draft.md](results/v2_design_draft.md), [v2a_preregistration.md](results/v2a_preregistration.md), [v2a_results.md](results/v2a_results.md), [v2b_preregistration.md](results/v2b_preregistration.md), [v2b_results.md](results/v2b_results.md) |
| v3 | depends on gate 2. Candidates: transformers at several sizes, Pythia checkpoints across scale, SAE features as variables | open | |
| v4 | closer to real models: text-trained transformers, own GPT-2-small runs, pre-trained open models (gates 3 to 5) | open | |

## Change log

### 20 November 2026: Part B run and analysed; a discussion about the research line

* **Part B** (100 objects, DR2b; [v2b_results.md](results/v2b_results.md); verified independently).
  * Every instrument check passed.
  * B1: the circuit formed in 60 of 60 runs; larger models change earlier (phase change at a median step of 692, 339
    and 211 for d_model 48, 96 and 192). Both predictions met.
  * B2: in 60 of 60 runs the induction heads form before the previous-token head becomes clean. Prediction met.
  * The θ/η mechanism comparison is a tie. η is favoured at C_U only when pairs of a circuit head with the layer-0
    t − 2 ("context") head count as non-circuit pairs ("referee-dependent", which does not count as ahead). That
    was the expected outcome, and it does not count toward gate 2.
  * Not predicted: at d_model 192, 9 of 20 runs grow a strong t − 2 head in layer 0.
* **The research line** (the researcher, 20 November). The aim is to find real anomalous phenomena in the survey
  and, from them, regularities that are verified, general and hold across scale. Whether RI sees more than the η
  view is a question about a tool, not the goal. The gates above are written around RI and may be rewritten. A
  discussion draft ([notes/reframing_draft_2026-11-20.md](../notes/reframing_draft_2026-11-20.md)) is being
  discussed with the advisor session and Antigravity; nothing is decided yet, and the plan above is unchanged.

### 16 November 2026 (later): Part B becomes a formation study; gate 2 stays open

* **Review of the first Part B draft** ([v2_analysis/v2b_prereg_review.md](results/v2_analysis/v2b_prereg_review.md)).
  * On this task no prediction referee can tell θ and η apart: the circuit forms abruptly, and every measure crosses
    any control-calibrated threshold at the same early checkpoint.
  * The mechanism referee's expected η win came from counting a functional layer-0 "context" head (it attends to
    t − 2) as outside the circuit. Excluding it gives a tie.
  * The design draft does not let a mechanism referee alone pass gate 2.
* **Decision** (the researcher, 16 November). Part B is a study of how the induction circuit forms and how this
  changes with size (B1, B2). The θ/η comparison of how well each agrees with head roles is reported, but does not
  count toward gate 2. Gate 2 stays open for a later test with a real prediction or intervention referee, designed
  with Prof. Sugiyama.
* **Next.** A consistency check of the revised pre-registration; the analysis code and its verification; the run.

### 16 November 2026: Part B pilot 1 analysed; tiled layout; pilot 2 ready

* **Pilot 1** (DR2b-pilot, 48 objects). The controls behaved as expected. The two-layer circuit formed fully in
  only 2 of 12 runs, and never at minimum repeat 12. In both full circuits, η's top head pair was the composition;
  θ's was not.
* **Local experiments.** Variants of the single-repeat layout (shorter minimum repeat, shorter repeats, a higher
  learning rate, longer training) formed the circuit in 0 to 1 of 3 to 6 runs each. Tiling a random segment over
  the whole sequence formed it in 6 of 6, and the 1-layer control formed none. On those circuits:
  * with 4 heads per layer, η's top pair was the composition in 3 of 3 and θ's in 1 of 3;
  * with 2 heads per layer, every measure put the redundant induction-head pair first.

  So the mechanism referee must count both composition and redundancy pairs.
* **Built.** The tiled layout in the induction task; an induction score that counts every matching earlier
  position; a duplicate-token score; `configs/sky_v2b_pilot2.yaml` and `ops/linux/start_v2b_pilot2.sh`.
* **Next.** Pilot 2 on the server, which sets the learning rate for the prediction referee; then the Part B
  pre-registration.

### 15 November 2026 (later): Part B started, pilot ready

* **Plan** ([v2b_pilot_plan.md](results/v2b_pilot_plan.md)). Part B studies induction heads in small
  attention-only transformers. Following Part A's decision, η is the detector and RI its descriptive companion.
  The variables are head-ablation patterns, all 2^H enumerated, as in Part A's pattern mode. The mechanism referee
  is the head roles read from attention patterns, which neither view uses.
* **Probes on this machine** (to be disclosed in the Part B pre-registration):
  * A fixed repeat offset lets one positional head solve the task, so the repeat now has a random length and
    position.
  * Whether the two-layer circuit (previous-token heads plus induction heads) forms depends on the seed and on the
    minimum repeat length. Some runs stall in a positional heuristic.
  * In one formed circuit, η's top head pair was a previous-token head with an induction head; θ's top pairs were
    redundant induction heads.
* **Built.** The induction task, head masks and attention maps in TinyGPT, `configs/sky_v2b_pilot.yaml` (48
  objects) and `ops/linux/start_v2b_pilot.sh`.
* **Next.** The pilot on the CPU server, then the Part B pre-registration.

### 15 November 2026: Part A run; "η wins" by the pre-registered rule

* **Run.** All 440 objects done on the CPU server in about 3 hours of wall time. The analysis took 7 minutes.
  Every instrument check passed.
* **Result** ([v2a_results.md](results/v2a_results.md); verified independently from the data release DR2a, which
  reproduced every number and label):
  * η wins 1 of 12 comparisons (T2 at C_U) and θ none. Four comparisons are value-dependent and seven are ties.
  * The two views are largely interchangeable on these referees. Large gaps occur only where outputs saturate (the
    {1, 2, 3} chain configurations, "1:2.0", parity at ρ_train = 0.8). There θ is mostly, but not always, the worse
    of the two.
  * The single win is weak. It rests mostly on one twin set, in which θ's values stay below the presence floor and
    the added term is close to unidentifiable.
  * Q1, Q3 and Q4 hold. Q2 (the divergence law of A1) fails on trained networks.
* **Consequence (decision table).** Gate 2 is not passed. Part B uses the η view as its detector, with RI as a
  descriptive companion. Part B's pre-registration needs a referee that intervenes on the network.

### 14 November 2026 (evening): Part A engineering done; ready to run

* **Built** (pre-registration §6):
  * pattern-mode observation, which stores the log-odds on every input pattern and the probe counts;
  * exact distributions;
  * correlated parity;
  * seed offsets with data_seed = seed;
  * the offline analysis;
  * `configs/sky_v2a.yaml` (440 objects) and `ops/linux/start_v2a.sh`;
  * pattern files in data releases.
* **Pilot** (32 objects, seeds 900–901). Every instrument check passed and every run converged. Markov runs take
  about 9 s and parity runs about 1.5 s on this machine.
* **Verification.** An independent agent reproduced every pilot number to rounding level. It found one defect, a
  logistic fit that could stall without a flag, which is now fixed
  ([v2_analysis/v2a_analysis_verification.md](results/v2_analysis/v2a_analysis_verification.md)).
* **Disclosure.** The pilot's smoke-test statistics (2 seeds) were seen and point to "η wins". They are recorded in
  the pre-registration's amendments; no definition was changed after seeing them.
* **Next.** On the CPU server: `git pull`, then `bash ops/linux/start_v2a.sh`. When all 440 objects are done, run
  the manifest and analysis commands of [OPS_LINUX.md](OPS_LINUX.md).

### 14 November 2026 (later): Part A pre-registered

* **What is fixed** ([v2a_preregistration.md](results/v2a_preregistration.md)):
  * Measures: two links (θ: log-odds, logistic; η: means, least squares) at four references (marginal or
    conditional table × training or uniform weights), as exact values from the network's output on every input
    pattern.
  * Runs: 320 Markov runs (16 coupling sets, chain and iid) and 120 parity runs on correlated bits, on new seeds.
  * Tests: T1, does a measure of spurious structure track fragility under a distribution shift? T2, does a
    measure's change between twin networks (data with and without one planted term) track how differently they
    behave? 12 comparisons; a link wins only with a margin of 0.10 in Spearman ρ, a Holm-adjusted bootstrap
    p < 0.05, both η values and the same sign with a squared-error outcome.
* **Review.** An independent review of the first draft
  ([v2_analysis/v2a_prereg_review.md](results/v2_analysis/v2a_prereg_review.md)) found that the pooled statistic
  ranked configurations rather than networks, that the draft's η value grows without bound as the output saturates,
  and that the registered seeds would have retrained DR1's networks. All three were fixed, with the other
  corrections it asked for. A second agent then checked the revision; it found two instrument checks that would
  have failed on correct data (sampling and slow mixing, not bugs), a bootstrap that ignored seeds shared across
  configurations, and overstatements about the preview. Those were fixed too, and the seeds moved to 100–109
  (Markov) and 200–219 (parity), clear of every DR1 seed.
* **Disclosure.** The review's pilot (seeds 0–3) previews the tests. It points to "η wins" under both the draft's
  and the revised rules. This is disclosed in §0 of the pre-registration; the expectations there say so.
* **Next.** The engineering of §6 (pattern storage, exact pmfs, correlated parity, seeds, offline analysis), a
  pilot with non-confirmatory seeds, then the run on the CPU server.

### 14 November 2026: A0 and A1 done

* **A0** ([v2_a0_a1.md](results/v2_a0_a1.md); independently verified, corrections applied):
  * The four measures are fixed, with one gate for all of them:
    * θ view: marginal RI and conditional RI;
    * η view: the uniform Walsh coefficient and the data moment, put in nats with the base rate.
  * Per-cell counts and sums are the storage format.
  * Conditional RI has a fast estimator (a weighted logistic regression, checked against iterative proportional
    fitting).
  * E8 calibrated both RI variants under correlated probes, including nulls that are correlated with used inputs.
    The χ²_1 test is valid on average and conservative (pooled false-positive rates 0.22 to 0.34 % at α = 0.01;
    single cells up to 2.7 %). Both variants use min_effect 0.001 at n ≥ 4,096; at n = 1,024 the thresholds are
    0.003 to 0.0044.
* **A1** (constructed known answers):
  * The views diverge only with saturation (largest |m| from about 0.74 at min_effect 0.001) and lower-order terms
    together, in both directions by construction.
  * Under correlated probes, marginal RI and the data moment report many correlated but unused inputs, while
    conditional RI and the uniform Walsh coefficient report none.
  * Conditional RI loses power when inputs are redundant: about 200 times smaller at ρ = 0.95.
  * The uniform coefficient's clean record relies on the teacher being defined off the data distribution, which a
    trained network is not.
* **Next.**
  * A2 tests this on trained networks, starting with the {1}:1.0 + {1,2,3}:1.0 chain cell.
  * The pre-registration of Part A comes first. Its sharpest question: does the uniform coefficient keep its
    advantage when the network was trained on correlated data?

### 12 November 2026 (later): marginal and conditional RI side by side; A3 done

* **Decision.** The discussion with Prof. Sugiyama is postponed: he is busy for two weeks and next week's meeting is
  cancelled. v2 does not wait for it. Both marginal and conditional RI are computed side by side, and the gate-2
  referees decide between them. The other open questions in v2_design_draft.md §9 stay open, and their answers can
  still change the v2 pre-registration before it is written.
* **A3 result** ([v2_a3_stored_data.md](results/v2_a3_stored_data.md); independently verified). In DR1 the θ and η
  views agree almost everywhere. They diverge only where the output is saturated and lower-order coefficients are
  large together. Every real divergence, in both directions, is in one configuration ({1}:1.0 + {1,2,3}:1.0,
  chain). A naive rebuild of θ from stored coefficients fails there. An exact reconstruction from regenerated
  probes works. The parity θ-only flags were artefacts of the uncentred η estimator.
* **Consequences for v2 Part A.**
  * Saturation and lower-order strength become crossed knobs.
  * f(∅) is stored, and the η coefficient is computed on uniform cells; per-cell counts and sums are stored for
    correlated probes.
  * That one DR1 cell is the first A2 experiment, with an ablation referee.

### 12 November 2026: v2 retargeted to gate 2; small transformers moved forward

**Before.** The roadmap of 10 November: v2 correlated inputs (RI on in-context Markov chains, settle RI for
correlated variables); v3 SAE features as hidden variables and grokking internals; v4 small transformers on text
and public checkpoint suites.

**After.**
* v2 keeps correlated inputs, but its stated goal is now gate 2, with explicit referees and pass, tie and fail
  outcomes fixed in advance.
* Small transformers with a known answer (the formation of induction heads) move from v4 into v2 (Part B).
* Model size becomes a survey knob from the start.
* Public checkpoint suites (Pythia) and a few own GPT-2-small runs become the scale anchors of v3 and v4.
* SAE variables and grokking internals are deferred.

**Why.**
1. v1 showed that uniform probes cannot test RI's added value. On those tasks RI is a function of the Walsh
   coefficients, so check 2 is a consistency test (v1_results.md, §1). Settling RI "for correlated variables" is
   not enough on its own: it needs a referee that is neither view.
2. v1 produced the first case where the two views disagree on one subset: a coefficient of 0.12 to 0.14 with RI
   near 1e-4, from a cancellation by learned lower-order terms (v1_results.md, §4). Gate 2 is where this kind of
   difference either matters or does not.
3. The lab has GPU servers (RTX PRO 6000; five A100 80 GB) besides the 88-core CPU server. Many-seed surveys of
   small transformers therefore become affordable now rather than at v4.
4. Results on small models may not hold at scale. A regularity is worth extrapolating only if it is measured as a
   trend across sizes, so size is built in from the start. What tends to transfer is a regularity in how things
   change with scale, or a mechanism, not a specific number or trick.
5. Induction heads give a known answer inside a transformer: a known mechanism, a reproducible phase change, existing
   measures and an ablation referee. It is the most direct step from toy MLPs toward language models.

**Order of work** (decided the same day): the CPU server first, for Part A and a Part B pilot on the smallest
transformers; then the GPU servers, for Part B at larger sizes.

**Discussion.** [notes/discussion_2026-11-12_after_v1.md](../notes/discussion_2026-11-12_after_v1.md) (in
Chinese). **Pending:** discussion with Prof. Sugiyama, especially the choice between marginal and conditional RI
for correlated variables, and the gate-2 bar (v2_design_draft.md, §9).
