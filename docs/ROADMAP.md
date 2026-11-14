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
| v2 | gate 2: θ vs η with referees. Part A: correlated inputs and constructed cases (CPU). Part B: induction heads in small transformers (pilot on the CPU server, then the GPU servers) | A0, A1, A3 done; Part A pre-registered | [v2_design_draft.md](results/v2_design_draft.md), [v2a_preregistration.md](results/v2a_preregistration.md) |
| v3 | depends on gate 2. Candidates: transformers at several sizes, Pythia checkpoints across scale, SAE features as variables | open | |
| v4 | closer to real models: text-trained transformers, own GPT-2-small runs, pre-trained open models (gates 3 to 5) | open | |

## Change log

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
  corrections it asked for.
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
