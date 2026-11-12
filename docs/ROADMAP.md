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
| v2 | gate 2: θ vs η with referees. Part A: correlated inputs and constructed cases (CPU). Part B: induction heads in small transformers (pilot on the CPU server, then the GPU servers) | design draft | [v2_design_draft.md](results/v2_design_draft.md) |
| v3 | depends on gate 2. Candidates: transformers at several sizes, Pythia checkpoints across scale, SAE features as variables | open | |
| v4 | closer to real models: text-trained transformers, own GPT-2-small runs, pre-trained open models (gates 3 to 5) | open | |

## Change log

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
