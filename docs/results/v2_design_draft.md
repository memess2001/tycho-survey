# Sky v2 design draft: does the θ view see what the η view does not?

Status: **draft for discussion**, 12 November 2026. This is not a pre-registration. Nothing here is fixed until the
v2 pre-registration is written, before any v2 data exist. Background: [v1_results.md](v1_results.md); why the plan
changed: [ROADMAP.md](../ROADMAP.md); the discussion that led to it:
[notes/discussion_2026-11-12_after_v1.md](../../notes/discussion_2026-11-12_after_v1.md).

## 1. Why v2 changes

The roadmap shown on 10 November planned four skies after v0:

| sky | plan of 10 November |
|---|---|
| v1 | sharper known-answer tasks (done: DR1) |
| v2 | correlated inputs: RI on in-context Markov chains, and how to use RI with correlated variables |
| v3 | hidden variables: SAE features as RI variables; grokking internals |
| v4 | closer to real models: small transformers on text, public checkpoint suites |

v1 settled what it was built for: the instrument finds planted interactions (288 of 288 parity runs, 202 of 202
Markov runs). It also showed what it cannot settle. With uniform probes, RI on a subset is a function of the
Walsh–Fourier coefficients on its sub-subsets, so no uniform-probe sky can show that RI adds anything to a Fourier or
moment analysis. v1 found one case where the two disagree on the same subset. In chain runs of {1}:1.0 with
{1,2,3}:1.0, the coefficient f{1,2,3} is 0.12 to 0.14 while RI stays near 1e-4, because learned lower-order terms
cancel the top-order log-linear term.

The research line needs to pass a series of gates before it can say anything useful about language models:

1. the instrument is trusted on known answers (passed in v1);
2. **the θ view sees something the η view does not, judged by a referee that is neither;**
3. a measurement predicts a later event, across settings;
4. an intervention based on it works in a toy setting (causal);
5. it transfers: small transformers, then pre-trained models.

Gate 2 decides whether RI is worth carrying toward language models. If it fails, everything after it can be done
with cheaper moment and Fourier tools. v2 therefore keeps the original direction (correlated inputs) but states its
goal as gate 2, adds explicit referees, and moves small transformers forward from v4. The lab's GPU servers (one
with RTX PRO 6000 GPUs, one with five A100 80 GB) make many-seed transformer skies affordable. SAE variables (old
v3) wait until gate 2 is settled.

## 2. The question, made precise

Take binary variables (inputs or features, and the model's binarised output ŷ) and their joint distribution under
some probe distribution.

* **η view (moments).** The Walsh–Fourier coefficient under uniform probes, f_u(A) = E_u[(2ŷ − 1) χ_A]: a property
  of the function, which ignores the data. The data moment m_D(A) = E_D[(2ŷ − 1) χ_A] under the data distribution D:
  correlation-type.
* **θ view (log-linear interactions).** Marginal RI, the survey's current spectral line: the top-order θ of the
  subset's own marginal table (RI^marg of Enouen & Sugiyama). Conditional RI (RI^cond, "purified"): the top-order
  term of the subset with the other variables of the table conditioned on. It is implemented as
  `tycho.ri.conditional_ri` but not used in any sky so far.

Under uniform probes the two views carry the same information (v1). They can still give different answers in two
ways. Both examples below were computed with tycho's own functions.

1. **Attribution at fixed information.** A saturating output turns first-order log-odds into higher-order moments.
   With x uniform on {−1, 1}³ and E[2ŷ − 1 | x] = tanh(x₁ + x₂ + x₃), the log-odds have no interaction, and the
   local RI of {x₁, x₂, x₃, ŷ} is 0. The Walsh coefficient f_u({1,2,3}) is −0.32. The v1 cancellation case is of
   this kind. Which description matches what a network computes is decided by neither view; it needs a referee.
2. **Correlated probes.** Let ŷ copy x_i (with 10 % noise) and let x_j be correlated with x_i (ρ = 0.6) but unused.
   The data moment of {j} and the marginal RI of {x_j, ŷ} (0.12 nats) both report a dependence on x_j. The
   conditional RI of {x_j, ŷ} with x_i in the table is 0. The uniform Walsh coefficient is also 0, but only because
   it evaluates the network off its data distribution, on inputs that may never occur (in a chain, impossible
   contexts).

## 3. Referees: what counts as "seeing more"

* **R1, mechanism.** Networks whose computation is known: hand-set weights, or trained networks with a planted
  structure confirmed by ablation. Score: which measure recovers the mechanism's interaction sets and orders
  (precision and recall).
* **R2, prediction.** Pre-registered forecasts from measurements up to step t of a later event: the induction phase
  change, a planted term's onset, final generalisation. Score: against moment-based and loss-based predictors.
* **R3, intervention.** Ablate or clamp what each measure singles out. The measure whose choices produce the larger
  and more specific behavioural effect wins.

**Gate 2 passes** if the θ view beats the η view on R2 or R3 by a pre-registered margin in at least one task family,
and the win does not follow from the definitions alone. R1 by itself shows that the two views answer different
questions, which is expected; it calibrates, it does not pass the gate. The other outcomes are informative too:

| outcome | meaning | next step |
|---|---|---|
| θ wins on R2 or R3 | RI detects something moments miss | carry RI toward language models (v3, v4) |
| tie | RI's value is in attribution (description), not detection | moment tools for detection, RI for interpretation |
| η wins | the θ view misleads in these settings | reconsider the instrument |

## 4. Part A: θ and η on known answers (CPU server)

**A0. Settle the definitions.**
* Marginal and conditional RI are both computed, side by side, and the referees decide between them (decided 12
  November; see ROADMAP.md). Conditional RI needs the full table over the variables it conditions on, so it is
  limited to about 16 to 20 binary variables, or to a chosen conditioning set.
* Storage: for every kept subset, the model's mean output per cell of the subset (for small inputs, the output on
  every distinct probe pattern) and the mean output f(∅). Then θ and η coefficients are both exact. A3 showed that
  rebuilding them from stored coefficients fails exactly where the two views diverge.
* Probe distribution: uniform, data, or both.
* Calibration under correlated probes. The v0 calibration (noise floor, false positives, min_effect) assumed
  independent uniform probes, so it has to be redone with correlated nulls (new calibration experiments E7 onward).

**A1. Constructed networks (no training).** The exact value of every measure is known in each case. The check is
that the estimators reproduce them: an instrument check, not a finding.
* additive log-odds through a saturating output (θ: no interaction; η: higher-order moments), with the logit scale
  (saturation) and the strength of the lower-order terms as two crossed knobs. A3 found divergences in DR1 only
  where both are large;
* probability-scale products such as soft AND and XOR (both nonzero, at different orders);
* the v1 cancellation pattern ({1}, {2}, {3} with {1,2,3});
* correlated inputs: Markov-chain and Ising-type inputs with tunable ρ, where unused inputs are correlated with used
  ones.

**A2. Trained networks with planted structure under correlated inputs.**
* v1's Markov language with RI probes drawn from the training distribution (chain) as well as uniform;
* sparse parity on correlated input bits at tunable ρ;
* the v1 cancellation sets, starting with {1}:1.0 + {1,2,3}:1.0 on chain contexts, the only DR1 cell where the
  views diverge: rerun with exact storage and an ablation of the three-way component (R3).

Questions: the false-interaction rate of each measure against the planted mechanism (R1), with ablations (R3).

**A3. What stored v1 data can already answer (no new training).**
* RI against the single-mode R(f), and against RI rebuilt from the coefficients, over all subsets stored in DR1
  (v1 Amendment J): where and how often the two views diverge. **Done:** [v2_a3_stored_data.md](v2_a3_stored_data.md).
* A dense replay of the Markov order-4 and order-5 objects (v1 Q5).
* Re-specified system checks 9 and 10 (a minimum prefix size; anomaly groups by configuration).

## 5. Part B: induction heads in small transformers (CPU pilot, then GPU servers)

**Why induction heads.** An induction head attends to the token after the previous occurrence of the current token
and copies it. It meets all three needs of gate 2 at once:
* the inputs are naturally correlated (sequences);
* the answer is known: the mechanism, the sharp and reproducible phase change in training (Olsson et al. 2022), and
  existing measures (in-context-learning score, prefix-matching score);
* there is a referee: head ablation.

It also matters for real language models, which makes it a natural first step from toy MLPs toward them.

**Objects.** Attention-only and standard transformers, 1 to 3 layers, 1 to 8 heads, d_model 32 to 256 (about 0.05M
to 5M parameters). They are trained on synthetic sequences that mix bigram (Markov) statistics with repeated random
segments, with 10 or more seeds per configuration.

**Controls built in.**
* negative: 1-layer models, where induction by composition of two heads cannot form;
* negative: data without repeats;
* positive: 2-layer models on data with repeats.

**Knobs (candidates).** Layers, heads, d_model, repeat fraction, vocabulary size, context length, learning rate,
warm-up.

**Variables (open design question).**
* **Token-level events** under the data distribution, which gives correlated probes. Candidates: "the current token
  occurred at position k", "the token after that occurrence equals the candidate", and ŷ = "the model's top
  prediction copies the token after the previous occurrence". The induction mechanism is a conjunction: a
  higher-order interaction among events.
* **Component-level.** Random head-ablation patterns, a design we choose, so we control the probe distribution,
  with ŷ = "in-context loss below a threshold". The composition of a previous-token head with an induction head is
  then a pairwise interaction between heads.

**Questions (candidates; to be fixed in the pre-registration).**
* B1 (R1, R3): do marginal and conditional RI single out the composing heads or events that ablation confirms, and do
  moments add spurious ones?
* B2 (R2): does the θ-measured onset of the induction interaction precede the phase change in loss and ICL score,
  and does it come earlier than moment-based and loss-based progress measures, by a pre-registered lead?
* B3: how does that onset scale with d_model, heads and repeat fraction? (Scale is a knob from the start; §6.)

**Compute: CPU first, then GPU.** The smallest models (2-layer attention-only, d_model 32 to 64, under about 1M
parameters) train on the CPU server; the run rate will be measured. That is where the transformer task, the
variables and the controls are first built and checked, in a pilot of the smallest sizes. The GPU servers take over
for the larger sizes of the scale knob and longer contexts. There a run takes minutes on one GPU, and several runs
share a GPU.

**Engineering before Part B.**
* for the GPU stage only: GPU workers in the scheduler (device assignment, several small jobs per GPU) and RI scans
  on the GPU;
* a transformer task module and data generators;
* a variable-extraction layer that maps a model and its data to binary variables;
* moment baselines under the data distribution.

## 6. Scale: designed in, not extrapolated at the end

* Model size is a knob in Part B from the start. A regularity is extrapolated only if its exponent or ordering is
  stable across sizes; one that vanishes with size is reported as such.
* Later anchors (v3, v4):
  * public checkpoint suites, such as Pythia (eight sizes from 70M to 12B parameters, the same data order, 154
    checkpoints each; multi-seed releases exist for the smaller sizes). They let the same measurement run across
    scale without training anything.
  * a few GPT-2-small (124M) runs from scratch on the A100 server, as an anchor we trained ourselves.
* What can be expected to transfer is a regularity in how things change with scale, or a mechanism. A specific
  number or trick usually does not. Scaling laws, μP hyperparameter transfer and induction heads across model sizes
  are examples of the first kind.

## 7. Deferred

* SAE features as RI variables (old v3).
* Grokking internals: v1's Q7 measure was not specific to grokking; it will be revisited with better variables.
* Post-training experiments on pre-trained open models (gate 5).

## 8. Order of work

The CPU server comes first, the GPU servers after it.

1. A3 (stored data). It informs the thresholds and cases of A0 and A1.
2. A0 and A1: definitions, calibration under correlated probes, constructed cases (instrument checks). CPU.
3. The Part A pre-registration (criteria, referees, margins), before any Part A data. Then A2. CPU.
4. The Part B pilot on the smallest models: transformer task module, variable extraction, controls. CPU.
5. The Part B pre-registration. Pilot results are disclosed in it, as v1's Amendment 1 disclosed its pilots.
6. The Part B survey at the larger sizes, after GPU workers are added to the scheduler. GPU.
7. Analysis with independent verification, as for v1.

## 9. Questions for discussion

1. For correlated variables, should the survey's spectral line be marginal RI, conditional RI, or both? (Working
   decision: both, side by side, with the referees deciding.) Conditional RI depends on which variables are in the
   table. How should that set be chosen in a transformer?
2. Is the referee framing right? Is "θ beats η on prediction or intervention" the right bar for gate 2, and what
   margin should count?
3. Interaction measures are known to depend on the scale: additive in log-odds is not additive in probability. Is
   there a principled reason, from information geometry, to prefer the θ view for describing what a network
   computes?
4. Part B variables: token-level events, head-ablation patterns, or both?
5. GPU server time: how much can the survey use, and when?

## References

* J. Enouen, M. Sugiyama. A complete decomposition of KL error using refined information and mode interaction
  selection. TMLR (2026); arXiv:2410.11964.
* C. Olsson et al. In-context learning and induction heads (2022).
* S. Biderman et al. Pythia: a suite for analyzing large language models across training and scaling (2023).
* G. Yang et al. Tensor Programs V: tuning large neural networks via zero-shot hyperparameter transfer (2022).
