# Sky v2, Part B: pilot plan (induction heads in small transformers)

Status: plan for the CPU pilot, 15 November 2026. This is not a pre-registration. The Part B pre-registration comes
after the pilot and discloses it, as Part A's did. Background: [v2_design_draft.md](v2_design_draft.md) §5 and
[v2a_results.md](v2a_results.md).

## 1. What Part A changes

* **Decision table.** Part A ended in "η wins" (1 of 12 comparisons; a weak win). So Part B uses the η view as its
  detector, and RI is computed side by side as a descriptive companion.
* **Stronger referees.** Part A's referees were behavioural and weak. Part A §7 asked that Part B include referees
  of the kinds the design draft calls R1 (mechanism) and R2 (prediction), and one that intervenes on the network
  itself. The plan is to use:
  * R1, mechanism: head roles identified from attention patterns, independently of either view;
  * R2, prediction: the training step of the induction phase change, forecast from earlier checkpoints.

  The ablation pattern is the probe design itself (§3). So the measures are computed from interventions on the
  network, and R1 checks what they single out against an independent reading of the circuit. Whether this gives
  θ a second, fair chance at gate 2 is a question for the pre-registration (§6, question 1).

## 2. What a quick probe on this machine showed (to be disclosed)

Setup: 2-layer attention-only TinyGPT, d_model 64, 4 heads per layer, vocabulary 64, sequences of 64 tokens,
Adam lr 1e-3, batch 32, one seed. Scratch code, not in the repository.

1. **The repeat must start at a random position.**
   * With a fixed offset (random first half, second half a copy), the model solves the task with one layer-1 head
     attending by position. That head's induction score is 0.97, and no two-layer circuit forms.
   * With a random repeat length (12 to 32 tokens) the shortcut is gone.
2. **Controls behave as expected.**
   * 2-layer: the loss on repeated tokens falls from 4.17 to 0.06 within 5,000 steps, with the drop around steps
     400–800.
   * 1-layer: it stays near 3.1.
   * Training runs at about 50 steps per second on one core.
3. **A clean circuit forms.**
   * Layer-1 heads 0, 2 and 3 are previous-token heads (attention to t − 1: 0.66, 0.41, 0.43).
   * The four layer-2 heads are induction heads (attention to the token after the previous occurrence: 0.82 to
     0.95).
   * Ablating all heads gives p(correct) 0.016; the intact model gives 0.96.
4. **The two views read the circuit differently.** Measures taken over all 2^8 zero-ablation patterns (uniform
   design, conditional reference, probability of the correct repeated token as output):
   * Both views put the single heads 0 (a previous-token head) and 6 (an induction head) first.
   * η's top pair is (0, 6), a previous-token head with an induction head: the composition. Bounded η is 0.014.
   * θ ranks that pair low (0.003); its top pairs are induction heads together (redundancy).

   This is one model and one seed, so it is a preview and not a result.

5. **Whether the circuit forms depends on the seed and the data.** A first full-size run under the runner showed
   this; it used d_model 64, 4 heads per layer, minimum repeat 12, random start and seed 300.
   * By 6,000 steps it had not formed the circuit. No head attended to t − 1; the layer-0 heads attended about 12
     positions back, a positional heuristic tied to the minimum repeat length. The loss on repeated tokens stalled
     near 1.1.
   * Four variants, one seed each, 8,000 to 24,000 steps:

     | variant | outcome |
     |---|---|
     | minimum repeat 4 | the circuit formed by step ~6,000: two previous-token heads at 0.97 and 0.99, three induction heads at 0.93 |
     | minimum repeat 12, another seed | formed by step 3,000–6,000 |
     | a fixed start | formed fastest |
     | vocabulary 256 | stalled in the non-circuit solution (loss 1.4) |

   * So the pilot maps the formation rate and its timing across seeds, the minimum repeat length and the head
     count, and trains longer.
6. **Cost.** On one core of this machine, a 2-layer, 8-head run takes about 5 minutes for 12,000 steps. The 256
   ablation patterns take about 13 s per checkpoint, so about 9 minutes for 40 checkpoints.

## 3. Pilot design

* **Task** (`induction`, new): uniform random tokens with one repeated segment per sequence.
  * The segment has a random length and starts at a random position, so matching must be by content.
  * Knobs:
    * vocabulary;
    * sequence length;
    * repeat length range;
    * the fraction of sequences that contain a repeat (0 is the no-repeat control);
    * later, a global bigram background, so that in-weights and in-context predictions can conflict.
* **Models.** Attention-only TinyGPT with d_model 64; layers {1 (control), 2}, heads per layer {2, 4}.
  * At most 8 heads in total, so that the 2^H ablation patterns stay cheap.
  * Minimum repeat length {4, 12}, as a knob.
  * The no-repeat control (p_repeat = 0).
  * 12,000 steps; 3 seeds (300–302).
  * 48 objects: `configs/sky_v2b_pilot.yaml`.
* **Variables.** One binary variable per head (kept or ablated), and the output "the model predicts the repeated
  token" (soft: the mean probability of the correct token over a fixed probe set of repeated positions).
  * The probe distribution over ablation patterns is ours to choose: uniform here, so D = U. Correlated designs
    (for example ablating heads of one layer together) can be added.
  * With at most 12 heads, all patterns are enumerated, as in Part A's pattern mode. Every θ and η value is then
    exact for the chosen probe set.
* **Light-curve metrics.**
  * Loss on repeated and on other positions (their difference is the in-context-learning score).
  * Per head: the previous-token score and the induction score. These are the R1 reference, measured from
    attention patterns, which neither view uses.
* **Pilot size.** 48 objects at 40 checkpoints each, about 6 CPU hours in total. That is 10 to 20 minutes of wall
  time on the 88-core server; the longest single run (8 heads) takes about 15 minutes.

## 4. Engineering

1. `tycho/tasks/induction.py`: data with random repeat positions, the controls, the light-curve metrics and the
   head scores.
2. Head ablation in TinyGPT: a per-head mask on the attention output (zero ablation; mean ablation as an option).
3. Pattern mode over head-ablation patterns, through the existing PatternTask interface. The interface will allow a
   task without a ground-truth conditional.
4. A pilot sky file and a pilot analysis: phase-change step, head roles, θ and η trajectories of the heads and their
   pairs.
5. For the later GPU stage: GPU workers in the scheduler.

## 5. What the pilot is for

* Check that the circuit forms reproducibly across seeds and sizes, and that the controls fail as expected.
* Measure run times.
* See whether head roles are stable enough to serve as R1.
* Choose the referees, statistics and margins of the Part B pre-registration. Everything seen is disclosed there.

## 6. Open questions (for the pre-registration)

1. **Does Part B pre-register a second θ-versus-η test?** Part A's decision makes η the detector. A second test with
   a mechanism referee (R1) and a prediction referee (R2) would give θ a fairer chance than Part A's behavioural
   referees did. The preview in §2 points against θ on R1.
2. **Zero or mean ablation.** Mean ablation is the more common choice in the circuits literature.
3. **Token-level variables under the data distribution.** These are where marginal and conditional readings differ.
   Add them in Part B, or keep head-ablation patterns only?
