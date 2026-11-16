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

7. **Pilot 1 on the server** (48 objects, DR2b-pilot; analysis: `v2_analysis/v2b_pilot.json`).
   * The controls behaved as expected:
     * training without repeats stays at chance on repeated tokens (loss 4.159);
     * 1-layer models learn partly (3.25 to 3.57), without a circuit.
   * The circuit was unreliable. Of the 12 two-layer repeat runs, it formed fully in 2 and partly in 2, all at
     minimum repeat 4; at minimum repeat 12 it never formed.
   * In both full circuits, η's top head pair (both values) was a previous-token head with an induction head; θ's
     was not.
   * An 8-head run takes about 22 minutes, mostly in the 256 ablation patterns.
8. **Making the circuit form reliably** (local, training only, seeds 300–302).
   * Variants of the single-repeat layout:

     | variant | full circuits |
     |---|---|
     | minimum repeat 2 | 1 of 6 |
     | repeats of 4 to 16 tokens | 0 of 6 |
     | learning rate 3e-3 | 0 of 3 |
     | 30,000 steps | 1 of 3 |

   * The cause appears to be weak signal: one short repeat per sequence leaves few predictable tokens, and the
     model settles into a shortcut first.
   * **The tiled layout fixes it.** A random segment of length 4 to 16 is repeated to fill the sequence, starting
     at a random phase.
     * 6 of 6 two-layer runs formed the textbook circuit: one previous-token head (0.92 to 0.98), induction heads
       (0.84 to 0.99), and duplicate-token attention ≤ 0.05.
     * The repeat loss fell to 0.004.
     * The 1-layer control formed no circuit (loss 1.7 to 1.8).
     * At learning rate 1e-3 the repeat loss halved by step 250, early, which the prediction referee cannot use.
   * **Preview of the mechanism referee on these six circuits.**
     * With 4 heads per layer, η's top pair was the composition (previous-token head with an induction head) in 3
       of 3; θ's in 1 of 3.
     * With 2 heads per layer, both layer-1 heads are induction heads, and every measure, θ and η alike, put that
       redundant pair first. η put the composition a close second.
     * So the mechanism referee must count both kinds of circuit pair, composition and redundancy, against pairs
       that involve a head outside the circuit. The top pair alone is not enough.
   * **The induction score** now counts attention to every earlier token that followed an occurrence of the current
     token, not only the most recent one. This matters in the tiled layout, where there are several such tokens.
     Pilot 1 used the old score.

9. **Pilot 2 on the server** (DR2b-pilot2). The 24 objects with 3 heads per layer failed: d_model 64 is not
   divisible by 3. That was a configuration error, now fixed and guarded by a test. In the 24 with 2 heads:
   * The tiled layout formed the circuit in 6 of 6 two-layer runs (previous-token score 0.88 to 0.91, induction
     score 0.93 to 0.96, repeat loss about 0.01).
   * The controls failed as expected: without repeats, chance (4.159); 1 layer, 1.65 to 1.89, roughly what
     in-context token counting alone achieves.
   * **Phase change.** The repeat loss stays near 3.2 and then falls below 1:
     * at learning rate 3e-4, between steps 300 and 430;
     * at 1e-3, at about step 200.

     The induction scores jump at the same time. The previous-token head becomes clean (≥ 0.5) much later: steps
     5,900 to 7,500 at 3e-4.
10. **3 heads per layer, locally** (d_model 96, seeds 310–312, both learning rates, 6 runs in pattern mode).
    * Circuit in 6 of 6: one previous-token head (0.91 to 0.95) and two or three induction heads.
    * **Mechanism preview.** At the last checkpoint, take the AUC with which a measure ranks circuit pairs
      (composition and redundancy) above pairs that involve a head outside the circuit (conditional reference,
      uniform ablation design).
      * θ: 0.63 to 0.94 (mean about 0.85).
      * η: 0.76 to 1.00 (mean about 0.94), both values.
      * η is ahead in 6 of 6.
    * **Prediction preview.** The first checkpoint where a composition pair reaches 0.001 nats:
      * η: at or one checkpoint after the loss falls below 1 (Gaussian at it; bounded one checkpoint later at
        3e-4);
      * θ: one or two checkpoints later still.

      No measure gave a lead before the loss drop at this checkpoint spacing.
    * **Expectation.** With these previews, a second θ-versus-η test on this task is expected to favour η again.
      Like everything in this section, this is disclosed in the Part B pre-registration.

## 3. Pilot 2 design

`configs/sky_v2b_pilot2.yaml`, 48 objects:
* tiled layout, segment length 4 to 16;
* d_model 64; layers {1 (control), 2}; heads per layer {2, 3}, so H ≤ 6 and 64 ablation patterns;
* learning rate {3e-4, 1e-3}, to move the phase change later;
* p_repeat {0 (control), 1};
* 12,000 steps; seeds 310–312.

About 6 minutes per run, so about 10 minutes of wall time on the server.

## 3a. Pilot 1 design (superseded)

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
