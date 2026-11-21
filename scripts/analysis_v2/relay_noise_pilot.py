"""Pilot (exploratory; local runs): what does the noise scale lr/√B act on?

``relay_period_pilot.py`` found that at rep_min 12 the circuit an unperturbed run ends with is ordered by the
learning rate over the root of the batch size (Adam): textbook circuits at 0.88e-4, relays at 1.77e-4, whatever the
learning rate and the batch size are separately. The reviewer's seventh reply (19 November, to commit 8351c3c) gives
an account of this in three steps, supported by measurements at steps 100 to 300 of five settings:

1. Early on Adam's denominators are set by the gradient noise, so a step moves a weight by about the learning rate
   whatever the batch size, with a drift of lr·√B·(mean over standard deviation of the gradient). Counted in
   sequences seen, N = B·t, the drift and the random walk of a weight are then both lr/√B times a function of N.
   (This is the square-root scaling rule for Adam of Malladi, Lyu, Panigrahi and Arora 2022, arXiv:2205.10287,
   Definition 5.2 and Theorem 5.3, without that rule's adjustment of the two decay rates and of ε. The reply does
   not cite it.)
2. The residual stream is normalised before each attention layer and before the unembedding, so only the direction
   of the embeddings matters; once their random walk exceeds the initial norm, the direction turns at a rate that
   no longer depends on the learning rate. Progress is therefore counted in sequences: a circuit forms after about
   the same N whatever the learning rate and the batch size.
3. The query and key weights are not normalised away. At equal N they have moved in proportion to lr/√B, so a
   larger lr/√B sharpens layer 0's attention on the band earlier in N, before layer 1 has begun to match content,
   and the relay is locked in.

Three checks.

a. Equal N, no new runs: the attention statistics and the repeat loss of the eight cells of
   ``relay_period_pilot.py`` at equal numbers of sequences. Steps 1 and 2 predict that cells with the same lr/√B
   coincide.
b. Displacement: the first 38,400 sequences of the eight cells again (these repeat the cached runs exactly),
   recording for seven groups of weights the norm, the distance from the initial weights and the cosine to them,
   every 3,200 sequences. Step 1 predicts a distance at equal N proportional to lr/√B: fitted as
   lr^a · B^b across the eight cells, a = 1 and b = −1/2 (an Adam without noise would give b = −1; a distance set
   by the number of steps alone, b = −1/2 with a = 1 is the same as a random walk of lr per step).
c. Intervention on the baseline (learning rate 1e-3, batch 32: 0 of 6 textbook circuits; at learning rate 5e-4:
   6 of 6): the learning rate is halved for layer 0's query and key weights only ("qk0_half"), and for every
   weight except those ("rest_half"). If step 3 is what decides (the noise acts through layer 0's hold on the
   band), qk0_half gives textbook circuits and rest_half relays. If the noise acts instead by keeping layer 1's
   content match from forming, it is the other way round. Both, or neither, flipping would mean that no single
   group carries the effect.

Recorded before the runs of (b) and (c) were started: the equal-N comparison (a) had been made (cells with batch 8
and 32 agree at equal N, cells with batch 128 lag by a few thousand sequences); neither outcome of (c) was known
or expected with any confidence.

Second set of (c), recorded after the first two arms and before its runs. The first two arms came out as step 3
has it: qk0_half 6 of 6 textbook circuits, rest_half 0 of 6. With the two reference cells that is a two-by-two in
which the circuit follows the learning rate of layer 0's query and key weights and nothing else. Stated as a
hypothesis H: the circuit is set by lr/√B of layer 0's query and key weights alone, with the threshold where the
whole network has it (textbook circuits at 0.88e-4 and 1.25e-4, relays at 1.77e-4 and above). Five arms test it
away from the two-by-two:

* qk1_double: the baseline with the learning rate of layer 1's query and key weights doubled. Layer 0's are at
  1.77e-4, so H predicts relays (at most 1 of 6). If what counts is instead how fast layer 0's attention learns
  relative to layer 1's (the ratio is one half here, as in qk0_half), it gives textbook circuits (at least 5).
* qk0_0.71: the baseline with layer 0's query and key weights at 1/√2 of the learning rate (1.25e-4, where the
  whole network gave 5 of 6). H predicts at least 4 of 6.
* lr5e-4_b8_qk0_half: learning rate 5e-4, batch 8 (1 of 6 without the change), layer 0's query and key weights at
  half (0.88e-4). H predicts at least 5 of 6.
* lr2e-3_qk0_quarter: learning rate 2e-3, batch 32 (0 of 6), layer 0's query and key weights at a quarter
  (0.88e-4) while everything else stays at 3.54e-4. H predicts at least 5 of 6.
* lr2e-3_qk0_half: the same cell with those weights at half (1.77e-4). H predicts at most 1 of 6.

H fails if qk1_double gives 3 or more textbook circuits, if either arm at 0.88e-4 gives 3 or fewer, or if
lr2e-3_qk0_half gives 3 or more. Expectation recorded with it: no confident view on qk1_double and on
lr2e-3_qk0_quarter (the rest of that network is at four times the noise scale of the weights that are slowed).

Third set of (c): the reviewer's eighth reply (20 November, to commit cb49962; it states that it ran none of
these). Its account of why the textbook circuit is lost when the weights other than the two layers' query and key
weights are at 3.54e-4: layer 0's output is a product of two weight matrices that are not normalised away (its
value rows and its output projection), so it grows as the square of the noise scale while the embeddings grow
linearly; at learning rate 2e-3 it makes up 90 per cent and more of what layer 1 reads, and layer 1 can no longer
see positions and tokens. Its predictions, in the cell at learning rate 2e-3 and batch 32 with layer 0's query and
key weights at a quarter (1 of 6 textbook circuits), with one more group of weights at half the learning rate:
the output side of both layers ("ov0" and "ov1"): at least 5 of 6; the positional embedding: at most 2; the token
embedding: at most 1; the unembedding: at most 2. In a note: of the two output sides it is layer 0's alone that
decides (at least 5), not layer 1's. It fails if the output side gives 3 or fewer, or one of the other three 4 or
more.
Recorded here before the runs. (i) One reference arm is added: every weight except layer 1's query and key weights
at half (layer 0's query and key weights stay at a quarter). That is the baseline with the one slowed and the
other sped up; expected: at least 5 of 6. Without it, four arms that each change one group say nothing if none of
them flips the outcome. (ii) The size of layer 0's output also rises when its attention sharpens (a sharper
attention averages fewer value vectors), and in the reviewer's own table the runs with the largest output are the
ones in which layer 0 had already taken the band. If the large output is a consequence of that and not the cause,
halving the output side does not bring the textbook circuit back. No confident expectation either way.

Fourth set of (c), added after the runs of the third set and before its own (20 November). The third set came out
mostly as the reviewer predicted: with the learning rate halved for the output side of both layers, 6 of 6
textbook circuits; for layer 0's output side alone, 6 of 6; for layer 1's alone, 0 of 6; for the token embedding,
0 of 6; for the unembedding, 0 of 6; for the positional embedding, 3 of 6, where at most 2 were predicted. That
fixes the lever (the step size of layer 0's output weights) but not the explanation. The reviewer's is that layer
0's output swamps what layer 1 reads, and that this is also why the whole network's threshold lies between 0.88e-4
and 1.77e-4. Another is that whatever slows layer 0's positional route helps, and its output weights are part of
that route. Three arms change the size of layer 0's output directly, by a fixed factor on it in the forward pass,
with every learning rate left alone (Adam does not respond to the scale of a gradient, so the weights move about
as before and the output is smaller or larger by about the factor):

* base_y0_quarter: the baseline (1.77e-4, 0 of 6) with layer 0's output times 0.25. If the size of that output is
  what separates the two sides of the threshold: textbook circuits, at least 5 of 6.
* lr5e-4_y0_times4: learning rate 5e-4 (0.88e-4, 6 of 6) with the output times 4: relays, at most 1 of 6.
* lr2e-3_qk0_quarter_y0_quarter: the cell of the third set (1 of 6) with the output times 0.25: textbook circuits,
  at least 5 of 6, as with the output weights at half the learning rate.

These predictions are Claude's reading of the reviewer's account, not the reviewer's own. Expectation recorded
with them: the third more likely than not (the size of layer 0's output at step 300 sorts the 48 runs of the third
set almost perfectly); the first two open.

Fifth set of (c): the reviewer's ninth reply (20 November, to commit b25f5ec; it states that it ran none of these).
The fourth set came out as its account has it (baseline with layer 0's output times 0.25: 5 of 6 textbook
circuits; learning rate 5e-4 with the output times 4: 0 of 6; the cell of the third set with the output times
0.25: 6 of 6). The reply makes two things of this.

First, how the factor c on layer 0's output trades against the noise scale s = lr/√B. Claude's question had
supposed c to be worth √c in s, because that output grows as the square of s. The reply: what layer 1 reads is the
ratio of that output to the embeddings, which themselves grow with s, so the ratio goes as c·s and c is worth c,
not √c. Four cells with no learning rate changed tell the two apart: learning rate 2e-3 (3.54e-4) with the output
times 0.25: at least 4 of 6 (√c would make it the baseline, relays); the same with 1/16: at least 4; the baseline
with 0.5: at least 5; learning rate 5e-4 with the output times 2: at most 2 (√c would make it 1.25e-4, textbook).
It fails if the first gives 2 or fewer, the fourth 4 or more, the third 3 or fewer, or the second 2 or fewer.

Second, one quantity computed before training,

    S = (s_qk0 / s_qk1) · 2(s0² + s_pos²) / (2 s0² + s_pos² + s_tok²) · (1 + R²),
    R = c · s_ov0² / (s* · √((s_pos² + s_tok²) / 2)),   s* = 1.0e-4,   s0 = 1.606e-4,

with s_g the noise scale of a group of weights, and a threshold at about 3.15 (textbook circuits below). On the 25
conditions it had, every condition with 3 or more textbook circuits of 6 lies below and every one with 1 or fewer
above. Three settings of its own choosing, all on the baseline: layer 0's query and key weights at half the
learning rate with the output times 2 (S = 6.75; alone, the slower weights gave 6 of 6): at most 1 of 6; every
weight but those at half with the output times 0.25 (S = 2.10; without the factor 0 of 6): at least 4; the token
embedding at twice the learning rate and nothing else (S = 1.24; baseline 0 of 6): at least 5. It fails if the
first gives 3 or more, the second 2 or fewer, or the third 3 or fewer.

Recorded here before the runs. (i) The form of S, its two constants and its threshold were chosen with the 25
outcomes in view, so its fit to them says little; the seven new conditions are the test. (ii) On the trade between
c and s the reply's argument is the better one, and Claude expects its four cells to come out its way more often
than not, with no confident view on the cell at 1/16 (so little of layer 0's output may leave the textbook circuit
without its key). (iii) The third of its own settings is the most surprising: a higher learning rate for one group
of weights is to turn relays into textbook circuits. In the third set the token embedding at half the learning
rate made matters worse, which points the same way; expected: an improvement, whether as far as 5 of 6 is open.
(iv) One arm is added for the gap the reply names itself (S has no term for layer 1's output weights): learning
rate 5e-4 (6 of 6, S = 1.78) with those weights at a quarter of the learning rate. Expected: at most 3 of 6,
against the formula. (v) Added while the runs were in progress and before any outcome was looked at, from the
settings and the committed counts alone: on the 25 conditions S is at the ceiling of any rule of the settings (139
of 150 runs, every condition with a majority on its side of the threshold), and so are three other forms: S
without its position factor; S with c entering as √c; and S with layer 0's output taken by itself instead of
relative to the embeddings. The constant s* can lie anywhere from 0.71e-4 to 1.41e-4 if the threshold moves with
it. The 25 conditions therefore do not tell these forms apart, and the new arms divide as follows. Relative to the
embeddings against by itself (c against √c): the first and fourth cells of the first set and the token-embedding
arm. The position factor: no arm (with and without it the calls for all eight arms are the same). The second
setting of the second set is on the textbook side only for s* below about 1.3e-4. For the second and third cells
of the first set and for Claude's arm every one of the forms says textbook circuits.

Outcome of the fifth set (textbook circuits of 6 at 12,000 steps). First set: 4, 2, 6 and 0. By the reply's own
criteria it fails, on the cell at 1/16 alone; no run of its first three cells is a relay, and those that are not
textbook circuits (2 and 4 runs) have layer 1 on the source with a one-token key. The two cells that tell c from
√c came out the reply's way, against Claude's supposition. Second set: 0, 3 and 1. It fails, on the
token-embedding arm (5 relays). Claude's arm: 0 of 6. On the eight arms S has 28 of 48 runs on its side of the
threshold (40 possible), against 139 of 150 on the 25 conditions it was built on. Of the remarks above, (ii) and
(iv) held and (iii) did not (1 of 6 against the baseline's 0 is no improvement).

Sixth set of (c): the reviewer's tenth reply (21 November, to commit 9566534; it states that it ran none of
these). It withdraws S. Two of its three tasks are here (the third, thresholds at rep_min 4 and 24, is in
``relay_period_pilot.py``).

(1) Its account of the arm with layer 1's output weights slowed: the outputs of the two layers share the early
loss, so when one is slow the other grows faster. From two existing pairs of arms that differ by a factor 2 in the
learning rate of layer 1's output weights (R1 at step 300 lower by 0.67 and 0.72), it predicts for the baseline
with those weights at twice the learning rate: R1 at step 300 (median of six runs) 0.96, between 0.80 and 1.15
(baseline 1.375), and at least 4 of 6 textbook circuits (baseline 0). It fails on R1 at 1.25 or more or below 0.70,
and on the count at 2 or fewer. R1 is the root-mean-square size of layer 0's output over that of the summed
embeddings at the predictable positions of the probe set.

(2) Why the second key token does not come when layer 0's output is scaled down. Its account: before layer 1
settles on the source the scaled output is too small for one head to carry the one-token key, so every head of
layer 0 is drawn to the previous token; once all of them carry that key together, moving any of them to the token
before it costs loss at first, and the state is stable. By the end of the first stage Adam has made up for the
factor, so the factor itself is no longer what is in the way. (2A) Learning rate 2e-3 with the output at 1/16 and
8 heads: at least 5 of 6 textbook circuits (3 of 3 with three seeds) and no relay; it fails at 3 of 6 or fewer.
(2B) Taking the factor away once the first stage is over, in the eight 2-head runs that stalled: at most 2 of 8
become textbook circuits (centre 0 to 1), whether the factor is folded into the weights (the function unchanged at
the switch) or simply dropped (the output jumps by 4 or 16); if dropped, the three runs at 1/16 show a spike in
the loss and at least one of them leaves the source; if folded, all eight stay on the source with a one-token key.
It fails at 4 of 8 or more.

Recorded here before the runs. The switch is made at step 1,000 in every run (in the twelve 2-head runs of the four
arms layer 1 settles on the source between steps 350 and 700, and none has its second key token by step 1,000);
with the fold, Adam's averages for the folded weights are rescaled, so that training goes on as it would without a
factor (see ``relay_band_pilot.train``). A trial of the two switches on the first 406 steps of one run (switch at
step 403, not a run of the set) showed what they do at once: with the fold the loss is unchanged at the switch and
then rises for some steps (1.25 to 1.75 within two), because the folded weights are small and Adam's steps are
large against them; with the factor dropped the loss is back at chance in one step (4.27). Added by Claude: the
four 2-head runs of the same arms that did become textbook circuits, under both switches, to see what the switch
itself does. The 8-head arm uses seeds 300 to 305.
Claude's expectations. (1) R1 below the baseline's, and more likely near the top of its interval or just above it
(1.0 to 1.25) than at its centre: at learning rate 5e-4 a factor 4 on these weights moves R1 at step 300 by only
0.9, and the two pairs it uses are at 2e-3. The count: 4 of 6 as a single guess, 2 to 6 unsurprising. (2A) 5 of 6.
(2B) With the fold 0 or 1 of 8, as it says; with the factor dropped no confident view (guess 2 of 8, and at least
one of the three runs at 1/16 ends off the source). The four control runs: textbook circuits with the fold.

Outcome of the sixth set. (1) R1 at step 300 is 1.01 and 6 of 6 runs are textbook circuits: both as the reply had
them (Claude's count was too low). (2A) 6 of 6. (2B) Of the eight runs that had stalled, 6 become textbook circuits
with the fold and 8 with the factor dropped; by the reply's criterion both fail, and so does Claude's expectation
for the fold. With the factor dropped the three runs at 1/16 lose everything for 100 to 200 steps and all three end
as textbook circuits on the source. The four control runs are textbook circuits under both switches.

d. A summary made after all of the above (post hoc; no prediction on record): which layer's band attention (mean
   over its heads) passes a level first, against the circuit at the end. The level is chosen on the 48 runs of
   the eight cells and then applied to the intervention arms, to the unperturbed runs at d_model 32 and 128 and to
   the cells at rep_min 16 of ``relay_period_pilot.py``.

e. The two stages on the way to a textbook circuit (post hoc, after the fifth set; no prediction on record). A run
   that ends as a textbook circuit first has layer 1 settle on the source with a one-token key (the loss on
   predictable tokens falls to between 0.5 and 0.8); the second key token comes later (the loss falls below 0.4
   and stays there). Per condition: the step of each stage, their ratio, and the runs that are on the source at
   the end without being textbook circuits, with the offset each head of layer 0 attends to most. Ten such runs,
   from the arms with layer 0's output at a quarter or less, are trained four times as long (48,000 steps) to see
   whether the second stage is late or absent.

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/relay_noise_pilot.py
--jobs 4`` (``--skip-runs`` analyses the cache). Needs the cached runs and the output of
``relay_period_pilot.py``. Runs are cached in ``work/relay_band``. Writes
``docs/results/v2_analysis/relay_noise_pilot.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("relay_period_pilot", Path(__file__).with_name("relay_period_pilot.py"))
pp = importlib.util.module_from_spec(_spec)
sys.modules["relay_period_pilot"] = pp
_spec.loader.exec_module(pp)
wp, rb, fp = pp.wp, pp.rb, pp.fp

#: the eight cells of the noise table of relay_period_pilot.py (rep_min 12, d_model 64), by lr/√B
CELLS = ("lr2.5e-4_b8", "lr5e-4", "b128", "b64", "lr5e-4_b8", "base", "lr2e-3_b128", "lr2e-3")
LOW, HIGH = ("lr2.5e-4_b8", "lr5e-4", "b128"), ("lr5e-4_b8", "base", "lr2e-3_b128")   # lr/√B 0.88e-4 and 1.77e-4
N_STEP, N_LAST = 3_200, 38_400                          # sequences between measurements, and in all
GROUPS = ("pos", "tok", "qk0", "ov0", "qk1", "ov1", "head")
KEYS = ("band_l0", "band_l1", "source_l1", "offsets_1_2_l0", "loss_rep")
LEVELS = (0.55, 0.65, 0.75, 0.85)                       # band attention levels of the summary (d)
#: intervention arms: (cell, learning-rate factors by group of weights (see relay_band_pilot.weight_groups),
#: the prediction on record: at least / at most this many textbook circuits, or None[, {layer: fixed factor on
#: that layer's attention output}])
ARMS = {"qk0_half": ("base", {"qk0": 0.5}, None), "rest_half": ("base", {"all": 0.5, "qk0": 1.0}, None),
        "qk1_double": ("base", {"qk1": 2.0}, ("at most", 1)),
        "qk0_0.71": ("base", {"qk0": 1 / math.sqrt(2)}, ("at least", 4)),
        "lr5e-4_b8_qk0_half": ("lr5e-4_b8", {"qk0": 0.5}, ("at least", 5)),
        "lr2e-3_qk0_quarter": ("lr2e-3", {"qk0": 0.25}, ("at least", 5)),
        "lr2e-3_qk0_half": ("lr2e-3", {"qk0": 0.5}, ("at most", 1)),
        "lr2e-3_qk0_quarter_ov_half": ("lr2e-3", {"qk0": 0.25, "ov0": 0.5, "ov1": 0.5}, ("at least", 5)),
        "lr2e-3_qk0_quarter_pos_half": ("lr2e-3", {"qk0": 0.25, "pos": 0.5}, ("at most", 2)),
        "lr2e-3_qk0_quarter_tok_half": ("lr2e-3", {"qk0": 0.25, "tok": 0.5}, ("at most", 1)),
        "lr2e-3_qk0_quarter_head_half": ("lr2e-3", {"qk0": 0.25, "head": 0.5}, ("at most", 2)),
        "lr2e-3_qk0_quarter_rest_half": ("lr2e-3", {"all": 0.5, "qk0": 0.25, "qk1": 1.0}, ("at least", 5)),
        "lr2e-3_qk0_quarter_ov0_half": ("lr2e-3", {"qk0": 0.25, "ov0": 0.5}, ("at least", 5)),
        "lr2e-3_qk0_quarter_ov1_half": ("lr2e-3", {"qk0": 0.25, "ov1": 0.5}, None),
        "base_y0_quarter": ("base", {}, ("at least", 5), {0: 0.25}),
        "lr5e-4_y0_times4": ("lr5e-4", {}, ("at most", 1), {0: 4.0}),
        "lr2e-3_qk0_quarter_y0_quarter": ("lr2e-3", {"qk0": 0.25}, ("at least", 5), {0: 0.25}),
        "lr2e-3_y0_quarter": ("lr2e-3", {}, ("at least", 4), {0: 0.25}),
        "lr5e-4_y0_times2": ("lr5e-4", {}, ("at most", 2), {0: 2.0}),
        "tok_double": ("base", {"tok": 2.0}, ("at least", 5)),
        "qk0_half_y0_times2": ("base", {"qk0": 0.5}, ("at most", 1), {0: 2.0}),
        "rest_half_y0_quarter": ("base", {"all": 0.5, "qk0": 1.0}, ("at least", 4), {0: 0.25}),
        "base_y0_half": ("base", {}, ("at least", 5), {0: 0.5}),
        "lr2e-3_y0_sixteenth": ("lr2e-3", {}, ("at least", 4), {0: 0.0625}),
        "lr5e-4_ov1_quarter": ("lr5e-4", {"ov1": 0.25}, ("at most", 3)),
        "ov1_double": ("base", {"ov1": 2.0}, ("at least", 4))}
#: the arms with a fixed factor on layer 0's output in the fourth set; the arms of the eighth round (third and
#: fourth sets); the arms of the ninth round (fifth set); whose prediction the third entry of an arm is
OUTPUT = ("base_y0_quarter", "lr5e-4_y0_times4", "lr2e-3_qk0_quarter_y0_quarter")
NINTH = ("lr2e-3_y0_quarter", "lr5e-4_y0_times2", "tok_double", "qk0_half_y0_times2", "rest_half_y0_quarter",
         "base_y0_half", "lr2e-3_y0_sixteenth", "lr5e-4_ov1_quarter")
TENTH = ("ov1_double",)
EIGHTH = tuple(a for a in ARMS if a.startswith("lr2e-3_qk0_quarter_")) + tuple(a for a in OUTPUT if a not in (
    "lr2e-3_qk0_quarter_y0_quarter",))
PREDICTED_BY = {**{a: "hypothesis H (Claude, seventh round)" for a in ARMS
                   if a not in EIGHTH + NINTH + TENTH and ARMS[a][2]},
                **{a: "the reviewer's eighth reply" for a in EIGHTH if ARMS[a][2]},
                "lr2e-3_qk0_quarter_rest_half": "Claude (reference arm, eighth round)",
                **{a: "the reviewer's account of layer 0's output, as read by Claude" for a in OUTPUT},
                **{a: "the reviewer's ninth reply" for a in NINTH}, "lr5e-4_ov1_quarter": "Claude (ninth round)",
                **{a: "the reviewer's tenth reply" for a in TENTH}}
#: runs that ended with layer 1 on the source and a one-token key (arms with layer 0's output at a quarter or
#: less), trained four times as long; chosen after the outcomes of the ninth round
LONG_STEPS = 48_000
LONG = {"lr2e-3_y0_sixteenth": ("h2_s300", "h2_s301", "h2_s302", "h4_s300"),
        "lr2e-3_y0_quarter": ("h2_s300", "h2_s301"), "rest_half_y0_quarter": ("h2_s300", "h2_s301", "h4_s302"),
        "base_y0_quarter": ("h2_s301",)}
#: the second stage of the textbook circuit: the loss on predictable tokens stays below this (with a one-token
#: key it sits at 0.5 to 0.8)
SECOND_KEY_LOSS = 0.4
#: tenth round, (1): R1 (layer 0's output over the embeddings) in the first steps of these conditions; the pairs
#: differ only in the learning rate of layer 1's output weights (slow, fast, factor)
RATIO_STEPS = (0, 100, 200, 300, 400)
RATIO_PAIRS = (("lr2e-3_qk0_quarter_ov_half", "lr2e-3_qk0_quarter_ov0_half", 2.0),
               ("lr2e-3_qk0_quarter_ov1_half", "lr2e-3_qk0_quarter", 2.0), ("lr5e-4_ov1_quarter", "lr5e-4", 4.0),
               ("base", "ov1_double", 2.0))
#: tenth round, (2A): the arm with layer 0's output at 1/16, with 8 heads
EIGHT_ARM, EIGHT_HEADS, EIGHT_SEEDS = "lr2e-3_y0_sixteenth", 8, (300, 301, 302, 303, 304, 305)
#: tenth round, (2B): the factor on layer 0's output taken away at this step, in the 2-head runs of the arms of
#: ``LONG``; "fold" keeps the function at the switch, "drop" lets the output jump
SWITCH_STEP, SWITCH_HOW = 1_000, ("fold", "drop")
#: the constants of the reviewer's combined quantity S (ninth reply): the noise scale at which layer 0's output
#: equals the embeddings in what layer 1 reads, the noise scale that stands for the initial embeddings, and the
#: threshold of S
S_STAR, S_ZERO, S_THRESHOLD = 1.0e-4, 1.606e-4, 3.15


def cell_of(setting: str) -> tuple[float, int]:
    change = pp.SETTINGS[setting]
    return change.get("lr", 1e-3), change.get("batch_size", 32)


def combined_quantity(lr: float, batch: int, factors: dict, output: dict | None, s_zero: float = S_ZERO) -> dict:
    """The reviewer's S (ninth reply) for a condition: learning rate, batch size, learning-rate factors by group of
    weights, and the fixed factor on layer 0's output. ``s_zero=0`` gives its asymptotic version."""
    def scale(g: str) -> float:
        return lr * factors.get(g, factors.get("all", 1.0)) / math.sqrt(batch)

    c = float((output or {}).get(0, 1.0))
    pos, tok = scale("pos"), scale("tok")
    ratio = c * scale("ov0") ** 2 / (S_STAR * math.sqrt((pos ** 2 + tok ** 2) / 2))
    position = 2 * (s_zero ** 2 + pos ** 2) / (2 * s_zero ** 2 + pos ** 2 + tok ** 2)
    speed = scale("qk0") / scale("qk1")
    return {"qk0_over_qk1": speed, "ratio_of_layer_0_output_to_embeddings": ratio, "position_share_factor": position,
            "S": speed * position * (1 + ratio ** 2)}


def displacement_stats():
    """A measurement function for ``rb.train``: the band statistics and, by group of weights, the norm, the distance
    from the first weights it sees (step 0) and the cosine to them."""
    import torch

    first = {}

    def fn(model, task) -> dict:
        out = rb.band_stats(model, task)
        for g, parts in rb.weight_groups(model).items():
            w = torch.cat([p.detach()[rows].reshape(-1) for p, rows in parts])
            w0 = first.setdefault(g, w.clone())
            out[f"norm_{g}"] = float(w.norm())
            out[f"moved_{g}"] = float((w - w0).norm())
            out[f"cos_{g}"] = float((w @ w0) / (w.norm() * w0.norm()))
        return out

    return fn


def clock_jobs() -> list[dict]:
    jobs = []
    for cell in CELLS:
        _, batch = cell_of(cell)
        for j in pp.arm_jobs(f"{cell}_natural"):
            jobs.append({**j, "cell": cell, "steps": N_LAST // batch, "every": N_STEP // batch})
    return jobs


def run_clock(args: tuple[dict, str]) -> str:
    job, work = args
    out = Path(work) / f"noise_clock_{job['cell']}_{job['base']}.json"
    if out.exists():
        return f"cached {out.name}"
    at = tuple(range(0, job["steps"] + 1, job["every"]))
    _, _, stats, _ = rb.train(job, job["steps"], at, block_from=0, block_until=0, layer=0, forbid="band",
                              stats_fn=displacement_stats())
    out.write_text(json.dumps({"cell": job["cell"], "run": job["base"], "step": list(at),
                               **{k: [stats[s][k] for s in at] for k in stats[0]}}) + "\n", encoding="utf-8")
    return f"done clock {job['cell']} {job['base']}"


def arm_jobs(arm: str) -> list[dict]:
    cell, factors = ARMS[arm][:2]
    output = ARMS[arm][3] if len(ARMS[arm]) > 3 else None
    return [{**j, "name": f"noise_{arm}_{j['base']}", "arm": f"noise_{arm}", "steps": pp.FULL_STEPS,
             "train": {**j["train"], "steps": pp.FULL_STEPS}, "lr_scale": factors or None, "out_scale": output}
            for j in pp.arm_jobs(f"{cell}_natural")]


def long_jobs() -> list[dict]:
    return [{**j, "name": f"noise_{arm}_long{LONG_STEPS}_{j['base']}", "arm": f"noise_{arm}_long{LONG_STEPS}",
             "steps": LONG_STEPS, "train": {**j["train"], "steps": LONG_STEPS}}
            for arm, picks in LONG.items() for j in arm_jobs(arm) if j["base"].endswith(picks)]


def eight_jobs() -> list[dict]:
    t = next(j for j in arm_jobs(EIGHT_ARM) if j["cfg"]["n_head"] == 4)
    return [{**t, "name": f"noise_{EIGHT_ARM}_single_rm12_h{EIGHT_HEADS}_s{seed}",
             "base": f"single_rm12_h{EIGHT_HEADS}_s{seed}", "arm": f"noise_{EIGHT_ARM}_h{EIGHT_HEADS}", "seed": seed,
             "cfg": {**t["cfg"], "n_head": EIGHT_HEADS, "data_seed": seed}} for seed in EIGHT_SEEDS]


def switch_jobs() -> list[dict]:
    return [{**j, "name": f"noise_{arm}_restore_{how}{SWITCH_STEP}_{j['base']}",
             "arm": f"noise_{arm}_restore_{how}{SWITCH_STEP}", "switched": arm, "how": how,
             "stalled_without_the_switch": j["base"].endswith(tuple(LONG[arm])),
             "out_switch": {"step": SWITCH_STEP, "fold": how == "fold"}}
            for how in SWITCH_HOW for arm in LONG for j in arm_jobs(arm) if j["cfg"]["n_head"] == 2]


def ratio_jobs() -> dict:
    """The conditions of ``RATIO_PAIRS`` (cells of ``relay_period_pilot.py`` or arms here), as jobs."""
    names = dict.fromkeys(c for pair in RATIO_PAIRS for c in pair[:2])
    return {c: arm_jobs(c) if c in ARMS else pp.arm_jobs(f"{c}_natural") for c in names}


def ratio_stats(model, task) -> dict:
    """The band statistics and, at the predictable positions of the probe set, the root-mean-square size of the
    summed embeddings, of layer 0's output and of layer 1's output (with any factor of ``out_scale`` on them)."""
    import torch

    out = rb.band_stats(model, task)
    pr = task._probe
    x, rep = pr["x"], pr["rep"]
    with torch.no_grad():
        mask = model.causal_mask[:x.shape[1], :x.shape[1]]
        h0 = model.tok(x) + model.pos(torch.arange(x.shape[1]))
        y0 = model.blocks[0].attn(model.blocks[0].ln1(h0), mask)
        y1 = model.blocks[1].attn(model.blocks[1].ln1(h0 + y0), mask)
    out.update({f"rms_{k}": float(torch.sqrt((v[rep] ** 2).mean())) for k, v in (("emb", h0), ("y0", y0), ("y1", y1))})
    out["r1"] = out["rms_y0"] / out["rms_emb"]
    return out


def run_ratio(args: tuple[str, dict, str]) -> str:
    condition, job, work = args
    out = Path(work) / f"noise_ratio_{condition}_{job['base']}.json"
    if out.exists():
        return f"cached {out.name}"
    _, _, stats, _ = rb.train(job, RATIO_STEPS[-1], RATIO_STEPS, block_from=0, block_until=0, layer=0, forbid="band",
                              stats_fn=ratio_stats, lr_scale=job.get("lr_scale"), out_scale=job.get("out_scale"))
    out.write_text(json.dumps({"condition": condition, "run": job["base"], "step": list(RATIO_STEPS),
                               **{k: [stats[s][k] for s in RATIO_STEPS] for k in stats[0]}}) + "\n", encoding="utf-8")
    return f"done ratio {condition} {job['base']}"


def stage_steps(job: dict, work: Path) -> dict:
    """The two stages on the way to a textbook circuit, from a run's curves: the first step with most of layer 1's
    attention on the source, and the first step from which the loss on predictable tokens stays below
    ``SECOND_KEY_LOSS``."""
    step, st = pp.curves(job, work)
    on = np.flatnonzero(st["source_l1"] > 0.5)
    low = st["loss_rep"] < SECOND_KEY_LOSS
    stay = np.flatnonzero(low & (np.cumsum(~low[::-1])[::-1] == 0))
    return {"on_the_source_at_the_end": bool(st["source_l1"][-1] > 0.5), "steps": float(step[-1]),
            "source_step": float(step[on[0]]) if len(on) else None,
            "second_key_step": float(step[stay[0]]) if len(stay) else None}


def on_the_source(row: dict) -> bool:
    """Layer 1 ends with most of its attention on the source itself (every textbook circuit does; so do the circuits
    with a one-token key that an offset-1 head in layer 0 supplies)."""
    return bool(row["formed"] and row["relay_delta"] == 0 and row["source_l1_at_the_end"] > 0.5)


def layer_1_first(job: dict, work: Path, level: float) -> bool | None:
    """Does layer 1's band attention pass the level before layer 0's? None if neither does."""
    step, st = pp.curves(job, work)
    t0 = pp.first_crossing(step, st["band_l0"], level, True)
    t1 = pp.first_crossing(step, st["band_l1"], level, True)
    if t0 is None and t1 is None:
        return None
    return t0 is None or (t1 is not None and t1 < t0)


def power_fit(cells: dict, value: dict) -> dict:
    """value ~ lr^a · B^b across the cells (least squares on the logarithms)."""
    names = [c for c in cells if value.get(c) and value[c] > 0]
    x = np.array([[1.0, math.log(cells[c][0]), math.log(cells[c][1])] for c in names])
    y = np.log([value[c] for c in names])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"lr_exponent": float(coef[1]), "batch_exponent": float(coef[2]),
            "rms_log_residual": float(np.sqrt(np.mean((x @ coef - y) ** 2))), "cells": len(names)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_band")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    ap.add_argument("--only", choices=("clock", "arms"), help="run only this part")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_noise_pilot.json")
    args = ap.parse_args(argv)
    clock = clock_jobs()
    arms = {arm: arm_jobs(arm) for arm in ARMS}
    longer, eight, switched, ratio_runs = long_jobs(), eight_jobs(), switch_jobs(), ratio_jobs()
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            if args.only != "arms":
                for msg in pool.imap(run_clock, [(j, str(args.work)) for j in clock]):
                    print(msg, flush=True)
                for msg in pool.imap(run_ratio, [(c, j, str(args.work)) for c, js in ratio_runs.items()
                                                 for j in js]):
                    print(msg, flush=True)
            if args.only != "clock":
                for msg in pool.imap(pp.run_arm, [(j, str(args.work)) for js in arms.values() for j in js] + [
                        (j, str(args.work)) for j in longer + eight + switched]):
                    print(msg, flush=True)
    cells = {c: cell_of(c) for c in CELLS}
    grid = list(range(0, N_LAST + 1, N_STEP))

    # (a) the cached unperturbed runs at equal numbers of sequences
    cached = {}
    for cell, (lr, batch) in cells.items():
        js = [j for j in pp.arm_jobs(f"{cell}_natural") if (args.work / f"period_{j['name']}.npz").exists()]
        cs = [pp.curves(j, args.work) for j in js]
        cached[cell] = {k: [float(np.median([np.interp(n / batch, c[0], c[1][k]) for c in cs])) for n in grid]
                        for k in KEYS} if cs else None
    equal_n = {"sequences": grid, "cells": {
        c: {"learning_rate": cells[c][0], "batch": cells[c][1], "lr_over_root_batch": cells[c][0] / math.sqrt(
            cells[c][1]), **cached[c]} for c in CELLS if cached[c]}}
    spread = {}
    if all(cached[c] for c in LOW + HIGH):
        for k in KEYS:
            lo = np.array([cached[c][k] for c in LOW])
            hi = np.array([cached[c][k] for c in HIGH])
            spread[k] = {"within_low": (lo.max(0) - lo.min(0)).tolist(),
                         "within_high": (hi.max(0) - hi.min(0)).tolist(), "between": (hi.mean(0) - lo.mean(0)).tolist(),
                         "within_low_batch_8_and_32": np.abs(lo[0] - lo[1]).tolist(),
                         "within_high_batch_8_and_32": np.abs(hi[0] - hi[1]).tolist()}
    equal_n["spread"] = spread

    # (b) displacement of the weights at equal numbers of sequences
    runs = {}
    for j in clock:
        f = args.work / f"noise_clock_{j['cell']}_{j['base']}.json"
        if f.exists():
            runs.setdefault(j["cell"], []).append(json.loads(f.read_text()))
    measured = {c: {k: np.median([r[k] for r in rs], axis=0).tolist() for k in rs[0]
                    if k not in ("cell", "run", "step")} for c, rs in runs.items() if len(rs) == 6}
    repeat = None                                        # the new short runs against the cached ones, at the steps
    if measured and all(cached.get(c) for c in measured):   # at which the cached runs were measured
        def on_grid(c: str, n: int) -> bool:
            t = n / cells[c][1]
            return t % 250 == 0 or (t <= pp.SHORT_STEPS and t % 50 == 0)

        repeat = float(max(abs(a - b) for c in measured for k in KEYS
                           for a, b, n in zip(measured[c][k], cached[c][k], grid) if on_grid(c, n)))
    displacement = {"sequences": grid, "cells": measured, "largest_difference_from_the_cached_runs": repeat,
                    "fit_lr_a_times_batch_b": {}, "moved_times_root_batch_over_lr": {},
                    "moved_ratio_when_the_learning_rate_doubles": {}, "one_minus_cosine_fit": {}}
    #: pairs of cells with the same batch size and twice the learning rate
    doubled = (("lr2.5e-4_b8", "lr5e-4_b8"), ("lr5e-4", "base"), ("b128", "lr2e-3_b128"), ("base", "lr2e-3"))
    if len(measured) == len(CELLS):
        # how far the batch-128 cell of each level is behind the other two, in sequences: the shift that brings
        # its layer-0 band attention onto the mean of the other two, over the first 19,200 sequences
        lag = {}
        early = [n for n in grid if n <= 19_200]
        for name, group in (("low", LOW), ("high", HIGH)):
            ref = np.mean([measured[c]["band_l0"] for c in group[:2]], axis=0)[:len(early)]
            mine = np.array(measured[group[2]]["band_l0"])
            shifts = np.arange(0, 9_601, 200)
            err = [float(np.mean((np.interp(np.array(early) + s, grid, mine) - ref) ** 2)) for s in shifts]
            best = int(shifts[int(np.argmin(err))])
            lag[name] = {"cell": group[2], "sequences": best, "steps_of_batch_128": best / 128}
        equal_n["batch_128_lag"] = lag
    if len(measured) == len(CELLS):
        for g in GROUPS:
            fits, coll, turn = {}, {}, {}
            for i, n in enumerate(grid):
                if i == 0:
                    continue
                fits[str(n)] = power_fit(cells, {c: measured[c][f"moved_{g}"][i] for c in CELLS})
                scaled = {c: measured[c][f"moved_{g}"][i] * math.sqrt(cells[c][1]) / cells[c][0] for c in CELLS}
                coll[str(n)] = {"by_cell": scaled, "largest_over_smallest": max(scaled.values()) / min(scaled.values()),
                                "largest_over_smallest_with_lr_over_batch": (lambda v: max(v) / min(v))(
                                    [measured[c][f"moved_{g}"][i] * cells[c][1] / cells[c][0] for c in CELLS])}
                turn[str(n)] = power_fit(cells, {c: 1.0 - measured[c][f"cos_{g}"][i] for c in CELLS})
            displacement["moved_ratio_when_the_learning_rate_doubles"][g] = {
                str(n): [measured[b][f"moved_{g}"][i] / measured[a][f"moved_{g}"][i] for a, b in doubled]
                for i, n in enumerate(grid) if i}
            displacement["fit_lr_a_times_batch_b"][g] = fits
            displacement["moved_times_root_batch_over_lr"][g] = coll
            displacement["one_minus_cosine_fit"][g] = turn

    # (c) the baseline with the learning rate halved for one group of weights
    samples = fp.sample(12, 32)
    reference = json.loads((REPO / "docs" / "results" / "v2_analysis" / "relay_period_pilot.json").read_text())
    intervention = {"reference_cells_without_a_change": {
        c: {**{k: v[k] for k in ("learning_rate", "batch", "lr_over_root_batch", "textbook", "of")}}
        for c, v in reference["noise_scale"]["cells"].items()}, "arms": {}}
    rows_out = {}
    for arm, js in arms.items():
        cell, factors, predicted = ARMS[arm][:3]
        lr, batch = cells[cell]
        have = [j for j in js if (args.work / f"period_{j['name']}.npz").exists()]
        rows = []
        for j in have:
            r = wp.row_of({**j, "name": f"period_{j['name']}"}, args.work, samples)
            step, st = pp.curves(j, args.work)
            r.update({"name": j["name"], "d1_l0_max": float(st["d1_l0_max"][-1]),
                      "d2_l0_max": float(st["d2_l0_max"][-1]),
                      **{f"{k}_at_equal_sequences": [float(np.interp(n / batch, step, st[k])) for n in grid]
                         for k in KEYS}})
            rows.append(r)
        rows_out[arm] = rows
        n_textbook = int(sum(r["textbook"] for r in rows))
        as_predicted = None
        if predicted and len(rows) == len(js):
            as_predicted = bool(n_textbook >= predicted[1] if predicted[0] == "at least" else
                                n_textbook <= predicted[1])
        intervention["arms"][arm] = {
            "cell": cell, "learning_rate": lr, "batch": batch,
            "textbook_in_the_cell_without_a_change": reference["noise_scale"]["cells"][cell]["textbook"],
            "learning_rate_factors": factors,
            "output_factor_by_layer": {str(k): v for k, v in ARMS[arm][3].items()} if len(ARMS[arm]) > 3 else None,
            "lr_over_root_batch_by_group": {g: lr * factors.get(g, factors.get("all", 1.0)) / math.sqrt(batch)
                                            for g in GROUPS},
            "predicted": list(predicted) if predicted else None, "predicted_by": PREDICTED_BY.get(arm),
            "as_predicted": as_predicted,
            "textbook": n_textbook,
            "relays": int(sum(r["formed"] and not r["textbook"] for r in rows)),
            # of those: layer 1's most attended position is the source itself (not displaced) but the key is not the
            # two-token one; and, of these, the runs with most of layer 1's attention there (the others are relays
            # through a band in layer 0 whose most attended position happens to be the source)
            "at_the_source_without_the_two_token_key": int(sum(
                r["formed"] and not r["textbook"] and r["relay_delta"] == 0 for r in rows)),
            "on_the_source_with_a_one_token_key": int(sum(on_the_source(r) and not r["textbook"] for r in rows)),
            "on_the_source_with_a_one_token_key_by_heads": {f"h{nh}": int(sum(
                on_the_source(r) and not r["textbook"] for r in rows if r["n_head"] == nh)) for nh in rb.HEADS},
            "not_formed": int(sum(not r["formed"] for r in rows)), "of": len(rows), "complete": len(rows) == len(js),
            "by_heads": {f"h{nh}": int(sum(r["textbook"] for r in rows if r["n_head"] == nh)) for nh in rb.HEADS},
            "relay_deltas": [r["relay_delta"] for r in rows],
            "deep_decoy_over_clean": [r["deep_decoy_over_clean"] for r in rows],
            "first_step_with_loss_rep_below_2": [r["first_step_with_loss_rep_below_2"] for r in rows],
            "medians_at_equal_sequences": {k: np.median([r[f"{k}_at_equal_sequences"] for r in rows], axis=0).tolist()
                                           for k in KEYS} if rows else {}}

    # every condition by the noise scale of three groups of weights
    by_scales = [{"condition": c, "kind": "cell", "learning_rate": v["learning_rate"], "batch": v["batch"],
                  **{g: v["lr_over_root_batch"] for g in GROUPS}, "textbook": v["textbook"], "of": v["of"]}
                 for c, v in reference["noise_scale"]["cells"].items()]
    by_scales += [{"condition": arm, "kind": "arm", "learning_rate": o["learning_rate"], "batch": o["batch"],
                   **o["lr_over_root_batch_by_group"], "textbook": o["textbook"], "of": o["of"]}
                  for arm, o in intervention["arms"].items()]
    intervention["every_condition_by_noise_scale_of_the_groups"] = by_scales

    # (d) which layer's band attention passes a level first, against the circuit at the end
    base_final = {r["name"]: r for r in json.loads(
        (REPO / "docs" / "results" / "v2_analysis" / "relay_followup_pilot.json").read_text())["runs"]}
    labelled = {"cells": [], "intervention arms": [], "d_model 32 and 128": [], "rep_min 16": [],
                "intervention arms of the eighth round": [], "rep_min 8 and 20": [],
                "intervention arms of the ninth round": [], "intervention arms of the tenth round": []}
    for cell in CELLS:
        rows = {r["name"]: r for r in reference["runs"].get(f"{cell}_natural", [])}
        for j in pp.arm_jobs(f"{cell}_natural"):
            if cell == "base":                           # the baseline's circuits at 12,000 steps (earlier pilots)
                f = base_final[j["base"]]
                label = bool(f["relay_delta"] == 0 and f["intact"]["deep_decoy_over_clean"] >= 0.9)
            elif j["name"] in rows:
                label = rows[j["name"]]["textbook"]
            else:
                continue
            labelled["cells"].append((j, label))
    for arm, js in arms.items():
        rows = {r["name"]: r for r in rows_out[arm]}
        group = ("intervention arms of the eighth round" if arm in EIGHTH else
                 "intervention arms of the ninth round" if arm in NINTH else
                 "intervention arms of the tenth round" if arm in TENTH else "intervention arms")
        labelled[group] += [(j, rows[j["name"]]["textbook"]) for j in js if j["name"] in rows]
    for group, names in (("d_model 32 and 128", ("d32_natural", "d128_natural")),
                         ("rep_min 16", tuple(f"{c}_natural" for c in pp.SETTINGS if c.startswith("rm16_"))),
                         ("rep_min 8 and 20", tuple(f"{c}_natural" for c in pp.SETTINGS
                                                    if c.startswith(("rm8_", "rm20_"))))):
        for name in names:
            rows = {r["name"]: r for r in reference["runs"].get(name, [])}
            labelled[group] += [(j, rows[j["name"]]["textbook"]) for j in pp.arm_jobs(name) if j["name"] in rows]
    first = {"levels": list(LEVELS), "groups": {}}
    for group, items in labelled.items():
        first["groups"][group] = {}
        for level in LEVELS:
            calls = [(j, layer_1_first(j, args.work, level), label) for j, label in items]
            decided = [(j, c, label) for j, c, label in calls if c is not None]
            first["groups"][group][str(level)] = {
                "runs": len(calls), "decided": len(decided), "correct": int(sum(c == label for _, c, label in decided)),
                "layer_0_first": {"runs": int(sum(not c for _, c, _ in decided)),
                                  "textbook": int(sum(label for _, c, label in decided if not c))},
                "layer_1_first": {"runs": int(sum(c for _, c, _ in decided)),
                                  "textbook": int(sum(label for _, c, label in decided if c))},
                "misses": [{"run": j["name"], "n_head": j["cfg"]["n_head"], "layer_1_first": c, "textbook": label}
                           for j, c, label in decided if c != label]}
    on_cells = first["groups"]["cells"]
    first["level_chosen_on_the_cells"] = max(LEVELS, key=lambda lv: (on_cells[str(lv)]["correct"], lv))

    # the reviewer's combined quantity S, for every condition (the cells and the arms)
    def at_the_source_in_a_cell(cell: str) -> int:
        """Runs of an unchanged cell that end with layer 1 on the source itself, with either key."""
        if cell == "base":                               # the baseline's six runs are displaced relays (earlier pilots)
            ends = [base_final[j["base"]] for j in pp.arm_jobs("base_natural")]
            return int(sum(f["relay_delta"] == 0 and f["intact"]["loss_rep"] < 2 for f in ends))
        return int(sum(on_the_source(r) for r in reference["runs"][f"{cell}_natural"]))

    conditions = []
    for cell, v in intervention["reference_cells_without_a_change"].items():
        lr, batch = cells[cell]
        conditions.append({"condition": cell, "set": "the 25 conditions the ninth reply had", "of": v["of"],
                           "textbook": v["textbook"], "at_the_source": at_the_source_in_a_cell(cell),
                           **combined_quantity(lr, batch, {}, None),
                           "S_asymptotic": combined_quantity(lr, batch, {}, None, 0.0)["S"]})
    for arm, o in intervention["arms"].items():
        output = ARMS[arm][3] if len(ARMS[arm]) > 3 else None
        conditions.append({"condition": arm, "of": o["of"], "textbook": o["textbook"],
                           "at_the_source": o["textbook"] + o["on_the_source_with_a_one_token_key"],
                           "set": ("the ninth round" if arm in NINTH else "the tenth round" if arm in TENTH else
                                   "the 25 conditions the ninth reply had"),
                           **combined_quantity(o["learning_rate"], o["batch"], ARMS[arm][1], output),
                           "S_asymptotic": combined_quantity(o["learning_rate"], o["batch"], ARMS[arm][1], output,
                                                             0.0)["S"]})

    def score(rows: list[dict], outcome: str = "textbook") -> dict:
        rows = [r for r in rows if r["of"]]
        runs = [(r["S"], k < r[outcome]) for r in rows for k in range(r["of"])]
        pos_, neg_ = [v for v, t in runs if t], [v for v, t in runs if not t]
        auc = (sum((a < b) + 0.5 * (a == b) for a in pos_ for b in neg_) / (len(pos_) * len(neg_))
               if pos_ and neg_ else None)
        return {"conditions": len(rows), "runs": len(runs),
                "runs_on_the_predicted_side": int(sum((v < S_THRESHOLD) == t for v, t in runs)),
                "most_any_rule_of_the_settings_could_get": int(sum(max(r[outcome], r["of"] - r[outcome])
                                                                   for r in rows)),
                "conditions_with_the_majority_on_the_predicted_side": int(sum(
                    (r["S"] < S_THRESHOLD) == (2 * r[outcome] > r["of"]) for r in rows
                    if 2 * r[outcome] != r["of"])),
                "conditions_split_evenly": int(sum(2 * r[outcome] == r["of"] for r in rows)),
                "area_under_the_roc_curve": auc}

    # the reply states S for textbook circuits; the second reading (layer 1 on the source with either key, against
    # displaced relays and runs that did not form) was added after the outcomes of the ninth round
    sets = ("the 25 conditions the ninth reply had", "the ninth round")
    combined = {"constants": {"s_star": S_STAR, "s_zero": S_ZERO, "threshold": S_THRESHOLD},
                "conditions": sorted(conditions, key=lambda r: r["S"]),
                **{k: score([r for r in conditions if r["set"] == k]) for k in sets},
                "with_layer_1_on_the_source_as_the_outcome": {
                    k: score([r for r in conditions if r["set"] == k], "at_the_source") for k in sets}}

    # the ninth reply's own criteria for its two sets of settings (None until every run of a set is in)
    def count(*names: str) -> list | None:
        got = [intervention["arms"][a] for a in names]
        return [o["textbook"] for o in got] if all(o["complete"] for o in got) else None

    by_factor = count("lr2e-3_y0_quarter", "lr2e-3_y0_sixteenth", "base_y0_half", "lr5e-4_y0_times2")
    by_quantity = count("qk0_half_y0_times2", "rest_half_y0_quarter", "tok_double")
    criteria = {"output_factor_against_noise_scale": None, "combined_quantity": None}
    if by_factor:
        a1, a2, a3, a4 = by_factor
        criteria["output_factor_against_noise_scale"] = {
            "textbook_of_6": dict(zip(("lr 2e-3, output times 0.25", "lr 2e-3, output times 1/16",
                                       "baseline, output times 0.5", "lr 5e-4, output times 2"), by_factor)),
            "passes": bool(a1 >= 4 and a2 >= 4 and a3 >= 5 and a4 <= 2 and a1 - a4 >= 3),
            "fails": bool(a1 <= 2 or a4 >= 4 or a3 <= 3 or a2 <= 2)}
    if by_quantity:
        c1, c2, c3 = by_quantity
        criteria["combined_quantity"] = {
            "textbook_of_6": dict(zip(("layer 0's query and key weights at half, output times 2",
                                       "every weight but those at half, output times 0.25",
                                       "token embedding at twice the learning rate"), by_quantity)),
            "passes": bool(c1 <= 1 and c2 >= 4 and c3 >= 5), "fails": bool(c1 >= 3 or c2 <= 2 or c3 <= 3)}
    combined["criteria_of_the_ninth_reply"] = criteria

    # (e) the two stages on the way to a textbook circuit, in every condition at rep_min 12 and d_model 64 without a
    # mask (the baseline cell excepted: its six runs are displaced relays), and the runs trained four times as long
    def layer_0_offsets(row: dict) -> list[int]:
        """The most attended offset of each head of layer 0 (1 is the previous token)."""
        return sorted(int(h["top_offsets"][0][0]) for h in row["layer0"].values())

    stage_sets = [(cell, None, pp.arm_jobs(f"{cell}_natural"),
                   {r["name"]: r for r in reference["runs"].get(f"{cell}_natural", [])})
                  for cell in CELLS if cell != "base"]
    stage_sets += [(arm, ARMS[arm][3].get(0) if len(ARMS[arm]) > 3 else None, js,
                    {r["name"]: r for r in rows_out[arm]}) for arm, js in arms.items() if arm not in TENTH]
    stage_rows, by_condition = [], {}
    for name, factor, js, label in stage_sets:
        got = []
        for j in js:
            if j["name"] not in label:
                continue
            got.append({"condition": name, "output_factor": factor, "run": j["base"], "n_head": j["cfg"]["n_head"],
                        "textbook": bool(label[j["name"]]["textbook"]),
                        "layer_0_offsets": layer_0_offsets(label[j["name"]]), **stage_steps(j, args.work)})
        there = [r for r in got if r["on_the_source_at_the_end"]]
        stage_rows += there
        by_condition[name] = {
            "runs": len(got), "on_the_source": len(there), "textbook": int(sum(r["textbook"] for r in there)),
            "source_step": sorted(r["source_step"] for r in there),
            "second_key_step_of_the_textbook_runs": sorted(r["second_key_step"] for r in there if r["textbook"]
                                                           and r["second_key_step"] is not None),
            "second_over_first": sorted(r["second_key_step"] / r["source_step"] for r in there if r["textbook"]
                                        and r["second_key_step"] is not None),
            "not_textbook": [{"run": r["run"], "n_head": r["n_head"], "source_step": r["source_step"],
                              "steps_over_source_step": r["steps"] / r["source_step"],
                              "layer_0_offsets": r["layer_0_offsets"]}
                             for r in there if not r["textbook"]]}
    ratios = np.array([r["second_key_step"] / r["source_step"] for r in stage_rows
                       if r["textbook"] and r["second_key_step"] is not None])
    small = [r for r in stage_rows if r["output_factor"] is not None and r["output_factor"] <= 0.25]
    rest = [r for r in stage_rows if r not in small]

    def tally(rows: list[dict]) -> dict:
        return {"on_the_source": len(rows), "textbook": int(sum(r["textbook"] for r in rows)),
                **{f"h{nh}": [int(sum(r["n_head"] == nh for r in rows)),
                              int(sum(r["textbook"] for r in rows if r["n_head"] == nh))] for nh in rb.HEADS}}

    def heads(rows: list[dict]) -> dict:
        return {"runs": len(rows), "with_a_head_on_offset_2": int(sum(2 in r["layer_0_offsets"] for r in rows)),
                "with_every_head_on_offset_1": int(sum(set(r["layer_0_offsets"]) == {1} for r in rows))}

    long_rows = []
    for j in longer:
        if not (args.work / f"period_{j['name']}.npz").exists():
            continue
        r = wp.row_of({**j, "name": f"period_{j['name']}"}, args.work, samples)
        step, st = pp.curves(j, args.work)
        long_rows.append({"arm": j["arm"][len("noise_"):-len(f"_long{LONG_STEPS}")], "run": j["base"],
                          "n_head": r["n_head"], "textbook": r["textbook"], "relay_delta": r["relay_delta"],
                          "deep_decoy_over_clean": r["deep_decoy_over_clean"], "loss_rep": r["loss_rep"],
                          "d1_l0_max": float(st["d1_l0_max"][-1]), "d2_l0_max": float(st["d2_l0_max"][-1]),
                          "source_l1_at_the_end": r["source_l1_at_the_end"], **stage_steps(j, args.work),
                          "layer_0_offsets": layer_0_offsets(r), "layer1_modal_deltas": r["layer1_modal_deltas"],
                          "loss_rep_every_6000_steps": [float(np.interp(t, step, st["loss_rep"]))
                                                        for t in range(6_000, LONG_STEPS + 1, 6_000)],
                          "d2_l0_max_every_6000_steps": [float(np.interp(t, step, st["d2_l0_max"]))
                                                         for t in range(6_000, LONG_STEPS + 1, 6_000)]})
    two_stages = {
        "second_key_loss": SECOND_KEY_LOSS, "by_condition": by_condition,
        "all_runs_on_the_source": tally(stage_rows),
        "output_factor_a_quarter_or_less": tally(small), "the_other_conditions": tally(rest),
        "layer_0_heads_of_the_textbook_runs": heads([r for r in stage_rows if r["textbook"]]),
        "layer_0_heads_of_the_other_runs_on_the_source": heads([r for r in stage_rows if not r["textbook"]]),
        "second_over_first_in_the_textbook_runs": {
            "runs": int(len(ratios)), "median": float(np.median(ratios)),
            "quartiles": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
            "range": [float(ratios.min()), float(ratios.max())]} if len(ratios) else None,
        "trained_four_times_as_long": {"steps": LONG_STEPS, "runs": long_rows,
                                       "textbook": int(sum(r["textbook"] for r in long_rows)), "of": len(long_rows),
                                       "planned": len(longer)}}

    # the tenth reply. (1) R1 in the first steps, and the arm with layer 1's output weights at twice the rate
    ratio = {}
    for cond, js in ratio_runs.items():
        fs = [args.work / f"noise_ratio_{cond}_{j['base']}.json" for j in js]
        rs = [json.loads(f.read_text()) for f in fs if f.exists()]
        if len(rs) == len(js):
            ratio[cond] = {**{k: np.median([r[k] for r in rs], axis=0).tolist()
                              for k in ("r1", "rms_emb", "rms_y0", "rms_y1", "band_l0", "band_l1")},
                           "r1_at_step_300_by_run": {r["run"]: r["r1"][RATIO_STEPS.index(300)] for r in rs}}
    pairs = [{"slow": a, "fast": b, "factor_on_layer_1_output_weights": f,
              "r1_fast_over_slow": [y / x for x, y in zip(ratio[a]["r1"], ratio[b]["r1"])][1:],
              "y0_fast_over_slow": [y / x for x, y in zip(ratio[a]["rms_y0"], ratio[b]["rms_y0"])][1:]}
             for a, b, f in RATIO_PAIRS if a in ratio and b in ratio]
    doubled = intervention["arms"]["ov1_double"]
    r1_300 = ratio["ov1_double"]["r1"][RATIO_STEPS.index(300)] if "ov1_double" in ratio else None
    n_doubled = doubled["textbook"] if doubled["complete"] else None
    shared_loss = {
        "steps": list(RATIO_STEPS), "conditions": ratio, "pairs": pairs, "r1_at_step_300": r1_300,
        "textbook_of_6": n_doubled,
        "r1_between_0.80_and_1.15": None if r1_300 is None else bool(0.80 <= r1_300 <= 1.15),
        "r1_fails": None if r1_300 is None else bool(r1_300 >= 1.25 or r1_300 < 0.70),
        "count_at_least_4": None if n_doubled is None else bool(n_doubled >= 4),
        "count_fails": None if n_doubled is None else bool(n_doubled <= 2)}

    def final_row(j: dict) -> dict:
        """The circuit a run ends with, where layer 0's heads attend, and its two stages."""
        r = wp.row_of({**j, "name": f"period_{j['name']}"}, args.work, samples)
        step, st = pp.curves(j, args.work)
        after = (step > SWITCH_STEP) & (step <= SWITCH_STEP + 500)
        return {"run": j["base"], "n_head": r["n_head"], "textbook": r["textbook"], "formed": r["formed"],
                "relay_delta": r["relay_delta"], "layer1_modal_deltas": r["layer1_modal_deltas"],
                "on_the_source": on_the_source(r), "deep_decoy_over_clean": r["deep_decoy_over_clean"],
                "loss_rep": r["loss_rep"], "source_l1_at_the_end": r["source_l1_at_the_end"],
                "layer_0_offsets": layer_0_offsets(r), **stage_steps(j, args.work),
                "loss_rep_at_the_switch_step": float(st["loss_rep"][step == SWITCH_STEP][0]),
                "largest_loss_rep_in_the_500_steps_after": float(st["loss_rep"][after].max())}

    # (2A) eight heads with layer 0's output at 1/16
    rows8 = [final_row(j) for j in eight if (args.work / f"period_{j['name']}.npz").exists()]
    n8 = int(sum(r["textbook"] for r in rows8)) if len(rows8) == len(eight) else None
    first3 = [r for r in rows8 if r["run"].endswith(tuple(f"s{x}" for x in EIGHT_SEEDS[:3]))]
    eight_heads = {
        "arm": EIGHT_ARM, "n_head": EIGHT_HEADS, "seeds": list(EIGHT_SEEDS), "runs": rows8, "textbook": n8,
        "textbook_of_the_first_three_seeds": int(sum(r["textbook"] for r in first3)) if len(first3) == 3 else None,
        "displaced_relays": int(sum(r["formed"] and r["relay_delta"] != 0 for r in rows8)),
        "on_the_source_with_a_one_token_key": int(sum(r["on_the_source"] and not r["textbook"] for r in rows8)),
        "at_least_5": None if n8 is None else bool(n8 >= 5), "fails": None if n8 is None else bool(n8 <= 3)}

    # (2B) the factor taken away at SWITCH_STEP, in the 2-head runs of the arms of LONG
    after_switch = {}
    for how in SWITCH_HOW:
        js = [j for j in switched if j["how"] == how]
        got = [{"arm": j["switched"], "stalled_without_the_switch": j["stalled_without_the_switch"], **final_row(j)}
               for j in js if (args.work / f"period_{j['name']}.npz").exists()]
        stalled = [r for r in got if r["stalled_without_the_switch"]]
        control = [r for r in got if not r["stalled_without_the_switch"]]
        sixteenth = [r for r in stalled if r["arm"] == "lr2e-3_y0_sixteenth"]
        done = len(got) == len(js)
        n_s = int(sum(r["textbook"] for r in stalled)) if done else None
        after_switch[how] = {
            "runs": got, "complete": done, "stalled_runs": len(stalled), "textbook_of_the_stalled_runs": n_s,
            "stalled_runs_still_on_the_source_with_a_one_token_key": int(sum(
                r["on_the_source"] and not r["textbook"] for r in stalled)) if done else None,
            "runs_at_one_sixteenth_off_the_source": int(sum(
                not r["on_the_source"] or any(d != 0 for d in r["layer1_modal_deltas"]) for r in sixteenth))
            if done else None,
            "loss_rise_after_the_switch_at_one_sixteenth": [
                r["largest_loss_rep_in_the_500_steps_after"] - r["loss_rep_at_the_switch_step"] for r in sixteenth],
            "control_runs": len(control), "textbook_of_the_control_runs": int(sum(r["textbook"] for r in control))
            if done else None,
            "at_most_2": None if n_s is None else bool(n_s <= 2), "fails": None if n_s is None else bool(n_s >= 4)}
    tenth = {"shared_loss": shared_loss, "eight_heads": eight_heads,
             "factor_taken_away": {"step": SWITCH_STEP, **after_switch}}

    result = {"status": "exploratory pilot; local runs; not independently recomputed",
              "missing": {"clock": [f"{j['cell']} {j['base']}" for j in clock if not (
                  args.work / f"noise_clock_{j['cell']}_{j['base']}.json").exists()],
                          "arms": [j["name"] for js in arms.values() for j in js if not (
                              args.work / f"period_{j['name']}.npz").exists()],
                          "longer": [j["name"] for j in longer if not (
                              args.work / f"period_{j['name']}.npz").exists()],
                          "tenth": [j["name"] for j in eight + switched if not (
                              args.work / f"period_{j['name']}.npz").exists()] + [
                              f"ratio {c} {j['base']}" for c, js in ratio_runs.items() for j in js if not (
                                  args.work / f"noise_ratio_{c}_{j['base']}.json").exists()]},
              "equal_numbers_of_sequences": equal_n, "displacement": displacement, "intervention": intervention,
              "first_layer_past_a_level": first, "combined_quantity_of_the_ninth_reply": combined,
              "two_stages": two_stages, "tenth_reply": tenth, "runs": rows_out}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    pick = [i for i, n in enumerate(grid) if n in (6400, 12800, 19200, 25600, 38400)]
    print("== (a) cached runs at equal numbers of sequences N (medians of six runs); N =", [grid[i] for i in pick])
    for k in KEYS:
        for c in CELLS:
            if cached[c]:
                print(f"   {k:>15s} {c:>12s} lr/sqrt(B) {cells[c][0] / math.sqrt(cells[c][1]):.2e}: "
                      + " ".join(f"{cached[c][k][i]:7.3f}" for i in pick))
    if spread:
        for k in ("band_l0", "band_l1"):
            print(f"   {k}: within the low level", [round(spread[k]["within_low"][i], 3) for i in pick],
                  "| within the high level", [round(spread[k]["within_high"][i], 3) for i in pick],
                  "| between", [round(spread[k]["between"][i], 3) for i in pick])
    if displacement["fit_lr_a_times_batch_b"]:
        print("   batch-128 cells behind the other two of their level:", equal_n["batch_128_lag"])
        print("== (b) distance from the initial weights at equal N, fitted as lr^a B^b over the eight cells "
              "(largest difference of the new runs from the cached ones:", repeat, ")")
        for g in GROUPS:
            f = displacement["fit_lr_a_times_batch_b"][g]
            c = displacement["moved_times_root_batch_over_lr"][g]
            print(f"   {g:>5s}: " + " | ".join(
                f"N {n}: a {f[str(n)]['lr_exponent']:+.2f} b {f[str(n)]['batch_exponent']:+.2f} (spread of "
                f"moved*sqrt(B)/lr x{c[str(n)]['largest_over_smallest']:.2f}; of moved*B/lr "
                f"x{c[str(n)]['largest_over_smallest_with_lr_over_batch']:.2f})" for n in (3200, 12800, 38400)))
        for g in GROUPS:
            r = displacement["moved_ratio_when_the_learning_rate_doubles"][g]
            print(f"   {g:>5s}, distance at twice the learning rate over distance at the same batch size: "
                  + " | ".join(f"N {n}: " + ", ".join(f"{x:.2f}" for x in r[str(n)])
                               for n in (3200, 6400, 12800, 25600, 38400)))
        for g in ("pos", "tok"):
            t = displacement["one_minus_cosine_fit"][g]
            print(f"   1 - cosine to the initial {g}: " + " | ".join(
                f"N {n}: lr^{t[str(n)]['lr_exponent']:+.2f} B^{t[str(n)]['batch_exponent']:+.2f}"
                for n in (3200, 12800, 38400)))
    print("== (c) a cell with the learning rate changed for one group of weights")
    for arm, o in intervention["arms"].items():
        print(f"   {arm:>28s} (cell {o['cell']}, {o['textbook_in_the_cell_without_a_change']}/6 without the change; "
              f"factors {o['learning_rate_factors']}, output {o['output_factor_by_layer']}): textbook "
              f"{o['textbook']}/{o['of']} (2 heads {o['by_heads']['h2']}, 4 heads {o['by_heads']['h4']}) | predicted "
              f"{o['predicted']} by {o['predicted_by']}: {o['as_predicted']} | relay positions "
              f"{o['relay_deltas']} | decoy "
              f"{[round(x, 2) for x in o['deep_decoy_over_clean']]} | loss<2 at "
              f"{o['first_step_with_loss_rep_below_2']}")
    print("   every condition by lr/sqrt(B) of each group of weights (units of 1e-4):")
    for c in sorted(by_scales, key=lambda c: (c["pos"], c["qk0"] / c["qk1"], c["qk0"])):
        print(f"      {c['condition']:>28s} {c['kind']:>4s}: " + " ".join(f"{g} {c[g] * 1e4:.2f}" for g in GROUPS)
              + f" -> textbook {c['textbook']}/{c['of']}")
    print("== (d) which layer's band attention passes a level first, against the circuit at the end (level chosen "
          "on the cells:", first["level_chosen_on_the_cells"], ")")
    for group, blk in first["groups"].items():
        print(f"   {group:>18s}: " + " | ".join(
            f"level {lv}: {b['correct']}/{b['decided']} of {b['runs']} (layer 0 first: {b['layer_0_first']['textbook']}"
            f"/{b['layer_0_first']['runs']} textbook; layer 1 first: {b['layer_1_first']['textbook']}/"
            f"{b['layer_1_first']['runs']})" for lv, b in blk.items()))
    chosen = str(first["level_chosen_on_the_cells"])
    print("   misses at the chosen level:", [(x["run"], f"h{x['n_head']}", x["layer_1_first"], x["textbook"])
                                           for blk in first["groups"].values() for x in blk[chosen]["misses"]])
    print("== the reviewer's combined quantity S (threshold", S_THRESHOLD, "; textbook circuits below)")
    for r in combined["conditions"]:
        print(f"   S {r['S']:7.3f} (asymptotic {r['S_asymptotic']:7.3f}) | qk0/qk1 {r['qk0_over_qk1']:.2f} R "
              f"{r['ratio_of_layer_0_output_to_embeddings']:.2f} position {r['position_share_factor']:.2f} | textbook "
              f"{r['textbook']}/{r['of']}, layer 1 on the source {r['at_the_source']}/{r['of']} | {r['condition']} "
              f"({r['set']})")
    for k in ("the 25 conditions the ninth reply had", "the ninth round", "criteria_of_the_ninth_reply"):
        print(f"   {k}:", json.dumps(combined[k]))
    for k, v in combined["with_layer_1_on_the_source_as_the_outcome"].items():
        print(f"   with layer 1 on the source (either key) as the outcome, {k}:", json.dumps(v))
    print("== (e) two stages: layer 1 on the source, then the second key token (loss below", SECOND_KEY_LOSS,
          "for good)")
    for name, v in two_stages["by_condition"].items():
        if v["on_the_source"]:
            print(f"   {name:>32s}: on the source {v['on_the_source']}/{v['runs']} from step "
                  f"{[int(x) for x in v['source_step']]} | textbook {v['textbook']}, second stage at "
                  f"{[int(x) for x in v['second_key_step_of_the_textbook_runs']]} | second / first "
                  f"{[round(x, 1) for x in v['second_over_first']]} | not textbook: "
                  f"{[(x['run'].split('_', 2)[2], round(x['steps_over_source_step'], 1)) for x in v['not_textbook']]}")
    for k in ("all_runs_on_the_source", "output_factor_a_quarter_or_less", "the_other_conditions",
              "layer_0_heads_of_the_textbook_runs", "layer_0_heads_of_the_other_runs_on_the_source",
              "second_over_first_in_the_textbook_runs"):
        print(f"   {k}:", json.dumps(two_stages[k]))
    longest = two_stages["trained_four_times_as_long"]
    print(f"   trained to {LONG_STEPS} steps: textbook {longest['textbook']}/{longest['of']} (planned "
          f"{longest['planned']})")
    for r in longest["runs"]:
        print(f"      {r['arm']} {r['run']}: textbook {r['textbook']} | decoy {r['deep_decoy_over_clean']:.2f} | loss "
              f"{r['loss_rep']:.2f} | layer-0 heads on offsets {r['layer_0_offsets']} | layer-1 positions "
              f"{r['layer1_modal_deltas']} | second stage at {r['second_key_step']} | loss every 6,000 steps "
              f"{[round(x, 2) for x in r['loss_rep_every_6000_steps']]}")
    print("== the tenth reply (1): R1 at steps", list(RATIO_STEPS)[1:], "(medians of six runs)")
    for cond, v in ratio.items():
        print(f"   {cond:>28s}: R1 {[round(x, 3) for x in v['r1'][1:]]} | layer 0's output "
              f"{[round(x, 3) for x in v['rms_y0'][1:]]} | layer 1's {[round(x, 3) for x in v['rms_y1'][1:]]} | "
              f"embeddings {[round(x, 3) for x in v['rms_emb'][1:]]}")
    for pr in pairs:
        print(f"   {pr['fast']} over {pr['slow']} (factor {pr['factor_on_layer_1_output_weights']}): R1 "
              f"{[round(x, 2) for x in pr['r1_fast_over_slow']]}, layer 0's output "
              f"{[round(x, 2) for x in pr['y0_fast_over_slow']]}")
    print("   ", json.dumps({k: v for k, v in shared_loss.items() if k not in ("conditions", "pairs", "steps")}))
    print("== the tenth reply (2A):", EIGHT_HEADS, "heads on", EIGHT_ARM, "| textbook", eight_heads["textbook"],
          "of", len(eight), "| first three seeds", eight_heads["textbook_of_the_first_three_seeds"],
          "| displaced relays", eight_heads["displaced_relays"], "| one-token key on the source",
          eight_heads["on_the_source_with_a_one_token_key"], "| at least 5:", eight_heads["at_least_5"], "| fails:",
          eight_heads["fails"])
    for r in rows8:
        print(f"      {r['run']}: textbook {r['textbook']} | decoy {r['deep_decoy_over_clean']:.2f} | loss "
              f"{r['loss_rep']:.2f} | layer-0 offsets {r['layer_0_offsets']} | layer-1 positions "
              f"{r['layer1_modal_deltas']} | stages at {r['source_step']}, {r['second_key_step']}")
    for how, v in after_switch.items():
        print(f"== the tenth reply (2B): the factor taken away at step {SWITCH_STEP} ({how}) | stalled runs: textbook "
              f"{v['textbook_of_the_stalled_runs']} of {v['stalled_runs']}, still one-token on the source "
              f"{v['stalled_runs_still_on_the_source_with_a_one_token_key']}, at 1/16 off the source "
              f"{v['runs_at_one_sixteenth_off_the_source']} | control runs: textbook "
              f"{v['textbook_of_the_control_runs']} of {v['control_runs']} | at most 2: {v['at_most_2']} | fails: "
              f"{v['fails']}")
        for r in v["runs"]:
            print(f"      {r['arm']} {r['run']} ({'stalled' if r['stalled_without_the_switch'] else 'control'}): "
                  f"textbook {r['textbook']} | delta {r['relay_delta']} | decoy {r['deep_decoy_over_clean']:.2f} | "
                  f"loss {r['loss_rep']:.2f} | layer-0 offsets {r['layer_0_offsets']} | layer-1 positions "
                  f"{r['layer1_modal_deltas']} | second stage at {r['second_key_step']} | loss at the switch "
                  f"{r['loss_rep_at_the_switch_step']:.2f}, largest in the next 500 steps "
                  f"{r['largest_loss_rep_in_the_500_steps_after']:.2f}")
    print("missing:", {k: len(v) for k, v in result["missing"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
