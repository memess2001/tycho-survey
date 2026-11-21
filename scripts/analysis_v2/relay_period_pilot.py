"""Pilot (exploratory; local runs): does the critical period move with the learning rate and the width?

``relay_window_pilot.py`` found that at rep_min 12 (learning rate 1e-3, d_model 64) keeping layer 0 off the
positional band for 500 steps flips the final circuit if the restriction starts by step 400, and does not if it
starts at step 500 or later. The reviewer's fifth reply (18 November, to commit 4b44e01) gives an account of this
critical period and predicts where it lies at two other learning rates and two other widths. It states that it ran
none of these settings; its predictions are on record here before the runs.

Its account. A masked attention logit gets no gradient, so what layer 0 has learned about the band before the mask
stays in its weights during the mask ("frozen, not erased"). Within the masked window the textbook circuit has to be
built in three stages that wait for each other: layer 1 returns to the source position, a layer-0 head grows on
offset 1, a second one on offset 2. Started at step 400 the three fit into 500 steps, started at step 500 they do
not. Two quantities of the unperturbed run, read before step 400, are said to locate the critical step:

    t_crit_loss = t2 + (0.295 − (loss(0) − loss(t2))) / v,   v = (loss(t1) − loss(t2)) / (t2 − t1)
    t_crit_attn = t2 + log(o(t2) / 0.0085) / u,              u = (log o(t1) − log o(t2)) / (t2 − t1)

with loss the repeat loss and o layer 0's attention on offsets 1 and 2. The two constants (0.295 nats, 0.0085) were
set on the six baseline runs, where the formulas give medians of 451 and 433.

Predictions (rep_min 12; 2 and 4 heads; seeds 300–302; band mask on layer 0 during [start, end); a "flip" is a
textbook circuit at 12,000 steps: relay at the source and a two-token key):

* 1A, learning rate 5e-4. At step 300: layer 0's band attention (median of the six runs) in [0.52, 0.65], its
  attention on offsets 1–2 in [0.028, 0.043]; t_crit_loss from steps 150 and 350 in [760, 980]. Window [600, 1600):
  at least 5 of 6 flip; [700, 1700): at least 4; [1050, 2050): at most 1.
* 1A, learning rate 2e-3. t_crit_loss and t_crit_attn from steps 100 and 200 in [200, 265]; at step 300 the band
  attention is at least 0.85 and the attention on offsets 1–2 at most 0.010. Window [300, 800): at most 1 of 6 flip;
  [150, 650): at least 5.
  1A fails if [600, 1600) at 5e-4 flips 3 or fewer, if [300, 800) at 2e-3 flips 3 or more, or if a median of
  t_crit_loss is outside its interval.
* 1B, d_model 128. At step 300: band attention at least 0.82, attention on offsets 1–2 at most 0.013, drop of the
  repeat loss at least 0.23 nats; t_crit_attn and t_crit_loss in [260, 350]. Window [400, 900): at most 2 of 6 flip
  (6 of 6 at d_model 64); [240, 740): at least 5.
* 1B, d_model 32. At step 300: band attention in [0.54, 0.67], attention on offsets 1–2 at least 0.026, drop of
  the repeat loss in [0.09, 0.14]; t_crit in [560, 760]. Window [500, 1000): at least 4 of 6 flip (0 of 6 at
  d_model 64).
  1B fails if the band attention at step 300 is not higher at d_model 128 and lower at d_model 32 than at 64, if
  [400, 900) at d_model 128 still flips 5 or more, or if [500, 1000) at d_model 32 flips 1 or fewer.

Recorded here before the runs, for comparison. (a) If training simply runs at a speed proportional to the learning
rate, the critical step is 800–1,000 at 5e-4 and 200–250 at 2e-3, and every window of 1A comes out as the reviewer
predicts; 1A cannot tell its account from that. (b) If the width did not matter, [400, 900) would flip 6 of 6 at
d_model 128 and [500, 1000) 0 of 6 at d_model 32: 1B is the informative half. (c) One window is added at d_model 32,
[900, 1400), to see whether there is a critical period at all at that width (the reviewer's interval for t_crit
implies that it fails).

Also run, to look inside the window: the six baseline runs without a mask and with the windows [400, 900) and
[500, 1000), for 2,500 steps, measured every 50 steps with the layer-0 attention on offsets 1 and 2 by head. These
repeat the first 2,500 steps of runs of ``relay_window_pilot.py`` exactly.

Follow-up, recorded after the runs above and before the two arms it adds. Without any mask, all six runs form
the textbook circuit at learning rate 5e-4 and none does at 1e-3 or 2e-3: the learning rate changes the outcome, not
only the clock. Two readings. (i) The learning rate itself decides. (ii) The noise of the updates decides, and a
smaller learning rate is one way to lower it; a larger batch is another. Two arms without a mask tell them apart:
learning rate 1e-3 with batch 128 (reading (i): relays, as at batch 32; reading (ii): textbook circuits), and
learning rate 5e-4 with batch 8 (reading (i): textbook circuits, as at batch 32; reading (ii): relays).
One more window is added at learning rate 2e-3, [400, 900), to bracket the critical period there: the window from
step 300 still flipped 5 of 6, against the reviewer's prediction.

Second follow-up: the reviewer's sixth reply (19 November, to commit 6062933; it states that it ran none of these).
(a) Which noise scale? Under Adam it argues for learning rate over the root of the batch size, not over the batch
size. The five cells above cannot tell the two apart; three more without a mask can: learning rate 2.5e-4 with
batch 8 (same lr/√B as the cells that give textbook circuits, same lr/B as the baseline: it predicts at least 5 of
6 textbook circuits, by step 24,000; lr/B would give at most 1), learning rate 2e-3 with batch 128 (same lr/√B as
the baseline, same lr/B as a textbook cell: it predicts at most 2 of 6; lr/B would give at least 5), and learning
rate 1e-3 with batch 64 (between the two in lr/√B: it predicts 2 to 4 of 6; lr/B would give 6). Its prediction
passes if the first gives at least 5 and the second at most 2, and fails in the opposite case or if the third gives
6 of 6. (b) Why layer 0's band preference comes back during a mask that starts at step 500: its explanation is that
the mask leaves the offset just below the band (10) and the offsets beyond it (32 and more) open, that layer 1 is
by then locked on a relay and drives layer 0 to those offsets, and that the shared weights carry the masked band
offsets along. Added here to test it: the same window with every offset from 4 on masked ("far"), which closes
those leaks, and the band-masked window again with layer 0's attention just below, low in and beyond the band
recorded. If the explanation holds, the preference does not come back under the wider mask.
Added after the first six runs with the wider mask (all six ended with a strong layer-0 head on offset 2, the
mark of the textbook circuit, where the band mask from the same step gave 0 of 6): two later windows with the
wider mask, from step 1,000 (layer 0 holds the band, no circuit yet) and from step 3,000 (the relay has formed).
They ask whether the critical period of the band mask is a point of no return for the network or a limit of that
mask. Either outcome was open when they were started.

Third follow-up: the reviewer's seventh reply (19 November, to commit 8351c3c; it states that it ran none of these).
(a) The noise threshold at rep_min 16 (d_model 64, no mask), where the baseline (learning rate 1e-3, batch 32,
lr/√B 1.77e-4) gives 5 of 6 textbook circuits. From two clocks read off existing runs it predicts the threshold at
lr/√B between 2.0e-4 and 2.9e-4, against about 1.5e-4 at rep_min 12: learning rate 2e-3 with batch 128 (1.77e-4):
at least 4 of 6; learning rate 1e-3 with batch 16 (2.5e-4): at most 4; learning rate 2e-3 with batch 32
(3.54e-4): at most 1; optionally learning rate 1e-3 with batch 8 (3.54e-4, 24,000 steps): at most 1. It fails if
the first gives 2 or fewer, if the third gives 3 or more, or if the second gives 6 of 6 (or 5 with the third at 2
or more). (b) The critical period under the wider mask at other widths (rep_min 12, 500-step windows, every offset
from 4 on masked). d_model 32: from step 900 at least 5 of 6 flip (the band mask from that step gave 0 of 6); from
step 1,500 between 1 and 5; from step 2,400 at most 1. d_model 128: from step 400 at least 5 of 6 are textbook
circuits (the band mask gave 2 of 6), with at least 2 of the three 4-head runs; from step 700 at most 4, with at
most 1 of the 4-head runs and the 2-head runs staying textbook. Optional: d_model 128 from step 550 (4 or 5),
d_model 64 from step 750 (1 to 4). It fails if the first window at d_model 32 flips 2 or fewer, if the third
flips 4 or more, if no 4-head run flips at d_model 128 from step 400, if 2 or more of them flip from step 700, or
if the 2-head runs at d_model 128 fall to 1 of 3 or fewer.
Recorded here before the runs: the baseline of (a) is one cell of six runs, so the first cell of (a) also tests
whether that cell repeats; and for (b) the plain expectation from the wider mask at d_model 64 (flips from step
500, not from step 1,000, where the unperturbed circuit forms at about 1,700) is a critical period at 0.3 to 0.6
of the formation time, that is steps 1,200 to 2,400 at d_model 32 and, for the 4-head runs, 280 to 560 at d_model
128, which agrees with the reviewer on every core window; the two optional windows are where they could differ.
Added after the two core windows at d_model 128 (both gave 6 of 6 textbook circuits; the reviewer had predicted at
most 4 for the one from step 700, and so had the plain expectation above): one later window at d_model 128, from
step 1,200, after the relays of the 4-head runs have formed (steps 840 to 1,030). It asks whether there is a point
of no return at this width at all; at d_model 64 the relay came back after such a window. No prediction on record.
Added after that window (6 of 6 textbook circuits at d_model 128: the three relays that had formed were converted
and did not come back): the same question at d_model 64 with a longer mask, steps 3,000 to 4,999 instead of 3,000
to 3,499. If the relay is converted when the mask lasts four times as long, the "critical period" of a 500-step
mask is a matter of how long the alternative takes to build, not a point of no return. Expectation recorded before
the runs: more likely converted than not (the textbook circuit has the lower loss and the mask leaves it free to
form), without confidence.

Fourth follow-up: the reviewer's eighth reply (20 November, to commit cb49962; it states that it ran none of these).
(a) Two more windows with the wider mask. d_model 128 from step 3,000 for 500 steps: at least 5 of 6 textbook
circuits, with at least 2 of the three 4-head runs (its reading: at this width the relay never empties the parts of
the textbook circuit, and they are more alive at step 3,000 than at step 1,200). d_model 64 from step 1,000 for
2,000 steps: at most 1 of 6, and at least 4 of 6 end in a relay next to the source (position 1 to 3, one-token
key), because the mask leaves offsets 0 to 3 open and such a relay forms inside it. It fails if the first gives 3
or fewer, or the second 3 or more.
(b) The noise threshold at rep_min 8 and 20, by the two clocks it used for rep_min 16 (which gave an interval that
held): rep_min 8 between 0.85e-4 and 1.25e-4, rep_min 20 between 2.3e-4 and 3.3e-4. Cells (d_model 64, no mask,
12,000 steps, six runs each). rep_min 8: learning rate 5e-4 with batch 64 (0.625e-4): at least 4; 5e-4 with 32
(0.884e-4): 2 to 6; 1e-3 with 64 (1.25e-4): at most 2; monotone within one run. Fails if the first gives 2 or
fewer or the third 4 or more. rep_min 20: 1e-3 with 16 (2.5e-4): at least 3; 2e-3 with 32 (3.54e-4): at most 3;
2e-3 with 16 (5e-4): at most 1, with relays at 18 or 19. Fails if the first gives 1 or fewer, the second 5 or
more, or the third 3 or more.
Recorded here before the runs. For (a): no different expectation on either window (the window from step 1,200 at
d_model 128 gave 6 of 6; 500 steps from step 1,000 at d_model 64 gave 0 of 6 with four relays next to the source).
For (b): the two thresholds already measured (rep_min 12 between 1.25e-4 and 1.77e-4, rep_min 16 between 1.77e-4
and 2.5e-4) differ by a factor of about 1.45; carrying that factor one step each way gives about 1.0e-4 at
rep_min 8 and about 3.2e-4 at rep_min 20, and with it nearly the same six predictions. Only the cell at 3.54e-4
for rep_min 20 could tell the reviewer's method from this extrapolation (the reviewer: at most 3, centre 0 to 2;
the extrapolation: 2 to 4), so a pass here says little about the method.

Fifth follow-up: the reviewer's tenth reply (21 November, to commit 9566534; it states that it ran none of these).
The noise threshold at rep_min 4 and 24. From the data distribution alone (no training) it computes, for each
rep_min, the gradient of layer 0 towards the band and of layer 1 towards the source under uniform attention, their
signal-to-noise ratios per sequence, and the gain from copying the band; it anchors these at rep_min 12 (threshold
1.45e-4, ignition after 22,000 sequences) and gets two bounds on the threshold: the ratio of the two gradients over
the growth curve of rep_min 12 at the ignition time (1.78e-4 at rep_min 4, 4.54e-4 at 24), and the gain from copying
the band (0.52e-4 and 6.54e-4). It takes the first below rep_min 12 and the second above: about 1.55e-4 to 1.78e-4
at rep_min 4 and 6.0e-4 to 6.5e-4 at rep_min 24, where it also expects runs that never form. Cells (d_model 64, no
mask, six runs each). rep_min 4: learning rate 1e-3 with batch 64 (1.25e-4; 12,000 steps): at least 4 textbook
circuits, with layer 1 on the source in at least 5; 2e-3 with 64 (2.5e-4; 12,000 steps): at most 1, the relays at
positions 6 to 16. Fails if the first gives 3 or fewer or the second 3 or more. rep_min 24: 1.414e-3 with batch 8
(5.0e-4; 24,000 steps): at least 4, and at most 1 clean relay; 2e-3 with 8 (7.07e-4; 24,000 steps): at most 2, a
mixture of relays at 22 or 23 and runs that do not form. Fails if the first gives 2 or fewer, or 3 or more clean
relays, or if the second gives 4 or more. It also gives the ignition time at rep_min 24 from the same quantities:
about 3,750 to 5,000 sequences.
Recorded here before the runs. The quantities were chosen with the thresholds at rep_min 8, 16 and 20 known, and
which bound applies on which side of rep_min 12 was read off those thresholds, so the cells at rep_min 24 and the
ignition time are what tests the account. The two cells at rep_min 4 are safe: the baseline there (1.77e-4) gave 2
of 6, so every reading of the existing cells puts the threshold between 1.25e-4 and 2.5e-4; Claude's guess is 5 and
1. At rep_min 24 the open question is whether runs form at all at batch 8 with these learning rates; the guess for
the first cell is 4 of 6, without confidence, and for the second at most 2 with most runs unformed, which would
say nothing about relays. Added by Claude: the baseline at rep_min 24 (learning rate 1e-3, batch 32, 1.77e-4;
12,000 steps), which no earlier pilot has; expected 6 of 6. It gives the ignition time at the noise scale at which
the reviewer measured it for rep_min 16 and 20. The ignition time is read as in its ninth reply, with its own
diagnostic (not part of this script): the number of sequences at which the median over the six runs of layer 1's
source score from the current token against layer 0's output passes 0.5, by linear interpolation between
measurements (this reading gives its 13,101 at rep_min 16); here on a grid of 1,600 sequences.
Added before that measurement was made, from the ninth round's rerun of its diagnostic: with the same reading the
baseline cells give 13,101 sequences at rep_min 16 and 10,058 at rep_min 20, a ratio of 0.77 where its formula has
0.56; its 7,300 for rep_min 20 lies between the three cells there (6,888, 10,058 and 10,251), and its anchor of
22,000 at rep_min 12 is the earliest runs, the median being beyond 25,600 in both textbook cells. Claude's
expectation for the baseline at rep_min 24 is therefore later than its 3,750 to 5,000: between 6,000 and 9,000.
Outcome of the fifth follow-up (textbook circuits of 6). rep_min 4: 4 at 1.25e-4, with layer 1 on the source in all
six, and 0 at 2.5e-4 (relays at 9 or 10 in the 2-head runs and at 4, 1 and 1 in the 4-head runs). rep_min 24: 6 at
the baseline; 4 at 5.0e-4, with no relay and two runs that never form; 0 at 7.07e-4, with two relays at 23 and four
runs that never form. All four cells are as the reply had them. The ignition time at the baseline of rep_min 24 is
9,073 sequences (13,205 and 10,507 at rep_min 16 and 20 on the same grid): later than its 3,750 to 5,000, and just
past the upper end of Claude's range.

Usage, from the repository root (needs torch): ``PYTHONPATH=. python scripts/analysis_v2/relay_period_pilot.py
--jobs 4`` (``--skip-runs`` analyses the cache). Needs the outputs of ``relay_followup_pilot.py``,
``relay_band_pilot.py`` and ``relay_window_pilot.py``. Runs are cached in ``work/relay_band``. Writes
``docs/results/v2_analysis/relay_period_pilot.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("relay_window_pilot", Path(__file__).with_name("relay_window_pilot.py"))
wp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp)
rb, fp, rp = wp.rb, wp.fp, wp.rp

FULL_STEPS, SHORT_STEPS, LONG_STEPS = 12_000, 2_500, 24_000
LOSS_DROP, NEAR_FLOOR = 0.295, 0.0085                   # the reviewer's two constants (set on the baseline runs)
SETTINGS = {"base": {}, "lr5e-4": {"lr": 5e-4}, "lr2e-3": {"lr": 2e-3}, "d128": {"d_model": 128},
            "d32": {"d_model": 32}, "b128": {"batch_size": 128}, "lr5e-4_b8": {"lr": 5e-4, "batch_size": 8},
            "lr2.5e-4_b8": {"lr": 2.5e-4, "batch_size": 8}, "lr2e-3_b128": {"lr": 2e-3, "batch_size": 128},
            "b64": {"batch_size": 64},
            "rm16_b16": {"rep_min": 16, "batch_size": 16}, "rm16_lr2e-3": {"rep_min": 16, "lr": 2e-3},
            "rm16_b8": {"rep_min": 16, "batch_size": 8},
            "rm16_lr2e-3_b128": {"rep_min": 16, "lr": 2e-3, "batch_size": 128},
            "rm20_b16": {"rep_min": 20, "batch_size": 16}, "rm20_lr2e-3": {"rep_min": 20, "lr": 2e-3},
            "rm20_lr2e-3_b16": {"rep_min": 20, "lr": 2e-3, "batch_size": 16},
            "rm8_lr5e-4_b64": {"rep_min": 8, "lr": 5e-4, "batch_size": 64}, "rm8_lr5e-4": {"rep_min": 8, "lr": 5e-4},
            "rm8_b64": {"rep_min": 8, "batch_size": 64},
            "rm4_b64": {"rep_min": 4, "batch_size": 64},
            "rm4_lr2e-3_b64": {"rep_min": 4, "lr": 2e-3, "batch_size": 64},
            "rm24_lr1.414e-3_b8": {"rep_min": 24, "lr": 1.414e-3, "batch_size": 8},
            "rm24_lr2e-3_b8": {"rep_min": 24, "lr": 2e-3, "batch_size": 8}, "rm24_base": {"rep_min": 24}}
MODEL_KEYS = ("d_model", "rep_min")                     # settings that change the task or model, not the training
#: arm -> (setting, window [start, end) of the mask on layer 0 or None, steps[, masked offsets: "band" unless
#: given]); in order of priority
ARMS = {
    "base_natural": ("base", None, SHORT_STEPS), "base_w400_900": ("base", (400, 900), SHORT_STEPS),
    "base_w500_1000": ("base", (500, 1000), SHORT_STEPS),
    "d32_natural": ("d32", None, FULL_STEPS), "d32_w500_1000": ("d32", (500, 1000), FULL_STEPS),
    "d32_w900_1400": ("d32", (900, 1400), FULL_STEPS),
    "d128_natural": ("d128", None, FULL_STEPS), "d128_w400_900": ("d128", (400, 900), FULL_STEPS),
    "d128_w240_740": ("d128", (240, 740), FULL_STEPS),
    "lr2e-3_natural": ("lr2e-3", None, FULL_STEPS), "lr2e-3_w300_800": ("lr2e-3", (300, 800), FULL_STEPS),
    "lr2e-3_w150_650": ("lr2e-3", (150, 650), FULL_STEPS),
    "lr5e-4_natural": ("lr5e-4", None, FULL_STEPS), "lr5e-4_w600_1600": ("lr5e-4", (600, 1600), FULL_STEPS),
    "lr5e-4_w700_1700": ("lr5e-4", (700, 1700), FULL_STEPS),
    "lr5e-4_w1050_2050": ("lr5e-4", (1050, 2050), FULL_STEPS),
    "b128_natural": ("b128", None, FULL_STEPS), "lr5e-4_b8_natural": ("lr5e-4_b8", None, FULL_STEPS),
    "lr2e-3_w400_900": ("lr2e-3", (400, 900), FULL_STEPS),
    "base_x_w500_1000": ("base", (500, 1000), SHORT_STEPS, "band"),
    "base_far_w500_1000": ("base", (500, 1000), FULL_STEPS, "far"),
    "base_far_w1000_1500": ("base", (1000, 1500), FULL_STEPS, "far"),
    "base_far_w3000_3500": ("base", (3000, 3500), FULL_STEPS, "far"),
    "lr2.5e-4_b8_natural": ("lr2.5e-4_b8", None, LONG_STEPS), "b64_natural": ("b64", None, FULL_STEPS),
    "lr2e-3_b128_natural": ("lr2e-3_b128", None, FULL_STEPS),
    "d32_far_w900_1400": ("d32", (900, 1400), FULL_STEPS, "far"),
    "d32_far_w1500_2000": ("d32", (1500, 2000), FULL_STEPS, "far"),
    "d32_far_w2400_2900": ("d32", (2400, 2900), FULL_STEPS, "far"),
    "rm16_b16_natural": ("rm16_b16", None, FULL_STEPS), "rm16_lr2e-3_natural": ("rm16_lr2e-3", None, FULL_STEPS),
    "rm16_b8_natural": ("rm16_b8", None, LONG_STEPS),
    "base_far_w750_1250": ("base", (750, 1250), FULL_STEPS, "far"),
    "d128_far_w400_900": ("d128", (400, 900), FULL_STEPS, "far"),
    "d128_far_w700_1200": ("d128", (700, 1200), FULL_STEPS, "far"),
    "rm16_lr2e-3_b128_natural": ("rm16_lr2e-3_b128", None, FULL_STEPS),
    "d128_far_w550_1050": ("d128", (550, 1050), FULL_STEPS, "far"),
    "d128_far_w1200_1700": ("d128", (1200, 1700), FULL_STEPS, "far"),
    "base_far_w3000_5000": ("base", (3000, 5000), FULL_STEPS, "far"),
    "base_far_w1000_3000": ("base", (1000, 3000), FULL_STEPS, "far"),
    "rm20_b16_natural": ("rm20_b16", None, FULL_STEPS),
    "rm20_lr2e-3_b16_natural": ("rm20_lr2e-3_b16", None, FULL_STEPS),
    "rm20_lr2e-3_natural": ("rm20_lr2e-3", None, FULL_STEPS),
    "rm8_lr5e-4_natural": ("rm8_lr5e-4", None, FULL_STEPS), "rm8_b64_natural": ("rm8_b64", None, FULL_STEPS),
    "rm8_lr5e-4_b64_natural": ("rm8_lr5e-4_b64", None, FULL_STEPS),
    "d128_far_w3000_3500": ("d128", (3000, 3500), FULL_STEPS, "far"),
    "rm4_b64_natural": ("rm4_b64", None, FULL_STEPS), "rm4_lr2e-3_b64_natural": ("rm4_lr2e-3_b64", None, FULL_STEPS),
    "rm24_base_natural": ("rm24_base", None, FULL_STEPS),
    "rm24_lr1.414e-3_b8_natural": ("rm24_lr1.414e-3_b8", None, LONG_STEPS),
    "rm24_lr2e-3_b8_natural": ("rm24_lr2e-3_b8", None, LONG_STEPS),
}
#: the steps (t1, t2) the reviewer names for its two formulas, by setting: (loss formula, attention formula)
FORMULA_STEPS = {"base": ((100, 300), (200, 300)), "lr5e-4": ((150, 350), (150, 350)),
                 "lr2e-3": ((100, 200), (100, 200)), "d128": ((100, 300), (200, 300)),
                 "d32": ((100, 300), (200, 300)), "b128": ((100, 300), (200, 300)),
                 "lr5e-4_b8": ((150, 350), (150, 350))}
STATS = ("band_l0", "band_l1", "source_l1", "offsets_1_2_l0", "loss_rep", "d1_l0_max", "d2_l0_max")
#: recorded from the second follow-up on (layer 0, mean over heads): attention on the offset just below the band,
#: on offsets 4 .. below the band, on the lowest nine offsets of the band, and beyond the band
EXTRA_STATS = ("just_below_band_l0", "below_band_l0", "low_band_l0", "beyond_band_l0")


def measure_at(steps: int) -> tuple[int, ...]:
    return tuple(sorted(set(range(0, min(steps, SHORT_STEPS) + 1, 50)) | set(range(0, steps + 1, 250))))


def detail_stats(model, task) -> dict:
    """``band_stats`` of ``relay_band_pilot.py`` and, for layer 0, the largest attention of a head on offset 1 and
    on offset 2 (from the predictable positions of the probe set)."""
    out = rb.band_stats(model, task)
    pr = task._probe
    a0 = model.attention_maps(pr["x"])[0].numpy()        # (B, H, T, T)
    bi, ti = np.nonzero(pr["rep"].numpy())
    out["d1_l0_max"] = float(a0[bi, :, ti, ti - 1].mean(0).max())
    out["d2_l0_max"] = float(a0[bi, :, ti, ti - 2].mean(0).max())

    def mass(offsets) -> float:                          # attention on a set of offsets, mean over heads
        total = np.zeros((len(bi), a0.shape[1]))
        for d in offsets:
            ok = ti >= d
            total[ok] += a0[bi[ok], :, ti[ok], ti[ok] - d]
        return float(total.mean())

    lo, hi = task.rep_min - 1, task.rep_max - 1           # the band
    out["just_below_band_l0"] = mass([lo - 1])
    out["below_band_l0"] = mass(range(4, lo))
    out["low_band_l0"] = mass(range(lo, lo + 9))
    out["beyond_band_l0"] = mass(range(hi + 1, a0.shape[-1]))
    return out


def arm_jobs(arm: str) -> list[dict]:
    setting, window, steps = ARMS[arm][:3]
    forbid = ARMS[arm][3] if len(ARMS[arm]) > 3 else "band"
    base = {j["name"]: j for j in rp.jobs_list()}
    change = SETTINGS[setting]
    rep_min = change.get("rep_min", 12)
    jobs = []
    for nh in rb.HEADS:
        for seed in rb.SEEDS:
            t = base[f"single_rm12_h{nh}_s{seed}"]
            name = f"single_rm{rep_min}_h{nh}_s{seed}"
            jobs.append({"name": f"{arm}_{name}", "base": name,
                         "arm": arm, "setting": setting, "seed": seed, "window": window or (0, 0), "steps": steps,
                         "forbid": forbid,
                         "cfg": {**t["cfg"], **{k: v for k, v in change.items() if k in MODEL_KEYS}},
                         "train": {**t["train"], **{k: v for k, v in change.items() if k not in MODEL_KEYS},
                                   "steps": steps},
                         "released_id": None})
    return jobs


def run_arm(args: tuple[dict, str]) -> str:
    job, work = args
    out = Path(work) / f"period_{job['name']}.npz"
    if out.exists():
        return f"cached {out.name}"
    from tycho.tasks.patterns import MAX_PATTERN_INPUTS

    at = measure_at(job["steps"])
    model, task, stats, _ = rb.train(job, job["steps"], at, block_from=job["window"][0],
                                     block_until=job["window"][1], layer=0, forbid=job["forbid"],
                                     stats_fn=detail_stats, lr_scale=job.get("lr_scale"),
                                     out_scale=job.get("out_scale"), out_switch=job.get("out_switch"))
    model.eval()
    arrays = {f"param/{k}": v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update({f"stat_{k}": np.asarray([stats[s][k] for s in at]) for k in STATS + EXTRA_STATS})
    heads = int(job["cfg"]["n_layer"]) * int(job["cfg"]["n_head"])    # the ablation table has 2^heads entries
    table = task.pattern_logodds(model)[None, :] if heads <= MAX_PATTERN_INPUTS else np.zeros((1, 0))
    np.savez_compressed(out, logodds=table, steps=np.asarray([job["steps"]]),
                        lc_step=np.asarray(at), lc_loss_rep=arrays["stat_loss_rep"],
                        object_id=np.asarray(f"{job['arm']}:{job['base']}"), **arrays)
    return f"done {job['name']}"


def curves(job: dict, work: Path) -> tuple[np.ndarray, dict]:
    with np.load(work / f"period_{job['name']}.npz") as f:
        return f["lc_step"].astype(float), {k: f[f"stat_{k}"] for k in STATS + EXTRA_STATS if f"stat_{k}" in f.files}


def first_crossing(step: np.ndarray, y: np.ndarray, level: float, rising: bool) -> float | None:
    """The first step at which y passes the level (linear interpolation between measurements)."""
    hit = (y >= level) if rising else (y <= level)
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    if i == 0:
        return float(step[0])
    return float(step[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (step[i] - step[i - 1]))


def early(job: dict, work: Path) -> dict:
    """What an unperturbed run shows early on: the reviewer's quantities at step 300, its two extrapolated critical
    steps, and where the same two levels are actually crossed."""
    step, st = curves(job, work)
    at = {int(s): i for i, s in enumerate(step)}
    loss, near = st["loss_rep"], st["offsets_1_2_l0"]
    (a1, a2), (b1, b2) = FORMULA_STEPS.get(job["setting"], ((100, 300), (200, 300)))
    v = (loss[at[a1]] - loss[at[a2]]) / (a2 - a1)
    u = (math.log(near[at[b1]]) - math.log(near[at[b2]])) / (b2 - b1)
    after = step >= 100                                  # the two layers are equal at step 0
    behind = np.nonzero(after & (st["band_l1"] >= st["band_l0"]))[0]
    return {"band_l0_at_300": float(st["band_l0"][at[300]]), "offsets_1_2_l0_at_300": float(near[at[300]]),
            "loss_drop_at_300": float(loss[0] - loss[at[300]]),
            "t_crit_loss": float(a2 + (LOSS_DROP - (loss[0] - loss[at[a2]])) / v) if v > 0 else None,
            "t_crit_attn": float(b2 + math.log(near[at[b2]] / NEAR_FLOOR) / u) if u > 0 else None,
            "measured": {"loss_drop_reaches_0.295": first_crossing(step, loss[0] - loss, LOSS_DROP, True),
                         "offsets_1_2_l0_falls_to_0.0085": first_crossing(step, near, NEAR_FLOOR, False),
                         "last_step_with_layer_1_ahead_in_the_band": float(step[behind[-1]]) if len(behind) else None,
                         "band_l1_falls_to_0.33": first_crossing(step[after], st["band_l1"][after], 0.33, False),
                         "loss_rep_falls_below_2": first_crossing(step, loss, 2.0, False)}}


def median(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_band")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-runs", action="store_true", help="analyse the cached runs only")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_period_pilot.json")
    args = ap.parse_args(argv)
    jobs = {arm: arm_jobs(arm) for arm in ARMS}
    args.work.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        with Pool(args.jobs) as pool:
            for msg in pool.imap(run_arm, [(j, str(args.work)) for js in jobs.values() for j in js]):
                print(msg, flush=True)
    have = {arm: [j for j in js if (args.work / f"period_{j['name']}.npz").exists()] for arm, js in jobs.items()}
    samples = {12: fp.sample(12, 32)}                    # the common sequences on which circuits are classified

    # final circuits of the full-length arms
    rows = {}
    for arm, js in have.items():
        if ARMS[arm][2] < FULL_STEPS:
            continue
        rows[arm] = []
        for j in js:
            rep_min = j["cfg"]["rep_min"]
            if rep_min not in samples:
                samples[rep_min] = fp.sample(rep_min, 32)
            r = wp.row_of({**j, "name": f"period_{j['name']}"}, args.work, samples[rep_min])
            step, st = curves(j, args.work)
            r.update({"name": j["name"], "d1_l0_max": float(st["d1_l0_max"][-1]),
                      "d2_l0_max": float(st["d2_l0_max"][-1])})
            rows[arm].append(r)

    def outcome(arm: str) -> dict:
        rs = rows.get(arm, [])
        return {"window": list(ARMS[arm][1]) if ARMS[arm][1] else None,
                "textbook": int(sum(r["textbook"] for r in rs)),
                "relays": int(sum(r["formed"] and not r["textbook"] for r in rs)),
                "not_formed": int(sum(not r["formed"] for r in rs)), "of": len(rs),
                "complete": len(rs) == len(jobs[arm]), "relay_deltas": [r["relay_delta"] for r in rs],
                "deep_decoy_over_clean": [r["deep_decoy_over_clean"] for r in rs],
                "loss_all": [r["loss_all"] for r in rs], "d2_l0_max": [r["d2_l0_max"] for r in rs],
                "first_step_with_loss_rep_below_2": [r["first_step_with_loss_rep_below_2"] for r in rs]}

    settings = {}
    for setting in SETTINGS:
        nat = f"{setting}_natural"
        per_run = {j["base"]: early(j, args.work) for j in have[nat]}
        keys = ("band_l0_at_300", "offsets_1_2_l0_at_300", "loss_drop_at_300", "t_crit_loss", "t_crit_attn")
        settings[setting] = {
            "formula_steps": dict(zip(("loss", "attention"), map(list, FORMULA_STEPS.get(
                setting, ((100, 300), (200, 300)))))),
            "unperturbed_runs": per_run,
            "medians": {**{k: median([r[k] for r in per_run.values()]) for k in keys},
                        "measured": {k: median([r["measured"][k] for r in per_run.values()])
                                     for k in next(iter(per_run.values()))["measured"]} if per_run else {}},
            "arms": {arm: outcome(arm) for arm in ARMS if ARMS[arm][0] == setting and ARMS[arm][2] >= FULL_STEPS}}

    # two things the reviewer's account does not cover: does the step-300 criterion of relay_window_pilot.py carry
    # over to other settings, and can a mask undo a textbook circuit that forms without it
    lost = []
    for setting, blk in settings.items():
        nat = {r["name"].split("_natural_")[1]: r for r in rows.get(f"{setting}_natural", [])}
        if not nat:
            continue
        blk["step_300_source_criterion"] = {
            "threshold": wp.SOURCE_THRESHOLD, "of": len(nat),
            "correct": int(sum((r["at_step_300"]["source_l1"] >= wp.SOURCE_THRESHOLD) == r["textbook"]
                               for r in nat.values())),
            "textbook_without_mask": int(sum(r["textbook"] for r in nat.values()))}
        for arm in blk["arms"]:
            if arm.endswith("_natural"):
                continue
            for r in rows.get(arm, []):
                base_name = r["name"].split(arm + "_")[1]
                if nat[base_name]["textbook"] and not r["textbook"]:
                    lost.append({"arm": arm, "run": base_name, "deep_decoy_over_clean": r["deep_decoy_over_clean"],
                                 "relay_delta": r["relay_delta"], "formed": r["formed"]})

    def flips(arm: str) -> int | None:
        o = settings[ARMS[arm][0]]["arms"][arm]
        return o["textbook"] if o["complete"] else None

    def med(setting: str, key: str):
        return settings[setting]["medians"].get(key)

    def within(x, lo, hi) -> bool | None:
        return None if x is None else bool(lo <= x <= hi)

    def test(x, fn) -> bool | None:
        return None if x is None else bool(fn(x))

    prediction_1a = {
        "lr5e-4": {"band_l0_at_300_in_0.52_0.65": within(med("lr5e-4", "band_l0_at_300"), 0.52, 0.65),
                   "offsets_1_2_l0_at_300_in_0.028_0.043": within(med("lr5e-4", "offsets_1_2_l0_at_300"), 0.028, 0.043),
                   "t_crit_loss_in_760_980": within(med("lr5e-4", "t_crit_loss"), 760, 980),
                   "w600_1600_at_least_5": test(flips("lr5e-4_w600_1600"), lambda n: n >= 5),
                   "w700_1700_at_least_4": test(flips("lr5e-4_w700_1700"), lambda n: n >= 4),
                   "w1050_2050_at_most_1": test(flips("lr5e-4_w1050_2050"), lambda n: n <= 1)},
        "lr2e-3": {"t_crit_loss_in_200_265": within(med("lr2e-3", "t_crit_loss"), 200, 265),
                   "t_crit_attn_in_200_265": within(med("lr2e-3", "t_crit_attn"), 200, 265),
                   "band_l0_at_300_at_least_0.85": test(med("lr2e-3", "band_l0_at_300"), lambda x: x >= 0.85),
                   "offsets_1_2_l0_at_300_at_most_0.010": test(med("lr2e-3", "offsets_1_2_l0_at_300"),
                                                               lambda x: x <= 0.010),
                   "w300_800_at_most_1": test(flips("lr2e-3_w300_800"), lambda n: n <= 1),
                   "w150_650_at_least_5": test(flips("lr2e-3_w150_650"), lambda n: n >= 5)},
        "fails_by_its_own_criteria": None}
    f600, f300 = flips("lr5e-4_w600_1600"), flips("lr2e-3_w300_800")
    if f600 is not None and f300 is not None:
        prediction_1a["fails_by_its_own_criteria"] = bool(
            f600 <= 3 or f300 >= 3 or prediction_1a["lr5e-4"]["t_crit_loss_in_760_980"] is False
            or prediction_1a["lr2e-3"]["t_crit_loss_in_200_265"] is False)
    b64, b128, b32 = (med(s, "band_l0_at_300") for s in ("base", "d128", "d32"))
    prediction_1b = {
        "d128": {"band_l0_at_300_at_least_0.82": test(b128, lambda x: x >= 0.82),
                 "offsets_1_2_l0_at_300_at_most_0.013": test(med("d128", "offsets_1_2_l0_at_300"),
                                                             lambda x: x <= 0.013),
                 "loss_drop_at_300_at_least_0.23": test(med("d128", "loss_drop_at_300"), lambda x: x >= 0.23),
                 "t_crit_loss_in_260_350": within(med("d128", "t_crit_loss"), 260, 350),
                 "t_crit_attn_in_260_350": within(med("d128", "t_crit_attn"), 260, 350),
                 "w400_900_at_most_2": test(flips("d128_w400_900"), lambda n: n <= 2),
                 "w240_740_at_least_5": test(flips("d128_w240_740"), lambda n: n >= 5)},
        "d32": {"band_l0_at_300_in_0.54_0.67": within(b32, 0.54, 0.67),
                "offsets_1_2_l0_at_300_at_least_0.026": test(med("d32", "offsets_1_2_l0_at_300"), lambda x: x >= 0.026),
                "loss_drop_at_300_in_0.09_0.14": within(med("d32", "loss_drop_at_300"), 0.09, 0.14),
                "t_crit_loss_in_560_760": within(med("d32", "t_crit_loss"), 560, 760),
                "t_crit_attn_in_560_760": within(med("d32", "t_crit_attn"), 560, 760),
                "w500_1000_at_least_4": test(flips("d32_w500_1000"), lambda n: n >= 4)},
        "fails_by_its_own_criteria": None}
    f400, f500 = flips("d128_w400_900"), flips("d32_w500_1000")
    if None not in (f400, f500, b64, b128, b32):
        prediction_1b["fails_by_its_own_criteria"] = bool(b128 <= b64 or b32 >= b64 or f400 >= 5 or f500 <= 1)

    # inside the window: medians over the six baseline runs, every 100 steps
    inside = {}
    for arm in ("base_natural", "base_w400_900", "base_w500_1000"):
        cs = [curves(j, args.work) for j in have[arm]]
        if not cs:
            continue
        step = cs[0][0]
        pick = [i for i, s in enumerate(step) if s % 100 == 0 and 300 <= s <= SHORT_STEPS]
        inside[arm] = {"step": [int(step[i]) for i in pick],
                       **{k: [float(np.median([c[1][k][i] for c in cs])) for i in pick] for k in STATS},
                       "runs": {j["base"]: {k: [float(c[1][k][i]) for i in pick] for k in STATS}
                                for j, c in zip(have[arm], cs)}}

    # second follow-up (a): the unperturbed outcome by learning rate and batch size (d_model 64)
    base_final = {r["name"]: r for r in json.loads(
        (REPO / "docs" / "results" / "v2_analysis" / "relay_followup_pilot.json").read_text())["runs"]}
    noise = {}
    for setting, change in SETTINGS.items():
        if any(k in change for k in MODEL_KEYS) or f"{setting}_natural" not in ARMS:
            continue
        lr, batch = change.get("lr", 1e-3), change.get("batch_size", 32)
        if setting == "base":                            # the baseline's circuits at 12,000 steps (earlier pilots)
            finals = [base_final[j["base"]] for j in jobs["base_natural"]]
            textbook = sum(r["relay_delta"] == 0 and r["intact"]["deep_decoy_over_clean"] >= 0.9 for r in finals)
            cell = {"textbook": int(textbook), "relays": len(finals) - int(textbook), "not_formed": 0,
                    "of": len(finals), "complete": True, "steps": FULL_STEPS}
        else:
            o = settings[setting]["arms"][f"{setting}_natural"]
            cell = {k: o[k] for k in ("textbook", "relays", "not_formed", "of", "complete")}
            cell.update({"steps": ARMS[f"{setting}_natural"][2], "relay_deltas": o["relay_deltas"],
                         "first_step_with_loss_rep_below_2": o["first_step_with_loss_rep_below_2"]})
        noise[setting] = {"learning_rate": lr, "batch": batch, "lr_over_root_batch": lr / math.sqrt(batch),
                          "lr_over_batch": lr / batch, "lr_times_root_batch": lr * math.sqrt(batch), **cell,
                          "loss_drop_at_300": settings[setting]["medians"].get("loss_drop_at_300"),
                          "loss_rep_falls_below_2": settings[setting]["medians"].get("measured", {}).get(
                              "loss_rep_falls_below_2")}
    long_arm = "lr2.5e-4_b8_natural"                     # what its runs look like half-way, from the statistics
    half_way = {}
    for j in have[long_arm]:
        step, st = curves(j, args.work)
        i = int(np.argmin(np.abs(step - FULL_STEPS)))
        half_way[j["base"]] = {k: float(st[k][i]) for k in ("loss_rep", "source_l1", "d1_l0_max", "d2_l0_max")}

    def cell_count(setting: str) -> int | None:
        c = noise.get(setting)
        return c["textbook"] if c and c["complete"] else None

    low, high, mid = cell_count("lr2.5e-4_b8"), cell_count("lr2e-3_b128"), cell_count("b64")
    noise_prediction = {
        "low_noise_by_root_batch_at_least_5": test(low, lambda n: n >= 5),
        "high_noise_by_root_batch_at_most_2": test(high, lambda n: n <= 2),
        "between_2_to_4": test(mid, lambda n: 2 <= n <= 4),
        "passes_by_its_own_criteria": None if None in (low, high) else bool(low >= 5 and high <= 2),
        "fails_by_its_own_criteria": None if None in (low, high, mid) else bool((high >= 5 and low <= 2) or mid == 6),
        "runs_at_lr_2.5e-4_batch_8_at_step_12000": half_way}

    # second follow-up (b): does layer 0's band preference come back when the leaks are closed
    leak = {}
    for arm in ("base_x_w500_1000", "base_far_w500_1000"):
        cs = [curves(j, args.work) for j in have[arm]]
        if not cs:
            continue
        step = cs[0][0]
        pick = [i for i, s in enumerate(step) if s in (400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000, 2500)]
        keys = ("band_l0", "just_below_band_l0", "below_band_l0", "low_band_l0", "beyond_band_l0", "band_l1",
                "source_l1", "d1_l0_max", "d2_l0_max", "loss_rep")
        leak[arm] = {"masked_offsets": ARMS[arm][3], "window": list(ARMS[arm][1]),
                     "step": [int(step[i]) for i in pick],
                     **{k: [float(np.median([c[1][k][i] for c in cs])) for i in pick] for k in keys}}
    if "base_far_w500_1000" in settings["base"]["arms"]:
        leak["outcome_with_every_offset_from_4_masked"] = settings["base"]["arms"]["base_far_w500_1000"]

    # third follow-up (a): the unperturbed outcome at rep_min 16 by learning rate and batch size
    def by_heads(rs: list[dict]) -> dict:
        return {f"h{nh}": {"textbook": int(sum(r["textbook"] for r in rs if r["n_head"] == nh)),
                           "of": int(sum(r["n_head"] == nh for r in rs))} for nh in rb.HEADS}

    def cells_at(rep_min: int) -> dict:
        """The unperturbed cells at one rep_min: the baseline of the earlier pilots (where there is one) and the
        settings here."""
        cells = {}
        if f"single_rm{rep_min}_h{rb.HEADS[0]}_s{rb.SEEDS[0]}" in base_final:
            finals = [{"n_head": nh, **base_final[f"single_rm{rep_min}_h{nh}_s{seed}"]}
                      for nh in rb.HEADS for seed in rb.SEEDS]
            for r in finals:
                r["textbook"] = bool(r["relay_delta"] == 0 and r["intact"]["deep_decoy_over_clean"] >= 0.9)
            cells["base"] = {"learning_rate": 1e-3, "batch": 32, "lr_over_root_batch": 1e-3 / math.sqrt(32),
                             "lr_over_batch": 1e-3 / 32, "textbook": int(sum(r["textbook"] for r in finals)),
                             "of": len(finals), "complete": True, "steps": FULL_STEPS, "by_heads": by_heads(finals),
                             "relay_deltas": [r["relay_delta"] for r in finals]}
        for setting, change in SETTINGS.items():
            if change.get("rep_min") != rep_min:
                continue
            lr, batch = change.get("lr", 1e-3), change.get("batch_size", 32)
            arm = f"{setting}_natural"
            o = settings[setting]["arms"][arm]
            rs = rows.get(arm, [])
            cells[setting] = {"learning_rate": lr, "batch": batch, "lr_over_root_batch": lr / math.sqrt(batch),
                              "lr_over_batch": lr / batch,
                              **{k: o[k] for k in ("textbook", "relays", "not_formed", "of", "complete",
                                                   "relay_deltas", "first_step_with_loss_rep_below_2")},
                              # of the runs that formed: layer 1 on the source itself, or displaced (a clean relay)
                              "on_the_source": int(sum(r["formed"] and r["relay_delta"] == 0 for r in rs)),
                              "displaced_relays": int(sum(r["formed"] and r["relay_delta"] != 0 for r in rs)),
                              "steps": ARMS[arm][2], "by_heads": by_heads(rs)}
        return cells

    noise16, noise8, noise20 = cells_at(16), cells_at(8), cells_at(20)
    noise4, noise24 = cells_at(4), cells_at(24)

    def count16(setting: str) -> int | None:
        c = noise16[setting]
        return c["textbook"] if c["complete"] else None

    def count_in(cells: dict, setting: str) -> int | None:
        return cells[setting]["textbook"] if cells[setting]["complete"] else None

    # fourth follow-up (b): the thresholds at rep_min 8 and 20
    a8, b8, c8 = (count_in(noise8, k) for k in ("rm8_lr5e-4_b64", "rm8_lr5e-4", "rm8_b64"))
    prediction8 = {
        "C8A_lr_5e-4_batch_64_at_least_4": test(a8, lambda n: n >= 4),
        "C8B_lr_5e-4_batch_32_between_2_and_6": test(b8, lambda n: 2 <= n <= 6),
        "C8C_lr_1e-3_batch_64_at_most_2": test(c8, lambda n: n <= 2),
        "monotone_within_one_run": None if None in (a8, b8, c8) else bool(a8 >= b8 - 1 and b8 >= c8 - 1),
        "passes_by_its_own_criteria": None if None in (a8, b8, c8) else bool(
            a8 >= 4 and c8 <= 2 and a8 >= b8 - 1 and b8 >= c8 - 1),
        "fails_by_its_own_criteria": None if None in (a8, c8) else bool(a8 <= 2 or c8 >= 4)}
    a20, b20, c20 = (count_in(noise20, k) for k in ("rm20_b16", "rm20_lr2e-3", "rm20_lr2e-3_b16"))
    prediction20 = {
        "C20A_lr_1e-3_batch_16_at_least_3": test(a20, lambda n: n >= 3),
        "C20B_lr_2e-3_batch_32_at_most_3": test(b20, lambda n: n <= 3),
        "C20C_lr_2e-3_batch_16_at_most_1": test(c20, lambda n: n <= 1),
        "passes_by_its_own_criteria": None if None in (a20, b20, c20) else bool(a20 >= 3 and b20 <= 3 and c20 <= 1),
        "fails_by_its_own_criteria": None if None in (a20, b20, c20) else bool(a20 <= 1 or b20 >= 5 or c20 >= 3)}

    # fifth follow-up: the thresholds at rep_min 4 and 24 (the reviewer's tenth reply)
    a4, b4 = (count_in(noise4, k) for k in ("rm4_b64", "rm4_lr2e-3_b64"))
    prediction4 = {
        "4A_lr_1e-3_batch_64_at_least_4": test(a4, lambda n: n >= 4),
        "4A_layer_1_on_the_source_in_at_least_5": None if a4 is None else bool(
            noise4["rm4_b64"]["on_the_source"] >= 5),
        "4B_lr_2e-3_batch_64_at_most_1": test(b4, lambda n: n <= 1),
        "4B_relay_positions": None if b4 is None else noise4["rm4_lr2e-3_b64"]["relay_deltas"],
        "passes_by_its_own_criteria": None if None in (a4, b4) else bool(a4 >= 4 and b4 <= 1),
        "fails_by_its_own_criteria": None if None in (a4, b4) else bool(a4 <= 3 or b4 >= 3)}
    a24, b24 = (count_in(noise24, k) for k in ("rm24_lr1.414e-3_b8", "rm24_lr2e-3_b8"))
    relays24 = noise24["rm24_lr1.414e-3_b8"]["displaced_relays"]
    prediction24 = {
        "24A_lr_1.414e-3_batch_8_at_least_4": test(a24, lambda n: n >= 4),
        "24A_at_most_1_clean_relay": None if a24 is None else bool(relays24 <= 1),
        "24B_lr_2e-3_batch_8_at_most_2": test(b24, lambda n: n <= 2),
        "passes_by_its_own_criteria": None if None in (a24, b24) else bool(a24 >= 4 and relays24 <= 1 and b24 <= 2),
        "fails_by_its_own_criteria": None if None in (a24, b24) else bool(a24 <= 2 or relays24 >= 3 or b24 >= 4)}

    n_a, n_b, n_c, n_d = (count16(k) for k in ("rm16_lr2e-3_b128", "rm16_b16", "rm16_lr2e-3", "rm16_b8"))
    prediction16 = {
        "1A_lr_2e-3_batch_128_at_least_4": test(n_a, lambda n: n >= 4),
        "1B_lr_1e-3_batch_16_at_most_4": test(n_b, lambda n: n <= 4),
        "1C_lr_2e-3_batch_32_at_most_1": test(n_c, lambda n: n <= 1),
        "1D_lr_1e-3_batch_8_at_most_1": test(n_d, lambda n: n <= 1),
        "passes_by_its_own_criteria": None if None in (n_a, n_b, n_c) else bool(
            n_a >= 4 and n_c <= 1 and n_b <= 4 and (n_d is None or n_d <= 1)),
        "fails_by_its_own_criteria": None if None in (n_a, n_b, n_c) else bool(
            n_a <= 2 or n_c >= 3 or n_b == 6 or (n_b == 5 and n_c >= 2))}

    # third follow-up (b): windows with the wider mask, by width
    far = {arm: {**outcome(arm), "by_heads": by_heads(rows.get(arm, []))} for arm in ARMS
           if len(ARMS[arm]) > 3 and ARMS[arm][3] == "far" and ARMS[arm][2] >= FULL_STEPS}

    def far_count(arm: str, heads: int | None = None) -> int | None:
        o = far[arm]
        if not o["complete"]:
            return None
        return o["textbook"] if heads is None else o["by_heads"][f"h{heads}"]["textbook"]

    f_a, f_b, f_c = (far_count(a) for a in ("d32_far_w900_1400", "d32_far_w1500_2000", "d32_far_w2400_2900"))
    f_d, f_e = far_count("d128_far_w400_900"), far_count("d128_far_w700_1200")
    d_h4, e_h4 = far_count("d128_far_w400_900", 4), far_count("d128_far_w700_1200", 4)
    d_h2, e_h2 = far_count("d128_far_w400_900", 2), far_count("d128_far_w700_1200", 2)
    core = (f_a, f_c, f_d, f_e)
    prediction_far = {
        "2A_d32_from_900_at_least_5": test(f_a, lambda n: n >= 5),
        "2B_d32_from_1500_between_1_and_5": test(f_b, lambda n: 1 <= n <= 5),
        "2C_d32_from_2400_at_most_1": test(f_c, lambda n: n <= 1),
        "2D_d128_from_400_at_least_5_with_2_of_the_4_head_runs": None if f_d is None else bool(f_d >= 5 and d_h4 >= 2),
        "2E_d128_from_700_at_most_4_with_at_most_1_of_the_4_head_runs": None if None in (f_d, f_e) else bool(
            f_e <= 4 and e_h4 <= 1 and e_h4 < d_h4 and e_h2 >= 2),
        "2F_d128_from_550_is_4_or_5": test(far_count("d128_far_w550_1050"), lambda n: 4 <= n <= 5),
        "2G_d64_from_750_between_1_and_4": test(far_count("base_far_w750_1250"), lambda n: 1 <= n <= 4),
        "passes_by_its_own_criteria": None if None in core else bool(
            f_a >= 5 and f_c <= 1 and f_d >= 5 and d_h4 >= 2 and f_e <= 4 and e_h4 <= 1 and e_h4 < d_h4
            and e_h2 >= 2),
        "fails_by_its_own_criteria": None if None in core else bool(
            f_a <= 2 or f_c >= 4 or d_h4 == 0 or e_h4 >= 2 or d_h2 <= 1 or e_h2 <= 1)}
    # fourth follow-up (a): two more windows
    g_b1, g_b1_h4, g_b2 = (far_count("d128_far_w3000_3500"), far_count("d128_far_w3000_3500", 4),
                           far_count("base_far_w1000_3000"))
    near = int(sum(r["formed"] and r["relay_delta"] in (1, 2, 3) and r["deep_decoy_over_clean"] < 0.75
                   for r in rows.get("base_far_w1000_3000", [])))
    prediction_far8 = {
        "B1_d128_from_3000_at_least_5_with_2_of_the_4_head_runs": None if g_b1 is None else bool(
            g_b1 >= 5 and g_b1_h4 >= 2),
        "B2_d64_from_1000_for_2000_steps_at_most_1": test(g_b2, lambda n: n <= 1),
        "B2_relays_next_to_the_source_with_a_one_token_key": None if g_b2 is None else near,
        "B2_at_least_4_of_them": None if g_b2 is None else bool(near >= 4),
        "passes_by_its_own_criteria": None if None in (g_b1, g_b2) else bool(
            g_b1 >= 5 and g_b1_h4 >= 2 and g_b2 <= 1 and near >= 4),
        "fails_by_its_own_criteria": None if None in (g_b1, g_b2) else bool(g_b1 <= 3 or g_b2 >= 3)}

    result = {"status": "exploratory pilot; local runs; not independently recomputed",
              "missing": [j["name"] for arm, js in jobs.items() for j in js if j not in have[arm]],
              "noise_scale": {"cells": noise, "prediction": noise_prediction}, "band_preference_under_a_mask": leak,
              "noise_scale_at_rep_min_16": {"cells": noise16, "prediction": prediction16},
              "noise_scale_at_rep_min_8": {"cells": noise8, "prediction": prediction8},
              "noise_scale_at_rep_min_20": {"cells": noise20, "prediction": prediction20},
              "noise_scale_at_rep_min_4": {"cells": noise4, "prediction": prediction4},
              "noise_scale_at_rep_min_24": {"cells": noise24, "prediction": prediction24},
              "wider_mask_windows": {"arms": far, "prediction": prediction_far,
                                     "prediction_of_the_eighth_reply": prediction_far8},
              "constants": {"loss_drop": LOSS_DROP, "offsets_1_2_floor": NEAR_FLOOR},
              "settings": settings, "prediction_1A": prediction_1a, "prediction_1B": prediction_1b,
              "textbook_without_mask_lost_with_mask": lost, "inside_the_window": inside, "runs": rows}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    for setting, blk in settings.items():
        m = blk["medians"]
        if m.get("band_l0_at_300") is None:
            continue
        print(f"== {setting}: at step 300 band_l0 {m['band_l0_at_300']:.3f}, offsets 1-2 "
              f"{m['offsets_1_2_l0_at_300']:.4f}, loss drop {m['loss_drop_at_300']:.3f}; t_crit_loss "
              f"{m['t_crit_loss']}, t_crit_attn {m['t_crit_attn']}")
        print("   measured (medians):", json.dumps(m["measured"]))
        for name, r in blk["unperturbed_runs"].items():
            print(f"     {name}: band_l0(300) {r['band_l0_at_300']:.3f} near {r['offsets_1_2_l0_at_300']:.4f} drop "
                  f"{r['loss_drop_at_300']:.3f} t_crit loss/attn {r['t_crit_loss']}/{r['t_crit_attn']} | "
                  + json.dumps(r["measured"]))
        for arm, o in blk["arms"].items():
            print(f"   {arm}: textbook {o['textbook']}/{o['of']}, relays {o['relays']}, not formed {o['not_formed']}; "
                  f"δ {o['relay_deltas']}; decoy {[round(x, 2) for x in o['deep_decoy_over_clean']]}; "
                  f"d2 {[round(x, 2) for x in o['d2_l0_max']]}; first<2 {o['first_step_with_loss_rep_below_2']}")
    print("step-300 criterion (source attention of layer 1 >= threshold) by setting:",
          json.dumps({k: v.get("step_300_source_criterion") for k, v in settings.items()}))
    print("textbook without the mask, not textbook with it:", json.dumps(lost))
    print("prediction 1A:", json.dumps(prediction_1a))
    print("prediction 1B:", json.dumps(prediction_1b))
    for arm, blk in inside.items():
        print(f"== inside, {arm} (medians): step / latent band_l0 / band_l1 / source_l1 / d1 max / d2 max / loss")
        for i, s in enumerate(blk["step"]):
            print(f"   {s:>5d} {blk['band_l0'][i]:.2f} {blk['band_l1'][i]:.2f} {blk['source_l1'][i]:.3f} "
                  f"{blk['d1_l0_max'][i]:.3f} {blk['d2_l0_max'][i]:.3f} {blk['loss_rep'][i]:.2f}")
    print("== unperturbed outcome by learning rate and batch (d_model 64)")
    for setting, c in sorted(noise.items(), key=lambda kv: kv[1]["lr_over_root_batch"]):
        print(f"   {setting:>14s}: lr {c['learning_rate']:.1e} batch {c['batch']:>3d} lr/sqrt(B) "
              f"{c['lr_over_root_batch']:.2e} lr/B {c['lr_over_batch']:.2e} lr*sqrt(B) {c['lr_times_root_batch']:.2e}"
              f" | textbook {c['textbook']}/{c['of']} relays {c['relays']} not formed {c['not_formed']} "
              f"({c['steps']} steps) | loss drop at 300 {c['loss_drop_at_300']} | loss<2 at "
              f"{c['loss_rep_falls_below_2']}")
    print("   prediction:", json.dumps({k: v for k, v in noise_prediction.items() if not k.startswith("runs_")}))
    for name, v in half_way.items():
        print(f"   lr 2.5e-4, batch 8, {name} at step 12,000: loss {v['loss_rep']:.2f} source {v['source_l1']:.2f} "
              f"d1 {v['d1_l0_max']:.2f} d2 {v['d2_l0_max']:.2f}")
    for arm, blk in leak.items():
        if "step" not in blk:
            continue
        print(f"== {arm} (mask on {blk['masked_offsets']}, steps {blk['window']}; medians): step / band_l0 / just "
              "below / below / low band / beyond / band_l1 / source_l1 / d1 / d2 / loss")
        for i, st_ in enumerate(blk["step"]):
            print(f"   {st_:>5d} {blk['band_l0'][i]:.2f} {blk['just_below_band_l0'][i]:.3f} "
                  f"{blk['below_band_l0'][i]:.3f} {blk['low_band_l0'][i]:.2f} {blk['beyond_band_l0'][i]:.3f} "
                  f"{blk['band_l1'][i]:.2f} {blk['source_l1'][i]:.3f} {blk['d1_l0_max'][i]:.3f} "
                  f"{blk['d2_l0_max'][i]:.3f} {blk['loss_rep'][i]:.2f}")
    print("== unperturbed outcome at rep_min 16 by learning rate and batch")
    for setting, c in sorted(noise16.items(), key=lambda kv: kv[1]["lr_over_root_batch"]):
        print(f"   {setting:>18s}: lr {c['learning_rate']:.1e} batch {c['batch']:>3d} lr/sqrt(B) "
              f"{c['lr_over_root_batch']:.2e} | textbook {c['textbook']}/{c['of']} "
              f"(2 heads {c['by_heads']['h2']['textbook']}, 4 heads {c['by_heads']['h4']['textbook']}) | relay "
              f"positions {c.get('relay_deltas')} | loss<2 at {c.get('first_step_with_loss_rep_below_2')}")
    print("   prediction:", json.dumps(prediction16))
    for rep_min, cells, pred in ((8, noise8, prediction8), (20, noise20, prediction20), (4, noise4, prediction4),
                                 (24, noise24, prediction24)):
        print(f"== unperturbed outcome at rep_min {rep_min} by learning rate and batch")
        for setting, c in sorted(cells.items(), key=lambda kv: kv[1]["lr_over_root_batch"]):
            print(f"   {setting:>18s}: lr {c['learning_rate']:.1e} batch {c['batch']:>3d} lr/sqrt(B) "
                  f"{c['lr_over_root_batch']:.2e} | textbook {c['textbook']}/{c['of']} "
                  f"(2 heads {c['by_heads']['h2']['textbook']}, 4 heads {c['by_heads']['h4']['textbook']}) | relay "
                  f"positions {c.get('relay_deltas')} | not formed {c.get('not_formed')} | loss<2 at "
                  f"{c.get('first_step_with_loss_rep_below_2')}")
        print("   prediction:", json.dumps(pred))
    print("== windows with the wider mask")
    for arm, o in far.items():
        print(f"   {arm:>20s} {o['window']}: textbook {o['textbook']}/{o['of']} (2 heads "
              f"{o['by_heads']['h2']['textbook']}, 4 heads {o['by_heads']['h4']['textbook']}) | relay positions "
              f"{o['relay_deltas']} | decoy {[round(x, 2) for x in o['deep_decoy_over_clean']]} | loss<2 at "
              f"{o['first_step_with_loss_rep_below_2']}")
    print("   prediction:", json.dumps(prediction_far))
    print("   prediction of the eighth reply:", json.dumps(prediction_far8))
    print("missing:", len(result["missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
