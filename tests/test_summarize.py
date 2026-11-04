"""Tests of tycho.analysis.summarize: exact definitions on hand-made curves, edge cases, and the sharp-drop detector."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tycho.analysis.summarize import (
    DROP_MAX_WIDTH,
    SUMMARY_KEYS,
    acc_threshold,
    find_drops,
    summarize,
)

LC_KEYS = ("step train_loss test_loss train_acc test_acc param_norm grad_norm update_norm dist_from_init "
           "erank_weights erank_hidden rep_cka_prev ntk_trace ntk_top_eig ntk_erank ntk_align ntk_velocity lr "
           "wall_s").split()


def eval_steps(total: int, n_log: int = 60, n_lin: int = 40) -> np.ndarray:
    """Log-spaced from 1 plus linear from 0, like tycho.train.schedule.eval_steps."""
    log = np.round(np.logspace(0, np.log10(total), n_log)).astype(int)
    lin = np.round(np.linspace(0, total, n_lin)).astype(int)
    return np.unique(np.concatenate([[0], log, lin, [total]]))


def make_lc(step, **cols) -> dict[str, np.ndarray]:
    """A light curve with every SPEC §4.4 key (NaN where not given)."""
    step = np.asarray(step)
    lc = {k: np.full(step.size, np.nan) for k in LC_KEYS}
    lc["step"] = step.astype(np.int64)
    lc.update({k: np.asarray(v, dtype=np.float64) for k, v in cols.items()})
    return lc


def logistic(z):
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=np.float64) / 2.0))


def same(a, b) -> bool:
    """Equality that treats NaN == NaN."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


# ---------------------------------------------------------------------------------------------- structure
def test_keys_types_and_json():
    st = eval_steps(1000)
    s = summarize(make_lc(st, train_loss=1.0 / (1 + st), test_loss=1.2 / (1 + st), train_acc=st / 1000,
                          test_acc=st / 1100), 0.1)
    assert tuple(s) == SUMMARY_KEYS
    for k in ("n_evals", "steps_total", "n_drops", "loss_spikes"):
        assert type(s[k]) is int, k
    assert type(s["diverged"]) is bool
    for k, v in s.items():
        assert type(v) in (int, float, bool, str), (k, type(v))
    assert json.loads(json.dumps(s))["steps_total"] == 1000
    assert s["n_evals"] == st.size and s["chance_acc"] == 0.1


@pytest.mark.parametrize("lc", [{}, {"step": np.zeros(0, dtype=np.int64)}, make_lc(np.zeros(0))])
def test_empty_light_curve(lc):
    s = summarize(lc, None)
    assert s["n_evals"] == 0 and s["steps_total"] == 0 and s["n_drops"] == 0 and s["loss_spikes"] == 0
    assert s["diverged"] is False
    assert math.isnan(s["final_train_loss"]) and math.isnan(s["plateau_frac"]) and math.isnan(s["chance_acc"])


def test_missing_and_all_nan_keys_give_nan():
    st = np.arange(0, 1001, 100)
    loss = 2.0 / (1 + st / 10.0)
    for lc in ({"step": st, "train_loss": loss},  # regression task: no accuracies, no test split given
               make_lc(st, train_loss=loss, test_loss=loss)):  # accuracies and eNTK present but all NaN
        s = summarize(lc, None)
        for k in ("final_train_acc", "max_train_acc", "t_train_acc90", "t_test_acc99", "grok_delay", "grok_ratio",
                  "test_acc_at_train99", "max_ntk_velocity", "step_max_ntk_velocity"):
            assert math.isnan(s[k]), k
        assert s["final_train_loss"] == pytest.approx(loss[-1])
        assert s["diverged"] is False


def test_malformed_input_raises():
    with pytest.raises(ValueError, match="train_loss"):
        summarize({"step": [0, 1, 2], "train_loss": [1.0, 0.5]}, None)
    with pytest.raises(ValueError, match="step"):
        summarize({"train_loss": [1.0, 0.5]}, None)
    with pytest.raises(ValueError, match="finite"):
        summarize({"step": [0.0, np.nan], "train_loss": [1.0, 0.5]}, None)


def test_unsorted_rows_and_duplicate_steps():
    rng = np.random.default_rng(3)
    st = eval_steps(5000)
    lc = make_lc(st, train_loss=0.01 + 0.69 * logistic((1500 - st) / 50.0), test_loss=0.7 / (1 + st / 300),
                 train_acc=0.5 + 0.5 * logistic((st - 1500) / 50.0), test_acc=0.5 + 0.4 * logistic((st - 1500) / 50.0),
                 ntk_velocity=rng.random(st.size))
    perm = rng.permutation(st.size)
    shuffled = {k: v[perm] for k, v in lc.items()}
    a, b = summarize(lc, 0.5), summarize(shuffled, 0.5)
    assert all(same(a[k], b[k]) for k in SUMMARY_KEYS)
    # a re-logged step: the last row wins
    s = summarize({"step": [0, 1, 1, 2], "train_loss": [1.0, 0.9, 0.8, 0.7], "train_acc": [0.0, 0.95, 0.2, 0.3]}, 0.1)
    assert s["n_evals"] == 3 and math.isnan(s["t_train_acc90"])


def test_input_not_mutated():
    st = np.array([3, 0, 2, 1])
    lc = {"step": st.copy(), "train_loss": np.array([0.1, 1.0, 0.2, 0.5]), "train_acc": np.array([1.0, 0, 0.9, 0.5])}
    snapshot = {k: v.copy() for k, v in lc.items()}
    summarize(lc, 0.1)
    for k, v in lc.items():
        np.testing.assert_array_equal(v, snapshot[k])


# ---------------------------------------------------------------------------------------------- accuracy times
@pytest.mark.parametrize("chance, level, expected", [
    (None, 50, 0.5), (0.1, 90, 0.9), (0.5, 99, 0.99), (0.6, 50, 0.8), (0.6, 90, 0.96), (0.8, 99, 0.998),
])
def test_acc_threshold(chance, level, expected):
    assert acc_threshold(level, chance) == pytest.approx(expected, abs=1e-12)


def test_crossing_times_and_grok_identities():
    st = np.array([0, 10, 100, 1000, 10000])
    lc = make_lc(st, train_acc=[0.1, 0.5, 0.9, 0.995, 1.0], test_acc=[0.1, 0.1, 0.2, 0.5, 0.95])
    s = summarize(lc, 0.1)
    assert (s["t_train_acc50"], s["t_train_acc90"], s["t_train_acc99"]) == (10.0, 100.0, 1000.0)  # >= : exact hits
    assert (s["t_test_acc50"], s["t_test_acc90"]) == (1000.0, 10000.0) and math.isnan(s["t_test_acc99"])
    assert s["grok_delay"] == s["t_test_acc90"] - s["t_train_acc99"] == 9000.0
    assert s["grok_ratio"] == s["t_test_acc90"] / s["t_train_acc99"] == 10.0
    assert s["test_acc_at_train99"] == 0.5
    assert s["max_train_acc"] == 1.0 and s["final_test_acc"] == 0.95


def test_rescaled_thresholds_above_half_chance():
    st = np.array([0, 10, 100, 1000])
    chance = 0.6  # thresholds 0.8, 0.96, 0.996 (exact hits must count despite float rounding of c + l(1-c))
    s = summarize(make_lc(st, train_acc=[0.6, 0.8, 0.96, 0.996]), chance)
    assert (s["t_train_acc50"], s["t_train_acc90"], s["t_train_acc99"]) == (10.0, 100.0, 1000.0)
    s_abs = summarize(make_lc(st, train_acc=[0.6, 0.8, 0.96, 0.996]), None)  # unknown chance: absolute thresholds
    assert (s_abs["t_train_acc50"], s_abs["t_train_acc90"], s_abs["t_train_acc99"]) == (0.0, 100.0, 1000.0)


def test_grok_ratio_guards_zero_and_negative_delay():
    s = summarize(make_lc([0, 100], train_acc=[1.0, 1.0], test_acc=[0.0, 0.95]), 0.0)
    assert s["t_train_acc99"] == 0.0 and s["grok_ratio"] == 100.0  # t90 / max(t99, 1)
    s = summarize(make_lc([0, 10, 100], train_acc=[0.0, 0.5, 1.0], test_acc=[0.0, 0.95, 0.95]), 0.0)
    assert s["grok_delay"] == -90.0 and s["grok_ratio"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------------------------- loss statistics
def test_final_values_and_minima():
    s = summarize(make_lc([0, 1, 2], train_loss=[1.0, 0.2, 0.3], test_loss=[1.1, 0.4, np.nan]), None)
    assert s["final_train_loss"] == 0.3 and s["min_train_loss"] == 0.2 and s["initial_train_loss"] == 1.0
    assert math.isnan(s["final_test_loss"]) and s["min_test_loss"] == 0.4  # final = value at the last evaluation
    assert s["train_loss_decrease"] == pytest.approx(0.8)


def test_plateau_frac_on_a_step_function():
    st = np.arange(0, 1001, 50)
    s = summarize(make_lc(st, train_loss=np.where(st < 300, 1.0, 0.1)), None)
    assert s["t_plateau_end"] == 300.0
    assert s["plateau_frac"] == pytest.approx(0.3)
    assert s["plateau_logfrac"] == pytest.approx(math.log(301) / math.log(1001))


def test_plateau_of_a_noisy_flat_curve_lasts_the_whole_run():
    rng = np.random.default_rng(0)
    st = eval_steps(20000)
    s = summarize(make_lc(st, train_loss=0.6931 + 0.0005 * rng.standard_normal(st.size)), None)
    assert s["plateau_frac"] == 1.0 and s["plateau_logfrac"] == 1.0 and math.isnan(s["t_plateau_end"])
    assert s["n_drops"] == 0 and s["loss_spikes"] == 0


@pytest.mark.parametrize("rel_noise", [0.003, 0.01, 0.03])
def test_noisy_flat_curve_keeps_its_plateau(rel_noise):
    """A single noisy evaluation (including the first one) must not end the plateau of a run that never learns."""
    st = eval_steps(20000)
    for seed in range(40):
        rng = np.random.default_rng(seed)
        loss = 0.6931 * (1 + rel_noise * rng.standard_normal(st.size))
        assert summarize(make_lc(st, train_loss=loss), None)["plateau_frac"] == 1.0
        loss[0] = 0.6931 * (1 + 5 * rel_noise)  # an outlying first evaluation
        assert summarize(make_lc(st, train_loss=loss), None)["plateau_frac"] == 1.0


def test_loss_spikes_count_after_the_first_drop_only():
    st = eval_steps(20000)
    loss = 0.002 + 0.691 * logistic((3000 - st) / 100)
    late = np.flatnonzero(st > 6000)[[3, 10, 20]]
    loss[late] *= 3.0  # three spikes above 2x the running minimum
    s = summarize(make_lc(st, train_loss=loss), None)
    assert s["t_plateau_end"] <= 3100 and s["loss_spikes"] == 3
    early = np.flatnonzero((st > 100) & (st < 2000))[5]
    loss[early] = 2.5  # a rise on the plateau, before the first drop: not counted
    assert summarize(make_lc(st, train_loss=loss), None)["loss_spikes"] == 3


def test_loss_spike_rise_floor_ignores_negligible_losses():
    loss = np.concatenate([np.geomspace(3.0, 1e-5, 20), [3e-5, 1e-5, 3e-5, 1e-5]])
    s = summarize(make_lc(np.arange(loss.size), train_loss=loss), None)
    assert s["loss_spikes"] == 0
    loss[-1] = 0.5  # a real spike
    assert summarize(make_lc(np.arange(loss.size), train_loss=loss), None)["loss_spikes"] == 1


def test_test_bump_definition():
    s = summarize(make_lc([0, 1, 2, 3], test_loss=[1.0, 0.5, 0.8, 0.3]), None)
    assert s["test_bump"] == pytest.approx(0.3) and s["test_bump_step"] == 2.0
    assert s["test_bump_recovery"] == pytest.approx(0.5 / 0.3) and s["test_first_descent"] == pytest.approx(0.5)
    s = summarize(make_lc([0, 1, 2, 3], test_loss=[2.0, 1.0, 1.5, 0.5]), None)
    assert s["test_bump"] == pytest.approx(0.25)  # normalised by test_loss(0)
    s = summarize(make_lc([0, 1, 2], test_loss=[1.0, 0.5, 0.9]), None)
    assert s["test_bump"] == pytest.approx(0.4) and math.isnan(s["test_bump_recovery"])  # peak is the last point
    s = summarize(make_lc([0, 1, 2], test_loss=[1.0, 0.8, 0.5]), None)
    assert s["test_bump"] == 0.0 and math.isnan(s["test_bump_step"]) and math.isnan(s["test_first_descent"])
    assert math.isnan(summarize(make_lc([0], test_loss=[1.0]), None)["test_bump"])


@pytest.mark.parametrize("train, test, expected", [
    ([1.0, 0.5, np.nan], [1.0, 0.5, 0.4], True),  # NaN after finite values
    ([1.0, 0.5, 0.4], [1.0, np.inf, 0.4], True),  # infinite test loss
    ([1.0, 0.5, 20.0], [1.0, 0.5, 0.4], True),  # finite explosion beyond 10x the initial loss
    ([1.0, 0.5, 9.0], [1.0, 0.5, 0.4], False),
    ([np.nan, 1.0, 0.5], [1.0, 0.5, 0.4], False),  # NaN before any finite value is not divergence
    ([np.nan, np.nan, np.nan], [1.0, 0.5, 0.4], False),  # train loss not measured
    ([1.0, 0.5, 0.4], [1.0, 0.5, 0.4], False),
])
def test_diverged(train, test, expected):
    assert summarize(make_lc([0, 1, 2], train_loss=train, test_loss=test), None)["diverged"] is expected


def test_max_ntk_velocity():
    s = summarize(make_lc([0, 1, 2, 3, 4], ntk_velocity=[np.nan, 0.1, 0.5, 0.2, 0.5]), None)
    assert s["max_ntk_velocity"] == 0.5 and s["step_max_ntk_velocity"] == 2.0  # first step attaining the max


# ---------------------------------------------------------------------------------------------- sharp drops
ST = eval_steps(20000).astype(np.float64)


@pytest.mark.parametrize("alpha", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("floor", [0.0, 0.05])
def test_power_laws_have_no_drops(alpha, floor):
    assert find_drops(ST, floor + 2.0 * (1 + ST / 20.0) ** -alpha) == []


@pytest.mark.parametrize("tau", [10.0, 100.0, 1000.0])
@pytest.mark.parametrize("floor", [0.0, 0.01, 0.1])
def test_exponential_relaxations_are_not_sharp(tau, floor):
    assert find_drops(ST, floor + (1.0 - floor) * np.exp(-ST / tau)) == []


@pytest.mark.parametrize("t_star", [300.0, 6000.0])
@pytest.mark.parametrize("rel_width", [0.02, 0.1])
def test_logistic_cliff_is_one_drop(t_star, rel_width):
    loss = 0.002 + 0.691 * logistic((t_star - ST) / (rel_width * t_star))
    (drop,) = find_drops(ST, loss)
    assert drop.step_lo <= t_star <= drop.step_hi
    assert drop.step_lo in ST and drop.step_hi in ST
    assert drop.width <= DROP_MAX_WIDTH and drop.pre_frac < 0.1
    total = math.log((loss[0] + 1e-3 * loss[0]) / (loss[-1] + 1e-3 * loss[0]))
    assert 0.8 * total < drop.depth <= total + 1e-9
    assert drop.steepness > 5.0  # a cliff in log-log coordinates


def test_two_separated_drops_and_pre_frac():
    loss = 0.60 + 0.35 * logistic((4000 - ST) / 200) + 0.15 * logistic((300 - ST) / 20)
    first, second = find_drops(ST, loss)
    assert first.step_lo <= 300 <= first.step_hi and second.step_lo <= 4000 <= second.step_hi
    frac_first = math.log(1.10 / 0.95) / math.log(1.10 / 0.60)
    assert second.pre_frac == pytest.approx(frac_first, abs=0.05)


def test_drops_closer_than_a_factor_two_merge():
    loss = 0.01 + 0.4 * logistic((2000 - ST) / 30) + 0.4 * logistic((2400 - ST) / 30)
    assert len(find_drops(ST, loss)) == 1


@pytest.mark.parametrize("background", [
    lambda t: 0.05 + 2.0 * (1 + t / 20) ** -0.6,
    lambda t: 2.0 * (1 + t / 500) ** -1.0,  # steep: without background subtraction this drop merges into it
])
def test_drop_on_a_power_law_background(background):
    loss = background(ST) * (1 - 0.6 * logistic((ST - 3000) / 50))
    (drop,) = find_drops(ST, loss)
    assert drop.step_lo <= 3000 <= drop.step_hi and drop.width < 0.6


def test_recovery_from_a_spike_is_not_a_drop():
    loss = (0.05 + 2.0 * (1 + ST / 20) ** -0.6) * (1 + 20 * np.exp(-0.5 * ((ST - 5000) / 300) ** 2))
    assert find_drops(ST, loss) == []


def test_drops_are_invariant_to_loss_scale_and_robust_to_resampling():
    def curve(t):
        return 0.002 + 0.691 * logistic((6000 - t) / 150)

    (base,) = find_drops(ST, curve(ST))
    (scaled,) = find_drops(ST, 7.3 * curve(ST))
    assert (scaled.step, scaled.step_lo, scaled.step_hi) == (base.step, base.step_lo, base.step_hi)
    assert (scaled.depth, scaled.width, scaled.pre_frac) == pytest.approx((base.depth, base.width, base.pre_frac))
    dense = eval_steps(20000, n_log=120, n_lin=200).astype(np.float64)
    (d,) = find_drops(dense, curve(dense))
    assert abs(math.log(d.step / base.step)) < 0.1


@pytest.mark.parametrize("seed", range(8))
def test_drop_counts_under_one_percent_noise(seed):
    rng = np.random.default_rng(seed)

    def noisy(loss):
        return loss * np.exp(0.01 * rng.standard_normal(loss.size))

    assert find_drops(ST, noisy(np.full(ST.size, 0.6931))) == []
    assert find_drops(ST, noisy(0.05 + 2.0 * (1 + ST / 20) ** -0.6)) == []
    assert len(find_drops(ST, noisy(0.002 + 0.691 * logistic((6000 - ST) / 150)))) == 1
    assert len(find_drops(ST, noisy(0.60 + 0.35 * logistic((4000 - ST) / 200) + 0.15 * logistic((300 - ST) / 20)))) == 2


def test_find_drops_degenerate_inputs():
    assert find_drops([0, 1, 2, 3], [1.0, 1.0, 0.1, 0.1]) == []  # fewer than DROP_MIN_POINTS evaluations
    assert find_drops(ST, np.zeros(ST.size)) == []
    loss = 0.691 * logistic((6000 - ST) / 150) - 0.001  # slightly negative tail is clipped, not an error
    assert len(find_drops(ST, loss)) == 1
    with pytest.raises(ValueError):
        find_drops([0, 1, 2], [1.0, 0.5])


def test_summary_reports_the_largest_drop_bracket():
    loss = 0.002 + 0.691 * logistic((6000 - ST) / 150)
    s = summarize(make_lc(ST, train_loss=loss), 0.5)
    assert s["n_drops"] == 1 and s["first_drop_step"] == s["largest_drop_step"]
    assert s["largest_drop_lo"] <= 6000 <= s["largest_drop_hi"]
    assert s["largest_drop_lo"] in ST and s["largest_drop_hi"] in ST
    assert s["largest_drop_depth"] > 4.0 and s["drop_width_max"] <= DROP_MAX_WIDTH
    assert s["first_drop_pre_frac"] < 0.1
    (drop,) = find_drops(ST, loss)
    d = json.loads(json.dumps(drop.to_dict()))
    assert (d["step"], d["step_lo"], d["step_hi"]) == (s["largest_drop_step"], s["largest_drop_lo"],
                                                       s["largest_drop_hi"])
