"""Tests of the anomaly score (tycho.analysis.anomaly) and of scripts/set_config.py."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from tycho.analysis.anomaly import MIN_GROUP, THRESHOLD, anomaly_scores, summary_features

ROOT = Path(__file__).resolve().parents[1]


def _summary(**kw):
    s = {"t_test_acc90": 1000.0, "t_train_acc90": 500.0, "final_train_acc": 1.0, "final_test_acc": 0.95,
         "final_train_loss": 0.01, "final_test_loss": 0.1, "plateau_logfrac": 0.2, "n_drops": 1, "test_bump": 0.0,
         "steps_total": 10_000}
    s.update(kw)
    return s


def test_an_outlier_is_flagged_and_its_feature_named():
    rng = np.random.default_rng(1)
    group = [_summary(final_test_loss=float(0.1 * (1 + 0.05 * rng.standard_normal()))) for _ in range(30)]
    group[4] = _summary(final_test_loss=50.0)
    scores = anomaly_scores(group)
    flagged = [i for i, s in enumerate(scores) if s is not None and s[0] >= THRESHOLD]
    assert flagged == [4] and scores[4][1] == "log_final_test_loss"


def test_identical_peers_score_zero_and_small_groups_are_not_scored():
    assert all(s == (0.0, s[1]) for s in anomaly_scores([_summary() for _ in range(MIN_GROUP)]))
    assert anomaly_scores([_summary() for _ in range(MIN_GROUP - 1)]) == [None] * (MIN_GROUP - 1)


def test_never_reached_times_count_as_late_but_finite():
    f = summary_features(_summary(t_test_acc90=math.nan, steps_total=10_000))
    assert f["log_t_test_acc90"] == pytest.approx(math.log10(20_000))
    assert math.isnan(summary_features({})["final_test_acc"])


def test_missing_summaries_are_skipped():
    group = [_summary() for _ in range(MIN_GROUP)] + [None]
    assert anomaly_scores(group)[-1] is None


def _set_config():
    spec = importlib.util.spec_from_file_location("set_config", ROOT / "scripts" / "set_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_set_config_keeps_comments_and_validates(tmp_path):
    text = (ROOT / "configs" / "tycho.example.yaml").read_text(encoding="utf-8")
    (tmp_path / "tycho.yaml").write_text(text, encoding="utf-8")
    mod = _set_config()
    assert mod.main(["--root", str(tmp_path), "survey.sky=configs/sky_v1.yaml",
                     "scheduler.followups_per_day=10000"]) == 0
    new = (tmp_path / "tycho.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(new)
    assert data["survey"]["sky"] == "configs/sky_v1.yaml" and data["scheduler"]["followups_per_day"] == 10000
    assert new.count("#") == text.count("#")  # comments kept
    changed = [(a, b) for a, b in zip(text.splitlines(), new.splitlines()) if a != b]
    assert len(changed) == 2 and len(new.splitlines()) == len(text.splitlines())


def test_set_config_adds_missing_keys_and_restores_on_bad_values(tmp_path):
    (tmp_path / "tycho.yaml").write_text("survey:\n  n_workers: 4\n", encoding="utf-8")
    mod = _set_config()
    assert mod.main(["--root", str(tmp_path), "survey.torch_threads=1", "scheduler.extra_seeds=3"]) == 0
    data = yaml.safe_load((tmp_path / "tycho.yaml").read_text(encoding="utf-8"))
    assert data == {"survey": {"n_workers": 4, "torch_threads": 1}, "scheduler": {"extra_seeds": 3}}
    before = (tmp_path / "tycho.yaml").read_text(encoding="utf-8")
    assert mod.main(["--root", str(tmp_path), "survey.n_workers=[unclosed"]) == 1
    assert (tmp_path / "tycho.yaml").read_text(encoding="utf-8") == before
    assert mod.main(["--root", str(tmp_path), "nodot=1"]) == 2


def test_sky_v1_loads_and_every_point_builds_its_task():
    from tycho.survey.sky import Sky
    from tycho.tasks import make_task

    sky = Sky.from_yaml(ROOT / "configs" / "sky_v1.yaml")
    total = 0
    for region in sky.regions:
        points = sky.points(region)
        total += len(points) * region.seeds
        for point in points:
            spec = sky.make_spec(region, point, 0)
            make_task(spec.task, spec.task_cfg, spec.seed)
    assert total == 680
    assert {r.name for r in sky.regions if r.ri_policy == "followup"} == {"grokking-v1"}
