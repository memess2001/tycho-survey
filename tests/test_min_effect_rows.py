"""min_effect follows the scan's actual number of probe rows when it comes from a calibration file (sky v1 fix).

Modular addition probes all p^2 input pairs, whatever ``n_probe`` says; in sky v0 its scans used the threshold
calibrated for n_probe = 4096 although they had 529 (p = 23) or 2209 (p = 47) rows.
"""
from __future__ import annotations

import json

import pytest
import torch

from tycho.calibration.report import recommended_min_effect
from tycho.survey.followup import _RI_FIELDS, resolve_ri_config
from tycho.train import ObjectSpec, RIConfig, TrainConfig, run_object


def _calibration(tmp_path, noise_at_512: float = 0.004):
    table = [{"order": 2, "n": 512, "noise": noise_at_512, "floor": 1e-3, "value": noise_at_512, "source": "E1"},
             {"order": 2, "n": 4096, "noise": noise_at_512 / 8, "floor": 1e-3, "value": 1e-3, "source": "E1"}]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"format": "tycho-calibration/1",
                                "recommended_min_effect": {"rule": "test", "table": table}}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_config_files_cannot_set_the_calibration_path():
    assert "min_effect_calibration" not in _RI_FIELDS
    assert "min_effect" in _RI_FIELDS


def test_resolve_records_the_calibration_only_when_used(tmp_path):
    cal = _calibration(tmp_path)
    cfg = {"ri": {"n_probe": 4096, "calibration_file": str(cal)}}
    ri = resolve_ri_config(cfg, None, root=tmp_path)
    assert ri.min_effect == pytest.approx(1e-3) and ri.min_effect_calibration == str(cal)
    explicit = resolve_ri_config({"ri": {"n_probe": 4096, "calibration_file": str(cal), "min_effect": 0.02}}, None,
                                 root=tmp_path)
    assert explicit.min_effect == pytest.approx(0.02) and explicit.min_effect_calibration is None


def _modadd(p: int) -> ObjectSpec:
    return ObjectSpec(task="modular_addition", task_cfg={"p": p, "train_frac": 0.5, "width": 16, "ri_units": 4},
                      seed=0, train=TrainConfig(steps=40, batch_size=64, optimizer="adamw", lr=1e-3, n_log_evals=5,
                                                n_lin_evals=0, ntk_every=0))


def test_all_pairs_scan_uses_the_threshold_at_its_row_count(tmp_path):
    cal = _calibration(tmp_path)
    ri = RIConfig(enabled=True, n_probe=4096, n_points=3, min_effect=1e-3, min_effect_calibration=str(cal))
    res = run_object(_modadd(23), work_dir=tmp_path / "w", ri=ri)
    assert res.status == "done"
    meta = res.ri_curve.meta
    assert meta["min_effect_n"] == 23 * 23
    expected = recommended_min_effect(cal, 23 * 23)
    assert meta["min_effect"] == pytest.approx(expected)
    assert expected > 3e-3  # about 0.004 * 512 / 529, far above the 1e-3 calibrated for 4096 rows


def test_without_calibration_or_at_n_probe_nothing_changes(tmp_path):
    res = run_object(_modadd(23), work_dir=tmp_path / "a", ri=RIConfig(enabled=True, n_probe=4096, n_points=3,
                                                                     min_effect=2e-3))
    assert res.ri_curve.meta["min_effect"] == pytest.approx(2e-3)
    cal = _calibration(tmp_path)
    spec = ObjectSpec(task="sparse_parity", task_cfg={"n": 8, "k": 2, "width": 16, "test_size": 256}, seed=1,
                      train=TrainConfig(steps=40, batch_size=32, optimizer="sgd", lr=0.05, n_log_evals=5,
                                        n_lin_evals=0, ntk_every=0))
    res = run_object(spec, work_dir=tmp_path / "b", ri=RIConfig(enabled=True, n_probe=512, n_points=3,
                                                                min_effect=0.004, min_effect_calibration=str(cal)))
    assert res.ri_curve.meta["min_effect_n"] == 512
    assert res.ri_curve.meta["min_effect"] == pytest.approx(0.004)
