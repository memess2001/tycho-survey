"""Instrument calibration (SPEC §3, "build Tycho's instrument before the survey").

* :mod:`~tycho.calibration.planted` — distributions with planted, exactly known RI structure (spin-coded binary
  tables, target-RI planting, samplers, switch-on schedules, dichotomized Gaussians).
* :mod:`~tycho.calibration.suite` — :class:`CalibrationConfig` and :func:`run_calibration`, experiments E1-E7.
* :mod:`~tycho.calibration.report` — :func:`write_report` (``CALIBRATION.md``, figures, ``calibration.json``) and
  :func:`recommended_min_effect`, the lookup the survey uses to set ``min_effect``.

Typical use::

    from tycho.calibration import CalibrationConfig, run_calibration, write_report
    results = run_calibration(CalibrationConfig.full(), out_dir, n_jobs=10)   # resumable; see suite docstring
    write_report(results, out_dir)
"""
from tycho.calibration.planted import (
    dichotomized_gaussian_table,
    equicorrelation,
    gaussian_binarized,
    latent_threshold,
    planted_binary,
    planted_binary_batch,
    random_interactions,
    sample_table,
    spin_features,
    switch_on,
    theta_for_local_ri,
    true_local_ri,
)
from tycho.calibration.report import (
    calibration_document,
    load_calibration,
    recommended_min_effect,
    write_report,
)
from tycho.calibration.suite import (
    EXPERIMENTS,
    CalibrationConfig,
    interpolate_min_effect,
    load_results,
    run_calibration,
)

__all__ = [
    # planted (SPEC §3)
    "planted_binary",
    "random_interactions",
    "sample_table",
    "true_local_ri",
    "switch_on",
    "gaussian_binarized",
    # planted (extras)
    "planted_binary_batch",
    "spin_features",
    "theta_for_local_ri",
    "equicorrelation",
    "latent_threshold",
    "dichotomized_gaussian_table",
    # suite
    "EXPERIMENTS",
    "CalibrationConfig",
    "run_calibration",
    "load_results",
    "interpolate_min_effect",
    # report
    "write_report",
    "calibration_document",
    "load_calibration",
    "recommended_min_effect",
]
