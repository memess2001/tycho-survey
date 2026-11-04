"""A tiny survey root for the runtime tests (worker, follow-up, daemon, CLI, release, dashboard).

Two small regions: ``ctrl`` (sparse parity, ``ri_policy: always``, in-situ RI) and ``grok`` (modular addition,
``ri_policy: followup``), each with 2 grid points x 2 seeds of a few hundred steps (well under a second per object).
The config makes every periodic daemon task fast and disables push channels and the disk guard threshold, so the
tests do not depend on the machine's free space or network.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from tycho.config import Paths, deep_merge, load_config

PARITY_TRAIN = {"steps": 300, "batch_size": 16, "optimizer": "sgd", "momentum": 0.0, "lr": 0.05, "n_log_evals": 12,
                "n_lin_evals": 6, "ntk_every": 3, "ntk_probe": 8}
MODADD_TRAIN = {"steps": 300, "batch_size": 32, "n_log_evals": 12, "n_lin_evals": 6, "ntk_every": 0}

TINY_SKY: dict[str, Any] = {
    "version": "tiny-v0",
    "description": "test sky",
    "regions": [
        {"name": "ctrl", "task": "sparse_parity", "seeds": 2, "priority": 2.0, "ri_policy": "always",
         "ri": {"n_probe": 512, "n_points": 6}, "fixed": {"n": 8, "k": 2, "test_size": 256}, "train": PARITY_TRAIN,
         "knobs": {"width": {"values": [16, 32], "kind": "log"}}},
        {"name": "grok", "task": "modular_addition", "seeds": 2, "priority": 1.0, "ri_policy": "followup",
         "ri": {"n_probe": 512, "n_points": 6}, "fixed": {"p": 7}, "train": MODADD_TRAIN,
         "knobs": {"width": {"values": [16, 32], "kind": "log"}}},
    ],
}

TINY_CONFIG: dict[str, Any] = {
    "survey": {"sky": "configs/tiny.yaml", "n_workers": 2, "heartbeat_s": 0.5, "stale_after_s": 60.0,
               "topup_every_s": 1.0, "followup_every_s": 1.0, "status_every_s": 1.0, "min_free_gb": 0.0,
               "code_version": "test"},
    "scheduler": {"target_queue": 8},
    "ri": {"n_probe": 512},
    "notify": {"channels": {"ntfy": {"enabled": False}, "webhook": {"enabled": False},
                            "macos": {"enabled": False}}},
}


def make_root(root: Path, *, config: dict | None = None, sky: dict | None = None) -> Paths:
    """Write the tiny sky and ``tycho.yaml`` (``config`` merged over :data:`TINY_CONFIG`) into ``root``."""
    paths = Paths(root).ensure()
    (paths.configs / "tiny.yaml").write_text(yaml.safe_dump(copy.deepcopy(sky or TINY_SKY), sort_keys=False),
                                             encoding="utf-8")
    cfg = deep_merge(TINY_CONFIG, config or {})
    (paths.root / "tycho.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return paths


def tiny_config(paths: Paths) -> dict:
    """The effective config of a root made by :func:`make_root`."""
    return load_config(paths.root)


# ---------------------------------------------------------------------------------------------- synthetic catalog
def _light_curve(final_acc: float, seed: int) -> dict:
    """A small hand-made light curve with a sharp loss drop (numpy only, no training)."""
    import numpy as np

    step = np.array([0, 1, 3, 10, 30, 100, 150, 200, 300], dtype=np.int64)
    loss = np.array([0.70, 0.70, 0.69, 0.69, 0.68, 0.40, 0.10, 0.05, 0.04]) + 0.001 * seed
    acc = np.minimum(final_acc, np.array([0.5, 0.5, 0.5, 0.5, 0.52, 0.8, 0.97, 0.99, 1.0]))
    return {"step": step, "train_loss": loss, "test_loss": loss * 1.1, "train_acc": acc, "test_acc": acc * 0.98,
            "wall_s": np.linspace(0.0, 1.0, step.size)}


def synthetic_catalog(paths: Paths, *, clock=None):
    """A catalog of the tiny sky's 8 objects without training: 5 done (light curves, summaries, rule
    classifications), 1 failed, 2 queued; one in-situ RI run (curve file) with one event; two transients (one tied
    to the event, one photometric); a heartbeat of worker ``w0``. Returns ``(catalog, done_object_ids)``."""
    import numpy as np

    from tycho.analysis.classify import classify
    from tycho.analysis.events import RIEvent, Transient
    from tycho.analysis.ricurve import RICurve
    from tycho.analysis.summarize import summarize
    from tycho.survey.catalog import Catalog
    from tycho.survey.sky import Sky
    from tycho.survey.worker import save_light_curve
    from tycho.train.spec import RunResult

    sky = Sky.from_dict(TINY_SKY)
    kwargs = {} if clock is None else {"clock": clock}
    catalog = Catalog(paths.db, root=paths.root, **kwargs)
    for region in sky.regions:
        specs = [sky.make_spec(region, p, seed) for p in sky.points(region) for seed in range(region.seeds)]
        catalog.add_objects(specs, region=region.name, source="coarse", priority=region.priority)
    claimed = [catalog.claim_next("w0") for _ in range(6)]
    done = []
    for i, spec in enumerate(claimed[:5]):
        oid = spec.object_id()
        lc = _light_curve(1.0 if i % 2 == 0 else 0.6, i)
        summary = summarize(lc, 0.5)
        lc_path = save_light_curve(paths.lightcurve_path(oid), lc)
        catalog.mark_done(RunResult(object_id=oid, status="done", light_curve=lc, summary=summary,
                                    wall_time_s=1.5 + i, device="cpu", n_params=100 + i), lc_path,
                          code_version="test")
        catalog.add_classification(oid, classify(summary), classifier="rules-v1")
        done.append(oid)
    catalog.mark_failed(claimed[5].object_id(), "Traceback: synthetic failure", worker_id="w0")

    oid = done[0]
    curve = RICurve(steps=[0, 100, 300], names=["x0", "x1", "yhat"], subsets=[(0, 2), (1, 2), (0, 1, 2)],
                    ri=np.array([[0.0, 0.001, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.6]]),
                    ri_plugin=np.array([[0.001, 0.002, 0.001], [0.0, 0.001, 0.0], [0.001, 0.5, 0.6]]),
                    p=np.array([[0.5, 0.2, 0.6], [0.9, 0.8, 0.7], [0.4, 1e-9, 1e-9]]),
                    q=np.array([[0.5, 0.3, 0.6], [0.9, 0.8, 0.7], [0.4, 1e-8, 1e-8]]), n_eff=[512, 512, 512],
                    ground_truth=[{"subset": ["x0", "x1", "yhat"], "order": 3}],
                    meta={"alpha": 0.01, "min_effect": 0.001, "max_order": 3, "kind": "insitu", "object_id": oid})
    ri_path = curve.save(paths.ri_path(oid, "insitu"))
    run_id = catalog.add_ri_run(oid, "insitu", ri_path, curve)
    event = RIEvent(subset=(0, 1, 2), subset_names=("x0", "x1", "yhat"), order=3, kind="activation", step=100,
                    step_lo=0, step_hi=100, ri_before=0.0, ri_after=0.55, peak_ri=0.6, confidence=0.99,
                    has_precursor=False, matches_ground_truth=True)
    event_ids = catalog.add_events(run_id, oid, [event])
    catalog.add_transients(oid, [Transient("pure_synergy", 3, "pure synergy of <x0, x1, yhat> & friends", 0),
                                 Transient("loss_spike", 1, "1 loss spike(s)", None)], event_ids)
    catalog.heartbeat("w0", None, "idle")
    return catalog, done
