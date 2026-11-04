"""Training and observation of survey objects (SPEC §4.4).

Modules: :mod:`~tycho.train.spec` (contract types), :mod:`~tycho.train.device` (device policy, torch settings),
:mod:`~tycho.train.schedule` (evaluation, RI and learning-rate schedules), :mod:`~tycho.train.photometry` (light-curve
observables), :mod:`~tycho.train.runner` (:func:`run_object`), :mod:`~tycho.train.replay` (:func:`replay_object`).

The names below are re-exported lazily (PEP 562): ``import tycho.train.spec`` — which the catalog, scheduler and
agent need — does not import torch; ``from tycho.train import run_object`` imports the runner on first use.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "TrainConfig": "tycho.train.spec", "RIConfig": "tycho.train.spec", "ObjectSpec": "tycho.train.spec",
    "RunResult": "tycho.train.spec",
    "pick_device": "tycho.train.device", "configure_torch": "tycho.train.device",
    "eval_steps": "tycho.train.schedule", "subsample_log": "tycho.train.schedule", "lr_at": "tycho.train.schedule",
    "LIGHT_CURVE_KEYS": "tycho.train.photometry", "Photometer": "tycho.train.photometry",
    "effective_rank": "tycho.train.photometry", "linear_cka": "tycho.train.photometry",
    "ntk_gram": "tycho.train.photometry", "ntk_stats": "tycho.train.photometry",
    "param_norm": "tycho.train.photometry",
    "run_object": "tycho.train.runner", "LeanOptimizer": "tycho.train.runner",
    "replay_object": "tycho.train.replay", "replay_fidelity": "tycho.train.replay",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
