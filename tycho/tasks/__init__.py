"""Survey objects: the task registry (SPEC §4.1-4.2).

``REGISTRY`` maps a task name to its :class:`~tycho.tasks.base.Task` subclass; :func:`make_task` builds one.
"""
from __future__ import annotations

from tycho.tasks.base import Task
from tycho.tasks.icl_markov import ICLMarkov
from tycho.tasks.induction import Induction
from tycho.tasks.markov_lang import MarkovLanguage
from tycho.tasks.modular_addition import ModularAddition
from tycho.tasks.sparse_parity import SparseParity
from tycho.tasks.toy_superposition import ToySuperposition

__all__ = ["REGISTRY", "Task", "make_task", "ICLMarkov", "Induction", "MarkovLanguage", "ModularAddition",
           "SparseParity", "ToySuperposition"]

REGISTRY: dict[str, type[Task]] = {cls.name: cls for cls in (SparseParity, MarkovLanguage, ModularAddition,
                                                             ICLMarkov, ToySuperposition, Induction)}


def make_task(name: str, cfg: dict | None, seed: int) -> Task:
    """Instantiate the task ``name`` with configuration ``cfg`` (merged over its defaults) and training seed.

    Raises:
        ValueError: for an unknown task name or invalid configuration.
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; known tasks: {sorted(REGISTRY)}") from None
    return cls(cfg, seed)
