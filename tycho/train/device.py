"""Device choice and global torch configuration for workers.

The survey's objects are tiny: their step time is dominated by Python and dispatcher overhead, not FLOPs, so one
single-threaded CPU worker per core is usually the fastest configuration and the only bit-for-bit reproducible
one. Apple's MPS backend is used only when asked for (``"mps"``) or, under ``"auto"``, for models large enough to
benefit (:data:`MPS_MIN_PARAMS`). **Exact replay determinism is only guaranteed on CPU** (for a fixed thread
count); MPS kernels make no such promise. MPS is never required: when it is unavailable the CPU is used.
"""
from __future__ import annotations

import logging

import torch

__all__ = ["DEVICE_POLICIES", "MPS_MIN_PARAMS", "configure_torch", "mps_available", "pick_device"]

logger = logging.getLogger(__name__)

#: Accepted device policies.
DEVICE_POLICIES = ("cpu", "mps", "auto")
#: Under the "auto" policy, models with at least this many parameters run on MPS (when available).
MPS_MIN_PARAMS = 2_000_000


def mps_available() -> bool:
    """True when the MPS backend can be used in this process."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def pick_device(policy: str, n_params: int) -> torch.device:
    """Device for a model with ``n_params`` parameters under ``policy``.

    ``"cpu"``: always CPU. ``"mps"``: MPS when available, else CPU (logged). ``"auto"``: MPS only when available
    and ``n_params >= MPS_MIN_PARAMS``, else CPU.

    Raises:
        ValueError: on an unknown policy.
    """
    policy = str(policy).lower()
    if policy not in DEVICE_POLICIES:
        raise ValueError(f"device policy must be one of {DEVICE_POLICIES}, got {policy!r}")
    if policy == "cpu":
        return torch.device("cpu")
    if not mps_available():
        if policy == "mps":
            logger.info("MPS requested but not available; using the CPU")
        return torch.device("cpu")
    if policy == "mps" or n_params >= MPS_MIN_PARAMS:
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch(threads: int, deterministic: bool = True) -> None:
    """Process-wide torch settings for a worker: intra-op (and, if still possible, inter-op) thread counts, and
    deterministic algorithms.

    ``deterministic`` calls ``torch.use_deterministic_algorithms(True, warn_only=True)``: operations without a
    deterministic implementation warn instead of failing (relevant on MPS; CPU kernels used here are
    deterministic). Results are reproducible bit-for-bit only for a fixed thread count, since reductions are split
    across threads.
    """
    threads = max(1, int(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:  # only settable once, before any inter-op parallel work has started
        logger.debug("inter-op thread count already fixed at %d", torch.get_num_interop_threads())
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
