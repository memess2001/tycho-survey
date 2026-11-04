"""Observe one survey object: train it, record its light curve, optionally its RI curve, and resume exactly.

:func:`run_object` trains ``spec.task`` for ``spec.train.steps`` updates and stops at *checkpoint steps*: the
evaluation steps (:func:`~tycho.train.schedule.eval_steps`), where :class:`~tycho.train.photometry.Photometer`
records a light-curve row, and, when RI is enabled, the RI steps (default
``subsample_log(eval steps, ri.n_points)``), where ``task.ri_variables`` is scanned with
:func:`~tycho.ri.estimators.ri_scan` and appended to an :class:`~tycho.analysis.ricurve.RICurve`.

Randomness and determinism
--------------------------
* Initial weights: ``torch.manual_seed(spec.seed)`` inside ``task.build_model``.
* Batches: a dedicated CPU generator seeded from ``(spec.seed, "data")``.
* RI probes: at step t a fresh generator seeded from ``(spec.seed, "ri", t)`` (and a numpy generator from
  ``(spec.seed, "ri-scan", t)`` for any bootstrap). Probing never touches the training generator or the weights,
  so the training light curve is identical with RI on or off, and the probe at step t is the same whichever
  other steps are probed (an in-situ curve and a denser replay agree at shared steps).
* On CPU with a fixed thread count the light curve is bit-for-bit reproducible (except ``wall_s``);
  ``torch.use_deterministic_algorithms(True, warn_only=True)`` is enabled for the duration of a run. MPS runs are
  not guaranteed to be reproducible.

Training loop
-------------
Gradients come from ``torch.autograd.grad`` and are applied by :class:`LeanOptimizer`, which implements the update
rules of ``torch.optim`` SGD / Adam / AdamW (foreach variants) without ``.grad`` accumulation and ``Optimizer``
bookkeeping. For the survey's tiny models this removes about a third of the per-step time. Global-norm clipping
follows ``torch.nn.utils.clip_grad_norm_`` (``coef = clip / (norm + 1e-6)`` when below 1). Gradient and update
norms are computed only for the update that precedes an evaluation.

Divergence
----------
A non-finite minibatch loss stops training before that update is applied: a final row is measured at that step
(if it is not already an evaluation row), and the result has status "done" with ``summary["diverged"] = True``.
A non-finite evaluation train loss or non-finite parameters at a checkpoint also end the run this way.

Interruption and resume
-----------------------
``work_dir/resume.pt`` is written every ``train.checkpoint_every_s`` seconds (checked every :data:`POLL_STEPS`
updates and at checkpoint steps) and whenever ``stop_flag()`` returns true (checked at the same times); the run
then returns status "interrupted". With ``resume=True`` a later call restores the model, optimizer, data
generator, global torch RNG, photometer memory, partial light curve and partial RI curve, and continues exactly:
the final light curve and RI curve equal those of an uninterrupted run. A checkpoint whose fingerprint (object id,
package version, schedule, RI settings) differs from the current call, or that cannot be read, is ignored with a
warning; one that matches but cannot be applied (e.g. written by code with another architecture) is renamed to
``resume.pt.bad`` and the run starts from scratch. The file is deleted when the run completes. Exceptions give
status "failed" with the traceback in ``error`` (and leave the last checkpoint in place for a retry).
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

import tycho
from tycho.analysis.ricurve import RICurve
from tycho.analysis.summarize import summarize
from tycho.ri.estimators import ri_scan
from tycho.tasks import make_task
from tycho.tasks.base import numpy_rng, torch_generator
from tycho.train.device import pick_device
from tycho.train.photometry import LIGHT_CURVE_KEYS, Photometer
from tycho.train.schedule import eval_steps, lr_at, subsample_log
from tycho.train.spec import ObjectSpec, RIConfig, RunResult, TrainConfig

__all__ = ["ADAM_EPS", "CLIP_EPS", "OPTIMIZERS", "POLL_STEPS", "RESUME_FILE", "RESUME_FORMAT", "LeanOptimizer",
           "light_curve_from_rows", "run_object"]

logger = logging.getLogger(__name__)

#: Name of the resume checkpoint inside ``work_dir``.
RESUME_FILE = "resume.pt"
#: Version of the checkpoint layout (a checkpoint with another version is ignored).
RESUME_FORMAT = 1
#: Updates between checks of ``stop_flag`` and of the checkpoint timer.
POLL_STEPS = 64
#: Adam's ε (the torch.optim default).
ADAM_EPS = 1e-8
#: ε of global-norm gradient clipping (as in ``torch.nn.utils.clip_grad_norm_``).
CLIP_EPS = 1e-6
#: Supported optimizers (``TrainConfig.optimizer``).
OPTIMIZERS = ("sgd", "adam", "adamw")

_NAN = float("nan")


class LeanOptimizer:
    """SGD, Adam and AdamW with the update rules of ``torch.optim`` (foreach variants), on explicit gradients.

    * ``"sgd"``: ``g ← g + wd·θ``; with momentum μ > 0, ``b ← g`` at the first step and ``b ← μ b + g`` after,
      ``g ← b``; then ``θ ← θ − lr·g`` (no dampening, no Nesterov).
    * ``"adam"``: L2 weight decay ``g ← g + wd·θ``; ``m ← β₁ m + (1−β₁) g``, ``v ← β₂ v + (1−β₂) g²``,
      ``θ ← θ − lr/(1−β₁ᵗ) · m / (√v / √(1−β₂ᵗ) + ε)``.
    * ``"adamw"``: decoupled weight decay ``θ ← θ (1 − lr·wd)`` first, then the Adam step without L2 term.

    Hyper-parameters come from :class:`~tycho.train.spec.TrainConfig` (``momentum`` is used by SGD only,
    ``beta1``/``beta2`` by Adam/AdamW; ε = :data:`ADAM_EPS`). The learning rate is passed to every :meth:`step`.
    The gradients passed to :meth:`step` may be modified in place.
    """

    def __init__(self, params: Sequence[Tensor], train: TrainConfig) -> None:
        if train.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {train.optimizer!r}")
        self.params = list(params)
        self.kind = train.optimizer
        self.weight_decay = float(train.weight_decay)
        self.momentum = float(train.momentum)
        self.beta1, self.beta2 = float(train.beta1), float(train.beta2)
        self.t = 0
        self.momentum_buffer: list[Tensor] | None = None
        self.exp_avg: list[Tensor] | None = None
        self.exp_avg_sq: list[Tensor] | None = None
        if self.kind != "sgd":
            self.exp_avg = [torch.zeros_like(p) for p in self.params]
            self.exp_avg_sq = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, grads: Sequence[Tensor], lr: float) -> None:
        params = self.params
        grads = list(grads)
        self.t += 1
        if self.kind == "sgd":
            if self.weight_decay != 0.0:
                grads = torch._foreach_add(grads, params, alpha=self.weight_decay)
            if self.momentum != 0.0:
                if self.momentum_buffer is None:
                    self.momentum_buffer = [g.clone() for g in grads]
                else:
                    torch._foreach_mul_(self.momentum_buffer, self.momentum)
                    torch._foreach_add_(self.momentum_buffer, grads)
                grads = self.momentum_buffer
            torch._foreach_add_(params, grads, alpha=-lr)
            return
        if self.weight_decay != 0.0:
            if self.kind == "adamw":
                torch._foreach_mul_(params, 1.0 - lr * self.weight_decay)
            else:
                grads = torch._foreach_add(grads, params, alpha=self.weight_decay)
        torch._foreach_lerp_(self.exp_avg, grads, 1.0 - self.beta1)
        torch._foreach_mul_(self.exp_avg_sq, self.beta2)
        torch._foreach_addcmul_(self.exp_avg_sq, grads, grads, 1.0 - self.beta2)
        step_size = (lr / (1.0 - self.beta1 ** self.t)) * -1.0
        denom = torch._foreach_sqrt(self.exp_avg_sq)
        torch._foreach_div_(denom, (1.0 - self.beta2 ** self.t) ** 0.5)
        torch._foreach_add_(denom, ADAM_EPS)
        torch._foreach_addcdiv_(params, self.exp_avg, denom, step_size)

    def state_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "t": self.t, "momentum_buffer": self.momentum_buffer, "exp_avg": self.exp_avg,
                "exp_avg_sq": self.exp_avg_sq}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["kind"] != self.kind:
            raise ValueError(f"optimizer state is for {state['kind']!r}, not {self.kind!r}")

        def to_params(tensors: list[Tensor] | None) -> list[Tensor] | None:
            if tensors is None:
                return None
            if len(tensors) != len(self.params):
                raise ValueError("optimizer state does not match the parameters")
            return [t.to(device=p.device, dtype=p.dtype).clone() for t, p in zip(tensors, self.params)]

        self.t = int(state["t"])
        self.momentum_buffer = to_params(state["momentum_buffer"])
        self.exp_avg = to_params(state["exp_avg"])
        self.exp_avg_sq = to_params(state["exp_avg_sq"])


def light_curve_from_rows(rows: Sequence[dict[str, float]]) -> dict[str, np.ndarray]:
    """Stack light-curve rows into arrays: :data:`LIGHT_CURVE_KEYS` first, then extra keys sorted; ``step`` is
    int64, everything else float64 (NaN where a row lacks a key)."""
    extra = sorted({k for row in rows for k in row} - set(LIGHT_CURVE_KEYS))
    lc = {k: np.asarray([row.get(k, _NAN) for row in rows], dtype=np.float64) for k in (*LIGHT_CURVE_KEYS, *extra)}
    lc["step"] = lc["step"].astype(np.int64)
    return lc


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _total_norm(tensors: Sequence[Tensor]) -> float:
    return float(torch.nn.utils.get_total_norm(tensors))


@contextlib.contextmanager
def _deterministic_algorithms() -> Iterator[None]:
    """Enable deterministic algorithms (warn-only) for the duration of a run, then restore the previous mode."""
    previous, previous_warn = torch.are_deterministic_algorithms_enabled(), \
        torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


class _Run:
    """State of one observation (see the module docstring)."""

    def __init__(self, spec: ObjectSpec, work_dir: Path, ri: RIConfig | None, ri_steps: Sequence[int] | None,
                 ri_kind: str, ri_meta: dict | None) -> None:
        self.spec, self.train = spec, spec.train
        self.object_id = spec.object_id()
        self.ckpt_path = work_dir / RESUME_FILE
        self.task = make_task(spec.task, spec.task_cfg, spec.seed)
        model = self.task.build_model()
        self.n_params = int(sum(p.numel() for p in model.parameters()))
        self.device = pick_device(spec.device, self.n_params)
        self.model = model.to(self.device)
        self.model.train()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = LeanOptimizer(self.params, self.train)
        self.gen = torch_generator(spec.seed, "data")
        self.photometer = Photometer(ntk_probe=self.train.ntk_probe)
        self.splits = {k: (x.to(self.device), y.to(self.device)) for k, (x, y) in self.task.eval_splits().items()}
        self.on_cpu = self.device.type == "cpu"
        self.clip = None if self.train.grad_clip is None else float(self.train.grad_clip)

        total = int(self.train.steps)
        if total < 0 or int(self.train.batch_size) < 1:
            raise ValueError("train.steps must be >= 0 and train.batch_size >= 1")
        self.total = total
        self.eval_steps = eval_steps(total, self.train.n_log_evals, self.train.n_lin_evals)
        self.ri_cfg = ri if (ri is not None and ri.enabled) else None
        if self.ri_cfg is None:
            self.ri_steps = np.zeros(0, dtype=np.int64)
        elif ri_steps is None:
            self.ri_steps = subsample_log(self.eval_steps, self.ri_cfg.n_points)
        else:
            self.ri_steps = np.unique(np.asarray(list(ri_steps), dtype=np.int64))
            if self.ri_steps.size and (self.ri_steps[0] < 0 or self.ri_steps[-1] > total):
                raise ValueError(f"ri_steps must lie in [0, {total}]")
        self.eval_set = {int(s) for s in self.eval_steps}
        self.ri_set = {int(s) for s in self.ri_steps}
        self.checkpoints = [int(s) for s in np.union1d(self.eval_steps, self.ri_steps)]
        self.max_order = None
        if self.ri_cfg is not None:
            self.max_order = int(self.ri_cfg.max_order or self.task.default_ri_order())
        self.ri_kind = str(ri_kind)
        self.ri_extra_meta = dict(ri_meta or {})
        self.fingerprint = {
            "object_id": self.object_id, "code": tycho.__version__, "eval_steps": [int(s) for s in self.eval_steps],
            "ri_steps": [int(s) for s in self.ri_steps], "ri": None if self.ri_cfg is None else asdict(self.ri_cfg),
            "ri_kind": self.ri_kind, "ri_meta": json.dumps(self.ri_extra_meta, sort_keys=True, default=_json_default)}

        # mutable state (saved in resume checkpoints)
        self.step = 0
        self.next_idx = 0
        self.rows: list[dict[str, float]] = []
        self.curve: RICurve | None = None
        self.ri_frozen: dict | None = None
        self.diverged = False
        self.last_grad_norm = _NAN
        self.last_update_norm = _NAN
        self.elapsed_before = 0.0
        self.session_start = time.perf_counter()
        self.last_save = self.session_start

    # ------------------------------------------------------------------------------------------ bookkeeping
    def elapsed(self) -> float:
        return self.elapsed_before + (time.perf_counter() - self.session_start)

    def light_curve(self) -> dict[str, np.ndarray]:
        return light_curve_from_rows(self.rows)

    def pruned_curve(self) -> RICurve | None:
        if self.curve is None:
            return None
        cfg = self.ri_cfg
        return self.curve.prune(cfg.keep_top, alpha=cfg.alpha, min_effect=cfg.min_effect)

    def result(self, status: str, *, error: str | None = None) -> RunResult:
        lc = self.light_curve()
        summary: dict = {}
        if lc["step"].size:
            summary = summarize(lc, self.task.chance_accuracy())
            if self.diverged:
                summary["diverged"] = True
        return RunResult(object_id=self.object_id, status=status, light_curve=lc, summary=summary,
                         ri_curve=self.pruned_curve(), wall_time_s=self.elapsed(), device=str(self.device),
                         n_params=self.n_params, error=error)

    # ------------------------------------------------------------------------------------------ checkpoints
    def save(self) -> None:
        curve = None
        if self.curve is not None:
            c = self.curve
            curve = {"steps": torch.from_numpy(c.steps.copy()), "names": list(c.names),
                     "subsets": [list(s) for s in c.subsets], "ri": torch.from_numpy(c.ri.copy()),
                     "ri_plugin": torch.from_numpy(c.ri_plugin.copy()), "p": torch.from_numpy(c.p.copy()),
                     "q": torch.from_numpy(c.q.copy()), "n_eff": torch.from_numpy(c.n_eff.copy()),
                     "ground_truth": json.dumps(c.ground_truth, default=_json_default),
                     "meta": json.dumps(c.meta, default=_json_default)}
        state = {"format": RESUME_FORMAT, "fingerprint": self.fingerprint, "step": self.step,
                 "next_idx": self.next_idx, "model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                 "gen_data": self.gen.get_state(), "torch_rng": torch.get_rng_state(),
                 "photometer": self.photometer.state_dict(), "rows": self.rows,
                 "ri_frozen": json.dumps(self.ri_frozen, default=_json_default), "curve": curve,
                 "last_grad_norm": self.last_grad_norm, "last_update_norm": self.last_update_norm,
                 "elapsed_s": self.elapsed()}
        tmp = self.ckpt_path.with_name(self.ckpt_path.name + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)
        self.last_save = time.perf_counter()
        logger.debug("%s: resume checkpoint at step %d", self.object_id, self.step)

    def maybe_save(self) -> None:
        if time.perf_counter() - self.last_save >= float(self.train.checkpoint_every_s):
            self.save()

    def restore(self) -> bool:
        """Load ``resume.pt`` if it exists and matches this run; return whether it was used."""
        if not self.ckpt_path.exists():
            return False
        try:
            state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # a torn or foreign file must not stop the survey: start over
            logger.warning("%s: unreadable resume checkpoint %s (%s); starting from scratch", self.object_id,
                           self.ckpt_path, exc)
            return False
        if state.get("format") != RESUME_FORMAT or state.get("fingerprint") != self.fingerprint:
            logger.warning("%s: resume checkpoint %s belongs to another object, schedule or RI setting; starting "
                           "from scratch", self.object_id, self.ckpt_path)
            return False
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.gen.set_state(state["gen_data"])
        torch.set_rng_state(state["torch_rng"])
        self.photometer.load_state_dict(state["photometer"])
        self.step, self.next_idx = int(state["step"]), int(state["next_idx"])
        self.rows = list(state["rows"])
        self.ri_frozen = json.loads(state["ri_frozen"])
        self.last_grad_norm, self.last_update_norm = float(state["last_grad_norm"]), float(state["last_update_norm"])
        self.elapsed_before = float(state["elapsed_s"])
        c = state["curve"]
        if c is not None:
            self.curve = RICurve(steps=c["steps"].numpy(), names=c["names"], subsets=[tuple(s) for s in c["subsets"]],
                                 ri=c["ri"].numpy(), ri_plugin=c["ri_plugin"].numpy(), p=c["p"].numpy(),
                                 q=c["q"].numpy(), n_eff=c["n_eff"].numpy(), ground_truth=json.loads(c["ground_truth"]),
                                 meta=json.loads(c["meta"]))
        logger.info("%s: resumed at step %d of %d", self.object_id, self.step, self.total)
        return True

    # ------------------------------------------------------------------------------------------ training
    def update(self, record: bool) -> bool:
        """One optimizer update; False (and no update) if the minibatch loss is not finite."""
        lr = lr_at(self.step, self.train)
        x, y = self.task.sample_batch(self.train.batch_size, self.gen)
        if not self.on_cpu:
            x, y = x.to(self.device), y.to(self.device)
        loss = self.task.loss(self.model(x), y)
        if not math.isfinite(loss.item()):
            return False
        grads = list(torch.autograd.grad(loss, self.params, allow_unused=True, materialize_grads=True))
        norm = _total_norm(grads) if (record or self.clip is not None) else _NAN
        if self.clip is not None:
            coef = self.clip / (norm + CLIP_EPS)
            if coef < 1.0:
                torch._foreach_mul_(grads, coef)
        before = [p.detach().clone() for p in self.params] if record else None
        self.opt.step(grads, lr)
        if record:
            self.last_grad_norm = norm
            self.last_update_norm = _total_norm(torch._foreach_sub([p.detach() for p in self.params], before))
        return True

    def advance(self, target: int, stop_flag: Callable[[], bool] | None) -> str:
        """Train up to step ``target``; returns "ok", "stopped" (stop_flag) or "diverged"."""
        record_last = target in self.eval_set
        countdown = POLL_STEPS
        while self.step < target:
            countdown -= 1
            if countdown <= 0:
                countdown = POLL_STEPS
                if stop_flag is not None and stop_flag():
                    return "stopped"
                self.maybe_save()
            if not self.update(record_last and self.step + 1 == target):
                return "diverged"
            self.step += 1
        return "ok"

    # ------------------------------------------------------------------------------------------ observation
    def params_finite(self) -> bool:
        return all(bool(torch.isfinite(p).all()) for p in self.params)

    def measure(self, step: int, *, do_ntk: bool | None = None) -> None:
        if do_ntk is None:
            every = int(self.train.ntk_every)
            do_ntk = every > 0 and len(self.rows) % every == 0
        grad_norm, update_norm = (self.last_grad_norm, self.last_update_norm) if step > 0 else (_NAN, _NAN)
        row = self.photometer.measure(step, self.model, self.task, self.splits, grad_norm, lr_at(step, self.train),
                                      do_ntk, update_norm=update_norm, wall_s=self.elapsed())
        self.rows.append(row)
        self.last_grad_norm = self.last_update_norm = _NAN

    def observe_ri(self, step: int) -> None:
        cfg = self.ri_cfg
        was_training = self.model.training
        self.model.eval()
        try:
            vs = self.task.ri_variables(self.model, cfg.n_probe, torch_generator(self.spec.seed, "ri", step),
                                        frozen=self.ri_frozen)
        finally:
            self.model.train(was_training)
        if self.ri_frozen is None and vs.meta.get("frozen") is not None:
            self.ri_frozen = dict(vs.meta["frozen"])
        scan = ri_scan(vs, max_order=self.max_order, alpha=cfg.alpha, min_effect=cfg.min_effect, null=cfg.null,
                       max_subsets=cfg.max_subsets, rng=numpy_rng(self.spec.seed, "ri-scan", step))
        if self.curve is None:
            meta = {"alpha": cfg.alpha, "min_effect": cfg.min_effect, "max_order": self.max_order, "null": cfg.null,
                    "n_probe": cfg.n_probe, "object_id": self.object_id, "kind": self.ri_kind,
                    "task": self.spec.task, "seed": int(self.spec.seed), "variables": dict(vs.meta),
                    **self.ri_extra_meta}
            self.curve = RICurve.empty(scan.names, scan.subsets, ground_truth=vs.ground_truth, meta=meta)
        self.curve = self.curve.append(step, scan)

    def observe(self, step: int) -> None:
        """Light-curve row (evaluation steps) and RI scan (RI steps) at a checkpoint step. Non-finite parameters
        end the run with a final row at this step; so does a non-finite evaluation train loss."""
        finite = self.params_finite()
        if step in self.eval_set or not finite:
            self.measure(step)
            if not finite or not math.isfinite(self.rows[-1]["train_loss"]):
                self.diverged = True
        if self.ri_cfg is not None and step in self.ri_set and not self.diverged:
            self.observe_ri(step)

    def on_divergence(self) -> None:
        """Non-finite minibatch loss at ``self.step``: record a final row there and stop."""
        self.diverged = True
        if not self.rows or int(self.rows[-1]["step"]) != self.step:
            self.measure(self.step, do_ntk=False)
        logger.info("%s: diverged at step %d", self.object_id, self.step)

    # ------------------------------------------------------------------------------------------ main loop
    def execute(self, stop_flag: Callable[[], bool] | None,
                on_progress: Callable[[int, int], None] | None) -> RunResult:
        while self.next_idx < len(self.checkpoints) and not self.diverged:
            target = self.checkpoints[self.next_idx]
            outcome = self.advance(target, stop_flag)
            if outcome == "stopped":
                self.save()
                return self.result("interrupted")
            if outcome == "diverged":
                self.on_divergence()
                break
            self.observe(target)
            self.next_idx += 1
            if on_progress is not None:
                on_progress(target, self.total)
            if self.diverged:
                break
            if stop_flag is not None and stop_flag():
                self.save()
                return self.result("interrupted")
            self.maybe_save()
        result = self.result("done")
        self.ckpt_path.unlink(missing_ok=True)
        return result


def run_object(spec: ObjectSpec, *, work_dir: Path, ri: RIConfig | None = None,
               stop_flag: Callable[[], bool] | None = None, on_progress: Callable[[int, int], None] | None = None,
               resume: bool = True, ri_steps: Sequence[int] | None = None, ri_kind: str = "insitu",
               ri_meta: dict | None = None) -> RunResult:
    """Train and observe one object (see the module docstring).

    Args:
        spec: the object.
        work_dir: directory for ``resume.pt`` (created if needed).
        ri: RI spectroscopy settings; RI runs only when ``ri.enabled``.
        stop_flag: polled during training; when it returns true the run checkpoints and returns "interrupted".
        on_progress: called as ``on_progress(step, total_steps)`` after each checkpoint step is observed.
        resume: continue from ``work_dir/resume.pt`` when it exists and matches.
        ri_steps: RI checkpoint steps (default ``subsample_log(eval_steps, ri.n_points)``); need not be eval steps.
        ri_kind: stored as ``ri_curve.meta["kind"]`` ("insitu" or "replay").
        ri_meta: extra entries for ``ri_curve.meta`` (JSON-friendly).

    Returns:
        A :class:`~tycho.train.spec.RunResult` with status "done", "interrupted" or "failed". The RI curve is pruned
        with ``RICurve.prune(ri.keep_top, ri.alpha, ri.min_effect)``.
    """
    work_dir = Path(work_dir)
    run: _Run | None = None
    t0 = time.perf_counter()
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        with _deterministic_algorithms():
            run = _Run(spec, work_dir, ri, ri_steps, ri_kind, ri_meta)
            if resume:
                try:
                    run.restore()
                except Exception as exc:  # e.g. a checkpoint written by code with another architecture
                    bad = run.ckpt_path.with_name(run.ckpt_path.name + ".bad")
                    os.replace(run.ckpt_path, bad)
                    logger.warning("%s: could not restore %s (%s: %s); moved it to %s and starting from scratch",
                                   run.object_id, run.ckpt_path, type(exc).__name__, exc, bad)
                    run = _Run(spec, work_dir, ri, ri_steps, ri_kind, ri_meta)
            return run.execute(stop_flag, on_progress)
    except Exception:
        error = traceback.format_exc()
        logger.error("object %s failed:\n%s", spec.object_id(), error)
        if run is not None:
            try:  # keep the partial light curve when it can still be assembled
                return run.result("failed", error=error)
            except Exception:
                logger.exception("object %s: could not assemble the partial result", spec.object_id())
        return RunResult(object_id=spec.object_id(), status="failed", wall_time_s=time.perf_counter() - t0,
                         device="cpu" if run is None else str(run.device),
                         n_params=0 if run is None else run.n_params, error=error)
