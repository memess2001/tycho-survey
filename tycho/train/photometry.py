"""Photometry: the cheap per-checkpoint observables of a training run (its light curve, SPEC §4.4).

At every evaluation step :class:`Photometer` records one row with the keys :data:`LIGHT_CURVE_KEYS` (always present,
NaN when not applicable) plus the task's extra metrics as ``x_<name>``:

* ``train_loss``, ``test_loss``, ``train_acc``, ``test_acc``: on the task's fixed evaluation splits (accuracy NaN
  when the task has none).
* ``param_norm`` ``‖θ‖₂``, ``dist_from_init`` ``‖θ − θ₀‖₂`` (all trainable parameters, θ₀ = first measurement).
* ``grad_norm`` / ``update_norm``: global norms of the minibatch gradient (before clipping) and of the parameter
  change ``θ_t − θ_{t−1}`` of the *last* update before this step; NaN at step 0. ``lr``: the schedule's rate at
  this step (the runner provides these three). ``wall_s``: seconds since the object started (all sessions).
* ``erank_weights``: :func:`effective_rank` of the model's largest 2-D weight matrix (:func:`monitored_weight`;
  the input layer of the MLP tasks, ``W`` of the superposition model).
* ``erank_hidden``: effective rank of the column-centred representation (``Task.hidden_representation``) of the
  first :data:`REP_PROBE` test inputs; ``rep_cka_prev``: linear CKA between that representation and the one at the
  previous evaluation (NaN at the first).
* eNTK statistics on the first ``ntk_probe`` training inputs, at the evaluations the runner selects (every
  ``ntk_every``-th): the Gram matrix ``K_ij = ⟨∇_θ f(x_i), ∇_θ f(x_j)⟩`` of the per-sample scalar
  ``f = Task.ntk_output`` (:func:`ntk_gram`), its trace, top eigenvalue, effective rank and centred kernel-target
  alignment (:func:`ntk_stats`), and ``ntk_velocity = 1 − cos(K_t, K_prev)`` (Frobenius cosine) over successive
  eNTK measurements. NaN at evaluations without eNTK.

Effective rank (Roy & Vetterli 2007): ``exp(H(σ / Σσ))`` for the singular values σ, between 1 (rank one) and the
rank; 0 for an all-zero matrix and NaN for non-finite input.

Linear CKA (Kornblith et al. 2019): ``‖Yᵀ X‖²_F / (‖Xᵀ X‖_F ‖Yᵀ Y‖_F)`` for column-centred X, Y (same rows).

Centred kernel-target alignment (Cortes et al. 2012): ``⟨K_c, T_c⟩_F / (‖K_c‖_F ‖T_c‖_F)`` with ``K_c = H K H``,
``H = I − 11ᵀ/n``, and ``T_c = Y_c Y_cᵀ`` for the centred one-hot label matrix ``Y_c`` (for two classes this is the
same as centred ±1 labels, since alignment is scale-invariant).

All statistics are computed in float64 on the CPU and are deterministic for a fixed thread count. Non-finite
parameters (a diverging run) give NaN statistics, never exceptions.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

if TYPE_CHECKING:  # pragma: no cover
    from tycho.tasks.base import Task

__all__ = ["EVAL_CHUNK", "LIGHT_CURVE_KEYS", "REP_PROBE", "Photometer", "effective_rank", "flat_params",
           "linear_cka", "monitored_weight", "ntk_gram", "ntk_stats", "param_norm"]

logger = logging.getLogger(__name__)

#: Keys of every light-curve row (SPEC §4.4), in storage order; task extras follow as ``x_<name>``.
LIGHT_CURVE_KEYS: tuple[str, ...] = (
    "step", "train_loss", "test_loss", "train_acc", "test_acc", "param_norm", "grad_norm", "update_norm",
    "dist_from_init", "erank_weights", "erank_hidden", "rep_cka_prev", "ntk_trace", "ntk_top_eig", "ntk_erank",
    "ntk_align", "ntk_velocity", "lr", "wall_s")
#: Rows of the fixed representation probe (the first test inputs).
REP_PROBE = 256
#: Evaluation inputs per forward pass (bounds memory on large evaluation splits).
EVAL_CHUNK = 4096

_NAN = float("nan")
_func_fallback_logged = False


# ---------------------------------------------------------------------------------------------- statistics
def flat_params(model: nn.Module) -> Tensor:
    """All trainable parameters concatenated into one float64 CPU vector."""
    return torch.cat([p.detach().reshape(-1).to("cpu", torch.float64) for p in model.parameters()
                      if p.requires_grad])


def param_norm(model: nn.Module) -> float:
    """Global L2 norm of the trainable parameters."""
    return float(torch.linalg.vector_norm(flat_params(model)))


def effective_rank(M: Tensor) -> float:
    """Roy–Vetterli effective rank ``exp(H(σ/Σσ))`` of a 2-D tensor (0 for a zero matrix, NaN if non-finite)."""
    M = M.detach().to("cpu", torch.float64)
    if M.dim() != 2:
        raise ValueError(f"effective_rank needs a 2-D tensor, got shape {tuple(M.shape)}")
    if M.numel() == 0 or not bool(torch.isfinite(M).all()):
        return _NAN
    s = torch.linalg.svdvals(M)
    return _entropy_rank(s)


def _entropy_rank(values: Tensor) -> float:
    """``exp(entropy)`` of non-negative values normalised to sum 1 (0 if they are all zero)."""
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def linear_cka(X: Tensor, Y: Tensor) -> float:
    """Linear CKA of two representations with the same rows (NaN if either is constant or non-finite)."""
    X = X.detach().to("cpu", torch.float64).reshape(X.shape[0], -1)
    Y = Y.detach().to("cpu", torch.float64).reshape(Y.shape[0], -1)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"linear_cka needs the same number of rows, got {X.shape[0]} and {Y.shape[0]}")
    Xc, Yc = X - X.mean(dim=0), Y - Y.mean(dim=0)
    den = float(torch.linalg.matrix_norm(Xc.T @ Xc) * torch.linalg.matrix_norm(Yc.T @ Yc))
    if not math.isfinite(den) or den <= 0.0:
        return _NAN
    return float(torch.linalg.matrix_norm(Yc.T @ Xc) ** 2) / den


def monitored_weight(model: nn.Module) -> Tensor | None:
    """The model's largest 2-D trainable parameter (by element count; the first one on ties), or None."""
    mats = [p for p in model.parameters() if p.requires_grad and p.dim() == 2]
    return max(mats, key=lambda p: p.numel()) if mats else None


def _jacobian_func(model: nn.Module, names: list[str], params: list[Tensor], x: Tensor, y: Tensor,
                   task: "Task") -> Tensor:
    """Per-sample gradients ``(n, P)`` of ``task.ntk_output`` with ``torch.func`` (``vmap(grad(·))``)."""
    p_dict = {k: p.detach() for k, p in zip(names, params)}
    buffers = {k: b.detach() for k, b in model.named_buffers()}

    def output(p: dict[str, Tensor], xi: Tensor, yi: Tensor) -> Tensor:
        out = functional_call(model, (p, buffers), (xi.unsqueeze(0),))
        return task.ntk_output(out, yi.unsqueeze(0)).squeeze(0)

    grads = vmap(grad(output), in_dims=(None, 0, 0))(p_dict, x, y)
    return torch.cat([grads[k].reshape(x.shape[0], -1) for k in names], dim=1)


def _jacobian_loop(params: list[Tensor], model: nn.Module, x: Tensor, y: Tensor, task: "Task") -> Tensor:
    """Per-sample gradients ``(n, P)`` with one backward pass per sample (fallback)."""
    rows = []
    with torch.enable_grad():
        for i in range(x.shape[0]):
            f = task.ntk_output(model(x[i:i + 1]), y[i:i + 1]).sum()
            grads = torch.autograd.grad(f, params, allow_unused=True)
            rows.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                   for g, p in zip(grads, params)]))
    return torch.stack(rows)


def ntk_gram(model: nn.Module, x: Tensor, y: Tensor, task: "Task", *, method: str = "auto") -> Tensor:
    """Empirical NTK Gram matrix ``(n, n)`` (float64, CPU) of ``task.ntk_output`` on the inputs ``x``.

    ``method``: ``"func"`` (``torch.func``), ``"loop"`` (per-sample backward) or ``"auto"`` (func, falling back to
    the loop if the model or task is not supported by ``torch.func``; the fallback is logged once per process).
    """
    global _func_fallback_logged
    if method not in ("auto", "func", "loop"):
        raise ValueError(f"method must be 'auto', 'func' or 'loop', got {method!r}")
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    names, params = [k for k, _ in named], [p for _, p in named]
    if method == "loop":
        J = _jacobian_loop(params, model, x, y, task)
    else:
        try:
            J = _jacobian_func(model, names, params, x, y, task)
        except Exception as exc:  # unsupported op under vmap: the loop computes the same matrix
            if method == "func":
                raise
            if not _func_fallback_logged:
                logger.warning("torch.func eNTK failed (%s: %s); using the per-sample loop", type(exc).__name__, exc)
                _func_fallback_logged = True
            J = _jacobian_loop(params, model, x, y, task)
    J = J.detach().to("cpu", torch.float64)
    return J @ J.T


def _target_gram(y: Tensor | None, n: int) -> Tensor | None:
    """Centred target Gram matrix: one-hot for 1-D integer labels, the targets themselves for float arrays."""
    if y is None:
        return None
    y = y.detach().cpu()
    if y.shape[0] != n:
        raise ValueError(f"labels have {y.shape[0]} rows, kernel has {n}")
    if y.dim() == 1 and not y.is_floating_point():
        Y = torch.nn.functional.one_hot(y.long(), num_classes=int(y.max()) + 1).to(torch.float64)
    else:
        Y = y.to(torch.float64).reshape(n, -1)
    Yc = Y - Y.mean(dim=0)
    return Yc @ Yc.T


def ntk_stats(K: Tensor, y: Tensor | None = None) -> dict[str, float]:
    """``trace``, ``top_eig``, ``erank`` (of the eigenvalues) and centred kernel-target ``align`` of a Gram matrix.

    ``y``: 1-D integer class labels (one-hot targets) or float targets ``(n,)`` / ``(n, k)``; None → align NaN.
    All values are NaN for a non-finite K.
    """
    K = K.detach().to("cpu", torch.float64)
    n = K.shape[0]
    out = {"trace": _NAN, "top_eig": _NAN, "erank": _NAN, "align": _NAN}
    if K.dim() != 2 or K.shape[1] != n:
        raise ValueError(f"K must be square, got shape {tuple(K.shape)}")
    if n == 0 or not bool(torch.isfinite(K).all()):
        return out
    K = 0.5 * (K + K.T)
    eig = torch.linalg.eigvalsh(K).clamp_min(0.0)
    out["trace"] = float(torch.trace(K))
    out["top_eig"] = float(eig[-1])
    out["erank"] = _entropy_rank(eig)
    T = _target_gram(y, n)
    if T is not None:
        Kc = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
        den = float(torch.linalg.matrix_norm(Kc) * torch.linalg.matrix_norm(T))
        if math.isfinite(den) and den > 0.0:
            out["align"] = float((Kc * T).sum()) / den
    return out


def _frobenius_cos(A: Tensor, B: Tensor) -> float:
    den = float(torch.linalg.matrix_norm(A) * torch.linalg.matrix_norm(B))
    if not math.isfinite(den) or den <= 0.0:
        return _NAN
    return float((A * B).sum()) / den


# ---------------------------------------------------------------------------------------------- photometer
class Photometer:
    """Measures light-curve rows and keeps the state they depend on (θ₀, the previous representation and eNTK).

    Args:
        ntk_probe: training inputs used for the eNTK (the first ``ntk_probe`` of the "train" split).
        rep_probe: test inputs used for the representation statistics.
        eval_chunk: evaluation inputs per forward pass.
    """

    def __init__(self, *, ntk_probe: int = 32, rep_probe: int = REP_PROBE, eval_chunk: int = EVAL_CHUNK) -> None:
        if min(ntk_probe, rep_probe, eval_chunk) < 1:
            raise ValueError("ntk_probe, rep_probe and eval_chunk must be >= 1")
        self.ntk_probe, self.rep_probe, self.eval_chunk = int(ntk_probe), int(rep_probe), int(eval_chunk)
        self.theta0: Tensor | None = None
        self.rep_prev: Tensor | None = None
        self.K_prev: Tensor | None = None

    def evaluate(self, model: nn.Module, task: "Task", x: Tensor, y: Tensor) -> tuple[float, float]:
        """Mean loss and accuracy (NaN if the task has none) of ``model`` on ``(x, y)``, in chunks."""
        n = x.shape[0]
        loss_sum, acc_sum, has_acc = 0.0, 0.0, True
        with torch.no_grad():
            for lo in range(0, n, self.eval_chunk):
                xb, yb = x[lo:lo + self.eval_chunk], y[lo:lo + self.eval_chunk]
                out = model(xb)
                m = xb.shape[0]
                loss_sum += float(task.loss(out, yb)) * m
                acc = task.accuracy(out, yb)
                if acc is None:
                    has_acc = False
                else:
                    acc_sum += acc * m
        return loss_sum / n, (acc_sum / n if has_acc else _NAN)

    def measure(self, step: int, model: nn.Module, task: "Task", splits: dict[str, tuple[Tensor, Tensor]],
                grad_norm: float, lr: float, do_ntk: bool, *, update_norm: float = _NAN,
                wall_s: float = _NAN) -> dict[str, float]:
        """One light-curve row at ``step`` (see the module docstring). ``splits`` must be on the model's device."""
        row: dict[str, Any] = {k: _NAN for k in LIGHT_CURVE_KEYS}
        row.update(step=int(step), grad_norm=float(grad_norm), update_norm=float(update_norm), lr=float(lr),
                   wall_s=float(wall_s))
        was_training = model.training
        model.eval()
        try:
            for name in ("train", "test"):
                row[f"{name}_loss"], row[f"{name}_acc"] = self.evaluate(model, task, *splits[name])
            theta = flat_params(model)
            if self.theta0 is None:
                self.theta0 = theta.clone()
            row["param_norm"] = float(torch.linalg.vector_norm(theta))
            row["dist_from_init"] = float(torch.linalg.vector_norm(theta - self.theta0))
            weight = monitored_weight(model)
            row["erank_weights"] = effective_rank(weight) if weight is not None else _NAN
            with torch.no_grad():
                rep = task.hidden_representation(model, splits["test"][0][: self.rep_probe])
                rep = rep.detach().to("cpu", torch.float64)
                extras = task.extra_metrics(model)
            row["erank_hidden"] = effective_rank(rep - rep.mean(dim=0))
            row["rep_cka_prev"] = linear_cka(self.rep_prev, rep) if self.rep_prev is not None else _NAN
            self.rep_prev = rep
            if do_ntk:
                x, y = splits["train"]
                x, y = x[: self.ntk_probe], y[: self.ntk_probe]
                K = ntk_gram(model, x, y, task)
                stats = ntk_stats(K, y if y.dim() == 1 else None)
                row.update(ntk_trace=stats["trace"], ntk_top_eig=stats["top_eig"], ntk_erank=stats["erank"],
                           ntk_align=stats["align"])
                row["ntk_velocity"] = 1.0 - _frobenius_cos(K, self.K_prev) if self.K_prev is not None else _NAN
                self.K_prev = K
            for key, value in extras.items():
                row[f"x_{key}"] = float(value)
        finally:
            model.train(was_training)
        return row

    def state_dict(self) -> dict[str, Tensor | None]:
        """The photometer's memory (for resume checkpoints)."""
        return {"theta0": self.theta0, "rep_prev": self.rep_prev, "K_prev": self.K_prev}

    def load_state_dict(self, state: dict[str, Tensor | None]) -> None:
        self.theta0, self.rep_prev, self.K_prev = state["theta0"], state["rep_prev"], state["K_prev"]
