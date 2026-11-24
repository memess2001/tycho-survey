"""88-core pre-registered confirmation runner for relay vs. textbook induction circuits.

Specification: ``docs/results/relay_confirmation_preregistration.md`` (upgraded from
``notes/relay_confirmation_plan_draft_2026-11-22.md``).

Matrix: 76 cells x 24 fresh runs = 1,824 runs across three suites:
  * Suite A (45 cells, 1,080 runs): frozen claims C1 (A1, A2), C2 (A3), C3 (A4, A5), C4 (A6).
  * Suite B (18 cells, 432 runs): generality (B1-W AdamW, B1-U warmup, B1-WU both, B2 +MLP, B3 data window).
  * Suite C (13 cells, 312 runs): width bridge (d_model = 32, 128, 256; d_model = 64 reuses Suite A).

Seeds:
  * Standard cell (24 runs): seeds 1000..1011 with 2 heads, seeds 1012..1023 with 4 heads.
  * C4 head-count ladder (A6-1..A6-5): all 24 seeds 1000..1023 with fixed head count (2, 4, or 8).
  * Top-up reserve (prereg §2.5): seeds 1024..1035 with 2 heads, 1036..1047 with 4 heads.

Instrument checks (``--suite ic``):
  * IC-1: 2 runs replayed twice on the same machine -> bit-identical curves & final weights.
  * IC-2: 18 exploratory runs (seeds 300..302, h2 & h4 on base, lr5e-4, ov1_double) -> <= 1 flip vs exploratory.
  * IC-4: default options off -> bit-identical to ``relay_band_pilot.train`` at step 600 on single_rm12_h2_s300.

Usage (from repository root):
  PYTHONPATH=. python scripts/analysis_v2/relay_confirm.py --list
  PYTHONPATH=. python scripts/analysis_v2/relay_confirm.py --smoke --jobs 4
  PYTHONPATH=. python scripts/analysis_v2/relay_confirm.py --suite ic --jobs 20
  PYTHONPATH=. python scripts/analysis_v2/relay_confirm.py --suite A --jobs 86
  PYTHONPATH=. python scripts/analysis_v2/relay_confirm.py --suite all --jobs 86
  PYTHONPATH=. python scripts/analysis_v2/relay_confirm.py --topup A1-2 A1-6 --jobs 48
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NEAR = 4
SEEDS_MAIN_H2 = tuple(range(1000, 1012))     # 12 seeds -> 2 heads
SEEDS_MAIN_H4 = tuple(range(1012, 1024))     # 12 seeds -> 4 heads
SEEDS_MAIN_ALL = tuple(range(1000, 1024))    # 24 seeds (for A6-1..A6-5 fixed head count)
SEEDS_TOPUP_H2 = tuple(range(1024, 1036))    # 12 reserve seeds -> 2 heads
SEEDS_TOPUP_H4 = tuple(range(1036, 1048))    # 12 reserve seeds -> 4 heads

STAT_KEYS = (
    "band_l0",
    "band_l1",
    "source_l1",
    "offsets_1_2_l0",
    "loss_rep",
    "rms_emb",
    "rms_y0",
    "rms_y1",
    "r1",
    "rms_m0",
    "rms_m1",
)


def measure_steps(total_steps: int) -> tuple[int, ...]:
    """Every 50 steps up to 2,500, then every 250 steps up to ``total_steps`` (prereg §2.1)."""
    early = list(range(0, min(total_steps, 2500) + 1, 50))
    late = list(range(2750, total_steps + 1, 250)) if total_steps > 2500 else []
    pts = sorted(set(early + late + [total_steps]))
    return tuple(pts)


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    suite: str
    group: str
    role: str  # "primary", "secondary", "descriptive", "exploratory", "gate"
    lr: float = 1e-3
    batch_size: int = 32
    steps: int = 12_000
    rep_min: int = 12
    d_model: int = 64
    attn_only: bool = True
    mlp_ratio: int = 4
    optimizer: str = "adam"
    weight_decay: float = 0.0
    warmup_steps: int = 0
    lr_schedule: str = "constant"
    fixed_heads: int | None = None  # if set, all 24 seeds use this n_head
    lr_scale: dict[str, float] | None = None
    init_scale: dict[str, float] | None = None
    out_scale: dict[int, float] | None = None
    out_switch: dict[str, Any] | None = None  # {"step": 1000, "fold": bool}
    mask_window: tuple[int, int] | None = None  # (block_from, block_until)
    mask_layer: int = 0
    mask_forbid: str = "far"
    data_window: dict[str, int] | None = None  # {"start": s, "end": e, "rep_min": rm}
    note: str = ""

    def run_pairs(self, topup: bool = False) -> list[tuple[int, int]]:
        if self.fixed_heads is not None:
            if topup:
                raise ValueError(f"Cell {self.cell_id} (fixed_heads={self.fixed_heads}) is excluded from §2.5 top-up.")
            return [(s, self.fixed_heads) for s in SEEDS_MAIN_ALL]
        if topup:
            return [(s, 2) for s in SEEDS_TOPUP_H2] + [(s, 4) for s in SEEDS_TOPUP_H4]
        return [(s, 2) for s in SEEDS_MAIN_H2] + [(s, 4) for s in SEEDS_MAIN_H4]


def build_cells() -> list[CellSpec]:
    """Exact 76-cell matrix from preregistration §3–§5."""
    c: list[CellSpec] = []

    # ---- Suite A: C1 (A1: 9 cells, A2: 14 cells) ----
    c.extend([
        CellSpec("A1-1", "A", "A1", "primary", lr=5e-4, batch_size=32, note="sigma=0.88e-4 ref"),
        CellSpec("A1-2", "A", "A1", "primary", lr=1e-3, batch_size=128, note="sigma=0.88e-4, same lr as base"),
        CellSpec("A1-3", "A", "A1", "secondary", lr=1e-3, batch_size=64, note="sigma=1.25e-4 threshold"),
        CellSpec("A1-4", "A", "A1", "secondary", lr=5e-4, batch_size=16, note="sigma=1.25e-4 threshold (new)"),
        CellSpec("A1-5", "A", "A1", "secondary", lr=2e-3, batch_size=256, note="sigma=1.25e-4 threshold (new)"),
        CellSpec("A1-6", "A", "A1", "primary", lr=1e-3, batch_size=32, note="sigma=1.77e-4 baseline"),
        CellSpec("A1-7", "A", "A1", "primary", lr=2e-3, batch_size=128, note="sigma=1.77e-4, same lr/B as A1-1"),
        CellSpec("A1-8", "A", "A1", "secondary", lr=5e-4, batch_size=8, note="sigma=1.77e-4, 3rd lr"),
        CellSpec("A1-9", "A", "A1", "primary", lr=2e-3, batch_size=32, note="sigma=3.54e-4 high noise"),
    ])

    c.extend([
        CellSpec("A2-1", "A", "A2", "primary", rep_min=16, lr=1e-3, batch_size=32),
        CellSpec("A2-2", "A", "A2", "primary", rep_min=16, lr=2e-3, batch_size=32),
        CellSpec("A2-3", "A", "A2", "descriptive", rep_min=20, lr=1e-3, batch_size=32),
        CellSpec("A2-4", "A", "A2", "primary", rep_min=20, lr=2e-3, batch_size=32),
        CellSpec("A2-5", "A", "A2", "descriptive", rep_min=24, lr=1e-3, batch_size=32),
        CellSpec("A2-6", "A", "A2", "secondary", rep_min=24, lr=2e-3, batch_size=32, note="new"),
        CellSpec("A2-7", "A", "A2", "descriptive", rep_min=4, lr=1e-3, batch_size=64),
        CellSpec("A2-8", "A", "A2", "secondary", rep_min=4, lr=1e-3, batch_size=32),
        CellSpec("A2-9", "A", "A2", "descriptive", rep_min=4, lr=2e-3, batch_size=64),
        CellSpec("A2-10", "A", "A2", "descriptive", rep_min=8, lr=1e-3, batch_size=64),
        CellSpec("A2-11", "A", "A2", "descriptive", rep_min=8, lr=1e-3, batch_size=32),
        CellSpec("A2-12", "A", "A2", "descriptive", rep_min=20, lr=2e-3, batch_size=16, note="wall"),
        CellSpec("A2-13", "A", "A2", "descriptive", rep_min=24, lr=1.414e-3, batch_size=8, steps=24_000, note="wall"),
        CellSpec("A2-14", "A", "A2", "descriptive", rep_min=24, lr=2e-3, batch_size=8, steps=24_000, note="wall"),
    ])

    # ---- Suite A: C2 (A3: 7 cells) ----
    c.extend([
        CellSpec("A3-1", "A", "A3", "primary", lr=1e-3, batch_size=32, out_scale={0: 0.5}),
        CellSpec("A3-2", "A", "A3", "primary", lr=5e-4, batch_size=32, out_scale={0: 2.0}),
        CellSpec("A3-3", "A", "A3", "primary", lr=2e-3, batch_size=32, out_scale={0: 0.25}),
        CellSpec("A3-4", "A", "A3", "secondary", lr=2e-3, batch_size=32, out_scale={0: 0.5}, note="blind linear vs sq"),
        CellSpec("A3-5", "A", "A3", "primary", lr=1e-3, batch_size=32, lr_scale={"ov1": 2.0}),
        CellSpec("A3-6", "A", "A3", "primary", lr=5e-4, batch_size=32, lr_scale={"ov1": 0.25}),
        CellSpec("A3-7", "A", "A3", "exploratory", lr=1e-3, batch_size=32, lr_scale={"ov0": 0.5}, note="blind ov0 only"),
    ])

    # ---- Suite A: C3 (A4: 4 cells, A5: 3 cells) ----
    c.extend([
        CellSpec("A4-1", "A", "A4", "primary", lr=1e-3, batch_size=32, lr_scale={"qk0": 0.5}),
        CellSpec("A4-2", "A", "A4", "primary", lr=1e-3, batch_size=32, lr_scale={"qk1": 2.0}),
        CellSpec("A4-3", "A", "A4", "primary", lr=5e-4, batch_size=32, lr_scale={"qk0": 2.0}),
        CellSpec("A4-4", "A", "A4", "secondary", lr=2e-3, batch_size=32, lr_scale={"qk0": 0.25}, note="boundary"),
        CellSpec("A5-1", "A", "A5", "primary", lr=1e-3, batch_size=32, mask_window=(0, 500), mask_layer=0, mask_forbid="far"),
        CellSpec("A5-2", "A", "A5", "primary", lr=1e-3, batch_size=32, mask_window=(1000, 3000), mask_layer=0, mask_forbid="far"),
        CellSpec("A5-3", "A", "A5", "secondary", rep_min=20, lr=1e-3, batch_size=32, mask_window=(0, 500), mask_layer=1, mask_forbid="band", note="mirror"),
    ])

    # ---- Suite A: C4 (A6: 8 cells) ----
    c.extend([
        CellSpec("A6-1", "A", "A6", "primary", lr=2e-3, batch_size=32, fixed_heads=2, out_scale={0: 0.0625}),
        CellSpec("A6-2", "A", "A6", "descriptive", lr=2e-3, batch_size=32, fixed_heads=4, out_scale={0: 0.0625}),
        CellSpec("A6-3", "A", "A6", "primary", lr=2e-3, batch_size=32, fixed_heads=8, out_scale={0: 0.0625}),
        CellSpec("A6-4", "A", "A6", "primary", lr=2e-3, batch_size=32, fixed_heads=2, out_scale={0: 0.0625}, out_switch={"step": 1000, "fold": True}),
        CellSpec("A6-5", "A", "A6", "secondary", lr=2e-3, batch_size=32, fixed_heads=2, out_scale={0: 0.0625}, out_switch={"step": 1000, "fold": False}),
        CellSpec("A6-6", "A", "A6", "exploratory", lr=5e-4, batch_size=32, steps=24_000, out_scale={0: 0.0625}, note="decomp ref"),
        CellSpec("A6-7", "A", "A6", "exploratory", lr=5e-4, batch_size=32, steps=24_000, lr_scale={"proj0": 0.0625}, note="decomp lr only"),
        CellSpec("A6-8", "A", "A6", "exploratory", lr=5e-4, batch_size=32, steps=24_000, init_scale={"proj0": 0.0625}, note="decomp init only"),
    ])

    # ---- Suite B: Generality (18 cells) ----
    noise3 = [("low", 1e-3, 128), ("mid", 1e-3, 32), ("high", 2e-3, 32)]
    for tag, lr, bs in noise3:
        c.append(CellSpec(f"B1-W-{tag}", "B", "B1-W", "exploratory", lr=lr, batch_size=bs, optimizer="adamw", weight_decay=0.1))
    for tag, lr, bs in noise3:
        c.append(CellSpec(f"B1-U-{tag}", "B", "B1-U", "exploratory", lr=lr, batch_size=bs, warmup_steps=200))
    for tag, lr, bs in noise3:
        c.append(CellSpec(f"B1-WU-{tag}", "B", "B1-WU", "exploratory", lr=lr, batch_size=bs, optimizer="adamw", weight_decay=0.1, warmup_steps=200))
    c.append(CellSpec("B1-WU-mid-c05", "B", "B1-WU", "exploratory", lr=1e-3, batch_size=32, optimizer="adamw", weight_decay=0.1, warmup_steps=200, out_scale={0: 0.5}))
    c.append(CellSpec("B1-WU-low-c2", "B", "B1-WU", "exploratory", lr=1e-3, batch_size=128, optimizer="adamw", weight_decay=0.1, warmup_steps=200, out_scale={0: 2.0}))

    for tag, lr, bs in noise3:
        c.append(CellSpec(f"B2-{tag}", "B", "B2", "exploratory", lr=lr, batch_size=bs, attn_only=False, mlp_ratio=4))
    c.append(CellSpec("B2-mid-c05", "B", "B2", "exploratory", lr=1e-3, batch_size=32, attn_only=False, mlp_ratio=4, out_scale={0: 0.5}))
    c.append(CellSpec("B2-low-c2", "B", "B2", "exploratory", lr=1e-3, batch_size=128, attn_only=False, mlp_ratio=4, out_scale={0: 2.0}))

    c.append(CellSpec("B3-early", "B", "B3", "exploratory", lr=1e-3, batch_size=32, data_window={"start": 0, "end": 1000, "rep_min": 20}))
    c.append(CellSpec("B3-late", "B", "B3", "exploratory", lr=1e-3, batch_size=32, data_window={"start": 1000, "end": 3000, "rep_min": 20}))

    # ---- Suite C: Width bridge (13 cells; d=64 reuses A1-2, A1-6, A1-9, A3-1, A3-4) ----
    for tag, lr, bs in noise3:
        c.append(CellSpec(f"C-d32-{tag}", "C", "C-d32", "gate", lr=lr, batch_size=bs, d_model=32))
    for d in (128, 256):
        for tag, lr, bs in noise3:
            c.append(CellSpec(f"C-d{d}-{tag}", "C", f"C-d{d}", "gate", lr=lr, batch_size=bs, d_model=d))
        c.append(CellSpec(f"C-d{d}-mid-c05", "C", f"C-d{d}", "gate", lr=1e-3, batch_size=32, d_model=d, out_scale={0: 0.5}))
        c.append(CellSpec(f"C-d{d}-high-c05", "C", f"C-d{d}", "gate", lr=2e-3, batch_size=32, d_model=d, out_scale={0: 0.5}))

    return c


CELLS: list[CellSpec] = build_cells()
CELL_BY_ID: dict[str, CellSpec] = {c.cell_id: c for c in CELLS}


def weight_groups(model) -> dict[str, list[tuple[Any, slice]]]:
    """Named weight groups matching ``relay_band_pilot.weight_groups`` plus ``proj0`` / ``proj1``."""
    d = model.d_model
    groups: dict[str, list[tuple[Any, slice]]] = {
        "pos": [(model.pos.weight, slice(None))],
        "tok": [(model.tok.weight, slice(None))],
        "head": [(model.head.weight, slice(None))],
    }
    for i, block in enumerate(model.blocks):
        a = block.attn
        groups[f"qk{i}"] = [(a.qkv.weight, slice(0, 2 * d)), (a.qkv.bias, slice(0, 2 * d))]
        groups[f"ov{i}"] = [
            (a.qkv.weight, slice(2 * d, 3 * d)),
            (a.qkv.bias, slice(2 * d, 3 * d)),
            (a.proj.weight, slice(None)),
            (a.proj.bias, slice(None)),
        ]
        groups[f"proj{i}"] = [(a.proj.weight, slice(None)), (a.proj.bias, slice(None))]
    return groups


def confirm_stats(model, task) -> dict[str, float]:
    """Exact prereg §2.1 & §4 B2 metrics on the probe set (zero RNG side effects)."""
    import torch

    pr = task._probe
    x = pr["x"]
    rep_t = pr["rep"]
    rep, src = rep_t.numpy(), pr["src"].numpy()
    maps = [a.numpy() for a in model.attention_maps(x)]
    bi, ti = np.nonzero(rep)
    out: dict[str, float] = {}
    for layer, a in enumerate(maps):
        band = np.zeros((len(bi), a.shape[1]))
        for d in range(task.rep_min - 1, task.rep_max):
            ok = ti >= d
            band[ok] += a[bi[ok], :, ti[ok], ti[ok] - d]
        near = (
            a[bi, :, ti, np.maximum(ti - 1, 0)] * (ti >= 1)[:, None]
            + a[bi, :, ti, np.maximum(ti - 2, 0)] * (ti >= 2)[:, None]
        )
        out[f"band_l{layer}"] = float(band.mean())
        out[f"offsets_1_2_l{layer}"] = float(near.mean())
        out[f"source_l{layer}"] = float(a[bi, :, ti, src[bi, ti]].mean())
    nll, mask = task._token_nll(model)
    out["loss_rep"] = float(nll[mask].mean())

    with torch.no_grad():
        T = x.shape[1]
        cmask = model.causal_mask[:T, :T]
        h0 = model.tok(x) + model.pos(torch.arange(T))
        b0, b1 = model.blocks[0], model.blocks[1]
        y0 = b0.attn(b0.ln1(h0), cmask)
        h_mid = h0 + y0
        if b0.mlp is not None:
            m0 = b0.mlp(b0.ln2(h_mid))
            h1 = h_mid + m0
            rms_m0 = float(torch.sqrt((m0[rep_t] ** 2).mean()))
        else:
            h1 = h_mid
            rms_m0 = 0.0
        y1 = b1.attn(b1.ln1(h1), cmask)
        if b1.mlp is not None:
            m1 = b1.mlp(b1.ln2(h1 + y1))
            rms_m1 = float(torch.sqrt((m1[rep_t] ** 2).mean()))
        else:
            rms_m1 = 0.0

    rms_emb = float(torch.sqrt((h0[rep_t] ** 2).mean()))
    rms_y0 = float(torch.sqrt((y0[rep_t] ** 2).mean()))
    rms_y1 = float(torch.sqrt((y1[rep_t] ** 2).mean()))
    out["rms_emb"] = rms_emb
    out["rms_y0"] = rms_y0
    out["rms_y1"] = rms_y1
    out["r1"] = rms_y0 / rms_emb if rms_emb > 0 else float("nan")
    out["rms_m0"] = rms_m0
    out["rms_m1"] = rms_m1
    return out


def make_job(spec: CellSpec, seed: int, n_head: int, *, steps_override: int | None = None,
             name_override: str | None = None) -> dict[str, Any]:
    steps = int(steps_override if steps_override is not None else spec.steps)
    cfg = {
        "vocab": 64,
        "seq_len": 64,
        "rep_min": spec.rep_min,
        "rep_max": 32,
        "layout": "single",
        "d_model": spec.d_model,
        "attn_only": spec.attn_only,
        "mlp_ratio": spec.mlp_ratio,
        "probe_size": 128,
        "n_head": n_head,
        "n_layer": 2,
        "p_repeat": 1.0,
        "data_seed": seed,
    }
    train = {
        "steps": steps,
        "batch_size": spec.batch_size,
        "optimizer": spec.optimizer,
        "lr": spec.lr,
        "weight_decay": spec.weight_decay,
        "warmup_steps": spec.warmup_steps,
        "lr_schedule": spec.lr_schedule,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    name = name_override or f"{spec.cell_id}_h{n_head}_s{seed}"
    return {
        "name": name,
        "cell_id": spec.cell_id,
        "suite": spec.suite,
        "group": spec.group,
        "role": spec.role,
        "seed": seed,
        "n_head": n_head,
        "cfg": cfg,
        "train": train,
        "lr_scale": dict(spec.lr_scale) if spec.lr_scale else None,
        "init_scale": dict(spec.init_scale) if spec.init_scale else None,
        "out_scale": dict(spec.out_scale) if spec.out_scale else None,
        "out_switch": dict(spec.out_switch) if spec.out_switch else None,
        "mask_window": list(spec.mask_window) if spec.mask_window else None,
        "mask_layer": spec.mask_layer,
        "mask_forbid": spec.mask_forbid,
        "data_window": dict(spec.data_window) if spec.data_window else None,
    }


def train_confirm(job: dict[str, Any], measure_at: tuple[int, ...] | None = None):
    """Confirmation training loop; bit-identical to ``relay_band_pilot.train`` when new options are off."""
    import torch

    from tycho.tasks import make_task
    from tycho.tasks.base import torch_generator
    from tycho.train.runner import LeanOptimizer
    from tycho.train.schedule import lr_at
    from tycho.train.spec import TrainConfig

    torch.set_num_threads(1)
    steps = int(job["train"]["steps"])
    at = measure_at if measure_at is not None else measure_steps(steps)
    at_set = set(at)

    tc = TrainConfig.from_dict(job["train"])
    task = make_task("induction", job["cfg"], job["seed"])
    win_cfg = job.get("data_window")
    win_task = (
        make_task("induction", {**job["cfg"], "rep_min": int(win_cfg["rep_min"])}, job["seed"])
        if win_cfg
        else None
    )

    model = task.build_model()
    if job.get("init_scale"):
        groups = weight_groups(model)
        with torch.no_grad():
            for grp, factor in job["init_scale"].items():
                for p, rows in groups[grp]:
                    p[rows].mul_(float(factor))

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = LeanOptimizer(params, tc)
    gen = torch_generator(job["seed"], "data")

    # Attention mask hook (identical to relay_band_pilot.train)
    mask_layer = int(job.get("mask_layer", 0))
    mask_forbid = str(job.get("mask_forbid", "far"))
    mw = job.get("mask_window")
    block_from, block_until = (int(mw[0]), int(mw[1])) if mw else (0, 0)
    attn = model.blocks[mask_layer].attn
    plain = attn.forward
    idx = torch.arange(model.max_len)
    offset = idx[:, None] - idx[None, :]
    band = (offset >= task.rep_min - 1) & (offset <= task.rep_max - 1)
    allowed = {"far": offset < NEAR, "band": ~band, "sham": (offset < NEAR) | band}[mask_forbid]
    blocked = {"on": False}

    def restricted(x, mask):
        return plain(x, mask & allowed[: mask.shape[0], : mask.shape[1]] if blocked["on"] else mask)

    attn.forward = restricted

    # Fixed attention output scaling hook
    factors = {int(lyr): float(factor) for lyr, factor in (job.get("out_scale") or {}).items()}
    for lyr in factors:
        inner = model.blocks[lyr].attn.forward
        model.blocks[lyr].attn.forward = (lambda x, mask, inner=inner, lyr=lyr: factors[lyr] * inner(x, mask))

    # Per-group learning rate scale (post-step rescaling; exact for Adam/AdamW w/o decay mismatch)
    lr_scale = job.get("lr_scale")
    scale = None
    if lr_scale:
        scale = [torch.full_like(p, float(lr_scale.get("all", 1.0))) for p in params]
        where = {id(p): i for i, p in enumerate(params)}
        groups = weight_groups(model)
        for grp, factor in lr_scale.items():
            if grp != "all":
                for p, rows in groups[grp]:
                    scale[where[id(p)]][rows] = float(factor)

    out_switch = job.get("out_switch")
    use_schedule = (tc.warmup_steps > 0) or (tc.lr_schedule != "constant")

    stats: dict[int, dict[str, float]] = {}
    for step in range(steps + 1):
        if step in at_set:
            blocked["on"] = False
            model.eval()
            stats[step] = confirm_stats(model, task)
            model.train()
        if step == steps:
            break
        if out_switch and step == int(out_switch["step"]):
            where_p = {id(p): i for i, p in enumerate(params)}
            for lyr, factor in factors.items():
                if out_switch["fold"] and opt.exp_avg is None:
                    raise ValueError("out_switch with fold needs Adam")
                if out_switch["fold"]:
                    a = model.blocks[lyr].attn
                    with torch.no_grad():
                        for w in (a.proj.weight, a.proj.bias):
                            w.mul_(factor)
                            opt.exp_avg[where_p[id(w)]].div_(factor)
                            opt.exp_avg_sq[where_p[id(w)]].div_(factor * factor)
                factors[lyr] = 1.0

        blocked["on"] = block_from <= step < block_until
        if win_task is not None and int(win_cfg["start"]) <= step < int(win_cfg["end"]):
            x, y = win_task.sample_batch(tc.batch_size, gen)
        else:
            x, y = task.sample_batch(tc.batch_size, gen)
        loss = task.loss(model(x), y)
        before = [p.detach().clone() for p in params] if scale else None
        step_lr = lr_at(step, tc) if use_schedule else tc.lr
        opt.step(list(torch.autograd.grad(loss, params, allow_unused=True, materialize_grads=True)), step_lr)
        if scale:
            with torch.no_grad():
                for p, old, f in zip(params, before, scale):
                    p.copy_(old + f * (p - old))

    blocked["on"] = False
    for lyr, factor in factors.items():
        a = model.blocks[lyr].attn
        a.forward = type(a).forward.__get__(a)
        with torch.no_grad():
            a.proj.weight.mul_(factor)
            a.proj.bias.mul_(factor)
    model.eval()
    return model, task, at, stats


def run_one_job(args: tuple[dict[str, Any], str, bool]) -> str:
    job, work_dir, overwrite = args
    import torch

    out = Path(work_dir) / f"{job['name']}.npz"
    if out.exists() and not overwrite:
        return f"cached {job['name']}"
    t0 = time.monotonic()
    model, _, at, stats = train_confirm(job)
    wall_s = time.monotonic() - t0
    arrays = {f"param/{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    for k in STAT_KEYS:
        arrays[f"stat_{k}"] = np.asarray([stats[s][k] for s in at], dtype=np.float64)
    meta = {
        **job,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cpu_model": platform.processor() or platform.machine(),
        "num_threads": 1,
        "wall_s": round(wall_s, 3),
    }
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        lc_step=np.asarray(at, dtype=np.int64),
        lc_loss_rep=arrays["stat_loss_rep"],
        meta_json=np.asarray(json.dumps(meta)),
        **arrays,
    )
    os.replace(tmp, out)
    return f"done {job['name']} ({wall_s:.1f}s, loss={stats[at[-1]]['loss_rep']:.3f}, src={stats[at[-1]]['source_l1']:.3f})"


def estimated_cost(job: dict[str, Any]) -> float:
    """Descending cost heuristic so the slowest runs start first on 86 workers (LPT scheduling)."""
    cfg = job["cfg"]
    tr = job["train"]
    d = float(cfg["d_model"])
    mlp = 2.0 if not cfg["attn_only"] else 1.0
    return (d * d) * float(tr["steps"]) * float(tr["batch_size"]) * mlp


def run_instrument_checks(work_dir: Path, jobs: int, *, skip_ic2: bool = False) -> dict[str, Any]:
    """Run IC-1 (same-machine repeatability), IC-2 (18 exploratory runs), and IC-4 (no-op bit-identity)."""
    import torch

    # Register modules in sys.modules BEFORE exec_module (required by Python 3.9 dataclasses + future annotations)
    if "relay_confirm" not in sys.modules:
        sys.modules["relay_confirm"] = sys.modules[__name__]

    # Load relay_band_pilot for IC-4 comparison
    if "relay_band_pilot" in sys.modules:
        rb = sys.modules["relay_band_pilot"]
    else:
        rb_spec = importlib.util.spec_from_file_location("relay_band_pilot", Path(__file__).with_name("relay_band_pilot.py"))
        rb = importlib.util.module_from_spec(rb_spec)
        sys.modules["relay_band_pilot"] = rb
        rb_spec.loader.exec_module(rb)

    # Load frozen classifier for IC-2 check
    if "relay_confirm_analyze" in sys.modules:
        an = sys.modules["relay_confirm_analyze"]
    else:
        an_spec = importlib.util.spec_from_file_location("relay_confirm_analyze", Path(__file__).with_name("relay_confirm_analyze.py"))
        an = importlib.util.module_from_spec(an_spec)
        sys.modules["relay_confirm_analyze"] = an
        an_spec.loader.exec_module(an)

    ic_dir = work_dir / "ic"
    ic_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"torch_version": torch.__version__, "platform": platform.platform()}

    # ---- IC-4: No-op bit-identity against relay_band_pilot.train (step 600, single_rm12_h2_s300) ----
    print("[IC-4] Checking bit-identity vs relay_band_pilot.train at step 600...", flush=True)
    base_spec = CELL_BY_ID["A1-6"]
    j_ic4 = make_job(base_spec, seed=300, n_head=2, steps_override=600, name_override="ic4_s300_h2")
    at_ic4 = (0, 100, 200, 300, 400, 600)
    m_new, _, _, st_new = train_confirm(j_ic4, measure_at=at_ic4)
    m_old, _, st_old, _ = rb.train(j_ic4, 600, at_ic4, block_from=0, block_until=0, layer=0, forbid="far")
    max_w_diff = max(
        float((m_new.state_dict()[k] - m_old.state_dict()[k]).abs().max())
        for k in m_new.state_dict()
    )
    max_s_diff = max(
        abs(st_new[s][k] - st_old[s][k])
        for s in at_ic4
        for k in ("band_l0", "band_l1", "source_l1", "offsets_1_2_l0", "loss_rep")
    )
    report["IC4"] = {"max_abs_weight_diff": max_w_diff, "max_abs_stat_diff": max_s_diff, "pass": max_w_diff == 0.0 and max_s_diff == 0.0}
    print(f"  IC-4: max_weight_diff={max_w_diff}, max_stat_diff={max_s_diff}, pass={report['IC4']['pass']}", flush=True)

    # ---- IC-1: Same-machine bit-identity (2 runs replayed twice) ----
    print("[IC-1] Running 2 repeatability pairs...", flush=True)
    ic1_specs = [
        ("ic1_base_h2_s300", make_job(CELL_BY_ID["A1-6"], 300, 2, steps_override=1500, name_override="ic1_base_a")),
        ("ic1_base_h2_s300", make_job(CELL_BY_ID["A1-6"], 300, 2, steps_override=1500, name_override="ic1_base_b")),
        ("ic1_low_h4_s300", make_job(CELL_BY_ID["A1-1"], 300, 4, steps_override=1500, name_override="ic1_low_a")),
        ("ic1_low_h4_s300", make_job(CELL_BY_ID["A1-1"], 300, 4, steps_override=1500, name_override="ic1_low_b")),
    ]
    with Pool(min(jobs, 4)) as pool:
        for msg in pool.imap_unordered(run_one_job, [(j, str(ic_dir), True) for _, j in ic1_specs]):
            print(" ", msg, flush=True)
    ic1_ok = True
    ic1_pairs = {}
    for pair_name, a_name, b_name in [("base_h2_s300", "ic1_base_a", "ic1_base_b"), ("low_h4_s300", "ic1_low_a", "ic1_low_b")]:
        with np.load(ic_dir / f"{a_name}.npz") as fa, np.load(ic_dir / f"{b_name}.npz") as fb:
            wdiff = max(float(np.abs(fa[k] - fb[k]).max()) for k in fa.files if k.startswith("param/"))
            sdiff = max(float(np.abs(fa[k] - fb[k]).max()) for k in fa.files if k.startswith("stat_"))
        ic1_pairs[pair_name] = {"max_weight_diff": wdiff, "max_stat_diff": sdiff, "bit_identical": wdiff == 0.0 and sdiff == 0.0}
        ic1_ok = ic1_ok and ic1_pairs[pair_name]["bit_identical"]
    report["IC1"] = {"pairs": ic1_pairs, "pass": ic1_ok}
    print(f"  IC-1: pass={ic1_ok}", flush=True)

    # ---- IC-2: Cross-machine agreement (18 exploratory runs: seeds 300..302, h2 & h4 on A1-6, A1-1, A3-5) ----
    if skip_ic2:
        report["IC2"] = {"skipped": True, "pass": True}
        print("[IC-2] Skipped (--ic-fast).", flush=True)
    else:
        print("[IC-2] Running 18 exploratory calibration runs (seeds 300..302, h2 & h4)...", flush=True)
        ic2_arms = [("A1-6", "base", 0), ("A1-1", "lr5e-4", 6), ("A3-5", "ov1_double", 6)]
        ic2_jobs = []
        for cid, label, _ in ic2_arms:
            sp = CELL_BY_ID[cid]
            for nh in (2, 4):
                for s in (300, 301, 302):
                    ic2_jobs.append(make_job(sp, s, nh, name_override=f"ic2_{label}_h{nh}_s{s}"))
        with Pool(min(jobs, len(ic2_jobs))) as pool:
            for msg in pool.imap_unordered(run_one_job, [(j, str(ic_dir), False) for j in ic2_jobs]):
                print(" ", msg, flush=True)

        smp12 = an.get_sample(12)
        flips_ts = 0
        flips_tsm = 0
        ic2_detail = {}
        for cid, label, exp_src in ic2_arms:
            arm_rows = []
            for nh in (2, 4):
                for s in (300, 301, 302):
                    name = f"ic2_{label}_h{nh}_s{s}"
                    arm_rows.append(an.classify_npz(ic_dir / f"{name}.npz", {12: smp12}))
            got_ts = sum(r["cat"] in ("T", "S") for r in arm_rows)
            got_tsm = sum(r["cat"] in ("T", "S", "M") for r in arm_rows)
            flips_ts += abs(got_ts - exp_src)
            flips_tsm += abs(got_tsm - exp_src)
            ic2_detail[label] = {
                "expected_source_of_6": exp_src,
                "got_T_S": got_ts,
                "got_T_S_M": got_tsm,
                "cats": [r["cat"] for r in arm_rows],
            }
        report["IC2"] = {
            "arms": ic2_detail,
            "total_flips_T_S": flips_ts,
            "total_flips_T_S_M": flips_tsm,
            "pass": max(flips_ts, flips_tsm) <= 1,
        }
        print(f"  IC-2: flips(T+S)={flips_ts}, flips(T+S+M)={flips_tsm}, pass={report['IC2']['pass']}", flush=True)
    report["all_pass"] = bool(report["IC1"]["pass"] and report["IC2"]["pass"] and report["IC4"]["pass"])

    out_json = ic_dir / "ic_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[IC] Summary saved to {out_json} | all_pass={report['all_pass']}", flush=True)
    return report


def run_smoke_tests(work_dir: Path, jobs: int) -> None:
    """Fast 150-step smoke test touching every code path (AdamW, warmup, MLP, proj0 lr/init, data_window, 8h, d=256, out_switch)."""
    smoke_dir = work_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    picks = [
        ("A1-6", 1000, 2),       # base
        ("A3-1", 1000, 2),       # out_scale
        ("A4-1", 1000, 2),       # qk0 lr_scale
        ("A5-1", 1000, 2),       # far mask
        ("A5-3", 1000, 2),       # band mask L1 rep_min 20
        ("A6-3", 1000, 8),       # 8 heads + out_scale
        ("A6-4", 1000, 2),       # out_switch fold (override step=100)
        ("A6-5", 1000, 2),       # out_switch drop (override step=100)
        ("A6-7", 1000, 2),       # proj0 lr_scale
        ("A6-8", 1000, 2),       # proj0 init_scale
        ("B1-WU-mid", 1000, 2),  # AdamW + warmup
        ("B2-mid-c05", 1000, 2), # MLP + out_scale
        ("B3-early", 1000, 2),   # data_window
        ("C-d256-mid", 1000, 4), # d_model=256
    ]
    sjobs = []
    for cid, seed, nh in picks:
        j = make_job(CELL_BY_ID[cid], seed, nh, steps_override=150, name_override=f"smoke_{cid}_h{nh}_s{seed}")
        if j.get("out_switch"):
            j["out_switch"] = {**j["out_switch"], "step": 100}
        if j.get("data_window"):
            j["data_window"] = {"start": 0, "end": 75, "rep_min": 20}
        sjobs.append(j)
    print(f"[SMOKE] Running {len(sjobs)} 150-step jobs across all code paths...", flush=True)
    with Pool(min(jobs, len(sjobs))) as pool:
        for msg in pool.imap_unordered(run_one_job, [(j, str(smoke_dir), True) for j in sjobs]):
            print(" ", msg, flush=True)
    # Verify every npz loads cleanly and has finite stats
    for j in sjobs:
        with np.load(smoke_dir / f"{j['name']}.npz") as f:
            for k in STAT_KEYS:
                arr = f[f"stat_{k}"]
                assert np.all(np.isfinite(arr)), f"Non-finite stat_{k} in {j['name']}"
            if not j["cfg"]["attn_only"]:
                assert f["stat_rms_m0"][-1] > 0.0 and f["stat_rms_m1"][-1] > 0.0, f"MLP RMS zero in {j['name']}"
    print("[SMOKE] All 14 code paths passed with finite metrics!", flush=True)


def print_cell_list() -> None:
    by_suite: dict[str, list[CellSpec]] = {"A": [], "B": [], "C": []}
    for sp in CELLS:
        by_suite[sp.suite].append(sp)
    total_runs = 0
    for s in ("A", "B", "C"):
        n_cells = len(by_suite[s])
        n_runs = sum(len(sp.run_pairs()) for sp in by_suite[s])
        total_runs += n_runs
        print(f"Suite {s}: {n_cells} cells, {n_runs} runs")
        for sp in by_suite[s]:
            heads = f"{sp.fixed_heads}h x 24" if sp.fixed_heads else "2h x 12 + 4h x 12"
            extras = []
            if sp.rep_min != 12:
                extras.append(f"rm={sp.rep_min}")
            if sp.d_model != 64:
                extras.append(f"d={sp.d_model}")
            if sp.steps != 12_000:
                extras.append(f"steps={sp.steps}")
            if not sp.attn_only:
                extras.append("MLP")
            if sp.optimizer != "adam":
                extras.append(f"{sp.optimizer}(wd={sp.weight_decay})")
            if sp.warmup_steps:
                extras.append(f"warmup={sp.warmup_steps}")
            if sp.out_scale:
                extras.append(f"c={sp.out_scale}")
            if sp.out_switch:
                extras.append(f"switch={sp.out_switch}")
            if sp.lr_scale:
                extras.append(f"lr_scale={sp.lr_scale}")
            if sp.init_scale:
                extras.append(f"init_scale={sp.init_scale}")
            if sp.mask_window:
                extras.append(f"mask=L{sp.mask_layer}:{sp.mask_forbid}{sp.mask_window}")
            if sp.data_window:
                extras.append(f"data_win={sp.data_window}")
            ext = " | " + ", ".join(extras) if extras else ""
            print(f"  {sp.cell_id:15s} [{sp.role:11s}] lr={sp.lr:g} B={sp.batch_size:<3d} ({heads}){ext}")
    print(f"TOTAL: {len(CELLS)} cells, {total_runs} runs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="88-core pre-registered confirmation runner (76 cells, 1,824 runs).")
    ap.add_argument("--suite", choices=("ic", "A", "B", "C", "all"), help="Which suite to run")
    ap.add_argument("--cells", nargs="+", help="Run specific cell IDs (main seeds 1000..1023)")
    ap.add_argument("--topup", nargs="+", help="Run §2.5 top-up reserve seeds (1024..1047) for specific cell IDs")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_confirm", help="Directory for cached run .npz files")
    ap.add_argument("--jobs", type=int, default=86, help="Number of single-thread worker processes")
    ap.add_argument("--list", action="store_true", help="Print the 76-cell table and exit")
    ap.add_argument("--smoke", action="store_true", help="Run fast 150-step smoke tests across all code paths")
    ap.add_argument("--ic-fast", action="store_true", help="Run IC-4 + IC-1 only (skip 18-run IC-2; for local verification)")
    ap.add_argument("--overwrite", action="store_true", help="Re-run even if cached .npz exists")
    args = ap.parse_args(argv)

    if args.list:
        print_cell_list()
        return 0

    args.work.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        run_smoke_tests(args.work, max(1, args.jobs))
        return 0

    if args.ic_fast:
        rep = run_instrument_checks(args.work, max(1, args.jobs), skip_ic2=True)
        return 0 if rep["all_pass"] else 2

    if args.suite == "ic":
        rep = run_instrument_checks(args.work, max(1, args.jobs), skip_ic2=False)
        return 0 if rep["all_pass"] else 2

    run_jobs: list[dict[str, Any]] = []
    if args.topup:
        for cid in args.topup:
            sp = CELL_BY_ID[cid]
            for seed, nh in sp.run_pairs(topup=True):
                run_jobs.append(make_job(sp, seed, nh))
    elif args.cells:
        for cid in args.cells:
            sp = CELL_BY_ID[cid]
            for seed, nh in sp.run_pairs(topup=False):
                run_jobs.append(make_job(sp, seed, nh))
    elif args.suite:
        suites = ("A", "B", "C") if args.suite == "all" else (args.suite,)
        for sp in CELLS:
            if sp.suite in suites:
                for seed, nh in sp.run_pairs(topup=False):
                    run_jobs.append(make_job(sp, seed, nh))
    else:
        ap.error("Specify one of --list, --smoke, --suite, --cells, or --topup")

    # Longest-processing-time-first ordering minimizes straggler tail on 86 cores
    run_jobs.sort(key=estimated_cost, reverse=True)
    todo = [j for j in run_jobs if args.overwrite or not (args.work / f"{j['name']}.npz").exists()]
    print(f"[RUN] Total selected: {len(run_jobs)} | To run: {len(todo)} | Cached: {len(run_jobs) - len(todo)} | Workers: {args.jobs}", flush=True)
    if not todo:
        return 0

    t0 = time.monotonic()
    done = 0
    with Pool(processes=max(1, args.jobs)) as pool:
        for msg in pool.imap_unordered(run_one_job, [(j, str(args.work), args.overwrite) for j in todo]):
            done += 1
            if done % 20 == 0 or done == len(todo) or done <= 5:
                elapsed = time.monotonic() - t0
                print(f"[{done:4d}/{len(todo):4d} | {elapsed/60:5.1f}m] {msg}", flush=True)
    print(f"[RUN] Finished {done} runs in {(time.monotonic() - t0)/60:.1f} min.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
