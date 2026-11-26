"""Frozen Phase 2 pre-registered confirmation runner & analyzer (16 cells, 384 fresh runs).

Specification: ``docs/results/relay_confirmation_phase2_preregistration.md``
Motivated by independent audit ``notes/relay_confirm_audit_2026-11-25.md`` (§6.1).

Matrix (16 cells x 24 fresh seeds = 384 runs, seeds 1048..1059 h2 + 1060..1071 h4):
  * M1 (5 cells, 120 runs): 2L abs-pos 4x MLP + AdamW(0.1) + 200wu, d=64, lr=1e-3, 12k steps:
      base (1,1), attn05 (0.5,1), mlp05 (1,0.5), both05 (0.5,0.5), both025 (0.25,0.25).
  * M2a (2 cells, 48 runs): 2L abs-pos 4x MLP + plain Adam (no wu), d=64, lr=5e-4, 12k steps:
      base (1,1), both025 (0.25,0.25).
  * M2b (2 cells, 48 runs, blind exploratory): 2L abs-pos 4x MLP + AdamW(0.1) + 200wu, d=128, lr=5e-4, 12k steps:
      base (1,1), both025 (0.25,0.25).
  * R1 (4 cells, 96 runs): 2L RoPE boundary (attn-only 1e-3 & 2e-3 blind; 4x MLP+AdamW+wu 2e-3 base & both025).
  * D1 (3 cells, 72 runs): 4L 4x MLP + AdamW(0.1) + 200wu, d=64, lr=1e-3, 24,000 steps:
      D1-abs-24k, D1-rope-24k, D1-abs-l01_05-24k (exploratory 24k truncation check).

Unified behavioral criterion (audit §3.2):
  ``B_T`` (Behavioral Textbook) = ``loss_rep < 0.3`` AND ``deep_decoy_over_clean >= 0.9``
  on the fixed 2,000-sequence evaluation set (seed 777).

Cache directory: ``work/relay_confirm_phase2/`` (strictly isolated from Phase 1 ``work/relay_confirm/``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if "relay_confirm" in sys.modules:
    rc = sys.modules["relay_confirm"]
else:
    _rc_spec = importlib.util.spec_from_file_location("relay_confirm", Path(__file__).with_name("relay_confirm.py"))
    rc = importlib.util.module_from_spec(_rc_spec)
    sys.modules["relay_confirm"] = rc
    _rc_spec.loader.exec_module(rc)

if "relay_confirm_analyze" in sys.modules:
    an = sys.modules["relay_confirm_analyze"]
else:
    _an_spec = importlib.util.spec_from_file_location("relay_confirm_analyze", Path(__file__).with_name("relay_confirm_analyze.py"))
    an = importlib.util.module_from_spec(_an_spec)
    sys.modules["relay_confirm_analyze"] = an
    _an_spec.loader.exec_module(an)

fp = an.fp

SEEDS_P2_H2 = tuple(range(1048, 1060))  # 12 fresh seeds -> 2 heads
SEEDS_P2_H4 = tuple(range(1060, 1072))  # 12 fresh seeds -> 4 heads
STAT_KEYS = ("band_l0", "band_l1", "source_l0", "source_l1", "max_src_upper_head", "loss_rep", "rms_emb", "rms_y0", "rms_m0", "r1")
LOSS_T_THRESH = 0.3
DECOY_T_THRESH = 0.9


# -------------------------------------------------------------------------------------- RoPE helpers
def rope_cos_sin(seq_len: int, d_head: int, base: float = 10000.0):
    import torch

    assert d_head % 2 == 0, f"RoPE requires even d_head, got {d_head}"
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x, cos, sin):
    import torch

    T = x.shape[2]
    c = cos[:, :, :T, :]
    s = sin[:, :, :T, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


def patch_model_for_rope(model) -> None:
    """Disable learned absolute pos (zeroed & frozen) and apply standard RoPE to Q and K in every block."""
    import torch

    with torch.no_grad():
        model.pos.weight.zero_()
    model.pos.weight.requires_grad_(False)

    for block in model.blocks:
        attn = block.attn
        cos, sin = rope_cos_sin(model.max_len, attn.d_head)

        def _rope_attn_forward(x, mask, attn=attn, cos=cos, sin=sin):
            B, T, C = x.shape
            q, k, v = attn.qkv(x).split(C, dim=-1)
            q = q.reshape(B, T, attn.n_head, attn.d_head).transpose(1, 2)
            k = k.reshape(B, T, attn.n_head, attn.d_head).transpose(1, 2)
            v = v.reshape(B, T, attn.n_head, attn.d_head).transpose(1, 2)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.d_head))
            att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            if attn.record is not None:
                attn.record.append(att.detach())
            y = att @ v
            if attn.head_scale is not None:
                y = y * attn.head_scale.to(device=y.device, dtype=y.dtype)[None, :, None, None]
            return attn.proj(y.transpose(1, 2).reshape(B, T, C))

        attn.forward = _rope_attn_forward


# -------------------------------------------------------------------------------------- cell matrix
@dataclass(frozen=True)
class P2Cell:
    cell_id: str
    group: str
    role: str  # "primary", "secondary", "exploratory"
    n_layer: int = 2
    use_rope: bool = False
    attn_only: bool = False
    d_model: int = 64
    lr: float = 1e-3
    batch_size: int = 32
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    warmup_steps: int = 200
    steps: int = 12_000
    rep_min: int = 12
    attn_out_scale: dict[int, float] | None = None
    mlp_out_scale: dict[int, float] | None = None
    note: str = ""


P2_CELLS: list[P2Cell] = [
    # ---- M1: 2L abs MLP joint Layer-0 residual scaling (5 cells, 120 runs) ----
    P2Cell("M1-base", "M1", "primary", note="2L abs 4x MLP AdamW+wu unscaled"),
    P2Cell("M1-attn05", "M1", "primary", attn_out_scale={0: 0.5}, note="Attn0 x0.5 only"),
    P2Cell("M1-mlp05", "M1", "primary", mlp_out_scale={0: 0.5}, note="MLP0 x0.5 only"),
    P2Cell("M1-both05", "M1", "primary", attn_out_scale={0: 0.5}, mlp_out_scale={0: 0.5}, note="Attn0+MLP0 x0.5"),
    P2Cell("M1-both025", "M1", "primary", attn_out_scale={0: 0.25}, mlp_out_scale={0: 0.25}, note="Attn0+MLP0 x0.25"),

    # ---- M2a: 2L abs MLP at second LR/optimizer (plain Adam, lr=5e-4, no wu; 2 cells, 48 runs) ----
    P2Cell("M2a-lr5e4-base", "M2a", "primary", lr=5e-4, optimizer="adam", weight_decay=0.0, warmup_steps=0, note="plain Adam 5e-4 unscaled"),
    P2Cell("M2a-lr5e4-both025", "M2a", "primary", lr=5e-4, optimizer="adam", weight_decay=0.0, warmup_steps=0, attn_out_scale={0: 0.25}, mlp_out_scale={0: 0.25}, note="plain Adam 5e-4 x0.25"),

    # ---- M2b: 2L abs MLP at second width d=128 (blind exploratory; 2 cells, 48 runs) ----
    P2Cell("M2b-d128-base", "M2b", "exploratory", d_model=128, lr=5e-4, note="blind d=128 5e-4 AdamW+wu unscaled"),
    P2Cell("M2b-d128-both025", "M2b", "exploratory", d_model=128, lr=5e-4, attn_out_scale={0: 0.25}, mlp_out_scale={0: 0.25}, note="blind d=128 5e-4 AdamW+wu x0.25"),

    # ---- R1: 2L RoPE boundary (4 cells, 96 runs) ----
    P2Cell("R1-attn-1e3", "R1", "secondary", use_rope=True, attn_only=True, lr=1e-3, optimizer="adam", weight_decay=0.0, warmup_steps=0, note="2L RoPE attn-only 1e-3"),
    P2Cell("R1-attn-2e3", "R1", "exploratory", use_rope=True, attn_only=True, lr=2e-3, optimizer="adam", weight_decay=0.0, warmup_steps=0, note="blind 2L RoPE attn-only 2e-3"),
    P2Cell("R1-mlp-2e3-base", "R1", "primary", use_rope=True, attn_only=False, lr=2e-3, note="2L RoPE MLP 2e-3 unscaled"),
    P2Cell("R1-mlp-2e3-both025", "R1", "primary", use_rope=True, attn_only=False, lr=2e-3, attn_out_scale={0: 0.25}, mlp_out_scale={0: 0.25}, note="2L RoPE MLP 2e-3 x0.25"),

    # ---- D1: 4L abs vs 4L RoPE at 24,000 steps (3 cells, 72 runs) ----
    P2Cell("D1-abs-24k", "D1", "primary", n_layer=4, use_rope=False, steps=24_000, note="4L abs 24k"),
    P2Cell("D1-rope-24k", "D1", "primary", n_layer=4, use_rope=True, steps=24_000, note="4L RoPE 24k"),
    P2Cell("D1-abs-l01_05-24k", "D1", "exploratory", n_layer=4, use_rope=False, steps=24_000, attn_out_scale={0: 0.5, 1: 0.5}, mlp_out_scale={0: 0.5, 1: 0.5}, note="exploratory 4L abs L0+L1 x0.5 24k"),
]

CELL_MAP: dict[str, P2Cell] = {c.cell_id: c for c in P2_CELLS}


def make_job(sp: P2Cell, seed: int, n_head: int, *, steps_override: int | None = None, name_override: str | None = None) -> dict[str, Any]:
    steps = int(steps_override if steps_override is not None else sp.steps)
    return {
        "name": name_override or f"{sp.cell_id}_h{n_head}_s{seed}",
        "cell_id": sp.cell_id,
        "group": sp.group,
        "role": sp.role,
        "use_rope": sp.use_rope,
        "seed": seed,
        "n_head": n_head,
        "cfg": {
            "vocab": 64,
            "seq_len": 64,
            "rep_min": sp.rep_min,
            "rep_max": 32,
            "layout": "single",
            "d_model": sp.d_model,
            "attn_only": sp.attn_only,
            "mlp_ratio": 4,
            "probe_size": 128,
            "n_head": n_head,
            "n_layer": sp.n_layer,
            "p_repeat": 1.0,
            "data_seed": seed,
        },
        "train": {
            "steps": steps,
            "batch_size": sp.batch_size,
            "optimizer": sp.optimizer,
            "lr": sp.lr,
            "weight_decay": sp.weight_decay,
            "warmup_steps": sp.warmup_steps,
            "lr_schedule": "constant",
            "beta1": 0.9,
            "beta2": 0.999,
        },
        "attn_out_scale": dict(sp.attn_out_scale) if sp.attn_out_scale else None,
        "mlp_out_scale": dict(sp.mlp_out_scale) if sp.mlp_out_scale else None,
    }


def measure_stats(model, task) -> dict[str, float]:
    import torch

    pr = task._probe
    x = pr["x"]
    rep_t = pr["rep"]
    rep, src = rep_t.numpy(), pr["src"].numpy()
    maps = [a.numpy() for a in model.attention_maps(x)]
    bi, ti = np.nonzero(rep)

    out: dict[str, float] = {}
    for layer_idx, tag in [(0, "l0"), (1, "l1")]:
        a = maps[layer_idx]
        band = np.zeros((len(bi), a.shape[1]))
        for d in range(task.rep_min - 1, task.rep_max):
            ok = ti >= d
            band[ok] += a[bi[ok], :, ti[ok], ti[ok] - d]
        out[f"band_{tag}"] = float(band.mean())
        out[f"source_{tag}"] = float(a[bi, :, ti, src[bi, ti]].mean())

    out["max_src_upper_head"] = max(
        float(maps[l][bi, :, ti, src[bi, ti]].mean(axis=0).max())
        for l in range(1, len(maps))
    )
    nll, mask = task._token_nll(model)
    out["loss_rep"] = float(nll[mask].mean())

    with torch.no_grad():
        T = x.shape[1]
        cmask = model.causal_mask[:T, :T]
        h0 = model.tok(x) + model.pos(torch.arange(T))
        b0 = model.blocks[0]
        y0 = b0.attn(b0.ln1(h0), cmask)
        h_mid = h0 + y0
        if b0.mlp is not None:
            m0 = b0.mlp(b0.ln2(h_mid))
            rms_m0 = float(torch.sqrt((m0[rep_t] ** 2).mean()))
        else:
            rms_m0 = 0.0
    rms_emb = float(torch.sqrt((h0[rep_t] ** 2).mean()))
    rms_y0 = float(torch.sqrt((y0[rep_t] ** 2).mean()))
    out["rms_emb"] = rms_emb
    out["rms_y0"] = rms_y0
    out["rms_m0"] = rms_m0
    out["r1"] = rms_y0 / rms_emb if rms_emb > 0 else float("nan")
    return out


def train_p2(job: dict[str, Any], *, return_pre_fold_logits: bool = False):
    import torch

    from tycho.tasks import make_task
    from tycho.tasks.base import torch_generator
    from tycho.train.runner import LeanOptimizer
    from tycho.train.schedule import lr_at
    from tycho.train.spec import TrainConfig

    torch.set_num_threads(1)
    steps = int(job["train"]["steps"])
    at = rc.measure_steps(steps)
    at_set = set(at)

    tc = TrainConfig.from_dict(job["train"])
    task = make_task("induction", job["cfg"], job["seed"])
    model = task.build_model()
    if job["use_rope"]:
        patch_model_for_rope(model)

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = LeanOptimizer(params, tc)
    gen = torch_generator(job["seed"], "data")

    a_factors = {int(lyr): float(factor) for lyr, factor in (job.get("attn_out_scale") or {}).items()}
    for lyr in a_factors:
        a = model.blocks[lyr].attn
        inner = a.forward
        a.forward = (lambda x, mask, inner=inner, lyr=lyr: a_factors[lyr] * inner(x, mask))

    m_factors = {int(lyr): float(factor) for lyr, factor in (job.get("mlp_out_scale") or {}).items()}
    for lyr in m_factors:
        m = model.blocks[lyr].mlp
        inner_m = m.forward
        m.forward = (lambda x, inner_m=inner_m, lyr=lyr: m_factors[lyr] * inner_m(x))

    use_schedule = (tc.warmup_steps > 0) or (tc.lr_schedule != "constant")
    stats: dict[int, dict[str, float]] = {}
    for step in range(steps + 1):
        if step in at_set:
            model.eval()
            stats[step] = measure_stats(model, task)
            model.train()
        if step == steps:
            break
        x, y = task.sample_batch(tc.batch_size, gen)
        loss = task.loss(model(x), y)
        step_lr = lr_at(step, tc) if use_schedule else tc.lr
        opt.step(list(torch.autograd.grad(loss, params, allow_unused=True, materialize_grads=True)), step_lr)

    pre_fold = None
    if return_pre_fold_logits:
        model.eval()
        with torch.no_grad():
            pre_fold = model(task._probe["x"][:8]).detach().clone()

    # Exact post-training fold into proj / mlp[2] and restore unscaled forward
    for lyr, factor in a_factors.items():
        a = model.blocks[lyr].attn
        if not job["use_rope"]:
            a.forward = type(a).forward.__get__(a)
        with torch.no_grad():
            a.proj.weight.mul_(factor)
            a.proj.bias.mul_(factor)
    for lyr, factor in m_factors.items():
        m = model.blocks[lyr].mlp
        m.forward = type(m).forward.__get__(m)
        with torch.no_grad():
            m[2].weight.mul_(factor)
            m[2].bias.mul_(factor)
    if job["use_rope"]:
        patch_model_for_rope(model)
    model.eval()
    if return_pre_fold_logits:
        return model, task, at, stats, pre_fold
    return model, task, at, stats


def run_one(args: tuple[dict[str, Any], str, bool]) -> str:
    job, work_dir, overwrite = args
    import torch

    out = Path(work_dir) / f"{job['name']}.npz"
    if out.exists() and not overwrite:
        return f"cached {job['name']}"
    t0 = time.monotonic()
    model, _, at, stats = train_p2(job)
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
    return f"done {job['name']} ({wall_s:.1f}s, loss={stats[at[-1]]['loss_rep']:.3f}, max_src={stats[at[-1]]['max_src_upper_head']:.3f})"


def estimated_cost(job: dict[str, Any]) -> float:
    cfg = job["cfg"]
    d = float(cfg["d_model"])
    mlp = 2.2 if not cfg["attn_only"] else 1.0
    return float(cfg["n_layer"]) * (d * d) * float(job["train"]["steps"]) * mlp * (1.2 if job["n_head"] == 4 else 1.0)


# -------------------------------------------------------------------------------------- IC-P2 checks
def verify_math_and_fold() -> dict[str, Any]:
    """Local unit check using ONLY seed 300 (never touches confirmatory seeds 1048..1071)."""
    import torch

    cos, sin = rope_cos_sin(64, 32)
    torch.manual_seed(0)
    q = torch.randn(1, 2, 1, 32).expand(1, 2, 64, 32).contiguous()
    k = torch.randn(1, 2, 1, 32).expand(1, 2, 64, 32).contiguous()
    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    norm_err = float((qr.norm(dim=-1) - q.norm(dim=-1)).abs().max())
    rel_err = abs((qr[0, 0, 10] * kr[0, 0, 5]).sum().item() - (qr[0, 0, 42] * kr[0, 0, 37]).sum().item())
    assert norm_err < 1e-5 and rel_err < 1e-5

    fold_diffs = {}
    for cid in ("M1-both05", "R1-mlp-2e3-both025", "D1-abs-l01_05-24k"):
        j = make_job(CELL_MAP[cid], 300, 2, steps_override=4, name_override=f"verify_{cid}")
        model, task, _, _, pre_fold = train_p2(j, return_pre_fold_logits=True)
        x = task._probe["x"][:8]
        with torch.no_grad():
            post_fold = model(x)
        # Reload fresh model from state_dict (as frozen analyzer does)
        m2 = task.build_model()
        m2.load_state_dict(model.state_dict())
        if j["use_rope"]:
            patch_model_for_rope(m2)
            assert float(m2.pos.weight.abs().max()) == 0.0
        m2.eval()
        with torch.no_grad():
            reloaded = m2(x)
        d_fold = float((pre_fold - post_fold).abs().max())
        d_reload = float((post_fold - reloaded).abs().max())
        assert d_fold < 1e-5 and d_reload == 0.0 and torch.isfinite(post_fold).all(), f"{cid}: fold={d_fold}, reload={d_reload}"
        fold_diffs[cid] = {"pre_vs_post_fold": d_fold, "post_vs_reload": d_reload}
    res = {"rope_norm_err": norm_err, "rope_rel_err": rel_err, "fold_diffs": fold_diffs, "pass": True}
    print(f"[IC-P2-MATH] PASSED: norm_err={norm_err:.2e}, rel_err={rel_err:.2e}, folds={fold_diffs}", flush=True)
    return res


def run_ic_p2(work_dir: Path, jobs: int) -> dict[str, Any]:
    import torch

    ic_dir = work_dir / "ic"
    ic_dir.mkdir(parents=True, exist_ok=True)
    math_res = verify_math_and_fold()

    print("[IC-P2-REPEAT] Checking same-machine bit-identity on 2L abs MLP & 4L RoPE MLP (seed 300, 600 steps)...", flush=True)
    specs = [
        ("m1_base_h2_s300", make_job(CELL_MAP["M1-base"], 300, 2, steps_override=600, name_override="ic_m1_a")),
        ("m1_base_h2_s300", make_job(CELL_MAP["M1-base"], 300, 2, steps_override=600, name_override="ic_m1_b")),
        ("d1_rope_h4_s300", make_job(CELL_MAP["D1-rope-24k"], 300, 4, steps_override=600, name_override="ic_d1_a")),
        ("d1_rope_h4_s300", make_job(CELL_MAP["D1-rope-24k"], 300, 4, steps_override=600, name_override="ic_d1_b")),
    ]
    with Pool(min(jobs, 4)) as pool:
        for msg in pool.imap_unordered(run_one, [(j, str(ic_dir), True) for _, j in specs]):
            print(" ", msg, flush=True)

    pairs = {}
    ok = True
    for label, a, b in [("m1_base_h2_s300", "ic_m1_a", "ic_m1_b"), ("d1_rope_h4_s300", "ic_d1_a", "ic_d1_b")]:
        with np.load(ic_dir / f"{a}.npz") as fa, np.load(ic_dir / f"{b}.npz") as fb:
            wdiff = max(float(np.abs(fa[k] - fb[k]).max()) for k in fa.files if k.startswith("param/"))
            sdiff = max(float(np.abs(fa[k] - fb[k]).max()) for k in fa.files if k.startswith("stat_"))
        bit_id = (wdiff == 0.0 and sdiff == 0.0)
        pairs[label] = {"max_weight_diff": wdiff, "max_stat_diff": sdiff, "bit_identical": bit_id}
        ok = ok and bit_id

    rep = {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "math_and_fold": math_res,
        "repeatability": {"pairs": pairs, "pass": ok},
        "all_pass": bool(math_res["pass"] and ok),
    }
    (ic_dir / "ic_p2_report.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"[IC-P2] all_pass={rep['all_pass']} | pairs={pairs}", flush=True)
    return rep


# -------------------------------------------------------------------------------------- frozen classifier & analyzer
def classify_p2_npz(npz_path: Path) -> dict[str, Any]:
    import torch

    from tycho.tasks.induction import Induction

    with np.load(npz_path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta_json"]))
        state = {k[6:]: torch.as_tensor(f[k]) for k in f.files if k.startswith("param/")}
        steps = f["lc_step"].astype(np.int64)
        stats = {k: f[f"stat_{k}"].astype(np.float64) for k in STAT_KEYS if f"stat_{k}" in f.files}

    cfg = meta["cfg"]
    seed = int(meta["seed"])
    rep_min = int(cfg["rep_min"])
    nh = int(cfg["n_head"])
    n_layer = int(cfg["n_layer"])

    task = Induction(cfg, seed)
    model = task.build_model()
    model.load_state_dict(state)
    if meta.get("use_rope"):
        patch_model_for_rope(model)
    model.eval()

    smp = an.get_sample(rep_min)
    probs = fp.probabilities(model, smp["x"])
    sc = fp.score(fp.true_token(probs, smp), smp)

    # Probe-set attention summary across all upper layers (1 .. n_layer-1)
    pr = task._probe
    x_pr, rep_pr, src_pr = pr["x"], pr["rep"].numpy(), pr["src"].numpy()
    maps = [a.numpy() for a in model.attention_maps(x_pr)]
    bi, ti = np.nonzero(rep_pr)
    lag = ti - src_pr[bi, ti] + 1
    lags = sorted(set(lag.tolist()))

    best_head = None
    best_src_att = 0.0
    for lyr in range(1, n_layer):
        for h in range(nh):
            a = maps[lyr][:, h]
            src_att = float(a[bi, ti, src_pr[bi, ti]].mean())
            if src_att > best_src_att:
                best_src_att = src_att

            def profile(sel: np.ndarray) -> np.ndarray:
                prof = np.full(len(an.DELTAS), np.nan)
                for k, dl in enumerate(an.DELTAS):
                    j = src_pr[bi[sel], ti[sel]] + dl
                    ok = (j >= 0) & (j <= ti[sel])
                    if ok.any():
                        prof[k] = a[bi[sel][ok], ti[sel][ok], j[ok]].mean()
                return prof

            by_lag = np.array([profile(lag == L) for L in lags])
            top_by_lag = [an.DELTAS[int(np.nanargmax(row))] for row in by_lag]
            modal, _ = Counter(top_by_lag).most_common(1)[0]
            med_att = float(np.nanmedian(by_lag[:, an.DELTAS.index(modal)]))
            if best_head is None or med_att > best_head["med_att"]:
                best_head = {"layer": lyr, "head": f"l{lyr}h{h}", "delta": int(modal), "med_att": med_att}

    relay_delta = int(best_head["delta"])
    loss_rep = float(sc["loss_rep"])
    decoy = float(sc["deep_decoy_over_clean"])
    source_l1_end = float(stats["source_l1"][-1])

    # Unified behavioral textbook criterion (audit §3.2): decoy >= 0.9 AND loss < 0.3
    is_bt = bool(math.isfinite(loss_rep) and math.isfinite(decoy) and loss_rep < LOSS_T_THRESH and decoy >= DECOY_T_THRESH)
    if not math.isfinite(loss_rep) or loss_rep >= 2.0:
        cat = "U"
    elif is_bt:
        cat = "T_beh"
    elif relay_delta != 0:
        cat = "R"
    elif (source_l1_end > 0.5 if n_layer == 2 else best_src_att > 0.5):
        cat = "S"
    else:
        cat = "M"

    lr = float(meta["train"]["lr"])
    r1_target = int(round(0.3 / lr))
    r1_idx = int(np.argmin(np.abs(steps - r1_target)))

    return {
        "name": meta["name"],
        "cell_id": meta["cell_id"],
        "group": meta["group"],
        "role": meta["role"],
        "use_rope": bool(meta.get("use_rope")),
        "n_layer": n_layer,
        "d_model": int(cfg["d_model"]),
        "lr": lr,
        "seed": seed,
        "n_head": nh,
        "cat": cat,
        "is_bt": is_bt,
        "loss_rep": loss_rep,
        "deep_decoy_over_clean": decoy,
        "relay_delta": relay_delta,
        "best_head": best_head["head"],
        "best_src_att": round(best_src_att, 4),
        "r1_aligned": float(stats["r1"][r1_idx]),
        "rms_m0_aligned": float(stats["rms_m0"][r1_idx]),
        "wall_s": meta.get("wall_s"),
    }


def _classify_worker(p_str: str) -> dict[str, Any]:
    return classify_p2_npz(Path(p_str))


def summarize_p2_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = {k: sum(r["cat"] == k for r in rows) for k in ("U", "R", "T_beh", "S", "M")}
    bt = sum(r["is_bt"] for r in rows)
    by_h = {}
    for nh in sorted({r["n_head"] for r in rows}):
        sub = [r for r in rows if r["n_head"] == nh]
        by_h[f"h{nh}"] = {
            "n": len(sub),
            "B_T": sum(r["is_bt"] for r in sub),
            "U": sum(r["cat"] == "U" for r in sub),
            "R": sum(r["cat"] == "R" for r in sub),
            "S": sum(r["cat"] == "S" for r in sub),
            "M": sum(r["cat"] == "M" for r in sub),
        }
    return {
        "n": n,
        "counts": counts,
        "B_T": bt,
        "B_T_rate": round(bt / n, 4) if n else None,
        "B_T_ci95": an.wilson_ci(bt, n),
        "by_head": by_h,
        "unstable": counts["U"] > 6,
        "loss_rep_med": an.bootstrap_median_ci([r["loss_rep"] for r in rows]),
        "r1_aligned": an.bootstrap_median_ci([r["r1_aligned"] for r in rows]),
        "rms_m0_aligned": an.bootstrap_median_ci([r["rms_m0_aligned"] for r in rows]),
    }


def paired_bt(cid_x: str, cid_y: str, by_cell: dict[str, list[dict[str, Any]]], *, head_filter: int | None = None) -> dict[str, Any]:
    rx = {r["seed"]: r for r in by_cell.get(cid_x, []) if head_filter is None or r["n_head"] == head_filter}
    ry = {r["seed"]: r for r in by_cell.get(cid_y, []) if head_filter is None or r["n_head"] == head_filter}
    common = sorted(set(rx) & set(ry))
    n = len(common)
    if n == 0:
        return {"x": cid_x, "y": cid_y, "n": 0, "verdict": "missing"}
    ux = sum(rx[s]["cat"] == "U" for s in common)
    uy = sum(ry[s]["cat"] == "U" for s in common)
    unstable = (ux > (6 if n == 24 else 3)) or (uy > (6 if n == 24 else 3))
    bx = [bool(rx[s]["is_bt"]) for s in common]
    by = [bool(ry[s]["is_bt"]) for s in common]
    kx, ky = sum(bx), sum(by)
    p_mc = an.mcnemar_exact_p(bx, by)
    verdict = "unstable" if unstable else an.three_tier(kx, ky, n)
    return {
        "x": cid_x, "y": cid_y, "n": n,
        "kx": kx, "ky": ky, "diff": kx - ky,
        "rx_R": sum(rx[s]["cat"] == "R" for s in common),
        "ry_R": sum(ry[s]["cat"] == "R" for s in common),
        "ux": ux, "uy": uy,
        "mcnemar_p": round(p_mc, 6),
        "verdict": verdict,
    }


def analyze_p2(work_dir: Path, jobs: int, out_json: Path, out_md: Path) -> dict[str, Any]:
    npz_files = sorted(p for p in work_dir.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    print(f"[ANALYZE-P2] Classifying {len(npz_files)} run files in {work_dir}...", flush=True)
    with Pool(processes=min(max(1, jobs), max(1, len(npz_files)))) as pool:
        rows = list(pool.imap_unordered(_classify_worker, [str(p) for p in npz_files]))
    rows.sort(key=lambda r: (r["cell_id"], r["seed"]))

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(r["cell_id"], []).append(r)

    cells = {sp.cell_id: summarize_p2_cell(by_cell[sp.cell_id]) for sp in P2_CELLS if sp.cell_id in by_cell}

    comparisons = {
        "PM1a_both05_vs_base": paired_bt("M1-both05", "M1-base", by_cell),
        "PM1b_both025_vs_base": paired_bt("M1-both025", "M1-base", by_cell),
        "PM1c_both05_vs_attn05": paired_bt("M1-both05", "M1-attn05", by_cell),
        "PM1d_both05_vs_mlp05": paired_bt("M1-both05", "M1-mlp05", by_cell),
        "PM2a_lr5e4_both025_vs_base": paired_bt("M2a-lr5e4-both025", "M2a-lr5e4-base", by_cell),
        "EM2b_d128_both025_vs_base": paired_bt("M2b-d128-both025", "M2b-d128-base", by_cell),
        "PR1_rope_mlp_2e3_all": paired_bt("R1-mlp-2e3-both025", "R1-mlp-2e3-base", by_cell),
        "PR1_rope_mlp_2e3_h2": paired_bt("R1-mlp-2e3-both025", "R1-mlp-2e3-base", by_cell, head_filter=2),
        "PD1_4L_rope_vs_abs_all": paired_bt("D1-rope-24k", "D1-abs-24k", by_cell),
        "PD1_4L_rope_vs_abs_h2": paired_bt("D1-rope-24k", "D1-abs-24k", by_cell, head_filter=2),
        "ED1_4L_abs_l01_vs_base": paired_bt("D1-abs-l01_05-24k", "D1-abs-24k", by_cell),
    }
    secondary = {
        "SM1_attn05_le12": (cells["M1-attn05"]["B_T"] <= 12) if "M1-attn05" in cells else None,
        "SM1_mlp05_le8": (cells["M1-mlp05"]["B_T"] <= 8) if "M1-mlp05" in cells else None,
        "SR1a_rope_attn_1e3": (cells["R1-attn-1e3"]["B_T"] >= 16 and cells["R1-attn-1e3"]["counts"]["R"] <= 2) if "R1-attn-1e3" in cells else None,
        "ER1b_rope_attn_2e3_BT": cells["R1-attn-2e3"]["B_T"] if "R1-attn-2e3" in cells else None,
        "ER1b_rope_attn_2e3_R": cells["R1-attn-2e3"]["counts"]["R"] if "R1-attn-2e3" in cells else None,
    }
    res = {
        "phase": "phase2_preregistered",
        "criterion": f"B_T: loss_rep < {LOSS_T_THRESH} AND deep_decoy_over_clean >= {DECOY_T_THRESH}",
        "n_runs_analyzed": len(rows),
        "n_cells_present": len(cells),
        "cells": cells,
        "comparisons": comparisons,
        "secondary": secondary,
        "runs": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 2 Pre-Registered Confirmation — Frozen Analysis Summary",
        "",
        f"- Runs analyzed: **{len(rows)}** across **{len(cells)}** cells (fresh seeds `1048..1071`).",
        f"- Unified behavioral criterion: **`B_T` = (`loss_rep < {LOSS_T_THRESH}` AND `deep_decoy_over_clean >= {DECOY_T_THRESH}`)**.",
        "",
        "## Comparisons (Paired on Identical Fresh Seeds)",
        "",
        "| ID | X vs Y | n | B_T (kx vs ky) | R (rx vs ry) | U (ux vs uy) | Verdict | McNemar p |",
        "|---|---|---:|---|---|---|---:|---:|",
    ]
    for k, d in comparisons.items():
        if d.get("n", 0) == 0:
            lines.append(f"| {k} | {d['x']} vs {d['y']} | 0 | missing | — | — | — | — |")
        else:
            lines.append(
                f"| {k} | {d['x']} vs {d['y']} | {d['n']} | {d['kx']} vs {d['ky']} | "
                f"{d['rx_R']} vs {d['ry_R']} | {d['ux']} vs {d['uy']} | **{d['verdict']}** | {d['mcnemar_p']} |"
            )
    lines.extend([
        "",
        "## Secondary & Blind Checks",
        "",
        f"- `secondary`: `{json.dumps(secondary)}`",
        "",
        "## All 16 Cells",
        "",
        "| Cell | Role | L | Pos | d | lr | Steps | n | U | R | B_T | S | M | h2 B_T | h4 B_T | med loss | med R_1 | med m0 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for sp in P2_CELLS:
        if sp.cell_id not in cells:
            continue
        cs = cells[sp.cell_id]
        c = cs["counts"]
        h2 = cs["by_head"].get("h2", {})
        h4 = cs["by_head"].get("h4", {})
        lines.append(
            f"| {sp.cell_id} | {sp.role} | {sp.n_layer} | {'RoPE' if sp.use_rope else 'abs'} | {sp.d_model} | {sp.lr:g} | {sp.steps} | "
            f"{cs['n']} | {c['U']} | {c['R']} | **{cs['B_T']}** | {c['S']} | {c['M']} | "
            f"{h2.get('B_T',0)}/12 | {h4.get('B_T',0)}/12 | {cs['loss_rep_med']['median']} | "
            f"{cs['r1_aligned']['median']} | {cs['rms_m0_aligned']['median']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 pre-registered confirmation runner & analyzer (16 cells, 384 runs).")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--ic", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_confirm_phase2")
    ap.add_argument("--jobs", type=int, default=86)
    ap.add_argument("--analyze-jobs", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-json", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_confirm_phase2_results.json")
    ap.add_argument("--out-md", type=Path, default=REPO / "docs" / "results" / "v2_analysis" / "relay_confirm_phase2_summary.md")
    args = ap.parse_args(argv)

    if args.list:
        for sp in P2_CELLS:
            print(f"  {sp.cell_id:20s} [{sp.role:11s}] L={sp.n_layer} {'RoPE' if sp.use_rope else 'abs ':4s} d={sp.d_model:<3d} lr={sp.lr:g} steps={sp.steps} a={sp.attn_out_scale} m={sp.mlp_out_scale}")
        print(f"TOTAL: {len(P2_CELLS)} cells, {len(P2_CELLS) * 24} runs (seeds 1048..1059 h2 + 1060..1071 h4)")
        return 0

    if args.verify_only:
        verify_math_and_fold()
        return 0

    args.work.mkdir(parents=True, exist_ok=True)
    if args.ic:
        rep = run_ic_p2(args.work, max(1, args.jobs))
        if not rep["all_pass"]:
            return 2

    if args.run:
        jobs: list[dict[str, Any]] = []
        for sp in P2_CELLS:
            for s in SEEDS_P2_H2:
                jobs.append(make_job(sp, s, 2))
            for s in SEEDS_P2_H4:
                jobs.append(make_job(sp, s, 4))
        jobs.sort(key=estimated_cost, reverse=True)
        todo = [j for j in jobs if args.overwrite or not (args.work / f"{j['name']}.npz").exists()]
        print(f"[RUN-P2] Selected={len(jobs)} | To run={len(todo)} | Cached={len(jobs)-len(todo)} | Workers={args.jobs}", flush=True)
        if todo:
            t0 = time.monotonic()
            done = 0
            with Pool(processes=max(1, args.jobs)) as pool:
                for msg in pool.imap_unordered(run_one, [(j, str(args.work), args.overwrite) for j in todo]):
                    done += 1
                    if done % 20 == 0 or done == len(todo) or done <= 5:
                        print(f"[{done:3d}/{len(todo):3d} | {(time.monotonic()-t0)/60:5.1f}m] {msg}", flush=True)
            print(f"[RUN-P2] Finished {done} runs in {(time.monotonic()-t0)/60:.1f} min.", flush=True)

    if args.analyze:
        analyze_p2(args.work, max(1, args.analyze_jobs), args.out_json, args.out_md)

    if not (args.ic or args.run or args.analyze):
        ap.error("Specify one or more of --list, --verify-only, --ic, --run, --analyze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
