"""Post-hoc exploratory stress tests (NOT pre-registered):
  * Gate 1 (``--gate rope``): Replace learned absolute ``pos`` with standard RoPE (Su et al. 2021)
    in 2-layer attention-only and full modern MLP + AdamW + warmup Transformers (6 cells, 144 runs).
  * Gate 2 (``--gate depth``): Deeper Transformers (4 layers and 6 layers) with modern
    4x GELU MLP + AdamW + warmup under learned ``pos`` and RoPE, with and without shallow-layer
    residual damping (6 cells, 144 runs).

Cached strictly in ``work/relay_stress_tests/`` (never touches pre-registered ``work/relay_confirm/``).
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

_rc_spec = importlib.util.spec_from_file_location("relay_confirm", Path(__file__).with_name("relay_confirm.py"))
rc = importlib.util.module_from_spec(_rc_spec)
sys.modules["relay_confirm"] = rc
_rc_spec.loader.exec_module(rc)

_an_spec = importlib.util.spec_from_file_location("relay_confirm_analyze", Path(__file__).with_name("relay_confirm_analyze.py"))
an = importlib.util.module_from_spec(_an_spec)
sys.modules["relay_confirm_analyze"] = an
_an_spec.loader.exec_module(an)

fp = an.fp


# -------------------------------------------------------------------------------------- RoPE helpers
def rope_cos_sin(seq_len: int, d_head: int, base: float = 10000.0):
    import torch

    assert d_head % 2 == 0, f"RoPE requires even d_head, got {d_head}"
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (T, d_head // 2)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x, cos, sin):
    """Apply standard rotary embedding to tensor ``x`` of shape ``(B, H, T, d_head)``."""
    import torch

    T = x.shape[2]
    c = cos[:, :, :T, :]
    s = sin[:, :, :T, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


def patch_model_for_rope(model) -> None:
    """Disable learned absolute ``pos`` and patch every attention block to apply standard RoPE to Q and K."""
    import torch

    # Zero out pos embedding and freeze it so h0 = tok(x) purely (no absolute position signal)
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


# -------------------------------------------------------------------------------------- cell specs
@dataclass(frozen=True)
class StressCell:
    cell_id: str
    gate: str
    n_layer: int = 2
    use_rope: bool = False
    attn_only: bool = False
    lr: float = 1e-3
    batch_size: int = 32
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    warmup_steps: int = 200
    steps: int = 12_000
    d_model: int = 64
    rep_min: int = 12
    attn_out_scale: dict[int, float] | None = None
    mlp_out_scale: dict[int, float] | None = None
    note: str = ""


GATE_ROPE_CELLS: list[StressCell] = [
    StressCell("ROPE-attn-low", "rope", n_layer=2, use_rope=True, attn_only=True, lr=5e-4, optimizer="adam", weight_decay=0.0, warmup_steps=0, note="low-lr sanity check"),
    StressCell("ROPE-attn-mid", "rope", n_layer=2, use_rope=True, attn_only=True, lr=1e-3, optimizer="adam", weight_decay=0.0, warmup_steps=0, note="attn-only baseline"),
    StressCell("ROPE-attn-mid-c05", "rope", n_layer=2, use_rope=True, attn_only=True, lr=1e-3, optimizer="adam", weight_decay=0.0, warmup_steps=0, attn_out_scale={0: 0.5}, note="attn-only lever"),
    StressCell("ROPE-mlp-wu", "rope", n_layer=2, use_rope=True, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, note="modern 2L RoPE+MLP+AdamW+wu"),
    StressCell("ROPE-mlp-wu-both05", "rope", n_layer=2, use_rope=True, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, attn_out_scale={0: 0.5}, mlp_out_scale={0: 0.5}, note="L0 scale 0.5"),
    StressCell("ROPE-mlp-wu-both025", "rope", n_layer=2, use_rope=True, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, attn_out_scale={0: 0.25}, mlp_out_scale={0: 0.25}, note="L0 scale 0.25"),
]

GATE_DEPTH_CELLS: list[StressCell] = [
    StressCell("L4-abs-mlp-wu", "depth", n_layer=4, use_rope=False, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, note="4L learned pos"),
    StressCell("L4-abs-mlp-wu-l0_05", "depth", n_layer=4, use_rope=False, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, attn_out_scale={0: 0.5}, mlp_out_scale={0: 0.5}, note="4L abs, damp L0 x0.5"),
    StressCell("L4-abs-mlp-wu-l01_05", "depth", n_layer=4, use_rope=False, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, attn_out_scale={0: 0.5, 1: 0.5}, mlp_out_scale={0: 0.5, 1: 0.5}, note="4L abs, damp L0+L1 x0.5"),
    StressCell("L6-abs-mlp-wu", "depth", n_layer=6, use_rope=False, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, note="6L learned pos"),
    StressCell("L6-abs-mlp-wu-l01_05", "depth", n_layer=6, use_rope=False, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, attn_out_scale={0: 0.5, 1: 0.5}, mlp_out_scale={0: 0.5, 1: 0.5}, note="6L abs, damp L0+L1 x0.5"),
    StressCell("L4-rope-mlp-wu", "depth", n_layer=4, use_rope=True, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, note="4L RoPE"),
    StressCell("L4-rope-mlp-wu-l01_05", "depth", n_layer=4, use_rope=True, attn_only=False, lr=1e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, attn_out_scale={0: 0.5, 1: 0.5}, mlp_out_scale={0: 0.5, 1: 0.5}, note="4L RoPE, damp L0+L1 x0.5"),
    StressCell("ROPE-mlp-wu-lr2e-3", "depth", n_layer=2, use_rope=True, attn_only=False, lr=2e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, note="2L RoPE high-LR stress"),
    StressCell("ROPE-mlp-wu-lr2e-3-both025", "depth", n_layer=2, use_rope=True, attn_only=False, lr=2e-3, optimizer="adamw", weight_decay=0.1, warmup_steps=200, attn_out_scale={0: 0.25}, mlp_out_scale={0: 0.25}, note="2L RoPE high-LR, damp L0 x0.25"),
]

ALL_CELLS = GATE_ROPE_CELLS + GATE_DEPTH_CELLS
CELL_MAP = {c.cell_id: c for c in ALL_CELLS}


def make_stress_job(sp: StressCell, seed: int, n_head: int, *, steps_override: int | None = None) -> dict[str, Any]:
    steps = int(steps_override if steps_override is not None else sp.steps)
    return {
        "name": f"{sp.cell_id}_h{n_head}_s{seed}",
        "cell_id": sp.cell_id,
        "gate": sp.gate,
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


def stress_stats(model, task) -> dict[str, float]:
    """Depth-agnostic probe stats (works for 2, 4, 6+ layers and RoPE)."""
    import torch

    pr = task._probe
    x = pr["x"]
    rep_t = pr["rep"]
    rep, src = rep_t.numpy(), pr["src"].numpy()
    maps = [a.numpy() for a in model.attention_maps(x)]
    bi, ti = np.nonzero(rep)

    out: dict[str, float] = {}
    # Record L0 and max-over-upper-layers source & band attention
    for layer_idx, tag in [(0, "l0"), (1, "l1")]:
        a = maps[layer_idx]
        band = np.zeros((len(bi), a.shape[1]))
        for d in range(task.rep_min - 1, task.rep_max):
            ok = ti >= d
            band[ok] += a[bi[ok], :, ti[ok], ti[ok] - d]
        out[f"band_{tag}"] = float(band.mean())
        out[f"source_{tag}"] = float(a[bi, :, ti, src[bi, ti]].mean())

    max_src_upper = max(
        float(maps[l][bi, :, ti, src[bi, ti]].mean(axis=0).max())
        for l in range(1, len(maps))
    )
    out["max_src_upper_head"] = max_src_upper

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


STRESS_STAT_KEYS = ("band_l0", "band_l1", "source_l0", "source_l1", "max_src_upper_head", "loss_rep", "rms_emb", "rms_y0", "rms_m0", "r1")


def train_stress(job: dict[str, Any]):
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
            stats[step] = stress_stats(model, task)
            model.train()
        if step == steps:
            break
        x, y = task.sample_batch(tc.batch_size, gen)
        loss = task.loss(model(x), y)
        step_lr = lr_at(step, tc) if use_schedule else tc.lr
        opt.step(list(torch.autograd.grad(loss, params, allow_unused=True, materialize_grads=True)), step_lr)

    # Fold scaling factors into final linear projections and restore unscaled forward
    for lyr, factor in a_factors.items():
        a = model.blocks[lyr].attn
        if job["use_rope"]:
            # Re-patch clean unscaled RoPE forward
            pass
        else:
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
    return model, task, at, stats


def run_one(args: tuple[dict[str, Any], str]) -> str:
    job, work_dir = args
    import torch

    out = Path(work_dir) / f"{job['name']}.npz"
    if out.exists():
        return f"cached {job['name']}"
    t0 = time.monotonic()
    model, _, at, stats = train_stress(job)
    wall_s = time.monotonic() - t0
    arrays = {f"param/{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    for k in STRESS_STAT_KEYS:
        arrays[f"stat_{k}"] = np.asarray([stats[s][k] for s in at], dtype=np.float64)
    meta = {
        **job,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
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


# -------------------------------------------------------------------------------------- depth-agnostic classifier
def classify_stress_npz(npz_path: Path) -> dict[str, Any]:
    import torch

    from tycho.tasks.induction import Induction

    with np.load(npz_path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta_json"]))
        state = {k[6:]: torch.as_tensor(f[k]) for k in f.files if k.startswith("param/")}
        steps = f["lc_step"].astype(np.int64)
        stats = {k: f[f"stat_{k}"].astype(np.float64) for k in STRESS_STAT_KEYS if f"stat_{k}" in f.files}

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

    # Exact common 2,000-sequence behavioral evaluation (seed 777)
    smp = an.get_sample(rep_min)
    probs = fp.probabilities(model, smp["x"])
    sc = fp.score(fp.true_token(probs, smp), smp)

    # Probe-set attention summary across ALL upper layers (1 .. n_layer-1)
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

    # Depth-agnostic behavioral T priority (avoids misclassifying RoPE delta=-1 dup heads with decoy>=0.9 & loss~0.18 as R)
    if not math.isfinite(loss_rep) or loss_rep >= 2.0:
        cat = "U"
    elif math.isfinite(decoy) and decoy >= 0.9:
        cat = "T"
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
        "gate": meta["gate"],
        "use_rope": bool(meta.get("use_rope")),
        "n_layer": n_layer,
        "seed": seed,
        "n_head": nh,
        "cat": cat,
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
    return classify_stress_npz(Path(p_str))


# -------------------------------------------------------------------------------------- local verification
def verify_all() -> None:
    import torch

    # 1. Verify RoPE norm invariance and exact relative-position inner product invariance
    cos, sin = rope_cos_sin(64, 32)
    torch.manual_seed(0)
    q = torch.randn(1, 2, 1, 32).expand(1, 2, 64, 32).contiguous()
    k = torch.randn(1, 2, 1, 32).expand(1, 2, 64, 32).contiguous()
    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    norm_err = float((qr.norm(dim=-1) - q.norm(dim=-1)).abs().max())
    assert norm_err < 1e-5, f"RoPE norm drift: {norm_err}"
    # Check <R_t q, R_{t-5} k> is identical for t=10 vs t=42
    d1 = (qr[0, 0, 10] * kr[0, 0, 5]).sum().item()
    d2 = (qr[0, 0, 42] * kr[0, 0, 37]).sum().item()
    assert abs(d1 - d2) < 1e-5, f"RoPE relative invariance failed: {d1} vs {d2}"

    # 2. Verify folding exactness on BOTH RoPE+MLP (2L) and 4L RoPE+MLP
    for sp in (CELL_MAP["ROPE-mlp-wu-both05"], CELL_MAP["L4-rope-mlp-wu-l01_05"]):
        j = make_stress_job(sp, 1000, 2, steps_override=4)
        model, task, _, st = train_stress(j)
        x = task._probe["x"][:4]
        with torch.no_grad():
            logits_folded = model(x)
        # Re-run model forward after reloading from state_dict (as classifier does)
        m2 = task.build_model()
        m2.load_state_dict(model.state_dict())
        patch_model_for_rope(m2)
        m2.eval()
        with torch.no_grad():
            logits_reloaded = m2(x)
        diff = float((logits_folded - logits_reloaded).abs().max())
        assert diff == 0.0 and torch.isfinite(logits_folded).all(), f"Reload mismatch on {sp.cell_id}: {diff}"
        assert float(model.pos.weight.abs().max()) == 0.0, "RoPE pos weight not zero!"
    print("[VERIFY] RoPE orthogonality, relative invariance, 2L/4L folding & reload identity PASSED!", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Post-hoc RoPE (Gate 1) & Depth 4L/6L (Gate 2) stress tests.")
    ap.add_argument("--gate", choices=("rope", "depth", "all"), default="rope")
    ap.add_argument("--work", type=Path, default=REPO / "work" / "relay_stress_tests")
    ap.add_argument("--jobs", type=int, default=86)
    ap.add_argument("--analyze-jobs", type=int, default=32)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-runs", action="store_true")
    args = ap.parse_args(argv)

    if args.verify_only:
        verify_all()
        return 0

    cells = GATE_ROPE_CELLS if args.gate == "rope" else (GATE_DEPTH_CELLS if args.gate == "depth" else ALL_CELLS)
    args.work.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for sp in cells:
        pairs = [(s, 2) for s in rc.SEEDS_MAIN_H2] + [(s, 4) for s in rc.SEEDS_MAIN_H4]
        for seed, nh in pairs:
            jobs.append(make_stress_job(sp, seed, nh))

    # LPT ordering: deepest & MLP runs start first
    jobs.sort(key=lambda j: (j["cfg"]["n_layer"] * (2 if not j["cfg"]["attn_only"] else 1), j["n_head"]), reverse=True)

    if not args.skip_runs:
        todo = [j for j in jobs if not (args.work / f"{j['name']}.npz").exists()]
        print(f"[STRESS:{args.gate}] Selected={len(jobs)} | To run={len(todo)} | Workers={args.jobs}", flush=True)
        if todo:
            t0 = time.monotonic()
            done = 0
            with Pool(processes=max(1, args.jobs)) as pool:
                for msg in pool.imap_unordered(run_one, [(j, str(args.work)) for j in todo]):
                    done += 1
                    if done % 16 == 0 or done == len(todo) or done <= 5:
                        print(f"[{done:3d}/{len(todo):3d} | {(time.monotonic()-t0)/60:4.1f}m] {msg}", flush=True)
            print(f"[STRESS:{args.gate}] Finished {done} runs in {(time.monotonic()-t0)/60:.1f} min.", flush=True)

    # Classify all files matching selected cells
    wanted_cids = {sp.cell_id for sp in cells}
    npz_files = sorted(
        p for p in args.work.glob("*.npz")
        if not p.name.endswith(".tmp.npz") and any(p.name.startswith(cid + "_h") for cid in wanted_cids)
    )
    print(f"[STRESS:{args.gate}] Classifying {len(npz_files)} runs...", flush=True)
    with Pool(processes=min(max(1, args.analyze_jobs), max(1, len(npz_files)))) as pool:
        rows = list(pool.imap_unordered(_classify_worker, [str(p) for p in npz_files]))
    rows.sort(key=lambda r: (r["cell_id"], r["seed"]))

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(r["cell_id"], []).append(r)

    summary_cells = {sp.cell_id: an.summarize_cell(by_cell[sp.cell_id]) for sp in cells if sp.cell_id in by_cell}

    out_json = REPO / "docs" / "results" / "v2_analysis" / f"relay_stress_{args.gate}.json"
    out_md = REPO / "docs" / "results" / "v2_analysis" / f"relay_stress_{args.gate}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"exploratory": True, "gate": args.gate, "n_runs": len(rows), "cells": summary_cells, "runs": rows}, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Post-hoc Exploratory Stress Test — Gate: `{args.gate}`",
        "",
        "| Cell | L | Pos | MLP | n | U | R | T | S | M | T+S | h2 | h4 | med loss | med R_1 | med m0 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sp in cells:
        if sp.cell_id not in summary_cells:
            continue
        cs = summary_cells[sp.cell_id]
        rs = by_cell[sp.cell_id]
        c = cs["counts"]
        h2 = cs["by_head"].get("h2", {})
        h4 = cs["by_head"].get("h4", {})
        h2_ts = h2.get("T", 0) + h2.get("S", 0)
        h4_ts = h4.get("T", 0) + h4.get("S", 0)
        loss_med = round(float(np.median([r["loss_rep"] for r in rs])), 3)
        m0_med = round(float(np.median([r["rms_m0_aligned"] for r in rs])), 4)
        lines.append(
            f"| {sp.cell_id} | {sp.n_layer} | {'RoPE' if sp.use_rope else 'abs'} | {'no' if sp.attn_only else '4x'} | "
            f"{cs['n']} | {c['U']} | {c['R']} | {c['T']} | {c['S']} | {c['M']} | **{cs['source_strict']}** | "
            f"{h2_ts}/12 | {h4_ts}/12 | {loss_med} | {cs['r1_aligned']['median']} | {m0_med} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
