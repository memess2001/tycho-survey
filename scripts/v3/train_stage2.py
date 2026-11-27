"""Tycho Survey v3 Stage 2 — Pre-Registered 5-Arm Causal Surgery Trainer & Online Probe.

Implements a modern OLMo-2 / Llama-style decoder-only Transformer:
  - RoPE (theta=10,000), SwiGLU MLP, QK-Norm (RMSNorm on Q & K), untied lm_head, no biases.
  - Configurable normalization & initialization arms (docs/v3_stage2_spec.md):
      1. pre_ln             : Standard Pre-RMSNorm baseline (emb_std=0.02, gamma=1.0)
      2. post_olmo2         : Default OLMo-2 Post-RMSNorm (emb_std=0.02, gamma=1.0 -> R_0(0)=50)
      3. post_matched_gamma : Matched-Skip Post-RMSNorm (emb_std=0.02, gamma=0.02 -> R_0(0)=1)
      4. post_matched_emb   : Gemma-style scaled embedding control (embed_scale=50, gamma=1.0 -> R_0(0)=1)
      5. post_frozen_gamma  : Peri-LN frozen-gamma lock-in control (emb_std=0.02, gamma=1.0 frozen -> R_0(0)=50)
      6. post_depth_graded  : Optional self-organized depth-graded gamma_l(0) in [0.02, 0.12]

Built-in online FP32 diagnostics at log-spaced checkpoints:
  - val_bpb, R_0 = gamma_A0 / eff_embed_rms, wR_0 = wRMS_A0 / eff_embed_rms
  - gamma_attn[l], gamma_mlp[l], wRMS_attn[l], wRMS_mlp[l]
  - cosA_stream[l], cosM_stream[l], min_cosA_stream, max_outlier_int, max_outlier_pos0
  - Logit Lens pred_depth, full single-block mean ablation dA[l], dM[l], causD
  - Layer-0 Hydra bridge E_same_L0 = dA0M0 - dA0 - dM0
  - IC-v3.2 cancellation-safe residual identity verification (< 1e-5)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture: RoPE + QK-Norm + SwiGLU + Pre-LN / Inside-Branch Post-RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, init_val: float = 1.0, learnable: bool = True):
        super().__init__()
        self.eps = eps
        w = torch.full((dim,), float(init_val))
        if learnable:
            self.weight = nn.Parameter(w)
        else:
            self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * inv_rms).to(x.dtype) * self.weight.to(x.dtype)


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (T, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    T = x.size(2)
    c = cos[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    s = sin[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.stack((out1, out2), dim=-1).flatten(-2)


@dataclass
class Stage2Config:
    arm: str = "post_olmo2"
    vocab_size: int = 50257
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1376
    seq_len: int = 512
    emb_std: float = 0.02
    init_std: float = 0.02
    rope_theta: float = 10000.0
    eps: float = 1e-6


class ModernAttention(nn.Module):
    def __init__(self, cfg: Stage2Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # OLMo-2 style QK-Norm (applied over full d_model before head reshape, matching OLMo-2)
        self.q_norm = RMSNorm(cfg.d_model, eps=cfg.eps, init_val=1.0, learnable=True)
        self.k_norm = RMSNorm(cfg.d_model, eps=cfg.eps, init_val=1.0, learnable=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_norm(self.q_proj(x)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(self.k_proj(x)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: Stage2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ModernBlock(nn.Module):
    def __init__(self, cfg: Stage2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_pre_ln = cfg.arm == "pre_ln"
        if cfg.arm == "post_matched_gamma":
            g_init = cfg.emb_std
            learnable = True
        elif cfg.arm == "post_frozen_gamma":
            g_init = 1.0
            learnable = False
        elif cfg.arm == "post_depth_graded":
            frac = layer_idx / max(1, cfg.n_layers - 1)
            g_init = cfg.emb_std + frac * (0.12 - cfg.emb_std)
            learnable = True
        else:
            # pre_ln, post_olmo2, post_matched_emb
            g_init = 1.0
            learnable = True

        self.attn = ModernAttention(cfg)
        self.mlp = SwiGLUMLP(cfg)
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.eps, init_val=g_init, learnable=learnable)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.eps, init_val=g_init, learnable=learnable)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ablate: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
        record: Optional[Dict[str, list]] = None,
    ) -> torch.Tensor:
        h_in = h
        if self.is_pre_ln:
            da = self.attn(self.attn_norm(h), cos, sin)
        else:
            da = self.attn_norm(self.attn(h, cos, sin))
        if ablate and (self.layer_idx, "attn") in ablate:
            vec = ablate[(self.layer_idx, "attn")].to(da.dtype).to(da.device)
            da = torch.cat([da[:, :1, :], vec.view(1, 1, -1).expand(da.size(0), da.size(1) - 1, -1)], dim=1)

        h_mid = h + da

        if self.is_pre_ln:
            dm = self.mlp(self.mlp_norm(h_mid))
        else:
            dm = self.mlp_norm(self.mlp(h_mid))
        if ablate and (self.layer_idx, "mlp") in ablate:
            vec = ablate[(self.layer_idx, "mlp")].to(dm.dtype).to(dm.device)
            dm = torch.cat([dm[:, :1, :], vec.view(1, 1, -1).expand(dm.size(0), dm.size(1) - 1, -1)], dim=1)

        h_out = h_mid + dm
        if record is not None:
            record["h_in"].append(h_in.detach())
            record["da"].append(da.detach())
            record["dm"].append(dm.detach())
            record["h_out"].append(h_out.detach())
        return h_out


class Stage2Transformer(nn.Module):
    def __init__(self, cfg: Stage2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_scale = 1.0
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([ModernBlock(cfg, l) for l in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps, init_val=1.0, learnable=True)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope_freqs(cfg.d_model // cfg.n_heads, cfg.seq_len + 16, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        eff_emb_std = 1.0 if self.cfg.arm == "post_matched_emb" else self.cfg.emb_std
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=eff_emb_std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.cfg.init_std)
        for mod in self.layers.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        ablate: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
        record: Optional[Dict[str, list]] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids) * self.embed_scale
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin, ablate=ablate, record=record)
        return self.lm_head(self.final_norm(h))


# ---------------------------------------------------------------------------
# Online FP32 4-Channel Diagnostics & IC-v3.2 Verification
# ---------------------------------------------------------------------------

def interior_ce_nats(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    # Predict tokens at positions 2..T-1 (strictly excluding BOS position 0 and predicting pos 1 from BOS alone)
    pred = logits[:, 1:-1, :].float()
    tgt = input_ids[:, 2:]
    return float(F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1)).item())


@torch.no_grad()
def evaluate_diagnostics(
    model: Stage2Transformer,
    val_batch: torch.Tensor,
    diag_batch: torch.Tensor,
    bytes_per_token: float,
    device: torch.device,
) -> dict:
    was_training = model.training
    model.eval()

    # 1. Full validation CE / bpb (fp32 for precision)
    val_ids = val_batch.to(device)
    val_nats_list = []
    bs_chunk = min(4, val_ids.size(0))
    for i in range(0, val_ids.size(0), bs_chunk):
        chunk = val_ids[i : i + bs_chunk]
        lg = model(chunk)
        val_nats_list.append(interior_ce_nats(lg, chunk))
    val_nats = float(np.mean(val_nats_list))
    val_bpb = val_nats / (math.log(2.0) * bytes_per_token)

    # 2. Clean forward on diagnostic batch with full residual recording
    d_ids = diag_batch.to(device)
    rec: Dict[str, list] = {"h_in": [], "da": [], "dm": [], "h_out": []}
    logits_clean = model(d_ids, record=rec)
    clean_nats = interior_ce_nats(logits_clean, d_ids)

    L = model.cfg.n_layers
    raw_emb_rms = float(torch.sqrt(model.embed_tokens.weight.detach().float().pow(2).mean()).item())
    eff_emb_rms = raw_emb_rms * float(model.embed_scale)

    gamma_attn, gamma_mlp = [], []
    w_a, w_m, cos_a, cos_m, cos_am = [], [], [], [], []
    rel_errs = []
    mean_deltas: Dict[Tuple[int, str], torch.Tensor] = {}

    def _lens_nats(h: torch.Tensor) -> float:
        lg = model.lm_head(model.final_norm(h[:, 1:-1, :])).float()
        return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), d_ids[:, 2:].reshape(-1)).item())

    lens_nats = [_lens_nats(rec["h_in"][0])]
    out_int_max, out_pos0_max = 0.0, 0.0

    for l in range(L):
        ga = model.layers[l].attn_norm.weight.detach().float()
        gm = model.layers[l].mlp_norm.weight.detach().float()
        gamma_attn.append(round(float(torch.sqrt(ga.pow(2).mean()).item()), 5))
        gamma_mlp.append(round(float(torch.sqrt(gm.pow(2).mean()).item()), 5))

        hi = rec["h_in"][l].float()
        dai = rec["da"][l].float()
        dmi = rec["dm"][l].float()
        ho = rec["h_out"][l].float()

        # IC-v3.2 cancellation-safe residual identity check
        ae = float((ho - (hi + dai + dmi)).abs().max().item())
        sc = max(
            float(hi.abs().max().item()),
            float(dai.abs().max().item()),
            float(dmi.abs().max().item()),
            float(ho.abs().max().item()),
        ) + 1e-8
        rel_errs.append(ae / sc)

        mean_deltas[(l, "attn")] = dai[:, 1:, :].mean(dim=(0, 1))
        mean_deltas[(l, "mlp")] = dmi[:, 1:, :].mean(dim=(0, 1))

        da_int = dai[:, 1:, :]
        dm_int = dmi[:, 1:, :]
        hi_int = hi[:, 1:, :]
        hm_int = (hi + dai)[:, 1:, :]

        w_a.append(round(float(torch.sqrt(da_int.pow(2).mean()).item()), 5))
        w_m.append(round(float(torch.sqrt(dm_int.pow(2).mean()).item()), 5))
        na, nm = da_int.norm(dim=-1), dm_int.norm(dim=-1)
        nha, nhm = hi_int.norm(dim=-1), hm_int.norm(dim=-1)
        cos_a.append(round(float(((da_int * hi_int).sum(-1) / (na * nha + 1e-8)).mean().item()), 4))
        cos_m.append(round(float(((dm_int * hm_int).sum(-1) / (nm * nhm + 1e-8)).mean().item()), 4))
        cos_am.append(round(float(((da_int * dm_int).sum(-1) / (na * nm + 1e-8)).mean().item()), 4))

        if l < L - 1:
            oi = ho[:, 1:, :].abs()
            o0 = ho[:, 0, :].abs()
            out_int_max = max(out_int_max, float(oi.max().item() / (oi.median().item() + 1e-8)))
            out_pos0_max = max(out_pos0_max, float(o0.max().item() / (o0.median().item() + 1e-8)))
        lens_nats.append(_lens_nats(ho))

    lens_arr = np.array(lens_nats)
    gains = np.maximum(0.0, lens_arr[:-1] - lens_arr[1:])
    pred_depth = float(np.sum(gains * (np.arange(1, L + 1) / L)) / max(float(gains.sum()), 1e-8))

    # 3. Full single-block mean ablation (dA[l], dM[l]) + Layer-0 Hydra bridge (dA0M0)
    dA, dM = [], []
    for l in range(L):
        lg_a = model(d_ids, ablate={(l, "attn"): mean_deltas[(l, "attn")]})
        lg_m = model(d_ids, ablate={(l, "mlp"): mean_deltas[(l, "mlp")]})
        dA.append(round(interior_ce_nats(lg_a, d_ids) - clean_nats, 4))
        dM.append(round(interior_ce_nats(lg_m, d_ids) - clean_nats, 4))

    lg_a0m0 = model(d_ids, ablate={(0, "attn"): mean_deltas[(0, "attn")], (0, "mlp"): mean_deltas[(0, "mlp")]})
    dA0M0 = round(interior_ce_nats(lg_a0m0, d_ids) - clean_nats, 4)
    E_same_L0 = round(dA0M0 - dA[0] - dM[0], 4)

    # Normalized causal depth across layers
    layer_imp = np.maximum(0.0, np.array(dA) + np.array(dM))
    causD = float(np.sum(layer_imp * (np.arange(1, L + 1) / L)) / max(float(layer_imp.sum()), 1e-8))

    # Adjacent-attention Hydra spot check across layers 0..L-2
    adj_E = []
    for l in range(L - 1):
        lg_j = model(
            d_ids,
            ablate={(l, "attn"): mean_deltas[(l, "attn")], (l + 1, "attn"): mean_deltas[(l + 1, "attn")]},
        )
        dj = interior_ce_nats(lg_j, d_ids) - clean_nats
        adj_E.append(round(dj - dA[l] - dA[l + 1], 4))

    R0 = gamma_attn[0] / max(eff_emb_rms, 1e-8)
    wR0 = w_a[0] / max(eff_emb_rms, 1e-8)
    ic_max = float(max(rel_errs))

    if was_training:
        model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_nats": round(val_nats, 4),
        "val_bpb": round(val_bpb, 4),
        "diag_clean_nats": round(clean_nats, 4),
        "ic_v3_2_max_rel_err": ic_max,
        "ic_v3_2_pass": bool(ic_max <= 1e-5),
        "raw_embed_rms": round(raw_emb_rms, 5),
        "eff_embed_rms": round(eff_emb_rms, 5),
        "R_0": round(R0, 4),
        "wR_0": round(wR0, 4),
        "gamma_attn": gamma_attn,
        "gamma_mlp": gamma_mlp,
        "wRMS_attn": w_a,
        "wRMS_mlp": w_m,
        "cosA_stream": cos_a,
        "cosM_stream": cos_m,
        "cos_am": cos_am,
        "min_cosA_stream": round(min(cos_a), 4),
        "min_cosM_stream": round(min(cos_m), 4),
        "max_outlier_int": round(out_int_max, 2),
        "max_outlier_pos0": round(out_pos0_max, 2),
        "pred_depth": round(pred_depth, 4),
        "causal_depth": round(causD, 4),
        "dA": dA,
        "dM": dM,
        "dA0": dA[0],
        "dM0": dM[0],
        "dA0M0": dA0M0,
        "E_same_L0": E_same_L0,
        "adj_attn_E": adj_E,
        "adj_attn_frac_red": round(float(np.mean(np.array(adj_E) > 0.02)), 3),
    }


# ---------------------------------------------------------------------------
# Data Streams: Synthetic (Offline Smoke), Pre-Tokenized .bin, & FineWeb-Edu
# ---------------------------------------------------------------------------

class SyntheticStream:
    """Fast deterministic Markov + copy-motif stream for offline CPU smoke tests."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        # Fixed transition table over top-256 frequent tokens so loss drops rapidly & cleanly
        self.core_v = min(256, vocab_size - 2)
        logits = self.rng.normal(0.0, 2.2, size=(self.core_v, self.core_v))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        self.trans = probs / probs.sum(axis=1, keepdims=True)
        self.cdf = np.cumsum(self.trans, axis=1)

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        out = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        out[:, 0] = self.vocab_size - 1  # BOS
        cur = self.rng.integers(0, self.core_v, size=batch_size)
        out[:, 1] = cur
        for t in range(2, self.seq_len):
            u = self.rng.random(batch_size)
            nxt = (self.cdf[cur] < u[:, None]).sum(axis=1)
            nxt = np.clip(nxt, 0, self.core_v - 1)
            # Periodic short-range copy motif to engage attention heads
            if t >= 10 and (t % 12 == 0):
                nxt = out[:, t - 8]
            out[:, t] = nxt
            cur = nxt
        return torch.from_numpy(out)


class BinaryShardStream:
    """Zero-overhead uint16 memmap stream for maximum PRO 6000 throughput."""

    def __init__(self, path: Path, seq_len: int, seed: int, offset_tokens: int = 0):
        self.data = np.memmap(str(path), dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_chunks = (len(self.data) - offset_tokens - 1) // seq_len
        assert self.n_chunks > 0, f"Binary shard {path} too small for seq_len={seq_len}"
        self.offset = offset_tokens
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.n_chunks)
        self.ptr = 0

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        rows = []
        for _ in range(batch_size):
            if self.ptr >= self.n_chunks:
                self.perm = self.rng.permutation(self.n_chunks)
                self.ptr = 0
            idx = int(self.perm[self.ptr])
            self.ptr += 1
            s = self.offset + idx * self.seq_len
            rows.append(self.data[s : s + self.seq_len].astype(np.int64))
        return torch.from_numpy(np.stack(rows, axis=0))

    def state_dict(self) -> dict:
        return {"ptr": int(self.ptr), "perm": self.perm.copy(), "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, st: dict) -> None:
        self.ptr = int(st["ptr"])
        self.perm = np.array(st["perm"], dtype=np.int64)
        self.rng.bit_generator.state = st["rng_state"]


class FineWebEduStream:
    """Streaming HuggingFaceFW/fineweb-edu (sample-10BT) packed into fixed seq_len chunks."""

    def __init__(self, seq_len: int, seed: int, skip_docs: int = 0):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.eos = self.tok.eos_token_id
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        if skip_docs > 0:
            ds = ds.skip(skip_docs)
        ds = ds.shuffle(seed=seed, buffer_size=2048)
        self.it = iter(ds)
        self.buf: List[int] = []
        self.bytes_seen = 0
        self.toks_seen = 0

    def _refill(self, min_tokens: int) -> None:
        while len(self.buf) < min_tokens:
            doc = next(self.it)["text"]
            ids = self.tok.encode(doc, add_special_tokens=False)
            self.bytes_seen += len(doc.encode("utf-8"))
            self.toks_seen += len(ids)
            self.buf.append(self.eos)
            self.buf.extend(ids)

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        need = batch_size * self.seq_len
        self._refill(need)
        arr = np.array(self.buf[:need], dtype=np.int64).reshape(batch_size, self.seq_len)
        del self.buf[:need]
        return torch.from_numpy(arr)

    @property
    def bytes_per_token(self) -> float:
        return self.bytes_seen / max(1, self.toks_seen)


# ---------------------------------------------------------------------------
# Optimizer, LR Schedule, & Training Loop
# ---------------------------------------------------------------------------

def build_optimizer(
    model: Stage2Transformer,
    *,
    peak_lr: float,
    weight_decay: float,
    wd_gamma: bool = True,
) -> torch.optim.AdamW:
    decay_params, nodecay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("embed_tokens"):
            nodecay_params.append(p)
        elif p.ndim >= 2:
            decay_params.append(p)
        else:
            # RMSNorm gamma parameters: OLMo-2 applies standard WD=0.1 unless --no-wd-gamma
            if wd_gamma:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=peak_lr, betas=(0.9, 0.95), eps=1e-8)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return peak_lr * float(step + 1) / max(1, warmup_steps)
    prog = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
    return peak_lr * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog)))


def default_eval_steps(total_steps: int) -> List[int]:
    if total_steps <= 50:
        return sorted(set([0, 5, 10, total_steps]))
    base = [0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7500, 9000, 10500, 12000, 13500, total_steps]
    return sorted(set(s for s in base if s <= total_steps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["pre_ln", "post_olmo2", "post_matched_gamma", "post_matched_emb", "post_frozen_gamma", "post_depth_graded"],
    )
    ap.add_argument("--preset", default="stage2_70m", choices=["stage2_70m", "tiny_smoke"])
    ap.add_argument("--dataset", default="fineweb-edu", choices=["fineweb-edu", "bin", "synthetic"])
    ap.add_argument("--train-bin", type=Path, default=None)
    ap.add_argument("--val-bin", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--peak-lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--no-wd-gamma", action="store_true")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sync-dir", type=Path, default=None, help="Optional Google Drive mirror dir for metrics/checkpoints")
    ap.add_argument("--sync-latest", action="store_true", help="Also mirror latest.pt to --sync-dir for cross-session resume")
    ap.add_argument("--resume", action="store_true", help="Resume from out_dir/latest.pt (or sync_dir/latest.pt) if present")
    ap.add_argument("--save-latest-every", type=int, default=1000, help="Save rolling latest.pt every N steps (0=disable)")
    ap.add_argument("--save-ckpts", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    use_bf16 = (args.dtype == "bf16") or (args.dtype == "auto" and device.type == "cuda")

    if args.preset == "stage2_70m":
        cfg = Stage2Config(arm=args.arm, vocab_size=50257, n_layers=12, d_model=512, n_heads=8, d_ff=1376, seq_len=512)
        total_steps = args.total_steps or 15260
        warmup_steps = args.warmup_steps or 400
        # Auto-select micro_batch by VRAM (all keep exact 65,536 tokens/step):
        # >=40GB (PRO 6000 96GB / A100 80GB / H100): 64x2 (or 128x1 on >=70GB); <40GB (L4 22.5GB): 32x4 (~10.5GB peak)
        vram_gb = (torch.cuda.get_device_properties(device).total_memory / (1024**3)) if device.type == "cuda" else 0.0
        if vram_gb >= 70.0:
            default_mb, default_ga = 128, 1
        elif vram_gb >= 38.0:
            default_mb, default_ga = 64, 2
        else:
            default_mb, default_ga = 32, 4
        micro_bs = args.micro_batch or default_mb
        grad_accum = args.grad_accum or default_ga  # all equal 65,536 tokens/step
        val_seqs, diag_seqs = 64, 8
    else:
        cfg = Stage2Config(arm=args.arm, vocab_size=512, n_layers=4, d_model=128, n_heads=4, d_ff=352, seq_len=64)
        total_steps = args.total_steps or 20
        warmup_steps = args.warmup_steps or 4
        micro_bs = args.micro_batch or 8
        grad_accum = args.grad_accum or 2
        val_seqs, diag_seqs = 16, 4

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.sync_dir is not None:
        args.sync_dir.mkdir(parents=True, exist_ok=True)

    done_path = args.out_dir / "DONE"
    sync_done = (args.sync_dir / "DONE") if args.sync_dir is not None else None
    if args.resume and (done_path.exists() or (sync_done is not None and sync_done.exists())):
        print(f"[{cfg.arm} s={args.seed}] SKIP already DONE in {args.out_dir}", flush=True)
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    metrics_path = args.out_dir / "metrics.jsonl"
    latest_path = args.out_dir / "latest.pt"

    # Pull prior artifacts from --sync-dir if resuming on a fresh Colab runtime
    if args.resume and args.sync_dir is not None:
        for fn in ["metrics.jsonl", "latest.pt"]:
            src = args.sync_dir / fn
            dst = args.out_dir / fn
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    if (not args.resume or not latest_path.exists()) and metrics_path.exists():
        metrics_path.unlink()

    # Build deterministic validation + training data streams
    if args.dataset == "synthetic":
        val_stream = SyntheticStream(cfg.vocab_size, cfg.seq_len, seed=99991)
        val_batch = val_stream.sample_batch(val_seqs)
        diag_batch = val_batch[:diag_seqs].clone()
        train_stream = SyntheticStream(cfg.vocab_size, cfg.seq_len, seed=args.seed)
        bytes_per_token = 1.0
    elif args.dataset == "bin":
        assert args.train_bin and args.val_bin, "--train-bin and --val-bin required for --dataset bin"
        val_stream = BinaryShardStream(args.val_bin, cfg.seq_len, seed=99991)
        val_batch = val_stream.sample_batch(val_seqs)
        diag_batch = val_batch[:diag_seqs].clone()
        train_stream = BinaryShardStream(args.train_bin, cfg.seq_len, seed=args.seed)
        bytes_per_token = 4.2
        dmeta = args.train_bin.parent / "dataset_meta.json"
        if dmeta.exists():
            try:
                bytes_per_token = float(json.loads(dmeta.read_text()).get("bytes_per_token", bytes_per_token))
            except Exception:
                pass
    else:
        val_stream = FineWebEduStream(cfg.seq_len, seed=99991, skip_docs=0)
        val_batch = val_stream.sample_batch(val_seqs)
        diag_batch = val_batch[:diag_seqs].clone()
        bytes_per_token = val_stream.bytes_per_token
        train_stream = FineWebEduStream(cfg.seq_len, seed=args.seed, skip_docs=2000)

    model = Stage2Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = build_optimizer(model, peak_lr=args.peak_lr, weight_decay=args.weight_decay, wd_gamma=not args.no_wd_gamma)

    start_step = 0
    prior_wall_s = 0.0
    logged_steps = set()
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        prior_wall_s = float(ckpt.get("wall_s", 0.0))
        if isinstance(train_stream, BinaryShardStream) and "stream" in ckpt:
            train_stream.load_state_dict(ckpt["stream"])
        if metrics_path.exists():
            # Deduplicate any metrics logged past start_step
            kept = []
            for line in metrics_path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if int(r["step"]) <= start_step:
                    kept.append(line)
                    logged_steps.add(int(r["step"]))
            metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"[{cfg.arm} s={args.seed}] RESUMED from step={start_step} (prior_wall={prior_wall_s:.1f}s)", flush=True)

    train_fn = torch.compile(model) if (args.compile and device.type == "cuda") else model

    meta = {
        "config": asdict(cfg),
        "n_params": n_params,
        "n_trainable": n_trainable,
        "seed": args.seed,
        "peak_lr": args.peak_lr,
        "weight_decay": args.weight_decay,
        "wd_gamma": not args.no_wd_gamma,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "micro_batch": micro_bs,
        "grad_accum": grad_accum,
        "tokens_per_step": micro_bs * grad_accum * cfg.seq_len,
        "dataset": args.dataset,
        "bytes_per_token": round(bytes_per_token, 5),
        "device": str(device),
        "bf16": use_bf16,
    }
    meta_path = args.out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    def sync_file(src: Path) -> None:
        if args.sync_dir is not None and src.exists():
            tmp = args.sync_dir / (src.name + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, args.sync_dir / src.name)

    sync_file(meta_path)

    eval_steps = set(default_eval_steps(total_steps))
    tokens_per_step = micro_bs * grad_accum * cfg.seq_len
    t_start = time.perf_counter()

    def save_latest(step: int) -> None:
        payload = {
            "step": step,
            "wall_s": prior_wall_s + (time.perf_counter() - t_start),
            "config": asdict(cfg),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
        }
        if isinstance(train_stream, BinaryShardStream):
            payload["stream"] = train_stream.state_dict()
        tmp = args.out_dir / "latest.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, latest_path)
        if args.sync_latest:
            sync_file(latest_path)

    def log_eval(step: int, train_nats: Optional[float], grad_norm: Optional[float], lr_val: float) -> None:
        diag = evaluate_diagnostics(model, val_batch, diag_batch, bytes_per_token, device)
        row = {
            "step": step,
            "tokens": step * tokens_per_step,
            "wall_s": round(prior_wall_s + (time.perf_counter() - t_start), 2),
            "lr": round(lr_val, 7),
            "train_nats": None if train_nats is None else round(train_nats, 4),
            "grad_norm": None if grad_norm is None else round(grad_norm, 4),
            **diag,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        sync_file(metrics_path)
        print(
            f"[{cfg.arm:18s} s={args.seed}] step={step:5d} tok={row['tokens']/1e6:6.1f}M "
            f"| val_bpb={diag['val_bpb']:.4f} R0={diag['R_0']:6.2f} emb={diag['eff_embed_rms']:.4f} gA0={diag['gamma_attn'][0]:.4f} "
            f"| minCosA={diag['min_cosA_stream']:+.3f} dA0={diag['dA0']:5.2f} dM0={diag['dM0']:4.2f} E_sL0={diag['E_same_L0']:+5.2f} "
            f"causD={diag['causal_depth']:.3f} ic={diag['ic_v3_2_max_rel_err']:.1e}",
            flush=True,
        )
        if args.save_ckpts:
            torch.save(
                {"step": step, "config": asdict(cfg), "model": model.state_dict()},
                args.out_dir / f"ckpt_step{step:06d}.pt",
            )

    if start_step == 0 and 0 in eval_steps and 0 not in logged_steps:
        log_eval(0, None, None, 0.0)

    model.train()
    for step in range(start_step + 1, total_steps + 1):
        lr_now = cosine_lr(step - 1, total_steps, warmup_steps, args.peak_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(grad_accum):
            batch = train_stream.sample_batch(micro_bs).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = train_fn(batch)
                loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
            (loss / grad_accum).backward()
            acc_loss += float(loss.item()) / grad_accum

        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        opt.step()

        if step in eval_steps:
            log_eval(step, acc_loss, gnorm, lr_now)
        if args.save_latest_every > 0 and (step % args.save_latest_every == 0 or step == total_steps):
            save_latest(step)

    done_path.write_text(f"completed_step={total_steps}\n")
    sync_file(done_path)
    # Clean up rolling ~1GB resume checkpoint from Drive once run is safely DONE
    if args.sync_dir is not None and (args.sync_dir / "latest.pt").exists():
        (args.sync_dir / "latest.pt").unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
