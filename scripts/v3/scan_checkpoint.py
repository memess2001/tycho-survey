"""Tycho Survey v3 Stage 1 — Unified 4-Channel Checkpoint Scanner.

Supports Pythia (GPTNeoX, parallel residual), OLMo-1 (Olmo, sequential pre-LN), and
OLMo-2 (Olmo2, sequential post-RMSNorm + QK-Norm) through a single architecture-verified
residual-delta adapter (IC-v3.2).

Channels computed per checkpoint (zero training):
  Ch1: Layer-wise Attn vs MLP residual write RMS, relative perturbation ratio, write energy share,
       write-stream cosine, within-layer Attn-MLP cosine, logit-lens bpb trajectory, and block DLA.
  Ch2: Block-level causal necessity (mean-ablation primary + zero-ablation audit) for all 2L blocks,
       plus structured O(L) pairwise Excess Joint Loss E(i,j) = dL(i,j) - dL(i) - dL(j)
       (E > 0: redundancy/backup; E < 0: serial/AND-gate composition).
  Ch3: 8 tokenizer-agnostic matched micro-probe contrasts (bpb & log-prob margins).
  Ch4: Prompt-endpoint centered Gram matrices K_l (M x M, tokenizer- & width-agnostic CKA),
       per-layer effective rank, massive-activation outlier ratios (t=0 vs t>=1),
       and attention entropy + BOS-sink mass.

Instrument checks built into every run:
  IC-v3.1: Determinism check (repeat Ch1 forward pass).
  IC-v3.2: Exact residual identity check h_{l+1} == h_l + delta_attn + delta_mlp at every layer.
  IC-v3.3: Ablation identity check (empty ablation == clean; all-block zero ablation == embed-only).
  IC-v3.4: Step-0 null sanity reporting.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probes import PairProbeItem, build_channel3_probes, build_core_passages, build_gram_prompts  # noqa: E402


# ---------------------------------------------------------------------------
# Unified Architecture Adapter (Pythia / OLMo-1 / OLMo-2)
# ---------------------------------------------------------------------------

class UnifiedAdapter:
    """Architecture-agnostic interface exposing residual writes (delta_attn, delta_mlp)."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model_type = str(getattr(model.config, "model_type", "unknown"))
        self.unembed = model.get_output_embeddings()
        self.embed_in = model.get_input_embeddings()
        if self.model_type == "gpt_neox":
            self.layers = list(model.gpt_neox.layers)
            self.final_norm = model.gpt_neox.final_layer_norm
            self.is_parallel = bool(getattr(model.config, "use_parallel_residual", True))
            self.is_post_norm = False
        elif self.model_type == "olmo":
            self.layers = list(model.model.layers)
            self.final_norm = model.model.norm
            self.is_parallel = False
            self.is_post_norm = False
        elif self.model_type == "olmo2":
            self.layers = list(model.model.layers)
            self.final_norm = model.model.norm
            self.is_parallel = False
            self.is_post_norm = True
        else:
            raise ValueError(f"Unsupported model_type={self.model_type!r}; expected gpt_neox, olmo, or olmo2")

        self.n_layers = len(self.layers)
        self.d_model = int(model.config.hidden_size)
        self.n_heads = int(model.config.num_attention_heads)
        self.vocab_size = int(model.config.vocab_size)

    def attn_delta_module(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.model_type == "gpt_neox":
            return layer.attention
        if self.model_type == "olmo":
            return layer.self_attn
        if self.model_type == "olmo2":
            # Post-norm residual write is emitted AFTER post_attention_layernorm
            return layer.post_attention_layernorm
        raise AssertionError

    def mlp_delta_module(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.model_type in ("gpt_neox", "olmo"):
            return layer.mlp
        if self.model_type == "olmo2":
            return layer.post_feedforward_layernorm
        raise AssertionError

    def norm_eps(self) -> float:
        for attr in ("eps", "variance_epsilon"):
            if hasattr(self.final_norm, attr):
                return float(getattr(self.final_norm, attr))
        return 1e-5

    def apply_final_norm_linearized(self, delta: torch.Tensor, h_final: torch.Tensor) -> torch.Tensor:
        """Apply first-order linearization of final_norm at clean h_final to residual delta."""
        eps = self.norm_eps()
        d32 = delta.float()
        h32 = h_final.float()
        is_rms = "rms" in self.final_norm.__class__.__name__.lower()
        if is_rms:
            scale = torch.sqrt(h32.pow(2).mean(dim=-1, keepdim=True) + eps)
            out = d32 / scale
        else:
            # Standard LayerNorm (parametric or non-parametric)
            var = h32.var(dim=-1, unbiased=False, keepdim=True)
            scale = torch.sqrt(var + eps)
            out = (d32 - d32.mean(dim=-1, keepdim=True)) / scale
        w = getattr(self.final_norm, "weight", None)
        if w is not None:
            out = out * w.detach().float()
        return out


# ---------------------------------------------------------------------------
# Tokenization & Core Batch Preparation
# ---------------------------------------------------------------------------

def prepare_core_batch(
    tokenizer: Any, passages: list[str], seq_len: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Tokenize passages into fixed-length (B, seq_len) block and count UTF-8 bytes of predicted interior tokens."""
    rows: list[list[int]] = []
    bos_id = tokenizer.bos_token_id
    total_pred_bytes = 0
    for text in passages:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if bos_id is not None and (len(ids) == 0 or ids[0] != bos_id):
            ids = [bos_id] + ids
        if len(ids) < seq_len:
            # Deterministically wrap around passage if shorter than seq_len
            while len(ids) < seq_len:
                ids = ids + tokenizer.encode(" " + text, add_special_tokens=False)
        ids = ids[:seq_len]
        rows.append(ids)
        # Interior predicted tokens are positions 2..seq_len-1 (predicted from t=1..seq_len-2)
        pred_text = tokenizer.decode(ids[2:], clean_up_tokenization_spaces=False)
        total_pred_bytes += max(1, len(pred_text.encode("utf-8")))
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    return input_ids, total_pred_bytes


def interior_ce_nats(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Mean cross-entropy in nats over interior predictions t=1..T-2 -> target t+1=2..T-1."""
    # logits[:, 1:-1] predicts input_ids[:, 2:]
    pred_logits = logits[:, 1:-1, :].float()
    targets = input_ids[:, 2:]
    return float(F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), targets.reshape(-1)).item())


def nats_to_bpb(mean_nats: float, n_pred_tokens: int, total_pred_bytes: int) -> float:
    return float((mean_nats * n_pred_tokens) / (max(1, total_pred_bytes) * math.log(2.0)))


# ---------------------------------------------------------------------------
# Channel 1 + Channel 4 (Core Pass) + IC-v3.2
# ---------------------------------------------------------------------------

def _effective_rank(mat: torch.Tensor) -> float:
    """Effective rank exp(H(sigma^2 / sum sigma^2)) of centered (N, d) matrix."""
    x = mat.float()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s.pow(2)
    tot = float(p.sum().item())
    if tot <= 0.0:
        return 0.0
    p = p / tot
    p = p[p > 1e-12]
    ent = float(-(p * torch.log(p)).sum().item())
    return float(math.exp(ent))


@torch.no_grad()
def run_channel1_and_core_ch4(
    adapter: UnifiedAdapter,
    input_ids: torch.Tensor,
    total_pred_bytes: int,
    *,
    compute_attn_stats: bool = True,
    attn_max_seqs: int = 16,
    erank_max_tokens: int = 1024,
) -> dict[str, Any]:
    """Run clean forward pass on D_core to collect Ch1, core Ch4 stats, mean ablation vectors, and IC-v3.2."""
    L = adapter.n_layers
    B, T = input_ids.shape
    n_pred_tokens = B * (T - 2)

    h_in_list: list[torch.Tensor | None] = [None] * L
    h_out_list: list[torch.Tensor | None] = [None] * L
    d_attn_list: list[torch.Tensor | None] = [None] * L
    d_mlp_list: list[torch.Tensor | None] = [None] * L

    handles = []
    for l in range(L):
        layer = adapter.layers[l]

        def _make_layer_hook(idx: int):
            def _hook(_mod, args, out):
                h_in_list[idx] = args[0].detach()
                h_out_list[idx] = (out[0] if isinstance(out, tuple) else out).detach()
            return _hook

        def _make_delta_hook(buf: list, idx: int):
            def _hook(_mod, _args, out):
                t = out[0] if isinstance(out, tuple) else out
                buf[idx] = t.detach()
            return _hook

        handles.append(layer.register_forward_hook(_make_layer_hook(l)))
        handles.append(adapter.attn_delta_module(l).register_forward_hook(_make_delta_hook(d_attn_list, l)))
        handles.append(adapter.mlp_delta_module(l).register_forward_hook(_make_delta_hook(d_mlp_list, l)))

    try:
        out = adapter.model(input_ids=input_ids, use_cache=False, output_attentions=False)
        logits_clean = out.logits
    finally:
        for h in handles:
            h.remove()

    clean_nats = interior_ce_nats(logits_clean, input_ids)
    clean_bpb = nats_to_bpb(clean_nats, n_pred_tokens, total_pred_bytes)

    # Verify IC-v3.2 residual identity at every layer (both absolute and relative to max |h_out|)
    resid_max_err = np.zeros(L, dtype=np.float64)
    resid_rel_err = np.zeros(L, dtype=np.float64)
    for l in range(L):
        reconstructed = h_in_list[l].float() + d_attn_list[l].float() + d_mlp_list[l].float()
        abs_err = float((h_out_list[l].float() - reconstructed).abs().max().item())
        scale = float(h_out_list[l].float().abs().max().item()) + 1e-8
        resid_max_err[l] = abs_err
        resid_rel_err[l] = abs_err / scale

    # Channel 1 metrics
    write_rms = np.zeros((L, 2), dtype=np.float64)      # [:, 0]=attn, [:, 1]=mlp
    rel_ratio = np.zeros((L, 2), dtype=np.float64)
    attn_share = np.zeros(L, dtype=np.float64)
    cos_stream = np.zeros((L, 2), dtype=np.float64)
    cos_am = np.zeros(L, dtype=np.float64)
    dla = np.zeros((L, 2), dtype=np.float64)
    lens_nats = np.zeros(L + 1, dtype=np.float64)       # index 0 = h_0 (embed-only), 1..L = h_1..h_L
    lens_bpb = np.zeros(L + 1, dtype=np.float64)

    # Channel 4 geometry on interior vs pos-0
    erank = np.zeros(L + 1, dtype=np.float64)
    erank_normed = np.zeros(L + 1, dtype=np.float64)
    outlier_pos0 = np.zeros(L + 1, dtype=np.float64)
    outlier_interior = np.zeros(L + 1, dtype=np.float64)

    # Mean vectors for Channel 2 mean-ablation (interior t >= 1)
    mean_deltas: dict[tuple[int, str], torch.Tensor] = {}

    # Prepare DLA target unembedding contrast vector: W_U[y_t] - mean(W_U)
    h_final = h_out_list[L - 1]
    W_u = adapter.unembed.weight.detach().float()  # (V, d)
    w_u_mean = W_u.mean(dim=0, keepdim=True)       # (1, d)
    targets_int = input_ids[:, 2:]                 # (B, T-2)
    w_tgt = W_u[targets_int] - w_u_mean            # (B, T-2, d)

    # Embed-only logit lens & geometry (h_0 = h_in_list[0])
    def _layer_geom(h: torch.Tensor, idx: int) -> None:
        h32 = h.float()
        h0 = h32[:, 0, :].abs()
        hi = h32[:, 1:, :].abs()
        outlier_pos0[idx] = float(h0.max().item() / (h0.median().item() + 1e-8))
        outlier_interior[idx] = float(hi.max().item() / (hi.median().item() + 1e-8))
        flat = h32[:, 1:, :].reshape(-1, h32.size(-1))
        if flat.size(0) > erank_max_tokens:
            step = max(1, flat.size(0) // erank_max_tokens)
            flat = flat[::step][:erank_max_tokens]
        erank[idx] = _effective_rank(flat)
        # Outlier-decoupled effective rank: standardize each feature channel by its std
        fc = flat - flat.mean(dim=0, keepdim=True)
        fstd = fc.std(dim=0, keepdim=True).clamp_min(1e-6)
        erank_normed[idx] = _effective_rank(fc / fstd)

    def _logit_lens(h: torch.Tensor) -> tuple[float, float]:
        # Compute on interior positions 1..T-2 only to save memory & compute
        hn = adapter.final_norm(h[:, 1:-1, :])
        lg = adapter.unembed(hn).float()
        ce = float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets_int.reshape(-1)).item())
        return ce, nats_to_bpb(ce, n_pred_tokens, total_pred_bytes)

    _layer_geom(h_in_list[0], 0)
    lens_nats[0], lens_bpb[0] = _logit_lens(h_in_list[0])

    eps = 1e-8
    for l in range(L):
        h_in = h_in_list[l]
        da = d_attn_list[l]
        dm = d_mlp_list[l]
        h_out = h_out_list[l]
        h_in_mlp = h_in if adapter.is_parallel else (h_in + da)

        # Store interior mean delta vectors for Channel 2
        mean_deltas[(l, "attn")] = da[:, 1:, :].mean(dim=(0, 1)).detach()
        mean_deltas[(l, "mlp")] = dm[:, 1:, :].mean(dim=(0, 1)).detach()

        da_i = da[:, 1:, :].float()
        dm_i = dm[:, 1:, :].float()
        ha_in_i = h_in[:, 1:, :].float()
        hm_in_i = h_in_mlp[:, 1:, :].float()

        # Write RMS
        wa = float(torch.sqrt(da_i.pow(2).mean()).item())
        wm = float(torch.sqrt(dm_i.pow(2).mean()).item())
        write_rms[l, 0] = wa
        write_rms[l, 1] = wm
        attn_share[l] = float((wa * wa) / (wa * wa + wm * wm + 1e-16))

        # Relative perturbation ratio
        na = da_i.norm(dim=-1)
        nm = dm_i.norm(dim=-1)
        nha = ha_in_i.norm(dim=-1)
        nhm = hm_in_i.norm(dim=-1)
        rel_ratio[l, 0] = float((na / (nha + eps)).mean().item())
        rel_ratio[l, 1] = float((nm / (nhm + eps)).mean().item())

        # Write-stream cosine alignment
        cos_stream[l, 0] = float(((da_i * ha_in_i).sum(dim=-1) / (na * nha + eps)).mean().item())
        cos_stream[l, 1] = float(((dm_i * hm_in_i).sum(dim=-1) / (nm * nhm + eps)).mean().item())

        # Within-layer Attn-MLP cosine
        cos_am[l] = float(((da_i * dm_i).sum(dim=-1) / (na * nm + eps)).mean().item())

        # DLA on predicting positions 1..T-2 -> target 2..T-1
        da_lin = adapter.apply_final_norm_linearized(da[:, 1:-1, :], h_final[:, 1:-1, :])
        dm_lin = adapter.apply_final_norm_linearized(dm[:, 1:-1, :], h_final[:, 1:-1, :])
        dla[l, 0] = float((da_lin * w_tgt).sum(dim=-1).mean().item())
        dla[l, 1] = float((dm_lin * w_tgt).sum(dim=-1).mean().item())

        # Logit lens & geometry at h_{l+1}
        _layer_geom(h_out, l + 1)
        lens_nats[l + 1], lens_bpb[l + 1] = _logit_lens(h_out)

    # Attention entropy & BOS-sink mass (compact pass on first attn_max_seqs sequences)
    attn_entropy = np.full((L, adapter.n_heads), np.nan, dtype=np.float64)
    attn_bos_mass = np.full((L, adapter.n_heads), np.nan, dtype=np.float64)
    if compute_attn_stats:
        sub_ids = input_ids[: min(B, attn_max_seqs)]
        out_a = adapter.model(input_ids=sub_ids, use_cache=False, output_attentions=True)
        attns = out_a.attentions
        if attns is not None and len(attns) == L:
            Ts = sub_ids.size(1)
            # Positions t >= 4 so normalized entropy denominator log(t+1) is meaningful
            t_idx = torch.arange(4, Ts, device=sub_ids.device, dtype=torch.float32)
            log_denom = torch.log(t_idx + 1.0).view(1, 1, -1)
            for l in range(L):
                a = attns[l].float()[:, :, 4:, :]  # (Bs, H, T-4, T)
                ent = -(a * torch.log(a.clamp_min(1e-12))).sum(dim=-1) / log_denom
                attn_entropy[l] = ent.mean(dim=(0, 2)).cpu().numpy()
                attn_bos_mass[l] = a[:, :, :, 0].mean(dim=(0, 2)).cpu().numpy()

    return {
        "clean_nats": clean_nats,
        "clean_bpb": clean_bpb,
        "resid_max_err": resid_max_err,
        "resid_rel_err": resid_rel_err,
        "write_rms": write_rms,
        "rel_ratio": rel_ratio,
        "attn_share": attn_share,
        "cos_stream": cos_stream,
        "cos_am": cos_am,
        "dla": dla,
        "lens_nats": lens_nats,
        "lens_bpb": lens_bpb,
        "erank": erank,
        "erank_normed": erank_normed,
        "outlier_pos0": outlier_pos0,
        "outlier_interior": outlier_interior,
        "attn_entropy": attn_entropy,
        "attn_bos_mass": attn_bos_mass,
        "mean_deltas": mean_deltas,
    }


# ---------------------------------------------------------------------------
# Channel 2 — Block-Level Causal Necessity & Pairwise Excess Joint Loss + IC-v3.3
# ---------------------------------------------------------------------------

def build_structured_pairs(n_layers: int) -> list[tuple[str, tuple[int, str], tuple[int, str]]]:
    """Build O(L) structured block pair set P (5L - 4 pairs)."""
    pairs: list[tuple[str, tuple[int, str], tuple[int, str]]] = []
    for l in range(n_layers):
        pairs.append((f"same_L{l}", (l, "attn"), (l, "mlp")))
    for l in range(n_layers - 1):
        pairs.append((f"adj_attn_L{l}_L{l+1}", (l, "attn"), (l + 1, "attn")))
        pairs.append((f"adj_mlp_L{l}_L{l+1}", (l, "mlp"), (l + 1, "mlp")))
        pairs.append((f"adj_am_L{l}_L{l+1}", (l, "attn"), (l + 1, "mlp")))
        pairs.append((f"adj_ma_L{l}_L{l+1}", (l, "mlp"), (l + 1, "attn")))
    return pairs


@torch.no_grad()
def eval_with_ablation(
    adapter: UnifiedAdapter,
    input_ids: torch.Tensor,
    ablate_blocks: list[tuple[int, str]],
    mode: str,
    mean_deltas: dict[tuple[int, str], torch.Tensor],
    *,
    all_positions: bool = False,
) -> float:
    """Evaluate interior CE (nats) while ablating the specified list of (layer_idx, 'attn'|'mlp') blocks."""
    handles = []
    for l, kind in ablate_blocks:
        mod = adapter.attn_delta_module(l) if kind == "attn" else adapter.mlp_delta_module(l)
        mu = mean_deltas[(l, kind)] if mode == "mean" else None

        def _make_hook(m_vec: torch.Tensor | None):
            def _hook(_m, _args, out):
                t = out[0] if isinstance(out, tuple) else out
                rep = t.clone()
                if all_positions:
                    if m_vec is None:
                        rep.zero_()
                    else:
                        rep.copy_(m_vec.to(dtype=t.dtype, device=t.device).view(1, 1, -1).expand_as(rep))
                else:
                    if m_vec is None:
                        rep[:, 1:, :].zero_()
                    else:
                        rep[:, 1:, :] = m_vec.to(dtype=t.dtype, device=t.device).view(1, 1, -1)
                if isinstance(out, tuple):
                    return (rep,) + out[1:]
                return rep
            return _hook

        handles.append(mod.register_forward_hook(_make_hook(mu)))

    try:
        out = adapter.model(input_ids=input_ids, use_cache=False, output_attentions=False)
        return interior_ce_nats(out.logits, input_ids)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_channel2(
    adapter: UnifiedAdapter,
    causal_ids: torch.Tensor,
    causal_pred_bytes: int,
    mean_deltas: dict[tuple[int, str], torch.Tensor],
    *,
    run_pairs: bool = True,
) -> dict[str, Any]:
    """Run single-block mean/zero ablation, structured pairwise Excess Joint Loss, and IC-v3.3."""
    L = adapter.n_layers
    B, T = causal_ids.shape
    n_pred = B * (T - 2)

    # Clean loss on D_causal + IC-v3.3a (empty ablation == clean)
    clean_direct = interior_ce_nats(
        adapter.model(input_ids=causal_ids, use_cache=False, output_attentions=False).logits, causal_ids
    )
    clean_empty_ablate = eval_with_ablation(adapter, causal_ids, [], "mean", mean_deltas)
    ic_empty_diff = abs(clean_direct - clean_empty_ablate)

    # IC-v3.3b: zero-ablating ALL 2L blocks (all positions) == embed-only logit lens
    all_blocks = [(l, k) for l in range(L) for k in ("attn", "mlp")]
    all_zero_nats = eval_with_ablation(
        adapter, causal_ids, all_blocks, "zero", mean_deltas, all_positions=True
    )
    # Direct embed-only baseline via layer-0 input hook
    h0_holder: list[torch.Tensor] = []
    h_hook = adapter.layers[0].register_forward_hook(lambda _m, args, _o: h0_holder.append(args[0].detach()))
    try:
        adapter.model(input_ids=causal_ids, use_cache=False, output_attentions=False)
    finally:
        h_hook.remove()
    h0 = h0_holder[0]
    embed_logits = adapter.unembed(adapter.final_norm(h0[:, 1:-1, :])).float()
    embed_only_nats = float(
        F.cross_entropy(embed_logits.reshape(-1, embed_logits.size(-1)), causal_ids[:, 2:].reshape(-1)).item()
    )
    ic_allzero_diff = abs(all_zero_nats - embed_only_nats)

    # Single-block mean & zero ablation
    single_mean_dnats = np.zeros((L, 2), dtype=np.float64)
    single_zero_dnats = np.zeros((L, 2), dtype=np.float64)
    for l in range(L):
        for ki, kind in enumerate(("attn", "mlp")):
            loss_m = eval_with_ablation(adapter, causal_ids, [(l, kind)], "mean", mean_deltas)
            loss_z = eval_with_ablation(adapter, causal_ids, [(l, kind)], "zero", mean_deltas)
            single_mean_dnats[l, ki] = loss_m - clean_direct
            single_zero_dnats[l, ki] = loss_z - clean_direct

    scale_bpb = n_pred / (max(1, causal_pred_bytes) * math.log(2.0))
    single_mean_dbpb = single_mean_dnats * scale_bpb
    single_zero_dbpb = single_zero_dnats * scale_bpb

    # Pairwise structured interactions (mean ablation)
    pairs = build_structured_pairs(L)
    pair_names: list[str] = []
    pair_joint_dnats = np.zeros(len(pairs), dtype=np.float64)
    pair_excess_nats = np.zeros(len(pairs), dtype=np.float64)
    pair_excess_norm = np.zeros(len(pairs), dtype=np.float64)

    if run_pairs:
        for pi, (pname, b1, b2) in enumerate(pairs):
            pair_names.append(pname)
            lj = eval_with_ablation(adapter, causal_ids, [b1, b2], "mean", mean_deltas)
            dj = lj - clean_direct
            d1 = single_mean_dnats[b1[0], 0 if b1[1] == "attn" else 1]
            d2 = single_mean_dnats[b2[0], 0 if b2[1] == "attn" else 1]
            exc = dj - d1 - d2
            pair_joint_dnats[pi] = dj
            pair_excess_nats[pi] = exc
            pair_excess_norm[pi] = exc / max(abs(d1) + abs(d2), 1e-3)
    else:
        pair_names = [p[0] for p in pairs]

    return {
        "causal_clean_nats": clean_direct,
        "causal_clean_bpb": clean_direct * scale_bpb,
        "ic_v3_3a_empty_diff": ic_empty_diff,
        "ic_v3_3b_allzero_diff": ic_allzero_diff,
        "single_mean_dnats": single_mean_dnats,
        "single_zero_dnats": single_zero_dnats,
        "single_mean_dbpb": single_mean_dbpb,
        "single_zero_dbpb": single_zero_dbpb,
        "pair_names": pair_names,
        "pair_joint_dnats": pair_joint_dnats,
        "pair_excess_nats": pair_excess_nats,
        "pair_excess_bpb": pair_excess_nats * scale_bpb,
        "pair_excess_norm": pair_excess_norm,
    }


# ---------------------------------------------------------------------------
# Channel 3 — Tokenizer-Agnostic Matched Micro-Probe Battery
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_continuations_batched(
    model: nn.Module,
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int = 32,
) -> list[tuple[float, float, int]]:
    """Return list of (sum_logprob_nats, bpb, n_target_tokens) for each (context, target) string pair.

    Uses character offset mapping to locate the exact target token span with zero BPE-boundary bugs.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    encoded: list[tuple[list[int], list[int], int]] = []
    for ctx, tgt in pairs:
        full = ctx + tgt
        cut = len(ctx)
        enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
        ids = list(enc["input_ids"])
        offs = list(enc["offset_mapping"])
        if bos_id is not None and (len(ids) == 0 or ids[0] != bos_id):
            ids = [bos_id] + ids
            offs = [(0, 0)] + offs
        tgt_pos = [i for i, (s, e) in enumerate(offs) if i >= 1 and e > cut and e > s]
        if not tgt_pos:
            # Fallback if tokenizer produced zero-width offset
            tgt_pos = [len(ids) - 1]
        n_bytes = max(1, len(tgt.encode("utf-8")))
        encoded.append((ids, tgt_pos, n_bytes))

    results: list[tuple[float, float, int]] = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        max_len = max(len(x[0]) for x in chunk)
        inp = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for bi, (ids, _, _) in enumerate(chunk):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            mask[bi, : len(ids)] = 1
        logits = model(input_ids=inp, attention_mask=mask, use_cache=False, output_attentions=False).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        for bi, (ids, tgt_pos, n_bytes) in enumerate(chunk):
            s_lp = 0.0
            for pos in tgt_pos:
                s_lp += float(logp[bi, pos - 1, ids[pos]].item())
            bpb = float((-s_lp) / (n_bytes * math.log(2.0)))
            results.append((s_lp, bpb, len(tgt_pos)))
    return results


@torch.no_grad()
def run_channel3(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_per_probe: int = 16,
    batch_size: int = 32,
) -> dict[str, dict[str, float]]:
    """Evaluate all 8 matched micro-probe contrasts in tokenizer-agnostic bpb and log-prob margins."""
    probe_dict = build_channel3_probes(max_per_probe=max_per_probe)
    summary: dict[str, dict[str, float]] = {}

    for name, items in probe_dict.items():
        if not items:
            continue
        kind = items[0].kind
        if kind == "ctx_pair":
            sc_a = score_continuations_batched(model, tokenizer, [(x.ctx_a, x.target_a) for x in items], device, batch_size)
            sc_b = score_continuations_batched(model, tokenizer, [(x.ctx_b, x.target_a) for x in items], device, batch_size)
            bpb_a = np.array([x[1] for x in sc_a])
            bpb_b = np.array([x[1] for x in sc_b])
            diff = bpb_b - bpb_a  # positive => clean/long context beats shuffled/short
            summary[name] = {
                "n": float(len(items)),
                "bpb_a_mean": float(bpb_a.mean()),
                "bpb_b_mean": float(bpb_b.mean()),
                "delta_bpb_b_minus_a": float(diff.mean()),
                "delta_bpb_se": float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0,
                "win_rate_a_beats_b": float((diff > 0).mean()),
            }
        elif kind == "tgt_pair":
            sc_a = score_continuations_batched(model, tokenizer, [(x.ctx_a, x.target_a) for x in items], device, batch_size)
            sc_b = score_continuations_batched(model, tokenizer, [(x.ctx_a, x.target_b) for x in items], device, batch_size)
            m = np.array([a[0] - b[0] for a, b in zip(sc_a, sc_b)])
            mbpb = np.array([b[1] - a[1] for a, b in zip(sc_a, sc_b)])  # positive => good has lower bpb
            summary[name] = {
                "n": float(len(items)),
                "bpb_good_mean": float(np.mean([x[1] for x in sc_a])),
                "bpb_foil_mean": float(np.mean([x[1] for x in sc_b])),
                "margin_bpb_mean": float(mbpb.mean()),
                "margin_nats_mean": float(m.mean()),
                "margin_nats_se": float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else 0.0,
                "acc_good_gt_foil": float((mbpb > 0).mean()),
            }
        elif kind == "dual_tgt_pair":
            sa_g = score_continuations_batched(model, tokenizer, [(x.ctx_a, x.target_a) for x in items], device, batch_size)
            sa_f = score_continuations_batched(model, tokenizer, [(x.ctx_a, x.target_b) for x in items], device, batch_size)
            sb_g = score_continuations_batched(model, tokenizer, [(x.ctx_b, x.target_a) for x in items], device, batch_size)
            sb_f = score_continuations_batched(model, tokenizer, [(x.ctx_b, x.target_b) for x in items], device, batch_size)
            ma = np.array([g[0] - f[0] for g, f in zip(sa_g, sa_f)])
            mb = np.array([g[0] - f[0] for g, f in zip(sb_g, sb_f)])
            ma_bpb = np.array([f[1] - g[1] for g, f in zip(sa_g, sa_f)])
            mb_bpb = np.array([f[1] - g[1] for g, f in zip(sb_g, sb_f)])
            shift = ma - mb
            shift_bpb = ma_bpb - mb_bpb
            summary[name] = {
                "n": float(len(items)),
                "margin_bpb_ctx_a": float(ma_bpb.mean()),
                "margin_bpb_ctx_b": float(mb_bpb.mean()),
                "shift_bpb_a_minus_b": float(shift_bpb.mean()),
                "margin_ctx_a_mean": float(ma.mean()),
                "margin_ctx_b_mean": float(mb.mean()),
                "margin_shift_a_minus_b": float(shift.mean()),
                "margin_shift_se": float(shift.std(ddof=1) / math.sqrt(len(shift))) if len(shift) > 1 else 0.0,
                "acc_ctx_a": float((ma_bpb > 0).mean()),
                "acc_ctx_b": float((mb_bpb > 0).mean()),
            }
        elif kind == "arith_pair":
            sc_nc = score_continuations_batched(model, tokenizer, [(x.ctx_a, x.target_a) for x in items], device, batch_size)
            sc_c = score_continuations_batched(model, tokenizer, [(x.ctx_b, x.target_b) for x in items], device, batch_size)
            bpb_nc = np.array([x[1] for x in sc_nc])
            bpb_c = np.array([x[1] for x in sc_c])
            gap = bpb_c - bpb_nc  # positive => carry is harder than no-carry
            summary[name] = {
                "n": float(len(items)),
                "bpb_nocarry_mean": float(bpb_nc.mean()),
                "bpb_carry_mean": float(bpb_c.mean()),
                "carry_penalty_bpb": float(gap.mean()),
                "carry_penalty_se": float(gap.std(ddof=1) / math.sqrt(len(gap))) if len(gap) > 1 else 0.0,
            }
    return summary


# ---------------------------------------------------------------------------
# Channel 4 — Tokenizer- & Width-Agnostic Prompt-Endpoint Gram Matrices
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_channel4_gram(
    adapter: UnifiedAdapter,
    tokenizer: Any,
    device: torch.device,
    n_prompts: int = 64,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute centered Gram matrices K_l = H_l H_l^T of shape (L+1, M, M) at final prompt tokens."""
    prompts = build_gram_prompts(n_prompts=n_prompts)
    M = len(prompts)
    L = adapter.n_layers
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    H_all = [np.zeros((M, adapter.d_model), dtype=np.float32) for _ in range(L + 1)]
    encoded: list[list[int]] = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if bos_id is not None and (len(ids) == 0 or ids[0] != bos_id):
            ids = [bos_id] + ids
        encoded.append(ids)

    for start in range(0, M, batch_size):
        chunk = encoded[start : start + batch_size]
        max_len = max(len(x) for x in chunk)
        inp = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        last_idx = []
        for bi, ids in enumerate(chunk):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            mask[bi, : len(ids)] = 1
            last_idx.append(len(ids) - 1)
        last_t = torch.tensor(last_idx, dtype=torch.long, device=device)
        row_idx = torch.arange(len(chunk), device=device)

        h0_buf: list[torch.Tensor] = []
        hout_buf: list[torch.Tensor | None] = [None] * L
        handles = [adapter.layers[0].register_forward_hook(lambda _m, a, _o: h0_buf.append(a[0].detach()))]
        for l in range(L):
            def _mk(idx: int):
                def _hk(_m, _a, o):
                    hout_buf[idx] = (o[0] if isinstance(o, tuple) else o).detach()
                return _hk
            handles.append(adapter.layers[l].register_forward_hook(_mk(l)))
        try:
            adapter.model(input_ids=inp, attention_mask=mask, use_cache=False, output_attentions=False)
        finally:
            for h in handles:
                h.remove()

        H_all[0][start : start + len(chunk)] = h0_buf[0][row_idx, last_t].float().cpu().numpy()
        for l in range(L):
            H_all[l + 1][start : start + len(chunk)] = hout_buf[l][row_idx, last_t].float().cpu().numpy()

    grams = np.zeros((L + 1, M, M), dtype=np.float32)
    for l in range(L + 1):
        hc = H_all[l].astype(np.float64)
        hc = hc - hc.mean(axis=0, keepdims=True)
        grams[l] = (hc @ hc.T).astype(np.float32)
    return grams


def linear_cka_from_grams(K_a: np.ndarray, K_b: np.ndarray) -> np.ndarray:
    """Per-layer linear CKA given centered Gram tensors of shape (L+1, M, M)."""
    L1 = K_a.shape[0]
    out = np.zeros(L1, dtype=np.float64)
    for l in range(L1):
        a = K_a[l].astype(np.float64)
        b = K_b[l].astype(np.float64)
        na = np.linalg.norm(a, "fro")
        nb = np.linalg.norm(b, "fro")
        out[l] = float(np.sum(a * b) / (na * nb + 1e-16))
    return out


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------

def scan_single_revision(
    model_id: str,
    revision: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    print(f"\n=== Scanning {model_id} @ {revision} (device={device}, dtype={dtype}) ===", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    t_load0 = time.perf_counter()
    # Use eager attention so output_attentions=True works cleanly across Pythia/OLMo-1/OLMo-2
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    t_load = time.perf_counter() - t_load0

    adapter = UnifiedAdapter(model)
    passages = build_core_passages(args.core_seqs)
    core_ids, core_bytes = prepare_core_batch(tok, passages, args.seq_len, device)
    causal_ids, causal_bytes = prepare_core_batch(tok, passages[: args.causal_seqs], args.seq_len, device)

    # Channel 1 + core Ch4 + IC-v3.2
    t_ch1_0 = time.perf_counter()
    ch1 = run_channel1_and_core_ch4(
        adapter,
        core_ids,
        core_bytes,
        compute_attn_stats=not args.skip_attn_stats,
        attn_max_seqs=min(args.core_seqs, 16),
    )
    t_ch1 = time.perf_counter() - t_ch1_0

    # IC-v3.1 Determinism check: repeat core pass (without attn stats) and measure max diff on Ch1
    ch1_rep = run_channel1_and_core_ch4(adapter, core_ids, core_bytes, compute_attn_stats=False)
    ic_det_diff = float(
        max(
            abs(ch1["clean_nats"] - ch1_rep["clean_nats"]),
            float(np.max(np.abs(ch1["write_rms"] - ch1_rep["write_rms"]))),
            float(np.max(np.abs(ch1["lens_bpb"] - ch1_rep["lens_bpb"]))),
            float(np.max(np.abs(ch1["dla"] - ch1_rep["dla"]))),
        )
    )

    # Channel 2 + IC-v3.3
    t_ch2_0 = time.perf_counter()
    ch2 = run_channel2(
        adapter,
        causal_ids,
        causal_bytes,
        ch1["mean_deltas"],
        run_pairs=not args.skip_pairs,
    )
    t_ch2 = time.perf_counter() - t_ch2_0

    # Channel 3
    t_ch3_0 = time.perf_counter()
    ch3 = run_channel3(model, tok, device, max_per_probe=args.probe_n, batch_size=args.batch_size)
    t_ch3 = time.perf_counter() - t_ch3_0

    # Channel 4 Gram matrices
    t_ch4_0 = time.perf_counter()
    grams = run_channel4_gram(adapter, tok, device, n_prompts=args.gram_m, batch_size=args.batch_size)
    t_ch4 = time.perf_counter() - t_ch4_0

    t_total = time.perf_counter() - t0
    tol_det = 1e-6 if dtype == torch.float32 else 1e-4
    tol_rel = 1e-6 if dtype == torch.float32 else 1e-2
    tol_allzero = 1e-5 if dtype == torch.float32 else 1e-2

    ic_report = {
        "IC_v3_1_determinism_max_diff": ic_det_diff,
        "IC_v3_1_pass": ic_det_diff <= tol_det,
        "IC_v3_2_residual_identity_max_abs_err": float(np.max(ch1["resid_max_err"])),
        "IC_v3_2_residual_identity_max_rel_err": float(np.max(ch1["resid_rel_err"])),
        "IC_v3_2_pass": float(np.max(ch1["resid_rel_err"])) <= tol_rel,
        "IC_v3_3a_empty_ablation_diff": float(ch2["ic_v3_3a_empty_diff"]),
        "IC_v3_3a_pass": float(ch2["ic_v3_3a_empty_diff"]) == 0.0,
        "IC_v3_3b_allzero_vs_embed_diff": float(ch2["ic_v3_3b_allzero_diff"]),
        "IC_v3_3b_pass": float(ch2["ic_v3_3b_allzero_diff"]) <= tol_allzero,
    }
    ic_report["all_pass"] = bool(
        ic_report["IC_v3_1_pass"]
        and ic_report["IC_v3_2_pass"]
        and ic_report["IC_v3_3a_pass"]
        and ic_report["IC_v3_3b_pass"]
    )

    # Save compact .npz + .json
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{model_id.replace('/', '__')}__{revision}"
    npz_path = out_dir / f"{slug}.npz"
    json_path = out_dir / f"{slug}.json"

    np.savez_compressed(
        npz_path,
        write_rms=ch1["write_rms"],
        rel_ratio=ch1["rel_ratio"],
        attn_share=ch1["attn_share"],
        cos_stream=ch1["cos_stream"],
        cos_am=ch1["cos_am"],
        dla=ch1["dla"],
        lens_nats=ch1["lens_nats"],
        lens_bpb=ch1["lens_bpb"],
        erank=ch1["erank"],
        erank_normed=ch1["erank_normed"],
        outlier_pos0=ch1["outlier_pos0"],
        outlier_interior=ch1["outlier_interior"],
        attn_entropy=ch1["attn_entropy"],
        attn_bos_mass=ch1["attn_bos_mass"],
        single_mean_dnats=ch2["single_mean_dnats"],
        single_zero_dnats=ch2["single_zero_dnats"],
        single_mean_dbpb=ch2["single_mean_dbpb"],
        single_zero_dbpb=ch2["single_zero_dbpb"],
        pair_names=np.array(ch2["pair_names"]),
        pair_joint_dnats=ch2["pair_joint_dnats"],
        pair_excess_nats=ch2["pair_excess_nats"],
        pair_excess_bpb=ch2["pair_excess_bpb"],
        pair_excess_norm=ch2["pair_excess_norm"],
        gram_matrices=grams,
    )

    # Top redundant (E > 0) and top serial/AND-gate (E < 0) block pairs for readable JSON summary
    order_exc = np.argsort(ch2["pair_excess_nats"])
    top_serial = [
        {
            "pair": ch2["pair_names"][i],
            "E_nats": float(ch2["pair_excess_nats"][i]),
            "E_norm": float(ch2["pair_excess_norm"][i]),
            "joint_dnats": float(ch2["pair_joint_dnats"][i]),
        }
        for i in order_exc[:5]
    ]
    top_redundant = [
        {
            "pair": ch2["pair_names"][i],
            "E_nats": float(ch2["pair_excess_nats"][i]),
            "E_norm": float(ch2["pair_excess_norm"][i]),
            "joint_dnats": float(ch2["pair_joint_dnats"][i]),
        }
        for i in order_exc[-5:][::-1]
    ]

    summary = {
        "model_id": model_id,
        "revision": revision,
        "model_type": adapter.model_type,
        "is_parallel": adapter.is_parallel,
        "is_post_norm": adapter.is_post_norm,
        "n_layers": adapter.n_layers,
        "d_model": adapter.d_model,
        "n_heads": adapter.n_heads,
        "vocab_size": adapter.vocab_size,
        "config": {
            "device": str(device),
            "dtype": str(dtype),
            "core_seqs": args.core_seqs,
            "causal_seqs": args.causal_seqs,
            "seq_len": args.seq_len,
            "probe_n": args.probe_n,
            "gram_m": args.gram_m,
            "skip_pairs": args.skip_pairs,
        },
        "timing_s": {
            "load": round(t_load, 2),
            "ch1_and_core_ch4": round(t_ch1, 2),
            "ch2_causal": round(t_ch2, 2),
            "ch3_probes": round(t_ch3, 2),
            "ch4_gram": round(t_ch4, 2),
            "total": round(t_total, 2),
        },
        "artifacts": {
            "npz_bytes": int(npz_path.stat().st_size),
            "npz_path": str(npz_path.name),
        },
        "ic_v3": ic_report,
        "ch1_summary": {
            "core_clean_nats": round(ch1["clean_nats"], 5),
            "core_clean_bpb": round(ch1["clean_bpb"], 5),
            "embed_only_bpb": round(float(ch1["lens_bpb"][0]), 5),
            "attn_share_by_layer": [round(float(x), 4) for x in ch1["attn_share"]],
            "cos_stream_attn_by_layer": [round(float(x), 4) for x in ch1["cos_stream"][:, 0]],
            "cos_stream_mlp_by_layer": [round(float(x), 4) for x in ch1["cos_stream"][:, 1]],
            "cos_am_by_layer": [round(float(x), 4) for x in ch1["cos_am"]],
            "lens_bpb_by_layer": [round(float(x), 4) for x in ch1["lens_bpb"]],
            "dla_attn_by_layer": [round(float(x), 4) for x in ch1["dla"][:, 0]],
            "dla_mlp_by_layer": [round(float(x), 4) for x in ch1["dla"][:, 1]],
        },
        "ch2_summary": {
            "causal_clean_bpb": round(ch2["causal_clean_bpb"], 5),
            "single_mean_dbpb_attn": [round(float(x), 4) for x in ch2["single_mean_dbpb"][:, 0]],
            "single_mean_dbpb_mlp": [round(float(x), 4) for x in ch2["single_mean_dbpb"][:, 1]],
            "single_zero_dbpb_attn": [round(float(x), 4) for x in ch2["single_zero_dbpb"][:, 0]],
            "single_zero_dbpb_mlp": [round(float(x), 4) for x in ch2["single_zero_dbpb"][:, 1]],
            "top_serial_pairs": top_serial,
            "top_redundant_pairs": top_redundant,
        },
        "ch3_probes": ch3,
        "ch4_summary": {
            "erank_by_layer": [round(float(x), 2) for x in ch1["erank"]],
            "erank_normed_by_layer": [round(float(x), 2) for x in ch1["erank_normed"]],
            "outlier_pos0_by_layer": [round(float(x), 2) for x in ch1["outlier_pos0"]],
            "outlier_interior_by_layer": [round(float(x), 2) for x in ch1["outlier_interior"]],
            "attn_entropy_mean_by_layer": [round(float(np.nanmean(x)), 4) for x in ch1["attn_entropy"]],
            "attn_bos_mass_mean_by_layer": [round(float(np.nanmean(x)), 4) for x in ch1["attn_bos_mass"]],
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"  DONE in {t_total:.1f}s (load={t_load:.1f}s ch1={t_ch1:.1f}s ch2={t_ch2:.1f}s "
        f"ch3={t_ch3:.1f}s ch4={t_ch4:.1f}s) | npz={npz_path.stat().st_size/1024:.1f} KB | "
        f"IC-v3 all_pass={ic_report['all_pass']} | clean_bpb={ch1['clean_bpb']:.4f}",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary, grams


def main() -> int:
    ap = argparse.ArgumentParser(description="Tycho v3 Stage 1 unified 4-channel checkpoint scanner")
    ap.add_argument("--model", required=True, help="HuggingFace model ID (e.g. EleutherAI/pythia-160m)")
    ap.add_argument("--revisions", nargs="+", required=True, help="One or more HF revisions (e.g. step2000 step143000)")
    ap.add_argument("--out-dir", default="work/v3_proto", help="Output directory for .npz and .json summaries")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--threads", type=int, default=0, help="CPU torch threads (0 = leave default)")
    ap.add_argument("--core-seqs", type=int, default=32)
    ap.add_argument("--causal-seqs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--probe-n", type=int, default=16)
    ap.add_argument("--gram-m", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--skip-attn-stats", action="store_true")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    out_dir = Path(args.out_dir)
    prev_rev: str | None = None
    prev_gram: np.ndarray | None = None
    prev_bpb: float | None = None
    velocities: list[dict[str, Any]] = []

    for rev in args.revisions:
        summary, gram = scan_single_revision(args.model, rev, args, device, dtype, out_dir)
        if not summary["ic_v3"]["all_pass"]:
            print("WARNING: IC-v3 failed:", summary["ic_v3"], file=sys.stderr)
        if prev_gram is not None and prev_rev is not None:
            cka = linear_cka_from_grams(prev_gram, gram)
            vel = 1.0 - cka
            dbpb = abs(summary["ch1_summary"]["core_clean_bpb"] - (prev_bpb or 0.0))
            v_rec = {
                "from": prev_rev,
                "to": rev,
                "cka_by_layer": [round(float(x), 4) for x in cka],
                "velocity_1_minus_cka": [round(float(x), 4) for x in vel],
                "abs_delta_bpb": round(dbpb, 5),
                "dark_ratio_by_layer": [round(float(v / max(dbpb, 1e-3)), 3) for v in vel],
            }
            velocities.append(v_rec)
            print(f"  Ch4 CKA({prev_rev} -> {rev}): {[round(float(x), 3) for x in cka]}")
        prev_rev = rev
        prev_gram = gram
        prev_bpb = summary["ch1_summary"]["core_clean_bpb"]

    if velocities:
        v_path = out_dir / f"{args.model.replace('/', '__')}__cka_velocity.json"
        v_path.write_text(json.dumps(velocities, indent=2) + "\n")
        print(f"Wrote CKA velocity summary: {v_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
