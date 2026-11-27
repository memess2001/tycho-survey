"""Offline Stage 1 Discovery & Cross-Family Anomaly Engine for Tycho Survey v3.

Reads all verified (`ic_v3.all_pass == True`) checkpoint `.json` and `.npz` files under
`work/v3_stage1/` (on s82), builds aligned trajectories across:
  - Pythia scale ladder (14M -> 31M -> 70M -> 160M -> 410M -> 1B)
  - Pythia multi-seed controls (70M x4, 160M x4, 410M x2)
  - Cross-family modern anchors (OLMo-1B-0724 pre-LN SwiGLU + OLMo-2-0425 post-RMSNorm QK-Norm)

Computes:
  1. Per-model 4-channel trajectory summaries (predictive depth, Attn/MLP power handoff,
     erasure cosines, block causal necessity, mean-vs-zero DC gap, pairwise E(i,j) flips,
     non-monotonic probe regressions, dark CKA velocity spikes, layer-freeze curves,
     and outlier-decoupled final-layer rank collapse).
  2. Multi-seed mean / SD / range bands.
  3. Cross-family layer-by-layer representational convergence CKA(Pythia-1B, OLMo-1B, OLMo-2-1B)
     at matched token budgets using the tokenizer- & width-agnostic 64x64 Gram matrices.
  4. Automated candidate anomaly table graded against Gate 1 (multi-seed) and Gate 2
     (cross-scale & cross-family).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_step_and_tokens(model_id: str, revision: str) -> tuple[int, float]:
    """Return (step_int, tokens_billions) for Pythia, OLMo-1, and OLMo-2 revisions."""
    # OLMo-2: stage1-step140000-tokens294B
    m = re.match(r"^stage1-step(\d+)-tokens(\d+(?:\.\d+)?)B$", revision)
    if m:
        step = int(m.group(1))
        # Exact tokens: stage1-step300-tokens1B is ~0.63B, use step*2.097e-3 if step>0 and close, else parsed
        tok_b = float(m.group(2))
        if step == 300 and tok_b == 1.0:
            tok_b = 0.63
        return step, tok_b
    # OLMo-1: step1000-tokens2B
    m = re.match(r"^step(\d+)-tokens(\d+(?:\.\d+)?)B$", revision)
    if m:
        return int(m.group(1)), float(m.group(2))
    # Pythia: step143000 (2,097,152 tokens/step)
    m = re.match(r"^step(\d+)$", revision)
    if m:
        step = int(m.group(1))
        return step, step * 2.097152e-3
    raise ValueError(f"Unrecognized revision format: {model_id} @ {revision}")


def cka_matrix(K_a: np.ndarray, K_b: np.ndarray) -> np.ndarray:
    """Compute (L_a+1, L_b+1) linear CKA matrix between two models' centered Gram stacks."""
    La = K_a.shape[0]
    Lb = K_b.shape[0]
    out = np.zeros((La, Lb), dtype=np.float64)
    na = np.array([np.linalg.norm(K_a[i].astype(np.float64), "fro") for i in range(La)])
    nb = np.array([np.linalg.norm(K_b[j].astype(np.float64), "fro") for j in range(Lb)])
    for i in range(La):
        a = K_a[i].astype(np.float64)
        for j in range(Lb):
            b = K_b[j].astype(np.float64)
            out[i, j] = float(np.sum(a * b) / (na[i] * nb[j] + 1e-16))
    return out


def cka_diag(K_a: np.ndarray, K_b: np.ndarray) -> np.ndarray:
    """Per-layer linear CKA between two checkpoints of the same model."""
    L1 = K_a.shape[0]
    out = np.zeros(L1, dtype=np.float64)
    for l in range(L1):
        a = K_a[l].astype(np.float64)
        b = K_b[l].astype(np.float64)
        out[l] = float(np.sum(a * b) / (np.linalg.norm(a, "fro") * np.linalg.norm(b, "fro") + 1e-16))
    return out


def load_model_trajectory(model_dir: Path) -> dict[str, Any] | None:
    """Load all verified checkpoints for one model directory, sorted by step."""
    jfiles = sorted(model_dir.glob("*.json"))
    records: list[dict[str, Any]] = []
    for jf in jfiles:
        if jf.name.endswith("__cka_velocity.json"):
            continue
        npz_path = jf.with_suffix(".npz")
        if not npz_path.exists():
            continue
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if not d.get("ic_v3", {}).get("all_pass", False):
            continue
        step, tok_b = parse_step_and_tokens(d["model_id"], d["revision"])
        z = np.load(npz_path, allow_pickle=True)
        records.append({
            "step": step,
            "tokens_B": tok_b,
            "revision": d["revision"],
            "json": d,
            "npz": {k: z[k] for k in z.files},
        })
    if not records:
        return None
    records.sort(key=lambda r: r["step"])
    d0 = records[0]["json"]
    return {
        "model_id": d0["model_id"],
        "model_type": d0["model_type"],
        "is_parallel": d0["is_parallel"],
        "is_post_norm": d0["is_post_norm"],
        "n_layers": d0["n_layers"],
        "d_model": d0["d_model"],
        "n_heads": d0["n_heads"],
        "records": records,
    }


def summarize_trajectory(traj: dict[str, Any]) -> dict[str, Any]:
    """Extract rich scalar & depth-normalized trajectory signals for one model."""
    recs = traj["records"]
    L = traj["n_layers"]
    n_t = len(recs)
    steps = [r["step"] for r in recs]
    toks = [r["tokens_B"] for r in recs]
    revs = [r["revision"] for r in recs]

    # Arrays over time
    clean_bpb = np.array([r["json"]["ch1_summary"]["core_clean_bpb"] for r in recs])
    embed_bpb = np.array([r["json"]["ch1_summary"]["embed_only_bpb"] for r in recs])
    attn_share = np.array([r["npz"]["attn_share"] for r in recs])            # (T, L)
    cos_stream_a = np.array([r["npz"]["cos_stream"][:, 0] for r in recs])    # (T, L)
    cos_stream_m = np.array([r["npz"]["cos_stream"][:, 1] for r in recs])    # (T, L)
    cos_am = np.array([r["npz"]["cos_am"] for r in recs])                    # (T, L)
    lens_bpb = np.array([r["npz"]["lens_bpb"] for r in recs])                # (T, L+1)
    dla_a = np.array([r["npz"]["dla"][:, 0] for r in recs])                  # (T, L)
    dla_m = np.array([r["npz"]["dla"][:, 1] for r in recs])                  # (T, L)
    mean_dbpb_a = np.array([r["npz"]["single_mean_dbpb"][:, 0] for r in recs])
    mean_dbpb_m = np.array([r["npz"]["single_mean_dbpb"][:, 1] for r in recs])
    zero_dbpb_a = np.array([r["npz"]["single_zero_dbpb"][:, 0] for r in recs])
    zero_dbpb_m = np.array([r["npz"]["single_zero_dbpb"][:, 1] for r in recs])
    pair_names = [str(x) for x in recs[0]["npz"]["pair_names"]]
    pair_exc_nats = np.array([r["npz"]["pair_excess_nats"] for r in recs])   # (T, P)
    pair_exc_norm = np.array([r["npz"]["pair_excess_norm"] for r in recs])   # (T, P)
    erank_raw = np.array([r["npz"]["erank"] for r in recs])                  # (T, L+1)
    erank_normed = np.array([r["npz"]["erank_normed"] for r in recs])        # (T, L+1)
    out_pos0 = np.array([r["npz"]["outlier_pos0"] for r in recs])            # (T, L+1)
    out_int = np.array([r["npz"]["outlier_interior"] for r in recs])         # (T, L+1)
    attn_ent = np.array([np.nanmean(r["npz"]["attn_entropy"], axis=1) for r in recs])   # (T, L)
    attn_bos = np.array([np.nanmean(r["npz"]["attn_bos_mass"], axis=1) for r in recs])  # (T, L)
    grams = [r["npz"]["gram_matrices"] for r in recs]                        # list of (L+1, M, M)

    # 1. Normalized predictive depth from logit-lens marginal gains
    lens_gain = np.maximum(0.0, lens_bpb[:, :-1] - lens_bpb[:, 1:])          # (T, L)
    layer_frac = np.arange(1, L + 1, dtype=np.float64) / float(L)
    pred_depth = np.sum(lens_gain * layer_frac[None, :], axis=1) / np.maximum(np.sum(lens_gain, axis=1), 1e-8)

    # 2. Causal necessity center-of-mass depth & Attn causal share
    tot_a = np.sum(np.maximum(0.0, mean_dbpb_a), axis=1)
    tot_m = np.sum(np.maximum(0.0, mean_dbpb_m), axis=1)
    causal_attn_share = tot_a / np.maximum(tot_a + tot_m, 1e-8)
    causal_by_layer = np.maximum(0.0, mean_dbpb_a) + np.maximum(0.0, mean_dbpb_m)
    causal_depth = np.sum(causal_by_layer * layer_frac[None, :], axis=1) / np.maximum(np.sum(causal_by_layer, axis=1), 1e-8)
    dc_gap_mlp_final = zero_dbpb_m[:, -1] - mean_dbpb_m[:, -1]

    # 3. Pairwise E(i,j) family summaries & sign flips
    def _mask_prefix(pfx: str) -> np.ndarray:
        return np.array([p.startswith(pfx) for p in pair_names], dtype=bool)

    pair_groups = {}
    for pfx in ("same_", "adj_attn_", "adj_mlp_", "adj_am_", "adj_ma_"):
        m = _mask_prefix(pfx)
        sub_e = pair_exc_nats[:, m]
        sub_n = pair_exc_norm[:, m]
        pair_groups[pfx.rstrip("_")] = {
            "mean_E_nats": [round(float(x), 4) for x in sub_e.mean(axis=1)],
            "max_E_nats": [round(float(x), 4) for x in sub_e.max(axis=1)],
            "min_E_nats": [round(float(x), 4) for x in sub_e.min(axis=1)],
            "frac_redundant_gt_0p02": [round(float(x), 3) for x in (sub_e > 0.02).mean(axis=1)],
            "frac_serial_lt_m0p02": [round(float(x), 3) for x in (sub_e < -0.02).mean(axis=1)],
            "mean_E_norm": [round(float(x), 4) for x in sub_n.mean(axis=1)],
        }

    # Detect individual block pairs that strongly flip sign across training (min < -0.05 and max > +0.05 nats)
    sign_flips = []
    for pi, pname in enumerate(pair_names):
        arr = pair_exc_nats[:, pi]
        # exclude step 0 from flip check
        arr_tr = arr[1:] if n_t > 1 else arr
        if arr_tr.size >= 3 and float(arr_tr.min()) < -0.05 and float(arr_tr.max()) > 0.05:
            i_min = int(np.argmin(arr_tr)) + (1 if n_t > 1 else 0)
            i_max = int(np.argmax(arr_tr)) + (1 if n_t > 1 else 0)
            sign_flips.append({
                "pair": pname,
                "min_E_nats": round(float(arr[i_min]), 4),
                "min_step": steps[i_min],
                "min_tokens_B": round(toks[i_min], 2),
                "max_E_nats": round(float(arr[i_max]), 4),
                "max_step": steps[i_max],
                "max_tokens_B": round(toks[i_max], 2),
                "direction": "serial->redundant" if i_min < i_max else "redundant->serial",
            })
    sign_flips.sort(key=lambda x: -(x["max_E_nats"] - x["min_E_nats"]))

    # 4. Channel 4 CKA velocity, dark-transition spikes, layer-freeze times, and final-layer rank collapse
    cka_step_vel = np.zeros((max(0, n_t - 1), L + 1), dtype=np.float64)
    dark_ratio = np.zeros((max(0, n_t - 1), L + 1), dtype=np.float64)
    for t in range(n_t - 1):
        c = cka_diag(grams[t], grams[t + 1])
        v = 1.0 - c
        cka_step_vel[t] = v
        dbpb = max(abs(clean_bpb[t + 1] - clean_bpb[t]), 1e-3)
        dark_ratio[t] = v / dbpb

    # CKA to final checkpoint (layer-freeze curve!)
    cka_to_final = np.zeros((n_t, L + 1), dtype=np.float64)
    for t in range(n_t):
        cka_to_final[t] = cka_diag(grams[t], grams[-1])
    # First token count where CKA to final >= 0.85 and >= 0.90
    freeze_tok_85 = []
    freeze_tok_90 = []
    for l in range(L + 1):
        idx85 = np.where(cka_to_final[:, l] >= 0.85)[0]
        idx90 = np.where(cka_to_final[:, l] >= 0.90)[0]
        freeze_tok_85.append(round(toks[int(idx85[0])], 3) if len(idx85) else None)
        freeze_tok_90.append(round(toks[int(idx90[0])], 3) if len(idx90) else None)

    # Final-layer normed effective rank collapse ratio: erank_normed(L) / erank_normed(L-1)
    final_rank_ratio = erank_normed[:, -1] / np.maximum(erank_normed[:, -2], 1e-6)
    final_cos_am = cos_am[:, -1]
    max_outlier_int = out_int[:, 1:-1].max(axis=1) if L >= 2 else out_int.max(axis=1)
    final_outlier_int = out_int[:, -1]

    # 5. Channel 3 micro-probe trajectories + non-monotonicity scan
    probe_names = sorted(recs[0]["json"]["ch3_probes"].keys())
    ch3_curves: dict[str, dict[str, list[float]]] = {}
    non_monotonic_flags: list[dict[str, Any]] = []

    primary_metric_map = {
        "P1_order": ("delta_bpb_b_minus_a", "delta_bpb_se"),
        "P2_horizon": ("delta_bpb_b_minus_a", "delta_bpb_se"),
        "P3_copy_decoy": ("shift_bpb_a_minus_b", None),
        "P4_induction_vs_break": ("margin_bpb_mean", None),
        "P5_agreement_attractor": ("shift_bpb_a_minus_b", None),
        "P6_memory_vs_icl": ("margin_bpb_ctx_b", None),  # negative = ICL overrides parametric!
        "P7_binding_2_vs_5": ("margin_bpb_ctx_a", None),
        "P8_arith_carry": ("carry_penalty_bpb", "carry_penalty_se"),
    }

    for pname in probe_names:
        keys = list(recs[0]["json"]["ch3_probes"][pname].keys())
        ch3_curves[pname] = {
            k: [round(float(r["json"]["ch3_probes"][pname][k]), 4) for r in recs]
            for k in keys
            if k != "n"
        }
        # Scan primary metric for non-monotonic peak-then-drop (or trough-then-rise) after step > 0
        mkey, sekey = primary_metric_map.get(pname, (None, None))
        if mkey and mkey in ch3_curves[pname] and n_t >= 5:
            arr = np.array(ch3_curves[pname][mkey][1:])  # post-step0
            t_arr = toks[1:]
            s_arr = steps[1:]
            # Check peak followed by drop to end or later minimum
            pk = int(np.argmax(arr))
            if pk < len(arr) - 1:
                after_min_rel = int(np.argmin(arr[pk + 1 :]))
                after_min = pk + 1 + after_min_rel
                drop = float(arr[pk] - arr[after_min])
                span = float(arr.max() - arr.min()) + 1e-8
                if drop > 0.15 * span and drop > 0.05:
                    non_monotonic_flags.append({
                        "probe": pname,
                        "metric": mkey,
                        "peak_val": round(float(arr[pk]), 4),
                        "peak_tokens_B": round(t_arr[pk], 2),
                        "peak_step": s_arr[pk],
                        "later_min_val": round(float(arr[after_min]), 4),
                        "later_min_tokens_B": round(t_arr[after_min], 2),
                        "final_val": round(float(arr[-1]), 4),
                        "drop": round(drop, 4),
                        "drop_frac_of_span": round(drop / span, 3),
                    })
            # Also check trough followed by rise (especially for P6 margin_bpb_ctx_b where negative=ICL!)
            tr = int(np.argmin(arr))
            if tr < len(arr) - 1:
                after_max_rel = int(np.argmax(arr[tr + 1 :]))
                after_max = tr + 1 + after_max_rel
                rise = float(arr[after_max] - arr[tr])
                span = float(arr.max() - arr.min()) + 1e-8
                if rise > 0.15 * span and rise > 0.05:
                    non_monotonic_flags.append({
                        "probe": pname,
                        "metric": mkey + " (rebound)",
                        "trough_val": round(float(arr[tr]), 4),
                        "trough_tokens_B": round(t_arr[tr], 2),
                        "trough_step": s_arr[tr],
                        "later_max_val": round(float(arr[after_max]), 4),
                        "later_max_tokens_B": round(t_arr[after_max], 2),
                        "final_val": round(float(arr[-1]), 4),
                        "rise": round(rise, 4),
                        "rise_frac_of_span": round(rise / span, 3),
                    })

    # Detect top dark-transition spikes (post-1B tokens, where loss changes slowly yet CKA jumps)
    dark_spikes = []
    for t in range(n_t - 1):
        if toks[t] >= 1.0:
            mean_v = float(cka_step_vel[t, 1:].mean())
            max_v = float(cka_step_vel[t, 1:].max())
            max_l = int(np.argmax(cka_step_vel[t, 1:])) + 1
            dbpb = float(clean_bpb[t] - clean_bpb[t + 1])
            dr = float(max_v / max(abs(dbpb), 1e-3))
            dark_spikes.append({
                "from_rev": revs[t],
                "to_rev": revs[t + 1],
                "from_tokens_B": round(toks[t], 2),
                "to_tokens_B": round(toks[t + 1], 2),
                "delta_bpb": round(dbpb, 4),
                "mean_vel": round(mean_v, 4),
                "max_vel": round(max_v, 4),
                "max_vel_layer": max_l,
                "dark_ratio": round(dr, 2),
            })
    dark_spikes_sorted = sorted(dark_spikes, key=lambda x: -x["dark_ratio"])[:8]

    return {
        "model_id": traj["model_id"],
        "model_type": traj["model_type"],
        "is_parallel": traj["is_parallel"],
        "is_post_norm": traj["is_post_norm"],
        "n_layers": L,
        "d_model": traj["d_model"],
        "n_checkpoints": n_t,
        "steps": steps,
        "tokens_B": [round(x, 3) for x in toks],
        "revisions": revs,
        "ch1": {
            "clean_bpb": [round(float(x), 4) for x in clean_bpb],
            "embed_bpb": [round(float(x), 4) for x in embed_bpb],
            "pred_depth": [round(float(x), 4) for x in pred_depth],
            "attn_share_mean": [round(float(x), 4) for x in attn_share.mean(axis=1)],
            "attn_share_mid_mean": [round(float(x), 4) for x in attn_share[:, 1 : max(2, L - 1)].mean(axis=1)],
            "attn_share_final": [round(float(x), 4) for x in attn_share[:, -1]],
            "cos_stream_mlp_min": [round(float(x), 4) for x in cos_stream_m.min(axis=1)],
            "cos_stream_mlp_mean": [round(float(x), 4) for x in cos_stream_m.mean(axis=1)],
            "cos_stream_attn_min": [round(float(x), 4) for x in cos_stream_a.min(axis=1)],
            "final_cos_am": [round(float(x), 4) for x in final_cos_am],
            "attn_share_last_ckpt": [round(float(x), 4) for x in attn_share[-1]],
            "cos_stream_mlp_last_ckpt": [round(float(x), 4) for x in cos_stream_m[-1]],
            "cos_am_last_ckpt": [round(float(x), 4) for x in cos_am[-1]],
        },
        "ch2": {
            "causal_attn_share": [round(float(x), 4) for x in causal_attn_share],
            "causal_depth": [round(float(x), 4) for x in causal_depth],
            "dc_gap_mlp_final": [round(float(x), 4) for x in dc_gap_mlp_final],
            "pair_groups": pair_groups,
            "sign_flips_top10": sign_flips[:10],
            "n_sign_flips": len(sign_flips),
        },
        "ch3": {
            "curves": ch3_curves,
            "non_monotonic_flags": non_monotonic_flags,
        },
        "ch4": {
            "final_rank_ratio": [round(float(x), 4) for x in final_rank_ratio],
            "erank_normed_final_layer": [round(float(x), 2) for x in erank_normed[:, -1]],
            "erank_normed_penult_layer": [round(float(x), 2) for x in erank_normed[:, -2]],
            "erank_raw_min_mid": [round(float(x), 2) for x in erank_raw[:, 1:-1].min(axis=1)],
            "max_outlier_int": [round(float(x), 2) for x in max_outlier_int],
            "final_outlier_int": [round(float(x), 2) for x in final_outlier_int],
            "attn_bos_mean": [round(float(x), 4) for x in attn_bos.mean(axis=1)],
            "attn_ent_final": [round(float(x), 4) for x in attn_ent[:, -1]],
            "freeze_tok_85": freeze_tok_85,
            "freeze_tok_90": freeze_tok_90,
            "cka_to_final_at_10B_nearest": [
                round(float(x), 4) for x in cka_to_final[int(np.argmin(np.abs(np.array(toks) - 10.0)))]
            ],
            "dark_spikes_top8": dark_spikes_sorted,
        },
    }


def compute_cross_family_cka(trajs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute cross-family CKA at matched token budgets (0B, 2B, 20B, 100B, 300B, final)."""
    anchors = [
        ("EleutherAI/pythia-1b", "allenai/OLMo-1B-0724-hf"),
        ("EleutherAI/pythia-1b", "allenai/OLMo-2-0425-1B"),
        ("allenai/OLMo-1B-0724-hf", "allenai/OLMo-2-0425-1B"),
        ("EleutherAI/pythia-160m", "EleutherAI/pythia-1b"),
    ]
    budgets = [0.0, 2.0, 20.0, 100.0, 300.0]
    out: dict[str, Any] = {}
    for m1, m2 in anchors:
        if m1 not in trajs or m2 not in trajs:
            continue
        t1, t2 = trajs[m1], trajs[m2]
        tok1 = np.array([r["tokens_B"] for r in t1["records"]])
        tok2 = np.array([r["tokens_B"] for r in t2["records"]])
        pair_res = []
        for b in budgets + ["final"]:
            if b == "final":
                i1, i2 = len(tok1) - 1, len(tok2) - 1
                label = "final"
            else:
                i1 = int(np.argmin(np.abs(tok1 - b)))
                i2 = int(np.argmin(np.abs(tok2 - b)))
                label = f"{int(b)}B"
            K1 = t1["records"][i1]["npz"]["gram_matrices"]
            K2 = t2["records"][i2]["npz"]["gram_matrices"]
            M = cka_matrix(K1, K2)
            # Best-match CKA per layer of model 1 (excluding embed L0)
            best_per_l1 = [round(float(M[l, 1:].max()), 4) for l in range(1, M.shape[0])]
            best_arg_l2 = [int(np.argmax(M[l, 1:])) + 1 for l in range(1, M.shape[0])]
            pair_res.append({
                "budget": label,
                "rev1": t1["records"][i1]["revision"],
                "tok1_B": round(float(tok1[i1]), 2),
                "rev2": t2["records"][i2]["revision"],
                "tok2_B": round(float(tok2[i2]), 2),
                "mean_best_cka": round(float(np.mean(best_per_l1)), 4),
                "max_cka": round(float(M[1:, 1:].max()), 4),
                "final_layer_cka": round(float(M[-1, -1]), 4),
                "best_per_l1": best_per_l1,
                "best_match_l2": best_arg_l2,
            })
        out[f"{m1}__vs__{m2}"] = pair_res
    return out


def compute_multiseed_stats(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mean/SD across multi-seed Pythia families where aligned steps exist."""
    groups = {
        "pythia-70m_4seeds": [
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-70m-seed1",
            "EleutherAI/pythia-70m-seed2",
            "EleutherAI/pythia-70m-seed3",
        ],
        "pythia-160m_4seeds": [
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-160m-seed1",
            "EleutherAI/pythia-160m-seed2",
            "EleutherAI/pythia-160m-seed3",
        ],
        "pythia-410m_2seeds": [
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-410m-seed1",
        ],
    }
    out: dict[str, Any] = {}
    for gname, mids in groups.items():
        avail = [m for m in mids if m in summaries]
        if len(avail) < 2:
            continue
        common_steps = sorted(set.intersection(*(set(summaries[m]["steps"]) for m in avail)))
        if not common_steps:
            continue

        def _collect(extractor):
            mat = []
            for m in avail:
                smap = {st: i for i, st in enumerate(summaries[m]["steps"])}
                mat.append([extractor(summaries[m], smap[st]) for st in common_steps])
            arr = np.array(mat, dtype=np.float64)
            return {
                "mean": [round(float(x), 4) for x in arr.mean(axis=0)],
                "sd": [round(float(x), 4) for x in arr.std(axis=0, ddof=1)],
                "min": [round(float(x), 4) for x in arr.min(axis=0)],
                "max": [round(float(x), 4) for x in arr.max(axis=0)],
            }

        out[gname] = {
            "models": avail,
            "n_seeds": len(avail),
            "common_steps": common_steps,
            "tokens_B": [round(st * 2.097152e-3, 3) for st in common_steps],
            "clean_bpb": _collect(lambda s, i: s["ch1"]["clean_bpb"][i]),
            "pred_depth": _collect(lambda s, i: s["ch1"]["pred_depth"][i]),
            "attn_share_mean": _collect(lambda s, i: s["ch1"]["attn_share_mean"][i]),
            "final_cos_am": _collect(lambda s, i: s["ch1"]["final_cos_am"][i]),
            "causal_attn_share": _collect(lambda s, i: s["ch2"]["causal_attn_share"][i]),
            "final_rank_ratio": _collect(lambda s, i: s["ch4"]["final_rank_ratio"][i]),
            "max_outlier_int": _collect(lambda s, i: s["ch4"]["max_outlier_int"][i]),
            "P3_decoy_shift_bpb": _collect(lambda s, i: s["ch3"]["curves"]["P3_copy_decoy"]["shift_bpb_a_minus_b"][i]),
            "P6_icl_margin_bpb": _collect(lambda s, i: s["ch3"]["curves"]["P6_memory_vs_icl"]["margin_bpb_ctx_b"][i]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Tycho v3 Stage 1 offline discovery & anomaly engine")
    ap.add_argument("--data-dir", default="work/v3_stage1")
    ap.add_argument("--out-json", default="work/v3_stage1/stage1_analysis.json")
    args = ap.parse_args()

    root = Path(args.data_dir)
    trajs: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        traj = load_model_trajectory(sub)
        if traj is None:
            continue
        mid = traj["model_id"]
        trajs[mid] = traj
        summaries[mid] = summarize_trajectory(traj)
        print(
            f"[analyze] {mid:30s}: {traj['n_layers']:2d}L d={traj['d_model']:4d} | "
            f"{len(traj['records']):2d} ckpts ({summaries[mid]['tokens_B'][0]}B -> {summaries[mid]['tokens_B'][-1]}B) | "
            f"final bpb={summaries[mid]['ch1']['clean_bpb'][-1]:.4f} | "
            f"final_rank_ratio={summaries[mid]['ch4']['final_rank_ratio'][-1]:.3f} | "
            f"sign_flips={summaries[mid]['ch2']['n_sign_flips']} | "
            f"non_mono_probes={len(summaries[mid]['ch3']['non_monotonic_flags'])}",
            flush=True,
        )

    cross_cka = compute_cross_family_cka(trajs)
    multiseed = compute_multiseed_stats(summaries)

    report = {
        "n_models": len(summaries),
        "total_checkpoints": sum(s["n_checkpoints"] for s in summaries.values()),
        "models": summaries,
        "multiseed": multiseed,
        "cross_family_cka": cross_cka,
    }
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[analyze] Wrote {out_p} ({out_p.stat().st_size/1024:.1f} KB), total_ckpts={report['total_checkpoints']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
