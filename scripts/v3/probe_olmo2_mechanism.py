"""Targeted mechanistic probe for Post-RMSNorm Layer-0 Evacuation & Anti-Residual Erasure.

Designed to run fast on both 1B (allenai/OLMo-2-0425-1B) and 7B (allenai/OLMo-2-1124-7B)
checkpoints on CPU (s82), with immediate per-checkpoint HF cache eviction.

For each checkpoint, measures:
  1. Direct parameter RMS of Post-RMSNorm weights gamma_attn(l), gamma_mlp(l), and embed_tokens.
  2. Clean forward pass on 4 passages x 96 tokens -> clean_bpb, predD, wRMS_A(l), wRMS_M(l),
     cosA_stream(l), cosM_stream(l), maxOutI, maxOut0, and IC-v3.2 relative residual identity error.
  3. Targeted Layer-0/Layer-1 causal ablations (Attn0, MLP0, Attn0+MLP0, Attn1, MLP1) ->
     dA0, dM0, dA0M0, Hydra Excess Joint Loss E(Attn0, MLP0), dA1, dM1.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probes import build_core_passages  # noqa: E402
from scan_checkpoint import (  # noqa: E402
    UnifiedAdapter,
    eval_with_ablation,
    interior_ce_nats,
    nats_to_bpb,
    prepare_core_batch,
)


@torch.no_grad()
def probe_one_checkpoint(
    model_id: str,
    revision: str,
    *,
    n_seqs: int,
    seq_len: int,
    cache_root: Path,
    threads: int,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tmp_cache = cache_root / f"olmo2_{os.getpid()}_{int(time.time()*1000)}"
    tmp_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(tmp_cache)
    t0 = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, cache_dir=str(tmp_cache))
        # Load in float32 on CPU (s82 Xeon Broadwell-EX AVX2 is faster & exact in fp32)
        t_ld0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.float32, attn_implementation="eager", cache_dir=str(tmp_cache)
        )
        model.eval()
        t_load = time.perf_counter() - t_ld0

        ad = UnifiedAdapter(model)
        L = ad.n_layers

        # 1. Exact parameter-level Post-RMSNorm gamma RMS & embedding RMS
        gamma_attn = []
        gamma_mlp = []
        for l in range(L):
            ga = ad.attn_delta_module(l).weight.detach().float()
            gm = ad.mlp_delta_module(l).weight.detach().float()
            gamma_attn.append(round(float(torch.sqrt(ga.pow(2).mean()).item()), 5))
            gamma_mlp.append(round(float(torch.sqrt(gm.pow(2).mean()).item()), 5))
        emb_w = ad.embed_in.weight.detach().float()
        embed_rms = float(torch.sqrt(emb_w.pow(2).mean()).item())

        # 2. Clean forward pass capturing h_in, h_out, delta_attn, delta_mlp
        passages = build_core_passages(n_seqs)
        inp, pred_bytes = prepare_core_batch(tok, passages, seq_len, torch.device("cpu"))
        n_pred = inp.size(0) * (inp.size(1) - 2)

        h_in = [None] * L
        h_out = [None] * L
        da = [None] * L
        dm = [None] * L
        handles = []
        for l in range(L):
            def _mk_l(i):
                def _h(_m, a, o):
                    h_in[i] = a[0].detach()
                    h_out[i] = (o[0] if isinstance(o, tuple) else o).detach()
                return _h

            def _mk_d(buf, i):
                def _h(_m, _a, o):
                    buf[i] = (o[0] if isinstance(o, tuple) else o).detach()
                return _h

            handles.append(ad.layers[l].register_forward_hook(_mk_l(l)))
            handles.append(ad.attn_delta_module(l).register_forward_hook(_mk_d(da, l)))
            handles.append(ad.mlp_delta_module(l).register_forward_hook(_mk_d(dm, l)))

        try:
            logits = model(input_ids=inp, use_cache=False, output_attentions=False).logits
        finally:
            for h in handles:
                h.remove()

        clean_nats = interior_ce_nats(logits, inp)
        clean_bpb = nats_to_bpb(clean_nats, n_pred, pred_bytes)

        # Metrics + IC-v3.2
        w_a, w_m, cos_a, cos_m, cos_am, rel_errs = [], [], [], [], [], []
        mean_deltas = {}
        lens_bpb = []

        def _lens(h):
            lg = ad.unembed(ad.final_norm(h[:, 1:-1, :])).float()
            ce = float(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), inp[:, 2:].reshape(-1)).item())
            return nats_to_bpb(ce, n_pred, pred_bytes)

        lens_bpb.append(_lens(h_in[0]))
        out_int_max = 0.0
        out_pos0_max = 0.0
        for l in range(L):
            recon = h_in[l].float() + da[l].float() + dm[l].float()
            ae = float((h_out[l].float() - recon).abs().max().item())
            sc = max(
                float(h_in[l].float().abs().max().item()),
                float(da[l].float().abs().max().item()),
                float(dm[l].float().abs().max().item()),
                float(h_out[l].float().abs().max().item()),
            ) + 1e-8
            rel_errs.append(ae / sc)

            mean_deltas[(l, "attn")] = da[l][:, 1:, :].mean(dim=(0, 1)).detach()
            mean_deltas[(l, "mlp")] = dm[l][:, 1:, :].mean(dim=(0, 1)).detach()

            dai = da[l][:, 1:, :].float()
            dmi = dm[l][:, 1:, :].float()
            hai = h_in[l][:, 1:, :].float()
            hmi = (h_in[l] + da[l])[:, 1:, :].float()

            w_a.append(round(float(torch.sqrt(dai.pow(2).mean()).item()), 4))
            w_m.append(round(float(torch.sqrt(dmi.pow(2).mean()).item()), 4))
            na, nm = dai.norm(dim=-1), dmi.norm(dim=-1)
            nha, nhm = hai.norm(dim=-1), hmi.norm(dim=-1)
            cos_a.append(round(float(((dai * hai).sum(-1) / (na * nha + 1e-8)).mean().item()), 4))
            cos_m.append(round(float(((dmi * hmi).sum(-1) / (nm * nhm + 1e-8)).mean().item()), 4))
            cos_am.append(round(float(((dai * dmi).sum(-1) / (na * nm + 1e-8)).mean().item()), 4))

            if l < L - 1:
                hi = h_out[l][:, 1:, :].float().abs()
                h0 = h_out[l][:, 0, :].float().abs()
                out_int_max = max(out_int_max, float(hi.max().item() / (hi.median().item() + 1e-8)))
                out_pos0_max = max(out_pos0_max, float(h0.max().item() / (h0.median().item() + 1e-8)))
            lens_bpb.append(_lens(h_out[l]))

        lens_arr = np.array(lens_bpb)
        gains = np.maximum(0.0, lens_arr[:-1] - lens_arr[1:])
        pred_depth = float(np.sum(gains * (np.arange(1, L + 1) / L)) / max(float(gains.sum()), 1e-8))

        # 3. Targeted L0/L1 causal ablations (just 5 forward passes!)
        dA0 = eval_with_ablation(ad, inp, [(0, "attn")], "mean", mean_deltas) - clean_nats
        dM0 = eval_with_ablation(ad, inp, [(0, "mlp")], "mean", mean_deltas) - clean_nats
        dA0M0 = eval_with_ablation(ad, inp, [(0, "attn"), (0, "mlp")], "mean", mean_deltas) - clean_nats
        dA1 = eval_with_ablation(ad, inp, [(1, "attn")], "mean", mean_deltas) - clean_nats
        dM1 = eval_with_ablation(ad, inp, [(1, "mlp")], "mean", mean_deltas) - clean_nats
        E_sL0 = dA0M0 - dA0 - dM0

        # Also check mid/deep causal spot check: L_{L//2} and L_{L-1} MLP
        dM_mid = eval_with_ablation(ad, inp, [(L // 2, "mlp")], "mean", mean_deltas) - clean_nats
        dM_last = eval_with_ablation(ad, inp, [(L - 1, "mlp")], "mean", mean_deltas) - clean_nats

        wall = time.perf_counter() - t0
        rec = {
            "model_id": model_id,
            "revision": revision,
            "n_layers": L,
            "d_model": ad.d_model,
            "wall_s": round(wall, 1),
            "load_s": round(t_load, 1),
            "ic_v3_2_max_rel_err": float(max(rel_errs)),
            "clean_bpb": round(clean_bpb, 4),
            "pred_depth": round(pred_depth, 4),
            "embed_rms": round(embed_rms, 5),
            "gamma_attn": gamma_attn,
            "gamma_mlp": gamma_mlp,
            "wRMS_attn": w_a,
            "wRMS_mlp": w_m,
            "cosA_stream": cos_a,
            "cosM_stream": cos_m,
            "cos_am": cos_am,
            "min_cosA_stream": round(min(cos_a), 4),
            "max_outlier_int": round(out_int_max, 2),
            "max_outlier_pos0": round(out_pos0_max, 2),
            "causal_nats": {
                "dA0": round(dA0, 4),
                "dM0": round(dM0, 4),
                "dA0M0": round(dA0M0, 4),
                "E_same_L0": round(E_sL0, 4),
                "dA1": round(dA1, 4),
                "dM1": round(dM1, 4),
                "dM_mid": round(dM_mid, 4),
                "dM_last": round(dM_last, 4),
            },
        }
        print(
            f"[olmo2-probe] {model_id} @ {revision} ({wall:.1f}s) | bpb={clean_bpb:.4f} predD={pred_depth:.3f} "
            f"| embRMS={embed_rms:.4f} gA0={gamma_attn[0]:.3f} gM0={gamma_mlp[0]:.3f} gA_last={gamma_attn[-1]:.3f} "
            f"| minCosA={min(cos_a):+.3f} | dA0={dA0:.2f} dM0={dM0:.2f} dA0M0={dA0M0:.2f} E_sL0={E_sL0:+.2f}",
            flush=True,
        )
        del model
        gc.collect()
        return rec
    finally:
        shutil.rmtree(tmp_cache, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revisions", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--threads", type=int, default=44)
    ap.add_argument("--cache-root", default="/tmp/hf_v3_olmo2_probe")
    args = ap.parse_args()

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_p.exists():
        try:
            for r in json.loads(out_p.read_text()):
                existing[r["revision"]] = r
        except Exception:
            pass

    results = []
    for rev in args.revisions:
        if rev in existing:
            print(f"[olmo2-probe] SKIP existing {args.model} @ {rev}", flush=True)
            results.append(existing[rev])
            continue
        rec = probe_one_checkpoint(
            args.model,
            rev,
            n_seqs=args.n_seqs,
            seq_len=args.seq_len,
            cache_root=Path(args.cache_root),
            threads=args.threads,
        )
        results.append(rec)
        # Save incrementally after every checkpoint
        out_p.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
