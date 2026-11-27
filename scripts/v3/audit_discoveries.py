"""Mechanistic audits of two Stage 1 discoveries:
  Audit A: Is adjacent-attention Hydra redundancy (E_norm = +2..+6) true functional compensation
           (holds in top-1 accuracy & KL & direct downstream Attn_{l+1} activation shift), or CE convexity?
  Audit B: Which exact tokens & hidden dimensions carry the transient 1,300x-3,270x interior massive
           activations in Pythia-1B and OLMo-1B-0724 before vs after their 12x-50x collapse?
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
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probes import build_core_passages  # noqa: E402
from scan_checkpoint import UnifiedAdapter, prepare_core_batch  # noqa: E402


@torch.no_grad()
def run_hydra_audit(model_id: str, revision: str, pairs_to_check: list[int], cache_root: Path, threads: int) -> dict:
    """Measure E(Attn_l, Attn_{l+1}) in CE, top-1 accuracy, KL, and direct Attn_{l+1} compensation delta."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tmp = cache_root / f"hydra_{os.getpid()}_{int(time.time()*1000)}"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(tmp)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, cache_dir=str(tmp))
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32, attn_implementation="eager", cache_dir=str(tmp))
        model.eval()
        ad = UnifiedAdapter(model)
        inp, _ = prepare_core_batch(tok, build_core_passages(8), 96, torch.device("cpu"))
        tgt = inp[:, 2:]

        # Clean pass capturing all Attn deltas
        L = ad.n_layers
        clean_da = [None] * L
        hs = []
        for l in range(L):
            def _mk(i):
                def _h(_m, _a, o):
                    clean_da[i] = (o[0] if isinstance(o, tuple) else o).detach().clone()
                return _h
            hs.append(ad.attn_delta_module(l).register_forward_hook(_mk(l)))
        try:
            lg_clean = model(input_ids=inp, use_cache=False).logits[:, 1:-1, :].float()
        finally:
            for h in hs:
                h.remove()

        p_clean = F.softmax(lg_clean, dim=-1)
        ce_clean = float(F.cross_entropy(lg_clean.reshape(-1, lg_clean.size(-1)), tgt.reshape(-1)).item())
        acc_clean = float((lg_clean.argmax(-1) == tgt).float().mean().item())
        means = {l: clean_da[l][:, 1:, :].mean(dim=(0, 1)) for l in range(L)}

        def _eval_ablate(layers_to_ablate: list[int], record_layer: int | None = None):
            rec_buf = []
            handles = []
            for l in layers_to_ablate:
                mu = means[l]
                def _mk_ab(m_vec):
                    def _h(_m, _a, o):
                        t = o[0] if isinstance(o, tuple) else o
                        r = t.clone()
                        r[:, 1:, :] = m_vec.to(dtype=t.dtype, device=t.device).view(1, 1, -1)
                        return (r,) + o[1:] if isinstance(o, tuple) else r
                    return _h
                handles.append(ad.attn_delta_module(l).register_forward_hook(_mk_ab(mu)))
            if record_layer is not None:
                handles.append(
                    ad.attn_delta_module(record_layer).register_forward_hook(
                        lambda _m, _a, o: rec_buf.append((o[0] if isinstance(o, tuple) else o).detach().clone())
                    )
                )
            try:
                lg = model(input_ids=inp, use_cache=False).logits[:, 1:-1, :].float()
            finally:
                for h in handles:
                    h.remove()
            ce = float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1)).item())
            acc = float((lg.argmax(-1) == tgt).float().mean().item())
            kl = float(F.kl_div(F.log_softmax(lg, dim=-1), p_clean, reduction="batchmean").item() / lg.size(1))
            return ce, acc, kl, (rec_buf[0] if rec_buf else None)

        rows = []
        for l in pairs_to_check:
            ce1, acc1, kl1, da_next_when_l_abl = _eval_ablate([l], record_layer=l + 1)
            ce2, acc2, kl2, _ = _eval_ablate([l + 1])
            ce12, acc12, kl12, _ = _eval_ablate([l, l + 1])
            d_ce1, d_ce2, d_ce12 = ce1 - ce_clean, ce2 - ce_clean, ce12 - ce_clean
            d_acc1, d_acc2, d_acc12 = acc_clean - acc1, acc_clean - acc2, acc_clean - acc12
            # Direct Hydra compensation test: when Attn_l is ablated, how does Attn_{l+1} respond?
            # Compare shift = (da'_{l+1} - da_{l+1}) with the removed signal (da_l - mu_l)!
            removed = (clean_da[l][:, 1:, :] - means[l].view(1, 1, -1)).float()
            shift_next = (da_next_when_l_abl[:, 1:, :] - clean_da[l + 1][:, 1:, :]).float()
            orig_next = clean_da[l + 1][:, 1:, :].float()
            # Positive cosine(shift_next, removed) => Attn_{l+1} ACTIVELY writes back the missing Attn_l direction!
            cos_comp = float(
                ((shift_next * removed).sum(-1) / (shift_next.norm(dim=-1) * removed.norm(dim=-1) + 1e-8)).mean().item()
            )
            rel_shift_norm = float((shift_next.norm(dim=-1) / (orig_next.norm(dim=-1) + 1e-8)).mean().item())
            # Also static similarity cos(da_l, da_{l+1}) (passive duplicate vs active compensation)
            cos_static = float(
                ((clean_da[l][:, 1:, :].float() * orig_next).sum(-1) / (clean_da[l][:, 1:, :].float().norm(dim=-1) * orig_next.norm(dim=-1) + 1e-8)).mean().item()
            )
            rows.append({
                "pair": f"Attn_L{l}_L{l+1}",
                "dCE_1": round(d_ce1, 4), "dCE_2": round(d_ce2, 4), "dCE_12": round(d_ce12, 4),
                "E_CE": round(d_ce12 - d_ce1 - d_ce2, 4),
                "E_CE_norm": round((d_ce12 - d_ce1 - d_ce2) / max(abs(d_ce1) + abs(d_ce2), 1e-3), 3),
                "dAcc_1": round(d_acc1, 4), "dAcc_2": round(d_acc2, 4), "dAcc_12": round(d_acc12, 4),
                "E_Acc": round(d_acc12 - d_acc1 - d_acc2, 4),
                "KL_1": round(kl1, 4), "KL_2": round(kl2, 4), "KL_12": round(kl12, 4),
                "E_KL": round(kl12 - kl1 - kl2, 4),
                "cos_active_compensation": round(cos_comp, 4),
                "rel_shift_next_norm": round(rel_shift_norm, 4),
                "cos_static_l_l1": round(cos_static, 4),
            })
        del model
        gc.collect()
        return {"model_id": model_id, "revision": revision, "ce_clean": round(ce_clean, 4), "acc_clean": round(acc_clean, 4), "pairs": rows}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@torch.no_grad()
def run_outlier_token_audit(model_id: str, revisions: list[str], cache_root: Path, threads: int) -> list[dict]:
    """Identify the exact tokens & hidden dimensions holding interior massive activations across checkpoints."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    out_rows = []
    for rev in revisions:
        tmp = cache_root / f"out_{os.getpid()}_{int(time.time()*1000)}"
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(tmp)
        try:
            tok = AutoTokenizer.from_pretrained(model_id, revision=rev, use_fast=True, cache_dir=str(tmp))
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.float32, attn_implementation="eager", cache_dir=str(tmp))
            model.eval()
            ad = UnifiedAdapter(model)
            inp, _ = prepare_core_batch(tok, build_core_passages(16), 96, torch.device("cpu"))
            L = ad.n_layers
            h_out = [None] * L
            hs = []
            for l in range(L):
                def _mk(i):
                    def _h(_m, _a, o):
                        h_out[i] = (o[0] if isinstance(o, tuple) else o).detach()
                    return _h
                hs.append(ad.layers[l].register_forward_hook(_mk(l)))
            try:
                model(input_ids=inp, use_cache=False)
            finally:
                for h in hs:
                    h.remove()

            # Find peak layer (excluding final layer) for interior outlier ratio
            best_l, best_ratio = 0, 0.0
            for l in range(L - 1):
                hi = h_out[l][:, 1:, :].float().abs()
                r = float(hi.max().item() / (hi.median().item() + 1e-8))
                if r > best_ratio:
                    best_ratio, best_l = r, l
            H = h_out[best_l].float()  # (B, T, d)
            med_int = float(H[:, 1:, :].abs().median().item())
            med_p0 = float(H[:, 0, :].abs().median().item())
            # Top 6 (b, t>=1, dim) entries
            flat_int = H[:, 1:, :].abs().reshape(-1)
            vals, idxs = torch.topk(flat_int, 6)
            B, Ti, D = H.size(0), H.size(1) - 1, H.size(2)
            top_int = []
            for v, idx in zip(vals.tolist(), idxs.tolist()):
                b = idx // (Ti * D)
                rem = idx % (Ti * D)
                t = (rem // D) + 1
                dim = rem % D
                tid = int(inp[b, t].item())
                tstr = tok.decode([tid])
                prev_str = tok.decode([int(inp[b, t - 1].item())])
                top_int.append({
                    "tok": tstr, "prev_tok": prev_str, "pos": t, "dim": dim,
                    "val": round(float(H[b, t, dim].item()), 1), "ratio": round(v / (med_int + 1e-8), 1),
                })
            # Also top pos0 dim
            v0, i0 = torch.topk(H[:, 0, :].abs().reshape(-1), 3)
            top_p0 = []
            for v, idx in zip(v0.tolist(), i0.tolist()):
                b = idx // D
                dim = idx % D
                top_p0.append({"pos": 0, "tok": tok.decode([int(inp[b, 0].item())]), "dim": dim, "val": round(float(H[b, 0, dim].item()), 1), "ratio": round(v / (med_p0 + 1e-8), 1)})
            rec = {
                "model_id": model_id, "revision": rev, "peak_layer": best_l + 1,
                "max_ratio_int": round(best_ratio, 1),
                "top_interior": top_int, "top_pos0": top_p0,
            }
            print(f"[outlier-audit] {model_id} @ {rev} L{best_l+1} maxR={best_ratio:.1f} | top_int={(top_int[0]['tok'], top_int[0]['pos'], 'd'+str(top_int[0]['dim']), top_int[0]['val'])} | top_p0={('d'+str(top_p0[0]['dim']), top_p0[0]['val'])}", flush=True)
            out_rows.append(rec)
            del model
            gc.collect()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="work/v3_stage1/mechanistic_audits.json")
    ap.add_argument("--threads", type=int, default=44)
    ap.add_argument("--cache-root", default="/tmp/hf_v3_audits")
    args = ap.parse_args()

    cache = Path(args.cache_root)
    hydra = [
        run_hydra_audit("EleutherAI/pythia-1b", "step143000", [1, 5, 6], cache, args.threads),
        run_hydra_audit("allenai/OLMo-1B-0724-hf", "step1454000-tokens3048B", [2, 3, 4], cache, args.threads),
        run_hydra_audit("allenai/OLMo-2-0425-1B", "stage1-step140000-tokens294B", [0, 4, 5], cache, args.threads),
    ]
    outliers = {
        "pythia-1b": run_outlier_token_audit("EleutherAI/pythia-1b", ["step80000", "step96000", "step112000", "step143000"], cache, args.threads),
        "olmo-1b": run_outlier_token_audit("allenai/OLMo-1B-0724-hf", ["step64000-tokens134B", "step500000-tokens1048B", "step750000-tokens1572B", "step1454000-tokens3048B"], cache, args.threads),
    }
    out = {"hydra_audit": hydra, "outlier_token_audit": outliers}
    Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
