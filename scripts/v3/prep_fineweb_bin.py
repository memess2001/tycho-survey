"""Fast batched packer: stream HuggingFaceFW/fineweb-edu (sample-10BT) -> uint16 .bin shards.

Produces:
  - <out_dir>/val.bin          (default 2M tokens, ~4 MB, strictly disjoint first documents)
  - <out_dir>/train.bin        (default 1.05B tokens, ~2.1 GB, subsequent documents)
  - <out_dir>/dataset_meta.json (exact bytes_per_token, token counts, tokenizer)

Uses HuggingFace fast Rust GPT-2 tokenizer in batches (saturates Colab G4's 48 vCPUs or s82)
so 1.05B tokens pack in ~2-4 minutes with minimal RAM (< 2 GB peak).
Also supports --synthetic for instant offline smoke tests.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np


def pack_synthetic(out_dir: Path, train_tokens: int, val_tokens: int, vocab_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    core_v = min(256, vocab_size - 2)
    logits = rng.normal(0.0, 2.2, size=(core_v, core_v))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    cdf = np.cumsum(probs / probs.sum(axis=1, keepdims=True), axis=1)

    def _gen(n: int) -> np.ndarray:
        arr = np.empty(n, dtype=np.uint16)
        cur = int(rng.integers(0, core_v))
        arr[0] = vocab_size - 1
        for i in range(1, n):
            u = float(rng.random())
            nxt = int(np.searchsorted(cdf[cur], u))
            nxt = min(nxt, core_v - 1)
            if i >= 10 and (i % 12 == 0):
                nxt = int(arr[i - 8])
            arr[i] = nxt
            cur = nxt
        return arr

    val_arr = _gen(val_tokens)
    trn_arr = _gen(train_tokens)
    val_arr.tofile(out_dir / "val.bin")
    trn_arr.tofile(out_dir / "train.bin")
    meta = {
        "source": "synthetic_markov_copy",
        "tokenizer": f"synthetic_v{vocab_size}",
        "vocab_size": vocab_size,
        "val_tokens": int(val_tokens),
        "train_tokens": int(train_tokens),
        "bytes_per_token": 1.0,
        "dtype": "uint16",
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[prep] Synthetic packed: val={val_tokens:,} train={train_tokens:,} -> {out_dir}", flush=True)


def pack_fineweb_edu(
    out_dir: Path,
    *,
    train_tokens: int,
    val_tokens: int,
    batch_docs: int,
    dataset_name: str,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tok.model_max_length = 10_000_000
    eos = int(tok.eos_token_id)
    assert tok.vocab_size <= 65535, "uint16 packing requires vocab_size <= 65535"

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=dataset_name, split="train", streaming=True)
    it = iter(ds)

    val_path = out_dir / "val.bin"
    trn_path = out_dir / "train.bin"
    val_tmp = out_dir / "val.bin.tmp"
    trn_tmp = out_dir / "train.bin.tmp"

    t0 = time.perf_counter()
    total_target = val_tokens + train_tokens
    written_val = 0
    written_trn = 0
    bytes_val, toks_val = 0, 0
    bytes_trn, toks_trn = 0, 0
    docs_seen = 0
    last_log = t0

    with val_tmp.open("wb") as fv, trn_tmp.open("wb") as ft:
        while written_val < val_tokens or written_trn < train_tokens:
            texts: List[str] = []
            for _ in range(batch_docs):
                try:
                    texts.append(next(it)["text"])
                except StopIteration:
                    break
            if not texts:
                raise RuntimeError("FineWeb-Edu stream exhausted before reaching target tokens.")

            enc = tok(texts, add_special_tokens=False)["input_ids"]
            docs_seen += len(texts)

            buf: List[int] = []
            batch_bytes = 0
            for txt, ids in zip(texts, enc):
                batch_bytes += len(txt.encode("utf-8"))
                buf.append(eos)
                buf.extend(ids)
            arr = np.asarray(buf, dtype=np.uint16)
            n_arr = len(arr)

            if written_val < val_tokens:
                need_v = val_tokens - written_val
                take_v = min(need_v, n_arr)
                arr[:take_v].tofile(fv)
                written_val += take_v
                bytes_val += int(batch_bytes * (take_v / max(1, n_arr)))
                toks_val += take_v
                rem = arr[take_v:]
                rem_bytes = batch_bytes - int(batch_bytes * (take_v / max(1, n_arr)))
            else:
                rem = arr
                rem_bytes = batch_bytes

            if len(rem) > 0 and written_trn < train_tokens:
                need_t = train_tokens - written_trn
                take_t = min(need_t, len(rem))
                rem[:take_t].tofile(ft)
                written_trn += take_t
                bytes_trn += int(rem_bytes * (take_t / max(1, len(rem))))
                toks_trn += take_t

            now = time.perf_counter()
            if now - last_log >= 10.0 or (written_val >= val_tokens and written_trn >= train_tokens):
                done = written_val + written_trn
                rate = done / max(1e-3, now - t0)
                eta = (total_target - done) / max(1.0, rate)
                print(
                    f"[prep] {done/1e6:7.1f}M / {total_target/1e6:7.1f}M tok "
                    f"({100*done/total_target:5.1f}%) | docs={docs_seen:,} "
                    f"| rate={rate/1e6:.2f}M tok/s | ETA={eta:.0f}s",
                    flush=True,
                )
                last_log = now

    os.replace(val_tmp, val_path)
    os.replace(trn_tmp, trn_path)

    bpt_val = bytes_val / max(1, toks_val)
    bpt_trn = bytes_trn / max(1, toks_trn)
    meta = {
        "source": f"HuggingFaceFW/fineweb-edu:{dataset_name}",
        "tokenizer": "gpt2",
        "vocab_size": int(tok.vocab_size),
        "eos_token_id": eos,
        "val_tokens": int(written_val),
        "train_tokens": int(written_trn),
        "docs_seen": int(docs_seen),
        "bytes_per_token_val": round(bpt_val, 5),
        "bytes_per_token_train": round(bpt_trn, 5),
        "bytes_per_token": round(bpt_val, 5),
        "dtype": "uint16",
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[prep] DONE in {meta['wall_s']}s | val={written_val:,} ({val_path.stat().st_size/1e6:.1f} MB) "
        f"| train={written_trn:,} ({trn_path.stat().st_size/1e9:.2f} GB) | bpt={bpt_val:.4f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-tokens", type=int, default=1_050_000_000, help="Default 1.05B tokens (~2.1 GB uint16)")
    ap.add_argument("--val-tokens", type=int, default=2_000_000, help="Default 2M tokens (~4 MB uint16)")
    ap.add_argument("--batch-docs", type=int, default=512)
    ap.add_argument("--dataset-name", default="sample-10BT")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "train.bin").exists() and (args.out_dir / "val.bin").exists() and (args.out_dir / "dataset_meta.json").exists():
        meta = json.loads((args.out_dir / "dataset_meta.json").read_text())
        if meta.get("train_tokens", 0) >= args.train_tokens and meta.get("val_tokens", 0) >= args.val_tokens:
            print(f"[prep] SKIP existing complete shard in {args.out_dir} ({meta['train_tokens']:,} train tok)", flush=True)
            return 0

    if args.synthetic:
        pack_synthetic(args.out_dir, args.train_tokens, args.val_tokens, args.vocab_size, args.seed)
    else:
        pack_fineweb_edu(
            args.out_dir,
            train_tokens=args.train_tokens,
            val_tokens=args.val_tokens,
            batch_docs=args.batch_docs,
            dataset_name=args.dataset_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
