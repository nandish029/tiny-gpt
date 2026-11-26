import argparse
import gc
import json
import os
import sys

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.tokenizer.subword import SubwordTokenizer


def stream_fineweb_edu(
    data_dir: str = "data/fineweb_edu",
    tokenizer_path: str = "data/fineweb_edu/fineweb_bpe_8192.json",
    vocab_size: int = 8192,
    tokenizer_docs: int = 40000,
    val_docs: int = 2000,
    target_train_tokens: int = 200_000_000,
    chunk_docs: int = 2000,
):
    os.makedirs(data_dir, exist_ok=True)
    ds_name = "HuggingFaceFW/fineweb-edu"
    ds_config = "sample-10BT"

    # 1. Train Byte-Level BPE tokenizer if not already present
    if not os.path.isfile(tokenizer_path):
        print(f"Training Byte-Level BPE (vocab={vocab_size}) on {tokenizer_docs:,} FineWeb-Edu docs...")
        ds_tok = load_dataset(ds_name, name=ds_config, split="train", streaming=True)
        tok_texts = []
        for item in ds_tok:
            t = item.get("text", "").strip()
            if t:
                tok_texts.append(t)
                if len(tok_texts) >= tokenizer_docs:
                    break
        tok = SubwordTokenizer(byte_level=True)
        tok.build_vocab(tok_texts, vocab_size=vocab_size)
        tok.save(tokenizer_path)
        del tok_texts, ds_tok
        gc.collect()
        print(f"Saved Byte-Level BPE tokenizer ({tok.vocab_size} tokens) -> {tokenizer_path}")
    else:
        tok = SubwordTokenizer(byte_level=True)
        tok.load(tokenizer_path)
        print(f"Loaded existing tokenizer ({tok.vocab_size} tokens) from {tokenizer_path}")

    if tok.vocab_size > 65535:
        raise ValueError("uint16 .bin requires vocab_size <= 65535")

    eot_id = tok.tokenizer.token_to_id("<|endoftext|>")
    eos_suffix = [eot_id] if eot_id is not None else []

    # 2. Stream validation docs first (strictly held out), then train docs
    ds = load_dataset(ds_name, name=ds_config, split="train", streaming=True)
    it = iter(ds)

    val_bin = os.path.join(data_dir, "val.bin")
    val_count = 0
    val_tokens = 0
    val_buf = []
    print(f"Streaming {val_docs:,} held-out validation docs -> {val_bin}...")
    with open(val_bin, "wb") as fv:
        while val_count < val_docs:
            try:
                item = next(it)
            except StopIteration:
                break
            t = item.get("text", "").strip()
            if not t:
                continue
            ids = tok.encode(t) + eos_suffix
            val_buf.extend(ids)
            val_count += 1
            if len(val_buf) >= 200_000:
                np.asarray(val_buf, dtype=np.uint16).tofile(fv)
                val_tokens += len(val_buf)
                val_buf.clear()
        if val_buf:
            np.asarray(val_buf, dtype=np.uint16).tofile(fv)
            val_tokens += len(val_buf)
            val_buf.clear()

    train_bin = os.path.join(data_dir, "train.bin")
    train_count = 0
    train_tokens = 0
    train_buf = []
    print(f"Streaming up to {target_train_tokens:,} train tokens -> {train_bin}...")
    with open(train_bin, "wb") as ft:
        while train_tokens + len(train_buf) < target_train_tokens:
            try:
                item = next(it)
            except StopIteration:
                break
            t = item.get("text", "").strip()
            if not t:
                continue
            ids = tok.encode(t) + eos_suffix
            train_buf.extend(ids)
            train_count += 1
            if len(train_buf) >= 500_000:
                np.asarray(train_buf, dtype=np.uint16).tofile(ft)
                train_tokens += len(train_buf)
                train_buf.clear()
                if train_count % 20000 == 0:
                    print(f"  [FineWeb-Edu] {train_count:,} docs ({train_tokens:,} tokens)...")
        if train_buf:
            np.asarray(train_buf, dtype=np.uint16).tofile(ft)
            train_tokens += len(train_buf)
            train_buf.clear()

    del it, ds
    gc.collect()

    manifest = {
        "dataset": f"{ds_name} ({ds_config})",
        "format": "uint16_bin",
        "tokenizer_path": tokenizer_path,
        "tokenizer_vocab_size": tok.vocab_size,
        "byte_level": True,
        "val": {
            "path": val_bin,
            "document_count": val_count,
            "token_count": val_tokens,
            "file_size_bytes": os.path.getsize(val_bin),
        },
        "train": {
            "path": train_bin,
            "document_count": train_count,
            "token_count": train_tokens,
            "file_size_bytes": os.path.getsize(train_bin),
        },
    }
    manifest_path = os.path.join(data_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared FineWeb-Edu in {data_dir}:")
    print(f"  val:   {val_count:,} docs, {val_tokens:,} tokens")
    print(f"  train: {train_count:,} docs, {train_tokens:,} tokens")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu subset for GPT-6")
    parser.add_argument("--data_dir", type=str, default="data/fineweb_edu")
    parser.add_argument("--tokenizer_path", type=str, default="data/fineweb_edu/fineweb_bpe_8192.json")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--tokenizer_docs", type=int, default=40000)
    parser.add_argument("--val_docs", type=int, default=2000)
    parser.add_argument("--target_train_tokens", type=int, default=200000000)
    args = parser.parse_args()
    stream_fineweb_edu(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
        tokenizer_docs=args.tokenizer_docs,
        val_docs=args.val_docs,
        target_train_tokens=args.target_train_tokens,
    )

