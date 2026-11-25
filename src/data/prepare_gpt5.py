import argparse
import gc
import hashlib
import json
import os
import random
import sys
from typing import Set

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.tokenizer.subword import SubwordTokenizer


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_normalized_texts(jsonl_path: str) -> Set[str]:
    texts = set()
    if not os.path.exists(jsonl_path):
        return texts
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            norm = " ".join(item.get("text", "").split())
            if norm:
                texts.add(norm)
    return texts


def stream_unique_split(
    split_name: str,
    target_count: int,
    exclude_texts: Set[str],
    seed: int,
):
    ds = load_dataset("roneneldan/TinyStories", split=split_name, streaming=True)
    iterator = iter(ds)

    collected = []
    seen = set(exclude_texts)
    skipped_empty = 0
    skipped_dup_or_excluded = 0

    while target_count < 0 or len(collected) < target_count:
        try:
            item = next(iterator)
        except StopIteration:
            break

        text = item.get("text", "")
        if not text or not text.strip():
            skipped_empty += 1
            continue

        norm = " ".join(text.split())
        if not norm:
            skipped_empty += 1
            continue

        if norm in seen:
            skipped_dup_or_excluded += 1
            continue

        seen.add(norm)
        collected.append({"text": norm})

    del iterator
    del ds
    gc.collect()

    rng = random.Random(seed)
    rng.shuffle(collected)
    return collected, skipped_empty, skipped_dup_or_excluded


def stream_to_bin(
    split_name: str,
    target_count: int,
    exclude_texts: Set[str],
    tokenizer: SubwordTokenizer,
    out_bin_path: str,
    chunk_docs: int = 5000,
):
    """Stream split directly into a uint16 .bin file in bounded chunks (low RAM)."""
    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories", split=split_name, streaming=True)
    iterator = iter(ds)

    unk_id = tokenizer.tokenizer.token_to_id("[UNK]")
    doc_count = 0
    token_count = 0
    unk_count = 0
    skipped_empty = 0
    skipped_excluded = 0
    buf = []

    with open(out_bin_path, "wb") as f:
        while target_count < 0 or doc_count < target_count:
            try:
                item = next(iterator)
            except StopIteration:
                break

            text = item.get("text", "")
            if not text or not text.strip():
                skipped_empty += 1
                continue

            norm = " ".join(text.split())
            if not norm:
                skipped_empty += 1
                continue

            if norm in exclude_texts:
                skipped_excluded += 1
                continue

            buf.append(norm)
            doc_count += 1

            if len(buf) >= chunk_docs:
                ids = tokenizer.encode(" ".join(buf))
                token_count += len(ids)
                if unk_id is not None:
                    unk_count += sum(1 for t in ids if t == unk_id)
                np.asarray(ids, dtype=np.uint16).tofile(f)
                buf.clear()
                if doc_count % 100000 == 0:
                    print(f"  [{split_name}] streamed {doc_count:,} docs ({token_count:,} tokens)...")

        if buf:
            ids = tokenizer.encode(" ".join(buf))
            token_count += len(ids)
            if unk_id is not None:
                unk_count += sum(1 for t in ids if t == unk_id)
            np.asarray(ids, dtype=np.uint16).tofile(f)
            buf.clear()

    del iterator
    del ds
    gc.collect()
    return {
        "document_count": doc_count,
        "token_count": token_count,
        "unk_token_id": unk_id,
        "unk_token_count": unk_count,
        "unk_token_rate": round((unk_count / token_count) if token_count > 0 else 0.0, 6),
        "skipped_empty": skipped_empty,
        "skipped_duplicate_or_excluded": skipped_excluded,
    }


def write_jsonl(path: str, records) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item) + "\n")
        f.flush()
        os.fsync(f.fileno())


def compute_token_stats(records, tokenizer: SubwordTokenizer):
    joined = " ".join(r["text"] for r in records)
    ids = tokenizer.encode(joined)
    unk_id = tokenizer.tokenizer.token_to_id("[UNK]")
    unk_count = sum(1 for tid in ids if tid == unk_id)
    total_tokens = len(ids)
    unk_rate = (unk_count / total_tokens) if total_tokens > 0 else 0.0
    return {
        "document_count": len(records),
        "char_count": len(joined),
        "token_count": total_tokens,
        "unk_token_id": unk_id,
        "unk_token_count": unk_count,
        "unk_token_rate": round(unk_rate, 6),
    }


def prepare_gpt5_data(
    num_train: int = 100000,
    num_val: int = 2000,
    seed: int = 42,
    data_dir: str = "data/gpt5",
    existing_val_path: str = "data/processed/val.jsonl",
    tokenizer_path: str = "data/processed/gpt3_tokenizer.json",
    write_bin: bool = False,
):
    print(f"Loading existing validation set from {existing_val_path} to enforce zero overlap...")
    existing_val_texts = load_normalized_texts(existing_val_path)
    print(f"Loaded {len(existing_val_texts)} existing validation documents for exclusion.")

    tokenizer = SubwordTokenizer()
    tokenizer.load(tokenizer_path)
    if write_bin and tokenizer.vocab_size > 65535:
        raise ValueError("uint16 .bin requires vocab_size <= 65535")

    if write_bin:
        train_bin = os.path.join(data_dir, "train.bin")
        val_bin = os.path.join(data_dir, "val.bin")
        print(f"Streaming train split ({'FULL' if num_train < 0 else num_train}) -> {train_bin}...")
        train_stats = stream_to_bin("train", num_train, existing_val_texts, tokenizer, train_bin)
        print(f"Streaming validation split ({'FULL' if num_val < 0 else num_val}) -> {val_bin}...")
        val_stats = stream_to_bin("validation", num_val, existing_val_texts, tokenizer, val_bin)
        train_entry = {"source_split": "train", "path": train_bin, "file_size_bytes": os.path.getsize(train_bin), **train_stats}
        val_entry = {"source_split": "validation", "path": val_bin, "file_size_bytes": os.path.getsize(val_bin), **val_stats}
    else:
        print(f"Streaming {num_train:,} unique TinyStories train documents...")
        train_records, train_empty, train_dup = stream_unique_split(
            split_name="train",
            target_count=num_train,
            exclude_texts=existing_val_texts,
            seed=seed,
        )
        train_texts = {r["text"] for r in train_records}
        val_exclude = existing_val_texts | train_texts

        print(f"Streaming {num_val:,} unique TinyStories validation documents...")
        val_records, val_empty, val_dup = stream_unique_split(
            split_name="validation",
            target_count=num_val,
            exclude_texts=val_exclude,
            seed=seed + 1,
        )

        train_path = os.path.join(data_dir, "train.jsonl")
        val_path = os.path.join(data_dir, "val.jsonl")
        write_jsonl(train_path, train_records)
        write_jsonl(val_path, val_records)

        train_stats = compute_token_stats(train_records, tokenizer)
        val_stats = compute_token_stats(val_records, tokenizer)
        train_entry = {
            "source_split": "train",
            "path": train_path,
            "file_size_bytes": os.path.getsize(train_path),
            "sha256": sha256_file(train_path),
            "skipped_empty": train_empty,
            "skipped_duplicate_or_excluded": train_dup,
            **train_stats,
        }
        val_entry = {
            "source_split": "validation",
            "path": val_path,
            "file_size_bytes": os.path.getsize(val_path),
            "sha256": sha256_file(val_path),
            "skipped_empty": val_empty,
            "skipped_duplicate_or_excluded": val_dup,
            **val_stats,
        }

    manifest = {
        "dataset": "roneneldan/TinyStories",
        "version": "0.0.0",
        "seed": seed,
        "format": "uint16_bin" if write_bin else "jsonl",
        "tokenizer_path": tokenizer_path,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "excluded_existing_val_path": existing_val_path,
        "excluded_existing_val_docs": len(existing_val_texts),
        "train": train_entry,
        "val": val_entry,
    }

    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared GPT-5 dataset in {data_dir} ({manifest['format']}):")
    print(f"  train: {train_entry['document_count']:,} docs, {train_entry['token_count']:,} tokens, UNK={train_entry['unk_token_count']:,} ({train_entry['unk_token_rate']:.4%})")
    print(f"  val:   {val_entry['document_count']:,} docs, {val_entry['token_count']:,} tokens, UNK={val_entry['unk_token_count']:,} ({val_entry['unk_token_rate']:.4%})")
    print(f"  manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset for GPT-5")
    parser.add_argument("--num_train", type=int, default=100000, help="Number of train docs (-1 for full split)")
    parser.add_argument("--num_val", type=int, default=2000, help="Number of val docs (-1 for full split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/gpt5")
    parser.add_argument("--tokenizer_path", type=str, default="data/processed/gpt3_tokenizer.json")
    parser.add_argument("--write_bin", action="store_true", help="Stream-tokenize directly to uint16 .bin files")
    args = parser.parse_args()
    prepare_gpt5_data(
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        write_bin=args.write_bin,
    )
