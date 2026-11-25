# GPT-5 Final Experiment Report — Data Scaling on TinyStories

**Experiment Name:** `gpt5`  
**Branch:** `feature/gpt4-architecture-experiments` (local uncommitted work)  
**Execution Device:** CPU (`Windows-11`, 12 CPU cores, no CUDA)  
**Execution Date:** 2026-11-25  
**Status:** COMPLETED & VERIFIED  

---

## 1. Objective

GPT-5 is the final model in the Tiny-GPT project. Having selected the 6-layer architecture from GPT-4 E1-C (`111,616` trainable parameters) and keeping the GPT-3/GPT-4 subword BPE tokenizer (`vocab_size = 512`) unchanged, GPT-5 isolates the effect of **training data scale and token exposure** on TinyStories.

---

## 2. Frozen Architecture & Tokenizer

| Component | Setting |
| :--- | :--- |
| **Model Class** | `src.model.gpt.GPT` |
| **Transformer Layers (`num_layers`)** | 6 |
| **Embedding Dimension (`embedding_dim`)** | 32 |
| **Attention Heads (`num_heads`)** | 4 ($d_k = 8$) |
| **Feed-Forward Dim (`feed_forward_dim`)** | 128 |
| **Context Length (`max_context_length`)** | 64 |
| **Vocabulary Size (`vocab_size`)** | 512 |
| **Trainable Parameters** | 111,616 |
| **Tokenizer** | `data/processed/gpt3_tokenizer.json` (`SubwordTokenizer`, BPE, `Whitespace` pre-tokenizer, `byte_fallback=false`) |

---

## 3. Dataset Selection & Preprocessing

- **Source:** `roneneldan/TinyStories` (Hugging Face `datasets` streaming, version `0.0.0`).
- **Full Split Context:** The full TinyStories `train` split contains `2,119,719` examples (~1.87 GB). Because `LanguageDataset` loads and tokenizes the entire corpus into CPU RAM at startup, a bounded deterministic subset of **100,000** training documents (~22.2× the existing 4,500-document training set) and **2,000** validation documents was prepared via [`src/data/prepare_gpt5.py`](../../../src/data/prepare_gpt5.py).
- **Preprocessing Rules:**
  1. Skip empty or whitespace-only records (`17` skipped in train stream).
  2. Normalize whitespace with `' '.join(text.split())`.
  3. Deduplicate exact normalized texts (`1,500` skipped as duplicates/excluded in train stream).
  4. Exclude all `500` documents present in [`data/processed/val.jsonl`](../../../data/processed/val.jsonl) to guarantee zero overlap with the shared GPT-4 validation set.
  5. Shuffle deterministically (`seed=42` for train, `seed=43` for val).
- **Prepared Data Manifest ([`data/gpt5/dataset_manifest.json`](../../../data/gpt5/dataset_manifest.json)):**
  - `data/gpt5/train.jsonl`: `100,000` docs, `90,168,544` bytes, **`30,482,584` tokens**, `120` `[UNK]` tokens (`0.0004%`).
  - `data/gpt5/val.jsonl`: `2,000` docs, `1,623,662` bytes, **`542,443` tokens**, `3` `[UNK]` tokens (`0.0006%`).
  - Zero overlap verified across `data/gpt5/train.jsonl`, `data/gpt5/val.jsonl`, and `data/processed/val.jsonl`.

---

## 4. Training Configuration & Compute Budget

- **Config:** [`experiments/gpt5/configs/gpt5.yaml`](../configs/gpt5.yaml)
- **Optimizer:** AdamW, `learning_rate = 3.0e-4`, `batch_size = 16`, `context_length = 64` (`1,024` tokens/step)
- **Optimizer Steps:** `10,000` (2× GPT-4 E1-C)
- **Total Token Exposure:** `10,240,000` token positions (~33.6% of one pass over the 30.48M-token training corpus)
- **Seed:** `42` (`val_seed = 10042`)
- **Duration & Throughput:** `612.84 s` (~10.2 minutes) on CPU (`61.28 ms/step`)
- **Training Metrics ([`experiments/gpt5/results/gpt5_results.json`](gpt5_results.json)):**
  - Final Train Loss: `3.6905`
  - Final Val Loss (`data/gpt5/val.jsonl`): `3.5478`
  - Minimum Val Loss (`data/gpt5/val.jsonl`): `3.4154`

---

## 5. Fair GPT-4 vs GPT-5 Evaluation

Both checkpoints were evaluated using `ExperimentRunner.evaluate()` (`batch_count = 50`, `batch_size = 16`, `context_length = 64`, **`51,200` sampled token positions**, `eval_seed = 10042`, same `SubwordTokenizer`):

| Evaluation Split (Sampled, 51,200 tokens) | Model | Train Docs | Steps | Token Exposure | Eval Loss | Perplexity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Shared TinyStories Val** (`data/processed/val.jsonl`) | GPT-4 (`e1_c`) | 4,500 | 5,000 | 5,120,000 | 3.7625 | 43.0553 |
| **Shared TinyStories Val** (`data/processed/val.jsonl`) | **GPT-5** | **100,000** | **10,000** | **10,240,000** | **3.5531** | **34.9209** |
| **GPT-5 Held-Out Val** (`data/gpt5/val.jsonl`) | GPT-4 (`e1_c`) | 4,500 | 5,000 | 5,120,000 | 3.7074 | 40.7472 |
| **GPT-5 Held-Out Val** (`data/gpt5/val.jsonl`) | **GPT-5** | **100,000** | **10,000** | **10,240,000** | **3.5084** | **33.3946** |

On the shared GPT-4 validation set, GPT-5 reduced evaluation loss by **`-0.2094` (`-5.57%`)** and perplexity by **`-8.1344` (`-18.89%`)**.

---

## 6. Text Generation Samples

Generated with `src.generation.generate.generate` (`temperature = 0.8`, `max_new_tokens = 30`, `seed = 42`):

- **Prompt:** `"Once upon a time"`  
  **GPT-5:** `"Once upon a time , there was a big and b ab s be ar . He was ing a c all ed it . She c le s ure to take it ."`
- **Prompt:** `"The magic"`  
  **GPT-5:** `"The m ag ic . She ran to Lily and b ab s be g ar . They p ret end from the sm o o s . They bo ok en ough the"`
- **Prompt:** `"One day, "`  
  **GPT-5:** `"One day , the e was very pr and b ran . It was good by ing ing . F in ally , he got s , but he didn ' t want"`

**Honest Quality Assessment:** Although perplexity dropped substantially, a `111,616`-parameter model (`D=32`, `vocab=512`, `context=64`) remains severely capacity-limited. It produces recognizable TinyStories vocabulary, punctuation, and short local phrases (`"Once upon a time , there was a big"`, `"didn ' t want"`), but still exhibits fragmented subwords (`"b ab s"`, `"sm o o s"`) and lacks multi-sentence narrative coherence or reasoning.

---

## 7. Artifact Paths & SHA-256 Checksums

- **Checkpoint:** [`checkpoints/gpt5/gpt5_baseline.pt`](../../../checkpoints/gpt5/gpt5_baseline.pt) (`1,464,515` bytes, `step=10000`)  
  SHA-256: `006cac53f7062967aadfa5eafb16aeb14bf4e35c68e3adf2776575af1402d4af`
- **Config:** [`experiments/gpt5/configs/gpt5.yaml`](../configs/gpt5.yaml)
- **Tokenizer:** [`data/processed/gpt3_tokenizer.json`](../../../data/processed/gpt3_tokenizer.json)
- **Dataset Manifest:** [`data/gpt5/dataset_manifest.json`](../../../data/gpt5/dataset_manifest.json)
- **Evaluation JSON:** [`experiments/gpt5/results/gpt5_evaluation.json`](gpt5_evaluation.json)

