# GPT-5 Final Experiment Report — Quality Mode & Full TinyStories Training

**Final Model Name:** `gpt5_full`  
**Branch:** `feature/gpt4-architecture-experiments`  
**Training Hardware:** Google Colab Tesla T4 GPU (final full run & sweep); verified locally on CPU  
**Completion Date:** 2026-11-25  
**Status:** COMPLETED & VERIFIED  

---

## 1. Motivation & Progression

1. **Initial GPT-5 Baseline (`gpt5`, `111,616` params):** Kept GPT-4 E1-C's `6L, D=32, V=512, T=64` architecture and trained `10,000` steps on `100,000` TinyStories docs (`10.24M` tokens). On the shared `V=512` validation set, perplexity dropped from `43.0553` (GPT-4 E1-C) to **`34.9209`** (`-18.89%`), proving data scaling helped — however, `D=32` and `V=512` remained capacity-bottlenecked and fragmented words.
2. **GPT-5 Quality Mode Sweep (Colab T4 GPU, `100,000` docs, `22.95M` tokens, `V=2048` BPE):**
   Tested 4 capacity/schedule configurations with `lr_schedule: cosine`, `warmup_steps: 200`, `grad_clip: 1.0`, `batch_size: 32`, `4,000` steps:

| Config | Params | Time (s) | Min Val Loss | Eval Loss | Eval PPL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `sweep_small` (`4L, D=128, T=128, lr=6e-4`) | 1,336,064 | 78.4 | 3.0103 | 3.0689 | 21.5185 |
| **`sweep_medium` (`6L, D=192, T=128, lr=6e-4`)** | **3,482,624** | **193.9** | **2.5662** | **2.6255** | **13.8116** |
| `sweep_medium_lr3e4` (`6L, D=192, T=128, lr=3e-4`) | 3,482,624 | 193.5 | 2.9031 | 2.9735 | 19.5597 |
| `sweep_ctx256` (`6L, D=192, T=256, lr=5e-4`) | 3,507,200 | 463.6 | 2.5849 | 2.6045 | 13.5244 |

**Winner:** `sweep_medium` achieved virtually identical perplexity to `sweep_ctx256` at **2.4× faster speed**, making `30,000`-step full-dataset training practical in one Colab T4 session.

---

## 2. Final GPT-5 (`gpt5_full`) Architecture & Tokenizer

| Component | Setting |
| :--- | :--- |
| **Model Class** | `src.model.gpt.GPT` |
| **Transformer Layers (`num_layers`)** | 6 |
| **Embedding Dimension (`embedding_dim`)** | 192 |
| **Attention Heads (`num_heads`)** | 6 ($d_k = 32$) |
| **Feed-Forward Dim (`feed_forward_dim`)** | 768 |
| **Context Length (`max_context_length`)** | 128 |
| **Vocabulary Size (`vocab_size`)** | 2,048 |
| **Trainable Parameters** | **3,482,624** |
| **Tokenizer** | `data/gpt5/gpt5_tokenizer.json` (`SubwordTokenizer`, BPE, `Whitespace` pre-tokenizer) |

---

## 3. Full 1.87 GB Dataset & Memmap Pipeline

- **Source:** `roneneldan/TinyStories` (full `train` and `validation` splits, streamed directly into `uint16` `.bin` via [`src/data/prepare_gpt5.py`](../../../src/data/prepare_gpt5.py) and loaded via `np.memmap` in [`src/data/language_dataset.py`](../../../src/data/language_dataset.py)).
- **Manifest ([`data/gpt5_full/dataset_manifest.json`](../../../data/gpt5_full/dataset_manifest.json)):**
  - `train.bin`: `2,119,489` docs, `990,128,818` bytes, **`495,064,409` tokens**, `887` `[UNK]` (`0.0002%`).
  - `val.bin`: `21,990` docs, `9,965,898` bytes, **`4,982,949` tokens**, `4` `[UNK]` (`0.0001%`).

---

## 4. Final Training & Evaluation Results

- **Config:** [`experiments/gpt5/configs/gpt5_full_run.yaml`](../configs/gpt5_full_run.yaml)
- **Training Budget:** `30,000` steps × `32` batch × `128` context = **`122,880,000` token positions** (`1,454.67 s` / ~24.2 min on T4 GPU).
- **Training Losses:** `final_train_loss = 2.0351`, `final_val_loss = 1.9296`, `min_val_loss = 1.8855`.
- **Sampled Validation Evaluation (`100` batches × `32` × `128` = `409,600` tokens, `eval_seed = 10042`):**
  - **Evaluation Loss:** **`1.9689`**
  - **Perplexity:** **`7.1628`** *(under `V=2048` BPE)*

---

## 5. Generation Samples (Coherent Multi-Sentence Stories)

- **Prompt (Colab T4, `seed=42`, `T=0.8`):** `"Once upon a time"`  
  `"Once upon a time , there was a little boy named Timmy . Timmy loved to play with his toy cars and truck s . One day , he saw a blue car zoom ing in the road . He wanted to play with it , but he was too small . Timmy ' s mom said , \" Timmy , let ' s go to the store and get some more car rots . You can play with your cars now .\" While"`
- **Prompt (Local CPU via `scripts/chat.py`, `seed=42`, `T=0.8`):** `"Once upon a time, a little girl found"`  
  `"Once upon a time , a little girl found a green tree . She was so happy to see the beautiful forest . She looked around and saw a big , blue tree like a tall tree . The little girl wondered what the tree was there . She decided to climb the tree and see what was there . She climbed higher and higher and higher until she"`

---

## 6. Final Artifact Paths & SHA-256

- **Final Checkpoint:** [`checkpoints/gpt5_full/gpt5_full_baseline.pt`](../../../checkpoints/gpt5_full/gpt5_full_baseline.pt) (`41,923,481` bytes, `step=30000`, `3,482,624` params)  
  SHA-256: `60e596659924dfc3e95c6e6dac5a0909039fb01db3d0845e1b1641d2490ca8d9`
- **Config:** [`experiments/gpt5/configs/gpt5_full_run.yaml`](../configs/gpt5_full_run.yaml)
- **Tokenizer:** [`data/gpt5/gpt5_tokenizer.json`](../../../data/gpt5/gpt5_tokenizer.json)
- **Evaluation JSON:** [`experiments/gpt5/results/gpt5_full_eval.json`](gpt5_full_eval.json), [`experiments/gpt5/results/sweep_summary.json`](sweep_summary.json)
- **Initial 111K Baseline (preserved):** [`checkpoints/gpt5/gpt5_baseline.pt`](../../../checkpoints/gpt5/gpt5_baseline.pt)
