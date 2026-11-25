# GPT-4 Experiment E1-A — 3 Layers Depth Scaling Report

**Experiment Name:** `e1_a`  
**Experiment Family:** `e1_depth`  
**Parent Plan:** GPT-4 Architecture Experiment Suite  
**Branch:** `feature/gpt4-architecture-experiments`  
**Execution Device:** CPU (AMD64)  
**Execution Timestamp:** 2026-11-04  
**Status:** COMPLETED & VERIFIED  

---

## 1. Scientific Overview & Objective

The **E1-A Experiment** evaluates the impact of increasing Transformer depth from 2 to 3 layers ($L = 3$) while strictly preserving all other dimensional, architectural, and optimization hyperparameters:
- **Vocabulary Size ($V$):** 512 (Subword BPE)
- **Embedding Dimension ($D$):** 32
- **Transformer Layers ($L$):** 3 ($\Delta = +1$ layer, $+12,704$ parameters)
- **Attention Heads ($H$):** 4 (head dimension $d_k = 8$)
- **Feed-Forward Dimension ($d_{ff}$):** 128 ($4 \times D$)
- **Context Length ($T$):** 64 tokens
- **Batch Size ($B$):** 16 sequences ($1,024$ tokens per optimizer step)
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = 0.01)
- **Learning Rate ($\eta$):** $3.0 \times 10^{-4}$
- **Training Budget:** Exactly 5,000 steps ($80,000$ sequences / $5,120,000$ total tokens)
- **Random Seed:** 42 (with independent seeded generators for training and validation)
- **Validation Interval:** Every 100 optimizer steps

---

## 2. Parameter Scaling Verification

Each Transformer block adds:
$$\Delta P = \underbrace{32 \times 96 + 96}_{\text{QKV}} + \underbrace{32 \times 32 + 32}_{\text{Out}} + \underbrace{64}_{\text{LN}_1} + \underbrace{32 \times 128 + 128}_{\text{FFN}_1} + \underbrace{128 \times 32 + 32}_{\text{FFN}_2} + \underbrace{64}_{\text{LN}_2} = 12,704$$

Total trainable parameters:
$$P(3) = 35,392 + 12,704 \times 3 = 73,504$$
- **Configured & Verified:** Exactly **73,504** trainable parameters ($\Delta = +12,704$ vs Baseline).

---

## 3. Training Execution & Comparison with Baseline

The 5,000 optimizer steps ran to completion on local CPU without errors or divergence:

| Metric | E1 Baseline ($L=2$) | E1-A ($L=3$) | Delta ($L=3$ vs $L=2$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer Layers** | 2 | 3 | +1 | +50.0% |
| **Trainable Parameters** | 60,800 | 73,504 | +12,704 | +20.9% |
| **Training Steps** | 5,000 | 5,000 | 0 | 0.0% |
| **Wall-Clock Duration** | 95.52 s | 143.04 s | +47.52 s | +49.7% |
| **Step Throughput** | 19.10 ms/step | 28.61 ms/step | +9.51 ms/step | +49.8% |
| **Final Train Loss** | 3.7990 | **3.7541** | -0.0449 | **-1.18% (Better)** |
| **Final Val Loss** | 3.9662 | **3.9174** | -0.0488 | **-1.23% (Better)** |
| **Minimum Val Loss** | 3.7648 | **3.6983** | -0.0665 | **-1.77% (Better)** |
| **Independent Eval Loss** | 3.9121 | **3.8478** | -0.0643 | **-1.64% (Better)** |
| **Independent Perplexity** | 50.0042 | **46.8913** | -3.1129 | **-6.23% (Better)** |

---

## 4. Independent Model Evaluation

Evaluated via the standardized evaluation procedure across the held-out validation split (`data/processed/val.jsonl`):
- **Evaluation Split:** `val.jsonl` (426,366 bytes, 143,808 tokens)
- **Evaluation Sampling:** 50 validation batches $\times$ batch size 16 $\times$ context 64 = **51,200 tokens**
- **Evaluation Seed:** 10042 (independent RNG generator)
- **Evaluation Loss:** **3.8478**
- **Evaluation Perplexity:** **46.8913**

### Scientific Finding
Increasing depth from 2 to 3 layers successfully reduced independent cross-entropy loss by **$0.0643$** and perplexity by **$3.1129$ points ($-6.23\%$)** under the identical token budget ($5.12\text{M tokens}$) and learning rate schedule. Computational cost scaled almost exactly linearly with depth ($+49.7\%$ runtime for $+50\%$ layer count).

---

## 5. Checkpoint Verification & Strict Reload

The saved checkpoint was verified for structure, strict parameter matching, and numerical stability:

| Attribute | Value |
| :--- | :--- |
| **Checkpoint Path** | `checkpoints/gpt4/e1_depth/e1_a/e1_a_baseline.pt` |
| **File Size** | 948,867 bytes |
| **SHA-256 Checksum** | `fdc7a878e6e9f32d6a46b44846336cd027baa09a1f8286ee2a61a37c5a359dd5` |
| **Checkpoint Step** | 5,000 |
| **Trainable Parameters** | 73,504 |
| **Strict Reload (`strict=True`)** | 0 missing keys, 0 unexpected keys |
| **Numerical Integrity** | All 73,504 parameters verified finite (`torch.isfinite`) |
| **State Dictionaries** | `model_state_dict`, `optimizer_state_dict`, `step`, `config` |

---

## 6. Text Generation Samples

Generated using the E1-A checkpoint and subword BPE tokenizer (`data/processed/gpt3_tokenizer.json`, temperature = 0.8, max_new_tokens = 30, seed = 42):

- **Prompt 1:** `"Once upon a time"`  
  **Output:** `"Once upon a time . She was very m and b and wal be e ar . They ing they found something very proud of her s . One day , and they have"`

- **Prompt 2:** `"The magic"`  
  **Output:** `"The m ag ic en se for his mom . He ran k in r ong . The b ro ck . S ac k f an ide a pa ma x , there"`

- **Prompt 3:** `"One day, "`  
  **Output:** `"One day , saw it d y started to f around the c ur ide . F rom that he was playing his new to sa me m in the sk y ."`

---

## 7. Scientific Isolation & Safety Verification

1. **Baseline Checkpoint Integrity:**
   The E1 baseline checkpoint (`checkpoints/gpt4/e1_depth/baseline/baseline_baseline.pt`, SHA-256: `8f3a95961b9d...`) was verified bit-for-bit identical to its pre-run state.
2. **Canonical GPT-1/2/3 Checkpoint Integrity:**
   All canonical artifacts pinned in `configs/artifact_manifest.json` match their original SHA-256 hashes. None were overwritten or modified.
3. **Subsequent Experiments Scope:**
   E1-B ($L=4$) and E1-C ($L=6$) remain **unstarted**.
4. **Git Hygiene:**
   No commits, pushes, merges, or external hub uploads were performed.

