# E1: Transformer Depth Scaling Experiments

**Status:** INFRASTRUCTURE READY (BUILD & TEST VERIFIED)
**Parent Plan:** GPT-4 Architecture Experiment Suite
**Branch:** `feature/gpt4-architecture-experiments`
**Baseline Reference:** GPT-3 Canonical Baseline ($L=2$, 60,800 parameters)

---

## 1. Research Question

How does increasing Transformer network depth ($L \in \{2, 3, 4, 6\}$) impact language modeling validation loss and perplexity on TinyStories when all other dimensional, architectural, and optimization hyperparameters remain strictly controlled?

Specifically:
- Does additional depth provide superior hierarchical abstraction and lower perplexity within a fixed 5,000-step training budget?
- At what layer count do diminishing returns or depth-induced optimization friction emerge given a compact embedding dimension of $D = 32$?

---

## 2. Hypothesis

Increasing Transformer depth from 2 to 3 and 4 layers will lower validation loss and perplexity relative to the GPT-3 baseline by enabling more sequential non-linear representations of text structure. However, expanding to 6 layers with $D = 32$ without residual scaling or wider dimensions may yield diminishing returns or slower early convergence under a fixed 5,000-step training budget.

---

## 3. Independent and Controlled Variables

### Independent Variable:
- **Transformer Layer Depth ($L$):**
  - Baseline: 2 layers
  - E1-A: 3 layers
  - E1-B: 4 layers
  - E1-C: 6 layers

### Controlled Variables:
To ensure strict scientific isolation and fair comparisons:
- **Vocabulary Size ($V$):** 512 (Subword BPE)
- **Embedding Dimension ($D$):** 32
- **Context Length ($T$):** 64 tokens
- **Attention Heads ($H$):** 4 heads ($d_k = 8$)
- **Feed-Forward Dimension ($d_{ff}$):** 128 (expansion factor of $4\times$)
- **Dataset & Split:** TinyStories processed splits (`data/processed/train.jsonl` and `data/processed/val.jsonl`)
- **Tokenizer:** Subword BPE tokenizer (`data/processed/gpt3_tokenizer.json`)
- **Optimizer:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay = 0.01)
- **Learning Rate:** $3.0 \times 10^{-4}$
- **Batch Size:** 16 sequences per step
- **Training Budget:** Exactly 5,000 steps (80,000 sequences / 5,120,000 tokens)
- **Validation Interval:** Every 100 steps
- **Evaluation Settings:** 50 validation batches of batch size 16 (800 sequences / 51,200 tokens)
- **Random Seed:** 42

---

## 4. Experiment Configurations & Parameter Scaling

Each additional Transformer block introduces exactly $12,704$ parameters:
$$\Delta P = \underbrace{32 \times 96 + 96}_{\text{QKV}} + \underbrace{32 \times 32 + 32}_{\text{Out}} + \underbrace{64}_{\text{LN}_1} + \underbrace{32 \times 128 + 128}_{\text{FFN}_1} + \underbrace{128 \times 32 + 32}_{\text{FFN}_2} + \underbrace{64}_{\text{LN}_2} = 12,704$$

Total parameter count as a function of depth $L$:
$$P(L) = 35,392 + 12,704 \times L$$

| Experiment ID | Configuration File | Layers ($L$) | Trainable Parameters | Parameter Delta vs Baseline | Estimated Training Memory |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (GPT-3)** | `baseline.yaml` | 2 | **60,800** | — (Reference) | ~1.93 MB |
| **E1-A** | `e1_a.yaml` | 3 | **73,504** | +12,704 (+20.89%) | ~2.62 MB |
| **E1-B** | `e1_b.yaml` | 4 | **86,208** | +25,408 (+41.79%) | ~3.32 MB |
| **E1-C** | `e1_c.yaml` | 6 | **111,616** | +50,816 (+83.58%) | ~4.70 MB |

---

## 5. Directory Structure & Isolation

All artifacts, checkpoints, and evaluation results are isolated into dedicated experiment subdirectories to prevent mutual collisions or overwriting canonical baselines:

```
experiments/gpt4/e1_depth/
├── README.md                                # This document
├── configs/
│   ├── baseline.yaml                        # 2 layers reference (GPT-3 identical)
│   ├── e1_a.yaml                            # 3 layers
│   ├── e1_b.yaml                            # 4 layers
│   └── e1_c.yaml                            # 6 layers
└── results/                                 # Dedicated JSON / Markdown results
    ├── .gitkeep
    └── ...

checkpoints/gpt4/e1_depth/
├── baseline/                                # Isolated from checkpoints/gpt3/
├── e1_a/
├── e1_b/
└── e1_c/
```

Canonical checkpoints (`checkpoints/gpt1`, `checkpoints/gpt2`, `checkpoints/gpt3`) and configs (`configs/gpt3.yaml`) remain strictly read-only and immutable.

---

## 6. Evaluation Metrics

1. **Validation Cross-Entropy Loss:** Evaluated every 100 steps during training and independently over 50 held-out validation batches.
2. **Perplexity (PPL):** Computed as $\exp(\mathcal{L}_{\text{val}})$.
3. **Training Loss Trajectory:** Convergence rate across steps $100 \dots 5000$.
4. **Minimum Validation Loss:** Best generalization checkpoint step and loss.
5. **Computational Throughput:** Steps per second and wall-clock training time.
6. **Qualitative Text Generation:** Sample generation across standard prompts (`"Once upon a time"`, `"The magic"`, `"One day, "`) at temperature 0.8.

---

## 7. Reusable Experiment Runner

The experiment infrastructure provides [`src/experiments/runner.py`](file:///C:/Users/U%20Naveen%20Kumar/.gemini/antigravity-ide/scratch/tiny-gpt/src/experiments/runner.py) and CLI [`scripts/run_experiment.py`](file:///C:/Users/U%20Naveen%20Kumar/.gemini/antigravity-ide/scratch/tiny-gpt/scripts/run_experiment.py):

- **Default Preflight Inspection (Read-Only):**
  ```bash
  python scripts/run_experiment.py --config experiments/gpt4/e1_depth/configs/e1_a.yaml
  ```
  Reports model dimensions, parameter counts, memory estimates, and paths without performing any training.
- **Evaluation Mode (Read-Only):**
  ```bash
  python scripts/run_experiment.py --config experiments/gpt4/e1_depth/configs/e1_a.yaml --evaluate
  ```
- **Training Mode (Explicit Opt-In Only):**
  ```bash
  python scripts/run_experiment.py --config experiments/gpt4/e1_depth/configs/e1_a.yaml --train
  ```
  Training is never executed accidentally.

---

## 8. Limitations & Scope

1. **Embedding Dimension Bottleneck:** With $D = 32$, deeper networks may encounter representational bottlenecks if residual stream width is insufficient.
2. **Compute Budget:** Training is limited to 5,000 steps per model to maintain accessible CPU iteration; asymptotic behavior beyond 5,000 steps is not evaluated.
3. **Deterministic Single-Seed Baseline:** Experiments use fixed random seed 42 to enable direct paired comparisons against canonical GPT-3.
