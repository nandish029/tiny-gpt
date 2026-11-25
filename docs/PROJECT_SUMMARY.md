# Tiny-GPT Project Summary (GPT-1 through GPT-5)

This document summarizes the progression of all five model versions in the Tiny-GPT PyTorch project. **GPT-5** is the final project model.

## Progression Overview

| Version | What Changed | Intended Demonstration | Key Architecture & Training Settings | Main Evaluation Result (Shared TinyStories Val) |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-1** | Initial baseline | End-to-end Transformer training from scratch | Char tokenizer (`V=89`), `L=2`, `D=32`, `H=4`, `FFN=128`, `T=64`, `33,305` params, `1,000` steps | Eval loss `~2.4250`, PPL `~11.3026` (char-level) |
| **GPT-2** | Training steps (`1k → 5k`) | Effect of longer optimization on fixed architecture | Same as GPT-1 (`V=89`, `33,305` params), `5,000` steps | Eval loss `2.0488`, PPL `7.7588` (char-level) |
| **GPT-3** | Subword BPE tokenizer (`V=512`) | Subword tokenization vs character-level modeling | Subword BPE (`V=512`), `L=2`, `D=32`, `H=4`, `FFN=128`, `T=64`, `60,800` params, `5,000` steps | Eval loss `3.8975`, PPL `49.2796` (subword-level, not directly comparable to char PPL) |
| **GPT-4 (`E1`)** | Depth scaling (`L=2, 3, 4, 6`) | Effect of Transformer depth under fixed `D=32`, `V=512`, `5k` steps | `L=6` selected (`E1-C`): `111,616` params, `4,500` train docs, `5,000` steps (`5.12M` tokens) | `L=2`: `3.9121` (`50.00`) → `L=3`: `3.8478` (`46.89`) → `L=4`: `3.7989` (`44.65`) → **`L=6`: `3.7625` (`43.0553`)** |
| **GPT-5 (Final)** | Data & token budget (`4.5k → 100k` docs, `5k → 10k` steps) | Whether larger TinyStories data exposure improves the selected 6-layer model | Frozen GPT-4 `E1-C` (`L=6`, `V=512`, `111,616` params), `100,000` train docs (`30.48M` tokens), `10,000` steps (`10.24M` tokens) | **Eval loss `3.5531`, PPL `34.9209`** (`-18.89%` PPL vs GPT-4 `E1-C`) |

## Final Project Model

- **Model:** GPT-5 ([`checkpoints/gpt5/gpt5_baseline.pt`](../checkpoints/gpt5/gpt5_baseline.pt))
- **Config:** [`experiments/gpt5/configs/gpt5.yaml`](../experiments/gpt5/configs/gpt5.yaml)
- **Tokenizer:** [`data/processed/gpt3_tokenizer.json`](../data/processed/gpt3_tokenizer.json)
- **Detailed Report:** [`experiments/gpt5/results/GPT5_report.md`](../experiments/gpt5/results/GPT5_report.md)

