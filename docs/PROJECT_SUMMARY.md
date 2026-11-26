# Tiny-GPT Project Summary (GPT-1 through GPT-5)

This document summarizes the progression of all five model versions in the Tiny-GPT PyTorch project. **`gpt5_full` (GPT-5 Quality Mode)** is the final project model.

## Progression Overview

| Version | What Changed | Intended Demonstration | Key Architecture & Training Settings | Main Evaluation Result |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-1** | Initial baseline | End-to-end Transformer training from scratch | Char (`V=89`), `L=2, D=32, H=4, FFN=128, T=64`, `33,305` params, `1,000` steps | Eval loss `~2.4250`, PPL `~11.3026` (char) |
| **GPT-2** | Training steps (`1k → 5k`) | Effect of longer optimization on fixed architecture | Same as GPT-1 (`33,305` params), `5,000` steps | Eval loss `2.0488`, PPL `7.7588` (char) |
| **GPT-3** | Subword BPE tokenizer (`V=512`) | Subword tokenization vs character-level modeling | BPE (`V=512`), `L=2, D=32, H=4, FFN=128, T=64`, `60,800` params, `5,000` steps | Eval loss `3.8975`, PPL `49.2796` (`V=512`) |
| **GPT-4 (`E1`)** | Depth scaling (`L=2, 3, 4, 6`) | Effect of Transformer depth under fixed `D=32, V=512` | `L=6` (`E1-C`): `111,616` params, `4,500` train docs, `5,000` steps (`5.12M` tokens) | `L=2`: `3.9121` (`50.00`) → **`L=6`: `3.7625` (`43.0553`)** (`V=512`) |
| **GPT-5 (111K Baseline)** | Data scaling (`4.5k → 100k` docs, `10k` steps) | Isolating TinyStories data scale on frozen `111K` model | `L=6, D=32, V=512, T=64`, `111,616` params, `100,000` docs, `10,000` steps (`10.24M` tokens) | Eval loss `3.5531`, PPL `34.9209` (`V=512`, `-18.89%` PPL vs GPT-4) |
| **GPT-5 (`gpt5_full`, Final)** | Quality Mode sweep + full `1.87 GB` TinyStories (`uint16` memmap) + cosine LR | Coherent multi-sentence story generation via capacity, vocab, and full-corpus scaling | `L=6, D=192, H=6, FFN=768, T=128, V=2048`, **`3,482,624` params**, `2.12M` docs (`495M` tokens), `30,000` steps (`122.88M` tokens) on T4 GPU | **Eval loss `1.9689`, PPL `7.1628`** (`V=2048`, `409,600` eval tokens) |

## Final Project Model

- **Checkpoint:** [`checkpoints/gpt5_full/gpt5_full_baseline.pt`](../checkpoints/gpt5_full/gpt5_full_baseline.pt)
- **Config:** [`experiments/gpt5/configs/gpt5_full_run.yaml`](../experiments/gpt5/configs/gpt5_full_run.yaml)
- **Tokenizer:** [`data/gpt5/gpt5_tokenizer.json`](../data/gpt5/gpt5_tokenizer.json)
- **Detailed Report:** [`experiments/gpt5/results/GPT5_report.md`](../experiments/gpt5/results/GPT5_report.md)
