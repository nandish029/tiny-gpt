# GPT-2 FROZEN

GPT-2 is officially **FROZEN**.

This document certifies that the **CANONICAL CPU GPT-2 BASELINE** experiment has concluded.
- GPT-2 architecture is frozen.
- GPT-2 character tokenizer is frozen.
- GPT-2 checkpoint (`checkpoints/gpt2/gpt2_baseline.pt`) is frozen.

## Details
- Architecture: 33,305 parameters (2 layers, 4 heads, 32 dim, 64 context)
- Tokenizer: Character-level (`data/processed/tokenizer.json`)
- Dataset: TinyStories (`data/processed/train.jsonl`)
- Configuration: `configs/gpt2.yaml`
- Training steps: 5000
- Device: CPU
- Final metrics: Training Loss (2.0906), Eval Loss (2.0488), Eval Perplexity (7.7588)
- Exact experimental difference from GPT-1: Training duration (5000 steps vs 1000 steps). All other architectures and tokenizers are strictly identical.

## Rules
- No further GPT-2 training or configuration changes should be made.
- Do NOT re-train GPT-2.
- Do NOT change the tokenizer methodology.
- Do NOT modify the baseline checkpoint.
- GPT-3 is a separate subword experiment and must be conducted in a separate configuration.
