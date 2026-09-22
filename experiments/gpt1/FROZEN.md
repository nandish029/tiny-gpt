# GPT-1 Canonical Freeze

**STATUS: FROZEN**

The canonical GPT-1 CPU baseline has been established and verified.
All architecture, parameter configurations, and experimental parameters are frozen and immutable.

## Architecture
- vocab size: 89
- embedding dimension: 32
- context length: 64
- heads: 4
- feed-forward dimension: 128
- layers: 2
- Total Parameters: 33,305

## Training Profile
- 1000 steps
- device: cpu
- dataset: 5000 deterministic samples
- tokenizer: char

## Integrity
- Model architecture verified statically against `configs/gpt1.yaml`.
- Model architecture verified dynamically during load.
- Generation checks passed.
- Ephemeral canonical checkpoint (`checkpoints/gpt1/gpt1_baseline.pt`) is locally available but purposefully untracked in Git.

Do not modify the GPT-1 architecture or configs.
