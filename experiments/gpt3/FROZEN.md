# GPT-3 CANONICAL CPU EXPERIMENT

**STATUS: FROZEN**

This document certifies the canonical GPT-3 baseline experiment. No further architecture, training, or configuration changes belong to this experiment.

## Architecture & Configuration
- tokenizer: subword BPE
- vocab: 512
- embedding dimension: 32 (D32)
- context length: 64 (context64)
- heads: 4 (H4)
- feed-forward dimension: 128 (FFN128)
- layers: 2
- parameters: 60,800
- batch size: 16
- learning rate: 3e-4
- seed: 42

## Training Details
- steps: 5000
- device: CPU
- training time: 99.24 seconds
- initial loss: 6.4330
- final train loss: 3.9417
- final validation loss: 3.9174
- minimum validation loss: 3.8512 at step 4600

## Evaluation
- evaluation loss: 3.8975
- perplexity: 49.2796

## Checkpoint
- checkpoint path: checkpoints/gpt3/gpt3_baseline.pt
- step: 5000

*GPT-3 is officially frozen.*
