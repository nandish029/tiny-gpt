# GPT-3 Experiment

## Objective
The primary objective of GPT-3 is to introduce a robust subword tokenizer (BPE) replacing the simple character-level tokenizer from GPT-1/GPT-2, and evaluate its impact on token efficiency, loss, and generated output quality on the TinyStories dataset.

## Controlled Variable
- **Tokenizer**: Changed from CharacterTokenizer to a BPE Subword Tokenizer (HuggingFace `tokenizers`).
- **Vocabulary Size**: Explicitly constrained and verified to exactly 512.

## Architecture (Constant)
The internal Transformer architecture (excluding embedding/lm_head matrices due to vocabulary change) remains strictly identical to the GPT-1 and GPT-2 frozen baselines:
- embedding dimension: 32
- max context length: 64
- heads: 4
- feed-forward dimension: 128
- layers: 2
- parameters: 60,800 (increased strictly due to the `V x D` embedding dimension scaling from `V=89` to `V=512`)

## Training Configuration
- Training steps: 5,000
- Batch size: 16
- Learning rate: 3e-4
- Seed: 42
- Device: NVIDIA Tesla T4

## Quantitative Results
- Training time: 37.26 seconds
- Final training loss: 3.8550
- Final validation loss during training: 3.7005
- Independent evaluation validation loss: 3.9218
- Independent evaluation perplexity: 50.4923

## Qualitative Generation Observations
- GPT-3 successfully generates text continuations from prompts.
- It learned story-like patterns, names, punctuation, and common TinyStories structures.
- Generated text is still substantially incoherent.
- Some subword spacing/fragments are visible, e.g. "happ ily", "f ir re", "ne ed".
- *Note:* GPT-3 is a foundational autoregressive pre-training experiment, not a chat or instruction-following model. The overall quality remains extremely low given the tiny 60K parameter scale.

## Limitations
- Subword fragmentation continues to limit readability.
- The 32-dimensional embedding and 2-layer depth severely cap representational capacity.
- 5,000 steps is still far short of full convergence for a 512-vocabulary distribution.

## What carries forward to GPT-4
- The Subword BPE Tokenizer (V=512) is verified and will remain frozen.
- The next step (GPT-4) will focus entirely on Architectural Scaling (e.g., increasing dimension, heads, and layers) while holding the subword tokenizer constant.
