# GPT-2 Experiment

## Motivation
GPT-2 explores the impact of a longer training duration on the identical GPT-1 architecture. 
By increasing the training steps from 1,000 to 5,000, we aim to observe if the model continues to learn character-level patterns and achieve a lower validation loss, without any changes to the model dimensions or tokenizer.

## Configuration
All architectural parameters (vocab_size, embedding_dim, max_context_length, num_heads, feed_forward_dim, num_layers) are strictly identical to GPT-1. 
The tokenizer is exactly the same character-level tokenizer built on the TinyStories dataset.

- tokenizer: CharacterTokenizer
- vocab size: 89
- embedding dimension: 32
- context length: 64
- heads: 4
- feed-forward dimension: 128
- layers: 2
- batch size: 16
- learning rate: 3e-4
- seed: 42

## Training
- dataset: TinyStories
- training steps: 5000
- validation interval: 100
- device: CUDA/T4
- parameter count: 33,305

## Results
- final training loss: 2.0906
- final validation loss: 2.0813
- minimum validation loss: 2.0039

## Checkpoint
- checkpoint path: checkpoints/gpt2/gpt2_baseline.pt
- final step: 5000

## Final GPT-2 Evaluation
- parameter count: 33,305
- architecture:
  - vocab_size: 89
  - embedding_dim: 32
  - max_context_length: 64
  - num_heads: 4
  - feed_forward_dim: 128
  - num_layers: 2
- tokenizer: CharacterTokenizer
- validation loss: 2.0498
- validation perplexity: 7.7661
- checkpoint path: checkpoints/gpt2/gpt2_baseline.pt
- checkpoint step: 5000

## GPT-2 Conclusion
GPT-2 successfully learned measurable character-level patterns and reduced training loss substantially. It produced valid autoregressive output and successfully preserves context. It serves as a fully verified baseline for further controlled improvements.

## GPT-1 vs GPT-2
- **GPT-1**: 1,000 training steps
- **GPT-2**: 5,000 training steps
- **Architecture**: Same 33,305-parameter architecture
- **Tokenizer**: Same CharacterTokenizer
- **Comparison**: GPT-2 achieved lower validation loss (2.0498) and perplexity (7.7661) in its independent evaluation compared to GPT-1. Therefore, the controlled experiment demonstrates the positive effect of longer training.

*(Note: Qualitative generation comparison is still pending)*
