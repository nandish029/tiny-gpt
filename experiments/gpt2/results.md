# GPT-2 Controlled Training Experiment

## Motivation
GPT-2 explores the impact of a longer training duration on the identical GPT-1 architecture. 
By increasing the training steps from 1,000 to 5,000, we aim to observe if the model continues to learn character-level patterns and achieve a lower validation loss, without any changes to the model dimensions or tokenizer.

## Configuration
All architectural parameters (vocab_size, embedding_dim, max_context_length, num_heads, feed_forward_dim, num_layers) are strictly identical to GPT-1. 
The tokenizer is exactly the same character-level tokenizer built on the TinyStories dataset.

**Changes from GPT-1**:
- `train_steps`: Increased from 1,000 to 5,000.

## Results
*(To be populated after the long GPU training run)*

## Comparison against GPT-1
*(To be populated after the long GPU training run)*

## Conclusion
*(To be populated after the long GPU training run)*
