# GPT-2 Canonical CPU Experiment

## Motivation
GPT-2 explores the impact of a longer training duration on the identical GPT-1 architecture. 
By increasing the training steps from 1,000 to 5,000, we aim to observe if the model continues to learn character-level patterns and achieve a lower validation loss, without any changes to the model dimensions or tokenizer.

## Configuration
All architectural parameters (vocab_size, embedding_dim, max_context_length, num_heads, feed_forward_dim, num_layers) are strictly identical to GPT-1. 
The tokenizer is exactly the same character-level tokenizer built on the TinyStories dataset.

- tokenizer: char
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
- device: CPU
- parameter count: 33,305

## Training Results
- initial loss: 3.3396
- final training loss: 2.0906
- final validation loss: 2.0813
- minimum validation loss: 2.0039
- total training time: 88.01s

## Checkpoint
- checkpoint path: checkpoints/gpt2/gpt2_baseline.pt
- final step: 5000

## Evaluation
- parameter count: 33,305
- architecture:
  - vocab_size: 89
  - embedding_dim: 32
  - max_context_length: 64
  - num_heads: 4
  - feed_forward_dim: 128
  - num_layers: 2
- tokenizer: char
- evaluation loss: 2.0488
- evaluation perplexity: 7.7588

## Generation Samples
Prompt: 'Once upon a time'
Generated: Once upon a time herlaid the frelplay. She the bim helom an lend seas nowing bor cap he tound they therle the ar he taip and the sto and tow he sould ther tald sand to he ovend a in the denunt bew po the hin ling san

Prompt: 'The little girl'
Generated: The little girlly lay fonthe war imering groued. "Lule dand at sthe to the ould she wand and belllte bou?" Sor. Shey mellapy a mastted. "Yo bon fry she aplke to the thra fon an warmald toond an in an's smon fand the

Prompt: 'One day'
Generated: One day shiry smay" haind his wand dand andd. She ucog woompy, the smound a and they from. he wand sher tereng jand hery."One le evennd sound to an oul snoy cicke this the wa thous the the tom to him bot mu

## GPT-1 vs GPT-2
- **GPT-1**: 1,000 training steps
- **GPT-2**: 5,000 training steps
- **Architecture**: Same 33,305-parameter architecture
- **Tokenizer**: Same character-level tokenizer (V=89)
- **Comparison**: The experimental variable was training duration. GPT-2 achieved lower validation loss (2.0488 vs 2.4250) and perplexity (7.7588 vs 11.3026) in its independent evaluation compared to GPT-1. Therefore, the controlled experiment demonstrates the positive effect of longer training.
