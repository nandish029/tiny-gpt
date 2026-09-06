# GPT-1 Frozen Baseline

GPT-1 is now frozen and will not be modified for subsequent experiments.

- checkpoint:
  checkpoints/gpt1/gpt1_baseline.pt

- parameter count:
  33,305

- architecture:
  vocab_size=89
  embedding_dim=32
  context_length=64
  num_heads=4
  feed_forward_dim=128
  num_layers=2

- tokenizer:
  character-level

- training steps:
  1000

- seed:
  42

- final training loss:
  2.4415

- final validation loss:
  2.4835

- minimum validation loss:
  2.4426

Future GPT-2 experiments must not modify this checkpoint.
