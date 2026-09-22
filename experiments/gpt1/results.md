# GPT-1 Canonical Training Results

## Configuration
- tokenizer type: char
- vocab size: 89
- embedding dimension: 32
- context length: 64
- heads: 4
- feed-forward dimension: 128
- layers: 2
- batch size: 16
- learning rate: 0.0003
- seed: 42

## Training
- dataset: TinyStories
- training steps: 1000
- device: CPU
- parameter count: 33,305

## Results
- final train loss: 2.4415
- final validation loss: 2.4835
- evaluation validation loss: approximately 2.4250
- validation perplexity: approximately 11.3026

## Checkpoint
- checkpoint path: checkpoints/gpt1/gpt1_baseline.pt
- final step: 1000

## Observations
The model successfully ran end-to-end and saved a baseline canonical checkpoint.
