# GPT-1 Baseline

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
- training steps: 1
- validation interval: 100
- device: cpu
- parameter count: 33305

## Results
| Step | Train Loss | Validation Loss |
|------|------------|-----------------|
| 1 | 5.3226 | 5.2386 |

- initial loss: 5.4060
- final training loss: 5.3226
- final validation loss: 5.2386
- minimum validation loss: 5.2386

## Checkpoint
- checkpoint path: C:\Users\UNAVEE~1\AppData\Local\Temp\tmpls64fk5h\checkpoints\gpt1_baseline.pt
- final step: 1

## Observations
The loss decreased meaningfully over the training. The model successfully ran end-to-end and saved a baseline checkpoint.
