# GPT-3 Canonical CPU Training

## Configuration
- tokenizer type: subword BPE
- vocab size: 512
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
- training steps: 5000
- validation interval: 100
- device: cpu
- parameter count: 60800
- training time: 99.24 seconds

## Results
| Step | Train Loss | Validation Loss |
|------|------------|-----------------|
| 100 | 5.7653 | 5.7659 |
| 200 | 5.4779 | 5.4569 |
| 300 | 5.3697 | 5.3361 |
| 400 | 5.3707 | 5.3008 |
| 500 | 5.2446 | 5.2361 |
| 600 | 5.2106 | 5.1269 |
| 700 | 5.0745 | 5.0841 |
| 800 | 4.9882 | 4.9414 |
| 900 | 4.9305 | 4.9559 |
| 1000 | 4.8381 | 4.8651 |
| 1100 | 4.6927 | 4.7364 |
| 1200 | 4.7479 | 4.7288 |
| 1300 | 4.6569 | 4.7289 |
| 1400 | 4.6885 | 4.6429 |
| 1500 | 4.5303 | 4.5652 |
| 1600 | 4.5296 | 4.4566 |
| 1700 | 4.4401 | 4.6293 |
| 1800 | 4.5179 | 4.4007 |
| 1900 | 4.4130 | 4.4287 |
| 2000 | 4.1871 | 4.3668 |
| 2100 | 4.3097 | 4.3739 |
| 2200 | 4.1808 | 4.2520 |
| 2300 | 4.2588 | 4.2167 |
| 2400 | 4.2485 | 4.2825 |
| 2500 | 4.1797 | 4.2264 |
| 2600 | 4.1806 | 4.1922 |
| 2700 | 4.1600 | 4.2118 |
| 2800 | 4.2145 | 4.1729 |
| 2900 | 4.1358 | 4.2483 |
| 3000 | 4.1887 | 3.9943 |
| 3100 | 4.1288 | 4.1329 |
| 3200 | 4.1164 | 4.2162 |
| 3300 | 4.0779 | 3.9807 |
| 3400 | 4.1679 | 4.0688 |
| 3500 | 4.0535 | 4.1963 |
| 3600 | 3.8911 | 4.1247 |
| 3700 | 4.0262 | 3.9346 |
| 3800 | 3.8605 | 4.0422 |
| 3900 | 4.1164 | 3.9433 |
| 4000 | 4.0122 | 3.9175 |
| 4100 | 3.7294 | 4.1763 |
| 4200 | 3.9522 | 4.0113 |
| 4300 | 4.0595 | 3.8767 |
| 4400 | 4.0115 | 3.9542 |
| 4500 | 3.8796 | 3.8880 |
| 4600 | 3.8415 | 3.8512 |
| 4700 | 3.8074 | 3.9059 |
| 4800 | 3.8378 | 3.8801 |
| 4900 | 3.9242 | 3.9616 |
| 5000 | 3.9417 | 3.9174 |

- initial loss: 6.4330
- final training loss: 3.9417
- final validation loss: 3.9174
- minimum validation loss: 3.8512 at step 4600

## Checkpoint
- checkpoint path: checkpoints/gpt3/gpt3_baseline.pt
- final step: 5000

## Observations
The loss decreased meaningfully over the training. The model successfully ran end-to-end and saved a baseline checkpoint.

## Evaluation Pipeline
- evaluation validation loss: 3.8975
- perplexity: 49.2796

## Generation Samples
**Prompt:** Once upon a time
**Output:** Once upon a time there was l u as one . One day , there was a time , there was a z um my . But then , but the be lo st

**Prompt:** The magic
**Output:** The m ag ic ed up and help s ur pr id . They started to make a lo ll him too make it was re gr ound ed ing at re . It

**Prompt:** One day, 
**Output:** One day , Ben . They have more s er be an im al s . The re pl an im my were mo to his mom asked , and said good in

