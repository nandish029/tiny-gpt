# GPT-1 Baseline

## Configuration
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
- training steps: 1000
- validation interval: 100
- device: cpu
- parameter count: 33305

## Results
| Step | Train Loss | Validation Loss |
|------|------------|-----------------|
| 100 | 3.3396 | 3.3334 |
| 200 | 2.9339 | 2.9607 |
| 300 | 2.8330 | 2.7996 |
| 400 | 2.7116 | 2.7044 |
| 500 | 2.5369 | 2.5927 |
| 600 | 2.5923 | 2.5957 |
| 700 | 2.5899 | 2.4918 |
| 800 | 2.4696 | 2.4493 |
| 900 | 2.4241 | 2.4426 |
| 1000 | 2.4415 | 2.4835 |

- initial loss: 4.7663
- final training loss: 2.4415
- final validation loss: 2.4835
- minimum validation loss: 2.4426

## Checkpoint
- checkpoint path: checkpoints/gpt1/gpt1_baseline.pt
- final step: 1000

## Observations
The loss decreased meaningfully over the 1000 steps. The model successfully ran end-to-end and saved a baseline checkpoint.

## Generation
- generation method: Autoregressive next-token sampling with temperature
- max_new_tokens: 200
- temperature: 0.8

### Samples

**Prompt 1:** `Once upon a time`
**Generated:** `Once upon a time Thelaithe ta d in thesad so - bho he aga helera Thas n wind bor cap he toune t wy themlin an ar he taip wip thoust ind s pwithe id, ser he alar ar tet to o hein tre the d au t ned pe t hit t ved b go`

**Prompt 2:** `The little girl`
**Generated:** `The little girllus ay s the soar ixhivire ncupound. an. anend H. h Inout.. Guto s T waney Qfengle te be ?isoor. he we Herapewa me Lked. " tadonond th. muprkited sge bera foy a p Oras hetto a an in an sisy. So ad Olo`

**Prompt 3:** `One day`
**Generated:** `One day shit wa an" hasoano s waniy The anad. Shed Bo ce omt wal so m felyamond theke r merhe wapindd ploeredat tand shy. wan S s punthecke ternd aw oul s he cicat t ssre lir mind o thet ache m tolus cmo her`

**Observations:** The generated text correctly preserves the prompt and generates the requested number of tokens using a character-level vocabulary. The text is largely gibberish as expected for a tiny 33K parameter model trained for only 1000 steps, but it clearly exhibits character-level space structuring, some valid small words ("in", "the", "an", "and", "he"), and punctuation patterns.

## Final GPT-1 Baseline Evaluation
- parameter count: 33305
- architecture:
  - vocab_size: 89
  - embedding_dim: 32
  - max_context_length: 64
  - num_heads: 4
  - feed_forward_dim: 128
  - num_layers: 2
- tokenizer: CharacterTokenizer
- validation loss: 2.4248
- validation perplexity: 11.3001
- training loss from actual training run: 2.4415
- generation settings: max_new_tokens=200, temperature=0.8
- checkpoint path: checkpoints/gpt1/gpt1_baseline.pt
- checkpoint step: 1000

### Final Generation Samples
**Prompt:** `Once upon a time`
**Generated:** `Once upon a time Thelaithe ta d in thesad so - bho he aga helera Thas n wind bor cap he toune t wy themlin an ar he taip wip thoust ind s pwithe id, ser he alar ar tet to o hein tre the d au t ned pe t hit t ved b go`

**Prompt:** `The little girl`
**Generated:** `The little girllus ay s the soar ixhivire ncupound. an. anend H. h Inout.. Guto s T waney Qfengle te be ?isoor. he we Herapewa me Lked. " tadonond th. muprkited sge bera foy a p Oras hetto a an in an sisy. So ad Olo`

**Prompt:** `One day`
**Generated:** `One day shit wa an" hasoano s waniy The anad. Shed Bo ce omt wal so m felyamond theke r merhe wapindd ploeredat tand shy. wan S s punthecke ternd aw oul s he cicat t ssre lir mind o thet ache m tolus cmo her`

## GPT-1 Baseline Conclusion
GPT-1 successfully learned measurable character-level patterns and reduced training loss substantially to ~2.44. It produced valid autoregressive output and successfully preserves context. As expected at this tiny scale (33K parameters) and short training duration, it generated mostly incoherent text, though it exhibits structural learning such as spacing and frequent character n-grams. It serves as a fully verified baseline for controlled GPT-2 improvements.
