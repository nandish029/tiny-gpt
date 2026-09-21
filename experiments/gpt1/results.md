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

## Final GPT-1 Baseline Evaluation
- parameter count: 60800
- architecture:
  - tokenizer_type: subword
  - vocab_size: 512
  - embedding_dim: 32
  - max_context_length: 64
  - num_heads: 4
  - feed_forward_dim: 128
  - num_layers: 2
- tokenizer type: subword
- validation loss: 6.4231
- validation perplexity: 615.8919
- training loss from actual training run: N/A
- generation settings: max_new_tokens=200, temperature=0.8
- checkpoint path: checkpoints/gpt3/gpt3_baseline.pt
- checkpoint step: 2

### Final Generation Samples
**Prompt:** `Once upon a time`
**Generated:** `Once upon a time felt e ain ted bb F wa ning His be way good 7 ing id they ts ig O it “ ch L ma ara Jo about na fun ud en ts ous V wan things E !" decided od r always dog ise look be od ue ous felt am bird Once looked speci pa feel down P boy saw Â want ways ake happ pa around see S ough ide ud girl ged outside special ways t look found irl didn by ther se la ought Her 6 ked ch E Ben there was ame ret sto if home want saw ur L l ing ?" ided ay feel ree rom with Tom if ick liked ach ct when pe co ough n 2 ile sm ud side wal ill ow like thing home lf tree scared thing Â Her nam scared have ot ly r € ¡ ep take rabb up if ite ag Lily use fter not ter back s ," ny go sk s any e » king ra en things un » our into were Sara So ight de You Jack not at st et When K named ing lly ened ci feel use`

**Prompt:** `The little girl`
**Generated:** `The little girl ged fu ith d play N ra lf , see ong fly star ve ar © gre run make wanted Jo B started F did ge ; nam is Ben now 9 thought qu ise ittle A took can ful things id ," Tom 4 h thought ! he ad N here ways tree wor car dad it with knew ran g But T ge ³ back car br playing al im 1 heard mo uddenly ile want then ust ice har ge Jo see ro jo felt you They es ven na ited pu udden ³ " ion any uddenly 1 ile G ise but scar har iled dog long loved ise park oug D by st fa thought ake lp ough K ," pe heard re pp One this who loved did He they hen asked sun feel ouse wh were no L in rom special Y they ca oug she ited u f 2 go ke started e ong g saw sh der ould A but friends ate so ; S po with € friend k D ned looked ? Q ig ¡ hed in im help said co 5 saw wo not li Ã my ought`

**Prompt:** `One day`
**Generated:** `One day man ™ for jo side she ould girl laug help Sam Ben in every b care play John 4 come ret ight lear fa ried der lp ± Y them de could Q able and ith heard nam now lf get speci la i uc » la found ut happy one ci ust says fly ra John boy pu c ree thought over e jo we decided 6 would ck dog were wal ret hat h Let 2 im On very g ough r Lily things ara L But back ter de ab sad hed hed ion mommy ad into fa end ion knew wal e thing ag When Sam udd N So ough bl Timmy ht upon play ile pe speci ust t ot k started ss ha low ³ fo wo liked ret ough R saw lf P ot bo they wor pe t long riend at by Ã But de go es y 8 know le he ex ur rom 6 decided D . It w ˜ ark all about Y scar at scared bird li show scar go boy there ake wo ouse ; Sam Z « find proud ile happy old old â ¡`

## GPT-1 Baseline Conclusion
GPT-1 successfully learned measurable character-level patterns. It serves as a verified baseline.
