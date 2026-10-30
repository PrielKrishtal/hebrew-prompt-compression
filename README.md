# Hebrew token tax

**Authors:** Priel Krishtal, Liad Albocher

**TL;DR**

- Hebrew needs more tokens than English on every tokenizer tested: 1.49× to 3.65× on average.
- OpenAI's gap fell from 3.65× (`gpt-4`) to 1.50× (`gpt-4o`). Gemini moved from 1.74× to 1.66× on one newer model.
- No tokenizer tested, including Hebrew-specialized open ones, goes below about 1.49×.

How many more tokens does Hebrew cost than English for the same content, on commercial LLM tokenizers? API pricing is per token, so more tokens means a higher bill.

## Hypotheses

- **H1:** for the same content, Hebrew needs more tokens than English (ratio > 1) on every tokenizer we test.
- **H2:** newer tokenizers narrow that gap.

The ratio is Hebrew tokens divided by English tokens, per paragraph. We use the ratio and not the difference because paragraph length varies: a difference grows with length, a ratio does not.

## Method

- **Data:** FLORES+ dev split (`openlanguagedata/flores_plus`), Hebrew and English versions of the same sentences. Sentences are grouped by source article into paragraphs, and 50 paragraphs are sampled (seed 42).
- **Counting:** token counts only, using free methods. No paid generation calls.
  - **OpenAI:** offline with `tiktoken` (`gpt-4`, `gpt-4o`, `gpt-5`).
  - **Gemini:** the `count_tokens` endpoint. The notebook lists the models available to the API key, keeps text models of generation 3 or newer, and probes each one. Every call adds a fixed overhead, which is measured per model and subtracted. The notebook checks that `"a"`, `"a a"` and `"a a a"` count as overhead + 1, 2 and 3.
  - **Reference tokenizers:** loaded locally with `transformers` (tokenizer files only, no model weights, no special tokens). They show what a Hebrew-aware vocabulary costs, and are not priced APIs: `dictalm2` and the DictaLM 3.0 models together with the base models they were initialized from.
- **Tokenizers, not model names:** different models can share one tokenizer, so results are grouped by tokenizer.
  - OpenAI models are grouped by the tokenizer name `tiktoken` reports.
  - Gemini and the reference models publish no comparable name, so two models count as one tokenizer only if they give identical counts on every paragraph in both languages.
- **Data handling:** the FLORES text is not saved or published. The results file holds counts only.
- **Notebook:** [notebooks/01_tokenizer_ratio_baseline.ipynb](notebooks/01_tokenizer_ratio_baseline.ipynb)

## Results

Commercial tokenizers:

| Tokenizer | Models | Mean ratio | Min – max | Hebrew chars/token | English chars/token |
|---|---|---|---|---|---|
| `cl100k_base` (legacy) | `gpt-4` | 3.65 | 2.95 – 4.78 | 1.05 | 4.86 |
| `o200k_base` | `gpt-4o`, `gpt-5` | 1.50 | 1.23 – 1.91 | 2.58 | 4.93 |
| Gemini A | 10 models: `gemini-3-flash-preview` through `gemini-3.8-flash` | 1.74 | 1.45 – 2.25 | 2.20 | 4.87 |
| Gemini B | `gemini-4-argon` | 1.66 | 1.32 – 2.11 | 2.30 | 4.87 |

Reference tokenizers (open models, not commercial APIs):

| Tokenizer | Models | Mean ratio | Min – max | Hebrew chars/token | English chars/token |
|---|---|---|---|---|---|
| `dictalm2` | `dictalm2.0` | 1.53 | 1.21 – 2.00 | 2.26 | 4.38 |
| Mistral vocabulary | DictaLM-3.0-24B, DictaLM-3.0-Nemotron-12B, Mistral-Small-3.1, Nemotron-Nano-v2 | 1.64 | 1.30 – 2.03 | 2.27 | 4.73 |
| Qwen vocabulary | DictaLM-3.0-1.7B, Qwen3-1.7B | 1.49 | 1.26 – 1.80 | 2.52 | 4.78 |

![Hebrew/English token ratio by tokenizer](results/01_tokenizer_ratios.png)

### Findings

- **H1 holds on every tokenizer tested.** On average a Hebrew paragraph needs 1.49× to 3.65× the tokens of its English version.
- **H2 is supported for OpenAI.** The gap fell from 3.65× (`gpt-4`) to 1.50× (`gpt-4o`). `gpt-5` uses the same tokenizer as `gpt-4o`, so it adds no further improvement.
- **Gemini is flat across generation 3.** Ten Gemini 3.x models, from flash-lite to pro, share one tokenizer. The single newer model, `gemini-4-argon`, is slightly cheaper for Hebrew (1.66× against 1.74×), which is one data point.
- **DictaLM 3.0 did not change the vocabulary.** Each DictaLM 3.0 model counts identically to its base model on all 100 measurements, so Dicta kept the base tokenizer. Its Hebrew ability comes from additional training, not from a Hebrew-specific vocabulary.
- **A Hebrew-aware vocabulary is not much cheaper than `o200k_base`.** `dictalm2`, built for Hebrew, reaches 1.53×, and the best reference tokenizer (Qwen) reaches 1.49×. Both are close to `o200k_base` at 1.50×. `dictalm2` also uses more English tokens than the others, which raises its ratio.

**Overall H2 verdict: partially supported.** The evidence is one OpenAI transition (`gpt-4` to `gpt-4o`), one small Gemini change (`gemini-4-argon`), and Claude is untested.

Raw per-paragraph counts are in [results/01_tokenizer_counts.csv](results/01_tokenizer_counts.csv).

## Implications

- At a ratio of about 1.5×, the same content takes about 50% more tokens in Hebrew. Per-token billing is about 50% higher, and the context window is used up about 50% faster.
- No tokenizer tested goes below about 1.49×, including the Hebrew-specialized ones. Choosing a Hebrew-aware vocabulary did not remove the gap in these measurements.

## Limitations

- One dataset (FLORES+) and 50 sampled paragraphs. Only 3 of them contain notable Latin-script text, so the "mixed" subgroup is too small to compare.
- FLORES+ is news and encyclopedic text. Other domains (code, chat, legal, social media) may give different ratios.
- `gpt-4o` and `gpt-5` share a tokenizer according to `tiktoken`. We have not verified that independently, and `gpt-5` is not a separate measurement.
- Gemini models are merged because their counts match exactly. Google does not publish the tokenizer, so this is strong evidence, not proof.
- `gemini-4-argon` has a fixed overhead of 31 tokens per request, against 1 for the 3.x models. We subtract it, and its counts grow linearly, but we cannot see what it contains.
- Gemini models older than generation 3 are not included. `gemini-2.5-flash` is listed by the API but rejects token counting for new users.
- Reference tokenizers show vocabulary behavior only. They are not prices, and the ratio is not an accuracy or quality measure.
- Claude is not tested yet.

## Next steps

- **Net-savings go/no-go test:** compare translate-then-query (Hebrew to English, then query in English) against querying in Hebrew directly. The translation tokens, input and output, count toward the translate-then-query cost.
- **Claude:** add its tokenizer when API access is available.
- **Real-world pairs:** add Hebrew/English pairs from other domains than FLORES+.

## Run it

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `HF_TOKEN` and `GEMINI_API_KEY`. The FLORES+ dataset is gated, so accept its terms on Hugging Face first. Then open the notebook with the `hebrew-token-tax` kernel and run all cells. The first run downloads the reference tokenizer files from Hugging Face (a few MB each).

## References

- FLORES+ dataset: [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus), CC BY-SA 4.0.
- Shmidman, S., Shmidman, A., Cohen, A. D. N., Koppel, M. (2024). *Adapting LLMs to Hebrew: Unveiling DictaLM 2.0 with Enhanced Vocabulary and Instruction Capabilities*. [arXiv:2407.07080](https://arxiv.org/abs/2407.07080).
- Shmidman, S., Shmidman, A., Cohen, A. D. N., Koppel, M. (2026). *Dicta-LM 3.0: Advancing The Frontier of Hebrew Sovereign LLMs*. [arXiv:2602.02104](https://arxiv.org/abs/2602.02104).
- Petrov, A., La Malfa, E., Torr, P. H. S., Bibi, A. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages*. [arXiv:2305.15425](https://arxiv.org/abs/2305.15425).
