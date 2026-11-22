# Test 3: how it works

## 1. Summary

Test 3 checks how many tokens we can save by shortening Hebrew chatbot requests (conversation
history + a policy document) and how much answer quality we lose by doing so. It also compares the
result with existing compressors, with translating to English first, and with the same methods on
English. Every method shortens each of the 48 cases. A chatbot model then answers 3 questions from
the shortened text, and a judge model (an LLM that grades answers) marks each answer as correct or
wrong. The outputs are:

- a CSV with one row per case/method/rate/language;
- a summary CSV;
- a chart of token savings against accuracy;
- a markdown summary.

The go/no-go decision is ours. The code does not write it.

## 2. Files

| File | What it is | What it does | Used in which stage |
|---|---|---|---|
| `notebooks/03_baselines_go_no_go.ipynb` | The Test 3 notebook | Sets the config and calls the functions in `src/`. It prints and saves the results. | All |
| `notebooks/03_spec.md` | Test spec (source of truth) | Defines methods, models, prompts, metrics and decision thresholds. The notebook reads the 4 prompts from it word for word. | Config, teachers, answers, judge |
| `data/03/batch_1.json` … `batch_6.json` | Hebrew cases, 8 per file | Raw input to `scripts/validate_cases.py` | Data preparation (before Test 3) |
| `data/03/batch_1_en.json` … `batch_6_en.json` | English cases, 8 per file | Same as above, in English | Data preparation |
| `data/03/cases_he.json`, `cases_en.json` | The 48 merged cases per language, same ids and order | Each case has `id`, `domain`, `history`, `doc`, 3 `questions`, 3 `correct_answers` and `critical_facts` | Load data, and every stage after it |
| `data/03/protected_he.txt`, `protected_en.txt` | Words that must never be dropped (negations, conditions, …) | Used by the rules method and by `build_filler_lists.py` | Local methods |
| `data/03/filler_he.txt`, `filler_en.txt` | Words the rules method may drop | spaCy stop words minus protected words, number words, anything with a digit and (English) n't forms | Local methods |
| `src/hebrew_token_tax/test3.py` | The pipeline | Loads prompts and cases, and builds the method grid. Runs each stage and stores its results. Computes the metrics, the chart and `summary.md`. | All |
| `src/hebrew_token_tax/compressors.py` | Local tools | The rules method, LLMLingua-2 (a small model that deletes the least useful tokens) and NLLB (a translation model). Also the teacher validator (checks the output only deletes words; attached punctuation, English letter case and Hebrew prefix letters moved from just-deleted words are allowed, see the spec) and timing. | Local methods, teachers |
| `src/hebrew_token_tax/llm.py` | API layer | One call path for OpenAI, Anthropic, Gemini and a fake provider. It covers caching, retries, max 4 calls at a time (Anthropic 2, streamed), cost tracking, the spend guard and billing errors. | Smoke test, teachers, answers, judge, spend |
| `src/hebrew_token_tax/wordlists.py` | Word-list helpers | Reads the word lists. Implements the Hebrew prefix rule (strip ו ה ש ב ל כ מ, with exceptions בעד, מבעד). | Local methods, `build_filler_lists.py` |
| `scripts/validate_cases.py` | Data check and merge | Checks every batch and compares Hebrew with English. If nothing is blocking, it writes `cases_he.json` and `cases_en.json`. | Data preparation |
| `scripts/build_filler_lists.py` | Word-list builder | Writes `filler_he.txt` and `filler_en.txt` from spaCy stop words | Data preparation |
| `scripts/check_compressors.py` | Check script | Runs the validator unit checks; without `--validator-only` it also runs every local method on 1 case per language. It downloads the models. | Before the local stage |
| `scripts/check_llm_fake.py` | Check script | Tests `llm.py` and the pipeline with the fake provider: no network, no keys, no cost. Also tests resume, billing errors and every fail-fast stop. | Before any paid run |
| `.env` / `.env.example` | API keys (`.env` is gitignored) | Names: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `HF_TOKEN`. The notebook checks only the names its providers need. It does not check `HF_TOKEN`, and the Hugging Face models are public. | Config / environment |
| `data/cache/03/` | Cache (gitignored, created on first run) | `llm/<role>_<hash>.json` holds one file per API reply, with its usage and cost. `local.json` holds the local-method outputs. `real/`, `gemini/` and `fake/` each hold `teachers.json`, `answers.json` and `judgements.json`, one folder per run mode. | Every stage |
| `results/03_results{SUFFIX}.csv` | Output | One row per case/method/rate/language, counts and scores only (no text) | Metrics |
| `results/03_summary{SUFFIX}.csv` | Output | Means, standard errors, per-question accuracy and failure counts per method/rate/language | Results (a) |
| `results/03_question_types{SUFFIX}.csv` | Output | Accuracy per question type (Q1 numbers/dates, Q2 negations/conditions, Q3 from history) | Results (b) |
| `results/03_savings_vs_accuracy{SUFFIX}.png`, `03_accuracy_bars{SUFFIX}.png`, `03_facts_kept_bars{SUFFIX}.png` | Output | Charts with ±1 standard-error bars, one panel per language | Results (c) |
| `results/03_domain_drop{SUFFIX}.csv` | Output | Accuracy drop vs. original per domain for llmlingua2_large and both teachers | Results (d) |
| `results/03_spend{SUFFIX}.csv` | Output | Per stage and provider: calls, new/cached this session, tokens in/out, $ | Results (e) |
| `results/03_summary{SUFFIX}.md` | Output | Token ratio, all tables and the 3 charts; no verdict | Results (f) |

`SUFFIX` is empty only for the real full run (`PROVIDER_OVERRIDE = None`, `DRY_RUN_N = None`).
Other runs add a tag such as `_real_n3` or `_fake_n3`, so they never overwrite the real results.

## 3. The process (run the notebook top to bottom, from the `notebooks/` folder)

| # | Step | What happens | Where | Input | Output | Cost |
|---|---|---|---|---|---|---|
| 1 | Prompts | Reads the 4 prompts from the spec and prints them | cell "1. Imports and prompts", `load_prompts` | `03_spec.md` | printed prompts | free |
| 2 | Config | Sets models, prices, flags and paths | cell "2. Config" | — | settings | free |
| 3 | Environment | Loads `.env` only when a real provider needs keys (never in fake mode) and checks that the needed key names exist. Creates the client with the budget caps. | cell "3. Environment and client", `resolve_roles`, `LLMClient` | `.env` | `client`, roles | free |
| 4 | Load data | Loads the cases and builds the 23 method/rate/language combos. Prints the Hebrew/English token ratio. | cell "4. Cases", `load_cases`, `method_grid`, `token_ratio` | `cases_he/en.json` | `cases`, `methods` | free |
| 5 | Smoke test | Sends "Reply with OK" to each model in use (4 calls) | cell "5", `smoke_requests` | roles | printed replies | paid (tiny) |
| 6 | Local methods | Runs rules, LLMLingua-2 large/small at 0.7/0.5, NLLB translation, and translation followed by LLMLingua-2, all on CPU. Warm-up first, so `latency_ms` excludes model loading. | cell "6", `run_local`, `compress_locally` | cases, word lists | `data/cache/03/local.json` | free |
| 7 | Teachers | Two LLMs shorten each text by deleting only, to an exact word target (round(words × keep rate)). Invalid output is retried. Valid output more than 0.10 above the keep rate is retried with a length-feedback line. There are at most 2 retries. The closest valid output is kept; a length miss is not a failure. A case is marked failed (original text kept) only if no attempt was valid. | cell "7", `run_teachers` | cases, teacher prompts | `<mode>/teachers.json` | paid |
| 8 | Chatbot answers | Answers the 3 questions from each shortened text, as JSON. Invalid JSON is retried once; if still invalid, the 3 answers count as incorrect. | cell "8", `run_answers` | compressions, chatbot prompt | `<mode>/answers.json` | paid |
| 9 | Judge | One blind call per answer set (the judge never sees the method). Grades the 3 answers against the correct answers. JSON failures count as incorrect. | cell "9", `run_judge` | answers, correct answers | `<mode>/judgements.json` | paid |
| 10 | Main chart | Per row: tokens, savings %, tokens saved, accuracy, drop vs. original, facts kept, latency, requested/actual keep rate, attempts and failure flags. Then the means. Main chart: tokens saved vs. answers correct, one line per method from keep rate 0.7 to 0.5, named on the chart; original = red star. | cell "10", `score_rows`, `summarize`, `plot_main_chart` | all stores | `03_results*.csv`, `03_summary*.csv`, `03_main_chart*.png` | free |
| 11 | (a) Summary table | Means per language × method × rate; plus requested vs. actual keep rate and attempts per teacher row | cell "11", `results_table`, `teacher_length_table` | summary, rows | `03_teacher_length*.csv` | free |
| 12 | (b) Question types | Accuracy for Q1 numbers/dates, Q2 negations/conditions and Q3 from history | cell "12", `question_type_table` | summary | `03_question_types*.csv` | free |
| 13 | (c) More charts | Savings vs. accuracy (±1 SE, numbered points), accuracy bars and facts-kept bars | cell "13", `plot_savings_vs_accuracy`, `plot_metric_bars` | summary | 3 PNGs | free |
| 14 | (d) Domains | Accuracy drop per domain for llmlingua2_large and both teachers | cell "14", `domain_drop_table` | rows | `03_domain_drop*.csv` | free |
| 15 | (e) Spend | Per stage and provider: calls, new/cached this session, tokens in/out and $. Also all-time spend against the caps, and output tokens used per role against its limit. | cell "15", `spend_by_stage`, `usage_by_role` | the LLM cache | `03_spend*.csv` | free |
| 16 | (f) Summary file | Main chart first, then the token ratio, all tables and the other charts; no verdict | cell "16", `write_summary_md` | tables, charts | `03_summary*.md` | free |

Facts are matched after cleaning both the fact and the text the same way (`normalize_for_match`):
runs of spaces collapse to one, and spaces next to punctuation or symbols are dropped.

Every API stage shows a progress bar with ETA, cache hits, failures and spend so far.

Savings are always counted with tiktoken `o200k_base` against the original text in that language.

## 4. How to run (flags in cell "2. Config")

| Flag | What it does | Dry run | Full run |
|---|---|---|---|
| `DRY_RUN_N` | Cases per language (`None` = all 48) | `3` | `None` |
| `PROVIDER_OVERRIDE` | `None` = the spec's models; `"gemini"` = free debug keys; `"fake"` = offline replies | `None` (or `"fake"` / `"gemini"` to debug) | `None` |
| `RUN_SMOKE` | Run the smoke test | `True` (once) | `False` |
| `RUN_LOCAL` | Run the local methods | `True` | `True` |
| `RUN_TEACHERS` / `RUN_ANSWERS` / `RUN_JUDGE` | Run that paid stage | `True` | `True` |
| `CONFIRM_SPEND` | Paid stages print an estimate and stop unless `True` | `False` first (read the estimate), then `True` | same |
| `BUDGET_CAPS` | Maximum all-time spend per provider: `{"anthropic": 25.0, "openai": 15.0}` | unchanged | unchanged |
| `*_MAX_OUTPUT_TOKENS` | Output limit per role, reasoning tokens included: teachers 32000, chatbot 4000, judge 8000 | unchanged | tighten if the dry run shows much lower use |

The smoke test checks that the 4 model ids work. The run plan at the top of the notebook lists
the exact flags for each step. The dry-run cases are the first 3 of each language; the full run
reuses them from the cache, so they are not paid for again.

## 5. Safety notes

- **Caching:** every API reply is saved under `data/cache/03/llm/`, keyed by provider, model,
  prompt, reasoning level and attempt. A rerun reads the cache and pays nothing twice. The output
  limit is not part of the key, so raising it keeps cached replies; only a reply that was cut off
  at a lower limit is requested again. Local outputs and stage results are saved too.
- **Spend guard:** before each paid stage, the notebook prints the calls to make, the calls already
  cached and the estimated cost per provider. It stops with `SpendBlocked` if `CONFIRM_SPEND` is
  `False` or a provider would pass its cap. During a stage, each uncached call is checked against
  the cap first (worst case: its full output limit). After the dry run, estimates use real usage.
- **Fail-fast:** these stop the stage at once, cancel the queued calls and print why. Calls
  already running still finish; their cost is printed and added to the spend. Anthropic runs at
  most 2 calls at a time (OpenAI 4) to keep that small.
  - `BillingError`: out of credit or quota (HTTP 402, OpenAI `insufficient_quota`, Anthropic
    `billing_error`, every Gemini key at its daily quota). Nothing is cached or counted as incorrect.
  - `FatalCallError`: a non-retryable error (400 bad request, 401/403 auth, 404 model not found),
    or 5 retries running out. 429, 5xx and timeouts are retried with backoff.
  - `StageAborted`: 3 failed calls in a row, or too many failures in a stage's first 10 calls
    (teachers > 30% invalid, chatbot/judge > 20% invalid JSON). Only first attempts count, so the
    spec's capped retries of one stubborn case don't stop the stage.
- **Keys:** they are read from `.env` only when a real provider is used. The notebook checks key
  names and never prints values.
- **If a stage stops midway:** read the message and fix the cause (for example, top up credit or
  correct a model id), then run the same cell again. Calls that already finished come from the
  cache, so the stage continues where it stopped, without repeats.
