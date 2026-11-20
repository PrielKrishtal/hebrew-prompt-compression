# Test 3 spec: final feasibility test (go / no-go)
Status: fixed before any run. Changes after the run must be reported.

## Question
How many tokens can be saved when shortening Hebrew chatbot requests, relative
to answer quality, and how does that compare with existing compressors,
translation, and the same methods on English?

## Data
data/03/cases_he.json, data/03/cases_en.json: 48 cases each, same ids/order.
6 domains × 8. Each case: history + doc, 3 questions, 3 correct_answers,
critical_facts. English = natural adaptation of the Hebrew (same facts).
Compressed text = history + doc. Questions are never compressed.

## Methods (keep rates 0.7 and 0.5 where marked *)
Hebrew:  original | rules | llmlingua2_large* | llmlingua2_small* |
         translate_only (NLLB he→en) | translate+llmlingua2_large* |
         teacher_sol* | teacher_opus*
English: original | rules | llmlingua2_large* | llmlingua2_small* |
         teacher_sol* | teacher_opus*

## Tools
- LLMLingua-2 large: microsoft/llmlingua-2-xlm-roberta-large-meetingbank
- LLMLingua-2 small: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
  (rate = keep fraction; savings are always measured with our own counter)
- NLLB: facebook/nllb-200-distilled-600M, heb_Hebr→eng_Latn, sentence by sentence
- All local models run on CPU.

## Rules method
Drop words found in data/03/filler_<lang>.txt. Never drop: words in
data/03/protected_<lang>.txt (Hebrew: also after stripping leading prefixes
ו ה ש ב ל כ מ, except listed exceptions), or anything containing a digit.

## Models and roles
| Role | Model | Reasoning | Why |
|---|---|---|---|
| Teacher 1 | GPT-6.1 Sol | high | Strong teacher at mid price |
| Teacher 2 | Claude Opus 5.5 | high | Strongest affordable teacher; defines the ceiling |
| Chatbot | GPT-6.1 Terra | low / default | Realistic cost-conscious production bot |
| Judge | Claude Sonnet 5.5 | medium | Simple grading task; different family than chatbot |
| Debug only | Gemini (free keys) | any | Pipeline debugging at $0 |
Same chatbot, settings and prompt for every method; lowest-randomness
setting each model supports. Exact API model ids are set in the notebook config.

## Budget
Cap $50 ($30 Anthropic + $20 OpenAI prepaid). Estimate one full run ≈ $32.
Run the 3-case paid dry run first and extrapolate before the full run.

## Prompts (use verbatim; {x} = filled by code)

### Teacher (Hebrew)
Shorten the following Hebrew text to about {pct}% of its length by DELETING
only. You may delete whole words, or one or more leading prefix letters
(ו ה ש ב ל כ מ, e.g. וה) from the start of a word. Do not add, reorder, rewrite or translate anything.
Never delete numbers, dates, names, negations or conditions.
Return only the shortened text.

### Teacher (English)
Shorten the following English text to about {pct}% of its length by DELETING
whole words only. Do not add, reorder, rewrite or translate anything.
Never delete numbers, dates, names, negations or conditions.
Return only the shortened text.

Validator: output must equal the original with only whole words (and, in
Hebrew, leading prefixes) removed. A kept word may drop attached punctuation.
If invalid: retry up to 2 times, then mark the case as failed and report the
failure count.

### Chatbot
You are a customer-service assistant. Answer the 3 questions using ONLY the
text below. Answer briefly, in {answer_language}. If the text does not contain
the answer, say so.
Return JSON only: {"answers": ["...", "...", "..."]}
Text:
{context}
Questions:
1. {q1}
2. {q2}
3. {q3}
(answer_language = Hebrew for all Hebrew methods, including translation ones;
English for English methods)

### Judge (blind: never sees the method)
For each numbered item, is the given answer correct, meaning it states the same
facts as the correct answer, even if worded differently?
1. Question: {q1}
   Correct answer: {a1}
   Given answer: {g1}
2. Question: {q2}
   Correct answer: {a2}
   Given answer: {g2}
3. Question: {q3}
   Correct answer: {a3}
   Given answer: {g3}
Return JSON only: {"correct": [true/false, true/false, true/false]}

## Metrics (per case, method, rate, language)
- savings = 1 − tokens(compressed) / tokens(original), tiktoken o200k_base,
  always vs. the original in that language
- accuracy = % of answers judged correct; drop = accuracy(original) − accuracy(method)
- facts_kept = % of critical_facts still present (exact substring);
  translation methods: digit-containing facts only
- latency_ms = time of the compression step only, CPU (not for teachers)
- teacher_failures = cases where the validator failed after retries
Also report: absolute tokens saved per case, and the Hebrew/English token
ratio of the originals.

## Run order
1. Smoke test (1 call per provider)  2. 3 cases on free Gemini (debug)
3. 3 cases on paid models (check cost)  4. Local methods, all cases
5. Teachers  6. Chatbot answers  7. Judge  8. Tables + chart

## Decision thresholds (fixed before results)
- Go (Hebrew): teacher beats LLMLingua-2 large by ≥10 points of savings
  at an accuracy drop ≤5 points.
- Rethink: teacher within 10 points of LLMLingua-2, OR translate+LLMLingua-2
  keeps ≥95% of digit facts, ≤100 ms latency, and saves more at a drop ≤5 points.
- Switch to English: only if the English (teacher − LLMLingua-2) gap is
  ≥10 points larger than the Hebrew gap.

## Known limitations
Synthetic data from one model family (Claude), which also provides Teacher 2
(Opus) and the judge (Sonnet). English is adapted from Hebrew, not native.
Latency is measured on a laptop CPU (upper bound). 48 cases: about ±7 points
of noise on accuracy.