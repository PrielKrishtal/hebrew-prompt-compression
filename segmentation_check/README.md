# Segmentation check: is DictaBERT-joint good enough to define our units?

## Purpose

Our compressor deletes **morphemes** (words and Hebrew prefixes ו ה ש ב ל כ מ).
To get those units we use DictaBERT-joint's prefix segmentation.
Before building on it, we check how often it splits our customer-service text correctly,
especially the dangerous error: cutting off a letter that belongs to the word
(e.g. "לקוח" → "ל + קוח"), which would let our student delete a real letter.

## Decision rule (set before labeling)
**Error types**
- **Harmful:** a letter that belongs to the word is split off as a prefix (`לקוח` → `ל + קוח`). Our student could delete it and break the word.
- **Not harmful:** a real prefix is missed (`ההחזר` → `ההחזר`). We only lose a little compression.

**Outcome** (base model, all rows)
- **Pass:** ≥ 90% correct and ≤ 2 harmful errors → use as is.
- **Fix:** 3–7 harmful errors → add those words to a never-split list + "unsure → keep whole word".
- **Rethink:** > 7 harmful errors or < 80% correct → fine-tune or try another segmenter.

**Tiny vs. base:** prefer tiny if it's within 3 points of base and has no extra harmful errors.


## Files
| File | What it does |
|---|---|
| `sample_words.py` | Picks 100 random prefix-letter words (with their sentences) from the Test 3 Hebrew cases → writes `results/sample.csv` |
| `traps.txt` | 20 words where the first letter is part of the word (לקוח, בנק, מספר…), tested on purpose |
| `run_segmentation.py` | Adds trap sentences, runs both models on every row → writes `results/review.csv` |
| `results/sample.csv` | Output of `sample_words.py`: the 100 random words + their sentences |
| `results/review.csv` | Output of `run_segmentation.py`: sample + trap rows with both models' splits, plus empty columns we label by hand |

## How to run

```bash
python sample_words.py        # once; seed 53, so the sample is reproducible
python run_segmentation.py    # first run downloads both models (~5–10 min total on CPU)
```


## What models do we run
We run two sizes of DictaBERT-joint (base and tiny) on every sentence and compare their splits.
Where they disagree, we check by hand; it also shows whether the faster tiny model is good enough.


## How DictaBERT-joint is used here

- We only use the model's output: the `seg` field, which gives each word's
  morpheme split (e.g. "בשנת" → ["ב", "שנת"]).
- The tokenizer is passed to `predict` only because the model needs it to read
  the sentence. Its own subword split (by frequency, not grammar) is internal
  and not used by us.
- Stacked prefixes come back as one block ("ובתולדות" → ["וב", "תולדות"]).
  Every Hebrew prefix is one letter, so we split the block into letters ourselves.
- In the main project, these morphemes become the units for our student model,
  which tokenizes them itself (`is_split_into_words=True` + `word_ids()`).





## How DictaBERT-joint processes a sentence
```
"לא קיבלתי את ההחזר."
        ↓  tokenizer: cut the sentence into tokens (subword pieces)
matrix [tokens × 768]   ← each row = one token
        ↓  model: reads all rows at once, predicts each word's split
        ↓  predict(): groups tokens back into words
result = [ {"tokens": [ {"token": "לא",     "seg": ["לא"]},
                        {"token": "קיבלתי", "seg": ["קיבלתי"]},
                        {"token": "את",     "seg": ["את"]},
                        {"token": "ההחזר.", "seg": ["ה", "החזר"]} ]} ]
```

Note: in the output, "token" means a whole word (not the subword token from the matrix).
Grouping tokens into words is bookkeeping (always correct).
The split itself is learned, and that's what this check tests.

## Columns in `review.csv`

| Column | Meaning |
|---|---|
| `word` | The word being tested |
| `sentence` | Its full sentence (context) |
| `first_letter` | Possible prefix letter |
| `domain` | Customer-service domain |
| `case_id` | Source case in the data |
| `segment` | From chat history or document |
| `is_trap` | Word comes from traps.txt |
| `base_split` | Base model's split |
| `tiny_split` | Tiny model's split |
| `agree` | Both models gave the same split |
| `correct` | Ours: Y / N (base) |
| `correct_split` | Ours: right split, if N |
| `note` | Ours: error type (optional) |


## How to label

- `correct`: Y / N for the base model's split.
- `correct_split`: only if N, the right split (e.g. `ה + החזר`).
- Judge only **where the word starts**; prefixes grouped as one block (`וה + חתול`) count as correct.
- Look hardest at rows with `agree = False`: one model is wrong for sure.

## FAQ

**DictaBERT vs. DictaBERT-joint?**
DictaBERT is the raw Hebrew BERT: it only outputs vectors and can't split anything. DictaBERT-joint is DictaBERT fine-tuned on several tasks, including prefix splitting. This check uses only DictaBERT-joint. Plain DictaBERT becomes our student later.

**Why send the whole sentence if we check one word?**
The split depends on context ("מספר" = "number" or "from a book"). The model splits every word in the sentence, and we pick out ours by matching the word text.

**Why traps, if we already have a random sample?**
Most trap words never came up in the random 100. Traps make sure the riskiest case (a letter that belongs to the word) gets tested.

**Where do trap sentences come from?**
From our own cases: up to 2 real sentences per trap word. If a trap sentence is already in the sample, the duplicate is dropped. Matching is on the exact word, so "המכשיר" doesn't count as "מכשיר".