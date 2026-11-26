import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent  # .../segmentation_check
REPO_ROOT = HERE.parent  # 1 level up = the repo root
CASES_PATH = REPO_ROOT / "feasibility_test" / "data" / "03" / "cases_he.json"
RESULTS = HERE / "results"

PREFIX_LETTERS = set("והשבלכמ")
PUNCTUATION = ".,?!:;()\"'"


def build_pool(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for case in cases:
        for segment in ("history", "doc"):
            text = case[segment]
            sentences = re.split(r"(?<=[.?!])\s+|\n", text)
            for sentence in sentences:
                sentence = re.sub(
                    r"^(לקוח|נציג):\s*", "", sentence
                )  # strip the speaker label
                for raw in sentence.split():
                    word = raw.strip(
                        PUNCTUATION
                    )  # clean the raw word before processing
                    if len(word) > 1 and word[0] in PREFIX_LETTERS:
                        rows.append(
                            {
                                "word": word,
                                "sentence": sentence,
                                "first_letter": word[0],
                                "domain": case["domain"],
                                "case_id": case["id"],
                                "segment": segment,
                            }
                        )

    return pd.DataFrame(rows).drop_duplicates(subset=["word", "sentence"])


def sample_pool(
    pool: pd.DataFrame, num_words: int = 100, random_state: int = 53
) -> pd.DataFrame:
    # keep at most 2 random sentences per word
    capped = pool.sample(frac=1, random_state=random_state).groupby("word").head(2)

    if len(capped) < num_words:
        raise ValueError(
            f"Pool has only {len(capped)} words, but {num_words} were requested."
        )

    return capped.sample(n=num_words, random_state=random_state).reset_index(drop=True)


if __name__ == "__main__":
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    pool = build_pool(cases)
    sample = sample_pool(pool)
    print(sample["domain"].value_counts().to_string())
    print(sample["segment"].value_counts().to_string())
    print(sample["first_letter"].value_counts().to_string())
    RESULTS.mkdir(exist_ok=True)
    sample.to_csv(RESULTS / "sample.csv", index=False, encoding="utf-8-sig")
