import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from sample_words import PUNCTUATION, build_pool

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
CASES_PATH = REPO_ROOT / "feasibility_test" / "data" / "03" / "cases_he.json"
TRAPS_PATH = HERE / "traps.txt"
RESULTS = HERE / "results"
SEED = 53

MODELS = {
    "base": "dicta-il/dictabert-joint",
    "tiny": "dicta-il/dictabert-tiny-joint",
}
MAX_SENTENCES_PER_TRAP = 2


def find_trap_rows(pool: pd.DataFrame, traps: set[str]) -> pd.DataFrame:
    """Find real sentences for the trap words (words we want to test on purpose)"""
    shuffled = pool.sample(frac=1, random_state=SEED)
    trap_rows = (
        shuffled[shuffled["word"].isin(traps)]
        .groupby("word")
        .head(MAX_SENTENCES_PER_TRAP)
    )
    missing = traps - set(trap_rows["word"])
    if missing:
        print(f"Traps not found in the cases: {', '.join(sorted(missing))}")
    return trap_rows


def load_model(name: str) -> tuple:
    """Load one DictaBERT-joint model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, trust_remote_code=True)
    model.eval()  # inference mode: turns off training-only behaviour
    return tokenizer, model


def segment_word(tokenizer, model, sentence: str, word: str) -> str:
    """Return how the model splits `word` inside `sentence`"""
    try:
        with torch.no_grad():  # prediction only: no training math, faster and lighter
            result = model.predict([sentence], tokenizer, output_style="json")
    except Exception as e:  # noqa: BLE001 - any failure on one sentence: mark ERROR, keep going
        print(f"Model failed on: {sentence[:40]}... ({e})")
        return "ERROR"

    for token in result[0]["tokens"]:  # result[0] = our one sentence
        if (
            token["token"].strip(PUNCTUATION) == word
        ):  # find only the row of the word we sent
            return " + ".join(token["seg"])
    return "NOT_FOUND"


def add_splits(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Run every model on every row and compare their splits.
    Adds one column per model ( base_split, tiny_split) and an agree column.
    """
    df = df.copy()  # don't change the caller's table

    for name, (tokenizer, model) in models.items():
        print(f"Running {name} model on {len(df)} rows...")
        df[f"{name}_split"] = df.apply(
            lambda row, tok=tokenizer, mdl=model: segment_word(
                tok, mdl, row["sentence"], row["word"]
            ),
            axis=1,  # call once per row
        )

    df["agree"] = df["base_split"] == df["tiny_split"]

    return df


if __name__ == "__main__":
    # Load the 100 sampled rows
    sample = pd.read_csv(RESULTS / "sample.csv", encoding="utf-8-sig")

    # Load the cases and the trap words
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    with open(TRAPS_PATH, encoding="utf-8") as f:
        traps = {line.strip() for line in f if line.strip()}

    pool = build_pool(cases)
    trap_rows = find_trap_rows(pool, traps)

    # Combine sample + trap rows(without duplicates and with extra col is_trap:bool)
    review = pd.concat([sample, trap_rows], ignore_index=True)
    review["is_trap"] = review["word"].isin(traps)
    review = review.drop_duplicates(subset=["word", "sentence"]).reset_index(drop=True)

    # Load both models once
    models = {name: load_model(path) for name, path in MODELS.items()}

    # Run both models on every row
    review = add_splits(review, models)

    # Empty columns for manual labeling
    review["correct"] = ""
    review["correct_split"] = ""
    review["note"] = ""

    # Save
    review.to_csv(RESULTS / "review.csv", index=False, encoding="utf-8-sig")

    # Summary
    print(f"Total rows:    {len(review)}")
    print(f"Trap rows:     {review['is_trap'].sum()}")
    print(f"Disagreements: {(~review['agree']).sum()}")
    print(
        f"Not found/errors: {review[['base_split', 'tiny_split']].isin(['NOT_FOUND', 'ERROR']).any(axis=1).sum()}"
    )
