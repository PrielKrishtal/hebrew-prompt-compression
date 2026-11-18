"""Load FLORES+ Hebrew/English sentence pairs and build article paragraphs."""

import pandas as pd
from datasets import load_dataset

DATASET = "openlanguagedata/flores_plus"
SPLIT = "dev"
HEBREW_SUBSET = "heb_Hebr"
ENGLISH_SUBSET = "eng_Latn"
LATIN_PATTERN = r"[A-Za-z]"
HEBREW_PATTERN = r"[֐-׿]"


def load_flores(subset: str) -> pd.DataFrame:
    """Load one FLORES+ language subset as a DataFrame with the columns we need."""
    ds = load_dataset(DATASET, subset, split=SPLIT)
    return ds.to_pandas()[["id", "text", "url", "topic"]]


def load_aligned_pairs() -> pd.DataFrame:
    """Return Hebrew/English sentence pairs, checking they are aligned and non-empty."""
    he = load_flores(HEBREW_SUBSET)
    en = load_flores(ENGLISH_SUBSET)

    assert len(he) == len(en), f"Row count mismatch: {len(he)} vs {len(en)}"
    assert set(he["id"]) == set(en["id"]), "Sentence IDs differ between languages"

    pairs = he.merge(en, on="id", suffixes=("_he", "_en"), validate="one_to_one")

    assert (pairs["url_he"] == pairs["url_en"]).all(), "Source URLs don't match"
    for col in ("text_he", "text_en"):
        assert not pairs[col].isna().any(), f"Missing values in {col}"
        assert not pairs[col].str.strip().eq("").any(), f"Empty strings in {col}"

    return pairs.sort_values("id")


def build_paragraphs(pairs: pd.DataFrame, n_paragraphs: int, seed: int) -> pd.DataFrame:
    """Join each source article's sentences into one paragraph and sample `n_paragraphs`."""
    paragraphs = (
        pairs.sort_values("id")
        .groupby("url_he", sort=False)
        .agg(
            first_id=("id", "min"),
            topic=("topic_he", "first"),
            n_sentences=("id", "size"),
            he=("text_he", " ".join),
            en=("text_en", " ".join),
        )
        .reset_index()
        .rename(columns={"url_he": "url"})
    )

    paragraphs = paragraphs.sample(
        n=min(n_paragraphs, len(paragraphs)), random_state=seed
    ).reset_index(drop=True)
    paragraphs["pair_id"] = "flores_" + paragraphs["first_id"].astype(str)
    paragraphs["source"] = "flores"

    latin_letters = paragraphs["he"].str.count(LATIN_PATTERN)
    hebrew_letters = paragraphs["he"].str.count(HEBREW_PATTERN)
    paragraphs["latin_share"] = latin_letters / (latin_letters + hebrew_letters)
    return paragraphs
