"""Build the notebook-03 filler lists: spaCy stop words minus protected words.

Usage: python scripts/build_filler_lists.py
Writes data/03/filler_he.txt and data/03/filler_en.txt.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.he.stop_words import STOP_WORDS as HE_STOP_WORDS

from hebrew_token_tax.wordlists import has_digit, is_prefixed_protected, load_protected

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "03"
CURLY_APOSTROPHES = str.maketrans({"‘": "'", "’": "'"})
EN_SPLIT_NEGATIONS = frozenset({
    "didn", "isn", "wasn", "doesn", "hasn", "haven", "weren", "won", "wouldn",
    "couldn", "shouldn", "mustn", "needn", "aren", "hadn", "ain", "ca",
})
EN_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "first", "second", "third", "once", "twice",
})
HE_NUMBER_WORDS = frozenset({
    "אפס", "אחד", "אחת", "שניים", "שתיים", "שני", "שתי", "שלוש", "שלושה", "שלושת",
    "ארבע", "ארבעה", "חמש", "חמישה", "שש", "שישה", "שבע", "שבעה", "שמונה", "תשע",
    "תשעה", "עשר", "עשרה", "עשרים", "שלושים", "ארבעים", "חמישים", "מאה", "מאות",
    "אלף", "אלפים", "ראשון", "ראשונה", "שנייה", "פעם", "פעמיים",
})

Rule = tuple[str, Callable[[str], bool]]


def is_contraction(word: str) -> bool:
    """Return True for n't contractions, including curly-apostrophe forms."""
    return word.translate(CURLY_APOSTROPHES).endswith("n't")


def hebrew_rules(protected_path: Path) -> list[Rule]:
    """Return the ordered removal rules for Hebrew."""
    protected = load_protected(protected_path, "he")
    return [
        ("protected", lambda word: word in protected),
        ("prefix(es) + protected", lambda word: is_prefixed_protected(word, protected)),
        ("number word", lambda word: word in HE_NUMBER_WORDS),
        ("contains a digit", has_digit),
    ]


def english_rules(protected_path: Path) -> list[Rule]:
    """Return the ordered removal rules for English."""
    protected = load_protected(protected_path, "en")
    return [
        ("protected", lambda word: word in protected),
        ("n't contraction", is_contraction),
        ("split n't form", lambda word: word in EN_SPLIT_NEGATIONS),
        ("number word", lambda word: word in EN_NUMBER_WORDS),
        ("contains a digit", has_digit),
    ]


def apply_rules(
    stop_words: frozenset[str] | set[str], rules: list[Rule]
) -> tuple[list[str], dict[str, list[str]]]:
    """Return (kept words, removed words per rule); each word goes to its first matching rule."""
    removed: dict[str, list[str]] = {name: [] for name, _ in rules}
    kept = []
    for word in sorted(stop_words):
        rule_name = next((name for name, matches in rules if matches(word)), None)
        if rule_name is None:
            kept.append(word)
        else:
            removed[rule_name].append(word)
    return kept, removed


def write_list(path: Path, words: list[str]) -> None:
    """Write one word per line under a spaCy version comment."""
    lines = [f"# spacy {spacy.__version__}", *words]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(language: str, kept: list[str], removed: dict[str, list[str]], total: int) -> None:
    """Print the final list and the words removed by each rule."""
    print(f"\n=== {language}: {total} spaCy stop words -> {len(kept)} filler words ===")
    print(" ".join(kept))
    for rule_name, words in removed.items():
        print(f"\nremoved by '{rule_name}' ({len(words)}):")
        print(" ".join(words) if words else "(none)")


def build(language: str, stop_words: frozenset[str] | set[str], rules: list[Rule]) -> None:
    """Filter one language's stop words, save the result, and report it."""
    kept, removed = apply_rules(stop_words, rules)
    write_list(DATA_DIR / f"filler_{language}.txt", kept)
    report(language, kept, removed, len(stop_words))


def main() -> None:
    """Build both filler lists."""
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"spacy {spacy.__version__}")
    build("he", HE_STOP_WORDS, hebrew_rules(DATA_DIR / "protected_he.txt"))
    build("en", EN_STOP_WORDS, english_rules(DATA_DIR / "protected_en.txt"))


if __name__ == "__main__":
    main()
