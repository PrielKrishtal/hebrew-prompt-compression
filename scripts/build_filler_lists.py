"""Build the notebook-03 filler lists: spaCy stop words minus protected words.

Usage: python scripts/build_filler_lists.py
Writes data/03/filler_he.txt and data/03/filler_en.txt.
"""

import re
import sys
from collections.abc import Callable
from pathlib import Path

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.he.stop_words import STOP_WORDS as HE_STOP_WORDS

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "03"
HE_PREFIXES = frozenset("והשבלכמ")
HE_PREFIX_RULE_EXCEPTIONS = frozenset({"בעד", "מבעד"})
MIN_PART_LENGTH = 2
ENTRY_SEPARATOR = re.compile(r"[\s\-]+")
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


def read_entries(path: Path) -> list[str]:
    """Return the non-empty, non-comment lines of a word list."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def single_word_entries(entries: list[str]) -> set[str]:
    """Return entries that are one word (a trailing hyphen is dropped); phrases are skipped."""
    singles = set()
    for entry in entries:
        parts = [part for part in ENTRY_SEPARATOR.split(entry) if part]
        if len(parts) == 1 and len(parts[0]) >= MIN_PART_LENGTH:
            singles.add(parts[0])
    return singles


def prefix_remainders(word: str) -> list[str]:
    """Return what is left after stripping 1, 2, ... leading Hebrew prefixes."""
    remainders = []
    for index, letter in enumerate(word[:-1]):
        if letter not in HE_PREFIXES:
            break
        remainders.append(word[index + 1:])
    return remainders


def has_digit(word: str) -> bool:
    """Return True if `word` contains any digit."""
    return any(char.isdigit() for char in word)


def is_contraction(word: str) -> bool:
    """Return True for n't contractions, including curly-apostrophe forms."""
    return word.translate(CURLY_APOSTROPHES).endswith("n't")


def hebrew_rules(protected_path: Path) -> list[Rule]:
    """Return the ordered removal rules for Hebrew."""
    protected = single_word_entries(read_entries(protected_path))
    return [
        ("protected", lambda word: word in protected),
        (
            "prefix(es) + protected",
            lambda word: word not in HE_PREFIX_RULE_EXCEPTIONS
            and any(rest in protected for rest in prefix_remainders(word)),
        ),
        ("number word", lambda word: word in HE_NUMBER_WORDS),
        ("contains a digit", has_digit),
    ]


def english_rules(protected_path: Path) -> list[Rule]:
    """Return the ordered removal rules for English."""
    protected = {entry.lower() for entry in read_entries(protected_path)}
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
