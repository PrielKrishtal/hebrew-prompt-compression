"""Protected and filler word lists (data/03) and the Hebrew prefix rule shared by the
filler-list builder and the rules compressor."""

import re
from pathlib import Path

HE_PREFIXES = frozenset("והשבלכמ")
HE_PREFIX_RULE_EXCEPTIONS = frozenset({"בעד", "מבעד"})
MIN_PART_LENGTH = 2
ENTRY_SEPARATOR = re.compile(r"[\s\-]+")


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


def is_prefixed_protected(word: str, protected: set[str]) -> bool:
    """True if stripping leading Hebrew prefixes from `word` leaves a protected word."""
    if word in HE_PREFIX_RULE_EXCEPTIONS:
        return False
    return any(rest in protected for rest in prefix_remainders(word))


def has_digit(word: str) -> bool:
    """Return True if `word` contains any digit."""
    return any(char.isdigit() for char in word)


def load_protected(path: Path, language: str) -> set[str]:
    """Return the protected words: single-word entries (Hebrew) or lowercased entries (English)."""
    entries = read_entries(path)
    if language == "he":
        return single_word_entries(entries)
    return {entry.lower() for entry in entries}


def load_filler(path: Path) -> set[str]:
    """Return the filler words of a built filler list."""
    return set(read_entries(path))
