"""Stage-1 check: validator unit checks, then every local method on 1 Hebrew and 1 English case.

Usage:
    python scripts/check_compressors.py --validator-only   (no models)
    python scripts/check_compressors.py                     (downloads/loads LLMLingua-2 and NLLB)
"""

import argparse
import sys
import tempfile
from pathlib import Path

from hebrew_token_tax.compressors import (
    LLMLINGUA2_LARGE,
    LLMLINGUA2_SMALL,
    LLMLingua2Compressor,
    NllbTranslator,
    RuleCompressor,
    is_valid_deletion,
)
from hebrew_token_tax.test3 import (
    TEACHER_METHODS,
    JsonStore,
    LocalCompressors,
    compress_locally,
    context,
    count_tokens,
    load_cases,
    method_grid,
    record_key,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "03"
KEEP_RATES = (0.7, 0.5)
HE_ORIGINAL = "ולא קיבלתי את הרכב, והבית שלי רחוק."
EN_ORIGINAL = "I didn't get the car, and my house is far."
VALIDATOR_CASES = (
    ("he", HE_ORIGINAL, "קיבלתי הרכב, בית רחוק.", True, "deletions + stacked prefix וה"),
    ("he", HE_ORIGINAL, "קיבלתי הרכב בית רחוק", True, "attached punctuation dropped"),
    ("he", HE_ORIGINAL, "לא קיבלתי רכב", True, "single prefixes stripped"),
    ("he", HE_ORIGINAL, "הרכב, קיבלתי", False, "reordered"),
    ("he", HE_ORIGINAL, "קיבלתי את הרכב החדש", False, "word added"),
    ("he", HE_ORIGINAL, "קיבלנו את הרכב", False, "word changed"),
    ("he", HE_ORIGINAL, "קיבלתי, הרכב.", True, "A: punctuation added/changed"),
    ("he", HE_ORIGINAL, "I did not get the car", False, "translated"),
    ("he", HE_ORIGINAL, "   ", False, "empty"),
    ("he", HE_ORIGINAL, " . , ", False, "punctuation only"),
    ("he", "שלום – עולם", "שלום – עולם", True, "punctuation-only token kept"),
    ("he", "שלום – עולם", "שלום , עולם", True, "A: punctuation-only token ignored"),
    ("he", "ובכלל לא חשבתי להתקשר", "ולא חשבתי להתקשר", True, "C: prefix moved from deleted word"),
    ("he", "לחשבון של צד שלישי", "לצד שלישי", True, "C: prefix from the deleted run"),
    ("he", "ועוד קיבלתי הודעה", "וקיבלתי הודעה", True, "C: ו moved onto next word"),
    ("he", "הוא בכלל לא בא", "הוא ולא בא", False, "C: prefix not in the deleted run"),
    ("he", "ועוד קיבלתי הודעה", "ועוד וקיבלתי הודעה", False, "C: prefix from a kept word"),
    ("he", "ועוד קיבלתי הודעה", "תקיבלתי הודעה", False, "C: non-prefix letter added"),
    ("en", EN_ORIGINAL, "didn't get car far.", True, "deletions"),
    ("en", EN_ORIGINAL, "I didn't get the car and my house is far", True, "punctuation dropped"),
    ("en", EN_ORIGINAL, "I Didn't get the car. My house is far.", True, "B: case-insensitive"),
    ("en", EN_ORIGINAL, "house my far", False, "reordered"),
    ("en", EN_ORIGINAL, "I nd my house", False, "no prefix stripping in English"),
    ("en", EN_ORIGINAL, "I didn't get the new car", False, "word added"),
)


def check_validator() -> bool:
    """Run the validator cases; return True if all pass."""
    ok = True
    for language, original, output, expected, label in VALIDATOR_CASES:
        actual = is_valid_deletion(original, output, language)
        ok &= actual == expected
        print(f"{'PASS' if actual == expected else 'FAIL'} [{language}] {label}: {actual}")
    return ok


def check_methods() -> bool:
    """Run every local method on the first case of each language; rules must be deletions."""
    tools = LocalCompressors(
        rules={lang: RuleCompressor(lang, DATA_DIR) for lang in ("he", "en")},
        llmlingua2_large=LLMLingua2Compressor(LLMLINGUA2_LARGE),
        llmlingua2_small=LLMLingua2Compressor(LLMLINGUA2_SMALL),
        translator=NllbTranslator(),
    )
    cases = {lang: load_cases(DATA_DIR, lang, limit=1) for lang in ("he", "en")}
    print("\nloading models and warming up ...", flush=True)
    tools.warm_up(context(cases["he"][0]))
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonStore(Path(tmp) / "compressions.json")
        print(f"\n{'method':34} {'orig':>6} {'comp':>6} {'saved%':>7} {'ms':>8}")
        for method in method_grid(KEEP_RATES):
            if method.name in TEACHER_METHODS:
                continue
            case = cases[method.language][0]
            record = compress_locally(method, case, tools, store)
            store.set(record_key(case, method), record)
            original_tokens = count_tokens(context(case))
            tokens = count_tokens(record["text"])
            label = f"{method.language} {method.name} {method.rate or ''}"
            print(
                f"{label:34} {original_tokens:6} {tokens:6} "
                f"{100 * (1 - tokens / original_tokens):7.1f} {record['latency_ms']:8.0f}"
            )
            if method.name == "rules":
                valid = is_valid_deletion(context(case), record["text"], method.language)
                ok &= valid
                print(f"  rules output is a valid deletion: {valid}")
    return ok


def main() -> int:
    """Run the checks; exit code 1 if any fails."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator-only", action="store_true")
    args = parser.parse_args()
    ok = check_validator()
    if not args.validator_only:
        ok &= check_methods()
    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
