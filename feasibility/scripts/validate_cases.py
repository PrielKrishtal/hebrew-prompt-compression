"""Validate the notebook-03 case batches (Hebrew + English) and merge them if clean.

Usage: python scripts/validate_cases.py
Exit code is 1 when there are BLOCKING problems (nothing is merged), else 0.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "03"
LANGUAGES = ("he", "en")
BATCH_COUNT = 6
CASES_PER_BATCH = 8
QUESTION_COUNT = 3
MIN_FACTS = 8
MAX_FACTS = 15
HISTORY_ANSWER_INDEX = 2
STRING_FIELDS = ("id", "domain", "history", "doc")
LIST_FIELDS = ("questions", "correct_answers", "critical_facts")
REQUIRED_FIELDS = STRING_FIELDS + LIST_FIELDS
NUMBER_PATTERN = re.compile(r"\d+", re.ASCII)
ENCODING_NAME = "o200k_base"


@dataclass
class Report:
    """Problems found during validation."""

    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseStats:
    """Size of one case (history + doc) in both languages."""

    case_id: str
    domain: str
    he_words: int
    en_words: int
    he_tokens: int
    en_tokens: int


def batch_path(index: int, language: str) -> Path:
    """Return the path of batch `index` in `language`."""
    suffix = "" if language == "he" else "_en"
    return DATA_DIR / f"batch_{index}{suffix}.json"


def load_batch(path: Path, report: Report) -> list:
    """Load one batch file; return [] and record a problem if it is unusable."""
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report.blocking.append(f"{path.name}: file not found")
        return []
    except json.JSONDecodeError as error:
        report.blocking.append(f"{path.name}: invalid JSON ({error})")
        return []
    if not isinstance(cases, list):
        report.blocking.append(f"{path.name}: top level is not a list")
        return []
    if len(cases) != CASES_PER_BATCH:
        report.blocking.append(
            f"{path.name}: {len(cases)} cases, expected {CASES_PER_BATCH}"
        )
    return cases


def is_string_list(value: object) -> bool:
    """Return True if `value` is a list of non-empty strings."""
    return isinstance(value, list) and all(
        isinstance(item, str) and item for item in value
    )


def check_shape(case: object, label: str, report: Report) -> bool:
    """Check fields and types; return True if the case is safe to inspect further."""
    if not isinstance(case, dict):
        report.blocking.append(f"{label}: case is not an object")
        return False
    missing = [name for name in REQUIRED_FIELDS if name not in case]
    if missing:
        report.blocking.append(f"{label}: missing fields {missing}")
        return False
    well_formed = True
    for name in STRING_FIELDS:
        if not isinstance(case[name], str) or not case[name]:
            report.blocking.append(f"{label}: '{name}' is not a non-empty string")
            well_formed = False
    for name in LIST_FIELDS:
        if not is_string_list(case[name]):
            report.blocking.append(f"{label}: '{name}' is not a list of non-empty strings")
            well_formed = False
    return well_formed


def check_counts(case: dict, label: str, report: Report) -> None:
    """Check the number of questions, answers and critical facts."""
    for name in ("questions", "correct_answers"):
        if len(case[name]) != QUESTION_COUNT:
            report.blocking.append(
                f"{label}: {len(case[name])} {name}, expected {QUESTION_COUNT}"
            )
    fact_count = len(case["critical_facts"])
    if not MIN_FACTS <= fact_count <= MAX_FACTS:
        report.blocking.append(
            f"{label}: {fact_count} critical_facts, expected {MIN_FACTS}-{MAX_FACTS}"
        )


def appears_in_text(snippet: str, case: dict) -> bool:
    """Return True if `snippet` appears verbatim in the history or the doc."""
    return snippet in case["history"] or snippet in case["doc"]


def check_grounding(case: dict, label: str, report: Report) -> None:
    """Check that answers and critical facts are copied word-for-word from the text."""
    for index, answer in enumerate(case["correct_answers"]):
        if not appears_in_text(answer, case):
            report.blocking.append(f"{label}: answer Q{index + 1} not in text: {answer!r}")
    answers = case["correct_answers"]
    if len(answers) > HISTORY_ANSWER_INDEX:
        q3_answer = answers[HISTORY_ANSWER_INDEX]
        if q3_answer not in case["history"]:
            report.blocking.append(f"{label}: answer Q3 not in history: {q3_answer!r}")
    for fact in case["critical_facts"]:
        if not appears_in_text(fact, case):
            report.blocking.append(f"{label}: critical fact not in text: {fact!r}")


def check_case(case: object, label: str, report: Report) -> bool:
    """Run every per-case check; return True if the case is well-formed."""
    if not check_shape(case, label, report):
        return False
    check_counts(case, label, report)
    check_grounding(case, label, report)
    return True


def check_unique_ids(cases: list, language: str, report: Report) -> None:
    """Record ids that appear more than once within one language."""
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str):
            continue
        if case_id in seen:
            report.blocking.append(f"[{language}] duplicate id across files: {case_id}")
        seen.add(case_id)


def case_text(case: dict) -> str:
    """Return the context text of a case (history + doc)."""
    return f"{case['history']}\n{case['doc']}"


def number_set(text: str) -> set[str]:
    """Return every digit sequence in `text`."""
    return set(NUMBER_PATTERN.findall(text))


def compare_numbers(he_case: dict, en_case: dict, report: Report) -> None:
    """Warn about numbers present in only one language."""
    he_numbers = number_set(case_text(he_case))
    en_numbers = number_set(case_text(en_case))
    case_id = he_case["id"]
    if only_he := sorted(he_numbers - en_numbers, key=int):
        report.warnings.append(f"{case_id}: numbers only in Hebrew: {only_he}")
    if only_en := sorted(en_numbers - he_numbers, key=int):
        report.warnings.append(f"{case_id}: numbers only in English: {only_en}")


def measure(he_case: dict, en_case: dict, encoding: tiktoken.Encoding) -> CaseStats:
    """Count words and tokens of one case pair."""
    he_text, en_text = case_text(he_case), case_text(en_case)
    return CaseStats(
        case_id=he_case["id"],
        domain=he_case["domain"],
        he_words=len(he_text.split()),
        en_words=len(en_text.split()),
        he_tokens=len(encoding.encode(he_text)),
        en_tokens=len(encoding.encode(en_text)),
    )


def compare_batch(
    index: int,
    he_cases: list,
    en_cases: list,
    well_formed: dict[int, bool],
    encoding: tiktoken.Encoding,
    report: Report,
) -> list[CaseStats]:
    """Compare a Hebrew batch with its English batch; return per-case stats."""
    he_ids = [case.get("id") if isinstance(case, dict) else None for case in he_cases]
    en_ids = [case.get("id") if isinstance(case, dict) else None for case in en_cases]
    if he_ids != en_ids:
        report.blocking.append(f"batch_{index}: id mismatch he={he_ids} en={en_ids}")
    stats = []
    for position, (he_case, en_case) in enumerate(zip(he_cases, en_cases, strict=False)):
        if not (well_formed.get(id(he_case)) and well_formed.get(id(en_case))):
            continue
        if he_case["domain"] != en_case["domain"]:
            report.blocking.append(
                f"batch_{index}#{position + 1}: domain mismatch "
                f"he={he_case['domain']!r} en={en_case['domain']!r}"
            )
        compare_numbers(he_case, en_case, report)
        stats.append(measure(he_case, en_case, encoding))
    return stats


def format_ratio(he_tokens: int, en_tokens: int) -> str:
    """Return the Hebrew/English token ratio, or 'n/a' when undefined."""
    return f"{he_tokens / en_tokens:.2f}" if en_tokens else "n/a"


def print_table(title: str, stats: list[CaseStats]) -> None:
    """Print one row per case with words, tokens and the token ratio."""
    header = ("id", "domain", "he_words", "en_words", "he_tokens", "en_tokens", "ratio")
    rows = [
        (
            row.case_id,
            row.domain,
            str(row.he_words),
            str(row.en_words),
            str(row.he_tokens),
            str(row.en_tokens),
            format_ratio(row.he_tokens, row.en_tokens),
        )
        for row in stats
    ]
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    print(f"\n{title}")
    for cells in (header, *rows):
        print("  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)))
    if not rows:
        print("  (no comparable cases)")


def print_problems(report: Report) -> None:
    """Print BLOCKING problems and WARNINGS."""
    for heading, problems in (("BLOCKING", report.blocking), ("WARNINGS", report.warnings)):
        print(f"\n{heading} ({len(problems)})")
        for problem in problems:
            print(f"  - {problem}")


def write_cases(path: Path, cases: list) -> None:
    """Write merged cases as UTF-8 JSON."""
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Validate all batches, print the report, and merge when nothing is blocking."""
    sys.stdout.reconfigure(encoding="utf-8")
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    report = Report()
    merged: dict[str, list] = {language: [] for language in LANGUAGES}
    well_formed: dict[int, bool] = {}
    all_stats: list[CaseStats] = []

    for index in range(1, BATCH_COUNT + 1):
        batches = {}
        for language in LANGUAGES:
            path = batch_path(index, language)
            cases = load_batch(path, report)
            for position, case in enumerate(cases):
                case_id = case.get("id", "?") if isinstance(case, dict) else "?"
                label = f"{path.name}#{position + 1} ({case_id})"
                well_formed[id(case)] = check_case(case, label, report)
            batches[language] = cases
            merged[language].extend(cases)
        stats = compare_batch(
            index, batches["he"], batches["en"], well_formed, encoding, report
        )
        print_table(f"batch_{index}", stats)
        all_stats.extend(stats)

    for language in LANGUAGES:
        check_unique_ids(merged[language], language, report)

    he_total = sum(row.he_tokens for row in all_stats)
    en_total = sum(row.en_tokens for row in all_stats)
    print(
        f"\nOverall ({len(all_stats)} cases, {ENCODING_NAME}): "
        f"he={he_total} en={en_total} ratio={format_ratio(he_total, en_total)}"
    )
    print_problems(report)

    if report.blocking:
        print("\nNot merged: fix BLOCKING problems first.")
        return 1
    for language in LANGUAGES:
        output = DATA_DIR / f"cases_{language}.json"
        write_cases(output, merged[language])
        print(f"\nMerged {len(merged[language])} cases -> {output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
