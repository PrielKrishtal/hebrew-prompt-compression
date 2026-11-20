"""Test 3 pipeline (notebooks/03_spec.md): prompts, methods, stages, metrics and reports."""

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
from pydantic import BaseModel, Field, StrictBool, StrictStr
from tqdm.auto import tqdm

from hebrew_token_tax.compressors import (
    LLMLingua2Compressor,
    NllbTranslator,
    RuleCompressor,
    is_valid_deletion,
    timed,
)
from hebrew_token_tax.llm import LLMClient, LLMRequest, LLMResult, StageMonitor
from hebrew_token_tax.wordlists import has_digit

LANGUAGES = ("he", "en")
ANSWER_LANGUAGE = {"he": "Hebrew", "en": "English"}
N_QUESTIONS = 3
PROMPT_TITLES = {
    "teacher_he": "Teacher (Hebrew)",
    "teacher_en": "Teacher (English)",
    "teacher_feedback": "Teacher length feedback",
    "chatbot": "Chatbot",
    "judge": "Judge (blind: never sees the method)",
}
CHATBOT_NOTE_MARKER = "(answer_language"
PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")
DIGITS = re.compile(r"\d+")
SPEC_METHODS: dict[str, tuple[tuple[str, bool], ...]] = {
    "he": (
        ("original", False),
        ("rules", False),
        ("llmlingua2_large", True),
        ("llmlingua2_small", True),
        ("translate_only", False),
        ("translate+llmlingua2_large", True),
        ("teacher_sol", True),
        ("teacher_opus", True),
    ),
    "en": (
        ("original", False),
        ("rules", False),
        ("llmlingua2_large", True),
        ("llmlingua2_small", True),
        ("teacher_sol", True),
        ("teacher_opus", True),
    ),
}
TEACHER_METHODS = frozenset({"teacher_sol", "teacher_opus"})
TRANSLATION_METHODS = frozenset({"translate_only", "translate+llmlingua2_large"})
TEACHER_MAX_ATTEMPTS = 3
MAIN_CHART_RED_INDEX = 3
LABEL_HEIGHT = 4.0
LABEL_CHAR_WIDTH = 1.05
LENGTH_TOLERANCE = 0.10
FEEDBACK_PREFIX = "Your previous output had"
TEACHER_MAX_FAILURE_RATE = 0.30
JSON_MAX_FAILURE_RATE = 0.20
SMOKE_PROMPT = "Reply with OK"
O200K = tiktoken.get_encoding("o200k_base")
SPACES = re.compile(r"\s+")
LANGUAGE_NAMES = {"he": "Hebrew", "en": "English"}
QUESTION_TYPES = ("Q1 numbers/dates", "Q2 negations/conditions", "Q3 from history")
DOMAIN_METHODS = ("llmlingua2_large", "teacher_sol", "teacher_opus")
RESULT_COLUMNS = {
    "cases": "cases",
    "savings_pct": "savings %",
    "tokens_saved": "tokens saved per case",
    "accuracy_pct": "accuracy %",
    "accuracy_drop": "accuracy drop vs original",
    "facts_kept_pct": "facts kept %",
    "latency_ms": "latency ms",
    "teacher_failures": "teacher failures",
    "json_failures": "JSON failures",
}


class Answers(BaseModel):
    answers: Annotated[list[StrictStr], Field(min_length=N_QUESTIONS, max_length=N_QUESTIONS)]


class Verdicts(BaseModel):
    correct: Annotated[list[StrictBool], Field(min_length=N_QUESTIONS, max_length=N_QUESTIONS)]


@dataclass(frozen=True)
class Role:
    """Which model plays a role, at which spec reasoning level, with which output-token limit
    (reasoning tokens included)."""

    provider: str
    model: str
    reasoning: str
    max_output_tokens: int


@dataclass(frozen=True)
class Method:
    """One compression method at one keep rate (None for methods without a rate)."""

    language: str
    name: str
    rate: float | None

    @property
    def key(self) -> str:
        return f"{self.language}|{self.name}|{self.rate}"


class JsonStore:
    """A dict persisted as one JSON file (stage outputs under data/cache/03/)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        exists = path is not None and path.exists()
        self.data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if exists else {}

    @classmethod
    def combined(cls, *stores: "JsonStore") -> "JsonStore":
        """A read-only, in-memory union of several stores."""
        union = cls(None)
        for store in stores:
            union.data.update(store.data)
        return union

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self) -> None:
        if self.path is None:
            raise ValueError("combined stores are read-only")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


def count_tokens(text: str) -> int:
    """o200k_base token count (the spec's counter)."""
    return len(O200K.encode(text))


def load_cases(data_dir: Path, language: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Return the merged cases of one language, optionally only the first `limit`."""
    cases = json.loads((data_dir / f"cases_{language}.json").read_text(encoding="utf-8"))
    return cases if limit is None else cases[:limit]


def context(case: dict[str, Any]) -> str:
    """The text that is compressed and given to the chatbot: history + doc."""
    return f"{case['history']}\n\n{case['doc']}"


def load_prompts(spec_path: Path) -> dict[str, str]:
    """Read each prompt verbatim from the spec: the lines after its ### title up to the first
    blank line (the chatbot's trailing answer_language note is not part of the prompt)."""
    lines = spec_path.read_text(encoding="utf-8").splitlines()
    prompts = {}
    for key, title in PROMPT_TITLES.items():
        start = lines.index(f"### {title}") + 1
        block = []
        for line in lines[start:]:
            if not line.strip() or line.startswith(CHATBOT_NOTE_MARKER):
                break
            block.append(line)
        prompts[key] = "\n".join(block)
    return prompts


def fill(template: str, **fields: str) -> str:
    """Replace exactly the template's {name} placeholders; JSON braces stay literal."""
    expected = set(PLACEHOLDER.findall(template))
    if expected != set(fields):
        raise ValueError(f"placeholders {sorted(expected)} != fields {sorted(fields)}")
    for name, value in fields.items():
        template = template.replace(f"{{{name}}}", value)
    return template


def method_grid(keep_rates: Sequence[float]) -> list[Method]:
    """Every (language, method, rate) the spec asks for, in spec order."""
    return [
        Method(language, name, rate)
        for language in LANGUAGES
        for name, rated in SPEC_METHODS[language]
        for rate in (keep_rates if rated else (None,))
    ]


def resolve_roles(
    roles: dict[str, Role], override: str | None, debug_models: dict[str, str]
) -> dict[str, Role]:
    """Route every role to `override` ("gemini" or "fake") with its debug model, if set."""
    if override is None:
        return roles
    return {
        name: Role(override, debug_models[override], role.reasoning, role.max_output_tokens)
        for name, role in roles.items()
    }


def make_request(role_name: str, role: Role, prompt: str, attempt: int = 0) -> LLMRequest:
    """A request for one role: the prompt alone, no system prompt or wrapper text."""
    return LLMRequest(
        role_name, role.provider, role.model, prompt, role.reasoning, role.max_output_tokens,
        attempt,
    )


def record_key(case: dict[str, Any], method: Method) -> str:
    return f"{case['id']}|{method.key}"


def smoke_requests(roles: dict[str, Role]) -> list[LLMRequest]:
    """One tiny request per distinct model in use."""
    distinct = {(role.provider, role.model): role for role in roles.values()}
    return [make_request("smoke", role, SMOKE_PROMPT) for _, role in sorted(distinct.items())]


@dataclass
class LocalCompressors:
    """The local tools; models load lazily on first use."""

    rules: dict[str, RuleCompressor]
    llmlingua2_large: LLMLingua2Compressor
    llmlingua2_small: LLMLingua2Compressor
    translator: NllbTranslator

    def warm_up(self, sample: str) -> None:
        """Load every model and run it once so latency excludes loading."""
        self.llmlingua2_large.compress(sample, 0.5)
        self.llmlingua2_small.compress(sample, 0.5)
        self.translator.translate(sample)


def compress_locally(
    method: Method, case: dict[str, Any], tools: LocalCompressors, store: JsonStore
) -> dict[str, Any]:
    """Run one local method on one case; returns the text and the latency in ms."""
    original = context(case)
    if method.name == "original":
        return {"text": original, "latency_ms": 0.0}
    if method.name == "rules":
        text, latency = timed(lambda: tools.rules[method.language].compress(original))
    elif method.name == "llmlingua2_large":
        text, latency = timed(lambda: tools.llmlingua2_large.compress(original, method.rate))
    elif method.name == "llmlingua2_small":
        text, latency = timed(lambda: tools.llmlingua2_small.compress(original, method.rate))
    elif method.name == "translate_only":
        text, latency = timed(lambda: tools.translator.translate(original))
    elif method.name == "translate+llmlingua2_large":
        translation = store.get(record_key(case, Method(method.language, "translate_only", None)))
        text, compress_latency = timed(
            lambda: tools.llmlingua2_large.compress(translation["text"], method.rate)
        )
        latency = translation["latency_ms"] + compress_latency
    else:
        raise ValueError(f"not a local method: {method.name}")
    return {"text": text, "latency_ms": latency}


def local_pending(
    cases: dict[str, list[dict[str, Any]]], methods: Sequence[Method], store: JsonStore
) -> int:
    """Number of local compressions not stored yet."""
    return sum(
        record_key(case, method) not in store
        for method in methods
        if method.name not in TEACHER_METHODS
        for case in cases[method.language]
    )


def run_local(
    cases: dict[str, list[dict[str, Any]]],
    methods: Sequence[Method],
    tools: LocalCompressors,
    store: JsonStore,
) -> int:
    """Compress every case with every local method not stored yet (saving after each one);
    returns how many ran."""
    jobs = [
        (case, method)
        for method in methods
        if method.name not in TEACHER_METHODS
        for case in cases[method.language]
        if record_key(case, method) not in store
    ]
    for case, method in tqdm(jobs, desc="local methods", disable=not jobs):
        record = compress_locally(method, case, tools, store)
        store.set(record_key(case, method), {**record, "teacher_failed": False, "attempts": 0})
        store.save()
    return len(jobs)


def word_count(text: str) -> int:
    """Whitespace-separated words (the unit of the teacher's word target)."""
    return len(text.split())


def teacher_request(
    roles: dict[str, Role],
    prompts: dict[str, str],
    case: dict[str, Any],
    method: Method,
    attempt: int,
    previous_words: int | None = None,
) -> LLMRequest:
    """The verbatim teacher prompt with the exact word target, a blank line and the text; on a
    length retry, a blank line and the verbatim length feedback follow."""
    n_words = word_count(context(case))
    target = round(n_words * method.rate)
    header = fill(
        prompts[f"teacher_{method.language}"],
        n_words=str(n_words),
        target_words=str(target),
        pct=str(round(method.rate * 100)),
    )
    prompt = f"{header}\n\n{context(case)}"
    if previous_words is not None:
        feedback = fill(
            prompts["teacher_feedback"], actual_words=str(previous_words), target_words=str(target)
        )
        prompt = f"{prompt}\n\n{feedback}"
    return make_request(method.name, roles[method.name], prompt, attempt)


@dataclass
class TeacherJob:
    """One teacher/case/rate: the closest valid output so far and the length-retry state."""

    case: dict[str, Any]
    method: Method
    best_text: str | None = None
    best_keep: float = math.nan
    long_words: int | None = None
    attempts: int = 0
    done: bool = False

    def keep_of(self, text: str) -> float:
        return word_count(text) / word_count(context(self.case))

    def offer(self, text: str) -> None:
        """Record a valid output: keep it if it is the closest so far; finish if on target."""
        keep = self.keep_of(text)
        rate = self.method.rate
        if self.best_text is None or abs(keep - rate) < abs(self.best_keep - rate):
            self.best_text, self.best_keep = text, keep
        if keep - rate > LENGTH_TOLERANCE:
            self.long_words = word_count(text)
        else:
            self.done = True

    def record(self) -> dict[str, Any]:
        """The stored compression; no valid output means failed (original text kept)."""
        failed = self.best_text is None
        return {
            "text": context(self.case) if failed else self.best_text,
            "latency_ms": None,
            "teacher_failed": failed,
            "attempts": self.attempts,
            "requested_keep": self.method.rate,
            "actual_keep": 1.0 if failed else self.best_keep,
            "length_miss": not failed and self.best_keep - self.method.rate > LENGTH_TOLERANCE,
        }


def pending_teacher_jobs(
    cases: dict[str, list[dict[str, Any]]], methods: Sequence[Method], store: JsonStore
) -> list[tuple[dict[str, Any], Method]]:
    return [
        (case, method)
        for method in methods
        if method.name in TEACHER_METHODS
        for case in cases[method.language]
        if not is_current_teacher_record(store.get(record_key(case, method)))
    ]


def is_current_teacher_record(record: dict[str, Any] | None) -> bool:
    """True for a record made with the length-aware teacher stage (older records are redone)."""
    return record is not None and "requested_keep" in record


def fingerprint(value: Any) -> str:
    """Short hash of a JSON-serializable value, to notice when a stage input changed."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_teachers(
    client: LLMClient,
    roles: dict[str, Role],
    prompts: dict[str, str],
    jobs: list[tuple[dict[str, Any], Method]],
    store: JsonStore,
) -> int:
    """Teacher compression with up to 2 retries (3 attempts). Invalid output is retried; valid
    output more than 0.10 above the requested keep rate is retried with the length feedback.
    The closest valid output is kept (a length miss is not a failure); a case with no valid
    output keeps the original text and is marked failed. Replies are validated as they arrive,
    so the stage stops early (3 invalid first attempts in a row, or > 30% of the first 10).
    Returns the number of failed cases."""
    monitor = StageMonitor("teachers", TEACHER_MAX_FAILURE_RATE, "invalid teacher output")
    states = [TeacherJob(case, method) for case, method in jobs]
    state_of: dict[str, TeacherJob] = {}

    def is_valid(result: LLMResult) -> bool:
        state = state_of[result.request.cache_key()]
        output = result.text.strip()
        language = state.method.language
        return not result.failed and is_valid_deletion(context(state.case), output, language)

    for attempt in range(TEACHER_MAX_ATTEMPTS):
        pending = [state for state in states if not state.done]
        if not pending:
            break
        requests = [
            teacher_request(roles, prompts, s.case, s.method, attempt, s.long_words)
            for s in pending
        ]
        state_of.update({r.cache_key(): s for r, s in zip(requests, pending, strict=True)})
        results = client.call_many(requests, is_valid, monitor, f"teachers (attempt {attempt + 1})")
        for state, result in zip(pending, results, strict=True):
            state.attempts = attempt + 1
            if is_valid(result):
                state.offer(result.text.strip())
            if state.done:
                store.set(record_key(state.case, state.method), state.record())
        store.save()
    for state in states:
        if not state.done:
            store.set(record_key(state.case, state.method), state.record())
    store.save()
    return sum(state.best_text is None for state in states)


def chatbot_request(
    role: Role, template: str, case: dict[str, Any], method: Method, text: str
) -> LLMRequest:
    """Chatbot prompt; answers are in the method's language (Hebrew for translation methods)."""
    questions = {f"q{i + 1}": question for i, question in enumerate(case["questions"])}
    prompt = fill(
        template, answer_language=ANSWER_LANGUAGE[method.language], context=text, **questions
    )
    return make_request("chatbot", role, prompt)


def answer_jobs(
    cases: dict[str, list[dict[str, Any]]],
    methods: Sequence[Method],
    compressions: JsonStore,
    answers: JsonStore,
) -> list[tuple[str, dict[str, Any], Method]]:
    """(key, case, method) for every compressed text that has no stored answers yet."""
    return [
        (key, case, method)
        for method in methods
        for case in cases[method.language]
        if (key := record_key(case, method)) in compressions
        and (answers.get(key) or {}).get("text_sha") != fingerprint(compressions.get(key)["text"])
    ]


def answer_requests(
    roles: dict[str, Role],
    prompts: dict[str, str],
    jobs: list[tuple[str, dict[str, Any], Method]],
    compressions: JsonStore,
) -> list[LLMRequest]:
    return [
        chatbot_request(roles["chatbot"], prompts["chatbot"], case, method,
                        compressions.get(key)["text"])
        for key, case, method in jobs
    ]


def run_answers(
    client: LLMClient,
    roles: dict[str, Role],
    prompts: dict[str, str],
    jobs: list[tuple[str, dict[str, Any], Method]],
    compressions: JsonStore,
    answers: JsonStore,
) -> int:
    """Ask the chatbot; invalid JSON is retried once, then stored as failed. Returns failures."""
    requests = answer_requests(roles, prompts, jobs, compressions)
    outcomes = client.call_json_many(requests, Answers, "answers", JSON_MAX_FAILURE_RATE)
    failures = 0
    for (key, _, _), (parsed, _) in zip(jobs, outcomes, strict=True):
        failures += parsed is None
        answers.set(
            key,
            {
                "answers": parsed.answers if parsed else None,
                "json_failed": parsed is None,
                "text_sha": fingerprint(compressions.get(key)["text"]),
            },
        )
    answers.save()
    return failures


def judge_request(
    role: Role, template: str, case: dict[str, Any], given: list[str]
) -> LLMRequest:
    """Judge prompt with the 3 (question, correct answer, given answer) items; no method info."""
    fields = {}
    for i in range(N_QUESTIONS):
        fields[f"q{i + 1}"] = case["questions"][i]
        fields[f"a{i + 1}"] = case["correct_answers"][i]
        fields[f"g{i + 1}"] = given[i]
    return make_request("judge", role, fill(template, **fields))


def judge_jobs(
    cases: dict[str, list[dict[str, Any]]],
    methods: Sequence[Method],
    answers: JsonStore,
    judgements: JsonStore,
) -> list[tuple[str, dict[str, Any]]]:
    """(key, case) for every stored answer set that is not judged yet."""
    return [
        (key, case)
        for method in methods
        for case in cases[method.language]
        if (key := record_key(case, method)) in answers
        and (judgements.get(key) or {}).get("answers_sha") != fingerprint(answers.get(key)["answers"])
    ]


def judge_requests(
    roles: dict[str, Role],
    prompts: dict[str, str],
    jobs: list[tuple[str, dict[str, Any]]],
    answers: JsonStore,
) -> list[LLMRequest]:
    """Judge requests for answer sets that parsed (failed answer sets need no judge call)."""
    return [
        judge_request(roles["judge"], prompts["judge"], case, answers.get(key)["answers"])
        for key, case in jobs
        if answers.get(key)["answers"] is not None
    ]


def run_judge(
    client: LLMClient,
    roles: dict[str, Role],
    prompts: dict[str, str],
    jobs: list[tuple[str, dict[str, Any]]],
    answers: JsonStore,
    judgements: JsonStore,
) -> int:
    """Judge each answer set in one call; failed answers or judge JSON count as incorrect.
    Returns the number of judge JSON failures."""
    all_wrong = [False] * N_QUESTIONS
    judged = [(key, case) for key, case in jobs if answers.get(key)["answers"] is not None]
    for key, _ in jobs:
        if answers.get(key)["answers"] is None:
            judgements.set(
                key,
                {"correct": all_wrong, "json_failed": False, "answers_sha": fingerprint(None)},
            )
    outcomes = client.call_json_many(
        judge_requests(roles, prompts, jobs, answers), Verdicts, "judge", JSON_MAX_FAILURE_RATE
    )
    failures = 0
    for (key, _), (parsed, _) in zip(judged, outcomes, strict=True):
        failures += parsed is None
        judgements.set(
            key,
            {
                "correct": parsed.correct if parsed else all_wrong,
                "json_failed": parsed is None,
                "answers_sha": fingerprint(answers.get(key)["answers"]),
            },
        )
    judgements.save()
    return failures


def is_punctuation_or_symbol(char: str) -> bool:
    """True for Unicode punctuation (P*) and symbols (S*), e.g. - ״ % ₪."""
    return bool(char) and unicodedata.category(char)[0] in "PS"


def normalize_for_match(text: str) -> str:
    """Collapse whitespace to one space and drop spaces next to punctuation or symbols, so
    "90 ש ״ח  ליום" matches "90 ש״ח ליום" and "windshield - only" matches "windshield-only".
    Facts and texts are cleaned the same way for every method."""
    collapsed = SPACES.sub(" ", text).strip()
    kept = []
    for index, char in enumerate(collapsed):
        before = collapsed[index - 1] if index else ""
        after = collapsed[index + 1] if index + 1 < len(collapsed) else ""
        if char == " " and (is_punctuation_or_symbol(before) or is_punctuation_or_symbol(after)):
            continue
        kept.append(char)
    return "".join(kept)


def facts_kept(facts: list[str], text: str, method_name: str) -> tuple[int, int]:
    """(facts checked, facts kept), after cleaning facts and text with normalize_for_match.
    Translation methods: digit-containing facts only, kept when every digit sequence of the
    fact appears in the text; other methods: the cleaned fact is a substring of the text."""
    clean_text = normalize_for_match(text)
    clean_facts = [normalize_for_match(fact) for fact in facts]
    if method_name in TRANSLATION_METHODS:
        text_numbers = set(DIGITS.findall(clean_text))
        checked = [fact for fact in clean_facts if has_digit(fact)]
        return len(checked), sum(set(DIGITS.findall(f)) <= text_numbers for f in checked)
    return len(clean_facts), sum(fact in clean_text for fact in clean_facts)


def score_row(
    case: dict[str, Any],
    method: Method,
    compression: dict[str, Any],
    answer: dict[str, Any] | None,
    judgement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Counts and scores for one case/method/rate/language (no text)."""
    tokens_original = count_tokens(context(case))
    tokens_compressed = count_tokens(compression["text"])
    facts_total, facts_ok = facts_kept(case["critical_facts"], compression["text"], method.name)
    correct = judgement["correct"] if judgement else [math.nan] * N_QUESTIONS
    n_correct = sum(correct) if judgement else math.nan
    return {
        "language": method.language,
        "case_id": case["id"],
        "domain": case["domain"],
        "method": method.name,
        "rate": method.rate,
        "tokens_original": tokens_original,
        "tokens_compressed": tokens_compressed,
        "tokens_saved": tokens_original - tokens_compressed,
        "savings_pct": 100 * (1 - tokens_compressed / tokens_original),
        "latency_ms": compression["latency_ms"],
        "requested_keep": method.rate,
        "actual_keep": word_count(compression["text"]) / word_count(context(case)),
        "attempts": compression.get("attempts", 0),
        "length_miss": int(compression.get("length_miss", False)),
        "facts_total": facts_total,
        "facts_kept": facts_ok,
        "facts_kept_pct": 100 * facts_ok / facts_total if facts_total else math.nan,
        "teacher_failed": int(compression["teacher_failed"]),
        "answer_json_failed": int(answer["json_failed"]) if answer else math.nan,
        "judge_json_failed": int(judgement["json_failed"]) if judgement else math.nan,
        **{f"q{i + 1}_correct": float(value) for i, value in enumerate(correct)},
        "n_correct": n_correct,
        "accuracy_pct": 100 * n_correct / N_QUESTIONS,
    }


def score_rows(
    cases: dict[str, list[dict[str, Any]]],
    methods: Sequence[Method],
    compressions: JsonStore,
    answers: JsonStore,
    judgements: JsonStore,
) -> pd.DataFrame:
    """One row per case/method/rate/language that has a compression, plus the accuracy drop
    against the original of the same case."""
    rows = [
        score_row(case, method, compressions.get(key), answers.get(key), judgements.get(key))
        for method in methods
        for case in cases[method.language]
        if (key := record_key(case, method)) in compressions
    ]
    frame = pd.DataFrame(rows)
    keys = ["language", "case_id"]
    baseline = frame.loc[frame["method"] == "original", [*keys, "accuracy_pct"]].rename(
        columns={"accuracy_pct": "original_accuracy_pct"}
    )
    frame = frame.merge(baseline, on=keys, how="left")
    frame["accuracy_drop"] = frame["original_accuracy_pct"] - frame["accuracy_pct"]
    return frame.drop(columns="original_accuracy_pct")


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Means per language/method/rate, standard errors, per-question accuracy and failure
    counts."""
    summary = (
        rows.groupby(["language", "method", "rate"], dropna=False, sort=False)
        .agg(
            cases=("case_id", "count"),
            savings_pct=("savings_pct", "mean"),
            savings_se=("savings_pct", "sem"),
            tokens_saved=("tokens_saved", "mean"),
            accuracy_pct=("accuracy_pct", "mean"),
            accuracy_se=("accuracy_pct", "sem"),
            accuracy_drop=("accuracy_drop", "mean"),
            facts_kept_pct=("facts_kept_pct", "mean"),
            facts_kept_se=("facts_kept_pct", "sem"),
            latency_ms=("latency_ms", "mean"),
            teacher_failures=("teacher_failed", "sum"),
            answer_json_failures=("answer_json_failed", "sum"),
            judge_json_failures=("judge_json_failed", "sum"),
            **{f"q{i + 1}_accuracy_pct": (f"q{i + 1}_correct", "mean") for i in range(N_QUESTIONS)},
        )
        .reset_index()
    )
    for i in range(N_QUESTIONS):
        summary[f"q{i + 1}_accuracy_pct"] *= 100
    summary["json_failures"] = summary["answer_json_failures"] + summary["judge_json_failures"]
    return summary


def results_table(summary: pd.DataFrame) -> pd.DataFrame:
    """(a) The summary table with readable column names."""
    return summary[["language", "method", "rate", *RESULT_COLUMNS]].rename(columns=RESULT_COLUMNS)


def question_type_table(summary: pd.DataFrame) -> pd.DataFrame:
    """(b) Accuracy % per question type, one row per language/method/rate."""
    columns = {f"q{i + 1}_accuracy_pct": name for i, name in enumerate(QUESTION_TYPES)}
    return summary[["language", "method", "rate", *columns]].rename(columns=columns)


def domain_drop_table(rows: pd.DataFrame, methods: Sequence[str] = DOMAIN_METHODS) -> pd.DataFrame:
    """(d) Mean accuracy drop vs. the original per domain, for the selected methods."""
    selected = rows[rows["method"].isin(methods)]
    return selected.pivot_table(
        index=["language", "method", "rate"],
        columns="domain",
        values="accuracy_drop",
        aggfunc="mean",
        dropna=False,
    ).reset_index()


def teacher_length_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per teacher row: requested keep rate, actual keep rate (words), attempts used, and
    whether the closest valid output still missed the length target."""
    teachers = rows[rows["method"].isin(TEACHER_METHODS)]
    columns = ["language", "case_id", "method", "requested_keep", "actual_keep", "attempts",
               "length_miss", "teacher_failed"]
    return teachers[columns].reset_index(drop=True)


def token_ratio(cases: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Hebrew/English o200k token ratio of the original texts: overall and mean per case."""
    he = [count_tokens(context(case)) for case in cases["he"]]
    en = [count_tokens(context(case)) for case in cases["en"]]
    per_case = [h / e for h, e in zip(he, en, strict=True)]
    return {
        "he_tokens": sum(he),
        "en_tokens": sum(en),
        "overall": sum(he) / sum(en),
        "mean_per_case": sum(per_case) / len(per_case),
    }


def method_label(method: str, rate: float) -> str:
    return method if pd.isna(rate) else f"{method} @{rate:g}"


def free_label_y(x: float, y: float, length: int, placed: list[tuple[float, float, int]]) -> float:
    """Move a label down until it does not overlap labels already placed (data units: a
    character is about 1 unit wide, a line about 4 units high)."""
    while any(
        abs(y - other_y) < LABEL_HEIGHT
        and x < other_x + other_length * LABEL_CHAR_WIDTH
        and other_x < x + length * LABEL_CHAR_WIDTH
        for other_x, other_y, other_length in placed
    ):
        y -= LABEL_HEIGHT
    return y


def plot_main_chart(summary: pd.DataFrame, path: Path) -> plt.Figure:
    """Main chart, one panel per language: x = tokens saved (%), y = answers correct (%).
    A method with keep rates is one colored line from its 0.7 to its 0.5 point, named next to
    its 0.5 point; methods without a rate are single named points; the original is a red star.
    Light ±1 standard-error bars."""
    names = [name for name in dict.fromkeys(summary["method"]) if name != "original"]
    palette = [plt.get_cmap("tab10")(i) for i in range(10) if i != MAIN_CHART_RED_INDEX]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(names)}
    colors["original"] = plt.get_cmap("tab10")(MAIN_CHART_RED_INDEX)
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(15, 6.5), sharey=True)
    for ax, language in zip(axes, LANGUAGES, strict=True):
        panel = summary[summary["language"] == language]
        placed: list[tuple[float, float, int]] = []
        for name, group in panel.groupby("method", sort=False):
            group = group.sort_values("rate", ascending=False, na_position="first")
            is_original = name == "original"
            color = colors[name]
            ax.errorbar(
                group["savings_pct"],
                group["accuracy_pct"],
                xerr=group["savings_se"].fillna(0),
                yerr=group["accuracy_se"].fillna(0),
                fmt="*" if is_original else "o-",
                markersize=16 if is_original else 6,
                color=color,
                ecolor=(*color[:3], 0.3),
                elinewidth=0.8,
                capsize=2,
            )
            last = group.iloc[-1]
            x, y = last["savings_pct"] + 1.2, last["accuracy_pct"] - 1.0
            y = free_label_y(x, y, len(name), placed)
            placed.append((x, y, len(name)))
            ax.text(x, y, name, fontsize=9, color=color, va="center")
        ax.set_title(LANGUAGE_NAMES[language])
        ax.set_xlabel("Tokens saved (%, o200k_base)")
        ax.set_xlim(-5, 100)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Answers correct (%)")
    fig.suptitle("Lines run from keep rate 0.7 to 0.5; light bars = ±1 SE", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def plot_savings_vs_accuracy(summary: pd.DataFrame, path: Path) -> plt.Figure:
    """One panel per language: mean token savings vs. accuracy, one numbered point per
    method/rate (legend maps numbers to labels) with ±1 standard-error bars; the original is
    marked with a star."""
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(15, 8.5), sharey=True)
    for ax, language in zip(axes, LANGUAGES, strict=True):
        panel = summary[summary["language"] == language]
        for number, (_, row) in enumerate(panel.iterrows(), start=1):
            is_original = row["method"] == "original"
            ax.errorbar(
                row["savings_pct"],
                row["accuracy_pct"],
                xerr=0 if pd.isna(row["savings_se"]) else row["savings_se"],
                yerr=0 if pd.isna(row["accuracy_se"]) else row["accuracy_se"],
                fmt="*" if is_original else "o",
                markersize=14 if is_original else 7,
                color="tab:red" if is_original else "tab:blue",
                ecolor="gray",
                capsize=3,
                label=f"{number}. {method_label(row['method'], row['rate'])}",
            )
            ax.annotate(
                str(number),
                (row["savings_pct"], row["accuracy_pct"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.legend(
            fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, handlelength=1
        )
        ax.set_title(LANGUAGE_NAMES[language])
        ax.set_xlabel("Mean token savings (%, o200k_base)")
        ax.set_xlim(-5, 100)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Answers judged correct (%), ±1 SE")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def plot_metric_bars(
    summary: pd.DataFrame, column: str, se_column: str, ylabel: str, path: Path
) -> plt.Figure:
    """One bar per method/rate, one panel per language, with ±1 standard-error bars."""
    fig, axes = plt.subplots(len(LANGUAGES), 1, figsize=(13, 9))
    for ax, language in zip(axes, LANGUAGES, strict=True):
        panel = summary[summary["language"] == language]
        labels = [method_label(m, r) for m, r in zip(panel["method"], panel["rate"], strict=True)]
        colors = ["tab:red" if m == "original" else "tab:blue" for m in panel["method"]]
        ax.bar(labels, panel[column], yerr=panel[se_column].fillna(0), color=colors, capsize=3)
        ax.set_title(LANGUAGE_NAMES[language])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def markdown_table(frame: pd.DataFrame, digits: int = 1) -> str:
    """Render a DataFrame as a GitHub markdown table."""
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(False)]
    return "\n".join([header, divider, *body])


def write_summary_md(
    path: Path,
    ratio: dict[str, float],
    n_cases: dict[str, int],
    tables: Sequence[tuple[str, pd.DataFrame]],
    charts: Sequence[tuple[str, str]],
    main_chart: tuple[str, str] | None = None,
) -> None:
    """All result tables and charts, the main chart first; no go/no-go verdict and no
    thresholds."""
    lines = [
        "# 03 — Baselines and teachers (Test 3)",
        "",
        f"Cases: {n_cases['he']} Hebrew, {n_cases['en']} English. Spec: `notebooks/03_spec.md`. "
        + "Savings use tiktoken `o200k_base`; error bars are ±1 standard error over cases.",
        "",
        "## Hebrew/English token ratio of the originals",
        "",
        f"- Overall: {ratio['overall']:.2f} ({ratio['he_tokens']} / {ratio['en_tokens']} tokens)",
        f"- Mean per case: {ratio['mean_per_case']:.2f}",
        "",
    ]
    if main_chart is not None:
        title, filename = main_chart
        lines = lines[:2] + [f"## {title}", "", f"![{title}]({filename})", ""] + lines[2:]
    for title, table in tables:
        lines += [f"## {title}", "", markdown_table(table, digits=3 if "$" in title else 1), ""]
    for title, filename in charts:
        lines += [f"## {title}", "", f"![{title}]({filename})", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def fake_reply(request: LLMRequest) -> str:
    """Offline replies for dry runs: teachers delete every third word (every second word after
    length feedback), the chatbot returns valid JSON, the judge returns verdicts that vary with
    the prompt (so charts have spread)."""
    if request.role in TEACHER_METHODS:
        text = request.prompt.split("\n\n", 1)[1]
        step = 3
        if FEEDBACK_PREFIX in text:
            text, step = text.rsplit(f"\n\n{FEEDBACK_PREFIX}", 1)[0], 2
        return "\n".join(
            " ".join(word for i, word in enumerate(line.split()) if i % step != step - 1)
            for line in text.split("\n")
        )
    if request.role == "chatbot":
        return json.dumps({"answers": ["(fake answer)"] * N_QUESTIONS})
    if request.role == "judge":
        digest = hashlib.sha256(request.prompt.encode("utf-8")).digest()
        return json.dumps({"correct": [digest[i] % 3 != 0 for i in range(N_QUESTIONS)]})
    return "OK"
