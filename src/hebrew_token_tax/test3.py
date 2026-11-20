"""Test 3 pipeline (notebooks/03_spec.md): prompts, methods, stages, metrics and reports."""

import json
import math
import re
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
TEACHER_MAX_FAILURE_RATE = 0.30
JSON_MAX_FAILURE_RATE = 0.20
SMOKE_PROMPT = "Reply with OK"
O200K = tiktoken.get_encoding("o200k_base")


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


def teacher_request(
    roles: dict[str, Role],
    prompts: dict[str, str],
    case: dict[str, Any],
    method: Method,
    attempt: int,
) -> LLMRequest:
    """The verbatim teacher prompt, a blank line, then the text to shorten."""
    role = roles[method.name]
    template = prompts[f"teacher_{method.language}"]
    prompt = f"{fill(template, pct=str(round(method.rate * 100)))}\n\n{context(case)}"
    return make_request(method.name, role, prompt, attempt)


def pending_teacher_jobs(
    cases: dict[str, list[dict[str, Any]]], methods: Sequence[Method], store: JsonStore
) -> list[tuple[dict[str, Any], Method]]:
    return [
        (case, method)
        for method in methods
        if method.name in TEACHER_METHODS
        for case in cases[method.language]
        if record_key(case, method) not in store
    ]


def run_teachers(
    client: LLMClient,
    roles: dict[str, Role],
    prompts: dict[str, str],
    jobs: list[tuple[dict[str, Any], Method]],
    store: JsonStore,
) -> int:
    """Teacher compression: up to 2 retries for invalid output, then keep the original text
    and mark the case failed. Each reply is validated as it arrives, so the stage stops early
    (3 invalid in a row, or > 30% of the first 10). Returns the number of failed cases."""
    monitor = StageMonitor("teachers", TEACHER_MAX_FAILURE_RATE, "invalid teacher output")
    job_of: dict[str, tuple[dict[str, Any], Method]] = {}

    def is_valid(result: LLMResult) -> bool:
        case, method = job_of[result.request.cache_key()]
        output = result.text.strip()
        return not result.failed and is_valid_deletion(context(case), output, method.language)

    pending = jobs
    for attempt in range(TEACHER_MAX_ATTEMPTS):
        if not pending:
            break
        requests = [teacher_request(roles, prompts, c, m, attempt) for c, m in pending]
        job_of.update({r.cache_key(): job for r, job in zip(requests, pending, strict=True)})
        results = client.call_many(requests, is_valid, monitor, f"teachers (attempt {attempt + 1})")
        still_invalid = []
        for (case, method), result in zip(pending, results, strict=True):
            output = result.text.strip()
            if is_valid(result):
                store.set(
                    record_key(case, method),
                    {"text": output, "latency_ms": None, "teacher_failed": False,
                     "attempts": attempt + 1},
                )
            else:
                still_invalid.append((case, method))
        pending = still_invalid
        store.save()
    for case, method in pending:
        store.set(
            record_key(case, method),
            {"text": context(case), "latency_ms": None, "teacher_failed": True,
             "attempts": TEACHER_MAX_ATTEMPTS},
        )
    store.save()
    return len(pending)


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
        if (key := record_key(case, method)) in compressions and key not in answers
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
            {"answers": parsed.answers if parsed else None, "json_failed": parsed is None},
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
        if (key := record_key(case, method)) in answers and key not in judgements
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
            judgements.set(key, {"correct": all_wrong, "json_failed": False})
    outcomes = client.call_json_many(
        judge_requests(roles, prompts, jobs, answers), Verdicts, "judge", JSON_MAX_FAILURE_RATE
    )
    failures = 0
    for (key, _), (parsed, _) in zip(judged, outcomes, strict=True):
        failures += parsed is None
        judgements.set(
            key, {"correct": parsed.correct if parsed else all_wrong, "json_failed": parsed is None}
        )
    judgements.save()
    return failures


def facts_kept(facts: list[str], text: str, method_name: str) -> tuple[int, int]:
    """(facts checked, facts kept). Translation methods: digit-containing facts only, kept
    when every digit sequence of the fact appears in the text; otherwise exact substring."""
    if method_name in TRANSLATION_METHODS:
        text_numbers = set(DIGITS.findall(text))
        checked = [fact for fact in facts if has_digit(fact)]
        return len(checked), sum(set(DIGITS.findall(f)) <= text_numbers for f in checked)
    return len(facts), sum(fact in text for fact in facts)


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
    n_correct = sum(judgement["correct"]) if judgement else math.nan
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
        "facts_total": facts_total,
        "facts_kept": facts_ok,
        "facts_kept_pct": 100 * facts_ok / facts_total if facts_total else math.nan,
        "teacher_failed": int(compression["teacher_failed"]),
        "answer_json_failed": int(answer["json_failed"]) if answer else math.nan,
        "judge_json_failed": int(judgement["json_failed"]) if judgement else math.nan,
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
    """Means per language/method/rate, plus failure counts."""
    return (
        rows.groupby(["language", "method", "rate"], dropna=False, sort=False)
        .agg(
            cases=("case_id", "count"),
            savings_pct=("savings_pct", "mean"),
            tokens_saved=("tokens_saved", "mean"),
            accuracy_pct=("accuracy_pct", "mean"),
            accuracy_drop=("accuracy_drop", "mean"),
            facts_kept_pct=("facts_kept_pct", "mean"),
            latency_ms=("latency_ms", "mean"),
            teacher_failures=("teacher_failed", "sum"),
            answer_json_failures=("answer_json_failed", "sum"),
            judge_json_failures=("judge_json_failed", "sum"),
        )
        .reset_index()
    )


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


def plot_savings_vs_accuracy(summary: pd.DataFrame, path: Path) -> plt.Figure:
    """One panel per language: mean token savings vs. accuracy, one point per method/rate."""
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(14, 6), sharey=True)
    for ax, language in zip(axes, LANGUAGES, strict=True):
        panel = summary[summary["language"] == language]
        ax.scatter(panel["savings_pct"], panel["accuracy_pct"], s=60)
        for _, row in panel.iterrows():
            ax.annotate(
                method_label(row["method"], row["rate"]),
                (row["savings_pct"], row["accuracy_pct"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title({"he": "Hebrew", "en": "English"}[language])
        ax.set_xlabel("Mean token savings (%, o200k_base)")
        ax.set_xlim(-5, 100)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Answers judged correct (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def markdown_table(frame: pd.DataFrame, digits: int = 1) -> str:
    """Render a DataFrame as a GitHub markdown table."""
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(False)]
    return "\n".join([header, divider, *body])


def write_summary_md(
    path: Path,
    summary: pd.DataFrame,
    ratio: dict[str, float],
    chart_name: str,
    n_cases: dict[str, int],
) -> None:
    """Tables and chart only; no go/no-go verdict."""
    lines = [
        "# 03 — Baselines and teachers (Test 3)",
        "",
        f"Cases: {n_cases['he']} Hebrew, {n_cases['en']} English. Spec: `notebooks/03_spec.md`. "
        + "Savings use tiktoken `o200k_base`.",
        "",
        "## Hebrew/English token ratio of the originals",
        "",
        f"- Overall: {ratio['overall']:.2f} ({ratio['he_tokens']} / {ratio['en_tokens']} tokens)",
        f"- Mean per case: {ratio['mean_per_case']:.2f}",
        "",
    ]
    for language, title in (("he", "Hebrew"), ("en", "English")):
        table = summary[summary["language"] == language].drop(columns="language")
        lines += [f"## {title}", "", markdown_table(table), ""]
    lines += ["## Chart", "", f"![savings vs accuracy]({chart_name})", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def fake_reply(request: LLMRequest) -> str:
    """Offline replies for dry runs: teachers delete every third word, the chatbot and judge
    return valid JSON."""
    if request.role in TEACHER_METHODS:
        text = request.prompt.split("\n\n", 1)[1]
        return "\n".join(
            " ".join(word for i, word in enumerate(line.split()) if i % 3 != 2)
            for line in text.split("\n")
        )
    if request.role == "chatbot":
        return json.dumps({"answers": ["(fake answer)"] * N_QUESTIONS})
    if request.role == "judge":
        return json.dumps({"correct": [True, False, True]})
    return "OK"
