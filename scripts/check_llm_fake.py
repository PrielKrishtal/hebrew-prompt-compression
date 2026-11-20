"""Stage-2 check: llm.py and the Test 3 pipeline with the fake provider only (no network,
no models, no API keys). Writes only to a temporary directory.

Usage: python scripts/check_llm_fake.py
"""

import json
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import anthropic
import httpx2
import openai

from hebrew_token_tax.compressors import (
    LLMLINGUA2_LARGE,
    LLMLINGUA2_SMALL,
    LLMLingua2Compressor,
    NllbTranslator,
    RuleCompressor,
)
from hebrew_token_tax.llm import (
    MAX_CONCURRENCY,
    AnthropicProvider,
    BillingError,
    FakeProvider,
    FatalCallError,
    LLMClient,
    LLMRequest,
    OpenAIProvider,
    Price,
    SpendBlocked,
    StageAborted,
    guard_spend,
    parse_json_reply,
)
from hebrew_token_tax.test3 import (
    TEACHER_METHODS,
    Answers,
    JsonStore,
    LocalCompressors,
    Method,
    Role,
    Verdicts,
    answer_jobs,
    context,
    facts_kept,
    fake_reply,
    fill,
    fingerprint,
    judge_jobs,
    load_cases,
    load_prompts,
    method_grid,
    pending_teacher_jobs,
    record_key,
    resolve_roles,
    run_answers,
    run_judge,
    run_local,
    run_teachers,
    score_rows,
    summarize,
    teacher_request,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "03"
SPEC_PATH = ROOT / "notebooks" / "03_spec.md"
PRICES = {"paid-model": Price(2.0, 10.0)}
OUTPUT_DEFAULTS = {"smoke": 10, "chatbot": 100, "judge": 50, "teacher_sol": 500,
                   "teacher_opus": 500}
TEXT_COLUMNS = {"history", "doc", "text", "answers", "questions"}
IN_FLIGHT = 1
results: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'} {name}")


def make_client(
    tmp: Path,
    name: str,
    responder: Callable[[LLMRequest], str],
    budget_caps: dict[str, float] | None = None,
    **fake_options: object,
) -> tuple[LLMClient, FakeProvider]:
    fake = FakeProvider(responder, **fake_options)
    client = LLMClient(
        tmp / name, {"fake": fake, "openai": fake}, PRICES, budget_caps,
        sleep=lambda seconds: None,
    )
    return client, fake


def request(
    prompt: str,
    provider: str = "fake",
    role: str = "chatbot",
    attempt: int = 0,
    max_output_tokens: int = 1000,
) -> LLMRequest:
    model = "paid-model" if provider == "openai" else "fake"
    return LLMRequest(role, provider, model, prompt, "low", max_output_tokens, attempt)


def raises(error_type: type[BaseException], function: Callable[[], object]) -> bool:
    """True if `function` raises `error_type`; prints the stop message as the notebook would."""
    try:
        function()
    except error_type as error:
        print(f"    stopped: {type(error).__name__}: {error}")
        return True
    return False


def check_cache(tmp: Path) -> None:
    client, fake = make_client(tmp, "cache", lambda r: "hello")
    first = client.call(request("p"))
    second = client.call(request("p"))
    check("cache: second call is a cache hit", not first.cached and second.cached)
    check("cache: provider called once", fake.calls == 1)
    client.call(request("p", attempt=1))
    check("cache: attempt index is part of the key", fake.calls == 2)


def check_output_limit(tmp: Path) -> None:
    """Raising the output limit keeps cached replies; a cut-off reply is requested again."""
    client, fake = make_client(tmp, "limit", lambda r: "full reply")
    client.call(request("p", provider="openai", max_output_tokens=16000))
    again = client.call(request("p", provider="openai", max_output_tokens=32000))
    check("limit: raising max_output_tokens reuses the cached reply", again.cached and fake.calls == 1)

    path = client.cache_path(request("q", provider="openai"))
    client.call(request("q", provider="openai", max_output_tokens=100))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["error"] = "stop_reason max_tokens"
    path.write_text(json.dumps(record), encoding="utf-8")
    first_cost = record["cost_usd"]
    redo = client.call(request("q", provider="openai", max_output_tokens=32000))
    new_record = json.loads(path.read_text(encoding="utf-8"))
    check("limit: reply cut off at a lower limit is requested again",
          not redo.cached and fake.calls == 3 and not new_record["error"])
    check("limit: the cut-off reply's cost stays counted",
          new_record["earlier_cost_usd"] == first_cost
          and abs(client.all_time_spend()["openai"] - client.session_spend()) < 1e-12)
    same = client.call(request("q", provider="openai", max_output_tokens=100))
    check("limit: a lower limit never re-requests", same.cached and fake.calls == 3)


def check_in_flight_spend(tmp: Path) -> None:
    """When a stage stops, calls already running finish and are counted in the spend."""
    client, fake = make_client(tmp, "in_flight", lambda r: "not json", max_workers=4,
                               delay_seconds=0.05)
    requests = [request(f"f{i}", provider="openai") for i in range(12)]
    stopped = raises(StageAborted, lambda: client.call_json_many(requests, Answers, "answers"))
    paid = sum(1 for _ in (tmp / "in_flight").glob("*.json"))
    check("in-flight: stage stops, running calls finish and are all counted in spend",
          stopped and paid == fake.successes == sum(client.ledger.calls.values()) > 3
          and abs(client.session_spend() - client.all_time_spend()["openai"]) < 1e-12)
    check("concurrency: Anthropic runs at most 2 calls at once",
          AnthropicProvider.max_workers == 2 and OpenAIProvider.max_workers == 4)


def check_backoff(tmp: Path) -> None:
    sleeps: list[float] = []
    fake = FakeProvider(lambda r: "ok", fail_first=2)
    client = LLMClient(tmp / "backoff", {"fake": fake}, PRICES, sleep=sleeps.append)
    result = client.call(request("p"))
    check("backoff: succeeds after 2 simulated 429s", result.text == "ok" and fake.calls == 3)
    check("backoff: delays grow", len(sleeps) == 2 and sleeps[1] > sleeps[0])

    def broken(_: LLMRequest) -> str:
        raise ValueError("bad request")

    client, fake = make_client(tmp, "broken", broken)
    check("fail-fast: non-retryable error stops at once (FatalCallError)",
          raises(FatalCallError, lambda: client.call(request("p"))) and fake.calls == 1)
    client, fake = make_client(tmp, "flaky", lambda r: "ok", fail_first=100)
    check("fail-fast: retries running out stop the stage",
          raises(FatalCallError, lambda: client.call(request("p"))) and fake.calls == 6)


def check_concurrency(tmp: Path) -> None:
    client, fake = make_client(
        tmp, "concurrency", lambda r: r.prompt, max_workers=10, delay_seconds=0.05
    )
    requests = [request(f"p{i}") for i in range(20)] + [request("p0")]
    replies = client.call_many(requests)
    check("concurrency: at most 4 calls at once", fake.max_active == 4)
    check("concurrency: duplicate request paid once", fake.calls == 20)
    check("concurrency: input order kept", [r.text for r in replies][:3] == ["p0", "p1", "p2"])


def check_json(tmp: Path) -> None:
    def second_try_valid(r: LLMRequest) -> str:
        return '{"answers": ["a", "b", "c"]}' if r.attempt == 1 else "not json"

    client, _ = make_client(tmp, "json", second_try_valid)
    [(parsed, calls)] = client.call_json_many([request("p")], Answers)
    check("json: invalid reply retried once, then parsed", parsed is not None and len(calls) == 2)
    client, _ = make_client(tmp, "json_fail", lambda r: '{"answers": ["only one"]}')
    [(parsed, calls)] = client.call_json_many([request("p")], Answers)
    check("json: still invalid after retry -> failed", parsed is None and len(calls) == 2)
    fenced = '```json\n{"correct": [true, false, true]}\n```'
    check("json: code fences accepted", parse_json_reply(fenced, Verdicts) is not None)
    loose = '{"correct": ["true", false, true]}'
    check("json: string 'true' rejected", parse_json_reply(loose, Verdicts) is None)


def check_cost(tmp: Path) -> None:
    check("cost: $2/$10 per 1M -> 1M in + 100k out = $3", Price(2, 10).cost(1_000_000, 100_000) == 3)
    client, _ = make_client(tmp, "cost", lambda r: "reply text")
    result = client.call(request("some prompt", provider="openai"))
    expected = PRICES["paid-model"].cost(result.input_tokens, result.output_tokens)
    check("cost: recorded per call", result.cost_usd == expected > 0)
    check("cost: ledger and all-time spend agree",
          client.ledger.cost["openai"] == client.all_time_spend()["openai"] == expected)


def check_guard(tmp: Path) -> None:
    paid = [request(f"q{i}", provider="openai") for i in range(3)]
    for confirm, cap, should_block, label in (
        (False, 100.0, True, "CONFIRM_SPEND=False stops"),
        (True, 0.0, True, "budget cap stops before the stage"),
        (True, 100.0, False, "confirmed and under cap passes"),
    ):
        client, fake = make_client(tmp, f"guard_{label[:6]}", lambda r: "x", {"openai": cap})
        estimate = client.estimate(paid, OUTPUT_DEFAULTS)
        blocked = raises(SpendBlocked, partial(guard_spend, client, estimate, confirm, "check"))
        check(f"guard: {label}", blocked == should_block and fake.calls == 0)
    check("guard: estimate counts calls to make", estimate.calls["openai"] == 3)
    client.call_many(paid)
    cached_estimate = client.estimate(paid, OUTPUT_DEFAULTS)
    check("guard: cached calls counted and cost nothing",
          cached_estimate.total == 0 and cached_estimate.cached["openai"] == 3)

    long_reply = " ".join(["word"] * 900)
    client, fake = make_client(tmp, "cap_mid", lambda r: long_reply, {"openai": 0.025})
    many = [request(f"m{i}", provider="openai", max_output_tokens=1000) for i in range(20)]
    check("fail-fast: budget cap stops mid-stage before an over-cap call",
          raises(SpendBlocked, lambda: client.call_many(many, desc=None))
          and 0 < fake.calls < 20 and client.spent("openai") <= 0.025)


def sdk_error(sdk: object, status: int, body: dict[str, object]) -> Exception:
    """A real SDK status error, as the SDK would raise it."""
    request_ = httpx2.Request("POST", "https://api.example.invalid")
    response = httpx2.Response(status, request=request_)
    return sdk.APIStatusError("simulated", response=response, body=body)


def check_classification() -> None:
    """Real SDK errors are classified as the fail-fast rules require."""
    providers = (("openai", OpenAIProvider(), openai), ("anthropic", AnthropicProvider(), anthropic))
    for name, provider, sdk in providers:
        for status in (400, 401, 403, 404):
            error = sdk_error(sdk, status, {"error": {"type": "invalid_request_error"}})
            check(f"classify {name}: {status} is fatal (not retried, not billing)",
                  not provider.is_retryable(error) and not provider.is_billing_error(error))
        for status in (429, 500, 529):
            error = sdk_error(sdk, status, {"error": {"type": "rate_limit_error"}})
            check(f"classify {name}: {status} is retried",
                  provider.is_retryable(error) and not provider.is_billing_error(error))
    quota = sdk_error(openai, 429, {"error": {"code": "insufficient_quota"}})
    check("classify openai: 429 insufficient_quota is a billing error",
          OpenAIProvider().is_billing_error(quota))
    credit = sdk_error(anthropic, 400, {"type": "error", "error": {"type": "billing_error"}})
    check("classify anthropic: billing_error is a billing error",
          AnthropicProvider().is_billing_error(credit))
    check("classify: HTTP 402 is a billing error",
          OpenAIProvider().is_billing_error(sdk_error(openai, 402, {})))


def check_fail_fast_rates(tmp: Path, prompts: dict[str, str]) -> None:
    """3 consecutive failures, teacher > 30% and JSON > 20% in the first 10 calls."""
    client, fake = make_client(tmp, "consecutive", lambda r: "not json", max_workers=1)
    requests = [request(f"c{i}") for i in range(12)]
    check("fail-fast: 3 consecutive failed calls stop the stage",
          raises(StageAborted, lambda: client.call_json_many(requests, Answers, "answers", 0.2))
          and fake.calls <= 3 + IN_FLIGHT)

    counter = iter(range(10_000))

    def every_third_bad(_: LLMRequest) -> str:
        return "broken" if next(counter) % 3 == 2 else '{"answers": ["a", "b", "c"]}'

    client, fake = make_client(tmp, "json_rate", every_third_bad, max_workers=1)
    check("fail-fast: > 20% invalid JSON in the first 10 calls stops the stage",
          raises(StageAborted, lambda: client.call_json_many(requests, Answers, "answers", 0.2))
          and fake.calls <= 10 + IN_FLIGHT)

    flip = iter(range(10_000))

    def alternating_teacher(r: LLMRequest) -> str:
        return fake_reply(r) if next(flip) % 2 == 0 else "rewritten, not a deletion"

    client, fake = make_client(tmp, "teacher_rate", alternating_teacher, max_workers=1)
    roles = resolve_roles(NOTEBOOK_ROLES, "fake", DEBUG_MODELS)
    cases = {lang: load_cases(DATA_DIR, lang, limit=3) for lang in ("he", "en")}
    methods = [m for m in method_grid((0.7, 0.5)) if m.name in TEACHER_METHODS]
    store = JsonStore(tmp / "teacher_rate" / "teachers.json")
    jobs = pending_teacher_jobs(cases, methods, store)
    check("fail-fast: > 30% invalid teacher output in the first 10 calls stops the stage",
          raises(StageAborted, lambda: run_teachers(client, roles, prompts, jobs, store))
          and fake.calls <= 10 + IN_FLIGHT and not store.data)


def check_prompts() -> dict[str, str]:
    prompts = load_prompts(SPEC_PATH)
    check("prompts: 5 prompts loaded (incl. teacher length feedback)",
          set(prompts) == {"teacher_he", "teacher_en", "teacher_feedback", "chatbot", "judge"})
    check("prompts: chatbot note excluded", "answer_language =" not in prompts["chatbot"])
    check("prompts: judge returns a 3-item list", '{"correct": [' in prompts["judge"])
    try:
        fill(prompts["teacher_he"], wrong="x")
        rejected = False
    except ValueError:
        rejected = True
    check("prompts: fill rejects missing/extra fields", rejected)
    return prompts


def check_pipeline(tmp: Path, prompts: dict[str, str]) -> None:
    roles = {name: Role("fake", "fake", "low", 1000)
             for name in ("teacher_sol", "teacher_opus", "chatbot", "judge")}
    cases = {lang: load_cases(DATA_DIR, lang, limit=2) for lang in ("he", "en")}
    methods = [m for m in method_grid((0.7, 0.5)) if m.name in TEACHER_METHODS | {"original"}]
    check("pipeline: spec grid has 23 method/rate/language combos",
          len(method_grid((0.7, 0.5))) == 23)
    compressions = JsonStore(tmp / "pipeline" / "compressions.json")
    for method in methods:
        if method.name == "original":
            for case in cases[method.language]:
                compressions.set(record_key(case, method),
                                 {"text": context(case), "latency_ms": 0.0,
                                  "teacher_failed": False, "attempts": 0})
    client, fake = make_client(tmp, "pipeline_llm", fake_reply)
    jobs = pending_teacher_jobs(cases, methods, compressions)
    failed = run_teachers(client, roles, prompts, jobs, compressions)
    check("pipeline: fake teachers pass the validator", failed == 0 and len(jobs) == 16)

    bad_client, bad_fake = make_client(tmp, "pipeline_bad", lambda r: "rewritten text")
    bad_store = JsonStore(tmp / "pipeline" / "bad.json")
    one_job = jobs[:1]
    bad_failed = run_teachers(bad_client, roles, prompts, one_job, bad_store)
    case, method = one_job[0]
    record = bad_store.get(record_key(case, method))
    check("pipeline: invalid teacher -> 3 attempts, original kept, marked failed",
          bad_failed == 1 and bad_fake.calls == 3 and record["teacher_failed"]
          and record["text"] == context(case))

    answers = JsonStore(tmp / "pipeline" / "answers.json")
    judgements = JsonStore(tmp / "pipeline" / "judgements.json")
    run_answers(client, roles, prompts, answer_jobs(cases, methods, compressions, answers),
                compressions, answers)
    run_judge(client, roles, prompts, judge_jobs(cases, methods, answers, judgements),
              answers, judgements)
    rows = score_rows(cases, methods, compressions, answers, judgements)
    summary = summarize(rows)
    check("pipeline: one row per case/method/rate/language", len(rows) == 20)
    check("pipeline: no text columns in results", not TEXT_COLUMNS & set(rows.columns))
    check("pipeline: every row judged, per-question columns filled",
          rows["n_correct"].notna().all()
          and (rows[["q1_correct", "q2_correct", "q3_correct"]].sum(axis=1) == rows["n_correct"]).all())
    check("facts: spaces around punctuation ignored (Hebrew)",
          facts_kept(["90 ש״ח ליום"], "עד 90 ש ״ח  ליום", "rules") == (1, 1))
    check("facts: spaces around punctuation ignored (English)",
          facts_kept(["windshield-only claims"], "for windshield - only claims", "rules") == (1, 1))
    check("facts: a deleted word still means not kept",
          facts_kept(["90 ש״ח ליום"], "עד 90 ליום", "rules") == (1, 0))
    check("pipeline: original savings are 0", (rows.loc[rows["method"] == "original",
                                                         "savings_pct"] == 0).all())
    check("pipeline: summary has a row per method/rate/language", len(summary) == 10)
    reruns_before = fake.calls
    run_answers(client, roles, prompts, answer_jobs(cases, methods, compressions, answers),
                compressions, answers)
    check("pipeline: rerun makes no new calls", fake.calls == reruns_before)


NOTEBOOK_ROLES = {
    "teacher_sol": Role("openai", "gpt-6.1-sol", "high", 16000),
    "teacher_opus": Role("anthropic", "claude-opus-5-5", "high", 16000),
    "chatbot": Role("openai", "gpt-6.1-terra", "low", 4000),
    "judge": Role("anthropic", "claude-sonnet-5-5", "medium", 8000),
}
DEBUG_MODELS = {"gemini": "gemini-3.5-flash-lite", "fake": "fake"}
RESUME_METHODS = TEACHER_METHODS | {"original", "rules"}
BILLING_AFTER = 7


def check_teacher_length(tmp: Path, prompts: dict[str, str]) -> None:
    """Exact word target in the prompt, length retry with feedback, closest output kept."""
    roles = resolve_roles(NOTEBOOK_ROLES, "fake", DEBUG_MODELS)
    case = load_cases(DATA_DIR, "en", limit=1)[0]
    method = Method("en", "teacher_sol", 0.5)
    n_words = len(context(case).split())
    first = teacher_request(roles, prompts, case, method, 0)
    check("length: prompt states the exact word target",
          first.prompt.startswith(f"The text has {n_words} words. Shorten it to about "
                                  f"{round(n_words * 0.5)} words\n(keep rate 50%)"))
    retry = teacher_request(roles, prompts, case, method, 1, previous_words=200)
    check("length: feedback line follows the text after a blank line",
          retry.prompt.endswith(f"\n\nYour previous output had 200 words; the target is "
                                f"{round(n_words * 0.5)}. Delete more."))

    client, _ = make_client(tmp, "length_ok", fake_reply)
    store = JsonStore(tmp / "length_ok.json")
    jobs = [(case, method), (case, Method("en", "teacher_sol", 0.7))]
    failed = run_teachers(client, roles, prompts, jobs, store)
    at_half, at_70 = (store.get(record_key(case, m)) for _, m in jobs)
    check("length: too long at 0.5 -> retried with feedback, then on target (2 attempts)",
          failed == 0 and at_half["attempts"] == 2 and not at_half["length_miss"]
          and abs(at_half["actual_keep"] - 0.5) <= 0.10)
    check("length: on target at 0.7 -> 1 attempt", at_70["attempts"] == 1)

    def never_shorter(r: LLMRequest) -> str:
        text = r.prompt.split("\n\n", 1)[1].rsplit("\n\nYour previous output had", 1)[0]
        keep = 3 if "Your previous output had" in r.prompt else 4
        return "\n".join(" ".join(w for i, w in enumerate(line.split()) if i % keep != keep - 1)
                         for line in text.split("\n"))

    client, _ = make_client(tmp, "length_miss", never_shorter)
    store = JsonStore(tmp / "length_miss.json")
    failed = run_teachers(client, roles, prompts, [(case, method)], store)
    record = store.get(record_key(case, method))
    check("length: still too long after retries -> closest valid kept, not a failure",
          failed == 0 and not record["teacher_failed"] and record["length_miss"]
          and record["attempts"] == 3 and abs(record["actual_keep"] - 2 / 3) < 0.03)


def check_stale_records(tmp: Path, prompts: dict[str, str]) -> None:
    """A changed teacher text re-runs its answer and judgement; unchanged rows are reused."""
    fake = FakeProvider(fake_reply)
    run_stages(tmp / "stale_llm", tmp / "stale_stage", 1, prompts, fake)
    teachers = JsonStore(tmp / "stale_stage" / "teachers.json")
    key = next(k for k in teachers.data if "teacher_opus" in k and "|0.5" in k)
    record = teachers.get(key)
    old_format = {k: record[k] for k in ("text", "latency_ms", "teacher_failed", "attempts")}
    old_format["text"] = record["text"] + " extra"
    teachers.set(key, old_format)
    teachers.save()
    before = fake.successes
    run_stages(tmp / "stale_llm", tmp / "stale_stage", 1, prompts, fake)
    answers = JsonStore(tmp / "stale_stage" / "answers.json")
    teachers = JsonStore(tmp / "stale_stage" / "teachers.json")
    check("stale: old-format teacher record is redone, its answer and judgement follow",
          "requested_keep" in teachers.get(key)
          and answers.get(key)["text_sha"] == fingerprint(teachers.get(key)["text"])
          and fake.successes == before)


def run_stages(
    cache: Path, stage_dir: Path, n_cases: int, prompts: dict[str, str], fake: FakeProvider
) -> None:
    """The notebook's flow with PROVIDER_OVERRIDE="fake": local (rules only, no models),
    teachers, answers, judge. Stage files and the LLM cache persist between calls."""
    roles = resolve_roles(NOTEBOOK_ROLES, "fake", DEBUG_MODELS)
    cases = {lang: load_cases(DATA_DIR, lang, limit=n_cases) for lang in ("he", "en")}
    methods = [m for m in method_grid((0.7, 0.5)) if m.name in RESUME_METHODS]
    client = LLMClient(cache, {"fake": fake}, PRICES, sleep=lambda seconds: None)
    tools = LocalCompressors(
        rules={lang: RuleCompressor(lang, DATA_DIR) for lang in ("he", "en")},
        llmlingua2_large=LLMLingua2Compressor(LLMLINGUA2_LARGE),
        llmlingua2_small=LLMLingua2Compressor(LLMLINGUA2_SMALL),
        translator=NllbTranslator(),
    )
    local = JsonStore(stage_dir / "local.json")
    run_local(cases, methods, tools, local)
    teachers = JsonStore(stage_dir / "teachers.json")
    run_teachers(client, roles, prompts, pending_teacher_jobs(cases, methods, teachers), teachers)
    compressions = JsonStore.combined(local, teachers)
    answers = JsonStore(stage_dir / "answers.json")
    run_answers(client, roles, prompts, answer_jobs(cases, methods, compressions, answers),
                compressions, answers)
    judgements = JsonStore(stage_dir / "judgements.json")
    run_judge(client, roles, prompts, judge_jobs(cases, methods, answers, judgements),
              answers, judgements)


def no_failures_recorded(stage_dir: Path, cache: Path) -> bool:
    """No stage record is marked failed and no cached reply carries an error."""
    for name, flag in (("teachers", "teacher_failed"), ("answers", "json_failed"),
                       ("judgements", "json_failed")):
        if any(record[flag] for record in JsonStore(stage_dir / f"{name}.json").data.values()):
            return False
    return not any(record["error"] for record in LLMClient(cache, {}, {}).cached_records())


def check_resume(tmp: Path, prompts: dict[str, str]) -> None:
    reference = FakeProvider(fake_reply)
    run_stages(tmp / "ref_llm", tmp / "ref_stage", 2, prompts, reference)
    dry_run_calls = reference.successes
    run_stages(tmp / "ref_llm", tmp / "ref_stage", 2, prompts, reference)
    check("resume: rerun makes 0 new calls", reference.successes == dry_run_calls)
    run_stages(tmp / "ref_llm", tmp / "ref_stage", 3, prompts, reference)
    increment = reference.successes - dry_run_calls

    scratch = FakeProvider(fake_reply)
    run_stages(tmp / "scratch_llm", tmp / "scratch_stage", 3, prompts, scratch)
    check("resume: dry-run cases reused (3 cases after 2 pays only the 3rd)",
          0 < increment == scratch.successes - dry_run_calls)

    run_stages(tmp / "int_llm", tmp / "int_stage", 2, prompts, FakeProvider(fake_reply))
    broke = FakeProvider(fake_reply, billing_after=BILLING_AFTER)
    try:
        run_stages(tmp / "int_llm", tmp / "int_stage", 3, prompts, broke)
        stopped = False
    except BillingError:
        stopped = True
    check("billing: insufficient credit stops the stage", stopped)
    check("billing: queued calls cancelled after the error",
          broke.calls <= BILLING_AFTER + MAX_CONCURRENCY)
    check("billing: nothing cached or counted as failed/incorrect",
          no_failures_recorded(tmp / "int_stage", tmp / "int_llm"))
    resumed = FakeProvider(fake_reply)
    run_stages(tmp / "int_llm", tmp / "int_stage", 3, prompts, resumed)
    check("resume: interrupted run finishes without repeating paid calls",
          broke.successes + resumed.successes == increment)
    check("resume: interrupted + resumed stages equal a clean run",
          all(JsonStore(tmp / "int_stage" / f"{name}.json").data
              == JsonStore(tmp / "ref_stage" / f"{name}.json").data
              for name in ("teachers", "answers", "judgements")))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        check_cache(tmp)
        check_output_limit(tmp)
        check_in_flight_spend(tmp)
        check_backoff(tmp)
        check_concurrency(tmp)
        check_json(tmp)
        check_cost(tmp)
        check_guard(tmp)
        prompts = check_prompts()
        check_pipeline(tmp, prompts)
        check_resume(tmp, prompts)
        check_teacher_length(tmp, prompts)
        check_stale_records(tmp, prompts)
        check_classification()
        check_fail_fast_rates(tmp, prompts)
    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
