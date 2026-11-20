"""One call() for OpenAI, Anthropic and Gemini: on-disk cache, backoff, concurrency, cost.

Reasoning levels (spec: high / medium / low) map to each SDK's real parameter:
- OpenAI (Responses API): reasoning={"effort": level}; no temperature is sent.
- Anthropic: output_config={"effort": level}; adaptive thinking is on by default
  (always on for Opus 5.5) and SDK 2.3.0 has no temperature parameter.
- Gemini: thinking_config=ThinkingConfig(thinking_level=LEVEL), temperature=0.
The output-token limit (max_output_tokens / max_tokens) includes reasoning tokens.

Fail-fast: non-retryable API errors (400/401/403/404), billing errors, retries running out,
3 consecutive failed calls, a too-high failure rate in the first 10 calls of a stage, and a
call that could pass the budget cap all stop the stage at once; queued calls are cancelled.
"""

import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

import tiktoken
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

REASONING_LEVELS = ("low", "medium", "high")
MAX_CONCURRENCY = 4
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_JITTER = 0.25
REQUEST_TIMEOUT_SECONDS = 600.0
MAX_CONSECUTIVE_FAILURES = 3
EARLY_CHECK_CALLS = 10
CAP_INPUT_MARGIN = 2.0
CACHE_KEY_CHARS = 16
TOKENS_PER_MILLION = 1_000_000
RATE_LIMIT_CODE = 429
PAYMENT_REQUIRED_CODE = 402
SERVER_ERROR_CODE = 500
BILLING_MARKERS = ("insufficient_quota", "billing_error", "credit balance", "billing_hard_limit")
DAILY_QUOTA_MARKER = "perday"
JSON_RETRY_ATTEMPT = 1
CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
O200K = tiktoken.get_encoding("o200k_base")


class SpendBlocked(RuntimeError):
    """Raised when a paid stage is not confirmed or would pass a budget cap."""


class BillingError(RuntimeError):
    """Fatal: the account is out of credit or quota. The stage stops; nothing is cached and
    nothing is counted as failed or incorrect, so a rerun after topping up resumes cleanly."""


class FatalCallError(RuntimeError):
    """Fatal: a non-retryable API error (auth, model not found, bad request) or retries ran out."""


class StageAborted(RuntimeError):
    """Fatal: too many failed calls (consecutive, or as a share of a stage's first calls)."""


@dataclass(frozen=True)
class Price:
    """USD per 1M input and output tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost of one call."""
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / TOKENS_PER_MILLION


FREE = Price(0.0, 0.0)


@dataclass(frozen=True)
class LLMRequest:
    """One prompt for one model. `role` names the cache file only; `attempt` is in the key."""

    role: str
    provider: str
    model: str
    prompt: str
    reasoning: str
    max_output_tokens: int
    attempt: int = 0

    def cache_key(self) -> str:
        """Hash of provider, model, prompt and params (reasoning, output limit, attempt)."""
        payload = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "prompt": self.prompt,
                "params": {
                    "reasoning": self.reasoning,
                    "max_output_tokens": self.max_output_tokens,
                    "attempt": self.attempt,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CACHE_KEY_CHARS]


@dataclass(frozen=True)
class Reply:
    """What a provider returned. `error` is set for paid but unusable replies (refusal, cut off)."""

    text: str
    input_tokens: int
    output_tokens: int
    error: str = ""


@dataclass(frozen=True)
class LLMResult:
    """A reply with its cost; `cached` is True when no API call was made."""

    request: LLMRequest
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    error: str = ""

    @property
    def failed(self) -> bool:
        """True for refused or cut-off replies."""
        return bool(self.error)


class Provider(Protocol):
    """A model API: generate one reply and say which errors are worth retrying."""

    name: str
    max_workers: int
    pause_seconds: float

    def generate(self, request: LLMRequest) -> Reply: ...

    def is_retryable(self, error: Exception) -> bool: ...

    def is_billing_error(self, error: Exception) -> bool: ...


def require_key(name: str) -> str:
    """Return an API key from the environment without ever printing it."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


def mentions_billing(error: Exception, status_error: type) -> bool:
    """True for HTTP 402 or an error body naming insufficient credit or quota."""
    if not isinstance(error, status_error):
        return False
    details = f"{json.dumps(getattr(error, 'body', None), default=str)} {error}".lower()
    is_payment = error.status_code == PAYMENT_REQUIRED_CODE
    return is_payment or any(marker in details for marker in BILLING_MARKERS)


def is_retryable_status(error: Exception, status_error: type, connection_error: type) -> bool:
    """True for 429, 5xx and connection errors of an OpenAI-style SDK."""
    if isinstance(error, connection_error):
        return True
    if isinstance(error, status_error):
        status = error.status_code
        return status == RATE_LIMIT_CODE or status >= SERVER_ERROR_CODE
    return False


class OpenAIProvider:
    """OpenAI Responses API; spec reasoning level -> reasoning.effort."""

    name = "openai"
    max_workers = MAX_CONCURRENCY
    pause_seconds = 0.0

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=require_key("OPENAI_API_KEY"),
                max_retries=0,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        return self._client

    def generate(self, request: LLMRequest) -> Reply:
        response = self._get_client().responses.create(
            model=request.model,
            input=request.prompt,
            reasoning={"effort": request.reasoning},
            max_output_tokens=request.max_output_tokens,
        )
        usage = response.usage
        error = "" if response.status == "completed" else f"status {response.status}"
        return Reply(response.output_text, usage.input_tokens, usage.output_tokens, error)

    def is_retryable(self, error: Exception) -> bool:
        import openai

        return is_retryable_status(error, openai.APIStatusError, openai.APIConnectionError)

    def is_billing_error(self, error: Exception) -> bool:
        import openai

        return mentions_billing(error, openai.APIStatusError)


class AnthropicProvider:
    """Anthropic Messages API; spec reasoning level -> output_config.effort."""

    name = "anthropic"
    max_workers = MAX_CONCURRENCY
    pause_seconds = 0.0

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=require_key("ANTHROPIC_API_KEY"),
                max_retries=0,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        return self._client

    def generate(self, request: LLMRequest) -> Reply:
        response = self._get_client().messages.create(
            model=request.model,
            max_tokens=request.max_output_tokens,
            output_config={"effort": request.reasoning},
            messages=[{"role": "user", "content": request.prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        stop = response.stop_reason
        error = f"stop_reason {stop}" if stop in ("refusal", "max_tokens") else ""
        usage = response.usage
        return Reply(text, usage.input_tokens, usage.output_tokens, error)

    def is_retryable(self, error: Exception) -> bool:
        import anthropic

        return is_retryable_status(error, anthropic.APIStatusError, anthropic.APIConnectionError)

    def is_billing_error(self, error: Exception) -> bool:
        import anthropic

        return mentions_billing(error, anthropic.APIStatusError)


class GeminiProvider:
    """Free Gemini keys for debugging: one call at a time, paced to the free-tier RPM,
    moving to the next key when a key's daily quota is used up (as in notebook 02)."""

    name = "gemini"
    max_workers = 1

    def __init__(
        self,
        key_names: Sequence[str],
        requests_per_minute: float,
        send_thinking_level: bool = True,
    ) -> None:
        self.key_names = tuple(key_names)
        self.pause_seconds = 60.0 / requests_per_minute + 1.0
        self.send_thinking_level = send_thinking_level
        self._clients: list[Any] = []
        self._key_index = 0

    def _get_client(self) -> Any:
        if not self._clients:
            from google import genai

            self._clients = [genai.Client(api_key=require_key(n)) for n in self.key_names]
        return self._clients[self._key_index]

    def _config(self, request: LLMRequest) -> Any:
        from google.genai import types

        thinking = (
            types.ThinkingConfig(thinking_level=types.ThinkingLevel[request.reasoning.upper()])
            if self.send_thinking_level
            else None
        )
        return types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=request.max_output_tokens,
            thinking_config=thinking,
            http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_SECONDS * 1000)),
        )

    def _is_daily_quota(self, error: Exception) -> bool:
        from google.genai import errors

        if not isinstance(error, errors.APIError) or error.code != RATE_LIMIT_CODE:
            return False
        return DAILY_QUOTA_MARKER in json.dumps(error.details, default=str).lower()

    def generate(self, request: LLMRequest) -> Reply:
        while True:
            try:
                response = self._get_client().models.generate_content(
                    model=request.model, contents=request.prompt, config=self._config(request)
                )
                break
            except Exception as error:
                if not self._is_daily_quota(error) or self._key_index + 1 >= len(self.key_names):
                    raise
                self._key_index += 1
                print(f"gemini: daily quota reached, switching to key {self._key_index + 1}")
        usage = response.usage_metadata
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        text = response.text or ""
        error = "" if text else "empty response"
        return Reply(text, usage.prompt_token_count or 0, output_tokens, error)

    def is_retryable(self, error: Exception) -> bool:
        from google.genai import errors

        if not isinstance(error, errors.APIError) or self._is_daily_quota(error):
            return False
        return error.code == RATE_LIMIT_CODE or error.code >= SERVER_ERROR_CODE

    def is_billing_error(self, error: Exception) -> bool:
        """A daily quota reaching the caller means every key is used up."""
        return self._is_daily_quota(error)


class TransientError(Exception):
    """A retryable error raised by the fake provider."""


class FakeBillingError(Exception):
    """A billing error raised by the fake provider."""


class FakeProvider:
    """Offline provider for tests and dry runs: `responder(request)` returns the reply text.
    The first `fail_first` calls raise TransientError to exercise the backoff; after
    `billing_after` successful calls every call raises FakeBillingError."""

    name = "fake"
    pause_seconds = 0.0

    def __init__(
        self,
        responder: Callable[[LLMRequest], str],
        max_workers: int = MAX_CONCURRENCY,
        fail_first: int = 0,
        delay_seconds: float = 0.0,
        billing_after: int | None = None,
    ) -> None:
        self.responder = responder
        self.max_workers = max_workers
        self.delay_seconds = delay_seconds
        self.billing_after = billing_after
        self.calls = 0
        self.successes = 0
        self.active = 0
        self.max_active = 0
        self._failures_left = fail_first
        self._lock = threading.Lock()

    def generate(self, request: LLMRequest) -> Reply:
        with self._lock:
            self.calls += 1
            if self.billing_after is not None and self.successes >= self.billing_after:
                raise FakeBillingError("simulated insufficient_quota")
            if self._failures_left > 0:
                self._failures_left -= 1
                raise TransientError("simulated 429")
            self.successes += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay_seconds)
            text = self.responder(request)
        finally:
            with self._lock:
                self.active -= 1
        return Reply(text, len(O200K.encode(request.prompt)), len(O200K.encode(text)))

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, TransientError)

    def is_billing_error(self, error: Exception) -> bool:
        return isinstance(error, FakeBillingError)


def retry_with_backoff[T](
    function: Callable[[], T],
    is_retryable: Callable[[Exception], bool],
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `function`, retrying retryable errors with exponential backoff and jitter."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return function()
        except Exception as error:
            if attempt == max_retries or not is_retryable(error):
                raise
            sleep(delay * (1 + BACKOFF_JITTER * random.random()))
            delay *= 2
    raise AssertionError("unreachable")


def parse_json_reply[Schema: BaseModel](text: str, schema: type[Schema]) -> Schema | None:
    """Parse the first {...} object in `text` (code fences allowed) into `schema`, or None."""
    stripped = CODE_FENCE.sub("", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        return schema.model_validate_json(stripped[start:end + 1])
    except ValidationError:
        return None


@dataclass
class SpendLedger:
    """Cost and number of API calls made in this session, per provider."""

    cost: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: Counter[str] = field(default_factory=Counter)
    new_by_stage: Counter[tuple[str, str]] = field(default_factory=Counter)
    cached_by_stage: Counter[tuple[str, str]] = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, request: LLMRequest, cost: float) -> None:
        """Record one paid (uncached) call."""
        with self._lock:
            self.cost[request.provider] += cost
            self.calls[request.provider] += 1
            self.new_by_stage[(request.role, request.provider)] += 1

    def add_cache_hit(self, request: LLMRequest) -> None:
        """Record one reply served from the cache."""
        with self._lock:
            self.cached_by_stage[(request.role, request.provider)] += 1


@dataclass(frozen=True)
class SpendEstimate:
    """Calls to make, calls already cached, and the estimated cost of the uncached calls
    (first attempts only: teacher and JSON retries are not included)."""

    calls: Counter[str]
    cached: Counter[str]
    cost: dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.cost.values())


class StageMonitor:
    """Stops a stage on 3 consecutive failed calls, or when more than `max_failure_rate` of its
    first 10 calls failed. Only fresh (uncached) first attempts count: a rerun is never
    re-judged on replies it already paid for, and the spec's capped retries (teacher: 2,
    JSON: 1) of one stubborn case do not stop the whole stage."""

    def __init__(
        self,
        stage: str,
        max_failure_rate: float | None = None,
        failure_label: str = "failed",
    ) -> None:
        self.stage = stage
        self.max_failure_rate = max_failure_rate
        self.failure_label = failure_label
        self.calls = 0
        self.failures = 0
        self.consecutive = 0
        self._lock = threading.Lock()

    def observe(self, result: LLMResult, ok: bool) -> None:
        """Record one result; raise StageAborted when a stop rule is hit."""
        if result.cached or result.request.attempt > 0:
            return
        with self._lock:
            self.calls += 1
            self.failures += not ok
            self.consecutive = 0 if ok else self.consecutive + 1
            if self.consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise StageAborted(
                    f"{self.stage}: {self.consecutive} consecutive failed calls "
                    f"({self.failure_label}, last error: {result.error or '-'}); stage stopped"
                )
            rate = self.failures / self.calls
            limit = self.max_failure_rate
            if self.calls == EARLY_CHECK_CALLS and limit is not None and rate > limit:
                raise StageAborted(
                    f"{self.stage}: {self.failures}/{self.calls} first calls "
                    f"{self.failure_label} (> {limit:.0%}); stage stopped"
                )


class LLMClient:
    """Cached, retried, concurrent calls to the configured providers, with cost tracking and
    a per-provider budget cap checked before every uncached call."""

    def __init__(
        self,
        cache_dir: Path,
        providers: dict[str, Provider],
        prices: dict[str, Price],
        budget_caps: dict[str, float] | None = None,
        max_retries: int = MAX_RETRIES,
        initial_backoff: float = INITIAL_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = cache_dir
        self.providers = providers
        self.prices = prices
        self.budget_caps = budget_caps or {}
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.sleep = sleep
        self.ledger = SpendLedger()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._spent_before = self.all_time_spend()
        self._reserved: dict[str, float] = defaultdict(float)
        self._budget_lock = threading.Lock()

    def price(self, request: LLMRequest) -> Price:
        """Price of the request's model; fake and Gemini (free keys) cost nothing."""
        if request.provider in ("fake", "gemini"):
            return FREE
        if request.model not in self.prices:
            raise SpendBlocked(f"No price configured for {request.model}")
        return self.prices[request.model]

    def spent(self, provider: str) -> float:
        """All-time spend of `provider`: earlier sessions plus this one."""
        return self._spent_before.get(provider, 0.0) + self.ledger.cost.get(provider, 0.0)

    def worst_case_cost(self, request: LLMRequest) -> float:
        """Upper bound for one call: generous input count plus the full output limit."""
        input_tokens = len(O200K.encode(request.prompt)) * CAP_INPUT_MARGIN
        return self.price(request).cost(input_tokens, request.max_output_tokens)

    def _reserve(self, request: LLMRequest, amount: float) -> None:
        cap = self.budget_caps.get(request.provider)
        with self._budget_lock:
            committed = self.spent(request.provider) + self._reserved[request.provider]
            if cap is not None and committed + amount > cap:
                raise SpendBlocked(
                    f"{request.provider}: next call could pass the ${cap:.2f} cap "
                    f"(spent ${self.spent(request.provider):.2f}); stage stopped"
                )
            self._reserved[request.provider] += amount

    def _release(self, request: LLMRequest, amount: float) -> None:
        with self._budget_lock:
            self._reserved[request.provider] -= amount

    def cache_path(self, request: LLMRequest) -> Path:
        return self.cache_dir / f"{request.role}_{request.cache_key()}.json"

    def is_cached(self, request: LLMRequest) -> bool:
        return self.cache_path(request).exists()

    def _cached_result(self, request: LLMRequest, path: Path) -> LLMResult:
        record = json.loads(path.read_text(encoding="utf-8"))
        return LLMResult(
            request,
            record["text"],
            record["input_tokens"],
            record["output_tokens"],
            record["cost_usd"],
            cached=True,
            error=record["error"],
        )

    def _generate(self, provider: Provider, request: LLMRequest) -> Reply:
        """Call the provider with backoff; billing and non-retryable errors become fatal."""
        try:
            return retry_with_backoff(
                lambda: provider.generate(request),
                lambda error: provider.is_retryable(error)
                and not provider.is_billing_error(error),
                self.max_retries,
                self.initial_backoff,
                self.sleep,
            )
        except Exception as error:
            where = f"{request.provider} ({request.model})"
            if provider.is_billing_error(error):
                raise BillingError(
                    f"{where}: out of credit or quota; stage stopped, nothing cached or counted"
                ) from error
            reason = "retries ran out" if provider.is_retryable(error) else "non-retryable"
            raise FatalCallError(
                f"{where}: {reason}: {type(error).__name__}: {error}; stage stopped"
            ) from error

    def call(self, request: LLMRequest) -> LLMResult:
        """Return the cached reply, or check the budget, call the provider (with backoff),
        cache the reply and pay once."""
        path = self.cache_path(request)
        if path.exists():
            self.ledger.add_cache_hit(request)
            return self._cached_result(request, path)
        provider = self.providers[request.provider]
        price = self.price(request)
        reservation = self.worst_case_cost(request)
        self._reserve(request, reservation)
        try:
            reply = self._generate(provider, request)
            cost = price.cost(reply.input_tokens, reply.output_tokens)
            record = {
                "request": asdict(request),
                "text": reply.text,
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "prompt_o200k_tokens": len(O200K.encode(request.prompt)),
                "cost_usd": cost,
                "error": reply.error,
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            self.ledger.add(request, cost)
        finally:
            self._release(request, reservation)
        if provider.pause_seconds:
            self.sleep(provider.pause_seconds)
        return LLMResult(
            request, reply.text, reply.input_tokens, reply.output_tokens, cost, False, reply.error
        )

    def call_many(
        self,
        requests: Sequence[LLMRequest],
        check: Callable[[LLMResult], bool] | None = None,
        monitor: StageMonitor | None = None,
        desc: str | None = None,
    ) -> list[LLMResult]:
        """Call each unique request once (at most 4 at a time per provider); keep input order.
        `check(result)` says whether a reply is usable (default: no refusal/cut-off); the
        monitor stops the stage on too many unusable replies. On any error the queued calls are
        cancelled and the error is raised; finished calls stay cached, so a rerun resumes."""
        check = check or (lambda result: not result.failed)
        monitor = monitor or StageMonitor(desc or "calls")
        unique = {request.cache_key(): request for request in requests}
        by_provider: dict[str, list[LLMRequest]] = defaultdict(list)
        for request in unique.values():
            by_provider[request.provider].append(request)
        results: dict[str, LLMResult] = {}
        counts: Counter[str] = Counter()
        with tqdm(total=len(unique), desc=desc, disable=desc is None or not unique) as bar:
            for provider_name, group in by_provider.items():
                workers = min(MAX_CONCURRENCY, self.providers[provider_name].max_workers)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures: dict[Future[LLMResult], LLMRequest] = {
                        pool.submit(self.call, request): request for request in group
                    }
                    try:
                        for future in as_completed(futures):
                            result = future.result()
                            ok = check(result)
                            counts["cached"] += result.cached
                            counts["failed"] += not ok
                            results[futures[future].cache_key()] = result
                            bar.update(1)
                            bar.set_postfix(
                                cached=counts["cached"],
                                failed=counts["failed"],
                                spend=f"${sum(self.ledger.cost.values()):.3f}",
                            )
                            monitor.observe(result, ok)
                    except BaseException:
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
        return [results[request.cache_key()] for request in requests]

    def call_json_many[Schema: BaseModel](
        self,
        requests: Sequence[LLMRequest],
        schema: type[Schema],
        desc: str | None = None,
        max_failure_rate: float | None = None,
    ) -> list[tuple[Schema | None, list[LLMResult]]]:
        """Call and parse each request; retry an invalid reply once (attempt 1), else None.
        Stops the stage when JSON failures pass `max_failure_rate` in the first 10 calls."""
        monitor = StageMonitor(desc or "json", max_failure_rate, "invalid JSON")

        def parses(result: LLMResult) -> bool:
            return parse_json_reply(result.text, schema) is not None

        first = self.call_many(requests, parses, monitor, desc)
        parsed = [parse_json_reply(result.text, schema) for result in first]
        retry_index = [i for i, value in enumerate(parsed) if value is None]
        retries = self.call_many(
            [replace(requests[i], attempt=JSON_RETRY_ATTEMPT) for i in retry_index],
            parses,
            monitor,
            f"{desc} (JSON retry)" if desc else None,
        )
        outcomes = [(value, [result]) for value, result in zip(parsed, first, strict=True)]
        for index, retry in zip(retry_index, retries, strict=True):
            outcomes[index] = (parse_json_reply(retry.text, schema), [first[index], retry])
        return outcomes

    def cached_records(self) -> list[dict[str, Any]]:
        """Every cached reply (all sessions)."""
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.cache_dir.glob("*.json"))
        ]

    def all_time_spend(self) -> dict[str, float]:
        """Total USD per provider over every cached call, including earlier sessions."""
        spend: dict[str, float] = defaultdict(float)
        for record in self.cached_records():
            spend[record["request"]["provider"]] += record["cost_usd"]
        return dict(spend)

    def recorded_usage(self) -> dict[tuple[str, str], tuple[float, float]]:
        """Per (role, model): mean input/o200k ratio and mean output tokens of cached calls."""
        samples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in self.cached_records():
            request = record["request"]
            samples[(request["role"], request["model"])].append(record)
        return {
            key: (
                mean(r["input_tokens"] / max(r["prompt_o200k_tokens"], 1) for r in records),
                mean(r["output_tokens"] for r in records),
            )
            for key, records in samples.items()
        }

    def spend_by_stage(self) -> list[dict[str, Any]]:
        """Per stage (role) and provider: paid calls, tokens and cost over all cached calls
        (all sessions), plus this session's new calls and cache hits."""
        totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        costs: dict[tuple[str, str], float] = defaultdict(float)
        for record in self.cached_records():
            key = (record["request"]["role"], record["request"]["provider"])
            totals[key].update(
                calls=1, tokens_in=record["input_tokens"], tokens_out=record["output_tokens"]
            )
            costs[key] += record["cost_usd"]
        keys = set(totals) | set(self.ledger.cached_by_stage)
        return [
            {
                "stage": role,
                "provider": provider,
                "calls (all time)": totals[(role, provider)]["calls"],
                "new this session": self.ledger.new_by_stage[(role, provider)],
                "cached this session": self.ledger.cached_by_stage[(role, provider)],
                "tokens in": totals[(role, provider)]["tokens_in"],
                "tokens out": totals[(role, provider)]["tokens_out"],
                "cost $": round(costs[(role, provider)], 4),
            }
            for role, provider in sorted(keys)
        ]

    def usage_by_role(self) -> list[dict[str, Any]]:
        """Per (role, model) over all cached calls: calls, cost, and mean/max output tokens
        against the output limit (to check that limits neither cut replies nor are far too big)."""
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in self.cached_records():
            request = record["request"]
            groups[(request["role"], request["model"])].append(record)
        return [
            {
                "role": role,
                "model": model,
                "calls": len(records),
                "cost_usd": round(sum(r["cost_usd"] for r in records), 4),
                "mean_output_tokens": round(mean(r["output_tokens"] for r in records)),
                "max_output_tokens": max(r["output_tokens"] for r in records),
                "output_limit": records[0]["request"]["max_output_tokens"],
                "cut_off": sum("max_tokens" in r["error"] or "incomplete" in r["error"]
                               for r in records),
            }
            for (role, model), records in sorted(groups.items())
        ]

    def estimate(
        self, requests: Sequence[LLMRequest], default_output_tokens: dict[str, int]
    ) -> SpendEstimate:
        """Estimate the cost of the uncached requests from o200k prompt sizes and, when
        available, the usage recorded so far for the same role and model (e.g. the dry run)."""
        recorded = self.recorded_usage()
        calls: Counter[str] = Counter()
        cached: Counter[str] = Counter()
        cost: dict[str, float] = defaultdict(float)
        unique = {r.cache_key(): r for r in requests}
        for request in unique.values():
            if self.is_cached(request):
                cached[request.provider] += 1
                continue
            input_ratio, output_tokens = recorded.get(
                (request.role, request.model), (1.0, default_output_tokens[request.role])
            )
            input_tokens = len(O200K.encode(request.prompt)) * input_ratio
            calls[request.provider] += 1
            cost[request.provider] += self.price(request).cost(input_tokens, output_tokens)
        return SpendEstimate(calls, cached, dict(cost))


def guard_spend(
    client: LLMClient, estimate: SpendEstimate, confirm_spend: bool, stage: str
) -> None:
    """Print calls to make, calls cached and the estimated cost per provider; raise
    SpendBlocked unless confirmed and within every provider's budget cap."""
    total_calls = sum(estimate.calls.values())
    print(
        f"{stage}: {total_calls} calls to make, {sum(estimate.cached.values())} cached, "
        f"est. ${estimate.total:.3f}"
    )
    for provider in sorted(set(estimate.calls) | set(estimate.cached)):
        cap = client.budget_caps.get(provider)
        print(
            f"  {provider}: {estimate.calls[provider]} to make, {estimate.cached[provider]} cached, "
            f"est. ${estimate.cost.get(provider, 0.0):.3f}, spent so far "
            f"${client.spent(provider):.2f}" + (f" of ${cap:.2f} cap" if cap is not None else "")
        )
    if estimate.total == 0:
        return
    for provider, cost in estimate.cost.items():
        cap = client.budget_caps.get(provider)
        if cap is not None and client.spent(provider) + cost > cap:
            raise SpendBlocked(f"{stage}: {provider} would pass its ${cap:.2f} cap")
    if not confirm_spend:
        raise SpendBlocked(f"{stage}: CONFIRM_SPEND is False; nothing was called")
