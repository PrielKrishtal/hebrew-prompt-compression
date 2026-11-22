"""Token counters: local tiktoken encodings and Gemini's free count_tokens endpoint."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import tiktoken
from google import genai
from google.genai import errors

BASELINE_TEXT = "a"
MAX_RETRIES = 6
RETRY_CODES = (429,)
UNAVAILABLE_CODES = (403, 404)


@dataclass(frozen=True)
class TokenCounter:
    """A model's token counter and the metadata used to group tokenizers."""

    provider: str
    model: str
    tokenizer_id: str | None
    count: Callable[[str], int]
    note: str | None = None


def make_openai_counter(model: str) -> TokenCounter:
    """Return a local tiktoken counter that records the model's tokenizer ID."""
    encoding = tiktoken.encoding_for_model(model)
    return TokenCounter(
        provider="OpenAI",
        model=model,
        tokenizer_id=encoding.name,
        count=lambda text: len(encoding.encode(text)),
    )


def with_backoff[T](
    call: Callable[[], T],
    initial_delay: float = 1.0,
    give_up: Callable[[errors.APIError], bool] = lambda error: False,
) -> T:
    """Run `call`, waiting `initial_delay`, then double that... while the API returns 429.

    A 429 for which `give_up` returns True is raised at once, without waiting.
    """
    delay = initial_delay
    for _ in range(MAX_RETRIES):
        try:
            return call()
        except errors.APIError as error:
            if error.code not in RETRY_CODES or give_up(error):
                raise
            time.sleep(delay)
            delay *= 2
    return call()


def raw_gemini_count(client: genai.Client, model: str, text: str) -> int:
    """Ask the free count_tokens endpoint, retrying on rate limits."""
    return with_backoff(
        lambda: client.models.count_tokens(model=model, contents=text).total_tokens
    )


def measure_overhead(model: str, raw_count: Callable[[str], int]) -> int:
    """Return the fixed per-request overhead of `raw_count`, asserting it is constant."""
    observed = [raw_count(" ".join([BASELINE_TEXT] * words)) for words in (1, 2, 3)]
    overhead = observed[0] - 1
    expected = [overhead + words for words in (1, 2, 3)]
    print(f"{model}: count('a'), count('a a'), count('a a a') = {observed}, expected {expected}")
    assert observed == expected, f"{model}: overhead is not constant"
    return overhead


def probe_gemini(client: genai.Client, model: str) -> dict[str, object]:
    """Probe `model` with count_tokens and check its per-request overhead is constant."""
    try:
        overhead = measure_overhead(model, partial(raw_gemini_count, client, model))
    except errors.APIError as error:
        if error.code not in UNAVAILABLE_CODES:
            raise
        reason = f"{error.code} {error.status}"
        print(f"unavailable: {model} ({reason})")
        return {"model": model, "available": "no", "overhead": None, "reason": reason}
    return {"model": model, "available": "yes", "overhead": overhead, "reason": ""}


def make_gemini_counter(
    model: str, overhead: int, raw_count: Callable[[str], int]
) -> TokenCounter:
    """Return a Gemini counter with the per-request overhead subtracted."""
    return TokenCounter(
        provider="Gemini",
        model=model,
        tokenizer_id=None,
        count=lambda text: raw_count(text) - overhead,
    )
