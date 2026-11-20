"""Local compressors for Test 3 (rules, LLMLingua-2, NLLB) and the teacher-output validator."""

import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hebrew_token_tax.wordlists import (
    HE_PREFIXES,
    has_digit,
    is_prefixed_protected,
    load_filler,
    load_protected,
)

LLMLINGUA2_LARGE = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
LLMLINGUA2_SMALL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_SOURCE = "heb_Hebr"
NLLB_TARGET = "eng_Latn"
NLLB_BATCH_SIZE = 16
NLLB_MAX_NEW_TOKENS = 256
DEVICE = "cpu"
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MILLISECONDS_PER_SECOND = 1000


def timed[T](function: Callable[[], T]) -> tuple[T, float]:
    """Run `function` and return its result and the wall-clock time in milliseconds."""
    started = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - started) * MILLISECONDS_PER_SECOND


def is_punctuation(char: str) -> bool:
    """True for Unicode punctuation characters."""
    return unicodedata.category(char).startswith("P")


def split_punctuation(token: str) -> tuple[str, str, str]:
    """Split a token into (leading punctuation, body, trailing punctuation)."""
    start, end = 0, len(token)
    while start < end and is_punctuation(token[start]):
        start += 1
    while end > start and is_punctuation(token[end - 1]):
        end -= 1
    return token[:start], token[start:end], token[end:]


def lookup_form(word: str, language: str) -> str:
    """Strip surrounding punctuation (and lowercase English) to look a word up in a list."""
    body = split_punctuation(word)[1]
    return body.lower() if language == "en" else body


class RuleCompressor:
    """Drop filler words; keep protected words, prefixed protected words (Hebrew) and digits."""

    def __init__(self, language: str, data_dir: Path) -> None:
        self.language = language
        self.filler = load_filler(data_dir / f"filler_{language}.txt")
        self.protected = load_protected(data_dir / f"protected_{language}.txt", language)

    def is_dropped(self, word: str) -> bool:
        """True if `word` is filler and nothing protects it."""
        if has_digit(word):
            return False
        form = lookup_form(word, self.language)
        if form not in self.filler or form in self.protected:
            return False
        return not (self.language == "he" and is_prefixed_protected(form, self.protected))

    def compress(self, text: str) -> str:
        """Return `text` without its filler words, keeping the line structure."""
        lines = text.split("\n")
        kept = (" ".join(w for w in line.split() if not self.is_dropped(w)) for line in lines)
        return "\n".join(kept)


class LLMLingua2Compressor:
    """LLMLingua-2 on CPU; `rate` is the fraction of tokens to keep. Loads on first use."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._compressor: Any = None

    def _load(self) -> Any:
        if self._compressor is None:
            from llmlingua import PromptCompressor

            self._compressor = PromptCompressor(
                model_name=self.model_name, use_llmlingua2=True, device_map=DEVICE
            )
        return self._compressor

    def compress(self, text: str, rate: float) -> str:
        """Return LLMLingua-2's compression of `text` keeping about `rate` of its tokens."""
        result = self._load().compress_prompt(text, rate=rate)
        return result["compressed_prompt"]


def split_sentences(line: str) -> list[str]:
    """Split one line into sentences at . ! ? followed by whitespace."""
    return [sentence for sentence in SENTENCE_END.split(line.strip()) if sentence]


class NllbTranslator:
    """NLLB-200 Hebrew to English on CPU, sentence by sentence. Loads on first use."""

    def __init__(self, model_name: str = NLLB_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=NLLB_SOURCE)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(DEVICE)
            self._model.eval()
        return self._model, self._tokenizer

    def translate_sentences(self, sentences: list[str]) -> list[str]:
        """Translate each sentence independently, in batches."""
        import torch

        model, tokenizer = self._load()
        target_id = tokenizer.convert_tokens_to_ids(NLLB_TARGET)
        translations = []
        for start in range(0, len(sentences), NLLB_BATCH_SIZE):
            batch = sentences[start:start + NLLB_BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, forced_bos_token_id=target_id, max_new_tokens=NLLB_MAX_NEW_TOKENS
                )
            translations.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        return translations

    def translate(self, text: str) -> str:
        """Translate `text` sentence by sentence, keeping its line structure."""
        lines = text.split("\n")
        sentences_per_line = [split_sentences(line) for line in lines]
        flat = [sentence for sentences in sentences_per_line for sentence in sentences]
        translated = iter(self.translate_sentences(flat)) if flat else iter(())
        return "\n".join(
            " ".join(next(translated) for _ in sentences) for sentences in sentences_per_line
        )


def is_subsequence(part: str, whole: str) -> bool:
    """True if the characters of `part` appear in `whole` in order."""
    remaining = iter(whole)
    return all(char in remaining for char in part)


def body_variants(body: str, language: str) -> set[str]:
    """The body itself plus, in Hebrew, the body with 1, 2, ... leading prefixes stripped."""
    variants = {body}
    if language == "he":
        for index, letter in enumerate(body[:-1]):
            if letter not in HE_PREFIXES:
                break
            variants.add(body[index + 1:])
    return variants


def token_matches(kept: str, original: str, language: str) -> bool:
    """True if `kept` is `original` with attached punctuation and (Hebrew) prefixes dropped."""
    kept_lead, kept_body, kept_trail = split_punctuation(kept)
    lead, body, trail = split_punctuation(original)
    if not kept_body:
        return not body and is_subsequence(kept, original)
    return (
        kept_body in body_variants(body, language)
        and is_subsequence(kept_lead, lead)
        and is_subsequence(kept_trail, trail)
    )


def first_violation(original: str, output: str, language: str) -> str | None:
    """Return the first output token that breaks the in-order deletion rule, or None if valid."""
    original_tokens = original.split()
    position = 0
    for token in output.split():
        while position < len(original_tokens) and not token_matches(
            token, original_tokens[position], language
        ):
            position += 1
        if position == len(original_tokens):
            return token
        position += 1
    return None


def is_valid_deletion(original: str, output: str, language: str) -> bool:
    """True if `output` is non-empty and only deletes from `original` (spec validator)."""
    return bool(output.strip()) and first_violation(original, output, language) is None
