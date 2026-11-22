"""Local compressors for Test 3 (rules, LLMLingua-2, NLLB) and the teacher-output validator."""

import re
import time
import unicodedata
from collections.abc import Callable, Sequence
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


def word_form(token: str, language: str) -> str:
    """A token without attached punctuation (rule A); English is lowercased (rule B).
    Punctuation-only tokens give "" and are ignored."""
    body = split_punctuation(token)[1]
    return body.lower() if language == "en" else body


def leading_prefixes(word: str) -> set[str]:
    """The Hebrew prefix-letter strings a word starts with, e.g. ובכלל -> {ו, וב}."""
    prefixes = set()
    for index, letter in enumerate(word[:-1]):
        if letter not in HE_PREFIXES:
            break
        prefixes.add(word[: index + 1])
    return prefixes


def body_variants(body: str, language: str) -> set[str]:
    """The body itself plus, in Hebrew, the body with 1, 2, ... leading prefixes stripped."""
    variants = {body}
    if language == "he":
        variants |= {body[len(prefix):] for prefix in leading_prefixes(body)}
    return variants


def word_matches(kept: str, original: str, deleted: Sequence[str], language: str) -> bool:
    """True if `kept` is `original` with (Hebrew) leading prefixes dropped, or - rule C - with
    prefix letters moved onto it from a word in the deleted run directly before it."""
    variants = body_variants(original, language)
    if kept in variants:
        return True
    if language != "he":
        return False
    moved = {prefix for word in deleted for prefix in leading_prefixes(word)}
    return any(kept.startswith(p) and kept[len(p):] in variants for p in moved)


def first_violation(original: str, output: str, language: str) -> str | None:
    """Return the first output word that breaks the in-order deletion rule, or None if valid.
    Rules: words only deleted, order kept; A) attached punctuation may change and
    punctuation-only tokens are ignored; B) English is compared case-insensitively;
    C) Hebrew may move prefix letters from the deleted run directly before a kept word."""
    original_words = [w for w in (word_form(t, language) for t in original.split()) if w]
    position = 0
    for token in output.split():
        kept = word_form(token, language)
        if not kept:
            continue
        start = position
        while position < len(original_words) and not word_matches(
            kept, original_words[position], original_words[start:position], language
        ):
            position += 1
        if position == len(original_words):
            return token
        position += 1
    return None


def is_valid_deletion(original: str, output: str, language: str) -> bool:
    """True if `output` keeps at least one word and only deletes from `original`."""
    has_words = any(word_form(token, language) for token in output.split())
    return has_words and first_violation(original, output, language) is None
