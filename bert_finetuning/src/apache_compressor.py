import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# MODULAR HARDWARE PROFILES (ALL APACHE 2.0 MODELS)
# ==============================================================================
PROFILES: dict[str, dict] = {
    # CPU-only laptop: lightweight Hebrew-native model for quick smoke tests
    "laptop": {
        "model_name": "dicta-il/DictaLM-3.0-1.7B-Instruct",
        "use_4bit": False,
    },
    # Main PC (RTX 4060 8GB): fast, low VRAM (~3 GB), solid instruction following
    "pc-fast": {
        "model_name": "Qwen/Qwen3-4B",
        "use_4bit": True,
    },
    # Main PC (RTX 4060 8GB): strongest strict extractive deletion (~5.5 GB VRAM)
    "pc-strong": {
        "model_name": "Qwen/Qwen3-8B",
        "use_4bit": True,
    },
    # Main PC (RTX 4060 8GB): strong Hebrew-native alternative (~4.8 GB VRAM)
    "pc-hebrew": {
        "model_name": "dicta-il/dictalm2.0-instruct",
        "use_4bit": True,
    },
}


@dataclass
class CompressionConfig:
    model_name: str
    use_4bit: bool = False
    min_drop_ratio: float = 0.15
    max_drop_ratio: float = 0.50
    force_drop_fillers: bool = False
    max_prompt_examples: int = 50


def normalize_word(word: str) -> str:
    """Strips leading/trailing punctuation while keeping Hebrew letters & digits."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", word)


def resolve_wordlist_path(explicit_path: Path | None, filename: str) -> Path | None:
    """Finds a wordlist via explicit path, bert_finetuning/data/, or feasibility_test/."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path("bert_finetuning/data") / filename,
        *Path("feasibility_test").rglob(filename),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_word_list(path: Path | None) -> tuple[set[str], list[list[str]]]:
    """
    Loads a UTF-8 word list (one word or multi-word phrase per line).
    Ignores blank lines and '#' comments.
    Returns (single_words_set, multi_word_phrases_list).
    """
    single_words: set[str] = set()
    phrases: list[list[str]] = []
    if path is None or not path.exists():
        return single_words, phrases

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [normalize_word(w) for w in line.split() if normalize_word(w)]
            if len(parts) == 1:
                single_words.add(parts[0])
            elif len(parts) > 1:
                phrases.append(parts)
    return single_words, phrases


class ApacheCompressor:
    """Modular extractive Hebrew prompt compressor backed by an Apache 2.0 LLM."""

    def __init__(
        self,
        config: CompressionConfig,
        protected_path: Path | None = None,
        fillers_path: Path | None = None,
    ):
        self.config = config
        self.protected_words, self.protected_phrases = load_word_list(protected_path)
        self.filler_words, self.filler_phrases = load_word_list(fillers_path)

        print(
            f"Loaded {len(self.protected_words)} protected words "
            f"({len(self.protected_phrases)} phrases) | "
            f"{len(self.filler_words)} filler words "
            f"({len(self.filler_phrases)} phrases)"
        )

        self.system_prompt = self._build_system_prompt()
        self.tokenizer, self.model = self._load_model()

    def _build_system_prompt(self) -> str:
        filler_preview = sorted(self.filler_words)[: self.config.max_prompt_examples]
        filler_preview += [" ".join(p) for p in self.filler_phrases[:15]]
        protected_preview = sorted(self.protected_words)[: self.config.max_prompt_examples]

        filler_hint = (
            f"4. Prioritize deleting conversational fillers such as: {', '.join(filler_preview)}.\n"
            if filler_preview
            else ""
        )
        protected_hint = (
            f"5. NEVER delete numbers, dates, names, or protected words such as: {', '.join(protected_preview)}.\n"
            if protected_preview
            else "5. NEVER delete numbers, dates, names, or negations.\n"
        )

        return (
            "You are a strict extractive Hebrew text compressor. "
            "Delete 20% to 40% of non-critical filler or redundant words while preserving "
            "the exact meaning and all factual details.\n"
            "RULES:\n"
            "1. ONLY delete whole words. NEVER rewrite, paraphrase, reorder, or add words.\n"
            "2. Keep the exact spelling of every word you keep.\n"
            "3. Output ONLY the compressed Hebrew text with no quotes or explanation.\n"
            f"{filler_hint}"
            f"{protected_hint}"
        )

    def _load_model(self):
        has_cuda = torch.cuda.is_available()
        use_4bit = self.config.use_4bit and has_cuda
        print(
            f"Loading Apache 2.0 Teacher: '{self.config.model_name}' | "
            f"Device: {'CUDA GPU' if has_cuda else 'CPU'} | 4-bit: {use_4bit}"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        if use_4bit:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                quantization_config=quant_cfg,
            )
        elif has_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
            )
            model.to("cpu")

        model.eval()
        return tokenizer, model

    def generate_compressed_text(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        text_token_len = len(self.tokenizer.encode(text, add_special_tokens=False))
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max(48, text_token_len + 24),
                do_sample=False,
                pad_token_id=pad_id,
            )

        raw = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        if not raw:
            return ""
        return raw.splitlines()[0].strip().strip("\"'`")

    def _apply_phrase_rule(self, orig_norm: list[str], labels: list[int], phrases: list[list[str]], value: int):
        n = len(orig_norm)
        for phrase in phrases:
            p_len = len(phrase)
            for i in range(n - p_len + 1):
                if orig_norm[i : i + p_len] == phrase:
                    for j in range(i, i + p_len):
                        labels[j] = value

    def align_and_label(self, original_text: str, compressed_text: str) -> dict | None:
        """
        Aligns original vs. compressed words with difflib.
        Rejects rewrites/insertions and enforces filler/protected rules + 15%-50% bounds.
        """
        orig_words = original_text.split()
        comp_words = compressed_text.split()
        if not orig_words or not comp_words or len(comp_words) >= len(orig_words):
            return None

        orig_norm = [normalize_word(w) for w in orig_words]
        comp_norm = [normalize_word(w) for w in comp_words]

        matcher = difflib.SequenceMatcher(None, orig_norm, comp_norm, autojunk=False)
        labels = [0] * len(orig_words)
        rewritten = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for idx in range(i1, i2):
                    labels[idx] = 1
            elif tag == "delete":
                for idx in range(i1, i2):
                    labels[idx] = 0
            else:
                rewritten += max(i2 - i1, j2 - j1)

        # Reject if the model paraphrased or inserted any words
        if rewritten > 0:
            return None

        # Optional: force-drop configured filler words/phrases
        if self.config.force_drop_fillers:
            for idx, clean in enumerate(orig_norm):
                if clean in self.filler_words:
                    labels[idx] = 0
            self._apply_phrase_rule(orig_norm, labels, self.filler_phrases, value=0)

        # Always enforce protected guardrails LAST so protected.txt overrides everything
        for idx, clean in enumerate(orig_norm):
            if clean in self.protected_words or any(ch.isdigit() for ch in clean):
                labels[idx] = 1
        self._apply_phrase_rule(orig_norm, labels, self.protected_phrases, value=1)

        drop_ratio = labels.count(0) / len(orig_words)
        if not (self.config.min_drop_ratio <= drop_ratio <= self.config.max_drop_ratio):
            return None

        return {
            "tokens": orig_words,
            "labels": labels,
            "drop_ratio": round(drop_ratio, 3),
            "original_text": original_text,
            "compressed_text": " ".join(w for w, lbl in zip(orig_words, labels) if lbl == 1),
        }

    def compress_and_label(self, text: str) -> dict | None:
        compressed = self.generate_compressed_text(text)
        return self.align_and_label(text, compressed)