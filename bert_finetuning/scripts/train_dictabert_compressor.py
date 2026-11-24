import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "dicta-il/dictabert"
DEFAULT_DATA_PATH = Path("bert_finetuning/data/processed/train_dataset.json")
DEFAULT_PROTECTED_PATH = Path("bert_finetuning/data/protected.txt")
DEFAULT_OUTPUT_DIR = Path("bert_finetuning/models/dictabert_compressor")
TARGET_GPT_MODEL = "gpt-4o"

ID2LABEL = {0: "DROP", 1: "KEEP"}
LABEL2ID = {"DROP": 0, "KEEP": 1}

SMOKE_TEST_DATA = [
    {
        "tokens": ["שלום", "היי", "רציתי", "בבקשה", "לבדוק", "מה", "היתרה", "שלי", "בחשבון", "הבנק", "היום"],
        "labels": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    },
    {
        "tokens": ["אהלן", "אממ", "אפשר", "בבקשה", "להזמין", "משלוח", "של", "פיצה", "משפחתית", "לרחוב", "הרצל", "15"],
        "labels": [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    },
    {
        "tokens": ["סליחה", "שאני", "מפריע", "אבל", "הכרטיס", "שלי", "לא", "עובד", "בכספומט", "מאז", "אתמול"],
        "labels": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    },
    {
        "tokens": ["היי", "תודה", "רבה", "מראש", "אני", "צריך", "לבטל", "את", "התור", "לפקיד", "ביום", "שלישי"],
        "labels": [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    },
    {
        "tokens": ["תשמע", "בעצם", "השליח", "עדיין", "לא", "הגיע", "והזמנה", "מספר", "4821", "מתעכבת", "שעה"],
        "labels": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    {
        "tokens": ["בוקר", "טוב", "רציתי", "רק", "לשאול", "האם", "אין", "עמלה", "על", "העברה", "בנקאית", "בסכום", "500", "שקל"],
        "labels": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    },
    {
        "tokens": ["היי", "כאילו", "רציתי", "לעדכן", "את", "כתובת", "המשלוח", "שלי", "לתל", "אביב", "דירה", "4"],
        "labels": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    },
    {
        "tokens": ["שלום", "אשמח", "מאוד", "אם", "תוכל", "להגדיל", "לי", "את", "מסגרת", "האשראי", "ל", "20000"],
        "labels": [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
    },
]


def normalize_word(word: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", word)


def load_protected_words(path: Path) -> tuple[set[str], list[list[str]]]:
    single_words: set[str] = set()
    phrases: list[list[str]] = []
    if not path.exists():
        print(f"Warning: protected list not found at '{path}'. Protecting digits only.")
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


def load_training_data(data_path: Path, smoke_test: bool) -> list[dict]:
    if smoke_test:
        print("Running in --smoke-test mode with 8 built-in Hebrew samples.")
        return SMOKE_TEST_DATA

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Run build_dataset.py first or pass --smoke-test."
        )

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) < 4:
        raise ValueError(
            f"Dataset at '{data_path}' has only {len(data)} samples (need at least 4 to train/validate). "
            "Run build_dataset.py with --profile pc-strong on your RTX 4060 PC, or test with --smoke-test."
        )

    cleaned = []
    for idx, item in enumerate(data):
        tokens, labels = item.get("tokens", []), item.get("labels", [])
        if len(tokens) == len(labels) and len(tokens) > 0:
            cleaned.append({"tokens": tokens, "labels": labels})
        else:
            print(f"Skipping malformed row {idx} (token/label length mismatch).")
    return cleaned


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True,
    )

    aligned_labels = []
    for i, word_labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == prev_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(word_labels[word_idx])
            prev_word_idx = word_idx
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)

    valid_preds, valid_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                valid_preds.append(p)
                valid_labels.append(l)

    accuracy = float(np.mean(np.array(valid_preds) == np.array(valid_labels)))
    keep_rate = float(np.mean(np.array(valid_preds) == 1))
    return {"token_accuracy": accuracy, "predicted_keep_rate": keep_rate}


def get_protected_indices(
    words: list[str],
    protected_words: set[str],
    protected_phrases: list[list[str]],
) -> set[int]:
    norm_words = [normalize_word(w) for w in words]
    protected_idx: set[int] = set()

    for idx, clean in enumerate(norm_words):
        if clean in protected_words or any(ch.isdigit() for ch in clean):
            protected_idx.add(idx)

    n = len(norm_words)
    for phrase in protected_phrases:
        p_len = len(phrase)
        for i in range(n - p_len + 1):
            if norm_words[i : i + p_len] == phrase:
                protected_idx.update(range(i, i + p_len))

    return protected_idx


def compress_text(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    gpt_tokenizer,
    protected_words: set[str],
    protected_phrases: list[list[str]],
    keep_threshold: float = 0.5,
) -> dict:
    start = time.perf_counter()
    words = text.split()

    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    word_ids = encoded.word_ids(batch_index=0)
    inputs = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        keep_probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

    word_scores = {}
    for tok_idx, w_idx in enumerate(word_ids):
        if w_idx is not None and w_idx not in word_scores:
            word_scores[w_idx] = float(keep_probs[tok_idx])

    protected_idx = get_protected_indices(words, protected_words, protected_phrases)
    kept_words = []
    decisions = []

    for idx, word in enumerate(words):
        score = word_scores.get(idx, 1.0)
        is_prot = idx in protected_idx
        keep = is_prot or (score >= keep_threshold)
        if keep:
            kept_words.append(word)
        status = "PROTECTED" if is_prot else ("KEEP" if keep else "DROP")
        decisions.append((word, round(score, 2), status))

    compressed = " ".join(kept_words)
    latency_ms = (time.perf_counter() - start) * 1000

    orig_tokens = len(gpt_tokenizer.encode(text))
    comp_tokens = len(gpt_tokenizer.encode(compressed))
    savings_pct = (1 - (comp_tokens / orig_tokens)) * 100 if orig_tokens > 0 else 0.0

    return {
        "original_text": text,
        "compressed_text": compressed,
        "orig_tokens": orig_tokens,
        "comp_tokens": comp_tokens,
        "savings_pct": savings_pct,
        "latency_ms": latency_ms,
        "decisions": decisions,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DictaBERT for extractive Hebrew prompt compression.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to train_dataset.json.")
    parser.add_argument("--protected", type=Path, default=DEFAULT_PROTECTED_PATH, help="Path to protected.txt.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save model weights.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs (default: 4).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per device.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Keep probability threshold for test inference.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick test with 8 built-in Hebrew samples.")
    return parser.parse_args()


def main():
    args = parse_args()
    use_gpu = torch.cuda.is_available()
    batch_size = args.batch_size or (8 if use_gpu else 4)
    epochs = 6 if args.smoke_test and args.epochs == 4 else args.epochs

    raw_data = load_training_data(args.data, args.smoke_test)
    protected_words, protected_phrases = load_protected_words(args.protected)
    print(f"Loaded {len(raw_data)} training examples | Protected: {len(protected_words)} words, {len(protected_phrases)} phrases")

    dataset = Dataset.from_list(raw_data)
    split = dataset.train_test_split(test_size=0.2, seed=42)

    print(f"Loading tokenizer and base model: {MODEL_NAME} | Device: {'CUDA GPU' if use_gpu else 'CPU'}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = split.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer),
        batched=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=use_gpu,
        logging_steps=max(1, len(split["train"]) // batch_size),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n--- Starting DictaBERT Fine-Tuning ---")
    trainer.train()

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nSaved fine-tuned DictaBERT model to '{args.output_dir}'.")

    # Quick inference verification
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)
    model.eval()
    gpt_tokenizer = tiktoken.encoding_for_model(TARGET_GPT_MODEL)

    sample = "שלום היי רציתי בבקשה לבדוק למה המשלוח שלי מספר 9042 עדיין לא הגיע היום"
    result = compress_text(
        sample,
        model,
        tokenizer,
        device,
        gpt_tokenizer,
        protected_words,
        protected_phrases,
        keep_threshold=args.threshold,
    )

    print("\n--- Sample Inference Check ---")
    print(f"Original ({result['orig_tokens']} GPT-4o tokens):   {result['original_text']}")
    print(f"Compressed ({result['comp_tokens']} GPT-4o tokens): {result['compressed_text']}")
    print(f"Token Savings: {result['savings_pct']:.1f}% | Latency: {result['latency_ms']:.1f} ms")
    for word, score, status in result["decisions"]:
        print(f"  {status:9s} | P(keep)={score:.2f} | {word}")


if __name__ == "__main__":
    main()