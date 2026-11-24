import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

# Allow importing from bert_finetuning/src/ regardless of working directory
BERT_ROOT = Path(__file__).resolve().parents[1]
if str(BERT_ROOT) not in sys.path:
    sys.path.insert(0, str(BERT_ROOT))

from src.apache_compressor import (
    PROFILES,
    ApacheCompressor,
    CompressionConfig,
    resolve_wordlist_path,
)

DEFAULT_INPUT_CSV = Path("bert_finetuning/data/raw/synthetic_interactions_500.csv")
DEFAULT_OUTPUT_JSON = Path("bert_finetuning/data/processed/train_dataset.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Compress Hebrew texts with an Apache 2.0 Teacher and produce 0/1 labels.")
    parser.add_argument(
        "--profile",
        choices=["auto", *PROFILES.keys()],
        default="auto",
        help="Hardware profile: laptop, pc-fast, pc-strong, pc-hebrew, or auto.",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional custom Hugging Face Apache 2.0 model ID.")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true", help="Force 4-bit quantization (with --model).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="Input CSV with a 'Hebrew_Text' column.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output JSON path.")
    parser.add_argument("--protected", type=Path, default=None, help="Path to protected.txt (auto-detected if omitted).")
    parser.add_argument("--fillers", type=Path, default=None, help="Path to fillerwords.txt (auto-detected if omitted).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (useful for quick tests).")
    parser.add_argument("--force-drop-fillers", action="store_true", help="Always label fillerwords.txt matches as 0 (unless protected).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Select hardware profile
    if args.model:
        model_name, use_4bit = args.model, args.use_4bit
        profile_label = "custom"
    else:
        profile_label = ("pc-strong" if torch.cuda.is_available() else "laptop") if args.profile == "auto" else args.profile
        model_name = PROFILES[profile_label]["model_name"]
        use_4bit = PROFILES[profile_label]["use_4bit"]

    protected_path = resolve_wordlist_path(args.protected, "protected.txt")
    fillers_path = resolve_wordlist_path(args.fillers, "fillerwords.txt")

    print(f"Profile: {profile_label} ({model_name})")
    print(f"protected.txt:   {protected_path or 'NOT FOUND (digits still protected)'}")
    print(f"fillerwords.txt: {fillers_path or 'NOT FOUND'}")

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    sentences = df["Hebrew_Text"].dropna().astype(str).str.strip().tolist()
    if args.limit:
        sentences = sentences[: args.limit]

    # Resume existing JSON if present
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled_dataset: list[dict] = []
    processed_originals: set[str] = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            labeled_dataset = json.load(f)
        processed_originals = {item["original_text"] for item in labeled_dataset}
        print(f"Resumed {len(labeled_dataset)} already-labeled samples from '{args.output}'.")

    config = CompressionConfig(
        model_name=model_name,
        use_4bit=use_4bit,
        min_drop_ratio=0.15,
        max_drop_ratio=0.50,
        force_drop_fillers=args.force_drop_fillers,
    )
    compressor = ApacheCompressor(config, protected_path=protected_path, fillers_path=fillers_path)

    rejected = 0
    print(f"\n--- Compressing & Labeling {len(sentences)} Sentences ---")
    for idx, text in enumerate(sentences, start=1):
        if text in processed_originals:
            continue

        sample = compressor.compress_and_label(text)
        if sample is not None:
            labeled_dataset.append(sample)
            processed_originals.add(text)
        else:
            rejected += 1

        if idx % 10 == 0 or idx == len(sentences):
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(labeled_dataset, f, ensure_ascii=False, indent=2)
            print(f"Progress: {idx}/{len(sentences)} | Accepted: {len(labeled_dataset)} | Rejected: {rejected}")

    print(f"\nDone! Saved {len(labeled_dataset)} clean samples to '{args.output}'.")


if __name__ == "__main__":
    main()