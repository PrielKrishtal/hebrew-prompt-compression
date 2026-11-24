import argparse
import random
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 1. HARDWARE PROFILES (ALL APACHE 2.0 MODELS)
# ==============================================================================
PROFILES = {
    # CPU-only laptop: lightweight Hebrew-native model for small tests
    "laptop": {
        "model_name": "dicta-il/DictaLM-3.0-1.7B-Instruct",
        "use_4bit": False,
        "items_per_call": 3,
        "max_new_tokens": 240,
    },
    # Main PC with RTX 4060 (8GB VRAM): fast, natural Hebrew in bfloat16 (~3.4GB VRAM)
    "rtx4060": {
        "model_name": "dicta-il/DictaLM-3.0-1.7B-Instruct",
        "use_4bit": False,
        "items_per_call": 5,
        "max_new_tokens": 380,
    },
    # Optional higher-diversity profile for RTX 4060 (requires: pip install bitsandbytes)
    "rtx4060-qwen4b": {
        "model_name": "Qwen/Qwen3-4B",
        "use_4bit": True,
        "items_per_call": 5,
        "max_new_tokens": 380,
    },
}

DEFAULT_OUTPUT_CSV = Path("bert_finetuning/data/raw/synthetic_interactions_500.csv")
MIN_WORDS = 10
MAX_WORDS = 55
RANDOM_SEED = 42

# ==============================================================================
# 2. SCENARIO & STYLE DIVERSITY MATRIX
# ==============================================================================
DOMAINS = {
    "banking": [
        "checking account balance and a suspicious charge",
        "credit card blocked at an ATM",
        "canceling or rescheduling an appointment with a bank teller",
        "requesting a credit limit increase",
        "disputing a monthly fee or commission",
        "updating a standing order or bank transfer",
        "asking about mortgage or loan payment dates",
    ],
    "delivery": [
        "food delivery running late with a specific order number",
        "missing or wrong item in a restaurant delivery",
        "updating the delivery street address and apartment number",
        "courier marked package as delivered but it never arrived",
        "canceling an active delivery before dispatch",
        "scheduling a courier pickup window for a parcel",
    ],
    "tech_support": [
        "home internet router disconnecting repeatedly",
        "cannot log in to account even after resetting password",
        "mobile app crashing during checkout",
        "error code appearing on smart TV or set-top box",
        "requesting a technician visit to fix a fiber line",
        "SIM card or eSIM not activating on a new phone",
    ],
    "ecommerce_billing": [
        "requesting a refund for a damaged product",
        "double charge appearing on an invoice",
        "tracking a delayed online order shipment",
        "returning an item bought on a specific date",
        "coupon code not applying to the final price",
    ],
    "appointments_and_services": [
        "rescheduling a clinic or service appointment",
        "canceling a monthly gym or subscription plan",
        "updating personal contact details and phone number",
        "asking why a service request is still pending",
    ],
}

SPEAKER_STYLES = [
    "polite and wordy, starting with casual greetings and filler phrases",
    "slightly frustrated and conversational, using everyday spoken fillers",
    "rushed customer typing quickly in chat with informal phrasing",
    "detailed customer giving dates, amounts, and explicit negations",
    "hesitant customer asking a follow-up question with conversational padding",
]

SYSTEM_PROMPT = (
    "You are a synthetic data generator creating realistic Israeli customer service "
    "messages written strictly in natural, everyday spoken Hebrew. "
    "Each message must be a single paragraph (1 to 3 sentences, 12 to 40 words) "
    "as typed or spoken by a human customer to a receptionist, support agent, or bot.\n"
    "REQUIREMENTS:\n"
    "1. Write ONLY in Hebrew (digits and currency numbers are allowed).\n"
    "2. Include natural conversational greetings, politeness, and filler words so "
    "the text reflects real human speech.\n"
    "3. Include concrete facts such as order numbers, dates, times, amounts, street names, "
    "or negations (e.g., something did not arrive or does not work).\n"
    "4. Output ONLY the requested messages, one per line, with NO numbering, NO bullet "
    "points, NO English text, and NO explanations."
)


# ==============================================================================
# 3. MODEL LOADING & VALIDATION HELPERS
# ==============================================================================
def load_generator(model_name: str, use_4bit: bool):
    has_cuda = torch.cuda.is_available()
    print(f"Loading '{model_name}' | Device: {'CUDA GPU' if has_cuda else 'CPU'} | 4-bit: {use_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_4bit and has_cuda:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
        )
    elif has_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model.to("cpu")

    model.eval()
    return tokenizer, model


def clean_line(line: str) -> str:
    """Strips list numbering, bullet markers, quotes, and extra whitespace."""
    cleaned = line.strip()
    cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[\.\)\-:])\s*", "", cleaned)
    cleaned = cleaned.strip("\"'`")
    return " ".join(cleaned.split())


def is_valid_hebrew_message(text: str) -> bool:
    """Verifies word count bounds and ensures the line is predominantly Hebrew."""
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False

    hebrew_chars = len(re.findall(r"[\u0590-\u05FF]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    return hebrew_chars >= 20 and latin_chars <= 6


def generate_batch(
    tokenizer,
    model,
    domain: str,
    scenario: str,
    style: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    user_prompt = (
        f"Generate {batch_size} completely different customer inquiries in natural spoken Hebrew.\n"
        f"Domain: {domain}\n"
        f"Scenario: {scenario}\n"
        f"Tone/Style: {style}\n"
        f"Make sure each line uses different names, numbers, dates, and phrasing. "
        f"Output exactly {batch_size} lines in Hebrew only."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=pad_id,
        )

    raw_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return [
        cleaned
        for raw_line in raw_text.splitlines()
        if is_valid_hebrew_message(cleaned := clean_line(raw_line))
    ]


# ==============================================================================
# 4. MAIN LOOP WITH CHECKPOINTING & RESUME
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Hebrew customer interactions.")
    parser.add_argument(
        "--profile",
        choices=["auto", "laptop", "rtx4060", "rtx4060-qwen4b"],
        default="auto",
        help="Hardware profile (default: auto-detects GPU vs CPU).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=500,
        help="Total number of synthetic Hebrew rows to collect (default: 500).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    profile_name = ("rtx4060" if torch.cuda.is_available() else "laptop") if args.profile == "auto" else args.profile
    cfg = PROFILES[profile_name]
    print(f"Selected profile: '{profile_name}' -> {cfg['model_name']}")

    random.seed(RANDOM_SEED)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing CSV if present
    records = []
    seen_texts = set()
    if args.output.exists():
        existing_df = pd.read_csv(args.output)
        for row in existing_df.to_dict("records"):
            txt = str(row["Hebrew_Text"]).strip()
            if txt and txt not in seen_texts:
                seen_texts.add(txt)
                records.append(row)
        print(f"Resumed {len(records)} existing samples from '{args.output}'.")

    if len(records) >= args.target:
        print(f"Target of {args.target} samples already reached.")
        return

    tokenizer, model = load_generator(cfg["model_name"], cfg["use_4bit"])
    domain_keys = list(DOMAINS.keys())
    call_count = 0
    max_calls = max(10, (args.target // cfg["items_per_call"]) * 4)

    print(f"\n--- Generating Synthetic Hebrew Interactions (Target: {args.target}) ---")
    while len(records) < args.target and call_count < max_calls:
        call_count += 1
        domain = random.choice(domain_keys)
        scenario = random.choice(DOMAINS[domain])
        style = random.choice(SPEAKER_STYLES)

        batch = generate_batch(
            tokenizer=tokenizer,
            model=model,
            domain=domain,
            scenario=scenario,
            style=style,
            batch_size=cfg["items_per_call"],
            max_new_tokens=cfg["max_new_tokens"],
        )

        added = 0
        for text in batch:
            if text not in seen_texts and len(records) < args.target:
                seen_texts.add(text)
                records.append({
                    "Domain": domain,
                    "Scenario": scenario,
                    "Style": style,
                    "Hebrew_Text": text,
                })
                added += 1

        if call_count % 3 == 0 or len(records) >= args.target:
            pd.DataFrame(records).to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"Call {call_count} | +{added} valid | Total: {len(records)}/{args.target}")

    pd.DataFrame(records).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nDone! Saved {len(records)} synthetic interactions to '{args.output}'.")


if __name__ == "__main__":
    main()