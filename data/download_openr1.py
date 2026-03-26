"""
Download open-r1/OpenR1-Math-220k from HuggingFace and convert to
the JSON format expected by dataset.py:
  [{"question": str, "steps": [str, ...], "answer": str}, ...]

Usage:
  python download_openr1.py                       # default config, all rows
  python download_openr1.py --config extended      # extended config
  python download_openr1.py --max_samples 10000    # limit samples
  python download_openr1.py --split_ratio 0.95     # 95% train, 5% test
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def pick_correct_generation(row):
    """Pick the first correct generation for a row, or None if none are correct."""
    correctness = row.get("correctness_math_verify") or []
    generations = row.get("generations") or []
    for i, (gen, correct) in enumerate(zip(generations, correctness)):
        if correct:
            return gen
    return None


def split_into_steps(solution_text: str) -> list[str]:
    """Split a reasoning trace into steps.

    Strategy: split on double-newlines (paragraphs) first, then fall back to
    single newlines if there's only one chunk. Strip the <think>...</think>
    wrapper if present and keep only the final answer portion outside it.
    """
    # Remove <think>...</think> tags but keep the content
    text = re.sub(r"<think>|</think>", "", solution_text).strip()

    # Split on double newlines first
    chunks = [c.strip() for c in re.split(r"\n\n+", text) if c.strip()]

    # If only one chunk, try single newlines
    if len(chunks) <= 1:
        chunks = [c.strip() for c in text.split("\n") if c.strip()]

    # Filter out very short fragments (< 5 chars) unless it's the only chunk
    if len(chunks) > 1:
        chunks = [c for c in chunks if len(c) >= 5]

    return chunks if chunks else [text]


def extract_answer(row) -> str:
    """Extract a clean final answer string from the row."""
    answer = str(row.get("answer", "")).strip()
    # Remove LaTeX boxing if present: \boxed{...}
    m = re.search(r"\\boxed\{(.+?)\}", answer)
    if m:
        answer = m.group(1)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Download & convert OpenR1-Math-220k")
    parser.add_argument("--config", default="default", choices=["default", "extended", "all"],
                        help="Dataset config (default: 'default', ~94k samples)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to keep (after filtering)")
    parser.add_argument("--split_ratio", type=float, default=0.95,
                        help="Train/test split ratio (default 0.95)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data", help="Output directory")
    args = parser.parse_args()

    print(f"Downloading open-r1/OpenR1-Math-220k (config={args.config})...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", args.config, split="train")
    print(f"Downloaded {len(ds)} rows.")

    # Convert
    converted = []
    skipped = 0
    for row in ds:
        gen = pick_correct_generation(row)
        if gen is None:
            skipped += 1
            continue

        answer = extract_answer(row)
        if not answer:
            skipped += 1
            continue

        steps = split_into_steps(gen)
        converted.append({
            "question": row["problem"],
            "steps": ".",
            "answer": answer,
        })

    print(f"Converted {len(converted)} samples ({skipped} skipped, no correct generation or answer).")

    # Shuffle and optionally limit
    random.seed(args.seed)
    random.shuffle(converted)
    if args.max_samples:
        converted = converted[:args.max_samples]
        print(f"Limited to {len(converted)} samples.")

    # Split
    split_idx = int(len(converted) * args.split_ratio)
    train_data = converted[:split_idx]
    test_data = converted[split_idx:]

    # Save
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    train_path = out / "openr1_train.json"
    test_path = out / "openr1_test.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(train_data)} train samples to {train_path}")
    print(f"Saved {len(test_data)} test samples to {test_path}")

    # Print a sample
    if converted:
        sample = converted[0]
        print(f"\n--- Sample ---")
        print(f"Question: {sample['question'][:200]}...")
        print(f"Steps ({len(sample['steps'])}): {sample['steps'][0][:150]}...")
        print(f"Answer: {sample['answer']}")


if __name__ == "__main__":
    main()
