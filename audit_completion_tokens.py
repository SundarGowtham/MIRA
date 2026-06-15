"""
audit_completion_tokens.py
---------------------------
Tokenizes the 'completion' field (= <think>...</think>\\n{response}) of
the SFT train/val/test JSONL files using the actual model tokenizer, and
reports percentiles. Use this to set --max-new-tokens for eval/generation
correctly — setting it too low truncates every completion mid-<think>
and produces zero valid JSON (as happened with the default 1024).

Usage:
    uv run python audit_completion_tokens.py --model Qwen/Qwen3-8B
    uv run python audit_completion_tokens.py --model Qwen/Qwen3-8B --data-dir data/sft
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    all_lengths = []
    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"  skip {path} (not found)")
            continue
        examples = load_jsonl(path)
        lengths = []
        for ex in examples:
            completion = ex.get("completion", "")
            ids = tok(completion, add_special_tokens=False)["input_ids"]
            lengths.append(len(ids))
        all_lengths.extend(lengths)

        arr = np.array(lengths)
        print(f"\n{split} (n={len(arr)}):")
        print(f"  mean:   {arr.mean():.1f}")
        print(f"  p50:    {np.percentile(arr, 50):.0f}")
        print(f"  p90:    {np.percentile(arr, 90):.0f}")
        print(f"  p95:    {np.percentile(arr, 95):.0f}")
        print(f"  p99:    {np.percentile(arr, 99):.0f}")
        print(f"  max:    {arr.max():.0f}")

    arr = np.array(all_lengths)
    print(f"\n{'='*50}")
    print(f"ALL SPLITS (n={len(arr)}):")
    print(f"  mean:   {arr.mean():.1f}")
    print(f"  p90:    {np.percentile(arr, 90):.0f}")
    print(f"  p95:    {np.percentile(arr, 95):.0f}")
    print(f"  p99:    {np.percentile(arr, 99):.0f}")
    print(f"  max:    {arr.max():.0f}")

    p99 = int(np.percentile(arr, 99))
    recommended = ((p99 // 256) + 2) * 256  # round up to next 256, +1 buffer block
    print(f"\nRecommended --max-new-tokens: {recommended}")
    print(f"  (p99={p99}, rounded up with headroom)")


if __name__ == "__main__":
    main()