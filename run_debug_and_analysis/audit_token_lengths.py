"""
audit_token_lengths.py
----------------------
Measure post-chat-template tokenized lengths across the SFT dataset.

This is what max_seq_len actually constrains. Char-count estimates aren't
reliable for Qwen3 because the chat template adds ~30-50 tokens of role
markers and special tokens, and the BPE tokenizer's chars-per-token ratio
depends on content (chemistry formulas tokenize differently than prose).

Reports min/max/mean/median and the truncation rate for several candidate
max_seq_len values, so you can pick one with eyes open.

Usage:
    python audit_token_lengths.py
    python audit_token_lengths.py --data-prefix sft_v2 --split train
    python audit_token_lengths.py --max-seq-lens 1024 1280 1536 2048
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--data-prefix", default="sft_v2")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument(
        "--max-seq-lens",
        type=int,
        nargs="+",
        default=[1024, 1280, 1536, 2048],
        help="Candidate values to report truncation rates for.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    path = args.data_dir / f"{args.data_prefix}_{args.split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"Loading tokenizer for {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Reading {path}...")
    with path.open() as f:
        examples = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(examples)} examples\n")

    # Measure three things per example:
    #   1. Full chat-templated length (what SFTTrainer feeds to the model)
    #   2. Prompt-only length (chat-templated, with add_generation_prompt)
    #   3. Completion-only length (raw, no template)
    full_lens = []
    prompt_lens = []
    completion_lens = []

    for ex in examples:
        full_text = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]},
             {"role": "assistant", "content": ex["completion"]}],
            tokenize=False, add_generation_prompt=False,
        )
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        completion_ids = tok(ex["completion"], add_special_tokens=False).input_ids
        full_lens.append(len(full_ids))
        prompt_lens.append(len(prompt_ids))
        completion_lens.append(len(completion_ids))

    def report(name, xs):
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        print(f"  {name}:")
        print(f"    min     = {min(xs)}")
        print(f"    p10     = {xs_sorted[n // 10]}")
        print(f"    p25     = {xs_sorted[n // 4]}")
        print(f"    median  = {xs_sorted[n // 2]}")
        print(f"    p75     = {xs_sorted[3 * n // 4]}")
        print(f"    p90     = {xs_sorted[9 * n // 10]}")
        print(f"    p95     = {xs_sorted[19 * n // 20]}")
        print(f"    p99     = {xs_sorted[99 * n // 100]}")
        print(f"    max     = {max(xs)}")
        print(f"    mean    = {statistics.mean(xs):.1f}")

    print("Token-length distributions:")
    report("full sequence (prompt+completion, chat-templated)", full_lens)
    print()
    report("prompt only (chat-templated, with gen prompt)", prompt_lens)
    print()
    report("completion only (raw)", completion_lens)
    print()

    print("Truncation rate at each candidate max_seq_len:")
    print("  (% of examples that would be truncated)")
    for L in sorted(args.max_seq_lens):
        n_trunc = sum(1 for x in full_lens if x > L)
        pct = 100.0 * n_trunc / len(full_lens)
        print(f"    max_seq_len={L:5d}: {n_trunc:5d} / {len(full_lens)} truncated  ({pct:.2f}%)")

    print()
    print("Recommendations:")
    p99 = sorted(full_lens)[99 * len(full_lens) // 100]
    p95 = sorted(full_lens)[19 * len(full_lens) // 20]
    print(f"  Set max_seq_len to ~p99 ({p99}) to truncate ~1% of long-tail examples.")
    print(f"  Set max_seq_len to ~p95 ({p95}) to truncate ~5% — tighter, more VRAM headroom.")
    print(f"  Set max_seq_len to max ({max(full_lens)}) for zero truncation but max VRAM cost.")


if __name__ == "__main__":
    main()