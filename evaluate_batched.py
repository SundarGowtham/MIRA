"""
evaluate_batched.py
-------------------
Batched evaluation of a trained checkpoint against the SFT test set.

Same scoring pipeline as evaluate.py but generates in batches for ~10-15x
speedup on a single GPU. Critical implementation details:

  - Left-padding (so generation continues from prompt end, not pad tokens)
  - Per-batch progress logging (flushed immediately)
  - Incremental JSON writes (kill-safe — partial results survive)
  - Skips already-evaluated examples on resume

Usage:
    python evaluate_batched.py --checkpoint runs/sft-qlora-sft-v2-qwen/final \\
        --model Qwen/Qwen3-8B --data-prefix sft_v2 --tag sft-v2 --skip-thermo
    python evaluate_batched.py --checkpoint base --model Qwen/Qwen3-8B \\
        --data-prefix sft_v2 --tag base --skip-thermo --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.data import load_jsonl
from core.reward import load_validator, parse_completion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="LoRA checkpoint path or 'base' for pretrained")
    p.add_argument("--model", default=None, help="Base model name (required for base or LoRA)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Directory containing split JSONL files. Defaults to data/sft/ "
                        "(output of split_dataset.py). Override to data/processed/ for legacy files.")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--data-prefix", default="sft",
                   help="Filename prefix for data files. E.g. --data-prefix sft_v2 "
                        "loads sft_v2_test.jsonl. Default 'sft' is back-compat.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8, help="Prompts per batch (8 fits comfortably on 3090, try 16 on A100)")
    p.add_argument("--max-new-tokens", type=int, default=11000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--skip-thermo", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model loading (mirrors evaluate.py but with explicit left-padding)
# ---------------------------------------------------------------------------

def load_eval_model(checkpoint: str, model_name: str | None):
    if checkpoint == "base":
        if not model_name:
            raise ValueError("--model required when --checkpoint=base")
        tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto",
        )
    else:
        ckpt_path = Path(checkpoint)
        tok_path = ckpt_path if (ckpt_path / "tokenizer_config.json").exists() else model_name
        if not tok_path:
            raise ValueError("--model required when checkpoint has no tokenizer")
        tok = AutoTokenizer.from_pretrained(tok_path, padding_side="left")

        is_lora = (ckpt_path / "adapter_config.json").exists()
        if is_lora:
            if not model_name:
                with (ckpt_path / "adapter_config.json").open() as f:
                    adapter_cfg = json.load(f)
                model_name = adapter_cfg.get("base_model_name_or_path")
                if not model_name:
                    raise ValueError("Adapter config missing base_model_name_or_path")
            base = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="auto",
            )
            model = PeftModel.from_pretrained(base, str(ckpt_path))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_path), dtype=torch.bfloat16, device_map="auto",
            )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Critical: left-padding for batched generation
    tok.padding_side = "left"
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_batch(model, tok, prompts: list[str], args) -> list[str]:
    """
    Generate completions for a batch of prompts. Left-padding required.
    Returns one decoded completion per prompt (excludes prompt tokens).
    """
    # Apply chat template per prompt
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]

    inputs = tok(
        texts, return_tensors="pt",
        padding=True, truncation=True, max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.pad_token_id,
        )

    # out shape: [batch, prompt_len + completion_len]
    # With left-padding, the prompt is right-aligned, so the completion
    # starts at index input_ids.shape[1] for ALL rows in the batch.
    prompt_len = inputs.input_ids.shape[1]
    completion_ids = out[:, prompt_len:]
    completions = tok.batch_decode(completion_ids, skip_special_tokens=True)
    return completions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_target(ex: dict) -> str:
    """
    Extract target formula from a record.
    split_dataset.py writes 'target' at top level.
    Legacy evaluate.py format wraps it in ex['metadata']['target_formula'].
    """
    if "target" in ex:
        return ex["target"]
    return ex.get("metadata", {}).get("target_formula", "")

    # try:
    #     route = parse_completion(completion, target)
    #     return validator.validate(route, target)
    # except Exception:
    #     return 0.0, {"error": 1.0}
def score_one(completion: str, target: str, validator) -> tuple[float, dict]:
    try:
        route = parse_completion(completion, target)
        return validator.validate(route, target)
    except Exception as e:
        return 0.0, {"error": 1.0}

def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    rewards = [r["reward"] for r in records]
    constraint_scores: dict[str, list[float]] = defaultdict(list)
    for r in records:
        for k, v in r["breakdown"].items():
            constraint_scores[k].append(v)

    agg = {
        "n_examples":       len(records),
        "mean_reward":      round(statistics.mean(rewards), 4),
        "median_reward":    round(statistics.median(rewards), 4),
        "std_reward":       round(statistics.stdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "min_reward":       round(min(rewards), 4),
        "max_reward":       round(max(rewards), 4),
        "format_fail_rate": round(
            sum(1 for r in records if r["reward"] < 0.05) / len(records), 4
        ),
    }
    for k, scores in constraint_scores.items():
        agg[f"mean_{k}"] = round(statistics.mean(scores), 4)
    return agg


def print_summary(agg: dict, tag: str) -> None:
    print()
    print("=" * 60)
    print(f"EVAL RESULTS: {tag}")
    print("=" * 60)
    print(f"Examples:           {agg['n_examples']}")
    print(f"Mean reward:        {agg['mean_reward']:.4f} (± {agg['std_reward']:.4f})")
    print(f"Median reward:      {agg['median_reward']:.4f}")
    print(f"Min / Max:          {agg['min_reward']:.4f} / {agg['max_reward']:.4f}")
    print(f"Format-fail rate:   {agg['format_fail_rate']:.1%}")
    print()
    print("Per-constraint mean scores:")
    for k, v in agg.items():
        if k.startswith("mean_") and k != "mean_reward":
            print(f"  {k[5:]:30s}: {v:.4f}")
    print(flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_results(args, records, agg, out_path):
    payload = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "data_prefix": args.data_prefix,
        "split": args.split,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "skip_thermo": args.skip_thermo,
        "aggregate": agg,
        "records": records,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Resolve data_dir default: split_dataset.py writes to data/sft/
    if args.data_dir is None:
        args.data_dir = Path("data/sft") if Path("data/sft").exists() else Path("data/processed")
        log(f"data-dir not specified; using {args.data_dir}")

    log(f"Loading model from {args.checkpoint}...")
    model, tok = load_eval_model(args.checkpoint, args.model)
    log(f"Model loaded. Device: {model.device}")

    log("Loading validator...")
    if args.skip_thermo:
        log("  (--skip-thermo: using neutral 0.5 for thermo)")
    validator = load_validator(
        formula_set_path=args.cache_dir / "mp_formula_set.pkl",
        pd_cache_path=None if args.skip_thermo else args.cache_dir / "phase_diagrams.pkl",
    )

    eval_path = args.data_dir / f"{args.data_prefix}_{args.split}.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval data not found at {eval_path}. "
            f"Check --data-prefix (currently '{args.data_prefix}') and --split "
            f"(currently '{args.split}')."
        )
    examples = load_jsonl(eval_path)
    if args.limit:
        examples = examples[: args.limit]
    log(f"Evaluating {len(examples)} examples from {eval_path.name}")

    # Resume support: load existing results if present
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.checkpoint).name or "base"
    out_path = args.output_dir / f"eval_{tag}_{args.split}.json"

    existing_records = []
    completed_targets = set()
    if out_path.exists():
        try:
            with out_path.open() as f:
                prev = json.load(f)
            existing_records = prev.get("records", [])
            completed_targets = {
                (r["target"], r.get("idx", i))
                for i, r in enumerate(existing_records)
            }
            log(f"Resuming: {len(existing_records)} examples already done")
        except Exception as e:
            log(f"Could not read prior results ({e}); starting fresh")
            existing_records = []
            completed_targets = set()

    # Filter to TODO
    todo = []
    for i, ex in enumerate(examples):
        target = get_target(ex)
        if (target, i) in completed_targets:
            continue
        todo.append((i, ex))

    if not todo:
        log("All examples already evaluated.")
        agg = aggregate(existing_records)
        print_summary(agg, tag)
        log(f"Results at: {out_path}")
        return

    log(f"  {len(todo)} examples remaining ({len(existing_records)} cached)")

    records = list(existing_records)
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size

    start_time = time.time()
    for batch_idx in range(n_batches):
        batch = todo[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        prompts = [ex["prompt"] for _, ex in batch]
        targets = [get_target(ex) for _, ex in batch]
        indices = [i for i, _ in batch]

        completions = generate_batch(model, tok, prompts, args)

        for i, target, completion in zip(indices, targets, completions):
            reward, breakdown = score_one(completion, target, validator)
            records.append({
                "idx": i,
                "target": target,
                "reward": reward,
                "breakdown": breakdown,
                "completion": completion,
            })

        # Incremental save after every batch — kill-safe
        agg = aggregate(records)
        write_results(args, records, agg, out_path)

        # Progress logging
        done = len(records)
        total = len(examples)
        elapsed = time.time() - start_time
        rate = (done - len(existing_records)) / max(elapsed, 1)
        remaining = len(todo) - (batch_idx + 1) * args.batch_size
        eta_min = max(remaining, 0) / max(rate * 60, 0.01)
        log(f"  batch {batch_idx + 1}/{n_batches}  "
            f"({done}/{total} examples)  "
            f"running mean reward = {agg['mean_reward']:.3f}  "
            f"rate = {rate:.2f}/s  "
            f"ETA = {eta_min:.0f} min")

    print_summary(agg, tag)
    log(f"\nResults at: {out_path}")


if __name__ == "__main__":
    main()