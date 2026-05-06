"""
evaluate.py — score a trained checkpoint (or base model) on the SFT test set.

Loads the model, generates completions, parses each through the validator,
and reports per-constraint reward breakdown.

Usage:
    python evaluate.py --checkpoint runs/sft-qlora/final
    python evaluate.py --checkpoint base --model meta-llama/Llama-3.1-8B-Instruct
    python evaluate.py --checkpoint runs/sft-qlora/final --limit 50 --tag debug
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from core.data import load_jsonl
from core.reward import parse_completion, load_validator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint, or 'base' for pretrained model")
    p.add_argument("--model", default=None, help="Base model name (required if checkpoint=base or LoRA)")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--n-samples", type=int, default=1, help="Generations per prompt (>1 enables pass@k)")
    p.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    p.add_argument("--tag", default=None, help="Suffix for output filename")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-thermo", action="store_true", help="Skip phase diagram cache load (fast dev iteration)")
    return p.parse_args()


def load_eval_model(checkpoint: str, model_name: str | None):
    """Returns (model, tokenizer). Handles base model or LoRA checkpoint."""
    if checkpoint == "base":
        if not model_name:
            raise ValueError("--model required when --checkpoint=base")
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto",
        )
    else:
        ckpt_path = Path(checkpoint)
        # If checkpoint dir has a tokenizer, use it; else fall back to base model
        tok_path = ckpt_path if (ckpt_path / "tokenizer_config.json").exists() else model_name
        if not tok_path:
            raise ValueError("--model required when checkpoint has no tokenizer")
        tok = AutoTokenizer.from_pretrained(tok_path)

        # Determine if this is a full checkpoint or a LoRA adapter
        is_lora = (ckpt_path / "adapter_config.json").exists()
        if is_lora:
            if not model_name:
                # Try to read base model from adapter config
                with (ckpt_path / "adapter_config.json").open() as f:
                    adapter_cfg = json.load(f)
                model_name = adapter_cfg.get("base_model_name_or_path")
                if not model_name:
                    raise ValueError("Adapter config missing base_model_name_or_path; pass --model")
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
    model.eval()
    return model, tok


def generate_completion(model, tok, prompt: str, args) -> str:
    """Single completion. Caller handles batching/sampling externally."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.pad_token_id,
        )
    completion_ids = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(completion_ids, skip_special_tokens=True)


def score_one(completion: str, target: str, validator) -> tuple[float, dict]:
    """Parse + validate. Returns (reward, per-constraint breakdown)."""
    try:
        route = parse_completion(completion, target)
        return validator.validate(route, target)
    except Exception:
        return 0.0, {"error": 1.0}


def aggregate(records: list[dict]) -> dict:
    """Compute aggregate stats across all evaluation records."""
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
            constraint = k[5:]
            print(f"  {constraint:30s}: {v:.4f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading model from {args.checkpoint}...")
    model, tok = load_eval_model(args.checkpoint, args.model)

    print("Loading validator...")
    validator = load_validator(
        formula_set_path=args.cache_dir / "mp_formula_set.pkl",
        pd_cache_path=None if args.skip_thermo else args.cache_dir / "phase_diagrams.pkl",
    )

    print("Loading validator...")
    if args.skip_thermo:
        print("  (--skip-thermo: thermo check disabled, using neutral 0.5)")
        

    eval_path = args.data_dir / f"sft_{args.split}.jsonl"
    examples = load_jsonl(eval_path)
    if args.limit:
        examples = examples[:args.limit]
    print(f"Evaluating {len(examples)} examples from {eval_path.name}...")

    records = []
    for i, ex in enumerate(examples, 1):
        target = ex["metadata"]["target_formula"]
        # Track best-of-N for pass@k style; default n=1 just scores the single sample
        best_reward, best_breakdown, best_completion = -1.0, {}, ""
        for _ in range(args.n_samples):
            completion = generate_completion(model, tok, ex["prompt"], args)
            reward, breakdown = score_one(completion, target, validator)
            if reward > best_reward:
                best_reward, best_breakdown, best_completion = reward, breakdown, completion

        records.append({
            "target": target,
            "reward": best_reward,
            "breakdown": best_breakdown,
            "completion": best_completion,
        })
        if i % 10 == 0 or i == len(examples):
            running = sum(r["reward"] for r in records) / len(records)
            print(f"  [{i}/{len(examples)}] running mean reward = {running:.3f}")

    agg = aggregate(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.checkpoint).name or "base"
    out_path = args.output_dir / f"eval_{tag}_{args.split}.json"
    with out_path.open("w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "model": args.model,
            "split": args.split,
            "n_samples_per_prompt": args.n_samples,
            "aggregate": agg,
            "records": records,
        }, f, indent=2)

    print_summary(agg, tag)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()