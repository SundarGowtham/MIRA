from __future__ import annotations
import json
from pathlib import Path
from datasets import Dataset


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def get_target(ex: dict) -> str:
    """
    Extract target formula from a record.
    split_dataset.py writes 'target' at top level.
    Legacy evaluate.py format wraps it in ex['metadata']['target_formula'].
    Mirrors evaluate_batched.py's get_target() exactly - the two files had
    diverged (this one only checked the legacy path, confirmed broken
    against real data 2026-07), now both use the same tolerant lookup.
    """
    if "target" in ex:
        return ex["target"]
    return ex.get("metadata", {}).get("target_formula", "")


def build_sft_dataset(jsonl_path: Path, tokenizer, limit: int | None = None) -> Dataset:
    """For SFT: format {prompt, completion} → {text} via chat template."""
    examples = load_jsonl(jsonl_path)
    if limit:
        examples = examples[:limit]
    rows = []
    for ex in examples:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]},
             {"role": "assistant", "content": ex["completion"]}],
            tokenize=False, add_generation_prompt=False,
        )
        rows.append({"text": text})
    return Dataset.from_list(rows)


def build_grpo_dataset(jsonl_path: Path, tokenizer, limit: int | None = None) -> Dataset:
    """For GRPO: format prompt only (model generates completion)."""
    examples = load_jsonl(jsonl_path)
    if limit:
        examples = examples[:limit]
    rows = []
    for ex in examples:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        rows.append({
            "prompt": prompt_text,
            "target_formula": get_target(ex),
        })
    return Dataset.from_list(rows)