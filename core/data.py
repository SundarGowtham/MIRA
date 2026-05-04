from __future__ import annotations
import json
from pathlib import Path
from datasets import Dataset


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


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
            "target_formula": ex["metadata"]["target_formula"],
        })
    return Dataset.from_list(rows)