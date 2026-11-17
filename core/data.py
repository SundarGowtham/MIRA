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


def is_capped_completion(completion: str) -> bool:
    """A self-generated completion that hit the sampling cap: no </think>."""
    return "</think>" not in completion


def build_sft_full_dataset(jsonl_path: Path, tokenizer, limit: int | None = None) -> Dataset:
    """Phase 20 SFT path: pre-tokenized rows (input_ids + completion_mask).

    Ordinary rows are tokenized exactly as TRL 1.8.0 tokenizes a
    conversational prompt-completion example (trl.data_utils._tokenize with
    the tokenizer's own chat template): prompt rendered with
    add_generation_prompt=True, full sequence rendered without it,
    completion_mask = 0 on the prompt and 1 on the assistant turn. TRL then
    builds labels from completion_mask (completion_only_loss=True) and, with
    max_length=None, never truncates.

    Capped rows (a sampled completion that hit the 8,192-token cap and never
    closed </think>; 2 such completions in data/e3 and data/e3c) are
    "rendered as sampled" (Phase 20 pre-training decision, 2026-11-17): the
    rendered generation prompt + the raw sampled text tokenized as-is, with no
    template-inserted empty <think></think>, no </think> and no <|im_end|>
    (the model never stopped). Loss on those completion tokens only.

    The old `build_sft_dataset` (flat "text", loss on every token) is unchanged.
    """
    from trl.data_utils import _tokenize
    examples = load_jsonl(jsonl_path)
    if limit:
        examples = examples[:limit]
    kw = {"chat_template": None, "tools": None}
    rows = []
    for ex in examples:
        prompt = [{"role": "user", "content": ex["prompt"]}]
        p_ids = list(_tokenize(tokenizer, prompt, add_generation_prompt=True, **kw)["input_ids"])
        if is_capped_completion(ex["completion"]):
            c_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
            ids = p_ids + list(c_ids)
        else:
            full = prompt + [{"role": "assistant", "content": ex["completion"]}]
            ids = list(_tokenize(tokenizer, full, **kw)["input_ids"])
            if ids[: len(p_ids)] != p_ids:
                raise ValueError(f"prompt is not a token prefix of the full sequence ({ex.get('target')})")
        rows.append({"input_ids": ids, "completion_mask": [0] * len(p_ids) + [1] * (len(ids) - len(p_ids))})
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