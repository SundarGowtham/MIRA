#!/usr/bin/env python
"""
format_only_parse_rate.py — PHASE_11_REVISED.md Step 2, parse-rate pass
condition ("must clear ~95%, else format-only SFT is not doing its one job").

Generates completions for the held-out format_only_val.jsonl prompts (same
open-book-with-thermo-context style the model was trained on -- unlike the
closed-book ASTRAL protocol) with the format-only checkpoint, and measures
the fraction that parse cleanly via core.reward.parse_completion.

Usage (tmux):
  PYTHONPATH=. uv run python run_debug_and_analysis/format_only_parse_rate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace  # noqa: E402

from evaluate_batched import load_eval_model, generate_batch  # noqa: E402
from core.reward import parse_completion  # noqa: E402

CHECKPOINT = "runs/sft-qlora-format-only/final"
VAL_PATH = Path("data/sft_v3/format_only_val.jsonl")
OUT = Path("misc/format_only_parse_rate.json")
N_PROMPTS = 100


def main():
    examples = [json.loads(l) for l in VAL_PATH.open() if l.strip()][:N_PROMPTS]
    print(f"loaded {len(examples)} held-out prompts from {VAL_PATH}", flush=True)

    model, tok = load_eval_model(checkpoint=CHECKPOINT, model_name="Qwen/Qwen3-8B")
    gen_args = SimpleNamespace(temperature=1.0, top_p=0.95, max_new_tokens=8192)

    prompts = [ex["prompt"] for ex in examples]
    completions = []
    for i in range(0, len(prompts), 8):
        chunk = prompts[i:i + 8]
        completions.extend(generate_batch(model, tok, chunk, gen_args))
        print(f"  generated {min(i + 8, len(prompts))}/{len(prompts)}", flush=True)

    n_ok, failures = 0, []
    for ex, comp in zip(examples, completions):
        try:
            parse_completion(comp, ex["target"])
            n_ok += 1
        except Exception as e:
            failures.append({"target": ex["target"], "error": str(e)[:200]})

    rate = n_ok / len(examples)
    print("\n" + "=" * 78)
    print(f"parse rate: {n_ok}/{len(examples)} = {rate:.1%}")
    print(f"pass condition (>=95%): {'PASS' if rate >= 0.95 else 'FAIL'}")
    if failures[:10]:
        print("sample failures:", failures[:10])

    OUT.write_text(json.dumps({
        "checkpoint": CHECKPOINT, "n": len(examples), "n_ok": n_ok,
        "parse_rate": rate, "pass_95pct": rate >= 0.95, "failures": failures,
    }, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
