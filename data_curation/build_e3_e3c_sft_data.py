#!/usr/bin/env python
"""
data_curation/build_e3_e3c_sft_data.py — Phase 16 §6.5 prep.

data/novelty/{e3,e3c}_round1_filtered.jsonl have {target, completion,
n_tokens, clipped, round, phase} -- no "prompt" field, since the
sampling scripts only stored the raw completion (core.data.
build_sft_dataset needs {prompt, completion}). Reconstructs the prompt
via the same closed_prompt(target) convention every other script in
this project uses, and splits train/val the same way
data_curation/build_rs_sft_dataset.py does: every 25th example held out
for val (eval loss is not decision-relevant for this arm).

Usage:
  uv run python data_curation/build_e3_e3c_sft_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402


def closed_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def build(name: str):
    src = REPO_ROOT / "data" / "novelty" / f"{name}_round1_filtered.jsonl"
    out_dir = REPO_ROOT / "data" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with src.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append({
                "prompt": closed_prompt(rec["target"]),
                "completion": rec["completion"],
                "target": rec["target"],
                "source": name,
            })

    val = rows[::25]
    train = [r for i, r in enumerate(rows) if i % 25 != 0]

    with (out_dir / f"{name}_train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with (out_dir / f"{name}_val.jsonl").open("w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} total -> {len(train)} train, {len(val)} val "
         f"-> {out_dir}/{name}_{{train,val}}.jsonl")


if __name__ == "__main__":
    build("e3")
    build("e3c")
