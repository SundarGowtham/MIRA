"""
sft_data.py
-----------
Convert cached synthesis records into SFT training examples.

Pipeline:
    data/raw/synthesis.json  ──▶  data/processed/sft_{train,val,test}.jsonl
    data/raw/summary.json   ─┘    data/processed/sft_stats.json

Each example is a {prompt, completion} pair:
    prompt     — task instruction + structured target description
    completion — the synthesis route in a stable, parseable format

"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from monty.serialization import loadfn
from pymatgen.core import Composition


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Train/val/test split
SPLIT_RATIOS = {"train": 0.85, "val": 0.10, "test": 0.05}
RANDOM_SEED  = 42

# Filter thresholds
MIN_OPERATIONS = 2     # routes with fewer than this are noise
MIN_PRECURSORS = 1     # routes with no precursors are useless
MAX_PRECURSORS = 6     # routes with too many usually parse errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def is_valid_concrete_formula(formula: str | None) -> bool:
    """Same logic as data_pull.py — reject templated/multi-target strings."""
    if not formula or "-" in formula:
        return False
    try:
        comp = Composition(formula)
        for el in comp.elements:
            if not str(el).isalpha() or len(str(el)) > 2:
                return False
            if str(el) in {"M", "L", "A", "B", "X", "R", "Ln", "An"}:
                return False
        return True
    except Exception:
        return False


def load_caches() -> tuple[list[dict], dict[str, dict]]:
    """Load synthesis records and a formula→summary lookup."""
    synth = loadfn(DATA_RAW / "synthesis.json")
    summary = loadfn(DATA_RAW / "summary.json")

    summary_by_formula: dict[str, dict] = {}
    for s in summary:
        try:
            key = Composition(s["formula_pretty"]).reduced_formula
            summary_by_formula[key] = s
        except Exception:
            pass

    log(f"Loaded {len(synth)} synthesis records, "
        f"{len(summary_by_formula)} summary entries")
    return synth, summary_by_formula


def filter_records(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """
    Remove records the model can't usefully learn from.

    Returns (kept, drop_reasons_counter).
    """
    kept: list[dict] = []
    drops: Counter = Counter()

    for r in records:
        target = r.get("target_formula")
        if not is_valid_concrete_formula(target):
            drops["invalid_target_formula"] += 1
            continue

        precursors = r.get("precursors", []) or []
        valid_precs = [
            p for p in precursors
            if is_valid_concrete_formula(p.get("formula"))
        ]
        if len(valid_precs) < MIN_PRECURSORS:
            drops["too_few_precursors"] += 1
            continue
        if len(valid_precs) > MAX_PRECURSORS:
            drops["too_many_precursors"] += 1
            continue

        ops = r.get("operations", []) or []
        if len(ops) < MIN_OPERATIONS:
            drops["too_few_operations"] += 1
            continue

        # Replace precursors with the valid subset for downstream formatting
        r = dict(r)
        r["precursors"] = valid_precs
        kept.append(r)

    return kept, drops


# ---------------------------------------------------------------------------
# Formatting — the heart of SFT
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """
<role>
You are a materials chemist designing a solid-state synthesis route.
</role>

<target material>
TARGET MATERIAL: {target}{target_context}
</target material>

<objective>
Propose a synthesis route. Output the route in the following structured format:
</objective>

<precursors>
- formula | amount
- ...
</precursors>

<operations>
1. operation_type | conditions
2. ...
</operations>

Use real, commercially available precursor compounds. Operations must be in physically valid order (mixing before heating). Specify temperatures in Celsius."""


def format_target_context(
    target_formula: str,
    summary: dict | None,
) -> str:
    """
    Optional structural/property context for the target.
    Empty string if no summary available.
    """
    if summary is None:
        return ""

    parts = []
    if summary.get("crystal_system"):
        parts.append(f"crystal system: {summary['crystal_system']}")
    if summary.get("spacegroup_number"):
        parts.append(f"space group: {summary['spacegroup_number']}")
    if summary.get("band_gap") is not None:
        bg = summary["band_gap"]
        parts.append(f"band gap: {bg:.2f} eV" if bg > 0 else "metallic")
    if summary.get("density"):
        parts.append(f"density: {summary['density']:.2f} g/cm³")

    if not parts:
        return ""
    return "\nKnown properties: " + ", ".join(parts)


def format_precursors(precursors: list[dict]) -> str:
    """
    Convert precursor list to the structured block in the completion.

    Amount inference: synthesis records don't always have stoichiometric
    amounts attached to precursors (the reaction object has them, but
    the precursor list is just the chemicals). For SFT we default to 1.0
    and let the LLM learn typical patterns from many examples.
    """
    lines = []
    for p in precursors:
        formula = p["formula"]
        amount = p.get("amount", 1.0)
        lines.append(f"- {formula} | {amount}")
    return "\n".join(lines)


def format_operations(operations: list[dict]) -> str:
    """
    Convert operation list to numbered structured lines.

    Each line: "N. OperationType | T=...°C, t=...h, atm=..."
    Conditions are only emitted if present.
    """
    lines = []
    for i, op in enumerate(operations, 1):
        op_type = op.get("type", "Unknown")

        # Build conditions string
        cond_parts = []

        temps = op.get("heating_temperature", []) or []
        flat_temps = [t for sub in temps for t in (sub if isinstance(sub, list) else [sub])]
        if flat_temps:
            avg_temp = sum(flat_temps) / len(flat_temps)
            cond_parts.append(f"T={avg_temp:.0f}°C")

        times = op.get("heating_time", []) or []
        flat_times = [t for sub in times for t in (sub if isinstance(sub, list) else [sub])]
        if flat_times:
            avg_time = sum(flat_times) / len(flat_times)
            cond_parts.append(f"t={avg_time:.1f}h")

        atm = op.get("heating_atmosphere", []) or []
        if atm:
            cond_parts.append(f"atm={','.join(atm)}")

        media = op.get("mixing_media")
        if media:
            cond_parts.append(f"media={media}")

        cond_str = ", ".join(cond_parts) if cond_parts else "none"
        lines.append(f"{i}. {op_type} | {cond_str}")

    return "\n".join(lines)


COMPLETION_TEMPLATE = """<precursors>
{precursors}
</precursors>

<operations>
{operations}
</operations>"""


def make_example(record: dict, summary_by_formula: dict[str, dict]) -> dict:
    """Build one {prompt, completion, metadata} example."""
    target = record["target_formula"]
    target_reduced = Composition(target).reduced_formula
    summary = summary_by_formula.get(target_reduced)

    prompt = PROMPT_TEMPLATE.format(
        target=target,
        target_context=format_target_context(target, summary),
    )
    completion = COMPLETION_TEMPLATE.format(
        precursors=format_precursors(record["precursors"]),
        operations=format_operations(record["operations"]),
    )

    return {
        "prompt": prompt,
        "completion": completion,
        # Metadata stays alongside but isn't fed to the model;
        # used for the validator at GRPO time and for evaluation.
        "metadata": {
            "target_formula": target,
            "doi": record.get("doi"),
            "synthesis_type": record.get("synthesis_type"),
            "n_precursors": len(record["precursors"]),
            "n_operations": len(record["operations"]),
            "has_summary": summary is not None,
        },
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_examples(
    examples: list[dict],
    seed: int = RANDOM_SEED,
) -> dict[str, list[dict]]:
    """
    Split by TARGET FORMULA, not by record. This prevents the same
    target appearing in both train and test (data leakage). If we
    split by record, the model could memorize "BaTiO3 needs BaCO3 +
    TiO2" from training, then ace test examples for the same target.
    """
    rng = random.Random(seed)

    # Group records by target formula
    by_target: dict[str, list[dict]] = {}
    for ex in examples:
        target = ex["metadata"]["target_formula"]
        by_target.setdefault(target, []).append(ex)

    targets = list(by_target.keys())
    rng.shuffle(targets)

    n = len(targets)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    train_targets = set(targets[:n_train])
    val_targets   = set(targets[n_train : n_train + n_val])
    test_targets  = set(targets[n_train + n_val :])

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for target, recs in by_target.items():
        if target in train_targets:
            splits["train"].extend(recs)
        elif target in val_targets:
            splits["val"].extend(recs)
        else:
            splits["test"].extend(recs)

    return splits


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def compute_stats(splits: dict[str, list[dict]]) -> dict:
    stats: dict[str, Any] = {}

    for split_name, examples in splits.items():
        targets = {e["metadata"]["target_formula"] for e in examples}
        n_ops = [e["metadata"]["n_operations"] for e in examples]
        n_precs = [e["metadata"]["n_precursors"] for e in examples]
        with_summary = sum(1 for e in examples if e["metadata"]["has_summary"])
        prompt_lens = [len(e["prompt"]) for e in examples]
        completion_lens = [len(e["completion"]) for e in examples]

        def med(xs: list) -> float:
            return sorted(xs)[len(xs) // 2] if xs else 0

        stats[split_name] = {
            "n_examples": len(examples),
            "n_unique_targets": len(targets),
            "median_n_operations": med(n_ops),
            "median_n_precursors": med(n_precs),
            "with_summary_pct": (
                round(100 * with_summary / len(examples), 1) if examples else 0
            ),
            "median_prompt_chars": med(prompt_lens),
            "median_completion_chars": med(completion_lens),
            "max_prompt_chars": max(prompt_lens) if prompt_lens else 0,
            "max_completion_chars": max(completion_lens) if completion_lens else 0,
        }

    return stats


def print_stats(stats: dict, drops: dict) -> None:
    print()
    print("=" * 60)
    print("SFT DATASET STATS")
    print("=" * 60)
    print()
    print("Records dropped during filtering:")
    for reason, count in drops.most_common():
        print(f"  {reason:30s}: {count}")
    print()
    for split_name in ["train", "val", "test"]:
        s = stats[split_name]
        print(f"{split_name.upper()}:")
        print(f"  examples:               {s['n_examples']}")
        print(f"  unique targets:         {s['n_unique_targets']}")
        print(f"  median operations:      {s['median_n_operations']}")
        print(f"  median precursors:      {s['median_n_precursors']}")
        print(f"  with summary context:   {s['with_summary_pct']}%")
        print(f"  median prompt chars:    {s['median_prompt_chars']}")
        print(f"  median completion chars:{s['median_completion_chars']}")
        print(f"  max prompt chars:       {s['max_prompt_chars']}")
        print(f"  max completion chars:   {s['max_completion_chars']}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    log("Loading caches...")
    synth, summary_by_formula = load_caches()

    log("Filtering records...")
    kept, drops = filter_records(synth)
    log(f"  kept {len(kept)} of {len(synth)} records")

    log("Building examples...")
    examples = [make_example(r, summary_by_formula) for r in kept]

    log("Splitting train/val/test by target formula...")
    splits = split_examples(examples)

    log("Writing JSONL files...")
    for name, exs in splits.items():
        path = DATA_PROCESSED / f"sft_{name}.jsonl"
        write_jsonl(path, exs)
        log(f"  {path} → {len(exs)} examples")

    log("Computing stats...")
    stats = compute_stats(splits)
    stats_path = DATA_PROCESSED / "sft_stats.json"
    with stats_path.open("w") as f:
        json.dump({"drops": dict(drops), "splits": stats}, f, indent=2)
    log(f"  saved → {stats_path}")

    print_stats(stats, drops)
    log("Done.")


if __name__ == "__main__":
    main()