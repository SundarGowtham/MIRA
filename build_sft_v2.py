"""
build_sft_v2.py
---------------
Assemble SFT training data from reasoning_traces JSONL files.

Pipeline:
    data/processed/reasoning_traces_120B.jsonl  ──┐
    data/processed/reasoning_traces.jsonl (20B) ──┴──> sft_v2_{train,val,test}.jsonl
                                                       sft_v2_stats.json

Each output example: {prompt, completion, metadata}
    prompt:     existing PROMPT_TEMPLATE (matches model's SFT-trained prompt structure)
    completion: <think>{reasoning}</think>\n\n<precursors>...</precursors>\n\n<operations>...</operations>
    metadata:   target_formula, source (closed_book | fallback),
                validator_score, n_precursors, n_operations, etc.

Filtering (the "balanced" preset):
  - Closed-book records:  keep if validator_score >= 0.65 (all of them by construction)
  - Fallback records:     re-validate the MP route, keep if mp_route_score >= 0.65
                          AND reasoning_length >= 1200 chars
  - Drop:                 records with reasoning < 800 chars, malformed schema

Train/val/test split:
  - 85/10/5 by target formula (prevents same target appearing in multiple splits)
  - Seed=42 for reproducibility

Usage:
    python build_sft_v2.py
    python build_sft_v2.py --min-closed-book-score 0.85   # stricter
    python build_sft_v2.py --include-fallback False        # closed-book only
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"


# ---------------------------------------------------------------------------
# Prompt template — matches existing sft_data.py so the prompt structure is
# stable across SFT-v1 (the broken one) and SFT-v2 (this one). The completion
# is what changes: reasoning is now actually populated.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
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


COMPLETION_TEMPLATE = """\
<think>
{reasoning}
</think>

<precursors>
{precursors}
</precursors>

<operations>
{operations}
</operations>"""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_traces(*paths: Path) -> list[dict]:
    """Load and merge trace JSONLs. Dedup on (target_formula, mp_record_idx).
    When the same record is present in multiple files, prefer:
      1. closed-book over fallback
      2. higher validator_score (for closed-book)
      3. longer reasoning (tiebreak)
    """
    by_key: dict[tuple, dict] = {}
    for p in paths:
        if not p.exists():
            print(f"  skipping (does not exist): {p}")
            continue
        n = 0
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r["target_formula"], r["mp_record_idx"])
                if key in by_key:
                    incumbent = by_key[key]
                    if _is_better(r, incumbent):
                        by_key[key] = r
                else:
                    by_key[key] = r
                n += 1
        print(f"  loaded {n} records from {p.name}")
    return list(by_key.values())


def _is_better(new: dict, incumbent: dict) -> bool:
    """Dedup tiebreaker: closed-book > fallback; higher score > lower; longer > shorter."""
    new_cb = not new.get("used_fallback", True)
    inc_cb = not incumbent.get("used_fallback", True)
    if new_cb and not inc_cb:
        return True
    if inc_cb and not new_cb:
        return False
    new_score = new.get("validator_score") or 0.0
    inc_score = incumbent.get("validator_score") or 0.0
    if new_score != inc_score:
        return new_score > inc_score
    return len(new.get("reasoning_raw", "")) > len(incumbent.get("reasoning_raw", ""))


# ---------------------------------------------------------------------------
# Fallback re-validation: score the MP route through the same validator
# ---------------------------------------------------------------------------

def revalidate_fallbacks(records: list[dict], validator) -> None:
    """In place. For each fallback record, run the MP route through the validator
    and attach mp_route_score / mp_route_breakdown. Closed-book records already
    have validator_score set; we leave those alone.
    """
    from validator import (
        PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions
    )

    n_done = 0
    n_failed = 0
    for r in records:
        if not r.get("used_fallback", False):
            continue
        target = r["target_formula"]
        try:
            # The MP precursors have varied schemas; the predicted_* fields were
            # already normalized at generation time into the validator-friendly
            # form, so we can use those.
            precs = [
                PredictedPrecursor(formula=p["formula"], amount=p["amount"])
                for p in r["predicted_precursors"]
            ]
            ops = []
            for op in r["predicted_operations"]:
                ops.append(PredictedOperation(
                    type=op["type"],
                    conditions=PredictedConditions(
                        heating_temperature=op.get("heating_temperature", []),
                        heating_time=op.get("heating_time", []),
                        heating_atmosphere=op.get("heating_atmosphere", []),
                    ),
                ))
            route = PredictedRoute(
                target_formula=target,
                precursors=precs,
                operations=ops,
            )
            score, breakdown = validator.validate(route, target)
            r["mp_route_score"] = score
            r["mp_route_breakdown"] = breakdown
            n_done += 1
        except Exception as e:
            r["mp_route_score"] = 0.0
            r["mp_route_breakdown"] = {"error": str(e)}
            n_failed += 1

    print(f"  re-validated {n_done} fallback records ({n_failed} failed)")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_records(records: list[dict], args) -> tuple[list[dict], dict[str, int]]:
    """Apply quality filters. Returns (kept, drop_reasons)."""
    kept = []
    drops: Counter = Counter()

    for r in records:
        target = r.get("target_formula")
        if not target:
            drops["no_target"] += 1
            continue

        reasoning = r.get("reasoning_raw", "")
        if len(reasoning) < args.min_reasoning_length:
            drops["short_reasoning"] += 1
            continue

        precs = r.get("predicted_precursors", [])
        ops = r.get("predicted_operations", [])
        if not precs or not ops:
            drops["missing_precs_or_ops"] += 1
            continue

        # Schema sanity check
        if any("amount" not in p or "formula" not in p for p in precs):
            drops["malformed_schema"] += 1
            continue

        if r.get("used_fallback", False):
            if not args.include_fallback:
                drops["excluded_fallback"] += 1
                continue
            mp_score = r.get("mp_route_score")
            if mp_score is None or mp_score < args.min_fallback_score:
                drops["low_mp_route_score"] += 1
                continue
            if len(reasoning) < args.min_fallback_reasoning_length:
                drops["short_fallback_reasoning"] += 1
                continue
        else:
            score = r.get("validator_score")
            if score is None or score < args.min_closed_book_score:
                drops["low_closed_book_score"] += 1
                continue

        kept.append(r)

    return kept, dict(drops)


# ---------------------------------------------------------------------------
# Format conversion: trace → SFT example
# ---------------------------------------------------------------------------

def format_precursors_block(precursors: list[dict]) -> str:
    return "\n".join(f"- {p['formula']} | {p['amount']}" for p in precursors)


def format_operations_block(operations: list[dict]) -> str:
    lines = []
    for i, op in enumerate(operations, 1):
        op_type = op.get("type", "Unknown")
        cond_parts = []
        temps = op.get("heating_temperature") or []
        if temps:
            cond_parts.append(f"T={temps[0]:.0f}°C")
        times = op.get("heating_time") or []
        if times:
            cond_parts.append(f"t={times[0]:.1f}h")
        atm = op.get("heating_atmosphere") or []
        if atm:
            cond_parts.append(f"atm={','.join(atm)}")
        media = op.get("media") or ""
        if media:
            cond_parts.append(f"media={media}")
        cond_str = ", ".join(cond_parts) if cond_parts else "none"
        lines.append(f"{i}. {op_type} | {cond_str}")
    return "\n".join(lines)


def format_target_context_from_record(r: dict) -> str:
    """We don't carry summary context through the trace pipeline, so this
    is empty. The target alone is sufficient — gpt-oss-120b reasoned without
    crystal-system context and still hit median 0.93 score."""
    return ""


def make_sft_example(r: dict) -> dict:
    target = r["target_formula"]
    prompt = PROMPT_TEMPLATE.format(
        target=target,
        target_context=format_target_context_from_record(r),
    )
    completion = COMPLETION_TEMPLATE.format(
        reasoning=r["reasoning_raw"].strip(),
        precursors=format_precursors_block(r["predicted_precursors"]),
        operations=format_operations_block(r["predicted_operations"]),
    )

    source = "fallback" if r.get("used_fallback", False) else "closed_book"
    score = r.get("validator_score") if source == "closed_book" else r.get("mp_route_score")

    return {
        "prompt": prompt,
        "completion": completion,
        "metadata": {
            "target_formula": target,
            "mp_record_idx": r.get("mp_record_idx"),
            "source": source,
            "validator_score": score,
            "n_precursors": len(r["predicted_precursors"]),
            "n_operations": len(r["predicted_operations"]),
            "reasoning_length": len(r["reasoning_raw"]),
            "generator": r.get("generator", "unknown"),
        },
    }


# ---------------------------------------------------------------------------
# Train/val/test split — by target formula, seeded
# ---------------------------------------------------------------------------

SPLIT_RATIOS = {"train": 0.85, "val": 0.10, "test": 0.05}


def split_by_target(examples: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_target: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_target[ex["metadata"]["target_formula"]].append(ex)

    targets = sorted(by_target.keys())  # sorted for reproducibility
    rng.shuffle(targets)

    n = len(targets)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    train_targets = set(targets[:n_train])
    val_targets = set(targets[n_train : n_train + n_val])
    test_targets = set(targets[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for t, exs in by_target.items():
        if t in train_targets:
            splits["train"].extend(exs)
        elif t in val_targets:
            splits["val"].extend(exs)
        else:
            splits["test"].extend(exs)
    return splits


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(splits: dict[str, list[dict]]) -> dict:
    out: dict[str, Any] = {}
    for name, exs in splits.items():
        if not exs:
            out[name] = {"n_examples": 0}
            continue
        targets = {e["metadata"]["target_formula"] for e in exs}
        sources = Counter(e["metadata"]["source"] for e in exs)
        scores = [e["metadata"]["validator_score"] for e in exs
                  if e["metadata"]["validator_score"] is not None]
        reason_lens = [e["metadata"]["reasoning_length"] for e in exs]
        prompt_lens = [len(e["prompt"]) for e in exs]
        comp_lens = [len(e["completion"]) for e in exs]
        n_precs = [e["metadata"]["n_precursors"] for e in exs]
        n_ops = [e["metadata"]["n_operations"] for e in exs]

        def med(xs): return sorted(xs)[len(xs)//2] if xs else 0

        out[name] = {
            "n_examples": len(exs),
            "n_unique_targets": len(targets),
            "source_breakdown": dict(sources),
            "score_mean": round(statistics.mean(scores), 4) if scores else None,
            "score_median": round(statistics.median(scores), 4) if scores else None,
            "median_reasoning_length": med(reason_lens),
            "median_prompt_chars": med(prompt_lens),
            "median_completion_chars": med(comp_lens),
            "max_completion_chars": max(comp_lens) if comp_lens else 0,
            "median_n_precursors": med(n_precs),
            "median_n_operations": med(n_ops),
        }
    return out


def print_stats(stats: dict, drops: dict) -> None:
    print()
    print("=" * 64)
    print("SFT-V2 DATASET STATS")
    print("=" * 64)
    print("\nFilter drops:")
    for reason, n in sorted(drops.items(), key=lambda x: -x[1]):
        print(f"  {reason:35s}: {n}")
    print()
    for name in ["train", "val", "test"]:
        s = stats[name]
        if s["n_examples"] == 0:
            print(f"{name.upper()}: empty"); continue
        print(f"{name.upper()}:")
        print(f"  examples:               {s['n_examples']}")
        print(f"  unique targets:         {s['n_unique_targets']}")
        print(f"  sources:                {s['source_breakdown']}")
        print(f"  score mean/median:      {s['score_mean']} / {s['score_median']}")
        print(f"  reasoning median chars: {s['median_reasoning_length']}")
        print(f"  prompt median chars:    {s['median_prompt_chars']}")
        print(f"  completion median chars:{s['median_completion_chars']}")
        print(f"  completion max chars:   {s['max_completion_chars']}")
        print(f"  median precursors:      {s['median_n_precursors']}")
        print(f"  median operations:      {s['median_n_operations']}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trace-files",
        nargs="+",
        type=Path,
        default=[
            DATA_PROCESSED / "reasoning_traces_120B.jsonl",
            DATA_PROCESSED / "reasoning_traces.jsonl",
        ],
        help="One or more reasoning-traces JSONL files (merged, deduplicated).",
    )
    p.add_argument("--output-dir", type=Path, default=DATA_PROCESSED)
    p.add_argument("--prefix", default="sft_v2",
                   help="Output filename prefix: sft_v2_train.jsonl etc.")
    p.add_argument("--formula-set", type=Path, default=DATA_CACHE / "mp_formula_set.pkl")
    p.add_argument("--min-closed-book-score", type=float, default=0.65)
    p.add_argument("--min-fallback-score", type=float, default=0.65,
                   help="Min validator score on the MP route (re-validated by this script).")
    p.add_argument("--min-reasoning-length", type=int, default=800,
                   help="Hard minimum: anything shorter is rejected as a generation failure.")
    p.add_argument("--min-fallback-reasoning-length", type=int, default=1200,
                   help="Higher bar for fallback reasoning — these are post-hoc, "
                        "so we want substantive ones.")
    p.add_argument("--include-fallback", type=lambda s: s.lower() != "false", default=True,
                   help="Set --include-fallback False for closed-book-only training.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-revalidation", action="store_true",
                   help="Skip re-scoring fallback records' MP routes. Faster but less rigorous.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[1/5] Loading traces from {len(args.trace_files)} file(s)...")
    records = load_traces(*args.trace_files)
    print(f"  total unique records (after dedup): {len(records)}")

    if args.include_fallback and not args.skip_revalidation:
        print("\n[2/5] Re-validating fallback records' MP routes...")
        from core.reward import load_validator
        validator = load_validator(
            formula_set_path=args.formula_set,
            pd_cache_path=None,
        )
        revalidate_fallbacks(records, validator)
    else:
        print("\n[2/5] Skipping fallback revalidation.")

    print("\n[3/5] Filtering records...")
    kept, drops = filter_records(records, args)
    print(f"  kept {len(kept)} of {len(records)} records")

    print("\n[4/5] Building SFT examples...")
    examples = [make_sft_example(r) for r in kept]

    print("\n[5/5] Splitting train/val/test by target formula...")
    splits = split_by_target(examples, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        out_path = args.output_dir / f"{args.prefix}_{name}.jsonl"
        with out_path.open("w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
        print(f"  {out_path} → {len(exs)} examples")

    stats = compute_stats(splits)
    stats_path = args.output_dir / f"{args.prefix}_stats.json"
    with stats_path.open("w") as f:
        json.dump({"drops": drops, "splits": stats, "args": vars(args)}, f,
                  indent=2, default=str)
    print(f"  {stats_path}")

    print_stats(stats, drops)
    print("Done.")


if __name__ == "__main__":
    main()