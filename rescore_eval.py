"""
rescore_eval.py
---------------
Re-validates an eval_results/eval_*.json file by re-parsing saved completions
with robust JSON extraction and re-running the full validator. No model
re-inference needed — uses the completions already saved by evaluate_batched.py.

Diagnoses three potential failure modes:
  1. parse_completion failed (JSON not extracted) → was scoring 0 on
     stoichiometry/precursors_exist/operation_order despite valid model output
  2. PredictedRoute validation failed (schema mismatch) → field name issue
  3. Validator scored correctly but wasn't using thermo (--skip-thermo)

Usage:
    uv run python rescore_eval.py --input eval_results/eval_v3-rank16-seed42_test.json
    uv run python rescore_eval.py --input eval_results/eval_v3-rank16-seed42_test.json --skip-thermo
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from core.reward import load_validator, parse_completion


# ---------------------------------------------------------------------------
# Scoring — mirrors evaluate_batched.py's score_one exactly. parse_completion
# already handles JSON extraction + conversion to PredictedRoute (including
# the nested PredictedConditions/PredictedOperation structure). Do not
# reinvent this here.
# ---------------------------------------------------------------------------

def score_one(completion: str, target: str, validator) -> tuple[float, dict, str]:
    """
    Returns (reward, breakdown, outcome) where outcome is one of:
      "ok", "no_think_close", "parse_failed"
    """
    if "</think>" not in completion:
        # Completion was truncated before reasoning finished — there is no
        # JSON to recover. This is a generation-budget problem, not a
        # parsing problem. rescoring cannot fix it.
        return 0.0, {"truncated_no_think": 1.0}, "no_think_close"
    try:
        route = parse_completion(completion, target)
        reward, breakdown = validator.validate(route, target)
        return reward, breakdown, "ok"
    except Exception as e:
        return 0.0, {"parse_failed": 1.0, "error": str(e)[:200]}, "parse_failed"


# ---------------------------------------------------------------------------
# Main rescore loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to eval_*.json file produced by evaluate_batched.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to rescored output (default: <input>_rescored.json)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--skip-thermo", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Don't write output, just print diagnostic stats")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_name(args.input.stem + "_rescored.json")

    # --- Load saved eval payload ---
    with args.input.open() as f:
        payload = json.load(f)
    records = payload.get("records", [])
    print(f"Loaded {len(records)} records from {args.input}")

    # --- Load validator ---
    print("Loading validator...")
    validator = load_validator(
        formula_set_path=args.cache_dir / "mp_formula_set.pkl",
        pd_cache_path=None if args.skip_thermo else args.cache_dir / "phase_diagrams.pkl",
    )

    # --- Diagnostic counters ---
    parse_outcomes = Counter()
    schema_errors = Counter()
    score_deltas = []

    rescored_records = []
    for i, rec in enumerate(records):
        completion = rec.get("completion", "")
        target = rec.get("target", "")
        old_reward = rec.get("reward", 0)
        old_breakdown = rec.get("breakdown", {})

        new_reward, new_breakdown, outcome = score_one(completion, target, validator)
        parse_outcomes[outcome] += 1
        if outcome == "parse_failed":
            err = new_breakdown.get("error", "Unknown")
            schema_errors[err.split(":")[0]] += 1

        score_deltas.append(new_reward - old_reward)

        rescored_records.append({
            **rec,
            "old_reward": old_reward,
            "old_breakdown": old_breakdown,
            "reward": new_reward,
            "breakdown": new_breakdown,
            "delta": round(new_reward - old_reward, 4),
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(records)}]  "
                  f"running mean new reward = {statistics.mean([r['reward'] for r in rescored_records]):.3f}")

    # --- Aggregate ---
    new_rewards = [r["reward"] for r in rescored_records]
    old_rewards = [r["old_reward"] for r in rescored_records]

    print()
    print("=" * 60)
    print("PARSING OUTCOMES")
    print("=" * 60)
    total = len(records)
    for outcome, count in parse_outcomes.most_common():
        print(f"  {outcome:25s}  {count:4d}  ({100*count/total:.1f}%)")
    if schema_errors:
        print(f"\n  parse_failed error types:")
        for err, count in schema_errors.most_common():
            print(f"    {err:25s}  {count:4d}")

    if parse_outcomes.get("no_think_close", 0) > total * 0.1:
        print(f"\n  WARNING: {parse_outcomes['no_think_close']}/{total} completions never "
              f"closed <think>. This is a generation-length problem, not a parsing\n"
              f"  problem — rescoring cannot recover these. Re-run evaluate_batched.py\n"
              f"  with a larger --max-new-tokens (see audit_completion_tokens.py).")

    # Per-constraint scoring breakdown
    per_constraint: dict[str, list] = defaultdict(list)
    for r in rescored_records:
        for k, v in r["breakdown"].items():
            per_constraint[k].append(v)

    print()
    print("=" * 60)
    print("RESCORED RESULTS")
    print("=" * 60)
    print(f"  N:               {len(records)}")
    print(f"  Old mean reward: {statistics.mean(old_rewards):.4f}")
    print(f"  New mean reward: {statistics.mean(new_rewards):.4f}")
    print(f"  Mean delta:      {statistics.mean(score_deltas):+.4f}")
    print(f"  Improved:        {sum(1 for d in score_deltas if d > 0.01)}")
    print(f"  Unchanged:       {sum(1 for d in score_deltas if abs(d) <= 0.01)}")
    print(f"  Worsened:        {sum(1 for d in score_deltas if d < -0.01)}")
    print()
    print("  Per-constraint mean scores (new):")
    for k in sorted(per_constraint):
        scores = per_constraint[k]
        print(f"    {k:30s}  {statistics.mean(scores):.4f}  (n={len(scores)})")

    # --- Write output ---
    if not args.diagnostic_only:
        agg = {
            "n_examples": len(records),
            "mean_reward": round(statistics.mean(new_rewards), 4),
            "median_reward": round(statistics.median(new_rewards), 4),
            "std_reward": round(statistics.stdev(new_rewards), 4) if len(new_rewards) > 1 else 0,
            "min_reward": round(min(new_rewards), 4),
            "max_reward": round(max(new_rewards), 4),
            "format_fail_rate": round(sum(1 for r in new_rewards if r < 0.05) / len(records), 4),
        }
        for k, scores in per_constraint.items():
            agg[f"mean_{k}"] = round(statistics.mean(scores), 4)

        out = {
            **{k: v for k, v in payload.items() if k != "records" and k != "aggregate"},
            "rescored_from": str(args.input),
            "skip_thermo": args.skip_thermo,
            "parse_outcomes": dict(parse_outcomes),
            "aggregate": agg,
            "records": rescored_records,
        }
        with args.output.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()