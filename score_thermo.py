"""
score_thermo.py
---------------
Add thermo_favorable scores to an existing eval result JSON, without
loading the model. Use when --skip-thermo was used during evaluation
because the 2GB PD cache wouldn't fit alongside the model.

Procedure:
  1. Load the model-free validator (formula set + thermo cache)
  2. Re-parse each saved completion in the eval JSON
  3. Score with full validator (including thermo)
  4. Write augmented JSON with thermo column added

Usage:
    python score_thermo.py --eval-json eval_results/eval_sft-v2_test.json
"""

from __future__ import annotations
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, required=True,
                   help="Path to existing eval result JSON.")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--output", type=Path, default=None,
                   help="Output path. Default: <input>_with_thermo.json")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.eval_json.exists():
        raise FileNotFoundError(args.eval_json)

    out_path = args.output or args.eval_json.with_name(
        args.eval_json.stem + "_with_thermo.json"
    )

    log(f"Reading {args.eval_json}...")
    with args.eval_json.open() as f:
        data = json.load(f)
    records = data.get("records", [])
    log(f"  {len(records)} eval records")

    log("Loading validator with thermo cache (this takes ~30-60s)...")
    from core.reward import load_validator, parse_completion
    validator = load_validator(
        formula_set_path=args.cache_dir / "mp_formula_set.pkl",
        pd_cache_path=args.cache_dir / "phase_diagrams.pkl",
    )
    log("Validator loaded.")

    log("Re-scoring records with thermo...")
    constraint_scores = defaultdict(list)
    n_thermo_hits = 0
    n_thermo_neutral = 0
    rewards_new = []

    for i, rec in enumerate(records):
        try:
            route = parse_completion(rec["completion"], rec["target"])
            reward, breakdown = validator.validate(route, rec["target"])
        except Exception as e:
            reward, breakdown = 0.0, {"error": 1.0}
        rec["reward"] = reward
        rec["breakdown"] = breakdown
        rewards_new.append(reward)
        for k, v in breakdown.items():
            constraint_scores[k].append(v)
        if "thermo_favorable" in breakdown:
            if breakdown["thermo_favorable"] >= 0.99:
                n_thermo_hits += 1
            elif 0.4 < breakdown["thermo_favorable"] < 0.6:
                n_thermo_neutral += 1
        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(records)} done")

    log("Done. Aggregating...")
    agg = {
        "n_examples":       len(records),
        "mean_reward":      round(statistics.mean(rewards_new), 4),
        "median_reward":    round(statistics.median(rewards_new), 4),
        "std_reward":       round(statistics.stdev(rewards_new), 4) if len(rewards_new) > 1 else 0.0,
        "min_reward":       round(min(rewards_new), 4),
        "max_reward":       round(max(rewards_new), 4),
    }
    for k, scores in constraint_scores.items():
        agg[f"mean_{k}"] = round(statistics.mean(scores), 4)
    data["aggregate"] = agg

    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    log(f"Wrote {out_path}")

    print()
    print("=" * 60)
    print("THERMO-AUGMENTED EVAL RESULTS")
    print("=" * 60)
    print(f"Mean reward:        {agg['mean_reward']:.4f} (± {agg['std_reward']:.4f})")
    print(f"Median reward:      {agg['median_reward']:.4f}")
    print()
    print("Per-constraint mean scores:")
    for k, v in agg.items():
        if k.startswith("mean_") and k != "mean_reward":
            print(f"  {k[5:]:30s}: {v:.4f}")
    print()
    print(f"Thermo: {n_thermo_hits} hits ({100*n_thermo_hits/len(records):.1f}%), "
          f"{n_thermo_neutral} neutral (~0.5)")


if __name__ == "__main__":
    main()