"""
regrade_eval.py
---------------
Re-scores existing eval result JSON files using the fixed reward.py
parser, WITHOUT re-running inference. The raw completions are already
on disk — only the grading logic was broken.

Schema (confirmed from eval_results/eval_v3-rank16-seed42_test.json):
  Top-level: {checkpoint, model, data_prefix, split, batch_size,
              max_new_tokens, temperature, top_p, skip_thermo,
              aggregate, records}
  Each record: {idx, target, reward, breakdown:{6 constraint keys},
                completion}
  aggregate: {n_examples, mean_reward, median_reward, std_reward,
              min_reward, max_reward, format_fail_rate,
              mean_<constraint> for each of the 6 constraints}

Usage:
  uv run python regrade_eval.py eval_results/eval_v3-rank16-seed42_test.json
  uv run python regrade_eval.py eval_results/eval_v3-rank*_test.json

  # to overwrite in place rather than writing _regraded.json files:
  uv run python regrade_eval.py eval_results/eval_v3-*_test.json --in-place
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
from pathlib import Path
from rich import print as rprint
from core.reward import parse_completion, ParseFailure, load_validator


CONSTRAINT_KEYS = [
    "stoichiometry", "charge_neutrality", "precursors_exist",
    "operation_order", "temperature_plausible", "target_match",
]


def regrade_file(path: Path, validator, verbose: bool = False) -> dict:
    with path.open() as f:
        data = json.load(f)

    records = data["records"]
    n_total = len(records)
    n_parse_failed = 0
    n_validate_failed = 0
    rewards = []
    constraint_totals: dict[str, list[float]] = {k: [] for k in CONSTRAINT_KEYS}
    old_rewards = []

    new_records = []
    for rec in records[:1]:
        completion = rec["completion"]
        target = rec["target"]
        old_rewards.append(rec.get("reward", 0.0))

        new_rec = {k: v for k, v in rec.items() if k != "breakdown" and k != "reward"}

        try:
            route = parse_completion(completion, target)
        except ParseFailure as e:
            n_parse_failed += 1
            new_rec["reward"] = 0.0
            new_rec["breakdown"] = {k: 0.0 for k in CONSTRAINT_KEYS}
            new_rec["parse_ok"] = False
            new_rec["parse_error"] = str(e)
            rewards.append(0.0)
            new_records.append(new_rec)
            if verbose:
                print(f"  [parse_fail] idx={rec.get('idx')} target={target}: {e}")
            continue

        try:
            reward, breakdown = validator.validate(route, target)
            rprint("breakdown: ", breakdown)
        except Exception as e:
            n_validate_failed += 1
            new_rec["reward"] = 0.0
            new_rec["breakdown"] = {k: 0.0 for k in CONSTRAINT_KEYS}
            new_rec["parse_ok"] = True
            new_rec["validate_error"] = str(e)
            rewards.append(0.0)
            new_records.append(new_rec)
            if verbose:
                print(f"  [validate_fail] idx={rec.get('idx')} target={target}: {e}")
            continue

        # Match the original aggregation's reward-shaping: raw reward, not r-0.30.
        # (The training reward function in make_reward_fn does subtract 0.30, but
        # the eval breakdown stored alongside is the raw constraint scores; we
        # report the raw mean reward here so it lines up with the original
        # aggregate's mean_reward semantics for direct comparison.)
        new_rec["reward"] = round(reward, 4)
        new_rec["breakdown"] = {k: round(float(breakdown.get(k, 0.0)), 4)
                                 for k in CONSTRAINT_KEYS}
        new_rec["parse_ok"] = True
        rewards.append(reward)
        for k in CONSTRAINT_KEYS:
            constraint_totals[k].append(float(breakdown.get(k, 0.0)))

        new_records.append(new_rec)

    mean_old = statistics.mean(old_rewards) if old_rewards else 0.0
    if rewards:
        mean_reward = statistics.mean(rewards)
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        median_reward = statistics.median(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
    else:
        mean_reward = std_reward = median_reward = min_reward = max_reward = 0.0

    new_aggregate = {
        "n_examples": n_total,
        "mean_reward": round(mean_reward, 4),
        "median_reward": round(median_reward, 4),
        "std_reward": round(std_reward, 4),
        "min_reward": round(min_reward, 4),
        "max_reward": round(max_reward, 4),
        "format_fail_rate": data.get("aggregate", {}).get("format_fail_rate", 0.0),
        "parse_fail_rate": round(n_parse_failed / n_total, 4) if n_total else 0.0,
        "validate_fail_rate": round(n_validate_failed / n_total, 4) if n_total else 0.0,
    }
    for k in CONSTRAINT_KEYS:
        vs = constraint_totals[k]
        new_aggregate[f"mean_{k}"] = round(statistics.mean(vs), 4) if vs else 0.0

    new_data = dict(data)
    new_data["aggregate_original"] = data.get("aggregate")
    new_data["aggregate"] = new_aggregate
    new_data["records"] = new_records
    new_data["regraded"] = True

    return {
        "data": new_data,
        "summary": {
            "n_total": n_total,
            "mean_old": mean_old,
            "mean_new": mean_reward,
            "parse_fail_rate": new_aggregate["parse_fail_rate"],
            "per_constraint_old": {k: data.get("aggregate", {}).get(f"mean_{k}") for k in CONSTRAINT_KEYS},
            "per_constraint_new": {k: new_aggregate[f"mean_{k}"] for k in CONSTRAINT_KEYS},
        }
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--formula-set", type=Path, default=Path("data/cache/mp_formula_set.pkl"))
    p.add_argument("--pd-index", type=Path, default=Path("data/cache/pd_index.json"))
    p.add_argument("--project-root", type=Path, default=Path(os.getcwd()),
                   help="Directory pd_index.json's shard paths are resolved against")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite source files instead of writing _regraded.json copies")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    pd_index = args.pd_index if args.pd_index and args.pd_index.exists() else None
    validator = load_validator(args.formula_set, pd_index, args.project_root)

    print(f"\n{'file':50s} {'old_mean':>10s} {'new_mean':>10s} {'parse_fail':>11s}")
    print("-" * 86)

    summaries = []
    for path in args.files:
        if args.verbose:
            print(f"\n=== {path.name} ===")
        result = regrade_file(path, validator, verbose=args.verbose)
        # rprint(f"result: {result}")
        s = result["summary"]
        summaries.append((path.name, s))
        print(f"{path.stem[:50]:50s} {s['mean_old']:>10.4f} {s['mean_new']:>10.4f} "
              f"{s['parse_fail_rate']*100:>10.2f}%")

        # if args.in_place:
        #     out_path = path
        # else:
        #     out_path = path.with_name(path.stem + "_regraded" + path.suffix)
        # with out_path.open("w") as f:
        #     json.dump(result["data"], f, indent=2)

    # Per-constraint breakdown across all files
    print(f"\n{'constraint':25s} ", end="")
    for name, _ in summaries:
        short = name.replace("eval_v3-", "").replace("_test.json", "")[:14]
        print(f"{short:>15s}", end="")
    print()
    print("-" * (25 + 15 * len(summaries)))
    for k in CONSTRAINT_KEYS:
        print(f"{k:25s} ", end="")
        for _, s in summaries:
            v = s["per_constraint_new"][k]
            print(f"{v:>15.4f}", end="")
        print()


if __name__ == "__main__":
    main()