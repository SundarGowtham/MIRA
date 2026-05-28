"""
score_thermo_sharded.py
-----------------------
Re-scores an existing eval JSON with full 6/6 validator constraints,
using the sharded PD cache instead of the monolithic 7GB pickle.

Does NOT re-run the model. Reads saved completions from the eval JSON,
re-parses and re-scores each one with thermo enabled.

Usage:
    python score_thermo_sharded.py
    python score_thermo_sharded.py \
        --eval-json eval_results/eval_sft-v2_test.json \
        --shards-dir data/cache/pd_shards \
        --out eval_results/eval_sft-v2_test_6of6.json
"""
from __future__ import annotations
import argparse, json, pickle, statistics, time
from pathlib import Path
from collections import defaultdict


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Load all PD shards into a single dict, mimicking the monolithic cache
# ---------------------------------------------------------------------------

def load_sharded_pd_cache(shards_dir: Path) -> dict:
    index_path = shards_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No index.json in {shards_dir}")

    index = json.loads(index_path.read_text())
    log(f"  index has {len(index)} entries")

    pd_cache = {}
    t0 = time.time()
    for chemsys, filename in index.items():
        shard_path = shards_dir / filename
        if not shard_path.exists():
            log(f"  WARNING: missing shard {filename} — skipping")
            continue
        with shard_path.open("rb") as f:
            pd_cache[chemsys] = pickle.load(f)

    elapsed = time.time() - t0
    log(f"  loaded {len(pd_cache)} PDs in {elapsed:.1f}s")
    return pd_cache


# ---------------------------------------------------------------------------
# Patch ThermoChecker to use an already-loaded dict
# ---------------------------------------------------------------------------

def make_thermo_checker_from_dict(pd_cache: dict):
    """
    Creates a ThermoChecker instance using an already-loaded PD dict,
    bypassing ThermoChecker.from_cache() which expects a monolithic pickle.
    We directly set the internal cache attribute without touching validator.py.
    """
    from validator import ThermoChecker

    # ThermoChecker.from_cache loads a pickle into self.pd_cache.
    # We create an instance and set the attribute directly.
    checker = ThermoChecker.__new__(ThermoChecker)
    checker.phase_diagrams = pd_cache  # was: checker.pd_cache
    return checker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path,
                   default=Path("eval_results/eval_sft-v2_test.json"))
    p.add_argument("--shards-dir", type=Path,
                   default=Path("data/cache/pd_shards"))
    p.add_argument("--formula-set", type=Path,
                   default=Path("data/cache/mp_formula_set.pkl"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Default: <input>_6of6.json")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or args.eval_json.with_name(
        args.eval_json.stem + "_6of6.json"
    )

    log(f"Reading {args.eval_json}...")
    with args.eval_json.open() as f:
        data = json.load(f)
    records = data["records"]
    log(f"  {len(records)} records")

    log(f"Loading PD shards from {args.shards_dir}...")
    pd_cache = load_sharded_pd_cache(args.shards_dir)

    log("Building validator with thermo...")
    from core.reward import load_validator, parse_completion
    # load_validator with skip-thermo to get the base validator
    validator = load_validator(
        formula_set_path=args.formula_set,
        pd_cache_path=None,
    )
    # Now inject the sharded thermo checker
    thermo_checker = make_thermo_checker_from_dict(pd_cache)
    validator.thermo_checker = thermo_checker
    log("  validator ready (6/6 constraints)")

    log("Re-scoring records...")
    constraint_scores = defaultdict(list)
    rewards = []
    n_thermo_scored = 0
    n_thermo_skipped = 0

    for i, rec in enumerate(records):
        target = rec["target"]
        completion = rec["completion"]
        try:
            route = parse_completion(completion, target)
            reward, breakdown = validator.validate(route, target)
        except Exception as e:
            reward = 0.0
            breakdown = {"error": str(e)}

        rec["reward"] = reward
        rec["breakdown"] = breakdown
        rewards.append(reward)

        for k, v in breakdown.items():
            constraint_scores[k].append(v)

        thermo_val = breakdown.get("thermodynamic_favorable")
        if thermo_val is not None and thermo_val != 0.5:
            n_thermo_scored += 1
        else:
            n_thermo_skipped += 1

        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(records)}  "
                f"running mean={statistics.mean(rewards):.4f}  "
                f"thermo_scored={n_thermo_scored}")

    # Update aggregate
    agg = {
        "n_examples":    len(records),
        "mean_reward":   round(statistics.mean(rewards), 4),
        "median_reward": round(statistics.median(rewards), 4),
        "std_reward":    round(statistics.stdev(rewards), 4),
        "min_reward":    round(min(rewards), 4),
        "max_reward":    round(max(rewards), 4),
        "thermo_scored": n_thermo_scored,
        "thermo_skipped_no_pd": n_thermo_skipped,
    }
    for k, scores in constraint_scores.items():
        agg[f"mean_{k}"] = round(statistics.mean(scores), 4)

    data["aggregate"] = agg
    data["shards_dir"] = str(args.shards_dir)

    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    log(f"Wrote {out_path}")

    print()
    print("=" * 60)
    print("6/6 EVAL RESULTS")
    print("=" * 60)
    print(f"Mean reward:    {agg['mean_reward']:.4f}")
    print(f"Median reward:  {agg['median_reward']:.4f}")
    print(f"Thermo scored:  {n_thermo_scored} / {len(records)}")
    print()
    print("Per-constraint means:")
    for k, v in agg.items():
        if k.startswith("mean_") and k != "mean_reward":
            print(f"  {k[5:]:30s}: {v:.4f}")


if __name__ == "__main__":
    main()