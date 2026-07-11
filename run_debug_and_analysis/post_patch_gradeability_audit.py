"""
post_patch_gradeability_audit.py
-----------------------------------
Closes the loop on the flat-GRPO-signal investigation. Everything before
this point either probed ThermoChecker internals directly or backtested
the interpolation function in isolation - useful for diagnosis, but NOT
the same as confirming the actual patched validate() pathway (the one
GRPO training calls through reward.py) now produces the right numbers on
real data.

This calls validator.validate(route, target) directly - the real
production entrypoint - and reads the gradeability tags straight out of
the returned breakdown dict:
  scores["thermodynamic_favorable_gradeability"]
  scores["target_stability_gradeability"]
  scores["chempot_atmosphere_gradeability"]

each one of "discrete" / "interpolated" / "ungradeable" / "sentinel_no_entry"
/ "not_applicable" / "no_thermo_checker", per validator.py's new contract.

DO NOT reuse coverage_audit.py / pd_coverage_attribution.py /
pd_failure_diagnosis.py / pd_frontier_check.py against the patched
validator.py - they call reaction_energy_per_atom/target_e_above_hull
directly and unpack the OLD single-value return shape. Since
reaction_energy_per_atom now returns (value, tag), a 2-tuple is never
None, so `if dG is not None` in those scripts is always True regardless
of whether the value inside is actually None. They will not crash - they
will silently misclassify every record as gradeable. This script replaces
them for post-patch verification.

Reports, per split:
  - gradeability tag distribution for each of the three thermo checks
  - reward distribution: raw validate() reward AND the post-floor
    adjusted reward (max(r - 0.30, 0), mirroring reward.py's
    make_reward_fn exactly) - this is the number that actually reaches
    GRPO, so it's the real test of whether the flat-signal problem is fixed
  - direct before/after framing for thermodynamic_favorable specifically,
    since that's the check the interpolation fallback targets

Usage:
  uv run python post_patch_gradeability_audit.py \
      --split data/sft/train.jsonl \
      --split data/sft/test.jsonl \
      --split data/sft/val.jsonl \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --out post_patch_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

try:
    from core.reward import parse_completion, ParseFailure, load_validator
except ImportError:
    from reward import parse_completion, ParseFailure, load_validator


FLOOR_CONST = 0.30  # must match reward.py's make_reward_fn


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def get_target_formula(ex: dict) -> tuple[str, str]:
    """
    Tolerant accessor. Confirmed schema for data/sft/*.jsonl is a flat
    ex["target"] key (NOT ex["metadata"]["target_formula"] - that was
    this script's author's mistaken assumption in earlier scripts this
    session, silently corrected by hand each time it was run). Tries the
    confirmed key first, falls back to the old guess, then a couple of
    other plausible shapes, and reports which one actually matched so a
    schema surprise on a different file is visible immediately instead of
    silently wrong.
    """
    if "target" in ex:
        return ex["target"], "target"
    if "metadata" in ex and isinstance(ex["metadata"], dict) and "target_formula" in ex["metadata"]:
        return ex["metadata"]["target_formula"], "metadata.target_formula"
    if "target_formula" in ex:
        return ex["target_formula"], "target_formula"
    raise KeyError(
        f"No recognized target-formula key in record. Keys present: {list(ex.keys())}"
    )


def audit_split(path: Path, validator, sample: int | None = None) -> dict:
    examples = load_jsonl(path)
    if sample:
        examples = examples[:sample]

    schema_seen = Counter()
    n_parse_fail = 0
    n_validate_exception = 0

    tag_counts = {
        "thermodynamic_favorable_gradeability": Counter(),
        "target_stability_gradeability": Counter(),
        "chempot_atmosphere_gradeability": Counter(),
    }
    raw_rewards = []
    adjusted_rewards = []
    interpolated_examples = []  # spot-check: records that specifically used interpolation

    for ex in examples:
        try:
            target, schema_key = get_target_formula(ex)
        except KeyError:
            continue
        schema_seen[schema_key] += 1

        try:
            route = parse_completion(ex["completion"], target)
        except ParseFailure:
            n_parse_fail += 1
            continue

        try:
            reward, breakdown = validator.validate(route, target)
        except Exception:
            n_validate_exception += 1
            continue

        raw_rewards.append(reward)
        adjusted_rewards.append(max(reward - FLOOR_CONST, 0.0))

        for tag_key in tag_counts:
            if tag_key in breakdown:
                tag_counts[tag_key][breakdown[tag_key]] += 1

        if breakdown.get("thermodynamic_favorable_gradeability") == "interpolated" and \
                len(interpolated_examples) < 10:
            interpolated_examples.append({
                "target": target,
                "thermodynamic_favorable_score": breakdown.get("thermodynamic_favorable"),
                "reward": reward,
            })

    def dist(xs):
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        return {
            "n": len(xs),
            "min": round(min(xs), 4),
            "median": round(statistics.median(xs), 4),
            "mean": round(statistics.mean(xs), 4),
            "max": round(max(xs), 4),
            "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "frac_zero": round(sum(1 for x in xs if x <= 1e-9) / len(xs), 3),
        }

    return {
        "path": str(path),
        "n_total": len(examples),
        "n_parse_fail": n_parse_fail,
        "n_validate_exception": n_validate_exception,
        "n_scored": len(raw_rewards),
        "schema_key_used": dict(schema_seen),
        "gradeability_tags": {
            k: dict(v) for k, v in tag_counts.items()
        },
        "raw_reward_distribution": dist(raw_rewards),
        "adjusted_reward_distribution_post_floor": dist(adjusted_rewards),
        "interpolated_examples": interpolated_examples,
    }


def print_summary(r: dict):
    print(f"\n=== {r['path']}  (n_total={r['n_total']}, scored={r['n_scored']}) ===")
    if r["schema_key_used"]:
        print(f"target-formula key used: {r['schema_key_used']}")
    if r["n_parse_fail"] or r["n_validate_exception"]:
        print(f"parse failures: {r['n_parse_fail']}  validate() exceptions: {r['n_validate_exception']}")

    print("\ngradeability tags (this is the direct successor to the old light_only metric):")
    for check_name, tags in r["gradeability_tags"].items():
        total = sum(tags.values())
        if total == 0:
            continue
        print(f"  {check_name}:")
        for tag, count in sorted(tags.items(), key=lambda kv: -kv[1]):
            print(f"    {tag:<16} {count:<6} ({count/total:.1%})")

    rd = r["raw_reward_distribution"]
    ad = r["adjusted_reward_distribution_post_floor"]
    if rd["n"]:
        print(f"\nraw reward:      n={rd['n']}  median={rd['median']}  mean={rd['mean']}  "
              f"stdev={rd['stdev']}  frac_zero={rd['frac_zero']}")
        print(f"adjusted reward: median={ad['median']}  mean={ad['mean']}  "
              f"stdev={ad['stdev']}  frac_zero={ad['frac_zero']}  "
              f"(this is what reward_fn actually returns to GRPO)")

    if r["interpolated_examples"]:
        print(f"\nspot-check — records now graded via interpolation ({len(r['interpolated_examples'])} shown):")
        for ex in r["interpolated_examples"][:5]:
            print(f"  {ex['target']:<24} thermo_score={ex['thermodynamic_favorable_score']}  "
                  f"total_reward={ex['reward']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, dest="splits")
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--sample", type=int, default=None, help="cap records per split (default: all)")
    ap.add_argument("--out", type=Path, default=Path("post_patch_results.json"))
    args = ap.parse_args()

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for split_path in args.splits:
        p = Path(split_path)
        print(f"Auditing {p} ...", file=sys.stderr)
        r = audit_split(p, validator, args.sample)
        all_results[p.name] = r
        print_summary(r)

    with args.out.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()