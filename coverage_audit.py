"""
coverage_audit.py
------------------
Answers the load-bearing question in HANDOFF_2 §2b / §7 #1: what fraction
of each split falls into the flat-reward (thermo-sentinel) region, and how
does that fraction split between in-MP-API and frontier targets?

This does NOT read stored `breakdown` dicts. A stored 0.5 is ambiguous
between "sentinel, can't compute" and "real value that happens to land
near 0.5" (cf. the Li2MnO3 case in HANDOFF_2 §2: thermodynamic_favorable
= 0.4988 is a TRUE computed value, numerically adjacent to the sentinel).
So gradeability is determined by probing the live ThermoChecker and
testing `is None`, never by reading a number back off disk.

Four gradeability tiers per record, in decreasing order of RL usefulness:
  real_dG    - reaction_energy_per_atom resolves (needs a PD shard
               covering target UNION precursor elements). All three
               thermo constraints vary meaningfully.
  hull_only  - target has a PD entry (target_e_above_hull resolves) but
               the full reaction doesn't balance against a covered shard
               (e.g. a precursor drags in an element outside any cached
               chemsys). target_stability still varies; the other two
               thermo constraints are flat.
  light_only - neither resolves. All three thermo constraints return the
               0.5 sentinel unconditionally. This is the true flat-reward
               region for GRPO.
  ungradeable - completion doesn't even parse into a PredictedRoute
               (ParseFailure). No constraint varies; reward_fn floors
               this to 0.0 regardless of chemistry.

Each record is also tagged in_mp (target formula, normalized, is in the
validator's mp_formula_set) as an approximate frontier/known split -
this is the same set the validator itself uses for precursors_exist, not
a separate frontier heuristic.

Additionally computes a `floor_risk` flag: whether the GROUND-TRUTH
completion's raw validate() reward is <= 0.30, the constant makred.py
subtracts before flooring at 0.0 (`adjusted = max(r - 0.30, 0.0)`). This
is a PROXY for GRPO's all-zero-reward failure mode, not a live measurement
of it - the real GRPO group is N sampled completions from the policy, not
the single ground-truth completion, so this only sizes "targets where even
a perfect answer starts near the floor," which is the necessary condition
for the whole group to floor together. The actual frac_reward_zero_std
must come from the effective-support probe (HANDOFF_2 §7 #6), not from
this script.

Usage (run on manifold, inside the MIRA project root so relative shard
paths in pd_index.json resolve):

  uv run python coverage_audit.py \
      --split data/processed/sft_v2_train.jsonl \
      --split data/processed/sft_v2_test.jsonl \
      --split data/processed/sft_v2_val.jsonl \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --out coverage_audit_results.json

Prints a per-split 4x2 cross-tab (gradeability x in_mp) to stdout and
writes full per-record classifications + the cross-tabs to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Layout tolerance: some checkouts have core/reward.py + core/validator.py
# as a package, others have reward.py / validator.py at repo root
# (both layouts appear in this project's history - see HANDOFF_2 imports
# vs the flat layout in /mnt/project). Try both.
try:
    from core.reward import parse_completion, ParseFailure, load_validator
except ImportError:
    from reward import parse_completion, ParseFailure, load_validator


GRADEABILITY_TIERS = ["real_dG", "hull_only", "light_only", "ungradeable"]
FLOOR_CONST = 0.30  # must match reward.py's make_reward_fn adjustment


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_record(ex: dict, validator) -> dict:
    """
    Returns a dict with: target, in_mp, gradeability, floor_risk,
    raw_reward (None if unparseable), parse_error (if any).
    """

    
    target = ex["target"]
    in_mp = validator._normalize_formula(target) in validator.mp_formula_set

    completion = ex["completion"]

    try:
        route = parse_completion(completion, target)
    except ParseFailure as e:
        return {
            "target": target,
            "in_mp": in_mp,
            "gradeability": "ungradeable",
            "floor_risk": True,  # reward_fn gives 0.0 on parse failure
            "raw_reward": None,
            "parse_error": str(e),
        }

    tc = validator.thermo_checker
    precs = [(p.formula, p.amount) for p in route.precursors]

    dG = None
    hull = None
    if tc is not None:
        try:
            dG = tc.reaction_energy_per_atom(precs, target, predicted_route=route)
        except Exception:
            dG = None
        try:
            hull = tc.target_e_above_hull(target)
        except Exception:
            hull = None

    if dG is not None:
        gradeability = "real_dG"
    elif hull is not None:
        gradeability = "hull_only"
    else:
        gradeability = "light_only"

    try:
        raw_reward, _breakdown = validator.validate(route, target)
    except Exception:
        raw_reward = None

    floor_risk = (raw_reward is None) or (raw_reward <= FLOOR_CONST)

    return {
        "target": target,
        "in_mp": in_mp,
        "gradeability": gradeability,
        "floor_risk": floor_risk,
        "raw_reward": raw_reward,
        "parse_error": None,
    }


def audit_split(path: Path, validator, verbose: bool = False) -> dict:
    examples = load_jsonl(path)
    n = len(examples)

    cross_tab = defaultdict(int)  # (gradeability, in_mp) -> count
    floor_risk_by_tier = defaultdict(int)
    records = []

    for i, ex in enumerate(examples):
        rec = classify_record(ex, validator)
        cross_tab[(rec["gradeability"], rec["in_mp"])] += 1
        if rec["floor_risk"]:
            floor_risk_by_tier[rec["gradeability"]] += 1
        records.append(rec)
        if verbose and (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{n}", file=sys.stderr)

    tier_totals = Counter(rec["gradeability"] for rec in records)
    in_mp_totals = Counter(rec["in_mp"] for rec in records)

    return {
        "path": str(path),
        "n_total": n,
        "cross_tab": {f"{g}|in_mp={m}": c for (g, m), c in cross_tab.items()},
        "tier_totals": dict(tier_totals),
        "tier_fractions": {k: round(v / n, 4) for k, v in tier_totals.items()} if n else {},
        "in_mp_totals": {str(k): v for k, v in in_mp_totals.items()},
        "floor_risk_by_tier": dict(floor_risk_by_tier),
        "floor_risk_fraction_overall": round(
            sum(floor_risk_by_tier.values()) / n, 4
        ) if n else 0.0,
        "records": records,
    }


def print_cross_tab(result: dict):
    n = result["n_total"]
    print(f"\n=== {result['path']}  (n={n}) ===")
    header = f"{'tier':<14}{'in_mp':<10}{'not_in_mp':<12}{'total':<10}{'frac':<8}{'floor_risk':<10}"
    print(header)
    print("-" * len(header))
    for tier in GRADEABILITY_TIERS:
        in_mp_c = result["cross_tab"].get(f"{tier}|in_mp=True", 0)
        not_mp_c = result["cross_tab"].get(f"{tier}|in_mp=False", 0)
        total = result["tier_totals"].get(tier, 0)
        frac = result["tier_fractions"].get(tier, 0.0)
        floor = result["floor_risk_by_tier"].get(tier, 0)
        print(f"{tier:<14}{in_mp_c:<10}{not_mp_c:<12}{total:<10}{frac:<8.4f}{floor:<10}")
    print(f"\nOverall floor-risk fraction (ground truth <= {FLOOR_CONST} raw reward): "
          f"{result['floor_risk_fraction_overall']:.4f}")
    light_only_frac = result["tier_fractions"].get("light_only", 0.0)
    print(f"Flat-thermo-reward fraction (light_only, the true GRPO blast radius): "
          f"{light_only_frac:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, dest="splits",
                     help="Path to a split jsonl file. Repeatable.")
    ap.add_argument("--formula-set", type=Path, required=True,
                     help="Path to mp_formula_set.pkl")
    ap.add_argument("--pd-index", type=Path, required=True,
                     help="Path to pd_index.json (sharded PD cache index, "
                          "NOT phase_diagrams.pkl - see HANDOFF_2 §2)")
    ap.add_argument("--project-root", type=Path, default=Path("."),
                     help="Repo root that pd_index.json shard paths are relative to")
    ap.add_argument("--out", type=Path, default=Path("coverage_audit_results.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print(
            "FATAL: validator.thermo_checker is None. Either --pd-index doesn't "
            "exist or load_validator's exists() check failed silently (this is "
            "the exact reward.py:175 short-circuit documented in HANDOFF_2 §2). "
            "The audit would report every record as light_only, which is not "
            "measuring coverage, it's measuring this bug again. Fix the path "
            "and rerun.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Loaded validator. thermo_checker active, "
          f"{len(validator.thermo_checker.pd_index)} shards indexed.", file=sys.stderr)

    all_results = {}
    for split_path in args.splits:
        p = Path(split_path)
        print(f"Auditing {p} ...", file=sys.stderr)
        result = audit_split(p, validator, verbose=args.verbose)
        all_results[p.name] = result
        print_cross_tab(result)

    # Records are large (one dict per example); keep them in the output
    # file for downstream stratification but don't blow up stdout.
    with args.out.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results (including per-record classifications) written to {args.out}")


if __name__ == "__main__":
    main()