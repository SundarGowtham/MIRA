"""
pd_frontier_check.py
----------------------
Closes the loop opened by pd_failure_diagnosis.py's target_entry_missing
finding (98% of light_only). Hypothesis, from one sample (SnB0.6P0.4O2.9)
and from data_pull_3.py:build_mp_formula_set (which turns out to be
self-referential - built FROM the corpus's own formulas, filtered only
for placeholder dopant symbols via is_valid_concrete_formula, never
queried against the real MP API despite the filename): most
target_entry_missing failures are doped/non-stoichiometric solid-solution
compositions that structurally cannot have a discrete DFT entry in any
MP snapshot, no matter how complete the PD cache is.

This checks that directly: for every record, parse the RAW (unreduced)
composition and test whether every element's amount is within 1e-6 of an
integer. Cross-tab against gradeability tier and (for light_only) the
failure stage from pd_failure_diagnosis.py's logic.

If the hypothesis is right: target_entry_missing should be ~all
fractional. Any target_entry_missing record that comes back INTEGER
stoichiometry is a genuine second bug (a clean compound MP should have,
missing from an otherwise-good hull for some other reason) and worth
listing explicitly - that's what --out is for.

Usage: same flags as the other audit scripts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pymatgen.core import Composition

try:
    from core.reward import parse_completion, ParseFailure, load_validator
except ImportError:
    from reward import parse_completion, ParseFailure, load_validator

try:
    from core.gibbs_corrector import _best_entry_for_formula as gibbs_best_entry
except ImportError:
    from gibbs_corrector import _best_entry_for_formula as gibbs_best_entry


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def is_fractional(formula: str, tol: float = 1e-6) -> bool:
    """True if any element's amount in the RAW (unreduced) parsed formula
    is not within tol of an integer. This is the doped/solid-solution
    signature: SnB0.6P0.4O2.9 -> {Sn:1, B:0.6, P:0.4, O:2.9} -> fractional."""
    try:
        amounts = Composition(formula).as_dict().values()
    except Exception:
        return False  # unparseable; don't count as fractional, it's a different problem
    return any(abs(a - round(a)) > tol for a in amounts)


def gradeability_of(dG, hull) -> str:
    if dG is not None:
        return "real_dG"
    if hull is not None:
        return "hull_only"
    return "light_only"


def check_split(path: Path, validator) -> dict:
    examples = load_jsonl(path)
    tc = validator.thermo_checker

    tier_by_fractional = defaultdict(Counter)  # {is_fractional: Counter(tier)}
    target_entry_missing_by_fractional = Counter()
    non_fractional_target_entry_missing_samples = []
    n_parse_fail = 0

    for ex in examples:
        target = ex["target"]
        frac = is_fractional(target)

        try:
            route = parse_completion(ex["completion"], target)
        except ParseFailure:
            n_parse_fail += 1
            continue

        precs = [(p.formula, p.amount) for p in route.precursors]
        try:
            dG = tc.reaction_energy_per_atom(precs, target, predicted_route=route)
        except Exception:
            dG = None
        try:
            hull = tc.target_e_above_hull(target)
        except Exception:
            hull = None

        tier = gradeability_of(dG, hull)
        tier_by_fractional[frac][tier] += 1

        if tier == "light_only":
            # replicate just the entry-lookup step to confirm the stage
            core_formulas = [target] + [f for f, _ in precs]
            pd, _ = tc._resolve_pd(core_formulas)
            if pd is not None:
                target_red = Composition(target).reduced_formula
                entry = gibbs_best_entry(pd, target_red)
                if entry is None:
                    target_entry_missing_by_fractional[frac] += 1
                    if not frac and len(non_fractional_target_entry_missing_samples) < 15:
                        non_fractional_target_entry_missing_samples.append(target)

    return {
        "path": str(path),
        "n_total": len(examples),
        "n_parse_fail": n_parse_fail,
        "tier_by_fractional": {
            ("fractional" if k else "integer"): dict(v)
            for k, v in tier_by_fractional.items()
        },
        "target_entry_missing_by_fractional": {
            ("fractional" if k else "integer"): v
            for k, v in target_entry_missing_by_fractional.items()
        },
        "non_fractional_target_entry_missing_samples": non_fractional_target_entry_missing_samples,
    }


def print_summary(result: dict):
    print(f"\n=== {result['path']} (n={result['n_total']}) ===")
    print("gradeability tier by stoichiometry type:")
    for stoich, tiers in result["tier_by_fractional"].items():
        total = sum(tiers.values())
        print(f"  {stoich:<12} n={total:<6} {tiers}")
    print("\ntarget_entry_missing by stoichiometry type:")
    for stoich, count in result["target_entry_missing_by_fractional"].items():
        print(f"  {stoich:<12} {count}")
    if result["non_fractional_target_entry_missing_samples"]:
        print(f"\nINTEGER-stoichiometry targets that still hit target_entry_missing "
              f"(genuine second bug candidates, {len(result['non_fractional_target_entry_missing_samples'])} shown):")
        for t in result["non_fractional_target_entry_missing_samples"]:
            print(f"    {t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, dest="splits")
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("frontier_check_results.json"))
    args = ap.parse_args()

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for split_path in args.splits:
        p = Path(split_path)
        print(f"Checking {p} ...", file=sys.stderr)
        result = check_split(p, validator)
        all_results[p.name] = result
        print_summary(result)

    with args.out.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()