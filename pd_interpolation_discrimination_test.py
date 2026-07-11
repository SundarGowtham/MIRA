"""
pd_interpolation_discrimination_test.py
------------------------------------------
post_patch_gradeability_audit.py confirmed thermodynamic_favorable is no
longer flat at a sentinel for fractional targets - but it also showed
most interpolated thermo_score values landing at exactly 1.0. That audit
only ever grades the ONE correct completion per record, so it can't tell
"real recipes are genuinely favorable" (fine) apart from "the scoring
saturated at the ceiling regardless of route quality" (the same
flat-signal problem, just moved from 0.5 to 1.0). GRPO's advantage
computation needs variance ACROSS SAMPLED COMPLETIONS for the SAME
target, not just a non-constant value across different targets.

This tests that directly: takes real fractional-target (interpolated-
tier) records and their ground-truth route, then constructs deliberately
degraded variants of the SAME route for the SAME target:
  amounts_scrambled  - all precursor amounts multiplied by random
                        off-stoichiometry factors (breaks mass balance,
                        keeps the same precursor SET)
  one_amount_doubled - a milder version: only one precursor's amount
                        doubled (tests sensitivity at the margin, not
                        just catastrophic breakage)
  precursor_dropped  - one precursor removed entirely (if >1 present)
  precursor_swapped  - one precursor's formula replaced with a
                        chemically irrelevant reagent that shares no
                        elements with the target (tests whether an
                        obviously-wrong route gets penalized OR correctly
                        falls to "ungradeable", either of which is a
                        real, useful discrimination signal - as opposed
                        to silently scoring the same as the real route)

For each record, reports baseline (real route) vs each perturbation's
thermodynamic_favorable score, gradeability tag, and total reward. The
question this answers: does baseline consistently score >= its
perturbations? If yes across most records, interpolation carries real
discriminative signal. If perturbations often score AS HIGH as baseline,
the check is saturated and isn't actually useful for GRPO's within-group
advantage computation regardless of what the ground-truth-only audit
showed.

Usage: same flags as the other audit scripts, plus --sample (how many
interpolated-tier records to test - this does ~5 validate() calls per
record so keep it modest).

  uv run python pd_interpolation_discrimination_test.py \
      --split data/sft/train.jsonl \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --sample 40 \
      --out discrimination_test_results.json
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from pathlib import Path

from pymatgen.core import Composition

try:
    from core.reward import parse_completion, ParseFailure, load_validator
except ImportError:
    from reward import parse_completion, ParseFailure, load_validator


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def get_target_formula(ex: dict) -> str | None:
    if "target" in ex:
        return ex["target"]
    if "metadata" in ex and isinstance(ex["metadata"], dict) and "target_formula" in ex["metadata"]:
        return ex["metadata"]["target_formula"]
    if "target_formula" in ex:
        return ex["target_formula"]
    return None


# Generic reagents unlikely to share elements with most oxide/borate/
# phosphate targets in this corpus. Picked to be a clean contamination
# test, not to be chemically meaningful.
_CONTAMINANT_POOL = ["NaCl", "KBr", "CaF2", "MgCl2", "AgI", "ZnBr2"]


def pick_contaminant(route) -> str | None:
    used_elements = set()
    used_elements.update(str(el) for el in Composition(route.target_formula).elements)
    for p in route.precursors:
        try:
            used_elements.update(str(el) for el in Composition(p.formula).elements)
        except Exception:
            pass
    for candidate in _CONTAMINANT_POOL:
        cand_elements = set(str(el) for el in Composition(candidate).elements)
        if not cand_elements & used_elements:
            return candidate
    return None  # every candidate collided (rare) - skip this perturbation


def make_perturbations(route, rng: random.Random) -> dict:
    variants = {}

    # amounts_scrambled: every precursor's amount hit with an off-ratio factor
    v = copy.deepcopy(route)
    for p in v.precursors:
        factor = rng.choice([0.3, 0.4, 2.5, 3.0])
        p.amount = round(p.amount * factor, 4)
    variants["amounts_scrambled"] = v

    # one_amount_doubled: mild, single-precursor perturbation
    if route.precursors:
        v = copy.deepcopy(route)
        idx = rng.randrange(len(v.precursors))
        v.precursors[idx].amount = round(v.precursors[idx].amount * 2.0, 4)
        variants["one_amount_doubled"] = v

    # precursor_dropped: remove one precursor (only if it leaves >=1 behind)
    if len(route.precursors) > 1:
        v = copy.deepcopy(route)
        idx = rng.randrange(len(v.precursors))
        del v.precursors[idx]
        variants["precursor_dropped"] = v

    # precursor_swapped: replace one precursor with an irrelevant reagent
    contaminant = pick_contaminant(route)
    if contaminant is not None and route.precursors:
        v = copy.deepcopy(route)
        idx = rng.randrange(len(v.precursors))
        v.precursors[idx].formula = contaminant
        variants["precursor_swapped"] = v

    return variants


def run(path: Path, validator, sample: int, seed: int = 0) -> dict:
    examples = load_jsonl(path)
    rng = random.Random(seed)

    records = []
    n_tested = 0

    for ex in examples:
        if n_tested >= sample:
            break
        target = get_target_formula(ex)
        if target is None:
            continue
        try:
            route = parse_completion(ex["completion"], target)
        except ParseFailure:
            continue

        try:
            baseline_reward, baseline_breakdown = validator.validate(route, target)
        except Exception:
            continue

        if baseline_breakdown.get("thermodynamic_favorable_gradeability") != "interpolated":
            continue  # only testing the population this whole exercise was about

        n_tested += 1
        variants = make_perturbations(route, rng)
        variant_results = {}
        for name, v_route in variants.items():
            try:
                v_reward, v_breakdown = validator.validate(v_route, target)
            except Exception:
                v_reward, v_breakdown = None, {}
            variant_results[name] = {
                "reward": v_reward,
                "thermodynamic_favorable": v_breakdown.get("thermodynamic_favorable"),
                "gradeability": v_breakdown.get("thermodynamic_favorable_gradeability"),
                "amount_accuracy": v_breakdown.get("amount_accuracy"),
                "amount_accuracy_gradeability": v_breakdown.get("amount_accuracy_gradeability"),
            }

        records.append({
            "target": target,
            "baseline_reward": baseline_reward,
            "baseline_thermo_score": baseline_breakdown.get("thermodynamic_favorable"),
            "baseline_amount_accuracy": baseline_breakdown.get("amount_accuracy"),
            "variants": variant_results,
        })

    def _discrimination_stats(records, perturbation_names, baseline_key, variant_key):
        """Same aggregation logic, parameterized so it can score any
        (baseline_key, variant_key) pair - thermodynamic_favorable was
        the only metric here before; amount_accuracy now needs the exact
        same treatment, not a copy-pasted second loop."""
        stats = {}
        for pname in sorted(perturbation_names):
            deltas = []
            n_applicable = 0
            n_baseline_strictly_better = 0
            n_tied = 0
            n_perturbed_better = 0
            for r in records:
                if pname not in r["variants"]:
                    continue
                v = r["variants"][pname]
                b_score = r.get(baseline_key)
                v_score = v.get(variant_key)
                if b_score is None or v_score is None:
                    continue
                n_applicable += 1
                delta = b_score - v_score
                deltas.append(delta)
                if delta > 1e-9:
                    n_baseline_strictly_better += 1
                elif delta < -1e-9:
                    n_perturbed_better += 1
                else:
                    n_tied += 1
            stats[pname] = {
                "n_applicable": n_applicable,
                "n_baseline_strictly_better": n_baseline_strictly_better,
                "n_tied": n_tied,
                "n_perturbed_better": n_perturbed_better,
                "mean_delta": round(statistics.mean(deltas), 4) if deltas else None,
                "median_delta": round(statistics.median(deltas), 4) if deltas else None,
            }
        return stats

    perturbation_names = set()
    for r in records:
        perturbation_names.update(r["variants"].keys())

    discrimination_stats = _discrimination_stats(
        records, perturbation_names, "baseline_thermo_score", "thermodynamic_favorable"
    )
    amount_discrimination_stats = _discrimination_stats(
        records, perturbation_names, "baseline_amount_accuracy", "amount_accuracy"
    )

    return {
        "path": str(path),
        "n_tested": n_tested,
        "discrimination_stats": discrimination_stats,
        "amount_discrimination_stats": amount_discrimination_stats,
        "records": records,
    }


def print_summary(r: dict):
    print(f"\n=== {r['path']}  (interpolated-tier records tested: {r['n_tested']}) ===")
    print("\ndiscrimination by perturbation type (baseline_thermo_score - perturbed_thermo_score):")
    for pname, stats in r["discrimination_stats"].items():
        n = stats["n_applicable"]
        if n == 0:
            continue
        print(f"  {pname:<20} n={n:<4} "
              f"baseline_better={stats['n_baseline_strictly_better']:<4} "
              f"tied={stats['n_tied']:<4} "
              f"perturbed_better={stats['n_perturbed_better']:<4} "
              f"mean_delta={stats['mean_delta']:<8} median_delta={stats['median_delta']}")

    print("\namount_accuracy discrimination by perturbation type:")
    for pname, stats in r["amount_discrimination_stats"].items():
        n = stats["n_applicable"]
        if n == 0:
            continue
        print(f"  {pname:<20} n={n:<4} "
              f"baseline_better={stats['n_baseline_strictly_better']:<4} "
              f"tied={stats['n_tied']:<4} "
              f"perturbed_better={stats['n_perturbed_better']:<4} "
              f"mean_delta={stats['mean_delta']:<8} median_delta={stats['median_delta']}")

    print("\nworked examples (baseline vs each perturbation's thermo_score):")
    for rec in r["records"][:6]:
        print(f"  {rec['target']:<24} baseline={rec['baseline_thermo_score']}")
        for pname, v in rec["variants"].items():
            print(f"      {pname:<20} score={v['thermodynamic_favorable']}  tag={v['gradeability']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, dest="splits")
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("discrimination_test_results.json"))
    args = ap.parse_args()

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for split_path in args.splits:
        p = Path(split_path)
        print(f"Testing {p} ...", file=sys.stderr)
        r = run(p, validator, args.sample, args.seed)
        all_results[p.name] = r
        print_summary(r)

    with args.out.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()