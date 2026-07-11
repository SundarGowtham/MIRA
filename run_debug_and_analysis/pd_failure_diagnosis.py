"""
pd_failure_diagnosis.py
-------------------------
Follow-up to pd_coverage_attribution.py. That script (once its bucket()
bug is fixed) will show most light_only records have a PD that loads
fine - the failure is downstream, inside compute_reaction_gibbs_per_atom
or the hull entry lookup, both of which get swallowed to a bare None by
`except Exception: return None` in validator.py. This script replicates
their internal steps OUTSIDE that swallow, so each record's failure gets
attributed to one specific stage instead of a generic None.

Reaction path stages (mirrors gibbs_corrector.compute_reaction_gibbs_per_atom):
  no_pd                    - _resolve_pd itself returned None (coverage gap;
                              already characterized by pd_coverage_attribution.py)
  target_entry_missing     - target formula not found in the resolved PD
  precursor_entry_missing  - a precursor formula not found in the resolved PD
  target_wrap_failed       - target entry has no .structure, or
                              get_form_energy_per_atom raised (feeds directly
                              back to the has-.structure finding from
                              pd_shard_probe.py / pd_shard_census.py)
  precursor_wrap_failed    - same, for a precursor entry
  reaction_error           - ComputedReaction couldn't balance the stoichiometry
  get_coeff_failed         - target composition not found in the balanced reaction
  nonpositive_coefficient  - target coefficient <= 1e-6 (degenerate balance)
  unexpectedly_succeeded   - diagnosis produced a value; the original None came
                              from something outside this replicated path (flags
                              a divergence between this script and the real code)

Hull path stages:
  no_pd                 - _resolve_pd returned None
  target_entry_missing  - target formula not in the resolved PD
  hull_calc_exception   - pd.get_e_above_hull raised even with on_error="ignore"
  unexpectedly_succeeded

Usage: same flags as coverage_audit.py / pd_coverage_attribution.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pymatgen.core import Composition
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError

try:
    from core.reward import parse_completion, ParseFailure, load_validator
except ImportError:
    from reward import parse_completion, ParseFailure, load_validator

try:
    from core.gibbs_corrector import (
        extract_synthesis_temperature_K,
        _best_entry_for_formula as gibbs_best_entry,
        _wrap_solid_at_T,
        make_nist_gas_entry,
        _GAS_SPECIES,
    )
except ImportError:
    from gibbs_corrector import (
        extract_synthesis_temperature_K,
        _best_entry_for_formula as gibbs_best_entry,
        _wrap_solid_at_T,
        make_nist_gas_entry,
        _GAS_SPECIES,
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def diagnose_reaction(tc, precursors, target_formula, predicted_route):
    """Returns (stage, detail_str)."""
    core_formulas = [target_formula] + [f for f, _ in precursors]
    pd, _ = tc._resolve_pd(core_formulas)
    if pd is None:
        return "no_pd", ""

    T_K = extract_synthesis_temperature_K(predicted_route)
    target_red = Composition(target_formula).reduced_formula
    target_raw = gibbs_best_entry(pd, target_red)
    if target_raw is None:
        return "target_entry_missing", target_red

    precursor_raw = []
    for f, _amt in precursors:
        p_red = Composition(f).reduced_formula
        p_entry = gibbs_best_entry(pd, p_red)
        if p_entry is None:
            return "precursor_entry_missing", p_red
        precursor_raw.append(p_entry)

    target_gibbs = _wrap_solid_at_T(target_raw, pd, T_K)
    if target_gibbs is None:
        has_struct = hasattr(target_raw, "structure") and target_raw.structure is not None
        return "target_wrap_failed", f"{target_red} has_structure={has_struct}"

    precursor_gibbs = []
    for raw in precursor_raw:
        wrapped = _wrap_solid_at_T(raw, pd, T_K)
        if wrapped is None:
            has_struct = hasattr(raw, "structure") and raw.structure is not None
            return "precursor_wrap_failed", f"{raw.composition.reduced_formula} has_structure={has_struct}"
        precursor_gibbs.append(wrapped)

    reactant_elements = set()
    for raw in [target_raw] + precursor_raw:
        reactant_elements.update(str(el) for el in raw.composition.elements)
    gas_entries = []
    for sp in _GAS_SPECIES:
        sp_elements = set(str(el) for el in Composition(sp).elements)
        if sp_elements.issubset(reactant_elements):
            gas_entries.append(make_nist_gas_entry(sp, T_K))

    reactants = list(precursor_gibbs)
    products = [target_gibbs] + gas_entries
    try:
        reaction = ComputedReaction(reactants, products)
    except ReactionError as exc:
        return "reaction_error", str(exc)

    target_comp_reduced = Composition(target_red)
    try:
        coeff = reaction.get_coeff(target_comp_reduced)
    except (ValueError, KeyError) as exc:
        return "get_coeff_failed", str(exc)

    if coeff <= 1e-6:
        return "nonpositive_coefficient", str(coeff)

    return "unexpectedly_succeeded", ""


def diagnose_hull(tc, target_formula):
    pd, _ = tc._resolve_pd([target_formula])
    if pd is None:
        return "no_pd", ""
    target_entry = tc._best_entry_for_formula(pd, target_formula)
    if target_entry is None:
        return "target_entry_missing", target_formula
    try:
        val = pd.get_e_above_hull(target_entry, on_error="ignore")
    except Exception as exc:
        return "hull_calc_exception", str(exc)
    if val is None:
        return "hull_calc_returned_none", ""
    return "unexpectedly_succeeded", str(val)


def gradeability_of(dG, hull) -> str:
    if dG is not None:
        return "real_dG"
    if hull is not None:
        return "hull_only"
    return "light_only"


def diagnose_split(path: Path, validator, max_examples_per_stage: int = 5) -> dict:
    examples = load_jsonl(path)
    tc = validator.thermo_checker

    reaction_stages = Counter()
    hull_stages = Counter()
    samples = defaultdict(list)
    n_light_only = 0

    for ex in examples:
        target = ex["target"]
        try:
            route = parse_completion(ex["completion"], target)
        except ParseFailure:
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

        if gradeability_of(dG, hull) != "light_only":
            continue
        n_light_only += 1

        r_stage, r_detail = diagnose_reaction(tc, precs, target, route)
        h_stage, h_detail = diagnose_hull(tc, target)
        reaction_stages[r_stage] += 1
        hull_stages[h_stage] += 1

        if len(samples[f"reaction:{r_stage}"]) < max_examples_per_stage:
            samples[f"reaction:{r_stage}"].append({"target": target, "detail": r_detail})
        if len(samples[f"hull:{h_stage}"]) < max_examples_per_stage:
            samples[f"hull:{h_stage}"].append({"target": target, "detail": h_detail})

    return {
        "path": str(path),
        "n_light_only": n_light_only,
        "reaction_stages": dict(reaction_stages),
        "hull_stages": dict(hull_stages),
        "samples": {k: v for k, v in samples.items()},
    }


def print_summary(result: dict):
    n = result["n_light_only"]
    print(f"\n=== {result['path']}  (light_only={n}) ===")
    print("reaction failure stage:")
    for stage, count in sorted(result["reaction_stages"].items(), key=lambda kv: -kv[1]):
        print(f"  {stage:<28}{count:<8}{count/n:.2%}")
    print("hull failure stage:")
    for stage, count in sorted(result["hull_stages"].items(), key=lambda kv: -kv[1]):
        print(f"  {stage:<28}{count:<8}{count/n:.2%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, dest="splits")
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("failure_diagnosis_results.json"))
    args = ap.parse_args()

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for split_path in args.splits:
        p = Path(split_path)
        print(f"Diagnosing {p} ...", file=sys.stderr)
        result = diagnose_split(p, validator)
        all_results[p.name] = result
        print_summary(result)

    with args.out.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()