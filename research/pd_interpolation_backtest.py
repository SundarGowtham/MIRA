"""
pd_interpolation_backtest.py
-------------------------------
Last check before wiring interpolation into validator.py/gibbs_corrector.py
for real. pd_interpolation_probe.py showed the interpolated dG distribution
is sane for fractional targets (no ground truth to check against there -
that's WHY those targets needed interpolation in the first place). This
script checks interpolation against targets where we DO have ground
truth: the real_dG tier, where a discrete target entry already exists.

Prediction being tested: get_decomposition(target) is built ONLY from
STABLE entries. If the target itself is stable (on the hull, e_above_hull
~ 0), decomposition trivially returns {target: 1.0} and interpolated_dG
should equal discrete_dG almost exactly. If the target is slightly
unstable (e_above_hull > 0, common for real synthesized-but-imperfect
compounds), decomposition excludes the target's own (unstable) entry and
instead uses the surrounding stable phases, which sit strictly LOWER in
energy - so interpolated_dG should read MORE NEGATIVE than discrete_dG by
approximately e_above_hull(target).

  predicted_interpolated_dG = discrete_dG - e_above_hull(target)
  residual = interpolated_dG - predicted_interpolated_dG

If residual clusters tightly around 0 across the real_dG population, the
interpolation math is doing exactly the physics it's supposed to and
there's real confidence to generalize it to the fractional population
(where this check is impossible, since there's no discrete entry to
compare against - this is as close to ground truth as this problem gets).

If residual does NOT cluster near 0, something in the interpolation path
disagrees with the established discrete path and needs fixing BEFORE
generalizing it to the ~40% of the corpus that has no other check.

Usage: same flags as pd_interpolation_probe.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from pymatgen.core import Composition

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


def interpolated_reaction_dG_per_atom(tc, precursors, target_formula, predicted_route):
    """Identical logic to pd_interpolation_probe.py's fixed version."""
    core_formulas = [target_formula] + [f for f, _ in precursors]
    pd, _ = tc._resolve_pd(core_formulas)
    if pd is None:
        return None, "no_pd"

    target_comp = Composition(target_formula)
    T_K = extract_synthesis_temperature_K(predicted_route)

    try:
        decomp = pd.get_decomposition(target_comp)
    except Exception:
        return None, "decomposition_failed"
    if not decomp:
        return None, "empty_decomposition"

    g_hull_per_atom = 0.0
    for entry, amount in decomp.items():
        wrapped = _wrap_solid_at_T(entry, pd, T_K)
        if wrapped is None:
            return None, "decomp_wrap_failed"
        g_hull_per_atom += amount * wrapped.energy_per_atom

    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError

    synthetic_target = ComputedEntry(target_comp, g_hull_per_atom * target_comp.num_atoms)

    precursor_entries = []
    for f, _amt in precursors:
        p_red = Composition(f).reduced_formula
        raw = gibbs_best_entry(pd, p_red)
        if raw is None:
            return None, "precursor_entry_missing"
        wrapped_p = _wrap_solid_at_T(raw, pd, T_K)
        if wrapped_p is None:
            return None, "precursor_wrap_failed"
        precursor_entries.append(wrapped_p)

    reactant_elements = set()
    for e in precursor_entries + [synthetic_target]:
        reactant_elements.update(str(el) for el in e.composition.elements)
    gas_entries = []
    for sp in _GAS_SPECIES:
        if set(str(el) for el in Composition(sp).elements).issubset(reactant_elements):
            gas_entries.append(make_nist_gas_entry(sp, T_K))

    try:
        reaction = ComputedReaction(precursor_entries, [synthetic_target] + gas_entries)
    except ReactionError:
        return None, "reaction_error"

    try:
        coeff = reaction.get_coeff(Composition(target_comp.reduced_formula))
    except (ValueError, KeyError):
        return None, "get_coeff_failed"
    if coeff <= 1e-6:
        return None, "nonpositive_coefficient"

    atoms_per_target = target_comp.reduced_composition.num_atoms
    return float(reaction.calculated_reaction_energy) / (coeff * atoms_per_target), "ok"


def run(path: Path, validator, sample: int) -> dict:
    examples = load_jsonl(path)
    tc = validator.thermo_checker

    residuals = []
    e_hulls = []
    n_checked = 0
    n_skipped_no_discrete = 0
    n_interp_failed = 0
    worked = []

    for ex in examples:
        if n_checked >= sample:
            break
        target = ex["target"]
        try:
            route = parse_completion(ex["completion"], target)
        except ParseFailure:
            continue

        precs = [(p.formula, p.amount) for p in route.precursors]
        try:
            discrete_dG = tc.reaction_energy_per_atom(precs, target, predicted_route=route)
        except Exception:
            discrete_dG = None
        if discrete_dG is None:
            n_skipped_no_discrete += 1
            continue  # only backtest where ground truth exists

        try:
            e_hull = tc.target_e_above_hull(target)
        except Exception:
            e_hull = None
        if e_hull is None:
            n_skipped_no_discrete += 1
            continue

        interp_dG, stage = interpolated_reaction_dG_per_atom(tc, precs, target, route)
        n_checked += 1
        if interp_dG is None:
            n_interp_failed += 1
            continue

        predicted = discrete_dG - e_hull
        residual = interp_dG - predicted
        residuals.append(residual)
        e_hulls.append(e_hull)

        if len(worked) < 10:
            worked.append({
                "target": target,
                "discrete_dG": round(discrete_dG, 4),
                "e_above_hull": round(e_hull, 4),
                "interpolated_dG": round(interp_dG, 4),
                "predicted_interpolated_dG": round(predicted, 4),
                "residual": round(residual, 4),
            })

    def dist(xs):
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        return {
            "n": len(xs),
            "min": round(min(xs), 5),
            "median": round(statistics.median(xs), 5),
            "max": round(max(xs), 5),
            "mean": round(statistics.mean(xs), 5),
            "stdev": round(statistics.stdev(xs), 5) if len(xs) > 1 else 0.0,
            "frac_within_5meV": round(sum(1 for x in xs if abs(x) < 0.005) / len(xs), 3),
            "frac_within_20meV": round(sum(1 for x in xs if abs(x) < 0.020) / len(xs), 3),
            "frac_within_50meV": round(sum(1 for x in xs if abs(x) < 0.050) / len(xs), 3),
        }

    return {
        "path": str(path),
        "n_checked": n_checked,
        "n_skipped_no_discrete": n_skipped_no_discrete,
        "n_interp_failed": n_interp_failed,
        "residual_distribution_eV": dist(residuals),
        "e_above_hull_distribution_eV": dist(e_hulls),
        "worked_examples": worked,
    }


def print_summary(r: dict):
    print(f"\n=== {r['path']} ===")
    print(f"backtested: {r['n_checked']} (had both discrete_dG and e_above_hull)")
    print(f"skipped (no ground truth to compare against): {r['n_skipped_no_discrete']}")
    print(f"interpolation failed on a backtestable record: {r['n_interp_failed']}")
    d = r["residual_distribution_eV"]
    if d["n"] == 0:
        print("no residuals computed - nothing to report")
        return
    print(f"\nresidual = interpolated_dG - (discrete_dG - e_above_hull), eV/atom:")
    print(f"  n={d['n']}  median={d['median']}  mean={d['mean']}  stdev={d['stdev']}")
    print(f"  range=[{d['min']}, {d['max']}]")
    print(f"  within 5 meV/atom:  {d['frac_within_5meV']:.1%}")
    print(f"  within 20 meV/atom: {d['frac_within_20meV']:.1%}")
    print(f"  within 50 meV/atom: {d['frac_within_50meV']:.1%}  (MP's own reaction-energy noise floor)")
    print("\nworked examples (discrete vs interpolated):")
    for ex in r["worked_examples"]:
        print(f"  {ex['target']:<22} discrete={ex['discrete_dG']:<9} e_hull={ex['e_above_hull']:<8} "
              f"interp={ex['interpolated_dG']:<9} predicted={ex['predicted_interpolated_dG']:<9} "
              f"residual={ex['residual']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, dest="splits")
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("interpolation_backtest_results.json"))
    args = ap.parse_args()

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for split_path in args.splits:
        p = Path(split_path)
        print(f"Backtesting {p} ...", file=sys.stderr)
        r = run(p, validator, args.sample)
        all_results[p.name] = r
        print_summary(r)

    with args.out.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()