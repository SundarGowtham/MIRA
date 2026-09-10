"""
inspect_lfp_chemsys.py
-------------------------
diagnose_lfp_coverage.py found LiFePO4 itself (the canonical, extremely
well-characterized base compound) landing in ok_but_no_entry - a PD loads
fine, but no entry matches "LiFePO4". Both _best_entry_for_formula
implementations are correct (straightforward reduced-formula matching),
so the real question is upstream: what chemsys did this actually resolve
to, and does that PD genuinely lack a LiFePO4-matching entry, or is
something else going on (e.g. the combined target+precursors chemsys
pulled in enough extra elements from an unusual precursor that MP's own
entry coverage for that specific large combination is sparse).

For each failing LiFePO4 record, prints: the required chemsys (target +
all precursor elements), whether it resolves via exact match or a
superset fallback, the resolved PD's total entry count, and every entry
in that PD whose composition lies purely within {Li, Fe, P, O} - if
LiFePO4 exists ANYWHERE in that PD it will show up here regardless of
exact reduced-formula string quirks.

Usage:
  uv run python inspect_lfp_chemsys.py \
      --synthesis data/raw/synthesis_clean.json \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --limit 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_COEFF_TERM = re.compile(r"([0-9]*\.?[0-9]+)\s+([A-Za-z0-9().]+)")


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_reaction_lhs_amounts(reaction_string: str) -> dict[str, float]:
    if not reaction_string or "==" not in reaction_string:
        return {}
    lhs = reaction_string.split("==", 1)[0]
    return {f: float(c) for c, f in _COEFF_TERM.findall(lhs) if _is_float(c)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthesis", type=Path, required=True)
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    from pymatgen.core import Composition
    try:
        from core.reward import load_validator
    except ImportError:
        from reward import load_validator

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)
    tc = validator.thermo_checker
    try:
        from core.validator import PredictedPrecursor, PredictedRoute
    except ImportError:
        from validator import PredictedPrecursor, PredictedRoute

    records = json.loads(args.synthesis.read_text())
    lfp_records = [
        r for r in records
        if r.get("target_formula") == "LiFePO4" and r.get("precursors")
    ]
    print(f"{len(lfp_records)} raw records with target_formula exactly 'LiFePO4'\n")

    for rec in lfp_records[:args.limit]:
        raw_precursor_formulas = [p.get("formula") for p in rec.get("precursors", []) if p.get("formula")]

        # Construct real objects instead of passing raw strings to _resolve_pd -
        # this is what actually exercises PredictedPrecursor's __post_init__
        # hydrate-notation normalization. Passing raw strings (the original bug
        # in this script) silently bypasses the fix entirely and produces
        # stale, misleading results - confirmed the hard way.
        precursor_objs = [PredictedPrecursor(formula=f) for f in raw_precursor_formulas]
        route = PredictedRoute(target_formula="LiFePO4", precursors=precursor_objs, operations=[])
        precursor_formulas = [p.formula for p in route.precursors]  # normalized
        core_formulas = [route.target_formula] + precursor_formulas

        all_elements = set()
        for f in core_formulas:
            try:
                all_elements.update(str(el) for el in Composition(f).elements)
            except Exception:
                pass
        exact_chemsys = "-".join(sorted(all_elements))

        print(f"=== record: raw precursors={raw_precursor_formulas} ===")
        if precursor_formulas != raw_precursor_formulas:
            print(f"  normalized precursors: {precursor_formulas}")
        print(f"  required chemsys (target + all precursors): {exact_chemsys}")
        print(f"  in pd_index (exact match)? {exact_chemsys in tc.pd_index}")

        pd, resolved_chemsys = tc._resolve_pd(core_formulas)
        if pd is None:
            print(f"  _resolve_pd: FAILED - no chemsys resolved at all\n")
            continue

        print(f"  _resolve_pd resolved to: {resolved_chemsys}  "
              f"({'exact match' if resolved_chemsys == exact_chemsys else 'SUPERSET fallback'})")
        print(f"  total entries in this PD: {len(pd.all_entries)}")

        # The actual ok_but_no_entry question: does a LiFePO4-matching entry
        # exist in this PD, and if so why does _best_entry_for_formula miss it?
        # (The old {Li,Fe,P,O}-subset filter was useless - it only ever
        # surfaced O2 reference entries, never answering whether the TARGET
        # compound is present.)
        target_red = Composition("LiFePO4").reduced_formula
        exact_target_entries = [
            e for e in pd.all_entries
            if e.composition.reduced_formula == target_red
        ]
        print(f"  entries with reduced_formula == '{target_red}': {len(exact_target_entries)}")
        for e in exact_target_entries[:5]:
            print(f"    {e.composition.reduced_formula:<16} "
                  f"energy_per_atom={e.energy_per_atom:.4f}  entry_id={getattr(e, 'entry_id', '?')}")

        # Directly exercise the actual lookup the validator uses
        try:
            best = validator.thermo_checker._best_entry_for_formula(pd, "LiFePO4")
            print(f"  _best_entry_for_formula('LiFePO4') returned: "
                  f"{'None' if best is None else best.composition.reduced_formula + ' @ ' + format(best.energy_per_atom, '.4f')}")
        except Exception as e:
            print(f"  _best_entry_for_formula('LiFePO4') RAISED: {type(e).__name__}: {e}")

        print()


if __name__ == "__main__":
    main()