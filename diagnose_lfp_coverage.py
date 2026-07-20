"""
diagnose_lfp_coverage.py
---------------------------
scan_family_coverage.py found LFP_LMP_olivine (LiFePO4/LiMnPO4-family)
at 253 matches, only 9 gradeable (96.4% ungradeable) - a severe outlier
against chemically adjacent families (LMO_spinel at 4.7% ungradeable,
LCO at 1.1%). Given LiFePO4 itself is one of the most characterized
compounds in Materials Project, this is almost certainly a PD-shard
coverage gap for the DOPED/SUBSTITUTED variants' multi-element
chemsystems, not a "the base compound is missing" problem.

Reuses the exact live _get_pd instrumentation from
pd_coverage_attribution.py (earlier this session) to distinguish:
  missing    - chemsys genuinely absent from pd_index (needs a new shard)
  corrupted  - chemsys IS in pd_index but fails to unpickle
  ok_but_no_entry - a PD loaded successfully but the target/precursor
                     formula isn't an entry in it (real chemistry gap,
                     not a coverage gap)
for every LFP-family record, and - the actually actionable output -
counts which SPECIFIC missing chemsystems recur most often, so you know
exactly what to build if the answer is "add more PD shards."

Usage:
  uv run python diagnose_lfp_coverage.py \
      --synthesis data/raw/synthesis_clean.json \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root .
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def lfp_lmp_classifier(comp, els: set[str]) -> bool:
    return ("Li" in els and "P" in els and "O" in els
            and ("Fe" in els or "Mn" in els)
            and "Ni" not in els and "Co" not in els)


_COEFF_TERM = re.compile(r"([0-9]*\.?[0-9]+)\s+([A-Za-z0-9().]+)")


def parse_reaction_lhs_amounts(reaction_string: str) -> dict[str, float]:
    if not reaction_string or "==" not in reaction_string:
        return {}
    lhs = reaction_string.split("==", 1)[0]
    amounts = {}
    for coeff, formula in _COEFF_TERM.findall(lhs):
        try:
            amounts[formula] = float(coeff)
        except ValueError:
            continue
    return amounts


def flatten_temps(heating_temperature) -> list[float]:
    flat = []
    for group in heating_temperature or []:
        if isinstance(group, (list, tuple)):
            for t in group:
                try:
                    flat.append(float(t))
                except (TypeError, ValueError):
                    pass
        else:
            try:
                flat.append(float(group))
            except (TypeError, ValueError):
                pass
    return flat


def build_route(record: dict):
    try:
        from core.validator import PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions
    except ImportError:
        from validator import PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions

    target = record.get("target_formula")
    raw_precursors = record.get("precursors", [])
    if not target or not raw_precursors:
        return None

    amounts = parse_reaction_lhs_amounts(record.get("reaction_string", ""))
    precursors = []
    for p in raw_precursors:
        formula = p.get("formula")
        if not formula:
            continue
        precursors.append(PredictedPrecursor(formula=formula, amount=amounts.get(formula, 1.0)))
    if not precursors:
        return None

    operations = []
    for op in record.get("operations", []):
        atmos = op.get("heating_atmosphere") or []
        conditions = PredictedConditions(
            heating_temperature=flatten_temps(op.get("heating_temperature")),
            heating_time=flatten_temps(op.get("heating_time")),
            heating_atmosphere=[str(a) for a in atmos if a],
            mixing_media=op.get("mixing_media"),
        )
        operations.append(PredictedOperation(type=op.get("type", ""), conditions=conditions))

    return PredictedRoute(target_formula=target, precursors=precursors, operations=operations,
                           reaction_string=record.get("reaction_string", ""))


def instrument(tc):
    original = tc._get_pd
    log = []

    def wrapped(chemsys):
        result = original(chemsys)
        if result is not None:
            status = "ok"
        elif chemsys in tc.pd_index:
            status = "corrupted"
        else:
            status = "missing"
        log.append((chemsys, status))
        return result

    tc._get_pd = wrapped

    def restore():
        tc._get_pd = original

    return restore, log


def classify(reaction_log, hull_log, gradeability_tag) -> str:
    """
    Uses the ACTUAL tag reaction_energy_per_atom returns (discrete/
    interpolated/ungradeable) - the same one thermo_tier is built from
    everywhere else this session - rather than inferring gradeability
    from dG/hull directly, which produced an inconsistent, looser
    definition on the first pass (counted hull-only success as
    "gradeable" even when thermo_tier would call it ungradeable, since
    target_stability_gradeability is a separate, independent tag never
    folded into thermo_tier anywhere else in this codebase).
    """
    if gradeability_tag in ("discrete", "interpolated"):
        return "gradeable"
    all_statuses = {s for _, s in reaction_log + hull_log}
    if "ok" in all_statuses:
        return "ok_but_no_entry"
    if "corrupted" in all_statuses:
        return "corrupted"
    if all_statuses <= {"missing"} or not all_statuses:
        return "missing"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthesis", type=Path, required=True)
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
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

    records = json.loads(args.synthesis.read_text())
    lfp_records = []
    for rec in records:
        target = rec.get("target_formula")
        if not target:
            continue
        try:
            comp = Composition(target)
        except Exception:
            continue
        els = {str(el) for el in comp.elements}
        if lfp_lmp_classifier(comp, els):
            lfp_records.append(rec)

    print(f"{len(lfp_records)} LFP/LMP-family records found in raw corpus.")

    category_counts = Counter()
    missing_chemsys_counts = Counter()
    ok_but_no_entry_examples = []
    corrupted_examples = []

    for rec in lfp_records:
        route = build_route(rec)
        if route is None:
            category_counts["build_failed"] += 1
            continue

        target = route.target_formula
        precs = [(p.formula, p.amount) for p in route.precursors]

        restore, log = instrument(tc)
        try:
            log.clear()
            try:
                dG, gradeability_tag = tc.reaction_energy_per_atom(precs, target, predicted_route=route)
            except Exception:
                dG, gradeability_tag = None, "ungradeable"
            reaction_log = list(log)

            log.clear()
            try:
                hull = tc.target_e_above_hull(target)
            except Exception:
                hull = None
            hull_log = list(log)
        finally:
            restore()

        cat = classify(reaction_log, hull_log, gradeability_tag)
        category_counts[cat] += 1

        if cat == "missing":
            # every attempted chemsys was absent - the exact chemsys of
            # (target + precursors) is the most informative one to report
            all_chemsys = {c for c, _ in reaction_log + hull_log}
            for c in all_chemsys:
                missing_chemsys_counts[c] += 1
        elif cat == "ok_but_no_entry" and len(ok_but_no_entry_examples) < 10:
            ok_but_no_entry_examples.append(target)
        elif cat == "corrupted" and len(corrupted_examples) < 10:
            corrupted_examples.append(target)

    print(f"\ncategory breakdown:")
    for cat, count in category_counts.most_common():
        print(f"  {cat:<20} {count:<6} ({count/len(lfp_records):.1%})")

    if missing_chemsys_counts:
        print(f"\nmost common MISSING chemsystems (need a new PD shard built for these):")
        for chemsys, count in missing_chemsys_counts.most_common(20):
            print(f"  {chemsys:<40} needed by {count} record(s)")

    if ok_but_no_entry_examples:
        print(f"\nexamples where PD loaded fine but target/precursor entry missing "
              f"(genuine chemistry gap, not a coverage gap):")
        for t in ok_but_no_entry_examples:
            print(f"  {t}")

    if corrupted_examples:
        print(f"\nexamples blocked by a corrupted shard (re-fetch candidates):")
        for t in corrupted_examples:
            print(f"  {t}")


if __name__ == "__main__":
    main()