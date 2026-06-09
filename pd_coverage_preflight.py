"""
pd_coverage_preflight.py
------------------------
Audit: for every target in synthesis.json, check whether the local pd_index
covers its chemical system. Report:
  - exact coverage (chemsys hits an indexed PD directly)
  - superset coverage (chemsys ⊂ some indexed PD, so the validator's
    _resolve_pd will find it)
  - missing coverage (no indexed PD contains all the target's elements)

Run before scaling generation. Records with missing coverage will get the
"No phase diagram data computed for this system" message in their prompt,
which means the model falls back to pretraining-only output for those
targets — exactly the K2Ti4O9 case from the live run.

Usage:
    python pd_coverage_preflight.py
    python pd_coverage_preflight.py --report missing_chemsystems.json
    python pd_coverage_preflight.py --include-precursors  # also check
                                                          # carbonate/etc. ext.

Output: stdout summary + optional JSON of missing chemsystems for downstream
batch fetching from the MP API.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pymatgen.core import Composition


PROJECT_ROOT = Path(__file__).parent
SYNTHESIS_FILE = PROJECT_ROOT / "data" / "raw" / "synthesis.json"
PD_INDEX_FILE = PROJECT_ROOT / "data" / "cache" / "pd_index.json"


def chemsys_of(formula: str) -> str | None:
    try:
        els = sorted(str(el) for el in Composition(formula).elements)
        return "-".join(els)
    except Exception:
        return None


def elements_of(chemsys: str) -> set[str]:
    return set(chemsys.split("-"))


def find_superset(target_els: set[str], indexed_chemsystems: list[set[str]]) -> str | None:
    """Return the first indexed chemsys whose elements are a superset of target's, else None."""
    for cs in indexed_chemsystems:
        if target_els.issubset(cs):
            return "-".join(sorted(cs))
    return None


def audit(
    synthesis_path: Path,
    pd_index_path: Path,
    include_precursors: bool = False,
) -> dict:
    if not synthesis_path.exists():
        sys.exit(f"ERROR: {synthesis_path} not found")
    if not pd_index_path.exists():
        sys.exit(f"ERROR: {pd_index_path} not found")

    with synthesis_path.open() as f:
        records = json.load(f)
    with pd_index_path.open() as f:
        pd_index = json.load(f)

    print(f"Loaded {len(records)} synthesis records")
    print(f"Loaded {len(pd_index)} cached phase diagrams")
    print()

    # Cache indexed chemsystems as element sets for fast superset checking
    indexed_sets = [elements_of(cs) for cs in pd_index.keys()]
    indexed_set_lookup = {frozenset(s): "-".join(sorted(s)) for s in indexed_sets}

    # Buckets
    exact_hits: list[str] = []
    superset_hits: dict[str, str] = {}   # target → covering chemsys
    missing: list[tuple[str, str]] = []  # (target, chemsys)
    unparseable: list[str] = []

    # Auxiliary: how often does each missing chemsys appear?
    missing_chemsys_counter: Counter[str] = Counter()
    # Cross-reference: which targets need each missing chemsys?
    missing_chemsys_targets: dict[str, list[str]] = defaultdict(list)

    for rec in records:
        target = rec.get("target_formula", "")
        if not target:
            continue

        chemsys = chemsys_of(target)
        if chemsys is None:
            unparseable.append(target)
            continue

        target_els = elements_of(chemsys)

        # Also consider precursor elements if requested (catches the
        # carbonate/nitrate gap: target=Na2Ti3O7 needs Na-Ti-O, but the
        # reaction-energy check needs Na-C-Ti-O once Na2CO3 is the precursor)
        if include_precursors:
            for p in rec.get("precursors", []):
                p_chemsys = chemsys_of(p.get("formula", ""))
                if p_chemsys:
                    target_els |= elements_of(p_chemsys)
            chemsys = "-".join(sorted(target_els))

        if chemsys in pd_index:
            exact_hits.append(target)
            continue

        covering = find_superset(target_els, indexed_sets)
        if covering:
            superset_hits[target] = covering
        else:
            missing.append((target, chemsys))
            missing_chemsys_counter[chemsys] += 1
            missing_chemsys_targets[chemsys].append(target)

    total = len(exact_hits) + len(superset_hits) + len(missing) + len(unparseable)
    return {
        "records_total": len(records),
        "records_audited": total,
        "exact_hits": exact_hits,
        "superset_hits": superset_hits,
        "missing": missing,
        "missing_chemsys_counter": missing_chemsys_counter,
        "missing_chemsys_targets": dict(missing_chemsys_targets),
        "unparseable": unparseable,
        "include_precursors": include_precursors,
    }


def print_report(audit_result: dict):
    print("=" * 72)
    print("  PD COVERAGE PRE-FLIGHT")
    print("=" * 72)

    n_exact = len(audit_result["exact_hits"])
    n_superset = len(audit_result["superset_hits"])
    n_missing = len(audit_result["missing"])
    n_unparseable = len(audit_result["unparseable"])
    n_total = audit_result["records_audited"]

    def pct(n):
        return f"{100 * n / n_total:.1f}%" if n_total else "—"

    print()
    print(f"Total records: {audit_result['records_total']}")
    print(f"Audited:       {n_total}")
    print(f"Mode:          {'target + precursors' if audit_result['include_precursors'] else 'target elements only'}")
    print()
    print("Coverage breakdown:")
    print(f"  exact PD hit       : {n_exact:6d}  ({pct(n_exact)})")
    print(f"  superset PD hit    : {n_superset:6d}  ({pct(n_superset)})")
    print(f"  no coverage        : {n_missing:6d}  ({pct(n_missing)})")
    print(f"  unparseable target : {n_unparseable:6d}  ({pct(n_unparseable)})")
    print()
    print(f"Effective coverage (exact + superset): "
          f"{n_exact + n_superset:6d}  ({pct(n_exact + n_superset)})")
    print()

    if n_missing > 0:
        # Top missing chemsystems — these are the highest-leverage PDs to fetch
        print("=" * 72)
        print("  TOP 20 MISSING CHEMSYSTEMS (most targets affected per system)")
        print("=" * 72)
        print()
        print(f"  {'count':>6s}  {'chemsys':25s}  {'example targets'}")
        print(f"  {'-' * 6}  {'-' * 25}  {'-' * 40}")
        for cs, count in audit_result["missing_chemsys_counter"].most_common(20):
            examples = audit_result["missing_chemsys_targets"][cs][:3]
            example_str = ", ".join(examples)
            if count > 3:
                example_str += f", ... ({count - 3} more)"
            print(f"  {count:6d}  {cs:25s}  {example_str}")
        print()

    if n_unparseable > 0:
        print(f"Unparseable target formulas ({n_unparseable}):")
        for t in audit_result["unparseable"][:10]:
            print(f"  {t!r}")
        if n_unparseable > 10:
            print(f"  ... and {n_unparseable - 10} more")
        print()

    # Closing recommendation
    print("=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)
    coverage_pct = (n_exact + n_superset) / n_total if n_total else 0
    if coverage_pct >= 0.95:
        print(f"  ✓ {coverage_pct*100:.1f}% coverage. Safe to scale generation.")
    elif coverage_pct >= 0.85:
        print(f"  ⚠ {coverage_pct*100:.1f}% coverage. Tolerable but consider")
        print(f"    fetching the top-N missing chemsystems before scaling.")
    else:
        print(f"  ✗ {coverage_pct*100:.1f}% coverage. Too low — a significant")
        print(f"    fraction of records will get ungrounded prompts. Expand")
        print(f"    the PD cache before scaling.")
    print()
    if audit_result["include_precursors"]:
        print("  Mode = target+precursors. Coverage gaps here include the")
        print("  carbonate/nitrate/hydroxide extensions (C, H, N elements)")
        print("  that come from common precursors. Refetch PDs with these")
        print("  elements included if you want the thermodynamic_favorable")
        print("  check to grade reactions involving SrCO3, K2CO3, NaNO3, etc.")
    else:
        print("  Mode = target only. Re-run with --include-precursors to see")
        print("  the additional coverage needed for the reaction-energy check")
        print("  (which needs PDs covering precursor elements like C, H, N).")


def write_missing_report(audit_result: dict, output_path: Path):
    """Write missing chemsystems to JSON for downstream batch PD fetching."""
    missing_list = sorted(
        audit_result["missing_chemsys_counter"].items(),
        key=lambda x: -x[1],
    )
    report = {
        "mode": "target+precursors" if audit_result["include_precursors"] else "target-only",
        "total_missing_chemsystems": len(missing_list),
        "total_affected_records": sum(c for _, c in missing_list),
        "missing": [
            {
                "chemsys": cs,
                "affected_record_count": count,
                "example_targets": audit_result["missing_chemsys_targets"][cs][:5],
            }
            for cs, count in missing_list
        ],
    }
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Missing-chemsystems report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis", type=Path, default=SYNTHESIS_FILE,
                        help="Path to synthesis records JSON")
    parser.add_argument("--pd-index", type=Path, default=PD_INDEX_FILE,
                        help="Path to pd_index JSON")
    parser.add_argument("--include-precursors", action="store_true",
                        help="Also check coverage of precursor element sets")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional path to write missing chemsystems JSON")
    args = parser.parse_args()

    result = audit(args.synthesis, args.pd_index, include_precursors=args.include_precursors)
    print_report(result)

    if args.report:
        write_missing_report(result, args.report)


if __name__ == "__main__":
    main()