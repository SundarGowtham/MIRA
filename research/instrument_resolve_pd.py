"""
instrument_resolve_pd.py
----------------------------
verify_refetch_reality.py confirmed the shard FILES for C-Fe-H-Li-N-O-P
and Fe-Li-O-P are completely fine (raw pickle.load succeeds, entry
counts match the refetch log exactly). Yet _resolve_pd, called on the
SAME chemistry via inspect_lfp_chemsys.py, reports total failure. The
bug is inside _resolve_pd/_get_pd's own logic, not the files.

This instruments _get_pd (same technique as pd_coverage_attribution.py
much earlier this session) to log every (chemsys_string, outcome) pair
_resolve_pd actually attempts internally, and prints the EXACT chemsys
string _resolve_pd computes from the given formulas - if that string
doesn't match what an independent computation expects, that's the bug.
If it matches but _get_pd still fails on it despite a raw pickle.load
succeeding on the identical path, the bug is inside _get_pd itself
(caching, project_root resolution, or something else entirely).

Usage:
  uv run python instrument_resolve_pd.py \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --target LiFePO4 \
      --precursor "FeC2O4·2H2O" --precursor "Li2CO3" --precursor "(NH4)2HPO4"
"""
from __future__ import annotations

import argparse
from pathlib import Path


def instrument(tc):
    original = tc._get_pd
    log = []

    def wrapped(chemsys):
        result = original(chemsys)
        status = "ok" if result is not None else ("in_index_but_failed" if chemsys in tc.pd_index else "not_in_index")
        log.append((chemsys, status))
        return result

    tc._get_pd = wrapped

    def restore():
        tc._get_pd = original

    return restore, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--target", required=True)
    ap.add_argument("--precursor", action="append", required=True)
    args = ap.parse_args()

    from pymatgen.core import Composition
    try:
        from core.reward import load_validator
    except ImportError:
        from reward import load_validator

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.")
        return
    tc = validator.thermo_checker

    core_formulas = [args.target] + args.precursor

    print(f"target: {args.target}")
    print(f"precursors: {args.precursor}\n")

    all_elements = set()
    per_formula_elements = {}
    for f in core_formulas:
        try:
            els = {str(el) for el in Composition(f).elements}
        except Exception as e:
            els = None
            print(f"  WARNING: '{f}' failed to parse independently: {e}")
        per_formula_elements[f] = els
        if els:
            all_elements.update(els)

    print("per-formula element parse (independent computation):")
    for f, els in per_formula_elements.items():
        print(f"  {f!r:<24} -> {sorted(els) if els else 'PARSE FAILED'}")

    independent_chemsys = "-".join(sorted(all_elements))
    print(f"\nindependently computed chemsys: {independent_chemsys}")
    print(f"is this key in pd_index?        {independent_chemsys in tc.pd_index}")

    print(f"\n--- now calling the REAL tc._resolve_pd(core_formulas) with instrumentation ---")
    restore, log = instrument(tc)
    try:
        pd, resolved_chemsys = tc._resolve_pd(core_formulas)
    finally:
        restore()

    print(f"\n_resolve_pd returned: pd={'<PhaseDiagram>' if pd is not None else 'None'}, "
          f"chemsys={resolved_chemsys!r}")
    print(f"\nfull internal attempt log ({len(log)} candidate(s) tried by _get_pd):")
    for chemsys, status in log:
        matches_independent = " <-- MATCHES independent computation" if chemsys == independent_chemsys else ""
        print(f"  {chemsys!r:<40} -> {status}{matches_independent}")

    if independent_chemsys not in [c for c, _ in log]:
        print(f"\n*** _resolve_pd NEVER EVEN TRIED the chemsys my independent computation "
              f"expected ({independent_chemsys!r}) - the string computation itself diverges "
              f"somewhere inside _resolve_pd. ***")
    elif pd is None:
        print(f"\n*** _resolve_pd DID try {independent_chemsys!r} (confirmed in the log above) "
              f"but _get_pd still returned None for it, despite a raw pickle.load() on the "
              f"identical file succeeding moments ago. The bug is inside _get_pd itself - "
              f"project_root resolution, path construction, or caching. ***")


if __name__ == "__main__":
    main()