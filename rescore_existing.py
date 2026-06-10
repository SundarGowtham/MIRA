"""
rescore_existing.py
-------------------
Re-run validator over existing records in synthesis_with_traces.jsonl
WITHOUT regenerating from the LLM. Useful for:
  - Verifying validator fixes work on already-generated data
  - Updating scores after a validator change without spending another $$
  - Inspecting which records change scores and by how much

Writes a new file rather than overwriting, so you can diff old vs new.

Usage:
    uv run python rescore_existing.py
    uv run python rescore_existing.py --input data/processed/synthesis_with_traces.jsonl
    uv run python rescore_existing.py --dry-run --limit 100   # just print diffs
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from validator import SynthesisValidator, ThermoChecker
from generate_traces_openrouter import convert_to_predicted_route

PROJECT_ROOT   = Path(__file__).parent
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "processed" / "synthesis_with_traces.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "synthesis_with_traces_rescored.jsonl"
PD_INDEX_FILE  = PROJECT_ROOT / "data" / "cache" / "pd_index.json"
FORMULA_SET_FILE = PROJECT_ROOT / "data" / "cache" / "mp_formula_set.pkl"


def _to_jsonable(obj):
    """
    Recursively convert a record to JSON-safe types.
    Handles numpy scalars, Python bools, and nested dicts/lists.
    The validator now emits float metadata keys (thermodynamic_T_K,
    thermodynamic_dG_eV_atom) that may come back as numpy floats on some
    platforms; this sanitizes everything before json.dumps.
    """
    try:
        import numpy as np
        np_int   = (np.integer,)
        np_float = (np.floating,)
        np_array = np.ndarray
    except ImportError:
        np_int = np_float = np_array = ()

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, bool):
        return int(obj)
    if np_int and isinstance(obj, np_int):
        return int(obj)
    if np_float and isinstance(obj, np_float):
        return float(obj)
    if np_array and isinstance(obj, np_array):
        return obj.tolist()
    return obj


def rescore_record(record: dict, validator: SynthesisValidator) -> dict | None:
    """
    Re-validate one record. Returns the updated record (with new score/breakdown)
    or None if the record can't be re-validated (e.g. malformed predicted_route).
    """
    target = record.get("target")
    predicted = record.get("predicted_route")
    if not target or not predicted:
        return None

    route = convert_to_predicted_route(target, predicted)
    if route is None:
        return None

    new_score, new_breakdown = validator.validate(route, target)

    updated = dict(record)
    # Preserve the old scores for diffing
    updated["validator_score_old"] = record.get("validator_score")
    updated["validator_breakdown_old"] = record.get("validator_breakdown")
    updated["validator_score"] = new_score
    updated["validator_breakdown"] = new_breakdown
    # Cast to int so json.dumps never sees a bare Python bool
    updated["passed_validator"] = int(new_score >= 0.5)
    return updated


def diff_summary(old: dict, new: dict) -> str | None:
    """Return a one-line diff if scores differ meaningfully, else None."""
    o = round(old.get("validator_score", 0), 3)
    n = round(new.get("validator_score", 0), 3)
    if abs(o - n) < 0.005:
        return None

    o_breakdown = old.get("validator_breakdown", {})
    n_breakdown = new.get("validator_breakdown", {})
    changed_checks = []
    for k in set(o_breakdown) | set(n_breakdown):
        oc = round(o_breakdown.get(k, 0), 3)
        nc = round(n_breakdown.get(k, 0), 3)
        if abs(oc - nc) >= 0.005:
            changed_checks.append(f"{k}: {oc}→{nc}")
    target = old.get("target", "?")
    return f"  {target:25s}  total: {o:.3f}→{n:.3f}   [{', '.join(changed_checks)}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pd-index", type=Path, default=PD_INDEX_FILE)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N records (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write output file, just print score diffs")
    parser.add_argument("--show-diffs", action="store_true",
                        help="Print every record whose score changed")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    print(f"Loading records from {args.input}")
    records = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    if args.limit:
        records = records[:args.limit]
        print(f"Limited to {len(records)} records")

    print(f"Initializing validator with PD index: {args.pd_index}")
    thermo = ThermoChecker.from_sharded_cache(args.pd_index, PROJECT_ROOT)
    print(f"PD index has {len(thermo.pd_index)} shards")

    if not FORMULA_SET_FILE.exists():
        sys.exit(f"mp_formula_set.pkl not found at {FORMULA_SET_FILE}")
    with FORMULA_SET_FILE.open("rb") as f:
        mp_formula_set = pickle.load(f)
    print(f"Loaded mp_formula_set: {len(mp_formula_set)} formulas")

    validator = SynthesisValidator(mp_formula_set=mp_formula_set, thermo_checker=thermo)

    # Score tracking
    n_rescored = 0
    n_failed = 0
    score_changes: list[tuple[str, float, float]] = []  # (target, old, new)
    check_changes: Counter = Counter()  # which checks improved most

    output_records = []
    for i, rec in enumerate(records, 1):
        if i % 500 == 0:
            print(f"  [{i}/{len(records)}] processed")

        new_rec = rescore_record(rec, validator)
        if new_rec is None:
            n_failed += 1
            output_records.append(rec)  # keep original
            continue

        n_rescored += 1
        old_score = rec.get("validator_score", 0)
        new_score = new_rec["validator_score"]
        score_changes.append((rec.get("target", "?"), old_score, new_score))

        # Track which checks improved
        old_b = rec.get("validator_breakdown", {})
        new_b = new_rec["validator_breakdown"]
        for k in new_b:
            if abs(new_b.get(k, 0) - old_b.get(k, 0)) >= 0.01:
                check_changes[k] += 1

        if args.show_diffs:
            diff = diff_summary(rec, new_rec)
            if diff:
                print(diff)

        output_records.append(new_rec)

    # Summary
    print()
    print("=" * 60)
    print("RESCORING SUMMARY")
    print("=" * 60)
    print(f"Rescored:           {n_rescored}")
    print(f"Failed (kept old):  {n_failed}")

    if score_changes:
        improved   = sum(1 for _, o, n in score_changes if n > o + 0.005)
        worsened   = sum(1 for _, o, n in score_changes if n < o - 0.005)
        unchanged  = len(score_changes) - improved - worsened
        avg_delta  = sum(n - o for _, o, n in score_changes) / len(score_changes)
        avg_old    = sum(o for _, o, _ in score_changes) / len(score_changes)
        avg_new    = sum(n for _, _, n in score_changes) / len(score_changes)
        print()
        print(f"Score deltas:")
        print(f"  Improved:    {improved}  ({100*improved/len(score_changes):.1f}%)")
        print(f"  Unchanged:   {unchanged}  ({100*unchanged/len(score_changes):.1f}%)")
        print(f"  Worsened:    {worsened}  ({100*worsened/len(score_changes):.1f}%)")
        print(f"  Avg old:     {avg_old:.3f}")
        print(f"  Avg new:     {avg_new:.3f}")
        print(f"  Avg delta:   {avg_delta:+.3f}")

        # Threshold counts
        old_pass = sum(1 for _, o, _ in score_changes if o >= 0.5)
        new_pass = sum(1 for _, _, n in score_changes if n >= 0.5)
        print(f"  Pass count:  {old_pass} → {new_pass}")

    if check_changes:
        print()
        print("Which checks changed (counts):")
        for check, count in check_changes.most_common():
            print(f"  {check:30s}  {count}")

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for rec in output_records:
                f.write(json.dumps(_to_jsonable(rec)) + "\n")
        print()
        print(f"Wrote rescored records to {args.output}")
        print(f"Original input untouched at {args.input}")
    else:
        print()
        print("Dry run — no output written. Re-run without --dry-run to save.")


if __name__ == "__main__":
    main()