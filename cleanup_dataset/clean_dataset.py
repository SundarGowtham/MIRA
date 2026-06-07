"""
clean_dataset.py
----------------
Clean the Kononova synthesis records by classifying each target formula:

  concrete    — parses cleanly as a Composition (BaTiO3, La0.5Sr0.5FeO3, ...)
  recovered   — parses after substituting fraction notation (LiNi1/3Co1/3Mn1/3O2)
  parametric  — contains variables (x, y, z, δ) or dummy element symbols
                (M, A, R, Ln, An) representing a *family* of materials, not a
                discrete compound. These have no single phase diagram and must
                be dropped.
  unparseable — fails to parse for some other reason (malformed syntax, etc.)

Concrete + recovered records are written to synthesis_clean.json with the
cleaned target_formula (recovered records get their fractions expanded).

Optionally also writes synthesis_dropped.json with one entry per dropped
record explaining why, and re-runs the PD coverage check on the cleaned
output so you see effective coverage numbers.

Usage:
    python clean_dataset.py
    python clean_dataset.py --keep-dropped --rerun-coverage
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from pymatgen.core import Composition, Element


PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
SYNTHESIS_FILE = DATA_RAW / "synthesis.json"
SYNTHESIS_CLEAN_FILE = DATA_RAW / "synthesis_clean.json"
SYNTHESIS_DROPPED_FILE = DATA_RAW / "synthesis_dropped.json"

# ---------------------------------------------------------------------------
# Parametric markers
# ---------------------------------------------------------------------------

# Lowercase letters that are subscript variables in solid-state chemistry
# notation (LiFexMn2-xO4 etc.). Note: deliberately NOT including any letter
# that's part of a real element symbol when standing alone — but lowercase
# alone never IS an element symbol, so this is safe.
PARAMETRIC_VARIABLES = frozenset({"x", "y", "z", "w", "v", "u", "δ", "ε", "ξ"})

# Uppercase tokens that look like element symbols but are placeholders for
# whole families (M = any transition metal, Ln = any lanthanide, etc.).
# These are not real elements, so the detection actually relies on the
# "unknown element symbol" check below — listed here for documentation.
KNOWN_DUMMY_ELEMENTS = frozenset({
    "M", "A", "R", "Ln", "An", "Ch", "Q", "T", "D", "E", "G", "J", "L",
    "Z", "X", "Re",  # Re=rhenium is real but often used as "rare earth"; ambiguous
})

# Set of all valid element symbols according to pymatgen.
VALID_ELEMENT_SYMBOLS = frozenset(str(e) for e in Element)

# Regex helpers
FRACTION_RE = re.compile(r"(\d+)/(\d+)")
ELEMENT_TOKEN_RE = re.compile(r"[A-Z][a-z]?")
SUBSCRIPT_VARIABLE_RE = re.compile(r"[\d\.\-\+]\s*([a-z]+)(?:[\d\.\-\+\(\)]|$)")
EXPRESSION_RE = re.compile(r"\d+(?:\.\d+)?\s*-\s*[a-zδεξ]")


def substitute_fractions(formula: str) -> str:
    """Replace n/m → decimal (rounded to 4 places). Skips n/0 silently."""
    def _repl(m):
        denom = int(m.group(2))
        if denom == 0:
            return m.group(0)   # leave n/0 unchanged; parser will reject it
        return str(round(int(m.group(1)) / denom, 4))
    return FRACTION_RE.sub(_repl, formula)


def detect_parametric_reason(formula: str) -> str | None:
    """
    If `formula` looks parametric (variables or dummy elements), return a
    short reason string. Otherwise return None.

    Tries to be specific about WHY a formula is parametric so the dropped
    bucket is auditable.
    """
    # Subtraction expressions: "1-x", "2-y", "0.5-z" — strongest signal
    if EXPRESSION_RE.search(formula):
        return "expression_with_variable"

    # Unknown capital-led element tokens (likely placeholders)
    for tok in ELEMENT_TOKEN_RE.findall(formula):
        if tok not in VALID_ELEMENT_SYMBOLS:
            return f"unknown_element_symbol:{tok}"

    # Lowercase subscript variables (x, y, z appearing as a subscript)
    for match in SUBSCRIPT_VARIABLE_RE.finditer(formula):
        var = match.group(1)
        if var in PARAMETRIC_VARIABLES:
            return f"variable_subscript:{var}"

    # Bare single lowercase letters not adjacent to numbers (rare but possible)
    bare = re.findall(r"\b[a-z]\b", formula)
    for tok in bare:
        if tok in PARAMETRIC_VARIABLES:
            return f"bare_variable:{tok}"

    return None


def classify_formula(raw: str) -> tuple[str, str | None, str]:
    """
    Classify a target formula.

    Returns (status, cleaned_formula, reason) where
        status ∈ {"concrete", "recovered", "parametric", "unparseable"}.
    """
    if not raw or not isinstance(raw, str):
        return ("unparseable", None, "empty_or_non_string")

    raw_stripped = raw.strip()

    # First check: do we have parametric markers? If so, bail immediately —
    # even if the formula happens to parse (some can, by accident), the
    # presence of a variable means it's a family, not a compound.
    reason = detect_parametric_reason(raw_stripped)
    if reason:
        return ("parametric", None, reason)

    # Try direct parse
    try:
        Composition(raw_stripped)
        return ("concrete", raw_stripped, "")
    except Exception:
        pass

    # Try with fraction substitution
    sub = substitute_fractions(raw_stripped)
    if sub != raw_stripped:
        # Re-check parametric (defensive — fraction sub shouldn't introduce them)
        if detect_parametric_reason(sub):
            return ("parametric", None, "after_fraction_sub_still_parametric")
        try:
            Composition(sub)
            return ("recovered", sub, "fractions_substituted")
        except Exception as e:
            return ("unparseable", None, f"parse_error_after_sub: {type(e).__name__}")

    # Couldn't parse and no fractions to substitute
    return ("unparseable", None, "parse_error")


# ---------------------------------------------------------------------------
# Main cleaning logic
# ---------------------------------------------------------------------------

def clean_dataset(synthesis_path: Path) -> tuple[list, list, dict]:
    """
    Return (clean_records, dropped_records, stats).

    Clean records have target_formula replaced with the cleaned version
    (concrete records are unchanged; recovered records have fractions
    expanded). Dropped records include the reason and the original formula.
    """
    if not synthesis_path.exists():
        sys.exit(f"ERROR: {synthesis_path} not found")

    with synthesis_path.open() as f:
        records = json.load(f)

    print(f"Loaded {len(records)} synthesis records")

    clean: list[dict] = []
    dropped: list[dict] = []
    status_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()

    for idx, rec in enumerate(records):
        raw = rec.get("target_formula", "")
        status, cleaned_formula, reason = classify_formula(raw)
        status_counter[status] += 1
        if reason:
            # Bucket reasons by their broad category for the summary
            # (e.g. "variable_subscript:x" → "variable_subscript")
            bucket = reason.split(":")[0]
            reason_counter[bucket] += 1

        if status in ("concrete", "recovered"):
            new_rec = dict(rec)
            new_rec["target_formula_original"] = raw
            new_rec["target_formula"] = cleaned_formula
            new_rec["_clean_status"] = status
            clean.append(new_rec)
        else:
            dropped.append({
                "index": idx,
                "target_formula": raw,
                "status": status,
                "reason": reason,
            })

    stats = {
        "total_records": len(records),
        "status_counts": dict(status_counter),
        "dropped_reason_buckets": dict(reason_counter),
        "kept": len(clean),
        "dropped": len(dropped),
    }
    return clean, dropped, stats


def print_summary(stats: dict, dropped: list):
    n = stats["total_records"]
    kept = stats["kept"]
    counts = stats["status_counts"]

    def pct(k):
        return f"{100 * k / n:.1f}%" if n else "—"

    print()
    print("=" * 72)
    print("  DATASET CLEANING SUMMARY")
    print("=" * 72)
    print()
    print(f"Total input records: {n}")
    print()
    print("Classification:")
    print(f"  concrete    : {counts.get('concrete', 0):6d}  ({pct(counts.get('concrete', 0))})")
    print(f"  recovered   : {counts.get('recovered', 0):6d}  ({pct(counts.get('recovered', 0))})  "
          f"← fraction substitution recovered these")
    print(f"  parametric  : {counts.get('parametric', 0):6d}  ({pct(counts.get('parametric', 0))})  "
          f"← variables/placeholders, dropped")
    print(f"  unparseable : {counts.get('unparseable', 0):6d}  ({pct(counts.get('unparseable', 0))})  "
          f"← syntactic errors, dropped")
    print()
    print(f"Kept (concrete + recovered): {kept:6d}  ({pct(kept)})")
    print()

    if stats["dropped_reason_buckets"]:
        print("Dropped-record reason breakdown:")
        for bucket, count in sorted(stats["dropped_reason_buckets"].items(),
                                    key=lambda x: -x[1]):
            print(f"  {bucket:35s} : {count:6d}")
        print()

    if dropped:
        # Show a sample of each reason type
        reason_examples: dict[str, list[str]] = {}
        for d in dropped:
            bucket = (d["reason"] or "").split(":")[0] or d["status"]
            reason_examples.setdefault(bucket, []).append(d["target_formula"])

        print("Sample dropped formulas by reason bucket:")
        for bucket, examples in sorted(reason_examples.items(),
                                       key=lambda x: -len(x[1])):
            print(f"  {bucket}:")
            for ex in examples[:4]:
                print(f"    {ex!r}")
            if len(examples) > 4:
                print(f"    ... and {len(examples) - 4} more")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis", type=Path, default=SYNTHESIS_FILE,
                        help="Input synthesis records JSON")
    parser.add_argument("--clean-out", type=Path, default=SYNTHESIS_CLEAN_FILE,
                        help="Output cleaned records JSON")
    parser.add_argument("--keep-dropped", action="store_true",
                        help="Also write synthesis_dropped.json with the dropped records")
    parser.add_argument("--rerun-coverage", action="store_true",
                        help="After cleaning, re-run pd_coverage_preflight on the clean output")
    parser.add_argument("--pd-index", type=Path, default=PROJECT_ROOT / "data" / "cache" / "pd_index.json",
                        help="Used only if --rerun-coverage is set")
    args = parser.parse_args()

    clean, dropped, stats = clean_dataset(args.synthesis)
    print_summary(stats, dropped)

    args.clean_out.parent.mkdir(parents=True, exist_ok=True)
    with args.clean_out.open("w") as f:
        json.dump(clean, f, indent=2)
    print(f"Wrote {len(clean)} clean records to {args.clean_out}")

    if args.keep_dropped:
        with SYNTHESIS_DROPPED_FILE.open("w") as f:
            json.dump(dropped, f, indent=2)
        print(f"Wrote {len(dropped)} dropped records to {SYNTHESIS_DROPPED_FILE}")

    if args.rerun_coverage:
        print()
        print("=" * 72)
        print("  RE-RUNNING PD COVERAGE ON CLEAN DATASET")
        print("=" * 72)
        try:
            import pd_coverage_preflight as preflight
            result = preflight.audit(args.clean_out, args.pd_index, include_precursors=False)
            preflight.print_report(result)
            print()
            print("Now with --include-precursors:")
            print()
            result2 = preflight.audit(args.clean_out, args.pd_index, include_precursors=True)
            preflight.print_report(result2)
        except ImportError:
            print("Could not import pd_coverage_preflight — run it manually:")
            print(f"  python pd_coverage_preflight.py --synthesis {args.clean_out}")


if __name__ == "__main__":
    main()