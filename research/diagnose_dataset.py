"""
diagnose_dataset.py
-------------------
Stratified analysis of the synthesis_with_traces.jsonl dataset.
Answers the key question: which of the 0.5 thermodynamic_favorable records
are "correct neutrals" (solid solutions, no MP entry) vs "validator failures"
(should have computed ΔG but didn't)?

Also profiles the full dataset by validator score band so you can make
a principled decision about what to keep for SFT.

Usage:
    uv run python diagnose_dataset.py
    uv run python diagnose_dataset.py --input data/processed/synthesis_with_traces.jsonl
    uv run python diagnose_dataset.py --sample 20  # sample N records per stratum for manual review
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "synthesis_with_traces.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_solid_solution(formula: str) -> bool:
    """
    Heuristic: does the formula contain fractional subscripts?
    These targets likely have no discrete MP entry.
    """
    # Patterns like 0.2, 0.33, 0.5, 1.2, etc. embedded in formula
    return bool(re.search(r'\d+\.\d+', formula))


def classify_thermo_neutral(record: dict) -> str:
    """
    Classify a record with thermodynamic_favorable == 0.5.

    Returns one of:
      'solid_solution'    - fractional formula, 0.5 is expected
      'has_dG'            - has dG_eV_atom despite 0.5 score (shouldn't happen; weird)
      'no_structure'      - Gibbs corrector likely failed (no structure in entry)
      'no_pd'             - chemsys not in cache (no T_K either)
      'gibbs_failed'      - has T_K but no dG_eV_atom → Bartel/reaction failed
      'unknown'           - can't determine
    """
    target = record.get("target", "")
    breakdown = record.get("validator_breakdown", {})

    dg = breakdown.get("thermodynamic_dG_eV_atom")
    t_k = breakdown.get("thermodynamic_T_K")

    if dg is not None:
        return "has_dG"  # scored 0.5 despite having ΔG — inspect manually

    if is_solid_solution(target):
        return "solid_solution"

    if t_k is None:
        # No T_K means _check_thermodynamics returned before reaching gibbs_corrector
        # → most likely chemsys not found in PD cache
        return "no_pd"

    # Has T_K but no dG → Gibbs corrector ran but returned None
    return "gibbs_failed"


# ---------------------------------------------------------------------------
# Score band classification
# ---------------------------------------------------------------------------

def score_band(score: float) -> str:
    if score >= 0.95:
        return "A: ≥0.95 (excellent)"
    elif score >= 0.85:
        return "B: 0.85–0.95 (good)"
    elif score >= 0.75:
        return "C: 0.75–0.85 (borderline)"
    elif score >= 0.65:
        return "D: 0.65–0.75 (weak)"
    else:
        return "E: <0.65 (poor)"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of records to sample per failure stratum for manual review")
    parser.add_argument("--export-failures", action="store_true",
                        help="Write gibbs_failed + no_pd records to a separate JSONL for inspection")
    args = parser.parse_args()

    print(f"Loading {args.input}")
    records = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records\n")

    # --- Overall score distribution ---
    band_counts: Counter = Counter()
    for r in records:
        s = r.get("validator_score", 0)
        band_counts[score_band(s)] += 1

    print("=" * 60)
    print("SCORE BAND DISTRIBUTION")
    print("=" * 60)
    for band in sorted(band_counts):
        count = band_counts[band]
        pct = 100 * count / len(records)
        bar = "█" * int(pct / 2)
        print(f"  {band:35s}  {count:5d}  ({pct:5.1f}%)  {bar}")
    print()

    # Recommended SFT cutoff
    keep_counts = sum(v for k, v in band_counts.items() if k[0] in ("A", "B"))
    keep_pct = 100 * keep_counts / len(records)
    print(f"  → Recommend keeping bands A+B (score ≥ 0.85): {keep_counts} records ({keep_pct:.1f}%)")
    print()

    # --- Thermodynamic favorable breakdown ---
    thermo_05 = [r for r in records if r.get("validator_breakdown", {}).get("thermodynamic_favorable") == 0.5]
    thermo_real = [r for r in records if r.get("validator_breakdown", {}).get("thermodynamic_favorable") != 0.5]

    print("=" * 60)
    print("THERMODYNAMIC_FAVORABLE = 0.5 BREAKDOWN")
    print("=" * 60)
    print(f"  Total records:         {len(records)}")
    print(f"  Thermo = 0.5:          {len(thermo_05)}  ({100*len(thermo_05)/len(records):.1f}%)")
    print(f"  Thermo computed:       {len(thermo_real)}  ({100*len(thermo_real)/len(records):.1f}%)")
    print()

    # Classify the 0.5 records
    class_counts: Counter = Counter()
    by_class: dict[str, list] = defaultdict(list)
    for r in thermo_05:
        cls = classify_thermo_neutral(r)
        class_counts[cls] += 1
        by_class[cls].append(r)

    print("  Classification of 0.5 records:")
    class_descriptions = {
        "solid_solution": "Solid solution (no discrete MP entry) — 0.5 is CORRECT",
        "no_pd":          "Chemsys not in PD cache — validator skipped thermo",
        "gibbs_failed":   "Gibbs corrector returned None (structure/balance issue)",
        "has_dG":         "Has ΔG but still scored 0.5 — BUG, inspect immediately",
        "unknown":        "Unclassified",
    }
    fixable = 0
    for cls in ["solid_solution", "no_pd", "gibbs_failed", "has_dG", "unknown"]:
        count = class_counts.get(cls, 0)
        pct = 100 * count / len(thermo_05) if thermo_05 else 0
        desc = class_descriptions[cls]
        print(f"    {cls:20s}  {count:5d}  ({pct:5.1f}%)  — {desc}")
        if cls in ("no_pd", "gibbs_failed", "has_dG"):
            fixable += count
    print()

    fixable_pct = 100 * fixable / len(thermo_05) if thermo_05 else 0
    print(f"  → Potentially fixable (no_pd + gibbs_failed + has_dG): {fixable} ({fixable_pct:.1f}% of 0.5 records)")
    solid_sol_pct = 100 * class_counts.get("solid_solution", 0) / len(thermo_05) if thermo_05 else 0
    print(f"  → Correct neutrals (solid_solution): {class_counts.get('solid_solution', 0)} ({solid_sol_pct:.1f}%)")
    print()

    # --- Sample of each failure class for manual review ---
    print("=" * 60)
    print(f"SAMPLE RECORDS PER FAILURE CLASS (n={args.sample})")
    print("=" * 60)
    for cls in ["gibbs_failed", "no_pd", "has_dG"]:
        recs = by_class.get(cls, [])
        if not recs:
            continue
        print(f"\n  [{cls.upper()}] — {len(recs)} records total, showing {min(args.sample, len(recs))}:")
        targets = [r.get("target", "?") for r in recs[:args.sample]]
        breakdowns = [r.get("validator_breakdown", {}) for r in recs[:args.sample]]
        for t, bd in zip(targets, breakdowns):
            t_k = bd.get("thermodynamic_T_K", "—")
            dg  = bd.get("thermodynamic_dG_eV_atom", "—")
            prec = [p.get("formula", "?") for p in
                    r.get("predicted_route", {}).get("precursors", [])
                    for r in [recs[targets.index(t)]]]
            print(f"    {t:40s}  T_K={t_k}  dG={dg}  precursors={prec}")

    # --- Check field: missing thermodynamic_T_K (old records) ---
    old_records = [r for r in records if "thermodynamic_T_K" not in r.get("validator_breakdown", {})]
    print()
    print("=" * 60)
    print("RECORDS MISSING thermodynamic_T_K (pre-Tier3 records)")
    print("=" * 60)
    print(f"  Count: {len(old_records)}")
    if old_records:
        print(f"  These records were scored before Tier 3.1 landed.")
        print(f"  → Rescore these with the new validator to get real ΔG values.")
    print()

    # --- Check field: stoichiometry failures ---
    stoich_fail = [r for r in records if r.get("validator_breakdown", {}).get("stoichiometry", 1.0) < 0.5]
    print("=" * 60)
    print("STOICHIOMETRY FAILURES (score < 0.5)")
    print("=" * 60)
    print(f"  Count: {len(stoich_fail)}")
    target_counts: Counter = Counter(r.get("target", "?") for r in stoich_fail)
    print("  Most common failing targets:")
    for t, c in target_counts.most_common(15):
        prec_sets = set()
        for r in stoich_fail:
            if r.get("target") == t:
                precs = tuple(sorted(p.get("formula","?") for p in r.get("predicted_route",{}).get("precursors",[])))
                prec_sets.add(precs)
        print(f"    {t:30s}  ×{c}  precursors: {list(prec_sets)[:2]}")
    print()

    # --- Score distribution of thermo=0.5 vs computed records ---
    print("=" * 60)
    print("IMPACT OF THERMO=0.5 ON OVERALL SCORE DISTRIBUTION")
    print("=" * 60)
    scores_with_thermo = [r.get("validator_score",0) for r in thermo_real]
    scores_without = [r.get("validator_score",0) for r in thermo_05]
    if scores_with_thermo:
        print(f"  Records with real ΔG:    mean={sum(scores_with_thermo)/len(scores_with_thermo):.3f}  n={len(scores_with_thermo)}")
    if scores_without:
        print(f"  Records with 0.5 thermo: mean={sum(scores_without)/len(scores_without):.3f}  n={len(scores_without)}")
    print()

    # --- Export failures for inspection ---
    if args.export_failures:
        failure_classes = {"gibbs_failed", "no_pd", "has_dG"}
        failures = [r for r in thermo_05 if classify_thermo_neutral(r) in failure_classes]
        out_path = args.input.parent / "thermo_failures.jsonl"
        with out_path.open("w") as f:
            for r in failures:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(failures)} failure records to {out_path}")

    # --- Final recommendation ---
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    high_quality = [r for r in records if r.get("validator_score", 0) >= 0.85]
    borderline   = [r for r in records if 0.65 <= r.get("validator_score", 0) < 0.85]
    poor         = [r for r in records if r.get("validator_score", 0) < 0.65]
    print(f"  High quality (≥0.85):   {len(high_quality):5d}  — safe for SFT")
    print(f"  Borderline (0.65–0.85): {len(borderline):5d}  — include after rescore")
    print(f"  Poor (<0.65):           {len(poor):5d}  — exclude from SFT")
    print()
    print(f"  If fixable thermo failures ({fixable}) can be resolved by rescore,")
    print(f"  many borderline records will likely move to high quality.")
    print()
    sft_estimate = len(high_quality) + len(borderline)
    print(f"  Estimated usable for SFT after rescore: ~{sft_estimate} records")


if __name__ == "__main__":
    main()