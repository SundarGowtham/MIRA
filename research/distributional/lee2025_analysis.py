#!/usr/bin/env python
"""
research/distributional/lee2025_analysis.py — Phase 15 Task 3.
Pre-registration: docs/phases/PHASE15_LEE_PREREG.md (committed before
this script touched any data; primary and secondary endpoints below are
[C], everything else [E], per that file).

Source: Lee, Cruse, Baibakova, Ceder, Jain, Sci. Data 12:1969 (2025).
data/external/lee2025/SS_rxns_80806.json (gitignored), downloaded from
https://ndownloader.figshare.com/files/58973674 (Figshare API-resolved
direct link; the DOI landing page and figshare.com itself both blocked
automated fetch -- api.figshare.com/v2/articles/30423274 worked).
md5 2cca759682be4689e7d0d3d882d12909, matches Figshare's supplied hash
exactly (verified before use).

SCHEMA [E], confirmed by direct inspection before writing this script
(not assumed): each of the 80,806 records has `target` (list of dicts
with `material_formula`), `precursors` (list of dicts with
`material_formula`), `target_reaction`, and `impurity_reaction` -- a
list, ALWAYS present (0/80806 missing), EMPTY when no impurity phase was
reported, non-empty (list of reaction tuples) when one was. This is the
impurity-phase indicator used for the secondary endpoint. Overall corpus
impurity rate (all records, not alkali-restricted): 5.9% (4746/80806).

Usage (tmux -- the file is 439MB uncompressed, pure Python/JSON, no
pymatgen needed for this script's own logic, but run in tmux per the
project's own "always tmux" convention regardless):
  uv run python research/distributional/lee2025_analysis.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/external/lee2025/SS_rxns_80806.json")
OUT_JSON = Path("results/external/lee2025_precursor_usage.json")
ASTRAL_PATH = Path("misc/astral_validation_set.json")

ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
BARE_OXIDE = {"Li2O": "Li", "Na2O": "Na", "K2O": "K", "Rb2O": "Rb", "Cs2O": "Cs"}
CARBONATE = {"Li2CO3": "Li", "Na2CO3": "Na", "K2CO3": "K", "Rb2CO3": "Rb", "Cs2CO3": "Cs"}

_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def strip_hydrate(formula: str | None) -> str:
    """A·nH2O -> A (light normalization; this dataset's own hydrate
    notation uses the same middle-dot convention validator.py's
    expand_hydrate_notation handles for this project's own model outputs).
    Some records have a null material_formula (text-mining miss) --
    treated as an empty, unmatchable formula, not an error."""
    if not formula:
        return ""
    if "·" in formula:
        return formula.split("·")[0]
    return formula


def formula_elements(formula: str) -> set[str]:
    """Cheap regex element extraction -- avoids a pymatgen dependency for
    a text-mined dataset whose formulas are not guaranteed pymatgen-
    parseable (oxygen-deficiency notation, composite/doped strings)."""
    formula = strip_hydrate(formula)
    try:
        return {m.group(1) for m in _ELEMENT_RE.finditer(formula) if m.group(1)}
    except Exception:
        return set()


def target_alkalis(target_formula: str) -> set[str]:
    return formula_elements(target_formula) & ALKALI


def classify_alkali_source(precursor_formulas: list[str], alkali_el: str) -> str:
    for p in precursor_formulas:
        p_stripped = strip_hydrate(p)
        if p_stripped in BARE_OXIDE and BARE_OXIDE[p_stripped] == alkali_el:
            return "bare_oxide"
    for p in precursor_formulas:
        p_stripped = strip_hydrate(p)
        if p_stripped in CARBONATE and CARBONATE[p_stripped] == alkali_el:
            return "carbonate"
    for p in precursor_formulas:
        if alkali_el in formula_elements(p):
            return "other"
    return "missing"


def record_sourcing_category(precursor_formulas: list[str], alkalis: set[str]) -> str:
    """Whole-record classification for the secondary (impurity-rate)
    endpoint: does this record use a bare-oxide alkali source, a
    carbonate one, both (multi-alkali target using different classes per
    element, or ambiguous), or other/none."""
    cats = {classify_alkali_source(precursor_formulas, el) for el in alkalis}
    has_bare = "bare_oxide" in cats
    has_carb = "carbonate" in cats
    if has_bare and has_carb:
        return "both"
    if has_bare:
        return "bare_oxide_only"
    if has_carb:
        return "carbonate_only"
    return "other_only"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def diff_ci_normal_approx(k1: int, n1: int, k2: int, n2: int, z: float = 1.96):
    """95% CI on (p1 - p2) via normal approximation to the difference of
    two independent proportions -- adequate given both group sizes here
    (thousands), per the pre-reg's own stated fallback rule."""
    if n1 == 0 or n2 == 0:
        return (None, None)
    p1, p2 = k1 / n1, k2 / n2
    se = ((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)) ** 0.5
    diff = p1 - p2
    return (diff - z * se, diff + z * se)


def main():
    print(f"loading {DATA_PATH} ({DATA_PATH.stat().st_size / 1e6:.0f} MB)...", flush=True)
    data = json.loads(DATA_PATH.read_text())
    print(f"n records: {len(data)}")

    overall_impure = sum(1 for r in data if r.get("impurity_reaction"))
    print(f"overall corpus impurity rate: {overall_impure}/{len(data)} = {100*overall_impure/len(data):.1f}%")

    # -----------------------------------------------------------------
    # PRIMARY [C]: per-(record, alkali-element) triple source-type fractions
    # -----------------------------------------------------------------
    triple_counts = Counter()
    n_triples = 0
    alkali_records = []  # (record, alkalis, precursor_formulas, is_impure) for records with >=1 alkali
    for r in data:
        targets = r.get("target") or []
        if not targets:
            continue
        target_formula = targets[0].get("material_formula", "")
        alkalis = target_alkalis(target_formula)
        if not alkalis:
            continue
        precursor_formulas = [p.get("material_formula", "") for p in (r.get("precursors") or [])]
        is_impure = bool(r.get("impurity_reaction"))
        alkali_records.append((r, target_formula, alkalis, precursor_formulas, is_impure))
        for el in alkalis:
            cat = classify_alkali_source(precursor_formulas, el)
            triple_counts[cat] += 1
            n_triples += 1

    print(f"\n== PRIMARY [C]: alkali-containing records: {len(alkali_records)}, "
          f"triples: {n_triples} ==")
    for cat in ["bare_oxide", "carbonate", "other", "missing"]:
        n = triple_counts[cat]
        share = n / n_triples if n_triples else None
        print(f"  {cat:<12} {n:6d}  ({share:.1%})" if share is not None else f"  {cat:<12} 0")

    bare_share = triple_counts["bare_oxide"] / n_triples
    carb_share = triple_counts["carbonate"] / n_triples
    primary_bare_pred_holds = bare_share < 0.05
    primary_carb_pred_holds = carb_share > 0.50
    print(f"\nPre-registered prediction: bare-oxide < 5% -> "
          f"{'HOLDS' if primary_bare_pred_holds else 'DOES NOT HOLD'} ({bare_share:.1%})")
    print(f"Pre-registered prediction: carbonate > 50% -> "
          f"{'HOLDS' if primary_carb_pred_holds else 'DOES NOT HOLD'} ({carb_share:.1%})")

    # -----------------------------------------------------------------
    # SECONDARY [C]: impurity-phase rate, bare-oxide-only vs carbonate-only
    # -----------------------------------------------------------------
    record_cats = Counter()
    cat_impure = Counter()
    target_to_cats: dict[str, set[str]] = {}
    for r, target_formula, alkalis, precursor_formulas, is_impure in alkali_records:
        cat = record_sourcing_category(precursor_formulas, alkalis)
        record_cats[cat] += 1
        if is_impure:
            cat_impure[cat] += 1
        target_to_cats.setdefault(target_formula, set()).add(cat)

    print(f"\n== SECONDARY [C]: per-record sourcing category, impurity rate ==")
    for cat in ["bare_oxide_only", "carbonate_only", "both", "other_only"]:
        n = record_cats[cat]
        n_imp = cat_impure[cat]
        rate = n_imp / n if n else None
        ci = wilson_ci(n_imp, n) if n else (None, None)
        print(f"  {cat:<16} n={n:6d}  impure={n_imp:5d}  rate={rate:.1%}  "
              f"95% CI=[{ci[0]:.1%},{ci[1]:.1%}]" if rate is not None else f"  {cat:<16} n=0")

    n_bare, n_imp_bare = record_cats["bare_oxide_only"], cat_impure["bare_oxide_only"]
    n_carb, n_imp_carb = record_cats["carbonate_only"], cat_impure["carbonate_only"]
    rate_bare = n_imp_bare / n_bare if n_bare else None
    rate_carb = n_imp_carb / n_carb if n_carb else None
    diff = (rate_bare - rate_carb) if (rate_bare is not None and rate_carb is not None) else None
    diff_ci = diff_ci_normal_approx(n_imp_bare, n_bare, n_imp_carb, n_carb) if diff is not None else (None, None)
    print(f"\nPOOLED: bare-oxide impurity rate={rate_bare:.1%} (n={n_bare}) vs "
          f"carbonate impurity rate={rate_carb:.1%} (n={n_carb})")
    print(f"  difference (bare - carb) = {diff:+.1%}, 95% CI=[{diff_ci[0]:+.1%},{diff_ci[1]:+.1%}]")
    pred_holds = diff is not None and diff >= 0
    print(f"  Pre-registered prediction (bare-oxide impurity rate >= carbonate): "
          f"{'HOLDS' if pred_holds else 'DOES NOT HOLD'}")

    # Restricted to targets that appear with BOTH sourcing choices somewhere in the corpus
    both_targets = {t for t, cats in target_to_cats.items()
                    if "bare_oxide_only" in cats and "carbonate_only" in cats}
    print(f"\nTargets appearing with BOTH bare-oxide-only and carbonate-only records: {len(both_targets)}")
    n_bare_r, n_imp_bare_r, n_carb_r, n_imp_carb_r = 0, 0, 0, 0
    for r, target_formula, alkalis, precursor_formulas, is_impure in alkali_records:
        if target_formula not in both_targets:
            continue
        cat = record_sourcing_category(precursor_formulas, alkalis)
        if cat == "bare_oxide_only":
            n_bare_r += 1
            n_imp_bare_r += int(is_impure)
        elif cat == "carbonate_only":
            n_carb_r += 1
            n_imp_carb_r += int(is_impure)
    rate_bare_r = n_imp_bare_r / n_bare_r if n_bare_r else None
    rate_carb_r = n_imp_carb_r / n_carb_r if n_carb_r else None
    diff_r = (rate_bare_r - rate_carb_r) if (rate_bare_r is not None and rate_carb_r is not None) else None
    diff_ci_r = diff_ci_normal_approx(n_imp_bare_r, n_bare_r, n_imp_carb_r, n_carb_r) if diff_r is not None else (None, None)
    print(f"RESTRICTED to those {len(both_targets)} targets: "
          f"bare={rate_bare_r:.1%} (n={n_bare_r}) vs carb={rate_carb_r:.1%} (n={n_carb_r})"
          if rate_bare_r is not None and rate_carb_r is not None else "insufficient data")
    if diff_r is not None:
        print(f"  difference = {diff_r:+.1%}, 95% CI=[{diff_ci_r[0]:+.1%},{diff_ci_r[1]:+.1%}]")

    # -----------------------------------------------------------------
    # [E] step 5: ammonium-phosphate vs H3PO4 vs P2O5
    # -----------------------------------------------------------------
    print(f"\n== [E] step 5: phosphorus source usage, targets containing P ==")
    def is_ammonium(f):
        els = formula_elements(f)
        return "N" in els and "H" in els
    p_counts = Counter()
    p_impure = Counter()
    for r in data:
        targets = r.get("target") or []
        if not targets:
            continue
        target_formula = targets[0].get("material_formula", "")
        if "P" not in formula_elements(target_formula):
            continue
        precursor_formulas = [p.get("material_formula", "") for p in (r.get("precursors") or [])]
        is_impure = bool(r.get("impurity_reaction"))
        has_ammonium = any(is_ammonium(p) for p in precursor_formulas)
        has_h3po4 = any(strip_hydrate(p) == "H3PO4" for p in precursor_formulas)
        has_p2o5 = any(strip_hydrate(p) in ("P2O5", "P4O10") for p in precursor_formulas)
        if has_ammonium:
            p_counts["ammonium_phosphate"] += 1
            p_impure["ammonium_phosphate"] += int(is_impure)
        elif has_h3po4:
            p_counts["H3PO4"] += 1
            p_impure["H3PO4"] += int(is_impure)
        elif has_p2o5:
            p_counts["P2O5"] += 1
            p_impure["P2O5"] += int(is_impure)
        else:
            p_counts["other"] += 1
            p_impure["other"] += int(is_impure)
    for cat, n in p_counts.items():
        rate = p_impure[cat] / n if n else None
        print(f"  {cat:<20} n={n:5d}  impure_rate={rate:.1%}" if rate is not None else f"  {cat:<20} n=0")

    # -----------------------------------------------------------------
    # [E] step 6: ASTRAL target overlap
    # -----------------------------------------------------------------
    print(f"\n== [E] step 6: ASTRAL 35-target overlap ==")
    astral = json.loads(ASTRAL_PATH.read_text())
    astral_targets = {t["target"] for t in astral["targets"]}
    lee_targets_by_formula: dict[str, list] = {}
    for r in data:
        targets = r.get("target") or []
        if not targets:
            continue
        tf = targets[0].get("material_formula", "")
        lee_targets_by_formula.setdefault(tf, []).append(r)
    overlap = astral_targets & set(lee_targets_by_formula.keys())
    print(f"ASTRAL targets appearing in Lee et al.: {len(overlap)}/35: {sorted(overlap)}")
    overlap_detail = []
    for t in sorted(overlap):
        recs = lee_targets_by_formula[t]
        for rec in recs[:5]:  # cap per target for readability
            precs = [p.get("material_formula") for p in (rec.get("precursors") or [])]
            overlap_detail.append({"target": t, "precursors": precs,
                                   "is_impure": bool(rec.get("impurity_reaction"))})
    for d in overlap_detail:
        print(f"  {d['target']:<16} {d['precursors']}  impure={d['is_impure']}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "source_md5": "2cca759682be4689e7d0d3d882d12909",
        "n_records": len(data),
        "overall_impurity_rate": overall_impure / len(data),
        "primary": {
            "n_alkali_records": len(alkali_records), "n_triples": n_triples,
            "triple_counts": dict(triple_counts),
            "bare_share": bare_share, "carbonate_share": carb_share,
            "prediction_bare_lt_5pct_holds": primary_bare_pred_holds,
            "prediction_carb_gt_50pct_holds": primary_carb_pred_holds,
        },
        "secondary_pooled": {
            "record_cats": dict(record_cats), "cat_impure": dict(cat_impure),
            "rate_bare_oxide": rate_bare, "rate_carbonate": rate_carb,
            "difference": diff, "difference_95ci": diff_ci,
            "prediction_bare_ge_carb_holds": pred_holds,
        },
        "secondary_restricted_both_sourcing_targets": {
            "n_targets": len(both_targets),
            "rate_bare_oxide": rate_bare_r, "n_bare": n_bare_r,
            "rate_carbonate": rate_carb_r, "n_carb": n_carb_r,
            "difference": diff_r, "difference_95ci": diff_ci_r,
        },
        "phosphorus_source_usage": {cat: {"n": n, "impure_rate": (p_impure[cat] / n if n else None)}
                                    for cat, n in p_counts.items()},
        "astral_overlap": {"n_overlap": len(overlap), "targets": sorted(overlap),
                           "detail": overlap_detail},
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
