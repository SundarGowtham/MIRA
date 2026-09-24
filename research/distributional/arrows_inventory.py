#!/usr/bin/env python
"""
research/distributional/arrows_inventory.py — Phase 15 Task 4a: ARROWS³
inventory only, no scoring. "The pair count decides the bar" -- this
script produces that count and stops.

Source: Szymanski et al., Nat. Commun. 2023 (arXiv 2304.09353).
Data: https://github.com/njszym/ARROWS, commit cb630e944315c8ac19f5b9a3c9d8984be93258a6
(main branch, cloned 2026-09-24). Git LFS files fetched directly via
media.githubusercontent.com (git-lfs CLI unavailable on this machine;
GIT_LFS_SKIP_SMUDGE=1 clone + direct HTTPS fetch of the real blob content,
sha256-verified against each LFS pointer's own oid before use):
  Examples/YBCO/Exp.json  (52,463,427 bytes, sha256 61a9efa9...)
  Examples/LTOPO/Exp.json (20,322,802 bytes, sha256 2fd8f33c...)
  Examples/NTMO/Exp.json  (2,118,607 bytes, sha256 3fa43477...)

Usage (tmux):
  uv run python research/distributional/arrows_inventory.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ARROWS_DIR = Path("data/external/arrows/ARROWS/Examples")
OUT_JSON = Path("results/external/arrows_inventory.json")

BA_CARBONATE = {"BaCO3"}
BA_BARE_OR_PEROXIDE = {"BaO", "BaO2"}


def load_target(name: str) -> dict:
    with (ARROWS_DIR / name / "Exp.json").open() as f:
        return json.load(f)["Universal File"]


def parse_precursor_set(key: str) -> list[str]:
    return [p.strip() for p in key.split(",")]


def main():
    results = {}
    for name in ["YBCO", "LTOPO", "NTMO"]:
        d = load_target(name)
        precursor_sets = {k: v for k, v in d.items() if k != "Common Experimental Conditions"}
        common_conditions = d.get("Common Experimental Conditions")

        n_sets = len(precursor_sets)
        all_temps = set()
        outcome_field_counts = Counter()
        n_entries_with_products = 0
        n_entries_with_xrd_only = 0
        n_entries_total = 0
        for pset, val in precursor_sets.items():
            temps = val.get("Temperatures", {})
            for temp, entry in temps.items():
                all_temps.add(temp)
                n_entries_total += 1
                if "products" in entry and "product weight fractions" in entry:
                    n_entries_with_products += 1
                    outcome_field_counts["products+weight_fractions"] += 1
                elif "XRD" in entry:
                    n_entries_with_xrd_only += 1
                    outcome_field_counts["XRD_only"] += 1
                else:
                    outcome_field_counts["other:" + ",".join(sorted(entry.keys()))] += 1

        print(f"\n{'='*90}\n{name}\n{'='*90}")
        print(f"n distinct precursor sets: {n_sets}")
        print(f"distinct temperatures observed: {sorted(all_temps, key=lambda t: float(t.split()[0]))}")
        print(f"n (precursor_set, temperature) entries total: {n_entries_total}")
        print(f"outcome field breakdown: {dict(outcome_field_counts)}")
        print(f"has 'Common Experimental Conditions': {common_conditions is not None} "
              f"({common_conditions})")

        # Step 3: within-target, same-temperature pairs of distinct
        # precursor sets where the outcome differs. Only meaningful where
        # an outcome (products+weight_fractions) exists -- for XRD-only
        # entries there is no pre-computed outcome to compare without
        # additional Rietveld-style processing, out of scope for
        # inventory-only Task 4a.
        by_temp: dict[str, dict[str, dict]] = defaultdict(dict)
        for pset, val in precursor_sets.items():
            for temp, entry in val.get("Temperatures", {}).items():
                if "products" in entry and "product weight fractions" in entry:
                    by_temp[temp][pset] = entry

        pair_count_per_temp = {}
        total_pairs = 0
        total_pairs_differing = 0
        total_pairs_diff_phaseset = 0
        tv_distances = []  # total-variation distance, when phase sets match exactly
        for temp, psets in by_temp.items():
            keys = list(psets.keys())
            n_pairs_this_temp = 0
            n_differing_this_temp = 0
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    n_pairs_this_temp += 1
                    prod_a = psets[a]["products"]
                    prod_b = psets[b]["products"]
                    wf_a = psets[a]["product weight fractions"]
                    wf_b = psets[b]["product weight fractions"]
                    if set(prod_a) == set(prod_b):
                        # same phase set -> total-variation distance between
                        # the two weight-fraction distributions (aligned by
                        # phase name), in [0, 100] (wt% scale)
                        map_a = dict(zip(prod_a, wf_a))
                        map_b = dict(zip(prod_b, wf_b))
                        tv = sum(abs(map_a[p] - map_b[p]) for p in map_a) / 2.0
                        differs = tv > 1e-9
                        if differs:
                            tv_distances.append(tv)
                    else:
                        differs = True
                        total_pairs_diff_phaseset += 1
                    if differs:
                        n_differing_this_temp += 1
            pair_count_per_temp[temp] = {"n_pairs": n_pairs_this_temp, "n_differing": n_differing_this_temp}
            total_pairs += n_pairs_this_temp
            total_pairs_differing += n_differing_this_temp

        print(f"\nwithin-target, same-temperature pairs (products+weight_fractions entries only):")
        print(f"  TOTAL pairs: {total_pairs}, differing outcome: {total_pairs_differing} "
              f"(different phase SET: {total_pairs_diff_phaseset}, "
              f"same phase set but different weight fractions: {len(tv_distances)})")
        for temp, c in sorted(pair_count_per_temp.items(), key=lambda kv: float(kv[0].split()[0])):
            print(f"  {temp:<8} n_pairs={c['n_pairs']:4d}  differing={c['n_differing']:4d}")
        if tv_distances:
            tv_sorted = sorted(tv_distances)
            n = len(tv_sorted)
            print(f"  total-variation distance distribution (wt%, same-phase-set pairs only, n={n}):")
            print(f"    min={tv_sorted[0]:.1f}  p25={tv_sorted[n//4]:.1f}  "
                  f"median={tv_sorted[n//2]:.1f}  p75={tv_sorted[3*n//4]:.1f}  max={tv_sorted[-1]:.1f}")

        results[name] = {
            "n_precursor_sets": n_sets,
            "temperatures": sorted(all_temps, key=lambda t: float(t.split()[0])),
            "n_entries_total": n_entries_total,
            "outcome_field_counts": dict(outcome_field_counts),
            "common_experimental_conditions": common_conditions,
            "within_target_same_temp_pairs": {
                "total_pairs": total_pairs, "total_differing": total_pairs_differing,
                "total_differing_phase_set": total_pairs_diff_phaseset,
                "total_differing_same_phase_set_wt_fraction": len(tv_distances),
                "per_temperature": pair_count_per_temp,
                "tv_distance_distribution_wt_pct": (
                    {"n": len(tv_distances), "min": min(tv_distances), "max": max(tv_distances),
                     "median": sorted(tv_distances)[len(tv_distances) // 2]}
                    if tv_distances else None
                ),
            },
        }

    # --- Ba carbonate vs Ba oxide/peroxide controlled-comparison check (YBCO only) ---
    print(f"\n{'='*90}\nBa carbonate vs Ba oxide/peroxide check (YBCO)\n{'='*90}")
    ybco = load_target("YBCO")
    ybco_sets = {k: v for k, v in ybco.items() if k != "Common Experimental Conditions"}

    def ba_source(pset_parts: list[str]) -> set[str]:
        return {p for p in pset_parts if p in ("BaO", "BaO2", "BaCO3", "BaCuO2", "Ba2(CuO2)3")}

    # Group by the NON-Ba precursors (the "co-precursors"), so we can find
    # matched triples that differ ONLY in which single Ba source is used.
    by_coprecursors: dict[tuple, dict[str, str]] = defaultdict(dict)
    for pset_key in ybco_sets:
        parts = parse_precursor_set(pset_key)
        ba_parts = [p for p in parts if p in ("BaO", "BaO2", "BaCO3")]
        non_ba_parts = tuple(sorted(p for p in parts if p not in ("BaO", "BaO2", "BaCO3")))
        if len(ba_parts) == 1:  # exactly one simple Ba source (not a mixed/pre-reacted set)
            by_coprecursors[non_ba_parts][ba_parts[0]] = pset_key

    matched_triples = {co: sources for co, sources in by_coprecursors.items()
                       if len(sources) >= 2}  # at least 2 of {BaO, BaO2, BaCO3} matched
    print(f"co-precursor combinations with >=2 matched simple-Ba-source sets: {len(matched_triples)}")
    triple_detail = []
    for co, sources in matched_triples.items():
        temps_by_source = {}
        for ba_type, pset_key in sources.items():
            temps = sorted(ybco_sets[pset_key].get("Temperatures", {}).keys(),
                           key=lambda t: float(t.split()[0]))
            temps_by_source[ba_type] = {"precursor_set": pset_key, "temperatures": temps}
        print(f"  co-precursors={co}: {list(sources.keys())}")
        for ba_type, info in temps_by_source.items():
            print(f"    {ba_type:<8} -> '{info['precursor_set']}'  T={info['temperatures']}")
        triple_detail.append({"co_precursors": list(co), "sources": temps_by_source})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "arrows_commit": "cb630e944315c8ac19f5b9a3c9d8984be93258a6",
        "per_target": results,
        "ybco_ba_source_matched_comparisons": {
            "n_matched_coprecursor_groups": len(matched_triples),
            "detail": triple_detail,
        },
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
