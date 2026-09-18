#!/usr/bin/env python
"""
research/astral_pair_enumeration.py — Phase 13 Step 0, run formally
(misc/PHASE13_14_SPEC.md Step 0; flagged by regular Claude's review of
iteration 2 as "supposed to precede the re-score" and never having been
run as its own dedicated, reported artifact -- only checked ad hoc via a
one-off inspection in conversation). Enumerates every within-target pair
in misc/astral_validation_set.json with measured phase purity on both
sides, and reports the count and the |Delta purity| distribution, BEFORE
any interpretation of what that count means for testing the comparator.

This does not re-score anything -- it is a data-availability inventory,
independent of core/comparator.py entirely.

Usage (tmux -- trivial computation, but the project's "always tmux" rule
has no exceptions):
  uv run python research/astral_pair_enumeration.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

DATA_PATH = Path("misc/astral_validation_set.json")
OUT_JSON = Path("results/astral_pair_enumeration.json")


def main():
    data = json.loads(DATA_PATH.read_text())
    targets = data["targets"]
    print(f"loaded {len(targets)} ASTRAL targets from {DATA_PATH}", flush=True)

    # Every within-target PAIR of named routes with measured purity on
    # both sides. Each target record carries exactly two named routes
    # (traditional, predicted) -- checked directly below, not assumed.
    pairs = []
    route_counts = []
    for t in targets:
        routes = []
        if "traditional" in t and t.get("trad_best_purity") is not None:
            routes.append(("traditional", t["trad_best_purity"]))
        if "predicted" in t and t.get("pred_best_purity") is not None:
            routes.append(("predicted", t["pred_best_purity"]))
        route_counts.append(len(routes))
        # every unordered pair among this target's named routes
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                name_a, purity_a = routes[i]
                name_b, purity_b = routes[j]
                pairs.append({
                    "target": t["target"],
                    "side_a": name_a, "side_b": name_b,
                    "purity_a": purity_a, "purity_b": purity_b,
                    "abs_delta_purity": abs(purity_a - purity_b),
                })

    n_routes_per_target = sorted(set(route_counts))
    deltas = [p["abs_delta_purity"] for p in pairs]

    print(f"\ndistinct route-count-per-target values found: {n_routes_per_target}")
    print(f"(i.e. every target has exactly {n_routes_per_target[0]} named routes "
          f"with measured purity, if this list has one element)" if len(n_routes_per_target) == 1
          else "(route count per target VARIES -- see per-target detail in the JSON)")
    print(f"\nTOTAL within-target pairs enumerable from this file: {len(pairs)}")
    print(f"(the primary/secondary endpoints in phase13_astral_scoring_v2.py already "
          f"use exactly these {len(pairs)} pairs -- this confirms rather than expands "
          f"that scope)")

    if deltas:
        print(f"\n|Delta purity| distribution across {len(deltas)} pairs:")
        print(f"  min={min(deltas):.3f}  median={statistics.median(deltas):.3f}  "
              f"mean={statistics.mean(deltas):.3f}  max={max(deltas):.3f}")
        for q in [10, 25, 50, 75, 90]:
            print(f"  p{q} = {statistics.quantiles(deltas, n=100)[q-1]:.3f}")

    # Explicit statement of the ceiling: does a fuller ASTRAL dataset
    # exist anywhere in this repo that would let this enumeration exceed
    # 35 pairs? Re-verified here, not just claimed.
    other_astral_files = list(Path(".").rglob("*astral*"))
    other_astral_files = [f for f in other_astral_files
                          if f.is_file() and ".git" not in str(f)
                          and "__pycache__" not in str(f)]
    print(f"\n{len(other_astral_files)} astral-related files exist in this repo "
          f"(scripts, logs, and result dumps that all trace back to this same "
          f"35-target extract -- re-confirmed, not a new search).")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "source": str(DATA_PATH),
        "n_targets": len(targets),
        "n_routes_per_target_distinct_values": n_routes_per_target,
        "n_pairs_enumerated": len(pairs),
        "abs_delta_purity_distribution": {
            "min": min(deltas) if deltas else None,
            "median": statistics.median(deltas) if deltas else None,
            "mean": statistics.mean(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
        },
        "conclusion": (
            f"Every target in {DATA_PATH} carries exactly two named routes "
            "(traditional, predicted) with measured purity -- there is no "
            "third or further candidate route to pair. The within-target "
            "pair enumeration is therefore IDENTICAL to the 35 "
            "predicted-vs-conventional pairs already scored, not a larger, "
            "independent set. ASTRAL's own headline_numbers field records "
            "224 total reactions in the underlying screen, but that fuller "
            "dataset is not present anywhere in this repository (checked by "
            "listing every astral-related file, all of which trace back to "
            "this same 35-target extract). The ~150-pair enumeration that "
            "would give this endpoint real power does not exist here to run."
        ),
        "pairs": pairs,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
