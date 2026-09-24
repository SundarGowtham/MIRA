#!/usr/bin/env python
"""
research/distributional/arrows_ba_source_preference.py — Phase 15 Task 4b,
Ba-source controlled test: compute the verifier's ordering of BaO / BaO2 /
BaCO3 in the 10 matched co-precursor groups (results/external/
arrows_inventory.json), BEFORE joining any outcome (target wt%) data.
This script's output is locked into docs/phases/PHASE15_ARROWS_PREREG.md
verbatim; Step 3 later joins outcomes against these ALREADY-COMMITTED
preferences, never the reverse.

For each group, for each temperature common to every available Ba-source
option in that group: build a route per option (same reconstruction as
Step 1 -- one heating op, air atmosphere, ARROWS's own declared
"Precursor stoichiometry" as amounts), score all pairwise comparisons
via (a) the validator "GDPO vote" (sign-vote sum over RUN3_CHECKS,
ungradeable-either-side=0, gate failure on one side=prefer the other)
and (b) core/comparator.py's iteration-2 margin sign. Aggregate pairwise
results into a full ordering via win-counting (round-robin) for 3-option
groups.

Usage (tmux, real PD cache):
  uv run python research/distributional/arrows_ba_source_preference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from validator import PredictedConditions, PredictedOperation, PredictedPrecursor, PredictedRoute  # noqa: E402
from core.reward import RUN3_CHECKS, load_validator  # noqa: E402
from core.comparator import ComparatorParams, load_comparator  # noqa: E402

INVENTORY_PATH = Path("results/external/arrows_inventory.json")
OUT_JSON = Path("results/external/arrows_ba_source_verifier_preference.json")
TARGET_FORMULA = "YBa2Cu3O7"
COMPARATOR_SCALES_PATH = Path("misc/comparator_scales_v2.json")


def build_route(precursors: list[str], amounts: list[float], temp_C: float) -> PredictedRoute:
    return PredictedRoute(
        target_formula=TARGET_FORMULA,
        precursors=[PredictedPrecursor(p, a) for p, a in zip(precursors, amounts)],
        operations=[PredictedOperation(
            type="HeatingOperation",
            conditions=PredictedConditions(heating_temperature=[temp_C], heating_atmosphere=["air"]))],
    )


def gdpo_vote(validator, route_a, route_b) -> tuple[int, str]:
    """Returns (winner, reason) where winner in {1 (A), -1 (B), 0 (tie)}.
    Gate failure on exactly one side -> prefer the other, unconditionally.
    Both fail -> excluded (winner=None handled by caller)."""
    try:
        _r_a, bd_a = validator.validate(route_a, TARGET_FORMULA)
    except Exception:
        bd_a = None
    try:
        _r_b, bd_b = validator.validate(route_b, TARGET_FORMULA)
    except Exception:
        bd_b = None

    from validator import SynthesisValidator
    a_gate_fail = bd_a is None
    b_gate_fail = bd_b is None
    if a_gate_fail and b_gate_fail:
        return None, "both_gate_fail_excluded"
    if a_gate_fail:
        return -1, "gate_failure_a"
    if b_gate_fail:
        return 1, "gate_failure_b"

    vote_sum = 0
    per_channel_votes = {}
    for ch in RUN3_CHECKS:
        grade_key = f"{ch}_gradeability"
        a_ungradeable = bd_a.get(grade_key) in SynthesisValidator.SENTINEL_TAGS
        b_ungradeable = bd_b.get(grade_key) in SynthesisValidator.SENTINEL_TAGS
        if a_ungradeable or b_ungradeable:
            per_channel_votes[ch] = 0
            continue
        va, vb = bd_a.get(ch), bd_b.get(ch)
        if va is None or vb is None:
            per_channel_votes[ch] = 0
            continue
        diff = va - vb
        vote = 1 if diff > 1e-9 else (-1 if diff < -1e-9 else 0)
        per_channel_votes[ch] = vote
        vote_sum += vote

    winner = 1 if vote_sum > 0 else (-1 if vote_sum < 0 else 0)
    return winner, f"channel_votes_sum={vote_sum}"


def comparator_preference(comparator, scales, route_a, route_b) -> int:
    params = ComparatorParams(c3_fraction=0.17)
    margin, _bd = comparator.compare(route_a, route_b, TARGET_FORMULA, scales, params)
    if margin > 1e-9:
        return 1
    if margin < -1e-9:
        return -1
    return 0


def main():
    inventory = json.loads(INVENTORY_PATH.read_text())
    groups = inventory["ybco_ba_source_matched_comparisons"]["detail"]

    print("loading validator and comparator...", flush=True)
    validator = load_validator(
        Path("data/cache/mp_formula_set.pkl"), Path("data/cache/pd_index.json"), Path("."))
    comparator = load_comparator(
        Path("data/cache/mp_formula_set.pkl"), Path("data/cache/pd_index.json"), Path("."))
    scales = json.loads(COMPARATOR_SCALES_PATH.read_text())["scales"]

    ybco_data = json.loads(Path("data/external/arrows/ARROWS/Examples/YBCO/Exp.json").read_text())["Universal File"]

    all_group_results = []
    for g in groups:
        co = g["co_precursors"]
        sources = g["sources"]  # {ba_type: {"precursor_set": key, "temperatures": [...]}}
        ba_free_coprecursors = not any(p.startswith("Ba") for p in co)

        # shared temperatures across ALL options in this group
        shared_temps = None
        for ba_type, info in sources.items():
            tset = set(info["temperatures"])
            shared_temps = tset if shared_temps is None else (shared_temps & tset)
        shared_temps = sorted(shared_temps, key=lambda t: float(t.split()[0]))

        cell_results = []
        for temp_str in shared_temps:
            temp_C = float(temp_str.split()[0])
            routes = {}
            for ba_type, info in sources.items():
                pset_key = info["precursor_set"]
                precursors = [p.strip() for p in pset_key.split(",")]
                amounts = ybco_data[pset_key]["Precursor stoichiometry"]
                routes[ba_type] = build_route(precursors, amounts, temp_C)

            ba_types = list(routes.keys())
            pairwise_gdpo = {}
            pairwise_comp = {}
            for i in range(len(ba_types)):
                for j in range(i + 1, len(ba_types)):
                    a, b = ba_types[i], ba_types[j]
                    winner, reason = gdpo_vote(validator, routes[a], routes[b])
                    pairwise_gdpo[f"{a}_vs_{b}"] = {
                        "winner": (a if winner == 1 else (b if winner == -1 else "tie"))
                                 if winner is not None else "excluded_both_gate_fail",
                        "reason": reason,
                    }
                    comp_winner = comparator_preference(comparator, scales, routes[a], routes[b])
                    pairwise_comp[f"{a}_vs_{b}"] = (a if comp_winner == 1 else
                                                    (b if comp_winner == -1 else "tie"))

            # round-robin win counts -> ordering
            def ordering_from_pairwise(pairwise: dict, key_fn) -> list[str]:
                wins = {bt: 0 for bt in ba_types}
                for pair_key, val in pairwise.items():
                    winner_label = key_fn(val)
                    if winner_label in wins:
                        wins[winner_label] += 1
                return sorted(ba_types, key=lambda bt: -wins[bt])

            gdpo_ordering = ordering_from_pairwise(pairwise_gdpo, lambda v: v["winner"])
            comp_ordering = ordering_from_pairwise(pairwise_comp, lambda v: v)

            cell_results.append({
                "temperature": temp_str,
                "pairwise_gdpo_vote": pairwise_gdpo,
                "pairwise_comparator": pairwise_comp,
                "gdpo_vote_ordering_best_first": gdpo_ordering,
                "comparator_ordering_best_first": comp_ordering,
            })

        all_group_results.append({
            "co_precursors": co, "ba_free_coprecursors": ba_free_coprecursors,
            "ba_sources_available": list(sources.keys()),
            "cells": cell_results,
        })

        print(f"\nco-precursors={co} (Ba-free co-precursors: {ba_free_coprecursors})")
        for cell in cell_results:
            print(f"  T={cell['temperature']}: "
                  f"GDPO-vote order={cell['gdpo_vote_ordering_best_first']}  "
                  f"comparator order={cell['comparator_ordering_best_first']}")

    OUT_JSON.write_text(json.dumps({"groups": all_group_results}, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
