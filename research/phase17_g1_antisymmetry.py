#!/usr/bin/env python
"""
research/phase17_g1_antisymmetry.py — Phase 17 Block G1, first step per
regular Claude's instruction: "measure antisymmetry, energy of
c(a,b)+c(b,a) over all pairs. Nonzero means an R-a violation for R5:
report it, then use the antisymmetric part as the edge flow. Reuse the
margins R5 already computed where possible. Run the checker/ranker
zero-curl control before the comparator."

Reuses the SAME ARROWS3 pools (YBCO/LTOPO/NTMO) R-f and R-g already
built via build_pool_indexed (research/phase17_rg_pool_bootstrap.py,
itself built on the frozen research/distributional/arrows_gate_scoring.py)
-- no new route-reconstruction infrastructure needed; this is explicitly
reusing R5's own margins per the instruction, not the larger universe.
jsonl/top-30-recipe construction.

CONTROL (checker/ranker, run first): c(a,b) = sign(scalar(a) - scalar(b))
is antisymmetric BY CONSTRUCTION for any scalar-valued reward -- this is
a sanity check on the test methodology itself, expected to show EXACTLY
zero violations.

COMPARATOR TEST: for every ordered pair (i,j), i != j, call
comparator_rule(route_i, route_j, ...) -> w_ij AND comparator_rule
(route_j, route_i, ...) -> w_ji (the reverse call, not inferred). Checks
w_ij == -w_ji. core/comparator.py's own docstring (compare(), not edited
here, only called) claims this is "Exactly antisymmetric ... enforced by
tests/test_comparator.py" for the margin computation itself; this script
empirically verifies the CLAIM on real ARROWS3 pairs including the
gate-failure branch comparator_rule adds on top of compare(), which is
not covered by tests/test_comparator.py (that file tests compare()
directly, not the gate-wrapped comparator_rule()).

NO EDIT to core/comparator.py, validator.py, or arrows_gate_scoring.py.
NO NEW GENERATION.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_g1_antisymmetry.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import load_comparator  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency  # noqa: E402
from core.reward import load_validator  # noqa: E402
from research.ranker_capacity_rssft import SYNTHESIS  # noqa: E402
from research.distributional.arrows_gate_scoring import comparator_rule, COMPARATOR_SCALES_PATH  # noqa: E402
from research.phase17_rg_pool_bootstrap import build_pool_indexed  # noqa: E402
from validator import ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "g1_antisymmetry.json"
TARGETS = ["YBCO", "LTOPO", "NTMO"]


def scalar_control_check(pool: list[dict], key: str) -> dict:
    """c(a,b) = sign(scalar_a - scalar_b); antisymmetric by construction.
    Sanity check, not a real test of anything -- confirms methodology."""
    n_pairs, n_violations, energy_terms = 0, 0, []
    for i in range(len(pool)):
        for j in range(len(pool)):
            if i == j:
                continue
            va, vb = pool[i][key], pool[j][key]
            if va is None or vb is None:
                continue
            diff = va - vb
            w_ij = 1 if diff > 1e-9 else (-1 if diff < -1e-9 else 0)
            diff_rev = vb - va
            w_ji = 1 if diff_rev > 1e-9 else (-1 if diff_rev < -1e-9 else 0)
            n_pairs += 1
            if w_ij != -w_ji:
                n_violations += 1
            energy_terms.append((w_ij + w_ji) ** 2)
    return {"n_ordered_pairs": n_pairs, "n_violations": n_violations,
            "mean_energy_c_ab_plus_c_ba_squared": round(sum(energy_terms) / len(energy_terms), 6) if energy_terms else None}


def comparator_antisymmetry_check(pool: list[dict], comparator, scales, target_formula: str) -> dict:
    n_pairs, n_violations, energy_terms, violations = 0, 0, [], []
    both_none = 0
    for i in range(len(pool)):
        for j in range(len(pool)):
            if i == j:
                continue
            w_ij, reason_ij = comparator_rule(comparator, scales, pool[i]["route"], pool[j]["route"], target_formula)
            w_ji, reason_ji = comparator_rule(comparator, scales, pool[j]["route"], pool[i]["route"], target_formula)
            if w_ij is None and w_ji is None:
                both_none += 1
                continue
            if w_ij is None or w_ji is None:
                n_pairs += 1
                n_violations += 1
                if len(violations) < 10:
                    violations.append({"i": i, "j": j, "w_ij": w_ij, "w_ji": w_ji,
                                       "reason_ij": reason_ij, "reason_ji": reason_ji,
                                       "kind": "one_side_gate_fail_only_one_direction"})
                continue
            n_pairs += 1
            energy_terms.append((w_ij + w_ji) ** 2)
            if w_ij != -w_ji:
                n_violations += 1
                if len(violations) < 10:
                    violations.append({"i": i, "j": j, "w_ij": w_ij, "w_ji": w_ji,
                                       "reason_ij": reason_ij, "reason_ji": reason_ji,
                                       "kind": "margin_or_gate_asymmetry"})
    return {"n_ordered_pairs_both_defined": n_pairs, "n_both_gate_fail_excluded": both_none,
            "n_violations": n_violations,
            "mean_energy_c_ab_plus_c_ba_squared": round(sum(energy_terms) / len(energy_terms), 6) if energy_terms else None,
            "violation_examples": violations}


def main():
    print("loading validator, comparator, ranker...", flush=True)
    validator = load_validator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                               REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    comparator = load_comparator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                                 REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    scales = json.loads(COMPARATOR_SCALES_PATH.read_text())["scales"]
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "G", "subproperty": "G1 step 1: antisymmetry",
        "statement": "c(a,b) + c(b,a) should be exactly 0 for every ordered pair (R-a's symmetry "
                     "axiom, applied to the pairwise comparator). Checker/ranker run first as a "
                     "trivial control (any scalar-derived pairwise rule is antisymmetric by "
                     "construction). R5 (comparator) is the real test -- its compare() claims exact "
                     "antisymmetry by its own docstring/tests/test_comparator.py, but that test "
                     "covers compare() directly, not the gate-wrapped comparator_rule() used "
                     "throughout R5/R-f/R-g/Block G, which adds its own branching on top.",
        "per_target": {},
    }
    any_violation = False
    for name in TARGETS:
        print(f"\n=== {name} ===", flush=True)
        pool, target_formula = build_pool_indexed(name, validator, ranker)
        print(f"  pool size = {len(pool)}", flush=True)

        r2_control = scalar_control_check(pool, "r2_scalar")
        r4_control = scalar_control_check(pool, "r4_scalar")
        print(f"  R2 control: {r2_control}", flush=True)
        print(f"  R4 control: {r4_control}", flush=True)

        comp_check = comparator_antisymmetry_check(pool, comparator, scales, target_formula)
        print(f"  R5 comparator: n_violations={comp_check['n_violations']}/{comp_check['n_ordered_pairs_both_defined']}, "
              f"mean_energy={comp_check['mean_energy_c_ab_plus_c_ba_squared']}", flush=True)
        if comp_check["n_violations"] > 0:
            any_violation = True

        report["per_target"][name] = {
            "pool_size": len(pool),
            "R2_checker_control": r2_control,
            "R4_ranker_control": r4_control,
            "R5_comparator": comp_check,
        }

    report["verdict"] = {
        "checker_ranker_controls_zero_violations": all(
            report["per_target"][t]["R2_checker_control"]["n_violations"] == 0 and
            report["per_target"][t]["R4_ranker_control"]["n_violations"] == 0
            for t in TARGETS),
        "comparator_any_violation": any_violation,
        "interpretation": ("R5 (comparator) shows a real R-a (symmetry axiom) violation on at least "
                           "one target -- the antisymmetric part, not the raw margin, should be used "
                           "as the edge flow going into the triangle-curl (Hodge) decomposition." if any_violation else
                           "R5 (comparator) shows ZERO antisymmetry violations across all three "
                           "ARROWS3 pools -- comparator_rule (compare()'s margin plus the gate-"
                           "failure wrapper) is exactly antisymmetric empirically, matching "
                           "compare()'s own docstring claim and extending it to the gate-wrapped "
                           "rule actually used throughout R5/R-f/R-g. The edge flow equals the raw "
                           "margin unchanged -- no symmetric part to subtract. This does NOT by "
                           "itself mean the comparator is curl-free (consistent with a global "
                           "ranking): antisymmetry is necessary but not sufficient -- the real "
                           "triangle-curl projection (does c(a,b)+c(b,c)+c(c,a) sum to zero over "
                           "every triple?) is the next, separate G1 step."),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nverdict: {report['verdict']}", flush=True)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
