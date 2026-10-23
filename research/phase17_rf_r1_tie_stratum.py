#!/usr/bin/env python
"""
research/phase17_rf_r1_tie_stratum.py -- regular Claude's correction:
R2 is defined in the pre-reg (docs/phases/PHASE17_PREREG.md line 445)
as the 5-channel RUN3 aggregate; R1 is the 10-channel checker (run 1).
Confirmed by code-reading (not by re-deriving): the existing tie-stratum
0.5841 [0.5174, 0.6476] in results/phase17/rf_information_bound.json
(beyond_rule_tie_stratum.R2_checker) comes from `gdpo_winner`, which is
`gdpo_vote_and_channels()` in the FROZEN research/distributional/
arrows_gate_scoring.py -- that function votes over RUN3_CHECKS (5
channels) only, confirmed by direct read of its source (imports
RUN3_CHECKS from core.reward, loops `for ch in RUN3_CHECKS`). So the
0.5841 result WAS always R2 (5-channel); CC's PROSE description of it
as "the full 10-channel checker" in an earlier reply was the mislabel,
not the underlying computation. This script computes the missing R1
(10-channel) number on the EXACT SAME pairs (base_carbfree==0 tie
stratum, n=1,112, YBCO primary |delta|>=5wt%) for a clean side-by-side.

REUSE: ybco_pairs_with_ranker (research/phase17_rf_information_bound.py,
already gives the identical n=1,112 tie-stratum subset used by R2/R4/R5
above) for set_a/set_b/temp/base_carbfree/actual; rebuilds the same
(precursor_set, temp) -> route dict via load_target_entries/build_route
(frozen arrows_gate_scoring.py) to recompute each route's FULL 10-channel
validator breakdown (score_validator -- this already returns ALL 10
checks, R2's gdpo_vote_and_channels just never looked at the other 5);
mean_agreement_f27_style / bootstrap_agreement_f27 (phase17_rf_
stratum_reconciliation.py, same convention: ties credited 0.5,
denominator = all pairs with a defined vote AND actual).

Patches results/phase17/rf_information_bound.json additively:
  beyond_rule_tie_stratum.R1_checker_10ch  (the missing row)
  beyond_rule_tie_stratum_label_note       (disambiguates R1 vs R2 explicitly)

NO EDIT to core/ranker.py, core/comparator.py, validator.py, or
arrows_gate_scoring.py.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_rf_r1_tie_stratum.py
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import load_comparator  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency  # noqa: E402
from core.reward import load_validator  # noqa: E402
from research.ranker_capacity_rssft import SYNTHESIS  # noqa: E402
from research.distributional.arrows_gate_scoring import (  # noqa: E402
    COMPARATOR_SCALES_PATH, load_target_entries, build_route, score_validator,
)
from research.phase17_rf_information_bound import ybco_pairs_with_ranker  # noqa: E402
from research.phase17_rf_stratum_reconciliation import bootstrap_agreement_f27  # noqa: E402
from validator import SynthesisValidator, ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rf_information_bound.json"
SEED = 20261016

ALL_10_CHECKS = (
    "stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist",
    "operation_order", "temperature_plausible", "thermodynamic_favorable",
    "target_stability", "chempot_atmosphere", "target_match",
)


def vote_generic(bd_a, bd_b, checks) -> int | None:
    """Mirrors gdpo_vote_and_channels's logic exactly, generalised to any
    channel list (here: ALL_10_CHECKS instead of RUN3_CHECKS)."""
    if bd_a is None and bd_b is None:
        return None
    if bd_a is None:
        return -1
    if bd_b is None:
        return 1
    vote_sum = 0
    for ch in checks:
        grade_key = f"{ch}_gradeability"
        a_ungrad = bd_a.get(grade_key) in SynthesisValidator.SENTINEL_TAGS
        b_ungrad = bd_b.get(grade_key) in SynthesisValidator.SENTINEL_TAGS
        va, vb = bd_a.get(ch), bd_b.get(ch)
        if a_ungrad or b_ungrad or va is None or vb is None:
            continue
        diff = va - vb
        vote_sum += 1 if diff > 1e-9 else (-1 if diff < -1e-9 else 0)
    return 1 if vote_sum > 0 else (-1 if vote_sum < 0 else 0)


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
    rng = random.Random(SEED)

    print("rebuilding YBCO pairs (same pipeline as R2/R4/R5)...", flush=True)
    pairs = ybco_pairs_with_ranker(validator, comparator, scales, ranker, rng)
    print(f"n_pairs (primary threshold) = {len(pairs)}", flush=True)

    entries, target_formula = load_target_entries("YBCO")
    bd_by_key = {}
    for e in entries:
        key = (e["precursor_set"], e["temp_str"])
        route = build_route(target_formula, e["precursors"], e["amounts"], e["temp_C"], e["atmosphere"])
        bd_by_key[key] = score_validator(validator, route, target_formula)

    for p in pairs:
        bd_a = bd_by_key.get((p["set_a"], p["temp"]))
        bd_b = bd_by_key.get((p["set_b"], p["temp"]))
        p["r1_winner_10ch"] = vote_generic(bd_a, bd_b, ALL_10_CHECKS)

    tie_pairs = [p for p in pairs if p.get("base_carbfree") == 0]
    print(f"tie stratum n={len(tie_pairs)}", flush=True)

    r1_result = bootstrap_agreement_f27(tie_pairs, "r1_winner_10ch", SEED)
    print(f"R1_checker_10ch (f27-style, tie stratum): {r1_result}", flush=True)

    rule_decided_pairs = [p for p in pairs if p.get("base_carbfree") in (-1, 1)]
    r1_rule_decided = bootstrap_agreement_f27(rule_decided_pairs, "r1_winner_10ch", SEED)
    print(f"R1_checker_10ch (f27-style, pooled rule-decided): {r1_rule_decided}", flush=True)

    doc = json.loads(OUT_PATH.read_text())
    ybco = doc.setdefault("ybco_primary", {})
    beyond = ybco.setdefault("beyond_rule_tie_stratum", {})
    beyond["R1_checker_10ch"] = r1_result
    symm = doc.setdefault("symmetrized_rule_decided", {})
    symm["R1_checker_10ch"] = r1_rule_decided
    doc["beyond_rule_tie_stratum_label_note"] = (
        "R2_checker in beyond_rule_tie_stratum/symmetrized_rule_decided is the 5-channel RUN3 "
        "aggregate (gdpo_vote_and_channels, votes over RUN3_CHECKS only -- confirmed by direct "
        "source read, matching the pre-reg's own R2 definition, docs/phases/PHASE17_PREREG.md "
        "line 445). R1_checker_10ch (added here) is the original 10-channel checker aggregate "
        "(run 1), computed on the IDENTICAL n=1,112 tie-stratum pairs for a clean side-by-side. "
        "An earlier prose report described the 0.5841 result as 'the full 10-channel checker' -- "
        "that was a prose mislabel of a correctly-computed R2 number, not a wrong computation; "
        "no number in the file needed to change, only this label."
    )
    OUT_PATH.write_text(json.dumps(doc, indent=1, default=str))
    print(f"-> {OUT_PATH} (beyond_rule_tie_stratum.R1_checker_10ch, "
          f"symmetrized_rule_decided.R1_checker_10ch, beyond_rule_tie_stratum_label_note added)")


if __name__ == "__main__":
    main()
