#!/usr/bin/env python
"""
research/phase17_r4_gate_channel_shapley.py — Phase 17 Block R, R-e's
"Gates" bullet for R4 (ranker v2): the same exact 2-player Shapley
attribution of a decision to {Gate, Channels} built for R5
(research/phase17_r5_comparator_capacity_rssft.py), applied here to
`core/ranker.py::Ranker.score()`'s own gate x weighted-objective
structure. R4's own capacity number (46.31%) already exists
(results/ranker_capacity.json, Phase 16) and is NOT recomputed here --
this script adds only the Shapley column R-e asks for.

core/ranker.py's score() is SCALAR per-sample (reward = 0.0 if any
gate fails, else the weighted objective sum), not pairwise like the
comparator -- so, unlike R5, no pairwise loop over groups is needed.
Per sample: G = 1.0 if every gate in GATE_NAMES passes, else 0.0.
C = the weighted-objective sum AS IF gates did not exist, computed by
calling the exact same private objective methods `score()` itself
calls (_driving_force, _temperature_economy, _volatility_risk,
_precursor_instability, _n_precursors, _slice_competing_phases,
_precursor_decomposition_match), in the same order, with the same
None-exclusion ("active" dict) and weighting -- but WITHOUT score()'s
`if not all(gates.values()): return 0.0` early exit. Verified below
(real_reward == G * C, sampled, exact) before trusting the Shapley
split. Shapley values for the 2-player game v(none)=0, v({channels})=C,
v({gate})=0, v({gate,channels})=G*C:
    phi_gate     = 0.5 * (G - 1) * C
    phi_channels = 0.5 * (G + 1) * C
summing exactly to G*C (efficiency) -- see phase17_r5_comparator_capacity_rssft.py's
module docstring for the full derivation; identical game, different reward.

NO EDIT to core/ranker.py. NO NEW GENERATION: same source/scope/
validator_version/literature table as the existing ranker_capacity.json
(research/ranker_capacity_rssft.py, reused directly by import, not
duplicated) -- RS-SFT's 2,400 resampling_rate samples, validator_version=2
(E1's own launch config).

Usage (tmux, CPU only -- thermo-backed Ranker calls, no GPU needed):
  PYTHONPATH=. uv run python research/phase17_r4_gate_channel_shapley.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.reward import ParseFailure, parse_completion  # noqa: E402
from core.ranker import GATE_NAMES, OBJECTIVE_NAMES, Ranker, build_precursor_frequency  # noqa: E402
from probe_hardening import load_literature  # noqa: E402
from research.ranker_capacity_rssft import (  # noqa: E402
    RSSFT_PATHS, TRIAGE, SYNTHESIS, load_resampling_records, chunk,
)
from validator import ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "r4_gate_channel_shapley.json"
GROUP_SIZE = 8


def objectives_nogate(ranker: Ranker, route, target_formula: str,
                      lit_T, lit_n_ops) -> dict:
    """Mirrors Ranker.score()'s objective-computation block EXACTLY
    (same private-method calls, same order, same arguments) but without
    the gate early-exit -- so a gate-failing route still gets its
    objectives computed, for the Shapley decomposition's C term."""
    if route is None:
        return {o: None for o in OBJECTIVE_NAMES}
    dG, _dG_grade = ranker._driving_force(route, target_formula)
    obj: dict = {}
    obj["temperature_economy"] = ranker._temperature_economy(route, lit_T, dG)
    obj["volatility_risk"] = ranker._volatility_risk(route, target_formula)
    obj["driving_force_margin"] = (
        None if dG is None else max(0.0, min(1.0, -dG / ranker.scales.dg_scale)))
    obj["precursor_instability"], _ = ranker._precursor_instability(route, target_formula)
    obj["n_precursors"], _ = ranker._n_precursors(route)
    obj["slice_competing_phases"], _ = ranker._slice_competing_phases(route, target_formula)
    obj["precursor_decomposition_match"], _ = ranker._precursor_decomposition_match(route)
    return obj


def weighted_sum(ranker: Ranker, obj: dict) -> tuple[float, int]:
    active = {k: v for k, v in obj.items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    wsum = sum(ranker.weights[k] for k in active)
    if wsum == 0:
        return 0.0, 0
    return sum((ranker.weights[k] / wsum) * v for k, v in active.items()), len(active)


def main():
    print("loading literature + validator formula set + PD cache + precursor frequency...", flush=True)
    lit = load_literature(TRIAGE, SYNTHESIS)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)  # matches E1's launch, same as ranker_capacity.json

    n_pairs = 0  # here "pairs" = samples (scalar reward, not pairwise)
    phi_gate_sum = 0.0
    phi_channels_sum = 0.0
    n_gate_fail = 0
    n_gate_pass = 0
    abs_C_when_gate_fail = []
    sanity = []

    for label, path in RSSFT_PATHS.items():
        print(f"\nscoring {path} (resampling_rate only, gates bypassed for raw capture)...", flush=True)
        by_target = load_resampling_records(path)
        lit_this = {t: lit.get(t, {}) for t in by_target}
        n_scored, n_parse_fail = 0, 0
        for target, recs in by_target.items():
            lit_rec = lit_this[target]
            groups = chunk(recs, GROUP_SIZE)
            for gi, group in enumerate(groups):
                for rec in group:
                    try:
                        route = parse_completion(rec["completion"], target)
                    except Exception:
                        route = None
                        n_parse_fail += 1
                    n_scored += 1
                    gate_pass = route is not None and all(ranker._check_gates(route).values())
                    obj = objectives_nogate(ranker, route, target, lit_rec.get("max_T"), lit_rec.get("n_ops"))
                    Cval, n_active = weighted_sum(ranker, obj)
                    G = 1.0 if gate_pass else 0.0

                    if n_active == 0 and not gate_pass:
                        # Nothing to attribute either way (route failed to parse / had no
                        # computable objective at all) -- skip, matches score()'s own
                        # reward=0.0-with-no-information case; not a gate decision.
                        continue

                    phi_gate = 0.5 * (G - 1.0) * Cval
                    phi_channels = 0.5 * (G + 1.0) * Cval
                    phi_gate_sum += phi_gate
                    phi_channels_sum += phi_channels
                    n_pairs += 1
                    if gate_pass:
                        n_gate_pass += 1
                    else:
                        n_gate_fail += 1
                        abs_C_when_gate_fail.append(abs(Cval))

                    if route is not None and n_pairs <= 20:
                        actual_reward, _info = ranker.score(route, target, lit_T=lit_rec.get("max_T"),
                                                             lit_n_ops=lit_rec.get("n_ops"))
                        sanity.append({"phi_sum": round(phi_gate + phi_channels, 6),
                                      "G_times_C": round(G * Cval, 6),
                                      "actual_reward": actual_reward})
        print(f"  {label}: n_scored={n_scored} n_parse_fail={n_parse_fail}", flush=True)

    max_sanity_diff_shapley = max((abs(s["phi_sum"] - s["G_times_C"]) for s in sanity), default=0.0)
    max_sanity_diff_real = max((abs(s["G_times_C"] - s["actual_reward"]) for s in sanity), default=0.0)

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-e Gates: exact Shapley attribution of each decision to gate vs channels (R4)",
        "reward_id": "R4",
        "reward": "Ranker v2, core/ranker.py",
        "policy": "RS-SFT",
        "source": "data/novelty/rssft_reference_{heldout,astral}.jsonl, sample_group=resampling_rate "
                   "(NO NEW GENERATION), same scope as results/ranker_capacity.json",
        "validator_version": 2,
        "gate_names": GATE_NAMES,
        "objective_names": OBJECTIVE_NAMES,
        "n_samples": n_pairs,
        "n_gate_pass": n_gate_pass,
        "n_gate_fail": n_gate_fail,
        "pct_gate_fail": round(100 * n_gate_fail / n_pairs, 2) if n_pairs else None,
        "mean_phi_gate": round(phi_gate_sum / n_pairs, 4) if n_pairs else None,
        "mean_phi_channels": round(phi_channels_sum / n_pairs, 4) if n_pairs else None,
        "mean_abs_channel_value_suppressed_by_gate_failure": (
            round(float(np.mean(abs_C_when_gate_fail)), 4) if abs_C_when_gate_fail else None),
        "efficiency_check_max_abs_diff_phi_sum_vs_GxC_first20": round(max_sanity_diff_shapley, 9),
        "correctness_check_max_abs_diff_GxC_vs_actual_score_first20": round(max_sanity_diff_real, 9),
        "comparison_reference": {
            "r5_comparator_shapley_pct_pairs_gate_blocked": 11.98,
            "note": "R4 is a per-SAMPLE scalar decomposition (one Shapley split per route); R5's is "
                    "per-PAIR (one split per within-group comparison), since the comparator itself is "
                    "pairwise. Not directly numerically comparable, but both answer the same question: "
                    "how much of this reward's output is actually the gate, not the scored channels.",
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print("\n" + json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
