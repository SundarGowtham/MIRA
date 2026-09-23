#!/usr/bin/env python
"""
research/distributional/task2c_gibbs_check.py — Phase 15 Task 2c.

Part 1: confirm exactly what thermodynamic_favorable computes (already
answered by reading validator.py/gibbs_corrector.py directly -- this
script re-derives it empirically for 5 representative pairs rather than
just asserting the code-reading answer).

Part 2: for 5 representative carbonate-vs-bare-oxide pairs (same alkali,
same co-precursor, same target), compute BOTH:
  (i)  the CURRENT PRODUCTION value: gibbs_corrector.compute_reaction_
       gibbs_per_atom -- Bartel-descriptor solids + NIST-JANAF gas ΔG(T)
       at the route's max_T. This is what thermodynamic_favorable
       actually scores in training right now.
  (ii) the NAIVE 0K value: validator.py's legacy ComputedReaction path
       (predicted_route=None), i.e. raw 0K DFT reaction energy with no
       temperature or gas-entropy correction at all.
The gap between (i) and (ii) is the size of the correction the pipeline
ALREADY applies. Since (i) already incorporates NIST-JANAF CO2 ΔfG°(T)
(gibbs_corrector.py's _NIST_DFG_KJMOL table, confirmed present for CO2),
this also directly answers "does gaseous CO2 entropy enter anywhere" --
yes, in (i), not as an additional step this script needs to invent.

3 of the 5 pairs are pulled from real GDPO training completions (the 50
mixed groups from Task 2b), not invented, per this project's
verify-by-execution discipline.

Usage (tmux, real PD cache):
  uv run python research/distributional/task2c_gibbs_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pymatgen.core import Composition  # noqa: E402
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError  # noqa: E402

from validator import (PredictedConditions, PredictedOperation,  # noqa: E402
                       PredictedPrecursor, PredictedRoute, ThermoChecker)
from gibbs_corrector import compute_reaction_gibbs_per_atom  # noqa: E402

OUT_JSON = Path("results/distributional/task2c_gibbs_check.json")

# Pairs 1-2: hand-specified, per the task's own examples.
# Pairs 3-5: REAL pairs pulled directly from GDPO training completions in
# the 50 mixed groups (one-off scan of runs/gdpo-qlora-gdpo-phase12-rssft-beta0/
# generations.jsonl -- see run_logs/find_real_pairs.log for the full scan
# output this was picked from). A single representative max_T is used for
# both sides of each pair so the comparison isolates carbonate-vs-oxide,
# not also T; the real completions used a range of T for each (documented
# per pair below), so this max_T is a reasonable representative choice,
# not the exact T either specific completion reported.
PAIRS = [
    {"target": "LiBO2", "carb": ["Li2CO3", "B2O3"], "bare": ["Li2O", "B2O3"], "max_T": 800.0,
     "source": "hand-specified per task instructions"},
    {"target": "KNbO3", "carb": ["K2CO3", "Nb2O5"], "bare": ["K2O", "Nb2O5"], "max_T": 800.0,
     "source": "hand-specified per task instructions"},
    {"target": "Li2Mn2O4", "carb": ["Li2CO3", "Mn2O3"], "bare": ["Li2O", "Mn2O3"], "max_T": 800.0,
     "source": "real GDPO completions, step scan: bare@800C, carb@750C observed"},
    {"target": "Li1.1V3O8", "carb": ["Li2CO3", "V2O5"], "bare": ["Li2O", "V2O5"], "max_T": 200.0,
     "source": "real GDPO completions, both observed at 200C -- fractional/doped target"},
    {"target": "Li2Si2O5", "carb": ["Li2CO3", "SiO2"], "bare": ["Li2O", "SiO2"], "max_T": 600.0,
     "source": "real GDPO completions, bare@600C observed, carb@500-900C range observed"},
]


def build_route(target, precursors, T):
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(p, 1.0) for p in precursors],
        operations=[PredictedOperation(
            type="HeatingOperation",
            conditions=PredictedConditions(heating_temperature=[float(T)], heating_atmosphere=["air"]))],
    )


def naive_0K(thermo: ThermoChecker, precursors, target):
    """The legacy path: predicted_route=None -> raw 0K DFT ComputedReaction,
    no Gibbs wrapping, no gas-entropy correction at all."""
    precursor_pairs = [(p, 1.0) for p in precursors]
    val, gradeability = thermo.reaction_energy_per_atom(precursor_pairs, target, predicted_route=None)
    return val, gradeability


def production_gibbs(thermo: ThermoChecker, precursors, target, route):
    core_formulas = [target] + precursors
    pd, _ = thermo._resolve_pd(core_formulas)
    if pd is None:
        return None, None
    delta_G, T_K = compute_reaction_gibbs_per_atom(target, precursors, pd, route)
    return delta_G, T_K


def main():
    print("loading thermo checker (real PD cache)...", flush=True)
    thermo = ThermoChecker.from_sharded_cache(Path("data/cache/pd_index.json"), Path("."))

    results = []
    for pair in PAIRS:
        target, carb, bare, T = pair["target"], pair["carb"], pair["bare"], pair["max_T"]
        carb_route = build_route(target, carb, T)
        bare_route = build_route(target, bare, T)

        carb_naive, carb_naive_grade = naive_0K(thermo, carb, target)
        bare_naive, bare_naive_grade = naive_0K(thermo, bare, target)
        carb_gibbs, carb_T_K = production_gibbs(thermo, carb, target, carb_route)
        bare_gibbs, bare_T_K = production_gibbs(thermo, bare, target, bare_route)

        naive_gap = (bare_naive - carb_naive) if (bare_naive is not None and carb_naive is not None) else None
        gibbs_gap = (bare_gibbs - carb_gibbs) if (bare_gibbs is not None and carb_gibbs is not None) else None

        row = {
            "target": target, "carbonate_precursors": carb, "bare_oxide_precursors": bare,
            "max_T_C": T,
            "naive_0K": {"carbonate": carb_naive, "carbonate_gradeability": carb_naive_grade,
                        "bare_oxide": bare_naive, "bare_oxide_gradeability": bare_naive_grade,
                        "gap_bare_minus_carb": naive_gap},
            "production_gibbs_corrected": {"carbonate": carb_gibbs, "T_K": carb_T_K,
                                           "bare_oxide": bare_gibbs, "gap_bare_minus_carb": gibbs_gap},
        }
        results.append(row)
        print(f"\n{target}: carbonate={carb}  bare-oxide={bare}  max_T={T}C")
        print(f"  naive 0K:          carb={carb_naive}  ({carb_naive_grade})   "
              f"bare={bare_naive}  ({bare_naive_grade})   gap(bare-carb)={naive_gap}")
        print(f"  production Gibbs:  carb={carb_gibbs}  bare={bare_gibbs}  "
              f"gap(bare-carb)={gibbs_gap}")
        if naive_gap is not None and gibbs_gap is not None:
            if abs(gibbs_gap) < abs(naive_gap) * 0.5:
                verdict = "SHRINKS substantially under Gibbs correction"
            elif (naive_gap > 0) != (gibbs_gap > 0):
                verdict = "REVERSES sign under Gibbs correction"
            else:
                verdict = "SURVIVES (similar magnitude/sign)"
            print(f"  verdict: {verdict}")
            row["verdict"] = verdict

    OUT_JSON.write_text(json.dumps(results, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
