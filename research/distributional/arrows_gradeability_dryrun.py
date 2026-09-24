#!/usr/bin/env python
"""
research/distributional/arrows_gradeability_dryrun.py — Phase 15 Task 4b
Step 1: gradeability dry-run, NO outcome data joined, NO scores looked at.

Renders every ARROWS³ (precursor set, temperature) entry as a route using
the same reconstruction convention as Task 2: one heating operation at
the entry's own temperature, the recorded atmosphere if present (else
"air"), amounts from ARROWS's own "Precursor stoichiometry" field
(the real declared mixing ratio, matching the precursor-set key's order
-- verified directly, not assumed). Runs validator.py's SynthesisValidator
and core/comparator.py's Comparator AS-IS (no modification). Records
ONLY gate pass/fail and gradeability flags per channel per entry.

STOP RULE (checked here, before any pre-registration is written):
  - the canonical target phase must resolve in the phase diagram for
    ALL THREE targets (YBCO, LTOPO, NTMO), or stop.
  - if any Phase-12 reward channel (RUN3_CHECKS: amount_accuracy,
    thermodynamic_favorable, stoichiometry, chempot_atmosphere,
    operation_order) is ungradeable on more than 50% of YBCO entries,
    stop.
If the stop rule triggers, this script reports that and
docs/phases/PHASE15_ARROWS_PREREG.md is NOT written.

Canonical target formulas (used for ALL entries of that target,
independent of what the ARROWS record's own recorded target phase
was for that specific reaction -- this step asks "does the verifier
consider this route toward the target ASTRAL-style canonical
compound," not the outcome-matching question from Task 4b Step 2):
  YBCO  -> YBa2Cu3O7  (reduces to Ba2YCu3O7, confirmed in the MP formula set)
  LTOPO -> LiTiOPO4   (reduces to LiTiPO5, confirmed in the MP formula set)
  NTMO  -> Na2Te3Mo3O16 (confirmed in the MP formula set)

Usage (tmux, real PD cache):
  uv run python research/distributional/arrows_gradeability_dryrun.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pymatgen.core import Composition  # noqa: E402
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError  # noqa: E402

from validator import (PredictedConditions, PredictedOperation,  # noqa: E402
                       PredictedPrecursor, PredictedRoute, SynthesisValidator,
                       VOLATILE_FORMULAS)
from core.reward import RUN3_CHECKS, load_validator  # noqa: E402
from core.comparator import CHANNEL_NAMES, load_comparator  # noqa: E402


def solve_balanced_amounts(precursors: list[str], target: str) -> list[float] | None:
    """Fallback for entries with no ARROWS-declared 'Precursor stoichiometry'
    (LTOPO has none at all, confirmed by direct inspection): solve the
    balanced reaction via pymatgen's Reaction, same candidate-volatile-set
    approach as validator.py's own _find_balanced_reaction, and return the
    solved reactant coefficients in precursor order. None if no balance
    is found with any candidate volatile set."""
    try:
        reactants = [Composition(p) for p in precursors]
        target_comp = Composition(target)
    except Exception:
        return None
    candidate_volatile_sets = [[], ["CO2"], ["H2O"], ["O2"],
                              ["CO2", "H2O", "O2"], list(VOLATILE_FORMULAS)]
    for volatile_strs in candidate_volatile_sets:
        try:
            volatile_set = [Composition(v) for v in volatile_strs]
            products = [target_comp] + volatile_set
            reaction = Reaction(reactants, products)
        except ReactionError:
            continue
        except Exception:
            continue
        try:
            coeffs = [abs(reaction.get_coeff(r)) for r in reactants]
        except (ValueError, KeyError):
            continue
        if all(c > 1e-9 for c in coeffs):
            return coeffs
    return None

ARROWS_DIR = Path("data/external/arrows/ARROWS/Examples")
OUT_JSON = Path("results/external/arrows_gradeability_dryrun.json")

CANONICAL_TARGET = {
    "YBCO": "YBa2Cu3O7",
    "LTOPO": "LiTiOPO4",
    "NTMO": "Na2Te3Mo3O16",
}


def parse_precursor_set(key: str) -> list[str]:
    return [p.strip() for p in key.split(",")]


def parse_temp_C(temp_str: str) -> float:
    return float(temp_str.split()[0])


def build_route(target: str, precursors: list[str], amounts: list[float],
                temp_C: float, atmosphere: str) -> PredictedRoute:
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(p, a) for p, a in zip(precursors, amounts)],
        operations=[PredictedOperation(
            type="HeatingOperation",
            conditions=PredictedConditions(
                heating_temperature=[temp_C], heating_atmosphere=[atmosphere]))],
    )


def load_target_entries(name: str, target_formula: str) -> tuple[list[dict], str | None, int, int]:
    with (ARROWS_DIR / name / "Exp.json").open() as f:
        d = json.load(f)["Universal File"]
    common = d.get("Common Experimental Conditions")
    atmosphere = (common or {}).get("atmosphere", "air")
    entries = []
    n_declared, n_solved_fallback, n_dropped = 0, 0, 0
    for pset_key, val in d.items():
        if pset_key == "Common Experimental Conditions":
            continue
        precursors = parse_precursor_set(pset_key)
        amounts = val.get("Precursor stoichiometry")
        source = "declared"
        if amounts is None or len(amounts) != len(precursors):
            # Fallback: ARROWS provides no declared stoichiometry for this
            # target (confirmed: LTOPO has none at all) -- solve the
            # balanced reaction directly, same candidate-volatile-set
            # approach validator.py's own balance solver uses.
            amounts = solve_balanced_amounts(precursors, target_formula)
            source = "solved_fallback"
            if amounts is None:
                n_dropped += 1
                continue
            n_solved_fallback += 1
        else:
            n_declared += 1
        for temp_str in val.get("Temperatures", {}):
            entries.append({
                "target_name": name, "precursor_set": pset_key,
                "precursors": precursors, "amounts": amounts, "amounts_source": source,
                "temp_C": parse_temp_C(temp_str), "temp_str": temp_str,
                "atmosphere": atmosphere,
            })
    return entries, atmosphere, n_declared, n_solved_fallback


def main():
    print("loading validator and comparator (real PD cache)...", flush=True)
    validator = load_validator(
        Path("data/cache/mp_formula_set.pkl"),
        Path("data/cache/pd_index.json"),
        Path("."),
    )
    comparator = load_comparator(
        Path("data/cache/mp_formula_set.pkl"),
        Path("data/cache/pd_index.json"),
        Path("."),
    )

    # --- Stop-rule check 1: target phase resolves in the PD for all three targets ---
    print("\n== Stop-rule check 1: does the canonical target resolve in the PD? ==")
    target_resolves = {}
    for name, formula in CANONICAL_TARGET.items():
        try:
            pd, chemsys = validator.thermo_checker._resolve_pd([formula])
            entry = validator.thermo_checker._best_entry_for_formula(pd, formula) if pd else None
            resolves = entry is not None
        except Exception as e:
            resolves = False
            print(f"  {name} ({formula}): EXCEPTION {e}")
        target_resolves[name] = resolves
        print(f"  {name} ({formula}): {'RESOLVES' if resolves else 'DOES NOT RESOLVE'}")

    if not all(target_resolves.values()):
        print("\n*** STOP RULE TRIGGERED: a target phase does not resolve. "
              "NOT writing the pre-registration. ***")
        OUT_JSON.write_text(json.dumps({
            "stopped": True, "reason": "target_phase_does_not_resolve",
            "target_resolves": target_resolves,
        }, indent=1))
        return

    # --- Build all entries, score gates + gradeability (no outcomes) ---
    all_results = {}
    for name in ["YBCO", "LTOPO", "NTMO"]:
        target_formula = CANONICAL_TARGET[name]
        entries, atmosphere, n_declared, n_solved = load_target_entries(name, target_formula)
        print(f"\n{name}: {len(entries)} entries, atmosphere={atmosphere!r}, "
              f"amounts: {n_declared} declared (ARROWS), {n_solved} solved-fallback")

        per_entry = []
        for e in entries:
            route = build_route(target_formula, e["precursors"], e["amounts"],
                                e["temp_C"], e["atmosphere"])
            # Validator: full validate() call, record gradeability per RUN3_CHECKS channel
            try:
                _reward, bd = validator.validate(route, target_formula)
            except Exception:
                bd = None
            val_gradeability = {}
            if bd is not None:
                for ch in RUN3_CHECKS:
                    grade_key = f"{ch}_gradeability"
                    tag = bd.get(grade_key)
                    is_ungradeable = tag in SynthesisValidator.SENTINEL_TAGS if tag is not None else False
                    val_gradeability[ch] = {"gradeability_tag": tag, "ungradeable": is_ungradeable,
                                            "value_present": bd.get(ch) is not None}
            else:
                for ch in RUN3_CHECKS:
                    val_gradeability[ch] = {"gradeability_tag": "parse_or_validate_exception",
                                            "ungradeable": True, "value_present": False}

            # Comparator: gates + per-channel gradeability
            gates = comparator._gater._check_gates(route)
            channels = comparator.score_channels(route, target_formula)
            comp_gradeability = {ch: (channels[ch] is not None) for ch in CHANNEL_NAMES}

            per_entry.append({
                "precursor_set": e["precursor_set"], "temp_C": e["temp_C"],
                "validator_gradeability": val_gradeability,
                "comparator_gates": gates,
                "comparator_channel_gradeable": comp_gradeability,
            })

        # Aggregate: per RUN3_CHECKS channel, % ungradeable
        n = len(per_entry)
        val_ungradeable_rate = {}
        for ch in RUN3_CHECKS:
            n_ungradeable = sum(1 for r in per_entry if r["validator_gradeability"][ch]["ungradeable"])
            val_ungradeable_rate[ch] = n_ungradeable / n if n else None

        gate_fail_rate = {}
        for g in ["format_ok", "balances", "precursors_exist", "charge_neutral", "temperature_physical"]:
            n_fail = sum(1 for r in per_entry if not r["comparator_gates"][g])
            gate_fail_rate[g] = n_fail / n if n else None

        comp_ungradeable_rate = {}
        for ch in CHANNEL_NAMES:
            n_ungradeable = sum(1 for r in per_entry if not r["comparator_channel_gradeable"][ch])
            comp_ungradeable_rate[ch] = n_ungradeable / n if n else None

        print(f"  validator RUN3_CHECKS ungradeable rate: {val_ungradeable_rate}")
        print(f"  comparator gate FAIL rate: {gate_fail_rate}")
        print(f"  comparator channel ungradeable rate: {comp_ungradeable_rate}")

        all_results[name] = {
            "n_entries": n, "atmosphere": atmosphere,
            "n_precursor_sets_declared_stoich": n_declared,
            "n_precursor_sets_solved_fallback": n_solved,
            "validator_ungradeable_rate": val_ungradeable_rate,
            "comparator_gate_fail_rate": gate_fail_rate,
            "comparator_channel_ungradeable_rate": comp_ungradeable_rate,
            "per_entry": per_entry,
        }

    # --- Stop-rule check 2: any RUN3_CHECKS channel ungradeable on >50% of YBCO entries ---
    print("\n== Stop-rule check 2: any Phase-12 channel ungradeable on >50% of YBCO entries? ==")
    ybco_rates = all_results["YBCO"]["validator_ungradeable_rate"]
    triggered_channels = {ch: r for ch, r in ybco_rates.items() if r is not None and r > 0.5}
    for ch, r in ybco_rates.items():
        print(f"  {ch:<24} {r:.1%}" + ("  <-- STOP RULE TRIGGERED" if ch in triggered_channels else ""))

    stop_triggered = bool(triggered_channels)
    OUT_JSON.write_text(json.dumps({
        "stopped": stop_triggered,
        "target_resolves": target_resolves,
        "ybco_channel_ungradeable_trigger": triggered_channels if stop_triggered else None,
        "results": all_results,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")

    if stop_triggered:
        print("\n*** STOP RULE TRIGGERED: a Phase-12 channel is ungradeable on >50% of "
              "YBCO entries. NOT writing the pre-registration. ***")
    else:
        print("\n*** NO STOP RULE TRIGGERED. Proceeding to Step 2 (pre-registration). ***")


if __name__ == "__main__":
    main()
