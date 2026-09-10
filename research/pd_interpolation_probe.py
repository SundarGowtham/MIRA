"""
pd_interpolation_probe.py
---------------------------
Tests the load-bearing question for Route 2 (multi-class reward): can the
~50% of the corpus that currently sentinels out (fractional / doped /
solid-solution targets with no discrete MP entry) be graded instead via
convex-hull interpolation?

Mechanism, from the MP phase-diagram + thermodynamic-stability docs:
e_above_hull needs a discrete entry for the exact composition, which
doped compositions never have. But PhaseDiagram.get_hull_energy(comp)
returns the equilibrium (tie-line interpolated) ground-state energy at
ANY composition the chemsys covers - it needs the surrounding stable
phases, not the target itself. So we manufacture a synthetic target
entry at its hull energy and run it through the SAME ComputedReaction
pipeline gibbs_corrector already uses. The resulting reaction energy is
"how far downhill the precursors sit relative to the thermodynamic
ground state at the target's stoichiometry" - gradeable for every
covered composition, fractional or not.

Caveat baked into the interpretation: the real single-phase solid
solution sits slightly above the 0K hull (entropy-stabilized), so the
hull energy is a lower bound on the target's true energy and this
delta_G is a mild upper bound on the true driving force.

All three reaction components are put on ONE consistent scale
(formation-Gibbs at synthesis T): precursors and the target's
decomposition phases via _wrap_solid_at_T (Bartel), gases via NIST
dfG(T). An earlier version of this probe mixed raw 0K DFT solid energies
with NIST formation-Gibbs gases, which put every gas-releasing reaction
(carbonate / hydroxide / boric-acid precursors) off by ~1 eV/atom - the
tell was that the one no-gas example graded correctly at dG~=0 while all
gas-releasing ones blew up positive.

What to look for in the output:
  - fractional targets should go from 0% gradeable to near-100% gradeable
  - the fractional delta_G distribution should overlap the integer
    real_dG distribution (spread across negative values), NOT collapse
    to a single degenerate value. Overlap => signal exists => Route 2
    viable. Degenerate pile => interpolation adds no usable signal.

Usage: same flags as the other audit scripts. --sample caps records per
tier for speed (hull interpolation over many compositions is not free).

  uv run python pd_interpolation_probe.py \
      --split data/sft/train.jsonl \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --sample 300 \
      --out interpolation_probe_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError


from core.reward import parse_completion, ParseFailure, load_validator


from gibbs_corrector import (
    extract_synthesis_temperature_K,
    _best_entry_for_formula as gibbs_best_entry,
    _wrap_solid_at_T,
    make_nist_gas_entry,
    _GAS_SPECIES,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def is_fractional(formula: str, tol: float = 1e-6) -> bool:
    try:
        amounts = Composition(formula).as_dict().values()
    except Exception:
        return False
    return any(abs(a - round(a)) > tol for a in amounts)


def interpolated_reaction_dG_per_atom(tc, precursors, target_formula, predicted_route):
    """
    ΔG_rxn per atom of target, with the target represented on the SAME
    formation-Gibbs-at-T scale as the wrapped precursors and NIST gases.

    Reference-frame consistency (the fix for the raw-solids + NIST-gas bug):
    everything is on the formation-Gibbs-at-T scale.
      - target:     synthetic entry whose per-atom energy is the finite-T
                    hull energy Sigma_i amount_i * g_form_i(T), where g_form_i
                    is the Bartel-wrapped (GibbsComputedStructureEntry) per-atom
                    formation Gibbs of each real decomposition phase i, and
                    amount_i are get_decomposition's barycentric weights
                    (which sum to 1). Mirrors pymatgen's own
                    get_decomp_and_hull_energy_per_atom exactly, but over
                    finite-T formation energies instead of raw 0K energies.
      - precursors: _wrap_solid_at_T (NOT raw - the earlier bug), same scale.
      - gases:      make_nist_gas_entry = dfG(T), already this scale.

    Returns (dG_per_atom, decomp_summary, stage_string).
    """
    core_formulas = [target_formula] + [f for f, _ in precursors]
    pd, _ = tc._resolve_pd(core_formulas)
    if pd is None:
        return None, None, "no_pd"

    target_comp = Composition(target_formula)
    T_K = extract_synthesis_temperature_K(predicted_route)

    try:
        decomp = pd.get_decomposition(target_comp)  # {entry: amount}, sum(amount) = 1
    except Exception:
        return None, None, "decomposition_failed"
    if not decomp:
        return None, None, "empty_decomposition"

    g_hull_per_atom = 0.0
    decomp_summary = {}
    for entry, amount in decomp.items():
        wrapped = _wrap_solid_at_T(entry, pd, T_K)
        if wrapped is None:  # decomposition phase missing structure -> can't Bartel-wrap
            return None, None, "decomp_wrap_failed"
        g_hull_per_atom += amount * wrapped.energy_per_atom
        decomp_summary[entry.composition.reduced_formula] = round(float(amount), 3)

    synthetic_target = ComputedEntry(target_comp, g_hull_per_atom * target_comp.num_atoms)

    precursor_entries = []
    for f, _amt in precursors:
        p_red = Composition(f).reduced_formula
        raw = gibbs_best_entry(pd, p_red)
        if raw is None:
            return None, decomp_summary, "precursor_entry_missing"
        wrapped_p = _wrap_solid_at_T(raw, pd, T_K)
        if wrapped_p is None:
            return None, decomp_summary, "precursor_wrap_failed"
        precursor_entries.append(wrapped_p)

    reactant_elements = set()
    for e in precursor_entries + [synthetic_target]:
        reactant_elements.update(str(el) for el in e.composition.elements)
    gas_entries = []
    for sp in _GAS_SPECIES:
        if set(str(el) for el in Composition(sp).elements).issubset(reactant_elements):
            gas_entries.append(make_nist_gas_entry(sp, T_K))

    try:
        reaction = ComputedReaction(precursor_entries, [synthetic_target] + gas_entries)
    except ReactionError:
        return None, decomp_summary, "reaction_error"

    try:
        coeff = reaction.get_coeff(Composition(target_comp.reduced_formula))
    except (ValueError, KeyError):
        return None, decomp_summary, "get_coeff_failed"
    if coeff <= 1e-6:
        return None, decomp_summary, "nonpositive_coefficient"

    atoms_per_target = target_comp.reduced_composition.num_atoms
    dG_per_atom = float(reaction.calculated_reaction_energy) / (coeff * atoms_per_target)
    return dG_per_atom, decomp_summary, "ok"


def run(path: Path, validator, sample: int) -> dict:
    examples = load_jsonl(path)
    tc = validator.thermo_checker

    per_tier = defaultdict(list)       # "fractional"/"integer" -> [dG]
    stages = defaultdict(Counter)      # tier -> Counter(stage)
    worked_examples = defaultdict(list)
    seen = Counter()

    for ex in examples:
        target = ex["target"]
        tier = "fractional" if is_fractional(target) else "integer"
        if seen[tier] >= sample:
            continue
        try:
            route = parse_completion(ex["completion"], target)
        except ParseFailure:
            continue
        seen[tier] += 1

        precs = [(p.formula, p.amount) for p in route.precursors]
        dG, decomp, stage = interpolated_reaction_dG_per_atom(tc, precs, target, route)
        stages[tier][stage] += 1
        if dG is not None:
            per_tier[tier].append(dG)
            if len(worked_examples[tier]) < 8:
                worked_examples[tier].append({
                    "target": target,
                    "dG_per_atom": round(dG, 4),
                    "decomposition": decomp,
                    "precursors": [f for f, _ in precs],
                })

    def dist(xs):
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        return {
            "n": len(xs),
            "min": round(min(xs), 4),
            "p10": round(xs_sorted[len(xs) // 10], 4),
            "median": round(statistics.median(xs), 4),
            "p90": round(xs_sorted[9 * len(xs) // 10], 4),
            "max": round(max(xs), 4),
            "mean": round(statistics.mean(xs), 4),
            "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "frac_negative": round(sum(1 for x in xs if x < 0) / len(xs), 3),
        }

    return {
        "path": str(path),
        "gradeable_via_interpolation": {
            tier: {"attempted": seen[tier], "graded": len(per_tier[tier]),
                    "rate": round(len(per_tier[tier]) / seen[tier], 3) if seen[tier] else 0.0}
            for tier in ("fractional", "integer")
        },
        "dG_distribution": {tier: dist(per_tier[tier]) for tier in ("fractional", "integer")},
        "stage_breakdown": {tier: dict(stages[tier]) for tier in ("fractional", "integer")},
        "worked_examples": {tier: worked_examples[tier] for tier in ("fractional", "integer")},
    }


def print_summary(r: dict):
    print(f"\n=== {r['path']} ===")
    print("gradeable via hull interpolation:")
    for tier, s in r["gradeable_via_interpolation"].items():
        print(f"  {tier:<12} {s['graded']}/{s['attempted']}  ({s['rate']:.1%})")
    print("\nΔG/atom distribution (eV):")
    for tier, d in r["dG_distribution"].items():
        if d["n"] == 0:
            print(f"  {tier:<12} (none graded)")
            continue
        print(f"  {tier:<12} n={d['n']:<5} median={d['median']:<8} "
              f"mean={d['mean']:<8} stdev={d['stdev']:<8} "
              f"[{d['min']}, {d['max']}]  frac_neg={d['frac_negative']}")
    print("\nstage breakdown (why records didn't grade):")
    for tier, sb in r["stage_breakdown"].items():
        print(f"  {tier:<12} {sb}")
    print("\nworked fractional examples (target -> ΔG, decomposition):")
    for ex in r["worked_examples"].get("fractional", [])[:5]:
        print(f"  {ex['target']:<22} ΔG={ex['dG_per_atom']:<9} "
              f"from {ex['precursors']}")
        print(f"      decomposes to: {ex['decomposition']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, dest="splits")
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--sample", type=int, default=300, help="max records per tier")
    ap.add_argument("--out", type=Path, default=Path("interpolation_probe_results.json"))
    args = ap.parse_args()

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for split_path in args.splits:
        p = Path(split_path)
        print(f"Probing {p} ...", file=sys.stderr)
        r = run(p, validator, args.sample)
        all_results[p.name] = r
        print_summary(r)

    with args.out.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()