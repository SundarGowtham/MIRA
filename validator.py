"""
validator.py
------------
Deterministic synthesis route validator for GRPO reward signal.

This version is pymatgen-grounded throughout: every chemistry check delegates
to a function in pymatgen.analysis.* that the Materials Project itself uses
in published workflows. Hand-coded heuristics (volatile-carrier ratios,
ad-hoc oxidation-state guessing) have been replaced.

Key pymatgen functions used (with the validator check they back):

  _check_stoichiometry          → pymatgen.analysis.reaction_calculator.Reaction
                                  (exact null-space balance, ReactionError on fail)
  _check_charge_neutrality      → Composition.oxi_state_guesses(all_oxi_states=True)
                                  + fractional-valence path using
                                  Element.oxidation_states (full known list)
  _check_thermodynamics         → reaction_calculator.ComputedReaction
                                  .calculated_reaction_energy
  _check_target_stability (NEW) → PhaseDiagram.get_e_above_hull
  _check_chempot_atmosphere     → PhaseDiagram.get_composition_chempots
    (NEW)                         (μ_O at target's facet → oxidizing/reducing/neutral)

Reward r ∈ [0.0, 1.0] — higher is better.

Modes:
  - Lightweight (default): 5 deterministic checks, no network.
  - Thermo-aware: adds ΔE_rxn (ComputedReaction), target stability, and
    chempot-atmosphere consistency. Requires a ThermoChecker with a
    precomputed PD cache.

Usage:
    from validator import SynthesisValidator
    validator = SynthesisValidator(mp_formula_set)
    reward, breakdown = validator.validate(predicted_route, "BaTiO3")

    # Thermo-aware mode (recommended for GRPO)
    from validator import SynthesisValidator, ThermoChecker
    thermo = ThermoChecker.from_sharded_cache("data/cache/pd_index.json", project_root)
    validator = SynthesisValidator(mp_formula_set, thermo_checker=thermo)
    reward, breakdown = validator.validate(predicted_route, "BaTiO3")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, TYPE_CHECKING

from pymatgen.core import Composition, Element
from pymatgen.analysis.reaction_calculator import (
    Reaction,
    ComputedReaction,
    ReactionError,
)

if TYPE_CHECKING:
    from pymatgen.analysis.phase_diagram import PhaseDiagram


# ---------------------------------------------------------------------------
# Data classes for predicted route (unchanged from prior version)
# ---------------------------------------------------------------------------

@dataclass
class PredictedConditions:
    heating_temperature: list[float] = field(default_factory=list)  # °C
    heating_time: list[float] = field(default_factory=list)          # hours
    heating_atmosphere: list[str] = field(default_factory=list)
    mixing_media: Optional[str] = None
    atmosphere: Optional[Literal["Ar", "N2", "vacuum", "air"]] = None


@dataclass
class PredictedOperation:
    type: str
    conditions: PredictedConditions = field(default_factory=PredictedConditions)


@dataclass
class PredictedPrecursor:
    formula: str
    amount: float = 1.0


@dataclass
class PredictedRoute:
    target_formula: str
    precursors: list[PredictedPrecursor]
    operations: list[PredictedOperation]
    reaction_string: str = ""


# ---------------------------------------------------------------------------
# Constraint weights
# ---------------------------------------------------------------------------

# Lightweight mode (no thermodynamics)
WEIGHTS_LIGHT = {
    "stoichiometry":         0.35,
    "charge_neutrality":     0.25,
    "precursors_exist":      0.20,
    "operation_order":       0.10,
    "temperature_plausible": 0.10,
}

# Thermo-aware mode — eight checks, all physics-grounded.
# Mass balance + reaction energy + target hull stability + atmosphere chemistry
# carry the most weight because they're the load-bearing chemistry signals.
WEIGHTS_THERMO = {
    "stoichiometry":            0.20,   # Reaction balances exactly?
    "thermodynamic_favorable":  0.20,   # ΔE_rxn from ComputedReaction
    "target_stability":         0.15,   # e_above_hull of target itself
    "chempot_atmosphere":       0.10,   # μ_O vs. operation atmosphere
    "charge_neutrality":        0.10,
    "precursors_exist":         0.10,
    "operation_order":          0.075,
    "temperature_plausible":    0.075,
}

assert abs(sum(WEIGHTS_LIGHT.values()) - 1.0) < 1e-9
assert abs(sum(WEIGHTS_THERMO.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Operation ordering rules (unchanged; no pymatgen analog for synthesis ontology)
# ---------------------------------------------------------------------------

OPERATION_ORDER = {
    "StartingSynthesis":  0,
    "MixingOperation":    1,
    "ShapingOperation":   2,
    "DryingOperation":    2,
    "HeatingOperation":   3,
    "QuenchingOperation": 4,
    "CleaningOperation":  5,
}

# Ops whose temperatures should be graded against TEMP_MIN/MAX
# (grinds, dryings, mixings happen at room temp — don't count them)
HEATING_OP_TYPES = frozenset({
    "HeatingOperation",
    "SinteringOperation",   # in case the parser emits this variant
    "QuenchingOperation",
})

TEMP_MIN = 100.0
TEMP_MAX = 2000.0


# ---------------------------------------------------------------------------
# Volatile byproducts admissible in solid-state synthesis.
# Used as candidate products in Reaction / ComputedReaction balancing.
# ---------------------------------------------------------------------------

VOLATILE_FORMULAS = ["CO2", "H2O", "O2", "N2", "NH3"]

# Anions whose oxidation state is essentially fixed in inorganic synthesis.
# Used only for the fractional-valence path in _check_charge_neutrality.
ANION_STATES: dict[str, int] = {
    "O":  -2, "F":  -1, "Cl": -1, "Br": -1, "I": -1,
    "S":  -2, "Se": -2, "Te": -2, "N": -3,
}

# Redox-active cations that can absorb fractional valence in mixed-valence
# phases (perovskites, spinels, layered oxides, etc.)
REDOX_METALS = frozenset({
    "Fe", "Mn", "Co", "Ni", "Cu", "V", "Cr", "Mo", "W", "Ti",
    "Ce", "Eu", "Pr", "Tb", "Sn", "Pb", "Bi", "Ru", "Ir", "Re",
})

# Default cation valences for non-redox cations in the fractional path
# (alkali, alkaline earth, rare earth excluding Ce/Eu/Pr/Tb)
FIXED_CATION_STATE: dict[str, int] = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    "Al": 3, "Ga": 3, "In": 3, "Sc": 3, "Y": 3,
    "La": 3, "Nd": 3, "Sm": 3, "Gd": 3, "Dy": 3,
    "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Zr": 4, "Hf": 4, "Ta": 5, "Nb": 5,
}


# ---------------------------------------------------------------------------
# Thermodynamic favorability thresholds (eV/atom)
# Based on MP-documented noise floors for GGA reaction energies.
# ---------------------------------------------------------------------------

RXN_ENERGY_FAVORABLE   = -0.025
RXN_ENERGY_BORDERLINE  =  0.025
RXN_ENERGY_UNFAVORABLE =  0.150

# Target-stability thresholds (eV/atom above hull)
HULL_STABLE     = 0.025   # on hull within DFT noise → 1.0
HULL_METASTABLE = 0.100   # 25–100 meV: metastable, synthesizable → linear decay
HULL_UNSTABLE   = 0.250   # > 250 meV: probably not a real phase → 0.0

# μ_O thresholds (eV, relative to elemental O2 reference at PD's facet)
# These are heuristic but grounded in MP convex-hull behavior for oxides.
MU_O_OXIDIZING_REQUIRED = -1.0   # Δμ_O > this → target needs air/O2
MU_O_REDUCING_REQUIRED  = -3.0   # Δμ_O < this → target needs inert/H2


# ---------------------------------------------------------------------------
# Atmosphere classification (used by chempot_atmosphere check)
# ---------------------------------------------------------------------------

ATMOSPHERE_OXIDIZING = frozenset({"air", "o2", "oxygen"})
ATMOSPHERE_REDUCING  = frozenset({"h2", "hydrogen", "co", "forming gas", "formgas"})
ATMOSPHERE_INERT     = frozenset({"ar", "argon", "n2", "nitrogen", "vacuum", "he", "helium"})


def _classify_atmosphere(atm: str) -> str:
    """Return one of: 'ox', 'red', 'inert', 'unknown'."""
    a = atm.lower().strip()
    if any(o in a for o in ATMOSPHERE_OXIDIZING):
        return "ox"
    if any(r in a for r in ATMOSPHERE_REDUCING):
        return "red"
    if any(i in a for i in ATMOSPHERE_INERT):
        return "inert"
    return "unknown"


# ---------------------------------------------------------------------------
# ThermoChecker — wraps a phase-diagram cache and exposes the pymatgen-native
# thermodynamic queries the validator uses.
# ---------------------------------------------------------------------------

class ThermoChecker:
    """
    Precomputed phase-diagram cache for thermodynamic checks.

    Three queries are exposed, each backed by a pymatgen function:

      reaction_energy_per_atom → ComputedReaction.calculated_reaction_energy
                                 (auto-balances reaction including volatiles)
      target_e_above_hull      → PhaseDiagram.get_e_above_hull
      composition_chempots     → PhaseDiagram.get_composition_chempots
    """

    def __init__(
        self,
        phase_diagrams: dict[str, "PhaseDiagram"],
        pd_index: dict | None = None,
        project_root: Path | None = None,
    ):
        self.phase_diagrams = phase_diagrams
        self.pd_index = pd_index or {}
        self.project_root = project_root

    @classmethod
    def from_sharded_cache(
        cls,
        index_path: str | Path,
        project_root: Path,
    ) -> "ThermoChecker":
        import json
        path = Path(index_path)
        if not path.exists():
            return cls(phase_diagrams={}, pd_index={}, project_root=project_root)
        with path.open("r") as f:
            pd_index = json.load(f)
        return cls(phase_diagrams={}, pd_index=pd_index, project_root=project_root)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pd(self, chemsys: str):
        """Lazy-load a PD shard."""
        if chemsys in self.phase_diagrams:
            return self.phase_diagrams[chemsys]
        if self.pd_index and chemsys in self.pd_index and self.project_root:
            shard_path = self.project_root / self.pd_index[chemsys]
            if shard_path.exists():
                try:
                    with shard_path.open("rb") as f:
                        pd = pickle.load(f)
                    self.phase_diagrams[chemsys] = pd
                    return pd
                except Exception:
                    pass
        return None

    def _resolve_pd(self, formulas: list[str]):
        """
        Find a cached PD whose chemsys covers every element in formulas.
        Tries exact match first, then any superset chemsys.
        Returns (pd, chemsys) or (None, "") if no covering PD exists.
        """
        try:
            all_els = set()
            for f in formulas:
                for el in Composition(f).elements:
                    all_els.add(str(el))
        except Exception:
            return None, ""
        chemsys = "-".join(sorted(all_els))
        pd = self._get_pd(chemsys)
        if pd is not None:
            return pd, chemsys
        if self.pd_index:
            for cs in self.pd_index:
                if all_els.issubset(set(cs.split("-"))):
                    pd = self._get_pd(cs)
                    if pd is not None:
                        return pd, cs
        return None, chemsys

    @staticmethod
    def _best_entry_for_formula(pd, formula: str):
        """Lowest-energy PD entry whose reduced formula matches the input."""
        try:
            target_red = Composition(formula).reduced_formula
        except Exception:
            return None
        matches = [
            e for e in pd.all_entries
            if e.composition.reduced_formula == target_red
        ]
        if not matches:
            return None
        return min(matches, key=lambda e: e.energy_per_atom)

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def reaction_energy_per_atom(
        self,
        precursors: list[tuple[str, float]],
        target_formula: str,
    ) -> Optional[float]:
        """
        Compute per-atom reaction energy using pymatgen's ComputedReaction.

        ComputedReaction re-balances the reaction from compositions and computes
        the energy from the entries' total energies. The model's stated
        coefficients are irrelevant (and rightly so — RL shouldn't be teaching
        the model to do stoichiometric arithmetic).

        Returns None if the reaction can't be balanced or required entries are
        missing from the PD.
        """
        try:
            all_formulas = [target_formula] + [f for f, _ in precursors] + VOLATILE_FORMULAS
            pd, _ = self._resolve_pd(all_formulas)
            if pd is None:
                return None

            target_entry = self._best_entry_for_formula(pd, target_formula)
            if target_entry is None:
                return None

            reactant_entries = []
            for formula, _ in precursors:
                e = self._best_entry_for_formula(pd, formula)
                if e is None:
                    return None
                reactant_entries.append(e)

            volatile_entries = []
            for v in VOLATILE_FORMULAS:
                e = self._best_entry_for_formula(pd, v)
                if e is not None:
                    volatile_entries.append(e)

            product_entries = [target_entry] + volatile_entries

            try:
                reaction = ComputedReaction(reactant_entries, product_entries)
            except ReactionError:
                return None

            # Sign convention: products positive, reactants negative.
            # Target must appear with positive coefficient.
            target_coeff = reaction.get_coeff(target_entry.composition)
            if target_coeff <= 1e-6:
                return None

            delta_E_total = reaction.calculated_reaction_energy  # eV
            atoms_target = target_coeff * target_entry.composition.num_atoms
            return delta_E_total / atoms_target

        except Exception:
            return None

    def target_e_above_hull(self, target_formula: str) -> Optional[float]:
        """
        Energy above convex hull of the target itself (eV/atom).

        Tells us whether the target is stable in MP's PD or metastable.
        Returns None if target isn't a PD entry (e.g., novel solid solutions
        not in MP).
        """
        try:
            pd, _ = self._resolve_pd([target_formula])
            if pd is None:
                return None
            target_entry = self._best_entry_for_formula(pd, target_formula)
            if target_entry is None:
                return None
            return pd.get_e_above_hull(target_entry, on_error="ignore")
        except Exception:
            return None

    def composition_chempots(self, target_formula: str) -> Optional[dict]:
        """
        ALL facet chempots for the target via PhaseDiagram.get_all_chempots.

        Returns {facet_name: {Element: chempot}} so the caller can compute
        the Δμ_O envelope rather than relying on a single facet (which
        get_composition_chempots picks arbitrarily — typically the O2-rich
        one, masking reducing-stability behavior).

        Returns None on failure.
        """
        try:
            pd, _ = self._resolve_pd([target_formula])
            if pd is None:
                return None
            target_comp = Composition(target_formula)
            return pd.get_all_chempots(target_comp)
        except Exception:
            return None

    def oxygen_reference_energy(self, target_formula: str) -> Optional[float]:
        """
        Per-atom energy of the O2 elemental reference in the relevant PD.
        Used to convert absolute μ_O to Δμ_O (relative to O2-rich limit).
        """
        try:
            pd, _ = self._resolve_pd([target_formula])
            if pd is None:
                return None
            o_el = Element("O")
            if o_el not in pd.el_refs:
                return None
            return pd.el_refs[o_el].energy_per_atom
        except Exception:
            return None

    def __contains__(self, chemsys: str) -> bool:
        return chemsys in self.phase_diagrams or chemsys in self.pd_index

    def __len__(self) -> int:
        return len(self.phase_diagrams) + len(self.pd_index)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SynthesisValidator:
    """
    Scores a predicted synthesis route against physical/chemical constraints.

    Args:
        mp_formula_set: set of known MP formula strings (reduced form).
        thermo_checker: optional ThermoChecker for thermo-aware mode.
    """

    def __init__(
        self,
        mp_formula_set: set[str],
        thermo_checker: Optional[ThermoChecker] = None,
    ):
        self.mp_formula_set = {self._normalize_formula(f) for f in mp_formula_set}
        self.thermo_checker = thermo_checker
        self.weights = (
            WEIGHTS_THERMO if thermo_checker is not None else WEIGHTS_LIGHT
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(
        self,
        predicted: PredictedRoute,
        ground_truth_target_formula: Optional[str] = None,
    ) -> tuple[float, dict[str, float]]:
        scores: dict[str, float] = {}

        scores["stoichiometry"]         = self._check_stoichiometry(predicted)
        scores["charge_neutrality"]     = self._check_charge_neutrality(predicted)
        scores["precursors_exist"]      = self._check_precursors_exist(predicted)
        scores["operation_order"]       = self._check_operation_order(predicted)
        scores["temperature_plausible"] = self._check_temperature(predicted)

        if self.thermo_checker is not None:
            scores["thermodynamic_favorable"] = self._check_thermodynamics(predicted)
            scores["target_stability"]        = self._check_target_stability(predicted)
            scores["chempot_atmosphere"]      = self._check_chempot_atmosphere(predicted)

        if ground_truth_target_formula is not None:
            scores["target_match"] = self._check_target_match(
                predicted, ground_truth_target_formula
            )

        active_weights = {k: v for k, v in self.weights.items() if k in scores}
        weight_sum = sum(active_weights.values())
        if weight_sum == 0:
            return 0.0, scores
        reward = sum(
            (w / weight_sum) * scores[k] for k, w in active_weights.items()
        )
        return round(reward, 4), scores

    # ------------------------------------------------------------------
    # Individual constraint checks
    # ------------------------------------------------------------------

    def _check_stoichiometry(self, predicted: PredictedRoute) -> float:
        """
        Mass balance via pymatgen.analysis.reaction_calculator.Reaction.

        Reaction performs exact null-space balancing on the elemental
        composition matrix. If a balanced equation exists with the target
        on the product side and the precursors on the reactant side
        (allowing CO2/H2O/O2/N2/NH3 as volatile byproducts), score is 1.0.
        Otherwise 0.0.

        Binary because the underlying mathematical question is binary:
        either a balance exists or it doesn't.
        """
        try:
            reactants = [Composition(p.formula) for p in predicted.precursors]
            if not reactants:
                return 0.0
            target_comp = Composition(predicted.target_formula)

            # Try with progressively more volatile candidates. Some routes
            # don't release any gases (e.g., oxide + oxide → oxide); Reaction
            # is happier with the minimum set that lets it balance.
            candidate_volatile_sets = [
                [],                                                # no volatiles
                [Composition(v) for v in ["CO2"]],                # carbonate routes
                [Composition(v) for v in ["H2O"]],                # hydrate routes
                [Composition(v) for v in ["O2"]],                 # redox routes
                [Composition(v) for v in ["CO2", "H2O", "O2"]],   # full common set
                [Composition(v) for v in VOLATILE_FORMULAS],      # everything
            ]

            for volatile_set in candidate_volatile_sets:
                products = [target_comp] + volatile_set
                try:
                    reaction = Reaction(reactants, products)
                except ReactionError:
                    continue
                # Target must be produced with positive coefficient
                target_coeff = reaction.get_coeff(target_comp)
                if target_coeff <= 1e-6:
                    continue
                # All precursors must actually be consumed (negative coeff)
                all_used = all(
                    reaction.get_coeff(r) < -1e-6 for r in reactants
                )
                if all_used:
                    return 1.0

            return 0.0

        except Exception:
            return 0.0

    def _check_charge_neutrality(self, predicted: PredictedRoute) -> float:
        """
        Two-stage check:

        1. Try Composition.oxi_state_guesses(all_oxi_states=True). If any
           integer assignment exists, score 1.0.

        2. For solid solutions with non-integer subscripts (e.g.,
           La0.5Sr0.5FeO3 where Fe averages +3.5), do a fractional-valence
           check: hold non-redox cations at their default valence, compute
           the required average valence on redox cations, and check it lies
           within their accessible range (Element.oxidation_states).

        Continuous outside the accessible range: distance-to-range decay
        with a 0.5-valence tolerance.
        """
        try:
            comp = Composition(predicted.target_formula)

            # Stage 1: pymatgen's integer oxidation-state guesser using
            # COMMON states only. all_oxi_states=True is too permissive — it
            # accepts Na^-1 + Cl^+0.5 for NaCl2 because it allows every known
            # state of every element. Default behavior uses ICSD-frequency
            # priors and rejects chemical nonsense.
            try:
                guesses = comp.oxi_state_guesses()
                if guesses:
                    return 1.0
            except Exception:
                pass

            # Stage 2: fractional-valence path for mixed-valence solid solutions
            # (La0.5Sr0.5FeO3, YBa2Cu3O7, La1-xCaxMnO3, etc.) — where the
            # integer guesser correctly fails because the average metal
            # valence is non-integer.
            return self._fractional_valence_check(comp)

        except Exception:
            return 0.0

    def _fractional_valence_check(self, comp: Composition) -> float:
        """
        Mixed-valence path: solve for required average valence on redox metals.
        """
        comp_dict = comp.as_dict()

        anion_charge = 0.0
        cations: list[tuple[str, float]] = []
        for el, amt in comp_dict.items():
            if el in ANION_STATES:
                anion_charge += ANION_STATES[el] * amt
            else:
                cations.append((el, amt))

        if not cations or anion_charge >= 0:
            return 0.5

        required_cation_charge = -anion_charge

        fixed_cations = [(el, amt) for el, amt in cations if el not in REDOX_METALS]
        redox_cations = [(el, amt) for el, amt in cations if el in REDOX_METALS]

        fixed_charge = 0.0
        for el, amt in fixed_cations:
            if el in FIXED_CATION_STATE:
                fixed_charge += FIXED_CATION_STATE[el] * amt
            else:
                # Unknown non-redox cation — fall back to most common positive
                # state from pymatgen's Element.oxidation_states
                try:
                    states = [s for s in Element(el).oxidation_states if s > 0]
                    if not states:
                        return 0.5
                    fixed_charge += states[0] * amt
                except Exception:
                    return 0.5

        redox_charge_needed = required_cation_charge - fixed_charge

        if not redox_cations:
            # No mixed-valence flexibility; exact integer match required
            return 1.0 if abs(redox_charge_needed) < 1e-3 else 0.0

        total_redox_amt = sum(amt for _, amt in redox_cations)
        if total_redox_amt <= 0:
            return 0.0
        avg_required = redox_charge_needed / total_redox_amt

        # Check accessibility for each redox cation (best score wins)
        best_score = 0.0
        for el, _ in redox_cations:
            try:
                states = [s for s in Element(el).oxidation_states if s > 0]
            except Exception:
                continue
            if not states:
                continue
            min_s, max_s = min(states), max(states)
            if min_s <= avg_required <= max_s:
                return 1.0
            # Distance-based continuous decay (tolerance = 0.5 valence units)
            if avg_required < min_s:
                score = max(0.0, 1.0 - (min_s - avg_required) / 0.5)
            else:
                score = max(0.0, 1.0 - (avg_required - max_s) / 0.5)
            best_score = max(best_score, score)

        return best_score

    def _check_precursors_exist(self, predicted: PredictedRoute) -> float:
        """Fraction of precursors found in the MP formula set."""
        if not predicted.precursors:
            return 0.0
        hits = sum(
            1 for p in predicted.precursors
            if self._normalize_formula(p.formula) in self.mp_formula_set
        )
        return hits / len(predicted.precursors)

    def _check_operation_order(self, predicted: PredictedRoute) -> float:
        """
        Fraction of adjacent op pairs in valid order.

        Solid-state synthesis routinely uses regrind cycles: calcine → grind
        → calcine → grind → sinter is standard practice (intermediate
        homogenization between heating steps). A naive monotone-rank check
        punishes this; we treat heating→mixing→heating as a legitimate
        regrind cycle, not a violation.
        """
        if not predicted.operations:
            return 0.0
        if len(predicted.operations) == 1:
            return 1.0

        ranks = [
            OPERATION_ORDER.get(self._normalize_op_type(op.type), 99)
            for op in predicted.operations
        ]
        HEATING_RANK = OPERATION_ORDER["HeatingOperation"]
        MIXING_RANK  = OPERATION_ORDER["MixingOperation"]

        pairs = list(zip(ranks[:-1], ranks[1:]))
        violations = 0
        for i, (a, b) in enumerate(pairs):
            if a > b:
                # Rank dropped. Check if it's a heating→mixing regrind cycle
                # where another heating step follows later.
                if a == HEATING_RANK and b <= MIXING_RANK + 1:
                    if any(r >= HEATING_RANK for r in ranks[i + 2:]):
                        continue   # legitimate regrind, not a violation
                violations += 1
        return 1.0 - violations / len(pairs)

    def _check_temperature(self, predicted: PredictedRoute) -> float:
        """
        Fraction of HEATING-type operation temperatures in [TEMP_MIN, TEMP_MAX].

        FIX from prior version: grinds, mixings, and dryings at room temperature
        no longer count against the heating-temperature plausibility check.
        """
        temps = []
        for op in predicted.operations:
            op_type = self._normalize_op_type(op.type)
            if op_type in HEATING_OP_TYPES:
                temps.extend(op.conditions.heating_temperature)

        if not temps:
            return 0.5  # no heating temps specified — neutral
        in_range = [1.0 if TEMP_MIN <= t <= TEMP_MAX else 0.0 for t in temps]
        return sum(in_range) / len(in_range)

    def _check_thermodynamics(self, predicted: PredictedRoute) -> float:
        """
        ΔE_rxn via ComputedReaction. Continuous score with piecewise-linear
        mapping from eV/atom to [0, 1].
        """
        if self.thermo_checker is None:
            return 0.5

        try:
            precursor_pairs = [(p.formula, p.amount) for p in predicted.precursors]
            delta_E = self.thermo_checker.reaction_energy_per_atom(
                precursor_pairs, predicted.target_formula
            )
        except Exception:
            return 0.5

        if delta_E is None:
            return 0.5

        if delta_E <= RXN_ENERGY_FAVORABLE:
            return 1.0
        elif delta_E <= RXN_ENERGY_BORDERLINE:
            t = (delta_E - RXN_ENERGY_FAVORABLE) / (RXN_ENERGY_BORDERLINE - RXN_ENERGY_FAVORABLE)
            return 1.0 - 0.5 * t
        elif delta_E <= RXN_ENERGY_UNFAVORABLE:
            t = (delta_E - RXN_ENERGY_BORDERLINE) / (RXN_ENERGY_UNFAVORABLE - RXN_ENERGY_BORDERLINE)
            return 0.5 - 0.5 * t
        else:
            return 0.0

    def _check_target_stability(self, predicted: PredictedRoute) -> float:
        """
        Score based on e_above_hull of the target itself.

        Continuous:
          ≤ 25 meV/atom    → 1.0 (on hull within DFT noise)
          25–100 meV/atom  → linear decay 1.0 → 0.5 (synthesizable metastable)
          100–250 meV/atom → linear decay 0.5 → 0.0
          > 250 meV/atom   → 0.0

        Returns 0.5 if the target isn't a discrete PD entry (e.g., novel
        solid solution). This is the right neutral signal: we shouldn't
        punish the model for working on a composition MP hasn't computed.
        """
        if self.thermo_checker is None:
            return 0.5
        try:
            e_hull = self.thermo_checker.target_e_above_hull(predicted.target_formula)
        except Exception:
            return 0.5
        if e_hull is None:
            return 0.5

        if e_hull <= HULL_STABLE:
            return 1.0
        elif e_hull <= HULL_METASTABLE:
            t = (e_hull - HULL_STABLE) / (HULL_METASTABLE - HULL_STABLE)
            return 1.0 - 0.5 * t
        elif e_hull <= HULL_UNSTABLE:
            t = (e_hull - HULL_METASTABLE) / (HULL_UNSTABLE - HULL_METASTABLE)
            return 0.5 - 0.5 * t
        else:
            return 0.0

    def _check_chempot_atmosphere(self, predicted: PredictedRoute) -> float:
        """
        Atmosphere consistency with the Δμ_O envelope at the target's
        stability facets, via PhaseDiagram.get_all_chempots.

        Logic:
          1. If target contains no oxygen → 1.0 (atmosphere check doesn't apply).
          2. If PD unavailable → 0.5 (neutral, can't determine).
          3. Compute Δμ_O envelope [mu_lo, mu_hi] across all stability facets,
             where mu_lo is the lowest μ_O the target survives.
          4. Classify atmosphere requirement using mu_lo (the binding constraint):
               mu_lo > -1.0  → oxidizing required
               mu_hi < -3.0  → reducing required
               otherwise     → flexible
          5. Compare against the atmospheres declared in heating operations:
               - omitted + chemistry demands it → 0.0
               - omitted + flexible regime     → 1.0
               - declared + matches need       → 1.0
               - declared + contradicts need   → 0.0
        """
        if self.thermo_checker is None:
            return 0.5

        # Step 1: short-circuit for non-oxide targets — the check doesn't apply.
        try:
            target_comp = Composition(predicted.target_formula)
            if Element("O") not in target_comp.elements:
                return 1.0
        except Exception:
            return 0.5

        # Step 2: fetch chempot envelope.
        try:
            all_chempots = self.thermo_checker.composition_chempots(predicted.target_formula)
            mu_O_ref = self.thermo_checker.oxygen_reference_energy(predicted.target_formula)
        except Exception:
            return 0.5

        if not all_chempots or mu_O_ref is None:
            return 0.5

        o_el = Element("O")
        o_mus_delta = [
            float(facet[o_el]) - float(mu_O_ref)
            for facet in all_chempots.values()
            if o_el in facet
        ]
        if not o_mus_delta:
            # PD has oxygen ref but no facet exposes μ_O — unusual; be neutral.
            return 0.5

        mu_lo, mu_hi = min(o_mus_delta), max(o_mus_delta)
        needs_oxidizing = mu_lo > MU_O_OXIDIZING_REQUIRED
        needs_reducing  = mu_hi < MU_O_REDUCING_REQUIRED

        # Step 5: compare against operation atmospheres.
        atms = []
        for op in predicted.operations:
            op_type = self._normalize_op_type(op.type)
            if op_type in HEATING_OP_TYPES:
                atms.extend(op.conditions.heating_atmosphere)
        classes = [_classify_atmosphere(a) for a in atms]

        if not classes:
            if needs_oxidizing or needs_reducing:
                return 0.0
            return 1.0

        if needs_oxidizing:
            return 1.0 if all(c == "ox" for c in classes) else 0.0
        if needs_reducing:
            return 1.0 if all(c in {"red", "inert"} for c in classes) else 0.0
        return 1.0

    def _check_target_match(
        self,
        predicted: PredictedRoute,
        ground_truth_formula: str,
    ) -> float:
        try:
            pred = Composition(predicted.target_formula).reduced_formula
            true = Composition(ground_truth_formula).reduced_formula
            return 1.0 if pred == true else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_formula(formula: str) -> str:
        try:
            return Composition(formula).reduced_formula
        except Exception:
            return formula.strip()

    @staticmethod
    def _normalize_op_type(op_type: str) -> str:
        """
        Map a model-emitted op type string to a canonical internal name.

        Recognized synonyms span the verbs the model actually emits in
        practice (mix, calcine, sinter, anneal, cool, quench, etc.) plus
        the longer Class-style names from the parser (HeatingOperation,
        etc.). Unknown strings fall through unchanged so downstream code
        can log them.
        """
        if "." in op_type:
            op_type = op_type.split(".")[-1]
        mapping = {
            # StartingSynthesis
            "starting":             "StartingSynthesis",
            "startingsynthesis":    "StartingSynthesis",

            # MixingOperation — every prep-stage verb collapses here
            "mix":                  "MixingOperation",
            "mixing":               "MixingOperation",
            "mixingoperation":      "MixingOperation",
            "grind":                "MixingOperation",
            "grinding":             "MixingOperation",
            "ball_mill":            "MixingOperation",
            "ballmill":             "MixingOperation",
            "ball-milling":         "MixingOperation",
            "ballmilling":          "MixingOperation",
            "mortar":               "MixingOperation",

            # DryingOperation
            "dry":                  "DryingOperation",
            "drying":               "DryingOperation",
            "dryingoperation":      "DryingOperation",

            # ShapingOperation
            "shape":                "ShapingOperation",
            "shaping":              "ShapingOperation",
            "shapingoperation":     "ShapingOperation",
            "press":                "ShapingOperation",
            "pellet":               "ShapingOperation",
            "pelletize":            "ShapingOperation",
            "pelletizing":          "ShapingOperation",

            # HeatingOperation — every furnace verb collapses here
            "heat":                 "HeatingOperation",
            "heating":              "HeatingOperation",
            "heatingoperation":     "HeatingOperation",
            "calcine":              "HeatingOperation",
            "calcination":          "HeatingOperation",
            "calcining":            "HeatingOperation",
            "fire":                 "HeatingOperation",
            "firing":               "HeatingOperation",
            "anneal":               "HeatingOperation",
            "annealing":            "HeatingOperation",
            "sinter":               "HeatingOperation",
            "sintering":            "HeatingOperation",
            "sinteringoperation":   "HeatingOperation",
            "reaction":             "HeatingOperation",
            "react":                "HeatingOperation",
            "sealed_tube_heating":  "HeatingOperation",
            "sealedtubeheating":    "HeatingOperation",
            "sealed-tube":          "HeatingOperation",
            "ampoule":              "HeatingOperation",
            "hydrothermal":         "HeatingOperation",
            "solidstate":           "HeatingOperation",
            "solid_state":          "HeatingOperation",

            # QuenchingOperation — both rapid and slow cooling
            "quench":               "QuenchingOperation",
            "quenching":            "QuenchingOperation",
            "quenchingoperation":   "QuenchingOperation",
            "cool":                 "QuenchingOperation",
            "cooling":              "QuenchingOperation",
            "slow_cool":            "QuenchingOperation",
            "slowcool":             "QuenchingOperation",
            "slow_cooling":         "QuenchingOperation",

            # CleaningOperation
            "clean":                "CleaningOperation",
            "cleaning":             "CleaningOperation",
            "cleaningoperation":    "CleaningOperation",
            "wash":                 "CleaningOperation",
            "washing":              "CleaningOperation",
            "filter":               "CleaningOperation",
            "filtering":            "CleaningOperation",
            "rinse":                "CleaningOperation",
        }
        return mapping.get(op_type.lower().strip(), op_type)


# ---------------------------------------------------------------------------
# GRPO scoring helpers (unchanged)
# ---------------------------------------------------------------------------

def score_group(
    validator: SynthesisValidator,
    routes: list[PredictedRoute],
    ground_truth_formula: str,
) -> list[tuple[float, dict]]:
    return [validator.validate(route, ground_truth_formula) for route in routes]


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    import statistics
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
    eps = 1e-8
    return [(r - mean) / (std + eps) for r in rewards]

