"""
validator.py
------------
Deterministic synthesis route validator for GRPO reward signal.

Takes a predicted synthesis route (dict) and scores it against
physical/chemical constraints. No LLM involved anywhere.
Every check is a rules engine or database lookup.

Reward r ∈ [0.0, 1.0] — higher is better.

The validator has two modes:
  - Lightweight (default): 5 deterministic checks, no network.
  - Thermo-aware: adds reaction energy check via MP phase diagrams,
    requires a ThermoChecker with a precomputed PD cache.

Usage:
    # Lightweight mode (training startup, no network)
    from validator import SynthesisValidator
    validator = SynthesisValidator(mp_formula_set)
    reward, breakdown = validator.validate(predicted_route, "BaTiO3")

    # Thermo-aware mode (recommended for GRPO training)
    from validator import SynthesisValidator, ThermoChecker
    thermo = ThermoChecker.from_cache("data/cache/phase_diagrams.pkl")
    validator = SynthesisValidator(mp_formula_set, thermo_checker=thermo)
    reward, breakdown = validator.validate(predicted_route, "BaTiO3")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from pymatgen.core import Composition

if TYPE_CHECKING:
    # Heavy imports only loaded when ThermoChecker is constructed
    from pymatgen.analysis.phase_diagram import PhaseDiagram


# ---------------------------------------------------------------------------
# Data classes for predicted route
# (mirrors the MP synthesis schema so SFT output maps cleanly)
# ---------------------------------------------------------------------------

@dataclass
class PredictedConditions:
    heating_temperature: list[float] = field(default_factory=list)  # °C
    heating_time: list[float] = field(default_factory=list)          # hours
    heating_atmosphere: list[str] = field(default_factory=list)
    mixing_media: Optional[str] = None


@dataclass
class PredictedOperation:
    type: str   # StartingSynthesis | MixingOperation | DryingOperation |
                # HeatingOperation | ShapingOperation | QuenchingOperation
    conditions: PredictedConditions = field(default_factory=PredictedConditions)


@dataclass
class PredictedPrecursor:
    formula: str                # e.g. "BaCO3"
    amount: float = 1.0         # stoichiometric coefficient


@dataclass
class PredictedRoute:
    """
    The structured output the LLM is trained to produce.
    Every field maps to a verifiable constraint.
    """
    target_formula: str                          # e.g. "BaTiO3"
    precursors: list[PredictedPrecursor]
    operations: list[PredictedOperation]
    reaction_string: str = ""                    # human-readable (not scored)


# ---------------------------------------------------------------------------
# Constraint weights — two configurations
# Both must sum to 1.0 within their respective scope
# ---------------------------------------------------------------------------

# Lightweight mode (no thermodynamics)
WEIGHTS_LIGHT = {
    "stoichiometry":         0.35,   # most important — mass must balance
    "charge_neutrality":     0.25,   # target must be charge neutral
    "precursors_exist":      0.20,   # precursors must be real materials
    "operation_order":       0.10,   # operations must be physically ordered
    "temperature_plausible": 0.10,   # temperatures must be realistic
}

# Thermo-aware mode — reduces other weights to fit thermodynamic check.
# Stoichiometry and thermodynamic favorability are both physics-grounded
# and weighted highest. Precursor existence drops because the thermo
# check partially covers it (unknown precursors → no PD entry → 0.0).
WEIGHTS_THERMO = {
    "stoichiometry":            0.25,   # mass balance
    "thermodynamic_favorable":  0.25,   # ΔE_rxn ≤ 0 via MP phase diagram
    "charge_neutrality":        0.15,
    "precursors_exist":         0.15,
    "operation_order":          0.10,
    "temperature_plausible":    0.10,
}

assert abs(sum(WEIGHTS_LIGHT.values()) - 1.0) < 1e-9, "Light weights must sum to 1.0"
assert abs(sum(WEIGHTS_THERMO.values()) - 1.0) < 1e-9, "Thermo weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Operation ordering rules
# Each operation type has a valid position range in the sequence.
# StartingSynthesis must be first. HeatingOperation must be after Mixing.
# ---------------------------------------------------------------------------

# Lower number = must come earlier
OPERATION_ORDER = {
    "StartingSynthesis":  0,
    "MixingOperation":    1,
    "ShapingOperation":   2,   # pressing/pelletizing comes after mixing
    "DryingOperation":    2,   # drying can happen at same stage as shaping
    "HeatingOperation":   3,   # always last (sintering/calcination)
    "QuenchingOperation": 4,   # after heating
    "CleaningOperation":  5,
}


# ---------------------------------------------------------------------------
# Temperature plausibility bounds (°C)
# Based on typical solid-state synthesis ranges in MP synthesis dataset
# ---------------------------------------------------------------------------

TEMP_MIN = 100.0    # below this = not a real firing step
TEMP_MAX = 2000.0   # above this = beyond standard lab furnace


# ---------------------------------------------------------------------------
# Known oxidation states for common elements in inorganic synthesis
# Used when MP API data is unavailable (fallback only)
# ---------------------------------------------------------------------------

COMMON_OX_STATES: dict[str, list[int]] = {
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1],
    "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2],
    "Al": [3], "Ga": [3], "In": [3],
    "Si": [4, -4], "Ge": [4, 2],
    "N":  [-3, 3, 5], "P": [-3, 3, 5], "As": [3, 5],
    "O":  [-2], "S": [-2, 4, 6], "Se": [-2, 4, 6],
    "F":  [-1], "Cl": [-1, 1, 3, 5, 7], "Br": [-1, 1],
    "Ti": [4, 3, 2], "Zr": [4], "Hf": [4],
    "V":  [5, 4, 3, 2], "Nb": [5, 4, 3], "Ta": [5],
    "Cr": [3, 6, 2], "Mo": [6, 4, 3], "W": [6, 4],
    "Mn": [2, 3, 4, 7], "Fe": [2, 3], "Co": [2, 3],
    "Ni": [2, 3], "Cu": [1, 2], "Zn": [2],
    "Y":  [3], "La": [3],
    "Ce": [3, 4], "Pr": [3], "Nd": [3], "Sm": [3],
    "Eu": [2, 3], "Gd": [3], "Tb": [3, 4], "Dy": [3],
    "Ho": [3], "Er": [3], "Tm": [3], "Yb": [2, 3], "Lu": [3],
    "Pb": [2, 4], "Bi": [3, 5], "Sn": [2, 4],
}


# ---------------------------------------------------------------------------
# Thermodynamic favorability thresholds (eV/atom)
# Based on MP-documented noise floors:
#   - GGA reaction energy std for ternary oxides: ~24 meV/atom
#   - Oxide → ternary oxide reaction MAE: ~14 kJ/mol-atom (~0.15 eV/atom)
# We give partial credit between strongly-favorable and clearly-uphill.
# ---------------------------------------------------------------------------

RXN_ENERGY_FAVORABLE   = -0.025   # ≤ this = full credit (ΔE/atom in eV)
RXN_ENERGY_BORDERLINE  =  0.025   # 0 ± noise floor → 1.0 → 0.5 credit linearly
RXN_ENERGY_UNFAVORABLE =  0.150   # > this = zero credit (clearly uphill)


# ---------------------------------------------------------------------------
# ThermoChecker
# ---------------------------------------------------------------------------

class ThermoChecker:
    """
    Precomputed phase-diagram cache for thermodynamic favorability checks.

    Building PhaseDiagrams from the MP API is expensive (~1-3s per chemsys
    + entries fetch). For training we precompute one PhaseDiagram per
    unique chemsys in the dataset, cache to disk, and load at validator init.

    The check answers: "Given precursors P_i with stoichiometric amounts a_i,
    does the reaction Σ a_i * P_i → target have ΔE ≤ 0 per atom?"
    Uses MP's mixed and corrected energies via the convex hull of the
    union chemsys.

    Cache structure:
        {chemsys_string: PhaseDiagram}

    where chemsys_string is e.g. "Ba-O-Ti" (alphabetized, hyphen-joined).
    """

    def __init__(self, phase_diagrams: dict[str, "PhaseDiagram"]):
        self.phase_diagrams = phase_diagrams

    @classmethod
    def from_cache(cls, cache_path: str | Path) -> "ThermoChecker":
        """Load a precomputed PD cache from disk (pickle)."""
        path = Path(cache_path)
        with path.open("rb") as f:
            phase_diagrams = pickle.load(f)
        return cls(phase_diagrams)

    @classmethod
    def build_cache(
        cls,
        target_formulas: list[str],
        api_key: Optional[str] = None,
        save_path: Optional[str | Path] = None,
    ) -> "ThermoChecker":
        """
        Build PD cache from MP API for a set of target formulas.

        For each target, computes the union chemsys with reasonable
        precursors and builds one PhaseDiagram per unique chemsys.
        Use this once during data preparation, not at training time.

        Args:
            target_formulas: list of target material formulas
            api_key: MP API key (None reads from env MP_API_KEY)
            save_path: optional path to pickle the resulting cache

        Returns:
            ThermoChecker instance
        """
        # Heavy imports only here, not at module load
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

        # Group targets by chemsys (one PD per unique chemsys is enough)
        chemsys_set = set()
        for formula in target_formulas:
            try:
                els = sorted(
                    str(el) for el in Composition(formula).elements
                )
                chemsys_set.add("-".join(els))
            except Exception:
                continue

        phase_diagrams: dict[str, "PhaseDiagram"] = {}
        scheme = MaterialsProjectDFTMixingScheme()

        with MPRester(api_key) as mpr:
            for chemsys in sorted(chemsys_set):
                try:
                    elements = chemsys.split("-")
                    entries = mpr.get_entries_in_chemsys(
                        elements=elements,
                        additional_criteria={
                            "thermo_types": ["GGA_GGA+U", "R2SCAN"]
                        },
                    )
                    # Re-apply mixing scheme locally — corrections from the
                    # API are scoped to each material's "home" chemsys, not
                    # the union we need for reaction energy.
                    entries = scheme.process_entries(entries)
                    if entries:
                        phase_diagrams[chemsys] = PhaseDiagram(entries)
                except Exception as e:
                    # Network or data issue — skip this chemsys
                    print(f"[ThermoChecker] skip {chemsys}: {e}")

        if save_path is not None:
            with Path(save_path).open("wb") as f:
                pickle.dump(phase_diagrams, f)

        return cls(phase_diagrams)

    def reaction_energy_per_atom(
        self,
        precursors: list[tuple[str, float]],   # [(formula, amount), ...]
        target_formula: str,
    ) -> Optional[float]:
        """
        Compute ΔE/atom in eV for the reaction:
            Σ a_i * P_i → 1 * target

        Returns None if any species is missing from MP entries or the
        chemsys has no PD cached. None means "can't evaluate", not zero.
        """
        try:
            target_comp = Composition(target_formula)
            target_els = {str(el) for el in target_comp.elements}

            # Union chemsys = elements in any precursor or the target
            all_els = set(target_els)
            for formula, _ in precursors:
                for el in Composition(formula).elements:
                    all_els.add(str(el))
            chemsys = "-".join(sorted(all_els))

            pd = self.phase_diagrams.get(chemsys)
            if pd is None:
                # Fall back: try a cached PD whose chemsys is a superset
                for cs, candidate_pd in self.phase_diagrams.items():
                    cs_els = set(cs.split("-"))
                    if all_els.issubset(cs_els):
                        pd = candidate_pd
                        break
            if pd is None:
                return None

            def get_entry_energy(formula: str) -> Optional[float]:
                """Total corrected energy (eV) for the given formula unit."""
                comp = Composition(formula)
                matching = [
                    e for e in pd.all_entries
                    if e.composition.reduced_formula == comp.reduced_formula
                ]
                if not matching:
                    return None
                best = min(matching, key=lambda e: e.energy_per_atom)
                # Scale to the requested formula amount
                scale = comp.num_atoms / best.composition.num_atoms
                return best.energy * scale

            target_energy = get_entry_energy(target_formula)
            if target_energy is None:
                return None

            precursor_energy_total = 0.0
            for formula, amount in precursors:
                e = get_entry_energy(formula)
                if e is None:
                    return None
                precursor_energy_total += amount * e

            # ΔE = E(target) - Σ a_i * E(precursor_i)
            # Normalized per atom in target product
            delta_E = target_energy - precursor_energy_total
            n_atoms_target = target_comp.num_atoms
            if n_atoms_target == 0:
                return None
            return delta_E / n_atoms_target

        except Exception:
            return None

    def __contains__(self, chemsys: str) -> bool:
        return chemsys in self.phase_diagrams

    def __len__(self) -> int:
        return len(self.phase_diagrams)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SynthesisValidator:
    """
    Scores a predicted synthesis route against physical/chemical constraints.

    Args:
        mp_formula_set: set of known MP formula strings (reduced form).
                        Used for precursor existence check.
                        Load from cached data/cache/mp_formula_set.pkl
        thermo_checker: optional ThermoChecker for thermodynamic favorability
                        check. If provided, the validator uses WEIGHTS_THERMO
                        and adds the rxn-energy constraint. If None, uses
                        WEIGHTS_LIGHT (5 deterministic checks, no network).
    """

    def __init__(
        self,
        mp_formula_set: set[str],
        thermo_checker: Optional[ThermoChecker] = None,
    ):
        # Normalize all formulas to reduced form on init (do once, not per call)
        self.mp_formula_set = {
            self._normalize_formula(f) for f in mp_formula_set
        }
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
        """
        Score a predicted route.

        Args:
            predicted: PredictedRoute from LLM output
            ground_truth_target_formula: optional — if provided, adds a
                target_match check (used during GRPO training).
                During inference this may not be available.

        Returns:
            (reward, breakdown) where reward ∈ [0,1] and breakdown
            maps constraint name → score for that constraint.
        """
        scores: dict[str, float] = {}

        scores["stoichiometry"]         = self._check_stoichiometry(predicted)
        scores["charge_neutrality"]     = self._check_charge_neutrality(predicted)
        scores["precursors_exist"]      = self._check_precursors_exist(predicted)
        scores["operation_order"]       = self._check_operation_order(predicted)
        scores["temperature_plausible"] = self._check_temperature(predicted)

        # Thermodynamic favorability — only when checker is configured.
        # When the checker can't evaluate (missing PD or entries), the
        # constraint receives a neutral 0.5 score so the model is neither
        # rewarded nor punished for cases we can't measure.
        if self.thermo_checker is not None:
            scores["thermodynamic_favorable"] = self._check_thermodynamics(
                predicted
            )

        # Optional target match (used in training, not inference)
        if ground_truth_target_formula is not None:
            scores["target_match"] = self._check_target_match(
                predicted, ground_truth_target_formula
            )

        # Use the appropriate weight set; ignore weights for absent scores.
        active_weights = {k: v for k, v in self.weights.items() if k in scores}
        # Renormalize active weights so reward stays in [0, 1] even if
        # the thermo check is configured but returned neutral for this route.
        # (target_match is not in self.weights so doesn't affect normalization.)
        weight_sum = sum(active_weights.values())
        if weight_sum == 0:
            return 0.0, scores
        reward = sum(
            (w / weight_sum) * scores[k] for k, w in active_weights.items()
        )
        return round(reward, 4), scores

    # ------------------------------------------------------------------
    # Individual constraint checks — all return float in [0, 1]
    # ------------------------------------------------------------------

    def _check_stoichiometry(self, predicted: PredictedRoute) -> float:
        """
        Verify mass balance: elements on precursor side must account for
        elements in target (allowing for volatile byproducts: CO2, H2O,
        NH3, O2, N2 — standard in solid-state synthesis).

        Strategy: compute precursor composition, subtract target composition,
        check that residual contains only known volatile species.
        """
        # Elements that exit as gases in solid-state synthesis:
        # C → CO2, H → H2O, N → NH3/N2
        # Oxygen bonded to these carriers also leaves — so we compute
        # how much O is "consumed" by volatile carriers and exempt it.
        VOLATILE_CARRIERS = {"C": 2.0, "H": 0.5, "N": 1.5}
        # C takes 2 O (CO2), H takes 0.5 O (H2O), N takes 1.5 O (NO avg — approx)

        try:
            # Sum precursor compositions weighted by stoichiometric amount
            precursor_comp: dict[str, float] = {}
            for p in predicted.precursors:
                comp = Composition(p.formula)
                for el, amt in comp.as_dict().items():
                    precursor_comp[el] = precursor_comp.get(el, 0.0) + amt * p.amount

            # Get target composition
            target_comp = Composition(predicted.target_formula).as_dict()

            # Check each element in target is covered by precursors
            for el, needed in target_comp.items():
                available = precursor_comp.get(el, 0.0)
                if available < needed * 0.9:   # 10% tolerance for rounding
                    return 0.0

            # Compute oxygen consumed by volatile carriers (leaves as CO2/H2O/NH3)
            o_consumed_by_volatiles = sum(
                precursor_comp.get(carrier, 0.0) * o_ratio
                for carrier, o_ratio in VOLATILE_CARRIERS.items()
            )

            # Check residual after accounting for target + volatile-bound oxygen
            for el, available in precursor_comp.items():
                needed = target_comp.get(el, 0.0)
                if el == "O":
                    # Oxygen can leave bonded to volatile carriers
                    effective_needed = needed + o_consumed_by_volatiles
                    excess = available - effective_needed
                elif el in VOLATILE_CARRIERS:
                    excess = 0.0   # volatile carrier — always ok to have
                else:
                    excess = available - needed

                if excess > 0.15:   # 15% tolerance for rounding/hydrates
                    return 0.0

            return 1.0

        except Exception:
            return 0.0

    def _check_charge_neutrality(self, predicted: PredictedRoute) -> float:
        """
        Check that the target formula is charge neutral using pymatgen's
        oxidation state guesser. Returns 1.0 if any valid assignment exists.

        This is the most forgiving check — if pymatgen can find ANY valid
        oxidation state assignment, we accept it.
        """
        try:
            comp = Composition(predicted.target_formula)

            # Try pymatgen's built-in oxi state guesser first
            try:
                oxi_comp = comp.add_charges_from_oxi_state_guesses(
                    max_sites=-1,
                    oxi_states_override=None
                )
                # If it returns without raising, a valid assignment exists
                return 1.0
            except Exception:
                pass

            # Fallback: manual check using COMMON_OX_STATES
            return self._manual_charge_check(comp)

        except Exception:
            return 0.0

    def _check_precursors_exist(self, predicted: PredictedRoute) -> float:
        """
        Fraction of predicted precursors that exist in the MP database.
        Returns mean score across all precursors (partial credit).

        E.g. if 2/3 precursors are known → 0.667
        """
        if not predicted.precursors:
            return 0.0

        scores = []
        for p in predicted.precursors:
            normalized = self._normalize_formula(p.formula)
            scores.append(1.0 if normalized in self.mp_formula_set else 0.0)

        return sum(scores) / len(scores)

    def _check_operation_order(self, predicted: PredictedRoute) -> float:
        """
        Verify operations are in a physically sensible order.
        Uses the OPERATION_ORDER rank dict — no operation should have
        a lower rank than the operation before it (monotone non-decreasing).

        Partial credit: fraction of consecutive pairs that are in order.
        """
        if not predicted.operations:
            return 0.0

        if len(predicted.operations) == 1:
            return 1.0

        ranks = []
        for op in predicted.operations:
            op_type = self._normalize_op_type(op.type)
            rank = OPERATION_ORDER.get(op_type, 99)
            ranks.append(rank)

        # Count consecutive pairs that are non-decreasing
        pairs = list(zip(ranks[:-1], ranks[1:]))
        ordered = sum(1 for a, b in pairs if a <= b)
        return ordered / len(pairs)

    def _check_temperature(self, predicted: PredictedRoute) -> float:
        """
        All specified temperatures must be within [TEMP_MIN, TEMP_MAX].
        If no temperatures specified (not uncommon in sparse data),
        give partial credit (0.5) rather than penalizing.

        Partial credit: fraction of temps in valid range.
        """
        temps = []
        for op in predicted.operations:
            temps.extend(op.conditions.heating_temperature)

        if not temps:
            return 0.5   # no temperature specified — not penalized but not rewarded

        in_range = [1.0 if TEMP_MIN <= t <= TEMP_MAX else 0.0 for t in temps]
        return sum(in_range) / len(in_range)

    def _check_thermodynamics(self, predicted: PredictedRoute) -> float:
        """
        Score the reaction Σ a_i * P_i → target by ΔE/atom (eV).
        Uses MP-corrected total energies via the cached PhaseDiagram.

        Scoring:
          ΔE ≤ -25 meV/atom        → 1.0  (clearly favorable)
          -25 < ΔE ≤ +25 meV/atom  → linear interpolation 1.0 → 0.5
          +25 < ΔE ≤ +150 meV/atom → linear interpolation 0.5 → 0.0
          ΔE > 150 meV/atom        → 0.0  (clearly unfavorable)

        Returns 0.5 (neutral) if the checker can't evaluate this reaction
        (missing PD or unknown precursors). Neutral means no signal — the
        score doesn't push the model in either direction.
        """
        if self.thermo_checker is None:
            return 0.5   # shouldn't be called, defensive

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
            # Linear: -25 meV → 1.0, +25 meV → 0.5
            t = (delta_E - RXN_ENERGY_FAVORABLE) / (
                RXN_ENERGY_BORDERLINE - RXN_ENERGY_FAVORABLE
            )
            return 1.0 - 0.5 * t
        elif delta_E <= RXN_ENERGY_UNFAVORABLE:
            # Linear: +25 meV → 0.5, +150 meV → 0.0
            t = (delta_E - RXN_ENERGY_BORDERLINE) / (
                RXN_ENERGY_UNFAVORABLE - RXN_ENERGY_BORDERLINE
            )
            return 0.5 - 0.5 * t
        else:
            return 0.0

    def _check_target_match(
        self,
        predicted: PredictedRoute,
        ground_truth_formula: str,
    ) -> float:
        """
        Check if predicted target formula matches ground truth.
        Uses reduced formula comparison (ignores stoichiometric scaling).
        Binary: 1.0 or 0.0.
        """
        try:
            pred = Composition(predicted.target_formula).reduced_formula
            true = Composition(ground_truth_formula).reduced_formula
            return 1.0 if pred == true else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_formula(formula: str) -> str:
        """
        Normalize a formula string to pymatgen reduced form.
        E.g. "Ba1Ti1O3" → "BaTiO3", "TiO2" → "TiO2"
        Returns original string if pymatgen can't parse it.
        """
        try:
            return Composition(formula).reduced_formula
        except Exception:
            return formula.strip()

    @staticmethod
    def _normalize_op_type(op_type: str) -> str:
        """
        Normalize operation type string to match OPERATION_ORDER keys.
        Handles both enum values and plain strings.
        """
        # Strip enum prefix if present (e.g. "OperationTypeEnum.heating")
        if "." in op_type:
            op_type = op_type.split(".")[-1]

        # Map common variants
        mapping = {
            "starting":         "StartingSynthesis",
            "startingsynthesis":"StartingSynthesis",
            "mixing":           "MixingOperation",
            "mixingoperation":  "MixingOperation",
            "drying":           "DryingOperation",
            "dryingoperation":  "DryingOperation",
            "heating":          "HeatingOperation",
            "heatingoperation": "HeatingOperation",
            "shaping":          "ShapingOperation",
            "shapingoperation": "ShapingOperation",
            "quenching":        "QuenchingOperation",
            "quenchingoperation":"QuenchingOperation",
            "cleaning":         "CleaningOperation",
            "cleaningoperation":"CleaningOperation",
        }
        return mapping.get(op_type.lower(), op_type)

    def _manual_charge_check(self, comp: Composition) -> float:
        """
        Fallback charge check using COMMON_OX_STATES.
        Returns 1.0 if any combination of valid oxidation states
        sums to zero, 0.0 otherwise.

        Only checks binary and ternary compositions (combinatorial
        explosion for larger systems — skip and give benefit of doubt).
        """
        elements = list(comp.as_dict().keys())

        if len(elements) > 3:
            return 0.5   # too complex to check manually — partial credit

        # Try all combinations of oxidation states
        def get_ox_states(el: str) -> list[int]:
            return COMMON_OX_STATES.get(el, [0])

        from itertools import product
        el_amounts = [(el, comp.as_dict()[el]) for el in elements]
        ox_options = [get_ox_states(el) for el, _ in el_amounts]

        for ox_combo in product(*ox_options):
            total = sum(
                ox * amt
                for (ox, (_, amt)) in zip(ox_combo, el_amounts)
            )
            if abs(total) < 0.1:
                return 1.0

        return 0.0


# ---------------------------------------------------------------------------
# Batch scoring (used in GRPO training loop)
# ---------------------------------------------------------------------------

def score_group(validator: SynthesisValidator, routes: list[PredictedRoute], ground_truth_formula: str) -> list[tuple[float, dict]]:
    """
    Score a group of G routes for one material.
    Used directly in GRPO: sample G routes, score all, compute advantages.

    Returns list of (reward, breakdown) in same order as input routes.
    """
    return [
        validator.validate(route, ground_truth_formula)
        for route in routes
    ]


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """
    Compute group-relative advantages for GRPO update.
    Advantage = (reward - group_mean) / (group_std + eps)

    Args:
        rewards: list of G reward values for one input

    Returns:
        list of G advantage values (zero-centered, unit variance)
    """
    import statistics

    if len(rewards) < 2:
        return [0.0] * len(rewards)

    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
    eps = 1e-8

    return [(r - mean) / (std + eps) for r in rewards]


# ---------------------------------------------------------------------------
# Test harness — run directly to verify validator works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a minimal fake MP formula set for testing
    # In production this comes from your cached data pull
    fake_mp_set = {
        "BaCO3", "TiO2", "BaTiO3",     # valid BaTiO3 synthesis
        "Gd2O3", "NH4Cl", "Er2O3",     # valid GdOCl synthesis (from your test)
        "SrCO3", "Fe2O3", "SrFeO3",    # valid SrFeO3 synthesis
    }

    validator = SynthesisValidator(fake_mp_set)

    print("=" * 60)
    print("TEST 1: Valid BaTiO3 synthesis route (expect reward ~1.0)")
    print("=" * 60)
    good_route = PredictedRoute(
        target_formula="BaTiO3",
        precursors=[
            PredictedPrecursor("BaCO3", amount=1.0),
            PredictedPrecursor("TiO2", amount=1.0),
        ],
        operations=[
            PredictedOperation(
                type="StartingSynthesis",
                conditions=PredictedConditions()
            ),
            PredictedOperation(
                type="MixingOperation",
                conditions=PredictedConditions(mixing_media="ethanol")
            ),
            PredictedOperation(
                type="HeatingOperation",
                conditions=PredictedConditions(
                    heating_temperature=[1200.0],
                    heating_time=[4.0],
                    heating_atmosphere=["air"]
                )
            ),
        ]
    )
    reward, breakdown = validator.validate(good_route, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:25s}: {v:.3f}")

    print()
    print("=" * 60)
    print("TEST 2: Broken stoichiometry (expect stoichiometry=0.0)")
    print("=" * 60)
    bad_stoich = PredictedRoute(
        target_formula="BaTiO3",
        precursors=[
            PredictedPrecursor("BaCO3", amount=1.0),
            # Missing TiO2 — Ba but no Ti
        ],
        operations=[
            PredictedOperation(type="MixingOperation",
                               conditions=PredictedConditions()),
            PredictedOperation(
                type="HeatingOperation",
                conditions=PredictedConditions(heating_temperature=[1200.0])
            ),
        ]
    )
    reward, breakdown = validator.validate(bad_stoich, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:25s}: {v:.3f}")

    print()
    print("=" * 60)
    print("TEST 3: Unknown precursor (expect precursors_exist < 1.0)")
    print("=" * 60)
    unknown_precursor = PredictedRoute(
        target_formula="BaTiO3",
        precursors=[
            PredictedPrecursor("BaCO3", amount=1.0),
            PredictedPrecursor("TiO2", amount=1.0),
            PredictedPrecursor("MagicChemical123", amount=0.1),  # fake
        ],
        operations=[
            PredictedOperation(type="StartingSynthesis",
                               conditions=PredictedConditions()),
            PredictedOperation(
                type="HeatingOperation",
                conditions=PredictedConditions(heating_temperature=[1200.0])
            ),
        ]
    )
    reward, breakdown = validator.validate(unknown_precursor, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:25s}: {v:.3f}")

    print()
    print("=" * 60)
    print("TEST 4: Wrong operation order — heating before mixing")
    print("=" * 60)
    wrong_order = PredictedRoute(
        target_formula="BaTiO3",
        precursors=[
            PredictedPrecursor("BaCO3", amount=1.0),
            PredictedPrecursor("TiO2", amount=1.0),
        ],
        operations=[
            PredictedOperation(
                type="HeatingOperation",   # heating FIRST — wrong
                conditions=PredictedConditions(heating_temperature=[1200.0])
            ),
            PredictedOperation(type="MixingOperation",
                               conditions=PredictedConditions()),
        ]
    )
    reward, breakdown = validator.validate(wrong_order, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:25s}: {v:.3f}")

    print()
    print("=" * 60)
    print("TEST 5: Impossible temperature (5000°C)")
    print("=" * 60)
    bad_temp = PredictedRoute(
        target_formula="BaTiO3",
        precursors=[
            PredictedPrecursor("BaCO3", amount=1.0),
            PredictedPrecursor("TiO2", amount=1.0),
        ],
        operations=[
            PredictedOperation(type="StartingSynthesis",
                               conditions=PredictedConditions()),
            PredictedOperation(
                type="HeatingOperation",
                conditions=PredictedConditions(
                    heating_temperature=[5000.0]   # impossible
                )
            ),
        ]
    )
    reward, breakdown = validator.validate(bad_temp, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:25s}: {v:.3f}")

    print()
    print("=" * 60)
    print("TEST 6: GRPO group scoring (G=4 routes)")
    print("=" * 60)
    group = [good_route, bad_stoich, unknown_precursor, wrong_order]
    results = score_group(validator, group, "BaTiO3")
    rewards = [r for r, _ in results]
    advantages = compute_grpo_advantages(rewards)
    print(f"Rewards:    {[round(r, 3) for r in rewards]}")
    print(f"Advantages: {[round(a, 3) for a in advantages]}")
    print()
    print("Higher advantage → route gets positive gradient update")
    print("Lower advantage  → route gets negative gradient update")

    # ----------------------------------------------------------------
    # Thermo-aware mode tests using a MOCK ThermoChecker
    # (real one needs the MP API; we mock the rxn_energy method here)
    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    print("THERMO MODE TESTS (using mock ThermoChecker)")
    print("=" * 60)

    class MockThermoChecker(ThermoChecker):
        """Returns hardcoded ΔE values for tests."""
        def __init__(self, rxn_table: dict):
            super().__init__(phase_diagrams={})
            self.rxn_table = rxn_table

        def reaction_energy_per_atom(self, precursors, target_formula):
            key = (
                tuple(sorted((f, a) for f, a in precursors)),
                target_formula,
            )
            return self.rxn_table.get(key)

    mock_thermo = MockThermoChecker({
        # Favorable (-50 meV/atom) → score 1.0
        (
            tuple(sorted([("BaCO3", 1.0), ("TiO2", 1.0)])),
            "BaTiO3",
        ): -0.050,
        # Borderline (+10 meV/atom) → ~0.65
        (
            tuple(sorted([("BaO", 1.0), ("TiO2", 1.0)])),
            "BaTiO3",
        ):  0.010,
        # Unfavorable (+200 meV/atom) → 0.0
        (
            tuple(sorted([("BaO2", 1.0), ("Ti", 1.0)])),
            "BaTiO3",
        ):  0.200,
    })
    thermo_validator = SynthesisValidator(
        fake_mp_set | {"BaO", "BaO2", "Ti"},   # add precursors to set
        thermo_checker=mock_thermo,
    )

    print()
    print("TEST 7: Thermodynamically favorable route (ΔE = -50 meV/atom)")
    print("-" * 60)
    reward, breakdown = thermo_validator.validate(good_route, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:30s}: {v:.3f}")

    print()
    print("TEST 8: Borderline reaction (ΔE = +10 meV/atom)")
    print("-" * 60)
    borderline_route = PredictedRoute(
        target_formula="BaTiO3",
        precursors=[
            PredictedPrecursor("BaO", amount=1.0),
            PredictedPrecursor("TiO2", amount=1.0),
        ],
        operations=[
            PredictedOperation(type="StartingSynthesis",
                               conditions=PredictedConditions()),
            PredictedOperation(
                type="HeatingOperation",
                conditions=PredictedConditions(heating_temperature=[1200.0])
            ),
        ]
    )
    reward, breakdown = thermo_validator.validate(borderline_route, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:30s}: {v:.3f}")

    print()
    print("TEST 9: Thermodynamically unfavorable (ΔE = +200 meV/atom)")
    print("-" * 60)
    unfavorable_route = PredictedRoute(
        target_formula="BaTiO3",
        precursors=[
            PredictedPrecursor("BaO2", amount=1.0),
            PredictedPrecursor("Ti", amount=1.0),
        ],
        operations=[
            PredictedOperation(type="StartingSynthesis",
                               conditions=PredictedConditions()),
            PredictedOperation(
                type="HeatingOperation",
                conditions=PredictedConditions(heating_temperature=[1200.0])
            ),
        ]
    )
    reward, breakdown = thermo_validator.validate(unfavorable_route, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:30s}: {v:.3f}")

    print()
    print("TEST 10: Unknown reaction (not in mock table) → neutral 0.5")
    print("-" * 60)
    unknown_route = PredictedRoute(
        target_formula="BaTiO3",
        precursors=[
            PredictedPrecursor("BaCO3", amount=1.0),
            PredictedPrecursor("TiO2", amount=2.0),    # different stoich
        ],
        operations=[
            PredictedOperation(type="StartingSynthesis",
                               conditions=PredictedConditions()),
            PredictedOperation(
                type="HeatingOperation",
                conditions=PredictedConditions(heating_temperature=[1200.0])
            ),
        ]
    )
    reward, breakdown = thermo_validator.validate(unknown_route, "BaTiO3")
    print(f"Reward: {reward}")
    for k, v in breakdown.items():
        print(f"  {k:30s}: {v:.3f}")
    print()
    print("Note: thermodynamic_favorable=0.5 = neutral (PD lookup failed)")