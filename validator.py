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
  _check_amount_accuracy (NEW)  → same Reaction object's solved get_coeff()
                                  values, compared against the model's
                                  predicted precursor amounts. Added 2026-07:
                                  .amount was previously read once and
                                  discarded, so the reward had zero signal
                                  on whether predicted quantities were
                                  correct, only on precursor SPECIES choice.
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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, TYPE_CHECKING
from rich import print as rprint

from pymatgen.core import Composition, Element
from pymatgen.analysis.reaction_calculator import (
    Reaction,
    ComputedReaction,
    ReactionError,
)

if TYPE_CHECKING:
    from pymatgen.analysis.phase_diagram import PhaseDiagram


# ---------------------------------------------------------------------------
# Data classes for predicted route
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


def expand_hydrate_notation(formula: str) -> str:
    """
    pymatgen's Composition() cannot parse the standard middle-dot hydrate
    notation "A·nB" (confirmed 2026-07: Composition("FeC2O4·2H2O") raises
    "·2 is an invalid formula!"). This is common, standard synthesis
    notation - FeC2O4·2H2O (iron oxalate dihydrate) is the conventional
    Fe(II) precursor for LiFePO4 specifically, chosen to keep iron from
    oxidizing to Fe(III) during synthesis. Found via a real LiFePO4
    record: _resolve_pd hit this precursor and aborted the ENTIRE
    chemsys resolution with zero PD candidates attempted - even though
    the target and every other precursor parsed fine and the needed PD
    shard was confirmed present and loadable. This one unhandled
    notation is the leading suspect for why LFP-family sat at 82%
    ungradeable despite the target chemistry itself being unremarkable.

    Rewrites "A·nB" -> "A(B)n", parenthetical-multiplier notation
    pymatgen DOES support natively (confirmed against Ca9Y(PO4)7,
    Li3V2(PO4)3C, and FeC2O4(H2O)2 - the last one verified to produce
    exactly {Fe:1, C:2, O:6, H:4}, matching the real chemistry).

    Only handles the CONFIRMED "·" (U+00B7 MIDDLE DOT) character -
    deliberately not guessing at other hydrate notations ("." with
    spaces, "*", unspecified "xH2O") without evidence they occur in
    this corpus too. Formulas without "·" pass through unchanged.
    """
    if "\u00b7" not in formula:
        return formula
    parts = formula.split("\u00b7")
    if len(parts) != 2:
        return formula  # more than one dot - don't guess, leave as-is
    base, hydrate = parts
    m = re.match(r"^(\d*)(.+)$", hydrate.strip())
    if not m:
        return formula
    count_str, hydrate_formula = m.groups()
    count = count_str if count_str else "1"
    return f"{base}({hydrate_formula}){count}"


@dataclass
class PredictedPrecursor:
    formula: str
    amount: float = 1.0

    def __post_init__(self):
        self.formula = expand_hydrate_notation(self.formula)


@dataclass
class PredictedRoute:
    target_formula: str
    precursors: list[PredictedPrecursor]
    operations: list[PredictedOperation]
    reaction_string: str = ""

    def __post_init__(self):
        self.target_formula = expand_hydrate_notation(self.target_formula)


# ---------------------------------------------------------------------------
# Constraint weights
# ---------------------------------------------------------------------------

# Lightweight mode (no thermodynamics)
# NOTE: stoichiometry's original 0.35 is split in half with the new
# amount_accuracy check (added 2026-07) rather than diluting every other
# weight proportionally. These are two sub-questions of the same
# underlying "is the recipe stoichiometrically sound" concept (does a
# balance exist vs. is the predicted ratio close to it), so splitting the
# existing budget between them is the natural starting allocation - not
# touching the other four checks' relative weights against each other.
WEIGHTS_LIGHT = {
    "stoichiometry":         0.175,
    "amount_accuracy":       0.175,
    "charge_neutrality":     0.25,
    "precursors_exist":      0.20,
    "operation_order":       0.10,
    "temperature_plausible": 0.10,
}

# Thermo-aware mode — nine checks, all physics-grounded.
# Mass balance + reaction energy + target hull stability + atmosphere chemistry
# carry the most weight because they're the load-bearing chemistry signals.
WEIGHTS_THERMO = {
    "stoichiometry":            0.10,   # Reaction balances exactly?
    "amount_accuracy":          0.10,   # Predicted ratio vs. solved ratio
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
    # Alkali (+1)
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    # Alkaline earth (+2)
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    # Group 13 (+3)
    "Al": 3, "Ga": 3, "In": 3, "Sc": 3, "Y": 3,
    # p-block oxide formers — the bug fix. Without these, the fallback
    # to Element(el).oxidation_states[0] picks the LOWEST positive state,
    # which for B/Si/P/Ge/As/Sb gives +1/+2 instead of the +3/+4/+5
    # they take in oxides. That mis-computation cascaded into wildly
    # wrong redox-cation requirements and 0.0 charge_neutrality scores
    # on every solid-solution borate/silicate/phosphate.
    "B":  3, "Si": 4, "P":  5, "Ge": 4, "As": 5, "Sb": 5,
    "Te": 4,  # in oxide environments — Te6+ also possible but rarer
    # Rare earths (+3, with the exception of Ce/Eu/Pr/Tb which are in REDOX_METALS)
    "La": 3, "Nd": 3, "Sm": 3, "Gd": 3, "Dy": 3,
    "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    # Transition metals with fixed high valences (the variable ones live in REDOX_METALS)
    "Zr": 4, "Hf": 4, "Ta": 5, "Nb": 5,
}


# ---------------------------------------------------------------------------
# Thermodynamic favorability thresholds (eV/atom)
# These thresholds apply to ΔG_rxn(T_synthesis) computed by gibbs_corrector.py
# (Bartel descriptor for solids + NIST-JANAF tabulated values for gases).
# Calibrated against the ~50 meV/atom MAE of the Bartel descriptor, so the
# ±0.025 eV/atom band around zero is treated as noise.
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

# Amount-accuracy thresholds. UNLIKE the thermo thresholds above, these are
# NOT derived from a physical noise floor - they're a starting design
# choice, chosen to tolerate the modest deliberate excess real recipes use
# (e.g. 5-20% extra alkali carbonate to compensate for volatilization
# losses, per real corpus reasoning traces - "we might use excess K2CO3
# due to volatility at high temps"). Validate/retune empirically against
# real predicted-vs-solved distributions before trusting these numbers the
# way the physically-grounded thermo thresholds can be trusted.
# Metric: sum(|predicted_i - solved_i|) / sum(solved_i) - a stoichiometric-
# mass-weighted mean relative error. Weighting by the solved coefficient
# itself means a large relative error on a small-coefficient dopant
# precursor (e.g. Eu2O3 in Sr0.99Eu0.01B2O4) barely moves the score, since
# its absolute contribution to the deviation is small - this matters for
# THIS corpus specifically, given ~40% of targets are doped solid
# solutions with genuinely minor dopant-source precursors.
AMOUNT_ERROR_TIGHT = 0.10   # within 10% of total stoichiometric mass → 1.0
AMOUNT_ERROR_LOOSE = 0.30   # up to 30% → linear decay 1.0 → 0.5
AMOUNT_ERROR_BAD   = 0.75   # up to 75% → linear decay 0.5 → 0.0; beyond → 0.0


# ---------------------------------------------------------------------------
# Atmosphere classification (used by chempot_atmosphere check)
# ---------------------------------------------------------------------------

ATMOSPHERE_OXIDIZING = frozenset({"air", "o2", "oxygen", "h2o", "steam", "water"})
ATMOSPHERE_REDUCING  = frozenset({"h2", "hydrogen", "co", "forming gas", "formgas"})
ATMOSPHERE_INERT     = frozenset({"ar", "argon", "n2", "nitrogen", "vacuum", "he", "helium"})


def _classify_atmosphere(atm: str) -> str:
    """Return one of: 'ox', 'red', 'inert', 'unknown'.

    Token-based matching on alphanumeric runs — NOT substring matching.
    The previous `in` checks misclassified real strings: "h2o" matched
    "h2" (steam flagged REDUCING, which is backwards — steam oxidizes
    metals at temperature, e.g. Fe + H2O -> FeO + H2), "co2" matched
    "o2", "cold"/"controlled" matched "co", and "carbon" matched "ar"
    (carbon is a reductant, not an inert gas). h2o/steam/water are
    classified oxidizing accordingly.
    """
    a = atm.lower().strip()
    # Multiword phrases whose individual tokens would lose their meaning.
    if "forming gas" in a or "formgas" in a:
        return "red"
    tokens = {t for t in re.split(r"[^a-z0-9]+", a) if t}
    if tokens & ATMOSPHERE_OXIDIZING:
        return "ox"
    if tokens & ATMOSPHERE_REDUCING:
        return "red"
    if tokens & ATMOSPHERE_INERT:
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
            # Deterministic candidate order: smallest covering superset first
            # (fewer extraneous competing phases distorting the hull),
            # alphabetical tiebreak. Previously iterated raw dict order, so
            # the chosen PD depended on the index file's key ordering and
            # could change across index rebuilds.
            candidates = sorted(
                (cs for cs in self.pd_index if all_els.issubset(set(cs.split("-")))),
                key=lambda c: (c.count("-"), c),
            )
            for cs in candidates:
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
        predicted_route=None,
    ) -> tuple[Optional[float], str]:
        """
        Compute per-atom reaction energy using pymatgen's ComputedReaction.

        If `predicted_route` is provided, computes ΔG_rxn at the synthesis
        temperature parsed from the route's heating operations, using
        gibbs_corrector.py (Bartel descriptor for solids + NIST-JANAF tabulated
        values for gases). This is the Tier 3.1 codepath and is what the
        validator should always use in production.

        If the target has no discrete PD entry (doped / non-stoichiometric
        solid solution - the dominant cause of sentinel scores, confirmed
        2026-07: ~98% of previously-flat records fail at exactly this step
        despite a fully covered, uncorrupted PD), falls back to
        gibbs_corrector.compute_reaction_gibbs_per_atom_interpolated, which
        represents the target via its convex-hull decomposition instead of
        a literal entry. Backtested against 400 records with a real discrete
        entry: median residual 0.0, 95.3% within MP's own 50 meV/atom
        reaction-energy noise floor. See that function's docstring for the
        known systematic bias (mild upper bound on the true driving force).

        If `predicted_route` is None, falls back to a 0K ΔE calculation. This
        is a diagnostic-only path; raw 0K ΔE is systematically endothermic for
        gas-releasing reactions (carbonates, hydroxides) and shouldn't be
        scored against the same thresholds as ΔG.

        Returns
        -------
        (value, gradeability) where gradeability is one of:
          "discrete"     - target had a literal PD entry; Tier 3.1 as before.
          "interpolated" - target had no entry; hull-decomposition fallback
                            was used. Caller should calibrate/weight this
                            differently from "discrete" (see bias note above).
          "ungradeable"  - neither method produced a value (value is None):
                            no covering PD, missing precursor entry, or the
                            reaction genuinely doesn't balance. This is a
                            real "can't grade" case, distinct from either
                            success path.
        """
        try:
            # CRITICAL: use ONLY target + precursors for the chemsys lookup.
            # Do NOT include VOLATILE_FORMULAS here. Including CO2/H2O/NH3/N2
            # would add C, H, N to every query — turning a lookup for
            # "C-O-Sr-Ti" (which data_pull built) into "C-H-N-O-Sr-Ti"
            # (which was never built). The PD for (target+precursors) already
            # contains all volatile sub-system entries (O2, CO2, etc.) because
            # pymatgen's PhaseDiagram includes every entry across the full
            # element set.
            core_formulas = [target_formula] + [f for f, _ in precursors]
            pd, _ = self._resolve_pd(core_formulas)
            if pd is None:
                return None, "ungradeable"

            # Tier 3.1: Gibbs-corrected reaction energy at synthesis T
            if predicted_route is not None:
                from gibbs_corrector import (
                    compute_reaction_gibbs_per_atom,
                    compute_reaction_gibbs_per_atom_interpolated,
                )
                precursor_formulas = [f for f, _ in precursors]
                delta_G, _T_K = compute_reaction_gibbs_per_atom(
                    target_formula, precursor_formulas, pd, predicted_route
                )
                if delta_G is not None:
                    return delta_G, "discrete"

                delta_G_interp, _T_K = compute_reaction_gibbs_per_atom_interpolated(
                    target_formula, precursor_formulas, pd, predicted_route
                )
                if delta_G_interp is not None:
                    return delta_G_interp, "interpolated"
                return None, "ungradeable"

            # Legacy 0K path (diagnostic only — not for production scoring)
            target_entry = self._best_entry_for_formula(pd, target_formula)
            if target_entry is None:
                return None, "ungradeable"

            reactant_entries = []
            for formula, _ in precursors:
                e = self._best_entry_for_formula(pd, formula)
                if e is None:
                    return None, "ungradeable"
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
                return None, "ungradeable"

            # ComputedReaction normalizes internally to reduced compositions.
            # Target entry may be the unreduced form (Sr2Ti2O6 for mp-4651
            # SrTiO3), so query with reduced composition or get_coeff raises.
            target_comp_reduced = Composition(
                target_entry.composition.reduced_formula
            )
            try:
                target_coeff = reaction.get_coeff(target_comp_reduced)
            except (ValueError, KeyError):
                return None, "ungradeable"
            if target_coeff <= 1e-6:
                return None, "ungradeable"

            delta_E_total = reaction.calculated_reaction_energy  # eV
            atoms_target = target_coeff * target_comp_reduced.num_atoms
            return delta_E_total / atoms_target, "discrete"

        except Exception:
            return None, "ungradeable"

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

    # Gradeability tags that mean "this check could not be computed" — as
    # opposed to a genuinely computed mid-band 0.5. validate() excludes
    # checks carrying one of these from the reward entirely (see there).
    SENTINEL_TAGS = frozenset({
        "ungradeable", "sentinel_no_entry", "no_balance_found", "no_thermo_checker",
    })

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def val_debug(self, predicted: PredictedRoute, ground_truth_target_formula: Optional[str] = None):


        reactants = [Composition(p.formula) for p in predicted.precursors]
        if not reactants:
            print("no reactants")
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
        #     try:
        #         reaction = Reaction(reactants, products)
        #     except ReactionError:
        #         continue
        #     # Target must be produced with positive coefficient
        #     target_coeff = reaction.get_coeff(target_comp)
        #     if target_coeff <= 1e-6:
        #         continue
        #     # All precursors must actually be consumed (negative coeff)
        #     all_used = all(
        #         reaction.get_coeff(r) < -1e-6 for r in reactants
        #     )
        #     if all_used:
        #         return 1.0




    def validate(
        self,
        predicted: PredictedRoute,
        ground_truth_target_formula: Optional[str] = None,
    ) -> tuple[float, dict[str, float | str]]:
        # NOTE: scores is float-valued for every check EXCEPT the three
        # "_gradeability"-suffixed sibling keys added below, which are
        # strings ("discrete" / "interpolated" / "ungradeable" / etc).
        # This is deliberate and additive: active_weights below only pulls
        # keys that also exist in self.weights, and no "_gradeability" key
        # is ever a weight name, so the reward computation is unaffected.
        # External callers that unpack validate() as (reward, breakdown)
        # and do dict lookups by check name see no change; only code that
        # blindly iterated breakdown.values() assuming all-float would
        # break, and nothing in this codebase currently does that.
        scores: dict[str, float | str] = {}

        scores["stoichiometry"]         = self._check_stoichiometry(predicted)
        scores["amount_accuracy"], scores["amount_accuracy_gradeability"] = \
            self._check_amount_accuracy(predicted)
        scores["charge_neutrality"]     = self._check_charge_neutrality(predicted)
        scores["precursors_exist"]      = self._check_precursors_exist(predicted)
        scores["operation_order"]       = self._check_operation_order(predicted)
        scores["temperature_plausible"] = self._check_temperature(predicted)

        if self.thermo_checker is not None:
            scores["thermodynamic_favorable"], scores["thermodynamic_favorable_gradeability"] = \
                self._check_thermodynamics(predicted)
            scores["target_stability"], scores["target_stability_gradeability"] = \
                self._check_target_stability(predicted)
            scores["chempot_atmosphere"], scores["chempot_atmosphere_gradeability"] = \
                self._check_chempot_atmosphere(predicted)

        if ground_truth_target_formula is not None:
            scores["target_match"] = self._check_target_match(
                predicted, ground_truth_target_formula
            )

        # None-propagation (Route C): checks whose gradeability tag marks a
        # can't-compute state are EXCLUDED from the reward and the remaining
        # weights are renormalized. Previously the 0.5 sentinel was paid at
        # full weight — free credit on eval (up to ~0.225 in thermo mode
        # when all three thermo checks are ungradeable) and constant,
        # advantage-zero mass inside every GRPO group on that target.
        # Genuinely computed mid-band 0.5 scores (tag "discrete"/
        # "interpolated") are unaffected.
        active_weights = {}
        for k, w in self.weights.items():
            if k not in scores:
                continue
            if scores.get(f"{k}_gradeability") in self.SENTINEL_TAGS:
                continue
            active_weights[k] = w
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

    def _find_balanced_reaction(self, predicted: PredictedRoute):
        """
        Shared balance-finding logic for _check_stoichiometry and
        _check_amount_accuracy - both need "does a balanced reaction
        exist, and if so, what are its solved coefficients", and must
        agree on which reaction that is (extracted here so they can't
        silently diverge into disagreeing about the same route).

        Returns (reaction, reactants). reaction is None if no balance was
        found across any volatile-set candidate. reactants is returned
        even on failure so callers can still report/use the reactant
        Compositions without reconstructing them.
        """
        reactants = [Composition(p.formula) for p in predicted.precursors]
        if not reactants:
            return None, reactants
        target_comp = Composition(predicted.target_formula)

        # Try with progressively more volatile candidates. Some routes
        # don't release any gases (e.g., oxide + oxide → oxide); Reaction
        # is happier with the minimum set that lets it balance.
        candidate_volatile_sets = [
            [],                          # no volatiles
            ["CO2"],                     # carbonate routes
            ["H2O"],                     # hydrate routes
            ["O2"],                      # redox routes
            ["CO2", "H2O", "O2"],        # full common set
            VOLATILE_FORMULAS,           # everything
        ]

        for volatile_strs in candidate_volatile_sets:
            volatile_set = [Composition(v) for v in volatile_strs]
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
            if not all_used:
                continue
            # pymatgen's null-space balance puts a volatile on whichever
            # side closes the equation — including the REACTANT side, i.e.
            # the balance can silently assume gas uptake (FeO -> Fe2O3
            # balanced as consuming O2 the route never declared). Only
            # accept such a balance if the route declares an atmosphere
            # that can actually supply the consumed gas.
            consumed = [s for s, c in zip(volatile_strs, volatile_set)
                        if reaction.get_coeff(c) < -1e-6]
            if consumed and not self._volatiles_supplied(consumed, predicted):
                continue
            return reaction, reactants

        return None, reactants

    @staticmethod
    def _volatiles_supplied(consumed: list[str], predicted: PredictedRoute) -> bool:
        """
        True iff every volatile the balance CONSUMES has a declared supplier
        in the route's operation atmospheres (e.g. O2 uptake requires an
        air/O2/oxidizing atmosphere; N2 uptake a nitrogen atmosphere).
        Atmospheres declared on any operation count — quench-step
        atmospheres are as real as furnace atmospheres for this purpose.
        """
        SUPPLIERS = {
            "O2":  {"o2", "oxygen", "air"},
            "H2O": {"h2o", "steam", "water"},
            "CO2": {"co2"},
            "N2":  {"n2", "nitrogen"},
            "NH3": {"nh3", "ammonia"},
        }
        tokens: set[str] = set()
        for op in predicted.operations:
            for atm in op.conditions.heating_atmosphere:
                tokens.update(t for t in re.split(r"[^a-z0-9]+", atm.lower()) if t)
        for v in consumed:
            if not (tokens & SUPPLIERS.get(v, set())):
                return False
        return True

    def _check_stoichiometry(self, predicted: PredictedRoute) -> float:
        """
        Mass balance via pymatgen.analysis.reaction_calculator.Reaction.

        Reaction performs exact null-space balancing on the elemental
        composition matrix. If a balanced equation exists with the target
        on the product side and the precursors on the reactant side
        (allowing CO2/H2O/O2/N2/NH3 as volatile byproducts), score is 1.0.
        Otherwise 0.0.

        Binary because the underlying mathematical question is binary:
        either a balance exists or it doesn't. See _check_amount_accuracy
        for whether the model's PREDICTED amounts match that balance -
        this check only asks whether some balance exists at all.
        """
        try:
            reaction, reactants = self._find_balanced_reaction(predicted)
            if not reactants:
                return 0.0
            return 1.0 if reaction is not None else 0.0
        except Exception:
            return 0.0

    def _check_amount_accuracy(self, predicted: PredictedRoute) -> tuple[float, str]:
        """
        Compares the model's PREDICTED precursor amounts against the
        solved stoichiometric coefficients pymatgen's Reaction computes -
        the same balance _check_stoichiometry finds and then discards.

        Added 2026-07 after discovering .amount was read in exactly one
        place in the entire validator (packed into a tuple, then
        immediately discarded before reaching any computation) - meaning
        the reward previously had ZERO signal on whether predicted
        amounts were correct, only on whether the right precursor SPECIES
        were chosen. This check closes that gap.

        SCALE-INVARIANCE BUG, found and fixed 2026-07: the first version
        of this check assumed the model always states amounts normalized
        to "produce exactly 1 mole of target" (true in the one real
        example it was validated against, KLaNb2O7, by coincidence - the
        model happened to solve its balance for 1 mole there). Reaction's
        solved coefficients ARE normalized that way (target_coeff always
        == 1.0 after the /target_coeff step below), but the model has no
        reason to pick that same batch size - "combine 1 mole La2O3 + 1
        mole In2O3" (-> 2 mole LaInO3) is an entirely standard, correct
        way to state a recipe, and compared unscaled against solved
        [0.5, 0.5] it scored as maximally wrong. A full-corpus rescore
        caught this: the overwhelming majority of amount_accuracy=0.0
        records showed a CONSISTENT scale factor between predicted and
        solved (not scattered ratio errors), across formulas as mundane
        as LaInO3, LiTaO3, FeNbO4. Fixed by finding the best-fit uniform
        scale factor (ratio of totals) and rescaling predicted amounts by
        it before computing distance - this is now comparing RATIOS, not
        absolute batch sizes, which is the chemically meaningful quantity.
        Verified against real flagged records post-fix: pure scale
        mismatches (LaInO3, CaMg0.8Al0.4Si1.8O6, Na2Mo2O7) now resolve to
        ~0 error; genuine ratio errors (Y0.99Dy0.01(P0.8V0.2)O4 at 27%,
        Ga5Ge20Sb10S65 at 48%) correctly retain substantial error - the
        fix discriminates rather than just zeroing everything out.

        Units: Reaction's solved coefficients are normalized by the
        target's own solved coefficient (putting them on a "per 1 mole
        target" basis) as before - that part was always correct. What's
        new is normalizing the PREDICTED side by its own best-fit scale
        before comparing, so the comparison is scale-free on both sides.

        Metric: sum(|scaled_predicted_i - solved_i|) / sum(solved_i) -
        total absolute deviation normalized by total reference
        stoichiometric mass, AFTER removing the model's chosen batch
        size. Still a stoichiometric-mass-WEIGHTED error: a large
        relative error on a small-coefficient dopant precursor (e.g.
        Eu2O3 in Sr0.99Eu0.01B2O4, solved coeff ~0.01) barely moves the
        score, matching the actual chemistry - getting a trace dopant's
        amount somewhat wrong matters far less than botching a major
        reagent.

        Returns (score, gradeability):
          "discrete"         - a balance was found and amounts compared
          "no_balance_found" - _find_balanced_reaction found nothing (0.5,
                                mirrors _check_stoichiometry's 0.0 failure
                                mode but as a sentinel rather than a
                                penalty, since this check's JOB is amount
                                comparison, not balance-existence - that
                                failure is already penalized via the
                                separate stoichiometry weight)
          "no_precursors"    - route has no precursors at all
          "ungradeable"      - unexpected exception, or degenerate solved/
                                predicted amounts (e.g. all-zero predicted
                                amounts, making the scale factor undefined)

        AMOUNT_ERROR_TIGHT/LOOSE/BAD thresholds are a starting design
        choice, not physically derived - see their definition comment.
        Worth re-examining empirically now that the metric measures real
        ratio error instead of being dominated by scale-mismatch noise.
        """
        try:
            reaction, reactants = self._find_balanced_reaction(predicted)
            if not reactants:
                return 0.0, "no_precursors"
            if reaction is None:
                return 0.5, "no_balance_found"

            target_comp = Composition(predicted.target_formula)
            target_coeff = reaction.get_coeff(target_comp)

            solved = [abs(reaction.get_coeff(c)) / target_coeff for c in reactants]
            predicted_amounts = [p.amount for p in predicted.precursors]

            sum_solved = sum(solved)
            sum_predicted = sum(predicted_amounts)
            if sum_solved < 1e-9 or sum_predicted < 1e-9:
                return 0.5, "ungradeable"

            # Best-fit uniform scale factor: what batch size did the model
            # implicitly choose, relative to the solved "1 mole target"
            # convention? Rescale predicted amounts by it before measuring
            # distance, so the comparison is between RATIOS, not batch sizes.
            scale = sum_predicted / sum_solved
            scaled_predicted = [pa / scale for pa in predicted_amounts]

            total_abs_dev = sum(
                abs(sp - s) for sp, s in zip(scaled_predicted, solved)
            )
            weighted_error = total_abs_dev / sum_solved

            if weighted_error <= AMOUNT_ERROR_TIGHT:
                return 1.0, "discrete"
            elif weighted_error <= AMOUNT_ERROR_LOOSE:
                t = (weighted_error - AMOUNT_ERROR_TIGHT) / (AMOUNT_ERROR_LOOSE - AMOUNT_ERROR_TIGHT)
                return 1.0 - 0.5 * t, "discrete"
            elif weighted_error <= AMOUNT_ERROR_BAD:
                t = (weighted_error - AMOUNT_ERROR_LOOSE) / (AMOUNT_ERROR_BAD - AMOUNT_ERROR_LOOSE)
                return 0.5 - 0.5 * t, "discrete"
            else:
                return 0.0, "discrete"

        except Exception:
            return 0.5, "ungradeable"

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
                # Unknown non-redox cation — fall back to the HIGHEST positive
                # oxidation state. These cations are in oxides (or other
                # anionic compounds), so they're oxidized, not reduced. Picking
                # states[0] (the lowest) here was the LiB/Si/P/Ge bug:
                # gave B = +1 instead of +3 in borates.
                try:
                    positive_states = [s for s in Element(el).oxidation_states if s > 0]
                    if not positive_states:
                        return 0.5
                    fixed_charge += max(positive_states) * amt
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
        assessed = 0
        for i, (a, b) in enumerate(pairs):
            if a == 99 or b == 99:
                # Unrecognized op type — ordering of this pair can't be
                # assessed. Previously such pairs silently passed, so a
                # route of entirely unknown op types scored a perfect 1.0.
                continue
            assessed += 1
            if a > b:
                # Rank dropped. Check if it's a heating→mixing regrind cycle
                # where another heating step follows later.
                if a == HEATING_RANK and b <= MIXING_RANK + 1:
                    if any(r >= HEATING_RANK for r in ranks[i + 2:]):
                        continue   # legitimate regrind, not a violation
                violations += 1
        if assessed == 0:
            return 0.5  # no recognizable op types — neutral, not perfect
        return 1.0 - violations / assessed

    def _check_temperature(self, predicted: PredictedRoute) -> float:
        """
        Fraction of HEATING-step temperatures in [TEMP_MIN, TEMP_MAX].

        Only counts operations whose normalized type is "HeatingOperation"
        (calcine, sinter, anneal, etc.). Excludes:
          - MixingOperation / DryingOperation / ShapingOperation (room-temp prep)
          - QuenchingOperation (the temperature_c on a quench/cool op is the
            DESTINATION, not a heating setpoint, so it's typically 25°C and
            would always fail TEMP_MIN).

        The chempot-atmosphere check still considers quench/cool atmospheres
        because those *are* meaningful (e.g., quenching in Ar vs. air); only
        temperature plausibility excludes them.
        """
        temps = []
        for op in predicted.operations:
            op_type = self._normalize_op_type(op.type)
            if op_type == "HeatingOperation":
                temps.extend(op.conditions.heating_temperature)

        if not temps:
            return 0.5  # no heating temps specified — neutral
        in_range = [1.0 if TEMP_MIN <= t <= TEMP_MAX else 0.0 for t in temps]
        return sum(in_range) / len(in_range)

    def _check_thermodynamics(self, predicted: PredictedRoute) -> tuple[float, str]:
        """
        ΔE_rxn via ComputedReaction. Continuous score with piecewise-linear
        mapping from eV/atom to [0, 1].

        Returns (score, gradeability) — gradeability is one of "discrete",
        "interpolated", "ungradeable", or "no_thermo_checker". This
        replaces the old bare-float return specifically so a genuine
        interpolated/discrete value is never indistinguishable from a
        0.5 can't-compute sentinel downstream (the schema gap flagged in
        HANDOFF_2 — a stored 0.5 used to mean either "computed and happens
        to be near 0.5" or "couldn't compute at all", with no way to tell
        which after the fact).
        """
        if self.thermo_checker is None:
            return 0.5, "no_thermo_checker"

        try:
            precursor_pairs = [(p.formula, p.amount) for p in predicted.precursors]
            delta_G, gradeability = self.thermo_checker.reaction_energy_per_atom(
                precursor_pairs, predicted.target_formula,
                predicted_route=predicted,
            )
        except Exception:
            return 0.5, "ungradeable"

        if delta_G is None:
            return 0.5, gradeability  # "ungradeable" from reaction_energy_per_atom

        # Piecewise-linear scoring applied to ΔG_rxn(T_synthesis), not 0K ΔE.
        if delta_G <= RXN_ENERGY_FAVORABLE:
            return 1.0, gradeability
        elif delta_G <= RXN_ENERGY_BORDERLINE:
            t = (delta_G - RXN_ENERGY_FAVORABLE) / (RXN_ENERGY_BORDERLINE - RXN_ENERGY_FAVORABLE)
            return 1.0 - 0.5 * t, gradeability
        elif delta_G <= RXN_ENERGY_UNFAVORABLE:
            t = (delta_G - RXN_ENERGY_BORDERLINE) / (RXN_ENERGY_UNFAVORABLE - RXN_ENERGY_BORDERLINE)
            return 0.5 - 0.5 * t, gradeability
        else:
            return 0.0, gradeability

    def _check_target_stability(self, predicted: PredictedRoute) -> tuple[float, str]:
        """
        Score based on e_above_hull of the target itself.

        Continuous:
          ≤ 25 meV/atom    → 1.0 (on hull within DFT noise)
          25-100 meV/atom  → linear decay 1.0 → 0.5 (synthesizable metastable)
          100-250 meV/atom → linear decay 0.5 → 0.0
          > 250 meV/atom   → 0.0

        Returns (0.5, "sentinel_no_entry") if the target isn't a discrete PD
        entry (e.g. novel solid solution). Deliberately NOT interpolated,
        unlike reaction_energy_per_atom: e_above_hull measures how far a
        discrete entry sits above the equilibrium mixture at its own
        composition. A target represented at its own interpolated hull
        value doesn't have a meaningful "distance above hull" - it would
        trivially be 0 by construction, which overstates confidence rather
        than fixing anything. This stays an honest, explicitly-tagged
        sentinel rather than a fabricated number.
        """
        if self.thermo_checker is None:
            return 0.5, "no_thermo_checker"
        try:
            e_hull = self.thermo_checker.target_e_above_hull(predicted.target_formula)
        except Exception:
            return 0.5, "ungradeable"
        if e_hull is None:
            return 0.5, "sentinel_no_entry"

        if e_hull <= HULL_STABLE:
            return 1.0, "discrete"
        elif e_hull <= HULL_METASTABLE:
            t = (e_hull - HULL_STABLE) / (HULL_METASTABLE - HULL_STABLE)
            return 1.0 - 0.5 * t, "discrete"
        elif e_hull <= HULL_UNSTABLE:
            t = (e_hull - HULL_METASTABLE) / (HULL_UNSTABLE - HULL_METASTABLE)
            return 0.5 - 0.5 * t, "discrete"
        else:
            return 0.0, "discrete"

    def _check_chempot_atmosphere(self, predicted: PredictedRoute) -> tuple[float, str]:
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

        Returns (score, gradeability). Unlike reaction_energy_per_atom, this
        check does NOT depend on the target having a discrete PD entry —
        get_all_chempots operates on facet geometry for any composition in
        the covered chemsys, discrete or not (confirmed against pymatgen
        source, 2026-07). So this was never actually blocked by the
        target_entry_missing failure mode; the tag is added here purely for
        schema consistency across all three thermo checks, not because a
        fix was needed.
        """
        if self.thermo_checker is None:
            return 0.5, "no_thermo_checker"

        # Step 1: short-circuit for non-oxide targets — the check doesn't apply.
        try:
            target_comp = Composition(predicted.target_formula)
            if Element("O") not in target_comp.elements:
                return 1.0, "not_applicable"
        except Exception:
            return 0.5, "ungradeable"

        # Step 2: fetch chempot envelope.
        try:
            all_chempots = self.thermo_checker.composition_chempots(predicted.target_formula)
            mu_O_ref = self.thermo_checker.oxygen_reference_energy(predicted.target_formula)
        except Exception:
            return 0.5, "ungradeable"

        if not all_chempots or mu_O_ref is None:
            return 0.5, "ungradeable"

        o_el = Element("O")
        o_mus_delta = [
            float(facet[o_el]) - float(mu_O_ref)
            for facet in all_chempots.values()
            if o_el in facet
        ]
        if not o_mus_delta:
            # PD has oxygen ref but no facet exposes μ_O — unusual; be neutral.
            return 0.5, "ungradeable"

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
                return 0.0, "discrete"
            return 1.0, "discrete"

        if needs_oxidizing:
            return (1.0 if all(c == "ox" for c in classes) else 0.0), "discrete"
        if needs_reducing:
            return (1.0 if all(c in {"red", "inert"} for c in classes) else 0.0), "discrete"
        return 1.0, "discrete"

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