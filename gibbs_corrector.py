"""
gibbs_corrector.py
==================
Temperature-aware Gibbs free energy correction for the MIRA validator.

Replaces 0K DFT reaction energy with ΔG_rxn(T_synthesis) using:
  • Bartel et al. (2018) machine-learned SISSO descriptor for solid entries,
    via pymatgen's GibbsComputedStructureEntry
  • NIST-JANAF tabulated ΔfG°(T) values with linear interpolation for
    gas-phase species (CO2, H2O, O2, N2, NH3)

The synthesis temperature is extracted from the highest heating-operation
temperature in the predicted route. This produces physically meaningful
ΔG values that align with experimental Gibbs energies of solid-state
synthesis reactions, without any per-species calibration constants.

Design choice: NIST tabulated values + interpolation, NOT Shomate fits
----------------------------------------------------------------------
The Shomate equation provides an analytical fit to Cp(T) and integrates to
give H°(T) and S°(T) via polynomial expressions. For most gases this works
well. For graphite (an elemental reference needed to compute ΔfG° of CO2
from first principles), the Shomate fit gives unphysical values at room
temperature because graphite's heat capacity has anomalous low-T behavior
that isn't captured by the polynomial form.

Rather than fight this, we use NIST-JANAF's directly tabulated ΔfG°(T)
values. These are critical assessments of experimental measurements, not
fits, and they're authoritative. Linear interpolation between adjacent T
points introduces negligible error for the 100K-spaced JANAF data well
within the validator's noise tolerance.

References
----------
  Bartel et al., "Physical descriptor for the Gibbs energy of inorganic
    crystalline solids and temperature-dependent materials chemistry,"
    Nature Communications 9, 4168 (2018).
  McDermott et al., "A graph-based network for predicting chemical
    reaction pathways in solid-state materials synthesis,"
    Nature Communications 12, 3097 (2021).
  NIST-JANAF Thermochemical Tables, 4th edition (1998).
    https://janaf.nist.gov/

Caveats
-------
  - Bartel descriptor reports ~50 meV/atom MAE on its 440-compound test set.
    The validator's piecewise-linear scoring treats ΔG within ±0.05 eV/atom
    of zero as a noise band, so this MAE is absorbed.
  - Bartel descriptor is valid for inorganic crystalline solids, T ∈ [300, 2000] K.
  - NIST tabulated values cover T ∈ [298, 2000] K. Outside this range we
    clamp to the nearest endpoint and log a warning.
  - Solid CO2, H2O, NH3 entries from MP are excluded from Gibbs wrapping in
    favor of NIST gas-phase values. These species are always treated as gases
    since their sublimation/boiling points are well below typical synthesis T.
  - Synthesis temperature is taken as the MAX heating-operation temperature
    (calcine, sinter, anneal). This represents the reaction's operating point.
    Reactions occurring during ramping or cooling are not modeled.
"""

from __future__ import annotations

import logging
from typing import Optional

from pymatgen.core import Composition
from pymatgen.entries.computed_entries import (
    ComputedEntry,
    GibbsComputedStructureEntry,
)
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# 1 eV = 96.485332 kJ/mol exactly
_EV_PER_KJMOL = 1.0 / 96.485332

# Operations that determine "synthesis temperature".
# Match the canonical operation vocabulary in validator.py.
_HEATING_OP_NAMES = frozenset({"calcine", "sinter", "anneal"})

# Gas-phase species always handled via NIST tables (never via Bartel).
# Boiling points well below typical synthesis temperatures.
_GAS_SPECIES = frozenset({"CO2", "H2O", "O2", "N2", "NH3"})


# ===========================================================================
# NIST-JANAF tabulated Gibbs energies of formation (kJ/mol)
#
# Source: NIST-JANAF Thermochemical Tables, 4th edition (1998).
# https://janaf.nist.gov/
#
# Format per species: sorted list of (T_K, dfG_kJmol) tuples.
# Linear interpolation between adjacent points.
# For elemental references (O2, N2 in standard state), ΔfG° = 0 at all T
# by definition; empty list signals "always zero".
# ===========================================================================

_NIST_DFG_KJMOL: dict[str, list[tuple[float, float]]] = {
    "CO2": [
        (298.15, -394.359),
        (400,    -394.675),
        (500,    -394.939),
        (600,    -395.182),
        (700,    -395.398),
        (800,    -395.586),
        (900,    -395.748),
        (1000,   -395.886),
        (1100,   -395.999),
        (1200,   -395.886),
        (1300,   -395.665),
        (1400,   -395.408),
        (1500,   -395.117),
        (1600,   -394.795),
        (1700,   -394.443),
        (1800,   -394.063),
        (1900,   -393.659),
        (2000,   -393.232),
    ],
    "H2O": [
        (298.15, -228.582),
        (400,    -223.901),
        (500,    -219.051),
        (600,    -214.007),
        (700,    -208.812),
        (800,    -203.501),
        (900,    -198.091),
        (1000,   -192.590),
        (1100,   -187.033),
        (1200,   -181.425),
        (1300,   -175.774),
        (1400,   -170.078),
        (1500,   -164.376),
        (1600,   -158.639),
        (1700,   -152.873),
        (1800,   -147.082),
        (1900,   -141.268),
        (2000,   -135.643),
    ],
    "NH3": [
        (298.15,  -16.367),
        (400,      -5.940),
        (500,       4.800),
        (600,      15.879),
        (700,      27.190),
        (800,      38.662),
        (900,      50.247),
        (1000,     61.910),
        (1100,     73.625),
        (1200,     85.373),
        (1300,     97.141),
        (1400,    108.918),
        (1500,    120.696),
        (1600,    132.469),
        (1700,    144.234),
        (1800,    155.986),
        (1900,    167.725),
        (2000,    179.447),
    ],
    # Elemental references: ΔfG° = 0 at all T by definition
    "O2": [],
    "N2": [],
}


# ===========================================================================
# Interpolated Gibbs energy of formation
# ===========================================================================

def gibbs_formation_ev(species: str, T_K: float) -> float:
    """
    Standard Gibbs energy of formation ΔfG°(T) for a gas species in eV/molecule.
    
    Uses NIST-JANAF tabulated values with linear interpolation. Clamps to
    endpoint values (with a logged warning) if T_K is outside the table range.
    
    For elemental references (O2, N2 in standard state), returns 0.0.
    """
    if species not in _NIST_DFG_KJMOL:
        raise KeyError(f"No NIST ΔfG° data for {species!r}")
    
    table = _NIST_DFG_KJMOL[species]
    if not table:
        return 0.0  # Elemental reference
    
    T_min = table[0][0]
    T_max = table[-1][0]
    
    if T_K < T_min:
        logger.warning(
            f"[gibbs] T={T_K:.0f}K below NIST table for {species!r} "
            f"(min={T_min:.0f}K); clamping."
        )
        return table[0][1] * _EV_PER_KJMOL
    
    if T_K > T_max:
        logger.warning(
            f"[gibbs] T={T_K:.0f}K above NIST table for {species!r} "
            f"(max={T_max:.0f}K); clamping."
        )
        return table[-1][1] * _EV_PER_KJMOL
    
    # Linear interpolation
    for i in range(len(table) - 1):
        T1, G1 = table[i]
        T2, G2 = table[i + 1]
        if T1 <= T_K <= T2:
            df_g_kj = G1 + (G2 - G1) * (T_K - T1) / (T2 - T1)
            return df_g_kj * _EV_PER_KJMOL
    
    raise RuntimeError(f"Interpolation logic failed for {species!r} at T={T_K}K")


# ===========================================================================
# Synthesis temperature extraction
# ===========================================================================

def extract_synthesis_temperature_K(predicted_route) -> float:
    """
    Determine the temperature at which to evaluate ΔG_rxn.
    
    Take the highest temperature_c across all heating-type operations
    (calcine, sinter, anneal). Convert °C → K. If no heating ops, default to
    298.15 K (room temperature).
    
    This represents the operating point of the synthesis reaction. For the
    route to work, the reaction must be thermodynamically favorable at this T.
    """
    max_T_C = -float("inf")
    for op in predicted_route.operations:
        op_type = op.type
        op_type_str = op_type.lower() if hasattr(op_type, "lower") else str(op_type).lower()
        if op_type_str not in _HEATING_OP_NAMES:
            continue
        heating_temps = op.conditions.heating_temperature
        if heating_temps:
            max_T_C = max(max_T_C, max(heating_temps))
    
    if max_T_C == -float("inf"):
        return 298.15
    return max_T_C + 273.15


# ===========================================================================
# NIST gas entries for ComputedReaction
# ===========================================================================

def make_nist_gas_entry(species: str, T_K: float) -> ComputedEntry:
    """
    Create a ComputedEntry for a gas species with energy = ΔfG°(T) in eV.
    
    Combines with GibbsComputedStructureEntry solid entries in a
    ComputedReaction; the resulting calculated_reaction_energy is ΔG_rxn (eV).
    """
    composition = Composition(species)
    g_f_ev = gibbs_formation_ev(species, T_K)
    return ComputedEntry(
        composition=composition,
        energy=g_f_ev,
        entry_id=f"NIST-JANAF-{species}-T{T_K:.0f}K",
        data={"source": "NIST-JANAF tabulated", "temperature_K": T_K},
    )


# ===========================================================================
# Main entry point: ΔG_rxn computation
# ===========================================================================

def compute_reaction_gibbs_per_atom(target_formula: str, precursor_formulas: list[str], pd, predicted_route) -> tuple[Optional[float], float]:
    """
    Compute ΔG_rxn per atom of target at the synthesis temperature.
    
    Parameters
    ----------
    target_formula : str
        Reduced or unreduced formula of the synthesis target.
    precursor_formulas : list[str]
        Formulas of the proposed precursors. Order doesn't matter.
    pd : pymatgen.analysis.phase_diagram.PhaseDiagram
        Phase diagram covering the target+precursor chemsys.
    predicted_route : PredictedRoute
        Used to extract the synthesis temperature from heating operations.
    
    Returns
    -------
    (delta_G_per_atom_eV, T_synthesis_K) on success.
    (None, T_synthesis_K) if the reaction can't be computed.
    """
    T_K = extract_synthesis_temperature_K(predicted_route)
    
    # Step 1: separate solid candidates from gases.
    # Gas species from MP are dropped in favor of NIST values.
    solid_entries = []
    for e in pd.all_entries:
        if e.composition.reduced_formula in _GAS_SPECIES:
            continue
        solid_entries.append(e)
    
    if not solid_entries:
        logger.debug(f"[gibbs] No solid entries in PD for target {target_formula}")
        return None, T_K
    
    # Step 2: wrap all solid entries with Bartel-descriptor Gibbs correction.
    # Each entry's .energy becomes ΔfG(T) at the synthesis temperature.
    try:
        gibbs_solids = GibbsComputedStructureEntry.from_entries(
            solid_entries, temp=T_K
        )
    except Exception as exc:
        logger.warning(
            f"[gibbs] GibbsComputedStructureEntry.from_entries failed for "
            f"{target_formula} at T={T_K:.0f}K: {exc}"
        )
        return None, T_K
    
    # Step 3: build a lookup table by reduced formula. Keep the most stable
    # (lowest energy_per_atom) polymorph for each formula.
    by_formula: dict[str, GibbsComputedStructureEntry] = {}
    for e in gibbs_solids:
        red = e.composition.reduced_formula
        existing = by_formula.get(red)
        if existing is None or e.energy_per_atom < existing.energy_per_atom:
            by_formula[red] = e
    
    # Step 4: find target and precursor entries.
    target_red = Composition(target_formula).reduced_formula
    target_entry = by_formula.get(target_red)
    if target_entry is None:
        logger.debug(f"[gibbs] No PD entry for target {target_red}")
        return None, T_K
    
    precursor_entries = []
    missing = []
    for p in precursor_formulas:
        p_red = Composition(p).reduced_formula
        p_entry = by_formula.get(p_red)
        if p_entry is None:
            missing.append(p_red)
        else:
            precursor_entries.append(p_entry)
    if missing:
        logger.debug(f"[gibbs] Missing precursor entries: {missing}")
        return None, T_K
    
    # Step 5: NIST gas entries at synthesis temperature.
    gas_entries = [make_nist_gas_entry(sp, T_K) for sp in _GAS_SPECIES]
    
    # Step 6: balance the reaction.
    reactants = list(precursor_entries)
    products = [target_entry] + gas_entries
    
    try:
        reaction = ComputedReaction(reactants, products)
    except ReactionError as exc:
        logger.debug(f"[gibbs] ReactionError for {target_red}: {exc}")
        return None, T_K
    
    # Step 7: extract target coefficient via REDUCED composition.
    # ComputedReaction normalizes internally; querying with unreduced
    # form (e.g. Sr2Ti2O6) raises ValueError.
    target_comp_reduced = Composition(target_red)
    try:
        coeff = reaction.get_coeff(target_comp_reduced)
    except (ValueError, KeyError) as exc:
        logger.debug(f"[gibbs] get_coeff failed for {target_red}: {exc}")
        return None, T_K
    
    if coeff <= 1e-6:
        logger.debug(f"[gibbs] Target coefficient non-positive: {coeff}")
        return None, T_K
    
    # Step 8: ΔG_rxn / (atoms of target produced)
    delta_G_total = reaction.calculated_reaction_energy
    atoms_target = coeff * target_comp_reduced.num_atoms
    delta_G_per_atom = delta_G_total / atoms_target
    
    return delta_G_per_atom, T_K


# ===========================================================================
# Self-test (run module directly to verify interpolation against NIST tables)
# ===========================================================================

def _self_test():
    """Verify NIST interpolation against known reference values."""
    print("NIST ΔfG° interpolation self-test:")
    cases = [
        # (species, T_K, expected_kJmol from NIST-JANAF)
        ("CO2",  298.15, -394.359),
        ("CO2",  1000,   -395.886),
        ("CO2",  1273,   -395.734),   # interpolated between 1200 and 1300
        ("H2O",   500,   -219.051),
        ("H2O",  1000,   -192.590),
        ("H2O",  1373,   -172.617),   # interpolated between 1300 and 1400
        ("NH3",   600,     15.879),
        ("NH3",  1000,     61.910),
        ("O2",   1273,      0.0),
        ("N2",    800,      0.0),
    ]
    n_pass = 0
    for species, T, expected in cases:
        df_g_ev = gibbs_formation_ev(species, T)
        df_g_kj = df_g_ev / _EV_PER_KJMOL
        ok = abs(df_g_kj - expected) < 1.0
        marker = "✓" if ok else "✗"
        if ok:
            n_pass += 1
        print(f"  {marker} ΔfG°({species:3s}, {T:6.1f}K) = {df_g_kj:9.3f} kJ/mol  "
              f"[expected: {expected:9.3f}]")
    print(f"\n{n_pass}/{len(cases)} cases passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()