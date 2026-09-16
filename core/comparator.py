"""
core/comparator.py — Phase 13 pairwise comparator.

Spec: misc/PHASE13_14_SPEC.md. Pre-registration: misc/PHASE13_PREREG.md
(read its dated addendum before touching C3 -- the compound melting-point
table below is its output, not an independent invention). Read CLAUDE.md
Central Diagnosis and findings 19/24/25 first.

`core/ranker.py` asked "does the model fail this channel sometimes" and
scored each route in isolation (gates x weighted objectives). This module
asks a narrower, harder question: "does this channel say route A is better
than route B, and does a chemist agree." It is PAIRWISE, not per-route --
`compare(a, b, target) -> (margin, breakdown)` -- and every channel's raw
quantity is UNCLIPPED (larger is better, enforced by test); scale comes
from a robust spread computed once over unlabeled generations (never
ASTRAL -- ASTRAL is the readout, never the calibration signal).

Gates (format_ok, balances, precursors_exist, charge_neutral,
temperature_physical) are reused unmodified from `core/ranker.py` --
this module does not reimplement or touch gate logic. A route failing any
gate scores None on every channel (see `score_channels`).

`ranker.py` and `validator.py` stay untouched -- Arm A/B remain
reproducible. Nothing here is fitted to ASTRAL: every functional form and
parameter below traces to a physical relationship (Tammann, Trouton/
Clausius-Clapeyron, combinatorics, DFT formation/reaction energetics) or a
sourced table, per PHASE13_14_SPEC.md's four design rules.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

from pymatgen.core import Composition, Element

from validator import (
    PredictedRoute,
    SynthesisValidator,
    ThermoChecker,
    VOLATILE_FORMULAS,
)
from core.ranker import (
    Ranker,
    RankerScales,
    GATE_NAMES,
    VOLATILE_T,
    PRECURSOR_DECOMP_T,
    _route_max_T,
)

COMPARATOR_VERSION = "2026-09-16-v1-phase13"

CHANNEL_NAMES = [
    "C1_selectivity_margin",
    "C2_unspent_driving_force",
    "C3_reactive_temperature_window",
    "C4_interface_count",
    "C5_volatilization",
    "C6_decomposition_clearance",
    "C7_gas_evolution",
]
# C8_diversity is deliberately absent: it is group-level (needs the other 7
# completions in a GDPO group), ASTRAL routes were not generated in groups
# (spec, C8 section), and it is validated only by Phase 14's manipulation
# check -- never scored in Phase 13's pairwise comparison.

_C_TO_K = 273.15


def _c_to_k(t_celsius: float) -> float:
    return t_celsius + _C_TO_K


# Compound melting points, KELVIN, verified via Wikipedia infobox WebFetch
# 2026-09-16 (2 exceptions below via WebSearch, flagged), keyed on the raw
# formula string as it appears in misc/astral_validation_set.json --
# normalized at lookup time via SynthesisValidator._normalize_formula so
# this shares the validator's key space, same convention as
# core.ranker.PRECURSOR_DECOMP_T.
#
# Built target-driven, NOT corpus-frequency-driven: every one of the 34
# unique precursor formulas across BOTH sides of all 35 ASTRAL target pairs
# is covered here (misc/PHASE13_PREREG.md addendum, 2026-09-16) -- this is
# what keeps the table from reintroducing finding 8's gradeability
# asymmetry (dense coverage for corpus-common precursors, sparse for
# ASTRAL-novel ones). Materials Project has NO compound melting-point
# field at all (confirmed via a direct `available_fields` query) and a
# regression from MP formation energy was tested and rejected (R^2=0.229
# global; residual std comparable to or exceeding this channel's own
# window width) -- see the pre-reg addendum for both negative results.
#
# Caveats carried from the addendum, not resolved here: Li2CO3, NH4H2PO4,
# ZnO report a temperature where decomposition and melting are concurrent,
# not a clean solid->liquid transition -- still usable as a window
# boundary, just not a "clean melt" in the strict sense. K3PO4 and LiPO3
# had no Wikipedia thermal data and were sourced via WebSearch from
# secondary chemical-supplier pages instead -- lower confidence, flagged
# for chemist spot-check, same posture as the unread Merkle & Maier
# Tammann-rule citation in the pre-registration.
_COMPOUND_MELT_T_K_RAW = {
    "Al2O3": 2327.0, "BaO": 2196.0, "Bi2O3": 1090.0, "B2O3": 723.0,
    "CuO": 1599.0, "Fe2O3": 1812.0, "GeO2": 1388.0, "K2CO3": 1164.0,
    "K3PO4": 1613.0,       # lower confidence -- single non-Wikipedia source
    "KNbO3": 1373.0, "KPO3": 1080.0, "Li2CO3": 996.0, "Li2TiO3": 1806.0,
    "LiBO2": 1122.0, "LiNbO3": 1510.0,
    "LiPO3": 929.0,         # lower confidence -- two sources disagreed
                            # (656/669 C); lower value used
    "MgO": 3125.0, "MnO": 2218.0, "Na2CO3": 1124.0, "NaBO2": 1239.0,
    "NH4H2PO4": 463.0, "NiO": 2228.0, "Pr6O11": 2456.0, "Sc2O3": 2758.0,
    "SiO2": 1986.0, "SrO": 2804.0, "Ta2O5": 2145.0, "TiO2": 2116.0,
    "V2O3": 2210.0, "WO3": 1746.0, "Y2O3": 2698.0, "ZnO": 2247.0,
    "ZrO2": 2988.0,
}
COMPOUND_MELT_T_K = {
    SynthesisValidator._normalize_formula(f): t
    for f, t in _COMPOUND_MELT_T_K_RAW.items()
}


def melt_point_k(formula: str) -> Optional[float]:
    """T_melt in Kelvin: pymatgen element data for single elements
    (already Kelvin, no conversion), COMPOUND_MELT_T_K for compounds. None
    if neither knows it -- C3 degrades gracefully per-species rather than
    refusing to score (see _species_ceiling_candidates_k)."""
    try:
        comp = Composition(formula)
        if len(comp.elements) == 1:
            el = comp.elements[0]
            mp = Element(str(el)).melting_point
            if mp:
                return float(mp)
    except Exception:
        pass
    key = SynthesisValidator._normalize_formula(formula)
    return COMPOUND_MELT_T_K.get(key)


def _species_ceiling_candidates_k(formula: str) -> list[float]:
    """Every known upper-bound temperature (Kelvin) for one species: its
    own melting point, its decomposition onset (PRECURSOR_DECOMP_T, if the
    formula happens to be one of those salts), and the volatilization
    onset of any volatile element (VOLATILE_T) present in its composition.
    T_high (see _c3) is the min across every species' candidates -- the
    single most restrictive ceiling wins, which is what "first loss
    mechanism to trigger" means physically."""
    out: list[float] = []
    m = melt_point_k(formula)
    if m is not None:
        out.append(m)
    d = PRECURSOR_DECOMP_T.get(SynthesisValidator._normalize_formula(formula))
    if d is not None:
        out.append(_c_to_k(d))
    try:
        elements = {str(e) for e in Composition(formula).elements}
    except Exception:
        elements = set()
    for el in elements:
        if el in VOLATILE_T:
            out.append(_c_to_k(VOLATILE_T[el]))
    return out


@dataclass
class ComparatorParams:
    """C3's width parameter -- the one sensitivity sweep the pre-reg
    requires (misc/PHASE13_PREREG.md, four arms: fraction 0.12/0.17/0.22,
    plus flat_100k for direct comparison to the original un-sourced
    design). `c3_fraction=None` selects the flat-100K arm; `c3_fraction=x`
    sets w_low = x * T_low's own melting-point anchor (Kelvin) and
    w_high = x * T_high itself (Kelvin) -- "of whichever precursor [or
    boundary mechanism] sets that boundary," per the addendum. All other
    parameters (VOLATILE_T onsets, PRECURSOR_DECOMP_T, Trouton's 10.6,
    C6's 40 K sigmoid width) are physical constants or existing tables,
    not swept."""
    c3_fraction: Optional[float] = 0.17  # pre-registered read: arm B


class Comparator:
    """Gate-and-channel scorer, then a pairwise diff. Mirrors `Ranker`'s
    posture (gate machinery reused verbatim via an internal `Ranker`
    instance -- ranker.py itself is never modified) but every channel
    method returns a single unclipped Optional[float] raw quantity, not a
    (clipped score, raw) pair: there is no clip step in this module by
    design (PHASE13_14_SPEC.md design rule 2 -- physics-shaped, no
    clip(0,1) chosen for convenience). Scale/normalization happens once,
    in `compare()`, from MAD values computed over unlabeled generations."""

    def __init__(self, mp_formula_set: set[str], thermo_checker: Optional[ThermoChecker]):
        # Gate machinery only -- precursor_freq is irrelevant here since no
        # ranker objective is ever read off this instance.
        self._gater = Ranker(mp_formula_set, thermo_checker, precursor_freq={})
        self._v = self._gater._v
        self.thermo = thermo_checker

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score_channels(
        self,
        predicted: Optional[PredictedRoute],
        target_formula: str,
        params: Optional[ComparatorParams] = None,
    ) -> dict[str, Optional[float]]:
        """One route -> {channel_name: raw value or None}. None on every
        channel if `predicted` is None or any gate fails -- the pairwise
        comparator test in tests/test_comparator.py checks this directly
        (PHASE13_14_SPEC.md's third required test, extended to C1-C7 since
        C8 isn't scored here at all)."""
        params = params or ComparatorParams()
        if predicted is None:
            return {c: None for c in CHANNEL_NAMES}
        gates = self._gater._check_gates(predicted)
        if not all(gates.values()):
            return {c: None for c in CHANNEL_NAMES}
        return {
            "C1_selectivity_margin": self._c1(predicted, target_formula),
            "C2_unspent_driving_force": self._c2(predicted, target_formula),
            "C3_reactive_temperature_window": self._c3(predicted, target_formula, params),
            "C4_interface_count": self._c4(predicted),
            "C5_volatilization": self._c5(predicted, target_formula),
            "C6_decomposition_clearance": self._c6(predicted),
            "C7_gas_evolution": self._c7(predicted, target_formula),
        }

    def compare(
        self,
        a: Optional[PredictedRoute],
        b: Optional[PredictedRoute],
        target_formula: str,
        scales: dict[str, float],
        params: Optional[ComparatorParams] = None,
    ) -> tuple[float, dict]:
        """(margin, breakdown). margin = sum over symmetrically-gradeable
        channels of (raw_a - raw_b) / MAD_scale[channel]. A channel enters
        the sum only if BOTH routes have a non-None value for it -- no
        renormalization, no payout for None (PHASE13_14_SPEC.md
        Aggregation section). Weights are uniform (1/MAD, no further
        per-channel weight) per the same section: "Weights. Uniform. Do
        not tune."

        Exactly antisymmetric: compare(b, a, ...) == (-margin, ...) with
        each per-channel diff negated -- enforced by
        tests/test_comparator.py, not just claimed here."""
        sa = self.score_channels(a, target_formula, params)
        sb = self.score_channels(b, target_formula, params)
        breakdown: dict = {"comparator_version": COMPARATOR_VERSION}
        margin = 0.0
        n_gradeable = 0
        for c in CHANNEL_NAMES:
            va, vb = sa[c], sb[c]
            breakdown[f"{c}_a"] = va
            breakdown[f"{c}_b"] = vb
            if va is None or vb is None:
                breakdown[f"{c}_gradeable"] = False
                breakdown[f"{c}_diff"] = None
                continue
            scale = scales.get(c)
            if not scale or scale <= 0:
                breakdown[f"{c}_gradeable"] = False
                breakdown[f"{c}_diff"] = None
                continue
            diff = va - vb
            breakdown[f"{c}_gradeable"] = True
            breakdown[f"{c}_diff"] = diff
            breakdown[f"{c}_diff_scaled"] = diff / scale
            margin += diff / scale
            n_gradeable += 1
        breakdown["n_gradeable_channels"] = n_gradeable
        breakdown["margin"] = margin
        return margin, breakdown

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------

    def _c1(self, predicted: PredictedRoute, target_formula: str) -> Optional[float]:
        """Selectivity margin (PHASE13_14_SPEC.md C1). For every pairwise
        precursor interface, InterfacialReactivity's tie-line kinks give a
        reaction energy per atom at each composition; margin at that
        interface = (most negative reaction energy among kinks whose
        products EXCLUDE the target) - (most negative reaction energy
        among kinks whose products INCLUDE the target). Positive = target
        favored. For n>2 precursors, take the MINIMUM over all pairwise
        interfaces (documented generalization, misc/PHASE13_PREREG.md
        "C1's n>2 generalization" section) -- the worst interface limits
        the route. None if there are <2 precursors (no interface exists)
        or no interface has both a target-forming and a competing kink."""
        precursors = predicted.precursors or []
        if len(precursors) < 2 or self.thermo is None:
            return None
        try:
            target_red = Composition(target_formula).reduced_formula
            core_formulas = [target_formula] + [p.formula for p in precursors]
            pd, _ = self.thermo._resolve_pd(core_formulas)
        except Exception:
            return None
        if pd is None:
            return None
        from pymatgen.analysis.interface_reactions import InterfacialReactivity
        margins = []
        for pa, pb in itertools.combinations(precursors, 2):
            try:
                ca = Composition(pa.formula)
                cb = Composition(pb.formula)
                ir = InterfacialReactivity(ca, cb, pd, norm=True, use_hull_energy=False)
                target_es, comp_es = [], []
                for _idx, _x, _energy, reaction, rxn_e in ir.get_kinks():
                    is_target = False
                    for prod in reaction.products:
                        try:
                            if prod.reduced_formula == target_red:
                                is_target = True
                                break
                        except Exception:
                            continue
                    (target_es if is_target else comp_es).append(rxn_e)
            except Exception:
                continue
            if not target_es or not comp_es:
                continue
            dG_target = min(target_es)
            dG_comp = min(comp_es)
            margins.append(dG_comp - dG_target)
        if not margins:
            return None
        return min(margins)

    def _c2(self, predicted: PredictedRoute, target_formula: str) -> Optional[float]:
        """Unspent driving force (C2). Mean formation energy per atom
        (relative to the elements) of the declared precursors -- larger
        (less negative) is better, replacing precursor_instability's
        e_above_hull (which sits at ~0 for any isolable solid, per
        RankerScales' docstring finding)."""
        precursors = predicted.precursors or []
        if not precursors or self.thermo is None:
            return None
        try:
            core_formulas = [target_formula] + [p.formula for p in precursors]
            pd, _ = self.thermo._resolve_pd(core_formulas)
            if pd is None:
                return None
            efs = []
            for p in precursors:
                entry = self.thermo._best_entry_for_formula(pd, p.formula)
                if entry is None:
                    continue
                ef = pd.get_form_energy_per_atom(entry)
                if ef is not None:
                    efs.append(ef)
        except Exception:
            return None
        if not efs:
            return None
        return sum(efs) / len(efs)

    def _c3(self, predicted: PredictedRoute, target_formula: str,
            params: ComparatorParams) -> Optional[float]:
        """Reactive temperature window (C3), Tammann's rule. Computed
        entirely in KELVIN internally -- the fractional relationship to
        T_melt (0.5x, or the swept 0.12/0.17/0.22x width) is only
        physically meaningful on an absolute temperature scale; mixing in
        the Celsius values used elsewhere in this codebase would silently
        shift T_low by ~270 K x fraction, a real error, not a units
        nicety. T_max (the route's own reported temperature) is Celsius
        everywhere else in this codebase and is converted here at the
        boundary.

        T_low = 0.5 x max over precursors of T_melt (Kelvin). Precursors
        with no known T_melt are skipped, not treated as blocking --
        None only if NO precursor has a known melting point at all.
        T_high = min over {every precursor, the target} of
        {T_melt, T_decomp if tabulated, T_volat if any volatile element is
        present} (Kelvin) -- the single most restrictive ceiling from any
        species/mechanism wins. None if nothing at all is known.

        Score = -(delta_low/w_low)^2 - (delta_high/w_high)^2, zero inside
        the window. w_low/w_high are either the flat 100 K original design
        (params.c3_fraction is None) or params.c3_fraction times the
        Kelvin anchor that set that boundary (T_low's own melting-point
        anchor; T_high itself) -- the fractional-width sweep required by
        misc/PHASE13_PREREG.md."""
        precursors = predicted.precursors or []
        if not precursors:
            return None
        T_max_c = _route_max_T(predicted)
        if T_max_c is None:
            return None
        T_max_k = _c_to_k(T_max_c)

        melt_vals_k = [melt_point_k(p.formula) for p in precursors]
        melt_vals_k = [v for v in melt_vals_k if v is not None]
        if not melt_vals_k:
            return None
        T_low_anchor_k = max(melt_vals_k)
        T_low_k = 0.5 * T_low_anchor_k

        ceiling_candidates_k: list[float] = []
        for p in precursors:
            ceiling_candidates_k.extend(_species_ceiling_candidates_k(p.formula))
        ceiling_candidates_k.extend(_species_ceiling_candidates_k(target_formula))
        if not ceiling_candidates_k:
            return None
        T_high_k = min(ceiling_candidates_k)

        delta_low = max(0.0, T_low_k - T_max_k)
        delta_high = max(0.0, T_max_k - T_high_k)

        if params.c3_fraction is None:
            w_low, w_high = 100.0, 100.0
        else:
            w_low = params.c3_fraction * T_low_anchor_k
            w_high = params.c3_fraction * T_high_k
            if w_low <= 0 or w_high <= 0:
                return None
        return -((delta_low / w_low) ** 2) - ((delta_high / w_high) ** 2)

    def _c4(self, predicted: PredictedRoute) -> Optional[float]:
        """Interface count (C4). k = n(n-1)/2 pairwise interfaces;
        score = -k -- quadratic in n by combinatorics (2->1, 3->3, 4->6),
        replacing n_precursors' linear-penalty approximation of the same
        idea. None for a 0- or 1-precursor route: n=1 has no interface
        (k=0, a genuine best case, not "not applicable") -- but a route
        with 0 precursors already fails format_ok/gate, so in practice
        this only returns None pre-gate; n=1 returns 0.0, correctly the
        best score."""
        n = len(predicted.precursors or [])
        if n == 0:
            return None
        k = n * (n - 1) / 2.0
        return -k

    def _c5(self, predicted: PredictedRoute, target_formula: str) -> Optional[float]:
        """Volatilization (C5). Score = -sum_el exp(excess/T_scale(el))
        for every volatile element present in target or precursors
        (VOLATILE_T onset table, Celsius, unchanged first-pass table from
        ranker.py -- flagged there for chemist verification). T_scale
        derived from Trouton's rule (DeltaH_vap ~= 88 J/mol/K x T_boil) as
        T_scale = T_boil/10.6, T_boil from pymatgen Element.boiling_point
        (Kelvin) -- a WIDTH, so using it directly against a Celsius excess
        is dimensionally fine (temperature INTERVALS are the same size in
        both scales; only the offset differs, and offset cancels in a
        difference). No max(0, ...) clamp on excess, unlike ranker.py's
        volatility_risk -- the exponential decay below onset is itself the
        physically continuous behavior Clausius-Clapeyron predicts, per
        the spec's literal form. None if no volatile element is present
        (not applicable, not a free 0) or T_max is undeclared."""
        T_max = _route_max_T(predicted)
        if T_max is None:
            return None
        elements: set[str] = set()
        try:
            elements |= {str(e) for e in Composition(target_formula).elements}
        except Exception:
            pass
        for p in predicted.precursors or []:
            try:
                elements |= {str(e) for e in Composition(p.formula).elements}
            except Exception:
                continue
        volatile_present = [el for el in elements if el in VOLATILE_T]
        if not volatile_present:
            return None
        total = 0.0
        any_scaled = False
        for el in volatile_present:
            try:
                t_boil = Element(el).boiling_point
            except Exception:
                t_boil = None
            if not t_boil:
                continue
            t_scale = float(t_boil) / 10.6
            excess = T_max - VOLATILE_T[el]
            total += math.exp(excess / t_scale)
            any_scaled = True
        if not any_scaled:
            return None
        return -total

    def _c6(self, predicted: PredictedRoute) -> Optional[float]:
        """Decomposition clearance (C6). margin = T_max - T_decomp for the
        worst (highest-onset) tabulated precursor (PRECURSOR_DECOMP_T,
        Celsius, unchanged first-pass table -- flagged there for chemist
        verification). Score = sigmoid(margin/40 K), w=40 K from typical
        DTA decomposition-peak widths, replacing the old linear
        0.5+margin/610 ramp. None if no declared precursor is in the
        table."""
        T_max = _route_max_T(predicted)
        if T_max is None:
            return None
        relevant = []
        for p in predicted.precursors or []:
            key = SynthesisValidator._normalize_formula(p.formula)
            t_decomp = PRECURSOR_DECOMP_T.get(key)
            if t_decomp is not None:
                relevant.append(t_decomp)
        if not relevant:
            return None
        worst_t_decomp = max(relevant)
        margin = T_max - worst_t_decomp
        return 1.0 / (1.0 + math.exp(-margin / 40.0))

    def _c7(self, predicted: PredictedRoute, target_formula: str) -> Optional[float]:
        """Gas evolution (C7). moles of gaseous product (from
        VOLATILE_FORMULAS = CO2/H2O/O2/N2/NH3, validator.py's own
        candidate set) per mole of target, from the balanced reaction
        (`SynthesisValidator._find_balanced_reaction`, reused unmodified).
        Score = -(mol_gas/mol_target), linear and unbounded. None if no
        balance was found or the target's own coefficient is ~0."""
        try:
            reaction, _reactants = self._v._find_balanced_reaction(predicted)
        except Exception:
            return None
        if reaction is None:
            return None
        try:
            target_comp = Composition(target_formula)
            target_coeff = reaction.get_coeff(target_comp)
        except Exception:
            return None
        if target_coeff is None or abs(target_coeff) < 1e-9:
            return None
        gas_moles = 0.0
        found_any = False
        for gas_formula in VOLATILE_FORMULAS:
            try:
                c = reaction.get_coeff(Composition(gas_formula))
            except Exception:
                continue
            if c is not None and c > 1e-9:
                gas_moles += c
                found_any = True
        if not found_any:
            return 0.0  # balanced with no gaseous product -- a real, best-case 0
        return -(gas_moles / target_coeff)


def load_comparator(
    formula_set_path,
    pd_index_path,
    project_root,
) -> Comparator:
    """Mirrors core.ranker.load_ranker's signature/argument meaning."""
    import pickle
    from pathlib import Path
    with Path(formula_set_path).open("rb") as f:
        formula_set = pickle.load(f)
    thermo = None
    if pd_index_path and Path(pd_index_path).exists():
        root = project_root or Path(pd_index_path).parent
        thermo = ThermoChecker.from_sharded_cache(Path(pd_index_path), Path(root))
    return Comparator(formula_set, thermo)
