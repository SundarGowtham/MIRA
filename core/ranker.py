"""
core/ranker.py — a quality-ranking verifier (Arm B).

Spec: misc/some_claude_files/RANKER_SPEC.md. Read CLAUDE.md findings 1, 6, 9,
15 first. The Central Diagnosis: `validator.py` measures validity, not
quality — a well-trained policy passes its checks >99% of the time, so most
channels have zero within-group variance and RLVR has nothing to optimize.

**The validator checks; the ranker ranks.** Architecture: gates x objectives.

    reward = Π(gates) x Σ_i w_i * objective_i

Gates are binary and multiplicative, NOT reward channels — they cost nothing
when passed (as validity gates almost always are) and block garbage when
failed. Objectives are continuous, unbounded-effort, mutually trading, and
every one of them must be something the policy actually fails at a useful
rate — that is the whole design rule this file exists to satisfy.

Gate failure -> every objective is None (-> NaN downstream), never 0.0.
Paying 0.0 on gate failure would inject a spurious full-strength z-score
into the group statistics — the exact sentinel-payout bug validator.py's
None-propagation was built to eliminate, reappearing one level up.

This module imports validator.py's data classes, ThermoChecker, and (via a
thermo-less SynthesisValidator instance held internally) its balance solver,
formula normalization, and a few individual checks — but composes the reward
differently and does NOT modify validator.py. Arm A must stay reproducible.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import stdev
from typing import Optional

from pymatgen.core import Composition, Element

from validator import (
    PredictedRoute,
    SynthesisValidator,
    ThermoChecker,
    RXN_ENERGY_BORDERLINE,
)

RANKER_VERSION = "2026-08-29-v1"

GATE_NAMES = [
    "balances", "precursors_exist", "charge_neutral", "format_ok",
    "temperature_physical",
]
OBJECTIVE_NAMES = [
    "temperature_economy", "step_economy", "precursor_availability",
    "volatility_risk", "driving_force_margin",
]
# phase_purity REMOVED from the active reward (misc/some_claude_files/
# ranker_fixes_instructions.md step 2, 2026-08-30): the capacity probe
# (misc/ranker_capacity_probe.json) showed it 100% zero-std -- it counts
# every entry in the RESOLVED CHEMSYS within the purity window, which is
# constant per target (all 8 group completions typically share a chemsys)
# and just measures how big that chemsys's phase diagram is, not how
# selective the route is. That makes it prompt-determined, the same failure
# mode that killed the validator's target_stability. `_phase_purity` and
# `_competing_phases` are kept below (still computed, still written to the
# breakdown as `phase_purity_n_competing` for later analysis) but excluded
# from scoring. Deferred fix: redefine it to count phases reachable from
# the DECLARED PRECURSOR SET specifically, not the whole chemsys.

TEMP_PHYSICAL_MIN = 100.0
TEMP_PHYSICAL_MAX = 2200.0  # deliberately wider than validator's 100-2000
CHARGE_NEUTRAL_GATE_MIN = 0.9  # _check_charge_neutrality's continuous decay
                                # -- near-1.0 counts as "balances", not just ==1.0

# Approximate onset of significant volatilization loss, degrees C.
# SPEC FLAG (RANKER_SPEC.md 2.4): these are a first pass, chemistry- and
# partial-pressure-dependent, not a verified reference table. They don't
# need to be exactly right to create gradient -- only directionally right
# and varying across routes. Ask the user to sanity-check before trusting
# this for anything beyond reward shaping.
VOLATILE_T = {
    "Ag": 950, "Pb": 850, "Bi": 900, "Li": 1100, "Na": 900, "K": 850,
    "Zn": 900, "Cd": 700, "Hg": 350, "Se": 700, "Te": 900, "P": 800,
    "S": 600, "Sb": 900, "Tl": 700, "As": 600,
}


def _route_max_T(predicted: PredictedRoute) -> Optional[float]:
    """Max heating-op temperature reported anywhere in the route."""
    temps = []
    for op in predicted.operations or []:
        for t in op.conditions.heating_temperature or []:
            try:
                if t is not None:
                    temps.append(float(t))
            except (TypeError, ValueError):
                pass
    return max(temps) if temps else None


def _route_n_ops(predicted: PredictedRoute) -> int:
    return len(predicted.operations or [])


def build_precursor_frequency(synthesis_clean_path: Path | str) -> dict[str, int]:
    """
    Precursor formula -> corpus frequency, from the 17.6k Kononova literature
    routes (data/raw/synthesis_clean.json). Normalized with
    SynthesisValidator._normalize_formula (reduced_formula) so lookups at
    score time use the same key space as the MP formula-set checks.
    """
    freq: dict[str, int] = {}
    records = json.loads(Path(synthesis_clean_path).read_text())
    for r in records:
        for p in r.get("precursors") or []:
            f = p.get("formula") if isinstance(p, dict) else p
            if not f:
                continue
            key = SynthesisValidator._normalize_formula(f)
            freq[key] = freq.get(key, 0) + 1
    return freq


@dataclass
class RankerScales:
    """Per-objective scale parameters -- everything RANKER_SPEC.md section 5's
    rail-calibration step is meant to tune. These are LIVE DEFAULTS, not
    just CLI-tunable knobs: the capacity probe (misc/ranker_capacity_probe.json)
    was run via `Ranker(formula_set, thermo, freq)` with no scales override,
    so whatever is written here is what run 4 actually trains against unless
    a caller overrides it -- keep this in sync with the current
    best-calibrated values, don't just tune a CLI flag and leave these stale
    (that gap is exactly what happened to the first capacity probe: rail
    calibration found n_max=5 was ~50x too small, but only the standalone
    calibration script's CLI got the fix, not this dataclass, so the real
    probe ran uncalibrated).

    Values below are ranker_fixes_instructions.md step 3 (2026-08-30),
    picked from misc/ranker_capacity_probe.json's actual per-channel rail
    rates (320 real completions, 40 groups) -- t_span/n_span/cost_scale
    widened to cut tie-at-the-rail rates; z-normalization within GDPO makes
    a constant rescale gradient-neutral, so the only thing these scales
    control is where clip(0,1) bites."""
    t_ref_margin: float = 150.0
    t_span: float = 800.0             # was 400 -- 42.8% of routes clipped at 0.0
    n_ref_margin: int = 2
    n_span: float = 5.0               # was 3-4 -- 41.3% clipped at 1.0
    cost_scale: float = 3.0           # was 10 -- no clipping, but std only 0.082
    dg_scale: float = 0.3             # eV/atom -- best-behaved channel, unchanged
    n_max: float = 250.0              # phase_purity is inactive (see OBJECTIVE_NAMES)
                                       # but kept sane for the retained diagnostic
    purity_window: float = 0.05       # eV/atom (50 meV) above hull


class Ranker:
    """
    Gates x objectives scorer. Drop-in analog of
    `SynthesisValidator.validate(route, target) -> (reward, breakdown)`,
    for use with probe_hardening.py's `--scorer {validator,ranker}`.
    """

    def __init__(
        self,
        mp_formula_set: set[str],
        thermo_checker: Optional[ThermoChecker],
        precursor_freq: dict[str, int],
        scales: Optional[RankerScales] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        # Thermo-less validator instance held ONLY to reuse gate machinery
        # (balance solver, formula normalization, the charge/temperature
        # checks) -- .validate() on this instance is never called; its
        # weight vector is irrelevant here.
        self._v = SynthesisValidator(mp_formula_set, thermo_checker=None)
        self.thermo = thermo_checker
        self.precursor_freq = precursor_freq
        self.total_freq = max(1, sum(precursor_freq.values()))
        self.max_cost = -math.log(1.0 / (2 * self.total_freq))
        self.scales = scales or RankerScales()
        self.weights = weights or {name: 1.0 / len(OBJECTIVE_NAMES)
                                   for name in OBJECTIVE_NAMES}
        self._competing_phase_cache: dict[str, list[tuple[str, float]]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(
        self,
        predicted: Optional[PredictedRoute],
        target_formula: str,
        lit_T: Optional[float] = None,
        lit_n_ops: Optional[int] = None,
    ) -> tuple[float, dict]:
        """Returns (reward, breakdown) like SynthesisValidator.validate().
        breakdown carries every gate (bool) and every objective (float or
        None), plus `*_gradeability` siblings where relevant, plus
        `ranker_version`."""
        info: dict = {"ranker_version": RANKER_VERSION}

        if predicted is None:
            for g in GATE_NAMES:
                info[f"gate_{g}"] = False
            for o in OBJECTIVE_NAMES:
                info[o] = None
            info["reward"] = 0.0
            return 0.0, info

        gates = self._check_gates(predicted)
        for g, v in gates.items():
            info[f"gate_{g}"] = v

        if not all(gates.values()):
            for o in OBJECTIVE_NAMES:
                info[o] = None
            info["reward"] = 0.0
            return 0.0, info

        dG, dG_grade = self._driving_force(predicted, target_formula)
        info["driving_force_margin_gradeability"] = dG_grade

        obj: dict[str, Optional[float]] = {}
        obj["temperature_economy"] = self._temperature_economy(predicted, lit_T, dG)
        obj["step_economy"] = self._step_economy(predicted, lit_n_ops)
        obj["precursor_availability"] = self._precursor_availability(predicted)
        obj["volatility_risk"] = self._volatility_risk(predicted, target_formula)
        # phase_purity: computed and logged (diagnostic only) but NOT added
        # to `obj` -- it must stay out of the weighted reward now that it's
        # not in OBJECTIVE_NAMES/self.weights (see the module-level note).
        info["phase_purity_INACTIVE"], info["phase_purity_n_competing"] = \
            self._phase_purity(predicted, target_formula)
        obj["driving_force_margin"] = (
            None if dG is None else
            max(0.0, min(1.0, -dG / self.scales.dg_scale))
        )
        info.update(obj)

        active = {k: v for k, v in obj.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
        wsum = sum(self.weights[k] for k in active)
        reward = 0.0 if wsum == 0 else sum(
            (self.weights[k] / wsum) * v for k, v in active.items()
        )
        info["reward"] = round(reward, 4)
        return info["reward"], info

    # ------------------------------------------------------------------
    # Gates (binary, multiplicative)
    # ------------------------------------------------------------------

    def _check_gates(self, predicted: PredictedRoute) -> dict[str, bool]:
        return {
            "format_ok": self._gate_format_ok(predicted),
            "balances": self._gate_balances(predicted),
            "precursors_exist": self._gate_precursors_exist(predicted),
            "charge_neutral": self._gate_charge_neutral(predicted),
            "temperature_physical": self._gate_temperature_physical(predicted),
        }

    @staticmethod
    def _gate_format_ok(predicted: PredictedRoute) -> bool:
        return bool(predicted.precursors) and bool(predicted.operations)

    def _gate_balances(self, predicted: PredictedRoute) -> bool:
        try:
            reaction, _ = self._v._find_balanced_reaction(predicted)
            return reaction is not None
        except Exception:
            return False

    def _gate_precursors_exist(self, predicted: PredictedRoute) -> bool:
        try:
            return self._v._check_precursors_exist(predicted) >= 0.999
        except Exception:
            return False

    def _gate_charge_neutral(self, predicted: PredictedRoute) -> bool:
        try:
            return self._v._check_charge_neutrality(predicted) >= CHARGE_NEUTRAL_GATE_MIN
        except Exception:
            return False

    @staticmethod
    def _gate_temperature_physical(predicted: PredictedRoute) -> bool:
        temps = []
        for op in predicted.operations or []:
            if SynthesisValidator._normalize_op_type(op.type) == "HeatingOperation":
                temps.extend(op.conditions.heating_temperature or [])
        if not temps:
            # No declared heating temp to gate on -- vacuous pass (garbage
            # with no temperature at all is caught by format_ok/balances,
            # not this gate's job).
            return True
        return all(TEMP_PHYSICAL_MIN <= t <= TEMP_PHYSICAL_MAX for t in temps)

    # ------------------------------------------------------------------
    # Objectives (continuous, gate failure already excluded by score())
    # ------------------------------------------------------------------

    def _driving_force(self, predicted: PredictedRoute,
                       target_formula: str) -> tuple[Optional[float], str]:
        if self.thermo is None:
            return None, "no_thermo_checker"
        try:
            precursor_pairs = [(p.formula, p.amount) for p in predicted.precursors]
            return self.thermo.reaction_energy_per_atom(
                precursor_pairs, target_formula, predicted_route=predicted)
        except Exception:
            return None, "ungradeable"

    def _temperature_economy(self, predicted: PredictedRoute,
                             lit_T: Optional[float],
                             dG: Optional[float]) -> Optional[float]:
        """RANKER_SPEC.md 2.1. Gated on feasibility at T_max (dG <= the
        validator's own borderline-favorable cutoff) -- else the model
        trivially maximizes by reporting room temperature regardless of
        whether the reaction would actually run there."""
        if lit_T is None or dG is None or dG > RXN_ENERGY_BORDERLINE:
            return None
        T_max = _route_max_T(predicted)
        if T_max is None:
            return None
        T_ref = lit_T + self.scales.t_ref_margin
        return max(0.0, min(1.0, (T_ref - T_max) / self.scales.t_span))

    def _step_economy(self, predicted: PredictedRoute,
                      lit_n_ops: Optional[int]) -> Optional[float]:
        if lit_n_ops is None:
            return None
        n_ops = _route_n_ops(predicted)
        n_ref = lit_n_ops + self.scales.n_ref_margin
        return max(0.0, min(1.0, (n_ref - n_ops) / self.scales.n_span))

    def _precursor_availability(self, predicted: PredictedRoute) -> Optional[float]:
        if not predicted.precursors:
            return None
        costs = []
        for p in predicted.precursors:
            key = SynthesisValidator._normalize_formula(p.formula)
            freq = self.precursor_freq.get(key, 0)
            cost = self.max_cost if freq <= 0 else -math.log(freq / self.total_freq)
            costs.append(cost)
        mean_cost = sum(costs) / len(costs)
        return max(0.0, min(1.0, 1.0 - mean_cost / self.scales.cost_scale))

    def _volatility_risk(self, predicted: PredictedRoute,
                         target_formula: str) -> Optional[float]:
        """None (not 1.0) when no volatile-risk element is present -- the
        channel doesn't apply to that route, it hasn't EARNED a perfect
        score (ranker_fixes_instructions.md step 1, 2026-08-30: a hardcoded
        1.0 here was most of a 53.9% ceiling pile in the capacity probe)."""
        elements: set[str] = set()
        try:
            elements |= {str(e) for e in Composition(target_formula).elements}
        except Exception:
            pass
        for p in predicted.precursors:
            try:
                elements |= {str(e) for e in Composition(p.formula).elements}
            except Exception:
                continue
        volatile_present = [el for el in elements if el in VOLATILE_T]
        if not volatile_present:
            return None  # not applicable -- excluded from group stats, not a free 1.0
        T_max = _route_max_T(predicted)
        if T_max is None:
            return None
        excess = max(max(0.0, T_max - VOLATILE_T[el]) for el in volatile_present)
        return max(0.0, min(1.0, 1.0 - excess / 400.0))

    def _competing_phases(self, chemsys_formulas: list[str]
                          ) -> Optional[list[tuple[str, float]]]:
        """(reduced_formula, e_above_hull) for every PD entry in the
        resolved chemsys, cached per chemsys -- all 8 completions in a
        GRPO group typically share (target, precursor-element-set), so
        this amortizes the PD walk across the group."""
        if self.thermo is None:
            return None
        pd, chemsys = self.thermo._resolve_pd(chemsys_formulas)
        if pd is None:
            return None
        if chemsys in self._competing_phase_cache:
            return self._competing_phase_cache[chemsys]
        out = []
        for e in pd.all_entries:
            try:
                eah = pd.get_e_above_hull(e, on_error="ignore")
            except Exception:
                continue
            if eah is not None:
                out.append((e.composition.reduced_formula, eah))
        self._competing_phase_cache[chemsys] = out
        return out

    def _phase_purity(self, predicted: PredictedRoute,
                      target_formula: str) -> tuple[Optional[float], Optional[int]]:
        """RANKER_SPEC.md 2.5. SIMPLIFICATION FLAG: uses the 0K DFT hull
        (same data target_e_above_hull uses), not a T_max-Gibbs-corrected
        hull for every competing phase -- that would need gibbs-correcting
        every entry in the chemsys, well beyond the existing per-reaction
        Bartel/NIST-JANAF machinery. Directionally right (different
        precursor sets resolve to different chemsys -> different competing
        sets) even at 0K; flag this as a known approximation, not exact.

        Returns (score, n_competing) -- the raw count is surfaced (not just
        the clipped [0,1] score) so rail calibration can pick N_MAX from an
        actual distribution instead of guessing."""
        if not predicted.precursors:
            return None, None
        try:
            target_red = Composition(target_formula).reduced_formula
        except Exception:
            return None, None
        core_formulas = [target_formula] + [p.formula for p in predicted.precursors]
        entries = self._competing_phases(core_formulas)
        if entries is None:
            return None, None
        n_competing = sum(
            1 for formula, eah in entries
            if formula != target_red and eah <= self.scales.purity_window
        )
        score = max(0.0, min(1.0, 1.0 - n_competing / self.scales.n_max))
        return score, n_competing


def load_ranker(
    formula_set_path: Path,
    pd_index_path: Optional[Path],
    project_root: Optional[Path],
    synthesis_clean_path: Path,
    scales: Optional[RankerScales] = None,
) -> Ranker:
    """Mirrors core.reward.load_validator's signature/argument meaning."""
    import pickle
    with formula_set_path.open("rb") as f:
        formula_set = pickle.load(f)
    thermo = None
    if pd_index_path and pd_index_path.exists():
        root = project_root or pd_index_path.parent
        thermo = ThermoChecker.from_sharded_cache(pd_index_path, root)
    freq = build_precursor_frequency(synthesis_clean_path)
    return Ranker(formula_set, thermo, freq, scales=scales)


# ---------------------------------------------------------------------------
# Instrumentation helpers (RANKER_SPEC.md section 4)
# ---------------------------------------------------------------------------

def rail_stats(breakdowns: list[dict]) -> dict[str, dict]:
    """Fraction at exactly 0.0 / exactly 1.0 per objective. >~15% at either
    rail means the scale is mismatched and that channel is wasting gradient
    (RANKER_SPEC.md section 4.2)."""
    out = {}
    for name in OBJECTIVE_NAMES:
        vals = [b[name] for b in breakdowns
                if isinstance(b.get(name), (int, float)) and not isinstance(b.get(name), bool)]
        if not vals:
            out[name] = {"n": 0, "pct_at_0": None, "pct_at_1": None}
            continue
        n = len(vals)
        out[name] = {
            "n": n,
            "pct_at_0": round(100 * sum(1 for v in vals if v <= 1e-9) / n, 1),
            "pct_at_1": round(100 * sum(1 for v in vals if v >= 1 - 1e-9) / n, 1),
            "mean": round(sum(vals) / n, 3),
        }
    return out


def make_ranker_reward_fns(
    ranker: Ranker,
    lit: dict[str, dict],
    dump_path: Optional[str] = None,
    checks: tuple[str, ...] = tuple(OBJECTIVE_NAMES),
    format_weight: float = 0.2,
):
    """
    Per-objective reward functions for GDPO multi-reward training via TRL's
    multi_objective_aggregation, in the exact shape
    core.reward.make_check_reward_fns returns: (reward_funcs, reward_names,
    reward_weights) for GRPOTrainer(reward_funcs=...) /
    GRPOConfig(reward_weights=...). Arm B's drop-in analog of that function
    -- the verifier is the only thing that should differ between arms, so
    this mirrors its caching, dump-to-JSONL, None-propagation, and
    within-group-std logging behavior line for line.

    Unlike make_check_reward_fns, no `*_gradeability` sentinel lookup is
    needed here -- Ranker.score() already returns None directly for every
    ungradeable/inapplicable objective, so `bd.get(check)` is the whole
    story.

    `lit`: the full load_literature() table (misc/kononova_triage_results3.json
    x data/raw/synthesis_clean.json), looked up per-target inside the reward
    closure -- training sees many more targets than any fixed probe set, so
    this is loaded once here rather than baked into the dataset.
    """
    cache: dict[tuple[str, str], Optional[dict]] = {}
    dump_fp = open(dump_path, "a", buffering=1) if dump_path else None

    def bank(completions, target_formula, trainer_state=None, **kwargs):
        from core.reward import parse_completion  # local: core.reward imports
                                                    # this module transitively
                                                    # via validator; avoid a cycle
        out = []
        new = []
        for c, t in zip(completions, target_formula):
            key = (c, t)
            if key not in cache:
                try:
                    route = parse_completion(c, t)
                except Exception:
                    route = None
                try:
                    lit_rec = lit.get(t, {})
                    _, bd = ranker.score(route, t, lit_T=lit_rec.get("max_T"),
                                        lit_n_ops=lit_rec.get("n_ops"))
                    cache[key] = bd
                except Exception:
                    cache[key] = None
                new.append((c, t, cache[key]))
            out.append(cache[key])
        if dump_fp is not None and new:
            step = getattr(trainer_state, "global_step", None)
            for c, t, bd in new:
                dump_fp.write(json.dumps(
                    {"step": step, "target": t, "completion": c, "breakdown": bd},
                    default=str) + "\n")
        return out

    checks = list(checks)
    missing = [c for c in checks if c not in OBJECTIVE_NAMES]
    if missing:
        raise ValueError(
            f"requested ranker reward checks {missing} not in OBJECTIVE_NAMES "
            f"({OBJECTIVE_NAMES}) -- refusing to run with a silently "
            f"truncated/misnamed reward vector")

    def format_ok(completions, target_formula, **kwargs):
        bds = bank(completions, target_formula, **kwargs)
        # ranker.score(None, ...) returns a real dict (reward 0.0, every
        # gate False) rather than None -- unlike make_check_reward_fns'
        # bank(), a parse failure here is NOT "bd is None" (that only
        # happens on a genuine exception inside ranker.score() itself).
        # gate_format_ok is the actual "did this parse into a sane route"
        # signal (Ranker._gate_format_ok: nonempty precursors + operations).
        return [0.0 if (bd is None or not bd.get("gate_format_ok")) else 1.0
               for bd in bds]
    format_ok.__name__ = "format_ok"

    def make_fn(check: str):
        def fn(completions, target_formula, log_metric=None, **kwargs):
            bds = bank(completions, target_formula, **kwargs)
            vals = []
            for bd in bds:
                if bd is None:
                    vals.append(None)
                else:
                    v = bd.get(check)
                    vals.append(float(v) if isinstance(v, (int, float))
                               and not isinstance(v, bool) else None)
            if log_metric is not None:
                groups: dict[str, list[float]] = {}
                for v, t in zip(vals, target_formula):
                    if v is not None:
                        groups.setdefault(t, []).append(v)
                stds = [stdev(g) for g in groups.values() if len(g) >= 2]
                if stds:
                    log_metric(f"within_group_std/{check}", sum(stds) / len(stds))
            return vals
        fn.__name__ = f"check_{check}"
        return fn

    funcs = [format_ok] + [make_fn(c) for c in checks]
    names = ["format_ok"] + checks
    weights = [format_weight] + [1.0] * len(checks)
    return funcs, names, weights


def gate_failure_rates(breakdowns: list[dict]) -> dict[str, float]:
    """Failure rate per gate. RANKER_SPEC.md section 4.3: if any gate fails
    more than a few percent, it is an objective in disguise and should move
    out of the gate set."""
    out = {}
    for g in GATE_NAMES:
        key = f"gate_{g}"
        vals = [b[key] for b in breakdowns if key in b]
        if not vals:
            out[g] = None
            continue
        out[g] = round(100 * (1 - sum(1 for v in vals if v) / len(vals)), 1)
    return out
