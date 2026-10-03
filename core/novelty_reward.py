"""
core/novelty_reward.py — Phase 16 §3.2 (misc/PHASE16_INSTRUCTIONS.md).

The novelty reward judges validity and novelty ONLY. It contains no
quality heuristics, because quality is judged outside training (ASTRAL
purities, ARROWS³ outcomes) — adding a quality term here would be exactly
the circularity Phase 16's rule 1 (§1) forbids.

Reward: R = G * (0.5 + 0.5 * N)
  G: binary validity gate (§ below). G=0 -> reward 0, no novelty computed.
  N = 0.7 * max_p rarity(p) + 0.3 * set_novelty, in [0, 1].

No path in this module reads misc/astral_validation_set.json,
results/astral_validation_set.json, or ASTRAL's precursor-selection
principles — guarded by tests/test_no_astral_circularity.py, which scans
this directory's own source text as part of its project-wide check.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from validator import PredictedRoute, SynthesisValidator

NOVELTY_REWARD_VERSION = "2026-10-03-v1-phase16"

CORPUS_PATH = Path("data/raw/synthesis_clean.json")
PRECURSOR_FREQ_PATH = Path("data/novelty/precursor_freq.json")
TARGET_SETS_PATH = Path("data/novelty/target_sets.json")

# T in the rarity formula. Precompute asserts the corpus actually has this
# many routes (data provenance it was calibrated against), rather than
# silently using whatever count happens to be found.
TOTAL_CORPUS_ROUTES = 17616

RARITY_WEIGHT = 0.7
SET_NOVELTY_WEIGHT = 0.3
# Elements a Reagent-plausibility precursor may contain beyond the
# target's own elements (§3.2's MP-fallback clause).
EXTRA_ALLOWED_ELEMENTS = frozenset({"O", "H", "C", "N"})
MAX_PRECURSOR_HULL_EV_PER_ATOM = 0.05


# ---------------------------------------------------------------------------
# Precompute: data/novelty/{precursor_freq,target_sets}.json
# ---------------------------------------------------------------------------

def precompute_counts(corpus_path: Path = CORPUS_PATH) -> tuple[dict[str, int], dict[str, list]]:
    """
    precursor_freq: normalized precursor formula -> number of corpus
      routes (data/raw/synthesis_clean.json) containing it at least once.
    target_sets: normalized target formula -> list of distinct precursor
      sets (each a sorted list of normalized formulas) that appear for
      that target anywhere in the corpus.
    Counts use SynthesisValidator._normalize_formula, per the spec.
    """
    data = json.loads(corpus_path.read_text())
    if len(data) != TOTAL_CORPUS_ROUTES:
        raise ValueError(
            f"{corpus_path} has {len(data)} routes, expected {TOTAL_CORPUS_ROUTES} "
            f"(the T the rarity formula was specified against) -- the corpus "
            f"changed under this module's feet; update TOTAL_CORPUS_ROUTES "
            f"deliberately, don't let this pass silently.")

    precursor_freq: dict[str, int] = {}
    target_sets: dict[str, list] = {}
    for entry in data:
        target = SynthesisValidator._normalize_formula(entry["target_formula"])
        prec_formulas = sorted({
            SynthesisValidator._normalize_formula(p["formula"])
            for p in entry.get("precursors", [])
        })
        for p in prec_formulas:
            precursor_freq[p] = precursor_freq.get(p, 0) + 1
        existing = target_sets.setdefault(target, [])
        if prec_formulas not in existing:
            existing.append(prec_formulas)
    return precursor_freq, target_sets


def build_and_save(corpus_path: Path = CORPUS_PATH,
                    freq_out: Path = PRECURSOR_FREQ_PATH,
                    sets_out: Path = TARGET_SETS_PATH) -> None:
    freq, sets_ = precompute_counts(corpus_path)
    freq_out.parent.mkdir(parents=True, exist_ok=True)
    freq_out.write_text(json.dumps(freq, indent=1, sort_keys=True))
    sets_out.write_text(json.dumps(sets_, indent=1, sort_keys=True))
    print(f"-> {freq_out} ({len(freq)} distinct precursors)")
    print(f"-> {sets_out} ({len(sets_)} distinct targets)")


def load_counts(freq_path: Path = PRECURSOR_FREQ_PATH,
                 sets_path: Path = TARGET_SETS_PATH) -> tuple[dict[str, int], dict[str, list]]:
    return (json.loads(freq_path.read_text()), json.loads(sets_path.read_text()))


# ---------------------------------------------------------------------------
# Novelty score N
# ---------------------------------------------------------------------------

def rarity(count: int, total: int = TOTAL_CORPUS_ROUTES) -> float:
    """rarity(p) = clip(log10(T / (count+1)) / log10(T), 0, 1).
    count=0 (never seen) -> 1.0. A precursor seen in every route -> ~0."""
    val = math.log10(total / (count + 1)) / math.log10(total)
    return max(0.0, min(1.0, val))


def set_novelty(precursor_formulas: list[str], target_formula: str,
                 target_sets: dict[str, list]) -> float:
    """1 if this exact precursor set never appears for this target in the
    corpus; 0 if it does; 0.5 if the target isn't in the corpus at all."""
    target = SynthesisValidator._normalize_formula(target_formula)
    if target not in target_sets:
        return 0.5
    this_set = sorted({SynthesisValidator._normalize_formula(p) for p in precursor_formulas})
    return 0.0 if this_set in target_sets[target] else 1.0


# ---------------------------------------------------------------------------
# NoveltyReward: validity gate + novelty score
# ---------------------------------------------------------------------------

class NoveltyReward:
    def __init__(self, validator: SynthesisValidator,
                 precursor_freq: dict[str, int], target_sets: dict[str, list]):
        if validator.thermo_checker is None:
            raise ValueError(
                "NoveltyReward requires a thermo-backed SynthesisValidator "
                "(reaction-energy-sign gate needs thermo_checker)")
        if validator.validator_version < 2:
            raise ValueError(
                "NoveltyReward requires validator_version=2 (the ammonium "
                "balance fix) -- the stoichiometry gate on version 1 would "
                "reject real ammonium-salt routes for a software reason, "
                "not a validity reason.")
        self.validator = validator
        self.precursor_freq = precursor_freq
        self.target_sets = target_sets

    def _reagent_plausible(self, formula: str, target_elements: set[str]) -> bool:
        norm = SynthesisValidator._normalize_formula(formula)
        if self.precursor_freq.get(norm, 0) >= 1:
            return True
        if norm not in self.validator.mp_formula_set:
            return False
        try:
            from pymatgen.core import Composition
            elements = {str(e) for e in Composition(formula).elements}
        except Exception:
            return False
        if not elements <= (target_elements | EXTRA_ALLOWED_ELEMENTS):
            return False
        e_hull = self.validator.thermo_checker.target_e_above_hull(formula)
        return e_hull is not None and e_hull <= MAX_PRECURSOR_HULL_EV_PER_ATOM

    def score(self, predicted: Optional[PredictedRoute],
              target_formula: str) -> tuple[float, dict]:
        info: dict = {"novelty_reward_version": NOVELTY_REWARD_VERSION}

        if predicted is None or not predicted.precursors or not predicted.operations:
            info["gate"] = False
            info["gate_reason"] = "parse_failure_or_empty_route"
            info["reward"] = 0.0
            return 0.0, info

        _, scores = self.validator.validate(predicted, target_formula)
        gate_stoichiometry = scores.get("stoichiometry", 0.0) >= 0.999
        gate_precursors_exist = scores.get("precursors_exist", 0.0) >= 0.999
        gate_temperature_plausible = scores.get("temperature_plausible", 0.0) >= 0.999

        try:
            precursor_pairs = [(p.formula, p.amount) for p in predicted.precursors]
            delta_g, _ = self.validator.thermo_checker.reaction_energy_per_atom(
                precursor_pairs, predicted.target_formula, predicted_route=predicted)
        except Exception:
            delta_g = None
        # Sign only (per spec): ungradeable (delta_g is None) cannot be
        # confirmed downhill, so it does not pass the gate either --
        # a validity gate that can't verify validity should not reward.
        gate_reaction_downhill = delta_g is not None and delta_g <= 0.0

        try:
            from pymatgen.core import Composition
            target_elements = {str(e) for e in Composition(target_formula).elements}
        except Exception:
            target_elements = set()
        gate_reagent_plausible = all(
            self._reagent_plausible(p.formula, target_elements)
            for p in predicted.precursors
        )

        info["gate_stoichiometry"] = gate_stoichiometry
        info["gate_precursors_exist"] = gate_precursors_exist
        info["gate_temperature_plausible"] = gate_temperature_plausible
        info["gate_reaction_downhill"] = gate_reaction_downhill
        info["gate_reaction_downhill_raw_delta_g"] = delta_g
        info["gate_reagent_plausible"] = gate_reagent_plausible

        gate = (gate_stoichiometry and gate_precursors_exist
                and gate_temperature_plausible and gate_reaction_downhill
                and gate_reagent_plausible)
        info["gate"] = gate
        if not gate:
            info["reward"] = 0.0
            return 0.0, info

        rarities = [rarity(self.precursor_freq.get(
            SynthesisValidator._normalize_formula(p.formula), 0))
            for p in predicted.precursors]
        max_rarity = max(rarities) if rarities else 0.0
        set_nov = set_novelty([p.formula for p in predicted.precursors],
                              target_formula, self.target_sets)
        n = RARITY_WEIGHT * max_rarity + SET_NOVELTY_WEIGHT * set_nov
        reward = 1.0 * (0.5 + 0.5 * n)

        info["max_rarity"] = max_rarity
        info["set_novelty"] = set_nov
        info["novelty_N"] = n
        info["reward"] = round(reward, 4)
        return info["reward"], info


def load_novelty_reward(formula_set_path: Path, pd_index_path: Path,
                         project_root: Path,
                         corpus_path: Path = CORPUS_PATH,
                         freq_path: Path = PRECURSOR_FREQ_PATH,
                         sets_path: Path = TARGET_SETS_PATH) -> NoveltyReward:
    """Mirrors core.reward.load_validator's signature/argument meaning."""
    import pickle
    from validator import ThermoChecker
    with formula_set_path.open("rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(pd_index_path, project_root)
    validator = SynthesisValidator(formula_set, thermo_checker=thermo, validator_version=2)
    if freq_path.exists() and sets_path.exists():
        freq, sets_ = load_counts(freq_path, sets_path)
    else:
        freq, sets_ = precompute_counts(corpus_path)
    return NoveltyReward(validator, freq, sets_)


# ---------------------------------------------------------------------------
# GDPO/GRPO reward-function factory, same shape as make_check_reward_fns /
# make_ranker_reward_fns: (reward_funcs, reward_names, reward_weights).
# A single channel -- the novelty reward is one scalar by spec, not a
# factored vector -- so this is the degenerate n=1 case of that interface.
# ---------------------------------------------------------------------------

def make_novelty_reward_fns(novelty_reward: NoveltyReward,
                             dump_path: Optional[str] = None):
    cache: dict[tuple[str, str], Optional[dict]] = {}
    dump_fp = open(dump_path, "a", buffering=1) if dump_path else None

    def novelty(completions, target_formula, trainer_state=None, **kwargs):
        from core.reward import parse_completion
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
                    _, bd = novelty_reward.score(route, t)
                except Exception:
                    bd = {"gate": False, "gate_reason": "exception", "reward": 0.0}
                cache[key] = bd
                new.append((c, t, bd))
            out.append(cache[key])
        if dump_fp is not None and new:
            step = getattr(trainer_state, "global_step", None)
            for c, t, bd in new:
                dump_fp.write(json.dumps(
                    {"step": step, "target": t, "completion": c, "breakdown": bd},
                    default=str) + "\n")
        return [bd.get("reward", 0.0) if bd else 0.0 for bd in out]

    novelty.__name__ = "novelty"
    return [novelty], ["novelty"], [1.0]


if __name__ == "__main__":
    build_and_save()
