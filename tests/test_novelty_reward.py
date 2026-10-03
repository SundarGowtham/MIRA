"""
Regression tests for core/novelty_reward.py (Phase 16 §3.2) — needs the
real sharded PD cache, like test_comparator.py; run in tmux per repo
convention.

Run:  uv run python tests/test_novelty_reward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator import (  # noqa: E402
    PredictedConditions,
    PredictedOperation,
    PredictedPrecursor,
    PredictedRoute,
    SynthesisValidator,
    ThermoChecker,
)
from core.novelty_reward import (  # noqa: E402
    NoveltyReward,
    load_counts,
    rarity,
    set_novelty,
)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def route(target, precursors, ops=None):
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(f, a) for f, a in precursors],
        operations=ops or [],
    )


def op(t, atm=None, temp=None):
    return PredictedOperation(
        type=t,
        conditions=PredictedConditions(
            heating_atmosphere=[atm] if atm else [],
            heating_temperature=[temp] if temp is not None else [],
        ),
    )


print("== rarity() and set_novelty() in isolation ==")
freq, target_sets = load_counts()
li2co3_rarity = rarity(freq.get(SynthesisValidator._normalize_formula("Li2CO3"), 0))
libo2_rarity = rarity(freq.get(SynthesisValidator._normalize_formula("LiBO2"), 0))
check("Li2CO3 gets low rarity, LiBO2 gets high rarity",
      li2co3_rarity < 0.5 < libo2_rarity,
      f"Li2CO3={li2co3_rarity:.3f} LiBO2={libo2_rarity:.3f}")
check("never-seen precursor gets rarity exactly 1.0", rarity(0) == 1.0)

check("set_novelty: an exact set pulled from the corpus's own BaTiO3 entries -> 0.0",
      set_novelty(["BaCO3", "TiO2"], "BaTiO3", target_sets) == 0.0,
      str(target_sets.get("BaTiO3", [])[:3]))
check("set_novelty: target absent from the corpus -> 0.5",
      set_novelty(["Zz9Qq2"], "Zz9Qq2Rr3", target_sets) == 0.5)
check("set_novelty: a never-before-seen precursor set for a KNOWN target -> 1.0",
      set_novelty(["Zz9Qq2WeirdPrecursor"], "BaTiO3", target_sets) == 1.0)

print("== NoveltyReward construction guards ==")
import pickle  # noqa: E402
with open("data/cache/mp_formula_set.pkl", "rb") as f:
    formula_set = pickle.load(f)
thermo = ThermoChecker.from_sharded_cache(Path("data/cache/pd_index.json"), Path("."))

try:
    NoveltyReward(SynthesisValidator(formula_set, thermo_checker=None, validator_version=2),
                 freq, target_sets)
    no_thermo_rejected = False
except ValueError:
    no_thermo_rejected = True
check("rejects a validator with no thermo_checker", no_thermo_rejected)

try:
    NoveltyReward(SynthesisValidator(formula_set, thermo_checker=thermo, validator_version=1),
                 freq, target_sets)
    v1_rejected = False
except ValueError:
    v1_rejected = True
check("rejects validator_version=1 (would reject real ammonium routes for a software reason)",
      v1_rejected)

v2_validator = SynthesisValidator(formula_set, thermo_checker=thermo, validator_version=2)
nr = NoveltyReward(v2_validator, freq, target_sets)

print("== score(): gate + novelty, real thermo cache ==")
unbalanced = route("LiFePO4", [("K2CO3", 1.0)], ops=[op("calcine", temp=700)])
reward, info = nr.score(unbalanced, "LiFePO4")
check("an unbalanced route gets reward 0", reward == 0.0 and info["gate"] is False
      and info["gate_stoichiometry"] is False, str(info))

# LiTiO5: not in data/raw/synthesis_clean.json, not in Materials Project --
# a genuinely invented/absurd compound, verified directly (not assumed)
# before writing this test. 1 LiTiO5 + 0.5 Li2O -> Li2TiO3 + 1.25 O2 is
# element-balanced (so this is NOT just re-testing the unbalanced case),
# but the precursor itself should fail reagent plausibility.
invented = route("Li2TiO3", [("LiTiO5", 1.0), ("Li2O", 0.5)],
                 ops=[op("calcine", temp=900)])
reward_inv, info_inv = nr.score(invented, "Li2TiO3")
check("a route with an invented/absurd compound gets reward 0", reward_inv == 0.0, str(info_inv))
check("...specifically because it's absent from the corpus and MP, not a balance failure",
      info_inv["gate_stoichiometry"] is True and info_inv["gate_reagent_plausible"] is False,
      str(info_inv))

# BaO + TiO2 -> BaTiO3: real, balanced, no gas, confirmed downhill (Task 2c
# precedent) -- the full validity gate should pass and a novelty score in
# [0, 1] should be computed.
plausible = route("BaTiO3", [("BaO", 1.0), ("TiO2", 1.0)], ops=[op("calcine", temp=1100)])
reward_pl, info_pl = nr.score(plausible, "BaTiO3")
check("a real, balanced, downhill route passes the gate", info_pl["gate"] is True, str(info_pl))
check("reward is in [0.5, 1.0] when the gate passes (R = G*(0.5+0.5N), G=1)",
      0.5 <= reward_pl <= 1.0, f"reward={reward_pl}")
check("novelty_N is in [0, 1]", 0.0 <= info_pl["novelty_N"] <= 1.0)

print("== degenerate inputs ==")
reward_none, info_none = nr.score(None, "BaTiO3")
check("predicted=None -> reward 0, gate False", reward_none == 0.0 and info_none["gate"] is False)

empty_ops = route("BaTiO3", [("BaO", 1.0), ("TiO2", 1.0)], ops=[])
reward_empty, info_empty = nr.score(empty_ops, "BaTiO3")
check("empty operations -> reward 0 (treated as a parse-failure-shaped route)",
      reward_empty == 0.0)


if __name__ == "__main__":
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("All tests passed.")
