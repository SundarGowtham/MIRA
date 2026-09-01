"""
Regression tests for core/ranker.py — dependency-free except the real PD
cache for the thermo-backed smoke tests (mirrors tests/test_validator.py).

Run:  uv run python tests/test_ranker.py

Every gate gets a pass and fail case (RANKER_SPEC.md section 5, step 1).
Every objective gets a hand-constructed route scoring high, one scoring low
or None, per the same instruction. Plus the None-propagation contract test:
a gate failure must produce None (not 0.0) for every objective.
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
from core.ranker import (  # noqa: E402
    GATE_NAMES,
    OBJECTIVE_NAMES,
    RANKER_VERSION,
    Ranker,
    RankerScales,
    _route_max_T,
    build_precursor_frequency,
    gate_failure_rates,
    rail_stats,
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
            heating_temperature=[temp] if temp is not None else [],
            heating_atmosphere=[atm] if atm else [],
        ),
    )


BATIO3 = route("BaTiO3", [("BaO", 1.0), ("TiO2", 1.0)],
               ops=[op("mix"), op("calcine", atm="air", temp=1000)])

FREQ = {"BaO": 500, "TiO2": 800, "Li2CO3": 5000, "NaCl": 1}
# BaO/TiO2 present in the formula set (precursors_exist gate should pass);
# NaCl deliberately absent (fail case).
RANKER = Ranker(mp_formula_set={"BaO", "TiO2", "Li2CO3"},
                thermo_checker=None, precursor_freq=FREQ)

# ---------------------------------------------------------------------------
print("== gates: pass/fail cases ==")

check("format_ok pass (real route)", RANKER._gate_format_ok(BATIO3))
check("format_ok fail (empty route)",
      not RANKER._gate_format_ok(route("BaTiO3", [])))

check("balances pass (BaO+TiO2 -> BaTiO3)", RANKER._gate_balances(BATIO3))
check("balances fail (NaCl -> BaTiO3, no covering reaction)",
      not RANKER._gate_balances(route("BaTiO3", [("NaCl", 1.0)])))

check("precursors_exist pass (both in formula set)",
      RANKER._gate_precursors_exist(BATIO3))
check("precursors_exist fail (NaCl not in formula set)",
      not RANKER._gate_precursors_exist(route("BaTiO3", [("NaCl", 1.0)])))

check("charge_neutral pass (BaTiO3, known-neutral)",
      RANKER._gate_charge_neutral(BATIO3))
check("charge_neutral fail (NaCl2, no valid assignment)",
      not RANKER._gate_charge_neutral(route("NaCl2", [("NaCl", 2.0)])))

check("temperature_physical pass (1000 C)",
      RANKER._gate_temperature_physical(BATIO3))
check("temperature_physical fail (5000 C, out of range)",
      not RANKER._gate_temperature_physical(
          route("BaTiO3", [("BaO", 1.0), ("TiO2", 1.0)],
                ops=[op("calcine", temp=5000)])))
check("temperature_physical vacuous pass (no heating temp declared)",
      RANKER._gate_temperature_physical(route("BaTiO3", [("BaO", 1.0)], ops=[op("mix")])))

# ---------------------------------------------------------------------------
print("== None-propagation: gate failure -> every objective is None, reward 0.0 ==")

bad_route = route("BaTiO3", [("NaCl", 1.0)])  # fails balances + precursors_exist
reward, info = RANKER.score(bad_route, "BaTiO3", lit_T=1000.0, lit_n_ops=2)
check("gate-failed reward is exactly 0.0", reward == 0.0, f"got {reward}")
check("gate-failed: every objective is None",
      all(info[o] is None for o in OBJECTIVE_NAMES),
      {o: info[o] for o in OBJECTIVE_NAMES})
check("gate-failed: no objective silently paid 0.0 instead of None",
      not any(info[o] == 0.0 for o in OBJECTIVE_NAMES))

reward_none, info_none = RANKER.score(None, "BaTiO3")
check("score(None) is also a clean 0.0/all-None", reward_none == 0.0
      and all(info_none[o] is None for o in OBJECTIVE_NAMES))

# ---------------------------------------------------------------------------
print("== objectives: high / low / None cases ==")

# temperature_economy
r_cold = route("X", [("A", 1.0)], ops=[op("calcine", temp=700)])
r_hot = route("X", [("A", 1.0)], ops=[op("calcine", temp=1400)])
e_cold = RANKER._temperature_economy(r_cold, lit_T=700.0, dG=-0.05)
e_hot = RANKER._temperature_economy(r_hot, lit_T=700.0, dG=-0.05)
check("temperature_economy: cold route scores higher than hot route",
      e_cold is not None and e_hot is not None and e_cold > e_hot,
      f"cold={e_cold} hot={e_hot}")
check("temperature_economy: gated to None when infeasible (dG above cutoff)",
      RANKER._temperature_economy(r_cold, lit_T=700.0, dG=1.0) is None)
check("temperature_economy: None without literature T",
      RANKER._temperature_economy(r_cold, lit_T=None, dG=-0.05) is None)

# step_economy
r_few = route("X", [("A", 1.0)], ops=[op("mix"), op("calcine", temp=800)])
r_many = route("X", [("A", 1.0)], ops=[op("mix")] * 8 + [op("calcine", temp=800)])
s_few = RANKER._step_economy(r_few, lit_n_ops=2)
s_many = RANKER._step_economy(r_many, lit_n_ops=2)
check("step_economy: fewer ops scores higher",
      s_few is not None and s_many is not None and s_few > s_many,
      f"few={s_few} many={s_many}")
check("step_economy: None without literature n_ops",
      RANKER._step_economy(r_few, lit_n_ops=None) is None)

# precursor_availability
r_common = route("X", [("Li2CO3", 1.0)])
r_rare = route("X", [("NaCl", 1.0)])
a_common = RANKER._precursor_availability(r_common)
a_rare = RANKER._precursor_availability(r_rare)
check("precursor_availability: common precursor scores higher than rare",
      a_common is not None and a_rare is not None and a_common > a_rare,
      f"common={a_common} rare={a_rare}")
check("precursor_availability: None for an empty precursor list",
      RANKER._precursor_availability(route("X", [])) is None)

# volatility_risk
r_no_volatile = route("BaTiO3", [("BaO", 1.0), ("TiO2", 1.0)],
                      ops=[op("calcine", temp=1400)])
r_hot_ag = route("Ag2O", [("AgNO3", 1.0)], ops=[op("calcine", temp=1400)])
r_ag_no_temp = route("Ag2O", [("AgNO3", 1.0)], ops=[op("mix")])
check("volatility_risk: None (not applicable, not a free 1.0) when no volatile element present",
      RANKER._volatility_risk(r_no_volatile, "BaTiO3") is None)
v_ag = RANKER._volatility_risk(r_hot_ag, "Ag2O")
check("volatility_risk: penalized when a volatile element (Ag) is hot",
      v_ag is not None and v_ag < 1.0, f"got {v_ag}")
check("volatility_risk: None when volatile element present but no T_max",
      RANKER._volatility_risk(r_ag_no_temp, "Ag2O") is None)

# ---------------------------------------------------------------------------
print("== rail_stats / gate_failure_rates ==")

breakdowns = []
for lit_T in (700.0, 900.0, 1100.0, 1300.0):
    # BaO + TiO2 -> BaTiO3: real, balanceable, both precursors in RANKER's
    # test formula set, so these 4 pass every gate and only bad_route
    # (added below) fails.
    r = route("BaTiO3", [("BaO", 1.0), ("TiO2", 1.0)],
              ops=[op("calcine", temp=1000)])
    _, info = RANKER.score(r, "BaTiO3", lit_T=lit_T, lit_n_ops=1)
    breakdowns.append(info)
_, bad_info = RANKER.score(bad_route, "BaTiO3", lit_T=1000.0, lit_n_ops=2)
breakdowns.append(bad_info)

rs = rail_stats(breakdowns)
check("rail_stats covers every objective", set(rs) == set(OBJECTIVE_NAMES))
gf = gate_failure_rates(breakdowns)
check("gate_failure_rates covers every gate", set(gf) == set(GATE_NAMES))
check("gate_failure_rates: balances failed exactly 1/5 of the batch",
      gf["balances"] == 20.0, f"got {gf['balances']}")

# ---------------------------------------------------------------------------
print("== build_precursor_frequency (real corpus) ==")

freq = build_precursor_frequency(Path("data/raw/synthesis_clean.json"))
check("frequency table is non-empty", len(freq) > 100, f"n={len(freq)}")
check("a common precursor outranks a rare one",
      freq.get(SynthesisValidator._normalize_formula("Li2CO3"), 0) >
      freq.get(SynthesisValidator._normalize_formula("HgO"), 0))

# ---------------------------------------------------------------------------
print("== score() smoke, real thermo cache ==")

thermo = ThermoChecker.from_sharded_cache(Path("data/cache/pd_index.json"), Path("."))
import pickle
with open("data/cache/mp_formula_set.pkl", "rb") as f:
    formula_set = pickle.load(f)
ranker_full = Ranker(formula_set, thermo, freq)
btio3_full = route("BaTiO3", [("BaCO3", 1.0), ("TiO2", 1.0)],
                   ops=[op("mix"), op("calcine", atm="air", temp=1100)])
reward, info = ranker_full.score(btio3_full, "BaTiO3", lit_T=1200.0, lit_n_ops=2)
check("BaTiO3 route scores in [0,1]", 0.0 <= reward <= 1.0, f"reward={reward}")
check("ranker_version stamped", info.get("ranker_version") == RANKER_VERSION)
check("driving_force_margin_gradeability present",
      "driving_force_margin_gradeability" in info)
check("phase_purity excluded from OBJECTIVE_NAMES but still logged inactive",
      "phase_purity" not in OBJECTIVE_NAMES and "phase_purity_INACTIVE" in info)
print(f"    (BaTiO3 reward={reward:.3f}  "
      f"dfm={info.get('driving_force_margin')}  "
      f"phase_purity_INACTIVE={info.get('phase_purity_INACTIVE')}  "
      f"temp_economy={info.get('temperature_economy')})")

# gate-passing route with a nonsense target should still degrade gracefully
weird = route("Zx7Qy2", [("BaCO3", 1.0)], ops=[op("calcine", temp=900)])
r2, info2 = ranker_full.score(weird, "Zx7Qy2", lit_T=900.0, lit_n_ops=1)
check("nonsense-formula route doesn't crash", isinstance(r2, float))

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} test(s): {FAILURES}")
    sys.exit(1)
print("All tests passed.")
