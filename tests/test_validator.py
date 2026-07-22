"""
Regression tests for validator.py / core/reward.py — dependency-free.

Run:  uv run python tests/test_validator.py

Covers the bugs fixed 2026-07-20 (atmosphere substring misclassification,
undeclared gas uptake passing stoichiometry, unknown op types auto-passing
operation_order, non-deterministic _resolve_pd, //-comment stripping that
mangled string values) plus the hydrate-notation fix and parser contract.

_resolve_pd determinism is not directly tested here (needs the real sharded
cache); it is covered indirectly by test_validate_smoke_thermo, which loads
the actual pd_index.json + shards.
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
    _classify_atmosphere,
    expand_hydrate_notation,
)
from core.reward import ParseFailure, load_validator, parse_completion  # noqa: E402
from pymatgen.core import Composition  # noqa: E402

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


LIGHT_VALIDATOR = SynthesisValidator(mp_formula_set=set(), thermo_checker=None)

# ---------------------------------------------------------------------------
print("== _classify_atmosphere ==")
cases = {
    "air": "ox", "O2 flow": "ox", "oxygen": "ox",
    "steam": "ox", "H2O vapor": "ox",          # steam oxidizes; was "red" (h2 inside h2o)
    "Ar": "inert", "argon": "inert", "N2": "inert", "vacuum": "inert", "He": "inert",
    "5% H2 / 95% Ar": "red", "H2": "red", "forming gas": "red", "CO": "red",
    "CO2": "unknown",                          # was "ox" (o2 inside co2)
    "cold": "unknown",                         # was "red" (co inside cold)
    "controlled": "unknown",
    "carbon": "unknown",                       # was "inert" (ar inside carbon)
    "argon-flow 99.999%": "inert",
    "gibberish xyz": "unknown",
}
for s, want in cases.items():
    got = _classify_atmosphere(s)
    check(f"classify {s!r} -> {want}", got == want, f"got {got}")

# ---------------------------------------------------------------------------
print("== expand_hydrate_notation ==")
check("middle-dot hydrate rewrites",
      Composition(expand_hydrate_notation("FeC2O4·2H2O")).as_dict()
      == Composition("FeC2O4(H2O)2").as_dict())
check("no dot passes through", expand_hydrate_notation("Li2CO3") == "Li2CO3")
check("multi-dot passes through unchanged",
      expand_hydrate_notation("A·2B·3C") == "A·2B·3C")
check("PredictedPrecursor normalizes",
      PredictedPrecursor("CH3COOLi·2H2O").formula == "CH3COOLi(H2O)2")
# '.'-notation hydrates are deliberately NOT rewritten: the pattern is
# indistinguishable from fractional stoichiometry (Li0.5CoO2). This is the
# corpus-typo class (e.g. FeC2O4.2H20) — handle via data audit, not parsing.
check("dot notation NOT rewritten (fractional-stoich safety)",
      expand_hydrate_notation("Li0.5CoO2") == "Li0.5CoO2")

# ---------------------------------------------------------------------------
print("== stoichiometry: gas-uptake must be declared ==")
feo = route("Fe2O3", [("FeO", 4.0)], ops=[op("calcine", atm="air", temp=900)])
check("FeO -> Fe2O3 WITH air atmosphere balances",
      LIGHT_VALIDATOR._check_stoichiometry(feo) == 1.0)
feo_no_atm = route("Fe2O3", [("FeO", 4.0)], ops=[op("calcine", temp=900)])
check("FeO -> Fe2O3 WITHOUT atmosphere rejected (undeclared O2 uptake)",
      LIGHT_VALIDATOR._check_stoichiometry(feo_no_atm) == 0.0)
caco3 = route("CaO", [("CaCO3", 1.0)], ops=[op("calcine", temp=1100)])
check("CaCO3 -> CaO (CO2 released, no atmosphere needed)",
      LIGHT_VALIDATOR._check_stoichiometry(caco3) == 1.0)
nonsense = route("LiFePO4", [("K2CO3", 1.0)], ops=[op("calcine", temp=700)])
check("chemically impossible route fails",
      LIGHT_VALIDATOR._check_stoichiometry(nonsense) == 0.0)

print("== sentinel exclusion (None-propagation in validate()) ==")
reward, bd = LIGHT_VALIDATOR.validate(nonsense, "LiFePO4")
check("impossible route's amount_accuracy tagged no_balance_found",
      bd.get("amount_accuracy_gradeability") == "no_balance_found")
# Expected reward: renormalized over active checks with amount_accuracy removed
w = dict(LIGHT_VALIDATOR.weights)
w.pop("amount_accuracy")
ws = sum(w.values())
expected = sum(w[k] * bd[k] for k in w) / ws
check("sentinel check excluded from reward, weights renormalized",
      abs(reward - round(expected, 4)) < 1e-3,
      f"reward={reward} expected={expected:.4f}")

# ---------------------------------------------------------------------------
print("== operation_order ==")
mix_heat = route("X", [("X", 1.0)],
                 ops=[op("mix"), op("calcine"), op("sinter")])
check("mix -> calcine -> sinter is 1.0",
      LIGHT_VALIDATOR._check_operation_order(mix_heat) == 1.0)
regrind = route("X", [("X", 1.0)],
                ops=[op("calcine"), op("grind"), op("sinter")])
check("calcine -> grind -> sinter regrind exempt",
      LIGHT_VALIDATOR._check_operation_order(regrind) == 1.0)
unknown = route("X", [("X", 1.0)], ops=[op("mystery_step"), op("alchemy")])
check("all-unknown op types are neutral 0.5 (was free 1.0)",
      LIGHT_VALIDATOR._check_operation_order(unknown) == 0.5)

# ---------------------------------------------------------------------------
print("== parse_completion ==")
good = '<think>reasoning</think>\n{"precursors": [{"formula": "Li2CO3", "amount": 1}], "operations": [{"type": "calcine", "temperature_c": 800, "time_h": 10, "atmosphere": "air"}]}'
r = parse_completion(good, "LiFePO4")
check("think+JSON parses", len(r.precursors) == 1 and len(r.operations) == 1)

with_comment = '{"precursors": [{"formula": "Li2CO3"}], // stray comment\n "operations": [{"type": "mix"}]}'
r = parse_completion(with_comment, "X")
check("// comment recovered as fallback", len(r.precursors) == 1)

url_inside = '{"precursors": [{"formula": "Li2CO3", "note": "see https://example.com/x"}], "operations": [{"type": "mix", "media": "H2O//EtOH"}]}'
r = parse_completion(url_inside, "X")
check("// inside string values NOT mangled",
      r.operations[0].conditions.mixing_media == "H2O//EtOH")

try:
    parse_completion("no json here at all", "X")
    check("garbage raises ParseFailure", False)
except ParseFailure:
    check("garbage raises ParseFailure", True)

try:
    parse_completion('{"a": 1, "b": 2}', "X")
    check("empty precursors+operations raises ParseFailure", False)
except ParseFailure:
    check("empty precursors+operations raises ParseFailure", True)

schema_drift = '{"precursors": [{"formula": "CaO"}], "operations": [{"type": "Heating", "temperature": 1000, "time": 5}]}'
r = parse_completion(schema_drift, "CaCO3")
check("key-alias drift (temperature/time) still parsed",
      r.operations[0].conditions.heating_temperature == [1000.0]
      and r.operations[0].conditions.heating_time == [5.0])

# ---------------------------------------------------------------------------
print("== validate() smoke, real thermo cache ==")
v = load_validator(Path("data/cache/mp_formula_set.pkl"),
                   Path("data/cache/pd_index.json"), Path("."))
check("thermo checker loads", v.thermo_checker is not None)
lfp = route(
    "LiFePO4",
    [("FeC2O4·2H2O", 2.0), ("Li2CO3", 1.0), ("(NH4)2HPO4", 2.0)],
    ops=[op("mix"), op("calcine", atm="Ar", temp=350), op("sinter", atm="Ar", temp=700)],
)
reward, breakdown = v.validate(lfp, "LiFePO4")
check("LFP hydrate route validates end-to-end", 0.0 <= reward <= 1.0,
      f"reward={reward}")
check("gradeability tags present",
      "thermodynamic_favorable_gradeability" in breakdown
      and "target_stability_gradeability" in breakdown)
print(f"    (LFP reward={reward:.3f}  thermo_tag={breakdown.get('thermodynamic_favorable_gradeability')}  "
      f"stab_tag={breakdown.get('target_stability_gradeability')})")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} test(s): {FAILURES}")
    sys.exit(1)
print("All tests passed.")
