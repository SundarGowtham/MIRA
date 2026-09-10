"""
Regression tests for core/reward.py's tolerant JSON parsing.

Added 2026-09-09 after the Phase 12 smoke gate found 18.2% of completions
failing to parse, traced to two root causes:
  1. Fraction-literal numbers ("amount": 5/12) -- invalid JSON syntax, model
     writes them disproportionately on fractional/doped-composition targets.
  2. The naive first-'{'-to-last-'}' span crossing multiple distinct
     top-level JSON objects ("Extra data" errors).
Both are now repaired before falling back to ParseFailure. Genuine
truncation (unbalanced braces -- generation hit max_new_tokens before the
JSON closed) must NOT be silently "fixed" into a fabricated route; it's a
different problem (completion length) and must still raise ParseFailure.

Run:  uv run python tests/test_reward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reward import (  # noqa: E402
    ParseFailure, parse_completion, _repair_fraction_literals,
    _balanced_json_objects, _try_parse_json_object,
    make_check_reward_fns, PARSE_FAIL_ALERT_THRESHOLD, PARSE_FAIL_ALERT_MIN_N,
)
from validator import SynthesisValidator  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


print("== _repair_fraction_literals ==")

check("simple fraction repaired",
      _repair_fraction_literals('{"amount": 5/12}') == '{"amount": 0.4166666666666667}')
check("fraction with trailing comma repaired",
      _repair_fraction_literals('{"amount": 1/8, "x": 1}') == '{"amount": 0.125, "x": 1}')
check("division by 1 repaired",
      _repair_fraction_literals('{"x": 0.75 / 1}') == '{"x": 0.75}')
check("does not touch a string value containing a slash",
      _repair_fraction_literals('{"note": "2/3 conversion"}') ==
      '{"note": "2/3 conversion"}')
check("division by zero left alone (not silently dropped)",
      "5/0" in _repair_fraction_literals('{"amount": 5/0}'))

print("== _balanced_json_objects ==")

check("finds two distinct top-level objects",
      len(_balanced_json_objects('{"a": 1}  {"b": 2}')) == 2)
check("ignores braces inside strings",
      _balanced_json_objects('{"a": "contains { and }"}') ==
      ['{"a": "contains { and }"}'])
check("picks up nested object as part of the outer one, not separately",
      _balanced_json_objects('{"a": {"b": 1}}') == ['{"a": {"b": 1}}'])

print("== parse_completion: recovers real Phase-12-smoke failure patterns ==")

fraction_completion = (
    '<think>ok</think>\n{\n  "precursors": [\n'
    '    {"formula": "Na2O", "amount": 5/12},\n'
    '    {"formula": "MnO2", "amount": 3/4}\n'
    '  ],\n  "operations": [{"type": "calcine", "temperature_c": 900}]\n}'
)
route = parse_completion(fraction_completion, "X")
check("fraction-literal completion parses",
      len(route.precursors) == 2, f"got {route.precursors}")
check("fraction value correctly converted",
      abs(route.precursors[0].amount - 5 / 12) < 1e-9)

extra_data_completion = (
    '<think>let me think</think>\n'
    '{"example": "ignore this one, no precursors here"}\n'
    'Actually here is my real answer:\n'
    '{"precursors": [{"formula": "BaO", "amount": 1.0}], '
    '"operations": [{"type": "mix", "temperature_c": 800}]}'
)
route2 = parse_completion(extra_data_completion, "X")
check("multi-object completion recovers the route-shaped candidate",
      len(route2.precursors) == 1 and route2.precursors[0].formula == "BaO",
      f"got {route2.precursors}")

print("== parse_completion: genuine truncation still raises, not silently accepted ==")

truncated_mid_think = "<think>Let's balance the reaction... I need to consider"
try:
    parse_completion(truncated_mid_think, "X")
    check("truncated-mid-think raises ParseFailure", False, "did not raise")
except ParseFailure:
    check("truncated-mid-think raises ParseFailure", True)

truncated_mid_json = (
    '<think>ok</think>\n{\n  "precursors": [\n'
    '    {"formula": "Na2O", "amount": 0.5},\n'
    '    {"formula": "MnO2"'
)
try:
    parse_completion(truncated_mid_json, "X")
    check("truncated-mid-json raises ParseFailure", False, "did not raise")
except ParseFailure:
    check("truncated-mid-json raises ParseFailure", True)

print("== _try_parse_json_object ==")

check("returns None (not a crash) on unparseable input",
      _try_parse_json_object("{not json at all") is None)
check("returns a dict for straightforwardly valid JSON",
      _try_parse_json_object('{"a": 1}') == {"a": 1})

print("== per-stratum parse_fail_rate logging + growth alert (mocked, no real network) ==")

import core.reward as reward_module  # noqa: E402

_alert_calls = []
_orig_alert = reward_module._fire_parse_fail_alert
reward_module._fire_parse_fail_alert = lambda stratum, rate, n_total: _alert_calls.append(
    (stratum, rate, n_total))

validator = SynthesisValidator(mp_formula_set={"BaO", "TiO2"}, thermo_checker=None)
target_strata = {"good_target": "stratumA", "bad_target": "stratumB"}
reward_funcs, _, _ = make_check_reward_fns(
    validator, dump_path=None, target_strata=target_strata,
    checks=("amount_accuracy", "stoichiometry", "operation_order"))

logged = {}
def _log(name, value):
    logged[name] = value

def good_completion(i):
    return (f'<think>ok {i}</think>\n{{"precursors": [{{"formula": "BaO", "amount": 1.0}}], '
           f'"operations": [{{"type": "mix", "temperature_c": {800 + i}}}]}}')

def bad_completion(i):
    return f"<think>never finishes, attempt {i}"

# cache is keyed on the literal completion string (real generations are
# never identical) -- each call below uses distinct text so the bank()
# cache treats every one as a new sample, matching real usage.

# n below PARSE_FAIL_ALERT_MIN_N -- must NOT alert yet even at 100% failure.
n_below = PARSE_FAIL_ALERT_MIN_N - 1
for fn in reward_funcs:
    fn(completions=[bad_completion(i) for i in range(n_below)],
      target_formula=["bad_target"] * n_below, log_metric=_log)
check(f"no alert below MIN_N ({PARSE_FAIL_ALERT_MIN_N}) even at 100% failure",
      len(_alert_calls) == 0, f"got {_alert_calls}")
check("parse_fail_rate logged for stratumB even without alerting",
      logged.get("parse_fail_rate/stratumB") == 1.0)

# Cross MIN_N while still over threshold -- must alert exactly once.
for fn in reward_funcs:
    fn(completions=[bad_completion(1000)], target_formula=["bad_target"], log_metric=_log)
check("alert fires exactly once on crossing MIN_N over threshold",
      len(_alert_calls) == 1, f"got {_alert_calls}")
check(f"alert reports rate above threshold ({PARSE_FAIL_ALERT_THRESHOLD})",
      _alert_calls and _alert_calls[0][1] > PARSE_FAIL_ALERT_THRESHOLD)

# Further failures on the same stratum must NOT re-fire (once-per-run guard).
for fn in reward_funcs:
    fn(completions=[bad_completion(2000)], target_formula=["bad_target"], log_metric=_log)
check("does not re-fire on the same stratum", len(_alert_calls) == 1, f"got {_alert_calls}")

# A healthy stratum never crosses the threshold regardless of n.
for fn in reward_funcs:
    fn(completions=[good_completion(i) for i in range(PARSE_FAIL_ALERT_MIN_N)],
      target_formula=["good_target"] * PARSE_FAIL_ALERT_MIN_N, log_metric=_log)
check("healthy stratum never alerts", len(_alert_calls) == 1, f"got {_alert_calls}")
check("healthy stratum's parse_fail_rate logged as 0.0",
      logged.get("parse_fail_rate/stratumA") == 0.0)

reward_module._fire_parse_fail_alert = _orig_alert

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} test(s): {FAILURES}")
    sys.exit(1)
print("All tests passed.")
