#!/usr/bin/env python3
"""Stats for deepseek/deepseek-v4-pro rows in synthesis_with_traces.jsonl."""

import json
from pathlib import Path

JSONL = Path("data/processed/synthesis_with_traces.jsonl")
MODEL1 = "deepseek/deepseek-v4-pro"
MODEL2 = "deepseek/deepseek-r1"

def is_true(val) -> bool:
    if val is True:
        return True
    if isinstance(val, str) and val.lower() in ("true", "1", "yes"):
        return True
    return False

def stats_for_model_1():
    total = 0
    passed = 0
    passed_fallback = 0
    
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("generator") != MODEL1:
                continue
            total += 1


            if is_true(rec.get("passed_validator")):
                passed += 1
                if is_true(rec.get("used_fallback")):
                    passed_fallback += 1


# total = passed = passed_fallback = 0

# with JSONL.open() as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         rec = json.loads(line)
#         if rec.get("generator") != MODEL:
#             continue
#         total += 1
#         if is_true(rec.get("passed_validator")):
#             passed += 1
#             if is_true(rec.get("used_fallback")):
#                 passed_fallback += 1

# passed_no_fallback = passed - passed_fallback
# failed = total - passed


# def pct(n: int, denom: int) -> str:
#     if denom == 0:
#         return "n/a"
#     return f"{100.0 * n / denom:.1f}%"


# print(f"generator = {MODEL!r}")
# print(f"  total:                          {total}")
# print()
# print("  Outcomes (of all records for this generator):")
# print(f"    passed, no fallback:            {passed_no_fallback:4d}  ({pct(passed_no_fallback, total)})")
# print(f"    passed, used fallback:          {passed_fallback:4d}  ({pct(passed_fallback, total)})")
# print(f"    failed validator:               {failed:4d}  ({pct(failed, total)})")
# print()
# print("  Validator summary:")
# print(f"    passed_validator=true:          {passed:4d}  ({pct(passed, total)})")
# print(f"    passed_validator=false:         {failed:4d}  ({pct(failed, total)})")
# if passed:
#     print()
#     print("  Among passed only:")
#     print(f"    used_fallback=false:            {passed_no_fallback:4d}  ({pct(passed_no_fallback, passed)})")
#     print(f"    used_fallback=true:             {passed_fallback:4d}  ({pct(passed_fallback, passed)})")