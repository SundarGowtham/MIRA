#!/usr/bin/env python3
"""Stats for deepseek/deepseek-v4-pro rows in synthesis_with_traces.jsonl."""

import json
from pathlib import Path
from rich import print as rprint

JSONL = Path("data/processed/synthesis_with_traces.jsonl")
MODEL1 = "deepseek/deepseek-v4-pro"
MODEL2 = "deepseek/deepseek-r1"

def is_true(val) -> bool:
    if val is True:
        return True
    if isinstance(val, str) and val.lower() in ("true", "1", "yes"):
        return True
    return False

def stats_for_model(MODEL:str):
    total = 0
    passed = 0
    passed_fallback = 0
    null_reasoning = 0
    non_null_reasoning = 0
    null_reasoning_arr = []
    non_null_reasoning_arr = []
    
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("generator") != MODEL:
                continue
            total += 1

            

            if rec.get("reasoning_raw") == None:
                null_reasoning += 1
                null_reasoning_arr.append(rec)
            else:
                non_null_reasoning += 1
                non_null_reasoning_arr.append(rec)



            if is_true(rec.get("passed_validator")):
                passed += 1
                if is_true(rec.get("used_fallback")):
                    passed_fallback += 1

    return {
        "total": total,
        "passed": passed,
        "passed_fallback": passed_fallback,
        "null_reasoning": null_reasoning,
        "non_null_reasoning": non_null_reasoning,
        "null_reasoning_arr": null_reasoning_arr,
        "non_null_reasoning_arr": non_null_reasoning_arr,
    }


M1 = stats_for_model(MODEL1)

print(f'total: {M1["total"]}')
print(f'passed: {M1["passed"]}')
print(f'passed_fallback: {M1["passed_fallback"]}')
print(f'null_reasoning: {M1["null_reasoning"]}')
print(f'non_null_reasoning: {M1["non_null_reasoning"]}')
if len(M1["null_reasoning_arr"]):
    rprint(M1["null_reasoning_arr"][0])
print('+'*100)
if len(M1["non_null_reasoning_arr"]):
    rprint(M1["non_null_reasoning_arr"][0])

print("="*100)
print("="*100)
print("="*100)
M2 = stats_for_model(MODEL2)


print(f'total: {M2["total"]}')
print(f'passed: {M2["passed"]}')
print(f'passed_fallback: {M2["passed_fallback"]}')
print(f'null_reasoning: {M2["null_reasoning"]}')
print(f'non_null_reasoning: {M2["non_null_reasoning"]}')
if len(M2["null_reasoning_arr"]):
    rprint(M2["null_reasoning_arr"][0])

print('+'*100)
if len(M2["non_null_reasoning_arr"]):
    rprint(M2["non_null_reasoning_arr"][0])