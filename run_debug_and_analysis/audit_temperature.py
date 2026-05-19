"""
audit_temperature.py
--------------------
Diagnose why temperature_plausible is at 0.675 on closed-book records.

Three questions to answer:
  1. What fraction of records fail temp_plausible, by how much?
  2. What temperatures is the model emitting in failing records?
  3. How do model temperatures compare to MP's recorded temperatures
     for the same target? (the ground-truth answer)
"""

from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics

TRACE_FILE = Path("data/processed/reasoning_traces_120B_clean.jsonl")


def extract_temps(operations: list[dict]) -> list[float]:
    """Pull all heating temperatures out of an ops list."""
    temps = []
    for op in operations:
        ts = op.get("heating_temperature") or []
        for t in ts:
            if isinstance(t, list):
                temps.extend([float(x) for x in t if x])
            elif t:
                temps.append(float(t))
    return temps


def extract_temps_with_op_type(operations: list[dict]) -> list[tuple[str, float]]:
    """Pull (op_type, temp) pairs to see WHICH ops have the bad temps."""
    out = []
    for op in operations:
        op_type = op.get("type", "Unknown")
        ts = op.get("heating_temperature") or []
        for t in ts:
            if isinstance(t, list):
                for x in t:
                    if x:
                        out.append((op_type, float(x)))
            elif t:
                out.append((op_type, float(t)))
    return out


def main():
    records = [json.loads(line) for line in open(TRACE_FILE)
               if not json.loads(line).get("used_fallback", True)]
    print(f"Loaded {len(records)} closed-book records\n")

    # Question 1: distribution of temp_plausible scores
    scores = [r["validator_breakdown"].get("temperature_plausible", 1.0)
              for r in records
              if r.get("validator_breakdown") and "temperature_plausible" in r["validator_breakdown"]]
    print("temp_plausible score distribution:")
    score_buckets = Counter(round(s, 2) for s in scores)
    for s in sorted(score_buckets.keys()):
        print(f"  {s:.2f}: {score_buckets[s]:6d}  ({100*score_buckets[s]/len(scores):.1f}%)")
    print(f"  mean: {statistics.mean(scores):.4f}")
    print()

    # Question 2: temperatures emitted by op_type, split by pass/fail
    passed = [r for r in records
              if r.get("validator_breakdown", {}).get("temperature_plausible", 0) >= 1.0]
    failed = [r for r in records
              if r.get("validator_breakdown", {}).get("temperature_plausible", 1) < 1.0]
    print(f"Passing temp_plausible: {len(passed)} ({100*len(passed)/len(records):.1f}%)")
    print(f"Failing temp_plausible: {len(failed)} ({100*len(failed)/len(records):.1f}%)")
    print()

    # Question 2a: WHERE are the temperatures in failing records?
    fail_pairs = []
    for r in failed:
        fail_pairs.extend(extract_temps_with_op_type(r["predicted_operations"]))
    pass_pairs = []
    for r in passed:
        pass_pairs.extend(extract_temps_with_op_type(r["predicted_operations"]))

    print("Temperature distribution by op_type (FAILING records):")
    by_op_fail = defaultdict(list)
    for op_type, t in fail_pairs:
        by_op_fail[op_type].append(t)
    for op_type, temps in sorted(by_op_fail.items()):
        print(f"  {op_type:25s} n={len(temps):4d}  "
              f"min={min(temps):6.0f}  med={statistics.median(temps):6.0f}  max={max(temps):6.0f}")
    print()

    print("Temperature distribution by op_type (PASSING records):")
    by_op_pass = defaultdict(list)
    for op_type, t in pass_pairs:
        by_op_pass[op_type].append(t)
    for op_type, temps in sorted(by_op_pass.items()):
        print(f"  {op_type:25s} n={len(temps):4d}  "
              f"min={min(temps):6.0f}  med={statistics.median(temps):6.0f}  max={max(temps):6.0f}")
    print()

    # Question 2b: temperature buckets in failing records (any op type, max temp per record)
    print("Max-temp-per-record distribution (FAILING):")
    max_temps_fail = []
    for r in failed:
        ts = extract_temps(r["predicted_operations"])
        if ts:
            max_temps_fail.append(max(ts))
    buckets = Counter(int(t // 100) * 100 for t in max_temps_fail)
    for b in sorted(buckets.keys()):
        bar = "#" * (buckets[b] // max(1, len(max_temps_fail) // 30))
        print(f"  {b:4d}-{b+99:4d}C: {buckets[b]:5d}  {bar}")
    print()

    # Question 3: model temp vs MP temp for same target
    print("Model max-temp vs MP max-temp for same target (FAILING records):")
    deltas = []
    for r in failed[:500]:  # sample
        model_temps = extract_temps(r["predicted_operations"])
        mp_temps = extract_temps(r.get("mp_operations") or [])
        if not model_temps or not mp_temps:
            continue
        deltas.append((max(model_temps), max(mp_temps), r["target_formula"]))

    if deltas:
        diff = [m - mp for m, mp, _ in deltas]
        print(f"  Sampled {len(deltas)} failing records with MP temperature data")
        print(f"  Mean (model - MP): {statistics.mean(diff):+.0f} C")
        print(f"  Median (model - MP): {statistics.median(diff):+.0f} C")
        print(f"  Model > MP: {sum(1 for d in diff if d > 0)}")
        print(f"  Model < MP: {sum(1 for d in diff if d < 0)}")
        print(f"  |diff| > 200C: {sum(1 for d in diff if abs(d) > 200)}")
        print()
        print("  Largest 5 over-predictions:")
        for m, mp, t in sorted(deltas, key=lambda x: -(x[0]-x[1]))[:5]:
            print(f"    {t:30s}  model={m:.0f}C  MP={mp:.0f}C  diff={m-mp:+.0f}C")
        print("  Largest 5 under-predictions:")
        for m, mp, t in sorted(deltas, key=lambda x: x[0]-x[1])[:5]:
            print(f"    {t:30s}  model={m:.0f}C  MP={mp:.0f}C  diff={m-mp:+.0f}C")
        print()

    # Question 4: how many failing records emit *no* heating temperature at all?
    no_temp = sum(1 for r in failed if not extract_temps(r["predicted_operations"]))
    print(f"Failing records that emit ZERO heating temperatures: {no_temp} "
          f"({100*no_temp/len(failed):.1f}% of failures)")


if __name__ == "__main__":
    main()