"""
clean_traces.py
---------------
Strip bogus low-temperature placeholders from non-heating operations.

Root cause: the Pydantic Operation schema asked the model to set
temperature_c to -1.0 when not applicable, but gpt-oss often emitted
25 (room temperature) as the "not applicable" default instead. The
serializer then preserved that as a real temperature, causing the
validator to flag MixingOperation/StartingSynthesis with 25C as
"implausible".

Fix: For operations that aren't HeatingOperation, only keep temps
above a heating-relevant threshold (default 200C). Below that, treat
the temp as a placeholder and drop it.

Run this on the trace JSONL, then re-run build_sft_v2.py to get
accurate stats and filter behavior.

Usage:
    python clean_traces.py
    python clean_traces.py --in-place    # overwrite instead of writing _clean.jsonl
"""

from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path

DATA_PROCESSED = Path("data/processed")

HEATING_OPS = {"HeatingOperation"}
# Drying can legitimately have a low temp (80-150C). Keep its temps as-is.
# StartingSynthesis, MixingOperation, ShapingOperation, QuenchingOperation:
# anything below 200C is the placeholder bug.
LOW_TEMP_PLACEHOLDER_THRESHOLD = 200.0


def clean_operation(op: dict) -> dict:
    """If op is not a heating op and its temp is below the placeholder threshold,
    strip the heating_temperature/time fields. Drying is exempt."""
    op_type = op.get("type", "")
    if op_type in HEATING_OPS or op_type == "DryingOperation":
        return op
    temps = op.get("heating_temperature") or []
    if not temps:
        return op
    # Check the max temp in this operation
    flat = []
    for t in temps:
        if isinstance(t, list):
            flat.extend([float(x) for x in t if x])
        else:
            try:
                flat.append(float(t))
            except (TypeError, ValueError):
                pass
    if flat and max(flat) < LOW_TEMP_PLACEHOLDER_THRESHOLD:
        out = dict(op)
        out["heating_temperature"] = []
        out["heating_time"] = []
        return out
    return op


def clean_record(r: dict) -> tuple[dict, int]:
    """Returns (cleaned_record, n_temps_stripped)."""
    if "predicted_operations" not in r:
        return r, 0
    new_ops = []
    n_stripped = 0
    for op in r["predicted_operations"]:
        cleaned = clean_operation(op)
        if cleaned is not op:
            n_stripped += 1
        new_ops.append(cleaned)
    r = dict(r)
    r["predicted_operations"] = new_ops
    return r, n_stripped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=DATA_PROCESSED / "reasoning_traces_120B.jsonl")
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <input>_clean.jsonl. Use --in-place to overwrite.")
    p.add_argument("--in-place", action="store_true")
    args = p.parse_args()

    if args.in_place:
        output = args.input
        backup = args.input.with_suffix(
            f".jsonl.bak.{time.strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy(args.input, backup)
        print(f"Backed up to: {backup}")
    else:
        output = args.output or args.input.with_name(
            args.input.stem + "_clean.jsonl"
        )

    n_records = 0
    n_records_with_strip = 0
    total_strips = 0

    cleaned_records = []
    with args.input.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cleaned, n_stripped = clean_record(r)
            cleaned_records.append(cleaned)
            n_records += 1
            if n_stripped > 0:
                n_records_with_strip += 1
                total_strips += n_stripped

    with output.open("w") as f:
        for r in cleaned_records:
            f.write(json.dumps(r) + "\n")

    print(f"Read    {n_records} records from {args.input}")
    print(f"Cleaned {n_records_with_strip} records "
          f"({100*n_records_with_strip/max(n_records,1):.1f}%)")
    print(f"Total operations stripped of placeholder temps: {total_strips}")
    print(f"Wrote   {len(cleaned_records)} records to {output}")


if __name__ == "__main__":
    main()