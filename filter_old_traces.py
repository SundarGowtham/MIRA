"""
filter_old_traces.py
--------------------
Filter the reasoning_traces.jsonl to remove records from prior buggy
generator versions (the un-guided run that produced 626 empty-precursor
records all scoring exactly 0.45).

Keeps only records where generator == "ray-gpt-oss-20b-guided" (the new
guided_json version). The Qwen3 HF fallback records are also dropped since
they were generated under the same flawed-prompt assumption.

Backs up the original file before filtering. Run this BEFORE relaunching
the new generator so resume logic doesn't skip the broken records.

Usage:
    python filter_old_traces.py
    python filter_old_traces.py --keep-fallback   # keep Qwen3 fallback records
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        type=Path,
        default=Path("data/processed/reasoning_traces.jsonl"),
    )
    p.add_argument(
        "--keep-fallback",
        action="store_true",
        help="Keep records with generator='hf' (Qwen3 fallback). "
             "Off by default; those records had the same prompt-format bug.",
    )
    args = p.parse_args()

    if not args.path.exists():
        print(f"{args.path} doesn't exist. Nothing to filter.")
        return

    # Backup
    backup = args.path.with_suffix(
        f".jsonl.bak.{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy(args.path, backup)
    print(f"Backed up to: {backup}")

    # Read + filter
    keep = []
    drop_counts = {}
    with args.path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            gen = r.get("generator", "unknown")
            if gen == "ray-gpt-oss-20b-guided":
                keep.append(r)
            elif gen == "hf" and args.keep_fallback:
                keep.append(r)
            else:
                drop_counts[gen] = drop_counts.get(gen, 0) + 1

    # Write back
    with args.path.open("w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")

    print(f"Kept:  {len(keep)} records")
    print(f"Dropped by generator:")
    for gen, n in sorted(drop_counts.items(), key=lambda x: -x[1]):
        print(f"  {gen}: {n}")


if __name__ == "__main__":
    main()