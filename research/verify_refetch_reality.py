"""
verify_refetch_reality.py
----------------------------
refetch_pd_shards.py reported 1691/1691 succeeded, including
C-Fe-H-Li-N-O-P and Fe-Li-O-P specifically. But inspect_lfp_chemsys.py,
run immediately after, still shows _resolve_pd failing for those exact
chemsystems. One of these two things is wrong, and this checks both
directly instead of trusting either:

1. What does refetch_log.json ACTUALLY say for these specific
   chemsystems - were they really attempted, and did they really report
   "Success"?
2. What does the file on disk ACTUALLY look like right now - size,
   modification time (was it touched recently, i.e. actually rewritten
   by the refetch run, or is it the same old file), and does a RAW
   pickle.load() (no exception swallowing, no _get_pd abstraction)
   succeed or fail, with the real exception if it fails?

Usage:
  uv run python verify_refetch_reality.py \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --refetch-log refetch_log.json \
      --chemsys C-Fe-H-Li-N-O-P \
      --chemsys C-Fe-H-Li-N-O-P-V \
      --chemsys Fe-Li-O-P
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--refetch-log", type=Path, default=Path("refetch_log.json"))
    ap.add_argument("--chemsys", action="append", required=True)
    args = ap.parse_args()

    pd_index = json.loads(args.pd_index.read_text())

    log_by_chemsys = {}
    if args.refetch_log.exists():
        log_entries = json.loads(args.refetch_log.read_text())
        for entry in log_entries:
            log_by_chemsys[entry["chemsys"]] = entry
        print(f"Loaded {len(log_entries)} entries from {args.refetch_log}\n")
    else:
        print(f"WARNING: {args.refetch_log} not found - can't check what the refetch actually logged.\n")

    now = time.time()

    for chemsys in args.chemsys:
        print(f"=== {chemsys} ===")

        log_entry = log_by_chemsys.get(chemsys)
        if log_entry is None:
            print(f"  refetch_log.json: NEVER ATTEMPTED - this chemsys was not in the queue at all")
        else:
            print(f"  refetch_log.json says: status={log_entry['status']!r}  "
                  f"shard_path={log_entry.get('shard_path')!r}  "
                  f"n_entries={log_entry.get('n_entries')}")

        in_index = chemsys in pd_index
        print(f"  pd_index.json: {'present -> ' + pd_index[chemsys] if in_index else 'ABSENT'}")

        if not in_index:
            print("  Can't check the file - not in pd_index.json at all.\n")
            continue

        shard_path = args.project_root / pd_index[chemsys]
        if not shard_path.exists():
            print(f"  FILE DOES NOT EXIST at {shard_path} - despite pd_index.json pointing at it\n")
            continue

        stat = shard_path.stat()
        age_minutes = (now - stat.st_mtime) / 60
        print(f"  file on disk: {shard_path}")
        print(f"    size: {stat.st_size:,} bytes")
        print(f"    last modified: {age_minutes:.1f} minutes ago")

        try:
            with shard_path.open("rb") as f:
                pd = pickle.load(f)
            print(f"    RAW pickle.load(): SUCCESS - {len(pd.all_entries)} entries")
        except Exception as e:
            print(f"    RAW pickle.load(): FAILED - {type(e).__name__}: {e}")

        print()


if __name__ == "__main__":
    main()