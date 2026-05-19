"""
inspect_pd_cache.py
-------------------
Inspect the phase_diagrams.pkl without loading everything into memory.
Tells us the structure, key format, and size distribution so we can
design the streaming replacement.

Uses pickletools to read the pickle stream without constructing objects,
plus a sample load of just the first few entries.

Usage:
    python inspect_pd_cache.py
    python inspect_pd_cache.py --sample 3   # load 3 PDs to see their structure
"""

from __future__ import annotations
import argparse
import os
import pickle
import struct
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/cache/phase_diagrams.pkl"))
    p.add_argument("--sample", type=int, default=0,
                   help="Load N entries from the cache to inspect their object structure.")
    return p.parse_args()


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    args = parse_args()
    path = args.cache
    if not path.exists():
        raise FileNotFoundError(path)

    size_bytes = path.stat().st_size
    print(f"File: {path}")
    print(f"Size: {human_bytes(size_bytes)}")
    print()

    # Strategy: load the dict in a way that gives us keys without fully
    # constructing all PhaseDiagram objects. Pickle loads lazily per
    # object, so we can read the top-level dict then stop.
    # However, Python's pickle isn't truly lazy — dict.load() builds everything.
    # Best we can do without custom unpickler: load and immediately grab keys.
    #
    # If even loading causes OOM, we can use pickletools to count SETITEM ops.
    print("Loading cache keys (this will use memory proportional to the full file)...")
    print("If this hangs, Ctrl-C and we'll use the pickletools approach instead.")
    t0 = time.time()

    try:
        with path.open("rb") as f:
            pd_cache = pickle.load(f)
        elapsed = time.time() - t0
        print(f"Loaded in {elapsed:.1f}s")
        print()

        print(f"Type: {type(pd_cache)}")
        print(f"Number of entries: {len(pd_cache)}")
        print()

        # Key format
        keys = list(pd_cache.keys())
        print("Key format examples (first 10):")
        for k in keys[:10]:
            print(f"  {repr(k)}")
        print()

        # Check value types
        sample_keys = keys[:3]
        print(f"Value types (first 3 entries):")
        for k in sample_keys:
            v = pd_cache[k]
            print(f"  {k!r}: {type(v).__name__}")
            if hasattr(v, '__dict__'):
                attrs = [a for a in vars(v) if not a.startswith('_')][:8]
                print(f"    attrs: {attrs}")
            # Estimate object size
            try:
                import sys
                sz = sys.getsizeof(v)
                print(f"    shallow size: {human_bytes(sz)}")
            except Exception:
                pass
        print()

        # Key structure analysis
        if keys and isinstance(keys[0], str):
            # Assume keys are like "Ba-O-Ti" (alphabetized hyphen-joined elements)
            element_counts = [len(k.split("-")) for k in keys]
            from collections import Counter
            ectr = Counter(element_counts)
            print("Chemsystem size distribution (number of elements in key):")
            for n_el, cnt in sorted(ectr.items()):
                print(f"  {n_el} elements: {cnt:5d} chemsystems")
            print()

        print(f"Total entries: {len(pd_cache)}")

    except MemoryError:
        print("MemoryError loading the full cache. Falling back to pickletools scan...")
        _scan_with_pickletools(path)
    except KeyboardInterrupt:
        print("\nInterrupted. Falling back to pickletools scan...")
        _scan_with_pickletools(path)


def _scan_with_pickletools(path: Path):
    """Count SETITEM / SETITEMS opcodes to estimate number of entries,
    and extract string keys by looking for SHORT_BINUNICODE / BINUNICODE opcodes."""
    import pickletools
    import io

    print("Scanning pickle stream (no object construction)...")
    keys_found = []
    n_setitem = 0

    with path.open("rb") as f:
        # Read in chunks to avoid loading everything
        data = f.read(10 * 1024 * 1024)  # first 10MB — enough to find key format

    stream = io.BytesIO(data)
    try:
        for opcode, arg, pos in pickletools.genops(stream):
            if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                if isinstance(arg, str) and "-" in arg and len(arg) < 50:
                    keys_found.append(arg)
            if opcode.name in ("SETITEM", "SETITEMS"):
                n_setitem += 1
    except Exception:
        pass

    print(f"Found {n_setitem} SETITEM ops in first 10MB (partial)")
    print(f"Sample string keys (likely chemsys keys):")
    for k in keys_found[:15]:
        print(f"  {k!r}")


if __name__ == "__main__":
    main()