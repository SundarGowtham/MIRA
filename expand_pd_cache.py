"""
expand_pd_cache.py
------------------
Extends the existing PD cache by building C and N extension chemsystems
for every oxide target that doesn't already have them.

WHY THIS IS NEEDED
------------------
data_pull_3.py already builds PDs that union target + precursor elements,
so if a Kononova record shows SrCO3 as a precursor for SrTiO3, "C-O-Sr-Ti"
is already in the cache. But there are two remaining gap types:

  1. Records where the Kononova data used oxide precursors (SrO, not SrCO3)
     → the cache has O-Sr-Ti but not C-O-Sr-Ti. When the LLM proposes
     SrCO3 (which it almost always will for hygroscopic alkaline earths),
     the validator needs C-O-Sr-Ti to compute the reaction energy.

  2. Records where validator's _resolve_pd was being called with the wrong
     chemsys (a pre-existing validator bug that has now been fixed), meaning
     some C-containing shards might have been missed entirely.

WHAT THIS SCRIPT DOES
---------------------
For every chemsys currently in pd_index where:
  - O is present (it's an oxide system)
  - Adding "C" doesn't exceed PD_MAX_ELEMENTS
  → build chemsys+C if not already in index

For every chemsys currently in pd_index where:
  - O is present and N is not already present
  - Adding "N" doesn't exceed PD_MAX_ELEMENTS
  → build chemsys+N if not already in index

Also scans synthesis_clean.json (if present) for any additional
(target + precursors) chemsystems not yet covered.

After running this + deploying the validator fix, thermodynamic_favorable
should return real values instead of 0.5 for carbonate and nitrate routes.

Usage:
    MP_API_KEY=... python expand_pd_cache.py
    MP_API_KEY=... python expand_pd_cache.py --dry-run      # show counts only
    MP_API_KEY=... python expand_pd_cache.py --workers 12   # default: cpu-2
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import time
import multiprocessing
import concurrent.futures
import warnings
from pathlib import Path

# Suppress noisy-but-harmless pymatgen/spglib internal warnings.
# These come from edge-case crystal symmetry analysis and mixing-scheme
# oxidation-state guessing; they don't affect PD correctness.
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", message="ssm_get_exact_positions")
warnings.filterwarnings("ignore", message="get_bravais_exact_positions")

import dotenv
from pymatgen.core import Composition

dotenv.load_dotenv()

# ── Match data_pull_3.py exactly ─────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
DATA_CACHE    = PROJECT_ROOT / "data" / "cache"
PD_SHARDS_DIR = DATA_CACHE / "pd_shards"
PD_INDEX_FILE = DATA_CACHE / "pd_index.json"
SYNTHESIS_FILE = PROJECT_ROOT / "data" / "raw" / "synthesis_clean.json"

PD_MAX_ELEMENTS = 8   # same cap as data_pull_3
PD_TIMEOUT_SECS = 600

MP_API_KEY = os.environ.get("MP_API_KEY")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Retry decorator (standalone copy — can't import from data_pull_3) ─────────
def with_retry(max_attempts: int = 5, base_delay: float = 5.0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
        return wrapper
    return decorator


# ── Worker (runs in isolated subprocess, same as data_pull_3) ─────────────────
def process_single_chemsys(chemsys: str):
    """
    Fetch entries, build PD, write pickle shard.
    Must be fully self-contained — runs in a subprocess.
    """
    import os, pickle, signal
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

    class _TimeoutError(Exception): pass
    def _timeout_handler(signum, frame): raise _TimeoutError("PD build timed out")

    scheme = MaterialsProjectDFTMixingScheme()

    def _with_retry(fn, max_attempts=5, base_delay=5.0):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    time.sleep(base_delay * (2 ** (attempt - 1)))
        return wrapper

    @_with_retry
    def fetch():
        with MPRester(os.environ.get("MP_API_KEY")) as mpr:
            return mpr.get_entries_in_chemsys(
                elements=chemsys.split("-"),
                additional_criteria={"thermo_types": ["GGA_GGA+U", "R2SCAN"]},
            )

    try:
        entries = fetch()
        entries = scheme.process_entries(entries)
        if not entries:
            return chemsys, None, 0, "no_entries"

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(PD_TIMEOUT_SECS)
        try:
            pd = PhaseDiagram(entries)
        finally:
            signal.alarm(0)

        shard_path = PD_SHARDS_DIR / f"{chemsys}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(pd, f)

        return chemsys, str(shard_path.relative_to(PROJECT_ROOT)), len(entries), "success"

    except _TimeoutError:
        return chemsys, None, 0, "timeout"
    except Exception as e:
        return chemsys, None, 0, f"error: {e}"


# ── Gap analysis ─────────────────────────────────────────────────────────────
def get_chemsys(formula: str) -> str | None:
    try:
        return "-".join(sorted(str(e) for e in Composition(formula).elements))
    except Exception:
        return None


def is_valid_concrete(formula: str | None) -> bool:
    if not formula:
        return False
    try:
        Composition(formula)
        return True
    except Exception:
        return False


def compute_needed_chemsystems(pd_index: dict, synthesis_path: Path) -> set[str]:
    """
    Return the set of chemsystems that SHOULD be in the cache but aren't.

    Two sources:
      1. +C and +N extensions of every existing oxide chemsys with room.
      2. Explicit (target+precursors) chemsystems from synthesis_clean.json.
    """
    existing = set(pd_index.keys())
    needed: set[str] = set()

    # ── Source 1: systematic +C / +N extensions ───────────────────────────
    for cs in existing:
        els = set(cs.split("-"))
        n_els = len(els)

        # Only extend oxide systems
        if "O" not in els:
            continue

        # +C extension (carbonate precursors — most common gap)
        if "C" not in els and n_els < PD_MAX_ELEMENTS:
            ext_C = "-".join(sorted(els | {"C"}))
            if ext_C not in existing:
                needed.add(ext_C)

        # +N extension (nitrate precursors — second most common)
        if "N" not in els and n_els < PD_MAX_ELEMENTS:
            ext_N = "-".join(sorted(els | {"N"}))
            if ext_N not in existing:
                needed.add(ext_N)

        # +C+N extension (both — less common but covers e.g. Ba(NO3)2 + BaCO3 mixtures)
        if "C" not in els and "N" not in els and n_els < PD_MAX_ELEMENTS - 1:
            ext_CN = "-".join(sorted(els | {"C", "N"}))
            if ext_CN not in existing:
                needed.add(ext_CN)

    # ── Source 2: synthesis_clean.json explicit precursor chemsystems ─────
    if synthesis_path.exists():
        with synthesis_path.open() as f:
            records = json.load(f)

        for rec in records:
            els: set[str] = set()
            target = rec.get("target_formula")
            if target and is_valid_concrete(target):
                cs = get_chemsys(target)
                if cs:
                    els.update(cs.split("-"))

            for p in rec.get("precursors", []):
                pf = p.get("formula") if isinstance(p, dict) else None
                if pf and is_valid_concrete(pf):
                    cs = get_chemsys(pf)
                    if cs:
                        els.update(cs.split("-"))

            if els and len(els) <= PD_MAX_ELEMENTS:
                cs = "-".join(sorted(els))
                if cs not in existing:
                    needed.add(cs)
    else:
        log(f"  synthesis_clean.json not found at {synthesis_path} — skipping source 2")

    return needed


def breakdown_by_extension(needed: set[str], existing: set[str]) -> dict[str, int]:
    """Break down needed chemsystems by what extension type they are."""
    counts = {"+C only": 0, "+N only": 0, "+C+N": 0, "from synthesis records": 0}
    for cs in needed:
        els = set(cs.split("-"))
        has_c = "C" in els
        has_n = "N" in els
        if has_c and has_n:
            counts["+C+N"] += 1
        elif has_c:
            counts["+C only"] += 1
        elif has_n:
            counts["+N only"] += 1
        else:
            counts["from synthesis records"] += 1
    return counts


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extend the PD cache with +C and +N chemsystems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended sequence for a solid-state synthesis dataset:

  Step 1 (most impactful, ~90 min):
      python expand_pd_cache.py --only-C

  Step 2 (nitrate routes, optional, ~2 h):
      python expand_pd_cache.py --only-N

  Step 3 (mixed routes, rarely needed, ~8 h):
      python expand_pd_cache.py   # no flags = everything remaining
""",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be built without touching the API")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (default: cpu_count - 2, capped at 24)")
    parser.add_argument("--only-C", action="store_true",
                        help="Only +C extensions (~3,300 PDs). "
                             "Covers carbonate precursors — the most common gap. "
                             "Do this first.")
    parser.add_argument("--only-N", action="store_true",
                        help="Only +N extensions (~3,500 PDs). "
                             "Covers nitrate precursors. Do after --only-C.")
    parser.add_argument("--skip-cn", action="store_true",
                        help="Skip +C+N combinations (~8,000 PDs). "
                             "Mixed carbonate+nitrate routes are rare in "
                             "solid-state synthesis. Equivalent to --only-C + --only-N "
                             "in two separate passes.")
    parser.add_argument("--pd-index", type=Path, default=PD_INDEX_FILE,
                        help="Path to pd_index.json")
    parser.add_argument("--synthesis", type=Path, default=SYNTHESIS_FILE,
                        help="Path to synthesis_clean.json")
    args = parser.parse_args()

    if not MP_API_KEY and not args.dry_run:
        raise RuntimeError("Set MP_API_KEY env var before running")

    PD_SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing index (resume-safe)
    pd_index: dict[str, str] = {}
    if args.pd_index.exists():
        with args.pd_index.open() as f:
            pd_index = json.load(f)
    log(f"Loaded pd_index: {len(pd_index)} existing shards")

    # Compute what's needed
    needed = compute_needed_chemsystems(pd_index, args.synthesis)

    # Apply mode filters. Default (no flags) builds everything.
    # Recommended first pass: --only-C (~3k PDs, ~90 min).
    if args.only_C:
        # Only +C extensions: carbonate precursors, highest-value single pass.
        needed = {cs for cs in needed
                  if "C" in cs.split("-") and "N" not in cs.split("-")}
    elif args.only_N:
        # Only +N extensions: nitrate precursors.
        needed = {cs for cs in needed
                  if "N" in cs.split("-") and "C" not in cs.split("-")}
    elif args.skip_cn:
        # Skip +C+N combinations (~8k PDs): drop anything that has BOTH C and N
        # added as new extensions. Mixed carbonate+nitrate routes are rare in
        # solid-state synthesis; do +C and +N separately first.
        needed = {cs for cs in needed
                  if not ("C" in cs.split("-") and "N" in cs.split("-"))}

    log(f"Total chemsystems to build: {len(needed)}")

    if needed:
        breakdown = breakdown_by_extension(needed, set(pd_index.keys()))
        for label, count in breakdown.items():
            if count:
                log(f"  {label}: {count}")

    if not needed:
        log("Cache is already complete — nothing to build.")
        return

    if args.dry_run:
        log("Dry run — top 20 chemsystems that would be built:")
        for cs in sorted(needed)[:20]:
            log(f"  {cs}")
        if len(needed) > 20:
            log(f"  ... and {len(needed) - 20} more")
        log("Re-run without --dry-run to build them.")
        return

    # Build
    n_workers = args.workers or min(24, max(1, multiprocessing.cpu_count() - 2))
    log(f"Building {len(needed)} PDs with {n_workers} workers ...")

    to_build = sorted(needed)
    n_success = n_skip = n_fail = 0

    def save_index():
        with args.pd_index.open("w") as f:
            json.dump(pd_index, f, indent=2)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_chemsys, cs): cs for cs in to_build}

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                chemsys, path, n_entries, status = future.result()
            except Exception as e:
                chemsys = futures[future]
                log(f"  [{i}/{len(to_build)}] {chemsys}: WORKER_CRASH ({e})")
                n_fail += 1
                continue

            if status == "success":
                pd_index[chemsys] = path
                n_success += 1
                log(f"  [{i}/{len(to_build)}] ✓ {chemsys:35s} {n_entries} entries")
            elif status == "no_entries":
                n_skip += 1
                log(f"  [{i}/{len(to_build)}] ○ {chemsys:35s} no entries in MP")
            else:
                n_fail += 1
                log(f"  [{i}/{len(to_build)}] ✗ {chemsys:35s} {status}")

            if i % 10 == 0:
                save_index()
                log(f"  → index checkpoint at {i} ({n_success} built, {n_skip} empty, {n_fail} failed)")

    save_index()
    log("=" * 60)
    log(f"Done. Built: {n_success}  Empty: {n_skip}  Failed: {n_fail}")
    log(f"Updated index: {args.pd_index} ({len(pd_index)} total shards)")


if __name__ == "__main__":
    main()