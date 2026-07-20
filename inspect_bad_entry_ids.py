"""
inspect_bad_entry_ids.py
----------------------------
debug_refetch_failure.py traced the crash to pymatgen's own
MaterialsProjectDFTMixingScheme._filter_and_sort_entries doing
`{e.entry_id for e in filtered_entries}` - a set comprehension that
requires entry_id to be hashable. For Al-Mg-N-O-Sb-Si-Ti at least one of
2279 entries has entry_id as a raw dict instead of a string.

This finds every entry with a non-hashable entry_id, shows what's
actually inside it, and checks run_type (GGA vs GGA+U vs R2SCAN) to see
if it's specific to one entry type - which would confirm the "API schema
drift for R2SCAN entries" hypothesis rather than being random/widespread.

Usage:
  export MP_API_KEY='your_key'
  uv run python inspect_bad_entry_ids.py --chemsys Al-Mg-N-O-Sb-Si-Ti
"""
from __future__ import annotations

import argparse
import os
import dotenv

dotenv.load_dotenv()

def is_hashable(x) -> bool:
    try:
        hash(x)
        return True
    except TypeError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemsys", required=True)
    args = ap.parse_args()

    from mp_api.client import MPRester

    with MPRester(os.environ.get("MP_API_KEY")) as mpr:
        entries = mpr.get_entries_in_chemsys(
            elements=args.chemsys.split("-"),
            additional_criteria={"thermo_types": ["GGA_GGA+U", "R2SCAN"]},
        )
    print(f"{len(entries)} entries retrieved.\n")

    bad = [e for e in entries if not is_hashable(e.entry_id)]
    good = [e for e in entries if is_hashable(e.entry_id)]
    print(f"{len(bad)}/{len(entries)} entries have a non-hashable entry_id")
    print(f"{len(good)}/{len(entries)} entries have a normal (hashable) entry_id\n")

    if good:
        print(f"example of a NORMAL entry_id: {good[0].entry_id!r} (type={type(good[0].entry_id)})\n")

    run_type_counts_bad = {}
    for e in bad:
        rt = e.data.get("run_type", "?") if isinstance(getattr(e, "data", None), dict) else "?"
        run_type_counts_bad[rt] = run_type_counts_bad.get(rt, 0) + 1
    print(f"run_type breakdown among the BAD (dict entry_id) entries: {run_type_counts_bad}")

    run_type_counts_good = {}
    for e in good:
        rt = e.data.get("run_type", "?") if isinstance(getattr(e, "data", None), dict) else "?"
        run_type_counts_good[rt] = run_type_counts_good.get(rt, 0) + 1
    print(f"run_type breakdown among the GOOD (hashable entry_id) entries: {run_type_counts_good}\n")

    print("first 5 bad entries in full detail:")
    for e in bad[:5]:
        print(f"  composition={e.composition.reduced_formula}")
        print(f"  entry_id (dict) = {e.entry_id}")
        print(f"  run_type = {e.data.get('run_type') if isinstance(getattr(e, 'data', None), dict) else '?'}")
        print(f"  material_id = {e.data.get('material_id') if isinstance(getattr(e, 'data', None), dict) else '?'}")
        print()


if __name__ == "__main__":
    main()