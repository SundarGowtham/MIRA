"""
debug_refetch_failure.py
---------------------------
process_single_chemsys's broad `except Exception as e: return ... f"SKIP
({e})"` threw away the traceback for the "unhashable type: 'dict'" error
on Al-Mg-N-O-Sb-Si-Ti - keeping only the bare message, no idea which of
the three stages (MPRester.get_entries_in_chemsys,
MaterialsProjectDFTMixingScheme.process_entries, or PhaseDiagram
construction) actually raised it.

This runs the exact same three stages, one at a time, with full
tracebacks kept intact, so the failure point is visible instead of
guessed at.

Usage:
  export MP_API_KEY='your_key'
  uv run python debug_refetch_failure.py --chemsys Al-Mg-N-O-Sb-Si-Ti
"""
from __future__ import annotations

import argparse
import os
import traceback
import dotenv

dotenv.load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemsys", required=True)
    args = ap.parse_args()

    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

    print(f"=== Stage 1: MPRester.get_entries_in_chemsys({args.chemsys}) ===")
    try:
        with MPRester(os.environ.get("MP_API_KEY")) as mpr:
            entries = mpr.get_entries_in_chemsys(
                elements=args.chemsys.split("-"),
                additional_criteria={"thermo_types": ["GGA_GGA+U", "R2SCAN"]},
            )
        print(f"OK - {len(entries)} entries retrieved.")
        if entries:
            e0 = entries[0]
            print(f"  first entry type: {type(e0)}")
            print(f"  first entry .data type: {type(getattr(e0, 'data', None))}")
            if hasattr(e0, "data") and isinstance(e0.data, dict):
                for k, v in e0.data.items():
                    print(f"    data[{k!r}] = {type(v)}"
                          + ("  <-- DICT VALUE, worth suspecting" if isinstance(v, dict) else ""))
    except Exception:
        print("STAGE 1 FAILED - full traceback:")
        traceback.print_exc()
        return

    print(f"\n=== Stage 2: MaterialsProjectDFTMixingScheme().process_entries(entries) ===")
    try:
        scheme = MaterialsProjectDFTMixingScheme()
        mixed_entries = scheme.process_entries(entries)
        print(f"OK - {len(mixed_entries)} entries after mixing scheme correction.")
    except Exception:
        print("STAGE 2 FAILED - full traceback:")
        traceback.print_exc()
        print(f"\nDiagnostic dump of the first few raw entries' .data fields, "
              f"since Stage 2 is where dict-valued entry.data fields most often "
              f"break internal deduplication/grouping logic that expects hashable keys:")
        for i, e in enumerate(entries[:5]):
            data = getattr(e, "data", None)
            print(f"  entry[{i}] composition={e.composition.reduced_formula}  "
                  f"data_type={type(data)}  data={data if not isinstance(data, dict) else list(data.keys())}")
        return

    print(f"\n=== Stage 3: PhaseDiagram(entries) ===")
    try:
        pd = PhaseDiagram(mixed_entries)
        print(f"OK - PhaseDiagram built, {len(pd.all_entries)} entries.")
    except Exception:
        print("STAGE 3 FAILED - full traceback:")
        traceback.print_exc()
        return

    print("\nAll three stages succeeded - the failure may be intermittent "
          "or specific to something else in process_single_chemsys "
          "(the pickle.dump step, the signal/timeout handling). Worth "
          "rerunning the real refetch_pd_shards.py for just this chemsys "
          "to see if it now succeeds.")


if __name__ == "__main__":
    main()