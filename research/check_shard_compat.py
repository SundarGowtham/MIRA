"""
check_shard_compat.py
------------------------
Verifies existing PD shards still unpickle AND still BEHAVE correctly
under upgraded mp_api/pymatgen, before committing to the upgrade in the
main environment. Checks more than "does it load without crashing" -
pickle compatibility across library versions can succeed while silently
changing internal state (cached values, correction application order,
etc.), which wouldn't show up as an error, just as wrong numbers later.

Two modes, meant to be run in two DIFFERENT environments against the
SAME sample of shards:

  --mode baseline   run this FIRST, in your CURRENT (pre-upgrade)
                     environment. Captures a fingerprint per shard:
                     entry count, stable entry count, a deterministic
                     sum of all entries' energy_per_atom (would shift if
                     ANYTHING about entry construction/corrections
                     changed), and e_above_hull + get_decomposition
                     results for a couple of real compositions as a
                     functional sanity check, not just a load check.

  --mode compare     run this SECOND, in a NEW isolated venv with
                     mp_api/pymatgen upgraded, pointed at the SAME
                     pd_shards directory. Loads the baseline JSON,
                     recomputes the same fingerprint for the same
                     sampled shards, and flags any shard whose
                     fingerprint changed - a load failure OR a silent
                     numeric drift are both reported, distinctly.

Uses a FIXED random seed so both runs sample the identical shards.

Usage:
  # in your current environment:
  uv run python check_shard_compat.py \
      --pd-index data/cache/pd_index.json --project-root . \
      --mode baseline --sample-size 200 --out shard_compat_baseline.json

  # in a fresh venv with mp_api/pymatgen upgraded:
  python check_shard_compat.py \
      --pd-index data/cache/pd_index.json --project-root . \
      --mode compare --baseline shard_compat_baseline.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path


def fingerprint_shard(chemsys: str, shard_path: Path) -> dict:
    result = {"chemsys": chemsys, "loaded": False}
    try:
        with shard_path.open("rb") as f:
            pd = pickle.load(f)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["loaded"] = True
    try:
        all_entries = list(pd.all_entries)
        stable_entries = list(pd.stable_entries)
        result["n_entries"] = len(all_entries)
        result["n_stable"] = len(stable_entries)

        # deterministic numeric fingerprint - shifts if entry construction,
        # corrections, or energy computation changed between versions at all
        energy_sum = sum(e.energy_per_atom for e in all_entries)
        result["energy_sum_fingerprint"] = round(energy_sum, 6)

        # functional sanity check: the lowest-energy entry should be on
        # (or extremely near) the hull - e_above_hull should be ~0
        lowest = min(all_entries, key=lambda e: e.energy_per_atom)
        try:
            e_above = float(pd.get_e_above_hull(lowest, on_error="ignore"))
            result["lowest_entry_e_above_hull"] = round(e_above, 6) if e_above is not None else None
        except Exception as e:
            result["lowest_entry_e_above_hull_error"] = str(e)

        # functional sanity check: a stable entry should trivially
        # decompose to itself with weight ~1.0
        if stable_entries:
            probe = stable_entries[0]
            try:
                decomp = pd.get_decomposition(probe.composition)
                result["probe_composition"] = probe.composition.reduced_formula
                result["probe_decomp_n_phases"] = len(decomp)
                result["probe_decomp_self_weight"] = round(
                    max(decomp.values()), 6
                ) if decomp else None
            except Exception as e:
                result["probe_decomp_error"] = str(e)

    except Exception as e:
        result["post_load_error"] = f"{type(e).__name__}: {e}"

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--mode", choices=["baseline", "compare"], required=True)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("shard_compat_baseline.json"))
    ap.add_argument("--baseline", type=Path, default=None, help="required for --mode compare")
    args = ap.parse_args()

    pd_index = json.loads(args.pd_index.read_text())

    if args.mode == "baseline":
        chemsystems = sorted(pd_index.keys())
        random.Random(args.seed).shuffle(chemsystems)
        sample = chemsystems[:args.sample_size]
        print(f"Fingerprinting {len(sample)} of {len(chemsystems)} shards (seed={args.seed})...")

        results = {}
        for i, cs in enumerate(sample):
            shard_path = args.project_root / pd_index[cs]
            results[cs] = fingerprint_shard(cs, shard_path)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(sample)}")

        n_loaded = sum(1 for r in results.values() if r["loaded"])
        print(f"\n{n_loaded}/{len(sample)} loaded successfully in THIS environment.")
        args.out.write_text(json.dumps({
            "seed": args.seed, "sample_size": args.sample_size, "results": results,
        }, indent=2, default=str))
        print(f"Wrote baseline to {args.out}")
        print(f"\nNow install upgraded mp_api/pymatgen in a FRESH venv and rerun with "
              f"--mode compare --baseline {args.out}")

    else:  # compare
        if not args.baseline:
            print("ERROR: --mode compare requires --baseline <path to baseline json>")
            return
        baseline_data = json.loads(args.baseline.read_text())
        baseline_results = baseline_data["results"]
        seed = baseline_data["seed"]
        sample_size = baseline_data["sample_size"]

        chemsystems = sorted(pd_index.keys())
        random.Random(seed).shuffle(chemsystems)
        sample = chemsystems[:sample_size]

        assert set(sample) == set(baseline_results.keys()), (
            "Sample mismatch - pd_index.json must be the SAME file (or at least contain "
            "the same chemsystems) as when the baseline was captured."
        )

        print(f"Re-fingerprinting {len(sample)} shards under the NEW environment...")
        new_results = {}
        for i, cs in enumerate(sample):
            shard_path = args.project_root / pd_index[cs]
            new_results[cs] = fingerprint_shard(cs, shard_path)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(sample)}")

        n_newly_broken = 0
        n_now_fixed = 0
        n_numeric_drift = 0
        n_unchanged_ok = 0
        n_still_broken_both = 0

        for cs in sample:
            old = baseline_results[cs]
            new = new_results[cs]

            if old["loaded"] and not new["loaded"]:
                n_newly_broken += 1
                print(f"  REGRESSION: {cs} loaded before, FAILS now: {new.get('error')}")
            elif not old["loaded"] and new["loaded"]:
                n_now_fixed += 1
                print(f"  IMPROVED: {cs} failed before, loads now (expected for the "
                      f"corrupted-shard population, not a concern)")
            elif not old["loaded"] and not new["loaded"]:
                n_still_broken_both += 1
                old_err = old.get("error", "?")
                new_err = new.get("error", "?")
                if old_err != new_err:
                    print(f"  NOTE: {cs} fails in both, but with a DIFFERENT error: "
                          f"{old_err!r} -> {new_err!r} (worth a look, though still broken either way)")
            elif old["loaded"] and new["loaded"]:
                keys_to_compare = ["n_entries", "n_stable", "energy_sum_fingerprint",
                                    "lowest_entry_e_above_hull", "probe_decomp_self_weight"]
                drifted = [k for k in keys_to_compare if old.get(k) != new.get(k)]
                if drifted:
                    n_numeric_drift += 1
                    print(f"  DRIFT: {cs} loads in both, but differs in: {drifted}")
                    for k in drifted:
                        print(f"      {k}: {old.get(k)} -> {new.get(k)}")
                else:
                    n_unchanged_ok += 1

        assert (n_newly_broken + n_now_fixed + n_numeric_drift + n_unchanged_ok
                + n_still_broken_both) == len(sample), (
            "Category counts don't sum to sample size - this itself would be a bug, "
            "not just a reporting gap. Should be unreachable now, but asserting so a "
            "future edit that reintroduces the gap fails loudly instead of silently."
        )

        print(f"\n{'='*60}")
        print(f"COMPATIBILITY CHECK RESULT ({len(sample)} shards sampled)")
        print(f"{'='*60}")
        print(f"  unchanged, fully OK:        {n_unchanged_ok}")
        print(f"  newly fixed (were corrupted, now load - expected/fine): {n_now_fixed}")
        print(f"  still broken in both (expected for genuine file corruption): {n_still_broken_both}")
        print(f"  REGRESSION (loaded before, fails now):  {n_newly_broken}")
        print(f"  NUMERIC DRIFT (loads both, different values): {n_numeric_drift}")
        print(f"  {'-'*40}")
        print(f"  TOTAL: {n_unchanged_ok + n_now_fixed + n_still_broken_both + n_newly_broken + n_numeric_drift} "
              f"(should equal sample size {len(sample)})")
        print(f"{'='*60}")
        if n_newly_broken > 0 or n_numeric_drift > 0:
            print("\nNOT clean - review the flagged shards above before upgrading "
                  "your main environment. A regression or drift here means the "
                  "upgrade changes behavior for existing data, not just fixes "
                  "the entry_id issue.")
        else:
            print("\nClean - no regressions, no numeric drift on the sampled shards. "
                  "Reasonable confidence the upgrade is safe for existing data.")


if __name__ == "__main__":
    main()