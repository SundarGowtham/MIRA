"""
refetch_pd_shards.py
-----------------------
Targeted re-fetch of specific PD shards, rather than rebuilding the whole
~19,862-shard cache. Reuses process_single_chemsys DIRECTLY from
data_pull_3.py (imported, not duplicated) - same MPRester call, same
GGA/GGA+U/R2SCAN criteria, same MaterialsProjectDFTMixingScheme
correction already confirmed correct earlier this session, same PD
construction. This guarantees re-fetched shards are built identically to
the rest of your cache, not a slightly-different one-off implementation.

Two chemsys sources, usable together:
  --from-corrupted-census   pulls the full list from corrupted_chemsys.json
                             (pd_shard_census.py's output, ~1689 chemsystems
                             confirmed to fail unpickling, from earlier this
                             session)
  --chemsys A-B-C           ad-hoc specific chemsystems (repeatable flag) -
                             e.g. the C-Fe-H-Li-N-O-P / Fe-Li-O-P pair found
                             via inspect_lfp_chemsys.py

For chemsystems ALREADY in pd_index.json (just corrupted), re-fetching
overwrites the shard file in place at the SAME path - no index change
needed. For chemsystems NOT in pd_index.json at all (genuinely missing),
a new entry is added after a successful fetch.

Sequential, not parallel - this hits the live MP API, and
process_single_chemsys already has its own retry/backoff built in
(data_pull_3.py's with_retry decorator). Hammering the API in parallel
isn't the same kind of embarrassingly-parallel-safe operation the local
PD-cache reads earlier this session were.

Requires MP_API_KEY set in the environment (data_pull_3.py checks this
at import time and raises immediately if it's missing).

Usage:
  export MP_API_KEY='your_key'
  uv run python refetch_pd_shards.py \
      --project-root . \
      --pd-index data/cache/pd_index.json \
      --from-corrupted-census corrupted_chemsys.json \
      --chemsys C-Fe-H-Li-N-O-P --chemsys Fe-Li-O-P \
      --dry-run
  # drop --dry-run once the list looks right
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import dotenv

dotenv.load_dotenv()

def write_pd_index(pd_index: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(pd_index, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--from-corrupted-census", type=Path, default=None,
                     help="corrupted_chemsys.json from pd_shard_census.py")
    ap.add_argument("--chemsys", action="append", default=[],
                     help="ad-hoc chemsys to re-fetch, e.g. C-Fe-H-Li-N-O-P (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be fetched, don't call the API")
    ap.add_argument("--log", type=Path, default=Path("refetch_log.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.project_root))
    import data_pull_3  # requires MP_API_KEY set - fails fast at import if not

    to_fetch: set[str] = set(args.chemsys)
    if args.from_corrupted_census:
        census = json.loads(args.from_corrupted_census.read_text())
        for entry in census.get("corrupted", []):
            to_fetch.add(entry["chemsys"])
        print(f"Loaded {len(census.get('corrupted', []))} chemsystems from corrupted census.")

    if not to_fetch:
        print("Nothing to fetch - pass --chemsys and/or --from-corrupted-census.")
        return

    pd_index = json.loads(args.pd_index.read_text())
    to_fetch_sorted = sorted(to_fetch)
    n_already_indexed = sum(1 for c in to_fetch_sorted if c in pd_index)
    n_new = len(to_fetch_sorted) - n_already_indexed

    print(f"\n{len(to_fetch_sorted)} chemsystems to re-fetch: "
          f"{n_already_indexed} already in pd_index (corrupted -> overwrite in place), "
          f"{n_new} genuinely new (will be added to pd_index on success)")

    if args.dry_run:
        print("\n--dry-run: not calling the API. Chemsystems that would be fetched:")
        for c in to_fetch_sorted:
            status = "corrupted, will overwrite" if c in pd_index else "new, will add to index"
            print(f"  {c:<40} ({status})")
        return

    results = []
    if args.log.exists():
        try:
            results = json.loads(args.log.read_text())
            done = {r["chemsys"] for r in results if r["status"] == "Success"}
            to_fetch_sorted = [c for c in to_fetch_sorted if c not in done]
            print(f"Resuming: {len(done)} already fetched successfully, "
                  f"{len(to_fetch_sorted)} remaining.")
        except Exception:
            pass

    for i, chemsys in enumerate(to_fetch_sorted):
        print(f"[{i+1}/{len(to_fetch_sorted)}] fetching {chemsys} ...", file=sys.stderr)
        t0 = time.time()
        try:
            cs, shard_path, n_entries, status = data_pull_3.process_single_chemsys(chemsys)
        except Exception as e:
            cs, shard_path, n_entries, status = chemsys, None, 0, f"EXCEPTION: {e}"
        elapsed = time.time() - t0
        print(f"  -> {status}  ({n_entries} entries, {elapsed:.1f}s)", file=sys.stderr)

        results.append({"chemsys": cs, "shard_path": shard_path, "n_entries": n_entries,
                         "status": status, "elapsed_s": round(elapsed, 1)})
        args.log.write_text(json.dumps(results, indent=2))

        if status == "Success" and shard_path:
            pd_index[cs] = shard_path
            write_pd_index(pd_index, args.pd_index)

    n_success = sum(1 for r in results if r["status"] == "Success")
    n_failed = len(results) - n_success
    # dedupe for the summary: keep only the LATEST attempt per chemsys, since
    # a failed chemsys gets retried (not skipped) on every rerun, and without
    # this the raw log length double-counts retried failures across reruns
    latest_by_chemsys = {}
    for r in results:
        latest_by_chemsys[r["chemsys"]] = r
    deduped = list(latest_by_chemsys.values())
    n_success_deduped = sum(1 for r in deduped if r["status"] == "Success")
    n_failed_deduped = len(deduped) - n_success_deduped

    print(f"\n{'='*60}")
    print(f"DONE (this run): {n_success} succeeded, {n_failed} failed/skipped")
    print(f"CUMULATIVE (deduped across all runs, {len(deduped)} distinct chemsystems): "
          f"{n_success_deduped} succeeded, {n_failed_deduped} failed/skipped")
    print(f"pd_index.json updated in place, log written to {args.log}")
    if n_failed_deduped:
        print(f"\nfailed/skipped chemsystems (latest attempt):")
        for r in deduped:
            if r["status"] != "Success":
                print(f"  {r['chemsys']:<40} {r['status']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()