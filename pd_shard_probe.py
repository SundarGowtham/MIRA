"""
pd_shard_probe.py  -  run on BOTH mac and manifold, then diff the stdout.

Answers: is the 48G/14G gap a truncated transfer, a leaner rebuild, or
structures-stripped entries? Each tells a different story and needs a
different fix.

  same entry counts, similar bytes  -> du/filesystem artifact, caches equal
  same entry counts, bytes differ   -> serialization/structure difference
  fewer entries on one machine       -> IMPOVERISHED HULLS (rebuild needed)
  entries lack .structure on one     -> anion corrections silently broken
                                        (peroxide/superoxide/oxide split
                                         needs bond lengths -> needs structures)

Usage:
  python pd_shard_probe.py --index data/cache/pd_index.json --root . --n 40
"""
import argparse, json, pickle, random, statistics
from pathlib import Path


def probe(index_path: Path, root: Path, n: int, seed: int = 0):
    idx = json.loads(index_path.read_text())
    keys = sorted(idx.keys())
    random.Random(seed).shuffle(keys)
    sample = keys[:n]

    n_entries, sizes, with_struct, corr_nonzero, entry_types = [], [], 0, 0, {}
    failed = []

    for cs in sample:
        shard = root / idx[cs]
        if not shard.exists():
            failed.append((cs, "missing"))
            continue
        sizes.append(shard.stat().st_size)
        try:
            pd = pickle.loads(shard.read_bytes())
        except Exception as e:
            failed.append((cs, f"unpicklable: {type(e).__name__}"))
            continue
        entries = list(pd.all_entries)
        n_entries.append(len(entries))
        e0 = entries[0] if entries else None
        if e0 is not None:
            entry_types[type(e0).__name__] = entry_types.get(type(e0).__name__, 0) + 1
            if getattr(e0, "structure", None) is not None:
                with_struct += 1
            # correction magnitude on a corrected entry, if any carry one
            if any(abs(getattr(e, "correction", 0.0)) > 1e-9 for e in entries):
                corr_nonzero += 1

    def stats(xs):
        if not xs:
            return "n/a"
        return (f"min={min(xs)} median={int(statistics.median(xs))} "
                f"max={max(xs)} mean={int(statistics.mean(xs))}")

    print(f"index: {index_path}   sampled {len(sample)} of {len(keys)} chemsys")
    print(f"shard bytes:   {stats(sizes)}")
    print(f"entries/hull:  {stats(n_entries)}")
    print(f"entry types:   {entry_types}")
    print(f"has .structure (first entry): {with_struct}/{len(n_entries)}")
    print(f"hulls with >=1 corrected entry: {corr_nonzero}/{len(n_entries)}")
    if failed:
        print(f"FAILED {len(failed)}: {failed[:5]}{' ...' if len(failed) > 5 else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    probe(a.index, a.root, a.n, a.seed)