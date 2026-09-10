"""
pd_shard_census.py
-------------------
Full pass over every shard in pd_index.json (not a sample). Answers: of
19862 indexed chemsys, how many shards actually fail to unpickle, and
which ones specifically.

This follows from pd_shard_probe.py's n=4000 finding: 333/4000 (8.3%)
EOFError, identical chemsys list on both mac and manifold - meaning the
corruption is baked into the shard files themselves, not a transfer
artifact. This script gets the exact count/list instead of a projection,
and writes corrupted_chemsys.json so a follow-up pass can check how many
training-corpus records were silently downgraded to light_only by a
corrupted shard rather than a genuinely absent one (validator.py's
_get_pd does `except Exception: pass; return None` - corrupted and
missing are indistinguishable downstream without this list).

Usage:
  python pd_shard_census.py --index data/cache/pd_index.json --root .

Writes:
  corrupted_chemsys.json  - {"corrupted": [...], "ok": <int>, "total": <int>}
"""
import argparse, json, pickle, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("corrupted_chemsys.json"))
    ap.add_argument("--progress-every", type=int, default=1000)
    a = ap.parse_args()

    idx = json.loads(a.index.read_text())
    keys = sorted(idx.keys())
    total = len(keys)

    corrupted = []
    missing = []
    ok = 0
    t0 = time.time()

    for i, cs in enumerate(keys):
        shard = a.root / idx[cs]
        if not shard.exists():
            missing.append(cs)
            continue
        try:
            with shard.open("rb") as f:
                pickle.load(f)
            ok += 1
        except Exception as e:
            corrupted.append({"chemsys": cs, "error": f"{type(e).__name__}: {e}"})

        if (i + 1) % a.progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else float("inf")
            print(
                f"  {i+1}/{total}  ok={ok} corrupted={len(corrupted)} "
                f"missing={len(missing)}  ({rate:.1f}/s, eta {eta/60:.1f}m)",
                file=sys.stderr,
            )

    print(f"\n=== census complete: {total} indexed chemsys ===")
    print(f"ok:        {ok}  ({ok/total:.2%})")
    print(f"corrupted: {len(corrupted)}  ({len(corrupted)/total:.2%})")
    print(f"missing:   {len(missing)}  ({len(missing)/total:.2%})")

    a.out.write_text(json.dumps(
        {"corrupted": corrupted, "missing": missing, "ok": ok, "total": total},
        indent=2,
    ))
    print(f"\nWrote {a.out} - corrupted chemsys list for corpus-overlap check.")


if __name__ == "__main__":
    main()