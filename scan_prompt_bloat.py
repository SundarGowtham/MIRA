"""
scan_prompt_bloat.py
-----------------------
Checks whether the unbounded stable_lines/competing_lines loop in
generate_traces_openrouter.py's get_stability_data (confirmed present,
unpatched, in production 2026-07) actually corrupted real training
prompts - not just theoretically capable of it, measured against the
real corpus.

The exact failure signature, discovered via stratified_difficulty_eval.py:
prompts for densely-studied chemsystems (e.g. Li-Mn-O) balloon to 20k+
characters because pd.stable_entries/pd.unstable_entries can contain
thousands of entries for heavily-studied systems, including near-
duplicate composition-series entries (LiMn437O874, LiMn438O876, ...)
that aren't true duplicates by reduced formula but are functionally
identical for a synthesis prompt's purposes.

This scans every record's stored "prompt" field for:
  1. raw length (chars) - distribution + flagged outliers
  2. the repetition signature specifically - does any single line/formula
     pattern repeat an implausible number of times within one prompt

Usage:
  uv run python scan_prompt_bloat.py \
      --synthesis data/processed/synthesis_with_traces.jsonl \
      --length-threshold 8000
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def detect_repetition(prompt: str, min_repeats: int = 10) -> dict | None:
    """Looks for the specific pathology: a formula-like token or a
    recurring phrase (e.g. 'above hull', 'delta Ef=') appearing far more
    times than a normal-length prompt would ever contain. Returns details
    if found, None if the prompt looks normal."""
    above_hull_count = prompt.count("above hull")
    def_ef_count = prompt.count("\u0394Ef=")  # delta Ef=

    # crude formula-token extraction: sequences like "LiMn437O874" - element
    # symbols followed by numbers, chained. Not a full chemical parser, just
    # good enough to catch composition-series near-duplicates repeating.
    tokens = re.findall(r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*", prompt)
    token_counts = Counter(t for t in tokens if len(t) > 3)
    most_common = token_counts.most_common(1)
    top_token, top_count = most_common[0] if most_common else (None, 0)

    if above_hull_count >= min_repeats or def_ef_count >= min_repeats or top_count >= min_repeats:
        return {
            "above_hull_count": above_hull_count,
            "delta_ef_count": def_ef_count,
            "most_repeated_token": top_token,
            "most_repeated_token_count": top_count,
        }
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthesis", type=Path, required=True)
    ap.add_argument("--length-threshold", type=int, default=8000,
                     help="prompts longer than this (chars) get flagged and inspected in detail")
    ap.add_argument("--show-worst", type=int, default=10)
    args = ap.parse_args()

    with args.synthesis.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    lengths = []
    flagged = []
    missing_prompt = 0

    for i, rec in enumerate(records):
        prompt = rec.get("prompt")
        target = rec.get("target", "?")
        if prompt is None:
            missing_prompt += 1
            continue
        n = len(prompt)
        lengths.append(n)
        if n >= args.length_threshold:
            rep = detect_repetition(prompt)
            flagged.append({"idx": i, "target": target, "length": n, "repetition": rep})

    if not lengths:
        print("No records had a 'prompt' field - nothing to scan.")
        return

    lengths_sorted = sorted(lengths)
    def pct(p):
        return lengths_sorted[int(len(lengths_sorted) * p)]

    print(f"n={len(lengths)} records with a prompt field ({missing_prompt} missing)")
    print(f"\nprompt length distribution (chars):")
    print(f"  min={min(lengths)}  p50={pct(0.5)}  p90={pct(0.9)}  p99={pct(0.99)}  max={max(lengths)}")
    print(f"  mean={statistics.mean(lengths):.0f}  stdev={statistics.stdev(lengths):.0f}")

    n_flagged = len(flagged)
    n_confirmed_repetitive = sum(1 for f in flagged if f["repetition"] is not None)
    print(f"\n{n_flagged}/{len(lengths)} ({n_flagged/len(lengths):.1%}) prompts >= "
          f"{args.length_threshold} chars")
    print(f"{n_confirmed_repetitive}/{n_flagged} of those show the specific repetition "
          f"signature (same formula/phrase repeating >=10x) - the rest may just be "
          f"legitimately verbose, not necessarily the same bug")

    if flagged:
        print(f"\nworst {min(args.show_worst, len(flagged))} by length:")
        for f in sorted(flagged, key=lambda x: -x["length"])[:args.show_worst]:
            rep = f["repetition"]
            rep_str = (f"repeats: {rep}" if rep else "no clear repetition signature - "
                       "may be legitimately long, worth eyeballing manually")
            print(f"  idx={f['idx']:<6} target={f['target']:<24} length={f['length']:<8} {rep_str}")


if __name__ == "__main__":
    main()