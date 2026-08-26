#!/usr/bin/env python
"""
analyze_passk.py — paired statistics on probe_passk.py output.

The marginal means (0.870 vs 0.880) hide what matters. These are PAIRED
measurements on the same targets, so the questions are:

  - How many targets did GDPO solve that SFT did not, and vice versa?
    (McNemar on discordant pairs -- the only thing that carries information)
  - Does the gap SHRINK as k grows (sharpening) or HOLD/WIDEN (expansion)?
  - Are the differences outside a paired bootstrap CI?

Usage:
    python analyze_passk.py --results misc/passk_n200.json
    python analyze_passk.py --results ... --bar 0.95    # re-bucket, no new GPU
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from math import comb
from pathlib import Path


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator. Undefined (None) when k > n."""
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load(path: Path, bar: float | None):
    """
    Returns {model: {target: {"n":, "c":, "rewards": [...]}}} and the stratum map.

    If `bar` is given, successes are recomputed from stored raw rewards, which
    lets you re-bucket at a stricter threshold with no new generation.
    """
    raw = json.loads(path.read_text())
    per_target = raw["per_target"] if isinstance(raw, dict) else raw

    data = defaultdict(dict)
    strata = {}
    for rec in per_target:
        target = rec.get("target")
        strata[target] = rec.get("stratum", "unknown")
        for model, m in rec.get("models", {}).items():
            rewards = m.get("rewards", [])
            if bar is not None and rewards:
                c = sum(1 for r in rewards if r >= bar)
                n = len(rewards)
            else:
                c, n = m.get("n_success", 0), m.get("n", len(rewards))
            data[model][target] = {"n": n, "c": c, "rewards": rewards}
    return data, strata


def mcnemar(b: int, c: int) -> float:
    """
    Exact two-sided binomial p-value on discordant pairs.
    b = A-only successes, c = B-only successes.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap(pairs, n_boot=10000, seed=0):
    """Bootstrap CI on the mean paired difference (resample targets)."""
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    diffs = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(a - b for a, b in s) / n)
    diffs.sort()
    mean = sum(a - b for a, b in pairs) / n
    return mean, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def compare(data, model_a: str, model_b: str, ks, strata=None):
    """model_a vs model_b (a is the 'treatment', e.g. gdpo300 vs sft)."""
    common = sorted(set(data[model_a]) & set(data[model_b]))
    print(f"\n{'='*72}")
    print(f"{model_a}  vs  {model_b}     ({len(common)} paired targets)")
    print(f"{'='*72}")

    print(f"\n{'k':>4}{'  '+model_b:>14}{'  '+model_a:>14}{'delta':>10}"
          f"{'95% CI':>22}")
    gaps = []
    for k in ks:
        pairs = []
        for t in common:
            ra, rb = data[model_a][t], data[model_b][t]
            pa, pb = pass_at_k(ra["n"], ra["c"], k), pass_at_k(rb["n"], rb["c"], k)
            if pa is None or pb is None:
                continue
            pairs.append((pa, pb))
        if not pairs:
            print(f"{k:>4}{'n/a (k > n_samples)':>60}")
            continue
        mb = sum(b for _, b in pairs) / len(pairs)
        ma = sum(a for a, _ in pairs) / len(pairs)
        mean, lo, hi = paired_bootstrap(pairs)
        sig = " *" if (lo > 0 or hi < 0) else ""
        gaps.append((k, mean))
        print(f"{k:>4}{mb:>14.4f}{ma:>14.4f}{mean:>+10.4f}"
              f"   [{lo:+.4f}, {hi:+.4f}]{sig}")

    print("\n* = paired bootstrap CI excludes zero.")

    # Sharpening vs expansion: does the gap shrink with k?
    if len(gaps) >= 2:
        k_lo, g_lo = gaps[0]
        k_hi, g_hi = gaps[-1]
        print(f"\ngap at k={k_lo}: {g_lo:+.4f}   gap at k={k_hi}: {g_hi:+.4f}")
        if g_hi < g_lo * 0.6:
            print(">> GAP SHRINKS with k. This is SHARPENING: the treatment model is")
            print("   more reliable at reaching routes the baseline can also reach")
            print("   given more attempts. Not a capability-boundary change.")
        elif g_hi > g_lo * 0.9:
            print(">> GAP HOLDS at large k. Consistent with EXPANSION: the treatment")
            print("   reaches targets the baseline does not reach at any k.")
        else:
            print(">> Gap partially shrinks -- ambiguous between sharpening and")
            print("   expansion at this n.")

    # McNemar at the largest usable k: 'solvable at all' comparison.
    k_max = max(k for k, _ in gaps) if gaps else None
    if k_max:
        both = a_only = b_only = neither = 0
        a_only_t, b_only_t = [], []
        for t in common:
            ra, rb = data[model_a][t], data[model_b][t]
            sa = ra["c"] > 0
            sb = rb["c"] > 0
            if sa and sb:
                both += 1
            elif sa:
                a_only += 1
                a_only_t.append(t)
            elif sb:
                b_only += 1
                b_only_t.append(t)
            else:
                neither += 1
        p = mcnemar(a_only, b_only)
        print(f"\n--- 'solvable at all' (>=1 success in n samples) ---")
        print(f"  both solve:        {both}")
        print(f"  {model_a} only:    {a_only}")
        print(f"  {model_b} only:    {b_only}")
        print(f"  neither:           {neither}")
        print(f"  McNemar exact p = {p:.4f}"
              f"{'  (significant)' if p < 0.05 else '  (not significant)'}")
        print(f"\n  NOTE: only the {a_only + b_only} discordant targets carry any")
        print(f"  information. The {both} both-solve and {neither} neither-solve")
        print("  targets contribute nothing to this comparison.")
        if a_only_t:
            print(f"\n  {model_a}-only targets: {', '.join(a_only_t[:12])}"
                  f"{' ...' if len(a_only_t) > 12 else ''}")
        if b_only_t:
            print(f"  {model_b}-only targets: {', '.join(b_only_t[:12])}"
                  f"{' ...' if len(b_only_t) > 12 else ''}")

    # Per-stratum breakdown at k=1.
    if strata:
        print(f"\n--- pass@1 by stratum ---")
        by = defaultdict(list)
        for t in common:
            ra, rb = data[model_a][t], data[model_b][t]
            pa = pass_at_k(ra["n"], ra["c"], 1)
            pb = pass_at_k(rb["n"], rb["c"], 1)
            if pa is not None and pb is not None:
                by[strata.get(t, "unknown")].append((pa, pb))
        print(f"{'stratum':<24}{model_b:>12}{model_a:>12}{'delta':>10}{'n':>6}")
        for s in sorted(by):
            pr = by[s]
            mb = sum(b for _, b in pr) / len(pr)
            ma = sum(a for a, _ in pr) / len(pr)
            print(f"{s[:23]:<24}{mb:>12.4f}{ma:>12.4f}{ma-mb:>+10.4f}{len(pr):>6}")


def headroom(data, models, ks):
    """How much room was there to begin with?"""
    print(f"\n{'='*72}")
    print("HEADROOM")
    print(f"{'='*72}")
    base = models[0]
    common = set(data[base])
    for m in models[1:]:
        common &= set(data[m])
    common = sorted(common)
    for k in ks:
        vals = {}
        for m in models:
            ps = [pass_at_k(data[m][t]["n"], data[m][t]["c"], k) for t in common]
            ps = [p for p in ps if p is not None]
            if ps:
                vals[m] = sum(ps) / len(ps)
        if len(vals) < 2:
            continue
        b = vals[base]
        print(f"\n  k={k}:  " + "   ".join(f"{m}={v:.4f}" for m, v in vals.items()))
        print(f"    room above {base}: {1 - b:.4f}")
        for m in models[1:]:
            if m in vals:
                captured = (vals[m] - b) / (1 - b) if (1 - b) > 1e-9 else float("nan")
                print(f"    {m} captured {100*captured:.1f}% of it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--bar", type=float, default=None,
                    help="re-bucket successes at this reward threshold "
                         "(uses stored raw rewards; no new generation)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--treatment", default="gdpo300")
    ap.add_argument("--baseline", default="sft")
    args = ap.parse_args()

    data, strata = load(args.results, args.bar)
    models = list(data)
    print(f"models found: {models}")
    print(f"success bar: {args.bar if args.bar is not None else 'as stored (0.9)'}")

    if args.treatment in data and args.baseline in data:
        compare(data, args.treatment, args.baseline, args.ks, strata)
    if "sft" in data and "base" in data:
        compare(data, "sft", "base", args.ks, strata)

    order = [m for m in ("base", "sft", args.treatment) if m in data]
    if len(order) >= 2:
        headroom(data, order, args.ks)


if __name__ == "__main__":
    main()