#!/usr/bin/env python
"""
ranker_rail_calibration.py — RANKER_SPEC.md step 2.

Score ~200 archived completions from
runs/gdpo-qlora-beta-ablation-probe/generations.jsonl with the ranker.
NO NEW GENERATION — this is CPU-only scoring of routes the model already
produced. Reports per-objective rail stats (fraction piled at exactly
0.0/1.0), per-gate failure rates, and inline reward capacity, so scales can
be tuned (RankerScales) before committing GPU to the step-3 capacity probe.

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/ranker_rail_calibration.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reward import ParseFailure, parse_completion  # noqa: E402
from core.ranker import (  # noqa: E402
    OBJECTIVE_NAMES, RANKER_VERSION, Ranker, RankerScales,
    build_precursor_frequency, gate_failure_rates, rail_stats,
)
from probe_hardening import capacity_metrics, load_literature  # noqa: E402
from validator import ThermoChecker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gens", type=Path,
                   default=Path("runs/gdpo-qlora-beta-ablation-probe/generations.jsonl"))
    p.add_argument("--triage", type=Path,
                   default=Path("misc/kononova_triage_results3.json"))
    p.add_argument("--synthesis", type=Path,
                   default=Path("data/raw/synthesis_clean.json"))
    p.add_argument("--n-sample", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("misc/ranker_rail_calibration.json"))
    # rail-tunable scales -- defaults are RANKER_SPEC.md's own starting points
    p.add_argument("--t-ref-margin", type=float, default=150.0)
    p.add_argument("--t-span", type=float, default=400.0)
    p.add_argument("--n-ref-margin", type=int, default=2)
    p.add_argument("--n-span", type=float, default=4.0)
    p.add_argument("--cost-scale", type=float, default=10.0)
    p.add_argument("--dg-scale", type=float, default=0.3)
    p.add_argument("--n-max", type=float, default=5.0)
    p.add_argument("--purity-window", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()

    print("loading literature (precursors, max_T, n_ops)...", flush=True)
    lit = load_literature(args.triage, args.synthesis)

    print("loading validator formula set + PD cache + precursor frequency...", flush=True)
    with open("data/cache/mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(Path("data/cache/pd_index.json"), Path("."))
    freq = build_precursor_frequency(args.synthesis)

    scales = RankerScales(
        t_ref_margin=args.t_ref_margin, t_span=args.t_span,
        n_ref_margin=args.n_ref_margin, n_span=args.n_span,
        cost_scale=args.cost_scale, dg_scale=args.dg_scale,
        n_max=args.n_max, purity_window=args.purity_window,
    )
    ranker = Ranker(formula_set, thermo, freq, scales=scales)

    print(f"loading + sampling {args.n_sample} archived completions from {args.gens}...",
          flush=True)
    records = [json.loads(l) for l in args.gens.open() if l.strip()]
    records = [r for r in records if r.get("target") in lit]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    sample = records[:args.n_sample]
    print(f"  {len(sample)}/{len(records)} candidates with usable literature data "
          f"(pool before that filter: unfiltered gens file)", flush=True)

    scored = []
    n_parse_fail = 0
    for i, r in enumerate(sample):
        target = r["target"]
        try:
            route = parse_completion(r["completion"], target)
        except ParseFailure:
            route = None
            n_parse_fail += 1
        lit_rec = lit[target]
        reward, info = ranker.score(
            route, target, lit_T=lit_rec.get("max_T"), lit_n_ops=lit_rec.get("n_ops"))
        scored.append({"target": target, "breakdown": info})
        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1}/{len(sample)}", flush=True)

    breakdowns = [s["breakdown"] for s in scored]
    rails = rail_stats(breakdowns)
    gates = gate_failure_rates(breakdowns)
    cap = capacity_metrics(scored, OBJECTIVE_NAMES)

    print("\n" + "=" * 78)
    print("RANKER RAIL CALIBRATION")
    print("=" * 78)
    print(f"n={len(sample)}  parse_failures={n_parse_fail}  "
          f"ranker_version={RANKER_VERSION}")
    print(f"\nscales: {scales}")
    print("\n--- gate failure rates (target: a few % at most; >few% -> objective in disguise) ---")
    for g, v in gates.items():
        print(f"  {g:<24}{v}%" if v is not None else f"  {g:<24}n/a")
    print("\n--- objective rail stats (target: <15% at either rail) ---")
    print(f"  {'objective':<24}{'n':>6}{'%@0.0':>8}{'%@1.0':>8}{'mean':>8}")
    for name in OBJECTIVE_NAMES:
        s = rails[name]
        if s["n"] == 0:
            print(f"  {name:<24}{'0':>6}{'--':>8}{'--':>8}{'--':>8}")
            continue
        flag = "  <-- RAIL PILE-UP" if (s["pct_at_0"] > 15 or s["pct_at_1"] > 15) else ""
        print(f"  {name:<24}{s['n']:>6}{s['pct_at_0']:>7.1f}%{s['pct_at_1']:>7.1f}%"
              f"{s['mean']:>8.3f}{flag}")
    print(f"\ninline capacity (n_channels={len(OBJECTIVE_NAMES)}): "
          f"{cap.get('capacity_pct')}%  (n_groups={cap.get('n_groups')})")
    print("per-channel z-var:", cap.get("per_channel_z_var"))
    print("per-channel zero-std %:", cap.get("per_channel_zero_std_pct"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ranker_version": RANKER_VERSION,
        "n": len(sample), "n_parse_fail": n_parse_fail,
        "scales": scales.__dict__,
        "gate_failure_rates": gates,
        "rail_stats": rails,
        "capacity": cap,
        "records": scored,
    }, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
