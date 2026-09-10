#!/usr/bin/env python
"""
ranker_rail_calibration_phase11.py — Phase 11 Step 1 rail calibration for the
rebuilt ranker (misc/some_claude_files/PHASE_11_INSTRUCTIONS.md).

Score ~200 archived completions from
runs/gdpo-qlora-beta-ablation-probe/generations.jsonl with the ranker.
NO NEW GENERATION -- CPU-only scoring of routes the model already produced.

Two things this does that ranker_rail_calibration.py (v1) didn't need to:
  1. Reports the RAW pre-clip quantity for every Phase 11 objective
     (precursor_instability_raw_mean_eah, inverse_hull_energy_raw_e_eq,
     n_precursors_raw_n, slice_competing_phases_raw_n_competing,
     precursor_decomposition_match_raw_margin) with 5th/95th percentiles --
     PHASE_11_INSTRUCTIONS.md Step 1 requires scales set from these
     percentiles, not hand-picked constants.
  2. Prints a ready-to-paste RankerScales block using those percentiles, so
     the *dataclass defaults* (not just this script's own CLI flags) can be
     kept in sync -- the exact gap that ran the first capacity probe
     uncalibrated (see RankerScales' docstring in core/ranker.py).

Usage (tmux):
  PYTHONPATH=. uv run python run_debug_and_analysis/ranker_rail_calibration_phase11.py
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

RAW_FIELDS = {
    "precursor_instability": "precursor_instability_raw_mean_eah",
    "inverse_hull_energy": "inverse_hull_energy_raw_e_eq",
    "n_precursors": "n_precursors_raw_n",
    "slice_competing_phases": "slice_competing_phases_raw_n_competing",
    "precursor_decomposition_match": "precursor_decomposition_match_raw_margin",
}


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
    p.add_argument("--out", type=Path, default=Path("misc/ranker_rail_calibration_phase11.json"))
    return p.parse_args()


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def main():
    args = parse_args()

    print("loading literature (precursors, max_T, n_ops)...", flush=True)
    lit = load_literature(args.triage, args.synthesis)

    print("loading validator formula set + PD cache + precursor frequency...", flush=True)
    with open("data/cache/mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(Path("data/cache/pd_index.json"), Path("."))
    freq = build_precursor_frequency(args.synthesis)

    # Score with the CURRENT (placeholder) RankerScales defaults -- the raw
    # fields are scale-independent (computed before clip), so this pass is
    # only about the raw distributions, not about tuning-by-rerun.
    scales = RankerScales()
    ranker = Ranker(formula_set, thermo, freq, scales=scales)

    print(f"loading + sampling {args.n_sample} archived completions from {args.gens}...",
          flush=True)
    records = [json.loads(l) for l in args.gens.open() if l.strip()]
    records = [r for r in records if r.get("target") in lit]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    sample = records[:args.n_sample]
    print(f"  {len(sample)}/{len(records)} candidates with usable literature data",
          flush=True)

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
    print("RANKER v2 (Phase 11) RAIL CALIBRATION")
    print("=" * 78)
    print(f"n={len(sample)}  parse_failures={n_parse_fail}  ranker_version={RANKER_VERSION}")

    print("\n--- gate failure rates (target: a few % at most) ---")
    for g, v in gates.items():
        print(f"  {g:<24}{v}%" if v is not None else f"  {g:<24}n/a")

    print("\n--- objective rail stats on CURRENT (placeholder) scales "
          "(target: <15% at either rail) ---")
    print(f"  {'objective':<30}{'n':>6}{'%@0.0':>8}{'%@1.0':>8}{'mean':>8}")
    for name in OBJECTIVE_NAMES:
        s = rails[name]
        if s["n"] == 0:
            print(f"  {name:<30}{'0':>6}{'--':>8}{'--':>8}{'--':>8}")
            continue
        flag = "  <-- RAIL PILE-UP" if (s["pct_at_0"] > 15 or s["pct_at_1"] > 15) else ""
        print(f"  {name:<30}{s['n']:>6}{s['pct_at_0']:>7.1f}%{s['pct_at_1']:>7.1f}%"
              f"{s['mean']:>8.3f}{flag}")

    print("\n--- raw pre-clip quantities: n / p5 / p50 / p95 "
          "(these set the scale defaults) ---")
    raw_percentiles = {}
    raw_vals: dict[str, list[float]] = {}
    for obj_name, raw_key in RAW_FIELDS.items():
        vals = [b[raw_key] for b in breakdowns
                if isinstance(b.get(raw_key), (int, float))
                and not isinstance(b.get(raw_key), bool)]
        raw_vals[obj_name] = vals
        if not vals:
            print(f"  {raw_key:<42}n=0 (no gradeable samples)")
            raw_percentiles[obj_name] = None
            continue
        p5, p50, p95 = percentile(vals, 0.05), percentile(vals, 0.50), percentile(vals, 0.95)
        print(f"  {raw_key:<42}n={len(vals):<5}p5={p5:<10.4f}p50={p50:<10.4f}p95={p95:<10.4f}")
        raw_percentiles[obj_name] = {"n": len(vals), "p5": p5, "p50": p50, "p95": p95}

    # Ready-to-paste RankerScales suggestion, each derived from that
    # objective's own raw distribution (PHASE_11_INSTRUCTIONS.md Step 1:
    # "Set every scale from the 5th/95th percentile of the raw quantity").
    suggested = dict(scales.__dict__)  # start from current (placeholder) values
    v = raw_vals["precursor_instability"]
    if v:
        suggested["instability_scale"] = round(max(1e-3, percentile(v, 0.95)), 4)
    v = raw_vals["inverse_hull_energy"]
    if v:
        magnitudes = [-x for x in v]
        suggested["inv_hull_scale"] = round(max(1e-3, percentile(magnitudes, 0.95)), 4)
    v = raw_vals["n_precursors"]
    if v:
        excess = [max(0, n - suggested["n_precursors_ref"]) for n in v]
        suggested["n_precursors_span"] = round(max(1.0, percentile(excess, 0.95)), 2)
    v = raw_vals["slice_competing_phases"]
    if v:
        suggested["slice_n_max"] = round(max(1.0, percentile(v, 0.95)), 2)
    v = raw_vals["precursor_decomposition_match"]
    if v:
        abs_margins = [abs(x) for x in v]
        suggested["decomp_span"] = round(max(50.0, percentile(abs_margins, 0.95)), 1)

    print("\n--- suggested RankerScales (ready to paste into RankerScales' "
          "class defaults in core/ranker.py) ---")
    for k, val in suggested.items():
        changed = " *" if suggested.get(k) != scales.__dict__.get(k) else ""
        print(f"  {k:<20}= {val}{changed}")

    print(f"\ninline capacity on placeholder scales "
          f"(n_channels={len(OBJECTIVE_NAMES)}): {cap.get('capacity_pct')}%  "
          f"(n_groups={cap.get('n_groups')})")
    print("per-channel z-var:", cap.get("per_channel_z_var"))
    print("per-channel zero-std %:", cap.get("per_channel_zero_std_pct"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ranker_version": RANKER_VERSION,
        "n": len(sample), "n_parse_fail": n_parse_fail,
        "placeholder_scales": scales.__dict__,
        "gate_failure_rates": gates,
        "rail_stats": rails,
        "raw_percentiles": raw_percentiles,
        "capacity_on_placeholder_scales": cap,
        "records": scored,
    }, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
