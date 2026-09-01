#!/usr/bin/env python
"""
ranker_capacity_recheck.py — ranker_fixes_instructions.md step 4, THE GATE.

Re-scores runs/gdpo-qlora-beta-ablation-probe/generations.jsonl (4,896
completions, 612 well-formed groups of 8 -- confirmed: every target appears
exactly once, at exactly one step, with exactly 8 samples) with the UPDATED
Ranker (volatility_risk fix, phase_purity dropped from the active reward,
widened scales). No new generation, no GPU. Reports capacity via the SAME
computation probe_hardening.py::capacity_metrics uses (within-group
z-normalize, covariance, sum positive eigenvalues / n_channels), plus
per-channel rails, z-variance, zero-std fraction, and gate failure rates.

Do NOT use misc/ranker_rail_calibration_v2.json's capacity number -- that
run sampled ~1.2 completions/target (no real groups), so its 13.49% was
never a valid capacity estimate, only usable for marginal rail distributions.

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/ranker_capacity_recheck.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reward import ParseFailure, parse_completion  # noqa: E402
from core.ranker import (  # noqa: E402
    OBJECTIVE_NAMES, RANKER_VERSION, Ranker, gate_failure_rates, rail_stats,
)
from probe_hardening import capacity_metrics, diversity_metrics, load_literature  # noqa: E402
from validator import ThermoChecker  # noqa: E402

GENS = Path("runs/gdpo-qlora-beta-ablation-probe/generations.jsonl")
TRIAGE = Path("misc/kononova_triage_results3.json")
SYNTHESIS = Path("data/raw/synthesis_clean.json")
OUT = Path("misc/ranker_capacity_recheck.json")


def main():
    print("loading literature (precursors, max_T, n_ops)...", flush=True)
    lit = load_literature(TRIAGE, SYNTHESIS)

    print("loading validator formula set + PD cache + precursor frequency...", flush=True)
    with open("data/cache/mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(Path("data/cache/pd_index.json"), Path("."))
    from core.ranker import build_precursor_frequency
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq)  # class defaults -- the fixed ones
    print(f"ranker scales in effect: {ranker.scales}", flush=True)

    print(f"loading {GENS} ...", flush=True)
    all_records = [json.loads(l) for l in GENS.open() if l.strip()]
    print(f"  {len(all_records)} completions total", flush=True)

    coverage = [r for r in all_records if r.get("target") in lit]
    n_no_lit = len(all_records) - len(coverage)
    print(f"  literature coverage: {len(coverage)}/{len(all_records)} "
          f"({n_no_lit} targets skipped, no triage/synthesis match)", flush=True)

    scored = []
    n_parse_fail = 0
    for i, r in enumerate(coverage):
        target = r["target"]
        try:
            route = parse_completion(r["completion"], target)
        except ParseFailure:
            route = None
            n_parse_fail += 1
        lit_rec = lit[target]
        reward, info = ranker.score(
            route, target, lit_T=lit_rec.get("max_T"), lit_n_ops=lit_rec.get("n_ops"))
        scored.append({
            "target": target, "reward": reward, "breakdown": info,
            "precursor_set": sorted({p.formula for p in route.precursors})
                            if route is not None else [],
        })
        if (i + 1) % 500 == 0:
            print(f"  scored {i + 1}/{len(coverage)}", flush=True)

    breakdowns = [s["breakdown"] for s in scored]
    rails = rail_stats(breakdowns)
    gates = gate_failure_rates(breakdowns)
    cap = capacity_metrics(scored, OBJECTIVE_NAMES)
    div = diversity_metrics([{"target": s["target"],
                              "precursor_set": s["precursor_set"]} for s in scored])

    print("\n" + "=" * 78)
    print("RANKER CAPACITY RECHECK (proper groups, post-fix)")
    print("=" * 78)
    print(f"n={len(coverage)}  n_groups={cap.get('n_groups')}  "
          f"parse_failures={n_parse_fail}  ranker_version={RANKER_VERSION}")
    print(f"\nCAPACITY: {cap.get('capacity_pct')}%  "
          f"(pre-fix probe was 48.7%)")
    print(f"diversity: {div}")

    print("\n--- gate failure rates ---")
    for g, v in gates.items():
        print(f"  {g:<24}{v}%" if v is not None else f"  {g:<24}n/a")

    print("\n--- objective rails / z-variance / zero-std ---")
    print(f"  {'objective':<24}{'n':>6}{'%@0.0':>8}{'%@1.0':>8}{'mean':>8}"
          f"{'z-var':>9}{'zero-std%':>11}")
    zvar = cap.get("per_channel_z_var", {})
    zstd = cap.get("per_channel_zero_std_pct", {})
    for name in OBJECTIVE_NAMES:
        s = rails[name]
        if s["n"] == 0:
            print(f"  {name:<24}{'0':>6}")
            continue
        print(f"  {name:<24}{s['n']:>6}{s['pct_at_0']:>7.1f}%{s['pct_at_1']:>7.1f}%"
              f"{s['mean']:>8.3f}{zvar.get(name, float('nan')):>9.3f}"
              f"{zstd.get(name, float('nan')):>10.1f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "ranker_version": RANKER_VERSION,
        "n": len(coverage), "n_parse_fail": n_parse_fail,
        "scales": ranker.scales.__dict__,
        "gate_failure_rates": gates,
        "rail_stats": rails,
        "capacity": cap,
        "diversity": div,
        "pre_fix_capacity_pct": 48.7,
        "records": scored,
    }, indent=1))
    print(f"\n-> {OUT}")

    # decision rule (ranker_fixes_instructions.md step 4)
    pct = cap.get("capacity_pct", 0.0)
    print("\n" + "=" * 78)
    if pct >= 60:
        print(f"DECISION: capacity {pct}% >= 60% -> proceed to step 5, launch run 4.")
    elif pct >= 48:
        print(f"DECISION: {pct}% in [48,60) -> STILL LAUNCH RUN 4 "
              f"(48.7% already cleared the pre-registered bar; fixes were "
              f"an optimization, not a prerequisite).")
    else:
        print(f"DECISION: capacity {pct}% < 48% -> DO NOT LAUNCH. Fixes made "
              f"things worse than the pre-fix probe. Revert steps 1-3, "
              f"re-run to confirm ~48.7% is restored, write findings to "
              f"misc/ranker_capacity_recheck.md, leave the GPU idle.")


if __name__ == "__main__":
    main()
