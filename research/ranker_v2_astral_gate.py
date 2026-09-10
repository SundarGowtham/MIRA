#!/usr/bin/env python
"""
ranker_v2_astral_gate.py — Phase 11 Step 2, THE EXTERNAL GATE (hard stop).

Scores the 35 ASTRAL targets with the REBUILT ranker (core/ranker.py,
RANKER_VERSION 2026-09-03-v2-phase11: precursor_instability, inverse_hull_energy,
n_precursors, slice_competing_phases, precursor_decomposition_match, plus
temperature_economy/driving_force_margin/volatility_risk carried over from v1)
against ASTRAL robot-measured phase purity (Chen/Cross/Sun, Nature Synthesis
2024, arXiv 2304.00743). CPU only, no generation, no training.

Same construction as astral_validation.py (Step 1, ranker v1): two
PredictedRoute objects per target (traditional / predicted precursors) at
that route's own best_T, air, single calcination. lit_T/lit_n_ops stay None
-- these are already-optimal experimental routes, not model predictions
against a literature reference, so temperature_economy is ungraded here (as
in v1); none of the five NEW Phase 11 objectives need that reference.

PRE-REGISTERED THRESHOLD (PHASE_11_INSTRUCTIONS.md Step 2):
  Baselines to beat: validator 17/34 = 50.0% (chance). Ranker v1 20/32 = 62.5%.
  >= 24/35 (69%) -> proceed to Step 3 (RS-SFT / run 4 prep).
  <  24/35       -> STOP. Report which channels disagree with experiment,
                    per target. Do not train.
This script computes and prints the verdict; it does not train anything
regardless of the outcome -- that decision is reported back, not automated.

Usage (tmux):
  PYTHONPATH=. uv run python run_debug_and_analysis/ranker_v2_astral_gate.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr  # noqa: E402

from validator import (  # noqa: E402
    PredictedConditions, PredictedOperation, PredictedPrecursor, PredictedRoute,
    ThermoChecker,
)
from core.ranker import RANKER_VERSION, Ranker, build_precursor_frequency  # noqa: E402

DATA_PATH = Path("misc/astral_validation_set.json")
OUT_JSON = Path("misc/ranker_v2_astral_gate.json")

THRESHOLD_N = 24  # of 35 -- pre-registered, PHASE_11_INSTRUCTIONS.md Step 2
BASELINE_VALIDATOR = 17 / 34
BASELINE_RANKER_V1 = 20 / 32


def build_route(target: str, precursors: list[str], T: float) -> PredictedRoute:
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(f, 1.0) for f in precursors],
        operations=[PredictedOperation(
            type="calcine",
            conditions=PredictedConditions(
                heating_temperature=[float(T)], heating_atmosphere=["air"]))],
    )


def score_with_ranker_v2(rows_meta, ranker):
    out = []
    for m in rows_meta:
        trad_route = build_route(m["target"], m["traditional"], m["trad_best_T"])
        pred_route = build_route(m["target"], m["predicted"], m["pred_best_T"])
        trad_score, trad_bd = ranker.score(trad_route, m["target"], lit_T=None, lit_n_ops=None)
        pred_score, pred_bd = ranker.score(pred_route, m["target"], lit_T=None, lit_n_ops=None)
        out.append({**m, "trad_score": trad_score, "pred_score": pred_score,
                    "trad_breakdown": trad_bd, "pred_breakdown": pred_bd})
    return out


def analyze(rows):
    scores = [r["trad_score"] for r in rows] + [r["pred_score"] for r in rows]
    purities = [r["trad_best_purity"] for r in rows] + [r["pred_best_purity"] for r in rows]
    rho, p = spearmanr(scores, purities)
    agree, ties = 0, 0
    disagreements = []
    for r in rows:
        dv = r["trad_score"] - r["pred_score"]
        dp = r["trad_best_purity"] - r["pred_best_purity"]
        if abs(dv) < 1e-9 or abs(dp) < 1e-9:
            ties += 1
            continue
        if (dv > 0) == (dp > 0):
            agree += 1
        else:
            disagreements.append(r["target"])
    n_decided = len(rows) - ties
    return {
        "spearman_rho": float(rho), "spearman_p": float(p), "n_routes": len(scores),
        "agreement_rate": (agree / n_decided) if n_decided else None,
        "agree": agree, "ties": ties, "n_decided": n_decided, "n_pairs": len(rows),
        "disagreements": disagreements,
    }


def per_channel_disagreement(rows, objective_names):
    """For each channel: how often does its sign (trad - pred) DISAGREE with
    the measured-purity sign, among pairs where both are gradeable? Phase 11
    Step 2: 'if <24/35, report which channels disagree with experiment, per
    target' -- computed regardless of the verdict, since it's cheap and
    informative either way."""
    out = {}
    for name in objective_names:
        agree, disagree, ungradeable, ties = 0, 0, 0, 0
        per_target = []
        for r in rows:
            tv = r["trad_breakdown"].get(name)
            pv = r["pred_breakdown"].get(name)
            dp = r["trad_best_purity"] - r["pred_best_purity"]
            if tv is None or pv is None:
                ungradeable += 1
                continue
            dv = tv - pv
            if abs(dv) < 1e-9 or abs(dp) < 1e-9:
                ties += 1
                continue
            ok = (dv > 0) == (dp > 0)
            (agree if ok else disagree)
            if ok:
                agree += 1
            else:
                disagree += 1
                per_target.append(r["target"])
        out[name] = {"agree": agree, "disagree": disagree, "ties": ties,
                     "ungradeable": ungradeable, "disagreeing_targets": per_target}
    return out


def main():
    data = json.loads(DATA_PATH.read_text())
    targets = data["targets"]
    print(f"loaded {len(targets)} ASTRAL targets", flush=True)

    print(f"loading ranker (RANKER_VERSION={RANKER_VERSION})...", flush=True)
    with open("data/cache/mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(Path("data/cache/pd_index.json"), Path("."))
    freq = build_precursor_frequency(Path("data/raw/synthesis_clean.json"))
    ranker = Ranker(formula_set, thermo, freq)  # class defaults -- the calibrated ones

    print("scoring 35 targets x 2 routes with ranker v2...", flush=True)
    rows = score_with_ranker_v2(targets, ranker)
    stats = analyze(rows)
    from core.ranker import OBJECTIVE_NAMES
    per_channel = per_channel_disagreement(rows, OBJECTIVE_NAMES)

    n_agree = stats["agree"]
    n_pairs = stats["n_pairs"]
    verdict = "PASS -- proceed to Step 3" if n_agree >= THRESHOLD_N else "FAIL -- STOP, do not train"

    print("\n" + "=" * 100)
    print(f"{'target':<16}{'trad_val':>9}{'trad_pur':>9}{'pred_val':>9}{'pred_pur':>9}{'agree?':>8}")
    for r in rows:
        dv = r["trad_score"] - r["pred_score"]
        dp = r["trad_best_purity"] - r["pred_best_purity"]
        tag = "tie" if (abs(dv) < 1e-9 or abs(dp) < 1e-9) else ("YES" if (dv > 0) == (dp > 0) else "no")
        print(f"{r['target']:<16}{r['trad_score']:>9.3f}{r['trad_best_purity']:>9.2f}"
              f"{r['pred_score']:>9.3f}{r['pred_best_purity']:>9.2f}{tag:>8}")

    print("\n" + "=" * 100)
    print("RANKER v2 (Phase 11 rebuild) vs ASTRAL robot-measured phase purity")
    print(f"  Spearman rho={stats['spearman_rho']:.3f}  p={stats['spearman_p']:.4f}  "
          f"n={stats['n_routes']}")
    print(f"  agreement: {n_agree}/{stats['n_decided']} decided pairs "
          f"(ties excluded: {stats['ties']}/{n_pairs})")
    print(f"  as N/35 pairs (Phase 11's own units): {n_agree}/{n_pairs} "
          f"= {n_agree / n_pairs:.1%}")
    print(f"\n  baselines: validator (Arm A) {BASELINE_VALIDATOR:.1%}  |  "
          f"ranker v1 {BASELINE_RANKER_V1:.1%}  |  pre-registered bar {THRESHOLD_N}/35 = "
          f"{THRESHOLD_N/35:.1%}")
    print(f"\n  *** VERDICT: {verdict} ***")
    if stats["disagreements"]:
        print(f"  disagreeing targets ({len(stats['disagreements'])}): "
              f"{stats['disagreements']}")

    print("\n--- per-channel disagreement (which objectives point the wrong way) ---")
    for name, d in per_channel.items():
        n_scored = d["agree"] + d["disagree"]
        rate = f"{d['disagree']}/{n_scored} = {d['disagree']/n_scored:.0%}" if n_scored else "n/a"
        print(f"  {name:<30}disagree={rate:<16}ties={d['ties']:<4}ungradeable={d['ungradeable']}")

    OUT_JSON.write_text(json.dumps({
        "ranker_version": RANKER_VERSION,
        "threshold_n": THRESHOLD_N, "threshold_pct": THRESHOLD_N / 35,
        "baseline_validator": BASELINE_VALIDATOR, "baseline_ranker_v1": BASELINE_RANKER_V1,
        "n_agree": n_agree, "n_pairs": n_pairs, "agree_rate_of_35": n_agree / n_pairs,
        "verdict": verdict,
        "stats": stats,
        "per_channel_disagreement": per_channel,
        "rows": rows,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
