#!/usr/bin/env python
"""
astral_validation.py 

External validation: does validator.py / ranker.py score correlate with
ASTRAL robot-measured phase purity (Chen/Cross/Sun, Nature Synthesis 2024,
arXiv 2304.00743)? CPU only, no generation. Arm A (validator.py) unmodified.

For each of 35 targets, builds two PredictedRoute objects (traditional vs
predicted precursors) at that route's own best_T, air atmosphere, single
calcination step, and scores both with validator.py and (for comparison)
core/ranker.py.

CAVEAT (flagged, not hidden): the ASTRAL data gives precursor SPECIES, not
molar amounts, so every precursor gets a placeholder amount=1.0. This is
applied identically to traditional and predicted routes, so it's a
consistent, not asymmetric, simplification -- but it means amount_accuracy
(validator) has no real signal here and should be read with that in mind.
The ranker's temperature_economy/step_economy need a literature T/n_ops
reference that doesn't exist for these already-optimal experimental routes
(there's no "predicted vs literature" gap to measure), so both are left
ungraded (lit_T=None, lit_n_ops=None) for this comparison -- ranker capacity
here reflects the 3 objectives that don't need that reference:
precursor_availability, volatility_risk, driving_force_margin.

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/astral_validation.py
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
from core.reward import load_validator  # noqa: E402

DATA_PATH = Path("misc/astral_validation_set.json")
OUT_JSON = Path("misc/astral_validator_correlation.json")
OUT_FIG = Path("manifold_visualization/figures/astral_correlation.png")


def build_route(target: str, precursors: list[str], T: float) -> PredictedRoute:
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(f, 1.0) for f in precursors],
        operations=[PredictedOperation(
            type="calcine",
            conditions=PredictedConditions(
                heating_temperature=[float(T)], heating_atmosphere=["air"]))],
    )


def score_with_validator(rows_meta):
    validator = load_validator(Path("data/cache/mp_formula_set.pkl"),
                               Path("data/cache/pd_index.json"), Path("."))
    out = []
    for m in rows_meta:
        trad_route = build_route(m["target"], m["traditional"], m["trad_best_T"])
        pred_route = build_route(m["target"], m["predicted"], m["pred_best_T"])
        trad_score, trad_bd = validator.validate(trad_route, m["target"])
        pred_score, pred_bd = validator.validate(pred_route, m["target"])
        out.append({**m, "trad_score": trad_score, "pred_score": pred_score,
                    "trad_breakdown": trad_bd, "pred_breakdown": pred_bd})
    return out


def score_with_ranker(rows_meta):
    from core.ranker import Ranker, build_precursor_frequency
    with open("data/cache/mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(Path("data/cache/pd_index.json"), Path("."))
    freq = build_precursor_frequency(Path("data/raw/synthesis_clean.json"))
    ranker = Ranker(formula_set, thermo, freq)
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
    for r in rows:
        dv = r["trad_score"] - r["pred_score"]
        dp = r["trad_best_purity"] - r["pred_best_purity"]
        if abs(dv) < 1e-9 or abs(dp) < 1e-9:
            ties += 1
            continue
        if (dv > 0) == (dp > 0):
            agree += 1
    n_decided = len(rows) - ties
    return {
        "spearman_rho": float(rho), "spearman_p": float(p), "n_routes": len(scores),
        "agreement_rate": (agree / n_decided) if n_decided else None,
        "agree": agree, "ties": ties, "n_decided": n_decided, "n_pairs": len(rows),
    }


def make_scatter(validator_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trad_scores = [r["trad_score"] for r in validator_rows]
    trad_purity = [r["trad_best_purity"] for r in validator_rows]
    pred_scores = [r["pred_score"] for r in validator_rows]
    pred_purity = [r["pred_best_purity"] for r in validator_rows]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(trad_scores, trad_purity, label="traditional precursors",
              color="tab:blue", alpha=0.8)
    ax.scatter(pred_scores, pred_purity, label="predicted precursors",
              color="tab:orange", alpha=0.8)
    ax.set_xlabel("validator score (Arm A)")
    ax.set_ylabel("measured phase purity (ASTRAL robot)")
    ax.set_title("Validator score vs. robot-measured phase purity\n"
                 "(35 ASTRAL targets, 70 routes)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"-> {OUT_FIG}")


def main():
    data = json.loads(DATA_PATH.read_text())
    targets = data["targets"]
    print(f"loaded {len(targets)} ASTRAL targets", flush=True)

    print("scoring with validator.py (Arm A, unmodified)...", flush=True)
    validator_rows = score_with_validator(targets)
    validator_stats = analyze(validator_rows)

    print("scoring with core/ranker.py (Arm B, for comparison)...", flush=True)
    ranker_rows = score_with_ranker(targets)
    ranker_stats = analyze(ranker_rows)

    print("\n" + "=" * 100)
    print(f"{'target':<16}{'trad_val':>9}{'trad_pur':>9}{'pred_val':>9}{'pred_pur':>9}")
    for r in validator_rows:
        print(f"{r['target']:<16}{r['trad_score']:>9.3f}{r['trad_best_purity']:>9.2f}"
              f"{r['pred_score']:>9.3f}{r['pred_best_purity']:>9.2f}")

    print("\n" + "=" * 100)
    print("VALIDATOR (Arm A):")
    print(f"  Spearman rho={validator_stats['spearman_rho']:.3f}  "
          f"p={validator_stats['spearman_p']:.4f}  n={validator_stats['n_routes']}")
    print(f"  agreement rate: {validator_stats['agree']}/{validator_stats['n_decided']} "
          f"= {validator_stats['agreement_rate']:.1%}  "
          f"(ties excluded: {validator_stats['ties']}/{validator_stats['n_pairs']} pairs; "
          f"chance = 50%)")

    print("\nRANKER (Arm B, for comparison):")
    print(f"  Spearman rho={ranker_stats['spearman_rho']:.3f}  "
          f"p={ranker_stats['spearman_p']:.4f}  n={ranker_stats['n_routes']}")
    print(f"  agreement rate: {ranker_stats['agree']}/{ranker_stats['n_decided']} "
          f"= {ranker_stats['agreement_rate']:.1%}  "
          f"(ties excluded: {ranker_stats['ties']}/{ranker_stats['n_pairs']} pairs)")

    make_scatter(validator_rows)

    OUT_JSON.write_text(json.dumps({
        "source": data["source"],
        "validator": {"stats": validator_stats, "rows": validator_rows},
        "ranker": {"stats": ranker_stats, "rows": ranker_rows},
        "caveats": [
            "precursor amounts are placeholder 1.0 (ASTRAL data gives species "
            "only, not molar ratios) -- applied identically to both routes, "
            "so amount_accuracy specifically carries no real signal here",
            "ranker's temperature_economy/step_economy are ungraded (lit_T/"
            "lit_n_ops=None) -- these are already-optimal experimental "
            "routes, not model predictions being compared to a literature "
            "reference, so that comparison doesn't apply",
        ],
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
