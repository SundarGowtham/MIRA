#!/usr/bin/env python
"""
astral_three_model_table.py — base vs SFT vs GDPO-300 on the ASTRAL protocol.

Recomputes SFT's summary from misc/astral_model_generations.json (no
regeneration -- its precursor matcher is byte-identical to what generated
misc/astral_gen_base.json / astral_gen_gdpo300.json, verified: pymatgen-
normalized via SynthesisValidator._normalize_formula, not raw string
comparison; see astral_model_generations.py's module docstring for the
verification), so all three are directly comparable.

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/astral_three_model_table.py
"""
from __future__ import annotations

import json
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_PATH = Path("misc/astral_validation_set.json")
SFT_LEGACY = Path("misc/astral_model_generations.json")
OUT = Path("misc/astral_three_model_comparison.json")

MODEL_FILES = {
    "base": Path("misc/astral_gen_base.json"),
    "sft": SFT_LEGACY,
    "gdpo300": Path("misc/astral_gen_gdpo300.json"),
}


def summarize(results: list[dict]) -> dict:
    n_targets_any_pred = sum(1 for r in results
                             if any(s["match"] in ("PREDICTED", "PREDICTED_SUPERSET")
                                   for s in r["samples"]))
    n_targets_any_trad = sum(1 for r in results
                             if any(s["match"] in ("TRADITIONAL", "TRADITIONAL_SUPERSET")
                                   for s in r["samples"]))
    trad_rewards = [s["reward"] for r in results for s in r["samples"]
                   if s["match"] in ("TRADITIONAL", "TRADITIONAL_SUPERSET") and s["reward"] is not None]
    pred_rewards = [s["reward"] for r in results for s in r["samples"]
                   if s["match"] in ("PREDICTED", "PREDICTED_SUPERSET") and s["reward"] is not None]
    all_rewards = [s["reward"] for r in results for s in r["samples"] if s["reward"] is not None]
    all_temps = [s["max_T"] for r in results for s in r["samples"] if s["max_T"] is not None]
    return {
        "n_targets": len(results),
        "n_targets_any_predicted": n_targets_any_pred,
        "n_targets_any_traditional": n_targets_any_trad,
        "mean_reward_traditional_matches": (sum(trad_rewards) / len(trad_rewards)
                                            if trad_rewards else None),
        "mean_reward_predicted_matches": (sum(pred_rewards) / len(pred_rewards)
                                          if pred_rewards else None),
        "mean_reward_all": (sum(all_rewards) / len(all_rewards) if all_rewards else None),
        "mean_max_T": (stats.mean(all_temps) if all_temps else None),
        "median_max_T": (stats.median(all_temps) if all_temps else None),
        "n_temps": len(all_temps),
    }


def main():
    astral = json.loads(DATA_PATH.read_text())
    astral_trad_T = [t["trad_best_T"] for t in astral["targets"]]
    astral_pred_T = [t["pred_best_T"] for t in astral["targets"]]

    summaries = {}
    for tag, path in MODEL_FILES.items():
        if not path.exists():
            print(f"[{tag}] {path} not found yet -- skipping")
            continue
        blob = json.loads(path.read_text())
        results = blob["results"]
        summaries[tag] = summarize(results)
        print(f"[{tag}] n_targets={len(results)}  "
              f"any_predicted={summaries[tag]['n_targets_any_predicted']}  "
              f"any_traditional={summaries[tag]['n_targets_any_traditional']}")

    print("\n" + "=" * 100)
    print(f"{'model':<10}{'any_pred/35':>13}{'any_trad/35':>13}{'mean_reward':>13}"
          f"{'mean_maxT':>11}{'median_maxT':>13}")
    for tag in ("base", "sft", "gdpo300"):
        s = summaries.get(tag)
        if not s:
            print(f"{tag:<10}  (not yet available)")
            continue
        print(f"{tag:<10}{s['n_targets_any_predicted']:>10}/{s['n_targets']:<3}"
              f"{s['n_targets_any_traditional']:>10}/{s['n_targets']:<3}"
              f"{(s['mean_reward_all'] or 0):>13.3f}"
              f"{(s['mean_max_T'] or 0):>11.1f}{(s['median_max_T'] or 0):>13.1f}")
    print(f"\nASTRAL traditional best-condition T: mean={stats.mean(astral_trad_T):.1f} "
          f"median={stats.median(astral_trad_T):.1f}")
    print(f"ASTRAL predicted best-condition T:    mean={stats.mean(astral_pred_T):.1f} "
          f"median={stats.median(astral_pred_T):.1f}")

    OUT.write_text(json.dumps({
        "models": summaries,
        "astral_reference_T": {
            "traditional_mean": stats.mean(astral_trad_T),
            "traditional_median": stats.median(astral_trad_T),
            "predicted_mean": stats.mean(astral_pred_T),
            "predicted_median": stats.median(astral_pred_T),
        },
        "note": "precursor matching normalizes with pymatgen Composition "
               "(SynthesisValidator._normalize_formula), not raw string "
               "comparison -- verified against every ASTRAL reference "
               "formula and spot-checked on real generated data "
               "(NH4H2PO4/MAP vs (NH4)2HPO4/DAP correctly distinguished "
               "as different compounds, not a matcher bug).",
    }, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
