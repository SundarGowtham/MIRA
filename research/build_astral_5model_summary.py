#!/usr/bin/env python
"""
build_astral_5model_summary.py — aggregates the five independently-generated
astral_gen_n32_*.json files (base, sft, gdpo300, format_only, rs_sft) into
one results/astral_5model_n32.json for the README/results table. Computed
from the tracked per-model artifacts, not hand-typed -- run this to
regenerate results/astral_5model_n32.json if any of the five source files
change.

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/build_astral_5model_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

MODELS = {
    "base": Path("results/astral_gen_n32_base.json"),
    "full_sft": Path("results/astral_gen_n32_sft.json"),
    "gdpo300": Path("results/astral_gen_n32_gdpo300.json"),
    "format_only_sft": Path("results/astral_gen_n32_format_only.json"),
    "rs_sft_from_base": Path("results/astral_gen_n32_rs_sft.json"),
}
OUT = Path("results/astral_5model_n32.json")


def summarize(tag: str, path: Path) -> dict:
    d = json.loads(path.read_text())
    s = d["summary"]
    results = d["results"]
    total = sum(len(r["samples"]) for r in results)
    n_fail = sum(1 for r in results for x in r["samples"] if x["match"] == "PARSE_FAILURE")
    pred = s.get("mean_reward_predicted_matches")
    trad = s.get("mean_reward_traditional_matches")
    gap = (pred - trad) if (pred is not None and trad is not None) else None
    return {
        "tag": tag,
        "source_file": str(path),
        "n_targets": s["n_targets"],
        "any_predicted": s["n_targets_any_predicted"],
        "any_traditional": s["n_targets_any_traditional"],
        "mean_reward_all": s["mean_reward_all"],
        "mean_reward_predicted_matches": pred,
        "mean_reward_traditional_matches": trad,
        "validator_gap_predicted_minus_traditional": gap,
        "mean_max_T_C": s["mean_max_T"],
        "median_max_T_C": s["median_max_T"],
        "parse_rate": (total - n_fail) / total,
        "n_samples": total,
    }


def main():
    summaries = {tag: summarize(tag, path) for tag, path in MODELS.items()}
    OUT.write_text(json.dumps({
        "protocol": "35 ASTRAL targets x 32 closed-book samples per model, "
                    "run_debug_and_analysis/astral_model_generations.py",
        "source": "Chen/Cross/Sun, Nature Synthesis 2024, arXiv 2304.00743",
        "models": summaries,
    }, indent=1))
    print(f"-> {OUT}")
    for tag, s in summaries.items():
        print(f"{tag:20s} predicted={s['any_predicted']:2d}/35  "
              f"traditional={s['any_traditional']:2d}/35  "
              f"gap={s['validator_gap_predicted_minus_traditional']}  "
              f"parse_rate={s['parse_rate']:.1%}")


if __name__ == "__main__":
    main()
