#!/usr/bin/env python
"""
research/distributional/rs_sft_bareoxide_step.py — Phase 15 Task 2b(a):
does RS-SFT's single-round bar-0.9 filter alone explain its 25.1%
bare-alkali-oxide share on ASTRAL, or did something amplify it further?

Three numbers:
  1. Base share (same proxy as the Phase 13 ammonium-propagation test,
     since build_rs_sft_dataset.py never persisted rejected samples --
     there is no base-model dump on the actual RS-SFT training-prompt
     universe (data/rl) to use instead): 8.6% on
     results/astral_gen_n32_base.json (matches analyze3.py exactly).
  2. One-round-filter PREDICTION: apply Task 2's measured pass@0.9 rates
     (bare-oxide-only vs NOT-bare-oxide, computed fresh here directly
     from the same base dump rather than reusing only the carbonate-only
     number) to the base share via Bayes, to get the predicted bare-oxide
     share among filter SURVIVORS.
  3. RS-SFT's ACTUAL training-set share, parsed directly from
     data/rs_sft/rs_sft_train.jsonl + rs_sft_val.jsonl (295 completions,
     real bar-0.9 survivors, not a prediction).

If (3) is close to (2), one filtering round explains it. If (3) is much
higher, or if RS-SFT's ASTRAL-inference-time share (25.1%, from
analyze3.py) is much higher than (3) too, something beyond simple
one-round survivor selection is amplifying the bare-oxide preference
further (e.g. the fine-tuning step itself, or a second-order effect).

Usage (tmux -- uses the real validator/PD cache for the base-dump pass
rate, matching Task 2's methodology exactly):
  uv run python research/distributional/rs_sft_bareoxide_step.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from validator import (PredictedConditions, PredictedOperation,  # noqa: E402
                       PredictedPrecursor, PredictedRoute)
from core.reward import ParseFailure, load_validator, parse_completion  # noqa: E402

BASE_GEN_PATH = Path("results/astral_gen_n32_base.json")
RS_SFT_TRAIN = Path("data/rs_sft/rs_sft_train.jsonl")
RS_SFT_VAL = Path("data/rs_sft/rs_sft_val.jsonl")
OUT_JSON = Path("results/distributional/rs_sft_bareoxide_step.json")

BARE_OXIDE = {"Li2O", "Na2O", "K2O", "Rb2O", "Cs2O"}
BAR = 0.9


def is_bare_oxide_route(precursors: list[str]) -> bool:
    return any(p in BARE_OXIDE for p in precursors)


def main():
    print("loading validator (thermo-aware, real PD cache)...", flush=True)
    validator = load_validator(
        Path("data/cache/mp_formula_set.pkl"),
        Path("data/cache/pd_index.json"),
        Path("."),
    )

    doc = json.loads(BASE_GEN_PATH.read_text())
    bare_all, nonbare_all = [], []
    for r in doc["results"]:
        target = r["target"]
        for s in r["samples"]:
            precursors = s.get("precursors")
            if not precursors:
                continue
            route = PredictedRoute(
                target_formula=target,
                precursors=[PredictedPrecursor(p, 1.0) for p in precursors],
                operations=[PredictedOperation(
                    type="HeatingOperation",
                    conditions=PredictedConditions(
                        heating_temperature=[float(s["max_T"])] if s.get("max_T") else [],
                        heating_atmosphere=["air"]))],
            )
            try:
                reward, _bd = validator.validate(route, target)
            except Exception:
                continue
            (bare_all if is_bare_oxide_route(precursors) else nonbare_all).append(reward)

    n_bare, n_nonbare = len(bare_all), len(nonbare_all)
    n_total = n_bare + n_nonbare
    base_share = n_bare / n_total
    pass_bare = sum(1 for r in bare_all if r >= BAR) / n_bare if n_bare else None
    pass_nonbare = sum(1 for r in nonbare_all if r >= BAR) / n_nonbare if n_nonbare else None

    print(f"base (ASTRAL proxy) share: bare-oxide-containing = {base_share:.1%} "
          f"(n_bare={n_bare}, n_nonbare={n_nonbare}, n_total={n_total})")
    print(f"pass@{BAR}: bare-oxide = {pass_bare:.1%}  non-bare-oxide = {pass_nonbare:.1%}")

    # Bayes: predicted survivor bare-oxide share after ONE round of bar-0.9
    # filtering, applying these two pass rates to the base share.
    num = base_share * pass_bare
    den = base_share * pass_bare + (1 - base_share) * pass_nonbare
    predicted_survivor_share = num / den
    print(f"\nONE-ROUND-FILTER PREDICTION: {predicted_survivor_share:.1%} bare-oxide "
          f"among survivors (Bayes update of {base_share:.1%} base share by the two pass rates)")

    # RS-SFT's actual training set.
    rssft_precursor_lists = []
    n_parse_fail = 0
    for path in [RS_SFT_TRAIN, RS_SFT_VAL]:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target = rec.get("target")
                completion = rec.get("completion", "")
                try:
                    route = parse_completion(completion, target)
                except Exception:
                    n_parse_fail += 1
                    continue
                rssft_precursor_lists.append([p.formula for p in (route.precursors or [])])

    n_rssft = len(rssft_precursor_lists)
    n_rssft_bare = sum(1 for ps in rssft_precursor_lists if is_bare_oxide_route(ps))
    rssft_actual_share = n_rssft_bare / n_rssft if n_rssft else None
    print(f"\nRS-SFT ACTUAL training-set share: {rssft_actual_share:.1%} bare-oxide "
          f"(n={n_rssft}, {n_parse_fail} parse failures excluded)")

    astral_inference_share = 0.251  # analyze3.py, RS-SFT column, this project's own number
    print(f"\nFor reference, RS-SFT's ASTRAL-inference-time bare-oxide share "
          f"(analyze3.py, model generating on ASTRAL prompts): {astral_inference_share:.1%}")

    print("\n== does the training-prompt number explain the ASTRAL number? ==")
    print(f"  base share:                      {base_share:.1%}")
    print(f"  one-round-filter prediction:      {predicted_survivor_share:.1%}")
    print(f"  RS-SFT actual training-set share: {rssft_actual_share:.1%}")
    print(f"  RS-SFT ASTRAL-inference share:    {astral_inference_share:.1%}")
    gap_prediction_vs_actual = rssft_actual_share - predicted_survivor_share
    gap_actual_vs_inference = astral_inference_share - rssft_actual_share
    print(f"\n  actual training-set share vs one-round prediction: {gap_prediction_vs_actual:+.1%}")
    print(f"  ASTRAL-inference share vs actual training-set share: {gap_actual_vs_inference:+.1%}")
    if abs(gap_prediction_vs_actual) < 0.03:
        print("  -> ONE ROUND OF FILTERING EXPLAINS THE TRAINING-SET SHARE.")
    else:
        print("  -> training-set share differs from the one-round prediction by more than 3pp.")
    if gap_actual_vs_inference > 0.05:
        print(f"  -> the fine-tuning step itself (not just survivor selection) further "
              f"amplifies bare-oxide preference by an additional {gap_actual_vs_inference:.1%} "
              f"at inference time on ASTRAL.")

    OUT_JSON.write_text(json.dumps({
        "base_share_astral_proxy": base_share, "n_bare": n_bare, "n_nonbare": n_nonbare,
        "pass_rate_bare": pass_bare, "pass_rate_nonbare": pass_nonbare,
        "one_round_filter_prediction": predicted_survivor_share,
        "rs_sft_actual_training_set_share": rssft_actual_share, "n_rssft": n_rssft,
        "rs_sft_astral_inference_share": astral_inference_share,
        "gap_prediction_vs_actual_training_set": gap_prediction_vs_actual,
        "gap_actual_training_set_vs_astral_inference": gap_actual_vs_inference,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
