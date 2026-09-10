#!/usr/bin/env python
"""
astral_model_generations.py — URGENT_PRESENTATION_PREP.md Step 4, extended
for the 3-model comparison (base / SFT / GDPO-300).

Does the model ever propose the ASTRAL-predicted (better) precursors, or does
it always fall back to the traditional (corpus-common) ones? 8 closed-book
samples per ASTRAL target (35 x 8 = 280 generations). Resume-safe: writes
after every target.

Precursor matching normalizes with pymatgen (SynthesisValidator.
_normalize_formula = Composition(f).reduced_formula) -- NOT raw string
comparison. Verified against every ASTRAL reference formula (all parse
cleanly, no fallback-to-raw) and spot-checked on real generated data:
NH4H2PO4 (monoammonium phosphate, MAP) normalizes to "PH6NO4" and correctly
matches wherever the model generates an equivalent formula; (NH4)2HPO4
(diammonium phosphate, DAP) normalizes to "PH9(NO2)2" and is correctly NOT
matched against the traditional MAP reference -- DAP and MAP are genuinely
different compounds, not a normalization bug.

Usage (tmux):
  PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python -u run_debug_and_analysis/astral_model_generations.py \
    --checkpoint base --tag base --out misc/astral_gen_base.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_batched import load_eval_model, generate_batch  # noqa: E402
from core.reward import parse_completion, load_validator  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402
from validator import SynthesisValidator  # noqa: E402

DATA_PATH = Path("misc/astral_validation_set.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs/sft-qlora-sft-v3-2nd-rank16/final",
                   help="'base' for no-adapter Qwen3-8B, or a checkpoint path.")
    p.add_argument("--tag", default="sft", help="model tag stamped into the output.")
    p.add_argument("--out", type=Path, default=None,
                   help="default: misc/astral_gen_<tag>.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=8,
                   help="closed-book samples per target (Phase 11 Step 0: 32)")
    return p.parse_args()


def closed_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def route_summary(route):
    precs = {SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}
    temps = []
    for op in route.operations:
        temps.extend(op.conditions.heating_temperature or [])
    return precs, (max(temps) if temps else None)


def classify(precs: set, trad: set, pred: set) -> str:
    if precs == trad:
        return "TRADITIONAL"
    if precs == pred:
        return "PREDICTED"
    if pred <= precs:
        return "PREDICTED_SUPERSET"
    if trad <= precs:
        return "TRADITIONAL_SUPERSET"
    return "OTHER"


def main():
    args = parse_args()
    out = args.out or Path(f"misc/astral_gen_{args.tag}.json")
    import random
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = json.loads(DATA_PATH.read_text())
    validator = load_validator(Path("data/cache/mp_formula_set.pkl"),
                               Path("data/cache/pd_index.json"), Path("."))
    model, tok = load_eval_model(checkpoint=args.checkpoint, model_name="Qwen/Qwen3-8B")
    gen_args = SimpleNamespace(temperature=1.0, top_p=0.95, max_new_tokens=8192)

    results = []
    done_targets = set()
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            results = prev.get("results", [])
            done_targets = {r["target"] for r in results}
            print(f"resuming: {len(done_targets)} targets already done", flush=True)
        except Exception as e:
            print(f"could not resume ({e}); starting fresh", flush=True)

    for i, t in enumerate(data["targets"]):
        target = t["target"]
        if target in done_targets:
            continue
        trad = {SynthesisValidator._normalize_formula(f) for f in t["traditional"]}
        pred = {SynthesisValidator._normalize_formula(f) for f in t["predicted"]}

        prompt = closed_prompt(target)
        # Chunked at 8/call (the proven-working batch size, CLAUDE.md's GPU
        # memory rule) -- args.n_samples=32 in one generate_batch call OOMs
        # the 32GB card (KV-cache scales with batch x max_new_tokens=8192).
        completions = []
        remaining = args.n_samples
        while remaining > 0:
            chunk = min(8, remaining)
            completions.extend(generate_batch(model, tok, [prompt] * chunk, gen_args))
            remaining -= chunk

        samples = []
        for comp in completions:
            try:
                route = parse_completion(comp, target)
                reward, _ = validator.validate(route, target)
                precs, max_T = route_summary(route)
                match = classify(precs, trad, pred)
            except Exception:
                reward, precs, max_T, match = None, [], None, "PARSE_FAILURE"
            samples.append({"precursors": sorted(precs), "max_T": max_T,
                            "reward": reward, "match": match})

        results.append({
            "target": target, "traditional": t["traditional"], "predicted": t["predicted"],
            "trad_best_T": t["trad_best_T"], "pred_best_T": t["pred_best_T"],
            "trad_best_purity": t["trad_best_purity"], "pred_best_purity": t["pred_best_purity"],
            "samples": samples,
        })
        out.write_text(json.dumps({"results": results}, indent=1))
        n_trad = sum(1 for s in samples if s["match"] in ("TRADITIONAL", "TRADITIONAL_SUPERSET"))
        n_pred = sum(1 for s in samples if s["match"] in ("PREDICTED", "PREDICTED_SUPERSET"))
        print(f"[{i+1}/{len(data['targets'])}] {target}: "
              f"trad={n_trad}/{args.n_samples} pred={n_pred}/{args.n_samples}",
              flush=True)

    # summary
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
    import statistics as stats
    print("\n" + "=" * 78)
    print(f"targets where model EVER proposed predicted set (or superset): "
          f"{n_targets_any_pred}/{len(results)}")
    print(f"targets where model EVER proposed traditional set (or superset): "
          f"{n_targets_any_trad}/{len(results)}")
    print(f"mean validator reward | traditional matches: "
          f"{sum(trad_rewards)/len(trad_rewards):.3f} (n={len(trad_rewards)})"
          if trad_rewards else "no traditional matches")
    print(f"mean validator reward | predicted matches: "
          f"{sum(pred_rewards)/len(pred_rewards):.3f} (n={len(pred_rewards)})"
          if pred_rewards else "no predicted matches")
    print(f"mean validator reward | all samples: "
          f"{sum(all_rewards)/len(all_rewards):.3f} (n={len(all_rewards)})"
          if all_rewards else "no valid rewards")
    print(f"proposed max-T | mean={stats.mean(all_temps):.1f} "
          f"median={stats.median(all_temps):.1f} (n={len(all_temps)})"
          if all_temps else "no temps")

    out.write_text(json.dumps({
        "tag": args.tag, "checkpoint": args.checkpoint, "n_samples": args.n_samples,
        "results": results,
        "summary": {
            "n_targets": len(results),
            "n_targets_any_predicted": n_targets_any_pred,
            "n_targets_any_traditional": n_targets_any_trad,
            "mean_reward_traditional_matches": (sum(trad_rewards)/len(trad_rewards)
                                                if trad_rewards else None),
            "mean_reward_predicted_matches": (sum(pred_rewards)/len(pred_rewards)
                                              if pred_rewards else None),
            "mean_reward_all": (sum(all_rewards)/len(all_rewards) if all_rewards else None),
            "mean_max_T": (stats.mean(all_temps) if all_temps else None),
            "median_max_T": (stats.median(all_temps) if all_temps else None),
            "n_temps": len(all_temps),
        },
    }, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
