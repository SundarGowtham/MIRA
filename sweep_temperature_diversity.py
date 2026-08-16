"""
sweep_temperature_diversity.py
------------------------------
Step 1 of the run-3 diagnostics: is the SFT policy's low route diversity
(mean ~2 distinct precursor sets per group, GDPO_RUN1 diagnostics D5)
reachable by sampling temperature, or structural?

For ~20 stratified targets x T in {1.0, 1.2, 1.5} x G=8 samples (closed-book
prompts — proven equivalent, 2x cheaper), count DISTINCT PRECURSOR SETS per
group of 8. If diversity opens with T, run 3 uses raised T / clip-higher;
if it stays ~2, the policy is peaked beyond sampling reach (support-seeding
territory).

Incremental JSON output, resume-safe.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_batched import load_eval_model, generate_batch  # noqa: E402
from core.reward import parse_completion  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402

# TEMPS = [1.0, 1.2, 1.5]
TEMPS = [1.0, 1.2, 1.5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/sft-qlora-sft-v3-2nd-rank16/final")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--targets-per-stratum", type=int, default=4)
    p.add_argument("--samples-per-target", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("misc/temp_diversity_sweep.json"))
    return p.parse_args()


def closed_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def precursor_set(completion: str, target: str):
    try:
        route = parse_completion(completion, target)
        return frozenset(p.formula for p in route.precursors)
    except Exception:
        return None


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # stratified target pool from the RL dataset (val = never trained on)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for l in open("data/rl/val.jsonl"):
        r = json.loads(l)
        by_stratum[r["stratum"]].append(r["target"])
    selected = []
    for s, ts in sorted(by_stratum.items()):
        for t in random.sample(ts, min(args.targets_per_stratum, len(ts))):
            selected.append((s, t))
    print(f"selected {len(selected)} targets", flush=True)

    results: dict[str, dict] = {}
    if args.out.exists():
        results = {r["target"]: r for r in json.load(args.out.open())["per_target"]}
        print(f"resuming: {len(results)} targets done", flush=True)

    model, tok = load_eval_model(args.checkpoint, args.model)
    t0 = time.time()
    for stratum, target in selected:
        rec = results.setdefault(target, {"target": target, "stratum": stratum, "temps": {}})
        for temp in TEMPS:
            key = f"T{temp}"
            if key in rec["temps"]:
                continue
            args.temperature = temp
            comps = []
            while len(comps) < args.samples_per_target:
                n = min(args.batch_size, args.samples_per_target - len(comps))
                comps.extend(generate_batch(model, tok, [closed_prompt(target)] * n, args))
            sets = [precursor_set(c, target) for c in comps]
            valid = [s for s in sets if s]
            rec["temps"][key] = {
                "n_distinct_precursor_sets": len(set(valid)),
                "n_valid": len(valid),
                "precursor_sets": [sorted(s) for s in set(valid)],
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w") as f:
                json.dump({"per_target": list(results.values())}, f)
        d = rec["temps"]
        print(f"  [{target}] " + "  ".join(
            f"{k}: {v['n_distinct_precursor_sets']}sets" for k, v in sorted(d.items()))
            + f"  ({(time.time()-t0)/60:.0f} min)", file=sys.stderr, flush=True)

    # summary
    print("\n=== TEMPERATURE DIVERSITY SWEEP ===")
    agg = defaultdict(list)
    for rec in results.values():
        for k, v in rec["temps"].items():
            agg[k].append(v["n_distinct_precursor_sets"])
    for k in sorted(agg):
        v = agg[k]
        print(f"  {k}: mean distinct sets/group = {sum(v)/len(v):.2f}  "
              f"(min {min(v)}, max {max(v)}, n={len(v)})")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
