"""
probe_passk.py
--------------
Step 2 of the run-3 diagnostics: the pass@k baseline. The central claim
(sharpening vs expansion) is a claim about the capability boundary, and
pass@1 cannot see it. Measures pass@k (unbiased estimator) for three
models on the same held-out stratified targets, graded by the full
validator:

  base   = Qwen/Qwen3-8B (no adapter)
  sft    = runs/sft-qlora-sft-v3-2nd-rank16/final
  gdpo   = runs/gdpo-qlora-gdpo-v3/checkpoint-300

Closed-book prompts (proven equivalent, 2x cheaper). Success bar: reward
>= 0.9 (also stores all rewards for re-bucketing). Incremental, resume-safe.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_batched import load_eval_model, generate_batch  # noqa: E402
from core.reward import parse_completion, load_validator  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402

# Decision-relevant pair first (gdpo300 vs sft); base is context.
MODELS = [
    ("sft", "runs/sft-qlora-sft-v3-2nd-rank16/final"),
    ("gdpo300", "runs/gdpo-qlora-gdpo-v3/checkpoint-300"),
    ("base", "base"),
]
KS = [1, 2, 4, 8, 16, 32, 48]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--targets-per-stratum", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--bar", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formula-set", type=Path, default=Path("data/cache/mp_formula_set.pkl"))
    p.add_argument("--pd-index", type=Path, default=Path("data/cache/pd_index.json"))
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("misc/passk_baseline.json"))
    p.add_argument("--extra-model", action="append", default=[],
                   metavar="NAME:PATH",
                   help="extra checkpoint to evaluate after the built-in three "
                        "(e.g. the equal-compute SFT continuation arm: "
                        "--extra-model sftcont:runs/sft-qlora-sft-v3-cont/final). "
                        "Repeatable; resume-safe per model.")
    return p.parse_args()


def closed_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for l in open("data/rl/val.jsonl"):
        r = json.loads(l)
        by_stratum[r["stratum"]].append(r["target"])
    selected = []
    for s, ts in sorted(by_stratum.items()):
        for t in random.sample(ts, min(args.targets_per_stratum, len(ts))):
            selected.append((s, t))
    print(f"selected {len(selected)} targets", flush=True)

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        sys.exit("FATAL: thermo_checker is None")

    results: dict[str, dict] = {}
    if args.out.exists():
        results = {r["target"]: r for r in json.load(args.out.open())["per_target"]}
        print(f"resuming: {len(results)} targets have partial results", flush=True)

    models = list(MODELS)
    for spec in args.extra_model:
        name, _, path = spec.partition(":")
        if not path:
            sys.exit(f"--extra-model must be NAME:PATH, got {spec!r}")
        models.append((name, path))

    for model_name, ckpt in models:
        print(f"\n=== loading {model_name} ({ckpt}) ===", flush=True)
        model, tok = load_eval_model(ckpt, args.model)
        t0 = time.time()
        for stratum, target in selected:
            rec = results.setdefault(target, {"target": target, "stratum": stratum, "models": {}})
            if model_name in rec["models"]:
                continue
            rewards = []
            while len(rewards) < args.n_samples:
                n = min(args.batch_size, args.n_samples - len(rewards))
                comps = generate_batch(model, tok, [closed_prompt(target)] * n, args)
                for c in comps:
                    try:
                        route = parse_completion(c, target)
                        r, _ = validator.validate(route, target)
                        rewards.append(r)
                    except Exception:
                        rewards.append(0.0)
            c = sum(1 for r in rewards if r >= args.bar)
            rec["models"][model_name] = {
                "n": len(rewards), "n_success": c,
                "mean_reward": round(sum(rewards) / len(rewards), 4),
                **{f"pass@{k}": round(pass_at_k(len(rewards), c, k), 4) for k in KS if k <= len(rewards)},
                "rewards": rewards,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w") as f:
                json.dump({"per_target": list(results.values())}, f)
            pk = rec["models"][model_name]
            print(f"  {target:<26} pass@1={pk['pass@1']:.3f} pass@16={pk.get('pass@16', 0):.3f} "
                  f"pass@48={pk.get('pass@48', 0):.3f}  ({(time.time()-t0)/60:.0f} min)",
                  file=sys.stderr, flush=True)
        del model
        torch.cuda.empty_cache()

    # summary: mean pass@k per model
    print("\n=== PASS@K SUMMARY (mean over targets) ===")
    for model_name, _ in models:
        row = [model_name]
        for k in [1, 8, 16, 48]:
            vals = [r["models"][model_name][f"pass@{k}"]
                    for r in results.values()
                    if model_name in r["models"] and f"pass@{k}" in r["models"][model_name]]
            row.append(f"pass@{k}={sum(vals)/len(vals):.3f}" if vals else f"pass@{k}=n/a")
        print("  " + "  ".join(row))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
