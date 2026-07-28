"""
probe_effective_support.py
--------------------------
Effective-support probe (HANDOFF_2 §7.6): measures the per-target success
mass p̂ of the current SFT policy across the corpus, i.e. the experiment
that decides "saturated vs. headroom" and produces the RL-dataset buckets.

For each sampled target:
  1. build the training-time prompt (SYSTEM_MSG + CLOSED_BOOK_USER with
     live PD stability data, same as stratified_difficulty_eval.py),
  2. draw N samples at high temperature from the checkpoint,
  3. grade each with the patched validator,
  4. p̂ = fraction of samples with reward >= threshold (computed at
     multiple thresholds post-hoc; all rewards are stored).

Strata: (is_fractional x thermo_tier) from the CURRENT triage results —
a parametric difficulty map (integer x discrete is the easy floor,
fractional x ungradeable the frontier), replacing the broken tier ladder.

Output JSON is written incrementally after every target; re-running
resumes by skipping targets already present.

Example (full run):
  PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python probe_effective_support.py \
      --checkpoint runs/sft-qlora-sft-v3-2nd-rank16/final \
      --model Qwen/Qwen3-8B

Smoke:
  PYTHONPATH=. uv run python probe_effective_support.py \
      --checkpoint runs/sft-qlora-sft-v3-2nd-rank16/final \
      --model Qwen/Qwen3-8B --targets-per-stratum 1 --samples-per-target 4
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

try:
    from core.evaluate_batched import load_eval_model, generate_batch
    from core.reward import parse_completion, load_validator
except ImportError:
    from evaluate_batched import load_eval_model, generate_batch
    from core.reward import parse_completion, load_validator

from stratified_difficulty_eval import (
    SYSTEM_MSG,
    CLOSED_BOOK_USER,
    get_stability_data_sync,
)

PASS_THRESHOLDS = [0.65, 0.9]  # p̂ reported at both bars (see docstring)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--triage-results", type=Path,
                   default=Path("misc/kononova_triage_results3.json"))
    p.add_argument("--checkpoint", required=True,
                   help="LoRA checkpoint dir, or 'base'")
    p.add_argument("--model", required=True)
    p.add_argument("--formula-set", type=Path,
                   default=Path("data/cache/mp_formula_set.pkl"))
    p.add_argument("--pd-index", type=Path,
                   default=Path("data/cache/pd_index.json"))
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--targets-per-stratum", type=int, default=30)
    p.add_argument("--samples-per-target", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=11000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("misc/support_probe_results.json"))
    return p.parse_args()


def build_strata(triage_path: Path) -> dict[str, list[str]]:
    """stratum key -> sorted unique targets. Key: 'int|frac' + 'x' + thermo_tier."""
    data = json.load(triage_path.open())
    records = data["records"] if isinstance(data, dict) else data
    strata: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.get("status") != "graded":
            continue
        fam = "frac" if r.get("is_fractional") else "int"
        tier = r.get("thermo_tier", "unknown")
        strata[f"{fam}x{tier}"].add(r["target"])
    return {k: sorted(v) for k, v in sorted(strata.items())}


def p_hats(rewards: list[float]) -> dict[str, float]:
    n = len(rewards)
    return {f"p_hat_ge_{t}": round(sum(1 for r in rewards if r >= t) / n, 4)
            for t in PASS_THRESHOLDS}


def summarize(results: dict[str, dict]) -> None:
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in results.values():
        by_stratum[r["stratum"]].append(r)
    print("\n" + "=" * 74)
    print("EFFECTIVE-SUPPORT PROBE — p̂ distribution per stratum")
    print("=" * 74)
    for stratum, recs in sorted(by_stratum.items()):
        print(f"\n{stratum}  (n_targets={len(recs)})")
        for t in PASS_THRESHOLDS:
            key = f"p_hat_ge_{t}"
            buckets = {"p=0": 0, "0<p<0.5": 0, "0.5<=p<0.95": 0, "p>=0.95": 0}
            for r in recs:
                p = r[key]
                if p == 0:
                    buckets["p=0"] += 1
                elif p < 0.5:
                    buckets["0<p<0.5"] += 1
                elif p < 0.95:
                    buckets["0.5<=p<0.95"] += 1
                else:
                    buckets["p>=0.95"] += 1
            mean_p = sum(r[key] for r in recs) / len(recs)
            print(f"  bar={t:<5} mean p̂={mean_p:.3f}   " +
                  "  ".join(f"{k}:{v}" for k, v in buckets.items()))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- stratified target selection ---
    strata = build_strata(args.triage_results)
    print("strata pool sizes:", {k: len(v) for k, v in strata.items()})
    selected: list[tuple[str, str]] = []  # (stratum, target)
    for stratum, targets in strata.items():
        k = min(args.targets_per_stratum, len(targets))
        if k == 0:
            continue
        for t in random.sample(targets, k):
            selected.append((stratum, t))
    print(f"selected {len(selected)} targets x {args.samples_per_target} samples")
    est_batches = len(selected) * (args.samples_per_target / args.batch_size)
    print(f"~{est_batches:.0f} generation batches expected")

    # --- resume state ---
    results: dict[str, dict] = {}
    if args.out.exists():
        try:
            results = {r["target"]: r for r in json.load(args.out.open())["per_target"]}
            print(f"resuming: {len(results)} targets already done")
        except Exception as e:
            print(f"could not read prior output ({e}); starting fresh")

    print("loading model...", flush=True)
    model, tok = load_eval_model(args.checkpoint, args.model)
    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    n_done = 0
    for stratum, target in selected:
        if target in results:
            n_done += 1
            continue

        stability_text, _ = get_stability_data_sync(target, validator)
        user_msg = CLOSED_BOOK_USER.format(
            target=target, context="", stability_data=stability_text)
        prompt = SYSTEM_MSG + "\n\n" + user_msg

        # draw samples-per-target completions in batches
        completions: list[str] = []
        while len(completions) < args.samples_per_target:
            n = min(args.batch_size, args.samples_per_target - len(completions))
            completions.extend(generate_batch(model, tok, [prompt] * n, args))

        rewards: list[float] = []
        n_error = 0
        for c in completions:
            try:
                route = parse_completion(c, target)
                r, _ = validator.validate(route, target)
                rewards.append(r)
            except Exception:
                rewards.append(0.0)
                n_error += 1

        results[target] = {
            "target": target,
            "stratum": stratum,
            "n_samples": len(completions),
            "n_error": n_error,
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            **p_hats(rewards),
            "rewards": rewards,
        }
        n_done += 1

        # incremental write (resume support)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({"per_target": list(results.values())}, f)

        rate = n_done / (time.time() - t0) * 60
        print(f"  [{n_done}/{len(selected)}] {target:<24} "
              f"p̂(0.9)={results[target]['p_hat_ge_0.9']:.3f}  "
              f"({rate:.1f} targets/min)", file=sys.stderr)

    summarize(results)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
