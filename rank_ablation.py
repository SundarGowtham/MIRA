"""
rank_ablation.py
----------------
Run a controlled comparison of LoRA ranks for MIRA SFT.

Trains identical configurations with r=16 and r=32 across multiple seeds,
evaluates on held-out test set, reports mean ± std of validator scores.
This is the empirical answer to "should we use r=32?"

Compute estimate (Qwen3-8B QLoRA on A100):
  ~25 min per epoch × 3 epochs × 2 ranks × 3 seeds = ~7.5 hours total

If GPU budget is tight, reduce to 2 seeds (~5 hours) or 1 seed (~2.5 hours).
With 1 seed you lose variance estimates but still get a point comparison.

Usage:
    uv run python rank_ablation.py
    uv run python rank_ablation.py --seeds 42 7 1337
    uv run python rank_ablation.py --ranks 16 32 64        # test r=64 too
    uv run python rank_ablation.py --quick                  # 1 epoch, 1 seed for smoke test
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).parent


def run_one(rank: int, seed: int, epochs: int,
            output_root: Path, dry_run: bool = False) -> dict:
    """
    Train one config and evaluate.

    Calls into your existing train.py / eval.py — pointing at the right
    output dir and passing rank/seed/epochs as flags.

    Returns:
      {
        "rank": int, "seed": int,
        "train_seconds": float,
        "test_mean_reward": float,
        "test_format_fail_rate": float,
        "checkpoint_path": str,
      }
    """
    run_name = f"sft-qlora-v3-rank{rank}-seed{seed}"
    output_dir = output_root / run_name

    train_cmd = [
        "python", "-u", "train.py", "sft",
        "--adapter",      "qlora",
        "--model",        "Qwen/Qwen3-8B",
        "--data-prefix",  "sft",
        "--data-dir",     "data/sft",
        "--lora-r",       str(rank),
        "--lora-alpha",   str(rank * 2),
        "--lora-dropout", "0.05",
        "--seed",         str(seed),
        "--output-root",  str(output_root),
        "--tag",          f"v3-rank{rank}-seed{seed}",
    ]

    eval_cmd = [
        "python", "-u", "evaluate_batched.py",
        "--checkpoint", str(output_dir / "final"),
        "--model",      "Qwen/Qwen3-8B",
        "--data-dir",   "data/sft",
        "--data-prefix", "sft",
        "--split",      "test",
        "--tag",        f"v3-rank{rank}-seed{seed}",
        "--skip-thermo",
    ]

    if dry_run:
        print(f"[dry-run] would run: {' '.join(train_cmd)}")
        print(f"[dry-run] then: {' '.join(eval_cmd)}")
        return {"rank": rank, "seed": seed, "dry_run": True}

    print(f"\n{'='*60}\nTraining {run_name}\n{'='*60}")
    t0 = time.time()
    subprocess.run(train_cmd, check=True)
    train_seconds = time.time() - t0

    print(f"\nEvaluating {run_name}")
    subprocess.run(eval_cmd, check=True)

    with (output_dir / "test_results.json").open() as f:
        results = json.load(f)
    agg = results.get("aggregate", results)  # evaluate_batched wraps under "aggregate"

    return {
        "rank": rank,
        "seed": seed,
        "train_seconds": train_seconds,
        "test_mean_reward": agg.get("mean_reward", 0),
        "test_format_fail_rate": agg.get("format_fail_rate", 0),
        "test_thermo_favorable_mean": agg.get("mean_thermodynamic_favorable", 0),
        "checkpoint_path": str(output_dir),
    }


def summarize(results: list[dict], ranks: list[int]) -> dict:
    """Aggregate results: mean ± std per rank, and report which wins."""
    summary = {}
    failed = [r for r in results if "error" in r]
    if failed:
        summary["failed_runs"] = [
            {"rank": r["rank"], "seed": r["seed"]} for r in failed
        ]

    for rank in ranks:
        rank_results = [
            r for r in results
            if r["rank"] == rank and not r.get("dry_run") and "error" not in r
        ]
        if not rank_results:
            continue
        rewards = [r["test_mean_reward"] for r in rank_results]
        train_times = [r["train_seconds"] for r in rank_results]
        summary[f"rank_{rank}"] = {
            "n_seeds": len(rank_results),
            "mean_reward": round(mean(rewards), 4),
            "std_reward":  round(stdev(rewards), 4) if len(rewards) > 1 else None,
            "mean_train_seconds": round(mean(train_times), 1),
            "individual_rewards": [round(r, 4) for r in rewards],
        }

    # Statistical comparison
    if len(ranks) == 2 and all(f"rank_{r}" in summary for r in ranks):
        r1, r2 = ranks
        m1 = summary[f"rank_{r1}"]["mean_reward"]
        m2 = summary[f"rank_{r2}"]["mean_reward"]
        s1 = summary[f"rank_{r1}"]["std_reward"]
        s2 = summary[f"rank_{r2}"]["std_reward"]

        if s1 is not None and s2 is not None:
            # Welch's t-test approximation for effect size
            pooled_std = ((s1**2 + s2**2) / 2) ** 0.5
            cohens_d = (m2 - m1) / pooled_std if pooled_std > 0 else 0

            summary["comparison"] = {
                "ranks_compared": [r1, r2],
                "absolute_diff": round(m2 - m1, 4),
                "relative_diff_pct": round(100 * (m2 - m1) / m1, 2) if m1 else 0,
                "cohens_d": round(cohens_d, 3),
                "interpretation": (
                    "negligible" if abs(cohens_d) < 0.2 else
                    "small"      if abs(cohens_d) < 0.5 else
                    "medium"     if abs(cohens_d) < 0.8 else
                    "large"
                ),
                "verdict": (
                    f"Use r={r1} (simpler, equivalent performance)" if abs(cohens_d) < 0.2
                    else f"r={r2} is meaningfully better (Cohen's d = {cohens_d:.2f})"
                    if cohens_d > 0.2
                    else f"r={r1} is meaningfully better (Cohen's d = {cohens_d:.2f})"
                ),
            }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1337])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="1 epoch, 1 seed per rank — smoke test only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.seeds = [42]
        print("[quick mode] 1 epoch, 1 seed — not statistically rigorous")

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Rank ablation experiment")
    print(f"  Ranks: {args.ranks}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Total runs: {len(args.ranks) * len(args.seeds)}")
    print()

    results = []
    for rank in args.ranks:
        for seed in args.seeds:
            try:
                result = run_one(
                    rank=rank, seed=seed, epochs=args.epochs,
                    output_root=args.output_root, dry_run=args.dry_run,
                )
                results.append(result)
                # Save partial after each run so a crash doesn't lose everything
                with (args.output_root / "results.json").open("w") as f:
                    json.dump(results, f, indent=2)
            except subprocess.CalledProcessError as e:
                print(f"FAILED: rank={rank} seed={seed}: {e}")
                results.append({"rank": rank, "seed": seed, "error": str(e)})

    if args.dry_run:
        return

    summary = summarize(results, args.ranks)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    with (args.output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
