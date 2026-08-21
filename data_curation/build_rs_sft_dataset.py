"""
build_rs_sft_dataset.py — RS-SFT (rejection-sampling SFT) control arm.

Claude's replacement for the equal-compute SFT continuation: "sample
8/target from SFT, filter at the same bar with the same validator, fine-tune
on survivors. 24,000 steps ≈ 200 epochs is an overfit strawman — RS-SFT is
the standard RLVR control and isolates 'does the RL objective matter' from
'does verifier-filtered data matter.'"

Samples targets uniformly from data/rl (the same SFT-disjoint universe runs
1-2 drew from), generates --samples-per-target completions from the SFT
checkpoint with closed-book prompts, keeps survivors at --bar under the full
validator, and writes an SFT-format jsonl ({prompt, completion, ...}) that
train.py sft --data-prefix rs_sft can consume directly.

  --keep best  -> highest-scoring survivor per target (cleaner, smaller set)
  --keep all   -> every survivor (RAFT-style; more data, more duplication)

Resume-safe: survivors are appended per target and skipped on rerun.

Usage (GPU, tmux):
  PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python data_curation/build_rs_sft_dataset.py --targets 400
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from evaluate_batched import load_eval_model, generate_batch  # noqa: E402
from core.reward import parse_completion, load_validator  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402
from validator import VALIDATOR_VERSION  # noqa: E402

SFT_CKPT = "runs/sft-qlora-sft-v3-2nd-rank16/final"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--checkpoint", default=SFT_CKPT)
    p.add_argument("--rl-dir", type=Path, default=Path("data/rl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/rs_sft"))
    p.add_argument("--targets", type=int, default=400)
    p.add_argument("--samples-per-target", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--bar", type=float, default=0.9,
                   help="survivor bar — the same bar the pass@k comparison uses")
    p.add_argument("--keep", choices=["best", "all"], default="best")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formula-set", type=Path, default=Path("data/cache/mp_formula_set.pkl"))
    p.add_argument("--pd-index", type=Path, default=Path("data/cache/pd_index.json"))
    p.add_argument("--project-root", type=Path, default=Path("."))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pool = []
    with (args.rl_dir / "rl_train.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pool.append((r["target"], r.get("stratum")))
    seen = sorted({t for t, _ in pool})
    random.shuffle(seen)
    targets = seen[: args.targets]
    strata = dict(pool)
    print(f"pool {len(seen)} targets -> sampling {len(targets)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "manifest.json").exists():
        sys.exit(f"{args.out_dir} already finalized (manifest.json exists); "
                 f"delete it to extend the dataset")
    out_path = args.out_dir / "rs_sft_train.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["target"])
        print(f"resuming: {len(done)} targets already have survivors", flush=True)

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        sys.exit("FATAL: thermo_checker is None")
    model, tok = load_eval_model(args.checkpoint, args.model)

    out_fp = out_path.open("a", buffering=1)
    n_surv = 0
    t0 = time.time()
    for k, target in enumerate(targets):
        if target in done:
            continue
        prompt = (SYSTEM_MSG + "\n\nTarget: " + target +
                  "\n\nProvide your synthesis route as a JSON object.")
        scored = []  # (reward, completion)
        while len(scored) < args.samples_per_target:
            n = min(args.batch_size, args.samples_per_target - len(scored))
            comps = generate_batch(model, tok, [prompt] * n, args)
            for c in comps:
                try:
                    route = parse_completion(c, target)
                    r, _ = validator.validate(route, target)
                    scored.append((r, c))
                except Exception:
                    scored.append((0.0, c))
        survivors = [(r, c) for r, c in scored if r >= args.bar]
        if args.keep == "best" and survivors:
            survivors = [max(survivors, key=lambda x: x[0])]
        for r, c in survivors:
            out_fp.write(json.dumps({
                "prompt": prompt, "completion": c, "target": target,
                "stratum": strata.get(target), "reward": r,
                "source": "rs_sft", "validator_version": VALIDATOR_VERSION,
            }) + "\n")
            n_surv += 1
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(targets)} targets, {n_surv} survivors, "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)

    # The trainer's convention requires <prefix>_val.jsonl: hold out every
    # 25th survivor (eval loss is not decision-relevant for this arm).
    out_fp.close()
    rows = [l for l in out_path.open() if l.strip()]
    val = rows[::25]
    train = [l for i, l in enumerate(rows) if i % 25 != 0]
    with out_path.open("w") as f:
        f.writelines(train)
    with (args.out_dir / "rs_sft_val.jsonl").open("w") as f:
        f.writelines(val)
    manifest = {
        "checkpoint": args.checkpoint, "bar": args.bar, "keep": args.keep,
        "samples_per_target": args.samples_per_target, "seed": args.seed,
        "targets_attempted": len(targets), "survivors": len(rows),
        "train": len(train), "val": len(val),
        "prompt_format": "closed-book (SYSTEM_MSG + bare Target)",
        "validator_version": VALIDATOR_VERSION,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {len(rows)} survivors from {len(targets)} targets -> "
          f"{out_path} (+{len(val)} val)")


if __name__ == "__main__":
    main()
