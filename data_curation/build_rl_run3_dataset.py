"""
build_rl_run3_dataset.py
------------------------
Builds data/rl_run3/ — the run-3 RL dataset — per the run-3 spec
(gdpo_v4_next_steps_claude_recommendation.md §5 step 4):

  - p̂ re-estimated ON-POLICY from run-2 generations
    (runs/gdpo-qlora-beta-ablation-probe/generations.jsonl), not the run-1
    pilot probe. Scalar rewards are reconstructed from the dumped per-check
    breakdowns with validate()'s own rule (sentinel-tagged checks excluded,
    WEIGHTS_THERMO renormalized) — the dump stores breakdowns, not scalars.
  - Mid-band selection weighted toward hard: keep targets with band weight
    w = p̂(1−p̂) > 0 at --bar (default 0.9; bar 0.65 is saturated on-policy)
    and resample ∝ w. NOT hardest-only: p̂=0 targets carry no gradient.
  - Gradeability-stability filter: within a target, every retained check's
    gradeability tag must be constant across all parseable samples. Up to
    60% of fractional targets otherwise get graded by different channel
    sets on different samples, which turns within-group reward spread into
    "did the validator manage to grade this one" instead of route quality.
  - CLOSED-BOOK prompts (SYSTEM_MSG + bare Target line, no PD stability
    block): measured equivalent to open-book (|Δp̂| ≤ 0.04, within noise)
    at ~2× cheaper. Matches probe_effective_support.py --closed-book.
  - Fixed probe set of --probe-size held-out targets (from rl_val,
    stratified 6 per stratum), the repeated-measurement instrument runs
    1-2 lacked (0/612 targets ever repeated across steps there).

Outputs (trainer convention <prefix>_{train,val,probe}.jsonl, prefix rl3):
  data/rl_run3/rl3_train.jsonl   — weighted resample, --train-size rows
  data/rl_run3/rl3_val.jsonl     — all held-out val targets, closed-book
  data/rl_run3/rl3_probe.jsonl   — the fixed probe set
  data/rl_run3/manifest.json     — provenance + per-target stats/tag vectors

Usage:
  PYTHONPATH=. uv run python data_curation/build_rl_run3_dataset.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_dataset import chemistry_class  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402
from validator import (  # noqa: E402
    SynthesisValidator,
    VALIDATOR_VERSION,
    WEIGHTS_THERMO,
)

SENTINEL = SynthesisValidator.SENTINEL_TAGS

# The run-3 reward channels — stability is judged on these only; the dropped
# channels no longer enter the reward, so their flap cannot contaminate it.
from core.reward import RUN3_CHECKS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gens", type=Path,
                   default=Path("runs/gdpo-qlora-beta-ablation-probe/generations.jsonl"),
                   help="run-2 generation dump for on-policy p-hat re-estimation")
    p.add_argument("--rl-dir", type=Path, default=Path("data/rl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/rl_run3"))
    p.add_argument("--bar", type=float, default=0.9,
                   help="pass bar for the selection p-hat (0.65 is saturated)")
    p.add_argument("--train-size", type=int, default=1200,
                   help="resampled train rows (2 prompts/step -> size/2 steps)")
    p.add_argument("--probe-size", type=int, default=30)
    p.add_argument("--min-samples", type=int, default=6,
                   help="min parseable run-2 samples for a target to be eligible")
    p.add_argument("--max-flap", type=int, default=0,
                   help="max samples allowed to deviate from a target's modal "
                        "gradeability tag vector (0 = strict, the run-3 spec; "
                        "1 would grow the pool 115 -> 164 at bounded "
                        "grading-path contamination)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def closed_book_prompt(target: str) -> str:
    """Verbatim the probe's closed-book format — no PD context anywhere, not
    even an empty header that would signal 'data missing'."""
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def scalar_reward(bd: dict) -> float | None:
    """Reconstruct validate()'s scalar from a dumped breakdown: sentinel-tagged
    checks excluded, remaining WEIGHTS_THERMO renormalized."""
    active = {}
    for k, w in WEIGHTS_THERMO.items():
        if bd.get(f"{k}_gradeability") in SENTINEL:
            continue
        v = bd.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            active[k] = w
    ws = sum(active.values())
    if ws == 0:
        return None
    return sum(w / ws * bd[k] for k, w in active.items())


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # ---- per-target run-2 sample stats ----------------------------------
    samples: dict[str, list[dict | None]] = defaultdict(list)
    with args.gens.open() as f:
        n_rows = 0
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            samples[r["target"]].append(r.get("breakdown"))
            n_rows += 1
    print(f"loaded {n_rows} generations over {len(samples)} targets from {args.gens}")

    stats = {}
    for target, bds in samples.items():
        parseable = [bd for bd in bds if bd]
        n_parse_fail = len(bds) - len(parseable)
        rewards = [scalar_reward(bd) for bd in parseable]
        rewards = [r for r in rewards if r is not None]
        if len(rewards) < args.min_samples:
            continue
        p65 = sum(r >= 0.65 for r in rewards) / len(rewards)
        p90 = sum(r >= 0.9 for r in rewards) / len(rewards)
        p_bar = p90 if args.bar == 0.9 else p65
        # gradeability tag per retained check per sample (absent tag == the
        # check is always computed, e.g. operation_order)
        tag_vecs = [
            tuple(bd.get(f"{c}_gradeability", "gradeable") for c in RUN3_CHECKS)
            for bd in parseable
        ]
        stable = (len(tag_vecs) - Counter(tag_vecs).most_common(1)[0][1]
                  ) <= args.max_flap
        stats[target] = {
            "n_samples": len(bds),
            "n_parse_fail": n_parse_fail,
            "p_hat_65": round(p65, 4),
            "p_hat_90": round(p90, 4),
            "band_weight": round(p_bar * (1 - p_bar), 4),
            "gradeability_stable": stable,
            "gradeability_tags": dict(zip(RUN3_CHECKS, Counter(tag_vecs).most_common(1)[0][0])),
        }

    n_stable = sum(s["gradeability_stable"] for s in stats.values())
    n_band = sum(s["band_weight"] > 0 for s in stats.values())
    pool = [t for t, s in stats.items()
            if s["gradeability_stable"] and s["band_weight"] > 0]
    print(f"eligible (>= {args.min_samples} parseable): {len(stats)}")
    print(f"  gradeability-stable: {n_stable}   mid-band (w>0 @{args.bar}): {n_band}")
    print(f"  both (train pool): {len(pool)}")
    if not pool:
        sys.exit("FATAL: empty train pool — relax the bar or the stability filter")

    # ---- source rows (stratum labels) ------------------------------------
    def load_rl(path: Path) -> dict[str, dict]:
        rows = {}
        with path.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["target"]] = r
        return rows

    rl_train = load_rl(args.rl_dir / "rl_train.jsonl")
    rl_val = load_rl(args.rl_dir / "rl_val.jsonl")
    missing = [t for t in pool if t not in rl_train]
    if missing:
        print(f"warn: {len(missing)} pool targets absent from rl_train, dropping")
        pool = [t for t in pool if t in rl_train]

    # ---- probe set: stratified held-out val ------------------------------
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for t, r in rl_val.items():
        by_stratum[r.get("stratum", "unknown")].append(t)
    per = max(1, args.probe_size // max(len(by_stratum), 1))
    probe_targets: list[str] = []
    for s in sorted(by_stratum):
        cand = sorted(t for t in by_stratum[s] if t not in rl_train)
        probe_targets.extend(rng.sample(cand, min(per, len(cand))))
    probe_targets = sorted(probe_targets)
    print(f"probe set: {len(probe_targets)} targets "
          f"({dict(Counter(rl_val[t].get('stratum') for t in probe_targets))})")
    assert not set(probe_targets) & set(pool), "probe/train overlap"

    # ---- emit -------------------------------------------------------------
    def record(target: str, src_rows: dict, annotated: bool) -> dict:
        src = src_rows[target]
        rec = {
            "prompt": closed_book_prompt(target),
            "target": target,
            "stratum": src.get("stratum"),
            "chemistry_class": chemistry_class(target),
            "source": "run3_band_from_" + args.gens.parent.name,
            "validator_version": VALIDATOR_VERSION,
        }
        if annotated and target in stats:
            rec.update(stats[target])
        return rec

    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights = [stats[t]["band_weight"] for t in pool]
    train_targets = rng.choices(pool, weights=weights, k=args.train_size)
    with (args.out_dir / "rl3_train.jsonl").open("w") as f:
        for t in train_targets:
            f.write(json.dumps(record(t, rl_train, annotated=True)) + "\n")

    with (args.out_dir / "rl3_val.jsonl").open("w") as f:
        for t in sorted(rl_val):
            f.write(json.dumps(record(t, rl_val, annotated=False)) + "\n")

    with (args.out_dir / "rl3_probe.jsonl").open("w") as f:
        for t in probe_targets:
            f.write(json.dumps(record(t, rl_val, annotated=False)) + "\n")

    manifest = {
        "validator_version": VALIDATOR_VERSION,
        "source_gens": str(args.gens),
        "source_rl_dir": str(args.rl_dir),
        "prompt_format": "SYSTEM_MSG + bare Target (CLOSED-BOOK, no PD stability data)",
        "selection": {
            "bar": args.bar,
            "band_weight": "p_hat(1-p_hat) at bar, on-policy from run-2 generations",
            "gradeability": f"tag vector stable across samples (max_flap={args.max_flap}), checks={list(RUN3_CHECKS)}",
            "min_samples": args.min_samples,
            "pool_size": len(pool),
            "train_size": args.train_size,
            "resampling": "with replacement, seed-fixed, ∝ band_weight",
        },
        "counts": {
            "train_rows": len(train_targets),
            "train_unique_targets": len(set(train_targets)),
            "val": len(rl_val),
            "probe": len(probe_targets),
        },
        "probe_targets": probe_targets,
        "probe_strata": dict(Counter(rl_val[t].get("stratum") for t in probe_targets)),
        "per_target": stats,
        "caveats": [
            "p̂ labels are on-policy for the run-2 policy (gdpo ckpt-300 + β/lr ablation); "
            "they drift as run 3 trains — mid-band is a starting condition, not a fixed property.",
            "Train rows repeat targets by design (weighted resampling); each occurrence "
            "still gets fresh G=8 samples at train time.",
            "Closed-book removes the MP-snapshot prompt pinning: prompts no longer carry "
            "live PD data, so validator/MP updates no longer stale the prompt set.",
        ],
    }
    with (args.out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {args.out_dir}/rl3_train.jsonl  ({len(train_targets)} rows, "
          f"{len(set(train_targets))} unique)")
    print(f"wrote {args.out_dir}/rl3_val.jsonl    ({len(rl_val)} rows)")
    print(f"wrote {args.out_dir}/rl3_probe.jsonl  ({len(probe_targets)} rows)")
    print(f"wrote {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
