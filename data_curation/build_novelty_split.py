#!/usr/bin/env python
"""
data_curation/build_novelty_split.py — Phase 16 §6.1
(misc/PHASE16_INSTRUCTIONS.md). Reproduces the RS-SFT target pool exactly
(same pool `build_rs_sft_dataset.py` draws from: distinct targets in
data/rl/rl_train.jsonl, sorted then shuffled with seed 42, first 400),
removes every ASTRAL target formula (all 35, unconditionally -- ASTRAL
stays eval-only for every Phase 16 arm, independent of the separate
overlap-disclosure analysis in results/phase16/astral_overlap.json),
then holds out a random 10% (seed 42) of the remainder as the novelty
evaluation split.

No GPU. Deterministic -- cross-checked against data/rs_sft/rs_sft_train.jsonl's
actual survivor targets (every survivor must be in the reproduced 400-pool).

Outputs:
  data/novelty/heldout_targets.json   -- the 10% held-out split
  data/novelty/training_targets.json  -- the remaining 90%, for the
                                          sampling/E3/E3c pipeline

Usage:
  uv run python data_curation/build_novelty_split.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RL_TRAIN_PATH = REPO_ROOT / "data" / "rl" / "rl_train.jsonl"
ASTRAL_PATH = REPO_ROOT / "results" / "astral_validation_set.json"
RS_SFT_TRAIN_PATH = REPO_ROOT / "data" / "rs_sft" / "rs_sft_train.jsonl"
OUT_DIR = REPO_ROOT / "data" / "novelty"

RS_SFT_POOL_SIZE = 400  # build_rs_sft_dataset.py's --targets default
SEED = 42
HELDOUT_FRACTION = 0.10


def reproduce_rs_sft_pool() -> list[str]:
    """Exact replica of build_rs_sft_dataset.py's target-selection logic."""
    pool = []
    with RL_TRAIN_PATH.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pool.append(r["target"])
    seen = sorted(set(pool))
    rng = random.Random(SEED)
    rng.shuffle(seen)
    return seen[:RS_SFT_POOL_SIZE]


def cross_check_against_rs_sft_survivors(pool: set[str]) -> None:
    if not RS_SFT_TRAIN_PATH.exists():
        print(f"  (skip cross-check: {RS_SFT_TRAIN_PATH} not found)")
        return
    survivors = set()
    with RS_SFT_TRAIN_PATH.open() as f:
        for line in f:
            if line.strip():
                survivors.add(json.loads(line)["target"])
    missing = survivors - pool
    if missing:
        raise RuntimeError(
            f"Reproduced RS-SFT pool does not contain {len(missing)} of "
            f"{len(survivors)} actual rs_sft_train.jsonl survivor targets "
            f"({sorted(missing)[:5]}...) -- the pool-reconstruction logic "
            f"has drifted from build_rs_sft_dataset.py's actual behaviour. "
            f"Stop and report rather than silently using a wrong pool.")
    print(f"  cross-check OK: all {len(survivors)} actual RS-SFT survivor "
          f"targets are in the reproduced {len(pool)}-target pool")


def main():
    pool_list = reproduce_rs_sft_pool()
    pool = set(pool_list)
    print(f"reproduced RS-SFT target pool: {len(pool)} targets")
    cross_check_against_rs_sft_survivors(pool)

    astral_targets = {t["target"] for t in json.loads(ASTRAL_PATH.read_text())["targets"]}
    overlap = pool & astral_targets
    remaining = sorted(pool - astral_targets)
    print(f"removed {len(overlap)} ASTRAL targets from the pool: {sorted(overlap)}")
    print(f"remaining pool: {len(remaining)} targets")

    rng = random.Random(SEED)
    shuffled = remaining[:]
    rng.shuffle(shuffled)
    n_heldout = round(len(shuffled) * HELDOUT_FRACTION)
    heldout = sorted(shuffled[:n_heldout])
    training = sorted(shuffled[n_heldout:])
    print(f"held out {len(heldout)} targets (10%), {len(training)} remain for E3/E3c sampling")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "heldout_targets.json").write_text(json.dumps(heldout, indent=1))
    (OUT_DIR / "training_targets.json").write_text(json.dumps(training, indent=1))
    print(f"-> {OUT_DIR / 'heldout_targets.json'}")
    print(f"-> {OUT_DIR / 'training_targets.json'}")


if __name__ == "__main__":
    main()
