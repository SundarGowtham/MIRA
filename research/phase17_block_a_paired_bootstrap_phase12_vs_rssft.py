#!/usr/bin/env python
"""
research/phase17_block_a_paired_bootstrap_phase12_vs_rssft.py — Phase 17
Block A, item 3 (regular Claude, 2026-10-28): paired bootstrap of
Phase12@300's hits minus RS-SFT's hits, SAME resamples.

phase17_block_a_hit_bootstrap.py already bootstraps each model
independently (fresh `random.Random(SEED)` per model); because every
model loop makes the identical sequence of `rng.randrange(32)` calls for
the same 35 targets in the same sorted order, replicate i happens to
already use the same resampled POSITIONS across models -- but that file
never stores or diffs the per-model hit sequences, only their final
summary stats. This script makes the pairing explicit and auditable:
for each of 10,000 replicates and each of the 35 targets, draw ONE set
of 32 resampled positions (indices into that target's own 32-sample
array) and apply it to BOTH Phase12@300's and RS-SFT's sample arrays for
that target (both have exactly 32 samples/target, so positions are
interchangeable) -- then diff = hits_phase12(replicate) -
hits_rssft(replicate). Report the resulting diff distribution's median
and 95% interval.

NO EDIT to any frozen file.

Usage (tmux, CPU only, seconds):
  PYTHONPATH=. uv run python research/phase17_block_a_paired_bootstrap_phase12_vs_rssft.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
R = REPO_ROOT / "results"
OUT_PATH = R / "phase17" / "block_a_paired_bootstrap_phase12_vs_rssft.json"
SEED = 20261016
N_BOOT = 10000

PHASE12_FILE = "astral_gen_n32_gdpo_phase12_beta0.json"
RSSFT_FILE = "astral_gen_n32_rs_sft.json"


def load(rel):
    return json.loads((R / rel).read_text())


def main():
    phase12 = load(PHASE12_FILE)
    rssft = load(RSSFT_FILE)
    phase12_by_target = {r["target"]: [s.get("match") == "PREDICTED" for s in r["samples"]] for r in phase12["results"]}
    rssft_by_target = {r["target"]: [s.get("match") == "PREDICTED" for s in r["samples"]] for r in rssft["results"]}

    targets = sorted(set(phase12_by_target) & set(rssft_by_target))
    skipped = sorted(set(phase12_by_target) ^ set(rssft_by_target))
    n_mismatch = sum(1 for t in targets if len(phase12_by_target[t]) != len(rssft_by_target[t]))
    print(f"targets in both files: {len(targets)}; symmetric-diff (skipped): {skipped}; "
          f"targets with mismatched sample counts: {n_mismatch}", flush=True)

    orig_phase12_hits = sum(1 for t in targets if any(phase12_by_target[t]))
    orig_rssft_hits = sum(1 for t in targets if any(rssft_by_target[t]))
    print(f"original hits: Phase12@300={orig_phase12_hits}/{len(targets)} RS-SFT={orig_rssft_hits}/{len(targets)} "
          f"diff={orig_phase12_hits - orig_rssft_hits}", flush=True)

    rng = random.Random(SEED)
    diffs = []
    for _ in range(N_BOOT):
        h_phase12, h_rssft = 0, 0
        for t in targets:
            n = min(len(phase12_by_target[t]), len(rssft_by_target[t]))
            idx = [rng.randrange(n) for _ in range(n)]
            if any(phase12_by_target[t][i] for i in idx):
                h_phase12 += 1
            if any(rssft_by_target[t][i] for i in idx):
                h_rssft += 1
        diffs.append(h_phase12 - h_rssft)

    diffs.sort()
    median = diffs[len(diffs) // 2]
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    print(f"paired bootstrap diff (Phase12@300 - RS-SFT): median={median} 95%=[{lo},{hi}]", flush=True)

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "A", "subproperty": "paired bootstrap, Phase12@300 vs RS-SFT hits",
        "seed": SEED, "n_boot": N_BOOT, "n_targets": len(targets), "skipped_targets": skipped,
        "orig_hits": {"Phase12@300": orig_phase12_hits, "RS-SFT": orig_rssft_hits,
                     "orig_diff": orig_phase12_hits - orig_rssft_hits},
        "paired_diff_hits_bootstrap": {"median": median, "ci95": [lo, hi]},
        "method_note": (
            "Paired by construction: each of the 10,000 replicates draws ONE set of resampled "
            "positions per target and applies it to BOTH models' 32-sample arrays for that "
            "target (both have exactly 32 samples/target), so within-replicate noise common to "
            "both models (e.g. a target where resampling happens to drop its only hit for one "
            "model) cancels in the diff rather than inflating the CI."
        ),
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
