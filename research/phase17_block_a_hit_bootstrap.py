#!/usr/bin/env python
"""
research/phase17_block_a_hit_bootstrap.py — Phase 17 Block A item 5
(regular Claude, 2026-10-27): the "replication" wasn't one. scripts/
astral_model_generations.py defaults --seed to 42, and the regen job
(scripts/run_phase17_block_a_regen.sh) never overrides it -- confirmed
directly by diffing the files: 1,120/1,120 samples byte-identical
between results/astral_gen_n32_gdpo_phase12_beta0.json and results/
phase17/block_a_astral_gen_n32_phase12_beta0_ckpt300_replication.json.
Same seed, same model, same sampling config -> deterministic re-run,
not an independent draw. research/phase17_block_a_replication_hits.py's
"0 discordant targets, p=1.0" McNemar result was trivially guaranteed
by this, not evidence of anything.

The REAL question those files can answer: how much of the headline hit
count is just sampling variance from a single n=32 draw per target?
Per model, per bootstrap replicate (10,000, seed 20261016): resample
each target's own 32 samples WITH REPLACEMENT, recompute hit() (exact
match only, same scripts/build_claims_ledger.py logic reimplemented in
phase17_block_a_replication_hits.py) per target, sum over the 35
targets. Report the resulting hits/35 distribution's median, 95%
interval, and P(hits>=13) -- one below the pre-registered ">12/35"
amplification bar -- for Phase12@300, E1@300, GDPO-300, Corpus SFT, and
RS-SFT (10/35, the init being amplified against).

NO EDIT to astral_model_generations.py or build_claims_ledger.py.

Usage (tmux, CPU only, seconds):
  PYTHONPATH=. uv run python research/phase17_block_a_hit_bootstrap.py
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
OUT_PATH = R / "phase17" / "block_a_hit_bootstrap.json"
SEED = 20261016
N_BOOT = 10000

MODELS = [
    {"name": "Phase12@300", "file": "astral_gen_n32_gdpo_phase12_beta0.json", "orig_hits": 14},
    {"name": "E1@300", "file": "astral_gen_n32_e1.json", "orig_hits": 14},
    {"name": "GDPO-300", "file": "astral_gen_n32_gdpo300.json", "orig_hits": 3},
    {"name": "Corpus SFT", "file": "astral_gen_n32_sft.json", "orig_hits": 0},
    {"name": "RS-SFT", "file": "astral_gen_n32_rs_sft.json", "orig_hits": 10},
]


def load(rel):
    return json.loads((R / rel).read_text())


def main():
    report = {"prereg_commit": PREREG_COMMIT, "block": "A", "subproperty": "hit-count bootstrap (sampling-noise floor)",
               "seed": SEED, "n_boot": N_BOOT,
               "deterministic_rerun_note": (
                   "Confirmed directly: astral_model_generations.py defaults --seed=42, "
                   "scripts/run_phase17_block_a_regen.sh never overrides it, and 1,120/1,120 "
                   "samples are byte-identical between the original Phase12@300 ASTRAL file and "
                   "its 'replication'. The *_replication.json files are deterministic re-runs, "
                   "not independent replications -- relabelled in block_a_manifest.json "
                   "accordingly. This bootstrap is the real check: how much of the headline hit "
                   "count is sampling variance from one n=32 draw per target, using ONLY the "
                   "original headline file's own 32 samples/target, resampled with replacement."
               ), "models": {}}

    for spec in MODELS:
        data = load(spec["file"])
        by_target = {r["target"]: [s.get("match") for s in r["samples"]] for r in data["results"]}
        targets = sorted(by_target.keys())
        n_targets = len(targets)
        rng = random.Random(SEED)
        hits_dist = []
        for _ in range(N_BOOT):
            total_hits = 0
            for t in targets:
                samples = by_target[t]
                n = len(samples)
                resampled = [samples[rng.randrange(n)] for _ in range(n)]
                if "PREDICTED" in resampled:
                    total_hits += 1
            hits_dist.append(total_hits)
        hits_dist.sort()
        median = hits_dist[len(hits_dist) // 2]
        lo = hits_dist[int(0.025 * len(hits_dist))]
        hi = hits_dist[min(int(0.975 * len(hits_dist)), len(hits_dist) - 1)]
        p_ge_13 = sum(1 for h in hits_dist if h >= 13) / len(hits_dist)
        report["models"][spec["name"]] = {
            "n_targets": n_targets, "orig_hits_stated": spec["orig_hits"],
            "bootstrap_median_hits": median, "bootstrap_95pct_interval": [lo, hi],
            "p_hits_ge_13": round(p_ge_13, 4),
        }
        print(f"{spec['name']}: orig={spec['orig_hits']}/{n_targets} "
              f"bootstrap median={median} 95%=[{lo},{hi}] P(hits>=13)={p_ge_13:.4f}", flush=True)

    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
