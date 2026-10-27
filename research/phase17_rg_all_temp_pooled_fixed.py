#!/usr/bin/env python
"""
research/phase17_rg_all_temp_pooled_fixed.py — fix to results/phase17/
rg_table2_verdict.json's "pooled_over_targets.all_temperatures": it pools
by CONCATENATING each target's 1,000 bootstrap replicate (diff) values
into one 3,000-long array, then takes percentiles of that. This is a
different bug from the n_hi-mixing one phase17_rg_table2_verdict_
stratified.py fixed (that script's own docstring explicitly, and
wrongly, said all-temperatures pooling "has no mixing problem to begin
with" -- it checked only for n_hi mixing, not this).

Concatenating replicates conflates within-target bootstrap uncertainty
with between-target variation in an uninterpretable way. The correct
pooled quantity (regular Claude, 2026-10-27): average the three targets'
diff values WITHIN each bootstrap replicate index (YBCO's SEED is reset
per target in rg_table2_verdict.py's main(), so replicate i for each
target is a deterministic, reproducible draw at that position -- not a
shared joint resample, but a well-defined index alignment), producing
ONE 1,000-long series of pooled-mean diffs, then take percentiles of
THAT series.

FIX: re-run the IDENTICAL deterministic bootstrap (same SEED, same
paired_bootstrap/build_pool_indexed/precompute_win_matrix, imported
unchanged from phase17_rg_table2_verdict.py) and pool this way instead.
Does NOT overwrite rg_table2_verdict.json.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_rg_all_temp_pooled_fixed.py
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import load_comparator  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency  # noqa: E402
from core.reward import load_validator  # noqa: E402
from research.ranker_capacity_rssft import SYNTHESIS  # noqa: E402
from research.distributional.arrows_gate_scoring import COMPARATOR_SCALES_PATH  # noqa: E402
from research.phase17_rg_pool_bootstrap import build_pool_indexed, precompute_win_matrix  # noqa: E402
from research.phase17_rg_table2_verdict import (  # noqa: E402
    paired_bootstrap, REWARDS, N_LO, N_HI_DEFAULT, TARGETS, SEED,
)
from validator import ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rg_all_temp_pooled_fixed.json"


def pct(vals, lo=0.025, hi=0.975):
    vals = sorted(vals)
    return [round(vals[int(lo * len(vals))], 4), round(vals[min(int(hi * len(vals)), len(vals) - 1)], 4)]


def main():
    print("loading validator, comparator, ranker...", flush=True)
    validator = load_validator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                               REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    comparator = load_comparator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                                 REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    scales = json.loads(COMPARATOR_SCALES_PATH.read_text())["scales"]
    freq = build_precursor_frequency(SYNTHESIS)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)

    per_target_pairs = {}
    for name in TARGETS:
        print(f"\n=== {name} ===", flush=True)
        pool, target_formula = build_pool_indexed(name, validator, ranker)
        win = precompute_win_matrix(pool, comparator, scales, target_formula)
        rng = random.Random(SEED)
        all_temp = paired_bootstrap(pool, win, N_LO, N_HI_DEFAULT, rng)
        per_target_pairs[name] = all_temp
        print(f"  n_replicates={ {m: len(all_temp[m]) for m in REWARDS} }", flush=True)

    lengths = {name: {m: len(per_target_pairs[name][m]) for m in REWARDS} for name in TARGETS}
    min_len_by_reward = {m: min(lengths[name][m] for name in TARGETS) for m in REWARDS}

    results = {}
    for m in REWARDS:
        n_use = min_len_by_reward[m]
        pooled_diffs = []
        for i in range(n_use):
            diffs_at_i = [per_target_pairs[name][m][i][1] - per_target_pairs[name][m][i][0] for name in TARGETS]
            pooled_diffs.append(sum(diffs_at_i) / len(diffs_at_i))
        ci = pct(pooled_diffs)
        mean_diff = sum(pooled_diffs) / len(pooled_diffs)
        results[m] = {
            "n_replicates_used": n_use, "lengths_per_target": {name: lengths[name][m] for name in TARGETS},
            "mean_diff": round(mean_diff, 4), "ci95": ci,
            "verdict": "climbs" if ci[0] > 0 else ("drops" if ci[1] < 0 else "no climb"),
        }
        print(f"{m}: mean_diff={mean_diff:.4f} ci95={ci} -> {results[m]['verdict']}", flush=True)

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "R-g", "subproperty": "all-temperatures pooled CI, fixed",
        "fix_note": (
            "Replaces concatenate-then-percentile (rg_table2_verdict.json's "
            "pooled_over_targets.all_temperatures) with average-within-replicate-then-"
            "percentile, per regular Claude's review (2026-10-27). Does not overwrite the "
            "original file."
        ),
        "seed": SEED, "n_hi": N_HI_DEFAULT, "results": results,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
