#!/usr/bin/env python
"""
research/phase17_rg_table2_verdict_stratified.py — fix to
results/phase17/rg_table2_verdict.json's own flagged problem: its
"pooled_over_targets.within_temperature" concatenated every qualifying
target's paired-bootstrap replicates into ONE array regardless of n_hi,
even though YBCO's within_temperature_n_hi=16 (its 4 largest strata) while
LTOPO's and NTMO's is 8 (they never reach a single-temperature-stratum
pool of 16) -- the original script's own note said as much ("mixes
targets whose n_hi differs ... not a single matched n") but did not act
on it. Mixing n=8-vs-16 comparisons into one pooled diff array compares
apples to oranges: a bigger n_hi mechanically has less best-of-n draw
noise, so the mixed pool's CI width is not interpretable as "the" CI for
either n.

FIX: re-run the IDENTICAL deterministic bootstrap (same SEED, same
paired_bootstrap/build_pool_indexed/precompute_win_matrix, imported
unchanged from phase17_rg_table2_verdict.py -- reproduces bit-identical
per-target replicate pairs) but pool only across targets that share the
SAME n_hi_within, reporting one CI per n_hi bucket instead of one mega-
pooled CI. (All-temperatures pooling is NOT touched -- N_HI_DEFAULT=16
is used uniformly there for all three targets already, so it has no
mixing problem to begin with; included here unchanged for a single
read-point, explicitly labeled as "no fix needed".)

Does NOT overwrite rg_table2_verdict.json -- new file, and the original
per-target numbers (which were always correct; only the TARGET-POOLED
within-temperature summary mixed n) are unchanged.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_rg_table2_verdict_stratified.py
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from collections import defaultdict
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
    paired_bootstrap, summarize_pairs, REWARDS, N_LO, N_HI_DEFAULT, TARGETS, SEED,
)
from validator import ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rg_table2_verdict_stratified.json"


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

    report = {"prereg_commit": PREREG_COMMIT, "block": "R-g",
               "subproperty": "Table 2 verdict, within-temperature pooling fixed to stratify by n_hi",
               "fix_note": "rg_table2_verdict.json's pooled_over_targets.within_temperature concatenated "
                           "YBCO's n_hi=16 replicates with LTOPO/NTMO's n_hi=8 replicates into one array. "
                           "This file reruns the identical deterministic bootstrap (same seed) and pools "
                           "only within matching n_hi -- one CI per n_hi bucket. all_temperatures is NOT "
                           "re-fixed: it already uses n_hi=16 uniformly for all three targets, so the "
                           "original mega-pooled CI there is already a single matched-n CI.",
               "by_n_hi": {}}

    all_temp_pairs = {m: [] for m in REWARDS}  # unchanged, no fix needed, kept for one read-point
    within_by_n_hi = defaultdict(lambda: {m: [] for m in REWARDS})
    per_target_n_hi = {}

    for name in TARGETS:
        print(f"\n=== {name} ===", flush=True)
        pool, target_formula = build_pool_indexed(name, validator, ranker)
        win = precompute_win_matrix(pool, comparator, scales, target_formula)
        rng = random.Random(SEED)

        all_temp = paired_bootstrap(pool, win, N_LO, N_HI_DEFAULT, rng)
        for m in REWARDS:
            all_temp_pairs[m].extend(all_temp[m])

        by_temp = defaultdict(list)
        for d in pool:
            by_temp[d["temp"]].append(d)
        strata_able_16 = [t for t, sub in by_temp.items() if len(sub) >= N_HI_DEFAULT]
        n_hi_within = N_HI_DEFAULT if strata_able_16 else max(
            (n for n in (8, 4, 2) if any(len(sub) >= n for sub in by_temp.values())), default=None)
        qualifying = [t for t, sub in by_temp.items() if len(sub) >= (n_hi_within or 10**9)]
        per_target_n_hi[name] = n_hi_within
        print(f"  n_hi_within={n_hi_within}, qualifying strata={qualifying}", flush=True)

        for temp in qualifying:
            sub = by_temp[temp]
            sub_reindexed = [{**d, "idx": i} for i, d in enumerate(sub)]
            win_sub = precompute_win_matrix(sub_reindexed, comparator, scales, target_formula)
            pairs = paired_bootstrap(sub_reindexed, win_sub, N_LO, n_hi_within, rng)
            for m in REWARDS:
                within_by_n_hi[n_hi_within][m].extend(pairs[m])

    report["per_target_n_hi_within"] = per_target_n_hi
    report["all_temperatures_pooled_unchanged"] = {m: summarize_pairs(all_temp_pairs[m]) for m in REWARDS}
    for n_hi, pairs_by_reward in sorted(within_by_n_hi.items()):
        targets_in_bucket = [t for t, n in per_target_n_hi.items() if n == n_hi]
        report["by_n_hi"][str(n_hi)] = {
            "targets_pooled": targets_in_bucket,
            **{m: summarize_pairs(pairs_by_reward[m]) for m in REWARDS},
        }
        print(f"\n=== within-temperature, n_hi={n_hi} (targets: {targets_in_bucket}) ===", flush=True)
        print(json.dumps({m: summarize_pairs(pairs_by_reward[m]) for m in REWARDS}, indent=1), flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
