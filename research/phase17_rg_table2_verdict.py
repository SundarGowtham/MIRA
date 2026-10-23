#!/usr/bin/env python
"""
research/phase17_rg_table2_verdict.py -- regular Claude's Table 2:
"for R2 (and R4, R5), does the climb from n=1 to n=16 survive (a) the
pool-level cluster bootstrap and (b) the within-temperature control?
Give, per target and pooled: yield at n=1 and n=16, the difference, and
its CI under each. One-line verdict per reward."

(a) ALL TEMPERATURES: reuses results/phase17/rg_pool_bootstrap.json's
already-computed pool_bootstrap_ci_a (same N_BOOT_POOL=1000, N_DRAWS_INNER
=300, seed) -- but that file only stored the SUMMARIZED mean/CI per n,
independently for n=1 and n=16. This script recomputes the SAME
procedure (reusing precompute_win_matrix/pick_best_fast by import, no
edits) but keeps each outer bootstrap replicate's (n=1, n=16) PAIR, so
the difference's CI is a proper paired percentile CI (mean_n16_r -
mean_n1_r per replicate r, not an independent-CI overlap check).

(b) WITHIN-TEMPERATURE: same pairing, but the pool is first restricted
to one measured temperature stratum (research/phase17_rg_pool_bootstrap.py
's within_temperature_curve only reported a POINT mean, no CI -- this
adds the same pool-cluster-bootstrap CI to it). Pool sizes per stratum
cap how large n can go (YBCO's four largest strata support n=16; every
other stratum tops out at n=8 or less) -- reported explicitly per
stratum, and pooled only across strata that support the SAME n_hi used
for that target's row (n_hi chosen as the largest value every one of
a target's qualifying strata supports, reported per target).

POOLING ACROSS TARGETS: equal-weight concatenation of each target's
(or stratum's) 1,000 paired-replicate diffs into one combined array,
then a straight percentile CI on the combined array -- the same
"pooled" convention already used elsewhere in Block R (e.g. the
carbonate-tie-stratum pooling in phase17_rf_stratum_reconciliation.py).

Does NOT overwrite rg_pool_bootstrap.json -- new file.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_rg_table2_verdict.py
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
from research.phase17_rg_pool_bootstrap import (  # noqa: E402
    build_pool_indexed, precompute_win_matrix, pick_best_fast, METHODS,
    N_BOOT_POOL, N_DRAWS_INNER, SEED,
)
from validator import ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rg_table2_verdict.json"
TARGETS = ["YBCO", "LTOPO", "NTMO"]
REWARDS = ["R2_checker", "R4_ranker", "R5_comparator"]
N_LO, N_HI_DEFAULT = 1, 16


def paired_bootstrap(pool: list[dict], win: dict, n_lo: int, n_hi: int, rng: random.Random) -> dict:
    """Same mechanics as pool_bootstrap_curve, but keeps each outer
    replicate's (n_lo, n_hi) pair per reward for a paired-diff CI."""
    by_set = defaultdict(list)
    for d in pool:
        by_set[d["precursor_set"]].append(d)
    set_labels = sorted(by_set.keys())
    n_sets = len(set_labels)

    pairs = {m: [] for m in REWARDS}
    for _ in range(N_BOOT_POOL):
        drawn_labels = [set_labels[rng.randrange(n_sets)] for _ in range(n_sets)]
        resampled_pool = [d for label in drawn_labels for d in by_set[label]]
        if len(resampled_pool) < max(2, n_hi):
            continue
        means = {}
        for n in (n_lo, n_hi):
            vals = {m: [] for m in REWARDS}
            for _ in range(N_DRAWS_INNER):
                draw = rng.sample(resampled_pool, n)
                for m in REWARDS:
                    pick = pick_best_fast(draw, m, win, rng)
                    vals[m].append(pick["wt_pct"])
            means[n] = {m: sum(vals[m]) / len(vals[m]) for m in REWARDS}
        for m in REWARDS:
            pairs[m].append((means[n_lo][m], means[n_hi][m]))
    return pairs


def summarize_pairs(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"n_replicates": 0}
    n1 = sorted(p[0] for p in pairs)
    n16 = sorted(p[1] for p in pairs)
    diffs = sorted(b - a for a, b in pairs)
    n = len(pairs)

    def pct(vals, lo=0.025, hi=0.975):
        return [round(vals[int(lo * len(vals))], 4), round(vals[min(int(hi * len(vals)), len(vals) - 1)], 4)]

    return {
        "n_replicates": n,
        "yield_n1": {"mean": round(sum(n1) / n, 4), "ci95": pct(n1)},
        "yield_n16": {"mean": round(sum(n16) / n, 4), "ci95": pct(n16)},
        "diff": {"mean": round(sum(diffs) / n, 4), "ci95": pct(diffs)},
        "verdict": "climbs" if pct(diffs)[0] > 0 else ("drops" if pct(diffs)[1] < 0 else "no climb"),
    }


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

    report = {"prereg_commit": PREREG_COMMIT, "block": "R-g", "subproperty": "Table 2 verdict",
               "n_lo": N_LO, "n_hi_default": N_HI_DEFAULT, "per_target": {}}

    all_temp_pairs = {m: [] for m in REWARDS}
    within_temp_pairs = {m: [] for m in REWARDS}

    for name in TARGETS:
        print(f"\n=== {name} ===", flush=True)
        pool, target_formula = build_pool_indexed(name, validator, ranker)
        win = precompute_win_matrix(pool, comparator, scales, target_formula)
        rng = random.Random(SEED)

        all_temp = paired_bootstrap(pool, win, N_LO, N_HI_DEFAULT, rng)
        all_temp_summary = {m: summarize_pairs(all_temp[m]) for m in REWARDS}
        for m in REWARDS:
            all_temp_pairs[m].extend(all_temp[m])
        print("  all-temperatures:", json.dumps(all_temp_summary, indent=1), flush=True)

        by_temp = defaultdict(list)
        for d in pool:
            by_temp[d["temp"]].append(d)
        strata_able_16 = [t for t, sub in by_temp.items() if len(sub) >= N_HI_DEFAULT]
        n_hi_within = N_HI_DEFAULT if strata_able_16 else max(
            (n for n in (8, 4, 2) if any(len(sub) >= n for sub in by_temp.values())), default=None)
        qualifying = [t for t, sub in by_temp.items() if len(sub) >= (n_hi_within or 10**9)]

        within_detail = {}
        target_within_pairs = {m: [] for m in REWARDS}
        for temp in qualifying:
            sub = by_temp[temp]
            sub_reindexed = [{**d, "idx": i} for i, d in enumerate(sub)]
            win_sub = precompute_win_matrix(sub_reindexed, comparator, scales, target_formula)
            pairs = paired_bootstrap(sub_reindexed, win_sub, N_LO, n_hi_within, rng)
            within_detail[temp] = {"pool_size": len(sub_reindexed),
                                   **{m: summarize_pairs(pairs[m]) for m in REWARDS}}
            for m in REWARDS:
                target_within_pairs[m].extend(pairs[m])
                within_temp_pairs[m].extend(pairs[m])
        print(f"  within-temperature (n_hi={n_hi_within}, strata={qualifying}):",
              json.dumps({m: summarize_pairs(target_within_pairs[m]) for m in REWARDS}, indent=1), flush=True)

        report["per_target"][name] = {
            "pool_size": len(pool),
            "all_temperatures": all_temp_summary,
            "within_temperature_n_hi": n_hi_within,
            "within_temperature_qualifying_strata": qualifying,
            "within_temperature_per_stratum": within_detail,
            "within_temperature_pooled_over_strata": {m: summarize_pairs(target_within_pairs[m]) for m in REWARDS},
        }

    report["pooled_over_targets"] = {
        "all_temperatures": {m: summarize_pairs(all_temp_pairs[m]) for m in REWARDS},
        "within_temperature": {m: summarize_pairs(within_temp_pairs[m]) for m in REWARDS},
        "note": "within_temperature pooling mixes targets whose n_hi differs (YBCO reaches 16 on "
                "4/5 strata; LTOPO/NTMO cap at 8) -- an equal-weight concatenation of whatever n_hi "
                "each qualifying stratum supports, not a single matched n. Per-target rows above give "
                "the matched-n detail.",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print("\n=== POOLED OVER TARGETS ===", flush=True)
    print(json.dumps(report["pooled_over_targets"], indent=1), flush=True)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
