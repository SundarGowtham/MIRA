#!/usr/bin/env python
"""
research/phase17_rh_recount_no_tail.py — Phase 17 Block R, R-h recount
WITHOUT the estimator tail (regular Claude, item 2, 2026-10-28): R-h's
original "unreachable share" used Block A's K=8-trace pi~ estimator in
the denominator -- exactly the quantity the tail-calibration check
(b1_tail_calibration.json) showed is biased low by up to ~1.7M x in the
lowest decade. This recount drops the estimator entirely.

Method: for R1 (init=corpus_sft), R2 (init=rs_sft), R4 (init=rs_sft),
per target, take the reward's top-decile recipes (same selection as
phase17_rh_reachability.py's reward_audit_for: >=10 reward-covered
recipes required, decile cutoff at the 90th percentile of mean reward,
ties at the cutoff included). For each such recipe, count its RAW
occurrence in the init model's own samples on disk for that target,
reusing phase17_gate_g1.py's held_out_frequencies EXACTLY UNCHANGED
(every raw completion for that model/target whose precset is in U(x),
excluding only the up-to-8 rows consumed as Block A's own tau_k traces
-- not an estimate, a literal count. For base/rs_sft this pool already
spans both the 32-sample resampling_rate draw and the 200-sample
reference pool, same as b1_tail_calibration.json's note).

Report: share of top-decile recipes with 0 observed occurrences, and a
Clopper-Pearson (alpha=0.05) upper bound on the true occurrence rate of
those zero-count recipes -- both per-recipe (using that target's own
n_held_out as the trial count) and pooled (summing n_held_out across
every zero-count recipe, treated as one binomial test of "a top-decile,
zero-observed recipe occurs at rate > bound" -- a supplementary number,
not a replacement for the per-recipe bounds).

The 4 Phase-12-gained targets' rule-picked set under RS-SFT is NOT an
arbitrary top-decile recipe selection -- it is copied directly from
results/phase17/b1_tail_calibration.json's l6_gains_detail (counts
across all ~232 on-disk RS-SFT samples/target, both sample_groups),
which already answers this exact question for exactly those 4 targets.

NO EDIT to any frozen file.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_rh_recount_no_tail.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase17_rh_reachability import (  # noqa: E402
    REWARDS, load_logp, load_reward_table, L6_GAINS_TARGETS,
)
from research.phase17_gate_g1 import held_out_frequencies, load_universe_node_keys  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
INVENTORY_PATH = REPO_ROOT / "results" / "phase17" / "inventory_v2.json"
TAIL_CAL_PATH = REPO_ROOT / "results" / "phase17" / "b1_tail_calibration.json"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rh_recount_no_tail.json"
ALPHA = 0.05


def cp_upper_bound_zero_successes(n: int, alpha: float = ALPHA) -> float | None:
    """Exact one-sided Clopper-Pearson upper bound when X=0 successes in
    n trials: 1 - alpha^(1/n)."""
    if n <= 0:
        return None
    return 1.0 - alpha ** (1.0 / n)


def main():
    inventory = json.loads(INVENTORY_PATH.read_text())
    astral_doc = json.loads((REPO_ROOT / "results" / "astral_validation_set.json").read_text())
    astral_targets = {t["target"] for t in astral_doc["targets"]}
    heldout_targets = set(json.loads((REPO_ROOT / "data" / "novelty" / "heldout_targets.json").read_text()))
    node_keys_by_target = load_universe_node_keys()

    print("computing held-out (raw, non-estimator) frequencies...", flush=True)
    held_out = held_out_frequencies(inventory, astral_targets, heldout_targets, node_keys_by_target)

    print("loading logp_universe.jsonl (only to enumerate each init model's U(x) membership)...", flush=True)
    logp = load_logp()

    by_reward = {}
    for reward_id, cfg in REWARDS.items():
        field, init_model = cfg["field"], cfg["init_model"]
        reward_table = load_reward_table(field)
        print(f"\n{reward_id}: init_model={init_model} field={field}", flush=True)

        per_target = {}
        all_cells = []  # (target, recipe, observed, n_held_out)
        for (model, target), by_recipe in logp.items():
            if model != init_model:
                continue
            covered = {y: reward_table[(target, y)] for y in by_recipe if (target, y) in reward_table}
            if len(covered) < 10:
                continue
            r_mean = {y: sum(vs) / len(vs) for y, vs in covered.items()}
            sorted_r = sorted(r_mean.values())
            decile_cut = sorted_r[int(0.9 * len(sorted_r))]
            top_decile = [y for y, r in r_mean.items() if r >= decile_cut]

            ho_counter = held_out.get((model, target), Counter())
            n_held_out = sum(ho_counter.values())
            recipe_rows = []
            for y in top_decile:
                obs = ho_counter.get(y, 0)
                recipe_rows.append({"recipe": list(y), "observed": obs, "n_held_out": n_held_out})
                all_cells.append((target, y, obs, n_held_out))
            per_target[target] = {
                "n_top_decile": len(top_decile), "decile_cutoff_reward": round(decile_cut, 4),
                "n_held_out_samples": n_held_out,
                "n_zero_observed": sum(1 for r in recipe_rows if r["observed"] == 0),
                "recipes": recipe_rows,
            }

        zero_cells = [(t, y, n) for (t, y, obs, n) in all_cells if obs == 0]
        n_total = len(all_cells)
        n_zero = len(zero_cells)
        share_zero = round(n_zero / n_total, 4) if n_total else None

        per_recipe_bounds = [cp_upper_bound_zero_successes(n) for (_t, _y, n) in zero_cells if n > 0]
        per_recipe_bounds = [b for b in per_recipe_bounds if b is not None]
        pooled_n = sum(n for (_t, _y, n) in zero_cells)
        pooled_bound = cp_upper_bound_zero_successes(pooled_n)

        print(f"  n_top_decile_total={n_total} n_zero_observed={n_zero} share_zero={share_zero} "
              f"pooled_n_held_out={pooled_n} pooled_cp_upper_bound={pooled_bound}", flush=True)

        by_reward[reward_id] = {
            "init_model": init_model, "reward_field": field,
            "n_targets_with_decile": len(per_target),
            "n_top_decile_recipes_total": n_total,
            "n_zero_observed": n_zero,
            "share_zero_observed": share_zero,
            "per_recipe_cp_upper_bound_alpha05": {
                "n_recipes_with_bound": len(per_recipe_bounds),
                "min": round(min(per_recipe_bounds), 6) if per_recipe_bounds else None,
                "median": round(sorted(per_recipe_bounds)[len(per_recipe_bounds) // 2], 6) if per_recipe_bounds else None,
                "max": round(max(per_recipe_bounds), 6) if per_recipe_bounds else None,
            },
            "pooled_cp_upper_bound_alpha05": {
                "pooled_n_held_out_trials": pooled_n,
                "upper_bound": round(pooled_bound, 6) if pooled_bound is not None else None,
                "note": ("Treats every zero-observed top-decile recipe as an independent binomial test "
                        "of its own occurrence rate, pools their trial counts (sum of each recipe's "
                        "target's n_held_out) into one Clopper-Pearson bound. A supplementary summary, "
                        "not a substitute for the per-recipe bounds above (targets differ in n_held_out)."),
            },
            "per_target": per_target,
        }

    l6_source = json.loads(TAIL_CAL_PATH.read_text())["l6_gains_detail"]
    l6_copied = {t: l6_source[t] for t in L6_GAINS_TARGETS if t in l6_source}

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "R", "subproperty": "R-h recount without the estimator tail",
        "method_note": (
            "Replaces Block A's pi~ marginal estimator with a literal on-disk occurrence count "
            "(held_out_frequencies, reused unchanged from phase17_gate_g1.py) in the denominator "
            "of R-h's unreachable-share check, per regular Claude's review (2026-10-27/28): the "
            "tail-calibration check (b1_tail_calibration.json) showed that estimator is biased low "
            "by up to ~1.7M x in the lowest decade, exactly where the original R-h result lived."
        ),
        "alpha": ALPHA,
        "by_reward": by_reward,
        "l6_gains_on_disk_counts": {
            "note": "Copied from results/phase17/b1_tail_calibration.json's l6_gains_detail -- the "
                    "rule-picked predicted set's exact occurrence count across every RS-SFT sample on "
                    "disk (~232/target, both sample_groups), not an arbitrary top-decile recipe.",
            "targets": l6_copied,
        },
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
