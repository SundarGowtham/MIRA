#!/usr/bin/env python
"""
research/review_2_c_calibrated_range.py — review check 2 (2026-11-01):
robustness of Block B1's split-trace slope c, restricted to (target,
recipe) rows whose FULL (non-split, all 8 traces) before-model estimate
pi~_before >= 1e-2 -- the range results/phase17/b1_tail_calibration.json
shows is roughly calibrated (observed_over_expected in [0.594, 2.23] at
and above the [1e-2,1e-1) bin, vs up to ~1.7 million x below it). Reading
rule pre-registered in docs/phases/REVIEW_RESULTS.md before this ran.

Reuses, unchanged: research/phase17_b1_split_trace_control.py's
load_per_trace/build_split_data (the split-trace rows), and
research/phase17_b1_signature_regression.py's load_logp/
load_recipe_rewards/pi_tilde_before/bootstrap_step (the full K=8-trace
marginal estimator, used ONLY to compute the filter -- the regression
itself still regresses on the split half pi~_A, exactly as
b1_split_trace_control_corrected.json does; only the ROW SET fed into
that regression changes here).

Filters split_rows to rows whose target+recipe has a FULL-estimator
pi~_before >= 1e-2 (from logp_universe.jsonl via load_logp(), the same
quantity b1_tail_calibration.json bins -- NOT the split-half pi~_A the
regression's regressor is built from).

NO EDIT to any frozen file. NO TRAINING. New output file only, under
results/review/.

Usage (tmux, CPU only, seconds):
  PYTHONHASHSEED=0 PYTHONPATH=. uv run python research/review_2_c_calibrated_range.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase17_b1_signature_regression import (  # noqa: E402
    STEPS, load_logp, load_recipe_rewards, pi_tilde_before, bootstrap_step,
)
from research.phase17_b1_split_trace_control import load_per_trace, build_split_data, SEED  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "review" / "c_calibrated_range.json"
PI_CUTOFF = 1e-2

FT_STEPS = ("S1", "S2", "S3", "S4")
PRACTICE_RULE_STEPS = ("S5", "S7")  # S6 reported but not part of the pass/fail rule


def full_pi_tilde_by_target(before_model: str, logp_after_all: dict) -> dict[str, dict]:
    """{target: {recipe_tuple: pi~_before}}, using the FULL (unsplit, all
    8 traces) marginal estimator for before_model -- logp_after_all
    already holds this for every model (it is the same file
    b1_signature_regression.py uses for 'after' models; before_model's
    own full-trace logp lives in it too)."""
    out = {}
    for (model, target), by_recipe in logp_after_all.items():
        if model == before_model:
            out[target] = pi_tilde_before(by_recipe)
    return out


def main():
    per_trace = load_per_trace()
    logp_after_all = load_logp()
    rewards = load_recipe_rewards()

    results = {}
    for step, (before, after, reward_field, _sens) in STEPS.items():
        split_rows, _placebo_rows = build_split_data(before, after, reward_field, per_trace, logp_after_all, rewards)
        n_before_filter = len(split_rows)

        pi_full = full_pi_tilde_by_target(before, logp_after_all)
        filtered_rows = [
            r for r in split_rows
            if pi_full.get(r["target"], {}).get(r["recipe"], 0.0) >= PI_CUTOFF
        ]
        n_after_filter = len(filtered_rows)
        n_targets_after = len({r["target"] for r in filtered_rows})

        fit = bootstrap_step(filtered_rows, SEED) if n_after_filter >= 10 else None
        c_point = fit["point"]["c"] if (fit and fit["point"]) else None
        c_ci = fit.get("c_ci95", [None, None]) if fit else [None, None]
        b_point = fit["point"]["b"] if (fit and fit["point"]) else None
        b_ci = fit.get("b_ci95", [None, None]) if fit else [None, None]
        # within_target_fe_ols silently drops any target with <2 rows (its
        # demeaned contribution is exactly 0); n_obs below is the ACTUAL count
        # of rows that entered the regression, not n_after_filter.
        n_obs_in_fit = fit["point"]["n_obs"] if (fit and fit["point"]) else 0
        by_t = defaultdict(int)
        for r in filtered_rows:
            by_t[r["target"]] += 1
        n_targets_with_ge2 = sum(1 for v in by_t.values() if v >= 2)

        print(f"{step} ({before}->{after}): n_split_rows={n_before_filter} "
              f"n_calibrated(pi>=1e-2)={n_after_filter} n_targets_any={n_targets_after} "
              f"n_targets_with_>=2obs={n_targets_with_ge2} n_obs_in_fit={n_obs_in_fit} "
              f"c_split={c_point} ci={c_ci}", flush=True)

        results[step] = {
            "before": before, "after": after,
            "n_split_rows_unfiltered": n_before_filter,
            "n_rows_calibrated": n_after_filter,
            "n_targets_calibrated_any": n_targets_after,
            "n_targets_calibrated_with_ge2_obs": n_targets_with_ge2,
            "n_obs_in_fe_fit": n_obs_in_fit,
            "c_split_calibrated": round(c_point, 4) if c_point is not None else None,
            "c_split_calibrated_ci95": c_ci,
            "b_split_calibrated": round(b_point, 4) if b_point is not None else None,
            "b_split_calibrated_ci95": b_ci,
        }

    reading_rule = {"pi_cutoff": PI_CUTOFF, "per_step": {}}
    all_ft_met = True
    for s in FT_STEPS:
        ci = results[s]["c_split_calibrated_ci95"]
        met = ci[0] is not None and ci[1] is not None and ci[1] < 0
        all_ft_met = all_ft_met and met
        reading_rule["per_step"][s] = {"role": "fine-tuning", "c_ci95": ci, "rule": "entirely < 0", "met": met}
    all_practice_met = True
    for s in PRACTICE_RULE_STEPS:
        ci = results[s]["c_split_calibrated_ci95"]
        met = ci[0] is not None and ci[1] is not None and (ci[0] <= 0 <= ci[1])
        all_practice_met = all_practice_met and met
        reading_rule["per_step"][s] = {"role": "practice (in rule)", "c_ci95": ci, "rule": "includes 0", "met": met}
    # S6 reported, not adjudicated
    reading_rule["per_step"]["S6"] = {
        "role": "practice (reported, NOT part of pass/fail rule)",
        "c_ci95": results["S6"]["c_split_calibrated_ci95"],
        "note": "finding 34: already known to carry a small, real, CI-excludes-0 residual; "
                "this rule does not adjudicate S6 either way.",
    }
    reading_rule["contrast_stands"] = bool(all_ft_met and all_practice_met)

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "review", "subproperty": "B1 c_split, calibrated-pi~ range only",
        "method": (
            "Same split-trace regression as b1_split_trace_control_corrected.json (traces 1-4 "
            "regressor, Delta=logp_after-logp(traces 5-8), within-target FE OLS, cluster bootstrap "
            "by target, seed 20261016), restricted to (target,recipe) rows whose FULL (unsplit, "
            "all 8 traces) before-model pi~_before >= 1e-2 -- the probability range "
            "b1_tail_calibration.json shows is roughly calibrated (observed/expected in "
            "[0.594, 2.23] at and above [1e-2,1e-1), vs up to ~1.7e6x below it). The filter uses "
            "the FULL estimator's pi~, not the split-half pi~_A the regression's own regressor is "
            "built from."
        ),
        "pi_cutoff": PI_CUTOFF,
        "results": results,
        "reading_rule": reading_rule,
        "low_power_caveat": (
            "The >=1e-2 filter removes 85-92% of split_rows per step (most recipes live in the "
            "miscalibrated tail by sheer diversity, not because the calibrated range is empty), "
            "and within_target_fe_ols additionally drops any target with <2 surviving rows (its "
            "demeaned contribution is exactly 0). n_obs_in_fe_fit / n_targets_calibrated_with_ge2_obs "
            "above are the ACTUAL regression sample per step, smaller than n_rows_calibrated -- as "
            "low as 36 obs over 12 targets for S1. Read any sign flip in this calibrated-range fit "
            "as suggestive, not as statistically equivalent in power to the full-range result."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\ncontrast_stands={reading_rule['contrast_stands']}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
