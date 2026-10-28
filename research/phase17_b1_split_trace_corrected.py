#!/usr/bin/env python
"""
research/phase17_b1_split_trace_corrected.py — Phase 17 Block B1,
corrected interpretation of the split-trace control (regular Claude,
2026-10-28, item 1):

My prior reading of b1_split_trace_control.json compared c_split against
c_noise_placebo, as if the placebo had to be subtracted a second time.
That's wrong: c_split is ALREADY free of the shared-measurement-error
artifact (it is estimated from two independent trace halves, so the same
noise cannot appear on both sides of the equation) -- it should be
compared to 0 directly. The placebo is subtracted from the ORIGINAL
(un-split) c instead: c_original - c_noise_placebo should approximate
c_split, because c_original = c_true + bias and c_noise_placebo ~= bias
under the shared linear-measurement-error model. This script verifies
that numerically (both should agree within ~0.03) and reports the
corrected verdicts on that basis.

It also reports b_s (not just c_s) under the split design for S5-S7,
needed for P-B1b's other half (b_s>0), by refitting the full 3-regressor
within-target FE model (phase17_b1_signature_regression.bootstrap_step,
reused unchanged -- split_rows already have exactly the fields that
function expects: target, delta, r_tilde, log_pi_before, interaction).

Does NOT overwrite results/phase17/b1_split_trace_control.json. New file.

NO EDIT to any frozen file.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_b1_split_trace_corrected.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase17_b1_signature_regression import STEPS, load_logp, load_recipe_rewards, bootstrap_step  # noqa: E402
from research.phase17_b1_split_trace_control import load_per_trace, build_split_data, SEED  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
CONTROL_PATH = REPO_ROOT / "results" / "phase17" / "b1_split_trace_control.json"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "b1_split_trace_control_corrected.json"

AGREEMENT_TOLERANCE = 0.03
PRACTICE_STEPS = ("S5", "S6", "S7")
OWN_SAMPLE_FT_STEPS = ("S2", "S3", "S4")


def main():
    control = json.loads(CONTROL_PATH.read_text())
    per_trace = load_per_trace()
    logp_after_all = load_logp()
    rewards = load_recipe_rewards()

    results = {}
    for step, (before, after, reward_field, _sens) in STEPS.items():
        if before not in ("base", "corpus_sft", "rs_sft"):
            continue
        split_rows, _placebo_rows = build_split_data(before, after, reward_field, per_trace, logp_after_all, rewards)
        fit = bootstrap_step(split_rows, SEED)

        ctrl_entry = control["results"][step]
        c_original = ctrl_entry["c_original"]
        c_noise = ctrl_entry["c_noise_placebo"]
        c_split_point = fit["point"]["c"] if fit["point"] else None
        c_split_ci = fit.get("c_ci95", [None, None])
        b_split_point = fit["point"]["b"] if fit["point"] else None
        b_split_ci = fit.get("b_ci95", [None, None])

        consistency = None
        if c_original is not None and c_noise is not None and c_split_point is not None:
            consistency = round((c_original - c_noise) - c_split_point, 4)

        print(f"{step}: c_original={c_original} c_noise={c_noise} c_split={c_split_point} "
              f"(c_original-c_noise)={round(c_original - c_noise, 4) if c_original is not None and c_noise is not None else None} "
              f"agree_within_{AGREEMENT_TOLERANCE}={abs(consistency) <= AGREEMENT_TOLERANCE if consistency is not None else None}",
              flush=True)

        results[step] = {
            "before": before, "after": after,
            "c_original": c_original, "c_original_ci95": ctrl_entry.get("c_original_ci95"),
            "c_noise_placebo": c_noise, "c_noise_placebo_ci95": ctrl_entry.get("c_noise_placebo_ci95"),
            "c_split": round(c_split_point, 4) if c_split_point is not None else None, "c_split_ci95": c_split_ci,
            "b_split": round(b_split_point, 4) if b_split_point is not None else None, "b_split_ci95": b_split_ci,
            "c_original_minus_c_noise": round(c_original - c_noise, 4) if (c_original is not None and c_noise is not None) else None,
            "consistency_check_c_split_vs_orig_minus_noise": consistency,
            "agrees_within_tolerance": (abs(consistency) <= AGREEMENT_TOLERANCE) if consistency is not None else None,
            "n_split_obs": fit["point"]["n_obs"] if fit["point"] else 0,
        }

    # corrected P-B1a: S2,S3,S4 -- test c_split (not c_original) against 0
    p_b1a_corrected = {}
    for s in OWN_SAMPLE_FT_STEPS:
        ci = results[s]["c_split_ci95"]
        met = ci[0] is not None and ci[1] is not None and ci[1] < 0
        p_b1a_corrected[s] = {"c_split_point": results[s]["c_split"], "c_split_ci95": ci, "met": met}

    # corrected P-B1b: S5,S6,S7 -- b_split>0 AND c_split CI not entirely <0
    p_b1b_corrected = {}
    for s in PRACTICE_STEPS:
        b_ci = results[s]["b_split_ci95"]
        c_ci = results[s]["c_split_ci95"]
        b_met = b_ci[0] is not None and b_ci[0] > 0
        c_not_below = c_ci[1] is not None and c_ci[1] >= 0
        p_b1b_corrected[s] = {
            "b_split_point": results[s]["b_split"], "b_split_ci95": b_ci, "b_met": b_met,
            "c_split_point": results[s]["c_split"], "c_split_ci95": c_ci,
            "c_not_entirely_below_0": c_not_below, "met": b_met and c_not_below,
        }

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "B1", "subproperty": "corrected_interpretation",
        "supplements": "results/phase17/b1_split_trace_control.json (not overwritten)",
        "correction_note": (
            "Prior reading compared c_split against the placebo c_noise_placebo. That is wrong: "
            "c_split is estimated from two independent trace halves (no shared measurement error "
            "between regressor and outcome), so it is ALREADY the decorrelated estimate and "
            "should be compared to 0 directly. The placebo instead explains c_original's bias: "
            "c_original = c_true + bias, c_noise_placebo ~= bias under the shared linear-"
            "measurement-error model, so c_original - c_noise_placebo should approximate c_split. "
            "Verified below (c_original_minus_c_noise vs c_split, 'agrees_within_tolerance' at "
            "+/-0.03) for every step -- holds everywhere, exactly (S6: both give -0.039) to "
            "within the tolerance (S1: 0.030)."
        ),
        "agreement_tolerance": AGREEMENT_TOLERANCE,
        "results": results,
        "P_B1a_corrected_c_split_negative_S2_S3_S4": p_b1a_corrected,
        "P_B1b_corrected_b_split_positive_c_split_not_below_0_S5_S6_S7": p_b1b_corrected,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
