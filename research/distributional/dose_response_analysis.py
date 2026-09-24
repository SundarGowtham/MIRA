#!/usr/bin/env python
"""
research/distributional/dose_response_analysis.py — Phase 15 Task 5,
questions 1-4. Pre-registration: docs/phases/PHASE15_DOSE_PREREG.md
(committed 0852277, before results/dose_response/teacher_forcing_raw.json
existed). This script only reads that raw file; it does not touch the
GPU, validator, comparator, or any training run.

Cluster bootstrap: resample precursor sets (the natural cluster; each
route is one precursor set) with replacement, recompute the statistic,
repeat N_BOOT times, take the 2.5/97.5 percentiles. Seed 20260925 per
the pre-reg (question 4); reused for questions 1-3 for consistency
(not itself pre-registered for those, since the instructions doc only
named a seed for question 4 -- noted as [E] methodology choice).

Usage:
  uv run python research/distributional/dose_response_analysis.py
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

RAW_PATH = Path("results/dose_response/teacher_forcing_raw.json")
OUT_JSON = Path("results/dose_response/analysis.json")
OUT_MD = Path("docs/phases/PHASE15_DOSE_RESULTS.md")

SEED = 20260925
N_BOOT = 10000
EPSILON = -20.0
PHASE_PURE_THRESHOLD = 50.0


def load_routes():
    return json.loads(RAW_PATH.read_text())


def cluster_bootstrap_ci(rows, stat_fn, rng, n_boot=N_BOOT):
    """rows: list of dicts, each already one cluster (one precursor set).
    stat_fn(rows) -> float. Returns (lo, hi) 95% CI."""
    n = len(rows)
    boot_stats = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))
    boot_stats.sort()
    lo = boot_stats[int(0.025 * n_boot)]
    hi = boot_stats[int(0.975 * n_boot) - 1]
    return (lo, hi)


def mean_of(rows, key):
    vals = [r[key] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def ols_coeffs(rows, y_key, x_keys):
    """Simple multiple OLS via normal equations (numpy-free, small n).
    Returns dict {x_key: coeff, 'intercept': c}."""
    import numpy as np
    X = np.array([[1.0] + [float(r[k]) for k in x_keys] for r in rows])
    y = np.array([float(r[y_key]) for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = {"intercept": beta[0]}
    for i, k in enumerate(x_keys):
        out[k] = beta[i + 1]
    return out


def main():
    routes = load_routes()
    rng = random.Random(SEED)

    rows = []
    for r in routes:
        route = r["route"]
        ck = r["checkpoints"]
        rows.append({
            "precursor_set": route["precursor_set"],
            "target": route["arrows_target"],
            "has_carbonate": route["has_carbonate"],
            "best_wt_pct": route["best_wt_pct"],
            "phase_pure": 1 if route["best_wt_pct"] >= PHASE_PURE_THRESHOLD else 0,
            "logp_base": ck["base"]["total_logp"],
            "logp_rs_sft": ck["rs_sft"]["total_logp"],
            "logp_gdpo": ck["gdpo_300"]["total_logp"],
            "logp_full_sft": ck["full_sft"]["total_logp"],
            "n_tokens_base": ck["base"]["n_tokens"],
            "top5_first_token_base": [t["token"] for t in (ck["base"]["top20_first_token"] or [])][:5],
        })
        rows[-1]["delta_gdpo_base"] = rows[-1]["logp_gdpo"] - rows[-1]["logp_base"]
        rows[-1]["delta_gdpo_rssft"] = rows[-1]["logp_gdpo"] - rows[-1]["logp_rs_sft"]
        rows[-1]["logp_base_per_tok"] = rows[-1]["logp_base"] / rows[-1]["n_tokens_base"]
        rows[-1]["delta_gdpo_base_per_tok"] = rows[-1]["delta_gdpo_base"] / rows[-1]["n_tokens_base"]

    results = {}

    # ================= Question 1: Leash =================
    below = [r for r in rows if r["logp_base"] < EPSILON]
    above = [r for r in rows if r["logp_base"] >= EPSILON]

    def mean_delta(subset):
        return mean_of(subset, "delta_gdpo_base") if subset else 0.0

    ci_below = cluster_bootstrap_ci(below, mean_delta, rng) if below else None
    ci_above = cluster_bootstrap_ci(above, mean_delta, rng) if above else None
    q1 = {
        "epsilon": EPSILON,
        "n_below": len(below), "mean_delta_below": mean_delta(below) if below else None,
        "ci_below": ci_below,
        "includes_zero_below": (ci_below[0] <= 0 <= ci_below[1]) if ci_below else None,
        "n_above": len(above), "mean_delta_above": mean_delta(above) if above else None,
        "ci_above": ci_above,
        "includes_zero_above": (ci_above[0] <= 0 <= ci_above[1]) if ci_above else None,
    }
    results["q1_leash"] = q1
    print("Q1 Leash:", json.dumps(q1, indent=1, default=str))

    # [E] supplement: the pre-registered epsilon=-20 was calibrated for a
    # per-token/short-sequence scale, but total_logp sums over the full
    # ~51-token array, so every route falls below it (n_above=0, degenerate
    # as specified -- not silently fixed, reported in q1 above). Post-hoc,
    # not a substitute for q1: repeat the same partition using per-token
    # mean log p, on the SAME fixed epsilon=-20 re-purposed as a per-token
    # cut (a different, more permissive threshold than intended, but the
    # only way to get a non-degenerate split without inventing a new number
    # after seeing data -- flagged as exploratory throughout).
    below_pt = [r for r in rows if r["logp_base_per_tok"] < EPSILON]
    above_pt = [r for r in rows if r["logp_base_per_tok"] >= EPSILON]

    def mean_delta_pt(subset):
        return mean_of(subset, "delta_gdpo_base_per_tok") if subset else 0.0

    ci_below_pt = cluster_bootstrap_ci(below_pt, mean_delta_pt, rng) if below_pt else None
    ci_above_pt = cluster_bootstrap_ci(above_pt, mean_delta_pt, rng) if above_pt else None
    q1e = {
        "note": "Pre-registered epsilon=-20 assumed a per-token/short-sequence "
                "log-p scale; total_logp sums over the full array span "
                "(mean n_tokens=51), so mean base log p = -96.2 and ALL 75 "
                "routes fall below epsilon in q1 above (n_above=0, degenerate "
                "as specified). This supplement reuses the same epsilon=-20 "
                "value applied to PER-TOKEN mean log p instead -- an ad hoc "
                "rescue, not a pre-registered test; reported as [E] only.",
        "epsilon": EPSILON,
        "n_below": len(below_pt), "mean_delta_per_tok_below": mean_delta_pt(below_pt) if below_pt else None,
        "ci_below": ci_below_pt,
        "n_above": len(above_pt), "mean_delta_per_tok_above": mean_delta_pt(above_pt) if above_pt else None,
        "ci_above": ci_above_pt,
    }
    results["q1e_per_token_supplement"] = q1e
    print("Q1e [E] per-token supplement:", json.dumps(q1e, indent=1, default=str))

    # ================= Question 2: Quality alignment =================
    coeffs = ols_coeffs(rows, "delta_gdpo_base", ["phase_pure", "logp_base"])

    def phase_pure_coef(subset):
        if len({r["phase_pure"] for r in subset}) < 2:
            return coeffs["phase_pure"]
        c = ols_coeffs(subset, "delta_gdpo_base", ["phase_pure", "logp_base"])
        return c["phase_pure"]

    ci_pp = cluster_bootstrap_ci(rows, phase_pure_coef, rng)
    q2 = {
        "phase_pure_threshold_wt_pct": PHASE_PURE_THRESHOLD,
        "n_phase_pure": sum(r["phase_pure"] for r in rows),
        "n_phase_impure": len(rows) - sum(r["phase_pure"] for r in rows),
        "coefficients": coeffs,
        "phase_pure_coef_ci": ci_pp,
        "excludes_zero": not (ci_pp[0] <= 0 <= ci_pp[1]),
        "positive_and_excludes_zero": coeffs["phase_pure"] > 0 and not (ci_pp[0] <= 0 <= ci_pp[1]),
    }
    results["q2_quality_alignment"] = q2
    print("Q2 Quality alignment:", json.dumps(q2, indent=1, default=str))

    # ================= Question 3: Top-5 (rescoped) =================
    q3_per_target = {}
    for target in ["YBCO", "LTOPO", "NTMO"]:
        target_rows = [r for r in rows if r["target"] == target]
        best = max(target_rows, key=lambda r: (r["best_wt_pct"], -len(r["precursor_set"].split(","))))
        first_formula = best["precursor_set"].split(",")[0].strip()
        first_char = first_formula[0]
        in_top5 = any(tok.strip().startswith(first_char) or first_char in tok for tok in best["top5_first_token_base"])
        # more precise: check literal formula-string prefix match against decoded tokens
        in_top5_strict = any(first_formula.startswith(tok) or tok == first_char for tok in best["top5_first_token_base"])
        q3_per_target[target] = {
            "best_precursor_set": best["precursor_set"],
            "best_wt_pct": best["best_wt_pct"],
            "first_precursor_formula": first_formula,
            "base_top5_first_token": best["top5_first_token_base"],
            "in_top5_loose_char_match": in_top5,
            "in_top5_strict_prefix_match": in_top5_strict,
        }
    n_hit = sum(1 for v in q3_per_target.values() if v["in_top5_strict_prefix_match"])
    q3 = {
        "per_target": q3_per_target,
        "n_targets_hit": n_hit,
        "n_targets_total": 3,
        "prediction_supported": n_hit >= 2,
    }
    results["q3_top5_rescoped"] = q3
    print("Q3 Top-5:", json.dumps(q3, indent=1, default=str))

    # ================= Question 4: Policy-level carbonate test =================
    rows4 = [dict(r, carbonate_status=1 if r["has_carbonate"] else 0) for r in rows]
    coeffs4 = ols_coeffs(rows4, "delta_gdpo_rssft", ["carbonate_status", "best_wt_pct"])

    def carb_coef(subset):
        if len({r["carbonate_status"] for r in subset}) < 2:
            return coeffs4["carbonate_status"]
        c = ols_coeffs(subset, "delta_gdpo_rssft", ["carbonate_status", "best_wt_pct"])
        return c["carbonate_status"]

    def wt_coef(subset):
        if len({r["carbonate_status"] for r in subset}) < 2:
            return coeffs4["best_wt_pct"]
        c = ols_coeffs(subset, "delta_gdpo_rssft", ["carbonate_status", "best_wt_pct"])
        return c["best_wt_pct"]

    ci_carb = cluster_bootstrap_ci(rows4, carb_coef, rng)
    ci_wt = cluster_bootstrap_ci(rows4, wt_coef, rng)
    q4 = {
        "n_carbonate": sum(r["carbonate_status"] for r in rows4),
        "n_non_carbonate": len(rows4) - sum(r["carbonate_status"] for r in rows4),
        "coefficients": coeffs4,
        "carbonate_status_ci": ci_carb,
        "carbonate_excludes_zero": not (ci_carb[0] <= 0 <= ci_carb[1]),
        "target_wt_pct_ci": ci_wt,
        "target_wt_pct_includes_zero": ci_wt[0] <= 0 <= ci_wt[1],
        "prediction1_supported": not (ci_carb[0] <= 0 <= ci_carb[1]),
        "prediction2_supported": ci_wt[0] <= 0 <= ci_wt[1],
        "mean_delta_gdpo_rssft_carbonate": mean_of([r for r in rows4 if r["carbonate_status"] == 1], "delta_gdpo_rssft"),
        "mean_delta_gdpo_rssft_noncarbonate": mean_of([r for r in rows4 if r["carbonate_status"] == 0], "delta_gdpo_rssft"),
    }
    results["q4_policy_carbonate"] = q4
    print("Q4 Policy carbonate:", json.dumps(q4, indent=1, default=str))

    # ================= [E] descriptive stats =================
    descriptive = {
        "n_routes": len(rows),
        "mean_logp_base": statistics.mean(r["logp_base"] for r in rows),
        "mean_logp_rs_sft": statistics.mean(r["logp_rs_sft"] for r in rows),
        "mean_logp_gdpo": statistics.mean(r["logp_gdpo"] for r in rows),
        "mean_logp_full_sft": statistics.mean(r["logp_full_sft"] for r in rows),
        "mean_delta_gdpo_base": statistics.mean(r["delta_gdpo_base"] for r in rows),
        "mean_delta_gdpo_rssft": statistics.mean(r["delta_gdpo_rssft"] for r in rows),
    }
    results["descriptive_e"] = descriptive
    print("[E] descriptive:", json.dumps(descriptive, indent=1, default=str))

    OUT_JSON.write_text(json.dumps(results, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
