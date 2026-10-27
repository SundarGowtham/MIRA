#!/usr/bin/env python
"""
research/phase17_b1_split_trace_control.py — errors-in-variables control
for Block B1 (regular Claude's item 2, 2026-10-27): is B1's negative c_s
for the practice steps (S5/S6/S7) a real effect, or classic regression-
to-the-mean from measurement noise in log pi_before appearing with
opposite sign on both sides of Delta = log pi_after - log pi_before?

Uses results/phase17/logp_per_trace_before_models.jsonl (8 individual
per-trace log pi_m(y|x,tau_k) values for base/corpus_sft/rs_sft, from
research/phase17_b1_split_trace_forward_pass.py -- a NEW GPU run,
estimated at <12h and approved, actually ~41min).

For every (model,target,recipe) cell with exactly 8 traces (fewer are
excluded, reported as a coverage note):
  log_pi_A = log(mean(exp(trace_1..4)))   -- half A, the REGRESSOR
  log_pi_B = log(mean(exp(trace_5..8)))   -- half B, used only in the
                                             outcome below
  Delta_split = log_pi_after(standard, logp_universe.jsonl) - log_pi_B
  Delta_noise = log_pi_B - log_pi_A                  (PLACEBO: no real
                                                       training step at
                                                       all, same model,
                                                       same (model,
                                                       target) -- any
                                                       negative slope
                                                       here is pure
                                                       measurement-noise
                                                       artifact)

Refits B1's exact within-target FE regression (reusing
phase17_b1_signature_regression's within_target_fe_ols/bootstrap_step,
not duplicated) for:
  (a) the SPLIT model: Delta_split ~ b*r~_A + c*log_pi_A + d*pi~_A*r~_A
  (b) the PLACEBO: Delta_noise ~ c_noise*log_pi_A (single regressor,
      within-target demeaned, no r~/interaction -- isolates the pure
      noise mechanism with nothing else in the equation)

Table: | step | c (original, signature.json) | c (split) | c_noise (placebo) | CI each |

NO EDIT to any frozen file.

Usage (tmux, CPU only -- the GPU work already happened in the forward-
pass script this reads from):
  PYTHONPATH=. uv run python research/phase17_b1_split_trace_control.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase17_b1_signature_regression import (  # noqa: E402
    STEPS, load_logp, load_recipe_rewards, pi_tilde_before, within_target_fe_ols, logsumexp,
)

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
PER_TRACE_PATH = REPO_ROOT / "results" / "phase17" / "logp_per_trace_before_models.jsonl"
SIGNATURE_PATH = REPO_ROOT / "results" / "phase17" / "signature.json"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "b1_split_trace_control.json"
SEED = 20261016
N_BOOT = 10000


def load_per_trace() -> dict:
    """{(model,target): {recipe_tuple: [8 per-trace logp values]}}"""
    out = defaultdict(dict)
    n_total, n_full8 = 0, 0
    with PER_TRACE_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            vals = r["per_trace_logp"]
            if len(vals) == 8:
                n_full8 += 1
                out[(r["model"], r["target"])][tuple(r["recipe"])] = vals
    print(f"per-trace coverage: {n_full8}/{n_total} cells have exactly 8 traces "
          f"({n_full8/n_total:.1%}) -- only these are used below.", flush=True)
    return out


def half_logp(vals: list[float], lo: int, hi: int) -> float:
    """log(mean(exp(vals[lo:hi])))."""
    sub = vals[lo:hi]
    return logsumexp(sub) - math.log(len(sub))


def build_split_data(before_model: str, after_model: str, reward_field: str,
                     per_trace: dict, logp_after_all: dict, rewards: dict) -> tuple[list, list]:
    """Returns (split_rows, placebo_rows). split_rows have delta=Delta_split,
    log_pi_before=log_pi_A, r_tilde/interaction built from pi~_A. placebo_rows
    have delta=Delta_noise, log_pi_before=log_pi_A (no r_tilde/interaction --
    the placebo is a single-regressor model)."""
    reward_table = rewards[reward_field]
    split_rows, placebo_rows = [], []

    by_target = defaultdict(dict)
    for (model, target), by_recipe in per_trace.items():
        if model == before_model:
            by_target[target] = by_recipe

    after_by_target = {t: r for (m, t), r in logp_after_all.items() if m == after_model}

    for target, before_recipes in by_target.items():
        after_recipes = after_by_target.get(target)
        if not after_recipes:
            continue
        log_pi_A = {y: half_logp(vals, 0, 4) for y, vals in before_recipes.items()}
        log_pi_B = {y: half_logp(vals, 4, 8) for y, vals in before_recipes.items()}
        pi_tilde_A = pi_tilde_before(log_pi_A)

        covered = {y: pi_tilde_A[y] for y in log_pi_A
                  if (target, y) in reward_table and reward_table[(target, y)]}
        w_sum = sum(covered.values())
        r_by_recipe, r_bar = {}, 0.0
        if w_sum > 0:
            r_by_recipe = {y: sum(reward_table[(target, y)]) / len(reward_table[(target, y)]) for y in covered}
            r_bar = sum(pi_tilde_A[y] / w_sum * r_by_recipe[y] for y in covered)

        for y in log_pi_A:
            if y not in after_recipes:
                continue
            # placebo: no reward coverage needed, pure noise mechanism
            placebo_rows.append({
                "target": target, "recipe": y,
                "delta": log_pi_B[y] - log_pi_A[y],
                "log_pi_before": log_pi_A[y],
                "r_tilde": 0.0, "interaction": 0.0,
            })
            if y in r_by_recipe:
                r_tilde = r_by_recipe[y] - r_bar
                split_rows.append({
                    "target": target, "recipe": y,
                    "delta": after_recipes[y] - log_pi_B[y],
                    "log_pi_before": log_pi_A[y],
                    "r_tilde": r_tilde,
                    "pi_tilde_before": pi_tilde_A[y],
                    "interaction": pi_tilde_A[y] * r_tilde,
                })
    return split_rows, placebo_rows


def bootstrap_simple(rows: list[dict], seed: int, use_full_model: bool) -> dict:
    """Same cluster-bootstrap-by-target convention as B1. If use_full_model,
    fits b/c/d (3-regressor); else fits c only (single-regressor placebo,
    still within-target demeaned)."""
    def fit(rows):
        if use_full_model:
            return within_target_fe_ols(rows)
        by_target = defaultdict(list)
        for r in rows:
            by_target[r["target"]].append(r)
        Y, X = [], []
        for target, trows in by_target.items():
            n = len(trows)
            if n < 2:
                continue
            mean_d = sum(r["delta"] for r in trows) / n
            mean_lp = sum(r["log_pi_before"] for r in trows) / n
            for r in trows:
                Y.append(r["delta"] - mean_d)
                X.append(r["log_pi_before"] - mean_lp)
        n = len(Y)
        if n < 10:
            return None
        sxx = sum(x * x for x in X)
        sxy = sum(x * y for x, y in zip(X, Y))
        if sxx < 1e-12:
            return None
        return {"c": sxy / sxx, "n_obs": n}

    point = fit(rows)
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)
    targets = sorted(by_target.keys())
    n = len(targets)
    rng = random.Random(seed)
    boots = []
    for _ in range(N_BOOT):
        sample_targets = [targets[rng.randrange(n)] for _ in range(n)]
        resampled = [r for t in sample_targets for r in by_target[t]]
        f = fit(resampled)
        if f is not None:
            boots.append(f["c"])
    boots.sort()
    ci = [boots[int(0.025 * len(boots))], boots[min(int(0.975 * len(boots)), len(boots) - 1)]] if boots else [None, None]
    return {"point_c": point["c"] if point else None, "c_ci95": ci,
           "n_obs": point["n_obs"] if point else 0, "n_boot_valid": len(boots)}


def main():
    per_trace = load_per_trace()
    logp_after_all = load_logp()
    rewards = load_recipe_rewards()
    signature = json.loads(SIGNATURE_PATH.read_text())

    results = {}
    for step, (before, after, reward_field, _sens) in STEPS.items():
        if before not in ("base", "corpus_sft", "rs_sft"):
            results[step] = {"status": "skipped", "reason": f"before model {before} not in the 3 re-scored models"}
            continue
        split_rows, placebo_rows = build_split_data(before, after, reward_field, per_trace, logp_after_all, rewards)
        print(f"\n{step}: {before}->{after}, n_split_rows={len(split_rows)} n_placebo_rows={len(placebo_rows)}", flush=True)
        split_fit = bootstrap_simple(split_rows, SEED, use_full_model=True)
        placebo_fit = bootstrap_simple(placebo_rows, SEED, use_full_model=False)
        orig_c = signature["steps"][step]["fit"]["point"]
        orig_c = orig_c["c"] if orig_c else None
        orig_ci = signature["steps"][step]["fit"].get("c_ci95")
        print(f"  c_original={orig_c} ci={orig_ci}", flush=True)
        print(f"  c_split={split_fit.get('point_c')} ci={split_fit.get('c_ci95')} n={split_fit.get('n_obs')}", flush=True)
        print(f"  c_noise(placebo)={placebo_fit.get('point_c')} ci={placebo_fit.get('c_ci95')} n={placebo_fit.get('n_obs')}", flush=True)
        results[step] = {
            "before": before, "after": after,
            "c_original": orig_c, "c_original_ci95": orig_ci,
            "c_split": split_fit.get("point_c"), "c_split_ci95": split_fit.get("c_ci95"),
            "c_noise_placebo": placebo_fit.get("point_c"), "c_noise_placebo_ci95": placebo_fit.get("c_ci95"),
            "n_split_obs": split_fit.get("n_obs"), "n_placebo_obs": placebo_fit.get("n_obs"),
        }

    report = {"prereg_commit": PREREG_COMMIT, "block": "B1", "subproperty": "errors-in-variables split-trace control",
              "method": "4/4 trace split; regressor=log_pi_A (traces 1-4), Delta_split=log_pi_after-log_pi_B (traces 5-8); "
                       "placebo Delta_noise=log_pi_B-log_pi_A regressed on log_pi_A (pure measurement noise, no training step)",
              "results": results}
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
