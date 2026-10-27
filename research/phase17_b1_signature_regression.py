#!/usr/bin/env python
"""
research/phase17_b1_signature_regression.py — Phase 17 Block B1: signature
regression. PHASE17_PREREG.md:

    Delta_s(y) = a_x + b_s*r~(y) + c_s*log(pi_before(y)) + d_s*pi~_before(y)*r~(y) + eps

Fit with target fixed effects (within-target demeaning = the standard FE
estimator) and a cluster bootstrap by target (10,000 resamples, seed
20261016).

STEP -> REWARD, per regular Claude's decision (2026-10-27), which also
corrects a pre-reg self-contradiction: the governing rule is "the reward
the step itself trained on" (line 121-124), which makes the pre-reg's own
"S5: 5-channel checker aggregate" wrong -- S5 is run 1, which trained on
the ORIGINAL 10-channel checker (R1 in Block R). Logged as a pre-reg
DEVIATION below, not silently followed or silently corrected.
  S1-S4 (fine-tuning, reward used only as covariate): 5-ch RUN3 ("r2_5ch")
  S5  (run 1, 10ch primary / 5ch sensitivity): "r1_10ch" primary, "r2_5ch" sensitivity
  S6  (Phase 12, 5ch):                          "r2_5ch"
  S7  (E1, ranker v2):                          "r4_ranker"

r(y): plain mean of gradeable (non-None) channels per sample, pooled
across ALL samples (any model) that produced y for that target --
results/phase17/universe_scores.jsonl, built by
research/phase17_universe_scores.py. r~(y) = r(y) - sum_{y' w/ r defined}
pi~_before(y')*r(y') where pi~_before is itself renormalised over ALL y
in U(x) with a defined log pi_before (Block A's marginal estimator, NOT
restricted to the reward-covered subset), then the weights used in the
r(y') sum are renormalised onto just the reward-covered subset (a
properly-weighted average of r over the targets' reward-covered
recipes using pi_before's own shape) -- this is the only point where
coverage (decision 3) forces a scope restriction narrower than literal
pi~ on the full U(x); documented explicitly in the output, not glossed
over. pi_before(y) itself (for the c_s/d_s terms) uses every y with a
defined logp regardless of reward coverage.

NO EDIT to any frozen file. Reads only existing results/phase17/
logp_universe.jsonl, universe_scores.jsonl, and the raw training files.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_b1_signature_regression.py
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

from core.reward import parse_completion  # noqa: E402
from research.support_growth_by_checkpoint import precset_of  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
LOGP_PATH = REPO_ROOT / "results" / "phase17" / "logp_universe.jsonl"
SCORES_PATH = REPO_ROOT / "results" / "phase17" / "universe_scores.jsonl"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "signature.json"
SEED = 20261016
N_BOOT = 10000

# (before_model, after_model, reward_field, sensitivity_reward_field_or_None)
STEPS = {
    "S1": ("base", "corpus_sft", "r2_5ch", None),
    "S2": ("base", "rs_sft", "r2_5ch", None),
    "S3": ("base", "e3", "r2_5ch", None),
    "S4": ("base", "e3c", "r2_5ch", None),
    "S5": ("corpus_sft", "gdpo_v3_ckpt300", "r1_10ch", "r2_5ch"),
    "S6": ("rs_sft", "phase12_ckpt300", "r2_5ch", None),
    "S7": ("rs_sft", "e1_ckpt300", "r4_ranker", None),
}
# supplementary intermediate checkpoints for S6/S7 (B2's territory, not B1's
# primary fit, but cheap to report alongside)
SUPPLEMENTARY_AFTER = {
    "S6": ["phase12_ckpt100", "phase12_ckpt200"],
    "S7": ["e1_ckpt100", "e1_ckpt200"],
}

OWN_SAMPLE_FT_STEPS = ("S2", "S3", "S4")
PRACTICE_STEPS = ("S5", "S6", "S7")

TRAINING_FILES = {
    "S2": [REPO_ROOT / "data" / "rs_sft" / "rs_sft_train.jsonl", REPO_ROOT / "data" / "rs_sft" / "rs_sft_val.jsonl"],
    "S3": [REPO_ROOT / "data" / "e3" / "e3_train.jsonl", REPO_ROOT / "data" / "e3" / "e3_val.jsonl"],
    "S4": [REPO_ROOT / "data" / "e3c" / "e3c_train.jsonl", REPO_ROOT / "data" / "e3c" / "e3c_val.jsonl"],
}


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def load_logp() -> dict:
    """{(model, target): {recipe_tuple: {"marginal": lp, "answer_only": lp}}}"""
    out = defaultdict(dict)
    with LOGP_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if r["marginal_logp"] is None:
                continue
            key = tuple(r["recipe"])
            out[(r["model"], r["target"])][key] = r["marginal_logp"]
    return out


def load_recipe_rewards() -> dict:
    """{reward_field: {(target, recipe_tuple): [scores...]}} -- pooled across
    every model/sample that produced that (target, recipe)."""
    fields = ("r1_10ch", "r2_5ch", "r4_ranker")
    out = {f: defaultdict(list) for f in fields}
    with SCORES_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["target"], tuple(r["recipe"]))
            for field in fields:
                v = r.get(field)
                if v is not None:
                    out[field][key].append(v)
    return out


def pi_tilde_before(logp_by_recipe: dict) -> dict:
    """Renormalise log pi_before over every recipe with a defined logp
    for this (model, target) -- softmax over the full marginal-estimator
    -scored universe, per the pre-reg's own renormalisation convention
    (same one Gate G1's epsilon-support fix used)."""
    if not logp_by_recipe:
        return {}
    keys = list(logp_by_recipe.keys())
    lps = [logp_by_recipe[k] for k in keys]
    log_z = logsumexp(lps)
    return {k: math.exp(lp - log_z) for k, lp in zip(keys, lps)}


def build_step_data(before_model: str, after_model: str, reward_field: str,
                    logp: dict, rewards: dict) -> tuple[list[dict], dict]:
    """One row per (target, recipe) with every quantity B1's regression
    needs, restricted to recipes with: logp_before AND logp_after defined
    (both from the marginal estimator), AND at least one gradeable sample
    for reward_field (coverage requirement, decision 3). Also returns
    coverage stats: the share of (target,recipe) pairs with both logp's
    defined that ALSO have reward coverage -- required by the pre-reg
    ("report coverage: the share of U(x) with at least one scored
    sample")."""
    rows = []
    n_both_logp_defined = 0
    reward_table = rewards[reward_field]
    before_by_target = defaultdict(dict)
    after_by_target = defaultdict(dict)
    for (model, target), by_recipe in logp.items():
        if model == before_model:
            before_by_target[target] = by_recipe
        elif model == after_model:
            after_by_target[target] = by_recipe

    for target, before_recipes in before_by_target.items():
        after_recipes = after_by_target.get(target)
        if not after_recipes:
            continue
        n_both_logp_defined += sum(1 for y in before_recipes if y in after_recipes)
        pi_tilde = pi_tilde_before(before_recipes)
        pi_tilde_after_map = pi_tilde_before(after_recipes)
        # weights renormalised onto the reward-covered subset, for r~(y)'s
        # pi-weighted-mean-reward term (decision 3's coverage scope)
        covered = {y: pi_tilde[y] for y in before_recipes
                  if (target, y) in reward_table and reward_table[(target, y)]}
        w_sum = sum(covered.values())
        if w_sum <= 0:
            continue
        r_by_recipe = {y: sum(reward_table[(target, y)]) / len(reward_table[(target, y)])
                      for y in covered}
        r_bar = sum(pi_tilde[y] / w_sum * r_by_recipe[y] for y in covered)

        for y, logp_before in before_recipes.items():
            if y not in after_recipes:
                continue
            if y not in r_by_recipe:
                continue
            logp_after = after_recipes[y]
            r_y = r_by_recipe[y]
            r_tilde = r_y - r_bar
            rows.append({
                "target": target, "recipe": y,
                "delta": logp_after - logp_before,
                "r_tilde": r_tilde,
                "log_pi_before": logp_before,
                "pi_tilde_before": pi_tilde[y],
                "pi_tilde_after": pi_tilde_after_map.get(y),
                "interaction": pi_tilde[y] * r_tilde,
                "r_raw": r_y, "n_pooled_samples": len(reward_table[(target, y)]),
            })
    coverage = {
        "n_recipes_both_logp_defined": n_both_logp_defined,
        "n_recipes_with_reward_coverage": len(rows),
        "coverage_share": round(len(rows) / n_both_logp_defined, 4) if n_both_logp_defined else None,
    }
    return rows, coverage


def within_target_fe_ols(rows: list[dict]) -> dict | None:
    """Within-target (fixed-effects) OLS: Delta = b*r~ + c*log_pi_before +
    d*interaction + eps, target FE absorbed by demeaning each variable
    within its own target group. No intercept (demeaning removes it)."""
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)
    # need >=2 obs per target to demean meaningfully; targets with 1 obs
    # contribute nothing (demeaned value is exactly 0)
    Y, X1, X2, X3 = [], [], [], []
    for target, trows in by_target.items():
        n = len(trows)
        if n < 2:
            continue
        mean_d = sum(r["delta"] for r in trows) / n
        mean_r = sum(r["r_tilde"] for r in trows) / n
        mean_lp = sum(r["log_pi_before"] for r in trows) / n
        mean_ia = sum(r["interaction"] for r in trows) / n
        for r in trows:
            Y.append(r["delta"] - mean_d)
            X1.append(r["r_tilde"] - mean_r)
            X2.append(r["log_pi_before"] - mean_lp)
            X3.append(r["interaction"] - mean_ia)
    n = len(Y)
    if n < 10:
        return None
    # closed-form (X'X)^-1 X'Y for a 3-column design, via Gaussian elimination
    cols = [X1, X2, X3]
    XtX = [[sum(a * b for a, b in zip(ci, cj)) for cj in cols] for ci in cols]
    XtY = [sum(a * b for a, b in zip(ci, Y)) for ci in cols]
    coef = solve_3x3(XtX, XtY)
    if coef is None:
        return None
    return {"b": coef[0], "c": coef[1], "d": coef[2], "n_obs": n, "n_targets": len(by_target)}


def solve_3x3(A: list[list[float]], b: list[float]) -> list[float] | None:
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(M[r][i]))
        if abs(M[piv][i]) < 1e-12:
            return None
        M[i], M[piv] = M[piv], M[i]
        pivval = M[i][i]
        M[i] = [v / pivval for v in M[i]]
        for r in range(3):
            if r != i:
                factor = M[r][i]
                M[r] = [M[r][k] - factor * M[i][k] for k in range(4)]
    return [M[i][3] for i in range(3)]


def bootstrap_step(rows: list[dict], seed: int) -> dict:
    point = within_target_fe_ols(rows)
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)
    targets = sorted(by_target.keys())
    n = len(targets)
    rng = random.Random(seed)
    boots = {"b": [], "c": [], "d": []}
    for _ in range(N_BOOT):
        sample_targets = [targets[rng.randrange(n)] for _ in range(n)]
        resampled_rows = [r for t in sample_targets for r in by_target[t]]
        fit = within_target_fe_ols(resampled_rows)
        if fit is not None:
            for k in ("b", "c", "d"):
                boots[k].append(fit[k])
    out = {"point": point}
    for k in ("b", "c", "d"):
        vals = sorted(boots[k])
        if not vals:
            out[f"{k}_ci95"] = [None, None]
            continue
        lo = vals[int(0.025 * len(vals))]
        hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
        out[f"{k}_ci95"] = [lo, hi]
    out["n_boot_valid"] = len(boots["b"])
    return out


# ---------------------------------------------------------------------------
# m-flow curve fit (S2-S4)
# ---------------------------------------------------------------------------

def load_training_recipes(step: str) -> dict[str, set]:
    """{target: set(recipe_tuple)} from that step's ACTUAL training file
    (train+val combined), per decision 4."""
    out = defaultdict(set)
    for path in TRAINING_FILES[step]:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                try:
                    route = parse_completion(rec["completion"], rec["target"])
                except Exception:
                    continue
                out[rec["target"]].add(precset_of(route))
    return out


def fit_m_flow_curve(rows: list[dict], train_recipes: dict[str, set]) -> dict:
    """Delta(y) = log(1 - lambda + lambda*q(y)/pi_before(y)), q uniform
    over that target's TRAINING recipes (decision 4). Only targets
    present in the training file qualify; <10 such targets -> not
    estimable."""
    qualifying_targets = {t for t in train_recipes if train_recipes[t]}
    fit_rows = [r for r in rows if r["target"] in qualifying_targets]
    targets_present = {r["target"] for r in fit_rows}
    if len(targets_present) < 10:
        return {"estimable": False, "n_qualifying_targets": len(targets_present),
               "reason": "fewer than 10 targets with training-file recipes present in U(x)"}

    def q_of(r):
        n_train = len(train_recipes[r["target"]])
        return 1.0 / n_train if r["recipe"] in train_recipes[r["target"]] else 0.0

    # the curve's q(y)/pi_before(y) ratio needs a common, within-target
    # absolute scale, so pi_tilde_before (renormalised on U(x)) is used
    # here, not the raw marginal-estimator probability.
    pi_tilde = [r["pi_tilde_before"] for r in fit_rows]
    q_vals = [q_of(r) for r in fit_rows]
    deltas = [r["delta"] for r in fit_rows]

    def predicted(lam):
        out = []
        for q, pt in zip(q_vals, pi_tilde):
            ratio = q / pt if pt > 1e-300 else (0.0 if q == 0 else 1e6)
            val = 1 - lam + lam * ratio
            out.append(math.log(val) if val > 1e-12 else math.log(1e-12))
        return out

    def sse(lam):
        pred = predicted(lam)
        return sum((d - p) ** 2 for d, p in zip(deltas, pred))

    # grid + local refine
    best_lam, best_sse = 0.5, sse(0.5)
    for lam in [i / 200 for i in range(1, 200)]:
        s = sse(lam)
        if s < best_sse:
            best_sse, best_lam = s, lam
    # golden-section-ish local refine around best_lam
    lo, hi = max(1e-4, best_lam - 0.01), min(0.9999, best_lam + 0.01)
    for _ in range(40):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if sse(m1) < sse(m2):
            hi = m2
        else:
            lo = m1
    best_lam = (lo + hi) / 2
    best_sse = sse(best_lam)

    mean_d = sum(deltas) / len(deltas)
    tss = sum((d - mean_d) ** 2 for d in deltas)
    r2 = 1 - best_sse / tss if tss > 0 else None

    # e-flow alternative: Delta = t*r~ - psi (OLS, with intercept)
    r_tilde = [r["r_tilde"] for r in fit_rows]
    n = len(r_tilde)
    mean_r = sum(r_tilde) / n
    mean_d2 = sum(deltas) / n
    cov = sum((r - mean_r) * (d - mean_d2) for r, d in zip(r_tilde, deltas))
    var_r = sum((r - mean_r) ** 2 for r in r_tilde)
    t_coef = cov / var_r if var_r > 1e-300 else None
    psi = -(mean_d2 - (t_coef or 0) * mean_r)
    if t_coef is not None:
        pred_e = [t_coef * r - psi for r in r_tilde]
        sse_e = sum((d - p) ** 2 for d, p in zip(deltas, pred_e))
        r2_e = 1 - sse_e / tss if tss > 0 else None
    else:
        r2_e = None

    return {"estimable": True, "n_qualifying_targets": len(targets_present),
           "n_rows": len(fit_rows), "lambda": round(best_lam, 4), "r2_m_flow": round(r2, 4) if r2 else r2,
           "e_flow_alt": {"t": round(t_coef, 4) if t_coef is not None else None,
                          "psi": round(psi, 4) if t_coef is not None else None,
                          "r2": round(r2_e, 4) if r2_e is not None else None}}


def main():
    print("loading logp_universe.jsonl and universe_scores.jsonl...", flush=True)
    logp = load_logp()
    rewards = load_recipe_rewards()

    step_results = {}
    step_rows_cache = {}
    for step, (before, after, reward_field, sens_field) in STEPS.items():
        print(f"\n{step}: {before} -> {after}, reward={reward_field}"
              f"{' (+sensitivity '+sens_field+')' if sens_field else ''}", flush=True)
        rows, coverage = build_step_data(before, after, reward_field, logp, rewards)
        step_rows_cache[step] = rows
        fit = bootstrap_step(rows, SEED)
        print(f"  n_obs={fit['point']['n_obs'] if fit['point'] else None} "
              f"n_targets={fit['point']['n_targets'] if fit['point'] else None} "
              f"coverage={coverage['coverage_share']} "
              f"b_ci={fit.get('b_ci95')} c_ci={fit.get('c_ci95')} d_ci={fit.get('d_ci95')}", flush=True)
        entry = {"before": before, "after": after, "reward_field": reward_field, "fit": fit,
                "coverage": coverage, "n_obs": fit["point"]["n_obs"] if fit["point"] else 0}
        if sens_field:
            sens_rows, sens_coverage = build_step_data(before, after, sens_field, logp, rewards)
            entry["sensitivity"] = {"reward_field": sens_field, "fit": bootstrap_step(sens_rows, SEED),
                                    "coverage": sens_coverage}
        step_results[step] = entry

    # supplementary intermediate checkpoints for S6/S7 (not part of B1's
    # primary predictions, reported alongside for free since the data's
    # already loaded)
    for step, after_list in SUPPLEMENTARY_AFTER.items():
        before, _after300, reward_field, _ = STEPS[step]
        supp = {}
        for after_ck in after_list:
            rows, cov = build_step_data(before, after_ck, reward_field, logp, rewards)
            supp[after_ck] = {"fit": bootstrap_step(rows, SEED), "coverage": cov}
        step_results[step]["supplementary_checkpoints"] = supp

    # P-B1a: S2,S3,S4 -- c_s<0, CI excludes 0 (upper bound < 0)
    p_b1a = {}
    for s in OWN_SAMPLE_FT_STEPS:
        ci = step_results[s]["fit"].get("c_ci95", [None, None])
        met = (ci[0] is not None and ci[1] is not None and ci[1] < 0)
        p_b1a[s] = {"c_point": step_results[s]["fit"]["point"]["c"] if step_results[s]["fit"]["point"] else None,
                    "c_ci95": ci, "met": met}

    # P-B1b: S5,S6,S7 -- b_s>0, CI excludes 0 (lower bound > 0), AND CI of
    # c_s not lying entirely below 0 (upper bound of c_s CI >= 0)
    p_b1b = {}
    for s in PRACTICE_STEPS:
        fit = step_results[s]["fit"]
        b_ci = fit.get("b_ci95", [None, None])
        c_ci = fit.get("c_ci95", [None, None])
        b_met = (b_ci[0] is not None and b_ci[0] > 0)
        c_not_below = (c_ci[1] is not None and c_ci[1] >= 0)
        p_b1b[s] = {"b_point": fit["point"]["b"] if fit["point"] else None, "b_ci95": b_ci, "b_met": b_met,
                    "c_point": fit["point"]["c"] if fit["point"] else None, "c_ci95": c_ci,
                    "c_not_entirely_below_0": c_not_below, "met": b_met and c_not_below}

    # P-B1c: CI of c_S3 - c_S4 includes 0 -- joint bootstrap (same
    # resampled target draw applied to both steps' data each replicate)
    by_target_s3 = defaultdict(list)
    for r in step_rows_cache["S3"]:
        by_target_s3[r["target"]].append(r)
    by_target_s4 = defaultdict(list)
    for r in step_rows_cache["S4"]:
        by_target_s4[r["target"]].append(r)
    shared_targets = sorted(set(by_target_s3) | set(by_target_s4))
    rng = random.Random(SEED)
    diffs = []
    for _ in range(N_BOOT):
        sample_targets = [shared_targets[rng.randrange(len(shared_targets))] for _ in range(len(shared_targets))]
        rows3 = [r for t in sample_targets for r in by_target_s3.get(t, [])]
        rows4 = [r for t in sample_targets for r in by_target_s4.get(t, [])]
        fit3 = within_target_fe_ols(rows3)
        fit4 = within_target_fe_ols(rows4)
        if fit3 is not None and fit4 is not None:
            diffs.append(fit3["c"] - fit4["c"])
    diffs.sort()
    c3 = step_results["S3"]["fit"]["point"]["c"] if step_results["S3"]["fit"]["point"] else None
    c4 = step_results["S4"]["fit"]["point"]["c"] if step_results["S4"]["fit"]["point"] else None
    point_diff = (c3 - c4) if (c3 is not None and c4 is not None) else None
    if diffs:
        lo = diffs[int(0.025 * len(diffs))]
        hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
        p_b1c = {"point_diff_c_S3_minus_c_S4": point_diff, "ci95": [lo, hi],
                "met_includes_0": lo <= 0 <= hi, "n_boot_valid": len(diffs)}
    else:
        p_b1c = {"point_diff_c_S3_minus_c_S4": point_diff, "ci95": [None, None],
                "met_includes_0": None, "n_boot_valid": 0}

    # measured-no-direction + does-not-support checks
    measured_no_direction = {
        "d_sign_S5_S7": {s: step_results[s]["fit"]["point"]["d"] if step_results[s]["fit"]["point"] else None
                        for s in PRACTICE_STEPS},
        "c_S1": step_results["S1"]["fit"]["point"]["c"] if step_results["S1"]["fit"]["point"] else None,
    }
    does_not_support = {
        "c_s_negative_for_any_practice_step":
            [s for s in PRACTICE_STEPS if step_results[s]["fit"].get("c_ci95", [None, None])[1] is not None
             and step_results[s]["fit"]["c_ci95"][1] < 0],
        "c_s_ci_entirely_above_0_for_any_own_sample_ft_step":
            [s for s in OWN_SAMPLE_FT_STEPS if step_results[s]["fit"].get("c_ci95", [None, None])[0] is not None
             and step_results[s]["fit"]["c_ci95"][0] > 0],
    }

    # m-flow curve fits, S2-S4
    curve_fits = {}
    for s in OWN_SAMPLE_FT_STEPS:
        train_recipes = load_training_recipes(s)
        curve_fits[s] = fit_m_flow_curve(step_rows_cache[s], train_recipes)
        print(f"{s} curve fit: {curve_fits[s]}", flush=True)

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "B1", "subproperty": "signature regression",
        "seed": SEED, "n_boot": N_BOOT,
        "prereg_deviation_S5_reward": (
            "The pre-reg (line 122) names S5's recipe reward as the 5-channel RUN3 "
            "checker aggregate, but its own governing rule (line 121, 'the reward the "
            "step itself trained on') requires the ORIGINAL 10-channel checker: S5 is "
            "run 1 (Corpus SFT -> GDPO-300), and run 1 trained against the 10-channel "
            "checker (R1 in Block R), not the 5-channel RUN3 subset introduced only for "
            "run 3 onward. Per regular Claude's decision (2026-10-27): S5 uses the "
            "10-channel checker as PRIMARY and the 5-channel aggregate as a sensitivity "
            "check (reported in each step's 'sensitivity' field)."
        ),
        "reward_per_step": {s: v[2] for s, v in STEPS.items()},
        "coverage_scope_note": (
            "r(y) pools every existing sample (any model, any target set) whose parsed "
            "route canonicalises to recipe y -- results/phase17/universe_scores.jsonl, "
            "scored over the FULL raw-completion corpus (54,320 rows), not a subsample. "
            "pi~_before(y) is renormalised over every y in U(x) with a defined marginal "
            "log-probability (Block A), regardless of reward coverage; the "
            "pi-weighted-mean-reward term in r~(y) then renormalises those same weights "
            "onto just the reward-covered subset of that renormalised pi~ -- the only "
            "point coverage narrows the pre-reg's literal 'renormalised on U(x)' scope, "
            "done because r(y') is undefined, not 0, for uncovered y'."
        ),
        "steps": step_results,
        "predictions": {
            "P_B1a_c_negative_S2_S3_S4": p_b1a,
            "P_B1b_b_positive_c_not_below_0_S5_S6_S7": p_b1b,
            "P_B1c_c_S3_minus_c_S4_CI_includes_0": p_b1c,
        },
        "measured_no_direction_predicted": measured_no_direction,
        "does_not_support_checks": does_not_support,
        "m_flow_curve_fits_S2_S4": curve_fits,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
