#!/usr/bin/env python
"""
research/phase17_rh_reachability.py — Phase 17 Block R, R-h: reachability
(how much of the reward lives where practice cannot go). PHASE17_PREREG.md:

    On-policy practice tilts; it cannot reach recipes the policy does not
    sample. Per target, score every recipe in U(x) with the reward, and
    take pi_init from Block A. Compute the UNREACHABLE REWARD SHARE: the
    share of recipes in the reward's top decile that the init assigns
    pi<epsilon. Compute the TILT CEILING: the largest gain in log-
    probability a tilt of the fitted strength b_s (B1) can give such
    recipes. Back-test on Phase 12's four gained targets (ledger
    L6.gains): was the gained rule-picked set above or below epsilon
    under RS-SFT's teacher-forced probability? RS-SFT never sampled it
    in 32 draws.

REWARDS SCOPED TO R1, R2, R4 ONLY (same precedent as R-c, PHASE17_
PREREG's own back-test table, and rc_price_equation.py): R3 is scored on
a disjoint target set (hardening-probe's 40 targets, not U(x)'s ASTRAL+
held-out 75) and was never trained with; R5/R6 were never trained with
either. Skipped, not silently omitted -- flagged in the output.

  R1 -> pi_init = Corpus SFT's marginal estimator (R1's "policy whose
        existing samples are scored", Rewards table); b_s = B1's S5 (10ch
        primary).
  R2 -> pi_init = RS-SFT; b_s = B1's S6.
  R4 -> pi_init = RS-SFT; b_s = B1's S7.

NO EDIT to any frozen file. Reads results/phase17/logp_universe.jsonl,
universe_scores.jsonl (reused from Block B1, same pooled-reward tables),
and signature.json (B1's b_s). Requires Block B1 to have been run first.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_rh_reachability.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pymatgen.core import Composition  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
LOGP_PATH = REPO_ROOT / "results" / "phase17" / "logp_universe.jsonl"
SCORES_PATH = REPO_ROOT / "results" / "phase17" / "universe_scores.jsonl"
SIGNATURE_PATH = REPO_ROOT / "results" / "phase17" / "signature.json"
ASTRAL_VALIDATION_SET = REPO_ROOT / "results" / "astral_validation_set.json"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "audit_reach.json"

EPSILONS = {"1/100": 1.0 / 100, "1/200": 1.0 / 200, "1/500": 1.0 / 500}
PRIMARY_EPSILON = "1/200"

REWARDS = {
    "R1": {"field": "r1_10ch", "init_model": "corpus_sft", "b_step": "S5", "b_source": "point"},
    "R2": {"field": "r2_5ch", "init_model": "rs_sft", "b_step": "S6", "b_source": "point"},
    "R4": {"field": "r4_ranker", "init_model": "rs_sft", "b_step": "S7", "b_source": "point"},
}
SKIPPED_REWARDS_NOTE = (
    "R3, R5, R6 skipped -- same precedent as R-c (research/phase17_rc_price_equation.py) "
    "and the pre-reg's own back-test table (line 'R1 -> GDPO-300 (run 1), R2 -> Phase 12, "
    "R4 -> E1 (the three rewards that were actually trained with)'). R3 is additionally "
    "scored on a disjoint target set (the hardening probe's 40 targets, not U(x)'s ASTRAL+"
    "held-out 75), so it has no pi_init from Block A at all. R5/R6 were never trained with."
)
L6_GAINS_TARGETS = ("BaLiBO3", "KLi(PO3)2", "Li2TiSiO5", "NaSrBO3")


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def canon(formula: str) -> str:
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return formula.strip()


def load_logp() -> dict:
    out = defaultdict(dict)
    with LOGP_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if r["marginal_logp"] is None:
                continue
            out[(r["model"], r["target"])][tuple(r["recipe"])] = r["marginal_logp"]
    return out


def load_reward_table(field: str) -> dict:
    out = defaultdict(list)
    with SCORES_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            v = r.get(field)
            if v is not None:
                out[(r["target"], tuple(r["recipe"]))].append(v)
    return out


def pi_tilde(logp_by_recipe: dict) -> dict:
    if not logp_by_recipe:
        return {}
    keys = list(logp_by_recipe.keys())
    lps = [logp_by_recipe[k] for k in keys]
    log_z = logsumexp(lps)
    return {k: math.exp(lp - log_z) for k, lp in zip(keys, lps)}


def reward_audit_for(reward_id: str, cfg: dict, logp: dict, signature: dict) -> dict:
    reward_table = load_reward_table(cfg["field"])
    init_model = cfg["init_model"]

    b_entry = signature["steps"][cfg["b_step"]]
    b_fit = b_entry["fit"]["point"]
    r_tilde_values_all = []  # for the tilt-ceiling range, pooled across this step's own fitted data
    b_s = b_fit["b"] if b_fit else None

    per_target = {}
    for (model, target), by_recipe in logp.items():
        if model != init_model:
            continue
        pi = pi_tilde(by_recipe)
        # recipes with reward coverage for this target
        covered = {y: reward_table[(target, y)] for y in by_recipe if (target, y) in reward_table}
        if len(covered) < 10:
            continue  # need a meaningful decile
        r_mean = {y: sum(vs) / len(vs) for y, vs in covered.items()}
        sorted_r = sorted(r_mean.values())
        decile_cut = sorted_r[int(0.9 * len(sorted_r))]
        top_decile = [y for y, r in r_mean.items() if r >= decile_cut]

        by_eps = {}
        for eps_label, eps_val in EPSILONS.items():
            unreachable = [y for y in top_decile if pi.get(y, 0.0) < eps_val]
            by_eps[eps_label] = {
                "n_top_decile": len(top_decile),
                "n_unreachable": len(unreachable),
                "unreachable_share": round(len(unreachable) / len(top_decile), 4) if top_decile else None,
            }
        per_target[target] = {"n_covered_recipes": len(covered), "decile_cutoff_reward": round(decile_cut, 4),
                              "by_epsilon": by_eps}
        r_tilde_values_all.extend(r_mean[y] - sum(r_mean.values()) / len(r_mean) for y in r_mean)

    if not per_target:
        return {"reward_id": reward_id, "status": "no_data", "note": f"no (model,target) rows for init_model={init_model}"}

    shares_primary = [v["by_epsilon"][PRIMARY_EPSILON]["unreachable_share"] for v in per_target.values()
                      if v["by_epsilon"][PRIMARY_EPSILON]["unreachable_share"] is not None]
    mean_unreachable_share = sum(shares_primary) / len(shares_primary) if shares_primary else None

    tilt_ceiling_nats = None
    if b_s is not None and r_tilde_values_all:
        tilt_ceiling_nats = b_s * (max(r_tilde_values_all) - min(r_tilde_values_all))

    return {
        "reward_id": reward_id, "status": "ok", "init_model": init_model,
        "b_step": cfg["b_step"], "b_s_point": b_s,
        "n_targets": len(per_target),
        "mean_unreachable_reward_share_primary_eps": round(mean_unreachable_share, 4) if mean_unreachable_share is not None else None,
        "passes_R_h_threshold_lt_50pct": (mean_unreachable_share < 0.5) if mean_unreachable_share is not None else None,
        "tilt_ceiling_nats": round(tilt_ceiling_nats, 4) if tilt_ceiling_nats is not None else None,
        "per_target": per_target,
    }


def l6_gains_backtest(logp: dict) -> dict:
    """RS-SFT's teacher-forced probability of each gained target's rule-
    picked precursor set: above or below epsilon? RS-SFT never sampled
    it in 32 draws (ledger L6.gains)."""
    doc = json.loads(ASTRAL_VALIDATION_SET.read_text())
    predicted_by_target = {t["target"]: frozenset(canon(p) for p in t["predicted"]) for t in doc["targets"]}

    out = {}
    for target in L6_GAINS_TARGETS:
        predicted = predicted_by_target.get(target)
        by_recipe = logp.get(("rs_sft", target), {})
        if predicted is None or not by_recipe:
            out[target] = {"status": "no_data", "predicted_set": sorted(predicted) if predicted else None}
            continue
        pi = pi_tilde(by_recipe)
        match_key = next((y for y in by_recipe if frozenset(y) == predicted), None)
        if match_key is None:
            out[target] = {"status": "predicted_set_not_in_U(x)_for_rs_sft",
                          "predicted_set": sorted(predicted),
                          "note": "RS-SFT's U(x) membership for this target never includes the rule-picked "
                                  "set at all (not merely below epsilon) -- its marginal estimator has no "
                                  "entry to evaluate."}
            continue
        pi_val = pi[match_key]
        by_eps = {label: pi_val < eps for label, eps in EPSILONS.items()}
        out[target] = {"status": "ok", "predicted_set": sorted(predicted),
                      "pi_tilde_rs_sft": round(pi_val, 8),
                      "below_epsilon_by_eps": by_eps,
                      "below_epsilon_primary": by_eps[PRIMARY_EPSILON]}
    return out


def main():
    if not SIGNATURE_PATH.exists():
        print("ERROR: results/phase17/signature.json not found -- run Block B1 first "
              "(R-h's tilt ceiling needs B1's fitted b_s).", flush=True)
        sys.exit(1)
    signature = json.loads(SIGNATURE_PATH.read_text())

    print("loading logp_universe.jsonl...", flush=True)
    logp = load_logp()

    reward_results = {}
    for reward_id, cfg in REWARDS.items():
        print(f"\n{reward_id}: init_model={cfg['init_model']} b_step={cfg['b_step']}", flush=True)
        res = reward_audit_for(reward_id, cfg, logp, signature)
        print(f"  mean_unreachable_share={res.get('mean_unreachable_reward_share_primary_eps')} "
              f"passes(<50%)={res.get('passes_R_h_threshold_lt_50pct')} "
              f"tilt_ceiling_nats={res.get('tilt_ceiling_nats')}", flush=True)
        reward_results[reward_id] = res

    print("\nL6.gains back-test (Phase 12's 4 gained targets)...", flush=True)
    backtest = l6_gains_backtest(logp)
    for t, v in backtest.items():
        print(f"  {t}: {v}", flush=True)

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "R", "subproperty": "R-h reachability",
        "skipped_rewards_note": SKIPPED_REWARDS_NOTE,
        "epsilons_tested": list(EPSILONS.values()), "primary_epsilon": PRIMARY_EPSILON,
        "by_reward": reward_results,
        "l6_gains_backtest_no_direction_predicted": backtest,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
