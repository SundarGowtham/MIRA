#!/usr/bin/env python
"""
research/phase17_rf_stratum_reconciliation.py — third follow-up to
results/phase17/rf_information_bound.json, per regular Claude's review:

1. n per stratum for every reward (already present from
   phase17_rf_directional_agreement.py's directional_agreement_by_stratum;
   restated here alongside the new fields for a single read-point).

2. The carbonate-TIE stratum (base_carbfree == 0: both recipes have the
   SAME carbonate status) is the one that isolates signal BEYOND the
   carbonate-free rule -- reported here as the PRIMARY "beyond the rule"
   number for R2/R4/R5, using finding 27's own agreement_score convention
   (ties credited 0.5, denominator = ALL pairs with a defined vote and a
   defined actual winner -- NOT decisive-only, which undercounts relative
   to finding 27's n=1,112). Finding 27's quoted number (0.522, CI
   [0.495, 0.557], n=1,112) is `thermodynamic_favorable` ALONE, restricted
   to this exact tie stratum via the same base_carbfree==0 cut
   (PHASE15_ARROWS_RESULTS.md, Task 4d(c)) -- same pairs, same cut,
   DIFFERENT reward (one channel vs the full checker vote) AND different
   agreement convention (decisive-only in the earlier follow-up vs
   ties-credited-0.5 here). Both differences are named explicitly.

3. Symmetrize: pools the two "rule decided" strata (-1, +1) into one
   "rule_decided" cell and recomputes, to test whether the asymmetry
   (R4 significant only when the rule prefers B) survives pooling. Also
   runs an explicit A/B-relabeling randomization: swap A<->B on a random
   half of pairs (negating base_carbfree, each reward's vote, and
   `actual` on the swapped half), then recompute per-stratum agreement --
   if the asymmetry washes out under random relabeling, it was an
   ordering artifact of how set_a/set_b were assigned, not a real
   composition difference between the two strata.

4. Holm-adjusted p-values (exact two-sided binomial test vs 0.5) across
   the 9 (reward x stratum) cells on the YBCO primary endpoint, reported
   beside the raw per-cell p.

Does NOT overwrite any existing field -- adds `beyond_rule_tie_stratum`,
`symmetrized_rule_decided`, `ab_randomization_check`, and
`holm_adjusted_significance` per reward / globally, additively.

Usage (tmux, CPU only, reuses ybco_pairs_with_ranker by import, no new
PD calls):
  PYTHONPATH=. uv run python research/phase17_rf_stratum_reconciliation.py
"""
from __future__ import annotations

import json
import math
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
from research.distributional.arrows_gate_scoring import cluster_bootstrap_ci, agreement_score  # noqa: E402
from research.phase17_rf_information_bound import (  # noqa: E402
    ybco_pairs_with_ranker, COMPARATOR_SCALES_PATH, OUT_PATH, SEED,
)
from validator import ThermoChecker  # noqa: E402

N_BOOT = 10000
REWARD_KEYS = [("R2_checker", "gdpo_winner"), ("R4_ranker", "ranker_winner"), ("R5_comparator", "comp_winner")]


def mean_agreement_f27_style(pairs: list[dict], r_key: str) -> float:
    """Finding-27's own convention: ties credited 0.5, denominator = all
    pairs with a defined vote AND a defined actual (NOT decisive-only)."""
    scored = [agreement_score(p[r_key], p["actual"]) for p in pairs if p.get(r_key) is not None and p.get("actual") is not None]
    return sum(scored) / len(scored) if scored else 0.5


def bootstrap_agreement_f27(pairs: list[dict], r_key: str, seed: int) -> dict:
    point = mean_agreement_f27_style(pairs, r_key)
    lo, hi = cluster_bootstrap_ci(pairs, lambda ps: mean_agreement_f27_style(ps, r_key), random.Random(seed), n_boot=N_BOOT)
    n = sum(1 for p in pairs if p.get(r_key) is not None and p.get("actual") is not None)
    return {"n": n, "agreement": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "clears_chance_above": lo > 0.5, "below_chance_significant": hi < 0.5}


def two_sided_binom_p(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test p-value vs null p0, no scipy dependency."""
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i)) for i in range(n + 1)]
    p_k = probs[k]
    tail = sum(pr for pr in probs if pr <= p_k + 1e-12)
    return min(1.0, tail)


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values in
    the ORIGINAL order of pvals."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def randomize_ab(pairs: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for p in pairs:
        if rng.random() < 0.5:
            swapped = dict(p)
            for k in ("base_carbfree", "gdpo_winner", "ranker_winner", "comp_winner", "actual"):
                if k in swapped and swapped[k] is not None:
                    swapped[k] = -swapped[k]
            swapped["set_a"], swapped["set_b"] = p.get("set_b"), p.get("set_a")
            out.append(swapped)
        else:
            out.append(p)
    return out


def main():
    rng = random.Random(SEED)
    print("rebuilding YBCO pairs...", flush=True)
    validator = load_validator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                               REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    comparator = load_comparator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                                 REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    scales = json.loads(COMPARATOR_SCALES_PATH.read_text())["scales"]
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)
    pairs = ybco_pairs_with_ranker(validator, comparator, scales, ranker, rng)
    print(f"n_pairs={len(pairs)}", flush=True)

    report = json.loads(OUT_PATH.read_text())

    # --- (2) beyond-the-rule tie stratum, finding-27-style agreement ---
    tie_pairs = [p for p in pairs if p.get("base_carbfree") == 0]
    print(f"tie stratum n={len(tie_pairs)}", flush=True)
    beyond_rule = {}
    for reward_id, r_key in REWARD_KEYS:
        res = bootstrap_agreement_f27(tie_pairs, r_key, SEED)
        beyond_rule[reward_id] = res
        print(f"  {reward_id} (f27-style, tie stratum): {res}", flush=True)
        report["ybco_primary"][reward_id]["beyond_rule_tie_stratum"] = res

    report["ybco_primary"]["beyond_rule_reconciliation_note"] = {
        "statement": "Finding 27's 0.522 [0.495,0.557] (PHASE15_ARROWS_RESULTS.md Task 4d(c)) is the "
                     "SAME pairs (base_carbfree==0 cut on YBCO primary, n=1,112) and the SAME agreement "
                     "convention used here (ties credited 0.5, all pairs with a defined vote in the "
                     "denominator) -- but it scores `thermodynamic_favorable` ALONE, not the full "
                     "checker vote reported as R2_checker here. Any difference between finding 27's "
                     "0.522 and R2_checker's tie-stratum number above is attributable to channel count "
                     "(1 channel vs the full 10-channel vote), not a methodology mismatch -- the earlier "
                     "phase17_rf_directional_agreement.py number for this stratum (0.6561) used a "
                     "DIFFERENT, decisive-only convention (excludes ties from both numerator and "
                     "denominator, n_decisive=599 of 1,112) and is not directly comparable to 0.522 "
                     "for that reason either; this field is the fair, apples-to-apples comparison.",
        "n_pairs_tie_stratum": len(tie_pairs),
        "finding_27_thermo_only_tie_stratum": {"agreement": 0.522, "ci95": [0.495, 0.557], "n": 1112},
    }

    # --- (3a) symmetrize: pool rule-decided strata ---
    rule_decided = [p for p in pairs if p.get("base_carbfree") in (-1, 1)]
    symmetrized = {}
    for reward_id, r_key in REWARD_KEYS:
        res = bootstrap_agreement_f27(rule_decided, r_key, SEED)
        symmetrized[reward_id] = res
        print(f"  {reward_id} (f27-style, pooled rule-decided): {res}", flush=True)

    # --- (3b) A/B randomization check ---
    rand_pairs = randomize_ab(pairs, SEED + 1)
    randomization_check = {}
    for reward_id, r_key in REWARD_KEYS:
        by_cf = defaultdict(list)
        for p in rand_pairs:
            cf = p.get("base_carbfree")
            if cf is not None:
                by_cf[cf].append(p)
        cells = {}
        for cf_val in (-1, 0, 1):
            subset = by_cf.get(cf_val, [])
            if subset:
                cells[str(cf_val)] = bootstrap_agreement_f27(subset, r_key, SEED)
        randomization_check[reward_id] = cells
        print(f"  {reward_id} (post A/B-randomization, per stratum): {cells}", flush=True)

    report["symmetrized_rule_decided"] = symmetrized
    report["ab_randomization_check"] = {
        "statement": "Swap A<->B on a random half of pairs (negating base_carbfree, each reward's "
                     "vote, and `actual` on the swapped half), then recompute per-stratum agreement. "
                     "If R4's original asymmetry (significant only in the 'rule prefers B' stratum) "
                     "washes out here -- i.e. the two non-tie strata converge to similar agreement -- "
                     "that is evidence the original asymmetry was an ordering artifact of how set_a/"
                     "set_b happened to be assigned, not a real composition difference between the "
                     "strata.",
        "per_reward": randomization_check,
    }

    # --- (4) Holm-adjusted p-values across 9 (reward x stratum) cells ---
    cells_for_holm = []
    for reward_id, r_key in REWARD_KEYS:
        strat = report["ybco_primary"][reward_id]["directional_agreement_by_stratum"]
        for cf_val in ("-1", "0", "1"):
            cell = strat.get(cf_val)
            if cell and cell.get("n_decisive"):
                n = cell["n_decisive"]
                k = round(cell["agreement_rate"] * n)
                p_raw = two_sided_binom_p(k, n, 0.5)
                cells_for_holm.append({"reward": reward_id, "stratum_base_carbfree": cf_val,
                                       "n_decisive": n, "k_agree": k,
                                       "agreement_rate": cell["agreement_rate"], "p_raw": round(p_raw, 5)})
    p_raws = [c["p_raw"] for c in cells_for_holm]
    p_holm = holm_adjust(p_raws)
    for c, p_adj in zip(cells_for_holm, p_holm):
        c["p_holm"] = round(p_adj, 5)
        c["significant_at_0.05_holm"] = p_adj < 0.05
    n_sig_holm = sum(1 for c in cells_for_holm if c["significant_at_0.05_holm"])
    report["holm_adjusted_significance"] = {
        "statement": "9 cells tested (3 rewards x 3 base_carbfree strata, decisive-pairs agreement "
                     "vs 0.5, YBCO primary). Exact two-sided binomial p per cell, Holm-Bonferroni "
                     "step-down adjustment across all 9.",
        "cells": cells_for_holm,
        "n_significant_raw_0.05": sum(1 for p in p_raws if p < 0.05),
        "n_significant_holm_0.05": n_sig_holm,
    }

    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nn_significant_holm_0.05 = {n_sig_holm}/9", flush=True)
    print(f"-> {OUT_PATH} (beyond_rule_tie_stratum, symmetrized_rule_decided, "
          f"ab_randomization_check, holm_adjusted_significance added)")


if __name__ == "__main__":
    main()
