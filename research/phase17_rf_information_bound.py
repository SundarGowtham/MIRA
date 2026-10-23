#!/usr/bin/env python
"""
research/phase17_rf_information_bound.py — Phase 17 Block R, R-f
(validity: how much the reward knows about the lab). PHASE17_PREREG.md:

    I(update; Q) <= I(R; Q)

What matters is information BEYOND a trivial rule: I(R;Q | carbonate-free).
Computed for R2 (checker RUN3 vote), R4 (ranker v2 vote, NEW here --
the frozen Phase 15 script research/distributional/arrows_gate_scoring.py
never had a ranker rule), R5 (comparator vote), and the carbonate-free
baseline itself, on ARROWS3 YBCO pairs (the primary |delta|>=5wt%
endpoint, finding 27's own data) and ASTRAL's 35 predicted-vs-traditional
pairs (flagged small, n=35 not n=1915).

REUSE, NOT REBUILD: research/distributional/arrows_gate_scoring.py is a
FROZEN pre-registered script ("no changes after seeing results" --
its own docstring) that already implements load_target_entries,
build_route, score_validator, gdpo_vote_and_channels (R2's vote),
comparator_rule (R5's vote), baseline_carbonate_free, actual_winner,
and cluster_bootstrap_ci. This script imports every one of them
UNCHANGED and calls score_target() itself for YBCO to get its real,
already-vetted _all_pairs_primary_threshold list (set_a/set_b/temp/
wt_a/wt_b/gdpo_winner/comp_winner/base_carbfree) -- then independently
rebuilds the SAME routes-by-(precursor_set,temp) dict load_target_entries
exposes, to add ONLY the one missing vote (ranker v2) per pair. No
edit to arrows_gate_scoring.py itself.

MI DEFINITION: R in {-1,0,1} (B wins / tie / A wins), Q in {-1,1}
(actual_winner, already excludes <5wt% pairs so ties are near-absent).
I(R;Q) = sum_{r,q} p(r,q) log2(p(r,q)/(p(r)p(q))) over the empirical
joint from the pair list (pairs are already deduplicated per the
frozen script's own construction). I(R;Q|CF) = sum_cf p(cf) * I(R;Q
on the cf-subset).

NO EDIT to core/ranker.py, core/comparator.py, validator.py, or
arrows_gate_scoring.py. NO NEW GENERATION/DATA. Bootstrap: cluster by
precursor-set label (reusing cluster_bootstrap_ci's own resampling,
not reinvented), 10,000 resamples, seed 20261016 (project convention;
overrides the frozen script's own SEED=20260924 for THIS script's new
computation only).

Usage (tmux, CPU only, real PD cache):
  PYTHONPATH=. uv run python research/phase17_rf_information_bound.py
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
from probe_hardening import load_literature  # noqa: E402
from research.ranker_capacity_rssft import TRIAGE, SYNTHESIS  # noqa: E402
from research.distributional.arrows_gate_scoring import (  # noqa: E402
    load_target_entries, build_route, score_validator, score_target,
    baseline_carbonate_free, actual_winner, cluster_bootstrap_ci,
    COMPARATOR_SCALES_PATH, CANONICAL_TARGET,
)
from validator import ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rf_information_bound.json"
ASTRAL_VALIDATION_SET = REPO_ROOT / "results" / "astral_validation_set.json"
SEED = 20261016
N_BOOT = 10000


def ranker_vote(ranker: Ranker, route_a, route_b, target_formula: str, lit_T=None, lit_n_ops=None) -> int:
    score_a, _ = ranker.score(route_a, target_formula, lit_T=lit_T, lit_n_ops=lit_n_ops)
    score_b, _ = ranker.score(route_b, target_formula, lit_T=lit_T, lit_n_ops=lit_n_ops)
    diff = score_a - score_b
    return 1 if diff > 1e-9 else (-1 if diff < -1e-9 else 0)


def mutual_info(pairs: list[dict], r_key: str, q_key: str = "actual") -> float:
    joint = defaultdict(int)
    r_marg = defaultdict(int)
    q_marg = defaultdict(int)
    n = 0
    for p in pairs:
        r, q = p.get(r_key), p.get(q_key)
        if r is None or q is None:
            continue
        joint[(r, q)] += 1
        r_marg[r] += 1
        q_marg[q] += 1
        n += 1
    if n == 0:
        return 0.0
    mi = 0.0
    for (r, q), c_rq in joint.items():
        p_rq = c_rq / n
        p_r = r_marg[r] / n
        p_q = q_marg[q] / n
        if p_rq > 0 and p_r > 0 and p_q > 0:
            mi += p_rq * math.log2(p_rq / (p_r * p_q))
    return max(mi, 0.0)


def conditional_mutual_info(pairs: list[dict], r_key: str, cf_key: str = "base_carbfree", q_key: str = "actual") -> float:
    by_cf = defaultdict(list)
    for p in pairs:
        cf = p.get(cf_key)
        if cf is None:
            continue
        by_cf[cf].append(p)
    n_total = sum(len(v) for v in by_cf.values())
    if n_total == 0:
        return 0.0
    cmi = 0.0
    for cf, subset in by_cf.items():
        cmi += (len(subset) / n_total) * mutual_info(subset, r_key, q_key)
    return cmi


def bootstrap_mi(pairs: list[dict], r_key: str, conditional: bool, seed: int) -> dict:
    rng = random.Random(seed)
    fn = (lambda ps: conditional_mutual_info(ps, r_key)) if conditional else (lambda ps: mutual_info(ps, r_key))
    point = fn(pairs)
    lo, hi = cluster_bootstrap_ci(pairs, fn, rng, n_boot=N_BOOT)
    return {"point_bits": round(point, 5), "ci95_bits": [round(lo, 5), round(hi, 5)]}


def add_ranker_votes(pairs: list[dict], routes: dict, ranker: Ranker, target_formula: str,
                     lit_T=None, lit_n_ops=None) -> list[dict]:
    out = []
    for p in pairs:
        key_a = (p["set_a"], p["temp"])
        key_b = (p["set_b"], p["temp"])
        route_a, route_b = routes.get(key_a), routes.get(key_b)
        if route_a is None or route_b is None:
            continue
        rv = ranker_vote(ranker, route_a, route_b, target_formula, lit_T, lit_n_ops)
        actual = actual_winner(p["wt_a"], p["wt_b"], 5)  # primary threshold, matches the frozen script
        if actual is None:
            continue
        out.append({**p, "ranker_winner": rv, "actual": actual})
    return out


def ybco_pairs_with_ranker(validator, comparator, scales, ranker, rng) -> list[dict]:
    ybco = score_target("YBCO", validator, comparator, scales, rng)
    pairs = ybco["_all_pairs_primary_threshold"]
    entries, target_formula = load_target_entries("YBCO")
    routes = {}
    for e in entries:
        key = (e["precursor_set"], e["temp_str"])
        routes[key] = build_route(target_formula, e["precursors"], e["amounts"], e["temp_C"], e["atmosphere"])
    return add_ranker_votes(pairs, routes, ranker, target_formula)


def astral_pairs(ranker, validator, comparator, scales) -> list[dict]:
    """ASTRAL's 35 predicted-vs-traditional pairs: Q = which recipe had
    higher measured phase purity (trad_best_purity vs pred_best_purity).
    Amounts are placeholder 1.0 (ASTRAL gives species, not stoichiometry
    -- same caveat astral_validation.py documents)."""
    doc = json.loads(ASTRAL_VALIDATION_SET.read_text())
    pairs = []
    for t in doc["targets"]:
        target = t["target"]
        route_trad = build_route(target, t["traditional"], [1.0] * len(t["traditional"]), t["trad_best_T"], "air")
        route_pred = build_route(target, t["predicted"], [1.0] * len(t["predicted"]), t["pred_best_T"], "air")
        bd_trad = score_validator(validator, route_trad, target)
        bd_pred = score_validator(validator, route_pred, target)
        from research.distributional.arrows_gate_scoring import gdpo_vote_and_channels, comparator_rule
        gdpo_w, _reason, _ = gdpo_vote_and_channels(bd_trad, bd_pred)  # a=trad, b=pred
        comp_w, _ = comparator_rule(comparator, scales, route_trad, route_pred, target)
        cf_w = baseline_carbonate_free(route_trad, route_pred)
        rk_w = ranker_vote(ranker, route_trad, route_pred, target)
        dp = t["trad_best_purity"] - t["pred_best_purity"]
        actual = 1 if dp > 1e-9 else (-1 if dp < -1e-9 else 0)
        pairs.append({
            "set_a": "trad::" + target, "set_b": "pred::" + target,
            "gdpo_winner": gdpo_w, "comp_winner": comp_w, "base_carbfree": cf_w, "ranker_winner": rk_w,
            "actual": actual,
        })
    return pairs


def report_for(pairs: list[dict], label: str) -> dict:
    out = {"label": label, "n_pairs": len(pairs)}
    for reward_id, r_key in [("R2_checker", "gdpo_winner"), ("R4_ranker", "ranker_winner"),
                              ("R5_comparator", "comp_winner"), ("carbonate_free_baseline", "base_carbfree")]:
        i_rq = bootstrap_mi(pairs, r_key, conditional=False, seed=SEED)
        entry = {"I(R;Q)_bits": i_rq}
        if reward_id != "carbonate_free_baseline":
            i_rq_given_cf = bootstrap_mi(pairs, r_key, conditional=True, seed=SEED)
            entry["I(R;Q|carbonate_free)_bits"] = i_rq_given_cf
            entry["ci_includes_zero"] = i_rq_given_cf["ci95_bits"][0] <= 0 <= i_rq_given_cf["ci95_bits"][1]
        out[reward_id] = entry
    return out


def main():
    rng = random.Random(SEED)
    print("loading validator, comparator, ranker...", flush=True)
    validator = load_validator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                               REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    comparator = load_comparator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                                 REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    scales = json.loads(COMPARATOR_SCALES_PATH.read_text())["scales"]
    lit = load_literature(TRIAGE, SYNTHESIS)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)

    print("\n=== ARROWS3 YBCO primary endpoint (|delta|>=5wt%) ===", flush=True)
    ybco_pairs = ybco_pairs_with_ranker(validator, comparator, scales, ranker, rng)
    print(f"  n_pairs={len(ybco_pairs)}", flush=True)
    ybco_report = report_for(ybco_pairs, "ARROWS3_YBCO_primary_5wtpct")
    print(json.dumps(ybco_report, indent=1), flush=True)

    print("\n=== ASTRAL 35 predicted-vs-traditional pairs (flagged small) ===", flush=True)
    a_pairs = astral_pairs(ranker, validator, comparator, scales)
    print(f"  n_pairs={len(a_pairs)}", flush=True)
    astral_report = report_for(a_pairs, "ASTRAL_35_pairs_SMALL")
    print(json.dumps(astral_report, indent=1), flush=True)

    prediction_rows = []
    for reward_id in ("R2_checker", "R4_ranker", "R5_comparator"):
        ci_incl_zero = ybco_report[reward_id]["ci_includes_zero"]
        prediction_rows.append({"reward": reward_id, "ci_includes_zero": ci_incl_zero})
    n_pass = sum(1 for r in prediction_rows if r["ci_includes_zero"])

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-f (validity: data-processing inequality, I(R;Q|carbonate-free))",
        "seed": SEED, "n_boot": N_BOOT,
        "ranker_vote_note": "R4's vote is NEW in this script -- the frozen Phase 15 "
                            "arrows_gate_scoring.py never had a ranker rule. Added here by "
                            "importing that script's routes/pairs unchanged and scoring each "
                            "pair's two routes with Ranker.score() directly, without editing it.",
        "ybco_primary": ybco_report,
        "astral_35_pairs_small": astral_report,
        "prediction_check": {
            "statement": "For R2, R4 and R5, the CI of I(R;Q|carbonate-free) includes 0 (finding "
                         "27 restated in information-theoretic units). Checked on the ARROWS3 YBCO "
                         "primary endpoint.",
            "rows": prediction_rows, "n_pass": n_pass, "n_total": 3,
            "passes": n_pass == 3,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nprediction_check: {n_pass}/3 (passes={report['prediction_check']['passes']})", flush=True)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
