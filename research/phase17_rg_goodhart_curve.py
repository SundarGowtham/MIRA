#!/usr/bin/env python
"""
research/phase17_rg_goodhart_curve.py — Phase 17 Block R, R-g (Goodhart
curve: best-of-n as a free simulation of practice). PHASE17_PREREG.md:

Picking the best of n samples by a reward is a cheap stand-in for
optimising against it; its KL from the policy is at most
log(n) - (n-1)/n (Beirami et al. 2024). Per ARROWS3 target, draw
n in {1,2,4,8,16} recipes at random from the measured recipe pool
(2,000 draws per n), select the best by each reward, record the
measured yield of the pick. Prediction: at every n, the CIs of R2 and
R4 overlap the carbonate-free rule's. [E]: does any reward's curve
turn down at large n (Gao/Schulman/Hilton 2023 over-optimisation shape).

REUSE: load_target_entries, build_route, score_validator (all from the
FROZEN research/distributional/arrows_gate_scoring.py, unedited) supply
the measured recipe pool per target (precursor_set x temp, each with
a real wt_pct). This script adds the scalar "best-of-n" selection
rules that pairwise/vector rewards need but never had:

  R2 (checker):     scalar = sum of RUN3_CHECKS' raw values (same
                     channel set nansum_rule already sums pairwise;
                     here summed per-route, not per-pair).
  R4 (ranker):       Ranker.score()'s own scalar output directly.
  R5 (comparator):   no native scalar (pairwise only) -- Copeland
                     score: count of pairwise wins (comparator_rule)
                     against the other n-1 draws in THIS SPECIFIC
                     n-sample, ties counting 0.5. A principled,
                     documented simplification; Block G's Hodge
                     gradient would be the rigorous version, not
                     built for this single sub-property.
  carbonate_free:    1 if the recipe avoids every carbonate, else 0.
  random:            uniform random pick among the n draws.
  oracle:            the draw with the actual highest measured yield
                     (upper bound; not a reward at all).

NO EDIT to core/ranker.py, core/comparator.py, validator.py, or
arrows_gate_scoring.py. NO NEW DATA. Draws: 2,000 per (target, n),
seed 20261016 (project convention).

Usage (tmux, CPU only, real PD cache):
  PYTHONPATH=. uv run python research/phase17_rg_goodhart_curve.py
"""
from __future__ import annotations

import json
import math
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import load_comparator, ComparatorParams  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency  # noqa: E402
from core.reward import RUN3_CHECKS, load_validator  # noqa: E402
from research.ranker_capacity_rssft import SYNTHESIS  # noqa: E402
from research.distributional.arrows_gate_scoring import (  # noqa: E402
    load_target_entries, build_route, score_validator, baseline_carbonate_free,
    comparator_rule, COMPARATOR_SCALES_PATH, CARBONATES,
)
from validator import ThermoChecker, SynthesisValidator  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rg_goodhart_curve.json"
SEED = 20261016
N_DRAWS = 2000
N_VALUES = [1, 2, 4, 8, 16]
TARGETS = ["YBCO", "LTOPO", "NTMO"]


def kl_bound(n: int) -> float:
    return math.log(n) - (n - 1) / n if n > 0 else 0.0


def checker_scalar(bd: dict | None) -> float | None:
    if bd is None:
        return None
    total = 0.0
    n_grad = 0
    for ch in RUN3_CHECKS:
        v = bd.get(ch)
        tag = bd.get(f"{ch}_gradeability")
        if v is not None and tag not in SynthesisValidator.SENTINEL_TAGS:
            total += v
            n_grad += 1
    return total if n_grad else None


def carbonate_free_scalar(route) -> float:
    return 0.0 if any(p.formula in CARBONATES for p in route.precursors) else 1.0


def build_pool(name: str, validator, ranker) -> tuple[list[dict], str]:
    entries, target_formula = load_target_entries(name)
    pool = []
    for e in entries:
        route = build_route(target_formula, e["precursors"], e["amounts"], e["temp_C"], e["atmosphere"])
        bd = score_validator(validator, route, target_formula)
        score_r4, _ = ranker.score(route, target_formula)
        pool.append({
            "precursor_set": e["precursor_set"], "temp": e["temp_str"], "wt_pct": e["wt_pct"],
            "route": route, "bd": bd, "r2_scalar": checker_scalar(bd), "r4_scalar": score_r4,
            "cf_scalar": carbonate_free_scalar(route),
        })
    return pool, target_formula


def pick_best(draw: list[dict], method: str, comparator, scales, target_formula, rng) -> dict:
    if method == "random":
        return rng.choice(draw)
    if method == "oracle":
        return max(draw, key=lambda d: d["wt_pct"])
    if method in ("R2_checker", "R4_ranker", "carbonate_free"):
        key = {"R2_checker": "r2_scalar", "R4_ranker": "r4_scalar", "carbonate_free": "cf_scalar"}[method]
        scored = [(d, d[key]) for d in draw if d[key] is not None]
        if not scored:
            return rng.choice(draw)
        best_val = max(v for _, v in scored)
        ties = [d for d, v in scored if abs(v - best_val) < 1e-9]
        return rng.choice(ties)
    if method == "R5_comparator":
        wins = {id(d): 0.0 for d in draw}
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                w, _reason = comparator_rule(comparator, scales, draw[i]["route"], draw[j]["route"], target_formula)
                if w == 1:
                    wins[id(draw[i])] += 1.0
                elif w == -1:
                    wins[id(draw[j])] += 1.0
                else:
                    wins[id(draw[i])] += 0.5
                    wins[id(draw[j])] += 0.5
        best_win = max(wins.values())
        ties = [d for d in draw if wins[id(d)] == best_win]
        return rng.choice(ties)
    raise ValueError(method)


def bootstrap_mean_ci(vals: list[float], seed: int, n_boot: int = 10000) -> dict:
    rng = random.Random(seed)
    n = len(vals)
    point = sum(vals) / n
    boots = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(int(0.975 * n_boot), n_boot - 1)]
    return {"mean": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)]}


METHODS = ["R2_checker", "R4_ranker", "R5_comparator", "carbonate_free", "random", "oracle"]


def main():
    print("loading validator, comparator, ranker...", flush=True)
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
    rng = random.Random(SEED)

    by_target = {}
    for name in TARGETS:
        pool, target_formula = build_pool(name, validator, ranker)
        by_target[name] = {"pool": pool, "target_formula": target_formula}
        print(f"{name}: pool size = {len(pool)}", flush=True)

    curve = {method: {n: [] for n in N_VALUES} for method in METHODS}
    for name, info in by_target.items():
        pool, target_formula = info["pool"], info["target_formula"]
        if len(pool) < 2:
            print(f"  {name}: pool too small ({len(pool)}), skipping", flush=True)
            continue
        for n in N_VALUES:
            if n > len(pool):
                continue
            for _ in range(N_DRAWS):
                draw = rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]
                for method in METHODS:
                    pick = pick_best(draw, method, comparator, scales, target_formula, rng)
                    curve[method][n].append(pick["wt_pct"])
        print(f"  {name}: drew {N_DRAWS} x {len(N_VALUES)} x {len(METHODS)} selections", flush=True)

    curve_summary = {}
    for method in METHODS:
        curve_summary[method] = {}
        for n in N_VALUES:
            vals = curve[method][n]
            if not vals:
                curve_summary[method][n] = None
                continue
            curve_summary[method][n] = {"n_draws": len(vals), "kl_bound": round(kl_bound(n), 4),
                                        **bootstrap_mean_ci(vals, seed=SEED)}
        print(f"{method}: {curve_summary[method]}", flush=True)

    # Prediction: at every n, R2 and R4's CIs overlap carbonate-free's.
    overlap_rows = []
    for n in N_VALUES:
        cf = curve_summary["carbonate_free"].get(n)
        if cf is None:
            continue
        for method in ("R2_checker", "R4_ranker"):
            m = curve_summary[method].get(n)
            if m is None:
                continue
            overlaps = not (m["ci95"][1] < cf["ci95"][0] or m["ci95"][0] > cf["ci95"][1])
            overlap_rows.append({"n": n, "method": method, "overlaps_carbonate_free": overlaps})
    n_pass = sum(1 for r in overlap_rows if r["overlaps_carbonate_free"])

    # [E]: does any curve turn down at large n?
    turn_down = {}
    for method in METHODS:
        vals_by_n = [(n, curve_summary[method][n]["mean"]) for n in N_VALUES if curve_summary[method].get(n)]
        vals_by_n.sort()
        turns_down = any(vals_by_n[i + 1][1] < vals_by_n[i][1] for i in range(len(vals_by_n) - 1)) if len(vals_by_n) > 1 else None
        turn_down[method] = {"values_by_n": vals_by_n, "turns_down_anywhere": turns_down}

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-g (Goodhart curve: best-of-n as a free simulation of practice)",
        "seed": SEED, "n_draws_per_n": N_DRAWS, "n_values": N_VALUES, "targets": TARGETS,
        "pool_sizes": {name: len(info["pool"]) for name, info in by_target.items()},
        "r5_scalar_method_note": "R5 (comparator) has no native per-sample scalar; best-of-n here uses "
                                 "a Copeland score (count of pairwise wins via comparator_rule against "
                                 "the other n-1 draws in THIS specific sample, ties=0.5) -- a documented "
                                 "simplification, not Block G's rigorous Hodge-gradient scalar.",
        "curve": curve_summary,
        "prediction_check": {
            "statement": "At every n, the CIs of R2 and R4 overlap the carbonate-free rule's.",
            "rows": overlap_rows, "n_pass": n_pass, "n_total": len(overlap_rows),
            "passes": n_pass == len(overlap_rows) if overlap_rows else None,
        },
        "exploratory_turn_down_E": turn_down,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nprediction_check: {n_pass}/{len(overlap_rows)} (passes={report['prediction_check']['passes']})", flush=True)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
