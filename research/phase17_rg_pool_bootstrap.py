#!/usr/bin/env python
"""
research/phase17_rg_pool_bootstrap.py — follow-up correction to
results/phase17/rg_goodhart_curve.json per regular Claude's review of R-g:

1. (a) The original CIs re-drew 2,000 best-of-n SAMPLES from a FIXED pool.
   That measures Monte-Carlo draw noise, which shrinks with more draws --
   it is not the real uncertainty, which comes from which recipes are in
   the pool to begin with (pool sizes are small: YBCO=200, LTOPO=58,
   NTMO=18). This script adds a CLUSTER BOOTSTRAP OVER THE POOL: resample
   each target's distinct precursor-SET labels with replacement (same
   clustering convention as cluster_bootstrap_ci in the frozen
   arrows_gate_scoring.py -- a precursor-set can appear at >1 temperature,
   and those entries must resample together), reconstruct the weighted
   pool, THEN redraw best-of-n from that resampled pool. Repeated many
   times for a CI that reflects pool uncertainty, not just draw noise.

   KEY OPTIMISATION: the original script called comparator_rule() fresh
   for every pairwise comparison inside every one of 2,000 draws (this is
   almost certainly why R-g took 3h22m instead of the ~1.5-2.5h estimate
   -- LTOPO/NTMO's chemistry apparently hits a slow PD-lookup path).
   comparator_rule's result for a given (pool_i, pool_j) pair never
   changes, so this script precomputes the FULL pairwise win-matrix per
   target ONCE (O(pool_size^2) calls -- 19,900/1,653/153 for YBCO/LTOPO/
   NTMO, all cheap) and looks it up thereafter. This makes the nested
   pool-bootstrap computationally tractable.

2. (b) [E, post-hoc] A within-temperature version: for each target,
   restrict the pool to one measured temperature at a time before
   drawing, so no reward can select on temperature (the checker's ΔG is
   evaluated at the reported T; the ranker has a temperature channel --
   either could "find" high yield by picking hot entries, not good
   chemistry).

3. (c) [E, post-hoc] Per-target curves (not only pooled across YBCO/
   LTOPO/NTMO), and the share of exact ties for carbonate_free at each n
   (broken by random tiebreak) -- to explain whether it sits flat because
   most of the pool is carbonate-containing (so "avoid carbonate" rarely
   discriminates within a draw) or some other reason.

4. (d) R2's n=16 dip (21.5->19.8%, the original "Goodhart" candidate) is
   NOT labelled Goodhart here unless it survives the pool-bootstrap CI --
   i.e. the pool-bootstrap CI at n=16 must sit clearly below the n=8 CI.

Does NOT overwrite results/phase17/rg_goodhart_curve.json or the original
draw-noise curve numbers -- writes a NEW file, cross-referencing it.

Usage (tmux, CPU only, reuses the frozen arrows_gate_scoring.py + the
original rg script's pool-building/scalar functions by import; no edits
to either):
  PYTHONPATH=. uv run python research/phase17_rg_pool_bootstrap.py
"""
from __future__ import annotations

import json
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
from research.distributional.arrows_gate_scoring import (  # noqa: E402
    load_target_entries, build_route, score_validator, comparator_rule, COMPARATOR_SCALES_PATH,
)
from research.phase17_rg_goodhart_curve import (  # noqa: E402
    checker_scalar, carbonate_free_scalar, kl_bound, SEED, N_DRAWS, N_VALUES, TARGETS,
    OUT_PATH as RG_ORIGINAL_PATH,
)
from validator import ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rg_pool_bootstrap.json"
N_BOOT_POOL = 1000      # outer: resample the pool
N_DRAWS_INNER = 300     # inner: best-of-n draws per bootstrap replicate
N_DRAWS_FRESH = 2000    # matches the original script's per-(target,n) draw count
METHODS = ["R2_checker", "R4_ranker", "R5_comparator", "carbonate_free", "random", "oracle"]


def build_pool_indexed(name: str, validator, ranker) -> tuple[list[dict], str]:
    """Like the original build_pool, but keeps a stable index and the
    precursor_set label per entry for cluster resampling."""
    entries, target_formula = load_target_entries(name)
    pool = []
    for idx, e in enumerate(entries):
        route = build_route(target_formula, e["precursors"], e["amounts"], e["temp_C"], e["atmosphere"])
        bd = score_validator(validator, route, target_formula)
        score_r4, _ = ranker.score(route, target_formula)
        pool.append({
            "idx": idx, "precursor_set": e["precursor_set"], "temp": e["temp_str"], "wt_pct": e["wt_pct"],
            "route": route, "bd": bd, "r2_scalar": checker_scalar(bd), "r4_scalar": score_r4,
            "cf_scalar": carbonate_free_scalar(route),
        })
    return pool, target_formula


def precompute_win_matrix(pool: list[dict], comparator, scales, target_formula: str) -> dict:
    win = {}
    n = len(pool)
    for i in range(n):
        for j in range(i + 1, n):
            w, _ = comparator_rule(comparator, scales, pool[i]["route"], pool[j]["route"], target_formula)
            win[(i, j)] = w
    return win


def copeland_pick(indices: list[int], pool_by_idx: dict, win: dict, rng: random.Random) -> dict:
    wins = {i: 0.0 for i in indices}
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            if i == j:
                wins[i] += 0.5
                wins[j] += 0.5
                continue
            lo, hi = (i, j) if i < j else (j, i)
            w = win.get((lo, hi), 0)
            if w == 1:
                wins[lo] += 1.0
            elif w == -1:
                wins[hi] += 1.0
            else:
                wins[i] += 0.5
                wins[j] += 0.5
    best = max(wins.values())
    ties = [i for i in indices if wins[i] == best]
    return pool_by_idx[rng.choice(ties)]


def pick_best_fast(draw_entries: list[dict], method: str, win: dict, rng: random.Random) -> dict:
    if method == "random":
        return rng.choice(draw_entries)
    if method == "oracle":
        return max(draw_entries, key=lambda d: d["wt_pct"])
    if method in ("R2_checker", "R4_ranker", "carbonate_free"):
        key = {"R2_checker": "r2_scalar", "R4_ranker": "r4_scalar", "carbonate_free": "cf_scalar"}[method]
        scored = [(d, d[key]) for d in draw_entries if d[key] is not None]
        if not scored:
            return rng.choice(draw_entries)
        best_val = max(v for _, v in scored)
        ties = [d for d, v in scored if abs(v - best_val) < 1e-9]
        return rng.choice(ties)
    if method == "R5_comparator":
        pool_by_idx = {d["idx"]: d for d in draw_entries}
        indices = [d["idx"] for d in draw_entries]
        return copeland_pick(indices, pool_by_idx, win, rng)
    raise ValueError(method)


def tie_count(draw_entries: list[dict], key: str) -> bool:
    vals = [d[key] for d in draw_entries if d.get(key) is not None]
    return len(vals) > 1 and len(set(vals)) == 1


def bootstrap_mean_ci_from_vals(vals: list[float]) -> dict:
    if not vals:
        return {"mean": None, "ci95": [None, None], "n": 0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = sum(vals_sorted) / n
    lo = vals_sorted[int(0.025 * n)]
    hi = vals_sorted[min(int(0.975 * n), n - 1)]
    return {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": n}


def fresh_curve_per_target(pool: list[dict], win: dict, n_values: list[int], seed: int) -> dict:
    """Reproduces the ORIGINAL per-(target,n) 2,000-draw curve, but using
    the precomputed win-matrix (fast), and keeps it UN-pooled (per target)
    plus records carbonate_free tie shares."""
    rng = random.Random(seed)
    out = {}
    for n in n_values:
        if n > len(pool):
            continue
        picks = {m: [] for m in METHODS}
        cf_ties = 0
        for _ in range(N_DRAWS_FRESH):
            draw = rng.sample(pool, n)
            for m in METHODS:
                pick = pick_best_fast(draw, m, win, rng)
                picks[m].append(pick["wt_pct"])
            if tie_count(draw, "cf_scalar"):
                cf_ties += 1
        out[n] = {}
        for m in METHODS:
            out[n][m] = {"n_draws": len(picks[m]), "mean": round(sum(picks[m]) / len(picks[m]), 4)}
        out[n]["carbonate_free_tie_share"] = round(cf_ties / N_DRAWS_FRESH, 4)
    return out


def pool_bootstrap_curve(pool: list[dict], win: dict, n_values: list[int], seed: int) -> dict:
    """Cluster-bootstrap over the pool (resample precursor_set labels with
    replacement), then redraw best-of-n from the reconstructed pool."""
    rng = random.Random(seed)
    by_set = defaultdict(list)
    for d in pool:
        by_set[d["precursor_set"]].append(d)
    set_labels = sorted(by_set.keys())
    n_sets = len(set_labels)

    boot_means = {n: {m: [] for m in METHODS} for n in n_values}
    for _ in range(N_BOOT_POOL):
        drawn_labels = [set_labels[rng.randrange(n_sets)] for _ in range(n_sets)]
        resampled_pool = [d for label in drawn_labels for d in by_set[label]]
        if len(resampled_pool) < 2:
            continue
        for n in n_values:
            if n > len(resampled_pool):
                continue
            vals = {m: [] for m in METHODS}
            for _ in range(N_DRAWS_INNER):
                draw = rng.sample(resampled_pool, n) if n <= len(resampled_pool) else [rng.choice(resampled_pool) for _ in range(n)]
                for m in METHODS:
                    pick = pick_best_fast(draw, m, win, rng)
                    vals[m].append(pick["wt_pct"])
            for m in METHODS:
                boot_means[n][m].append(sum(vals[m]) / len(vals[m]))

    out = {}
    for n in n_values:
        out[n] = {}
        for m in METHODS:
            out[n][m] = bootstrap_mean_ci_from_vals(boot_means[n][m])
    return out


def within_temperature_curve(pool: list[dict], comparator, scales, target_formula: str, n_values: list[int], seed: int) -> dict:
    """[E, post-hoc]: restrict to one measured temperature at a time."""
    rng = random.Random(seed)
    by_temp = defaultdict(list)
    for d in pool:
        by_temp[d["temp"]].append(d)
    out = {}
    for temp, sub in by_temp.items():
        if len(sub) < 2:
            continue
        sub_reindexed = [{**d, "idx": i} for i, d in enumerate(sub)]
        win = precompute_win_matrix(sub_reindexed, comparator, scales, target_formula)
        temp_out = {}
        for n in n_values:
            if n > len(sub_reindexed):
                continue
            picks = {m: [] for m in METHODS}
            for _ in range(N_DRAWS_INNER):
                draw = rng.sample(sub_reindexed, n)
                for m in METHODS:
                    pick = pick_best_fast(draw, m, win, rng)
                    picks[m].append(pick["wt_pct"])
            temp_out[n] = {m: {"n_draws": len(picks[m]), "mean": round(sum(picks[m]) / len(picks[m]), 4)}
                           for m in METHODS}
        out[temp] = {"pool_size": len(sub_reindexed), "curve": temp_out}
    return out


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

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-g follow-up (pool bootstrap, within-temperature control, per-target curves)",
        "cross_reference": str(RG_ORIGINAL_PATH.relative_to(REPO_ROOT)),
        "seed": SEED,
        "deviation_note": f"Nominal convention is 10,000 bootstrap resamples; this uses a NESTED "
                          f"design (resample the pool, THEN redraw best-of-n from it), which is "
                          f"combinatorially heavier than a flat resample. Used N_BOOT_POOL={N_BOOT_POOL} "
                          f"outer x N_DRAWS_INNER={N_DRAWS_INNER} inner draws per (target,n,method) -- "
                          f"a documented reduction from 10,000, made tractable chiefly by precomputing "
                          f"each target's full pairwise comparator win-matrix ONCE instead of per-draw.",
        "per_target": {},
    }

    for name in TARGETS:
        print(f"\n=== {name} ===", flush=True)
        pool, target_formula = build_pool_indexed(name, validator, ranker)
        print(f"  pool size = {len(pool)}", flush=True)
        n_values_here = [n for n in N_VALUES if n <= len(pool)]
        cf_share = sum(d["cf_scalar"] for d in pool) / len(pool)
        print(f"  carbonate-free share of pool = {cf_share:.3f}", flush=True)

        win = precompute_win_matrix(pool, comparator, scales, target_formula)
        print(f"  win-matrix precomputed ({len(win)} pairs)", flush=True)

        fresh = fresh_curve_per_target(pool, win, n_values_here, SEED)
        print(f"  fresh per-target curve done", flush=True)

        pool_boot = pool_bootstrap_curve(pool, win, n_values_here, SEED)
        print(f"  pool-bootstrap curve done", flush=True)

        within_temp = within_temperature_curve(pool, comparator, scales, target_formula, n_values_here, SEED)
        print(f"  within-temperature curve done ({len(within_temp)} temps)", flush=True)

        report["per_target"][name] = {
            "pool_size": len(pool),
            "carbonate_free_pool_share": round(cf_share, 4),
            "fresh_curve_per_target_c": fresh,
            "pool_bootstrap_ci_a": pool_boot,
            "within_temperature_E_b": within_temp,
        }

    # (d) Goodhart claim check for R2 at n=16, pooled pool-bootstrap CIs across targets
    # (weight each target's bootstrap means equally per replicate is not directly poolable
    # post-hoc without re-running jointly; report per-target n=8 vs n=16 CI comparison instead,
    # and a pooled-mean-of-per-target-CIs as a conservative approximation.)
    goodhart_rows = []
    for name, info in report["per_target"].items():
        pb = info["pool_bootstrap_ci_a"]
        if 8 in pb and 16 in pb and "R2_checker" in pb[8] and "R2_checker" in pb[16]:
            r2_8, r2_16 = pb[8]["R2_checker"], pb[16]["R2_checker"]
            if r2_8["mean"] is not None and r2_16["mean"] is not None:
                clearly_below = r2_16["ci95"][1] < r2_8["ci95"][0]
                goodhart_rows.append({"target": name, "n8": r2_8, "n16": r2_16,
                                      "n16_clearly_below_n8": clearly_below})
    any_survives = any(r["n16_clearly_below_n8"] for r in goodhart_rows)
    report["r2_n16_dip_goodhart_check_d"] = {
        "statement": "R2_checker's original 21.5%->19.8% (n=8->n=16) dip is called 'Goodhart' only if "
                     "the pool-bootstrap CI at n=16 sits clearly below (no overlap with) the CI at n=8, "
                     "per target.",
        "rows": goodhart_rows,
        "any_target_confirms_goodhart": any_survives,
        "verdict": ("At least one target's pool-bootstrap CI confirms the dip survives pool uncertainty "
                    "-- 'Goodhart' is a defensible label." if any_survives else
                    "No target's pool-bootstrap CI confirms the dip survives pool uncertainty -- the "
                    "original point-estimate dip is likely noise; do NOT call it Goodhart."),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\ngoodhart_check (d): any_target_confirms={any_survives}", flush=True)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
