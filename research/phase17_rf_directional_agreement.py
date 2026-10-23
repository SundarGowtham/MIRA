#!/usr/bin/env python
"""
research/phase17_rf_directional_agreement.py — second follow-up correction
to results/phase17/rf_information_bound.json, per regular Claude's review:
mutual information is SIGN-BLIND. A reward that consistently picks the
WORSE recipe carries exactly as many bits as one that picks the better one
-- and finding 27 already showed R5 scores BELOW CHANCE (0.474) on these
exact YBCO pairs. The permutation-null correction (phase17_rf_permutation_null.py)
confirmed R2/R4/R5's I(R;Q|carbonate_free) is real signal, not estimator
bias -- but "real signal" is not yet "right-direction signal".

This script adds, per reward, per CARBONATE-FREE STRATUM (base_carbfree in
{-1, 0, 1}):
  - directional agreement rate (vote == actual, among decisive pairs where
    both vote != 0 and actual != 0), with a cluster-bootstrap CI (same
    cluster_bootstrap_ci reused from the frozen arrows_gate_scoring.py).
  - n_decisive pairs in that stratum.
  - an explicit anti_signal flag: True if the CI's upper bound is < 0.5
    (reward is SIGNIFICANTLY WORSE than chance in that stratum -- the
    bits it carries point the wrong way).

Also adds an unconditional permutation-null (shuffle `actual` globally,
not stratified) beside I(R;Q)_bits, matching the null-mean-beside-point
request for the unconditional quantity (the conditional one already has
this from phase17_rf_permutation_null.py).

Does NOT overwrite any existing field -- adds `directional_agreement` and
`unconditional_permutation_null` per reward, additively.

Usage (tmux, CPU only, reuses the same pair construction, no new PD calls):
  PYTHONPATH=. uv run python research/phase17_rf_directional_agreement.py
"""
from __future__ import annotations

import json
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
from research.distributional.arrows_gate_scoring import cluster_bootstrap_ci  # noqa: E402
from research.phase17_rf_information_bound import (  # noqa: E402
    ybco_pairs_with_ranker, mutual_info, COMPARATOR_SCALES_PATH, OUT_PATH, SEED,
)
from validator import ThermoChecker  # noqa: E402

N_PERM = 2000
N_BOOT = 10000


def directional_agreement(pairs: list[dict], r_key: str) -> dict:
    decisive = [p for p in pairs if p.get(r_key) not in (None, 0) and p.get("actual") not in (None, 0)]
    if not decisive:
        return {"n_decisive": 0, "agreement_rate": None, "ci95": [None, None], "anti_signal": None}
    rate_fn = lambda ps: (sum(1 for p in ps if p[r_key] == p["actual"]) / len(ps)) if ps else 0.5  # noqa: E731
    point = rate_fn(decisive)
    lo, hi = cluster_bootstrap_ci(decisive, rate_fn, random.Random(SEED), n_boot=N_BOOT)
    return {"n_decisive": len(decisive), "agreement_rate": round(point, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "anti_signal": hi < 0.5}


def by_stratum_agreement(pairs: list[dict], r_key: str) -> dict:
    by_cf = defaultdict(list)
    for p in pairs:
        cf = p.get("base_carbfree")
        if cf is None:
            continue
        by_cf[cf].append(p)
    out = {}
    for cf, subset in sorted(by_cf.items()):
        out[str(cf)] = {"n_pairs_in_stratum": len(subset), **directional_agreement(subset, r_key)}
    out["overall"] = directional_agreement(pairs, r_key)
    return out


def unconditional_permutation_null(pairs: list[dict], r_key: str, seed: int, n_perm: int) -> dict:
    rng = random.Random(seed)
    actuals = [p["actual"] for p in pairs]
    observed = mutual_info(pairs, r_key)
    nulls = []
    for _ in range(n_perm):
        shuffled_actuals = actuals[:]
        rng.shuffle(shuffled_actuals)
        shuffled = [{**p, "actual": a} for p, a in zip(pairs, shuffled_actuals)]
        nulls.append(mutual_info(shuffled, r_key))
    nulls.sort()
    p95 = nulls[int(0.95 * len(nulls))]
    mean_null = sum(nulls) / len(nulls)
    return {
        "observed_I(R;Q)_bits": round(observed, 5),
        "null_mean_bits": round(mean_null, 5),
        "null_p95_bits": round(p95, 5),
        "indistinguishable_from_null": observed <= p95,
    }


def main():
    rng = random.Random(SEED)
    print("rebuilding YBCO pairs (same construction as rf_information_bound.py)...", flush=True)
    validator = load_validator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                               REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    comparator = load_comparator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                                 REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    scales = json.loads(COMPARATOR_SCALES_PATH.read_text())["scales"]
    import pickle
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)

    pairs = ybco_pairs_with_ranker(validator, comparator, scales, ranker, rng)
    print(f"n_pairs={len(pairs)}", flush=True)

    report = json.loads(OUT_PATH.read_text())
    any_anti_signal = []
    for reward_id, r_key in [("R2_checker", "gdpo_winner"), ("R4_ranker", "ranker_winner"),
                              ("R5_comparator", "comp_winner")]:
        strat = by_stratum_agreement(pairs, r_key)
        report["ybco_primary"][reward_id]["directional_agreement_by_stratum"] = strat
        uncond_null = unconditional_permutation_null(pairs, r_key, SEED, N_PERM)
        report["ybco_primary"][reward_id]["I(R;Q)_bits"]["unconditional_permutation_null"] = uncond_null
        overall = strat["overall"]
        print(f"{reward_id}: overall agreement={overall['agreement_rate']} ci={overall['ci95']} "
              f"anti_signal={overall['anti_signal']}", flush=True)
        for cf_val, cell in strat.items():
            if cf_val == "overall":
                continue
            if cell.get("anti_signal"):
                any_anti_signal.append({"reward": reward_id, "stratum_base_carbfree": cf_val,
                                        "agreement_rate": cell["agreement_rate"], "ci95": cell["ci95"]})

    cf_baseline_strat = by_stratum_agreement(pairs, "base_carbfree")
    report["ybco_primary"]["carbonate_free_baseline"]["directional_agreement_by_stratum_note"] = (
        "Not meaningful: base_carbfree IS the stratifying variable here, so its own "
        "'agreement with itself' is definitionally 1.0 wherever it is decisive. "
        "Its unconditional agreement with `actual` (finding 27's 0.583) is already "
        "reported elsewhere (PHASE15_ARROWS_RESULTS.md) and is not recomputed here."
    )

    report["directional_signal_summary"] = {
        "statement": "Per regular Claude's review: MI is sign-blind, so a positive I(R;Q|carbonate-free) "
                     "does not by itself mean the reward points the RIGHT way within a carbonate stratum. "
                     "This reports agreement rate (vote==actual) per base_carbfree stratum, with CIs. "
                     "anti_signal=true means the CI's upper bound is below 0.5 -- the reward is "
                     "SIGNIFICANTLY WORSE than chance in that stratum, i.e. the information it carries "
                     "points the wrong way.",
        "any_anti_signal_strata": any_anti_signal,
        "interpretation": ("At least one (reward, stratum) cell is significantly anti-signal -- the "
                           "permutation-null correction's 'real signal' finding must be read together "
                           "with this: real does not mean right-direction." if any_anti_signal else
                           "No (reward, stratum) cell is significantly anti-signal (all CIs either "
                           "exclude 0.5 on the correct side or include it) -- the permutation-null "
                           "correction's 'real signal' plausibly does point the right way, though CIs "
                           "should still be read individually before leading with this."),
    }

    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nany_anti_signal_strata: {len(any_anti_signal)} cells flagged", flush=True)
    print(f"-> {OUT_PATH} (directional_agreement_by_stratum + unconditional_permutation_null added)")


if __name__ == "__main__":
    main()
