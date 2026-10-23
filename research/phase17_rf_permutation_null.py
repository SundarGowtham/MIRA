#!/usr/bin/env python
"""
research/phase17_rf_permutation_null.py — follow-up correction to
results/phase17/rf_information_bound.json. The cluster-bootstrap CI
for I(R;Q|carbonate_free) excluded 0 for R2/R4/R5 on the YBCO primary
endpoint (0/3 pass, prediction FAILS as first computed) -- but
plug-in discrete mutual-information estimators are KNOWN to carry a
positive finite-sample bias (Paninski 2003): even a TRUE zero MI
reads as significantly positive under naive bootstrap CIs, because
MI>=0 always and sampling noise pushes the plug-in estimate up, not
down, when the truth is near the boundary. A positive point estimate
with a CI excluding 0 is not, by itself, evidence of real R-Q
association once this bias is accounted for.

CORRECTION: a permutation null. Shuffle Q (the actual winner label)
independently WITHIN each carbonate-free stratum (preserving the
carbonate-free split and each reward's own marginal exactly, breaking
only the R<->Q association) 2,000 times; recompute the conditional MI
each time. If the OBSERVED value sits inside this null's own range
(not just at/above 0), the apparent "CI excludes 0" result is
estimator bias, not signal -- consistent with finding 27's qualitative
"below chance / no better than baseline" conclusion. If the observed
value sits ABOVE the null's 95th percentile, that IS a real signal
the naive CI correctly flagged, and finding 27 would need revisiting.

Does NOT overwrite the original bootstrap-CI numbers -- adds a
`permutation_null_check` field per reward, additively, to the
existing JSON.

Usage (tmux, CPU only -- reuses already-built pairs, no new PD calls):
  PYTHONPATH=. uv run python research/phase17_rf_permutation_null.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import load_comparator  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency  # noqa: E402
from core.reward import load_validator  # noqa: E402
from research.ranker_capacity_rssft import SYNTHESIS  # noqa: E402
from research.phase17_rf_information_bound import (  # noqa: E402
    ybco_pairs_with_ranker, conditional_mutual_info, COMPARATOR_SCALES_PATH, OUT_PATH, SEED,
)
from validator import ThermoChecker  # noqa: E402

N_PERM = 2000


def permutation_null(pairs: list[dict], r_key: str, cf_key: str, seed: int, n_perm: int) -> list[float]:
    """Shuffle 'actual' WITHIN each cf stratum, n_perm times."""
    rng = random.Random(seed)
    by_cf: dict = {}
    for p in pairs:
        by_cf.setdefault(p.get(cf_key), []).append(p)
    nulls = []
    for _ in range(n_perm):
        shuffled = []
        for cf, subset in by_cf.items():
            actuals = [p["actual"] for p in subset]
            rng.shuffle(actuals)
            for p, a in zip(subset, actuals):
                shuffled.append({**p, "actual": a})
        nulls.append(conditional_mutual_info(shuffled, r_key, cf_key, "actual"))
    return nulls


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
    for reward_id, r_key in [("R2_checker", "gdpo_winner"), ("R4_ranker", "ranker_winner"),
                              ("R5_comparator", "comp_winner")]:
        observed = report["ybco_primary"][reward_id]["I(R;Q|carbonate_free)_bits"]["point_bits"]
        null_vals = permutation_null(pairs, r_key, "base_carbfree", SEED, N_PERM)
        null_vals.sort()
        p95 = null_vals[int(0.95 * len(null_vals))]
        mean_null = sum(null_vals) / len(null_vals)
        pct_null_above_or_eq_observed = sum(1 for v in null_vals if v >= observed) / len(null_vals)
        indistinguishable_from_null = observed <= p95
        report["ybco_primary"][reward_id]["permutation_null_check"] = {
            "n_perm": N_PERM,
            "observed_I(R;Q|CF)_bits": observed,
            "null_mean_bits": round(mean_null, 5),
            "null_p95_bits": round(p95, 5),
            "empirical_p_value": round(pct_null_above_or_eq_observed, 4),
            "indistinguishable_from_null": indistinguishable_from_null,
            "interpretation": ("The naive bootstrap CI's exclusion of 0 is consistent with known "
                              "plug-in-MI positive finite-sample bias, not a real R-Q association "
                              "beyond carbonate-free -- the observed value falls within the "
                              "permutation null's own range." if indistinguishable_from_null else
                              "The observed value exceeds the permutation null's 95th percentile -- "
                              "this IS a real signal beyond carbonate-free, not estimator bias, and "
                              "contradicts finding 27's qualitative 'no better than baseline' read "
                              "when restated in information-theoretic units."),
        }
        print(f"{reward_id}: observed={observed} null_p95={round(p95,5)} "
              f"indistinguishable={indistinguishable_from_null}", flush=True)

    n_indist = sum(1 for rid in ("R2_checker", "R4_ranker", "R5_comparator")
                   if report["ybco_primary"][rid]["permutation_null_check"]["indistinguishable_from_null"])
    report["prediction_check_corrected"] = {
        "statement": "Corrected prediction check: a reward 'passes' (CI finding is not an artifact "
                     "of plug-in-MI bias) if its I(R;Q|carbonate_free) is indistinguishable from the "
                     "permutation null, OR if the original bootstrap CI already included 0.",
        "n_indistinguishable_from_null": n_indist, "n_total": 3,
        "passes": n_indist == 3,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\ncorrected prediction check: {n_indist}/3 indistinguishable from null "
          f"(passes={report['prediction_check_corrected']['passes']})", flush=True)
    print(f"-> {OUT_PATH} (permutation_null_check added)")


if __name__ == "__main__":
    main()
