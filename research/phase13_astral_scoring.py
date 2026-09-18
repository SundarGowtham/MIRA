#!/usr/bin/env python
"""
research/phase13_astral_scoring.py — Phase 13 Steps 0 and 4
(misc/PHASE13_14_SPEC.md): does core/comparator.py prefer the ASTRAL
predicted-set routes over the conventional ones? Pre-registration:
misc/PHASE13_PREREG.md (locked before this script was run). CPU only, no
generation, no training.

Route construction mirrors research/ranker_v2_astral_gate.py (Phase 11
Step 2) exactly, for direct comparability: two PredictedRoute objects per
target (traditional / predicted precursors) at that route's own best_T,
air, single calcination, amount=1.0 for every precursor (ASTRAL supplies
species, not molar ratios -- finding 19's caveat, unchanged here).

STEP 0 NOTE: misc/astral_validation_set.json has exactly two named routes
per target (traditional, predicted) with measured purity on both sides --
there is no third or further route candidate in this dataset to enumerate.
"Full pair enumeration" therefore coincides with the primary 35 pairs; this
is a fact about the available data, not a shortcut taken here, and is
reported as such rather than silently treated as if a larger enumeration
had been run.

PRE-REGISTERED PRIMARY (misc/PHASE13_PREREG.md): N_pref = targets where
compare(predicted, traditional, target) margin > 0, arm B (C3 fraction
0.17). Bar: N_pref >= 25/35. Also reports arms A (0.12), C (0.22), D
(flat 100 K) per the required sensitivity sweep.

Usage (tmux):
  uv run python research/phase13_astral_scoring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import binomtest  # noqa: E402

from validator import (  # noqa: E402
    PredictedConditions, PredictedOperation, PredictedPrecursor, PredictedRoute,
)
from core.comparator import (  # noqa: E402
    CHANNEL_NAMES, COMPARATOR_VERSION, ComparatorParams, load_comparator,
)

DATA_PATH = Path("misc/astral_validation_set.json")
SCALES_PATH = Path("misc/comparator_scales_v1.json")
OUT_JSON = Path("results/phase13_astral_scoring.json")

PRIMARY_BAR = 25  # of 35, misc/PHASE13_PREREG.md
CHANCE = 17.5

ARMS = {
    "A_0.12": ComparatorParams(c3_fraction=0.12),
    "B_0.17_PREREGISTERED": ComparatorParams(c3_fraction=0.17),
    "C_0.22": ComparatorParams(c3_fraction=0.22),
    "D_flat_100K": ComparatorParams(c3_fraction=None),
}


def build_route(target: str, precursors: list[str], T: float) -> PredictedRoute:
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(f, 1.0) for f in precursors],
        operations=[PredictedOperation(
            type="calcine",
            conditions=PredictedConditions(
                heating_temperature=[float(T)], heating_atmosphere=["air"]))],
    )


def score_all_arms(targets, comparator, scales):
    """rows[i] = {..target meta.., "arms": {arm_name: {"margin":.., "breakdown":..}}}"""
    rows = []
    for m in targets:
        trad_route = build_route(m["target"], m["traditional"], m["trad_best_T"])
        pred_route = build_route(m["target"], m["predicted"], m["pred_best_T"])
        arm_results = {}
        for arm_name, params in ARMS.items():
            margin, bd = comparator.compare(
                pred_route, trad_route, m["target"], scales, params)
            arm_results[arm_name] = {"margin": margin, "breakdown": bd}
        rows.append({**m, "arms": arm_results})
    return rows


def primary_endpoint(rows, arm_name):
    n_pref, n_against, n_tie = 0, 0, 0
    pref_targets, against_targets = [], []
    for r in rows:
        margin = r["arms"][arm_name]["margin"]
        if margin > 1e-12:
            n_pref += 1
            pref_targets.append(r["target"])
        elif margin < -1e-12:
            n_against += 1
            against_targets.append(r["target"])
        else:
            n_tie += 1
    n = len(rows)
    # one-sided binomial test: is n_pref significantly above chance (17.5/35, p=0.5)?
    bt = binomtest(n_pref, n, 0.5, alternative="greater")
    return {
        "n_pref": n_pref, "n_against": n_against, "n_tie": n_tie, "n_total": n,
        "chance": CHANCE, "bar": PRIMARY_BAR,
        "clears_bar": n_pref >= PRIMARY_BAR,
        "binomial_p_one_sided": bt.pvalue,
        "pref_targets": pref_targets, "against_targets": against_targets,
    }


def sign_agreement_with_purity(rows, arm_name):
    """Secondary endpoint: does the comparator's margin sign agree with the
    sign of measured phase-purity difference (pred_purity - trad_purity)?
    Only the 35 primary pairs exist in this dataset (Step 0 note above)."""
    agree, disagree, tie = 0, 0, 0
    disagreements = []
    for r in rows:
        margin = r["arms"][arm_name]["margin"]
        dp = r["pred_best_purity"] - r["trad_best_purity"]
        if abs(dp) < 1e-9 or abs(margin) < 1e-12:
            tie += 1
            continue
        if (margin > 0) == (dp > 0):
            agree += 1
        else:
            disagree += 1
            disagreements.append(r["target"])
    n_decided = agree + disagree
    ci = binomtest(agree, n_decided, 0.5).proportion_ci() if n_decided else None
    return {
        "agree": agree, "disagree": disagree, "tie": tie, "n_decided": n_decided,
        "agreement_rate": agree / n_decided if n_decided else None,
        "ci_95": [ci.low, ci.high] if ci else None,
        "disagreeing_targets": disagreements,
    }


def per_channel_agreement(rows, arm_name):
    """For each of C1-C7: does its sign ALONE (raw diff sign, scale-
    invariant since MAD scales are all positive) agree with the sign of
    measured phase-purity difference, on gradeable pairs."""
    out = {}
    for c in CHANNEL_NAMES:
        agree, disagree, tie, ungradeable = 0, 0, 0, 0
        for r in rows:
            bd = r["arms"][arm_name]["breakdown"]
            diff = bd.get(f"{c}_diff")
            dp = r["pred_best_purity"] - r["trad_best_purity"]
            if diff is None:
                ungradeable += 1
                continue
            if abs(diff) < 1e-12 or abs(dp) < 1e-9:
                tie += 1
                continue
            ok = (diff > 0) == (dp > 0)
            if ok:
                agree += 1
            else:
                disagree += 1
        n_scored = agree + disagree
        out[c] = {
            "agree": agree, "disagree": disagree, "tie": tie,
            "ungradeable": ungradeable, "n_scored": n_scored,
            "agreement_rate": agree / n_scored if n_scored else None,
            "pct_gradeable_of_35": round(100 * (35 - ungradeable) / 35, 1),
        }
    return out


def channel_ablation(rows, scales, arm_name, params):
    """Drop one channel at a time from the aggregate margin, recount
    N_pref -- identifies which channels are load-bearing for the primary
    result (PHASE13_14_SPEC.md secondary endpoint 4)."""
    out = {}
    for dropped in CHANNEL_NAMES:
        ablated_scales = dict(scales)
        ablated_scales[dropped] = None  # forces this channel out of every pair
        n_pref = 0
        for r in rows:
            bd = r["arms"][arm_name]["breakdown"]
            margin = 0.0
            for c in CHANNEL_NAMES:
                if c == dropped:
                    continue
                diff_scaled = bd.get(f"{c}_diff_scaled")
                if diff_scaled is not None:
                    margin += diff_scaled
            if margin > 1e-12:
                n_pref += 1
        out[dropped] = {"n_pref_without_this_channel": n_pref,
                        "delta_vs_full": n_pref - primary_full_n_pref[0]}
    return out


# module-level mutable cell so channel_ablation can report delta vs the
# full-aggregate N_pref without threading an extra argument everywhere
primary_full_n_pref = [None]


def main():
    data = json.loads(DATA_PATH.read_text())
    targets = data["targets"]
    print(f"loaded {len(targets)} ASTRAL targets", flush=True)

    scales_doc = json.loads(SCALES_PATH.read_text())
    scales = scales_doc["scales"]
    print(f"loaded MAD scales ({SCALES_PATH}): {scales}", flush=True)

    print("loading comparator...", flush=True)
    comparator = load_comparator(
        Path("data/cache/mp_formula_set.pkl"),
        Path("data/cache/pd_index.json"),
        Path("."),
    )
    assert comparator.thermo is not None

    print("scoring 35 targets x 2 routes x 4 C3 arms...", flush=True)
    rows = score_all_arms(targets, comparator, scales)

    PRIMARY_ARM = "B_0.17_PREREGISTERED"
    primary = primary_endpoint(rows, PRIMARY_ARM)
    primary_full_n_pref[0] = primary["n_pref"]

    sweep = {arm: primary_endpoint(rows, arm) for arm in ARMS}
    sign_agree = sign_agreement_with_purity(rows, PRIMARY_ARM)
    per_channel = per_channel_agreement(rows, PRIMARY_ARM)
    ablation = channel_ablation(rows, scales, PRIMARY_ARM, ARMS[PRIMARY_ARM])

    verdict = "CLEARS BAR -- Phase 14 becomes a real prospect" if primary["clears_bar"] \
        else "DOES NOT CLEAR BAR -- STOP, per pre-registration"

    print("\n" + "=" * 100)
    print(f"{'target':<16}{'margin(B)':>12}{'trad_pur':>10}{'pred_pur':>10}{'pref?':>8}")
    for r in rows:
        m = r["arms"][PRIMARY_ARM]["margin"]
        tag = "PRED" if m > 1e-12 else ("TRAD" if m < -1e-12 else "tie")
        print(f"{r['target']:<16}{m:>12.4f}{r['trad_best_purity']:>10.2f}"
              f"{r['pred_best_purity']:>10.2f}{tag:>8}")

    print("\n" + "=" * 100)
    print(f"PRIMARY ENDPOINT (arm B, fraction=0.17): "
          f"N_pref = {primary['n_pref']}/{primary['n_total']} "
          f"(against={primary['n_against']}, tie={primary['n_tie']})")
    print(f"  chance={CHANCE}  bar={PRIMARY_BAR}  "
          f"one-sided binomial p={primary['binomial_p_one_sided']:.4g}")
    print(f"  *** VERDICT: {verdict} ***")

    print("\n--- C3 sensitivity sweep (all 4 arms) ---")
    for arm, res in sweep.items():
        print(f"  {arm:<24} N_pref={res['n_pref']}/35  clears_bar={res['clears_bar']}")

    print("\n--- sign agreement with measured phase purity (arm B) ---")
    print(f"  agree={sign_agree['agree']} disagree={sign_agree['disagree']} "
          f"tie={sign_agree['tie']} rate={sign_agree['agreement_rate']}")
    if sign_agree["disagreeing_targets"]:
        print(f"  disagreements: {sign_agree['disagreeing_targets']}")

    print("\n--- per-channel agreement with measured phase purity (arm B) ---")
    for c, d in per_channel.items():
        print(f"  {c:<32} agree={d['agree']:>3} disagree={d['disagree']:>3} "
              f"tie={d['tie']:>3} ungradeable={d['ungradeable']:>3} "
              f"({d['pct_gradeable_of_35']}% of 35)  rate={d['agreement_rate']}")

    print("\n--- channel ablation (drop one channel, recount N_pref, arm B) ---")
    print(f"  FULL aggregate N_pref = {primary['n_pref']}")
    for c, d in ablation.items():
        print(f"  drop {c:<32} N_pref={d['n_pref_without_this_channel']:>3} "
              f"(delta={d['delta_vs_full']:+d})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "comparator_version": COMPARATOR_VERSION,
        "scales_used": scales,
        "primary_arm": PRIMARY_ARM,
        "primary": primary,
        "verdict": verdict,
        "c3_sensitivity_sweep": sweep,
        "sign_agreement_with_purity": sign_agree,
        "per_channel_agreement": per_channel,
        "channel_ablation": ablation,
        "rows": rows,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
