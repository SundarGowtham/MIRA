#!/usr/bin/env python
"""
research/phase13_astral_scoring_v2.py — Phase 13 iteration 2 scoring
(misc/PHASE13_PREREG.md addendum 2, 2026-09-18). Supersedes
research/phase13_astral_scoring.py (kept as the iteration-1 record,
misc/PHASE13_RESULTS.md).

Changes from iteration 1, all locked in the addendum before this script
was written:
  - Loads misc/comparator_scales_v2.json (rank-transform reference
    distributions), not v1's MAD scales.
  - PRIMARY endpoint is now sign agreement with measured phase purity,
    reported descriptively (point estimate + 95% CI), not a binary bar --
    a power calculation showed no bar at n<=35 (the real ceiling; ASTRAL's
    own screen has 224 reactions but only 35 pairs exist anywhere in this
    repo) is both achievable and meaningful.
  - N_pref is now a SECONDARY endpoint, always reported with the label
    confound stated (every traditional route here is 3-precursor, every
    predicted route 2-precursor, so C4/C7 reproduce ASTRAL's principle 1
    by construction).
  - Uses the balance-solver-fixed comparator (core.comparator installs
    _ComparatorValidator internally -- no change needed in this script).
  - C1 stays diagnostic-only (core.comparator.DIAGNOSTIC_CHANNEL_NAMES);
    reported in the per-channel table but never in the scored aggregate
    or the ablation.

Route construction is unchanged from iteration 1 (mirrors
research/ranker_v2_astral_gate.py): two PredictedRoute objects per target,
that route's own best_T, air, single calcination, amount=1.0 (ASTRAL
supplies species not molar ratios -- finding 19's caveat).

Usage (tmux):
  uv run python research/phase13_astral_scoring_v2.py
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
    CHANNEL_NAMES, COMPARATOR_VERSION, LABEL_CONFOUNDED_CHANNELS,
    SCORED_CHANNEL_NAMES, ComparatorParams, load_comparator,
)

DATA_PATH = Path("misc/astral_validation_set.json")
SCALES_PATH = Path("misc/comparator_scales_v2.json")
OUT_JSON = Path("results/phase13_astral_scoring_v2.json")

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


def sign_agreement_with_purity(rows, arm_name):
    """PRIMARY (iteration 2): does the comparator's margin sign agree with
    the sign of measured phase-purity difference? Descriptive only -- no
    binary bar (see module docstring)."""
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


def n_pref_secondary(rows, arm_name):
    """SECONDARY (demoted from iteration 1's primary): N_pref, always
    reported with the label confound stated (module docstring)."""
    n_pref, n_against, n_tie = 0, 0, 0
    for r in rows:
        margin = r["arms"][arm_name]["margin"]
        if margin > 1e-12:
            n_pref += 1
        elif margin < -1e-12:
            n_against += 1
        else:
            n_tie += 1
    n = len(rows)
    bt = binomtest(n_pref, n, 0.5, alternative="greater")
    return {
        "n_pref": n_pref, "n_against": n_against, "n_tie": n_tie, "n_total": n,
        "chance": CHANCE,
        "binomial_p_one_sided": bt.pvalue,
        "LABEL_CONFOUND_WARNING": (
            "Every traditional route in this dataset has exactly 3 "
            "precursors and every predicted route exactly 2 -- "
            "C4_interface_count and C7_gas_evolution reproduce this label "
            "by construction (see LABEL_CONFOUNDED_CHANNELS). This number "
            "is not independent physics evidence to the extent it is "
            "driven by either channel -- see channel_ablation."
        ),
    }


def per_channel_agreement(rows, arm_name):
    """For each of C1-C7 (including diagnostic-only C1): does its sign
    ALONE agree with the sign of measured phase-purity difference, on
    gradeable pairs."""
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
            "diagnostic_only": c not in SCORED_CHANNEL_NAMES,
            "label_confounded": c in LABEL_CONFOUNDED_CHANNELS,
        }
    return out


def channel_ablation(rows, arm_name):
    """Drop one SCORED channel at a time from the aggregate margin
    (using the already-computed rank-transformed diffs in the breakdown),
    recount N_pref and sign agreement."""
    out = {}
    full_n_pref = n_pref_secondary(rows, arm_name)["n_pref"]
    for dropped in SCORED_CHANNEL_NAMES:
        n_pref = 0
        agree, disagree = 0, 0
        for r in rows:
            bd = r["arms"][arm_name]["breakdown"]
            margin = 0.0
            for c in SCORED_CHANNEL_NAMES:
                if c == dropped:
                    continue
                diff_scaled = bd.get(f"{c}_diff_scaled")
                if diff_scaled is not None:
                    margin += diff_scaled
            if margin > 1e-12:
                n_pref += 1
            dp = r["pred_best_purity"] - r["trad_best_purity"]
            if abs(dp) >= 1e-9 and abs(margin) >= 1e-12:
                if (margin > 0) == (dp > 0):
                    agree += 1
                else:
                    disagree += 1
        n_decided = agree + disagree
        out[dropped] = {
            "n_pref_without_this_channel": n_pref,
            "delta_n_pref_vs_full": n_pref - full_n_pref,
            "sign_agreement_without_this_channel": agree / n_decided if n_decided else None,
            "label_confounded": dropped in LABEL_CONFOUNDED_CHANNELS,
        }
    return out


def main():
    data = json.loads(DATA_PATH.read_text())
    targets = data["targets"]
    print(f"loaded {len(targets)} ASTRAL targets", flush=True)

    scales_doc = json.loads(SCALES_PATH.read_text())
    scales = scales_doc["scales"]
    print(f"loaded rank-transform scales ({SCALES_PATH}), "
          f"n per channel: { {c: (len(v) if v else 0) for c, v in scales.items()} }",
          flush=True)

    print("loading comparator (balance-solver fix installed)...", flush=True)
    comparator = load_comparator(
        Path("data/cache/mp_formula_set.pkl"),
        Path("data/cache/pd_index.json"),
        Path("."),
    )
    assert comparator.thermo is not None

    print("scoring 35 targets x 2 routes x 4 C3 arms...", flush=True)
    rows = score_all_arms(targets, comparator, scales)

    PRIMARY_ARM = "B_0.17_PREREGISTERED"
    sign_agree = sign_agreement_with_purity(rows, PRIMARY_ARM)
    n_pref = n_pref_secondary(rows, PRIMARY_ARM)
    sweep = {arm: sign_agreement_with_purity(rows, arm) for arm in ARMS}
    per_channel = per_channel_agreement(rows, PRIMARY_ARM)
    ablation = channel_ablation(rows, PRIMARY_ARM)

    print("\n" + "=" * 100)
    print(f"{'target':<16}{'margin(B)':>12}{'trad_pur':>10}{'pred_pur':>10}{'pref?':>8}")
    for r in rows:
        m = r["arms"][PRIMARY_ARM]["margin"]
        tag = "PRED" if m > 1e-12 else ("TRAD" if m < -1e-12 else "tie")
        print(f"{r['target']:<16}{m:>12.4f}{r['trad_best_purity']:>10.2f}"
              f"{r['pred_best_purity']:>10.2f}{tag:>8}")

    print("\n" + "=" * 100)
    print("PRIMARY ENDPOINT (iteration 2): sign agreement with measured phase purity")
    print(f"  arm B: agree={sign_agree['agree']} disagree={sign_agree['disagree']} "
          f"tie={sign_agree['tie']} n_decided={sign_agree['n_decided']}")
    print(f"  agreement rate = {sign_agree['agreement_rate']:.1%}  "
          f"95% CI = [{sign_agree['ci_95'][0]:.1%}, {sign_agree['ci_95'][1]:.1%}]")
    if sign_agree["disagreeing_targets"]:
        print(f"  disagreements: {sign_agree['disagreeing_targets']}")

    print("\n--- C3 sensitivity sweep on the primary (sign agreement, all 4 arms) ---")
    for arm, res in sweep.items():
        ci = res["ci_95"]
        print(f"  {arm:<24} agreement={res['agreement_rate']:.1%}  "
              f"CI=[{ci[0]:.1%},{ci[1]:.1%}]  n_decided={res['n_decided']}")

    print("\n" + "=" * 100)
    print("SECONDARY ENDPOINT: N_pref -- LABEL-CONFOUNDED, see warning")
    print(f"  N_pref = {n_pref['n_pref']}/{n_pref['n_total']} "
          f"(against={n_pref['n_against']}, tie={n_pref['n_tie']})")
    print(f"  {n_pref['LABEL_CONFOUND_WARNING']}")

    print("\n--- per-channel sign agreement with measured phase purity (arm B) ---")
    for c, d in per_channel.items():
        tags = []
        if d["diagnostic_only"]:
            tags.append("DIAGNOSTIC-ONLY")
        if d["label_confounded"]:
            tags.append("LABEL-CONFOUNDED")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"  {c:<32}{tag_str:<28} agree={d['agree']:>3} disagree={d['disagree']:>3} "
              f"tie={d['tie']:>3} ({d['pct_gradeable_of_35']}% of 35)  rate={d['agreement_rate']}")

    print("\n--- channel ablation (drop one SCORED channel, arm B) ---")
    print(f"  FULL aggregate: N_pref={n_pref['n_pref']}  "
          f"sign_agreement={sign_agree['agreement_rate']:.1%}")
    for c, d in ablation.items():
        conf = " [LABEL-CONFOUNDED]" if d["label_confounded"] else ""
        sa = d["sign_agreement_without_this_channel"]
        print(f"  drop {c:<32}{conf:<22} N_pref={d['n_pref_without_this_channel']:>3} "
              f"(delta={d['delta_n_pref_vs_full']:+d})  "
              f"sign_agreement_without={sa:.1%}" if sa is not None else
              f"  drop {c:<32}{conf:<22} N_pref={d['n_pref_without_this_channel']:>3} "
              f"(delta={d['delta_n_pref_vs_full']:+d})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "comparator_version": COMPARATOR_VERSION,
        "iteration": 2,
        "primary_arm": PRIMARY_ARM,
        "primary_sign_agreement": sign_agree,
        "primary_sign_agreement_c3_sweep": sweep,
        "secondary_n_pref": n_pref,
        "per_channel_agreement": per_channel,
        "channel_ablation": ablation,
        "rows": rows,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
