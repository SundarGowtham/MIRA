#!/usr/bin/env python
"""
research/distributional/arrows_task4d_diagnostics.py — Phase 15 Task 4d
diagnostics [E], appended to docs/phases/PHASE15_ARROWS_RESULTS.md.
Reuses arrows_gate_scoring.py's route construction and rule
implementations exactly (imported, not reimplemented) so results are
directly comparable to the pre-registered Task 4c numbers.

1. Rebuilt lowest threshold: |Delta target wt%| > 0 STRICT, excluding
   pairs with IDENTICAL target wt% (including both-zero) -- those have
   no correct answer to agree or disagree with.
2. Thermo-only same-pairs analysis (YBCO, primary threshold >=5):
   (a) thermo-only rule vs best baseline, cluster-bootstrap CI.
   (b) thermo's decisive pairs only: n, thermo/carbonate-free/fewer-
       precursors agreement on the SAME pairs, and fraction where
       thermo's pick == carbonate-free's pick.
   (c) thermo agreement restricted to pairs where carbonate-free ties
       (both routes same carbonate status) -- does thermo carry signal
       beyond the carbonate dimension?
3. Comparator agreement restricted to the 1,660 channel-decided pairs
   (gate-decided and both-gate-fail pairs excluded).

Usage (tmux, real PD cache):
  uv run python research/distributional/arrows_task4d_diagnostics.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validator import SynthesisValidator  # noqa: E402
from core.reward import RUN3_CHECKS, load_validator  # noqa: E402
from core.comparator import ComparatorParams, load_comparator  # noqa: E402

# Reuse arrows_gate_scoring.py's building blocks directly.
import arrows_gate_scoring as base  # noqa: E402

OUT_JSON = Path("results/external/arrows_task4d_diagnostics.json")
SEED = 20260924
N_BOOT = 10000


def build_all(name, validator, comparator, scales):
    entries, target_formula = base.load_target_entries(name)
    routes, val_bd, comp_gates = {}, {}, {}
    for e in entries:
        key = (e["precursor_set"], e["temp_str"])
        route = base.build_route(target_formula, e["precursors"], e["amounts"], e["temp_C"], e["atmosphere"])
        routes[key] = route
        val_bd[key] = base.score_validator(validator, route, target_formula)
        comp_gates[key] = comparator._gater._check_gates(route)
    from collections import defaultdict
    by_temp = defaultdict(list)
    for e in entries:
        by_temp[e["temp_str"]].append(e)
    return entries, target_formula, routes, val_bd, comp_gates, by_temp


def enumerate_pairs(by_temp):
    pairs = []
    for temp_str, es in by_temp.items():
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                pairs.append((es[i], es[j]))
    return pairs


def thermo_only_vote(bd_a, bd_b):
    """Single-channel vote using ONLY thermodynamic_favorable, same
    gate-failure handling as the full GDPO-vote rule."""
    if bd_a is None and bd_b is None:
        return None
    if bd_a is None:
        return -1
    if bd_b is None:
        return 1
    ch = "thermodynamic_favorable"
    a_ungrad = bd_a.get(f"{ch}_gradeability") in SynthesisValidator.SENTINEL_TAGS
    b_ungrad = bd_b.get(f"{ch}_gradeability") in SynthesisValidator.SENTINEL_TAGS
    va, vb = bd_a.get(ch), bd_b.get(ch)
    if a_ungrad or b_ungrad or va is None or vb is None:
        return 0  # ungradeable -> treated as tie (0.5), per instruction
    diff = va - vb
    return 1 if diff > 1e-9 else (-1 if diff < -1e-9 else 0)


def main():
    rng = random.Random(SEED)
    print("loading validator and comparator...", flush=True)
    validator = load_validator(Path("data/cache/mp_formula_set.pkl"), Path("data/cache/pd_index.json"), Path("."))
    comparator = load_comparator(Path("data/cache/mp_formula_set.pkl"), Path("data/cache/pd_index.json"), Path("."))
    scales = json.loads(base.COMPARATOR_SCALES_PATH.read_text())["scales"]

    results = {}

    # ================= 1. Rebuilt threshold table (strict >0) =================
    print("\n" + "=" * 90 + "\n1. REBUILT LOWEST THRESHOLD: |delta| > 0 STRICT\n" + "=" * 90)
    rebuilt = {}
    for name in ["YBCO", "LTOPO", "NTMO"]:
        entries, target_formula, routes, val_bd, comp_gates, by_temp = build_all(name, validator, comparator, scales)
        all_pairs_raw = enumerate_pairs(by_temp)
        n_total = len(all_pairs_raw)
        n_identical = sum(1 for ea, eb in all_pairs_raw if abs(ea["wt_pct"] - eb["wt_pct"]) < 1e-9)
        n_strict = n_total - n_identical

        scored = []
        for ea, eb in all_pairs_raw:
            d = ea["wt_pct"] - eb["wt_pct"]
            if abs(d) < 1e-9:
                continue  # excluded: identical outcome, no correct answer
            actual = 1 if d > 0 else -1
            key_a, key_b = (ea["precursor_set"], ea["temp_str"]), (eb["precursor_set"], eb["temp_str"])
            route_a, route_b = routes[key_a], routes[key_b]
            bd_a, bd_b = val_bd[key_a], val_bd[key_b]
            gdpo_winner, _reason, _pc = base.gdpo_vote_and_channels(bd_a, bd_b)
            comp_winner, _creason = base.comparator_rule(comparator, scales, route_a, route_b, target_formula)
            base_fewer = base.baseline_fewer_precursors(route_a, route_b)
            base_carbfree = base.baseline_carbonate_free(route_a, route_b)
            scored.append({
                "set_a": ea["precursor_set"], "set_b": eb["precursor_set"],
                "agree_gdpo": base.agreement_score(gdpo_winner, actual) if gdpo_winner is not None else None,
                "agree_comp": base.agreement_score(comp_winner, actual) if comp_winner is not None else None,
                "agree_chance": base.agreement_score(0, actual),
                "agree_fewer": base.agreement_score(base_fewer, actual),
                "agree_carbfree": base.agreement_score(base_carbfree, actual),
            })
        agreements = {
            "gdpo_vote": base.mean_agreement(scored, "agree_gdpo"),
            "comparator": base.mean_agreement(scored, "agree_comp"),
            "chance": base.mean_agreement(scored, "agree_chance"),
            "fewer_precursors": base.mean_agreement(scored, "agree_fewer"),
            "carbonate_free": base.mean_agreement(scored, "agree_carbfree"),
        }
        rebuilt[name] = {
            "n_total_pairs": n_total, "n_excluded_identical": n_identical,
            "n_strict_gt_0": n_strict, "agreements": agreements,
        }
        print(f"{name}: n_total={n_total}  excluded(identical outcome)={n_identical}  "
              f"n(|delta|>0)={n_strict}")
        print(f"  agreements: {agreements}")
    results["rebuilt_threshold_gt_0"] = rebuilt

    # ================= 2. Thermo-only same-pairs analysis (YBCO, thr>=5) =================
    print("\n" + "=" * 90 + "\n2. THERMO-ONLY CHANNEL ANALYSIS (YBCO, |delta|>=5)\n" + "=" * 90)
    entries, target_formula, routes, val_bd, comp_gates, by_temp = build_all("YBCO", validator, comparator, scales)
    all_pairs_raw = enumerate_pairs(by_temp)

    scored5 = []
    for ea, eb in all_pairs_raw:
        d = ea["wt_pct"] - eb["wt_pct"]
        if abs(d) < 5 - 1e-9:
            continue
        actual = 1 if d > 0 else (-1 if d < 0 else 0)
        key_a, key_b = (ea["precursor_set"], ea["temp_str"]), (eb["precursor_set"], eb["temp_str"])
        route_a, route_b = routes[key_a], routes[key_b]
        bd_a, bd_b = val_bd[key_a], val_bd[key_b]
        thermo_winner = thermo_only_vote(bd_a, bd_b)
        base_fewer = base.baseline_fewer_precursors(route_a, route_b)
        base_carbfree = base.baseline_carbonate_free(route_a, route_b)
        a_carb = any(p.formula in base.CARBONATES for p in route_a.precursors)
        b_carb = any(p.formula in base.CARBONATES for p in route_b.precursors)
        carbfree_ties = (a_carb == b_carb)
        scored5.append({
            "set_a": ea["precursor_set"], "set_b": eb["precursor_set"],
            "thermo_winner": thermo_winner, "actual": actual,
            "agree_thermo": base.agreement_score(thermo_winner, actual),
            "agree_fewer": base.agreement_score(base_fewer, actual),
            "agree_carbfree": base.agreement_score(base_carbfree, actual),
            "thermo_decisive": thermo_winner != 0,
            "carbfree_decisive": base_carbfree != 0,
            "carbfree_ties": carbfree_ties,
        })

    # (a) thermo-only rule vs best baseline, cluster bootstrap
    best_baseline_agreement_5 = max(
        base.mean_agreement(scored5, "agree_fewer"),
        base.mean_agreement(scored5, "agree_carbfree"),
    )
    best_baseline_name = "fewer_precursors" if base.mean_agreement(scored5, "agree_fewer") >= base.mean_agreement(scored5, "agree_carbfree") else "carbonate_free"
    thermo_agreement_all = base.mean_agreement(scored5, "agree_thermo")

    def diff_thermo(subset):
        v = base.mean_agreement(subset, "agree_thermo")
        key = "agree_fewer" if best_baseline_name == "fewer_precursors" else "agree_carbfree"
        b = base.mean_agreement(subset, key)
        return (v - b) if (v is not None and b is not None) else 0.0

    ci_thermo = base.cluster_bootstrap_ci(scored5, diff_thermo, rng)
    print(f"(a) thermo-only agreement (all {len(scored5)} primary pairs) = {thermo_agreement_all:.4f}")
    print(f"    best baseline ({best_baseline_name}) = {best_baseline_agreement_5:.4f}")
    print(f"    diff = {thermo_agreement_all - best_baseline_agreement_5:+.4f}, 95% CI = {ci_thermo}")

    # (b) thermo's decisive pairs only
    decisive = [p for p in scored5 if p["thermo_decisive"]]
    n_decisive = len(decisive)
    thermo_agree_decisive = base.mean_agreement(decisive, "agree_thermo")
    fewer_agree_decisive = base.mean_agreement(decisive, "agree_fewer")
    carbfree_agree_decisive = base.mean_agreement(decisive, "agree_carbfree")
    # does thermo's predicted side match carbonate-free's predicted side,
    # among pairs where BOTH are decisive?
    match_count, comparable_count = 0, 0
    for ea, eb in all_pairs_raw:
        d = ea["wt_pct"] - eb["wt_pct"]
        if abs(d) < 5 - 1e-9:
            continue
        key_a, key_b = (ea["precursor_set"], ea["temp_str"]), (eb["precursor_set"], eb["temp_str"])
        route_a, route_b = routes[key_a], routes[key_b]
        bd_a, bd_b = val_bd[key_a], val_bd[key_b]
        tw = thermo_only_vote(bd_a, bd_b)
        if tw == 0:
            continue
        cf = base.baseline_carbonate_free(route_a, route_b)
        if cf == 0:
            continue
        comparable_count += 1
        if tw == cf:
            match_count += 1
    frac_matches_carbfree = match_count / comparable_count if comparable_count else None

    print(f"(b) thermo decisive pairs: n={n_decisive}")
    print(f"    thermo agreement (decisive only) = {thermo_agree_decisive}")
    print(f"    carbonate-free agreement (same pairs) = {carbfree_agree_decisive}")
    print(f"    fewer-precursors agreement (same pairs) = {fewer_agree_decisive}")
    print(f"    fraction where thermo's pick == carbonate-free's pick "
          f"(both decisive, n={comparable_count}) = {frac_matches_carbfree}")

    # (c) thermo agreement restricted to carbonate-free-tie pairs
    carbfree_tie_pairs = [p for p in scored5 if p["carbfree_ties"]]
    n_carbfree_tie = len(carbfree_tie_pairs)
    thermo_agree_carbfree_tie = base.mean_agreement(carbfree_tie_pairs, "agree_thermo")

    def thermo_agree_fn(subset):
        v = base.mean_agreement(subset, "agree_thermo")
        return v if v is not None else 0.0

    ci_carbfree_tie = base.cluster_bootstrap_ci(carbfree_tie_pairs, thermo_agree_fn, rng) if carbfree_tie_pairs else (None, None)
    print(f"(c) pairs where carbonate-free TIES (same carbonate status both sides): n={n_carbfree_tie}")
    print(f"    thermo agreement on these = {thermo_agree_carbfree_tie}, "
          f"95% CI (cluster bootstrap) = {ci_carbfree_tie}")

    results["thermo_analysis"] = {
        "a_thermo_vs_best_baseline": {
            "thermo_agreement": thermo_agreement_all, "best_baseline": best_baseline_name,
            "best_baseline_agreement": best_baseline_agreement_5,
            "diff": thermo_agreement_all - best_baseline_agreement_5, "diff_95ci": ci_thermo,
        },
        "b_decisive_pairs_only": {
            "n_decisive": n_decisive, "thermo_agreement": thermo_agree_decisive,
            "carbonate_free_agreement_same_pairs": carbfree_agree_decisive,
            "fewer_precursors_agreement_same_pairs": fewer_agree_decisive,
            "n_comparable_both_decisive": comparable_count,
            "frac_thermo_matches_carbfree": frac_matches_carbfree,
        },
        "c_carbfree_tie_pairs": {
            "n": n_carbfree_tie, "thermo_agreement": thermo_agree_carbfree_tie,
            "thermo_agreement_95ci": ci_carbfree_tie,
        },
    }

    # ================= 3. Comparator agreement on channel-decided-only pairs =================
    print("\n" + "=" * 90 + "\n3. COMPARATOR AGREEMENT, CHANNEL-DECIDED PAIRS ONLY (YBCO, |delta|>=5)\n" + "=" * 90)
    channel_decided = []
    for ea, eb in all_pairs_raw:
        d = ea["wt_pct"] - eb["wt_pct"]
        if abs(d) < 5 - 1e-9:
            continue
        actual = 1 if d > 0 else -1
        key_a, key_b = (ea["precursor_set"], ea["temp_str"]), (eb["precursor_set"], eb["temp_str"])
        route_a, route_b = routes[key_a], routes[key_b]
        gates_a, gates_b = comp_gates[key_a], comp_gates[key_b]
        a_fail, b_fail = not all(gates_a.values()), not all(gates_b.values())
        if a_fail or b_fail:
            continue  # gate-decided or both-fail -- excluded, we want channel-decided only
        comp_winner, _ = base.comparator_rule(comparator, scales, route_a, route_b, target_formula)
        channel_decided.append({"agree": base.agreement_score(comp_winner, actual)})
    n_channel_decided = len(channel_decided)
    channel_decided_agreement = base.mean_agreement(channel_decided, "agree")
    print(f"n channel-decided pairs (both gates pass, |delta|>=5) = {n_channel_decided}")
    print(f"comparator agreement on these = {channel_decided_agreement}")
    results["comparator_channel_decided_only"] = {
        "n": n_channel_decided, "agreement": channel_decided_agreement,
    }

    # ================= Gate-failure audit summary (from direct inspection, recorded here) =================
    results["gate_failure_audit"] = {
        "note": "Computed by direct inspection of results/external/arrows_gradeability_dryrun.json, not re-scored here.",
        "precursors_exist_failures": {
            "n_ybco_sets": 16, "cause": "100% attributable to Y2(CO3)3 (absent from Materials Project entirely -- confirmed via direct mp_formula_set.pkl membership check; every other precursor across all 3 targets, all 30 distinct formulas, IS present in MP)",
            "category": "(a) absent from MP",
        },
        "balances_failures": {
            "n_ybco_sets": 5,
            "breakdown": "3/5 also contain Y2(CO3)3 (already broken via precursors_exist). 2/5 do NOT contain Y2(CO3)3: ('BaCuO2','BaO2','Cu2O','Y2O3') and ('BaO','BaO2','Cu2O','Y2O3') -- all formulas parse cleanly and are in MP, so this is NOT a notation/parsing issue.",
            "category": "(c) genuinely unbalanceable with the validator's finite candidate-volatile-set search, for these specific 4-precursor combinations (likely an overdetermined system: two separate Ba sources plus Cu2O plus Y2O3 simultaneously)",
        },
        "notation_parsing_failures_category_b": {
            "n": 0,
            "note": "ZERO instances. All 30 distinct precursor formulas across all three targets -- including parenthetical forms (Ba2(CuO2)3), peroxides (BaO2, Na2O2), and complex ammonium-phosphate-style formulas (PH9(NO2)2, MoH8(NO2)2) -- parse without error via pymatgen Composition(). Category (b) does not dominate; it is empty.",
        },
        "ltopo_ntmo": {"precursors_exist_failures": 0, "balances_failures": 0},
    }

    OUT_JSON.write_text(json.dumps(results, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
