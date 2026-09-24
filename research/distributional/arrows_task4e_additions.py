#!/usr/bin/env python
"""
research/distributional/arrows_task4e_additions.py — three small [E]
additions to docs/phases/PHASE15_ARROWS_RESULTS.md, requested on review:

1. Direct agreement on the 2,115 gate-decided pairs (not inferred from
   pooled minus channel-decided). Also: mean YBCO target-phase yield for
   routes containing Y2(CO3)3 vs. other Y sources, at matched temperature.
2. How many of YBCO's 47 precursor sets contain Y2(CO3)3.
3. Carbonate-free baseline's own agreement vs 0.5, cluster-bootstrap CI --
   is "avoid carbonates" itself better than chance on this data?

Usage (tmux, real PD cache):
  uv run python research/distributional/arrows_task4e_additions.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.reward import load_validator  # noqa: E402
from core.comparator import load_comparator  # noqa: E402

import arrows_gate_scoring as base  # noqa: E402

OUT_JSON = Path("results/external/arrows_task4e_additions.json")
SEED = 20260924
N_BOOT = 10000


def main():
    rng = random.Random(SEED)
    print("loading validator and comparator...", flush=True)
    validator = load_validator(Path("data/cache/mp_formula_set.pkl"), Path("data/cache/pd_index.json"), Path("."))
    comparator = load_comparator(Path("data/cache/mp_formula_set.pkl"), Path("data/cache/pd_index.json"), Path("."))
    scales = json.loads(base.COMPARATOR_SCALES_PATH.read_text())["scales"]

    entries, target_formula, routes, val_bd, comp_gates, by_temp = None, None, None, None, None, None
    import arrows_task4d_diagnostics as t4d
    entries, target_formula, routes, val_bd, comp_gates, by_temp = t4d.build_all("YBCO", validator, comparator, scales)
    all_pairs_raw = t4d.enumerate_pairs(by_temp)

    # ================= 1. Direct gate-decided-only agreement =================
    print("\n1. Direct gate-decided-pair agreement (YBCO, primary threshold >=5)")
    gate_decided, channel_decided_check = [], []
    for ea, eb in all_pairs_raw:
        d = ea["wt_pct"] - eb["wt_pct"]
        if abs(d) < 5 - 1e-9:
            continue
        actual = 1 if d > 0 else -1
        key_a, key_b = (ea["precursor_set"], ea["temp_str"]), (eb["precursor_set"], eb["temp_str"])
        route_a, route_b = routes[key_a], routes[key_b]
        gates_a, gates_b = comp_gates[key_a], comp_gates[key_b]
        a_fail, b_fail = not all(gates_a.values()), not all(gates_b.values())
        comp_winner, reason = base.comparator_rule(comparator, scales, route_a, route_b, target_formula)
        if reason == "both_gate_fail_excluded":
            continue  # excluded entirely, neither gate-decided nor channel-decided
        if a_fail or b_fail:
            gate_decided.append({"agree": base.agreement_score(comp_winner, actual)})
        else:
            channel_decided_check.append({"agree": base.agreement_score(comp_winner, actual)})

    n_gate_decided = len(gate_decided)
    gate_decided_agreement = base.mean_agreement(gate_decided, "agree")
    n_channel_check = len(channel_decided_check)
    channel_agreement_check = base.mean_agreement(channel_decided_check, "agree")
    print(f"   n gate-decided (one-sided failure) = {n_gate_decided}, agreement = {gate_decided_agreement:.4f}")
    print(f"   n channel-decided (cross-check vs Task 4d's 704/0.513) = {n_channel_check}, "
          f"agreement = {channel_agreement_check:.4f}")

    # ================= Y2(CO3)3 vs other Y-source yield, MATCHED BY TEMPERATURE =================
    # Pooling across all temperatures would conflate the huge temperature
    # effect (0% at 600-700C regardless of precursor choice) with the
    # Y-source effect. Report per-temperature means, plus the pooled
    # figure for reference only.
    print("\n   Mean YBCO target-phase yield: Y2(CO3)3-containing vs other-Y-source routes")
    from collections import defaultdict
    by_temp_y2co3, by_temp_other = defaultdict(list), defaultdict(list)
    for e in entries:
        precs = set(e["precursors"])
        if "Y2(CO3)3" in precs:
            by_temp_y2co3[e["temp_str"]].append(e["wt_pct"])
        elif any(p in precs for p in ["Y2O3", "Y2Cu2O5"]):
            by_temp_other[e["temp_str"]].append(e["wt_pct"])
    per_temp_comparison = {}
    for temp in sorted(set(by_temp_y2co3) | set(by_temp_other), key=lambda t: float(t.split()[0])):
        ys = by_temp_y2co3.get(temp, [])
        os_ = by_temp_other.get(temp, [])
        m_y = sum(ys) / len(ys) if ys else None
        m_o = sum(os_) / len(os_) if os_ else None
        per_temp_comparison[temp] = {"y2co3_3_mean": m_y, "y2co3_3_n": len(ys),
                                     "other_y_mean": m_o, "other_y_n": len(os_)}
        print(f"   {temp}: Y2(CO3)3 mean={m_y} (n={len(ys)})   other-Y mean={m_o} (n={len(os_)})")
    y2co33_yields = [v for vs in by_temp_y2co3.values() for v in vs]
    other_y_yields = [v for vs in by_temp_other.values() for v in vs]
    mean_y2co33 = sum(y2co33_yields) / len(y2co33_yields) if y2co33_yields else None
    mean_other = sum(other_y_yields) / len(other_y_yields) if other_y_yields else None
    print(f"   POOLED (reference only, not matched): Y2(CO3)3 n={len(y2co33_yields)} mean={mean_y2co33:.2f}%   "
          f"other-Y n={len(other_y_yields)} mean={mean_other:.2f}%")

    # ================= 2. How many of 47 YBCO sets contain Y2(CO3)3 =================
    all_sets = {e["precursor_set"] for e in entries}
    sets_with_y2co33 = {s for s in all_sets if "Y2(CO3)3" in [p.strip() for p in s.split(",")]}
    print(f"\n2. {len(sets_with_y2co33)}/{len(all_sets)} YBCO precursor sets contain Y2(CO3)3")

    # ================= 3. Carbonate-free baseline's own significance =================
    print("\n3. Carbonate-free baseline's own agreement vs chance (YBCO, primary threshold >=5)")
    carbfree_pairs = []
    for ea, eb in all_pairs_raw:
        d = ea["wt_pct"] - eb["wt_pct"]
        if abs(d) < 5 - 1e-9:
            continue
        actual = 1 if d > 0 else -1
        key_a, key_b = (ea["precursor_set"], ea["temp_str"]), (eb["precursor_set"], eb["temp_str"])
        route_a, route_b = routes[key_a], routes[key_b]
        cf = base.baseline_carbonate_free(route_a, route_b)
        carbfree_pairs.append({
            "set_a": ea["precursor_set"], "set_b": eb["precursor_set"],
            "agree": base.agreement_score(cf, actual),
        })
    carbfree_agreement = base.mean_agreement(carbfree_pairs, "agree")

    def carbfree_diff_from_chance(subset):
        v = base.mean_agreement(subset, "agree")
        return (v - 0.5) if v is not None else 0.0

    ci_carbfree = base.cluster_bootstrap_ci(carbfree_pairs, carbfree_diff_from_chance, rng)
    print(f"   carbonate-free agreement = {carbfree_agreement:.4f}")
    print(f"   diff from chance (0.5) = {carbfree_agreement - 0.5:+.4f}, "
          f"95% CI (cluster bootstrap) = {ci_carbfree}")
    print(f"   significantly better than chance? {ci_carbfree[0] > 0}")

    OUT_JSON.write_text(json.dumps({
        "gate_decided_direct": {
            "n": n_gate_decided, "agreement": gate_decided_agreement,
        },
        "channel_decided_crosscheck": {
            "n": n_channel_check, "agreement": channel_agreement_check,
        },
        "y2co3_3_yield_comparison": {
            "per_temperature": per_temp_comparison,
            "pooled_reference_only": {
                "y2co3_3_containing": {"n": len(y2co33_yields), "mean_yield": mean_y2co33},
                "other_y_source": {"n": len(other_y_yields), "mean_yield": mean_other},
            },
        },
        "n_ybco_sets_with_y2co33": len(sets_with_y2co33),
        "n_ybco_sets_total": len(all_sets),
        "carbonate_free_baseline_significance": {
            "agreement": carbfree_agreement, "diff_from_chance": carbfree_agreement - 0.5,
            "diff_95ci": ci_carbfree, "significantly_better_than_chance": ci_carbfree[0] > 0,
        },
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
