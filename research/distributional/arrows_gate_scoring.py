#!/usr/bin/env python
"""
research/distributional/arrows_gate_scoring.py — Phase 15 Task 4c.
Scores exactly as pre-registered in docs/phases/PHASE15_ARROWS_PREREG.md
(committed before this script touched any outcome data). One iteration;
no changes after seeing results.

Outputs: results/external/arrows_gate.json, and this script's console
output feeds docs/phases/PHASE15_ARROWS_RESULTS.md directly.

Usage (tmux, real PD cache):
  uv run python research/distributional/arrows_gate_scoring.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pymatgen.core import Composition  # noqa: E402
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError  # noqa: E402

from validator import (PredictedConditions, PredictedOperation,  # noqa: E402
                       PredictedPrecursor, PredictedRoute, SynthesisValidator,
                       VOLATILE_FORMULAS)
from core.reward import RUN3_CHECKS, load_validator  # noqa: E402
from core.comparator import ComparatorParams, load_comparator  # noqa: E402

ARROWS_DIR = Path("data/external/arrows/ARROWS/Examples")
OUT_JSON = Path("results/external/arrows_gate.json")
COMPARATOR_SCALES_PATH = Path("misc/comparator_scales_v2.json")
BA_PREFERENCE_PATH = Path("results/external/arrows_ba_source_verifier_preference.json")

CANONICAL_TARGET = {"YBCO": "YBa2Cu3O7", "LTOPO": "LiTiOPO4", "NTMO": "Na2Te3Mo3O16"}
CARBONATES = {"Li2CO3", "Na2CO3", "K2CO3", "BaCO3", "SrCO3", "CaCO3", "MgCO3"}
SEED = 20260924
N_BOOT = 10000
THRESHOLDS = [0, 5, 10]


# ---------------------------------------------------------------------------
# Target-match rule (locked in the pre-reg, verified against real labels)
# ---------------------------------------------------------------------------

def base_formula(label: str) -> str:
    return label.split("_")[0]


def is_ybco_match(label: str) -> bool:
    f = base_formula(label)
    try:
        amt = Composition(f).get_el_amt_dict()
    except Exception:
        return False
    y, ba, cu, o = amt.get("Y", 0), amt.get("Ba", 0), amt.get("Cu", 0), amt.get("O", 0)
    if y == 0 or ba == 0 or cu == 0:
        return False
    ba_n, cu_n, o_n = ba / y, cu / y, o / y
    return abs(ba_n - 2) < 0.05 and abs(cu_n - 3) < 0.05 and 6 - 0.05 <= o_n <= 7 + 0.05


def target_match(name: str, label: str) -> bool:
    if name == "YBCO":
        return is_ybco_match(label)
    f = base_formula(label)
    try:
        target_red = Composition(CANONICAL_TARGET[name]).reduced_formula
        return Composition(f).reduced_formula == target_red
    except Exception:
        return False


def target_wt_pct(name: str, entry: dict) -> float:
    products = entry.get("products", [])
    wfs = entry.get("product weight fractions", [])
    total = 0.0
    for p, wf in zip(products, wfs):
        if target_match(name, p):
            total += wf
    return total


# ---------------------------------------------------------------------------
# Route construction (identical to Step 1)
# ---------------------------------------------------------------------------

def solve_balanced_amounts(precursors: list[str], target: str) -> list[float] | None:
    try:
        reactants = [Composition(p) for p in precursors]
        target_comp = Composition(target)
    except Exception:
        return None
    for volatile_strs in [[], ["CO2"], ["H2O"], ["O2"], ["CO2", "H2O", "O2"], list(VOLATILE_FORMULAS)]:
        try:
            volatile_set = [Composition(v) for v in volatile_strs]
            reaction = Reaction(reactants, [target_comp] + volatile_set)
        except (ReactionError, Exception):
            continue
        try:
            coeffs = [abs(reaction.get_coeff(r)) for r in reactants]
        except (ValueError, KeyError):
            continue
        if all(c > 1e-9 for c in coeffs):
            return coeffs
    return None


def build_route(target: str, precursors: list[str], amounts: list[float], temp_C: float,
                atmosphere: str) -> PredictedRoute:
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(p, a) for p, a in zip(precursors, amounts)],
        operations=[PredictedOperation(
            type="HeatingOperation",
            conditions=PredictedConditions(heating_temperature=[temp_C], heating_atmosphere=[atmosphere]))],
    )


def load_target_entries(name: str) -> tuple[list[dict], str]:
    with (ARROWS_DIR / name / "Exp.json").open() as f:
        d = json.load(f)["Universal File"]
    common = d.get("Common Experimental Conditions")
    atmosphere = (common or {}).get("atmosphere", "air")
    target_formula = CANONICAL_TARGET[name]
    entries = []
    for pset_key, val in d.items():
        if pset_key == "Common Experimental Conditions":
            continue
        precursors = [p.strip() for p in pset_key.split(",")]
        amounts = val.get("Precursor stoichiometry")
        if amounts is None or len(amounts) != len(precursors):
            amounts = solve_balanced_amounts(precursors, target_formula)
            if amounts is None:
                continue
        for temp_str, entry in val.get("Temperatures", {}).items():
            if "products" not in entry or "product weight fractions" not in entry:
                continue
            entries.append({
                "precursor_set": pset_key, "precursors": precursors, "amounts": amounts,
                "temp_C": float(temp_str.split()[0]), "temp_str": temp_str,
                "atmosphere": atmosphere, "wt_pct": target_wt_pct(name, entry),
            })
    return entries, target_formula


# ---------------------------------------------------------------------------
# Verifier rules
# ---------------------------------------------------------------------------

def score_validator(validator, route, target_formula):
    try:
        _r, bd = validator.validate(route, target_formula)
        return bd
    except Exception:
        return None


def gdpo_vote_and_channels(bd_a, bd_b):
    """Returns (winner in {1,-1,0,None}, reason, per_channel_votes dict)."""
    if bd_a is None and bd_b is None:
        return None, "both_gate_fail_excluded", {}
    if bd_a is None:
        return -1, "gate_failure_a", {}
    if bd_b is None:
        return 1, "gate_failure_b", {}
    vote_sum = 0
    per_channel = {}
    for ch in RUN3_CHECKS:
        grade_key = f"{ch}_gradeability"
        a_ungrad = bd_a.get(grade_key) in SynthesisValidator.SENTINEL_TAGS
        b_ungrad = bd_b.get(grade_key) in SynthesisValidator.SENTINEL_TAGS
        va, vb = bd_a.get(ch), bd_b.get(ch)
        if a_ungrad or b_ungrad or va is None or vb is None:
            per_channel[ch] = ("ungradeable", 0)
            continue
        diff = va - vb
        vote = 1 if diff > 1e-9 else (-1 if diff < -1e-9 else 0)
        per_channel[ch] = ("tie" if vote == 0 else ("A" if vote == 1 else "B"), vote)
        vote_sum += vote
    winner = 1 if vote_sum > 0 else (-1 if vote_sum < 0 else 0)
    return winner, f"vote_sum={vote_sum}", per_channel


def nansum_rule(bd_a, bd_b):
    if bd_a is None and bd_b is None:
        return None, "both_gate_fail_excluded"
    if bd_a is None:
        return -1, "gate_failure_a"
    if bd_b is None:
        return 1, "gate_failure_b"
    sum_a, sum_b = 0.0, 0.0
    for ch in RUN3_CHECKS:
        va, vb = bd_a.get(ch), bd_b.get(ch)
        grade_a = bd_a.get(f"{ch}_gradeability") in SynthesisValidator.SENTINEL_TAGS
        grade_b = bd_b.get(f"{ch}_gradeability") in SynthesisValidator.SENTINEL_TAGS
        if va is not None and not grade_a:
            sum_a += va
        if vb is not None and not grade_b:
            sum_b += vb
    diff = sum_a - sum_b
    winner = 1 if diff > 1e-9 else (-1 if diff < -1e-9 else 0)
    return winner, f"sum_a={sum_a:.4f} sum_b={sum_b:.4f}"


def comparator_rule(comparator, scales, route_a, route_b, target_formula):
    gates_a = comparator._gater._check_gates(route_a)
    gates_b = comparator._gater._check_gates(route_b)
    a_fail, b_fail = not all(gates_a.values()), not all(gates_b.values())
    if a_fail and b_fail:
        return None, "both_gate_fail_excluded"
    if a_fail:
        return -1, "gate_failure_a"
    if b_fail:
        return 1, "gate_failure_b"
    params = ComparatorParams(c3_fraction=0.17)
    margin, _bd = comparator.compare(route_a, route_b, target_formula, scales, params)
    winner = 1 if margin > 1e-9 else (-1 if margin < -1e-9 else 0)
    return winner, f"margin={margin:.4f}"


def baseline_fewer_precursors(route_a, route_b):
    na, nb = len(route_a.precursors), len(route_b.precursors)
    return 1 if na < nb else (-1 if nb < na else 0)


def baseline_carbonate_free(route_a, route_b):
    a_has = any(p.formula in CARBONATES for p in route_a.precursors)
    b_has = any(p.formula in CARBONATES for p in route_b.precursors)
    if a_has == b_has:
        return 0
    return 1 if not a_has else -1


def actual_winner(wt_a, wt_b, threshold):
    d = wt_a - wt_b
    if abs(d) < threshold - 1e-9:
        return None  # doesn't clear this threshold, excluded from that arm
    if d > 0:
        return 1
    if d < 0:
        return -1
    return 0


def agreement_score(predicted_winner, actual):
    if predicted_winner == 0:
        return 0.5
    return 1.0 if predicted_winner == actual else 0.0


# ---------------------------------------------------------------------------
# Cluster bootstrap (resample precursor sets)
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(pairs: list[dict], value_fn, rng: random.Random, n_boot=N_BOOT):
    """pairs: list of dicts with 'set_a','set_b', plus whatever value_fn needs.
    value_fn(pairs_subset) -> float. Resamples the distinct precursor-set
    labels with replacement, reconstructs the multiset of pairs whose both
    endpoints were drawn (weighted by draw multiplicity), recomputes
    value_fn on each resample."""
    all_sets = sorted({p["set_a"] for p in pairs} | {p["set_b"] for p in pairs})
    n_sets = len(all_sets)
    boot_vals = []
    for _ in range(n_boot):
        drawn = rng.choices(all_sets, k=n_sets)
        counts = defaultdict(int)
        for s in drawn:
            counts[s] += 1
        resampled_pairs = []
        for p in pairs:
            mult = counts.get(p["set_a"], 0) * counts.get(p["set_b"], 0)
            if mult > 0:
                resampled_pairs.extend([p] * mult)
        if resampled_pairs:
            boot_vals.append(value_fn(resampled_pairs))
    boot_vals.sort()
    n = len(boot_vals)
    lo = boot_vals[int(0.025 * n)]
    hi = boot_vals[int(0.975 * n)]
    return lo, hi


def mean_agreement(pairs: list[dict], key: str) -> float:
    vals = [p[key] for p in pairs if p[key] is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Main per-target scoring
# ---------------------------------------------------------------------------

def score_target(name: str, validator, comparator, scales, rng):
    entries, target_formula = load_target_entries(name)
    print(f"\n{'='*90}\n{name}: {len(entries)} entries\n{'='*90}")

    # build routes once, cache validator/comparator breakdowns per entry
    routes = {}
    val_bd = {}
    for e in entries:
        key = (e["precursor_set"], e["temp_str"])
        route = build_route(target_formula, e["precursors"], e["amounts"], e["temp_C"], e["atmosphere"])
        routes[key] = route
        val_bd[key] = score_validator(validator, route, target_formula)

    by_temp = defaultdict(list)
    for e in entries:
        by_temp[e["temp_str"]].append(e)

    all_pairs = []  # every within-temp pair, regardless of threshold, with |delta| tagged
    gate_decided_gdpo, gate_decided_comp, excluded_both_gdpo, excluded_both_comp = 0, 0, 0, 0
    channel_agree = {ch: {"agree": 0, "disagree": 0, "tie": 0} for ch in RUN3_CHECKS}

    for temp_str, es in by_temp.items():
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                ea, eb = es[i], es[j]
                key_a = (ea["precursor_set"], ea["temp_str"])
                key_b = (eb["precursor_set"], eb["temp_str"])
                route_a, route_b = routes[key_a], routes[key_b]
                bd_a, bd_b = val_bd[key_a], val_bd[key_b]

                gdpo_winner, gdpo_reason, per_channel = gdpo_vote_and_channels(bd_a, bd_b)
                if gdpo_reason.startswith("gate_failure"):
                    gate_decided_gdpo += 1
                if gdpo_reason == "both_gate_fail_excluded":
                    excluded_both_gdpo += 1

                nansum_winner, _ = nansum_rule(bd_a, bd_b)

                comp_winner, comp_reason = comparator_rule(comparator, scales, route_a, route_b, target_formula)
                if comp_reason.startswith("gate_failure"):
                    gate_decided_comp += 1
                if comp_reason == "both_gate_fail_excluded":
                    excluded_both_comp += 1

                base_chance = 0
                base_fewer = baseline_fewer_precursors(route_a, route_b)
                base_carbfree = baseline_carbonate_free(route_a, route_b)

                delta = abs(ea["wt_pct"] - eb["wt_pct"])

                for ch, (label, vote) in per_channel.items():
                    dp = ea["wt_pct"] - eb["wt_pct"]
                    if label == "ungradeable":
                        continue
                    if abs(dp) < 1e-9 or vote == 0:
                        channel_agree[ch]["tie"] += 1
                        continue
                    channel_pred = 1 if vote == 1 else -1
                    actual = 1 if dp > 0 else -1
                    if channel_pred == actual:
                        channel_agree[ch]["agree"] += 1
                    else:
                        channel_agree[ch]["disagree"] += 1

                all_pairs.append({
                    "set_a": ea["precursor_set"], "set_b": eb["precursor_set"],
                    "temp": temp_str, "delta_wt_pct": delta,
                    "wt_a": ea["wt_pct"], "wt_b": eb["wt_pct"],
                    "gdpo_winner": gdpo_winner, "nansum_winner": nansum_winner,
                    "comp_winner": comp_winner, "base_chance": base_chance,
                    "base_fewer": base_fewer, "base_carbfree": base_carbfree,
                })

    # per-threshold agreement
    threshold_results = {}
    for thr in THRESHOLDS:
        scored_pairs = []
        for p in all_pairs:
            actual = actual_winner(p["wt_a"], p["wt_b"], thr)
            if actual is None:
                continue
            scored_pairs.append({
                **p, "actual": actual,
                "agree_gdpo": agreement_score(p["gdpo_winner"], actual) if p["gdpo_winner"] is not None else None,
                "agree_nansum": agreement_score(p["nansum_winner"], actual) if p["nansum_winner"] is not None else None,
                "agree_comp": agreement_score(p["comp_winner"], actual) if p["comp_winner"] is not None else None,
                "agree_chance": agreement_score(p["base_chance"], actual),
                "agree_fewer": agreement_score(p["base_fewer"], actual),
                "agree_carbfree": agreement_score(p["base_carbfree"], actual),
            })
        agreements = {
            "gdpo_vote": mean_agreement(scored_pairs, "agree_gdpo"),
            "nansum": mean_agreement(scored_pairs, "agree_nansum"),
            "comparator": mean_agreement(scored_pairs, "agree_comp"),
            "chance": mean_agreement(scored_pairs, "agree_chance"),
            "fewer_precursors": mean_agreement(scored_pairs, "agree_fewer"),
            "carbonate_free": mean_agreement(scored_pairs, "agree_carbfree"),
        }
        tie_rates = {
            "gdpo_vote": sum(1 for p in scored_pairs if p["gdpo_winner"] == 0) / len(scored_pairs) if scored_pairs else None,
            "comparator": sum(1 for p in scored_pairs if p["comp_winner"] == 0) / len(scored_pairs) if scored_pairs else None,
        }
        threshold_results[thr] = {
            "n_pairs": len(scored_pairs), "agreements": agreements, "tie_rates": tie_rates,
            "scored_pairs": scored_pairs if thr == 5 else None,  # keep detail only for primary
        }
        print(f"  threshold |delta|>={thr}: n={len(scored_pairs)}  agreements={agreements}")

    per_channel_report = {}
    for ch, d in channel_agree.items():
        n = d["agree"] + d["disagree"]
        per_channel_report[ch] = {**d, "agreement_rate": d["agree"] / n if n else None}
    print(f"  per-channel agreement (all pairs, any delta): {per_channel_report}")
    print(f"  gate-decided pairs: GDPO={gate_decided_gdpo} comparator={gate_decided_comp}  "
          f"both-gate-fail excluded: GDPO={excluded_both_gdpo} comparator={excluded_both_comp}")

    return {
        "n_entries": len(entries), "n_total_pairs": len(all_pairs),
        "threshold_results": {str(k): v for k, v in threshold_results.items()},
        "per_channel_agreement": per_channel_report,
        "gate_decided_pairs": {"gdpo": gate_decided_gdpo, "comparator": gate_decided_comp},
        "excluded_both_gate_fail": {"gdpo": excluded_both_gdpo, "comparator": excluded_both_comp},
        "_all_pairs_primary_threshold": [p for p in all_pairs if p["delta_wt_pct"] >= 5],
    }


def primary_endpoint(target_results: dict, rng) -> dict:
    thr5 = target_results["threshold_results"]["5"]
    scored = thr5["scored_pairs"]
    if not scored:
        return {"error": "no pairs at threshold 5"}
    baselines = ["chance", "fewer_precursors", "carbonate_free"]
    baseline_agreements = {b: thr5["agreements"][{"chance": "chance", "fewer_precursors": "fewer_precursors",
                                                   "carbonate_free": "carbonate_free"}[b]] for b in baselines}
    best_baseline = max(baseline_agreements, key=lambda b: baseline_agreements[b])
    best_baseline_val = baseline_agreements[best_baseline]

    key_map = {"chance": "agree_chance", "fewer_precursors": "agree_fewer", "carbonate_free": "agree_carbfree"}
    best_key = key_map[best_baseline]

    def diff_gdpo(pairs_subset):
        v = mean_agreement(pairs_subset, "agree_gdpo")
        b = mean_agreement(pairs_subset, best_key)
        return (v - b) if (v is not None and b is not None) else 0.0

    def diff_comp(pairs_subset):
        v = mean_agreement(pairs_subset, "agree_comp")
        b = mean_agreement(pairs_subset, best_key)
        return (v - b) if (v is not None and b is not None) else 0.0

    obs_diff_gdpo = thr5["agreements"]["gdpo_vote"] - best_baseline_val
    obs_diff_comp = thr5["agreements"]["comparator"] - best_baseline_val

    ci_gdpo = cluster_bootstrap_ci(scored, diff_gdpo, rng)
    ci_comp = cluster_bootstrap_ci(scored, diff_comp, rng)

    return {
        "best_baseline": best_baseline, "best_baseline_agreement": best_baseline_val,
        "validator_gdpo_vote_agreement": thr5["agreements"]["gdpo_vote"],
        "validator_diff": obs_diff_gdpo, "validator_diff_95ci": ci_gdpo,
        "validator_pass": ci_gdpo[0] > 0,
        "comparator_agreement": thr5["agreements"]["comparator"],
        "comparator_diff": obs_diff_comp, "comparator_diff_95ci": ci_comp,
        "comparator_pass": ci_comp[0] > 0,
    }


def ba_source_test(rng):
    pref = json.loads(BA_PREFERENCE_PATH.read_text())
    ybco_entries, _ = load_target_entries("YBCO")
    wt_lookup = {(e["precursor_set"], e["temp_str"]): e["wt_pct"] for e in ybco_entries}

    group_rows = []
    for g in pref["groups"]:
        co = g["co_precursors"]
        inv = json.loads(INVENTORY_PATH.read_text())
        matching_inv_group = next(
            (mg for mg in inv["ybco_ba_source_matched_comparisons"]["detail"]
             if mg["co_precursors"] == co), None)
        if matching_inv_group is None:
            continue
        sources = matching_inv_group["sources"]
        for cell in g["cells"]:
            temp = cell["temperature"]
            gdpo_order = cell["gdpo_vote_ordering_best_first"]
            preferred, least_preferred = gdpo_order[0], gdpo_order[-1]
            pset_pref = sources[preferred]["precursor_set"]
            pset_least = sources[least_preferred]["precursor_set"]
            wt_pref = wt_lookup.get((pset_pref, temp))
            wt_least = wt_lookup.get((pset_least, temp))
            if wt_pref is None or wt_least is None:
                continue
            diff = wt_pref - wt_least
            group_rows.append({
                "co_precursors": co, "temperature": temp,
                "preferred_source": preferred, "least_preferred_source": least_preferred,
                "wt_preferred": wt_pref, "wt_least_preferred": wt_least,
                "diff": diff, "sign": (1 if diff > 0 else (-1 if diff < 0 else 0)),
            })

    print("\n-- Ba-source controlled test: full group x temperature table --")
    for r in group_rows:
        print(f"  {r['co_precursors']} @ {r['temperature']}: "
              f"{r['preferred_source']}={r['wt_preferred']:.1f}%  "
              f"{r['least_preferred_source']}={r['wt_least_preferred']:.1f}%  "
              f"diff={r['diff']:+.1f}  sign={r['sign']:+d}")

    # bootstrap over GROUPS (not cells)
    groups_list = sorted({tuple(r["co_precursors"]) for r in group_rows})
    def mean_diff(rows_subset):
        vals = [r["diff"] for r in rows_subset]
        return sum(vals) / len(vals) if vals else 0.0

    boot_means = []
    for _ in range(N_BOOT):
        drawn_groups = rng.choices(groups_list, k=len(groups_list))
        subset = [r for r in group_rows if tuple(r["co_precursors"]) in drawn_groups]
        # weight by draw count
        weighted = []
        from collections import Counter
        draw_counts = Counter(drawn_groups)
        for r in group_rows:
            c = draw_counts.get(tuple(r["co_precursors"]), 0)
            weighted.extend([r] * c)
        if weighted:
            boot_means.append(mean_diff(weighted))
    boot_means.sort()
    n = len(boot_means)
    ci = (boot_means[int(0.025 * n)], boot_means[int(0.975 * n)]) if n else (None, None)
    obs_mean = mean_diff(group_rows)
    n_positive = sum(1 for r in group_rows if r["sign"] > 0)
    n_negative = sum(1 for r in group_rows if r["sign"] < 0)
    n_zero = sum(1 for r in group_rows if r["sign"] == 0)

    print(f"\nmean diff (preferred - least_preferred) = {obs_mean:+.2f} wt%, "
          f"95% CI (bootstrap over {len(groups_list)} groups) = [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    print(f"sign distribution across {len(group_rows)} cells: "
          f"positive(verifier claim holds)={n_positive}  negative(claim fails)={n_negative}  zero={n_zero}")
    claim_supported = ci[0] > 0
    print(f"Verifier's claim (preferred source gives higher wt%) supported? {claim_supported}")

    return {
        "group_rows": group_rows, "n_groups": len(groups_list),
        "mean_diff": obs_mean, "diff_95ci": ci,
        "n_positive": n_positive, "n_negative": n_negative, "n_zero": n_zero,
        "verifier_claim_supported": claim_supported,
    }


INVENTORY_PATH = Path("results/external/arrows_inventory.json")


def main():
    rng = random.Random(SEED)
    print("loading validator and comparator...", flush=True)
    validator = load_validator(Path("data/cache/mp_formula_set.pkl"), Path("data/cache/pd_index.json"), Path("."))
    comparator = load_comparator(Path("data/cache/mp_formula_set.pkl"), Path("data/cache/pd_index.json"), Path("."))
    scales = json.loads(COMPARATOR_SCALES_PATH.read_text())["scales"]

    all_results = {}
    for name in ["YBCO", "LTOPO", "NTMO"]:
        all_results[name] = score_target(name, validator, comparator, scales, rng)

    print("\n" + "=" * 90 + "\nPRIMARY ENDPOINT (YBCO)\n" + "=" * 90)
    primary = primary_endpoint(all_results["YBCO"], rng)
    print(json.dumps(primary, indent=1, default=str))

    print("\n" + "=" * 90 + "\nSECONDARY REPLICATIONS (LTOPO, NTMO)\n" + "=" * 90)
    secondary = {}
    for name in ["LTOPO", "NTMO"]:
        secondary[name] = primary_endpoint(all_results[name], rng)
        print(f"{name}: {json.dumps(secondary[name], indent=1, default=str)}")

    print("\n" + "=" * 90 + "\nBA-SOURCE CONTROLLED TEST\n" + "=" * 90)
    ba_result = ba_source_test(rng)

    # trim large per-pair detail before final JSON dump to keep file size reasonable
    for name in all_results:
        all_results[name].pop("_all_pairs_primary_threshold", None)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "seed": SEED, "n_boot": N_BOOT,
        "per_target": all_results,
        "primary_endpoint_ybco": primary,
        "secondary_replications": secondary,
        "ba_source_test": ba_result,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
