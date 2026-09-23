#!/usr/bin/env python
"""
research/distributional/carbonate_penalty_analysis.py — Phase 15 Task 2:
which validator.py channel(s) account for the base-model pass@0.9 gap
between bare-alkali-oxide routes and carbonate routes (Task 1's README
claimed 98.9% vs 78.2%; not reproduced by any Task-1 script, re-derived
here directly against the CURRENT, UNMODIFIED SynthesisValidator).

Classification: a sample is bare-oxide-only if it declares >=1 bare
alkali oxide and 0 carbonates; carbonate-only if the reverse; "both" if
it declares at least one of each; "neither" otherwise. Same convention
extended to ammonium-phosphate vs H3PO4 on phosphate targets (step 5).

Uses the same validator every RS-SFT filtering and GDPO training actually
scored against (`core.reward.load_validator`, thermo-aware, real PD
cache) -- this is NOT core/comparator.py's balance-solver-patched
validator; that fix is scoped to the comparator only and is irrelevant
here, since the question is what the CURRENT production validator does.

Usage (tmux, real PD cache):
  uv run python research/distributional/carbonate_penalty_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from validator import WEIGHTS_THERMO  # noqa: E402
from core.reward import ParseFailure, load_validator, parse_completion  # noqa: E402

BASE_GEN_PATH = Path("results/astral_gen_n32_base.json")
OUT_JSON = Path("results/distributional/carbonate_penalty_by_channel.json")

BARE_OXIDE = {"Li2O", "Na2O", "K2O", "Rb2O", "Cs2O"}
CARBONATE = {"Li2CO3", "Na2CO3", "K2CO3", "BaCO3", "SrCO3", "CaCO3", "MgCO3"}
H3PO4_SET = {"H3PO4", "PH3O4"}  # PH3O4: alt element ordering some parses/normalizers emit

CHANNELS = list(WEIGHTS_THERMO.keys())
BAR = 0.9


def is_ammonium(formula: str) -> bool:
    return ("N" in formula) and ("H" in formula) and ("NO3" not in formula)


def classify(precursors: list[str], group_a: set[str], group_b: set[str]) -> str:
    has_a = any(p in group_a for p in precursors)
    has_b = any(p in group_b for p in precursors)
    if has_a and has_b:
        return "both"
    if has_a:
        return "a_only"
    if has_b:
        return "b_only"
    return "neither"


def score_all(doc, validator, group_a: set[str], group_b: set[str],
              custom_classifier=None):
    """Returns {category: [ (breakdown_dict, reward, route, target) ]}"""
    out = {"a_only": [], "b_only": [], "both": [], "neither": []}
    n_parse_fail = 0
    for r in doc["results"]:
        target = r["target"]
        for s in r["samples"]:
            precursors = s.get("precursors")
            if not precursors:
                continue
            if custom_classifier is not None:
                cat = custom_classifier(precursors)
            else:
                cat = classify(precursors, group_a, group_b)
            if cat == "neither":
                continue
            # LIMITATION, stated up front: astral_gen_n32_base.json stores
            # only the analyzed summary (precursors, max_T, reward, match)
            # from research/astral_model_generations.py, not the raw
            # completion text or the model's real operation list/atmosphere
            # -- those were consumed at generation time and not persisted.
            # Reconstruct a minimal single-heating-step PredictedRoute
            # instead, atmosphere defaulted to "air" (the same convention
            # research/ranker_v2_astral_gate.py and
            # research/phase13_astral_scoring*.py use for constructed
            # ASTRAL routes). This is symmetric across both groups being
            # compared (bare-oxide vs carbonate, or ammonium-P vs H3PO4),
            # so it cannot manufacture a fake gap between them, but it CAN
            # wash out signal on channels that depend on the real operation
            # sequence or a non-air atmosphere (operation_order trivially
            # passes with one op; chempot_atmosphere may read differently
            # under an assumed "air" than whatever the model actually
            # declared). Flagged explicitly in the report, not silently
            # assumed away.
            from validator import (PredictedConditions, PredictedOperation,
                                   PredictedPrecursor, PredictedRoute)
            route = PredictedRoute(
                target_formula=target,
                precursors=[PredictedPrecursor(p, 1.0) for p in precursors],
                operations=[PredictedOperation(
                    type="HeatingOperation",
                    conditions=PredictedConditions(
                        heating_temperature=[float(s["max_T"])] if s.get("max_T") else [],
                        heating_atmosphere=["air"]))],
            )
            try:
                reward, bd = validator.validate(route, target)
            except Exception:
                n_parse_fail += 1
                continue
            out[cat].append({"breakdown": bd, "reward": reward, "target": target,
                             "precursors": precursors})
    return out, n_parse_fail


def channel_table(scored: dict, cat_a: str, cat_b: str, label_a: str, label_b: str):
    """For each channel x {cat_a, cat_b}: mean score, frac <1.0, frac
    ungradeable, n. Uses SynthesisValidator.SENTINEL_TAGS to detect
    ungradeable via the `<channel>_gradeability` sibling key."""
    from validator import SynthesisValidator
    table = {}
    for ch in CHANNELS:
        table[ch] = {}
        for cat, label in [(cat_a, label_a), (cat_b, label_b)]:
            vals, ungradeable = [], 0
            for rec in scored[cat]:
                bd = rec["breakdown"]
                grade_key = f"{ch}_gradeability"
                if grade_key in bd and bd[grade_key] in SynthesisValidator.SENTINEL_TAGS:
                    ungradeable += 1
                    continue
                v = bd.get(ch)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(v)
            n_total = len(scored[cat])
            table[ch][label] = {
                "n": n_total,
                "n_gradeable": len(vals),
                "mean_score": sum(vals) / len(vals) if vals else None,
                "frac_below_1": sum(1 for v in vals if v < 1.0 - 1e-9) / len(vals) if vals else None,
                "frac_ungradeable": ungradeable / n_total if n_total else None,
            }
    return table


def pass_rate(scored: dict, cat: str, bar: float = BAR) -> tuple[float, int]:
    recs = scored[cat]
    if not recs:
        return None, 0
    n_pass = sum(1 for r in recs if r["reward"] >= bar)
    return n_pass / len(recs), len(recs)


def print_table(table: dict, label_a: str, label_b: str):
    print(f"{'channel':<24}{'weight':>7}  "
          f"{label_a+' mean':>14}{label_a+' <1.0':>10}{label_a+' ungr':>10}{label_a+' n':>6}  "
          f"{label_b+' mean':>14}{label_b+' <1.0':>10}{label_b+' ungr':>10}{label_b+' n':>6}")
    for ch in CHANNELS:
        a, b = table[ch][label_a], table[ch][label_b]
        def fmt(d):
            m = f"{d['mean_score']:.3f}" if d['mean_score'] is not None else "n/a"
            f1 = f"{d['frac_below_1']:.1%}" if d['frac_below_1'] is not None else "n/a"
            fu = f"{d['frac_ungradeable']:.1%}" if d['frac_ungradeable'] is not None else "n/a"
            return m, f1, fu, d['n']
        am, a1, au, an = fmt(a)
        bm, b1, bu, bn = fmt(b)
        w = WEIGHTS_THERMO.get(ch, 0)
        print(f"{ch:<24}{w:>7.3f}  {am:>14}{a1:>10}{au:>10}{an:>6}  {bm:>14}{b1:>10}{bu:>10}{bn:>6}")


def find_examples(scored: dict, cat: str, channel: str, n: int = 5):
    out = []
    for rec in scored[cat]:
        bd = rec["breakdown"]
        v = bd.get(channel)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 1.0 - 1e-9:
            out.append({"target": rec["target"], "precursors": rec["precursors"],
                       "channel_value": v, "reward": rec["reward"],
                       "gradeability": bd.get(f"{channel}_gradeability")})
            if len(out) >= n:
                break
    return out


def main():
    print("loading validator (thermo-aware, real PD cache)...", flush=True)
    validator = load_validator(
        Path("data/cache/mp_formula_set.pkl"),
        Path("data/cache/pd_index.json"),
        Path("."),
    )
    assert validator.thermo_checker is not None

    doc = json.loads(BASE_GEN_PATH.read_text())

    print("\n" + "=" * 100)
    print("BARE-ALKALI-OXIDE vs CARBONATE (base model, ALL samples)")
    print("=" * 100)
    scored_bo, n_fail_bo = score_all(doc, validator, BARE_OXIDE, CARBONATE)
    print(f"n_score_fail={n_fail_bo}  counts: bare-only={len(scored_bo['a_only'])} "
          f"carbonate-only={len(scored_bo['b_only'])} both={len(scored_bo['both'])} "
          f"neither={len(scored_bo['neither'])}")
    n_carb_with_ammonium = sum(1 for rec in scored_bo["b_only"]
                               if any(is_ammonium(p) for p in rec["precursors"]))
    n_bare_with_ammonium = sum(1 for rec in scored_bo["a_only"]
                               if any(is_ammonium(p) for p in rec["precursors"]))
    print(f"CONFOUND CHECK: of carbonate-only samples, {n_carb_with_ammonium}/"
          f"{len(scored_bo['b_only'])} also declare an ammonium precursor "
          f"(N+H, not nitrate) -- these overlap with the Phase 13 balance-solver "
          f"finding, not an independent carbonate effect. Of bare-oxide-only, "
          f"{n_bare_with_ammonium}/{len(scored_bo['a_only'])} do.")


    p_a, n_a = pass_rate(scored_bo, "a_only")
    p_b, n_b = pass_rate(scored_bo, "b_only")
    print(f"\npass@{BAR} rate: bare-oxide-only = {p_a:.1%} (n={n_a})  "
          f"carbonate-only = {p_b:.1%} (n={n_b})")
    print("(Task 1's README claimed 98.9% / 78.2% -- compare against this re-derivation)")

    table_bo = channel_table(scored_bo, "a_only", "b_only", "bare_oxide", "carbonate")
    print()
    print_table(table_bo, "bare_oxide", "carbonate")

    # Identify the channel with the largest mean-score gap (bare_oxide - carbonate),
    # ranked by ABSOLUTE magnitude -- a large gap that happens to run the
    # "wrong" sign is still the dominant channel, not a channel to bury at
    # the bottom of a signed sort (a mistake caught and fixed after the
    # ammonium-vs-H3PO4 comparison below showed exactly this).
    gaps = {}
    for ch in CHANNELS:
        a, b = table_bo[ch]["bare_oxide"], table_bo[ch]["carbonate"]
        if a["mean_score"] is not None and b["mean_score"] is not None:
            gaps[ch] = a["mean_score"] - b["mean_score"]
    top_channels = sorted(gaps.items(), key=lambda kv: -abs(kv[1]))
    print("\n-- channels ranked by |bare_oxide mean - carbonate mean|, largest gap first --")
    for ch, g in top_channels:
        print(f"  {ch:<24} gap={g:+.4f}")

    # CLEANED re-run: exclude any sample (either group) that also declares
    # an ammonium precursor, to isolate whether carbonate has an
    # independent stoichiometry effect once the Phase-13-diagnosed
    # ammonium/balance-solver confound is removed.
    print("\n-- CLEANED: same comparison, samples with an ammonium precursor excluded from both groups --")
    def clean_classifier(precursors):
        if any(is_ammonium(p) for p in precursors):
            return "neither"  # drop entirely, not "both"
        return classify(precursors, BARE_OXIDE, CARBONATE)
    scored_bo_clean, n_fail_bo_clean = score_all(doc, validator, BARE_OXIDE, CARBONATE,
                                                 custom_classifier=clean_classifier)
    print(f"counts (ammonium-free): bare-only={len(scored_bo_clean['a_only'])} "
          f"carbonate-only={len(scored_bo_clean['b_only'])}")
    pa_c, na_c = pass_rate(scored_bo_clean, "a_only")
    pb_c, nb_c = pass_rate(scored_bo_clean, "b_only")
    if pa_c is not None and pb_c is not None:
        print(f"pass@{BAR} rate (ammonium-free): bare-oxide-only = {pa_c:.1%} (n={na_c})  "
              f"carbonate-only = {pb_c:.1%} (n={nb_c})")
    table_bo_clean = channel_table(scored_bo_clean, "a_only", "b_only", "bare_oxide", "carbonate")
    print()
    print_table(table_bo_clean, "bare_oxide", "carbonate")
    gaps_clean = {}
    for ch in CHANNELS:
        a, b = table_bo_clean[ch]["bare_oxide"], table_bo_clean[ch]["carbonate"]
        if a["mean_score"] is not None and b["mean_score"] is not None:
            gaps_clean[ch] = a["mean_score"] - b["mean_score"]
    top_channels_clean = sorted(gaps_clean.items(), key=lambda kv: -abs(kv[1]))
    print("\n-- (ammonium-free) channels ranked by |gap|, largest first --")
    for ch, g in top_channels_clean:
        print(f"  {ch:<24} gap={g:+.4f}")

    top_channel = top_channels[0][0] if top_channels else None
    examples = find_examples(scored_bo, "b_only", top_channel, n=5) if top_channel else []
    print(f"\n-- 5 example carbonate routes failing top channel '{top_channel}' --")
    for ex in examples:
        print(f"  {ex['target']:<16} precursors={ex['precursors']}  "
              f"{top_channel}={ex['channel_value']:.3f}  gradeability={ex['gradeability']}  "
              f"overall_reward={ex['reward']:.3f}")

    print("\n" + "=" * 100)
    print("AMMONIUM-PHOSPHATE vs H3PO4 (base model, phosphate targets only)")
    print("=" * 100)

    def phosphate_target(t):
        import re
        return "P" in re.sub(r"Pb|Pr|Pd|Pt|Pm|Po|Pu", "", t)

    phosphate_doc = {"results": [r for r in doc["results"] if phosphate_target(r["target"])]}
    print(f"phosphate targets: {[r['target'] for r in phosphate_doc['results']]}")

    def ammon_classifier(precursors):
        has_a = any(is_ammonium(p) for p in precursors)
        has_b = any(p in H3PO4_SET for p in precursors)
        if has_a and has_b:
            return "both"
        if has_a:
            return "a_only"
        if has_b:
            return "b_only"
        return "neither"

    scored_p, n_fail_p = score_all(phosphate_doc, validator, set(), set(),
                                   custom_classifier=ammon_classifier)
    print(f"n_score_fail={n_fail_p}  counts: ammonium-only={len(scored_p['a_only'])} "
          f"H3PO4-only={len(scored_p['b_only'])} both={len(scored_p['both'])} "
          f"neither={len(scored_p['neither'])}")

    pa, na = pass_rate(scored_p, "a_only")
    pb, nb = pass_rate(scored_p, "b_only")
    if pa is not None and pb is not None:
        print(f"\npass@{BAR} rate: ammonium-phosphate-only = {pa:.1%} (n={na})  "
              f"H3PO4-only = {pb:.1%} (n={nb})")
    else:
        print(f"\ninsufficient samples for a clean pass-rate comparison "
              f"(ammonium n={na}, H3PO4 n={nb})")

    table_p = channel_table(scored_p, "a_only", "b_only", "ammonium_P", "H3PO4")
    print()
    print_table(table_p, "ammonium_P", "H3PO4")

    gaps_p = {}
    for ch in CHANNELS:
        a, b = table_p[ch]["ammonium_P"], table_p[ch]["H3PO4"]
        if a["mean_score"] is not None and b["mean_score"] is not None:
            gaps_p[ch] = a["mean_score"] - b["mean_score"]
    top_channels_p = sorted(gaps_p.items(), key=lambda kv: -abs(kv[1]))
    print("\n-- channels ranked by |ammonium_P mean - H3PO4 mean|, largest gap first --")
    for ch, g in top_channels_p:
        print(f"  {ch:<24} gap={g:+.4f}")
    top_channel_p = top_channels_p[0][0] if top_channels_p else None
    same_channel = (top_channel_p == top_channel)
    print(f"\nSame top channel as carbonate/bare-oxide (all samples)? "
          f"{'YES: ' + str(top_channel) if same_channel else f'NO ({top_channel} vs {top_channel_p})'}")

    same_channel_clean = (top_channels_clean and top_channels_clean[0][0] == top_channel_p)
    print(f"Same top channel as CLEANED carbonate/bare-oxide (ammonium excluded)? "
          f"{'YES: ' + str(top_channel_p) if same_channel_clean else 'NO'}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "bare_oxide_vs_carbonate_all_samples": {
            "pass_rate_bare_oxide": p_a, "n_bare_oxide": n_a,
            "pass_rate_carbonate": p_b, "n_carbonate": n_b,
            "n_carbonate_with_ammonium_precursor": n_carb_with_ammonium,
            "n_bare_oxide_with_ammonium_precursor": n_bare_with_ammonium,
            "channel_table": table_bo,
            "channel_gaps_sorted_by_abs": top_channels,
            "top_channel": top_channel,
            "examples_failing_top_channel": examples,
        },
        "bare_oxide_vs_carbonate_ammonium_excluded": {
            "pass_rate_bare_oxide": pa_c, "n_bare_oxide": na_c,
            "pass_rate_carbonate": pb_c, "n_carbonate": nb_c,
            "channel_table": table_bo_clean,
            "channel_gaps_sorted_by_abs": top_channels_clean,
        },
        "ammonium_vs_h3po4": {
            "pass_rate_ammonium": pa, "n_ammonium": na,
            "pass_rate_h3po4": pb, "n_h3po4": nb,
            "channel_table": table_p,
            "channel_gaps_sorted_by_abs": top_channels_p,
            "top_channel": top_channel_p,
            "same_top_channel_as_carbonate_all_samples": same_channel,
            "same_top_channel_as_carbonate_ammonium_excluded": same_channel_clean,
        },
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
