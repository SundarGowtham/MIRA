#!/usr/bin/env python
"""
research/phase12_astral_final_analysis.py — Phase 12's pre-registered
primary-endpoint readout: GDPO from RS-SFT, beta=0, checkpoint 300, vs the
RS-SFT baseline it was initialized from. Same methodology as the prior
three-model/five-model ASTRAL comparisons (research/build_astral_5model_
summary.py's summary schema; research/analyze_passk.py's exact McNemar on
discordant pairs).

Pre-registered thresholds (misc/PHASE13_PREREG.md "Phase 12 note";
originally set when Phase 12 was launched): ASTRAL predicted-set hits N/35
at n=32, checkpoint 300 (single pre-registered read, not resumed to 600 --
settled).
  >12/35 -> RL amplified support in RS-SFT, the project's first positive
            result.
  10-12/35 -> preserved but not amplified.
  <10/35 -> RL degrades even a good starting point.

Usage (tmux):
  uv run python research/phase12_astral_final_analysis.py
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

RS_SFT_PATH = Path("results/astral_gen_n32_rs_sft.json")
PHASE12_PATH = Path("misc/astral_gen_n32_gdpo_phase12_beta0.json")
PHASE12_RESULTS_COPY = Path("results/astral_gen_n32_gdpo_phase12_beta0.json")
OUT_JSON = Path("results/phase12_astral_final_analysis.json")

THRESHOLD_AMPLIFIED = 12  # >12/35
THRESHOLD_PRESERVED_LOW = 10  # 10-12/35 band


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial p-value on discordant pairs (b = A-only
    successes, c = B-only successes) -- identical to
    research/analyze_passk.py's implementation, reused for consistency."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def per_target_hits(doc: dict) -> dict[str, dict[str, bool]]:
    """target -> {"predicted": any sample PREDICTED/PREDICTED_SUPERSET,
    "traditional": any sample TRADITIONAL/TRADITIONAL_SUPERSET}."""
    out = {}
    for r in doc["results"]:
        matches = {s["match"] for s in r["samples"]}
        out[r["target"]] = {
            "predicted": bool(matches & {"PREDICTED", "PREDICTED_SUPERSET"}),
            "traditional": bool(matches & {"TRADITIONAL", "TRADITIONAL_SUPERSET"}),
        }
    return out


def summarize(doc: dict) -> dict:
    s = doc["summary"]
    results = doc["results"]
    total = sum(len(r["samples"]) for r in results)
    n_fail = sum(1 for r in results for x in r["samples"] if x["match"] == "PARSE_FAILURE")
    pred = s.get("mean_reward_predicted_matches")
    trad = s.get("mean_reward_traditional_matches")
    gap = (pred - trad) if (pred is not None and trad is not None) else None
    return {
        "n_targets": s["n_targets"],
        "any_predicted": s["n_targets_any_predicted"],
        "any_traditional": s["n_targets_any_traditional"],
        "mean_reward_all": s["mean_reward_all"],
        "mean_reward_predicted_matches": pred,
        "mean_reward_traditional_matches": trad,
        "validator_gap_predicted_minus_traditional": gap,
        "mean_max_T_C": s["mean_max_T"],
        "median_max_T_C": s["median_max_T"],
        "parse_rate": (total - n_fail) / total,
        "n_samples": total,
    }


def paired_mcnemar(hits_a: dict, hits_b: dict, key: str, label_a: str, label_b: str) -> dict:
    common = sorted(set(hits_a) & set(hits_b))
    a_only, b_only, both, neither = [], [], [], []
    for t in common:
        sa, sb = hits_a[t][key], hits_b[t][key]
        if sa and sb:
            both.append(t)
        elif sa:
            a_only.append(t)
        elif sb:
            b_only.append(t)
        else:
            neither.append(t)
    p = mcnemar(len(a_only), len(b_only))
    return {
        "n_targets": len(common),
        f"{label_a}_only": a_only, f"{label_a}_only_n": len(a_only),
        f"{label_b}_only": b_only, f"{label_b}_only_n": len(b_only),
        "both_n": len(both), "neither_n": len(neither),
        "mcnemar_p": p,
    }


def main():
    rs_sft = json.loads(RS_SFT_PATH.read_text())
    phase12 = json.loads(PHASE12_PATH.read_text())

    rs_sft_summary = summarize(rs_sft)
    phase12_summary = summarize(phase12)

    rs_sft_hits = per_target_hits(rs_sft)
    phase12_hits = per_target_hits(phase12)

    pred_mcnemar = paired_mcnemar(rs_sft_hits, phase12_hits, "predicted", "rs_sft", "phase12_beta0")
    trad_mcnemar = paired_mcnemar(rs_sft_hits, phase12_hits, "traditional", "rs_sft", "phase12_beta0")

    n_pred = phase12_summary["any_predicted"]
    if n_pred > THRESHOLD_AMPLIFIED:
        verdict = f"AMPLIFIED ({n_pred}/35 > {THRESHOLD_AMPLIFIED}) -- RL amplified what was in RS-SFT's support. First positive result."
    elif n_pred >= THRESHOLD_PRESERVED_LOW:
        verdict = f"PRESERVED, NOT AMPLIFIED ({n_pred}/35, in [{THRESHOLD_PRESERVED_LOW},{THRESHOLD_AMPLIFIED}])"
    else:
        verdict = f"INTRINSIC COLLAPSE ({n_pred}/35 < {THRESHOLD_PRESERVED_LOW})"

    print("=" * 100)
    print("PHASE 12 PRE-REGISTERED PRIMARY ENDPOINT -- checkpoint 300, n=32, settled read")
    print("=" * 100)
    print(f"{'':30s}{'RS-SFT (init)':>18s}{'GDPO-phase12-beta0':>22s}")
    print(f"{'predicted hits':30s}{rs_sft_summary['any_predicted']:>18d}{phase12_summary['any_predicted']:>22d}")
    print(f"{'conventional hits':30s}{rs_sft_summary['any_traditional']:>18d}{phase12_summary['any_traditional']:>22d}")
    print(f"{'mean reward (all)':30s}{rs_sft_summary['mean_reward_all']:>18.3f}{phase12_summary['mean_reward_all']:>22.3f}")
    print(f"{'validator gap (pred-trad)':30s}{rs_sft_summary['validator_gap_predicted_minus_traditional']:>18.3f}"
          f"{phase12_summary['validator_gap_predicted_minus_traditional']:>22.3f}")
    print(f"{'mean max-T (C)':30s}{rs_sft_summary['mean_max_T_C']:>18.1f}{phase12_summary['mean_max_T_C']:>22.1f}")
    print(f"{'parse rate':30s}{rs_sft_summary['parse_rate']:>18.1%}{phase12_summary['parse_rate']:>22.1%}")

    print(f"\n*** VERDICT: {verdict} ***")

    print(f"\nMcNemar (predicted-hit, RS-SFT vs GDPO-phase12-beta0):")
    print(f"  RS-SFT-only: {pred_mcnemar['rs_sft_only_n']} {pred_mcnemar['rs_sft_only']}")
    print(f"  phase12_beta0-only: {pred_mcnemar['phase12_beta0_only_n']} {pred_mcnemar['phase12_beta0_only']}")
    print(f"  both: {pred_mcnemar['both_n']}  neither: {pred_mcnemar['neither_n']}")
    print(f"  exact McNemar p = {pred_mcnemar['mcnemar_p']:.4g}")

    print(f"\nMcNemar (conventional-hit, RS-SFT vs GDPO-phase12-beta0):")
    print(f"  RS-SFT-only: {trad_mcnemar['rs_sft_only_n']} {trad_mcnemar['rs_sft_only']}")
    print(f"  phase12_beta0-only: {trad_mcnemar['phase12_beta0_only_n']} {trad_mcnemar['phase12_beta0_only']}")
    print(f"  both: {trad_mcnemar['both_n']}  neither: {trad_mcnemar['neither_n']}")
    print(f"  exact McNemar p = {trad_mcnemar['mcnemar_p']:.4g}")

    # Copy the raw generation dump into results/ (tracked), matching the
    # existing astral_gen_n32_*.json convention -- per the outstanding
    # user request to move this out of gitignored misc/ once done.
    PHASE12_RESULTS_COPY.write_text(PHASE12_PATH.read_text())
    print(f"\ncopied raw dump -> {PHASE12_RESULTS_COPY}")

    OUT_JSON.write_text(json.dumps({
        "protocol": "35 ASTRAL targets x 32 closed-book samples, checkpoint 300, "
                    "pre-registered settled read (misc/PHASE13_PREREG.md 'Phase 12 note')",
        "rs_sft_baseline": rs_sft_summary,
        "phase12_gdpo_beta0_ckpt300": phase12_summary,
        "verdict": verdict,
        "thresholds": {"amplified_gt": THRESHOLD_AMPLIFIED, "preserved_low": THRESHOLD_PRESERVED_LOW},
        "mcnemar_predicted_hit": pred_mcnemar,
        "mcnemar_conventional_hit": trad_mcnemar,
    }, indent=1, default=str))
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
