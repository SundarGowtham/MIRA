#!/usr/bin/env python
"""
research/phase16_e1_support_growth.py — Phase 16, prediction 1's
support-growth half (docs/phases/PHASE16_PREREG.md, verbatim):

  "Support growth of model M relative to reference R, on a target set:
  the share of M's distinct precursor sets (32 samples per target) that
  are absent from R's 200-sample run on the same targets."
  "Resampling rate of R: the same quantity computed for a fresh 32-sample
  draw from R itself. This is the noise floor. Support growth counts only
  beyond it."

M = E1 (checkpoint runs/gdpo-qlora-gdpo-e1-ranker-rssft/final).
R = RS-SFT (runs/sft-qlora-rs-sft-from-base/final), already drawn:
    data/novelty/rssft_reference_{astral,heldout}.jsonl, sample_group
    "reference" (200/target) and "resampling_rate" (32/target).

Reported SEPARATELY for the ASTRAL targets (35) and the held-out split
(40), per the pre-registration. Pooled aggregation: numerator = sum over
targets of (# of M's distinct precursor sets for that target absent
from R's 200-sample reference set for that target); denominator = sum
over targets of (# of M's distinct precursor sets for that target).
Parse failures are dropped before counting distinct sets (consistent
with every other Phase 16 script's convention).

Usage:
  uv run python research/phase16_e1_support_growth.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validator import SynthesisValidator  # noqa: E402
from core.reward import parse_completion, ParseFailure  # noqa: E402

E1_ASTRAL_PATH = REPO_ROOT / "results" / "astral_gen_n32_e1.json"
E1_HELDOUT_PATH = REPO_ROOT / "data" / "novelty" / "e1_heldout_n32.jsonl"
RSSFT_ASTRAL_PATH = REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl"
RSSFT_HELDOUT_PATH = REPO_ROOT / "data" / "novelty" / "rssft_reference_heldout.jsonl"
OUT_PATH = REPO_ROOT / "results" / "phase16" / "e1_support_growth.json"


def precset_of(completion: str, target_formula: str) -> tuple | None:
    try:
        route = parse_completion(completion, target_formula)
    except ParseFailure:
        return None
    if not route.precursors:
        return None
    return tuple(sorted({SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}))


def load_e1_astral() -> dict[str, list[str]]:
    doc = json.loads(E1_ASTRAL_PATH.read_text())
    out: dict[str, list[str]] = defaultdict(list)
    for r in doc["results"]:
        for s in r["samples"]:
            precs = s.get("precursors")
            if precs:
                out[r["target"]].append(tuple(sorted(precs)))
    return out


def load_jsonl_precsets(path: Path, target_filter=None) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if target_filter is not None and not target_filter(rec):
                continue
            ps = precset_of(rec["completion"], rec["target"])
            if ps is not None:
                out[rec["target"]].append(ps)
    return out


def load_e1_heldout() -> dict[str, list[tuple]]:
    return load_jsonl_precsets(E1_HELDOUT_PATH)


def load_rssft(path: Path, group: str) -> dict[str, list[tuple]]:
    return load_jsonl_precsets(path, target_filter=lambda rec: rec.get("sample_group") == group)


def support_growth(m_sets_by_target: dict[str, list[tuple]], r_sets_by_target: dict[str, set]) -> dict:
    num, den = 0, 0
    per_target = {}
    for target, m_sets in m_sets_by_target.items():
        distinct_m = set(m_sets)
        r_sets = r_sets_by_target.get(target, set())
        absent = distinct_m - r_sets
        num += len(absent)
        den += len(distinct_m)
        per_target[target] = {
            "n_distinct_m": len(distinct_m),
            "n_absent_from_r": len(absent),
            "share": (len(absent) / len(distinct_m)) if distinct_m else None,
        }
    return {
        "n_targets": len(m_sets_by_target),
        "numerator_absent": num,
        "denominator_distinct": den,
        "pooled_share": (num / den) if den else None,
        "per_target": per_target,
    }


def main():
    report = {}

    print("=== ASTRAL targets (35) ===")
    e1_astral = load_e1_astral()
    rssft_astral_ref = load_rssft(RSSFT_ASTRAL_PATH, "reference")
    rssft_astral_resamp = load_rssft(RSSFT_ASTRAL_PATH, "resampling_rate")
    rssft_astral_ref_sets = {t: set(v) for t, v in rssft_astral_ref.items()}

    e1_growth_astral = support_growth(e1_astral, rssft_astral_ref_sets)
    rssft_noise_astral = support_growth(rssft_astral_resamp, rssft_astral_ref_sets)
    print(f"E1 support growth vs RS-SFT reference: {e1_growth_astral['pooled_share']:.4f} "
          f"({e1_growth_astral['numerator_absent']}/{e1_growth_astral['denominator_distinct']}, "
          f"{e1_growth_astral['n_targets']} targets)")
    print(f"RS-SFT resampling rate (noise floor): {rssft_noise_astral['pooled_share']:.4f} "
          f"({rssft_noise_astral['numerator_absent']}/{rssft_noise_astral['denominator_distinct']}, "
          f"{rssft_noise_astral['n_targets']} targets)")
    report["astral"] = {"e1_support_growth": e1_growth_astral, "rssft_resampling_noise_floor": rssft_noise_astral}

    print("\n=== Held-out targets (40) ===")
    e1_heldout = load_e1_heldout()
    rssft_heldout_ref = load_rssft(RSSFT_HELDOUT_PATH, "reference")
    rssft_heldout_resamp = load_rssft(RSSFT_HELDOUT_PATH, "resampling_rate")
    rssft_heldout_ref_sets = {t: set(v) for t, v in rssft_heldout_ref.items()}

    e1_growth_heldout = support_growth(e1_heldout, rssft_heldout_ref_sets)
    rssft_noise_heldout = support_growth(rssft_heldout_resamp, rssft_heldout_ref_sets)
    print(f"E1 support growth vs RS-SFT reference: {e1_growth_heldout['pooled_share']:.4f} "
          f"({e1_growth_heldout['numerator_absent']}/{e1_growth_heldout['denominator_distinct']}, "
          f"{e1_growth_heldout['n_targets']} targets)")
    print(f"RS-SFT resampling rate (noise floor): {rssft_noise_heldout['pooled_share']:.4f} "
          f"({rssft_noise_heldout['numerator_absent']}/{rssft_noise_heldout['denominator_distinct']}, "
          f"{rssft_noise_heldout['n_targets']} targets)")
    report["heldout"] = {"e1_support_growth": e1_growth_heldout, "rssft_resampling_noise_floor": rssft_noise_heldout}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
