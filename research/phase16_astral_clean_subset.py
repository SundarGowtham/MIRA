#!/usr/bin/env python
"""
research/phase16_astral_clean_subset.py — Phase 16, "Decisions after
pass 2" item 3 (misc/PHASE16_INSTRUCTIONS.md): recompute every ASTRAL
headline number on the 35 targets minus the union of all overlapping
targets (results/phase16/astral_overlap.json), side by side with the
full-35 number. No new GPU work -- reuses the existing n=32 generation
dumps.

Recomputes:
  1. Rule-picked (predicted-set) hits: base, full SFT, RS-SFT, Phase 12
     GDPO-beta0.
  2. base -> full SFT erasure (the "10->0/1" finding) + exact McNemar.
  3. RS-SFT -> Phase 12 GDPO amplification (the "10->14" finding) +
     one-sided exact McNemar (matching finding 22's own posture).
  4. Carbonate/bare-oxide/other alkali-source shares, each model.

Usage:
  uv run python research/phase16_astral_clean_subset.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research" / "distributional"))

from alkali_source_share import target_alkalis, classify_alkali_source  # noqa: E402

OVERLAP_PATH = REPO_ROOT / "results" / "phase16" / "astral_overlap.json"
GEN_FILES = {
    "base": REPO_ROOT / "results" / "astral_gen_n32_base.json",
    "full_sft": REPO_ROOT / "results" / "astral_gen_n32_sft.json",
    "rs_sft": REPO_ROOT / "results" / "astral_gen_n32_rs_sft.json",
    "gdpo_phase12_beta0": REPO_ROOT / "results" / "astral_gen_n32_gdpo_phase12_beta0.json",
}
OUT_PATH = REPO_ROOT / "results" / "phase16" / "astral_clean_subset.json"


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial p-value on discordant pairs (research/analyze_passk.py)."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar_one_sided(b: int, c: int) -> float:
    """One-sided exact binomial, matching finding 22's posture when all
    discordant pairs favor the same side: P(X<=min(b,c) | n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    return sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)


def load_hits(path: Path) -> dict[str, bool]:
    """target -> True if any of its 32 samples matched PREDICTED."""
    doc = json.loads(path.read_text())
    return {r["target"]: any(s.get("match") == "PREDICTED" for s in r["samples"])
            for r in doc["results"]}


def load_carbonate_shares(path: Path, targets: set[str]) -> dict:
    doc = json.loads(path.read_text())
    counts = {"bare_oxide": 0, "carbonate": 0, "other": 0, "missing": 0}
    n_triples = 0
    for r in doc["results"]:
        if r["target"] not in targets:
            continue
        alkalis = target_alkalis(r["target"])
        if not alkalis:
            continue
        for s in r["samples"]:
            precursors = s.get("precursors")
            if not precursors:
                continue
            for el in alkalis:
                counts[classify_alkali_source(precursors, el)] += 1
                n_triples += 1
    return {"counts": counts, "n_triples": n_triples,
            "shares": {k: round(v / n_triples, 4) if n_triples else None
                      for k, v in counts.items()}}


def main():
    overlap = json.loads(OVERLAP_PATH.read_text())
    overlapping = set(overlap["union_of_all_overlapping_targets"])
    all_hits = {name: load_hits(path) for name, path in GEN_FILES.items()}
    all_targets = set(next(iter(all_hits.values())).keys())
    clean_targets = all_targets - overlapping

    print(f"full-35: {len(all_targets)} targets. overlapping: {len(overlapping)} "
          f"({sorted(overlapping)}). clean subset: {len(clean_targets)}.")

    report: dict = {"overlapping_targets": sorted(overlapping),
                    "n_full": len(all_targets), "n_clean": len(clean_targets),
                    "hits": {}, "erasure_mcnemar": {}, "amplification_mcnemar": {},
                    "carbonate_shares": {}}

    print("\n=== 1. Rule-picked hits, full-35 vs clean-subset ===")
    for name, hits in all_hits.items():
        full_n = sum(hits.values())
        clean_n = sum(1 for t in clean_targets if hits.get(t))
        report["hits"][name] = {"full_35": f"{full_n}/{len(all_targets)}",
                                "clean_subset": f"{clean_n}/{len(clean_targets)}"}
        print(f"  {name}: full {full_n}/{len(all_targets)}  clean {clean_n}/{len(clean_targets)}")

    print("\n=== 2. base -> full SFT erasure ===")
    for subset_name, subset in [("full_35", all_targets), ("clean_subset", clean_targets)]:
        base_hits = all_hits["base"]
        sft_hits = all_hits["full_sft"]
        b = sum(1 for t in subset if base_hits.get(t) and not sft_hits.get(t))
        c = sum(1 for t in subset if sft_hits.get(t) and not base_hits.get(t))
        p = mcnemar(b, c)
        report["erasure_mcnemar"][subset_name] = {"b_base_only": b, "c_sft_only": c, "p_two_sided": p}
        print(f"  {subset_name}: base-only={b} sft-only={c} McNemar p={p:.4f}")

    print("\n=== 3. RS-SFT -> Phase 12 GDPO amplification ===")
    for subset_name, subset in [("full_35", all_targets), ("clean_subset", clean_targets)]:
        rssft_hits = all_hits["rs_sft"]
        gdpo_hits = all_hits["gdpo_phase12_beta0"]
        b = sum(1 for t in subset if rssft_hits.get(t) and not gdpo_hits.get(t))
        c = sum(1 for t in subset if gdpo_hits.get(t) and not rssft_hits.get(t))
        p_two = mcnemar(b, c)
        p_one = mcnemar_one_sided(b, c) if (b == 0 or c == 0) else None
        gained = sorted(t for t in subset if gdpo_hits.get(t) and not rssft_hits.get(t))
        lost = sorted(t for t in subset if rssft_hits.get(t) and not gdpo_hits.get(t))
        report["amplification_mcnemar"][subset_name] = {
            "b_rssft_only": b, "c_gdpo_only": c, "p_two_sided": p_two,
            "p_one_sided_if_unidirectional": p_one,
            "gdpo_gained": gained, "rssft_lost": lost,
        }
        print(f"  {subset_name}: rs_sft-only={b} gdpo-only={c} gained={gained} lost={lost} "
              f"McNemar p(two-sided)={p_two:.4f}" + (f" p(one-sided)={p_one:.4f}" if p_one is not None else ""))

    print("\n=== 4. Carbonate/bare-oxide/other alkali-source shares ===")
    for name, path in GEN_FILES.items():
        full_share = load_carbonate_shares(path, all_targets)
        clean_share = load_carbonate_shares(path, clean_targets)
        report["carbonate_shares"][name] = {"full_35": full_share, "clean_subset": clean_share}
        print(f"  {name}: full {full_share['shares']}  clean {clean_share['shares']}")

    OUT_PATH.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
