#!/usr/bin/env python
"""
research/test_rssft_ammonium_propagation.py — tests regular Claude's
propagation hypothesis on the ammonium-precursor balance-solver bug
(misc/PHASE13_RESULTS.md iteration 2 + the historical audit): if RS-SFT's
bar-0.9 filter selected against routes the bug was silently zeroing
(`stoichiometry`/`amount_accuracy` both key off `_find_balanced_reaction`),
ammonium-precursor prevalence should have COLLAPSED between the base
model's raw, unfiltered generations and the RS-SFT survivor set kept for
training.

Detection method, corrected from a first pass that used substring
matching on formula text: `results/astral_gen_n32_base.json` stores some
precursor formulas in a fully-expanded elemental (Hill-like) form --
`NH4H2PO4`'s own reduced formula is `PH6NO4` via pymatgen, which contains
neither "NH4" nor "NH3" as a substring, so text matching silently misses
it. Detection here instead parses each precursor with pymatgen
`Composition` and checks whether BOTH nitrogen and hydrogen are present
-- a chemically reliable proxy for "ammonium-type precursor" in this
domain, since no other common inorganic solid-state precursor class
combines N and H (hydroxides have H, no N; nitrates have N, no H;
carbonates have neither). Applied consistently to both sources below.

Data-availability caveat, stated up front: `data_curation/
build_rs_sft_dataset.py` only ever persists SURVIVORS
(`data/rs_sft/rs_sft_train.jsonl` + `rs_sft_val.jsonl`, 295 total) --
rejected samples were never dumped, so the exact pre-filter distribution
for RS-SFT's own 400-target run cannot be reconstructed from anything on
disk. The closest available proxy for "base model, unfiltered" is
`results/astral_gen_n32_base.json` (35 ASTRAL targets x 32 samples = 1120
completions, same base checkpoint, same closed-book generation
methodology, DIFFERENT target universe from RS-SFT's `data/rl` sample).
This is a real limitation, reported as such -- a proxy test, not a
matched pre/post-filter comparison.

Usage (tmux):
  uv run python research/test_rssft_ammonium_propagation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymatgen.core import Composition, Element  # noqa: E402

from core.reward import ParseFailure, parse_completion  # noqa: E402

BASE_UNFILTERED = Path("results/astral_gen_n32_base.json")
RS_SFT_TRAIN = Path("data/rs_sft/rs_sft_train.jsonl")
RS_SFT_VAL = Path("data/rs_sft/rs_sft_val.jsonl")
OUT_JSON = Path("results/rssft_ammonium_propagation_test.json")


def is_ammonium_like(formula: str) -> bool:
    try:
        els = {str(e) for e in Composition(formula).elements}
    except Exception:
        return False
    return "N" in els and "H" in els


def scan_base_unfiltered():
    d = json.loads(BASE_UNFILTERED.read_text())
    n_total, n_mentioned = 0, 0
    for r in d["results"]:
        for s in r["samples"]:
            precursors = s.get("precursors") or []
            n_total += 1
            if any(is_ammonium_like(p) for p in precursors):
                n_mentioned += 1
    return n_total, n_mentioned


def scan_rs_sft():
    n_total, n_mentioned, n_parse_fail = 0, 0, 0
    for path in [RS_SFT_TRAIN, RS_SFT_VAL]:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_total += 1
                target = rec.get("target")
                completion = rec.get("completion", "")
                try:
                    route = parse_completion(completion, target)
                except Exception:
                    n_parse_fail += 1
                    continue
                formulas = [p.formula for p in (route.precursors or [])]
                if any(is_ammonium_like(f) for f in formulas):
                    n_mentioned += 1
    return n_total, n_mentioned, n_parse_fail


def main():
    base_total, base_mentioned = scan_base_unfiltered()
    rssft_total, rssft_mentioned, rssft_parse_fail = scan_rs_sft()

    base_pct = 100 * base_mentioned / base_total if base_total else None
    rssft_pct = 100 * rssft_mentioned / rssft_total if rssft_total else None

    print("=" * 90)
    print("RS-SFT ammonium-precursor propagation test (proxy comparison -- see caveat)")
    print("=" * 90)
    print(f"base model, unfiltered (ASTRAL n=32 proxy, DIFFERENT target universe):")
    print(f"  {base_mentioned}/{base_total} = {base_pct:.2f}% have an N+H (ammonium-like) precursor")
    print(f"RS-SFT survivor set (bar=0.9, data/rl universe, {rssft_parse_fail} parse failures excluded):")
    print(f"  {rssft_mentioned}/{rssft_total} = {rssft_pct:.2f}% have an N+H (ammonium-like) precursor")

    verdict = None
    if base_pct and rssft_pct is not None:
        ratio = rssft_pct / base_pct if base_pct else None
        print(f"\nratio (RS-SFT / base) = {ratio:.2f}" if ratio is not None else "")
        if ratio is not None and ratio < 0.5:
            verdict = "CONSISTENT with the propagation hypothesis: prevalence roughly halved or more."
        elif ratio is not None:
            verdict = "NOT clearly consistent with the propagation hypothesis at this sample size/proxy."
        print(verdict)

    print("\nCAVEAT: different target universes (ASTRAL's 35 vs RS-SFT's data/rl 400) -- "
          "this is a proxy test, not a matched pre/post-filter comparison, because "
          "build_rs_sft_dataset.py never persisted rejected (non-survivor) samples.")

    OUT_JSON.write_text(json.dumps({
        "detection_method": "pymatgen Composition: precursor contains both N and H elements",
        "base_unfiltered": {"source": str(BASE_UNFILTERED), "n_total": base_total,
                            "n_mentioned": base_mentioned, "pct": base_pct},
        "rs_sft_survivors": {"source": [str(RS_SFT_TRAIN), str(RS_SFT_VAL)],
                             "n_total": rssft_total, "n_mentioned": rssft_mentioned,
                             "n_parse_fail": rssft_parse_fail, "pct": rssft_pct},
        "verdict": verdict,
        "caveat": "Different target universes (ASTRAL 35 vs RS-SFT's data/rl 400); "
                  "build_rs_sft_dataset.py never persisted rejected samples, so no "
                  "matched pre/post-filter comparison exists on disk. This is the "
                  "best available proxy, not a clean test.",
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
