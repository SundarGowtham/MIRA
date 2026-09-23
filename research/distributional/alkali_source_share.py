#!/usr/bin/env python
"""
research/distributional/alkali_source_share.py — Phase 15 Task 2b(a),
CORRECTED per regular Claude's review: the original 8.6%-vs-6.4% "bare
alkali oxide share of all samples" comparison crossed prompt
distributions (ASTRAL's 35 targets vs data/rl's 400-target universe RS-SFT
was built from) -- some targets contain no alkali metal at all and
couldn't source one via either route, diluting the comparison in ways
unrelated to any real behavioral shift.

Corrected metric: restricted to targets that actually contain an alkali
metal (Li, Na, K, Rb, Cs), for each (target, sample, alkali-element)
triple, classify how that specific alkali element is sourced in that
sample: bare_oxide / carbonate / other. Reported as a fraction of these
triples (a target needing 2 alkali elements contributes 2 triples -- an
intentional, documented choice, not hidden).

Four corpora:
  1. base-on-ASTRAL       (results/astral_gen_n32_base.json)
  2. RS-SFT training set  (data/rs_sft/rs_sft_train.jsonl + rs_sft_val.jsonl, 295 total)
  3. RS-SFT-on-ASTRAL     (results/astral_gen_n32_rs_sft.json)
  4. GDPO-on-ASTRAL       (results/astral_gen_n32_gdpo_phase12_beta0.json)

Also reports the fraction of RS-SFT's 295 training targets that contain
an alkali metal at all (determines whether the training set had enough
alkali-relevant exposure to teach a preference in the first place).

Usage (tmux):
  uv run python research/distributional/alkali_source_share.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pymatgen.core import Composition  # noqa: E402
from core.reward import parse_completion  # noqa: E402

ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
BARE_OXIDE = {"Li2O": "Li", "Na2O": "Na", "K2O": "K", "Rb2O": "Rb", "Cs2O": "Cs"}
CARBONATE = {"Li2CO3": "Li", "Na2CO3": "Na", "K2CO3": "K", "Rb2CO3": "Rb", "Cs2CO3": "Cs"}

ASTRAL_BASE = Path("results/astral_gen_n32_base.json")
ASTRAL_RSSFT = Path("results/astral_gen_n32_rs_sft.json")
ASTRAL_GDPO = Path("results/astral_gen_n32_gdpo_phase12_beta0.json")
RS_SFT_TRAIN = Path("data/rs_sft/rs_sft_train.jsonl")
RS_SFT_VAL = Path("data/rs_sft/rs_sft_val.jsonl")
OUT_JSON = Path("results/distributional/alkali_source_share.json")


def target_alkalis(target: str) -> set[str]:
    try:
        els = {str(e) for e in Composition(target).elements}
    except Exception:
        return set()
    return els & ALKALI


def classify_alkali_source(precursors: list[str], alkali_el: str) -> str:
    """For one alkali element needed by the target, what kind of precursor
    (among those declared) supplies it? bare_oxide / carbonate / other /
    missing (no declared precursor contains this element at all)."""
    for p in precursors:
        if p in BARE_OXIDE and BARE_OXIDE[p] == alkali_el:
            return "bare_oxide"
    for p in precursors:
        if p in CARBONATE and CARBONATE[p] == alkali_el:
            return "carbonate"
    for p in precursors:
        try:
            p_els = {str(e) for e in Composition(p).elements}
        except Exception:
            continue
        if alkali_el in p_els:
            return "other"
    return "missing"


def tally_astral_dump(path: Path) -> dict:
    doc = json.loads(path.read_text())
    counts = {"bare_oxide": 0, "carbonate": 0, "other": 0, "missing": 0}
    n_triples = 0
    for r in doc["results"]:
        target = r["target"]
        alkalis = target_alkalis(target)
        if not alkalis:
            continue
        for s in r["samples"]:
            precursors = s.get("precursors")
            if not precursors:
                continue
            for el in alkalis:
                cat = classify_alkali_source(precursors, el)
                counts[cat] += 1
                n_triples += 1
    return {"counts": counts, "n_triples": n_triples,
           "shares": {k: (v / n_triples if n_triples else None) for k, v in counts.items()}}


def tally_rssft_training_set() -> dict:
    counts = {"bare_oxide": 0, "carbonate": 0, "other": 0, "missing": 0}
    n_triples = 0
    targets_seen = set()
    targets_with_alkali = set()
    n_parse_fail = 0
    for path in [RS_SFT_TRAIN, RS_SFT_VAL]:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target = rec.get("target")
                targets_seen.add(target)
                alkalis = target_alkalis(target)
                if not alkalis:
                    continue
                targets_with_alkali.add(target)
                completion = rec.get("completion", "")
                try:
                    route = parse_completion(completion, target)
                except Exception:
                    n_parse_fail += 1
                    continue
                precursors = [p.formula for p in (route.precursors or [])]
                for el in alkalis:
                    cat = classify_alkali_source(precursors, el)
                    counts[cat] += 1
                    n_triples += 1
    return {
        "counts": counts, "n_triples": n_triples,
        "shares": {k: (v / n_triples if n_triples else None) for k, v in counts.items()},
        "n_training_examples": len(targets_seen) if False else None,  # examples, not targets -- see below
        "n_distinct_targets": len(targets_seen),
        "n_distinct_targets_with_alkali": len(targets_with_alkali),
        "frac_targets_with_alkali": len(targets_with_alkali) / len(targets_seen) if targets_seen else None,
        "n_parse_fail": n_parse_fail,
    }


def n_rssft_examples() -> int:
    n = 0
    for path in [RS_SFT_TRAIN, RS_SFT_VAL]:
        if path.exists():
            n += sum(1 for _ in path.open())
    return n


def print_corpus(name: str, result: dict):
    print(f"\n{name}: n_triples={result['n_triples']}")
    for cat in ["bare_oxide", "carbonate", "other", "missing"]:
        share = result["shares"][cat]
        n = result["counts"][cat]
        print(f"  {cat:<12} {n:5d}  ({share:.1%})" if share is not None else f"  {cat:<12} 0")


def main():
    n_examples = n_rssft_examples()
    print(f"RS-SFT trained on {n_examples} examples "
          f"({RS_SFT_TRAIN.name}+{RS_SFT_VAL.name})")

    base_result = tally_astral_dump(ASTRAL_BASE)
    rssft_astral_result = tally_astral_dump(ASTRAL_RSSFT)
    gdpo_result = tally_astral_dump(ASTRAL_GDPO)
    rssft_train_result = tally_rssft_training_set()

    print_corpus("1. base-on-ASTRAL", base_result)
    print_corpus("2. RS-SFT training set", rssft_train_result)
    print_corpus("3. RS-SFT-on-ASTRAL", rssft_astral_result)
    print_corpus("4. GDPO-on-ASTRAL", gdpo_result)

    print(f"\nRS-SFT training set: {rssft_train_result['n_distinct_targets_with_alkali']}/"
          f"{rssft_train_result['n_distinct_targets']} distinct targets "
          f"({rssft_train_result['frac_targets_with_alkali']:.1%}) contain an alkali metal "
          f"({rssft_train_result['n_parse_fail']} parse failures excluded)")

    OUT_JSON.write_text(json.dumps({
        "n_rssft_training_examples": n_examples,
        "base_on_astral": base_result,
        "rs_sft_training_set": rssft_train_result,
        "rs_sft_on_astral": rssft_astral_result,
        "gdpo_on_astral": gdpo_result,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
