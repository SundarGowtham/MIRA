#!/usr/bin/env python
"""
research/audit_ammonium_precursor_history.py — Phase 13 iteration 2 side
investigation (misc/PHASE13_PREREG.md addendum 2, per regular Claude's
review): how often, across ALL prior RL training generation dumps (not
just the ASTRAL comparison), did the MODEL ITSELF propose an
ammonium-containing precursor and get silently scored zero everywhere by
validator.py's `_find_balanced_reaction` candidate-set gap
(misc/PHASE13_RESULTS.md iteration 1 diagnosis)?

Every one of these dumps comes from a GDPO run trained against
validator.py's own reward (`stoichiometry`, `amount_accuracy`, and every
thermo-aware check all key off `_find_balanced_reaction` finding a
balance) -- so if the model proposed an ammonium salt on its own during
training, not just in the curated ASTRAL literature routes, the exact
same gap would have zeroed those checks for that completion, for a
software reason unrelated to whether the route was actually sound.

Method: for each dump, text-scan the raw completion for ammonium-formula
patterns (NH4, (NH4)x, NH3-derived salts written as ammonium precursors)
-- cheap, no parsing needed for the prevalence count. For a random sample
of matches, actually parse and gate-check BOTH the original (unfixed)
validator.py behavior and the comparator's fixed behavior, to confirm the
mechanism generalizes beyond NH4H2PO4 specifically and to quantify how
often the bug actually bit (vs. the ammonium precursor appearing in a
route that fails to balance for an unrelated, legitimate reason).

Usage (tmux):
  uv run python research/audit_ammonium_precursor_history.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator import SynthesisValidator, ThermoChecker  # noqa: E402
from core.comparator import _ComparatorValidator  # noqa: E402
from core.reward import ParseFailure, parse_completion  # noqa: E402

DUMPS = [
    Path("runs/gdpo-qlora-gdpo-v3/generations.jsonl"),
    Path("runs/gdpo-qlora-gdpo-v4/generations.jsonl"),
    Path("runs/gdpo-qlora-beta-ablation-probe/generations.jsonl"),
    Path("runs/gdpo-qlora-gdpo-phase12-smoke/generations.jsonl"),
    Path("runs/gdpo-qlora-gdpo-phase12-rssft/generations.jsonl"),
    Path("runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl"),
    Path("runs/gdpo-qlora-gdpo-run4-ranker/generations.jsonl"),
]

AMMONIUM_PATTERN = re.compile(r"NH4|\(NH4\)|NH3\s*[·.]")
SEED = 20260918
SAMPLE_N = 200  # per-dump cap for the parse+gate-check confirmation pass
OUT_PATH = Path("results/ammonium_precursor_history_audit.json")


def scan_dump(path: Path) -> dict:
    n_total = 0
    n_ammonium_mentioned = 0
    ammonium_records = []
    with path.open() as f:
        for line in f:
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            completion = rec.get("completion", "")
            if AMMONIUM_PATTERN.search(completion):
                n_ammonium_mentioned += 1
                ammonium_records.append(rec)
    return {
        "n_total": n_total,
        "n_ammonium_mentioned": n_ammonium_mentioned,
        "pct": round(100 * n_ammonium_mentioned / n_total, 2) if n_total else None,
        "records": ammonium_records,
    }


def confirm_bug_impact(records: list[dict], rng: random.Random,
                        original_validator: SynthesisValidator,
                        fixed_validator: _ComparatorValidator) -> dict:
    sample = rng.sample(records, min(SAMPLE_N, len(records)))
    n_parsed = 0
    n_no_ammonium_precursor = 0  # mentioned in reasoning text but not an actual precursor
    n_original_balances_fail = 0
    n_fixed_balances_pass = 0  # among original failures, does the fix recover them?
    n_original_pass_already = 0
    for rec in sample:
        target = rec.get("target")
        completion = rec.get("completion")
        if not target or not completion:
            continue
        try:
            route = parse_completion(completion, target)
        except Exception:
            continue
        n_parsed += 1
        formulas = {p.formula for p in (route.precursors or [])}
        if not any("NH4" in f or "NH3" in f for f in formulas):
            n_no_ammonium_precursor += 1
            continue
        try:
            orig_reaction, _ = original_validator._find_balanced_reaction(route)
        except Exception:
            orig_reaction = None
        if orig_reaction is None:
            n_original_balances_fail += 1
            try:
                fixed_reaction, _ = fixed_validator._find_balanced_reaction(route)
            except Exception:
                fixed_reaction = None
            if fixed_reaction is not None:
                n_fixed_balances_pass += 1
        else:
            n_original_pass_already += 1
    return {
        "sample_size": len(sample),
        "n_parsed": n_parsed,
        "n_no_ammonium_precursor_declared": n_no_ammonium_precursor,
        "n_original_balances_fail": n_original_balances_fail,
        "n_original_pass_already": n_original_pass_already,
        "n_fixed_recovers": n_fixed_balances_pass,
        "recovery_rate_of_failures": (
            round(n_fixed_balances_pass / n_original_balances_fail, 3)
            if n_original_balances_fail else None
        ),
    }


def main():
    rng = random.Random(SEED)
    original_validator = SynthesisValidator(mp_formula_set=set(), thermo_checker=None)
    fixed_validator = _ComparatorValidator(mp_formula_set=set(), thermo_checker=None)

    results = {}
    for dump in DUMPS:
        if not dump.exists():
            print(f"MISSING (skipped): {dump}")
            continue
        print(f"scanning {dump}...", flush=True)
        scan = scan_dump(dump)
        impact = confirm_bug_impact(scan["records"], rng, original_validator, fixed_validator)
        results[str(dump)] = {
            "n_total": scan["n_total"],
            "n_ammonium_mentioned": scan["n_ammonium_mentioned"],
            "pct_ammonium_mentioned": scan["pct"],
            "impact_sample": impact,
        }
        print(f"  n_total={scan['n_total']} ammonium_mentioned={scan['n_ammonium_mentioned']} "
              f"({scan['pct']}%)  impact_sample={impact}", flush=True)

    total_completions = sum(r["n_total"] for r in results.values())
    total_ammonium = sum(r["n_ammonium_mentioned"] for r in results.values())
    total_orig_fail = sum(r["impact_sample"]["n_original_balances_fail"] for r in results.values())
    total_fixed_recover = sum(r["impact_sample"]["n_fixed_recovers"] for r in results.values())

    print("\n" + "=" * 100)
    print(f"TOTAL across {len(results)} dumps: {total_completions} completions, "
          f"{total_ammonium} ({100*total_ammonium/total_completions:.2f}%) mention an ammonium species")
    print(f"Sampled impact check: {total_orig_fail} balances-failures under the ORIGINAL validator, "
          f"{total_fixed_recover} recovered by the comparator's fix "
          f"({100*total_fixed_recover/total_orig_fail:.1f}% of failures, if any)"
          if total_orig_fail else "Sampled impact check: no original balances-failures found in sample")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "pattern": AMMONIUM_PATTERN.pattern,
        "seed": SEED,
        "sample_n_per_dump": SAMPLE_N,
        "per_dump": results,
        "totals": {
            "total_completions": total_completions,
            "total_ammonium_mentioned": total_ammonium,
            "total_original_balances_fail_sampled": total_orig_fail,
            "total_fixed_recovers_sampled": total_fixed_recover,
        },
    }, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
