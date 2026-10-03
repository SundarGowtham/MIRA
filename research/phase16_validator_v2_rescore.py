#!/usr/bin/env python
"""
research/phase16_validator_v2_rescore.py — Phase 16 §2.4.3 evaluation:
"re-score Phase 12's ASTRAL generations with version 2 and report how
much changes." Reads the already-generated (no new GPU work)
results/astral_gen_n32_{rs_sft,gdpo_phase12_beta0}.json dumps.

Reconstruction limitation carried over from Phase 15 Task 2 (docs/phases/
PHASE15_DISTRIBUTIONAL.md): these dumps store only precursors/max_T/
reward/match per sample, not the raw completion, declared amounts, or
real operation/atmosphere sequence. Each sample is re-scored as a single
HeatingOperation at the reported max_T, atmosphere defaulted to "air",
amounts defaulted to 1.0 each (amounts were never stored, so
amount_accuracy cannot be faithfully recomputed from this data --
_check_stoichiometry, which only asks "does ANY valid balance exist," is
unaffected by the placeholder amounts and is what this script reports).

Usage:
  uv run python research/phase16_validator_v2_rescore.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator import (  # noqa: E402
    PredictedConditions,
    PredictedOperation,
    PredictedPrecursor,
    PredictedRoute,
    SynthesisValidator,
)

OUT_DIR = Path("results/phase16")
FILES = {
    "rs_sft": Path("results/astral_gen_n32_rs_sft.json"),
    "gdpo_phase12_beta0": Path("results/astral_gen_n32_gdpo_phase12_beta0.json"),
}

V1 = SynthesisValidator(mp_formula_set=set(), thermo_checker=None, validator_version=1)
V2 = SynthesisValidator(mp_formula_set=set(), thermo_checker=None, validator_version=2)


def to_route(target: str, sample: dict) -> PredictedRoute:
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(p, 1.0) for p in sample["precursors"]],
        operations=[PredictedOperation(
            type="calcine",
            conditions=PredictedConditions(
                heating_atmosphere=["air"],
                heating_temperature=[sample["max_T"]],
            ),
        )],
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for model_name, path in FILES.items():
        if not path.exists():
            print(f"skip {model_name}: {path} not found")
            continue
        data = json.loads(path.read_text())
        n_total = 0
        n_v1_balances = 0
        n_v2_balances = 0
        n_flipped = 0  # v1 fails to balance, v2 succeeds
        flipped_examples = []
        for entry in data["results"]:
            target = entry["target"]
            for sample in entry["samples"]:
                n_total += 1
                r = to_route(target, sample)
                s1 = V1._check_stoichiometry(r)
                s2 = V2._check_stoichiometry(r)
                v1_ok = s1 == 1.0
                v2_ok = s2 == 1.0
                n_v1_balances += int(v1_ok)
                n_v2_balances += int(v2_ok)
                if (not v1_ok) and v2_ok:
                    n_flipped += 1
                    if len(flipped_examples) < 10:
                        flipped_examples.append({
                            "target": target, "precursors": sample["precursors"],
                            "max_T": sample["max_T"],
                        })
        report[model_name] = {
            "n_total_samples": n_total,
            "n_balances_under_v1": n_v1_balances,
            "n_balances_under_v2": n_v2_balances,
            "pct_balances_v1": round(100 * n_v1_balances / n_total, 2) if n_total else None,
            "pct_balances_v2": round(100 * n_v2_balances / n_total, 2) if n_total else None,
            "n_flipped_v1_fail_v2_pass": n_flipped,
            "pct_flipped": round(100 * n_flipped / n_total, 2) if n_total else None,
            "flipped_examples": flipped_examples,
        }
        print(f"{model_name}: n={n_total}  v1 balances={n_v1_balances} "
              f"({report[model_name]['pct_balances_v1']}%)  "
              f"v2 balances={n_v2_balances} ({report[model_name]['pct_balances_v2']}%)  "
              f"flipped={n_flipped} ({report[model_name]['pct_flipped']}%)")

    out_path = OUT_DIR / "validator_v2_rescore.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
