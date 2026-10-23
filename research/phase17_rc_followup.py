#!/usr/bin/env python
"""
research/phase17_rc_followup.py — follow-up to results/phase17/rc_price_equation.json
per regular Claude's review:

1. R1's bare-oxide mismatch (the 1/7 miss) needs its RAW COUNTS next to it --
   R1's bare-oxide share sits near zero, so a sign flip there can be a
   handful-of-recipes artifact. Report n with the feature true / n total,
   early and late.
2. State explicitly whether "observed shift" was measured the SAME WAY for
   R1 (training-trajectory step0-29 vs step270-299 pooling) and R2/R4
   (matched before/after eval files) -- it was NOT, and this must be said
   in one line, not left implicit in the per-reward observed_shift_method
   strings alone.

Does NOT overwrite rc_price_equation.json's existing fields -- adds
`r1_bare_oxide_mismatch_counts` and `observed_shift_method_parity`.

Usage (tmux, CPU only, reuses r1_records_and_shift by import -- re-parses
the same training dump once, no new generation):
  PYTHONPATH=. uv run python research/phase17_rc_followup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase17_rc_price_equation import OUT_PATH  # noqa: E402


def counts(feats_list: list[dict], feature: str) -> dict:
    vals = [f[feature] for f in feats_list if f[feature] is not None]
    n_true = sum(1 for v in vals if v > 0.5)
    return {"n_total_with_feature_defined": len(vals), "n_true": n_true,
            "share": round(n_true / len(vals), 4) if vals else None}


def main():
    print("rebuilding R1 early/late cohorts (same construction as rc_price_equation.py)...", flush=True)
    import json as _json
    from research.phase17_rc_price_equation import features_from_precursors, astral_predicted_sets, RUN1_GENERATIONS
    from research.support_growth_by_checkpoint import build_reference
    from core.reward import parse_completion

    ref_astral = build_reference("astral")
    ref_heldout = build_reference("heldout")
    astral_pred = astral_predicted_sets()
    early, late = [], []
    with RUN1_GENERATIONS.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = _json.loads(line)
            step, target, completion = rec["step"], rec["target"], rec["completion"]
            try:
                route = parse_completion(completion, target)
            except Exception:
                continue
            precs = [p.formula for p in (route.precursors or [])]
            if not precs:
                continue
            max_T = None
            for op in (route.operations or []):
                for v in (getattr(op.conditions, "heating_temperature", None) or []):
                    if max_T is None or v > max_T:
                        max_T = v
            ref = ref_astral.get(target) or ref_heldout.get(target)
            feats = features_from_precursors(target, precs, max_T, ref, astral_pred.get(target))
            if step <= 29:
                early.append(feats)
            elif step >= 270:
                late.append(feats)

    bare_oxide_counts = {
        "early_step0_29": counts(early, "bare_alkali_oxide_used"),
        "late_step270_299": counts(late, "bare_alkali_oxide_used"),
    }
    print(f"bare_oxide counts: {bare_oxide_counts}", flush=True)

    report = json.loads(OUT_PATH.read_text())
    report["r1_bare_oxide_mismatch_counts"] = {
        "note": "R1's bare-oxide S_phi predicted an INCREASE but the measured step0->step299 "
                "trajectory showed a decrease (the one named mismatch in prediction_check, "
                "6/7 overall). Raw counts below -- a near-zero base rate means this sign is "
                "sensitive to a handful of recipes; reported, not explained away.",
        **bare_oxide_counts,
    }
    report["observed_shift_method_parity"] = {
        "statement": "Observed shift was measured DIFFERENTLY for R1 vs R2/R4, not the same way. "
                     "R1's observed shift: pooled step 0-29 vs step 270-299 cohorts of the SAME "
                     "run's own training-generation dump (a within-run trajectory). R2's and R4's "
                     "observed shift: two SEPARATE, already-existing eval files (results/astral_gen_"
                     "n32_rs_sft.json before vs results/astral_gen_n32_gdpo_phase12_beta0.json / "
                     "..._e1.json after), both n=32/target closed-book ASTRAL generations on the "
                     "identical 35 targets. Flagged in each reward's own observed_shift_method "
                     "string; this field states the difference in one place.",
        "same_method": False,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"-> {OUT_PATH} (r1_bare_oxide_mismatch_counts + observed_shift_method_parity added)")


if __name__ == "__main__":
    main()
