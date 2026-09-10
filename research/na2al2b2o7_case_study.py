#!/usr/bin/env python
"""
na2al2b2o7_case_study.py — URGENT_PRESENTATION_PREP.md Step 2.

Na2Al2B2O7 is in the pass@k held-out val set with pass@1=1.000 (SFT), but
misc/passk_n200.json only stored rewards, not completions -- regenerate 8
closed-book samples from the SFT checkpoint (same prompt/settings as
probe_passk.py) to see WHAT the model actually proposed.

Usage:
  PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python run_debug_and_analysis/na2al2b2o7_case_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_batched import load_eval_model, generate_batch  # noqa: E402
from core.reward import parse_completion, load_validator  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402
from validator import SynthesisValidator  # noqa: E402

TARGET = "Na2Al2B2O7"
CHECKPOINT = "runs/sft-qlora-sft-v3-2nd-rank16/final"
TRAD_SET = {"Na2CO3", "Al2O3", "B2O3"}
PRED_SET = {"Al2O3", "NaBO2"}
OUT_MD = Path("misc/case_study_na2al2b2o7.md")


def closed_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def route_summary(route):
    precs = sorted({SynthesisValidator._normalize_formula(p.formula)
                    for p in route.precursors})
    temps = []
    for op in route.operations:
        temps.extend(op.conditions.heating_temperature or [])
    return precs, (max(temps) if temps else None)


def classify(precs: set) -> str:
    if precs == {SynthesisValidator._normalize_formula(f) for f in TRAD_SET}:
        return "TRADITIONAL"
    if precs == {SynthesisValidator._normalize_formula(f) for f in PRED_SET}:
        return "PREDICTED"
    if precs <= ({SynthesisValidator._normalize_formula(f) for f in TRAD_SET} |
                {SynthesisValidator._normalize_formula(f) for f in PRED_SET}):
        return "PARTIAL/OTHER (subset of known sets)"
    return "NEITHER (novel precursor choice)"


def main():
    validator = load_validator(Path("data/cache/mp_formula_set.pkl"),
                               Path("data/cache/pd_index.json"), Path("."))
    model, tok = load_eval_model(checkpoint=CHECKPOINT, model_name="Qwen/Qwen3-8B")

    gen_args = SimpleNamespace(temperature=1.0, top_p=0.95, max_new_tokens=8192)
    prompt = closed_prompt(TARGET)

    print(f"generating 8 samples for {TARGET} ...", flush=True)
    completions = generate_batch(model, tok, [prompt] * 8, gen_args)

    rows = []
    for i, comp in enumerate(completions):
        try:
            route = parse_completion(comp, TARGET)
            reward, breakdown = validator.validate(route, TARGET)
            precs, max_T = route_summary(route)
            match = classify(set(precs))
        except Exception as e:
            reward, breakdown, precs, max_T, match = None, {"error": str(e)}, [], None, "PARSE_FAILURE"
        rows.append({"i": i, "precursors": precs, "max_T": max_T, "reward": reward,
                    "match": match, "completion": comp})
        print(f"  [{i}] precursors={precs} T={max_T} reward={reward} match={match}",
              flush=True)

    lines = [
        "# Case study — Na2Al2B2O7 (ASTRAL target 30)",
        "",
        f"*Regenerated 2026-09-01: {OUT_MD} did not exist in "
        f"`misc/passk_n200.json` (rewards only, no completions stored). "
        f"8 fresh closed-book samples, SFT checkpoint ({CHECKPOINT}), "
        f"temperature=1.0, top_p=0.95, matching `probe_passk.py`'s settings.*",
        "",
        "MIRA's pass@k held-out set: **pass@1 = 1.000** (16/16 successes, "
        "mean_reward=0.9825, `misc/passk_n200.json`).",
        "",
        "ASTRAL ground truth: traditional route (Na2CO3 + Al2O3 + B2O3) yields "
        "**0.00 phase fraction at 600°C**, 0.46 at 700°C. Predicted route "
        "(Al2O3 + NaBO2) yields 0.52 at 600°C, **0.60 at 700°C**.",
        "",
        "## What the model actually proposed (8 samples, verbatim precursor sets)",
        "",
        "| # | precursors | max T (°C) | validator reward | matches |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['i']} | {', '.join(r['precursors']) or '(parse failure)'} "
                     f"| {r['max_T']} | {r['reward']} | {r['match']} |")

    n_trad = sum(1 for r in rows if r["match"] == "TRADITIONAL")
    n_pred = sum(1 for r in rows if r["match"] == "PREDICTED")
    n_other = len(rows) - n_trad - n_pred
    lines += [
        "",
        f"**{n_trad}/8 matched the traditional set, {n_pred}/8 matched the "
        f"predicted set, {n_other}/8 other/novel/parse-failure.**",
        "",
        "## The slide, if the model matched traditional",
        "",
        ("The model proposed the TRADITIONAL route (Na2CO3+Al2O3+B2O3) and "
         "scored near-perfect on the validator (mean reward "
         f"{sum(r['reward'] for r in rows if r['reward'] is not None) / max(1, sum(1 for r in rows if r['reward'] is not None)):.3f}) "
         "while the robot measured 0.00 phase fraction at 600°C for exactly "
         "that route (0.46 at the model's typical 700°C). "
         "**Validator-perfect, experimentally near-zero at the lower "
         "temperature.**") if n_trad > n_pred else
        ("The model's actual precursor choices were mixed/novel rather than "
         "cleanly matching either published set -- see the per-sample table "
         "above for the real distribution; the clean single-slide framing "
         "the spec anticipated (100% traditional, validator-perfect) did "
         "NOT hold verbatim. Report what's actually there."),
        "",
        "## Raw completions",
        "",
    ]
    for r in rows:
        lines.append(f"### sample {r['i']} (reward={r['reward']}, match={r['match']})")
        lines.append("```")
        lines.append(r["completion"][:4000])
        lines.append("```")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
