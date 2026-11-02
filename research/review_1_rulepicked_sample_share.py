#!/usr/bin/env python
"""
research/review_1_rulepicked_sample_share.py — review check 1 (2026-11-01):
sample-level ASTRAL rule-picked ("predicted") share across ALL generated
samples (parse failures / no-precursor samples count as non-matches, not
excluded from the denominator), for 8 models, on the full 35 ASTRAL
targets and the clean 22, with a target-clustered bootstrap CI and the
paired difference vs each model's init. Reading rule pre-registered in
docs/phases/REVIEW_RESULTS.md before this ran.

Checks/corrects the share implied by
results/phase17/rc_price_equation.json's observed_shift for
matches_astral_rule_picked (RS-SFT 3.82% -> Phase12 3.31%, RS-SFT 3.82% ->
E1 5.11%): that file's shift_from_astral_files() excludes
PARSE_FAILURE/no-precursor samples from the denominator; this script uses
the literal ALL-samples denominator instead, so a small, explainable
difference is expected, not an error.

Normalisation matches precset_of (research/support_growth_by_checkpoint.py)
and rc_price_equation.py's canon() exactly: both ultimately call
validator.SynthesisValidator._normalize_formula, i.e.
pymatgen.Composition(formula).reduced_formula.

NO EDIT to any frozen file. NO TRAINING. New output file only, under
results/review/.

Usage (tmux, CPU only, seconds):
  PYTHONHASHSEED=0 PYTHONPATH=. uv run python research/review_1_rulepicked_sample_share.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validator import SynthesisValidator  # noqa: E402

R = REPO_ROOT / "results"
OUT_DIR = R / "review"
OUT_PATH = OUT_DIR / "rulepicked_sample_share.json"
SEED = 20261016
N_BOOT = 10000

MODEL_FILES = {
    "base": "astral_gen_n32_base.json",
    "corpus_sft": "astral_gen_n32_sft.json",
    "rs_sft": "astral_gen_n32_rs_sft.json",
    "gdpo300": "astral_gen_n32_gdpo300.json",
    "phase12_ckpt300": "astral_gen_n32_gdpo_phase12_beta0.json",
    "e1_ckpt300": "astral_gen_n32_e1.json",
    "e3": "astral_gen_n32_e3.json",
    "e3c": "astral_gen_n32_e3c.json",
}
# model -> its initialisation model, per CLAUDE.md's Phase 12/16 provenance
INIT_OF = {
    "corpus_sft": "base",
    "rs_sft": "base",
    "gdpo300": "corpus_sft",
    "phase12_ckpt300": "rs_sft",
    "e1_ckpt300": "rs_sft",
    "e3": "base",
    "e3c": "base",
}


def norm(formula: str) -> str:
    return SynthesisValidator._normalize_formula(formula)


def load_predicted_sets() -> dict[str, frozenset]:
    doc = json.loads((R / "astral_validation_set.json").read_text())
    return {t["target"]: frozenset(norm(p) for p in t["predicted"]) for t in doc["targets"]}


def load_match_arrays(path: Path, predicted: dict[str, frozenset]) -> dict[str, list[bool]]:
    doc = json.loads(path.read_text())
    out = {}
    for r in doc["results"]:
        target = r["target"]
        pred = predicted.get(target)
        arr = []
        for s in r["samples"]:
            precs = s.get("precursors")
            if not precs or pred is None:
                arr.append(False)
                continue
            sample_set = frozenset(norm(p) for p in precs)
            arr.append(sample_set == pred)
        out[target] = arr
    return out


def share(by_target: dict[str, list[bool]], targets: list[str]) -> tuple[float | None, int, int]:
    n_match = sum(sum(by_target[t]) for t in targets)
    n_total = sum(len(by_target[t]) for t in targets)
    return (n_match / n_total if n_total else None), n_match, n_total


def cluster_bootstrap_ci(by_target: dict[str, list[bool]], targets: list[str], seed: int) -> list[float | None]:
    rng = random.Random(seed)
    n = len(targets)
    vals = []
    for _ in range(N_BOOT):
        sample_targets = [targets[rng.randrange(n)] for _ in range(n)]
        n_match = sum(sum(by_target[t]) for t in sample_targets)
        n_total = sum(len(by_target[t]) for t in sample_targets)
        if n_total:
            vals.append(n_match / n_total)
    vals.sort()
    if not vals:
        return [None, None]
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
    return [lo, hi]


def paired_diff_bootstrap(model_bt: dict, init_bt: dict, targets: list[str], seed: int) -> tuple[float | None, list]:
    """Cluster bootstrap by target: ONE resampled target list per replicate,
    applied to BOTH model and init, so common per-target sampling noise
    cancels in the diff rather than inflating the CI (same logic as
    block_a_paired_bootstrap_phase12_vs_rssft.json, at the target-cluster
    level instead of the within-target sample-position level)."""
    rng = random.Random(seed)
    n = len(targets)
    diffs = []
    for _ in range(N_BOOT):
        sample_targets = [targets[rng.randrange(n)] for _ in range(n)]
        nm_model = sum(sum(model_bt[t]) for t in sample_targets)
        nt_model = sum(len(model_bt[t]) for t in sample_targets)
        nm_init = sum(sum(init_bt[t]) for t in sample_targets)
        nt_init = sum(len(init_bt[t]) for t in sample_targets)
        if nt_model and nt_init:
            diffs.append(nm_model / nt_model - nm_init / nt_init)
    diffs.sort()
    if not diffs:
        return None, [None, None]
    median = diffs[len(diffs) // 2]
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    return median, [lo, hi]


def main():
    predicted = load_predicted_sets()
    clean_overlap = set(json.loads((R / "phase16" / "astral_clean_subset.json").read_text())["overlapping_targets"])

    by_target_all = {model: load_match_arrays(R / fname, predicted) for model, fname in MODEL_FILES.items()}

    full_35 = sorted(by_target_all["base"].keys())
    clean_22 = sorted(set(full_35) - clean_overlap)
    assert len(full_35) == 35, len(full_35)
    assert len(clean_22) == 22, len(clean_22)
    for model, bt in by_target_all.items():
        assert sorted(bt.keys()) == full_35, f"{model} target set mismatch"

    report = {
        "seed": SEED, "n_boot": N_BOOT, "full_35_n": 35, "clean_22_n": 22,
        "clean_22_excludes": sorted(clean_overlap),
        "per_model": {}, "paired_diff_vs_init": {},
    }

    for model, bt in by_target_all.items():
        entry = {}
        for label, tlist in (("full_35", full_35), ("clean_22", clean_22)):
            pt, nm, nt = share(bt, tlist)
            ci = cluster_bootstrap_ci(bt, tlist, SEED)
            entry[label] = {
                "share": pt, "pct": round(100 * pt, 3) if pt is not None else None,
                "n_match": nm, "n_total": nt,
                "ci95_pct": [round(100 * c, 3) if c is not None else None for c in ci],
            }
        report["per_model"][model] = entry
        print(f"{model}: full_35={entry['full_35']['pct']}% clean_22={entry['clean_22']['pct']}%", flush=True)

    for model, init in INIT_OF.items():
        d = {}
        for label, tlist in (("full_35", full_35), ("clean_22", clean_22)):
            median, ci = paired_diff_bootstrap(by_target_all[model], by_target_all[init], tlist, SEED)
            model_pt = report["per_model"][model][label]["share"]
            init_pt = report["per_model"][init][label]["share"]
            point_diff = (model_pt - init_pt) if (model_pt is not None and init_pt is not None) else None
            d[label] = {
                "model": model, "init": init,
                "point_diff_pct": round(100 * point_diff, 3) if point_diff is not None else None,
                "boot_median_diff_pct": round(100 * median, 3) if median is not None else None,
                "ci95_diff_pct": [round(100 * c, 3) if c is not None else None for c in ci],
            }
        report["paired_diff_vs_init"][model] = d
        print(f"{model} vs init={init}: full_35 diff={d['full_35']}", flush=True)

    rc = json.loads((R / "phase17" / "rc_price_equation.json").read_text())
    rc_r2 = rc["by_reward"]["R2"]["observed_shift"]["matches_astral_rule_picked"]
    rc_r4 = rc["by_reward"]["R4"]["observed_shift"]["matches_astral_rule_picked"]
    report["rc_price_equation_cross_check"] = {
        "rc_implied_pct_valid_samples_only": {
            "rs_sft_before": round(100 * rc_r2["before"], 3),
            "phase12_after": round(100 * rc_r2["after"], 3),
            "e1_after": round(100 * rc_r4["after"], 3),
        },
        "this_script_pct_all_samples_full_35": {
            "rs_sft": report["per_model"]["rs_sft"]["full_35"]["pct"],
            "phase12_ckpt300": report["per_model"]["phase12_ckpt300"]["full_35"]["pct"],
            "e1_ckpt300": report["per_model"]["e1_ckpt300"]["full_35"]["pct"],
        },
        "explanation": (
            "rc_price_equation.json's shift_from_astral_files() excludes PARSE_FAILURE/"
            "no-precursor samples from the denominator; this script uses the literal "
            "ALL-samples denominator (parse failures count as non-matches), so a small "
            "systematic difference is expected here, not an error."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
