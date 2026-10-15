#!/usr/bin/env python
"""
research/ranker_capacity_rssft.py — regular Claude's task (2026-10-15):
compute reward capacity under Ranker v2 (RANKER_VERSION,
core/ranker.py, as E1 trained it) on RS-SFT's OWN completions, so the
paper can compare it with the checker's 14% (finding 1) / 16.4%
(finding 9, hardening-probe baseline).

NO NEW GENERATION. Source: data/novelty/rssft_reference_{heldout,
astral}.jsonl, sample_group == "resampling_rate" -- 32 completions per
target, drawn directly from the RS-SFT checkpoint via vLLM
(data_curation/phase16_rssft_reference_sampling.py), zero GDPO
training applied. These already exist on disk; this script only
re-scores them with the ranker.

Group structure: capacity_metrics (probe_hardening.py, unchanged) keys
groups by a "target" string and needs >=2 samples/group to compute a
within-group std. To get the hardening-probe's own 8-samples-per-group
shape from RS-SFT's 32-sample draw per real target, each target's 32
completions are split into 4 non-overlapping synthetic groups of 8
(labeled "<target>#g<i>"), in file order (the 32 are already IID draws
from one sampling call, so order carries no structure to preserve).

Scored once combined over both target sets (35 ASTRAL + 40 held-out =
300 groups of 8), and separately per target set, because ASTRAL's
targets are deliberately novel/synthetic and have much sparser
Kononova literature coverage than the held-out split (drawn from the
corpus-derived RS-SFT training pool) -- this affects the
temperature_economy channel specifically (it needs a literature max_T
reference; returns None, not 0, when none exists -- handled by
capacity_metrics' own NaN-on-non-numeric convention, no samples
dropped).

Cross-check (item 4): pulls E1's own W&B training history
(train/frac_reward_zero_std, train/rewards/check_*/std) over the full
300-step run and reports it next to this script's offline numbers.
Any mismatch is reported, not smoothed over.

Usage (tmux, CPU only -- thermo-backed ranker calls, no GPU needed):
  uv run python research/ranker_capacity_rssft.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.reward import ParseFailure, parse_completion  # noqa: E402
from core.ranker import (  # noqa: E402
    OBJECTIVE_NAMES, RANKER_VERSION, Ranker, build_precursor_frequency,
    gate_failure_rates, rail_stats,
)
from probe_hardening import capacity_metrics, load_literature  # noqa: E402
from validator import ThermoChecker  # noqa: E402

TRIAGE = REPO_ROOT / "misc" / "kononova_triage_results3.json"
SYNTHESIS = REPO_ROOT / "data" / "raw" / "synthesis_clean.json"
RSSFT_PATHS = {
    "heldout": REPO_ROOT / "data" / "novelty" / "rssft_reference_heldout.jsonl",
    "astral": REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl",
}
OUT_PATH = REPO_ROOT / "results" / "ranker_capacity.json"
GROUP_SIZE = 8


def load_resampling_records(path: Path) -> dict[str, list[dict]]:
    by_target: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("sample_group") == "resampling_rate":
                by_target[rec["target"]].append(rec)
    return by_target


def chunk(lst: list, n: int) -> list[list]:
    return [lst[i:i + n] for i in range(0, len(lst), n) if len(lst[i:i + n]) == n]


def score_target_set(ranker, lit: dict, by_target: dict[str, list[dict]],
                     label: str) -> tuple[list[dict], int, int]:
    """Returns (records_for_capacity_metrics, n_scored, n_parse_fail)."""
    records = []
    n_scored = 0
    n_parse_fail = 0
    for target, recs in by_target.items():
        groups = chunk(recs, GROUP_SIZE)
        lit_rec = lit.get(target, {})
        for gi, group in enumerate(groups):
            group_key = f"{target}#g{gi}"
            for rec in group:
                try:
                    route = parse_completion(rec["completion"], target)
                except ParseFailure:
                    route = None
                    n_parse_fail += 1
                except Exception:
                    route = None
                    n_parse_fail += 1
                reward, info = ranker.score(
                    route, target, lit_T=lit_rec.get("max_T"), lit_n_ops=lit_rec.get("n_ops"))
                records.append({"target": group_key, "breakdown": info,
                                "reward": reward, "source_target": target, "target_set": label})
                n_scored += 1
    return records, n_scored, n_parse_fail


def main():
    print(f"ranker version in effect: {RANKER_VERSION}", flush=True)
    print("loading literature (precursors, max_T, n_ops)...", flush=True)
    lit = load_literature(TRIAGE, SYNTHESIS)

    print("loading validator formula set + PD cache + precursor frequency...", flush=True)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    freq = build_precursor_frequency(SYNTHESIS)
    # validator_version=2 exactly as E1's own launch (scripts/run_gdpo_e1.sh
    # --validator-version 2), threaded per commit fd1ffa7. Scales/weights
    # left at class defaults -- the same defaults E1 trained with, since
    # no ranker.py edit has landed since E1 launched.
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)
    print(f"ranker scales in effect: {ranker.scales}", flush=True)

    all_records = []
    per_set_summary = {}
    for label, path in RSSFT_PATHS.items():
        print(f"\nloading {path} (resampling_rate only)...", flush=True)
        by_target = load_resampling_records(path)
        n_targets = len(by_target)
        lit_covered = sum(1 for t in by_target if t in lit)
        print(f"  {n_targets} targets, {lit_covered}/{n_targets} with literature coverage", flush=True)
        records, n_scored, n_parse_fail = score_target_set(ranker, lit, by_target, label)
        all_records.extend(records)
        n_groups_this = len({r["target"] for r in records})
        cap_this = capacity_metrics(records, OBJECTIVE_NAMES)
        print(f"  {label}: n_scored={n_scored} n_parse_fail={n_parse_fail} "
              f"n_groups={n_groups_this} capacity={cap_this.get('capacity_pct')}%", flush=True)
        per_set_summary[label] = {
            "n_targets": n_targets, "n_lit_covered": lit_covered,
            "n_scored": n_scored, "n_parse_fail": n_parse_fail,
            "capacity": cap_this,
        }

    print("\n" + "=" * 78)
    cap_combined = capacity_metrics(all_records, OBJECTIVE_NAMES)
    print(f"COMBINED (35 ASTRAL + 40 held-out, {cap_combined.get('n_groups')} groups of 8): "
          f"capacity = {cap_combined.get('capacity_pct')}%")
    print("(compare: checker finding 1 = 14%, finding 9 hardening-probe baseline = 16.4%)")

    print("\n--- per-channel z-variance, zero-std-group %, and ceiling ---")
    print("ceiling = 1 - (zero-std-group fraction): within_group_z normalizes every "
          "non-zero-std group to unit variance by construction, so a channel's "
          "z-variance cannot exceed this regardless of how strong its real signal is.")
    zvar = cap_combined.get("per_channel_z_var", {})
    zstd_pct = cap_combined.get("per_channel_zero_std_pct", {})
    rail_combined = rail_stats([r["breakdown"] for r in all_records])
    gate_combined = gate_failure_rates([r["breakdown"] for r in all_records])
    ceiling = {}
    print(f"  {'channel':<28}{'z-var':>8}{'zero-std%':>11}{'ceiling':>9}")
    for name in OBJECTIVE_NAMES:
        frac0 = zstd_pct.get(name, float("nan"))
        ceil = round(1 - frac0 / 100, 4) if frac0 == frac0 else float("nan")
        ceiling[name] = ceil
        print(f"  {name:<28}{zvar.get(name, float('nan')):>8.4f}{frac0:>10.1f}%{ceil:>9.4f}")

    print("\n--- gate failure rates (combined) ---")
    for g, v in gate_combined.items():
        print(f"  {g:<24}{v}%" if v is not None else f"  {g:<24}n/a")

    # ------------------------------------------------------------------
    # Item 4: cross-check against E1's own training logs (W&B)
    # ------------------------------------------------------------------
    print("\n--- cross-check: E1's own training logs (W&B run 238lefg3) ---")
    wandb_cross_check = None
    try:
        import os
        import wandb
        api = wandb.Api()
        run = api.run("gowthamsundar1998-supportvectors/mira/238lefg3")
        keys = ["train/frac_reward_zero_std"] + [
            f"train/rewards/check_{name}/std" for name in OBJECTIVE_NAMES]
        hist = run.history(keys=keys, samples=10000)
        import math
        means = {}
        for k in keys:
            if k in hist.columns:
                vals = [v for v in hist[k].tolist() if v is not None and not (isinstance(v, float) and math.isnan(v))]
                means[k] = {"mean": round(sum(vals) / len(vals), 4) if vals else None,
                           "min": round(min(vals), 4) if vals else None,
                           "max": round(max(vals), 4) if vals else None,
                           "n_logged": len(vals)}
            else:
                means[k] = None
        wandb_cross_check = means
        print(f"  train/frac_reward_zero_std over full run: {means.get('train/frac_reward_zero_std')}")
        for name in OBJECTIVE_NAMES:
            k = f"train/rewards/check_{name}/std"
            print(f"  {name:<28}{means.get(k)}")
    except Exception as e:
        print(f"  WARNING: could not pull W&B history ({e}). "
              f"Cross-check must be filled in from the earlier manual W&B pull "
              f"in this conversation instead.", flush=True)

    report = {
        "ranker_version": RANKER_VERSION,
        "source": "data/novelty/rssft_reference_{heldout,astral}.jsonl, sample_group=resampling_rate (NO NEW GENERATION)",
        "group_size": GROUP_SIZE,
        "per_target_set": per_set_summary,
        "combined": {
            "n_records": len(all_records),
            "capacity": cap_combined,
            "ceiling_by_channel": ceiling,
            "rail_stats": rail_combined,
            "gate_failure_rates": gate_combined,
        },
        "wandb_cross_check_e1_run_238lefg3": wandb_cross_check,
        "comparison_reference": {
            "checker_finding_1_capacity_pct": 14.0,
            "checker_finding_9_hardening_probe_baseline_pct": 16.4,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
