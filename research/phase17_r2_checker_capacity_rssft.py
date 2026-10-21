#!/usr/bin/env python
"""
research/phase17_r2_checker_capacity_rssft.py — Phase 17 Block R, R2/R-b:
score RS-SFT's OWN existing samples with the 5-channel RUN3 checker
aggregate, so the ranker's 46.3% capacity (finding 32,
results/ranker_capacity.json) has a like-for-like comparison. Per
regular Claude (2026-10-19): "the checker's 14-16% capacity came from
Corpus SFT's samples and the ranker's 46.3% from RS-SFT's, so the '3x'
gap isn't like-for-like... Block R's R-a step fixes this by scoring
RS-SFT's samples with the checker. Don't quote the 3x until that's done."

NO NEW GENERATION, NO VALIDATOR EDIT. Mirrors
research/ranker_capacity_rssft.py exactly: same source files, same
sample_group filter, same group-of-8 chunking, same capacity_metrics
call -- only the scorer changes (validator.validate() restricted to
RUN3_CHECKS, instead of Ranker.score()).

Source: data/novelty/rssft_reference_{heldout,astral}.jsonl,
sample_group == "resampling_rate" -- the identical 2400-row/300-group
scope results/ranker_capacity.json used.

validator_version=1 (default), matching Phase 12's own launch
(scripts/run_gdpo_phase12_beta0.sh passes no --validator-version flag;
train.py's default is 1). This is deliberately NOT validator_version=2
(E1/ranker's choice) -- R2 audits the checker exactly as Phase 12
trained against it.

Usage (tmux, CPU only -- thermo-backed validator calls, no GPU needed):
  PYTHONPATH=. uv run python research/phase17_r2_checker_capacity_rssft.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.reward import ParseFailure, parse_completion, RUN3_CHECKS  # noqa: E402
from probe_hardening import capacity_metrics  # noqa: E402
from validator import SynthesisValidator, ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
RSSFT_PATHS = {
    "heldout": REPO_ROOT / "data" / "novelty" / "rssft_reference_heldout.jsonl",
    "astral": REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl",
}
OUT_PATH = REPO_ROOT / "results" / "phase17" / "r2_checker_capacity_rssft.json"
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


def score_target_set(validator, by_target: dict[str, list[dict]], label: str) -> tuple[list[dict], int, int]:
    records = []
    n_scored = 0
    n_parse_fail = 0
    for target, recs in by_target.items():
        groups = chunk(recs, GROUP_SIZE)
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
                if route is None:
                    breakdown = {"error": "parse_failure"}
                    reward = 0.0
                else:
                    reward, breakdown = validator.validate(route, target)
                records.append({
                    "target": group_key, "breakdown": breakdown, "reward": reward,
                    "source_target": target, "target_set": label,
                })
                n_scored += 1
    return records, n_scored, n_parse_fail


def tie_rate_stats(records: list[dict], check_names: tuple[str, ...]) -> dict:
    """Per-channel: predicted tie rate from per-target pass rates
    (P_tie = p^G + (1-p)^G, pass/fail at bar 0.5 on that channel's own
    [0,1] scale) vs the observed tie rate (zero within-group std).
    R-b's prediction: these agree within 5 points for binary channels."""
    by_group = defaultdict(lambda: defaultdict(list))
    for r in records:
        bd = r.get("breakdown") or {}
        if bd.get("error"):
            continue
        for name in check_names:
            v = bd.get(name)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                by_group[name][r["target"]].append(v)

    out = {}
    for name in check_names:
        groups = by_group[name]
        if not groups:
            out[name] = None
            continue
        n_groups = 0
        n_ties = 0
        pass_rates = []
        for gvals in groups.values():
            if len(gvals) < 2:
                continue
            n_groups += 1
            if max(gvals) == min(gvals):
                n_ties += 1
            pass_rates.append(sum(1 for v in gvals if v >= 0.5) / len(gvals))
        if n_groups == 0:
            out[name] = None
            continue
        observed_tie_rate = n_ties / n_groups
        g = GROUP_SIZE
        predicted_tie_rate = sum(
            (p ** g + (1 - p) ** g) for p in pass_rates
        ) / len(pass_rates) if pass_rates else float("nan")
        share_mid_band = sum(1 for p in pass_rates if 0.1 <= p <= 0.9) / len(pass_rates) if pass_rates else float("nan")
        out[name] = {
            "n_groups": n_groups,
            "observed_tie_rate": round(observed_tie_rate, 4),
            "predicted_tie_rate_from_pass_rates": round(predicted_tie_rate, 4),
            "agrees_within_5pp": abs(observed_tie_rate - predicted_tie_rate) <= 0.05,
            "share_targets_pass_rate_in_0.1_0.9": round(share_mid_band, 4),
        }
    return out


def main():
    print("loading validator formula set + PD cache (validator_version=1, matching Phase 12's launch)...", flush=True)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    validator = SynthesisValidator(formula_set, thermo_checker=thermo, validator_version=1)
    print(f"RUN3_CHECKS (5ch): {RUN3_CHECKS}", flush=True)

    all_records = []
    per_set_summary = {}
    for label, path in RSSFT_PATHS.items():
        print(f"\nloading {path} (resampling_rate only)...", flush=True)
        by_target = load_resampling_records(path)
        n_targets = len(by_target)
        records, n_scored, n_parse_fail = score_target_set(validator, by_target, label)
        all_records.extend(records)
        n_groups_this = len({r["target"] for r in records})
        cap_this = capacity_metrics(records, list(RUN3_CHECKS))
        print(f"  {label}: n_targets={n_targets} n_scored={n_scored} n_parse_fail={n_parse_fail} "
              f"n_groups={n_groups_this} capacity={cap_this.get('capacity_pct')}%", flush=True)
        per_set_summary[label] = {
            "n_targets": n_targets, "n_scored": n_scored, "n_parse_fail": n_parse_fail,
            "capacity": cap_this,
        }

    print("\n" + "=" * 78)
    cap_combined = capacity_metrics(all_records, list(RUN3_CHECKS))
    print(f"COMBINED (35 ASTRAL + 40 held-out, {cap_combined.get('n_groups')} groups of 8): "
          f"checker (RUN3, 5ch) capacity on RS-SFT's own samples = {cap_combined.get('capacity_pct')}%")
    print("(compare: ranker v2 on the SAME samples, results/ranker_capacity.json = 46.3%)")
    print("(compare: checker on CORPUS SFT's samples -- finding 1 = 14.0%, finding 9 baseline = 16.4% -- NOT the same policy)")

    ties = tie_rate_stats(all_records, RUN3_CHECKS)
    print("\n--- R-b tie-rate check (predicted vs observed, per channel) ---")
    for name, stats in ties.items():
        print(f"  {name:<28}{stats}")

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-b (signal/capacity) + tie-rate check; closes the R2-vs-R4 like-for-like gap",
        "reward_id": "R2",
        "reward": "Checker, 5-channel RUN3 aggregate",
        "policy": "RS-SFT",
        "validator_version": 1,
        "run3_checks": list(RUN3_CHECKS),
        "source": "data/novelty/rssft_reference_{heldout,astral}.jsonl, sample_group=resampling_rate (NO NEW GENERATION), same 2400-row/300-group scope as results/ranker_capacity.json",
        "group_size": GROUP_SIZE,
        "per_target_set": per_set_summary,
        "combined": {
            "n_records": len(all_records),
            "capacity": cap_combined,
        },
        "tie_rate_check": ties,
        "comparison_reference": {
            "ranker_v2_on_rssft_samples_capacity_pct": 46.31,
            "checker_on_corpus_sft_samples_finding_1_pct": 14.0,
            "checker_on_corpus_sft_samples_finding_9_baseline_pct": 16.4,
            "note": "This file's combined capacity_pct is the first like-for-like (same policy, same samples) checker-vs-ranker comparison. Any '3x' claim must cite THIS number against 46.31%, not finding 1/9.",
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
