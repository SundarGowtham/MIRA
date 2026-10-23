#!/usr/bin/env python
"""
research/phase17_r1_r3_checker_capacity.py — Block R triage (regular
Claude, 2026-10-22), items 1d/1e:

R1: the original 10-channel checker aggregate, recomputed with
probe_hardening.capacity_metrics directly on results/hardening.json's
"baseline" condition (SFT policy, closed-book, 40 targets x 8 samples
-- the hardening probe's own dedicated sample pool), NOT quoted from
finding 1 (14.0%, an older/different sample source with no surviving
flat-record file) or finding 9 (16.4%, prose only). This file's number
IS finding 9's 16.4% reproduced from code, with full provenance
(sha256, record counts) instead of prose.

R3: the same recomputation on misc/hardening_low_temp.json's "low_temp"
condition (same 40 targets x 8 samples, SFT policy, soft low-T
preference), at both 10 channels and 11 (+temperature_economy) --
reproducing finding 15's whole table (16.4% / 19.0% / 23.8%) from
code.

Both get the R-b tie-rate check (predicted p^G+(1-p)^G vs observed,
per channel) and R1 additionally gets the R-e noise-amplification
check (aggregation_diagnostics, reused by import from
phase17_r5_comparator_capacity_rssft.py) -- the pre-registered R-e
prediction is "at least one R1 channel has noise-amplified live
groups > 10%".

2x2 TABLE (item 1e, labeled [E]): {10-channel checker, 5-channel RUN3}
x {hardening-probe baseline samples (SFT, closed-book, 40 targets),
RS-SFT's own samples (data/novelty/rssft_reference_*, 75 targets)}.
Separates "fewer channels raises the average mechanically" from "a
different policy is more gradeable" -- the RS-SFT side reuses
phase17_r2_checker_capacity_rssft.py's own scoring functions by
import (one cheap CPU rerun, validator_version=1, matching that
file's choice) to get full 10-channel breakdowns, which that file's
own saved JSON does not retain (it only kept the 5ch aggregate).

NO NEW GENERATION. NO VALIDATOR EDIT. New output file only:
results/phase17/r1_r3_checker_capacity_2x2.json

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_r1_r3_checker_capacity.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from probe_hardening import capacity_metrics  # noqa: E402
from research.phase17_r5_comparator_capacity_rssft import aggregation_diagnostics  # noqa: E402
from research.phase17_r2_checker_capacity_rssft import (  # noqa: E402
    RSSFT_PATHS, load_resampling_records, score_target_set as r2_score_target_set,
)
from validator import SynthesisValidator, ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "r1_r3_checker_capacity_2x2.json"
HARDENING_PATH = REPO_ROOT / "results" / "hardening.json"
LOW_TEMP_PATH = REPO_ROOT / "misc" / "hardening_low_temp.json"
GROUP_SIZE = 8

CHECK_10 = ["stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist",
            "operation_order", "temperature_plausible", "thermodynamic_favorable",
            "target_stability", "chempot_atmosphere", "target_match"]
RUN3_CHECKS_5 = ["amount_accuracy", "thermodynamic_favorable", "stoichiometry",
                 "chempot_atmosphere", "operation_order"]
CHECK_11 = CHECK_10 + ["temperature_economy"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tie_rate_stats(records: list[dict], check_names: list[str]) -> dict:
    from collections import defaultdict
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
        n_groups, n_ties, pass_rates = 0, 0, []
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
        observed = n_ties / n_groups
        g = GROUP_SIZE
        predicted = sum((p ** g + (1 - p) ** g) for p in pass_rates) / len(pass_rates)
        share_mid = sum(1 for p in pass_rates if 0.1 <= p <= 0.9) / len(pass_rates)
        out[name] = {"n_groups": n_groups, "observed_tie_rate": round(observed, 4),
                      "predicted_tie_rate_from_pass_rates": round(predicted, 4),
                      "agrees_within_5pp": abs(observed - predicted) <= 0.05,
                      "share_targets_pass_rate_in_0.1_0.9": round(share_mid, 4)}
    return out


def load_hardening_condition(path: Path, condition: str) -> list[dict]:
    d = json.loads(path.read_text())
    return [r for r in d["records"] if r.get("condition") == condition]


def main():
    print(f"R1 source: {HARDENING_PATH} sha256={sha256_of(HARDENING_PATH)}", flush=True)
    print(f"R3 source: {LOW_TEMP_PATH} sha256={sha256_of(LOW_TEMP_PATH)}", flush=True)

    # ---------------- R1: checker (10ch), hardening-probe baseline ----------------
    r1_records = load_hardening_condition(HARDENING_PATH, "baseline")
    r1_cap = capacity_metrics(r1_records, CHECK_10)
    r1_ties = tie_rate_stats(r1_records, CHECK_10)
    r1_noise = aggregation_diagnostics(r1_records, CHECK_10)
    r1_noise_pred_passes = any(
        (v is not None and v > 10.0) for v in (r1_noise.get("noise_amplified_pct_by_channel") or {}).values()
    )
    print(f"\nR1 (checker, 10ch, hardening baseline): capacity={r1_cap.get('capacity_pct')}% "
          f"n_records={len(r1_records)}", flush=True)
    print("R1 tie-rate:", json.dumps(r1_ties, indent=1), flush=True)
    print("R1 noise-amplification by channel:", r1_noise.get("noise_amplified_pct_by_channel"), flush=True)
    print(f"R-e prediction (>=1 R1 channel noise-amplified >10% of live groups): "
          f"{'PASSES' if r1_noise_pred_passes else 'FAILS'}", flush=True)

    # ---------------- R3: low-temp objective reward, 10ch and 11ch ----------------
    r3_baseline = load_hardening_condition(LOW_TEMP_PATH, "baseline")
    r3_low_temp = load_hardening_condition(LOW_TEMP_PATH, "low_temp")
    r3_cap_baseline_10 = capacity_metrics(r3_baseline, CHECK_10)
    r3_cap_lowtemp_10 = capacity_metrics(r3_low_temp, CHECK_10)
    r3_cap_lowtemp_11 = capacity_metrics(r3_low_temp, CHECK_11)
    r3_ties_11 = tie_rate_stats(r3_low_temp, CHECK_11)
    print(f"\nR3 baseline(10ch)={r3_cap_baseline_10.get('capacity_pct')}% "
          f"low_temp(10ch)={r3_cap_lowtemp_10.get('capacity_pct')}% "
          f"low_temp(11ch incl temperature_economy)={r3_cap_lowtemp_11.get('capacity_pct')}%", flush=True)
    print("(compare: finding 15's prose table -- 16.4% / 19.0% / 23.8%)", flush=True)

    # ---------------- 2x2: channel-count x policy, isolating each axis ----------------
    print("\nscoring RS-SFT samples with the FULL 10-channel validator output "
          "(cheap rerun of phase17_r2's own scoring, validator_version=1)...", flush=True)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    validator = SynthesisValidator(formula_set, thermo_checker=thermo, validator_version=1)

    rssft_records = []
    for label, path in RSSFT_PATHS.items():
        by_target = load_resampling_records(path)
        recs, n_scored, n_parse_fail = r2_score_target_set(validator, by_target, label)
        rssft_records.extend(recs)
        print(f"  {label}: n_scored={n_scored} n_parse_fail={n_parse_fail}", flush=True)

    cap_rssft_10 = capacity_metrics(rssft_records, CHECK_10)
    cap_rssft_5 = capacity_metrics(rssft_records, RUN3_CHECKS_5)
    cap_hardening_10 = r1_cap
    cap_hardening_5 = capacity_metrics(r1_records, RUN3_CHECKS_5)

    two_by_two = {
        "10_channel": {
            "hardening_probe_baseline_SFT": cap_hardening_10.get("capacity_pct"),
            "rssft_own_samples": cap_rssft_10.get("capacity_pct"),
        },
        "5_channel_RUN3": {
            "hardening_probe_baseline_SFT": cap_hardening_5.get("capacity_pct"),
            "rssft_own_samples": cap_rssft_5.get("capacity_pct"),
        },
    }
    print("\n2x2 table (capacity_pct):", json.dumps(two_by_two, indent=1), flush=True)
    chan_effect_on_hardening = cap_hardening_5.get("capacity_pct") - cap_hardening_10.get("capacity_pct")
    chan_effect_on_rssft = cap_rssft_5.get("capacity_pct") - cap_rssft_10.get("capacity_pct")
    policy_effect_at_10ch = cap_rssft_10.get("capacity_pct") - cap_hardening_10.get("capacity_pct")
    policy_effect_at_5ch = cap_rssft_5.get("capacity_pct") - cap_hardening_5.get("capacity_pct")
    print(f"channel-count effect: +{chan_effect_on_hardening:.2f}pp (hardening policy), "
          f"+{chan_effect_on_rssft:.2f}pp (RS-SFT policy)", flush=True)
    print(f"policy effect: +{policy_effect_at_10ch:.2f}pp (10ch), "
          f"+{policy_effect_at_5ch:.2f}pp (5ch RUN3)", flush=True)

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-b (R1, R3 recomputed from code, not quoted) + R-e (R1 noise-amplification) "
                       "+ [E] a 2x2 channel-count x policy capacity table",
        "r1": {
            "reward_id": "R1", "reward": "Checker, 10-channel original aggregate",
            "policy": "SFT (hardening-probe baseline condition)",
            "source": str(HARDENING_PATH.relative_to(REPO_ROOT)),
            "source_sha256": sha256_of(HARDENING_PATH),
            "check_names": CHECK_10, "n_records": len(r1_records),
            "capacity": r1_cap, "tie_rate_check": r1_ties,
            "noise_amplification_by_channel": r1_noise.get("noise_amplified_pct_by_channel"),
            "re_prediction_noise_amplified_channel_gt_10pct": {
                "statement": "At least one R1 channel has noise-amplified live groups > 10%.",
                "passes": r1_noise_pred_passes,
            },
            "note": "This number IS finding 9's baseline (16.4%), reproduced here directly from "
                    "code with sha256 provenance, not finding 1's 14.0% (an older, different, "
                    "untracked sample source with no surviving flat-record file to recompute from).",
        },
        "r3": {
            "reward_id": "R3", "reward": "Low-temperature objective (checker + temperature_economy), finding 15",
            "policy": "SFT (hardening-probe low_temp condition)",
            "source": str(LOW_TEMP_PATH.relative_to(REPO_ROOT)),
            "source_sha256": sha256_of(LOW_TEMP_PATH),
            "capacity_baseline_10ch": r3_cap_baseline_10, "capacity_low_temp_10ch": r3_cap_lowtemp_10,
            "capacity_low_temp_11ch_incl_temperature_economy": r3_cap_lowtemp_11,
            "tie_rate_check_11ch": r3_ties_11,
            "note": "Reproduces finding 15's table (16.4% / 19.0% / 23.8%) from code.",
        },
        "two_by_two_table_E": {
            "statement": "[E] Separates channel-count from policy. Rows: 10-channel checker vs "
                         "5-channel RUN3 subset. Columns: hardening-probe baseline samples (SFT "
                         "policy, 40 targets x 8, closed-book) vs RS-SFT's own samples (75 targets "
                         "x 32, resampling_rate). Until this ran, '14% -> 27%' conflated two "
                         "changes (fewer channels, different policy) -- this isolates each.",
            "capacity_pct": two_by_two,
            "channel_count_effect_pp": {
                "on_hardening_policy": round(chan_effect_on_hardening, 2),
                "on_rssft_policy": round(chan_effect_on_rssft, 2),
            },
            "policy_effect_pp": {
                "at_10_channels": round(policy_effect_at_10ch, 2),
                "at_5_channels_RUN3": round(policy_effect_at_5ch, 2),
            },
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
