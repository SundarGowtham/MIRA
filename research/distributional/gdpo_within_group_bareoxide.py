#!/usr/bin/env python
"""
research/distributional/gdpo_within_group_bareoxide.py — Phase 15 Task
2b(b) and 2b(c): within GDPO training groups that contain BOTH a
bare-oxide and a carbonate completion, is the raw per-channel score gap
small but the z-scored advantage large (because within-group std is
small)? And how do ungradeable rates for thermodynamic_favorable differ
between bare-oxide and carbonate completions?

Source: runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl
(the actual Phase 12 training dump -- confirmed structure: 2 targets per
step, 8 completions each = 16 lines/step, matching G=8, batch 1 x accum 2
groups). A "group" here is (step, target).

Channels: RUN3_CHECKS (core/reward.py) -- the five channels Phase 12's
reward vector actually used: amount_accuracy, thermodynamic_favorable,
stoichiometry, chempot_atmosphere, operation_order.

Usage (tmux -- pure JSON processing, no model/PD calls, but the
project's "always tmux" rule has no exceptions):
  uv run python research/distributional/gdpo_within_group_bareoxide.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from validator import SynthesisValidator  # noqa: E402
from core.reward import ParseFailure, RUN3_CHECKS, parse_completion  # noqa: E402

DUMP_PATH = Path("runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl")
OUT_JSON = Path("results/distributional/gdpo_within_group_bareoxide.json")

BARE_OXIDE = {"Li2O", "Na2O", "K2O", "Rb2O", "Cs2O"}
CARBONATE = {"Li2CO3", "Na2CO3", "K2CO3", "BaCO3", "SrCO3", "CaCO3", "MgCO3"}


def classify(precursors: list[str]) -> str:
    has_bare = any(p in BARE_OXIDE for p in precursors)
    has_carb = any(p in CARBONATE for p in precursors)
    if has_bare and has_carb:
        return "both"
    if has_bare:
        return "bare"
    if has_carb:
        return "carbonate"
    return "neither"


def load_groups():
    """(step, target) -> list of {precursors, breakdown} for each completion."""
    groups = defaultdict(list)
    n_lines, n_parse_fail = 0, 0
    with DUMP_PATH.open() as f:
        for line in f:
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bd = rec.get("breakdown")
            if bd is None:
                n_parse_fail += 1
                continue
            try:
                route = parse_completion(rec["completion"], rec["target"])
                precursors = [p.formula for p in (route.precursors or [])]
            except Exception:
                n_parse_fail += 1
                continue
            groups[(rec["step"], rec["target"])].append({
                "precursors": precursors, "breakdown": bd, "cat": classify(precursors),
            })
    print(f"n_lines={n_lines}  n_parse_fail_or_no_breakdown={n_parse_fail}  "
          f"n_groups={len(groups)}")
    return groups


def main():
    groups = load_groups()

    mixed_groups = {k: v for k, v in groups.items()
                    if any(m["cat"] == "bare" for m in v) and any(m["cat"] == "carbonate" for m in v)}
    print(f"groups containing >=1 bare-oxide AND >=1 carbonate completion: "
          f"{len(mixed_groups)} / {len(groups)}")

    per_channel = {}
    for ch in RUN3_CHECKS:
        raw_gaps, z_gaps, within_std_list = [], [], []
        bare_z_all, carb_z_all = [], []
        for (step, target), members in mixed_groups.items():
            vals = []
            for m in members:
                bd = m["breakdown"]
                grade_key = f"{ch}_gradeability"
                if grade_key in bd and bd[grade_key] in SynthesisValidator.SENTINEL_TAGS:
                    v = None
                else:
                    raw = bd.get(ch)
                    v = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
                vals.append(v)
            gradeable = [v for v in vals if v is not None]
            if len(gradeable) < 2:
                continue
            mean = statistics.mean(gradeable)
            std = statistics.pstdev(gradeable)  # population std within this group of <=8
            bare_vals = [v for v, m in zip(vals, members) if m["cat"] == "bare" and v is not None]
            carb_vals = [v for v, m in zip(vals, members) if m["cat"] == "carbonate" and v is not None]
            if not bare_vals or not carb_vals:
                continue
            raw_gap = statistics.mean(bare_vals) - statistics.mean(carb_vals)
            if std > 0:
                bare_z = [(v - mean) / std for v in bare_vals]
                carb_z = [(v - mean) / std for v in carb_vals]
            else:
                bare_z = [0.0] * len(bare_vals)
                carb_z = [0.0] * len(carb_vals)
            z_gap = statistics.mean(bare_z) - statistics.mean(carb_z)
            raw_gaps.append(raw_gap)
            z_gaps.append(z_gap)
            within_std_list.append(std)
            bare_z_all.extend(bare_z)
            carb_z_all.extend(carb_z)

        per_channel[ch] = {
            "n_mixed_groups_with_both_gradeable": len(raw_gaps),
            "mean_raw_gap": statistics.mean(raw_gaps) if raw_gaps else None,
            "mean_within_group_std": statistics.mean(within_std_list) if within_std_list else None,
            "mean_z_gap": statistics.mean(z_gaps) if z_gaps else None,
            "mean_bare_z": statistics.mean(bare_z_all) if bare_z_all else None,
            "mean_carb_z": statistics.mean(carb_z_all) if carb_z_all else None,
        }

    print(f"\n{'channel':<24}{'n_groups':>9}{'raw_gap':>10}{'within_std':>12}{'z_gap':>10}{'bare_z':>9}{'carb_z':>9}")
    for ch, d in per_channel.items():
        if d["mean_raw_gap"] is None:
            print(f"{ch:<24}{'--':>9}")
            continue
        print(f"{ch:<24}{d['n_mixed_groups_with_both_gradeable']:>9}"
              f"{d['mean_raw_gap']:>10.4f}{d['mean_within_group_std']:>12.4f}"
              f"{d['mean_z_gap']:>10.3f}{d['mean_bare_z']:>9.3f}{d['mean_carb_z']:>9.3f}")

    # Step-wise trend, bucketed (dump has 334 steps; bucket into 5 ranges
    # for a readable trend rather than 334 individual points).
    print("\n-- step-wise trend, bucketed --")
    all_steps = sorted({s for s, _ in groups.keys()})
    n_buckets = 5
    bucket_edges = [all_steps[i * len(all_steps) // n_buckets] for i in range(n_buckets)] + [all_steps[-1] + 1]
    step_trend = {}
    for ch in RUN3_CHECKS:
        bucket_z = [[] for _ in range(n_buckets)]
        for (step, target), members in mixed_groups.items():
            bidx = next(i for i in range(n_buckets) if bucket_edges[i] <= step < bucket_edges[i + 1])
            vals = []
            for m in members:
                bd = m["breakdown"]
                grade_key = f"{ch}_gradeability"
                if grade_key in bd and bd[grade_key] in SynthesisValidator.SENTINEL_TAGS:
                    v = None
                else:
                    raw = bd.get(ch)
                    v = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
                vals.append(v)
            gradeable = [v for v in vals if v is not None]
            if len(gradeable) < 2:
                continue
            mean = statistics.mean(gradeable)
            std = statistics.pstdev(gradeable)
            bare_vals = [v for v, m in zip(vals, members) if m["cat"] == "bare" and v is not None]
            carb_vals = [v for v, m in zip(vals, members) if m["cat"] == "carbonate" and v is not None]
            if not bare_vals or not carb_vals or std == 0:
                continue
            bare_z = statistics.mean([(v - mean) / std for v in bare_vals])
            carb_z = statistics.mean([(v - mean) / std for v in carb_vals])
            bucket_z[bidx].append(bare_z - carb_z)
        step_trend[ch] = [statistics.mean(b) if b else None for b in bucket_z]
        ranges = [f"[{bucket_edges[i]}-{bucket_edges[i+1]})" for i in range(n_buckets)]
        vals_str = "  ".join(f"{r}:{v:.2f}" if v is not None else f"{r}:n/a"
                             for r, v in zip(ranges, step_trend[ch]))
        print(f"  {ch:<24} z_gap by step-bucket: {vals_str}")

    # 2b(c): ungradeable rate for thermodynamic_favorable, bare vs carbonate,
    # across the WHOLE dump (not just mixed groups).
    print("\n-- 2b(c): thermodynamic_favorable ungradeable rate, bare-oxide vs carbonate (whole dump) --")
    all_members = [m for members in groups.values() for m in members]
    for cat in ["bare", "carbonate"]:
        members_cat = [m for m in all_members if m["cat"] == cat]
        n = len(members_cat)
        n_ungradeable = sum(
            1 for m in members_cat
            if m["breakdown"].get("thermodynamic_favorable_gradeability") in SynthesisValidator.SENTINEL_TAGS
        )
        rate = n_ungradeable / n if n else None
        print(f"  {cat:<10} n={n:5d}  ungradeable={n_ungradeable:5d}  rate={rate:.1%}" if rate is not None
              else f"  {cat:<10} n=0")

    OUT_JSON.write_text(json.dumps({
        "n_groups_total": len(groups),
        "n_mixed_groups": len(mixed_groups),
        "per_channel": per_channel,
        "step_trend_z_gap_by_channel": step_trend,
        "bucket_edges": bucket_edges,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
