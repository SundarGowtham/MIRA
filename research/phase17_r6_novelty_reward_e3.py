#!/usr/bin/env python
"""
research/phase17_r6_novelty_reward_e3.py — Phase 17 Block R, R6
(novelty reward, Phase 16, on E3's own samples): R-b (signal/capacity,
tie-rate) and R-e's Gates column, for the one reward this project
built but never actually trained against (E4, which would have used
it, never ran per Phase 16's pre-registered stopping rule -- finding
31/CLAUDE.md). AUDIT ONLY: there is no training outcome to back-test
against, unlike R1/R2/R4; this closes Block 0's R6 gap (no
novelty-scored sample file existed) but the result is read as "what
this reward would have said," not "what it caused."

core/novelty_reward.py::NoveltyReward.score() is R = G * (0.5 + 0.5*N),
the SAME gate x channel product form as R4's ranker and R5's
comparator -- a single scalar channel ("the novelty reward is one
scalar by spec, not a factored vector," per that module's own
docstring), so capacity here is the N=1 degenerate case of R-b's own
Fisher-theorem formula: capacity_pct = 0.875 x share-of-live-groups
exactly, for one channel. The gate-vs-channel Shapley reuses the exact
2-player game from phase17_r5_comparator_capacity_rssft.py /
phase17_r4_gate_channel_shapley.py: G = the 5-subgate AND, C = (0.5 +
0.5*N) computed unconditionally (N's two free functions, `rarity` and
`set_novelty`, are never themselves gated -- only `score()`'s final
`if not gate: return 0.0` is bypassed here).

NO EDIT to core/novelty_reward.py. NO NEW GENERATION: E3's own raw
samples, data/novelty/e3_{astral,heldout}_n32.jsonl (35 + 40 targets,
32 samples each, no sample_group filter needed -- confirmed these
files hold only the resampling draws, unlike rssft_reference_*.jsonl's
mixed reference/resampling_rate rows).

Usage (tmux, CPU only -- thermo-backed NoveltyReward calls, no GPU):
  PYTHONPATH=. uv run python research/phase17_r6_novelty_reward_e3.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.novelty_reward import (  # noqa: E402
    NOVELTY_REWARD_VERSION, load_novelty_reward, rarity, set_novelty,
)
from core.reward import ParseFailure, parse_completion  # noqa: E402
from probe_hardening import capacity_metrics  # noqa: E402
from validator import SynthesisValidator  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
E3_PATHS = {
    "heldout": REPO_ROOT / "data" / "novelty" / "e3_heldout_n32.jsonl",
    "astral": REPO_ROOT / "data" / "novelty" / "e3_astral_n32.jsonl",
}
OUT_PATH = REPO_ROOT / "results" / "phase17" / "r6_novelty_reward_e3.json"
GROUP_SIZE = 8


def load_records(path: Path) -> dict[str, list[dict]]:
    by_target: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_target[rec["target"]].append(rec)
    return by_target


def chunk(lst: list, n: int) -> list[list]:
    return [lst[i:i + n] for i in range(0, len(lst), n) if len(lst[i:i + n]) == n]


def channel_free(nr, route, target_formula: str) -> float | None:
    """C: (0.5 + 0.5*N) computed unconditionally, bypassing score()'s
    gate early-return. rarity/set_novelty are free functions, never
    gated internally -- only score()'s final `if not gate` short-
    circuits them in the real reward."""
    if route is None or not route.precursors:
        return None
    rarities = [rarity(nr.precursor_freq.get(
        SynthesisValidator._normalize_formula(p.formula), 0)) for p in route.precursors]
    max_rarity = max(rarities) if rarities else 0.0
    set_nov = set_novelty([p.formula for p in route.precursors], target_formula, nr.target_sets)
    n = 0.7 * max_rarity + 0.3 * set_nov
    return 0.5 + 0.5 * n


def tie_rate_stats(records: list[dict], channel: str = "reward") -> dict:
    by_group = defaultdict(list)
    for r in records:
        bd = r.get("breakdown") or {}
        if bd.get("error"):
            continue
        v = bd.get(channel)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            by_group[r["target"]].append(v)
    n_groups, n_ties, pass_rates = 0, 0, []
    for gvals in by_group.values():
        if len(gvals) < 2:
            continue
        n_groups += 1
        if max(gvals) == min(gvals):
            n_ties += 1
        pass_rates.append(sum(1 for v in gvals if v >= 0.5) / len(gvals))
    if n_groups == 0:
        return {}
    observed = n_ties / n_groups
    g = GROUP_SIZE
    predicted = sum((p ** g + (1 - p) ** g) for p in pass_rates) / len(pass_rates)
    return {
        "n_groups": n_groups, "observed_tie_rate": round(observed, 4),
        "predicted_tie_rate_from_pass_rates": round(predicted, 4),
        "agrees_within_5pp": abs(observed - predicted) <= 0.05,
        "caveat": "Nearly every gate-passed sample scores in [0.5, 1.0] by construction "
                  "(reward = G*(0.5+0.5N)); the 0.5 pass/fail cutoff this formula assumes "
                  "is therefore almost exactly the gate boundary itself, not an independent "
                  "channel threshold -- read this as a gate-liveness check, not a channel one.",
    }


def main():
    print(f"novelty reward version: {NOVELTY_REWARD_VERSION}", flush=True)
    nr = load_novelty_reward(
        REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
        REPO_ROOT / "data" / "cache" / "pd_index.json",
        REPO_ROOT,
    )

    all_records = []
    per_set_summary = {}
    n_gate_pass_total, n_pairs_total = 0, 0
    phi_gate_sum, phi_channels_sum = 0.0, 0.0
    n_gate_fail, abs_C_when_gate_fail = 0, []
    sanity = []

    for label, path in E3_PATHS.items():
        print(f"\nscoring {path}...", flush=True)
        by_target = load_records(path)
        records = []
        n_scored, n_parse_fail, n_gate_pass = 0, 0, 0
        for target, recs in by_target.items():
            for gi, group in enumerate(chunk(recs, GROUP_SIZE)):
                group_key = f"{target}#g{gi}"
                for rec in group:
                    try:
                        route = parse_completion(rec["completion"], target)
                    except Exception:
                        route = None
                        n_parse_fail += 1
                    reward, info = nr.score(route, target)
                    records.append({"target": group_key, "breakdown": {"reward": reward, **info}, "reward": reward})
                    n_scored += 1
                    if info.get("gate"):
                        n_gate_pass += 1

                    # Gate-vs-channel Shapley (reuses the exact 2-player game; see module docstring).
                    Cval = channel_free(nr, route, target)
                    if Cval is None:
                        continue
                    G = 1.0 if info.get("gate") else 0.0
                    phi_gate = 0.5 * (G - 1.0) * Cval
                    phi_channels = 0.5 * (G + 1.0) * Cval
                    phi_gate_sum += phi_gate
                    phi_channels_sum += phi_channels
                    n_pairs_total += 1
                    if not info.get("gate"):
                        n_gate_fail += 1
                        abs_C_when_gate_fail.append(abs(Cval))
                    if route is not None and len(sanity) < 20:
                        sanity.append({"phi_sum": round(phi_gate + phi_channels, 6),
                                      "G_times_C": round(G * Cval, 6), "actual_reward": reward})
        all_records.extend(records)
        n_gate_pass_total += n_gate_pass
        cap = capacity_metrics(records, ["reward"])
        per_set_summary[label] = {"n_targets": len(by_target), "n_scored": n_scored,
                                   "n_parse_fail": n_parse_fail, "n_gate_pass": n_gate_pass,
                                   "pct_gate_pass": round(100 * n_gate_pass / max(n_scored, 1), 1),
                                   "capacity": cap}
        print(f"  {label}: n_scored={n_scored} n_parse_fail={n_parse_fail} "
              f"n_gate_pass={n_gate_pass} ({per_set_summary[label]['pct_gate_pass']}%) "
              f"capacity={cap.get('capacity_pct')}%", flush=True)

    print("\n" + "=" * 78)
    cap_combined = capacity_metrics(all_records, ["reward"])
    print(f"COMBINED novelty-reward capacity (N=1 degenerate case): {cap_combined.get('capacity_pct')}%")
    ties = tie_rate_stats(all_records)
    print(f"tie-rate check: {ties}")

    max_sanity_shapley = max((abs(s["phi_sum"] - s["G_times_C"]) for s in sanity), default=0.0)

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-b (signal/capacity, tie-rate) + R-e Gates (Shapley), AUDIT ONLY -- E4 never ran",
        "reward_id": "R6",
        "reward": "Novelty reward (Phase 16), core/novelty_reward.py",
        "policy": "E3",
        "novelty_reward_version": NOVELTY_REWARD_VERSION,
        "source": "data/novelty/e3_{astral,heldout}_n32.jsonl (NO NEW GENERATION, no sample_group filter needed)",
        "group_size": GROUP_SIZE,
        "per_target_set": per_set_summary,
        "combined": {"n_records": len(all_records), "n_gate_pass": n_gate_pass_total, "capacity": cap_combined},
        "tie_rate_check": ties,
        "gate_channel_shapley": {
            "n_samples": n_pairs_total, "n_gate_fail": n_gate_fail,
            "pct_gate_fail": round(100 * n_gate_fail / n_pairs_total, 2) if n_pairs_total else None,
            "mean_phi_gate": round(phi_gate_sum / n_pairs_total, 4) if n_pairs_total else None,
            "mean_phi_channels": round(phi_channels_sum / n_pairs_total, 4) if n_pairs_total else None,
            "mean_abs_channel_value_suppressed_by_gate_failure": (
                round(sum(abs_C_when_gate_fail) / len(abs_C_when_gate_fail), 4) if abs_C_when_gate_fail else None),
            "efficiency_check_max_abs_diff_phi_sum_vs_GxC_first20": round(max_sanity_shapley, 9),
        },
        "audit_only_caveat": "E3 was trained with checker-validity self-distillation, NOT this novelty "
                             "reward -- this file measures what the novelty reward would have scored on "
                             "E3's existing samples, not an outcome the novelty reward caused. E4 (which "
                             "would have trained against it) never ran, per Phase 16's pre-registered "
                             "stopping rule. Read as audit-only throughout.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
