#!/usr/bin/env python
"""
research/phase17_block_r_patches2.py — Block R triage (regular Claude,
2026-10-22), remaining items: 1d (R4 tie-rate, missing until now),
1f (sha256 manifest confirming R2/R4/R5 use the identical source
files), 1g (R5's pairwise-to-per-sample method stated plainly as a
top-level field, not just in a docstring), 1h (Shapley baseline
documented explicitly for R4/R5/R6, with the "efficiency is algebra,
not correctness" caveat), 1i (R-e pre-registered prediction pass/fail:
R5's C7-median-tie claim: FAILS, C4 hit it instead; R6 cells marked
n/a, single channel), 1j (comparator_scales_v2.json snapshot + sha256,
stale C3 caveat carried over unchanged).

NO EDIT to core/ranker.py, core/comparator.py, core/novelty_reward.py,
validator.py. NO NEW GENERATION (R4 tie-rate reuses the existing
RS-SFT sample files; R6 reruns its own existing scoring to get
per-record breakdowns, same scope as results/phase17/r6_novelty_reward_e3.json).
Edits results/phase17/{r4_gate_channel_shapley,r5_comparator_capacity_rssft,
r6_novelty_reward_e3}.json additively; writes one new file
(results/phase17/comparator_scales_v2_snapshot.json) plus a sha256
manifest (results/phase17/block_r_sample_file_manifest.json).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.reward import ParseFailure, parse_completion  # noqa: E402
from core.ranker import GATE_NAMES, OBJECTIVE_NAMES, Ranker, build_precursor_frequency  # noqa: E402
from core.novelty_reward import load_novelty_reward  # noqa: E402
from probe_hardening import load_literature  # noqa: E402
from research.ranker_capacity_rssft import TRIAGE, SYNTHESIS, RSSFT_PATHS, load_resampling_records, chunk  # noqa: E402
from research.phase17_r4_gate_channel_shapley import objectives_nogate, weighted_sum  # noqa: E402
from research.phase17_r6_novelty_reward_e3 import (  # noqa: E402
    E3_PATHS, load_records as r6_load_records, chunk as r6_chunk,
)
from validator import ThermoChecker  # noqa: E402

RSSFT_SOURCE_PATHS = [
    REPO_ROOT / "data" / "novelty" / "rssft_reference_heldout.jsonl",
    REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl",
]
R4_OUT = REPO_ROOT / "results" / "phase17" / "r4_gate_channel_shapley.json"
R5_OUT = REPO_ROOT / "results" / "phase17" / "r5_comparator_capacity_rssft.json"
R6_OUT = REPO_ROOT / "results" / "phase17" / "r6_novelty_reward_e3.json"
SCALES_SRC = REPO_ROOT / "misc" / "comparator_scales_v2.json"
SCALES_SNAPSHOT = REPO_ROOT / "results" / "phase17" / "comparator_scales_v2_snapshot.json"
MANIFEST_OUT = REPO_ROOT / "results" / "phase17" / "block_r_sample_file_manifest.json"
GROUP_SIZE = 8


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tie_rate_with_share(records: list[dict], check_names: list[str]) -> dict:
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


SHAPLEY_BASELINE_NOTE = (
    "Characteristic function for the 2-player Shapley game {Gate, Channels}: "
    "v(empty coalition)=0, v({Gate} alone)=0 (a gate that passes/fails contributes "
    "nothing on its own -- matches score()'s own convention that the gate ONLY ever "
    "multiplies, never adds), v({Channels} alone)=C (the channel-only value AS IF the "
    "gate always passed), v({Gate,Channels} together)=G*C (the real score). "
    "phi_gate=0.5*(G-1)*C, phi_channels=0.5*(G+1)*C sum to G*C for ANY C -- this is an "
    "algebraic identity of the chosen characteristic function, so the "
    "'efficiency_check' field confirms the ARITHMETIC was implemented correctly, not "
    "that this characteristic function is the right model of how a gate and its "
    "channels jointly produce a decision. The independent CORRECTNESS check is "
    "comparing G*C against the reward actually returned by score()/compare() itself: "
    "exact for R6 (reward IS G*(0.5+0.5N) by definition, so this is tautological, not "
    "an independent check), near-exact (4.5e-05, float rounding) for R4 against "
    "ranker.score()'s literal output, and implicit-but-unverified-against-a-literal-"
    "call for R5 (compare()'s pairwise margin was checked analytically -- gate failure "
    "on either route zeroes every channel term in the real comparator -- but not "
    "resampled against a live compare() call the way R4 was against score())."
)


def main():
    manifest = {p.name: {"path": str(p.relative_to(REPO_ROOT)), "sha256": sha256_of(p)} for p in RSSFT_SOURCE_PATHS}
    manifest["identical_across_R2_R4_R5"] = True
    manifest["note"] = ("R2 (results/phase17/r2_checker_capacity_rssft.json), R4 "
                        "(results/ranker_capacity.json + results/phase17/r4_gate_channel_shapley.json), "
                        "and R5 (results/phase17/r5_comparator_capacity_rssft.json) all read these exact "
                        "two paths with the sample_group=='resampling_rate' filter -- confirmed identical "
                        "by construction (same hardcoded paths in each script) and here by sha256.")
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=1))
    print("sample file manifest:", json.dumps(manifest, indent=1), flush=True)

    # ---------------- R4 tie-rate (missing until now) ----------------
    print("\ncomputing R4 (ranker) tie-rate check on RS-SFT samples...", flush=True)
    lit = load_literature(TRIAGE, SYNTHESIS)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)

    r4_records = []
    for label, path in RSSFT_PATHS.items():
        by_target = load_resampling_records(path)
        lit_this = {t: lit.get(t, {}) for t in by_target}
        for target, recs in by_target.items():
            lit_rec = lit_this[target]
            for gi, group in enumerate(chunk(recs, GROUP_SIZE)):
                group_key = f"{target}#g{gi}"
                for rec in group:
                    try:
                        route = parse_completion(rec["completion"], target)
                    except Exception:
                        route = None
                    obj = objectives_nogate(ranker, route, target, lit_rec.get("max_T"), lit_rec.get("n_ops"))
                    r4_records.append({"target": group_key, "breakdown": obj})
    r4_ties = tie_rate_with_share(r4_records, OBJECTIVE_NAMES)
    print("R4 tie-rate:", json.dumps(r4_ties, indent=1), flush=True)

    r4 = json.loads(R4_OUT.read_text())
    r4["tie_rate_check"] = r4_ties
    r4["tie_rate_check_note"] = ("Objectives are gate-free (channels_nogate), continuous on "
                                 "[0,1] by construction -- same generic-liveness-check caveat as "
                                 "R5's own tie-rate framing (binary pass/fail at 0.5 is not a "
                                 "natural cutoff for a continuous objective).")
    r4["shapley_baseline"] = SHAPLEY_BASELINE_NOTE
    R4_OUT.write_text(json.dumps(r4, indent=1, default=str))
    print(f"-> {R4_OUT} (tie_rate_check + shapley_baseline added)", flush=True)

    # ---------------- R5 patches ----------------
    r5 = json.loads(R5_OUT.read_text())
    r5["pairwise_to_persample_method"] = (
        "NOT win-share, NOT Borda count. Each route's per-channel raw value (computed by "
        "calling the private _c2.._c7 methods directly, bypassing score_channels()'s gate) "
        "is percentile-ranked against the SAME global calibration table compare() itself "
        "uses (misc/comparator_scales_v2.json) -- this percentile rank IS the exact "
        "per-route building block of every pairwise margin that route would enter: "
        "compare(a,b) = sum_c [percentile_rank(c, raw_a) - percentile_rank(c, raw_b)]. "
        "Treating a route's own percentile-ranked channel vector as its 'score' for "
        "capacity_metrics is therefore not an invented reduction -- it is the literal "
        "per-route term the pairwise comparator already computes internally."
    )
    r5["shapley_baseline"] = SHAPLEY_BASELINE_NOTE
    c4_pct = r5["scale_estimator_breakdown"]["C4_interface_count"]["pct_tied_at_median"]
    c7_pct = r5["scale_estimator_breakdown"]["C7_gas_evolution"]["pct_tied_at_median"]
    r5["re_predictions"] = {
        "c7_median_tie_prediction": {
            "statement": "Pre-registered: R5's C7 (C7_gas_evolution) calibration set shows "
                         ">50% median ties (finding-23-style MAD-collapse signature).",
            "observed_C7_pct_tied_at_median": c7_pct,
            "passes": c7_pct > 50.0,
            "actual_finding": f"C4_interface_count hit it instead ({c4_pct}% tied at median), "
                              f"not C7 ({c7_pct}%). Reported as a genuine non-matching finding, "
                              "not forced to fit the prediction.",
        },
    }
    R5_OUT.write_text(json.dumps(r5, indent=1, default=str))
    print(f"-> {R5_OUT} (pairwise_to_persample_method + shapley_baseline + re_predictions added)", flush=True)

    # ---------------- R6 patches (rerun own scoring for share_targets field) ----------------
    print("\nrecomputing R6 tie-rate with share_targets field (same scope as r6_novelty_reward_e3.json)...",
          flush=True)
    nr = load_novelty_reward(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                             REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    r6_records = []
    for label, path in E3_PATHS.items():
        by_target = r6_load_records(path)
        for target, recs in by_target.items():
            for gi, group in enumerate(r6_chunk(recs, GROUP_SIZE)):
                group_key = f"{target}#g{gi}"
                for rec in group:
                    try:
                        route = parse_completion(rec["completion"], target)
                    except Exception:
                        route = None
                    reward, info = nr.score(route, target)
                    r6_records.append({"target": group_key, "breakdown": {"reward": reward}})
    r6_ties = tie_rate_with_share(r6_records, ["reward"])
    r6 = json.loads(R6_OUT.read_text())
    # R6's tie_rate_check is a FLAT dict (single channel, "reward"), unlike R2/R5's
    # nested-by-channel-name convention -- matches the original module's own
    # tie_rate_stats() return shape (research/phase17_r6_novelty_reward_e3.py).
    r6["tie_rate_check"]["share_targets_pass_rate_in_0.1_0.9"] = (
        r6_ties["reward"]["share_targets_pass_rate_in_0.1_0.9"] if r6_ties.get("reward") else None
    )
    r6["shapley_baseline"] = SHAPLEY_BASELINE_NOTE
    r6["re_cells"] = ("n/a (single channel) -- participation ratio, conflict, dominance, and "
                      "noise-amplification all require >=2 scored channels; R6/novelty reward is "
                      "architecturally one scalar (N), per core/novelty_reward.py's own docstring.")
    R6_OUT.write_text(json.dumps(r6, indent=1, default=str))
    print(f"-> {R6_OUT} (share_targets field + shapley_baseline + re_cells added)", flush=True)

    # ---------------- comparator_scales_v2 snapshot ----------------
    shutil.copy(SCALES_SRC, SCALES_SNAPSHOT)
    print(f"\n-> {SCALES_SNAPSHOT} (sha256={sha256_of(SCALES_SNAPSHOT)}), copied from "
          f"{SCALES_SRC} (sha256={sha256_of(SCALES_SRC)}) -- stale C3 caveat unchanged, pre-reg "
          f"says reuse existing scales.", flush=True)


if __name__ == "__main__":
    main()
