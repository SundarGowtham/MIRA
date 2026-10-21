#!/usr/bin/env python
"""
research/phase17_r5_comparator_capacity_rssft.py — Phase 17 Block R,
R5 (comparator on RS-SFT's own samples): R-b (signal/capacity,
tie-rate) and R-e (aggregation: participation ratio, conflict,
dominance, noise amplification, scale-estimator breakdown, and an
exact Shapley attribution of each pairwise decision to gates vs
channels).

NO NEW GENERATION, NO EDIT to core/comparator.py or core/ranker.py.
Source: data/novelty/rssft_reference_{heldout,astral}.jsonl,
sample_group == "resampling_rate" -- the identical 2400-row/300-group
scope as R2 (results/phase17/r2_checker_capacity_rssft.json) and R4
(results/ranker_capacity.json).

DESIGN NOTE (R5 is pairwise, R1/R2/R4 are per-sample scalar -- this is
the gap flagged at the end of the R2 report). core/comparator.py's
`compare(a, b, ...)` needs two routes; `score_channels(route, target)`
does NOT -- it returns {channel: raw unclipped value or None}, gated
by `_check_gates`, for ONE route. This script uses `score_channels`'s
per-route raw channel values, percentile-ranked against the existing
calibration scales (misc/comparator_scales_v2.json, the same table
`compare()` itself uses), as each route's own "channel contribution"
-- exactly what it would contribute to any pairwise margin it entered.
This makes R1/R2/R4's `capacity_metrics`-on-a-breakdown-dict pattern
apply directly, with no pairwise-to-scalar reduction invented: a
route's percentile score for channel c IS the per-route building block
of every compare() call it is party to.

STALE-CALIBRATION CAVEAT (flagged, not fixed): misc/comparator_scales_v2.json
was built 2026-09-18 (seed 20260916), before commit e7e1f4d (2026-10-03,
COMPARATOR_VERSION v2->v3) changed C3_reactive_temperature_window's
ceiling candidates (dropped precursor-decomposition onset, per Phase 16
Sec 2.4.1). That commit's own message confirms C6 is unaffected, and no
other channel's computation changed -- so only C3's calibration
distribution is stale; the other 5 scored channels are not. Recomputing
the scale is a separate, cheap CPU job (research/compute_comparator_scales_v2.py)
but is out of scope here; if C3 drives a headline number downstream,
redo it first.

GATE-VS-CHANNEL SHAPLEY (R-e "Gates" bullet, the one genuinely new
piece of methodology this script introduces -- documented here since
the pre-reg specifies the measurement but not the exact game). Per
unordered pair (i, j) within a group of 8: let G = 1 if BOTH routes
pass every gate, else 0 (`Ranker._check_gates`, reused via the
internal `Comparator._gater`, never modified). Let C = the
channel-only margin as if gates did not exist: sum over
SCORED_CHANNEL_NAMES of (percentile_rank(raw_i) - percentile_rank(raw_j))
for every channel where BOTH routes' raw value (computed by calling
the private `_c2`.._c7 methods DIRECTLY, bypassing `score_channels`'s
gate check) is not None. The true compare() margin is M = G * C
exactly (verified below on a subsample) -- a route that fails any gate
contributes None on every channel in the real comparator, which zeroes
every term, so gate failure forces M = 0 regardless of what the
channels would have said. For this exact product-form game with
players {Gate, Channels} and characteristic function v(None)=0,
v({Channels})=C, v({Gate})=0, v({Gate,Channels})=G*C, the Shapley
values are:
    phi_Gate    = 0.5 * (G - 1) * C
    phi_Channels = 0.5 * (G + 1) * C
which sum exactly to G*C = M (efficiency), and reduce to
(phi_Gate=0, phi_Channels=M) whenever both routes pass gates -- gates
get zero credit for a decision they did not touch -- and split the
*unrealized* channel margin C evenly (-C/2 attributed to the gate that
suppressed it, +C/2 still credited to the channels that would have
produced it) whenever a gate forces M=0 despite C != 0.

Usage (tmux, CPU only -- thermo-backed Comparator calls, no GPU):
  PYTHONPATH=. uv run python research/phase17_r5_comparator_capacity_rssft.py
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import (  # noqa: E402
    ComparatorParams,
    Comparator,
    SCORED_CHANNEL_NAMES,
    _percentile_rank,
)
from core.reward import ParseFailure, parse_completion  # noqa: E402
from probe_hardening import capacity_metrics, within_group_z  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
RSSFT_PATHS = {
    "heldout": REPO_ROOT / "data" / "novelty" / "rssft_reference_heldout.jsonl",
    "astral": REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl",
}
SCALES_PATH = REPO_ROOT / "misc" / "comparator_scales_v2.json"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "r5_comparator_capacity_rssft.json"
GROUP_SIZE = 8
STALE_CHANNELS = ["C3_reactive_temperature_window"]  # calibration predates v2->v3 fix


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


def raw_channels_nogate(comp: Comparator, route, target: str, params: ComparatorParams) -> dict:
    """Every SCORED channel's raw value computed by calling the private
    _c2.._c7 methods directly -- bypasses _check_gates entirely. None
    only for a channel's own inapplicability (e.g. C1 needs >=2
    precursors, C5 needs a volatile element present), never for a gate
    failure."""
    if route is None:
        return {c: None for c in SCORED_CHANNEL_NAMES}
    return {
        "C2_unspent_driving_force": comp._c2(route, target),
        "C3_reactive_temperature_window": comp._c3(route, target, params),
        "C4_interface_count": comp._c4(route),
        "C5_volatilization": comp._c5(route, target),
        "C6_decomposition_clearance": comp._c6(route),
        "C7_gas_evolution": comp._c7(route, target),
    }


def score_target_set(comp: Comparator, params: ComparatorParams,
                      by_target: dict[str, list[dict]], label: str) -> tuple[list[dict], int, int]:
    """One entry per sample: route, gate_pass, raw channel values
    (ungated), target_set, group key. No percentile-ranking here --
    done once, globally, after all samples are collected (so every
    sample uses the same scales regardless of which target-set loop
    produced it)."""
    samples = []
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
                gate_pass = route is not None and all(comp._gater._check_gates(route).values())
                raw = raw_channels_nogate(comp, route, target, params)
                samples.append({
                    "target": group_key, "source_target": target, "target_set": label,
                    "gate_pass": gate_pass, "raw": raw,
                })
                n_scored += 1
    return samples, n_scored, n_parse_fail


def tie_rate_stats(records: list[dict], check_names: list[str]) -> dict:
    """Same formula as research/phase17_r2_checker_capacity_rssft.py's
    tie_rate_stats: predicted tie rate from per-target pass rates
    (P_tie = p^G + (1-p)^G at a 0.5 cutoff) vs observed tie rate.
    CAVEAT carried from R2: this binary pass/fail framing is literal
    for R1/R2's [0,1]-bounded checker channels; for R5's percentile-
    ranked channels (continuous on [0,1] by construction, no natural
    "pass" semantics at 0.5) it is still computable and reported, but
    should be read as a generic liveness check, not a pass/fail model."""
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


def aggregation_diagnostics(records: list[dict], check_names: list[str],
                             resolution: float = 1e-3) -> dict:
    """R-e's measured items that capacity_metrics does not already give:
    participation ratio (PR) of the within-group channel CORRELATION
    matrix, conflict (mean pairwise cosine of channel advantage
    vectors), dominance (share of live groups where one channel >50%
    of sum_k|A_k|), noise amplification (share of live groups where a
    channel's raw within-group RANGE < resolution yet z-normalization
    still scales it to unit variance -- i.e. a "live" group by
    capacity_metrics' zero-std test, but on a range smaller than the
    channel's own stated resolution)."""
    gindex = {}
    rows_raw, rows_gid = [], []
    for r in records:
        bd = r.get("breakdown") or {}
        if bd.get("error"):
            continue
        key = r["target"]
        gindex.setdefault(key, len(gindex))
        rows_raw.append([bd.get(n) if isinstance(bd.get(n), (int, float))
                          and not isinstance(bd.get(n), bool) else np.nan
                          for n in check_names])
        rows_gid.append(gindex[key])
    if not rows_raw:
        return {}
    R = np.asarray(rows_raw, dtype=float)
    gids = np.asarray(rows_gid)
    Z, _ = within_group_z(R, gids)
    C = Z.shape[1]

    cov = np.zeros((C, C))
    for i in range(C):
        for j in range(i, C):
            both = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            if both.sum() > 2:
                cov[i, j] = cov[j, i] = np.cov(Z[both, i], Z[both, j])[0, 1]
    diag = np.diag(cov).copy()
    diag[diag <= 0] = np.nan
    denom = np.sqrt(np.outer(diag, diag))
    corr = cov / denom
    corr_valid = corr[~np.isnan(corr).any(axis=1)][:, ~np.isnan(corr).any(axis=0)] if np.isnan(corr).any() else corr
    eig = np.linalg.eigvalsh((corr_valid + corr_valid.T) / 2) if corr_valid.size else np.array([np.nan])
    eig = eig[~np.isnan(eig)]
    pr = float((eig.sum() ** 2) / (eig ** 2).sum()) if eig.size and (eig ** 2).sum() > 0 else float("nan")

    cos_vals = []
    for i in range(C):
        for j in range(i + 1, C):
            both = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            if both.sum() > 2:
                vi, vj = Z[both, i], Z[both, j]
                denom_ij = np.linalg.norm(vi) * np.linalg.norm(vj)
                if denom_ij > 1e-12:
                    cos_vals.append(float(np.dot(vi, vj) / denom_ij))
    conflict = float(np.mean(cos_vals)) if cos_vals else float("nan")

    n_live_groups = 0
    n_dominance_groups = 0
    n_noise_amplified = {c: 0 for c in range(C)}
    n_live_channel_groups = {c: 0 for c in range(C)}
    for g in np.unique(gids):
        m = gids == g
        block_raw = R[m]
        block_z = Z[m]
        any_live = False
        abs_A_sum = np.zeros(C)
        for c in range(C):
            col_raw = block_raw[:, c]
            valid = ~np.isnan(col_raw)
            if valid.sum() < 2:
                continue
            vals = col_raw[valid]
            rng = float(vals.max() - vals.min())
            z_valid = block_z[valid, c]
            is_live = np.nanstd(z_valid) > 1e-9 if z_valid.size else False
            if is_live:
                any_live = True
                n_live_channel_groups[c] += 1
                if rng < resolution:
                    n_noise_amplified[c] += 1
            abs_A_sum[c] = np.nansum(np.abs(z_valid)) if z_valid.size else 0.0
        if any_live:
            n_live_groups += 1
            total_abs = abs_A_sum.sum()
            if total_abs > 1e-9 and (abs_A_sum.max() / total_abs) > 0.5:
                n_dominance_groups += 1

    return {
        "participation_ratio": round(pr, 3) if pr == pr else None,
        "conflict_mean_pairwise_cosine": round(conflict, 4) if conflict == conflict else None,
        "n_channel_pairs_for_conflict": len(cos_vals),
        "dominance_pct_of_live_groups": round(100 * n_dominance_groups / n_live_groups, 1) if n_live_groups else None,
        "n_live_groups": n_live_groups,
        "noise_amplified_pct_by_channel": {
            check_names[c]: (round(100 * n_noise_amplified[c] / n_live_channel_groups[c], 1)
                              if n_live_channel_groups[c] else None)
            for c in range(C)
        },
        "resolution_used": resolution,
    }


def scale_estimator_breakdown(scales: dict) -> dict:
    """Share of each SCORED channel's calibration values tied at the
    median -- R5's C7 prediction is specifically that this exceeds 50%
    (finding-23-style MAD-collapse signature, now checked directly
    against the percentile-rank table instead of the retired MAD
    scale)."""
    out = {}
    for c in SCORED_CHANNEL_NAMES:
        vals = scales.get(c)
        if not vals:
            out[c] = None
            continue
        n = len(vals)
        med = vals[n // 2]
        n_tied = sum(1 for v in vals if v == med)
        out[c] = {
            "n_calibration_values": n,
            "median": med,
            "pct_tied_at_median": round(100 * n_tied / n, 1),
        }
    return out


def gate_channel_shapley(samples: list[dict], pct_lookup, max_pairs_per_group: int = 28) -> dict:
    """Exact 2-player Shapley attribution of each pairwise margin to
    {Gate, Channels} -- see module docstring for the game definition.
    Computed over every unordered pair within each group of 8
    (C(8,2)=28 pairs/group, all of them, max_pairs_per_group=28 kept
    as an explicit cap rather than assumed)."""
    by_group = defaultdict(list)
    for s in samples:
        by_group[s["target"]].append(s)

    phi_gate_sum = 0.0
    phi_channels_sum = 0.0
    n_pairs = 0
    n_gate_blocked = 0
    n_both_pass = 0
    n_both_fail = 0
    abs_C_when_blocked = []
    sanity_checks = []
    for group_key, recs in by_group.items():
        if len(recs) < 2:
            continue
        pairs = list(itertools.combinations(range(len(recs)), 2))[:max_pairs_per_group]
        for i, j in pairs:
            ri, rj = recs[i], recs[j]
            G = 1.0 if (ri["gate_pass"] and rj["gate_pass"]) else 0.0
            Cval = 0.0
            n_grad = 0
            for c in SCORED_CHANNEL_NAMES:
                va, vb = ri["raw"].get(c), rj["raw"].get(c)
                if va is None or vb is None:
                    continue
                Cval += pct_lookup(c, va) - pct_lookup(c, vb)
                n_grad += 1
            if n_grad == 0:
                continue
            phi_gate = 0.5 * (G - 1.0) * Cval
            phi_channels = 0.5 * (G + 1.0) * Cval
            phi_gate_sum += phi_gate
            phi_channels_sum += phi_channels
            n_pairs += 1
            if G == 1.0:
                n_both_pass += 1
            elif ri["gate_pass"] or rj["gate_pass"]:
                n_gate_blocked += 1
                abs_C_when_blocked.append(abs(Cval))
            else:
                n_both_fail += 1
            if n_pairs <= 20:
                sanity_checks.append({"M_shapley_sum": round(phi_gate + phi_channels, 6),
                                       "G_times_C": round(G * Cval, 6)})

    max_sanity_diff = max((abs(s["M_shapley_sum"] - s["G_times_C"]) for s in sanity_checks), default=0.0)
    return {
        "n_pairs": n_pairs,
        "n_both_pass_gates": n_both_pass,
        "n_gate_asymmetric_blocked": n_gate_blocked,
        "n_both_fail_gates": n_both_fail,
        "pct_pairs_gate_blocked": round(100 * n_gate_blocked / n_pairs, 2) if n_pairs else None,
        "mean_phi_gate": round(phi_gate_sum / n_pairs, 4) if n_pairs else None,
        "mean_phi_channels": round(phi_channels_sum / n_pairs, 4) if n_pairs else None,
        "mean_abs_channel_margin_suppressed_by_gate_asymmetry": (
            round(float(np.mean(abs_C_when_blocked)), 4) if abs_C_when_blocked else None),
        "efficiency_check_max_abs_diff_shapley_sum_vs_GxC_first20pairs": round(max_sanity_diff, 9),
    }


def main():
    print("loading comparator (mp formula set + PD cache)...", flush=True)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    from validator import ThermoChecker
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    comp = Comparator(formula_set, thermo)
    params = ComparatorParams(c3_fraction=0.17)  # the pre-registered read (arm B), matches scales' own calibration run

    print(f"loading calibration scales: {SCALES_PATH}", flush=True)
    scales_doc = json.loads(SCALES_PATH.read_text())
    scales = scales_doc["scales"]
    print(f"  built {scales_doc.get('seed')}, COMPARATOR_VERSION label in file: {scales_doc.get('comparator_version')} "
          f"(module path only, not the version string -- stale-calibration caveat applies to {STALE_CHANNELS})", flush=True)

    def pct_lookup(channel: str, value: float) -> float:
        sorted_vals = scales.get(channel)
        if not sorted_vals:
            return 0.5
        return _percentile_rank(value, sorted_vals)

    all_samples = []
    per_set_summary = {}
    for label, path in RSSFT_PATHS.items():
        print(f"\nscoring {path} (resampling_rate only, gates bypassed for raw capture)...", flush=True)
        by_target = load_resampling_records(path)
        samples, n_scored, n_parse_fail = score_target_set(comp, params, by_target, label)
        all_samples.extend(samples)
        n_gate_pass = sum(1 for s in samples if s["gate_pass"])
        print(f"  {label}: n_targets={len(by_target)} n_scored={n_scored} n_parse_fail={n_parse_fail} "
              f"n_gate_pass={n_gate_pass} ({round(100*n_gate_pass/max(n_scored,1),1)}%)", flush=True)
        per_set_summary[label] = {"n_targets": len(by_target), "n_scored": n_scored, "n_parse_fail": n_parse_fail,
                                   "n_gate_pass": n_gate_pass}

    # Build gated percentile breakdowns (the "real" comparator semantics:
    # None if gate failed, matching score_channels exactly).
    for s in all_samples:
        if s["gate_pass"]:
            bd = {c: pct_lookup(c, s["raw"][c]) if s["raw"][c] is not None else None
                  for c in SCORED_CHANNEL_NAMES}
        else:
            bd = {c: None for c in SCORED_CHANNEL_NAMES}
        s["breakdown"] = bd
        s["reward"] = sum(v for v in bd.values() if v is not None)

    print("\n" + "=" * 78)
    cap_combined = capacity_metrics(all_samples, SCORED_CHANNEL_NAMES)
    print(f"COMBINED comparator capacity (percentile-ranked channels, gated): {cap_combined.get('capacity_pct')}%")
    print("(compare: checker R2 on the SAME RS-SFT samples = 26.75%; ranker v2 (R4) = 46.31%)")

    per_set_cap = {}
    for label in RSSFT_PATHS:
        recs_label = [s for s in all_samples if s["target_set"] == label]
        per_set_cap[label] = capacity_metrics(recs_label, SCORED_CHANNEL_NAMES)
        per_set_summary[label]["capacity"] = per_set_cap[label]
        print(f"  {label}: capacity={per_set_cap[label].get('capacity_pct')}%")

    ties = tie_rate_stats(all_samples, SCORED_CHANNEL_NAMES)
    print("\n--- R-b tie-rate check (predicted vs observed, per channel) ---")
    for name, stats in ties.items():
        print(f"  {name:<32}{stats}")

    agg = aggregation_diagnostics(all_samples, SCORED_CHANNEL_NAMES)
    print("\n--- R-e aggregation diagnostics ---")
    print(json.dumps(agg, indent=1))

    scale_breakdown = scale_estimator_breakdown(scales)
    print("\n--- R-e scale-estimator breakdown (median-tie share, prediction: C7 > 50%) ---")
    for c, s in scale_breakdown.items():
        print(f"  {c:<32}{s}")

    print("\n--- R-e Gates: exact Shapley attribution (gate vs channels), all pairs within groups ---")
    shapley = gate_channel_shapley(all_samples, pct_lookup)
    print(json.dumps(shapley, indent=1))

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-b (signal/capacity, tie-rate) + R-e (aggregation: PR, conflict, dominance, "
                        "noise amplification, scale-estimator breakdown, gate-vs-channel Shapley)",
        "reward_id": "R5",
        "reward": "Comparator, core/comparator.py, 6 scored channels (C1 diagnostic-only, excluded)",
        "policy": "RS-SFT",
        "comparator_version": __import__("core.comparator", fromlist=["COMPARATOR_VERSION"]).COMPARATOR_VERSION,
        "c3_fraction": params.c3_fraction,
        "scored_channels": SCORED_CHANNEL_NAMES,
        "stale_calibration_caveat": {
            "channels_affected": STALE_CHANNELS,
            "note": ("misc/comparator_scales_v2.json built 2026-09-18, before commit e7e1f4d (2026-10-03) "
                     "changed C3's temperature-ceiling candidates (COMPARATOR_VERSION v2->v3). The commit's own "
                     "message confirms C6 is unaffected and no other channel's computation changed. "
                     "C3's percentile ranks in this report use a calibration distribution computed under the "
                     "PRE-fix C3 definition -- flagged, not recomputed, in this pass."),
        },
        "source": "data/novelty/rssft_reference_{heldout,astral}.jsonl, sample_group=resampling_rate "
                   "(NO NEW GENERATION), same 2400-row/300-group scope as R2/R4",
        "group_size": GROUP_SIZE,
        "per_target_set": per_set_summary,
        "combined": {
            "n_samples": len(all_samples),
            "n_gate_pass": sum(1 for s in all_samples if s["gate_pass"]),
            "capacity": cap_combined,
        },
        "tie_rate_check": ties,
        "aggregation_diagnostics": agg,
        "scale_estimator_breakdown": scale_breakdown,
        "gate_channel_shapley": shapley,
        "comparison_reference": {
            "checker_r2_on_rssft_samples_capacity_pct": 26.75,
            "ranker_r4_on_rssft_samples_capacity_pct": 46.31,
            "note": "This file's combined capacity_pct is R5's own first computed capacity number -- R5 was "
                    "never trained, so there is no training-log capacity to compare against (unlike R1/R2/R4).",
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
