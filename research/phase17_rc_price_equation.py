#!/usr/bin/env python
"""
research/phase17_rc_price_equation.py — Phase 17 Block R, R-c
(direction: the Price equation). PHASE17_PREREG.md:

    Delta E[phi] ~= eta * Cov_pi(A, phi)

A reward's selection differential on a feature phi -- standardised
S_phi = Cov(A, phi) / sd(phi), A = the within-group advantage GDPO
actually computes -- predicts, before training, which way practice
will push that feature. Back-tested against R1 -> GDPO-300 (run 1),
R2 -> Phase 12, R4 -> E1 (the three rewards that were actually
trained with; R5/R6 were never trained and are skipped here for the
same audit-only reason R-c's own back-test table omits them).

FEATURES (per sampled route): alkali_carbonate_used, bare_alkali_oxide
_used, ammonium_precursor_used, n_precursors, max_reported_temperature,
novel_vs_200_reference, matches_astral_rule_picked (ASTRAL targets
only -- None elsewhere).

A, PER REWARD (the EXACT within-group advantage GDPO's TRL
multi_objective_aggregation computes, per core/reward.py's own
docstring: "checks: run-3 vector by default (RUN3_CHECKS, uniform
weights)" -- i.e. A = sum over the reward's channels of each
channel's own within-group z-score, UNWEIGHTED):
  R1: 10-channel checker, z-summed, on Corpus SFT's own samples --
      runs/gdpo-qlora-gdpo-v3/generations.jsonl's step=0 records (the
      literal pre-update policy on run 1's own training prompts,
      data/rl_run3 -- a different, but equally valid, "Corpus SFT"
      sample from R1's R-b provenance (results/hardening.json's
      baseline condition); flagged, not silently swapped).
  R2: RUN3 5-channel checker, z-summed, on RS-SFT's samples (same
      2400-row scope as R2's own R-b file).
  R4: Ranker v2's own scalar score() output (gate-multiplied, as
      actually used), z-normalized within group, on RS-SFT's samples
      (same scope as R4's own Shapley file).

OBSERVED SHIFT (what to back-test S_phi's SIGN against):
  R1 -> GDPO-300: pooled early (step 0-29) vs late (step 270-299)
      cohorts of runs/gdpo-qlora-gdpo-v3/generations.jsonl itself --
      a genuine, directly-measured training-trajectory shift, not a
      finding quoted from prose. [Caveat: per-step curves are target-
      sampling noise (CLAUDE.md's Statistics rules); pooling 30
      consecutive steps on each end is the mitigation, not a fix --
      flagged as a lower-confidence back-test than R2/R4's.]
  R2 -> Phase 12: results/astral_gen_n32_rs_sft.json (init) vs
      results/astral_gen_n32_gdpo_phase12_beta0.json (Phase 12 ckpt
      300) on the identical 35 ASTRAL targets, n=32 each -- the exact
      files findings 22/24-26 are drawn from.
  R4 -> E1: results/astral_gen_n32_rs_sft.json (init) vs
      results/astral_gen_n32_e1.json (E1 ckpt 300), same targets.

NO EDIT to core/reward.py, core/ranker.py, validator.py. NO NEW
GENERATION (reuses existing training dumps and existing ASTRAL
generation comparison files). Bootstrap: cluster by target, 10,000
resamples, seed 20261016 (project standing convention).

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_rc_price_equation.py
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pymatgen.core import Composition  # noqa: E402
from core.reward import RUN3_CHECKS, parse_completion  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency  # noqa: E402
from probe_hardening import load_literature, within_group_z  # noqa: E402
from research.distributional.alkali_source_share import (  # noqa: E402
    ALKALI, BARE_OXIDE, CARBONATE, target_alkalis, classify_alkali_source,
)
from research.support_growth_by_checkpoint import build_reference  # noqa: E402
from research.ranker_capacity_rssft import TRIAGE, SYNTHESIS, RSSFT_PATHS, load_resampling_records, chunk  # noqa: E402
from validator import ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "rc_price_equation.json"
SEED = 20261016
N_BOOT = 10000
GROUP_SIZE = 8

RUN1_GENERATIONS = REPO_ROOT / "runs" / "gdpo-qlora-gdpo-v3" / "generations.jsonl"
ASTRAL_RSSFT = REPO_ROOT / "results" / "astral_gen_n32_rs_sft.json"
ASTRAL_PHASE12 = REPO_ROOT / "results" / "astral_gen_n32_gdpo_phase12_beta0.json"
ASTRAL_E1 = REPO_ROOT / "results" / "astral_gen_n32_e1.json"
ASTRAL_VALIDATION_SET = REPO_ROOT / "results" / "astral_validation_set.json"

FEATURE_NAMES = [
    "alkali_carbonate_used", "bare_alkali_oxide_used", "ammonium_precursor_used",
    "n_precursors", "max_reported_temperature", "novel_vs_200_reference",
    "matches_astral_rule_picked",
]


def astral_predicted_sets() -> dict[str, frozenset]:
    doc = json.loads(ASTRAL_VALIDATION_SET.read_text())
    return {t["target"]: frozenset(canon(p) for p in t["predicted"]) for t in doc["targets"]}


def canon(formula: str) -> str:
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return formula.strip()


def features_from_precursors(target: str, precursor_formulas: list[str], max_T, precset_ref: set | None,
                              astral_predicted: frozenset | None) -> dict:
    alks = target_alkalis(target)
    carb = any(classify_alkali_source(precursor_formulas, el) == "carbonate" for el in alks) if alks else False
    bare = any(classify_alkali_source(precursor_formulas, el) == "bare_oxide" for el in alks) if alks else False
    ammonium = any("nh4" in p.lower() for p in precursor_formulas)
    n_prec = len(precursor_formulas)
    canon_set = frozenset(canon(p) for p in precursor_formulas)
    novel = (canon_set not in precset_ref) if precset_ref is not None else None
    astral_match = (canon_set == astral_predicted) if astral_predicted is not None else None
    return {
        "alkali_carbonate_used": float(carb) if alks else None,
        "bare_alkali_oxide_used": float(bare) if alks else None,
        "ammonium_precursor_used": float(ammonium),
        "n_precursors": float(n_prec),
        "max_reported_temperature": float(max_T) if max_T is not None else None,
        "novel_vs_200_reference": float(novel) if novel is not None else None,
        "matches_astral_rule_picked": float(astral_match) if astral_match is not None else None,
    }


def cov_sd_sphi(A: list[float], phi: list[float]) -> float | None:
    pairs = [(a, p) for a, p in zip(A, phi) if p is not None]
    if len(pairs) < 10:
        return None
    a_vals = [a for a, _ in pairs]
    p_vals = [p for _, p in pairs]
    n = len(pairs)
    a_mean = sum(a_vals) / n
    p_mean = sum(p_vals) / n
    cov = sum((a - a_mean) * (p - p_mean) for a, p in pairs) / (n - 1)
    sd = (sum((p - p_mean) ** 2 for p in p_vals) / (n - 1)) ** 0.5
    if sd < 1e-12:
        return None
    return cov / sd


def bootstrap_sphi(records: list[dict], feature: str, seed: int) -> dict:
    """Cluster bootstrap by target: resample TARGETS (the real target,
    NOT the group-of-8 sub-key used for within-group z-normalization --
    a target has multiple groups, and resampling must pull them in
    together) with replacement, recompute S_phi over the resampled
    records each time."""
    by_target = defaultdict(list)
    for r in records:
        by_target[r["real_target"]].append(r)
    targets = sorted(by_target.keys())
    A_all = [r["A"] for r in records]
    phi_all = [r["features"][feature] for r in records]
    point = cov_sd_sphi(A_all, phi_all)

    rng = random.Random(seed)
    n = len(targets)
    vals = []
    for _ in range(N_BOOT):
        sample_targets = [targets[rng.randrange(n)] for _ in range(n)]
        recs = [r for t in sample_targets for r in by_target[t]]
        A = [r["A"] for r in recs]
        phi = [r["features"][feature] for r in recs]
        v = cov_sd_sphi(A, phi)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return {"point": point, "ci95": [None, None], "n_boot_valid": 0}
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
    return {"point": point, "ci95": [lo, hi], "n_boot_valid": len(vals)}


# ---------------------------------------------------------------------------
# R1: Corpus SFT's own samples from run 1's training dump (step=0)
# ---------------------------------------------------------------------------

def r1_records_and_shift() -> tuple[list[dict], dict]:
    ref_astral = build_reference("astral")
    ref_heldout = build_reference("heldout")
    astral_pred = astral_predicted_sets()

    by_target_step0 = defaultdict(list)
    early, late = [], []
    with RUN1_GENERATIONS.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            step, target, completion, bd = rec["step"], rec["target"], rec["completion"], rec["breakdown"]
            if bd is None:
                bd = {}
            try:
                route = parse_completion(completion, target)
            except Exception:
                continue
            precs = [p.formula for p in (route.precursors or [])]
            if not precs:
                continue
            max_T = None
            for op in (route.operations or []):
                ht = getattr(op.conditions, "heating_temperature", None) or []
                for v in ht:
                    if max_T is None or v > max_T:
                        max_T = v
            ref = ref_astral.get(target) or ref_heldout.get(target)
            feats = features_from_precursors(target, precs, max_T, ref, astral_pred.get(target))
            if step == 0:
                by_target_step0[target].append({"target_group": target, "real_target": target, "breakdown": bd, "features": feats})
            if step <= 29:
                early.append(feats)
            elif step >= 270:
                late.append(feats)

    # A for R1 = sum of within-group z-scores across the 10 checker channels, step=0 only.
    check_names = sorted({k for recs in by_target_step0.values() for r in recs for k in r["breakdown"]
                          if isinstance(r["breakdown"].get(k), (int, float)) and not isinstance(r["breakdown"].get(k), bool)
                          and not k.endswith("_gradeability")})
    flat = [r for recs in by_target_step0.values() for r in recs]
    import numpy as np
    rows, gids, gidx = [], [], {}
    for r in flat:
        gidx.setdefault(r["target_group"], len(gidx))
        rows.append([r["breakdown"].get(c) if isinstance(r["breakdown"].get(c), (int, float))
                     and not isinstance(r["breakdown"].get(c), bool) else np.nan for c in check_names])
        gids.append(gidx[r["target_group"]])
    R = np.asarray(rows, dtype=float)
    Z, _ = within_group_z(R, np.asarray(gids))
    A = np.nansum(Z, axis=1)
    records = [{"target_group": flat[i]["target_group"], "real_target": flat[i]["real_target"],
               "A": float(A[i]), "features": flat[i]["features"]}
               for i in range(len(flat))]

    def share_or_mean(feats_list: list[dict], feature: str) -> float | None:
        vals = [f[feature] for f in feats_list if f[feature] is not None]
        return (sum(vals) / len(vals)) if vals else None

    shift = {}
    for feat in FEATURE_NAMES:
        e, l = share_or_mean(early, feat), share_or_mean(late, feat)
        shift[feat] = {"early_step0_29": e, "late_step270_299": l,
                       "sign": (None if (e is None or l is None) else (1 if l > e else (-1 if l < e else 0)))}
    return records, shift


# ---------------------------------------------------------------------------
# R2 / R4: RS-SFT's samples (A), ASTRAL generation-comparison files (shift)
# ---------------------------------------------------------------------------

def r2_records() -> list[dict]:
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    from validator import SynthesisValidator
    validator = SynthesisValidator(formula_set, thermo_checker=thermo, validator_version=1)
    ref_astral = build_reference("astral")
    ref_heldout = build_reference("heldout")
    astral_pred = astral_predicted_sets()

    import numpy as np
    all_rows = []
    for label, path in RSSFT_PATHS.items():
        by_target = load_resampling_records(path)
        ref = ref_astral if label == "astral" else ref_heldout
        for target, recs in by_target.items():
            for gi, group in enumerate(chunk(recs, GROUP_SIZE)):
                group_key = f"{target}#g{gi}"
                group_rows = []
                for rec in group:
                    try:
                        route = parse_completion(rec["completion"], target)
                    except Exception:
                        route = None
                    if route is None:
                        continue
                    _, bd = validator.validate(route, target)
                    precs = [p.formula for p in (route.precursors or [])]
                    max_T = None
                    for op in (route.operations or []):
                        for v in (getattr(op.conditions, "heating_temperature", None) or []):
                            if max_T is None or v > max_T:
                                max_T = v
                    feats = features_from_precursors(target, precs, max_T, ref.get(target), astral_pred.get(target))
                    group_rows.append({"target_group": group_key, "real_target": target, "breakdown": bd, "features": feats})
                all_rows.extend(group_rows)

    rows, gids, gidx = [], [], {}
    for r in all_rows:
        gidx.setdefault(r["target_group"], len(gidx))
        rows.append([r["breakdown"].get(c) if isinstance(r["breakdown"].get(c), (int, float))
                     and not isinstance(r["breakdown"].get(c), bool) else np.nan for c in RUN3_CHECKS])
        gids.append(gidx[r["target_group"]])
    R = np.asarray(rows, dtype=float)
    Z, _ = within_group_z(R, np.asarray(gids))
    A = np.nansum(Z, axis=1)
    return [{"target_group": all_rows[i]["target_group"], "real_target": all_rows[i]["real_target"],
            "A": float(A[i]), "features": all_rows[i]["features"]}
            for i in range(len(all_rows))]


def r4_records() -> list[dict]:
    lit = load_literature(TRIAGE, SYNTHESIS)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)
    ref_astral = build_reference("astral")
    ref_heldout = build_reference("heldout")
    astral_pred = astral_predicted_sets()

    import numpy as np
    all_rows = []
    for label, path in RSSFT_PATHS.items():
        by_target = load_resampling_records(path)
        ref = ref_astral if label == "astral" else ref_heldout
        lit_this = {t: lit.get(t, {}) for t in by_target}
        for target, recs in by_target.items():
            lit_rec = lit_this[target]
            for gi, group in enumerate(chunk(recs, GROUP_SIZE)):
                group_key = f"{target}#g{gi}"
                group_rows = []
                for rec in group:
                    try:
                        route = parse_completion(rec["completion"], target)
                    except Exception:
                        route = None
                    if route is None:
                        continue
                    score, _info = ranker.score(route, target, lit_T=lit_rec.get("max_T"), lit_n_ops=lit_rec.get("n_ops"))
                    precs = [p.formula for p in (route.precursors or [])]
                    max_T = None
                    for op in (route.operations or []):
                        for v in (getattr(op.conditions, "heating_temperature", None) or []):
                            if max_T is None or v > max_T:
                                max_T = v
                    feats = features_from_precursors(target, precs, max_T, ref.get(target), astral_pred.get(target))
                    group_rows.append({"target_group": group_key, "real_target": target, "score": score, "features": feats})
                all_rows.extend(group_rows)

    rows, gids, gidx = [], [], {}
    for r in all_rows:
        gidx.setdefault(r["target_group"], len(gidx))
        rows.append([r["score"]])
        gids.append(gidx[r["target_group"]])
    R = np.asarray(rows, dtype=float)
    Z, _ = within_group_z(R, np.asarray(gids))
    A = Z[:, 0]
    return [{"target_group": all_rows[i]["target_group"], "real_target": all_rows[i]["real_target"],
             "A": float(A[i]) if A[i] == A[i] else 0.0,
             "features": all_rows[i]["features"]} for i in range(len(all_rows))]


def shift_from_astral_files(path_before: Path, path_after: Path) -> dict:
    ref_astral = build_reference("astral")
    astral_pred = astral_predicted_sets()

    def gather(path):
        doc = json.loads(path.read_text())
        feats_list = []
        for r in doc["results"]:
            target = r["target"]
            for s in r["samples"]:
                if s.get("match") == "PARSE_FAILURE" or not s.get("precursors"):
                    continue
                feats_list.append(features_from_precursors(
                    target, s["precursors"], s.get("max_T"), ref_astral.get(target), astral_pred.get(target)))
        return feats_list

    before, after = gather(path_before), gather(path_after)

    def share_or_mean(feats_list, feature):
        vals = [f[feature] for f in feats_list if f[feature] is not None]
        return (sum(vals) / len(vals)) if vals else None

    shift = {}
    for feat in FEATURE_NAMES:
        b, a = share_or_mean(before, feat), share_or_mean(after, feat)
        shift[feat] = {"before": b, "after": a,
                       "sign": (None if (b is None or a is None) else (1 if a > b else (-1 if a < b else 0)))}
    return shift


def main():
    results = {}

    print("=== R1 (checker, run-1 training dump) ===", flush=True)
    r1_recs, r1_shift = r1_records_and_shift()
    print(f"  n_records={len(r1_recs)}", flush=True)
    r1_sphi = {feat: bootstrap_sphi(r1_recs, feat, SEED) for feat in FEATURE_NAMES}
    for feat in FEATURE_NAMES:
        print(f"  {feat}: S_phi={r1_sphi[feat]['point']}, observed_shift={r1_shift[feat]}", flush=True)
    results["R1"] = {"reward": "Checker, 10ch (run 1)", "policy": "Corpus SFT (run1 training dump, step=0)",
                     "n_records": len(r1_recs), "s_phi": r1_sphi, "observed_shift": r1_shift,
                     "observed_shift_method": "pooled step 0-29 (early) vs step 270-299 (late) cohorts of "
                                              "runs/gdpo-qlora-gdpo-v3/generations.jsonl -- a genuine measured "
                                              "trajectory, lower-confidence than R2/R4's matched-file comparison "
                                              "because per-step target sampling is noisy (CLAUDE.md statistics rules)."}

    print("\n=== R2 (checker RUN3, RS-SFT) ===", flush=True)
    r2_recs = r2_records()
    print(f"  n_records={len(r2_recs)}", flush=True)
    r2_sphi = {feat: bootstrap_sphi(r2_recs, feat, SEED) for feat in FEATURE_NAMES}
    r2_shift = shift_from_astral_files(ASTRAL_RSSFT, ASTRAL_PHASE12)
    for feat in FEATURE_NAMES:
        print(f"  {feat}: S_phi={r2_sphi[feat]['point']}, observed_shift={r2_shift[feat]}", flush=True)
    results["R2"] = {"reward": "Checker RUN3 5ch", "policy": "RS-SFT", "n_records": len(r2_recs),
                     "s_phi": r2_sphi, "observed_shift": r2_shift,
                     "observed_shift_method": "results/astral_gen_n32_rs_sft.json (before) vs "
                                              "results/astral_gen_n32_gdpo_phase12_beta0.json (after, ckpt 300), "
                                              "identical 35 ASTRAL targets, n=32 each."}

    print("\n=== R4 (ranker v2, RS-SFT) ===", flush=True)
    r4_recs = r4_records()
    print(f"  n_records={len(r4_recs)}", flush=True)
    r4_sphi = {feat: bootstrap_sphi(r4_recs, feat, SEED) for feat in FEATURE_NAMES}
    r4_shift = shift_from_astral_files(ASTRAL_RSSFT, ASTRAL_E1)
    for feat in FEATURE_NAMES:
        print(f"  {feat}: S_phi={r4_sphi[feat]['point']}, observed_shift={r4_shift[feat]}", flush=True)
    results["R4"] = {"reward": "Ranker v2", "policy": "RS-SFT", "n_records": len(r4_recs),
                     "s_phi": r4_sphi, "observed_shift": r4_shift,
                     "observed_shift_method": "results/astral_gen_n32_rs_sft.json (before) vs "
                                              "results/astral_gen_n32_e1.json (after, ckpt 300), identical "
                                              "35 ASTRAL targets, n=32 each."}

    # ---- Prediction check: among |S_phi|>=0.1 pairs, sign matches observed shift >=75% ----
    n_checked, n_match = 0, 0
    pair_detail = []
    for reward_id in ("R1", "R2", "R4"):
        for feat in FEATURE_NAMES:
            s = results[reward_id]["s_phi"][feat]["point"]
            shift = results[reward_id]["observed_shift"][feat].get("sign")
            if s is None or shift is None or shift == 0 or abs(s) < 0.1:
                continue
            n_checked += 1
            s_sign = 1 if s > 0 else -1
            matched = (s_sign == shift)
            n_match += int(matched)
            pair_detail.append({"reward": reward_id, "feature": feat, "S_phi": round(s, 4),
                                "predicted_sign": s_sign, "observed_sign": shift, "matched": matched})

    r2_predicts_carbonate_to_bare_oxide = results["R2"]["s_phi"]["bare_alkali_oxide_used"]["point"]
    r2_predicts_temp_rise = results["R2"]["s_phi"]["max_reported_temperature"]["point"]
    r4_predicts_temp_fall = results["R4"]["s_phi"]["max_reported_temperature"]["point"]

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-c (direction: the Price equation)",
        "seed": SEED, "n_boot": N_BOOT,
        "feature_names": FEATURE_NAMES,
        "by_reward": results,
        "prediction_check": {
            "statement": "Among (reward, feature) pairs with |S_phi| >= 0.1, the sign of S_phi "
                         "matches the sign of the observed shift in at least 75%.",
            "n_pairs_checked": n_checked, "n_matched": n_match,
            "pct_matched": round(100 * n_match / n_checked, 1) if n_checked else None,
            "passes": (n_match / n_checked >= 0.75) if n_checked else None,
            "pair_detail": pair_detail,
        },
        "named_cases": {
            "r2_predicts_bare_oxide_increase": {"S_phi": r2_predicts_carbonate_to_bare_oxide,
                                                "predicted_direction": "bare_oxide UP (carbonate->bare-oxide shift)",
                                                "matches_prediction": (r2_predicts_carbonate_to_bare_oxide or 0) > 0},
            "r2_predicts_temperature_rise": {"S_phi": r2_predicts_temp_rise,
                                             "matches_prediction": (r2_predicts_temp_rise or 0) > 0},
            "r4_predicts_temperature_fall": {"S_phi": r4_predicts_temp_fall,
                                             "matches_prediction": (r4_predicts_temp_fall or 0) < 0,
                                             "note": "Opposite sign to R2's temperature prediction, per pre-reg."},
        },
        "r5_r6_skipped_note": "R5 (comparator) and R6 (novelty reward) were never trained, matching the "
                              "pre-reg's own back-test table (R-c section) which names only R1/R2/R4 -- "
                              "there is no training outcome to back-test a Price prediction against for "
                              "either, so they are skipped here rather than computed with no way to verify.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nprediction_check: {report['prediction_check']['n_matched']}/{report['prediction_check']['n_pairs_checked']} "
          f"= {report['prediction_check']['pct_matched']}% (passes={report['prediction_check']['passes']})", flush=True)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
