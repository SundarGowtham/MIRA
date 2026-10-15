#!/usr/bin/env python
"""
research/phase16_e3_e3c_final_battery.py — Phase 16 closing battery for
E3/E3c, per regular Claude's stopping decision (2026-10-15): prediction
2 is a reportable null (docs/phases/PHASE16_RESULTS.md, 2026-10-15
entry); E4 and E3 round 2 are NOT run. Before closing the phase, finish
the pre-registered battery on EXISTING samples only, CPU only, no new
training or sampling:

  1. Cluster bootstrap by target (10,000 resamples) on ASTRAL support
     growth: E3 vs base's own resampling floor, and E3c vs base's own
     floor, separately.
  2. Checker (validator) pass rate at bar 0.9, restricted to "new
     recipes" (gate-passing AND novel, same definition as the round-1
     filter and the 2026-10-15 evaluation), for E3 and E3c, on both
     target sets.
  3. ASTRAL rule-picked hits (clean-22 and full-35) and carbonate share,
     for E3 and E3c -- reusing research/phase16_astral_clean_subset.py's
     own load_hits/load_carbonate_shares against a freshly-built
     astral_gen_n32-shaped file for each, produced from the EXISTING
     data/novelty/{e3,e3c}_astral_n32.jsonl completions (no new
     generation -- same completions already scored in the 2026-10-15
     entry, just re-packaged into the standard astral_gen_n32_*.json
     shape so the existing clean-subset tooling applies unmodified).

Usage (tmux, CPU -- thermo-backed validator calls, no GPU needed):
  uv run python research/phase16_e3_e3c_final_battery.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))
sys.path.insert(0, str(REPO_ROOT / "research" / "distributional"))

from validator import SynthesisValidator  # noqa: E402
from core.reward import parse_completion, ParseFailure  # noqa: E402
from core.novelty_reward import load_novelty_reward  # noqa: E402
from phase16_astral_clean_subset import load_hits, load_carbonate_shares  # noqa: E402

RARITY_COUNT_THRESHOLD = 17
CHECKER_BAR = 0.9
SEED = 42
N_BOOT = 10000

ASTRAL_VALIDATION_PATH = REPO_ROOT / "results" / "astral_validation_set.json"
OVERLAP_PATH = REPO_ROOT / "results" / "phase16" / "astral_overlap.json"

GEN_JSONL = {
    ("e3", "heldout"): REPO_ROOT / "data" / "novelty" / "e3_heldout_n32.jsonl",
    ("e3", "astral"): REPO_ROOT / "data" / "novelty" / "e3_astral_n32.jsonl",
    ("e3c", "heldout"): REPO_ROOT / "data" / "novelty" / "e3c_heldout_n32.jsonl",
    ("e3c", "astral"): REPO_ROOT / "data" / "novelty" / "e3c_astral_n32.jsonl",
}
BASE_REF_PATHS = {
    "heldout": REPO_ROOT / "data" / "novelty" / "base_reference_heldout.jsonl",
    "astral": REPO_ROOT / "data" / "novelty" / "base_reference_astral.jsonl",
}
ASTRAL_GEN_OUT = {
    "e3": REPO_ROOT / "results" / "astral_gen_n32_e3.json",
    "e3c": REPO_ROOT / "results" / "astral_gen_n32_e3c.json",
}
OUT_PATH = REPO_ROOT / "results" / "phase16" / "e3_e3c_final_battery.json"


def precset_of(route) -> tuple:
    return tuple(sorted({SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}))


def route_summary(route):
    precs = {SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}
    temps = []
    for op in route.operations:
        temps.extend(op.conditions.heating_temperature or [])
    return precs, (max(temps) if temps else None)


def classify(precs: set, trad: set, pred: set) -> str:
    if precs == trad:
        return "TRADITIONAL"
    if precs == pred:
        return "PREDICTED"
    if pred <= precs:
        return "PREDICTED_SUPERSET"
    if trad <= precs:
        return "TRADITIONAL_SUPERSET"
    return "OTHER"


def load_jsonl(path: Path, target_filter=None) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if target_filter is not None and not target_filter(rec):
                continue
            recs.append(rec)
    return recs


def is_novel(nr, route, info: dict) -> bool:
    has_rare = any(
        nr.precursor_freq.get(SynthesisValidator._normalize_formula(p.formula), 0) <= RARITY_COUNT_THRESHOLD
        for p in route.precursors
    )
    return has_rare or (info.get("set_novelty") == 1.0)


# ---------------------------------------------------------------------------
# 1. Cluster bootstrap: E3 vs base floor, E3c vs base floor, on ASTRAL
# ---------------------------------------------------------------------------

def per_target_support_counts(nr, recs: list[dict], ref_by_target: dict[str, set]) -> dict[str, dict]:
    precsets_by_target: dict[str, list[tuple]] = defaultdict(list)
    for rec in recs:
        try:
            route = parse_completion(rec["completion"], rec["target"])
        except Exception:
            continue
        precsets_by_target[rec["target"]].append(precset_of(route))
    out = {}
    for target, precsets in precsets_by_target.items():
        distinct = set(precsets)
        ref = ref_by_target.get(target, set())
        out[target] = {"n_absent": len(distinct - ref), "n_distinct": len(distinct)}
    return out


def pooled_support(counts: dict[str, dict], targets: list[str]) -> float | None:
    num = sum(counts[t]["n_absent"] for t in targets if t in counts)
    den = sum(counts[t]["n_distinct"] for t in targets if t in counts)
    return (num / den) if den else None


def bootstrap_vs_floor(model_counts: dict, floor_counts: dict, all_targets: list[str],
                       n_boot: int, rng: random.Random) -> dict:
    diffs = []
    n = len(all_targets)
    for _ in range(n_boot):
        sample = [all_targets[rng.randrange(n)] for _ in range(n)]
        m_val = pooled_support(model_counts, sample)
        f_val = pooled_support(floor_counts, sample)
        if m_val is None or f_val is None:
            continue
        diffs.append(m_val - f_val)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    point = pooled_support(model_counts, all_targets) - pooled_support(floor_counts, all_targets)
    return {"point_diff": point, "ci95": [lo, hi], "n_boot_valid": len(diffs)}


def run_bootstrap_vs_floor(nr) -> dict:
    ref_recs = load_jsonl(BASE_REF_PATHS["astral"], target_filter=lambda r: r.get("sample_group") == "reference")
    resamp_recs = load_jsonl(BASE_REF_PATHS["astral"], target_filter=lambda r: r.get("sample_group") == "resampling_rate")

    ref_by_target: dict[str, set] = defaultdict(set)
    for rec in ref_recs:
        try:
            route = parse_completion(rec["completion"], rec["target"])
        except Exception:
            continue
        ref_by_target[rec["target"]].add(precset_of(route))
    ref_by_target = dict(ref_by_target)

    floor_counts = per_target_support_counts(nr, resamp_recs, ref_by_target)
    e3_recs = load_jsonl(GEN_JSONL[("e3", "astral")])
    e3c_recs = load_jsonl(GEN_JSONL[("e3c", "astral")])
    e3_counts = per_target_support_counts(nr, e3_recs, ref_by_target)
    e3c_counts = per_target_support_counts(nr, e3c_recs, ref_by_target)

    all_targets = sorted(set(floor_counts) | set(e3_counts) | set(e3c_counts))
    rng1 = random.Random(SEED)
    e3_result = bootstrap_vs_floor(e3_counts, floor_counts, all_targets, N_BOOT, rng1)
    rng2 = random.Random(SEED + 1)
    e3c_result = bootstrap_vs_floor(e3c_counts, floor_counts, all_targets, N_BOOT, rng2)

    print(f"[1] ASTRAL support growth bootstrap (n_targets={len(all_targets)}):")
    print(f"    E3 vs base floor: point={e3_result['point_diff']:.4f} CI95={e3_result['ci95']}")
    print(f"    E3c vs base floor: point={e3c_result['point_diff']:.4f} CI95={e3c_result['ci95']}")
    return {"e3_vs_floor": e3_result, "e3c_vs_floor": e3c_result,
           "floor_pooled": pooled_support(floor_counts, all_targets),
           "e3_pooled": pooled_support(e3_counts, all_targets),
           "e3c_pooled": pooled_support(e3c_counts, all_targets)}


# ---------------------------------------------------------------------------
# 2. Checker pass rate restricted to new recipes
# ---------------------------------------------------------------------------

def checker_pass_rate_new_recipes(nr) -> dict:
    result = {}
    for target_set in ("heldout", "astral"):
        for checkpoint in ("e3", "e3c"):
            recs = load_jsonl(GEN_JSONL[(checkpoint, target_set)])
            n_new = 0
            n_pass = 0
            for rec in recs:
                target = rec["target"]
                try:
                    route = parse_completion(rec["completion"], target)
                except Exception:
                    continue
                try:
                    _, info = nr.score(route, target)
                except Exception:
                    continue
                if not info["gate"] or not is_novel(nr, route, info):
                    continue
                n_new += 1
                reward, _ = nr.validator.validate(route, target)
                if reward >= CHECKER_BAR:
                    n_pass += 1
            rate = round(n_pass / n_new, 4) if n_new else None
            result[f"{checkpoint}_{target_set}"] = {
                "n_new_recipes": n_new, "n_pass_at_0.9": n_pass, "pass_rate": rate,
            }
            print(f"[2] checker pass rate, new recipes, {checkpoint}/{target_set}: "
                  f"{n_pass}/{n_new} = {rate}")
    return result


# ---------------------------------------------------------------------------
# 3. ASTRAL rule-picked hits + carbonate share, clean-22 and full-35
# ---------------------------------------------------------------------------

def build_astral_gen_shape(validator, checkpoint: str) -> dict:
    astral_data = json.loads(ASTRAL_VALIDATION_PATH.read_text())
    recs_by_target: dict[str, list[dict]] = defaultdict(list)
    for rec in load_jsonl(GEN_JSONL[(checkpoint, "astral")]):
        recs_by_target[rec["target"]].append(rec)

    results = []
    for t in astral_data["targets"]:
        target = t["target"]
        trad = {SynthesisValidator._normalize_formula(f) for f in t["traditional"]}
        pred = {SynthesisValidator._normalize_formula(f) for f in t["predicted"]}
        samples = []
        for rec in recs_by_target.get(target, []):
            try:
                route = parse_completion(rec["completion"], target)
                reward, _ = validator.validate(route, target)
                precs, max_T = route_summary(route)
                match = classify(precs, trad, pred)
            except Exception:
                reward, precs, max_T, match = None, set(), None, "PARSE_FAILURE"
            samples.append({"precursors": sorted(precs), "max_T": max_T,
                            "reward": reward, "match": match})
        results.append({
            "target": target, "traditional": t["traditional"], "predicted": t["predicted"],
            "trad_best_T": t["trad_best_T"], "pred_best_T": t["pred_best_T"],
            "trad_best_purity": t["trad_best_purity"], "pred_best_purity": t["pred_best_purity"],
            "samples": samples,
        })
    return {"tag": checkpoint, "checkpoint": checkpoint, "results": results}


def hits_and_carbonate(validator) -> dict:
    overlap = json.loads(OVERLAP_PATH.read_text())
    overlapping = set(overlap["union_of_all_overlapping_targets"])

    result = {}
    for checkpoint in ("e3", "e3c"):
        doc = build_astral_gen_shape(validator, checkpoint)
        ASTRAL_GEN_OUT[checkpoint].write_text(json.dumps(doc, indent=1))
        print(f"[3] wrote {ASTRAL_GEN_OUT[checkpoint]}")

        hits = {r["target"]: any(s.get("match") in ("PREDICTED", "PREDICTED_SUPERSET") for s in r["samples"])
               for r in doc["results"]}
        all_targets = set(hits.keys())
        clean_targets = all_targets - overlapping
        n_full = sum(hits.values())
        n_clean = sum(hits[t] for t in clean_targets)

        cs_full = load_carbonate_shares(ASTRAL_GEN_OUT[checkpoint], all_targets)
        cs_clean = load_carbonate_shares(ASTRAL_GEN_OUT[checkpoint], clean_targets)

        print(f"[3] {checkpoint}: predicted hits clean={n_clean}/{len(clean_targets)} "
              f"full={n_full}/{len(all_targets)}; carbonate share full={cs_full['shares']['carbonate']} "
              f"clean={cs_clean['shares']['carbonate']}; bare_oxide full={cs_full['shares']['bare_oxide']}")

        result[checkpoint] = {
            "predicted_hits_clean": f"{n_clean}/{len(clean_targets)}",
            "predicted_hits_full": f"{n_full}/{len(all_targets)}",
            "carbonate_shares": {"full_35": cs_full, "clean_subset": cs_clean},
        }
    return result


def main():
    nr = load_novelty_reward(
        formula_set_path=REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
        pd_index_path=REPO_ROOT / "data" / "cache" / "pd_index.json",
        project_root=REPO_ROOT,
    )
    report = {}
    report["1_astral_support_growth_bootstrap"] = run_bootstrap_vs_floor(nr)
    report["2_checker_pass_rate_new_recipes"] = checker_pass_rate_new_recipes(nr)
    report["3_astral_hits_and_carbonate"] = hits_and_carbonate(nr.validator)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
