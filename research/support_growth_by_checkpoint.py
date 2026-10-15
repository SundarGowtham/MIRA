#!/usr/bin/env python
"""
research/support_growth_by_checkpoint.py — regular Claude's "does the
new-recipe rate change over practice" task (2026-10-15): does support
growth rise, fall, or stay flat across training (RS-SFT init ->
checkpoint 100 -> 200 -> 300), for both E1 (ranker reward) and Phase 12
(validator reward, same RS-SFT init)?

READING RULE, stated before looking at the numbers (per instruction):
flat or falling support growth across checkpoints = no trend to
extrapolate, practice is not finding new support over time. Rising
AND clearing RS-SFT's own resampling floor (10.5% ASTRAL / 11.7%
held-out) at a later checkpoint = a real trend worth reporting and
extrapolating.

Sources (new sampling only for checkpoints that didn't already exist --
see data_curation/phase16_support_growth_trajectory_sampling.py's own
docstring for exactly which):
  rs_sft (step 0):     data/novelty/rssft_reference_{heldout,astral}.jsonl
  e1_ckpt100/200:      data/novelty/e1_ckpt{100,200}_{heldout,astral}_n32.jsonl
  e1_ckpt300 (final):  data/novelty/e1_heldout_n32.jsonl + results/astral_gen_n32_e1.json
  phase12_ckpt100/200: data/novelty/phase12_ckpt{100,200}_{heldout,astral}_n32.jsonl
  phase12_ckpt300:     data/novelty/phase12_ckpt300_heldout_n32.jsonl (new)
                       + results/astral_gen_n32_gdpo_phase12_beta0.json (existing)

Reference for support growth, every checkpoint: RS-SFT's own 200-sample
draw (data/novelty/rssft_reference_{heldout,astral}.jsonl,
sample_group=="reference") -- both E1 and Phase 12 are RS-SFT-
initialized, so this is the correct, single reference throughout
(matching the existing E1 support-growth analysis's own protocol).

Usage (tmux, CPU -- thermo-backed validator calls, no GPU needed):
  uv run python research/support_growth_by_checkpoint.py
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
from core.reward import parse_completion, load_validator  # noqa: E402
from phase16_astral_clean_subset import load_hits, load_carbonate_shares  # noqa: E402

SEED = 42
N_BOOT = 10000

HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
ASTRAL_VALIDATION_PATH = REPO_ROOT / "results" / "astral_validation_set.json"
OVERLAP_PATH = REPO_ROOT / "results" / "phase16" / "astral_overlap.json"

RSSFT_PATHS = {
    "heldout": REPO_ROOT / "data" / "novelty" / "rssft_reference_heldout.jsonl",
    "astral": REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl",
}

# (checkpoint_label, target_set) -> either a raw jsonl Path, or an
# ("enriched", Path) tuple pointing at an existing astral_gen_n32-shaped
# file (has "results"[i]["samples"][j]["precursors"]).
SOURCES = {
    ("rs_sft", "heldout"): ("raw_reference", RSSFT_PATHS["heldout"]),
    ("rs_sft", "astral"): ("raw_reference", RSSFT_PATHS["astral"]),
    ("e1_ckpt100", "heldout"): ("raw", REPO_ROOT / "data" / "novelty" / "e1_ckpt100_heldout_n32.jsonl"),
    ("e1_ckpt100", "astral"): ("raw", REPO_ROOT / "data" / "novelty" / "e1_ckpt100_astral_n32.jsonl"),
    ("e1_ckpt200", "heldout"): ("raw", REPO_ROOT / "data" / "novelty" / "e1_ckpt200_heldout_n32.jsonl"),
    ("e1_ckpt200", "astral"): ("raw", REPO_ROOT / "data" / "novelty" / "e1_ckpt200_astral_n32.jsonl"),
    ("e1_ckpt300", "heldout"): ("raw", REPO_ROOT / "data" / "novelty" / "e1_heldout_n32.jsonl"),
    ("e1_ckpt300", "astral"): ("enriched", REPO_ROOT / "results" / "astral_gen_n32_e1.json"),
    ("phase12_ckpt100", "heldout"): ("raw", REPO_ROOT / "data" / "novelty" / "phase12_ckpt100_heldout_n32.jsonl"),
    ("phase12_ckpt100", "astral"): ("raw", REPO_ROOT / "data" / "novelty" / "phase12_ckpt100_astral_n32.jsonl"),
    ("phase12_ckpt200", "heldout"): ("raw", REPO_ROOT / "data" / "novelty" / "phase12_ckpt200_heldout_n32.jsonl"),
    ("phase12_ckpt200", "astral"): ("raw", REPO_ROOT / "data" / "novelty" / "phase12_ckpt200_astral_n32.jsonl"),
    ("phase12_ckpt300", "heldout"): ("raw", REPO_ROOT / "data" / "novelty" / "phase12_ckpt300_heldout_n32.jsonl"),
    ("phase12_ckpt300", "astral"): ("enriched", REPO_ROOT / "results" / "astral_gen_n32_gdpo_phase12_beta0.json"),
}

CHECKPOINT_ORDER = ["rs_sft", "e1_ckpt100", "e1_ckpt200", "e1_ckpt300",
                   "phase12_ckpt100", "phase12_ckpt200", "phase12_ckpt300"]

OUT_PATH = REPO_ROOT / "results" / "support_growth_by_checkpoint.json"


def precset_of(route) -> tuple:
    return tuple(sorted({SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}))


def load_precsets_by_target(kind: str, path: Path, sample_group_filter: str | None = None) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = defaultdict(list)
    if not path.exists():
        return out
    if kind == "enriched":
        doc = json.loads(path.read_text())
        for r in doc["results"]:
            for s in r["samples"]:
                precs = s.get("precursors")
                if precs:
                    out[r["target"]].append(tuple(sorted(precs)))
        return out
    # raw jsonl (possibly with a sample_group filter, for rs_sft's own files)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if sample_group_filter is not None and rec.get("sample_group") != sample_group_filter:
                continue
            try:
                route = parse_completion(rec["completion"], rec["target"])
            except Exception:
                continue
            out[rec["target"]].append(precset_of(route))
    return out


def pooled_support(counts: dict[str, dict], targets: list[str]) -> float | None:
    num = sum(counts[t]["n_absent"] for t in targets if t in counts)
    den = sum(counts[t]["n_distinct"] for t in targets if t in counts)
    return (num / den) if den else None


def per_target_counts(precsets_by_target: dict[str, list[tuple]], ref_by_target: dict[str, set]) -> dict[str, dict]:
    out = {}
    for target, precsets in precsets_by_target.items():
        distinct = set(precsets)
        ref = ref_by_target.get(target, set())
        out[target] = {"n_absent": len(distinct - ref), "n_distinct": len(distinct)}
    return out


def bootstrap_ci(counts: dict[str, dict], all_targets: list[str], n_boot: int, seed: int) -> dict:
    rng = random.Random(seed)
    n = len(all_targets)
    vals = []
    for _ in range(n_boot):
        sample = [all_targets[rng.randrange(n)] for _ in range(n)]
        v = pooled_support(counts, sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    point = pooled_support(counts, all_targets)
    if not vals:
        return {"point": point, "ci95": [None, None], "n_boot_valid": 0}
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return {"point": point, "ci95": [lo, hi], "n_boot_valid": len(vals)}


def support_growth_for(checkpoint: str, target_set: str, ref_by_target: dict[str, set]) -> dict | None:
    kind, path = SOURCES[(checkpoint, target_set)]
    if kind == "raw_reference":
        precsets = load_precsets_by_target("raw", path, sample_group_filter="resampling_rate")
    else:
        precsets = load_precsets_by_target(kind, path)
    if not precsets:
        return None
    counts = per_target_counts(precsets, ref_by_target)
    all_targets = sorted(counts.keys())
    boot = bootstrap_ci(counts, all_targets, N_BOOT, seed=hash((checkpoint, target_set)) % (2**31))
    return {"n_targets": len(all_targets), **boot}


def build_reference(target_set: str) -> dict[str, set]:
    by_target = load_precsets_by_target("raw", RSSFT_PATHS[target_set], sample_group_filter="reference")
    return {t: set(v) for t, v in by_target.items()}


def hits_and_carbonate_for(checkpoint: str, validator) -> dict | None:
    kind, path = SOURCES.get((checkpoint, "astral"), (None, None))
    if path is None or not path.exists():
        return None
    overlap = json.loads(OVERLAP_PATH.read_text())
    overlapping = set(overlap["union_of_all_overlapping_targets"])

    if kind == "enriched":
        enriched_path = path
    else:
        # build the standard astral_gen_n32 shape from raw completions,
        # same pattern as research/phase16_e3_e3c_final_battery.py
        astral_data = json.loads(ASTRAL_VALIDATION_PATH.read_text())
        recs_by_target: dict[str, list[dict]] = defaultdict(list)
        with path.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    # rs_sft's own file is RSSFT_PATHS, which holds BOTH the
                    # 200-sample reference AND the 32-sample resampling_rate
                    # draw tagged by sample_group. Every other checkpoint's
                    # file here is exactly 32/target with no such field.
                    # Restrict to resampling_rate so rs_sft's hit/carbonate
                    # numbers are computed from n=32/target too -- apples to
                    # apples across the whole trajectory table, not n=232
                    # for rs_sft alone (the bug this comment replaces).
                    if kind == "raw_reference" and rec.get("sample_group") != "resampling_rate":
                        continue
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
                    precs = {SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}
                    if precs == trad:
                        match = "TRADITIONAL"
                    elif precs == pred:
                        match = "PREDICTED"
                    elif pred <= precs:
                        match = "PREDICTED_SUPERSET"
                    elif trad <= precs:
                        match = "TRADITIONAL_SUPERSET"
                    else:
                        match = "OTHER"
                except Exception:
                    reward, precs, match = None, set(), "PARSE_FAILURE"
                samples.append({"precursors": sorted(precs), "reward": reward, "match": match})
            results.append({"target": target, "samples": samples})
        # NOTE: "astral_gen_n32_rs_sft.json" already exists as the committed
        # Phase 11 / finding-18 headline evidence file (different schema --
        # has "checkpoint"/"n_samples"/"summary" keys this script doesn't
        # produce). Do not reuse that filename for this script's rebuilt,
        # n=32-only enrichment -- write to a distinctly-suffixed path instead
        # so a second run can never again clobber tracked evidence.
        out_name = f"astral_gen_n32_{checkpoint}_trajectory.json"
        enriched_path = REPO_ROOT / "results" / out_name
        enriched_path.write_text(json.dumps({"tag": checkpoint, "results": results}, indent=1))

    hits = {r["target"]: any(s.get("match") in ("PREDICTED", "PREDICTED_SUPERSET") for s in r["samples"])
           for r in json.loads(enriched_path.read_text())["results"]}
    all_targets = set(hits.keys())
    clean_targets = all_targets - overlapping
    cs = load_carbonate_shares(enriched_path, all_targets)
    return {
        "predicted_hits_clean": f"{sum(hits[t] for t in clean_targets)}/{len(clean_targets)}",
        "predicted_hits_full": f"{sum(hits.values())}/{len(all_targets)}",
        "carbonate_share_full_35": cs["shares"]["carbonate"],
        "bare_oxide_share_full_35": cs["shares"]["bare_oxide"],
        "enriched_file": str(enriched_path),
    }


def main():
    print("READING RULE (stated before looking): flat or falling support growth "
          "across checkpoints = no trend to extrapolate. Rising AND clearing "
          "RS-SFT's own resampling floor (10.5% ASTRAL / 11.7% held-out) at a "
          "later checkpoint = a real trend, report it.\n", flush=True)

    ref = {ts: build_reference(ts) for ts in ("heldout", "astral")}
    validator = load_validator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                               REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)

    report: dict = {"reading_rule": "flat/falling = no trend; rising AND clearing the floor = real trend",
                    "rssft_floor_reference": {"astral_pct": 10.5, "heldout_pct": 11.7},
                    "by_checkpoint": {}}

    print(f"{'checkpoint':<20}{'heldout growth':>22}{'astral growth':>22}{'astral hits (clean/full)':>28}{'carbonate%':>12}")
    for ckpt in CHECKPOINT_ORDER:
        row = {}
        for ts in ("heldout", "astral"):
            if (ckpt, ts) not in SOURCES:
                row[ts] = None
                continue
            kind, path = SOURCES[(ckpt, ts)]
            if not path.exists():
                print(f"  [skip] {ckpt}/{ts}: {path} not found yet")
                row[ts] = None
                continue
            row[ts] = support_growth_for(ckpt, ts, ref[ts])

        hc = hits_and_carbonate_for(ckpt, validator)
        row["hits_and_carbonate"] = hc

        h = row.get("heldout")
        a = row.get("astral")
        h_str = f"{h['point']*100:.1f}% [{h['ci95'][0]*100:.1f},{h['ci95'][1]*100:.1f}]" if h else "n/a"
        a_str = f"{a['point']*100:.1f}% [{a['ci95'][0]*100:.1f},{a['ci95'][1]*100:.1f}]" if a else "n/a"
        hits_str = f"{hc['predicted_hits_clean']} / {hc['predicted_hits_full']}" if hc else "n/a"
        carb_str = f"{hc['carbonate_share_full_35']*100:.1f}%" if hc and hc['carbonate_share_full_35'] is not None else "n/a"
        print(f"{ckpt:<20}{h_str:>22}{a_str:>22}{hits_str:>28}{carb_str:>12}")

        report["by_checkpoint"][ckpt] = row

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
