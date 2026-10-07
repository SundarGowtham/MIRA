#!/usr/bin/env python
"""
data_curation/build_e3_e3c_filter.py — Phase 16 §6.3/§6.4
(misc/PHASE16_INSTRUCTIONS.md), per regular Claude's instruction to
report before training anything.

Reads data/novelty/round1_samples.jsonl (23,040 raw base-model samples,
360 targets x 64), applies:

  E3 filter: validity gate G=1 AND (>=1 precursor with corpus count <=17,
    "rarer than 1 in 1,000", design choice 1) OR set_novelty==1).
    Deduplicate by precursor set, at most 4 per target (design choice 3).
  E3c filter: same gate, NO novelty condition, same dedup+cap, then
    randomly subsampled (seed 42) to exactly E3's final size.

STOP RULE (pre-registered, not adjustable after seeing the data): if the
final E3 size is < 150, or more than half of the 360 targets keep zero
samples, STOP -- write the diagnostic report, do NOT write the final
training-ready E3/E3c jsonl files, and do not loosen any threshold. A
small set is itself a result (how rarely base produces new recipes at
this rarity bar), not a reason to retune.

Usage (tmux, CPU -- thermo-backed validator calls, no GPU needed):
  uv run python data_curation/build_e3_e3c_filter.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validator import SynthesisValidator  # noqa: E402
from core.reward import parse_completion, ParseFailure  # noqa: E402
from core.novelty_reward import load_novelty_reward  # noqa: E402

POOL_PATH = REPO_ROOT / "data" / "novelty" / "round1_samples.jsonl"
OUT_E3 = REPO_ROOT / "data" / "novelty" / "e3_round1_filtered.jsonl"
OUT_E3C = REPO_ROOT / "data" / "novelty" / "e3c_round1_filtered.jsonl"
REPORT_PATH = REPO_ROOT / "results" / "phase16" / "e3_e3c_filter_report.json"
RESULTS_MD = REPO_ROOT / "docs" / "phases" / "PHASE16_RESULTS.md"

RARITY_COUNT_THRESHOLD = 17  # design choice 1: <= 17 of 17,616 (~1/1000)
PER_TARGET_CAP = 4           # design choice 3
MIN_E3_SIZE = 150            # stop rule, pre-registered
MOST_EMPTY_FRACTION = 0.5    # "most targets keep nothing"
SEED = 42


def precset_of(route) -> tuple:
    return tuple(sorted({SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}))


def dedup_and_cap(candidates: list[tuple[tuple, dict]], cap: int) -> list[dict]:
    seen = set()
    kept = []
    for precset, rec in candidates:
        if precset in seen:
            continue
        seen.add(precset)
        kept.append(rec)
        if len(kept) >= cap:
            break
    return kept


def main():
    nr = load_novelty_reward(
        formula_set_path=REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
        pd_index_path=REPO_ROOT / "data" / "cache" / "pd_index.json",
        project_root=REPO_ROOT,
    )

    samples_by_target: dict[str, list[dict]] = defaultdict(list)
    with POOL_PATH.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                samples_by_target[rec["target"]].append(rec)
    n_targets_total = len(samples_by_target)
    n_total = sum(len(v) for v in samples_by_target.values())
    print(f"loaded {n_total} samples across {n_targets_total} targets", flush=True)

    n_gate_pass = 0
    n_parse_fail = 0
    n_novel_among_gate_pass = 0
    e3_cands: dict[str, list] = defaultdict(list)
    e3c_cands: dict[str, list] = defaultdict(list)

    t0 = time.time()
    i = 0
    for target, recs in samples_by_target.items():
        for rec in recs:
            i += 1
            try:
                route = parse_completion(rec["completion"], target)
            except ParseFailure:
                n_parse_fail += 1
                continue
            except Exception:
                n_parse_fail += 1
                continue
            try:
                _, info = nr.score(route, target)
            except Exception:
                continue
            if not info["gate"]:
                continue
            n_gate_pass += 1
            has_rare = any(
                nr.precursor_freq.get(SynthesisValidator._normalize_formula(p.formula), 0) <= RARITY_COUNT_THRESHOLD
                for p in route.precursors
            )
            is_novel = has_rare or (info.get("set_novelty") == 1.0)
            if is_novel:
                n_novel_among_gate_pass += 1
            precset = precset_of(route)
            e3c_cands[target].append((precset, rec))
            if is_novel:
                e3_cands[target].append((precset, rec))
        if i % 2000 < 64:
            print(f"  {i}/{n_total} scored, {(time.time() - t0):.0f}s elapsed", flush=True)

    e3_final: list[dict] = []
    n_targets_keep_atleast1 = 0
    n_targets_keep_none = 0
    for target in samples_by_target:
        kept = dedup_and_cap(e3_cands.get(target, []), PER_TARGET_CAP)
        if kept:
            n_targets_keep_atleast1 += 1
        else:
            n_targets_keep_none += 1
        e3_final.extend(kept)

    e3c_pool: list[dict] = []
    for target in samples_by_target:
        e3c_pool.extend(dedup_and_cap(e3c_cands.get(target, []), PER_TARGET_CAP))

    rng = random.Random(SEED)
    if len(e3c_pool) >= len(e3_final):
        e3c_final = rng.sample(e3c_pool, len(e3_final))
    else:
        e3c_final = e3c_pool  # fewer gate-passing candidates than E3's size -- can't match up, report as-is

    report = {
        "n_total_pool_samples": n_total,
        "n_parse_fail": n_parse_fail,
        "n_gate_pass": n_gate_pass,
        "gate_pass_rate": round(n_gate_pass / n_total, 4),
        "n_novel_among_gate_pass": n_novel_among_gate_pass,
        "novelty_rate_among_gate_pass": round(n_novel_among_gate_pass / n_gate_pass, 4) if n_gate_pass else None,
        "n_targets_total": n_targets_total,
        "n_targets_keep_at_least_one": n_targets_keep_atleast1,
        "n_targets_keep_none": n_targets_keep_none,
        "fraction_targets_keep_none": round(n_targets_keep_none / n_targets_total, 4),
        "e3_final_size": len(e3_final),
        "e3c_pool_size_before_subsample": len(e3c_pool),
        "e3c_final_size": len(e3c_final),
        "rarity_count_threshold": RARITY_COUNT_THRESHOLD,
        "per_target_cap": PER_TARGET_CAP,
        "seed": SEED,
    }
    print(json.dumps(report, indent=1), flush=True)

    stop = (len(e3_final) < MIN_E3_SIZE) or (n_targets_keep_none / n_targets_total > MOST_EMPTY_FRACTION)
    report["stop_rule_triggered"] = stop
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1))
    print(f"-> {REPORT_PATH}", flush=True)

    if stop:
        print("STOP RULE TRIGGERED. Not writing final E3/E3c training files. "
             "Per pre-registration, the threshold is not adjusted after seeing this.", flush=True)
        with RESULTS_MD.open("a") as f:
            f.write(f"\n## E3/E3c filter: STOP RULE TRIGGERED\n\n"
                    f"```\n{json.dumps(report, indent=1)}\n```\n\n"
                    f"Final E3 size {len(e3_final)} (bar: >=150) or "
                    f"{n_targets_keep_none}/{n_targets_total} targets kept zero samples "
                    f"(bar: <=50%). Per the pre-registration, this is reported as a "
                    f"finding, not adjusted. Stopping before writing training-ready "
                    f"E3/E3c files; no training launched.\n")
        return

    with OUT_E3.open("w") as f:
        for r in e3_final:
            f.write(json.dumps(r) + "\n")
    with OUT_E3C.open("w") as f:
        for r in e3c_final:
            f.write(json.dumps(r) + "\n")
    print(f"-> {OUT_E3} ({len(e3_final)} examples)")
    print(f"-> {OUT_E3C} ({len(e3c_final)} examples)")


if __name__ == "__main__":
    main()
