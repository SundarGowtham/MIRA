#!/usr/bin/env python
"""
scripts/astral_overlap_report.py — Phase 16, "Decisions after pass 2"
item 1 (misc/PHASE16_INSTRUCTIONS.md). Informational: one row per
(dataset, ASTRAL target) giving the record count, the record's role
("prompt_only" or "prompt_plus_answer"), and, where there is a stored
answer, whether its precursor set equals ASTRAL's conventional
("traditional") set, ASTRAL's rule-picked ("predicted") set, or neither.

Also asserts zero overlap for every dataset PHASE 16 ITSELF CREATES
(currently just data/novelty/ -- E3/E3c's own training sets land there
later and get checked the same way once they exist).

This script does NOT decide pass/fail for the existing, pre-Phase-16
datasets (data/sft_v3, data/rl_run3, data/rl) -- that overlap is
disclosed, not gated. tests/test_no_astral_answer_leakage.py is the
strict gate: it fails only on the thing rule 1 actually forbids
(training on an ASTRAL answer).

Usage:
  uv run python scripts/astral_overlap_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validator import SynthesisValidator  # noqa: E402

ASTRAL_PATH = REPO_ROOT / "results" / "astral_validation_set.json"
OUT_PATH = REPO_ROOT / "results" / "phase16" / "astral_overlap.json"

# (dataset label, path) for every dataset currently in active use.
# data/processed/reasoning_traces.jsonl is excluded -- confirmed
# unreferenced by any .py file in the repo (tests/test_no_astral_
# answer_leakage.py re-verifies this each run, not assumed here).
PRE_EXISTING_DATASETS = {
    "sft_v3_train": REPO_ROOT / "data/sft_v3/sft_train.jsonl",
    "sft_v3_val": REPO_ROOT / "data/sft_v3/sft_val.jsonl",
    "sft_v3_test": REPO_ROOT / "data/sft_v3/sft_test.jsonl",
    "sft_v3_format_only_train": REPO_ROOT / "data/sft_v3/format_only_train.jsonl",
    "sft_v3_format_only_val": REPO_ROOT / "data/sft_v3/format_only_val.jsonl",
    "rl_run3_train": REPO_ROOT / "data/rl_run3/rl3_train.jsonl",
    "rl_run3_val": REPO_ROOT / "data/rl_run3/rl3_val.jsonl",
    "rl_run3_probe": REPO_ROOT / "data/rl_run3/rl3_probe.jsonl",
    "rs_sft_train": REPO_ROOT / "data/rs_sft/rs_sft_train.jsonl",
    "rs_sft_val": REPO_ROOT / "data/rs_sft/rs_sft_val.jsonl",
    "rl_train": REPO_ROOT / "data/rl/rl_train.jsonl",
    "rl_val": REPO_ROOT / "data/rl/rl_val.jsonl",
}

# Datasets Phase 16 itself creates -- must have exactly zero overlap,
# asserted (not just reported) below. Empty for now; E3/E3c's training
# jsonl files get added here once Section 6 is implemented.
PHASE16_CREATED_DATASETS: dict[str, Path] = {}


def norm(formula: str) -> str:
    return SynthesisValidator._normalize_formula(formula)


def extract_answer_precursors(rec: dict) -> list[str] | None:
    """Returns the normalized precursor-formula list from a record's
    stored answer, or None if there is no answer field at all (the
    rl_run3/rl/rs_sft schema -- prompt-only, the model generates
    on-policy, there is no fixed answer in the file).

    `response` (when present) is already bare JSON. `completion` is the
    full <think>...</think>{json} training text and needs the project's
    own parser (core.reward.parse_completion), which strips the think
    block and brace-balances -- a naive json.loads on `completion`
    silently fails (caught, returning None) and was an actual bug caught
    while writing this script: checking `rec.get("completion") or
    rec.get("response")` grabbed the unparsed think-block text first
    because it's non-empty, never reaching the already-clean `response`."""
    if rec.get("response"):
        try:
            obj = json.loads(rec["response"])
        except (json.JSONDecodeError, TypeError):
            obj = None
        if obj is not None:
            precs = obj.get("precursors")
            if isinstance(precs, list):
                return sorted({norm(p.get("formula", "")) for p in precs
                              if isinstance(p, dict) and p.get("formula")})
    if rec.get("completion"):
        from core.reward import parse_completion, ParseFailure
        try:
            route = parse_completion(rec["completion"], rec.get("target") or rec.get("target_formula") or "")
        except ParseFailure:
            return None
        return sorted({norm(p.formula) for p in route.precursors})
    return None


def classify(answer_precs: list[str], astral_entry: dict) -> str:
    traditional = sorted({norm(p) for p in astral_entry["traditional"]})
    predicted = sorted({norm(p) for p in astral_entry["predicted"]})
    if answer_precs == traditional:
        return "matches_traditional"
    if answer_precs == predicted:
        return "matches_predicted"
    return "neither"


def scan_dataset(path: Path, astral_by_target: dict) -> list[dict]:
    if not path.exists():
        return []
    rows: dict[tuple[str, str], dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = rec.get("target") or rec.get("target_formula")
        if target not in astral_by_target:
            continue
        answer_precs = extract_answer_precursors(rec)
        role = "prompt_plus_answer" if answer_precs is not None else "prompt_only"
        match = classify(answer_precs, astral_by_target[target]) if answer_precs is not None else None
        key = (target, role, match)
        if key not in rows:
            rows[key] = {"target": target, "role": role, "match": match, "count": 0}
        rows[key]["count"] += 1
    return list(rows.values())


def main():
    astral_entries = json.loads(ASTRAL_PATH.read_text())["targets"]
    astral_by_target = {t["target"]: t for t in astral_entries}

    report = {"pre_existing": {}, "phase16_created": {}}
    all_overlapping_targets: set[str] = set()

    for name, path in PRE_EXISTING_DATASETS.items():
        rows = scan_dataset(path, astral_by_target)
        if rows:
            report["pre_existing"][name] = rows
            for r in rows:
                all_overlapping_targets.add(r["target"])
            print(f"{name}: {sum(r['count'] for r in rows)} overlapping record(s) "
                  f"across {len({r['target'] for r in rows})} target(s)")
            for r in rows:
                print(f"    target={r['target']} role={r['role']} match={r['match']} count={r['count']}")
        else:
            print(f"{name}: 0 overlap")

    print("\n=== datasets Phase 16 itself creates: must be zero ===")
    violations = []
    for name, path in PHASE16_CREATED_DATASETS.items():
        rows = scan_dataset(path, astral_by_target)
        report["phase16_created"][name] = rows
        if rows:
            violations.append((name, rows))
        print(f"{name}: {'0 overlap' if not rows else f'{len(rows)} VIOLATIONS'}")
    if not PHASE16_CREATED_DATASETS:
        print("(none created yet -- E3/E3c land here once Section 6 runs)")

    report["union_of_all_overlapping_targets"] = sorted(all_overlapping_targets)
    report["n_overlapping_targets"] = len(all_overlapping_targets)
    report["n_astral_targets_total"] = len(astral_entries)
    report["clean_subset_size"] = len(astral_entries) - len(all_overlapping_targets)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nUnion of overlapping targets ({len(all_overlapping_targets)}/35): "
          f"{sorted(all_overlapping_targets)}")
    print(f"Clean subset size: {report['clean_subset_size']}/35")
    print(f"-> {OUT_PATH}")

    if violations:
        print("\nFATAL: Phase-16-created datasets have ASTRAL overlap:", violations)
        sys.exit(1)


if __name__ == "__main__":
    main()
