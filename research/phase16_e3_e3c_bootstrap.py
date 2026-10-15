#!/usr/bin/env python
"""
research/phase16_e3_e3c_bootstrap.py — Phase 16 §prediction 2's own
stated test (docs/phases/PHASE16_PREREG.md): "on the held-out split,
E3's novelty rate exceeds E3c's, with a bootstrap 95% CI on the
difference excluding zero" and "E3's support growth relative to base
exceeds E3c's support growth relative to base."

Cluster-bootstraps by TARGET (not by sample -- samples within a target
are correlated, same convention as every other bootstrap CI in this
project, e.g. research/distributional/arrows_gate_scoring.py): resample
the 40 held-out targets with replacement, recompute each arm's pooled
novelty rate and support growth over the resampled targets, take E3 -
E3c, repeat N_BOOT times, report the percentile 95% CI.

Reuses research/phase16_e3_e3c_evaluation.py's scoring logic per-target
rather than only pooled.

Usage (tmux, CPU -- thermo-backed validator calls, no GPU needed):
  uv run python research/phase16_e3_e3c_bootstrap.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validator import SynthesisValidator  # noqa: E402
from core.reward import parse_completion, ParseFailure  # noqa: E402
from core.novelty_reward import load_novelty_reward  # noqa: E402

RARITY_COUNT_THRESHOLD = 17
SEED = 42
N_BOOT = 10000

GEN_PATHS = {
    "e3": REPO_ROOT / "data" / "novelty" / "e3_heldout_n32.jsonl",
    "e3c": REPO_ROOT / "data" / "novelty" / "e3c_heldout_n32.jsonl",
}
BASE_REF_PATH = REPO_ROOT / "data" / "novelty" / "base_reference_heldout.jsonl"
OUT_PATH = REPO_ROOT / "results" / "phase16" / "e3_e3c_bootstrap.json"


def precset_of(route) -> tuple:
    return tuple(sorted({SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}))


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


def per_target_counts(nr, recs: list[dict], ref_precsets_by_target: dict[str, set]) -> dict[str, dict]:
    """Returns target -> {n_gate_pass, n_novel, n_absent, n_distinct}."""
    out: dict[str, dict] = defaultdict(lambda: {"n_gate_pass": 0, "n_novel": 0, "precsets": []})
    for rec in recs:
        target = rec["target"]
        try:
            route = parse_completion(rec["completion"], target)
        except Exception:
            continue
        out[target]["precsets"].append(precset_of(route))
        try:
            _, info = nr.score(route, target)
        except Exception:
            continue
        if not info["gate"]:
            continue
        out[target]["n_gate_pass"] += 1
        has_rare = any(
            nr.precursor_freq.get(SynthesisValidator._normalize_formula(p.formula), 0) <= RARITY_COUNT_THRESHOLD
            for p in route.precursors
        )
        if has_rare or info.get("set_novelty") == 1.0:
            out[target]["n_novel"] += 1

    result = {}
    for target, d in out.items():
        distinct = set(d["precsets"])
        ref = ref_precsets_by_target.get(target, set())
        absent = distinct - ref
        result[target] = {
            "n_gate_pass": d["n_gate_pass"],
            "n_novel": d["n_novel"],
            "n_absent": len(absent),
            "n_distinct": len(distinct),
        }
    return result


def pooled_novelty_rate(counts: dict[str, dict], targets: list[str]) -> float | None:
    num = sum(counts[t]["n_novel"] for t in targets if t in counts)
    den = sum(counts[t]["n_gate_pass"] for t in targets if t in counts)
    return (num / den) if den else None


def pooled_support_growth(counts: dict[str, dict], targets: list[str]) -> float | None:
    num = sum(counts[t]["n_absent"] for t in targets if t in counts)
    den = sum(counts[t]["n_distinct"] for t in targets if t in counts)
    return (num / den) if den else None


def bootstrap_diff(e3_counts: dict, e3c_counts: dict, all_targets: list[str], metric_fn, n_boot: int, rng: random.Random) -> dict:
    diffs = []
    n = len(all_targets)
    for _ in range(n_boot):
        sample = [all_targets[rng.randrange(n)] for _ in range(n)]
        e3_val = metric_fn(e3_counts, sample)
        e3c_val = metric_fn(e3c_counts, sample)
        if e3_val is None or e3c_val is None:
            continue
        diffs.append(e3_val - e3c_val)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    point = metric_fn(e3_counts, all_targets) - metric_fn(e3c_counts, all_targets)
    return {"point_diff_e3_minus_e3c": point, "ci95": [lo, hi], "n_boot_valid": len(diffs)}


def main():
    nr = load_novelty_reward(
        formula_set_path=REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
        pd_index_path=REPO_ROOT / "data" / "cache" / "pd_index.json",
        project_root=REPO_ROOT,
    )

    base_ref_recs = load_jsonl(BASE_REF_PATH, target_filter=lambda r: r.get("sample_group") == "reference")
    ref_precsets_by_target: dict[str, set] = defaultdict(set)
    for rec in base_ref_recs:
        try:
            route = parse_completion(rec["completion"], rec["target"])
        except Exception:
            continue
        ref_precsets_by_target[rec["target"]].add(precset_of(route))

    e3_recs = load_jsonl(GEN_PATHS["e3"])
    e3c_recs = load_jsonl(GEN_PATHS["e3c"])
    e3_counts = per_target_counts(nr, e3_recs, dict(ref_precsets_by_target))
    e3c_counts = per_target_counts(nr, e3c_recs, dict(ref_precsets_by_target))

    all_targets = sorted(set(e3_counts.keys()) | set(e3c_counts.keys()))
    print(f"n_targets={len(all_targets)}")

    rng = random.Random(SEED)
    novelty_result = bootstrap_diff(e3_counts, e3c_counts, all_targets, pooled_novelty_rate, N_BOOT, rng)
    print(f"novelty_rate diff (E3-E3c): point={novelty_result['point_diff_e3_minus_e3c']:.4f} "
          f"95% CI={novelty_result['ci95']}")

    rng2 = random.Random(SEED + 1)
    support_result = bootstrap_diff(e3_counts, e3c_counts, all_targets, pooled_support_growth, N_BOOT, rng2)
    print(f"support_growth diff (E3-E3c): point={support_result['point_diff_e3_minus_e3c']:.4f} "
          f"95% CI={support_result['ci95']}")

    gate_rate_e3 = sum(e3_counts[t]["n_gate_pass"] for t in all_targets if t in e3_counts) / sum(1 for t in all_targets if t in e3_counts) / 32
    gate_rate_e3c = sum(e3c_counts[t]["n_gate_pass"] for t in all_targets if t in e3c_counts) / sum(1 for t in all_targets if t in e3c_counts) / 32
    print(f"gate_pass_rate: E3={gate_rate_e3:.4f} E3c={gate_rate_e3c:.4f} diff_pp={100*(gate_rate_e3-gate_rate_e3c):.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "n_targets": len(all_targets),
        "novelty_rate_diff": novelty_result,
        "support_growth_diff": support_result,
        "gate_pass_rate": {"e3": gate_rate_e3, "e3c": gate_rate_e3c},
    }, indent=2))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
