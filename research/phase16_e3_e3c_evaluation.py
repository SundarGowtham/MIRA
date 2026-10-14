#!/usr/bin/env python
"""
research/phase16_e3_e3c_evaluation.py — Phase 16, the critical-path
item regular Claude named while the GPU was idle after E1: "compute
the novelty rate and support growth" for E3 and E3c's own n=32
generations (data/novelty/{e3,e3c}_{heldout,astral}_n32.jsonl,
data_curation/phase16_e3_e3c_sampling.py).

Both E3 and E3c are LoRA-on-BASE arms (misc/PHASE16_INSTRUCTIONS.md's
own arm table), so their matching reference distribution for support
growth is BASE's own 200+32 draws
(data/novelty/base_reference_{heldout,astral}.jsonl) -- NOT RS-SFT's
(that comparison belongs to E1 only).

Novelty rate: same definition as the round-1 filter
(data_curation/build_e3_e3c_filter.py) -- gate pass rate over all
samples, and among gate-passers, the share meeting the novelty
condition (>=1 precursor at corpus count <= 17, OR set_novelty == 1).
Applied here to E3's and E3c's OWN generations (not the round-1 base
pool), i.e. "did the self-distillation retain its novelty property
when resampled from the resulting checkpoint" -- a different question
from the round-1 filter's "how much of the base pool qualifies."

Support growth: identical definition and methodology to
research/phase16_e1_support_growth.py (pooled across targets: M's
distinct precursor sets absent from R's 200-sample reference / M's
total distinct sets), with M in {E3, E3c} and R = base.

Usage (tmux, CPU -- thermo-backed validator calls, no GPU needed):
  uv run python research/phase16_e3_e3c_evaluation.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validator import SynthesisValidator  # noqa: E402
from core.reward import parse_completion, ParseFailure  # noqa: E402
from core.novelty_reward import load_novelty_reward  # noqa: E402

RARITY_COUNT_THRESHOLD = 17  # matches build_e3_e3c_filter.py's design choice 1

GEN_PATHS = {
    ("e3", "heldout"): REPO_ROOT / "data" / "novelty" / "e3_heldout_n32.jsonl",
    ("e3", "astral"): REPO_ROOT / "data" / "novelty" / "e3_astral_n32.jsonl",
    ("e3c", "heldout"): REPO_ROOT / "data" / "novelty" / "e3c_heldout_n32.jsonl",
    ("e3c", "astral"): REPO_ROOT / "data" / "novelty" / "e3c_astral_n32.jsonl",
}
BASE_REF_PATHS = {
    "heldout": REPO_ROOT / "data" / "novelty" / "base_reference_heldout.jsonl",
    "astral": REPO_ROOT / "data" / "novelty" / "base_reference_astral.jsonl",
}
OUT_PATH = REPO_ROOT / "results" / "phase16" / "e3_e3c_evaluation.json"


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


def score_novelty_and_support(nr, recs: list[dict], ref_precsets_by_target: dict[str, set]) -> dict:
    n_total = len(recs)
    n_parse_fail = 0
    n_gate_pass = 0
    n_novel_among_gate_pass = 0
    precsets_by_target: dict[str, list[tuple]] = defaultdict(list)

    for rec in recs:
        target = rec["target"]
        try:
            route = parse_completion(rec["completion"], target)
        except ParseFailure:
            n_parse_fail += 1
            continue
        except Exception:
            n_parse_fail += 1
            continue
        precsets_by_target[target].append(precset_of(route))
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

    num, den = 0, 0
    for target, precsets in precsets_by_target.items():
        distinct = set(precsets)
        ref = ref_precsets_by_target.get(target, set())
        absent = distinct - ref
        num += len(absent)
        den += len(distinct)

    return {
        "n_total_samples": n_total,
        "n_parse_fail": n_parse_fail,
        "n_gate_pass": n_gate_pass,
        "gate_pass_rate": round(n_gate_pass / n_total, 4) if n_total else None,
        "n_novel_among_gate_pass": n_novel_among_gate_pass,
        "novelty_rate_among_gate_pass": round(n_novel_among_gate_pass / n_gate_pass, 4) if n_gate_pass else None,
        "support_growth_numerator_absent": num,
        "support_growth_denominator_distinct": den,
        "support_growth_pooled_share": (num / den) if den else None,
    }


def main():
    nr = load_novelty_reward(
        formula_set_path=REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
        pd_index_path=REPO_ROOT / "data" / "cache" / "pd_index.json",
        project_root=REPO_ROOT,
    )

    report: dict = {}
    for target_set in ("heldout", "astral"):
        base_ref_recs = load_jsonl(BASE_REF_PATHS[target_set],
                                   target_filter=lambda r: r.get("sample_group") == "reference")
        base_resamp_recs = load_jsonl(BASE_REF_PATHS[target_set],
                                      target_filter=lambda r: r.get("sample_group") == "resampling_rate")

        ref_precsets_by_target: dict[str, set] = defaultdict(set)
        for rec in base_ref_recs:
            try:
                route = parse_completion(rec["completion"], rec["target"])
            except Exception:
                continue
            ref_precsets_by_target[rec["target"]].add(precset_of(route))

        base_noise = score_novelty_and_support(nr, base_resamp_recs, dict(ref_precsets_by_target))
        print(f"=== {target_set}: base's own resampling noise floor ===")
        print(f"  support_growth (noise floor) = {base_noise['support_growth_pooled_share']:.4f} "
              f"({base_noise['support_growth_numerator_absent']}/{base_noise['support_growth_denominator_distinct']})")
        report.setdefault(target_set, {})["base_resampling_noise_floor"] = base_noise

        for checkpoint in ("e3", "e3c"):
            recs = load_jsonl(GEN_PATHS[(checkpoint, target_set)])
            result = score_novelty_and_support(nr, recs, dict(ref_precsets_by_target))
            print(f"=== {target_set}: {checkpoint} ===")
            print(f"  gate_pass_rate={result['gate_pass_rate']} "
                  f"novelty_rate_among_gate_pass={result['novelty_rate_among_gate_pass']} "
                  f"support_growth={result['support_growth_pooled_share']:.4f} "
                  f"({result['support_growth_numerator_absent']}/{result['support_growth_denominator_distinct']})")
            report[target_set][checkpoint] = result

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
