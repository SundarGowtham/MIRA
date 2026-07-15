"""
split_dataset.py
----------------
Filters, deduplicates, and stratified-splits the rescored synthesis dataset
into train/val/test JSONL files ready for SFT and GRPO.

Stratification axes:
  1. Validator score band (A/B/C/D) — ensures val/test see the same
     difficulty distribution as train
  2. Target chemistry class (oxide / phosphate / sulfide / halide / other)
     — ensures val/test cover the same chemistry as train
  3. Thermodynamic path (has_dG / solid_solution / no_pd) — ensures the
     three thermodynamic regimes are equally represented

Deduplication: when a target appears multiple times (model was run more than
once on the same target), keep the record with the highest validator_score.
Ties broken by reasoning length (longer = more informative for SFT).

Output files:
  train.jsonl  — 80% of kept records
  val.jsonl    — 10%
  test.jsonl   — 10%
  split_stats.json — full breakdown for the writeup

Usage:
    uv run python split_dataset.py
    uv run python split_dataset.py --input data/processed/synthesis_with_traces_rescored.jsonl
    uv run python split_dataset.py --min-score 0.65 --val-frac 0.1 --test-frac 0.1
    uv run python split_dataset.py --keep-dupes  # skip dedup, keep all records
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "synthesis_with_traces_rescored.jsonl"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "sft"


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def score_band(score: float) -> str:
    if score >= 0.95: return "A"
    if score >= 0.85: return "B"
    if score >= 0.65: return "C"
    return "D"


def chemistry_class(formula: str) -> str:
    """
    Coarse chemistry class from the target formula.
    Used as one axis of stratification so val/test see all chemistry types.
    """
    f = formula.upper()
    elements = set(re.findall(r'[A-Z][A-Z]?', f))  # rough element extraction
    if "P" in elements and "O" in elements:
        return "phosphate"
    if "S" in elements and "O" not in elements:
        return "sulfide"
    if "S" in elements and "O" in elements and any(e in elements for e in ["Ba","Sr","Ca","La"]):
        return "sulfate"
    if any(e in elements for e in ["F","CL","BR","I"]) and "O" not in elements:
        return "halide"
    if "N" in elements and "O" not in elements:
        return "nitride"
    if "C" in elements and "O" not in elements:
        return "carbide"
    if "SI" in elements and "O" in elements:
        return "silicate"
    if "B" in elements and "O" in elements:
        return "borate"
    if "O" in elements:
        return "oxide"
    return "other"


def thermo_class(record: dict) -> str:
    """Which thermodynamic regime this record falls into."""
    bd = record.get("validator_breakdown", {})
    thermo = bd.get("thermodynamic_favorable")
    dg     = bd.get("thermodynamic_dG_eV_atom")
    t_k    = bd.get("thermodynamic_T_K")

    if dg is not None:
        return "has_dG"
    if thermo == 0.5:
        formula = record.get("target", "")
        if re.search(r'\d+\.\d+', formula):
            return "solid_solution"
        if t_k is None:
            return "no_pd"
        return "gibbs_failed"
    return "other"


def stratification_key(record: dict) -> str:
    """Single string key combining all stratification axes."""
    score  = record.get("validator_score", 0)
    band   = score_band(score)
    chem   = chemistry_class(record.get("target", ""))
    thermo = thermo_class(record)
    return f"{band}_{chem}_{thermo}"


# ---------------------------------------------------------------------------
# SFT formatting
# ---------------------------------------------------------------------------

def format_for_sft(record: dict) -> dict:
    """
    Convert a raw record to SFT-ready format.

    The training signal is (prompt → thinking + response), where:
      - prompt: the synthesis task specification (already in record["prompt"])
      - thinking: the <think>...</think> chain of thought
      - response: the structured JSON predicted_route

    Returns a dict with 'prompt', 'thinking', 'response' fields plus
    all validator metadata preserved for downstream filtering.
    """
    thinking = record.get("thinking", "").strip()
    # Strip the wrapping <think>...</think> tags if present — the model
    # should generate them, not receive them in the target
    thinking = re.sub(r'^<think>\s*', '', thinking)
    thinking = re.sub(r'\s*</think>$', '', thinking)

    response = json.dumps(
        record.get("predicted_route", {}),
        indent=2,
        ensure_ascii=False,
    )

    # core/data.py's build_sft_dataset expects a single "completion" string
    # that becomes the assistant turn's content. Reconstruct the original
    # <think>...</think> + JSON format the model was trained to produce —
    # this is what the model needs to learn to generate end-to-end.
    completion = f"<think>\n{thinking}\n</think>\n{response}"

    return {
        "prompt":           record.get("prompt", ""),
        "completion":       completion,
        "thinking":         thinking,
        "response":         response,
        "target":           record.get("target", ""),
        "validator_score":  record.get("validator_score", 0),
        "validator_breakdown": record.get("validator_breakdown", {}),
        "score_band":       score_band(record.get("validator_score", 0)),
        "chemistry_class":  chemistry_class(record.get("target", "")),
        "thermo_class":     thermo_class(record),
        "generator":        record.get("generator", ""),
        "provider":         record.get("provider", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-score",  type=float, default=0.65,
                        help="Minimum validator_score to keep (default 0.65)")
    parser.add_argument("--val-frac",   type=float, default=0.10)
    parser.add_argument("--test-frac",  type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--keep-dupes", action="store_true",
                        help="Skip deduplication (keep all records per target)")
    parser.add_argument("--no-sft-format", action="store_true",
                        help="Write raw records instead of SFT-formatted output")
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Load ---
    print(f"Loading {args.input}")
    records = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded: {len(records)} records")

    # --- Filter by min score ---
    before = len(records)
    records = [r for r in records if r.get("validator_score", 0) >= args.min_score]
    print(f"  After score filter (≥{args.min_score}): {len(records)}  "
          f"(dropped {before - len(records)})")

    # --- Deduplicate by target: keep highest-scoring record ---
    if not args.keep_dupes:
        by_target: dict[str, list] = defaultdict(list)
        for r in records:
            by_target[r.get("target", "__unknown__")].append(r)

        deduped = []
        n_dupes = 0
        for target, recs in by_target.items():
            if len(recs) == 1:
                deduped.append(recs[0])
            else:
                n_dupes += len(recs) - 1
                # Sort by (validator_score DESC, reasoning length DESC)
                best = max(
                    recs,
                    key=lambda r: (
                        r.get("validator_score", 0),
                        len(r.get("thinking", "")),
                    )
                )
                deduped.append(best)
        records = deduped
        print(f"  After dedup: {len(records)}  (removed {n_dupes} lower-scoring duplicates)")
    else:
        print(f"  Dedup skipped (--keep-dupes)")

    # --- Stratified split ---
    # Group by stratification key, shuffle within each group, then
    # pull val_frac + test_frac from each group proportionally.
    by_stratum: dict[str, list] = defaultdict(list)
    for r in records:
        by_stratum[stratification_key(r)].append(r)

    train_recs, val_recs, test_recs = [], [], []

    for stratum, recs in sorted(by_stratum.items()):
        random.shuffle(recs)
        n = len(recs)
        n_test = max(1, round(n * args.test_frac)) if n >= 3 else 0
        n_val  = max(1, round(n * args.val_frac))  if n >= 3 else 0
        # Cap so we don't over-pull small strata
        n_test = min(n_test, n // 3)
        n_val  = min(n_val,  (n - n_test) // 2)

        test_recs  += recs[:n_test]
        val_recs   += recs[n_test:n_test + n_val]
        train_recs += recs[n_test + n_val:]

    total = len(train_recs) + len(val_recs) + len(test_recs)
    assert total == len(records), f"Split total {total} != records {len(records)}"

    print(f"\nSplit:")
    print(f"  train: {len(train_recs):5d}  ({100*len(train_recs)/total:.1f}%)")
    print(f"  val:   {len(val_recs):5d}  ({100*len(val_recs)/total:.1f}%)")
    print(f"  test:  {len(test_recs):5d}  ({100*len(test_recs)/total:.1f}%)")

    # --- Score distribution per split ---
    print()
    for split_name, split_recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        scores = [r.get("validator_score", 0) for r in split_recs]
        mean   = sum(scores) / len(scores) if scores else 0
        bands  = Counter(score_band(s) for s in scores)
        print(f"  {split_name:5s}  mean={mean:.3f}  "
              f"A={bands['A']} B={bands['B']} C={bands['C']} D={bands['D']}")

    # --- Write ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_split(recs: list, name: str):
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for r in recs:
                out = format_for_sft(r) if not args.no_sft_format else r
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(recs):5d} records → {path}")

    print()
    write_split(train_recs, "train")
    write_split(val_recs,   "val")
    write_split(test_recs,  "test")

    # --- Stats JSON for writeup ---
    chem_dist = Counter(chemistry_class(r.get("target", "")) for r in records)
    thermo_dist = Counter(thermo_class(r) for r in records)

    stats = {
        "total_after_filter": len(records),
        "train": len(train_recs),
        "val":   len(val_recs),
        "test":  len(test_recs),
        "min_score_threshold": args.min_score,
        "seed": args.seed,
        "score_distribution": {
            "mean":   round(sum(r.get("validator_score",0) for r in records) / len(records), 4),
            "band_A": sum(1 for r in records if score_band(r.get("validator_score",0)) == "A"),
            "band_B": sum(1 for r in records if score_band(r.get("validator_score",0)) == "B"),
            "band_C": sum(1 for r in records if score_band(r.get("validator_score",0)) == "C"),
            "band_D": sum(1 for r in records if score_band(r.get("validator_score",0)) == "D"),
        },
        "chemistry_distribution": dict(chem_dist.most_common()),
        "thermodynamic_path_distribution": dict(thermo_dist.most_common()),
        "strata_count": len(by_stratum),
        "strata_sizes": {k: len(v) for k, v in sorted(
            by_stratum.items(), key=lambda x: -len(x[1]))[:20]},
    }
    stats_path = args.output_dir / "split_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote stats → {stats_path}")


if __name__ == "__main__":
    main()