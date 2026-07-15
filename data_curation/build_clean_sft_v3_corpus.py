"""
build_clean_sft_v3_corpus.py
-------------------------------
The final step of this session's data-cleaning arc. Does three things in
one authoritative pass, reusing every piece of logic validated this
session (rescore_sft_corpus.py's route-building, scan_dash_notation.py's
composition-corruption check, the scale-invariant amount_accuracy fix):

1. Excludes any record whose target formula silently corrupts under
   Composition() (dash-immediately-followed-by-digit notation, e.g.
   "Na2O-2SiO2" -> {Na:2, Si:1}, oxygen silently dropped, no error
   anywhere). Checked BEFORE validation - a corrupted composition makes
   every downstream score meaningless, not worth computing.

2. Fully rescores every remaining record with the patched validator and
   excludes anything below VALIDATOR_THRESHOLD (0.65, the same gate
   generate_traces_openrouter.py used originally).

3. Writes surviving records to a clean output jsonl with COMPLETELY
   FRESH validator_score / validator_breakdown / passed_validator /
   score_band (not just the score, the full breakdown - this is a
   genuine re-score, not a patch of stale fields), tagged with a
   validator_version field so provenance is unambiguous going forward -
   this file's whole reason for existing is the confusion from
   untagged validator_score / validator_score_old fields in the
   original corpus, don't reproduce that problem here.

Writes two files:
  --out            clean jsonl, ready for the SFT v3 train/test/val split
  --manifest        JSON: every excluded record, its reason, and the
                     scores/composition data that justified exclusion

Usage:
  uv run python build_clean_sft_v3_corpus.py \
      --synthesis data/processed/synthesis_with_traces.jsonl \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --workers 8 \
      --out data/processed/synthesis_with_traces_clean.jsonl \
      --manifest sft_v3_exclusion_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

VALIDATOR_THRESHOLD = 0.65  # generate_traces_openrouter.py:69
VALIDATOR_VERSION = "patched-2026-07-scale-invariant-amount-accuracy"


def approximate_score_band(score: float) -> str:
    if score >= 0.90:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= VALIDATOR_THRESHOLD:
        return "C"
    return "D"


def parse_segment(segment: str):
    m = re.match(r"^(\d+)(.*)$", segment.strip())
    if m and m.group(2):
        return int(m.group(1)), m.group(2)
    return 1, segment.strip()


def dash_corruption_check(target: str):
    """Returns 'silently_wrong', 'raises_safely', 'fine', or 'no_dash'.
    See scan_dash_notation.py for the full mechanism - this is the same
    logic, condensed for use as a pre-validation gate."""
    if "-" not in target:
        return "no_dash"

    from pymatgen.core import Composition

    try:
        actual = Composition(target).as_dict()
        actual_raised = False
    except Exception:
        actual = None
        actual_raised = True

    true_comp: dict[str, float] = {}
    for raw_segment in target.split("-"):
        if not raw_segment.strip():
            continue
        coeff, formula = parse_segment(raw_segment)
        try:
            comp = Composition(formula)
        except Exception:
            return "raises_safely" if actual_raised else "fine_but_unverifiable"
        for el, amt in comp.as_dict().items():
            true_comp[el] = true_comp.get(el, 0.0) + coeff * amt

    if actual_raised:
        return "raises_safely"
    if set(actual.keys()) == set(true_comp.keys()) and all(
        abs(actual.get(k, 0) - v) < 1e-6 for k, v in true_comp.items()
    ):
        return "fine"
    return "silently_wrong"


def build_route(record: dict):
    try:
        from core.validator import PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions
    except ImportError:
        from validator import PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions

    target = record.get("target")
    pr = record.get("predicted_route")
    if not target or not pr:
        return None
    raw_precursors = pr.get("precursors", [])
    if not raw_precursors:
        return None
    precursors = []
    for p in raw_precursors:
        formula = p.get("formula")
        if not formula:
            continue
        precursors.append(PredictedPrecursor(formula=formula, amount=p.get("amount", 1.0)))
    if not precursors:
        return None

    operations = []
    for op in pr.get("operations", []):
        temp_c = op.get("temperature_c")
        time_h = op.get("time_h")
        atmos = op.get("atmosphere")
        conditions = PredictedConditions(
            heating_temperature=[float(temp_c)] if temp_c is not None else [],
            heating_time=[float(time_h)] if time_h is not None else [],
            heating_atmosphere=[str(atmos)] if atmos else [],
            mixing_media=op.get("media"),
        )
        operations.append(PredictedOperation(type=op.get("type", ""), conditions=conditions))

    return PredictedRoute(target_formula=target, precursors=precursors, operations=operations)


_worker_validator = None
_worker_n_since_clear = 0
_worker_cache_clear_every = 200


def _worker_init(formula_set_path, pd_index_path, project_root, cache_clear_every):
    global _worker_validator, _worker_n_since_clear, _worker_cache_clear_every
    try:
        from core.reward import load_validator
    except ImportError:
        from reward import load_validator
    _worker_validator = load_validator(Path(formula_set_path), Path(pd_index_path), Path(project_root))
    _worker_n_since_clear = 0
    _worker_cache_clear_every = cache_clear_every


def _process_batch(batch: list[tuple[int, dict]]) -> list[dict]:
    global _worker_n_since_clear

    results = []
    for idx, record in batch:
        target = record.get("target", "")

        dash_status = dash_corruption_check(target)
        if dash_status == "silently_wrong":
            results.append({
                "idx": idx, "decision": "excluded", "target": target,
                "reason": "silent_composition_corruption",
                "detail": f"Composition('{target}') silently drops elements - dash-adjacent-digit notation bug",
            })
            continue

        route = build_route(record)
        if route is None:
            results.append({
                "idx": idx, "decision": "excluded", "target": target,
                "reason": "missing_fields", "detail": "no target or predicted_route",
            })
            continue

        try:
            new_score, new_breakdown = _worker_validator.validate(route, target)
        except Exception as e:
            results.append({
                "idx": idx, "decision": "excluded", "target": target,
                "reason": "validate_exception", "detail": str(e)[:200],
            })
            continue

        new_passed = new_score >= VALIDATOR_THRESHOLD

        if not new_passed:
            results.append({
                "idx": idx, "decision": "excluded", "target": target,
                "reason": "failed_validator_threshold",
                "detail": f"score={new_score:.4f} < {VALIDATOR_THRESHOLD}",
                "old_score": record.get("validator_score"),
                "new_score": new_score,
                "new_breakdown": new_breakdown,
            })
            continue

        results.append({
            "idx": idx, "decision": "included", "target": target,
            "old_score": record.get("validator_score"),
            "new_score": new_score,
            "new_breakdown": new_breakdown,
            "new_band": approximate_score_band(new_score),
        })

        _worker_n_since_clear += 1
        if _worker_n_since_clear >= _worker_cache_clear_every:
            _worker_validator.thermo_checker.phase_diagrams.clear()
            _worker_n_since_clear = 0

    return results


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthesis", type=Path, required=True)
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--cache-clear-every", type=int, default=200)
    ap.add_argument("--checkpoint", type=Path, default=Path(".build_clean_corpus_checkpoint.json"),
                     help="internal resume checkpoint - separate from --out, which is only written at the end")
    args = ap.parse_args()

    with args.synthesis.open() as f:
        all_records = [json.loads(line) for line in f if line.strip()]
    total = len(all_records)
    print(f"Loaded {total} records from {args.synthesis}", file=sys.stderr)

    done_results: list[dict] = []
    done_idx: set[int] = set()
    if args.checkpoint.exists():
        try:
            done_results = json.loads(args.checkpoint.read_text())
            done_idx = {r["idx"] for r in done_results}
            print(f"Resuming: {len(done_results)} already processed.", file=sys.stderr)
        except Exception as e:
            print(f"Could not read checkpoint ({e}); starting fresh.", file=sys.stderr)

    todo = [(i, rec) for i, rec in enumerate(all_records) if i not in done_idx]
    if todo:
        print(f"{len(todo)} remaining, {args.workers} workers.", file=sys.stderr)
        batches = list(chunked(todo, args.batch_size))
        t0 = time.time()
        n_since_checkpoint = 0
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_worker_init,
                initargs=(str(args.formula_set), str(args.pd_index), str(args.project_root), args.cache_clear_every),
            ) as executor:
                futures = [executor.submit(_process_batch, b) for b in batches]
                for fut in as_completed(futures):
                    batch_results = fut.result()
                    done_results.extend(batch_results)
                    n_since_checkpoint += len(batch_results)
                    if n_since_checkpoint >= 200:
                        args.checkpoint.write_text(json.dumps(done_results))
                        n_since_checkpoint = 0
                        elapsed = time.time() - t0
                        rate = len(done_results) / elapsed if elapsed > 0 else 0
                        print(f"  {len(done_results)}/{total}  ({rate:.1f} rec/s)", file=sys.stderr)
        finally:
            args.checkpoint.write_text(json.dumps(done_results))

    # --- assemble final outputs, in original idx order ---
    by_idx = {r["idx"]: r for r in done_results}
    included_count = 0
    excluded_manifest = []

    with args.out.open("w") as out_f:
        for idx in range(total):
            decision = by_idx.get(idx)
            if decision is None:
                continue  # shouldn't happen if the run completed
            if decision["decision"] == "excluded":
                excluded_manifest.append(decision)
                continue

            record = dict(all_records[idx])  # preserve every original field
            record["validator_score"] = decision["new_score"]
            record["validator_breakdown"] = decision["new_breakdown"]
            record["passed_validator"] = True
            record["score_band"] = decision["new_band"]
            record["validator_version"] = VALIDATOR_VERSION
            record.pop("validator_score_old", None)
            record.pop("validator_breakdown_old", None)
            out_f.write(json.dumps(record) + "\n")
            included_count += 1

    args.manifest.write_text(json.dumps({
        "source": str(args.synthesis),
        "validator_version": VALIDATOR_VERSION,
        "validator_threshold": VALIDATOR_THRESHOLD,
        "n_total": total,
        "n_included": included_count,
        "n_excluded": len(excluded_manifest),
        "exclusion_reason_counts": {
            reason: sum(1 for r in excluded_manifest if r["reason"] == reason)
            for reason in set(r["reason"] for r in excluded_manifest)
        },
        "excluded_records": excluded_manifest,
    }, indent=2, default=str))

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"DONE: {included_count}/{total} included -> {args.out}", file=sys.stderr)
    print(f"{len(excluded_manifest)}/{total} excluded -> {args.manifest}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()