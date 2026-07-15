"""
kononova_triage.py  (v2 — parallel, memory-bounded, resumable)
------------------------------------------------------------------
Cost-free pre-flight triage over the FULL raw Kononova synthesis corpus
(data/raw/synthesis_clean.json, ~17,616 records), using the patched
validator's own gradeability logic - NO LLM / OpenRouter calls.

v2 changes from the original single-threaded version, both driven by
real problems hit running it on the full corpus:

1. PARALLEL (ProcessPoolExecutor, not threading - this workload is
   CPU-bound Python/numpy inside pymatgen, threads would just fight the
   GIL). Each of --workers processes builds its OWN persistent
   ThermoChecker/validator ONCE at worker startup (expensive to build,
   must not be rebuilt per record), then processes batches of records.

2. MEMORY-BOUNDED. ThermoChecker._get_pd caches every PhaseDiagram it
   ever loads in self.phase_diagrams with NO eviction (confirmed by
   reading validator.py directly - `self.phase_diagrams[chemsys] = pd`,
   plain dict, never cleared). Fine for a normal run touching a small,
   repeated chemsys set; NOT fine for a 17.6k-record sweep across
   diverse raw Kononova chemistry, which is what actually OOM'd a
   single-threaded run at 93GB/123GB with the process barely half done.
   Naive parallelism makes this WORSE (N workers = N independent
   unbounded caches), so each worker clears its OWN
   validator.thermo_checker.phase_diagrams every --cache-clear-every
   records it personally processes.

3. RESUMABLE / KILL-SAFE, mirroring evaluate_batched.py's established
   pattern in this codebase: full JSON overwrite (all records-so-far)
   every --checkpoint-every completed records, and on startup, read any
   existing output file back in and skip indices already present. With
   parallel workers, results arrive OUT OF ORDER (as_completed, not
   map), so resume matching is by explicit idx, not by loop position.

Usage:
  uv run python kononova_triage.py \
      --synthesis data/raw/synthesis_clean.json \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --workers 16 \
      --batch-size 50 \
      --cache-clear-every 200 \
      --checkpoint-every 500 \
      --out kononova_triage_results.json

--workers: start conservative (e.g. 8-16 on a 32-core box) and watch
`free -h` for the first couple of checkpoints. Memory is now bounded
PER WORKER, but total = workers x per-worker footprint, so more workers
means more concurrent shard-loading pressure even with clearing. If
memory looks fine, bump it up on the next run (this script resumes, so
nothing is lost by stopping and restarting with a different --workers).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


# reaction_string looks like:
#   "0.98 BaCO3 + 0.01 La2O3 + 1 SnO2 == 1 Ba0.98La0.02SnO3 + 0.98 CO2 + 0.005 O2"
_COEFF_TERM = re.compile(r"([0-9]*\.?[0-9]+)\s+([A-Za-z0-9().]+)")


def parse_reaction_lhs_amounts(reaction_string: str) -> dict[str, float]:
    if not reaction_string or "==" not in reaction_string:
        return {}
    lhs = reaction_string.split("==", 1)[0]
    amounts = {}
    for coeff, formula in _COEFF_TERM.findall(lhs):
        try:
            amounts[formula] = float(coeff)
        except ValueError:
            continue
    return amounts


def flatten_temps(heating_temperature) -> list[float]:
    flat = []
    for group in heating_temperature or []:
        if isinstance(group, (list, tuple)):
            for t in group:
                try:
                    flat.append(float(t))
                except (TypeError, ValueError):
                    pass
        else:
            try:
                flat.append(float(group))
            except (TypeError, ValueError):
                pass
    return flat


def build_route(record: dict):
    """Constructs a PredictedRoute directly from a raw Kononova record.
    Imports are LOCAL (inside the function) so this module can be
    imported by the main process without requiring pymatgen/validator to
    already be importable there - only workers need the heavy imports,
    and importing them lazily inside the worker process (after fork)
    avoids any issue with unpicklable module state crossing the
    process boundary."""
    try:
        from core.validator import PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions
    except ImportError:
        from validator import PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions

    target = record.get("target_formula")
    if not target:
        return None
    raw_precursors = record.get("precursors", [])
    if not raw_precursors:
        return None

    amounts = parse_reaction_lhs_amounts(record.get("reaction_string", ""))

    precursors = []
    for p in raw_precursors:
        formula = p.get("formula")
        if not formula:
            continue
        amount = amounts.get(formula, 1.0)
        precursors.append(PredictedPrecursor(formula=formula, amount=amount))
    if not precursors:
        return None

    operations = []
    for op in record.get("operations", []):
        op_type = op.get("type", "")
        atmos = op.get("heating_atmosphere") or []
        conditions = PredictedConditions(
            heating_temperature=flatten_temps(op.get("heating_temperature")),
            heating_time=flatten_temps(op.get("heating_time")),
            heating_atmosphere=[str(a) for a in atmos if a],
            mixing_media=op.get("mixing_media"),
        )
        operations.append(PredictedOperation(type=op_type, conditions=conditions))

    return PredictedRoute(
        target_formula=target,
        precursors=precursors,
        operations=operations,
        reaction_string=record.get("reaction_string", ""),
    )


def is_fractional(formula: str, tol: float = 1e-6) -> bool:
    try:
        from pymatgen.core import Composition
        amounts = Composition(formula).as_dict().values()
    except Exception:
        return False
    return any(abs(a - round(a)) > tol for a in amounts)


# ---------------------------------------------------------------------------
# Worker process state. Set once per worker by _worker_init (runs after
# fork, in each worker's own memory space - NOT shared with the main
# process or other workers).
# ---------------------------------------------------------------------------
_worker_validator = None
_worker_n_since_clear = 0
_worker_cache_clear_every = 200


def _worker_init(formula_set_path, pd_index_path, project_root, cache_clear_every):
    global _worker_validator, _worker_n_since_clear, _worker_cache_clear_every
    try:
        from core.reward import load_validator
    except ImportError:
        from reward import load_validator
    _worker_validator = load_validator(
        Path(formula_set_path), Path(pd_index_path), Path(project_root)
    )
    _worker_n_since_clear = 0
    _worker_cache_clear_every = cache_clear_every


def _score_batch(batch: list[tuple[int, dict]]) -> list[dict]:
    """Runs in a worker process. batch = [(idx, raw_record), ...].
    Returns one result dict per record. Never raises - all failures are
    captured as a status field so one bad record can't kill a batch."""
    global _worker_n_since_clear

    results = []
    for idx, record in batch:
        route = build_route(record)
        if route is None:
            results.append({"idx": idx, "status": "build_failed"})
            continue

        try:
            reward, breakdown = _worker_validator.validate(route, route.target_formula)
        except Exception as exc:
            results.append({"idx": idx, "status": "validate_exception", "error": str(exc)[:200]})
            continue

        rs_amounts = parse_reaction_lhs_amounts(record.get("reaction_string", ""))
        n_from_rs = sum(1 for p in route.precursors if p.formula in rs_amounts)

        results.append({
            "idx": idx,
            "status": "graded",
            "target": route.target_formula,
            "is_fractional": is_fractional(route.target_formula),
            "reward": reward,
            "thermo_tier": breakdown.get("thermodynamic_favorable_gradeability"),
            "target_stability_tier": breakdown.get("target_stability_gradeability"),
            "amount_accuracy": breakdown.get("amount_accuracy"),
            "amount_accuracy_tier": breakdown.get("amount_accuracy_gradeability"),
            "stoichiometry": breakdown.get("stoichiometry"),
            "n_precursors": len(route.precursors),
            "n_amounts_from_reaction_string": n_from_rs,
        })

        _worker_n_since_clear += 1
        if _worker_n_since_clear >= _worker_cache_clear_every:
            _worker_validator.thermo_checker.phase_diagrams.clear()
            _worker_n_since_clear = 0

    return results


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def summarize(records: list[dict]) -> dict:
    from collections import Counter
    import statistics

    graded = [r for r in records if r["status"] == "graded"]
    status_counts = Counter(r["status"] for r in records)
    thermo_tier = Counter(r.get("thermo_tier") for r in graded)
    frac_counts = Counter("fractional" if r["is_fractional"] else "integer" for r in graded)
    tier_by_frac = {}
    for r in graded:
        key = "fractional" if r["is_fractional"] else "integer"
        tier_by_frac.setdefault(key, Counter())[r.get("thermo_tier")] += 1

    rewards = [r["reward"] for r in graded if isinstance(r.get("reward"), (int, float))]
    amount_scores = [r["amount_accuracy"] for r in graded if isinstance(r.get("amount_accuracy"), (int, float))]
    stoich_scores = [r["stoichiometry"] for r in graded if isinstance(r.get("stoichiometry"), (int, float))]

    def dist(xs):
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs), "median": round(statistics.median(xs), 4),
            "mean": round(statistics.mean(xs), 4),
            "frac_ge_0.9": round(sum(1 for x in xs if x >= 0.9) / len(xs), 3),
        }

    n_graded = len(graded)
    gradeable = thermo_tier.get("discrete", 0) + thermo_tier.get("interpolated", 0)

    return {
        "n_total_processed": len(records),
        "status_counts": dict(status_counts),
        "n_graded": n_graded,
        "thermo_gradeability_tiers": dict(thermo_tier),
        "thermo_gradeability_fractions": (
            {k: round(v / n_graded, 4) for k, v in thermo_tier.items()} if n_graded else {}
        ),
        "target_stoichiometry_type": dict(frac_counts),
        "tier_by_stoichiometry": {k: dict(v) for k, v in tier_by_frac.items()},
        "reward_distribution": dist(rewards),
        "amount_accuracy_distribution": dist(amount_scores),
        "stoichiometry_distribution": dist(stoich_scores),
        "headline_gradeable_fraction": round(gradeable / n_graded, 4) if n_graded else 0.0,
        "headline_gradeable_count": gradeable,
    }


def write_checkpoint(records: list[dict], out_path: Path, meta: dict):
    payload = {"meta": meta, "summary": summarize(records), "records": records}
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(out_path)  # atomic on POSIX - avoids a truncated file if killed mid-write


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthesis", type=Path, required=True)
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("kononova_triage_results.json"))
    ap.add_argument("--workers", type=int, default=8,
                     help="start conservative and watch memory; bump up on a resumed rerun if headroom allows")
    ap.add_argument("--batch-size", type=int, default=50,
                     help="records per task submitted to a worker - amortizes IPC overhead")
    ap.add_argument("--cache-clear-every", type=int, default=200,
                     help="records a worker processes before clearing its own PD cache")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                     help="results accumulated before rewriting the output file")
    args = ap.parse_args()

    all_records = json.loads(args.synthesis.read_text())
    if args.sample:
        all_records = all_records[:args.sample]
    total = len(all_records)
    print(f"Loaded {total} raw Kononova records.", file=sys.stderr)

    records: list[dict] = []
    done_idx: set[int] = set()
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text())
            records = prev.get("records", [])
            done_idx = {r["idx"] for r in records}
            print(f"Resuming: {len(records)} already processed.", file=sys.stderr)
        except Exception as e:
            print(f"Could not read prior output ({e}); starting fresh.", file=sys.stderr)

    todo = [(i, rec) for i, rec in enumerate(all_records) if i not in done_idx]
    if not todo:
        print("All records already processed.", file=sys.stderr)
        write_checkpoint(records, args.out, {"synthesis_file": str(args.synthesis), "n_total": total})
        print(json.dumps(summarize(records), indent=2))
        return

    print(f"{len(todo)} remaining, {args.workers} workers, batch size {args.batch_size}.", file=sys.stderr)

    batches = list(chunked(todo, args.batch_size))
    meta = {
        "synthesis_file": str(args.synthesis), "n_total": total,
        "workers": args.workers, "batch_size": args.batch_size,
    }

    t0 = time.time()
    n_since_checkpoint = 0
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(str(args.formula_set), str(args.pd_index), str(args.project_root), args.cache_clear_every),
        ) as executor:
            futures = [executor.submit(_score_batch, b) for b in batches]
            for fut in as_completed(futures):
                batch_results = fut.result()
                records.extend(batch_results)
                n_since_checkpoint += len(batch_results)

                if n_since_checkpoint >= args.checkpoint_every:
                    write_checkpoint(records, args.out, meta)
                    n_since_checkpoint = 0
                    elapsed = time.time() - t0
                    rate = len(records) / elapsed if elapsed > 0 else 0
                    remaining = (len(todo) + len(done_idx) - len(records)) / rate if rate > 0 else float("inf")
                    print(f"timestamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]")
                    print(
                        f"  {len(records)}/{total}  ",
                        f"({rate:.1f} rec/s, ~{remaining/60:.1f}m remaining)",
                        file=sys.stderr,
                    )
    finally:
        # Always checkpoint on the way out, including on Ctrl-C or a worker
        # crash surfaced as an exception - never lose completed work.
        write_checkpoint(records, args.out, meta)

    print(f"\nWrote {args.out}", file=sys.stderr)
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()