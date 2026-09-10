"""
pd_coverage_attribution.py
----------------------------
Follows coverage_audit.py + pd_shard_census.py. Census established: 8.5%
of indexed shards (1689/19861) fail to unpickle, 0 missing, on BOTH mac
and manifold - i.e. the corruption is baked into the shard files, not a
transfer artifact. But _resolve_pd (validator.py) retries every superset
candidate in pd_index before giving up, so a corrupted exact-match shard
doesn't necessarily block a record if some other covering superset in
the cache happens to be intact. This script measures how often it
actually does block a record, for the real corpus, not a random sample.

For every light_only record, re-runs reaction_energy_per_atom and
target_e_above_hull with ThermoChecker._get_pd instrumented to log every
chemsys candidate it tries and how that attempt resolved:
  ok         - shard loaded successfully
  corrupted  - chemsys IS in pd_index, but _get_pd still returned None
               (census already proved this means unpicklable, not missing)
  absent     - chemsys is not a key in pd_index at all

Each of the two resolve paths (reaction / hull) is then bucketed:
  blocked_by_corruption  - at least one candidate was corrupted, none ok
  genuinely_uncovered     - every candidate attempted was absent (i.e. only
                            the initial exact-match attempt happened, and
                            it wasn't in the index - no superset exists
                            in the cache at all for these elements)
  shard_ok_other_failure  - a PD DID load (some candidate was "ok") but the
                            final value was still None - i.e. the hull/
                            reaction loaded fine but the specific formula
                            entry wasn't in it, or ComputedReaction
                            couldn't balance. This is a different problem
                            than coverage and re-fetching won't fix it.

Top-line number this produces: of current light_only records, how many
would plausibly move to real_dG/hull_only if the 1689 corrupted shards
were simply re-fetched from the MP API, vs how many need a shard that
was never built at all, vs how many have a working shard already and
the failure is somewhere else (entry mismatch, reaction imbalance).

Usage: same flags as coverage_audit.py.

  uv run python pd_coverage_attribution.py \
      --split data/sft/train.jsonl \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --out attribution_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from core.reward import parse_completion, ParseFailure, load_validator
except ImportError:
    from reward import parse_completion, ParseFailure, load_validator


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def instrument(tc):
    """
    Wraps tc._get_pd to log (chemsys, status) for every call, without
    touching validator.py. Returns (restore_fn, log_list). Caller clears
    log_list.clear() between resolve calls it wants attributed separately.
    """
    original = tc._get_pd
    log = []

    def wrapped(chemsys):
        result = original(chemsys)
        if result is not None:
            status = "ok"
        elif chemsys in tc.pd_index:
            status = "corrupted"  # census proved missing=0 -> must be corrupted
        else:
            status = "absent"
        log.append((chemsys, status))
        return result

    tc._get_pd = wrapped

    def restore():
        tc._get_pd = original

    return restore, log


def bucket(log: list[tuple[str, str]]) -> str:
    """
    NOTE: earlier version took a `resolved` bool derived from the final
    dG/hull scalar (e.g. `dG is not None`). That's always False here,
    since this is only ever called on already-filtered light_only
    records - so the shard_ok_other_failure branch could never fire and
    everything real fell into 'unknown'. Fixed: whether a PD was found
    is read off the log itself (any "ok" entry), independent of whether
    the final scalar came back None.
    """
    statuses = {s for _, s in log}
    if "ok" in statuses:
        # a PD loaded successfully (possibly after a corrupted candidate
        # failed first) but the final value was still None - the failure
        # is downstream of coverage: entry lookup, reaction balance, or
        # an exception inside gibbs_corrector.
        return "shard_ok_other_failure"
    if "corrupted" in statuses:
        return "blocked_by_corruption"
    if statuses <= {"absent"} or not statuses:
        return "genuinely_uncovered"
    return "unknown"


def gradeability_of(dG, hull) -> str:
    if dG is not None:
        return "real_dG"
    if hull is not None:
        return "hull_only"
    return "light_only"


def attribute_split(path: Path, validator) -> dict:
    examples = load_jsonl(path)
    tc = validator.thermo_checker
    restore, log = instrument(tc)

    reaction_buckets = Counter()
    hull_buckets = Counter()
    combined = Counter()
    n_light_only = 0
    n_total = len(examples)
    n_parse_fail = 0
    sample_corruption_blocked = []  # first few, for spot-checking

    try:
        for ex in examples:
            target = ex["target"]
            try:
                route = parse_completion(ex["completion"], target)
            except ParseFailure:
                n_parse_fail += 1
                continue

            precs = [(p.formula, p.amount) for p in route.precursors]

            log.clear()
            try:
                dG = tc.reaction_energy_per_atom(precs, target, predicted_route=route)
            except Exception:
                dG = None
            reaction_log = list(log)

            log.clear()
            try:
                hull = tc.target_e_above_hull(target)
            except Exception:
                hull = None
            hull_log = list(log)

            if gradeability_of(dG, hull) != "light_only":
                continue

            n_light_only += 1
            rb = bucket(reaction_log)
            hb = bucket(hull_log)
            reaction_buckets[rb] += 1
            hull_buckets[hb] += 1
            combined[(rb, hb)] += 1

            if rb == "blocked_by_corruption" and len(sample_corruption_blocked) < 10:
                sample_corruption_blocked.append({
                    "target": target,
                    "reaction_log": reaction_log,
                    "hull_log": hull_log,
                })
    finally:
        restore()

    corruption_implicated = sum(
        c for (rb, hb), c in combined.items()
        if rb == "blocked_by_corruption" or hb == "blocked_by_corruption"
    )
    genuinely_uncovered_both = combined.get(("genuinely_uncovered", "genuinely_uncovered"), 0)

    return {
        "path": str(path),
        "n_total": n_total,
        "n_parse_fail": n_parse_fail,
        "n_light_only": n_light_only,
        "reaction_buckets": dict(reaction_buckets),
        "hull_buckets": dict(hull_buckets),
        "combined_cross_tab": {f"{rb}|{hb}": c for (rb, hb), c in combined.items()},
        "corruption_implicated": corruption_implicated,
        "corruption_implicated_frac_of_light_only": (
            round(corruption_implicated / n_light_only, 4) if n_light_only else 0.0
        ),
        "genuinely_uncovered_both_frac_of_light_only": (
            round(genuinely_uncovered_both / n_light_only, 4) if n_light_only else 0.0
        ),
        "sample_corruption_blocked": sample_corruption_blocked,
    }


def print_summary(result: dict):
    print(f"\n=== {result['path']}  (light_only={result['n_light_only']}/{result['n_total']}) ===")
    print(f"reaction resolve buckets: {result['reaction_buckets']}")
    print(f"hull resolve buckets:     {result['hull_buckets']}")
    print(f"\ncorruption-implicated (reaction OR hull blocked by a corrupted shard): "
          f"{result['corruption_implicated']}/{result['n_light_only']} "
          f"({result['corruption_implicated_frac_of_light_only']:.2%} of light_only)")
    print(f"genuinely uncovered on both paths (no shard exists at all): "
          f"{result['genuinely_uncovered_both_frac_of_light_only']:.2%} of light_only")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, dest="splits")
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("attribution_results.json"))
    args = ap.parse_args()

    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for split_path in args.splits:
        p = Path(split_path)
        print(f"Attributing {p} ...", file=sys.stderr)
        result = attribute_split(p, validator)
        all_results[p.name] = result
        print_summary(result)

    with args.out.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()