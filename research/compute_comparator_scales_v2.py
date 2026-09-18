"""
research/compute_comparator_scales_v2.py — Phase 13 iteration 2
(misc/PHASE13_PREREG.md addendum 2, 2026-09-18): rank-transform reference
distributions for core/comparator.py's channels, replacing iteration 1's
raw-diff/MAD scale (research/compute_comparator_scales.py ->
misc/comparator_scales_v1.json, KEPT as the iteration-1 historical
record, not overwritten).

Why: iteration 1's MAD scale for C7_gas_evolution was 2.085e-05 (most
calibration completions release no gas at all), inflating a raw diff of
~1 by ~50,000x and dominating every margin (misc/PHASE13_RESULTS.md).
Iteration 2 stores the SORTED array of each channel's calibration raw
values instead of a single scale number; at scoring time a route's value
maps to its percentile rank (0-1) within that distribution, so the
pairwise diff (percentile_a - percentile_b) is bounded in [-1, 1]
regardless of the channel's own scale -- this also fixes C4's exact-zero
MAD and C6's saturation in the same change (core/comparator.py's
_percentile_rank).

Same sources, same sampling seed as iteration 1, for direct comparability
(both required by the spec; never ASTRAL):
  - runs/gdpo-qlora-beta-ablation-probe/generations.jsonl
  - runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl

Uses the FIXED comparator (core.comparator.Comparator, which installs
_ComparatorValidator internally) -- so this calibration pass also picks
up any ammonium-precursor completions in the corpus that iteration 1's
gate bug silently zeroed out everywhere (see the separate historical
audit, research/audit_ammonium_precursor_history.py).

Run in tmux (touches ThermoChecker / the PD cache):
  tmux new-session -d -s comparator_scales_v2 \
    "uv run python research/compute_comparator_scales_v2.py \
     > run_logs/compute_comparator_scales_v2.log 2>&1"
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.comparator import CHANNEL_NAMES, ComparatorParams, load_comparator  # noqa: E402
from core.reward import ParseFailure, parse_completion  # noqa: E402

SOURCES = [
    Path("runs/gdpo-qlora-beta-ablation-probe/generations.jsonl"),
    Path("runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl"),
]
N_PER_SOURCE = 500  # unchanged from iteration 1, for comparability
SEED = 20260916     # unchanged from iteration 1, for comparability
PARAMS = ComparatorParams(c3_fraction=0.17)  # the pre-registered read (arm B)
OUT_PATH = Path("misc/comparator_scales_v2.json")


def load_sample(path: Path, n: int, rng: random.Random) -> list[dict]:
    lines = path.read_text().splitlines()
    if len(lines) > n:
        lines = rng.sample(lines, n)
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main():
    rng = random.Random(SEED)
    comparator = load_comparator(
        Path("data/cache/mp_formula_set.pkl"),
        Path("data/cache/pd_index.json"),
        Path("."),
    )
    assert comparator.thermo is not None, "real PD cache required"

    records = []
    for src in SOURCES:
        if not src.exists():
            print(f"MISSING (skipped): {src}")
            continue
        sample = load_sample(src, N_PER_SOURCE, rng)
        print(f"{src}: sampled {len(sample)} completions")
        records.extend(sample)

    values: dict[str, list[float]] = {c: [] for c in CHANNEL_NAMES}
    n_parsed, n_parse_fail, n_gate_fail = 0, 0, 0
    n_ammonium_recovered = 0  # gate-passed completions using an ammonium precursor
    for i, rec in enumerate(records):
        target = rec.get("target")
        completion = rec.get("completion")
        if not target or not completion:
            continue
        try:
            route = parse_completion(completion, target)
        except ParseFailure:
            n_parse_fail += 1
            continue
        except Exception:
            n_parse_fail += 1
            continue
        n_parsed += 1
        channels = comparator.score_channels(route, target, PARAMS)
        all_none = all(v is None for v in channels.values())
        if all_none:
            n_gate_fail += 1
        else:
            formulas = {p.formula for p in (route.precursors or [])}
            if any("NH4" in f or "NH3" in f for f in formulas):
                n_ammonium_recovered += 1
        for c, v in channels.items():
            if v is not None:
                values[c].append(v)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(records)} scored", flush=True)

    print()
    print(f"n_records={len(records)} n_parsed={n_parsed} "
          f"n_parse_fail={n_parse_fail} n_gate_fail_all_none={n_gate_fail} "
          f"n_ammonium_precursor_gate_passed={n_ammonium_recovered}")
    print()

    scales = {}
    diagnostics = {}
    for c in CHANNEL_NAMES:
        vals = sorted(values[c])
        n = len(vals)
        n_gradeable_pct = round(100 * n / max(n_parsed, 1), 1)
        if not vals:
            scales[c] = None
            diagnostics[c] = {"n": 0, "pct_gradeable_of_parsed": 0.0}
            print(f"{c}: NO GRADEABLE VALUES -- unusable")
            continue
        scales[c] = vals
        diagnostics[c] = {
            "n": n, "pct_gradeable_of_parsed": n_gradeable_pct,
            "min": vals[0], "median": vals[n // 2], "max": vals[-1],
        }
        print(f"{c}: n={n} ({n_gradeable_pct}% of parsed) "
              f"min={vals[0]:.4g} median={vals[n // 2]:.4g} max={vals[-1]:.4g}")

    out = {
        "comparator_version": comparator.__class__.__module__,
        "iteration": 2,
        "method": "sorted raw-value arrays for rank-transform "
                  "(core.comparator._percentile_rank), replacing "
                  "iteration 1's raw-diff/MAD scale",
        "seed": SEED,
        "c3_fraction_used_for_calibration": PARAMS.c3_fraction,
        "sources": [str(s) for s in SOURCES],
        "n_per_source": N_PER_SOURCE,
        "n_records_sampled": len(records),
        "n_parsed": n_parsed,
        "n_parse_fail": n_parse_fail,
        "n_gate_fail_all_none": n_gate_fail,
        "n_ammonium_precursor_gate_passed": n_ammonium_recovered,
        "scales": scales,
        "diagnostics": diagnostics,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
