"""
research/compute_comparator_scales.py — Phase 13 step 2
(misc/PHASE13_14_SPEC.md): MAD scales for core/comparator.py's channels,
computed from UNLABELED generations only (never ASTRAL — ASTRAL is the
readout, never the calibration signal, per the spec's own rule).

Sources (both required by the spec):
  - runs/gdpo-qlora-beta-ablation-probe/generations.jsonl
  - runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl  ("the Phase
    12 dump" — the beta=0 relaunch, the only Phase 12 dump that exists)

For each completion: parse -> score_channels() at the pre-registered C3
arm (fraction=0.17) -> collect the raw per-channel value where gradeable
(gate passed, channel not None). Scale = MAD * 1.4826 (the standard
normal-consistent robust-sigma estimator), computed once here and used
unchanged across all four C3 sensitivity-sweep arms in Phase 13 scoring —
C3's raw quantity is already self-normalized by its own w_low/w_high
inside the score (-(delta/w)^2), so one MAD computed at arm B is a
reasonable, disclosed choice rather than recomputing per arm.

Run in tmux (touches ThermoChecker / the PD cache):
  tmux new-session -d -s comparator_scales \
    "uv run python research/compute_comparator_scales.py \
     > run_logs/compute_comparator_scales.log 2>&1"
"""
import json
import random
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.comparator import CHANNEL_NAMES, ComparatorParams, load_comparator  # noqa: E402
from core.reward import ParseFailure, parse_completion  # noqa: E402

SOURCES = [
    Path("runs/gdpo-qlora-beta-ablation-probe/generations.jsonl"),
    Path("runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl"),
]
N_PER_SOURCE = 500  # matches the Phase 11 rail-calibration precedent's order
                     # of magnitude (200 completions); 500/source here since
                     # PD resolution is cached per chemsys and most sampled
                     # completions share a small number of targets.
SEED = 20260916  # today's date, fixed for reproducibility
PARAMS = ComparatorParams(c3_fraction=0.17)  # the pre-registered read (arm B)
OUT_PATH = Path("misc/comparator_scales_v1.json")


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
        if all(v is None for v in channels.values()):
            n_gate_fail += 1
        for c, v in channels.items():
            if v is not None:
                values[c].append(v)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(records)} scored", flush=True)

    print()
    print(f"n_records={len(records)} n_parsed={n_parsed} "
          f"n_parse_fail={n_parse_fail} n_gate_fail_all_none={n_gate_fail}")
    print()

    scales = {}
    diagnostics = {}
    for c in CHANNEL_NAMES:
        vals = values[c]
        if not vals:
            scales[c] = None
            diagnostics[c] = {"n": 0}
            print(f"{c}: NO GRADEABLE VALUES -- scale is None, channel unusable")
            continue
        med = median(vals)
        mad = median(abs(v - med) for v in vals)
        scale = mad * 1.4826
        n_gradeable_pct = round(100 * len(vals) / max(n_parsed, 1), 1)
        scales[c] = scale if scale > 0 else None
        diagnostics[c] = {
            "n": len(vals), "pct_gradeable_of_parsed": n_gradeable_pct,
            "median": med, "mad": mad, "scale": scale,
            "min": min(vals), "max": max(vals),
        }
        flag = "  <-- ZERO SPREAD, scale unusable" if scale == 0 else ""
        print(f"{c}: n={len(vals)} ({n_gradeable_pct}% of parsed) "
              f"median={med:.4g} MAD={mad:.4g} scale={scale:.4g}{flag}")

    out = {
        "comparator_version": comparator.__class__.__module__,
        "seed": SEED,
        "c3_fraction_used_for_calibration": PARAMS.c3_fraction,
        "sources": [str(s) for s in SOURCES],
        "n_per_source": N_PER_SOURCE,
        "n_records_sampled": len(records),
        "n_parsed": n_parsed,
        "n_parse_fail": n_parse_fail,
        "n_gate_fail_all_none": n_gate_fail,
        "scales": scales,
        "diagnostics": diagnostics,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
