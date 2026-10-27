#!/usr/bin/env python
"""
research/phase17_universe_scores.py — Phase 17 Block B prep: score every
existing raw completion (across all 12 models, both target sets) with
both checker variants and the ranker, so Block B's recipe reward r_s(y)
can be built by pooling these per-sample scores by (target, precset).

Per regular Claude's B1 decisions (2026-10-27):
  1. Reward per step = what the step actually trained on. S5 (run 1)
     trained on the ORIGINAL 10-channel checker (primary), with the
     5-channel RUN3 aggregate as a sensitivity check -- the pre-reg's
     own "S5: 5-channel" line contradicts its stated rule ("the reward
     the step itself trained on") and is logged as a pre-reg deviation
     in signature.json, not silently followed. S6 = 5-ch RUN3. S7 =
     ranker v2. S1-S4's covariate = 5-ch RUN3.
  2. r(y) per sample = plain mean of the gradeable (non-None) channels
     in that reward's channel set, pooled across all samples (any
     model) that produced y, averaged. Ungradeable channels excluded,
     not treated as 0 or 0.5 (validator's own None-propagation
     convention, generalized to this simple mean). GDPO's actual
     training-time advantage is a z-normalized SUM within a group of 8
     and is NOT reused here -- r(y) is a separate, simpler, absolute
     [0,1]-ish proxy Block B's own within-target advantage formula
     recenters itself (r~(y) = r(y) - sum_y' pi~(y')r(y')).
  3. Coverage: score every existing completion (not a subsample) whose
     parsed recipe is in that target's U(x) -- which is every
     successfully-parsed completion, since U(x) is BUILT from these
     same completions (research/phase17_block0_inventory_v2.py). CPU
     cost was timed empirically before running in full (200-sample
     timing: ~15.4ms/sample for validate(), ~12.0ms/sample for
     Ranker.score() -> ~1,480s / ~25min for the full 54,320-row corpus
     at both scorers, far under the 6h cap regular Claude set -- so no
     8-completions-per-recipe cap is applied).
  4. Ranker v2's reward is already a single gated/aggregated scalar
     (Ranker.score() returns (reward, breakdown)) -- there is no
     multi-channel mean to take for R4; its "reward" field is used
     directly.

TEN_CHANNELS is validator.py's full original check set (run 1, R1 in
Block R); RUN3_CHECKS (5ch, imported from core.reward) is the subset
R2/R-b/R-c already treat as "the checker" post-run-3.

NO EDIT to validator.py, core/reward.py, core/ranker.py. NO NEW
GENERATION -- scores only existing raw completion files listed in
results/phase17/inventory_v2.json.

Usage (tmux, CPU only -- thermo-backed validator/ranker calls):
  PYTHONPATH=. uv run python research/phase17_universe_scores.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.reward import ParseFailure, parse_completion, RUN3_CHECKS  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency  # noqa: E402
from research.support_growth_by_checkpoint import precset_of  # noqa: E402
from validator import SynthesisValidator, ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
INVENTORY_PATH = REPO_ROOT / "results" / "phase17" / "inventory_v2.json"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "universe_scores.jsonl"

TEN_CHANNELS = (
    "stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist",
    "operation_order", "temperature_plausible", "thermodynamic_favorable",
    "target_stability", "chempot_atmosphere", "target_match",
)
FIVE_CHANNELS = RUN3_CHECKS


def mean_gradeable(breakdown: dict, names: tuple[str, ...]) -> float | None:
    vals = [breakdown[n] for n in names if n in breakdown
            and isinstance(breakdown[n], (int, float)) and not isinstance(breakdown[n], bool)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main():
    print("loading formula set, PD cache, checker (v1), ranker (v2)...", flush=True)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    checker = SynthesisValidator(formula_set, thermo_checker=thermo, validator_version=1)
    freq = build_precursor_frequency(REPO_ROOT / "data" / "raw" / "synthesis_clean.json")
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)

    inventory = json.loads(INVENTORY_PATH.read_text())

    done_keys = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["model"], r["target_set"], r["row_index"]))
        print(f"resuming: {len(done_keys)} rows already scored", flush=True)

    out_fp = OUT_PATH.open("a")
    t_start = time.time()
    n_total = 0
    n_scored = 0
    n_parse_fail = 0

    for model, by_set in inventory["raw_completions"].items():
        for target_set, info in by_set.items():
            path_str = info.get("raw_completions_file")
            if not path_str:
                continue
            path = REPO_ROOT / path_str
            if not path.exists():
                continue
            print(f"\nscoring {model}/{target_set}: {path}", flush=True)
            with path.open() as f:
                for row_index, line in enumerate(f):
                    n_total += 1
                    if (model, target_set, row_index) in done_keys:
                        continue
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    target = rec.get("target")
                    completion = rec.get("completion", "")
                    try:
                        route = parse_completion(completion, target)
                    except Exception:
                        n_parse_fail += 1
                        continue
                    precset = precset_of(route)

                    try:
                        _, checker_bd = checker.validate(route, target)
                    except Exception:
                        checker_bd = {}
                    r1_10ch = mean_gradeable(checker_bd, TEN_CHANNELS)
                    r2_5ch = mean_gradeable(checker_bd, FIVE_CHANNELS)

                    try:
                        r4_reward, _ranker_bd = ranker.score(route, target)
                    except Exception:
                        r4_reward = None

                    out_fp.write(json.dumps({
                        "model": model, "target_set": target_set, "row_index": row_index,
                        "target": target, "recipe": list(precset),
                        "r1_10ch": r1_10ch, "r2_5ch": r2_5ch, "r4_ranker": r4_reward,
                    }) + "\n")
                    n_scored += 1
                    if n_scored % 2000 == 0:
                        out_fp.flush()
                        elapsed = time.time() - t_start
                        print(f"  ...{n_scored} scored, {n_parse_fail} parse-fail, "
                              f"{elapsed:.0f}s elapsed", flush=True)

    out_fp.close()
    elapsed = time.time() - t_start
    print(f"\nDONE: n_total_rows={n_total} n_scored={n_scored} n_parse_fail={n_parse_fail} "
          f"elapsed={elapsed:.0f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
