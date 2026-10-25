#!/usr/bin/env python
"""
research/phase17_block0_inventory_v2.py — rebuild Block 0's universe/probe
set once, now that Block A's GPU regeneration job filled two of its six
raw-completions gaps: corpus_sft/heldout and gdpo_v3_ckpt300/heldout now
have real <think>-trace completions (data/novelty/corpus_sft_heldout_n32
.jsonl, data/novelty/gdpo300_heldout_n32.jsonl -- vLLM pipeline, saves
full completion text, unlike the HF ASTRAL pipeline used for the other
four legs).

Reuses research/phase17_block0_inventory.py's logic UNCHANGED (import,
not copy-paste) -- model inventory, Block R scored-sample inventory,
training-set membership are untouched by this rebuild (none of that
depends on raw completions). Only RAW_COMPLETIONS is patched (module-
global override, since build_universe_and_probe() reads that global
directly rather than taking it as a parameter) to point the two now-
resolved gaps at their new files; build_completions_inventory() and
build_universe_and_probe() are then re-run against the patched global.

IMPORTANT, surfaced here rather than silently absorbed: the other four
raw-completions gaps (corpus_sft/astral, gdpo_v3_ckpt300/astral,
phase12_ckpt300/astral, e1_ckpt300/astral) are NOT resolved by Block A.
Block A's three remaining ASTRAL legs (corpus_sft, gdpo300, and
phase12_beta0_ckpt300 was already done before this was noticed) reused
research/astral_model_generations.py, which only ever persists the
structured summary (precursors/max_T/reward/match) -- it never saves
completion text, by design (route_summary() discards it after parsing).
So Block A's ~39h GPU job resolved BOTH held-out gaps but ZERO of the
four ASTRAL gaps: those four (model, ASTRAL) combinations still cannot
serve Block A's real teacher-forcing marginal estimator (which needs
K=8 actual sampled <think> traces as tau_k), only the answer-only
(empty-think) estimator. This changes the real Block A instrument's
coverage and is reported in inventory_v2.json's raw_completions_gaps
and a new top-level block_a_marginal_estimator_coverage_gap field,
rather than being quietly worked around.

Outputs (new files, Block 0's originals untouched):
  results/phase17/inventory_v2.json
  results/phase17/universe_v2.jsonl
  results/phase17/probe_targets_v2.json

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_block0_inventory_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

import research.phase17_block0_inventory as b0  # noqa: E402

PREREG_COMMIT = b0.PREREG_COMMIT
OUT_DIR = REPO_ROOT / "results" / "phase17"

NEWLY_RESOLVED = {
    ("corpus_sft", "heldout"): REPO_ROOT / "data/novelty/corpus_sft_heldout_n32.jsonl",
    ("gdpo_v3_ckpt300", "heldout"): REPO_ROOT / "data/novelty/gdpo300_heldout_n32.jsonl",
}

STILL_GAPPED_ASTRAL = [
    "corpus_sft/astral", "gdpo_v3_ckpt300/astral", "phase12_ckpt300/astral", "e1_ckpt300/astral",
]


def main():
    for (model, target_set), path in NEWLY_RESOLVED.items():
        assert path.exists(), f"expected {path} to exist (Block A should have produced it)"
        b0.RAW_COMPLETIONS[model][target_set] = path
        print(f"patched RAW_COMPLETIONS[{model}][{target_set}] -> {path}", flush=True)

    model_inv = b0.build_model_inventory()
    completions_inv, completions_gaps = b0.build_completions_inventory()
    reward_files_inv, block_r_gaps = b0.build_block_r_inventory()
    training_membership = b0.build_training_membership()
    universe_rows, probe_doc = b0.build_universe_and_probe(completions_inv)

    inventory = {
        "prereg_commit": PREREG_COMMIT,
        "seed": b0.SEED,
        "rebuild_note": "v2 of Block 0's inventory/universe/probe set, after Block A's GPU "
                        "regeneration job (data_curation/phase17_block_a_heldout_sampling.py + "
                        "scripts/run_phase17_block_a_regen.sh) resolved the corpus_sft/heldout and "
                        "gdpo_v3_ckpt300/heldout raw-completions gaps. Block 0's original "
                        "inventory.json/universe.jsonl/probe_targets.json are untouched.",
        "block_a_marginal_estimator_coverage_gap": {
            "statement": "Block A's forward-pass marginal estimator needs K=8 real sampled "
                         "<think> traces per (model, target) as tau_k. These four (model, ASTRAL) "
                         "combinations still have NO raw trace file after Block A's regeneration "
                         "-- only the pre-existing structured-only fallback (precursors/max_T/"
                         "reward/match, no reasoning text) -- because astral_model_generations.py "
                         "(the script Block A's ASTRAL legs used) never persists completion text.",
            "still_gapped": STILL_GAPPED_ASTRAL,
            "resolved_by_block_a": [f"{m}/{s}" for (m, s) in NEWLY_RESOLVED],
            "consequence": "the marginal estimator cannot be computed for these 4 models x 35 "
                           "ASTRAL targets = 140 (model,target) cells; only the answer-only "
                           "(empty-think) estimator can be. Gate G1's own rule (median Spearman "
                           ">=0.5 under the MARGINAL estimator, every model) has no marginal-"
                           "estimator number to check for these 4 models on ASTRAL at all -- this "
                           "is a coverage gap, not a predicted gate failure, and needs a decision "
                           "(skip ASTRAL for these 4 under G1 and gate on held-out only, or spend "
                           "a 7th GPU leg re-generating ASTRAL samples with completion text saved) "
                           "before Block A's real forward-pass job is scoped.",
        },
        "models": model_inv,
        "raw_completions": completions_inv,
        "raw_completions_gaps": completions_gaps,
        "block_r_scored_sample_files": reward_files_inv,
        "block_r_gaps": block_r_gaps,
        "training_set_membership": training_membership,
        "universe_n_target_recipe_rows": len(universe_rows),
        "probe_targets": probe_doc,
    }

    (OUT_DIR / "inventory_v2.json").write_text(json.dumps(inventory, indent=1))
    with (OUT_DIR / "universe_v2.jsonl").open("w") as f:
        for row in universe_rows:
            f.write(json.dumps(row) + "\n")
    (OUT_DIR / "probe_targets_v2.json").write_text(json.dumps({**probe_doc, "prereg_commit": PREREG_COMMIT}, indent=1))

    print("\n=== Block 0 v2 rebuild complete ===")
    print(f"RAW-COMPLETIONS GAPS remaining ({len(completions_gaps)}):")
    for g in completions_gaps:
        print(f"  - {g}")
    print(f"\nUniverse rows: {len(universe_rows)} (was {sum(1 for _ in open(OUT_DIR / 'universe.jsonl'))} in v1)")
    print(f"\nOutputs written to {OUT_DIR}/ (inventory_v2.json, universe_v2.jsonl, probe_targets_v2.json)")


if __name__ == "__main__":
    main()
