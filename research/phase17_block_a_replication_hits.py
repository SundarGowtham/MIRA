#!/usr/bin/env python
"""
research/phase17_block_a_replication_hits.py — Phase 17 Block A, item 1:
replication hit counts for the four models Block A regenerated, using the
IDENTICAL matching code that produced the headline ASTRAL numbers (findings
16/18/22/30) -- scripts/build_claims_ledger.py's hit()/conv() functions,
reimplemented verbatim here (that file is a script, not a module, so it is
re-read rather than imported), not re-derived.

hit(m, t, superset): "PREDICTED" in the target's 32 sample match-tags
(exact rule-picked set), or also "PREDICTED_SUPERSET" if superset=True.
clean 22: results/phase16/astral_clean_subset.json's 35 targets minus its
13 "overlapping_targets" (the same clean-22 definition finding 16/22/30 use).

Original (headline) files vs Block A's new replication files, same 35
targets, same schema (classify() in research/astral_model_generations.py,
unedited, used by both):
  Phase12@300 (orig results/astral_gen_n32_gdpo_phase12_beta0.json, 14/35)
    vs results/phase17/block_a_astral_gen_n32_phase12_beta0_ckpt300_replication.json
  E1@300      (orig results/astral_gen_n32_e1.json, 14/35)
    vs results/phase17/block_a_astral_gen_n32_e1_replication.json
  GDPO-300    (orig results/astral_gen_n32_gdpo300.json, 3/35)
    vs results/phase17/block_a_astral_gen_n32_gdpo300_replication.json
  Corpus SFT  (orig results/astral_gen_n32_sft.json, 0/35 exact, 1/35 superset)
    vs results/phase17/block_a_astral_gen_n32_corpus_sft_replication.json

Exact McNemar (original vs replication, paired on the 35 targets, exact
match only) reported alongside -- this is a REPLICATION check (same model,
same sampling procedure, independent seed/run), not a model-vs-model
comparison; a significant p here would mean the sampling pipeline itself
is not stable, not that either number is wrong.

Whatever comes out is reported as-is; no headline number (14/35, 14/35,
3/35, 0/35) is replaced by this script.

NO EDIT to research/astral_model_generations.py or scripts/build_claims_ledger.py.

Usage (CPU only, seconds -- no tmux needed, but run inside one anyway per
project convention):
  PYTHONPATH=. uv run python research/phase17_block_a_replication_hits.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
R = REPO_ROOT / "results"
OUT_PATH = R / "phase17" / "block_a_replication_hits.json"

MODELS = [
    {"name": "Phase12@300", "orig_hits_stated": 14,
     "orig_file": "astral_gen_n32_gdpo_phase12_beta0.json",
     "repl_file": "phase17/block_a_astral_gen_n32_phase12_beta0_ckpt300_replication.json"},
    {"name": "E1@300", "orig_hits_stated": 14,
     "orig_file": "astral_gen_n32_e1.json",
     "repl_file": "phase17/block_a_astral_gen_n32_e1_replication.json"},
    {"name": "GDPO-300", "orig_hits_stated": 3,
     "orig_file": "astral_gen_n32_gdpo300.json",
     "repl_file": "phase17/block_a_astral_gen_n32_gdpo300_replication.json"},
    {"name": "Corpus SFT", "orig_hits_stated": "0 (1 superset)",
     "orig_file": "astral_gen_n32_sft.json",
     "repl_file": "phase17/block_a_astral_gen_n32_corpus_sft_replication.json"},
]


def load(rel):
    return json.loads((R / rel).read_text())


def hit(gen_by_target: dict, t: str, superset: bool = False) -> bool:
    ms = [s.get("match") for s in gen_by_target[t]["samples"]]
    return ("PREDICTED" in ms) or (superset and "PREDICTED_SUPERSET" in ms)


def mcnemar_exact(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_two_sided": 1.0, "p_one_sided": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_two_sided": round(min(1.0, 2 * tail), 4), "p_one_sided": round(tail, 4)}


def main():
    clean_subset = load("phase16/astral_clean_subset.json")
    overlap = set(clean_subset["overlapping_targets"])

    report = {"prereg_commit": PREREG_COMMIT, "block": "A", "subproperty": "replication hit counts",
               "note": "hit()/conv() logic is scripts/build_claims_ledger.py's, reimplemented verbatim "
                       "(not imported -- that file is a script with module-level side effects, not a "
                       "library). clean-22 = the 35 ASTRAL targets minus results/phase16/astral_clean_"
                       "subset.json's 13 overlapping_targets, the same definition used for every clean22 "
                       "number in findings 16/18/22/30.",
               "models": {}}

    rows = []
    for spec in MODELS:
        orig = {r["target"]: r for r in load(spec["orig_file"])["results"]}
        repl = {r["target"]: r for r in load(spec["repl_file"])["results"]}
        targets = sorted(set(orig) & set(repl))
        if set(orig) != set(repl):
            missing = set(orig) ^ set(repl)
            report.setdefault("warnings", []).append(
                f"{spec['name']}: orig/repl target sets differ, symmetric diff = {sorted(missing)}")

        orig_exact = sum(hit(orig, t) for t in targets)
        orig_super = sum(hit(orig, t, True) for t in targets)
        orig_clean = sum(hit(orig, t) for t in targets if t not in overlap)
        repl_exact = sum(hit(repl, t) for t in targets)
        repl_super = sum(hit(repl, t, True) for t in targets)
        repl_clean = sum(hit(repl, t) for t in targets if t not in overlap)

        only_orig = sum(hit(orig, t) and not hit(repl, t) for t in targets)
        only_repl = sum(hit(repl, t) and not hit(orig, t) for t in targets)
        mc = mcnemar_exact(only_orig, only_repl)

        model_report = {
            "n_targets_compared": len(targets),
            "n_clean22_in_compared": sum(1 for t in targets if t not in overlap),
            "original_hits_exact": orig_exact,
            "original_hits_superset": orig_super,
            "original_hits_clean22": orig_clean,
            "original_hits_stated_in_claude_md": spec["orig_hits_stated"],
            "replication_hits_exact": repl_exact,
            "replication_hits_superset": repl_super,
            "replication_hits_clean22": repl_clean,
            "mcnemar_original_vs_replication_exact": {
                "original_only": only_orig, "replication_only": only_repl,
                "p_two_sided": mc["p_two_sided"], "p_one_sided": mc["p_one_sided"]},
        }
        report["models"][spec["name"]] = model_report
        rows.append({"model": spec["name"], "orig": f"{orig_exact}/{len(targets)}",
                     "repl": f"{repl_exact}/{len(targets)}", "repl_superset": f"{repl_super}/{len(targets)}",
                     "repl_clean22": f"{repl_clean}/{model_report['n_clean22_in_compared']}",
                     "mcnemar_p_two_sided": mc["p_two_sided"]})
        print(f"{spec['name']}: orig={orig_exact}/{len(targets)} (stated {spec['orig_hits_stated']}), "
              f"repl={repl_exact}/{len(targets)}, repl_superset={repl_super}/{len(targets)}, "
              f"repl_clean22={repl_clean}/{model_report['n_clean22_in_compared']}, "
              f"mcnemar p={mc['p_two_sided']} (orig_only={only_orig}, repl_only={only_repl})", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
