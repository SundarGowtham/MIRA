#!/usr/bin/env python
"""
research/phase18_block0_inventory.py — Phase 18 Block 0 (inventory, CPU).
docs/phases/PHASE18_PREREG.md, "Block 0".

Records, for Base and every model in S1-S7: raw-completion files per
target set with samples per target (rows and parsed recipes), adapter
paths, intermediate checkpoints, LoRA rank/alpha. Also: Phase 17 outputs,
bf16 base weights in the local HF cache, ASTRAL lab outcomes, ARROWS3
YBCO data (via the frozen arrows_gate_scoring.load_target_entries), judge
versions, and a per-block input-readiness check so later blocks skip
(rather than improvise) if an input is missing.

Also builds results/phase18/recipes_by_model.json (canonical recipes per
model/target-set/target, precset_of canonicaliser), reused by 18A/18E.

ASTRAL success definition. Phase 16's "hit" is an exact match of the
canonical recipe to ASTRAL's rule-picked (predicted) set
(research/phase16_astral_clean_subset.py::load_hits, match == PREDICTED).
That is a set-identity definition, not a purity threshold. The robot data
additionally gives best measured phase purity for the traditional and the
predicted set; ASTRAL's own scale calls >0.5 "high". Both are recorded.
Blocks that need a binary lab outcome PER RECIPE (18B-ASTRAL, 18E, 18G)
use purity > 0.5 (lab_success_sets); 18E additionally reports the Phase-16
singleton V(x) = {predicted} as a sensitivity row. Decided here, before
any Phase 18 result was computed.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase18_block0_inventory.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

from research.phase18_common import (  # noqa: E402
    ADAPTERS, MODELS, SOURCES, STEPS, OUT_DIR, provenance, write_json, notify,
    load_recipes, target_sets, astral_lab,
)

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def lora_info(path: Path) -> dict | None:
    cfg = path / "adapter_config.json"
    if not cfg.exists():
        return None
    c = json.loads(cfg.read_text())
    return {"r": c.get("r"), "lora_alpha": c.get("lora_alpha"),
            "target_modules": sorted(c.get("target_modules") or []),
            "base_model_name_or_path": c.get("base_model_name_or_path")}


def model_inventory() -> dict:
    out = {"base": {"adapter_path": None, "weights": "Qwen/Qwen3-8B (no adapter)"}}
    for m, rel in ADAPTERS.items():
        p = REPO_ROOT / rel
        run_dir = p.parent
        ckpts = sorted((d.name for d in run_dir.glob("checkpoint-*") if d.is_dir()),
                       key=lambda s: int(s.split("-")[1]))
        has_weights = any((p / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin"))
        out[m] = {"adapter_path": rel, "adapter_weights_exist": has_weights,
                  "lora": lora_info(p), "run_dir_checkpoints": ckpts,
                  "run_dir_has_final": (run_dir / "final").exists()}
    return out


def completions_inventory(cache: dict) -> dict:
    out = {}
    for (model, ts), srcs in SOURCES.items():
        e = cache.get(model, {}).get(ts, {})
        if "missing" in e:
            out.setdefault(model, {})[ts] = {"missing": e["missing"]}
            continue
        parsed = {t: len(v) for t, v in e["samples"].items()}
        groups = {}
        for t, g in e.get("by_group", {}).items():
            for gname, v in g.items():
                groups.setdefault(gname, []).append(len(v))
        out.setdefault(model, {})[ts] = {
            "file": e["source"], "kind": e["kind"],
            "n_targets": len(e["n_rows"]),
            "rows_per_target": {"min": min(e["n_rows"].values()), "median": statistics.median(e["n_rows"].values()),
                                "max": max(e["n_rows"].values())},
            "parsed_recipes_per_target": {"min": min(parsed.values()) if parsed else 0,
                                          "median": statistics.median(parsed.values()) if parsed else 0,
                                          "max": max(parsed.values()) if parsed else 0},
            "parsed_per_sample_group_median": {g: statistics.median(v) for g, v in groups.items()},
            "n_rows_total": sum(e["n_rows"].values()), "n_parsed_total": sum(parsed.values()),
        }
    return out


def phase17_outputs() -> dict:
    files = sorted(p.name for p in (REPO_ROOT / "results" / "phase17").iterdir())
    return {
        "files": files,
        "blocks_ran": ["0", "A", "B (B1-B4)", "C (infeasible as specified: 6/385 converged)", "G (G1)", "R (R-a..R-h)"],
        "blocks_not_run": ["D", "E", "F", "H1 (weight geometry)"],
        "source": "CLAUDE.md 'PHASE 17 (CLOSED)' + docs/phases/PHASE17_RESULTS.md",
    }


def base_weights() -> dict:
    snap_root = HF_HOME / "hub" / "models--Qwen--Qwen3-8B" / "snapshots"
    snaps = sorted(snap_root.glob("*")) if snap_root.exists() else []
    if not snaps:
        return {"present": False, "searched": str(snap_root)}
    s = snaps[-1]
    cfg = json.loads((s / "config.json").read_text())
    shards = sorted(p.name for p in s.glob("*.safetensors"))
    size = sum(p.resolve().stat().st_size for p in s.glob("*.safetensors"))
    return {"present": bool(shards), "snapshot": str(s), "torch_dtype": cfg.get("torch_dtype"),
            "n_shards": len(shards), "total_bytes": size,
            "is_bf16_full_precision": cfg.get("torch_dtype") == "bfloat16" and size > 15e9}


def arrows_ybco() -> dict:
    from research.distributional.arrows_gate_scoring import load_target_entries
    entries, target_formula = load_target_entries("YBCO")
    recipes = sorted({e["precursor_set"] for e in entries})
    temps = sorted({e["temp_C"] for e in entries})
    return {"target": target_formula, "n_entries": len(entries), "n_recipes": len(recipes),
            "temperatures_C": temps,
            "wt_pct_range": [min(e["wt_pct"] for e in entries), max(e["wt_pct"] for e in entries)],
            "loader": "research/distributional/arrows_gate_scoring.py::load_target_entries('YBCO')"}


def judge_versions() -> dict:
    import validator
    from core import comparator, ranker
    return {"VALIDATOR_VERSION": validator.VALIDATOR_VERSION,
            "COMPARATOR_VERSION": comparator.COMPARATOR_VERSION,
            "RANKER_VERSION": ranker.RANKER_VERSION,
            "ranker_v2": "core/ranker.py::Ranker(formula_set, thermo, freq, validator_version=2) "
                         "(the construction Phase 17 R-f/R-b used)",
            "comparator_scales": "misc/comparator_scales_v2.json "
                                 "(snapshot: results/phase17/comparator_scales_v2_snapshot.json)"}


def readiness(models: dict, comps: dict) -> dict:
    def has(m, ts):
        return "missing" not in comps.get(m, {}).get(ts, {"missing": 1})
    r = {}
    r["18A"] = {"needs": "recipes for both models of S1-S7 on ASTRAL+held-out",
                "ok": all(has(a, ts) and has(b, ts) for a, b in STEPS.values() for ts in ("astral", "heldout"))}
    r["18B"] = {"needs": "ARROWS3 YBCO + comparator", "ok": (REPO_ROOT / "misc/comparator_scales_v2.json").exists()}
    need_c = ["corpus_sft", "rs_sft", "e3", "e3c", "gdpo_v3_ckpt300", "phase12_ckpt100", "phase12_ckpt200",
              "phase12_ckpt300", "e1_ckpt100", "e1_ckpt200", "e1_ckpt300"]
    r["18C"] = {"needs": "adapters for S1-S7 incl. practice inits + ckpt100/200/300 for S6/S7 + bf16 base",
                "ok": all(models[m]["adapter_weights_exist"] for m in need_c),
                "missing": [m for m in need_c if not models[m]["adapter_weights_exist"]]}
    r["18D"] = {"needs": "R-a 200-route suite source (rssft_reference_astral.jsonl) + judges",
                "ok": (REPO_ROOT / "data/novelty/rssft_reference_astral.jsonl").exists()}
    e_models = ["base", "corpus_sft", "rs_sft", "e3", "e3c", "gdpo_v3_ckpt300", "phase12_ckpt300", "e1_ckpt300"]
    r["18E"] = {"needs": "ASTRAL recipes for 8 models + lab outcomes", "ok": all(has(m, "astral") for m in e_models)}
    r["18F"] = {"needs": "ARROWS3 pairs (Phase 17 R-f setup) + validator channels", "ok": True}
    r["18G"] = {"needs": "existing scored samples (rescored per channel; no new sampling)", "ok": True}
    r["18H"] = {"needs": "ARROWS3 pairs + R2/R4/R5 margins", "ok": True}
    return r


def main():
    prov = provenance()
    notify("MIRA Phase 18 Block 0 (inventory) started")
    print("building recipe cache (parse every existing completion)...", flush=True)
    cache = load_recipes()
    models = model_inventory()
    comps = completions_inventory(cache)
    sets = target_sets()
    lab = astral_lab()
    doc = {
        **prov, "block": "0",
        "step_table": {s: {"before": a, "after": b} for s, (a, b) in STEPS.items()},
        "target_sets": {k: {"n": len(v), "targets": v} for k, v in sets.items()},
        "models": models,
        "raw_completions": comps,
        "intermediate_checkpoints_note": (
            "Fine-tuning runs (S1-S4) each have exactly one intermediate checkpoint dir "
            "(corpus_sft checkpoint-117, rs_sft checkpoint-27, e3/e3c checkpoint-48); see "
            "run_dir_checkpoints. Practice runs have checkpoint-100/200/300."),
        "phase17_outputs": phase17_outputs(),
        "base_weights": base_weights(),
        "astral_lab": {
            "success_definition": {
                "phase16_hit": "exact canonical match to ASTRAL's rule-picked (predicted) set "
                               "(research/phase16_astral_clean_subset.py::load_hits)",
                "per_recipe_lab_success": "best measured phase purity > 0.5 (ASTRAL's 'high' band)",
                "used_by": {"18B_ASTRAL[E]": "per_recipe_lab_success", "18E": "per_recipe_lab_success "
                            "primary; phase16_hit singleton as sensitivity", "18G": "per_recipe_lab_success"},
            },
            "n_targets": len(lab),
            "n_targets_with_a_success_set": sum(1 for v in lab.values() if v["lab_success_sets"]),
            "n_targets_with_a_failed_set": sum(1 for v in lab.values() if v["lab_fail_sets"]),
            "per_target": lab,
        },
        "arrows_ybco": arrows_ybco(),
        "judge_versions": judge_versions(),
        "block_readiness": None,
    }
    doc["block_readiness"] = readiness(models, comps)
    path = write_json("inventory.json", doc)
    print(json.dumps({k: doc[k] for k in ("base_weights", "judge_versions", "block_readiness")}, indent=1))
    print(f"-> {path}")
    notify("MIRA Phase 18 Block 0 done: " + ", ".join(f"{k}={'ok' if v['ok'] else 'MISSING'}"
                                                       for k, v in doc["block_readiness"].items()))


if __name__ == "__main__":
    main()
