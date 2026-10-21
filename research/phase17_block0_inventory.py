#!/usr/bin/env python
"""
research/phase17_block0_inventory.py — Phase 17 Block 0 (inventory, CPU
only). Per docs/phases/PHASE17_PREREG.md's Block 0 spec:

  1. For every model in the step table (S1-S7's before/after), record
     adapter path/storage/checkpoints/LoRA config.
  2. For every model and target set, find the files holding raw
     completions with <think> traces and per-sample checker/ranker
     scores.
  3. Record training-set membership of every ASTRAL (35) and held-out
     (40) target, per step.
  4. Build U(x) (the recipe universe) and the fixed probe set (20
     ASTRAL from the clean 22 + 20 held-out, seed 20261016).

Also inventories the six scored-sample files Block R needs (R1-R6),
per instruction, and reports any gap before Block A starts.

No training, no generation, no git writes. New files only:
  results/phase17/inventory.json
  results/phase17/universe.jsonl
  results/phase17/probe_targets.json

Usage (CPU, tmux per standing rule even though this is quick):
  PYTHONPATH=. uv run python research/phase17_block0_inventory.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

from validator import SynthesisValidator  # noqa: E402
from core.reward import parse_completion  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
SEED = 20261016

OUT_DIR = REPO_ROOT / "results" / "phase17"
OUT_DIR.mkdir(exist_ok=True)


def precset_of(route) -> tuple:
    return tuple(sorted({SynthesisValidator._normalize_formula(p.formula) for p in route.precursors}))


# ---------------------------------------------------------------------------
# 1. Models in the step table: adapter path / storage / checkpoints / LoRA
# ---------------------------------------------------------------------------

def read_adapter_config(adapter_dir: Path) -> dict | None:
    f = adapter_dir / "adapter_config.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text())
    return {
        "r": c.get("r"),
        "lora_alpha": c.get("lora_alpha"),
        "target_modules": sorted(c.get("target_modules", [])),
        "base_model_name_or_path": c.get("base_model_name_or_path"),
    }


def checkpoints_of(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    return sorted(
        (p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda s: int(s.split("-")[1]),
    )


MODELS = {
    "base": {
        "step_role": "S1/S2/S3/S4 before",
        "adapter_path": None,
        "storage": "no adapter (Qwen/Qwen3-8B base weights)",
        "continued_from": None,
    },
    "corpus_sft": {
        "step_role": "S1 after, S5 before",
        "adapter_path": "runs/sft-qlora-sft-v3-2nd-rank16/final",
        "storage": "LoRA adapter, fine-tuned fresh from base",
        "continued_from": "base",
    },
    "rs_sft": {
        "step_role": "S2 after, S6/S7 before",
        "adapter_path": "runs/sft-qlora-rs-sft-from-base/final",
        "storage": "LoRA adapter, fine-tuned fresh from base",
        "continued_from": "base",
    },
    "e3": {
        "step_role": "S3 after",
        "adapter_path": "runs/sft-qlora-e3-round1/final",
        "storage": "LoRA adapter, fine-tuned fresh from base",
        "continued_from": "base",
    },
    "e3c": {
        "step_role": "S4 after",
        "adapter_path": "runs/sft-qlora-e3c-round1/final",
        "storage": "LoRA adapter, fine-tuned fresh from base",
        "continued_from": "base",
    },
    "gdpo_v3_ckpt300": {
        "step_role": "S5 after (Corpus SFT -> GDPO-300, checker reward)",
        "adapter_path": "runs/gdpo-qlora-gdpo-v3/checkpoint-300",
        "storage": "LoRA adapter, CONTINUES corpus_sft's adapter (core/model.py::load_with_adapter, is_trainable=True) -- weight change is delta-W = B1A1 - B0A0, not stacked",
        "continued_from": "corpus_sft (runs/sft-qlora-sft-v3-2nd-rank16/final, confirmed via scripts/run_gdpo.sh:SFT_INIT)",
    },
    "phase12_ckpt100": {
        "step_role": "S6 @100 (RS-SFT -> Phase 12, checker reward, beta=0)",
        "adapter_path": "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-100",
        "storage": "LoRA adapter, continues rs_sft's adapter",
        "continued_from": "rs_sft (runs/sft-qlora-rs-sft-from-base/final, confirmed via scripts/run_gdpo_phase12_beta0.sh:RS_SFT_INIT)",
    },
    "phase12_ckpt200": {
        "step_role": "S6 @200",
        "adapter_path": "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-200",
        "storage": "LoRA adapter, continues rs_sft's adapter",
        "continued_from": "rs_sft",
    },
    "phase12_ckpt300": {
        "step_role": "S6 @300 (pre-registered stopping point, finding 22)",
        "adapter_path": "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300",
        "storage": "LoRA adapter, continues rs_sft's adapter",
        "continued_from": "rs_sft",
    },
    "e1_ckpt100": {
        "step_role": "S7 @100 (RS-SFT -> E1, ranker v2 reward, beta=0)",
        "adapter_path": "runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-100",
        "storage": "LoRA adapter, continues rs_sft's adapter",
        "continued_from": "rs_sft (confirmed via scripts/run_gdpo_e1.sh:RS_SFT_INIT)",
    },
    "e1_ckpt200": {
        "step_role": "S7 @200",
        "adapter_path": "runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-200",
        "storage": "LoRA adapter, continues rs_sft's adapter",
        "continued_from": "rs_sft",
    },
    "e1_ckpt300": {
        "step_role": "S7 @300",
        "adapter_path": "runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-300",
        "storage": "LoRA adapter, continues rs_sft's adapter",
        "continued_from": "rs_sft",
    },
}


def build_model_inventory() -> dict:
    out = {}
    for name, meta in MODELS.items():
        entry = dict(meta)
        if meta["adapter_path"] is None:
            entry["lora_config"] = None
            entry["adapter_exists"] = True  # base has no adapter to check
            entry["intermediate_checkpoints"] = []
        else:
            adapter_dir = REPO_ROOT / meta["adapter_path"]
            entry["lora_config"] = read_adapter_config(adapter_dir)
            entry["adapter_exists"] = adapter_dir.exists() and entry["lora_config"] is not None
            # run_dir = parent of "final" or "checkpoint-N"
            run_dir = adapter_dir.parent
            entry["intermediate_checkpoints"] = checkpoints_of(run_dir)
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# 2. Raw-completion files per model / target set
# ---------------------------------------------------------------------------

def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    n = 0
    with path.open() as f:
        for _ in f:
            n += 1
    return n


def has_think_trace(path: Path) -> bool | None:
    if not path.exists():
        return None
    with path.open() as f:
        line = f.readline()
    try:
        rec = json.loads(line)
    except Exception:
        return None
    return "<think>" in rec.get("completion", "")


def enriched_sample_schema(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    results = doc.get("results", [])
    if not results or not results[0].get("samples"):
        return []
    return sorted(results[0]["samples"][0].keys())


# model -> target_set -> raw-completions file (None if missing)
RAW_COMPLETIONS = {
    "base": {
        "astral": REPO_ROOT / "data/novelty/base_reference_astral.jsonl",
        "heldout": REPO_ROOT / "data/novelty/base_reference_heldout.jsonl",
    },
    "corpus_sft": {
        "astral": None,   # GAP: only results/astral_gen_n32_sft.json exists (structured, no <think>)
        "heldout": None,  # GAP: no file found at all
    },
    "rs_sft": {
        "astral": REPO_ROOT / "data/novelty/rssft_reference_astral.jsonl",
        "heldout": REPO_ROOT / "data/novelty/rssft_reference_heldout.jsonl",
    },
    "e3": {
        "astral": REPO_ROOT / "data/novelty/e3_astral_n32.jsonl",
        "heldout": REPO_ROOT / "data/novelty/e3_heldout_n32.jsonl",
    },
    "e3c": {
        "astral": REPO_ROOT / "data/novelty/e3c_astral_n32.jsonl",
        "heldout": REPO_ROOT / "data/novelty/e3c_heldout_n32.jsonl",
    },
    "gdpo_v3_ckpt300": {
        "astral": None,   # GAP: only results/astral_gen_n32_gdpo300.json exists (structured, no <think>)
        "heldout": None,  # GAP: no file found at all
    },
    "phase12_ckpt100": {
        "astral": REPO_ROOT / "data/novelty/phase12_ckpt100_astral_n32.jsonl",
        "heldout": REPO_ROOT / "data/novelty/phase12_ckpt100_heldout_n32.jsonl",
    },
    "phase12_ckpt200": {
        "astral": REPO_ROOT / "data/novelty/phase12_ckpt200_astral_n32.jsonl",
        "heldout": REPO_ROOT / "data/novelty/phase12_ckpt200_heldout_n32.jsonl",
    },
    "phase12_ckpt300": {
        "astral": None,  # GAP: results/astral_gen_n32_gdpo_phase12_beta0.json is structured-only
        "heldout": REPO_ROOT / "data/novelty/phase12_ckpt300_heldout_n32.jsonl",
    },
    "e1_ckpt100": {
        "astral": REPO_ROOT / "data/novelty/e1_ckpt100_astral_n32.jsonl",
        "heldout": REPO_ROOT / "data/novelty/e1_ckpt100_heldout_n32.jsonl",
    },
    "e1_ckpt200": {
        "astral": REPO_ROOT / "data/novelty/e1_ckpt200_astral_n32.jsonl",
        "heldout": REPO_ROOT / "data/novelty/e1_ckpt200_heldout_n32.jsonl",
    },
    "e1_ckpt300": {
        "astral": None,  # GAP: results/astral_gen_n32_e1.json is structured-only
        "heldout": REPO_ROOT / "data/novelty/e1_heldout_n32.jsonl",
    },
}

# Structured-but-not-raw fallback files (precursors/max_T/reward/match only,
# no <think> trace) -- these exist and are useful for Blocks B-C's U(x) but
# CANNOT serve Block A's teacher-forcing estimator.
STRUCTURED_ONLY_FALLBACK = {
    ("corpus_sft", "astral"): "results/astral_gen_n32_sft.json",
    ("gdpo_v3_ckpt300", "astral"): "results/astral_gen_n32_gdpo300.json",
    ("phase12_ckpt300", "astral"): "results/astral_gen_n32_gdpo_phase12_beta0.json",
    ("e1_ckpt300", "astral"): "results/astral_gen_n32_e1.json",
}


def build_completions_inventory() -> tuple[dict, list[str]]:
    out = {}
    gaps = []
    for model, by_set in RAW_COMPLETIONS.items():
        out[model] = {}
        for target_set, path in by_set.items():
            if path is None:
                fallback = STRUCTURED_ONLY_FALLBACK.get((model, target_set))
                fb_schema = enriched_sample_schema(REPO_ROOT / fallback) if fallback else None
                out[model][target_set] = {
                    "raw_completions_file": None,
                    "n_rows": None,
                    "has_think_trace": False,
                    "structured_only_fallback": fallback,
                    "structured_only_schema": fb_schema,
                }
                gaps.append(
                    f"{model}/{target_set}: NO raw <think>-trace completions"
                    + (f" (structured-only fallback exists: {fallback}, schema {fb_schema} -- no reasoning text)" if fallback else " (no file at all)")
                )
            else:
                out[model][target_set] = {
                    "raw_completions_file": str(path.relative_to(REPO_ROOT)),
                    "n_rows": count_jsonl(path),
                    "has_think_trace": has_think_trace(path),
                    "structured_only_fallback": None,
                    "structured_only_schema": None,
                }
    return out, gaps


# ---------------------------------------------------------------------------
# Block R: scored-sample files needed for R1-R6
# ---------------------------------------------------------------------------

REWARD_FILES = {
    "R1": {
        "reward": "Checker, 10 channels (run 1)",
        "policy": "Corpus SFT",
        "file": "results/hardening.json",
        "filter": "condition == 'baseline'",
        "note": "40 targets x 8 samples = 320 rows at condition=='baseline'; 10-channel breakdown present.",
    },
    "R2": {
        "reward": "Checker, 5-channel RUN3 aggregate",
        "policy": "RS-SFT",
        "file": None,
        "filter": None,
        "note": (
            "GAP (the one regular Claude flagged as not like-for-like with R4). "
            "Raw RS-SFT samples exist (data/novelty/rssft_reference_{astral,heldout}.jsonl, "
            "sample_group=='resampling_rate', same 2400-row/300-group scope as R4/results/ranker_capacity.json) "
            "but have NEVER been scored with the 5-channel checker aggregate -- no breakdown field in "
            "the raw jsonl. This is the computation R-a/R-b needs to run first."
        ),
    },
    "R3": {
        "reward": "Low-temperature objective (temperature_economy added)",
        "policy": "Corpus SFT (hardening-probe samples)",
        "file": "misc/hardening_low_temp2.json",
        "filter": "condition in {'baseline','low_temp'}",
        "note": (
            "misc/ is gitignored scratch, not git-tracked, but present on disk. "
            "640 rows at baseline+low_temp (40 targets x 8 samples x 2 conditions) carry temperature_economy; "
            "superset of misc/hardening_low_temp.json (same two conditions, earlier version) and adds "
            "low_temp_ceiling/low_temp_inventory (1280 rows total, 4 conditions)."
        ),
    },
    "R4": {
        "reward": "Ranker v2",
        "policy": "RS-SFT",
        "file": "results/ranker_capacity.json",
        "filter": "n_records=2400, n_groups=300 (both astral+heldout, sample_group=='resampling_rate')",
        "note": "Aggregated capacity file, not a per-sample jsonl -- per-sample breakdown would need rerunning research/ranker_capacity_rssft.py with raw output kept, or reading its source if it persisted one.",
    },
    "R5": {
        "reward": "Comparator (core/comparator.py)",
        "policy": "RS-SFT",
        "file": None,
        "filter": None,
        "note": (
            "GAP. Only two comparator-scored artifacts exist and neither is RS-SFT's own group-of-8 samples: "
            "(a) results/phase13_astral_scoring_v2.json -- ASTRAL's 35 fixed traditional-vs-predicted ground-truth "
            "pairs, not model samples; (b) results/external/arrows_gate.json -- ARROWS3 pairs (useful for R-f/R-g "
            "directly, see below). R-b (capacity/tie-rate) and R-e (aggregation) need the comparator run pairwise "
            "within RS-SFT's existing groups of 8 (data/novelty/rssft_reference_{astral,heldout}.jsonl, "
            "resampling_rate) -- not yet computed."
        ),
    },
    "R6": {
        "reward": "Novelty reward (Phase 16)",
        "policy": "E3",
        "file": None,
        "filter": None,
        "note": (
            "GAP, but audit-only / lowest priority (E4 never ran). core/novelty_reward.py is applied live inside "
            "research/phase16_e3_e3c_*.py and never persisted per-sample; only aggregate stats survive "
            "(results/phase16/e3_e3c_bootstrap.json). E3's raw samples exist in groups-of-32 "
            "(data/novelty/e3_{astral,heldout}_n32.jsonl) and could be rescored, but no scored file exists yet."
        ),
    },
    "R-f_R-g_arrows": {
        "reward": "ARROWS3 pair data (R-f info bound, R-g Goodhart curve inputs)",
        "policy": "n/a (external robot-lab data, not a model's own samples)",
        "file": "results/external/arrows_gate.json",
        "filter": None,
        "note": "Already exists from Phase 15; covers R2/R4/R5/carbonate-free baseline comparator+validator votes on YBCO pairs. Usable as-is for R-f and R-g's per-reward comparison columns.",
    },
}


def build_block_r_inventory() -> tuple[dict, list[str]]:
    gaps = []
    for rid, meta in REWARD_FILES.items():
        if meta["file"] is None:
            gaps.append(f"{rid} ({meta['reward']} on {meta['policy']}): {meta['note']}")
    return REWARD_FILES, gaps


# ---------------------------------------------------------------------------
# 3. Training-set membership per step, for every ASTRAL(35)/held-out(40) target
# ---------------------------------------------------------------------------

def load_json(path: Path):
    return json.loads(path.read_text())


def build_training_membership() -> dict:
    astral_targets = [r["target"] for r in load_json(REPO_ROOT / "results/astral_gen_n32_rs_sft.json")["results"]]
    heldout_targets = load_json(REPO_ROOT / "data/novelty/heldout_targets.json")
    corpus_sft_training = set(load_json(REPO_ROOT / "data/novelty/training_targets.json"))

    # held-out 40 is held out from E3/E3c by construction (PHASE17_PREREG.md
    # Target sets); confirm disjointness directly from the filtered sets.
    e3_filtered_targets = {json.loads(l)["target"] for l in (REPO_ROOT / "data/novelty/e3_round1_filtered.jsonl").read_text().splitlines()}
    e3c_filtered_targets = {json.loads(l)["target"] for l in (REPO_ROOT / "data/novelty/e3c_round1_filtered.jsonl").read_text().splitlines()}

    # RS-SFT's training set = its own filtered survivors (295/400 base-generated
    # targets per finding 18), recovered exactly from the persisted train+val
    # split (data/rs_sft/rs_sft_{train,val}.jsonl) -- a distinct target pool
    # from the Kononova corpus (data/rl's 400-target sample, not
    # training_targets.json).
    rs_sft_dir = REPO_ROOT / "data" / "rs_sft"
    rs_sft_training = set()
    for fname in ("rs_sft_train.jsonl", "rs_sft_val.jsonl"):
        fp = rs_sft_dir / fname
        if fp.exists():
            for line in fp.read_text().splitlines():
                rs_sft_training.add(json.loads(line)["target"])

    membership = {}
    for step, after_model, training_set, note in [
        ("S1", "corpus_sft", corpus_sft_training, "Kononova corpus, filtered >=0.65 (data/novelty/training_targets.json, n=360)"),
        ("S2", "rs_sft", rs_sft_training, f"base's own validator-passing samples, drawn from data/rl's 400-target pool (distinct from the Kononova corpus); recovered exactly from data/rs_sft/rs_sft_{{train,val}}.jsonl, n={len(rs_sft_training)} (finding 18: 295/400 survival)"),
        ("S3", "e3", e3_filtered_targets, "E3: base's own checker-valid + rare samples (data/novelty/e3_round1_filtered.jsonl)"),
        ("S4", "e3c", e3c_filtered_targets, "E3c: base's own checker-valid samples, rarity ignored (data/novelty/e3c_round1_filtered.jsonl)"),
    ]:
        if training_set is None:
            membership[step] = {"note": note, "astral_overlap": None, "heldout_overlap": None}
            continue
        astral_overlap = sorted(set(astral_targets) & training_set)
        heldout_overlap = sorted(set(heldout_targets) & training_set)
        membership[step] = {
            "note": note,
            "n_training_targets": len(training_set),
            "astral_overlap": astral_overlap,
            "heldout_overlap": heldout_overlap,
        }

    membership["held_out_40_by_construction"] = {
        "e3_excludes_heldout": len(set(heldout_targets) & e3_filtered_targets) == 0,
        "e3c_excludes_heldout": len(set(heldout_targets) & e3c_filtered_targets) == 0,
    }
    membership["astral_targets_n"] = len(astral_targets)
    membership["heldout_targets_n"] = len(heldout_targets)
    return membership


# ---------------------------------------------------------------------------
# 4. Universe U(x) and probe set
# ---------------------------------------------------------------------------

def load_precsets_from_raw_jsonl(path: Path) -> dict[str, dict[tuple, int]]:
    """target -> {precset: count}, from a raw completions jsonl, parsed with
    core.reward.parse_completion. Skips parse failures silently (counted
    elsewhere; Block 0 does not re-litigate parse-failure rates)."""
    out: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            target, completion = rec["target"], rec["completion"]
            try:
                route = parse_completion(completion, target)
            except Exception:
                continue
            if route is None:
                continue
            try:
                ps = precset_of(route)
            except Exception:
                continue
            out[target][ps] += 1
    return out


def load_precsets_from_structured(path: Path) -> dict[str, dict[tuple, int]]:
    """target -> {precset: count}, from an astral_gen_n32-shaped structured
    file (precursors list already extracted, no parsing needed)."""
    out: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
    if not path.exists():
        return out
    doc = load_json(path)
    for r in doc["results"]:
        target = r["target"]
        for s in r["samples"]:
            precs = s.get("precursors")
            if not precs:
                continue
            ps = tuple(sorted({SynthesisValidator._normalize_formula(p) for p in precs}))
            out[target][ps] += 1
    return out


def build_universe_and_probe(completions_inv: dict) -> tuple[list[dict], dict]:
    """Build U(x) by pooling every model's existing samples (raw where
    available, structured-only fallback otherwise) per target, over both
    target sets. Returns (universe_rows, probe_targets_doc)."""
    # target -> precset -> {model: count}
    pooled: dict[str, dict[tuple, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))

    for model, by_set in RAW_COMPLETIONS.items():
        for target_set, path in by_set.items():
            if path is not None:
                counts = load_precsets_from_raw_jsonl(path)
            else:
                fallback = STRUCTURED_ONLY_FALLBACK.get((model, target_set))
                if fallback is None:
                    continue
                counts = load_precsets_from_structured(REPO_ROOT / fallback)
            for target, ps_counts in counts.items():
                for ps, n in ps_counts.items():
                    pooled[target][ps][model] = pooled[target][ps].get(model, 0) + n

    universe_rows = []
    for target, ps_map in pooled.items():
        total_models_seeing_ps = {ps: sum(m.values()) for ps, m in ps_map.items()}
        for ps, model_counts in ps_map.items():
            universe_rows.append({
                "target": target,
                "recipe": list(ps),
                "pooled_count": total_models_seeing_ps[ps],
                "per_model_sample_counts": model_counts,
            })

    # Probe set: 20 ASTRAL (from clean 22) + 20 held-out, seed 20261016.
    astral_full = [r["target"] for r in load_json(REPO_ROOT / "results/astral_gen_n32_rs_sft.json")["results"]]
    clean_overlap = set(load_json(REPO_ROOT / "results/phase16/astral_clean_subset.json")["overlapping_targets"])
    clean_22 = sorted(set(astral_full) - clean_overlap)
    heldout_40 = load_json(REPO_ROOT / "data/novelty/heldout_targets.json")

    rng = random.Random(SEED)
    probe_astral = sorted(rng.sample(clean_22, 20))
    rng2 = random.Random(SEED)
    probe_heldout = sorted(rng2.sample(heldout_40, 20))

    probe_doc = {
        "seed": SEED,
        "astral_clean_22_pool_size": len(clean_22),
        "heldout_40_pool_size": len(heldout_40),
        "probe_astral_20": probe_astral,
        "probe_heldout_20": probe_heldout,
    }
    return universe_rows, probe_doc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_inv = build_model_inventory()
    completions_inv, completions_gaps = build_completions_inventory()
    reward_files_inv, block_r_gaps = build_block_r_inventory()
    training_membership = build_training_membership()
    universe_rows, probe_doc = build_universe_and_probe(completions_inv)

    inventory = {
        "prereg_commit": PREREG_COMMIT,
        "seed": SEED,
        "models": model_inv,
        "raw_completions": completions_inv,
        "raw_completions_gaps": completions_gaps,
        "block_r_scored_sample_files": reward_files_inv,
        "block_r_gaps": block_r_gaps,
        "training_set_membership": training_membership,
        "universe_n_target_recipe_rows": len(universe_rows),
        "probe_targets": probe_doc,
    }

    (OUT_DIR / "inventory.json").write_text(json.dumps(inventory, indent=1))
    with (OUT_DIR / "universe.jsonl").open("w") as f:
        for row in universe_rows:
            f.write(json.dumps(row) + "\n")
    (OUT_DIR / "probe_targets.json").write_text(json.dumps({**probe_doc, "prereg_commit": PREREG_COMMIT}, indent=1))

    print("=== Block 0 inventory complete ===")
    print(f"prereg_commit: {PREREG_COMMIT}")
    print(f"\nRAW-COMPLETIONS GAPS ({len(completions_gaps)}):")
    for g in completions_gaps:
        print(f"  - {g}")
    print(f"\nBLOCK R SCORED-SAMPLE GAPS ({len(block_r_gaps)}):")
    for g in block_r_gaps:
        print(f"  - {g}")
    print(f"\nUniverse rows: {len(universe_rows)}")
    print(f"Probe set: {len(probe_doc['probe_astral_20'])} ASTRAL + {len(probe_doc['probe_heldout_20'])} held-out")
    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
