#!/usr/bin/env python
"""
research/phase18_common.py — shared constants and loaders for Phase 18
(docs/phases/PHASE18_PREREG.md). Every phase18_* script imports from here
so the step table, target sets, canonicaliser and provenance are identical
across blocks.

PROVENANCE. The pre-registration rule is "commit the spec unedited, then
write its commit hash into every results JSON as prereg_commit". Gowtham
commits his own work and asked (2026-11-15) for every block to run
unattended, so the spec could not be committed before computation began.
Instead every JSON carries:
  prereg_blob_sha  = `git hash-object docs/phases/PHASE18_PREREG.md`,
                     computed live at run time and asserted equal to the
                     value frozen below (so an edited spec aborts the run);
  prereg_commit    = the commit containing that exact blob if one exists
                     at run time, else None (with prereg_commit_note).
Once the spec is committed, `git log --format=%H --find-object=<blob>`
recovers the commit and proves the spec predates the results.

Canonicaliser: precset_of from research/support_growth_by_checkpoint.py
(sorted set of SynthesisValidator._normalize_formula). Structured
astral_gen_n32 files already store extracted precursor formulas; they are
passed through the same _normalize_formula, so both source kinds yield
identical canonical recipes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

PREREG_PATH = REPO_ROOT / "docs" / "phases" / "PHASE18_PREREG.md"
PREREG_BLOB_SHA_FROZEN = "5c0a84ce0161f03f19a6198ac0aa7fffe949f5ac"
SEED = 20261115
N_BOOT = 10000
OUT_DIR = REPO_ROOT / "results" / "phase18"
NTFY_TOPIC = "mira-g5x7k2-status"


def notify(msg: str) -> None:
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode(), method="POST")
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


def provenance() -> dict:
    blob = subprocess.run(["git", "hash-object", str(PREREG_PATH)], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()
    if blob != PREREG_BLOB_SHA_FROZEN:
        raise SystemExit(f"PHASE18_PREREG.md changed after freeze: {blob} != {PREREG_BLOB_SHA_FROZEN}")
    commit = subprocess.run(["git", "log", "--format=%H", "--find-object=" + blob, "--", str(PREREG_PATH)],
                            cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    commit = commit[-1] if commit else None
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "prereg_commit": commit,
        "prereg_blob_sha": blob,
        "prereg_commit_note": None if commit else (
            "PHASE18_PREREG.md was not yet committed when this ran (Gowtham asked for "
            "unattended execution; he commits his own work). prereg_blob_sha is the git "
            "content hash of the spec at run time; once committed, `git log "
            "--find-object=<blob>` recovers the commit."),
        "repo_head_at_run": head,
        "seed": SEED,
    }


def write_json(name: str, obj: dict) -> Path:
    """New files only: refuse to overwrite a git-tracked file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=REPO_ROOT,
                             capture_output=True).returncode == 0
    if tracked:
        raise SystemExit(f"refusing to overwrite tracked results file {path}")
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


# ---------------------------------------------------------------------------
# Step table and target sets (PHASE17_PREREG.md, reused unchanged)
# ---------------------------------------------------------------------------
STEPS = {
    "S1": ("base", "corpus_sft"),
    "S2": ("base", "rs_sft"),
    "S3": ("base", "e3"),
    "S4": ("base", "e3c"),
    "S5": ("corpus_sft", "gdpo_v3_ckpt300"),
    "S6": ("rs_sft", "phase12_ckpt300"),
    "S7": ("rs_sft", "e1_ckpt300"),
}
FINE_TUNING_STEPS = ("S1", "S2", "S3", "S4")
OWN_SAMPLE_FT_STEPS = ("S2", "S3", "S4")
PRACTICE_STEPS = ("S5", "S6", "S7")

ADAPTERS = {
    "corpus_sft": "runs/sft-qlora-sft-v3-2nd-rank16/final",
    "rs_sft": "runs/sft-qlora-rs-sft-from-base/final",
    "e3": "runs/sft-qlora-e3-round1/final",
    "e3c": "runs/sft-qlora-e3c-round1/final",
    "gdpo_v3_ckpt100": "runs/gdpo-qlora-gdpo-v3/checkpoint-100",
    "gdpo_v3_ckpt200": "runs/gdpo-qlora-gdpo-v3/checkpoint-200",
    "gdpo_v3_ckpt300": "runs/gdpo-qlora-gdpo-v3/checkpoint-300",
    "phase12_ckpt100": "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-100",
    "phase12_ckpt200": "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-200",
    "phase12_ckpt300": "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300",
    "e1_ckpt100": "runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-100",
    "e1_ckpt200": "runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-200",
    "e1_ckpt300": "runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-300",
}
INIT_OF = {"gdpo_v3": "corpus_sft", "phase12": "rs_sft", "e1": "rs_sft"}

HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
ASTRAL_VALIDATION_PATH = REPO_ROOT / "results" / "astral_validation_set.json"
OVERLAP_PATH = REPO_ROOT / "results" / "phase16" / "astral_overlap.json"


def target_sets() -> dict[str, list[str]]:
    astral = [t["target"] for t in json.loads(ASTRAL_VALIDATION_PATH.read_text())["targets"]]
    overlap = set(json.loads(OVERLAP_PATH.read_text())["union_of_all_overlapping_targets"])
    heldout = json.loads(HELDOUT_TARGETS_PATH.read_text())
    return {
        "astral_35": astral,
        "astral_clean_22": [t for t in astral if t not in overlap],
        "heldout_40": list(heldout),
    }


# ---------------------------------------------------------------------------
# Sample sources: (model, target_set) -> list of (kind, path, sample_group filter)
# "raw" = completions jsonl parsed with core.reward.parse_completion
# "structured" = astral_gen_n32-shaped file with extracted precursors
# ---------------------------------------------------------------------------
D = REPO_ROOT / "data" / "novelty"
R = REPO_ROOT / "results"
SOURCES = {
    ("base", "astral"): [("raw", D / "base_reference_astral.jsonl", None)],
    ("base", "heldout"): [("raw", D / "base_reference_heldout.jsonl", None)],
    ("corpus_sft", "astral"): [("structured", R / "astral_gen_n32_sft.json", None)],
    ("corpus_sft", "heldout"): [("raw", D / "corpus_sft_heldout_n32.jsonl", None)],
    ("rs_sft", "astral"): [("raw", D / "rssft_reference_astral.jsonl", None)],
    ("rs_sft", "heldout"): [("raw", D / "rssft_reference_heldout.jsonl", None)],
    ("e3", "astral"): [("raw", D / "e3_astral_n32.jsonl", None)],
    ("e3", "heldout"): [("raw", D / "e3_heldout_n32.jsonl", None)],
    ("e3c", "astral"): [("raw", D / "e3c_astral_n32.jsonl", None)],
    ("e3c", "heldout"): [("raw", D / "e3c_heldout_n32.jsonl", None)],
    ("gdpo_v3_ckpt300", "astral"): [("structured", R / "astral_gen_n32_gdpo300.json", None)],
    ("gdpo_v3_ckpt300", "heldout"): [("raw", D / "gdpo300_heldout_n32.jsonl", None)],
    ("phase12_ckpt100", "astral"): [("raw", D / "phase12_ckpt100_astral_n32.jsonl", None)],
    ("phase12_ckpt100", "heldout"): [("raw", D / "phase12_ckpt100_heldout_n32.jsonl", None)],
    ("phase12_ckpt200", "astral"): [("raw", D / "phase12_ckpt200_astral_n32.jsonl", None)],
    ("phase12_ckpt200", "heldout"): [("raw", D / "phase12_ckpt200_heldout_n32.jsonl", None)],
    ("phase12_ckpt300", "astral"): [("structured", R / "astral_gen_n32_gdpo_phase12_beta0.json", None)],
    ("phase12_ckpt300", "heldout"): [("raw", D / "phase12_ckpt300_heldout_n32.jsonl", None)],
    ("e1_ckpt100", "astral"): [("raw", D / "e1_ckpt100_astral_n32.jsonl", None)],
    ("e1_ckpt100", "heldout"): [("raw", D / "e1_ckpt100_heldout_n32.jsonl", None)],
    ("e1_ckpt200", "astral"): [("raw", D / "e1_ckpt200_astral_n32.jsonl", None)],
    ("e1_ckpt200", "heldout"): [("raw", D / "e1_ckpt200_heldout_n32.jsonl", None)],
    ("e1_ckpt300", "astral"): [("structured", R / "astral_gen_n32_e1.json", None)],
    ("e1_ckpt300", "heldout"): [("raw", D / "e1_heldout_n32.jsonl", None)],
}
MODELS = ["base", "corpus_sft", "rs_sft", "e3", "e3c", "gdpo_v3_ckpt300",
          "phase12_ckpt100", "phase12_ckpt200", "phase12_ckpt300",
          "e1_ckpt100", "e1_ckpt200", "e1_ckpt300"]

RECIPE_CACHE = OUT_DIR / "recipes_by_model.json"


def _normalize(f: str) -> str:
    from validator import SynthesisValidator
    return SynthesisValidator._normalize_formula(f)


def _load_raw(path: Path, group_filter):
    from core.reward import parse_completion
    from support_growth_by_checkpoint import precset_of
    out = defaultdict(list)
    n_rows = defaultdict(int)
    groups = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec["target"]
            n_rows[t] += 1
            try:
                route = parse_completion(rec["completion"], t)
                ps = precset_of(route) if route is not None else None
            except Exception:
                ps = None
            if ps:
                out[t].append(list(ps))
                groups[t][rec.get("sample_group") or "all"].append(list(ps))
    return out, n_rows, groups


def _load_structured(path: Path):
    doc = json.loads(path.read_text())
    out = defaultdict(list)
    n_rows = defaultdict(int)
    for r in doc["results"]:
        t = r["target"]
        for s in r["samples"]:
            n_rows[t] += 1
            precs = s.get("precursors")
            if precs:
                out[t].append(sorted({_normalize(p) for p in precs}))
    return out, n_rows


def build_recipe_cache() -> dict:
    """{model: {target_set: {"samples": {target: [recipe,...]}, "n_rows": {...},
    "by_group": {target: {group: [...]}}, "source": ...}}}"""
    cache = {}
    for (model, ts), srcs in SOURCES.items():
        kind, path, _ = srcs[0]
        if not path.exists():
            cache.setdefault(model, {})[ts] = {"missing": str(path)}
            continue
        if kind == "raw":
            samples, n_rows, groups = _load_raw(path, None)
            by_group = {t: dict(g) for t, g in groups.items()}
        else:
            samples, n_rows = _load_structured(path)
            by_group = {t: {"all": v} for t, v in samples.items()}
        cache.setdefault(model, {})[ts] = {
            "source": str(path.relative_to(REPO_ROOT)), "kind": kind,
            "samples": dict(samples), "n_rows": dict(n_rows), "by_group": by_group,
        }
        print(f"  recipes: {model}/{ts}: {sum(len(v) for v in samples.values())} parsed "
              f"of {sum(n_rows.values())} rows over {len(n_rows)} targets", flush=True)
    return cache


def load_recipes() -> dict:
    if RECIPE_CACHE.exists():
        return json.loads(RECIPE_CACHE.read_text())
    cache = build_recipe_cache()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RECIPE_CACHE.write_text(json.dumps(cache))
    return cache


def recipes(cache: dict, model: str, ts: str, target: str, group: str | None = None) -> list[tuple]:
    entry = cache.get(model, {}).get(ts, {})
    if group is None:
        return [tuple(r) for r in entry.get("samples", {}).get(target, [])]
    return [tuple(r) for r in entry.get("by_group", {}).get(target, {}).get(group, [])]


def ts_of(target: str, sets: dict) -> str:
    return "astral" if target in sets["astral_35"] else "heldout"


# ---------------------------------------------------------------------------
# ASTRAL lab data
# ---------------------------------------------------------------------------
ASTRAL_PURITY_SUCCESS = 0.5   # ASTRAL's own ">0.5 high" purity band


def astral_lab() -> dict:
    """target -> {traditional, predicted (canonical tuples), purities, T,
    phase16_hit_set (= predicted, Phase 16's hit definition),
    lab_success_sets (sets whose best measured purity > 0.5)}"""
    doc = json.loads(ASTRAL_VALIDATION_PATH.read_text())
    out = {}
    for t in doc["targets"]:
        trad = tuple(sorted({_normalize(p) for p in t["traditional"]}))
        pred = tuple(sorted({_normalize(p) for p in t["predicted"]}))
        succ = []
        fail = []
        for s, pur in ((trad, t["trad_best_purity"]), (pred, t["pred_best_purity"])):
            if pur is None:
                continue
            (succ if pur > ASTRAL_PURITY_SUCCESS else fail).append(s)
        out[t["target"]] = {
            "traditional": trad, "predicted": pred,
            "trad_best_purity": t["trad_best_purity"], "pred_best_purity": t["pred_best_purity"],
            "trad_best_T": t["trad_best_T"], "pred_best_T": t["pred_best_T"],
            "phase16_hit_set": pred,
            "lab_success_sets": succ, "lab_fail_sets": fail,
        }
    return out


def jaccard_distance(a, b) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return 0.0 if not u else 1.0 - len(sa & sb) / len(u)
