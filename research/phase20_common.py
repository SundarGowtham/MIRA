#!/usr/bin/env python
"""
research/phase20_common.py -- shared constants and provenance for Phase 20
(docs/phases/PHASE20_PREREG.md).

provenance(require_commit=True) asserts the live blob of PHASE20_PREREG.md
equals the frozen blob saved at the start of the phase and that the blob is
in a commit. Every ANALYSIS step (anything that computes a P-20 number) calls
it with require_commit=True. Stage 0 gates, training and sampling call it with
require_commit=False (they are not analyses), and record whatever commit
exists at the time.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

PREREG_PATH = REPO_ROOT / "docs" / "phases" / "PHASE20_PREREG.md"
PREREG_BLOB_SHA_FROZEN = "15d89d8a177920db0b4ae4d9a65972d9ebeb8ccf"
SEED = 20261116
N_BOOT = 10000
OUT_DIR = REPO_ROOT / "results" / "phase20"
STAGE_DIR = REPO_ROOT / "runs" / "phase20"
NTFY_TOPIC = "mira-g5x7k2-status"
BASE_MODEL = "Qwen/Qwen3-8B"

# (data dir, prefix) of every SFT dataset in the phase. Same files as the old runs.
SFT_DATASETS = {
    "rs_sft": ("data/rs_sft", "rs_sft"),
    "e3": ("data/e3", "e3"),
    "e3c": ("data/e3c", "e3c"),
    "sft_v3": ("data/sft_v3", "sft"),
}
# new tag -> (dataset key, run dir created by train.py: runs/sft-qlora-<tag>)
NEW_SFT = {
    "sft-rssft-full": "rs_sft",
    "sft-e3-full": "e3",
    "sft-e3c-full": "e3c",
    "sft-corpus-full": "sft_v3",
}

# Sampling settings fixed by the spec (Sampling table).
SAMPLING_SPEC = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 8192, "seed": 42}


def notify(msg: str) -> None:
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode(), method="POST")
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def prereg_commit() -> str | None:
    blob = _git("hash-object", str(PREREG_PATH))
    if blob != PREREG_BLOB_SHA_FROZEN:
        raise SystemExit(f"PHASE20_PREREG.md changed after it was saved: {blob} != {PREREG_BLOB_SHA_FROZEN}")
    commits = _git("log", "--format=%H", "--find-object=" + blob, "--", str(PREREG_PATH)).splitlines()
    return commits[-1] if commits else None


def provenance(require_commit: bool = True) -> dict:
    commit = prereg_commit()
    if require_commit and not commit:
        raise SystemExit("PHASE20_PREREG.md (blob %s) is not committed; analyses wait for the commit"
                         % PREREG_BLOB_SHA_FROZEN)
    return {"prereg_commit": commit, "prereg_blob_sha": PREREG_BLOB_SHA_FROZEN,
            "repo_head_at_run": _git("rev-parse", "HEAD"), "seed": SEED}


def write_json(name: str, obj: dict) -> Path:
    """New files only: refuse to overwrite a git-tracked file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=REPO_ROOT,
                      capture_output=True).returncode == 0:
        raise SystemExit(f"refusing to overwrite tracked results file {path}")
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


def base_model_path() -> str:
    snaps = sorted((Path("/models/huggingface/hub/models--Qwen--Qwen3-8B/snapshots")).glob("*"))
    return str(snaps[-1]) if snaps else BASE_MODEL


# ---------------------------------------------------------------------------
# Models, sample sources and the recipe cache (new models added to Phase 18's)
# ---------------------------------------------------------------------------
NOV = REPO_ROOT / "data" / "novelty"
P20 = NOV / "phase20"

# label -> adapter dir (old labels as in research/phase18_common.ADAPTERS)
NEW_ADAPTERS = {
    "sft_rssft_full": "runs/sft-qlora-sft-rssft-full/final",
    "sft_e3_full": "runs/sft-qlora-sft-e3-full/final",
    "sft_e3c_full": "runs/sft-qlora-sft-e3c-full/final",
    "sft_corpus_full": "runs/sft-qlora-sft-corpus-full/final",
    "e1full_ckpt100": "runs/gdpo-qlora-gdpo-e1-ranker-rssft-full/checkpoint-100",
    "e1full_ckpt200": "runs/gdpo-qlora-gdpo-e1-ranker-rssft-full/checkpoint-200",
    "e1full_ckpt300": "runs/gdpo-qlora-gdpo-e1-ranker-rssft-full/checkpoint-300",
}
# old model each new one replaces
REPLACES = {"sft_rssft_full": "rs_sft", "sft_e3_full": "e3", "sft_e3c_full": "e3c",
            "sft_corpus_full": "corpus_sft", "e1full_ckpt300": "e1_ckpt300",
            "e1full_ckpt100": "e1_ckpt100", "e1full_ckpt200": "e1_ckpt200"}

# (model, target_set) -> raw jsonl. sft_rssft_full uses its 232-sample reference
# when present (Phase 18 convention for RS-SFT: 200 "reference" + 32
# "resampling_rate"), else its n32 draw.
NEW_SOURCES = {}
for _m in ("sft_e3_full", "sft_e3c_full", "sft_corpus_full",
           "e1full_ckpt100", "e1full_ckpt200", "e1full_ckpt300"):
    for _ts in ("astral", "heldout"):
        NEW_SOURCES[(_m, _ts)] = P20 / f"{_m}_{_ts}_n32.jsonl"
for _ts in ("astral", "heldout"):
    NEW_SOURCES[("sft_rssft_full", _ts)] = P20 / f"sft_rssft_full_reference_{_ts}.jsonl"
    NEW_SOURCES[("sft_rssft_full_n32", _ts)] = P20 / f"sft_rssft_full_{_ts}_n32.jsonl"
# Stage 5 temperature-fix resamples of four old models (ASTRAL only)
T09_SOURCES = {m: P20 / f"{m}_astral_n32_t09vllm.jsonl"
               for m in ("corpus_sft", "gdpo_v3_ckpt300", "phase12_ckpt300", "e1_ckpt300")}

NEW_RECIPE_CACHE = OUT_DIR / "recipes_new_models.json"


def _src_sig(path: Path):
    return [str(path), path.stat().st_size, int(path.stat().st_mtime)] if path.exists() else None


def load_recipes20(rebuild: bool = False) -> dict:
    """Phase 18's recipe cache (results/phase18/recipes_by_model.json, unchanged)
    plus the Phase 20 models and the Stage 5 resamples (as '<model>_t09'),
    parsed by the same code (phase18_common._load_raw: parse_completion +
    precset_of). New-model entries are cached keyed by source file size/mtime."""
    import research.phase18_common as c18
    from research.phase18_common import _load_raw
    # patch_phase18 replaces c18.load_recipes with this function; use the original
    load_recipes = getattr(c18, "_orig_load_recipes", c18.load_recipes)
    cache = json.loads(json.dumps(load_recipes()))
    new = json.loads(NEW_RECIPE_CACHE.read_text()) if NEW_RECIPE_CACHE.exists() and not rebuild else {}
    srcs = dict(NEW_SOURCES)
    srcs.update({(f"{m}_t09", "astral"): p for m, p in T09_SOURCES.items()})
    changed = False
    ref_done = (STAGE_DIR / "STAGE_S4_sample_ref.done").exists()
    for (m, ts), path in srcs.items():
        sig = _src_sig(path)
        if sig is None:
            continue
        # the 232-sample reference is written over ~a day; never read it half-written
        if m == "sft_rssft_full" and not ref_done:
            new.get(m, {}).pop(ts, None)
            continue
        ent = new.get(m, {}).get(ts)
        if ent and ent.get("sig") == sig:
            continue
        samples, n_rows, groups = _load_raw(path, None)
        new.setdefault(m, {})[ts] = {
            "source": str(path.relative_to(REPO_ROOT)), "kind": "raw", "sig": sig,
            "samples": dict(samples), "n_rows": dict(n_rows),
            "by_group": {t: dict(g) for t, g in groups.items()}}
        changed = True
        print(f"  recipes20: {m}/{ts}: {sum(len(v) for v in samples.values())} parsed "
              f"of {sum(n_rows.values())} rows over {len(n_rows)} targets", flush=True)
    if changed:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        NEW_RECIPE_CACHE.write_text(json.dumps(new))
    for m, d in new.items():
        cache[m] = d
    # before the 232-sample reference exists, sft_rssft_full falls back to its
    # n32 draw (flagged in "source"; P-20-1/P-20-3 final runs require the reference)
    if "sft_rssft_full_n32" in cache:
        for ts, ent in cache["sft_rssft_full_n32"].items():
            cache.setdefault("sft_rssft_full", {}).setdefault(ts, ent)
    return cache


def have_samples(*models, sets=("astral", "heldout")) -> bool:
    for m in models:
        for ts in sets:
            p = NEW_SOURCES.get((m, ts))
            if p is None or not p.exists():
                return False
    return True


def all_adapters() -> dict:
    from research.phase18_common import ADAPTERS
    return {**ADAPTERS, **NEW_ADAPTERS}


def patch_phase18(steps: dict, own_ft=("S2", "S3", "S4"), practice=("S5", "S6", "S7"),
                  out_prefix: str = "", require_commit: bool = True):
    """Run Phase 18 block code unchanged with a Phase 20 step table: redirect its
    step table, recipe cache, provenance and outputs. Must be called before
    importing the block module."""
    import research.phase18_common as c
    c.STEPS = dict(steps)
    c.FINE_TUNING_STEPS = tuple(s for s in steps if s in ("S1",) + tuple(own_ft))
    c.OWN_SAMPLE_FT_STEPS = tuple(own_ft)
    c.PRACTICE_STEPS = tuple(practice)
    c.ADAPTERS = all_adapters()
    if not hasattr(c, "_orig_load_recipes"):
        c._orig_load_recipes = c.load_recipes
    c.load_recipes = load_recipes20
    prov = provenance(require_commit=require_commit)
    c.provenance = lambda: dict(prov)
    c.write_json = lambda name, obj: write_json(out_prefix + name, obj)
    c.OUT_DIR = OUT_DIR
    return c
