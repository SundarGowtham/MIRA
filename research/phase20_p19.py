#!/usr/bin/env python
"""
research/phase20_p19.py -- run Phase 19 Blocks G, E and F1 UNCHANGED
(research/phase19_g_gate.py, phase19_e_location.py, phase19_f1_dimension.py,
the 671 probe strings of results/phase19/probe_strings.jsonl, the same
context, scoring, grid, seeds) on the Phase 20 step tables. Only the step
table, the model list, the adapter map, the per-model recipe counts added to
the probe rows, provenance and the output directory are redirected.

Step tables (labels keep Phase 19's so the block code's own predictions and
orderings run unmodified):
  --stage 2b : S3 = Base -> sft-e3-full, S4 = Base -> sft-e3c-full,
               S6 = RS-SFT -> Phase 12 @300 (old), S7 = RS-SFT -> E1 @300 (old)
  --stage 4  : S3, S4 as above, S6 old, S7 = sft-rssft-full -> E1-full @300
Old steps' Block E configurations are taken from Phase 19's cache
(results/phase19/location_scores.jsonl; same code, same shapes); new ones
are computed. In stage 4, S3/S4 come from the stage-2b cache.

Block G is run on Base + the new models only (old models' gate is Phase 19's).
Counts for the new models' rows come from their own samples, parsed by the
Phase 18 canonicaliser (research/phase20_common.load_recipes20):
sft-rssft-full uses its 232-sample reference when present, else its n32 set.

Outputs: results/phase20/p19_<stage>/{string_scores.jsonl, gate.json,
location_scores.jsonl, location.json, figdata_location.json, dimension.json}

Usage: PYTHONPATH=. uv run python research/phase20_p19.py --stage 2b --block G
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

from research.phase20_common import (OUT_DIR as P20_OUT, all_adapters, load_recipes20,  # noqa: E402
                                     provenance)

P19_OUT = REPO_ROOT / "results" / "phase19"
NEW_SFT = ["sft_rssft_full", "sft_e3_full", "sft_e3c_full"]

STEP_TABLES = {
    "2b": {"S3": ("base", "sft_e3_full"), "S4": ("base", "sft_e3c_full"),
           "S6": ("rs_sft", "phase12_ckpt300"), "S7": ("rs_sft", "e1_ckpt300")},
    "4": {"S3": ("base", "sft_e3_full"), "S4": ("base", "sft_e3c_full"),
          "S6": ("rs_sft", "phase12_ckpt300"), "S7": ("sft_rssft_full", "e1full_ckpt300")},
}
G_MODELS = {"2b": ["base"] + NEW_SFT, "4": ["base"] + NEW_SFT + ["e1full_ckpt300"]}
# Block E configurations reused from earlier caches: (cache file, step labels)
E_SEED = {
    "2b": [(P19_OUT / "location_scores.jsonl", ("S6", "S7"))],
    "4": [(P19_OUT / "location_scores.jsonl", ("S6",)),
          (P20_OUT / "p19_2b" / "location_scores.jsonl", ("S3", "S4"))],
}


def build_probe_file(out: Path) -> dict:
    """Phase 19 probe rows unchanged (strings, ranks, keys), with per-model
    counts / sampled flags / n_samples_parsed added for the Phase 20 models."""
    from research.phase18_common import recipes
    cache = load_recipes20()
    rows = [json.loads(l) for l in (P19_OUT / "probe_strings.jsonl").read_text().splitlines() if l.strip()]
    src = {}
    for m in NEW_SFT + ["e1full_ckpt300", "sft_corpus_full"]:
        if m not in cache:
            continue
        for r in rows:
            ts = r["target_set"]
            ent = cache[m].get(ts, {})
            if "samples" not in ent:
                continue
            lst = [tuple(x) for x in ent["samples"].get(r["target"], [])]
            src[m] = ent["source"]
            y = tuple(r["recipe"])
            r["counts"][m] = sum(1 for x in lst if x == y)
            r["sampled"][m] = r["counts"][m] > 0
            r.setdefault("n_samples_parsed", {})[m] = len(lst)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return src


def setup(stage: str, block: str):
    out = P20_OUT / f"p19_{stage}"
    out.mkdir(parents=True, exist_ok=True)
    import research.phase19_common as c
    steps = STEP_TABLES[stage]
    c.STEPS = dict(steps)
    c.METHOD = {"S3": "SFT", "S4": "SFT", "S6": "RL", "S7": "RL"}
    c.PRIMARY_STEPS = ("S3", "S4", "S6", "S7")
    c.SECONDARY_STEPS = ()
    if block == "G":
        c.MODELS = list(G_MODELS[stage])
    else:
        c.MODELS = ["base"] + sorted({m for b, a in steps.values() for m in (b, a) if m != "base"})
    ad = all_adapters()
    c.adapters = lambda: dict(ad)
    prov = provenance(require_commit=True)
    prov["phase20_stage"] = stage
    prov["phase20_step_table"] = steps
    c.provenance = lambda: dict(prov)
    c.OUT_DIR = out

    def write_json(name, obj):
        p = out / name
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=REPO_ROOT,
                          capture_output=True).returncode == 0:
            raise SystemExit(f"refusing to overwrite tracked file {p}")
        p.write_text(json.dumps(obj, indent=1, default=str))
        return p
    c.write_json = write_json
    c.SCORES_PATH = out / "string_scores.jsonl"
    c.PROBE_STRINGS_PATH = out / "probe_strings_with_counts.jsonl"
    src = build_probe_file(c.PROBE_STRINGS_PATH)
    (out / "count_sources.json").write_text(json.dumps(src, indent=1))
    return c, out


def seed_e_cache(stage: str, cache_path: Path):
    have = set()
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                have.add(json.loads(line)["tag"])
    with cache_path.open("a") as f:
        for src, labels in E_SEED[stage]:
            if not src.exists():
                raise SystemExit(f"Block E seed cache missing: {src}")
            for line in src.read_text().splitlines():
                if not line.strip():
                    continue
                tag = json.loads(line)["tag"]
                if tag.split("|")[0] in labels and tag not in have:
                    f.write(line + "\n")
                    have.add(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STEP_TABLES))
    ap.add_argument("--block", required=True, choices=["G", "E", "F1"])
    a = ap.parse_args()
    c, out = setup(a.stage, a.block)
    if a.block == "G":
        import research.phase19_g_gate as g
        g.GATED = [m for m in c.MODELS if m != "base"]
        g.main()
    elif a.block == "E":
        import research.phase19_e_location as e
        e.CACHE = out / "location_scores.jsonl"
        seed_e_cache(a.stage, e.CACHE)
        e.main()
    else:
        import research.phase19_f1_dimension as f1
        f1.main()


if __name__ == "__main__":
    main()
