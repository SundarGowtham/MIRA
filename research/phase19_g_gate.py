#!/usr/bin/env python
"""
research/phase19_g_gate.py -- Phase 19 Block G (GPU, forward only).
docs/phases/PHASE19_PREREG.md "Block G".

For each model, l_m(y) = exact teacher-forced log p of every probe string
(answer-only context, research/phase19_common.py). Per target, Spearman
between l_m(y) and the recipe's count in the model's own samples. Gate:
median >= 0.4 for every model in the step table (S3, S4, S6, S7: Base, E3,
E3c, RS-SFT, Phase 12 @300, E1 @300). Corpus SFT and GDPO-300 (S5,
secondary) are scored and reported, not gated. Targets whose counts are
constant across the strings (Spearman undefined) are dropped per model and
counted. Bootstrap CI of each median: cluster by target, 10,000, seed
20261116.

Outputs: results/phase19/string_scores.jsonl (reused by E, D, F1),
results/phase19/gate.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from research.phase19_common import (  # noqa: E402
    MultiAdapterModel, Prepared, score_prepared, load_probe_strings, provenance, write_json, notify,
    SCORES_PATH, MODELS, SEED, N_BOOT,
)

GATED = ["base", "e3", "e3c", "rs_sft", "phase12_ckpt300", "e1_ckpt300"]


def main():
    prov = provenance()
    ps = load_probe_strings()
    mm = MultiAdapterModel([m for m in MODELS if m != "base"])
    preps = {t: Prepared(mm.tok, t, [r["string"] for r in rows]) for t, rows in ps.items()}
    done = set()
    if SCORES_PATH.exists():
        for line in SCORES_PATH.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["target"]))
    with SCORES_PATH.open("a") as f:
        for m in MODELS:
            mm.set_all(None if m == "base" else m)
            t0 = time.time()
            for t, rows in ps.items():
                if (m, t) in done:
                    continue
                sc = score_prepared(mm, preps[t])
                for r, l in zip(rows, sc):
                    f.write(json.dumps({"model": m, "target": t, "recipe_key": r["recipe_key"], "l": l,
                                        "n_answer_tokens": int(sum(preps[t].masks[r["rank"]]))}) + "\n")
                f.flush()
            print(f"  scored {m} in {time.time() - t0:.0f}s", flush=True)
    scores = {}
    for line in SCORES_PATH.read_text().splitlines():
        r = json.loads(line)
        scores[(r["model"], r["target"], r["recipe_key"])] = r["l"]
    rng = np.random.default_rng(SEED)
    res = {}
    for m in MODELS:
        rhos, dropped = {}, []
        for t, rows in ps.items():
            l = [scores[(m, t, r["recipe_key"])] for r in rows]
            c = [r["counts"][m] for r in rows]
            if len(set(c)) < 2 or len(rows) < 3:
                dropped.append(t)
                continue
            rhos[t] = float(spearmanr(l, c).correlation)
        v = np.array(list(rhos.values()))
        idx = rng.integers(0, len(v), (N_BOOT, len(v)))
        bs = np.median(v[idx], axis=1)
        res[m] = {"median_spearman": float(np.median(v)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "n_targets": len(v), "dropped_constant_counts": dropped, "per_target": rhos,
                  "gated": m in GATED, "passes_0.4": bool(np.median(v) >= 0.4)}
        print(f"  {m:18s} median rho {res[m]['median_spearman']:.3f} CI {res[m]['ci95']} n={len(v)} "
              f"dropped={len(dropped)}", flush=True)
    passes = all(res[m]["passes_0.4"] for m in GATED)
    doc = {**prov, "block": "G", "gate_rule": "median per-target Spearman(l_m, count_m) >= 0.4 for every model in the step table",
           "gated_models": GATED, "gate_passes": passes, "by_model": res,
           "phase17_answer_only_reference": "0.45-0.57 (Phase 17 answer-only medians)"}
    p = write_json("gate.json", doc)
    print(f"GATE passes={passes} -> {p}")
    notify(f"MIRA 19 Block G: gate passes={passes}; " + ", ".join(f"{m}={res[m]['median_spearman']:.2f}" for m in GATED))


if __name__ == "__main__":
    main()
