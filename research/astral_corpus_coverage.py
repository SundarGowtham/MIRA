#!/usr/bin/env python
"""
astral_corpus_coverage.py — URGENT_PRESENTATION_PREP.md Step 3.

The ASTRAL paper's winning ("predicted") precursors are reported absent from
the text-mined synthesis corpus (Kononova, MIRA's SFT training data). Verify
directly on our own copy: count occurrences in data/raw/synthesis_clean.json
(17.6k routes) for the novel precursors, contrasted with traditional ones.
Then plot corpus frequency of each ASTRAL route's precursor set against its
measured phase purity, one point per route (70 total).

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/astral_corpus_coverage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ranker import build_precursor_frequency  # noqa: E402
from validator import SynthesisValidator  # noqa: E402

DATA_PATH = Path("misc/astral_validation_set.json")
SYNTHESIS_PATH = Path("data/raw/synthesis_clean.json")
OUT_JSON = Path("misc/corpus_coverage.json")
OUT_FIG = Path("manifold_visualization/figures/corpus_coverage.png")

TRADITIONAL_CONTRAST = ["Li2CO3", "B2O3", "BaO", "NH4H2PO4", "K2CO3", "Na2CO3"]


def norm(f: str) -> str:
    return SynthesisValidator._normalize_formula(f)


def main():
    data = json.loads(DATA_PATH.read_text())
    freq = build_precursor_frequency(SYNTHESIS_PATH)
    total = sum(freq.values())
    print(f"corpus: {len(freq)} distinct precursor formulas, {total} total mentions "
          f"across {SYNTHESIS_PATH}", flush=True)

    novel = data["novel_precursors_absent_from_literature_corpus"]
    print("\n--- novel (ASTRAL-winning) precursors: corpus frequency ---")
    novel_counts = {}
    for p in novel:
        c = freq.get(norm(p), 0)
        novel_counts[p] = c
        print(f"  {p:<10} -> {c}")

    print("\n--- traditional precursors: corpus frequency (contrast) ---")
    trad_counts = {}
    for p in TRADITIONAL_CONTRAST:
        c = freq.get(norm(p), 0)
        trad_counts[p] = c
        print(f"  {p:<10} -> {c}")

    print(f"\nmean novel freq: {sum(novel_counts.values())/len(novel_counts):.1f}  "
          f"mean traditional freq: {sum(trad_counts.values())/len(trad_counts):.1f}")

    # per-route: mean corpus frequency across that route's precursor set
    rows = []
    for t in data["targets"]:
        for kind in ("traditional", "predicted"):
            precs = t[kind]
            mean_freq = sum(freq.get(norm(p), 0) for p in precs) / len(precs)
            purity = t["trad_best_purity"] if kind == "traditional" else t["pred_best_purity"]
            rows.append({"target": t["target"], "kind": kind, "precursors": precs,
                        "mean_corpus_freq": mean_freq, "measured_purity": purity})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "novel_precursor_corpus_freq": novel_counts,
        "traditional_precursor_corpus_freq": trad_counts,
        "corpus_total_mentions": total,
        "corpus_n_distinct_precursors": len(freq),
        "routes": rows,
    }, indent=1))
    print(f"\n-> {OUT_JSON}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for kind, color, label in [("traditional", "tab:blue", "traditional precursors"),
                               ("predicted", "tab:orange", "predicted precursors")]:
        xs = [r["mean_corpus_freq"] for r in rows if r["kind"] == kind]
        ys = [r["measured_purity"] for r in rows if r["kind"] == kind]
        ax.scatter(xs, ys, color=color, alpha=0.8, label=label)
    ax.set_xlabel("mean corpus frequency of route's precursor set\n"
                  "(occurrences in data/raw/synthesis_clean.json, 17.6k routes)")
    ax.set_ylabel("measured phase purity (ASTRAL robot)")
    ax.set_title("Corpus frequency vs. measured phase purity\n(35 ASTRAL targets, 70 routes)")
    ax.set_xscale("symlog")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"-> {OUT_FIG}")


if __name__ == "__main__":
    main()
