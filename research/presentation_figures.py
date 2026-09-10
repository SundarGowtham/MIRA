#!/usr/bin/env python
"""
presentation_figures.py — URGENT_PRESENTATION_PREP.md Step 5, figures 3-5.
(Figures 1/2 -- astral_correlation.png, corpus_coverage.png -- already made
by astral_validation.py / astral_corpus_coverage.py in steps 1/3.)

3. passk_gap.png     — mean pass@k, base/sft/gdpo300, with gap-vs-k inset
                        (finding 4's sharpening signature, +2.4 -> +1.0)
4. reward_capacity.png — per-channel within-group z-variance, validator
                        (10 channels, baseline condition, misc/hardening.json)
                        vs ranker (6 channels, baseline condition,
                        misc/ranker_capacity_probe.json -- the pre-fix 48.7%
                        run, so "6 channels, 5 live" matches exactly)
5. intervention_ladder.png — capacity across every condition tried, 40% bar

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/presentation_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probe_hardening import capacity_metrics  # noqa: E402

FIGDIR = Path("manifold_visualization/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def fig_passk_gap():
    d = json.load(open("misc/passk_n200.json"))
    ks = [1, 2, 4, 8, 16]
    models = ["base", "sft", "gdpo300"]
    means = {m: [] for m in models}
    for k in ks:
        for m in models:
            vals = [r["models"][m][f"pass@{k}"] for r in d["per_target"]]
            means[m].append(sum(vals) / len(vals))
    gap = [ (g - s) * 100 for g, s in zip(means["gdpo300"], means["sft"]) ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for m, style in [("base", "tab:gray"), ("sft", "tab:blue"), ("gdpo300", "tab:red")]:
        ax1.plot(ks, means[m], marker="o", label=m, color=style)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ks)
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlabel("k")
    ax1.set_ylabel("pass@k (mean over 200 targets)")
    ax1.set_title("pass@k: base / SFT / GDPO-300")
    ax1.legend()

    ax2.plot(ks, gap, marker="o", color="tab:purple")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ks)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("k")
    ax2.set_ylabel("GDPO-300 - SFT pass@k (points)")
    ax2.set_title(f"Gap shrinks with k: {gap[0]:+.1f} -> {gap[-1]:+.1f} pts\n"
                  "(sharpening, not expansion; McNemar p=0.77 at k=16)")
    fig.tight_layout()
    out = FIGDIR / "passk_gap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}  (gap[1]={gap[0]:.2f} gap[16]={gap[-1]:.2f})")


def fig_reward_capacity():
    h = json.load(open("misc/hardening.json"))
    base_recs = [r for r in h["records"] if r["condition"] == "baseline"]
    val_checks = sorted({"stoichiometry", "amount_accuracy", "charge_neutrality",
                         "precursors_exist", "operation_order", "temperature_plausible",
                         "thermodynamic_favorable", "target_stability",
                         "chempot_atmosphere", "target_match"})
    val_cap = capacity_metrics(base_recs, val_checks)

    rp = json.load(open("misc/ranker_capacity_probe.json"))
    rbase = [r for r in rp["records"] if r["condition"] == "baseline"]
    ranker_checks = ["temperature_economy", "step_economy", "precursor_availability",
                     "volatility_risk", "phase_purity", "driving_force_margin"]
    ranker_cap = capacity_metrics(rbase, ranker_checks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cap, checks, title in [
        (ax1, val_cap, val_checks, f"Validator (Arm A) -- capacity {val_cap['capacity_pct']}%"),
        (ax2, ranker_cap, ranker_checks, f"Ranker (Arm B) -- capacity {ranker_cap['capacity_pct']}%"),
    ]:
        zvar = [cap["per_channel_z_var"][c] for c in checks]
        colors = ["tab:red" if z < 0.05 else "tab:green" for z in zvar]
        ax.barh(checks, zvar, color=colors)
        ax.set_title(title)
        ax.set_xlabel("within-group z-variance")
    n_dead_val = sum(1 for v in val_cap["per_channel_z_var"].values() if v < 0.05)
    n_dead_rk = sum(1 for v in ranker_cap["per_channel_z_var"].values() if v < 0.05)
    fig.suptitle(f"Per-channel within-group z-variance: validator "
                f"({len(val_checks)-n_dead_val}/{len(val_checks)} live) vs. ranker "
                f"({len(ranker_checks)-n_dead_rk}/{len(ranker_checks)} live)")
    fig.tight_layout()
    out = FIGDIR / "reward_capacity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


def fig_intervention_ladder():
    conditions = [
        ("baseline", 16.4), ("temp_ceiling", 17.7), ("inventory", 16.4),
        ("atmosphere", 16.3), ("combined", 16.1), ("low_temp", 23.8),
        ("ranker", 48.7),
    ]
    labels = [c[0] for c in conditions]
    vals = [c[1] for c in conditions]
    colors = ["tab:blue"] * 6 + ["tab:green"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=colors)
    ax.axhline(40, color="tab:red", linestyle="--", linewidth=1.5,
              label="pre-registered bar (40%)")
    ax.set_ylabel("reward capacity (%)")
    ax.set_title("Every intervention tried, Arm A (blue) vs. Arm B (green)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out = FIGDIR / "intervention_ladder.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    fig_passk_gap()
    fig_reward_capacity()
    fig_intervention_ladder()
