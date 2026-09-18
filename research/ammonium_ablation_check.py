#!/usr/bin/env python
"""
research/ammonium_ablation_check.py — the decisive check regular Claude
flagged as urgent: if the balance-solver bug's silent zeroing of
NH4H2PO4-containing traditional routes drove RS-SFT's rejection filter to
systematically delete the dominant CONVENTIONAL route for exactly those
targets (an "accidental ablation"), then Phase 12's positive result
(10/35 -> 14/35 predicted hits) should concentrate on the 18
ammonium-affected targets rather than being spread evenly.

Splits the 35 ASTRAL targets into ammonium-affected (traditional route
contains NH4H2PO4, n=18) and not (n=17), and reports the predicted-hit
rate within each group for base, RS-SFT, and GDPO-phase12-beta0 (ckpt
300) -- using the same per-target "any predicted" definition as
research/phase12_astral_final_analysis.py.

Usage (tmux):
  uv run python research/ammonium_ablation_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path("misc/astral_validation_set.json")
MODELS = {
    "base": Path("results/astral_gen_n32_base.json"),
    "rs_sft": Path("results/astral_gen_n32_rs_sft.json"),
    "gdpo_phase12_beta0_ckpt300": Path("results/astral_gen_n32_gdpo_phase12_beta0.json"),
}
OUT_JSON = Path("results/ammonium_ablation_check.json")


def per_target_predicted_hit(doc: dict) -> dict[str, bool]:
    out = {}
    for r in doc["results"]:
        matches = {s["match"] for s in r["samples"]}
        out[r["target"]] = bool(matches & {"PREDICTED", "PREDICTED_SUPERSET"})
    return out


def main():
    data = json.loads(DATA_PATH.read_text())
    targets = data["targets"]
    ammonium = sorted(t["target"] for t in targets if "NH4H2PO4" in t["traditional"])
    non_ammonium = sorted(t["target"] for t in targets if "NH4H2PO4" not in t["traditional"])
    print(f"ammonium-affected targets (n={len(ammonium)}): {ammonium}")
    print(f"not affected (n={len(non_ammonium)}): {non_ammonium}\n")

    results = {}
    for model_name, path in MODELS.items():
        doc = json.loads(path.read_text())
        hits = per_target_predicted_hit(doc)
        amm_hits = [hits[t] for t in ammonium if t in hits]
        non_hits = [hits[t] for t in non_ammonium if t in hits]
        results[model_name] = {
            "ammonium_affected": {
                "n": len(amm_hits), "n_hit": sum(amm_hits),
                "rate": sum(amm_hits) / len(amm_hits) if amm_hits else None,
                "hit_targets": [t for t in ammonium if hits.get(t)],
            },
            "not_affected": {
                "n": len(non_hits), "n_hit": sum(non_hits),
                "rate": sum(non_hits) / len(non_hits) if non_hits else None,
                "hit_targets": [t for t in non_ammonium if hits.get(t)],
            },
        }

    print(f"{'model':<28}{'ammonium-affected (n=18)':>28}{'not affected (n=17)':>24}")
    for model_name, r in results.items():
        a = r["ammonium_affected"]
        n = r["not_affected"]
        print(f"{model_name:<28}{a['n_hit']:>3}/{a['n']:<3} ({a['rate']:.1%})"
              f"{'':>10}{n['n_hit']:>3}/{n['n']:<3} ({n['rate']:.1%})")

    # The specific gain-concentration question: of the 4 targets GDPO
    # gained over RS-SFT (0 losses, 4 gains, per the pre-registered
    # McNemar), how many fall in the ammonium-affected group?
    base_hits = per_target_predicted_hit(json.loads(MODELS["base"].read_text()))
    rs_sft_hits = per_target_predicted_hit(json.loads(MODELS["rs_sft"].read_text()))
    gdpo_hits = per_target_predicted_hit(json.loads(MODELS["gdpo_phase12_beta0_ckpt300"].read_text()))
    common = sorted(set(rs_sft_hits) & set(gdpo_hits))
    gained = [t for t in common if gdpo_hits[t] and not rs_sft_hits[t]]
    lost = [t for t in common if rs_sft_hits[t] and not gdpo_hits[t]]
    gained_ammonium = [t for t in gained if t in ammonium]
    print(f"\nGDPO-phase12 gains over RS-SFT (n={len(gained)}): {gained}")
    print(f"  of which ammonium-affected: {gained_ammonium} ({len(gained_ammonium)}/{len(gained)})")
    print(f"GDPO-phase12 losses vs RS-SFT (n={len(lost)}): {lost}")

    OUT_JSON.write_text(json.dumps({
        "ammonium_affected_targets": ammonium,
        "not_affected_targets": non_ammonium,
        "per_model_group_rates": results,
        "rs_sft_to_gdpo_gains": gained,
        "rs_sft_to_gdpo_gains_ammonium_affected": gained_ammonium,
        "rs_sft_to_gdpo_losses": lost,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
