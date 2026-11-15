#!/usr/bin/env python
"""
research/phase18_arrows_pairs.py — shared pair table for Phase 18 blocks 18F
and 18H (and nothing else). Builds, with the frozen Phase 15 loader and
Phase 17 R-f's exact pair set, one record per lab pair carrying every judge's
per-route output, so 18F/18H read identical inputs.

YBCO (primary): research/distributional/arrows_gate_scoring.score_target(
"YBCO")["_all_pairs_primary_threshold"] -- within-temperature pairs with
|delta wt%| >= 5, the R-f / finding 27 primary endpoint; actual =
actual_winner(wt_a, wt_b, 5).
ASTRAL (flagged small): R-f's 35 traditional-vs-rule-picked pairs, routes
built exactly as phase17_rf_information_bound.astral_pairs builds them
(amounts 1.0, each set at its own best temperature, air); actual = sign of
the purity difference.

Per route: validator breakdown -> 10 channels (None if ungradeable or gate
failure), 5-channel aggregate; ranker v2 reward + objectives; comparator
gate pass. Per pair: R2 vote and vote_sum (gdpo_vote_and_channels), R5
comparator margin (compare(), both gates pass) or gate decision, carbonate-
free baseline.

Output: results/phase18/arrows_pair_table.json (cache; rebuilt if absent).
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase18_common import OUT_DIR, SEED, provenance  # noqa: E402

TABLE = OUT_DIR / "arrows_pair_table.json"
ALL_10 = ("stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist", "operation_order",
          "temperature_plausible", "thermodynamic_favorable", "target_stability", "chempot_atmosphere",
          "target_match")


def build():
    from core.comparator import ComparatorParams, load_comparator, SCORED_CHANNEL_NAMES
    from core.ranker import Ranker, build_precursor_frequency, OBJECTIVE_NAMES
    from core.reward import RUN3_CHECKS, load_validator
    from research.ranker_capacity_rssft import SYNTHESIS
    from research.distributional.arrows_gate_scoring import (
        load_target_entries, build_route, score_validator, score_target, gdpo_vote_and_channels,
        baseline_carbonate_free, actual_winner, COMPARATOR_SCALES_PATH)
    from validator import SynthesisValidator, ThermoChecker

    validator = load_validator(REPO_ROOT / "data/cache/mp_formula_set.pkl", REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    comparator = load_comparator(REPO_ROOT / "data/cache/mp_formula_set.pkl", REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    scales = json.loads((REPO_ROOT / COMPARATOR_SCALES_PATH).read_text())["scales"]
    params = ComparatorParams(c3_fraction=0.17)
    with open(REPO_ROOT / "data/cache/mp_formula_set.pkl", "rb") as f:
        fs = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    ranker = Ranker(fs, thermo, build_precursor_frequency(SYNTHESIS), validator_version=2)
    comp_max = float(len(SCORED_CHANNEL_NAMES))

    def route_info(route, target):
        bd = score_validator(validator, route, target)
        ch = {}
        for c in ALL_10:
            v = None if bd is None else bd.get(c)
            ch[c] = None if (v is None or bd.get(f"{c}_gradeability") in SynthesisValidator.SENTINEL_TAGS) else v
        g5 = [ch[c] for c in RUN3_CHECKS if ch[c] is not None]
        rr, rbd = ranker.score(route, target)
        return {"bd": bd, "val_ch": ch, "val_gate_fail": bd is None, "agg5": (sum(g5) / len(g5)) if g5 else None,
                "ranker": rr, "ranker_ch": {o: rbd.get(o) for o in OBJECTIVE_NAMES},
                "comp_gate": all(comparator._gater._check_gates(route).values())}

    def pair_record(ra, rb, ia, ib, target):
        w, reason, per_ch = gdpo_vote_and_channels(ia["bd"], ib["bd"])
        vote_sum = sum(v for _l, v in per_ch.values()) if per_ch else None
        if ia["comp_gate"] and ib["comp_gate"]:
            m, _ = comparator.compare(ra, rb, target, scales, params)
            comp = {"margin": float(m), "decided_by": "channels"}
        elif ia["comp_gate"] != ib["comp_gate"]:
            comp = {"margin": comp_max if ia["comp_gate"] else -comp_max, "decided_by": "gate"}
        else:
            comp = {"margin": None, "decided_by": "both_gate_fail"}
        return {"R2_vote": w, "R2_reason": reason, "R2_vote_sum": vote_sum,
                "R4_margin": float(ia["ranker"] - ib["ranker"]), "R5": comp,
                "base_carbfree": baseline_carbonate_free(ra, rb),
                "a": {k: v for k, v in ia.items() if k != "bd"}, "b": {k: v for k, v in ib.items() if k != "bd"}}

    rng = random.Random(SEED)
    ybco = score_target("YBCO", validator, comparator, scales, rng)
    pairs = ybco["_all_pairs_primary_threshold"]
    entries, target = load_target_entries("YBCO")
    routes, info = {}, {}
    for e in entries:
        k = (e["precursor_set"], e["temp_str"])
        routes[k] = build_route(target, e["precursors"], e["amounts"], e["temp_C"], e["atmosphere"])
        info[k] = route_info(routes[k], target)
    ybco_rows = []
    for p in pairs:
        ka, kb = (p["set_a"], p["temp"]), (p["set_b"], p["temp"])
        actual = actual_winner(p["wt_a"], p["wt_b"], 5)
        if actual is None:
            continue
        rec = pair_record(routes[ka], routes[kb], info[ka], info[kb], target)
        ybco_rows.append({"set_a": p["set_a"], "set_b": p["set_b"], "temp": p["temp"], "wt_a": p["wt_a"],
                          "wt_b": p["wt_b"], "actual": actual, **rec})
    print(f"YBCO pairs: {len(ybco_rows)}", flush=True)

    doc = json.loads((REPO_ROOT / "results/astral_validation_set.json").read_text())
    astral_rows = []
    for t in doc["targets"]:
        tg = t["target"]
        ra = build_route(tg, t["traditional"], [1.0] * len(t["traditional"]), t["trad_best_T"], "air")
        rb = build_route(tg, t["predicted"], [1.0] * len(t["predicted"]), t["pred_best_T"], "air")
        dp = t["trad_best_purity"] - t["pred_best_purity"]
        actual = 1 if dp > 1e-9 else (-1 if dp < -1e-9 else 0)
        rec = pair_record(ra, rb, route_info(ra, tg), route_info(rb, tg), tg)
        astral_rows.append({"set_a": "trad::" + tg, "set_b": "pred::" + tg, "target": tg, "actual": actual, **rec})
    out = {**provenance(), "block": "18F/18H shared input", "ybco_primary_5wtpct": ybco_rows,
           "astral_35_small": astral_rows, "comparator_gate_margin_magnitude": comp_max,
           "channels_validator": ALL_10, "channels_ranker": OBJECTIVE_NAMES}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(out, default=str))
    return out


def load_table():
    if TABLE.exists():
        return json.loads(TABLE.read_text())
    return build()


if __name__ == "__main__":
    t = load_table()
    print(len(t["ybco_primary_5wtpct"]), len(t["astral_35_small"]))
