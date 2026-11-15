#!/usr/bin/env python
"""
research/phase18_d_audit_representation.py — Phase 18 Block 18D: does a
judge score the physics or the text? PHASE18_PREREG.md "18D".

Suite: Phase 17 R-a's 200 base routes, selected by the identical rule (first
200 completions of data/novelty/rssft_reference_astral.jsonl that parse with
core.reward.parse_completion and have >= 1 precursor). The transformation is
applied to the route as WRITTEN: each completion's own JSON object (extracted
with the same naive-span-then-balanced-candidates logic parse_completion
uses) is edited, re-serialised as "<think>\n</think>\n\n" + json, and run
through parse_completion -> judge. The untransformed object, re-serialised
the same way, is the reference. This tests the full text -> score path,
which is what a reward sees during training.

Transformations (all leave the physical recipe unchanged; decisions made
before running):
  T1  every temperature written in kelvin: the temperature value becomes the
      string "<T+273.15> K" in the same key (the schema has no unit field;
      the parser's documented handling of unit strings like "1 h" is the
      reason for writing the unit into the value). [E] T1b: key renamed to
      "temperature_k" with a numeric kelvin value.
  T2  every duration written in minutes: "<60 t> min" in the same key.
      [E] T2b: key "time_min" with a numeric value.
  T3  the first isothermal hold (first op with temperature > 100 C and
      time > 0) split into two consecutive identical ops of time t/2.
  T4  a zero-duration op inserted right after that hold, same type,
      temperature and atmosphere, time 0.
  T5  explicit ramp rate: NOT APPLICABLE. The route schema
      (PredictedConditions) has no ramp field and parse_completion drops
      unknown keys, so "only if the schema allows" excludes it.
  T6  precursors permuted (seeded shuffle; control, already in R-a).
Routes with no hold are excluded from T3/T4 (counted).

Judges and violations (|delta| > 1e-3):
  * validator: the 5-channel aggregate (mean of gradeable RUN3_CHECKS) and
    each RUN3 channel; a channel that changes gradeability (value <-> None /
    sentinel) counts as a violation of that channel. The other 5 validator
    channels are reported too ([E]).
  * ranker v2 (Ranker(..., validator_version=2).score): reward and each
    objective, same rule.
  * comparator: compare(orig, transformed) margin, |m| > 1e-3 = violation;
    if exactly one side fails the gates that is a violation ("gate flip");
    both failing = undefined (counted).
  * parse failures of the transformed text are their own category and are
    not counted as violations.
R-a' representation passes for a judge only with 0 violations.

Usage (tmux, CPU): scripts/run_phase18_block.sh 18D research/phase18_d_audit_representation.py
"""
from __future__ import annotations

import copy
import json
import pickle
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import ComparatorParams, load_comparator  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency, OBJECTIVE_NAMES  # noqa: E402
from core.reward import (RUN3_CHECKS, load_validator, parse_completion, THINK_RE, FENCE_RE,  # noqa: E402
                         _try_parse_json_object, _balanced_json_objects)
from research.ranker_capacity_rssft import SYNTHESIS  # noqa: E402
from research.distributional.arrows_gate_scoring import COMPARATOR_SCALES_PATH  # noqa: E402
from research.phase18_common import SEED, provenance, write_json, notify  # noqa: E402
from validator import SynthesisValidator, ThermoChecker  # noqa: E402

SOURCE = REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl"
N_ROUTES = 200
TOL = 1e-3
ALL_10 = ("stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist", "operation_order",
          "temperature_plausible", "thermodynamic_favorable", "target_stability", "chempot_atmosphere",
          "target_match")
G = {}


def extract_obj(text):
    t = FENCE_RE.sub("", THINK_RE.sub("", text, count=1).strip().strip())
    s, e = t.find("{"), t.rfind("}")
    data = _try_parse_json_object(t[s:e + 1]) if (s != -1 and e > s) else None
    if data is None:
        cands = []
        for c in _balanced_json_objects(t):
            o = _try_parse_json_object(c)
            if o is not None and ("precursors" in o or "operations" in o):
                cands.append((len(c), o))
        data = max(cands, key=lambda x: x[0])[1] if cands else None
    return data


def load_suite():
    out = []
    with SOURCE.open() as f:
        for line in f:
            if len(out) >= N_ROUTES:
                break
            rec = json.loads(line)
            try:
                route = parse_completion(rec["completion"], rec["target"])
            except Exception:
                continue
            if not route.precursors:
                continue
            obj = extract_obj(rec["completion"])
            out.append({"target": rec["target"], "obj": obj})
    return out


def keyget(op, names):
    for k in op:
        if k.lower() in names:
            return k
    return None


TEMP_KEYS = ("temperature_c", "temperature", "temperature_celsius")
TIME_KEYS = ("time_h", "time", "time_hours")


def num(v):
    from core.reward import _coerce_float
    return _coerce_float(v)


def hold_index(ops):
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            continue
        tk, hk = keyget(op, TEMP_KEYS), keyget(op, TIME_KEYS)
        if tk and hk and (num(op[tk]) or 0) > 100 and (num(op[hk]) or 0) > 0:
            return i
    return None


def transforms(obj, rng):
    out = {}
    ops = obj.get("operations") or []

    def each_op(fn):
        o = copy.deepcopy(obj)
        o["operations"] = [fn(copy.deepcopy(op)) if isinstance(op, dict) else op for op in (o.get("operations") or [])]
        return o

    def t1(op):
        k = keyget(op, TEMP_KEYS)
        if k and num(op[k]) is not None:
            op[k] = f"{round(num(op[k]) + 273.15, 2)} K"
        return op

    def t1b(op):
        k = keyget(op, TEMP_KEYS)
        if k and num(op[k]) is not None:
            v = round(num(op[k]) + 273.15, 2)
            del op[k]
            op["temperature_k"] = v
        return op

    def t2(op):
        k = keyget(op, TIME_KEYS)
        if k and num(op[k]) is not None:
            op[k] = f"{round(num(op[k]) * 60, 4)} min"
        return op

    def t2b(op):
        k = keyget(op, TIME_KEYS)
        if k and num(op[k]) is not None:
            v = round(num(op[k]) * 60, 4)
            del op[k]
            op["time_min"] = v
        return op

    out["T1"], out["T1b"], out["T2"], out["T2b"] = each_op(t1), each_op(t1b), each_op(t2), each_op(t2b)
    h = hold_index(ops)
    if h is not None:
        o = copy.deepcopy(obj)
        op = o["operations"][h]
        hk = keyget(op, TIME_KEYS)
        half = copy.deepcopy(op)
        half[hk] = num(op[hk]) / 2
        o["operations"] = o["operations"][:h] + [half, copy.deepcopy(half)] + o["operations"][h + 1:]
        out["T3"] = o
        o = copy.deepcopy(obj)
        z = copy.deepcopy(o["operations"][h])
        z[keyget(z, TIME_KEYS)] = 0
        o["operations"] = o["operations"][:h + 1] + [z] + o["operations"][h + 1:]
        out["T4"] = o
    o = copy.deepcopy(obj)
    precs = list(o.get("precursors") or [])
    if len(precs) >= 2:
        for _ in range(10):
            p2 = precs[:]
            rng.shuffle(p2)
            if p2 != precs:
                break
        o["precursors"] = p2
        out["T6"] = o
    return out


def as_text(obj):
    return "<think>\n</think>\n\n" + json.dumps(obj)


def judge(route, target):
    v = G["validator"]
    _r, bd = v.validate(route, target)
    vch = {}
    for ch in ALL_10:
        val = bd.get(ch)
        vch[ch] = None if (val is None or bd.get(f"{ch}_gradeability") in SynthesisValidator.SENTINEL_TAGS) else val
    grad = [vch[c] for c in RUN3_CHECKS if vch[c] is not None]
    agg5 = sum(grad) / len(grad) if grad else None
    rr, rbd = G["ranker"].score(route, target)
    rch = {o: rbd.get(o) for o in OBJECTIVE_NAMES}
    gate = all(G["comparator"]._gater._check_gates(route).values())
    return {"validator_agg5": agg5, "validator": vch, "ranker": rr, "ranker_ch": rch, "comp_gate": gate}


def diff_viol(a, b):
    if a is None and b is None:
        return False
    if (a is None) != (b is None):
        return True
    return bool(abs(a - b) > TOL)


def run_one(args):
    i, item = args
    rng = random.Random(SEED + i)
    target, obj = item["target"], item["obj"]
    try:
        ref_route = parse_completion(as_text(obj), target)
    except Exception as e:  # noqa: BLE001
        return {"i": i, "ref_parse_failure": repr(e)[:200]}
    ref = judge(ref_route, target)
    res = {"i": i, "target": target, "by_T": {}}
    for tname, tobj in transforms(obj, rng).items():
        try:
            route = parse_completion(as_text(tobj), target)
        except Exception as e:  # noqa: BLE001
            res["by_T"][tname] = {"parse_failure": repr(e)[:200]}
            continue
        j = judge(route, target)
        r = {"validator_agg5": diff_viol(ref["validator_agg5"], j["validator_agg5"]),
             "validator_ch": {c: diff_viol(ref["validator"][c], j["validator"][c]) for c in ALL_10},
             "ranker": diff_viol(ref["ranker"], j["ranker"]),
             "ranker_ch": {o: diff_viol(ref["ranker_ch"][o], j["ranker_ch"][o]) for o in OBJECTIVE_NAMES},
             "delta_agg5": (None if ref["validator_agg5"] is None or j["validator_agg5"] is None
                            else float(j["validator_agg5"] - ref["validator_agg5"])),
             "delta_ranker": float(j["ranker"] - ref["ranker"]),
             "parsed_temps": [c.conditions.heating_temperature for c in route.operations][:4],
             "parsed_times": [c.conditions.heating_time for c in route.operations][:4]}
        if ref["comp_gate"] and j["comp_gate"]:
            m, _ = G["comparator"].compare(ref_route, route, target, G["scales"], G["params"])
            r["comparator"] = bool(abs(m) > TOL)
            r["comparator_margin"] = float(m)
        elif ref["comp_gate"] != j["comp_gate"]:
            r["comparator"] = True
            r["comparator_margin"] = "gate_flip"
        else:
            r["comparator"] = None
            r["comparator_margin"] = "both_gate_fail"
        res["by_T"][tname] = r
    return res


def init_globals():
    G["validator"] = load_validator(REPO_ROOT / "data/cache/mp_formula_set.pkl", REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    G["comparator"] = load_comparator(REPO_ROOT / "data/cache/mp_formula_set.pkl", REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    G["scales"] = json.loads((REPO_ROOT / COMPARATOR_SCALES_PATH).read_text())["scales"]
    G["params"] = ComparatorParams(c3_fraction=0.17)
    with open(REPO_ROOT / "data/cache/mp_formula_set.pkl", "rb") as f:
        fs = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    G["ranker"] = Ranker(fs, thermo, build_precursor_frequency(SYNTHESIS), validator_version=2)


def main():
    prov = provenance()
    suite = load_suite()
    print(f"suite: {len(suite)} routes", flush=True)
    init_globals()  # before fork: workers inherit the loaded judges
    with Pool(16) as pool:
        results = list(pool.imap(run_one, list(enumerate(suite))))
    T_NAMES = ["T1", "T2", "T3", "T4", "T6", "T1b", "T2b"]
    table = {}
    for t in T_NAMES:
        rows = [r["by_T"][t] for r in results if "by_T" in r and t in r["by_T"]]
        parsed = [x for x in rows if "parse_failure" not in x]
        n = len(parsed)

        def rate(fn):
            v = [fn(x) for x in parsed]
            v = [x for x in v if x is not None]
            return {"n_violations": sum(v), "n": len(v), "rate": (sum(v) / len(v)) if v else None}
        entry = {"n_routes_applicable": len(rows), "n_parse_failures": len(rows) - n,
                 "validator_agg5": rate(lambda x: x["validator_agg5"]),
                 "validator_channels": {c: rate(lambda x, c=c: x["validator_ch"][c]) for c in ALL_10},
                 "ranker": rate(lambda x: x["ranker"]),
                 "ranker_channels": {o: rate(lambda x, o=o: x["ranker_ch"][o]) for o in OBJECTIVE_NAMES},
                 "comparator": rate(lambda x: x["comparator"]),
                 "comparator_gate_flips": sum(1 for x in parsed if x["comparator_margin"] == "gate_flip"),
                 "comparator_both_gate_fail": sum(1 for x in parsed if x["comparator_margin"] == "both_gate_fail")}
        d5 = [x["delta_agg5"] for x in parsed if x["delta_agg5"] is not None and abs(x["delta_agg5"]) > TOL]
        entry["validator_agg5_mean_delta_when_violated"] = (sum(d5) / len(d5)) if d5 else None
        dr = [x["delta_ranker"] for x in parsed if abs(x["delta_ranker"]) > TOL]
        entry["ranker_mean_delta_when_violated"] = (sum(dr) / len(dr)) if dr else None
        table[t] = entry
    primary = ["T1", "T2", "T3", "T4", "T6"]
    card = {}
    for jname, key in (("validator_5ch", "validator_agg5"), ("ranker_v2", "ranker"), ("comparator", "comparator")):
        tot = sum(table[t][key]["n_violations"] for t in primary)
        card[jname] = {"total_violations_T1_T6": tot, "R_a_prime_passes": tot == 0,
                       "by_T": {t: table[t][key]["n_violations"] for t in primary}}
    examples = [{"i": r["i"], "target": r["target"], "T1_parsed_temps": r["by_T"].get("T1", {}).get("parsed_temps"),
                 "T2_parsed_times": r["by_T"].get("T2", {}).get("parsed_times")}
                for r in results[:3] if "by_T" in r]
    doc = {**prov, "block": "18D", "tolerance": TOL, "n_routes": len(suite),
           "n_ref_parse_failures": sum(1 for r in results if "ref_parse_failure" in r),
           "T5": {"applied": False, "reason": "route schema has no ramp-rate field; parse_completion drops unknown "
                                              "keys, so the schema does not allow stating a ramp rate."},
           "transformations": {"T1": "temperature value -> '<K> K' string, same key",
                               "T2": "time value -> '<min> min' string, same key",
                               "T3": "first isothermal hold (T>100C, t>0) split into two consecutive t/2 holds",
                               "T4": "zero-duration op inserted after that hold",
                               "T6": "precursor order permuted",
                               "T1b [E]": "key renamed temperature_k, numeric kelvin",
                               "T2b [E]": "key renamed time_min, numeric minutes"},
           "violation_table": table, "report_card_R_a_prime": card, "parse_examples": examples}
    p = write_json("audit_representation.json", doc)
    for t in T_NAMES:
        e = table[t]
        print(f"{t}: n={e['n_routes_applicable']} parsefail={e['n_parse_failures']} "
              f"val5={e['validator_agg5']['n_violations']}/{e['validator_agg5']['n']} "
              f"ranker={e['ranker']['n_violations']}/{e['ranker']['n']} comp={e['comparator']['n_violations']}/{e['comparator']['n']}",
              flush=True)
        print("    validator channels:", {c: v["n_violations"] for c, v in e["validator_channels"].items() if v["n_violations"]})
        print("    ranker channels:", {c: v["n_violations"] for c, v in e["ranker_channels"].items() if v["n_violations"]})
    print(json.dumps(card, indent=1))
    print(f"-> {p}")
    notify("MIRA 18D done: " + ", ".join(f"{k} viol={v['total_violations_T1_T6']}" for k, v in card.items()))


if __name__ == "__main__":
    main()
