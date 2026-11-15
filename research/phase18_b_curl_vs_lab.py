#!/usr/bin/env python
"""
research/phase18_b_curl_vs_lab.py — Phase 18 Block 18B: does judge
inconsistency (curl) sit where the judge is wrong? PHASE18_PREREG.md "18B".

Primary: ARROWS3 YBCO (frozen loader research/distributional/arrows_gate_scoring.py).
  * Nodes = recipes (precursor_set keys of Exp.json).
  * Lab flow Y_lab(a,b) = mean over shared temperatures T of
    sign(yield_{a,T} - yield_{b,T}) (Phase 17 G2's definition).
  * Comparator flow Y(a,b) = margin from core/comparator.py::compare (current
    COMPARATOR_VERSION, misc/comparator_scales_v2.json, ComparatorParams(
    c3_fraction=0.17) as everywhere else). A recipe's route depends on
    temperature, so -- decided here before running -- Y(a,b) is the mean of
    compare(route_{a,T}, route_{b,T}) over the shared temperatures at which
    both routes pass the comparator's gates (comparator._gater._check_gates,
    the convention of arrows_gate_scoring.comparator_rule). An edge exists
    where both Y and Y_lab are defined.
  * Hodge: weighted least squares for the potential s, weight w_ab = number
    of shared temperatures used for Y(a,b) (the G2 weighting).
    rho_e = |Y_e - (s_a - s_b)|.
  * d_e = 1[sign Y_e != sign Y_lab_e] on edges with sign(Y_lab_e) != 0
    (a comparator margin of exactly 0 has sign 0 and counts as disagreement).
  * AUC of rho for predicting d (Mann-Whitney, ties = 1/2).
  * Bootstrap: 10,000 resamples of recipes (nodes) with replacement; an edge
    enters with multiplicity count(a)*count(b) (the cluster scheme of
    arrows_gate_scoring.cluster_bootstrap_ci). rho is held fixed at its
    full-graph value (the decomposition is not refit per resample).
  * Permutation null: 1,000 shuffles of rho across edges; one-sided
    p = (1 + #{AUC_null >= AUC_obs}) / 1001.

Prediction [C] P-18B-1: AUC > 0.5, CI excluding 0.5, permutation p < 0.05.

[E] ASTRAL: only two measured recipes per target (traditional, rule-picked),
so each target contributes a single edge and the pooled graph is a forest
with no cycles: every residual is 0 and the AUC is undefined. Reported as
structurally not computable, not run.
[E] Channel-conflict score for R2 (5-channel checker, RUN3_CHECKS) and R4
(ranker v2): per (pair, shared temperature), the share of channels gradeable
on both routes whose pairwise sign opposes the aggregate's sign; averaged
over shared temperatures. AUC for predicting that reward's own disagreement
with the lab sign (aggregate sign averaged over shared temperatures the same
way as the comparator flow).

Usage (tmux, CPU): scripts/run_phase18_block.sh 18B research/phase18_b_curl_vs_lab.py
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import ComparatorParams, load_comparator, COMPARATOR_VERSION  # noqa: E402
from core.ranker import Ranker, build_precursor_frequency, OBJECTIVE_NAMES  # noqa: E402
from core.reward import RUN3_CHECKS, load_validator  # noqa: E402
from research.ranker_capacity_rssft import SYNTHESIS  # noqa: E402
from research.distributional.arrows_gate_scoring import (  # noqa: E402
    load_target_entries, build_route, score_validator, COMPARATOR_SCALES_PATH,
)
from research.phase18_common import SEED, N_BOOT, provenance, write_json, notify  # noqa: E402
from validator import SynthesisValidator, ThermoChecker  # noqa: E402

N_PERM = 1000


def sgn(x, eps=1e-9):
    return 1 if x > eps else (-1 if x < -eps else 0)


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    # rank-based Mann-Whitney with ties
    allv = sorted((s, i) for i, s in enumerate(scores))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[allv[k][1]] = r
        i = j + 1
    rp = sum(r for r, l in zip(ranks, labels) if l == 1)
    n1, n0 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n0)


def weighted_auc(scores, labels, weights):
    """AUC with integer multiplicities (bootstrap); O(n log n) via sorting."""
    items = sorted(zip(scores, labels, weights))
    tot_pos = sum(w for _, l, w in items if l == 1)
    tot_neg = sum(w for _, l, w in items if l == 0)
    if tot_pos == 0 or tot_neg == 0:
        return None
    acc = 0.0
    neg_below = 0.0
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][0] == items[i][0]:
            j += 1
        grp_pos = sum(w for _, l, w in items[i:j + 1] if l == 1)
        grp_neg = sum(w for _, l, w in items[i:j + 1] if l == 0)
        acc += grp_pos * (neg_below + 0.5 * grp_neg)
        neg_below += grp_neg
        i = j + 1
    return acc / (tot_pos * tot_neg)


def hodge_weighted(nodes, edges, weights):
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n))
    div = np.zeros(n)
    for (a, b), y in edges.items():
        i, j, w = idx[a], idx[b], weights[(a, b)]
        L[i, i] += w; L[j, j] += w; L[i, j] -= w; L[j, i] -= w
        div[i] += w * y; div[j] -= w * y
    s = np.linalg.pinv(L) @ div
    rho = {e: abs(y - (s[idx[e[0]]] - s[idx[e[1]]])) for e, y in edges.items()}
    Y = np.array(list(edges.values()))
    W = np.array([weights[e] for e in edges])
    grad = np.array([s[idx[a]] - s[idx[b]] for (a, b) in edges])
    frac_nongrad = float(np.sum(W * (Y - grad) ** 2) / np.sum(W * Y ** 2)) if np.sum(W * Y ** 2) > 0 else None
    return {nd: float(s[idx[nd]]) for nd in nodes}, rho, frac_nongrad


def auc_with_ci(edge_list, score, label, rng):
    sc = [score[e] for e in edge_list]
    lb = [label[e] for e in edge_list]
    point = auc(sc, lb)
    nodes = sorted({a for a, _ in edge_list} | {b for _, b in edge_list})
    boots = []
    for _ in range(N_BOOT):
        drawn = defaultdict(int)
        for _k in range(len(nodes)):
            drawn[nodes[rng.randrange(len(nodes))]] += 1
        w = [drawn.get(a, 0) * drawn.get(b, 0) for a, b in edge_list]
        keep = [(s, l, ww) for s, l, ww in zip(sc, lb, w) if ww > 0]
        if keep:
            v = weighted_auc([k[0] for k in keep], [k[1] for k in keep], [k[2] for k in keep])
            if v is not None:
                boots.append(v)
    boots.sort()
    ci = [boots[int(0.025 * len(boots))], boots[min(int(0.975 * len(boots)), len(boots) - 1)]] if boots else [None, None]
    null = []
    for _ in range(N_PERM):
        perm = sc[:]
        rng.shuffle(perm)
        v = auc(perm, lb)
        if v is not None:
            null.append(v)
    p = (1 + sum(1 for v in null if v >= point)) / (1 + len(null)) if point is not None else None
    return {"auc": point, "ci95": ci, "n_boot_valid": len(boots), "perm_p_one_sided": p,
            "perm_null_mean": float(np.mean(null)) if null else None,
            "n_edges": len(edge_list), "n_disagree": sum(lb), "n_nodes": len(nodes)}


def main():
    prov = provenance()
    rng = random.Random(SEED)
    validator = load_validator(REPO_ROOT / "data/cache/mp_formula_set.pkl", REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    comparator = load_comparator(REPO_ROOT / "data/cache/mp_formula_set.pkl", REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    scales = json.loads((REPO_ROOT / COMPARATOR_SCALES_PATH).read_text())["scales"]
    with open(REPO_ROOT / "data/cache/mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)
    ranker = Ranker(formula_set, thermo, build_precursor_frequency(SYNTHESIS), validator_version=2)
    params = ComparatorParams(c3_fraction=0.17)

    entries, target = load_target_entries("YBCO")
    by_recipe = defaultdict(dict)  # recipe -> temp_str -> entry
    for e in entries:
        by_recipe[e["precursor_set"]][e["temp_str"]] = e
    routes, gates, r2, r4, r2ch, r4ch = {}, {}, {}, {}, {}, {}
    for rcp, temps in by_recipe.items():
        for ts, e in temps.items():
            route = build_route(target, e["precursors"], e["amounts"], e["temp_C"], e["atmosphere"])
            k = (rcp, ts)
            routes[k] = route
            g = comparator._gater._check_gates(route)
            gates[k] = all(g.values())
            bd = score_validator(validator, route, target)
            chans = {}
            if bd is not None:
                for ch in RUN3_CHECKS:
                    v = bd.get(ch)
                    if v is not None and bd.get(f"{ch}_gradeability") not in SynthesisValidator.SENTINEL_TAGS:
                        chans[ch] = v
            r2ch[k] = chans
            r2[k] = (sum(chans.values()) / len(chans)) if chans else None
            rv, rbd = ranker.score(route, target)
            r4[k] = rv
            r4ch[k] = {o: rbd.get(o) for o in OBJECTIVE_NAMES if rbd.get(o) is not None}
    print(f"{len(by_recipe)} recipes, {len(routes)} routes, gate pass {sum(gates.values())}", flush=True)

    recipes = sorted(by_recipe)
    Y, W, Ylab, d = {}, {}, {}, {}
    conflict = {"R2": {}, "R4": {}}
    agg = {"R2": {}, "R4": {}}
    n_pairs_shared = 0
    for a, b in combinations(recipes, 2):
        shared = sorted(set(by_recipe[a]) & set(by_recipe[b]))
        if not shared:
            continue
        n_pairs_shared += 1
        lab = [sgn(by_recipe[a][t]["wt_pct"] - by_recipe[b][t]["wt_pct"]) for t in shared]
        ylab = float(np.mean(lab))
        margins = []
        for t in shared:
            ka, kb = (a, t), (b, t)
            if gates[ka] and gates[kb]:
                m, _bd = comparator.compare(routes[ka], routes[kb], target, scales, params)
                margins.append(m)
        if margins:
            Y[(a, b)] = float(np.mean(margins))
            W[(a, b)] = len(margins)
            Ylab[(a, b)] = ylab
        # [E] channel conflict for scalar rewards
        for name, sc, chd in (("R2", r2, r2ch), ("R4", r4, r4ch)):
            confs, aggs = [], []
            for t in shared:
                ka, kb = (a, t), (b, t)
                if sc[ka] is None or sc[kb] is None:
                    continue
                s_agg = sgn(sc[ka] - sc[kb])
                aggs.append(sc[ka] - sc[kb])
                both = [c for c in chd[ka] if c in chd[kb]]
                if both:
                    opp = sum(1 for c in both if s_agg != 0 and sgn(chd[ka][c] - chd[kb][c]) == -s_agg)
                    confs.append(opp / len(both))
            if aggs:
                agg[name][(a, b)] = (float(np.mean(aggs)), ylab)
                if confs:
                    conflict[name][(a, b)] = float(np.mean(confs))

    nodes = sorted({x for e in Y for x in e})
    s, rho, frac_nongrad = hodge_weighted(nodes, Y, W)
    edges_lab = [e for e in Y if sgn(Ylab[e]) != 0]
    for e in edges_lab:
        d[e] = int(sgn(Y[e]) != sgn(Ylab[e]))
    res = auc_with_ci(edges_lab, rho, d, rng)
    passes = (res["auc"] is not None and res["auc"] > 0.5 and res["ci95"][0] is not None
              and res["ci95"][0] > 0.5 and res["perm_p_one_sided"] < 0.05)
    print(f"comparator graph: {len(nodes)} nodes, {len(Y)} edges, non-gradient energy {frac_nongrad}", flush=True)
    print(f"P-18B-1: AUC={res['auc']} CI={res['ci95']} perm p={res['perm_p_one_sided']} passes={passes}", flush=True)

    extra = {}
    for name in ("R2", "R4"):
        es = [e for e in conflict[name] if sgn(agg[name][e][1]) != 0]
        dd = {e: int(sgn(agg[name][e][0]) != sgn(agg[name][e][1])) for e in es}
        extra[name] = auc_with_ci(es, conflict[name], dd, random.Random(SEED + 1))
        extra[name]["disagreement_rate"] = (sum(dd.values()) / len(dd)) if dd else None
        print(f"[E] {name} channel-conflict AUC={extra[name]['auc']} CI={extra[name]['ci95']} "
              f"p={extra[name]['perm_p_one_sided']}", flush=True)

    # descriptive: comparator disagreement rate, rho by d
    rho_d1 = [rho[e] for e in edges_lab if d[e] == 1]
    rho_d0 = [rho[e] for e in edges_lab if d[e] == 0]
    doc = {**prov, "block": "18B", "comparator_version": COMPARATOR_VERSION,
           "method": {"lab_flow": "mean over shared temperatures of sign(yield_a,T - yield_b,T)",
                      "comparator_flow": "mean over shared, both-gate-passing temperatures of compare() margin",
                      "hodge": "weighted LS potential, w = #temperatures in the comparator flow",
                      "bootstrap": f"{N_BOOT} node (recipe) resamples, rho fixed at full-graph value",
                      "null": f"{N_PERM} shuffles of rho across edges"},
           "graph": {"n_recipes": len(recipes), "n_routes": len(routes), "n_routes_gate_pass": sum(gates.values()),
                     "n_recipe_pairs_sharing_a_temperature": n_pairs_shared,
                     "n_nodes": len(nodes), "n_edges_comparator_defined": len(Y),
                     "n_edges_nonzero_lab_sign": len(edges_lab),
                     "weighted_nongradient_energy_fraction": frac_nongrad},
           "comparator_disagreement_rate": sum(d.values()) / len(d) if d else None,
           "rho_mean_when_disagree": float(np.mean(rho_d1)) if rho_d1 else None,
           "rho_mean_when_agree": float(np.mean(rho_d0)) if rho_d0 else None,
           "P_18B_1": {"statement": "AUC of rho for predicting comparator-lab disagreement > 0.5, CI excluding "
                                    "0.5, beating the permutation null at p < 0.05.", **res, "passes": passes},
           "E_astral": {"computed": False,
                        "reason": "ASTRAL measured two recipes per target; each target gives one edge, the pooled "
                                  "graph is a forest with no cycles, every Hodge residual is 0 and the AUC is "
                                  "undefined."},
           "E_channel_conflict": extra,
           "per_edge": [{"a": a, "b": b, "Y": Y[(a, b)], "w": W[(a, b)], "Y_lab": Ylab[(a, b)],
                         "rho": rho[(a, b)], "d": d.get((a, b))} for (a, b) in Y]}
    p = write_json("curl_vs_lab.json", doc)
    print(f"-> {p}")
    notify(f"MIRA 18B done: AUC={res['auc']:.3f} CI={[round(x,3) for x in res['ci95']]} p={res['perm_p_one_sided']:.3f}")


if __name__ == "__main__":
    main()
