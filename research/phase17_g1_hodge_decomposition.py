#!/usr/bin/env python
"""
research/phase17_g1_hodge_decomposition.py -- Phase 17 Block G1, REAL
pre-registered graph (docs/phases/PHASE17_PREREG.md, "Block G" section),
per regular Claude's correction: the earlier ARROWS3-pair antisymmetry
check (research/phase17_g1_antisymmetry.py, results/phase17/g1_antisymmetry.json)
is G2 groundwork, NOT G1. G1's graph is:

  per target (ASTRAL 35 + held-out 40), the top-30 recipes of U(x),
  each represented by its most frequent full parsed route.

NODE SET: results/phase17/universe.jsonl (built in Block 0, already on
disk) gives every distinct recipe (canonicalised precursor SET only,
via precset_of) per target with its pooled sample count across all 13
models. Per target: top 30 by pooled_count, plus any recipe sampled
>=2 times by a single model not already in that top 30 (pre-reg rule,
verbatim).

FULL ROUTE RECONSTRUCTION: universe.jsonl's "recipe" field has no
amounts/temperature/atmosphere (precset_of only needs the formula set).
To get "the most frequent full parsed route" for each node, this script
re-scans every model's RAW completions file (results/phase17/inventory.json's
raw_completions_file, has_think_trace=true only -- the enriched/
structured_only_fallback files for corpus_sft/gdpo_v3_ckpt300/
phase12_ckpt300 on ASTRAL lack amounts+atmosphere entirely, confirmed by
inspection, so they contribute to universe.jsonl's frequency COUNT but
cannot supply a full-route surface form; Block A's regeneration is
filling exactly this gap for those three, but with 9/13 models already
supplying full `<think>`-trace completions across thousands of samples
per target, every top-30 node is expected to have >=1 full-route sample
from an unaffected model -- this script reports explicitly, per target,
any node that has NO full-route sample available, and drops it from the
graph rather than guessing one). No new generation.

EDGES: all pairs (i,j) with a defined margin. Y_ij = comparator margin
(current COMPARATOR_VERSION, existing scales, core/comparator.py::compare,
imported unedited) -- defined only when BOTH routes clear the comparator's
gates (score_channels not all-None for both); gate-failing pairs are
excluded from the graph, per the pre-reg's literal "all pairs with a
DEFINED margin", not assigned a +-1 sign (that +-1 gate-failure branch
belongs to comparator_rule()'s DECISION wrapper used elsewhere in R-f/
R-g/the G2 antisymmetry check, not to this continuous-margin graph).

CONTROLS (same edge sets, same nodes):
  1. R1 (checker, 10-channel original aggregate) and R2 (checker, 5-channel
     RUN3 aggregate) scalar differences: Y_ij = s(i) - s(j). Any scalar is
     a gradient flow by construction -- zero curl+harmonic is the
     methodology check.
  2. R4 (ranker v2) scalar difference, same control.
  3. R5_fixed: a NEW thin wrapper (fixed_channel_compare, below, in this
     file only -- core/comparator.py is NOT edited) that sums over the
     SAME channel set (SCORED_CHANNEL_NAMES) for every pair, with a
     channel missing on either side entering at percentile 0.5 (not
     excluded) -- per regular Claude's instruction. This makes the margin
     a pure per-node-potential difference again (percentile sum per node
     doesn't depend on the pair), so it is ALSO a gradient flow by
     construction; comparing its curl+harmonic (expected ~0) against the
     real R5 (expected >0) isolates per-pair channel-set variability as
     the curl source, exactly as the pre-reg's Block G rationale states.

DECOMPOSITION: standard combinatorial Hodge/HodgeRank split (Jiang, Lim,
Yao & Ye, Math. Programming 2011), computed exactly (not assumed zero
harmonic), via:
  1. grad: least-squares potential s minimizing sum_{(i,j) in E} (Y_ij -
     (s_i - s_j))^2, solved as L s = div(Y) with L the (unweighted, w=1
     for every present edge -- the pre-reg does not specify a weighting
     scheme) graph Laplacian, via pseudo-inverse (Laplacian has a 1-dim
     constant nullspace).
  2. residual r_ij = Y_ij - (s_i - s_j) (divergence-free by the normal
     equations' optimality condition).
  3. curl potential Phi over every triangle (i<j<k) whose all 3 edges
     are present in E, via the adjoint/curl operator C (rows=triangles,
     cols=edges; C[t,(i,j)]=+1, C[t,(j,k)]=+1, C[t,(i,k)]=-1, matching
     curl(Y)(i,j,k)=Y_ij+Y_jk-Y_ik=Y_ij+Y_jk+Y_ki): Phi* = argmin
     ||r - C^T Phi||^2 via np.linalg.lstsq(C.T, r). curl_component = C^T
     Phi*. harmonic = r - curl_component. This is the GENERAL solution,
     not an assumed zero -- it reduces to harmonic=0 automatically on a
     complete graph (the theorem the pre-reg cites), without needing to
     special-case completeness.
  Reported per target and pooled (ASTRAL, held-out, overall): energy
  fractions ||grad||^2/||Y||^2, ||curl||^2/||Y||^2, ||harmonic||^2/||Y||^2
  (pooled = sum of squared components over every target's edges, divided
  by the summed ||Y||^2 -- an edge-count-weighted pool, not a mean of
  per-target fractions).

CYCLIC TRIADS: among R5's real edges, a triangle (i,j,k) is a cyclic
triad iff sign(Y_ij), sign(Y_jk), sign(Y_ki) [Y_ki = -Y_ik] are all the
same non-zero sign (i beats j beats k beats i, or the reverse) -- an
intransitive triple no scalar ranking can represent even locally.

CHANNEL-SET / CURL PREDICTION: per triangle (R5 only), compare whether
its 3 edges share an identical gradeable-channel-set (recorded from
compare()'s own breakdown, `{c}_gradeable` flags) against the raw
per-triangle curl magnitude |Y_ij+Y_jk+Y_ki|; report mean curl on
same-channel-set vs different-channel-set triangles and their ratio
(pre-reg prediction: >=1.5x on different-channel-set triangles).

NO EDIT to core/comparator.py, core/ranker.py, validator.py, or
research/distributional/arrows_gate_scoring.py.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_g1_hodge_decomposition.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comparator import (  # noqa: E402
    Comparator, ComparatorParams, SCORED_CHANNEL_NAMES, _percentile_rank, load_comparator,
)
from core.ranker import Ranker, build_precursor_frequency  # noqa: E402
from core.reward import RUN3_CHECKS, load_validator, parse_completion  # noqa: E402
from research.ranker_capacity_rssft import SYNTHESIS  # noqa: E402
from research.distributional.arrows_gate_scoring import COMPARATOR_SCALES_PATH  # noqa: E402
from research.support_growth_by_checkpoint import precset_of  # noqa: E402
from validator import SynthesisValidator, ThermoChecker  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "hodge_comparator.json"
UNIVERSE_PATH = REPO_ROOT / "results" / "phase17" / "universe.jsonl"
INVENTORY_PATH = REPO_ROOT / "results" / "phase17" / "inventory.json"
ASTRAL_SET_PATH = REPO_ROOT / "results" / "astral_validation_set.json"
HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
TOP_N = 30
MIN_MODEL_COUNT_ALWAYS_INCLUDE = 2
C3_FRACTION = 0.17  # matches comparator_rule's own ComparatorParams, arrows_gate_scoring.py:219


# ---------------------------------------------------------------------------
# R5_fixed: fixed-channel-set comparator wrapper (NEW, this file only)
# ---------------------------------------------------------------------------

def fixed_channel_compare(comparator: Comparator, a, b, target_formula: str, scales: dict, params) -> float | None:
    """Like compare(), but sums over the FULL SCORED_CHANNEL_NAMES list for
    every pair (not just the channels gradeable for both routes); a missing
    channel enters at percentile 0.5 rather than being excluded. Returns
    None if either route fails gates entirely (score_channels all-None),
    matching compare()'s own gate behaviour. This makes the margin a pure
    per-node-potential difference: f(c, x) = percentile(x's raw value) if
    gradeable else 0.5, margin(a,b) = sum_c f(c,a) - sum_c f(c,b) -- a
    gradient flow by construction, the point of the control."""
    sa = comparator.score_channels(a, target_formula, params)
    sb = comparator.score_channels(b, target_formula, params)
    if all(v is None for v in sa.values()) or all(v is None for v in sb.values()):
        return None
    margin = 0.0
    for c in SCORED_CHANNEL_NAMES:
        sorted_vals = scales.get(c)
        va, vb = sa.get(c), sb.get(c)
        pa = _percentile_rank(va, sorted_vals) if (va is not None and sorted_vals) else 0.5
        pb = _percentile_rank(vb, sorted_vals) if (vb is not None and sorted_vals) else 0.5
        margin += pa - pb
    return margin


def gate_pass(comparator: Comparator, route, target_formula: str, params) -> bool:
    if route is None:
        return False
    sc = comparator.score_channels(route, target_formula, params)
    return not all(v is None for v in sc.values())


# ---------------------------------------------------------------------------
# Scalars for R1 (10ch), R2 (5ch), R4 (ranker)
# ---------------------------------------------------------------------------

ALL_10_CHECKS = (
    "stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist",
    "operation_order", "temperature_plausible", "thermodynamic_favorable",
    "target_stability", "chempot_atmosphere", "target_match",
)


def checker_scalar_generic(bd: dict | None, checks: tuple[str, ...]) -> float | None:
    if bd is None:
        return None
    total, n_grad = 0.0, 0
    for ch in checks:
        v = bd.get(ch)
        tag = bd.get(f"{ch}_gradeability")
        if v is not None and tag not in SynthesisValidator.SENTINEL_TAGS:
            total += v
            n_grad += 1
    return total if n_grad else None


# ---------------------------------------------------------------------------
# Universe / top-30 node set
# ---------------------------------------------------------------------------

def load_universe() -> dict[str, list[dict]]:
    by_target = defaultdict(list)
    with UNIVERSE_PATH.open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                by_target[d["target"]].append(d)
    return by_target


def top30_node_keys(records: list[dict]) -> list[tuple]:
    records_sorted = sorted(records, key=lambda d: -d["pooled_count"])
    top = records_sorted[:TOP_N]
    keys = {tuple(sorted(d["recipe"])) for d in top}
    for d in records_sorted:
        key = tuple(sorted(d["recipe"]))
        if key in keys:
            continue
        if max(d.get("per_model_sample_counts", {}).values(), default=0) >= MIN_MODEL_COUNT_ALWAYS_INCLUDE:
            keys.add(key)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Full-route reconstruction: scan raw completions, find modal surface form
# ---------------------------------------------------------------------------

def surface_form_key(route) -> tuple:
    precs = tuple(sorted((p.formula, round(p.amount, 3)) for p in route.precursors))
    ops = tuple(
        (o.type,
         tuple(round(t, 0) for t in o.conditions.heating_temperature),
         tuple(round(t, 2) for t in o.conditions.heating_time),
         tuple(o.conditions.heating_atmosphere),
         o.conditions.atmosphere)
        for o in route.operations
    )
    return (precs, ops)


def raw_files_for_target_set(inventory: dict, target_set: str) -> list[tuple[str, Path]]:
    out = []
    for model, sets in inventory["raw_completions"].items():
        info = sets.get(target_set, {})
        p = info.get("raw_completions_file")
        if p:
            out.append((model, REPO_ROOT / p))
    return out


def build_modal_routes(targets: list[str], target_set: str, node_keys_by_target: dict[str, list[tuple]],
                        inventory: dict) -> dict[str, dict[tuple, object]]:
    """Returns {target: {node_key: route}} for whichever nodes a full-route
    sample was found for (missing nodes are simply absent -- reported, not
    guessed)."""
    targets_set = set(targets)
    needed_keys = {t: set(node_keys_by_target[t]) for t in targets}
    surface_counts: dict[str, dict[tuple, dict[tuple, int]]] = {t: defaultdict(lambda: defaultdict(int)) for t in targets}
    surface_example: dict[str, dict[tuple, dict[tuple, object]]] = {t: defaultdict(dict) for t in targets}

    for model, path in raw_files_for_target_set(inventory, target_set):
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                target = rec.get("target")
                if target not in targets_set:
                    continue
                try:
                    route = parse_completion(rec["completion"], target)
                except Exception:
                    continue
                key = precset_of(route)
                if key not in needed_keys[target]:
                    continue
                sform = surface_form_key(route)
                sc = surface_counts[target][key]
                sc[sform] += 1
                if sform not in surface_example[target][key]:
                    surface_example[target][key][sform] = route

    modal: dict[str, dict[tuple, object]] = {}
    missing: dict[str, list[tuple]] = {}
    for t in targets:
        modal[t] = {}
        missing[t] = []
        for key in node_keys_by_target[t]:
            sc = surface_counts[t].get(key)
            if not sc:
                missing[t].append(key)
                continue
            best_form = max(sc.items(), key=lambda kv: kv[1])[0]
            modal[t][key] = surface_example[t][key][best_form]
    return modal, missing


# ---------------------------------------------------------------------------
# Hodge decomposition
# ---------------------------------------------------------------------------

def hodge_decompose(n: int, edges: dict[tuple[int, int], float]) -> dict:
    """edges: {(i,j): Y_ij} for i<j. Returns grad/curl/harmonic per-edge
    arrays (same keys) plus energy totals."""
    edge_list = sorted(edges.keys())
    m = len(edge_list)
    if m == 0:
        return {"m": 0}
    idx_of = {e: k for k, e in enumerate(edge_list)}
    Y = np.array([edges[e] for e in edge_list], dtype=float)

    # Laplacian (unweighted), div(Y)
    L = np.zeros((n, n))
    div = np.zeros(n)
    for (i, j), k in idx_of.items():
        y = Y[k]
        L[i, i] += 1; L[j, j] += 1
        L[i, j] -= 1; L[j, i] -= 1
        div[i] += y     # edge i->j carries +Y_ij out of i
        div[j] -= y
    s = np.linalg.pinv(L) @ div
    grad = np.array([s[i] - s[j] for (i, j) in edge_list])
    resid = Y - grad

    # triangles with all 3 edges present
    adj = defaultdict(set)
    for (i, j) in edge_list:
        adj[i].add(j); adj[j].add(i)
    triangles = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k > j:
                    triangles.append((i, j, k))
    triangles.sort()

    curl_component = np.zeros(m)
    curl_vals = {}
    if triangles:
        t = len(triangles)
        C = np.zeros((t, m))
        for ti, (i, j, k) in enumerate(triangles):
            C[ti, idx_of[(i, j)]] += 1
            C[ti, idx_of[(j, k)]] += 1
            C[ti, idx_of[(i, k)]] -= 1
        curl_Y = C @ Y
        for ti, tri in enumerate(triangles):
            curl_vals[tri] = float(curl_Y[ti])
        phi, *_ = np.linalg.lstsq(C.T, resid, rcond=None)
        curl_component = C.T @ phi
    harmonic = resid - curl_component

    def energy(v):
        return float(np.sum(v ** 2))

    e_total = energy(Y)
    out = {
        "m_edges": m, "n_triangles": len(triangles),
        "energy_total": e_total,
        "energy_grad": energy(grad), "energy_curl": energy(curl_component),
        "energy_harmonic": energy(harmonic),
        "frac_grad": (energy(grad) / e_total) if e_total > 1e-12 else None,
        "frac_curl": (energy(curl_component) / e_total) if e_total > 1e-12 else None,
        "frac_harmonic": (energy(harmonic) / e_total) if e_total > 1e-12 else None,
        "edge_list": edge_list, "Y": Y, "grad": grad, "resid": resid,
        "curl_component": curl_component, "harmonic": harmonic,
        "triangles": triangles, "triangle_curl_raw": curl_vals,
    }
    return out


def cyclic_triads(triangles: list[tuple], Y_lookup) -> dict:
    n_cyc, n_checked = 0, 0
    for (i, j, k) in triangles:
        yij = Y_lookup(i, j)
        yjk = Y_lookup(j, k)
        yki = -Y_lookup(i, k)
        if yij is None or yjk is None or yki is None:
            continue
        n_checked += 1
        if (yij > 0 and yjk > 0 and yki > 0) or (yij < 0 and yjk < 0 and yki < 0):
            n_cyc += 1
    return {"n_triangles_checked": n_checked, "n_cyclic": n_cyc,
            "frac_cyclic": (n_cyc / n_checked) if n_checked else None}


def main():
    print("loading validator, comparator, ranker, universe, inventory...", flush=True)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    validator = load_validator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                               REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    comparator = load_comparator(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl",
                                 REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)
    scales = json.loads(COMPARATOR_SCALES_PATH.read_text())["scales"]
    freq = build_precursor_frequency(SYNTHESIS)
    ranker = Ranker(formula_set, thermo, freq, validator_version=2)
    params = ComparatorParams(c3_fraction=C3_FRACTION)

    universe = load_universe()
    inventory = json.loads(INVENTORY_PATH.read_text())
    astral_targets = [d["target"] for d in json.loads(ASTRAL_SET_PATH.read_text())["targets"]]
    heldout_targets = json.loads(HELDOUT_TARGETS_PATH.read_text())

    node_keys_by_target = {t: top30_node_keys(universe[t]) for t in universe if t in universe}

    astral_in_universe = [t for t in astral_targets if t in universe]
    heldout_in_universe = [t for t in heldout_targets if t in universe]
    print(f"ASTRAL targets in universe: {len(astral_in_universe)}/35; "
          f"held-out targets in universe: {len(heldout_in_universe)}/40", flush=True)

    modal_astral, missing_astral = build_modal_routes(astral_in_universe, "astral", node_keys_by_target, inventory)
    modal_heldout, missing_heldout = build_modal_routes(heldout_in_universe, "heldout", node_keys_by_target, inventory)
    modal_routes = {**modal_astral, **modal_heldout}
    missing_nodes = {**missing_astral, **missing_heldout}

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "G1",
        "graph_spec": "per target (ASTRAL 35 + held-out 40), top-30 recipes of U(x), each "
                      "represented by its most frequent full parsed route (universe.jsonl's "
                      "node set + a re-scan of raw <think>-trace completions for the modal "
                      "surface form); edges = all pairs with a defined comparator margin.",
        "comparator_version": None, "edge_weighting": "uniform (w=1 per present edge; "
                      "the pre-reg does not specify a weighting scheme)",
        "c3_fraction": C3_FRACTION,
        "per_target": {},
    }

    pooled = {"ASTRAL": defaultdict(lambda: defaultdict(float)), "heldout": defaultdict(lambda: defaultdict(float)),
              "overall": defaultdict(lambda: defaultdict(float))}
    pooled_cyc = {"ASTRAL": [0, 0], "heldout": [0, 0], "overall": [0, 0]}
    curl_by_channelset = {"same": [], "different": []}

    all_targets = astral_in_universe + heldout_in_universe
    debug_limit = int(os.environ.get("PHASE17_G1_LIMIT", "0") or "0")
    if debug_limit:
        all_targets = all_targets[:debug_limit]
        print(f"[debug] PHASE17_G1_LIMIT set -- restricting to first {debug_limit} targets", flush=True)
    for ti, target in enumerate(all_targets):
        which = "ASTRAL" if target in astral_in_universe else "heldout"
        nodes = {k: modal_routes[target][k] for k in node_keys_by_target[target] if k in modal_routes.get(target, {})}
        node_keys = list(nodes.keys())
        n = len(node_keys)
        if ti % 10 == 0:
            print(f"[{ti+1}/{len(all_targets)}] {target} ({which}): {n}/{len(node_keys_by_target[target])} "
                  f"nodes have a full-route sample", flush=True)
        if n < 3:
            report["per_target"][target] = {"which": which, "n_nodes": n,
                "note": "fewer than 3 nodes with a reconstructable full route -- skipped (no triangle possible)"}
            continue

        routes = [nodes[k] for k in node_keys]
        bds = [validator.validate(r, r.target_formula)[1] for r in routes]
        s1 = [checker_scalar_generic(bd, ALL_10_CHECKS) for bd in bds]
        s2 = [checker_scalar_generic(bd, RUN3_CHECKS) for bd in bds]
        s4 = [ranker.score(r, r.target_formula)[0] for r in routes]

        edges_r1, edges_r2, edges_r4, edges_r5, edges_r5fixed = {}, {}, {}, {}, {}
        channel_sets = {}
        gate_ok = [gate_pass(comparator, r, r.target_formula, params) for r in routes]
        for a, b in combinations(range(n), 2):
            if s1[a] is not None and s1[b] is not None:
                edges_r1[(a, b)] = s1[a] - s1[b]
            if s2[a] is not None and s2[b] is not None:
                edges_r2[(a, b)] = s2[a] - s2[b]
            if s4[a] is not None and s4[b] is not None:
                edges_r4[(a, b)] = s4[a] - s4[b]
            if gate_ok[a] and gate_ok[b]:
                margin, bd = comparator.compare(routes[a], routes[b], routes[a].target_formula, scales, params)
                edges_r5[(a, b)] = margin
                channel_sets[(a, b)] = frozenset(c for c in SCORED_CHANNEL_NAMES if bd.get(f"{c}_gradeable"))
                mf = fixed_channel_compare(comparator, routes[a], routes[b], routes[a].target_formula, scales, params)
                if mf is not None:
                    edges_r5fixed[(a, b)] = mf

        decomp = {
            "R1_checker_10ch": hodge_decompose(n, edges_r1),
            "R2_checker_5ch": hodge_decompose(n, edges_r2),
            "R4_ranker": hodge_decompose(n, edges_r4),
            "R5_comparator": hodge_decompose(n, edges_r5),
            "R5_fixed_channel_wrapper": hodge_decompose(n, edges_r5fixed),
        }

        def ylookup_for(edges):
            return lambda i, j: edges.get((i, j)) if i < j else (-edges.get((j, i)) if (j, i) in edges else None)

        cyc = cyclic_triads(decomp["R5_comparator"].get("triangles", []), ylookup_for(edges_r5))

        for key, which_scopes in [("ASTRAL", ["ASTRAL", "overall"]), ("heldout", ["heldout", "overall"])]:
            if which != key:
                continue
            for scope in which_scopes:
                pooled_cyc[scope][0] += cyc["n_cyclic"] or 0
                pooled_cyc[scope][1] += cyc["n_triangles_checked"] or 0
                for reward_id, d in decomp.items():
                    if d.get("m_edges", 0) == 0:
                        continue
                    pooled[scope][reward_id]["energy_total"] += d["energy_total"]
                    pooled[scope][reward_id]["energy_grad"] += d["energy_grad"]
                    pooled[scope][reward_id]["energy_curl"] += d["energy_curl"]
                    pooled[scope][reward_id]["energy_harmonic"] += d["energy_harmonic"]

        for (i, j, k), raw_curl in decomp["R5_comparator"].get("triangle_curl_raw", {}).items():
            csets = [channel_sets.get((min(a,b), max(a,b))) for a, b in [(i, j), (j, k), (i, k)]]
            if any(c is None for c in csets):
                continue
            group = "same" if len(set(csets)) == 1 else "different"
            curl_by_channelset[group].append(abs(raw_curl))

        report["per_target"][target] = {
            "which": which, "n_nodes": n,
            "missing_nodes": len(missing_nodes.get(target, [])),
            "cyclic_triads": cyc,
            **{k: {kk: vv for kk, vv in v.items() if kk not in
                   ("edge_list", "Y", "grad", "resid", "curl_component", "harmonic", "triangles", "triangle_curl_raw")}
               for k, v in decomp.items()},
        }

    def summarize_pooled(scope):
        out = {}
        for reward_id, acc in pooled[scope].items():
            et = acc["energy_total"]
            out[reward_id] = {
                "energy_total": et,
                "frac_grad": acc["energy_grad"] / et if et > 1e-12 else None,
                "frac_curl": acc["energy_curl"] / et if et > 1e-12 else None,
                "frac_harmonic": acc["energy_harmonic"] / et if et > 1e-12 else None,
            }
        n_cyc, n_check = pooled_cyc[scope]
        out["cyclic_triads"] = {"n_cyclic": n_cyc, "n_checked": n_check,
                                 "frac_cyclic": (n_cyc / n_check) if n_check else None}
        return out

    report["pooled"] = {"ASTRAL": summarize_pooled("ASTRAL"), "heldout": summarize_pooled("heldout"),
                        "overall": summarize_pooled("overall")}

    mean_same = sum(curl_by_channelset["same"]) / len(curl_by_channelset["same"]) if curl_by_channelset["same"] else None
    mean_diff = sum(curl_by_channelset["different"]) / len(curl_by_channelset["different"]) if curl_by_channelset["different"] else None
    same_is_numerically_zero = mean_same is not None and mean_same < 1e-9
    report["channel_set_curl_prediction"] = {
        "n_triangles_same_channel_set": len(curl_by_channelset["same"]),
        "n_triangles_different_channel_set": len(curl_by_channelset["different"]),
        "mean_abs_curl_same_channel_set": mean_same,
        "mean_abs_curl_different_channel_set": mean_diff,
        "ratio_diff_over_same": (mean_diff / mean_same) if (mean_same and mean_same > 1e-9) else
                                 ("undefined -- same-channel-set curl is numerically zero (<1e-9); "
                                  "the entire curl signal is attributable to channel-set heterogeneity "
                                  "across the triangle's three comparisons" if same_is_numerically_zero else None),
        "same_channel_set_curl_is_numerically_zero": same_is_numerically_zero,
        "prereg_prediction": ">= 1.5x higher on different-channel-set triangles",
    }

    report["missing_nodes_by_target"] = {t: len(v) for t, v in missing_nodes.items() if v}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["pooled"], indent=1), flush=True)
    print(json.dumps(report["channel_set_curl_prediction"], indent=1), flush=True)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
