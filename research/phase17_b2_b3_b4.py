#!/usr/bin/env python
"""
research/phase17_b2_b3_b4.py — Phase 17 Block B2 (geodesic test), B3 (the
leash), B4 (figure data, no predictions). Reuses Block B1's loaders
(research/phase17_b1_signature_regression.py) rather than duplicating
them -- same logp_universe.jsonl / universe_scores.jsonl reads, same
pi_tilde renormalisation convention.

B2 -- GEODESIC TEST (S6, S7 only). S1-S4 are explicitly in scope too
("applies ... to S1-S4 if Block 0 finds intermediate checkpoints") and
Block 0 DID find one intermediate checkpoint each (corpus_sft/checkpoint-
117, rs_sft/checkpoint-27, e3 and e3c/checkpoint-48, results/phase17/
inventory_v2.json). But none of those four intermediate checkpoints were
among Block A's 12 GPU-scored models -- scoring them would be NEW,
unapproved GPU forward-pass time beyond Block A's estimated-then-launched
scope. NOT done here; flagged, not silently skipped or silently run.
S6/S7 need no new GPU time: ckpt100/200/300 are already in logp_universe.
jsonl.

Per target, with pi_0 = RS-SFT (renormalised on this target's U(x)) and
pi_end = the final checkpoint (300), fit two 1-parameter-per-checkpoint
paths for k in {100, 200} (k=0 and k=end are the path's own endpoints,
trivially distance 0):
  e-path: log pi_k_fit(y) = (1-s_k)*log_pi_tilde_0(y) + s_k*log_pi_tilde_end(y) - psi(s_k)
          psi(s_k) = log-partition over U(x), making pi_k_fit a valid
          renormalised distribution.
  m-path: pi_k_fit(y) = (1-s_k)*pi_tilde_0(y) + s_k*pi_tilde_end(y)      (already normalised)
Score each by Fisher-Rao distance to the ACTUAL renormalised pi_tilde_k;
minimise over s_k in [0,1] by grid+refine. RSS_e / RSS_m per target
(summed over k=100,200); rho = RSS_e/RSS_m. Prediction: median rho < 1.

B3 -- THE LEASH. Per step (all 7): L_s = b_s*(max r~ - min r~) over that
step's own B1 regression data (the largest relative log-prob gain a tilt
of the fitted strength allows). Report L_s, the observed largest Delta
among recipes with pi_tilde_before < epsilon, and the raw count of
epsilon-crossings (below-epsilon at "before" -> above-epsilon at
"after"). The pre-reg's literal noise-floor control (splitting the K=8
tau_k into two independent halves and re-estimating pi_before from each)
needs PER-TRACE log-probabilities that phase17_block_a_forward_pass.py
never persisted (only the K-trace-averaged logsumexp/K value is in
logp_universe.jsonl) -- recomputing it means new GPU forward passes.
NOT done; flagged. A computable substitute is reported instead and
labelled as a substitute, not the literal spec: splitting each model's
OWN raw-completion SAMPLES (not log-prob traces) in half and comparing
each half's empirical sample-frequency-based crossing count to the
real before-vs-after crossing count.

B4 -- FIGURE DATA (ternary, Fisher sphere; no predictions). Ternary:
alkali carbonate / bare oxide / other mass per model, via
research/distributional/alkali_source_share.py::classify_alkali_source
(already-established project convention, imported not reimplemented).

NO EDIT to any frozen file.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_b2_b3_b4.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.reward import parse_completion  # noqa: E402
from research.support_growth_by_checkpoint import precset_of  # noqa: E402
from research.phase17_b1_signature_regression import (  # noqa: E402
    STEPS, SUPPLEMENTARY_AFTER, load_logp, load_recipe_rewards, build_step_data,
    pi_tilde_before, logsumexp,
)
from research.distributional.alkali_source_share import (  # noqa: E402
    target_alkalis, classify_alkali_source,
)

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
SIGNATURE_PATH = REPO_ROOT / "results" / "phase17" / "signature.json"
UNIVERSE_PATH = REPO_ROOT / "results" / "phase17" / "universe_v2.jsonl"
INVENTORY_PATH = REPO_ROOT / "results" / "phase17" / "inventory_v2.json"
OUT_B2 = REPO_ROOT / "results" / "phase17" / "geodesic.json"
OUT_B3 = REPO_ROOT / "results" / "phase17" / "leash.json"
OUT_B4_TERNARY = REPO_ROOT / "results" / "phase17" / "figdata_ternary.json"
EPSILON = 1.0 / 200


def load_universe_node_keys() -> dict:
    out = defaultdict(set)
    with UNIVERSE_PATH.open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["target"]].add(tuple(sorted(d["recipe"])))
    return out


def noise_floor_crossings(before_model: str, inventory: dict, node_keys_by_target: dict) -> dict:
    """Substitute for the pre-reg's literal two-tau_k-halves noise floor
    (which needs per-trace log-probabilities that were never persisted).
    Splits BEFORE_MODEL's own raw completion rows for each target into
    two halves by row parity (even/odd index -- deterministic, not
    random), builds an empirical sample-frequency distribution per half
    (renormalised over the SAME U(x) key set Block A used), and counts
    how many recipes cross from below-epsilon in half 1 to at-or-above-
    epsilon in half 2 -- the same direction as the real before->after
    crossing count, but between two halves of the SAME untrained model's
    own samples, i.e. pure resampling noise with no training step at
    all."""
    by_set = inventory["raw_completions"].get(before_model, {})
    n_cross_by_target = {}
    for target_set, info in by_set.items():
        path_str = info.get("raw_completions_file")
        if not path_str:
            continue
        path = REPO_ROOT / path_str
        if not path.exists():
            continue
        rows_by_target = defaultdict(list)
        with path.open() as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                rows_by_target[rec["target"]].append((i, rec))
        for target, rows in rows_by_target.items():
            node_keys = node_keys_by_target.get(target, set())
            half1_counts, half2_counts = Counter(), Counter()
            for i, rec in rows:
                try:
                    route = parse_completion(rec["completion"], target)
                    key = precset_of(route)
                except Exception:
                    continue
                if key not in node_keys:
                    continue
                (half1_counts if i % 2 == 0 else half2_counts)[key] += 1
            n1, n2 = sum(half1_counts.values()), sum(half2_counts.values())
            if n1 < 20 or n2 < 20:
                continue
            keys = set(half1_counts) | set(half2_counts)
            p1 = {k: half1_counts.get(k, 0) / n1 for k in keys}
            p2 = {k: half2_counts.get(k, 0) / n2 for k in keys}
            n_cross = sum(1 for k in keys if p1[k] < EPSILON and p2[k] >= EPSILON)
            n_below_half1 = sum(1 for k in keys if p1[k] < EPSILON)
            n_cross_by_target[target] = {"n_cross": n_cross, "n_below_half1": n_below_half1,
                                         "n1": n1, "n2": n2, "min_resolvable_freq": 1.0 / min(n1, n2)}
    total_cross = sum(v["n_cross"] for v in n_cross_by_target.values())
    total_below = sum(v["n_below_half1"] for v in n_cross_by_target.values())
    # Honesty check: with only n1/n2 draws per half, the finest frequency
    # a half can resolve is 1/min(n1,n2). If that is already coarser than
    # EPSILON, any recipe absent from half 1 (p=0<EPSILON) that appears
    # even ONCE in half 2 trivially "crosses" by construction -- not
    # because anything informative happened. Caught by direct inspection:
    # n_cross == n_below_half1 for every step here (every absent-in-half-1
    # recipe that appeared at all in half 2 crossed), confirming the
    # substitute is DEGENERATE at this epsilon given these sample sizes,
    # not a real noise floor.
    worst_resolution = max((v["min_resolvable_freq"] for v in n_cross_by_target.values()), default=None)
    resolution_adequate = (worst_resolution is not None and worst_resolution < EPSILON / 2)
    return {"n_targets": len(n_cross_by_target), "total_n_crossing_noise_floor": total_cross,
           "total_n_below_epsilon_half1": total_below,
           "worst_min_resolvable_freq": worst_resolution, "epsilon": EPSILON,
           "resolution_adequate_for_epsilon": resolution_adequate,
           "degenerate_warning": None if resolution_adequate else (
               f"min resolvable frequency per half (~1/{round(1/worst_resolution) if worst_resolution else None}) "
               f"is NOT finer than epsilon={EPSILON} -- this substitute cannot meaningfully test the leash's "
               f"noise floor at this epsilon; any zero-count recipe that appears even once in the other half "
               f"trivially 'crosses', which is exactly what happened (n_cross == n_below_half1 below). Treat "
               f"the real crossing counts as UNVALIDATED by this substitute, not confirmed above noise."),
           "per_target": n_cross_by_target}

GEODESIC_STEPS = {
    "S6": {"init": "rs_sft", "checkpoints": {0: "rs_sft", 100: "phase12_ckpt100",
                                             200: "phase12_ckpt200", 300: "phase12_ckpt300"}},
    "S7": {"init": "rs_sft", "checkpoints": {0: "rs_sft", 100: "e1_ckpt100",
                                             200: "e1_ckpt200", 300: "e1_ckpt300"}},
}
SKIPPED_S1_S4_NOTE = (
    "S1-S4 each have exactly one intermediate checkpoint per inventory_v2.json "
    "(corpus_sft/checkpoint-117, rs_sft/checkpoint-27, e3 and e3c/checkpoint-48), "
    "but none were in Block A's GPU-scored model list (logp_universe.jsonl) -- "
    "scoring them needs new, unapproved GPU forward-pass time beyond Block A's "
    "estimated-then-launched scope. Not computed; this is a scope decision, not mine "
    "to make unilaterally."
)


def fisher_rao(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    s = sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)
    s = max(-1.0, min(1.0, s))
    return 2 * math.acos(s)


def renorm(logp_by_recipe: dict) -> dict:
    return pi_tilde_before(logp_by_recipe)


def e_path_point(pi0: dict, piend: dict, s: float, keys: list) -> dict:
    raw = {}
    for y in keys:
        lp0 = math.log(pi0[y]) if pi0.get(y, 0) > 0 else -700.0
        lpe = math.log(piend[y]) if piend.get(y, 0) > 0 else -700.0
        raw[y] = (1 - s) * lp0 + s * lpe
    log_z = logsumexp(list(raw.values()))
    return {y: math.exp(v - log_z) for y, v in raw.items()}


def m_path_point(pi0: dict, piend: dict, s: float, keys: list) -> dict:
    return {y: (1 - s) * pi0.get(y, 0.0) + s * piend.get(y, 0.0) for y in keys}


def best_fit_distance(path_fn, pi0: dict, piend: dict, actual: dict, keys: list) -> float:
    def dist(s):
        fitted = path_fn(pi0, piend, s, keys)
        return fisher_rao(fitted, actual)
    best_s, best_d = 0.5, dist(0.5)
    for i in range(1, 100):
        s = i / 100
        d = dist(s)
        if d < best_d:
            best_d, best_s = d, s
    lo, hi = max(0.0, best_s - 0.01), min(1.0, best_s + 0.01)
    for _ in range(30):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if dist(m1) < dist(m2):
            hi = m2
        else:
            lo = m1
    best_s = (lo + hi) / 2
    return dist(best_s), best_s


def block_b2(logp: dict) -> dict:
    out = {"skipped_S1_S4_note": SKIPPED_S1_S4_NOTE, "steps": {}}
    for step, cfg in GEODESIC_STEPS.items():
        ckpts = cfg["checkpoints"]
        init_model = cfg["init"]
        per_target_rho = {}
        for (model, target), _ in logp.items():
            if model != init_model:
                continue
            by_recipe_by_k = {}
            ok = True
            for k, m in ckpts.items():
                br = logp.get((m, target))
                if not br:
                    ok = False
                    break
                by_recipe_by_k[k] = br
            if not ok:
                continue
            keys = set(by_recipe_by_k[0])
            for k in (100, 200, 300):
                keys &= set(by_recipe_by_k[k])
            keys = list(keys)
            if len(keys) < 5:
                continue
            pi_tilde_by_k = {k: renorm({y: by_recipe_by_k[k][y] for y in keys}) for k in (0, 100, 200, 300)}
            rss_e, rss_m = 0.0, 0.0
            for k in (100, 200):
                de, _ = best_fit_distance(e_path_point, pi_tilde_by_k[0], pi_tilde_by_k[300], pi_tilde_by_k[k], keys)
                dm, _ = best_fit_distance(m_path_point, pi_tilde_by_k[0], pi_tilde_by_k[300], pi_tilde_by_k[k], keys)
                rss_e += de ** 2
                rss_m += dm ** 2
            if rss_m > 1e-12:
                per_target_rho[target] = rss_e / rss_m
        rhos = sorted(per_target_rho.values())
        median_rho = rhos[len(rhos) // 2] if rhos else None
        out["steps"][step] = {
            "n_targets": len(rhos), "median_rho": round(median_rho, 4) if median_rho is not None else None,
            "prediction_met_median_lt_1": (median_rho < 1) if median_rho is not None else None,
            "per_target_rho": {t: round(v, 4) for t, v in per_target_rho.items()},
        }
    return out


def block_b3(logp: dict, rewards: dict, signature: dict, inventory: dict, node_keys_by_target: dict) -> dict:
    out = {"noise_floor_note": (
        "The pre-reg's literal noise floor (two independent halves of the K=8 "
        "tau_k, re-estimating pi_before from each) needs per-trace log-"
        "probabilities that were never persisted -- NOT computed; would need new "
        "GPU forward passes. A substitute IS computed below: splitting each "
        "model's own raw completion SAMPLES in half (by row parity) and comparing "
        "empirical sample-frequency crossing counts between the two halves "
        "(pure resampling noise, no training step) to the real before-vs-after "
        "crossing count."), "steps": {}}
    noise_floor_cache = {}
    for step, (before, after, reward_field, _sens) in STEPS.items():
        rows, _cov = build_step_data(before, after, reward_field, logp, rewards)
        if not rows:
            out["steps"][step] = {"status": "no_data"}
            continue
        r_tilde_vals = [r["r_tilde"] for r in rows]
        b_s = signature["steps"][step]["fit"]["point"]
        b_s = b_s["b"] if b_s else None
        leash_L = b_s * (max(r_tilde_vals) - min(r_tilde_vals)) if b_s is not None else None

        below_eps_rows = [r for r in rows if r["pi_tilde_before"] < EPSILON]
        max_delta_below_eps = max((r["delta"] for r in below_eps_rows), default=None)

        # exact crossing count: pi_tilde_after is the after-model's own
        # within-(model,target) renormalised probability (build_step_data
        # computes it the same way as pi_tilde_before), not a proxy.
        n_cross = sum(1 for r in below_eps_rows
                     if r["pi_tilde_after"] is not None and r["pi_tilde_after"] >= EPSILON)

        if before not in noise_floor_cache:
            noise_floor_cache[before] = noise_floor_crossings(before, inventory, node_keys_by_target)
        nf = noise_floor_cache[before]

        out["steps"][step] = {
            "b_s_point": b_s, "leash_L_s_nats": round(leash_L, 4) if leash_L is not None else None,
            "n_recipes_below_epsilon_at_before": len(below_eps_rows),
            "max_observed_delta_among_below_epsilon": round(max_delta_below_eps, 4) if max_delta_below_eps is not None else None,
            "n_crossing_below_to_above_epsilon": n_cross,
            "crossing_count_note": (
                "exact: pi_tilde_after(y) is the after-model's own probability, "
                "renormalised over its own (model,target) key set the same way "
                "pi_tilde_before is -- not a proxy derived from before*exp(delta)."
            ),
            "noise_floor_crossing_substitute": {
                "before_model_own_samples_split_in_half": nf["total_n_crossing_noise_floor"],
                "n_below_epsilon_in_half1": nf["total_n_below_epsilon_half1"],
                "n_targets": nf["n_targets"],
                "resolution_adequate_for_epsilon": nf["resolution_adequate_for_epsilon"],
                "degenerate_warning": nf["degenerate_warning"],
            },
            "n_obs": len(rows),
        }
    return out


def block_b4_ternary(logp: dict) -> dict:
    out = {}
    for (model, target), by_recipe in logp.items():
        alks = target_alkalis(target)
        if not alks:
            continue
        pi = renorm(by_recipe)
        mass = {"carbonate": 0.0, "bare_oxide": 0.0, "other": 0.0}
        for y, p in pi.items():
            classes = [classify_alkali_source(list(y), el) for el in alks]
            if "carbonate" in classes:
                mass["carbonate"] += p
            elif "bare_oxide" in classes:
                mass["bare_oxide"] += p
            else:
                mass["other"] += p
        out.setdefault(model, {})[target] = {k: round(v, 4) for k, v in mass.items()}
    return out


def main():
    print("loading logp_universe.jsonl, universe_scores.jsonl, signature.json...", flush=True)
    logp = load_logp()
    rewards = load_recipe_rewards()
    signature = json.loads(SIGNATURE_PATH.read_text()) if SIGNATURE_PATH.exists() else None

    print("\nBlock B2: geodesic test (S6, S7)...", flush=True)
    b2 = block_b2(logp)
    for step, v in b2["steps"].items():
        print(f"  {step}: n_targets={v['n_targets']} median_rho={v['median_rho']} "
              f"met={v['prediction_met_median_lt_1']}", flush=True)
    OUT_B2.write_text(json.dumps({"prereg_commit": PREREG_COMMIT, "block": "B2", **b2}, indent=1, default=str))
    print(f"-> {OUT_B2}", flush=True)

    if signature is None:
        print("\nWARNING: signature.json not found -- B3 needs B1's b_s. Run B1 first.", flush=True)
    else:
        print("\nBlock B3: the leash...", flush=True)
        inventory = json.loads(INVENTORY_PATH.read_text())
        node_keys_by_target = load_universe_node_keys()
        b3 = block_b3(logp, rewards, signature, inventory, node_keys_by_target)
        for step, v in b3["steps"].items():
            print(f"  {step}: {v}", flush=True)
        OUT_B3.write_text(json.dumps({"prereg_commit": PREREG_COMMIT, "block": "B3", **b3}, indent=1, default=str))
        print(f"-> {OUT_B3}", flush=True)

    print("\nBlock B4: ternary figure data...", flush=True)
    b4 = block_b4_ternary(logp)
    OUT_B4_TERNARY.write_text(json.dumps({"prereg_commit": PREREG_COMMIT, "block": "B4", "ternary": b4}, indent=1, default=str))
    print(f"-> {OUT_B4_TERNARY}", flush=True)


if __name__ == "__main__":
    main()
