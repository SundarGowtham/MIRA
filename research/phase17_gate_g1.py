#!/usr/bin/env python
"""
research/phase17_gate_g1.py — Phase 17 Block A's GATE G1 verdict, computed
from results/phase17/logp_universe.jsonl (written by phase17_block_a_
forward_pass.py). Per docs/phases/PHASE17_PREREG.md:

  Hold out the samples not used as tau_k. Per model and target, compute
  the Spearman correlation between log pi_hat(y|x) and the empirical
  frequency of y among the held-out samples. Pass: the median over
  targets is >= 0.5 for every model under the marginal estimator.
  Report the same for the answer-only estimator (sensitivity / how far
  Phase 15's empty-think numbers can be trusted). If G1 fails, stop --
  do not run B-F.

  Consistency check: the count of recipes entering the epsilon-support
  per step must agree in sign with Phase 16's support growth
  (results/support_growth_by_checkpoint.json). A disagreement is
  reported before anything else.

HELD-OUT FREQUENCY: re-scans every model's raw completions for each
target (same traversal order forward_pass.py used, so the SAME first-
K-per-(model,target) rows become "used as tau_k"), and counts precset_
of(route) over every OTHER successfully-parsed row for that (model,
target) -- i.e. excluding exactly the rows that became one of the K
traces, never a reconstruction or approximation of "held out".

Only node_keys that forward_pass.py actually scored (present in
logp_universe.jsonl for that model+target) enter the Spearman -- this
is U(x) restricted to whatever that (model,target) pair could score,
consistent with the coverage gap already flagged in inventory_v2.json.

EPSILON-SUPPORT SIGN CHECK: for phase12_ckpt{100,200,300} and
e1_ckpt{100,200,300}, counts |{y in U(x): pi_hat_marginal(y|x) > 1/200}|
per target, averaged over targets in each set (astral/heldout
separately), compared step-to-step (100->200, 200->300) against the
SAME-DIRECTION comparison of Phase 16's own support-growth "point"
values in results/support_growth_by_checkpoint.json.

NO EDIT to phase17_block_a_forward_pass.py or any frozen file -- only
reads logp_universe.jsonl and the raw completion files it was built
from (read-only re-scan, CPU, no GPU).

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_gate_g1.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

from scipy.stats import spearmanr  # noqa: E402

from core.reward import parse_completion  # noqa: E402
from research.support_growth_by_checkpoint import precset_of  # noqa: E402
from research.phase17_block_a_forward_pass import cut_prefix_at_precursors, K_TRACES  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
INVENTORY_PATH = REPO_ROOT / "results" / "phase17" / "inventory_v2.json"
LOGP_PATH = REPO_ROOT / "results" / "phase17" / "logp_universe.jsonl"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "instrument_check.json"
SUPPORT_GROWTH_PATH = REPO_ROOT / "results" / "support_growth_by_checkpoint.json"
EPSILON = 1.0 / 200


def held_out_frequencies(inventory: dict, astral_targets: set, heldout_targets: set,
                         node_keys_by_target: dict[str, set]) -> dict:
    """{(model,target): Counter(precset -> count)} over every raw sample
    NOT used as one of the K tau_k, replicating phase17_block_a_forward_
    pass.py's scan_surface_forms_and_traces EXACTLY: only rows whose
    precset is in that target's U(x) are considered at all (same filter,
    same order), so the same first-K-such-rows become traces here as
    they did there -- otherwise a row skipped by that scan's U(x) filter
    could be miscounted here as consuming one of the K trace slots."""
    held_out: dict[tuple, Counter] = defaultdict(Counter)
    n_used_as_trace: dict[tuple, int] = defaultdict(int)

    for model, by_set in inventory["raw_completions"].items():
        for target_set, info in by_set.items():
            path_str = info.get("raw_completions_file")
            if not path_str:
                continue
            path = REPO_ROOT / path_str
            if not path.exists():
                continue
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    target = rec.get("target")
                    if target not in astral_targets and target not in heldout_targets:
                        continue
                    completion = rec.get("completion", "")
                    try:
                        route = parse_completion(completion, target)
                        key = precset_of(route)
                    except Exception:
                        continue
                    if key not in node_keys_by_target.get(target, set()):
                        continue
                    mt = (model, target)
                    if n_used_as_trace[mt] < K_TRACES and cut_prefix_at_precursors(completion) is not None:
                        n_used_as_trace[mt] += 1
                        continue  # this row WAS one of the K traces -- excluded from held-out freq
                    held_out[mt][key] += 1
    return held_out


def gate_for_estimator(rows_by_model_target: dict, held_out: dict, logp_field: str) -> dict:
    per_model_target_rho = {}
    for (model, target), rows in rows_by_model_target.items():
        pairs = [(r[logp_field], held_out.get((model, target), Counter()).get(tuple(r["recipe"]), 0))
                 for r in rows if r[logp_field] is not None]
        if len(pairs) < 3:
            continue
        logps = [p[0] for p in pairs]
        freqs = [p[1] for p in pairs]
        if len(set(freqs)) < 2 or len(set(logps)) < 2:
            continue  # spearman undefined / degenerate
        rho, _p = spearmanr(logps, freqs)
        if rho == rho:  # not NaN
            per_model_target_rho[(model, target)] = float(rho)

    by_model = defaultdict(list)
    for (model, target), rho in per_model_target_rho.items():
        by_model[model].append(rho)

    per_model_summary = {}
    for model, rhos in by_model.items():
        per_model_summary[model] = {
            "n_targets": len(rhos), "median_rho": round(st.median(rhos), 4),
            "mean_rho": round(st.mean(rhos), 4), "passes_0.5": st.median(rhos) >= 0.5,
        }
    return {"per_model_target_rho": {f"{m}|{t}": round(r, 4) for (m, t), r in per_model_target_rho.items()},
           "per_model": per_model_summary}


def epsilon_support_count(rows: list, epsilon: float) -> int | None:
    """|{y in U(x): pi_tilde(y) > epsilon}|, pi_tilde renormalised over
    this (model,target)'s scored node_keys -- per the pre-reg's own
    definition (line 128: "epsilon-support ... with pi renormalised on
    U(x)", same convention as effective support / Fisher-Rao, lines
    126-127). Raw sequence log p is NOT comparable to epsilon=1/200
    directly: a ~50-150 token JSON completion has a raw summed log p of
    -15 to -30 nats even when it is the model's overwhelmingly preferred
    choice, so every raw-logp count was trivially 0 (caught by direct
    inspection of logp_universe.jsonl's value range before trusting the
    check)."""
    logps = [r["marginal_logp"] for r in rows if r["marginal_logp"] is not None]
    if not logps:
        return None
    log_z = logsumexp(logps)
    return sum(1 for lp in logps if math.exp(lp - log_z) > epsilon)


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def epsilon_support_sign_check() -> dict:
    """Per regular Claude (2026-10-27): report Phase 16's point AND ci95
    (not just the point) alongside Block A's eps-support change, and
    repeat the whole check at epsilon in {1/100, 1/200, 1/500} -- a
    disagreement that is really just two overlapping-CI point estimates
    moving in opposite directions by noise should show up as unstable
    across these three epsilons; a real disagreement should not."""
    sg = json.loads(SUPPORT_GROWTH_PATH.read_text())["by_checkpoint"]
    rows_by_model_target = defaultdict(list)
    with LOGP_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            rows_by_model_target[(r["model"], r["target"])].append(r)

    epsilons = {"1/100": 1.0 / 100, "1/200": 1.0 / 200, "1/500": 1.0 / 500}
    by_epsilon = {}
    for eps_label, eps_val in epsilons.items():
        checks = []
        for family, ckpts in (("phase12", ["phase12_ckpt100", "phase12_ckpt200", "phase12_ckpt300"]),
                              ("e1", ["e1_ckpt100", "e1_ckpt200", "e1_ckpt300"])):
            for tset in ("astral", "heldout"):
                eps_counts = []
                for ck in ckpts:
                    counts = []
                    for (model, target), rows in rows_by_model_target.items():
                        if model != ck:
                            continue
                        row_tset = rows[0]["target_set"]
                        if row_tset != tset:
                            continue
                        n_eps = epsilon_support_count(rows, eps_val)
                        if n_eps is not None:
                            counts.append(n_eps)
                    eps_counts.append(st.mean(counts) if counts else None)
                sg_cells = [sg[ck][tset] for ck in ckpts]
                sg_points = [c["point"] for c in sg_cells]
                sg_cis = [c["ci95"] for c in sg_cells]
                for i in range(len(ckpts) - 1):
                    a, b = eps_counts[i], eps_counts[i + 1]
                    sa, sb = sg_points[i], sg_points[i + 1]
                    cia, cib = sg_cis[i], sg_cis[i + 1]
                    ci_overlap = (cia[0] <= cib[1]) and (cib[0] <= cia[1])
                    if a is None or b is None:
                        sign_a = None
                    else:
                        sign_a = (b - a > 1e-9) - (b - a < -1e-9)
                    sign_b = (sb - sa > 1e-9) - (sb - sa < -1e-9)
                    checks.append({
                        "family": family, "target_set": tset,
                        "step": f"{ckpts[i]}->{ckpts[i+1]}",
                        "block_a_mean_eps_support": [a, b],
                        "phase16_support_growth_point": [round(sa, 4), round(sb, 4)],
                        "phase16_support_growth_ci95": [cia, cib],
                        "phase16_cis_overlap": ci_overlap,
                        "block_a_sign": sign_a, "phase16_sign": sign_b,
                        "agrees": (sign_a == sign_b) if sign_a is not None else None,
                    })
        by_epsilon[eps_label] = {
            "checks": checks,
            "n_agree": sum(1 for c in checks if c["agrees"] is True),
            "n_disagree": sum(1 for c in checks if c["agrees"] is False),
            "n_undetermined": sum(1 for c in checks if c["agrees"] is None),
            "n_disagree_with_overlapping_phase16_cis":
                sum(1 for c in checks if c["agrees"] is False and c["phase16_cis_overlap"]),
        }
    return {"epsilons_tested": list(epsilons.values()), "by_epsilon": by_epsilon,
           "primary_epsilon": "1/200", **by_epsilon["1/200"]}


def load_universe_node_keys() -> dict[str, set]:
    by_target = defaultdict(set)
    with (REPO_ROOT / "results" / "phase17" / "universe_v2.jsonl").open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                by_target[d["target"]].add(tuple(sorted(d["recipe"])))
    return by_target


def main():
    inventory = json.loads(INVENTORY_PATH.read_text())
    astral_doc = json.loads((REPO_ROOT / "misc" / "astral_validation_set.json").read_text())
    astral_targets = {t["target"] for t in astral_doc["targets"]}
    heldout_targets = set(json.loads((REPO_ROOT / "data" / "novelty" / "heldout_targets.json").read_text()))
    node_keys_by_target = load_universe_node_keys()

    print("computing held-out frequencies (excluding the K rows used as traces)...", flush=True)
    held_out = held_out_frequencies(inventory, astral_targets, heldout_targets, node_keys_by_target)

    rows_by_model_target = defaultdict(list)
    with LOGP_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            rows_by_model_target[(r["model"], r["target"])].append(r)

    print("Gate G1, marginal estimator...", flush=True)
    marginal_gate = gate_for_estimator(rows_by_model_target, held_out, "marginal_logp")
    print(json.dumps(marginal_gate["per_model"], indent=1), flush=True)

    print("\nSensitivity check, answer-only estimator...", flush=True)
    answer_only_gate = gate_for_estimator(rows_by_model_target, held_out, "answer_only_logp")
    print(json.dumps(answer_only_gate["per_model"], indent=1), flush=True)

    print("\nEpsilon-support sign-consistency check...", flush=True)
    sign_check = epsilon_support_sign_check()
    print(json.dumps(sign_check, indent=1), flush=True)

    all_models = json.loads(INVENTORY_PATH.read_text())["models"].keys()
    marginal_pass_per_model = {m: marginal_gate["per_model"].get(m, {}).get("passes_0.5")
                               for m in all_models}
    models_with_no_marginal_data = [m for m, v in marginal_pass_per_model.items() if v is None]
    models_failing = [m for m, v in marginal_pass_per_model.items() if v is False]
    overall_pass = len(models_failing) == 0 and len(models_with_no_marginal_data) == 0

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "A", "subproperty": "Gate G1 verdict",
        "rule": "median Spearman(log pi_hat_marginal, held-out frequency) over targets >= 0.5, for EVERY model",
        "marginal_estimator": marginal_gate,
        "answer_only_estimator_sensitivity_check": answer_only_gate,
        "epsilon_support_sign_consistency_check": sign_check,
        "verdict": {
            "models_passing": [m for m, v in marginal_pass_per_model.items() if v is True],
            "models_failing": models_failing,
            "models_with_no_marginal_data_at_all": models_with_no_marginal_data,
            "overall_pass": overall_pass,
            "note": "'no marginal data at all' means every target for that model returned too few "
                    "scoreable node_keys (<3) or degenerate rank data for Spearman -- distinct from "
                    "a genuine <0.5 FAIL. Per the pre-reg: 'if G1 fails, stop -- do not run B-F.'",
        },
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nOVERALL PASS: {overall_pass}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
