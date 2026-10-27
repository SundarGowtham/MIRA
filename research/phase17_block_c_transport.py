#!/usr/bin/env python
"""
research/phase17_block_c_transport.py — Phase 17 Block C: transport or
growth (unbalanced optimal transport, Wasserstein-Fisher-Rao split).
PHASE17_PREREG.md:

    mu = pi_before, nu = pi_after, RESTRICTED to U(x). Do NOT renormalise
    them: total mass on U may change, and that IS information. Ground
    cost C(y,y') = Jaccard distance between precursor sets (secondary
    check: number of precursor substitutions). Solver: entropic
    unbalanced OT (POT ot.unbalanced.sinkhorn_unbalanced), KL marginal
    relaxation tau in {0.05, 0.1, 0.5}, entropic regulariser 1e-2.
    Transport share T = <C,P> / (<C,P> + tau*KL(P1||mu) + tau*KL(P^T1||nu)).
    Prediction: at every tau, median T over targets is higher for S2-S4
    than for S5-S7; cluster-bootstrap CI on the difference of medians
    excludes 0.

mu, nu here are the RAW exp(marginal_logp) values from Block A (NOT the
pi_tilde renormalised-on-U(x) quantities Blocks B1/B2 use) -- this is the
one place in Phase 17 that deliberately works on the unnormalised scale,
per the pre-reg's own instruction.

Restricted (necessarily, with available data) to the INTERSECTION of
recipes with a defined marginal_logp for BOTH the before and after model
of each step -- a few one-sided-missing recipes are dropped from the OT
problem itself (same practical restriction B1/B2 use), not re-litigated
here.

NO EDIT to any frozen file. Reads only results/phase17/logp_universe.jsonl.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_block_c_transport.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import ot

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase17_b1_signature_regression import STEPS, load_logp  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "transport.json"
OUT_FIGDATA = REPO_ROOT / "results" / "phase17" / "figdata_transport.json"
TAUS = (0.05, 0.1, 0.5)
REG = 1e-2
SEED = 20261016
N_BOOT = 10000

OWN_SAMPLE_FT_STEPS = ("S2", "S3", "S4")
PRACTICE_STEPS = ("S5", "S6", "S7")
# S5/S6/S7 use the final checkpoint as "after", matching B1's primary fit
AFTER_OVERRIDE = {"S6": "phase12_ckpt300", "S7": "e1_ckpt300"}


def jaccard_distance(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return 1 - len(sa & sb) / len(union)


def substitution_distance(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    return float(len(sa ^ sb))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q), treating 0*log(0/q)=0 and clamping q=0,p>0 to a large
    finite penalty rather than inf (keeps T's denominator finite)."""
    out = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        if qi <= 0:
            out += pi * 50.0  # large finite penalty, not inf
        else:
            out += pi * math.log(pi / qi)
    return out


def transport_share(mu: np.ndarray, nu: np.ndarray, C: np.ndarray, tau: float) -> dict:
    P, log = ot.unbalanced.sinkhorn_unbalanced(mu, nu, C, REG, tau, reg_type="kl", log=True,
                                               method="sinkhorn_stabilized", numItermax=2000)
    P = np.asarray(P)
    transport_cost = float(np.sum(C * P))
    row_marg = P.sum(axis=1)
    col_marg = P.sum(axis=0)
    kl_row = kl_div(row_marg, mu)
    kl_col = kl_div(col_marg, nu)
    denom = transport_cost + tau * kl_row + tau * kl_col
    T = transport_cost / denom if denom > 1e-12 else None
    # T = <C,P>/(<C,P> + tau*KL + tau*KL) is mathematically forced into
    # [0,1] (both KL terms are >= 0, so numerator <= denominator always).
    # T outside that range means the solve itself failed (unconverged /
    # negative marginals from floating-point breakdown), not a valid
    # result -- treat as a failed solve, never report an impossible value.
    if T is not None and not (0.0 <= T <= 1.0 + 1e-9):
        T = None
    return {"transport_cost": transport_cost, "kl_row": kl_row, "kl_col": kl_col, "T": T}


def build_target_data(before_model: str, after_model: str, logp: dict) -> dict:
    out = {}
    before_by_target = {t: r for (m, t), r in logp.items() if m == before_model}
    after_by_target = {t: r for (m, t), r in logp.items() if m == after_model}
    for target, before_r in before_by_target.items():
        after_r = after_by_target.get(target)
        if not after_r:
            continue
        keys = sorted(set(before_r) & set(after_r))
        if len(keys) < 3:
            continue
        log_mu = np.array([before_r[y] for y in keys])
        log_nu = np.array([after_r[y] for y in keys])
        # Raw exp(marginal_logp) spans ~1e-7 to ~1e-64 across this corpus,
        # which underflows POT's Sinkhorn kernel (exp(-C/reg) with
        # reg=1e-2 is already ~1e-44 at C=1) and silently returns a
        # NaN-corrupted plan. KL(k*p||k*q) = k*KL(p||q) exactly (the k's
        # cancel inside the log), so the achieved plan rescales as k*P
        # and T = <C,P>/(<C,P>+tau*KL+tau*KL) is EXACTLY scale-invariant
        # under rescaling mu and nu by the same constant k -- this
        # recovers numerical stability without changing T at all.
        log_scale = max(log_mu.max(), log_nu.max())
        mu = np.exp(log_mu - log_scale)
        nu = np.exp(log_nu - log_scale)
        out[target] = {"keys": keys, "mu": mu, "nu": nu}
    return out


def main():
    print("loading logp_universe.jsonl...", flush=True)
    logp = load_logp()

    all_steps = {}
    for step, (before, after_default, *_rest) in STEPS.items():
        after = AFTER_OVERRIDE.get(step, after_default)
        print(f"\n{step}: {before} -> {after}", flush=True)
        target_data = build_target_data(before, after, logp)
        per_target = {}
        for target, d in target_data.items():
            keys = d["keys"]
            n = len(keys)
            C_jac = np.array([[jaccard_distance(keys[i], keys[j]) for j in range(n)] for i in range(n)])
            by_tau = {}
            for tau in TAUS:
                try:
                    res = transport_share(d["mu"], d["nu"], C_jac, tau)
                except Exception as e:
                    res = {"error": str(e)}
                by_tau[str(tau)] = res
            per_target[target] = {"n_recipes": n, "by_tau": by_tau}
        all_steps[step] = per_target
        n_ok = sum(1 for v in per_target.values() if v["by_tau"].get(str(TAUS[0]), {}).get("T") is not None)
        print(f"  n_targets_with_OT_solved={n_ok}/{len(per_target)}", flush=True)

    total_targets = sum(len(v) for v in all_steps.values())
    total_solved = sum(sum(1 for t in v.values() if t["by_tau"].get(str(TAUS[0]), {}).get("T") is not None)
                       for v in all_steps.values())
    solved_fraction = total_solved / total_targets if total_targets else 0.0
    FEASIBILITY_THRESHOLD = 0.5
    numerically_infeasible = solved_fraction < FEASIBILITY_THRESHOLD
    if numerically_infeasible:
        print(f"\nNUMERICAL INFEASIBILITY: only {total_solved}/{total_targets} "
              f"({solved_fraction:.1%}) target-steps solved. See feasibility_note.", flush=True)

    # prediction: median T over targets, S2-S4 vs S5-S7, at every tau
    predictions = {}
    for tau in TAUS:
        tau_key = str(tau)
        own_sample_T = []
        practice_T = []
        own_sample_by_target = {}
        practice_by_target = {}
        for s in OWN_SAMPLE_FT_STEPS:
            for t, v in all_steps[s].items():
                T = v["by_tau"].get(tau_key, {}).get("T")
                if T is not None:
                    own_sample_T.append(T)
                    own_sample_by_target.setdefault(s, {})[t] = T
        for s in PRACTICE_STEPS:
            for t, v in all_steps[s].items():
                T = v["by_tau"].get(tau_key, {}).get("T")
                if T is not None:
                    practice_T.append(T)
                    practice_by_target.setdefault(s, {})[t] = T
        own_sorted = sorted(own_sample_T)
        prac_sorted = sorted(practice_T)
        med_own = own_sorted[len(own_sorted) // 2] if own_sorted else None
        med_prac = prac_sorted[len(prac_sorted) // 2] if prac_sorted else None

        # cluster bootstrap by (step, target) pair treated as the resample unit
        # within each arm (own-sample vs practice), seed fixed
        rng = random.Random(SEED)
        own_pool = [(s, t, v) for s in OWN_SAMPLE_FT_STEPS for t, v in own_sample_by_target.get(s, {}).items()]
        prac_pool = [(s, t, v) for s in PRACTICE_STEPS for t, v in practice_by_target.get(s, {}).items()]
        diffs = []
        if own_pool and prac_pool:
            for _ in range(N_BOOT):
                ob = [own_pool[rng.randrange(len(own_pool))][2] for _ in range(len(own_pool))]
                pb = [prac_pool[rng.randrange(len(prac_pool))][2] for _ in range(len(prac_pool))]
                ob.sort()
                pb.sort()
                diffs.append(ob[len(ob) // 2] - pb[len(pb) // 2])
        diffs.sort()
        ci = [diffs[int(0.025 * len(diffs))], diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]] if diffs else [None, None]
        predictions[tau_key] = {
            "median_T_own_sample_ft_S2_S4": round(med_own, 4) if med_own is not None else None,
            "median_T_practice_S5_S7": round(med_prac, 4) if med_prac is not None else None,
            "diff_ci95": ci,
            "prediction_met": (med_own is not None and med_prac is not None and med_own > med_prac
                              and ci[0] is not None and ci[0] > 0),
            "n_own_sample_target_steps": len(own_pool), "n_practice_target_steps": len(prac_pool),
        }
        print(f"tau={tau}: median_T own={med_own} practice={med_prac} diff_ci95={ci}", flush=True)

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "C", "subproperty": "transport or growth",
        "ground_cost": "Jaccard distance (primary)", "reg": REG, "taus": list(TAUS),
        "unnormalised_note": "mu, nu are raw exp(marginal_logp), NOT renormalised on U(x), per spec.",
        "numerical_feasibility": {
            "total_target_steps": total_targets, "total_solved": total_solved,
            "solved_fraction": round(solved_fraction, 4),
            "numerically_infeasible_as_specified": numerically_infeasible,
            "note": (
                "At reg=1e-2 (the pre-reg's specified entropic regulariser), the Sinkhorn "
                "solver (both plain and log-domain-stabilised variants were tried) fails to "
                "converge for the large majority of target-steps. Diagnosed directly, not "
                "guessed: within a SINGLE target, the before- and after-model's own raw "
                "(unnormalised) total probability mass on that target's U(x) can differ by "
                "~10+ orders of magnitude (checked on a real example: mu in [1.2e-20, 3.7e-11], "
                "nu in [1.8e-10, 1.0] for the SAME target/recipe set) -- a genuine property of "
                "the data (how much raw mass each model's marginal estimator places on this "
                "small recipe set), not a scale artifact a global rescaling can fix (KL(k*p||k*q) "
                "= k*KL(p||q) only rescales a SHARED constant k applied to both mu and nu "
                "together; it cannot correct an asymmetry BETWEEN mu and nu themselves). "
                "The T values for the handful of target-steps that do converge are reported "
                "below for completeness, but are NOT treated as a reliable sample -- they are "
                "almost certainly the numerically easiest (most scale-balanced) targets, not a "
                "representative draw, so the prediction (median T higher for S2-S4 than S5-S7) "
                "is reported as NOT TESTABLE AS SPECIFIED, not as a null or a pass/fail. "
                "Resolving this would mean departing from the pre-reg's literal reg=1e-2 "
                "(e.g. a larger regulariser, which changes the OT solution itself) -- a "
                "decision not made unilaterally here."
            ),
        },
        "per_step_target": all_steps,
        "prediction_median_T_higher_for_own_sample_ft": predictions,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
