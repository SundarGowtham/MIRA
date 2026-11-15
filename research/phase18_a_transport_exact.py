#!/usr/bin/env python
"""
research/phase18_a_transport_exact.py — Phase 18 Block 18A: transport or
reweighting, solved exactly. docs/phases/PHASE18_PREREG.md, "18A".

Measures: empirical recipe frequencies from existing samples (canonical
recipes, results/phase18/recipes_by_model.json), never teacher-forced
probabilities. Per (step, target): subsample both models to n = min of
their parsed-sample counts, 200 repeats, average.

LP (HiGHS via scipy.optimize.linprog), TV penalty on created/destroyed mass:
  min_{P>=0} <C,P> + lam(||mu - P1||_1 + ||nu - P^T1||_1)  s.t. P1<=mu, P^T1<=nu
which, since the constraints make both residuals non-negative, equals
  lam(|mu|+|nu|) + min <C - 2 lam, P>.
C = Jaccard distance between precursor sets; lam in {0.35, 0.5}.
Tie rule (decided before running): an entry with C >= 2 lam is given upper
bound 0, i.e. at exact indifference (C == 2 lam: moving costs the same as
destroy+create) mass is counted as destroyed+created, not moved. The spec's
"moving is only cheaper when C < 2 lam" fixes this convention; without it the
LP solution (and T) at C == 2 lam would be solver-arbitrary.

Quantities: m_stay = sum_i P_ii (identical recipes), m_move = sum_{i!=j} P_ij,
m_new = |nu| - sum P, m_gone = |mu| - sum P, T = <C,P>/(<C,P>+lam(m_new+m_gone))
(undefined when the denominator is 0, i.e. identical distributions -> NaN,
excluded and counted). New-support mass = after-model's mass on recipes absent
from the before-model's reference.

Noise floor and reference (implementation decisions, made before running):
  * per repeat, hold out a random subsample B2 of the before-model of size
    n_f = min(n, floor(N_before/2)); the reference for new-support mass is
    the before-model's remaining samples (B minus B2), for BOTH the step and
    the floor, so both are measured against the same-size reference;
  * floor quantities are computed between B2 and a disjoint before-subsample
    B1 of size n_f (taken from B minus B2), at sample size n_f;
  * for Base and RS-SFT (232/target) n_f = n (about 32); for Corpus SFT
    (S5's before-model, 32/target) n_f = 16 -- flagged in the output.
Secondary solver: POT ot.unbalanced.sinkhorn_unbalanced(method=
"sinkhorn_stabilized", reg=0.05, reg_m=lam) on the first repeat only,
for comparison (its marginal penalty is KL, not TV, so its T is not
expected to equal the LP's).

Target pools: primary = ASTRAL clean 22 + held-out 40 (62 targets, Phase 17's
primary sets); secondary = ASTRAL 35 + held-out 40. Bootstrap = 10,000
resamples of targets (cluster by target), seed 20261115.

Predictions [C]:
  P-18A-1: at both lam, median T is higher for S2-S4 than S5-S7; bootstrap CI
           on the difference of medians excludes 0.
  P-18A-2: new-support mass minus its noise floor has a CI including 0 for
           each of S5-S7 and excluding 0 for S3 and S4.

Usage (tmux, CPU): scripts/run_phase18_block.sh 18A research/phase18_a_transport_exact.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase18_common import (  # noqa: E402
    STEPS, OWN_SAMPLE_FT_STEPS, PRACTICE_STEPS, SEED, N_BOOT, provenance, write_json,
    load_recipes, recipes, target_sets, jaccard_distance, notify,
)

LAMBDAS = (0.35, 0.5)
N_REP = 200
SINKHORN_REG = 0.05
QUANTS = ("m_stay", "m_move", "m_new", "m_gone", "cost", "T", "new_support")


def solve_lp(mu: dict, nu: dict, lam: float):
    a_keys, b_keys = list(mu), list(nu)
    k, l = len(a_keys), len(b_keys)
    C = np.array([[jaccard_distance(x, y) for y in b_keys] for x in a_keys])
    c = (C - 2 * lam).ravel()
    ub = np.where(C.ravel() >= 2 * lam - 1e-12, 0.0, None)
    bounds = [(0.0, u) for u in ub]
    A_rows = np.zeros((k, k * l))
    for i in range(k):
        A_rows[i, i * l:(i + 1) * l] = 1
    A_cols = np.zeros((l, k * l))
    for j in range(l):
        A_cols[j, j::l] = 1
    A = np.vstack([A_rows, A_cols])
    b = np.concatenate([[mu[x] for x in a_keys], [nu[y] for y in b_keys]])
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.status != 0:
        return None, res.status, None, None, None
    P = res.x.reshape(k, l)
    return P, 0, C, a_keys, b_keys


def plan_stats(mu, nu, lam):
    P, status, C, a_keys, b_keys = solve_lp(mu, nu, lam)
    if P is None:
        return {"status": status}
    stay = sum(P[i, j] for i, x in enumerate(a_keys) for j, y in enumerate(b_keys) if x == y)
    tot = P.sum()
    move = tot - stay
    m_new = sum(nu.values()) - tot
    m_gone = sum(mu.values()) - tot
    cost = float((C * P).sum())
    den = cost + lam * (m_new + m_gone)
    T = cost / den if den > 1e-12 else float("nan")
    moved_d = [(float(C[i, j]), float(P[i, j])) for i in range(len(a_keys)) for j in range(len(b_keys))
               if P[i, j] > 1e-12 and a_keys[i] != b_keys[j]]
    return {"status": 0, "m_stay": stay, "m_move": move, "m_new": max(m_new, 0.0),
            "m_gone": max(m_gone, 0.0), "cost": cost, "T": T, "moved": moved_d,
            "P": P, "a_keys": a_keys, "b_keys": b_keys}


def freqs(lst):
    c = Counter(lst)
    n = len(lst)
    return {k: v / n for k, v in c.items()}


def sinkhorn_T(mu, nu, lam):
    import ot
    a_keys, b_keys = list(mu), list(nu)
    C = np.array([[jaccard_distance(x, y) for y in b_keys] for x in a_keys])
    a = np.array([mu[x] for x in a_keys])
    b = np.array([nu[y] for y in b_keys])
    try:
        P, log = ot.unbalanced.sinkhorn_unbalanced(a, b, C, SINKHORN_REG, lam, method="sinkhorn_stabilized",
                                                    numItermax=10000, stopThr=1e-9, log=True)
        P = np.asarray(P)
        ok = bool(np.all(np.isfinite(P)))
        if not ok:
            return {"converged": False}
        cost = float((C * P).sum())
        m_new = max(b.sum() - P.sum(), 0.0)
        m_gone = max(a.sum() - P.sum(), 0.0)
        den = cost + lam * (m_new + m_gone)
        err = log.get("err", [None])
        return {"converged": True, "T": cost / den if den > 1e-12 else None,
                "final_err": float(err[-1]) if err and err[-1] is not None else None}
    except Exception as e:  # noqa: BLE001
        return {"converged": False, "error": repr(e)[:200]}


def run_target_step(args):
    step, target, B, A, seed, cap = args
    rng = random.Random(seed)
    NB, NA = len(B), len(A)
    if NB < 2 or NA < 1:
        return step, target, {"skipped": f"NB={NB} NA={NA}"}
    n = min(NB, NA) if cap is None else min(NB, NA, cap)
    n_f = min(n, NB // 2)
    acc = {lam: {"step": defaultdict(list), "floor": defaultdict(list)} for lam in LAMBDAS}
    status_fail = {lam: 0 for lam in LAMBDAS}
    n_lp = 0
    moved_hist = {lam: [] for lam in LAMBDAS}
    example = None
    sink = None
    for rep in range(N_REP):
        Bs = rng.sample(B, n)
        As = rng.sample(A, n)
        idx = list(range(NB))
        rng.shuffle(idx)
        B2 = [B[i] for i in idx[:n_f]]
        rest = [B[i] for i in idx[n_f:]]
        B1 = rest[:n_f]  # disjoint from B2 (rest is already a random order)
        ref = set(rest)
        mu, nu = freqs(Bs), freqs(As)
        mu1, nu2 = freqs(B1), freqs(B2)
        new_step = sum(1 for y in As if y not in ref) / len(As)
        new_floor = sum(1 for y in B2 if y not in ref) / len(B2)
        for lam in LAMBDAS:
            for kind, (m1, m2, ns) in (("step", (mu, nu, new_step)), ("floor", (mu1, nu2, new_floor))):
                st = plan_stats(m1, m2, lam)
                n_lp += 1
                if st["status"] != 0:
                    status_fail[lam] += 1
                    continue
                for q in ("m_stay", "m_move", "m_new", "m_gone", "cost", "T"):
                    acc[lam][kind][q].append(st[q])
                acc[lam][kind]["new_support"].append(ns)
                if kind == "step":
                    moved_hist[lam].extend(st["moved"])
                    if rep == 0 and lam == 0.5:
                        example = {"a_keys": [list(x) for x in st["a_keys"]],
                                   "b_keys": [list(y) for y in st["b_keys"]],
                                   "mu": [mu[x] for x in st["a_keys"]], "nu": [nu[y] for y in st["b_keys"]],
                                   "P": st["P"].round(6).tolist()}
        if rep == 0:
            sink = {str(lam): sinkhorn_T(mu, nu, lam) for lam in LAMBDAS}
    out = {"n": n, "n_floor": n_f, "N_before": NB, "N_after": NA, "n_lp": n_lp,
           "lp_failures": {str(l): status_fail[l] for l in LAMBDAS}, "sinkhorn_rep0": sink, "by_lambda": {}}
    for lam in LAMBDAS:
        d = {}
        for kind in ("step", "floor"):
            for q in QUANTS:
                vals = [v for v in acc[lam][kind][q] if not (isinstance(v, float) and math.isnan(v))]
                d[f"{kind}_{q}"] = float(np.mean(vals)) if vals else None
            d[f"{kind}_T_n_undefined"] = sum(1 for v in acc[lam][kind]["T"] if math.isnan(v))
        d["new_support_minus_floor"] = (d["step_new_support"] - d["floor_new_support"]
                                        if d["step_new_support"] is not None and d["floor_new_support"] is not None else None)
        # distance histogram of moved mass (mass-weighted, summed over reps)
        hist = defaultdict(float)
        for dist, m in moved_hist[lam]:
            hist[round(dist, 3)] += m / N_REP
        d["moved_mass_by_distance"] = dict(sorted(hist.items()))
        out["by_lambda"][str(lam)] = d
    out["example_plan_rep0_lam0.5"] = example
    return step, target, out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def boot_ci(vals, q=(0.025, 0.975)):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return [None, None]
    return [vals[int(q[0] * len(vals))], vals[min(int(q[1] * len(vals)), len(vals) - 1)]]


def analyse(per, pool, label):
    rng = random.Random(SEED)
    out = {"pool": label, "n_targets": len(pool)}
    # P-18A-1
    p1 = {}
    for lam in LAMBDAS:
        key = str(lam)

        def Ts(targets, steps):
            v = []
            for t in targets:
                for s in steps:
                    r = per.get(s, {}).get(t)
                    if r and "by_lambda" in r:
                        T = r["by_lambda"][key]["step_T"]
                        if T is not None:
                            v.append(T)
            return v
        own, prac = Ts(pool, OWN_SAMPLE_FT_STEPS), Ts(pool, PRACTICE_STEPS)
        point = median(own) - median(prac)
        diffs = []
        for _ in range(N_BOOT):
            samp = [pool[rng.randrange(len(pool))] for _ in pool]
            o, p = Ts(samp, OWN_SAMPLE_FT_STEPS), Ts(samp, PRACTICE_STEPS)
            if o and p:
                diffs.append(median(o) - median(p))
        ci = boot_ci(diffs)
        per_step_med = {s: median(Ts(pool, (s,))) for s in STEPS}
        p1[key] = {"median_T_S2_S4": median(own), "median_T_S5_S7": median(prac),
                   "n_target_steps_S2_S4": len(own), "n_target_steps_S5_S7": len(prac),
                   "diff": point, "ci95": ci, "ci_excludes_0": ci[0] is not None and (ci[0] > 0 or ci[1] < 0),
                   "direction_holds": point > 0, "median_T_by_step": per_step_med}
    out["P_18A_1"] = {"statement": "At both lambda, median T higher for S2-S4 than S5-S7; bootstrap CI on the "
                                   "difference of medians excludes 0.",
                      "by_lambda": p1,
                      "passes": all(v["direction_holds"] and v["ci95"][0] is not None and v["ci95"][0] > 0
                                    for v in p1.values()),
                      "reversed": any(v["ci95"][1] is not None and v["ci95"][1] < 0 for v in p1.values())}
    # P-18A-2 (new support doesn't depend on lambda; use lam=0.5 record)
    p2 = {}
    for s in STEPS:
        diffs_t = {t: per[s][t]["by_lambda"]["0.5"]["new_support_minus_floor"]
                   for t in pool if t in per.get(s, {}) and "by_lambda" in per[s][t]}
        diffs_t = {t: v for t, v in diffs_t.items() if v is not None}
        ts = list(diffs_t)
        point = float(np.mean(list(diffs_t.values())))
        boots = []
        for _ in range(N_BOOT):
            samp = [ts[rng.randrange(len(ts))] for _ in ts]
            boots.append(float(np.mean([diffs_t[t] for t in samp])))
        ci = boot_ci(boots)
        step_ns = float(np.mean([per[s][t]["by_lambda"]["0.5"]["step_new_support"] for t in ts]))
        floor_ns = float(np.mean([per[s][t]["by_lambda"]["0.5"]["floor_new_support"] for t in ts]))
        p2[s] = {"new_support_mass": step_ns, "noise_floor": floor_ns, "diff": point, "ci95": ci,
                 "ci_includes_0": ci[0] <= 0 <= ci[1], "ci_above_0": ci[0] > 0, "n_targets": len(ts)}
    out["P_18A_2"] = {"statement": "New-support mass minus noise floor: CI includes 0 for each of S5-S7, "
                                   "excludes 0 for S3 and S4.",
                      "by_step": p2,
                      "passes": all(p2[s]["ci_includes_0"] for s in PRACTICE_STEPS)
                      and all(not p2[s]["ci_includes_0"] for s in ("S3", "S4")),
                      "practice_step_above_floor": [s for s in PRACTICE_STEPS if p2[s]["ci_above_0"]]}
    # descriptive means by step
    desc = {}
    for s in STEPS:
        rows = [per[s][t] for t in pool if t in per.get(s, {}) and "by_lambda" in per[s][t]]
        desc[s] = {lam: {q: float(np.mean([r["by_lambda"][lam][f"step_{q}"] for r in rows
                                            if r["by_lambda"][lam][f"step_{q}"] is not None]))
                         for q in ("m_stay", "m_move", "m_new", "m_gone", "cost")}
                   for lam in ("0.35", "0.5")}
    out["mean_mass_split_by_step"] = desc
    return out


def classical_mds(D):
    n = D.shape[0]
    if n == 1:
        return np.zeros((1, 2))
    J = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:2]
    w2 = np.clip(w[idx], 0, None)
    X = V[:, idx] * np.sqrt(w2)
    if X.shape[1] < 2:
        X = np.hstack([X, np.zeros((n, 2 - X.shape[1]))])
    return X


N_CAP_SENSITIVITY = 32


def run_all(jobs_base, cap):
    jobs = [j + (cap,) for j in jobs_base]
    per = defaultdict(dict)
    with Pool(24) as pool:
        for i, (s, t, r) in enumerate(pool.imap_unordered(run_target_step, jobs, chunksize=2)):
            per[s][t] = r
            if (i + 1) % 100 == 0:
                print(f"  cap={cap}: {i+1}/{len(jobs)} done", flush=True)
    return per


def main():
    prov = provenance()
    cache = load_recipes()
    sets = target_sets()
    all_targets = sets["astral_35"] + sets["heldout_40"]
    jobs = []
    for si, (s, (before, after)) in enumerate(STEPS.items()):
        for ti, t in enumerate(all_targets):
            ts = "astral" if t in sets["astral_35"] else "heldout"
            B = recipes(cache, before, ts, t)
            A = recipes(cache, after, ts, t)
            jobs.append((s, t, B, A, SEED + 1000 * si + ti))
    print(f"{len(jobs)} target-steps; {N_REP} reps x 2 lambdas x (step+floor) LPs each", flush=True)
    per = run_all(jobs, None)
    per_cap = run_all(jobs, N_CAP_SENSITIVITY)
    # convergence
    total_lp = sum(r.get("n_lp", 0) for d in per.values() for r in d.values())
    fails = sum(sum(r.get("lp_failures", {}).values()) for d in per.values() for r in d.values())
    skipped = [(s, t) for s, d in per.items() for t, r in d.items() if "skipped" in r]
    ts_solved = sum(1 for d in per.values() for r in d.values()
                    if "lp_failures" in r and sum(r["lp_failures"].values()) == 0)
    sink_conv = defaultdict(lambda: [0, 0])
    for d in per.values():
        for r in d.values():
            for lam, v in (r.get("sinkhorn_rep0") or {}).items():
                sink_conv[lam][1] += 1
                sink_conv[lam][0] += int(bool(v.get("converged")))
    # LP vs sinkhorn T agreement (rep 0 is not the averaged T; report rank corr over target-steps)
    primary_pool = sets["astral_clean_22"] + sets["heldout_40"]
    secondary_pool = all_targets
    res_primary = analyse(per, primary_pool, "primary: ASTRAL clean 22 + held-out 40")
    res_secondary = analyse(per, secondary_pool, "secondary: ASTRAL 35 + held-out 40")
    by_set = {lab: analyse(per, pool, lab) for lab, pool in
              (("astral_clean_22_only", sets["astral_clean_22"]), ("heldout_40_only", sets["heldout_40"]))}

    # [E] distance histogram of moved mass, pooled, by step group
    hist = {}
    for lam in ("0.35", "0.5"):
        for grp, steps in (("S2_S4", OWN_SAMPLE_FT_STEPS), ("S5_S7", PRACTICE_STEPS), ("S1", ("S1",))):
            h = defaultdict(float)
            for s in steps:
                for t in primary_pool:
                    r = per[s].get(t, {})
                    for dist, m in (r.get("by_lambda", {}).get(lam, {}).get("moved_mass_by_distance") or {}).items():
                        h[float(dist)] += m
            tot = sum(h.values())
            one_swap = sum(m for d_, m in h.items() if d_ <= 0.5 + 1e-9)
            hist[f"lam{lam}_{grp}"] = {"mass_by_jaccard_distance": dict(sorted(h.items())),
                                       "share_within_one_swap_(d<=0.5)": one_swap / tot if tot else None}
    # [E] six example plans: the 3 primary-pool targets with most moved mass in S2-S4 and in S5-S7 (lam=0.5)
    figdata = {"examples": []}
    for grp, steps in (("own_sample_ft", OWN_SAMPLE_FT_STEPS), ("practice", PRACTICE_STEPS)):
        cand = []
        for s in steps:
            for t in primary_pool:
                r = per[s].get(t, {})
                if "by_lambda" in r and r.get("example_plan_rep0_lam0.5"):
                    cand.append((r["by_lambda"]["0.5"]["step_m_move"], s, t))
        cand.sort(reverse=True)
        seen = set()
        picked = []
        for mv, s, t in cand:
            if t in seen:
                continue
            seen.add(t)
            picked.append((s, t))
            if len(picked) == 3:
                break
        for s, t in picked:
            ex = per[s][t]["example_plan_rep0_lam0.5"]
            keys = [tuple(x) for x in ex["a_keys"]] + [tuple(y) for y in ex["b_keys"] if tuple(y) not in
                                                          {tuple(x) for x in ex["a_keys"]}]
            Dm = np.array([[jaccard_distance(x, y) for y in keys] for x in keys])
            xy = classical_mds(Dm)
            figdata["examples"].append({"group": grp, "step": s, "target": t, "lambda": 0.5,
                                        "recipes": [list(k) for k in keys], "mds_xy": xy.round(4).tolist(),
                                        **ex})
    figdata["moved_mass_histograms"] = hist

    doc = {**prov, "block": "18A",
           "method": {"lambdas": LAMBDAS, "n_rep": N_REP, "cost": "Jaccard distance on canonical precursor sets",
                      "solver": "scipy.optimize.linprog(method='highs')",
                      "tie_rule": "entries with C >= 2*lambda bounded at 0 (indifference -> destroy+create)",
                      "noise_floor": "two disjoint before-model subsamples of size n_f = min(n, N_before//2); "
                                     "new-support reference = before-model samples minus the held-out floor subsample",
                      "secondary_solver": f"POT sinkhorn_unbalanced sinkhorn_stabilized reg={SINKHORN_REG} reg_m=lambda, rep 0 only",
                      "bootstrap": f"{N_BOOT} target resamples, seed {SEED}"},
           "convergence": {"lp_total": total_lp, "lp_failures": fails,
                           "lp_solved_share": 1 - fails / total_lp if total_lp else None,
                           "target_steps_all_lps_solved": ts_solved,
                           "target_steps_total": sum(len(d) for d in per.values()),
                           "target_steps_skipped": skipped,
                           "sinkhorn_rep0_converged": {k: f"{v[0]}/{v[1]}" for k, v in sink_conv.items()}},
           "s5_floor_caveat": "S5's before-model (Corpus SFT) has 32 samples/target, so its noise floor uses two "
                              "disjoint 16-sample halves (n_f=16) and a 16-sample new-support reference.",
           "primary": res_primary, "secondary": res_secondary, "by_target_set": by_set,
           "E_moved_mass_distance_histogram": hist,
           "E_sensitivity_n_cap_32": {
               "note": "[E] decided after the smoke test showed S2 (Base->RS-SFT, both 232/target) is compared at "
                       "n~224 under the literal 'subsample to the smaller n' rule while every other step is at "
                       "n~32. Same analysis with every step's n capped at 32.",
               "primary": analyse(per_cap, primary_pool, "primary, n<=32"),
               "secondary": analyse(per_cap, secondary_pool, "secondary, n<=32")},
           "per_target_step": {s: {t: {k: v for k, v in r.items() if k != "example_plan_rep0_lam0.5"}
                                   for t, r in d.items()} for s, d in per.items()}}
    p = write_json("transport_exact.json", doc)
    write_json("figdata_transport_exact.json", {**prov, "block": "18A", **figdata})
    print("convergence:", doc["convergence"]["lp_solved_share"], doc["convergence"]["sinkhorn_rep0_converged"])
    for lab, r in (("primary", res_primary), ("secondary", res_secondary)):
        print(f"== {lab}")
        for lam, v in r["P_18A_1"]["by_lambda"].items():
            print(f"  P-18A-1 lam={lam}: med S2-S4={v['median_T_S2_S4']:.3f} S5-S7={v['median_T_S5_S7']:.3f} "
                  f"diff={v['diff']:.3f} CI={v['ci95']}  by step={ {k: round(x,3) for k,x in v['median_T_by_step'].items()} }")
        print("  P-18A-1 passes:", r["P_18A_1"]["passes"])
        for s, v in r["P_18A_2"]["by_step"].items():
            print(f"  P-18A-2 {s}: new={v['new_support_mass']:.4f} floor={v['noise_floor']:.4f} "
                  f"diff={v['diff']:.4f} CI={[round(x,4) for x in v['ci95']]}")
        print("  P-18A-2 passes:", r["P_18A_2"]["passes"])
    for s_, v in doc["E_sensitivity_n_cap_32"]["primary"]["P_18A_1"]["by_lambda"].items():
        print(f"  [E n<=32] P-18A-1 lam={s_}: diff={v['diff']:.3f} CI={v['ci95']} by step={ {k: round(x,3) for k,x in v['median_T_by_step'].items()} }")
    for s_, v in doc["E_sensitivity_n_cap_32"]["primary"]["P_18A_2"]["by_step"].items():
        print(f"  [E n<=32] P-18A-2 {s_}: diff={v['diff']:.4f} CI={[round(x,4) for x in v['ci95']]}")
    print(f"-> {p}")
    notify(f"MIRA 18A done: P-18A-1 {res_primary['P_18A_1']['passes']}, P-18A-2 {res_primary['P_18A_2']['passes']}")


if __name__ == "__main__":
    main()
