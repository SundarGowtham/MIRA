#!/usr/bin/env python
"""
research/phase18_h_conformal.py — Phase 18 Block 18H: a calibrated judge that
abstains (split conformal, UQNO-style). PHASE18_PREREG.md "18H".

Data: ARROWS3 YBCO pairs labelled by lab sign (results/phase18/
arrows_pair_table.json, the R-f primary endpoint, |delta| >= 5 wt%).
Judges and margins (decided before running):
  R2  5-channel checker: GDPO vote_sum (sum of per-channel signs, -5..5);
      a validator gate failure on one side -> +-5; both fail -> 0.
  R4  ranker v2: score_a - score_b.
  R5  comparator: compare() margin; one-sided gate failure -> +-(number of
      scored channels); both fail -> 0.
  CF  carbonate-free rule: -1/0/+1 ([E] per the pre-registration).
p(a > b) = sigmoid(margin / scale); 1/scale fit by maximum likelihood (one
parameter, no intercept, bounded >= 0) on the calibration half.
Split conformal for the binary label y in {+1,-1}: nonconformity
s = 1 - p(y); qhat = ceil((n_cal+1)(1-alpha))-th smallest calibration score;
prediction set = {y : 1 - p(y) <= qhat}. Size 1 = decisive, 2 = abstain,
0 = empty (reported).
Splits: recipes (nodes) randomly halved; a pair is in calibration if both its
recipes are, in test if both are, and dropped otherwise. 100 splits, seed
20261115. Reported at target coverage 80% and 90%: mean achieved coverage,
decisive share, abstention share, empty share, with 2.5/97.5 percentiles
over splits.

Prediction [C] P-18H-1: for R2, R4 and R5, the decisive share at 90% coverage
is <= 20% (mean over splits).
[E]: the carbonate-free rule; ASTRAL (35 pairs; nodes are per-target, so
grouping by recipe = grouping by target).

Usage (tmux, CPU): scripts/run_phase18_block.sh 18H research/phase18_h_conformal.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase18_arrows_pairs import load_table  # noqa: E402
from research.phase18_common import SEED, provenance, write_json, notify  # noqa: E402

N_SPLITS = 100
COVERAGES = (0.8, 0.9)


def margins(rows, comp_max):
    out = {"R2": [], "R4": [], "R5": [], "CF": []}
    for r in rows:
        if r["R2_reason"].startswith("gate_failure"):
            out["R2"].append(5.0 * (r["R2_vote"] or 0))
        elif r["R2_vote_sum"] is None:
            out["R2"].append(0.0)
        else:
            out["R2"].append(float(r["R2_vote_sum"]))
        out["R4"].append(float(r["R4_margin"]))
        m = r["R5"]["margin"]
        out["R5"].append(0.0 if m is None else float(m))
        out["CF"].append(float(r["base_carbfree"]))
    return {k: np.array(v) for k, v in out.items()}


def fit_beta(m, y):
    def nll(b):
        z = y * m * b
        return float(np.sum(np.logaddexp(0, -z)))
    hi = 50.0 / (np.abs(m).max() + 1e-12) if np.abs(m).max() > 0 else 1.0
    res = minimize_scalar(nll, bounds=(0.0, hi), method="bounded")
    return float(res.x)


def p_of(m, b, y):
    return 1 / (1 + np.exp(-y * m * b))


def one_split(m, y, cal, test, alpha):
    b = fit_beta(m[cal], y[cal])
    s = 1 - p_of(m[cal], b, y[cal])
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    qhat = np.inf if k > n else np.sort(s)[k - 1]
    in_pos = (1 - p_of(m[test], b, 1)) <= qhat + 1e-12
    in_neg = (1 - p_of(m[test], b, -1)) <= qhat + 1e-12
    size = in_pos.astype(int) + in_neg.astype(int)
    covered = np.where(y[test] == 1, in_pos, in_neg)
    return {"coverage": float(covered.mean()), "decisive": float((size == 1).mean()),
            "abstain": float((size == 2).mean()), "empty": float((size == 0).mean()),
            "beta": b, "n_cal": int(n), "n_test": int(len(test)),
            "decisive_accuracy": float(covered[size == 1].mean()) if (size == 1).any() else None}


def run(rows, comp_max, rng, label):
    y = np.array([r["actual"] for r in rows], dtype=float)
    keep = y != 0
    rows = [r for r, k in zip(rows, keep) if k]
    y = y[keep]
    M = margins(rows, comp_max)
    nodes = sorted({r["set_a"] for r in rows} | {r["set_b"] for r in rows})
    ia = np.array([nodes.index(r["set_a"]) for r in rows])
    ib = np.array([nodes.index(r["set_b"]) for r in rows])
    splits = []
    for _ in range(N_SPLITS):
        side = rng.integers(0, 2, len(nodes))
        cal = np.where((side[ia] == 0) & (side[ib] == 0))[0]
        test = np.where((side[ia] == 1) & (side[ib] == 1))[0]
        splits.append((cal, test))
    out = {"label": label, "n_pairs": len(rows), "n_recipes": len(nodes), "by_judge": {}}
    for j, m in M.items():
        res = {}
        for cov in COVERAGES:
            runs = [one_split(m, y, cal, test, 1 - cov) for cal, test in splits if len(cal) >= 5 and len(test) >= 1]
            agg = {}
            for key in ("coverage", "decisive", "abstain", "empty", "beta"):
                v = np.array([r[key] for r in runs])
                agg[key] = {"mean": float(v.mean()), "p2.5": float(np.percentile(v, 2.5)),
                            "p97.5": float(np.percentile(v, 97.5))}
            da = [r["decisive_accuracy"] for r in runs if r["decisive_accuracy"] is not None]
            agg["decisive_accuracy_mean"] = float(np.mean(da)) if da else None
            agg["n_splits_used"] = len(runs)
            agg["mean_n_cal"] = float(np.mean([r["n_cal"] for r in runs]))
            agg["mean_n_test"] = float(np.mean([r["n_test"] for r in runs]))
            res[str(cov)] = agg
        out["by_judge"][j] = res
        print(f"  [{label}] {j}: " + "; ".join(
            f"cov{c}: achieved {res[c]['coverage']['mean']:.3f} decisive {res[c]['decisive']['mean']:.3f} "
            f"abstain {res[c]['abstain']['mean']:.3f} empty {res[c]['empty']['mean']:.3f}" for c in res), flush=True)
    return out


def main():
    prov = provenance()
    rng = np.random.default_rng(SEED)
    t = load_table()
    cm = t["comparator_gate_margin_magnitude"]
    ybco = run(t["ybco_primary_5wtpct"], cm, rng, "YBCO_primary")
    ast = run(t["astral_35_small"], cm, rng, "ASTRAL_35_E")
    rows = {j: ybco["by_judge"][j]["0.9"]["decisive"]["mean"] for j in ("R2", "R4", "R5")}
    passes = all(v <= 0.20 for v in rows.values())
    doc = {**prov, "block": "18H", "n_splits": N_SPLITS,
           "P_18H_1": {"statement": "For R2, R4 and R5, the decisive share at 90% coverage is <= 20%.",
                       "decisive_share_at_90": rows, "passes": passes,
                       "per_judge_pass": {j: v <= 0.20 for j, v in rows.items()}},
           "ybco_primary": ybco, "E_astral": ast}
    p = write_json("conformal.json", doc)
    print("P-18H-1:", rows, "passes:", passes)
    print(f"-> {p}")
    notify(f"MIRA 18H done: decisive@90 {', '.join(f'{k}={v:.2f}' for k, v in rows.items())} passes={passes}")


if __name__ == "__main__":
    main()
