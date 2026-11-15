#!/usr/bin/env python
"""
research/phase18_f_channel_validity.py — Phase 18 Block 18F: do specific
channels know the lab better than the aggregate? PHASE18_PREREG.md "18F".

Data: Phase 17 R-f's setup via results/phase18/arrows_pair_table.json
(research/phase18_arrows_pairs.py): ARROWS3 YBCO primary endpoint (1,915
within-temperature pairs, |delta| >= 5 wt%), ASTRAL 35 pairs flagged small.
Q = lab winner (+-1). CF = carbonate-free rule (-1/0/1).

Per validator channel c (all 10, on pairs where c is gradeable on both
routes): R_c = sign(c_a - c_b); I(R_c; Q | CF) in bits (R-f's estimator:
plug-in MI per CF stratum, stratum-weighted). Aggregates on the full pair
set, exactly as R-f: R2 = GDPO vote (5-channel checker), R4 = sign of the
ranker-v2 score difference, R5 = comparator vote (gate decisions included,
both-gate-fail excluded).

Uncertainty (decided before running):
  * CI: cluster bootstrap over recipes (R-f's scheme: resample recipe
    labels, a pair enters with multiplicity count(a)*count(b)), 10,000
    resamples, seed 20261115, percentile 95% CI.
  * The plug-in CMI is >= 0 and biased upward, so a percentile CI almost
    never contains 0 literally. "Corrected CI excludes 0" is therefore
    operationalised as: stratified permutation test (Q shuffled within CF
    strata, 2,000 permutations, one-sided p for CMI >= observed), Holm-
    corrected across the 10 channels at 0.05; the reported Holm-level CI is
    the percentile bootstrap CI of the null-centred CMI (CMI minus the
    permutation-null mean) at level 1 - 0.05/(m - rank + 1).
  * Best channel minus aggregate: best = highest point CMI among channels
    (selected on the full data; selection noted). Compared with each
    aggregate on the SAME pair subset (pairs where the best channel is
    gradeable), paired over the same bootstrap draws.
[C-measure], no direction: number of channels surviving; best - aggregate
with CI. Interpretation fixed in the pre-registration: a channel beating the
aggregate => aggregation destroys validity; none => no channel knows the lab.
[E]: ranker-v2 objectives as channels (a separate Holm family).

Usage (tmux, CPU): scripts/run_phase18_block.sh 18F research/phase18_f_channel_validity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase18_arrows_pairs import load_table  # noqa: E402
from research.phase18_common import SEED, N_BOOT, provenance, write_json, notify  # noqa: E402

N_PERM = 2000


def sgn(x):
    return 0 if x is None else (1 if x > 1e-9 else (-1 if x < -1e-9 else 0))


def cmi(r, q, cf, w):
    """I(R;Q|CF) bits with integer weights w. r,q,cf int arrays."""
    tot = w.sum()
    if tot == 0:
        return 0.0
    out = 0.0
    for s in (-1, 0, 1):
        m = (cf == s) & (w > 0)
        ws = w[m].sum()
        if ws == 0:
            continue
        rs, qs, ww = r[m] + 1, (q[m] + 1) // 2, w[m]
        J = np.zeros((3, 2))
        np.add.at(J, (rs, qs), ww)
        P = J / ws
        pr, pq = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        out += (ws / tot) * float((P[nz] * np.log2(P[nz] / (pr @ pq)[nz])).sum())
    return max(out, 0.0)


class Boot:
    def __init__(self, set_a, set_b, rng):
        labels = sorted(set(set_a) | set(set_b))
        idx = {l: i for i, l in enumerate(labels)}
        self.ia = np.array([idx[x] for x in set_a])
        self.ib = np.array([idx[x] for x in set_b])
        n = len(labels)
        self.W = np.empty((N_BOOT, len(set_a)), dtype=np.int64)
        for k in range(N_BOOT):
            c = np.bincount(rng.integers(0, n, n), minlength=n)
            self.W[k] = c[self.ia] * c[self.ib]


def pct(v, lo=0.025, hi=0.975):
    v = np.sort(np.asarray(v))
    return [float(v[int(lo * len(v))]), float(v[min(int(hi * len(v)), len(v) - 1)])]


def analyse(rows, label, rng):
    q = np.array([r["actual"] for r in rows])
    cf = np.array([r["base_carbfree"] for r in rows])
    keep_q = q != 0
    rows = [r for r, k in zip(rows, keep_q) if k]
    q, cf = q[keep_q], cf[keep_q]
    boot = Boot([r["set_a"] for r in rows], [r["set_b"] for r in rows], rng)
    ones = np.ones(len(rows), dtype=np.int64)

    def stat(r, mask):
        point = cmi(r[mask], q[mask], cf[mask], ones[mask])
        bs = np.array([cmi(r[mask], q[mask], cf[mask], boot.W[k][mask]) for k in range(N_BOOT)])
        null = []
        qm, cfm = q[mask].copy(), cf[mask]
        for _ in range(N_PERM):
            qp = qm.copy()
            for s in (-1, 0, 1):
                ii = np.where(cfm == s)[0]
                qp[ii] = qp[rng.permutation(ii)]
            null.append(cmi(r[mask], qp, cfm, ones[mask]))
        null = np.array(null)
        p = (1 + int((null >= point - 1e-12).sum())) / (1 + N_PERM)
        return {"point_bits": point, "ci95_bits": pct(bs), "perm_p": p, "null_mean_bits": float(null.mean()),
                "n_pairs": int(mask.sum())}, bs, float(null.mean())

    aggs = {
        "R2_checker_vote": np.array([sgn(r["R2_vote"]) if r["R2_vote"] is not None else 0 for r in rows]),
        "R4_ranker": np.array([sgn(r["R4_margin"]) for r in rows]),
        "R5_comparator": np.array([sgn(r["R5"]["margin"]) for r in rows]),
    }
    agg_def = {"R2_checker_vote": np.array([r["R2_vote"] is not None for r in rows]),
               "R4_ranker": np.ones(len(rows), bool),
               "R5_comparator": np.array([r["R5"]["margin"] is not None for r in rows])}
    out = {"label": label, "n_pairs": len(rows), "aggregates": {}, "channels": {}, "E_ranker_channels": {}}
    agg_boot = {}
    for name, r in aggs.items():
        s, bs, _ = stat(r, agg_def[name])
        out["aggregates"][name] = s
        agg_boot[name] = bs
        print(f"  [{label}] {name}: {s['point_bits']:.4f} CI {s['ci95_bits']} p={s['perm_p']:.4f} n={s['n_pairs']}", flush=True)

    def family(chan_names, side_key, store):
        res, boots, nulls = {}, {}, {}
        for c in chan_names:
            va = [r["a"][side_key].get(c) for r in rows]
            vb = [r["b"][side_key].get(c) for r in rows]
            mask = np.array([x is not None and y is not None for x, y in zip(va, vb)])
            rc = np.array([sgn(x - y) if (x is not None and y is not None) else 0 for x, y in zip(va, vb)])
            if mask.sum() < 10 or len(set(rc[mask])) < 2:
                res[c] = {"point_bits": 0.0, "n_pairs": int(mask.sum()), "perm_p": 1.0,
                          "note": "fewer than 10 gradeable pairs or no variation in sign(c_a - c_b)"}
                continue
            s, bs, nm = stat(rc, mask)
            s["mask"] = mask
            s["rc"] = rc
            res[c], boots[c], nulls[c] = s, bs, nm
            print(f"  [{label}] {side_key}.{c}: {s['point_bits']:.4f} CI {s['ci95_bits']} p={s['perm_p']:.4f} "
                  f"n={s['n_pairs']}", flush=True)
        # Holm
        order = sorted(res, key=lambda c: res[c]["perm_p"])
        m = len(order)
        stop = False
        for rank, c in enumerate(order):
            alpha_i = 0.05 / (m - rank)
            survives = (not stop) and res[c]["perm_p"] <= alpha_i
            if not survives:
                stop = True
            res[c]["holm_alpha"] = alpha_i
            res[c]["holm_survives"] = survives
            if c in boots:
                lvl = alpha_i
                res[c]["holm_level_ci_null_centred_bits"] = pct(boots[c] - nulls[c], lvl / 2, 1 - lvl / 2)
        store.update({c: {k: v for k, v in s.items() if k not in ("mask", "rc")} for c, s in res.items()})
        return res, boots

    vres, vboots = family(load_table()["channels_validator"], "val_ch", out["channels"])
    family(load_table()["channels_ranker"], "ranker_ch", out["E_ranker_channels"])
    n_surv = sum(1 for s in vres.values() if s.get("holm_survives"))
    graded = {c: s for c, s in vres.items() if "mask" in s}
    best_cmp = None
    if graded:
        best = max(graded, key=lambda c: graded[c]["point_bits"])
        mask = graded[best]["mask"]
        best_cmp = {"best_channel": best, "best_point_bits": graded[best]["point_bits"], "vs": {}}
        for name, r in aggs.items():
            mm = mask & agg_def[name]
            rc = graded[best]["rc"]
            pt = cmi(rc[mm], q[mm], cf[mm], ones[mm]) - cmi(r[mm], q[mm], cf[mm], ones[mm])
            bs = np.array([cmi(rc[mm], q[mm], cf[mm], boot.W[k][mm]) - cmi(r[mm], q[mm], cf[mm], boot.W[k][mm])
                           for k in range(N_BOOT)])
            ci = pct(bs)
            best_cmp["vs"][name] = {"diff_bits_same_pairs": pt, "ci95": ci, "n_pairs": int(mm.sum()),
                                    "ci_excludes_0": ci[0] > 0 or ci[1] < 0}
        best_cmp["selection_note"] = "best channel chosen on the full data by point CMI; the CI does not account for selection"
    out["n_channels_surviving_holm"] = n_surv
    out["best_minus_aggregate"] = best_cmp
    return out


def main():
    prov = provenance()
    rng = np.random.default_rng(SEED)
    t = load_table()
    ybco = analyse(t["ybco_primary_5wtpct"], "YBCO_primary", rng)
    ast = analyse(t["astral_35_small"], "ASTRAL_35_SMALL", rng)
    b = ybco["best_minus_aggregate"]
    if b:
        beats = [k for k, v in b["vs"].items() if v["ci95"][0] > 0]
        interp = (f"{b['best_channel']} beats {beats} => aggregation destroys validity (pre-registered reading)"
                  if beats else "no channel beats the aggregates => no channel knows the lab better (pre-registered reading)")
    else:
        interp = "no gradeable channel"
    doc = {**prov, "block": "18F", "n_boot": N_BOOT, "n_perm": N_PERM,
           "C_measure": {"n_validator_channels_surviving_holm_YBCO": ybco["n_channels_surviving_holm"],
                         "best_minus_aggregate_YBCO": b, "interpretation": interp},
           "ybco_primary": ybco, "astral_35_small": ast}
    p = write_json("channel_validity.json", doc)
    print("channels surviving Holm (YBCO):", ybco["n_channels_surviving_holm"])
    print("best vs aggregates:", b)
    print(interp)
    print(f"-> {p}")
    notify(f"MIRA 18F done: {ybco['n_channels_surviving_holm']} channels survive Holm; {interp[:80]}")


if __name__ == "__main__":
    main()
