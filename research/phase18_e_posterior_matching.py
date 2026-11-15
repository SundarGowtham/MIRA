#!/usr/bin/env python
"""
research/phase18_e_posterior_matching.py — Phase 18 Block 18E: posterior
matching instead of best sample. PHASE18_PREREG.md "18E".

Models: Base, Corpus SFT, RS-SFT, E3, E3c, GDPO-300, Phase 12 @300, E1 @300.
p_m = empirical canonical-recipe distribution from existing ASTRAL samples
(results/phase18/recipes_by_model.json). Base and RS-SFT have 232
samples/target (200 reference + 32 resampling); to keep sample size equal
across models -- coverage depends on n -- they use their 32-sample
"resampling_rate" draw, as Phase 16's hit computation did (decided before
running).

Lab-valid set V(x) (Block 0's definition, decided before Phase 18 results):
primary = ASTRAL sets with best measured purity > 0.5; targets with empty V
are excluded. Sensitivity = Phase 16's singleton {rule-picked set}, defined
for every target.

Metrics per (model, target): lab-valid mass p_m(V); coverage |V cap supp|/|V|;
MMD between p_m and uniform(V) with kernel k(y,y') = 1 - J(y,y') (J = Jaccard
distance; MMD^2 = E_pp k + E_vv k - 2 E_pv k, MMD = sqrt(max(0, .)));
near-miss distance E_{y~p_m} min_{v in V} J(y,v).
Aggregation: mean over targets. Cluster bootstrap over targets, 10,000,
seed 20261115. Targets: clean 22 primary, full 35 secondary.

Prediction [C] P-18E-1: E3's MMD is lower than Phase 12 @300's; the cluster-
bootstrap CI on the difference (E3 - Phase12, mean over targets) excludes 0.

[E]: Spearman rank agreement between the models' MMD ordering and Phase 16's
hit ordering (hit = any of the 32 samples equals the rule-picked set);
share of practice samples within one precursor swap of a valid set
(|y symmetric-difference v| <= 2 for some v in V) and not in V.

Usage (tmux, CPU): scripts/run_phase18_block.sh 18E research/phase18_e_posterior_matching.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase18_common import (  # noqa: E402
    SEED, N_BOOT, provenance, write_json, notify, load_recipes, recipes, target_sets, astral_lab, jaccard_distance,
)

MODELS = ["base", "corpus_sft", "rs_sft", "e3", "e3c", "gdpo_v3_ckpt300", "phase12_ckpt300", "e1_ckpt300"]
PRACTICE = ["gdpo_v3_ckpt300", "phase12_ckpt300", "e1_ckpt300"]


def samples_for(cache, m, t):
    if m in ("base", "rs_sft"):
        return recipes(cache, m, "astral", t, group="resampling_rate")
    return recipes(cache, m, "astral", t)


def metrics(samples, V):
    n = len(samples)
    if n == 0 or not V:
        return None
    cnt = Counter(samples)
    ys = list(cnt)
    p = np.array([cnt[y] / n for y in ys])
    Kpp = np.array([[1 - jaccard_distance(a, b) for b in ys] for a in ys])
    Kvv = np.array([[1 - jaccard_distance(a, b) for b in V] for a in V])
    Kpv = np.array([[1 - jaccard_distance(a, b) for b in V] for a in ys])
    u = np.full(len(V), 1 / len(V))
    mmd2 = float(p @ Kpp @ p + u @ Kvv @ u - 2 * p @ Kpv @ u)
    Vs = set(V)
    near = float(sum(cnt[y] / n * min(jaccard_distance(y, v) for v in V) for y in ys))
    one_swap = sum(cnt[y] for y in ys if y not in Vs and any(len(set(y) ^ set(v)) <= 2 for v in V)) / n
    return {"lab_valid_mass": sum(cnt[y] for y in ys if y in Vs) / n,
            "coverage": len(Vs & set(ys)) / len(Vs), "mmd": float(np.sqrt(max(0.0, mmd2))),
            "near_miss": near, "share_within_one_swap_not_valid": one_swap, "n": n}


def boot_mean_diff(a, b, targets, rng):
    ts = [t for t in targets if a.get(t) is not None and b.get(t) is not None]
    da = np.array([a[t] - b[t] for t in ts])
    idx = rng.integers(0, len(ts), (N_BOOT, len(ts)))
    bs = da[idx].mean(1)
    return {"diff": float(da.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_targets": len(ts)}


def run(cache, lab, targets, vdef, rng, hits):
    per = {m: {} for m in MODELS}
    for m in MODELS:
        for t in targets:
            V = lab[t]["lab_success_sets"] if vdef == "purity" else [lab[t]["phase16_hit_set"]]
            V = [tuple(v) for v in V]
            r = metrics(samples_for(cache, m, t), V)
            if r is not None:
                per[m][t] = r
    used = sorted(set.intersection(*[set(per[m]) for m in MODELS]))
    summ = {m: {k: float(np.mean([per[m][t][k] for t in used])) for k in
                ("lab_valid_mass", "coverage", "mmd", "near_miss", "share_within_one_swap_not_valid")}
            for m in MODELS}
    p1 = boot_mean_diff({t: per["e3"][t]["mmd"] for t in used}, {t: per["phase12_ckpt300"][t]["mmd"] for t in used},
                        used, rng)
    p1["passes"] = p1["diff"] < 0 and p1["ci95"][1] < 0
    mmd_order = [summ[m]["mmd"] for m in MODELS]
    hit_counts = [sum(1 for t in used if hits[m].get(t)) for m in MODELS]
    rho, pv = spearmanr([-x for x in mmd_order], hit_counts)
    prac_share = float(np.mean([per[m][t]["share_within_one_swap_not_valid"] for m in PRACTICE for t in used]))
    return {"targets_used": used, "n_targets": len(used), "summary_by_model": summ,
            "P_18E_1_E3_minus_Phase12_mmd": p1,
            "E_rank_agreement": {"spearman_neg_mmd_vs_hits": float(rho), "p": float(pv),
                                 "hits_by_model_on_these_targets": dict(zip(MODELS, hit_counts))},
            "E_practice_share_within_one_swap_not_valid": prac_share,
            "per_target": {m: per[m] for m in MODELS}}


def main():
    prov = provenance()
    cache = load_recipes()
    sets = target_sets()
    lab = astral_lab()
    hits = {m: {t: any(tuple(y) == tuple(lab[t]["predicted"]) for y in samples_for(cache, m, t))
                for t in sets["astral_35"]} for m in MODELS}
    out = {}
    for vdef in ("purity", "phase16_singleton"):
        for tname in ("astral_clean_22", "astral_35"):
            rng = np.random.default_rng(SEED)
            res = run(cache, lab, sets[tname], vdef, rng, hits)
            out[f"{vdef}__{tname}"] = res
            p1 = res["P_18E_1_E3_minus_Phase12_mmd"]
            print(f"== V={vdef} targets={tname} (n={res['n_targets']})", flush=True)
            for m, s in res["summary_by_model"].items():
                print(f"   {m:18s} mass={s['lab_valid_mass']:.3f} cov={s['coverage']:.3f} mmd={s['mmd']:.3f} "
                      f"near={s['near_miss']:.3f} oneswap={s['share_within_one_swap_not_valid']:.3f}", flush=True)
            print(f"   P-18E-1 E3-P12 mmd diff={p1['diff']:.4f} CI={p1['ci95']} passes={p1['passes']}", flush=True)
            print(f"   [E] rank agreement {res['E_rank_agreement']}", flush=True)
    primary = out["purity__astral_clean_22"]["P_18E_1_E3_minus_Phase12_mmd"]
    doc = {**prov, "block": "18E",
           "P_18E_1": {"statement": "E3's MMD lower than Phase 12 @300's; cluster-bootstrap CI on the difference "
                                    "excludes 0 (primary: purity-defined V, clean 22).", **primary},
           "sample_size_note": "base and rs_sft use their 32-sample resampling_rate draw; every model n<=32/target",
           "results": out}
    p = write_json("posterior_matching.json", doc)
    print(f"-> {p}")
    notify(f"MIRA 18E done: P-18E-1 diff={primary['diff']:.3f} CI={[round(x,3) for x in primary['ci95']]} "
           f"passes={primary['passes']}")


if __name__ == "__main__":
    main()
