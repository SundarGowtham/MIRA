#!/usr/bin/env python
"""
research/phase19_d_kernel.py -- Phase 19 Block D (GPU, backward at fixed
weights). docs/phases/PHASE19_PREREG.md "Block D".

Gradients: newly written Kronecker sketch (research/phase19_grad.py)
following PHASE17_PREREG.md lines 286-293, every LoRA-targeted module, k=8
per side, seeded (20261116). Gradient checkpointing (non-reentrant); no
truncation of any sequence beyond what training itself applied.

Sketch check: exact (autograd .grad of 4 base weights) vs sketched per-module
inner products on 50 random pairs of probe strings at Base. Pearson >= 0.95
in every one of the 4 modules.

Update directions g_s = sum_i w_i grad log p(y_i), at the before-model:
  S2/S3/S4 (SFT): training sequences AS TRAINED -- chat-template text truncated
    to its first 1024 tokens (TRL max_length=1024, keep_start; Block 0), loss on
    every token; w_i = 1/|D|.
  S6/S7 (RL): first 50 groups (400 completions, steps 0-24), prompt + full
    completion (+ <|im_end|> when shorter than the 8192-token cap), loss on
    completion tokens; w_i = exact GDPO advantage (Block 0 reconstruction).
  S5 (RL, secondary): not run in D -- its run-1 training log uses a different
    reward configuration that Block 0 did not reconstruct.
Predicted change: D_hat(y') = <sketch(y'), g_s> (up to scale), probe strings
sketched at the before-model.

Lazy check: centered kernel alignment (linear CKA) between K at before and
after weights over all probe strings (pooled); per-target median [E].

P-19D-1: per-target Spearman(D_hat, D exact from Block G scores); median > 0,
  cluster-bootstrap CI excludes 0, for S3, S4, S6, S7.
P-19D-2 (S6, S7): reach ratio per usable target (before-model sampled >= 1
  recipe for the target AND >= 3 top-30 recipes with zero before-model count);
  median < 1 with CI excluding 1; < 10 usable -> untestable. [E] length-
  normalized ratio; partial Spearman(|D_hat|, unsampled | length, grad norm).
P-19D-3: S3 on the 20 held-out probe targets (none overlap E3/E3c training,
  Block 0); positives = E3 count >= 1 and Base count == 0; AUC of D_hat > 0.5,
  CI excludes 0.5.
[C-measure]: PSR/NSR halves of RL's g; within-target slope of standardized
  D_hat on l_before (positive = favours already-likely strings = narrowing);
  participation ratio of K at each before-model.
[E]: kernel block structure by alkali-source family (carbonate / oxide /
  other; alkali-containing targets only), cosine-normalised K, within minus
  across family mean, 1,000 within-target label permutations.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr, rankdata  # noqa: E402

from research.phase19_common import (  # noqa: E402
    MultiAdapterModel, load_probe_strings, load_scores, provenance, write_json, notify, context_text,
    answer_token_mask, render_prompt, closed_prompt, STEPS, METHOD, MODELS, SEED, N_BOOT, OUT_DIR,
)
from research.phase19_grad import Sketcher, prepare_for_backward, seq_logprob_sketch  # noqa: E402

CACHE_DIR = REPO_ROOT / "runs" / "phase19_cache" / "sketches"
D_STEPS = ["S3", "S4", "S6", "S7", "S2"]
SFT_FILES = {"S2": "data/rs_sft/rs_sft_train.jsonl", "S3": "data/e3/e3_train.jsonl", "S4": "data/e3c/e3c_train.jsonl"}
PROBE = json.loads((REPO_ROOT / "results/phase17/probe_targets.json").read_text())
MAX_COMPLETION = 8192


def cfg(m):
    return None if m == "base" else m


def probe_sketches(mm, sk, ps, model, batch=8):
    path = CACHE_DIR / f"probe_{model}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["S"], list(z["keys"]), z["f"]
    mm.set_all(cfg(model))
    S_all, keys, fs = [], [], []
    for t in sorted(ps):
        rows = ps[t]
        ctx = context_text(mm.tok, t)
        seqs = [answer_token_mask(mm.tok, ctx, ctx + r["string"]) for r in rows]
        for b0 in range(0, len(seqs), batch):
            ch = seqs[b0:b0 + batch]
            S, vals, _ = seq_logprob_sketch(mm, sk, [x for x, _ in ch], [m for _, m in ch])
            S_all.append(S.cpu().numpy())
            fs.extend(vals)
        keys.extend([f"{t}||{r['recipe_key']}" for r in rows])
    S = np.concatenate(S_all)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, S=S, keys=np.array(keys, dtype=object), f=np.array(fs))
    return S, keys, np.array(fs)


def update_sketch(mm, sk, step, before):
    path = CACHE_DIR / f"update_{step}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["S"], z["w"], z["ntok"]
    mm.set_all(cfg(before))
    tok = mm.tok
    S_all, w, ntok = [], [], []
    if METHOD[step] == "SFT":
        rows = [json.loads(l) for l in open(REPO_ROOT / SFT_FILES[step]) if l.strip()]
        seqs = []
        for r in rows:
            txt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]},
                                           {"role": "assistant", "content": r["completion"]}], tokenize=False)
            ids = tok(txt, add_special_tokens=False).input_ids[:1024]
            seqs.append(ids)
        for b0 in range(0, len(seqs), 4):
            ch = seqs[b0:b0 + 4]
            S, _, _ = seq_logprob_sketch(mm, sk, ch, [[True] * len(x) for x in ch])
            S_all.append(S.cpu().numpy())
            ntok.extend(len(x) for x in ch)
        w = [1.0 / len(seqs)] * len(seqs)
    else:
        rows = [json.loads(l) for l in open(OUT_DIR / f"rl_update_groups_{step}.jsonl")]
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        for i, r in enumerate(rows):
            pid = tok(render_prompt(tok, r["target"]), add_special_tokens=False).input_ids
            cid = tok(r["completion"], add_special_tokens=False).input_ids
            if len(cid) < MAX_COMPLETION:
                cid = cid + [eos]
            ids = pid + cid
            S, _, _ = seq_logprob_sketch(mm, sk, [ids], [[False] * len(pid) + [True] * len(cid)])
            S_all.append(S.cpu().numpy())
            w.append(r["advantage"])
            ntok.append(len(cid))
            if i % 50 == 0:
                print(f"    {step} update {i}/{len(rows)}", flush=True)
    S = np.concatenate(S_all)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, S=S, w=np.array(w), ntok=np.array(ntok))
    return S, np.array(w), np.array(ntok)


def sketch_check(mm, ps, rng):
    """Exact (autograd .grad) vs sketched inner products, 50 random pairs, 4 modules, at Base."""
    mm.set_all(None)
    names = [(5, "q_proj"), (17, "v_proj"), (26, "down_proj"), (33, "o_proj")]
    idxs = [[i for i, (li, qn, _m) in enumerate(mm.lora) if li == L and qn.endswith(n)][0] for L, n in names]
    sk = Sketcher(mm, SEED)
    Ws = [mm.lora[i][2].get_base_layer().weight for i in idxs]
    flat = [(t, r) for t in sorted(ps) for r in ps[t]]
    pick = rng.choice(len(flat), 20, replace=False)
    exact = [[] for _ in idxs]
    sket = [[] for _ in idxs]
    for W in Ws:
        W.requires_grad_(True)
    for p in pick:
        t, r = flat[p]
        ctx = context_text(mm.tok, t)
        ids, m = answer_token_mask(mm.tok, ctx, ctx + r["string"])
        S, _, _ = seq_logprob_sketch(mm, sk, [ids], [m])
        S = S[0].reshape(len(sk.mods), 64)
        for k, (i, W) in enumerate(zip(idxs, Ws)):
            exact[k].append(W.grad.float().flatten().cpu())
            sket[k].append(S[i].cpu())
            W.grad = None
    for W in Ws:
        W.requires_grad_(False)
    pairs = set()
    while len(pairs) < 50:
        a, b = sorted(rng.choice(20, 2, replace=False).tolist())
        pairs.add((a, b))
    pairs = sorted(pairs)
    out = {}
    for k, (L, n) in enumerate(names):
        e = np.array([float(exact[k][a] @ exact[k][b]) for a, b in pairs])
        s = np.array([float(sket[k][a] @ sket[k][b]) for a, b in pairs])
        out[f"layer{L}.{n}"] = {"pearson": float(np.corrcoef(e, s)[0, 1]),
                                "exact_ip_range": [float(e.min()), float(e.max())]}
    del exact
    return {"modules": out, "n_pairs": 50, "n_strings": 20,
            "min_pearson": min(v["pearson"] for v in out.values()),
            "passes_0.95_every_module": all(v["pearson"] >= 0.95 for v in out.values())}


def cka(K1, K2):
    n = K1.shape[0]
    H = np.eye(n) - 1.0 / n
    a, b = H @ K1 @ H, H @ K2 @ H
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def pr(K):
    ev = np.clip(np.linalg.eigvalsh(K), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def boot_median(vals, rng):
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), (N_BOOT, len(v)))
    bs = np.median(v[idx], axis=1)
    return float(np.median(v)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def auc(score, label):
    r = rankdata(score)
    n1 = label.sum()
    n0 = len(label) - n1
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def family_of(recipe, target):
    from research.distributional.alkali_source_share import target_alkalis, classify_alkali_source
    al = sorted(target_alkalis(target))
    if not al:
        return None
    c = classify_alkali_source(list(recipe), al[0])
    return {"bare_oxide": "oxide", "carbonate": "carbonate"}.get(c, "other")


def within_slope(dhat, lb, tg, targets, rng):
    """Pooled within-target OLS slope of standardized dhat on l_before; cluster bootstrap."""
    z = dhat / dhat.std()
    xs, ys = {}, {}
    for t in targets:
        m = tg == t
        if m.sum() < 3:
            continue
        xs[t] = lb[m] - lb[m].mean()
        ys[t] = z[m] - z[m].mean()
    T = sorted(xs)
    sxy = np.array([(xs[t] * ys[t]).sum() for t in T])
    sxx = np.array([(xs[t] ** 2).sum() for t in T])
    idx = rng.integers(0, len(T), (N_BOOT, len(T)))
    bs = sxy[idx].sum(1) / sxx[idx].sum(1)
    return {"slope": float(sxy.sum() / sxx.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_targets": len(T)}


def main():
    prov = provenance()
    ps = load_probe_strings()
    scores = load_scores()
    rng = np.random.default_rng(SEED)
    mm = MultiAdapterModel([m for m in MODELS if m != "base"])
    prepare_for_backward(mm, checkpointing=True)
    t0 = time.time()
    check = sketch_check(mm, ps, rng)
    print(f"sketch check: {check['modules']} passes={check['passes_0.95_every_module']} [{time.time() - t0:.0f}s]", flush=True)
    sk = Sketcher(mm, SEED)
    needed_probe = ["base", "rs_sft", "e3", "e3c", "phase12_ckpt300", "e1_ckpt300"]
    P = {}
    for m in needed_probe:
        t0 = time.time()
        P[m] = probe_sketches(mm, sk, ps, m)
        print(f"probe sketches {m}: {P[m][0].shape} [{time.time() - t0:.0f}s]", flush=True)
    keys = P["base"][1]
    tg = np.array([k.split("||")[0] for k in keys])
    rk = [k.split("||")[1] for k in keys]
    rows_by_key = {f"{t}||{r['recipe_key']}": r for t in ps for r in ps[t]}
    targets = sorted(ps)
    U = {}
    for s in D_STEPS:
        t0 = time.time()
        U[s] = update_sketch(mm, sk, s, STEPS[s][0])
        print(f"update sketches {s}: {U[s][0].shape} [{time.time() - t0:.0f}s]", flush=True)
        notify(f"MIRA 19 Block D: update sketches {s} done")

    res = {}
    for s in D_STEPS:
        b, a = STEPS[s]
        Sb, keys_b, _ = P[b]
        assert list(keys_b) == list(keys)
        Su, w, ntok = U[s]
        g = (w[:, None] * Su).sum(0)
        dhat = Sb @ g
        lb = np.array([scores[(b, t)][k] for t, k in zip(tg, rk)])
        la = np.array([scores[(a, t)][k] for t, k in zip(tg, rk)])
        d = la - lb
        # D-1
        rhos = {}
        for t in targets:
            m = tg == t
            if m.sum() >= 3:
                rhos[t] = float(spearmanr(dhat[m], d[m]).correlation)
        med, ci = boot_median(list(rhos.values()), rng)
        r = {"before": b, "after": a, "method": METHOD[s], "n_update_sequences": int(len(w)),
             "update_tokens_mean": float(ntok.mean()),
             "D1": {"median_spearman": med, "ci95": ci, "n_targets": len(rhos), "per_target": rhos,
                    "passes": med > 0 and ci[0] > 0},
             "pooled_spearman_dhat_vs_delta": float(spearmanr(dhat, d).correlation)}
        # lazy check
        if a in P:
            Sa = P[a][0]
            Kb, Ka = Sb @ Sb.T, Sa @ Sa.T
            per_t = [cka(Kb[np.ix_(tg == t, tg == t)], Ka[np.ix_(tg == t, tg == t)]) for t in targets if (tg == t).sum() >= 3]
            r["lazy_check"] = {"cka_pooled": cka(Kb, Ka), "cka_per_target_median_E": float(np.median(per_t)),
                               "below_0.8_note": None}
            if r["lazy_check"]["cka_pooled"] < 0.8:
                r["lazy_check"]["below_0.8_note"] = "CKA < 0.8: D-1 describes only the start of training."
        # slopes (C-measure)
        r["slope_dhat_on_l_before"] = within_slope(dhat, lb, tg, targets, rng)
        if METHOD[s] == "RL":
            for name, sel in (("PSR", w > 0), ("NSR", w < 0)):
                gh = (w[sel, None] * Su[sel]).sum(0)
                r[f"slope_{name}"] = within_slope(Sb @ gh, lb, tg, targets, rng)
                r[f"D1_{name}_pooled_spearman"] = float(spearmanr(Sb @ gh, d).correlation)
            # D-2 reach ratio
            cnt_b = np.array([rows_by_key[k]["counts"][b] for k in keys])
            ratios, ratios_len, usable = {}, {}, []
            ntoks = np.array([scores and 0 for _ in keys], float)
            # answer-token count of each string (from Block G file)
            sc_rows = {}
            for line in (OUT_DIR / "string_scores.jsonl").read_text().splitlines():
                q = json.loads(line)
                if q["model"] == b:
                    sc_rows[f"{q['target']}||{q['recipe_key']}"] = q["n_answer_tokens"]
            ntoks = np.array([sc_rows[k] for k in keys], float)
            for t in targets:
                m = tg == t
                uns = m & (cnt_b == 0)
                sam = m & (cnt_b > 0)
                if sam.sum() >= 1 and uns.sum() >= 3:
                    usable.append(t)
                    ratios[t] = float(np.abs(dhat[uns]).mean() / np.abs(dhat[sam]).mean())
                    ratios_len[t] = float((np.abs(dhat[uns]) / ntoks[uns]).mean() / (np.abs(dhat[sam]) / ntoks[sam]).mean())
            D2 = {"n_usable_targets": len(usable), "usable_targets": usable, "per_target_ratio": ratios}
            if len(usable) < 10:
                D2.update({"status": "untestable", "passes": None})
            else:
                med2, ci2 = boot_median(list(ratios.values()), rng)
                D2.update({"status": "tested", "median_ratio": med2, "ci95": ci2, "passes": med2 < 1 and ci2[1] < 1})
            if ratios_len:
                ml, cl = boot_median(list(ratios_len.values()), rng)
                D2["E_length_normalized"] = {"median_ratio": ml, "ci95": cl}
                mu = np.isin(tg, usable)
                gn = np.linalg.norm(Sb, axis=1)
                y = rankdata(np.abs(dhat[mu]))
                x = (cnt_b[mu] == 0).astype(float)
                Z = np.column_stack([np.ones(mu.sum()), rankdata(ntoks[mu]), rankdata(gn[mu])] +
                                    [(tg[mu] == t).astype(float) for t in usable[1:]])
                ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
                rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
                D2["E_partial_spearman_absdhat_unsampled_given_len_gradnorm_target"] = float(np.corrcoef(ry, rx)[0, 1])
            r["D2"] = D2
        # D-3
        if s == "S3":
            held = [t for t in PROBE["probe_heldout_20"]]
            mh = np.isin(tg, held)
            lab = np.array([(rows_by_key[k]["counts"]["e3"] >= 1 and rows_by_key[k]["counts"]["base"] == 0)
                            for k in keys], int)
            D3 = {"n_targets": len(held), "n_positives": int(lab[mh].sum()), "n_negatives": int((1 - lab[mh]).sum())}
            if len(held) < 10 or lab[mh].sum() == 0:
                D3.update({"status": "untestable", "passes": None})
            else:
                A = auc(dhat[mh], lab[mh])
                bs = []
                for _ in range(N_BOOT):
                    pick = rng.choice(held, len(held), replace=True)
                    ii = np.concatenate([np.where(tg == t)[0] for t in pick])
                    if 0 < lab[ii].sum() < len(ii):
                        bs.append(auc(dhat[ii], lab[ii]))
                ci3 = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                D3.update({"status": "tested", "auc": A, "ci95": ci3, "passes": A > 0.5 and ci3[0] > 0.5,
                           "n_boot_valid": len(bs)})
            r["D3"] = D3
        res[s] = r
        print(f"== {s}: D1 median {med:.3f} CI {ci}; lazy {r.get('lazy_check', {}).get('cka_pooled')}; "
              f"slope {r['slope_dhat_on_l_before']['slope']:.3f}"
              + (f"; D2 {r['D2'].get('median_ratio')} {r['D2'].get('ci95')} usable={r['D2']['n_usable_targets']}" if 'D2' in r else "")
              + (f"; D3 {r['D3']}" if 'D3' in r else ""), flush=True)

    # effective dimension and family structure at each before-model
    eff, fam = {}, {}
    for b in ("base", "rs_sft"):
        Sb = P[b][0]
        K = Sb @ Sb.T
        per_t = [pr(K[np.ix_(tg == t, tg == t)]) for t in targets if (tg == t).sum() >= 3]
        eff[b] = {"participation_ratio_pooled": pr(K), "participation_ratio_per_target_median": float(np.median(per_t)),
                  "n_strings": int(len(K))}
        nrm = np.linalg.norm(Sb, axis=1)
        C = K / np.outer(nrm, nrm)
        labs = np.array([family_of(rows_by_key[k]["recipe"], t) for k, t in zip(keys, tg)], dtype=object)

        def stat(lv):
            wi, ac = [], []
            for t in targets:
                ii = np.where((tg == t) & (lv != None))[0]  # noqa: E711
                for x in range(len(ii)):
                    for y in range(x + 1, len(ii)):
                        (wi if lv[ii[x]] == lv[ii[y]] else ac).append(C[ii[x], ii[y]])
            return (np.mean(wi) - np.mean(ac)) if wi and ac else np.nan
        obs = stat(labs)
        null = []
        for _ in range(1000):
            lp = labs.copy()
            for t in targets:
                ii = np.where((tg == t) & (labs != None))[0]  # noqa: E711
                lp[ii] = lp[rng.permutation(ii)]
            null.append(stat(lp))
        null = np.array(null)
        fam[b] = {"within_minus_across_cosine": float(obs), "perm_p_one_sided": float((1 + (null >= obs).sum()) / 1001),
                  "null_mean": float(np.nanmean(null)), "label_counts": {k: int((labs == k).sum()) for k in ("carbonate", "oxide", "other")}}
    # figdata: 3 targets with most strings, cosine kernel at Base, sorted by family then l_before
    fig = {}
    Sb = P["base"][0]
    nrm = np.linalg.norm(Sb, axis=1)
    for t in sorted(targets, key=lambda t: -(tg == t).sum())[:3]:
        ii = np.where(tg == t)[0]
        fams = [family_of(rows_by_key[keys[i]]["recipe"], t) or "n/a" for i in ii]
        lbs = [scores[("base", t)][rk[i]] for i in ii]
        order = sorted(range(len(ii)), key=lambda j: (fams[j], -lbs[j]))
        jj = ii[order]
        C = (Sb[jj] @ Sb[jj].T) / np.outer(nrm[jj], nrm[jj])
        fig[t] = {"recipes": [rk[i] for i in jj], "family": [fams[j] for j in order], "cosine_kernel_base": C.tolist()}
    prim = ("S3", "S4", "S6", "S7")
    doc = {**prov, "block": "D", "sketch": {"k_per_side": 8, "n_modules": 252, "dims": 252 * 64, "seed": SEED},
           "sketch_check": check,
           "P_19D_1": {"statement": "Median per-target Spearman(D_hat, D) > 0 with CI excluding 0, for S3, S4, S6, S7.",
                       "per_step": {s: res[s]["D1"]["passes"] for s in prim},
                       "passes": all(res[s]["D1"]["passes"] for s in prim)},
           "P_19D_2": {"statement": "S6, S7: median reach ratio < 1 with CI excluding 1.",
                       "per_step": {s: {k: res[s]["D2"].get(k) for k in ("status", "median_ratio", "ci95", "passes", "n_usable_targets")}
                                    for s in ("S6", "S7")}},
           "P_19D_3": {"statement": "S3 spillover AUC > 0.5 with CI excluding 0.5.", **res["S3"]["D3"]},
           "C_measure_psr_nsr": {s: {k: res[s].get(k) for k in ("slope_dhat_on_l_before", "slope_PSR", "slope_NSR")}
                                 for s in ("S6", "S7")},
           "C_measure_effective_dimension": eff,
           "E_family_block_structure": fam,
           "S5_note": "S5 not run in D (run-1 reward configuration not reconstructed in Block 0); S2 run as secondary.",
           "by_step": res}
    p = write_json("kernel_predictions.json", doc)
    write_json("kernel_meta.json", {**prov, "block": "D", "sketch_check": check,
                                    "lazy_check": {s: res[s].get("lazy_check") for s in res},
                                    "update_directions": {s: {"n": res[s]["n_update_sequences"],
                                                              "mean_tokens": res[s]["update_tokens_mean"]} for s in res}})
    write_json("reach.json", {**prov, "block": "D", "P_19D_2": doc["P_19D_2"], "by_step": {s: res[s]["D2"] for s in ("S6", "S7")}})
    write_json("spillover.json", {**prov, "block": "D", "P_19D_3": doc["P_19D_3"]})
    write_json("figdata_kernel.json", {**prov, "block": "D", "targets": fig})
    print(f"-> {p}")
    notify(f"MIRA 19 Block D done: D1 {doc['P_19D_1']['per_step']}; D2 {[(s, v['status'], v['passes']) for s, v in doc['P_19D_2']['per_step'].items()]}; "
           f"D3 {doc['P_19D_3'].get('auc')} passes={doc['P_19D_3'].get('passes')}")


if __name__ == "__main__":
    main()
