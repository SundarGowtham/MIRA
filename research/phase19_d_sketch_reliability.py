#!/usr/bin/env python
"""
research/phase19_d_sketch_reliability.py -- [E] (decided after the
pre-registered per-module sketch check failed: Pearson 0.67-0.77 < 0.95).

The registered check compares a single module's 64-dim sketch with its exact
inner products. Block D uses the FULL kernel (sum over 252 modules), whose
sketch error averages across modules. Exact full-kernel inner products are
infeasible (8B-parameter gradients), so this measures the full kernel's
reliability instead: probe strings at Base sketched with a second,
independent seed (20261117); Pearson between the two seeds' kernel entries
over (a) all within-target pairs and (b) 2,000 random pairs. If each kernel
equals truth + independent noise, Pearson(A, B) ~ r_true^2, so
r_true ~ sqrt(Pearson(A, B)) is reported alongside (an approximation).
Also: D-1 recomputed for S3 with the second seed's probe sketches and the
first seed's update direction recomputed... is not possible without
re-sketching updates; instead the seed-B Spearman(D_hat_A, D_hat_B) over
probe strings is reported for S3 using seed-B update sketches of E3's 505
training sequences (cheap: 1024 tokens each).
Output: results/phase19/sketch_reliability.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from research.phase19_common import (  # noqa: E402
    MultiAdapterModel, load_probe_strings, provenance, write_json, notify, context_text, answer_token_mask,
    MODELS, SEED,
)
from research.phase19_grad import Sketcher, prepare_for_backward, seq_logprob_sketch  # noqa: E402

CACHE = REPO_ROOT / "runs" / "phase19_cache" / "sketches"
SEED_B = SEED + 1


def main():
    prov = provenance()
    ps = load_probe_strings()
    mm = MultiAdapterModel([m for m in MODELS if m != "base"])
    prepare_for_backward(mm, checkpointing=True)
    mm.set_all(None)
    skB = Sketcher(mm, SEED_B)
    SB, keys = [], []
    for t in sorted(ps):
        ctx = context_text(mm.tok, t)
        seqs = [answer_token_mask(mm.tok, ctx, ctx + r["string"]) for r in ps[t]]
        for b0 in range(0, len(seqs), 8):
            ch = seqs[b0:b0 + 8]
            S, _, _ = seq_logprob_sketch(mm, skB, [x for x, _ in ch], [m for _, m in ch])
            SB.append(S.cpu().numpy())
        keys.extend(f"{t}||{r['recipe_key']}" for r in ps[t])
    SB = np.concatenate(SB)
    zA = np.load(CACHE / "probe_base.npz", allow_pickle=True)
    SA = zA["S"]
    assert list(zA["keys"]) == keys
    KA, KB = SA @ SA.T, SB @ SB.T
    tg = np.array([k.split("||")[0] for k in keys])
    iu = np.triu_indices(len(keys), 1)
    within = tg[iu[0]] == tg[iu[1]]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(iu[0]), 2000, replace=False)
    r_within = float(np.corrcoef(KA[iu][within], KB[iu][within])[0, 1])
    r_rand = float(np.corrcoef(KA[iu][pick], KB[iu][pick])[0, 1])
    nA, nB = np.sqrt(np.diag(KA)), np.sqrt(np.diag(KB))
    r_norm = float(np.corrcoef(nA, nB)[0, 1])
    CA, CB = KA / np.outer(nA, nA), KB / np.outer(nB, nB)
    r_cos_within = float(np.corrcoef(CA[iu][within], CB[iu][within])[0, 1])
    # S3 prediction under seed B
    rows = [json.loads(l) for l in open(REPO_ROOT / "data/e3/e3_train.jsonl") if l.strip()]
    UB = []
    for b0 in range(0, len(rows), 4):
        ids = []
        for r in rows[b0:b0 + 4]:
            txt = mm.tok.apply_chat_template([{"role": "user", "content": r["prompt"]},
                                              {"role": "assistant", "content": r["completion"]}], tokenize=False)
            ids.append(mm.tok(txt, add_special_tokens=False).input_ids[:1024])
        S, _, _ = seq_logprob_sketch(mm, skB, ids, [[True] * len(x) for x in ids])
        UB.append(S.cpu().numpy())
    gB = np.concatenate(UB).mean(0)
    zU = np.load(CACHE / "update_S3.npz", allow_pickle=True)
    gA = (zU["w"][:, None] * zU["S"]).sum(0)
    dA, dB = SA @ gA, SB @ gB
    per_t = [float(spearmanr(dA[tg == t], dB[tg == t]).correlation) for t in sorted(set(tg)) if (tg == t).sum() >= 3]
    doc = {**prov, "block": "D [E] sketch reliability", "seed_A": SEED, "seed_B": SEED_B,
           "pearson_kernel_seedA_vs_seedB_within_target_pairs": r_within,
           "pearson_kernel_seedA_vs_seedB_random_pairs": r_rand,
           "approx_r_true_within": float(np.sqrt(max(r_within, 0))),
           "approx_r_true_random": float(np.sqrt(max(r_rand, 0))),
           "pearson_gradient_norm_seedA_vs_seedB": r_norm,
           "pearson_cosine_kernel_within_target": r_cos_within,
           "S3_dhat_seedA_vs_seedB_spearman_pooled": float(spearmanr(dA, dB).correlation),
           "S3_dhat_seedA_vs_seedB_spearman_per_target_median": float(np.median(per_t))}
    p = write_json("sketch_reliability.json", doc)
    print(json.dumps(doc, indent=1, default=str))
    print(f"-> {p}")
    notify(f"MIRA 19 D [E] sketch reliability: kernel seedA~seedB r={r_within:.3f} (within), {r_rand:.3f} (random); "
           f"S3 dhat rho={doc['S3_dhat_seedA_vs_seedB_spearman_pooled']:.3f}")


if __name__ == "__main__":
    main()
