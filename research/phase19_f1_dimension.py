#!/usr/bin/env python
"""
research/phase19_f1_dimension.py -- Phase 19 Block F1 (GPU, forward only).
docs/phases/PHASE19_PREREG.md "Block F1".

Per primary step (S3, S4, S6, S7):
1. L* = L80 from Block E (results/phase19/location.json), else block 35.
2. Probe targets split in half, seeded (20261116), stratified 10 ASTRAL + 10
   held out per half. On the first half, delta = h_after - h_before at the
   output of block L*, at ALL positions of the teacher-forced sequence (prompt
   and answer string; the same sequence for both models). Principal
   components = top right singular vectors of the stacked delta vectors
   (uncentered: the patch adds P_k delta, so the directions that carry delta's
   energy are the relevant ones).
3. On the other half, run the before model with h_before + P_k delta patched at
   ALL positions; R(k) on the answer-string log-probability with Block E's
   recovery formula (pooled over the half's strings), k in {1,2,4,8,16,32,64}.
4. Null: 20 random k-dimensional subspaces (QR of Gaussian), seeded.
[E] one global steering vector: mean delta over the first half's positions,
added at every position.
P-19F-1: k80 (smallest k with R >= 0.8) smaller for S6 and S7 than for S3 and S4.
Output: results/phase19/dimension.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from research.phase19_common import (  # noqa: E402
    MultiAdapterModel, Prepared, score_prepared, load_probe_strings, provenance, write_json, notify,
    STEPS, METHOD, MODELS, N_LAYERS, SEED, OUT_DIR,
)

KS = [1, 2, 4, 8, 16, 32, 64]
N_NULL = 20
PROBE = json.loads((REPO_ROOT / "results/phase17/probe_targets.json").read_text())


class Hook:
    def __init__(self, layer):
        self.mode, self.key = None, None
        self.rec, self.add = {}, {}
        self.h = layer.register_forward_hook(self._f)

    def _f(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.mode == "record":
            self.rec[self.key] = hs.detach().float()
            return None
        if self.mode == "add":
            a = self.add.get(self.key)
            if a is None:
                return None
            hs = (hs.float() + a).to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return None


def run(mm, preps, targets, hook, mode):
    hook.mode = mode
    out = {}
    for t in targets:
        out[t] = score_prepared(mm, preps[t],
                                on_prefix=lambda t=t: setattr(hook, "key", (t, "prefix")),
                                on_cont=lambda b0, n, Lc, t=t: setattr(hook, "key", (t, "cont", b0)))
    hook.mode = None
    return out


def rec(lh, la, lb):
    d, dh = la - lb, lh - lb
    return float(1 - ((dh - d) ** 2).sum() / (d ** 2).sum())


def main():
    prov = provenance()
    loc = json.loads((OUT_DIR / "location.json").read_text())
    ps = load_probe_strings()
    rng = np.random.default_rng(SEED)
    a_ = list(rng.permutation(sorted(PROBE["probe_astral_20"])))
    h_ = list(rng.permutation(sorted(PROBE["probe_heldout_20"])))
    train = sorted(a_[:10] + h_[:10])
    test = sorted(a_[10:] + h_[10:])
    mm = MultiAdapterModel([m for m in MODELS if m != "base"])
    preps = {t: Prepared(mm.tok, t, [r["string"] for r in ps[t]]) for t in ps}
    res = {}
    for s in ("S3", "S4", "S6", "S7"):
        t0 = time.time()
        b, a = STEPS[s]
        B, A = (None if b == "base" else b), (None if a == "base" else a)
        L80 = loc["by_step"][s]["L80"]
        Ls = L80 if L80 is not None else N_LAYERS - 1
        hook = Hook(mm.layers[Ls])
        mm.set_all(B)
        lb = run(mm, preps, train + test, hook, "record")
        hb = hook.rec
        hook.rec = {}
        mm.set_all(A)
        la = run(mm, preps, train + test, hook, "record")
        ha = hook.rec
        hook.rec = {}
        # deltas with padding zeroed
        delta = {}
        for key in hb:
            d = ha[key] - hb[key]
            if key[1] == "cont":
                t, _, b0 = key
                lens = [len(c) for c in preps[t].conts[b0:b0 + d.shape[0]]]
                for i, L in enumerate(lens):
                    d[i, L:] = 0
            delta[key] = d
        del ha, hb
        X = torch.cat([delta[k].reshape(-1, delta[k].shape[-1]) for k in delta if k[0] in train])
        X = X[X.abs().sum(1) > 0]
        G = X.T @ X
        ev, V = torch.linalg.eigh(G)
        V = V[:, torch.argsort(ev, descending=True)][:, :max(KS)]
        mean_delta = X.mean(0)
        frac_energy = {k: float(ev.sort(descending=True).values[:k].sum() / ev.sum()) for k in KS}
        del X, G
        lb_t = np.concatenate([lb[t] for t in test])
        la_t = np.concatenate([la[t] for t in test])
        mm.set_all(B)

        def patched(proj=None, vec=None):
            hook.add = {}
            for key, d in delta.items():
                if key[0] not in test:
                    continue
                if proj is not None:
                    add = (d @ proj) @ proj.T
                else:
                    add = torch.zeros_like(d) + vec
                    if key[1] == "cont":
                        t, _, b0 = key
                        for i, L in enumerate([len(c) for c in preps[t].conts[b0:b0 + d.shape[0]]]):
                            add[i, L:] = 0
                hook.add[key] = add
            out = run(mm, preps, test, hook, "add")
            hook.add = {}
            return np.concatenate([out[t] for t in test])

        R, Rnull = {}, {}
        g = torch.Generator(device="cpu").manual_seed(SEED)
        for k in KS:
            R[k] = rec(patched(proj=V[:, :k]), la_t, lb_t)
            nulls = []
            for _ in range(N_NULL):
                Q, _r = torch.linalg.qr(torch.randn(V.shape[0], k, generator=g))
                nulls.append(rec(patched(proj=Q.to(V.device)), la_t, lb_t))
            Rnull[k] = {"mean": float(np.mean(nulls)), "max": float(np.max(nulls)), "values": nulls}
            print(f"  {s} k={k}: R={R[k]:.3f} null mean {Rnull[k]['mean']:.3f} max {Rnull[k]['max']:.3f}", flush=True)
        R_full = rec(patched(proj=torch.eye(V.shape[0], device=V.device)), la_t, lb_t)
        R_global = rec(patched(vec=mean_delta), la_t, lb_t)
        k80 = next((k for k in KS if R[k] >= 0.8), None)
        res[s] = {"method": METHOD[s], "L_star": Ls, "L80_from_E": L80, "R": R, "R_null_random_subspace": Rnull,
                  "R_full_residual_patch_at_L_star": R_full, "k80": k80, "E_global_steering_vector_R": R_global,
                  "pca_energy_fraction_train": frac_energy, "n_test_strings": int(len(la_t))}
        print(f"== {s} L*={Ls} k80={k80} full-patch R={R_full:.3f} global-vector R={R_global:.3f} [{time.time() - t0:.0f}s]",
              flush=True)
        hook.h.remove()
        del delta
        torch.cuda.empty_cache()
        notify(f"MIRA 19 F1 {s}: L*={Ls} k80={k80} fullR={R_full:.2f}")
    big = lambda x: 10 ** 6 if x is None else x  # noqa: E731
    order = {f"{r}<{t}": big(res[r]["k80"]) < big(res[t]["k80"]) for r in ("S6", "S7") for t in ("S3", "S4")}
    doc = {**prov, "block": "F1", "split": {"first_half_pca": train, "second_half_test": test},
           "P_19F_1": {"statement": "k80 is smaller for S6 and S7 than for S3 and S4.",
                       "k80": {s: res[s]["k80"] for s in res}, "orderings": order, "passes": all(order.values())},
           "by_step": res}
    p = write_json("dimension.json", doc)
    print(f"P-19F-1 {doc['P_19F_1']} -> {p}")
    notify(f"MIRA 19 F1 done: k80 {doc['P_19F_1']['k80']} passes={doc['P_19F_1']['passes']}")


if __name__ == "__main__":
    main()
