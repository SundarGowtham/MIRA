#!/usr/bin/env python
"""
research/phase19_e_location.py -- Phase 19 Block E (GPU, forward only).
docs/phases/PHASE19_PREREG.md "Block E".

Adapter swapping. For step s (before b, after a; b = no adapter for SFT
steps from Base) and depth L in {0,2,...,34,35}:
  bottom-up hybrid: a in blocks 0..L, b above;   reverse: b in 0..L, a above.
Score every probe string (exact teacher-forced l, identical computation
shapes for every configuration). Recovery pooled over targets:
  R_s(L) = 1 - sum (D_hyb - D)^2 / sum D^2,  D = l_after - l_before,
  D_hyb = l_hybrid - l_before.
L50 / L80 = first grid depth where bottom-up recovery >= 0.5 / 0.8.
[E] single-layer swap: a only in block j (j = 0..35), b elsewhere.
[E] residual-stream activation patching (Phase 17 Block E style): before
model, layer-L output replaced at every position by the after model's.
Bootstrap (cluster by target, 10,000, seed 20261116) CI on R at each depth
and on L50/L80.

Predictions [C]: P-19E-1 L50 later for each RL step (S6, S7) than each SFT
step (S3, S4) -- all four orderings; P-19E-2 L50(S6) >= 27 and L50(S7) >= 27.
Primary steps first (S3, S4, S6, S7), then S2, S5 (secondary).
Outputs: results/phase19/location.json, results/phase19/figdata_location.json
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
    STEPS, METHOD, PRIMARY_STEPS, SECONDARY_STEPS, MODELS, N_LAYERS, SEED, N_BOOT, OUT_DIR,
)

GRID = list(range(0, 35, 2)) + [35]
CACHE = OUT_DIR / "location_scores.jsonl"


def cfg_none(m):
    return None if m == "base" else m


class LayerPatch:
    """Replace / record the output of decoder layer L, staged per forward."""

    def __init__(self, mm, L):
        self.layer = mm.layers[L]
        self.mode = None      # "record" | "replace" | None
        self.store = {}
        self.key = None
        self.h = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.mode == "record":
            self.store[self.key] = hs.detach().clone()
        elif self.mode == "replace":
            new = self.store[self.key]
            hs = new
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return None

    def remove(self):
        self.h.remove()


def run_config(mm, preps, per_layer, tag, done, f, patch=None, patch_mode=None, tkey=""):
    if tag in done:
        return done[tag]
    mm.set_per_layer(per_layer)
    out = {}
    for t, prep in preps.items():
        if patch is not None:
            patch.mode = patch_mode

            def on_prefix(t=t):
                patch.key = (t, "prefix")

            def on_cont(b0, n, Lc, t=t):
                patch.key = (t, "cont", b0)
            sc = score_prepared(mm, prep, on_prefix=on_prefix, on_cont=on_cont)
            patch.mode = None
        else:
            sc = score_prepared(mm, prep)
        out[t] = sc
    f.write(json.dumps({"tag": tag, "scores": out}) + "\n")
    f.flush()
    done[tag] = out
    return out


def pooled(scores_by_t, targets):
    return np.concatenate([np.asarray(scores_by_t[t], float) for t in targets])


def rec(lh, la, lb):
    d = la - lb
    dh = lh - lb
    return float(1 - ((dh - d) ** 2).sum() / (d ** 2).sum())


def boot_rec(hyb, after, before, targets, rng):
    idx = rng.integers(0, len(targets), (N_BOOT, len(targets)))
    per_t = [(np.asarray(hyb[t]), np.asarray(after[t]), np.asarray(before[t])) for t in targets]
    num = np.array([(((h - b) - (a - b)) ** 2).sum() for h, a, b in per_t])
    den = np.array([((a - b) ** 2).sum() for h, a, b in per_t])
    return 1 - num[idx].sum(1) / den[idx].sum(1), num, den


def first_reach(curve, thr):
    for L, r in zip(GRID, curve):
        if r >= thr:
            return L
    return None


def main():
    prov = provenance()
    ps = load_probe_strings()
    targets = sorted(ps)
    mm = MultiAdapterModel([m for m in MODELS if m != "base"])
    preps = {t: Prepared(mm.tok, t, [r["string"] for r in ps[t]]) for t in targets}
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            done[r["tag"]] = r["scores"]
    f = CACHE.open("a")
    res, fig = {}, {}
    for s in list(PRIMARY_STEPS) + list(SECONDARY_STEPS):
        b, a = STEPS[s]
        B, A = cfg_none(b), cfg_none(a)
        t0 = time.time()
        before = run_config(mm, preps, [B] * N_LAYERS, f"{s}|before", done, f)
        after = run_config(mm, preps, [A] * N_LAYERS, f"{s}|after", done, f)
        lb, la = pooled(before, targets), pooled(after, targets)
        bu, td, single = {}, {}, {}
        for L in GRID:
            bu[L] = run_config(mm, preps, [A] * (L + 1) + [B] * (N_LAYERS - L - 1), f"{s}|bu|{L}", done, f)
            td[L] = run_config(mm, preps, [B] * (L + 1) + [A] * (N_LAYERS - L - 1), f"{s}|td|{L}", done, f)
        for j in range(N_LAYERS):
            single[j] = run_config(mm, preps, [A if i == j else B for i in range(N_LAYERS)], f"{s}|single|{j}", done, f)
        # [E] residual-stream activation patching, primary steps only
        act = {}
        if s in PRIMARY_STEPS:
            for L in GRID:
                tag = f"{s}|act|{L}"
                if tag in done:
                    act[L] = done[tag]
                    continue
                patch = LayerPatch(mm, L)
                mm.set_per_layer([A] * N_LAYERS)
                patch.mode = "record"
                for t, prep in preps.items():
                    score_prepared(mm, prep, on_prefix=lambda t=t: setattr(patch, "key", (t, "prefix")),
                                   on_cont=lambda b0, n, Lc, t=t: setattr(patch, "key", (t, "cont", b0)))
                patch.mode = None
                act[L] = run_config(mm, preps, [B] * N_LAYERS, tag, done, f, patch=patch, patch_mode="replace")
                patch.remove()
                patch.store = {}
                torch.cuda.empty_cache()
        rng = np.random.default_rng(SEED)
        # bootstrap draws shared across depths so L50 CI is coherent
        idx = rng.integers(0, len(targets), (N_BOOT, len(targets)))

        def curve(cfgs):
            pt = [rec(pooled(cfgs[L], targets), la, lb) for L in GRID]
            num = np.stack([np.array([(((np.asarray(cfgs[L][t]) - np.asarray(before[t])) -
                                        (np.asarray(after[t]) - np.asarray(before[t]))) ** 2).sum() for t in targets])
                            for L in GRID])
            den = np.array([((np.asarray(after[t]) - np.asarray(before[t])) ** 2).sum() for t in targets])
            bs = 1 - num[:, idx].sum(2) / den[idx].sum(1)[None, :]        # [n_L, N_BOOT]
            return pt, bs

        bu_pt, bu_bs = curve(bu)
        td_pt, td_bs = curve(td)
        L50 = first_reach(bu_pt, 0.5)
        L80 = first_reach(bu_pt, 0.8)
        L50_bs = [first_reach(bu_bs[:, k], 0.5) for k in range(N_BOOT)]
        L80_bs = [first_reach(bu_bs[:, k], 0.8) for k in range(N_BOOT)]
        single_rec = {j: rec(pooled(single[j], targets), la, lb) for j in range(N_LAYERS)}
        act_rec = {L: rec(pooled(act[L], targets), la, lb) for L in act}
        d = la - lb
        res[s] = {"before": b, "after": a, "method": METHOD[s], "n_strings": int(len(d)),
                  "delta_rms": float(np.sqrt((d ** 2).mean())), "delta_mean": float(d.mean()),
                  "bottom_up_recovery": dict(zip(GRID, bu_pt)),
                  "bottom_up_ci95": {L: [float(np.percentile(bu_bs[i], 2.5)), float(np.percentile(bu_bs[i], 97.5))]
                                     for i, L in enumerate(GRID)},
                  "top_down_recovery": dict(zip(GRID, td_pt)),
                  "L50": L50, "L80": L80,
                  "L50_ci95": [float(np.percentile([x if x is not None else 99 for x in L50_bs], q)) for q in (2.5, 97.5)],
                  "L80_ci95": [float(np.percentile([x if x is not None else 99 for x in L80_bs], q)) for q in (2.5, 97.5)],
                  "L50_boot_none_share": float(np.mean([x is None for x in L50_bs])),
                  "E_single_layer_recovery": single_rec,
                  "E_activation_patching_recovery": act_rec,
                  "E_activation_patching_L50": first_reach([act_rec[L] for L in GRID], 0.5) if act_rec else None,
                  "_L50_boot": L50_bs}
        fig[s] = {"grid": GRID, "bottom_up": bu_pt, "top_down": td_pt,
                  "single_layer": [single_rec[j] for j in range(N_LAYERS)],
                  "activation_patching": [act_rec.get(L) for L in GRID]}
        print(f"== {s} ({METHOD[s]}, {b}->{a}) L50={L50} CI {res[s]['L50_ci95']} L80={L80} "
              f"rms delta {res[s]['delta_rms']:.2f} [{time.time() - t0:.0f}s]", flush=True)
        print("   bottom-up:", " ".join(f"{L}:{r:.2f}" for L, r in zip(GRID, bu_pt)), flush=True)
        notify(f"MIRA 19 Block E {s}: L50={L50} L80={L80}")
    f.close()
    # predictions
    L = {s: res[s]["L50"] for s in PRIMARY_STEPS}
    big = lambda x: 99 if x is None else x  # noqa: E731
    orderings = {f"{r}>{t}": big(L[r]) > big(L[t]) for r in ("S6", "S7") for t in ("S3", "S4")}
    # bootstrap support for each ordering using shared draws (same seed -> same idx per step)
    boot_ord = {}
    for k in orderings:
        r, t = k.split(">")
        br = np.array([big(x) for x in res[r]["_L50_boot"]])
        bt = np.array([big(x) for x in res[t]["_L50_boot"]])
        boot_ord[k] = float(np.mean(br > bt))
    p1 = all(orderings.values())
    p2 = {s: (L[s] is not None and L[s] >= 27) for s in ("S6", "S7")}
    for s in res:
        res[s].pop("_L50_boot")
    doc = {**prov, "block": "E", "grid": GRID,
           "P_19E_1": {"statement": "L50 is later for each RL step (S6, S7) than for each SFT step (S3, S4); all four orderings required.",
                       "L50": L, "orderings": orderings, "bootstrap_share_ordering_holds": boot_ord, "passes": p1},
           "P_19E_2": {"statement": "L50(S6) >= 27 and L50(S7) >= 27.", "per_step": p2, "passes": all(p2.values()),
                       "does_not_support_rl_written_late": not any(p2.values())},
           "numerics_note": ("bf16 teacher forcing; identical computation shapes for every configuration. Cached vs "
                             "uncached scoring differed by up to ~1 nat per string in the Block 0 smoke (median ~0.1)."),
           "by_step": res}
    p = write_json("location.json", doc)
    write_json("figdata_location.json", {**prov, "block": "E", "curves": fig})
    print(f"P-19E-1 passes={p1} {orderings}; P-19E-2 {p2} -> {p}")
    notify(f"MIRA 19 Block E done: L50 {L}; P-19E-1 {p1}; P-19E-2 {p2}")


if __name__ == "__main__":
    main()
