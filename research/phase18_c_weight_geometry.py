#!/usr/bin/env python
"""
research/phase18_c_weight_geometry.py — Phase 18 Block 18C: low-rank
structure of the updates, and the *Path Not Taken* check (Phase 17 H1, never
run). PHASE18_PREREG.md "18C".

Per targeted module (7 per layer x 36 layers = 252):
  * Update: dW = (alpha/r) B A for fine-tuning steps (S1-S4, from base);
    (alpha/r)(B1 A1 - B0 A0) for practice steps (S5-S7), which continue their
    init adapter (checked here: cosine of A1 vs A0 per module, reported).
  * W_base from the bf16 Qwen3-8B safetensors in the local HF cache, loaded
    per tensor (never the 4-bit weights). Top-256 left/right singular vectors
    by exact SVD (torch.linalg.svd, float32, on GPU if available).
  * Metrics: effective rank exp H(s^2/sum s^2); stable rank ||dW||_F^2/s_1^2;
    E_k = ||U_k^T dW||_F^2/||dW||_F^2 (left) and ||dW V_k||_F^2/||dW||_F^2
    (right), k in {16, 64, 256}, vs the random-subspace expectation k/d_out
    (k/d_in). dW is never formed: with dW = P Q^T (P = [B1, -B0] scaled,
    Q = [A1; A0]^T, rank <= 2r = 32), thin QR of P and Q gives the exact SVD
    from a 32x32 core.
  * Trajectory: cumulative dW_t at checkpoints 100/200/300 for S5 (GDPO-v3),
    S6 (Phase 12) and S7 (E1). Fine-tuning runs have one intermediate
    checkpoint each (corpus_sft 117, rs_sft 27, e3/e3c 48); these are reported
    as an extra point, flagged where identical to final.

Predictions [C]:
  P-18C-1: pooled median of E_64 (left) across modules lower for S5-S7 than
           S2-S4; module-bootstrap CI on the difference excludes 0.
           (Pooling decided here: median over all (step, module) values in a
           group; bootstrap resamples the 252 modules, 10,000 times.)
  P-18C-2: for S6 and S7, effective rank of cumulative dW_t non-decreasing
           over 100 -> 200 -> 300 in at least 75% of modules (ties count as
           non-decreasing; equality tested at relative tolerance 1e-6).

Caveats carried into the write-up: Euclidean weight-space angles do not
respect the Fisher metric (why the project deprecated this diagnostic in
September); reported only for comparability with The Path Not Taken, which
used full fine-tuning -- every rank here is capped at 2r = 32 (r for S1-S4).

Usage (tmux; GPU used for base SVDs if free):
  scripts/run_phase18_block.sh 18C research/phase18_c_weight_geometry.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.phase18_common import (  # noqa: E402
    ADAPTERS, STEPS, OWN_SAMPLE_FT_STEPS, PRACTICE_STEPS, SEED, N_BOOT, provenance, write_json, notify,
)

SNAP = Path("/models/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
MODULES = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"), ("self_attn", "o_proj"),
           ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
N_LAYERS = 36
KS = (16, 64, 256)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

FT_INTERMEDIATE = {"corpus_sft": "runs/sft-qlora-sft-v3-2nd-rank16/checkpoint-117",
                   "rs_sft": "runs/sft-qlora-rs-sft-from-base/checkpoint-27",
                   "e3": "runs/sft-qlora-e3-round1/checkpoint-48",
                   "e3c": "runs/sft-qlora-e3c-round1/checkpoint-48"}
TRAJ = {"S5": ("corpus_sft", ["runs/gdpo-qlora-gdpo-v3/checkpoint-100", "runs/gdpo-qlora-gdpo-v3/checkpoint-200",
                              "runs/gdpo-qlora-gdpo-v3/checkpoint-300"]),
        "S6": ("rs_sft", [ADAPTERS["phase12_ckpt100"], ADAPTERS["phase12_ckpt200"], ADAPTERS["phase12_ckpt300"]]),
        "S7": ("rs_sft", [ADAPTERS["e1_ckpt100"], ADAPTERS["e1_ckpt200"], ADAPTERS["e1_ckpt300"]])}


class Adapter:
    def __init__(self, rel):
        p = REPO_ROOT / rel
        cfg = json.loads((p / "adapter_config.json").read_text())
        self.scale = cfg["lora_alpha"] / cfg["r"]
        self.f = safe_open(str(p / "adapter_model.safetensors"), "pt")
        self.rel = rel

    def AB(self, layer, grp, mod):
        k = f"base_model.model.model.layers.{layer}.{grp}.{mod}"
        A = self.f.get_tensor(k + ".lora_A.weight").double()
        B = self.f.get_tensor(k + ".lora_B.weight").double()
        return A, B


def factor_svd(P, Q):
    """dW = P @ Q.T (P: out x m, Q: in x m). Returns U (out x m'), s, V (in x m')."""
    Qp, Rp = torch.linalg.qr(P)
    Qq, Rq = torch.linalg.qr(Q)
    core = Rp @ Rq.T
    u, s, vt = torch.linalg.svd(core)
    return Qp @ u, s, Qq @ vt.T


def metrics(U, s, V, Ub, Vb, d_out, d_in):
    s2 = (s ** 2)
    tot = float(s2.sum())
    if tot <= 0:
        return None
    p = (s2 / tot).clamp_min(1e-300)
    H = float(-(p * p.log()).sum())
    out = {"fro2": tot, "sigma1": float(s[0]), "eff_rank": math.exp(H), "stable_rank": tot / float(s2[0])}
    # U diag(s) is the column-space representation of dW; ||Uk^T dW||^2 = ||Uk^T U diag(s)||^2
    US = U * s
    VS = V * s
    for k in KS:
        out[f"E{k}_left"] = float(((Ub[:, :k].T @ US) ** 2).sum()) / tot
        out[f"E{k}_right"] = float(((Vb[:, :k].T @ VS) ** 2).sum()) / tot
        out[f"E{k}_left_random"] = k / d_out
        out[f"E{k}_right_random"] = k / d_in
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def main():
    prov = provenance()
    idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def base_tensor(name):
        shard = idx[name]
        if shard not in handles:
            handles[shard] = safe_open(str(SNAP / shard), "pt")
        return handles[shard].get_tensor(name)

    assert base_tensor("model.layers.0.self_attn.q_proj.weight").dtype == torch.bfloat16
    # also load any trajectory init outside the four (Phase 20 adds "rs_sft_old"); Phase 18 unchanged
    ad = {m: Adapter(ADAPTERS[m]) for m in ("corpus_sft", "rs_sft", "e3", "e3c", *(i for i, _ in TRAJ.values()))}
    traj_ad = {s: [Adapter(r) for r in rels] for s, (_, rels) in TRAJ.items()}
    inter_ad = {m: Adapter(r) for m, r in FT_INTERMEDIATE.items()}

    rows = []  # per module, per update label
    init_check = []
    for layer in range(N_LAYERS):
        for grp, mod in MODULES:
            W = base_tensor(f"model.layers.{layer}.{grp}.{mod}.weight").to(DEV, torch.float32)
            Ub, sb, Vbt = torch.linalg.svd(W, full_matrices=False)
            Ub = Ub[:, :max(KS)].double().cpu()
            Vb = Vbt[:max(KS), :].T.double().cpu()
            d_out, d_in = W.shape
            del W, Vbt
            mod_id = f"L{layer}.{mod}"

            def rec(label, P, Q, extra=None):
                U, s, V = factor_svd(P, Q)
                m = metrics(U, s, V, Ub, Vb, d_out, d_in)
                if m is not None:
                    rows.append({"module": mod_id, "layer": layer, "proj": mod, "update": label, **m, **(extra or {})})

            # fine-tuning steps
            for step, model in (("S1", "corpus_sft"), ("S2", "rs_sft"), ("S3", "e3"), ("S4", "e3c")):
                A, B = ad[model].AB(layer, grp, mod)
                rec(step, ad[model].scale * B, A.T)
                Ai, Bi = inter_ad[model].AB(layer, grp, mod)
                same = bool(torch.equal(Ai, A) and torch.equal(Bi, B))
                rec(f"{step}@intermediate", inter_ad[model].scale * Bi, Ai.T, {"identical_to_final": same})
            # practice steps: cumulative dW_t = s(B_t A_t - B_0 A_0)
            for step, (init, _) in TRAJ.items():
                A0, B0 = ad[init].AB(layer, grp, mod)
                for ci, a in enumerate(traj_ad[step]):
                    A1, B1 = a.AB(layer, grp, mod)
                    if ci == 0:
                        cosA = float((A1 * A0).sum() / (A1.norm() * A0.norm()))
                        init_check.append({"step": step, "module": mod_id, "cos_A_ckpt100_vs_init": cosA})
                    P = torch.cat([a.scale * B1, -ad[init].scale * B0], dim=1)
                    Q = torch.cat([A1, A0], dim=0).T
                    label = step if ci == 2 else f"{step}@{(ci + 1) * 100}"
                    rec(label, P, Q)
            if DEV == "cuda":
                torch.cuda.empty_cache()
        print(f"layer {layer} done ({len(rows)} rows)", flush=True)

    rng = random.Random(SEED)
    by = {}
    for r in rows:
        by.setdefault(r["update"], {})[r["module"]] = r
    modules = sorted(by["S1"].keys())

    def group_median(key, steps, mods):
        return median([by[s][m][key] for s in steps for m in mods if m in by[s]])

    p1 = {}
    for side in ("left", "right"):
        key = f"E64_{side}"
        ft = group_median(key, OWN_SAMPLE_FT_STEPS, modules)
        pr = group_median(key, PRACTICE_STEPS, modules)
        diffs = []
        for _ in range(N_BOOT):
            samp = [modules[rng.randrange(len(modules))] for _ in modules]
            diffs.append(group_median(key, PRACTICE_STEPS, samp) - group_median(key, OWN_SAMPLE_FT_STEPS, samp))
        diffs.sort()
        ci = [diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]]
        p1[side] = {"median_E64_S2_S4": ft, "median_E64_S5_S7": pr, "diff_practice_minus_ft": pr - ft, "ci95": ci,
                    "random_expectation_median": median([by["S2"][m][f"E64_{side}_random"] for m in modules]),
                    "by_step_median": {s: group_median(key, (s,), modules) for s in STEPS}}
    passes1 = p1["left"]["diff_practice_minus_ft"] < 0 and p1["left"]["ci95"][1] < 0

    p2 = {}
    for step in ("S6", "S7", "S5"):
        nondec = 0
        n = 0
        for m in modules:
            e = [by[f"{step}@100"][m]["eff_rank"], by[f"{step}@200"][m]["eff_rank"], by[step][m]["eff_rank"]]
            n += 1
            tol = 1e-6 * max(e)
            if e[1] >= e[0] - tol and e[2] >= e[1] - tol:
                nondec += 1
        p2[step] = {"share_modules_nondecreasing": nondec / n, "n_modules": n,
                    "median_eff_rank_100_200_300": [group_median("eff_rank", (f"{step}@100",), modules),
                                                    group_median("eff_rank", (f"{step}@200",), modules),
                                                    group_median("eff_rank", (step,), modules)]}
    passes2 = all(p2[s]["share_modules_nondecreasing"] >= 0.75 for s in ("S6", "S7"))

    # [E] per-layer profile and rank comparison
    profile = {}
    for s in list(STEPS):
        profile[s] = {L: {"E64_left": median([by[s][m]["E64_left"] for m in modules if by[s][m]["layer"] == L]),
                          "eff_rank": median([by[s][m]["eff_rank"] for m in modules if by[s][m]["layer"] == L])}
                      for L in range(N_LAYERS)}
    rank_cmp = {s: {"median_eff_rank": group_median("eff_rank", (s,), modules),
                    "median_stable_rank": group_median("stable_rank", (s,), modules),
                    "median_E16_left": group_median("E16_left", (s,), modules),
                    "median_E256_left": group_median("E256_left", (s,), modules)} for s in STEPS}
    inter = {s: {"identical_to_final_share": float(np.mean([by[f"{s}@intermediate"][m]["identical_to_final"]
                                                             for m in modules])),
                 "median_eff_rank": group_median("eff_rank", (f"{s}@intermediate",), modules)}
             for s in ("S1", "S2", "S3", "S4")}
    cos = {s: median([c["cos_A_ckpt100_vs_init"] for c in init_check if c["step"] == s]) for s in TRAJ}

    doc = {**prov, "block": "18C", "device": DEV,
           "caveats": ["Euclidean weight-space quantities ignore the Fisher metric; reported for comparability with "
                       "The Path Not Taken only.",
                       "QLoRA r=16: fine-tuning updates have rank <= 16, practice updates (B1A1-B0A0) rank <= 32; "
                       "the paper used full fine-tuning."],
           "practice_continues_init_check": {"median_cos_A_ckpt100_vs_init_A": cos,
                                             "note": "near 1 = the practice run's A matrices are its init adapter's, "
                                                     "continued (not re-initialised)"},
           "P_18C_1": {"statement": "Pooled median E_64 (left) lower for S5-S7 than S2-S4; module-bootstrap CI on "
                                    "the difference excludes 0.", "by_side": p1, "passes": passes1},
           "P_18C_2": {"statement": "For S6 and S7, effective rank of cumulative dW_t non-decreasing over 100->200->300 "
                                    "in >= 75% of modules.", "by_step": p2, "passes": passes2,
                       "S5_note": "S5 reported alongside, not part of the prediction."},
           "E_rank_comparison_by_step": rank_cmp,
           "E_per_layer_profile": profile,
           "E_ft_intermediate_checkpoints": inter,
           "per_module": rows}
    p = write_json("weight_geometry.json", doc)
    print(json.dumps({"P_18C_1": doc["P_18C_1"], "P_18C_2": doc["P_18C_2"], "rank": rank_cmp,
                      "init": cos, "inter": inter}, indent=1, default=str))
    print(f"-> {p}")
    notify(f"MIRA 18C done: P-18C-1 {passes1}, P-18C-2 {passes2}")


if __name__ == "__main__":
    main()
