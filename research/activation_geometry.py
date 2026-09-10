#!/usr/bin/env python
"""
activation_geometry.py — the channel-subspace experiment (item 3).

Question: do the two live reward channels — amount_accuracy (arithmetic) and
thermodynamic_favorable (physics) — occupy DIFFERENT representational
subspaces? The dead channels are ~0 BY CONSTRUCTION and are therefore not a
null (Claude's amendment); the control is a PERMUTATION NULL: shuffle each
channel's within-group z-scores across the group's completions, recompute the
channel-aligned direction, and ask whether the real one separates from the
permuted distribution. No separation -> no effect, whatever the means say.

  pass 1 (base geometry): forward the fixed probe prompts; per-layer CKA +
        principal angles between checkpoint pairs — where did SFT/RL move
        representations at all?
  pass 2 (channel directions): forward ARCHIVED run-2 completions (no new
        sampling); per layer, channel-aligned direction d_c = A_c^T z_c
        (z_c = within-group z-score of channel c's reward, A_c = centered
        pooled completion-token activations over rows where z_c is finite).
        Reports ||d_c|| (with permutation p) and cos(d_amount, d_thermo)
        (with permutation band), per layer, per checkpoint.

Activations are cached per checkpoint to <out>/cache/*.npz; analysis reruns
are free. Prompts are closed-book (current default); z-scores are a property
of the archived completions + validator, independent of the prompt wrapper.

Usage:
  PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python run_debug_and_analysis/activation_geometry.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluate_batched import load_eval_model  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402
from validator import SynthesisValidator  # noqa: E402

SENTINEL = SynthesisValidator.SENTINEL_TAGS
LIVE = ["amount_accuracy", "thermodynamic_favorable"]

CHECKPOINTS = {
    "base": "base",
    "sft": "runs/sft-qlora-sft-v3-2nd-rank16/final",
    "gdpo300": "runs/gdpo-qlora-gdpo-v3/checkpoint-300",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--gens", type=Path,
                   default=Path("runs/gdpo-qlora-beta-ablation-probe/generations.jsonl"))
    p.add_argument("--probe", type=Path, default=Path("data/rl_run3/rl3_probe.jsonl"))
    p.add_argument("--out", type=Path, default=Path("manifold_visualization/act_geo"))
    p.add_argument("--targets", type=int, default=24)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=9216)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pass1", action="store_true")
    p.add_argument("--pass2", action="store_true")
    p.add_argument("--all", action="store_true")
    return p.parse_args()


def closed_book_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


# --------------------------------------------------------------------------
# archive loading (row order preserved — z alignment depends on it)
# --------------------------------------------------------------------------

def load_archive(args, targets: set[str] | None = None):
    """rows: list of (target, completion, {channel: value-or-None}).
    Every dump row is kept, including parse failures (all-None), so row
    indices stay aligned with the generation dump."""
    rows = []
    with args.gens.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = r["target"]
            if targets is not None and t not in targets:
                continue
            bd = r.get("breakdown") or {}
            vals = {}
            for c in LIVE:
                raw = bd.get(c)
                tag = bd.get(f"{c}_gradeability")
                ok = (tag not in SENTINEL) and isinstance(raw, (int, float)) \
                    and not isinstance(raw, bool)
                vals[c] = float(raw) if ok else None
            rows.append((t, r["completion"], vals))
    return rows


def select_targets(rows, n_targets, seed):
    """Targets where BOTH live channels carry within-group variance (>=6
    finite samples each, std > 0)."""
    by_target = defaultdict(list)
    for t, _, vals in rows:
        by_target[t].append(vals)
    ok = []
    for t, samples in by_target.items():
        good = True
        for c in LIVE:
            v = [s[c] for s in samples if s[c] is not None]
            if len(v) < 6 or np.std(v) < 1e-9:
                good = False
                break
        if good:
            ok.append(t)
    rng = np.random.default_rng(seed)
    rng.shuffle(ok)
    return sorted(ok[:n_targets])


def group_z(rows, channel):
    """z[i] = within-target-group z-score of channel for row i (NaN if the
    check was ungradeable for that row, or the group is degenerate)."""
    z = np.full(len(rows), np.nan)
    by_target = defaultdict(list)
    for i, (t, _, _) in enumerate(rows):
        by_target[t].append(i)
    for idxs in by_target.values():
        v = np.array([rows[i][2][channel] if rows[i][2][channel] is not None
                      else np.nan for i in idxs])
        m = np.isfinite(v)
        if m.sum() < 2 or np.nanstd(v) < 1e-12:
            continue
        zg = (v[m] - v[m].mean()) / v[m].std()
        for j, i in enumerate(np.array(idxs)[m]):
            z[i] = zg[j]
    return z


def permute_within_groups(z, rows, rng):
    zp = z.copy()
    by_target = defaultdict(list)
    for i, (t, _, _) in enumerate(rows):
        by_target[t].append(i)
    for idxs in by_target.values():
        zp[idxs] = rng.permutation(z[idxs])
    return zp


# --------------------------------------------------------------------------
# forward passes (pooled activations, cached)
# --------------------------------------------------------------------------

@torch.no_grad()
def pooled_activations(model, tok, texts, span_starts, max_tokens):
    """-> [n_texts, n_layers(+emb), hidden] float32. Mean-pool tokens in
    [span_start:]. One sequence at a time: 9k tokens x 152k-vocab logits is
    the card's memory ceiling."""
    out = []
    for k, (text, s) in enumerate(zip(texts, span_starts)):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_tokens).input_ids.to(model.device)
        hs = model(ids, output_hidden_states=True).hidden_states
        s = min(s, ids.shape[1] - 1)
        out.append(np.stack([h[0, s:, :].mean(0).float().cpu().numpy() for h in hs]))
        if (k + 1) % 25 == 0:
            print(f"    {k + 1}/{len(texts)}", flush=True)
    return np.stack(out)


def get_acts(args, ckpt_name, split, build_fn):
    p = args.out / "cache" / f"{ckpt_name}_{split}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return z["acts"], json.loads(z["meta"].item())
    print(f"[{ckpt_name}/{split}] forwarding ...", flush=True)
    model, tok = load_eval_model(CHECKPOINTS[ckpt_name], args.model)
    model.eval()
    acts, meta = build_fn(model, tok)
    np.savez_compressed(p, acts=acts, meta=json.dumps(meta))
    del model
    torch.cuda.empty_cache()
    return acts, meta


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def center(A):
    return A - A.mean(0, keepdims=True)


def linear_cka(X, Y):
    Xc, Yc = center(X), center(Y)
    hsic = float((Xc.T @ Yc).pow(2).sum()) if torch.is_tensor(Xc) else \
        float(((Xc.T @ Yc) ** 2).sum())
    nx = float(((Xc.T @ Xc) ** 2).sum()) ** 0.5
    ny = float(((Yc.T @ Yc) ** 2).sum()) ** 0.5
    return hsic / max(nx * ny, 1e-12)


def principal_angles(X, Y, k=8):
    Xc, Yc = center(X), center(Y)
    Ux = np.linalg.svd(Xc, full_matrices=False)[0][:, :k]
    Uy = np.linalg.svd(Yc, full_matrices=False)[0][:, :k]
    return np.clip(np.linalg.svd(Ux.T @ Uy, compute_uv=False), 0, 1)


def direction(A, z):
    """d = A_c^T z over rows with finite z (A centered on those rows)."""
    m = np.isfinite(z)
    if m.sum() < 2:
        return None
    d = center(A[m]).T @ z[m]
    return d, float(np.linalg.norm(d))


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    run_all = args.all or not (args.pass1 or args.pass2)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------- pass 1 -------------------------------------------
    if run_all or args.pass1:
        probe_prompts = [json.loads(l)["prompt"] for l in args.probe.open()
                         if l.strip()]
        acts = {}
        for name in CHECKPOINTS:
            def build(model, tok, prompts=probe_prompts):
                texts = [tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True) for p in prompts]
                starts = [len(tok(t).input_ids) - 1 for t in texts]  # last token
                return pooled_activations(model, tok, texts, starts,
                                          args.max_tokens), {"pool": "last_prompt_token"}
            acts[name], _ = get_acts(args, name, "probe", build)
        L = acts["base"].shape[1]
        names = list(CHECKPOINTS)
        cka, angles = {}, {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                cka[f"{a}|{b}"] = [linear_cka(acts[a][:, l, :], acts[b][:, l, :])
                                   for l in range(L)]
                angles[f"{a}|{b}"] = [principal_angles(acts[a][:, l, :],
                                                       acts[b][:, l, :]).tolist()
                                      for l in range(L)]
        out = {"n_probe": len(probe_prompts), "layers": L,
               "cka": cka, "principal_angles_top8": angles}
        (args.out / "pass1_geometry.json").write_text(json.dumps(out, indent=1))
        print(f"pass1 -> {args.out}/pass1_geometry.json")
        for pair, v in cka.items():
            print(f"  CKA {pair}: mean {np.mean(v):.3f}, "
                  f"min {np.min(v):.3f} (layer {int(np.argmin(v))})")

    # ---------------- pass 2 -------------------------------------------
    if run_all or args.pass2:
        all_rows = load_archive(args)
        targets = select_targets(all_rows, args.targets, args.seed)
        rows = [r for r in all_rows if r[0] in targets]
        print(f"pass2: {len(targets)} targets, {len(rows)} completions "
              f"(both channels variant within group)")
        if len(targets) < 4:
            sys.exit("too few qualifying targets")

        zmap = {c: group_z(rows, c) for c in LIVE}

        def build2(model, tok):
            texts, starts, meta = [], [], []
            for t, comp, _ in rows:
                cp = tok.apply_chat_template(
                    [{"role": "user", "content": closed_book_prompt(t)}],
                    tokenize=False, add_generation_prompt=True)
                starts.append(len(tok(cp).input_ids))
                texts.append(cp + comp)
                meta.append(t)
            return pooled_activations(model, tok, texts, starts,
                                      args.max_tokens), {"rows": meta}

        results = {}
        for name in CHECKPOINTS:
            acts, _ = get_acts(args, name, "completions", build2)
            L = acts.shape[1]
            per_chan = {}
            dirs = {}
            for c in LIVE:
                z = zmap[c]
                norms, ps, dlist = [], [], []
                for l in range(L):
                    out = direction(acts[:, l, :], z)
                    if out is None:
                        norms.append(None); ps.append(None); dlist.append(None)
                        continue
                    d, nrm = out
                    dlist.append(d)
                    nulls = []
                    for _ in range(args.perms):
                        on = direction(acts[:, l, :],
                                       permute_within_groups(z, rows, rng))
                        nulls.append(on[1] if on else 0.0)
                    nulls = np.array(nulls)
                    norms.append(nrm)
                    ps.append(float((nulls >= nrm).mean()))
                per_chan[c] = {"norm": norms, "perm_p": ps}
                dirs[c] = dlist
            cos_real, cos_band = [], []
            for l in range(L):
                da, dt = dirs[LIVE[0]][l], dirs[LIVE[1]][l]
                if da is None or dt is None:
                    cos_real.append(None); cos_band.append(None)
                    continue
                cos_real.append(float(da @ dt / (np.linalg.norm(da)
                                                 * np.linalg.norm(dt) + 1e-12)))
                cn = []
                for _ in range(args.perms):
                    o1 = direction(acts[:, l, :],
                                   permute_within_groups(zmap[LIVE[0]], rows, rng))
                    o2 = direction(acts[:, l, :],
                                   permute_within_groups(zmap[LIVE[1]], rows, rng))
                    if o1 and o2:
                        cn.append(float(o1[0] @ o2[0] /
                                        (np.linalg.norm(o1[0]) * np.linalg.norm(o2[0])
                                         + 1e-12)))
                cos_band.append([float(np.percentile(cn, 5)),
                                 float(np.percentile(cn, 95))] if cn else None)
            results[name] = {"per_channel": per_chan,
                             "cos_amount_vs_thermo": cos_real,
                             "cos_null_band_5_95": cos_band}
            print(f"  [{name}] done", flush=True)
        out = {"targets": targets, "n_completions": len(rows),
               "perms": args.perms, "live_channels": LIVE,
               "note": "perm_p = fraction of within-group permutations with "
                       "direction norm >= real (small = real channel-aligned "
                       "direction exists). cos band = 5-95% permutation band "
                       "for cos(d_amount, d_thermo); real cos OUTSIDE the band "
                       "= channels occupy distinguishable subspaces.",
               "results": results}
        (args.out / "pass2_channels.json").write_text(json.dumps(out, indent=1))
        print(f"pass2 -> {args.out}/pass2_channels.json")
        for name, res in results.items():
            for c in LIVE:
                ps = [p for p in res["per_channel"][c]["perm_p"] if p is not None]
                print(f"  [{name}] {c}: median perm_p {np.median(ps):.3f}, "
                      f"layers with p<0.05: {sum(p < 0.05 for p in ps)}/{len(ps)}")
            cr = [c for c in res["cos_amount_vs_thermo"] if c is not None]
            print(f"  [{name}] cos(d_amount, d_thermo): mean {np.mean(cr):.3f}")


if __name__ == "__main__":
    main()
