#!/usr/bin/env python
"""
research/phase19_block0_gpu_smoke.py -- Phase 19 Block 0, item 4 (time
estimate) plus two instrument checks run before any registered quantity:
  (a) KV-cached string scoring == uncached full-sequence scoring;
  (b) the hook-based Kronecker sketch of one module == R_out^T (autograd
      param .grad) R_in for the same module (catches double counting under
      gradient checkpointing).
Times: one target's probe strings under one adapter configuration; probe-
string gradient sketches; one SFT as-trained sequence (1024 tokens); the
longest of the 400 RL update completions (peak memory reported; no
truncation -- if it does not fit, that is reported, per the user's rule).
Output: results/phase19/time_estimate.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from research.phase19_common import (  # noqa: E402
    MultiAdapterModel, Prepared, score_prepared, score_full_reference, load_probe_strings, provenance,
    write_json, notify, OUT_DIR, N_LAYERS, context_text, answer_token_mask, SEED,
)
from research.phase19_grad import Sketcher, prepare_for_backward, seq_logprob_sketch  # noqa: E402


def sync():
    torch.cuda.synchronize()


def main():
    prov = provenance()
    ps = load_probe_strings()
    n_strings = sum(len(v) for v in ps.values())
    t0 = time.time()
    mm = MultiAdapterModel(["rs_sft", "e3"])
    load_s = time.time() - t0
    tgt = sorted(ps, key=lambda t: -len(ps[t]))[0]
    strings = [r["string"] for r in ps[tgt]]
    prep = Prepared(mm.tok, tgt, strings)
    mm.set_all("rs_sft")
    score_prepared(mm, prep)  # warmup
    sync(); t0 = time.time()
    cached = score_prepared(mm, prep)
    sync(); t_cfg_target = time.time() - t0
    full = score_full_reference(mm, tgt, strings[:8])
    kv_maxdiff = max(abs(a - b) for a, b in zip(cached[:8], full))
    mm.set_per_layer(["e3"] * 18 + [None] * 18)
    hyb = score_prepared(mm, prep)
    print(f"target {tgt}: {len(strings)} strings, prefix {prep.P} tokens, one config {t_cfg_target:.2f}s, "
          f"KV max |diff| {kv_maxdiff:.4f} nats", flush=True)

    # --- backward instrument check + timing
    mm.set_all("rs_sft")
    prepare_for_backward(mm, checkpointing=True)
    check_idx = [i for i, (li, qn, m) in enumerate(mm.lora) if li == 20 and qn.endswith("v_proj")][0]
    W = mm.lora[check_idx][2].get_base_layer().weight
    W.requires_grad_(True)
    sk = Sketcher(mm, SEED)
    ctx = context_text(mm.tok, tgt)
    ids, m = answer_token_mask(mm.tok, ctx, ctx + strings[0])
    S, vals, _ = seq_logprob_sketch(mm, sk, [ids], [m])
    gW = W.grad.float()
    ref = (sk.R_out[check_idx].T @ gW @ sk.R_in[check_idx]).flatten()
    got = S[0].reshape(len(sk.mods), 64)[check_idx]
    rel = float((ref - got).norm() / ref.norm())
    W.grad = None
    W.requires_grad_(False)
    print(f"sketch vs autograd param grad (layer 20 v_proj): rel err {rel:.4f}; f={vals[0]:.3f} vs scored {cached[0]:.3f}",
          flush=True)

    # probe-string batch timing
    seqs = [answer_token_mask(mm.tok, ctx, ctx + s) for s in strings[:8]]
    sync(); t0 = time.time()
    seq_logprob_sketch(mm, sk, [x for x, _ in seqs], [y for _, y in seqs])
    sync(); t_probe8 = time.time() - t0
    torch.cuda.reset_peak_memory_stats()

    # SFT as-trained (1024 tokens, loss on every token)
    row = json.loads(open(REPO_ROOT / "data/e3/e3_train.jsonl").readline())
    txt = mm.tok.apply_chat_template([{"role": "user", "content": row["prompt"]},
                                      {"role": "assistant", "content": row["completion"]}], tokenize=False)
    sid = mm.tok(txt, add_special_tokens=False).input_ids[:1024]
    sync(); t0 = time.time()
    seq_logprob_sketch(mm, sk, [sid], [[True] * len(sid)])
    sync(); t_sft = time.time() - t0

    # longest RL completion
    rl = [json.loads(l) for l in open(OUT_DIR / "rl_update_groups_S6.jsonl")]
    lens = []
    for r in rl:
        p = mm.tok.apply_chat_template([{"role": "user", "content": __import__("research.phase19_common", fromlist=["closed_prompt"]).closed_prompt(r["target"])}],
                                       tokenize=False, add_generation_prompt=True)
        lens.append((len(mm.tok(p + r["completion"], add_special_tokens=False).input_ids), r, p))
    lens.sort(key=lambda z: -z[0])
    mean_len = sum(z[0] for z in lens) / len(lens)
    Lmax, r, p = lens[0]
    pid = mm.tok(p, add_special_tokens=False).input_ids
    fid = mm.tok(p + r["completion"], add_special_tokens=False).input_ids
    mask = [False] * len(pid) + [True] * (len(fid) - len(pid))
    torch.cuda.reset_peak_memory_stats()
    sync(); t0 = time.time()
    ok, err = True, None
    try:
        seq_logprob_sketch(mm, sk, [fid], [mask])
    except torch.cuda.OutOfMemoryError as e:
        ok, err = False, str(e)[:300]
    sync(); t_rl = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"probe batch of 8: {t_probe8:.2f}s; SFT 1024-token seq {t_sft:.2f}s; longest RL seq {Lmax} tokens "
          f"{t_rl:.2f}s ok={ok} peak {peak:.1f} GiB; mean RL len {mean_len:.0f}", flush=True)

    # estimates
    n_targets = len(ps)
    cfg_all = t_cfg_target * n_targets * (n_strings / n_targets) / len(strings)
    est = {}
    n_steps = 6
    est["G"] = 8 * cfg_all
    est["E"] = n_steps * (2 * 19 + N_LAYERS + 2) * cfg_all
    probe_bw = t_probe8 / 8 * n_strings
    est["D"] = (3 + 6) * probe_bw + 2 * 505 * t_sft + 283 * t_sft + 1217 * t_sft * 0 + 800 * t_rl * (mean_len / Lmax) ** 1.2
    est["F1"] = 4 * (2 + 7 * 21 + 1) * cfg_all / 2
    doc = {**prov, "block": "0-time", "load_seconds": load_s, "probe_target_for_timing": tgt,
           "n_probe_strings": n_strings, "seconds_one_config_one_target": t_cfg_target,
           "kv_cache_vs_full_max_abs_diff_nats": kv_maxdiff, "hybrid_smoke_first3": hyb[:3],
           "sketch_vs_autograd_rel_err": rel, "seconds_probe_backward_batch8": t_probe8,
           "seconds_sft_1024": t_sft, "rl_longest_tokens": Lmax, "rl_mean_tokens": mean_len,
           "seconds_rl_longest": t_rl, "rl_longest_fits": ok, "rl_oom": err, "peak_gib_rl_longest": peak,
           "estimate_gpu_hours": {k: v / 3600 for k, v in est.items()},
           "estimate_notes": ("G: 8 models x all probe strings. E: 6 steps x (19 bottom-up + 19 top-down + 36 single-"
                              "layer + before/after) configs. D: probe sketches at 3 before-models + 6 after-models "
                              "(lazy check), SFT as-trained sequences (E3, E3c, RS-SFT), 800 RL completions. F1: 4 primary "
                              "steps x (collection + 7 k x 21 subspaces + global vector) on half the targets.")}
    p = write_json("time_estimate.json", doc)
    print(json.dumps(doc["estimate_gpu_hours"], indent=1))
    print(f"-> {p}")
    notify(f"MIRA 19 Block 0 GPU smoke: KV diff {kv_maxdiff:.3f}, sketch rel err {rel:.4f}, RL longest {Lmax} tok fits={ok}; "
           f"est h: " + ", ".join(f"{k}={v/3600:.1f}" for k, v in est.items()))


if __name__ == "__main__":
    main()
