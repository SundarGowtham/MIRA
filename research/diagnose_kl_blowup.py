#!/usr/bin/env python
"""
diagnose_kl_blowup.py — Phase 12 KL-estimator blowup diagnostic (2026-09-10).

Finds the highest per-token r = ref_logp - policy_logp among completions
generated at the training steps immediately before the KL estimator
(exp(r) - r - 1) exploded (steps ~50-75, runs/gdpo-qlora-gdpo-phase12-rssft),
to test the hypothesis that RS-SFT sharpened specific (likely fractional-
number) tokens to near-zero probability relative to base.

APPROXIMATION, stated plainly: no checkpoint was saved between step 0 and
the kill at step 96 (save_steps=100), so the exact policy weights at step
75 aren't recoverable. This uses the RS-SFT-from-base checkpoint itself
(the run's step-0 init) as a stand-in for "policy at step 75" -- reasonable
given only ~75 clipped (max_grad_norm=1.0), lr=1e-5 steps separate them,
but not exact. Treat the result as directionally diagnostic, not a precise
reconstruction of the actual blowup event.

Usage (tmux):
  PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python research/diagnose_kl_blowup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

GENERATIONS = Path("runs/gdpo-qlora-gdpo-phase12-rssft/generations.jsonl")
RS_SFT_CKPT = "runs/sft-qlora-rs-sft-from-base/final"
MODEL_NAME = "Qwen/Qwen3-8B"
STEP_RANGE = set(range(50, 76))
N_SAMPLE = 500
OUT = Path("results/kl_blowup_diagnostic.json")


def load_target_prompts() -> dict[str, str]:
    prompts = {}
    for p in [Path("data/rl_run3/rl3_train.jsonl"), Path("data/rl_run3/rl3_val.jsonl"),
              Path("data/rl_run3/rl3_probe.jsonl")]:
        if not p.exists():
            continue
        for line in p.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec.get("target")
            pr = rec.get("prompt")
            if t and pr:
                prompts[t] = pr
    return prompts


def per_token_logps(model, tok, prompt_text: str, completion_text: str, device):
    """Teacher-forced log-prob of each completion token under `model`."""
    chat_prompt = tok.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(chat_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = tok(chat_prompt + completion_text, return_tensors="pt",
                   add_special_tokens=False, truncation=True, max_length=8192).input_ids[0]
    n_prompt = len(prompt_ids)
    if n_prompt >= len(full_ids):
        return None, None
    with torch.no_grad():
        out = model(full_ids.unsqueeze(0).to(device))
        logits = out.logits[0].float()  # (seq, vocab)
    logprobs = torch.log_softmax(logits, dim=-1)
    # token i's logprob comes from logits at position i-1
    completion_token_ids = full_ids[n_prompt:]
    gather_positions = torch.arange(n_prompt - 1, len(full_ids) - 1)
    token_logps = logprobs[gather_positions, completion_token_ids]
    return token_logps.cpu(), completion_token_ids.cpu()


def main():
    target_prompts = load_target_prompts()
    recs = [json.loads(l) for l in GENERATIONS.open()]
    window = [r for r in recs if r.get("step") in STEP_RANGE and r["target"] in target_prompts]
    print(f"{len(window)} completions in steps 50-75 with a recoverable prompt", flush=True)
    sample = window[:N_SAMPLE]

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    # Sequential loading, not simultaneous: two bf16 8B models at once
    # (~16GB each) would sit right at the 32GB card's ceiling. Compute all
    # reference logps first, free that model, then load the policy proxy.
    print("loading reference (base Qwen3-8B)...", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device)
    ref_model.eval()

    ref_cache = {}
    for i, r in enumerate(sample):
        prompt_text = target_prompts[r["target"]]
        ref_lp, tok_ids = per_token_logps(ref_model, tok, prompt_text, r["completion"], device)
        if ref_lp is not None:
            ref_cache[i] = (ref_lp, tok_ids)
        if (i + 1) % 25 == 0:
            print(f"  ref [{i+1}/{len(sample)}]", flush=True)

    del ref_model
    torch.cuda.empty_cache()

    print("loading policy proxy (RS-SFT-from-base)...", flush=True)
    policy_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device)
    policy_model = PeftModel.from_pretrained(policy_base, RS_SFT_CKPT)
    policy_model.eval()

    results = []
    for i, r in enumerate(sample):
        if i not in ref_cache:
            continue
        ref_lp, tok_ids = ref_cache[i]
        prompt_text = target_prompts[r["target"]]
        pol_lp, _ = per_token_logps(policy_model, tok, prompt_text, r["completion"], device)
        if pol_lp is None or len(pol_lp) != len(ref_lp):
            continue
        r_vals = (ref_lp - pol_lp).numpy()
        max_idx = int(r_vals.argmax())
        max_r = float(r_vals[max_idx])
        token_str = tok.decode([int(tok_ids[max_idx])])
        context = tok.decode(tok_ids[max(0, max_idx - 8):max_idx + 8])
        results.append({
            "target": r["target"], "step": r["step"],
            "max_r": max_r, "max_r_token": token_str, "context": context,
            "n_tokens": len(r_vals),
        })
        if (i + 1) % 25 == 0 or max_r > 15:
            print(f"  [{i+1}/{len(sample)}] {r['target']}: max_r={max_r:.2f} token={token_str!r}",
                  flush=True)

    results.sort(key=lambda x: -x["max_r"])
    print("\n" + "=" * 78)
    print("TOP 10 HIGHEST-r TOKENS ACROSS THE SAMPLE")
    for x in results[:10]:
        print(f"  r={x['max_r']:7.2f}  target={x['target']:<30} token={x['max_r_token']!r}")
        print(f"           context: ...{x['context']!r}...")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": "APPROXIMATION: policy proxy is the RS-SFT-from-base checkpoint "
               "(step-0 init), not the exact step-75 weights (no checkpoint was "
               "saved before the kill). Directionally diagnostic only.",
        "n_sampled": len(sample), "n_scored": len(results),
        "results": results,
    }, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
