#!/usr/bin/env python
"""
verify_ref_model_identity.py — PHASE_11_REVISED.md Step 1 precheck.

Question: does GDPO's current KL reference actually equal pi_SFT (as
PHASE_11_REVISED.md / CLAUDE.md finding 17 assume) or pi_base (as TRL's PEFT
disable-adapter mechanism implies, given core/model.py's load_with_adapter
never creates a second adapter named "ref")?

Loads the model exactly as experiments/grpo.py does (base Qwen3-8B + SFT
LoRA adapter via PeftModel.from_pretrained(..., adapter_name defaults to
"default")), then compares log-probs on a fixed prompt under three
conditions:
  1. policy (adapter enabled)              -- what GRPOTrainer trains
  2. model.disable_adapter()                -- what GRPOTrainer uses as the
                                               reference when no "ref"
                                               adapter exists (grpo_trainer.py
                                               line ~2494, trl/trainer/utils.py
                                               use_adapter(model, None))
  3. raw base Qwen3-8B loaded fresh, no adapter at all

If (2) matches (3) (not (1)), the current reference is pi_base, not pi_SFT --
PHASE_11_REVISED.md's Step 1 ("re-anchor KL to base") would be a config
no-op against the actual current setup, and finding 17's stated mechanism
(anchored to the collapsed pi_SFT) needs revision even though the ASTRAL
n=32 empirical numbers (finding 16) stand on their own.

No training. A few forward passes only.

Usage (tmux):
  PYTHONPATH=. uv run python run_debug_and_analysis/verify_ref_model_identity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from core.model import load_with_adapter, load_model, load_tokenizer  # noqa: E402

SFT_ADAPTER = "runs/sft-qlora-sft-v3-2nd-rank16/final"
PROMPT = "Provide a solid-state synthesis route for BaTiO3 as a JSON object."


def get_logits(model, tok, prompt: str):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**ids)
    return out.logits[0, -1, :].float().cpu()


def main():
    print("loading policy (base + SFT adapter, is_trainable=True, exactly as grpo.py does)...",
          flush=True)
    model, tok = load_with_adapter("Qwen/Qwen3-8B", "qlora", smoke=False, init_from=SFT_ADAPTER)
    print("peft_config keys:", list(model.peft_config.keys()), flush=True)
    print("active_adapter:", model.active_adapter, flush=True)

    logits_policy = get_logits(model, tok, PROMPT)

    print("disabling adapter (this is what TRL's use_adapter(model, None) does for the "
          "reference logprobs)...", flush=True)
    with model.disable_adapter():
        logits_disabled = get_logits(model, tok, PROMPT)

    del model
    torch.cuda.empty_cache()

    print("loading a completely fresh base Qwen3-8B (same qlora quantization path, "
          "no PEFT/LoRA wrapper at all)...", flush=True)
    base_model = load_model("Qwen/Qwen3-8B", "qlora", smoke=False)
    logits_base_fresh = get_logits(base_model, tok, PROMPT)

    d_policy_vs_disabled = (logits_policy - logits_disabled).abs().max().item()
    d_disabled_vs_base = (logits_disabled - logits_base_fresh).abs().max().item()
    d_policy_vs_base = (logits_policy - logits_base_fresh).abs().max().item()

    print("\n" + "=" * 78)
    print(f"max |logit diff| policy vs disable_adapter():   {d_policy_vs_disabled:.6f}")
    print(f"max |logit diff| disable_adapter() vs fresh base: {d_disabled_vs_base:.6f}")
    print(f"max |logit diff| policy vs fresh base:            {d_policy_vs_base:.6f}")
    print()
    if d_disabled_vs_base < 1e-3 and d_policy_vs_disabled > 1e-2:
        print("VERDICT: disable_adapter() == base, != policy (SFT). "
              "The current GDPO reference is pi_base, NOT pi_SFT.")
    elif d_policy_vs_disabled < 1e-3:
        print("VERDICT: disable_adapter() == policy (adapter had no effect?). Unexpected -- "
              "investigate before trusting either PHASE_11_REVISED.md's premise or this script.")
    else:
        print("VERDICT: inconclusive -- neither clean match. Investigate before proceeding.")


if __name__ == "__main__":
    main()
