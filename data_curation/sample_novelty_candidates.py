#!/usr/bin/env python
"""
data_curation/sample_novelty_candidates.py — Phase 16 §6.2
(misc/PHASE16_INSTRUCTIONS.md). Samples --samples-per-target (64)
completions per target from a checkpoint (base Qwen3-8B for round 1; the
round-1-trained model, as a LoRA adapter, for round 2 per §6.6) using
offline vLLM, with the Phase 12 sampling settings (rule 2, SS1): closed-book
prompt, temperature 0.9, top_p 0.95, 8192-token completion cap.

This script ONLY samples and dumps raw (prompt, completion, target)
records -- no validity/novelty filtering. §6.3/§6.4's E3/E3c filters are
a separate step applied to this script's output.

Environment guard (Phase 16 "Decisions after pass 2", the mis-install
correction): this is a sampling entry point that runs inside mira-vllm;
it asserts sys.prefix and refuses to run anywhere else. Invoke with
mira-vllm's own python by absolute path:

  ~/envs/mira-vllm/bin/python data_curation/sample_novelty_candidates.py \
      --round 1 --out data/novelty/round1_samples.jsonl

NOT YET RUN. Written per the pass-2 "build the novelty split and
sampling script" step; launching it is a separate, later step.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_ENV = (Path.home() / "envs" / "mira-vllm").resolve()


def _assert_mira_vllm_environment() -> None:
    actual = Path(sys.prefix).resolve()
    if actual != EXPECTED_ENV:
        raise RuntimeError(
            f"sample_novelty_candidates.py refuses to run outside the "
            f"mira-vllm environment. Expected sys.prefix={EXPECTED_ENV}, "
            f"got {actual}. Invoke with ~/envs/mira-vllm/bin/python by "
            f"absolute path, not bare `python` or `uv run`.")


from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402

DEFAULT_CHECKPOINT = "Qwen/Qwen3-8B"
# Phase 12 sampling settings (rule 2) -- NOT build_rs_sft_dataset.py's own
# defaults (temperature=1.0 there), these must match GDPO's actual settings.
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 8192


def closed_prompt(target: str) -> str:
    """Identical construction to data_curation/build_rs_sft_dataset.py's
    own prompt string, for parity with every other closed-book sampling
    script in this project."""
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, choices=[1, 2], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--lora-path", type=Path, default=None,
                   help="Round 2 only: path to the round-1-trained model's "
                        "LoRA adapter, served via vLLM's LoRA support.")
    p.add_argument("--targets-file", type=Path,
                   default=REPO_ROOT / "data" / "novelty" / "training_targets.json")
    p.add_argument("--samples-per-target", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="Offline sampling, no training sharing the GPU -- "
                        "can run much higher than colocate mode's ~0.3-0.45.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    _assert_mira_vllm_environment()
    args = parse_args()
    if args.round == 2 and args.lora_path is None:
        sys.exit("--round 2 requires --lora-path (the round-1-trained model)")

    targets = json.loads(args.targets_file.read_text())
    print(f"sampling {args.samples_per_target} completions x {len(targets)} targets "
          f"(round {args.round}, checkpoint={args.lora_path or args.model})", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm_kwargs = dict(model=args.model, dtype="bfloat16",
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      max_model_len=2048 + MAX_NEW_TOKENS, seed=args.seed)
    lora_request = None
    if args.lora_path is not None:
        from vllm.lora.request import LoRARequest
        llm_kwargs["enable_lora"] = True
        lora_request = LoRARequest("round1", 1, str(args.lora_path))
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)

    prompts, prompt_targets = [], []
    for target in targets:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": closed_prompt(target)}],
            tokenize=False, add_generation_prompt=True)
        for _ in range(args.samples_per_target):
            prompts.append(rendered)
            prompt_targets.append(target)

    t0 = time.time()
    gen_kwargs = {"lora_request": lora_request} if lora_request else {}
    outputs = llm.generate(prompts, sampling_params, **gen_kwargs)
    print(f"generation done in {(time.time() - t0) / 60:.1f} min, "
          f"{len(outputs)} completions", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for prompt, target, out in zip(prompts, prompt_targets, outputs):
            f.write(json.dumps({
                "prompt": closed_prompt(target), "rendered_prompt": prompt,
                "completion": out.outputs[0].text, "target": target,
                "round": args.round, "checkpoint": str(args.lora_path or args.model),
            }) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
