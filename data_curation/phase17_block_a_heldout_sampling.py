#!/usr/bin/env python
"""
data_curation/phase17_block_a_heldout_sampling.py — Phase 17 Block A GPU
regeneration (user-approved, generation only, no training), held-out-set
half. Modeled directly on data_curation/phase16_support_growth_trajectory_
sampling.py (same vLLM pipeline, same sampling config) -- no held-out n=32
draw has EVER been taken for "GDPO-300" (the original run-1 checkpoint,
runs/gdpo-qlora-gdpo-v3/checkpoint-300, full-SFT init) or "Corpus SFT"
(runs/sft-qlora-sft-v3-2nd-rank16/final) -- data/novelty/ has e1_heldout,
e3_heldout, e3c_heldout, phase12_ckpt*_heldout, rssft_reference_heldout,
base_reference_heldout, but nothing for these two. This fills that gap;
the ASTRAL half for both of these checkpoints already exists
(misc/astral_gen_n32_gdpo300.json 3/35, misc/astral_gen_n32_sft.json
1/35, findings 16/18) and is separately RE-generated (not via this
script) by scripts/run_phase17_block_a_astral_regen.sh, which uses the
DIFFERENT HF-transformers pipeline (temp=1.0) that produced those
originals -- that pipeline must be replicated exactly for a valid
replication check, so it is kept as its own script, not folded in here.

This script's config (vLLM, temp=0.9, top_p=0.95, max_tokens=8192,
n=32/target) matches every OTHER held-out n=32 draw in the project
exactly (E1, Phase 12, RS-SFT, base) -- the established convention for
held-out sets, which has always differed from the ASTRAL HF-pipeline's
temp=1.0. New filenames only.

Environment guard: refuses to run outside mira-vllm.

Usage (tmux, mira-vllm, source-activated):
  source ~/envs/mira-vllm/bin/activate && \\
    python data_curation/phase17_block_a_heldout_sampling.py --job gdpo300
  source ~/envs/mira-vllm/bin/activate && \\
    python data_curation/phase17_block_a_heldout_sampling.py --job corpus_sft
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_ENV = (Path.home() / "envs" / "mira-vllm").resolve()
NTFY_TOPIC = "mira-g5x7k2-status"

BASE_MODEL = "Qwen/Qwen3-8B"
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
N_SAMPLES = 32

HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"

JOBS = {
    "gdpo300": ("gdpo300", REPO_ROOT / "runs" / "gdpo-qlora-gdpo-v3" / "checkpoint-300",
               REPO_ROOT / "data" / "novelty" / "gdpo300_heldout_n32.jsonl"),
    "corpus_sft": ("corpus_sft", REPO_ROOT / "runs" / "sft-qlora-sft-v3-2nd-rank16" / "final",
                   REPO_ROOT / "data" / "novelty" / "corpus_sft_heldout_n32.jsonl"),
}


def _assert_mira_vllm_environment() -> None:
    actual = Path(sys.prefix).resolve()
    if actual != EXPECTED_ENV:
        raise RuntimeError(
            f"phase17_block_a_heldout_sampling.py refuses to run outside mira-vllm. "
            f"Expected sys.prefix={EXPECTED_ENV}, got {actual}.")


def notify(msg: str) -> None:
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}",
                                   data=msg.encode(), method="POST"),
            timeout=10)
    except Exception as e:
        print(f"[ntfy failed, continuing]: {e}", flush=True)
    print(f"[ntfy] {msg}", flush=True)


from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402


def closed_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def run_one(llm, tok, sampling_params, lora_request, label: str,
           targets: list[str], out: Path) -> None:
    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["target"])
    print(f"{label}: {len(targets)} targets, {len(done)} already done", flush=True)
    t0 = time.time()
    for i, target in enumerate(targets):
        if target in done:
            continue
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": closed_prompt(target)}],
            tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([rendered] * N_SAMPLES, sampling_params, lora_request=lora_request)
        with out.open("a") as f:
            for o in outputs:
                n_tokens = len(o.outputs[0].token_ids)
                f.write(json.dumps({
                    "target": target, "completion": o.outputs[0].text,
                    "n_tokens": n_tokens, "clipped": n_tokens >= MAX_NEW_TOKENS,
                    "checkpoint": label,
                }) + "\n")
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i + 1}/{len(targets)} targets done, "
                  f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    print(f"{label}: DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=sorted(JOBS.keys()))
    args = ap.parse_args()
    label, path, out = JOBS[args.job]

    _assert_mira_vllm_environment()
    if not path.exists():
        raise FileNotFoundError(f"{label}: checkpoint not found at {path}")

    notify(f"Phase 17 Block A held-out sampling ({label}): loading via vLLM...")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
             max_model_len=2048 + MAX_NEW_TOKENS, seed=42, enable_lora=True,
             max_lora_rank=64)
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)

    heldout_targets = json.loads(HELDOUT_TARGETS_PATH.read_text())
    notify(f"Phase 17 Block A held-out sampling ({label}): loaded. "
          f"{len(heldout_targets) * N_SAMPLES} completions to draw. Starting.")

    lora_request = LoRARequest(label, 1, str(path))
    run_one(llm, tok, sampling_params, lora_request, label, heldout_targets, out)

    notify(f"Phase 17 Block A held-out sampling ({label}): COMPLETE.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        notify(f"Phase 17 Block A held-out sampling CRASHED:\n{tb[-1500:]}")
        raise
