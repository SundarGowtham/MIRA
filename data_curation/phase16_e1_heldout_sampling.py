#!/usr/bin/env python
"""
data_curation/phase16_e1_heldout_sampling.py — Phase 16 §7 prediction 1,
support-growth half: E1's own 32-sample draw per target on the 40
held-out targets, to compare against RS-SFT's already-drawn 200-sample
reference (data/novelty/rssft_reference_heldout.jsonl) per the
pre-registration's "Support growth of model M relative to reference R"
definition. The ASTRAL-target analog of this 32-sample draw is just
E1's existing n=32 ASTRAL eval generation (results/astral_gen_n32_e1.json)
-- no second draw needed there.

Via vLLM LoRA serving of E1's checkpoint (same pattern as
phase16_rssft_reference_sampling.py for RS-SFT). Not E1's own
training settings; same closed-book, Phase-12-matching sampling
settings every other script in this project uses.

Environment guard: refuses to run outside mira-vllm.

Usage (tmux, mira-vllm, source-activated):
  source ~/envs/mira-vllm/bin/activate && python data_curation/phase16_e1_heldout_sampling.py
"""
from __future__ import annotations

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
E1_LORA_PATH = REPO_ROOT / "runs" / "gdpo-qlora-gdpo-e1-ranker-rssft" / "final"
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
N_SAMPLES = 32

HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
OUT_PATH = REPO_ROOT / "data" / "novelty" / "e1_heldout_n32.jsonl"


def _assert_mira_vllm_environment() -> None:
    actual = Path(sys.prefix).resolve()
    if actual != EXPECTED_ENV:
        raise RuntimeError(
            f"phase16_e1_heldout_sampling.py refuses to run outside mira-vllm. "
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


def main():
    _assert_mira_vllm_environment()
    if not E1_LORA_PATH.exists():
        raise FileNotFoundError(f"E1 checkpoint not found at {E1_LORA_PATH}")

    notify("Phase 16 E1 held-out sampling: loading base + E1 LoRA adapter via vLLM...")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # Tokenizer from base, not the checkpoint dir -- the RS-SFT reference
    # sampling run hit a transformers-version incompatibility loading a
    # checkpoint's own saved tokenizer_config.json (mira-vllm's
    # transformers downgraded to 4.57.6 by the vllm install); same fix.
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
             max_model_len=2048 + MAX_NEW_TOKENS, seed=42, enable_lora=True,
             max_lora_rank=64)
    lora_request = LoRARequest("e1", 1, str(E1_LORA_PATH))
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)

    targets = json.loads(HELDOUT_TARGETS_PATH.read_text())
    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["target"])
    notify(f"Phase 16 E1 held-out sampling: loaded. {len(targets)} targets x {N_SAMPLES}, "
          f"{len(done)} already done. Starting.")

    t0 = time.time()
    for i, target in enumerate(targets):
        if target in done:
            continue
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": closed_prompt(target)}],
            tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([rendered] * N_SAMPLES, sampling_params, lora_request=lora_request)
        with OUT_PATH.open("a") as f:
            for out in outputs:
                n_tokens = len(out.outputs[0].token_ids)
                f.write(json.dumps({
                    "target": target, "completion": out.outputs[0].text,
                    "n_tokens": n_tokens, "clipped": n_tokens >= MAX_NEW_TOKENS,
                    "checkpoint": "e1",
                }) + "\n")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(targets)} targets done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)

    notify(f"Phase 16 E1 held-out sampling COMPLETE: {len(targets)} targets x {N_SAMPLES} done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        notify(f"Phase 16 E1 held-out sampling CRASHED:\n{tb[-1500:]}")
        raise
