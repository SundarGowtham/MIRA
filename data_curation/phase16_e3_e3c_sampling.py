#!/usr/bin/env python
"""
data_curation/phase16_e3_e3c_sampling.py — Phase 16, per regular
Claude's instruction while GPU was idle after E1's evaluation: "have
CC start sampling E3 and E3c now, 32 samples per compound on the
held-out and ASTRAL compounds, then compute the novelty rate and
support growth. That's the critical path. E4 waits on that result."

Draws 32 samples/target from both E3 (runs/sft-qlora-e3-round1/final)
and E3c (runs/sft-qlora-e3c-round1/final) on the 40 held-out targets
and the 35 ASTRAL targets, via vLLM LoRA serving (both adapters loaded
in one vLLM instance, same pattern as phase16_rssft_reference_sampling.py
and phase16_e1_heldout_sampling.py). Both E3 and E3c are LoRA adapters
on BASE Qwen3-8B (not RS-SFT), per misc/PHASE16_INSTRUCTIONS.md's own
arm table ("E3 | ... | base | manifold"). The matching reference
distribution for E3/E3c's support-growth and novelty-rate computation
is therefore BASE's own 200+32 reference draws
(data/novelty/base_reference_{heldout,astral}.jsonl), which already
exist from the original sampling session -- no new reference draw
needed here, only E3/E3c's own 32-sample comparison draw.

Environment guard: refuses to run outside mira-vllm.

Usage (tmux, mira-vllm, source-activated):
  source ~/envs/mira-vllm/bin/activate && python data_curation/phase16_e3_e3c_sampling.py
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
E3_LORA_PATH = REPO_ROOT / "runs" / "sft-qlora-e3-round1" / "final"
E3C_LORA_PATH = REPO_ROOT / "runs" / "sft-qlora-e3c-round1" / "final"
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
N_SAMPLES = 32

HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
ASTRAL_PATH = REPO_ROOT / "results" / "astral_validation_set.json"

OUT_PATHS = {
    ("e3", "heldout"): REPO_ROOT / "data" / "novelty" / "e3_heldout_n32.jsonl",
    ("e3", "astral"): REPO_ROOT / "data" / "novelty" / "e3_astral_n32.jsonl",
    ("e3c", "heldout"): REPO_ROOT / "data" / "novelty" / "e3c_heldout_n32.jsonl",
    ("e3c", "astral"): REPO_ROOT / "data" / "novelty" / "e3c_astral_n32.jsonl",
}


def _assert_mira_vllm_environment() -> None:
    actual = Path(sys.prefix).resolve()
    if actual != EXPECTED_ENV:
        raise RuntimeError(
            f"phase16_e3_e3c_sampling.py refuses to run outside mira-vllm. "
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


def run_one(llm, tok, sampling_params, lora_request, checkpoint_label: str,
           targets: list[str], out_path: Path) -> None:
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["target"])
    print(f"{checkpoint_label}: {len(targets)} targets, {len(done)} already done", flush=True)
    t0 = time.time()
    for i, target in enumerate(targets):
        if target in done:
            continue
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": closed_prompt(target)}],
            tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([rendered] * N_SAMPLES, sampling_params, lora_request=lora_request)
        with out_path.open("a") as f:
            for out in outputs:
                n_tokens = len(out.outputs[0].token_ids)
                f.write(json.dumps({
                    "target": target, "completion": out.outputs[0].text,
                    "n_tokens": n_tokens, "clipped": n_tokens >= MAX_NEW_TOKENS,
                    "checkpoint": checkpoint_label,
                }) + "\n")
        if (i + 1) % 10 == 0:
            print(f"  {checkpoint_label}: {i + 1}/{len(targets)} targets done, "
                  f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    print(f"{checkpoint_label}: DONE", flush=True)


def main():
    _assert_mira_vllm_environment()
    for p in (E3_LORA_PATH, E3C_LORA_PATH):
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found at {p}")

    notify("Phase 16 E3/E3c sampling: loading base + E3/E3c LoRA adapters via vLLM...")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # Tokenizer from base, not the checkpoint dir -- same fix as every
    # other Phase 16 vLLM script (mira-vllm's downgraded transformers
    # can't parse these checkpoints' own saved tokenizer_config.json).
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
             max_model_len=2048 + MAX_NEW_TOKENS, seed=42, enable_lora=True,
             max_lora_rank=64)
    e3_request = LoRARequest("e3", 1, str(E3_LORA_PATH))
    e3c_request = LoRARequest("e3c", 2, str(E3C_LORA_PATH))
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)

    heldout_targets = json.loads(HELDOUT_TARGETS_PATH.read_text())
    astral_targets = [t["target"] for t in json.loads(ASTRAL_PATH.read_text())["targets"]]

    n_total = (len(heldout_targets) + len(astral_targets)) * N_SAMPLES * 2
    notify(f"Phase 16 E3/E3c sampling: loaded. {n_total} completions to draw "
          f"(40 held-out + 35 ASTRAL targets x 32, x2 checkpoints). Starting.")

    print("=== E3, held-out (40 x 32) ===", flush=True)
    run_one(llm, tok, sampling_params, e3_request, "e3",
           heldout_targets, OUT_PATHS[("e3", "heldout")])

    print("=== E3, ASTRAL (35 x 32) ===", flush=True)
    run_one(llm, tok, sampling_params, e3_request, "e3",
           astral_targets, OUT_PATHS[("e3", "astral")])

    print("=== E3c, held-out (40 x 32) ===", flush=True)
    run_one(llm, tok, sampling_params, e3c_request, "e3c",
           heldout_targets, OUT_PATHS[("e3c", "heldout")])

    print("=== E3c, ASTRAL (35 x 32) ===", flush=True)
    run_one(llm, tok, sampling_params, e3c_request, "e3c",
           astral_targets, OUT_PATHS[("e3c", "astral")])

    notify("Phase 16 E3/E3c sampling COMPLETE: E3 + E3c, held-out + ASTRAL, all done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        notify(f"Phase 16 E3/E3c sampling CRASHED:\n{tb[-1500:]}")
        raise
