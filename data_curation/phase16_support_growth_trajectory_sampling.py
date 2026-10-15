#!/usr/bin/env python
"""
data_curation/phase16_support_growth_trajectory_sampling.py — regular
Claude's "does the new-recipe rate change over practice" task
(2026-10-15): does E1's support growth rise, fall, or stay flat across
training (RS-SFT init -> checkpoint 100 -> 200 -> 300), and does Phase
12 (the validator-reward GDPO arm, same RS-SFT init) show the same
pattern.

Draws 32 samples/target on the 40 held-out targets and the 35 ASTRAL
targets from the checkpoints NOT already sampled:
  - E1 checkpoint-100, checkpoint-200 (both target sets; checkpoint-300
    == final, already sampled -- results/astral_gen_n32_e1.json,
    data/novelty/e1_heldout_n32.jsonl, byte-identical adapter hash
    confirmed before launch, not re-drawn here).
  - Phase 12 (gdpo-phase12-rssft-beta0) checkpoint-100, checkpoint-200
    (both target sets) and checkpoint-300 (held-out only -- ASTRAL
    already exists, results/astral_gen_n32_gdpo_phase12_beta0.json,
    confirmed checkpoint-300 by its own "checkpoint" field).

RS-SFT itself (step 0) needs no new sampling: its 200-sample reference
and 32-sample resampling-rate draws already exist
(data/novelty/rssft_reference_{astral,heldout}.jsonl) and ARE the
reference distribution every checkpoint below is compared against
(both E1 and Phase 12 are initialized from RS-SFT).

Environment guard: refuses to run outside mira-vllm.

Usage (tmux, mira-vllm, source-activated):
  source ~/envs/mira-vllm/bin/activate && python data_curation/phase16_support_growth_trajectory_sampling.py
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
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
N_SAMPLES = 32

HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
ASTRAL_PATH = REPO_ROOT / "results" / "astral_validation_set.json"

# (label, lora_path, target_sets_to_draw)
JOBS = [
    ("e1_ckpt100", REPO_ROOT / "runs" / "gdpo-qlora-gdpo-e1-ranker-rssft" / "checkpoint-100", ("heldout", "astral")),
    ("e1_ckpt200", REPO_ROOT / "runs" / "gdpo-qlora-gdpo-e1-ranker-rssft" / "checkpoint-200", ("heldout", "astral")),
    ("phase12_ckpt100", REPO_ROOT / "runs" / "gdpo-qlora-gdpo-phase12-rssft-beta0" / "checkpoint-100", ("heldout", "astral")),
    ("phase12_ckpt200", REPO_ROOT / "runs" / "gdpo-qlora-gdpo-phase12-rssft-beta0" / "checkpoint-200", ("heldout", "astral")),
    ("phase12_ckpt300", REPO_ROOT / "runs" / "gdpo-qlora-gdpo-phase12-rssft-beta0" / "checkpoint-300", ("heldout",)),
]


def out_path(label: str, target_set: str) -> Path:
    return REPO_ROOT / "data" / "novelty" / f"{label}_{target_set}_n32.jsonl"


def _assert_mira_vllm_environment() -> None:
    actual = Path(sys.prefix).resolve()
    if actual != EXPECTED_ENV:
        raise RuntimeError(
            f"phase16_support_growth_trajectory_sampling.py refuses to run "
            f"outside mira-vllm. Expected sys.prefix={EXPECTED_ENV}, got {actual}.")


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
    _assert_mira_vllm_environment()
    for label, path, _ in JOBS:
        if not path.exists():
            raise FileNotFoundError(f"{label}: checkpoint not found at {path}")

    notify("Phase 16 support-growth trajectory sampling: loading base + "
          "E1/Phase12 checkpoint LoRA adapters via vLLM...")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
             max_model_len=2048 + MAX_NEW_TOKENS, seed=42, enable_lora=True,
             max_lora_rank=64)
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)

    heldout_targets = json.loads(HELDOUT_TARGETS_PATH.read_text())
    astral_targets = [t["target"] for t in json.loads(ASTRAL_PATH.read_text())["targets"]]
    targets_by_set = {"heldout": heldout_targets, "astral": astral_targets}

    n_total = sum(len(targets_by_set[ts]) for _, _, sets in JOBS for ts in sets) * N_SAMPLES
    notify(f"Phase 16 support-growth trajectory sampling: loaded. {n_total} completions to draw. Starting.")

    for lora_int_id, (label, path, target_sets) in enumerate(JOBS, start=1):
        lora_request = LoRARequest(label, lora_int_id, str(path))
        for target_set in target_sets:
            print(f"=== {label}, {target_set} ({len(targets_by_set[target_set])} x {N_SAMPLES}) ===", flush=True)
            run_one(llm, tok, sampling_params, lora_request, label,
                   targets_by_set[target_set], out_path(label, target_set))
        notify(f"Phase 16 support-growth trajectory sampling: {label} done ({', '.join(target_sets)}).")

    notify("Phase 16 support-growth trajectory sampling COMPLETE: all checkpoints, all target sets done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        notify(f"Phase 16 support-growth trajectory sampling CRASHED:\n{tb[-1500:]}")
        raise
