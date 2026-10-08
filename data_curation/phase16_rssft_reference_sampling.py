#!/usr/bin/env python
"""
data_curation/phase16_rssft_reference_sampling.py — Phase 16, scheduled
by regular Claude after the E3/E3c filter: "E1's support test compares
against RS-SFT's own 200 + 32 samples per target, which haven't been
drawn yet. Schedule them after the filter, before E1 is evaluated."

Same shape as data_curation/phase16_sampling_session.py's Phase 2a/2b
(200 reference + 32 independent resampling-rate draw per target, on the
40 held-out targets and the 35 ASTRAL targets), but from the RS-SFT
checkpoint (runs/sft-qlora-rs-sft-from-base/final) via vLLM LoRA
serving instead of base Qwen3-8B directly. Base's own reference samples
(data/novelty/base_reference_{heldout,astral}.jsonl) already exist;
this produces the RS-SFT analogs so E1's support-growth prediction
("no support growth beyond RS-SFT's own resampling rate") can actually
be computed once E1 exists.

No smoke run gate here (this phase's generation settings/backend/parsing
were already validated by the base-model sampling session) -- but still
writes per-target, crash-safe, with ntfy at start/crash/completion.

Environment guard: refuses to run outside mira-vllm.

Usage (tmux, mira-vllm, MUST be source-activated -- see
phase16_sampling_session.py's own usage note re: ninja/PATH):
  source ~/envs/mira-vllm/bin/activate && python data_curation/phase16_rssft_reference_sampling.py
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
RS_SFT_LORA_PATH = REPO_ROOT / "runs" / "sft-qlora-rs-sft-from-base" / "final"
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 8192

HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
ASTRAL_PATH = REPO_ROOT / "results" / "astral_validation_set.json"

HELDOUT_REF_OUT = REPO_ROOT / "data" / "novelty" / "rssft_reference_heldout.jsonl"
ASTRAL_REF_OUT = REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl"

REF_SAMPLES_PER_TARGET = 200
REF_RESAMPLE_PER_TARGET = 32


def _assert_mira_vllm_environment() -> None:
    actual = Path(sys.prefix).resolve()
    if actual != EXPECTED_ENV:
        raise RuntimeError(
            f"phase16_rssft_reference_sampling.py refuses to run outside "
            f"mira-vllm. Expected sys.prefix={EXPECTED_ENV}, got {actual}.")


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


def sample_target(llm, tok, sampling_params, lora_request, target: str, n: int) -> list[dict]:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": closed_prompt(target)}],
        tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([rendered] * n, sampling_params, lora_request=lora_request)
    records = []
    for out in outputs:
        text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        records.append({
            "target": target, "completion": text, "n_tokens": n_tokens,
            "clipped": n_tokens >= MAX_NEW_TOKENS,
            "checkpoint": "rs_sft",
        })
    return records


def append_records(path: Path, records: list[dict], extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in records:
            row = {**r, **(extra or {})}
            f.write(json.dumps(row) + "\n")


def run_reference(llm, tok, sampling_params, lora_request, targets: list[str],
                  out_path: Path, label: str) -> None:
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("sample_group") == "reference":
                        done.add(rec["target"])
    print(f"{label}: {len(targets)} targets, {len(done)} already done", flush=True)
    for i, target in enumerate(targets):
        if target in done:
            continue
        ref_records = sample_target(llm, tok, sampling_params, lora_request, target, REF_SAMPLES_PER_TARGET)
        append_records(out_path, ref_records, extra={"sample_group": "reference"})
        resample_records = sample_target(llm, tok, sampling_params, lora_request, target, REF_RESAMPLE_PER_TARGET)
        append_records(out_path, resample_records, extra={"sample_group": "resampling_rate"})
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i + 1}/{len(targets)} targets done", flush=True)
    print(f"{label}: DONE", flush=True)


def main():
    _assert_mira_vllm_environment()
    if not RS_SFT_LORA_PATH.exists():
        raise FileNotFoundError(f"RS-SFT checkpoint not found at {RS_SFT_LORA_PATH}")

    notify(f"Phase 16 RS-SFT reference sampling: loading {BASE_MODEL} + "
          f"RS-SFT LoRA adapter via vLLM...")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # Load from base, not the checkpoint dir: RS-SFT is a LoRA adapter on
    # base and does not change the tokenizer; the checkpoint's own copied
    # tokenizer_config.json hit a transformers-version incompatibility
    # (mira-vllm's transformers got downgraded to 4.57.6 by the vllm
    # install -- a different save-time format than whatever produced
    # that checkpoint). Confirmed by direct execution, not assumed.
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
             max_model_len=2048 + MAX_NEW_TOKENS, seed=42, enable_lora=True,
             max_lora_rank=64)
    lora_request = LoRARequest("rs_sft", 1, str(RS_SFT_LORA_PATH))
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)

    n_total = 75 * (REF_SAMPLES_PER_TARGET + REF_RESAMPLE_PER_TARGET)
    notify(f"Phase 16 RS-SFT reference sampling: loaded. {n_total} completions "
          f"to draw (40 held-out + 35 ASTRAL targets x 232 each). Starting.")

    print("=== RS-SFT reference, held-out (40 targets x 232) ===", flush=True)
    heldout_targets = json.loads(HELDOUT_TARGETS_PATH.read_text())
    run_reference(llm, tok, sampling_params, lora_request, heldout_targets,
                  HELDOUT_REF_OUT, "rssft_heldout_ref")

    print("=== RS-SFT reference, ASTRAL (35 targets x 232) ===", flush=True)
    astral_targets = [t["target"] for t in json.loads(ASTRAL_PATH.read_text())["targets"]]
    run_reference(llm, tok, sampling_params, lora_request, astral_targets,
                  ASTRAL_REF_OUT, "rssft_astral_ref")

    notify("Phase 16 RS-SFT reference sampling COMPLETE: held-out + ASTRAL reference both done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        notify(f"Phase 16 RS-SFT reference sampling CRASHED:\n{tb[-1500:]}")
        raise
