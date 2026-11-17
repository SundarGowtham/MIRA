#!/usr/bin/env python
"""
data_curation/phase20_sampling.py -- every Phase 20 sampling job
(docs/phases/PHASE20_PREREG.md, "Sampling").

Same code path as data_curation/phase16_e3_e3c_sampling.py and
phase16_rssft_reference_sampling.py: vLLM, bf16 base Qwen3-8B + LoRA,
LLM(seed=42), SamplingParams(temperature=0.9, top_p=0.95, max_tokens=8192),
chat-template closed-book prompt (add_generation_prompt=True), the 35 ASTRAL
targets and the 40 held-out targets of data/novelty/heldout_targets.json.
The only addition: every row records the sampling settings ("meta": model,
adapter path, adapter sha256, temperature, top_p, max_tokens, seed, vLLM
version, GPU name), so the chain can verify files drawn on another machine.

Resume-safe: a target already present in an output file (for the reference,
present in its "reference" group) is skipped.

Jobs (one per chain stage; each can run on any machine with a clone of the
repo plus the named adapter folders, see docs/phases/PHASE20_RUNPOD.md):
  stage2       sft-rssft-full, sft-e3-full, sft-e3c-full: ASTRAL + held-out, 32/target
  stage4_e1    E1-full checkpoints 100, 200, 300: ASTRAL + held-out, 32/target
  stage4_ref   sft-rssft-full reference: 200 "reference" + 32 "resampling_rate" per target
  stage5       old corpus_sft, gdpo_v3_ckpt300, phase12_ckpt300, e1_ckpt300: ASTRAL only, 32/target
  stage6       sft-corpus-full: ASTRAL + held-out, 32/target

Usage (mira-vllm env):  python data_curation/phase20_sampling.py --job stage2
                        python data_curation/phase20_sampling.py --job stage2 --list-outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NTFY_TOPIC = "mira-g5x7k2-status"
BASE_MODEL = "Qwen/Qwen3-8B"
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
SEED = 42
N32 = 32
REF_N = 200
REF_RESAMPLE_N = 32

OUT = REPO_ROOT / "data" / "novelty" / "phase20"
HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
ASTRAL_PATH = REPO_ROOT / "results" / "astral_validation_set.json"

# model label -> adapter directory (relative to the repo root)
ADAPTERS = {
    "sft_rssft_full": "runs/sft-qlora-sft-rssft-full/final",
    "sft_e3_full": "runs/sft-qlora-sft-e3-full/final",
    "sft_e3c_full": "runs/sft-qlora-sft-e3c-full/final",
    "sft_corpus_full": "runs/sft-qlora-sft-corpus-full/final",
    "e1full_ckpt100": "runs/gdpo-qlora-gdpo-e1-ranker-rssft-full/checkpoint-100",
    "e1full_ckpt200": "runs/gdpo-qlora-gdpo-e1-ranker-rssft-full/checkpoint-200",
    "e1full_ckpt300": "runs/gdpo-qlora-gdpo-e1-ranker-rssft-full/checkpoint-300",
    # Stage 5: old models, unchanged adapters
    "corpus_sft": "runs/sft-qlora-sft-v3-2nd-rank16/final",
    "gdpo_v3_ckpt300": "runs/gdpo-qlora-gdpo-v3/checkpoint-300",
    "phase12_ckpt300": "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300",
    "e1_ckpt300": "runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-300",
}


def n32_jobs(models, sets=("astral", "heldout"), suffix=""):
    return [{"model": m, "set": s, "kind": "n32",
             "out": f"{m}_{s}_n32{suffix}.jsonl"} for m in models for s in sets]


JOBS = {
    "stage2": n32_jobs(["sft_rssft_full", "sft_e3_full", "sft_e3c_full"]),
    "stage4_e1": n32_jobs(["e1full_ckpt100", "e1full_ckpt200", "e1full_ckpt300"]),
    "stage4_ref": [{"model": "sft_rssft_full", "set": s, "kind": "reference",
                    "out": f"sft_rssft_full_reference_{s}.jsonl"} for s in ("astral", "heldout")],
    "stage5": n32_jobs(["corpus_sft", "gdpo_v3_ckpt300", "phase12_ckpt300", "e1_ckpt300"],
                       sets=("astral",), suffix="_t09vllm"),
    "stage6": n32_jobs(["sft_corpus_full"]),
}


def notify(msg: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode(), method="POST"), timeout=10)
    except Exception as e:
        print(f"[ntfy failed, continuing]: {e}", flush=True)
    print(f"[ntfy] {msg}", flush=True)


def adapter_sha256(adapter_dir: Path) -> str:
    h = hashlib.sha256()
    with (adapter_dir / "adapter_model.safetensors").open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def targets(which: str) -> list[str]:
    if which == "astral":
        return [t["target"] for t in json.loads(ASTRAL_PATH.read_text())["targets"]]
    return list(json.loads(HELDOUT_TARGETS_PATH.read_text()))


def closed_prompt(target: str) -> str:
    from stratified_difficulty_eval import SYSTEM_MSG
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def expected_rows(job: dict) -> int:
    per = REF_N + REF_RESAMPLE_N if job["kind"] == "reference" else N32
    return per * len(targets(job["set"]))


def done_targets(path: Path, kind: str) -> set:
    done = set()
    if path.exists():
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                if kind != "reference" or r.get("sample_group") == "resampling_rate":
                    done.add(r["target"])
    return done


def draw(llm, tok, params, lora, target, n, label, meta, extra=None):
    rendered = tok.apply_chat_template([{"role": "user", "content": closed_prompt(target)}],
                                       tokenize=False, add_generation_prompt=True)
    outs = llm.generate([rendered] * n, params, lora_request=lora)
    rows = []
    for o in outs:
        k = len(o.outputs[0].token_ids)
        rows.append({"target": target, "completion": o.outputs[0].text, "n_tokens": k,
                     "clipped": k >= MAX_NEW_TOKENS, "checkpoint": label, **(extra or {}),
                     "meta": meta})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--list-outputs", action="store_true")
    a = ap.parse_args()
    jobs = JOBS[a.job]
    if a.list_outputs:
        for j in jobs:
            print(OUT / j["out"], ADAPTERS[j["model"]], expected_rows(j))
        return

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    models = sorted({j["model"] for j in jobs})
    for m in models:
        if not (REPO_ROOT / ADAPTERS[m] / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"adapter for {m} not found at {ADAPTERS[m]}")
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    notify(f"MIRA phase20 sampling {a.job}: loading {BASE_MODEL} + {len(models)} adapters on {gpu}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048 + MAX_NEW_TOKENS, seed=SEED, enable_lora=True, max_lora_rank=64)
    params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)
    requests, metas = {}, {}
    for i, m in enumerate(models, start=1):
        path = REPO_ROOT / ADAPTERS[m]
        requests[m] = LoRARequest(m, i, str(path))
        metas[m] = {"model": m, "base_model": BASE_MODEL, "adapter": ADAPTERS[m],
                    "adapter_sha256": adapter_sha256(path), "temperature": TEMPERATURE,
                    "top_p": TOP_P, "max_tokens": MAX_NEW_TOKENS, "seed": SEED,
                    "vllm_version": vllm.__version__, "gpu": gpu, "backend": "vllm",
                    "dtype": "bfloat16", "job": a.job}
    for j in jobs:
        m, path = j["model"], OUT / j["out"]
        tl = targets(j["set"])
        done = done_targets(path, j["kind"])
        print(f"=== {m} {j['set']} {j['kind']}: {len(tl)} targets, {len(done)} done -> {path}", flush=True)
        t0 = time.time()
        for i, t in enumerate(tl):
            if t in done:
                continue
            if j["kind"] == "reference":
                rows = draw(llm, tok, params, requests[m], t, REF_N, m, metas[m], {"sample_group": "reference"})
                rows += draw(llm, tok, params, requests[m], t, REF_RESAMPLE_N, m, metas[m],
                             {"sample_group": "resampling_rate"})
            else:
                rows = draw(llm, tok, params, requests[m], t, N32, m, metas[m])
            with path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            if (i + 1) % 5 == 0:
                print(f"  {m} {j['set']}: {i + 1}/{len(tl)} targets, {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        print(f"=== {m} {j['set']} DONE", flush=True)
    notify(f"MIRA phase20 sampling {a.job}: COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        notify(f"MIRA phase20 sampling CRASHED:\n{traceback.format_exc()[-1500:]}")
        raise
