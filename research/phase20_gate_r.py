#!/usr/bin/env python
"""
research/phase20_gate_r.py -- Phase 20 Gate R (after each SFT run).

  R1 (main env, --check): the trainer's logged tokens per example (TRL's
     `num_tokens`, which INCLUDES prompt tokens) must be within 5% of the
     dataset's mean full length including the prompt. Guards against a silent
     truncation. Deliberately NOT loss tokens (spec).
     num_tokens is cumulative at the last logged step s (logging_steps=25);
     examples seen by then = e*N + min(r*32, N) with k=ceil(N/32) optimizer
     steps per epoch, e = s // k, r = s % k (the last accumulation of each
     epoch takes the remainder). Mean full length comes from Gate T's
     tokenization of the same training file (results/phase20/gate_t.json).
  R2 (mira-vllm env, --sample): 8 samples on each of 8 held-out targets
     (the first 8 of data/novelty/heldout_targets.json), same vLLM settings as
     every Phase 20 draw (T 0.9, top-p 0.95, 8192 tokens, seed 42). Parse rate
     (core.reward.parse_completion) must be >= 70%. Median completion length
     reported either way.

Usage:
  ~/envs/mira-vllm/bin/python research/phase20_gate_r.py --tag sft-rssft-full --sample
  uv run python research/phase20_gate_r.py --tag sft-rssft-full --check
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

from research.phase20_common import NEW_SFT, SFT_DATASETS, notify, provenance, write_json  # noqa: E402

EFFECTIVE_BATCH = 32
N_TARGETS, N_PER = 8, 8


def run_dir(tag: str) -> Path:
    return REPO_ROOT / "runs" / f"sft-qlora-{tag}"


def samples_path(tag: str) -> Path:
    return REPO_ROOT / "runs" / "phase20" / f"gate_r_{tag}_samples.jsonl"


def sample(tag: str):
    from data_curation import phase20_sampling as S
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    adapter = run_dir(tag) / "final"
    tl = list(json.loads(S.HELDOUT_TARGETS_PATH.read_text()))[:N_TARGETS]
    tok = AutoTokenizer.from_pretrained(S.BASE_MODEL)
    llm = LLM(model=S.BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048 + S.MAX_NEW_TOKENS, seed=S.SEED, enable_lora=True, max_lora_rank=64)
    params = SamplingParams(temperature=S.TEMPERATURE, top_p=S.TOP_P, max_tokens=S.MAX_NEW_TOKENS)
    lora = LoRARequest(tag, 1, str(adapter))
    out = samples_path(tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tl:
            for r in S.draw(llm, tok, params, lora, t, N_PER, tag, {"adapter": str(adapter)}):
                f.write(json.dumps(r) + "\n")
    print(f"[gate R] sampled {N_TARGETS}x{N_PER} -> {out}")


def check(tag: str) -> bool:
    from core.reward import parse_completion
    key = NEW_SFT[tag]
    gate_t = json.loads((REPO_ROOT / "results" / "phase20" / "gate_t.json").read_text())
    ds = gate_t["datasets"][f"{key}/train"]
    n, mean_full = ds["n_examples"], ds["mean_full_len"]
    rd = run_dir(tag)
    ckpts = sorted(rd.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    logs = [h for h in state["log_history"] if "num_tokens" in h]
    last = logs[-1]
    s, k = last["step"], math.ceil(n / EFFECTIVE_BATCH)
    e, r = divmod(s, k)
    seen = e * n + min(r * EFFECTIVE_BATCH, n)
    per_ex = last["num_tokens"] / seen
    rel = per_ex / mean_full - 1
    r1 = abs(rel) <= 0.05

    rows = [json.loads(l) for l in samples_path(tag).open() if l.strip()]
    parsed = 0
    for row in rows:
        try:
            parsed += parse_completion(row["completion"], row["target"]) is not None
        except Exception:
            pass
    rate = parsed / len(rows)
    r2 = rate >= 0.70 and len(rows) == N_TARGETS * N_PER
    ok = r1 and r2
    res = {"gate": "R", "tag": tag, "dataset": key, "pass": ok,
           "R1": {"pass": r1, "logged_step": s, "num_tokens_at_step": last["num_tokens"],
                  "examples_seen": seen, "tokens_per_example": per_ex,
                  "dataset_mean_full_len_incl_prompt": mean_full, "relative_diff": rel,
                  "global_step_final": state["global_step"], "checkpoint": ckpts[-1].name},
           "R2": {"pass": r2, "n": len(rows), "parse_rate": rate,
                  "median_completion_tokens": statistics.median(r["n_tokens"] for r in rows),
                  "clipped_share": sum(r["clipped"] for r in rows) / len(rows),
                  "targets": sorted({r["target"] for r in rows})},
           **provenance(require_commit=False)}
    write_json(f"gate_r_{tag}.json", res)
    print(json.dumps(res, indent=1, default=str))
    print(f"[gate R] {tag}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        notify(f"MIRA phase20 Gate R FAILED for {tag}: tokens/ex rel diff {rel:+.3f}, parse rate {rate:.2f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(NEW_SFT))
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.sample:
        sample(a.tag)
    if a.check:
        sys.exit(0 if check(a.tag) else 1)


if __name__ == "__main__":
    main()
