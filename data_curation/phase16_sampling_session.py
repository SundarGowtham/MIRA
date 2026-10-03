#!/usr/bin/env python
"""
data_curation/phase16_sampling_session.py — Phase 16 SCOPE FREEZE launch
(misc/PHASE16_INSTRUCTIONS.md). One tmux session, one vLLM load of base
Qwen3-8B, three phases in order:

  0. Smoke run: 2 targets x 4 samples. Checks parse rate, mean completion
     length, and clipped-ratio against Phase 12's own (mean 9.7%, range
     7.25-12.25%, pulled directly from W&B run dpa0fqvb). Abort (write to
     docs/phases/PHASE16_RESULTS.md, ntfy, exit nonzero) if abnormal.
  1. Full E3/E3c pool: data/novelty/training_targets.json (360 targets) x
     64 samples -> data/novelty/round1_samples.jsonl. One sampling run
     feeds both E3 (novelty-filtered) and E3c (matched-size control) --
     they filter the SAME pool differently, not two separate draws.
  2. Base reference samples the pre-registration needs for the
     support-growth test: 200 (reference) + 32 (independent resampling-
     rate draw) per target, on data/novelty/heldout_targets.json (40) and
     on ASTRAL's 35 targets (results/astral_validation_set.json) --
     ASTRAL targets appear here ONLY as evaluation prompts for base's own
     reference distribution, never as training data (no circularity).

Crash-safety: each target's samples are appended and flushed immediately
after its own generate() call returns, so a crash loses at most one
target's samples, never a whole phase. ntfy at start (with a real,
throughput-based ETA from the smoke run), on any crash, and at
completion.

Environment guard: refuses to run outside mira-vllm (Phase 16
"Decisions after pass 2", the mis-install correction).

Usage (tmux, mira-vllm) -- must be `source`-activated, not invoked by
bare binary path: flashinfer JIT-compiles sampling kernels at startup via
a subprocess call to `ninja` on PATH, and invoking
`~/envs/mira-vllm/bin/python script.py` directly does NOT prepend the
venv's bin/ to PATH the way `source activate` does (confirmed the hard
way during the Sec 2.2a standalone smoke test and again here):

  source ~/envs/mira-vllm/bin/activate && python data_curation/phase16_sampling_session.py
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

MODEL = "Qwen/Qwen3-8B"
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 8192

TRAINING_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "training_targets.json"
HELDOUT_TARGETS_PATH = REPO_ROOT / "data" / "novelty" / "heldout_targets.json"
ASTRAL_PATH = REPO_ROOT / "results" / "astral_validation_set.json"

POOL_OUT = REPO_ROOT / "data" / "novelty" / "round1_samples.jsonl"
HELDOUT_REF_OUT = REPO_ROOT / "data" / "novelty" / "base_reference_heldout.jsonl"
ASTRAL_REF_OUT = REPO_ROOT / "data" / "novelty" / "base_reference_astral.jsonl"

POOL_SAMPLES_PER_TARGET = 64
REF_SAMPLES_PER_TARGET = 200
REF_RESAMPLE_PER_TARGET = 32

# Phase 12's own clipped ratio, pulled directly from W&B run dpa0fqvb
# (gdpo-qlora-gdpo-phase12-rssft-beta0), 13 logged points over the full
# 300-step run -- not assumed, read via the wandb API before writing this
# script.
PHASE12_CLIPPED_RATIO_MEAN = 0.0971
PHASE12_CLIPPED_RATIO_RANGE = (0.0725, 0.1225)
PHASE12_MEAN_TERMINATED_LENGTH = 4960


def _assert_mira_vllm_environment() -> None:
    actual = Path(sys.prefix).resolve()
    if actual != EXPECTED_ENV:
        raise RuntimeError(
            f"phase16_sampling_session.py refuses to run outside mira-vllm. "
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


def sample_target(llm, tok, sampling_params, target: str, n: int) -> list[dict]:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": closed_prompt(target)}],
        tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([rendered] * n, sampling_params)
    records = []
    for out in outputs:
        text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        records.append({
            "target": target, "completion": text, "n_tokens": n_tokens,
            "clipped": n_tokens >= MAX_NEW_TOKENS,
        })
    return records


def append_records(path: Path, records: list[dict], extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in records:
            row = {**r, **(extra or {})}
            f.write(json.dumps(row) + "\n")


def run_smoke(llm, tok, sampling_params) -> tuple[bool, dict, float]:
    targets = json.loads(TRAINING_TARGETS_PATH.read_text())[:2]
    t0 = time.time()
    all_records = []
    for target in targets:
        all_records.extend(sample_target(llm, tok, sampling_params, target, 4))
    elapsed = time.time() - t0

    n_total = len(all_records)
    n_parsed = 0
    from core.reward import parse_completion, ParseFailure
    for r in all_records:
        try:
            parse_completion(r["completion"], r["target"])
            n_parsed += 1
        except ParseFailure:
            pass
        except Exception:
            pass
    parse_rate = n_parsed / n_total if n_total else 0.0
    mean_length = sum(r["n_tokens"] for r in all_records) / n_total if n_total else 0.0
    clip_rate = sum(1 for r in all_records if r["clipped"]) / n_total if n_total else 0.0
    completions_per_sec = n_total / elapsed if elapsed > 0 else 0.0

    stats = {
        "n_total": n_total, "parse_rate": round(parse_rate, 3),
        "mean_length": round(mean_length, 1), "clip_rate": round(clip_rate, 3),
        "elapsed_sec": round(elapsed, 1), "completions_per_sec": round(completions_per_sec, 4),
    }

    # Abnormal if: parse rate is low (routes don't even parse), mean length
    # is wildly outside Phase 12's own range (e.g. near-zero or pegged at
    # the cap), or clip rate is far outside Phase 12's observed range.
    # "Far outside" = outside even a generous 3x band around Phase 12's
    # own min/max -- a real abnormality, not noise at n=8.
    lo, hi = PHASE12_CLIPPED_RATIO_RANGE
    clip_band = (max(0.0, lo - 0.15), min(1.0, hi + 0.15))
    ok = (parse_rate >= 0.5
          and 200 <= mean_length <= MAX_NEW_TOKENS
          and clip_band[0] <= clip_rate <= clip_band[1])
    return ok, stats, completions_per_sec


def run_pool(llm, tok, sampling_params) -> None:
    targets = json.loads(TRAINING_TARGETS_PATH.read_text())
    done = set()
    if POOL_OUT.exists():
        with POOL_OUT.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["target"])
    print(f"pool: {len(targets)} targets, {len(done)} already done", flush=True)
    for i, target in enumerate(targets):
        if target in done:
            continue
        records = sample_target(llm, tok, sampling_params, target, POOL_SAMPLES_PER_TARGET)
        append_records(POOL_OUT, records, extra={"round": 1, "phase": "pool"})
        if (i + 1) % 20 == 0:
            print(f"  pool: {i + 1}/{len(targets)} targets done", flush=True)
    print("pool: DONE", flush=True)


def run_reference(llm, tok, sampling_params, targets: list[str], out_path: Path, label: str) -> None:
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
        ref_records = sample_target(llm, tok, sampling_params, target, REF_SAMPLES_PER_TARGET)
        append_records(out_path, ref_records, extra={"sample_group": "reference"})
        resample_records = sample_target(llm, tok, sampling_params, target, REF_RESAMPLE_PER_TARGET)
        append_records(out_path, resample_records, extra={"sample_group": "resampling_rate"})
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i + 1}/{len(targets)} targets done", flush=True)
    print(f"{label}: DONE", flush=True)


def main():
    _assert_mira_vllm_environment()
    notify("Phase 16 sampling session: loading base Qwen3-8B via vLLM...")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
             max_model_len=2048 + MAX_NEW_TOKENS, seed=42)
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)

    print("=== Phase 0: smoke run (2 targets x 4 samples) ===", flush=True)
    ok, stats, rate = run_smoke(llm, tok, sampling_params)
    print(f"smoke stats: {stats}", flush=True)
    if not ok:
        msg = (f"Phase 16 sampling smoke run ABNORMAL: {stats}. "
              f"Phase 12 reference: clipped_ratio mean={PHASE12_CLIPPED_RATIO_MEAN} "
              f"range={PHASE12_CLIPPED_RATIO_RANGE}, mean_length~{PHASE12_MEAN_TERMINATED_LENGTH}. "
              f"Stopping per instruction.")
        notify(msg)
        results_md = REPO_ROOT / "docs" / "phases" / "PHASE16_RESULTS.md"
        with results_md.open("a") as f:
            f.write(f"\n## Sampling session smoke run ABNORMAL\n\n{msg}\n")
        sys.exit(1)

    n_remaining = (360 * POOL_SAMPLES_PER_TARGET
                  + 40 * (REF_SAMPLES_PER_TARGET + REF_RESAMPLE_PER_TARGET)
                  + 35 * (REF_SAMPLES_PER_TARGET + REF_RESAMPLE_PER_TARGET))
    eta_min = (n_remaining / rate / 60) if rate > 0 else float("nan")
    notify(f"Phase 16 sampling smoke run OK ({stats}). Launching full run: "
          f"~{n_remaining} completions remaining, throughput {rate:.3f}/sec, "
          f"ETA ~{eta_min:.0f} min.")

    print("=== Phase 1: E3/E3c pool (360 targets x 64) ===", flush=True)
    run_pool(llm, tok, sampling_params)

    print("=== Phase 2a: base reference, held-out (40 targets x 232) ===", flush=True)
    heldout_targets = json.loads(HELDOUT_TARGETS_PATH.read_text())
    run_reference(llm, tok, sampling_params, heldout_targets, HELDOUT_REF_OUT, "heldout_ref")

    print("=== Phase 2b: base reference, ASTRAL (35 targets x 232) ===", flush=True)
    astral_targets = [t["target"] for t in json.loads(ASTRAL_PATH.read_text())["targets"]]
    run_reference(llm, tok, sampling_params, astral_targets, ASTRAL_REF_OUT, "astral_ref")

    notify("Phase 16 sampling session COMPLETE: pool + held-out reference + ASTRAL reference all done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        notify(f"Phase 16 sampling session CRASHED:\n{tb[-1500:]}")
        raise
