#!/usr/bin/env python
"""
research/phase17_b1_split_trace_forward_pass.py — errors-in-variables
control for Block B1 (regular Claude, 2026-10-27): per-trace
log pi_m(y|x,tau_k) values are NOT stored anywhere (phase17_block_a_
forward_pass.py's main() computes them in a local per_trace_logp list,
then collapses to the single K-trace-averaged marginal_logp before
writing a row -- the per-trace values are discarded). Without them,
B1's regression cannot be split 4-traces/4-traces to check whether its
negative c_s for the practice steps (S5/S6/S7) is a real effect or
classic errors-in-variables regression-to-the-mean (noisy log pi_before
appearing with opposite sign on both sides of Delta = log pi_after -
log pi_before).

This reruns Block A's exact forward-pass logic (scan_surface_forms_and_
traces, score_continuations, closed_prompt, etc. -- ALL imported from
the frozen research/phase17_block_a_forward_pass.py, not duplicated or
edited) for ONLY the three BEFORE models B1 actually needs (base,
corpus_sft, rs_sft -- base->* covers S1-S4, corpus_sft is S5's before,
rs_sft is S6/S7's before), and writes the 8 individual per-trace
logsumexp-over-forms values per (model,target,recipe), not just their
average.

Cost estimated empirically (see run_logs/phase17_b1_split_trace_cost_
estimate.txt) before the full run: ~1/4 of Block A's original 12-model,
~6h16m run (same per-sample GPU compute, just also persisting the 8
per-trace values instead of collapsing them) -- approved by regular
Claude's own instruction if <= 12h.

Output: results/phase17/logp_per_trace_before_models.jsonl
  {model, target, target_set, recipe, per_trace_logp: [v1..v8 or fewer],
   n_traces_used}

NO EDIT to phase17_block_a_forward_pass.py.

Usage (tmux, GPU):
  PYTHONPATH=. uv run python research/phase17_b1_split_trace_forward_pass.py
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

import torch  # noqa: E402

from evaluate_batched import load_eval_model  # noqa: E402
from research.phase17_block_a_forward_pass import (  # noqa: E402
    closed_prompt, form_to_json_array, load_universe, scan_surface_forms_and_traces,
    score_continuations, logsumexp, INVENTORY_PATH, K_TRACES, MAX_FORMS,
)

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "logp_per_trace_before_models.jsonl"
MODELS = ["base", "corpus_sft", "rs_sft"]

LIMIT_TARGETS = int(os.environ.get("PHASE17_SPLIT_TRACE_LIMIT_TARGETS", "0")) or None
LIMIT_MODELS_FOR_SMOKE = os.environ.get("PHASE17_SPLIT_TRACE_SMOKE_MODEL")


def main():
    inventory = json.loads(INVENTORY_PATH.read_text())
    universe = load_universe()

    astral_doc = json.loads((REPO_ROOT / "misc" / "astral_validation_set.json").read_text())
    astral_targets = {t["target"] for t in astral_doc["targets"]}
    heldout_targets = set(json.loads((REPO_ROOT / "data" / "novelty" / "heldout_targets.json").read_text()))

    all_targets = sorted(universe.keys())
    if LIMIT_TARGETS:
        all_targets = all_targets[:LIMIT_TARGETS]
    targets_by_set = {"astral": [t for t in all_targets if t in astral_targets],
                      "heldout": [t for t in all_targets if t in heldout_targets]}
    print(f"targets: {len(all_targets)} ({len(targets_by_set['astral'])} astral, "
          f"{len(targets_by_set['heldout'])} heldout)", flush=True)

    node_keys_by_target = {t: universe[t] for t in all_targets}

    print("scanning raw completions for surface forms + traces (one pass, CPU)...", flush=True)
    t0 = time.time()
    forms_by_target, traces_by_model_target = scan_surface_forms_and_traces(
        inventory, targets_by_set, node_keys_by_target)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    top_forms: dict[str, dict[tuple, list[tuple]]] = {}
    for t in all_targets:
        top_forms[t] = {}
        for key in node_keys_by_target[t]:
            counter = forms_by_target[t].get(key, Counter())
            best = counter.most_common(MAX_FORMS)
            if not best:
                continue
            top_forms[t][key] = [(form, form_to_json_array(form)) for form, _ in best]

    models_list = [LIMIT_MODELS_FOR_SMOKE] if LIMIT_MODELS_FOR_SMOKE else MODELS

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_model_targets = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_model_targets.add((r["model"], r["target"]))
        print(f"resuming: {len(done_model_targets)} (model,target) rows already done", flush=True)
    out_f = OUT_PATH.open("a")
    n_rows_written = 0
    t_gpu_start = time.time()

    for model_name in models_list:
        minfo = inventory["models"][model_name]
        checkpoint = minfo["adapter_path"] or "base"
        print(f"\n=== loading {model_name} ({checkpoint}) ===", flush=True)
        tm0 = time.time()
        model, tok = load_eval_model(checkpoint=checkpoint, model_name="Qwen/Qwen3-8B")
        model.eval()
        print(f"  loaded in {time.time()-tm0:.1f}s", flush=True)

        for target in all_targets:
            if (model_name, target) in done_model_targets:
                continue
            tset = "astral" if target in astral_targets else "heldout"
            prompt = closed_prompt(target)
            node_keys = [k for k in node_keys_by_target[target] if k in top_forms[target]]
            if not node_keys:
                continue
            all_conts = []
            cont_index = []
            for key in node_keys:
                for form_idx, (_form, json_text) in enumerate(top_forms[target][key]):
                    all_conts.append(json_text)
                    cont_index.append((key, form_idx))

            traces = traces_by_model_target.get(model_name, {}).get(target, [])
            per_trace_logp_by_key: dict[tuple, list[float]] = defaultdict(list)
            for trace_prefix in traces:
                full_prefix = prompt + trace_prefix
                scores = score_continuations(model, tok, full_prefix, all_conts)
                by_key = defaultdict(list)
                for (key, _fi), sc in zip(cont_index, scores):
                    by_key[key].append(sc)
                for key in node_keys:
                    if key in by_key:
                        per_trace_logp_by_key[key].append(logsumexp(by_key[key]))

            for key in node_keys:
                row = {"model": model_name, "target": target, "target_set": tset,
                      "recipe": list(key),
                      "per_trace_logp": [round(v, 4) for v in per_trace_logp_by_key.get(key, [])],
                      "n_traces_used": len(traces)}
                out_f.write(json.dumps(row) + "\n")
                n_rows_written += 1
            out_f.flush()

            print(f"  {target} ({tset}): {len(node_keys)} recipes, "
                  f"{len(traces)} traces, {len(all_conts)} continuations scored, "
                  f"elapsed={time.time()-t_gpu_start:.0f}s", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"=== {model_name} DONE, GPU freed ===", flush=True)

    out_f.close()
    print(f"\n-> {OUT_PATH} ({n_rows_written} rows)")


if __name__ == "__main__":
    main()
