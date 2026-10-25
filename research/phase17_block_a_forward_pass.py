#!/usr/bin/env python
"""
research/phase17_block_a_forward_pass.py — Phase 17's REAL Block A (the
instrument, GATE G1), per docs/phases/PHASE17_PREREG.md:

  Compute log pi_hat_m(y|x) under both estimators, for every model x
  target x recipe y in U(x), then Gate G1: Spearman(log pi_hat, held-out
  empirical frequency), median over targets >= 0.5 for every model
  under the marginal estimator.

NAMING COLLISION, inherited from the pre-reg, not introduced here: the
GPU GENERATION job already run this week (data_curation/phase17_block_a_
heldout_sampling.py + scripts/run_phase17_block_a_regen.sh) was also
called "Block A" in its own docstrings, but it is NOT this. That job
filled a Block-0 raw-completions gap (sampling n=32 completions for 4
models that lacked them); THIS script is the pre-reg's actual Block A:
a forward-pass-only log-probability instrument used to GATE everything
downstream (B-F). The two are related (the generation job's outputs are
part of what THIS script reads as raw completions) but distinct blocks.

MODELS (12, not 13 -- the pre-reg's own prose says "(13)" but lists only
12 named checkpoints; this is a pre-existing miscount in the pre-reg
text, not something resolved here): Base, Corpus SFT, RS-SFT, E3, E3c,
GDPO-300, Phase12@100/200/300, E1@100/200/300. Paths read from
results/phase17/inventory_v2.json, not hardcoded.

RECIPE y = an unordered precursor SET (precset_of), per the pre-reg's
own Definitions section -- NOT a full route with operations/temperature
(that is Block G1's node definition, a different "G1"; see
phase17_g1_hodge_decomposition.py's own docstring for that collision).
U(x) is read from results/phase17/universe_v2.jsonl, UNCAPPED (the
top-30 cap is for Blocks D-G only, not Block A).

SURFACE FORMS: "one recipe is written in several ways (precursor order,
amounts)". Extracted by scanning every model's raw <think>-trace
completions for a target set once (pooled across models, matching U(x)'s
own pooled definition), grouping by (target, precset) -> Counter of the
literal ORDER-PRESERVING (formula, amount-rounded-to-2dp) sequence as
parsed. Up to the 8 most frequent forms per (target, precset) are kept;
each is re-serialized as a canonical single-line JSON array for scoring
(NOT the original raw text span -- a documented approximation: minor
whitespace/formatting differences from the literal training text are
possible, but the array content, token-for-token on formula/amount
values, is exact). The top-1-form log-prob is recorded separately too,
as a sensitivity check per the pre-reg's own instruction.

ESTIMATORS:
- Marginal (primary): for model m, target x, up to K=8 of THAT model's
  own raw completions for x serve as traces tau_k. Each trace's prefix
  is cut at its own literal '"precursors":' marker (so the <think>
  block used is real, sampled text, never reconstructed). For each
  tau_k, log pi_m(y|x,tau_k) = logsumexp over y's up-to-8 surface forms
  of logp(form | prefix=tau_k), scored via ONE prefill of tau_k's prefix
  (cached) then one short forward pass per candidate form reusing that
  cache (the pre-reg's own cost-control instruction). pi_hat_m(y|x) =
  mean_k exp(log pi_m(y|x,tau_k)); reported as its log.
  COVERAGE GAP (surfaced here, not silently absorbed): 4 models have
  NO raw-trace file for the ASTRAL target set (corpus_sft, gdpo_v3_
  ckpt300, phase12_ckpt300, e1_ckpt300 -- confirmed in inventory_v2.json,
  unresolved by the recent GPU regeneration job, which used a script
  that never persists completion text on the ASTRAL side). For these,
  the marginal estimator is reported as null for all 35 ASTRAL targets;
  Gate G1's per-model median is computed over whatever targets DO have
  marginal data (n reported explicitly), not silently padded.
- Answer-only (D-F, cost control): single fixed empty-think prefix
  ('<think>\\n</think>\\n{"precursors": ', matching research/
  distributional/dose_response_teacher_forcing.py's own FIXED_PREFIX
  convention exactly), same logsumexp-over-forms scoring, K=1.

GATE G1: per model, per target, Spearman(log pi_hat_marginal(y|x),
empirical frequency of y among the model's OWN raw samples for x that
were NOT used as one of the K tau_k). Median over targets; pass if
>= 0.5 for EVERY model. Reported for the answer-only estimator too
(sensitivity / Phase-15-empty-think-trust check), not gated on.

Cost estimate (results/phase17/block_a_forward_pass_cost_estimate.json,
written by a separate quick smoke timing, see that file for the real
measured numbers this script's design was sized against) put the full
12-model x 75-target run at single-digit GPU-hours, comfortably under
the pre-reg's 24h decision threshold.

NO EDIT to validator.py, core/*.py, evaluate_batched.py, or any frozen
scoring file -- only reads them.

Usage (tmux, GPU; smoke test first):
  PHASE17_BLOCK_A_FWD_LIMIT_TARGETS=2 PHASE17_BLOCK_A_FWD_LIMIT_MODELS=2 \
    PYTHONPATH=. uv run python research/phase17_block_a_forward_pass.py
  # full run:
  PYTHONPATH=. uv run python research/phase17_block_a_forward_pass.py
"""
from __future__ import annotations

import copy
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
from scipy.stats import spearmanr  # noqa: E402

from evaluate_batched import load_eval_model  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402
from core.reward import parse_completion  # noqa: E402
from research.support_growth_by_checkpoint import precset_of  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
UNIVERSE_PATH = REPO_ROOT / "results" / "phase17" / "universe_v2.jsonl"
INVENTORY_PATH = REPO_ROOT / "results" / "phase17" / "inventory_v2.json"
LOGP_OUT_PATH = REPO_ROOT / "results" / "phase17" / "logp_universe.jsonl"
# Gate G1's verdict (instrument_check.json, per the pre-reg's naming) is computed by the
# follow-up script research/phase17_gate_g1.py, which reads LOGP_OUT_PATH -- kept separate
# so a Gate G1 re-analysis never needs to re-run the GPU pass.

K_TRACES = 8
MAX_FORMS = 8
FIXED_ANSWER_ONLY_PREFIX = '<think>\n</think>\n{"precursors": '
PRECURSOR_MARKER = '"precursors":'

LIMIT_TARGETS = int(os.environ.get("PHASE17_BLOCK_A_FWD_LIMIT_TARGETS", "0")) or None
LIMIT_MODELS = int(os.environ.get("PHASE17_BLOCK_A_FWD_LIMIT_MODELS", "0")) or None


def closed_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def surface_form_of(route) -> tuple:
    return tuple((p.formula, round(p.amount, 2)) for p in route.precursors)


def form_to_json_array(form: tuple) -> str:
    return json.dumps([{"formula": f, "amount": a} for f, a in form])


def cut_prefix_at_precursors(text: str) -> str | None:
    idx = text.find(PRECURSOR_MARKER)
    if idx < 0:
        return None
    idx += len(PRECURSOR_MARKER)
    # advance past whitespace up to (and including) the opening bracket
    while idx < len(text) and text[idx] in " \n\t":
        idx += 1
    if idx >= len(text) or text[idx] != "[":
        return None
    return text[:idx + 1]  # include the '['


def load_universe() -> dict[str, list[tuple]]:
    by_target = defaultdict(list)
    with UNIVERSE_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            by_target[d["target"]].append(tuple(sorted(d["recipe"])))
    return by_target


def target_set_of(target: str, astral_targets: set, heldout_targets: set) -> str | None:
    if target in astral_targets:
        return "astral"
    if target in heldout_targets:
        return "heldout"
    return None


def scan_surface_forms_and_traces(inventory: dict, targets_by_set: dict[str, list[str]],
                                   node_keys_by_target: dict[str, list[tuple]]) -> tuple:
    """One pass over every raw-completions file. Returns:
    - forms_by_target[target][node_key] = Counter(surface_form) pooled over ALL models
    - traces_by_model_target[model][target] = list of (prefix_text,) up to K_TRACES,
      preserving file order (not re-shuffled -- deterministic re-runs)
    """
    forms_by_target: dict[str, dict[tuple, Counter]] = defaultdict(lambda: defaultdict(Counter))
    traces_by_model_target: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for model, by_set in inventory["raw_completions"].items():
        for target_set, info in by_set.items():
            path_str = info.get("raw_completions_file")
            if not path_str:
                continue
            path = REPO_ROOT / path_str
            if not path.exists():
                continue
            wanted_targets = set(targets_by_set.get(target_set, []))
            if not wanted_targets:
                continue
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    target = rec.get("target")
                    if target not in wanted_targets:
                        continue
                    completion = rec.get("completion", "")
                    try:
                        route = parse_completion(completion, target)
                    except Exception:
                        continue
                    try:
                        key = precset_of(route)
                    except Exception:
                        continue
                    if key not in set(node_keys_by_target.get(target, [])):
                        continue
                    forms_by_target[target][key][surface_form_of(route)] += 1
                    if len(traces_by_model_target[model][target]) < K_TRACES:
                        prefix = cut_prefix_at_precursors(completion)
                        if prefix is not None:
                            traces_by_model_target[model][target].append(prefix)
    return forms_by_target, traces_by_model_target


def score_continuations(model, tok, prefix_text: str, continuations: list[str]) -> list[float]:
    """Prefill prefix_text once (cached KV), then score each continuation's
    total log-prob (sum of per-token log p of ITS OWN tokens only, teacher-
    forced, reusing the cached prefix) -- the pre-reg's cost-control
    instruction. Returns one float per continuation, same order as input.

    CORRECTNESS NOTE (verified empirically against a from-scratch full-
    sequence forward pass before this script was trusted with real GPU
    hours -- see /tmp scratch kv_correctness_check*.py from that session):
    Qwen's tokenizer does NOT guarantee that tok(prefix) is a literal
    token-id prefix of tok(prefix + continuation) -- standalone-tokenizing
    prefix_text ending in '[' gives a DIFFERENT last token than the same
    text followed immediately by '{' (they merge into one token). Naively
    caching tok(prefix_text) and scoring tok(continuation) separately gave
    results off by 20-40 nats (checked against the correct full-sequence
    score) -- a silent, large, and otherwise undetectable error. The fix:
    tokenize prefix+continuation TOGETHER per continuation, then use only
    the LONGEST COMMON TOKEN PREFIX across all of THIS call's continuations
    (vs. the standalone prefix tokenization) as the safe cached length --
    never assume the full standalone length is safe. This is exact up to
    small bf16 split-computation noise (measured <1 nat vs a fp32 from-
    scratch reference on a representative prefix/continuation set),
    immaterial for Spearman rank correlation (Gate G1's actual use)."""
    device = model.device
    prefix_ids_standalone = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    combined_list = [tok(prefix_text + c, return_tensors="pt", add_special_tokens=False).input_ids[0]
                     for c in continuations]

    safe_plen = len(prefix_ids_standalone)
    for c in combined_list:
        m = min(safe_plen, len(c))
        i = 0
        while i < m and c[i] == prefix_ids_standalone[i]:
            i += 1
        safe_plen = min(safe_plen, i)

    prefix_ids = prefix_ids_standalone[:safe_plen].unsqueeze(0).to(device)
    with torch.no_grad():
        prefix_out = model(prefix_ids, use_cache=True)
    base_past = prefix_out.past_key_values
    full_logits_first = prefix_out.logits[:, -1, :]

    results = []
    for combined in combined_list:
        cont_ids = combined[safe_plen:].unsqueeze(0).to(device)
        with torch.no_grad():
            if cont_ids.shape[1] > 1:
                out = model(cont_ids[:, :-1], past_key_values=copy.deepcopy(base_past), use_cache=True)
                logits = torch.cat([full_logits_first.unsqueeze(1), out.logits], dim=1)
            else:
                logits = full_logits_first.unsqueeze(1)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_lp = logprobs[0, torch.arange(cont_ids.shape[1]), cont_ids[0]]
        results.append(float(token_lp.sum().item()))
    return results


def logsumexp(vals: list[float]) -> float:
    if not vals:
        return float("-inf")
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


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

    # cache top-MAX_FORMS forms per (target, node_key), as (form, json_text) list
    top_forms: dict[str, dict[tuple, list[tuple]]] = {}
    for t in all_targets:
        top_forms[t] = {}
        for key in node_keys_by_target[t]:
            counter = forms_by_target[t].get(key, Counter())
            best = counter.most_common(MAX_FORMS)
            if not best:
                continue
            top_forms[t][key] = [(form, form_to_json_array(form)) for form, _ in best]

    models_list = list(inventory["models"].keys())
    if LIMIT_MODELS:
        models_list = models_list[:LIMIT_MODELS]

    LOGP_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_model_targets = set()
    if LOGP_OUT_PATH.exists():
        with LOGP_OUT_PATH.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_model_targets.add((r["model"], r["target"]))
        print(f"resuming: {len(done_model_targets)} (model,target) rows already done", flush=True)
    out_f = LOGP_OUT_PATH.open("a")
    n_rows_written = 0

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
            cont_index = []  # (node_key, form_idx) aligned with all_conts
            for key in node_keys:
                for form_idx, (_form, json_text) in enumerate(top_forms[target][key]):
                    all_conts.append(json_text)
                    cont_index.append((key, form_idx))

            # --- answer-only estimator (K=1, fixed empty-think prefix) ---
            ao_prefix = prompt + FIXED_ANSWER_ONLY_PREFIX
            ao_scores = score_continuations(model, tok, ao_prefix, all_conts)
            ao_logp_by_key: dict[tuple, list[float]] = defaultdict(list)
            for (key, _fi), sc in zip(cont_index, ao_scores):
                ao_logp_by_key[key].append(sc)
            ao_logp = {key: logsumexp(v) for key, v in ao_logp_by_key.items()}

            # --- marginal estimator (K<=8 of the MODEL's own real traces) ---
            traces = traces_by_model_target.get(model_name, {}).get(target, [])
            marg_logp = {}
            if traces:
                per_trace_logp = []  # list of {key: logsumexp_over_forms}
                for trace_prefix in traces:
                    full_prefix = prompt + trace_prefix
                    scores = score_continuations(model, tok, full_prefix, all_conts)
                    by_key = defaultdict(list)
                    for (key, _fi), sc in zip(cont_index, scores):
                        by_key[key].append(sc)
                    per_trace_logp.append({key: logsumexp(v) for key, v in by_key.items()})
                for key in node_keys:
                    probs = [math.exp(d[key]) for d in per_trace_logp if key in d]
                    if probs:
                        marg_logp[key] = math.log(sum(probs) / len(probs)) if sum(probs) > 0 else float("-inf")

            for key in node_keys:
                row = {"model": model_name, "target": target, "target_set": tset,
                      "recipe": list(key), "answer_only_logp": round(ao_logp.get(key, float("-inf")), 4),
                      "marginal_logp": round(marg_logp[key], 4) if key in marg_logp else None,
                      "n_traces_used": len(traces), "n_forms": len(top_forms[target][key]),
                      "top1_form_answer_only_logp": round(ao_logp_by_key[key][0], 4) if ao_logp_by_key.get(key) else None}
                out_f.write(json.dumps(row) + "\n")
                n_rows_written += 1
            out_f.flush()

            print(f"  {target} ({tset}): {len(node_keys)} recipes, "
                  f"{len(traces)} traces, {len(all_conts)} continuations scored", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"=== {model_name} DONE, GPU freed ===", flush=True)

    out_f.close()
    print(f"\n-> {LOGP_OUT_PATH} ({n_rows_written} rows)")
    print("Run research/phase17_gate_g1.py next to compute Gate G1's verdict from this file.")


if __name__ == "__main__":
    main()
