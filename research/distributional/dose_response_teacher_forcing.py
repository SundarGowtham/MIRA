#!/usr/bin/env python
"""
research/distributional/dose_response_teacher_forcing.py — Phase 15 Task 5c.
Pre-registration: docs/phases/PHASE15_DOSE_PREREG.md, committed before
this script ran. GPU, forward passes only, no sampling.

For each of 4 checkpoints (base, RS-SFT, GDPO-300, full SFT) and each of
75 ARROWS routes: teacher-force closed_prompt(target) + a fixed
'<think>\n</think>\n{"precursors": ' prefix + the exact JSON precursor
array, and record total log p / per-token log p of the array span only
(never the shared prefix), plus the top-20 tokens at the first
precursor's first formula token.

Usage (tmux, GPU):
  uv run python research/distributional/dose_response_teacher_forcing.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402

ARROWS_DIR = Path("data/external/arrows/ARROWS/Examples")
OUT_DIR = Path("results/dose_response")
CANONICAL_TARGET = {"YBCO": "YBa2Cu3O7", "LTOPO": "LiTiOPO4", "NTMO": "Na2Te3Mo3O16"}
CARBONATES = {"Li2CO3", "Na2CO3", "K2CO3", "BaCO3", "SrCO3", "CaCO3", "MgCO3"}

CHECKPOINTS = {
    "base": {"checkpoint": "base", "model": "Qwen/Qwen3-8B"},
    "rs_sft": {"checkpoint": "runs/sft-qlora-rs-sft-from-base/final", "model": "Qwen/Qwen3-8B"},
    "gdpo_300": {"checkpoint": "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300", "model": "Qwen/Qwen3-8B"},
    "full_sft": {"checkpoint": "runs/sft-qlora-sft-v3-2nd-rank16/final", "model": "Qwen/Qwen3-8B"},
}

FIXED_PREFIX = '<think>\n</think>\n{"precursors": '


def closed_prompt(target: str) -> str:
    return (SYSTEM_MSG + "\n\nTarget: " + target +
            "\n\nProvide your synthesis route as a JSON object.")


def load_target_entries_with_best(name: str):
    """Same route construction as Task 4, but here we only need
    (precursor_set, precursors, amounts, best_wt_pct) -- one row per set,
    using its max wt% across temperatures (per the pre-reg)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import arrows_gate_scoring as g
    entries, target_formula = g.load_target_entries(name)
    best_by_set = {}
    for e in entries:
        key = e["precursor_set"]
        if key not in best_by_set or e["wt_pct"] > best_by_set[key]["wt_pct"]:
            best_by_set[key] = e
    return list(best_by_set.values()), target_formula


def build_routes():
    routes = []
    for name in ["YBCO", "LTOPO", "NTMO"]:
        entries, target_formula = load_target_entries_with_best(name)
        for e in entries:
            routes.append({
                "arrows_target": name, "target_formula": target_formula,
                "precursor_set": e["precursor_set"], "precursors": e["precursors"],
                "amounts": e["amounts"], "best_wt_pct": e["wt_pct"],
                "has_carbonate": any(p in CARBONATES for p in e["precursors"]),
            })
    return routes


def precursor_json_array(precursors: list[str], amounts: list[float]) -> str:
    arr = [{"formula": p, "amount": round(float(a), 4)} for p, a in zip(precursors, amounts)]
    return json.dumps(arr)


def load_checkpoint(name: str):
    cfg = CHECKPOINTS[name]
    checkpoint, model_name = cfg["checkpoint"], cfg["model"]
    if checkpoint == "base":
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    else:
        ckpt_path = Path(checkpoint)
        tok = AutoTokenizer.from_pretrained(ckpt_path)
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, str(ckpt_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def teacher_force_route(model, tok, route: dict) -> dict:
    prompt_text = closed_prompt(route["target_formula"])
    rendered_prompt = tok.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
    array_json = precursor_json_array(route["precursors"], route["amounts"])
    full_text = rendered_prompt + FIXED_PREFIX + array_json

    enc = tok(full_text, return_tensors="pt", return_offsets_mapping=True).to(model.device)
    offsets = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"]
    del enc["offset_mapping"]

    # character span of the array within full_text
    array_start_char = len(rendered_prompt) + len(FIXED_PREFIX)
    array_end_char = array_start_char + len(array_json)

    out = model(**enc)
    logits = out.logits[0]  # (seq_len, vocab)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # token i's logit predicts token i+1. Find which token INDICES (i+1)
    # fall inside the array's character span -- their predicting logit is
    # at position i = idx-1.
    array_token_positions = []  # (predicting_logit_idx, predicted_token_id, char_start)
    n_tokens = input_ids.shape[1]
    for idx in range(1, n_tokens):
        char_start, char_end = offsets[idx]
        if char_start >= array_start_char and char_end <= array_end_char and char_end > char_start:
            array_token_positions.append((idx - 1, input_ids[0, idx].item(), char_start))

    per_token_logp = []
    for pred_idx, tok_id, char_start in array_token_positions:
        lp = log_probs[pred_idx, tok_id].item()
        per_token_logp.append(lp)
    total_logp = sum(per_token_logp)

    # first precursor's first formula token: find char offset of the
    # first formula string's first character (right after {"formula": ")
    first_formula_marker = '{"formula": "'
    first_formula_char = full_text.find(first_formula_marker, array_start_char) + len(first_formula_marker)
    top20 = None
    for idx in range(1, n_tokens):
        char_start, char_end = offsets[idx]
        if char_start == first_formula_char:
            pred_idx = idx - 1
            topk = torch.topk(log_probs[pred_idx], k=20)
            top20 = [{"token": tok.decode([tid.item()]), "token_id": tid.item(), "logp": lp.item()}
                    for lp, tid in zip(topk.values, topk.indices)]
            break

    return {
        "total_logp": total_logp, "n_tokens": len(per_token_logp),
        "per_token_logp": per_token_logp, "top20_first_token": top20,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = build_routes()
    print(f"built {len(routes)} routes", flush=True)

    all_results = {r["precursor_set"] + "@" + r["arrows_target"]: {"route": r, "checkpoints": {}}
                  for r in routes}

    for ckpt_name in ["base", "rs_sft", "gdpo_300", "full_sft"]:
        print(f"\nloading checkpoint: {ckpt_name}...", flush=True)
        model, tok = load_checkpoint(ckpt_name)
        for i, route in enumerate(routes):
            key = route["precursor_set"] + "@" + route["arrows_target"]
            result = teacher_force_route(model, tok, route)
            all_results[key]["checkpoints"][ckpt_name] = result
            if (i + 1) % 20 == 0:
                print(f"  {ckpt_name}: {i + 1}/{len(routes)} routes done", flush=True)
        unload(model)
        print(f"  {ckpt_name}: DONE, GPU freed", flush=True)

    OUT_PATH = OUT_DIR / "teacher_forcing_raw.json"
    OUT_PATH.write_text(json.dumps(list(all_results.values()), indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
