"""
analyze_pd_attention.py
-----------------------
Mechanistic check of the closed-book finding: does the policy actually READ
the PD stability block in open-book prompts? Measures attention mass
directed onto the stability-data token span vs. its share of the sequence,
for a few open-book prompts through the SFT checkpoint.

Ratio mass/share << 1  -> the block is under-attended (behavioral null
confirmed mechanistically). Ratio ~1 -> attended but unused (stranger).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.reward import load_validator  # noqa: E402
from stratified_difficulty_eval import (  # noqa: E402
    SYSTEM_MSG, CLOSED_BOOK_USER, get_stability_data_sync,
)

MARK_BEGIN = "Thermodynamic Context (Phase Stability Data):\n"
MARK_END = "\n\nProvide your synthesis route as a JSON object."
OUT = Path("misc/pd_attention.json")
N_PROMPTS = 5


def load_eager(ckpt: str):
    """Self-contained loader with attn_implementation='eager' — sdpa does
    NOT return attention tensors, and load_eval_model doesn't expose the
    knob. Returns (model, tok)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16,
        attn_implementation="eager", device_map="auto")
    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    return model, tok


def main():
    targets = []
    for l in open("data/rl/val.jsonl"):
        targets.append(json.loads(l)["target"])
        if len(targets) >= N_PROMPTS:
            break

    model, tok = load_eager("runs/sft-qlora-sft-v3-2nd-rank16/final")
    validator = load_validator(Path("data/cache/mp_formula_set.pkl"),
                               Path("data/cache/pd_index.json"), Path("."))

    per_prompt = []
    for target in targets:
        stab, _ = get_stability_data_sync(target, validator)
        prompt = SYSTEM_MSG + "\n\n" + CLOSED_BOOK_USER.format(
            target=target, context="", stability_data=stab)

        enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True).to(model.device)
        offsets = enc.pop("offset_mapping")[0].tolist()

        b = prompt.index(MARK_BEGIN) + len(MARK_BEGIN)
        e = prompt.index(MARK_END)
        pd_tokens = [i for i, (s, t) in enumerate(offsets) if t > b and s < e]
        if not pd_tokens:
            print(f"  warn: no PD span found for {target}", file=sys.stderr)
            continue
        pd_lo, pd_hi = min(pd_tokens), max(pd_tokens)
        seq_len = len(offsets)

        with torch.no_grad():
            out = model(**enc, output_attentions=True)

        # mean attention mass onto PD-span keys, per query row after the span,
        # averaged over heads and layers. Rows are softmax-normalized already.
        masses = []
        for att in out.attentions:  # [1, heads, seq, seq]
            m = att[0, :, pd_lo:, pd_lo:pd_hi + 1].sum(dim=-1).mean().item()
            masses.append(m)
        share = len(pd_tokens) / seq_len
        mean_mass = sum(masses) / len(masses)
        # late layers carry the "read for use" signal; report last-quarter too
        late = masses[3 * len(masses) // 4:]
        late_mass = sum(late) / len(late)
        per_prompt.append({
            "target": target,
            "seq_len": seq_len,
            "pd_tokens": len(pd_tokens),
            "pd_share": round(share, 4),
            "attn_mass_all_layers": round(mean_mass, 4),
            "attn_mass_late_layers": round(late_mass, 4),
            "ratio_all": round(mean_mass / share, 3),
            "ratio_late": round(late_mass / share, 3),
        })
        print(f"  {target}: share={share:.2%} mass={mean_mass:.2%} "
              f"late={late_mass:.2%} ratio={mean_mass/share:.2f}", flush=True)
        del out, enc
        torch.cuda.empty_cache()

    OUT.write_text(json.dumps({"per_prompt": per_prompt}, indent=2))
    r_all = sum(p["ratio_all"] for p in per_prompt) / len(per_prompt)
    r_late = sum(p["ratio_late"] for p in per_prompt) / len(per_prompt)
    print(f"\n=== PD ATTENTION ===")
    print(f"mean mass/share ratio (all layers):  {r_all:.2f}")
    print(f"mean mass/share ratio (late layers): {r_late:.2f}")
    print("(1.0 = attended proportional to size; <<1 = under-attended)")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
