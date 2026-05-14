"""
generate_reasoning_traces_hf.py
-------------------------------
HuggingFace transformers-based reasoning trace generator.


Produces JSONL output in the IDENTICAL schema to the vLLM/Ollama versions,
so you can swap to vLLM later without rework.

Throughput estimate (3090, Qwen3-14B 4-bit, batch=4): ~30-60 records/hour.
That's ~150-300 records in a 5-hour run.

Usage:
    # Smoke test on 4 records
    uv run python -u generate_reasoning_traces.py --limit 4 --batch-size 2
    uv run python -u generate_reasoning_traces_hf.py --limit 4 --batch-size 2

    # Background run
    nohup python -u generate_reasoning_traces_hf.py 2>&1 | tee logs/hf_traces.log &
    disown

To switch to vLLM later:
    1. Kill this script
    2. Run generate_reasoning_traces.py — it skips records already in the JSONL.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Prompts (IDENTICAL across all generator scripts)
# ---------------------------------------------------------------------------

CLOSED_BOOK_PROMPT = """You are a materials chemist. Design a solid-state synthesis route for the following target compound.

TARGET: {target}{context}

Reason carefully about:
1. The target's stoichiometry and oxidation states
2. Suitable precursor compounds (real, commercially available)
3. The exact molar ratio of precursors required to balance the equation
4. Reaction operations and conditions (temperature, time, atmosphere)

After your reasoning, output the final route in this exact format:

<precursors>
- formula | amount
- formula | amount
</precursors>

<operations>
1. operation_type | conditions
2. operation_type | conditions
</operations>

Use these operation types: StartingSynthesis, MixingOperation, DryingOperation, HeatingOperation, ShapingOperation, QuenchingOperation.
Conditions can include: T=...°C, t=...h, atm=..., media=...
"""

OPEN_BOOK_PROMPT = """You are a materials chemist. The following is a verified solid-state synthesis route from the published literature.

TARGET: {target}{context}

VERIFIED ROUTE:
{route_block}

Reconstruct the chemistry reasoning that leads to this route. Reason carefully about:
1. The target's stoichiometry and oxidation states
2. Why these specific precursors were chosen
3. How the molar ratios balance the reaction equation
4. Why these temperatures, times, and atmosphere are appropriate

After your reasoning, output the route in this exact format:

<precursors>
- formula | amount
- formula | amount
</precursors>

<operations>
1. operation_type | conditions
2. operation_type | conditions
</operations>
"""


# ---------------------------------------------------------------------------
# MP record loading
# ---------------------------------------------------------------------------

def load_mp_records(path: Path) -> list[dict]:
    from monty.serialization import loadfn
    return loadfn(path)


def load_summary(path: Path) -> dict[str, dict]:
    from monty.serialization import loadfn
    from pymatgen.core import Composition
    if not path.exists():
        return {}
    summary = loadfn(path)
    out = {}
    for s in summary:
        try:
            key = Composition(s["formula_pretty"]).reduced_formula
            out[key] = s
        except Exception:
            pass
    return out


def format_target_context(target: str, summary_by_formula: dict) -> str:
    from pymatgen.core import Composition
    try:
        key = Composition(target).reduced_formula
    except Exception:
        return ""
    s = summary_by_formula.get(key)
    if not s:
        return ""
    parts = []
    if s.get("crystal_system"):
        parts.append(f"crystal system: {s['crystal_system']}")
    if s.get("spacegroup_number"):
        parts.append(f"space group: {s['spacegroup_number']}")
    if s.get("band_gap") is not None:
        bg = s["band_gap"]
        parts.append(f"band gap: {bg:.2f} eV" if bg > 0 else "metallic")
    if not parts:
        return ""
    return "\nKnown properties: " + ", ".join(parts)


def format_route_block(precursors: list[dict], operations: list[dict]) -> str:
    prec_lines = []
    for p in precursors:
        formula = p.get("formula", "")
        amount = p.get("amount", 1.0)
        prec_lines.append(f"- {formula} | {amount}")
    op_lines = []
    for i, op in enumerate(operations, 1):
        op_type = op.get("type", "Unknown")
        cond_parts = []
        temps = op.get("heating_temperature") or []
        flat_temps = [t for sub in temps for t in (sub if isinstance(sub, list) else [sub])]
        if flat_temps:
            cond_parts.append(f"T={sum(flat_temps)/len(flat_temps):.0f}°C")
        times = op.get("heating_time") or []
        flat_times = [t for sub in times for t in (sub if isinstance(sub, list) else [sub])]
        if flat_times:
            cond_parts.append(f"t={sum(flat_times)/len(flat_times):.1f}h")
        atm = op.get("heating_atmosphere") or []
        if atm:
            cond_parts.append(f"atm={','.join(atm)}")
        cond_str = ", ".join(cond_parts) if cond_parts else "none"
        op_lines.append(f"{i}. {op_type} | {cond_str}")
    return (
        "<precursors>\n" + "\n".join(prec_lines) + "\n</precursors>\n\n"
        "<operations>\n" + "\n".join(op_lines) + "\n</operations>"
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

PRECURSORS_RE = re.compile(r"<precursors>(.*?)</precursors>", re.DOTALL)
OPERATIONS_RE = re.compile(r"<operations>(.*?)</operations>", re.DOTALL)
TEMP_RE = re.compile(r"T=([0-9.]+)")
TIME_RE = re.compile(r"t=([0-9.]+)")
ATM_RE = re.compile(r"atm=([^,|]+)")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking_and_output(text: str) -> tuple[str, str]:
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text.strip()


def parse_route(text: str) -> tuple[list[dict], list[dict]]:
    precursors = []
    operations = []
    pm = PRECURSORS_RE.search(text)
    if pm:
        for line in pm.group(1).strip().splitlines():
            line = line.strip().lstrip("-").strip()
            if not line or "|" not in line:
                continue
            try:
                formula, amount = [p.strip() for p in line.split("|", 1)]
                precursors.append({"formula": formula, "amount": float(amount)})
            except (ValueError, IndexError):
                continue
    om = OPERATIONS_RE.search(text)
    if om:
        for line in om.group(1).strip().splitlines():
            line = re.sub(r"^\d+\.\s*", "", line.strip())
            if not line or "|" not in line:
                continue
            op_type, _, cond_str = line.partition("|")
            cond_str = cond_str.strip()
            temps = [float(t) for t in TEMP_RE.findall(cond_str)]
            times = [float(t) for t in TIME_RE.findall(cond_str)]
            atm_match = ATM_RE.search(cond_str)
            atm = [a.strip() for a in atm_match.group(1).split(",")] if atm_match else []
            operations.append({
                "type": op_type.strip(),
                "heating_temperature": temps,
                "heating_time": times,
                "heating_atmosphere": atm,
            })
    return precursors, operations


def make_predicted_route(target: str, precursors: list[dict], operations: list[dict]):
    from validator import (
        PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions
    )
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(formula=p["formula"], amount=p["amount"])
                    for p in precursors],
        operations=[
            PredictedOperation(
                type=op["type"],
                conditions=PredictedConditions(
                    heating_temperature=op.get("heating_temperature", []),
                    heating_time=op.get("heating_time", []),
                    heating_atmosphere=op.get("heating_atmosphere", []),
                ),
            )
            for op in operations
        ],
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen3(model_name: str, use_4bit: bool = True):
    log(f"Loading tokenizer from {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log(f"Loading model from {model_name} (4bit={use_4bit})...")
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def build_chat_prompts(tok, user_contents: list[str], enable_thinking: bool = True) -> list[str]:
    """Apply Qwen3 chat template with thinking enabled."""
    out = []
    for content in user_contents:
        messages = [{"role": "user", "content": content}]
        try:
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older transformers without enable_thinking kwarg
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        out.append(text)
    return out


@torch.no_grad()
def generate_batch(
    model, tok, prompts: list[str],
    max_new_tokens: int, temperature: float, top_p: float,
) -> list[str]:
    """Static batched generation. Returns one completion per prompt."""
    inputs = tok(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=2048,
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    # Left-padding: completion starts at fixed offset for all rows
    prompt_len = inputs.input_ids.shape[1]
    completion_ids = out[:, prompt_len:]
    return tok.batch_decode(completion_ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B",
                   help="HF model name. Use Qwen/Qwen3-8B if 14B OOMs at any batch size.")
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization (needs much more VRAM).")
    p.add_argument("--records", type=Path, default=DATA_RAW / "synthesis.json")
    p.add_argument("--summary", type=Path, default=DATA_RAW / "summary.json")
    p.add_argument("--formula-set", type=Path, default=DATA_CACHE / "mp_formula_set.pkl")
    p.add_argument("--output", type=Path, default=DATA_PROCESSED / "reasoning_traces.jsonl")
    p.add_argument("--validator-threshold", type=float, default=0.55)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Prompts per forward. 4 fits Qwen3-14B 4bit on 3090; try 8 if VRAM allows.")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["target_formula"], r["mp_record_idx"]))
            except Exception:
                continue
    return done


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    log("Loading MP synthesis records...")
    mp_records = load_mp_records(args.records)
    if args.limit:
        mp_records = mp_records[: args.limit]
    log(f"  {len(mp_records)} records loaded")

    log("Loading summary cache...")
    summary_by_formula = load_summary(args.summary)
    log(f"  {len(summary_by_formula)} summary entries")

    log("Loading validator (skip-thermo)...")
    from core.reward import load_validator
    validator = load_validator(
        formula_set_path=args.formula_set,
        pd_cache_path=None,
    )

    log("Checking resume state...")
    completed = load_completed(args.output)
    log(f"  {len(completed)} records already done")

    todo = []
    for idx, rec in enumerate(mp_records):
        target = rec.get("target_formula")
        if not target:
            continue
        if (target, idx) in completed:
            continue
        todo.append((idx, rec))
    log(f"  {len(todo)} records to process")
    if not todo:
        log("Nothing to do.")
        return

    model, tok = load_qwen3(args.model, use_4bit=not args.no_4bit)
    log("Model ready.")

    n_closed_book = 0
    n_fallback = 0
    n_failed = 0
    start_time = time.time()

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(n_batches):
        batch = todo[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

        # --- Pass 1: closed-book ---
        user_contents = []
        for _, rec in batch:
            target = rec["target_formula"]
            ctx = format_target_context(target, summary_by_formula)
            user_contents.append(CLOSED_BOOK_PROMPT.format(target=target, context=ctx))

        prompts = build_chat_prompts(tok, user_contents, enable_thinking=True)
        try:
            completions = generate_batch(
                model, tok, prompts,
                args.max_new_tokens, args.temperature, args.top_p,
            )
        except torch.cuda.OutOfMemoryError:
            log(f"  OOM at batch {batch_idx}. Try lowering --batch-size or --max-new-tokens.")
            raise

        closed_results = []
        for (idx, rec), comp in zip(batch, completions):
            thinking, post = split_thinking_and_output(comp)
            precursors, operations = parse_route(post)
            target = rec["target_formula"]
            try:
                route = make_predicted_route(target, precursors, operations)
                score, breakdown = validator.validate(route, target)
            except Exception:
                score, breakdown = 0.0, {"error": 1.0}
            closed_results.append({
                "idx": idx, "rec": rec, "raw": comp, "thinking": thinking,
                "precursors": precursors, "operations": operations,
                "score": score, "breakdown": breakdown,
            })

        # --- Pass 2: open-book fallback ---
        needs_fb = [r for r in closed_results if r["score"] < args.validator_threshold]
        fb_completions: dict[int, str] = {}
        if needs_fb:
            fb_user_contents = []
            for r in needs_fb:
                rec = r["rec"]
                target = rec["target_formula"]
                ctx = format_target_context(target, summary_by_formula)
                route_block = format_route_block(
                    rec.get("precursors", []) or [],
                    rec.get("operations", []) or [],
                )
                fb_user_contents.append(OPEN_BOOK_PROMPT.format(
                    target=target, context=ctx, route_block=route_block,
                ))
            fb_prompts = build_chat_prompts(tok, fb_user_contents, enable_thinking=True)
            try:
                fb_outputs = generate_batch(
                    model, tok, fb_prompts,
                    args.max_new_tokens, args.temperature, args.top_p,
                )
            except torch.cuda.OutOfMemoryError:
                log(f"  OOM on fallback at batch {batch_idx}. Skipping fallback for this batch.")
                fb_outputs = [""] * len(needs_fb)
            for r, out in zip(needs_fb, fb_outputs):
                fb_completions[r["idx"]] = out

        # --- Write results ---
        for r in closed_results:
            idx = r["idx"]
            rec = r["rec"]
            target = rec["target_formula"]
            mp_precursors = rec.get("precursors", []) or []
            mp_operations = rec.get("operations", []) or []

            if r["score"] >= args.validator_threshold:
                append_jsonl(args.output, {
                    "target_formula": target,
                    "mp_record_idx": idx,
                    "thinking": r["thinking"],
                    "predicted_precursors": r["precursors"],
                    "predicted_operations": r["operations"],
                    "validator_score": r["score"],
                    "validator_breakdown": r["breakdown"],
                    "mp_precursors": mp_precursors,
                    "mp_operations": mp_operations,
                    "used_fallback": False,
                    "raw_text_len": len(r["raw"]),
                    "generator": "hf",
                })
                n_closed_book += 1
            elif idx in fb_completions and fb_completions[idx]:
                fb_thinking, _ = split_thinking_and_output(fb_completions[idx])
                append_jsonl(args.output, {
                    "target_formula": target,
                    "mp_record_idx": idx,
                    "thinking": fb_thinking,
                    "predicted_precursors": mp_precursors,
                    "predicted_operations": mp_operations,
                    "validator_score": None,
                    "validator_breakdown": None,
                    "mp_precursors": mp_precursors,
                    "mp_operations": mp_operations,
                    "used_fallback": True,
                    "raw_text_len": len(fb_completions[idx]),
                    "generator": "hf",
                })
                n_fallback += 1
            else:
                n_failed += 1

        # --- Progress ---
        done = n_closed_book + n_fallback
        elapsed = time.time() - start_time
        rate = done / max(elapsed, 1)
        remaining = len(todo) - done
        eta_hours = remaining / max(rate * 3600, 0.001)
        log(
            f"batch {batch_idx + 1}/{n_batches}  "
            f"done={done}  "
            f"closed_book={n_closed_book}  "
            f"fallback={n_fallback}  "
            f"failed={n_failed}  "
            f"rate={rate*3600:.0f}/h  "
            f"ETA={eta_hours:.1f}h"
        )

    log("=" * 60)
    log(f"DONE in {(time.time()-start_time)/3600:.2f}h")
    log(f"  closed-book: {n_closed_book}  fallback: {n_fallback}  failed: {n_failed}")
    log(f"  output: {args.output}")


if __name__ == "__main__":
    main()