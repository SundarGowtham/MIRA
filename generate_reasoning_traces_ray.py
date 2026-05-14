"""
generate_reasoning_traces_ray.py
--------------------------------
Reasoning trace generator using the SupportVectors Ray cluster
(vLLM-served openai/gpt-oss-20b), with vLLM's guided_json constrained
decoding for guaranteed structured output.

Why guided_json:
  gpt-oss uses the "Harmony" response format which natively emits reasoning
  narratives even when not asked. Our earlier run produced 626 records that
  were 100% prose-in-<think>-tags with empty <precursors>/<operations>
  blocks. Prompt engineering can't fix this — gpt-oss is doing what it was
  trained to do.

  guided_json enforces a Pydantic JSON Schema at the *token level* via
  XGrammar (vLLM's default backend). Invalid tokens are masked to -inf,
  so the model physically cannot emit malformed output. Reasoning lives
  inside a dedicated `reasoning` field (placed first — the "reasoning-first"
  pattern that the structured-output literature flags as the single most
  important design choice).

JSONL output schema:
  - thinking:              "<think>{reasoning}</think>" — ready for SFT
  - reasoning_raw:         the raw reasoning text (no tags)
  - predicted_precursors:  list of {formula, amount}
  - predicted_operations:  list of {type, heating_temperature, ...}
  - validator_score / breakdown
  - mp_precursors / mp_operations
  - used_fallback          (closed-book vs open-book pass)
  - generator              "ray-gpt-oss-20b-guided"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError


PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"


# SV Ray cluster config
CHAT_BASE_URL = "http://10.0.10.51:8124"
CHAT_API_BASE_URL = f"{CHAT_BASE_URL}/v1"
SV_API_KEY = "sv-openai-api-key"
SV_MODEL = "openai/gpt-oss-20b"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Response schema — reasoning-first ordering is intentional
# ---------------------------------------------------------------------------

OperationType = Literal[
    "StartingSynthesis",
    "MixingOperation",
    "DryingOperation",
    "HeatingOperation",
    "ShapingOperation",
    "QuenchingOperation",
]


class Precursor(BaseModel):
    formula: str = Field(
        description="Chemical formula of the precursor, e.g., 'BaCO3', 'TiO2'."
    )
    amount: float = Field(
        description=(
            "Stoichiometric coefficient that balances the synthesis equation. "
            "E.g., for Na2Ti3O7 from Na2CO3 + 3 TiO2, Na2CO3 has amount 1.0 and "
            "TiO2 has amount 3.0. Do NOT default to 1.0."
        )
    )


class Operation(BaseModel):
    type: OperationType = Field(
        description="Operation type. Typical order: StartingSynthesis, "
                    "MixingOperation, DryingOperation, HeatingOperation."
    )
    temperature_c: float = Field(
        description="Temperature in Celsius. Use -1.0 if not applicable."
    )
    time_h: float = Field(
        description="Duration in hours. Use -1.0 if not applicable."
    )
    atmosphere: str = Field(
        description="Atmosphere, e.g., 'air', 'N2', 'Ar', 'O2'. Empty string if N/A."
    )
    media: str = Field(
        description="Mixing media, e.g., 'ethanol', 'water'. Empty string if N/A."
    )


class SynthesisResponse(BaseModel):
    """Solid-state synthesis route with chemistry reasoning.

    Field order is significant: `reasoning` MUST come first so the model
    generates analysis tokens before committing to precursors/operations.
    """

    reasoning: str = Field(
        description=(
            "Detailed chemistry analysis (200-600 words) covering: "
            "(1) target stoichiometry and oxidation states; "
            "(2) why each precursor is appropriate; "
            "(3) the balanced reaction equation with explicit molar ratios; "
            "(4) justification for temperatures, times, and atmosphere. "
            "Reason like a working materials chemist."
        )
    )
    precursors: list[Precursor] = Field(
        description="Precursors with stoichiometric coefficients matching the "
                    "balanced reaction. Must include all target elements."
    )
    operations: list[Operation] = Field(
        description="Ordered synthesis operations. Mixing before heating."
    )


RESPONSE_SCHEMA = SynthesisResponse.model_json_schema()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MSG = """You are a working materials chemist designing solid-state synthesis routes.

CRITICAL FORMATTING RULE — read this first:
You output JSON. Chemical formulas and oxidation states MUST use plain ASCII characters only.
- WRITE: Cu2Nb8O21, BaTiO3, Na2CO3, TiO2, Cs2Ti5O11
- DO NOT WRITE: Cu₂Nb₈O₂₁, BaTiO₃, Na₂CO₃ (Unicode subscripts)
- WRITE: Cu2+, Ti4+, O2-, Nb5+
- DO NOT WRITE: Cu²⁺, Ti⁴⁺, O²⁻, Nb⁵⁺ (Unicode superscripts)
- WRITE: → as -> or simply "yields"
- DO NOT WRITE: arrows, bullets (•, ‣), em-dashes (—), or other special characters

Subscripts/superscripts inside JSON strings cause downstream parsing to fail. Use ASCII consistently throughout the reasoning, the precursors, and the operations.

For every target compound, your `reasoning` field MUST address ALL of the following, in order:

(1) STOICHIOMETRY: Decompose the target formula into its elements and oxidation states. State each cation's required oxidation state explicitly.

(2) PRECURSOR CHOICE: For each element in the target, name the chosen precursor and justify why it is preferred over alternatives (cost, stability, decomposition behavior, oxidation state match). Use real reagents (binary oxides, carbonates, nitrates, hydroxides, sulfides — NOT elemental metals unless required).

(3) BALANCED EQUATION: Write the explicit balanced reaction equation including any gaseous byproducts (CO2, H2O, NOx). State the integer molar coefficients that satisfy element conservation. THESE coefficients are what you place in the `amount` field — NEVER default to 1.0 for every precursor.

(4) CONDITIONS: Justify the calcination/sintering temperature in relation to precursor decomposition temperatures and target phase formation. Justify the atmosphere (air for typical oxides, inert for sulfides/nitrides, reducing for low-oxidation-state cations). Justify hold times.

After reasoning, emit the precursors with their balanced-equation coefficients and the ordered operations. Operations sequence: StartingSynthesis (first), then mixing/drying/shaping as needed, then HeatingOperation for the high-temperature step.

EXAMPLE for BaTiO3:
- Reasoning would identify Ba2+, Ti4+, O2-; choose BaCO3 (decomposes cleanly above ~900C releasing CO2) and TiO2 (already in the +4 state); write BaCO3 + TiO2 -> BaTiO3 + CO2 (coefficients 1, 1, 1, 1); justify 1200C/4h/air based on the carbonate decomposition profile and perovskite formation temperature.
- Precursors: BaCO3 amount=1.0, TiO2 amount=1.0
- Operations: StartingSynthesis, MixingOperation (media=ethanol), HeatingOperation (T=1200, t=4, atmosphere=air)
"""

CLOSED_BOOK_USER = """Target: {target}{context}

Write the four reasoning sections (STOICHIOMETRY, PRECURSOR CHOICE, BALANCED EQUATION, CONDITIONS), then emit the precursors with their stoichiometric coefficients and the ordered operations."""

OPEN_BOOK_USER = """Target: {target}{context}

A published solid-state synthesis route exists for this target:

PRECURSORS: {precursor_summary}
OPERATIONS: {operations_summary}

Write the four reasoning sections (STOICHIOMETRY, PRECURSOR CHOICE, BALANCED EQUATION, CONDITIONS) that justify this exact route. In your output `precursors` and `operations` fields, emit the verified route — match the precursor formulas/amounts and operation sequence given above."""


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
    return " (" + ", ".join(parts) + ")"


def summarize_mp_precursors(precursors: list[dict]) -> str:
    parts = []
    for p in precursors:
        f = p.get("formula", "")
        a = p.get("amount", 1.0)
        parts.append(f"{f} (amount={a})" if a != 1.0 else f)
    return ", ".join(parts) if parts else "(none)"


def summarize_mp_operations(operations: list[dict]) -> str:
    parts = []
    for i, op in enumerate(operations, 1):
        bits = [op.get("type", "Unknown")]
        temps = op.get("heating_temperature") or []
        flat_temps = [t for sub in temps for t in (sub if isinstance(sub, list) else [sub])]
        if flat_temps:
            bits.append(f"T={sum(flat_temps)/len(flat_temps):.0f}C")
        times = op.get("heating_time") or []
        flat_times = [t for sub in times for t in (sub if isinstance(sub, list) else [sub])]
        if flat_times:
            bits.append(f"t={sum(flat_times)/len(flat_times):.1f}h")
        atm = op.get("heating_atmosphere") or []
        if atm:
            bits.append(f"atm={','.join(atm)}")
        parts.append(f"{i}. " + " ".join(bits))
    return " | ".join(parts) if parts else "(none)"


# ---------------------------------------------------------------------------
# Conversion: SynthesisResponse → validator's PredictedRoute / JSONL fields
# ---------------------------------------------------------------------------

def mp_precursors_to_predicted_schema(mp_precursors: list[dict]) -> list[dict]:
    """Convert MP precursor records into the {formula, amount} schema used in
    the predicted_precursors field. MP records have 'formula' + 'elements'
    + sometimes 'amount'; we strip the elements field and default amount to
    1.0 if absent (MP doesn't always carry stoich coefficients).
    """
    out = []
    for p in mp_precursors:
        out.append({
            "formula": p.get("formula", ""),
            "amount": float(p.get("amount", 1.0)),
        })
    return out


def mp_operations_to_predicted_schema(mp_operations: list[dict]) -> list[dict]:
    """Convert MP operation records into the schema used in
    predicted_operations. MP nests temperatures as [[1000.0]] etc.; we flatten.
    """
    out = []
    for op in mp_operations:
        temps = op.get("heating_temperature") or []
        flat_temps = [t for sub in temps for t in (sub if isinstance(sub, list) else [sub])]
        times = op.get("heating_time") or []
        flat_times = [t for sub in times for t in (sub if isinstance(sub, list) else [sub])]
        atm = op.get("heating_atmosphere") or []
        out.append({
            "type": op.get("type", "Unknown"),
            "heating_temperature": flat_temps,
            "heating_time": flat_times,
            "heating_atmosphere": atm,
            "media": op.get("mixing_media") or "",
        })
    return out


def to_predicted_route(target: str, resp: SynthesisResponse):
    from validator import (
        PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions
    )
    precursors = [
        PredictedPrecursor(formula=p.formula, amount=p.amount)
        for p in resp.precursors
    ]
    operations = []
    for op in resp.operations:
        temps = [op.temperature_c] if op.temperature_c > 0 else []
        times = [op.time_h] if op.time_h > 0 else []
        atm = [op.atmosphere] if op.atmosphere else []
        operations.append(PredictedOperation(
            type=op.type,
            conditions=PredictedConditions(
                heating_temperature=temps,
                heating_time=times,
                heating_atmosphere=atm,
            ),
        ))
    return PredictedRoute(
        target_formula=target,
        precursors=precursors,
        operations=operations,
    )


def serialize_response(resp: SynthesisResponse) -> tuple[list[dict], list[dict]]:
    prec = [{"formula": p.formula, "amount": p.amount} for p in resp.precursors]
    ops = []
    for op in resp.operations:
        ops.append({
            "type": op.type,
            "heating_temperature": [op.temperature_c] if op.temperature_c > 0 else [],
            "heating_time": [op.time_h] if op.time_h > 0 else [],
            "heating_atmosphere": [op.atmosphere] if op.atmosphere else [],
            "media": op.media if op.media else "",
        })
    return prec, ops


# ---------------------------------------------------------------------------
# Robust parse: gpt-oss can sometimes append junk after the closing }
# ---------------------------------------------------------------------------

# Unicode subscript and superscript translations. gpt-oss has a strong
# habit of emitting these despite prompt instructions, and they cause
# vLLM strict mode FSM crashes and downstream parsing pain. We normalize
# everywhere we ingest model output.
_UNICODE_SUBS = str.maketrans({
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
    "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
    "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
    "\u207a": "+", "\u207b": "-",
    "\u2192": "->", "\u27f6": "->", "\u21f6": "->",
    "\u2013": "-", "\u2014": "-",  # en/em dash
    "\u2022": "*", "\u2023": "*", "\u2043": "-",  # bullets
    "\u00b0": "",  # degree sign (we already handle "°C" => just "C")
    "\u202f": " ",  # narrow no-break space
    "\u00a0": " ",  # non-breaking space
})


def normalize_text(s: str) -> str:
    """ASCII-fy Unicode subscripts/superscripts/arrows/bullets in model output."""
    return s.translate(_UNICODE_SUBS) if s else s


def parse_response(text: str) -> SynthesisResponse | None:
    if not text:
        return None
    # Normalize Unicode artifacts that gpt-oss emits despite prompt rules
    text = normalize_text(text)
    try:
        return SynthesisResponse.model_validate_json(text)
    except (ValidationError, ValueError):
        pass
    s = text.strip()
    last = s.rfind("}")
    while last > 0:
        candidate = s[: last + 1]
        try:
            return SynthesisResponse.model_validate_json(candidate)
        except (ValidationError, ValueError):
            last = s.rfind("}", 0, last)
    return None


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

async def chat_completion(
    client: AsyncOpenAI,
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    """Constrained JSON output via response_format=json_schema.

    NOTE: strict=False intentionally. With strict=True, vLLM aborts generation
    mid-stream when gpt-oss emits Unicode subscript characters (₂, ₈, etc.)
    inside JSON string fields — the FSM enters states it can't continue from
    and we get truncated/empty responses. The system prompt below forbids
    Unicode subscripts, and our Pydantic parser handles the rest robustly.
    """
    resp = await client.chat.completions.create(
        model=SV_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SynthesisResponse",
                "schema": RESPONSE_SCHEMA,
                "strict": False,
            },
        },
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Per-record worker
# ---------------------------------------------------------------------------

async def process_record(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    record_idx: int,
    rec: dict,
    summary_by_formula: dict,
    validator,
    validator_threshold: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> dict | None:
    target = rec.get("target_formula")
    if not target:
        return None

    ctx = format_target_context(target, summary_by_formula)
    mp_precursors = rec.get("precursors", []) or []
    mp_operations = rec.get("operations", []) or []

    # Pass 1: closed-book
    # Empty responses from gpt-oss under strict json_schema appear to be a
    # gpt-oss + Harmony interaction issue. Retry once at slightly higher
    # temperature, which sometimes breaks the model out of the empty-emit
    # mode without compromising reasoning quality.
    closed_user = CLOSED_BOOK_USER.format(target=target, context=ctx)
    closed_raw = ""
    for attempt_temp in (temperature, min(temperature + 0.3, 0.9)):
        async with sem:
            try:
                closed_raw = await chat_completion(
                    client, SYSTEM_MSG, closed_user,
                    max_tokens, attempt_temp, top_p, timeout,
                )
            except Exception as e:
                log(f"  [idx={record_idx} {target}] closed-book error: {type(e).__name__}: {e}")
                return None
        if closed_raw:
            break  # got something, stop retrying
        # else: empty response, retry at higher temp
        log(f"  [idx={record_idx} {target}] empty closed-book at temp={attempt_temp}, retrying...")

    closed_resp = parse_response(closed_raw)
    if closed_resp is None:
        log(f"  [idx={record_idx} {target}] closed-book parse failed; raw_len={len(closed_raw)}")
        score = 0.0
        breakdown = {"error": 1.0}
    else:
        try:
            route = to_predicted_route(target, closed_resp)
            score, breakdown = validator.validate(route, target)
        except Exception as e:
            log(f"  [idx={record_idx} {target}] validator error on closed-book: {e}")
            score, breakdown = 0.0, {"error": 1.0}

    if closed_resp is not None and score >= validator_threshold:
        prec, ops = serialize_response(closed_resp)
        return {
            "target_formula": target,
            "mp_record_idx": record_idx,
            "thinking": f"<think>\n{closed_resp.reasoning}\n</think>",
            "reasoning_raw": closed_resp.reasoning,
            "predicted_precursors": prec,
            "predicted_operations": ops,
            "validator_score": score,
            "validator_breakdown": breakdown,
            "mp_precursors": mp_precursors,
            "mp_operations": mp_operations,
            "used_fallback": False,
            "raw_text_len": len(closed_raw),
            "generator": "ray-gpt-oss-20b-guided",
        }

    # Pass 2: open-book fallback (with same empty-response retry)
    open_user = OPEN_BOOK_USER.format(
        target=target,
        context=ctx,
        precursor_summary=summarize_mp_precursors(mp_precursors),
        operations_summary=summarize_mp_operations(mp_operations),
    )
    open_raw = ""
    for attempt_temp in (temperature, min(temperature + 0.3, 0.9)):
        async with sem:
            try:
                open_raw = await chat_completion(
                    client, SYSTEM_MSG, open_user,
                    max_tokens, attempt_temp, top_p, timeout,
                )
            except Exception as e:
                log(f"  [idx={record_idx} {target}] open-book error: {type(e).__name__}: {e}")
                return None
        if open_raw:
            break
        log(f"  [idx={record_idx} {target}] empty open-book at temp={attempt_temp}, retrying...")

    open_resp = parse_response(open_raw)
    if open_resp is None:
        log(f"  [idx={record_idx} {target}] open-book parse failed; raw_len={len(open_raw)}")
        return None

    return {
        "target_formula": target,
        "mp_record_idx": record_idx,
        "thinking": f"<think>\n{open_resp.reasoning}\n</think>",
        "reasoning_raw": open_resp.reasoning,
        "predicted_precursors": mp_precursors_to_predicted_schema(mp_precursors),
        "predicted_operations": mp_operations_to_predicted_schema(mp_operations),
        "validator_score": None,
        "validator_breakdown": None,
        "mp_precursors": mp_precursors,
        "mp_operations": mp_operations,
        "used_fallback": True,
        "raw_text_len": len(open_raw),
        "generator": "ray-gpt-oss-20b-guided",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, default=DATA_RAW / "synthesis.json")
    p.add_argument("--summary", type=Path, default=DATA_RAW / "summary.json")
    p.add_argument("--formula-set", type=Path, default=DATA_CACHE / "mp_formula_set.pkl")
    p.add_argument("--output", type=Path, default=DATA_PROCESSED / "reasoning_traces.jsonl")
    p.add_argument("--validator-threshold", type=float, default=0.65,
                   help="With guided_json, empty outputs are impossible, so we can "
                        "use the real threshold (not the 0.45 floor).")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=3072,
                   help="Bumped from 2048 — gpt-oss reasoning is long, and strict "
                        "json_schema mode aborts to empty content if generation "
                        "doesn't fit the schema in the token budget.")
    p.add_argument("--temperature", type=float, default=0.4,
                   help="Lowered from 0.7 — for structured output we want reliability.")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--request-timeout", type=float, default=240.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--retries", type=int, default=2)
    return p.parse_args()


def load_completed(path: Path) -> set[tuple[str, int]]:
    """Resume by reading existing records.

    NOTE: This skips records from ALL prior generator versions. The 626
    broken records from the un-guided run will be skipped. To regenerate
    them, run `filter_old_traces.py` (see below) first.
    """
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


async def append_jsonl(lock: asyncio.Lock, path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    async with lock:
        with path.open("a") as f:
            f.write(line)


async def amain():
    args = parse_args()

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
    log(f"  {len(completed)} records already done in {args.output}")

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

    log(f"Ray endpoint: {CHAT_API_BASE_URL}  model={SV_MODEL}")
    log(f"Mode: response_format=json_schema (strict)  fields: {list(SynthesisResponse.model_fields.keys())}")
    log(f"Concurrency: {args.concurrency}  max_tokens={args.max_tokens}  "
        f"threshold={args.validator_threshold}  temp={args.temperature}")

    client = AsyncOpenAI(
        base_url=CHAT_API_BASE_URL,
        api_key=SV_API_KEY,
        timeout=args.request_timeout,
        max_retries=0,
    )

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    n_closed_book = 0
    n_fallback = 0
    n_failed = 0
    start_time = time.time()
    last_log_done = 0

    async def worker(idx: int, rec: dict):
        nonlocal n_closed_book, n_fallback, n_failed, last_log_done
        result = None
        for attempt in range(args.retries + 1):
            try:
                result = await process_record(
                    client, sem, idx, rec, summary_by_formula, validator,
                    args.validator_threshold, args.max_tokens,
                    args.temperature, args.top_p, args.request_timeout,
                )
                break
            except Exception as e:
                if attempt < args.retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                log(f"  [idx={idx}] giving up: {e}")
                break

        if result is None:
            n_failed += 1
        else:
            await append_jsonl(write_lock, args.output, result)
            if result["used_fallback"]:
                n_fallback += 1
            else:
                n_closed_book += 1

        done = n_closed_book + n_fallback
        if done - last_log_done >= args.log_every:
            last_log_done = done
            elapsed = time.time() - start_time
            rate_h = done / max(elapsed, 1) * 3600
            remaining = len(todo) - done
            eta_h = remaining / max(rate_h, 0.001)
            cb_pct = 100 * n_closed_book / max(done, 1)
            log(
                f"  done={done}/{len(todo)}  "
                f"closed_book={n_closed_book} ({cb_pct:.0f}%)  "
                f"fallback={n_fallback}  "
                f"failed={n_failed}  "
                f"rate={rate_h:.0f}/h  "
                f"ETA={eta_h:.1f}h"
            )

    tasks = [asyncio.create_task(worker(idx, rec)) for idx, rec in todo]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        log("Interrupted. Partial progress saved.")
        for t in tasks:
            t.cancel()
        raise
    finally:
        await client.close()

    elapsed = time.time() - start_time
    log("=" * 60)
    log(f"DONE in {elapsed/3600:.2f}h")
    log(f"  closed-book: {n_closed_book}  fallback: {n_fallback}  failed: {n_failed}")
    log(f"  output: {args.output}")


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()