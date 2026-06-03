"""
generate_traces_openrouter.py
-----------------------------
Async pipeline to generate chemistry reasoning traces using DeepSeek-R1 via OpenRouter.
Uses a worker pool architecture to prevent disk I/O deadlocks.
Incorporates Thermo-Aware SynthesisValidator and Closed/Open-Book Fallback logic.
"""

import os
import json
import asyncio
import aiohttp
import pickle
import logging
from pathlib import Path
from typing import Literal
from pymatgen.analysis.phase_diagram import PhaseDiagram
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from pymatgen.core import Composition
import dotenv
import aiofiles
from aiohttp import ClientTimeout
from rich import print as rprint
# from pymatgen import analysis.phase_diagram.PhaseDiagram

# Import your validator classes
from validator import (
    SynthesisValidator, 
    ThermoChecker, 
    PredictedRoute, 
    PredictedPrecursor, 
    PredictedOperation, 
    PredictedConditions
)

TIMEOUT = ClientTimeout(
    total=300,       # 5 min total — R1 can be slow on long traces
    connect=10,      
    sock_read=120    
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TRACER")

dotenv.load_dotenv()

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_ID = "deepseek/deepseek-v4-pro" 
NUM_WORKERS = 15                  
VALIDATOR_THRESHOLD = 0.65

PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"

SYNTHESIS_FILE = DATA_RAW / "synthesis.json"
SUMMARY_FILE = DATA_RAW / "summary.json"
PD_INDEX_FILE = DATA_CACHE / "pd_index.json"
FORMULA_SET_FILE = DATA_CACHE / "mp_formula_set.pkl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "synthesis_with_traces.jsonl"

# --- PYDANTIC SCHEMAS ---
class PrecursorSchema(BaseModel):
    formula: str
    amount: float

class OperationSchema(BaseModel):
    type: str
    temperature_c: float
    time_h: float
    atmosphere: str
    media: str

class RouteSchema(BaseModel):
    precursors: list[PrecursorSchema]
    operations: list[OperationSchema]
    thermodynamic_checks: list[str] = Field(default_factory=list)

# --- PROMPTS ---
SYSTEM_MSG = """You are a working materials chemist designing solid-state synthesis routes.

For every target compound, your internal reasoning MUST address:
1. STOICHIOMETRY: Oxidation states and balancing.
2. PRECURSOR CHOICE: Justify reagents.
3. BALANCED EQUATION: Explicit molar coefficients.
4. CONDITIONS: Justify temps/times/atmosphere based on thermodynamics.

You must output your final answer as a pure JSON object matching this schema:
{
  "precursors": [{"formula": "str", "amount": float}],
  "operations": [{"type": "str", "temperature_c": float, "time_h": float, "atmosphere": "str", "media": "str"}],
  "thermodynamic_checks": ["str"]
}
Do not include markdown formatting or backticks in the final output. Just the JSON object."""

CLOSED_BOOK_USER = """Target: {target}{context}

Thermodynamic Context (Phase Stability Data):
{stability_data}

Provide your synthesis route as a JSON object."""

OPEN_BOOK_USER = """Target: {target}{context}

A published solid-state synthesis route exists for this target:
PRECURSORS: {precursor_summary}
OPERATIONS: {operations_summary}

Thermodynamic Context (Phase Stability Data):
{stability_data}

Provide your synthesis route as a JSON object that closely follows this published route."""

# --- HELPER FUNCTIONS ---
def get_chemsys(formula: str) -> str | None:
    try:
        els = sorted(str(el) for el in Composition(formula).elements)
        return "-".join(els)
    except Exception:
        return None

def summarize_mp_precursors(precursors: list[dict]) -> str:
    parts = [f"{p.get('formula', '')} (amount={p.get('amount', 1.0)})" for p in precursors]
    return ", ".join(parts) if parts else "(none)"

def summarize_mp_operations(operations: list[dict]) -> str:
    parts = []
    for i, op in enumerate(operations, 1):
        parts.append(f"{i}. {op.get('type', 'Unknown')}")
    return " | ".join(parts) if parts else "(none)"

def extract_json(text: str) -> dict | None:
    """Aggressively extract JSON from the LLM content output."""
    try:
        # Strip markdown code blocks if the model ignored instructions
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception:
        return None

def convert_to_predicted_route(target: str, data: dict) -> PredictedRoute | None:
    try:
        schema = RouteSchema(**data)
        precs = [PredictedPrecursor(formula=p.formula, amount=p.amount) for p in schema.precursors]
        ops = []
        for op in schema.operations:
            ops.append(PredictedOperation(
                type=op.type,
                conditions=PredictedConditions(
                    heating_temperature=[op.temperature_c] if op.temperature_c > 0 else [],
                    heating_time=[op.time_h] if op.time_h > 0 else [],
                    heating_atmosphere=[op.atmosphere] if op.atmosphere else [],
                    mixing_media=op.media if op.media else None
                )
            ))
        return PredictedRoute(target_formula=target, precursors=precs, operations=ops)
    except Exception:
        return None

async def get_stability_data(target: str, pd_index: dict) -> str:
    chemsys = get_chemsys(target)
    pd_shard_path = pd_index.get(chemsys) if chemsys else None
    
    if not pd_shard_path or not Path(pd_shard_path).exists():
        return "No phase diagram data computed for this system."
        
    try:
        async with aiofiles.open(pd_shard_path, "rb") as f:
            pd: PhaseDiagram = pickle.loads(await f.read())
            
        target_comp = Composition(target)
        
        # 1. TARGET-SPECIFIC DECOMPOSITION ANALYSIS (Fixed for fractional formulas)
        try:
            # Normalizing to fractional composition prevents QHull spatial errors
            decomp, e_above = pd.get_decomp_and_e_above_hull(target_comp.fractional_composition)
            if e_above <= 0.001: 
                target_status = f"TARGET STATUS: {target} is THERMODYNAMICALLY STABLE (on the convex hull)."
            else:
                decomp_str = " + ".join([f"{amt:.3f} {entry.composition.reduced_formula}" for entry, amt in decomp.items()])
                target_status = (
                    f"TARGET STATUS: {target} is METASTABLE (+{e_above:.3f} eV/atom above hull).\n"
                    f"WARNING: It will spontaneously decompose into: {decomp_str}"
                )
        except Exception as e:
            target_status = f"TARGET STATUS: Could not compute specific stability for {target} (Likely complex solid-solution)."

        # 2. GENERAL STABLE PHASES
        stable_lines = []
        for entry in pd.stable_entries:
            form_e = pd.get_form_energy_per_atom(entry)
            formula = entry.composition.reduced_formula
            stable_lines.append(f"{formula} (ΔEf={form_e:.2f})")
            
        # 3. DANGEROUS COMPETING PHASES (Fixed to Deduplicate Polymorphs)
        competing_dict = {}
        for entry in pd.unstable_entries:
            e_above = pd.get_e_above_hull(entry)
            if e_above < 0.05:  # 50 meV/atom threshold
                formula = entry.composition.reduced_formula
                # If we haven't seen this formula, or this polymorph is more dangerous (lower e_above), save it
                if formula not in competing_dict or e_above < competing_dict[formula]:
                    competing_dict[formula] = e_above

        # Format the deduplicated dictionary back into a list
        competing_lines = [f"{form} (+{e:.3f} above hull)" for form, e in competing_dict.items()]

        # 4. CONSTRUCT THE FINAL RAG STRING
        stability_data = (
            "--- THERMODYNAMIC PHASE COMPETITION ---\n"
            f"{target_status}\n\n"
            "SYSTEM STABLE PHASES (Formation Energy in eV/atom):\n"
            f"  {', '.join(stable_lines)}\n\n"
            "DANGEROUS METASTABLE SIDE-PHASES (Energy above hull in eV/atom):\n"
            f"  {', '.join(competing_lines) if competing_lines else 'None within 50 meV/atom threshold.'}"
        )
        return stability_data
        
    except Exception as e:
        return f"[Error reading phase diagram: {e}]"

# --- WORKER LOOP ---
async def worker(queue: asyncio.Queue, session: aiohttp.ClientSession, pd_index: dict, validator: SynthesisValidator, pbar: tqdm, f_out: any, file_lock: asyncio.Lock):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/mira-project",
        "X-Title": "MIRA Capstone",
        "Content-Type": "application/json"
    }

    while True:
        record = await queue.get()
        if record is None:
            queue.task_done()
            break

        target = record.get("target_formula")
        precursors = record.get("precursors", [])
        operations = record.get("operations", [])
        ctx = "" # Add your summary context logic here if needed
        
        stability_data = await get_stability_data(target, pd_index)

        async def fetch_llm(prompt: str) -> tuple[str, str] | None:
            """Helper to hit OpenRouter and return (reasoning, final_json)."""
            payload = {
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt}
                ],
                "include_reasoning": True,
                "temperature": 0.6,
            }
            for attempt in range(3):
                try:
                    async with session.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=TIMEOUT) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "choices" in data and data["choices"]:
                                msg = data["choices"][0].get("message", {})
                                return msg.get("reasoning", ""), msg.get("content", "")
                        elif response.status == 429:
                            await asyncio.sleep(5 * (attempt + 1))
                        else:
                            break
                except Exception as e:
                    logger.debug(f"API attempt failed: {e}")
                    await asyncio.sleep(2)
            return None

        # ==========================================
        # PASS 1: CLOSED-BOOK ATTEMPT
        # ==========================================
        closed_prompt = CLOSED_BOOK_USER.format(target=target, context=ctx, stability_data=stability_data)
        llm_resp = await fetch_llm(closed_prompt)
        
        final_result = None
        closed_score = 0.0

        if llm_resp:
            reasoning, content = llm_resp
            json_data = extract_json(content)
            if json_data:
                route = convert_to_predicted_route(target, json_data)
                if route:
                    closed_score, closed_breakdown = validator.validate(route, target)
                    if closed_score >= VALIDATOR_THRESHOLD:
                        final_result = {
                            "target": target,
                            "thinking": f"<think>\n{reasoning}\n</think>",
                            "reasoning_raw": reasoning,
                            "predicted_route": json_data,
                            "thermodynamic_checks": json_data.get("thermodynamic_checks", []),
                            "validator_score": closed_score,
                            "validator_breakdown": closed_breakdown,
                            "passed_validator": True,
                            "used_fallback": False,
                            "prompt": closed_prompt,
                            "stability_data": stability_data,
                            "generator": MODEL_ID
                        }

        # ==========================================
        # PASS 2: OPEN-BOOK FALLBACK
        # ==========================================
        if not final_result:
            open_prompt = OPEN_BOOK_USER.format(
                target=target, 
                context=ctx, 
                precursor_summary=summarize_mp_precursors(precursors),
                operations_summary=summarize_mp_operations(operations),
                stability_data=stability_data
            )
            llm_resp = await fetch_llm(open_prompt)
            
            if llm_resp:
                reasoning, content = llm_resp
                json_data = extract_json(content)
                if json_data:
                    route = convert_to_predicted_route(target, json_data)
                    open_score, open_breakdown = 0.0, {"error": 1.0}
                    if route:
                        open_score, open_breakdown = validator.validate(route, target)
                    
                    final_result = {
                        "target": target,
                        "thinking": f"<think>\n{reasoning}\n</think>",
                        "reasoning_raw": reasoning,
                        "predicted_route": json_data,
                        "thermodynamic_checks": json_data.get("thermodynamic_checks", []),
                        "validator_score": open_score,
                        "validator_breakdown": open_breakdown,
                        "passed_validator": open_score >= VALIDATOR_THRESHOLD,
                        "used_fallback": True,
                        "prompt": open_prompt,
                        "stability_data": stability_data,
                        "generator": MODEL_ID
                    }

        # Save to disk if either pass succeeded in generating valid JSON
        if final_result:
            async with file_lock:
                await f_out.write(json.dumps(final_result) + "\n")
                await f_out.flush()

        pbar.update(1)
        queue.task_done()

async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Please set OPENROUTER_API_KEY environment variable.")
        return

    print("Loading data files...")
    async with aiofiles.open(SYNTHESIS_FILE, "r") as f:
        records = json.loads(await f.read())
    async with aiofiles.open(PD_INDEX_FILE, "r") as f:
        pd_index = json.loads(await f.read())
        
    print("Initializing Sharded Thermo-Validator...")
    with open(FORMULA_SET_FILE, "rb") as f:
        mp_formula_set = pickle.load(f)
        
    # The new lazy-loading ThermoChecker from the updated validator.py
    thermo_checker = ThermoChecker.from_sharded_cache(PD_INDEX_FILE, PROJECT_ROOT)
    validator = SynthesisValidator(mp_formula_set=mp_formula_set, thermo_checker=thermo_checker)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    completed_targets = set()
    if OUTPUT_FILE.exists():
        async with aiofiles.open(OUTPUT_FILE, "r") as f:
            async for line in f:
                if line.strip():
                    try:
                        completed_targets.add(json.loads(line)["target"])
                    except Exception:
                        pass
    
    pending_records = [r for r in records if r.get("target_formula") not in completed_targets]
    print(f"Total records: {len(records)} | Already completed: {len(completed_targets)} | Pending: {len(pending_records)}")

    if not pending_records:
        return

    queue = asyncio.Queue()
    for r in pending_records:
        queue.put_nowait(r)
    for _ in range(NUM_WORKERS):
        queue.put_nowait(None)

    file_lock = asyncio.Lock()
    pbar = tqdm(total=len(pending_records), desc="Generating Traces")

    async with aiohttp.ClientSession() as session:
        async with aiofiles.open(OUTPUT_FILE, "a") as f_out:
            workers = [
                asyncio.create_task(worker(queue, session, pd_index, validator, pbar, f_out, file_lock))
                for _ in range(NUM_WORKERS)
            ]
            await asyncio.gather(*workers)
            
    pbar.close()

if __name__ == "__main__":
    asyncio.run(main())