"""
generate_traces_openrouter.py
-----------------------------
Async pipeline to generate chemistry reasoning traces using DeepSeek-R1 via OpenRouter.
Uses a worker pool architecture to prevent disk I/O and memory deadlocks.
"""

import os
import json
import asyncio
import aiohttp
import pickle
from pathlib import Path
from tqdm import tqdm
from pymatgen.core import Composition
import dotenv
import aiofiles
from aiohttp import ClientTimeout

TIMEOUT = ClientTimeout(
    total=300,       # 5 min total — R1 can be slow on long traces
    connect=10,      # fail fast if we can't even connect
    sock_read=120    # reset if we stop receiving bytes for 2 min
)



import logging
# --- DEEP LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TRACER")


dotenv.load_dotenv()

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_ID = "deepseek/deepseek-r1-0528" 
NUM_WORKERS = 15                  # Number of concurrent API streams

PROJECT_ROOT = Path(__file__).parent
SYNTHESIS_FILE = PROJECT_ROOT / "data" / "raw" / "synthesis.json"
PD_INDEX_FILE = PROJECT_ROOT / "data" / "cache" / "pd_index.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "synthesis_with_traces.jsonl"

def get_chemsys(formula: str) -> str | None:
    try:
        els = sorted(str(el) for el in Composition(formula).elements)
        return "-".join(els)
    except Exception:
        return None

async def build_prompt(target: str, precursors: list, pd_shard_path: str | None) -> str:
    precursor_formulas = [p.get("formula") for p in precursors if p.get("formula")]
    
    stability_data = "No phase diagram data computed for this system."
    if pd_shard_path and os.path.exists(pd_shard_path):
        try:
            async with aiofiles.open(pd_shard_path, "rb") as f:
                raw_bytes = await f.read()
                pd = pickle.loads(raw_bytes)
                stable_phases = [entry.composition.reduced_formula for entry in pd.stable_entries]
                stability_data = f"Known stable phases in this system: {', '.join(stable_phases)}"
        except Exception:
            pass

    prompt = f"""You are an expert materials scientist. 
Given the target formula: {target}
And the available precursors: {precursor_formulas}

Thermodynamic Context (Phase Stability Data):
{stability_data}

Think step-by-step to derive the exact stoichiometric coefficients required to synthesize the target from the precursors.
In your thinking process:
1. Break down the target composition elements.
2. Cross-reference the stability data to ensure your balanced equation doesn't favor an unintended stable side-phase.
3. Balance the mass equations carefully.

Output only your reasoning process. End your reasoning by explicitly stating the final balanced chemical equation."""
    return prompt

async def worker(queue: asyncio.Queue, session: aiohttp.ClientSession, pd_index: dict, pbar: tqdm, f_out: any, file_lock: asyncio.Lock):
    """Worker loop that processes items from the queue sequentially."""
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
        
        chemsys = get_chemsys(target) if target else None
        pd_shard_path = pd_index.get(chemsys) if chemsys else None

        # Fixed: File I/O happens safely inside the controlled worker scope
        prompt = await build_prompt(target, precursors, pd_shard_path)

        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "include_reasoning": True,
            "temperature": 0.6,
        }

        result = None
        for attempt in range(3):
            try:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        message = data["choices"][0]["message"]
                        result = {
                            "target": target,
                            "precursors": precursors,
                            "reasoning_trace": message.get("reasoning", ""),
                            "final_answer": message.get("content", ""),
                            "prompt_used": prompt
                        }
                        break
                    elif response.status == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {target}: {type(e).__name__}: {e}")
                await asyncio.sleep(2)

        if result:
            # Thread-safe async file append using a lock
            async with file_lock:
                await f_out.write(json.dumps(result) + "\n")
                await f_out.flush()

        pbar.update(1)
        queue.task_done()

async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Please set OPENROUTER_API_KEY environment variable.")
        return

    print("Loading data...")
    async with aiofiles.open(SYNTHESIS_FILE, "r") as f:
        records = json.loads(await f.read())
        
    async with aiofiles.open(PD_INDEX_FILE, "r") as f:
        pd_index = json.loads(await f.read())

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
        print("All records processed!")
        return

    # Populate the queue
    queue = asyncio.Queue()
    for r in pending_records:
        queue.put_nowait(r)
    
    # Add sentinel values to cleanly shut down workers when done
    for _ in range(NUM_WORKERS):
        queue.put_nowait(None)

    file_lock = asyncio.Lock()
    pbar = tqdm(total=len(pending_records), desc="Generating Traces")

    async with aiohttp.ClientSession() as session:
        async with aiofiles.open(OUTPUT_FILE, "a") as f_out:
            # Spawn exactly NUM_WORKERS tasks
            workers = [
                asyncio.create_task(worker(queue, session, pd_index, pbar, f_out, file_lock))
                for _ in range(NUM_WORKERS)
            ]
            await asyncio.gather(*workers)
            
    pbar.close()

if __name__ == "__main__":
    asyncio.run(main())