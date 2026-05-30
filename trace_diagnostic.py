import os
import json
import asyncio
import aiohttp
import pickle
import logging
from pathlib import Path
from pymatgen.core import Composition
import dotenv
import aiofiles

# --- DEEP LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TRACER")

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
PROJECT_ROOT = Path(__file__).parent
SYNTHESIS_FILE = PROJECT_ROOT / "data" / "raw" / "synthesis.json"
PD_INDEX_FILE = PROJECT_ROOT / "data" / "cache" / "pd_index.json"

def get_chemsys(formula: str) -> str | None:
    try:
        els = sorted(str(el) for el in Composition(formula).elements)
        return "-".join(els)
    except Exception:
        return None

async def worker(worker_id: int, queue: asyncio.Queue, session: aiohttp.ClientSession, pd_index: dict):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/mira-project",
        "X-Title": "MIRA Capstone"
    }

    while True:
        record = await queue.get()
        if record is None:
            break

        target = record.get("target_formula")
        logger.info(f"[W{worker_id}] Picked up {target} from queue.")
        
        chemsys = get_chemsys(target) if target else None
        pd_shard_path = pd_index.get(chemsys) if chemsys else None

        # --- TRACE 1: FILE I/O ---
        if pd_shard_path and os.path.exists(pd_shard_path):
            logger.info(f"[W{worker_id}] -> Opening shard: {pd_shard_path}")
            try:
                async with aiofiles.open(pd_shard_path, "rb") as f:
                    raw_bytes = await f.read()
                logger.info(f"[W{worker_id}] -> Shard read into memory ({len(raw_bytes)} bytes). Unpickling...")
                
                # If it hangs here, the pickle file is corrupted or too complex!
                pd = pickle.loads(raw_bytes)
                logger.info(f"[W{worker_id}] -> Unpickling successful!")
            except Exception as e:
                logger.error(f"[W{worker_id}] -> FILE/PICKLE ERROR: {e}")
        else:
            logger.info(f"[W{worker_id}] -> No shard found. Skipping RAG.")

        # --- TRACE 2: NETWORK ---
        payload = {
            "model": "deepseek/deepseek-r1:free", # Testing with the free endpoint first
            "messages": [{"role": "user", "content": f"Test prompt for {target}"}],
        }
        
        logger.info(f"[W{worker_id}] -> Initiating OpenRouter POST request...")
        try:
            # Added a very strict 15-second timeout to force it to fail if it hangs
            async with session.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=15) as response:
                logger.info(f"[W{worker_id}] -> RECEIVED HTTP {response.status}")
                if response.status != 200:
                    text = await response.text()
                    logger.warning(f"[W{worker_id}] -> API ERROR MSG: {text}")
                else:
                    logger.info(f"[W{worker_id}] -> API Success!")
        except asyncio.TimeoutError:
            logger.error(f"[W{worker_id}] -> NETWORK TIMEOUT! Connection hung for 15 seconds.")
        except Exception as e:
            logger.error(f"[W{worker_id}] -> NETWORK ERROR: {type(e).__name__} - {e}")
            
        logger.info(f"[W{worker_id}] Finished {target}. Moving to next.\n")
        queue.task_done()

async def main():
    logger.info("Starting script...")
    
    with open(SYNTHESIS_FILE, "r") as f:
        records = json.load(f)
    with open(PD_INDEX_FILE, "r") as f:
        pd_index = json.load(f)

    # GRAB EXACTLY 5 RECORDS FOR TESTING
    test_records = records[:5]
    logger.info(f"Loaded {len(test_records)} test records.")

    queue = asyncio.Queue()
    for r in test_records:
        queue.put_nowait(r)
    
    # Poison pills
    for _ in range(2):
        queue.put_nowait(None)

    logger.info("Spawning 2 concurrent workers...")
    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(worker(i, queue, session, pd_index))
            for i in range(2)
        ]
        await asyncio.gather(*workers)
        
    logger.info("Diagnostic complete.")

if __name__ == "__main__":
    asyncio.run(main())