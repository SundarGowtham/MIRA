"""
data_pull_3.py
------------
Run modes:
    SMOKE = False  → pulls full solid-state corpus, builds full cache

Outputs (relative to PROJECT_ROOT):
    data/raw/synthesis.json          — raw synthesis records
    data/raw/summary.json            — summary docs for synthesis targets
    data/raw/robocrys.json           — natural language descriptions
    data/cache/mp_formula_set.pkl    — set of known MP formulas
    data/cache/pd_shards/            — INDIVIDUAL PhaseDiagram pickles
    data/cache/pd_index.json         — Lightweight mapping for the RL loop
    data/raw/dataset_stats.json      — descriptive statistics
"""

from __future__ import annotations

import json
import os
import pickle
import time
import signal
from collections import Counter
from pathlib import Path
from typing import Any

from monty.serialization import dumpfn, loadfn
from pymatgen.core import Composition
import dotenv

dotenv.load_dotenv()

# =====================================================================
# CONFIGURATION
# =====================================================================

SMOKE = False  
SKIP_ROBOCRYS = True

# THE V3 DATA UPGRADES: Unleashing the full thermodynamic space
PD_MAX_ELEMENTS  = 8     # Bumped from 5 to capture complex doped systems
PD_TIMEOUT_SECS  = 600   # 10 minutes per hull. Let the CPU cook.

PROJECT_ROOT = Path(__file__).parent
DATA_RAW   = PROJECT_ROOT / "data" / "raw"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"
PD_SHARDS_DIR = DATA_CACHE / "pd_shards"

SMOKE_SAMPLE_SIZE = 20

MP_API_KEY = os.environ.get("MP_API_KEY")
if MP_API_KEY is None:
    raise RuntimeError("Set MP_API_KEY env var: `export MP_API_KEY='your_key'`")

# =====================================================================
# CORE HELPERS
# =====================================================================

def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    PD_SHARDS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def with_retry(max_attempts: int = 5, base_delay: float = 2.0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except KeyboardInterrupt:
                    raise   
                except Exception as e:
                    if attempt == max_attempts:
                        log(f"  RETRY EXHAUSTED after {max_attempts} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    log(f"  retry {attempt}/{max_attempts} after {delay:.0f}s: {e}")
                    time.sleep(delay)
            return None  
        return wrapper
    return decorator

def synthesis_doc_to_dict(doc) -> dict[str, Any]:
    return {
        "doi": doc.doi,
        "synthesis_type": str(doc.synthesis_type),
        "target_formula": doc.target.material_formula if doc.target else None,
        "precursors": [
            {"formula": p.material_formula, "elements": p.composition[0].elements}
            for p in (doc.precursors or [])
        ],
        "operations": [
            {
                "type": str(op.type).split(".")[-1],
                "heating_temperature": [v.values for v in (op.conditions.heating_temperature or [])],
                "heating_time": [v.values for v in (op.conditions.heating_time or [])],
            }
            for op in (doc.operations or [])
        ],
    }

def get_chemsys(formula: str) -> str | None:
    try:
        els = sorted(str(el) for el in Composition(formula).elements)
        return "-".join(els)
    except Exception:
        return None

def is_valid_concrete_formula(formula: str | None) -> bool:
    if not formula or "-" in formula:
        return False
    try:
        comp = Composition(formula)
        for el in comp.elements:
            if not str(el).isalpha() or len(str(el)) > 2:
                return False
            if str(el) in {"M", "L", "A", "B", "X", "R", "Ln", "An"}:
                return False
        return True
    except Exception:
        return False

# =====================================================================
# STEPS 1-4 (UNCHANGED)
# =====================================================================

def pull_synthesis_records(mpr) -> list[dict]:
    log("Step 1: pulling synthesis records...")
    out_path = DATA_RAW / "synthesis.json"
    if out_path.exists():
        return loadfn(out_path)

    docs = mpr.materials.synthesis.search(synthesis_type=["solid-state"])
    if SMOKE: docs = docs[:SMOKE_SAMPLE_SIZE]
    
    records = [synthesis_doc_to_dict(d) for d in docs if is_valid_concrete_formula(d.target.material_formula if d.target else None)]
    dumpfn(records, out_path)
    return records

def pull_summary_for_targets(mpr, synth_records: list[dict]) -> list[dict]:
    log("Step 2: pulling summary docs for synthesis targets...")
    out_path = DATA_RAW / "summary.json"
    if out_path.exists():
        return loadfn(out_path)

    target_formulas = sorted({r["target_formula"] for r in synth_records if r.get("target_formula")})
    chemsys_set = {get_chemsys(f) for f in target_formulas if get_chemsys(f)}

    docs = mpr.materials.summary.search(
        chemsys=sorted(chemsys_set),
        fields=["material_id", "formula_pretty", "structure", "energy_above_hull", "is_stable", "nsites"]
    )
    
    target_reduced = {Composition(f).reduced_formula for f in target_formulas}
    by_formula = {}
    for d in docs:
        if not d.formula_pretty: continue
        key = Composition(d.formula_pretty).reduced_formula
        if key in target_reduced:
            if key not in by_formula or (d.energy_above_hull or 1e9) < (by_formula[key].energy_above_hull or 1e9):
                by_formula[key] = d

    records = [{"material_id": str(d.material_id), "formula_pretty": d.formula_pretty} for d in by_formula.values()]
    dumpfn(records, out_path)
    return records

def build_mp_formula_set(mpr, synth_records: list[dict]) -> set[str]:
    log("Step 4: building MP formula set...")
    out_path = DATA_CACHE / "mp_formula_set.pkl"
    if out_path.exists():
        with out_path.open("rb") as f:
            return pickle.load(f)

    formulas = set()
    for r in synth_records:
        if r.get("target_formula"): formulas.add(r["target_formula"])
        for p in r.get("precursors", []):
            if p.get("formula"): formulas.add(p["formula"])

    normalized = {Composition(f).reduced_formula for f in formulas if is_valid_concrete_formula(f)}
    
    with out_path.open("wb") as f:
        pickle.dump(normalized, f)
    return normalized

# =====================================================================
# STEP 5: THE SHARDED PHASE DIAGRAM CACHE (MULTIPROCESSING)
# =====================================================================

def process_single_chemsys(chemsys: str):
    """
    This function runs entirely inside an isolated worker process.
    It opens its own API connection, computes the hull, and writes the pickle.
    """
    import os
    import pickle
    import signal
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

    class _TimeoutError(Exception): pass
    def _timeout_handler(signum, frame): raise _TimeoutError("PD build exceeded timeout")

    scheme = MaterialsProjectDFTMixingScheme()

    @with_retry(max_attempts=5, base_delay=5.0)
    def fetch_entries():
        with MPRester(os.environ.get("MP_API_KEY")) as worker_mpr:
            return worker_mpr.get_entries_in_chemsys(
                elements=chemsys.split("-"),
                additional_criteria={"thermo_types": ["GGA_GGA+U", "R2SCAN"]}
            )

    try:
        entries = fetch_entries()
        entries = scheme.process_entries(entries)
        if not entries:
            return chemsys, None, 0, "No entries"

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(PD_TIMEOUT_SECS)
        try:
            pd = PhaseDiagram(entries)
        finally:
            signal.alarm(0)   
            
        shard_path = PD_SHARDS_DIR / f"{chemsys}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(pd, f)
            
        return chemsys, str(shard_path.relative_to(PROJECT_ROOT)), len(entries), "Success"

    except _TimeoutError:
        return chemsys, None, 0, "TIMEOUT"
    except Exception as e:
        return chemsys, None, 0, f"SKIP ({e})"


def build_sharded_phase_diagram_cache(mpr, synth_records: list[dict]) -> dict:
    import concurrent.futures
    import multiprocessing
    
    log("Step 5: building sharded phase diagram cache (PARALLEL MODE)...")
    index_path = DATA_CACHE / "pd_index.json"
    
    pd_index = {}
    if index_path.exists():
        with open(index_path, "r") as f:
            pd_index = json.load(f)
        log(f"  Resuming from index: {len(pd_index)} PDs already sharded.")

    chemsys_set = set()
    for r in synth_records:
        elements = set()
        if r.get("target_formula") and is_valid_concrete_formula(r["target_formula"]):
            elements.update(str(el) for el in Composition(r["target_formula"]).elements)
        for p in r.get("precursors", []):
            if p.get("formula") and is_valid_concrete_formula(p["formula"]):
                elements.update(str(el) for el in Composition(p["formula"]).elements)
        if elements and len(elements) <= PD_MAX_ELEMENTS:
            chemsys_set.add("-".join(sorted(elements)))

    # Filter out the ones we already completed
    chemsys_list = sorted([cs for cs in chemsys_set if cs not in pd_index])
    if SMOKE: chemsys_list = chemsys_list[:SMOKE_SAMPLE_SIZE]

    if not chemsys_list:
        log("  All phase diagrams already cached!")
        return pd_index

    # Use max cores minus 2 (to leave OS breathing room)
    # Cap at 24 to avoid instantly triggering an MP API HTTP 429 Rate Limit
    max_workers = min(24, max(1, multiprocessing.cpu_count() - 2))
    log(f"  Launching process pool with {max_workers} concurrent workers for {len(chemsys_list)} systems...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chemsys = {executor.submit(process_single_chemsys, cs): cs for cs in chemsys_list}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_chemsys), 1):
            chemsys, path, n_entries, status = future.result()
            
            if status == "Success":
                pd_index[chemsys] = path
                log(f"  [{i}/{len(chemsys_list)}] {chemsys}: {n_entries} entries → Shard saved")
            else:
                log(f"  [{i}/{len(chemsys_list)}] {chemsys}: {status}")

            # Safely write the index from the main thread only
            if i % 10 == 0:
                with open(index_path, "w") as f:
                    json.dump(pd_index, f, indent=2)

    # Final index save
    with open(index_path, "w") as f:
        json.dump(pd_index, f, indent=2)
        
    log(f"  saved → {index_path} ({len(pd_index)} PDs mapped)")
    return pd_index


# =====================================================================
# STEP 6: STATS
# =====================================================================

def compute_dataset_stats(synth_records, formula_set, pd_index) -> dict:
    log("Step 6: computing dataset statistics...")
    stats = {
        "n_synthesis_records": len(synth_records),
        "n_formulas_in_set": len(formula_set),
        "n_phase_diagram_shards": len(pd_index),
    }
    out_path = DATA_RAW / "dataset_stats.json"
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)
    return stats

def main():
    ensure_dirs()
    log(f"=== MIRA data pull 3 starting (SMOKE={SMOKE}) ===")
    from mp_api.client import MPRester
    
    with MPRester(MP_API_KEY) as mpr:
        synth = pull_synthesis_records(mpr)
        _ = pull_summary_for_targets(mpr, synth)
        formula_set = build_mp_formula_set(mpr, synth)
        pd_index = build_sharded_phase_diagram_cache(mpr, synth)
        
    stats = compute_dataset_stats(synth, formula_set, pd_index)
    log(f"Completed Phase Diagram Shards: {stats['n_phase_diagram_shards']}")
    log("=== Done ===")

if __name__ == "__main__":
    main()