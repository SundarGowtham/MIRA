import os
import json
import asyncio
import aiohttp
import pickle
import logging
from pathlib import Path
from typing import Literal
from monty.json import MontyEncoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
import dotenv
import aiofiles
from aiohttp import ClientTimeout
from rich import print as rprint

from validator import SynthesisValidator, ThermoChecker, PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions
# from pymatgen import analysis.phase_diagram.PhaseDiagram


# SynthesisValidator
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


def best_pd_entry(pd: PhaseDiagram, formula: str):
    """Lowest-energy PD entry matching formula's reduced composition (MP polymorph)."""
    try:
        target_red = Composition(formula).reduced_formula
    except Exception:
        return None
    matches = [e for e in pd.all_entries if e.composition.reduced_formula == target_red]
    if not matches:
        return None
    return min(matches, key=lambda e: e.energy_per_atom)

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


async def main2():
    a = """
        <think>

        We need to design a solid-state synthesis route for La0.5Sr0.5FeO3. This is a perovskite solid solution. Let's analyze stoichiometry: La0.5Sr0.5FeO3. La is typically +3, Sr +2, Fe likely +3 or mixed? To balance, assume La3+, Sr2+, Fe? O is -2. Total positive charge: 0.5*3 + 0.5*2 + Fe = 1.5+1+Fe = 2.5 + Fe. For neutrality with 3 oxygens: 2.5+Fe = 6 => Fe = +3.5? That would be mixed Fe3+/Fe4+. So Fe oxidation state is +3.5, meaning equal amounts of Fe3+ and Fe4+ (since average 3.5). So La0.5Sr0.5FeO3 is a mixed-valence perovskite. This requires oxidizing conditions to stabilize Fe4+. So likely we need to use La2O3, SrCO3 or SrO, and Fe2O3 as iron source, and heat in air or oxygen to achieve Fe4+. The precursors: La2O3, SrCO3 (or SrO), Fe2O3. Let's calculate molar amounts: Target formula: La0.5Sr0.5FeO3. For 1 mole of product, need 0.5 La, 0.5 Sr, 1 Fe, 3 O. From La2O3: each mole provides 2 La. So need 0.25 mol La2O3. Sr: from SrCO3 (decomposes to SrO) gives 1 Sr per SrCO3. So need 0.5 mol SrCO3. Fe: from Fe2O3 gives 2 Fe per mole. Need 1 Fe, so 0.5 mol Fe2O3. But check oxygen balance: La2O3 supplies 0.25*3=0.75 O, SrCO3 gives after decomposition SrO + CO2, so SrO gives 0.5 O, Fe2O3 gives 0.5*3=1.5 O. Total O from precursors: 0.75+0.5+1.5 = 2.75 O. Need 3 O. So we need extra oxygen from atmosphere (air/O2) to oxidize Fe to +4 partially. Reaction will require O2 uptake: 0.25 O2 per formula unit. So we'll need calcination in O2 or air. Balanced equation: 0.25 La2O3 + 0.5 SrCO3 + 0.5 Fe2O3 + (0.125 O2?) Let's compute: La0.5Sr0.5FeO3 from La2O3, SrCO3, Fe2O3, O2.
        Write: 0.25 La2O3 + 0.5 SrCO3 + 0.5 Fe2O3 → La0.5Sr0.5FeO3 + 0.5 CO2 + ? O2? Actually we need O2 on left to balance. Let's tally: Left: La0.5, Sr0.5, Fe1, O: from 0.25 La2O3: 0.75 O; SrCO3 will decompose to SrO + CO2 during heating, so effectively SrO provides 0.5 O; Fe2O3 provides 1.5 O. Total O = 2.75. Right: target has 3 O. So we need 0.25 extra O atoms = 0.125 O2. So balanced: 0.25 La2O3 + 0.5 SrCO3 + 0.5 Fe2O3 + 0.125 O2 → La0.5Sr0.5FeO3 + 0.5 CO2. Thus the synthesis must be in O2 flow or static air? Air has O2 partial pressure of 0.21 atm, might suffice? Need thermodynamic justification: The compound requires oxidation, so we should heat in O2 atmosphere to favor Fe4+. Also consider competing phases: stable phases like SrFeO3, LaFeO3, Sr3Fe2O7, etc. The solid solution La0.5Sr0.5FeO3 is a known compound, typically synthesized by solid-state reaction at high temperature (e.g., 1100-1300 °C) in air or O2. Given the metastable phases close to hull, we need to avoid formation of LaFeO3, SrFeO3, etc. The target may be stable enough but careful control of stoichiometry and oxidation.

        We'll design a standard solid-state route: mix La2O3 (pre-dried), SrCO3, Fe2O3 in stoichiometric amounts, grind, calcine at 1100-1200 °C in O2 or air for 12-24 h, with intermediate grinding. Possibly multiple calcinations to ensure homogeneity. Sintering at 1300 °C. But we need to output JSON with precursors list (each formula, amount in moles relative to 1 mol target? The amount field: we can normalize to produce 1 mole of target. So amounts: La2O3: 0.25 mol, SrCO3: 0.5 mol, Fe2O3: 0.5 mol. Might also consider SrO instead of SrCO3 to avoid CO2, but SrCO3 is common, decomposes during heating. So use SrCO3. Atmosphere: oxygen. Operations: mix/grind, calcine, regrind, sinter. We'll give multiple steps: initial calcination at 1000 °C for 12 h in O2 to decompose SrCO3 and react, then higher temperature 1200 °C for 24 h with intermediate grinding. Finally maybe sinter at 1300 °C for 12 h. But the schema: list of operations with type (e.g., "grind", "calcine", "sinter"), temperature_c, time_h, atmosphere, media (e.g., "air", "oxygen", "argon"?). We'll use "dry" grinding with mortar and pestle, or ball mill. No media (or "none"?). We'll specify operations:
        Mixing: grind precursors in agate mortar (type: "grind", temperature: 25, time: 0.5, atmosphere: "air", media: "agate mortar").
        First calcination: heat in O2 at 1000°C for 12h (type: "calcine", temperature: 1000, time: 12, atmosphere: "oxygen", media: "alumina crucible").
        Regrind: grind again.
        Second calcination: 1200°C, 24h, O2.
        Final sintering: 1300°C, 12h, O2.
        But perhaps sintering is optional; we want single-phase powder. Many syntheses do 1200°C for 24h with multiple intermediate grindings. We'll propose: calcine at 1000°C/12h, then 1200°C/24h with regrinding. Atmosphere: O2. Thermodynamic checks: ensure oxidation potential, avoid formation of SrFeO3, LaFeO3. The compound is likely stable under O2 at high T. We'll note that excess Fe2O3 or La2O3 could lead to side phases, so precise stoichiometry is crucial. Also O2 flow helps incorporate oxygen. So JSON output.
        Define precursors: La2O3, SrCO3, Fe2O3. Amounts: La2O3: 0.25 mol, SrCO3: 0.5 mol, Fe2O3: 0.5 mol. Operations array. Thermodynamic checks: list of statements like "Oxidizing atmosphere required to stabilize Fe4+ and avoid reduction to Fe3+ phases (LaFeO3, SrFeO3).", "Precise stoichiometric control necessary to avoid SrLa(FeO3)2 and other metastable phases." Also mention that O2 partial pressure avoids SrFeO2. Include "Thermodynamic analysis indicates target phase is stable relative to decomposition into LaFeO3 + SrFeO3 under oxygen."

        Will output JSON.


        """
    b = """
            {
        "precursors": [
            {"formula": "La2O3", "amount": 0.25},
            {"formula": "SrCO3", "amount": 0.5},
            {"formula": "Fe2O3", "amount": 0.5}
        ],
        "operations": [
            {"type": "grind", "temperature_c": 25, "time_h": 0.5, "atmosphere": "air", "media": "agate mortar"},
            {"type": "calcine", "temperature_c": 1000, "time_h": 12, "atmosphere": "oxygen", "media": "alumina crucible"},
            {"type": "grind", "temperature_c": 25, "time_h": 0.5, "atmosphere": "air", "media": "agate mortar"},
            {"type": "calcine", "temperature_c": 1200, "time_h": 24, "atmosphere": "oxygen", "media": "alumina crucible"},
            {"type": "grind", "temperature_c": 25, "time_h": 0.5, "atmosphere": "air", "media": "agate mortar"},
            {"type": "sinter", "temperature_c": 1300, "time_h": 12, "atmosphere": "oxygen", "media": "alumina crucible"}
        ],
        "thermodynamic_checks": [
            "Oxygen atmosphere required to oxidize Fe to average +3.5 oxidation state, necessary for charge neutrality in La0.5Sr0.5FeO3.",
            "Stoichiometric control avoids LaFeO3 (ΔEf=-2.85) and SrFeO3 (ΔEf=-2.22) as competing stable phases; the perovskite solid solution is favored at high temperature under O2.",
            "Decomposition of SrCO3 provides in-situ SrO; total oxygen from precursors is insufficient, requiring 0.125 mol O2 per formula unit to complete oxidation, justifying calcination in pure oxygen.",
            "Regrinding and repeated calcinations at increasing temperatures ensure homogeneity and prevent trapping of unreacted La2O3 or SrO that could lead to SrLa(FeO3)2 (+0.005 eV/atom above hull)."
        ]
        }
    """
    llm_resp = (a, b)


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

    target = "La0.5Sr0.5FeO3"
    reasoning, content = llm_resp
    json_data = extract_json(content)
    if json_data:
        route = convert_to_predicted_route(target, json_data)
        if route:
            closed_score, closed_breakdown = validator.validate(route, target)
            if closed_score >= VALIDATOR_THRESHOLD:
                rprint(closed_breakdown)




async def main1():
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

    print("-"*100)
    print("-"*100)
    print("-"*100)

    record = records[400]

    target = record.get("target_formula")
    precursors = record.get("precursors", [])
    operations = record.get("operations", [])

    chemsys = get_chemsys(target)
    pd_shard_path = pd_index.get(chemsys) if chemsys else None
    
    if not pd_shard_path or not Path(pd_shard_path).exists():
        return "No phase diagram data computed for this system."
        
    try:
        async with aiofiles.open(pd_shard_path, "rb") as f:
            pd: PhaseDiagram = pickle.loads(await f.read())
            
        # target_comp = Composition(target)
        target_entry = best_pd_entry(pd, target)
        target_comp = Composition("La0.5Sr0.5FeO3")

        rprint("+"*30)
        rprint("get_composition_chempots")
        rprint(pd.get_composition_chempots(target_comp))
        rprint("+"*30)
        rprint("get_all_chempots")
        rprint(pd.get_all_chempots(target_comp))
        rprint("+"*30)
        # rprint("get_decomposition")
        # rprint(pd.get_decomposition(target_comp))
        # rprint("+"*30)
        # rprint("get_all_chempots")
        # rprint(pd.get_all_chempots(target_comp))
        # rprint("+"*30)
        # rprint("get_decomp_and_hull_energy_per_atom")
        # rprint(pd.get_decomp_and_hull_energy_per_atom(target_comp))
        # rprint("+"*30)
        # rprint("get_hull_energy")
        # rprint(pd.get_hull_energy(target_comp))
        # rprint("+"*30)
        # rprint("get_hull_energy_per_atom")
        # rprint(pd.get_hull_energy_per_atom(target_comp))
        # rprint("+"*30)
        # rprint("get_reference_energy")
        # rprint(pd.get_reference_energy(target_comp))
        # rprint("+"*30)
        # rprint("get_reference_energy_per_atom")
        # rprint(pd.get_reference_energy_per_atom(target_comp))
        # rprint("+"*30)
        # rprint("pd_coords")
        # rprint(pd.pd_coords(target_comp))

        # rprint("+"*30)
        # rprint("DANGEROUS COMPETING PHASES (Fixed to Deduplicate Polymorphs)")
        # competing_dict = {}
        # get_phase_separation_energy_dict_dangerous = {}
        # for entry in pd.unstable_entries:
        #     e_above = pd.get_e_above_hull(entry)
        #     pse = pd.get_phase_separation_energy(entry)
        #     get_phase_separation_energy_dict_dangerous[entry.composition.reduced_formula] = pse
        #     if e_above < 0.05:  # 50 meV/atom threshold
        #         formula = entry.composition.reduced_formula
        #         # If we haven't seen this formula, or this polymorph is more dangerous (lower e_above), save it
        #         if formula not in competing_dict or e_above < competing_dict[formula]:
        #             competing_dict[formula] = e_above

        # # Format the deduplicated dictionary back into a list
        # competing_lines = [f"{form} (+{e:.3f} above hull)" for form, e in competing_dict.items()]
        # rprint(competing_lines)
        # rprint("+"*30)

        # # Phase separation needs a PDEntry (composition + energy), not Composition alone.
        # if target_entry is None:
        #     rprint(
        #         f"[yellow]No MP entry for {target} ({target_comp.reduced_formula}) in this PD shard.[/yellow]\n"
        #         "Phase separation needs a PDEntry. Below: synthetic ComputedEntry at hull energy + 50 meV/atom."
        #     )
        #     _, hull_e_per_atom = pd.get_decomp_and_hull_energy_per_atom(target_comp)
        #     synthetic = ComputedEntry(
        #         target_comp,
        #         (hull_e_per_atom + 0.05) * target_comp.num_atoms,
        #     )
        #     rprint("+"*30)
        #     rprint("get_decomp_and_phase_separation_energy")
        #     rprint(f"Synthetic entry @ {synthetic.energy_per_atom:.4f} eV/atom")
        #     rprint(pd.get_decomp_and_phase_separation_energy(synthetic))
        #     rprint("+"*30)
        #     rprint("get_e_above_hull")
        #     rprint(pd.get_e_above_hull(synthetic))
        #     rprint("+"*30)
        #     rprint("get_equilibrium_reaction_energy")
        #     rprint(pd.get_equilibrium_reaction_energy(synthetic))
        #     rprint("+"*30)
        #     rprint("get_form_energy")
        #     rprint(pd.get_form_energy(synthetic))
        #     rprint("+"*30)
        #     rprint("get_form_energy_per_atom")
        #     rprint(pd.get_form_energy_per_atom(synthetic))
        #     rprint("+"*30)
        #     rprint("get_phase_separation_energy")
        #     rprint(pd.get_phase_separation_energy(synthetic))
        #     rprint("+"*30)
            

        # else:
        #     rprint(f"Using PD entry: {target_entry.composition} @ {target_entry.energy_per_atom:.4f} eV/atom")
        #     rprint(pd.get_decomp_and_phase_separation_energy(target_entry))

        rprint("+"*30)


        ## IDK HOW TO USE HTE BELOW ONES
        # rprint(pd.get_transition_chempots())
        # rprint(pd.getmu_vertices_stability_phase())
        # rprint(pd.numerical_tol)



    except Exception as e:
        rprint(f"[red]Error reading phase diagram:[/red] {e!r}")
        return


    print("-"*100)
    print("-"*100)
    print("-"*100)

    # stability_data = await get_stability_data(target, pd_index)
    # rprint(stability_data)
    # ctx = ""
    # closed_prompt = CLOSED_BOOK_USER.format(target=target, context=ctx, stability_data=stability_data)

    # rprint(f"{SYSTEM_MSG}\n{closed_prompt}")





if __name__ == "__main__":
    asyncio.run(main1())
