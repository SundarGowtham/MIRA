"""
generate_traces_openrouter.py  (v2)
-----------------------------------
Changes vs v1:
  - SYSTEM_MSG now describes the structured ThermoClaim schema (matches the
    Pydantic discriminated union below). The v1 prompt asked for ["str"]
    while the schema expected typed claims — a silent 100% parse-failure
    bug that would have wasted the entire run.
  - get_stability_data() now correctly handles solid-solution targets via
    pd.get_decomp_and_hull_energy_per_atom (which accepts a Composition,
    unlike get_decomp_and_e_above_hull which requires a PDEntry).
  - get_stability_data() injects an oxygen-chemical-potential classification
    derived from pd.get_all_chempots, telling the model whether the target
    requires oxidizing, reducing, or flexible atmosphere.
  - Write safety: numpy scalars in the validator breakdown are normalized
    to Python natives; serialization tries droppable fields before giving
    up; worker iterations are wrapped in try/finally so one bad record
    can't kill a worker or deadlock the queue.
"""

import os
import json
import asyncio
import aiohttp
import pickle
import logging
import warnings
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
import dotenv
import aiofiles
from aiohttp import ClientTimeout
from rich import print as rprint

from validator import (
    SynthesisValidator,
    ThermoChecker,
    PredictedRoute,
    PredictedPrecursor,
    PredictedOperation,
    PredictedConditions,
)

# Quiet the uncertainties UserWarning that pymatgen triggers from its
# internal error-propagation calls. Doesn't affect correctness.
warnings.filterwarnings("ignore", category=UserWarning, module="uncertainties")

TIMEOUT = ClientTimeout(total=300, connect=10, sock_read=120)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TRACER")

dotenv.load_dotenv()

# --- CONFIGURATION ------------------------------------------------------------
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

# Δμ_O thresholds (eV, relative to O2 reference) used to classify atmosphere
# requirements. These match the validator's MU_O_OXIDIZING_REQUIRED / REDUCING
# constants so the prompt hint and the validator agree on what "needs oxidizing"
# means.
MU_O_OXIDIZING_REQUIRED = -1.0
MU_O_REDUCING_REQUIRED = -3.0


# --- PYDANTIC SCHEMAS ---------------------------------------------------------
class PrecursorSchema(BaseModel):
    formula: str
    amount: float


# Canonical operation vocabulary. The schema below restricts type to this set;
# the normalization map (further down) maps common model-emitted synonyms onto
# these canonical values before Pydantic validation, so the model still has
# vocabulary freedom but the downstream pipeline sees a controlled set.
CANONICAL_OP_TYPES = (
    "mix",      # grinding, ball-milling, mortar-and-pestle prep (room temp)
    "dry",      # moisture/solvent removal
    "press",    # pelletization, shaping
    "calcine",  # initial heating; precursor decomposition / reaction
    "sinter",   # final densification heating
    "anneal",   # intermediate or final heat treatment
    "quench",   # rapid cooling
    "cool",     # controlled slow cooling
    "wash",     # post-synthesis solvent/acid wash
)

OP_TYPE_NORMALIZATION: dict[str, str] = {
    # mix
    "mix": "mix", "mixing": "mix",
    "grind": "mix", "grinding": "mix",
    "ball_mill": "mix", "ballmill": "mix",
    "ball-milling": "mix", "ballmilling": "mix",
    "mortar": "mix", "mortarandpestle": "mix",

    # dry
    "dry": "dry", "drying": "dry",

    # press
    "press": "press", "pressing": "press",
    "pellet": "press", "pelletize": "press", "pelletizing": "press",
    "shape": "press", "shaping": "press",

    # calcine — initial heating / precursor decomposition / reaction
    "calcine": "calcine", "calcination": "calcine", "calcining": "calcine",
    "heat": "calcine", "heating": "calcine",
    "fire": "calcine", "firing": "calcine",
    "reaction": "calcine", "react": "calcine",
    "sealed_tube_heating": "calcine", "sealedtubeheating": "calcine",
    "sealed-tube": "calcine", "ampoule": "calcine",
    "hydrothermal": "calcine",
    "solidstate": "calcine", "solid_state": "calcine",

    # sinter — final densification
    "sinter": "sinter", "sintering": "sinter",

    # anneal — intermediate heat treatment
    "anneal": "anneal", "annealing": "anneal",
    "heat_treatment": "anneal", "heattreatment": "anneal",

    # quench — rapid cooling
    "quench": "quench", "quenching": "quench",

    # cool — controlled slow cooling
    "cool": "cool", "cooling": "cool",
    "slow_cool": "cool", "slowcool": "cool", "slow_cooling": "cool",

    # wash
    "wash": "wash", "washing": "wash",
    "filter": "wash", "filtering": "wash",
    "rinse": "wash", "rinsing": "wash",
    "clean": "wash", "cleaning": "wash",
}


def normalize_op_type_for_parsing(raw_type: str) -> str:
    """
    Map any model-emitted op type onto the canonical set. Unmapped strings
    pass through unchanged (and will trip Pydantic's Literal validation,
    logged downstream as a vocabulary gap to be added to this map).
    """
    return OP_TYPE_NORMALIZATION.get(raw_type.lower().strip(), raw_type.lower().strip())


class OperationSchema(BaseModel):
    # Restricted Literal — anything outside CANONICAL_OP_TYPES fails Pydantic
    # validation. The pre-normalization step in convert_to_predicted_route()
    # maps synonyms first, so the model can emit "grinding" or "calcination"
    # naturally and they're collapsed to "mix"/"calcine" before reaching here.
    type: Literal[
        "mix", "dry", "press",
        "calcine", "sinter", "anneal",
        "quench", "cool",
        "wash",
    ]
    temperature_c: float
    time_h: float
    atmosphere: str
    media: str


class CompetingPhaseClaim(BaseModel):
    type: Literal["competing_phase"]
    formula: str
    form_energy_per_atom: float | None = None
    e_above_hull: float | None = None


class OxidationStateClaim(BaseModel):
    type: Literal["oxidation_state"]
    element: str
    avg_valence: float
    requires_atmosphere: Literal["oxidizing", "reducing", "inert", "any"]


class StoichiometryClaim(BaseModel):
    type: Literal["stoichiometric_constraint"]
    species: str
    moles_per_formula_unit: float
    role: Literal["consumed", "released"]


class HullStabilityClaim(BaseModel):
    type: Literal["hull_stability"]
    formula: str
    e_above_hull: float


ThermoClaim = Annotated[
    CompetingPhaseClaim | OxidationStateClaim | StoichiometryClaim | HullStabilityClaim,
    Field(discriminator="type"),
]


class RouteSchema(BaseModel):
    precursors: list[PrecursorSchema]
    operations: list[OperationSchema]
    thermodynamic_checks: list[ThermoClaim] = Field(default_factory=list)


# --- PROMPTS ------------------------------------------------------------------
SYSTEM_MSG = """You are a working materials chemist designing solid-state synthesis routes.

For every target compound, your internal reasoning MUST address:
1. STOICHIOMETRY: Oxidation states and balancing.
2. PRECURSOR CHOICE: Justify reagents.
3. BALANCED EQUATION: Explicit molar coefficients.
4. CONDITIONS: Justify temps/times/atmosphere based on thermodynamics.

You must output your final answer as a pure JSON object matching this schema:

{
  "precursors": [{"formula": "str", "amount": float}],
  "operations": [
    {"type": "str", "temperature_c": float, "time_h": float, "atmosphere": "str", "media": "str"}
  ],
  "thermodynamic_checks": [
    // Each entry must be one of the four structured claim types below.
    // Numeric values will be verified against the Materials Project convex hull.

    {"type": "oxidation_state", "element": "Fe", "avg_valence": 3.5, "requires_atmosphere": "oxidizing"},
    // Use when the target requires a specific cation valence.
    // requires_atmosphere is one of: "oxidizing", "reducing", "inert", "any".

    {"type": "competing_phase", "formula": "LaFeO3", "form_energy_per_atom": -2.85},
    // Use to name a phase that competes with the target during synthesis.
    // form_energy_per_atom is optional but graded against MP if provided.

    {"type": "stoichiometric_constraint", "species": "O2", "moles_per_formula_unit": 0.125, "role": "consumed"},
    // Use for non-precursor species exchanged with the atmosphere.
    // role is one of: "consumed", "released".

    {"type": "hull_stability", "formula": "SrLa(FeO3)2", "e_above_hull": 0.005}
    // Use to flag a metastable competitor by its energy above the convex hull (eV/atom).
  ]
}

The "type" field of each operation MUST be one of the following nine values
(any synonym will be normalized but using these exactly avoids ambiguity):

  - "mix"     — grinding, ball-milling, mortar-and-pestle prep (room temperature)
  - "dry"     — moisture or solvent removal
  - "press"   — pelletization, shaping, pressing into pellets
  - "calcine" — initial heating step; precursor decomposition or reaction
                (use for sealed-tube heating and hydrothermal — describe the
                vessel in "media", not in "type")
  - "sinter"  — final densification heating
  - "anneal"  — intermediate or final heat treatment (post-sinter)
  - "quench"  — rapid cooling (water, oil, gas blast)
  - "cool"    — controlled slow cooling at a defined rate
  - "wash"    — post-synthesis solvent/acid wash

Output 3-6 thermodynamic claims total. Reference only formulas that appear
in the SYSTEM STABLE PHASES or DANGEROUS METASTABLE SIDE-PHASES sections of
the prompt; do not invent phase names. When a numeric value (formation
energy, e_above_hull) is given in the prompt for a phase you reference,
include that exact number in your claim. Do not include markdown formatting,
backticks, or comments in the final output. Just the JSON object."""

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


# --- HELPERS ------------------------------------------------------------------
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
    parts = [f"{i}. {op.get('type', 'Unknown')}" for i, op in enumerate(operations, 1)]
    return " | ".join(parts) if parts else "(none)"


def extract_json(text: str) -> dict | None:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception:
        return None


def convert_to_predicted_route(target: str, data: dict) -> PredictedRoute | None:
    try:
        # Pre-normalize op-type vocabulary BEFORE Pydantic validation so that
        # model-emitted synonyms ("calcination", "sealed_tube_heating",
        # "ball-milling", "cooling") get mapped to the canonical Literal set
        # rather than failing validation outright. Anything unmapped passes
        # through and Pydantic will reject it, which we log for vocab growth.
        if isinstance(data, dict) and isinstance(data.get("operations"), list):
            unknown_types: list[str] = []
            for op in data["operations"]:
                if isinstance(op, dict) and "type" in op:
                    raw = str(op["type"])
                    normalized = normalize_op_type_for_parsing(raw)
                    if normalized not in CANONICAL_OP_TYPES:
                        unknown_types.append(raw)
                    op["type"] = normalized
            if unknown_types:
                logger.warning(
                    f"[parse] {target}: unmapped op types {unknown_types} "
                    f"(add to OP_TYPE_NORMALIZATION)"
                )

        schema = RouteSchema(**data)
        precs = [PredictedPrecursor(formula=p.formula, amount=p.amount) for p in schema.precursors]
        ops = []
        for op in schema.operations:
            ops.append(
                PredictedOperation(
                    type=op.type,
                    conditions=PredictedConditions(
                        heating_temperature=[op.temperature_c] if op.temperature_c > 0 else [],
                        heating_time=[op.time_h] if op.time_h > 0 else [],
                        heating_atmosphere=[op.atmosphere] if op.atmosphere else [],
                        mixing_media=op.media if op.media else None,
                    ),
                )
            )
        return PredictedRoute(target_formula=target, precursors=precs, operations=ops)
    except (ValidationError, ValueError, TypeError) as e:
        logger.debug(f"[parse] route conversion failed for {target}: {e}")
        return None


# --- WRITE SAFETY -------------------------------------------------------------
def _to_jsonable(obj):
    """
    Recursively convert numpy scalars / arrays and other non-native types to
    JSON-serializable Python natives. Handles NaN/Inf by replacing with None
    (so `allow_nan=False` doesn't trip on stray pymatgen values).
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    return obj


def safe_dump_record(record: dict, droppable_fields=("prompt", "stability_data")) -> str | None:
    """
    Serialize a record to JSON with three layers of fallback:
      1. Normalize numpy/NaN/Inf preemptively, try full dump.
      2. If that fails, drop optional fields one at a time and retry.
      3. If even that fails, fall back to essential keys only.

    Droppable fields (prompt, stability_data) are deterministically
    regenerable from `target` + the PD shards, so dropping them costs
    nothing — you can rebuild them in a postscript later.
    """
    cleaned = _to_jsonable(record)
    target_str = cleaned.get("target", "?") if isinstance(cleaned, dict) else "?"

    try:
        return json.dumps(cleaned, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        logger.warning(f"[serialize] full dump failed for {target_str}: {e}")

    for field in droppable_fields:
        if isinstance(cleaned, dict) and field in cleaned:
            cleaned = {k: v for k, v in cleaned.items() if k != field}
            try:
                result = json.dumps(cleaned, ensure_ascii=True, allow_nan=False)
                logger.info(f"[serialize] recovered for {target_str} by dropping '{field}'")
                return result
            except (TypeError, ValueError):
                continue

    essential_keys = ("target", "predicted_route", "validator_score", "passed_validator",
                      "thermodynamic_checks", "reasoning_raw")
    minimal = {k: cleaned.get(k) for k in essential_keys if isinstance(cleaned, dict) and k in cleaned}
    try:
        result = json.dumps(minimal, ensure_ascii=True, allow_nan=False)
        logger.warning(f"[serialize] minimal dump only for {target_str}")
        return result
    except Exception as e:
        logger.error(f"[serialize] even minimal dump failed for {target_str}: {e}")
        return None


# --- STABILITY DATA (v2: corrected solid-solution branch + μ_O injection) -----
def _classify_atmosphere(pd: PhaseDiagram, target_comp: Composition) -> str:
    """
    Return a one-line atmosphere-requirement hint based on the range of Δμ_O
    across the target composition's stability facets.

    Uses get_all_chempots (robust for edge/solid-solution compositions) and
    keys off the LOWER bound of Δμ_O: that's "the most reducing condition
    the target survives," which is what actually determines whether you need
    oxidizing conditions.
    """
    try:
        all_chempots = pd.get_all_chempots(target_comp)
    except Exception as e:
        logger.debug(f"[chempot] get_all_chempots failed for {target_comp.reduced_formula}: {e}")
        return "ATMOSPHERE: chempot analysis unavailable for this composition."

    if not all_chempots:
        return "ATMOSPHERE: no stability facets returned."

    o_el = Element("O")
    if o_el not in pd.el_refs:
        return "ATMOSPHERE: target contains no oxygen (analysis skipped)."

    o_ref = pd.el_refs[o_el].energy_per_atom
    o_mus_delta = [
        float(facet[o_el]) - float(o_ref)
        for facet in all_chempots.values()
        if o_el in facet
    ]

    if not o_mus_delta:
        return "ATMOSPHERE: oxygen chempot not present on any facet (unusual)."

    mu_lo, mu_hi = min(o_mus_delta), max(o_mus_delta)

    if mu_lo > MU_O_OXIDIZING_REQUIRED:
        return (
            f"ATMOSPHERE REQUIRED: oxidizing (air or O2). "
            f"Target requires Δμ_O > {mu_lo:.2f} eV relative to O2 reference "
            f"across all stability facets; reducing conditions will decompose it."
        )
    if mu_hi < MU_O_REDUCING_REQUIRED:
        return (
            f"ATMOSPHERE REQUIRED: reducing or inert (Ar, N2, H2, vacuum). "
            f"Target lies in low-μ_O regime, Δμ_O ∈ [{mu_lo:.2f}, {mu_hi:.2f}] eV; "
            f"oxidizing conditions will oxidize it away."
        )
    return (
        f"ATMOSPHERE: flexible. Target stable across Δμ_O ∈ [{mu_lo:.2f}, {mu_hi:.2f}] eV; "
        f"air, inert, or mildly reducing all acceptable."
    )


def _target_status_line(pd: PhaseDiagram, target: str, target_comp: Composition) -> str:
    """
    Honest stability framing for any target.

    For compositions with an explicit MP entry (e.g. BaTiO3), reports its
    e_above_hull from the lowest-energy matching entry.

    For solid solutions and other compositions without an entry, uses
    get_decomp_and_hull_energy_per_atom (which accepts a Composition) and
    reports the decomposition products explicitly. There is no honest
    "above hull" number for a composition without an energy — the
    composition lies on the convex envelope, period. The model needs to
    know what it will decompose into; that's the useful chemistry.
    """
    target_red = target_comp.reduced_formula

    matches = [
        e for e in pd.all_entries
        if e.composition.reduced_formula == target_red
    ]
    if matches:
        try:
            best = min(matches, key=lambda e: e.energy_per_atom)
            e_hull = float(pd.get_e_above_hull(best, on_error="ignore"))
            if e_hull <= 0.001:
                return f"TARGET STATUS: {target} is THERMODYNAMICALLY STABLE (on the convex hull)."
            return (
                f"TARGET STATUS: {target} is METASTABLE (+{e_hull:.3f} eV/atom above hull). "
                f"Will tend to decompose into more stable phases listed below."
            )
        except Exception as e:
            logger.debug(f"[status] e_above_hull failed for {target}: {e}")

    try:
        decomp, hull_e = pd.get_decomp_and_hull_energy_per_atom(target_comp)
        decomp_str = " + ".join(
            f"{amt:.3f} {entry.composition.reduced_formula}"
            for entry, amt in decomp.items()
        )
        return (
            f"TARGET STATUS: {target} is a non-discrete composition (no MP entry — "
            f"likely a solid solution or doped phase). At this composition the convex "
            f"hull lies at {hull_e:.3f} eV/atom and decomposes into: {decomp_str}. "
            f"Your synthesis must stabilize the target against this decomposition "
            f"(typically via configurational entropy at high T plus controlled cooling)."
        )
    except Exception as e:
        logger.debug(f"[status] decomp failed for {target}: {e}")
        return f"TARGET STATUS: stability analysis unavailable for {target}."


async def get_stability_data(target: str, pd_index: dict) -> str:
    chemsys = get_chemsys(target)
    pd_shard_path = pd_index.get(chemsys) if chemsys else None

    if not pd_shard_path or not Path(pd_shard_path).exists():
        return "No phase diagram data computed for this system."

    try:
        async with aiofiles.open(pd_shard_path, "rb") as f:
            pd: PhaseDiagram = pickle.loads(await f.read())

        target_comp = Composition(target)

        # 1. Stability framing — entry-based when MP has one, hull-based otherwise.
        target_status = _target_status_line(pd, target, target_comp)

        # 2. Atmosphere requirement from Δμ_O envelope.
        atmosphere_hint = _classify_atmosphere(pd, target_comp)

        # 3. Stable phases in the chemsys with formation energies.
        stable_lines = []
        for entry in pd.stable_entries:
            try:
                form_e = float(pd.get_form_energy_per_atom(entry))
                stable_lines.append(f"{entry.composition.reduced_formula} (ΔEf={form_e:.2f})")
            except Exception:
                continue

        # 4. Dangerous competing phases (within 50 meV of hull), deduped by formula.
        competing_dict: dict[str, float] = {}
        for entry in pd.unstable_entries:
            try:
                e_above = float(pd.get_e_above_hull(entry, on_error="ignore"))
            except Exception:
                continue
            if e_above is None or e_above >= 0.05:
                continue
            formula = entry.composition.reduced_formula
            if formula not in competing_dict or e_above < competing_dict[formula]:
                competing_dict[formula] = e_above

        competing_lines = [f"{form} (+{e:.3f} above hull)" for form, e in competing_dict.items()]

        return (
            "--- THERMODYNAMIC PHASE COMPETITION ---\n"
            f"{target_status}\n\n"
            f"{atmosphere_hint}\n\n"
            "SYSTEM STABLE PHASES (Formation Energy in eV/atom):\n"
            f"  {', '.join(stable_lines) if stable_lines else 'None resolved.'}\n\n"
            "DANGEROUS METASTABLE SIDE-PHASES (Energy above hull in eV/atom):\n"
            f"  {', '.join(competing_lines) if competing_lines else 'None within 50 meV/atom threshold.'}"
        )

    except Exception as e:
        logger.warning(f"[stability] failed to build context for {target}: {e}")
        return f"[Error reading phase diagram: {e}]"


# --- WORKER LOOP --------------------------------------------------------------
async def worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    pd_index: dict,
    validator: SynthesisValidator,
    pbar: tqdm,
    f_out,
    file_lock: asyncio.Lock,
):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/mira-project",
        "X-Title": "MIRA Capstone",
        "Content-Type": "application/json",
    }

    while True:
        record = await queue.get()
        if record is None:
            queue.task_done()
            break

        target = record.get("target_formula", "<unknown>")

        try:
            precursors = record.get("precursors", [])
            operations = record.get("operations", [])
            ctx = ""

            stability_data = await get_stability_data(target, pd_index)

            async def fetch_llm(prompt: str):
                payload = {
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                    "include_reasoning": True,
                    "temperature": 0.6,
                }
                for attempt in range(3):
                    try:
                        async with session.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            json=payload,
                            headers=headers,
                            timeout=TIMEOUT,
                        ) as response:
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
                        logger.debug(f"[api] attempt {attempt} failed for {target}: {e}")
                        await asyncio.sleep(2)
                return None

            # --- PASS 1: CLOSED-BOOK -----------------------------------------
            closed_prompt = CLOSED_BOOK_USER.format(
                target=target, context=ctx, stability_data=stability_data
            )
            llm_resp = await fetch_llm(closed_prompt)

            final_result = None
            closed_score = 0.0

            if llm_resp:
                reasoning, content = llm_resp
                # Don't let None render as the literal string "None" inside
                # the chain-of-thought field — that pollutes the training data
                # and teaches the model to emit "<think>\nNone\n</think>".
                reasoning_str = reasoning if reasoning else ""
                thinking_block = f"<think>\n{reasoning_str}\n</think>" if reasoning_str else None
                json_data = extract_json(content)
                if json_data:
                    route = convert_to_predicted_route(target, json_data)
                    if route:
                        closed_score, closed_breakdown = validator.validate(route, target)
                        if closed_score >= VALIDATOR_THRESHOLD:
                            final_result = {
                                "target": target,
                                "thinking": thinking_block,
                                "reasoning_raw": reasoning_str or None,
                                "predicted_route": json_data,
                                "thermodynamic_checks": json_data.get("thermodynamic_checks", []),
                                "validator_score": closed_score,
                                "validator_breakdown": closed_breakdown,
                                "passed_validator": True,
                                "used_fallback": False,
                                "prompt": closed_prompt,
                                "stability_data": stability_data,
                                "generator": MODEL_ID,
                            }

            # --- PASS 2: OPEN-BOOK FALLBACK ----------------------------------
            if not final_result:
                open_prompt = OPEN_BOOK_USER.format(
                    target=target,
                    context=ctx,
                    precursor_summary=summarize_mp_precursors(precursors),
                    operations_summary=summarize_mp_operations(operations),
                    stability_data=stability_data,
                )
                llm_resp = await fetch_llm(open_prompt)

                if llm_resp:
                    reasoning, content = llm_resp
                    reasoning_str = reasoning if reasoning else ""
                    thinking_block = f"<think>\n{reasoning_str}\n</think>" if reasoning_str else None
                    json_data = extract_json(content)
                    if json_data:
                        route = convert_to_predicted_route(target, json_data)
                        open_score, open_breakdown = 0.0, {"error": 1.0}
                        if route:
                            open_score, open_breakdown = validator.validate(route, target)

                        final_result = {
                            "target": target,
                            "thinking": thinking_block,
                            "reasoning_raw": reasoning_str or None,
                            "predicted_route": json_data,
                            "thermodynamic_checks": json_data.get("thermodynamic_checks", []),
                            "validator_score": open_score,
                            "validator_breakdown": open_breakdown,
                            "passed_validator": open_score >= VALIDATOR_THRESHOLD,
                            "used_fallback": True,
                            "prompt": open_prompt,
                            "stability_data": stability_data,
                            "generator": MODEL_ID,
                        }

            # --- WRITE (safe) ------------------------------------------------
            if final_result:
                serialized = safe_dump_record(final_result)
                if serialized:
                    async with file_lock:
                        await f_out.write(serialized + "\n")
                        await f_out.flush()
                else:
                    logger.error(f"[worker] dropping record for {target}; could not serialize")

        except Exception as e:
            # Swallow ANY exception so one bad record can't kill a worker
            # or deadlock the queue. Log with stack trace for postmortem.
            logger.exception(f"[worker] crashed on {target}: {type(e).__name__}: {e}")
        finally:
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
    print(
        f"Total records: {len(records)} | "
        f"Already completed: {len(completed_targets)} | "
        f"Pending: {len(pending_records)}"
    )

    if not pending_records:
        return

    queue: asyncio.Queue = asyncio.Queue()
    for r in pending_records:
        queue.put_nowait(r)
    for _ in range(NUM_WORKERS):
        queue.put_nowait(None)

    file_lock = asyncio.Lock()
    pbar = tqdm(total=len(pending_records), desc="Generating Traces")

    async with aiohttp.ClientSession() as session:
        async with aiofiles.open(OUTPUT_FILE, "a") as f_out:
            workers = [
                asyncio.create_task(
                    worker(queue, session, pd_index, validator, pbar, f_out, file_lock)
                )
                for _ in range(NUM_WORKERS)
            ]
            # return_exceptions=True so the gather doesn't abort the whole run
            # if some worker hits a truly unexpected condition outside the
            # per-record try/except.
            await asyncio.gather(*workers, return_exceptions=True)

    pbar.close()


if __name__ == "__main__":
    asyncio.run(main())