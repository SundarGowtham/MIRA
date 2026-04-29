"""
data_pull.py
------------
Fetch and cache all data needed for MIRA training.

Run modes:
    SMOKE = True   → pulls 20 synthesis records, builds tiny cache
                      (~2 min, validates pipeline end-to-end)
    SMOKE = False  → pulls full solid-state corpus, builds full cache
                      (~4-7 hours depending on chemsys diversity)

Outputs (relative to PROJECT_ROOT):
    data/raw/synthesis.json          — raw synthesis records
    data/raw/summary.json            — summary docs for synthesis targets
    data/raw/robocrys.json           — natural language descriptions
    data/cache/mp_formula_set.pkl    — set of known MP formulas
    data/cache/phase_diagrams.pkl    — precomputed PDs for thermo check
    data/raw/dataset_stats.json      — descriptive statistics

Cache philosophy:
    - JSON for serializable structured data (uses monty for pymatgen objects)
    - Pickle for PhaseDiagrams and the formula set (faster, larger objects)
    - Idempotent: re-running won't re-fetch unless you delete the files

Usage:
    export MP_API_KEY="your_key"
    python data_pull.py
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

from monty.serialization import dumpfn, loadfn
from pymatgen.core import Composition
import dotenv


dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SMOKE = False  # Set False for full run

# Robocrys data is only needed for the future structure-text alignment
# module (week 2). It's the slowest step (~3 hours per-material API calls)
# and is NOT needed for the validator, SFT, or GRPO training. Skip it
# until we actually need it.
SKIP_ROBOCRYS = True

# Phase diagram caps to bound runtime. Doped/multi-additive synthesis
# records produce chemsystems with 6-8 elements where PD construction
# can take minutes per system. The chemistry of those systems is not
# meaningfully different from their parent (e.g. doped BaTiO3 has the
# same thermodynamics as undoped BaTiO3 for our purposes), so we skip.
PD_MAX_ELEMENTS  = 5     # skip chemsystems with more than this many elements
PD_TIMEOUT_SECS  = 60    # skip individual PD builds taking longer than this

PROJECT_ROOT = Path(__file__).parent
DATA_RAW   = PROJECT_ROOT / "data2" / "raw"
DATA_CACHE = PROJECT_ROOT / "data2" / "cache"

SMOKE_SAMPLE_SIZE = 20

# MP API key from environment (Materials Project recommends env var)
MP_API_KEY = os.environ.get("MP_API_KEY")
if MP_API_KEY is None:
    raise RuntimeError(
        "Set MP_API_KEY env var: `export MP_API_KEY='your_key'`"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def with_retry(max_attempts: int = 5, base_delay: float = 2.0):
    """
    Decorator: retry on transient network/API failures with
    exponential backoff. Gives up after max_attempts.

    Use on any function that wraps an MP API call.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except KeyboardInterrupt:
                    raise   # always pass through Ctrl+C
                except Exception as e:
                    if attempt == max_attempts:
                        log(f"  RETRY EXHAUSTED after {max_attempts} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    log(f"  retry {attempt}/{max_attempts} after {delay:.0f}s: {e}")
                    time.sleep(delay)
            return None  # unreachable
        return wrapper
    return decorator


def synthesis_doc_to_dict(doc) -> dict[str, Any]:
    """
    Convert MP synthesis MPDataDoc to a JSON-serializable dict.
    Keep only fields the validator + SFT training need.
    """
    return {
        "doi": doc.doi,
        "paragraph_string": doc.paragraph_string,
        "synthesis_type": str(doc.synthesis_type),
        "reaction_string": doc.reaction_string,
        "target_formula": (
            doc.target.material_formula if doc.target else None
        ),
        "target_additives": (
            doc.target.additives if doc.target else []
        ),
        "precursors": [
            {"formula": p.material_formula, "elements": p.composition[0].elements}
            for p in (doc.precursors or [])
        ],
        "operations": [
            {
                "type": str(op.type).split(".")[-1],
                "token": op.token,
                "heating_temperature": [
                    v.values for v in (op.conditions.heating_temperature or [])
                ],
                "heating_time": [
                    v.values for v in (op.conditions.heating_time or [])
                ],
                "heating_atmosphere": op.conditions.heating_atmosphere or [],
                "mixing_media": op.conditions.mixing_media,
            }
            for op in (doc.operations or [])
        ],
    }


def get_chemsys(formula: str) -> str | None:
    """Alphabetized hyphen-joined chemsys, or None if unparseable."""
    try:
        els = sorted(str(el) for el in Composition(formula).elements)
        return "-".join(els)
    except Exception:
        return None


def is_valid_concrete_formula(formula: str | None) -> bool:
    """
    True iff the formula can be parsed by pymatgen AND contains only
    concrete elements (no placeholders like 'M', 'Ln', or 'A' that are
    common in templated synthesis records).

    Excludes formulas with hyphens (e.g. 'Cu2O-M2O5'), which indicate
    multi-target or templated reactions.
    """
    if not formula or "-" in formula:
        return False
    try:
        comp = Composition(formula)
        # Composition will accept 'M', 'Ln' etc. as DummySpecies — reject these
        for el in comp.elements:
            if not str(el).isalpha() or len(str(el)) > 2:
                return False
            # Reject single-letter pseudo-elements that aren't real:
            # M (metal), L (ligand), A/B/X (perovskite placeholders), Ln, R
            if str(el) in {"M", "L", "A", "B", "X", "R", "Ln", "An"}:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 1 — Synthesis records
# ---------------------------------------------------------------------------

def pull_synthesis_records(mpr) -> list[dict]:
    log("Step 1: pulling synthesis records...")
    out_path = DATA_RAW / "synthesis.json"
    if out_path.exists():
        log(f"  cached → loading {out_path}")
        return loadfn(out_path)

    docs = mpr.materials.synthesis.search(synthesis_type=["solid-state"])
    log(f"  fetched {len(docs)} solid-state records")

    if SMOKE:
        docs = docs[:SMOKE_SAMPLE_SIZE]
        log(f"  SMOKE: truncated to {len(docs)}")

    records = []
    for d in docs:
        try:
            records.append(synthesis_doc_to_dict(d))
        except Exception as e:
            log(f"  skip record: {e}")

    dumpfn(records, out_path)
    log(f"  saved → {out_path}")
    return records


# ---------------------------------------------------------------------------
# Step 2 — Summary docs for synthesis targets
# ---------------------------------------------------------------------------

def pull_summary_for_targets(mpr, synth_records: list[dict]) -> list[dict]:
    log("Step 2: pulling summary docs for synthesis targets...")
    out_path = DATA_RAW / "summary.json"
    if out_path.exists():
        log(f"  cached → loading {out_path}")
        return loadfn(out_path)

    # Filter to valid concrete formulas only (drops 'Cu2O-M2O5' etc.)
    target_formulas = sorted({
        r["target_formula"] for r in synth_records
        if is_valid_concrete_formula(r.get("target_formula"))
    })
    dropped = sum(
        1 for r in synth_records
        if r.get("target_formula") and not is_valid_concrete_formula(r["target_formula"])
    )
    log(f"  {len(target_formulas)} valid target formulas "
        f"({dropped} dropped: templated/hyphenated)")

    if not target_formulas:
        log("  no valid targets — skipping")
        dumpfn([], out_path)
        return []

    # The MP summary endpoint doesn't accept a list of formulas in one call.
    # We query by chemsys (which DOES accept multiple) and filter client-side
    # to the exact formulas we want. This is much faster than per-formula calls.
    target_set = set(target_formulas)
    chemsys_set: set[str] = set()
    for f in target_formulas:
        cs = get_chemsys(f)
        if cs:
            chemsys_set.add(cs)

    log(f"  querying {len(chemsys_set)} unique chemsystems...")

    docs = mpr.materials.summary.search(
        chemsys=sorted(chemsys_set),
        fields=[
            "material_id", "formula_pretty", "structure",
            "band_gap", "formation_energy_per_atom",
            "energy_above_hull", "is_stable", "is_metal",
            "symmetry", "density", "nsites",
            "possible_species", "has_props",
        ],
    )
    log(f"  fetched {len(docs)} summary docs across requested chemsystems")

    # Keep only the docs whose formula matches a target we asked about.
    # Use reduced_formula on both sides to handle "BaTiO3" vs "Ba(TiO3)" etc.
    target_reduced = {
        Composition(f).reduced_formula for f in target_formulas
    }
    filtered_docs = [
        d for d in docs
        if d.formula_pretty
        and Composition(d.formula_pretty).reduced_formula in target_reduced
    ]
    log(f"  {len(filtered_docs)} match a synthesis target")

    # When multiple polymorphs exist for the same formula, keep the one
    # with lowest energy_above_hull (the most stable polymorph).
    by_formula: dict[str, Any] = {}
    for d in filtered_docs:
        key = Composition(d.formula_pretty).reduced_formula
        if key not in by_formula:
            by_formula[key] = d
        else:
            existing = by_formula[key]
            cur_hull = d.energy_above_hull if d.energy_above_hull is not None else 1e9
            ext_hull = existing.energy_above_hull if existing.energy_above_hull is not None else 1e9
            if cur_hull < ext_hull:
                by_formula[key] = d

    log(f"  {len(by_formula)} unique materials after polymorph dedup")

    records = [
        {
            "material_id": str(d.material_id),
            "formula_pretty": d.formula_pretty,
            "structure": d.structure.as_dict() if d.structure else None,
            "band_gap": d.band_gap,
            "formation_energy_per_atom": d.formation_energy_per_atom,
            "energy_above_hull": d.energy_above_hull,
            "is_stable": d.is_stable,
            "is_metal": d.is_metal,
            "crystal_system": (
                str(d.symmetry.crystal_system) if d.symmetry else None
            ),
            "spacegroup_number": (
                d.symmetry.number if d.symmetry else None
            ),
            "density": d.density,
            "nsites": d.nsites,
            "possible_species": d.possible_species or [],
            "has_synthesis": (
                d.has_props.get("synthesis", False) if d.has_props else False
            ),
        }
        for d in by_formula.values()
    ]

    dumpfn(records, out_path)
    log(f"  saved → {out_path}")
    return records


# ---------------------------------------------------------------------------
# Step 3 — Robocrys descriptions
# ---------------------------------------------------------------------------

def pull_robocrys_descriptions(mpr, summary_records: list[dict]) -> list[dict]:
    """
    Pull natural language descriptions for our target materials.

    Writes incrementally to a JSONL file so a kill mid-run doesn't lose
    progress. On resume, skips materials whose IDs are already in the file.

    SKIPPED entirely when SKIP_ROBOCRYS = True. Robocrys is only needed
    for the structure-text alignment module (week 2 of the project),
    not for the validator, SFT, or GRPO training.
    """
    log("Step 3: pulling robocrys descriptions...")

    if SKIP_ROBOCRYS:
        log("  SKIP_ROBOCRYS = True — skipping (run separately later)")
        # Write empty file so downstream code doesn't break
        out_path_json = DATA_RAW / "robocrys.json"
        if not out_path_json.exists():
            dumpfn([], out_path_json)
        return []

    # We use JSONL for durability — one record per line, appended as we go.
    # The legacy JSON array file is built ONCE at the end from the JSONL.
    out_path_jsonl = DATA_RAW / "robocrys.jsonl"
    out_path_json  = DATA_RAW / "robocrys.json"

    # If the consolidated JSON exists, we already finished a previous run
    if out_path_json.exists():
        log(f"  cached → loading {out_path_json}")
        return loadfn(out_path_json)

    # Resume: read whatever JSONL we have so far
    seen_ids: set[str] = set()
    records: list[dict] = []
    if out_path_jsonl.exists():
        with out_path_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    seen_ids.add(rec["material_id"])
                    records.append(rec)
                except Exception:
                    pass
        log(f"  resuming: {len(seen_ids)} descriptions already cached")

    material_ids = [r["material_id"] for r in summary_records]
    todo = [mid for mid in material_ids if mid not in seen_ids]

    if not todo:
        log(f"  all {len(material_ids)} already cached")
        # Consolidate to canonical JSON for downstream code
        dumpfn(records, out_path_json)
        return records

    log(f"  fetching {len(todo)} of {len(material_ids)} descriptions "
        f"({len(seen_ids)} already cached)...")

    # Open in append mode so each fetched description is durable immediately
    fetched_this_run = 0
    with out_path_jsonl.open("a") as out_f:
        for i, mid in enumerate(todo, 1):
            doc = None
            try:
                results = mpr.materials.robocrys.search(keywords=[mid])
                for r in results:
                    if str(r.material_id) == mid:
                        doc = r
                        break
            except Exception:
                pass

            if doc is None:
                try:
                    doc = mpr.materials.robocrys.get_data_by_id(mid)
                except Exception:
                    pass

            if doc is not None:
                desc = getattr(doc, "description", None)
                if desc:
                    rec = {
                        "material_id": str(doc.material_id),
                        "description": desc,
                    }
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()       # force OS write — survives kill
                    records.append(rec)
                    fetched_this_run += 1

            # Progress log every 100 to give live visibility
            if i % 100 == 0:
                log(f"  [{i}/{len(todo)}] fetched (total cached: "
                    f"{len(seen_ids) + fetched_this_run})")

    # Final consolidation to JSON for downstream consumers
    dumpfn(records, out_path_json)
    log(f"  saved → {out_path_json} ({len(records)} non-empty descriptions)")
    return records


# ---------------------------------------------------------------------------
# Step 4 — MP formula set (for validator precursors_exist check)
# ---------------------------------------------------------------------------

def build_mp_formula_set(
    mpr,
    synth_records: list[dict],
    summary_records: list[dict],
) -> set[str]:
    log("Step 4: building MP formula set...")
    out_path = DATA_CACHE / "mp_formula_set.pkl"
    if out_path.exists():
        log(f"  cached → loading {out_path}")
        with out_path.open("rb") as f:
            return pickle.load(f)

    # Start with formulas from synthesis dataset (precursors + targets) —
    # these are guaranteed to be real materials referenced in literature
    formulas: set[str] = set()
    for r in synth_records:
        if r.get("target_formula"):
            formulas.add(r["target_formula"])
        for p in r.get("precursors", []):
            if p.get("formula"):
                formulas.add(p["formula"])

    log(f"  {len(formulas)} formulas from synthesis records")

    # Augment with the broader set of MP materials. We pull formula_pretty
    # for ALL stable materials so unknown precursors still get matched if
    # they exist anywhere in MP. This is a one-time bulk query.
    if not SMOKE:
        log("  pulling broader MP formula list (this may take a minute)...")
        all_mp = mpr.materials.summary.search(
            is_stable=True,
            fields=["formula_pretty"],
        )
        for d in all_mp:
            formulas.add(d.formula_pretty)
        log(f"  added {len(all_mp)} stable MP formulas")

    # Normalize to reduced form
    normalized: set[str] = set()
    for f in formulas:
        try:
            normalized.add(Composition(f).reduced_formula)
        except Exception:
            normalized.add(f)

    with out_path.open("wb") as f:
        pickle.dump(normalized, f)
    log(f"  saved → {out_path} ({len(normalized)} unique reduced formulas)")
    return normalized


# ---------------------------------------------------------------------------
# Step 5 — Phase diagram cache
# ---------------------------------------------------------------------------

def build_phase_diagram_cache(
    mpr,
    synth_records: list[dict],
) -> dict:
    log("Step 5: building phase diagram cache...")
    out_path = DATA_CACHE / "phase_diagrams.pkl"
    if out_path.exists():
        log(f"  cached → loading {out_path}")
        with out_path.open("rb") as f:
            return pickle.load(f)

    # Heavy imports only here
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

    # Compute union chemsys for each record (target + all precursors).
    # The PD must cover ALL species in the reaction for the energy
    # calculation to be valid. Skip records with templated/hyphenated
    # formulas (e.g. 'Cu2O-M2O5').
    chemsys_set: set[str] = set()
    skipped = 0
    for r in synth_records:
        target = r.get("target_formula")
        if not is_valid_concrete_formula(target):
            skipped += 1
            continue

        elements: set[str] = set()
        for el in Composition(target).elements:
            elements.add(str(el))
        for p in r.get("precursors", []):
            pf = p.get("formula")
            if not is_valid_concrete_formula(pf):
                continue
            try:
                for el in Composition(pf).elements:
                    elements.add(str(el))
            except Exception:
                pass
        if elements:
            chemsys_set.add("-".join(sorted(elements)))

    log(f"  {len(chemsys_set)} unique chemsystems "
        f"({skipped} records skipped: templated/invalid)")

    # Apply element-count cap — skip chemsystems with too many elements
    chemsys_filtered = [
        cs for cs in chemsys_set
        if len(cs.split("-")) <= PD_MAX_ELEMENTS
    ]
    n_capped = len(chemsys_set) - len(chemsys_filtered)
    log(f"  filtered to {len(chemsys_filtered)} chemsystems "
        f"(≤{PD_MAX_ELEMENTS} elements; {n_capped} excluded)")
    chemsys_set = set(chemsys_filtered)
    if SMOKE:
        chemsys_list = sorted(chemsys_set)[:SMOKE_SAMPLE_SIZE]
        log(f"  SMOKE: truncated to {len(chemsys_list)}")
    else:
        chemsys_list = sorted(chemsys_set)

    scheme = MaterialsProjectDFTMixingScheme()
    phase_diagrams: dict[str, PhaseDiagram] = {}

    # Resume from existing partial cache if present
    partial_path = DATA_CACHE / "phase_diagrams.partial.pkl"
    if partial_path.exists():
        with partial_path.open("rb") as f:
            phase_diagrams = pickle.load(f)
        log(f"  resuming from partial cache: {len(phase_diagrams)} PDs already built")

    # Per-chemsys timeout via SIGALRM — skips slow PDs gracefully.
    # Only works on Unix, which is fine since we're on a Linux server.
    import signal as _sig

    class _TimeoutError(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _TimeoutError("PD build exceeded timeout")

    # Per-chemsys fetch + PD construction with retry
    @with_retry(max_attempts=4, base_delay=5.0)
    def fetch_and_build(chemsys: str):
        elements = chemsys.split("-")
        entries = mpr.get_entries_in_chemsys(
            elements=elements,
            additional_criteria={
                "thermo_types": ["GGA_GGA+U", "R2SCAN"]
            },
        )
        entries = scheme.process_entries(entries)
        if not entries:
            return None
        # Wrap the actual hull computation in a timeout
        _sig.signal(_sig.SIGALRM, _timeout_handler)
        _sig.alarm(PD_TIMEOUT_SECS)
        try:
            pd = PhaseDiagram(entries)
        finally:
            _sig.alarm(0)   # always cancel
        return pd, len(entries)

    # Save checkpoint every N successful PDs (full run only — smoke is fast)
    CHECKPOINT_EVERY = 25
    builds_since_checkpoint = 0

    for i, chemsys in enumerate(chemsys_list, 1):
        if chemsys in phase_diagrams:
            continue   # already built (resumed run)
        try:
            result = fetch_and_build(chemsys)
            if result is None:
                log(f"  [{i}/{len(chemsys_list)}] {chemsys}: no entries")
                continue
            pd, n_entries = result
            phase_diagrams[chemsys] = pd
            builds_since_checkpoint += 1
            log(f"  [{i}/{len(chemsys_list)}] {chemsys}: "
                f"{n_entries} entries → PD built")

            if builds_since_checkpoint >= CHECKPOINT_EVERY:
                with partial_path.open("wb") as f:
                    pickle.dump(phase_diagrams, f)
                log(f"  checkpoint saved: {len(phase_diagrams)} PDs")
                builds_since_checkpoint = 0
        except _TimeoutError:
            log(f"  [{i}/{len(chemsys_list)}] {chemsys}: TIMEOUT "
                f"(>{PD_TIMEOUT_SECS}s) — skipped")
        except Exception as e:
            log(f"  [{i}/{len(chemsys_list)}] {chemsys}: SKIP after retries ({e})")

    with out_path.open("wb") as f:
        pickle.dump(phase_diagrams, f)
    # Remove partial cache once final cache is written
    if partial_path.exists():
        partial_path.unlink()
    log(f"  saved → {out_path} ({len(phase_diagrams)} PDs cached)")
    return phase_diagrams


# ---------------------------------------------------------------------------
# Step 6 — Dataset statistics
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    synth_records: list[dict],
    summary_records: list[dict],
    robocrys_records: list[dict],
    formula_set: set,
    phase_diagrams: dict,
) -> dict:
    log("Step 6: computing dataset statistics...")

    # Target formula stats
    target_formulas = [
        r["target_formula"] for r in synth_records if r.get("target_formula")
    ]
    target_counter = Counter(target_formulas)

    # Precursor stats
    precursor_formulas: list[str] = []
    for r in synth_records:
        for p in r.get("precursors", []):
            if p.get("formula"):
                precursor_formulas.append(p["formula"])
    precursor_counter = Counter(precursor_formulas)

    # Operation stats
    op_counter: Counter = Counter()
    n_ops_per_record: list[int] = []
    n_temps_per_record: list[int] = []
    for r in synth_records:
        ops = r.get("operations", [])
        n_ops_per_record.append(len(ops))
        n_temps = 0
        for op in ops:
            op_counter[op["type"]] += 1
            n_temps += len(op.get("heating_temperature", []))
        n_temps_per_record.append(n_temps)

    # Chemsys diversity
    chemsystems = set()
    for r in synth_records:
        cs = get_chemsys(r.get("target_formula", "") or "")
        if cs:
            chemsystems.add(cs)

    # Synthesis ↔ summary join coverage
    synth_target_set = set(target_formulas)
    summary_formula_set = {r["formula_pretty"] for r in summary_records}
    overlap = synth_target_set & summary_formula_set

    stats = {
        "n_synthesis_records":     len(synth_records),
        "n_summary_records":       len(summary_records),
        "n_robocrys_descriptions": len(robocrys_records),
        "n_formulas_in_set":       len(formula_set),
        "n_phase_diagrams":        len(phase_diagrams),
        "n_unique_targets":        len(target_counter),
        "n_unique_precursors":     len(precursor_counter),
        "n_unique_chemsystems":    len(chemsystems),
        "synth_summary_overlap":   len(overlap),
        "operation_type_counts":   dict(op_counter),
        "median_ops_per_route": (
            sorted(n_ops_per_record)[len(n_ops_per_record) // 2]
            if n_ops_per_record else 0
        ),
        "median_temps_per_route": (
            sorted(n_temps_per_record)[len(n_temps_per_record) // 2]
            if n_temps_per_record else 0
        ),
        "top_10_targets":          target_counter.most_common(10),
        "top_10_precursors":       precursor_counter.most_common(10),
    }

    out_path = DATA_RAW / "dataset_stats.json"
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2, default=str)
    log(f"  saved → {out_path}")
    return stats


def print_stats(stats: dict) -> None:
    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Synthesis records:       {stats['n_synthesis_records']}")
    print(f"Summary docs:            {stats['n_summary_records']}")
    print(f"Robocrys descriptions:   {stats['n_robocrys_descriptions']}")
    print(f"Formulas in set:         {stats['n_formulas_in_set']}")
    print(f"Phase diagrams cached:   {stats['n_phase_diagrams']}")
    print(f"Unique targets:          {stats['n_unique_targets']}")
    print(f"Unique precursors:       {stats['n_unique_precursors']}")
    print(f"Unique chemsystems:      {stats['n_unique_chemsystems']}")
    print(f"Synth↔Summary overlap:   {stats['synth_summary_overlap']}")
    print(f"Median ops per route:    {stats['median_ops_per_route']}")
    print(f"Median temps per route:  {stats['median_temps_per_route']}")
    print()
    print("Operation types:")
    for op, count in sorted(stats["operation_type_counts"].items(),
                             key=lambda x: -x[1]):
        print(f"  {op:25s}: {count}")
    print()
    print("Top 10 targets:")
    for formula, count in stats["top_10_targets"]:
        print(f"  {formula:25s}: {count}")
    print()
    print("Top 10 precursors:")
    for formula, count in stats["top_10_precursors"]:
        print(f"  {formula:25s}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dirs()
    mode = "SMOKE" if SMOKE else "FULL"
    log(f"=== MIRA data pull starting (mode: {mode}) ===")

    from mp_api.client import MPRester

    with MPRester(MP_API_KEY) as mpr:
        synth = pull_synthesis_records(mpr)
        summary = pull_summary_for_targets(mpr, synth)
        robocrys = pull_robocrys_descriptions(mpr, summary)
        formula_set = build_mp_formula_set(mpr, synth, summary)
        phase_diagrams = build_phase_diagram_cache(mpr, synth)

    stats = compute_dataset_stats(
        synth, summary, robocrys, formula_set, phase_diagrams
    )
    print_stats(stats)

    log("=== Done ===")


if __name__ == "__main__":
    main()