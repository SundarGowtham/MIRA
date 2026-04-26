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

SMOKE = True   # Set False for full run

PROJECT_ROOT = Path(__file__).parent
DATA_RAW   = PROJECT_ROOT / "data" / "raw"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"

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

    The robocrys endpoint doesn't accept `material_ids=` as a filter.
    Two supported approaches:
      1. Bulk fetch all robocrys docs, filter client-side (slow but works)
      2. Per-material lookup via mpr.materials.robocrys.get_data_by_id()

    For the smoke run we use approach #2 (cheap, only ~10 materials).
    For the full run we use approach #1 once and reuse the cached file.

    Robocrys data is for the future RAG/alignment module, NOT for the
    validator or SFT data. Failures here are non-fatal — we log and skip.
    """
    log("Step 3: pulling robocrys descriptions...")
    out_path = DATA_RAW / "robocrys.json"
    if out_path.exists():
        log(f"  cached → loading {out_path}")
        return loadfn(out_path)

    material_ids = [r["material_id"] for r in summary_records]
    if not material_ids:
        log("  no material IDs — skipping")
        dumpfn([], out_path)
        return []

    records = []

    if SMOKE or len(material_ids) < 100:
        # Per-material lookup — cheap when we have few targets.
        # Different mp-api versions have different method signatures, so
        # try a few patterns before giving up on each material.
        log(f"  fetching {len(material_ids)} descriptions individually...")
        for mid in material_ids:
            doc = None
            try:
                # Newer API: search by material_id field via the keyword search
                results = mpr.materials.robocrys.search(keywords=[mid])
                # Filter to exact match (search may return fuzzy hits)
                for r in results:
                    if str(r.material_id) == mid:
                        doc = r
                        break
            except Exception:
                pass

            if doc is None:
                # Older API fallback
                try:
                    doc = mpr.materials.robocrys.get_data_by_id(mid)
                except Exception:
                    pass

            if doc is not None:
                desc = getattr(doc, "description", None)
                if desc:
                    records.append({
                        "material_id": str(doc.material_id),
                        "description": desc,
                    })
    else:
        # Full run — bulk fetch all robocrys docs and filter client-side.
        # This is one big query but only happens once (cached after).
        log("  bulk fetching all robocrys descriptions (one-time)...")
        try:
            all_docs = mpr.materials.robocrys.search(keywords=[""])
            wanted = set(material_ids)
            for doc in all_docs:
                if str(doc.material_id) in wanted:
                    desc = getattr(doc, "description", None)
                    if desc:
                        records.append({
                            "material_id": str(doc.material_id),
                            "description": desc,
                        })
        except Exception as e:
            log(f"  bulk fetch failed ({e}) — robocrys data will be empty")

    dumpfn(records, out_path)
    log(f"  saved → {out_path} ({len(records)} non-empty descriptions)")
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
    if SMOKE:
        chemsys_list = sorted(chemsys_set)[:SMOKE_SAMPLE_SIZE]
        log(f"  SMOKE: truncated to {len(chemsys_list)}")
    else:
        chemsys_list = sorted(chemsys_set)

    scheme = MaterialsProjectDFTMixingScheme()
    phase_diagrams: dict[str, PhaseDiagram] = {}

    for i, chemsys in enumerate(chemsys_list, 1):
        try:
            elements = chemsys.split("-")
            entries = mpr.get_entries_in_chemsys(
                elements=elements,
                additional_criteria={
                    "thermo_types": ["GGA_GGA+U", "R2SCAN"]
                },
            )
            entries = scheme.process_entries(entries)
            if entries:
                phase_diagrams[chemsys] = PhaseDiagram(entries)
                log(f"  [{i}/{len(chemsys_list)}] {chemsys}: "
                    f"{len(entries)} entries → PD built")
            else:
                log(f"  [{i}/{len(chemsys_list)}] {chemsys}: no entries")
        except Exception as e:
            log(f"  [{i}/{len(chemsys_list)}] {chemsys}: SKIP ({e})")

    with out_path.open("wb") as f:
        pickle.dump(phase_diagrams, f)
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