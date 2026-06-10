"""
diagnose_thermo.py
------------------
Trace exactly what happens when the validator tries to compute
reaction_energy_per_atom for SrTiO3 from SrCO3 + TiO2.

Run from the FT-CAPSTONE project root:
    uv run python diagnose_thermo.py

This will tell you which step is failing:
  1. Is "C-O-Sr-Ti" in pd_index.json?
  2. Does the shard file actually exist on disk?
  3. Can the shard be unpickled?
  4. Does the loaded PD contain SrTiO3, SrCO3, TiO2, CO2, O2 entries?
  5. Does ComputedReaction balance the equation?
  6. Is the resulting reaction energy reasonable?

Every step gets printed. Whichever step fails is your real bug.
"""

import json
import pickle
import sys
from pathlib import Path

from pymatgen.core import Composition

PROJECT_ROOT = Path(__file__).parent
PD_INDEX_FILE = PROJECT_ROOT / "data" / "cache" / "pd_index.json"

TARGET = "SrTiO3"
PRECURSORS = ["SrCO3", "TiO2"]
VOLATILES = ["CO2", "H2O", "O2", "N2", "NH3"]


def step(n, msg):
    print(f"\n{'='*60}\nSTEP {n}: {msg}\n{'='*60}")


# ── STEP 1: index ──────────────────────────────────────────────────────────
step(1, "Check pd_index.json")
if not PD_INDEX_FILE.exists():
    sys.exit(f"FAIL: {PD_INDEX_FILE} does not exist")

with PD_INDEX_FILE.open() as f:
    pd_index = json.load(f)
print(f"Loaded pd_index: {len(pd_index)} entries")

elements = {"C", "O", "Sr", "Ti"}
chemsys = "-".join(sorted(elements))
print(f"Looking for chemsys: {chemsys!r}")
print(f"In index: {chemsys in pd_index}")

if chemsys not in pd_index:
    # Try to find it case-insensitively or as a superset
    matches = [k for k in pd_index if set(k.split("-")) == elements]
    print(f"Exact element-set matches: {matches}")
    supersets = [k for k in pd_index if elements.issubset(set(k.split("-")))][:5]
    print(f"First 5 supersets: {supersets}")
    if not matches and not supersets:
        sys.exit("FAIL: no shard covers C-O-Sr-Ti. Run expand_pd_cache.py --only-C.")
    chemsys = matches[0] if matches else supersets[0]
    print(f"Using chemsys: {chemsys!r}")

shard_rel = pd_index[chemsys]
print(f"pd_index[{chemsys!r}] = {shard_rel!r}")

# ── STEP 2: shard file ─────────────────────────────────────────────────────
step(2, "Check the shard file on disk")
shard_path = PROJECT_ROOT / shard_rel
print(f"Resolved path: {shard_path}")
print(f"Exists: {shard_path.exists()}")
print(f"Is file: {shard_path.is_file() if shard_path.exists() else 'N/A'}")
if shard_path.exists():
    print(f"Size: {shard_path.stat().st_size:,} bytes")
else:
    sys.exit("FAIL: shard file does not exist at the path stored in pd_index. "
             "Your pd_index paths don't match where shards actually live.")

# ── STEP 3: unpickle ───────────────────────────────────────────────────────
step(3, "Try to unpickle the shard")
try:
    with shard_path.open("rb") as f:
        pd = pickle.load(f)
    print(f"OK. Type: {type(pd).__name__}")
    print(f"Has all_entries: {hasattr(pd, 'all_entries')}")
    print(f"Number of entries: {len(pd.all_entries)}")
    print(f"Number of stable entries: {len(pd.stable_entries)}")
except Exception as e:
    sys.exit(f"FAIL: pickle.load raised {type(e).__name__}: {e}\n"
             f"This is likely a pymatgen version mismatch between the machine "
             f"that built the shards (GCP) and your local machine.")

# ── STEP 4: entry lookup ───────────────────────────────────────────────────
step(4, "Look up each formula in the loaded PD")

def find_entry(pd, formula):
    target_red = Composition(formula).reduced_formula
    matches = [e for e in pd.all_entries if e.composition.reduced_formula == target_red]
    if not matches:
        return None
    return min(matches, key=lambda e: e.energy_per_atom)

found = {}
for f in [TARGET] + PRECURSORS + VOLATILES:
    e = find_entry(pd, f)
    if e is None:
        print(f"  ✗ {f:8s} NOT FOUND in PD entries")
    else:
        print(f"  ✓ {f:8s} found: {e.entry_id if hasattr(e, 'entry_id') else '?'}  "
              f"E/atom={e.energy_per_atom:.3f}  n_atoms={int(e.composition.num_atoms)}")
    found[f] = e

missing_critical = [f for f in [TARGET] + PRECURSORS if found[f] is None]
if missing_critical:
    sys.exit(f"FAIL: critical entries missing from PD: {missing_critical}\n"
             f"The C-O-Sr-Ti hull doesn't contain {missing_critical}. "
             f"Either the shard is incomplete or MP doesn't have these entries.")


# ── STEP 5: ComputedReaction ───────────────────────────────────────────────
step(5, "Try to build ComputedReaction")
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.analysis.reaction_calculator import ReactionError

reactant_entries = [found[f] for f in PRECURSORS]
volatile_entries = [found[v] for v in VOLATILES if found[v] is not None]
product_entries = [found[TARGET]] + volatile_entries

print(f"Reactants:    {[e.composition.reduced_formula for e in reactant_entries]}")
print(f"Products:     {[e.composition.reduced_formula for e in product_entries]}")

try:
    reaction = ComputedReaction(reactant_entries, product_entries)
    print(f"OK. Reaction: {reaction}")
except ReactionError as e:
    sys.exit(f"FAIL: ReactionError: {e}")
except Exception as e:
    sys.exit(f"FAIL: {type(e).__name__}: {e}")

target_comp_reduced = Composition(found[TARGET].composition.reduced_formula)
print(f"Querying with reduced composition: {target_comp_reduced}")
target_coeff = reaction.get_coeff(target_comp_reduced)
print(f"target_coeff for {TARGET}: {target_coeff}")

if target_coeff <= 1e-6:
    sys.exit(f"FAIL: target_coeff is {target_coeff} (must be > 0). "
             f"Reaction is balanced wrong direction or doesn't actually produce target.")

# ── STEP 6: energy ─────────────────────────────────────────────────────────
step(6, "Compute reaction energy")
delta_E_total = reaction.calculated_reaction_energy
atoms_target = target_coeff * target_comp_reduced.num_atoms
delta_E_per_atom = delta_E_total / atoms_target
print(f"delta_E_total:    {delta_E_total:.4f} eV")
print(f"atoms_target:     {atoms_target:.2f}")
print(f"delta_E_per_atom: {delta_E_per_atom:.4f} eV/atom")

if delta_E_per_atom < -0.5 or delta_E_per_atom > 0.5:
    print("WARNING: reaction energy is unphysically large. Something may be off.")
else:
    print("✓ Reaction energy looks reasonable. The validator should score this > 0.5.")
print()
print("If you got here, the math works. The bug must be in how the worker calls "
      "the validator, not the validator itself. Check that precursors are getting "
      "passed correctly (as list of tuples, not list of dicts).")