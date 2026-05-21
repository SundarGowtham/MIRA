uv run python3 -c "
from monty.serialization import loadfn
from mace.calculators import mace_mp

summary = loadfn('data/raw/summary.json')
s = summary[0]
struct = s['structure']

print(f'Testing MACE on {s[\"formula_pretty\"]} ({s[\"material_id\"]})')
print(f'Sites: {len(struct)}')

calc = mace_mp(model='small', dispersion=False, default_dtype='float32', device='cpu')

import numpy as np
from ase import Atoms

# Convert pymatgen Structure to ASE Atoms
ase_atoms = Atoms(
    symbols=[str(site.specie) for site in struct],
    positions=[site.coords for site in struct],
    cell=struct.lattice.matrix,
    pbc=True,
)

# Get descriptors (the embedding)
desc = calc.get_descriptors(ase_atoms, invariants_only=True)
print(f'Descriptor shape: {desc.shape}')
print(f'Mean embedding (first 5): {desc.mean(axis=0)[:5]}')
print('MACE works.')
"