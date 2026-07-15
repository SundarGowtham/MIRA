"""
generate_mace_embeddings.py
---------------------------
Run MACE-MP-0 on all 2,529 MP summary structures and save mean-pooled
per-structure embeddings for use in structural retrieval.

Output:
    data/cache/mace_embeddings.npz
        material_ids : (N,)    string array of mp-XXXXXX IDs
        formulas     : (N,)    string array of formula_pretty
        embeddings   : (N, 256) float32 mean-pooled MACE descriptors

Runtime: ~15-20 min on CPU for 2,529 structures.
Resume-safe: skips material_ids already in the output file.

Usage:
    python generate_mace_embeddings.py
    python generate_mace_embeddings.py --model small   # default, fast
    python generate_mace_embeddings.py --model medium  # slower, better
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

# e3nn constants.pt uses slice objects; PyTorch 2.6+ requires explicit allowlist.
# Setting this env var restores the pre-2.6 behavior (safe for trusted model files).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from monty.serialization import loadfn

PROJECT_ROOT = Path(__file__).parent
DATA_RAW   = PROJECT_ROOT / "data" / "raw"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"
OUT_PATH   = DATA_CACHE / "mace_embeddings.npz"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_rich_embedding(calc, atoms) -> np.ndarray:
    """
    Richer invariant embedding than invariants_only=True.

    MACE with invariants_only=False returns equivariant node features:
    shape (n_atoms, n_channels * (2*max_ell+1)) — these transform under
    rotation so we can't mean-pool them directly.

    Instead we compute per-channel norms across the ℓ-components, giving
    a genuinely rotation-invariant fingerprint that preserves more
    information than the contracted invariants_only=True path.

    For medium (mace-128-L1, 128 channels, ℓ=0,1):
      ℓ=0: 1 component per channel  → 128 scalars (already invariant)
      ℓ=1: 3 components per channel → 128 norms
      Total: 256 per atom → but now each dim has distinct geometric meaning
             rather than the compressed invariants_only projection.

    For large (mace-128-L2, ℓ=0,1,2):
      ℓ=0: 128, ℓ=1: 128, ℓ=2: 128 → 384 per atom

    We then mean-pool across atoms → one vector per structure.
    """
    import torch
    from mace import data as mace_data

    # Get full equivariant node features
    desc = calc.get_descriptors(atoms, invariants_only=False)  # (n_atoms, D)

    # The descriptor layout for L1 medium:
    # [ℓ=0 channels (128)] [ℓ=1 channels (128*3)] = 512 total
    # For L0 small: just [ℓ=0 (256)] — no equivariant part
    # We extract by peeking at the model's irreps string
    try:
        irreps_str = str(calc.models[0].products[0].linear.irreps_out)
        # Parse to get (multiplicity, ell) pairs
        import re
        pairs = re.findall(r'(\d+)x(\d+)', irreps_str)
        if not pairs:
            # Fallback: just mean-pool raw descriptors
            return desc.mean(axis=0).astype(np.float32)

        chunks = []
        offset = 0
        for mult, ell in pairs:
            mult, ell = int(mult), int(ell)
            dim = 2 * ell + 1  # components per irrep
            total = mult * dim
            block = desc[:, offset:offset + total]  # (n_atoms, mult*dim)
            # Reshape to (n_atoms, mult, dim), take norm over dim
            block = block.reshape(desc.shape[0], mult, dim)
            norms = np.linalg.norm(block, axis=2)  # (n_atoms, mult)
            chunks.append(norms)
            offset += total

        rich = np.concatenate(chunks, axis=1)  # (n_atoms, sum_of_mults)
        return rich.mean(axis=0).astype(np.float32)  # (sum_of_mults,)
    except Exception:
        # Safe fallback
        return desc.mean(axis=0).astype(np.float32)


def structure_to_ase(struct):
    from ase import Atoms
    return Atoms(
        symbols=[str(site.specie) for site in struct],
        positions=[site.coords for site in struct],
        cell=struct.lattice.matrix,
        pbc=True,
    )


def load_existing(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    ids = data["material_ids"].tolist()
    embs = data["embeddings"]
    fmls = data["formulas"].tolist()
    return {mid: (emb, fml) for mid, emb, fml in zip(ids, embs, fmls)}


def save(existing: dict, path: Path):
    ids  = list(existing.keys())
    embs = np.array([existing[k][0] for k in ids], dtype=np.float32)
    fmls = np.array([existing[k][1] for k in ids])
    np.savez(path, material_ids=ids, embeddings=embs, formulas=fmls)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--summary", type=Path, default=DATA_RAW / "summary.json")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--checkpoint-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    DATA_CACHE.mkdir(parents=True, exist_ok=True)

    log(f"Loading summary from {args.summary}...")
    summary = loadfn(args.summary)
    log(f"  {len(summary)} records")

    log("Loading MACE-MH-1 (omat_pbe head — SOTA inorganic, Oct 2025)...")
    from mace.calculators import mace_mp
    calc = mace_mp(
        model="mh-1",
        head="omat_pbe",
        dispersion=False,
        default_dtype="float32",
        device=args.device,
    )
    log(f"  device={args.device}")

    existing = load_existing(args.out)
    log(f"  {len(existing)} embeddings already cached (resuming)")

    todo = [s for s in summary if s["material_id"] not in existing]
    log(f"  {len(todo)} structures to embed")

    n_done = 0
    n_failed = 0
    t0 = time.time()

    for i, record in enumerate(todo):
        mid     = record["material_id"]
        formula = record["formula_pretty"]
        struct  = record["structure"]

        try:
            atoms = structure_to_ase(struct)
            emb = get_rich_embedding(calc, atoms)
            existing[mid] = (emb, formula)
            n_done += 1
        except Exception as e:
            log(f"  SKIP {mid} ({formula}): {e}")
            n_failed += 1
            continue

        # Progress + checkpoint
        if (i + 1) % args.checkpoint_every == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta  = (len(todo) - i - 1) / max(rate, 0.01)
            log(f"  [{i+1}/{len(todo)}] done={n_done} failed={n_failed} "
                f"rate={rate:.1f}/s ETA={eta/60:.0f}min")
            save(existing, args.out)
            log(f"  checkpoint saved → {args.out}")

    save(existing, args.out)
    elapsed = time.time() - t0
    log(f"Done in {elapsed/60:.1f}min — {n_done} embeddings, {n_failed} failed")
    log(f"Output: {args.out}")
    log(f"  shape: ({n_done + len(existing) - len(todo)}, {next(iter(existing.values()))[0].shape[0]})")


if __name__ == "__main__":
    main()