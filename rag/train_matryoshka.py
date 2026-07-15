"""
train_matryoshka.py
-------------------
Fine-tune a Matryoshka projection on top of MACE-MH-1 embeddings.

Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022):
  A single linear projection W: R^1024 -> R^D is trained such that
  the first d dims are meaningful for ANY d in {32, 64, 128, 256, 512, 1024}.

  Loss = sum over d in dims of MSELoss(cosine_sim at d, label)

  This lets Stage 1 retrieval use 32-dim for fast ANN search,
  Stage 2 re-rank with 256-dim, without re-embedding anything.

Positive pairs:  same crystal system + same chemsys  (label = 1.0)
Hard negatives:  same crystal system, different chemsys (label = 0.0)
Easy negatives:  different crystal system              (label = -1.0)

Output:
    data/cache/matryoshka_projection.pt   — W matrix (1024 x 1024)
    data/cache/matryoshka_embeddings.npz  — projected embeddings, same shape

Usage:
    python train_matryoshka.py
    python train_matryoshka.py --epochs 50 --lr 1e-3
"""

from __future__ import annotations
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

DATA_CACHE = Path("data/cache")
DATA_RAW   = Path("data/raw")

MATRYOSHKA_DIMS = [32, 64, 128, 256, 512, 1024]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Data loading + pair construction
# ---------------------------------------------------------------------------

def load_embeddings():
    data = np.load(DATA_CACHE / "mace_embeddings.npz", allow_pickle=True)
    ids    = data["material_ids"].tolist()
    embs   = torch.tensor(data["embeddings"], dtype=torch.float32)
    fmls   = data["formulas"].tolist()
    return ids, embs, fmls


def build_metadata(ids: list[str], fmls: list[str]):
    """Build chemsys and crystal_system lookup from summary.json."""
    from monty.serialization import loadfn
    from pymatgen.core import Composition

    summary = loadfn(DATA_RAW / "summary.json")
    mid_to_meta = {
        s["material_id"]: {
            "crystal_system": s.get("crystal_system", "Unknown"),
            "chemsys": "-".join(sorted(
                str(e) for e in Composition(s["formula_pretty"]).elements
            )) if s.get("formula_pretty") else "Unknown",
        }
        for s in summary
    }
    crystal_systems = [mid_to_meta.get(mid, {}).get("crystal_system", "Unknown") for mid in ids]
    chemsys         = [mid_to_meta.get(mid, {}).get("chemsys", "Unknown") for mid in ids]
    return crystal_systems, chemsys


def build_pairs(crystal_systems, chemsys, n_pairs=50000):
    """
    Sample training pairs with labels:
      +1.0  same chemsys (structurally very similar)
       0.0  same crystal system, different chemsys (loosely similar)
      -1.0  different crystal system (dissimilar)
    """
    n = len(crystal_systems)
    idx_by_cs    = {}  # crystal_system -> [indices]
    idx_by_chem  = {}  # chemsys -> [indices]
    for i, (cs, ch) in enumerate(zip(crystal_systems, chemsys)):

        # print(f"i: {i} || cs: {cs} || ch: {ch}")
        idx_by_cs.setdefault(cs, []).append(i)
        idx_by_chem.setdefault(ch, []).append(i)

    pairs = []
    rng = random.Random(42)

    print("-----------------------------------")

    # Positives: same chemsys
    for ch, idxs in idx_by_chem.items():
        if len(idxs) < 2:
            # print(f"idxs is less than 2: {idxs}")
            continue
        
        # print(f"ch: {ch} || idxs: {idxs}")
        for _ in range(min(3, len(idxs))):
            i, j = rng.sample(idxs, 2)
            pairs.append((i, j, 1.0))

    # Hard negatives: same crystal system, different chemsys
    for cs, idxs in idx_by_cs.items():
        if len(idxs) < 4:
            continue
        # group by chemsys within this crystal system
        chem_groups = {}
        for i in idxs:
            chem_groups.setdefault(chemsys[i], []).append(i)
        cg_list = list(chem_groups.values())
        if len(cg_list) < 2:
            continue
        for _ in range(min(5, len(cg_list))):
            g1, g2 = rng.sample(cg_list, 2)
            i, j = rng.choice(g1), rng.choice(g2)
            pairs.append((i, j, 0.0))

    # Easy negatives: different crystal systems
    cs_list = list(idx_by_cs.keys())
    if len(cs_list) >= 2:
        for _ in range(len(pairs)):  # balance
            cs1, cs2 = rng.sample(cs_list, 2)
            i = rng.choice(idx_by_cs[cs1])
            j = rng.choice(idx_by_cs[cs2])
            pairs.append((i, j, -1.0))

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    log(f"  Built {len(pairs)} pairs  "
        f"(pos={sum(1 for p in pairs if p[2]>0.5)}  "
        f"hard_neg={sum(1 for p in pairs if -0.1<p[2]<0.1)}  "
        f"easy_neg={sum(1 for p in pairs if p[2]<-0.5)})")
    return pairs


# ---------------------------------------------------------------------------
# Matryoshka model
# ---------------------------------------------------------------------------

class MatryoshkaProjection(nn.Module):
    """
    A single linear layer W: R^1024 -> R^1024.
    Matryoshka property: first d dims of W(x) give the best d-dim embedding
    for any d in MATRYOSHKA_DIMS.
    """
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        # Initialize as identity so training starts from MACE representations
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor, d: int | None = None) -> torch.Tensor:
        z = self.proj(x)
        z = F.normalize(z, dim=-1)
        if d is not None:
            z = z[..., :d]
            z = F.normalize(z, dim=-1)
        return z


def matryoshka_loss(model, xi, xj, labels, dims=MATRYOSHKA_DIMS):
    """
    Sum of cosine-similarity MSE losses across all Matryoshka dimensions.
    Each dimension level is equally weighted.
    """
    total = 0.0
    for d in dims:
        zi = model(xi, d=d)
        zj = model(xj, d=d)
        sim = (zi * zj).sum(dim=-1)  # cosine sim (already normalized)
        total = total + F.mse_loss(sim, labels)
    return total / len(dims)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    log("Loading embeddings...")
    ids, embs, fmls = load_embeddings()
    log(f"  {len(ids)} embeddings, dim={embs.shape[1]}")

    log("Building metadata and pairs...")
    crystal_systems, chemsys = build_metadata(ids, fmls)
    pairs = build_pairs(crystal_systems, chemsys, n_pairs=args.n_pairs)

    dim = embs.shape[1]
    model = MatryoshkaProjection(dim=dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Convert pairs to tensors
    idx_i  = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    idx_j  = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    labels = torch.tensor([p[2] for p in pairs], dtype=torch.float32)

    log(f"Training Matryoshka projection — {args.epochs} epochs, lr={args.lr}")
    log(f"  dims={MATRYOSHKA_DIMS}  pairs={len(pairs)}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Mini-batch SGD
        perm = torch.randperm(len(pairs))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(pairs), args.batch_size):
            batch = perm[start:start + args.batch_size]
            xi = embs[idx_i[batch]]
            xj = embs[idx_j[batch]]
            lb = labels[batch]

            optimizer.zero_grad()
            loss = matryoshka_loss(model, xi, xj, lb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % max(1, args.epochs // 10) == 0:
            log(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.5f}  "
                f"best={best_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # Load best weights
    model.load_state_dict(best_state)

    # Save projection weights
    proj_path = DATA_CACHE / "matryoshka_projection.pt"
    torch.save({"state_dict": model.state_dict(), "dim": dim,
                "dims": MATRYOSHKA_DIMS}, proj_path)
    log(f"Saved projection → {proj_path}")

    # Project all embeddings and save
    model.eval()
    with torch.no_grad():
        projected = model.proj(embs).numpy().astype(np.float32)
        # L2-normalize
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / np.maximum(norms, 1e-8)

    out_path = DATA_CACHE / "matryoshka_embeddings.npz"
    data = np.load(DATA_CACHE / "mace_embeddings.npz", allow_pickle=True)
    np.savez(out_path,
             material_ids=data["material_ids"],
             formulas=data["formulas"],
             embeddings=projected.astype(np.float32))
    log(f"Saved projected embeddings → {out_path}  shape={projected.shape}")

    # Quick sanity check: BaTiO3 vs SrTiO3 similarity
    formulas_list = data["formulas"].tolist()
    def find(f):
        return next((i for i, x in enumerate(formulas_list) if f in x), None)
    i_ba, i_sr = find("BaTiO3"), find("SrTiO3")
    if i_ba is not None and i_sr is not None:
        for d in MATRYOSHKA_DIMS:
            zi = projected[i_ba, :d] / np.linalg.norm(projected[i_ba, :d])
            zj = projected[i_sr, :d] / np.linalg.norm(projected[i_sr, :d])
            sim = float(np.dot(zi, zj))
            log(f"  BaTiO3 vs SrTiO3 @ d={d:4d}: cosine={sim:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--n-pairs",    type=int,   default=50000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)