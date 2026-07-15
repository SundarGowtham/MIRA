"""
build_retrieval_index.py
------------------------
Build a Qdrant collection for 3-stage structural + text retrieval.

Each point in the collection represents one MP-matched material and carries:
  - structural vector: 1024-dim Matryoshka MACE-MH-1 embedding (L2-normalized)
  - text vector:       768-dim nomic-embed-text-v1.5 of robocrys description
  - payload:           material_id, formula, crystal_system, synthesis records

Stage 1 uses the 32-dim Matryoshka subspace (fast ANN recall, k=20)
Stage 2 re-ranks with ColBERT (separate script)
Stage 3 cross-encoder final reranking (separate script)

Output:
    Local Qdrant DB at data/cache/qdrant_db/
    Collection name: mira_materials

Usage:
    pip install qdrant-client sentence-transformers
    python build_retrieval_index.py
    python build_retrieval_index.py --reset  # rebuild from scratch
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
from monty.serialization import loadfn

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

DATA_CACHE    = Path("data/cache")
DATA_RAW      = Path("data/raw")
QDRANT_PATH   = DATA_CACHE / "qdrant_db"
COLLECTION    = "mira_materials"

# Matryoshka subspace used for Stage 1 ANN (fast recall)
STAGE1_DIM    = 32
# Full structural embedding dim
STRUCT_DIM    = 1024
# Text embedding dim (nomic-embed-text-v1.5)
TEXT_DIM      = 768


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_structural_embeddings() -> tuple[list[str], np.ndarray, list[str]]:
    path = DATA_CACHE / "matryoshka_embeddings.npz"
    if not path.exists():
        log("  matryoshka_embeddings.npz not found, using raw MACE embeddings")
        path = DATA_CACHE / "mace_embeddings.npz"
    data = np.load(path, allow_pickle=True)
    ids  = data["material_ids"].tolist()
    embs = data["embeddings"].astype(np.float32)
    fmls = data["formulas"].tolist()
    return ids, embs, fmls


def load_robocrys() -> dict[str, str]:
    """Load material_id -> robocrys description."""
    path = DATA_RAW / "robocrys.json"
    if not path.exists():
        log("  robocrys.json not found, text vectors will be empty")
        return {}
    records = loadfn(path)
    return {r["material_id"]: r.get("description", "") for r in records}


def load_summary_meta() -> dict[str, dict]:
    """Load material_id -> metadata dict."""
    summary = loadfn(DATA_RAW / "summary.json")
    return {
        s["material_id"]: {
            "formula":        s["formula_pretty"],
            "crystal_system": s.get("crystal_system", "Unknown"),
            "spacegroup":     s.get("spacegroup_number"),
            "energy_hull":    s.get("energy_above_hull"),
            "is_stable":      s.get("is_stable"),
            "nsites":         s.get("nsites"),
        }
        for s in summary
    }


def load_synthesis_index() -> dict[str, list[dict]]:
    """
    Build a lookup: reduced_formula -> list of synthesis records.
    Used to attach synthesis context to each retrieval result.
    """
    from pymatgen.core import Composition
    synth = loadfn(DATA_RAW / "synthesis.json")
    index: dict[str, list[dict]] = {}
    for r in synth:
        f = r.get("target_formula")
        if not f:
            continue
        try:
            key = Composition(f).reduced_formula
        except Exception:
            key = f
        index.setdefault(key, []).append({
            "target_formula": f,
            "precursors":     r.get("precursors", []),
            "operations":     r.get("operations", []),
            "doi":            r.get("doi", ""),
        })
    return index


def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Embed a list of texts using nomic-embed-text-v1.5."""
    from sentence_transformers import SentenceTransformer
    log("  Loading nomic-embed-text-v1.5...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    # Truncate texts to 400 tokens worth of chars (~1600 chars) to avoid
    # the 192GB attention buffer OOM from very long robocrys descriptions
    MAX_CHARS = 1600
    texts = [t[:MAX_CHARS] if t else f"crystal structure" for t in texts]

    log(f"  Embedding {len(texts)} texts (max {MAX_CHARS} chars each)...")
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        prompt_name="document",
    )
    return embs.astype(np.float32)


# ---------------------------------------------------------------------------
# Qdrant collection setup
# ---------------------------------------------------------------------------

def setup_collection(client, reset: bool):
    from qdrant_client.models import (
        VectorParams, Distance,
        HnswConfigDiff, OptimizersConfigDiff,
    )

    exists = any(c.name == COLLECTION for c in client.get_collections().collections)

    if exists and reset:
        log(f"  Deleting existing collection '{COLLECTION}'...")
        client.delete_collection(COLLECTION)
        exists = False

    if not exists:
        log(f"  Creating collection '{COLLECTION}'...")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                # Stage 1: fast ANN on 32-dim Matryoshka subspace
                "struct_32": VectorParams(
                    size=STAGE1_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                ),
                # Full structural embedding for later re-ranking
                "struct_full": VectorParams(
                    size=STRUCT_DIM,
                    distance=Distance.COSINE,
                ),
                # Text embedding from robocrys description
                "text": VectorParams(
                    size=TEXT_DIM,
                    distance=Distance.COSINE,
                ),
            },
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )
        log(f"  Collection created.")
    else:
        log(f"  Collection '{COLLECTION}' already exists (use --reset to rebuild)")


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(args):
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    setup_collection(client, args.reset)

    # Check how many points already indexed
    count = client.count(COLLECTION).count
    if count > 0 and not args.reset:
        log(f"  {count} points already in collection. Use --reset to rebuild.")
        return

    log("Loading structural embeddings...")
    ids, struct_embs, fmls = load_structural_embeddings()
    log(f"  {len(ids)} materials, dim={struct_embs.shape[1]}")

    log("Loading metadata...")
    meta   = load_summary_meta()
    robocrys = load_robocrys()
    synth_idx = load_synthesis_index()
    from pymatgen.core import Composition

    log("Generating text embeddings from robocrys descriptions...")
    texts = [robocrys.get(mid, f"Crystal structure of {fml}")
             for mid, fml in zip(ids, fmls)]
    text_embs = embed_texts(texts, batch_size=args.text_batch_size)
    log(f"  text embeddings shape: {text_embs.shape}")

    log("Uploading points to Qdrant...")
    points = []
    for i, (mid, fml) in enumerate(zip(ids, fmls)):
        s_emb = struct_embs[i]
        t_emb = text_embs[i]

        # Matryoshka: first 32 dims for Stage 1 ANN
        s32   = s_emb[:STAGE1_DIM].copy()
        s32   = s32 / np.maximum(np.linalg.norm(s32), 1e-8)

        # Synthesis records for this material
        try:
            reduced = Composition(fml).reduced_formula
        except Exception:
            reduced = fml
        synth_records = synth_idx.get(reduced, [])
        # Keep at most 3 synthesis records in payload (size limit)
        synth_payload = [
            {
                "target_formula": r["target_formula"],
                "precursors": [p.get("formula", "") for p in r["precursors"][:6]],
                "n_ops": len(r["operations"]),
                "doi": r["doi"],
            }
            for r in synth_records[:3]
        ]

        m = meta.get(mid, {})
        payload = {
            "material_id":    mid,
            "formula":        fml,
            "crystal_system": m.get("crystal_system", "Unknown"),
            "spacegroup":     m.get("spacegroup"),
            "energy_hull":    m.get("energy_hull"),
            "is_stable":      m.get("is_stable"),
            "nsites":         m.get("nsites"),
            "has_robocrys":   mid in robocrys,
            "n_synth_records": len(synth_records),
            "synth_records":  synth_payload,
        }

        points.append(PointStruct(
            id=i,
            vector={
                "struct_32":   s32.tolist(),
                "struct_full": s_emb.tolist(),
                "text":        t_emb.tolist(),
            },
            payload=payload,
        ))

    # Upload in batches
    batch_size = args.upload_batch_size
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)
        log(f"  uploaded {min(start + batch_size, len(points))}/{len(points)}")

    final_count = client.count(COLLECTION).count
    log(f"Done. Collection '{COLLECTION}' has {final_count} points.")
    log(f"Qdrant DB at: {QDRANT_PATH}")

    # Quick smoke test
    log("\nSmoke test — querying BaTiO3...")
    _smoke_test(client, struct_embs, text_embs, ids, fmls)


def _smoke_test(client, struct_embs, text_embs, ids, fmls):
    """Find BaTiO3 and retrieve its nearest structural neighbors."""
    target = next((i for i, f in enumerate(fmls) if "BaTiO3" in f), None)
    if target is None:
        log("  BaTiO3 not in index — skipping smoke test")
        return

    q32 = struct_embs[target, :STAGE1_DIM].copy()
    q32 = q32 / np.maximum(np.linalg.norm(q32), 1e-8)

    results = client.search(
        collection_name=COLLECTION,
        query_vector=("struct_32", q32.tolist()),
        limit=6,
        with_payload=True,
    )
    log(f"  Query: {fmls[target]}")
    log(f"  Top-5 structural neighbors (32-dim Matryoshka):")
    for r in results[1:6]:
        log(f"    {r.payload['formula']:20s}  "
            f"crystal_sys={r.payload['crystal_system']:12s}  "
            f"score={r.score:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reset",             action="store_true",
                   help="Delete and rebuild the collection from scratch.")
    p.add_argument("--text-batch-size",   type=int, default=64)
    p.add_argument("--upload-batch-size", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args)