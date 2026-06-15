"""
prep_viz_lora_data.py
----------------
Extracts geometric statistics from trained LoRA adapter pairs (r=16 vs r=32)
and writes a single JSON consumed by the trajectory visualization HTML.

What it dumps, per module, per rank:
  1. singular_values[]        — top-k singular values of ΔW = BA (capacity profile)
  2. col_norms[]              — L2 norm of each column of A (input-side activation)
  3. row_norms[]              — L2 norm of each row of B (output-side push)
  4. principal_angles[]       — 5 angles between r=16 and r=32 top-5 subspaces
  5. shared_basis_coords      — top-5 right singular vectors of EACH rank projected
                                into a shared 3D basis (the global PCA of the union),
                                so 5 points per rank can be plotted in the same 3D
                                scene and visually compared.

Output schema:
{
  "checkpoint_r16": "...",
  "checkpoint_r32": "...",
  "n_layers": 36,
  "modules": {
    "model.layers.0.self_attn.q_proj": {
      "r16": {"singular_values":[...], "col_norms":[...], "row_norms":[...]},
      "r32": {"singular_values":[...], "col_norms":[...], "row_norms":[...]},
      "principal_angles_deg": [...],
      "shared_basis_coords_r16": [[x,y,z], ...],   # 5 points
      "shared_basis_coords_r32": [[x,y,z], ...]
    },
    ...
  }
}

Usage:
  uv run python prep_viz_data.py \\
      runs/sft-qlora-v3-rank16-seed42/final \\
      runs/sft-qlora-v3-rank32-seed42/final \\
      --output lora_viz/adapter_geometry_seed42.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def load_adapter(checkpoint_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    safetensors_path = checkpoint_path / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {checkpoint_path}")
    state = load_file(str(safetensors_path))

    matrices: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for full_key, tensor in state.items():
        for ab in ("A", "B"):
            tag = f"lora_{ab}"
            if tag in full_key:
                module_name = full_key
                for suffix in (f".{tag}.weight", f".{tag}.default.weight"):
                    module_name = module_name.replace(suffix, "")
                module_name = module_name.replace("base_model.model.", "")
                matrices[module_name][ab] = tensor
                break

    return {k: v for k, v in matrices.items() if "A" in v and "B" in v}


@torch.no_grad()
def module_stats(A: torch.Tensor, B: torch.Tensor, device: str,
                 top_sv: int = 32, top_vecs: int = 5) -> dict:
    """
    Compute the statistics we need for one (A, B) pair.
    Returns a dict of plain-Python lists, JSON-safe.
    """
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)
    delta_w = B @ A  # shape: [d_out, d_in]

    # ---- singular value decomposition of ΔW (capacity profile) ----
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    k = min(top_sv, S.shape[0])
    singular_values = S[:k].cpu().tolist()

    # ---- column-wise L2 norm of A (one number per input direction) ----
    # A shape: [r, d_in]. Column norm = how much each input dim feeds into adapter.
    # For a 4096-d input space, that's 4096 values — we subsample by averaging
    # contiguous blocks down to 128 for visualization.
    col_norms_full = A.norm(dim=0).cpu()                 # [d_in]
    col_norms = aggregate(col_norms_full, target_len=128)

    # ---- row-wise L2 norm of B (one number per output direction) ----
    # B shape: [d_out, r]. Row norm = how strongly each output dim is pushed.
    row_norms_full = B.norm(dim=1).cpu()                 # [d_out]
    row_norms = aggregate(row_norms_full, target_len=128)

    # ---- top-k right singular vectors (the actual "directions" learned) ----
    # Vh shape: [min(d_out,d_in), d_in]. Take first top_vecs rows.
    V = Vh[:top_vecs].cpu()  # [top_vecs, d_in]

    return {
        "singular_values": [round(float(s), 6) for s in singular_values],
        "col_norms":       [round(float(v), 5) for v in col_norms.tolist()],
        "row_norms":       [round(float(v), 5) for v in row_norms.tolist()],
        "_V_for_basis":    V,   # kept in-memory for shared-basis computation below
        "_delta_w_norm":   float(delta_w.norm().item()),
    }


def aggregate(t: torch.Tensor, target_len: int = 128) -> torch.Tensor:
    """Aggregate a long 1D tensor down to target_len by mean-pooling contiguous blocks."""
    n = t.shape[0]
    if n <= target_len:
        return t
    block = n // target_len
    trimmed = t[: block * target_len]
    return trimmed.reshape(target_len, block).mean(dim=1)


@torch.no_grad()
def principal_angles(V1: torch.Tensor, V2: torch.Tensor, k: int = 5) -> list[float]:
    """
    V1, V2 are row-vectors of right-singular-vectors (each row = one direction).
    Principal angles between row spans.
    """
    # Orthonormalize (they should already be orthonormal from SVD)
    Q1, _ = torch.linalg.qr(V1[:k].T)  # [d, k]
    Q2, _ = torch.linalg.qr(V2[:k].T)
    M = Q1.T @ Q2
    s = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles = torch.acos(s) * 180.0 / math.pi
    return [round(float(a), 3) for a in angles.tolist()]


@torch.no_grad()
def shared_basis_projection(V1: torch.Tensor, V2: torch.Tensor, k: int = 5,
                             out_dim: int = 3) -> tuple[list, list]:
    """
    Project the top-k right singular vectors of TWO adapters into a shared
    `out_dim`-dimensional basis. The shared basis is the top out_dim left
    singular vectors of the concatenated [V1; V2] matrix — i.e., the
    directions of greatest combined variation.

    Returns two lists of k points each, each point an out_dim-dim coord,
    suitable for plotting in a shared 3D scene.
    """
    # V1: [k, d], V2: [k, d]
    stacked = torch.cat([V1[:k], V2[:k]], dim=0)  # [2k, d]
    # PCA of the stacked rows: SVD, then take top-out_dim right singular vectors
    _, _, Vh = torch.linalg.svd(stacked, full_matrices=False)
    basis = Vh[:out_dim]  # [out_dim, d]

    coords_1 = (V1[:k] @ basis.T).cpu().tolist()  # [k, out_dim]
    coords_2 = (V2[:k] @ basis.T).cpu().tolist()
    return (
        [[round(float(x), 5) for x in row] for row in coords_1],
        [[round(float(x), 5) for x in row] for row in coords_2],
    )


def parse_module_key(name: str) -> tuple[int, str] | None:
    m = re.match(r'model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)', name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_r16", type=Path)
    parser.add_argument("ckpt_r32", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to output JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-sv",   type=int, default=32,
                        help="How many singular values to keep per module")
    parser.add_argument("--top-vecs", type=int, default=5,
                        help="How many singular vectors to keep for the shared-basis plot")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Loading r=16 adapter from {args.ckpt_r16}")
    mats_r16 = load_adapter(args.ckpt_r16)
    print(f"  {len(mats_r16)} modules")

    print(f"Loading r=32 adapter from {args.ckpt_r32}")
    mats_r32 = load_adapter(args.ckpt_r32)
    print(f"  {len(mats_r32)} modules")

    shared = sorted(set(mats_r16) & set(mats_r32))
    print(f"  {len(shared)} shared modules")

    # Sanity check ranks
    r16_actual = mats_r16[shared[0]]["A"].shape[0]
    r32_actual = mats_r32[shared[0]]["A"].shape[0]
    print(f"  configured ranks: r={r16_actual} / r={r32_actual}")

    n_layers = 0
    modules_out: dict[str, dict] = {}

    for i, name in enumerate(shared):
        parsed = parse_module_key(name)
        if parsed is None:
            continue
        layer, proj = parsed
        n_layers = max(n_layers, layer + 1)

        stats_r16 = module_stats(mats_r16[name]["A"], mats_r16[name]["B"],
                                 args.device, args.top_sv, args.top_vecs)
        stats_r32 = module_stats(mats_r32[name]["A"], mats_r32[name]["B"],
                                 args.device, args.top_sv, args.top_vecs)

        # Principal angles + shared basis projection
        V1 = stats_r16.pop("_V_for_basis")
        V2 = stats_r32.pop("_V_for_basis")
        del stats_r16["_delta_w_norm"], stats_r32["_delta_w_norm"]
        pa = principal_angles(V1, V2, k=args.top_vecs)
        coords_r16, coords_r32 = shared_basis_projection(V1, V2,
                                                          k=args.top_vecs, out_dim=3)

        modules_out[name] = {
            "layer": layer,
            "proj": proj,
            "configured_rank_r16": int(mats_r16[name]["A"].shape[0]),
            "configured_rank_r32": int(mats_r32[name]["A"].shape[0]),
            "r16": stats_r16,
            "r32": stats_r32,
            "principal_angles_deg": pa,
            "shared_basis_coords_r16": coords_r16,
            "shared_basis_coords_r32": coords_r32,
        }

        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(shared)}] processed")

    # Global normalization hints for the visualization layer
    all_max_sv = max(max(m["r16"]["singular_values"][:1] + m["r32"]["singular_values"][:1])
                     for m in modules_out.values())
    all_max_col = max(max(max(m["r16"]["col_norms"]), max(m["r32"]["col_norms"]))
                      for m in modules_out.values())
    all_max_row = max(max(max(m["r16"]["row_norms"]), max(m["r32"]["row_norms"]))
                      for m in modules_out.values())

    payload = {
        "checkpoint_r16": str(args.ckpt_r16),
        "checkpoint_r32": str(args.ckpt_r32),
        "rank_r16": r16_actual,
        "rank_r32": r32_actual,
        "n_layers": n_layers,
        "n_modules": len(modules_out),
        "col_norm_aggregation": "mean-pooled to 128 bins from 4096-d input",
        "row_norm_aggregation": "mean-pooled to 128 bins from 4096-d output",
        "global_normalization": {
            "max_singular_value": all_max_sv,
            "max_col_norm":       all_max_col,
            "max_row_norm":       all_max_row,
        },
        "modules": modules_out,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f)

    import os
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nWrote {args.output}  ({size_kb:.1f} KB, {len(modules_out)} modules)")


if __name__ == "__main__":
    main()