"""
lora_spectral_analysis.py
-------------------------
GPU-accelerated spectral diagnostics for LoRA adapters. Answers questions
that scalar rewards can't:

  1. What rank is each ΔW = BA actually using? (effective rank via entropy)
  2. Do r=16 and r=32 learn the same subspace? (principal angles)
  3. How fast does spectral information decay? (power-law fit)
  4. Which layers want more rank? (per-layer rollup)

All computations on GPU via torch.linalg.svd — much faster than CPU numpy.

Usage:
    # Single adapter — full spectral profile
    python lora_spectral_analysis.py \\
        --checkpoint runs/sft-qlora-v3-rank16-seed42/final

    # Compare two adapters — subspace alignment + side-by-side
    python lora_spectral_analysis.py \\
        --checkpoint runs/sft-qlora-v3-rank16-seed42/final \\
        --compare runs/sft-qlora-v3-rank32-seed42/final

    # Scan all checkpoints in a run for training-dynamics view
    python lora_spectral_analysis.py \\
        --checkpoint-dir runs/sft-qlora-v3-rank16-seed42
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


# ============================================================================
# Loading adapters
# ============================================================================

def load_adapter_weights(checkpoint_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load LoRA A and B matrices from a PEFT checkpoint."""
    safetensors_path = checkpoint_path / "adapter_model.safetensors"
    bin_path = checkpoint_path / "adapter_model.bin"

    if safetensors_path.exists():
        state = load_file(str(safetensors_path))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No adapter weights in {checkpoint_path}. "
            f"Expected adapter_model.safetensors or adapter_model.bin"
        )

    matrices: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for full_key, tensor in state.items():
        for ab in ("A", "B"):
            tag = f"lora_{ab}"
            if tag in full_key:
                module_name = full_key
                for suffix in (f".{tag}.weight", f".{tag}.default.weight"):
                    module_name = module_name.replace(suffix, "")
                matrices[module_name][ab] = tensor
                break

    cleaned = {}
    for name, mats in matrices.items():
        clean_name = name.replace("base_model.model.", "")
        if "A" in mats and "B" in mats:
            cleaned[clean_name] = mats
    return cleaned


# ============================================================================
# Spectral primitives (GPU)
# ============================================================================

@torch.no_grad()
def compute_delta_w(A: torch.Tensor, B: torch.Tensor, device: str) -> torch.Tensor:
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)
    return B @ A


@torch.no_grad()
def singular_values(delta_w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(delta_w)


def effective_rank(sigma: torch.Tensor, eps: float = 1e-12) -> float:
    """
    exp(H) of normalized squared singular value spectrum.
    Rank-r matrix with r equal σ → effective rank exactly r.
    """
    s2 = sigma.pow(2)
    p = s2 / (s2.sum() + eps)
    p = p[p > eps]
    H = -(p * p.log()).sum().item()
    return math.exp(H)


def stable_rank(sigma: torch.Tensor, eps: float = 1e-12) -> float:
    """||A||_F² / ||A||_2². Always ≤ true rank."""
    return (sigma.pow(2).sum() / (sigma.max().pow(2) + eps)).item()


def spectral_decay_exponent(sigma: torch.Tensor) -> dict:
    """Power-law fit σ_i ~ i^(-α). Larger α = faster decay = lower-rank suffices."""
    s = sigma.cpu().numpy()
    s = s[s > 1e-12]
    if len(s) < 4:
        return {"alpha": float("nan"), "r_squared": float("nan")}
    i = np.arange(1, len(s) + 1)
    log_i = np.log(i)
    log_s = np.log(s)
    slope, intercept = np.polyfit(log_i, log_s, 1)
    predicted = slope * log_i + intercept
    ss_res = np.sum((log_s - predicted) ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"alpha": -float(slope), "r_squared": float(r2)}


@torch.no_grad()
def principal_angles(delta_w_1: torch.Tensor, delta_w_2: torch.Tensor, k: int) -> dict:
    """
    Principal angles between top-k left singular subspaces.
    Angles ≈ 0° → same subspace; ≈ 90° → orthogonal solutions.
    """
    U1, _, _ = torch.linalg.svd(delta_w_1, full_matrices=False)
    U2, _, _ = torch.linalg.svd(delta_w_2, full_matrices=False)
    U1k = U1[:, :k]
    U2k = U2[:, :k]
    M = U1k.T @ U2k
    s = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles_rad = torch.acos(s)
    angles_deg = (angles_rad * 180.0 / math.pi).cpu().numpy().tolist()
    grassmann = float(torch.sqrt(angles_rad.pow(2).sum()).item())
    return {
        "principal_angles_deg": [round(a, 2) for a in angles_deg],
        "mean_angle_deg": round(float(np.mean(angles_deg)), 2),
        "max_angle_deg":  round(float(np.max(angles_deg)), 2),
        "grassmann_distance": round(grassmann, 4),
    }


# ============================================================================
# Per-adapter profile
# ============================================================================

@torch.no_grad()
def random_adapter_like(
    reference: dict[str, dict[str, torch.Tensor]],
    device: str,
    seed: int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Generate a random LoRA adapter with the same module names and matrix
    shapes as `reference`, using PEFT's default LoRA init convention:
    A ~ Kaiming-uniform (matches nn.Linear default), B = zeros.

    Since B=0 gives ΔW=0 identically (useless for comparison), we instead
    draw BOTH A and B from the same Kaiming-uniform distribution used to
    init A — this represents "a randomly initialized but untrained adapter
    that has SOME update", giving a meaningful zero-alignment baseline for
    principal-angle comparisons. This is an artificial construction (real
    PEFT init has B=0) but answers the right question: "what do principal
    angles look like between two UNRELATED random subspaces of this shape?"
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    random_mats: dict[str, dict[str, torch.Tensor]] = {}
    for module_name, mats in reference.items():
        A_shape = mats["A"].shape
        B_shape = mats["B"].shape
        # Kaiming-uniform, same as nn.Linear default init
        bound_a = 1.0 / math.sqrt(A_shape[1])
        bound_b = 1.0 / math.sqrt(B_shape[1])
        A = (torch.rand(A_shape, generator=gen) * 2 - 1) * bound_a
        B = (torch.rand(B_shape, generator=gen) * 2 - 1) * bound_b
        random_mats[module_name] = {"A": A, "B": B}
    return random_mats


@torch.no_grad()
def compare_against_random_baseline(
    ckpt: Path,
    device: str,
    angle_k: int = 5,
    n_random_seeds: int = 3,
) -> dict:
    """
    Compare a trained adapter's top-k subspaces against freshly-initialized
    random adapters of the same shape. Gives the empirical "zero alignment"
    reference point for principal angles at this k, d (ambient dim).

    If trained-vs-trained angles (e.g. 50.9°) are well below
    random-vs-random angles (likely 75-88° per the asymptotic estimate
    cos²θ ≈ k/d), that confirms the trained subspaces share real structure
    despite not being identical.
    """
    print(f"\nRandom baseline for {ckpt} (k={angle_k}, {n_random_seeds} random seeds)")
    mats_trained = load_adapter_weights(ckpt)
    module_names = sorted(mats_trained.keys())

    all_seed_results = []
    for rseed in range(n_random_seeds):
        random_mats = random_adapter_like(mats_trained, device, seed=rseed)
        angles_this_seed = []
        for module_name in module_names:
            dw_trained = compute_delta_w(
                mats_trained[module_name]["A"], mats_trained[module_name]["B"], device
            )
            dw_random = compute_delta_w(
                random_mats[module_name]["A"], random_mats[module_name]["B"], device
            )
            k = min(
                mats_trained[module_name]["A"].shape[0],
                random_mats[module_name]["A"].shape[0],
                angle_k,
            )
            result = principal_angles(dw_trained, dw_random, k=k)
            angles_this_seed.append(result["mean_angle_deg"])
        all_seed_results.append(float(np.mean(angles_this_seed)))

    # Also compute random-vs-random (two independent random adapters)
    random_vs_random = []
    for rseed in range(n_random_seeds):
        mats_a = random_adapter_like(mats_trained, device, seed=100 + rseed)
        mats_b = random_adapter_like(mats_trained, device, seed=200 + rseed)
        angles = []
        for module_name in module_names:
            dw_a = compute_delta_w(mats_a[module_name]["A"], mats_a[module_name]["B"], device)
            dw_b = compute_delta_w(mats_b[module_name]["A"], mats_b[module_name]["B"], device)
            k = min(mats_a[module_name]["A"].shape[0], mats_b[module_name]["A"].shape[0], angle_k)
            result = principal_angles(dw_a, dw_b, k=k)
            angles.append(result["mean_angle_deg"])
        random_vs_random.append(float(np.mean(angles)))

    return {
        "checkpoint": str(ckpt),
        "angle_k": angle_k,
        "n_random_seeds": n_random_seeds,
        "trained_vs_random_mean_deg": round(float(np.mean(all_seed_results)), 2),
        "trained_vs_random_std_deg":  round(float(np.std(all_seed_results)), 2),
        "random_vs_random_mean_deg":  round(float(np.mean(random_vs_random)), 2),
        "random_vs_random_std_deg":   round(float(np.std(random_vs_random)), 2),
        "per_seed_trained_vs_random": [round(a, 2) for a in all_seed_results],
        "per_seed_random_vs_random":  [round(a, 2) for a in random_vs_random],
    }


def print_random_baseline_summary(rb: dict, trained_vs_trained_deg: float | None = None) -> None:
    print("\n" + "=" * 70)
    print(f"RANDOM BASELINE  (k={rb['angle_k']}, {rb['n_random_seeds']} seeds)")
    print("=" * 70)
    print(f"  trained vs random:  {rb['trained_vs_random_mean_deg']:.1f}° "
          f"± {rb['trained_vs_random_std_deg']:.1f}°")
    print(f"  random vs random:   {rb['random_vs_random_mean_deg']:.1f}° "
          f"± {rb['random_vs_random_std_deg']:.1f}°")
    if trained_vs_trained_deg is not None:
        print(f"\n  trained vs trained (r16 vs r32): {trained_vs_trained_deg:.1f}°")
        gap_to_random = rb["random_vs_random_mean_deg"] - trained_vs_trained_deg
        full_range = rb["random_vs_random_mean_deg"]
        pct = 100 * gap_to_random / full_range if full_range > 0 else 0
        print(f"\n  → trained subspaces are {pct:.0f}% of the way from "
              f"'fully random' (0%) to 'identical' (100%, i.e. 0°),")
        print(f"    relative to the {rb['random_vs_random_mean_deg']:.1f}° random-vs-random ceiling.")


# ============================================================================
# Per-adapter profile
# ============================================================================

@torch.no_grad()
def profile_adapter(checkpoint_path: Path, device: str) -> dict:
    print(f"\nProfiling {checkpoint_path}")
    matrices = load_adapter_weights(checkpoint_path)
    print(f"  Found {len(matrices)} LoRA modules")

    profile = {
        "checkpoint": str(checkpoint_path),
        "n_modules": len(matrices),
        "modules": {},
    }

    by_layer_type: dict[str, list[float]] = defaultdict(list)
    by_layer_type_alpha: dict[str, list[float]] = defaultdict(list)

    for module_name, mats in sorted(matrices.items()):
        delta_w = compute_delta_w(mats["A"], mats["B"], device)
        sigma = singular_values(delta_w)
        r_config = mats["A"].shape[0]

        eff_r  = effective_rank(sigma)
        stab_r = stable_rank(sigma)
        decay  = spectral_decay_exponent(sigma)

        profile["modules"][module_name] = {
            "configured_rank":     int(r_config),
            "effective_rank":      round(eff_r, 3),
            "stable_rank":         round(stab_r, 3),
            "utilization":         round(eff_r / r_config, 3),
            "spectral_alpha":      round(decay["alpha"], 3),
            "spectral_fit_r2":     round(decay["r_squared"], 3),
            "top_5_sigma":         [round(float(s), 5) for s in sigma[:5].cpu().tolist()],
            "frobenius_norm":      round(float(delta_w.norm().item()), 4),
        }

        layer_type = module_name.split(".")[-1]
        by_layer_type[layer_type].append(eff_r)
        by_layer_type_alpha[layer_type].append(decay["alpha"])

    profile["per_layer_type"] = {
        lt: {
            "n_modules":           len(ranks),
            "mean_effective_rank": round(float(np.mean(ranks)), 3),
            "std_effective_rank":  round(float(np.std(ranks)), 3),
            "min_effective_rank":  round(float(np.min(ranks)), 3),
            "max_effective_rank":  round(float(np.max(ranks)), 3),
            "mean_spectral_alpha": round(float(np.nanmean(by_layer_type_alpha[lt])), 3),
        }
        for lt, ranks in sorted(by_layer_type.items())
    }

    all_eff  = [m["effective_rank"] for m in profile["modules"].values()]
    all_util = [m["utilization"]     for m in profile["modules"].values()]
    profile["global"] = {
        "mean_effective_rank":   round(float(np.mean(all_eff)), 3),
        "median_effective_rank": round(float(np.median(all_eff)), 3),
        "mean_utilization":      round(float(np.mean(all_util)), 3),
        "n_modules":             len(all_eff),
    }

    return profile


# ============================================================================
# Comparison (r=16 vs r=32)
# ============================================================================

@torch.no_grad()
def compare_adapters(ckpt1: Path, ckpt2: Path, device: str, angle_k: int = 16) -> dict:
    print(f"\nComparing adapters:")
    print(f"  A: {ckpt1}")
    print(f"  B: {ckpt2}")

    mats1 = load_adapter_weights(ckpt1)
    mats2 = load_adapter_weights(ckpt2)
    shared = sorted(set(mats1) & set(mats2))
    print(f"  {len(shared)} shared modules")

    comparison = {
        "checkpoint_a": str(ckpt1),
        "checkpoint_b": str(ckpt2),
        "rank_a": int(mats1[shared[0]]["A"].shape[0]) if shared else None,
        "rank_b": int(mats2[shared[0]]["A"].shape[0]) if shared else None,
        "principal_angle_k": angle_k,
        "per_module": {},
    }

    all_mean_angles = []
    all_grassmann = []

    for module_name in shared:
        dw1 = compute_delta_w(mats1[module_name]["A"], mats1[module_name]["B"], device)
        dw2 = compute_delta_w(mats2[module_name]["A"], mats2[module_name]["B"], device)
        k = min(
            mats1[module_name]["A"].shape[0],
            mats2[module_name]["A"].shape[0],
            angle_k,
        )
        result = principal_angles(dw1, dw2, k=k)
        comparison["per_module"][module_name] = result
        all_mean_angles.append(result["mean_angle_deg"])
        all_grassmann.append(result["grassmann_distance"])

    comparison["global"] = {
        "mean_of_mean_angles_deg": round(float(np.mean(all_mean_angles)), 2),
        "mean_grassmann_distance": round(float(np.mean(all_grassmann)), 4),
        "interpretation": _interpret_subspace_overlap(float(np.mean(all_mean_angles))),
    }

    return comparison


def _interpret_subspace_overlap(mean_angle_deg: float) -> str:
    if mean_angle_deg < 15:
        return ("Subspaces highly aligned — larger rank mostly adds capacity in the "
                "SAME directions the smaller rank already learned. Extra parameters "
                "likely redundant.")
    elif mean_angle_deg < 45:
        return ("Subspaces partially aligned — larger rank shares core directions with "
                "smaller rank but also explores new ones. Some genuine capacity gain.")
    elif mean_angle_deg < 75:
        return ("Subspaces substantially different — larger rank converges to a "
                "different solution. Rank changes the optimization geometry, not just "
                "the capacity ceiling.")
    else:
        return ("Subspaces nearly orthogonal — the two ranks learn essentially "
                "unrelated solutions. Suggests a rugged loss landscape.")


# ============================================================================
# Training dynamics
# ============================================================================

@torch.no_grad()
def scan_training_dynamics(checkpoint_dir: Path, device: str) -> dict:
    print(f"\nScanning training dynamics in {checkpoint_dir}")

    checkpoint_paths = sorted(
        [p for p in checkpoint_dir.iterdir()
         if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if (checkpoint_dir / "final").exists():
        checkpoint_paths.append(checkpoint_dir / "final")

    if not checkpoint_paths:
        print("  No checkpoints found")
        return {"error": "no checkpoints"}

    print(f"  Found {len(checkpoint_paths)} checkpoints")

    dynamics = {
        "checkpoint_dir": str(checkpoint_dir),
        "n_checkpoints": len(checkpoint_paths),
        "checkpoints": [],
    }

    for ckpt in checkpoint_paths:
        step = int(ckpt.name.split("-")[-1]) if ckpt.name != "final" else None
        profile = profile_adapter(ckpt, device)
        dynamics["checkpoints"].append({
            "step":                       step,
            "checkpoint":                 ckpt.name,
            "global_mean_effective_rank": profile["global"]["mean_effective_rank"],
            "global_mean_utilization":    profile["global"]["mean_utilization"],
            "per_layer_type_mean_rank": {
                lt: stats["mean_effective_rank"]
                for lt, stats in profile["per_layer_type"].items()
            },
        })

    return dynamics


# ============================================================================
# Pretty-print
# ============================================================================

def print_profile_summary(profile: dict) -> None:
    print("\n" + "=" * 70)
    print(f"SPECTRAL PROFILE: {profile['checkpoint']}")
    print("=" * 70)
    g = profile["global"]
    print(f"\nGlobal effective rank:")
    print(f"  Mean:        {g['mean_effective_rank']:.2f}")
    print(f"  Median:      {g['median_effective_rank']:.2f}")
    print(f"  Utilization: {g['mean_utilization']:.1%}  (effective / configured)")
    print(f"\nPer layer type:")
    for lt, stats in profile["per_layer_type"].items():
        print(f"  {lt:15s}  eff_rank: {stats['mean_effective_rank']:5.2f} "
              f"± {stats['std_effective_rank']:4.2f}  "
              f"(range {stats['min_effective_rank']:.1f}–{stats['max_effective_rank']:.1f})  "
              f"α={stats['mean_spectral_alpha']:.2f}")

    util = g["mean_utilization"]
    if util < 0.7:
        print(f"\n  → Effective rank well below configured rank ({util:.0%}).")
        print(f"    Smaller LoRA rank would likely give equivalent performance.")
    elif util < 0.9:
        print(f"\n  → Effective rank slightly below configured rank ({util:.0%}).")
        print(f"    Modest headroom in current rank choice.")
    else:
        print(f"\n  → Effective rank near configured rank ({util:.0%}).")
        print(f"    Larger rank might provide additional useful capacity.")


def print_comparison_summary(comp: dict) -> None:
    print("\n" + "=" * 70)
    print(f"SUBSPACE COMPARISON: r={comp['rank_a']} vs r={comp['rank_b']}")
    print("=" * 70)
    g = comp["global"]
    print(f"\nTop-{comp['principal_angle_k']} subspace alignment:")
    print(f"  Mean principal angle:     {g['mean_of_mean_angles_deg']:.1f}°")
    print(f"  Mean Grassmann distance:  {g['mean_grassmann_distance']:.3f}")
    print(f"\n  Interpretation:")
    print(f"  {g['interpretation']}")

    angles_by_module = [
        (name, m["mean_angle_deg"])
        for name, m in comp["per_module"].items()
    ]
    angles_by_module.sort(key=lambda x: -x[1])
    print(f"\n  Most divergent modules (top 5):")
    for name, ang in angles_by_module[:5]:
        print(f"    {name:60s}  {ang:5.1f}°")
    print(f"\n  Most aligned modules (top 5):")
    for name, ang in reversed(angles_by_module[-5:]):
        print(f"    {name:60s}  {ang:5.1f}°")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     type=Path, default=None)
    parser.add_argument("--compare",        type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir",     type=Path, default=Path("analysis"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--angle-k", type=int, default=16)
    parser.add_argument("--random-baseline", action="store_true",
                        help="Compare --checkpoint against random adapters of the same "
                             "shape, and random-vs-random, to establish a zero-alignment "
                             "reference for principal angles at --angle-k.")
    parser.add_argument("--random-seeds", type=int, default=3)
    parser.add_argument("--trained-vs-trained-deg", type=float, default=None,
                        help="If you already have a trained-vs-trained mean angle "
                             "(e.g. from --compare), pass it here to show where it "
                             "falls relative to the random baseline.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available — running on CPU")

    if args.checkpoint_dir:
        dyn = scan_training_dynamics(args.checkpoint_dir, args.device)
        out = args.output_dir / f"dynamics_{args.checkpoint_dir.name}.json"
        with out.open("w") as f:
            json.dump(dyn, f, indent=2)
        print(f"\nWrote {out}")
        print(f"\n{'step':>6}  {'eff_rank':>9}  {'util':>6}")
        for cp in dyn["checkpoints"]:
            step_str = str(cp["step"]) if cp["step"] is not None else "final"
            print(f"{step_str:>6}  {cp['global_mean_effective_rank']:>9.3f}  "
                  f"{cp['global_mean_utilization']:>6.3f}")
        return

    if not args.checkpoint:
        parser.error("Must specify --checkpoint or --checkpoint-dir")

    if args.random_baseline:
        rb = compare_against_random_baseline(
            args.checkpoint, args.device,
            angle_k=args.angle_k, n_random_seeds=args.random_seeds,
        )
        out = args.output_dir / f"random_baseline_{args.checkpoint.parent.name}_k{args.angle_k}.json"
        with out.open("w") as f:
            json.dump(rb, f, indent=2)
        print_random_baseline_summary(rb, args.trained_vs_trained_deg)
        print(f"\nWrote {out}")
        return

    profile_a = profile_adapter(args.checkpoint, args.device)
    out_a = args.output_dir / f"profile_{args.checkpoint.parent.name}.json"
    with out_a.open("w") as f:
        json.dump(profile_a, f, indent=2)
    print_profile_summary(profile_a)
    print(f"\nWrote {out_a}")

    if args.compare:
        profile_b = profile_adapter(args.compare, args.device)
        out_b = args.output_dir / f"profile_{args.compare.parent.name}.json"
        with out_b.open("w") as f:
            json.dump(profile_b, f, indent=2)
        print_profile_summary(profile_b)
        print(f"\nWrote {out_b}")

        comparison = compare_adapters(args.checkpoint, args.compare, args.device, args.angle_k)
        out_c = args.output_dir / f"compare_{args.checkpoint.parent.name}_vs_{args.compare.parent.name}.json"
        with out_c.open("w") as f:
            json.dump(comparison, f, indent=2)
        print_comparison_summary(comparison)
        print(f"\nWrote {out_c}")


if __name__ == "__main__":
    main()