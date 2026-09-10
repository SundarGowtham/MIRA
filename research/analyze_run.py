"""
analyze_run.py
--------------
Retrospective analysis of a W&B training run.

Pulls all logged history for a run, generates diagnostic plots,
and identifies causally interesting moments (entropy crash, reward
variance collapse, gradient death by layer group).

Usage:
    python analyze_run.py <wandb_run_path>
    python analyze_run.py mira/grpo-qlora-stage2-stage2-grpo
    python analyze_run.py "user/project/run_id" --output-dir analysis/grpo_v1

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dotenv

dotenv.load_dotenv()



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_path",
                   help="W&B run path: 'entity/project/run_id' or 'project/run_name'")
    p.add_argument("--output-dir", type=Path, default=Path("analysis"))
    p.add_argument("--samples", type=int, default=20000,
                   help="Max history rows to pull (default 20000)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_history(run_path: str, samples: int) -> tuple[pd.DataFrame, dict]:
    """Pulls full step-by-step history + run config."""
    import wandb
    api = wandb.Api()
    run = api.run(run_path)
    df = run.history(samples=samples, pandas=True)
    config = {k: v for k, v in run.config.items()
              if not k.startswith("_") and isinstance(v, (str, int, float, bool))}
    print(f"Loaded {len(df)} steps, {len(df.columns)} metrics")
    print(f"Run state: {run.state}, total runtime: {run.summary.get('_runtime', 'unknown')}s")
    return df, config


def find_step_column(df: pd.DataFrame) -> str:
    """Find the step counter column (W&B uses different names)."""
    for candidate in ["train/global_step", "global_step", "_step", "step"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No step column found. Available: {df.columns.tolist()[:20]}")


# ---------------------------------------------------------------------------
# Causal moment detection
# ---------------------------------------------------------------------------

def find_change_point(series: pd.Series, window: int = 50) -> int | None:
    """
    Find the step where a series shows the largest change in mean.

    Uses a sliding-window mean-shift detector. Returns the step index
    where the difference between trailing window and leading window is
    maximized.
    """
    if series is None:
        return None
        
    s = series.dropna().values
    if len(s) < window * 2:
        return None
    diffs = np.zeros(len(s) - window * 2)
    for i in range(window, len(s) - window):
        before = s[i - window:i].mean()
        after = s[i:i + window].mean()
        diffs[i - window] = abs(after - before)
    if len(diffs) == 0:
        return None
    return int(np.argmax(diffs)) + window


def find_collapse_step(df: pd.DataFrame, step_col: str, metric: str, threshold: float, direction: str = "above") -> int | None:
    """
    Find first step where metric crosses threshold and stays there.

    direction='above' → first step >= threshold
    direction='below' → first step <= threshold
    """
    if metric not in df.columns:
        return None
    sub = df[[step_col, metric]].dropna()
    if direction == "above":
        crossed = sub[sub[metric] >= threshold]
    else:
        crossed = sub[sub[metric] <= threshold]
    if len(crossed) == 0:
        return None
    return int(crossed.iloc[0][step_col])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(df: pd.DataFrame, step_col: str, output_dir: Path):
    """Top-level training health: reward, KL, entropy, loss."""
    metrics = [
        ("train/rewards/reward_fn/mean", "Mean Reward", "tab:blue"),
        ("train/rewards/reward_fn/std", "Reward Std (within-group)", "tab:orange"),
        ("train/kl", "KL Divergence", "tab:green"),
        ("train/entropy", "Entropy", "tab:red"),
        ("train/loss", "Loss", "tab:purple"),
        ("train/frac_reward_zero_std", "Frac Groups with Zero Std", "tab:brown"),
    ]
    available = [(k, label, color) for k, label, color in metrics if k in df.columns]
    n = len(available)
    if n == 0:
        print("No reward metrics found, skipping overview plot")
        return

    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 3 * ((n + 1) // 2)))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (key, label, color) in zip(axes, available):
        sub = df[[step_col, key]].dropna()
        ax.plot(sub[step_col], sub[key], color=color, alpha=0.6, linewidth=0.8)
        # Smoothed line
        if len(sub) > 50:
            window = max(20, len(sub) // 50)
            smoothed = sub[key].rolling(window, min_periods=1, center=True).mean()
            ax.plot(sub[step_col], smoothed, color=color, linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(available):]:
        ax.set_visible(False)

    fig.tight_layout()
    out = output_dir / "01_overview.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_completion_dynamics(df: pd.DataFrame, step_col: str, output_dir: Path):
    """Completion length, clipping, termination — generation health."""
    metrics_groups = [
        ("Completion Length",
         ["train/completions/min_length",
          "train/completions/mean_length",
          "train/completions/max_length"]),
        ("Terminated Length (excludes clipped)",
         ["train/completions/min_terminated_length",
          "train/completions/mean_terminated_length",
          "train/completions/max_terminated_length"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, metrics) in zip(axes, metrics_groups):
        for metric in metrics:
            if metric not in df.columns:
                continue
            sub = df[[step_col, metric]].dropna()
            label = metric.split("/")[-1].replace("_", " ").title()
            ax.plot(sub[step_col], sub[metric], alpha=0.7, label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "02_completion_dynamics.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_per_layer_gradients(df: pd.DataFrame, step_col: str, output_dir: Path):
    """Per-layer-group gradient norms (from your GradientStatsCallback)."""
    grad_cols = [c for c in df.columns if c.startswith("grad_norm/")]
    weight_cols = [c for c in df.columns if c.startswith("weight_norm/")]
    update_cols = [c for c in df.columns if c.startswith("update_ratio/")]

    if not grad_cols and not update_cols:
        print("No per-layer-group gradient stats found (callback not active?)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for col in grad_cols:
        sub = df[[step_col, col]].dropna()
        if len(sub) > 0:
            axes[0].plot(sub[step_col], sub[col], label=col.split("/")[-1], alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Gradient Norm by Layer Group (log scale)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("||∇L||")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for col in weight_cols:
        sub = df[[step_col, col]].dropna()
        if len(sub) > 0:
            axes[1].plot(sub[step_col], sub[col], label=col.split("/")[-1], alpha=0.8)
    axes[1].set_title("Weight Norm by Layer Group")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("||W||")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for col in update_cols:
        sub = df[[step_col, col]].dropna()
        if len(sub) > 0:
            axes[2].plot(sub[step_col], sub[col], label=col.split("/")[-1], alpha=0.8)
    axes[2].set_yscale("log")
    axes[2].set_title("Update-to-Weight Ratio by Layer Group (log scale)")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("||∇L|| / ||W||")
    axes[2].axhline(1e-3, color="green", linestyle="--", alpha=0.5, label="1e-3 (healthy upper)")
    axes[2].axhline(1e-4, color="orange", linestyle="--", alpha=0.5, label="1e-4 (healthy lower)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "03_per_layer_gradients.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_causal_alignment(df: pd.DataFrame, step_col: str, output_dir: Path, events: dict):
    """
    Overlay multiple metrics on shared x-axis with vertical lines marking
    detected causal moments. Most important plot for the writeup.
    """
    metrics_to_overlay = [
        ("train/rewards/reward_fn/mean", "Mean Reward"),
        ("train/rewards/reward_fn/std", "Reward Std"),
        ("train/entropy", "Entropy"),
        ("train/frac_reward_zero_std", "Frac Zero-Std Groups"),
    ]
    available = [(k, l) for k, l in metrics_to_overlay if k in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(14, 2.5 * len(available)),
                             sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, available):
        sub = df[[step_col, key]].dropna()
        ax.plot(sub[step_col], sub[key], linewidth=1, alpha=0.6)
        if len(sub) > 50:
            window = max(20, len(sub) // 50)
            smoothed = sub[key].rolling(window, min_periods=1, center=True).mean()
            ax.plot(sub[step_col], smoothed, linewidth=2)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)

        for event_name, event_step in events.items():
            if event_step is not None:
                ax.axvline(event_step, color="red", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Step")
    axes[0].set_title("Causal Alignment: Vertical Lines Mark Detected Events")

    # Annotate events on top axis
    if events:
        ymin, ymax = axes[0].get_ylim()
        for i, (name, step) in enumerate(events.items()):
            if step is not None:
                axes[0].annotate(
                    f"{name}\nstep {step}",
                    xy=(step, ymax),
                    xytext=(step, ymax + (ymax - ymin) * 0.1 * (i + 1)),
                    fontsize=8,
                    color="red",
                    ha="center",
                )

    fig.tight_layout()
    out = output_dir / "04_causal_alignment.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Causal events
# ---------------------------------------------------------------------------

def detect_events(df: pd.DataFrame, step_col: str) -> dict:
    """Identify diagnostic moments in the training trajectory."""
    events = {}

    events["entropy_crash"] = find_change_point(df.get("train/entropy"), window=30)
    events["mode_collapse_onset"] = find_collapse_step(
        df, step_col, "train/frac_reward_zero_std",
        threshold=0.5, direction="above",
    )
    events["full_mode_collapse"] = find_collapse_step(
        df, step_col, "train/frac_reward_zero_std",
        threshold=0.85, direction="above",
    )
    events["reward_std_collapse"] = find_change_point(
        df.get("train/rewards/reward_fn/std"), window=30,
    )

    # Try to identify gradient death by layer group
    for col in df.columns:
        if col.startswith("grad_norm/"):
            group = col.split("/")[-1]
            cp = find_change_point(df[col], window=30)
            if cp is not None:
                events[f"grad_change_{group}"] = cp

    # Convert window-relative indices to real steps
    real_events = {}
    for name, idx in events.items():
        if idx is None:
            real_events[name] = None
            continue
        if name in ("mode_collapse_onset", "full_mode_collapse"):
            real_events[name] = idx
            continue
        # Map sliding-window index back to actual step
        sub = df[step_col].dropna().values
        if 0 <= idx < len(sub):
            real_events[name] = int(sub[idx])
        else:
            real_events[name] = None

    return real_events


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(df: pd.DataFrame, step_col: str, config: dict, events: dict, output_dir: Path):
    """Plain-text summary of the run for the writeup."""
    lines = []
    lines.append("=" * 60)
    lines.append("RUN ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Configuration:")
    for k, v in sorted(config.items()):
        lines.append(f"  {k:30s}: {v}")
    lines.append("")
    lines.append(f"Total steps logged: {len(df)}")
    lines.append(f"Step range: {df[step_col].min()} → {df[step_col].max()}")
    lines.append("")

    lines.append("Final values (last 50 steps mean):")
    final = df.tail(50)
    for metric in ["train/rewards/reward_fn/mean",
                   "train/rewards/reward_fn/std",
                   "train/entropy",
                   "train/kl",
                   "train/frac_reward_zero_std"]:
        if metric in df.columns:
            v = final[metric].dropna().mean()
            label = metric.split("/")[-1]
            lines.append(f"  {label:30s}: {v:.4f}")
    lines.append("")

    lines.append("Detected causal events:")
    for name, step in events.items():
        if step is not None:
            lines.append(f"  {name:30s}: step {step}")
        else:
            lines.append(f"  {name:30s}: not detected")
    lines.append("")

    # Causal interpretation
    lines.append("Interpretation:")
    entropy_crash = events.get("entropy_crash")
    mc_onset = events.get("mode_collapse_onset")
    full_mc = events.get("full_mode_collapse")
    if entropy_crash and mc_onset:
        if entropy_crash < mc_onset:
            lines.append(f"  Entropy collapse ({entropy_crash}) preceded mode collapse")
            lines.append(f"  onset ({mc_onset}) by {mc_onset - entropy_crash} steps. This")
            lines.append(f"  is consistent with policy convergence preceding reward-")
            lines.append(f"  variance death — the model became deterministic, then groups")
            lines.append(f"  of generations stopped exhibiting reward variance.")
        else:
            lines.append(f"  Mode collapse onset ({mc_onset}) preceded entropy crash")
            lines.append(f"  ({entropy_crash}). This is unusual — investigate reward")
            lines.append(f"  function for plateau structure causing premature std collapse.")
    lines.append("")

    text = "\n".join(lines)
    out = output_dir / "summary.txt"
    out.write_text(text)
    print(f"  saved {out}")
    print()
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading run: {args.run_path}")
    df, config = load_run_history(args.run_path, args.samples)

    step_col = find_step_column(df)
    print(f"Using step column: {step_col}")

    # Save raw history for further offline analysis
    csv_path = args.output_dir / "history.csv"
    df.to_csv(csv_path, index=False)
    print(f"  saved raw history → {csv_path}")

    print("\nDetecting causal events...")
    events = detect_events(df, step_col)
    for name, step in events.items():
        print(f"  {name}: {step}")

    print("\nGenerating plots...")
    plot_overview(df, step_col, args.output_dir)
    plot_completion_dynamics(df, step_col, args.output_dir)
    plot_per_layer_gradients(df, step_col, args.output_dir)
    plot_causal_alignment(df, step_col, args.output_dir, events)

    print("\nWriting summary...")
    write_summary(df, step_col, config, events, args.output_dir)

    # Save events as JSON for downstream analysis
    events_json = args.output_dir / "events.json"
    with events_json.open("w") as f:
        json.dump(events, f, indent=2)


if __name__ == "__main__":
    main()