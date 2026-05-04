"""
train.py — Orchestrator for MIRA training experiments.

Experiments live in experiments/ and register themselves in EXPERIMENTS.
This file contains zero training logic; it only resolves config and
dispatches to the chosen experiment.

Usage:
    python train.py sft --adapter qlora
    python train.py grpo --adapter qlora --init-from base
    python train.py sft-grpo --adapter qlora --sft-checkpoint runs/sft-qlora/final
    python train.py sft --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import EXPERIMENTS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=sorted(EXPERIMENTS.keys()))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--adapter", choices=["full", "lora", "qlora"], default="qlora")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    p.add_argument("--init-from", type=str, default=None,
                   help="Checkpoint to resume from (path or 'base' for pretrained)")
    p.add_argument("--sft-checkpoint", type=str, default=None,
                   help="For sft-grpo: path to SFT checkpoint to start GRPO from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None,
                   help="Optional run name suffix for W&B")
    return p.parse_args()


def main():
    args = parse_args()
    cls = EXPERIMENTS[args.experiment]
    experiment = cls(args)
    experiment.run()


if __name__ == "__main__":
    main()