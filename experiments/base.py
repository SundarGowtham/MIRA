"""
Experiment base class.

An Experiment owns:
  - its name and run_name (for W&B and output dir)
  - its hyperparameter defaults (overridable via CLI in subclass.parse_extra)
  - its training method (run())
  - its checkpoint conventions (final_dir property)

Shared logic lives here: output dir resolution, W&B init, seeding.
Subclasses implement only what's actually different.
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    name: str
    smoke: bool
    adapter: str
    model: str | None
    data_dir: Path
    output_root: Path
    seed: int
    tag: str | None
    extras: dict[str, Any] = field(default_factory=dict)


class Experiment(ABC):
    name: str = "experiment"

    SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
    FULL_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"

    def __init__(self, args):
        self.args = args
        self.cfg = self._build_config(args)
        self._set_seed(self.cfg.seed)

    def _build_config(self, args) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.name,
            smoke=args.smoke,
            adapter=args.adapter,
            model=args.model or (self.SMOKE_MODEL if args.smoke else self.FULL_MODEL),
            data_dir=args.data_dir,
            output_root=args.output_root,
            seed=args.seed,
            tag=args.tag,
        )

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def run_name(self) -> str:
        parts = [self.name, self.cfg.adapter]
        if self.cfg.smoke:
            parts.append("smoke")
        if self.cfg.tag:
            parts.append(self.cfg.tag)
        return "-".join(parts)

    @property
    def output_dir(self) -> Path:
        d = self.cfg.output_root / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def final_dir(self) -> Path:
        return self.output_dir / "final"

    def init_wandb(self, extra_config: dict | None = None):
        if os.environ.get("WANDB_API_KEY") is None:
            return None
        try:
            import wandb
        except ImportError:
            return None
        config = {
            "experiment": self.name,
            "adapter": self.cfg.adapter,
            "model": self.cfg.model,
            "smoke": self.cfg.smoke,
            "seed": self.cfg.seed,
        }
        if extra_config:
            config.update(extra_config)
        return wandb.init(
            project="mira",
            name=self.run_name,
            config=config,
            reinit=True,
        )

    @abstractmethod
    def run(self) -> Path:
        """Train and return the final checkpoint path."""
        ...