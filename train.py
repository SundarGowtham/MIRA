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
import sys
from pathlib import Path

from experiments import EXPERIMENTS


def _assert_expected_environment(use_vllm: str) -> None:
    """Environment guard (Phase 16, "Decisions after pass 2", the
    mis-install correction: `uv pip install` honored this machine's
    auto-activated VIRTUAL_ENV over UV_PROJECT_ENVIRONMENT and briefly
    installed vllm/torch into the frozen main environment -- see
    docs/phases/PHASE16_RESULTS.md). --use-vllm off (default, Phase 12
    behaviour) requires the main project .venv -- vLLM must never be
    installed there (rule 7). --use-vllm auto/on requires mira-vllm,
    the only environment vLLM is installed in. Either way, refuse to
    run in the wrong one rather than silently drift."""
    main_env = (Path(__file__).resolve().parent / ".venv").resolve()
    vllm_env = (Path.home() / "envs" / "mira-vllm").resolve()
    actual = Path(sys.prefix).resolve()
    if use_vllm == "off":
        if actual != main_env:
            raise RuntimeError(
                f"train.py --use-vllm off refuses to run outside the main "
                f"project environment. Expected sys.prefix={main_env}, got "
                f"{actual}.")
    else:
        if actual != vllm_env:
            raise RuntimeError(
                f"train.py --use-vllm {use_vllm} refuses to run outside "
                f"mira-vllm (the only environment vLLM is installed in, "
                f"per rule 7 -- it must never be installed in the main "
                f"project env). Expected sys.prefix={vllm_env}, got {actual}. "
                f"Invoke with `source ~/envs/mira-vllm/bin/activate && "
                f"python train.py ...`.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=sorted(EXPERIMENTS.keys()))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--adapter", choices=["full", "lora", "qlora"], default="qlora")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    p.add_argument("--init-from", type=str, default=None, help="Checkpoint to resume from (path or 'base' for pretrained)")
    p.add_argument("--sft-checkpoint", type=str, default=None, help="For sft-grpo: path to SFT checkpoint to start GRPO from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None, help="Optional run name suffix for W&B")
    p.add_argument("--data-prefix", default=None,
                   help="Filename prefix for data files (default: 'sft'). E.g. --data-prefix sft_v2 uses sft_v2_train.jsonl etc.")
    p.add_argument("--lora-r", type=int, default=None,
                   help="LoRA rank. Overrides the experiment's default. Used by rank_ablation.py.")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha. Overrides the experiment's default. Conventional: 2 * lora_r.")
    p.add_argument("--lora-dropout", type=float, default=None,
                   help="LoRA dropout. Overrides the experiment's default.")
    p.add_argument("--reward-aggregation",
                   choices=["normalize_then_sum", "sum_then_normalize"],
                   default="normalize_then_sum",
                   help="GRPO only: multi-reward aggregation. normalize_then_sum = "
                        "GDPO (per-check z-normalize, then combine); sum_then_normalize "
                        "= classic GRPO (combine, then normalize). Grid axis A.")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate (e.g. ablation probes).")
    p.add_argument("--kl-beta", type=float, default=None,
                   help="Override GRPO KL coefficient beta (default 0.04).")
    p.add_argument("--fresh-restart", action="store_true",
                   help="Load adapter weights from --init-from but start a FRESH "
                        "trainer (new optimizer/scheduler, step 0). Use when changing "
                        "lr/beta: a normal resume restores the old scheduler state "
                        "and silently re-imposes the old LR schedule.")
    p.add_argument("--probe-eval-steps", type=int, default=None,
                   help="GRPO only: evaluate the fixed probe set "
                        "(<data-dir>/<prefix>_probe.jsonl) every N steps "
                        "(default 50; 1 in --smoke). Probe generations are "
                        "archived in generations.jsonl like training ones.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="SFT only: hard step cap, overrides epochs (e.g. the "
                        "equal-compute continuation arm matched to GDPO-300's "
                        "~75 GPU-hours: ~24000 steps at ~11 s/step).")
    p.add_argument("--scorer", choices=["validator", "ranker"], default="validator",
                   help="GRPO/GDPO only: validator = Arm A (core/reward.py's "
                        "SynthesisValidator, validity checks). ranker = Arm B "
                        "(core/ranker.py's gates x objectives quality scorer, "
                        "RANKER_SPEC.md). The verifier is the only thing this "
                        "flag should change -- everything else (beta, lr, G, "
                        "data pipeline) stays identical between arms.")
    p.add_argument("--save-steps", type=int, default=None,
                   help="GRPO only: checkpoint save interval. Default 100 "
                        "(Phase 12 behaviour, experiments/grpo.py). Cloud runs "
                        "use 10 (Phase 16, more frequent B2 sync points).")
    p.add_argument("--validator-version", type=int, choices=[1, 2], default=1,
                   help="GRPO/GDPO only (validator scorer): 1 (default) "
                        "reproduces every prior run's scoring exactly. 2 "
                        "enables the ammonium-balance fix in "
                        "validator.py::_find_balanced_reaction (Phase 16 "
                        "§2.4.3). All Phase 16 arms use 2.")
    p.add_argument("--use-vllm", choices=["off", "auto", "on"], default="off",
                   help="GRPO/GDPO only (Phase 16 §2.2): off (default) is "
                        "Phase 12 behaviour exactly (HF generate). auto "
                        "tries vLLM (requires `import vllm` to succeed and "
                        "an Ampere-or-newer GPU) and falls back to HF "
                        "generate with one loud log line if not. on fails "
                        "loudly instead of falling back. Record the backend "
                        "actually used as generation_backend in W&B.")
    p.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate",
                   help="GRPO/GDPO + --use-vllm only: colocate (default) "
                        "shares the training GPU with sleep mode enabled. "
                        "server expects --vllm-server-host/--vllm-server-port "
                        "to point at a separately-launched vLLM server "
                        "(two-GPU setup).")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                   help="GRPO/GDPO + --use-vllm colocate only. Default "
                        "(None) lets experiments/grpo.py pick a config-"
                        "appropriate value -- measure peak memory in the "
                        "§5 benchmark before overriding, do not guess.")
    p.add_argument("--vllm-server-host", default="0.0.0.0")
    p.add_argument("--vllm-server-port", type=int, default=8000)


    return p.parse_args()


def main():
    args = parse_args()
    _assert_expected_environment(args.use_vllm)
    cls = EXPERIMENTS[args.experiment]
    experiment = cls(args)
    experiment.run()


if __name__ == "__main__":
    main()