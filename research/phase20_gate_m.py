#!/usr/bin/env python
"""
research/phase20_gate_m.py -- Phase 20 Gate M (memory).

For one dataset: run ONE optimizer step of the real --sft-full training path
(experiments/sft.py via train.py's own argument parser, QLoRA NF4, r16/a32,
gradient checkpointing, bf16, TRL's default chunked_nll loss) on that
dataset's single longest training example (by full tokenized length). The
only overrides are the dataset (that one example), accumulation 1 and
--max-steps 1. Reports peak CUDA memory. No truncation, nothing dropped from
the real run -- this only measures whether the longest sequence fits.

Usage: PYTHONPATH=. uv run python research/phase20_gate_m.py <dataset_key>
Writes results/phase20/gate_m_<key>.json; exit 1 on OOM or any error.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

from research.phase20_common import REPO_ROOT, SFT_DATASETS, notify, provenance, write_json


def main(key: str):
    os.environ.pop("WANDB_API_KEY", None)          # no W&B run for a gate
    ddir, prefix = SFT_DATASETS[key]
    import experiments.sft as sft_mod
    from core.data import build_sft_full_dataset as real_builder

    info = {}

    def longest_only(path, tok, limit=None):
        ds = real_builder(path, tok, None)
        if path.name.endswith("_train.jsonl"):
            lens = [len(r["input_ids"]) for r in ds]
            i = max(range(len(lens)), key=lens.__getitem__)
            info.update(longest_index=i, longest_full_len=lens[i], n_train=len(ds))
            return ds.select([i])
        return ds.select([0])                        # val: unused (eval never fires at 1 step)

    sft_mod.build_sft_full_dataset = longest_only
    orig_hp = sft_mod.SFTExperiment.hyperparams

    def hp(self):
        h = dict(orig_hp(self))
        h["accum"] = 1
        info["hyperparams_used"] = h
        return h
    sft_mod.SFTExperiment.hyperparams = hp

    from trl import SFTTrainer
    orig_train = SFTTrainer.train

    def train(self, *a, **k):
        ids = self.train_dataset[0]["input_ids"]
        lab = self.train_dataset[0]["labels"]
        info["trainer_seq_len"] = len(ids)
        info["trainer_loss_tokens"] = sum(l != -100 for l in lab)
        info["max_length_cfg"] = self.args.max_length
        info["completion_only_loss"] = self.completion_only_loss
        info["loss_type"] = self.args.loss_type
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = orig_train(self, *a, **k)
        info["step_seconds"] = time.time() - t0
        return out
    SFTTrainer.train = train

    sys.argv = ["train.py", "sft", "--adapter", "qlora", "--model", "Qwen/Qwen3-8B",
                "--data-dir", ddir, "--data-prefix", prefix,
                "--lora-r", "16", "--lora-alpha", "32", "--seed", "42",
                "--output-root", str(REPO_ROOT / "runs" / "phase20" / "gate_m"),
                "--tag", f"gate-m-{key}", "--sft-full", "--max-steps", "1"]
    import train as train_mod
    args = train_mod.parse_args()
    exp = train_mod.EXPERIMENTS[args.experiment](args)
    ok, err = True, None
    try:
        exp.run()
    except torch.cuda.OutOfMemoryError as e:
        ok, err = False, "OOM: " + str(e)[:500]
    except Exception:
        ok, err = False, traceback.format_exc()[-2000:]
    info.update(
        dataset=key, pass_=ok, error=err,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        gpu=torch.cuda.get_device_name(0),
        gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
    )
    if ok and info.get("trainer_seq_len") != info.get("longest_full_len"):
        ok = False
        info["pass_"] = False
        info["error"] = "trainer sequence length != rendered full length (truncation?)"
    p = write_json(f"gate_m_{key}.json", {"gate": "M", **info, **provenance(require_commit=False)})
    print(json.dumps({k: v for k, v in info.items() if k != "hyperparams_used"}, indent=1, default=str))
    print(f"[gate M] {key}: {'PASS' if ok else 'FAIL'} -> {p}", flush=True)
    if not ok:
        notify(f"MIRA phase20 Gate M FAILED for {key}: {str(info['error'])[:300]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
