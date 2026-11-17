"""
tests/test_sft_masking.py -- Phase 20 Gate T as a regression test.

Tokenizes examples of every SFT dataset through the exact --sft-full training
pipeline (TRL 1.8.0 SFTTrainer._prepare_dataset, completion_only_loss=True,
max_length=None) and asserts: no truncation; the rendered prompt is an exact
token prefix of the full sequence; zero prompt tokens in the loss; the loss
covers the whole assistant turn (<think>...</think>, the JSON answer through
its closing brace, <|im_end|>); a capped row (no </think>, hit the sampling
cap) trains on exactly its sampled text with nothing added. Also asserts the OLD path is untouched
(max_length 1024, keep_start) so old runs stay reproducible.

The full-dataset gate (every example, written to results/phase20/gate_t.json)
is research/phase20_gate_t.py; this test checks the first N examples of each
file so it runs in about a minute.

Run: PYTHONPATH=. uv run python tests/test_sft_masking.py   (or pytest)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

N_PER_FILE = 25


def _datasets():
    from research.phase20_common import SFT_DATASETS
    for key, (ddir, prefix) in SFT_DATASETS.items():
        for split in ("train", "val"):
            yield key, split, REPO_ROOT / ddir / f"{prefix}_{split}.jsonl"


def test_sft_full_masking():
    from datasets import Dataset
    from core.data import load_jsonl
    from research import phase20_gate_t as G
    tok = G.load_tokenizer()
    failures = []
    with tempfile.TemporaryDirectory() as td:
        cfg = G.sft_full_config(td)
        assert cfg.max_length is None and cfg.completion_only_loss is True
        for key, split, path in _datasets():
            allr = load_jsonl(path)
            # first N ordinary rows + every capped row (rendered as sampled)
            raw = [ex for ex in allr if "</think>" in ex["completion"]][:N_PER_FILE] + \
                  [ex for ex in allr if "</think>" not in ex["completion"]]
            sub = Path(td) / f"{key}_{split}.jsonl"
            import json
            sub.write_text("".join(json.dumps(ex) + "\n" for ex in raw))
            _, prepared = G.prepare(sub, tok, cfg)
            assert len(prepared) == len(raw)
            for ex, row in zip(raw, prepared):
                errs = G.check_example(tok, ex["prompt"], ex["completion"], row["input_ids"], row["labels"])
                if errs:
                    failures.append((key, split, ex.get("target"), errs))
    assert not failures, failures[:5]


def test_old_path_unchanged():
    """The pre-Phase-20 SFT hyperparameters must be exactly what the old runs used."""
    from types import SimpleNamespace
    from experiments.sft import SFTExperiment
    args = SimpleNamespace(smoke=False, adapter="qlora", model="Qwen/Qwen3-8B", data_dir=Path("data"),
                           output_root=Path("runs"), seed=42, tag="t", sft_full=False)
    h = SFTExperiment(args).hyperparams()
    assert (h["batch_size"], h["accum"], h["max_seq_len"]) == (4, 8, 1024)
    args.sft_full = True
    h = SFTExperiment(args).hyperparams()
    assert (h["batch_size"], h["accum"], h["max_seq_len"]) == (1, 32, None)
    assert (h["epochs"], h["lr"], h["weight_decay"], h["warmup_ratio"], h["max_grad_norm"]) == (3, 2e-4, 0.01, 0.1, 1.0)


if __name__ == "__main__":
    test_old_path_unchanged()
    print("test_old_path_unchanged: OK")
    test_sft_full_masking()
    print("test_sft_full_masking: OK")
