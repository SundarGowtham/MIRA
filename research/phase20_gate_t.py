#!/usr/bin/env python
"""
research/phase20_gate_t.py -- Phase 20 Gate T (masking, before any training).

Tokenizes every example of every SFT dataset through the exact training
pipeline -- core.data.build_sft_full_dataset (the --sft-full builder) then
TRL 1.8.0 SFTTrainer._prepare_dataset with the SFTConfig the --sft-full path
builds (completion_only_loss=True, max_length=None) -- and asserts:
  1. no example is truncated;
  2. the rendered prompt is an exact token prefix of the rendered full sequence;
  3. the loss mask contains zero prompt tokens;
  4. the loss mask covers the whole assistant turn: the full <think>...</think>,
     the complete JSON answer through its closing brace, and <|im_end|>.
     Exception (Phase 20 pre-training decision, 2026-11-17): a capped row (a
     sampled completion that hit the 8,192-token cap, no </think>) is checked
     "as sampled": loss tokens == exactly the raw completion's tokens, no
     </think> and no <|im_end|> expected. Every other row is checked unchanged.
Reports per dataset: example count, max full length, median full length,
median loss tokens. Writes results/phase20/gate_t.json. Exit code 1 if any
assertion fails (the chain stops and pings).

Usage: PYTHONPATH=. uv run python research/phase20_gate_t.py [dataset ...]
"""
from __future__ import annotations

import statistics
import sys
import types
from pathlib import Path

from research.phase20_common import (REPO_ROOT, SFT_DATASETS, base_model_path, notify,
                                     provenance, write_json)

IM_END = "<|im_end|>"


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model_path())
    if tok.pad_token is None:          # identical to core/model.py
        tok.pad_token = tok.eos_token
    return tok


def sft_full_config(tmpdir: str):
    """The SFTConfig fields that affect tokenization, exactly as experiments/sft.py
    sets them on the --sft-full path."""
    from trl import SFTConfig
    return SFTConfig(output_dir=tmpdir, max_length=None, packing=False,
                     dataset_text_field="text", completion_only_loss=True,
                     report_to="none", dataset_num_proc=None)


def prepare(rows_path: Path, tok, cfg):
    """The --sft-full builder, then TRL's own _prepare_dataset (the code
    SFTTrainer.__init__ runs)."""
    from trl import SFTTrainer
    from core.data import build_sft_full_dataset
    ds = build_sft_full_dataset(rows_path, tok)
    fake = types.SimpleNamespace(completion_only_loss=True, chat_template=None,
                                 _is_vision_dataset=False, _is_vlm=False)
    out = SFTTrainer._prepare_dataset(fake, ds, tok, cfg, False, None, "train")
    return ds, out


def check_example(tok, raw_prompt: str, raw_completion: str, input_ids, labels) -> list[str]:
    errs = []
    prompt_ids = tok.apply_chat_template([{"role": "user", "content": raw_prompt}],
                                         tokenize=True, add_generation_prompt=True)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    full_ids = tok.apply_chat_template([{"role": "user", "content": raw_prompt},
                                        {"role": "assistant", "content": raw_completion}],
                                       tokenize=True, add_generation_prompt=False)
    if hasattr(full_ids, "input_ids"):
        full_ids = full_ids["input_ids"]
    full_ids, prompt_ids = list(full_ids), list(prompt_ids)
    from core.data import is_capped_completion
    capped = is_capped_completion(raw_completion)
    if capped:   # rendered as sampled: generation prompt + raw sampled text, nothing added
        full_ids = prompt_ids + list(tok(raw_completion, add_special_tokens=False)["input_ids"])
    # 1. no truncation
    if list(input_ids) != full_ids:
        errs.append(f"truncated_or_altered: pipeline len {len(input_ids)} vs rendered {len(full_ids)}")
    # 2. prompt is an exact token prefix
    if full_ids[: len(prompt_ids)] != prompt_ids:
        errs.append("prompt_not_token_prefix")
    # 3. zero prompt tokens in the loss
    n_p = len(prompt_ids)
    if any(l != -100 for l in labels[:n_p]):
        errs.append(f"prompt_tokens_in_loss: {sum(l != -100 for l in labels[:n_p])}")
    # 4. loss covers the whole assistant turn
    if any(l == -100 for l in labels[n_p:]):
        errs.append(f"assistant_tokens_masked: {sum(l == -100 for l in labels[n_p:])}")
    loss_ids = [l for l in labels if l != -100]
    loss_text = tok.decode(loss_ids, skip_special_tokens=False)
    if capped:
        if loss_text != raw_completion:
            errs.append("capped_loss_text_not_exact_sampled_text")
        if "</think>" in loss_text or IM_END in loss_text or loss_text.count("<think>") != 1:
            errs.append("capped_row_has_added_tokens")
        return errs
    else:
        reasoning = raw_completion.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip("\n")
        answer = raw_completion.split("</think>")[-1].lstrip("\n")
        expect = "<think>\n" + reasoning + "\n</think>\n\n" + answer + IM_END
        if not loss_text.startswith(expect):
            errs.append("loss_text_not_exact_assistant_turn")
        if raw_completion.count("<think>") != 1 or not raw_completion.lstrip().startswith("<think>"):
            errs.append("data_think_open_not_single_leading")
        if not answer.rstrip().endswith("}"):
            errs.append("answer_not_closed_by_brace")
    if not loss_text.rstrip("\n").endswith(IM_END):
        errs.append("im_end_not_in_loss")
    if loss_text.count(IM_END) != 1:
        errs.append(f"im_end_count_{loss_text.count(IM_END)}")
    return errs


def run(keys: list[str]) -> dict:
    import tempfile
    from core.data import load_jsonl
    tok = load_tokenizer()
    report = {}
    with tempfile.TemporaryDirectory() as td:
        cfg = sft_full_config(td)
        for key in keys:
            ddir, prefix = SFT_DATASETS[key]
            for split in ("train", "val"):
                path = REPO_ROOT / ddir / f"{prefix}_{split}.jsonl"
                raw = load_jsonl(path)
                _, prepared = prepare(path, tok, cfg)
                from core.data import is_capped_completion
                capped = [ex.get("target") for ex in raw if is_capped_completion(ex["completion"])]
                full_lens, loss_lens, failures = [], [], []
                for i, (ex, row) in enumerate(zip(raw, prepared)):
                    labels = row["labels"]
                    errs = check_example(tok, ex["prompt"], ex["completion"], row["input_ids"], labels)
                    full_lens.append(len(row["input_ids"]))
                    loss_lens.append(sum(l != -100 for l in labels))
                    if errs:
                        failures.append({"index": i, "target": ex.get("target"), "errors": errs,
                                         "full_len": len(row["input_ids"])})
                name = f"{key}/{split}"
                report[name] = {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "n_examples": len(raw), "n_prepared": len(prepared),
                    "max_full_len": max(full_lens), "median_full_len": statistics.median(full_lens),
                    "mean_full_len": statistics.mean(full_lens),
                    "median_loss_tokens": statistics.median(loss_lens),
                    "n_failures": len(failures), "failures": failures,
                    "capped_rows_rendered_as_sampled": capped,
                    "pass": len(failures) == 0 and len(raw) == len(prepared),
                }
                print(f"[gate T] {name}: n={len(raw)} max={max(full_lens)} "
                      f"median_full={statistics.median(full_lens)} "
                      f"median_loss={statistics.median(loss_lens)} failures={len(failures)}", flush=True)
                for f in failures:
                    print(f"    FAIL idx={f['index']} target={f['target']} {f['errors']}", flush=True)
    return report


def main():
    keys = sys.argv[1:] or list(SFT_DATASETS)
    report = run(keys)
    ok = all(v["pass"] for v in report.values())
    out = {"gate": "T", "pass": ok, "datasets": report, **provenance(require_commit=False)}
    p = write_json("gate_t.json", out)
    print(f"[gate T] {'PASS' if ok else 'FAIL'} -> {p}")
    if not ok:
        notify("MIRA phase20 Gate T FAILED: " + ", ".join(
            f"{k} {v['n_failures']}" for k, v in report.items() if not v["pass"]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
