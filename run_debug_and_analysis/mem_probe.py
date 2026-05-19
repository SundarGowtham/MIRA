"""
mem_probe.py
------------
Load the model exactly as SFTExperiment does, then walk through what
sits in VRAM at each stage. Tells us where the OOM headroom is going.
"""

from __future__ import annotations
import gc
import torch
from pathlib import Path

def gb(x):
    return x / (1024 ** 3)


def show(stage: str):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    print(f"  [{stage:30s}] alloc={gb(alloc):.2f}GB  reserved={gb(reserv):.2f}GB  peak={gb(peak):.2f}GB")


def main():
    print("Importing...")
    from core.model import load_with_adapter
    from core.data import build_sft_dataset

    torch.cuda.reset_peak_memory_stats()
    show("start")

    print("\nLoading model (qlora)...")
    model, tok = load_with_adapter("Qwen/Qwen3-8B", "qlora", smoke=False, init_from=None)
    show("after model load")

    print(f"\nModel device map summary:")
    if hasattr(model, "hf_device_map"):
        from collections import Counter
        devs = Counter(model.hf_device_map.values())
        for d, n in devs.items():
            print(f"  device={d}: {n} modules")
    else:
        print(f"  model.device = {next(model.parameters()).device}")

    # Check the dtypes of trainable parameters
    print(f"\nTrainable parameter dtypes:")
    from collections import Counter
    dtypes = Counter()
    n_trainable = 0
    n_total_params = 0
    for name, p in model.named_parameters():
        n_total_params += p.numel()
        if p.requires_grad:
            dtypes[str(p.dtype)] += p.numel()
            n_trainable += p.numel()
    for dt, n in dtypes.items():
        print(f"  {dt}: {n/1e6:.1f}M params")
    print(f"  total trainable: {n_trainable/1e6:.1f}M / {n_total_params/1e6:.0f}M ({100*n_trainable/n_total_params:.2f}%)")

    print(f"\nGradient checkpointing enabled? "
          f"{getattr(model, 'is_gradient_checkpointing', False)}")
    if hasattr(model, "base_model"):
        base = model.base_model
        print(f"  base_model.is_gradient_checkpointing = "
              f"{getattr(base, 'is_gradient_checkpointing', 'n/a')}")

    print("\nBuilding train dataset (first batch only)...")
    ds = build_sft_dataset(Path("data/processed/sft_v2_train.jsonl"), tok, limit=4)
    show("after dataset build")

    # Manually tokenize one batch
    print("\nTokenizing 1 batch of 4 examples at max_seq_len=1024...")
    texts = [ds[i]["text"] for i in range(4)]
    enc = tok(texts, padding="max_length", truncation=True, max_length=1024,
              return_tensors="pt")
    input_ids = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].cuda()
    show("after tokenize")

    print(f"  input_ids shape: {input_ids.shape}  dtype: {input_ids.dtype}")

    # One forward pass
    print("\nRunning one forward pass...")
    model.train()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    show("after forward")
    print(f"  loss: {out.loss.item():.4f}  logits dtype: {out.logits.dtype}  shape: {out.logits.shape}")

    print("\nBackward pass...")
    out.loss.backward()
    show("after backward")

    del out, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()
    show("after cleanup")


if __name__ == "__main__":
    main()