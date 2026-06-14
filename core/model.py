from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, adapter: str, smoke: bool):
    """adapter ∈ {'full', 'lora', 'qlora'}. smoke disables quantization for CPU."""
    if adapter == "qlora" and not smoke:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)
        # prepare_model_for_kbit_training upcasts lm_head and LayerNorms to
        # fp32 for "training stability". On Qwen3 (vocab≈152k), this produces
        # fp32 logits — 2.5 GB per forward at batch=4 seq=1024, plus another
        # 2.5 GB for SFTTrainer's .contiguous() copy in compute_loss. That's
        # the OOM source. Cast them back to bf16; bf16 has ample numerical
        # precision for fine-tuning and matches the rest of the forward pass.
        for module_name, module in model.named_modules():
            if "lm_head" in module_name or "embed_tokens" in module_name:
                module.to(torch.bfloat16)
            elif "norm" in module_name.lower():
                module.to(torch.bfloat16)
    else:
        dtype = torch.float32 if smoke else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if not smoke else None,
            attn_implementation="sdpa"
        )
    return model


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, cfg)
    # PEFT initializes LoRA adapter weights in fp32 by default. On a 4-bit
    # quantized base, this causes the LoRA-wrapped linear projections to
    # output fp32, including the final lm_head — which blows up memory on
    # long-vocab models like Qwen3 (vocab≈152k):
    #   logits at fp32, batch=4, seq=1024 → 2.49 GB per forward
    #   logits at bf16, same shape       → 1.24 GB
    # plus the .contiguous() copy in compute_loss is also fp32 → another 2.49 GB
    # Casting LoRA params to bf16 keeps the whole forward in bf16.
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" in name.lower():
            p.data = p.data.to(torch.bfloat16)
    return model


def load_with_adapter(model_name: str, adapter: str, smoke: bool, init_from: str | None = None, lora_r: int | None = None, lora_alpha: int | None = None, lora_dropout: float | None = None,):
    """
    Returns (model, tokenizer). If init_from is a path, loads adapter weights from there.
    lora_r / lora_alpha / lora_dropout override the defaults in attach_lora when provided.
    """
    tok = load_tokenizer(model_name)
    model = load_model(model_name, adapter, smoke)
    if adapter in ("lora", "qlora"):
        if init_from and init_from != "base":
            model = PeftModel.from_pretrained(model, init_from, is_trainable=True)
            for name, p in model.named_parameters():
                if p.requires_grad and "lora_" in name.lower():
                    p.data = p.data.to(torch.bfloat16)
        else:
            # Build kwargs — only pass what was explicitly overridden
            lora_kwargs: dict = {}
            if lora_r is not None:
                lora_kwargs["r"] = lora_r
            if lora_alpha is not None:
                lora_kwargs["alpha"] = lora_alpha
            if lora_dropout is not None:
                lora_kwargs["dropout"] = lora_dropout
            model = attach_lora(model, **lora_kwargs)
    return model, tok