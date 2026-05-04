from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


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
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.float32 if smoke else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
            device_map="auto" if not smoke else None,
        )
    return model


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    return get_peft_model(model, cfg)


def load_with_adapter(model_name: str, adapter: str, smoke: bool, init_from: str | None = None):
    """
    Returns (model, tokenizer). If init_from is a path, loads adapter weights from there.
    """
    tok = load_tokenizer(model_name)
    model = load_model(model_name, adapter, smoke)
    if adapter in ("lora", "qlora"):
        if init_from and init_from != "base":
            model = PeftModel.from_pretrained(model, init_from, is_trainable=True)
        else:
            model = attach_lora(model)
    return model, tok