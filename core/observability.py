from __future__ import annotations
import torch
from transformers import TrainerCallback


class GradientStatsCallback(TrainerCallback):
    """Logs per-layer-group gradient norms and update-to-weight ratios to W&B."""

    GROUPS = {
        "attn":      ("q_proj", "k_proj", "v_proj", "o_proj"),
        "mlp":       ("gate_proj", "up_proj", "down_proj"),
        "embed":     ("embed_tokens", "lm_head"),
        "layernorm": ("norm",),
    }

    def __init__(self, log_every: int = 25):
        self.log_every = log_every
        self._prev_norms: dict[str, float] = {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0 or model is None:
            return
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        stats = {}
        for group_name, patterns in self.GROUPS.items():
            grad_sq = 0.0
            weight_sq = 0.0
            n = 0
            for name, p in model.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                if not any(pat in name for pat in patterns):
                    continue
                grad_sq += p.grad.detach().pow(2).sum().item()
                weight_sq += p.detach().pow(2).sum().item()
                n += 1
            if n == 0:
                continue
            grad_norm = grad_sq ** 0.5
            weight_norm = weight_sq ** 0.5
            stats[f"grad_norm/{group_name}"] = grad_norm
            stats[f"weight_norm/{group_name}"] = weight_norm
            if weight_norm > 0:
                stats[f"update_ratio/{group_name}"] = grad_norm / weight_norm
        if stats:
            wandb.log(stats, step=state.global_step)


class EvalModeGuard(TrainerCallback):
    """
    Works around a TRL 1.8.0 bug: the continuous-batching generation path
    calls `unwrapped_model.train()` unconditionally after EVERY generation
    round (grpo_trainer.py:1685). During evaluate() that flips the model
    back to train mode after the first eval batch, so _prepare_inputs'
    `mode = "train" if self.model.training else "eval"` misroutes later
    eval batches through the TRAIN branch — train num_generations (8)
    instead of num_generations_eval (2) at the group view, which crashed
    gdpo-v4's first probe eval ("shape [-1, 8, 6] invalid for size 12").
    Force the correct mode at each loop boundary instead. Needs the
    trainer (its .model may be a wrapper distinct from the module we
    built), so it is added post-construction via trainer.add_callback.
    """

    def __init__(self, trainer):
        self._trainer = trainer

    def on_prediction_step(self, args, state, control, **kwargs):
        self._trainer.model.eval()

    def on_step_begin(self, args, state, control, **kwargs):
        self._trainer.model.train()


class SampleCompletionCallback(TrainerCallback):
    """Generates completions for fixed prompts each eval and logs to W&B as a table."""

    def __init__(self, prompts: list[str], tokenizer, max_new_tokens: int = 256):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        rows = []
        model.eval()
        with torch.no_grad():
            for prompt in self.prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                completion = self.tokenizer.decode(
                    out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
                )
                rows.append([state.global_step, prompt[:200], completion])
        model.train()

        table = wandb.Table(columns=["step", "prompt", "completion"], data=rows)
        wandb.log({"samples": table}, step=state.global_step)