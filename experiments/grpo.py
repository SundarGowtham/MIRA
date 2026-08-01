from __future__ import annotations
import os
from pathlib import Path
from trl import GRPOConfig, GRPOTrainer

from experiments.base import Experiment
from core.data import build_grpo_dataset
from core.model import load_with_adapter
from core.reward import load_validator, make_check_reward_fns
from core.observability import GradientStatsCallback
from validator import VALIDATOR_VERSION


class GRPOExperiment(Experiment):
    name = "grpo"

    @property
    def data_prefix(self) -> str:
        prefix = getattr(self.args, "data_prefix", None)
        return prefix if prefix else "sft"

    def hyperparams(self) -> dict:
        if self.cfg.smoke:
            return dict(epochs=1, batch_size=2, lr=1e-5, accum=1,
                        num_generations=2, max_prompt_len=512,
                        max_completion_len=256, limit=8, kl_beta=0.04)
        # G=8: GDPO's per-check z-normalization divides by group stds
        # estimated from G samples; at G=4 each std has ~40% relative
        # error. accum=16 -> effective batch 16 = two G=8 groups per step,
        # and generation_batch_size stays 16 (generation speed unaffected).
        # batch_size=1: the training forward materializes the full
        # 152k-vocab logits per sequence (~5GB fp32) and the backward
        # needs its gradient too — at batch 2 that chain is ~20GB and
        # OOM'd inside loss.backward() on the 32GB card (2026-07-31).
        # max_completion_len=8192: the trained format is multi-KB
        # think+JSON (the old 512 truncated everything). 6144 still
        # clipped ~20% of generations live on gdpo-v3 (clipped_ratio=0.2)
        # -> ParseFailure -> format-only signal; completions p99 ~6.4k, so
        # 8192 covers the tail. Memory at batch_size=1: ~10.2k-token seq ->
        # fp32 logits ~6.2GB + grad ~6.2GB + 5GB model ~ 24GB peak, fits.
        return dict(epochs=1, batch_size=1, lr=5e-6, accum=16,
                    num_generations=8, max_prompt_len=1024,
                    max_completion_len=8192, limit=2000, kl_beta=0.04)

    def run(self) -> Path:
        h = self.hyperparams()
        reward_aggregation = getattr(
            self.args, "reward_aggregation", "normalize_then_sum")
        self.init_wandb(extra_config={
            **h, "data_prefix": self.data_prefix,
            "reward_aggregation": reward_aggregation,
            "validator_version": VALIDATOR_VERSION,
        })

        model, tok = load_with_adapter(
            self.cfg.model, self.cfg.adapter, self.cfg.smoke,
            init_from=self.args.init_from,
            lora_r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
        )

        train_path = self.cfg.data_dir / f"{self.data_prefix}_train.jsonl"
        val_path   = self.cfg.data_dir / f"{self.data_prefix}_val.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                f"Pass --data-prefix sft_v2 to use SFT-v2 data."
            )

        train_ds = build_grpo_dataset(train_path, tok, h["limit"])
        val_ds   = build_grpo_dataset(val_path, tok)
        print(f"[{self.run_name}] data_prefix={self.data_prefix} "
              f"train={len(train_ds)} val={len(val_ds)}")

        validator = load_validator(
            formula_set_path=Path("data/cache/mp_formula_set.pkl"),
            pd_index_path=Path("data/cache/pd_index.json"),
            project_root=Path("."),
        )
        reward_funcs, reward_names, reward_weights = make_check_reward_fns(
            validator, dump_path=str(self.output_dir / "generations.jsonl"))
        print(f"[{self.run_name}] reward funcs: {reward_names} "
              f"(aggregation={reward_aggregation})")

        grpo_config = GRPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=h["epochs"],
            per_device_train_batch_size=h["batch_size"],
            gradient_accumulation_steps=h["accum"],
            learning_rate=h["lr"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            logging_steps=5 if self.cfg.smoke else 25,
            # save_strategy="epoch",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            # save_total_limit=2,
            bf16=not self.cfg.smoke,
            gradient_checkpointing=not self.cfg.smoke,
            # max_prompt_length=h["max_prompt_len"],
            max_completion_length=h["max_completion_len"],
            num_generations=h["num_generations"],
            generation_batch_size=h["batch_size"] * h["accum"],
            multi_objective_aggregation=reward_aggregation,
            reward_weights=reward_weights,
            log_completions=True,
            # Lockstep generate() lets the slowest (clipped, 6.1k-token)
            # completion gate every batch — dominant cost at ~900s/step.
            # Continuous batching retires finished sequences early.
            use_transformers_continuous_batching=True,
            transformers_continuous_batching_config={
                "use_cuda_graph": False,
                "max_memory_percent": 0.4,
            },
            temperature=0.9,
            top_p=0.95,
            beta=h["kl_beta"],
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else "none",
            run_name=self.run_name,
            seed=self.cfg.seed,
            optim="adamw_8bit",
            remove_unused_columns=False,
            
            # epsilon_high=5.0
        )

        trainer = GRPOTrainer(
            model=model, args=grpo_config,
            train_dataset=train_ds,
            reward_funcs=reward_funcs,
            processing_class=tok,
            callbacks=[GradientStatsCallback(log_every=25 if not self.cfg.smoke else 5)],
        )

        # If init_from points to a checkpoint directory, tell the trainer to resume states
        resume_path = self.args.init_from if (self.args.init_from and "checkpoint" in self.args.init_from) else None

        trainer.train(resume_from_checkpoint=resume_path)

        trainer.save_model(str(self.final_dir))
        tok.save_pretrained(str(self.final_dir))
        print(f"[{self.run_name}] saved → {self.final_dir}")
        return self.final_dir