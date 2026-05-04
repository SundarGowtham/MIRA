from __future__ import annotations
import os
from pathlib import Path
from trl import GRPOConfig, GRPOTrainer

from experiments.base import Experiment
from core.data import build_grpo_dataset
from core.model import load_with_adapter
from core.reward import load_validator, make_reward_fn
from core.observability import GradientStatsCallback


class GRPOExperiment(Experiment):
    name = "grpo"

    def hyperparams(self) -> dict:
        if self.cfg.smoke:
            return dict(epochs=1, batch_size=1, lr=1e-5, accum=1,
                        num_generations=4, max_prompt_len=512,
                        max_completion_len=256, limit=8, kl_beta=0.04)
        return dict(epochs=2, batch_size=2, lr=5e-6, accum=4,
                    num_generations=8, max_prompt_len=1024,
                    max_completion_len=512, limit=None, kl_beta=0.04)

    def run(self) -> Path:
        h = self.hyperparams()
        self.init_wandb(extra_config=h)

        model, tok = load_with_adapter(
            self.cfg.model, self.cfg.adapter, self.cfg.smoke,
            init_from=self.args.init_from,
        )

        train_ds = build_grpo_dataset(self.cfg.data_dir / "sft_train.jsonl", tok, h["limit"])
        val_ds   = build_grpo_dataset(self.cfg.data_dir / "sft_val.jsonl", tok)
        print(f"[{self.run_name}] train={len(train_ds)} val={len(val_ds)}")

        validator = load_validator(
            formula_set_path=Path("data/cache/mp_formula_set.pkl"),
            pd_cache_path=Path("data/cache/phase_diagrams.pkl"),
        )
        reward_fn = make_reward_fn(validator)

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
            save_strategy="epoch",
            save_total_limit=2,
            bf16=not self.cfg.smoke,
            gradient_checkpointing=not self.cfg.smoke,
            max_prompt_length=h["max_prompt_len"],
            max_completion_length=h["max_completion_len"],
            num_generations=h["num_generations"],
            temperature=0.9,
            top_p=0.95,
            beta=h["kl_beta"],
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else "none",
            run_name=self.run_name,
            seed=self.cfg.seed,
        )

        trainer = GRPOTrainer(
            model=model, args=grpo_config,
            train_dataset=train_ds,
            reward_funcs=[reward_fn],
            processing_class=tok,
            callbacks=[GradientStatsCallback(log_every=25 if not self.cfg.smoke else 5)],
        )
        trainer.train()
        trainer.save_model(str(self.final_dir))
        tok.save_pretrained(str(self.final_dir))
        print(f"[{self.run_name}] saved → {self.final_dir}")
        return self.final_dir