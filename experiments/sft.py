from __future__ import annotations
import os
from pathlib import Path
from trl import SFTConfig, SFTTrainer

from experiments.base import Experiment
from core.data import build_sft_dataset, load_jsonl
from core.model import load_with_adapter
from core.observability import GradientStatsCallback, SampleCompletionCallback


class SFTExperiment(Experiment):
    name = "sft"

    @property
    def data_prefix(self) -> str:
        """Filename prefix for data files: <prefix>_{train,val,test}.jsonl.
        Default 'sft' preserves backward compatibility. Override via --data-prefix.
        """
        prefix = getattr(self.args, "data_prefix", None)
        return prefix if prefix else "sft"

    def hyperparams(self) -> dict:
        if self.cfg.smoke:
            return dict(
                epochs=1, batch_size=1, lr=5e-5, accum=1,
                max_seq_len=1024, limit=8,
                weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0,
                logging_steps=5, eval_steps=5, save_steps=20,
            )
        # Sized to the actual data (per audit_token_lengths.py):
        #   p99 full-sequence length = 858 tokens; max = 1115.
        #   max_seq_len=1024 truncates 0.11% (9 / 8005 examples) — negligible.
        # batch_size=4 + accum=8 = effective batch 32 (same as SFT-v1).
        # adamw_8bit halves optimizer-state memory.
        return dict(
            epochs=3, batch_size=4, lr=2e-4, accum=8,
            max_seq_len=1024, limit=None,
            weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0,
            logging_steps=25, eval_steps=100, save_steps=200,
        )

    def run(self) -> Path:
        h = self.hyperparams()
        self.init_wandb(extra_config={**h, "data_prefix": self.data_prefix})

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
                f"Use --data-prefix to point at a different filename stem "
                f"(e.g. --data-prefix sft_v2 looks for sft_v2_train.jsonl)."
            )

        train_ds = build_sft_dataset(train_path, tok, h["limit"])
        val_limit = h["limit"] * 2 if h["limit"] else None
        val_ds   = build_sft_dataset(val_path, tok, val_limit)
        print(f"[{self.run_name}] data_prefix={self.data_prefix}  "
              f"train={len(train_ds)} val={len(val_ds)}")

        sample_prompts = self._sample_prompts(tok)

        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=h["epochs"],
            per_device_train_batch_size=h["batch_size"],
            per_device_eval_batch_size=h["batch_size"],
            gradient_accumulation_steps=h["accum"],
            learning_rate=h["lr"],
            lr_scheduler_type="cosine",
            warmup_ratio=h["warmup_ratio"],
            weight_decay=h["weight_decay"],
            max_grad_norm=h["max_grad_norm"],
            logging_steps=h["logging_steps"],
            eval_strategy="steps",
            eval_steps=h["eval_steps"],
            save_strategy="steps",
            save_steps=h["save_steps"],
            save_total_limit=3,
            bf16=not self.cfg.smoke,
            gradient_checkpointing=not self.cfg.smoke,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=h["max_seq_len"],
            packing=False,
            dataset_text_field="text",
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else "none",
            run_name=self.run_name,
            seed=self.cfg.seed,
            optim="adamw_8bit",
        )

        callbacks = [GradientStatsCallback(log_every=25 if not self.cfg.smoke else 5)]
        if sample_prompts:
            callbacks.append(SampleCompletionCallback(sample_prompts, tok))

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tok,
            callbacks=callbacks,
        )
        trainer.train()
        trainer.save_model(str(self.final_dir))
        tok.save_pretrained(str(self.final_dir))
        print(f"[{self.run_name}] saved → {self.final_dir}")
        return self.final_dir

    def _sample_prompts(self, tokenizer) -> list[str]:
        val_path = self.cfg.data_dir / f"{self.data_prefix}_val.jsonl"
        if not val_path.exists():
            return []
        examples = load_jsonl(val_path)[:5]
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for ex in examples
        ]