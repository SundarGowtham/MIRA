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

    def hyperparams(self) -> dict:
        if self.cfg.smoke:
            return dict(epochs=1, batch_size=1, lr=5e-5, accum=1,
                        max_seq_len=1024, limit=8)
        return dict(epochs=3, batch_size=4, lr=2e-4, accum=8,
                    max_seq_len=2048, limit=None)

    def run(self) -> Path:
        h = self.hyperparams()
        self.init_wandb(extra_config=h)

        model, tok = load_with_adapter(
            self.cfg.model, self.cfg.adapter, self.cfg.smoke,
            init_from=self.args.init_from,
        )

        train_ds = build_sft_dataset(self.cfg.data_dir / "sft_train.jsonl", tok, h["limit"])
        val_ds   = build_sft_dataset(self.cfg.data_dir / "sft_val.jsonl", tok)
        print(f"[{self.run_name}] train={len(train_ds)} val={len(val_ds)}")

        sample_prompts = self._sample_prompts(tok)

        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=h["epochs"],
            per_device_train_batch_size=h["batch_size"],
            per_device_eval_batch_size=h["batch_size"],
            gradient_accumulation_steps=h["accum"],
            learning_rate=h["lr"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=5 if self.cfg.smoke else 25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            bf16=not self.cfg.smoke,
            gradient_checkpointing=not self.cfg.smoke,
            max_seq_length=h["max_seq_len"],
            packing=False,
            dataset_text_field="text",
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else "none",
            run_name=self.run_name,
            seed=self.cfg.seed,
        )

        callbacks = [GradientStatsCallback(log_every=25 if not self.cfg.smoke else 5)]
        if sample_prompts:
            callbacks.append(SampleCompletionCallback(sample_prompts, tok))

        trainer = SFTTrainer(
            model=model, args=sft_config,
            train_dataset=train_ds, eval_dataset=val_ds,
            processing_class=tok, callbacks=callbacks,
        )
        trainer.train()
        trainer.save_model(str(self.final_dir))
        tok.save_pretrained(str(self.final_dir))
        print(f"[{self.run_name}] saved → {self.final_dir}")
        return self.final_dir

    def _sample_prompts(self, tokenizer) -> list[str]:
        val_path = self.cfg.data_dir / "sft_val.jsonl"
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