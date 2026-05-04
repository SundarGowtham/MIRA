from __future__ import annotations
from pathlib import Path

from experiments.base import Experiment
from experiments.sft import SFTExperiment
from experiments.grpo import GRPOExperiment


class SFTGRPOExperiment(Experiment):
    """Composite: run SFT (or reuse a checkpoint), then GRPO from there."""
    name = "sft-grpo"

    def run(self) -> Path:
        if self.args.sft_checkpoint:
            sft_final = Path(self.args.sft_checkpoint)
            print(f"[{self.run_name}] reusing SFT checkpoint: {sft_final}")
        else:
            print(f"[{self.run_name}] stage 1/2: SFT")
            sft_args = self._stage_args(tag="stage1-sft")
            sft_final = SFTExperiment(sft_args).run()

        print(f"[{self.run_name}] stage 2/2: GRPO from {sft_final}")
        grpo_args = self._stage_args(tag="stage2-grpo", init_from=str(sft_final))
        return GRPOExperiment(grpo_args).run()

    def _stage_args(self, tag: str, init_from: str | None = None):
        import copy
        a = copy.copy(self.args)
        a.tag = "-".join(filter(None, [self.cfg.tag, tag]))
        if init_from is not None:
            a.init_from = init_from
        return a