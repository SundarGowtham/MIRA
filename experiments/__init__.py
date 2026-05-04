from experiments.sft import SFTExperiment
from experiments.grpo import GRPOExperiment
from experiments.sft_grpo import SFTGRPOExperiment

EXPERIMENTS = {
    "sft":      SFTExperiment,
    "grpo":     GRPOExperiment,
    "sft-grpo": SFTGRPOExperiment,
}