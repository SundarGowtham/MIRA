from experiments.sft import SFTExperiment
from experiments.grpo import GRPOExperiment
from experiments.gdpo import GDPOExperiment
from experiments.sft_grpo import SFTGRPOExperiment

EXPERIMENTS = {
    "sft":      SFTExperiment,
    "grpo":     GRPOExperiment,
    "gdpo":     GDPOExperiment,
    "sft-grpo": SFTGRPOExperiment,
}