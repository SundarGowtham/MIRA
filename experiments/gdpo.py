from experiments.grpo import GRPOExperiment


class GDPOExperiment(GRPOExperiment):
    """
    GRPO with GDPO reward aggregation (TRL: multi_objective_aggregation=
    "normalize_then_sum"). Each validator check's reward is z-normalized
    within its generation group BEFORE aggregation, so the factored
    per-check reward vector survives into the gradient. Classic scalar
    GRPO — the `grpo` experiment with --reward-aggregation
    sum_then_normalize — collapses distinct check combinations into
    identical advantages (arXiv 2601.05242).

    The per-check reward functions come from
    core.reward.make_check_reward_fns (shared cached validate, None ->
    NaN for can't-compute checks, generation dump to
    runs/<run>/generations.jsonl). Everything else is inherited.
    """
    name = "gdpo"

    def run(self):
        self.args.reward_aggregation = "normalize_then_sum"
        return super().run()
