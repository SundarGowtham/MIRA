# Ranker v2 reward capacity — 10-line summary

Source: `results/ranker_capacity.json` (`research/ranker_capacity_rssft.py`).
No new generation — scored RS-SFT's own existing 32-sample/target resampling
draws (`data/novelty/rssft_reference_{heldout,astral}.jsonl`), chunked into
4 synthetic groups of 8 per target to match the hardening-probe/Phase 12
group shape. `probe_hardening.capacity_metrics` unchanged.

1. **Combined capacity = 46.31%** (held-out 45.41%, ASTRAL 45.97%) — roughly 3x the checker's 14% (finding 1) / 16.4% (finding 9 baseline), same metric, same group size, both measured pre-RL on the policy's own samples.
2. Four channels carry the capacity: `driving_force_margin` (z-var 0.778, ceiling 0.907), `precursor_decomposition_match` (0.750, ceiling 0.909), `temperature_economy` (0.744, ceiling 0.876), `volatility_risk` (0.666, ceiling 0.787) — all near their own ceiling, not an artifact of a low ceiling.
3. `n_precursors` (0.155) and `slice_competing_phases` (0.120) are weak, 81.0%/84.0% zero-std groups. `precursor_instability` is nearly dead: z-var 0.026, 96.4% zero-std — the same signature as the validator's permanently-dead channels.
4. Gate failure rates (combined): balance-solver failures 18.8% (the largest single cause), `precursors_exist`/`charge_neutral`/`format_ok`/`temperature_physical` each 5–8%.
5. **Cross-check against E1's own W&B training log (run `238lefg3`, full 300 steps) — reported as asked, not smoothed over.** The two sources measure different statistics: this script's z-variance is driven almost entirely by how often a channel is tied within a group (z-normalization forces every non-tied group to ~unit variance, so z-var ≈ 1 − zero-std fraction), while the training log only exposes the raw, unnormalized per-channel std — a magnitude, not a tie-frequency. A decimal-for-decimal match was never going to be available from existing logs; rank order is the fair comparison.
6. **At the extremes, the two sources agree**: `driving_force_margin` is the strongest channel in both (offline z-var 0.778, rank 1; training mean std 0.221, rank 1), and `precursor_instability` is the weakest in both (offline z-var 0.026; training mean std 0.033 — both dead last).
7. **In the middle, two channels genuinely swap rank — this is the real mismatch.** `volatility_risk` reads as one of the three strongest channels offline (z-var 0.666, rank 4 of 7) but is one of the *weakest* in actual E1 training (mean std 0.090, rank 6 of 7, barely above dead `precursor_instability`). `n_precursors` reads as nearly dead offline (z-var 0.155, 81% zero-std, rank 5) but is the *second-strongest* channel throughout real training (mean std 0.160, rank 2, never dropping below 0.121 across all 12 logged points).
8. One plausible, unconfirmed explanation: the offline read scores RS-SFT's own behavior on 75 targets it never trained on; the training log reflects actual GDPO training on `rl_run3`'s targets as the policy drifts under the ranker reward over 300 steps. If the reward is actively pushing `n_precursors` variation (it's one of the 7 weighted channels), training could be *creating* the variance this channel lacks at init — consistent with the direction of the swap, but not verified here.
9. `train/frac_reward_zero_std` (all 7 channels simultaneously tied) averaged 0.67% across training — unsurprising on its own (needing every channel tied at once is a much rarer joint event than any single channel's own tie rate, consistent with but not independent confirmation of either ranking above).
10. Net: Ranker v2 clears the "can this reward discriminate" bar by a wide margin over the checker (46% vs 14–16%), and 2 of its 4 strongest offline channels (`driving_force_margin`, `precursor_decomposition_match`/`temperature_economy`) plausibly explain E1's steady reward climb and stable grad_norm — but `volatility_risk`'s offline strength doesn't carry into training, and `n_precursors`'s offline weakness doesn't either, in opposite directions. Don't cite this offline capacity number's per-channel breakdown as a direct description of what drove E1's training without flagging that caveat.

Files: `results/ranker_capacity.json`, `research/ranker_capacity_rssft.py`,
`run_logs/ranker_capacity_rssft.log`. No edits to `validator.py` or
`core/ranker.py`. Nothing deleted. Not committed.
