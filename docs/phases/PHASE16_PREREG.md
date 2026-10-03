# Phase 16 pre-registration

Written before any of E1, E3, E3c or E4 launches. Once Gowtham approves, this file is committed unedited and its commit hash is recorded in every run's W&B config before any GPU job starts. No changes after the commit.

## Scope (frozen)

Phase 16 answers three questions, and nothing else gets built:

| Arm | Question it answers |
|---|---|
| E1: GDPO with the ranker reward, from RS-SFT | Does a different per-route reward escape the sharpening seen with the validator? |
| E3 vs E3c: novelty-filtered self-distillation vs a matched control | Can new recipes be put into the starting distribution? |
| E4: GDPO with the novelty reward, from the final E3 model | Once new recipes are in the starting distribution, does practice amplify them? |

**E2 (comparator reward) is dropped.** The comparator already scored below chance on the ARROWS³ robot data, and E1 already supplies a second trained reward. The comparator's `C3` temperature-window issue is recorded as a limitation in the paper's reward-design section and is not fixed further.

## Evaluation sets

- **ASTRAL, clean subset (primary):** the 22 of 35 targets with no prompt or answer overlap in any project dataset (`results/phase16/astral_overlap.json`). The full 35 are reported alongside as secondary. ASTRAL targets are excluded from all E3, E3c and E4 training data, so they are held out for every Phase 16 arm.
- **Novelty held-out split:** 10% of the RS-SFT target pool (seed 42), in `data/novelty/heldout_targets.json`, never sampled into E3 or E3c training data.

## Definitions

- **Gate pass rate:** share of a model's samples passing the novelty reward's validity gate. Reported for every model, so a novelty gain that comes from producing less valid output is visible.
- **Novelty rate:** among gate-passing samples, the share with at least one precursor in ≤ 17 of the 17,616 corpus routes, or a precursor set never used for that target in the corpus.
- **Support growth of model M relative to reference R, on a target set:** the share of M's distinct precursor sets (32 samples per target) that are absent from R's 200-sample run on the same targets.
- **Resampling rate of R:** the same quantity computed for a fresh 32-sample draw from R itself. This is the noise floor. Support growth counts only beyond it.

## Design choices (signed off by Gowtham)

1. Rarity threshold for the E3 filter and the novelty rate: ≤ 17 corpus routes (under 1 in 1,000).
2. Novelty weights: `N = 0.7 * max_p rarity(p) + 0.3 * set_novelty`.
3. At most 4 kept recipes per target after the E3 filter and deduplication; E3c is subsampled to the same size.
4. Held-out fraction: 10% of the RS-SFT target pool.
5. Samples per target for E3 and E3c sampling: 64 from base Qwen3-8B per round.
6. Step-200 fallback: if budget cannot cover 300 steps for E1, it stops at step 200 and is compared against Phase 12's saved checkpoint-200. E4 always runs 300 steps.

## Predictions

### 1. [C] E1 sharpens like the validator did

- **Predicted:** relative to RS-SFT, the carbonate-free share and ASTRAL rule-picked hits rise (clean-22 baseline 9/22; full-35 baseline 10/35).
- **Predicted:** no support growth beyond RS-SFT's resampling rate, on the ASTRAL targets and on the held-out split, reported separately.
- **Predicted:** ARROWS³ gate agreement no better than the carbonate-free baseline (Phase 15's `research/distributional/arrows_gate_scoring.py`, unchanged).
- **Does not support:** support growth beyond the resampling rate on either target set.

### 2. [C] E3 puts new recipes into the starting distribution; E3c does not, or does less

- **Predicted:** on the held-out split, E3's novelty rate exceeds E3c's, with a bootstrap 95% CI on the difference excluding zero.
- **Predicted:** on the held-out split, E3's support growth relative to base exceeds E3c's support growth relative to base.
- **Predicted:** E3's gate pass rate is within 5 percentage points of E3c's.
- **Does not support:** E3 ≤ E3c on either of the first two. That would mean the rarity filter adds nothing beyond sampling more. It is a reportable null and is not re-run with a different threshold.

### 3. [C] E4 amplifies only what E3 already samples

- **Predicted:** E4's novelty rate on the held-out split rises relative to E3's, and E4 shows no support growth relative to E3 beyond E3's resampling rate.
- **Does not support:** support growth relative to E3 beyond the resampling rate. That would mean practice reached recipes the starting model did not sample, through shared parameters. It is the prediction most likely to fail if the paper's geometric account is wrong.

### 4. [E] Quality of the new recipes (no direction predicted)

For E3, E3c and E4 on the ASTRAL targets (held out from all three):
- the share of samples matching ASTRAL's rule-picked set, and, where a sample matches a set the robot measured, that set's measured purity;
- ARROWS³ gate agreement and teacher-forcing results, with Phase 15's scripts unchanged.

No direction is predicted. This is the open question the paper asks: whether recipes that are new are also any good.

## Not covered here

The vLLM parity checks and the cost benchmark are preconditions for running, not predictions. If parity fails, runs use Hugging Face generation instead.