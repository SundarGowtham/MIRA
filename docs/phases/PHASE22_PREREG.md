# Phase 22 pre-registration: search, a learned lab verifier, and RL paid to explore

## Disclosed before registration

Known before this file: all Phase 20 results (`PHASE20_RESULTS.md`, commit c7c649d), including:
- Corpus SFT new-support +0.068 [+0.040, +0.099];
- ASTRAL hits at n = 32: Base 7/35, Corpus SFT 2/35, RS-SFT 6/35, E1-full @300 9/35;
- the ranker's 0.523 agreement on ARROWS3 tie-stratum pairs (finding 32).

Nothing below has been computed.

## Fixed definitions (shared by all blocks)

| item | value |
|---|---|
| recipe | canonical precursor set from `precset_of` in `research/support_growth_by_checkpoint.py` |
| validity gate | validator version 2 pass, as used in the E1 runs |
| fitness | ranker v2 score, applied only to recipes that pass the validity gate |
| sampling | vLLM, bf16 base + LoRA, temperature 0.9, top-p 0.95, max new tokens 8,192, Qwen3 thinking on, same system message as `stratified_difficulty_eval.py` |
| targets | ASTRAL 35 and the P-18A-2 held-out list used in Phases 18–21 |
| new-recipe share | share of a sample set whose recipe is absent from the before-model's 232-sample reference set for that target |
| CIs | 95%, cluster bootstrap over targets, 10,000 draws, seed 20261125 |
| ASTRAL hits | the finding 18/22 rule-picked metric |
| paired hit test | exact McNemar on per-target hit indicators |

## Stage 0 (CPU plus a short GPU check)

1. **Inventory:**
   - confirm every input below exists (Base and E1 adapters, `sft-rssft-full`, `sft-corpus-full`, Base's and `sft-rssft-full`'s 232-sample references, ASTRAL and ARROWS3 data);
   - record the training-target list E1 used.
2. **Finding 32:** read finding 32's results JSON and report, for the ranker's 0.523, the number of tie-stratum pairs and its 95% CI. If the file lacks a CI, compute one with a bootstrap over recipes (seed 20261125) and label it [E].
3. **Tie-stratum counts:** count the pairs in each split used in 22B (definitions below). If any primary split has fewer than 30 pairs, mark P-22B-1 untestable on that split.
4. **GPU estimates:** give a GPU-hour estimate for each GPU stage from the old runs' logs and a 1-target dry run of 22A.
5. **Gate S (22A prompt check):** run one generation of the 22A mutation prompt on 4 targets and report the parse rate of the children. It must be at least 70%. If it is lower, stop and ping with 5 example outputs.

## 22A. Verifier-guided evolutionary search (GPU generation only; no training)

**Model:** Base Qwen3-8B, no adapter. This is the same starting model as every "adds support" claim.

**Search loop, per target:**
1. **Generation 0:** draw 16 fresh samples (seed 2201) and keep those that parse.
2. **Generations 1–8:** pick 4 parents and generate 8 children per generation. Each child uses 2 parents shown in its prompt (random pair from the 4).
3. **Parent selection:** tournament selection on fitness among all valid recipes found so far for that target. Invalid recipes are never parents. If fewer than 2 valid recipes exist, parents are drawn from all parsed recipes.
4. **Budget:** 16 + 64 = 80 model calls per target.

**Mutation prompt (fixed):** the normal task prompt for the target, followed by the 2 parent recipes as JSON, each with its validity result and ranker score. Then this instruction: "Propose one new synthesis recipe for this target. You may swap a precursor, combine precursors from the two recipes, or change conditions. Return the JSON format as usual." Save the exact template to `research/phase22_prompts.py` before generation.

**Arms:**
- **22A-F (fitness selection):** as above.
- **22A-R (random parents):** identical, but parents are drawn uniformly from valid recipes, ignoring fitness. This separates in-context recombination from selection.
- **Floor:** 80 plain Base samples per target, seed 2203, with no parents in context. This is the budget-matched floor.

**Analyses:**
- **P-22A-1 (search adds support).** The new-recipe share of all 80 calls in 22A-F, minus the same share for the budget-matched floor, has a CI entirely above 0 on the held-out targets.
- **P-22A-2 (search does not improve lab quality).** Take the 32 children of generations 5–8 as 22A-F's n32 set for each ASTRAL target. Its ASTRAL hits do not differ from Base's existing n32 hits (paired McNemar, p ≥ 0.05).
- **Measurements (no prediction):**
  - the same two quantities for 22A-R;
  - the validity rate per generation;
  - mean fitness per generation;
  - for each new recipe, its level against Base's reference (Phase 21's definitions: structural, compositional, transformational);
  - novelty tiers (in `data/sft_v3` for any target, in the Kononova corpus, in none);
  - the share of children that copy a parent's recipe exactly.

## 22B. A learned lab verifier (CPU; the optional GPU arm needs Gowtham's OK)

**Pairs:**
- **ARROWS3:** pairs with measured yields, loaded with `research/distributional/arrows_gate_scoring.py`.
- **ASTRAL:** within-target pairs ordered by phase purity.
- **Tie stratum:** pairs whose two recipes have the same carbonate status (both use a carbonate precursor or neither does).

**Features (fixed):**
- every ranker v2 objective value;
- every validator v2 channel;
- precursor-class flags (carbonate, hydroxide, nitrate, oxide, other);
- the number of precursors;
- the maximum temperature.

The model sees the feature difference between the two recipes in a pair.

**Verifiers:**
- **V1:** L2 logistic regression (C = 1.0) on feature differences.
- **V2:** gradient boosting (depth 3, 200 trees, learning rate 0.05).
- **V3 (Eureka-style):**
  - **Setup:** Base Qwen3-8B (vLLM) writes Python scoring functions over a fixed feature API (the features above).
  - **Search:** 16 programs per round, 10 rounds. Each round shows the top 4 programs and their training agreement.
  - **Selection:** the final program is chosen by training agreement only.
  - **Cost:** this uses the GPU for a few hours. Run it after 22A finishes, or interleaved if the chain allows.

**Splits:**
- **Primary:** train on ARROWS3 (all pairs), test on ASTRAL tie-stratum pairs.
- **Secondary:** train on ASTRAL, test on ARROWS3 tie-stratum pairs; leave-one-target-out within ASTRAL.
- **Rule:** no hyperparameter is tuned on a test split.

**P-22B-1 (gate).** On the primary test split, the best of V1–V3 must pass both:
- (a) agreement > 0.5, with a CI (bootstrap over recipes) excluding 0.5;
- (b) a paired bootstrap on the same pairs shows its agreement minus the ranker's has a CI above 0.

Prediction: **fails.** 224 ASTRAL reactions and a few ARROWS3 targets are probably too little data.

If it fails, report the result as "no verifier learnable from the available robot data beat chance or the ranker", and skip 22C-2.

**Measurements:** all splits and verifiers, with the carbonate-free rule as a baseline.

## 22D. Where the good recipes sit, and can the model's own frequency rank them (GPU sampling plus CPU; runs right after 22A)

**Why:** for precursors the candidate space can be enumerated, so ranking may matter more than reaching. This block decides whether the recipes ASTRAL found better than convention are in Base's tail or outside it.

**Sampling:** Base, no adapter, 1,024 samples per ASTRAL target (fixed settings, seed 2204). If Stage 0 estimates this above 60 GPU-hours, use 512 and say so.

**Analyses:**
- **P-22D-1 (the good recipes are in the tail).** At least 18 of the 35 ASTRAL-picked sets appear at least once in the 1,024 samples. Report the per-target counts and, for sets never seen, the rule-of-three bound 3/n.
- **P-22D-2 (frequency as a verifier).** On ASTRAL within-target pairs and ARROWS3 pairs (tie stratum for both), score each recipe by its frequency in Base's samples (ASTRAL: the 1,024 set; ARROWS3: Base's existing 232 reference where the targets overlap, otherwise report untestable). Agreement with the lab ordering is > 0.5 with a CI excluding 0.5. Prediction: **fails**.
- **Measurements:** Good–Turing unseen mass (recipes seen exactly once divided by n) per target, on the 1,024 set and on Base's 232 reference; Chao1 richness estimate per target; how many of 22A's new recipes fall inside the 1,024 set.

## 22C-1. RL paid to explore (GPU, about 130 hours; needs Gowtham's OK after 22B and 22D report)

**Hold:** do not start this stage until I reply OK to report 2. Support may not be the bottleneck; 22B and 22D decide that.

**Run:** `gdpo-e1-novelty-rssft-full`, a copy of `scripts/run_gdpo_e1.sh` with exactly one change: one added GDPO reward channel. Everything else is unchanged from `gdpo-e1-ranker-rssft-full`:
- init `sft-rssft-full`;
- the ranker reward;
- validator v2;
- β 0;
- HF generation;
- 300 steps;
- same seed and data;
- checkpoints 100, 200 and 300 kept.

**Added channel: a count-based novelty bonus.**
- **Formula:** b = 1/sqrt(1 + N), where N is the number of times this recipe appeared for this training target in all previous steps' rollouts.
- **Gate:** b = 0 if the recipe fails the validity gate.
- **Counts:** update after each step. Save them to `runs/phase22/novelty_counts.json` at every checkpoint.
- **Normalization:** GDPO z-normalizes the channel within each group like the others, so its scale needs no tuning.

**Sampling:** n32 on all targets at steps 100, 200 and 300 (fixed settings).

**Analyses:**
- **P-22C-1 (the thesis prediction).** At step 300, the new-recipe rate against `sft-rssft-full`'s floor (the P-20-2 procedure) does not exceed the floor on held-out targets.
  - The thesis ("support has to come from outside the model") predicts this.
  - The exploration literature predicts the opposite.
  - Either outcome is reported as registered.
- **P-22C-2.** ASTRAL hits at step 300 do not differ from `sft-rssft-full`'s (paired McNemar, p ≥ 0.05).
- **Measurements:**
  - new-recipe rate at 100, 200 and 300 next to E1-full's;
  - the validity rate;
  - the level classification and novelty tiers of new recipes;
  - the carbonate share;
  - the per-step share of groups with nonzero novelty-channel variance.

## 22C-2. A small two-loop run (GPU, about 130 hours; runs ONLY if P-22B-1 passes, and needs Gowtham's OK)

**Run:** RL from `sft-corpus-full` (outside-data support) with the reward = the 22B gate-passing verifier's score, applied only to recipes that pass the validity gate. β 0.05, with Base as the KL reference. Otherwise the same as 22C-1 without the novelty channel.

**Evaluation:** on ASTRAL only, which the verifier never trained on.

**P-22C-3.** ASTRAL hits at step 300 exceed `sft-corpus-full`'s (paired McNemar, p < 0.05), and its new-support mass against Base's floor keeps a CI above 0.

## Reporting

- Write a dated `docs/phases/PHASE22_RESULTS.md`, with every registered result in the registered wording.
- Report a failed prediction as failed. Label anything decided after the commit [E].
- End with one paragraph on what each result does to the Phase 20 headline:
  - P-22A-1 holding means support can come from search;
  - P-22C-1 failing means support can come from RL with an exploration incentive;
  - P-22B-1 passing means a lab-agreeing reward is learnable from existing data;
  - P-22D-1 holding means better-than-convention recipes were already in Base's tail, so finding and ranking, not support, is the bottleneck.