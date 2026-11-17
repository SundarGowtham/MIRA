# Phase 20 pre-registration: retrain the SFT models correctly, then rerun RL from the corrected RS-SFT

## Why

Phase 19 Block 0 found that every SFT run in the project was trained with TRL `max_length = 1024` and `truncation_mode = keep_start`. The cap was set on 2026-05-17 from an audit of `sft_v2`, which had no reasoning traces (p99 858 tokens). Every later SFT dataset has `<think>` traces with median full lengths of 4,800 to 5,800 tokens. As a result:

- No SFT model ever trained on an answer. Each trained on the prompt plus the first ~180 (RS-SFT, E3, E3c) or ~550 (Corpus SFT) tokens of reasoning.
- The loss also covered prompt tokens. For RS-SFT, E3 and E3c, about 82% of trained tokens were the prompt.

TRL's default `max_length` is also 1024, so removing the argument alone would not fix this.

RL was not affected: TRL 1.8.0 GRPO has no prompt cap, and the completion cap of 8,192 tokens matches the sampling cap. The RS-SFT, E3 and E3c training data were generated from Base, so the data are fine. Only the training was wrong.

## Fixed decisions

### SFT objective

The SFT the paper describes is maximum likelihood of the data's completions given their prompts, so it is fixed to that:

- The loss is on assistant-turn tokens only: a prompt-completion conversational dataset with TRL `completion_only_loss=True`.
- No truncation: `max_length=None`.

These two changes are the only ones. Everything else is identical to the old runs:

| setting | value |
|---|---|
| data files | same as the old runs |
| init | Base Qwen3-8B |
| quantization | QLoRA NF4 |
| LoRA | r 16, α 32, dropout 0.05, all 7 modules in all 36 blocks |
| epochs | 3 |
| learning rate | 2e-4, cosine schedule, warmup ratio 0.1 |
| weight decay | 0.01 |
| max grad norm | 1.0 |
| optimizer | adamw_8bit |
| effective batch | 32 (per-device 1 × accumulation 32, so the same number of optimizer steps as before) |
| seed | same as the old runs |

The old code path stays in the repo for reproducibility. The new path is a separate function or flag.

### Models trained in this phase

| new tag | data | replaces |
|---|---|---|
| `sft-rssft-full` | `data/rs_sft` | RS-SFT |
| `sft-e3-full` | `data/e3` | E3 |
| `sft-e3c-full` | `data/e3c` | E3c |
| `gdpo-e1-ranker-rssft-full` | RL | E1, now from `sft-rssft-full` |
| `sft-corpus-full` (optional, last) | `data/sft_v3` | Corpus SFT |

### RL

`gdpo-e1-ranker-rssft-full` runs `scripts/run_gdpo_e1.sh` unchanged except for two things: `RS_SFT_INIT` points to `sft-rssft-full`, and the tag is new. That keeps the ranker reward, validator version 2, β 0, HF generation (`--use-vllm off`), 300 steps and the same seed.

Checkpoints 100, 200 and 300 must all survive, including across automatic restarts. For this run set `save_total_limit` high enough that no checkpoint is ever deleted (this changes no training math), and verify it before launch.

Phase 12 (checker) is **not** rerun. It used validator version 1, which has the ammonium-balance bug. A faithful rerun would reproduce a known bug, and a fixed rerun would change two things at once. It is also about 127 GPU-hours, which the schedule cannot fit.

Old Phase 12 and old E1 stay in the paper, described as "RL from the truncated-SFT init". That gives the RL claims a second starting point.

### Sampling

All sampling uses the existing vLLM scripts with the same settings:

| setting | value |
|---|---|
| backend | vLLM, bf16 base + LoRA |
| temperature | 0.9 |
| top-p | 0.95 |
| max new tokens | 8,192 |
| seed | 42 |
| targets | the ASTRAL 35 and the held-out list used in Phase 18 |

| sample set | models | samples per target |
|---|---|---|
| n32 | `sft-rssft-full`, `sft-e3-full`, `sft-e3c-full`, E1-full @100, @200, @300 | 32 |
| reference | `sft-rssft-full`, same grouping as the old `rssft_reference_*` | 232 |
| temperature fix (Stage 5) | old `corpus_sft`, old `gdpo_v3_ckpt300`, old `phase12_ckpt300`, old `e1_ckpt300`, ASTRAL only | 32 |

The temperature-fix set is needed because those four models' existing ASTRAL files (`astral_gen_n32_*.json`) were sampled at temperature 1.0 through HF. Every other model's ASTRAL file was sampled at 0.9 through vLLM.

## Stages (GPU order)

**Stage 0. Fixes, audits and gates (CPU plus a short GPU check).**
1. Implement the new SFT path. Add `tests/test_sft_masking.py` to hold Gate T as a regression test.
2. Write `docs/phases/PHASE20_FINDINGS_AUDIT.md`. Mark every numbered finding in CLAUDE.md as one of:
   - unaffected;
   - SFT-side affected;
   - RL from a truncated-SFT init only;
   - uses temperature-mixed ASTRAL sample files.

   Do not edit CLAUDE.md. Draft the text of a new CLAUDE.md finding for the truncation bug, with that list, at the end of the audit file, for Gowtham to paste in.
3. List every result file in `results/` whose computation mixes temperature 1.0 HF ASTRAL files with temperature 0.9 vLLM files.
4. Run Gates T and M. Give a per-stage GPU time estimate based on the old runs' logs (old E1 took about 127 h).
5. Write the standalone sampling script and `PHASE20_RUNPOD.md` (see kickoff).

**Stage 1. Training.** Train `sft-rssft-full`, then `sft-e3-full`, then `sft-e3c-full`. Run Gate R after each.

**Stage 2. Sampling.** Draw the n32 sets for the three new SFT models.

**Stage 2b. Scoring (GPU, about 1 h).**
- Rerun Phase 19 Block G on the three new SFT models.
- Run Phase 19 Blocks E and F1 for the steps Base → `sft-e3-full` and Base → `sft-e3c-full`, using the Phase 19 code and probe strings unchanged.
- Start the CPU analyses for P-20-1 and the SFT halves of the measurements.

**Stage 3. RL.** Run `gdpo-e1-ranker-rssft-full` for 300 steps.

**Stage 4. RL sampling and scoring.**
- Draw the E1-full @100, @200 and @300 n32 sets.
- Draw the `sft-rssft-full` reference (232 per target).
- Score the step `sft-rssft-full` → E1-full@300 with Blocks G, E and F1.

**Stage 5. Temperature fix.** Draw the temperature 0.9 ASTRAL resamples of the four old models.

**Stage 6 (optional).** Only if Stages 1–5 are done: train `sft-corpus-full`, run Gate R, then draw its n32 sets.

**Final analysis.** Run everything below, then write a dated `docs/phases/PHASE20_RESULTS.md`.

## Gates (automatic; a failure stops the chain and pings)

**Gate T (masking, before any training).** Tokenize every training example of every dataset through the exact training pipeline. Assert:
1. No example is truncated.
2. The rendered prompt is an exact token prefix of the rendered full sequence.
3. The loss mask contains zero prompt tokens.
4. The loss mask covers the whole assistant turn: the full `<think>…</think>`, the complete JSON answer through its closing brace, and `<|im_end|>`.

Report for each dataset: example count, maximum full length, median full length, and median loss tokens.

**Gate M (memory).** For each dataset, run one optimizer step on its single longest example. It must not run out of memory; report peak memory.

If it runs out of memory, try these in order:
1. a fused or chunked cross-entropy (for example Liger), keeping bf16 logits;
2. if that still fails, stop and ping.

Never truncate and never drop examples.

**Gate R (after each SFT run).**
1. The trainer's logged tokens per example (TRL's `num_tokens`, which includes prompt tokens) must be within 5% of the dataset's mean full length including the prompt. This guards against a silent truncation. Do not change this to loss tokens.
2. Take 8 samples on each of 8 held-out targets (64 completions). The parse rate must be at least 70%.

Report the parse rate and the median completion length either way.

## Predictions

Analyses run only after this file's commit exists. Run Block G first on every new model. If G fails for a model (median below 0.4), its P-20-3, P-20-4 and P-20-5 results are labeled exploratory.

**P-20-1 (SFT adds support).** Use the new-support measure of P-18A-2, with the same code (`research/phase18_a_transport_exact.py`), the same λ, the same floor definition and the same targets. The rate above Base's resampling floor is positive, with the lower bound of the same CI above 0, for each of `sft-rssft-full`, `sft-e3-full` and `sft-e3c-full`.

**P-20-2 (RL adds none).** E1-full at steps 100, 200 and 300 does not exceed `sft-rssft-full`'s resampling floor for new recipes. Use the same procedure as finding 33.

**P-20-3 (exact reach).** This is the registered version of Phase 19's exploratory check.
- Inputs: Phase 19's 671 probe strings (`results/phase19/probe_strings.jsonl`) and the exact string scores l from the Phase 19 G/E code.
- For each step, define Δ = l_after − l_before. Center Δ within each target by subtracting that target's mean Δ.
- A string is "sampled" if the before-model's reference set contains its recipe at least once. The reference sets are Base's 232 and `sft-rssft-full`'s 232.
- Usable targets have at least 3 unsampled strings and at least 1 sampled string.
- Per target, take mean centered Δ over unsampled strings minus mean centered Δ over sampled strings.
- The statistic is the median over usable targets, with a 95% CI from a 10,000-draw cluster bootstrap over targets, seed 20261116.

Prediction:
- The CI lies entirely below 0 for `sft-rssft-full` → E1-full@300.
- The CI lies entirely above 0 for Base → `sft-e3-full` and for Base → `sft-e3c-full`.

Report the number of usable targets for each step.

**P-20-4 (location).** L50 for `sft-rssft-full` → E1-full@300 is greater than L50 for Base → `sft-e3-full` and greater than L50 for Base → `sft-e3c-full`. Use the Block E code unchanged; both orderings must hold.

**P-20-5 (dimension).** This registers the direction Phase 19 F1 measured. R(4) for each SFT step (Base → `sft-e3-full`, Base → `sft-e3c-full`) is greater than R(4) for `sft-rssft-full` → E1-full@300. Use the F1 code unchanged, with L* from this phase's Block E.

## Measurements (no prediction; report all)

- **Probe-set coverage:** P-20-3, P-20-4 and P-20-5 reuse Phase 19's 671 probe strings, which were chosen from the old models' samples. For each new model, report the share of its samples whose recipe is in the probe set, next to the same share for the old model it replaces.

- **Old against new RS-SFT:** per-target recipe distributions, MMD with the 18E kernel.
- **ASTRAL rule-picked hits** for each new model, using the finding 18/22 metric.
- **Alkali-source shares** for `sft-rssft-full` and E1-full@300, following finding 24's wording rule.
- **The uniform shift:** whether the old SFT models' uniform +7 to +8 nat shift on answer strings appears for the new SFT models.
- **Weight geometry:** Phase 18C (E₆₄, effective rank) for the new adapters.
- **Phase 18 reruns:** 18A transport share and 18E posterior matching, rerun with the new models in place of the old SFT models.
- **Temperature sensitivity:** for each comparison listed in Stage 0 item 3, recompute it with the Stage 5 resamples and report the old and new values side by side.
- **Optional:** if Stage 6 ran, findings 10 and 16 recomputed for `sft-corpus-full`.

## Reporting

- Write a dated `PHASE20_RESULTS.md` with every registered result in the registered wording.
- For every old finding that Phase 20 touches, add one line: confirmed, changed, or reversed.
- Label anything decided after the commit [E].
