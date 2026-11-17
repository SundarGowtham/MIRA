# Phase 19 results

Pre-registration: `docs/phases/PHASE19_PREREG.md`, committed in `7412c23` before any Phase 19 computation (blob `4df94b3a`). Every script asserts that blob at startup, and every results JSON records `prereg_commit = 7412c23db0a833d0fa3d0e7e8d00d7273f462ccf`. Anything decided after that commit is labeled **[E]**. RL = on-policy GDPO; SFT = supervised training on a fixed set.

---

## 2026-11-16 — Block 0: inventory

Code: `research/phase19_block0_inventory.py` (CPU), `research/phase19_block0_gpu_smoke.py` (GPU timing and instrument checks), shared `research/phase19_common.py`, gradient sketch `research/phase19_grad.py`. Outputs: `results/phase19/inventory.json`, `results/phase19/probe_strings.jsonl`, `results/phase19/rl_update_groups_S6.jsonl`, `results/phase19/rl_update_groups_S7.jsonl`, `results/phase19/time_estimate.json`. Logs: `run_logs/phase19_block0.log`, `run_logs/phase19_block0_gpu.log`.

### Finding that comes before everything else: no SFT run ever trained on an answer

`experiments/sft.py` passes `max_length = max_seq_len = 1024` to TRL's `SFTConfig`, which truncates with `keep_start`. The comment beside it ("p99 full-sequence length = 858 tokens") describes an older dataset. Every SFT dataset used since then has `<think>` traces of several thousand tokens:

| run (step) | examples | prompt tokens (median) | `</think>` at token (median) | full length (median) | answers inside the first 1024 tokens | trained tokens per example (from `trainer_state.json`) |
|---|---|---|---|---|---|---|
| Corpus SFT (init of S5) | 1,217 | 477 | 4,267 | 4,825 | 0 | 1,027 |
| RS-SFT (S2) | 283 | 840 | 5,069 | 5,468 | 0 | 1,025 |
| E3 (S3) | 505 | 841 | 5,486 | 5,843 | 2 | 1,026 |
| E3c (S4) | 505 | 841 | 5,439 | 5,843 | 2 | 1,026 |

So every SFT model in this project was trained on the prompt plus the first ~180 (RS-SFT, E3, E3c) or ~550 (Corpus SFT) tokens of the reasoning trace, with the loss on every token including the prompt. None of them saw the JSON answer during training. Everything they changed about which recipes they write came from training on the opening of the reasoning. This bears on findings 16, 18, 31 and Phase 18A, which describe what "SFT" does. Block D uses the sequences **as trained** (first 1024 tokens), which is the faithful update direction, not a truncation introduced here. RL training was not truncated (`max_completion_length = 8192`; no prompt cap).

### 1. Probe strings

- 40 probe targets (`results/phase17/probe_targets.json`: 20 ASTRAL clean, 20 held out). **671 strings in total, not 1,200**: many targets have fewer than 30 distinct recipes across the 8 step-table models (range 3–30 per target; only 4 targets reach 30).
- Pooled count = sum of the 8 step-table models' counts (Base and RS-SFT 232 samples per target, the others 32).
- String = the most frequent literal precursor-array text `[...]` among raw completions that parse to the recipe (median 2 distinct literal forms per recipe). 29 recipes in the pooled top 30 had no raw text anywhere (they occur only in the four ASTRAL files that store extracted precursors, not text) and were skipped in favor of the next recipe; they are listed in `inventory.json`.
- Context, fixed before any GPU scoring and identical for every model: the chat-template prompt (`add_generation_prompt=True`, as every sampling script rendered it) + `<think>\n\n</think>\n\n{\n  "precursors": `. 100% of the parsed samples checked use exactly that answer preamble.
- Share of probe strings each model sampled at least once: Base 66.6%, RS-SFT 65.9%, E3 34.1%, E3c 35.8%, Phase 12 @300 30.1%, E1 @300 30.1%.

### 2. Adapters

All step-table adapters (Corpus SFT, RS-SFT, E3, E3c, GDPO-300, Phase 12 @300, E1 @300) are r=16, α=32, with LoRA on q/k/v/o/gate/up/down in **all 36 blocks**.

### 3. Training data for Block D

- S3 / S4: `data/e3/e3_train.jsonl`, `data/e3c/e3c_train.jsonl` (505 each). S2: `data/rs_sft/rs_sft_train.jsonl` (283).
- **Overlap with probe targets: none.** No probe target (ASTRAL or held out) is in E3's or E3c's training or validation set, so P-19D-3 keeps all **20** held-out targets.
- S6 / S7: first 50 groups = steps 0–24 of `runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl` and `runs/gdpo-qlora-gdpo-e1-ranker-rssft/generations.jsonl`, 16 rows per step (2 groups of 8), no duplicate-dump gaps. Advantages recomputed exactly as TRL 1.8.0 `normalize_then_sum` does: per-reward z within group (nanstd + 1e-4), weighted nansum (format_ok 0.2, each channel 1.0), z over the 16-completion generation batch, NaN → 0.
  - S6 (checker: format_ok + the 5 run-3 channels): 200 positive, 176 negative, 24 zero advantages; 3 groups entirely zero.
  - S7 (ranker v2: format_ok + 7 objectives): 221 positive, 173 negative, 6 zero.
- **E1's `generations.jsonl` breakdown includes all 7 ranker objectives** (`driving_force_margin`, `n_precursors`, `precursor_decomposition_match`, `precursor_instability`, `slice_competing_phases`, `temperature_economy`, `volatility_risk`) plus the gates and scalar reward.
- S5 (GDPO-300 from Corpus SFT) is not reconstructed: run 1 used a different reward configuration. S5 is secondary, so Block D runs S2 as its secondary step instead.

### 4. Instrument checks and time estimate

- **Hook sketch vs autograd:** the 8×8 sketch of one module (block 20 `v_proj`) formed by the hooks equals R_outᵀ (autograd `.grad`) R_in to relative error 0.003 under gradient checkpointing. No double counting.
- **Numerics:** bf16 teacher forcing. The same string scored with a KV-cached batch, a cached single sequence, and an uncached full sequence differs by up to ~1 nat (typically 0.0–0.3 nats on totals of −22 to −38). This is bf16 rounding that depends on computation shape. Every model and hybrid is scored with identical shapes, so comparisons are like for like; the noise is stated here once and applies to all blocks.
- **Long sequences:** the longest of the 400 S6 RL completions (9,044 tokens with prompt) backpropagates in 4.2 s at 21.1 GiB peak with gradient checkpointing. **No truncation is needed.** Mean RL sequence 6,214 tokens.
- **GPU time estimate:** G 0.02 h, E 0.9 h, D 1.4 h, F1 0.6 h; about 3 h for the chain. S2 and S5 fit, so Block E runs them as secondary steps; Block D runs S2 (not S5, see above). F2 (generation) is estimated separately before asking for approval.

---

## 2026-11-16 — Block G: does the exact string score track behavior?

Code: `research/phase19_g_gate.py`. Outputs: `results/phase19/string_scores.jsonl` (reused by E, D, F1), `results/phase19/gate.json`. Log: `run_logs/phase19_G.log`.

**Gate: median ≥ 0.4 for every model. PASSES.** No target was dropped (counts vary within every target for every model; n = 40).

| model | median per-target Spearman(l_m, count_m) | 95% CI (cluster bootstrap) | gated |
|---|---|---|---|
| Base | 0.462 | [0.391, 0.599] | yes |
| RS-SFT | 0.445 | [0.348, 0.568] | yes |
| E3 | 0.478 | [0.420, 0.632] | yes |
| E3c | 0.467 | [0.397, 0.530] | yes |
| Phase 12 @300 | 0.506 | [0.387, 0.593] | yes |
| E1 @300 | 0.470 | [0.397, 0.595] | yes |
| Corpus SFT (S5, secondary) | 0.435 | [0.382, 0.607] | no |
| GDPO-300 (S5, secondary) | 0.478 | [0.352, 0.576] | no |

All medians fall in Phase 17's answer-only range (0.45–0.57) or just below it. The gate is passed on the point estimates; several lower CI bounds sit below 0.4 (RS-SFT 0.348, Phase 12 0.387), so the margin is thin. E, D and F1 are reported as registered, not as [E].

---

## 2026-11-16 — Block E: where is the change written?

Code: `research/phase19_e_location.py`. Outputs: `results/phase19/location.json`, `results/phase19/figdata_location.json`, per-configuration scores `results/phase19/location_scores.jsonl`. Log: `run_logs/phase19_E.log`. 671 probe strings; recovery pooled over targets; CIs from a 10,000-draw cluster bootstrap over targets (seed 20261116).

### Registered predictions

- **P-19E-1 — HOLDS.** L50 is later for each RL step than for each SFT step, in all four orderings: S6 (26) > S3 (6), S6 > S4 (6), S7 (16) > S3, S7 > S4. Each ordering holds in 100% of bootstrap draws.
- **P-19E-2 — FAILS for both RL steps.** L50(S6) = 26, CI [26, 28]; L50(S7) = 16, CI [14, 16]. Neither reaches 27. By the registered "does not support" line, this result **does not support "RL is written late"** in the sense of the last quarter of the network. S6's interval touches 28, so S6 misses the bar by one block; S7 misses it clearly.

### Table

| step | method | before → after | rms Δ (nats) | mean Δ | L50 [CI] | L80 [CI] |
|---|---|---|---|---|---|---|
| S3 | SFT | Base → E3 | 8.61 | +7.22 | 6 [6, 6] | 14 [12, 14] |
| S4 | SFT | Base → E3c | 8.91 | +7.57 | 6 [6, 6] | 14 [14, 14] |
| S6 | RL | RS-SFT → Phase 12 @300 | 2.09 | +0.72 | 26 [26, 28] | 32 [30, 34] |
| S7 | RL | RS-SFT → E1 @300 | 1.72 | +0.48 | 16 [14, 16] | 26 [26, 28] |
| S2 (secondary) | SFT | Base → RS-SFT | 9.38 | +8.04 | 6 [6, 6] | 14 [12, 14] |
| S5 (secondary) | RL | Corpus SFT → GDPO-300 | 0.32 | +0.18 | 30 [28, 32] | 35 [35, 35] |

### Shape of the curves (bottom-up recovery R(L), from `figdata_location.json`)

- **SFT (S2, S3, S4) is a step, not a ramp.** Recovery jumps from ~0.2 at block 4 to ~0.65 at block 6 and is ~0.87 by block 14. Then it **collapses at blocks 22–24** (S3 0.36, S4 0.41, S2 0.01) before climbing back to 1.0. So SFT's lower-half change, combined with the Base upper half, reproduces most of the shift. But the block 22–26 adapters are needed to stay consistent with the blocks above them: putting those adapters in without the ones above undoes the effect. The single-block swaps [E] agree: block 6 alone recovers 0.59–0.65 and block 34 alone 0.50–0.66.
- **RL is a ramp.** S6 rises slowly (0.24 at block 12, 0.39 at 18–20, 0.57 at 26, 0.82 at 32). S7 rises earlier and smoother (0.44 at 12, 0.56 at 16, 0.82 at 26). No single RL block recovers more than 0.13 [E]: the RL change is spread thinly across many blocks, while SFT's is concentrated in a few.
- **Top-down hybrids** (before-model adapters below L, after above) mirror this: SFT recovery stays ≥ 0.6 until block 32 and drops sharply only at 34; RL recovery declines steadily from block 12 onward.
- **S5 (GDPO-300)** has a very small change (rms 0.32 nats, near the bf16 noise noted in Block 0) and the latest location (L50 30, L80 35); its curve is noisy at low depths.
- RL's change is about a quarter the size of SFT's in rms (1.7–2.1 vs 8.6–9.4 nats), and SFT's is mostly a uniform upward shift (mean Δ ≈ rms Δ): SFT raises the log-probability of almost every probe string by ~7–8 nats under the answer-only context.

### [E] residual-stream activation patching

Patching block L's output (all positions) from the after-model into the before-model gives **exactly** the bottom-up adapter-swap curve at every depth for S3, S4, S6 and S7. This is expected rather than a separate finding: with the after-model's adapters in blocks 0..L, the residual stream at block L's output is the after-model's at every position, so the two interventions are identical. The match confirms that both code paths compute the same thing; it adds no independent evidence.

### Reading

The pre-registered ordering holds cleanly: RL's change sits deeper than SFT's in all four comparisons. The "last quarter" claim fails. RL with the checker reward (S6) concentrates its change in blocks ~22–34; RL with the ranker reward (S7) is spread over blocks ~8–28. SFT's change (all three SFT steps alike) is concentrated near block 6 and near block 34, with a consistency requirement around blocks 22–26.

---

## 2026-11-16 — Block D: the coupling through shared weights

Code: `research/phase19_d_kernel.py`, sketch `research/phase19_grad.py`. Outputs: `results/phase19/kernel_predictions.json`, `kernel_meta.json`, `reach.json`, `spillover.json`, `figdata_kernel.json`. Sketch caches (not tracked): `runs/phase19_cache/sketches/`. Log: `run_logs/phase19_D.log`.

**Read this first: the registered sketch check FAILED.** The newly written Kronecker sketch (k=8 per side, 252 modules, 16,128 dims) was compared with exact autograd gradients on 50 random pairs of probe strings in 4 modules. Pearson: block 5 `q_proj` 0.733, block 17 `v_proj` 0.706, block 26 `down_proj` 0.669, block 33 `o_proj` 0.770. All are below the 0.95 bar. A single module's 64-dimensional sketch is too noisy at k=8. Block D's predictions use the sum over all 252 modules, whose sketch noise partly averages out, but the registered instrument check does not certify that. Every D number below is therefore conditional on an instrument that failed its pre-registered validation. An [E] seed-to-seed reliability check of the full kernel runs after the chain (`research/phase19_d_sketch_reliability.py`) and is reported in its own entry. The hook mechanics themselves are correct: Block 0 matched one module's sketch to R_outᵀ(autograd grad)R_in at relative error 0.003. The failure is the sketch's resolution, not a bug.

**Lazy check.** Pooled CKA between the before-model and after-model kernels over all 671 probe strings: S3 0.43, S4 0.53, S2 0.42, S7 0.62, **S6 0.87**. Only S6 clears 0.8. For S2, S3, S4 and S7 the linearized picture describes only the start of training, as the pre-registration says the write-up must state. Per-target CKA medians [E] are high everywhere (0.86–0.98): within a target the kernel keeps its shape, and what changes is the across-target structure.

### Registered predictions

- **P-19D-1 (first-order validity) — FAILS** as registered (needs S3, S4, S6 and S7). Median per-target Spearman(Δ̂, Δ):

  | step | median | 95% CI | holds? |
  |---|---|---|---|
  | S3 (SFT) | 0.053 | [−0.068, 0.226] | no |
  | S4 (SFT) | 0.101 | [0.027, 0.236] | yes |
  | S6 (RL, checker) | **0.429** | [0.375, 0.506] | yes |
  | S7 (RL, ranker) | **−0.147** | [−0.268, −0.035] | no, CI entirely below 0 |
  | S2 (SFT, secondary) | 0.104 | [−0.007, 0.227] | no |

  Only S6, the one step whose lazy check passes, is predicted well. SFT's predictions are near zero. Two reasons are visible in the data: the SFT update direction is computed from the prompt and the first ~180 reasoning tokens (as trained, Block 0), not from answers; and SFT's actual change is mostly a uniform +7 to +8 nat shift that a per-target rank correlation discards. S7's first-order prediction is anti-correlated with what happened.
- **P-19D-2 (RL's reach) — FAILS for both RL steps.** 29 targets usable for each step (≥ 3 recipes RS-SFT never sampled), so both steps are testable.
  - S6: median ratio 1.04, CI [0.83, 1.30], which includes 1. RL's update moves recipes RS-SFT never sampled about as much as ones it did.
  - S7: median ratio **1.20**, CI [1.06, 1.29], which excludes 1 on the **wrong side**. The ranker's update moves unsampled recipes *more* than sampled ones.
  - [E] Length-normalized: S6 0.99 [0.88, 1.15], S7 1.18 [1.00, 1.26]. Partial Spearman(|Δ̂|, unsampled | length, gradient norm, target): S6 0.007, S7 0.060. String length does not explain the result.
  - By the pre-registration's own reading ("D-2 fails"): RL's update does push unsampled recipes through shared weights, about as hard as sampled ones; it does not cross Phase 18A's floor in 300 steps. The limit is not a hard wall at the level of the gradient. With the failed sketch check, this reading is conditional.
- **P-19D-3 (SFT spillover) — FAILS.** 20 held-out targets (no overlap with E3/E3c training), 11 positives (recipes E3 sampled and Base never did), 303 negatives. AUC 0.658, CI [0.438, 0.833]; the CI includes 0.5. The point estimate leans the predicted way, but 11 positives cannot resolve it.

### [C-measure] which half of RL narrows

Slope of standardized Δ̂ on l_before, within target (positive = the update favors strings already likely = narrowing):

| step | full update | PSR only (A > 0) | NSR only (A < 0) |
|---|---|---|---|
| S6 | −0.005 [−0.015, 0.006] | **−0.020 [−0.035, −0.007]** | **+0.018 [0.005, 0.031]** |
| S7 | **+0.032 [0.021, 0.044]** | −0.011 [−0.026, 0.003] | **+0.019 [0.007, 0.032]** |
| SFT (S2, S3, S4) | −0.014, CI excludes 0 in all three | — | — |

In both RL runs **the negative-advantage half narrows** (positive slope, CI excludes 0), and the positive-advantage half does not (negative or null slope). This is Ren and Sutherland's squeezing effect, not Zhu et al.'s prediction. In S6 the two halves cancel; in S7 the negative half wins and the full update narrows. SFT's predicted change slightly favors less likely strings in all three SFT steps. PSR's predictions also correlate with the actual change (pooled Spearman S6 +0.23, S7 +0.17), while NSR's anti-correlate (S6 −0.14, S7 −0.21).

### [C-measure] effective dimension

Participation ratio of the probe-string kernel: Base 10.9 pooled (3.35 per-target median); RS-SFT 12.8 (3.74). Within one target, the model's answer strings move along about 3–4 independent gradient directions; across all 40 targets, about 11–13.

### [E] chemistry in the kernel

Cosine kernel, alkali-source family (carbonate 163, oxide 65, other 258 strings; alkali-containing targets only). Within-family minus across-family mean cosine: Base +0.097, RS-SFT +0.068; permutation p = 0.001 for both (1,000 within-target permutations, null mean ≈ 0). Recipes that source the alkali the same way share gradient directions, so pushing one carbonate route pushes the other carbonate routes for the same target.

---

## 2026-11-16 — Block F1: how many directions does the change need?

Code: `research/phase19_f1_dimension.py`. Output: `results/phase19/dimension.json`. Log: `run_logs/phase19_F1.log`. L* = L80 from Block E. Principal directions from δ at all positions on 20 targets (10 ASTRAL + 10 held out, seeded); patching at all positions on the other 20; 20 random subspaces per k as the null.

**P-19F-1 — FAILS (not evaluable as registered).** k80 does not exist for any step: no k ≤ 64 reaches R ≥ 0.8, so none of the four orderings can hold. Even patching the **whole** residual stream at L* reaches only 0.86 (S3), 0.87 (S4), 0.82 (S6) and **0.79** (S7) on the test half, because L80 was set on all 40 targets in Block E. So the 0.8 bar was barely reachable at k = 4,096 and out of reach at k ≤ 64.

| step | L* | R(1) | R(2) | R(4) | R(8) | R(16) | R(32) | R(64) | full residual | random-subspace max (any k) | [E] global vector |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S3 (SFT) | 14 | **0.61** | 0.70 | 0.73 | **0.75** | 0.71 | 0.69 | 0.60 | 0.86 | 0.13 | 0.42 |
| S4 (SFT) | 14 | **0.59** | 0.71 | **0.76** | 0.76 | 0.73 | 0.68 | 0.53 | 0.87 | 0.14 | 0.46 |
| S6 (RL) | 32 | 0.02 | 0.10 | 0.39 | 0.54 | 0.62 | 0.63 | **0.70** | 0.82 | 0.08 | 0.05 |
| S7 (RL) | 26 | 0.03 | **0.51** | 0.55 | 0.64 | 0.67 | 0.69 | **0.69** | 0.79 | 0.04 | 0.33 |

Every k beats its random-subspace null by a wide margin (null maximum ≤ 0.14), so the principal directions are real.

**Reading, as measured rather than as registered.** The direction of the registered prediction is reversed:
- **SFT's change is low-dimensional.** One direction at block 14 recovers ~0.6 of SFT's effect on the answer strings, and 4–8 directions recover ~0.75. The first component holds 93–97% of δ's energy. Adding more components *lowers* recovery on the held-out half (0.75 → 0.53–0.60 at k = 64), so components beyond ~8 fit target-specific variation that does not transfer.
- **RL's change needs more directions.** S6 needs 8 directions to reach 0.54 and 64 to reach 0.70; one direction does nothing (0.02). S7 jumps to 0.51 at k = 2 and then climbs slowly. The first component holds 75% (S6) or 35% (S7) of δ's energy, yet recovers almost nothing alone.
- A single global steering vector (the mean δ) recovers 0.42–0.46 of SFT's change but only 0.05 of S6's and 0.33 of S7's.

With Block E (SFT concentrated in a few blocks, RL spread across many), the picture is consistent: **SFT writes a few large, shared directions; RL writes many small ones, later in the network.**

---

## 2026-11-16 — [E] Block D: reliability of the full sketched kernel

Code: `research/phase19_d_sketch_reliability.py` (decided after the registered sketch check failed). Output: `results/phase19/sketch_reliability.json`. The probe strings at Base were sketched a second time with an independent seed (20261117), and S3's update direction was recomputed with it.

| quantity | seed A vs seed B |
|---|---|
| kernel entries, within-target pairs (Pearson) | 0.88 (≈ 0.94 vs truth if noise is independent) |
| kernel entries, 2,000 random pairs (Pearson) | 0.61 (≈ 0.78 vs truth) |
| gradient norms | 0.96 |
| cosine kernel, within target | 0.87 |
| **S3 predicted change Δ̂, pooled Spearman** | **0.35** |
| **S3 Δ̂, per-target median Spearman** | **0.33** |

The full kernel is fairly reliable within a target. But **the predicted changes Δ̂ are not**: two sketch seeds agree on S3's ranking of probe strings only at Spearman ~0.33–0.35. Δ̂ is the inner product of a probe sketch with the summed update direction. The update direction's signal is small next to the sketch noise (SFT's per-example gradients largely cancel, and RL's advantages are signed), so the noise dominates. **Block D's registered results (D-1, D-2, D-3) and the PSR/NSR slopes should not be cited as findings.** The sketch at k = 8 is too coarse for predicted changes. Seed-to-seed agreement for S6/S7 was not measured; S6's D-1 of 0.43 exceeds S3's seed-to-seed agreement, but without that measurement it is not certified either. A larger sketch (k = 32 per side, 16× the dimensions) or exact LoRA-parameter gradients would be needed to rerun D.

The parts of D that use the kernel itself, not Δ̂, survive this check better: effective dimension (participation ratio) and the alkali-family block structure use within-target kernel entries (seed-to-seed 0.87–0.88).

---

## 2026-11-16 — Phase 19 summary (chain 0 → G → E → D → F1)

| block | registered quantity | result |
|---|---|---|
| 0 | inventory | done; **every SFT run trained only on prompt + first ~180–550 reasoning tokens, never on answers** (TRL max_length = 1024) |
| G | gate: median Spearman ≥ 0.4, every model | **passes** (0.445–0.506) |
| E | P-19E-1: L50 later for RL than SFT, all 4 orderings | **holds** (RL 26 and 16 vs SFT 6 and 6) |
| E | P-19E-2: L50 ≥ 27 for S6 and S7 | **fails** (26, 16): does not support "RL is written in the last quarter" |
| D | sketch check ≥ 0.95 per module | **fails** (0.67–0.77) |
| D | P-19D-1: first-order validity, S3/S4/S6/S7 | **fails** (holds for S4 and S6 only); conditional on the failed sketch |
| D | P-19D-2: RL reach ratio < 1 | **fails** (S6 1.04 incl. 1; S7 1.20, CI above 1); conditional on the failed sketch |
| D | P-19D-3: SFT spillover AUC > 0.5 | **fails** (0.66, CI [0.44, 0.83]); conditional on the failed sketch |
| D [E] | Δ̂ reproducibility across sketch seeds | Spearman 0.35 for S3: D's predictions are not reliable |
| F1 | P-19F-1: k80 smaller for RL than SFT | **fails**: no k80 exists; the measured direction is reversed (SFT ~1–8 directions, RL more) |
| F2 | dose (generation) | **not run**: needs approval; estimate ~25 GPU-hours (18 adapter configurations × 40 targets × 32 samples at ~82 min each) |

What Phase 19 can support: RL's change to the answer distribution is written deeper than SFT's and spread across many blocks and many small directions. SFT's change is concentrated in a few blocks and one to a few directions and is mostly a uniform upward shift, and it was learned from the opening of the reasoning trace alone. What it cannot support: any statement about the coupling mechanism (Block D) at this sketch resolution.
