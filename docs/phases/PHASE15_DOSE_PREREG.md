# Phase 15 Task 5 pre-registration — dose-response via teacher-forcing

*Written and committed before any GPU forward pass is run. Four
questions, all **[C]**. One iteration — no changes after seeing results.*

## Route set (Step 5a) — scope decision, stated explicitly

**Built from ARROWS³ only** (`data/external/arrows/ARROWS`, commit
`cb630e94...`, already fully processed in Task 4): all **75
(target, precursor-set) combinations** — 47 YBCO + 15 LTOPO + 13 NTMO —
each paired with its **best (max-yield) measured target-phase wt%**
across the temperatures that precursor set was tried at (using the
target-match rule locked in `docs/phases/PHASE15_ARROWS_PREREG.md`).
Precursor amounts: ARROWS's own declared `Precursor stoichiometry`
where present, the same solved-balanced-reaction fallback used
throughout Task 4 otherwise (LTOPO).

**Lee (Task 3) and ASTRAL are NOT incorporated**, despite the original
Task 5a instruction naming them as candidate sources "where targets
overlap." Checked directly: none of ARROWS's three targets (YBa₂Cu₃O₇,
LiTiOPO₄, Na₂Te₃Mo₃O₁₆) appear among ASTRAL's 35 targets, and cross-
referencing Lee's 80,806-record corpus for a compatible, differently-
labeled route set for these same three targets was judged out of scope
given the GPU-time budget. This is a scope decision, not an oversight —
stated here rather than left implicit.

**Consequence for question 3** (below): the original instructions-doc
wording references "the ASTRAL-predicted precursor" specifically. Since
ASTRAL is not part of this route set, question 3 is **rescoped** to
ARROWS's own outcome labels — stated explicitly in that question, not
silently substituted.

**Rendering**: each route's "precursor segment" is the exact JSON array
value of the `"precursors"` key from the training schema
(`stratified_difficulty_eval.py`'s `SYSTEM_MSG`) —
`[{"formula": "X1", "amount": A1}, {"formula": "X2", "amount": A2}, ...]`
— appended after a minimal, fixed `<think>\n</think>\n{"precursors": `
prefix common to every route and every checkpoint (so the "runway" into
JSON never differs between checkpoints or routes; log-probabilities are
read only over the array's own tokens, never the fixed prefix). Prompt:
`closed_prompt(target)` from `research/astral_model_generations.py`
(`SYSTEM_MSG` + `"\n\nTarget: " + target + "\n\nProvide your synthesis
route as a JSON object."`), rendered via each checkpoint's own tokenizer
`apply_chat_template([{"role": "user", "content": prompt}],
add_generation_prompt=True)` — identical to how these checkpoints were
prompted during training/eval generation, per `evaluate_batched.py`.

## Checkpoints

`Qwen/Qwen3-8B` (base), `runs/sft-qlora-rs-sft-from-base/final` (RS-SFT),
`runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300` (GDPO-300),
`runs/sft-qlora-sft-v3-2nd-rank16/final` (full SFT, reference only, not
part of any pre-registered question below).

## Question 1 [C] — Leash

**Is there a base log-probability below which GDPO's change (GDPO log p
− base log p, of the full precursor-segment span) is indistinguishable
from zero?** Fixed threshold, before measuring: **ε = −20** (natural log,
matching the instructions doc's own suggested value). **Prediction**:
among routes with base log p < ε, the mean GDPO − base change is not
significantly different from 0 (95% CI, cluster bootstrap over precursor
sets, includes 0); among routes with base log p ≥ ε, it is.

## Question 2 [C] — Quality alignment

**At matched base log-probability, did GDPO raise phase-pure routes more
than phase-impure ones?** "Phase-pure" defined here, before measuring,
as **target-phase wt% ≥ 50%** (a round, defensible majority-phase cutoff
on ARROWS's continuous yield label — stated explicitly since ARROWS has
no native pure/impure boolean). **Test**: linear regression of
`Δlog p (GDPO − base)` on `phase_pure` (binary) with `base log p` as a
covariate, across all 75 routes. **Prediction**: the `phase_pure`
coefficient is positive with a 95% CI (cluster bootstrap over precursor
sets) excluding 0.

## Question 3 [C] — Top-5 (rescoped to ARROWS, stated above)

**At the first token of the first precursor's formula string, is the
formula belonging to the target's highest-yielding precursor set (by
max wt%, ties broken by lower precursor count) in base's top-5 tokens?**
Computed per target (3 targets, one "best" precursor set each — not
per route). **Prediction, stated before measuring**: yes for at least
2 of 3 targets — published work on RL promoting already-likely tokens
predicts the eventually-best route is not a base long-shot at the very
first decision point, even before RS-SFT/GDPO act on it.

## Question 4 [C] — Policy-level carbonate test (added 2026-09-25)

**On ARROWS³ routes with measured yield, teacher-force the precursor
segment through RS-SFT and GDPO-300.** Same 75 routes, same rendering.

**Predictions, stated before measuring**:
1. `Δlog p (GDPO − RS-SFT)` differs by carbonate status (whether the
   route contains ≥1 of `Li2CO3, Na2CO3, K2CO3, BaCO3, SrCO3, CaCO3,
   MgCO3` — reusing this project's standing carbonate set).
2. **Conditional on carbonate status, `Δlog p (GDPO − RS-SFT)` is
   uncorrelated with target yield** — i.e., GDPO's policy-level move
   away from carbonates is not itself tracking real synthesis quality;
   it is a class-level shift, not a yield-sensitive one.

**Test**: linear regression of `Δlog p (GDPO − RS-SFT)` on
`carbonate_status` (binary) and `target_wt_pct` (continuous), across all
75 routes. **95% CI by cluster bootstrap over precursor sets, 10,000
resamples, fixed seed `20260925`.** Prediction 1 is supported if the
`carbonate_status` coefficient's CI excludes 0; prediction 2 is supported
if the `target_wt_pct` coefficient's CI includes 0 **once
`carbonate_status` is in the model** (i.e., no residual yield-sensitivity
after accounting for the class-level carbonate effect).

## Rules restated

- One design iteration. This file is not edited after Step 5c's forward
  passes are run.
- The route set, rendering convention, phase-pure threshold, carbonate
  set, and regression specifications above are locked and not adjusted
  after seeing results.
- No sampling anywhere in Step 5c — exact teacher-forced log-probabilities
  only, same prompts and same forced tokens across all four checkpoints.
- `validator.py`, `core/ranker.py`, `core/comparator.py` are not touched.
  No new training runs.

## Output

`results/dose_response/*.json`, `docs/phases/PHASE15_DOSE_RESULTS.md`.
Questions 1–4 tagged **[C]**; anything else **[E]**.
