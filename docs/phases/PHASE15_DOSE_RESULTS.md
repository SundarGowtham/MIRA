# Phase 15 Task 5 results — dose-response via teacher-forcing

Pre-registration: `docs/phases/PHASE15_DOSE_PREREG.md` (committed `0852277`,
before any GPU forward pass). Raw output: `results/dose_response/teacher_forcing_raw.json`
(75 ARROWS³ routes × 4 checkpoints: base, RS-SFT, GDPO-300, full SFT —
teacher-forced, exact log-probabilities, no sampling). Analysis:
`research/distributional/dose_response_analysis.py` → `results/dose_response/analysis.json`.
Robustness/addition round (requested on review, all [E] except where noted):
`research/distributional/dose_response_robustness.py` → `results/dose_response/robustness.json`.

GPU confirmed free before the run (RTX 5090, 0% util, 2 MiB used). All four
questions below were locked before this file or `analysis.json` existed.

## Question 1 [C] — Leash: **test is degenerate as specified, not confirmed or falsified**

Pre-registered ε = −20 (natural log) on `total_logp` (the full precursor-array
span, mean 51 tokens). Result: **all 75/75 routes have base log p below ε**
(mean base log p = **−96.2**). The "above ε" bucket is empty (n=0). The
partition the question depends on cannot be formed — there is no comparison
group, so the leash hypothesis is neither confirmed nor falsified by this run.

**Cause, stated plainly**: ε=−20 was calibrated for a per-token or
short-sequence log-probability scale; `total_logp` here sums log p over an
entire JSON precursor array (~51 tokens), which is unavoidably far more
negative. **This ε value was a specification error in `PHASE15_INSTRUCTIONS.md`
itself, not an implementation choice made while running Task 5** — recorded
here so it is not mis-attributed to the pipeline. The teacher-forcing
pipeline itself is validated (see the smoke test: sensible top-1 predictions,
correct per-token span extraction).

**[E] post-hoc supplement (not a substitute for Q1, reported only for
completeness)**: re-running the identical ε=−20 split on **per-token mean**
log p instead of the summed total. Result: the opposite degeneracy — **all
75/75 routes are ABOVE ε** on the per-token scale (mean per-token base log p
≈ −1.9, far above −20). Neither scale produces a non-degenerate split at this
ε. For the record, the "all-above" group's mean per-token Δ(GDPO−base) =
**+0.274**, 95% CI [0.247, 0.301] (cluster bootstrap) — GDPO raises per-token
likelihood of the precursor segment almost uniformly, which is a separate,
unconditional observation, not a leash result.

**Honest conclusion**: Question 1 as pre-registered cannot be answered from
this data at ε=−20 in either natural reading of the threshold. No claim about
a leash effect is supported or refuted here.

## Question 2 [C] — Quality alignment: **uninformative**

Regression of `Δlog p (GDPO − base)` on `phase_pure` (wt% ≥ 50%, n=49 pure /
26 impure) with base log p as covariate, n=75, cluster bootstrap over
precursor sets (10,000 resamples, seed 20260925):

| coefficient | value |
|---|---|
| intercept | 7.763 |
| `phase_pure` | **−0.133** |
| `logp_base` (covariate) | −0.070 |

`phase_pure` 95% CI: **uninformative — CI spans −2.887 to +2.585 nats**, wide
enough to be consistent with a moderate effect in either direction. The point
estimate is negative, not positive, but the interval is far too wide to
distinguish "no effect," "a real negative effect," or "a real positive effect
smaller than 2.6 nats." At matched base log p, this test does not show GDPO
raising phase-pure routes' log-probability more than phase-impure ones, but it
does not establish the absence of such an effect either.

## Question 3 [C] — Top-5 (rescoped to ARROWS): **prediction SUPPORTED, 3/3**

Per target, the highest-yielding precursor set's first-listed precursor's
formula-string prefix was checked against base's top-5 tokens at that exact
forced position:

| target | best set (max wt%) | wt% | first precursor | base top-5 at that position | prefix match |
|---|---|---|---|---|---|
| YBCO | `Ba2(CuO2)3, Y2O3` | 100.0 | `Ba2(CuO2)3` | Y, YO, Ce, YS, **Ba** | yes |
| LTOPO | `Li2CO3, PH9(NO2)2, TiO2` | 74.0 | `Li2CO3` | **Li**, Ti, L, " Li", Fe | yes |
| NTMO | `MoO3, Na2O, TeO2` | 62.0 | `MoO3` | Na, Te, **Mo**, NaN, " Na" | yes |

3/3 ≥ 2/3 threshold. **Prediction supported.** Even before any fine-tuning,
base already assigns non-trivial probability to the eventually-best route's
opening move — RL/SFT are refining an already-plausible base distribution,
not discovering a base long-shot. (Caveat, consistent with the smoke test's
observation on a different YBCO route: base's actual top-1 for YBCO is often
a Y-source-first ordering; here the best set happens to start with the
Ba-compound and "Ba" is still base's rank-5 choice, so the match is real but
not dominant.)

## Question 4 [C] — Policy-level carbonate test: **both predictions SUPPORTED**

Regression of `Δlog p (GDPO − RS-SFT)` on `carbonate_status` (binary, 19
carbonate-containing / 56 non-carbonate) and `target_wt_pct` (continuous),
n=75, cluster bootstrap over precursor sets (10,000 resamples, seed
20260925):

| coefficient | value | 95% CI | excludes 0? |
|---|---|---|---|
| `carbonate_status` | **−1.979** | **[−3.554, −0.352]** | **yes** |
| `target_wt_pct` | +0.020 | [−0.006, 0.043] | no (includes 0) |

- **Prediction 1 supported**: carbonate status significantly predicts
  `Δlog p (GDPO−RS-SFT)`. Mean Δ for carbonate-containing routes = **+2.24**;
  for non-carbonate routes = **+4.33** — GDPO increases log-probability
  relative to RS-SFT for *both* groups (RS-SFT already moved away from full
  SFT's carbonate bias, per Phase 12/17), but carbonate-containing routes
  gain **~2.1 nats less**, i.e. GDPO further widens the RS-SFT gap between
  carbonate and non-carbonate routes.
- **Prediction 2 supported**: once `carbonate_status` is in the model,
  `target_wt_pct` (measured synthesis yield) has **no detectable effect** —
  CI includes 0. But the CI upper bound (**0.043 nats/wt%**) does not exclude
  a moderate effect; at the top of the range that would be ~4.3 nats across
  the 0–100 wt% span, comparable in size to the carbonate coefficient itself.
  "No detectable effect" is the accurate statement; "no effect" is not
  established.

Consistent with GDPO inheriting the thermodynamic channel's carbonate
preference (Tasks 2c, 4d); **this test does not establish the causal path** —
it shows the policy-level signature is compatible with that mechanism, not
that the mechanism is confirmed at the policy level.

### [E] Q4 robustness round (target control, carbonate definition, length)

Three follow-up checks, requested on review, all reading only the already-committed
`results/dose_response/teacher_forcing_raw.json`: `research/distributional/dose_response_robustness.py`
→ `results/dose_response/robustness.json`.

**1. Target control.** Carbonate counts by ARROWS target (narrow, standing
definition — alkali/alkaline-earth carbonates only):

| target | n total | n carbonate | n non-carbonate |
|---|---|---|---|
| YBCO | 47 | 13 | 34 |
| LTOPO | 15 | 3 | 12 |
| NTMO | 13 | 3 | 10 |

Re-running the Q4 regression with target fixed effects (YBCO, LTOPO dummies,
NTMO reference): `carbonate_status` coefficient **−2.348**, 95% CI
**[−3.893, −0.808]** — **survives**, excludes 0, and the point estimate is if
anything larger in magnitude than the no-FE result (−1.979). The carbonate
effect is not an artifact of YBCO's larger sample or different target-level
base rate.

**2. Broader carbonate definition.** The standing carbonate set
(`Li2CO3, Na2CO3, K2CO3, BaCO3, SrCO3, CaCO3, MgCO3`) is alkali/alkaline-earth
only. Redefining `has_carbonate` as *any* precursor containing the `CO3`
substring (catching `CuCO3` and `Y2(CO3)3`, both YBCO-only): **17/75 routes
reclassify** from non-carbonate to carbonate (all YBCO; narrow 19/75 → broad
36/75).

| | coefficient | 95% CI | survives? |
|---|---|---|---|
| broad definition, no target FE | −1.125 | [−2.422, 0.263] | **no** |
| broad definition, with target FE | −2.337 | [−3.623, −1.046] | **yes** |

Without target fixed effects the broader definition's effect is
**not** significant — the 17 reclassified routes are concentrated entirely
in YBCO, and folding a target-level (not carbonate-level) pattern into a
pooled binary washes out the coefficient. With target FE it is recovered and
close to the narrow-definition, target-FE result (−2.348). **Read**: the
carbonate effect is robust to this broader definition once target is
controlled for, but the broader definition is more entangled with target
identity than the narrow one, which is why FE matters more here.

**3. Length covariate.** Carbonate-containing routes are on average
**longer** (mean 56.1 base tokens vs 52.8 for non-carbonate, n=19 vs 56).
Adding `n_tokens_base` to the Q4 regression: `n_tokens_base` coefficient
**+0.250**, 95% CI [0.207, 0.304] — positive and significant. Because
mean Δ(GDPO−RS-SFT) is positive overall, a uniform per-token effect pushes
*longer* segments' Δ *up*; since carbonate routes are the longer ones, this
length effect works **against** finding a negative carbonate coefficient.
With length controlled for, `carbonate_status` is **−2.797**, 95% CI
[−3.657, −1.915] — **more negative**, not less, than the original Q4
estimate (−1.979). **The original coefficient is conservative with respect
to length**: correcting for the length confound makes the carbonate penalty
larger, not smaller.

## [E] Original Q3 on ASTRAL — predicted vs traditional, prompt-only

Requested on review: for four named ASTRAL targets, conditioned on the
prompt only (no route teacher-forced), report the rank and log p of the
ASTRAL-predicted first precursor's first token vs. the traditional first
precursor's first token, under base / RS-SFT / GDPO-300. Same rendering as
Task 5 (`closed_prompt` + `apply_chat_template` + the fixed
`<think>\n</think>\n{"precursors": ` prefix + `[{"formula": "`).
`research/distributional/dose_response_robustness.py` (part 4) →
`results/dose_response/robustness.json`.

| target | predicted[0] | traditional[0] |
|---|---|---|
| NaSrBO₃ | NaBO2 | Na2CO3 |
| BaLiBO₃ | LiBO2 | Li2CO3 |
| KLi(PO₃)₂ | KPO3 | K2CO3 |
| Li₂TiSiO₅ | SiO2 | SiO2 |

**The literal test is uninformative for 3 of 4 targets**: NaBO2/Na2CO3,
LiBO2/Li2CO3, and KPO3/K2CO3 each share the same leading cation symbol
("Na", "Li", "K"), so the tokenizer's **first token is identical** for
predicted and traditional in all three cases — rank and log p at that
position are, correctly, exactly equal (confirmed at all 3 checkpoints).
For Li₂TiSiO₅, predicted and traditional are literally the same formula
(`SiO2`), so equality there is expected and not informative either. As
requested, the numbers are:

| target | checkpoint | predicted rank | predicted logp | traditional rank | traditional logp |
|---|---|---|---|---|---|
| NaSrBO₃ | base | 1 | −0.002 | 1 | −0.002 |
| NaSrBO₃ | RS-SFT | 1 | −0.002 | 1 | −0.002 |
| NaSrBO₃ | GDPO-300 | 1 | −0.001 | 1 | −0.001 |
| BaLiBO₃ | base | 2 | −11.500 | 2 | −11.500 |
| BaLiBO₃ | RS-SFT | 2 | −13.500 | 2 | −13.500 |
| BaLiBO₃ | GDPO-300 | 4 | −16.625 | 4 | −16.625 |
| KLi(PO₃)₂ | base | 1 | −0.001 | 1 | −0.001 |
| KLi(PO₃)₂ | RS-SFT | 1 | −0.002 | 1 | −0.002 |
| KLi(PO₃)₂ | GDPO-300 | 1 | −0.001 | 1 | −0.001 |
| Li₂TiSiO₅ | base | 14 | −24.750 | 14 | −24.750 |
| Li₂TiSiO₅ | RS-SFT | 18 | −21.501 | 18 | −21.501 |
| Li₂TiSiO₅ | GDPO-300 | 15 | −21.251 | 15 | −21.251 |

**[E, added beyond the literal ask]** — since the first-token equality makes
the requested comparison uninformative as specified, the actual point of
divergence between predicted and traditional was also checked: the first
token *after* the shared cation prefix (e.g. "Na" → "BO2" vs "2CO3"), same
rendering, same 3 checkpoints:

| target | common prefix | checkpoint | predicted cont. | pred. rank | pred. logp | traditional cont. | trad. rank | trad. logp |
|---|---|---|---|---|---|---|---|---|
| NaSrBO₃ | Na | base | BO2 | 2 | −5.257 | 2CO3 | **1** | −0.007 |
| NaSrBO₃ | Na | RS-SFT | BO2 | 4 | −3.660 | 2CO3 | **1** | −0.410 |
| NaSrBO₃ | Na | GDPO-300 | BO2 | 4 | −4.481 | 2CO3 | **1** | −0.231 |
| BaLiBO₃ | Li | base | BO2 | 2 | −0.826 | 2CO3 | **1** | −0.576 |
| BaLiBO₃ | Li | RS-SFT | BO2 | 2 | −1.164 | 2CO3 | **1** | −0.414 |
| BaLiBO₃ | Li | GDPO-300 | BO2 | 2 | −1.526 | 2CO3 | **1** | −0.276 |
| KLi(PO₃)₂ | K | base | PO3 | 2 | −6.504 | 2CO3 | **1** | −0.004 |
| KLi(PO₃)₂ | K | RS-SFT | PO3 | 6 | −6.618 | 2CO3 | **1** | −0.118 |
| KLi(PO₃)₂ | K | GDPO-300 | PO3 | 5 | −5.826 | 2CO3 | **1** | −0.076 |

**At the actual point of divergence, the traditional (carbonate) continuation
is rank 1 under every checkpoint for all three targets, and the predicted
continuation is never better than rank 2** (and gets *worse* under RS-SFT/GDPO
for KLi(PO₃)₂: rank 2→6→5). This is a single-token, prompt-only, no-sampling
read at one fork in the sequence — it is not the same statistic as Phase
12/18's "predicted precursor set, N/35 at n=32" (which is a full-generation,
sampled-completion outcome over many tokens and many forks). The two are not
in tension: this probe shows the *local* prior at this one decision point
still favors the conventional continuation strongly across all three
checkpoints on these four held-out targets, consistent with the project's
standing account that conventional routes remain highly likely throughout —
RS-SFT/GDPO's aggregate gains on the predicted-set metric come from shifting
mass elsewhere in the generation, not from flipping this local token-level
preference on these specific targets.

## [E] Descriptive summary (all 75 routes)

| checkpoint | mean total log p |
|---|---|
| base | −96.19 |
| RS-SFT | −85.61 |
| GDPO-300 | −81.81 |
| full SFT (reference only) | −50.07 |

Mean Δ(GDPO−base) = +14.38 nats; mean Δ(GDPO−RS-SFT) = +3.80 nats. Full SFT's
log p is far higher than any other checkpoint's — expected, since full SFT is
trained directly on Kononova's corpus distribution (finding 10 of `CLAUDE.md`),
not on these three specific real-lab targets; not further interpreted here
since full SFT is explicitly reference-only per the pre-reg.

## Summary

| question | tag | result |
|---|---|---|
| Q1 Leash | [C] | **degenerate at ε=−20** (a specification error in `PHASE15_INSTRUCTIONS.md`, not a Task-5 implementation choice) — no comparison group exists on the pre-registered total-logp scale (n_above=0), nor on the [E] per-token alternative (n_below=0). Neither confirmed nor falsified. |
| Q2 Quality alignment | [C] | **uninformative** — `phase_pure` coefficient −0.133, CI [−2.887, 2.585] spans a wide enough range to be consistent with no effect or a moderate effect either direction. |
| Q3 Top-5 (rescoped) | [C] | **supported**, 3/3 targets. |
| Q4 Policy carbonate | [C] | **both predictions supported** — carbonate status significant (CI excludes 0) and **robust to target fixed effects, a broader carbonate definition (with target FE), and a length covariate** (all [E] follow-ups); yield shows no detectable effect (CI includes 0), though the CI upper bound does not rule out a moderate one. |

Net: two of four pre-registered questions land clean and hold up under
robustness checks (Q3, Q4); one is uninformative by miscalibration rather
than by a real null (Q1); one is uninformative by a too-wide interval, not a
clean negative (Q2). Q4 is the headline — it is the first direct,
policy-level (not validator-level) signature consistent with GDPO's
carbonate pressure being class-based and yield-blind, using real ARROWS³
measurements GDPO never trained on; this test does not itself establish the
causal mechanism. The [E] ASTRAL prompt-only probe additionally shows that,
at the one token-level fork actually tested, all three checkpoints still
strongly favor the conventional continuation on held-out targets — a local
read that is not in tension with the aggregate predicted-set gains reported
in Phase 12/18, since those are full-generation, sampled outcomes over many
forks, not this single position.

No changes were made to `validator.py`, `core/ranker.py`, or `core/comparator.py`.
No new training run was launched. Task 6 (Precursor Genome check) remains
cancelled for this paper, noted as future work only.
