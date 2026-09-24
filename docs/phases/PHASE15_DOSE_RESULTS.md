# Phase 15 Task 5 results — dose-response via teacher-forcing

Pre-registration: `docs/phases/PHASE15_DOSE_PREREG.md` (committed `0852277`,
before any GPU forward pass). Raw output: `results/dose_response/teacher_forcing_raw.json`
(75 ARROWS³ routes × 4 checkpoints: base, RS-SFT, GDPO-300, full SFT —
teacher-forced, exact log-probabilities, no sampling). Analysis:
`research/distributional/dose_response_analysis.py` → `results/dose_response/analysis.json`.

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
negative. This is a specification miscalibration in the pre-reg, not a code
bug — the teacher-forcing pipeline itself is validated (see the smoke test:
sensible top-1 predictions, correct per-token span extraction).

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

## Question 2 [C] — Quality alignment: **prediction NOT supported**

Regression of `Δlog p (GDPO − base)` on `phase_pure` (wt% ≥ 50%, n=49 pure /
26 impure) with base log p as covariate, n=75, cluster bootstrap over
precursor sets (10,000 resamples, seed 20260925):

| coefficient | value |
|---|---|
| intercept | 7.763 |
| `phase_pure` | **−0.133** |
| `logp_base` (covariate) | −0.070 |

`phase_pure` 95% CI: **[−2.887, 2.585]** — includes 0, and the point estimate
is negative, not positive. **Prediction not supported**: at matched base log
p, GDPO does not raise phase-pure routes' log-probability more than
phase-impure ones — the coefficient is indistinguishable from (and nominally
opposite the sign of) the predicted effect. Consistent with this project's
standing finding that the validator/GDPO signal tracks carbonate-vs-not, not
measured synthesis quality (Tasks 2b, 4d).

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
  `target_wt_pct` (measured synthesis yield) has **no residual effect** — CI
  includes 0. GDPO's carbonate-avoidance move is a class-level shift, not
  sensitive to whether the specific carbonate-containing route actually
  performs worse in the lab.

**This is the cleanest, most direct confirmation in Phase 15 of the causal
chain established in Tasks 2/2b/4d**: GDPO's differential pressure on
carbonates operates at the level of the chemical class (carbonate vs. not),
independent of real measured outcome, because it inherits it from
`thermodynamic_favorable`'s 0K/finite-T ΔG bias — not from anything in
ARROWS's yield data, which GDPO never saw during training.

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
| Q1 Leash | [C] | **degenerate at ε=−20** — no comparison group exists on the pre-registered total-logp scale (n_above=0), nor on the [E] per-token alternative (n_below=0). Neither confirmed nor falsified. |
| Q2 Quality alignment | [C] | **not supported** — `phase_pure` coefficient −0.133, CI [−2.887, 2.585], includes 0 and wrong-signed. |
| Q3 Top-5 (rescoped) | [C] | **supported**, 3/3 targets. |
| Q4 Policy carbonate | [C] | **both predictions supported** — carbonate status significant (CI excludes 0), yield not (CI includes 0) once carbonate status is in the model. |

Net: two of four pre-registered questions land clean (Q3, Q4); one is
uninformative by miscalibration rather than by a real null (Q1); one lands a
clear negative (Q2). Q4 is the headline — it is the first direct,
policy-level (not validator-level) confirmation that GDPO's carbonate
pressure is class-based and yield-blind, using real ARROWS³ measurements
GDPO never trained on.

No changes were made to `validator.py`, `core/ranker.py`, or `core/comparator.py`.
No new training run was launched. Task 6 (Precursor Genome check) remains
cancelled for this paper, noted as future work only.
