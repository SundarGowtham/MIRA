# Phase 12 results — GDPO from RS-SFT

*Per `misc/PHASE12_INSTRUCTIONS.md`. Timestamped as results land.*

---

## Smoke gate — PASSED with one caveat, before the real launch (2026-09-08)

20-step smoke run from `runs/sft-qlora-rs-sft-from-base/final`, everything
else identical to Arm A. All 6 reward functions constructed correctly,
4/5 active channels showed real variance, `within_group_std/*` logging
confirmed populated, generations dumped exactly once per completion
(380/380 unique), no fatal crash, checkpoint saved cleanly, init resolved
to RS-SFT (not full SFT) confirmed directly from the launch command.

**Caveat flagged before launch**: `operation_order` was exactly 1.000
across all 311 scored completions in the smoke sample -- zero observed
variance. Historical zero-std-group rate is 78.8% (varies in ~21% of
groups); at ~39 groups, P(observing zero variance | baseline rate)
≈ 0.788³⁹ ≈ 1 in 10,000. **Not consistent with historical behavior --
a real change.** Mechanism: RS-SFT was trained on validator-filtered
survivors at bar 0.9, so every training example had good operation
ordering; the policy inherited that consistency by selection.
**Prediction recorded before the full run: `operation_order` stays dead
throughout.** (Not yet confirmed or falsified -- the run was killed at
step 96 for an unrelated reason before this could be checked at scale;
see below. Worth re-checking on the beta=0 relaunch.)

## Parse-failure investigation — before launch (2026-09-09)

Smoke sample showed 18.2% parse-failure rate, concentrated on
fractional/doped-composition targets (non-random attrition correlated
with the stratum the whole experiment is about). Traced to two root
causes in `core/reward.py::parse_completion`:
1. Fraction-literal numbers (`"amount": 5/12`) -- invalid JSON syntax,
   the model writes them wherever it wants a non-terminating decimal.
2. The naive first-`{`-to-last-`}` span crossing multiple distinct
   top-level JSON objects ("Extra data" errors).

Both repaired (fraction-literal regex substitution + balanced-brace
multi-candidate extraction, ported from `reward_geometry.py`'s
analysis-only extractor). Re-scored the existing smoke generations, no
new GPU: **18.2% -> 12.4%** (69 -> 47 failures / 380). Verified
programmatically (not sampled) that **46/47 remaining failures show a
truncation signature** (unbalanced braces or no closing `</think>` tag) --
the model running out of `max_new_tokens=8192` budget, disproportionately
on fractional/doped targets that apparently need more reasoning tokens to
plan. Only 1 residual (`AgNbO3`) is a genuine token-repetition/corruption
glitch. Confirmed both of Claude's specific sub-questions: failures do
correlate with hitting the completion-length cap, and 100% of the
original 69 failures came from `parse_completion` raising, never from
`validator.validate()` crashing on a route that parsed fine.

Added `tests/test_reward.py` (20 tests) covering the repair logic and the
per-stratum alerting below. Added `parse_fail_rate/<stratum>` logging
(via `target_strata` threaded through `make_check_reward_fns` /
`experiments/grpo.py`) plus a growth alert (`wandb.alert()` + the
project's ntfy channel, fires once per stratum crossing 30% failure at
n>=20) so this bias is visible live during training, not reconstructed
after the fact.

**Decision: launch at 12.4%, do not raise the completion-length cap.**
Reasoning (Claude): the bias is now instrumented and visible in real
time; 12.4% attrition is survivable while a confounded run isn't; and the
truncation is itself a finding -- the model needs more `<think>` budget
for doped/fractional targets specifically, corroborating the fractional
stratum being harder from a new, validator-independent direction. Raising
the cap would also change two variables against Arm A (init AND token
budget) instead of one.

---

## THE RUN, attempt 1 (beta=0.001) — KILLED at step 96 (2026-09-10)

`gdpo-qlora-gdpo-phase12-rssft`. GDPO from RS-SFT-from-base, validator
scorer, `data/rl_run3`, beta=0.001 (unchanged from Arm A), everything else
per `PHASE12_INSTRUCTIONS.md`. Confirmed live via the W&B API
(`gowthamsundar1998-supportvectors/mira/cy4c8waa`), not the console log
alone.

**KL estimator exploded across the first three logged steps:**

| step | loss | KL | reward | grad_norm |
|---|---|---|---|---|
| 25 | 0.043 | 1.32 | 3.88 | 616.0 |
| 50 | 0.142 | 97.2 | 3.80 | 1.84 |
| 75 | **2.07e8** | **1.99e11** | 3.68 | 0.072 |

- **Loss is ~99.99% KL penalty by step 75**: beta*KL = 0.001 * 1.99e11 ≈
  1.99e8, matching the logged loss of 2.07e8 almost exactly. The
  policy-gradient term is a rounding error by comparison.
- **Grad norm collapsed 616 -> 1.84 -> 0.072.** At step 25 the pre-clip
  gradient norm was 616x the clip threshold (max_grad_norm=1.0), so the
  applied update direction was the KL outlier, normalized -- not a
  coherent training signal. By step 75 the gradient is effectively zero:
  learning had stopped.
- **Reward is declining, not stable**: 3.88 -> 3.80 -> 3.68, monotone
  across all three points. (Read as "stable" in the first pass at this
  data -- it is not; three points moving the same direction is a trend,
  not noise, at this magnitude of everything else breaking.)
- **What did NOT break**: per-check within-group std looked completely
  normal (`thermodynamic_favorable` 0.20->0.14->0.11, `stoichiometry`
  ~0.15-0.21, `operation_order` exactly 0.0 throughout -- matching the
  smoke-gate prediction), `parse_fail_rate/fracxinterpolated` tracked
  14%->20%->20% as expected, and actual generated completions at step 95
  were read directly and are still coherent, well-formed JSON with sane
  chemistry reasoning -- no visible degeneration yet. The pathology is
  specific to the KL/loss computation, not (yet) visible in behavior.

**Mechanism** (TRL 1.8.0's actual source, `grpo_trainer.py` ~line 2937):
the KL estimator is the k3/Schulman estimator,
`per_token_kl = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1`,
unbounded above for any single token where the policy assigns much lower
probability than the reference. `epsilon_high=5.0` (this run's wide
DAPO-style clip) only clamps the *policy-loss* importance ratio, not this
KL term at all -- one outlier token in one completion can dominate a
batch-mean KL. To reach a batch-mean of 1.99e11 via `exp(r)` requires a
single token with r = ref_logp - policy_logp ≈ 35-37 nats (base assigning
real probability to a token RS-SFT has driven to ~1e-16).

**Interpretation, not just a bug**: RS-SFT was fine-tuned on its own
validator-filtered survivors (bar 0.9, 73.75% survival) and sharpened
hard as a result -- that's the same mechanism finding 18 already
measured as a *good* thing (RS-SFT moved further from corpus convention
than base itself). Sharpening away from base is exactly what produces
near-zero-probability tokens under a base reference. The k3 estimator is
unbiased but has infinite variance under heavy-tailed log-ratios, so the
true KL is large-but-finite while the *estimator* explodes. The blowup is
consistent with the support-change intervention having worked, surfacing
through the one instrument (KL) that wasn't being watched as closely as
reward/capacity.

**Root-cause token hunt — inconclusive on magnitude, suggestive on
direction, reported honestly rather than force-fit to the hypothesis.**
Diagnostic (`research/diagnose_kl_blowup.py`): computed per-token
r = ref_logp - policy_logp for **all 476** completions generated in steps
50-75, using base Qwen3-8B as reference and the RS-SFT-from-base
checkpoint itself as an **approximate stand-in for the step-75 policy**
(no checkpoint was saved before the step-96 kill, since save_steps=100 --
the exact step-75 weights are not recoverable).

Top 10 highest-r tokens found (full results: `results/kl_blowup_diagnostic.json`):

| r (nats) | target | token | context |
|---|---|---|---|
| 15.88 | (Ba0.5Sr0.5)3Co2Fe24O41 | `2` | "...2O19 and SrFe12O19). So when referencing..." |
| 14.50 | Sr10Bi6O24 | `3` | "...0 + 6x = -43.2? Wait, maybe I..." |
| 9.06 | La0.7Ca0.2Sr0.1MnO3 | `"` | "...+ 0.1 + 0"1 = 1.1 per..." |
| 7.72 | Sr1.96Eu0.04Al2SiO7 | ` phase` | "...oxidation states and any explicit phase names..." |
| 7.25 | Ca(Zn0.3333Nb0.6667)O3 | `"}\n` | "...'inert', 'media': 'ar'}..." |
| 7.20 | La0.85K0.15MnO3 | `'t` | "...35 / 2 = 0't75 moles..." |
| 6.90 | Ba6Mn24O48 | `0` | "...12 + 20avg_Mn -96 =..." |
| 6.88 | Pb(Mg0.5W0.5)O3 | `/` | "...Mg(NO3)2, 0/5 WO3. But handling..." |
| 6.88 | Sr2Ti0.9Co0.1O4 | ` ` | "...0.9 TiO2 + 0 0.1 CoO -> Sr..." |
| 6.66 | Nd10Mo6O33 | `9` | "...MoO3 melts around 795C. So the calcining..." |

**Max r found: 15.88 -- well short of the ~35-37 nats a single token would
need to explain a 1.99e11 batch-mean KL under a naive mean-over-tokens
reduction.** The specific "one token, r~35" mechanism is **not confirmed**
at this magnitude. But the result is not a clean refutation either: **7 of
the top 10 hits are digit/numeric tokens embedded in fractional-
stoichiometry arithmetic** (mid-reasoning coefficient math on doped/
fractional targets specifically -- exactly the stratum with the elevated
parse-failure rate), directionally consistent with the hypothesis even
though the magnitude doesn't close the gap.

Most likely explanation for the gap: the stated approximation limit.
Using RS-SFT's **step-0 init** weights as a stand-in for the actual
**step-75** policy may materially understate how sharp the real policy had
become -- 75 real (if norm-clipped) gradient steps, especially once the KL
term started dominating the loss around step 50, could plausibly have
driven the policy sharper on the offending token(s) than the untrained
checkpoint shows, in a compounding feedback loop (loss dominated by KL ->
gradient chases the KL outlier -> policy sharpens further on it -> KL
grows further). It's also possible the true aggregate isn't driven by one
single extreme token at all, but by a broader shift across many
moderately-elevated-r tokens (in which case "reach r~35" is the wrong
bar to test against in the first place). **Reported as inconclusive, not
resolved** -- the qualitative fractional/numeric-token pattern is worth
keeping as a lead, not a finding.

**Fix: beta = 0 for the relaunch**, not a per-token clamp. Three reasons,
none of them a second variable against Arm A:
1. **Numerically absent in Arm A already.** Arm A's own KL ran
   0.001-0.003, so beta*KL ~ 2e-6 against a loss of order 0.03 -- removing
   a term that contributed one part in ~10,000 changes Arm A's result by
   nothing measurable. "Init + beta" was never really two live variables;
   it was one variable plus removing a term that was already zero in the
   baseline and is pathological in the treatment.
2. **Standard practice** -- DAPO drops the KL term entirely for
   long-horizon RLVR once the policy is meant to move away from the
   reference.
3. **The reference was already established not to bind at beta=0.001**
   (finding 17, the Phase 11 Step 1 correction) -- the leash was never
   doing anything at this beta. Now it is not doing anything *and* it is
   numerically exploding. Removing it costs nothing real and removes a
   pure liability.

Relaunched as `gdpo-phase12-rssft-beta0` (`scripts/run_gdpo_phase12_beta0.sh`),
one config line changed (`--kl-beta 0.0`), everything else identical.
Pre-registered read is unchanged.

---

## beta=0 relaunch — training health, checkpoint 300 (2026-09-16)

`gdpo-qlora-gdpo-phase12-rssft-beta0`. No repeat of the KL pathology: zero
`train/kl` metric logged at any point (beta=0 correctly removes the term
from TRL's loss computation entirely, confirmed via the run's live W&B
config, not assumed from the CLI flag). Loss stable throughout (0.033-0.050
range), reward trending mildly upward (3.91 -> 4.32 across steps 50-175),
grad_norm never exceeded ~0.45 (vs. the 616 spike in the killed beta=0.001
run). Checkpoints 100/200/300 all saved cleanly.

Fixed 30-target probe evaluation, within-group std across 4 readings
(steps 50/150/200/250): **`operation_order` confirmed dead at 0.000 on
every single reading** -- matches the smoke-gate prediction exactly
(RS-SFT was filtered by validator bar 0.9, so it inherited near-perfect
operation ordering by selection, and on-policy RL cannot revive a channel
it never fails). `chempot_atmosphere` picked up a small amount of life late
(0.0 -> 0.0 -> 0.028 -> 0.031) that wasn't expected. `stoichiometry` stable
(~0.05-0.06 throughout). `amount_accuracy` and `thermodynamic_favorable`
noisy, no clean monotonic trend either direction across only 4 points.
`parse_fail_rate/fracxinterpolated` held steady around 16%, well under the
30% alert threshold throughout -- no alerts fired.

**The run was stopped deliberately just past step 300** (the pre-registered
evaluation point), not resumed toward the full 600. This is the final read
for Phase 12, not an interim one -- see "Phase 12 is settled" below.

## Phase 12 is settled at checkpoint 300 (2026-09-16)

Per `misc/PHASE13_14_SPEC.md`'s pre-registration discipline, and stated
explicitly here so it cannot be quietly revisited: **checkpoint 300 is
Phase 12's final, pre-registered read.** The run will not be resumed to 600
to get a second look. Resuming later and comparing would turn one
pre-registered read into two reads with the opportunity to pick the more
favorable one -- exactly the failure mode pre-registration exists to
prevent. If Phase 12 is ever resumed toward 600 for other reasons, any
result from that continuation is exploratory and must be labeled as such,
kept separate from the checkpoint-300 number below. Phase 14's launch does
not wait on Phase 12 reaching 600.

## ASTRAL n=32 at checkpoint 300 — RESULT: AMPLIFIED, the project's first positive result (2026-09-17)

`research/astral_model_generations.py --checkpoint
runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300`, completed
2026-09-17 04:28 UTC (`run_logs/astral_phase12_beta0_n32.log`). Analysis:
`research/phase12_astral_final_analysis.py` -> `results/
phase12_astral_final_analysis.json`; raw generations copied to
`results/astral_gen_n32_gdpo_phase12_beta0.json` (previously gitignored
`misc/` only — now tracked, matching the existing `astral_gen_n32_*.json`
convention).

|  | RS-SFT (init) | GDPO-phase12-beta0 (ckpt 300) |
|---|---|---|
| predicted hits | 10/35 | **14/35** |
| conventional hits | 17/35 | 17/35 |
| mean reward (all) | 0.972 | 0.982 |
| validator gap (pred − trad) | +0.007 | +0.007 |
| mean max-T | 934.2 °C | 976.1 °C |
| parse rate | 95.8% | 97.1% |

**VERDICT: AMPLIFIED (14/35 > the pre-registered 12/35 threshold).** RL
moved the policy further toward the ASTRAL-predicted routes than RS-SFT's
own init already was — the project's first positive result, per the
pre-registered read settled at checkpoint 300 (see above; not resumed to
600).

**McNemar on the predicted-hit discordant pairs is not significant, and
this must be stated plainly rather than glossed over.** The two-sided
exact test gives p = 0.125; but since the gain is entirely one-directional
(RS-SFT-only losses: 0; GDPO-phase12-only gains: 4 — `BaLiBO3`,
`KLi(PO3)2`, `Li2TiSiO5`, `NaSrBO3`; both: 10; neither: 21), the correct
one-sided test is the more informative number: **p = 0.0625** (exact
binomial, P(X≤0 | n=4, p=0.5) = 1/16) — the smallest p-value the test can
return at this discordant count, since all 4 favor the same side. Still
short of the conventional 0.05 bar. Report as "consistent with
amplification, not yet statistically decisive," the same posture finding
16 used for GDPO's partial recovery from SFT ("consistent with," never
"recovers").

**Conventional hits held exactly flat (17/17, McNemar p=1)** — unlike the
full-SFT-to-GDPO lineage (17→31, a strong pull back toward convention),
this RL run did not regress toward conventional routes. RS-SFT's
convention-avoidance property (finding 18) survived RL intact.

**Temperature overshoot increased but modestly**: +41.9 °C (934.2→976.1),
well short of the full-SFT-to-GDPO lineage's +53.9 °C jump (963.2→1017.1).
The temperature hack reasserts somewhat under RL even from a good init, but
far less aggressively than from a corpus-converged one.

**This is also the Phase 13/14 gate answer**: Phase 13's comparator reward
(`misc/PHASE13_14_SPEC.md`) has a genuine positive result to try to build
on — the RS-SFT retention result (finding 18) does not stand alone as the
project's only positive finding.

## Footnote on RS-SFT's provenance, added 2026-09-19 (Phase 13's ammonium-precursor bug bears on Phase 12's init)

Phase 13's investigation (`misc/PHASE13_RESULTS.md`) found that
`validator.py`'s balance-solver has a gap causing any ammonium-precursor
route (e.g. containing `NH4H2PO4`) to silently fail `stoichiometry`/
`amount_accuracy` for a software reason, not a chemistry one — confirmed
across the project's largest early training dumps. A direct test
(`research/test_rssft_ammonium_propagation.py`) found ammonium-precursor
prevalence dropped from 12.14% in unfiltered base-model generations to
2.03% in RS-SFT's own bar-0.9 survivor set — a ~6x reduction, directionally
consistent with the bug shaping which routes could even pass RS-SFT's
filter (caveat: a proxy comparison across different target universes, not
a matched pre/post-filter test — see the Phase 13 doc for the full
caveat).

**This does not undo Phase 12's measured result.** The routes RS-SFT
actually retained are real, and GDPO's amplification of them (10/35 →
14/35 predicted hits) is a real, measured effect on that real data. What
this footnote adds: RS-SFT's "convention-avoidance" property (finding 18)
may be partly an artifact of which routes could pass a broken gate to be
counted at all, not purely a chemistry preference the filtering correctly
identified.

### The decisive check (2026-09-19): the positive result does NOT concentrate on the bug-affected targets

Regular Claude's sharper version of this concern: 18/35 ASTRAL targets
have `NH4H2PO4` in their traditional route, and the balance bug zeroed
those — so RS-SFT's rejection filter may have systematically deleted the
dominant *conventional* route for exactly those targets, an "accidental
ablation." If that mechanism drove the 10→14 predicted-hit gain, the gain
should concentrate on those 18 targets.

**It does not — `research/ammonium_ablation_check.py`, run directly:**

| model | ammonium-affected (n=18) | not affected (n=17) |
|---|---|---|
| base | 2/18 (11.1%) | 8/17 (47.1%) |
| RS-SFT | 1/18 (5.6%) | 9/17 (52.9%) |
| GDPO-phase12-β0 (ckpt 300) | 2/18 (11.1%) | **12/17 (70.6%)** |

**3 of the 4 targets GDPO gained over RS-SFT (`BaLiBO3`, `Li2TiSiO5`,
`NaSrBO3`) are in the NOT-affected group; only 1 (`KLi(PO3)2`) is
ammonium-affected.** The entire base→RS-SFT→GDPO improvement trajectory
(47.1%→52.9%→70.6%) sits in the group where the bug is irrelevant; the
ammonium-affected group is flat and low throughout (11.1%→5.6%→11.1%,
never exceeding base). This is the opposite of what the "accidental
ablation" mechanism predicts.

**Conclusion: the alternative explanation is checked and cleared.** Phase
12's positive result is not an artifact of the balance-solver bug's
effect on RS-SFT's training data — it is concentrated exactly where that
bug has no purchase. The RS-SFT provenance caveat above (ammonium
prevalence 12.14%→2.03%) stands as a real, disclosed limitation on what
RS-SFT's filter selected for in general, but it does not explain Phase
12's measured amplification.
