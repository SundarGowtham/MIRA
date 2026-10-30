# Proposed CLAUDE.md updates for Phase 17's close

**This file is NOT an instruction to any future reader of CLAUDE.md — it is a
patch proposal.** Per this session's standing memory rule, CLAUDE.md in this
project is edited only by the "regular Claude" chat session the user runs
separately; this Claude Code session reads CLAUDE.md but never writes to it,
even when explicitly asked to in a relayed instruction. The pasted instruction
this responds to ("Update CLAUDE.md...") is implemented here as a diff for
that other session (or the user) to apply by hand, not as a direct edit.

Three changes, in CLAUDE.md's own section order:

---

## 1. New finding, inserted after finding 33 (before "## Journey")

```markdown
34. **Phase 17's reward-design audit closes with two real, narrow survivors
    and one withdrawn headline (2026-10-28).** Phase 17 ran the pre-registered
    reward audit (symmetry, Price equation, information bound, best-of-n,
    Hodge decomposition, reachability) and a mechanistic regression of each
    training step's log-probability shift against the model's own pre-step
    log-probability ("signature regression", the shape of fine-tuning vs
    practice). An early read of three results together — B1's confidently
    negative practice-step coefficient (c_s<0 for S5/S6/S7), R-h's reachability
    back-test (3 of Phase 12's 4 gained ASTRAL targets sat far below an
    unreachability threshold under RS-SFT's estimated probability), and B3's
    leash-ceiling overshoot (6-7x) — would have read as "on-policy practice
    reaches recipes a pure tilt cannot." **That headline is withdrawn before
    publication.** A tail-calibration check found the shared K=8-trace
    marginal probability estimator underlying all three results is biased low
    by up to ~1.7 million x in its lowest probability decade (pooled
    observed/expected ratio 7,425x for pi~<1e-4) — exactly the range the
    headline depended on. An errors-in-variables split-trace control (fitting
    the same regression from two independent trace halves, so the regressor
    and outcome no longer share a common noise source) confirmed this directly
    for B1: the corrected c_split is statistically indistinguishable from 0
    for S5 and S7 (the "confidently negative for all three practice steps"
    reading does not survive decorrelation), leaving only a small, real
    residual at S6 (c_split=-0.039, 95% CI [-0.075,-0.002]) — real, but far
    narrower than the original claim, and failing the full pre-registered
    joint criterion (b_split's CI excludes 0 only at S6, [0.72,8.02]; its own
    c_split CI is then entirely <0, failing that criterion's other half). A
    literal, non-estimator recount of R-h (counting each reward's top-decile
    recipes' raw occurrences in the init model's own on-disk samples, no
    probability estimate at all) found the same split: R1 (init corpus_sft)
    still shows a genuine majority-unreachable top decile by a raw count
    (78.7%, clearing the pre-registered >50% bar); R2 and R4 (init rs_sft) do
    **not** survive the correction (43.6% and 30.6%, both now below the bar).
    B3's overshoot was not given its own correction and is carried as "likely
    explained by the same tail bias, not independently confirmed." **What
    survives for the paper, replacing the three-line headline: R1's genuine
    unreachable majority by a raw count, and S6's small but real residual
    negative c_split — two narrower findings, not one broad one.** The reward
    audit's other pre-registered checks returned real, standalone results of
    their own, independent of this correction: the Price-equation sign-check
    passed (6/7 matched at the >=75% bar); Hodge decomposition found the
    learned comparator has zero empirical antisymmetry violations and only
    5.2% curl on ASTRAL, fully attributable to channel-set-heterogeneous
    triangles (confirming the pre-registered mechanism exactly); the
    information-bound check found real (not permutation-null) mutual
    information between every one of the three rewards and the real ARROWS3
    yield outcome even conditioned on carbonate status — a predicted null that
    failed, though a follow-up found no reward is significantly anti-signal in
    any carbonate stratum. Gate G1 (median Spearman between the marginal
    estimator's log-probability and held-out sampled frequency) passed for
    all 12 models checked (range 0.59-0.68, bar >=0.5) — this gate certifies
    rank-correlation in the estimator's bulk, which is a different claim from
    tail calibration, so G1 passing and the tail being badly biased are not in
    tension. Source: `docs/phases/PHASE17_RESULTS.md`, `results/ledger/CLAIMS_LEDGER.md`
    Layer 11.
```

---

## 2. Fragility caveat added to the end of finding 22's paragraph

Current last sentence of finding 22:

> This is also Phase 13's gate answer: it gave Phase 13's comparator reward a
> genuine positive result to try to build on, so finding 18 (RS-SFT retention)
> does not stand alone as the project's only positive result. Source:
> `docs/phases/PHASE12_RESULTS.md`.

Append, before the "Source:" sentence:

> **Fragility caveat added 2026-10-28 (Phase 17, Block A)**: a paired
> bootstrap of Phase12@300's hits minus RS-SFT's hits, using the SAME
> resampled draw positions for both models per target (so common per-target
> sampling noise cancels in the diff rather than inflating the CI), gives
> median diff = **2** hits, 95% CI **[-1, 5]** — includes 0. The 14-vs-10
> point estimate is unchanged and the paired check does not overturn the
> result, but "consistent with amplification, not yet statistically decisive"
> (already this finding's own framing) should be read with this interval
> attached, not as a stronger claim than that.

---

## 3. Phase 17 marked closed

In "## CURRENT WORK: PHASE 17", change the header to "## PHASE 17 (CLOSED,
2026-10-28)" and prepend before the existing bullet list:

```markdown
**STATUS: CLOSED.** Blocks 0, A, B, C, G (G1) and R ran. **Blocks D, E and F
were not run** — nothing in Phase 17's later work depended on them, and no
later result needed them to interpret what did run. **Block C (unbalanced OT)
is infeasible as specified**: only 1.6% of target-steps (6/385) converge at
the pre-registered regularization, because before/after raw mass differs by
10+ orders of magnitude within a single target — a genuine data property, not
a solver bug; the pre-registration's own fallback condition (rerun with a
tail-calibrated renormalization, only if the tail-calibration check shows the
tails ARE calibrated) was correctly never triggered, since they are not
(7,425x pooled bias, see finding 34). Headline result and full narrative:
finding 34 and `docs/phases/PHASE17_RESULTS.md`.
```

Then move the existing `## CURRENT WORK: PHASE 17` bullet list (block order,
gate rule, reuse-before-writing list) down as historical planning record
(same treatment as the `## PHASE 12 — GDPO FROM RS-SFT` section already gets
— "kept as the original planning record... do not treat its forward-looking
language as an open task").

---

## What this patch does NOT include

- No change to finding 27 (ARROWS³ gates), finding 25/26 (precursor shifts) —
  Phase 17 did not revisit those.
- No claim that B3 is corrected — the patch explicitly carries it as
  uncorrected, per the pre-registration brief's own instruction to the paper's
  writer not to report it.
- No mention of a true fresh-seed Phase 12 replication — that was offered as
  optional, compute-costing, and explicitly skipped this round.
