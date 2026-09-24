# Phase 15 Task 4c — ARROWS³ verifier gate: results

*Pre-registration: `docs/phases/PHASE15_ARROWS_PREREG.md`, committed
before any outcome data was joined. Script:
`research/distributional/arrows_gate_scoring.py`. Output:
`results/external/arrows_gate.json`. Pre-registered results tagged
**[C]**; everything else **[E]**. One iteration — no changes after
seeing these numbers.*

## Primary endpoint (YBCO) — [C] **NEITHER verifier passes the gate**

| | agreement (primary, \|Δ\|≥5, n=1915) | diff vs. best baseline | 95% CI | pass (CI lower > 0)? |
|---|---|---|---|---|
| **validator GDPO-vote** | 0.616 | +0.033 | [−0.043, +0.112] | **NO** |
| **comparator** | 0.474 | −0.109 | [−0.226, +0.007] | **NO** |
| best baseline | carbonate-free, 0.583 | — | — | — |

**Neither the validator's GDPO-vote nor the comparator's margin
significantly outperforms the best of three trivial baselines** on
4,870 real robot-lab pairs (1,915 at the primary threshold). The
validator's CI straddles zero (can't distinguish from
"always prefer the carbonate-free route"); the **comparator scores
BELOW chance itself** (0.474 < 0.500) and is 0.109 behind the best
baseline, with a CI that barely misses excluding a positive difference
in the *other* direction.

**Why the comparator does this badly — diagnosed, not just reported
[E]**: of YBCO's 4,390 total pairs, **2,115 are decided by one-sided
gate failure alone, and 615 more have both sides failing (excluded
entirely)** — only 1,660 pairs (38%) reach an actual channel-based
comparator decision. The comparator's five gates (`balances`,
`precursors_exist` especially — Step 1's dry-run found these fail on
10.5% and 33.5% of individual YBCO routes respectively) are validity
checks, not quality signals; on this dataset they end up doing most of
the "work," and validity-passing is not the same question as
"which route makes more of the target phase." The comparator's poor
showing here is substantially a consequence of gates dominating the
decision, not the six scored channels failing on their own terms.

## Per-channel agreement (YBCO, all pairs, any Δ) — [E]

| channel | agree | disagree | tie | agreement rate |
|---|---|---|---|---|
| **`thermodynamic_favorable`** | 406 | 165 | 1,325 | **71.1%** |
| `amount_accuracy` | 516 | 236 | 2,747 | 68.6% |
| `stoichiometry` | 225 | 196 | 3,969 | 53.4% (near chance) |
| `chempot_atmosphere` | 0 | 0 | 4,390 | dead (100% tied) |
| `operation_order` | 0 | 0 | 4,390 | dead (100% tied) |

**The one channel with a real, causally-established mechanism
(`thermodynamic_favorable` — Task 2c) is also the best-agreeing channel
in isolation (71.1%).** But the 5-channel GDPO-vote sum (which zeros out
any channel that's ungradeable or tied) only reaches 61.6% overall
agreement and still fails to beat the baseline — `stoichiometry`'s
near-chance signal and the two structurally dead channels dilute rather
than reinforce the one channel that works. This is consistent with, and
extends, the standing finding that per-channel signal does not
straightforwardly compose into aggregate signal in this reward design.

## Secondary replications (LTOPO, NTMO) — [C]

| target | validator diff | 95% CI | pass? | comparator diff | 95% CI | pass? |
|---|---|---|---|---|---|---|
| LTOPO | −0.084 | [−0.291, +0.171] | NO | −0.019 | [−0.315, +0.254] | NO |
| NTMO | −0.007 | [−0.222, +0.197] | NO | −0.045 | [−0.308, +0.212] | NO |

Both targets' best baseline is chance itself (0.5) — neither
fewer-precursors nor carbonate-free baseline beats chance here, and
neither verifier beats chance either. Wide CIs reflect the much smaller
pair counts (77–88 pairs vs. YBCO's 1,915) — genuinely underpowered
replications, consistent with a null but not a strong one on their own.

## Ba-source controlled test — [C] verifier's claim NOT supported

**Mean yield difference (verifier's preferred Ba source minus
least-preferred): +5.31 wt%, 95% CI (bootstrap over 10 groups) =
[−3.36, +14.60].** CI includes zero — **the verifier's claim (its
preferred source gives higher measured YBCO wt%) is not supported**,
exactly the pre-registered prediction.

**Sign test is the more informative number here**: of 40 `(group,
temperature)` cells, 22 have **both** options at 0% target phase
(too-low temperature, uninformative — neither source works at all) and
are excluded from the sign count. **Of the 18 informative cells: 9
positive (support the verifier's preference), 9 negative (contradict
it) — an exact 50/50 split.** The pooled mean being nominally positive
(+5.31) is driven by two large-magnitude cells (`CuCO3,Y2Cu2O5` and
`CuO,Y2Cu2O5` at 900°C, both +96.5 points) that happen to favor the
verifier's pick, not by a consistent directional effect — the sign test
shows no directional consistency at all once those outliers are counted
equally with everything else.

**Power caveat restated, as pre-registered**: 10 groups is genuinely
few; this result is directional, not decisive, on its own — but it is
fully consistent with, and reinforces, the primary endpoint's null.

## All three thresholds, side by side — [E]

| target | threshold | n pairs | GDPO-vote agreement | comparator agreement | best baseline |
|---|---|---|---|---|---|
| YBCO | ≥0 | 4,390 | 0.401 | 0.234 | carbonate-free 0.434 |
| YBCO | ≥5 (primary) | 1,915 | 0.616 | 0.474 | carbonate-free 0.583 |
| YBCO | ≥10 | 1,671 | 0.630 | 0.475 | carbonate-free 0.590 |
| LTOPO | ≥0 | 392 | 0.291 | 0.115 | chance 0.500 |
| LTOPO | ≥5 | 77 | 0.416 | 0.481 | chance 0.500 |
| LTOPO | ≥10 | 58 | 0.414 | 0.500 | chance 0.500 |
| NTMO | ≥0 | 88 | 0.443 | 0.369 | chance 0.500 |
| NTMO | ≥5 | 67 | 0.493 | 0.455 | chance 0.500 |
| NTMO | ≥10 | 53 | 0.425 | 0.387 | chance 0.500 |

The pattern is stable across thresholds: GDPO-vote is consistently
somewhat below or near the best baseline; the comparator is consistently
at or below chance on every target, every threshold.

## Tie rates — [E]

At the primary YBCO threshold: GDPO-vote ties on a small fraction of
decided pairs (most pairs resolve to a nonzero vote sum); the comparator
similarly rarely ties once gate-decided pairs are set aside. Full
per-threshold tie rates in `results/external/arrows_gate.json`.

## Gate-decided pairs — [E]

**YBCO**: GDPO-vote rule has **0** pairs decided by gate failure (the
validator's own `validate()` almost never raises/returns nothing for
these routes — it degrades gracefully to `0.5`/ungradeable internally
rather than failing outright). The **comparator rule has 2,115** pairs
decided by one-sided gate failure and **615** excluded for both-sided
failure — see the diagnosis above. LTOPO and NTMO: 0 gate-decided pairs
for either rule (smaller, evidently cleaner precursor sets from a
gate-validity standpoint).

## Summary

**A fourth independent verifier evaluation — the most rigorously
controlled one yet (real robot-lab data, matched precursor sets, three
explicit baselines, proper cluster-bootstrap CIs) — lands at or below
chance.** This extends the pattern already established by the validator
(50.0%, Kononova/ASTRAL), ranker v2 (60.0%, ASTRAL), and the comparator
(47.1%→57.1% across iterations, ASTRAL): **no verifier design tried in
this project has been shown to beat a trivial baseline on real,
externally-measured synthesis outcomes**, and on this dataset
specifically, the comparator does measurably *worse* than the simplest
possible heuristics. The one channel with a real, causally-verified
mechanism (`thermodynamic_favorable`) is also the best individual
performer (71.1% agreement) — but aggregating it with weaker and dead
channels erases the advantage at the whole-verifier level. The Ba-source
controlled test reinforces this with a genuine robot-lab experiment: the
verifier's confident, near-unanimous ordering (BaCO3 last in 39-40 of 40
cells) does not translate into a measurable real-yield advantage.
