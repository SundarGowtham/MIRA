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

**Corrected framing (2026-09-25)**: inconclusive — 9/9 sign split; the
+5.31 wt% mean is driven by two cells. The pre-registered prediction is
satisfied only in a weak sense — with 10 groups, a CI including zero is
expected under most effect sizes, so this result cannot distinguish "no
effect" from "a real effect this test is too small to see." **Power
caveat restated, as pre-registered**: 10 groups is genuinely few; this
result is directional, not decisive, on its own.

## All three thresholds, side by side — [E], lowest threshold REBUILT 2026-09-25

**Correction**: the original "≥0" row included pairs with *identical*
target wt% (overwhelmingly both-zero — too-low-temperature entries where
neither route produced any target phase), which have no correct answer
to agree or disagree with. Rebuilt as **strict `|Δ| > 0`**, excluding
those pairs entirely (`research/distributional/arrows_task4d_diagnostics.py`,
§1). Exclusion counts, reported per target as instructed:

| target | total pairs | excluded (identical outcome) | remaining (\|Δ\|>0) |
|---|---|---|---|
| YBCO | 4,390 | 2,266 (51.6%) | 2,124 |
| LTOPO | 392 | 302 (77.0%) | 90 |
| NTMO | 88 | 16 (18.2%) | 72 |

Over half of YBCO's and more than three-quarters of LTOPO's original
"pairs" carried no information at all.

| target | threshold | n pairs | GDPO-vote agreement | comparator agreement | best baseline |
|---|---|---|---|---|---|
| YBCO | **\|Δ\|>0 (rebuilt)** | 2,124 | 0.614 | 0.485 | carbonate-free 0.577 |
| YBCO | ≥5 (primary) | 1,915 | 0.616 | 0.474 | carbonate-free 0.583 |
| YBCO | ≥10 | 1,671 | 0.630 | 0.475 | carbonate-free 0.590 |
| LTOPO | **\|Δ\|>0 (rebuilt)** | 90 | 0.533 | 0.500 | chance 0.500 |
| LTOPO | ≥5 | 77 | 0.416 | 0.481 | chance 0.500 |
| LTOPO | ≥10 | 58 | 0.414 | 0.500 | chance 0.500 |
| NTMO | **\|Δ\|>0 (rebuilt)** | 72 | 0.493 | 0.451 | carbonate-free 0.438 |
| NTMO | ≥5 | 67 | 0.493 | 0.455 | chance 0.500 |
| NTMO | ≥10 | 53 | 0.425 | 0.387 | chance 0.500 |

**"Stable across thresholds" does not hold uniformly — kept only where
the rebuilt row actually supports it.** YBCO is genuinely stable once
the flawed threshold is fixed (0.614/0.616/0.630 GDPO-vote,
0.485/0.474/0.475 comparator — the instability in the original table was
an artifact of the uninformative-pairs bug, not a real threshold
sensitivity). **LTOPO is NOT stable**: GDPO-vote moves from 0.533
(rebuilt) to 0.416 (≥5), a 12-point swing on only 90→77 pairs — too
noisy at this n to call a pattern either way. NTMO is roughly flat but
on only 72–88 pairs.

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

---

# Task 4d diagnostics [E] (2026-09-25), appended after review

*Script: `research/distributional/arrows_task4d_diagnostics.py`, output:
`results/external/arrows_task4d_diagnostics.json`. Reuses
`arrows_gate_scoring.py`'s route construction and rules directly
(imported, not reimplemented) so every number here is comparable to the
pre-registered Task 4c figures above.*

## Thermo channel, same-pairs analysis (YBCO, primary threshold ≥5)

**(a) Thermo-only rule vs. best baseline, all 1,915 primary pairs**
(ties/ungradeable = 0.5): thermo-only agreement = **0.558**, best
baseline (carbonate-free) = 0.583, diff = **−0.025, 95% CI
[−0.078, +0.027]**. Even the single best channel, used alone across
every pair (most of which it can't actually decide), does not beat the
carbonate-free baseline either.

**(b) On thermo's decisive pairs only** (n=521, the 27% of primary pairs
where `thermodynamic_favorable` is actually gradeable on both sides and
not tied): thermo agreement = **0.712**, carbonate-free agreement on the
*same* 521 pairs = 0.662, fewer-precursors on the same pairs = 0.515.
Thermo does outperform carbonate-free by ~5 points where it has an
opinion. **But of the 285 pairs where both thermo and carbonate-free are
simultaneously decisive, thermo's pick equals carbonate-free's pick
87.7% of the time** — the two are heavily redundant, not independent
signals; thermo's edge lives almost entirely in the 12% where they
disagree.

**(c) Key test — thermo agreement restricted to pairs where
carbonate-free TIES** (both routes have the same carbonate status,
n=1,112 — this isolates whatever `thermodynamic_favorable` knows *beyond*
the carbonate/bare-oxide dimension): **agreement = 0.522, 95% CI
[0.495, 0.557]**. **The CI is centered almost exactly on chance and does
not clear it.** Once the carbonate dimension is held constant,
`thermodynamic_favorable` shows no established signal of its own — its
apparent quality substantially *is* the carbonate-vs-bare-oxide
distinction (Task 2c's own mechanism), not a separate, additional
capability.

## Gate-failure audit

**`precursors_exist` failures (16 YBCO precursor sets, 0 for LTOPO/NTMO):
100% attributable to a single precursor, `Y2(CO3)3`** — confirmed absent
from Materials Project entirely by direct `mp_formula_set.pkl`
membership check. Every other one of the 30 distinct precursor formulas
appearing across all three targets — including parenthetical forms
(`Ba2(CuO2)3`, `Ba2Y(CuO2)3`), peroxides (`BaO2`, `Na2O2`), and complex
ammonium/phosphate-style formulas (`PH9(NO2)2`, `MoH8(NO2)2`) — **is
present in MP**. Category (a) — absent from MP — fully explains this
failure mode.

**`balances` failures (5 YBCO sets)**: 3 of 5 also contain `Y2(CO3)3`
(already broken). **The other 2 — `(BaCuO2, BaO2, Cu2O, Y2O3)` and
`(BaO, BaO2, Cu2O, Y2O3)` — do not**, and every formula in them parses
cleanly and is in MP. This is category **(c): genuinely unbalanceable**
with the validator's finite candidate-volatile-set search for these
specific 4-precursor combinations (plausibly an overdetermined system —
two separate Ba sources, `Cu2O`, and `Y2O3` simultaneously).

**Category (b) — notation/parsing failure — is empty. Zero instances.**
All 30 distinct precursor formulas, across every target, parse without
error via pymatgen `Composition()`. **(b) does not dominate; it does not
occur at all** — stated plainly, as instructed, in the negative.

**16 of YBCO's 47 precursor sets (34%) contain `Y2(CO3)3`** — this is not
a marginal edge case; over a third of the dataset's precursor space is
invisibly excluded from `precursors_exist`-gated scoring.

**Does `Y2(CO3)3` actually perform worse in the real experiment? No —
if anything, slightly better**, at the temperatures where the reaction
proceeds at all (matched by temperature, not pooled, since pooling would
conflate the huge temperature effect with any precursor-choice effect):

| temperature | `Y2(CO3)3` mean yield (n) | other-Y-source mean yield (n) |
|---|---|---|
| 600–700°C | 0.0% (both) | 0.0% (both) — uninformative, neither works yet |
| 800°C | **26.2%** (n=16) | 24.7% (n=31) |
| 900°C | **66.2%** (n=16) | 62.1% (n=31) |
| 1000°C | 22.8% (n=3) | 42.6% (n=9) — small n, reversed |

**Recorded finding: `precursors_exist` tests Materials Project coverage,
not chemical existence.** `Y2(CO3)3` is a real compound used in a real,
published robotic synthesis — it works about as well as, or slightly
better than, the alternative Y sources at the temperatures where YBCO
actually forms. Its absence from MP's database (a DFT-computed materials
database, not an exhaustive registry of known chemistry) is a coverage
gap in the reference data, not a signal about whether the precursor is
sound. Every route using it is currently invisible to `precursors_exist`-
gated scoring for a database-completeness reason unrelated to its real
performance.

## Is "avoid carbonates" itself a real signal, or just a trivial baseline?

**Real signal, confirmed**: the carbonate-free baseline's own agreement
(0.583) is significantly better than chance — **diff from 0.5 = +0.083,
95% CI (cluster bootstrap over precursor sets) [+0.010, +0.154], lower
bound above zero.** "Avoid carbonates" is not merely a convenient
strawman baseline that happens to look good by construction; on this
robot-lab dataset it is itself a real, if modest, predictor of which
route yields more target phase. Neither verifier managed to add anything
measurable on top of this already-real signal — which sharpens, rather
than undermines, the negative result: the validator and comparator are
not failing to find a real effect that isn't there; they are failing to
improve on a real effect a two-line heuristic already captures.

**Comparator agreement restricted to the channel-decided-only pairs**
(both gates pass, YBCO, primary threshold, n=704 — excluding the
gate-decided/excluded majority): **0.513** — barely above chance, and
still far short of the 0.583 baseline. **Directly computed (not
inferred), the one-sided gate-decided subset (n=943) agrees only 0.445
of the time** — worse than chance itself, and worse than the pooled
figure (0.474) and the channel-decided figure (0.513) alike. Since
channel-decided (0.513) is better than pooled (0.474), gate-decided
necessarily had to be worse than both — now confirmed directly rather
than left as an inference: **the comparator's gates are actively hurting
its agreement on this dataset, not merely diluting an otherwise-good
channel signal.**

## GDPO-vote rule: gradeability-tag verification

**Confirmed by direct code inspection, not assumed**: the GDPO-vote rule
never uses a raw sentinel value (validator.py returns `0.5` as a
placeholder score for several ungradeable checks, e.g.
`_check_thermodynamics`'s `return 0.5, gradeability` when `delta_G is
None`) as if it were a real score. `research/distributional/
arrows_gate_scoring.py:174-177`, inside `gdpo_vote_and_channels`, checks
each channel's own `{channel}_gradeability` tag against
`SynthesisValidator.SENTINEL_TAGS` **before** reading the raw value into
the vote computation — if either side is ungradeable, the function
records `("ungradeable", 0)` and `continue`s, never subtracting the two
sides' raw (possibly-sentinel) scores. `nansum_rule` (lines 198-199) does
the same check before adding either value into its running sum. Two of
the five `RUN3_CHECKS` channels (`stoichiometry`, `operation_order`) have
no gradeability sibling key at all in `validator.py` — confirmed by
grep — because they are always directly computed with no ungradeable
state in this codebase's design, not an oversight in this script.

---

# Summary — corrected wording (2026-09-25)

**No verifier beat the best trivial baseline. The validator's GDPO-vote
(0.616) is above chance but indistinguishable from "prefer the
carbonate-free route" (0.583, 95% CI on the difference [−0.043,
+0.112]). The comparator (0.474) is below chance and below every
baseline.** This is the most rigorously controlled evaluation in this
project's history (real robot-lab data, matched precursor sets, three
explicit baselines, proper cluster-bootstrap CIs), and it does not show
either verifier clearing the bar a real signal would need to clear.

**Project-wide, the same corrected framing applies**: the validator here
(0.616) sits close to ranker v2's 60.0% (ASTRAL) and above the plain
validator's own 50.0% (ASTRAL) and the comparator's 47.1–57.1% (ASTRAL,
across iterations) — **but in every one of these four evaluations, the
number that matters is not the raw agreement rate, it's whether the
verifier beats a trivial, baseline-matched alternative, and in none of
them has it done so with a CI that excludes zero.** The Task 4d diagnosis
sharpens *why*, for the one channel with a real, causally-verified
mechanism: `thermodynamic_favorable` beats carbonate-free by a real
margin when it has an opinion (0.712 vs 0.662 on its own decisive pairs)
but is 88% redundant with that same baseline in its predictions, and
shows no established signal beyond the carbonate dimension once that
dimension is held constant (0.522, CI centered on chance). The
comparator's specific weakness here is diagnosed concretely: its gates,
not its scored channels, decide most pairs on this dataset, and they
decide badly (channel-decided agreement 0.513 vs. gate-decided 0.445,
directly computed — the gates make things worse, not just noisier). And
the bar itself is real: the carbonate-free baseline alone is
significantly better than chance (+0.083, CI [+0.010, +0.154]) — neither
verifier improves on an effect a two-line heuristic already captures.
The Ba-source controlled test adds a genuine robot-lab experiment
pointing the same direction, though inconclusively at n=10 groups (see
the corrected framing above).

---

**Task 6 (Precursor Genome check) is cancelled for this paper** — noted
as future work, not attempted. No data was downloaded and no analysis
was started.
