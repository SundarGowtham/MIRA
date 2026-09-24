# Phase 15 Task 4b — ARROWS³ verifier gate: pre-registration

*Written and committed before any outcome (target weight fraction) data
is joined against the verifier's predictions. Step 1 (gradeability
dry-run, `docs/phases/PHASE15_ARROWS_INVENTORY.md` §"Step 1" /
`results/external/arrows_gradeability_dryrun.json`) found no stop-rule
trigger: all three canonical targets resolve in the phase diagram, and
YBCO's worst Phase-12 reward channel (`thermodynamic_favorable`) is
33.5% ungradeable, under the 50% threshold. One design iteration. This
file is not edited after scoring begins — a correction found on review
gets a dated addendum, not a silent edit.*

## Outcome definition

**Outcome**: target-phase weight fraction per `(precursor set,
temperature)` entry, read directly from ARROWS's own `product weight
fractions` field (0% if the target phase does not appear in that entry's
`products` list at all).

**Target-match rule**, verified against the actual product-label
vocabulary in each `Exp.json` before locking this (not assumed):

- **YBCO**: any product whose reduced formula, after stripping ARROWS's
  trailing `_<id>` suffix, satisfies Ba:Y:Cu = 2:1:3 (±5% tolerance) and
  O per formula unit (Y=1 normalized) in `[6, 7]`. **Verified matches in
  the real data: `Ba2YCu3O7` and `Ba2Y(CuO2)3`** (the latter reduces to
  Ba2YCu3O6, the oxygen-deficient end-member — same cation stoichiometry,
  counted). Non-matches confirmed excluded by this rule despite
  superficial similarity: `BaYCu3O7` (Ba:Y:Cu = 1:1:3, wrong ratio),
  `Ba4Y(CuO3)3` (4:1:3, wrong ratio), `Ba2(CuO2)3` (no Y at all).
- **LTOPO**: reduced formula equals `LiTiOPO4`'s own reduced formula
  (`LiTiPO5`). Verified present in the data as `LiTiPO5`.
- **NTMO**: reduced formula equals `Na2Te3Mo3O16`. Verified present
  verbatim in the data.

## Pairs

Within-target, same-temperature, distinct precursor sets. **Primary
threshold: `|Δ target wt%| ≥ 5`.** Sensitivity arms: `≥ 0` (every
differing pair, including sub-5-point differences) and `≥ 10` (stricter),
both reported for every endpoint below, not just the primary.

## Validator primary rule — "GDPO vote"

For each pair `(A, B)`: for each of the five Phase-12 reward channels
(`RUN3_CHECKS` — `amount_accuracy`, `thermodynamic_favorable`,
`stoichiometry`, `chempot_atmosphere`, `operation_order`), compute
`sign(score_A − score_B)`; **ungradeable on either side counts as vote
0** for that channel (not excluded from the denominator — a 5-channel
vote always sums 5 terms, some possibly 0). Sum the 5 votes. **Positive
sum → prefer A. Negative → prefer B. Zero → tie (scored 0.5).**

**Gate failure**: if `validator.validate()` raises or returns no
breakdown for exactly one side, that side loses unconditionally — **prefer
the other side, regardless of channel votes.** If both sides fail,
**the pair is excluded from this endpoint entirely** (locked here, not
decided after seeing counts — there is no principled way to prefer
either side when neither can be evaluated at all). The count of pairs
decided by gate failure alone, and the count excluded for double gate
failure, are both reported.

**Secondary rule**: raw `nansum` of the five raw channel scores per side
(ungradeable = NaN, excluded from that side's sum, not zero) — compare
totals directly, no per-channel voting. Same tie/gate-failure handling.

## Comparator rule

`core/comparator.py` iteration-2 (`misc/comparator_scales_v2.json`
rank-transform scales, arm B / `c3_fraction=0.17`, the pre-registered
Phase 13 read) — margin sign from `Comparator.compare()`. Positive margin
→ prefer A, negative → prefer B, **zero → tie (0.5)**. Gate failure
handled identically to the validator rule (comparator gates are the same
five: `format_ok`, `balances`, `precursors_exist`, `charge_neutral`,
`temperature_physical`).

## Baselines (same pairs, same tie convention)

- **Chance**: every pair scored 0.5 — the reference point a real verifier
  must beat, not just an assumed 50%.
- **Fewer-precursors**: prefers whichever side has fewer declared
  precursors; equal count → tie (0.5).
- **Carbonate-free**: prefers whichever side has zero carbonate
  precursors (`Li2CO3, Na2CO3, K2CO3, BaCO3, SrCO3, CaCO3, MgCO3`); both
  carbonate-free or both carbonate-containing → tie (0.5).

Phase 13 already showed a constant-preference rule can look deceptively
good against an uncorrected baseline — these three, scored on the
identical pairs, are the correction.

## Primary endpoint (YBCO)

**`agreement(validator GDPO vote) − agreement(best baseline)`**, where
`agreement` = mean over pairs of `1` (verifier/baseline prefers the
higher-wt% side), `0` (prefers the lower side), `0.5` (either side tied
at the primary `|Δ|≥5` threshold, ties are structurally rare but possible
via the vote-sum-zero case). "Best baseline" = whichever of the three
baselines scores highest agreement on the same pairs (chosen after
seeing baseline agreement, not before — the verifier is compared against
the best available naive rule, not a strawman).

**95% CI on the difference by cluster bootstrap, resampling PRECURSOR
SETS (not pairs — pairs sharing a precursor set are not independent),
10,000 resamples, fixed seed `20260924`.** Procedure: sample the set of
distinct precursor sets with replacement to the original count; keep
every pair whose both members were drawn (a pair may appear 0, 1, or
multiple times depending on how many times each of its two sets was
drawn); recompute agreement on this resampled pair multiset; repeat
10,000 times; take the 2.5/97.5 percentiles of the resampled
`agreement(validator) − agreement(best baseline)` distribution.

**Pass rule**: the CI lower bound is above 0. **Identical procedure for
the comparator** (its own agreement minus the same best-baseline
agreement, its own bootstrap).

**LTOPO and NTMO**: reported the same way (same endpoint, same bootstrap
procedure) as **secondary replications** — not gating, given their much
smaller pair counts (392 and 88 total pairs respectively, from Task 4a;
after the primary `|Δ|≥5` filter and gate-failure exclusions, fewer
still).

## Always reported, regardless of the primary result

- Tie rate per verifier (validator GDPO-vote, validator nansum,
  comparator) and per baseline.
- Per-channel agreement and tie rate for each of the five `RUN3_CHECKS`
  channels individually (not just the aggregate vote) — **especially
  `thermodynamic_favorable`**, given its role in the causal chain
  (`docs/phases/PHASE15_DISTRIBUTIONAL.md`).
- Per-temperature breakdown (600/700/800/900/1000°C for YBCO; 400-700°C
  for LTOPO; 300-400°C for NTMO).
- All three `|Δ|` thresholds (0, 5, 10) side by side.
- How many pairs are decided by gate failure alone (both the validator
  rule and the comparator rule).

## Ba-source controlled test — verifier's ordering LOCKED here, before any outcome is joined

*Computed by `research/distributional/arrows_ba_source_preference.py`,
run and committed alongside this file, in the same commit as this
pre-registration — the commit timestamp is the proof the ordering below
came before any wt% lookup for these specific cells.*

**Verifier's claim** (what "preferred" means, stated before checking):
the verifier's preferred Ba source gives higher YBCO target wt% than the
less-preferred one(s), in the matched group it was computed from.

**Our prediction, stated before checking**: the verifier's claim does
NOT hold — the matched wt% difference (preferred minus non-preferred) is
≤ 0, or its bootstrap CI includes 0.

**Result, locked, counted precisely (not eyeballed)**: across all 10
matched groups × 4 shared temperatures = **40 `(group, temperature)`
cells**, the **comparator ranks BaCO3 last in 40/40 (100%)**. The
**GDPO-vote ranks BaCO3 last in 39/40 (97.5%) — one exception**:
`CuCO3, Y2O3` at 900°C, where the GDPO-vote ordering is `['BaO', 'BaCO3',
'BaO2']` (BaCO3 ranks 2nd, ahead of BaO2, not last). Stated exactly as
found; not rounded up to "zero exceptions." Full table (GDPO-vote
ordering; comparator ordering in parentheses when it differs):

| co-precursors | Ba-free co-precursors? | ordering (best→worst), all shared temperatures |
|---|---|---|
| CuCO3, Y2(CO3)3 | yes | BaO2, BaO, BaCO3 (all 4 temps, both methods agree) |
| CuO, Y2(CO3)3 | yes | BaO2, BaO, BaCO3 (all 4 temps, both methods agree) |
| CuCO3, Y2Cu2O5 | yes | BaO2, BaO, BaCO3 (all 4 temps, both methods agree) |
| CuO, Y2Cu2O5 | yes | GDPO: BaO2, BaO, BaCO3 — comparator: BaO, BaO2, BaCO3 (all 4 temps; methods disagree on 1st/2nd, agree BaCO3 last) |
| Ba2(CuO2)3, Cu2O, Y2O3 | no | BaO, BaCO3 (2-way, all 4 temps, both agree) |
| Ba2(CuO2)3, Cu2O, Y2Cu2O5 | no | BaO, BaCO3 (2-way, all 4 temps, both agree) |
| Ba2(CuO2)3, Cu2O, Y2(CO3)3 | no | BaO, BaCO3 (2-way, all 4 temps, both agree) |
| Ba2(CuO2)3, Y2Cu2O5 | no | GDPO: BaO,BaO2,BaCO3 at 600-700°C, swaps to BaO2,BaO,BaCO3 at 800-900°C — comparator: BaO2,BaO,BaCO3 at all 4 temps. BaCO3 last in every cell, both methods. |
| CuO, Y2O3 | yes | BaO, BaO2, BaCO3 (all 4 temps, both methods agree) |
| CuCO3, Y2O3 | yes | GDPO: BaO,BaO2,BaCO3 at 600-800°C, **BaO,BaCO3,BaO2 at 900°C (the one exception — BaCO3 2nd, not last)** — comparator: BaO2,BaO,BaCO3 at all 4 temps (BaCO3 last throughout). |

Full per-cell pairwise detail (every `A_vs_B` comparison, every
temperature): `results/external/arrows_ba_source_verifier_preference.json`.

**Test**: per `(group, temperature)` cell, sign of the yield difference
(preferred source's wt% minus the least-preferred source's wt%, using
the verifier's own top-ranked vs. bottom-ranked choice from the table
above). **95% CI by bootstrap resampling over GROUPS** (not cells — cells
within a group share a co-precursor pair and are not independent), 10,000
resamples, fixed seed `20260924`.

**Report the full group × temperature yield table** (preferred source's
wt%, least-preferred source's wt%, difference, sign) once outcomes are
joined in Step 3.

**Power statement, locked**: with **10 groups** (6 of them the "clean"
Ba-free-co-precursor case), this is a **directional check with limited
power** — a single bootstrap CI over 10 clusters will be wide. Reported
as a directional finding, not treated as a decisive statistical test on
its own; it stands alongside the pooled YBCO primary endpoint (4,390
pairs), not in place of it.

## Rules restated

- One design iteration. This file is not edited after Task 4c's scoring
  begins — a correction found on review gets a dated addendum.
- The target-match rule, pair thresholds, GDPO-vote mechanics, gate-
  failure handling, baseline definitions, and bootstrap procedure are all
  locked above and are not adjusted after seeing agreement numbers.
- The Ba-source ordering above is locked BEFORE any outcome/wt% lookup
  for those specific cells — Step 3 joins outcomes against this table,
  never recomputes the ordering after seeing them.
- `validator.py` and `core/ranker.py` are not modified. `core/
  comparator.py` is used as-is (iteration-2, already committed).
