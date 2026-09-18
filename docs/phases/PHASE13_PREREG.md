# Phase 13 pre-registration — locked before `core/comparator.py` exists

*Per `misc/PHASE13_14_SPEC.md`. Written and committed to before any channel is
implemented or scored, per the spec's own rule: "one design iteration... do not
adjust after seeing a number." This file is the contract. If the bar fails, we
report the per-channel table and stop — we do not come back and edit this file
retroactively to explain why.*

---

## Primary endpoint

On the 35 ASTRAL targets, N_pref = number of targets where the comparator
prefers the ASTRAL-*predicted* precursor set over the *conventional* set
(pairwise margin > 0, summed uniformly across gradeable channels C1–C7; C8
excluded — ASTRAL routes were not generated in groups, per the spec).

Chance = 17.5/35.

**Bar: N_pref ≥ 25/35** (one-sided binomial p ≈ 0.008 against chance).

- **≥ 25/35** → Phase 13 clears. Phase 14 becomes a real prospect, contingent
  on Phase 12's own readout (see "Phase 12 note" below).
- **< 25/35** → STOP. Report the per-channel disagreement table and the
  channel-ablation breakdown (which channels, if dropped, would have changed
  the count). Do not adjust channel forms, parameters, or membership after
  seeing this number. A physics-grounded, chemist-legible verifier still
  failing to rank real experimental outcomes is a stronger negative than
  Phase 11's ranker gate failure (21/35), and is reported as such, not
  quietly re-engineered.

## The C3 temperature-window parameter — locked before scoring

Original design used a flat 100 K additive width (`w_low = w_high = 100 K`)
with no source. Investigated during pre-registration (not after seeing a
result): the solid-state-reactivity-onset literature describes the Tammann
transition as a **fractional range of T_melt**, not a fixed Kelvin offset —
commonly stated as reactivity onset over roughly **0.5–0.67 × T_melt**, with
related interfacial-mobility studies showing continued broadening out to
~0.75 × T_melt. A flat Kelvin width doesn't scale correctly between a
600 K-melting and an 1800 K-melting precursor; a fractional width does.

**Citation stance**: cited here as standard solid-state synthesis practice,
not attributed to a specific primary paper — the specific source found during
this investigation (Merkle & Maier, *"On the Tammann Rule,"* Z. Anorg. Allg.
Chem. 2005) was read only via search-engine summaries, not the primary text
directly, by either Claude session that touched this spec. Citing an unread
primary source in a pre-registration is exactly the kind of thing an
external reviewer would catch and is a gratuitous risk for a channel that
works fine on a hedged citation. **Flagged for chemist verification** before
this goes in any writeup that cites Merkle & Maier by name.

**Form change**: `w_low`, `w_high` are now expressed as a **fraction of
T_melt** (of whichever precursor sets that boundary) rather than a flat
Kelvin value.

**Sensitivity sweep — four arms, all reported**:

| arm | width definition |
|---|---|
| A | fraction = 0.12 × T_melt |
| **B (pre-registered read)** | **fraction = 0.17 × T_melt** |
| C | fraction = 0.22 × T_melt |
| D | flat 100 K (the original, pre-fractional design, kept for direct comparison) |

The primary N_pref bar is evaluated under arm B. All four are reported in
`misc/PHASE13_RESULTS.md`. If N_pref is stable across all four arms, the
width parameter doesn't matter for this endpoint and we say so plainly. If
it swings, that is a stated limitation of C3, not something discovered
later and quietly absorbed.

## C1's n>2 generalization — documented, not re-derived

ASTRAL principle 4 is stated for exactly two precursors (the composition
slice between them). C1's "minimum over all pairwise interfaces" rule for
n > 2 precursors is a generalization beyond the paper's literal scope,
justified by a standard weakest-link/parallel-competing-pathway argument
(a strongly favored competing product at even one interface can dominate
regardless of the other interfaces) — not a claim the paper itself makes
for n > 2. Documented here as a modeling extension, kept as specified.

## Secondary endpoints (reported regardless of the primary bar)

1. **Full pair enumeration** (Step 0 of the spec): every within-target pair
   in `misc/astral_validation_set.json` with measured phase purity on both
   sides, not just predicted-vs-conventional. Report counts and the |Δ
   purity| distribution. Does not gate Phase 13; sets the power of this
   secondary analysis.
2. **Sign agreement** with measured phase-purity differences across the
   full enumeration, with binomial CI.
3. **Per-channel agreement table**: for each of C1–C7 individually, how
   often does its sign alone agree with the measured-purity sign, on
   gradeable pairs.
4. **Channel ablation**: drop one channel at a time from the aggregate,
   recount N_pref. Identifies which channels are load-bearing for the
   primary result (positive or negative).

## Gates

Unchanged from `core/ranker.py`: `format_ok`, `balances`, `precursors_exist`,
`charge_neutral`, `temperature_physical`. A route failing any gate is
excluded from that pair's comparison on every channel (symmetric
gradeability, per the aggregation spec).

## Rules restated (from the spec, binding)

- ASTRAL is the readout, never the training or calibration signal. MAD
  scales for the comparator come from unlabeled generations
  (`runs/gdpo-qlora-beta-ablation-probe/generations.jsonl` and the Phase 12
  dump), never from ASTRAL.
- One design iteration. This file is not edited after scoring begins. A
  second iteration is a new pre-registration and a stated limitation, not a
  quiet revision.
- Do not touch `validator.py` or `ranker.py`. New code lives in
  `core/comparator.py`.
- Phase 13 is CPU-only and does not depend on Phase 12's step count.

## Phase 12 note (settled, not left open)

Phase 12 (GDPO from RS-SFT, beta=0) was pre-registered to read at checkpoint
300 for the matched Arm A comparison. It was run to just past step 300 and
stopped there deliberately, and **that is the read** — not resumed to 600
for a second look. Resuming later to see whether a longer run changes the
answer would turn one pre-registered read into two reads with the
opportunity to pick, which is exactly the failure mode pre-registration
exists to prevent. If Phase 12 is ever resumed toward 600, any number from
that continuation is exploratory and must be labeled as such, separately
from the checkpoint-300 result already on record in `misc/PHASE12_RESULTS.md`.
This also means Phase 14's launch does not wait on Phase 12 reaching 600 —
checkpoint 300 is Phase 12's final answer.

---

## Addendum, 2026-09-16 — C3's melting-point data source

*Written and committed before `core/comparator.py` exists or any channel is
scored. This is an addendum, not a revision — the sections above are
unchanged. Follows the investigation ordered by regular Claude: confirm the
data gap, test the preferred fix, measure gradeability by side before
choosing among the remaining options.*

**The gap.** C3's Tammann window needs a melting point for every precursor.
Materials Project has none: a direct query of
`mpr.materials.summary.available_fields` for melt/boil/thermal fields
returned zero results. MP is a DFT ground-state energetics database; it does
not carry experimental thermal data for compounds (`Element.melting_point`
covers elements only, not compounds).

**Option 1 (regression from MP formation energy per atom) — tested, rejected.**
Fit `T_melt ~ a + b * E_f/atom` on 16 real, individually-verified reference
compounds (Wikipedia infobox, WebFetch-checked) pulled through the same PD
cache the validator uses. Global fit R² = 0.229. Stratified by anion class:
oxides R² = 0.551 (n=12, resid. std 395 K), halides R² = 0.816 (n=4, almost
certainly overfit at that n). Residual uncertainty (±400–600 K for oxides) is
comparable to or larger than C3's own window width (100–340 K depending on
sweep arm) — a regression this noisy would misplace compounds inside/outside
the window essentially at random. Rejected on the numbers, not on principle.

**Option 2 (chemistry-class-driven table) — adopted, built target-driven
rather than corpus-frequency-driven.** The failure mode regular Claude
flagged for a hand-curated table is corpus-frequency bias: dense coverage for
precursors common in the Kononova/SFT training corpus, sparse for
ASTRAL-novel ones — reproducing finding 8's gradeability asymmetry in a new
channel. The table built here avoids that by construction: it was not built
from corpus frequency at all. It was built by enumerating every unique
precursor formula that actually appears across all 35 ASTRAL target pairs —
both the traditional and predicted side — and sourcing a melting point for
each one, regardless of how common or novel that compound is.

Enumeration of `misc/astral_validation_set.json`: 26 unique formulas on the
traditional side, 28 on the predicted side, 34 unique in the union (20
shared, 5 traditional-only — carbonates and one phosphate — 8 predicted-only
— the niobates/borates/metaphosphates ASTRAL favors).

**Measured gradeability: 34/34 precursors covered, both sides, no
drop-out.** Every formula on both the traditional and predicted side now has
a sourced melting point. There is no asymmetry to adjudicate — coverage is
100% and symmetric by construction, which is what "target-driven, not
corpus-frequency-driven" is supposed to guarantee. Per regular Claude's own
rule ("if the ASTRAL side drops out materially more often, that decides it
without a judgment call"): it doesn't drop out at all, on either side, so
Option 2 is adopted outright. Option 3 (dropping C3 from Phase 13 scoring)
is not needed.

Sources: Wikipedia infoboxes via WebFetch for 32/34 compounds; 2 compounds
(K3PO4, LiPO3) had no Wikipedia thermal data and were sourced via WebSearch
from secondary chemical-supplier/literature pages instead — flagged below as
lower-confidence, same posture as the Tammann citation hedge above.

Full table (°C, matching `VOLATILE_T`/`PRECURSOR_DECOMP_T` convention in
`core/ranker.py`; keys normalized via `SynthesisValidator._normalize_formula`
when this is implemented in `core/comparator.py`):

| formula | T_melt (°C) | note |
|---|---|---|
| Al2O3 | 2054 | |
| BaO | 1923 | |
| Bi2O3 | 817 | |
| B2O3 | 450 | trigonal crystalline form; common amorphous/glass form has no distinct mp |
| CuO | 1326 | |
| Fe2O3 | 1539 | |
| GeO2 | 1115 | |
| K2CO3 | 891 | |
| K3PO4 | 1340 | lower confidence — single non-Wikipedia source, not cross-checked |
| KNbO3 | 1100 | |
| KPO3 | 807 | |
| Li2CO3 | 723 | concurrent decomposition reported near 1300°C in some sources |
| Li2TiO3 | 1533 | |
| LiBO2 | 849 | |
| LiNbO3 | 1240 | |
| LiPO3 | 656 | lower confidence — two sources disagree (656 vs. 669°C), split difference not taken, lower value used |
| MgO | 2852 | |
| MnO | 1945 | |
| Na2CO3 | 851 | anhydrous form (hydrates decompose well below this) |
| NaBO2 | 966 | |
| NH4H2PO4 | 190 | reported "melting point" is concurrent with onset of decomposition to NH3 + molten H3PO4 |
| NiO | 1955 | |
| Pr6O11 | 2183 | |
| Sc2O3 | 2485 | |
| SiO2 | 1713 | |
| SrO | 2531 | |
| Ta2O5 | 1872 | |
| TiO2 | 1843 | |
| V2O3 | 1940 | |
| WO3 | 1473 | |
| Y2O3 | 2425 | |
| ZnO | 1974 | reported value is a decomposition point, not a clean liquid-phase melt |
| ZrO2 | 2715 | |

**Caveat carried forward, not resolved here**: several entries (Li2CO3,
NH4H2PO4, ZnO) report a temperature where decomposition and melting are
concurrent rather than a clean solid→liquid transition. This doesn't block
C3's usability — the channel only needs a boundary temperature past which
the window closes, and a decomposition onset serves that role exactly like a
melting point does. It is flagged here as a factual caveat about what the
number means, not a gap in coverage. K3PO4 and LiPO3's single/conflicting
sources are flagged for chemist spot-check before any writeup states them
without qualification, same posture as the unread Merkle & Maier citation
above.

**Next step**: build `core/comparator.py` using this table for C3, per the
rest of this pre-registration, unchanged.

---

## Addendum 2, 2026-09-18 — iteration 2: the primary endpoint was a design error

*Written and committed before any of the fixes below are applied to
`core/comparator.py`, per the same discipline as addendum 1: this locks
the new design first, iteration 1's number stands as reported
(`misc/PHASE13_RESULTS.md`), and this is a disclosed second iteration —
"a new pre-registration and a stated limitation, not a quiet revision,"
per this file's own rule above.*

**Why iteration 1 is being reopened.** `N_pref ≥ 25/35` can be satisfied
by a constant comparator that always prefers the predicted set, with zero
physics inside it — the metric cannot distinguish that degenerate case
from a working ranker. What was measured (17 preferences, 0 against, 18
forced ties) is close to that degenerate case for a diagnosed reason: one
channel's calibration scale was miscalibrated by a factor of ~50,000
(C7's MAD scale, 2.085e-05, computed from a corpus where nearly every
completion releases no gas), and two channels (C4, C7) are perfectly
confounded with the label on this dataset (every traditional route has
exactly 3 precursors, every predicted route exactly 2). Full diagnosis:
`misc/PHASE13_RESULTS.md`.

**The full-enumeration primary endpoint does not have the power originally
hoped for, and this is now a settled fact, not an open question.**
`misc/astral_validation_set.json`'s own `headline_numbers` records 224
total reactions in ASTRAL's underlying screen, but this file extracts only
the best-traditional and best-predicted route per target — 35 pairs, no
more, confirmed by exhaustive key-set inspection of all 35 target records
and a repo-wide search that found no fuller ASTRAL dataset anywhere in
this codebase. The "full within-target pair enumeration" (Step 0) is
therefore capped at the same 35 pairs already scored, not the ~150 a power
calculation would want:

| n | agreement rate needed to detect anything (80% power, α=0.05, two-sided, vs. chance=50%) |
|---|---|
| 17 (iteration 1's gradeable count) | 84.0% |
| 35 (ceiling with the data on hand) | 73.7% |
| 150 (hoped-for) | 61.4% |
| 224 (ASTRAL's full screen, not available here) | 59.4% |

Every agreement rate this project has measured across three independent
physics-grounded verifiers (validator 50.0%, ranker v2 60.0%, comparator
47.1%) is below what n=35 could detect at 80% power. **Conclusion: no
binary bar at n≤35 is both achievable by any historically-observed effect
size and statistically meaningful.** This is itself the finding — about
ASTRAL data availability, not about the comparator — and is reported as
such rather than forcing a bar that no real effect in this project's
history could clear.

### New primary endpoint: descriptive, not a binary bar

Sign agreement between the comparator's margin and the sign of measured
phase-purity difference, on every gradeable pair in the 35-pair set (not
just predicted-vs-conventional — this is unchanged from iteration 1's
Step 0 scope, since no larger enumeration exists), reported as a point
estimate with a 95% binomial CI. No pass/fail bar is set, for the power
reason above. This mirrors exactly how the validator's 50.0% and ranker
v2's 60.0% have been reported in this project — descriptively, with the
CI doing the work a binary bar cannot at this n.

### New secondary endpoint: `N_pref`, confound stated every time

`N_pref` is retained and reported, but every report showing it must state
the label confound explicitly: on this specific 35-pair set, traditional
routes are always 3-precursor and predicted routes are always 2-precursor,
so any channel keyed to precursor count or total gas release reproduces
ASTRAL's own principle 1 by construction, not by an independent physics
read. `N_pref` is never reported alone after this addendum.

### The fixes, locked before implementation, in order

1. **Rank-transform every channel** over the calibration corpus
   (`runs/gdpo-qlora-beta-ablation-probe/generations.jsonl` +
   `runs/gdpo-qlora-gdpo-phase12-rssft-beta0/generations.jsonl`, unchanged
   sources) instead of raw-diff/MAD-scale. Per channel: store the sorted
   array of calibration raw values; at scoring time, a route's value maps
   to its percentile rank within that distribution (0 to 1); the pairwise
   diff is `percentile(a) − percentile(b)`, bounded in [−1, 1] regardless
   of the channel's raw scale. This fixes C7's ~50,000x scale blowup,
   C4's exact-zero MAD, and C6's saturation in the same change, because
   none of the three depend on a MAD estimate any more. Stored as
   `misc/comparator_scales_v2.json` — v1 is kept, not overwritten, as the
   iteration-1 record.
2. **Fix the balance-solver candidate-set gap inside `core/comparator.py`
   only.** A `SynthesisValidator` subclass local to `comparator.py`
   overrides `_find_balanced_reaction` to add the one missing candidate
   diagnosed in `misc/PHASE13_RESULTS.md` (`["CO2","H2O","O2","NH3"]`,
   which balances NH4H2PO4-containing routes cleanly with no spurious gas
   uptake) between the existing `["CO2","H2O","O2"]` entry and the
   full-`VOLATILE_FORMULAS` fallback. `validator.py` and `core/ranker.py`
   are not touched — Arm A and Arm B stay reproducible, per this file's
   restated rule above. Separately (not a comparator change): audit every
   prior training-generation dump for how often the model itself declared
   an ammonium precursor and was silently scored zero everywhere by this
   same gap — reported in `misc/PHASE13_RESULTS.md`'s iteration-2 section,
   not acted on further here.
3. **Demote C1 to diagnostic-only.** Computed and logged in every
   breakdown, excluded from the scored aggregate and from `N_pref`. 0/35
   gradeable on ASTRAL and 5.4% on the calibration corpus — the n>2
   pairwise-interface generalization does not hold up in practice, per
   addendum 1's discussion of C1 as a documented extension "kept as
   specified"; iteration 2 stops scoring it rather than re-deriving it.
4. **Keep C4 and C7 in the scored aggregate**, both permanently flagged
   as label-confounded on the ASTRAL dataset in every table that shows
   them (this addendum's "New secondary" section above applies to any
   result driven substantially by either channel, not only to `N_pref`
   itself).

**One design iteration per addendum, restated**: this file is not edited
again after iteration 2's scoring begins. A third iteration would be a new
addendum and a stated limitation, exactly as this one is for iteration 1.
