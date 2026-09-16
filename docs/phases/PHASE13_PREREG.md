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
