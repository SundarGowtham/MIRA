# Phase 13 results — iteration 1 (honest record, before any fixes)

*Per `misc/PHASE13_PREREG.md`. This is the record of what the first,
locked design actually did when scored against the 35 ASTRAL pairs —
written and committed **before** any of the fixes below are applied, per
regular Claude's instruction: "write it now, before any fixes... this is
the honest record of iteration one." The pre-registration's own rule
("one design iteration... do not adjust after seeing a number") is
respected by NOT silently re-scoring with a different design and
reporting only the improved number — iteration 2, if it happens, gets its
own dated addendum to the pre-registration (see there) and is reported as
a disclosed second iteration, not a quiet revision.*

---

## The headline number was uninformative by construction — a design error

Raw output: **N_pref = 17/35, N_against = 0/35, N_tie = 18/35** — fails
the pre-registered ≥25/35 bar (one-sided binomial p = 0.63 against
chance).

**This number cannot be trusted as a physics result, and the reason is a
flaw in the endpoint itself, not just noisy data.** `N_pref` counts how
often the comparator prefers the predicted set over the traditional one.
A comparator that returns "prefer predicted" *unconditionally* — a
constant, with zero physics inside it — scores 35/35 on this exact metric
and clears the bar. The metric cannot distinguish a working ranker from a
constant that happens to agree with ASTRAL's own selection principle by
construction. What was actually measured (17 preferences, 0 against) is
close to that degenerate case, for a concrete, diagnosed reason below —
not because the comparator is subtly right 17 times out of 35, but
because two channels evaluate to a near-constant "prefer predicted" and
happen to dominate the sum by four orders of magnitude.

## 99.99% of every nonzero margin is one channel, and it is a scale-calibration failure

Every decided (non-tied) target's margin is in the tens of thousands
(e.g. `BaLiBO3` = 23,984.3), which is not a plausible magnitude for a sum
of MAD-normalized physics differences. The arithmetic:

`C7_gas_evolution`'s raw diff (predicted − traditional) takes exactly one
of three values across every gradeable pair: **+0.5, +1.0, or +1.5**
(verified directly against all 17 gradeable pairs — no other value
occurs). Its MAD scale, computed from the unlabeled calibration corpus
(`misc/comparator_scales_v1.json`), is **2.085×10⁻⁵** — because most
generated completions in that corpus release *no* gas at all
(`C7_gas_evolution` clusters extremely tightly near 0), so the median
absolute deviation is almost zero even though the channel's own range is
real. Dividing a diff of ~1 by a scale of ~2×10⁻⁵ inflates it by a factor
of **~50,000**:

```
0.5 / 2.085e-05  ≈ 23,982     (matches BaLiBO3's margin of 23,984.3 almost exactly)
```

The other three channels that manage to be gradeable at all (C2, C3, C5)
contribute a combined ~2.4 points to that same margin — noise by
comparison. **This is finding 20's failure mode in a different file**:
Phase 12's KL loss was 99.99% penalty term; Phase 13's margin is 99.99%
one miscalibrated channel. Same mechanism (a robust-scale estimator
degenerating on a near-constant or heavily-clustered raw distribution),
same lesson (a per-channel scale computed once and trusted blindly can
silently become the entire signal).

## C4 and C7 are perfectly confounded with the label on this dataset — not approximately, exactly

Every one of the 35 ASTRAL pairs has **traditional = exactly 3
precursors, predicted = exactly 2** (verified directly, no exceptions —
this is baked into ASTRAL's own principle 1, not an artifact of this
evaluation). Consequences, verified directly against the scored data:

- **`C4_interface_count`'s raw diff is the same constant, +2.0, on every
  single gate-passing pair** (predicted: k=1 interface → score −1;
  traditional: k=3 interfaces → score −3; diff = −1 − (−3) = +2.0,
  always). It is not merely correlated with the label here — it is a
  deterministic function of it. (C4 did not actually enter the reported
  margin above, because its own MAD scale is exactly 0 — see below — but
  it will as soon as the scale-degeneracy fix lands, so this is flagged
  now, before that happens.)
- **`C7_gas_evolution`'s diff is always positive** (+0.5/+1.0/+1.5,
  above) because 3-precursor carbonate/phosphate routes mechanically
  evolve more gas per mole of target than 2-precursor oxide/borate
  routes — again, a function of precursor *count and class*, which is
  the label here, not an independent physics read on any given target's
  chemistry.

Any channel keyed to precursor count or total gas release will reproduce
ASTRAL's principle 1 on this specific comparison by construction. That is
not evidence the comparator understands selectivity; it is evidence the
dataset's two sides differ systematically in a way both channels happen
to key on directly.

## The real read is the secondary endpoint, and it lands at chance — for the third time

Sign agreement between the comparator's margin and the sign of *measured*
phase-purity difference, on the 17 pairs where anything was gradeable at
all:

**8 agree, 9 disagree — 47.1%, 95% CI [23.0%, 72.2%].** Consistent with
chance. Disagreeing targets: `Li2TiSiO5`, `LiNbGeO5`, `Li2TiGeO5`,
`BaNaBO3`, `NaSrBO3`, `KNbWO6`, `NaSiBO4`, `KTiNbO5`, `LiSi2BO6`.

Per-channel sign agreement (independent of the confound above — this asks
each channel's own sign, alone, against experiment):

| channel | agreement | n gradeable (of 35) |
|---|---|---|
| C1 selectivity margin | ungradeable | 0 (0.0%) |
| C2 unspent driving force | 50.0% | 14 (40.0%) |
| C3 reactive temperature window | 50.0% (3 ties) | 14 (48.6%) |
| C4 interface count | ungradeable (MAD=0) | 0 (0.0%) |
| C5 volatilization | 42.9% | 7 (48.6%) |
| C6 decomposition clearance | ungradeable | 0 (0.0%) |
| C7 gas evolution | 47.1% | 17 (48.6%) |

**Four independently-derived, physics-shaped channels that managed to be
gradeable at all, every one at chance against real experimental
outcomes.** This is the third instrument in this project to land there:
the validator itself (finding-level baseline, 50.0% = 17/34), ranker v2
(finding 19, 60.0% = 21/35, and even that was on a dataset that couldn't
exercise 3/8 of its channels), and now the comparator (47.1% = 8/17, on a
dataset where 3/7 channels never fired at all). The confidence intervals
are far too wide at these sample sizes to call any single one of these a
result — but a physics-grounded, chemist-legible verifier landing at
chance against real synthesis outcomes on three separately-designed
attempts is the pattern worth reporting, more than any one of the three
numbers alone.

## C1's 0/35 and C6's 0/35 are structural, not calibration artifacts

Unlike C4/C7 (confounded) and the MAD-scale bug (fixable), C1 and C6
being 0/35 gradeable are **separate, already-diagnosed structural facts**,
not consequences of the scale problem above:

- **C1 (`selectivity_margin`)**: every traditional route has 3
  precursors; C1's n>2 generalization requires the target to appear as a
  kink product on some pairwise tie line, which is geometrically
  near-impossible for a genuine 3-component target (verified directly on
  `BaLiBO3`: 0/3 pairwise interfaces show the target as a kink product).
  Gradeable on only 5.4% of unlabeled training generations too (51/940) —
  this is not an ASTRAL-specific artifact, it is close to structurally
  dead in general for 3+-precursor routes.
- **C6 (`decomposition_clearance`)**: gradeable per-route reasonably
  often (31.5% on the calibration corpus), but the *symmetric*
  gradeability requirement (both sides of a pair must have a tabulated
  decomposition-onset precursor) almost never holds when one side is
  carbonate-heavy and the other is not — the same structural asymmetry
  mechanism the pre-reg addendum worried about for C3's melting-point
  coverage, showing up instead in C6 (0/35 on ASTRAL despite nonzero
  single-route gradeability).

## What this iteration establishes and does not

- **Does not establish**: that the comparator prefers ASTRAL's routes
  "on physics" (the 17/0/18 split is an artifact of two dominant,
  label-confounded/miscalibrated channels, not evidence of selectivity).
- **Does not establish**: that the comparator fails to rank real routes
  either — the sign-agreement read (47.1%, CI [23%,72%]) is consistent
  with chance but the interval is wide enough to be consistent with a
  real, moderate effect too. Underpowered, not negative.
- **Does establish**: `N_pref` against a fixed-direction bar is the wrong
  primary endpoint for this design space, because it can be satisfied by
  a constant. A pre-registration addendum replacing it is required before
  any further scoring (see `misc/PHASE13_PREREG.md`'s dated addendum).
- **Does establish**: the calibration MAD scale for at least one channel
  (C7) is unusable as computed, and the fix (rank-transform) needs to be
  decided and written down before it is applied, not applied first and
  reported after.

## Next steps (see the pre-registration addendum for the locked plan)

1. New primary endpoint: sign agreement with measured phase purity,
   full within-target pair enumeration, with a bar set by a power
   calculation on the actual usable pair count.
2. Rank-transform every channel over the calibration corpus instead of
   raw-diff/MAD — fixes C7's domination, C4's zero-MAD, and C6's
   saturation in one change.
3. Fix the balance-solver candidate-set gap **inside `core/comparator.py`
   only** (a locally-subclassed `SynthesisValidator`), never touching
   `validator.py` — Arm A stays reproducible.
4. Demote C1 to a diagnostic-only channel (computed, logged, not scored).
5. Keep C4 and C7 in the scored aggregate but permanently flag both as
   label-confounded on the ASTRAL dataset in every report that shows
   them.
6. Separately: audit all prior training-generation dumps for how often
   the model itself declared an ammonium precursor and was silently
   scored zero everywhere by the same balance-solver gap.

---

# Iteration 2 — fixes applied per addendum 2 (2026-09-18)

*All fixes below were locked in `misc/PHASE13_PREREG.md`'s addendum 2
before any of this code was written, per the pre-registration discipline.
Iteration 1's numbers above are unchanged and stand as the record of what
that design actually did.*

## The fixes, as implemented

1. **Rank-transform** (`core.comparator._percentile_rank`): every
   channel's raw value now maps to its mid-rank percentile (0–1) within
   the calibration corpus's distribution for that channel
   (`misc/comparator_scales_v2.json`, sorted raw-value arrays from the
   same two unlabeled sources as v1). Pairwise diff is
   `percentile(a) − percentile(b)`, bounded in [−1, 1] per channel
   regardless of scale. `misc/comparator_scales_v1.json` is kept
   untouched as the iteration-1 record.
2. **Balance-solver fix, scoped to `core/comparator.py` only**: a new
   `_ComparatorValidator(SynthesisValidator)` subclass overrides
   `_find_balanced_reaction` to add the missing candidate set
   (`["CO2","H2O","O2","NH3"]`) between the existing
   `["CO2","H2O","O2"]` entry and the full-`VOLATILE_FORMULAS` fallback.
   `Comparator.__init__` installs this instance in place of `Ranker`'s
   default `_v`, so both the `balances` gate and C7's own lookup use the
   fix consistently. `validator.py` and `core/ranker.py` are unmodified —
   verified directly on the diagnosed case: `LiZnPO4`'s traditional route
   (`Li2CO3, NH4H2PO4, ZnO`) now passes `balances: True` and all six
   scored channels compute real values.
3. **C1 demoted to diagnostic-only** (`core.comparator.
   DIAGNOSTIC_CHANNEL_NAMES`): still computed and reported per-channel,
   excluded from `SCORED_CHANNEL_NAMES` and the margin sum.
4. **C4 and C7 kept scored, permanently flagged**
   (`core.comparator.LABEL_CONFOUNDED_CHANNELS`) in every breakdown and
   every report that shows them.

`tests/test_comparator.py` (antisymmetry, one sign test per channel,
gate-failure → all-None) passes unchanged against the rewritten module —
17/17 checks (`run_logs/test_comparator_v2.log`).

## Rescoring the 35 ASTRAL pairs (`research/phase13_astral_scoring_v2.py`)

**Every one of the 35 pairs is now gradeable on every scored channel**
(the balance-solver fix recovered all 18 previously gate-blocked
traditional routes; rank-transform gave C4 a usable scale for the first
time). No more forced ties.

### CORRECTION (2026-09-19): the "new primary" section below was wrong

*The text immediately below this note is the ORIGINAL iteration-2 writeup,
kept verbatim (not deleted) per the pre-registration's own discipline
("this file is not edited after scoring... a stated limitation, not a
quiet revision") — struck through in spirit, corrected in the section
that follows it. No re-scoring was done to produce this correction; only
the interpretation was wrong, caught on review by regular Claude and
verified directly against the data already on hand.*

~~**20/35 agree = 57.1%, 95% CI [39.4%, 73.7%].** ... the point estimate is
now the highest of the three instruments measured in this project
(validator 50.0%, ranker v2 60.0%, comparator iteration 2 57.1%)~~

**This comparison to 50% was the error.** `N_pref = 35/35` means
`sign(margin) > 0` on every single pair — the comparator prefers
"predicted" unconditionally on this dataset. Once that is true, "sign
agreement with measured purity" reduces exactly to counting how often
predicted purity actually beats traditional purity, because the
comparator's own sign never varies to disagree with anything. That count
is a property of ASTRAL's 35 curated pairs, not a measurement of the
comparator: **verified directly, 20/35 targets have measured predicted
purity > traditional purity** (0 exact ties) — the identical number, on
the identical targets, as the "sign agreement" figure above. `C4` alone,
a pure constant (+2.0 on every pair, no per-target information
whatsoever), reproduces this exact 20/35 by itself.

**The correct statement: on these 35 pairs, the comparator's measured
discriminative power is zero.** The right baseline for this dataset was
never 50% — it is 57.1%, the base rate of "predicted wins on purity," and
the comparator's full aggregate sits exactly on that base rate with zero
pairs differing from what a constant "always prefer predicted" rule would
say. This is not "the highest point estimate of three instruments"; it is
not an estimate of the comparator at all.

**Why this happened, structurally, not as a channel-design flaw**: all 35
pairs are 3-precursor-traditional vs. 2-precursor-predicted, with no
exceptions (verified, iteration 1). Any channel keyed to precursor count,
interface count, or total gas release is therefore a CONSTANT on this
specific comparison by construction — and a constant channel scores
exactly the dataset's base rate, not "chance." **The 35 curated ASTRAL
pairs cannot evaluate a comparator of this kind at all** — this is a
dataset-confound limitation, not a defect in C1–C7's design.

**The within-target pair enumeration — the only design that could compare
routes on the same side of that divide — was checked formally
(`research/astral_pair_enumeration.py`, run 2026-09-19, not assumed from
the earlier ad hoc inspection): every one of the 35 targets carries
exactly 2 named routes with measured purity (traditional, predicted), so
the enumeration is identically the same 35 pairs already scored, not a
larger set. `|Δ purity|` distribution across these 35: min 0.020, median
0.120, mean 0.193, max 0.630 (`results/astral_pair_enumeration.json`).
ASTRAL's own `headline_numbers` field records 224 total reactions in the
underlying screen, but that fuller dataset is not present anywhere in
this repository (confirmed by listing every astral-related file — 53 of
them, all tracing back to this same 35-target extract). **The ~150-pair
enumeration that would give this endpoint real power, and let it compare
same-precursor-count routes, does not exist in this repo. This is the
finding**, not a step that was skipped: ASTRAL's public data, as held
here, cannot validate a comparator whose channels are sensitive to
precursor count or class, because the dataset's only comparison axis IS
precursor count and class.

### Secondary: `N_pref`, confound stated, ablation inconsistency resolved

**N_pref = 35/35** (against=0, tie=0) — every single pair favors
predicted, the fully degenerate case. Per-target ablation: dropping
either C4 or C7 *alone* leaves N_pref at 35/35 unchanged (the other
confounded channel alone still pins every pair positive); dropping
**both** simultaneously moves it to 31/35 (4 targets flip sign:
`Li3Sc2(PO4)3`, `KNbWO6`, `KTiNbO5`, `LiNbWO6`).

**Apparent inconsistency, checked and resolved, not asserted away**:
dropping both C4 and C7 leaves sign agreement unchanged at 20/35 even
though 4 pairs flip margin sign — flagged by regular Claude as needing a
direct check, since a flipped pair should change its own agreement
status unless its `Δpurity` is exactly zero. None of the 4 flipped
targets has a purity tie (`Li3Sc2(PO4)3`: Δpurity=+0.53;
`LiNbWO6`: Δpurity=+0.12; `KNbWO6`: Δpurity=−0.05; `KTiNbO5`:
Δpurity=−0.09 — verified directly against the scored data). What actually
happens: **2 of the 4 flip from agree→disagree** (`Li3Sc2(PO4)3`,
`LiNbWO6`, both Δpurity>0, margin flips from favoring-predicted to
favoring-traditional) **and the other 2 flip from disagree→agree**
(`KNbWO6`, `KTiNbO5`, both Δpurity<0, same margin flip now happens to
match). Net change = 0, exactly and only because the flips split 2-and-2
in opposite directions on this specific set of 4 targets — not a bug, not
a coincidence requiring purity ties, a real cancellation confirmed by
direct arithmetic. Given the interpretation above, `N_pref` was already
known to be uninformative on this dataset regardless of this specific
arithmetic point.

### Per-channel sign agreement (arm B)

| channel | agreement | n gradeable (of 35) | flags |
|---|---|---|---|
| C1 selectivity margin | — | 0 (0.0%) | diagnostic-only, still structurally dead |
| C2 unspent driving force | 44.4% (12/27) | 27 (77.1%) | |
| C3 reactive temperature window | 59.4% (19/32) | 32 (100% decided-or-tied) | |
| C4 interface count | 57.1% (20/35) | 35 (100%) | label-confounded |
| C5 volatilization | 50.0% (8/16) | 16 (100% decided-or-tied) | |
| C6 decomposition clearance | — | 0 (0.0%) | structurally 0/35, unchanged from iteration 1 |
| C7 gas evolution | 57.1% (20/35) | 35 (100%) | label-confounded |

C2's 44.4% is now BELOW chance on its own (n=27, small enough that this
isn't a strong claim either way) — worth flagging as a direction to watch
if this channel is revisited, not acted on here (one design iteration).

## Historical ammonium-precursor audit (`research/audit_ammonium_precursor_history.py`)

Regular Claude's side request: how often did the MODEL ITSELF (not just
ASTRAL's curated literature routes) declare an ammonium precursor during
training and get silently zeroed by the same validator.py gap? Scanned
all 7 real generation dumps in `runs/` (19,380 completions total),
confirmed the mechanism generalizes beyond `NH4H2PO4` specifically by
directly re-checking a sample of matches against both the original and
fixed balance solver.

**1,113/19,380 completions (5.74%) mention an ammonium species.** Impact
is concentrated in the two largest, earliest runs:

| dump | n completions | % mention ammonium | sampled balance-failures (original) | recovered by fix |
|---|---|---|---|---|
| `gdpo-qlora-gdpo-v3` | 5,264 | 10.0% | 97/121 with an actual ammonium precursor (~80%) | 93 (95.9%) |
| `gdpo-qlora-beta-ablation-probe` | 4,896 | 9.1% | 109/138 (~79%) | 106 (97.2%) |
| `gdpo-qlora-gdpo-v4` | 1,364 | 2.0% | 0 sampled | — |
| `gdpo-qlora-gdpo-phase12-rssft` | 1,612 | 1.7% | 0 sampled | — |
| `gdpo-qlora-gdpo-phase12-rssft-beta0` | 5,704 | 1.2% | 1 | 1 |
| `gdpo-qlora-gdpo-run4-ranker` | 160 | 5.0% | 0 sampled | — |
| `gdpo-qlora-gdpo-phase12-smoke` | 380 | 2.9% | 0 sampled | — |

**Total sampled: 207 balance-failures under the original validator, 200
(96.6%) recovered by the fix.** This is not an ASTRAL-specific artifact —
whenever the model itself proposed an ammonium-salt precursor in the
project's two largest early training runs (`gdpo-v3`, the same
`beta-ablation-probe` used as half of this comparator's own calibration
corpus), roughly 4 times out of 5 that completion's `stoichiometry` and
`amount_accuracy` validator checks (both keyed on
`_find_balanced_reaction` finding a balance) were silently zeroed for a
software reason, not a chemistry one. Ammonium-precursor prevalence drops
sharply in later runs (RS-SFT and its descendants).

### RS-SFT propagation test (`research/test_rssft_ammonium_propagation.py`, 2026-09-19)

Regular Claude's hypothesis, tested directly rather than left open: if
RS-SFT's bar-0.9 filter selected against routes the balance-solver bug
was zeroing, ammonium-precursor prevalence should collapse between base
model's raw generations and RS-SFT's kept survivors. Detection corrected
from the historical audit's text-substring method to a chemistry-based
one: a precursor counts as ammonium-like if its parsed composition
contains BOTH nitrogen and hydrogen (substring matching on `"NH4"` missed
cases where a formula is stored in expanded elemental form — e.g.
`NH4H2PO4`'s own pymatgen reduced formula is `PH6NO4`, containing neither
`"NH4"` nor `"NH3"` as text).

**Base model, unfiltered (`results/astral_gen_n32_base.json`, 1120
completions): 136/1120 = 12.14%.**
**RS-SFT survivors (`data/rs_sft/rs_sft_train.jsonl` +
`rs_sft_val.jsonl`, 295 completions): 6/295 = 2.03%.**
**Ratio: 0.17 — a ~6x drop.** Directionally strongly consistent with the
propagation hypothesis.

**Caveat, stated plainly**: this is a proxy comparison, not a matched
pre/post-filter test — `build_rs_sft_dataset.py` never persisted rejected
(non-survivor) samples, so RS-SFT's own exact pre-filter distribution
cannot be reconstructed from anything on disk. The base-model figure comes
from ASTRAL's 35-target evaluation, a different target universe from
RS-SFT's `data/rl` 400-target sample. The result is a strong directional
signal, not a controlled measurement — reported as such, not overclaimed
as proof.

**Implication for Phase 12, flagged not retracted**: RS-SFT is Phase 12's
initialization, and its positive result (14/35 amplified,
`docs/phases/PHASE12_RESULTS.md`) rests on RS-SFT's own filtered survivor
set. If that set under-represents ammonium-precursor routes because of a
software bug rather than real chemistry, RS-SFT's "convention-avoidance"
property (finding 18) may be partly an artifact of which routes could
even pass the gate to be counted, not purely a preference the filtering
correctly identified. This does not undo Phase 12's measured result — the
routes RS-SFT did retain are real and the RL amplification measured
against them is real — but it is a footnote every future reference to
RS-SFT's provenance should carry.

**Not acted on further**: re-scoring historical capacity/gate-failure
numbers with the fix, or rebuilding RS-SFT with the fixed balance solver,
are both larger undertakings than this investigation's scope and are not
done here.

## What iteration 2 establishes

- **The primary endpoint, as originally written up in this iteration, was
  itself wrong** — see the correction above. `N_pref = 35/35` means the
  comparator's sign never varies, so "sign agreement" collapses to
  counting the dataset's own base rate (20/35 = 57.1%), not measuring the
  comparator. **The comparator's demonstrated discriminative power on
  these 35 pairs is zero.**
- The within-target pair enumeration was run formally and confirms this
  is a hard data-availability ceiling, not a step that was skipped: only
  35 pairs exist anywhere in this repository, and any channel sensitive
  to precursor count or class is a constant on this specific comparison
  by construction. **ASTRAL's 35 curated pairs, as held in this
  repository, cannot evaluate a comparator of this kind at all** — this
  is the finding, not a limitation to work around.
- The RS-SFT ammonium-propagation test is directionally strong (12.14% →
  2.03%, ratio 0.17) though a proxy, not a matched comparison — the
  balance-solver bug plausibly shaped what RS-SFT's own filtering kept,
  with a stated but unresolved implication for Phase 12's provenance.
- The balance-solver gap was real, generalizes well beyond the ASTRAL
  comparison, and materially affected the project's two largest early
  training runs — a genuine historical finding, not acted on further here.
- No claim is made that the comparator ranks real synthesis outcomes
  better or worse than a trivial rule. This specific dataset cannot
  settle that question either way, for a structural reason (the label is
  confounded with precursor count) rather than a statistical-power one —
  a stronger and more useful negative than "underpowered," because it
  says what evidence would actually be needed (routes matched on
  precursor count, which ASTRAL's public 35-pair extract does not
  provide) rather than just "more of the same data."
