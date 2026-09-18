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

### New primary: sign agreement with measured phase purity

**20/35 agree = 57.1%, 95% CI [39.4%, 73.7%].** Stable across all four C3
sensitivity-sweep arms (A/B/C/D all read exactly 57.1% — C3's own
quadratic-penalty self-normalization means its width parameter barely
moves the rank it ends up at). Still consistent with chance (CI straddles
50% comfortably), but the point estimate is now the highest of the three
instruments measured in this project (validator 50.0%, ranker v2 60.0%,
comparator iteration 2 57.1%) — still far short of a claim, given the
overlapping CIs, but no longer indistinguishable from the other two
either.

**Important clarification, checked directly rather than assumed**:
dropping BOTH label-confounded channels (C4, C7) simultaneously leaves
sign agreement completely unchanged — still exactly 20/35 = 57.1%, the
identical 20 targets agreeing. Only `N_pref` moves when C4/C7 are removed
(35 → 31). **The primary endpoint is not an artifact of the confounded
channels** — it is carried by C2/C3/C5 (C6 stays structurally 0/35
gradeable on ASTRAL, C1 is diagnostic-only), which is reassuring for
treating 57.1% as a genuine read on this design rather than a confound
side-effect, even though the CI is still too wide to call it a result.

### Secondary: `N_pref`, confound stated

**N_pref = 35/35** (against=0, tie=0) — every single pair now favors
predicted, exactly the degenerate case iteration 1's diagnosis warned
about. Per-target ablation confirms why: dropping either C4 or C7 *alone*
leaves N_pref at 35/35 unchanged (the other confounded channel alone is
still enough to pin every pair positive); only dropping **both**
simultaneously moves it, to 31/35. This number is reported only with this
context attached, never alone, per the addendum.

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
sharply in later runs (RS-SFT and its descendants), for reasons not
investigated further here — plausibly RS-SFT's own bar-0.9 filtering
selected against routes that would have scored zero on these checks,
compounding the original bug's effect on what survived into later
training data, though this is a hypothesis, not verified.

**Not acted on further**: re-scoring historical capacity/gate-failure
numbers with the fix is a larger undertaking than this investigation's
scope and is not done here. Flagged as a limitation on any
`stoichiometry`/`amount_accuracy` capacity number already reported for
`gdpo-v3` or `beta-ablation-probe` specifically.

## What iteration 2 establishes

- The primary endpoint is no longer satisfiable by a constant (sign
  agreement, unlike `N_pref`, cannot be gamed by "always prefer
  predicted" — a constant comparator would score at chance on this
  metric by construction).
- **57.1% (20/35), CI [39.4%, 73.7%]** is the comparator's real read
  against experiment — the highest point estimate of the three
  instruments tried in this project, still statistically indistinguishable
  from the other two given overlapping CIs, and confirmed not to be an
  artifact of the two label-confounded channels.
- The balance-solver gap was real, generalizes well beyond the ASTRAL
  comparison, and materially affected the project's two largest early
  training runs — a genuine historical finding, not acted on further here.
- No claim is made that the comparator ranks real synthesis outcomes
  better than chance. No claim is made that it doesn't, either — every
  CI produced by this project's three verifier designs is too wide to
  settle this at the sample sizes ASTRAL's public data provides.
