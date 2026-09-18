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
