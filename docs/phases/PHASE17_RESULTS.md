# Phase 17 results

Dated entries, one per block, written as each closes. `prereg_commit` on
every results JSON: `790d0852f11d62d442fc299f8c277840b9d8a16d`.

## 2026-10-28 — Headline withdrawn; Blocks A, B, C, G, R corrected

**WITHDRAWN.** On 2026-10-27 this doc would have reported: "practice
(on-policy RL) reaches recipes a pure tilt cannot" — B1's negative c_s for
the three practice steps (S5/S6/S7), R-h's L6.gains back-test (3/4 of
Phase 12's gained targets sat far below epsilon under RS-SFT's estimated
probability, yet were reached), and B3's leash overshoot (6–7×), read
together as three independent lines of evidence. They were never
written to this file, on regular Claude's explicit instruction, pending
two controls. **That instruction was correct: the three lines are not
independent, and the shared cause — the K=8-trace marginal estimator's
severe tail miscalibration and errors-in-variables bias — explains most
of what made the headline look real.** The corrected results below
replace it.

### Block A — replication, bootstrap, paired diff

- The four `*_replication.json` ASTRAL legs reused `--seed=42` (the
  default `astral_model_generations.py` never overrides), making them
  byte-identical to the original headline files (1,120/1,120 samples for
  Phase12@300) — **deterministic re-runs, not replications.** Relabelled
  in `block_a_manifest.json`'s `astral_legs_relabel` block.
- Real sampling-noise floor (10,000-resample bootstrap of each target's
  own 32 draws, `block_a_hit_bootstrap.json`): Phase12@300 orig 14/35,
  bootstrap median 12, 95% [9,14]; E1@300 orig 14/35, median 13, 95%
  [11,14]; RS-SFT orig 10/35, median 9, 95% [8,10]. Phase12@300's
  interval overlaps RS-SFT's at [9,10].
- **Paired bootstrap, Phase12@300 − RS-SFT hits, same resampled positions
  per target** (`block_a_paired_bootstrap_phase12_vs_rssft.json`): orig
  diff = 4 (14−10). Bootstrap median = **2**, 95% CI **[−1, 5]** —
  includes 0. Amplification from Phase 12 over its own RS-SFT init is
  **consistent with a real gain but not statistically decisive**, exactly
  the finding-22 framing ("consistent with amplification, not yet
  statistically decisive") — this paired check does not sharpen it to a
  significant result, and does not overturn it either.

### Block B — B1 signature regression, corrected

**The errors-in-variables control is real: every step's original c_s was
inflated by shared measurement noise in log π_before.** Split-trace
control (`b1_split_trace_control.json`, 4-vs-4 trace split, before model
re-scored on a dedicated GPU forward pass): for every step, the original
c (regressor and outcome sharing the same noisy log π_before) is more
negative than the split estimate (regressor and outcome from independent
trace halves). A noise-only placebo (Δ_before,B − Δ_before,A regressed on
Δ_before,A, no real training step) reproduces most of that gap.

**Corrected reading (`b1_split_trace_control_corrected.json`): c_split is
already decorrelated from the shared-noise artifact and should be
compared to 0 directly, not to the placebo** (the placebo instead
explains c_original's bias: c_original − c_noise_placebo ≈ c_split,
confirmed within 0.03 nats for 6/7 steps, within 0.031 for the 7th, S1).

| step | c_original [CI] | c_split [CI] | b_split [CI] |
|---|---|---|---|
| S1 (base→corpus_sft) | −0.583 [−0.632,−0.534] | −0.363 [−0.423,−0.301] | 7.524 [0.283,15.073] |
| S2 (base→rs_sft) | −0.493 [−0.550,−0.443] | **−0.224 [−0.298,−0.161]** | 3.480 [−1.205,8.657] |
| S3 (base→e3) | −0.482 [−0.533,−0.435] | **−0.229 [−0.288,−0.177]** | 0.349 [−4.415,5.715] |
| S4 (base→e3c) | −0.496 [−0.550,−0.444] | **−0.243 [−0.307,−0.186]** | 1.437 [−3.550,7.163] |
| S5 (corpus_sft→GDPO) | −0.094 [−0.171,−0.021] | **+0.045 [−0.001,0.092]** | 2.211 [−2.027,6.502] |
| S6 (rs_sft→Phase12) | −0.111 [−0.155,−0.081] | **−0.039 [−0.075,−0.002]** | 4.296 [0.717,8.018] |
| S7 (rs_sft→E1) | −0.072 [−0.112,−0.034] | **+0.005 [−0.036,0.049]** | 1.749 [−0.320,3.564] |

- **P-B1a (fine-tuning, S2/S3/S4: c<0) — MET under the corrected split
  estimate** (all three CIs entirely below 0). The original "MET" was not
  wrong in direction, but its magnitude (c≈−0.49) was substantially
  inflated by the shared-noise artifact (true c≈−0.22 to −0.24).
- **P-B1b (practice, S5/S6/S7) — still NOT MET under the strict
  pre-registered joint criterion (b>0 with CI excluding 0, AND c's CI
  not entirely below 0), but the character of the failure has changed
  completely.** c_split is statistically indistinguishable from 0 for S5
  and S7 (the original "c confidently negative for all three practice
  steps" does not survive decorrelation). S6 keeps a small, barely
  significant negative c_split (−0.039, CI excludes 0 but only just).
  b_split is positive in point estimate for all three (2.2, 4.3, 1.7) but
  its CI excludes 0 only for S6 (0.72–8.02); S5 and S7's b_split CIs are
  wide and span 0 — **underpowered, not falsified.** Net: practice no
  longer shows the clean negative-c signature the original fit
  suggested; what remains is noise-consistent for 2/3 steps and a small,
  real residual for the third (S6), with no step clearing the full
  pre-registered amplification bar.
- Tail-calibration check (`b1_tail_calibration.json`): pooled
  observed/expected ratio for π̃<1e-4 = **7,425×**; per-model `<1e-6` bin
  ratios 100,378×–1,724,611×. The estimator this regression's c_s/d_s
  terms and R-h/B3 all depend on is severely biased low in exactly the
  probability range where the withdrawn headline lived.
- B2 (geodesic test, S6/S7 only) and B4 (ternary figure data) are
  unaffected by this correction (neither uses π_before as a regressor
  against itself); B2's razor-thin median ρ (S6=0.9946, S7=1.0) stands.
- **B3 (leash) is NOT corrected here** — its overshoot (6–7× the leash
  ceiling for S5/S6/S7) is built on the same biased tail estimates the
  calibration check indicts, but no split-trace-equivalent control was
  run for B3 specifically. Treat as **likely explained by tail bias**,
  not confirmed corrected.

### Block C — unbalanced OT

No change: **numerically infeasible as specified** (6/385 target-steps,
1.6%, converge at the pre-reg's reg=1e-2; before/after raw mass differs
by 10+ orders of magnitude within a single target, a genuine data
property no shared rescale fixes). Per regular Claude's item 4
conditional ("only if tail calibration shows tails are calibrated"): they
are **not** (7,425× pooled bias) — the π-renormalized-over-U(x) rerun is
correctly **not attempted**.

### Gate G1 / instrument check

Sign-consistency re-check (`instrument_check.json`): all 3 Block-A-vs-
Phase-16 sign comparisons have overlapping CIs at ε∈{1/100,1/200,1/500}
— confirmed noise, no real disagreement between the two measurements.

### Block R — R-g, R-h

**R-g, all-temperatures pooled CI, fixed** (averaging the three ARROWS³
targets' paired diffs *within* each bootstrap replicate instead of
concatenating replicates, `rg_all_temp_pooled_fixed.json`): R2_checker
mean_diff=−0.126, CI=[−5.08,4.26] → no climb. R5_comparator
mean_diff=2.71, CI=[−5.67,13.89] → no climb. **R4_ranker mean_diff=11.27,
CI=[1.88,20.62] → climbs** — flips the original concatenation-based
verdict (CI was [−13.89,29.40], "no climb"); the point estimate barely
moved, only the (previously invalid) CI width did.

**R-h, recounted without the estimator tail**
(`rh_recount_no_tail.json`): replaces π̃ in the unreachable-share
denominator with a literal on-disk occurrence count (same held-out scan
Gate G1 uses, no estimator). Share of each reward's top-decile recipes
with **zero** observed occurrences in the init model's own samples:

| reward | init | top-decile recipes | zero-observed | share zero | pooled CP upper bound (α=0.05) |
|---|---|---|---|---|---|
| R1 | corpus_sft | 89 | 70 | **78.7%** | 0.0018 |
| R2 | rs_sft | 243 | 106 | **43.6%** | 0.00014 |
| R4 | rs_sft | 193 | 59 | **30.6%** | 0.00026 |

R1 still exceeds R-h's 50% "unreachable" bar on a raw count, no estimator
involved — this one holds up. **R2 and R4 do not**: their raw zero-
observed share (43.6%, 30.6%) sits *below* the 50% bar the original
estimator-based read failed (83.8%, 79.2%). The tiny Clopper-Pearson
upper bounds (pooled, α=0.05) say these specific zero-observed recipes
are genuinely rare in a strict statistical sense — but "rare" and
"unreachable (>50% of the reward's top decile below ε)" are different
claims, and R2/R4 only support the former now.

L6.gains back-test, recounted on the full on-disk RS-SFT pool (232
samples/target, both sample_groups, `b1_tail_calibration.json`'s
`l6_gains_detail`, cross-referenced here):

| target | π̃_rs_sft (estimator) | hits / 232 on disk | empirical rate |
|---|---|---|---|
| BaLiBO3 | 3.16e-06 | 10/232 | 4.31% |
| KLi(PO3)2 | 6.0e-07 | 0/232 | 0.00% |
| Li2TiSiO5 | 1.0e-08 | 1/232 | 0.43% |
| NaSrBO3 | 0.492 | 18/232 | 7.76% |

3/4 targets the estimator placed at π̃=1e-6 to 1e-8 actually occur at a
real empirical rate 100–1000× higher (BaLiBO3, Li2TiSiO5) — direct,
target-level confirmation of the tail-bias mechanism Block B1's
calibration check found in aggregate. KLi(PO3)2 shows 0/232 even on the
full on-disk count — consistent with genuine rarity, not just estimator
bias, for that one target. NaSrBO3 remains the known over-estimate
anomaly (finding already explained: estimator-variance artifact from
which 8 traces were drawn, not a canonicaliser mismatch).

### Net effect on the withdrawn headline

- **B1's practice-step signature**: does not survive decorrelation for
  2/3 steps; a small, real residual remains for S6 only. **Withdrawn.**
- **R-h's reachability claim**: holds for R1, does not hold for R2/R4
  once the estimator tail is removed from the denominator. **Downgraded,
  not withdrawn wholesale** — one of three rewards (R1) still shows a
  genuine >50% unreachable share by a raw count.
- **B3's leash overshoot**: built on the same biased estimator; likely
  explained by tail bias, not independently re-verified.
- **Block C**: remains uninformative (numerically infeasible), contributes
  nothing to either side.

No single clean "practice reaches the unreachable" result survives. What
remains, worth carrying forward: R1's genuine unreachable majority, and
S6's small but real residual negative c — both narrower, real findings,
not the broad three-line headline originally reported.

## 2026-10-30 — Housekeeping: ledger regen, a reproducibility bug found and
confirmed, Gate G1's source located. Phase 17 closed.

No new experiments. `scripts/build_claims_ledger.py` gained a Layer 11
("Phase 17: why, where, and the reward audit") with ten entries covering the
capacity 2×2 table, the Price-equation sign-check, the information bound,
best-of-n/Goodhart, the Hodge decomposition, R-h's corrected reachability,
the corrected B1 split-trace table, tail calibration, and Block A's
bootstraps — see `results/ledger/CLAIMS_LEDGER.md` Layer 11 for the full
entries; not duplicated here.

**A real reproducibility bug, confirmed, not just noted**: the 5-channel
checker capacity on RS-SFT's own samples is reported as two different
numbers in two tracked files — `r2_checker_capacity_rssft.json` says
26.75%, `r1_r3_checker_capacity_2x2.json` says 26.69% — despite byte-
identical inputs, byte-identical parse/scored counts, and literally shared
scoring code (`r1_r3` imports `r2`'s own functions rather than
reimplementing them). Three controlled reruns this session (default hash
seed, then `PYTHONHASHSEED=0`, `=1`, `=2` pinned explicitly) reproduced
26.75% under seed 0 and 26.69% under seeds 1 and 2 — **confirmed: this
project's validator pipeline is PYTHONHASHSEED-sensitive**, somewhere in
`validator.validate()`'s thermodynamic/competing-phase scoring (not in
`mp_formula_set`'s membership test, which doesn't depend on set iteration
order; the exact line was not pinned down — out of scope for this pass, and
`validator.py` is Gowtham's file to edit). The effect is small here (±0.06
percentage points of capacity) but is a correctness concern for ANY future
rerun of validator-backed scoring intended to be bit-reproducible: pin
`PYTHONHASHSEED=0` (or any fixed value) before launching such a job.
Ledger now quotes 26.75% for this cell (matching the separately-tracked
`r2_checker_capacity_rssft.json`), with the finding and both files' values
recorded in `L11.capacity_2x2`.

**Gate G1's "0.59 to 0.68 median Spearman across 12 models" located**:
`results/phase17/instrument_check.json`, field
`marginal_estimator.per_model` — min 0.5913 (phase12_ckpt200), max 0.6761
(rs_sft), all 12 checkpoints clear the pre-registered ≥0.5 bar. Recorded
in `L11.gate_g1`.

**CLAUDE.md**: not edited by this session (standing rule — a separate chat
session owns that file). The findings write-up, finding-22 fragility
caveat, and "Phase 17 closed" status change regular Claude asked for are
drafted in full, ready to paste, in
`docs/phases/PHASE17_CLAUDE_MD_PATCH.md`.

**Phase 17 is closed.** Blocks 0, A, B, C, G1 and R ran. Blocks D, E, F were
not run — nothing later in Phase 17 depended on them. Block C stayed
infeasible (6/385 target-steps converge; raw mass differs by 10+ orders of
magnitude within a target) and its conditional rerun was correctly never
triggered (the tail-calibration check shows the tails are NOT calibrated).
Skipped by explicit instruction, not run: a true fresh-seed Phase 12
replication (the only item that would have cost GPU/generation time) and a
split-trace-style control for Block B3 (optional; the pre-registration
brief already tells the paper's writer not to report B3).
