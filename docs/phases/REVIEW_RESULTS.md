# Review results

Dated entries. These are CPU-only spot-checks requested after Phase 17's
close (see `docs/phases/PHASE17_RESULTS.md`, CLAUDE.md finding 34) — not a
new phase, no training, new files only under `results/review/`.
`prereg_commit` (where applicable, reused from the Phase 17 scripts these
build on): `790d0852f11d62d442fc299f8c277840b9d8a16d`.

## 2026-11-01 — Two CPU checks: reading rules (written before running)

### Check 1: sample-level ASTRAL rule-picked share

**Reading rule, pre-registered before running.** From
`results/astral_gen_n32_*.json`, compute the share of **ALL** generated
samples (parse failures / no-precursor samples count as non-matches, not
excluded from the denominator) whose precursor set equals the target's
ASTRAL rule-picked ("predicted") set, using the same normalisation
`precset_of` and `rc_price_equation.py`'s `canon()` both use
(`validator.SynthesisValidator._normalize_formula` ==
`pymatgen.Composition(formula).reduced_formula` — the two functions are
identical, so there is no canonicalisation discrepancy to chase here).
Models: Base, Corpus SFT, RS-SFT, GDPO-300, Phase12@300, E1@300, E3, E3c.
Scopes: the full 35 ASTRAL targets, and the clean 22 (35 minus the 13
targets in `results/phase16/astral_clean_subset.json`'s
`overlapping_targets`). CI: target-clustered bootstrap (resample
*targets*, not samples, with replacement; 10,000 reps; seed 20261016 —
the project's standing bootstrap seed). Paired difference vs each model's
init (Corpus SFT, RS-SFT, E3, E3c vs Base; GDPO-300 vs Corpus SFT;
Phase12@300, E1@300 vs RS-SFT): same resampled target list applied to
both model and init within each replicate, so common per-target
sampling noise cancels in the diff rather than inflating the CI (same
logic as `block_a_paired_bootstrap_phase12_vs_rssft.json`, at the
target-cluster level instead of the within-target position level).

`results/phase17/rc_price_equation.json`'s own
`observed_shift.matches_astral_rule_picked` field implies RS-SFT 3.82% →
Phase12@300 3.31%, and RS-SFT 3.82% → E1@300 5.11% — but that file's
`shift_from_astral_files()` excludes `PARSE_FAILURE`/no-precursor samples
from the denominator, which is NOT the literal "share of ALL samples"
this check asks for. **Confirmed** if this check's ALL-samples numbers
match those three figures within a few tenths of a percentage point once
the parse-failure-exclusion difference is accounted for and stated.
**Corrected** if they diverge for any other, unexplained reason.

Output: `results/review/rulepicked_sample_share.json`.

### Check 2: robustness of B1's split-trace slope c, restricted to the calibrated probability range

**Reading rule, pre-registered before running.** Refit
`results/phase17/b1_split_trace_control_corrected.json`'s regression
(split traces 1–4 vs 5–8, within-target fixed-effects OLS, cluster
bootstrap by target, seed 20261016 — `research/phase17_b1_signature_regression.py`'s
`bootstrap_step`/`within_target_fe_ols`, reused unchanged) restricted to
`(target, recipe)` rows whose **FULL** (non-split, all 8 traces)
before-model estimate `pi~_before` ≥ 1e-2 — the range
`results/phase17/b1_tail_calibration.json` shows is roughly calibrated
(`observed_over_expected` in [0.594, 2.23] at and above the `[1e-2,1e-1)`
bin, vs up to ~1.7 million× below it). The filter uses the full K=8-trace
estimator's renormalised `pi~_before` (`logp_universe.jsonl` via
`pi_tilde_before()`), not the split-half `pi~_A` the regression's own
regressor is built from, since that is what the tail-calibration file
itself bins.

**The contrast stands if S1–S4 (fine-tuning: Corpus SFT, RS-SFT, E3, E3c)
have `c` intervals entirely below 0, and S5 and S7 (practice: GDPO-300,
E1@300) have intervals that include 0.** Report `n` per step, both
before and after the filter. S6 (Phase 12) is reported but not part of
this pass/fail rule — it is already known to carry a small, real,
CI-excludes-0 residual (finding 34) that this rule does not adjudicate
either way.

Output: `results/review/c_calibrated_range.json`.

### Results

**Check 1: confirmed.** Using the literal ALL-samples denominator (parse
failures counted as non-matches), full-35 shares are RS-SFT **3.661%**,
Phase12@300 **3.214%**, E1@300 **5.000%** — within a few tenths of a
point of `rc_price_equation.json`'s implied 3.82% / 3.31% / 5.11%, in the
expected direction (slightly lower everywhere, because parse failures
that file's denominator excludes are counted here) and with the same
sign and ordering: RS-SFT → Phase12@300 falls, RS-SFT → E1@300 rises.
Clean-22 shares and the target-clustered bootstrap CIs and paired
diffs vs init are in `results/review/rulepicked_sample_share.json`
(e.g. E3 vs Base, clean 22: boot median diff +3.125 points, 95% CI
[0.71, 6.25] — full detail in the file, not reproduced here). Nothing
in this check required correcting rc_price_equation.json's numbers.

**Check 2: the reading rule FAILS, and not narrowly — the sign flips.**
Restricted to the calibrated probability range (`pi~_before >= 1e-2`),
every one of the seven steps' `c_split` is **positive**, not negative or
near-zero:

| step | before→after | c_split (calibrated) | 95% CI | n in fit (targets) |
|---|---|---|---|---|
| S1 | base→corpus_sft | **+0.636** | [0.498, 1.007] | 36 (12) |
| S2 | base→rs_sft | **+0.809** | [0.691, 0.945] | 97 (31) |
| S3 | base→e3 | **+0.691** | [0.534, 0.821] | 97 (31) |
| S4 | base→e3c | **+0.721** | [0.475, 0.907] | 97 (31) |
| S5 | corpus_sft→gdpo300 | **+0.784** | [0.022, 1.400] | 32 (12) |
| S6 | rs_sft→phase12_ckpt300 | **+0.752** | [0.429, 1.035] | 79 (26) |
| S7 | rs_sft→e1_ckpt300 | **+0.988** | [0.679, 1.225] | 79 (26) |

Every fine-tuning step (S1–S4), which the full (uncalibrated) range read
as confidently *negative* (−0.363 to −0.224, finding 34), is confidently
**positive** here. S6 and S7, meant to include 0 under the reading rule,
are instead confidently positive too; only S5's lower CI bound (0.022)
sits near 0. `contrast_stands = false`.

**Read with the low-power caveat, not dismissed.** The `>=1e-2` filter
keeps only 8–19% of each step's split rows (most recipes live in the
miscalibrated tail by sheer diversity, not because the calibrated range
is empty), and `within_target_fe_ols` additionally drops any target with
fewer than 2 surviving rows. The regression that actually produced each
row above used as few as **36 observations over 12 targets** (S1) to
**97 over 31** (S2–S4) — far smaller than the full-range fit's hundreds
of targets. This does not make the sign flip noise (every CI above is
tight and clears, or nearly clears, 0 on its own), but it means this
check cannot certify the calibrated-range slope with the same power the
full-range result had.

**What this means for finding 34.** The "main surviving Phase 17 result"
(CLAUDE.md finding 34: fine-tuning's c_split confidently negative,
practice's c_split near 0) was computed over the FULL range of
`pi~_before`, dominated by recipes in the severely-miscalibrated tail —
exactly the range the tail-calibration check (`b1_tail_calibration.json`)
says the K=8-trace marginal estimator cannot be trusted in. Restricted
to the one range that check certifies as roughly calibrated, the
relationship is not merely "not negative" — it is **positive everywhere
tested**, for both fine-tuning and practice alike, which is a different
and arguably more coherent story (sharpening toward already-common
recipes) than the original "fine-tuning moves away from likely recipes,
practice doesn't" framing. **This check does not by itself replace
finding 34's main result — it is a single pre-registered robustness
probe, on a much smaller effective sample, and regular Claude should
decide how to fold it in** — but finding 34 as currently written in
CLAUDE.md should not be read as surviving unconditionally; the
full-range negative slope for S1–S4 is now a tail-range-specific result,
not a calibrated-range one.
