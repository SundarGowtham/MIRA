# Phase 15 summary — closed 2026-09-25

Phase 15 investigated the two largest precursor-class shifts GDPO produces
(bare-alkali-oxide vs carbonate, ammonium-phosphate vs H₃PO₄), validated them
against real literature (Lee et al. 2025) and real robot-lab data (ARROWS³),
pre-registered and ran two external verifier gates, and ran a policy-level
teacher-forcing dose-response study. **No further Phase 15 experiments are
planned.** Every result below is tagged **[C]** (pre-registered before the
data was seen/joined) or **[E]** (exploratory, found by looking). n, effect
size, and 95% CI are given wherever the source doc computed them; a **"does
not support"** line states what each result does *not* establish, so this
page cannot be skimmed into an overclaim.

Source docs: `docs/phases/PHASE15_DISTRIBUTIONAL.md` (Tasks 1, 2, 2b, 2c),
`docs/phases/PHASE15_EXTERNAL.md` (Task 3, pre-reg
`PHASE15_LEE_PREREG.md`), `docs/phases/PHASE15_ARROWS_INVENTORY.md` (Task
4a, data inventory only, no scoring), `docs/phases/PHASE15_ARROWS_RESULTS.md`
(Tasks 4c/4d/4e, pre-reg `PHASE15_ARROWS_PREREG.md`), `docs/phases/
PHASE15_DOSE_RESULTS.md` (Task 5, pre-reg `PHASE15_DOSE_PREREG.md`). Task 6
(Precursor Genome check) was cancelled before any data was downloaded —
recorded as future work only, no result to report.

---

## Task 1 — distributional analysis of ASTRAL n=32 generations (all [E])

Source: `docs/phases/PHASE15_DISTRIBUTIONAL.md` §Task 1.

| # | result | n | effect size / CI | does not support |
|---|---|---|---|---|
| 1.1 | Base and RS-SFT both hit 10/35 ASTRAL predicted sets, sharing only 7 targets | 35 targets | 10/35 each, 7 shared | that RS-SFT and base solve the *same* problems — they match on aggregate rate, not identity |
| 1.2 | 16/35 targets ever hit across 4 models; GDPO covers 14 | 35 targets | 16 union, 14/16 GDPO | broad generalization — per-target hit counts are 1–3 (noise-scale, see 1.3) |
| 1.3 | Most hit-count changes are noise; `NaSrBO3` (0/32→6/32 under GDPO) is not | 32 samples/target | P(0/32\|p=1/32)=0.36 | that any *other* single-target change is a real RL effect — this is the one exception identified |
| 1.4 | Bare-oxide/carbonate share shift dwarfs the ASTRAL-hit-rate shift: bare-oxide 8.6%→0.5%→25.1%→56.3% (base/SFT/RS-SFT/GDPO); carbonate 74.7%→98.5%→48.0%→19.1%; predicted-set share flat 2–4% | 32×35 samples/model | descriptive, no CI | that this shift is chemically beneficial, or that it is captured by the headline ASTRAL hit-rate metric at all (Task 3 shows it is a shift *away* from real practice) |
| 1.5 | Ammonium-phosphate share on phosphate targets: 23.8%→82.6%→4.9%→2.1% (base/SFT/RS-SFT/GDPO) | subset, phosphate targets | descriptive | a chemistry-driven cause — traced to a balance-solver bug (Task 2) and a divergence from real practice (Task 3), not a genuine improvement |
| 1.6 | GDPO's `OTHER`-category output is dominated by bare-oxide 3-precursor routes; ~22–26% of (target, set) pairs across SFT/RS-SFT/GDPO never appear in base's 32 samples, but full SFT shows the same 22.0% rate | 32 samples/model | descriptive | that RL discovers genuinely novel precursor combinations outside base's support — full SFT shows the same "novelty" rate by this metric |
| 1.7 | Solution-space overlap: split-half noise floor shows RS-SFT→GDPO Jaccard (0.525) is indistinguishable from GDPO's own self-noise ([0.523, 0.588]); every base-involving pair sits clearly below both self-noise floors | 35 targets, 200 resamples/target | Jaccard 0.52–0.68 (self), 0.328–0.525 (cross) | that GDPO adds qualitatively new solutions beyond RS-SFT's support — RL redistributes mass within a support that was already set at the base→RS-SFT step |
| 1.8 | Hit-definition resolution: full SFT is 0/35 exact match, 1/35 only under a superset-inclusive count (`LiZnBO3`); base-vs-SFT collapse significant either way | 35 targets | exact McNemar p=0.0020 (b=10,c=0); superset-inclusive p=0.0039 (b=9,c=0) | that the hit-definition choice changes *whether* the collapse is real — only which specific number is quoted |

## Task 2 — which validator channel penalizes carbonates? (all [E])

Source: `docs/phases/PHASE15_DISTRIBUTIONAL.md` §Task 2.

| # | result | n | effect size / CI | does not support |
|---|---|---|---|---|
| 2.1 | Re-derived pass@0.9: bare-oxide-only vs carbonate-only, close to but not identical to Task 1's claimed 98.9%/78.2% | bare n=95, carb n=825 | 95.8% vs 75.9% | an exact reproduction of the original claim — residual gap attributed to a stated reconstruction limitation (single-op, air-default) |
| 2.2 | Naive carbonate penalty is mostly the already-known ammonium/balance-solver bug: 130/825 (15.8%) of "carbonate-only" samples also contain an ammonium precursor. Removing them: pass@0.9 gap closes ~20pt→6.7pt; `stoichiometry` gap collapses +0.155→+0.007 | carb n=825→695 (ammonium removed) | 89.1% vs 95.8% pass@0.9; stoichiometry gap +0.007 | that `stoichiometry` is an independent carbonate-vs-bare-oxide discriminator — it is almost entirely the ammonium confound |
| 2.3 | Ammonium-phosphate vs H₃PO₄ on phosphate targets confirms the Phase 13 balance-solver bug directly on base's own spontaneous generations | ammonium n=137, H₃PO₄ n=215 | pass@0.9 9.5% vs 99.1%; stoichiometry gap −0.905 | that this is a new, distinct mechanism — it is the same NH₃-without-N₂ balance-solver gap Phase 13 diagnosed |
| 2.4 | After removing the ammonium confound, a real but much smaller residual carbonate-specific effect remains via `thermodynamic_favorable` | carb n=695 (cleaned) | gap +0.046, carbonate mean 0.954 | a single unified "carbonate penalty" mechanism — there are two distinct mechanisms, one bug-driven, one real-and-small |

## Task 2b — three checks on the bare-oxide/carbonate mechanism (all [E])

Source: `docs/phases/PHASE15_DISTRIBUTIONAL.md` §Task 2b.

| # | result | n | effect size / CI | does not support |
|---|---|---|---|---|
| 2b.1 | **Retraction**: the original RS-SFT-step comparison (8.6% vs 6.4%) crossed prompt distributions and was flawed; corrected alkali-source-share metric (triples, not samples) | base-on-ASTRAL n=1166 triples; RS-SFT train n=66; RS-SFT-on-ASTRAL n=1134; GDPO-on-ASTRAL n=1151 | bare-oxide 9.1% / 28.8% / 25.8% / 56.9% | the original "fine-tuning amplifies beyond training-set representation" claim — RS-SFT's inference-time rate (25.8%) sits almost exactly at its own training-set rate (28.8%); no amplification there |
| 2b.2 | Real amplification is at the GDPO step: within 50 mixed training groups, `thermodynamic_favorable` is the only channel with a meaningful within-group z-gap | 50 mixed groups (of 841 total), 49 gradeable | z-gap 1.292 SD (raw gap 0.250, within-group std 0.135); other channels ≤0.099 | a claim that this gap *grows* over training — retracted; z-gap is positive in every one of 5 step-buckets (1.75→0.85→1.94→2.14→2.07) but bucket n (4–17) is too small to establish a trend either way |
| 2b.3 | `thermodynamic_favorable` ungradeable rate differs slightly by precursor class | bare n=655, carb n=576 | 1.2% vs 2.6% ungradeable | a primary mechanism — small, direction-consistent addition to 2b.2, not the driver |

## Task 2c — what does `thermodynamic_favorable` actually compute? ([E], code-verification + controlled test)

Source: `docs/phases/PHASE15_DISTRIBUTIONAL.md` §Task 2c.

| # | result | n | effect size / CI | does not support |
|---|---|---|---|---|
| 2c.1 | Confirmed by direct code read: it is a Gibbs-corrected ΔG_rxn(T) (Bartel SISSO descriptor + NIST-JANAF gas ΔfG°(T) for CO₂/H₂O/O₂/N₂/NH₃), always the production codepath | n/a (code inspection) | n/a | a claim that this was previously unknown or in doubt — it settles which codepath production actually uses |
| 2c.2 | The bare-oxide thermodynamic preference **survives** a finite-temperature correction in every computable pair | 5 matched pairs (4 computable) | naive-0K gap −0.222 to −0.365 eV/atom → Gibbs-corrected −0.187 to −0.317 eV/atom (shrinks 11–25%) | that the entropy correction eliminates or reverses the preference — it is real, physically grounded, and robust, just somewhat smaller than the naive number |

## Task 3 — Lee et al. 2025 literature validation

Source: `docs/phases/PHASE15_EXTERNAL.md`; pre-reg `docs/phases/PHASE15_LEE_PREREG.md`.

| # | result | tag | n | effect size / CI | does not support |
|---|---|---|---|---|---|
| 3.1 | Primary endpoint: real practice strongly avoids bare alkali oxides | [C] | 13,849 (record, alkali-element) triples | bare-oxide 2.5% (341) vs carbonate 60.0% (8,311); both pre-registered predictions HOLD | anything about whether the avoidance is chemically *justified* — that's 3.2 |
| 3.2 | Secondary endpoint: bare-oxide impurity rate is NOT higher than carbonate's among published syntheses | [C] | bare n=338, carb n=7,616 (target-matched: bare n=141, carb n=1,295) | pooled diff −0.3%, CI [−3.0%,+2.3%]; matched diff +0.5%, CI [−3.9%,+5.0%] | that bare-oxide routes succeed equally often in *unselected* (unpublished) attempts — literature is outcome-selected; pre-registered prediction (bare-oxide impurity ≥ carbonate) explicitly NOT supported |
| 3.3 | Phosphorus source: ammonium phosphate is real chemists' overwhelming choice (10:1 over H₃PO₄+P₂O₅), opposite the direction RS-SFT/GDPO moved | [E] | ammonium n=2,564; H₃PO₄ n=134; P₂O₅ n=129; other n=2,092 | impurity 2.8% / 2.2% / 1.6% / 2.0% | a second, independent mechanism — very likely the same balance-solver bug as Task 2 |
| 3.4 | 20/35 ASTRAL targets appear in Lee et al.; none of their literature routes use a bare alkali oxide | [E] | 20 targets | 0/20 use bare oxide | a corpus-wide generalization beyond this smaller, target-matched sample (though it is the most directly relevant evidence available) |

## Task 4 — ARROWS³ robot-lab verifier gate

Source: `docs/phases/PHASE15_ARROWS_RESULTS.md`; pre-reg `docs/phases/PHASE15_ARROWS_PREREG.md`;
data inventory `docs/phases/PHASE15_ARROWS_INVENTORY.md` (no scoring, not itself a tagged result).

| # | result | tag | n | effect size / CI | does not support |
|---|---|---|---|---|---|
| 4.1 | Primary endpoint (YBCO): **neither verifier beats the best trivial baseline** | [C] | 1,915 pairs (\|Δ\|≥5) | validator 0.616, diff +0.033 CI [−0.043,+0.112] (fails); comparator 0.474 (below chance), diff −0.109 CI [−0.226,+0.007] (fails); best baseline carbonate-free 0.583 | that either verifier adds signal beyond a two-line carbonate-free heuristic |
| 4.2 | Per-channel diagnosis: `thermodynamic_favorable` is the best single channel, but the 5-channel vote dilutes it | [E] | n≈1,896–1,915 | thermo 71.1%, amount_accuracy 68.6%, stoichiometry 53.4% (near chance), 2 channels dead (100% tied); 5-channel vote only 61.6% | that aggregate GDPO-vote scoring reliably preserves a strong individual channel's signal |
| 4.3 | Secondary replications (LTOPO, NTMO) also fail to beat baseline | [C] | LTOPO n=77; NTMO n=67 | LTOPO validator diff −0.084 CI[−0.291,+0.171]; comparator −0.019 CI[−0.315,+0.254]. NTMO validator −0.007 CI[−0.222,+0.197]; comparator −0.045 CI[−0.308,+0.212] | strong evidence on its own — both targets are underpowered (small n), consistent with but not independently decisive for the null |
| 4.4 | Ba-source controlled test: verifier's preferred Ba source does NOT yield more target phase | [C] | 10 matched groups; 18 informative (group,T) cells | mean diff +5.31 wt%, 95% CI [−3.36,+14.60] (includes 0); sign test 9/9 split (exact chance) | that there is "no effect" — n=10 groups is underpowered; this is inconclusive, not a strong null |
| 4.5 | Threshold robustness: YBCO stable once uninformative (identical-outcome) pairs are excluded; LTOPO is NOT stable | [E] | YBCO n=1,671–2,124 across 3 thresholds; LTOPO n=58–90 | YBCO GDPO-vote 0.614/0.616/0.630, comparator 0.485/0.474/0.475 (stable); LTOPO GDPO-vote 0.533→0.416, a 12-pt swing | a claim that verifier behavior is threshold-robust in general — true for YBCO only, at this n |
| 4.6 | Comparator's poor showing is driven by its gates, not its scored channels — and the gates actively hurt | [E] | gate-decided n=943; channel-decided n=704 (of 4,390 YBCO pairs) | gate-decided agreement 0.445 (below chance); channel-decided 0.513 (barely above chance); pooled 0.474 | a claim that the comparator's scored channels are the weak point — they are merely mediocre; the gates are actively harmful |
| 4.7 | Gate-failure audit: `precursors_exist` failures are 100% a Materials Project coverage gap (`Y2(CO3)3`), not unsound chemistry | [E] | 16/47 YBCO sets (34%) contain `Y2(CO3)3` | at matched T, `Y2(CO3)3` yield ≥ other-Y-source yield (e.g. 900°C: 66.2% n=16 vs 62.1% n=31) | a claim that gate failures indicate bad chemistry — recorded finding: `precursors_exist` tests MP coverage, not chemical existence; notation/parsing failures are zero instances |
| 4.8 | "Avoid carbonates" is itself a real, non-trivial signal on this dataset | [E] | YBCO n=1,915 | agreement 0.583, diff from chance +0.083, 95% CI [+0.010,+0.154] | a claim that this baseline is a weak strawman — it is real; neither verifier improves on it |
| 4.9 | `thermodynamic_favorable`'s apparent quality is ~88% redundant with the carbonate dimension | [E] | decisive pairs n=521; both-decisive n=285; carbonate-free-tie subset n=1,112 | thermo 0.712 vs carbonate-free-on-same-pairs 0.662 (edges it by ~5pt where it has an opinion); same pick 87.7% of the time when both decisive; on the carbonate-tie subset, thermo agreement 0.522, CI [0.495,0.557] (chance) | a claim that `thermodynamic_favorable` carries signal beyond the carbonate/bare-oxide dimension — none established once that dimension is held constant |
| 4.10 | Gradeability-tag handling in the GDPO-vote rule verified correct by direct code inspection | [E] | n/a | n/a | a claim that this was a suspected bug — it was checked and confirmed correct, not a defect |

## Task 5 — dose-response via teacher-forcing

Source: `docs/phases/PHASE15_DOSE_RESULTS.md`; pre-reg `docs/phases/PHASE15_DOSE_PREREG.md`;
robustness round in the same file, script `research/distributional/dose_response_robustness.py`.

| # | result | tag | n | effect size / CI | does not support |
|---|---|---|---|---|---|
| 5.1 | Q1 Leash: **degenerate as specified**, neither confirmed nor falsified — ε=−20 was a specification error in the instructions doc | [C] | 75 routes | total-logp scale: 75/75 below ε (n_above=0); [E] per-token scale: 75/75 above ε (n_below=0) | any claim about a leash effect existing or not — the test could not be run as specified |
| 5.2 | Q2 Quality alignment: uninformative, not a clean negative | [C] | 75 routes (49 pure/26 impure) | `phase_pure` coefficient −0.133, 95% CI [−2.887,+2.585] | a claim that GDPO does or does not preferentially raise phase-pure routes' log p — the interval is too wide for either |
| 5.3 | Q3 Top-5 (rescoped to ARROWS): supported | [C] | 3 targets | 3/3 hit (≥2/3 threshold) | strong evidence beyond these 3 targets — small, target-level n |
| 5.4 | Q4 Policy-level carbonate test: both predictions supported, and robust | [C]+[E] robustness | 75 routes (19 carb/56 non-carb) | `carbonate_status` −1.979, CI [−3.554,−0.352]; robust to target FE (−2.348, CI[−3.893,−0.808]), a broader CO₃-substring definition with target FE (−2.337, CI[−3.623,−1.046]), and a length covariate (more negative once controlled, −2.797, CI[−3.657,−1.915]); `target_wt_pct` +0.020, CI[−0.006,+0.043] | a claim that yield has "no effect" — only "no detectable effect"; CI upper bound does not exclude a moderate one. Does not establish the causal path — consistent with, not proof of, inheritance from `thermodynamic_favorable` |
| 5.5 | ASTRAL prompt-only rank/logp: literal first-token test uninformative for 3/4 targets (shared leading cation token); [E, beyond the ask] divergence-point check shows the traditional continuation at rank 1 under every checkpoint | [E] | 4 targets, 3 checkpoints | e.g. NaSrBO₃: traditional rank 1 (logp −0.007 to −0.410) vs predicted rank 2–4 (logp −3.660 to −5.257) at the actual fork | a claim in tension with Phase 12/18's aggregate predicted-set gains — this is a single-token, empty-`<think>` local read, not the same statistic as a full sampled generation (see the dose-results doc's limitation section) |
| 5.6 | Descriptive: mean total log p by checkpoint | [E] | 75 routes | base −96.19, RS-SFT −85.61, GDPO-300 −81.81, full SFT (ref only) −50.07 | any causal or quality claim on its own — purely descriptive |

**Limitation applying to all of Task 5** (`docs/phases/PHASE15_DOSE_RESULTS.md`,
final section): every number above conditions on an **empty `<think>` block**.
Relative (Δ) comparisons across checkpoints are valid since the context is
identical; absolute ranks/log p describe the model's answer distribution with
reasoning skipped, not real generation behavior. [E, untested] GDPO's
rank-1 preference for the carbonate continuation in 5.5 contrasts with its
19.1% carbonate share in real sampled ASTRAL generations (Task 1, finding
1.4) — consistent with, but not established by, the shift being expressed
partly through the reasoning trace this probe cannot see.

---

## `CLAUDE.md` additions — APPLIED 2026-09-25

**Reconciliation performed first, before applying anything.** The
committed `CLAUDE.md` (verified in sync with `origin/main`, no divergence
in git history, stash, or worktrees — searched exhaustively) ended at
finding 19; it did not yet contain findings 20 (Phase 12 KL-estimator
blowup) or 21 (Phase 12 pre-launch diagnostics) referenced from a separate
local copy, nor any record of Phase 12's actual 14/35 AMPLIFIED result or
Phase 13's stopping decision. Findings 20 and 21 were drafted here from
`docs/phases/PHASE12_RESULTS.md` (which already documents both in full,
and itself cross-references "finding 20" for the KL blowup, confirming
that numbering). **Two gaps were found and filled**: neither the Phase 12
14/35 result nor the Phase 13 stopping decision had a finding number
anywhere — both are now finding 22 and finding 23 respectively.

The six Phase 15 findings below were then renumbered 24–29 to continue
after 23. The seventh original draft ("Phase 15 is closed") was **moved
into the Journey section as item 10, not applied as a finding** — it is a
status/wrap-up statement, not a result. `CLAUDE.md` now also carries a
one-line `STATUS: COMPLETE` note at the top of the `## PHASE 12 — GDPO
FROM RS-SFT` section, pointing to finding 22, since that section's
forward-looking planning language was stale (the run it describes already
happened). **All of this has been written into `CLAUDE.md` and committed.**
The text below is kept as the record of what was drafted and applied.

### Finding 24 (applied) — Phase 15: the bare-oxide/carbonate shift is real, larger than the ASTRAL metric, and correctly computed but wrong-construct

GDPO's largest distributional effect on ASTRAL generations is invisible to
the headline hit-rate metric. Bare-alkali-oxide share climbs base 8.6% →
full SFT 0.5% → RS-SFT 25.1% → GDPO 56.3%, with carbonate share moving
inversely (74.7% → 98.5% → 48.0% → 19.1%), while ASTRAL-predicted-set share
stays flat at 2–4% throughout. The mechanism is now fully traced: within 50
GDPO training groups that pit a bare-oxide completion against a carbonate
one, `thermodynamic_favorable` is the only channel with a real within-group
z-gap (1.29 SD), and this reflects a genuinely-computed, Gibbs-corrected
ΔG(T) (Bartel descriptor + NIST-JANAF gas thermochemistry, CO₂ entropy
included) that favors bare oxides by 0.19–0.32 eV/atom and **survives a
finite-temperature correction** (shrinks 11–25%, never reverses, across
every computable one of 5 matched pairs). This is not a validator bug.

But real solid-state synthesis practice does the opposite: literature-scale
data (Lee et al. 2025, *Sci. Data* 12:1969, 80,806 records) shows chemists
choose carbonate over bare oxide 60.0% vs 2.5% of the time among
alkali-containing targets (n=13,849 triples) — a ~24x preference — and among
the 20/35 ASTRAL targets that appear in Lee's corpus, **zero** literature
routes use a bare alkali oxide at all. Critically, this avoidance is **not**
explained by worse published outcomes: bare-oxide-only and carbonate-only
syntheses show no significant impurity-rate difference (pooled diff −0.3%,
95% CI [−3.0%,+2.3%], n=338 vs 7,616; target-matched +0.5%, CI
[−3.9%,+5.0%], n=141 vs 1,295) — though outcome-selection in published
literature means this does not settle unselected-attempt success rates.
**GDPO is optimizing a real, correctly-computed thermodynamic quantity that
turns out not to be the construct that determines real synthesis success**
— kinetics, handling, and nucleation constraints the validator has no
channel for. Source: `docs/phases/PHASE15_DISTRIBUTIONAL.md`,
`docs/phases/PHASE15_EXTERNAL.md`.

### Finding 25 (applied) — Phase 15: two distinct precursor-shift mechanisms, only one a bug

GDPO's two largest precursor-class shifts on ASTRAL both move away from
real chemist practice, but for unrelated reasons that should not be
conflated. (1) **Ammonium-phosphate avoidance is a software bug**: the
balance solver never tries a candidate volatile set containing NH₃ without
N₂, so ammonium-salt routes fail `stoichiometry` catastrophically (gap
−0.905, pass@0.9 9.5% n=137 vs H₃PO₄'s 99.1% n=215) — confirmed directly on
base's own spontaneous ASTRAL generations, the broadest confirmation yet of
the Phase 13 diagnosis. Real chemists prefer ammonium phosphate 10:1 over
H₃PO₄+P₂O₅ (Lee et al., n=2,564 vs 263); RS-SFT/GDPO trained away from it
for a software reason, not a chemistry one (share 23.8%→82.6%→4.9%→2.1%,
base/SFT/RS-SFT/GDPO). (2) **Bare-oxide preference is not a bug** (see
finding 24) — a real, correctly-computed thermodynamic quantity that simply
isn't what determines synthesis success. Source: `docs/phases/
PHASE15_DISTRIBUTIONAL.md`, `docs/phases/PHASE15_EXTERNAL.md`.

### Finding 26 (applied) — Phase 15: RS-SFT does not amplify its own training-set bias; GDPO does

Correcting an earlier flawed measurement (which crossed prompt
distributions): RS-SFT's ASTRAL-inference-time bare-oxide share (25.8%,
n=1,134 triples) sits almost exactly at its own training set's share
(28.8%, n=66 triples, only 64/295 training targets alkali-relevant) — no
amplification between training composition and RS-SFT's own generation
behavior. **The real amplification happens at the GDPO step**: RS-SFT-on-
ASTRAL (25.8%) → GDPO-on-ASTRAL (56.9%) is a genuine ~2.2x further
increase, driven by the within-group `thermodynamic_favorable` reward
advantage (finding 24's z-gap of 1.29 SD), positive in every one of 5
step-buckets across training (too few groups per bucket, 4–17, to confirm a
growth trend). Solution-space analysis with a split-half noise-floor control
confirms the real distributional shift happens at base→RS-SFT (Jaccard
0.328–0.484, clearly below both models' self-noise floors of 0.52–0.68);
RS-SFT→GDPO's Jaccard (0.525) is statistically indistinguishable from GDPO's
own resampling noise — **GDPO redistributes probability mass within RS-SFT's
existing support, it does not add qualitatively new solutions.** Source:
`docs/phases/PHASE15_DISTRIBUTIONAL.md`.

### Finding 27 (applied) — Phase 15: two independently pre-registered external verifier gates, neither beats a trivial baseline

The validator and `core/comparator.py` were both tested against real,
independent data outside the training loop, pre-registered before scoring.

**ARROWS³ robot-lab data** (Szymanski et al. 2023, 4,870 real synthesis
outcome pairs, YBCO/LTOPO/NTMO): on the primary YBCO endpoint (n=1,915,
|Δ|≥5wt%), the validator's GDPO-vote rule (0.616 agreement) does not
significantly beat the best trivial baseline (carbonate-free, 0.583; diff
+0.033, 95% CI [−0.043,+0.112]), and the comparator (0.474) scores **below
chance** and below every baseline (diff −0.109, CI [−0.226,+0.007]).
Secondary replications on LTOPO/NTMO also fail to beat baseline (both
underpowered, n=67–77). A direct robot-lab Ba-source controlled test finds
no significant yield advantage for the verifier's preferred source (mean
diff +5.31 wt%, 95% CI [−3.36,+14.60], n=10 groups; sign test exactly 9/9).
Diagnosis: the comparator's gates (not its scored channels) decide most
pairs and decide them badly (gate-decided agreement 0.445, below chance,
n=943, vs channel-decided 0.513, n=704) — 16/47 (34%) of YBCO's
`precursors_exist` gate failures trace entirely to `Y2(CO3)3`'s absence
from Materials Project, a real, comparably-performing compound (**a
database coverage gap, not unsound chemistry**). The best single channel
(`thermodynamic_favorable`, 71.1% agreement alone) is 88% redundant with
the carbonate-free baseline on pairs where both are decisive, and shows no
signal beyond the carbonate dimension once it's held constant (agreement
0.522, CI [0.495,0.557], centered on chance). Note the baseline itself is
real: "avoid carbonates" beats chance significantly (diff +0.083, CI
[+0.010,+0.154]) — neither verifier improves on a two-line heuristic.

**ASTRAL-based ranker v2 gate** (finding 19, pre-Phase-15): failed its own
pre-registered bar (21/35=60.0% vs 24/35 required).

Combined: every external verifier evaluation this project has run — plain
validator (ASTRAL, ~50%), ranker v2 (ASTRAL, 60.0%, failed gate), validator
GDPO-vote (ARROWS³, 61.6%, failed gate), comparator (ASTRAL 47–57%, ARROWS³
47.4%, failed both gates) — has failed to clear a baseline-matched bar with
a CI that excludes zero. Source: `docs/phases/PHASE15_ARROWS_RESULTS.md`,
`docs/phases/PHASE15_ARROWS_INVENTORY.md`.

### Finding 28 (applied) — Phase 15: policy-level teacher-forcing confirms the carbonate shift is class-based and yield-blind, with caveats

Teacher-forcing exact log-probabilities (no sampling) of the precursor
segment through base/RS-SFT/GDPO-300/full-SFT on 75 real ARROWS³ routes,
pre-registered, found: (1) a leash test (is there a base-log-p floor below
which GDPO's effect vanishes) was **degenerate as specified** — ε=−20 was
miscalibrated for a ~51-token summed span (all 75 routes fall on one side)
— a specification error, not a null result. (2) A quality-alignment test
(does GDPO raise phase-pure routes' log p more than impure ones) was
**uninformative**, CI [−2.887,+2.585] too wide to distinguish an effect from
none. (3) At the first precursor-token position, base already assigns
non-trivial probability to each target's eventually-best route (3/3 ARROWS
targets). (4) **The headline result**: `Δlog p(GDPO−RS-SFT)` on real
measured-yield routes depends significantly on carbonate status
(coefficient −1.979, 95% CI [−3.554,−0.352], n=75, robust to target fixed
effects, a broader carbonate definition, and a length covariate — the raw
coefficient is if anything conservative) but shows **no detectable
dependence on the actual measured yield** (coefficient +0.020, CI
[−0.006,+0.043]) once carbonate status is in the model — the CI does not
however exclude a moderate yield effect. This is a policy-level (not
validator-score) signature consistent with GDPO inheriting
`thermodynamic_favorable`'s carbonate preference; it does not itself
establish the causal path. **Limitation**: every number above conditions on
an empty `<think>` block, so absolute ranks/log p describe the answer
distribution with reasoning skipped, not real generation behavior — an
exploratory follow-up found GDPO's local, empty-think preference still
favors the carbonate continuation at rank 1 on 3 held-out targets, which
contrasts with its 19.1% carbonate share in real *sampled* generations
[E, untested hypothesis: the shift may be expressed partly through the
reasoning trace this probe cannot see]. Source: `docs/phases/
PHASE15_DOSE_RESULTS.md`.

### Finding 29 (applied) — Phase 15: `precursors_exist` measures database coverage, not chemical soundness

Auditing every ARROWS³ gate failure by category found zero notation/parsing
failures (category b is empty) and that 100% of YBCO's 16 `precursors_exist`
failures trace to a single precursor, `Y2(CO3)3`, confirmed absent from
Materials Project's formula set by direct membership check — while all 29
other distinct precursor formulas across all three targets (including
parenthetical forms, peroxides, and complex ammonium-phosphate notations)
parse cleanly and are present in MP. `Y2(CO3)3` appears in 16/47 (34%) of
YBCO's real precursor sets and, at matched temperature, performs
comparably to or slightly better than alternative Y-sources (e.g. 900°C:
66.2% n=16 vs 62.1% n=31) — it is a real, working, published-robotic-
synthesis compound. Materials Project is a DFT-computed database, not an
exhaustive chemistry registry; a third of this dataset's precursor space is
invisibly excluded from `precursors_exist`-gated scoring for a database-
completeness reason unrelated to real performance. Source: `docs/phases/
PHASE15_ARROWS_RESULTS.md`.

### Moved to Journey, not applied as a finding — Phase 15 is closed

This seventh original draft was a status/wrap-up statement, not a result,
so it was not given a finding number. It is now **Journey item 10** in
`CLAUDE.md`: six of seven planned Phase 15 tasks completed (Task 6,
Precursor Genome check, cancelled before any data was downloaded — future
work only); no further Phase 15 experiments are planned; full
result-by-result breakdown pointed to this file. The project's
structural-limitation framing gains a second, independent confirmation
this phase: the validator's dominant learnable signal
(`thermodynamic_favorable`) is a real, physically-correct quantity that
diverges from real-world synthesis success — the Raccuglia et al.-style
"verifier built only from successes" limitation named in `CLAUDE.md` is now
directly demonstrated with external data, not just argued from first
principles.
