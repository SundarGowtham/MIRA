# Presentation results — external validation against ASTRAL robot data

*Per `misc/URGENT_PRESENTATION_PREP.md`. ALL STEPS (1-5) DONE as of
2026-09-01 02:28 UTC. Read the MASTER SUMMARY section for the talk-ready
version; the step-by-step sections below it have full detail and caveats.*

---

## Step 1 — Validator vs. robot-measured phase purity — DONE (2026-09-01)

**The money result.** Built two `PredictedRoute`s (traditional / predicted
precursors) per ASTRAL target at that route's own `best_T`, air, single
calcination, scored unmodified `validator.py` (Arm A) and `core/ranker.py`
(Arm B) on all 70 routes.

| | Spearman ρ (n=70) | p | agreement rate | chance |
|---|---|---|---|---|
| **Validator (Arm A)** | 0.228 | 0.058 (n.s.) | **17/34 = 50.0%** | 50% |
| **Ranker (Arm B)** | 0.111 | 0.360 (n.s.) | 20/32 = 62.5% (n.s., n this small) | 50% |

Neither scorer significantly predicts which of two real, published routes the
robot found to work better. The validator's agreement rate landing at *exactly*
50.0% is a clean, external, non-project-internal confirmation of the Central
Diagnosis: **the verifier measures validity, not quality.**

Full table (validator scores) and per-route breakdowns:
`misc/astral_validator_correlation.json`. Scatter plot:
`manifold_visualization/figures/astral_correlation.png`.

**Caveats, stated not hidden:**
- ASTRAL data gives precursor species, not molar ratios — every precursor got
  a placeholder amount 1.0, applied identically to both routes per target, so
  `amount_accuracy` specifically carries no real signal, but this doesn't bias
  the trad-vs-pred comparison.
- Ranker's `temperature_economy`/`step_economy` need a literature-T/n_ops
  reference that doesn't apply to already-optimal experimental routes, so
  they were left ungraded for this comparison; the ranker number above
  reflects only `precursor_availability`, `volatility_risk`,
  `driving_force_margin`.

---

## Step 2 — Na2Al2B2O7 case study — DONE (2026-09-01)

Target 30 in the ASTRAL set, in MIRA's pass@k val set at **pass@1=1.000**.
`misc/passk_n200.json` stored only rewards, not completions — regenerated 8
closed-book SFT samples (same settings as `probe_passk.py`).

**7/8 samples matched the traditional precursor set (Na2CO3+Al2O3+B2O3)
exactly**, 0/8 matched predicted, 1/8 novel. Mean validator reward: **0.982**
(near-perfect), at temperatures 850–1200°C — all above ASTRAL's tested range
(600/700°C).

ASTRAL ground truth for that exact traditional precursor set: **0.00 phase
fraction at 600°C, 0.46 at 700°C** (the best they tested) — worse at every
tested temperature than the predicted route's 0.52/0.60. The model converges
overwhelmingly on the precursor set the robot found to be the *worse* choice,
while the validator can't tell the difference (near-perfect regardless).

Full table + raw completions: `misc/case_study_na2al2b2o7.md`.

---

## Step 3 — Corpus coverage of the winning precursors — DONE (2026-09-01)

Counted occurrences in `data/raw/synthesis_clean.json` (17.6k Kononova routes,
1,408 distinct precursor formulas, 50,371 total mentions).

| | mean corpus frequency |
|---|---|
| **novel (ASTRAL-winning) precursors** (LiPO3, LiBO2, LiNbO3, Li2TiO3, KPO3, K3PO4, KNbO3, NaBO2, NaPO3) | **2.6** |
| **traditional precursors** (Li2CO3, B2O3, BaO, NH4H2PO4, K2CO3, Na2CO3) | **674.8** |

Per-precursor: LiNbO3, KPO3, NaBO2 literally **0** occurrences in the corpus;
Li2CO3 alone appears **1,654** times. A ~260x disparity in mean frequency —
the anthropogenic bias quantified directly on MIRA's own training data, not
inferred.

`misc/corpus_coverage.json`, `manifold_visualization/figures/corpus_coverage.png`
(corpus frequency of each route's precursor set vs. measured purity, 70 points).

---

## Step 5 — Figures for the talk — DONE (2026-09-01)

All 5 in `manifold_visualization/figures/`:
1. `astral_correlation.png` (step 1)
2. `corpus_coverage.png` (step 3)
3. `passk_gap.png` — pass@k base/SFT/GDPO-300 + gap-vs-k inset. Gap: **+2.41 pts
   at k=1 → +1.00 pt at k=16** — matches finding 4 exactly.
4. `reward_capacity.png` — per-channel z-variance, validator (baseline
   condition, `misc/hardening.json`, 10 channels, capacity 16.41%) vs. ranker
   (baseline condition, `misc/ranker_capacity_probe.json` — the pre-fix
   48.7% run, so this is "6 channels, 5 live" as specified) side by side.
5. `intervention_ladder.png` — capacity across baseline(16.4)/temp_ceiling(17.7)/
   inventory(16.4)/atmosphere(16.3)/combined(16.1)/low_temp(23.8)/ranker(48.7),
   40% line drawn.

---

## MASTER SUMMARY — every number for the talk, in order (2026-09-01, updated as Step 4 lands)

**1. Arm A result.** pass@k, n=200 targets, k=16, closed-book, SFT vs GDPO-300:
paired diff **+1.0 pt** (188/200 ties). Gap **shrinks with k: +2.4 at k=1 →
+1.0 at k=16** (`passk_gap.png`), McNemar p=0.77 on discordant targets. The
textbook **sharpening, not expansion** signature — 300 steps of GDPO did not
expand the capability boundary.

**2. Diagnosis.** Reward capacity **14%** (z-variance 1.43/10 channels).
**8 of 10 channels have zero within-group variance** — the model passes them
>99% of the time regardless of policy, so there's no gradient there to
exploit (`reward_capacity.png`, left panel).

**3. Five interventions, all failed to raise capacity.** baseline 16.4% →
temp_ceiling 17.7%, inventory 16.4%, atmosphere 16.3%, combined 16.1%.
**Inventory restriction actively reduced route diversity** (2.17 → 1.65
distinct precursor sets/group of 8) — constraining the action space removed
the alternatives the model had been varying over, the opposite of the
prediction.

**4. The reward hack.** Median **+200°C** above literature temperature,
unconstrained. Under an explicit hard ceiling: **100% compliance**, mean max-T
993°C — confirmed by intervention (not correlation) that the model can
control T precisely and only inflates it because the validator scores ΔG at
the model's own reported T.

**5. NEW — external validation (Step 1, the money result).** Validator score
vs. ASTRAL robot-measured phase purity, 70 real published routes: Spearman
ρ=0.228 (p=0.058, not significant), and the validator's route-ranking
**agreement rate with the robot is exactly 50.0%** — chance. `astral_correlation.png`.
This is the Central Diagnosis confirmed against experimental ground truth
external to the project, not just internal reward-capacity diagnostics.

**6. NEW — the case study.** Na2Al2B2O7 (pass@1=1.000 in MIRA's own held-out
set): the SFT model proposes the traditional precursor set (Na2CO3+Al2O3+B2O3)
in **7/8 samples**, scoring near-perfect on the validator (mean 0.982) — while
the robot measured that exact precursor set at **0.00 phase purity at 600°C,
0.46 at 700°C (its best tested case)**, worse than the predicted route
(Al2O3+NaBO2) at every temperature ASTRAL tested (0.52/0.60).
`misc/case_study_na2al2b2o7.md`.

**7. NEW — corpus coverage: the best precursors are absent from the training
corpus, and the model never finds them either.** ASTRAL's winning precursors
(LiPO3, LiBO2, LiNbO3, KPO3, NaBO2, ...) average **2.6 occurrences** in the
17.6k-route Kononova corpus (MIRA's SFT data) vs. **674.8** for traditional
precursors — a ~260x disparity, several (LiNbO3, KPO3, NaBO2) appearing
**zero** times (`corpus_coverage.png`). Confirmed directly on the model itself
(Step 4, 35 targets x 8 samples = 280 generations): **the model proposed the
ASTRAL-predicted (better) precursor set for 0 of 35 targets, in 0 of 280
generations** — not once. It proposed the traditional set for 30/35 targets
(mean validator reward on those matches: 0.930, n=139 samples). The echo
chamber isn't an inference from corpus statistics — it's measured directly on
the trained policy. The model also runs consistently hot: mean proposed max-T
**964.6°C** vs. ASTRAL's own best-condition mean of 762.9°C (traditional) /
745.7°C (predicted) — the reward hack (item 4) reproduced on this external
target set too.

**8. Next.** Rebuild the verifier around the five published precursor-selection
principles from the ASTRAL paper (initiate between only 2 precursors;
precursors should be high-energy/unstable; target should be the deepest hull
point; the 2-precursor composition slice should intersect as few competing
phases as possible; large inverse hull energy if byproducts are unavoidable)
— and validate the rebuild against this same 35-target experimental set
instead of only internal reward-capacity diagnostics.

---

## Step 4 — Does the model ever propose the good precursors? — DONE (2026-09-01 02:28 UTC)

35 targets x 8 samples = 280 generations, SFT checkpoint, closed-book, tmux
`astral_step4` session, `run_astral_step4.sh` (resume-safe, ran clean in one
attempt, ~2h8m wall time).

| | count |
|---|---|
| targets where model EVER proposed the predicted (better) set | **0/35** |
| targets where model EVER proposed the traditional set | 30/35 |
| mean validator reward on traditional matches | 0.930 (n=139 samples) |
| mean validator reward on predicted matches | n/a — zero occurrences |

**Temperature (reward hack, reproduced on this external set):**

| | mean max-T | median max-T |
|---|---|---|
| model proposed | **964.6°C** | 950°C |
| ASTRAL traditional best-condition | 762.9°C | 800°C |
| ASTRAL predicted best-condition | 745.7°C | 700°C |

Zero out of 280 generations across 35 targets ever produced the better,
corpus-rare precursor set — not a near-miss, a clean zero. Sanity check:
Na2Al2B2O7 here independently reproduces Step 2's case study exactly (7/8
traditional matches both times, generated separately).

`misc/astral_model_generations.json` has every sample.

---
