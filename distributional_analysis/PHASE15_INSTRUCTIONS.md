# Phase 15 — instructions for Claude Code

*Written 2026-09-20. Read this whole file before starting. Do tasks in order.
Stop and report after each task — do not chain into the next one without a report.*

---

## Context in five lines

1. Phase 12 (GDPO β=0 from RS-SFT) is done: 14/35 ASTRAL predicted-set hits, pre-registered, settled. Do not touch.
2. Phase 13 stopped by pre-registration. Phase 14 was never launched. Do not reopen either.
3. A new analysis of the existing n=32 ASTRAL generations found that the biggest thing training changed was **not** ASTRAL routes (share stayed 2–4%). It was a swap from **carbonate precursors to bare alkali oxides** (Li₂O, Na₂O, K₂O): 8.6% of samples in base → 25.1% RS-SFT → 56.3% GDPO. In base samples, bare-oxide routes pass the 0.9 validator bar 98.9% of the time vs 78.2% for carbonate routes. Hypothesis: the validator penalizes gas-releasing precursors, and both RS-SFT filtering and GDPO followed that penalty.
4. Three public external datasets can fix the "only 35 confounded pairs" problem: Lee et al. 2025 (literature, 80k syntheses with purity labels), ARROWS³ (robot lab, one target, 47 precursor sets, continuous yield), Precursor Genome (robot lab, 1,035 pairwise reactions).
5. The GPU is free.

## Rules that apply to every task

- **Never delete anything.** `git mv` only. Never modify `core/validator.py`, `core/ranker.py`, `core/comparator.py`, or any existing results file.
- New scripts go in `research/`. New outputs go in `results/` (tracked). New docs go in `docs/phases/`.
- Every finding gets a tag in the results doc: **[C]** = confirmatory (pre-registered before seeing data) or **[E]** = exploratory (found by looking).
- For any task marked PREREG: write the pre-registration file, **commit it**, then run. The commit timestamp is the proof it came first.
- Report every number with its n. Report per-channel / per-family tables, never only an aggregate.
- If a task turns up something that doesn't match this spec, stop and report. Do not improvise around it.

---

## Task 1 — Commit the distributional analysis  (15 min, CPU)

Files provided alongside this doc: `distributional_analysis/analyze.py`, `analyze2.py`, `analyze3.py`, `per_target.json`, `README.md`.

1. Copy the three scripts into `research/distributional/`. Copy `per_target.json` and `README.md` into `results/distributional/`.
2. Re-run all three scripts against `results/astral_gen_n32_{base,sft,rs_sft,gdpo_phase12_beta0}.json` and confirm the numbers match the README. If any number differs, report it — don't fix silently.
3. Write `docs/phases/PHASE15_DISTRIBUTIONAL.md` containing the README's findings, all tagged **[E]**.
4. Add one line to that doc resolving the hit definition: full SFT is **0/35 under exact match** and **1/35 if superset matches count** (the superset is LiZnBO₃: Li₂CO₃ + LiBO₂ + ZnO). State that the paper will use exact match, and list every place in `CLAUDE.md` / `docs/` that cites "1/35" so they can be footnoted.

**Report:** confirmation that numbers reproduce, plus the list of "1/35" citations.

---

## Task 2 — Which validator channel penalizes carbonates?  (2–3 h, CPU)

Goal: turn the bare-oxide finding from a correlation into a mechanism.

1. From `results/astral_gen_n32_base.json`, take every parsed sample. Classify each as **bare-oxide-only**, **carbonate-only**, **both**, or **neither**, using:
   - bare alkali oxide = {Li2O, Na2O, K2O, Rb2O, Cs2O}
   - carbonate = {Li2CO3, Na2CO3, K2CO3, BaCO3, SrCO3, CaCO3, MgCO3}
2. Re-score each sample with the **current, unmodified** `SynthesisValidator` and capture the **full per-channel breakdown** (every channel, its value, and whether it was gradeable).
3. Produce a table: for each channel × {bare-oxide-only, carbonate-only} → mean score, fraction scoring below 1.0, fraction ungradeable, n.
4. Identify which channel(s) account for the pass-rate gap at the 0.9 bar (98.9% vs 78.2%). For the top offending channel, pull 5 example carbonate routes that fail it and print the exact reason.
5. Repeat step 3 on ammonium-phosphate routes (any precursor containing both N and H, excluding nitrates) vs H3PO4 routes on phosphate targets, to see whether it's the same channel.

**Output:** `results/distributional/carbonate_penalty_by_channel.json`, section added to `PHASE15_DISTRIBUTIONAL.md`, tagged **[E]**.
**Report:** the table and the named channel(s).

---

## Task 3 — Lee et al. 2025: do chemists actually use bare alkali oxides?  (3–4 h, CPU)

Source: Lee, Cruse, Baibakova, Ceder, Jain, *Sci. Data* 12:1969 (2025). Dataset: https://doi.org/10.6084/m9.figshare.30423274 (single JSON, CC-BY). Post-processing code: https://github.com/slee-lab/solid-state-recipes-with-impurity

1. Download into `data/external/lee2025/` (gitignored). Record the file hash in the results doc.
2. Read the data record format before parsing. Report the schema back if anything is ambiguous.
3. Count, across all 80,806 syntheses: how many use each bare alkali oxide as a precursor vs the corresponding carbonate (Li2O vs Li2CO3, Na2O vs Na2CO3, K2O vs K2CO3).
4. For each, report the **phase-impure rate** (fraction of those syntheses that report an impurity phase).
5. Same count for ammonium phosphates vs H3PO4 vs P2O5 as phosphorus source.
6. Check overlap: how many of the 35 ASTRAL targets appear as targets in Lee? List them with their precursor sets and purity labels.

**Output:** `results/external/lee2025_precursor_usage.json`, `docs/phases/PHASE15_EXTERNAL.md` section. Tag **[E]**.
**Report:** the usage table. The key question is whether our models drifted **toward** or **away from** what chemists actually use and what works.

---

## Task 4 — ARROWS³ verifier gate  (1 day, CPU, PREREG)

Source: Szymanski et al., *Nat. Commun.* 2023 (arXiv 2304.09353). Data: https://github.com/njszym/ARROWS — `Examples/YBCO/Exp.json`, `Examples/LTOPO/Exp.json`, `Examples/NTMO/Exp.json`. **These are Git LFS files** (YBCO ≈ 52 MB); clone with `git lfs`.

### 4a — inventory first (no scoring)
1. Clone into `data/external/arrows/` (gitignored). Record commit hash.
2. Report the schema of each `Exp.json`, the number of distinct precursor sets, temperatures, and what outcome field exists (yield wt% / phase fractions / pure-impure).
3. Enumerate every **within-target, same-temperature** pair of distinct precursor sets where the outcome differs. Report the count per target and per temperature, and the distribution of outcome differences.

**Stop and report after 4a.** The pair count decides the bar.

### 4b — pre-registration (after I reply to 4a)
Write `docs/phases/PHASE15_ARROWS_PREREG.md` with:
- **Primary endpoint:** fraction of same-temperature pairs where the verifier's preferred route has the higher measured yield.
- **Baselines, both required:** 50% (chance) **and** the "always prefer fewer precursors" rule and the "always prefer the carbonate-free route" rule. A verifier must beat all of them, not just chance — Phase 13 showed a constant can look good.
- **Bar:** set from a power calculation on the actual pair count (one-sided α=0.05, 80% power to detect 60% agreement). Write the numbers.
- **Verifiers scored:** (i) `validator.py` as-is, (ii) `comparator.py` as-is. No tuning.
- **Grouping:** because pairs share precursors, report a bootstrap CI that resamples **precursor sets**, not pairs.
- One iteration. No changes after seeing results.

Commit, then:

### 4c — score and report
Output `results/external/arrows_gate.json` and `docs/phases/PHASE15_ARROWS_RESULTS.md` with the primary endpoint, all baselines, per-channel agreement, per-temperature breakdown. Tag **[C]**.

---

## Task 5 — Dose-response via teacher-forcing  (1–2 days, GPU, forward passes only, PREREG)

Goal: for routes with known lab outcomes, measure (a) how likely the base model was to propose them and (b) how much RS-SFT and GDPO changed that. Exact probabilities, no sampling noise.

### 5a — build the route set
From ARROWS³ (Task 4) and, where targets overlap, Lee (Task 3) and ASTRAL: for each target, list precursor sets with an outcome label (pure/impure, or yield). Render each as the **precursor segment only**, in the exact output format our models emit, conditioned on the standard prompt for that target.

### 5b — pre-registration
Write `docs/phases/PHASE15_DOSE_PREREG.md` with three questions and predictions:
1. **Leash:** is there a base log-probability below which GDPO's change is indistinguishable from zero? Fix ε (e.g. log p = −20) **before** measuring.
2. **Quality alignment:** at matched base log-probability, did GDPO raise phase-pure routes more than phase-impure ones? (Test: regression of Δlog p on purity with base log p as covariate.)
3. **Top-5 prediction:** published work says RL's promoted token is always already in base's top-5. At the first precursor token, is the ASTRAL-predicted precursor in base's top-5? Pre-register yes/no expectation.

Commit, then:

### 5c — run
Checkpoints: `Qwen/Qwen3-8B` (base), `runs/sft-qlora-rs-sft-from-base/final`, `runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300`. Also full SFT `runs/sft-qlora-sft-v3-2nd-rank16` for reference.
For each route: total log p of the precursor segment, per-token log p, and at the first precursor token the top-20 tokens with log-probs.
**No sampling. Same prompts, same tokens, all checkpoints.**

Output `results/dose_response/*.json` and `docs/phases/PHASE15_DOSE_RESULTS.md`. Tag **[C]** for the three pre-registered questions, **[E]** for anything else.

---

## Task 6 — Precursor Genome check  (a few days, CPU) — do not start until Tasks 1–5 are reported

Source: Walters et al., arXiv 2607.09903 (July 2026), 1,035 pairwise A-Lab reactions.
Only step for now: find the data release, report its schema and licence, and report how many reactions involve a bare alkali oxide or a carbonate. Nothing more until I reply.

---

## Not in scope

- No new RL training runs.
- No changes to reward functions.
- No reopening Phase 12, 13, or 14 results.
