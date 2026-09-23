# Phase 15 Task 3 pre-registration — Lee et al. 2025 vs. the causal chain's link 5

*Written and committed BEFORE downloading or opening any Lee et al. data,
per explicit instruction — the commit timestamp is the proof this came
first. Source: Lee, Cruse, Baibakova, Ceder, Jain, Sci. Data 12:1969
(2025). Dataset: https://doi.org/10.6084/m9.figshare.30423274 (single
JSON, CC-BY). Post-processing code:
https://github.com/slee-lab/solid-state-recipes-with-impurity.*

*Everything in this file not explicitly tagged **[C]** is exploratory —
Task 3's steps 1, 3, 5, 6 (schema inspection, ammonium-phosphate/H3PO4/P2O5
counting, ASTRAL-overlap check) are **[E]**, reported regardless of the
primary/secondary results below, per the instructions doc's own rule
("[C]" = confirmatory, pre-registered before seeing data; "[E]" =
exploratory, found by looking).*

## Why this pre-registration exists

The causal chain established in `docs/phases/PHASE15_DISTRIBUTIONAL.md`
(links 1–4) shows the validator's `thermodynamic_favorable` channel
genuinely, robustly prefers bare alkali oxides over carbonates (a real
ΔG(T) effect, survives finite-temperature correction), and that this
preference drove GDPO to roughly double the model's own bare-oxide
sourcing rate on ASTRAL (25.8% → 56.9%) despite alkali bare oxides being
rare in real solid-state synthesis practice (hygroscopic, awkward to
handle, generally avoided in favor of carbonates as a matter of standard
lab practice — the premise motivating this task). Link 5 — whether the
model's shift moves it toward or away from what actually works in a real
lab — is untested until now. This is the first external, literature-scale
check.

## Primary endpoint [C]

**Restricted to syntheses whose target contains at least one of Li, Na,
K, Rb, Cs**: of the syntheses that source that alkali via either a bare
oxide (Li2O, Na2O, K2O, Rb2O, Cs2O) or the corresponding carbonate
(Li2CO3, Na2CO3, K2CO3, Rb2CO3, Cs2CO3) — the two-way comparison this
whole causal chain is about — report the fraction sourcing via bare oxide
and the fraction via carbonate.

**Pre-registered prediction**: bare-oxide fraction **< 5%**, carbonate
fraction **> 50%**. This is stated as a directional prediction to be
checked against the data, not a pass/fail bar with consequences for
whether Task 3 "clears" — report the actual fractions and state plainly
whether the prediction held, regardless of outcome.

**Denominator convention** (locked here, not decided after seeing
counts): the same per-`(target, synthesis, alkali-element)` triple
convention used throughout Task 2b — a synthesis needing two alkali
elements contributes two triples. "Other" alkali sources (hydroxides,
nitrates, acetates, ternary compounds) are counted and reported but are
NOT part of the bare-oxide/carbonate denominator for this specific
primary-endpoint fraction (consistent with Task 2's own bare-oxide vs
carbonate framing, which never included "other" in either side's rate).

## Secondary endpoint [C]

**For the same alkali-restricted synthesis set**: the impurity-phase rate
of bare-oxide-sourced syntheses is **≥** the impurity-phase rate of
carbonate-sourced syntheses.

**Impurity-phase rate**, as currently specified (to be confirmed, not
redefined, once the schema is inspected — see the schema-confirmation
note below): fraction of syntheses in each group whose record reports at
least one impurity/secondary phase (i.e., is NOT reported as
single-phase / fully phase-pure). The exact field name and its possible
values are unknown until the data is opened (Task 3 step 2, **[E]**,
reports this back before the secondary endpoint is scored) — this
pre-registration commits to using whatever field the dataset's own schema
provides for phase purity/impurity, applied identically and without
adjustment to both groups, not to a specific field name chosen after
looking.

**Report**:
- Both rates (bare-oxide impurity rate, carbonate impurity rate).
- The difference (bare − carbonate).
- A 95% confidence interval on that difference (two-proportion CI; exact
  method to be chosen based on the actual n's once known — Wilson or
  Newcombe for small n, normal approximation acceptable only if both
  group sizes clearly support it).
- The same three numbers **restricted to targets that appear with BOTH
  sourcing choices** in the dataset (a within-target comparison,
  controlling for target-specific difficulty — the pairwise-matched
  version of the same question, more informative than the pooled rate if
  the target sets differ systematically between the two sourcing
  choices).

## Rules restated (from this project's standing pre-registration discipline)

- One design iteration. This file is not edited after Task 3's data is
  opened. A correction found on review gets a dated addendum, not a
  silent edit — same discipline as every other pre-registration in this
  project (`misc/PHASE13_PREREG.md` and its addenda are the precedent).
- The primary and secondary endpoints' definitions (alkali set, bare-oxide
  set, carbonate set, denominator convention, "restricted to targets with
  both sourcing choices") are locked above and are not adjusted after
  seeing the counts.
- Steps 3, 5, 6 of the original Task 3 spec (ammonium-phosphate/H3PO4/P2O5
  counting; the 35-ASTRAL-target overlap check) are exploratory, reported
  in full regardless of what they show, never used to retroactively
  justify the primary/secondary result.
- Record the dataset file's hash once downloaded, in the results doc, per
  the original Task 3 instruction.

## What this settles, and what it doesn't

- If the primary prediction holds (bare-oxide rare, carbonate dominant in
  real practice) AND the secondary prediction holds (bare-oxide sourcing
  associated with equal-or-higher impurity rates in practice), this
  completes causal-chain link 5: the model's shift moves it toward
  validator-favored chemistry and away from both real practice and real
  outcomes — a genuine, externally-grounded reward-hacking finding, not
  merely a verifier-internal one.
- If the primary prediction holds but the secondary does NOT (bare-oxide
  sourcing shows lower impurity rates when chemists do use it), the
  finding is more nuanced: bare-oxide chemistry may be avoided for
  practical/handling reasons unrelated to phase purity, and the model's
  shift would be toward a real thermodynamic edge that practice
  under-exploits — a different, still-interesting story.
- If the primary prediction fails (bare-oxide is not actually rare, or
  carbonate is not actually dominant, in this literature corpus), the
  premise motivating Task 3 itself needs revisiting before link 5 can be
  claimed either way.
