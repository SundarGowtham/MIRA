# Phase 15 Task 4a — ARROWS³ inventory (no scoring)

*Script: `research/distributional/arrows_inventory.py`. Output:
`results/external/arrows_inventory.json`. Per instruction: inventory
only — "the pair count decides the bar" for Task 4b's pre-registration,
not computed here.*

## Data provenance

Source: Szymanski et al., *Nat. Commun.* 2023 (arXiv 2304.09353). Data:
https://github.com/njszym/ARROWS, **commit
`cb630e944315c8ac19f5b9a3c9d8984be93258a6`** (main branch, cloned
2026-09-24 into `data/external/arrows/ARROWS/`, gitignored).

**Git LFS note**: `git-lfs` is not installed on this machine. Cloned with
`GIT_LFS_SKIP_SMUDGE=1` (real clone, correct commit, LFS files left as
pointer stubs), then fetched the three `Exp.json` files' actual content
directly via `media.githubusercontent.com` (GitHub's LFS media endpoint
accepts plain HTTPS GET without the LFS CLI). **Every file's sha256 was
verified against its own LFS pointer's `oid` before use**:
- `Examples/YBCO/Exp.json` — 52,463,427 bytes, sha256 `61a9efa9...` ✓
- `Examples/LTOPO/Exp.json` — 20,322,802 bytes, sha256 `2fd8f33c...` ✓
- `Examples/NTMO/Exp.json` — 2,118,607 bytes, sha256 `3fa43477...` ✓

## Schema

Each `Exp.json` is `{"Universal File": {precursor_set_string:
{"Temperatures": {temp_string: outcome_entry}, "Precursor
stoichiometry": [...]}}}` (LTOPO additionally has one top-level
`"Common Experimental Conditions"` key: heating rate 20°C/min, natural
cooling, air atmosphere).

**Every outcome entry across all three targets has `products` (list of
phase labels) and `product weight fractions` (list of numbers, wt%)** —
a genuine continuous, multi-phase outcome, not just pure/impure. Some
entries additionally carry raw `XRD` (x/y diffraction arrays) alongside
the derived products/fractions — the derived field is present everywhere,
confirmed by direct count, not assumed from a single truncated example.
An `"Experimentally Verified"` boolean and a `"Source"` provenance string
are also present per entry.

| target | precursor sets | temperatures | (set, T) entries | outcome field |
|---|---|---|---|---|
| **YBCO** | 47 | 600, 700, 800, 900, 1000 °C | 200 | 100% products+weight_fractions |
| **LTOPO** | 15 | 400, 500, 600, 700 °C | 58 | 100% products+weight_fractions |
| **NTMO** | 13 | 300, 400 °C | 18 | 100% products+weight_fractions |

## Within-target, same-temperature pairs where the outcome differs

| target | total pairs | differing | different phase SET | same set, different wt% |
|---|---|---|---|---|
| **YBCO** | **4,390** | 4,296 (97.9%) | 3,707 | 589 |
| LTOPO | 392 | 392 (100%) | 158 | 234 |
| NTMO | 88 | 88 (100%) | 73 | 15 |

**Total across all three targets: 4,870 within-target, same-temperature
pairs, 4,776 with a differing outcome.** YBCO alone provides 4,390 pairs
— far more than the ~150 that would give Task 4b's power calculation a
comfortable bar, and orders of magnitude more than ASTRAL's hard ceiling
of 35. This is a genuinely well-powered dataset for the verifier gate.

**Total-variation distance** (wt%, only for pairs sharing an identical
phase set — the more conservative, cleanly-comparable subset):

| target | n (same-phase-set, differing) | median TV distance | max |
|---|---|---|---|
| YBCO | 589 | 21.0 wt% | 59.5 wt% |
| LTOPO | 234 | 8.0 wt% | 38.0 wt% |
| NTMO | 15 | 17.0 wt% | 58.0 wt% |

Differences are large in absolute terms, not marginal — even restricted
to the conservative same-phase-set subset, a typical pair differs by
~8–21 percentage points of product yield.

## Ba carbonate vs. Ba oxide/peroxide: a controlled robot-lab test exists

**Yes — YBCO's 47-set design is a genuine crossed experiment over Ba
source (BaO, BaO2, BaCO3, plus pre-reacted BaCuO2/Ba2(CuO2)3), Cu source
(CuO, Cu2O, CuCO3), and Y source (Y2O3, Y2Cu2O5, Y2(CO3)3).** Grouping by
identical non-Ba co-precursors and requiring ≥2 of {BaO, BaO2, BaCO3} as
the sole Ba source finds **10 matched groups**, of which **6 are full
three-way matches** (BaO, BaO2, AND BaCO3 all present for the identical
Cu/Y co-precursor pair), all at the same 4–5 temperatures (600–900°C, plus
1000°C for the BaCO3/BaO2 members of two groups):

| co-precursors | Ba sources matched | temperatures |
|---|---|---|
| CuCO3, Y2(CO3)3 | BaO2, BaO, BaCO3 | 600–900°C (BaCO3 also 1000°C) |
| CuO, Y2(CO3)3 | BaO2, BaO, BaCO3 | 600–900°C (BaCO3 also 1000°C) |
| CuCO3, Y2Cu2O5 | BaO2, BaO, BaCO3 | 600–900°C (BaCO3 also 1000°C) |
| CuO, Y2Cu2O5 | BaO2, BaO, BaCO3 | 600–900°C (BaCO3 also 1000°C) |
| CuO, Y2O3 | BaO, BaCO3, BaO2 | 600–900°C (BaCO3, BaO2 also 1000°C) |
| CuCO3, Y2O3 | BaO, BaCO3, BaO2 | 600–900°C (BaCO3, BaO2 also 1000°C) |

Plus 4 more two-way (BaCO3 vs. BaO, or BaCO3/BaO/BaO2 via the
pre-reacted `Ba2(CuO2)3` route) matched groups. **This directly answers
the causal-chain question with a controlled experiment, not an
observational literature comparison**: same target (YBCO), same
co-precursors, same temperature, only the Ba source's chemical class
differs. Full detail (which exact precursor-set string maps to which Ba
source, every temperature) in
`results/external/arrows_inventory.json`'s `ybco_ba_source_matched_comparisons`.

**Not scored here** — per Task 4a's explicit "inventory only, no
scoring" instruction. Whether BaO/BaO2 actually outperform BaCO3 in
these matched groups (or vice versa) is Task 4b/4c's question, after a
pre-registration locks the comparison method.

## What this settles for Task 4b's pre-registration

The pair count is large (4,870 total, 4,390 from YBCO alone) and the
Ba-source-matched subset offers a genuinely controlled test beyond the
observational Lee et al. comparison. Task 4b's power calculation and bar
should be set from these real counts, not an assumed number.
