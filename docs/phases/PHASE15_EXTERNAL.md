# Phase 15 Task 3 — Lee et al. 2025: do chemists actually use bare alkali oxides?

*Pre-registration: `docs/phases/PHASE15_LEE_PREREG.md`, committed before
any Lee et al. data was downloaded or opened. Script:
`research/distributional/lee2025_analysis.py`. Output:
`results/external/lee2025_precursor_usage.json`.*

## Data provenance

Source: Lee, Cruse, Baibakova, Ceder, Jain, *Sci. Data* 12:1969 (2025).
DOI `10.6084/m9.figshare.30423274` redirects to a figshare.com article
page that blocks automated fetch (403); the DOI landing page itself
redirects through an authenticated session wall. Resolved instead via
Figshare's public API (`api.figshare.com/v2/articles/30423274`), which
returned the direct file link without authentication.

- File: `SS_rxns_80806.json.gz` → `data/external/lee2025/` (gitignored).
- **md5: `2cca759682be4689e7d0d3d882d12909`** — matches Figshare's own
  supplied hash exactly, verified before use.
- License: CC BY 4.0. 80,806 records.

**Schema** [E], confirmed by direct inspection before writing any
analysis code: each record has `target` (list, `material_formula` field),
`precursors` (list, `material_formula` field), `target_reaction`
(balanced equation), and **`impurity_reaction`** — a list, present in
100% of records (0/80,806 missing), empty when no impurity phase was
reported, non-empty when one was. This is the impurity indicator used
below. **Overall corpus impurity rate (unrestricted): 5.9% (4,746/80,806)**
— a useful sanity baseline for the alkali-restricted rates that follow.

## Primary endpoint [C] — BOTH predictions HOLD

Restricted to the 12,847 records whose target contains an alkali metal
(13,849 `(record, alkali-element)` triples, per the pre-registered
denominator convention):

| category | n | share |
|---|---|---|
| **bare_oxide** | 341 | **2.5%** |
| **carbonate** | 8,311 | **60.0%** |
| other (hydroxide, nitrate, ternary, acetate, ...) | 5,109 | 36.9% |
| missing (no declared precursor supplies that element) | 88 | 0.6% |

**Pre-registered prediction "bare-oxide < 5%": HOLDS (2.5%).**
**Pre-registered prediction "carbonate > 50%": HOLDS (60.0%).**

Real solid-state synthesis practice, at literature scale, overwhelmingly
avoids bare alkali oxides in favor of carbonates — by a factor of ~24x
among the routes that use one or the other.

## Secondary endpoint [C] — prediction DOES NOT hold

| sourcing | n | impure | rate | 95% CI |
|---|---|---|---|---|
| bare_oxide_only | 338 | 21 | 6.2% | [4.1%, 9.3%] |
| carbonate_only | 7,616 | 498 | 6.5% | [6.0%, 7.1%] |
| other_only | 4,893 | 233 | 4.8% | [4.2%, 5.4%] |
| both (multi-alkali, mixed) | 0 | — | — | — |

**Pooled: difference (bare − carbonate) = −0.3%, 95% CI [−3.0%, +2.3%].**
**Pre-registered prediction (bare-oxide impurity rate ≥ carbonate): DOES
NOT HOLD** — the point estimate is marginally in the *opposite* direction
(bare-oxide slightly lower), and the CI comfortably straddles zero either
way. No significant difference.

**Restricted to the 76 targets that appear with both sourcing choices
somewhere in the corpus** (the pairwise-matched, more informative
version): bare-oxide 7.1% (n=141) vs carbonate 6.6% (n=1,295), difference
+0.5%, 95% CI [−3.9%, +5.0%] — this time the point estimate leans the
*predicted* direction, but the CI is wide and includes zero comfortably.
Still no significant difference, either direction, at this n.

**Honest reading**: real-world impurity outcomes do not distinguish
bare-oxide from carbonate sourcing in this corpus. The literature-mining
dataset is also a real limitation here, not just a sample-size one —
self-reported publication success bias likely compresses impurity rates
across the board (5.9% overall is almost certainly an underestimate of
true failure rates in the lab), which could mask a real difference in
either direction. Reported as inconclusive, not as a null result strong
enough to rule out a real effect.

## Causal chain link 5, resolved with nuance

**Part 1 (does practice avoid bare alkali oxides?): YES, strongly.**
2.5% vs 60.0%, a ~24x preference for carbonate among routes using either.
GDPO's shift to 56.9% bare-oxide sourcing on ASTRAL moves the model
**dramatically away from what practicing chemists actually do.**

**Part 2 (is that practice justified by worse bare-oxide outcomes?):
NOT CONFIRMED.** No significant impurity-rate difference either direction
in this dataset. The model's shift is not shown to be moving toward
worse real-world outcomes — only away from real-world *practice*. These
are different claims, and only the first is supported.

**Combined with Task 2c** (the carbonate-vs-bare-oxide ΔG(T) preference is
a real, Gibbs-corrected thermodynamic effect, not an artifact): the
overall picture is that GDPO is chasing a genuine thermodynamic signal
that real chemists apparently do not act on — plausibly because bare
alkali oxides are hygroscopic and awkward to handle at bench scale (a
practical constraint the validator has no channel for), not because they
perform worse when used. This is a more precise, better-evidenced
version of the reward-hacking story than "moves away from good chemistry
practice" alone would have been.

## [E] Step 5 — phosphorus source usage: the same pattern, independently

| source | n | impurity rate |
|---|---|---|
| **ammonium phosphate** | **2,564** | 2.8% |
| other | 2,092 | 2.0% |
| H3PO4 | 134 | 2.2% |
| P2O5 | 129 | 1.6% |

**Ammonium phosphate is the overwhelmingly dominant real phosphorus
source** (2,564 vs. 134+129=263 for H3PO4/P2O5 combined, ~10:1). This is
the OPPOSITE direction from what the model's own training moved toward
(Task 2, ammonium-phosphate share on phosphate targets: 23.8% base →
82.6% full SFT → **4.9% RS-SFT → 2.1% GDPO**, i.e. RS-SFT and GDPO moved
sharply AWAY from ammonium phosphate, toward H3PO4). **This is not a
second, independent finding — it is very likely the SAME balance-solver
bug** (Phase 13's NH3-without-N2 gap; Task 2's finding that ammonium
routes fail `stoichiometry` catastrophically, gap −0.905) forcing the
model away from the precursor real chemists actually prefer 10:1, for a
software reason, not a chemistry one. **Both of the model's largest
precursor-class shifts (away from carbonates, away from ammonium
phosphate) move it away from real practice, and both are traceable to
validator artifacts (a genuine but impractical ΔG preference; a balance-
solver bug) rather than genuine chemical improvement.**

## [E] Step 6 — ASTRAL target overlap: even more stark on exactly these targets

**20 of ASTRAL's 35 targets appear as targets in Lee et al.**: `K2Zr(PO4)2,
KBaPO4, KMgPO4, KNbWO6, KNiPO4, KTiNbO5, Li2CuP2O7, Li2TiGeO5, Li2TiSiO5,
Li3Fe2(PO4)3, Li3Sc2(PO4)3, Li3V2(PO4)3, LiGeBO4, LiMgPO4, LiMnPO4,
LiNbWO6, LiZnBO3, LiZnPO4, Na2Al2B2O7, NaSrBO3`.

Full precursor-set listing in `results/external/lee2025_precursor_usage.json`
(`astral_overlap.detail`) — **not one of the displayed literature routes
for these 20 exact ASTRAL targets uses a bare alkali oxide**; every
alkali source shown is a carbonate (occasionally a hydroxide or nitrate).
This is a smaller, target-matched sample than the full corpus-wide 2.5%,
but it is the most directly relevant evidence available: for the precise
chemistry space ASTRAL and this project's models operate in, real
synthesis practice does not use bare alkali oxides at all in the
literature records recovered here.

## Summary

| question | answer |
|---|---|
| Do chemists avoid bare alkali oxides in practice? | **Yes, strongly (2.5% vs 60.0%; 0 instances among the 20 ASTRAL-overlap targets)** |
| Is that avoidance justified by worse outcomes when bare oxides ARE used? | **Not confirmed (no significant impurity-rate difference, either pooled or target-matched)** |
| Does the same divergence-from-practice pattern show up elsewhere? | **Yes — ammonium phosphate (real chemists' overwhelming choice, 10:1) is exactly what RS-SFT/GDPO moved away from, likely the same balance-solver bug** |
