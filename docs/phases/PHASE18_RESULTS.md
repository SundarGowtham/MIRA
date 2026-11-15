# Phase 18 results

Pre-registration: `docs/phases/PHASE18_PREREG.md`, git blob
`5c0a84ce0161f03f19a6198ac0aa7fffe949f5ac`. The spec was not yet committed when
the blocks ran (Gowtham asked for unattended execution and commits his own
work), so every results JSON carries `prereg_blob_sha` plus `prereg_commit`
(None until committed; `git log --find-object=<blob>` recovers it). Every
script asserts the spec's blob hash at start-up, so an edited spec aborts the
run. Anything decided after the spec was written is labelled **[E]**.

Run order (kickoff prompt): 0, 18A, 18B, 18C, 18D, 18F, 18H, 18E, 18G.

---

## 2026-11-15 — Block 0 (inventory)

Output: `results/phase18/inventory.json` (+ `results/phase18/recipes_by_model.json`,
canonical recipes per model/target-set/target, reused by 18A and 18E).
Script: `research/phase18_block0_inventory.py`.

- **Models and adapters.** All 12 adapters for S1–S7 exist (Corpus SFT,
  RS-SFT, E3, E3c, GDPO-v3 @100/200/300, Phase 12 @100/200/300, E1
  @100/200/300); every one is LoRA r = 16, α = 32, the same 7 target modules.
  Intermediate checkpoints: Corpus SFT `checkpoint-117`, RS-SFT
  `checkpoint-27`, E3 and E3c `checkpoint-48`; practice runs 100/200/300.
- **Raw completions.** Every (model, target set) has samples for all 35 ASTRAL
  and 40 held-out targets. Base and RS-SFT: 232 rows per target (200
  reference + 32 resampling); all others 32. Parse rates 96–99%. Four ASTRAL
  sources (Corpus SFT, GDPO-300, Phase 12 @300, E1 @300) are the structured
  `astral_gen_n32_*` files (precursors only, no reasoning text); that is enough
  for every recipe-level block.
- **Phase 17 outputs.** 50 files in `results/phase17/`. Blocks 0, A, B, C
  (infeasible as specified), G1 and R ran; D, E, F and H1 did not.
- **Base weights.** bf16 Qwen3-8B present in the local HF cache
  (`/models/huggingface/hub/models--Qwen--Qwen3-8B`, 5 shards, 16.4 GB). No
  download needed.
- **ASTRAL lab data.** 35 targets, two measured sets each (traditional,
  rule-picked) with best phase purity. Phase 16's hit = exact match to the
  rule-picked set. For a per-recipe binary outcome the blocks use purity > 0.5
  (ASTRAL's "high" band): 23 targets have a successful set, 23 a failed one.
  Decided in Block 0, before any Phase 18 result; 18E also reports the Phase 16
  singleton definition.
- **ARROWS³ YBCO.** 200 (recipe, temperature) entries, 47 recipes, 600–1000 °C,
  yields 0–100 wt%, via `arrows_gate_scoring.load_target_entries`.
- **Judges.** `VALIDATOR_VERSION` 2026-10-03-noneprop+hydrate+gasuptake+atmtokens+nh3nitrogengate;
  `COMPARATOR_VERSION` 2026-10-03-v3-phase16-c3c6fix; ranker v2 = `core/ranker.py`
  `Ranker(..., validator_version=2)`, `RANKER_VERSION` 2026-10-03-v3-phase16.
- **Readiness.** Inputs for all eight blocks are present; none is skipped.

---

## 2026-11-15 — 18A (transport or reweighting, solved exactly)

Output: `results/phase18/transport_exact.json`, `results/phase18/figdata_transport_exact.json`.
Script: `research/phase18_a_transport_exact.py`. Primary pool: ASTRAL clean 22 +
held-out 40 (62 targets × 7 steps); secondary: ASTRAL 35 + held-out 40.

**Convergence.** The LP (HiGHS) solved **100%** of 420,000 problems across all
525 target-steps (Phase 17 Block C: 6/385). The POT Sinkhorn comparison
returned finite plans on 525/525 at rep 0, but POT warned of non-convergence
on some of them; its "converged" field records finiteness only, so it is not
quoted as a convergence rate.

**Implementation decisions made before running** (in the script docstring):
entries with C ≥ 2λ are bounded at 0, so at exact indifference mass counts as
destroyed + created; the noise floor uses two disjoint before-model subsamples of
size n_f = min(n, N_before/2), and the new-support reference is the
before-model's samples minus the floor subsample for both step and floor. S5's
before-model (Corpus SFT, 32 samples/target) therefore has n_f = 16.

**P-18A-1 — FAILED, reversed.** Median T, primary pool:

| λ | S2–S4 | S5–S7 | diff | 95% CI |
|---|---|---|---|---|
| 0.35 | 0.489 | 0.728 | −0.239 | [−0.327, −0.090] |
| 0.5 | 0.850 | 0.984 | −0.134 | [−0.192, −0.064] |

Per step (λ = 0.35): S1 0.60, S2 0.52, S3 0.49, S4 0.51, S5 0.97, S6 0.61,
S7 0.66. The secondary pool gives the same sign and the CIs exclude 0. This
is the pre-registered "does not support the thesis" outcome: practice steps
have the *higher* transport share, not the lower. **[E] n ≤ 32 sensitivity**
(decided after the smoke test showed S2 compared at n ≈ 224): unchanged, diff
−0.236 [−0.326, −0.105] and −0.151 [−0.203, −0.082].

What sits under T (mean mass split, λ = 0.35, primary): own-sample fine-tuning
keeps 0.73 in place, moves 0.15 and creates/destroys 0.12; S6 keeps 0.63,
moves 0.23, creates 0.14; S7 0.67 / 0.22 / 0.11; S5 0.88 / 0.10 / 0.02. So T
is high for practice because a large share of what it changes lands on a
recipe within one precursor swap (λ = 0.35: 80% of moved practice mass within
Jaccard 0.5, vs 73% for S2–S4), not because it changes less. **[E] reading,
not a test:** T is a ratio and cannot tell "reweight between two recipes
already in support that differ by one swap" from "move mass to a new recipe";
P-18A-2 is the support measure.

**P-18A-2 — FAILED as registered (S6), with the "does not support" condition
NOT triggered.** New-support mass minus its noise floor, primary pool:

| step | new-support mass | floor | diff | 95% CI | |
|---|---|---|---|---|---|
| S1 | 0.080 | 0.022 | +0.058 | [+0.030, +0.090] | |
| S2 | 0.062 | 0.031 | +0.032 | [+0.018, +0.045] | |
| **S3** | 0.033 | 0.022 | +0.011 | [+0.001, +0.022] | excludes 0 ✓ |
| **S4** | 0.042 | 0.022 | +0.020 | [+0.007, +0.034] | excludes 0 ✓ |
| **S5** | 0.060 | 0.066 | −0.006 | [−0.019, +0.006] | includes 0 ✓ |
| **S6** | 0.010 | 0.021 | −0.011 | [−0.015, −0.007] | **excludes 0, below** ✗ |
| **S7** | 0.017 | 0.021 | −0.004 | [−0.010, +0.002] | includes 0 ✓ |

The prediction required S6's CI to include 0; it lies entirely *below* 0
(Phase 12 puts less mass on recipes unseen by RS-SFT than RS-SFT's own resample
does), so P-18A-2 fails as worded. The "does not support" condition (a practice
step's CI above its floor) is not met for any of S5–S7. Every fine-tuning step,
S1–S4, lies above its floor. The secondary pool and the n ≤ 32 sensitivity
match this.

---

## 2026-11-15 — 18B (does judge inconsistency sit where the judge is wrong?)

Output: `results/phase18/curl_vs_lab.json`. Script: `research/phase18_b_curl_vs_lab.py`.

Graph: ARROWS³ YBCO, 47 recipes / 200 routes; 125 routes pass the comparator's
gates; 29 nodes and 406 edges with a defined comparator flow; 247 edges with a
nonzero lab sign. Weighted non-gradient energy of the comparator flow on this
graph: **0.9%** (G1 found 5.2% on model-sampled ASTRAL graphs). The comparator
disagrees with the lab sign on 119/247 edges (48.2%).

**P-18B-1 — FAILED.** AUC of ρ_e for predicting disagreement = **0.484**,
95% CI [0.312, 0.624] (node bootstrap), permutation p = 0.67 (null mean ≈ 0.50).
Mean ρ is 0.036 on disagreeing edges and 0.035 on agreeing ones. Curl does not
locate the comparator's errors on YBCO. As pre-registered, that is still
reported: on this graph the comparator is almost a pure gradient (0.9%
non-gradient), so its errors are mostly *consistent* errors that a potential
explains, not cycles.

**[E] ASTRAL:** not computable. Two measured recipes per target give one edge
each; the pooled graph has no cycles, so every residual is 0.

**[E] Channel conflict, scalar rewards.** R2 (5-channel checker): AUC 0.545,
node-bootstrap CI [0.494, 0.610], permutation p = 0.001 (678 edges,
disagreement rate 37.8%). The permutation test rejects, the cluster bootstrap
just includes 0.5: a weak signal that does not survive clustering by recipe.
R4 (ranker v2): AUC 0.469 [0.344, 0.606], p = 0.83.

---

## 2026-11-15 — 18D (does a judge score the physics or the text?)

Output: `results/phase18/audit_representation.json`. Script:
`research/phase18_d_audit_representation.py`. Suite: R-a's 200 routes (same
selection rule). Each transformation edits the route **as written** (the
completion's own JSON), re-serialises it and passes it through
`parse_completion` → judge; the untransformed JSON, re-serialised the same
way, is the reference. 0 reference parse failures, 0 parse failures under any
transformation. Threshold 1e-3. 10 routes fail the comparator's gates on both
sides (comparator undefined, n = 190).

**[C-measure] violation counts (routes violating / routes scored):**

| | T1 °C→K | T2 h→min | T3 split hold | T4 zero hold | T6 permute | T5 ramp |
|---|---|---|---|---|---|---|
| validator, 5-ch aggregate | **10/200** | 0/200 | 0/200 | 0/200 | 0/200 | n/a |
| ranker v2 | **190/200** | 0/200 | 0/200 | 0/200 | 0/200 | n/a |
| comparator | **190/190** | 0/190 | 0/190 | 0/190 | **154/190** | n/a |

**Report-card column R-a′ (representation): FAIL for all three judges** (validator 10,
ranker v2 190, comparator 344 violations over T1–T6).

- **T1.** The schema has no unit field. A temperature written as "1073.15 K"
  is parsed as 1073.15 °C (the parser strips the unit), so every route is
  heated 273 °C hotter. The validator's aggregate moves on 10/200 routes (all
  through `thermodynamic_favorable`, mean Δ = +0.12 when it moves:
  hotter reads as *more* favourable, the temperature hack's direction);
  `temperature_plausible`'s 100–2000 °C window absorbs the shift. Ranker v2
  moves on 190/200 (`volatility_risk` 185, `precursor_decomposition_match`
  160, `driving_force_margin` 106; mean Δ −0.054). **[E] T1b** (key renamed
  `temperature_k`): the parser drops the temperature entirely; validator
  172/200, ranker 190/200, comparator 0/190.
- **T2, T3, T4: 0 violations, but vacuously.** The parser also misreads
  "720 min" as 720 h (verified in `parse_examples`), yet nothing moves,
  because **no channel of any judge reads duration**, and splitting or
  padding a hold changes no channel either. These judges are invariant to
  how a schedule is written only because they ignore the schedule's time
  axis. They are not discretisation-invariant in the neural-operator sense.
- **T6 (control): the comparator is not order-invariant** (154/190). **[E]
  root cause, traced:** `C7_gas_evolution`'s raw value differs by ~1e-15
  between orderings (floating-point summation order, e.g. −2.0 vs
  −1.9999999999999991). C7's raw values sit exactly on a few discrete values
  (finding 23), so this wobble crosses a step of its percentile scale and the
  margin jumps by ~0.012 (one bucket). Every per-route channel value is
  otherwise identical. R-a tested order invariance on the validator only;
  the validator and ranker pass it here as well.
- T5 not applied: `PredictedConditions` has no ramp-rate field and the parser
  drops unknown keys, so the schema does not allow stating one.

---

## 2026-11-15 — 18C (low-rank structure of the updates; *The Path Not Taken*)

Output: `results/phase18/weight_geometry.json`. Script:
`research/phase18_c_weight_geometry.py`. 252 modules (36 layers × 7); base =
bf16 Qwen3-8B, exact SVD (float32, GPU); updates' exact SVD through their
rank ≤ 32 factorisation. **Caveats (as registered):** Euclidean weight-space
angles ignore the Fisher metric (why the project deprecated this diagnostic
in September); reported for comparability with *The Path Not Taken*, which
used full fine-tuning. Here every fine-tuning update has rank ≤ 16 and every
practice update rank ≤ 32.

**Practice continues its init adapter (checked):** median cosine between the
checkpoint-100 `lora_A` and the init adapter's `lora_A` is 0.999999 (S5),
0.999995 (S6), 0.999995 (S7). **[E] consequence:** practice barely moves A, so
ΔW ≈ (α/r)(B₁ − B₀)A₀ and practice's right (input-side) singular subspace is
inherited from the init adapter's A, by LoRA's construction rather than
chosen by the RL signal.

**P-18C-1 — FAILED.** Pooled median E₆₄ (left, base's top-64 output directions):

| | S2–S4 | S5–S7 | practice − FT | 95% module-bootstrap CI | random k/d_out |
|---|---|---|---|---|---|
| left (registered) | 0.0337 | 0.0339 | +0.0002 | [−0.0022, +0.0030] | 0.0156 |
| right [E side] | 0.0217 | 0.0205 | −0.0012 | [−0.0016, −0.0004] | 0.0156 |

Per step (left): S1 0.033, S2 0.035, S3 0.033, S4 0.034, S5 0.030, S6 0.037,
S7 0.037. Practice updates do not avoid the base weights' principal output
directions any more than own-sample fine-tuning does; both sit at about 2× the
random expectation, i.e. both are somewhat *aligned* with the principal
subspace. The input-side E₆₄ is lower for practice with a CI excluding 0, but
that side was not the registered one, and under the A-freezing above it
mostly reflects the init adapters' A. *The Path Not Taken*'s signature did
not appear under LoRA in this domain, which is the outcome the pre-registration
said the paper would report.

**P-18C-2 — FAILED (S6), holds for S7.** Share of modules whose effective
rank of the cumulative ΔW_t is non-decreasing over 100 → 200 → 300:
S6 **45.2%**, S7 **96.8%** (bar 75%, both required). Median effective rank
S6 6.40 → 6.28 → 6.44; S7 6.02 → 6.39 → 6.41. (S5, not part of the
prediction: 30.2%, 10.72 → 10.80 → 10.76.)

**[E]** Median effective rank of the update: S1 4.6, S2 7.2, S3 5.6, S4 5.6,
S5 10.8, S6 6.4, S7 6.4 (stable rank 1.8–2.9). So practice updates are not
lower-rank than fine-tuning updates; S5 is the highest. By module type, E₆₄ is
highest in `down_proj` and `o_proj` (2.2–2.6× random) for every step and
practice is marginally higher there (S6/S7 2.4–2.6 vs S2–S4 2.0–2.3). The fine-
tuning runs' only intermediate checkpoints are byte-identical to their final
adapters, so they give no trajectory.

---

## 2026-11-15 — 18F (do specific channels know the lab better than the aggregate?)

Output: `results/phase18/channel_validity.json` (input:
`results/phase18/arrows_pair_table.json`, built by
`research/phase18_arrows_pairs.py` from R-f's exact pair set). Script:
`research/phase18_f_channel_validity.py`. ARROWS³ YBCO primary endpoint, 1,915
pairs; cluster (recipe) bootstrap 10,000; stratified permutation null 2,000.

**Operationalisation decided before running:** the plug-in CMI is ≥ 0, so a
percentile CI almost never contains 0. "Corrected CI excludes 0" is read as
Holm-corrected (10 validator channels) stratified-permutation p < 0.05. Each
channel's Holm-level CI of the null-centred CMI is also stored.

I(R;Q | carbonate-free), bits:

| | CMI | 95% CI | perm p | null mean | pairs | Holm |
|---|---|---|---|---|---|---|
| R2 checker vote (aggregate) | 0.0253 | [0.0063, 0.0856] | 0.0005 | 0.0024 | 1915 | |
| R4 ranker v2 (aggregate) | 0.0077 | [0.0032, 0.0547] | 0.002 | 0.0023 | 1915 | |
| R5 comparator (aggregate) | 0.0127 | [0.0011, 0.0635] | 0.0005 | 0.0013 | 1647 | |
| `precursors_exist` | **0.0248** | [0.0048, 0.0924] | 0.0005 | 0.0023 | 1915 | ✓ |
| `amount_accuracy` | 0.0197 | [0.0049, 0.0979] | 0.0005 | 0.0029 | 1524 | ✓ |
| `thermodynamic_favorable` | 0.0176 | [0.0033, 0.1128] | 0.0015 | 0.0046 | 802 | ✓ |
| `stoichiometry` | 0.0031 | [0.0012, 0.0438] | 0.25 | 0.0024 | 1915 | ✗ |

The other six channels (`charge_neutrality`, `operation_order`,
`temperature_plausible`, `target_stability`, `chempot_atmosphere`,
`target_match`) never differ in sign within a YBCO pair, so they carry 0 bits
here by construction.

**[C-measure] results.** 3 of 10 channels survive Holm. Best channel
(`precursors_exist`, chosen on the full data) minus aggregate on the same pairs:
vs R2 −0.0005 bits [−0.055, +0.057]; vs R4 +0.017 [−0.013, +0.057];
vs R5 +0.003 [−0.017, +0.043]. No CI excludes 0. **Pre-registered reading: no
channel knows the lab better than the aggregate.** Aggregation does not
destroy validity measurably here: the channels carry about as much
carbonate-adjusted lab information as the aggregate built from them, and
all of it is small (≤ 0.025 bits out of ≤ 1 bit for a binary outcome).
**[E] caveat:** the best channel, `precursors_exist`, is the one finding 29
showed measures Materials Project coverage (Y₂(CO₃)₃ is absent from MP). Its
lab information on YBCO plausibly comes from that one compound's routes,
not from chemistry.

**[E] ranker-v2 objectives as channels:** only `driving_force_margin`
survives Holm (0.019 bits, p = 0.004, 704 pairs); `temperature_economy` and
`volatility_risk` are ungradeable on every pair. **ASTRAL (35, small):** no
channel survives; R2 0.048 bits p = 0.75, R4 0.046 p = 0.24, R5 0 bits.

---

## 2026-11-15 — 18H (a calibrated judge that abstains)

Output: `results/phase18/conformal.json`. Script: `research/phase18_h_conformal.py`.
YBCO 1,915 pairs; 100 recipe-grouped splits (about 494 calibration and 461
test pairs each, cross pairs dropped); p(a ≻ b) = sigmoid(β·margin), β fit by
maximum likelihood on the calibration half (β ≥ 0).

| judge | target | achieved coverage | decisive | abstain | accuracy when decisive | β |
|---|---|---|---|---|---|---|
| R2 checker | 80% | 0.858 | 0.441 | 0.559 | 0.695 | 0.68 |
| R2 checker | 90% | 0.973 | **0.087** | 0.913 | 0.681 | |
| R4 ranker v2 | 80% | 0.792 | 0.482 | 0.518 | 0.574 | 0.83 |
| R4 ranker v2 | 90% | 0.883 | **0.300** [0.08, 0.49] | 0.700 | 0.624 | |
| R5 comparator | 80% | 0.983 | 0.027 | 0.973 | 0.353 | 0.009 |
| R5 comparator | 90% | 1.000 | **0.000** | 1.000 | — | |
| [E] carbonate-free | 80% | 0.877 | 0.404 | 0.596 | 0.694 | 0.87 |
| [E] carbonate-free | 90% | 0.954 | 0.122 | 0.878 | 0.613 | |

No split produced an empty set. (Bracketed range = 2.5–97.5 percentile over splits.)

**P-18H-1 — FAILED (R4).** Decisive share at 90% coverage: R2 8.7% ✓,
R5 0.0% ✓, **R4 30.0%** ✗ (bar ≤ 20%). The calibrated ranker commits on 30% of
lab decisions, but at 62% accuracy when it commits, so its decisiveness reflects
a margin that sorts some pairs confidently rather than validity much above
chance. The comparator's fitted β ≈ 0.009 (its margin is not predictive, or
slightly anti-predictive, on YBCO), so a calibrated comparator abstains on
every pair. The checker and the two-line carbonate-free rule behave almost
identically (decisive 44% vs 40% at 80%, accuracy 0.70 vs 0.69).
**[E] ASTRAL (35):** at 90%, decisive R2 4.6%, R4 3.5%, R5 13.7%,
carbonate-free 0%.

---

## 2026-11-15 — 18E (posterior matching instead of best sample)

Output: `results/phase18/posterior_matching.json`. Script:
`research/phase18_e_posterior_matching.py`. ASTRAL samples, n ≤ 32 per target for
every model (Base and RS-SFT use their 32-sample `resampling_rate` draw, decided
before running, because coverage depends on n). V(x) primary = measured sets
with purity > 0.5 (Block 0). That leaves **15 of the clean 22** targets
(23 of 35) with a non-empty V. The Phase 16 singleton {rule-picked set} is the
sensitivity definition. Kernel 1 − Jaccard distance; mean over targets; target
cluster bootstrap 10,000.

Primary (purity V, clean 22 → 15 targets):

| model | lab-valid mass | coverage | MMD | near-miss | within one swap, not valid |
|---|---|---|---|---|---|
| Base | 0.207 | 0.300 | 0.882 | 0.568 | 0.313 |
| Corpus SFT | 0.261 | 0.200 | 0.930 | 0.535 | 0.191 |
| RS-SFT | 0.140 | 0.233 | 0.867 | 0.627 | 0.290 |
| E3 | 0.173 | **0.400** | 0.839 | 0.591 | 0.374 |
| E3c | 0.175 | 0.367 | **0.836** | 0.603 | 0.305 |
| GDPO-300 | **0.277** | 0.300 | 0.925 | 0.520 | 0.183 |
| Phase 12 @300 | 0.065 | 0.400 | 0.932 | 0.670 | 0.336 |
| E1 @300 | 0.071 | 0.333 | 0.923 | 0.670 | 0.324 |

**P-18E-1 — HOLDS.** E3 − Phase 12 @300 MMD = **−0.092**, 95% CI
[−0.168, −0.020]. Same sign with the CI excluding 0 in every sensitivity cut:
purity V on 35 (23 targets) −0.085 [−0.143, −0.030]; singleton V, clean 22
−0.054 [−0.110, −0.005]; singleton V, 35 −0.036 [−0.075, −0.001]. The
self-distilled model's whole distribution sits closer to the set of
lab-successful recipes than the practice model's does, even though Phase 12
matches the rule-picked set on as many or more targets.

**[E] readings.**
- *Mass vs. distribution.* Under the purity definition Corpus SFT and
  GDPO-300 put the most mass on lab-valid sets (0.26–0.28). Their valid sets
  are mostly the conventional routes, which ASTRAL measured at > 0.5 purity
  on many targets. Their MMD is nonetheless among the worst, because they
  concentrate on one member of V. Phase 12 and E1 have the lowest valid mass
  (0.07) and the largest near-miss distance (0.67).
- *Rank agreement* between MMD ordering and Phase 16's hit ordering on the
  same targets: purity V, Spearman 0.11 (p = 0.80, clean) / 0.05 (35);
  singleton V 0.62 (p = 0.10) / 0.69 (p = 0.056). The distributional metric
  and the best-sample metric do not order the models the same way once V
  contains the lab-successful conventional route. (Hit counts here come from
  these exact samples. On 35 targets: Phase 12 14, E1 14, E3 13, E3c 11,
  RS-SFT 10, Base 7, GDPO-300 2, Corpus SFT 0. Base's 7 differs from finding
  16's 10 because Base's 32-sample draw here is the `resampling_rate` draw,
  not `astral_gen_n32_base.json`.)
- *"The needle is nearby."* 33% of practice samples (GDPO-300, Phase 12,
  E1, pooled; purity V, clean 22) are within one precursor swap of a valid
  set without being in it; E3 37%, Base 31%. Under the singleton V the
  practice share is 4.5% (0.9–6.4% per model) vs E3 11%. Being near a valid set
  is not specific to practice.

---

## 2026-11-15 — 18G (what hidden variable do the channels measure?)

Output: `results/phase18/tensor_latent.json` (+ `results/phase18/channel_scores.jsonl`,
per-channel validator scores for every existing raw completion). Script:
`research/phase18_g_tensor_latent.py`. 52,626 existing raw completions
rescored (12 models, ASTRAL + held-out; the four structured-only ASTRAL files
cannot be scored and are excluded). No new sampling.

**Decisions made before the scores were looked at.** `validator.py` defines no
per-channel pass threshold, so pass = score ≥ 0.9 (the project's validator
bar); [E] sensitivity pass = 1.0. Channels passing on > 99.5% or < 0.5% of
samples are dropped: `operation_order` (99.7%), `temperature_plausible`
(100%), `target_match` (100%). Seven channels are kept; 25,531 complete cases
go to the moment method. **Method change made in development, before the real
data were touched:** the single-view moment method with iterative diagonal
correction failed to recover known parameters on synthetic data, so it was
replaced by the paper's multi-view estimator (§3.3: three views, each one-hot
over its channels' joint configuration), which recovered synthetic k = 2 and
k = 3 models to within ~0.01.

Pass rates: stoichiometry 0.82, amount_accuracy 0.88, thermodynamic_favorable
0.84, charge_neutrality 0.97, target_stability 0.98, chempot_atmosphere 0.97,
precursors_exist 0.99. Lab-outcome samples (recipe = one of its target's two
measured ASTRAL sets): 4,921 samples over 35 targets, 1,855 of them lab
successes (purity > 0.5).

**The moment method finds at most one strong latent direction.** The views'
cross-moment matrices have singular values 0.786, 0.002, 0, 0 (pass ≥ 0.9). So
k = 3 and k = 4 are **not identifiable** by the moment method. At k = 2 it
returns a 96% / 4% split, the minor class being `target_stability` failures
(pass 0.20) with weak `thermodynamic_favorable` (0.54). Every lab-outcome
sample falls into the major component, so the odds ratio is undefined
(Haldane-only, 0.61 [0.17, 1.61]).

**EM latent-class check:**

| k | weights | what the minor classes are | OR (all-pass vs lab success) | 95% CI (target bootstrap) |
|---|---|---|---|---|
| 2 | 0.81 / 0.19 | stoichiometry + amount_accuracy fail together (0.07 / 0.01) | 1.09 | [0.18, 9.55] |
| 3 | 0.69 / 0.17 / 0.15 | + a thermodynamic_favorable-fail class | 0.53 | [0.11, 4.46] |
| 4 | 0.60 / 0.21 / 0.14 / 0.06 | | 0.53 | [0.12, 4.53] |

[E] pass = 1.0 gives the same picture (EM ORs 1.54, 1.09, 0.64; every CI
includes 1; MoM again identifiable only at k = 2).

**P-18G-1 — NOT EVALUABLE under the registered estimator; CONSISTENT under the
EM check.** The prediction (OR CI includes 1) was read strictly as "at every
k, MoM assignment". The MoM gives no usable OR at any k: k = 2 is degenerate
(all lab-outcome samples in one component) and k = 3, 4 are not
identifiable. The script's `passes: false` records exactly that and is **not**
a failed prediction. Under EM, every OR CI includes 1 at k = 2, 3, 4 and both
thresholds, as predicted. The CIs are wide (35 target clusters), so the
result is "no evidence the all-pass class is lab-better", not "evidence it is
not".

**[E] What the shared latent is.** The dominant failure mode the channels
share is the balance solver not closing: `stoichiometry` and
`amount_accuracy` fail together (EM k = 2 minor class). That is a validity
property of the written equation, not lab quality, consistent with the
pre-registered reading. Residual within-component pairwise correlation stays
around 0.14–0.16 (mean |r|), so conditional independence holds only
approximately.

---

## 2026-11-15 — Phase 18 summary (all blocks run)

| block | prediction | outcome |
|---|---|---|
| 0 | inventory | all inputs present, nothing skipped |
| 18A | P-18A-1: transport share higher for own-sample FT than practice | **failed, reversed** (practice higher, CI excludes 0) |
| 18A | P-18A-2: practice new support at floor; S3, S4 above | **failed as worded** (S6 *below* its floor); S3/S4 above ✓; no practice step above its floor |
| 18B | P-18B-1: curl locates comparator errors (AUC > 0.5) | **failed** (AUC 0.48 [0.31, 0.62]) |
| 18C | P-18C-1: practice avoids base principal directions | **failed** (no difference, left side) |
| 18C | P-18C-2: InRank, rank non-decreasing ≥ 75% of modules | **failed** (S6 45%; S7 97%) |
| 18D | R-a′ representation (measure) | **all three judges fail** (K-unit misread; duration ignored; comparator order-sensitive via C7 round-off) |
| 18E | P-18E-1: E3 MMD < Phase 12 | **holds** (−0.092 [−0.168, −0.020]; robust to every cut) |
| 18F | channels vs aggregate (measure) | 3/10 channels survive Holm; none beats the aggregate |
| 18G | P-18G-1: all-pass class not lab-better | **not evaluable by MoM**; consistent under EM |
| 18H | P-18H-1: calibrated R2, R4, R5 decisive ≤ 20% at 90% | **failed (R4 30%)**; R2 9%, R5 0% |
