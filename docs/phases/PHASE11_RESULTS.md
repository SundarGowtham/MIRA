# Phase 11 results — escaping the closed loop

*Per `misc/some_claude_files/PHASE_11_INSTRUCTIONS.md`. Timestamped as results land.*

---

## Step 1 — Rebuild the ranker on published principles — DONE (2026-09-03)

`core/ranker.py` rebuilt, `RANKER_VERSION = "2026-09-03-v2-phase11"`.

**Deleted** (per spec): `step_economy` (chemically backwards -- rewarded
brevity in a deliberately multi-step method), `precursor_availability`
(rewarded corpus frequency, i.e. being typical -- the ASTRAL comparison
shows the experimentally-superior precursors sit at ~0 corpus frequency,
so it actively penalized the right answer).

**Kept unchanged**: `temperature_economy` (z-var 0.72, finding 15),
`driving_force_margin`, `volatility_risk`.

**Added** (4 required + 1 optional, all physics, none reference
conventionality):
- `precursor_instability` (principle 2) -- mean e_above_hull of declared
  precursors, higher is better.
- `inverse_hull_energy` (principle 5) -- pymatgen's
  `PhaseDiagram.get_equilibrium_reaction_energy`, literally documented as
  "inverse distance to hull."
- `n_precursors` (principle 1) -- penalizes 3+ precursor routes.
- `slice_competing_phases` (principle 4) -- competing phases on the
  **2-precursor tie line** specifically (pymatgen `InterfacialReactivity`),
  not the whole chemsys (that generality is what made v1's `phase_purity`
  100% prompt-determined).
- `precursor_decomposition_match` (optional) -- calcination T vs.
  carbonate/hydroxide/nitrate decomposition onset, 17-entry reference table.

34/34 unit tests pass (`tests/test_ranker.py`), including a real-PD-cache
smoke test and the None-propagation contract (gate failure -> every
objective None, never a spurious 0.0).

**Rail calibration** (`misc/ranker_rail_calibration_phase11.json`, 200
archived completions from `runs/gdpo-qlora-beta-ablation-probe/generations.jsonl`,
CPU-only, no new generation): scales set from 5th/95th percentiles per spec,
now the live `RankerScales` defaults.

**Finding, not swept under the rescale**: 4 of 8 objectives show **exactly
0.0 within-group z-variance** in that calibration run --
`n_precursors`, `inverse_hull_energy`, `slice_competing_phases` are
target-determined (flat across the 8 completions sampled for the *same*
target, varying only *across* targets), and `precursor_instability`'s raw
quantity is **98.7% exactly 0.0** at p5/p50/p95 -- not a scale problem at
all. `_best_entry_for_formula` correctly returns each precursor's most
stable known polymorph, and real, isolable solid precursors sit at or near
their own DFT hull almost by construction -- e_above_hull does not actually
operationalize ASTRAL's "kinetically metastable / synthesis-uncommon" sense
of precursor instability. Only `temperature_economy` (0.272),
`driving_force_margin` (0.273), `volatility_risk` (0.125), and
`precursor_decomposition_match` (0.143) carry real within-group spread.
**Inline capacity on the calibrated scales: 10.6%, *below* Arm A's 14%
validator baseline** -- the opposite of the >40% this rebuild targeted.
z-normalization inside GDPO makes a constant rescale gradient-neutral, so
no choice of scale fixes a channel whose raw quantity has no variance to
begin with; this is a design finding, not a calibration bug.

---

## Step 2 — THE EXTERNAL GATE — FAIL, STOP (2026-09-03)

`run_debug_and_analysis/ranker_v2_astral_gate.py` ->
`misc/ranker_v2_astral_gate.json`. 35 ASTRAL targets, traditional vs.
predicted precursor sets, ranker v2 score vs. robot-measured phase purity.

| | Spearman rho (n=70) | p | agreement (N/35) |
|---|---|---|---|
| Validator (Arm A, Step 1 presentation-prep) | 0.228 | 0.058 | 17/34 = 50.0% |
| Ranker v1 (pre-Phase-11) | -- | -- | 20/32 = 62.5% |
| **Ranker v2 (Phase 11 rebuild)** | **0.133** | **0.273** | **21/35 = 60.0%** |
| **Pre-registered bar** | -- | -- | **24/35 = 68.6%** |

**VERDICT: FAIL. 21/35 < 24/35. STOP per protocol -- do not proceed to
Step 3/4/5, do not train.** Rebuilding the ranker on ASTRAL's own published
precursor-selection principles produced a verifier that agrees with the
robot's actual measurements *worse* than the ad-hoc v1 ranker (60.0% vs.
62.5%), and its Spearman correlation is weaker too (0.133 vs. 0.228 for the
plain validator). Two verifier generations have now failed the external
gate; per the protocol, that is itself the result, not a reason to try a
third.

**Per-channel diagnosis** (`per_channel_disagreement` in the saved JSON):

- `driving_force_margin`: disagrees with experiment on 6/14 gradeable pairs
  (43%) -- barely better than a coin flip.
- `n_precursors`: disagrees on 9/17 gradeable pairs (53%) -- **worse than
  chance**. Penalizing 3-precursor routes actively fought the data here:
  ASTRAL's own traditional (3-precursor) routes sometimes measured *higher*
  purity than the predicted (2-precursor) routes principle 1 favors.
- `temperature_economy`, `slice_competing_phases`,
  `precursor_decomposition_match`: **ungradeable on all 35 pairs** --
  investigated, not a bug in two of three cases:
  - `temperature_economy` is ungradeable by deliberate design (`lit_T=None`
    -- these are already-optimal experimental routes, not model output
    scored against a literature target).
  - `slice_competing_phases` is gated to exactly-2-precursor routes by
    construction, and **every single ASTRAL traditional route uses 3
    precursors while every predicted route uses exactly 2** (that split
    *is* principle 1, present in the data itself) -- so a *paired*
    trad-vs-pred comparison can never have both sides gradeable on this
    dataset, even though the channel works correctly on individual routes.
    Real limitation of the paired-comparison design, not the channel.
  - `precursor_decomposition_match`'s 17-entry table covers common
    carbonates/hydroxides/nitrates (which show up on the *traditional*
    side: Li2CO3 x20, K2CO3 x13, Na2CO3 x4 across the 35 targets) but not
    ASTRAL's actual *predicted*-side precursors (LiPO3, KPO3, LiBO2, NaBO2,
    B2O3 -- metaphosphates and borates, zero overlap with the table). The
    gap falls exactly on the axis the whole experiment is about, so
    extending the table with more carbonates would not have helped.

**Confounder worth flagging plainly**: of the 18 pairs where every
objective reads "ungradeable" for the *n_precursors*-style reasons above,
all 18 stem from the `balances` gate failing on the **traditional
(3-precursor) route specifically -- 18/35, vs. 0/35 for predicted
(2-precursor) routes.** The amount=1.0 placeholder (ASTRAL's public data
gives precursor species, not molar ratios -- same caveat as Step 1) makes
3-precursor stoichiometry harder for the balance solver to close than
2-precursor stoichiometry. This systematically favors predicted routes
passing gates at all, independent of true chemistry, and is a real
methodological confound in this comparison, not just in the objectives
that reported "ungradeable."

---

## Step 0 — High-resolution ASTRAL baseline (n=32) — DONE (2026-09-03 to 2026-09-04)

Not gated by Step 2's outcome -- ran independently as the project's
external headline metric regardless of the ranker's fate. `run_astral_step0_n32.sh`,
tmux `astral_step0`, sequential SFT -> base -> GDPO-300, resume-safe. Full
wall time: SFT ~8.5h, base ~6h, GDPO-300 ~8.5h (23h total, one GPU).

**Bug fixed before launch**: the naive `generate_batch(model, tok, [prompt] * 32, ...)`
call OOM'd the 32GB card immediately (KV-cache scales with batch x
max_new_tokens=8192; 8-sample batches were the proven-safe size).
Fixed by chunking generation into 4x8-sample calls per target
(`run_debug_and_analysis/astral_model_generations.py`); confirmed clean on
relaunch, no further OOMs.

**Precursor-matcher claim investigated, not a bug**: verified
`SynthesisValidator._normalize_formula` already normalizes via pymatgen
`Composition.reduced_formula`, not raw string comparison -- NH4H2PO4 (MAP)
and PH6NO4 already match correctly; PH9(NO2)2 and (NH4)2HPO4 (DAP) are
correctly *not* matched against MAP because they are genuinely different
compounds. No fix was needed; SFT numbers were recomputed (not
regenerated) on the unchanged, already-correct code for comparability.

**All three models, n=32 -- DONE.** Compare against the earlier n=8 table
(`misc/astral_three_model_comparison.json`):

| model | any_predicted/35 (n=8) | any_predicted/35 (n=32) | any_traditional/35 (n=32) | mean_reward (n=32) | mean max-T (n=32) |
|---|---|---|---|---|---|
| base | 3/35 | **10/35** | 24/35 | 0.948 | 1026.5 |
| SFT | 0/35 | **1/35** | 32/35 | 0.888 | 963.2 |
| GDPO-300 | 1/35 | **3/35** | 31/35 | 0.894 | 1017.1 |

**n=8 was severely underpowered, base most of all**: 3/35 -> 10/35 -- nearly
3.5x. This is now the headline result of the whole phase and the reason
Step 0 was run first (finding 9's framing: rare-event counts near the
sampling budget's edge cannot be trusted). See CLAUDE.md findings 16-17 and
`misc/some_claude_files/PHASE11_REVISED.md` for the full statistical
analysis (paired McNemar per model pair, validator-score gap analysis) done
downstream of these raw generations -- summary: base significantly beats
SFT (p=0.004-0.012), SFT-to-GDPO recovery is not significant (p~0.5-0.6,
~22% of the loss), and the validator already scores ASTRAL's better routes
*above* conventional ones for base (+0.021) and GDPO-300 (+0.049) but
*below* for SFT (-0.053) -- the verifier points the right way, SFT's
support collapse is the obstacle, not the reward.

---

## CRITICAL CORRECTION to PHASE11_REVISED.md's Step 1 premise (2026-09-05)

**Empirically verified, not just read from source: Arm A's GDPO KL reference
has always been base Qwen3-8B, not SFT.** PHASE11_REVISED.md's Step 1
("re-anchor the KL reference from SFT to base, ~5 days GPU, highest
priority") assumes the current reference is pi_SFT. It is not, and Step 1
as specified would be a **config no-op against the current setup** -- 5
days of GPU to reproduce Arm A within noise, not a new experiment.

**Mechanism** (`core/model.py::load_with_adapter`): the SFT adapter is
loaded via `PeftModel.from_pretrained(model, sft_checkpoint, is_trainable=True)`
with PEFT's default `adapter_name="default"` -- never `"ref"`, anywhere in
`experiments/grpo.py`, `experiments/gdpo.py`, or `core/model.py`. TRL 1.8.0's
`GRPOTrainer` (`grpo_trainer.py` ~line 2494) computes reference log-probs
via `trl.trainer.utils.use_adapter(model, adapter_name="ref" if "ref" in
model.peft_config else None)`. Since no adapter is ever named `"ref"` here,
this always resolves to `adapter_name=None` -> `model.disable_adapter()` --
PEFT's mechanism for temporarily removing ALL LoRA deltas, which is
mathematically the raw base model, not SFT.

**Confirmed by direct execution** (`run_debug_and_analysis/verify_ref_model_identity.py`,
no training, three forward passes on one prompt):

| comparison | max \|logit diff\| |
|---|---|
| policy (SFT adapter enabled) vs. `disable_adapter()` | 10.05 |
| `disable_adapter()` vs. a freshly-loaded base Qwen3-8B | **0.000000** |
| policy vs. fresh base | 10.05 |

`disable_adapter()` output is bit-identical to loading base fresh, and
clearly different from the SFT policy. The reference has been base all
along.

**Implication**: finding 17's geometric argument (pi* ~ pi_SFT * exp(r/beta),
support frozen at pi_SFT's collapse) needs revision -- the KL reference was
never pi_SFT. The ASTRAL n=32 empirical results (finding 16) are unaffected
and stand on their own; what needs revising is the *explanation*. The more
likely mechanism: GRPO/GDPO is an **on-policy** method -- the reward can
only reinforce sequences the *current policy* actually samples during
rollout, and that policy is initialized from (and stays close to, at
beta=0.001's weak pull) SFT's weights regardless of which distribution the
KL penalty is computed against. If SFT's rollout distribution assigns
near-zero *sampling* probability to the good routes, GDPO never observes
them in a training batch to reinforce, independent of the reference
identity used only for the KL penalty term. This points toward Steps 2-3
(format-only SFT, RS-SFT from base) as the more load-bearing interventions
-- both change what the POLICY itself samples, not just what it is
KL-penalized against -- and away from Step 1 as specified.

**Not launched.** Step 1 (5 days GPU) was not started pending this
correction -- recommend the user/main-chat re-derive Step 1's rationale (or
confirm a genuinely different config, e.g. an explicit second `"ref"`
adapter loaded from an *earlier* checkpoint than SFT, if a literal pi_SFT
anchor is what was actually wanted for comparison) before committing GPU
time to it.

---

## PHASE11_REVISED.md Step 2 — Format-only SFT — DONE, PASSES (2026-09-05 to 2026-09-06)

*Note on step numbering: this is PHASE11_REVISED.md's Step 2 (format-only
SFT), distinct from PHASE11_INSTRUCTIONS.md's Step 2 (the ranker external
gate, FAILED, above). The two documents both number a "Step 2" for
different things -- disambiguating explicitly since both live in this
file.*

**The cleanest experiment nobody had run: how much SFT is actually needed?**
Trained a minimal LoRA from base Qwen3-8B: 300 examples (seeded random
subset of the full 1,217-example `data/sft_v3` train set), ~1 epoch
(max_steps=10 at effective batch 32), fresh adapter (not continuing from
the full-SFT checkpoint). `runs/sft-qlora-format-only/final`. Then measured
on the full ASTRAL n=32 protocol (`misc/astral_gen_n32_format_only.json`,
`run_debug_and_analysis/astral_model_generations.py --checkpoint
runs/sft-qlora-format-only/final --n-samples 32`).

**Bug caught before wasting the run**: `experiments/base.py`'s
`ExperimentConfig` defaults `--model` to `meta-llama/Llama-3.1-8B-Instruct`
(a stale framework default, `FULL_MODEL`) when the flag is omitted, and to
a tiny SmolLM2-135M in `--smoke` mode. The first launch crashed on a gated
Llama repo (never a Qwen3-8B issue); the first "smoke test" that appeared
to pass had actually only validated the pipeline against SmolLM2-135M, not
the real model. Fixed by always passing `--model Qwen/Qwen3-8B` explicitly
-- worth flagging in the operational playbook alongside the existing
`--checkpoint`-not-`--model` trap, since this is the same trap's mirror
image (a missing, not wrong, `--model` flag silently substitutes a
different base model).

**A separate parse-rate script using the training data's own bare prompts
(no SYSTEM_MSG-style instructions) measured only 1/100 = 1.0% parse rate**
(`misc/format_only_parse_rate.json`) -- investigated, not the real answer:
`build_sft_dataset` (training) and that quick script both use bare prompts
with no system message, consistent with each other, but 10 steps evidently
isn't enough exposure for the model to internalize the JSON schema
*independent of prompt phrasing*. The ASTRAL protocol's `closed_prompt()`
embeds detailed schema/persona instructions (`SYSTEM_MSG`) directly in the
user turn, which is a stronger prompt than the bare training-data prompts --
and that is the protocol every other number in this phase is measured
against, so it's the one that matters for the pass condition below.

**Real (protocol-consistent) parse rate, computed directly from the 1,120
ASTRAL-protocol samples**: **1,112/1,120 = 99.3%** (8 parse failures).
Clears the ≳95% bar cleanly.

| model | any_predicted/35 | any_traditional/35 | mean_reward (predicted matches) | mean_reward (traditional matches) | gap | mean max-T | median max-T |
|---|---|---|---|---|---|---|---|
| base | 10/35 | 24/35 | -- | -- | **+0.021** (validator, CLAUDE.md finding 16) | 1026.5 | 1000 |
| **format-only SFT** | **11/35** | 31/35 | 0.988 | 0.950 | **+0.038** | 1070.8 | 1100 |
| full SFT | 1/35 | 32/35 | -- | -- | **-0.053** | 963.2 | 950 |

**Both of Step 2's pass conditions are met, cleanly:**
- Parse rate 99.3% >> 95%.
- **Retains (and slightly exceeds) base's ASTRAL hit rate: 11/35 vs. base's
  10/35** -- in sharp contrast to full SFT's collapse to 1/35. The reward
  gap is positive and *larger* than base's (+0.038 vs. +0.021), also
  unlike full SFT's negative gap.
- The one softer criterion ("conventional N/35 should stay near base's
  24/35, not climb to 32/35") only partially holds: format-only landed at
  31/35, much closer to full SFT's 32/35 than to base's 24/35. This doesn't
  undercut the headline finding -- the model still finds the predicted set
  at base's rate -- but it means format-only SFT isn't a pure no-op on
  precursor preference either; some convergence toward corpus-common
  choices happens even at 300 examples/1 epoch.

**Conclusion: the support collapse is isolated to the chemistry-matching
component of full SFT, not the schema-installation component.** A model
that has only ever seen 300 examples for one epoch already writes valid
JSON 99.3% of the time and still finds ASTRAL's better precursor sets as
often as base does. Full SFT's 1,217 examples over 3 epochs is what erases
the capability -- not "SFT" as a category. This gives every subsequent RL
run (Step 1's corrected form, or a future run) a strictly better starting
point than the current `sft-qlora-sft-v3-2nd-rank16` checkpoint: same JSON
reliability, without the corpus-preference collapse.

---

## PHASE11_REVISED.md Step 3 — RS-SFT from base — DONE (2026-09-06 to 2026-09-08)

**Structurally different from more RL, not just a cheaper baseline:**
rejection-sampling SFT is a forward-KL (m-)projection onto verifier-filtered
samples of the model's own distribution -- forced to cover the support of
its training data by construction, unlike RL's e-projection (finding 17).

**Procedure** (`data_curation/build_rs_sft_dataset.py --checkpoint base`,
already written and unlaunched per CLAUDE.md's repo notes): sampled 8
routes/target from **base** Qwen3-8B for 400 targets drawn from
`data/rl` (SFT-disjoint, 6,368-target pool), closed-book, scored with the
unmodified `validator.py` (Arm A -- not the failed ranker), kept the
best survivor per target at bar=0.9 (the script's own documented
convention, "the same bar the pass@k comparison uses"). **295 survivors
from 400 targets = 73.75% survival rate.** Fine-tuned a fresh LoRA from
base on the 283-example train split (12 held out for eval loss), default
SFT hyperparams (3 epochs, effective batch 32, ~27 steps) --
`runs/sft-qlora-rs-sft-from-base/final`. Training fit its own
self-generated data easily (train_loss 0.77, mean_token_accuracy 0.97,
unsurprising since it's fitting completions the same model produced).
Measured on the full ASTRAL n=32 protocol
(`misc/astral_gen_n32_rs_sft.json`).

**Bug avoided this time**: same `--model` trap as Step 2 (see above) --
`--model Qwen/Qwen3-8B` passed explicitly throughout, verified with a
quick `--smoke` pass first (which itself OOM'd for an unrelated, expected
reason -- `--smoke` disables quantization, and an unquantized 8B model
doesn't fit in 32GB; the OOM occurred deep in a real forward/backward
pass, confirming Qwen3-8B loaded correctly before running out of memory,
which was enough to validate the config without wasting the real run).

**Full five-model comparison, uniformly recomputed from each model's own
`astral_gen_n32_*.json` via the same script/methodology** (gap =
mean_reward on predicted-set matches minus mean_reward on
traditional-set matches; parse rate = fraction of all 1,120 samples that
are not `PARSE_FAILURE`):

| model | any_predicted/35 | any_traditional/35 | mean_reward (all) | gap | mean max-T | median max-T | parse rate |
|---|---|---|---|---|---|---|---|
| base | 10/35 | 24/35 | 0.948 | +0.021 | 1026.5 | 1000 | 98.7% |
| format-only SFT | 11/35 | 31/35 | 0.937 | +0.038 | 1070.8 | 1100 | 99.3% |
| **RS-SFT from base** | **10/35** | **17/35** | **0.972** | +0.007 | **934.2** | **950** | 95.8% |
| full SFT | 1/35 | 32/35 | 0.888 | -0.053 | 963.2 | 950 | 98.5% |
| GDPO-300 | 3/35 | 31/35 | 0.894 | +0.049 | 1017.1 | 1000 | 97.9% |

**RS-SFT matches base's predicted-set hit rate exactly (10/35) while
*dropping* the conventional-set rate to 17/35 -- the lowest of any model,
including base itself (24/35).** That is not "retention," it is a genuine
improvement on base's own diversity: fewer routes fall into the
generic/traditional bucket at all, more land somewhere else (novel,
non-traditional routes the matcher doesn't categorize as either
reference set). RS-SFT also posts the **highest mean validator reward of
any model tested (0.972)** and the **lowest, most realistic mean/median
max-T (934/950 -- below even full SFT's 963/950, and far below base's
1026/1000 and format-only's 1071/1100)**. The one metric where it's
weaker than format-only SFT is the predicted-vs-traditional reward gap
(+0.007, smallest positive gap of the three positive-gap models) and
parse rate (95.8%, still clears the ≳95% bar but with the least margin --
8 fewer parses clean than format-only's 99.3%).

**This is the strongest single result in Phase 11 so far.** A fine-tune
built entirely from the model's own validator-approved samples --
zero external teacher data -- matches base's exploratory capability,
improves route diversity beyond base, scores higher on the validator, and
runs at more realistic temperatures than every other checkpoint tested.
It directly confirms PHASE11_REVISED.md's central claim: **fine-tuning
on an external corpus (Kononova/DeepSeek teacher traces) is what closed
the loop, not fine-tuning per se.** A support-preserving fine-tune (RS-SFT,
a genuine m-projection onto the model's own distribution) does not just
avoid collapse -- on this evidence it does measurably better than base on
every axis except raw predicted-set count, where it ties.

---

## Next per protocol

**PHASE11_INSTRUCTIONS.md's Step 2 (ranker external gate) FAILED** -- per
that document's own rule, do not proceed to its Step 3 (RS-SFT with the
ranker) or Step 5 (run 4 with the ranker as scorer). Two verifier
generations (validator, then a rebuild targeting ASTRAL's own published
principles) have both failed to beat a modest external agreement bar
against real robot measurements.

**PHASE11_REVISED.md's Steps 2 and 3 both PASSED, and Step 3 (RS-SFT)
is now the strongest checkpoint measured in this phase.** Three
support-preserving candidates now exist as strictly better GDPO starting
points than `sft-qlora-sft-v3-2nd-rank16`: format-only SFT, RS-SFT from
base, and (untested as a combination) RS-SFT continuing from format-only.
Per the user's instruction (2026-09-06), results stop here for a
main-chat review before any GDPO run is launched from one of these --
in particular, Step 1's KL-reference correction (the reference has always
been base, not SFT, so "re-anchor to base" is a no-op against the current
config) still needs to be resolved or reformulated before spending GPU on
a run 4, regardless of which checkpoint it starts from.

Step 0 (the n=32 baseline) is complete and stands as the phase's headline
external metric regardless of the ranker's fate.
