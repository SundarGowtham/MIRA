# Phase 16 results

Per `misc/PHASE16_INSTRUCTIONS.md`. Dated entries as they land.

## 2026-10-03 — §1 rule 1 check FAILED: pre-existing ASTRAL contamination in active training data

**Check that failed**: the mandated unit test (`tests/test_no_astral_circularity.py`,
§1 rule 1 — "no ASTRAL target formula appears in any training or sampling
prompt file") fails. Per the instructions' own protocol ("if a check
fails, stop... and wait for Gowtham"), stopping here rather than
proceeding to E3 sampling or anything else that depends on this data.

**This is a pre-existing condition, not something Phase 16 introduced.**
The contaminated files were all built well before this phase (SFT v3,
RL run3, and the general RL pool all predate Phase 16's own changes).
The rule-1 test is simply the first time anyone has checked for it.

**Exact scope, by file** (contaminated records / total, which ASTRAL
target(s)):

| file | contaminated / total | targets |
|---|---|---|
| `data/rs_sft/rs_sft_train.jsonl` | **0 / 283** | — |
| `data/rs_sft/rs_sft_val.jsonl` | **0 / 12** | — |
| `data/rl_run3/rl3_train.jsonl` | 0 / 1200 | — |
| `data/rl_run3/rl3_val.jsonl` | 1 / 200 | `Na2Al2B2O7` |
| `data/rl_run3/rl3_probe.jsonl` | **1 / 30** | `Na2Al2B2O7` |
| `data/sft_v3/{train,sft_train}.jsonl` | 3 / 1217 each | `KTiNbO5`, `LiMnPO4`, `Li3V2(PO4)3` |
| `data/sft_v3/format_only_train.jsonl` | 2 / 300 | `KTiNbO5`, `LiMnPO4` |
| `data/sft_v3/{val,test,sft_val,sft_test,format_only_val}.jsonl` | 0 each | — |
| `data/rl/{train,rl_train}.jsonl` | 9 / 6368 each | `Li3Fe2(PO4)3`, `NaSrBO3`, `KBaPO4`, `LiMgPO4`, `LiZnBO3`, `Li3Sc2(PO4)3`, `KNbWO6`, `Li2CuP2O7`, `KMgPO4` |
| `data/rl/{val,rl_val}.jsonl` | 1 / 200 | `Na2Al2B2O7` |

**The single most consequential dataset for Phase 16 — `data/rs_sft/` itself,
RS-SFT's actual training set, which every Phase 16 arm (E1/E2/E3/E4) either
inits from or distills further from — is clean, 0/295 contaminated.** The
Phase 11/12/16 RS-SFT lineage was not trained on any ASTRAL target.

**What IS affected**: full SFT (`data/sft_v3`, 3/1217 ≈ 0.25%, used in the
Phase 10/11 "full SFT" comparison arm) and, more notably, the **fixed
30-target probe set** (`data/rl_run3/rl3_probe.jsonl`, 1/30 = **3.3%** —
`Na2Al2B2O7`). This probe set is re-evaluated every N steps throughout
every GDPO run (Phase 12 and the planned Phase 16 E1/E2/E4 runs) to
produce the per-check training-health curves `CLAUDE.md` and
`docs/phases/PHASE12_RESULTS.md` report from. One target out of 30 being
an ASTRAL target is a real data-hygiene gap, not previously flagged
anywhere in this project, though its effect on any already-reported
32-target-averaged probe metric is necessarily small (1/30 weight).

**Not fixed or filtered here.** Removing records from datasets that
already produced completed, reported Phase 11/12 results is a decision
with real retroactive implications (does this change how those numbers
should be read, does the probe set need to be rebuilt and old curves
re-labeled, etc.) that is Gowtham's call, not something to decide
unilaterally mid-Phase-16. `tests/test_no_astral_circularity.py` is left
failing honestly rather than narrowed in scope to pass — scoping it to
only the directories real experiment code reads (not literally every
`.jsonl` in the repo, which also turned up one confirmed-dead legacy file,
`data/processed/reasoning_traces.jsonl`, excluded on that verified basis)
already reflects the real training/sampling surface; the remaining
failures are real.

ntfy sent to `mira-g5x7k2-status` at the time of this finding.

**Everything else in pass 2 items 1-2 (mira-vllm rebuild, standalone
smoke test, `core/novelty_reward.py` + its own tests) passed and is
unaffected by this** — it is the project-wide circularity guardrail
specifically that is failing, not novelty_reward.py's own tests (all of
which pass). Pending Gowtham's decision on this before any pre-registration
drafting or data-touching step proceeds.

## 2026-10-03 — resolved: this was test-set contamination, not rule-1 circularity. Erasure and carbonate findings survive on the clean subset.

**Gowtham's correction, confirmed by direct re-check, not assumed**: rule 1
forbids training on ASTRAL's *answers* (its precursor sets or its
principles), not a target formula merely appearing as a prompt. The
schemas of the contaminated files settle this directly —
`data/rl_run3/*.jsonl` and `data/rl/*.jsonl` have no `completion`/
`response` field at all (prompt-only; the model generates on-policy, there
is nothing fixed to memorize). Only `data/sft_v3` stores a real answer.

**The test was split, per instruction**:
- `tests/test_no_astral_answer_leakage.py` — the strict gate. **PASSES.**
  No `core/`/`data_curation/` module references an ASTRAL path, and zero
  training records anywhere have a stored answer equal to ASTRAL's
  rule-picked ("predicted") set for their target.
- `scripts/astral_overlap_report.py` → `results/phase16/astral_overlap.json`
  — informational, not gated. Also asserts zero overlap for any dataset
  Phase 16 itself creates (vacuously true today; nothing created yet).
- The old combined test moved to `research/test_no_astral_circularity_superseded.py`
  (never deleted, per project convention), with a header explaining the split.

**Full overlap classification** (dataset, target, role, match-against-ASTRAL):

| dataset | target | role | matches |
|---|---|---|---|
| `sft_v3_train`/`sft_v3_sft_train` | `KTiNbO5` | prompt+answer | **traditional** (exact) |
| `sft_v3_train`/`sft_v3_sft_train` | `LiMnPO4` | prompt+answer | neither |
| `sft_v3_train`/`sft_v3_sft_train` | `Li3V2(PO4)3` | prompt+answer | neither (close: `(NH4)2HPO4` vs traditional's `NH4H2PO4`) |
| `sft_v3_format_only_train` | `KTiNbO5`, `LiMnPO4` | prompt+answer | same as above (shared records) |
| `rl_run3_val` | `Na2Al2B2O7` | prompt-only | — |
| `rl_run3_probe` | `Na2Al2B2O7` | prompt-only | — |
| `rl_run3_train` | — | — | **0 overlap, confirmed explicitly, per instruction item 2** |
| `rl_train`/`rl_rl_train` | 9 targets (`Li3Fe2(PO4)3`, `NaSrBO3`, `KBaPO4`, `LiMgPO4`, `LiZnBO3`, `Li3Sc2(PO4)3`, `KNbWO6`, `Li2CuP2O7`, `KMgPO4`) | prompt-only | — |
| `rs_sft_train`/`rs_sft_val` | — | — | **0 overlap** |

**Zero records anywhere match ASTRAL's rule-picked set** — the only
non-trivial match is `KTiNbO5`'s exact match to the *conventional*
(traditional) literature recipe, a genuine, literal memorization case.
`LiMnPO4` and `Li3V2(PO4)3` do not exactly match either ASTRAL set.

**Union of all overlapping targets across every dataset: 13/35** —
`KBaPO4, KMgPO4, KNbWO6, KTiNbO5, Li2CuP2O7, Li3Fe2(PO4)3, Li3Sc2(PO4)3,
Li3V2(PO4)3, LiMgPO4, LiMnPO4, LiZnBO3, Na2Al2B2O7, NaSrBO3`. Clean
subset: **22/35**.

### Clean-subset sensitivity analysis (`research/phase16_astral_clean_subset.py` → `results/phase16/astral_clean_subset.json`)

**No headline conclusion changes.** Every number moves by the expected
small amount from removing ~37% of targets; none flips sign, none loses
significance in a way that changes what's claimed.

**1. Rule-picked hits, full-35 vs clean-22:**

| model | full-35 | clean-22 |
|---|---|---|
| base | 10/35 (28.6%) | 8/22 (36.4%) |
| full SFT | 0/35 | 0/22 |
| RS-SFT | 10/35 (28.6%) | 9/22 (40.9%) |
| GDPO-phase12-β0 | 14/35 (40.0%) | 12/22 (54.5%) |

**2. base → full SFT erasure**: full-35 base-only=10, sft-only=0,
McNemar p=0.0020 (finding 16/18's own number). **Clean-22: base-only=8,
sft-only=0, McNemar p=0.0078.** Still fully unidirectional, still
significant. **This directly answers the memorization concern**: the
erasure is not reliant on the 13 overlapping targets (including
`KTiNbO5`, the one literal-memorization case) — removing them, the same
base→SFT collapse pattern survives at essentially the same strength.

**3. RS-SFT → Phase 12 GDPO amplification**: full-35 gained=4
(`BaLiBO3`, `KLi(PO3)2`, `Li2TiSiO5`, `NaSrBO3`), one-sided p=0.0625
(finding 22's own number). **Clean-22: `NaSrBO3` is itself one of the 13
overlapping targets (prompt-only, in `data/rl` and `rl_run3`'s val/probe
— not an answer), so it drops out: gained=3 (`BaLiBO3`, `KLi(PO3)2`,
`Li2TiSiO5`), one-sided p=0.125.** The *direction* (RS-SFT 9/22 → GDPO
12/22, zero losses) is unchanged and, proportionally, slightly stronger
on the clean subset than on the full 35 — but note `NaSrBO3` was also
separately flagged (Task 1 finding 1.3) as "the one robust,
noise-exceeding GDPO-specific gain" (0/32→6/32). That claim rests on a
target that happens to be prompt-contaminated (not answer-contaminated)
in the general RL pool — worth keeping in mind when citing it, though
prompt-only overlap does not itself undermine a behavioral claim about
sampled generations. No pre-registered bar exists for the 22-target
subset (the `>12/35` amplification bar was set against the full 35), so
no new pass/fail claim is made here — only the side-by-side numbers.

**4. Carbonate/bare-oxide/other alkali-source shares** — virtually
unchanged full-35 vs clean-22 for every model:

| model | bare_oxide (full→clean) | carbonate (full→clean) |
|---|---|---|
| base | 0.091 → 0.097 | 0.729 → 0.734 |
| full SFT | 0.005 → 0.008 | 0.982 → 0.983 |
| RS-SFT | 0.258 → 0.282 | 0.459 → 0.440 |
| GDPO-phase12-β0 | 0.569 → 0.628 | 0.181 → 0.151 |

The base→SFT→RS-SFT→GDPO carbonate-to-bare-oxide shift (finding 24) is
not an artifact of the overlapping targets.

**Decision, per instruction**: `data/rl_run3` stays unchanged (new arms
stay comparable with Phase 12). For every arm, Phase 12 included, the
**clean-22-subset ASTRAL metrics are now primary**; full-35 metrics are
reported alongside as secondary. The E3/E3c target pool (from
`data/rs_sft`, 0/295 overlap) already excludes all ASTRAL targets per
§6.1, independent of this analysis.

### Environment guard (the mis-install correction)

Added to `train.py`: `_assert_main_environment()`, called at the top of
`main()`, raises immediately if `sys.prefix` isn't the main project
`.venv` — verified it fires correctly both ways (passes under the main
env, raises under `mira-vllm`'s python). Standing rule for all future
Phase 16 code: any script touching `mira-vllm` invokes
`~/envs/mira-vllm/bin/python` by absolute path, never bare `python` or
`uv pip` without an explicit environment target. End-of-pass check run
and confirmed: `uv sync --frozen` reports zero changes in the main
environment ("Checked 165 packages"), and all 6 test suites
(`test_validator`, `test_ranker`, `test_comparator`, `test_reward`,
`test_novelty_reward`, `test_no_astral_answer_leakage`) pass.

**Next**: draft `docs/phases/PHASE16_PREREG.md` with the clean-subset
metrics as primary, and stop for Gowtham's approval — per instruction.

## 2026-10-06/07 — E3/E3c pool, filter, and RS-SFT reference samples

**Pool (§6.2)**: 23,040 raw samples from base Qwen3-8B (360 training
targets x 64), via offline vLLM in mira-vllm. Smoke run (2x4) passed
cleanly before the full launch. Full-run quality check (not just the
8-sample smoke): parse rate 97.7%, mean length 4,847 tokens, clip rate
2.4% — all healthy, in line with Phase 12's own ~4,960-token mean.

**E3/E3c filter (§6.3/6.4)**, `data_curation/build_e3_e3c_filter.py`:

| metric | value |
|---|---|
| gate pass rate (of 23,040) | 52.7% |
| novelty rate among gate-passers | 54.7% |
| targets keeping >=1 sample | 256/360 (71.1%) |
| targets keeping 0 samples | 104/360 (28.9%) |
| **E3 final size** | **527** |
| E3c pool before subsample | 686 |
| **E3c final size** | **527** (seed-42 matched) |

**Stop rule did not trigger** (bar: E3 >= 150, >50% targets empty) --
final training-ready files written. One operational note: this CPU-only
scoring pass took far longer than its 50-sample dry run predicted
(~45 min vs an extrapolated ~18 min), confirmed NOT hung throughout
(98.5% CPU, `ps etime` matched accumulated CPU time) -- traced to two
pathological cases logged by `gibbs_corrector` (`NdIr3` at 2473K, `HCl`
at 300K) hitting a slow fallback path. Not killed/retried since the
script holds all state in memory with no incremental checkpointing --
worth adding if a similar pass needs to run again.

**RS-SFT reference samples** (scheduled by regular Claude: "E1's support
test compares against RS-SFT's own 200+32 samples per target... schedule
them after the filter, before E1 is evaluated"), `data_curation/
phase16_rssft_reference_sampling.py`: 17,400 completions (40 held-out +
35 ASTRAL targets x 232), via vLLM LoRA serving of the RS-SFT adapter.
One bug hit and fixed: loading the tokenizer from the checkpoint
directory crashed (`AttributeError` in transformers' special-tokens
handling -- mira-vllm's downgraded transformers 4.57.6 couldn't parse
that checkpoint's saved tokenizer_config.json format); fixed by loading
the tokenizer from base instead, since a LoRA adapter does not change it.

Quality check, and a notable cross-check against base's own reference
samples on the identical 75 targets:

| set | n | parse rate | mean length | clip rate | carbonate share |
|---|---|---|---|---|---|
| RS-SFT, held-out | 9,280 | 93.8% | 5,163 | 6.3% | **23.9%** |
| RS-SFT, ASTRAL | 8,120 | 97.0% | 5,272 | 3.0% | **48.2%** |
| base, held-out | 9,280 | 97.9% | 4,779 | 2.2% | 36.8% |
| base, ASTRAL | 8,120 | 99.3% | 4,977 | 0.8% | 74.8% |

RS-SFT's carbonate share is already well below base's on this fresh,
independent 232-sample draw (never computed before at this n) -- an
independent corroboration of the standing carbonate-to-bare-oxide shift
(Phase 15 findings 24/26) using new data on both target sets, not just
a repeat of the existing n=32 ASTRAL result.

## 2026-10-08 — vLLM colocate does not fit GDPO training on manifold's 32GB card

**What was tested**: whether `--use-vllm on` (colocate mode) can
actually run a real GDPO training step on manifold, not just the
offline sampling it already works for. The instructions' own prescribed
test (`--use-vllm on --smoke`) was tried first and is **not a valid
test**: `core/model.py::load_model`'s own docstring says "smoke disables
quantization for CPU" -- in `--smoke` mode the model loads as full fp32
with no device_map, bypassing QLoRA's 4-bit quantization entirely
regardless of `--adapter qlora`. An fp32 8B model alone is ~32GB, which
is why that attempt failed immediately with "Free memory on device
cuda:0 (0.09/31.36 GiB)" before vLLM even requested its slice -- a
smoke-mode artifact, not a finding about vLLM or QLoRA.

**Corrected test**: a real (non-smoke) `--max-steps 2` run, Phase
12's exact hyperparameters (QLoRA r=16/alpha=32, G=8, batch=1, accum=16,
max_completion_length=8192), `--use-vllm on`, from the RS-SFT
checkpoint. Two attempts, bracketing the failure from both sides:

| `--vllm-gpu-memory-utilization` | result |
|---|---|
| 0.3 (~9.4 GiB for vLLM) | Both models loaded and **generation ran successfully** (confirmed: the LoRA-merge-into-vLLM warning fired, meaning TRL's weight-sync path engaged). GPU at 19 GiB/32 GiB mid-run. **OOM'd in the training backward pass**: "Tried to allocate 4.64 GiB... 4.25 GiB free... 27.10 GiB in use." |
| 0.12 (~3.8 GiB for vLLM) | vLLM's own engine init failed before training even started: "No available memory for the cache blocks." |

**Conclusion: there is no `vllm_gpu_memory_utilization` split on this
32GB card that leaves both vLLM's minimum KV cache and the training
backward pass's peak memory satisfied, at Phase 12's real
hyperparameters.** This confirms §2.2's own anticipated outcome
("may not fit on manifold's RTX 5090... should fit on a 48GB card").
Per the instruction's own directive, this is reported rather than
forced by shrinking `max_completion_length` or batch/accum/G. **E1 (and
any other Phase 16 GDPO arm run on manifold) uses `--use-vllm off`
(HF generate) -- which is already the default**, so no change to how
E1 launches. vLLM's proven, working use on this machine remains the
offline sampling path (E3/E3c pool, base and RS-SFT reference samples),
not colocated training.

One sub-question left genuinely open: whether TRL's weight-sync
mechanism (does step N+1's generation actually reflect step N's
gradient update, or silently keep sampling from a stale policy) produces
*correct* updated generations could not be fully verified, since no
training ever reached a completed optimizer step under colocate -- it
OOM'd on the very first backward pass. This is moot in practice (colocate
isn't usable for training here regardless), but stated plainly rather
than implied as resolved.

**Next**: build E3 and E3c SFT training (data exists, nothing blocks
this), then E1 launch (HF generate, `--use-vllm off`).

## 2026-10-08 — pre-existing smoke-mode bug found before E1 launch, not a Phase 16 regression

Before launching E1, ran the project's own standard smoke gate
(`--smoke`) with `--scorer ranker --validator-version 2`. It crashed in
TRL/transformers' generation machinery, before any ranker/validator
code runs at all:

```
File ".../transformers/generation/continuous_batching/cache.py", line 738
ValueError: Invalid values: max_batch_tokens = 256, num_blocks = -2
```

**Confirmed NOT caused by anything built for Phase 16**: reran smoke
mode with the plain, unmodified validator scorer (no `--scorer ranker`,
no `--validator-version 2`) -- identical crash. This is a pre-existing
incompatibility between smoke mode's tiny dimensions
(`max_completion_len=256`, `core/model.py`'s own fp32/no-device-map
smoke loading) and the installed transformers' `PagedAttentionCache`
block-count calculation, which goes negative at these dimensions. Not
something to fix here (out of scope, risky to touch transformers
internals) and not something that blocks the real run: Phase 12 already
proved the real (non-smoke) hyperparameters work correctly with this
exact continuous-batching code path over 300 real steps, and the
ranker/validator_version changes are entirely downstream of generation
(reward computation, never touched by this crash). Proceeding directly
to the real E1 launch on that basis, logged here rather than silently
skipped.

## 2026-10-14 — E1 training completed cleanly, full evaluation against prediction 1

**Training** (W&B run `238lefg3`, `gdpo-qlora-gdpo-e1-ranker-rssft`,
`scripts/run_gdpo_e1.sh`): 300/300 steps in **one attempt, no restarts**,
~127h wall-clock. GDPO from RS-SFT init, `--scorer ranker
--validator-version 2` (ammonium-balance fix threaded into the ranker's
gate, commit `fd1ffa7`), `--kl-beta 0.0` (Phase 12's confirmed-safe
choice, finding 20), `--use-vllm off` (HF generate — colocate doesn't
fit on this card, see 2026-10-08 entry above).

Logged every 25 steps:

| step | reward | reward_std | loss | grad_norm | entropy | clipped_completions |
|---|---|---|---|---|---|---|
| 25 | 1.291 | 0.791 | 0.051 | 0.077 | 0.388 | 9.75% |
| 50 | 1.284 | 0.815 | 0.012 | 0.062 | 0.387 | 11.50% |
| 75 | 1.292 | 0.878 | 0.031 | 0.074 | 0.385 | 8.75% |
| 100 | 1.464 | 0.735 | 0.004 | 0.069 | 0.393 | 7.00% |
| 125 | 1.363 | 0.801 | 0.026 | 0.097 | 0.397 | 8.25% |
| 150 | 1.584 | 0.784 | 0.013 | 0.063 | 0.404 | 6.00% |
| 175 | 1.410 | 0.671 | 0.033 | 0.071 | 0.404 | 7.00% |
| 200 | 1.625 | 0.736 | 0.027 | 0.077 | 0.402 | 5.00% |
| 225 | 1.517 | 0.664 | 0.032 | 0.077 | 0.400 | 10.25% |
| 250 | 1.492 | 0.843 | 0.026 | 0.065 | 0.402 | 9.25% |
| 275 | 1.552 | 0.809 | 0.035 | 0.279 | 0.405 | 9.50% |
| 300 | 1.752 | 0.802 | 0.017 | 0.084 | 0.404 | 8.25% |

Reward climbed 1.29→1.75 (real, not noise — `frac_reward_zero_std` was
0.00 at nearly every logged step). `grad_norm` stayed low and stable
(0.06–0.28, one mild blip at step 275). Per-channel reward std stayed
alive in most of the 7 ranker channels for most of training
(`driving_force_margin` 0.18–0.27, `n_precursors` 0.12–0.24,
`precursor_decomposition_match`/`volatility_risk` generally >0.05,
`temperature_economy` 0.11–0.17, `slice_competing_phases` intermittently
zero at steps 75/125/175 but alive elsewhere, `precursor_instability`
smallest, 0.00–0.08 but nonzero most steps) — the contrast with the
validator's standing 8/10-dead-channels finding (Central Diagnosis,
CLAUDE.md) that this arm exists to test.

**Evaluation generations**: ASTRAL n=32 (`results/astral_gen_n32_e1.json`,
`research/astral_model_generations.py`, HF generate, main env) and
held-out-split n=32 (`data/novelty/e1_heldout_n32.jsonl`,
`data_curation/phase16_e1_heldout_sampling.py`, vLLM LoRA serving,
mira-vllm env), chained to launch automatically back-to-back
(`scripts/chain_e1_heldout_after_astral.sh`) since this machine has one
GPU. Both completed cleanly, rc=0.

### Prediction 1, bullet 1 — carbonate-free share and rule-picked hits rise relative to RS-SFT

| | predicted hits, clean-22 | predicted hits, full-35 | traditional hits, full-35 | carbonate share, full-35 | carbonate share, clean-22 | bare-oxide share, full-35 |
|---|---|---|---|---|---|---|
| RS-SFT (E1's init) | 9/22 | 10/35 | 17/35 | 45.94% | 43.95% | 25.84% |
| **E1** | **11/22** | **14/35** | **15/35** | **18.31%** | **16.58%** | **48.36%** |
| (reference) GDPO-phase12-β0 (validator-reward arm) | 12/22 | 14/35 | 17/35 (flat) | 18.07% | 15.11% | 56.91% |

**Rises exactly as predicted**, on both counts: predicted-set hits
10/35→14/35 and 9/22→11/22, carbonate share collapses 45.94%→18.31%
(carbonate-free 54.06%→81.69%), bare-oxide share nearly doubles. Unlike
the validator-reward arm (phase12-β0, flat 17→17 traditional), **E1's
traditional-set hits actually fell slightly, 17→15** — movement further
from convention, not a partial pull-back. **This is independent
corroboration of the carbonate/bare-oxide mechanism (Phase 15 findings
24/26) via a completely different reward** — the ranker (Arm B, 7
channels, no `thermodynamic_favorable` channel at all) reproduces
essentially the same shift as the validator's own `thermodynamic_favorable`-driven
effect, at a similar magnitude. Either the mechanism survives the
reward swap, or it is substantially inherited from the RS-SFT init
rather than caused by either reward — this run cannot distinguish the
two, but either reading is a real finding.

Mean validator reward: 0.9786 (all samples, n=1095 parsed/1120),
0.9661 on traditional-set matches (n=53), 0.9948 on predicted-set
matches (n=56) — gap +0.0287, between RS-SFT's +0.007 and
phase12-β0's +0.049. Parse rate 97.77% (25/1120 `PARSE_FAILURE`).

**Temperature — did the hack reassert? No, the opposite.** Mean
reported max-T = 779.4 °C (median 800), vs. the fixed ASTRAL
literature-best reference (mean `trad_best_T` across the 35 targets =
762.9 °C). **Overshoot = +16.5 °C** — far below RS-SFT's own +171 °C
(the previous best number in the project) and nowhere near
phase12-β0's +254 °C reassertion. Mechanistically expected, not a
surprise once seen: the ranker's `temperature_economy` channel stayed
reward-informative (nonzero std) through nearly all of training (see
table above), directly penalizing reported T — this is the first case
in the project where a temperature-economy-style objective (finding 15,
the earlier `temperature_economy` probe, which stalled at 23.8% capacity
and only partial compliance) actually drove reported temperature down
in a full GDPO run, not just a probe.

### Prediction 1, bullet 2 — no support growth beyond RS-SFT's own resampling rate

Per the pre-registration's exact definition (`docs/phases/PHASE16_PREREG.md`):
support growth of E1 relative to RS-SFT's 200-sample reference
(`data/novelty/rssft_reference_{astral,heldout}.jsonl`), vs. RS-SFT's
own resampling noise floor (its own fresh 32-sample draw vs. its own
200-sample reference, same targets). Computed by
`research/phase16_e1_support_growth.py` → `results/phase16/e1_support_growth.json`
(precursor sets normalized via `SynthesisValidator._normalize_formula`,
parse failures dropped, pooled across targets: numerator = E1's distinct
sets absent from RS-SFT's reference set for that target, denominator =
E1's total distinct sets):

| target set | E1 support growth | RS-SFT's own resampling rate (noise floor) |
|---|---|---|
| ASTRAL (35 targets) | **8.26%** (18/218) | 10.46% (25/239) |
| Held-out split (40 targets) | **10.00%** (15/150) | 11.67% (21/180) |

**Confirmed, and more cleanly than predicted**: E1's measured support
growth sits *below* RS-SFT's own resampling noise floor on both target
sets, not merely within it. GDPO sharpened inside RS-SFT's existing
support; it did not add new precursor-set combinations beyond what
resampling RS-SFT itself would turn up. Consistent with finding 26
(RS-SFT→phase12-GDPO's Jaccard indistinguishable from GDPO's own
resampling noise) via an independent metric and an independent reward.

### Prediction 1, bullet 3 — ARROWS³ gate agreement no better than carbonate-free baseline

**Flagging an interpretation rather than deciding it silently.** The
pre-registration cites "Phase 15's `research/distributional/arrows_gate_scoring.py`,
**unchanged**." Read literally, this means: reuse finding 27's existing,
already-settled result (the **validator's** GDPO-vote rule scored
0.616 agreement against real ARROWS³ lab outcomes vs. a 0.583
carbonate-free baseline, diff +0.033, 95% CI [−0.043,+0.112] — not
significantly better than baseline) rather than building a new
ranker-specific ARROWS³ gate. No new ranker-vs-ARROWS analysis exists
or was built for this entry — building one would mean adapting
~600 lines of pre-registered methodology (`arrows_gate_scoring.py`)
to the ranker's channel set outside anything this phase's scope-freeze
commissioned, and ARROWS³'s precursor-only (no molar ratio, no
per-route T sweep) format is exactly what caused 3/8 ranker channels to
be ungradeable in finding 19's ASTRAL gate test — a new ranker-ARROWS³
gate would likely hit the same ungradeable-channel problem.
**Carried forward as predicted, unchanged, pending regular Claude's
confirmation that this reading is the intended one** rather than a
request for new scope.

### Verdict

All three bullets of prediction 1 are **supported** by this evaluation:
rule-picked hits and carbonate-free share rose relative to RS-SFT
(bullet 1, clearly), support growth stayed at or below RS-SFT's own
noise floor on both target sets (bullet 2, cleanly), and the carried-forward
ARROWS³ result remains unchanged (bullet 3, pending confirmation of
that reading). E1 is a clean positive replication of the
"GDPO sharpens, does not expand support" finding (17/26) with a
materially different reward signal, plus a new, unanticipated result:
the first GDPO run in this project where reported synthesis temperature
fell rather than rose or held — the ranker's `temperature_economy`
channel appears to be doing real work where the standalone probe
(finding 15) only partially succeeded.

## 2026-10-14 (amended) — regular Claude's review of the E1 evaluation entry above

Three corrections to the entry above, all accepted as written.

**Bullet 1, overstated.** The predicted-set hit rise is weaker than the
"supported, clearly" framing above suggested. On the primary clean-22
metric the rise is 9/22→11/22: **2 gains, 0 losses, exact two-sided
McNemar p = 0.5** — not distinguishable from chance at this n. (Full-35
is 4 gains, 0 losses, p = 0.125 — still not decisive.) Report as "weak,
directionally consistent, not decisive," not as a confirmed rise.

Also corrected: the entry called the ranker a "completely different
reward" because it lacks `thermodynamic_favorable`. That's misleading —
`driving_force_margin`, one of the ranker's own 7 channels, is *also* a
thermodynamic driving-force quantity. **The accurate claim is that two
different thermodynamics-based rewards (validator's
`thermodynamic_favorable` and ranker's `driving_force_margin`) both push
toward bare oxides**, not that an unrelated reward independently
reproduced the shift. The carbonate collapse itself (45.9%→18.3%,
matching phase12-β0's 46%→18%) stands as reported.

**Bullet 3, wrongly marked "supported."** Re-reading: Phase 15's ARROWS³
gate (`arrows_gate_scoring.py`) scored the **validator** and the
**comparator** against real lab outcomes — it never scored the
**ranker**. Reusing that result for E1 (a ranker-reward arm) tests
nothing about the ranker. Correct status: **not tested**, for the
reason already given in the entry above (ARROWS³ has no molar ratios or
per-route temperature sweeps, so several ranker channels would be
ungradeable — the same problem finding 19 hit on ASTRAL). Per the scope
freeze, no new ranker-ARROWS³ gate is being built to resolve this.

**Revised verdict: 2 of 3 supported** (support growth, cleanly; carbonate
collapse, cleanly, but via a shared thermodynamic mechanism rather than
an independent reward confirmation) **and 1 not tested** (ARROWS³
agreement — no ranker-specific result exists). The predicted-hit-rise
half of bullet 1 is directionally consistent but not statistically
decisive.

**Temperature claim corrected.** The "+16.5 °C overshoot" figure in the
entry above used a single pooled mean-vs-mean comparison that hides
target-by-target spread. The right comparison is per-target: the
model's own mean reported max-T vs. **the best temperature of whichever
reference route (traditional or predicted) has the higher lab purity for
that target**, bucketed at ±100 °C (reproduced exactly from
`results/astral_gen_n32_{base,rs_sft,gdpo_phase12_beta0,e1}.json`):

| model | avg \|error\| | within 100 °C | too hot by >100 °C | too cold by >100 °C |
|---|---|---|---|---|
| base | 272 °C | 5 | 30 | 0 |
| RS-SFT | 189 °C | 10 | 25 | 0 |
| GDPO from RS-SFT (phase12-β0) | 224 °C | 10 | 25 | 0 |
| **E1** | **120 °C** | **13** | **12** | **10** |

**Honest line: E1 cut the average temperature error by roughly a third
(189→120 °C vs. its RS-SFT init), but it also created a failure mode no
earlier model had** — running more than 100 °C too *cold* on 10/35
targets. No prior model (base, RS-SFT, or the validator-reward GDPO arm)
ever undershot by more than 100 °C on a single target. This is still
the first reward in this project that measurably corrected the
temperature hack rather than feeding it (findings 6, 9, 15, 20/22's own
reassertion pattern) — but it overcorrected on roughly a third of
targets rather than converging tightly. Report both halves together.

**Next (critical path per regular Claude, GPU was idle):** E3 and E3c
n=32 sampling on both target sets launched immediately
(`data_curation/phase16_e3_e3c_sampling.py`, tmux `e3_e3c_sampling`,
`run_logs/phase16_e3_e3c_sampling.log`) — both are LoRA-on-base arms, so
their matching reference distribution is base's own existing 200+32
draws (`data/novelty/base_reference_{heldout,astral}.jsonl`), not
RS-SFT's. Novelty-rate and support-growth computation follows once this
finishes. E4 waits on that result.

## 2026-10-15 — E3/E3c evaluation: prediction 2 FAILS its own pre-registered gate

E3 and E3c's own n=32 generations on both target sets
(`data_curation/phase16_e3_e3c_sampling.py`) and the novelty-rate/
support-growth evaluation (`research/phase16_e3_e3c_evaluation.py`,
against **base's** own 200-sample reference — both E3 and E3c are
LoRA-on-base arms per the arm table, not RS-SFT) both completed
cleanly. Followed by the pre-registration's own named test
(`research/phase16_e3_e3c_bootstrap.py`): cluster bootstrap by target
(10,000 resamples, seed 42) on the held-out split, the primary
comparison for prediction 2.

**Point estimates** (`results/phase16/e3_e3c_evaluation.json`):

| target set | model | gate pass rate | novelty rate (among gate-pass) | support growth | (noise floor) |
|---|---|---|---|---|---|
| held-out (40) | E3 | 56.87% | 75.41% | 14.97% (25/167) | 12.95% (25/193) |
| held-out (40) | E3c | 57.89% | 76.22% | 15.85% (29/183) | 12.95% (25/193) |
| ASTRAL (35) | E3 | 63.12% | 60.82% | 21.46% (50/233) | 10.33% (22/213) |
| ASTRAL (35) | E3c | 65.18% | 61.78% | 18.85% (46/244) | 10.33% (22/213) |

**On the held-out split (the primary, pre-registered comparison), E3
is numerically BELOW E3c on both metrics the prediction named**:
novelty rate 75.41% vs 76.22%, support growth 14.97% vs 15.85%. The
pre-registration's own named test (`results/phase16/e3_e3c_bootstrap.json`):

- novelty rate diff (E3−E3c): **−0.0084**, 95% CI **[−0.0324, +0.0164]** — includes zero, point estimate negative.
- support growth diff (E3−E3c): **−0.0088**, 95% CI **[−0.0789, +0.0546]** — includes zero, point estimate negative.
- gate pass rate diff: **−1.02 pp** — within the pre-registered ±5pp tolerance (this one condition is met).

**This trips the pre-registration's own "does not support" rule
exactly as written**: "E3 ≤ E3c on either of the first two... a
reportable null and is not re-run with a different threshold." Per
that rule, **not re-tuning the rarity threshold or re-running with a
different filter** — reporting as-is.

Fisher-exact supplementary checks (not bootstrapped, single-table):
neither E3 (p=0.65) nor E3c (p=0.46) significantly beats base's own
resampling noise floor on held-out either — both arms are statistically
indistinguishable from noise there. On ASTRAL (secondary target set,
not covered by prediction 2's own wording, which names only the
held-out split), both E3 (p=0.0018) and E3c (p=0.012) *do* clear the
noise floor — real support growth happens on ASTRAL for both arms —
but E3 vs E3c there is also not significant (p=0.49, Fisher exact,
single-table, not bootstrapped).

**Reading**: self-distillation on base's own gate-passing samples (E3c,
no novelty filter) produces about as much new support as self-distillation
on the novelty-filtered subset (E3) does. The rarity/novelty filter is
not pulling its weight relative to just resampling and keeping whatever
passes the validity gate. Whatever support growth exists on ASTRAL
appears to come from the self-distillation step itself (both arms clear
the noise floor there), not from the novelty condition specifically.

**Flagging rather than deciding: does E4 still launch?** E4's own
question (prediction 3 — "does practice amplify what E3 already
samples") is phrased as a property of E4 vs. E3 specifically, and could
still be tested on its own terms regardless of this result. But E4's
premise, as stated in the arm table ("once new recipes are in the
starting distribution, does practice amplify them?"), assumes E3 put
something novel-beyond-control into the starting distribution — and
that premise just failed its own gate. Per this project's standing
practice at pre-registered gate failures (Phase 13, finding 23; Phase
15/ranker, finding 19), stopping here to report rather than
auto-launching E4, since this is a decision about whether the next
arm's premise still holds, not a routine execution step.
